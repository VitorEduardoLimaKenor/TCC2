"""Cache de rascunhos, retomada e proveniência (E04).

Cada fator que pode mudar um rascunho, um vetor ou uma linha do log precisa
invalidar o reaproveitamento; o que não muda nada precisa ser reaproveitado.
"""

from __future__ import annotations

import importlib.metadata
import json
import re
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tcc_rag import runner as runner_mod
from tcc_rag.cli import _indice_conferido, app
from tcc_rag.config import EmbeddingConfig, ExperimentConfig
from tcc_rag.data.spider import DbSchema
from tcc_rag.evaluation import em_oficial, ex_oficial
from tcc_rag.evaluation.ex_oficial import AvaliadorEX, carregar_oficial
from tcc_rag.generation import prompt as prompt_mod
from tcc_rag.generation import rascunhos
from tcc_rag.generation.llm import DummyBackend
from tcc_rag.logging_io import AVALIADOR_EX_HISTORICO, RunLog, report_rows
from tcc_rag.pipeline import identidade_indice, runs_dir
from tcc_rag.retrieval.embedder import Embedder
from tcc_rag.retrieval.index import DemoIndex
from tcc_rag.runner import Runner
from tcc_rag.schemas import Demo, Selection
from tcc_rag.schemas import TestItem as ItemDeTeste  # nome sem "Test": o pytest não tenta coletá-lo

TABELA = {
    "db_id": "escola",
    "table_names_original": ["aluno"],
    "column_names_original": [[-1, "*"], [0, "id"], [0, "nome"], [0, "nota"], [0, "turma"]],
    "foreign_keys": [],
}


class ContaGeracoes:
    def __init__(self, cfg):
        self.backend = DummyBackend(cfg)
        self.prompts = 0

    def generate(self, prompts):
        self.prompts += len(prompts)
        return self.backend.generate(prompts)

    def count_tokens(self, text):
        return self.backend.count_tokens(text)

    @property
    def descriptor(self):
        return self.backend.descriptor


@pytest.fixture
def mundo(tmp_path: Path):
    db = tmp_path / "escola.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE aluno (id INT, nome TEXT, nota REAL, turma INT)")
    conn.executemany("INSERT INTO aluno VALUES (?,?,?,?)", [(i, f"A{i}", 5.0 + i, i % 3) for i in range(9)])
    conn.commit()
    conn.close()
    cfg = ExperimentConfig()
    cfg.paths.work_dir = str(tmp_path / "artifacts")
    cfg.generation.backend = "dummy"
    schemas = {"escola": DbSchema("escola", "CREATE TABLE aluno (id INT, nome TEXT, nota REAL, turma INT);",
                                  str(db), 1, 4, tables_entry=TABELA)}
    itens = [ItemDeTeste(example_id=f"e{i}", db_id="escola", lang="pt",
                      question_raw=f"quantos alunos na turma {i}",
                      question_delex="quantos alunos na turma <num>",
                      gold_sql=f"SELECT count(*) FROM aluno WHERE turma = {i}")
             for i in range(3)]
    return cfg, schemas, itens, tmp_path


SELETOR_VAZIO = SimpleNamespace(select=lambda query, condition: Selection(), drafter=None)


def _runner(cfg, schemas, log_path, run_id="r", backend=None, **kwargs):
    return Runner(cfg, backend or ContaGeracoes(cfg.generation), SELETOR_VAZIO, schemas,
                  RunLog(log_path), run_id, **kwargs)


# ------------------------------------------------------------ rascunhos

def _rascunhar(cfg, schemas, itens, tmp_path):
    backend = ContaGeracoes(cfg.generation)
    runner = _runner(cfg, schemas, tmp_path / "rascunho.jsonl", backend=backend)
    runner.prepare_drafts(itens)
    return backend.prompts, {v["origem"] for v in runner._rascunhos.values()}


def test_rascunho_e_reutilizado_quando_a_identidade_confere(mundo):
    cfg, schemas, itens, tmp_path = mundo
    assert _rascunhar(cfg, schemas, itens, tmp_path) == (3, {"gerado"})
    assert _rascunhar(cfg, schemas, itens, tmp_path) == (0, {"cache"})


def _mudar_geracao(campo, valor):
    def mudar(cfg, schemas, itens, monkeypatch):
        setattr(cfg.generation, campo, valor)
        return cfg, schemas, itens
    return mudar


def _mudar_pergunta(cfg, schemas, itens, monkeypatch):
    return cfg, schemas, [replace(i, question_raw=i.question_raw + " (tradução real)") for i in itens]


def _mudar_ddl(cfg, schemas, itens, monkeypatch):
    return cfg, {"escola": replace(schemas["escola"], ddl=schemas["escola"].ddl + "\n-- outro")}, itens


def _mudar_instrucao(cfg, schemas, itens, monkeypatch):
    monkeypatch.setattr(prompt_mod, "INSTRUCTION_PT", prompt_mod.INSTRUCTION_PT + " Seja breve.")
    return cfg, schemas, itens


def _mudar_idioma(cfg, schemas, itens, monkeypatch):
    return cfg, schemas, [replace(i, lang="en") for i in itens]


def _mudar_biblioteca(cfg, schemas, itens, monkeypatch):
    original = rascunhos.versoes
    monkeypatch.setattr(rascunhos, "versoes", lambda p: {**original(p), "transformers": "0.0.1"})
    return cfg, schemas, itens


FATORES = {
    "modelo": _mudar_geracao("model", "outro/modelo"),
    "revisao": _mudar_geracao("revision", "abc123"),
    "backend": _mudar_geracao("backend", "hf"),
    "quantizacao": _mudar_geracao("quantization", "awq"),
    "dtype": _mudar_geracao("dtype", "bfloat16"),
    "temperatura": _mudar_geracao("temperature", 0.7),
    "max_new_tokens": _mudar_geracao("max_new_tokens", 64),
    "max_context_tokens": _mudar_geracao("max_context_tokens", 2048),
    "semente": _mudar_geracao("seed", 1),
    "lote": _mudar_geracao("batch_size", 1),
    "pergunta": _mudar_pergunta,
    "ddl": _mudar_ddl,
    "instrucao": _mudar_instrucao,
    "idioma": _mudar_idioma,
    "biblioteca": _mudar_biblioteca,
}


@pytest.mark.parametrize("fator", list(FATORES))
def test_cada_fator_invalida_o_rascunho(mundo, monkeypatch, fator):
    cfg, schemas, itens, tmp_path = mundo
    assert _rascunhar(cfg, schemas, itens, tmp_path)[0] == 3
    cfg, schemas, itens = FATORES[fator](cfg, schemas, itens, monkeypatch)
    assert _rascunhar(cfg, schemas, itens, tmp_path) == (3, {"gerado"})


def test_cache_legado_vai_para_quarentena_e_nao_e_usado(mundo, capsys):
    cfg, schemas, itens, tmp_path = mundo
    cache = cfg.work / "cache"
    cache.mkdir(parents=True)
    legado = cache / "drafts_f30f11a36f20.json"
    legado.write_text(json.dumps({f"{i.example_id}|pt": "SELECT 'antigo'" for i in itens}), encoding="utf-8")
    assert _rascunhar(cfg, schemas, itens, tmp_path) == (3, {"gerado"})
    assert not legado.exists()
    assert (cache / "quarentena" / legado.name).exists()
    assert (cache / "quarentena" / f"{legado.name}.motivo.txt").exists()
    assert "não será usado" in capsys.readouterr().err


def test_entrada_com_identidade_adulterada_e_ignorada(mundo, capsys):
    cfg, schemas, itens, tmp_path = mundo
    _rascunhar(cfg, schemas, itens, tmp_path)
    arquivo = cfg.work / "cache" / rascunhos.ARQUIVO
    linhas = [json.loads(l) for l in arquivo.read_text(encoding="utf-8").splitlines()]
    linhas[0]["identidade"]["pergunta_sha256"] = "0" * 64
    arquivo.write_text("".join(json.dumps(l) + "\n" for l in linhas), encoding="utf-8")
    assert _rascunhar(cfg, schemas, itens, tmp_path)[0] == 1
    assert "não confere" in capsys.readouterr().err


def test_linha_de_c4_registra_origem_e_custo_do_rascunho(mundo):
    cfg, schemas, itens, tmp_path = mundo
    seletor = SimpleNamespace(drafter=None)
    seletor.select = lambda q, c: Selection(preliminary_sql=seletor.drafter(q), parse_ok=True)
    log = tmp_path / "c4.jsonl"
    Runner(cfg, ContaGeracoes(cfg.generation), seletor, schemas, RunLog(log), "c4").run(itens, ("C4",))
    linhas = RunLog(log).rows()
    assert {l["rascunho_origem"] for l in linhas} == {"gerado"}
    assert all(l["rascunho_chave"] and l["rascunho_prompt_tokens"] > 0 for l in linhas)
    assert all(l["preliminary_sql"].startswith("SELECT 1 AS dummy_") for l in linhas)


# ------------------------------------------------------------- retomada

def _primeira(mundo, **kwargs):
    cfg, schemas, itens, tmp_path = mundo
    log = tmp_path / "retomada.jsonl"
    assert _runner(cfg, schemas, log, **kwargs).run(itens, ("C0",))["C0"] == 3
    return log


def test_retomada_compativel_nao_duplica(mundo):
    cfg, schemas, itens, _ = mundo
    log = _primeira(mundo, partition_digest="d1", split="val", indice_digest="i1")
    backend = ContaGeracoes(cfg.generation)
    runner = _runner(cfg, schemas, log, backend=backend, partition_digest="d1", split="val", indice_digest="i1")
    assert runner.run(itens, ("C0",)) == {"C0": 0}
    assert backend.prompts == 0 and len(RunLog(log).rows()) == 3


def _avaliador_outro(monkeypatch):
    monkeypatch.setattr(AvaliadorEX, "identificador", property(lambda self: "outro avaliador"))


def _em_outro(monkeypatch):
    monkeypatch.setattr(em_oficial.AvaliadorEM, "identificador", property(lambda self: "outro em"))


INCOMPATIVEIS = {
    "config_hash": (lambda cfg, mp: setattr(cfg.retrieval, "k", 8), {}, "outra configuração"),
    "split": (None, {"split": "test"}, "outro split"),
    "partition_digest": (None, {"partition_digest": "d2"}, "outra partição"),
    "indice_digest": (None, {"indice_digest": "i2"}, "outro índice"),
    "ex_avaliador": (lambda cfg, mp: _avaliador_outro(mp), {}, "outro avaliador de EX"),
    "em_avaliador": (lambda cfg, mp: _em_outro(mp), {}, "outro avaliador de EM"),
}


@pytest.mark.parametrize("campo", list(INCOMPATIVEIS))
def test_retomada_incompativel_e_recusada(mundo, monkeypatch, campo):
    cfg, schemas, itens, _ = mundo
    base = {"partition_digest": "d1", "split": "val", "indice_digest": "i1"}
    log = _primeira(mundo, **base)
    mudar, kwargs, mensagem = INCOMPATIVEIS[campo]
    if mudar:
        mudar(cfg, monkeypatch)
    with pytest.raises(RuntimeError, match=mensagem):
        _runner(cfg, schemas, log, **{**base, **kwargs})


def test_retomada_sobre_log_antigo_e_recusada(mundo):
    cfg, schemas, itens, tmp_path = mundo
    log = tmp_path / "antigo.jsonl"
    log.write_text(json.dumps({"run_id": "r", "condition": "C0", "lang": "pt", "example_id": "e0",
                               "config_hash": cfg.config_hash}) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="outro formato de log"):
        _runner(cfg, schemas, log)


def test_retomada_com_outro_texto_de_pergunta_e_recusada(mundo):
    cfg, schemas, itens, _ = mundo
    log = _primeira(mundo)
    novos = [replace(i, question_raw=i.question_raw + " (tradução real)") for i in itens]
    with pytest.raises(RuntimeError, match="outro conteúdo"):
        _runner(cfg, schemas, log).run(novos, ("C0",))


def test_retomada_com_outra_revisao_do_codigo_avisa(mundo, monkeypatch, capsys):
    cfg, schemas, itens, _ = mundo
    log = _primeira(mundo)
    monkeypatch.setattr(runner_mod, "codigo_revisao", lambda: "outra-revisao")
    assert _runner(cfg, schemas, log).run(itens, ("C0",)) == {"C0": 0}
    assert "outra revisão do código" in capsys.readouterr().err


def test_linha_registra_identidade_e_proveniencia(mundo):
    log = _primeira(mundo, partition_digest="d1", split="val", indice_digest="i1")
    for linha in RunLog(log).rows():
        assert linha["formato_log"] == runner_mod.FORMATO_LOG
        assert re.fullmatch(r"[0-9a-f]{16}", linha["item_sha256"])
        assert linha["indice_digest"] == "i1"
        assert linha["codigo_revisao"]
        assert "sqlparse" in linha["ambiente"]["bibliotecas"]
        assert linha["rascunho_origem"] is None


# ------------------------------------------------------- log e relatório

def test_cauda_incompleta_vai_para_incidente_e_o_log_continua_legivel(mundo, capsys):
    cfg, schemas, itens, _ = mundo
    log = _primeira(mundo, partition_digest="d1", split="val")
    with log.open("a", encoding="utf-8") as fh:
        fh.write('{"run_id": "r", "condi')
    reaberto = RunLog(log)
    assert reaberto.incidente is not None and reaberto.incidente.read_text(encoding="utf-8").startswith('{"run_id"')
    assert log.read_bytes().endswith(b"\n")
    assert "linha incompleta" in capsys.readouterr().err
    destino = log.parent / "r.jsonl"
    log.rename(destino)
    assert len(report_rows(destino.parent, config_hash=cfg.config_hash, run_ids=["r"])) == 3


def _linha(run_id, **campos):
    return {"run_id": run_id, "config_hash": "cfg", "partition_digest": "p", "split": "val",
            "condition": "C2", "lang": "pt", "example_id": "e1", "db_id": "db", "ex_correct": True,
            **campos}


def _gravar(pasta, nome, linhas):
    pasta.mkdir(parents=True, exist_ok=True)
    (pasta / f"{nome}.jsonl").write_text("".join(json.dumps(l) + "\n" for l in linhas), encoding="utf-8")


def test_relatorio_recusa_misturar_avaliadores_de_ex(tmp_path):
    _gravar(tmp_path, "historico", [_linha("historico")])
    _gravar(tmp_path, "novo", [_linha("novo", example_id="e2", ex_status="coincide", ex_avaliador="oficial")])
    _gravar(tmp_path, "outro", [_linha("outro", example_id="e3", ex_status="coincide", ex_avaliador="outro")])
    assert len(report_rows(tmp_path, config_hash="cfg", run_ids=["historico"])) == 1
    for par in (["historico", "novo"], ["novo", "outro"]):
        with pytest.raises(ValueError, match="avaliadores de EX diferentes"):
            report_rows(tmp_path, config_hash="cfg", run_ids=par)


@pytest.mark.parametrize("campos,mensagem", [
    ({"ex_status": "coincide"}, "ex_status sem ex_avaliador"),
    ({"ex_avaliador": "oficial"}, "sem ex_status"),
    ({"ex_status": "coincide", "ex_avaliador": "oficial", "em_status": "coincide", "em_oficial": True},
     "em_status sem em_avaliador"),
])
def test_relatorio_exige_avaliador_identificado(tmp_path, campos, mensagem):
    _gravar(tmp_path, "r", [_linha("r", **campos)])
    with pytest.raises(ValueError, match=mensagem):
        report_rows(tmp_path, config_hash="cfg", run_ids=["r"])


def test_evaluate_le_logs_novos_e_historicos_isolados(tmp_path):
    cfg = ExperimentConfig()
    cfg.paths.work_dir = str(tmp_path / "artifacts")
    config = tmp_path / "config.yaml"
    cfg.dump(config)
    novo = _linha("novo", config_hash=cfg.config_hash, ex_status="coincide", ex_avaliador="oficial",
                  codigo_revisao="abc")
    _gravar(runs_dir(cfg), "novo", [novo])
    assert app(["evaluate", "--config", str(config)]) == 0
    selecao = json.loads((cfg.work / "summary_selection.json").read_text(encoding="utf-8"))
    assert selecao["ex_avaliador"] == "oficial" and selecao["codigo_revisoes"] == ["abc"]
    assert selecao["runs_dir"] == str(runs_dir(cfg))
    historicos = cfg.work / "runs"
    _gravar(historicos, "velho", [_linha("velho", config_hash=cfg.config_hash)])
    assert app(["evaluate", "--config", str(config), "--runs-dir", str(historicos)]) == 0
    selecao = json.loads((cfg.work / "summary_selection.json").read_text(encoding="utf-8"))
    assert selecao["ex_avaliador"] == AVALIADOR_EX_HISTORICO


# ---------------------------------------------------- versões do avaliador

def test_versao_errada_do_sqlparse_e_recusada(monkeypatch):
    real = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version",
                        lambda p: "0.5.0" if p == "sqlparse" else real(p))
    with pytest.raises(RuntimeError, match="sqlparse==0.6.0"):
        carregar_oficial()


def test_versoes_fixadas_no_codigo_batem_com_o_pyproject():
    texto = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text(encoding="utf-8")
    for fixadas in (ex_oficial.VERSOES_FIXADAS, em_oficial.VERSOES_FIXADAS):
        for pacote, versao in fixadas.items():
            assert f'"{pacote}=={versao}"' in texto
    from tcc_rag.config import SandboxConfig
    from tcc_rag.evaluation.sandbox import Sandbox

    assert "sqlparse=0.6.0" in AvaliadorEX(Sandbox(SandboxConfig())).identificador


# ------------------------------------------------------------ embeddings

class ModeloFalso:
    def __init__(self):
        self.entradas = []

    def encode(self, batch, **kwargs):
        self.entradas.extend(batch)
        return np.ones((len(batch), 4), dtype=np.float32)


def _vetorizar(tmp_path, **cfg):
    emb = Embedder(EmbeddingConfig(**cfg), tmp_path / "cache")
    emb._model = ModeloFalso()
    emb.encode(["quantos alunos"], kind="query")
    return emb._model.entradas


def test_prefixo_e_configuracao_entram_na_identidade_do_vetor(tmp_path):
    assert _vetorizar(tmp_path) == ["query: quantos alunos"]
    assert _vetorizar(tmp_path) == []  # mesmo texto e mesma identidade: do cache
    assert _vetorizar(tmp_path, query_prefix="") == ["quantos alunos"]
    assert _vetorizar(tmp_path, normalize=False) == ["query: quantos alunos"]
    assert _vetorizar(tmp_path, device="cuda") == ["query: quantos alunos"]


# --------------------------------------------------------------- índice

def _pool(pergunta="quantos alunos"):
    return [Demo(uid=f"d{i}::pt", example_id=f"d{i}", db_id="escola", lang="pt", question_raw=pergunta,
                 question_delex=pergunta, sql="SELECT count(*) FROM aluno", signature={}, tables=["aluno"],
                 columns=[], family_id="f", partition="pool") for i in range(2)]


def test_indice_sem_identidade_ou_desatualizado_e_recusado(tmp_path):
    cfg = ExperimentConfig().qdrant
    cfg.path = str(tmp_path / "qdrant")
    embedder = SimpleNamespace(identidade={"modelo": "falso"})
    index = DemoIndex(cfg, dimension=4)
    index.recreate()
    index.upsert(_pool(), np.ones((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="sem identidade"):
        _indice_conferido(index, _pool(), embedder)
    index.gravar_identidade(identidade_indice(_pool(), embedder))
    assert _indice_conferido(index, _pool(), embedder) == identidade_indice(_pool(), embedder)["digest"]
    with pytest.raises(ValueError, match="outro pool"):
        _indice_conferido(index, _pool("how many students"), embedder)
    with pytest.raises(ValueError, match="outro pool"):
        _indice_conferido(index, _pool(), SimpleNamespace(identidade={"modelo": "outro"}))
    index.recreate()
    assert index.ler_identidade() is None
    index.close()


def test_relatorio_recusa_misturar_avaliadores_de_em(tmp_path):
    """B2 da revisão da E03: dois logs do EM oficial com identificadores
    diferentes (por exemplo, outro nltk ou outro punkt_tab) não se combinam."""
    base = {"ex_status": "coincide", "ex_avaliador": "oficial", "em_status": "coincide", "em_oficial": True}
    _gravar(tmp_path, "a", [_linha("a", **base, em_avaliador="em nltk=3.9.4")])
    _gravar(tmp_path, "b", [_linha("b", example_id="e2", **base, em_avaliador="em nltk=3.10.4")])
    assert len(report_rows(tmp_path, config_hash="cfg", run_ids=["a"])) == 1
    with pytest.raises(ValueError, match="avaliadores de EM diferentes"):
        report_rows(tmp_path, config_hash="cfg", run_ids=["a", "b"])
    _gravar(tmp_path, "c", [_linha("c", example_id="e3", **base, em_avaliador="")])
    with pytest.raises(ValueError, match="em_status sem em_avaliador"):
        report_rows(tmp_path, config_hash="cfg", run_ids=["c"])
