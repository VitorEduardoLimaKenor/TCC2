"""EX oficial: o adaptador decide como o avaliador oficial do Spider.

Cada caso roda duas vezes sobre o mesmo banco sintético: pelo adaptador
(`AvaliadorEX`, execução no Sandbox, somente leitura) e pela função oficial
`eval_exec_match`, numa cópia descartável do banco — a oficial abre o arquivo
para escrita. Os dois vereditos precisam coincidir entre si e com o esperado, que
documenta a regra oficial em cada caso, inclusive as que surpreendem.
"""

from __future__ import annotations

import random
import shutil
import sqlite3
from pathlib import Path

import pytest

from tcc_rag.config import SandboxConfig
from tcc_rag.evaluation import ex_oficial
from tcc_rag.evaluation.ex_oficial import AvaliadorEX, carregar_oficial, comparar
from tcc_rag.evaluation.sandbox import Sandbox


@pytest.fixture
def banco(tmp_path: Path) -> Path:
    caminho = tmp_path / "original" / "escola" / "escola.sqlite"
    caminho.parent.mkdir(parents=True)
    conn = sqlite3.connect(caminho)
    conn.executescript(
        """
        CREATE TABLE aluno (id INTEGER, nome TEXT, nota REAL, turma TEXT);
        INSERT INTO aluno VALUES (1, 'Ana', 9.0, 'A'), (2, 'Bruno', 7.5, 'B'),
            (3, 'Célia', 8.25, 'A'), (4, 'Davi', 7.5, 'B'), (5, 'Ana', 6.0, 'C');
        CREATE TABLE valores (id INTEGER, value INTEGER);
        INSERT INTO valores VALUES (1, 42);
        CREATE TABLE largo (c1 INT, c2 INT, c3 INT, c4 INT, c5 INT, c6 INT, c7 INT);
        INSERT INTO largo VALUES (1, 2, 3, 4, 5, 6, 7), (11, 12, 13, 14, 15, 16, 17);
        CREATE TABLE ruim (id INTEGER, t TEXT);
        INSERT INTO ruim VALUES (1, CAST(X'41FF42' AS TEXT));
        CREATE TABLE arquivos (id INTEGER, dado BLOB);
        INSERT INTO arquivos VALUES (1, X'0001FF');
        """
    )
    conn.commit()
    conn.close()
    return caminho


def _oficial(banco: Path, tmp_path: Path, pred: str, gold: str, keep_distinct: bool) -> int:
    """Como `evaluation.py --etype exec`: strip, "value" -> "1", eval_exec_match."""
    copia = tmp_path / "oficial" / "escola" / "escola.sqlite"
    copia.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(banco, copia)
    exec_eval, _ = carregar_oficial()
    return exec_eval.eval_exec_match(
        db=str(copia),
        p_str=pred.strip().replace("value", "1"),
        g_str=gold.strip(),
        plug_value=False,
        keep_distinct=keep_distinct,
        progress_bar_for_each_datapoint=False,
    )


CASOS = [
    # (caso, predição, referência, keep_distinct, veredito oficial esperado)
    ("ordem_livre_sem_order_by", "SELECT nome FROM aluno ORDER BY nome DESC",
     "SELECT nome FROM aluno", False, 1),
    ("ordem_exigida_com_order_by", "SELECT nome FROM aluno ORDER BY nome DESC",
     "SELECT nome FROM aluno ORDER BY nome", False, 0),
    # "order by" em qualquer ponto do texto da referência torna a ordem relevante.
    ("order_by_so_na_subconsulta",
     "SELECT nome FROM aluno WHERE id IN (SELECT id FROM aluno ORDER BY nota LIMIT 3) ORDER BY nome DESC",
     "SELECT nome FROM aluno WHERE id IN (SELECT id FROM aluno ORDER BY nota LIMIT 3)", False, 0),
    ("permutacao_de_2_colunas", "SELECT nota, nome FROM aluno",
     "SELECT nome, nota FROM aluno", False, 1),
    ("permutacao_de_7_colunas", "SELECT c7, c6, c5, c4, c3, c2, c1 FROM largo",
     "SELECT c1, c2, c3, c4, c5, c6, c7 FROM largo", False, 1),
    ("multiplicidade_diferente", "SELECT turma FROM aluno WHERE id IN (1, 2, 4)",
     "SELECT turma FROM aluno WHERE id IN (1, 2, 3)", False, 0),
    # Padrão oficial: todo DISTINCT sai das duas consultas antes de executar.
    ("distinct_removido", "SELECT DISTINCT nota FROM aluno WHERE nota < 8",
     "SELECT nota FROM aluno WHERE nota < 8", False, 1),
    ("distinct_mantido", "SELECT DISTINCT nota FROM aluno WHERE nota < 8",
     "SELECT nota FROM aluno WHERE nota < 8", True, 0),
    ("count_distinct_removido", "SELECT count(nome) FROM aluno",
     "SELECT count(DISTINCT nome) FROM aluno", False, 1),
    ("count_distinct_mantido", "SELECT count(nome) FROM aluno",
     "SELECT count(DISTINCT nome) FROM aluno", True, 0),
    ("ambos_vazios", "SELECT nome FROM aluno WHERE nota > 100",
     "SELECT nome FROM aluno WHERE id < 0", False, 1),
    ("predicao_com_mais_linhas", "SELECT nome FROM aluno",
     "SELECT nome FROM aluno WHERE id = 1", False, 0),
    ("predicao_com_menos_linhas", "SELECT nome FROM aluno WHERE id = 1",
     "SELECT nome FROM aluno", False, 0),
    ("inteiro_igual_a_real", "SELECT 1", "SELECT 1.0", False, 1),
    ("real_sem_arredondamento", "SELECT 0.1 + 0.2", "SELECT 0.3", False, 0),
    # A oficial troca "value" por "1" só na predição: coluna "value" deixa de bater.
    ("marcador_value_na_predicao", "SELECT value FROM valores WHERE id = 1",
     "SELECT value FROM valores WHERE id = 1", False, 0),
    ("operador_com_espaco", "SELECT nome FROM aluno WHERE nota > = 8",
     "SELECT nome FROM aluno WHERE nota >= 8", False, 1),
    ("ano_corrente_vira_2020", "SELECT nome FROM aluno WHERE id = YEAR(CURDATE()) - 2019",
     "SELECT nome FROM aluno WHERE id = 1", False, 1),
    ("nulo", "SELECT NULL", "SELECT NULL", False, 1),
    ("texto_com_bytes_invalidos", "SELECT 'AB'", "SELECT t FROM ruim WHERE id = 1", False, 1),
    ("blob", "SELECT X'0001FF'", "SELECT dado FROM arquivos WHERE id = 1", False, 1),
    ("erro_de_sintaxe", "SELEC nome FROM aluno", "SELECT nome FROM aluno", False, 0),
]


@pytest.mark.parametrize("caso,pred,gold,keep_distinct,esperado", CASOS, ids=[c[0] for c in CASOS])
def test_adaptador_concorda_com_o_oficial(banco, tmp_path, caso, pred, gold, keep_distinct, esperado):
    oficial = _oficial(banco, tmp_path, pred, gold, keep_distinct)
    veredito = AvaliadorEX(Sandbox(SandboxConfig()), keep_distinct=keep_distinct).avaliar(banco, pred, gold)
    assert oficial == esperado
    assert veredito.correto is bool(esperado)
    assert veredito.situacao in {"coincide", "difere", "erro_predicao"}


def test_detalhes_do_veredito(banco):
    avaliador = AvaliadorEX(Sandbox(SandboxConfig()))
    mais = avaliador.avaliar(banco, "SELECT nome FROM aluno", "SELECT nome FROM aluno WHERE id = 1")
    assert (mais.situacao, mais.linhas_ref, mais.linhas_pred) == ("difere", 1, 2)
    vazio = avaliador.avaliar(banco, "SELECT nome FROM aluno WHERE nota > 100", "SELECT nome FROM aluno WHERE id < 0")
    assert vazio.correto and vazio.ambos_vazios and vazio.linhas_ref == 0
    ordenado = avaliador.avaliar(banco, "SELECT nome FROM aluno", "SELECT nome FROM aluno ORDER BY id")
    assert ordenado.ordem_importa and ordenado.correto


def _alunos(banco: Path) -> int:
    conn = sqlite3.connect(banco)
    try:
        return conn.execute("SELECT count(*) FROM aluno").fetchone()[0]
    finally:
        conn.close()


@pytest.mark.parametrize("pred", ["DELETE FROM aluno", "UPDATE aluno SET nota = 0", "", "   "])
def test_predicao_que_nao_executa_e_erro(banco, pred):
    veredito = AvaliadorEX(Sandbox(SandboxConfig())).avaliar(banco, pred, "SELECT nome FROM aluno")
    assert veredito.correto is False and veredito.situacao == "erro_predicao"
    assert veredito.pred_executou is False
    assert _alunos(banco) == 5


def test_varias_instrucoes(banco, tmp_path):
    # remove_distinct reconstrói só a primeira instrução (sqlparse.parse(s)[0]):
    # no padrão oficial, o que vem depois do ";" é descartado antes de executar.
    pred, gold = "SELECT nome FROM aluno; DROP TABLE aluno", "SELECT nome FROM aluno"
    assert _oficial(banco, tmp_path, pred, gold, keep_distinct=False) == 1
    padrao = AvaliadorEX(Sandbox(SandboxConfig())).avaliar(banco, pred, gold)
    assert padrao.correto is True
    com_distinct = AvaliadorEX(Sandbox(SandboxConfig()), keep_distinct=True).avaliar(banco, pred, gold)
    assert com_distinct.correto is False and com_distinct.situacao == "erro_predicao"
    assert _alunos(banco) == 5


def test_predicao_sem_fim_e_tempo_esgotado(banco):
    infinita = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT count(*) FROM r"
    veredito = AvaliadorEX(Sandbox(SandboxConfig(timeout_s=2.0))).avaliar(banco, infinita, "SELECT 1")
    assert veredito.correto is False and veredito.situacao == "tempo_predicao"


def test_referencia_invalida_nao_tem_veredito(banco):
    veredito = AvaliadorEX(Sandbox(SandboxConfig())).avaliar(
        banco, "SELECT nome FROM aluno", "SELECT inexistente FROM aluno"
    )
    assert veredito.correto is None and veredito.situacao == "referencia_invalida"
    assert veredito.pred_executou is True and "referência" in veredito.erro


def test_referencia_executa_uma_vez_por_consulta(banco, monkeypatch):
    sandbox = Sandbox(SandboxConfig())
    chamadas = []
    original = sandbox.run
    monkeypatch.setattr(sandbox, "run", lambda *a, **k: chamadas.append(k.get("limite")) or original(*a, **k))
    avaliador = AvaliadorEX(sandbox)
    for pred in ("SELECT nome FROM aluno", "SELECT nome FROM aluno WHERE id = 1"):
        avaliador.avaliar(banco, pred, "SELECT nome FROM aluno")
    assert chamadas == [None, 5, 5]


def test_comparacao_preserva_o_estado_global_do_random():
    random.seed(7)
    antes = random.getstate()
    linhas = [(1, 2, 3, 4, 5), (6, 7, 8, 9, 10)]
    assert comparar(linhas, [tuple(reversed(r)) for r in linhas], ordem_importa=False)
    assert random.getstate() == antes


def test_copia_alterada_do_avaliador_e_recusada(tmp_path):
    pasta = tmp_path / "copia"
    shutil.copytree(ex_oficial._PASTA, pasta)
    (pasta / "exec_eval.py").write_text("TIMEOUT = 1\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="alterada"):
        carregar_oficial(pasta)


def test_copia_fixada_confere_com_o_upstream():
    for arquivo, blob in ex_oficial._BLOBS.items():
        assert ex_oficial.blob_git(ex_oficial._PASTA / arquivo) == blob
