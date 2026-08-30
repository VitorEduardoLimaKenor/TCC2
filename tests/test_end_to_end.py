"""Fumaça do pipeline inteiro, sem GPU e sem modelo.

Exercita o caminho completo — partição, auditoria, indexação no Qdrant,
seleção nas cinco condições, montagem de prompt, execução no sandbox e escrita
do log — com um gerador falso e um vetorizador determinístico.

Serve para pegar erro de fiação antes de gastar horas de GPU, e é o teste que
deve rodar em toda mudança.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from tcc_rag.config import ExperimentConfig
from tcc_rag.data import leakage as leak
from tcc_rag.data import partition as part
from tcc_rag.data.normalize import family_id
from tcc_rag.data.spider import DbSchema
from tcc_rag.generation.llm import DummyBackend
from tcc_rag.logging_io import RunLog
from tcc_rag.pipeline import build_demos, build_test_items
from tcc_rag.retrieval.index import DemoIndex
from tcc_rag.retrieval.selectors import DemoSelector
from tcc_rag.runner import Runner
from tcc_rag.schemas import Example

DIM = 32


class StubEmbedder:
    """Vetor determinístico derivado do texto. Sem modelo, sem download."""

    dimension = DIM

    def encode(self, texts: list[str], kind: str = "query") -> np.ndarray:
        return np.stack([self.encode_one(t, kind) for t in texts])

    def encode_one(self, text: str, kind: str = "query") -> np.ndarray:
        seed = int.from_bytes(
            hashlib.sha256(text.lower().encode("utf-8")).digest()[:8], "big"
        )
        raw = np.random.default_rng(seed).normal(size=DIM).astype(np.float32)
        return raw / np.linalg.norm(raw)


TEMPLATES = [
    ("SELECT nome FROM aluno WHERE nota > {n}", "alunos com nota acima de {n}"),
    ("SELECT count(*) FROM aluno WHERE turma = {n}", "quantos alunos na turma {n}"),
    ("SELECT turma, count(*) FROM aluno GROUP BY turma HAVING count(*) > {n}",
     "turmas com mais de {n} alunos"),
    ("SELECT a.nome FROM aluno a JOIN nota b ON a.id = b.aid WHERE b.valor > {n}",
     "nomes de alunos com avaliacao acima de {n}"),
    ("SELECT nome FROM aluno ORDER BY nota DESC LIMIT {n}",
     "os {n} alunos de maior nota"),
]


@pytest.fixture
def universo(tmp_path: Path):
    db_path = tmp_path / "escola.sqlite"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE aluno (id INT, nome TEXT, nota REAL, turma INT)")
    conn.execute("CREATE TABLE nota (id INT, aid INT, valor REAL)")
    conn.executemany(
        "INSERT INTO aluno VALUES (?,?,?,?)",
        [(i, f"Aluno {i}", 5.0 + (i % 5), i % 3) for i in range(40)],
    )
    conn.executemany(
        "INSERT INTO nota VALUES (?,?,?)", [(i, i, 4.0 + (i % 6)) for i in range(40)]
    )
    conn.commit()
    conn.close()

    exemplos: list[Example] = []
    idx = 0
    for sql_tpl, q_tpl in TEMPLATES:
        for n in range(1, 13):
            sql = sql_tpl.format(n=n)
            exemplos.append(
                Example(
                    example_id=f"ex{idx}",
                    db_id="escola",
                    question_en=q_tpl.format(n=n),
                    question_pt=q_tpl.format(n=n) + " (pt)",
                    sql=sql,
                    difficulty="medium",
                    family_id=family_id("escola", sql),
                )
            )
            idx += 1

    schema = DbSchema(
        db_id="escola",
        ddl="CREATE TABLE aluno (id INT, nome TEXT, nota REAL, turma INT);\n"
            "CREATE TABLE nota (id INT, aid INT, valor REAL);",
        sqlite_path=str(db_path),
        n_tables=2,
        n_columns=7,
    )
    return exemplos, {"escola": schema}, tmp_path


def test_pipeline_completo(universo):
    exemplos, schemas, tmp_path = universo

    cfg = ExperimentConfig()
    cfg.paths.work_dir = str(tmp_path / "artifacts")
    cfg.data.min_questions_per_db = 10
    cfg.data.min_families_per_db = 3
    cfg.qdrant.mode = "local"
    cfg.qdrant.path = str(tmp_path / "qdrant")
    cfg.generation.backend = "dummy"
    cfg.retrieval.k = 2
    cfg.retrieval.n_candidates = 6

    # --- partição e auditoria
    atribuidos = part.partition(exemplos, cfg.data)
    relatorio = leak.audit(atribuidos, cfg.data.near_duplicate_threshold)
    assert not relatorio.families_crossing, relatorio.render()
    assert relatorio.counts["pool"] > 0 and relatorio.counts["test"] > 0

    # --- índice
    pool = build_demos(atribuidos, partition=part.POOL)
    assert pool and all(d.lang in ("pt", "en") for d in pool)

    embedder = StubEmbedder()
    vetores = embedder.encode([d.question_delex for d in pool], kind="passage")
    index = DemoIndex(cfg.qdrant, dimension=DIM)
    index.recreate()
    assert index.upsert(pool, vetores) == len(pool)
    assert index.count(lang="pt", partition="pool") == len(pool) // 2

    # --- execução
    itens = build_test_items(atribuidos, partition=part.TEST)
    assert itens
    backend = DummyBackend(cfg.generation)
    seletor = DemoSelector(cfg, index, embedder, pool)
    log = RunLog(tmp_path / "run.jsonl")
    runner = Runner(cfg, backend, seletor, schemas, log, run_id="smoke")

    escritos = runner.run(itens, ("C0", "C1", "C2", "C3", "C4"))
    assert all(n == len(itens) for n in escritos.values()), escritos

    linhas = log.rows()
    assert len(linhas) == 5 * len(itens)

    por_condicao = {c: [r for r in linhas if r["condition"] == c] for c in escritos}

    # C0 nunca recebe demonstração; as demais recebem k.
    assert all(r["n_demos_used"] == 0 for r in por_condicao["C0"])
    for c in ("C1", "C2", "C3", "C4"):
        assert all(r["n_demos_used"] > 0 for r in por_condicao[c]), c

    # C4 é a única que produz consulta preliminar.
    assert all(r["preliminary_sql"] is None for r in por_condicao["C2"])
    assert all(r["preliminary_sql"] is not None for r in por_condicao["C4"])

    # A recuperação nunca devolve item de fora do pool nem de outro idioma.
    uids = {d.uid: d for d in pool}
    for c in ("C2", "C3", "C4"):
        for r in por_condicao[c]:
            for uid in r["selected_demo_ids"]:
                assert uids[uid].partition == "pool"
                assert uids[uid].lang == r["lang"]

    # O gabarito do item de teste jamais aparece entre as demonstrações.
    ids_teste = {i.example_id for i in itens}
    for r in linhas:
        for uid in r["selected_demo_ids"]:
            assert uids[uid].example_id not in ids_teste

    index.close()


def test_retomada_nao_duplica(universo):
    exemplos, schemas, tmp_path = universo

    cfg = ExperimentConfig()
    cfg.paths.work_dir = str(tmp_path / "art2")
    cfg.data.min_questions_per_db = 10
    cfg.data.min_families_per_db = 3
    cfg.qdrant.mode = "local"
    cfg.qdrant.path = str(tmp_path / "qdrant2")
    cfg.generation.backend = "dummy"
    cfg.retrieval.k = 2

    atribuidos = part.partition(exemplos, cfg.data)
    pool = build_demos(atribuidos, partition=part.POOL)
    embedder = StubEmbedder()
    index = DemoIndex(cfg.qdrant, dimension=DIM)
    index.recreate()
    index.upsert(pool, embedder.encode([d.question_delex for d in pool], kind="passage"))

    itens = build_test_items(atribuidos, partition=part.TEST)
    caminho = tmp_path / "retomada.jsonl"

    def executa():
        log = RunLog(caminho)
        seletor = DemoSelector(cfg, index, embedder, pool)
        runner = Runner(cfg, DummyBackend(cfg.generation), seletor, schemas, log, "r1")
        return runner.run(itens, ("C0", "C2"))

    primeira = executa()
    segunda = executa()

    assert sum(primeira.values()) == 2 * len(itens)
    assert sum(segunda.values()) == 0, "a segunda execução não pode reescrever nada"
    assert len(RunLog(caminho).rows()) == 2 * len(itens)

    index.close()
