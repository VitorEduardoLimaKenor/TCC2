"""Assinatura, seleção e o que chega ao prompt (E05).

Casos pequenos com resposta conhecida à mão: contagens da assinatura,
similaridade, compatibilidade, desempates e a equivalência de C4 com λ=1 e C3.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tcc_rag.config import ExperimentConfig
from tcc_rag.data.spider import DbSchema
from tcc_rag.generation.llm import DummyBackend
from tcc_rag.generation.prompt import build_prompt
from tcc_rag.logging_io import RunLog
from tcc_rag.retrieval.selectors import DemoSelector
from tcc_rag.runner import Runner
from tcc_rag.schemas import Demo, RetrievalQuery, Selection
from tcc_rag.schemas import TestItem as ItemDeTeste  # nome sem "Test": o pytest não tenta coletá-lo
from tcc_rag.structure import signature as sig

# ----------------------------------------------------------- assinatura

CONTAGENS = [
    # (caso, sql, n_subqueries, n_predicates)
    ("sem_subconsulta", "SELECT a FROM t WHERE x = 1 AND y > 2", 0, 2),
    ("escalar", "SELECT a FROM t WHERE x > (SELECT avg(x) FROM t)", 1, 1),
    ("in_com_where_interno", "SELECT a FROM t WHERE x IN (SELECT y FROM u WHERE z = 1)", 1, 2),
    ("not_exists", "SELECT a FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.k = t.k)", 1, 2),
    ("tabela_derivada", "SELECT s.a FROM (SELECT a FROM t GROUP BY a) AS s", 1, 0),
    ("duas_subconsultas", "SELECT a FROM t WHERE x > (SELECT avg(x) FROM t) AND y IN (SELECT y FROM u)", 2, 2),
    ("subconsulta_no_having", "SELECT a FROM t GROUP BY a HAVING count(*) > (SELECT count(*) FROM u)", 1, 0),
    ("union", "SELECT a FROM t UNION SELECT a FROM u", 0, 0),
    ("except_com_in", "SELECT a FROM t EXCEPT SELECT a FROM u WHERE b IN (SELECT b FROM v)", 1, 1),
    ("in_de_union", "SELECT a FROM t WHERE x IN (SELECT a FROM u UNION SELECT a FROM v)", 1, 1),
    ("intersect_com_where", "SELECT a FROM t WHERE x = 1 INTERSECT SELECT a FROM t WHERE y = 2", 0, 2),
]


@pytest.mark.parametrize("caso,sql,subconsultas,predicados", CONTAGENS, ids=[c[0] for c in CONTAGENS])
def test_contagens_da_assinatura(caso, sql, subconsultas, predicados):
    s = sig.signature(sql)
    assert (s["n_subqueries"], s["n_predicates"]) == (subconsultas, predicados)


def test_demais_campos_da_assinatura():
    s = sig.signature("SELECT DISTINCT T1.a, count(*) FROM t AS T1 JOIN u AS T2 ON T1.k = T2.k "
                      "JOIN v ON v.k = T2.k WHERE T1.x > 1 GROUP BY T1.a ORDER BY count(*) DESC LIMIT 3")
    assert s == {
        "n_joins": 2, "n_tables": 3, "agg_funcs": ["count"], "has_group_by": True,
        "has_having": False, "has_order_by": True, "has_limit": True, "has_distinct": True,
        "n_subqueries": 0, "set_ops": [], "n_predicates": 1, "n_select_items": 2,
    }


# ------------------------------------------------- similaridade e compatibilidade

def test_similaridade_conhecida():
    """Só n_tables (1 × 2) e n_joins (0 × 1) diferem: 1 − (2·1 + 1·0,5)/15,75."""
    a = sig.signature("SELECT a FROM t WHERE x = 1")
    b = sig.signature("SELECT a FROM t JOIN u ON t.k = u.k WHERE x = 1")
    total = sum(sig._WEIGHTS.values())
    assert total == pytest.approx(15.75)
    assert sig.similarity(a, b) == pytest.approx(1 - (2.0 * 1.0 + 1.0 * 0.5) / total)
    assert sig.similarity(a, b) == pytest.approx(sig.similarity(b, a))


def test_similaridade_nos_limites():
    a = sig.signature("SELECT a FROM t")
    assert sig.similarity(a, a) == 1.0
    assert sig.similarity(a, None) == 0.0 and sig.similarity({}, a) == 0.0
    b = sig.signature("SELECT max(x), count(*) FROM t JOIN u ON t.k = u.k GROUP BY y HAVING "
                      "count(*) > 1 ORDER BY 1 LIMIT 1")
    assert 0.0 <= sig.similarity(a, b) < sig.similarity(a, a)


def test_compatibilidade_separa_operacao_de_conjunto_e_subconsulta():
    """Antes da E05, todo UNION tinha n_subqueries > 0, e um UNION com e outro
    sem subconsulta pareciam compatíveis."""
    com = sig.signature("SELECT a FROM t WHERE x IN (SELECT x FROM u) UNION SELECT a FROM v")
    sem = sig.signature("SELECT a FROM t UNION SELECT a FROM v")
    assert sig.compatible(sem, sig.signature("SELECT b FROM w UNION SELECT b FROM z"))
    assert not sig.compatible(com, sem)


def test_compatibilidade_agrupa_tres_ou_mais_junções():
    tres = sig.signature("SELECT a FROM t JOIN u ON 1 JOIN v ON 1 JOIN w ON 1")
    cinco = sig.signature("SELECT a FROM t JOIN u ON 1 JOIN v ON 1 JOIN w ON 1 JOIN y ON 1 JOIN z ON 1")
    duas = sig.signature("SELECT a FROM t JOIN u ON 1 JOIN v ON 1")
    assert sig.compatible(tres, cinco) and not sig.compatible(tres, duas)


# ---------------------------------------------------------------- seleção

def _demo(uid: str, sql: str, pergunta: str = "q") -> Demo:
    return Demo(uid=f"{uid}::pt", example_id=uid, db_id="db", lang="pt", question_raw=pergunta,
                question_delex=pergunta, sql=sql, signature=sig.signature(sql) or {}, tables=[],
                columns=[], family_id=uid, partition="pool")


class IndiceFalso:
    """Busca exata por escores fixos, como o Qdrant local: decrescente, `limit` primeiros.

    Escores iguais saem na ordem de inserção, que faz o papel da ordem interna
    do Qdrant nos empates.
    """

    def __init__(self, escores: dict[str, dict[str, float]]):
        self.escores = escores  # texto da consulta -> uid -> escore
        self.demos: dict[str, Demo] = {}

    def search(self, vector, db_id, lang, partition="pool", limit=10, exclude_family=None):
        texto = vector  # o vetorizador falso devolve o próprio texto
        pares = [(self.demos[f"{uid}::pt"], s) for uid, s in self.escores[texto].items()]
        return sorted(pares, key=lambda p: -p[1])[:limit]


class VetorizadorFalso:
    def encode_one(self, text, kind="query"):
        return text


def _seletor(lam: float, k: int, n: int, demos: list[Demo], escores, rascunho: str):
    cfg = ExperimentConfig()
    cfg.retrieval.k, cfg.retrieval.n_candidates, cfg.retrieval.lambda_question = k, n, lam
    indice = IndiceFalso(escores)
    indice.demos = {d.uid: d for d in demos}
    return DemoSelector(cfg, indice, VetorizadorFalso(), demos, drafter=lambda q: rascunho)


POOL = [
    _demo("d1", "SELECT a FROM t WHERE x = 1"),
    _demo("d2", "SELECT count(*) FROM t GROUP BY a"),
    _demo("d3", "SELECT a FROM t JOIN u ON t.k = u.k"),
    _demo("d4", "SELECT a FROM t WHERE x = 2"),
    _demo("d5", "SELECT count(*) FROM t GROUP BY b"),
    _demo("d6", "SELECT a FROM t ORDER BY x LIMIT 1"),
]
CONSULTA = RetrievalQuery(example_id="e1", db_id="db", lang="pt",
                          question_raw="quantos por grupo", question_delex="quantos por <col>")
ESCORES = {
    "quantos por <col>": {"d1": 0.91, "d3": 0.90, "d6": 0.88, "d4": 0.85, "d2": 0.80, "d5": 0.79},
    "quantos por grupo": {"d4": 0.95, "d1": 0.93, "d2": 0.70, "d3": 0.60, "d5": 0.50, "d6": 0.40},
}


@pytest.mark.parametrize("k", [1, 2, 4])
def test_c4_com_lambda_um_e_c3(k):
    """Com λ=1 a parte estrutural some: C4 devolve os k primeiros candidatos da
    busca pela pergunta delexicalizada, que são o C3."""
    seletor = _seletor(1.0, k, 25, POOL, ESCORES, "SELECT count(*) FROM t GROUP BY a")
    c3, c4 = seletor.select(CONSULTA, "C3"), seletor.select(CONSULTA, "C4")
    assert [d.uid for d in c4.demos] == [d.uid for d in c3.demos]
    ddl = "CREATE TABLE t (a INT);"
    assert build_prompt("q", ddl, c4.demos, "pt", 4096).sha256 == build_prompt("q", ddl, c3.demos, "pt", 4096).sha256


def test_c4_com_lambda_menor_promove_estrutura_compativel():
    seletor = _seletor(0.0, 2, 25, POOL, ESCORES, "SELECT count(*) FROM t GROUP BY c")
    assert [d.uid for d in seletor.select(CONSULTA, "C4").demos] == ["d2::pt", "d5::pt"]


def test_desempate_preserva_a_ordem_da_primeira_etapa():
    """Escore combinado igual (λ=0 e assinaturas idênticas): vale a ordem da
    busca vetorial, porque a ordenação do Python é estável."""
    seletor = _seletor(0.0, 3, 25, POOL, ESCORES, "SELECT a FROM t WHERE x = 9")
    assert [d.uid for d in seletor.select(CONSULTA, "C4").demos][:2] == ["d1::pt", "d4::pt"]


def test_c4_so_reordena_os_n_candidatos():
    """Com N=2, a demonstração estruturalmente idêntica fora dos dois primeiros não entra."""
    seletor = _seletor(0.0, 2, 2, POOL, ESCORES, "SELECT count(*) FROM t GROUP BY c")
    c4 = seletor.select(CONSULTA, "C4")
    assert c4.candidate_ids == ["d1::pt", "d3::pt"]
    assert {d.uid for d in c4.demos} == {"d1::pt", "d3::pt"}


def test_c2_usa_a_pergunta_original_e_c3_a_delexicalizada():
    seletor = _seletor(1.0, 2, 25, POOL, ESCORES, "")
    assert [d.uid for d in seletor.select(CONSULTA, "C2").demos] == ["d4::pt", "d1::pt"]
    assert [d.uid for d in seletor.select(CONSULTA, "C3").demos] == ["d1::pt", "d3::pt"]


# ------------------------------------------------------ o que chega ao prompt

def test_prompt_registra_demonstracoes_efetivas_e_removidas():
    demos = [_demo(f"d{i}", "SELECT a FROM t", "pergunta " * 30) for i in range(6)]
    built = build_prompt("pergunta", "CREATE TABLE t (a INT);" * 20, demos, "pt", 600, 64)
    assert built.n_demos_dropped > 0
    assert built.demos + built.removidas == demos
    assert built.demos == demos[: built.n_demos_used]
    assert all(f"Pergunta: {d.question_raw}" in built.text for d in built.demos)
    assert all(built.text.count(d.question_raw) <= len(built.demos) for d in built.removidas)


@pytest.fixture
def banco(tmp_path: Path):
    db = tmp_path / "db.sqlite"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE t (a INT, x INT)")
    conn.execute("INSERT INTO t VALUES (1, 1)")
    conn.commit()
    conn.close()
    return db


def test_log_mede_o_acerto_estrutural_sobre_o_que_o_modelo_recebeu(banco, tmp_path):
    """A única demonstração compatível com a referência é a última da seleção,
    e o orçamento a corta: o acerto tem de ser falso, e o log tem de mostrar o corte."""
    longas = [_demo(f"d{i}", "SELECT a FROM t WHERE x = 1", "pergunta longa " * 40) for i in range(3)]
    compativel = _demo("d9", "SELECT count(*) FROM t GROUP BY a", "pergunta longa " * 40)
    selecao = longas + [compativel]
    cfg = ExperimentConfig()
    cfg.paths.work_dir = str(tmp_path / "artifacts")
    cfg.generation.backend = "dummy"
    cfg.generation.max_context_tokens, cfg.generation.max_new_tokens = 900, 64
    tabela = {"db_id": "db", "table_names_original": ["t"],
              "column_names_original": [[-1, "*"], [0, "a"], [0, "x"]], "foreign_keys": []}
    schemas = {"db": DbSchema("db", "CREATE TABLE t (a INT, x INT);", str(banco), 1, 2, tables_entry=tabela)}
    seletor = SimpleNamespace(select=lambda q, c: Selection(demos=list(selecao)), drafter=None)
    item = ItemDeTeste(example_id="e1", db_id="db", lang="pt", question_raw="quantos por a",
                       question_delex="quantos por a", gold_sql="SELECT count(*) FROM t GROUP BY a")
    log = RunLog(tmp_path / "r.jsonl")
    Runner(cfg, DummyBackend(cfg.generation), seletor, schemas, log, "r").run([item], ("C2",))
    linha = RunLog(tmp_path / "r.jsonl").rows()[0]
    assert linha["selected_demo_ids"] == [d.uid for d in selecao]
    assert linha["prompt_demos_removidas"] and linha["prompt_demos_removidas"][-1] == "d9::pt"
    assert linha["prompt_demo_ids"] + linha["prompt_demos_removidas"] == linha["selected_demo_ids"]
    assert linha["k_efetivo"] == len(linha["prompt_demo_ids"]) == linha["n_demos_used"]
    assert linha["structural_hit"] is False
    assert sig.compatible(sig.signature(item.gold_sql), compativel.signature)
    assert linha["prompt_no_limite_de_entrada"] is False
    assert linha["prompt_tokens_estimados"] > 0
