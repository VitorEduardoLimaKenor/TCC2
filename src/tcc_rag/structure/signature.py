"""Assinatura estrutural das consultas (Seção 4.2.3).

Registra as construções relacionais empregadas, independentemente dos
identificadores concretos. A granularidade é deliberadamente mais grossa que a
da família canônica usada no particionamento: como famílias inteiras vão para
uma única partição, uma representação fina tornaria improvável qualquer
correspondência entre o pool e o teste, e a reordenação não teria o que ordenar.
"""

from __future__ import annotations

from typing import Any

import sqlglot
from sqlglot import exp

DIALECT = "sqlite"

_AGGREGATES = {"count", "sum", "avg", "min", "max", "total", "group_concat"}

#: Pesos da similaridade estrutural. Junções, agrupamento e operações de
#: conjunto pesam mais porque são as construções em que os modelos mais erram.
_WEIGHTS: dict[str, float] = {
    "n_joins": 2.0,
    "n_tables": 1.0,
    "agg_funcs": 1.5,
    "has_group_by": 2.0,
    "has_having": 2.0,
    "has_order_by": 0.75,
    "has_limit": 0.75,
    "has_distinct": 0.5,
    "n_subqueries": 1.5,
    "set_ops": 2.0,
    "n_predicates": 1.0,
    "n_select_items": 0.75,
}

EMPTY: dict[str, Any] = {
    "n_joins": 0,
    "n_tables": 0,
    "agg_funcs": [],
    "has_group_by": False,
    "has_having": False,
    "has_order_by": False,
    "has_limit": False,
    "has_distinct": False,
    "n_subqueries": 0,
    "set_ops": [],
    "n_predicates": 0,
    "n_select_items": 0,
}


def signature(sql: str) -> dict[str, Any] | None:
    """Extrai a assinatura. Devolve None se o SQL não puder ser analisado.

    O None é significativo: em C4 ele dispara o desvio para a ordenação da
    primeira etapa, e a frequência disso é reportada como métrica de robustez.
    """
    try:
        tree = sqlglot.parse_one(sql, read=DIALECT)
    except Exception:
        return None
    if tree is None:
        return None

    set_ops: set[str] = set()
    for klass, name in ((exp.Union, "union"), (exp.Except, "except"), (exp.Intersect, "intersect")):
        if list(tree.find_all(klass)):
            set_ops.add(name)

    aggs: set[str] = set()
    for func in tree.find_all(exp.Func):
        name = (func.sql_name() or "").lower()
        if name in _AGGREGATES:
            aggs.add(name)

    selects = list(tree.find_all(exp.Select))
    n_subqueries = max(0, len(selects) - 1) + len(list(tree.find_all(exp.Subquery)))

    n_predicates = 0
    for where in tree.find_all(exp.Where):
        n_predicates += len(list(where.find_all(exp.Predicate)))

    n_select_items = 0
    if selects:
        n_select_items = len(selects[0].expressions)

    tables = {t.name.lower() for t in tree.find_all(exp.Table) if t.name}

    return {
        "n_joins": len(list(tree.find_all(exp.Join))),
        "n_tables": len(tables),
        "agg_funcs": sorted(aggs),
        "has_group_by": bool(list(tree.find_all(exp.Group))),
        "has_having": bool(list(tree.find_all(exp.Having))),
        "has_order_by": bool(list(tree.find_all(exp.Order))),
        "has_limit": bool(list(tree.find_all(exp.Limit))),
        "has_distinct": bool(list(tree.find_all(exp.Distinct))),
        "n_subqueries": n_subqueries,
        "set_ops": sorted(set_ops),
        "n_predicates": n_predicates,
        "n_select_items": n_select_items,
    }


def _numeric_similarity(a: int, b: int) -> float:
    if a == b:
        return 1.0
    return 1.0 - abs(a - b) / max(1, max(a, b))


def _jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def similarity(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float:
    """Similaridade estrutural em [0, 1].

    Média ponderada por campo: numéricos por distância relativa, booleanos por
    igualdade, conjuntos por Jaccard.
    """
    if not a or not b:
        return 0.0

    total = 0.0
    acc = 0.0
    for field, weight in _WEIGHTS.items():
        va, vb = a.get(field), b.get(field)
        if isinstance(va, bool) or isinstance(vb, bool):
            score = 1.0 if bool(va) == bool(vb) else 0.0
        elif isinstance(va, list) or isinstance(vb, list):
            score = _jaccard(va or [], vb or [])
        else:
            score = _numeric_similarity(int(va or 0), int(vb or 0))
        acc += weight * score
        total += weight
    return acc / total if total else 0.0


def _join_bucket(n: int) -> int:
    return min(int(n), 3)


def compatible(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    """Compatibilidade estrutural, usada na métrica de acerto entre os k
    primeiros (Seção 4.7.3).

    A definição é explícita e binária: duas consultas são compatíveis quando
    coincidem na faixa de número de junções (0, 1, 2, 3 ou mais), na presença
    de agrupamento, na presença de filtragem posterior ao agrupamento, na
    presença de subconsulta e no conjunto de operações de conjunto empregadas.
    """
    if not a or not b:
        return False
    return (
        _join_bucket(a.get("n_joins", 0)) == _join_bucket(b.get("n_joins", 0))
        and bool(a.get("has_group_by")) == bool(b.get("has_group_by"))
        and bool(a.get("has_having")) == bool(b.get("has_having"))
        and (int(a.get("n_subqueries", 0)) > 0) == (int(b.get("n_subqueries", 0)) > 0)
        and set(a.get("set_ops", [])) == set(b.get("set_ops", []))
    )
