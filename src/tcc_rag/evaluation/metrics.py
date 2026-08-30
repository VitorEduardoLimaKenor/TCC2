"""Métricas de avaliação (Seção 4.7).

Desfecho principal: acurácia de execução. A consulta gerada é executada e seu
conjunto de resultados é comparado ao da consulta de referência. Uma mesma
pergunta admite formulações SQL distintas e semanticamente equivalentes, o que
inviabiliza a correspondência textual como desfecho.

A comparação segue a convenção do Spider: sensível à ordem apenas quando a
consulta de referência tem ORDER BY, e tolerante à permutação de colunas.
"""

from __future__ import annotations

from itertools import permutations
from typing import Any, Iterable

from ..data.normalize import canonicalize
from ..schemas import ExecResult
from ..structure import signature as sig

_MAX_PERMUTED_COLUMNS = 6


def _as_multiset(rows: Iterable[tuple]) -> list[tuple]:
    return sorted(rows, key=lambda row: tuple(str(v) for v in row))


def _project(rows: list[tuple], order: tuple[int, ...]) -> list[tuple]:
    return [tuple(row[i] for i in order) for row in rows]


def _gold_is_ordered(gold_sql: str) -> bool:
    s = sig.signature(gold_sql)
    return bool(s and s.get("has_order_by"))


def execution_match(pred: ExecResult, gold: ExecResult, gold_sql: str) -> bool:
    """Acurácia de execução para um par (predição, referência)."""
    if not pred.ok or not gold.ok:
        return False

    pred_rows = pred.rows or []
    gold_rows = gold.rows or []

    if len(pred_rows) != len(gold_rows):
        return False
    if not gold_rows:
        return True

    n_gold = len(gold_rows[0])
    n_pred = len(pred_rows[0])
    if n_gold != n_pred:
        return False

    ordered = _gold_is_ordered(gold_sql)

    def equal(a: list[tuple], b: list[tuple]) -> bool:
        if ordered:
            return a == b
        return _as_multiset(a) == _as_multiset(b)

    if equal(pred_rows, gold_rows):
        return True

    # Permutação de colunas: a consulta pode estar correta e ter projetado as
    # colunas em outra ordem. Acima de seis colunas o custo não compensa e a
    # comparação fica restrita à ordem direta.
    if n_gold <= _MAX_PERMUTED_COLUMNS:
        for order in permutations(range(n_gold)):
            if order == tuple(range(n_gold)):
                continue
            if equal(_project(pred_rows, order), gold_rows):
                return True

    return False


def exact_match(pred_sql: str, gold_sql: str) -> bool:
    """Correspondência estrutural normalizada.

    Não é a métrica oficial do Spider, que compara componentes da consulta
    analisada. É uma aproximação por forma canônica, e o texto deve reportá-la
    com a ressalva de que penaliza consultas corretas escritas de outro modo.
    """
    try:
        return canonicalize(pred_sql) == canonicalize(gold_sql)
    except Exception:
        return False


def structural_hit(selected_signatures: list[dict[str, Any] | None], gold_sql: str) -> bool:
    """Acerto estrutural entre os k primeiros (Seção 4.7.3).

    Verdadeiro quando ao menos uma demonstração recuperada tem assinatura
    compatível com a da consulta de referência. A definição de compatibilidade
    está em `structure.signature.compatible` e é reportada no texto.
    """
    gold = sig.signature(gold_sql)
    if gold is None:
        return False
    return any(sig.compatible(gold, s) for s in selected_signatures)


def summarize(rows: list[dict]) -> dict[str, Any]:
    """Agregados por condição e idioma, prontos para as tabelas do Capítulo 5."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["condition"], row["lang"])
        bucket = out.setdefault(
            key,
            {
                "n": 0,
                "ex": 0,
                "em": 0,
                "not_executable": 0,
                "fallback": 0,
                "structural_hit": 0,
                "prompt_tokens": [],
                "latency_ms": [],
                "by_difficulty": {},
            },
        )
        bucket["n"] += 1
        bucket["ex"] += int(bool(row.get("ex_correct")))
        bucket["em"] += int(bool(row.get("em_correct")))
        bucket["not_executable"] += int(not row.get("exec_ok"))
        bucket["fallback"] += int(bool(row.get("used_fallback")))
        bucket["structural_hit"] += int(bool(row.get("structural_hit")))
        bucket["prompt_tokens"].append(row.get("prompt_tokens") or 0)
        bucket["latency_ms"].append(row.get("latency_ms") or 0.0)

        diff = row.get("difficulty", "unknown")
        slot = bucket["by_difficulty"].setdefault(diff, {"n": 0, "ex": 0})
        slot["n"] += 1
        slot["ex"] += int(bool(row.get("ex_correct")))

    result: dict[str, Any] = {}
    for (condition, lang), bucket in sorted(out.items()):
        n = max(1, bucket["n"])
        tokens = sorted(bucket["prompt_tokens"])
        lat = sorted(bucket["latency_ms"])
        result[f"{condition}|{lang}"] = {
            "n": bucket["n"],
            "ex": bucket["ex"] / n,
            "em": bucket["em"] / n,
            "not_executable": bucket["not_executable"],
            "fallback_rate": bucket["fallback"] / n,
            "structural_hit": bucket["structural_hit"] / n,
            "prompt_tokens_mean": sum(tokens) / n,
            "latency_p50": _percentile(lat, 50),
            "latency_p95": _percentile(lat, 95),
            "by_difficulty": {
                d: {"n": v["n"], "ex": v["ex"] / max(1, v["n"])}
                for d, v in sorted(bucket["by_difficulty"].items())
            },
        }
    return result


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, int(round((pct / 100) * (len(values) - 1)))))
    return float(values[idx])
