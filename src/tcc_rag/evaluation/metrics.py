"""Métricas de avaliação (Seção 4.7).

Desfecho principal: acurácia de execução, decidida pelo avaliador oficial do
Spider em `ex_oficial`. A correspondência exata e a dificuldade oficiais vêm de
`em_oficial`. Este módulo guarda as medidas próprias e os agregados. O
comparador próprio de EX usado até a validação de setembro de 2026 foi
substituído na E02; ele continua no commit 267a4fe, de onde a comparação entre
avaliadores o reproduz.

Nos logs, os nomes separam o oficial do próprio. `em_oficial` e
`dificuldade_oficial` são do avaliador oficial; `em_canonico_proprio` e
`dificuldade_aprox` são aproximações deste projeto. Logs anteriores à E03
gravaram as mesmas aproximações como `em_correct` e `difficulty`, e não têm EM
nem dificuldade oficiais.
"""

from __future__ import annotations

from typing import Any

from ..data.normalize import canonicalize
from ..structure import signature as sig


def em_canonico_proprio(pred_sql: str, gold_sql: str) -> bool:
    """Igualdade das formas canônicas próprias (a que define as famílias).

    Não é o EM oficial do Spider (`em_oficial`), que compara componentes da
    consulta analisada, sem valores. É uma medida própria, registrada como
    `em_canonico_proprio` (`em_correct` nos logs anteriores à E03), e o texto deve
    reportá-la com a ressalva de que penaliza consultas corretas escritas de
    outro modo.
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
    """Agregados por condição e idioma, prontos para as tabelas do Capítulo 5.

    Linha sem veredito de EX (`ex_correct` nulo: referência inválida) fica fora
    de todos os denominadores e é contada em `referencia_invalida`.

    `em_oficial` é calculado só sobre linhas com veredito do EM oficial; a
    referência que o analisador oficial recusa sai desse denominador e é contada
    em `em_referencia_nao_analisavel`. Linhas sem EM oficial (logs anteriores à
    E03) são contadas em `sem_em_oficial`, e nenhuma medida própria as
    substitui. `by_difficulty` usa a dificuldade oficial; a linha sem ela fica
    em "nao_classificada".
    """
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["condition"], row["lang"])
        bucket = out.setdefault(
            key,
            {
                "n": 0,
                "ex": 0,
                "em_oficial": 0,
                "em_n": 0,
                "em_referencia_nao_analisavel": 0,
                "em_predicao_nao_analisavel": 0,
                "sem_em_oficial": 0,
                "em_canonico_proprio": 0,
                "not_executable": 0,
                "fallback": 0,
                "structural_hit": 0,
                "prompt_tokens": [],
                "latency_ms": [],
                "by_difficulty": {},
                "referencia_invalida": 0,
                "tempo_predicao": 0,
                "acertos_ambos_vazios": 0,
            },
        )
        if row.get("ex_correct") is None:
            bucket["referencia_invalida"] += 1
            continue
        bucket["n"] += 1
        bucket["tempo_predicao"] += int(row.get("ex_status") == "tempo_predicao")
        bucket["acertos_ambos_vazios"] += int(bool(row.get("ex_ambos_vazios")))
        bucket["ex"] += int(bool(row.get("ex_correct")))
        if "em_status" not in row:
            bucket["sem_em_oficial"] += 1
        elif row.get("em_oficial") is None:
            bucket["em_referencia_nao_analisavel"] += 1
        else:
            bucket["em_n"] += 1
            bucket["em_oficial"] += int(bool(row["em_oficial"]))
            bucket["em_predicao_nao_analisavel"] += int(row["em_status"] == "predicao_nao_analisavel")
        # Antes da E03, a mesma medida própria era gravada como `em_correct`.
        proprio = row["em_canonico_proprio"] if "em_canonico_proprio" in row else row.get("em_correct")
        bucket["em_canonico_proprio"] += int(bool(proprio))
        bucket["not_executable"] += int(not row.get("exec_ok"))
        bucket["fallback"] += int(bool(row.get("used_fallback")))
        bucket["structural_hit"] += int(bool(row.get("structural_hit")))
        bucket["prompt_tokens"].append(row.get("prompt_tokens") or 0)
        bucket["latency_ms"].append(row.get("latency_ms") or 0.0)

        diff = row.get("dificuldade_oficial") or "nao_classificada"
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
            "em_oficial": bucket["em_oficial"] / bucket["em_n"] if bucket["em_n"] else None,
            "em_n": bucket["em_n"],
            "em_referencia_nao_analisavel": bucket["em_referencia_nao_analisavel"],
            "em_predicao_nao_analisavel": bucket["em_predicao_nao_analisavel"],
            "sem_em_oficial": bucket["sem_em_oficial"],
            "em_canonico_proprio": bucket["em_canonico_proprio"] / n,
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
            "referencia_invalida": bucket["referencia_invalida"],
            "tempo_predicao": bucket["tempo_predicao"],
            "acertos_ambos_vazios": bucket["acertos_ambos_vazios"],
        }
    return result


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    idx = min(len(values) - 1, max(0, int(round((pct / 100) * (len(values) - 1)))))
    return float(values[idx])
