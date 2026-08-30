"""Particionamento por família e recorte de domínio único (Seção 4.5).

A partição oficial do Spider separa *bancos* entre treino e desenvolvimento,
arranjo feito para medir generalização entre domínios. Recuperar demonstrações
do mesmo banco da pergunta exige o contrário: dentro de cada banco, três
partições disjuntas.

Duas invariantes governam a alocação e são verificadas em `leakage.py`:

1. famílias inteiras vão para uma única partição, nunca linhas soltas;
2. a pergunta em português e sua correspondente em inglês são o mesmo exemplo,
   portanto compartilham a partição por construção.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import replace

from ..config import DataConfig
from ..schemas import Example

POOL, VAL, TEST = "pool", "val", "test"


def eligible_databases(examples: list[Example], cfg: DataConfig) -> list[str]:
    """Bancos com volume suficiente para o recorte de domínio único."""
    by_db: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_db[ex.db_id].append(ex)

    keep: list[str] = []
    for db_id, items in sorted(by_db.items()):
        families = {ex.family_id for ex in items}
        if len(items) >= cfg.min_questions_per_db and len(families) >= cfg.min_families_per_db:
            keep.append(db_id)
    return keep


def database_report(examples: list[Example]) -> list[dict]:
    """Contagem por banco, para a porta de decisão de 04/09."""
    by_db: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        by_db[ex.db_id].append(ex)
    rows = []
    for db_id, items in by_db.items():
        rows.append(
            {
                "db_id": db_id,
                "n_questions": len(items),
                "n_families": len({ex.family_id for ex in items}),
            }
        )
    rows.sort(key=lambda r: (-r["n_families"], -r["n_questions"]))
    return rows


def partition(examples: list[Example], cfg: DataConfig) -> list[Example]:
    """Aloca famílias a pool/val/test, dentro de cada banco.

    A alocação é gulosa sobre famílias embaralhadas: a família vai para a
    partição que estiver mais distante da sua cota. Isso mantém as proporções
    aproximadas mesmo quando as famílias têm tamanhos muito desiguais.
    """
    rng = random.Random(cfg.seed)
    keep = set(eligible_databases(examples, cfg))
    out: list[Example] = []

    quotas = {POOL: cfg.pool_ratio, VAL: cfg.val_ratio, TEST: cfg.test_ratio}
    total_ratio = sum(quotas.values())
    quotas = {k: v / total_ratio for k, v in quotas.items()}

    by_db: dict[str, list[Example]] = defaultdict(list)
    for ex in examples:
        if ex.db_id in keep:
            by_db[ex.db_id].append(ex)

    for db_id in sorted(by_db):
        items = by_db[db_id]
        families: dict[str, list[Example]] = defaultdict(list)
        for ex in items:
            families[ex.family_id].append(ex)

        order = sorted(families)
        rng.shuffle(order)

        assigned: dict[str, str] = {}
        counts = {POOL: 0, VAL: 0, TEST: 0}
        total = len(items)

        for fam in order:
            deficit = {
                name: quotas[name] - (counts[name] / total if total else 0.0)
                for name in counts
            }
            target = max(deficit, key=lambda n: deficit[n])
            assigned[fam] = target
            counts[target] += len(families[fam])

        for ex in items:
            out.append(replace(ex, partition=assigned[ex.family_id]))

    return out


def summarize(examples: list[Example]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ex in examples:
        summary[ex.db_id][ex.partition] += 1
        summary["__total__"][ex.partition] += 1
    return {k: dict(v) for k, v in summary.items()}
