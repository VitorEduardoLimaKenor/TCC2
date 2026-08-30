"""Relatório de vazamento entre pool e teste (Seção 4.5.1).

Este módulo é a defesa mais importante do experimento: se o pool contiver itens
equivalentes às perguntas de teste, o ganho medido é artificial e o trabalho
inteiro perde valor. O relatório roda **antes de toda execução** e a função
`assert_clean` levanta exceção em vez de avisar.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from ..schemas import Example
from .normalize import canonicalize
from .partition import POOL, TEST, VAL


class LeakageError(RuntimeError):
    pass


@dataclass
class LeakageReport:
    families_crossing: list[str] = field(default_factory=list)
    canonical_sql_shared: list[tuple[str, str]] = field(default_factory=list)
    near_duplicates: list[tuple[str, str, float]] = field(default_factory=list)
    unpaired: list[str] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not (
            self.families_crossing
            or self.canonical_sql_shared
            or self.near_duplicates
            or self.unpaired
        )

    def render(self) -> str:
        lines = ["=== Relatório de vazamento ==="]
        for key, value in sorted(self.counts.items()):
            lines.append(f"  {key}: {value}")
        lines.append(f"  famílias cruzando partições: {len(self.families_crossing)}")
        lines.append(f"  SQL canônico compartilhado pool/teste: {len(self.canonical_sql_shared)}")
        lines.append(f"  paráfrases quase idênticas: {len(self.near_duplicates)}")
        lines.append(f"  exemplos sem par PT/EN: {len(self.unpaired)}")
        for fam in self.families_crossing[:5]:
            lines.append(f"    ! família em mais de uma partição: {fam}")
        for a, b in self.canonical_sql_shared[:5]:
            lines.append(f"    ! mesmo SQL canônico: {a} ~ {b}")
        for a, b, score in self.near_duplicates[:5]:
            lines.append(f"    ! paráfrase {score:.3f}: {a} ~ {b}")
        lines.append("  VEREDITO: " + ("limpo" if self.clean else "VAZAMENTO DETECTADO"))
        return "\n".join(lines)


_TOKEN = re.compile(r"\w+", re.UNICODE)


def _shingles(text: str, n: int = 3) -> set[str]:
    tokens = _TOKEN.findall(text.lower())
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def audit(examples: list[Example], threshold: float = 0.92) -> LeakageReport:
    report = LeakageReport()

    by_partition: dict[str, list[Example]] = defaultdict(list)
    fam_partitions: dict[str, set[str]] = defaultdict(set)
    for ex in examples:
        by_partition[ex.partition].append(ex)
        fam_partitions[ex.family_id].add(ex.partition)

    report.counts = {
        "pool": len(by_partition[POOL]),
        "val": len(by_partition[VAL]),
        "test": len(by_partition[TEST]),
        "bancos": len({ex.db_id for ex in examples}),
        "famílias": len(fam_partitions),
    }

    report.families_crossing = sorted(
        fam for fam, parts in fam_partitions.items() if len(parts) > 1
    )

    report.unpaired = sorted(
        ex.example_id
        for ex in examples
        if not ex.question_pt.strip() or not ex.question_en.strip()
    )

    pool = by_partition[POOL]
    test = by_partition[TEST]

    # Mesmo SQL canônico atravessando pool e teste. Se o particionamento por
    # família estiver correto isto é vazio; a verificação é redundante de
    # propósito, porque é barata e o custo de um falso negativo é alto.
    canon_pool: dict[str, str] = {}
    for ex in pool:
        canon_pool.setdefault(f"{ex.db_id}||{canonicalize(ex.sql)}", ex.example_id)
    for ex in test:
        key = f"{ex.db_id}||{canonicalize(ex.sql)}"
        if key in canon_pool:
            report.canonical_sql_shared.append((canon_pool[key], ex.example_id))

    # Paráfrases: só faz sentido comparar dentro do mesmo banco.
    pool_by_db: dict[str, list[Example]] = defaultdict(list)
    for ex in pool:
        pool_by_db[ex.db_id].append(ex)

    for ex in test:
        for lang_attr in ("question_en", "question_pt"):
            probe = _shingles(getattr(ex, lang_attr))
            for other in pool_by_db.get(ex.db_id, []):
                score = _jaccard(probe, _shingles(getattr(other, lang_attr)))
                if score >= threshold:
                    report.near_duplicates.append(
                        (other.example_id, ex.example_id, round(score, 4))
                    )
                    break

    return report


def drop_near_duplicates(examples: list[Example], report: LeakageReport) -> list[Example]:
    """Remove do pool os itens apontados como paráfrase de alguma pergunta de
    teste. O teste nunca é alterado: ele está selado."""
    drop = {pool_id for pool_id, _, _ in report.near_duplicates}
    return [ex for ex in examples if not (ex.partition == POOL and ex.example_id in drop)]


def assert_clean(examples: list[Example], threshold: float = 0.92) -> LeakageReport:
    report = audit(examples, threshold)
    if not report.clean:
        raise LeakageError(report.render())
    return report
