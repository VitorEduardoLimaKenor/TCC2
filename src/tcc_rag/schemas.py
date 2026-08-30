"""Estruturas de dados que circulam pelo pipeline.

Uma regra atravessa o módulo inteiro e é verificada em teste: `gold_sql` só
existe em `TestItem`, e `TestItem` nunca é passado para o recuperador. O que
chega ao recuperador é `RetrievalQuery`, que não tem o campo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

Partition = str  # "pool" | "val" | "test"


@dataclass(frozen=True)
class Example:
    """Um item do conjunto bilíngue, antes de virar demonstração ou teste."""

    example_id: str
    db_id: str
    question_en: str
    question_pt: str
    sql: str
    difficulty: str = "unknown"
    family_id: str = ""
    partition: Partition = ""
    source: str = "spider"


@dataclass(frozen=True)
class Demo:
    """Registro indexado no Qdrant (Seção 4.2.1 da monografia).

    Só `vector` participa da busca; o resto é metadado usado na reordenação,
    na montagem do prompt e na auditoria.
    """

    uid: str
    example_id: str
    db_id: str
    lang: str
    question_raw: str
    question_delex: str
    sql: str
    signature: dict[str, Any]
    tables: list[str]
    columns: list[str]
    family_id: str
    partition: Partition
    difficulty: str = "unknown"

    def payload(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "example_id": self.example_id,
            "db_id": self.db_id,
            "lang": self.lang,
            "question_raw": self.question_raw,
            "question_delex": self.question_delex,
            "sql": self.sql,
            "signature": self.signature,
            "tables": self.tables,
            "columns": self.columns,
            "family_id": self.family_id,
            "partition": self.partition,
            "difficulty": self.difficulty,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "Demo":
        return cls(
            uid=payload["uid"],
            example_id=payload["example_id"],
            db_id=payload["db_id"],
            lang=payload["lang"],
            question_raw=payload["question_raw"],
            question_delex=payload["question_delex"],
            sql=payload["sql"],
            signature=payload.get("signature", {}),
            tables=payload.get("tables", []),
            columns=payload.get("columns", []),
            family_id=payload.get("family_id", ""),
            partition=payload.get("partition", ""),
            difficulty=payload.get("difficulty", "unknown"),
        )


@dataclass(frozen=True)
class RetrievalQuery:
    """O que o recuperador enxerga. Sem gabarito, por construção."""

    example_id: str
    db_id: str
    lang: str
    question_raw: str
    question_delex: str


@dataclass(frozen=True)
class TestItem:
    """Item de avaliação. O gabarito fica aqui e só o avaliador o lê."""

    example_id: str
    db_id: str
    lang: str
    question_raw: str
    question_delex: str
    gold_sql: str
    difficulty: str = "unknown"

    def as_query(self) -> RetrievalQuery:
        return RetrievalQuery(
            example_id=self.example_id,
            db_id=self.db_id,
            lang=self.lang,
            question_raw=self.question_raw,
            question_delex=self.question_delex,
        )


@dataclass
class Selection:
    """Resultado da seleção de demonstrações, com o rastro do como."""

    demos: list[Demo] = field(default_factory=list)
    candidate_ids: list[str] = field(default_factory=list)
    candidate_scores: list[float] = field(default_factory=list)
    preliminary_sql: str | None = None
    parse_ok: bool | None = None
    signature: dict[str, Any] | None = None
    used_fallback: bool = False


@dataclass
class Generation:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


@dataclass
class ExecResult:
    ok: bool
    rows: list[tuple] | None = None
    error: str | None = None
    timed_out: bool = False
