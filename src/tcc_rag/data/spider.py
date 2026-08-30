"""Carga do Spider e serialização de esquema.

O esquema é serializado em DDL e é **idêntico em todas as condições**
experimentais, inclusive nas que não recebem demonstrações (Seção 4.4). Por
isso a serialização é determinística: ordem de tabelas e colunas preservada do
`tables.json`, sem qualquer dependência da pergunta.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .delex import delexicalize
from .normalize import family_id
from ..schemas import Example


@dataclass(frozen=True)
class DbSchema:
    db_id: str
    ddl: str
    sqlite_path: str
    n_tables: int
    n_columns: int


def _quote(name: str) -> str:
    return f'"{name}"' if not name.replace("_", "").isalnum() else name


def build_ddl(entry: dict[str, Any]) -> str:
    """Monta o DDL de um banco a partir de uma entrada do `tables.json`."""
    tables: list[str] = entry["table_names_original"]
    columns: list[list] = entry["column_names_original"]
    types: list[str] = entry["column_types"]
    primary: list[int] = entry.get("primary_keys", []) or []
    foreign: list[list[int]] = entry.get("foreign_keys", []) or []

    by_table: dict[int, list[tuple[int, str, str]]] = {i: [] for i in range(len(tables))}
    for col_idx, (table_idx, col_name) in enumerate(columns):
        if table_idx < 0:  # a coluna sintética "*"
            continue
        by_table[table_idx].append((col_idx, col_name, types[col_idx]))

    pk_by_table: dict[int, list[str]] = {}
    for col_idx in primary:
        if col_idx < 0 or col_idx >= len(columns):
            continue
        table_idx, col_name = columns[col_idx]
        pk_by_table.setdefault(table_idx, []).append(col_name)

    fk_by_table: dict[int, list[str]] = {}
    for src, dst in foreign:
        if src >= len(columns) or dst >= len(columns):
            continue
        src_table, src_col = columns[src]
        dst_table, dst_col = columns[dst]
        clause = (
            f"FOREIGN KEY ({_quote(src_col)}) REFERENCES "
            f"{_quote(tables[dst_table])}({_quote(dst_col)})"
        )
        fk_by_table.setdefault(src_table, []).append(clause)

    blocks: list[str] = []
    for table_idx, table_name in enumerate(tables):
        lines = [
            f"  {_quote(name)} {ctype.upper()}"
            for _, name, ctype in by_table.get(table_idx, [])
        ]
        if table_idx in pk_by_table:
            cols = ", ".join(_quote(c) for c in pk_by_table[table_idx])
            lines.append(f"  PRIMARY KEY ({cols})")
        lines.extend(f"  {clause}" for clause in fk_by_table.get(table_idx, []))
        body = ",\n".join(lines)
        blocks.append(f"CREATE TABLE {_quote(table_name)} (\n{body}\n);")

    return "\n\n".join(blocks)


def load_schemas(spider_root: str | Path) -> dict[str, DbSchema]:
    root = Path(spider_root)
    entries = json.loads((root / "tables.json").read_text(encoding="utf-8"))
    out: dict[str, DbSchema] = {}
    for entry in entries:
        db_id = entry["db_id"]
        out[db_id] = DbSchema(
            db_id=db_id,
            ddl=build_ddl(entry),
            sqlite_path=str(root / "database" / db_id / f"{db_id}.sqlite"),
            n_tables=len(entry["table_names_original"]),
            n_columns=len(entry["column_names_original"]) - 1,
        )
    return out


def hardness(sig: dict[str, Any] | None) -> str:
    """Faixa de dificuldade, aproximando a regra oficial do Spider.

    A regra original conta componentes da consulta analisada; aqui a contagem é
    derivada da assinatura estrutural. É uma aproximação, e o texto deve
    declará-la como tal ao estratificar os resultados.
    """
    if not sig:
        return "unknown"
    comp = (
        sig.get("n_joins", 0)
        + int(bool(sig.get("has_group_by")))
        + 2 * int(bool(sig.get("has_having")))
        + int(bool(sig.get("has_order_by")))
        + int(bool(sig.get("has_limit")))
        + 2 * sig.get("n_subqueries", 0)
        + 2 * len(sig.get("set_ops", []))
        + max(0, sig.get("n_predicates", 0) - 1)
        + max(0, len(sig.get("agg_funcs", [])) - 1)
    )
    if comp <= 1:
        return "easy"
    if comp <= 3:
        return "medium"
    if comp <= 6:
        return "hard"
    return "extra"


def load_examples(
    spider_root: str | Path,
    questions_pt_path: str | Path,
    splits: tuple[str, ...] = ("train_spider.json", "dev.json"),
) -> list[Example]:
    """Junta o Spider original com a tradução em português.

    `questions_pt_path` deve conter um JSON mapeando o identificador do exemplo
    para a pergunta traduzida. São aceitos dois formatos:

    * ``{"spider_train_0": "Quantos ...", ...}``
    * ``[{"id": "spider_train_0", "question_pt": "Quantos ..."}, ...]``

    O identificador é ``spider_<arquivo sem extensão>_<índice>``. Exemplos sem
    tradução são descartados e contados — a avaliação é pareada, e um item sem
    par não pode entrar.
    """
    from ..structure.signature import signature

    root = Path(spider_root)
    raw_pt = json.loads(Path(questions_pt_path).read_text(encoding="utf-8"))
    if isinstance(raw_pt, list):
        pt_by_id = {
            item["id"]: item.get("question_pt") or item.get("question")
            for item in raw_pt
        }
    else:
        pt_by_id = dict(raw_pt)

    examples: list[Example] = []
    missing = 0
    for split in splits:
        path = root / split
        if not path.exists():
            continue
        stem = path.stem
        for idx, row in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            example_id = f"spider_{stem}_{idx}"
            question_pt = pt_by_id.get(example_id)
            if not question_pt:
                missing += 1
                continue
            sql = row["query"].strip()
            db_id = row["db_id"]
            examples.append(
                Example(
                    example_id=example_id,
                    db_id=db_id,
                    question_en=row["question"].strip(),
                    question_pt=question_pt.strip(),
                    sql=sql,
                    difficulty=hardness(signature(sql)),
                    family_id=family_id(db_id, sql),
                    source=stem,
                )
            )
    if missing:
        print(f"[spider] {missing} exemplos sem tradução em português foram descartados")
    return examples


def question_for(example: Example, lang: str) -> str:
    return example.question_pt if lang == "pt" else example.question_en


def delex_for(example: Example, lang: str) -> str:
    return delexicalize(question_for(example, lang))
