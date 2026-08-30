"""Normalização de consultas SQL e agrupamento em famílias canônicas.

A família é a unidade de particionamento (Seção 4.5.1): duas consultas com o
mesmo padrão e literais diferentes pertencem à mesma família e vão para a mesma
partição. Isso impede que uma pergunta de teste tenha, no pool, um gêmeo que
difere apenas no valor de um filtro.

A família é *fina* de propósito. A assinatura estrutural, usada na reordenação,
é deliberadamente mais grossa — ver `tcc_rag.structure.signature`.
"""

from __future__ import annotations

import hashlib
import re

import sqlglot
from sqlglot import exp

DIALECT = "sqlite"
_PLACEHOLDER = "?"
_WS = re.compile(r"\s+")


def _alias_map(tree: exp.Expression) -> dict[str, str]:
    """alias -> nome real da tabela."""
    mapping: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        name = table.name
        alias = table.alias
        if alias and name:
            mapping[alias.lower()] = name.lower()
    return mapping


def canonicalize(sql: str) -> str:
    """Devolve a forma canônica de uma consulta.

    Elimina três fontes de variação irrelevante: apelidos de tabela, literais e
    formatação. Preserva a qualificação das colunas pelo nome real da tabela,
    de modo que `t1.name` e `t2.name` não colapsem no mesmo texto.
    """
    try:
        tree = sqlglot.parse_one(sql, read=DIALECT)
    except Exception:
        # Consulta que o parser não aceita ainda precisa de uma chave estável.
        return _WS.sub(" ", sql.strip().lower())

    aliases = _alias_map(tree)

    for column in tree.find_all(exp.Column):
        qualifier = column.table
        if qualifier:
            real = aliases.get(qualifier.lower(), qualifier.lower())
            column.set("table", exp.to_identifier(real))

    for table in tree.find_all(exp.Table):
        table.set("alias", None)

    for node in tree.find_all(exp.Alias):
        # Apelidos de coluna (SELECT x AS y) não distinguem padrões.
        node.set("alias", exp.to_identifier("c"))

    for literal in tree.find_all(exp.Literal):
        literal.replace(exp.Literal.string(_PLACEHOLDER))

    for ident in tree.find_all(exp.Identifier):
        ident.set("this", str(ident.this).lower())
        ident.set("quoted", False)

    try:
        out = tree.sql(dialect=DIALECT, normalize=True, pretty=False)
    except Exception:
        out = tree.sql(pretty=False)
    return _WS.sub(" ", out.strip().lower())


def family_id(db_id: str, sql: str) -> str:
    """Identificador da família canônica, escopado ao banco."""
    canon = canonicalize(sql)
    blob = f"{db_id.lower()}||{canon}".encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def referenced_tables_columns(sql: str) -> tuple[list[str], list[str]]:
    """Tabelas e colunas citadas pela consulta, para os metadados do Qdrant."""
    try:
        tree = sqlglot.parse_one(sql, read=DIALECT)
    except Exception:
        return [], []

    aliases = _alias_map(tree)
    tables = sorted({t.name.lower() for t in tree.find_all(exp.Table) if t.name})

    columns: set[str] = set()
    for column in tree.find_all(exp.Column):
        name = (column.name or "").lower()
        if not name or name == "*":
            continue
        qualifier = (column.table or "").lower()
        real = aliases.get(qualifier, qualifier)
        columns.add(f"{real}.{name}" if real else name)

    return tables, sorted(columns)


def is_read_only(sql: str) -> bool:
    """True apenas para SELECT e WITH, e apenas para uma instrução."""
    stripped = sql.strip().rstrip(";").strip()
    if ";" in stripped:
        return False
    try:
        statements = sqlglot.parse(stripped, read=DIALECT)
    except Exception:
        return False
    if len(statements) != 1 or statements[0] is None:
        return False
    root = statements[0]
    return isinstance(root, (exp.Select, exp.Union, exp.Except, exp.Intersect, exp.With))


def extract_sql(text: str) -> str:
    """Extrai a consulta da resposta do modelo.

    Modelos instruídos costumam devolver a consulta dentro de uma cerca de
    código, às vezes precedida de explicação. Esta função é aplicada de forma
    idêntica a todas as condições — se falhar em uma, falha em todas, e isso
    aparece na contagem de consultas não executáveis.
    """
    fence = re.search(r"```(?:sql)?\s*(.+?)```", text, re.S | re.I)
    body = fence.group(1) if fence else text

    match = re.search(r"\b(WITH|SELECT)\b", body, re.I)
    if match:
        body = body[match.start():]

    body = body.strip()
    if ";" in body:
        body = body.split(";", 1)[0]
    return _WS.sub(" ", body).strip()
