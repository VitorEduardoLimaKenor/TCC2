"""Montagem do prompt (Seção 4.4).

Quatro elementos, sempre na mesma ordem: instrução, esquema completo em DDL,
demonstrações e pergunta. A definição do esquema é **idêntica em todas as
condições**, inclusive nas que não recebem demonstrações — variar esquema e
demonstrações ao mesmo tempo impediria atribuir as diferenças observadas à
seleção de exemplos.

O `sha256` do prompt vai para o log. É a prova auditável de que, entre duas
condições, só o bloco de demonstrações mudou.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..schemas import Demo

INSTRUCTION_PT = (
    "Você é um tradutor de linguagem natural para SQL. Dado o esquema de um "
    "banco de dados SQLite e uma pergunta, escreva uma única consulta SQL que "
    "a responda.\n"
    "Responda apenas com a consulta, sem explicação, dentro de um bloco ```sql."
)

INSTRUCTION_EN = (
    "You are a natural language to SQL translator. Given the schema of a "
    "SQLite database and a question, write a single SQL query that answers it.\n"
    "Answer with the query only, no explanation, inside a ```sql block."
)

_LABELS = {
    "pt": ("-- Esquema do banco de dados", "-- Exemplos", "Pergunta:", "Consulta:"),
    "en": ("-- Database schema", "-- Examples", "Question:", "Query:"),
}


@dataclass
class BuiltPrompt:
    text: str
    sha256: str
    n_demos_used: int
    n_demos_dropped: int
    estimated_tokens: int


def _estimate_tokens(text: str) -> int:
    """Estimativa grosseira usada só para decidir o corte de demonstrações.

    A contagem exata vem do tokenizador do backend e é a que vai para o log de
    custo; aqui basta uma aproximação conservadora.
    """
    return int(len(text) / 3.2) + 1


def _render_demo(demo: Demo, lang: str) -> str:
    question_label = _LABELS[lang][2]
    query_label = _LABELS[lang][3]
    return f"{question_label} {demo.question_raw}\n{query_label} {demo.sql}"


def build_prompt(
    question: str,
    ddl: str,
    demos: list[Demo],
    lang: str,
    max_context_tokens: int,
    reserve_for_completion: int = 384,
) -> BuiltPrompt:
    """Monta o prompt e, se necessário, corta demonstrações.

    O corte remove a partir da **última** demonstração da lista, que é a de
    menor pontuação — a ordem chega aqui já decrescente. A ocorrência é
    registrada em `n_demos_dropped`.
    """
    lang = lang if lang in _LABELS else "en"
    schema_label, examples_label, question_label, query_label = _LABELS[lang]
    instruction = INSTRUCTION_PT if lang == "pt" else INSTRUCTION_EN
    budget = max_context_tokens - reserve_for_completion

    used = list(demos)
    dropped = 0

    while True:
        parts = [instruction, "", schema_label, ddl]
        if used:
            parts += ["", examples_label]
            parts += [_render_demo(d, lang) for d in used]
        parts += ["", f"{question_label} {question}", query_label]
        text = "\n".join(parts)

        tokens = _estimate_tokens(text)
        if tokens <= budget or not used:
            return BuiltPrompt(
                text=text,
                sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                n_demos_used=len(used),
                n_demos_dropped=dropped,
                estimated_tokens=tokens,
            )
        used.pop()
        dropped += 1


def build_draft_prompt(question: str, ddl: str, lang: str, max_context_tokens: int) -> BuiltPrompt:
    """Prompt da consulta preliminar de C4: idêntico ao de C0.

    Reusar o mesmo montador garante que o rascunho seja produzido nas mesmas
    condições da linha de base sem demonstrações, e não num formato paralelo
    que introduziria uma diferença não controlada.
    """
    return build_prompt(question, ddl, [], lang, max_context_tokens)
