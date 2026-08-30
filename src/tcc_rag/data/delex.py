"""Delexicalização de perguntas (Seção 4.2.2).

Substitui literais específicos do domínio por marcadores de tipo, preservando o
esqueleto interrogativo, para que duas perguntas com o mesmo padrão de consulta
fiquem próximas no espaço vetorial ainda que tratem de entidades distintas.

Restrição de projeto: a operação é **puramente superficial**. Ela não olha a
consulta SQL de referência, porque precisa ser idêntica para os registros do
pool e para as perguntas de teste — e no teste o gabarito não está disponível
no momento da inferência. `test_delex.py` verifica essa propriedade.
"""

from __future__ import annotations

import re

VAL = "<VAL>"
NUM = "<NUM>"
DATE = "<DATE>"

_QUOTED = re.compile(r"""(["'‘’“”])(.+?)\1""")
_DATE_SLASH = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
_DATE_ISO = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_YEAR = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")

_MONTHS = (
    "janeiro fevereiro março abril maio junho julho agosto setembro outubro "
    "novembro dezembro january february march april may june july august "
    "september october november december"
).split()
_MONTH_RE = re.compile(r"\b(" + "|".join(_MONTHS) + r")\b", re.I)


def delexicalize(question: str) -> str:
    """Aplica os marcadores na ordem em que as regras não se atropelam.

    Datas antes de números, senão `2024-01-15` vira três marcadores de número.
    """
    text = question.strip()
    text = _QUOTED.sub(VAL, text)
    text = _DATE_ISO.sub(DATE, text)
    text = _DATE_SLASH.sub(DATE, text)
    text = _MONTH_RE.sub(DATE, text)
    text = _YEAR.sub(DATE, text)
    text = _NUMBER.sub(NUM, text)
    return re.sub(r"\s+", " ", text).strip()


def delexicalize_batch(questions: list[str]) -> list[str]:
    return [delexicalize(q) for q in questions]
