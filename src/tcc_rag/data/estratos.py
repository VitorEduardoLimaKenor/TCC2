"""Estrato de literal traduzido (LT/SL), pré-registrado em 28/10/2026.

Fonte: `orquestracao/preregistro-literais-traduzidos-2026-10-28.md`. Um item é
**LT** quando o SQL de referência tem ao menos um literal de texto que:

1. está entre aspas simples ou duplas;
2. não é numérico: depois de tirar espaços e `%` das bordas, não casa com
   `[\\d.\\-]+`;
3. tem 2 caracteres ou mais;
4. aparece na pergunta em inglês como palavra ou sequência de palavras
   inteira (fronteiras de palavra, sem distinguir maiúsculas);
5. não aparece da mesma forma na pergunta em PT-BR.

Os demais são **SL**. Duas leituras do texto, registradas aqui e no relatório
da integração:

- os critérios 3–5 usam o literal já sem espaços e `%` das bordas (o mesmo
  recorte do critério 2): `LIKE '%Mar%'` é avaliado como `Mar`;
- "fronteira de palavra" é a ausência de caractere de palavra (`\\w`, Unicode)
  imediatamente antes e depois do literal, o que vale também para literais que
  começam ou terminam em pontuação.

A classificação depende só do texto das perguntas e do SQL. Não usa predições
nem resultados e não muda a partição.
"""

from __future__ import annotations

import re

_LITERAL = re.compile(r"'((?:[^']|'')*)'|\"([^\"]*)\"")
_NUMERICO = re.compile(r"[\d.\-]+")


def literais_de_texto(sql: str) -> list[str]:
    """Literais entre aspas, sem espaços e `%` das bordas, não numéricos, com 2+ caracteres."""
    saida = []
    for m in _LITERAL.finditer(sql):
        bruto = m.group(1) if m.group(1) is not None else m.group(2)
        valor = bruto.strip().strip("%").strip()
        if len(valor) < 2 or _NUMERICO.fullmatch(valor):
            continue
        saida.append(valor)
    return saida


def aparece(literal: str, texto: str) -> bool:
    padrao = r"(?<!\w)" + re.escape(literal) + r"(?!\w)"
    return re.search(padrao, texto, flags=re.IGNORECASE) is not None


def literais_traduzidos(sql: str, pergunta_en: str, pergunta_pt: str) -> list[str]:
    """Os literais que justificam o estrato LT (vazio: SL)."""
    return [lit for lit in literais_de_texto(sql)
            if aparece(lit, pergunta_en) and not aparece(lit, pergunta_pt)]


def estrato(sql: str, pergunta_en: str, pergunta_pt: str) -> str:
    return "LT" if literais_traduzidos(sql, pergunta_en, pergunta_pt) else "SL"
