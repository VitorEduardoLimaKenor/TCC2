"""Quadro das doze características da assinatura estrutural (Seção 4.2.3), com os pesos do código.

Uso: python scripts/quadro_assinatura.py <saida.tex>

Os pesos são lidos de `tcc_rag.structure.signature._WEIGHTS`, a mesma tabela usada na reordenação; o script
recusa gerar o quadro se o código tiver uma característica que ele não sabe descrever.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tcc_rag.structure.signature import _WEIGHTS  # noqa: E402

# (característica, tipo, comparação)
DESCRICAO = {
    "n_joins": ("Número de junções", "contagem"),
    "n_tables": ("Número de tabelas distintas", "contagem"),
    "n_subqueries": ("Número de subconsultas", "contagem"),
    "n_predicates": ("Número de predicados de filtragem", "contagem"),
    "n_select_items": ("Número de itens projetados", "contagem"),
    "agg_funcs": ("Funções de agregação empregadas (\\texttt{COUNT}, \\texttt{SUM}, \\texttt{AVG}...)", "conjunto"),
    "set_ops": ("Operações de conjunto (\\texttt{UNION}, \\texttt{INTERSECT}, \\texttt{EXCEPT})", "conjunto"),
    "has_group_by": ("Agrupamento (\\texttt{GROUP BY})", "presença"),
    "has_having": ("Filtragem após o agrupamento (\\texttt{HAVING})", "presença"),
    "has_order_by": ("Ordenação (\\texttt{ORDER BY})", "presença"),
    "has_limit": ("Limite de linhas (\\texttt{LIMIT})", "presença"),
    "has_distinct": ("Cláusula \\texttt{DISTINCT}", "presença"),
}
COMPARACAO = {
    "contagem": "$1 - |a - b| / \\max(a, b)$",
    "conjunto": "índice de Jaccard",
    "presença": "1 se iguais; 0 se diferentes",
}


def num(v: float) -> str:
    return (f"{v:.2f}".rstrip("0").rstrip(".")).replace(".", ",")


def main(saida: Path) -> None:
    faltam = set(_WEIGHTS) - set(DESCRICAO)
    if faltam:
        raise SystemExit(f"características sem descrição: {sorted(faltam)}")
    linhas = []
    for chave, (nome, tipo) in DESCRICAO.items():
        linhas.append(f"{nome} & {tipo} & {COMPARACAO[tipo]} & {num(_WEIGHTS[chave])} \\\\\n\\hline")
    tex = (
        "\\begin{tabframed}[!htbp]\n\\centering\n"
        "\\caption{Características da assinatura estrutural e pesos na similaridade estrutural.}\n\\footnotesize\n"
        "\\begin{tabularx}{\\linewidth}{|>{\\raggedright\\arraybackslash}X|l|l|c|}\n\\hline\n"
        "\\textbf{Característica} & \\textbf{Tipo} & \\textbf{Comparação entre duas consultas} & \\textbf{Peso} \\\\\n"
        "\\hline\n" + "\n".join(linhas) + "\n\\end{tabularx}\n"
        "\\par\\vspace{0.3\\baselineskip}\n\\parbox{\\linewidth}{\\footnotesize\\raggedright "
        "$a$ e $b$: valores da característica nas duas consultas (a comparação vale 1 quando ambos são zero). "
        "A similaridade estrutural é a média das comparações, ponderada pelos pesos.}\n"
        "\\vspace{0.1\\baselineskip}\n\\newline \\footnotesize \\textbf{Fonte: Autoria própria (2026).}\n"
        "\\label{tab:assinatura}\n\\end{tabframed}\n"
    )
    saida.write_text(tex, encoding="utf-8")
    print(saida)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
