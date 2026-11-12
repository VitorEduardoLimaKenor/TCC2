"""Quadro do exemplo do mecanismo (F03) em LaTeX, a partir do JSON de `exemplo_mecanismo.py`.

Uso: python scripts/tabela_exemplo.py <exemplo.json> <saida.tex> [n_linhas]

Mostra a pergunta, sua forma delexicalizada, a consulta preliminar de C4 e os
candidatos da primeira etapa com as duas similaridades e a posição final. As
quatro primeiras posições da primeira etapa são a seleção de C3; as quatro
primeiras posições finais, a de C4.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ESC = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
        "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}


def esc(texto: str) -> str:
    return "".join(_ESC.get(ch, ch) for ch in " ".join(texto.split()))


def tt(texto: str) -> str:
    return f"{{\\ttfamily {esc(texto)}}}"


def num(v: float, casas: int = 3) -> str:
    return f"{v:.{casas}f}".replace(".", ",")


def assinatura(a: dict) -> str:
    partes = [f"{a['n_joins']} junção" if a["n_joins"] == 1 else f"{a['n_joins']} junções",
              f"{a['n_tables']} tabela" if a["n_tables"] == 1 else f"{a['n_tables']} tabelas"]
    if a["agg_funcs"]:
        partes.append("agregação (" + ", ".join(a["agg_funcs"]) + ")")
    for campo, rotulo in (("has_group_by", "agrupamento"), ("has_having", "filtragem após o agrupamento"),
                          ("has_order_by", "ordenação"), ("has_limit", "limite"), ("has_distinct", "DISTINCT")):
        if a[campo]:
            partes.append(rotulo)
    if a["n_subqueries"]:
        partes.append(f"{a['n_subqueries']} subconsulta(s)")
    if a["set_ops"]:
        partes.append("operação de conjunto (" + ", ".join(a["set_ops"]) + ")")
    partes.append(f"{a['n_predicates']} predicado(s) de filtragem")
    return "; ".join(partes)


def main(entrada: str, saida: str, n_linhas: str = "8") -> int:
    e = json.loads(Path(entrada).read_text(encoding="utf-8"))
    cfg = e["configuracao"]
    cands = e["c4"]["candidatos"]
    # as selecionadas por C3 (posições 1..k da primeira etapa) e por C4, mais as primeiras da etapa 1
    k = cfg["k"]
    mostrar = {c["uid"] for c in cands if c["posicao_primeira_etapa"] <= max(k, int(n_linhas)) or c["selecionada"]}
    linhas = []
    for c in sorted((c for c in cands if c["uid"] in mostrar), key=lambda c: c["posicao_primeira_etapa"]):
        c3 = "C3" if c["posicao_primeira_etapa"] <= k else ""
        c4 = "C4" if c["selecionada"] else ""
        marcas = ", ".join(m for m in (c3, c4) if m) or "---"
        linhas.append(
            f"{c['posicao_primeira_etapa']} & {esc(c['pergunta'])} & {num(c['sim_pergunta'])} & "
            f"{num(c['sim_estrutural'])} & {num(c['escore'])} & {c['posicao_final']} & {marcas} \\\\"
        )
    corpo = "\n\\hline\n".join(linhas)
    lam = num(cfg["lambda_question"], 2)
    tex = (
        "\\begin{table}[H]\n\\centering\n"
        f"\\caption{{Exemplo do mecanismo com uma pergunta do conjunto de validação (banco \\texttt{{{esc(e['db_id'])}}}): "
        f"candidatos da primeira etapa, similaridades e seleção em C3 e C4, com $k = {k}$, $N = {cfg['n_candidates']}$ "
        f"e $\\lambda = {lam}$.}}\n\\footnotesize\n"
        "\\begin{tabular}{p{2.9cm}|p{12.3cm}}\n\\hline\n"
        f"\\textbf{{Pergunta}} & {esc(e['pergunta'])} \\\\\n\\hline\n"
        f"\\textbf{{Delexicalizada}} & {esc(e['pergunta_delexicalizada'])} \\\\\n\\hline\n"
        f"\\textbf{{Consulta preliminar (C4)}} & {tt(e['c4']['consulta_preliminar'])} \\\\\n\\hline\n"
        f"\\textbf{{Assinatura da consulta preliminar}} & {esc(assinatura(e['c4']['assinatura_preliminar']))} \\\\\n\\hline\n"
        "\\end{tabular}\n\n\\vspace{0.6\\baselineskip}\n"
        "\\begin{tabular}{c|p{7.4cm}|c|c|c|c|c}\n\\hline\n"
        "\\shortstack{\\rule{0pt}{2.5ex}\\textbf{Pos.}\\\\{}\\textbf{1ª et.}} & \\textbf{Pergunta da demonstração candidata} & "
        "\\shortstack{\\rule{0pt}{2.5ex}\\textbf{Sim.}\\\\{}\\textbf{perg.}} & "
        "\\shortstack{\\rule{0pt}{2.5ex}\\textbf{Sim.}\\\\{}\\textbf{estr.}} & \\textbf{Escore} & "
        "\\shortstack{\\rule{0pt}{2.5ex}\\textbf{Pos.}\\\\{}\\textbf{final}} & \\textbf{Sel.} \\\\\n\\hline\n"
        f"{corpo}\n\\hline\n\\end{{tabular}}\n"
        "\\par\\vspace{0.3\\baselineskip}\n\\parbox{0.94\\linewidth}{\\footnotesize\\raggedright "
        "Pos. 1ª et.: posição pela similaridade da pergunta delexicalizada (as $k$ primeiras são a seleção de C3). "
        "Sim. estr.: similaridade entre a assinatura da consulta preliminar e a da consulta da demonstração. "
        "Sel.: condições que incluíram a demonstração no \\textit{prompt}. A consulta de referência não participa "
        "da seleção.}\n"
        "\\vspace{0.1\\baselineskip}\n\\newline \\footnotesize \\textbf{Fonte: Autoria própria (2026).}\n"
        "\\label{tab:exemplo_mecanismo}\n\\end{table}\n"
    )
    Path(saida).write_text(tex, encoding="utf-8")
    print(saida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
