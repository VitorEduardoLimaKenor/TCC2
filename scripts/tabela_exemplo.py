"""Tabela do exemplo do mecanismo (F03) em LaTeX, a partir do JSON de `exemplo_mecanismo.py`.

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
        partes.append(f"{a['n_subqueries']} subconsulta" + ("" if a["n_subqueries"] == 1 else "s"))
    if a["set_ops"]:
        partes.append("operação de conjunto (" + ", ".join(a["set_ops"]) + ")")
    partes.append(f"{a['n_predicates']} predicado de filtragem" if a["n_predicates"] == 1
                  else f"{a['n_predicates']} predicados de filtragem")
    partes.append(f"{a['n_select_items']} item projetado" if a["n_select_items"] == 1
                  else f"{a['n_select_items']} itens projetados")
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
        # em cinza, as demonstrações que entram em só uma das duas seleções: é ali que C3 e C4 diferem
        cor = "\\rowcolor{black!8}" if bool(c3) != bool(c4) else ""
        linhas.append(
            f"{cor}{c['posicao_primeira_etapa']} & {esc(c['pergunta'])} & {num(c['sim_pergunta'])} & "
            f"{num(c['sim_estrutural'])} & {num(c['escore'])} & {c['posicao_final']} & {marcas} \\\\"
        )
    corpo = "\n".join(linhas)
    lam = num(cfg["lambda_question"], 2)

    def marcadores(texto: str) -> str:
        t = esc(texto)
        for m in ("<VAL>", "<NUM>", "<DATE>"):
            t = t.replace(m, f"\\texttt{{{m}}}")
        return t

    tex = (
        "\\begin{table}[!htb]\n\\centering\n"
        f"\\caption{{Exemplo das duas etapas de recuperação (banco \\texttt{{{esc(e['db_id'])}}}).}}\n\\footnotesize\n"
        "\\begin{tabularx}{\\linewidth}{@{}>{\\raggedright\\arraybackslash}p{3.3cm}X@{}}\n\\toprule\n"
        f"\\textbf{{Pergunta}} & {esc(e['pergunta'])} \\\\\n"
        f"\\textbf{{Delexicalizada}} & {marcadores(e['pergunta_delexicalizada'])} \\\\\n"
        f"\\textbf{{Consulta preliminar (C4)}} & {tt(e['c4']['consulta_preliminar'])} \\\\\n"
        f"\\textbf{{Assinatura (preliminar)}} & {esc(assinatura(e['c4']['assinatura_preliminar']))} \\\\\n"
        "\\bottomrule\n\\end{tabularx}\n\n\\vspace{0.8\\baselineskip}\n"
        "\\begin{tabularx}{\\linewidth}{@{}c>{\\raggedright\\arraybackslash}Xccccc@{}}\n\\toprule\n"
        " & & \\multicolumn{3}{c}{\\textbf{Pontuação}} & & \\\\\n\\cmidrule(lr){3-5}\n"
        "\\textbf{Pos. inicial} & \\textbf{Pergunta da demonstração candidata} & "
        "$\\boldsymbol{\\mathrm{sim}_{\\mathrm{perg}}}$ & $\\boldsymbol{\\mathrm{sim}_{\\mathrm{estr}}}$ & "
        "\\textbf{Escore} & \\textbf{Pos. final} & \\textbf{Seleção} \\\\\n\\midrule\n"
        f"{corpo}\n\\bottomrule\n\\end{{tabularx}}\n"
        "\\par\\vspace{0.3\\baselineskip}\n\\parbox{\\linewidth}{\\footnotesize\\raggedright "
        f"Pergunta do conjunto de validação, com $k = {k}$, $N = {cfg['n_candidates']}$ e $\\lambda = {lam}$. "
        "Assinatura: só as características presentes (as demais são zero ou ausentes). "
        "Pos. inicial: posição pela similaridade da pergunta delexicalizada ($\\mathrm{sim}_{\\mathrm{perg}}$); "
        "as $k$ primeiras formam a seleção de C3. $\\mathrm{sim}_{\\mathrm{estr}}$: similaridade entre a assinatura "
        "da consulta preliminar e a da consulta da demonstração. Escore e posição final: Equação \\ref{eq:escore}; as "
        "$k$ primeiras formam a seleção de C4. Seleção: condições que incluíram a demonstração no \\textit{prompt}; em "
        "cinza, as que entram em só uma das duas. A consulta de referência não participa da seleção. As perguntas são "
        "reproduzidas literalmente do conjunto de dados, inclusive em suas repetições.}\n"
        "\\vspace{0.1\\baselineskip}\n\\newline \\footnotesize \\textbf{Fonte: Autoria própria (2026).}\n"
        "\\label{tab:exemplo_mecanismo}\n\\end{table}\n"
    )
    Path(saida).write_text(tex, encoding="utf-8")
    print(saida)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
