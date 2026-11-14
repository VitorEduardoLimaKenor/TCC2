"""Tabelas LaTeX do Capítulo 5, geradas do JSON de `tcc-rag analise`.

Uso:
    python scripts/tabelas_cap5.py <analise-val-*.json> <analise-test-*.json|-> <dir_saida>

Cada tabela vira um arquivo `.tex` completo (ambiente `table`, legenda, rótulo e
fonte), incluído no capítulo com `\\input`. Nenhum número é digitado à mão: o
que aparece na monografia é o que está no arquivo de análise. O segundo
argumento pode ser `-` enquanto o teste não tiver sido executado.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

CONDS = ("C0", "C1", "C2", "C3", "C4")
NOME = {
    "C0": "C0 -- Sem demonstrações",
    "C1": "C1 -- Exemplos fixos",
    "C2": "C2 -- Similaridade da pergunta",
    "C3": "C3 -- Pergunta delexicalizada",
    "C4": "C4 -- Duas etapas",
}
FONTE = "\\vspace{0.1\\baselineskip}\n\\newline \\footnotesize \\textbf{Fonte: Autoria própria (2026).}"


def num(v: float | None, casas: int = 1) -> str:
    if v is None:
        return "---"
    return f"{v:.{casas}f}".replace(".", ",")


def pct(v: float | None) -> str:
    return "---" if v is None else num(100 * v)


def sinal(v: float | None, casas: int = 1) -> str:
    if v is None:
        return "---"
    return ("$+$" if v >= 0 else "$-$") + num(abs(v), casas)


def ic(par) -> str:
    return f"[{sinal(par[0])}; {sinal(par[1])}]"


def pval(p: float | None) -> str:
    if p is None:
        return "---"
    return "$<$\\,0,001" if p < 0.001 else num(p, 3)


def duas(a: str, b: str) -> str:
    """Célula ou cabeçalho em duas linhas."""
    return f"\\shortstack{{\\rule{{0pt}}{{2.5ex}}{a}\\\\{{}}{b}}}"  # {} impede que "[" vire argumento de \\\\


def cab(a: str, b: str) -> str:
    return duas(f"\\textbf{{{a}}}", f"\\textbf{{{b}}}")


def inteiro(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def tabela(colunas: str, cabecalho: str, linhas: list[str], legenda: str, rotulo: str, nota: str = "") -> str:
    corpo = "\n\\hline\n".join(linhas)
    return (
        "\\begin{table}[H]\n\\centering\n"
        f"\\caption{{{legenda}}}\n\\footnotesize\n"
        f"\\begin{{tabular}}{{{colunas}}}\n\\hline\n{cabecalho} \\\\\n\\hline\n{corpo}\n\\hline\n\\end{{tabular}}\n"
        + (f"\\par\\vspace{{0.3\\baselineskip}}\n\\parbox{{0.94\\linewidth}}{{\\footnotesize\\raggedright {nota}}}\n"
           if nota else "")
        + f"{FONTE}\n\\label{{{rotulo}}}\n\\end{{table}}\n"
    )


# ------------------------------------------------------------------ validação


def val_condicoes(a: dict) -> str:
    linhas = []
    for c in CONDS:
        pt, en = a["celulas"].get(f"{c}|pt"), a["celulas"].get(f"{c}|en")
        if not pt:
            continue
        linhas.append(
            f"{NOME[c]} & {pct(pt['ex'])} & {pct(en['ex']) if en else '---'} & {pct(pt['acerto_estrutural'])} & "
            f"{num(pt['prompt_tokens_total_medio'], 0)} \\\\"
        )
    n = a["n_exemplos"]
    return tabela(
        "l|c|c|c|c",
        "\\textbf{Condição} & " + cab("EX PT", "(\\%)") + " & " + cab("EX EN", "(\\%)") + " & "
        + cab("Acerto estr.", "PT (\\%)") + " & " + cab("Tokens ent.", "PT"),
        linhas,
        f"Validação com a tradução definitiva: acurácia de execução por condição e idioma ($n = {n}$ perguntas, "
        "configuração candidata).",
        "tab:validacao_ptbr",
        "Tokens de entrada: média por pergunta; em C4, somadas as duas chamadas ao modelo.",
    )


def contrastes(a: dict, rotulo: str, legenda: str, com_holm: bool) -> str:
    papeis = (("C3-C2", "C3 $-$ C2 (primária)"), ("C4-C2", "C4 $-$ C2 (H1)"), ("C4-C3", "C4 $-$ C3 (componente)"),
              ("C2-C1", "C2 $-$ C1"), ("C1-C0", "C1 $-$ C0"))
    holm = a.get("familia_confirmatoria_holm") or {}
    por_familia = (a.get("sensibilidade_familias") or {}).get("contrastes") or {}
    linhas = []
    for chave, nome in papeis:
        c = a["contrastes"].get(f"{chave}|pt")
        if not c:
            continue
        m = c["mcnemar"]
        celulas = [nome, sinal(c["diferenca_pp"]), ic(c["ic95"])]
        if por_familia:
            f = por_familia.get(f"{chave}|pt")
            celulas.append(ic(f["ic95_por_familia"]) if f else "---")
        celulas += [f"{m['only_a']} / {m['only_b']}", pval(m["p_value"])]
        if com_holm:
            h = holm.get(f"{chave}|pt")
            celulas.append(pval(h["p_holm"]) if h else "---")
        linhas.append(" & ".join(celulas) + " \\\\")
    cabecalho = ("\\textbf{Contraste} & " + cab("Diferença", "(pp)") + " & \\textbf{IC de 95\\%} & "
                 + (cab("IC de 95\\%", "(por família)") + " & " if por_familia else "")
                 + cab("Discor-", "dantes") + " & \\textbf{Valor-p}" + (" & " + cab("Valor-p", "(Holm)") if com_holm else ""))
    return tabela(
        "l|c|c|c|c" + ("|c" if por_familia else "") + ("|c" if com_holm else ""), cabecalho, linhas, legenda, rotulo,
        "IC de 95\\%: reamostragem pareada estratificada por banco" + ("; por família: a unidade reamostrada é a "
        "família canônica (análise de sensibilidade)" if por_familia else "") + ". Discordantes: perguntas em que só a "
        "primeira condição acerta / só a segunda acerta. Valor-p: teste exato de McNemar, bilateral.",
    )


# ---------------------------------------------------------------------- teste


def teste_dificuldade(a: dict) -> str:
    linhas = []
    for c in CONDS:
        cel = a["celulas"].get(f"{c}|pt")
        if not cel:
            continue
        d = cel["por_dificuldade"]
        linhas.append(f"{NOME[c]} & " + " & ".join(pct(d[k]["ex"]) for k in ("easy", "medium", "hard", "extra"))
                      + f" & \\textbf{{{pct(cel['ex'])}}} & {cel['nao_executaveis']} \\\\")
    d0 = a["celulas"]["C0|pt"]["por_dificuldade"]
    ns = ", ".join(f"{rot} $n = {d0[k]['n']}$" for k, rot in
                   (("easy", "fácil"), ("medium", "médio"), ("hard", "difícil"), ("extra", "extra")))
    return tabela(
        "l|c|c|c|c|c|c",
        "\\textbf{Condição} & \\textbf{Fácil} & \\textbf{Médio} & \\textbf{Difícil} & \\textbf{Extra} & \\textbf{Total} & "
        "\\textbf{Não exec.}",
        linhas,
        "Acurácia de execução (\\%) em português brasileiro no conjunto de teste, por nível de dificuldade "
        f"(classificação oficial do \\textit{{Spider}}; $n = {a['n_exemplos']}$).",
        "tab:resultados_ptbr",
        f"Perguntas por nível: {ns}. Não exec.: número de predições que não executam ou excedem o limite de tempo.",
    )


def teste_mcnemar(a: dict) -> str:
    m = a["contrastes"]["C3-C2|pt"]["mcnemar"]
    linhas = [f"\\textbf{{C3 correta}} & {m['both_correct']} & {m['only_a']} \\\\",
              f"\\textbf{{C3 incorreta}} & {m['only_b']} & {m['both_wrong']} \\\\"]
    return tabela("l|c|c", " & \\textbf{C2 correta} & \\textbf{C2 incorreta}", linhas,
                  f"Comparação pareada entre as condições C3 e C2 em português brasileiro ($n = {m['n']}$).",
                  "tab:mcnemar")


def teste_recuperador(a: dict) -> str:
    linhas = []
    for c in ("C1", "C2", "C3", "C4"):
        pt, en = a["celulas"].get(f"{c}|pt"), a["celulas"].get(f"{c}|en")
        if not pt:
            continue
        recurso = pct(pt["recurso_ordenacao_inicial"]) if c == "C4" else "n.a."
        linhas.append(f"{NOME[c]} & {pct(pt['acerto_estrutural'])} & {pct(en['acerto_estrutural']) if en else '---'} & "
                      f"{num(pt['k_efetivo_medio'], 2)} & {recurso} \\\\")
    return tabela(
        "l|c|c|c|c",
        "\\textbf{Condição} & " + cab("Acerto estr.", "PT (\\%)") + " & " + cab("Acerto estr.", "EN (\\%)") + " & "
        + cab("$k$", "efetivo") + " & " + cab("Recurso à ord.", "inicial (\\%)"),
        linhas,
        "Acerto estrutural entre as demonstrações que entraram no \\textit{prompt}, número efetivo de demonstrações e "
        "recurso à ordenação inicial, no conjunto de teste.",
        "tab:recuperador",
        "n.a.: não se aplica, pois apenas C4 gera consulta preliminar.",
    )


def teste_idiomas(a: dict) -> str:
    linhas = []
    for c in CONDS:
        pt, en = a["celulas"].get(f"{c}|pt"), a["celulas"].get(f"{c}|en")
        d = (a.get("en_menos_pt") or {}).get(c)
        if not (pt and en and d):
            continue
        linhas.append(f"{NOME[c]} & {pct(en['ex'])} & {pct(pt['ex'])} & {sinal(d['diferenca_pp'])} & {ic(d['ic95'])} \\\\")
    return tabela(
        "l|c|c|c|c",
        "\\textbf{Condição} & \\textbf{Inglês (\\%)} & \\textbf{Português (\\%)} & \\textbf{EN $-$ PT (pp)} & "
        "\\textbf{IC de 95\\%}",
        linhas,
        f"Acurácia de execução por idioma no conjunto de teste e diferença entre os idiomas ($n = {a['n_exemplos']}$ "
        "perguntas pareadas).",
        "tab:idiomas",
    )


def teste_custo(a: dict) -> str:
    lat = (a.get("custo") or {}).get("latencia") or {}
    linhas = []
    for c in CONDS:
        cel = a["celulas"].get(f"{c}|pt")
        cu = a["custo"]["por_celula"].get(f"{c}|pt")
        if not cel:
            continue
        l = lat.get(f"{c}|pt")
        ganho = "---" if c == "C0" or cu["ganho_pp_por_100_tokens"] is None else sinal(cu["ganho_pp_por_100_tokens"], 2)
        linhas.append(
            f"{NOME[c]} & {pct(cel['ex'])} & {num(cel['prompt_tokens_total_medio'], 0)} & "
            f"{num(cel['completion_tokens_total_medio'], 0)} & "
            f"{num(l['total_ms_p50'] / 1000, 2) if l else '---'} & {num(l['total_ms_p95'] / 1000, 2) if l else '---'} & "
            f"{ganho} \\\\"
        )
    prot = (a.get("custo") or {}).get("latencia_protocolo") or {}
    nota = ("Tokens: média por pergunta no conjunto de teste em português; em C4, somadas as duas chamadas ao modelo. "
            + (f"Latência: passagem dedicada com uma pergunta por vez ({prot.get('n_medicoes')} medições; "
               f"{prot.get('gpu')}). " if prot else "")
            + "Ganho marginal: pontos percentuais de acurácia por cem tokens de entrada adicionais em relação a C0.")
    return tabela(
        "l|c|c|c|c|c|c",
        "\\textbf{Condição} & " + cab("EX", "(\\%)") + " & " + cab("Tokens", "entrada") + " & " + cab("Tokens", "saída")
        + " & " + cab("Lat. p50", "(s)") + " & " + cab("Lat. p95", "(s)") + " & " + cab("Ganho", "marg."),
        linhas,
        "Acurácia, custo computacional e ganho marginal por condição, em português brasileiro.",
        "tab:custo",
        nota,
    )


def teste_erros(a: dict) -> str:
    rub = a["erros"]["rubrica"]
    conds = [c for c in CONDS if f"{c}|pt" in a["erros"]["por_celula"]]
    ordem = ("identificador", "invalida", "tabelas", "agregacao", "filtro", "literal", "projecao", "ordem", "outras",
             "nao_analisada")
    linhas = []
    for k in ordem:
        valores = [a["erros"]["por_celula"][f"{c}|pt"][k] for c in conds]
        if k == "nao_analisada" and not any(valores):
            continue
        linhas.append(f"{rub[k]} & " + " & ".join(str(v) for v in valores) + " \\\\")
    linhas.append("\\textbf{Total de predições incorretas} & "
                  + " & ".join(f"\\textbf{{{a['erros']['por_celula'][f'{c}|pt']['total']}}}" for c in conds) + " \\\\")
    return tabela(
        "p{7.4cm}|" + "|".join("c" for _ in conds),
        "\\textbf{Categoria (primeira regra que se aplica)} & " + " & ".join(f"\\textbf{{{c}}}" for c in conds),
        linhas,
        f"Predições incorretas em português brasileiro no conjunto de teste, por categoria da rubrica automática "
        f"($n = {a['n_exemplos']}$ perguntas por condição).",
        "tab:erros",
    )


def teste_h3(a: dict) -> str:
    linhas = []
    for c in ("C1", "C2", "C3", "C4"):
        h = a["h3"].get(f"{c}|pt")
        if not h:
            continue
        t = h["tabela"]
        n_com, n_sem = t["hit_correta"] + t["hit_incorreta"], t["sem_hit_correta"] + t["sem_hit_incorreta"]
        dif = "---" if h["diferenca_pp"] is None else duas(sinal(h["diferenca_pp"]), ic(h["ic95"]))
        aj = ("---" if h["diferenca_ajustada_dificuldade_pp"] is None else
              duas(sinal(h["diferenca_ajustada_dificuldade_pp"]), ic(h["ic95_ajustada"])))
        linhas.append(f"{c} & {pct(h['ex_com_acerto_estrutural'])} ({n_com}) & {pct(h['ex_sem_acerto_estrutural'])} "
                      f"({n_sem}) & {dif} & {aj} \\\\")
    return tabela(
        "l|c|c|c|c",
        "\\textbf{Cond.} & " + cab("EX com acerto", "estrutural ($n$)") + " & " + cab("EX sem acerto", "estrutural ($n$)")
        + " & " + cab("Diferença (pp)", "e IC de 95\\%") + " & " + cab("Diferença ajustada", "pela dificuldade"),
        linhas,
        "Acurácia de execução (\\%) em português brasileiro conforme o acerto estrutural das demonstrações, no conjunto "
        "de teste.",
        "tab:h3",
    )


def teste_literal(a: dict) -> str:
    lt = a.get("literal_traduzido")
    if not lt:
        return ""
    linhas = []
    for est, nome in (("LT", "Com literal traduzido"), ("SL", "Sem literal traduzido")):  # noqa: B007
        b = lt["por_estrato"].get(est)
        if not b:
            continue
        ex = b["ex"]
        d0 = (b.get("en_menos_pt") or {}).get("C0")
        d4 = (b.get("en_menos_pt") or {}).get("C4")
        c32 = b.get("c3_menos_c2_pt")
        def cel(d):
            return duas(sinal(d["diferenca_pp"]), ic(d["ic95"])) if d else "---"

        linhas.append(f"{nome} & {b['n']} & {pct(ex.get('C0|pt'))} / {pct(ex.get('C0|en'))} & "
                      f"{cel(d0)} & {cel(d4)} & {cel(c32)} \\\\")
    return tabela(
        "l|c|c|c|c|c",
        "\\textbf{Estrato} & \\textbf{$n$} & " + cab("EX em C0,", "PT / EN (\\%)") + " & " + cab("EN $-$ PT", "em C0")
        + " & " + cab("EN $-$ PT", "em C4") + " & " + cab("C3 $-$ C2", "em PT"),
        linhas,
        "Análise estratificada pela presença de literal traduzido no conjunto de teste (exploratória). Diferenças em "
        "pontos percentuais, com IC de 95\\%.",
        "tab:literal",
    )


def teste_em(a: dict) -> str:
    linhas = []
    for c in CONDS:
        pt, en = a["celulas"].get(f"{c}|pt"), a["celulas"].get(f"{c}|en")
        if not pt:
            continue
        linhas.append(f"{NOME[c]} & {pct(pt['em_oficial'])} & {pt['em_predicao_nao_analisavel']} & "
                      f"{pct(en['em_oficial']) if en else '---'} & {en['em_predicao_nao_analisavel'] if en else '---'} \\\\")
    return tabela(
        "l|c|c|c|c",
        "\\textbf{Condição} & " + cab("EM PT", "(\\%)") + " & " + cab("Não analis.", "PT") + " & " + cab("EM EN", "(\\%)")
        + " & " + cab("Não analis.", "EN"),
        linhas,
        "Correspondência exata de conjuntos (EM oficial) no conjunto de teste e número de predições que o analisador "
        "do avaliador não interpreta, contadas como incorretas.",
        "tab:em",
    )


def main(val: Path, teste: Path | None, saida: Path) -> None:
    saida.mkdir(parents=True, exist_ok=True)
    arquivos: dict[str, str] = {}
    av = json.loads(val.read_text(encoding="utf-8"))
    arquivos["val_condicoes"] = val_condicoes(av)
    arquivos["val_contrastes"] = contrastes(
        av, "tab:validacao_contrastes",
        f"Validação com a tradução definitiva: contrastes pareados em português brasileiro ($n = {av['n_exemplos']}$).",
        com_holm=False)
    if teste is not None:
        at = json.loads(teste.read_text(encoding="utf-8"))
        arquivos["teste_dificuldade"] = teste_dificuldade(at)
        arquivos["teste_mcnemar"] = teste_mcnemar(at)
        arquivos["teste_contrastes"] = contrastes(
            at, "tab:contrastes",
            f"Contrastes pareados em português brasileiro no conjunto de teste ($n = {at['n_exemplos']}$).",
            com_holm=True)
        arquivos["teste_recuperador"] = teste_recuperador(at)
        arquivos["teste_idiomas"] = teste_idiomas(at)
        arquivos["teste_custo"] = teste_custo(at)
        arquivos["teste_erros"] = teste_erros(at)
        arquivos["teste_h3"] = teste_h3(at)
        arquivos["teste_literal"] = teste_literal(at)
        arquivos["teste_em"] = teste_em(at)
    for nome, texto in arquivos.items():
        if texto:
            (saida / f"{nome}.tex").write_text(texto, encoding="utf-8")
            print(saida / f"{nome}.tex")


if __name__ == "__main__":
    main(Path(sys.argv[1]), None if sys.argv[2] == "-" else Path(sys.argv[2]), Path(sys.argv[3]))
