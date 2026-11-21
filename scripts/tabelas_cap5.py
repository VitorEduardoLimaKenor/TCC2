"""Tabelas LaTeX do Capítulo 5, geradas do JSON de `tcc-rag analise`.

Uso:
    python scripts/tabelas_cap5.py <analise-val-*.json> <analise-test-*.json|-> <dir_saida>

Cada tabela vira um arquivo `.tex` completo (ambiente `table`, legenda, rótulo e
fonte), incluído no capítulo com `\\input`. Nenhum número é digitado à mão: o
que aparece na monografia é o que está no arquivo de análise. O segundo
argumento pode ser `-` enquanto o teste não tiver sido executado.

Formato (desde 20/11/2026): tabelas abertas nas laterais, sem traços verticais
e só com os traços horizontais do cabeçalho e do fim (pacote `booktabs`), como
recomendam as normas de apresentação tabular do IBGE seguidas pela ABNT. As
tabelas numéricas ocupam a largura do texto, para que a nota fique alinhada a
elas; cabeçalhos com subcolunas usam `\\cmidrule`; nenhuma célula tem duas
linhas empilhadas.
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
NOME_CURTO = {
    "C0": "C0 -- Sem demonstr.",
    "C1": "C1 -- Fixos",
    "C2": "C2 -- Similaridade",
    "C3": "C3 -- Delexicalizada",
    "C4": "C4 -- Duas etapas",
}
FONTE = "\\vspace{0.1\\baselineskip}\n\\newline \\footnotesize \\textbf{Fonte: Autoria própria (2026).}"
DESTAQUE = "\\cellcolor{black!10}"
ARRED = (" As diferenças são calculadas sobre as contagens de acertos e podem diferir, no último dígito, da subtração "
         "dos valores arredondados.")  # fundo cinza-claro, legível em impressão monocromática


def num(v: float | None, casas: int = 1) -> str:
    if v is None:
        return "---"
    return f"{v:.{casas}f}".replace(".", ",")


def pct(v: float | None) -> str:
    return "---" if v is None else num(100 * v)


def sinal(v: float | None, casas: int = 1) -> str:
    """Número com sinal explícito; o zero (depois do arredondamento) fica sem sinal."""
    if v is None:
        return "---"
    if round(abs(v), casas) == 0:
        return num(0.0, casas)
    return ("$+$" if v > 0 else "$-$") + num(abs(v), casas)


def ic(par, casas: int = 1) -> str:
    return f"[{sinal(par[0], casas)}; {sinal(par[1], casas)}]"


def pval(p: float | None) -> str:
    if p is None:
        return "---"
    return "$<$\\,0,001" if p < 0.001 else num(p, 3)


def b(texto: str) -> str:
    return f"\\textbf{{{texto}}}"


def grupo(titulo: str, n: int) -> str:
    """Título de um grupo de `n` colunas no cabeçalho."""
    return f"\\multicolumn{{{n}}}{{c}}{{{b(titulo)}}}"


def bloco(titulo: str, ncols: int) -> str:
    """Linha de subtítulo que separa grupos de linhas no corpo da tabela."""
    return f"\\multicolumn{{{ncols}}}{{l}}{{\\textit{{{titulo}}}}} \\\\"


def inteiro(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def tabela(colunas: str, cabecalho: str, linhas: list[str], legenda: str, rotulo: str, nota: str = "",
           ajuste: str = "", posicao: str = "H", largura: bool | str = True, fonte: str = "\\small") -> str:
    """Monta o ambiente `table`.

    `colunas`: especificação sem traços verticais (por exemplo, "lcccc").
    `cabecalho`: uma ou mais linhas de cabeçalho; a última não leva `\\\\` (ele é acrescentado aqui), as
    anteriores podem trazer `\\cmidrule`.
    `linhas`: linhas do corpo, cada uma terminada em `\\\\`; também aceita `\\midrule` e `\\addlinespace`.
    `largura`: True estende a tabela à largura do texto (`tabular*` com espaço elástico entre as colunas);
    "x" usa `tabularx`, com as colunas X da especificação flexíveis; False mantém a largura natural.
    `ajuste`: comandos locais ao ambiente (por exemplo, o espaço entre colunas de uma tabela larga).
    `posicao`: especificador do ambiente; "H" fixa a tabela no ponto do texto, e "!htbp" a deixa flutuar
    para evitar uma página com espaço em branco quando ela não cabe no fim da página.
    """
    if largura == "x":  # coluna(s) X de largura flexível (tabularx), para textos longos na primeira coluna
        abre = f"\\begin{{tabularx}}{{\\linewidth}}{{@{{}}{colunas}@{{}}}}"
        fecha = "\\end{tabularx}"
    elif largura:
        # O espaço elástico (\\fill) distribui a sobra entre as colunas; o espaço fixo entre elas fica pequeno
        # para que a largura natural nunca passe da do texto.
        ajuste = ajuste or "\\setlength{\\tabcolsep}{2pt}\n"
        abre = f"\\begin{{tabular*}}{{\\linewidth}}{{@{{\\extracolsep{{\\fill}}}}{colunas}@{{}}}}"
        fecha = "\\end{tabular*}"
    else:
        abre = f"\\begin{{tabular}}{{{colunas}}}"
        fecha = "\\end{tabular}"
    corpo = "\n".join(linhas)
    return (
        f"\\begin{{table}}[{posicao}]\n\\centering\n"
        f"\\caption{{{legenda}}}\n{fonte}\n{ajuste}"
        f"{abre}\n\\toprule\n{cabecalho} \\\\\n\\midrule\n{corpo}\n\\bottomrule\n{fecha}\n"
        + (f"\\par\\vspace{{0.3\\baselineskip}}\n\\parbox{{\\linewidth}}{{\\footnotesize\\raggedright {nota}}}\n"
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
            f"{inteiro(round(pt['prompt_tokens_total_medio']))} \\\\"
        )
    n = a["n_exemplos"]
    cabecalho = (" & " + grupo("Acurácia de execução (\\%)", 2) + " & & \\\\\n\\cmidrule(lr){2-3}\n"
                 + b("Condição") + " & " + b("Português") + " & " + b("Inglês") + " & "
                 + b("Acerto estr. (\\%)") + " & " + b("Tokens ent."))
    return tabela(
        "lcccc", cabecalho, linhas,
        f"Acurácia de execução por condição e idioma na validação ($n = {n}$).",
        "tab:validacao_ptbr",
        "Configuração candidata. Acerto estrutural e \\textit{tokens} de entrada: perguntas em português; "
        "\\textit{tokens}: média por pergunta, em C4 somadas as duas chamadas ao modelo.",
    )


def contrastes(a: dict, rotulo: str, legenda: str, com_holm: bool) -> str:
    """Contrastes pareados, em dois blocos: a família confirmatória (corrigida por Holm) e os exploratórios."""
    papeis = (("C3-C2", "C3 $-$ C2 (primária)"), ("C4-C2", "C4 $-$ C2 (H1)"), ("C4-C3", "C4 $-$ C3 (componente)"),
              ("C2-C1", "C2 $-$ C1 (recuperação)"), ("C1-C0", "C1 $-$ C0 (exemplos fixos)"))
    holm = a.get("familia_confirmatoria_holm") or {}
    por_familia = (a.get("sensibilidade_familias") or {}).get("contrastes") or {}
    ncols = 5 + (1 if por_familia else 0) + (1 if com_holm else 0)

    def linha(chave: str, nome: str) -> str | None:
        c = a["contrastes"].get(f"{chave}|pt")
        if not c:
            return None
        m = c["mcnemar"]
        celulas = [nome, sinal(c["diferenca_pp"], 2), ic(c["ic95"], 2)]
        if por_familia:
            f = por_familia.get(f"{chave}|pt")
            celulas.append(ic(f["ic95_por_familia"], 2) if f else "---")
        celulas += [f"{m['only_a']} / {m['only_b']}", pval(m["p_value"])]
        if com_holm:
            h = holm.get(f"{chave}|pt")
            celulas.append(pval(h["p_holm"]) if h else "---")
        return " & ".join(celulas) + " \\\\"

    linhas: list[str] = []
    if com_holm:
        conf = [ch for ch, _ in papeis if f"{ch}|pt" in holm]
        linhas.append(bloco("Família confirmatória (correção de Holm)", ncols))
        linhas += [x for ch, nome in papeis if ch in conf and (x := linha(ch, nome))]
        linhas.append("\\addlinespace")
        linhas.append(bloco("Contrastes exploratórios", ncols))
        linhas += [x for ch, nome in papeis if ch not in conf and (x := linha(ch, nome))]
    else:
        linhas += [x for ch, nome in papeis if (x := linha(ch, nome))]

    n_ic = 2 if por_familia else 1
    n_p = 2 if com_holm else 1
    sup = (" & & " + grupo("IC de 95\\%", n_ic) + " & & " + grupo("Valor-p", n_p) + " \\\\\n"
           + f"\\cmidrule(lr){{3-{2 + n_ic}}}\\cmidrule(lr){{{4 + n_ic}-{3 + n_ic + n_p}}}\n")
    inf = (b("Contraste") + " & " + b("Diferença (pp)") + " & " + (b("Pergunta") + " & " + b("Família")
           if por_familia else b("Pergunta")) + " & " + b("Discordantes") + " & " + b("Exato")
           + (" & " + b("Holm") if com_holm else ""))
    return tabela(
        "l" + "c" * (ncols - 1), sup + inf, linhas, legenda, rotulo,
        "pp: pontos percentuais. IC: intervalo de confiança de 95\\%, por reamostragem pareada estratificada por "
        "banco, com a pergunta como unidade reamostrada" + ("; na coluna Família, a unidade reamostrada é a família "
        "canônica (análise de sensibilidade)" if por_familia else "") + ". Discordantes: perguntas em que só a "
        "primeira condição acerta / só a segunda acerta. Valor-p: teste exato de McNemar, bilateral"
        + ("; Holm: valor-p ajustado da família confirmatória" if com_holm else "") + "." + ARRED,
        fonte="\\footnotesize",
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
                      + f" & {b(pct(cel['ex']))} & {cel['nao_executaveis']} \\\\")
    d0 = a["celulas"]["C0|pt"]["por_dificuldade"]
    niveis = (("easy", "Fácil"), ("medium", "Médio"), ("hard", "Difícil"), ("extra", "Extra"))
    n_total = a["n_exemplos"]
    cabecalho = (" & " + grupo("Nível de dificuldade", 4) + " & & \\\\\n\\cmidrule(lr){2-5}\n"
                 + b("Condição") + " & " + " & ".join(f"{b(rot)}" for _, rot in niveis) + " & " + b("Total")
                 + " & " + b("Não exec.") + " \\\\\n"
                 + " & " + " & ".join(f"\\footnotesize ($n = {d0[k]['n']}$)" for k, _ in niveis)
                 + f" & \\footnotesize ($n = {n_total}$) &")
    return tabela(
        "lcccccc", cabecalho, linhas,
        "Acurácia de execução em português brasileiro por nível de dificuldade.",
        "tab:resultados_ptbr",
        "Acurácia em \\%, pela classificação oficial de dificuldade do \\textit{Spider}, atribuída à consulta de "
        "referência. Não exec.: número de predições que não executam ou excedem o limite de tempo.",
        fonte="\\footnotesize",
    )


def teste_mcnemar(a: dict) -> str:
    c = a["contrastes"]["C3-C2|pt"]
    m = c["mcnemar"]
    a_c, a_i = m["both_correct"] + m["only_a"], m["only_b"] + m["both_wrong"]  # linhas: C3
    b_c, b_i = m["both_correct"] + m["only_b"], m["only_a"] + m["both_wrong"]  # colunas: C2
    linhas = [f"Correta & {m['both_correct']} & {DESTAQUE}{b(m['only_a'])} & {a_c} \\\\",
              f"Incorreta & {DESTAQUE}{b(m['only_b'])} & {m['both_wrong']} & {a_i} \\\\",
              "\\midrule",
              f"Total & {b_c} & {b_i} & {m['n']} \\\\"]
    cabecalho = (" & " + grupo("C2", 2) + " & \\\\\n\\cmidrule(lr){2-3}\n"
                 + b("C3") + " & " + b("Correta") + " & " + b("Incorreta") + " & " + b("Total"))
    return tabela(
        "lccc", cabecalho, linhas,
        f"Comparação pareada entre as condições C3 e C2 em português brasileiro ($n = {m['n']}$).",
        "tab:mcnemar",
        f"Em destaque, os {m['only_a'] + m['only_b']} pares discordantes, os únicos considerados pelo teste exato "
        f"de McNemar (valor-p de {pval(m['p_value'])}).",
        largura=False,
    )


def teste_recuperador(a: dict) -> str:
    conds = [c for c in ("C1", "C2", "C3", "C4") if a["celulas"].get(f"{c}|pt")]
    k_ef = {a["celulas"][f"{c}|pt"]["k_efetivo_medio"] for c in conds}
    k_unico = len(k_ef) == 1
    linhas = []
    for c in conds:
        pt, en = a["celulas"][f"{c}|pt"], a["celulas"].get(f"{c}|en")
        extra = "" if k_unico else f" & {num(pt['k_efetivo_medio'], 2)}"
        linhas.append(f"{NOME[c]} & {pct(pt['acerto_estrutural'])} & "
                      f"{pct(en['acerto_estrutural']) if en else '---'}{extra} \\\\")
    rec = a["celulas"].get("C4|pt", {}).get("recurso_ordenacao_inicial")
    cabecalho = (" & " + grupo("Acerto estrutural (\\%)", 2) + (" \\\\" if k_unico else " & \\\\")
                 + "\n\\cmidrule(lr){2-3}\n" + b("Condição") + " & " + b("Português") + " & " + b("Inglês")
                 + ("" if k_unico else " & " + b("$k$ efetivo")))
    nota = ("Proporção de perguntas com ao menos uma demonstração, entre as que entraram no \\textit{prompt}, de "
            "assinatura compatível com a da consulta de referência. "
            + (f"O número efetivo de demonstrações foi {num(next(iter(k_ef)), 2)} em todas as condições. "
               if k_unico else "$k$ efetivo: número médio de demonstrações no \\textit{prompt}. ")
            + (f"Em C4, a consulta preliminar recorreu à ordenação inicial em {pct(rec)}\\% das perguntas."
               if rec is not None else ""))
    return tabela(
        "lcc" + ("" if k_unico else "c"), cabecalho, linhas,
        "Desempenho do recuperador no conjunto de teste.", "tab:recuperador", nota,
    )


def teste_idiomas(a: dict) -> str:
    linhas = []
    for c in CONDS:
        pt, en = a["celulas"].get(f"{c}|pt"), a["celulas"].get(f"{c}|en")
        d = (a.get("en_menos_pt") or {}).get(c)
        if not (pt and en and d):
            continue
        linhas.append(f"{NOME[c]} & {pct(pt['ex'])} & {pct(en['ex'])} & {sinal(d['diferenca_pp'])} & {ic(d['ic95'])} \\\\")
    if not linhas:
        return ""  # análise parcial, sem o inglês
    cabecalho = (" & " + grupo("Acurácia de execução (\\%)", 2) + " & " + grupo("Diferença EN $-$ PT", 2)
                 + " \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
                 + b("Condição") + " & " + b("Português") + " & " + b("Inglês") + " & " + b("pp") + " & "
                 + b("IC de 95\\%"))
    return tabela(
        "lcccc", cabecalho, linhas,
        f"Acurácia de execução por idioma e diferença entre os idiomas ($n = {a['n_exemplos']}$).",
        "tab:idiomas",
        "pp: pontos percentuais. IC: intervalo de confiança de 95\\%, por reamostragem pareada estratificada por "
        "banco." + ARRED,
    )


def teste_custo(a: dict) -> str:
    """Custo por condição. As colunas de latência só existem quando a passagem dedicada foi medida."""
    lat = (a.get("custo") or {}).get("latencia") or {}
    tem_lat = all(lat.get(f"{c}|pt") for c in CONDS)
    linhas = []
    for c in CONDS:
        cel = a["celulas"].get(f"{c}|pt")
        cu = a["custo"]["por_celula"].get(f"{c}|pt")
        if not cel:
            continue
        l = lat.get(f"{c}|pt")
        ganho = "---" if c == "C0" or cu["ganho_pp_por_100_tokens"] is None else sinal(cu["ganho_pp_por_100_tokens"], 2)
        latencia = (f"{num(l['total_ms_p50'] / 1000, 2)} & {num(l['total_ms_p95'] / 1000, 2)} & " if tem_lat else "")
        linhas.append(
            f"{c} & {pct(cel['ex'])} & {cu['chamadas_ao_modelo']} & "
            f"{inteiro(round(cel['prompt_tokens_total_medio']))} & "
            f"{num(cel['completion_tokens_total_medio'], 0)} & {num(cel['latencia_lote_por_item_ms_media'] / 1000, 2)} & "
            f"{latencia}{ganho} \\\\"
        )
    prot = (a.get("custo") or {}).get("latencia_protocolo") or {}
    nota = ("Condições como no Quadro \\ref{tab:condicoes}. EX: acurácia de execução. Rodada principal: médias por pergunta no conjunto de teste em português; em "
            "C4, os \\textit{tokens} somam as duas chamadas ao modelo. Tempo: tempo de geração do lote de quatro "
            "perguntas dividido por quatro; em C4, não inclui a consulta preliminar, cujo tempo é discutido no texto. "
            + (f"Latência: passagem dedicada, com uma pergunta por vez ({prot.get('n_medicoes')} medições; "
               f"{prot.get('gpu')}), mediana (p50) e percentil 95 (p95). " if tem_lat and prot else "")
            + "Ganho marginal: pontos percentuais de acurácia por cem \\textit{tokens} de entrada adicionais em "
              "relação a C0.")
    sup = (" & & " + grupo("Rodada principal (média por pergunta)", 4)
           + (" & " + grupo("Latência (s)", 2) if tem_lat else "")
           + " & " + b("Ganho") + " \\\\\n\\cmidrule(lr){3-6}" + ("\\cmidrule(lr){7-8}" if tem_lat else "") + "\n")
    inf = (b("Condição") + " & " + b("EX (\\%)") + " & " + b("Chamadas") + " & " + b("Tokens ent.") + " & "
           + b("Tokens saída") + " & " + b("Tempo (s)") + (" & " + b("p50") + " & " + b("p95") if tem_lat else "")
           + " & " + b("marginal"))
    return tabela(
        "lcccccc" + ("cc" if tem_lat else ""), sup + inf,
        linhas,
        "Acurácia, custo e ganho marginal por condição, em português brasileiro.",
        "tab:custo",
        nota,
        posicao="!htbp",
        fonte="\\footnotesize",
    )


def teste_erros(a: dict) -> str:
    rub = a["erros"]["rubrica"]
    conds = [c for c in CONDS if f"{c}|pt" in a["erros"]["por_celula"]]
    nao_executa = ("identificador", "invalida")
    executa = ("tabelas", "agregacao", "filtro", "literal", "projecao", "ordem", "outras", "nao_analisada")
    # Os rótulos da rubrica trazem, entre parênteses, se a consulta executa; com os blocos, isso fica implícito.
    curto = {k: rub[k].split(" (")[0] if k in nao_executa else rub[k] for k in rub}
    ncols = 1 + len(conds)

    def linha(k: str) -> str | None:
        valores = [a["erros"]["por_celula"][f"{c}|pt"][k] for c in conds]
        if k == "nao_analisada" and not any(valores):
            return None
        return "\\quad " + curto[k] + " & " + " & ".join(str(v) for v in valores) + " \\\\"

    linhas = [bloco("A consulta não executa (ou excede o limite de tempo)", ncols)]
    linhas += [x for k in nao_executa if (x := linha(k))]
    linhas.append("\\addlinespace")
    linhas.append(bloco("A consulta executa, mas difere da referência em", ncols))
    linhas += [x for k in executa if (x := linha(k))]
    linhas.append("\\midrule")
    linhas.append(b("Total de predições incorretas") + " & "
                  + " & ".join(b(a['erros']['por_celula'][f'{c}|pt']['total']) for c in conds) + " \\\\")
    return tabela(
        ">{\\raggedright\\arraybackslash}X" + "c" * len(conds),
        b("Categoria (primeira regra que se aplica)") + " & " + " & ".join(b(c) for c in conds),
        linhas,
        "Predições incorretas em português brasileiro, por categoria de erro.",
        "tab:erros",
        f"Contagens por condição, sobre {a['n_exemplos']} perguntas; categorias da rubrica automática da "
        "Seção \\ref{sec:RubricaErros}, aplicadas na ordem em que aparecem.",
        largura="x",
    )


def teste_h3(a: dict) -> str:
    linhas = []
    for c in ("C1", "C2", "C3", "C4"):
        h = a["h3"].get(f"{c}|pt")
        if not h:
            continue
        t = h["tabela"]
        n_com, n_sem = t["hit_correta"] + t["hit_incorreta"], t["sem_hit_correta"] + t["sem_hit_incorreta"]
        dif = "---" if h["diferenca_pp"] is None else f"{sinal(h['diferenca_pp'])} {ic(h['ic95'])}"
        aj = ("---" if h["diferenca_ajustada_dificuldade_pp"] is None else
              f"{sinal(h['diferenca_ajustada_dificuldade_pp'])} {ic(h['ic95_ajustada'])}")
        linhas.append(f"{c} & {pct(h['ex_com_acerto_estrutural'])} & {n_com} & "
                      f"{pct(h['ex_sem_acerto_estrutural'])} & {n_sem} & {dif} & {aj} \\\\")
    cabecalho = (" & " + grupo("Com acerto estr.", 2) + " & " + grupo("Sem acerto estr.", 2) + " & "
                 + grupo("Diferença (pp) e IC de 95\\%", 2) + " \\\\\n"
                 "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\n"
                 + b("Condição") + " & " + b("EX (\\%)") + " & " + b("$n$") + " & " + b("EX (\\%)") + " & "
                 + b("$n$") + " & " + b("Bruta") + " & " + b("Ajustada"))
    return tabela(
        "lcccccc", cabecalho, linhas,
        "Acurácia de execução com e sem acerto estrutural, em português brasileiro.",
        "tab:h3",
        "Acerto estrutural: presença, no \\textit{prompt}, de ao menos uma demonstração com assinatura compatível "
        "com a da consulta de referência. $n$: número de perguntas do grupo. Diferença: acurácia com acerto "
        "estrutural menos acurácia sem ele, em pontos percentuais (pp), com intervalo de confiança de 95\\% entre "
        "colchetes; a ajustada é a média das diferenças por nível de dificuldade, ponderada pelo número de "
        "perguntas de cada nível. Associações observacionais.",
        fonte="\\footnotesize",
    )


def teste_literal(a: dict) -> str:
    """Tabela transposta: uma linha por medida e uma coluna por estrato, com o intervalo na mesma célula."""
    lt = a.get("literal_traduzido")
    if not lt:
        return ""
    estratos = [(e, lt["por_estrato"][e]) for e in ("LT", "SL") if lt["por_estrato"].get(e)]
    if not estratos:
        return ""

    def cel(d):
        return f"{sinal(d['diferenca_pp'])} {ic(d['ic95'])}" if d else "---"

    medidas = [
        ("EX em C0, português (\\%)", lambda e: pct(e["ex"].get("C0|pt"))),
        ("EX em C0, inglês (\\%)", lambda e: pct(e["ex"].get("C0|en"))),
        ("Diferença EN $-$ PT em C0 (pp)", lambda e: cel((e.get("en_menos_pt") or {}).get("C0"))),
        ("Diferença EN $-$ PT em C4 (pp)", lambda e: cel((e.get("en_menos_pt") or {}).get("C4"))),
        ("Contraste C3 $-$ C2 em português (pp)", lambda e: cel(e.get("c3_menos_c2_pt"))),
    ]
    linhas = [f"{nome} & " + " & ".join(f(e) for _, e in estratos) + " \\\\" for nome, f in medidas]
    rot = {"LT": "Com literal traduzido", "SL": "Sem literal traduzido"}
    cabecalho = (b("Medida") + " & " + " & ".join(b(rot[k]) for k, _ in estratos) + " \\\\\n"
                 + " & " + " & ".join(f"\\footnotesize ($n = {e['n']}$)" for _, e in estratos))
    nota = ("EX: acurácia de execução; pp: pontos percentuais; entre colchetes, o intervalo de confiança de 95\\%, "
            "por reamostragem pareada estratificada por banco.")
    c32 = dict(estratos).get("LT", {}).get("c3_menos_c2_pt")
    if c32 and c32["ic95"][0] == c32["ic95"][1]:
        nota += (f" No estrato com literal traduzido, o intervalo do contraste C3 $-$ C2 tem amplitude nula: as "
                 f"{c32['discordantes']} perguntas discordantes, ambas favoráveis a C3, são as únicas do estrato em "
                 "seu banco, e a reamostragem estratificada por banco as inclui em todas as reamostras.")
    return tabela(
        "lcc", cabecalho, linhas,
        "Análise estratificada por literal traduzido (exploratória).",
        "tab:literal", nota,
    )


def teste_em(a: dict) -> str:
    linhas = []
    for c in CONDS:
        pt, en = a["celulas"].get(f"{c}|pt"), a["celulas"].get(f"{c}|en")
        if not pt:
            continue
        linhas.append(f"{NOME[c]} & {pct(pt['em_oficial'])} & {pt['em_predicao_nao_analisavel']} & "
                      f"{pct(en['em_oficial']) if en else '---'} & {en['em_predicao_nao_analisavel'] if en else '---'} \\\\")
    cabecalho = (" & " + grupo("Português", 2) + " & " + grupo("Inglês", 2)
                 + " \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\n"
                 + b("Condição") + " & " + b("EM (\\%)") + " & " + b("Não analis.") + " & " + b("EM (\\%)")
                 + " & " + b("Não analis."))
    return tabela(
        "lcccc", cabecalho, linhas,
        "Correspondência exata de conjuntos (EM oficial) no conjunto de teste.",
        "tab:em",
        "Não analis.: número de predições que o analisador do avaliador oficial não interpreta, contadas como "
        "incorretas.",
    )


def main(val: Path, teste: Path | None, saida: Path) -> None:
    saida.mkdir(parents=True, exist_ok=True)
    arquivos: dict[str, str] = {}
    av = json.loads(val.read_text(encoding="utf-8"))
    arquivos["val_condicoes"] = val_condicoes(av)
    arquivos["val_contrastes"] = contrastes(
        av, "tab:validacao_contrastes",
        f"Contrastes pareados em português brasileiro na validação ($n = {av['n_exemplos']}$).",
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
