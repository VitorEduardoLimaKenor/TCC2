"""Análise dos registros de execução (E08, E09, E10 e E17).

Uma única função, `analisar`, recebe as linhas de um conjunto de logs já
validado por `logging_io.report_rows` e devolve um dicionário com tudo o que a
monografia reporta: acurácia por condição e idioma, contrastes pareados,
correção de Holm, diferença entre idiomas (H2), associação entre acerto
estrutural e correção (H3), estratos de literal traduzido, custo, sobreposição
das demonstrações, rubrica automática de erros e, na validação, o poder do
teste primário para o tamanho do conjunto de teste.

Regras fixadas antes da abertura do conjunto de teste (manifesto de
congelamento de 11/11/2026):

* um exemplo com referência inválida em qualquer condição ou idioma sai de
  todas as análises (D-006);
* comparação primária: C3 − C2 em português; H1: C4 − C2 em português; as duas
  formam a família confirmatória, com correção de Holm a 5% bilateral (D-009);
* todo o resto é exploratório e sai com intervalo de confiança, sem linguagem
  de confirmação;
* os intervalos são de 95%, por reamostragem pareada estratificada por banco,
  com 10.000 reamostras e a semente da configuração.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import asdict
from typing import Any, Iterable, Sequence

import numpy as np

from .evaluation import stats

CONDICOES = ("C0", "C1", "C2", "C3", "C4")
IDIOMAS = ("pt", "en")
DIFICULDADES = ("easy", "medium", "hard", "extra")
REAMOSTRAS = 10000

CONTRASTES = (
    # (tratada, referência, papel)
    ("C3", "C2", "primaria"),
    ("C4", "C2", "h1"),
    ("C4", "C3", "componente"),
    ("C2", "C1", "exploratoria"),
    ("C1", "C0", "exploratoria"),
    ("C3", "C0", "exploratoria"),
)

RUBRICA = {
    "identificador": "Identificador inexistente ou ambíguo (consulta não executa)",
    "invalida": "Consulta inválida por outro motivo (não executa ou excede o tempo)",
    "tabelas": "Tabelas ou junções diferentes das da referência",
    "agregacao": "Agregação ou agrupamento diferente",
    "filtro": "Condição de filtragem diferente (coluna ou operador)",
    "literal": "Mesma condição de filtragem com valor literal diferente",
    "projecao": "Colunas projetadas diferentes",
    "ordem": "Ordenação, limite, DISTINCT ou operação de conjunto diferente",
    "outras": "Mesma estrutura nos itens acima, com resultado diferente",
    "nao_analisada": "Consulta que o analisador sintático não interpretou",
}


# ------------------------------------------------------------------ utilidades


def _media(valores: Iterable[float]) -> float | None:
    valores = [v for v in valores if v is not None]
    return float(np.mean(valores)) if valores else None


def _indices_estratificados(estratos: Sequence[str], rng: np.random.Generator,
                            reamostras: int = REAMOSTRAS) -> np.ndarray:
    """Matriz (reamostras × n) de índices sorteados com reposição dentro de cada estrato."""
    por_estrato: dict[str, list[int]] = defaultdict(list)
    for i, chave in enumerate(estratos):
        por_estrato[chave].append(i)
    blocos = []
    for indices in por_estrato.values():
        arr = np.asarray(indices)
        blocos.append(arr[rng.integers(0, len(arr), size=(reamostras, len(arr)))])
    return np.concatenate(blocos, axis=1)


def _ic(amostras: np.ndarray, nivel: float = 0.95) -> tuple[float, float]:
    amostras = amostras[~np.isnan(amostras)]
    if amostras.size == 0:
        return (float("nan"), float("nan"))
    alfa = (1 - nivel) / 2
    return (float(np.quantile(amostras, alfa)), float(np.quantile(amostras, 1 - alfa)))


def ic_por_familia(a: Sequence[bool], b: Sequence[bool], familias: Sequence[str], bancos: Sequence[str],
                   rng: np.random.Generator, reamostras: int = REAMOSTRAS) -> tuple[float, float]:
    """IC de 95% da diferença de acurácia reamostrando famílias inteiras dentro de cada banco.

    Perguntas da mesma família canônica têm a mesma consulta de referência, a
    menos de literais, e seus acertos não são independentes. Aqui a unidade
    sorteada é a família, com todas as suas perguntas, e não a pergunta.
    """
    dif = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    grupos: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for i, (fam, banco) in enumerate(zip(familias, bancos)):
        grupos[banco][fam].append(i)
    soma = np.zeros(reamostras)
    total = np.zeros(reamostras)
    for por_familia in grupos.values():
        d = np.array([dif[idx].sum() for idx in por_familia.values()])
        n = np.array([len(idx) for idx in por_familia.values()], dtype=float)
        sorteio = rng.integers(0, len(d), size=(reamostras, len(d)))
        soma += d[sorteio].sum(axis=1)
        total += n[sorteio].sum(axis=1)
    return _ic(100 * soma / total)


def holm(pvalores: dict[str, float], alfa: float = 0.05) -> dict[str, dict[str, Any]]:
    """Correção de Holm: valor-p ajustado e decisão a `alfa`, por hipótese."""
    ordenados = sorted(pvalores.items(), key=lambda kv: kv[1])
    m = len(ordenados)
    saida: dict[str, dict[str, Any]] = {}
    anterior = 0.0
    rejeitando = True
    for posicao, (nome, p) in enumerate(ordenados):
        ajustado = min(1.0, max(anterior, (m - posicao) * p))
        anterior = ajustado
        rejeita = rejeitando and p <= alfa / (m - posicao)
        rejeitando = rejeita
        saida[nome] = {"p": p, "p_holm": ajustado, "rejeita_h0": bool(rejeita)}
    return saida


# ------------------------------------------------------------- poder do teste


def _rejeita_mcnemar(b: int, d: int, alfa: float) -> bool:
    """O teste exato bilateral rejeita com `b` de `d` pares discordantes a favor de A?"""
    if d == 0:
        return False
    cauda = sum(math.comb(d, k) for k in range(0, min(b, d - b) + 1)) / 2 ** d
    return min(1.0, 2 * cauda) <= alfa


def poder_mcnemar(n: int, taxa_discordancia: float, diferenca: float, alfa: float = 0.05) -> float:
    """Poder do teste exato de McNemar bilateral.

    `taxa_discordancia` é a proporção de pares em que as duas condições
    divergem; `diferenca` é a diferença de acurácia (proporção, A − B). Os pares
    a favor de A têm probabilidade (taxa + diferenca)/2 e os a favor de B,
    (taxa − diferenca)/2. O cálculo é exato: soma sobre o número de pares
    discordantes e sobre a divisão entre eles.
    """
    if taxa_discordancia <= 0 or abs(diferenca) > taxa_discordancia:
        return float("nan")
    from scipy.stats import binom

    p_a = (taxa_discordancia + diferenca) / (2 * taxa_discordancia)
    poder = 0.0
    lo, hi = binom.interval(1 - 1e-10, n, taxa_discordancia)
    for d in range(int(lo), int(hi) + 1):
        peso = binom.pmf(d, n, taxa_discordancia)
        if peso < 1e-14 or d == 0:
            continue
        rejeicoes = [b for b in range(d + 1) if _rejeita_mcnemar(b, d, alfa)]
        if rejeicoes:
            poder += peso * float(np.sum(binom.pmf(rejeicoes, d, p_a)))
    return float(poder)


def diferenca_minima_detectavel(n: int, taxa_discordancia: float, alfa: float = 0.05,
                                poder: float = 0.80) -> float | None:
    """Menor diferença de acurácia (proporção) detectável com o poder pedido."""
    if taxa_discordancia <= 0:
        return None
    passo = 0.0025
    delta = passo
    while delta <= taxa_discordancia + 1e-12:
        if poder_mcnemar(n, taxa_discordancia, delta, alfa) >= poder:
            return round(delta, 4)
        delta += passo
    return None


# ----------------------------------------------------------- rubrica de erros


def _estrutura(sql: str) -> dict[str, Any] | None:
    import sqlglot
    from sqlglot import exp

    try:
        arvore = sqlglot.parse_one(sql, read="sqlite")
    except Exception:  # noqa: BLE001
        return None
    if arvore is None:
        return None

    def nome(col: exp.Column) -> str:
        return col.name.lower()

    tabelas = sorted({t.name.lower() for t in arvore.find_all(exp.Table)})
    agregacoes = sorted(type(a).__name__.lower() for a in arvore.find_all(exp.AggFunc))
    grupos = sorted({nome(c) for g in arvore.find_all(exp.Group) for c in g.find_all(exp.Column)})
    tem_having = arvore.find(exp.Having) is not None
    condicoes, literais = set(), set()
    for where in list(arvore.find_all(exp.Where)) + list(arvore.find_all(exp.Having)):
        for pred in where.find_all(exp.Binary, exp.In, exp.Between, exp.Like, exp.Is, exp.Not):
            if isinstance(pred, (exp.And, exp.Or)):
                continue
            colunas = sorted({nome(c) for c in pred.find_all(exp.Column)})
            if not colunas:
                continue
            condicoes.add((type(pred).__name__.lower(), tuple(colunas)))
            for lit in pred.find_all(exp.Literal):
                literais.add(str(lit.this).strip().strip("%").lower())
    selecao = arvore.find(exp.Select)
    projecao = sorted(
        {nome(c) for e in (selecao.expressions if selecao else []) for c in e.find_all(exp.Column)}
        | {"*" for e in (selecao.expressions if selecao else []) if e.find(exp.Star)}
    )
    ordem = (
        sorted(nome(c) for o in arvore.find_all(exp.Order) for c in o.find_all(exp.Column)),
        sorted(str(o.args.get("desc")) for o in arvore.find_all(exp.Ordered)),
        arvore.find(exp.Limit) is not None,
        arvore.find(exp.Distinct) is not None,
        sorted(type(s).__name__ for s in arvore.find_all(exp.Union, exp.Intersect, exp.Except)),
    )
    return {"tabelas": tabelas, "agregacoes": agregacoes, "grupos": grupos, "having": tem_having,
            "condicoes": sorted(condicoes), "literais": sorted(literais), "projecao": projecao,
            "ordem": ordem}


def classificar_erro(row: dict[str, Any]) -> str | None:
    """Categoria da rubrica automática para uma predição incorreta; `None` se correta.

    As regras são aplicadas em ordem e a primeira que se verifica decide. A
    rubrica compara a estrutura da predição com a da referência e não julga
    equivalência semântica: uma predição correta escrita de outra forma não
    chega aqui, e uma referência com erro de anotação aparece como divergência
    da predição.
    """
    if row.get("ex_correct") is not False:
        return None
    if row.get("ex_status") in ("erro_predicao", "tempo_predicao") or not row.get("exec_ok", True):
        erro = (row.get("exec_error") or "").lower()
        if "no such column" in erro or "no such table" in erro or "ambiguous column" in erro:
            return "identificador"
        return "invalida"
    pred, ref = _estrutura(row.get("generated_sql") or ""), _estrutura(row.get("gold_sql") or "")
    if pred is None or ref is None:
        return "nao_analisada"
    if pred["tabelas"] != ref["tabelas"]:
        return "tabelas"
    if (pred["agregacoes"], pred["grupos"], pred["having"]) != (ref["agregacoes"], ref["grupos"], ref["having"]):
        return "agregacao"
    if pred["condicoes"] != ref["condicoes"]:
        return "filtro"
    if pred["literais"] != ref["literais"]:
        return "literal"
    if pred["projecao"] != ref["projecao"]:
        return "projecao"
    if pred["ordem"] != ref["ordem"]:
        return "ordem"
    return "outras"


# ------------------------------------------------------------------- análise


class _Dados:
    """Linhas organizadas por (condição, idioma) e alinhadas pelos mesmos exemplos."""

    def __init__(self, rows: list[dict[str, Any]]):
        invalidos = {r["example_id"] for r in rows if r.get("ex_correct") is None}
        self.invalidos = sorted(invalidos)
        self.celulas: dict[tuple[str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
        for r in rows:
            if r["example_id"] in invalidos:
                continue
            self.celulas[(r["condition"], r["lang"])][r["example_id"]] = r
        self.idiomas = [l for l in IDIOMAS if any(k[1] == l for k in self.celulas)]
        self.condicoes = [c for c in CONDICOES if any(k[0] == c for k in self.celulas)]
        conjuntos = [set(v) for v in self.celulas.values()]
        if not conjuntos or any(c != conjuntos[0] for c in conjuntos[1:]):
            raise ValueError("pareamento incompleto: as condições e os idiomas precisam cobrir "
                             "os mesmos exemplos")
        self.ids = sorted(conjuntos[0])
        qualquer = next(iter(self.celulas.values()))
        self.banco = [qualquer[i]["db_id"] for i in self.ids]
        self.dificuldade = [qualquer[i].get("dificuldade_oficial") for i in self.ids]

    def tem(self, cond: str, lang: str) -> bool:
        return (cond, lang) in self.celulas

    def vetor(self, cond: str, lang: str, campo: str = "ex_correct") -> list[bool]:
        cel = self.celulas[(cond, lang)]
        return [bool(cel[i].get(campo)) for i in self.ids]

    def linhas(self, cond: str, lang: str) -> list[dict[str, Any]]:
        cel = self.celulas[(cond, lang)]
        return [cel[i] for i in self.ids]


def _celula(linhas: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(linhas)
    com_em = [r for r in linhas if r.get("em_oficial") is not None]
    por_dif = {}
    for dif in DIFICULDADES:
        sub = [r for r in linhas if r.get("dificuldade_oficial") == dif]
        por_dif[dif] = {"n": len(sub), "acertos": sum(bool(r["ex_correct"]) for r in sub),
                        "ex": (sum(bool(r["ex_correct"]) for r in sub) / len(sub)) if sub else None}
    rascunho = [r.get("rascunho_prompt_tokens") for r in linhas]
    tokens_total = [(r.get("prompt_tokens") or 0) + (r.get("rascunho_prompt_tokens") or 0) for r in linhas]
    saida_total = [(r.get("completion_tokens") or 0) + (r.get("rascunho_completion_tokens") or 0)
                   for r in linhas]
    picos = [r["gpu_mem_pico_mb"] for r in linhas if r.get("gpu_mem_pico_mb") is not None]
    return {
        "n": n,
        "acertos": sum(bool(r["ex_correct"]) for r in linhas),
        "ex": sum(bool(r["ex_correct"]) for r in linhas) / n,
        "em_oficial": (sum(bool(r["em_oficial"]) for r in com_em) / len(com_em)) if com_em else None,
        "em_n": len(com_em),
        "em_predicao_nao_analisavel": sum(r.get("em_status") == "predicao_nao_analisavel" for r in linhas),
        "nao_executaveis": sum(r.get("ex_status") in ("erro_predicao", "tempo_predicao") for r in linhas),
        "tempo_excedido": sum(r.get("ex_status") == "tempo_predicao" for r in linhas),
        "acertos_ambos_vazios": sum(bool(r.get("ex_ambos_vazios")) and bool(r["ex_correct"]) for r in linhas),
        "acerto_estrutural": sum(bool(r.get("structural_hit")) for r in linhas) / n,
        "recurso_ordenacao_inicial": sum(bool(r.get("used_fallback")) for r in linhas) / n,
        "k_efetivo_medio": _media(r.get("k_efetivo", r.get("n_demos_used")) for r in linhas),
        "com_demonstracao_removida": sum((r.get("n_demos_dropped") or 0) > 0 for r in linhas),
        "no_limite_de_entrada": sum(bool(r.get("prompt_no_limite_de_entrada")) for r in linhas),
        "prompt_tokens_medio": _media(r.get("prompt_tokens") for r in linhas),
        "completion_tokens_medio": _media(r.get("completion_tokens") for r in linhas),
        "rascunho_prompt_tokens_medio": _media(rascunho),
        "prompt_tokens_total_medio": float(np.mean(tokens_total)),
        "completion_tokens_total_medio": float(np.mean(saida_total)),
        "latencia_lote_por_item_ms_media": _media(r.get("latency_ms") for r in linhas),
        "gpu_mem_pico_mb_max": max(picos) if picos else None,
        "por_dificuldade": por_dif,
    }


def _contraste(a: list[bool], b: list[bool], bancos: list[str], semente: int) -> dict[str, Any]:
    mc = stats.mcnemar_exact(a, b)
    ic = stats.paired_bootstrap_diff(a, b, bancos, resamples=REAMOSTRAS, seed=semente)
    return {"mcnemar": asdict(mc), "discordantes": mc.discordant,
            "taxa_discordancia": mc.discordant / max(1, mc.n),
            "diferenca_pp": ic.point, "ic95": [ic.low, ic.high], "ic_exclui_zero": ic.excludes_zero}


def _h3(dados: _Dados, cond: str, lang: str, rng: np.random.Generator) -> dict[str, Any]:
    acerto = np.array(dados.vetor(cond, lang, "ex_correct"), dtype=float)
    hit = np.array(dados.vetor(cond, lang, "structural_hit"), dtype=float)
    n11 = int(np.sum((hit == 1) & (acerto == 1)))
    n10 = int(np.sum((hit == 1) & (acerto == 0)))
    n01 = int(np.sum((hit == 0) & (acerto == 1)))
    n00 = int(np.sum((hit == 0) & (acerto == 0)))

    def diferenca(ac: np.ndarray, ht: np.ndarray) -> np.ndarray:
        com = ht.sum(axis=-1)
        sem = (1 - ht).sum(axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return 100 * ((ac * ht).sum(axis=-1) / com - (ac * (1 - ht)).sum(axis=-1) / sem)

    idx = _indices_estratificados(dados.banco, rng)
    bruta = float(diferenca(acerto, hit)) if (n11 + n10) and (n01 + n00) else None
    ic_bruta = _ic(diferenca(acerto[idx], hit[idx])) if bruta is not None else None

    # Ajuste pela dificuldade oficial: média das diferenças dentro de cada
    # nível, ponderada pelo número de perguntas do nível (só níveis com as
    # duas situações de acerto estrutural).
    niveis = np.array([d or "?" for d in dados.dificuldade])

    def ajustada(ac: np.ndarray, ht: np.ndarray, nv: np.ndarray) -> float:
        soma = peso = 0.0
        for nivel in DIFICULDADES:
            m = nv == nivel
            com, sem = ht[m].sum(), (1 - ht[m]).sum()
            if com == 0 or sem == 0:
                continue
            d = (ac[m] * ht[m]).sum() / com - (ac[m] * (1 - ht[m])).sum() / sem
            soma += m.sum() * d
            peso += m.sum()
        return 100 * soma / peso if peso else float("nan")

    aj = ajustada(acerto, hit, niveis)
    amostras = np.array([ajustada(acerto[i], hit[i], niveis[i]) for i in idx[:2000]])
    por_nivel = {}
    for nivel in DIFICULDADES:
        m = niveis == nivel
        com, sem = int(hit[m].sum()), int((1 - hit[m]).sum())
        por_nivel[nivel] = {
            "com_acerto_estrutural": {"n": com, "ex": float(acerto[m][hit[m] == 1].mean()) if com else None},
            "sem_acerto_estrutural": {"n": sem, "ex": float(acerto[m][hit[m] == 0].mean()) if sem else None},
        }
    return {
        "tabela": {"hit_correta": n11, "hit_incorreta": n10, "sem_hit_correta": n01, "sem_hit_incorreta": n00},
        "ex_com_acerto_estrutural": n11 / (n11 + n10) if (n11 + n10) else None,
        "ex_sem_acerto_estrutural": n01 / (n01 + n00) if (n01 + n00) else None,
        "diferenca_pp": bruta, "ic95": list(ic_bruta) if ic_bruta else None,
        "diferenca_ajustada_dificuldade_pp": None if math.isnan(aj) else aj,
        "ic95_ajustada": list(_ic(amostras)), "reamostras_ajustada": int(len(amostras)),
        "por_dificuldade": por_nivel,
    }


def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def analisar(
    rows: list[dict[str, Any]],
    estratos: dict[str, Any] | None = None,
    custo_rows: list[dict[str, Any]] | None = None,
    semente: int = 20260904,
    n_teste: int = 471,
    ex_distinct: dict[tuple[str, str, str], bool | None] | None = None,
    familias: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Todas as estimativas reportadas, a partir das linhas de um split.

    `estratos` é o conteúdo de `estratos_literais.json`; `custo_rows`, as
    medições de `tcc-rag custo`; `ex_distinct`, o veredito da variante do
    avaliador que preserva DISTINCT, por (condição, idioma, exemplo);
    `familias`, a família canônica de cada exemplo (do manifesto), para a
    análise de sensibilidade em que a unidade reamostrada é a família.
    """
    dados = _Dados(rows)
    rng = np.random.default_rng(semente)
    split = rows[0].get("split")
    saida: dict[str, Any] = {
        "split": split,
        "n_exemplos": len(dados.ids),
        "referencias_invalidas": dados.invalidos,
        "condicoes": dados.condicoes,
        "idiomas": dados.idiomas,
        "bancos": dict(sorted(Counter(dados.banco).items())),
        "dificuldade": dict(Counter(d or "sem_classificacao" for d in dados.dificuldade)),
        "reamostras": REAMOSTRAS,
        "semente": semente,
    }

    # ---- 1. por condição e idioma
    celulas = {}
    for cond in dados.condicoes:
        for lang in dados.idiomas:
            if dados.tem(cond, lang):
                celulas[f"{cond}|{lang}"] = _celula(dados.linhas(cond, lang))
    saida["celulas"] = celulas

    # ---- 2. contrastes pareados
    contrastes: dict[str, Any] = {}
    for tratada, referencia, papel in CONTRASTES:
        for lang in dados.idiomas:
            if dados.tem(tratada, lang) and dados.tem(referencia, lang):
                c = _contraste(dados.vetor(tratada, lang), dados.vetor(referencia, lang), dados.banco, semente)
                c["papel"] = papel if lang == "pt" else "exploratoria"
                contrastes[f"{tratada}-{referencia}|{lang}"] = c
    saida["contrastes"] = contrastes
    familia = {k: contrastes[k]["mcnemar"]["p_value"] for k in ("C3-C2|pt", "C4-C2|pt") if k in contrastes}
    saida["familia_confirmatoria_holm"] = holm(familia) if len(familia) == 2 else None

    # ---- 2b. sensibilidade: reamostragem por família canônica (unidade = família)
    if familias:
        fam = [familias.get(i, i) for i in dados.ids]
        tamanhos = Counter(fam)
        sens: dict[str, Any] = {"n_familias": len(tamanhos), "maior_familia": max(tamanhos.values()),
                                "familias_com_mais_de_uma_pergunta": sum(1 for v in tamanhos.values() if v > 1),
                                "contrastes": {}}
        for tratada, referencia in (("C3", "C2"), ("C4", "C2"), ("C4", "C3")):
            for lang in dados.idiomas:
                if dados.tem(tratada, lang) and dados.tem(referencia, lang):
                    lo, hi = ic_por_familia(dados.vetor(tratada, lang), dados.vetor(referencia, lang), fam,
                                            dados.banco, rng)
                    sens["contrastes"][f"{tratada}-{referencia}|{lang}"] = {
                        "diferenca_pp": contrastes[f"{tratada}-{referencia}|{lang}"]["diferenca_pp"],
                        "ic95_por_familia": [lo, hi], "ic_exclui_zero": lo > 0 or hi < 0}
        saida["sensibilidade_familias"] = sens

    # ---- 3. idiomas: EN − PT por condição e diferenças em diferenças (H2)
    if set(dados.idiomas) == {"pt", "en"}:
        entre = {}
        for cond in dados.condicoes:
            if dados.tem(cond, "pt") and dados.tem(cond, "en"):
                c = _contraste(dados.vetor(cond, "en"), dados.vetor(cond, "pt"), dados.banco, semente)
                entre[cond] = c
        saida["en_menos_pt"] = entre
        did = {}
        for tratada, rotulo in (("C4", "h2_original_c4_c0"), ("C3", "exploratoria_c3_c0")):
            if all(dados.tem(c, l) for c in (tratada, "C0") for l in ("pt", "en")):
                d = stats.difference_in_differences(
                    dados.vetor(tratada, "pt"), dados.vetor("C0", "pt"),
                    dados.vetor(tratada, "en"), dados.vetor("C0", "en"),
                    dados.banco, resamples=REAMOSTRAS, seed=semente)
                did[rotulo] = {**asdict(d), "conclusivo": d.conclusive}
        saida["diferencas_em_diferencas"] = did

    # ---- 4. H3: acerto estrutural × correção
    saida["h3"] = {
        f"{cond}|{lang}": _h3(dados, cond, lang, rng)
        for cond in dados.condicoes if cond != "C0"
        for lang in dados.idiomas if dados.tem(cond, lang)
    }

    # ---- 5. estratos de literal traduzido (pré-registro de 28/10/2026)
    if estratos:
        itens = estratos.get("itens", estratos)
        rotulo = [itens.get(i, {}).get("estrato") for i in dados.ids]
        lt: dict[str, Any] = {"tamanho": dict(Counter(r or "sem_estrato" for r in rotulo)), "por_estrato": {}}
        for estrato in ("LT", "SL"):
            m = [i for i, r in enumerate(rotulo) if r == estrato]
            if not m:
                continue
            bancos = [dados.banco[i] for i in m]
            bloco: dict[str, Any] = {"n": len(m), "ex": {}}
            for cond in dados.condicoes:
                for lang in dados.idiomas:
                    if dados.tem(cond, lang):
                        v = dados.vetor(cond, lang)
                        bloco["ex"][f"{cond}|{lang}"] = sum(v[i] for i in m) / len(m)
            if set(dados.idiomas) == {"pt", "en"}:
                bloco["en_menos_pt"] = {}
                for cond in ("C0", "C4", "C3"):
                    if dados.tem(cond, "pt") and dados.tem(cond, "en"):
                        en, pt = dados.vetor(cond, "en"), dados.vetor(cond, "pt")
                        bloco["en_menos_pt"][cond] = _contraste([en[i] for i in m], [pt[i] for i in m],
                                                                bancos, semente)
            if dados.tem("C3", "pt") and dados.tem("C2", "pt"):
                c3, c2 = dados.vetor("C3", "pt"), dados.vetor("C2", "pt")
                bloco["c3_menos_c2_pt"] = _contraste([c3[i] for i in m], [c2[i] for i in m], bancos, semente)
            lt["por_estrato"][estrato] = bloco
        saida["literal_traduzido"] = lt

    # ---- 6. custo
    custo: dict[str, Any] = {"por_celula": {}}
    for chave, cel in celulas.items():
        cond, lang = chave.split("|")
        base = celulas.get(f"C0|{lang}")
        extra = cel["prompt_tokens_total_medio"] - base["prompt_tokens_total_medio"] if base else None
        ganho = 100 * (cel["ex"] - base["ex"]) if base else None
        custo["por_celula"][chave] = {
            "prompt_tokens_total_medio": cel["prompt_tokens_total_medio"],
            "completion_tokens_total_medio": cel["completion_tokens_total_medio"],
            "tokens_entrada_adicionais": extra,
            "ganho_ex_pp_sobre_c0": ganho,
            "ganho_pp_por_100_tokens": (100 * ganho / extra) if (extra and cond != "C0") else None,
            "chamadas_ao_modelo": 2 if cond == "C4" else 1,
        }
    if custo_rows:
        from .custo import resumir

        custo["latencia"] = resumir(custo_rows)
        custo["latencia_protocolo"] = {
            "n_medicoes": len(custo_rows),
            "split": custo_rows[0].get("split"),
            "lote": 1,
            "gpu": (custo_rows[0].get("ambiente") or {}).get("gpu"),
        }
    saida["custo"] = custo

    # ---- 7. sobreposição das demonstrações que entraram no prompt
    sobre: dict[str, Any] = {}
    for lang in dados.idiomas:
        for a, b in (("C2", "C3"), ("C3", "C4"), ("C2", "C4"), ("C1", "C2")):
            if dados.tem(a, lang) and dados.tem(b, lang):
                la, lb = dados.linhas(a, lang), dados.linhas(b, lang)
                pares = [(x.get("prompt_demo_ids") or x.get("selected_demo_ids") or [],
                          y.get("prompt_demo_ids") or y.get("selected_demo_ids") or []) for x, y in zip(la, lb)]
                sobre[f"{a}-{b}|{lang}"] = {
                    "jaccard_medio": float(np.mean([_jaccard(x, y) for x, y in pares])),
                    "mesmo_conjunto": float(np.mean([set(x) == set(y) for x, y in pares])),
                    "mesma_ordem": float(np.mean([list(x) == list(y) for x, y in pares])),
                }
    saida["sobreposicao_demonstracoes"] = sobre

    # ---- 8. rubrica automática de erros
    erros: dict[str, Any] = {}
    for cond in dados.condicoes:
        for lang in dados.idiomas:
            if dados.tem(cond, lang):
                cont = Counter(c for c in (classificar_erro(r) for r in dados.linhas(cond, lang)) if c)
                erros[f"{cond}|{lang}"] = {k: cont.get(k, 0) for k in RUBRICA} | {"total": sum(cont.values())}
    saida["erros"] = {"rubrica": RUBRICA, "por_celula": erros}

    # ---- 9. por banco (heterogeneidade, descritivo)
    por_banco: dict[str, Any] = {}
    for banco in sorted(set(dados.banco)):
        m = [i for i, b in enumerate(dados.banco) if b == banco]
        por_banco[banco] = {"n": len(m)}
        for cond in dados.condicoes:
            for lang in dados.idiomas:
                if dados.tem(cond, lang):
                    v = dados.vetor(cond, lang)
                    por_banco[banco][f"{cond}|{lang}"] = sum(v[i] for i in m) / len(m)
    saida["por_banco"] = por_banco

    # ---- 10. sensibilidade: avaliador que preserva DISTINCT
    if ex_distinct:
        sens = {}
        for cond in dados.condicoes:
            for lang in dados.idiomas:
                if dados.tem(cond, lang):
                    v = [ex_distinct.get((cond, lang, i)) for i in dados.ids]
                    validos = [x for x in v if x is not None]
                    padrao = dados.vetor(cond, lang)
                    sens[f"{cond}|{lang}"] = {
                        "ex_keep_distinct": sum(validos) / len(validos) if validos else None,
                        "n": len(validos),
                        "vereditos_que_mudam": sum(1 for x, y in zip(v, padrao) if x is not None and x != y),
                    }
        saida["sensibilidade_distinct"] = sens

    # ---- 11. poder do teste primário, com a discordância observada aqui
    if split == "val":
        poder = {}
        for chave in ("C3-C2|pt", "C4-C2|pt", "C4-C3|pt"):
            if chave not in contrastes:
                continue
            taxa = contrastes[chave]["taxa_discordancia"]
            ic_taxa = _wilson(contrastes[chave]["discordantes"], contrastes[chave]["mcnemar"]["n"])
            bloco = {"taxa_discordancia_observada": taxa, "ic95_taxa_wilson": ic_taxa, "n_teste": n_teste}
            for alfa in (0.05, 0.025):
                bloco[f"alfa_{alfa}"] = {
                    "dmd_80": diferenca_minima_detectavel(n_teste, taxa, alfa, 0.80),
                    "dmd_80_taxa_ic_inferior": diferenca_minima_detectavel(n_teste, ic_taxa[0], alfa, 0.80),
                    "dmd_80_taxa_ic_superior": diferenca_minima_detectavel(n_teste, ic_taxa[1], alfa, 0.80),
                    "poder_para_diferenca_observada": (
                        poder_mcnemar(n_teste, taxa, abs(contrastes[chave]["diferenca_pp"]) / 100, alfa)
                        if 0 < abs(contrastes[chave]["diferenca_pp"]) / 100 <= taxa else None),
                }
            poder[chave] = bloco
        saida["poder"] = poder

    # ---- 12. vetores por item, para figuras e conferência independente
    saida["itens"] = {
        "example_id": dados.ids,
        "db_id": dados.banco,
        "dificuldade_oficial": dados.dificuldade,
        "ex_correct": {f"{c}|{l}": "".join("1" if x else "0" for x in dados.vetor(c, l))
                       for c in dados.condicoes for l in dados.idiomas if dados.tem(c, l)},
        "structural_hit": {f"{c}|{l}": "".join("1" if x else "0" for x in dados.vetor(c, l, "structural_hit"))
                           for c in dados.condicoes for l in dados.idiomas if dados.tem(c, l)},
    }
    if estratos:
        itens = estratos.get("itens", estratos)
        saida["itens"]["estrato_literal"] = [itens.get(i, {}).get("estrato") for i in dados.ids]
    return saida


def _wilson(k: int, n: int, z: float = 1.959964) -> list[float]:
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centro - meia), min(1.0, centro + meia)]
