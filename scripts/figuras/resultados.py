"""F06, F07 e F08 — figuras dos resultados, a partir do JSON de `tcc-rag analise`.

Uso: python scripts/figuras/resultados.py <analise-<split>-<hash>.json> <dir_saida> [sufixo]

Nenhum número é digitado aqui: tudo vem do arquivo de análise.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import estilo  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

CONDS = ("C0", "C1", "C2", "C3", "C4")


def _sinal(v: float) -> str:
    return ("+" if v >= 0 else "−") + estilo.pct(abs(v))


def _wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    p = k / n
    den = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / den
    meia = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return 100 * (centro - meia), 100 * (centro + meia)


def f06_efeitos(a: dict, saida: Path, sufixo: str) -> None:
    """Diferenças pareadas de acurácia de execução, com IC de 95%."""
    linhas = [
        ("C3-C2", "C3 − C2", "comparação primária"),
        ("C4-C2", "C4 − C2", "hipótese H1"),
        ("C4-C3", "C4 − C3", "reordenação estrutural"),
        ("C2-C1", "C2 − C1", "recuperação"),
        ("C1-C0", "C1 − C0", "exemplos do domínio"),
    ]
    did = a.get("diferencas_em_diferencas") or {}
    extras = [(k, r1, r2) for k, r1, r2 in (
        ("h2_original_c4_c0", "Δ com C4 e C0", "hipótese H2"),
        ("exploratoria_c3_c0", "Δ com C3 e C0", "exploratória"),
    ) if k in did]
    n_linhas = len(linhas) + len(extras)
    fig, ax = plt.subplots(figsize=(estilo.LARGURA, 0.42 * n_linhas + 1.25))
    ys, rotulos = [], []
    y = 0.0
    lim = 1.0
    for chave, nome, papel in linhas:
        for lang, dy, cor, marc, cheio in (("pt", 0.14, estilo.AZUL, "o", True), ("en", -0.14, estilo.LARANJA, "s", False)):
            c = a["contrastes"].get(f"{chave}|{lang}")
            if not c:
                continue
            lo, hi = c["ic95"]
            lim = max(lim, abs(lo), abs(hi))
            ax.plot([lo, hi], [y + dy, y + dy], color=cor, linewidth=1.6, solid_capstyle="round", zorder=2)
            ax.plot([c["diferenca_pp"]], [y + dy], marker=marc, markersize=6.5, color=cor,
                    markerfacecolor=cor if cheio else "#ffffff", markeredgewidth=1.4, zorder=3, linestyle="none")
        ys.append(y)
        rotulos.append(f"{nome}\n{papel}")
        y -= 1.0
    if extras:
        y -= 0.35
        for chave, nome, papel in extras:
            d = did[chave]
            lim = max(lim, abs(d["low"]), abs(d["high"]))
            ax.plot([d["low"], d["high"]], [y, y], color=estilo.TINTA_2, linewidth=1.6, solid_capstyle="round", zorder=2)
            ax.plot([d["delta_pp"]], [y], marker="D", markersize=6, color=estilo.TINTA_2, zorder=3, linestyle="none")
            ys.append(y)
            rotulos.append(f"{nome}\n{papel}")
            y -= 1.0
    lim = math.ceil(lim / 2.0) * 2.0 + 1
    ax.axvline(0, color=estilo.TINTA_3, linewidth=0.8, zorder=1)
    ax.set_xlim(-lim, lim * 1.9)
    # rótulos numéricos à direita: PT (e EN) com o intervalo
    x_txt = lim * 1.05
    y = 0.0
    for chave, _, _ in linhas:
        partes = []
        for lang in ("pt", "en"):
            c = a["contrastes"].get(f"{chave}|{lang}")
            if c:
                partes.append(f"{lang.upper()}  {_sinal(c['diferenca_pp'])}  [{_sinal(c['ic95'][0])}; {_sinal(c['ic95'][1])}]")
        ax.text(x_txt, y, "\n".join(partes), ha="left", va="center", fontsize=7.6, color=estilo.TINTA, linespacing=1.45)
        y -= 1.0
    if extras:
        y -= 0.35
        for chave, _, _ in extras:
            d = did[chave]
            ax.text(x_txt, y, f"{_sinal(d['delta_pp'])}  [{_sinal(d['low'])}; {_sinal(d['high'])}]", ha="left",
                    va="center", fontsize=7.6)
            y -= 1.0
    ax.set_yticks(ys)
    ax.set_yticklabels(rotulos, linespacing=1.15)
    for rot in ax.get_yticklabels():
        rot.set_fontsize(8)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(ys[-1] - 0.6, 0.6)
    passo = 2 if lim <= 9 else 5
    maximo = int(lim // passo) * passo
    ticks = list(range(-maximo, maximo + 1, passo))
    ax.set_xticks(ticks)
    ax.set_xticklabels([("+" if t > 0 else "−" if t < 0 else "") + str(abs(t)) for t in ticks])
    ax.set_xlabel("Diferença de acurácia de execução (pontos percentuais) e IC de 95%", loc="left")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    marcas = [
        plt.Line2D([], [], color=estilo.AZUL, marker="o", markersize=6, linewidth=1.6, label="Português brasileiro"),
        plt.Line2D([], [], color=estilo.LARANJA, marker="s", markersize=6, markerfacecolor="#ffffff",
                   markeredgewidth=1.4, linewidth=1.6, label="Inglês"),
    ]
    if extras:
        marcas.append(plt.Line2D([], [], color=estilo.TINTA_2, marker="D", markersize=5.5, linewidth=1.6,
                                 label="Diferenças em diferenças (PT − EN)"))
    ax.legend(handles=marcas, ncol=len(marcas), loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False,
              columnspacing=1.6, handlelength=1.8, borderaxespad=0.2)
    estilo.salvar(fig, saida, f"efeitos_pareados{sufixo}")


def f07_custo(a: dict, saida: Path, sufixo: str) -> None:
    """Acurácia × tokens de entrada (PT) e latência por pergunta."""
    lat = (a.get("custo") or {}).get("latencia") or {}
    tem_lat = any(f"{c}|pt" in lat for c in CONDS)
    fig, eixos = plt.subplots(1, 2 if tem_lat else 1, figsize=(estilo.LARGURA, 2.9),
                              gridspec_kw={"width_ratios": [1.25, 1]} if tem_lat else None)
    ax = eixos[0] if tem_lat else eixos
    xs, ys = [], []
    brutos = {c: a["celulas"][f"{c}|pt"]["prompt_tokens_total_medio"] for c in CONDS if f"{c}|pt" in a["celulas"]}
    largura_x = (max(brutos.values()) - min(brutos.values())) or 1
    # condições com praticamente o mesmo custo são afastadas na horizontal só o
    # bastante para os intervalos não se sobreporem; o rótulo traz o valor exato
    desvio = {c: 0.0 for c in brutos}
    ordenadas = sorted(brutos, key=brutos.get)
    for i, j in zip(ordenadas, ordenadas[1:]):
        if brutos[j] - brutos[i] < 0.03 * largura_x:
            desvio[i] -= 0.012 * largura_x
            desvio[j] += 0.012 * largura_x
    for c in CONDS:
        cel = a["celulas"].get(f"{c}|pt")
        if not cel:
            continue
        x, y = cel["prompt_tokens_total_medio"] + desvio[c], 100 * cel["ex"]
        lo, hi = _wilson(cel["acertos"], cel["n"])
        ax.plot([x, x], [lo, hi], color=estilo.COR_CONDICAO[c], linewidth=1.2, zorder=2)
        ax.plot([x], [y], marker=estilo.MARCADOR[c], markersize=7.5, color=estilo.COR_CONDICAO[c],
                markeredgecolor="#ffffff", markeredgewidth=1.0, zorder=3, linestyle="none")
        xs.append(x)
        ys.append((lo, y, hi))
    # rótulos diretos, afastando os que colidem
    usados: list[tuple[float, float]] = []
    span_x = max(xs) - min(xs) or 1
    for c in CONDS:
        cel = a["celulas"].get(f"{c}|pt")
        if not cel:
            continue
        x, y = cel["prompt_tokens_total_medio"] + desvio[c], 100 * cel["ex"]
        dx, ha = 0.035 * span_x, "left"
        yy = y
        while any(abs(x - ux) < 0.22 * span_x and abs(yy - uy) < 1.3 for ux, uy in usados):
            yy -= 1.3
        usados.append((x, yy))
        ax.text(x + dx, yy, f"{c}  {estilo.pct(y)}%", ha=ha, va="center", fontsize=8, color=estilo.TINTA)
    ax.set_xlabel("Tokens de entrada por pergunta (média)")
    ax.set_ylabel("Acurácia de execução (%)")
    ax.set_xlim(min(xs) - 0.12 * span_x, max(xs) + 0.34 * span_x)
    lo_all, hi_all = min(v[0] for v in ys), max(v[2] for v in ys)
    ax.set_ylim(math.floor(lo_all / 5) * 5 - 1, math.ceil(hi_all / 5) * 5 + 1)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_title("(a) Acurácia e tokens de entrada", loc="left", color=estilo.TINTA_2)

    if tem_lat:
        bx = eixos[1]
        nomes = [c for c in CONDS if f"{c}|pt" in lat]
        for i, c in enumerate(nomes):
            v = lat[f"{c}|pt"]
            p50, p95 = v["total_ms_p50"] / 1000, v["total_ms_p95"] / 1000
            bx.plot([p50, p95], [i, i], color=estilo.COR_CONDICAO[c], linewidth=1.6, solid_capstyle="round", zorder=2)
            bx.plot([p50], [i], marker=estilo.MARCADOR[c], markersize=7.5, color=estilo.COR_CONDICAO[c],
                    markeredgecolor="#ffffff", markeredgewidth=1.0, zorder=3, linestyle="none")
            bx.plot([p95], [i], marker="|", markersize=8, color=estilo.COR_CONDICAO[c], zorder=3, linestyle="none")
            bx.text(p95 + 0.02 * max(lat[f"{k}|pt"]["total_ms_p95"] for k in nomes) / 1000, i,
                    f"{estilo.pct(p50)} s", ha="left", va="center", fontsize=8)
        bx.set_yticks(range(len(nomes)))
        bx.set_yticklabels(nomes)
        bx.invert_yaxis()
        bx.set_xlim(0, 1.22 * max(lat[f"{k}|pt"]["total_ms_p95"] for k in nomes) / 1000)
        bx.set_xlabel("Segundos por pergunta (mediana e percentil 95)")
        bx.xaxis.grid(True, zorder=0)
        bx.set_axisbelow(True)
        bx.tick_params(axis="y", length=0)
        bx.set_title("(b) Latência fim a fim", loc="left", color=estilo.TINTA_2)
    fig.tight_layout(w_pad=1.6)
    estilo.salvar(fig, saida, f"acuracia_custo{sufixo}")


def f08_h3(a: dict, saida: Path, sufixo: str) -> None:
    """Acurácia com e sem acerto estrutural, por condição (PT)."""
    conds = [c for c in ("C1", "C2", "C3", "C4") if f"{c}|pt" in a["h3"]]
    fig, ax = plt.subplots(figsize=(estilo.LARGURA, 0.55 * len(conds) + 1.2))
    for i, c in enumerate(conds):
        h = a["h3"][f"{c}|pt"]
        t = h["tabela"]
        n_com, n_sem = t["hit_correta"] + t["hit_incorreta"], t["sem_hit_correta"] + t["sem_hit_incorreta"]
        com = 100 * h["ex_com_acerto_estrutural"] if h["ex_com_acerto_estrutural"] is not None else None
        sem = 100 * h["ex_sem_acerto_estrutural"] if h["ex_sem_acerto_estrutural"] is not None else None
        if com is not None and sem is not None:
            ax.plot([sem, com], [i, i], color=estilo.CINZA_MEDIO, linewidth=2.2, zorder=1, solid_capstyle="round")
        if sem is not None:
            ax.plot([sem], [i], marker="o", markersize=7.5, markerfacecolor="#ffffff", markeredgecolor=estilo.TINTA_2,
                    markeredgewidth=1.5, zorder=3, linestyle="none")
        if com is not None:
            ax.plot([com], [i], marker="o", markersize=7.5, color=estilo.AZUL, markeredgecolor="#ffffff",
                    markeredgewidth=1.0, zorder=3, linestyle="none")
        ic = h.get("ic95")
        texto = (f"{_sinal(h['diferenca_pp'])} pp  [{_sinal(ic[0])}; {_sinal(ic[1])}]" if h.get("diferenca_pp") is not None
                 and ic else "—")
        ax.text(103, i, f"{texto}\ncom: n = {n_com} · sem: n = {n_sem}", ha="left", va="center", fontsize=7.6,
                linespacing=1.4)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels(conds)
    ax.invert_yaxis()
    ax.set_ylim(len(conds) - 0.5, -0.5)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Acurácia de execução (%)", loc="left")
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    marcas = [
        plt.Line2D([], [], color=estilo.AZUL, marker="o", markersize=7, linestyle="none",
                   label="Com acerto estrutural"),
        plt.Line2D([], [], markerfacecolor="#ffffff", markeredgecolor=estilo.TINTA_2, markeredgewidth=1.5, marker="o",
                   markersize=7, linestyle="none", label="Sem acerto estrutural"),
    ]
    ax.legend(handles=marcas, ncol=2, loc="lower left", bbox_to_anchor=(0.0, 1.0), frameon=False,
              columnspacing=1.6, borderaxespad=0.2, handletextpad=0.3)
    ax.text(103, -0.95, "Diferença e IC de 95%", ha="left", va="center", fontsize=7.6, color=estilo.TINTA_2,
            fontweight="bold")
    fig.subplots_adjust(right=0.60)
    estilo.salvar(fig, saida, f"acerto_estrutural_h3{sufixo}")


def main(entrada: Path, saida: Path, sufixo: str = "") -> None:
    a = json.loads(entrada.read_text(encoding="utf-8"))
    estilo.aplicar()
    f06_efeitos(a, saida, sufixo)
    f07_custo(a, saida, sufixo)
    f08_h3(a, saida, sufixo)
    print("figuras em", saida)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "")
