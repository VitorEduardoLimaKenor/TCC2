"""F01 — O que muda entre as cinco condições (diagrama, sem dados).

Uso: python scripts/figuras/f01_arquitetura.py <dir_saida>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import estilo  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

BORDA = estilo.TINTA_3
EXTRA = "#fbe3d8"  # fundo claro do laranja: a chamada adicional de C4


def caixa(ax, x, y, w, h, texto, cor="#ffffff", borda=BORDA, fs=7.6, negrito=False, lw=0.8, cor_texto=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                facecolor=cor, edgecolor=borda, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, texto, ha="center", va="center", fontsize=fs, zorder=3,
            fontweight="bold" if negrito else "normal", color=cor_texto or estilo.TINTA, linespacing=1.25)


def seta(ax, x0, y0, x1, y1, cor=estilo.TINTA_2, curva=0.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7,
                                 linewidth=0.8, color=cor, zorder=4,
                                 connectionstyle=f"arc3,rad={curva}", shrinkA=0, shrinkB=0))


def main(saida: Path) -> None:
    estilo.aplicar()
    fig, ax = plt.subplots(figsize=(estilo.LARGURA, 4.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(-7, 74)
    ax.axis("off")

    x_txt, w_txt = 10.5, 23.5
    x_sel, w_sel = 38.0, 30.0
    px, pw = 76.0, 24.0

    # ---- cabeçalhos das colunas
    for x, t in ((4.5, "Condição"), (x_txt + w_txt / 2, "Texto vetorizado\nna busca"),
                 (x_sel + w_sel / 2, "Seleção das\ndemonstrações"), (px + pw / 2, "Comum a todas\nas condições")):
        ax.text(x, 73.2, t, ha="center", va="top", fontsize=7.6, color=estilo.TINTA_2, fontweight="bold",
                linespacing=1.2)

    linhas = {"C0": 61.0, "C1": 51.5, "C2": 42.0, "C3": 32.5, "C4": 15.5}
    papel = {"C0": "ponto de\npartida", "C1": "controle", "C2": "comparação\nprincipal",
             "C3": "método\nprincipal", "C4": "análise de\ncomponente"}
    for c, y in linhas.items():
        ax.text(4.5, y + 1.3, c, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(4.5, y - 2.5, papel[c], ha="center", va="center", fontsize=6.3, color=estilo.TINTA_2,
                linespacing=1.15)

    # ---- bloco comum: prompt -> modelo -> SQL
    topo, base = 66.5, 28.0
    caixa(ax, px, base, pw, topo - base, "", cor=estilo.CINZA_CLARO, borda=BORDA)
    ax.text(px + pw / 2, topo - 2.3, "Prompt", ha="center", va="center", fontsize=8.0, fontweight="bold")
    itens = (("1. Instrução", 5.0), ("2. Esquema completo\n(DDL), sem poda", 7.4),
             ("3. Demonstrações\n(nenhuma em C0;\n$k$ em C1–C4)", 10.0), ("4. Pergunta original", 5.0))
    y = topo - 4.8
    centro_demo = None
    for i, (t, h) in enumerate(itens):
        y -= h
        caixa(ax, px + 1.6, y, pw - 3.2, h, t, fs=6.8, cor="#ffffff",
              borda=estilo.LARANJA if i == 2 else BORDA, lw=1.2 if i == 2 else 0.8)
        if i == 2:
            centro_demo = y + h / 2
        y -= 1.3
    caixa(ax, px, 14.5, pw, 9.5, "Modelo de linguagem\n(mesmos pesos e\ndecodificação gulosa)", fs=6.8)
    caixa(ax, px, 4.0, pw, 6.0, "Consulta SQL gerada", fs=7.0, negrito=True)
    seta(ax, px + pw / 2, base, px + pw / 2, 24.0)
    seta(ax, px + pw / 2, 14.5, px + pw / 2, 10.0)

    traco = dict(ha="center", va="center", color=estilo.TINTA_3, fontsize=9)
    # C0
    y = linhas["C0"]
    ax.text(x_txt + w_txt / 2, y, "—", **traco)
    caixa(ax, x_sel, y - 3, w_sel, 6, "Nenhuma demonstração", fs=7.0, borda=estilo.CINZA_MEDIO,
          cor_texto=estilo.TINTA_2)
    # C1
    y = linhas["C1"]
    ax.text(x_txt + w_txt / 2, y, "—", **traco)
    caixa(ax, x_sel, y - 3.2, w_sel, 6.4, "$k$ exemplos fixos do banco,\nsorteados uma única vez", fs=6.8)
    # C2 e C3
    for c, texto in (("C2", "Pergunta original"), ("C3", "Pergunta delexicalizada")):
        y = linhas[c]
        caixa(ax, x_txt, y - 3, w_txt, 6, texto, fs=6.9)
        caixa(ax, x_sel, y - 3.2, w_sel, 6.4, "$k$ vizinhos mais próximos\n(similaridade do cosseno)", fs=6.8)
        seta(ax, x_txt + w_txt, y, x_sel, y)
    # C4: duas etapas
    y = linhas["C4"]
    caixa(ax, x_txt, y + 4.0, w_txt, 6, "Pergunta delexicalizada", fs=6.9)
    caixa(ax, x_sel, y + 3.8, w_sel, 6.4, "1ª etapa: $N$ vizinhos mais próximos\n(similaridade do cosseno)", fs=6.5)
    seta(ax, x_txt + w_txt, y + 7.0, x_sel, y + 7.0)
    caixa(ax, x_txt, y - 11.5, w_txt, 11.5, "Consulta preliminar:\nchamada adicional\nao modelo, sem\ndemonstrações",
          cor=EXTRA, borda=estilo.LARANJA, lw=1.2, fs=6.8)
    caixa(ax, x_sel, y - 11.5, w_sel, 11.5,
          "2ª etapa: reordenação pela\nassinatura estrutural da\nconsulta preliminar,\n"
          "$\\lambda\\,\\mathrm{sim}_{perg} + (1-\\lambda)\\,\\mathrm{sim}_{estr}$;\nficam os $k$ melhores", fs=6.5)
    seta(ax, x_txt + w_txt, y - 5.75, x_sel, y - 5.75)
    seta(ax, x_sel + w_sel / 2, y + 3.8, x_sel + w_sel / 2, y)

    # ---- das seleções para o bloco "Demonstrações" do prompt, por um tronco comum
    xb = px + 1.6
    xt = (x_sel + w_sel + px) / 2
    saidas = {"C1": linhas["C1"], "C2": linhas["C2"], "C3": linhas["C3"], "C4": linhas["C4"] - 5.75}
    for ys in saidas.values():
        ax.plot([x_sel + w_sel, xt], [ys, ys], color=estilo.TINTA_2, linewidth=0.8, zorder=4, solid_capstyle="round")
    ax.plot([xt, xt], [min(saidas.values()), max(saidas.values())], color=estilo.TINTA_2, linewidth=0.8, zorder=4,
            solid_capstyle="round")
    seta(ax, xt, centro_demo, xb, centro_demo)

    ax.text(0.5, -6.6, "Nas condições C1 a C4, as demonstrações vêm da base de demonstrações do mesmo banco e do mesmo idioma "
            "da pergunta.\nA consulta de referência não participa de nenhuma etapa: só é usada depois, na avaliação por "
            "execução.", ha="left", va="bottom", fontsize=6.4, color=estilo.TINTA_2, linespacing=1.3)

    for caminho in estilo.salvar(fig, saida, "arquitetura_condicoes"):
        print(caminho)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
