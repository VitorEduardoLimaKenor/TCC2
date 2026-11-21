"""Primitivas dos diagramas da monografia (caixas, setas, rótulos e marcadores de etapa).

Todas as figuras de diagrama usam um sistema de coordenadas próprio (eixos sem escala, em geral de 0 a 100),
a paleta de `estilo` e texto em português. A cor nunca é o único código: há rótulo, borda tracejada ou número.
"""

from __future__ import annotations

import estilo
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle

BORDA = estilo.TINTA_3
REALCE = "#fbe3d8"  # fundo claro do laranja
REALCE_AZUL = "#dde9f8"  # fundo claro do azul
MONO = ["Liberation Mono", "DejaVu Sans Mono", "monospace"]


def caixa(ax, x, y, w, h, texto="", cor="#ffffff", borda=BORDA, fs=7.6, negrito=False, lw=0.8, ha="center",
          mono=False, cor_texto=None, tracejada=False, raio=1.0, va="center", italico=False, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={raio}",
                                facecolor=cor, edgecolor=borda, linewidth=lw, zorder=z,
                                linestyle=(0, (3, 2)) if tracejada else "solid"))
    if texto:
        tx = x + w / 2 if ha == "center" else x + 1.4
        ty = y + h / 2 if va == "center" else y + h - 1.2
        ax.text(tx, ty, texto, ha=ha, va=va, fontsize=fs, zorder=z + 1,
                fontweight="bold" if negrito else "normal", color=cor_texto or estilo.TINTA, linespacing=1.3,
                family=MONO if mono else None, style="italic" if italico else "normal")


def painel(ax, x, y, w, h, titulo="", cor=estilo.CINZA_CLARO, fs=7.8):
    """Área de fundo que agrupa caixas, com título no canto superior esquerdo."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.4",
                                facecolor=cor, edgecolor="none", zorder=0))
    if titulo:
        ax.text(x + 1.6, y + h - 1.4, titulo, ha="left", va="top", fontsize=fs, fontweight="bold",
                color=estilo.TINTA_2, zorder=1)


def seta(ax, x0, y0, x1, y1, cor=estilo.TINTA_2, tracejada=False, curva=0.0, lw=0.9, dupla=False, ms=8):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="<|-|>" if dupla else "-|>", mutation_scale=ms,
                                 linewidth=lw, color=cor, zorder=4, shrinkA=0, shrinkB=0,
                                 connectionstyle=f"arc3,rad={curva}",
                                 linestyle=(0, (3, 2)) if tracejada else "solid"))


def linha(ax, xs, ys, cor=estilo.TINTA_2, tracejada=False, lw=0.9):
    ax.plot(xs, ys, color=cor, linewidth=lw, zorder=1, solid_capstyle="butt",
            linestyle=(0, (3, 2)) if tracejada else "solid")


def rotulo(ax, x, y, texto, fs=7.0, ha="center", va="center", cor=None, negrito=False, italico=False, mono=False):
    ax.text(x, y, texto, ha=ha, va=va, fontsize=fs, color=cor or estilo.TINTA_2, linespacing=1.25, zorder=5,
            fontweight="bold" if negrito else "normal", style="italic" if italico else "normal",
            family=MONO if mono else None)


def etapa(ax, x, y, n, r=1.9, fs=7.0, cor=estilo.TINTA):
    """Marcador circular com o número de uma etapa descrita no texto.

    O círculo é um marcador (tamanho em pontos), e não um patch, para continuar redondo quando as escalas dos
    eixos x e y diferem; `r` define o tamanho do marcador (diâmetro de r × 5,6 pt)."""
    ax.plot([x], [y], marker="o", markersize=r * 5.6, markerfacecolor=cor,
            markeredgecolor="#ffffff", markeredgewidth=0.8, zorder=6, linestyle="none")
    ax.text(x, y, str(n), ha="center", va="center", fontsize=fs, color="#ffffff", fontweight="bold", zorder=7)


def cilindro(ax, x, y, w, h, texto="", fs=7.0, cor="#ffffff"):
    """Símbolo de base de dados."""
    e = h * 0.18
    ax.add_patch(Rectangle((x, y + e / 2), w, h - e, facecolor=cor, edgecolor="none", zorder=2))
    ax.add_patch(Ellipse((x + w / 2, y + e / 2), w, e, facecolor=cor, edgecolor=BORDA, linewidth=0.8, zorder=2))
    ax.plot([x, x], [y + e / 2, y + h - e / 2], color=BORDA, linewidth=0.8, zorder=3)
    ax.plot([x + w, x + w], [y + e / 2, y + h - e / 2], color=BORDA, linewidth=0.8, zorder=3)
    ax.add_patch(Ellipse((x + w / 2, y + h - e / 2), w, e, facecolor=cor, edgecolor=BORDA, linewidth=0.8, zorder=3))
    if texto:
        ax.text(x + w / 2, y + (h - e) / 2, texto, ha="center", va="center", fontsize=fs, zorder=4,
                linespacing=1.2, color=estilo.TINTA)


def eixos(fig_w, fig_h, xlim=(0, 100), ylim=(0, 100)):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.axis("off")
    return fig, ax
