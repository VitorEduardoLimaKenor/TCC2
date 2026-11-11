"""Estilo comum das figuras da monografia.

Figuras para impressão em A4 (largura útil de 16 cm), legíveis em tons de
cinza: a cor nunca é o único código — há rótulo direto, marcador ou hachura.
A paleta é a de referência validada (azul/laranja + rampa ordinal de um matiz).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LARGURA = 6.3  # polegadas = 16 cm, a largura do texto
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_3 = "#8a8983"
GRADE = "#dcdbd6"
SUPERFICIE = "#ffffff"
AZUL = "#2a78d6"
LARANJA = "#eb6834"
AQUA = "#1baf7a"
CINZA_CLARO = "#eeedea"
CINZA_MEDIO = "#c9c8c2"
# Rampa ordinal de um matiz (fácil -> extra), do claro ao escuro.
RAMPA = ("#86b6ef", "#3987e5", "#1c5cab", "#0d366b")
# Condições: cor por entidade (ordem fixa), com marcador próprio.
COR_CONDICAO = {"C0": "#8a8983", "C1": "#eda100", "C2": AZUL, "C3": LARANJA, "C4": AQUA}
MARCADOR = {"C0": "o", "C1": "s", "C2": "D", "C3": "^", "C4": "v"}
NOME_CONDICAO = {
    "C0": "C0 – Sem demonstrações",
    "C1": "C1 – Exemplos fixos",
    "C2": "C2 – Similaridade da pergunta",
    "C3": "C3 – Pergunta delexicalizada",
    "C4": "C4 – Duas etapas",
}


def aplicar() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Nimbus Sans", "DejaVu Sans"],
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.edgecolor": TINTA_3,
        "axes.labelcolor": TINTA_2,
        "axes.linewidth": 0.6,
        "xtick.color": TINTA_2,
        "ytick.color": TINTA_2,
        "text.color": TINTA,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "grid.color": GRADE,
        "grid.linewidth": 0.6,
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "pdf.fonttype": 42,  # texto como fonte TrueType embutida, selecionável
        "ps.fonttype": 42,
        "hatch.linewidth": 0.5,
    })


def salvar(fig, destino: Path, nome: str) -> list[Path]:
    """Grava PDF vetorial (para o LaTeX) e PNG (para conferência visual)."""
    destino.mkdir(parents=True, exist_ok=True)
    saidas = []
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
        caminho = destino / f"{nome}.{ext}"
        fig.savefig(caminho, bbox_inches="tight", pad_inches=0.04, **kw)
        saidas.append(caminho)
    plt.close(fig)
    return saidas


def pct(valor: float, casas: int = 1) -> str:
    """Número com vírgula decimal, como no texto da monografia."""
    return f"{valor:.{casas}f}".replace(".", ",")
