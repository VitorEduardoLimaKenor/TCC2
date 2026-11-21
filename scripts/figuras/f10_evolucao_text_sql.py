"""F10 — Evolução da acurácia de execução no Spider (Figura 1, Capítulo 1).

Redesenho, em português, da Figura 2 de Shi et al. (2025), "The evolution of
Text-to-SQL approach over time": acurácia de execução, no conjunto de teste do
Spider, das abordagens que lideraram o placar, separadas em abordagens baseadas
em modelos de linguagem pré-treinados (PLM) e em LLMs.

Os valores são os do placar oficial do Spider 1.0 ("Leaderboard - Execution with
Values", coluna Test; https://yale-lily.github.io/spider, consultado em
16/11/2026), os mesmos representados na figura original. Os rótulos de data são
os da figura original (ano e mês); no placar, a entrada de SmBoP + GraPPa é de
10/03/2021, e a figura original a posiciona em maio de 2021.

Uso: python scripts/figuras/f10_evolucao_text_sql.py <dir_saida>
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import estilo  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

MES = ("jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez")

# (ano, mês na figura original, método, acurácia de execução no teste, família)
PONTOS = (
    (2020, 5, "BRIDGE + BERT", 59.9, "PLM"),
    (2020, 11, "BRIDGE v2 + BERT", 64.3, "PLM"),
    (2021, 1, "COMBINE", 68.2, "PLM"),
    (2021, 5, "SmBoP + GraPPa", 71.1, "PLM"),
    (2021, 7, "PICARD + T5-3B", 75.1, "PLM"),
    (2022, 6, "PICARD + RASAT + T5-3B", 75.5, "PLM"),
    (2022, 9, "CatSQL + GraPPa", 78.0, "PLM"),
    (2023, 2, "NatSQL + RESDSQL-3B", 79.9, "PLM"),
    (2023, 4, "DIN-SQL + Codex", 78.2, "LLM"),
    (2023, 6, "C3-SQL + ChatGPT\n+ Zero-shot", 82.3, "LLM"),
    (2023, 7, "Hindsight CoT\n+ GPT-4", 83.9, "LLM"),
    (2023, 8, "DAIL-SQL + GPT-4\n+ Self-Consistency", 86.6, "LLM"),
    (2023, 11, "MiniSeek", 91.2, "LLM"),
)

# Posição do rótulo de cada ponto: deslocamento em pontos e alinhamento.
# Acima e à esquerda nos trechos ascendentes; abaixo e à direita quando o
# espaço acima está ocupado.
ACIMA_ESQ = ((-3, 5), "right", "bottom")
ACIMA = ((0, 7), "center", "bottom")
ABAIXO_DIR = ((4, -5), "left", "top")
ABAIXO = ((0, -8), "center", "top")
POSICAO = {
    "BRIDGE + BERT": ABAIXO_DIR,
    "BRIDGE v2 + BERT": ABAIXO_DIR,
    "COMBINE": ACIMA_ESQ,
    "SmBoP + GraPPa": ABAIXO_DIR,
    "PICARD + T5-3B": ACIMA_ESQ,
    "PICARD + RASAT + T5-3B": ABAIXO_DIR,
    "CatSQL + GraPPa": ACIMA_ESQ,
    "NatSQL + RESDSQL-3B": ACIMA,
    "DIN-SQL + Codex": ABAIXO,
    "C3-SQL + ChatGPT\n+ Zero-shot": ACIMA_ESQ,
    "Hindsight CoT\n+ GPT-4": ABAIXO_DIR,
    "DAIL-SQL + GPT-4\n+ Self-Consistency": ACIMA_ESQ,
    "MiniSeek": ACIMA_ESQ,
}

FAMILIA = {
    "PLM": ("Abordagens baseadas em modelos pré-treinados", estilo.LARANJA, "o"),
    "LLM": ("Abordagens baseadas em LLMs", estilo.AZUL, "s"),
}


def evolucao(saida: Path) -> None:
    fig, ax = plt.subplots(figsize=(estilo.LARGURA, 3.3))
    for familia, (rotulo, cor, marcador) in FAMILIA.items():
        idx = [i for i, p in enumerate(PONTOS) if p[4] == familia]
        ax.plot(idx, [PONTOS[i][3] for i in idx], color=cor, linewidth=1.4, marker=marcador, markersize=4.6,
                markeredgecolor="#ffffff", markeredgewidth=0.6, zorder=3, label=rotulo)
    for i, (_, _, metodo, ex, _) in enumerate(PONTOS):
        (dx, dy), ha, va = POSICAO[metodo]
        ax.annotate(metodo, (i, ex), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, fontsize=7.2,
                    color=estilo.TINTA_2, linespacing=1.15, zorder=4)
    ax.set_xticks(range(len(PONTOS)))
    ax.set_xticklabels([f"{MES[m - 1]}\n{a}" for a, m, *_ in PONTOS], fontsize=7.6)
    ax.set_xlim(-0.6, len(PONTOS) - 0.4)
    ax.set_ylim(55, 95)
    ax.set_yticks(range(55, 96, 5))
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_ylabel("Acurácia de execução (%)")
    ax.legend(loc="upper left", frameon=False, handlelength=2.2, borderaxespad=0.2)
    estilo.salvar(fig, saida, "evolucao_text_sql")


def main(saida: Path) -> None:
    estilo.aplicar()
    evolucao(saida)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
