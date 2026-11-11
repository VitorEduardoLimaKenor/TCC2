"""F04 — Perfil de dificuldade oficial: Spider × recorte × partições.

Uso: python scripts/figuras/f04_dificuldade.py <perfil-dificuldade.json> <dir_saida>
Os dados vêm de `scripts/perfil_dificuldade.py`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import estilo  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

NIVEIS = (("easy", "Fácil"), ("medium", "Médio"), ("hard", "Difícil"), ("extra", "Extra"))
GRUPOS = (
    ("spider_train", "Spider – treinamento"),
    ("spider_dev", "Spider – desenvolvimento"),
    ("recorte", "Recorte (17 bancos)"),
    ("recorte_pool", "· Base de demonstrações"),
    ("recorte_val", "· Validação"),
    ("recorte_test", "· Teste"),
)


def main(entrada: Path, saida: Path) -> None:
    dados = json.loads(entrada.read_text(encoding="utf-8"))["grupos"]
    estilo.aplicar()
    fig, ax = plt.subplots(figsize=(estilo.LARGURA, 2.75))
    ys = [0, 1, 2.35, 3.35, 4.35, 5.35]
    for y, (chave, rotulo) in zip(ys, GRUPOS):
        g = dados[chave]
        total = sum(g[n] for n, _ in NIVEIS)
        esquerda = 0.0
        for (nivel, _), cor in zip(NIVEIS, estilo.RAMPA):
            largura = 100 * g[nivel] / total
            ax.barh(y, largura - 0.5, left=esquerda + 0.25, height=0.66, color=cor, linewidth=0)
            clara = cor in estilo.RAMPA[:1]
            ax.text(esquerda + largura / 2, y, f"{estilo.pct(largura)}%", ha="center", va="center",
                    fontsize=8, color=estilo.TINTA if clara else "#ffffff")
            esquerda += largura
        ax.text(101.2, y, f"n = {total:,}".replace(",", "."), ha="left", va="center",
                fontsize=8, color=estilo.TINTA_2)
    ax.set_yticks(ys)
    ax.set_yticklabels([r for _, r in GRUPOS])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Proporção das perguntas por nível de dificuldade da consulta de referência")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    marcas = [plt.Rectangle((0, 0), 1, 1, color=c) for c in estilo.RAMPA]
    ax.legend(marcas, [r for _, r in NIVEIS], ncol=4, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              frameon=False, handlelength=1.1, handleheight=0.9, columnspacing=1.6, borderaxespad=0.2)
    for caminho in estilo.salvar(fig, saida, "perfil_dificuldade"):
        print(caminho)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
