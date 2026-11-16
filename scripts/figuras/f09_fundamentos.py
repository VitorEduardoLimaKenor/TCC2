"""F09 — Figuras da fundamentação teórica (diagramas, sem dados do experimento).

Gera três figuras do Capítulo 2:
  esquema_rag             fluxo geral de um sistema RAG
  busca_vetorial          consulta e seus k vizinhos mais próximos (esquema em 2D)
  exemplo_text_to_sql     pergunta, esquema e consulta de um item do Spider

Uso: python scripts/figuras/f09_fundamentos.py <dir_saida>

O exemplo de Text-to-SQL é o item `spider_train_spider_3289` do manifesto
(banco `college_1`): a pergunta em português é a da tradução de José e Cozman
(2021), a consulta é a de referência do Spider e o resultado (5) é o da sua
execução sobre o banco original.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import estilo  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch  # noqa: E402

BORDA = estilo.TINTA_3
REALCE = "#fbe3d8"
MONO = ["Liberation Mono", "DejaVu Sans Mono", "monospace"]


def caixa(ax, x, y, w, h, texto="", cor="#ffffff", borda=BORDA, fs=8.0, negrito=False, lw=0.8, ha="center",
          mono=False, cor_texto=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                facecolor=cor, edgecolor=borda, linewidth=lw, zorder=2))
    if texto:
        tx = x + w / 2 if ha == "center" else x + 1.6
        ax.text(tx, y + h / 2, texto, ha=ha, va="center", fontsize=fs, zorder=3,
                fontweight="bold" if negrito else "normal", color=cor_texto or estilo.TINTA, linespacing=1.3,
                family=MONO if mono else None)


def seta(ax, x0, y0, x1, y1, cor=estilo.TINTA_2, estilo_seta="-|>"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=estilo_seta, mutation_scale=8,
                                 linewidth=0.9, color=cor, zorder=4, shrinkA=0, shrinkB=0))


def rotulo(ax, x, y, texto, fs=7.0, ha="center", va="center"):
    ax.text(x, y, texto, ha=ha, va=va, fontsize=fs, color=estilo.TINTA_2, linespacing=1.2, zorder=5)


def esquema_rag(saida: Path) -> None:
    fig, ax = plt.subplots(figsize=(estilo.LARGURA, 2.55))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")
    y, h = 25, 13
    caixa(ax, 0, y, 15, h, "Entrada\ndo usuário", negrito=True)
    caixa(ax, 21, y, 17, h, "Recuperador", cor=estilo.CINZA_CLARO, negrito=True)
    caixa(ax, 44, y - 3, 25, h + 6, "", cor="#ffffff")
    ax.text(56.5, y + h + 0.6, "Prompt", ha="center", va="top", fontsize=8.0, fontweight="bold")
    for i, t in enumerate(("1. Instrução", "2. Itens recuperados", "3. Entrada do usuário")):
        caixa(ax, 46, y + 6.6 - i * 4.7, 21, 3.9, t, cor=REALCE if i == 1 else "#ffffff",
              borda=estilo.LARANJA if i == 1 else BORDA, fs=7.0, lw=1.1 if i == 1 else 0.7)
    caixa(ax, 74, y, 14, h, "Modelo de\nlinguagem", cor=estilo.CINZA_CLARO, negrito=True, fs=7.6)
    caixa(ax, 92, y, 8, h, "Saída", negrito=True)
    seta(ax, 15, y + h / 2, 21, y + h / 2)
    seta(ax, 38, y + h / 2, 44, y + h / 2)
    seta(ax, 69, y + h / 2, 74, y + h / 2)
    seta(ax, 88, y + h / 2, 92, y + h / 2)
    # base externa
    caixa(ax, 17.5, 1.5, 24, 12.5, "Base externa indexada\n(documentos, esquema\nou exemplos resolvidos)", fs=7.2)
    seta(ax, 26.5, y, 26.5, 14.0)
    seta(ax, 32.5, 14.0, 32.5, y)
    rotulo(ax, 25.6, 19.5, "consulta", ha="right")
    rotulo(ax, 33.4, 19.5, "itens mais\npróximos", ha="left")
    # caminho da entrada original até o prompt (por cima)
    ax.plot([7.5, 7.5, 56.5], [y + h, 42.0, 42.0], color=estilo.TINTA_2, linewidth=0.9, zorder=1)
    seta(ax, 56.5, 42.0, 56.5, y + h + 3.2)
    estilo.salvar(fig, saida, "esquema_rag")


def busca_vetorial(saida: Path) -> None:
    rng = np.random.default_rng(20260904)
    centros = np.array([[2.3, 6.9], [7.3, 7.4], [6.6, 2.5], [2.5, 2.6]])
    fundo = np.clip(np.vstack([c + rng.normal(0, 1.05, size=(10, 2)) for c in centros]), 0.4, 9.6)
    consulta = np.array([6.3, 3.4])
    # Posições desenhadas (não são dados): quatro vizinhos em torno da consulta e o
    # restante dos itens fora do raio, para que a figura seja legível.
    vizinhos = consulta + np.array([[0.78, 0.36], [-0.55, 0.68], [-0.82, -0.44], [0.33, -0.88]])
    fundo = fundo[np.linalg.norm(fundo - consulta, axis=1) > 1.6]
    pontos = np.vstack([fundo, vizinhos])
    dist = np.linalg.norm(pontos - consulta, axis=1)
    k = 4
    viz = np.argsort(dist)[:k]
    resto = np.setdiff1d(np.arange(len(pontos)), viz)

    fig, ax = plt.subplots(figsize=(estilo.LARGURA * 0.72, 3.25))
    ax.scatter(pontos[resto, 0], pontos[resto, 1], s=22, facecolor="#ffffff", edgecolor=estilo.TINTA_3,
               linewidth=0.9, zorder=2, label="Itens indexados")
    for i in viz:
        ax.plot([consulta[0], pontos[i, 0]], [consulta[1], pontos[i, 1]], color=estilo.AZUL, linewidth=0.8,
                linestyle=(0, (3, 2)), zorder=1)
    ax.scatter(pontos[viz, 0], pontos[viz, 1], s=42, facecolor=estilo.AZUL, edgecolor="#ffffff", linewidth=0.8,
               zorder=3, label=f"{k} vizinhos mais próximos")
    ax.scatter([consulta[0]], [consulta[1]], s=120, marker="*", facecolor=estilo.LARANJA, edgecolor="#ffffff",
               linewidth=0.7, zorder=4, label="Vetor de consulta")
    ax.add_patch(Circle(consulta, dist[viz[-1]] + 0.12, facecolor="none", edgecolor=estilo.TINTA_3,
                        linewidth=0.7, linestyle=(0, (1, 2)), zorder=1))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Dimensão 1")
    ax.set_ylabel("Dimensão 2")
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, handletextpad=0.4, borderaxespad=0)
    estilo.salvar(fig, saida, "busca_vetorial")


def exemplo_text_to_sql(saida: Path) -> None:
    fig, ax = plt.subplots(figsize=(estilo.LARGURA, 2.75))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 48)
    ax.axis("off")
    # coluna esquerda: pergunta e esquema
    ax.text(0, 47.2, "Pergunta", fontsize=7.4, fontweight="bold", color=estilo.TINTA_2, va="top")
    caixa(ax, 0, 31.5, 37, 12, "Quantos professores são do\ndepartamento de Contabilidade\nou Biologia?", fs=7.6)
    ax.text(0, 27.2, "Esquema do banco (trecho)", fontsize=7.4, fontweight="bold", color=estilo.TINTA_2, va="top")
    caixa(ax, 0, 2.5, 37, 21,
          "PROFESSOR(EMP_NUM, DEPT_CODE,\n          PROF_OFFICE, ...)\n"
          "DEPARTMENT(DEPT_CODE, DEPT_NAME,\n           SCHOOL_CODE, ...)",
          fs=6.2, ha="left", mono=True, cor=estilo.CINZA_CLARO)
    # modelo
    caixa(ax, 41.5, 18.5, 14.5, 12, "Sistema\nText-to-SQL", cor=REALCE, borda=estilo.LARANJA, negrito=True, fs=7.0,
          lw=1.1)
    seta(ax, 37, 37.5, 41.5, 28.0)
    seta(ax, 37, 13.0, 41.5, 21.0)
    seta(ax, 56, 24.5, 59.5, 24.5)
    # consulta
    ax.text(59.5, 47.2, "Consulta SQL", fontsize=7.4, fontweight="bold", color=estilo.TINTA_2, va="top")
    caixa(ax, 59.5, 16.5, 40.5, 27,
          "SELECT count(*)\nFROM professor AS T1\nJOIN department AS T2\n"
          "  ON T1.dept_code = T2.dept_code\nWHERE T2.dept_name = 'Accounting'\n"
          "   OR T2.dept_name = 'Biology'",
          fs=6.2, ha="left", mono=True)
    seta(ax, 81, 16.5, 81, 11.0)
    rotulo(ax, 82.2, 13.7, "execução no banco", ha="left")
    caixa(ax, 71, 2.5, 20, 8.5, "Resultado: 5", negrito=True, fs=7.8, cor=estilo.CINZA_CLARO)
    estilo.salvar(fig, saida, "exemplo_text_to_sql")


def main(saida: Path) -> None:
    estilo.aplicar()
    esquema_rag(saida)
    busca_vetorial(saida)
    exemplo_text_to_sql(saida)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
