"""F02 — Do Spider ao recorte e às três partições, por família canônica.

As contagens são lidas do manifesto (e conferidas contra os arquivos do Spider);
nada vem de predições nem de resultados.

Uso: python scripts/figuras/f02_particao.py <spider_root> <manifest.json> <dir_saida>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import estilo  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402


def milhar(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def caixa(ax, x, y, w, h, titulo, linhas, cor="#ffffff", borda=estilo.TINTA_3, lw=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
                                facecolor=cor, edgecolor=borda, linewidth=lw, zorder=2))
    ax.text(x + w / 2, y + h - 2.6, titulo, ha="center", va="center", fontsize=8.2, fontweight="bold", zorder=3)
    ax.text(x + w / 2, y + (h - 4.6) / 2, linhas, ha="center", va="center", fontsize=7.2, zorder=3,
            linespacing=1.35, color=estilo.TINTA)


def seta(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, linewidth=0.8,
                                 color=estilo.TINTA_2, zorder=1, shrinkA=0, shrinkB=0))


def main(spider_root: Path, manifesto: Path, saida: Path) -> None:
    treino = json.loads((spider_root / "train_spider.json").read_text(encoding="utf-8"))
    dev = json.loads((spider_root / "dev.json").read_text(encoding="utf-8"))
    n_spider = len(treino) + len(dev)
    bancos_spider = len({x["db_id"] for x in treino + dev})
    exemplos = json.loads(manifesto.read_text(encoding="utf-8"))["examples"]
    familias, perguntas = defaultdict(set), defaultdict(int)
    for e in exemplos:
        familias[e["partition"]].add(e["family_id"])
        perguntas[e["partition"]] += 1
    todas = set().union(*familias.values())
    assert sum(len(f) for f in familias.values()) == len(todas), "família em mais de uma partição"
    bancos = len({e["db_id"] for e in exemplos})
    n = len(exemplos)

    estilo.aplicar()
    fig, ax = plt.subplots(figsize=(estilo.LARGURA, 4.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    caixa(ax, 17, 54, 66, 11, "Spider 1.0 (treinamento e desenvolvimento)",
          f"{bancos_spider} bancos · {milhar(n_spider)} perguntas, cada uma em inglês\ne em português brasileiro, "
          "com a mesma consulta de referência")
    seta(ax, 50, 54, 50, 46.5)
    ax.text(51.5, 50.2, "bancos com pelo menos 90 perguntas e 25 famílias", ha="left", va="center", fontsize=6.9,
            color=estilo.TINTA_2)
    caixa(ax, 17, 35.5, 66, 11, "Recorte de domínio único",
          f"{bancos} bancos · {milhar(n)} perguntas · {len(todas)} famílias canônicas\n"
          "(consultas iguais após normalizar apelidos, formatação e literais)")
    ax.text(51.5, 31.2, "alocação por família, dentro de cada banco;\nos dois idiomas de uma pergunta ficam juntos",
            ha="left", va="center", fontsize=6.9, color=estilo.TINTA_2, linespacing=1.25)

    partes = (
        ("pool", "Base de demonstrações", estilo.CINZA_CLARO, estilo.TINTA_3, 0.8,
         "indexadas nos dois idiomas:\n{d} demonstrações"),
        ("val", "Validação", "#ffffff", estilo.TINTA_3, 0.8,
         "escolha de $k$, $N$ e $\\lambda$;\nredesignação do método"),
        ("test", "Teste", "#fbe3d8", estilo.LARANJA, 1.2,
         "selado até o congelamento\nda configuração"),
    )
    xs = (1.0, 35.0, 69.0)
    largura = 30.0
    ax.plot([50, 50], [35.5, 28.5], color=estilo.TINTA_2, linewidth=0.8, zorder=1)
    ax.plot([xs[0] + largura / 2, xs[2] + largura / 2], [28.5, 28.5], color=estilo.TINTA_2, linewidth=0.8, zorder=1)
    for x, (chave, titulo, cor, borda, lw, extra) in zip(xs, partes):
        seta(ax, x + largura / 2, 28.5, x + largura / 2, 25.0)
        pct = 100 * perguntas[chave] / n
        texto = (f"{milhar(perguntas[chave])} perguntas ({estilo.pct(pct)}%)\n{len(familias[chave])} famílias\n"
                 + extra.format(d=milhar(2 * perguntas[chave])))
        caixa(ax, x, 6.5, largura, 18.5, titulo, texto, cor=cor, borda=borda, lw=lw)

    ax.text(50, 2.2, "Auditoria automática: nenhuma família em mais de uma partição, nenhuma consulta canônica compartilhada "
            "entre a base\nde demonstrações e o teste, nenhuma paráfrase de alta similaridade e nenhuma pergunta sem par "
            "entre os idiomas.", ha="center", va="center", fontsize=6.6, color=estilo.TINTA_2, linespacing=1.3)

    for caminho in estilo.salvar(fig, saida, "particao_familias"):
        print(caminho)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
