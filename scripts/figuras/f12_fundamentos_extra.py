"""F12 — Diagramas adicionais do Capítulo 2 e linha do tempo do Capítulo 3.

Gera quatro figuras:
  esquema_x_demonstracoes recuperação de esquema (ilustrativa) e recuperação de demonstrações (Seção 2.3.2)
  selecao_demonstracoes   a mesma pergunta e duas demonstrações candidatas: uma mais parecida na pergunta,
                          outra com a mesma estrutura de consulta (Seção 2.5.4)
  comparacao_pareada      (a) tabela de pares do teste de McNemar e (b) reamostragem bootstrap pareada e
                          estratificada (Seção 2.6); esquemática, sem dados do experimento
  linha_do_tempo          trabalhos do Capítulo 3 por ano e por eixo de análise

Uso: python scripts/figuras/f12_fundamentos_extra.py <dir_saida>
     (executar na raiz do repositório: lê resultados/confirmatorio/artifacts/manifest.json)

As perguntas e consultas de `selecao_demonstracoes` são lidas do manifesto: a pergunta de validação
`spider_train_spider_109` (banco bike_1) e as demonstrações `spider_train_spider_176` e `..._199` da base.
É o mesmo item da tabela do exemplo do mecanismo (Cap. 4): pela similaridade da pergunta delexicalizada, a
primeira fica à frente da segunda e entra na seleção de C3; a reordenação estrutural a troca pela segunda (C4).
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import estilo  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Rectangle  # noqa: E402
from diagrama import BORDA, MONO, REALCE, REALCE_AZUL, caixa, eixos, linha, painel, rotulo, seta  # noqa: E402

RAIZ = Path(__file__).resolve().parents[2]
MANIFESTO = RAIZ / "resultados/confirmatorio/artifacts/manifest.json"
PERGUNTA, DEMO_A, DEMO_B = "spider_train_spider_109", "spider_train_spider_176", "spider_train_spider_199"
# pergunta de teste do banco college_1 (Figura 5 e Quadro do literal traduzido) e uma das demonstrações que C3
# recuperou para ela no teste (registro do teste, campo prompt_demo_ids)
PERGUNTA_C1, DEMO_C1 = "spider_train_spider_3289", "spider_train_spider_3184"


def _sql(s: str) -> str:
    """Espaços múltiplos do Spider reduzidos a um, e quebra antes das cláusulas, só para a leitura na figura."""
    s = " ".join(s.split()).replace(" ,", ",")
    for kw in (" FROM ", " WHERE ", " ORDER BY "):
        s = s.replace(kw, "\n" + kw.strip() + " ")
    s = s.replace(" BETWEEN ", "\n  BETWEEN ")
    linhas = []
    for ln in s.split("\n"):  # lista de colunas longa: uma quebra depois da vírgula
        if len(ln) > 40 and ", " in ln:
            corte = ln.rfind(", ", 0, 40) + 1
            linhas += [ln[:corte], "       " + ln[corte:].strip()]
        else:
            linhas.append(ln)
    return "\n".join(linhas)


def selecao_demonstracoes(saida: Path) -> None:
    man = {e["example_id"]: e for e in json.loads(MANIFESTO.read_text(encoding="utf-8"))["examples"]}
    p, a, b = man[PERGUNTA], man[DEMO_A], man[DEMO_B]
    fig, ax = eixos(estilo.LARGURA, 3.75, ylim=(0, 66))
    aspas = ("“", "”")

    # pergunta de teste
    caixa(ax, 21, 50, 58, 15, "", cor="#ffffff")
    rotulo(ax, 50, 62.6, "Pergunta a responder", fs=7.4, negrito=True, cor=estilo.TINTA)
    rotulo(ax, 50, 58.3, aspas[0] + p["question_pt"] + aspas[1], fs=7.2, cor=estilo.TINTA)
    rotulo(ax, 50, 54.6, "consulta esperada (desconhecida no momento da geração):", fs=6.2, italico=True)
    rotulo(ax, 50, 51.9, " ".join(p["sql"].split()), fs=6.2, mono=True)

    def candidata(x, cor, borda, titulo, d, marca):
        caixa(ax, x, 0, 45, 36, "", cor=cor, borda=borda, lw=1.0)
        rotulo(ax, x + 22.5, 33.4, titulo, fs=7.2, negrito=True, cor=estilo.TINTA)
        q = textwrap.fill(aspas[0] + d["question_pt"] + aspas[1], 46)
        ax.text(x + 2, 30.2, q, ha="left", va="top", fontsize=6.5, color=estilo.TINTA, linespacing=1.25)
        ax.text(x + 2, 22.0, _sql(d["sql"]), ha="left", va="top", fontsize=5.9, family=MONO,
                color=estilo.TINTA, linespacing=1.3)
        ax.text(x + 22.5, 1.4, marca, ha="center", va="bottom", fontsize=6.2, color=estilo.TINTA_2,
                style="italic", linespacing=1.2)

    candidata(0, REALCE_AZUL, estilo.AZUL, "Demonstração A: pergunta mais parecida", a,
              f"palavras em comum ({aspas[0]}data{aspas[1]}, {aspas[0]}temperatura{aspas[1]}),\n"
              "mas outra estrutura: ordenação e limite,\nsem filtro")
    candidata(55, REALCE, estilo.LARANJA, "Demonstração B: mesma estrutura", b,
              "outro assunto (pressão ao nível do mar),\nmas a mesma estrutura: uma tabela\ne um filtro em WHERE")
    seta(ax, 35, 50, 22.5, 36.6, cor=estilo.AZUL)
    seta(ax, 65, 50, 77.5, 36.6, cor=estilo.LARANJA)
    rotulo(ax, 25.5, 44.5, "preferida pela similaridade\nda pergunta delexicalizada", fs=6.6, ha="right",
           cor=estilo.AZUL)
    rotulo(ax, 74.5, 44.5, "preferida pela similaridade\nestrutural", fs=6.6, ha="left", cor=estilo.LARANJA)
    estilo.salvar(fig, saida, "selecao_demonstracoes")


def comparacao_pareada(saida: Path) -> None:
    fig, ax = eixos(estilo.LARGURA, 3.3, ylim=(0, 60))

    # ---------------- (a) McNemar
    rotulo(ax, 0, 58.5, "(a) Teste de McNemar", fs=7.6, ha="left", negrito=True, cor=estilo.TINTA)
    x0, y0, w, h = 14, 25, 14, 10.5
    rotulo(ax, x0 + w, y0 + 2 * h + 6.6, "Sistema B", fs=7.0, negrito=True)
    rotulo(ax, x0 + w / 2, y0 + 2 * h + 2.6, "acerta", fs=6.6)
    rotulo(ax, x0 + 1.5 * w, y0 + 2 * h + 2.6, "erra", fs=6.6)
    ax.text(1.5, y0 + h, "Sistema A", rotation=90, ha="center", va="center", fontsize=7.0, fontweight="bold",
            color=estilo.TINTA_2)
    rotulo(ax, x0 - 1.5, y0 + 1.5 * h, "acerta", fs=6.6, ha="right")
    rotulo(ax, x0 - 1.5, y0 + 0.5 * h, "erra", fs=6.6, ha="right")
    celulas = {(0, 1): ("a", "ambos acertam", False), (1, 1): ("b", "só A acerta", True),
               (0, 0): ("c", "só B acerta", True), (1, 0): ("d", "ambos erram", False)}
    for (i, j), (letra, desc, disc) in celulas.items():
        x, y = x0 + i * w, y0 + j * h
        ax.add_patch(Rectangle((x, y), w, h, facecolor=REALCE if disc else "#ffffff",
                               edgecolor=estilo.LARANJA if disc else BORDA, linewidth=1.1 if disc else 0.7,
                               zorder=3 if disc else 2))
        ax.text(x + w / 2, y + h * 0.62, letra, ha="center", va="center", fontsize=9.5, fontweight="bold",
                style="italic", zorder=4)
        ax.text(x + w / 2, y + h * 0.24, desc, ha="center", va="center", fontsize=5.9, color=estilo.TINTA_2,
                zorder=4)
    ax.text(0, 19.5,
            "Em destaque, os pares discordantes (b e c). Os pares em que os\n"
            "dois sistemas acertam ou erram juntos não informam sobre a\n"
            "diferença entre eles. Se os sistemas forem equivalentes, cada par\n"
            "discordante cai em b ou em c com a mesma chance, e o valor-p\n"
            "vem da distribuição binomial de b em b + c.",
            ha="left", va="top", fontsize=6.3, color=estilo.TINTA_2, linespacing=1.25)
    rotulo(ax, 0, 2.5, "diferença de acurácia entre A e B = (b $-$ c) / n", fs=6.6, ha="left", cor=estilo.TINTA)

    # ---------------- (b) bootstrap
    xb = 53
    rotulo(ax, xb, 58.5, "(b) Bootstrap pareado e estratificado", fs=7.6, ha="left", negrito=True,
           cor=estilo.TINTA)
    cores_banco = (estilo.AZUL, estilo.AQUA, "#eda100")
    rotulo(ax, xb + 7.2, 53.0, "amostra original", fs=6.4, italico=True)
    for bi, cor in enumerate(cores_banco):
        for k in range(4):
            ax.add_patch(Rectangle((xb + 0.6 + k * 3.6, 46.6 - bi * 5.0), 3.0, 3.8, facecolor=cor,
                                   edgecolor="#ffffff", alpha=0.85, zorder=2))
    seta(ax, xb + 16.0, 43.0, xb + 22.0, 43.0)
    rotulo(ax, xb + 36.0, 53.0, "$B$ reamostras", fs=6.4, italico=True)
    rng = np.random.default_rng(7)
    for r in range(3):
        yy = 47.8 - r * 5.4
        for bi, cor in enumerate(cores_banco):
            sorteio = np.sort(rng.integers(0, 4, size=4))  # com reposição: repetições e ausências
            for k, q in enumerate(sorteio):
                ax.add_patch(Rectangle((xb + 23.0 + bi * 7.0 + k * 1.55, yy), 1.35, 3.2, facecolor=cor,
                                       edgecolor="none", alpha=0.45 + 0.13 * q, zorder=2))
        rotulo(ax, xb + 46.0, yy + 1.6, f"$d^*_{r + 1}$", fs=7.0)
    rotulo(ax, xb + 33.5, 33.0, "...", fs=8)
    rotulo(ax, xb + 46.0, 33.0, "$d^*_B$", fs=7.0)
    ax.text(xb, 28.5,
            "Cada quadrado é uma pergunta, com os desfechos dos dois sistemas;\n"
            "a cor indica o banco. Cada reamostra sorteia, com reposição, o mesmo\n"
            "número de perguntas de cada banco e recalcula a diferença $d^*$.",
            ha="left", va="top", fontsize=6.0, color=estilo.TINTA_2, linespacing=1.25)
    # distribuição das diferenças
    xs = np.linspace(-3, 3, 200)
    ys = np.exp(-xs ** 2 / 2)
    gx, gy, gw, gh = xb + 3, 3.5, 36, 12
    px, py = gx + (xs + 3) / 6 * gw, gy + ys * gh
    ax.fill_between(px, gy, py, color=estilo.CINZA_MEDIO, zorder=1, linewidth=0)
    sel = (xs > -1.96) & (xs < 1.96)
    ax.fill_between(px[sel], gy, py[sel], color=estilo.AZUL, alpha=0.35, zorder=2, linewidth=0)
    linha(ax, [gx, gx + gw], [gy, gy], cor=estilo.TINTA_3, lw=0.6)
    for q, t in ((-1.96, "percentil 2,5"), (1.96, "percentil 97,5")):
        xq = gx + (q + 3) / 6 * gw
        linha(ax, [xq, xq], [gy, gy + gh * 0.5], cor=estilo.AZUL, lw=0.8)
        rotulo(ax, xq, gy - 1.9, t, fs=5.8, cor=estilo.AZUL)
    rotulo(ax, gx + gw / 2, gy + gh * 0.32, "IC de 95%", fs=6.6, negrito=True, cor=estilo.TINTA)
    rotulo(ax, gx + gw + 1.0, gy + 5.0, "distribuição\nde $d^*_1, \\ldots, d^*_B$", fs=5.9, ha="left")
    estilo.salvar(fig, saida, "comparacao_pareada")


def esquema_x_demonstracoes(saida: Path) -> None:
    """Recuperação de esquema (ilustrativa) e recuperação de demonstrações, com a pergunta do banco college_1."""
    man = {e["example_id"]: e for e in json.loads(MANIFESTO.read_text(encoding="utf-8"))["examples"]}
    p, demo = man[PERGUNTA_C1], man[DEMO_C1]
    aspas = ("\u201c", "\u201d")
    fig, ax = eixos(estilo.LARGURA, 4.0, ylim=(0, 70))
    pergunta = textwrap.fill(aspas[0] + p["question_pt"] + aspas[1], 36)

    def coluna(x0, titulo, indice, item, item_mono, blocos, destaque, rodape, ilustrativo=False, item_texto=""):
        painel(ax, x0, 4.6, 48.5, 64.9, titulo)
        cx = x0 + 24.25
        caixa(ax, x0 + 4, 52, 40.5, 9, pergunta, fs=6.6)
        seta(ax, cx, 52, cx, 48.5)
        caixa(ax, x0 + 4, 41.5, 40.5, 7, indice, cor=estilo.CINZA_MEDIO, fs=6.6)
        seta(ax, cx, 41.5, cx, 38)
        caixa(ax, x0 + 4, 24.2, 40.5, 13.8, "", cor="#ffffff", borda=estilo.LARANJA if destaque else BORDA, lw=1.0)
        ax.text(x0 + 5.5, 36.6, item, ha="left", va="top", fontsize=6.0, color=estilo.TINTA_2, style="italic")
        y_mono = 33.6
        if item_texto:
            ax.text(x0 + 5.5, 33.8, item_texto, ha="left", va="top", fontsize=6.2, color=estilo.TINTA,
                    linespacing=1.2)
            y_mono = 33.8 - 2.3 * (item_texto.count("\n") + 1) - 0.4
        ax.text(x0 + 5.5, y_mono, item_mono, ha="left", va="top", fontsize=5.6, family=MONO, color=estilo.TINTA,
                linespacing=1.25)
        seta(ax, cx, 24.2, cx, 22.4)
        caixa(ax, x0 + 4, 6.2, 40.5, 16.2, "", cor="#ffffff")
        rotulo(ax, cx, 20.5, "Prompt", fs=6.8, negrito=True, cor=estilo.TINTA)
        for i, (t, real) in enumerate(blocos):
            ax.text(x0 + 6, 17.9 - i * 2.5, t, ha="left", va="top", fontsize=6.1,
                    color=estilo.LARANJA if real else estilo.TINTA, fontweight="bold" if real else "normal")
        rotulo(ax, cx, 1.8, rodape, fs=6.6, italico=True, cor=estilo.TINTA)
        if ilustrativo:
            rotulo(ax, x0 + 47, 66.4, "(ilustrativo)", fs=5.8, ha="right", italico=True)

    coluna(0, "(a) Recuperação de esquema", "recuperador sobre um índice\nde tabelas e colunas",
           "item recuperado: parte do esquema",
           "PROFESSOR(EMP_NUM, DEPT_CODE, ...)\nDEPARTMENT(DEPT_CODE, DEPT_NAME, ...)",
           [("1. Instrução", False), ("2. Esquema reduzido às tabelas recuperadas", True), ("3. Pergunta", False)],
           True, "informa ao modelo sobre o que consultar", ilustrativo=True)
    sql = " ".join(demo["sql"].split()).replace(" WHERE ", "\nWHERE ")
    coluna(51.5, "(b) Recuperação de demonstrações", "recuperador sobre um índice\nde pares pergunta\u2013SQL",
           "item recuperado: um par pergunta\u2013consulta",
           sql,
           [("1. Instrução", False), ("2. Esquema completo", False), ("3. Demonstrações recuperadas", True),
            ("4. Pergunta", False)],
           True, "informa ao modelo como consultar (objeto deste trabalho)",
           item_texto=textwrap.fill(aspas[0] + demo["question_pt"] + aspas[1], 50))
    estilo.salvar(fig, saida, "esquema_x_demonstracoes")


TRABALHOS = [  # (ano, eixo, nome, descrição curta) — eixos como no Capítulo 3; a ordem define o empilhamento
    (2023, 0, "Chang e Fosler-\nLussier (2023a)", "exemplos do banco\nno prompt"),
    (2023, 0, "ODIS (Chang e Fosler-\nLussier, 2023b)", "a distribuição das\nconsultas SQL importa"),
    (2024, 0, "LPE-SQL", "base de exemplos\nque cresce"),
    (2023, 1, "ReFSQL", "estrutura, com\nrecuperador treinado"),
    (2024, 1, "DAIL-SQL", "pergunta mascarada\n+ SQL preliminar"),
    (2024, 1, "ASTReS", "reordenação pela\nárvore sintática"),
    (2024, 1, "Solid-SQL", "esqueleto da pergunta\ne da consulta"),
    (2024, 1, "Dubo-SQL", "pergunta, com\ndiversidade"),
    (2025, 1, "OpenSearch-SQL", "pergunta\nmascarada"),
    (2024, 2, "Ziletti et al.", "saúde: entidades\nmascaradas"),
    (2025, 2, "Carvalho et al.;\nPedroso et al.", "português × inglês,\nsem demonstrações"),
    (2026, 2, "Bouattour et al.", "banco industrial;\ncusto das demonstrações"),
    (2026, 2, "Nobre et al.", "BIRD em português;\nefeito do raciocínio"),
]
EIXOS = ("Por que as\ndemonstrações\najudam", "Critérios de\nseleção", "Condições\nrealistas e\noutros idiomas")


def linha_do_tempo(saida: Path) -> None:
    fig, ax = eixos(estilo.LARGURA, 5.1, xlim=(0, 100), ylim=(12, 101))
    anos = (2023, 2024, 2025, 2026)
    x_ano = {a: 28 + i * 20.5 for i, a in enumerate(anos)}
    faixas = {0: (72.5, 96.5), 1: (35.5, 70.5), 2: (14, 33.5)}
    larg, folga = 19.0, 1.2

    def altura(nome: str, desc: str) -> float:
        return 1.7 + 2.45 * (nome.count("\n") + 1) + 2.1 * (desc.count("\n") + 1)

    for e, (y_inf, y_sup) in faixas.items():
        ax.add_patch(FancyBboxPatch((0, y_inf), 100, y_sup - y_inf, boxstyle="round,pad=0,rounding_size=1.2",
                                    facecolor=estilo.CINZA_CLARO, edgecolor="none", zorder=0))
        rotulo(ax, 1.4, (y_inf + y_sup) / 2, EIXOS[e], fs=6.8, ha="left", negrito=True, cor=estilo.TINTA_2)
    for a, x in x_ano.items():
        rotulo(ax, x, 99.6, str(a), fs=7.8, negrito=True, cor=estilo.TINTA)
        linha(ax, [x, x], [14, 97.6], cor=estilo.GRADE, lw=0.6)

    def cartao(cx, topo, nome, desc, cor="#ffffff", borda=BORDA, lw=0.7):
        h = altura(nome, desc)
        caixa(ax, cx - larg / 2, topo - h, larg, h, cor=cor, borda=borda, lw=lw, raio=0.8, z=2)
        ax.text(cx, topo - 1.0, nome, ha="center", va="top", fontsize=6.1, fontweight="bold", color=estilo.TINTA,
                linespacing=1.1, zorder=3)
        ax.text(cx, topo - h + 0.9, desc, ha="center", va="bottom", fontsize=5.5, color=estilo.TINTA_2,
                linespacing=1.1, zorder=3)
        return h

    por_celula: dict[tuple[int, int], list] = {}
    for t in TRABALHOS:
        por_celula.setdefault((t[0], t[1]), []).append(t)
    for (ano, eixo), lista in por_celula.items():
        y_inf, y_sup = faixas[eixo]
        alturas = [altura(t[2], t[3]) for t in lista]
        total = sum(alturas) + folga * (len(lista) - 1)
        topo = (y_inf + y_sup) / 2 + total / 2
        for t, h in zip(lista, alturas):
            cartao(x_ano[ano], topo, t[2], t[3])
            topo -= h + folga
    # este trabalho
    yc = (faixas[1][0] + faixas[1][1]) / 2
    cartao(x_ano[2026], yc + 6.0, "Este trabalho", "pergunta, pergunta\ndelexicalizada e\nreordenação estrutural,\n"
           "em português e inglês", cor=REALCE, borda=estilo.LARANJA, lw=1.1)
    estilo.salvar(fig, saida, "linha_do_tempo")


def main(saida: Path) -> None:
    estilo.aplicar()
    esquema_x_demonstracoes(saida)
    selecao_demonstracoes(saida)
    comparacao_pareada(saida)
    linha_do_tempo(saida)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
