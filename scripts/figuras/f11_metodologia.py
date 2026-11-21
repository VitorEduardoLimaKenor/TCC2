"""F11 — Diagramas do Capítulo 4 (metodologia).

Gera três figuras:
  fluxo_c4                 as cinco etapas do framework na condição C4, numeradas como no texto (Seção 4.1)
  estrutura_prompt         o prompt real de uma pergunta de teste em C3, com os quatro blocos (Seção 4.4)
  protocolo_experimental   a ordem das decisões e das execuções, do início da validação ao teste (Seção 4.6)

Uso: python scripts/figuras/f11_metodologia.py <dir_saida>
     (executar na raiz do repositório: lê data/spider/tables.json e resultados/confirmatorio/)

O prompt de `estrutura_prompt` é remontado pelo próprio código do experimento (`build_prompt`), a partir do
manifesto e das demonstrações gravadas no registro do teste, e o script confere que seu SHA-256 coincide com o
`prompt_sha256` do registro antes de desenhar: a figura mostra exatamente o texto que o modelo recebeu, com o
esquema abreviado e as linhas quebradas para caber na largura.
"""

from __future__ import annotations

import dataclasses
import json
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import estilo  # noqa: E402
from diagrama import (BORDA, MONO, REALCE, caixa, cilindro, eixos, etapa, linha, painel, rotulo,  # noqa: E402
                      seta)

RAIZ = Path(__file__).resolve().parents[2]
REGISTRO_TESTE = RAIZ / "resultados/confirmatorio/artifacts/runs-v2/confirmatorio-test-a99f7394fedc945b-fb350ed2.jsonl"
MANIFESTO = RAIZ / "resultados/confirmatorio/artifacts/manifest.json"
TABELAS = RAIZ / "data/spider/tables.json"
EXEMPLO = "spider_train_spider_3289"  # banco college_1; o mesmo item da tabela do literal traduzido (Cap. 5)


# --------------------------------------------------------------------------- fluxo de C4


def fluxo_c4(saida: Path) -> None:
    fig, ax = eixos(estilo.LARGURA, 5.5, ylim=(0, 100))
    azul = estilo.AZUL

    # pergunta
    caixa(ax, 33, 92, 34, 7, "Pergunta do usuário\n(em linguagem natural)", negrito=True, fs=7.6)

    # etapa 1 — busca vetorial
    painel(ax, 0, 42, 48.5, 45.5, "Etapa 1 · Busca vetorial")
    etapa(ax, 45.2, 84.8, 1)
    xb, wb = 17.5, 30
    caixa(ax, xb, 74.5, wb, 7.5, "Delexicalização da pergunta\n(valores → <VAL>, <NUM>, <DATE>)", fs=6.4)
    caixa(ax, xb, 64.5, wb, 7.0, "Vetorização da pergunta\n(modelo de embeddings multilíngue)", fs=6.4)
    caixa(ax, xb, 54.5, wb, 7.0, "Busca pela similaridade do cosseno\n(mesmo banco e mesmo idioma)", fs=6.4)
    caixa(ax, xb + 5.5, 44.0, wb - 11, 4.6, "$N$ candidatos", fs=7.2, negrito=True)
    cilindro(ax, 2.0, 53.5, 10.5, 9.5)
    rotulo(ax, 7.25, 48.6, "Base de\ndemonstrações\n(Qdrant)", fs=6.0, cor=estilo.TINTA)
    seta(ax, 12.5, 58.0, xb, 58.0, dupla=True, ms=6)
    for y0, y1 in ((74.5, 71.5), (64.5, 61.5), (54.5, 48.6)):
        seta(ax, xb + wb / 2, y0, xb + wb / 2, y1)

    # etapa 2 — sinal estrutural
    painel(ax, 51.5, 42, 48.5, 45.5, "Etapa 2 · Sinal estrutural")
    xr, wr = 57.5, 37
    caixa(ax, xr, 74.5, wr, 7.5, "Modelo de linguagem, sem demonstrações\n(mesmo prompt da condição C0)", fs=6.4)
    etapa(ax, xr - 3.2, 78.25, 2)
    caixa(ax, xr, 64.5, wr, 7.0, "Consulta preliminar $\\hat{s}$", fs=7.0)
    caixa(ax, xr, 54.5, wr, 7.0, "Análise sintática da consulta preliminar\n(árvore sintática abstrata)", fs=6.4)
    etapa(ax, xr - 3.2, 58.0, 3)
    caixa(ax, 53.5, 43.5, 21.5, 5.8, "Assinatura estrutural\n(12 características)", fs=6.5, negrito=True)
    caixa(ax, 77.0, 43.5, 22.0, 5.8, "Se a análise falhar,\nmantém a ordem da etapa 1", fs=6.3,
          borda=estilo.LARANJA, tracejada=True, lw=1.0)
    for y0, y1 in ((74.5, 71.5), (64.5, 61.5)):
        seta(ax, xr + wr / 2, y0, xr + wr / 2, y1)
    xm = xr + wr / 2
    linha(ax, [xm, xm], [54.5, 52.2])
    linha(ax, [64.25, xm], [52.2, 52.2])
    linha(ax, [xm, 88.0], [52.2, 52.2], cor=estilo.LARANJA, tracejada=True)
    seta(ax, 64.25, 52.2, 64.25, 49.3)
    seta(ax, 88.0, 52.2, 88.0, 49.3, cor=estilo.LARANJA, tracejada=True)

    # da pergunta às duas etapas
    linha(ax, [50, 50], [92, 89.5])
    linha(ax, [xb + wb / 2, xm], [89.5, 89.5])
    seta(ax, xb + wb / 2, 89.5, xb + wb / 2, 82)
    seta(ax, xm, 89.5, xm, 82)

    # reordenação
    caixa(ax, 8, 28.5, 84, 10.5, "", cor=estilo.CINZA_CLARO)
    etapa(ax, 8.2, 39, 4)
    rotulo(ax, 50, 36.2, "Reordenação dos $N$ candidatos e escolha dos $k$ de maior pontuação", fs=7.4,
           negrito=True, cor=estilo.TINTA)
    rotulo(ax, 50, 31.3, r"$\mathrm{score}(e) = \lambda \cdot \mathrm{sim}_{\mathrm{perg}}(q, e)"
                         r" + (1 - \lambda) \cdot \mathrm{sim}_{\mathrm{estr}}(\hat{s}, s_e)$", fs=8.2,
           cor=estilo.TINTA)
    seta(ax, xb + wb / 2, 44.0, xb + wb / 2, 39)
    seta(ax, 64.25, 43.5, 64.25, 39)
    seta(ax, 88.0, 43.5, 88.0, 39, cor=estilo.LARANJA, tracejada=True)

    # geração
    caixa(ax, 8, 2, 46, 21, "", cor="#ffffff")
    etapa(ax, 8.2, 23, 5)
    rotulo(ax, 31, 20.3, "Prompt final", fs=7.6, negrito=True, cor=estilo.TINTA)
    itens = ("1. Instrução da tarefa", "2. Esquema completo do banco (DDL)",
             "3. $k$ demonstrações (pares pergunta–SQL)", "4. Pergunta original do usuário")
    for i, t in enumerate(itens):
        destaque = i == 2
        caixa(ax, 11, 14.2 - i * 3.9, 40, 3.3, t, fs=6.8, cor=REALCE if destaque else "#ffffff",
              borda=estilo.LARANJA if destaque else BORDA, lw=1.1 if destaque else 0.7)
    seta(ax, xb + wb / 2, 28.5, xb + wb / 2, 23)
    caixa(ax, 60, 8, 17, 9, "Modelo de\nlinguagem", cor=estilo.CINZA_CLARO, negrito=True, fs=7.4)
    caixa(ax, 82, 8, 17, 9, "Consulta SQL\ngerada", negrito=True, fs=7.4)
    seta(ax, 54, 12.5, 60, 12.5)
    seta(ax, 77, 12.5, 82, 12.5)
    estilo.salvar(fig, saida, "fluxo_c4")


# --------------------------------------------------------------------------- estrutura do prompt


def _prompt_do_registro():
    """Remonta o prompt de C3 em português do item EXEMPLO e confere o SHA-256 com o registro do teste."""
    from tcc_rag.data.spider import build_ddl
    from tcc_rag.generation.prompt import build_prompt
    from tcc_rag.schemas import Demo

    tabelas = {t["db_id"]: t for t in json.loads(TABELAS.read_text(encoding="utf-8"))}
    manifesto = {e["example_id"]: e for e in json.loads(MANIFESTO.read_text(encoding="utf-8"))["examples"]}
    reg = None
    with REGISTRO_TESTE.open(encoding="utf-8") as fh:
        for linha_txt in fh:
            r = json.loads(linha_txt)
            if r["example_id"] == EXEMPLO and r["condition"] == "C3" and r["lang"] == "pt":
                reg = r
                break
    assert reg is not None, "item não encontrado no registro"
    campos = [f.name for f in dataclasses.fields(Demo)]

    def demo(uid: str) -> Demo:
        eid = uid.split("::")[0]
        e = manifesto[eid]
        kw = dict(uid=uid, example_id=eid, db_id=e["db_id"], lang="pt", question_raw=e["question_pt"],
                  question_delex="", sql=e["sql"], signature={}, tables=[])
        kw.update({c: None for c in campos if c not in kw})
        return Demo(**kw)

    ddl = build_ddl(tabelas[reg["db_id"]])
    # limite de contexto (4096) e reserva para a geração (256 + 64), como na configuração congelada
    built = build_prompt(reg["question_raw"], ddl, [demo(u) for u in reg["prompt_demo_ids"]], "pt", 4096, 320)
    if built.sha256 != reg["prompt_sha256"]:
        raise SystemExit("o prompt remontado difere do registrado; a figura não seria fiel")
    return built.text, ddl


def estrutura_prompt(saida: Path) -> None:
    texto, ddl = _prompt_do_registro()
    antes_ddl, resto = texto.split(ddl, 1)
    # esquema abreviado: só a tabela DEPARTMENT por inteiro; as demais recolhidas
    blocos_ddl = ddl.split("\n\n")
    ddl_curto = []
    for b_ in blocos_ddl:
        nome = b_.split("(")[0].replace("CREATE TABLE", "").strip()
        ddl_curto += b_.split("\n") if nome == "DEPARTMENT" else [f"CREATE TABLE {nome} ( … );"]
    instrucao, rotulo_esquema = antes_ddl.rstrip("\n").rsplit("\n", 1)
    exemplos, pergunta = resto.strip("\n").split("\n\n")
    largura = 70

    def quebra(s: str, recuo: str = "") -> list[str]:
        out = []
        for ln in s.split("\n"):
            out += textwrap.wrap(ln, largura, subsequent_indent=recuo, break_long_words=True,
                                 replace_whitespace=False, drop_whitespace=True) or [""]
        return out

    secoes = [  # (bloco, linhas)
        (1, quebra(instrucao)),
        (0, [""]),
        (2, [rotulo_esquema] + ddl_curto),
        (0, [""]),
        (3, sum((quebra(x, "    ") for x in exemplos.split("\n")), [])),
        (0, [""]),
        (4, quebra(pergunta)),
    ]
    n = sum(len(l) for _, l in secoes)
    passo = 1.0
    altura = n * passo + 2
    fig, ax = eixos(estilo.LARGURA, 0.115 * altura + 0.3, ylim=(0, altura))
    x0, x1 = 0.0, 73.0
    y = altura - 1.0
    faixas = {}
    for bloco_id, linhas in secoes:
        topo = y
        for ln in linhas:
            ax.text(x0 + 1.2, y, ln, ha="left", va="top", fontsize=5.7, family=MONO, color=estilo.TINTA, zorder=3)
            y -= passo
        if bloco_id:
            faixas[bloco_id] = (y + 0.15, topo + 0.25)
    cores = {1: "#ffffff", 2: estilo.CINZA_CLARO, 3: REALCE, 4: "#ffffff"}
    bordas = {1: BORDA, 2: BORDA, 3: estilo.LARANJA, 4: BORDA}
    for k, (yb, yt) in faixas.items():
        caixa(ax, x0, yb, x1 - x0, yt - yb, cor=cores[k], borda=bordas[k], lw=1.0 if k == 3 else 0.7, raio=0.6,
              z=1)
    notas = {
        1: ("Instrução", "igual em todas as condições"),
        2: ("Esquema completo (DDL)", "igual em todas as condições;\nseis das sete tabelas\nrecolhidas nesta figura"),
        3: ("Demonstrações", "$k$ = 4 pares pergunta–consulta\nescolhidos pela condição (aqui, C3);\n"
                             "ausentes em C0 e na consulta\npreliminar de C4"),
        4: ("Pergunta original", "a consulta gerada pelo modelo\nvem depois de “Consulta:”"),
    }
    for k, (yb, yt) in faixas.items():
        xm = x1 + 1.5
        linha(ax, [xm, xm + 0.8, xm + 0.8, xm], [yt, yt, yb, yb], cor=estilo.TINTA_3, lw=0.8)
        ym = (yb + yt) / 2
        etapa(ax, xm + 3.6, ym + 0.2, k, r=1.35, fs=6.2, cor=estilo.LARANJA if k == 3 else estilo.TINTA)
        titulo, desc = notas[k]
        ax.text(xm + 6.0, ym + 0.9, titulo, ha="left", va="bottom", fontsize=7.0, fontweight="bold",
                color=estilo.TINTA)
        ax.text(xm + 6.0, ym + 0.5, desc, ha="left", va="top", fontsize=6.3, color=estilo.TINTA_2,
                linespacing=1.2)
    estilo.salvar(fig, saida, "estrutura_prompt")


# --------------------------------------------------------------------------- protocolo


def protocolo_experimental(saida: Path) -> None:
    fig, ax = eixos(estilo.LARGURA, 3.0, ylim=(0, 56))
    y0 = 27
    eventos = [  # (x, acima?, tipo, título, descrição)
        (6, True, "exec", "Varredura de validação", "150 perguntas; arquivo\nprovisório (perguntas em\ninglês); avaliador anterior"),
        (19, False, "decisao", "Redesignação (22/09/2026)", "C3 $-$ C2 passa a ser a\ncomparação primária; C4,\nanálise de componente"),
        (32, True, "decisao", "Registro de 28/10/2026", "C3 mantida qualquer que\nseja a nova validação;\nanálise por literal traduzido"),
        (45, False, "decisao", "Registro de 11/11/2026", "C3 $-$ C2 e H1 formam a\nfamília corrigida por Holm"),
        (57, True, "exec", "Validação definitiva", "275 perguntas; tradução\nde José e Cozman (2021);\n5 condições × 2 idiomas"),
        (69, False, "decisao", "Congelamento (14/11/2026)", "configuração, revisões e\npoder gravados; só então\no código libera o teste"),
        (82, True, "exec", "Teste confirmatório", "471 perguntas;\n5 condições × 2 idiomas"),
        (94, False, "exec", "Análises posteriores", "rubrica de erros e\nlatência (15/11/2026)"),
    ]
    # faixa do teste selado (o eixo passa pela faixa; os rótulos ficam acima dele)
    from matplotlib.patches import Rectangle
    x_cong = 69
    ax.add_patch(Rectangle((0, y0 - 1.6), x_cong, 6.6, facecolor=estilo.CINZA_CLARO, edgecolor="none", zorder=0))
    rotulo(ax, 44, y0 + 2.7, "conjunto de teste selado", fs=6.6, italico=True, cor=estilo.TINTA_2)
    rotulo(ax, 88, y0 + 2.7, "teste aberto", fs=6.6, italico=True, cor=estilo.TINTA_2)
    seta(ax, 0, y0, 100, y0, cor=estilo.TINTA_3, lw=0.8)
    for x, acima, tipo, titulo, desc in eventos:
        cor = estilo.LARANJA if tipo == "decisao" else estilo.AZUL
        mk = "D" if tipo == "decisao" else "o"
        ax.plot([x], [y0], marker=mk, markersize=5.5, color=cor, markeredgecolor="#ffffff",
                markeredgewidth=0.7, zorder=5)
        if acima:
            linha(ax, [x, x], [y0 + 1.2, y0 + 8.0], cor=estilo.TINTA_3, lw=0.6)
            ax.text(x, y0 + 8.8, titulo, ha="center", va="bottom", fontsize=6.6, fontweight="bold", color=cor)
            ax.text(x, y0 + 14.4, desc, ha="center", va="bottom", fontsize=6.0, color=estilo.TINTA_2,
                    linespacing=1.15)
        else:
            linha(ax, [x, x], [y0 - 1.2, y0 - 5.5], cor=estilo.TINTA_3, lw=0.6)
            ax.text(x, y0 - 6.2, titulo, ha="center", va="top", fontsize=6.6, fontweight="bold", color=cor)
            ax.text(x, y0 - 10.6, desc, ha="center", va="top", fontsize=6.0, color=estilo.TINTA_2,
                    linespacing=1.15)
    # legenda
    ax.plot([2], [2.0], marker="o", markersize=5, color=estilo.AZUL, markeredgecolor="#ffffff")
    rotulo(ax, 3.6, 2.0, "execução", fs=6.6, ha="left")
    ax.plot([14], [2.0], marker="D", markersize=5, color=estilo.LARANJA, markeredgecolor="#ffffff")
    rotulo(ax, 15.6, 2.0, "decisão registrada", fs=6.6, ha="left")
    rotulo(ax, 99, 2.0, "eventos em ordem cronológica, sem escala de tempo", fs=6.2, ha="right", italico=True)
    estilo.salvar(fig, saida, "protocolo_experimental")


def main(saida: Path) -> None:
    estilo.aplicar()
    fluxo_c4(saida)
    estrutura_prompt(saida)
    protocolo_experimental(saida)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
