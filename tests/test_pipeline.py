"""Testes das invariantes que sustentam o experimento.

Não são testes de acurácia — são testes das propriedades que, se quebrarem,
invalidam o resultado sem dar sinal: vazamento entre partições, uso do gabarito
na recuperação, e divergência de prompt entre condições.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tcc_rag.config import DataConfig, ExperimentConfig, SandboxConfig
from tcc_rag.data import leakage as leak
from tcc_rag.data import partition as part
from tcc_rag.data.delex import delexicalize
from tcc_rag.data.normalize import canonicalize, extract_sql, family_id, is_read_only
from tcc_rag.evaluation.sandbox import Sandbox
from tcc_rag.evaluation.stats import difference_in_differences, mcnemar_exact, paired_bootstrap_diff
from tcc_rag.generation.prompt import build_prompt
from tcc_rag.schemas import Demo, Example
from tcc_rag.structure import signature as sig


# ----------------------------------------------------------- normalização


def test_canonicalizacao_ignora_apelido_e_literal():
    a = "SELECT T1.name FROM singer AS T1 WHERE T1.age > 30"
    b = "SELECT x.name FROM singer AS x WHERE x.age > 45"
    assert canonicalize(a) == canonicalize(b)


def test_canonicalizacao_distingue_estruturas():
    a = "SELECT name FROM singer WHERE age > 30"
    b = "SELECT name FROM singer GROUP BY name"
    assert canonicalize(a) != canonicalize(b)


def test_familia_e_escopada_ao_banco():
    sql = "SELECT count(*) FROM t"
    assert family_id("concert_singer", sql) != family_id("pets_1", sql)


def test_apenas_leitura():
    assert is_read_only("SELECT 1")
    assert is_read_only("WITH x AS (SELECT 1) SELECT * FROM x")
    assert not is_read_only("DROP TABLE alunos")
    assert not is_read_only("SELECT 1; DROP TABLE alunos")


@pytest.mark.parametrize(
    "raw,esperado",
    [
        ("```sql\nSELECT 1\n```", "SELECT 1"),
        ("Claro! Aqui está:\n```sql\nSELECT a FROM b\n```", "SELECT a FROM b"),
        ("SELECT a FROM b; extra", "SELECT a FROM b"),
    ],
)
def test_extracao_de_sql(raw, esperado):
    assert extract_sql(raw) == esperado


# --------------------------------------------------------- delexicalização


def test_delex_mascara_literais():
    assert delexicalize('quantos pedidos o cliente "Silva" fez em 2024?') == (
        "quantos pedidos o cliente <VAL> fez em <DATE>?"
    )


def test_delex_e_puramente_superficial():
    """A operação não pode consultar o gabarito: no teste ele não existe no
    momento da inferência, e pool e teste precisam ser tratados igual."""
    import inspect

    from tcc_rag.data import delex

    fonte = inspect.getsource(delex)
    for proibido in ("sql", "gold", "query"):
        assert f"def delexicalize({proibido}" not in fonte
    assert len(inspect.signature(delex.delexicalize).parameters) == 1


# ---------------------------------------------------------------- assinatura


def test_assinatura_captura_construcoes():
    s = sig.signature(
        "SELECT c.nome, count(*) FROM cliente c JOIN pedido p ON c.id = p.cid "
        "GROUP BY c.nome HAVING count(*) > 3 ORDER BY 2 DESC LIMIT 5"
    )
    assert s["n_joins"] == 1
    assert s["has_group_by"] and s["has_having"] and s["has_order_by"] and s["has_limit"]
    assert "count" in s["agg_funcs"]


def test_assinatura_devolve_none_em_sql_invalido():
    assert sig.signature("isto não é sql (((") is None


def test_similaridade_e_maxima_para_iguais():
    s = sig.signature("SELECT a FROM t JOIN u ON t.id = u.id GROUP BY a")
    assert sig.similarity(s, s) == pytest.approx(1.0)


def test_compatibilidade_ignora_identificadores():
    a = sig.signature("SELECT x FROM t1 JOIN t2 ON t1.id = t2.id GROUP BY x")
    b = sig.signature("SELECT y FROM alfa JOIN beta ON alfa.k = beta.k GROUP BY y")
    assert sig.compatible(a, b)


def test_incompatibilidade_por_agrupamento():
    a = sig.signature("SELECT x FROM t GROUP BY x")
    b = sig.signature("SELECT x FROM t")
    assert not sig.compatible(a, b)


# --------------------------------------------------- partição e vazamento


def _exemplos(n: int = 60) -> list[Example]:
    out = []
    for i in range(n):
        sql = f"SELECT nome FROM tabela WHERE col{i % 12} = {i}"
        out.append(
            Example(
                example_id=f"ex{i}",
                db_id="banco_a" if i % 2 == 0 else "banco_b",
                question_en=f"question number {i}",
                question_pt=f"pergunta numero {i}",
                sql=sql,
                family_id=family_id("banco_a" if i % 2 == 0 else "banco_b", sql),
            )
        )
    return out


def test_familia_nunca_cruza_particao():
    cfg = DataConfig(min_questions_per_db=5, min_families_per_db=2)
    assigned = part.partition(_exemplos(), cfg)
    por_familia: dict[str, set[str]] = {}
    for ex in assigned:
        por_familia.setdefault(ex.family_id, set()).add(ex.partition)
    assert all(len(p) == 1 for p in por_familia.values())


def test_auditoria_detecta_familia_cruzando():
    exemplos = _exemplos(8)
    exemplos[0] = Example(**{**exemplos[0].__dict__, "partition": "pool"})
    exemplos[1] = Example(
        **{**exemplos[1].__dict__, "partition": "test", "family_id": exemplos[0].family_id}
    )
    for i in range(2, len(exemplos)):
        exemplos[i] = Example(**{**exemplos[i].__dict__, "partition": "pool"})
    relatorio = leak.audit(exemplos)
    assert not relatorio.clean
    assert relatorio.families_crossing


def test_assert_clean_levanta():
    exemplos = _exemplos(6)
    exemplos[0] = Example(**{**exemplos[0].__dict__, "partition": "pool"})
    exemplos[1] = Example(
        **{**exemplos[1].__dict__, "partition": "test", "family_id": exemplos[0].family_id}
    )
    with pytest.raises(leak.LeakageError):
        leak.assert_clean(exemplos)


# -------------------------------------------------------------- prompt


def _demo(uid: str) -> Demo:
    return Demo(
        uid=uid,
        example_id=uid,
        db_id="banco",
        lang="pt",
        question_raw=f"pergunta {uid}",
        question_delex=f"pergunta {uid}",
        sql="SELECT 1",
        signature={},
        tables=[],
        columns=[],
        family_id="f",
        partition="pool",
    )


def test_prompt_difere_apenas_no_bloco_de_demonstracoes():
    ddl = "CREATE TABLE t (a INT);"
    sem = build_prompt("qual o total?", ddl, [], "pt", 4096)
    com = build_prompt("qual o total?", ddl, [_demo("d1")], "pt", 4096)
    assert sem.sha256 != com.sha256
    # o esquema e a instrução seguem presentes e intactos
    assert ddl in sem.text and ddl in com.text
    assert sem.text.split(ddl)[0] == com.text.split(ddl)[0]


def test_prompt_corta_demonstracoes_quando_estoura_contexto():
    ddl = "CREATE TABLE t (a INT);" * 50
    built = build_prompt("pergunta", ddl, [_demo(f"d{i}") for i in range(8)], "pt", 700)
    assert built.n_demos_dropped > 0
    assert built.n_demos_used + built.n_demos_dropped == 8


# ------------------------------------------------------------- sandbox


@pytest.fixture
def banco(tmp_path: Path) -> str:
    caminho = tmp_path / "teste.sqlite"
    conn = sqlite3.connect(caminho)
    conn.execute("CREATE TABLE aluno (id INT, nome TEXT, nota REAL)")
    conn.executemany(
        "INSERT INTO aluno VALUES (?, ?, ?)",
        [(1, "Ana", 9.0), (2, "Bruno", 7.5), (3, "Célia", 8.25)],
    )
    conn.commit()
    conn.close()
    return str(caminho)


def test_sandbox_executa_leitura(banco):
    resultado = Sandbox(SandboxConfig()).run(banco, "SELECT nome FROM aluno ORDER BY nome")
    assert resultado.ok
    assert resultado.rows == [("Ana",), ("Bruno",), ("Célia",)]


def _contagem(banco: str) -> int:
    conn = sqlite3.connect(banco)
    try:
        return conn.execute("SELECT count(*) FROM aluno").fetchone()[0]
    finally:
        conn.close()


@pytest.mark.parametrize("sql", [
    "DELETE FROM aluno",
    "UPDATE aluno SET nota = 0",
    "DROP TABLE aluno",
    "SELECT 1; DROP TABLE aluno",
    "PRAGMA query_only = OFF",
])
def test_sandbox_nao_escreve(banco, sql):
    resultado = Sandbox(SandboxConfig()).run(banco, sql)
    assert not resultado.ok
    assert _contagem(banco) == 3


def test_sandbox_nega_attach(banco, tmp_path):
    alvo = tmp_path / "outro.sqlite"
    resultado = Sandbox(SandboxConfig()).run(banco, f"ATTACH DATABASE '{alvo}' AS x")
    assert not resultado.ok
    assert not alvo.exists()


def test_sandbox_nao_altera_o_banco(banco):
    Sandbox(SandboxConfig()).run(banco, "SELECT * FROM aluno")
    assert _contagem(banco) == 3


def test_sandbox_sinaliza_linhas_alem_do_limite(banco):
    sandbox = Sandbox(SandboxConfig())
    cortado = sandbox.run(banco, "SELECT id FROM aluno ORDER BY id", limite=2)
    assert cortado.ok and cortado.truncated and cortado.rows == [(1,), (2,)]
    exato = sandbox.run(banco, "SELECT id FROM aluno ORDER BY id", limite=3)
    assert exato.ok and not exato.truncated and len(exato.rows) == 3
    inteiro = sandbox.run(banco, "SELECT id FROM aluno")
    assert inteiro.ok and not inteiro.truncated and len(inteiro.rows) == 3


def test_sandbox_preserva_valores(banco):
    # Sem arredondamento e com BLOB intacto: a comparação oficial vê os valores
    # exatamente como o SQLite os devolve.
    resultado = Sandbox(SandboxConfig()).run(banco, "SELECT 0.1 + 0.2, X'00FF10', 7, NULL")
    assert resultado.rows == [(0.1 + 0.2, b"\x00\xff\x10", 7, None)]


def test_sandbox_interrompe_consulta_sem_fim(banco):
    infinita = "WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r) SELECT count(*) FROM r"
    resultado = Sandbox(SandboxConfig(timeout_s=2.0)).run(banco, infinita)
    assert not resultado.ok and resultado.timed_out


# ------------------------------------------------------------ estatística


def test_mcnemar_conta_discordantes():
    a = [True, True, False, True, False]
    b = [True, False, False, False, True]
    r = mcnemar_exact(a, b)
    assert r.only_a == 2 and r.only_b == 1
    assert r.discordant == 3
    assert 0.0 <= r.p_value <= 1.0


def test_mcnemar_sem_discordancia_da_p_um():
    a = b = [True, False, True]
    assert mcnemar_exact(a, b).p_value == 1.0


def test_bootstrap_devolve_intervalo_contendo_o_ponto():
    a = [True] * 30 + [False] * 10
    b = [True] * 20 + [False] * 20
    strata = ["db1"] * 20 + ["db2"] * 20
    ci = paired_bootstrap_diff(a, b, strata, resamples=400)
    assert ci.low <= ci.point <= ci.high


def test_diferencas_em_diferencas():
    n = 40
    pt_c4 = [True] * 30 + [False] * 10
    pt_c0 = [True] * 15 + [False] * 25
    en_c4 = [True] * 32 + [False] * 8
    en_c0 = [True] * 28 + [False] * 12
    did = difference_in_differences(
        pt_c4, pt_c0, en_c4, en_c0, ["db"] * n, resamples=400
    )
    # PT ganhou 37,5 pp e EN ganhou 10 pp: o contraste tem de ser positivo.
    assert did.delta_pp > 0
    assert did.low <= did.delta_pp <= did.high


# ------------------------------------------------------------- configuração


def test_hash_muda_com_parametro_e_nao_com_caminho():
    a = ExperimentConfig()
    b = ExperimentConfig()
    assert a.config_hash == b.config_hash

    b.paths.work_dir = "outro/lugar"
    assert a.config_hash == b.config_hash

    b.retrieval.k = 8
    assert a.config_hash != b.config_hash


# ------------------------------------------------------- integridade do repo


def test_todo_codigo_esta_versionado():
    """Todo .py sob src/ precisa estar rastreado pelo git.

    Existe porque um `.gitignore` com `data/` sem barra inicial engoliu o
    pacote `src/tcc_rag/data/` inteiro: o código funcionava na máquina de
    desenvolvimento e sumia no clone, com o erro aparecendo só no Colab.
    """
    import subprocess

    raiz = Path(__file__).resolve().parent.parent
    if not (raiz / ".git").exists():
        pytest.skip("fora de um repositório git")

    try:
        saida = subprocess.run(
            ["git", "ls-files", "src"],
            cwd=raiz, capture_output=True, text=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        pytest.skip("git indisponível")

    rastreados = {
        (raiz / linha).resolve()
        for linha in saida.stdout.splitlines()
        if linha.strip()
    }
    no_disco = {
        p.resolve()
        for p in (raiz / "src").rglob("*.py")
        if "__pycache__" not in p.parts
    }

    faltando = sorted(str(p.relative_to(raiz)) for p in no_disco - rastreados)
    assert not faltando, (
        "arquivos existem no disco mas não iriam no clone: " + ", ".join(faltando)
    )


# ------------------------------------------------------------- amostragem


def _itens_de_teste(n_bancos=4, por_banco=20, langs=("pt", "en")):
    from tcc_rag.schemas import TestItem

    itens = []
    for b in range(n_bancos):
        for i in range(por_banco):
            for lang in langs:
                itens.append(TestItem(
                    example_id=f"db{b}_ex{i}", db_id=f"db{b}", lang=lang,
                    question_raw="q", question_delex="q", gold_sql="SELECT 1",
                ))
    return itens


def test_fatiar_o_comeco_nao_e_amostrar():
    """O bug que motivou `subsample`: os itens vêm agrupados por banco, então
    pegar os primeiros N cobre só os primeiros bancos."""
    from tcc_rag.pipeline import subsample

    itens = _itens_de_teste()
    cabeca = {i.db_id for i in itens[:40]}
    amostra = {i.db_id for i in subsample(itens, 40)}

    assert len(cabeca) == 1, "fatiar o começo pega um banco só"
    assert len(amostra) == 4, "a amostra tem de cobrir todos os bancos"


def test_amostra_mantem_os_pares_de_idioma():
    from collections import Counter

    from tcc_rag.pipeline import subsample

    amostra = subsample(_itens_de_teste(), 40)
    por_exemplo = Counter(i.example_id for i in amostra)
    assert set(por_exemplo.values()) == {2}, "cada exemplo entra nos dois idiomas"


def test_amostra_e_deterministica_e_respeita_o_teto():
    from tcc_rag.pipeline import subsample

    itens = _itens_de_teste()
    a = subsample(itens, 40)
    b = subsample(itens, 40)
    assert [i.example_id for i in a] == [i.example_id for i in b]
    assert len(a) <= 40
    assert subsample(itens, 10_000) is itens


def test_digito_do_conjunto_distingue_amostras():
    from tcc_rag.pipeline import item_set_digest, subsample

    itens = _itens_de_teste()
    assert item_set_digest(subsample(itens, 40)) != item_set_digest(subsample(itens, 80))
    assert item_set_digest(subsample(itens, 40)) == item_set_digest(subsample(itens, 40))
