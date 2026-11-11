"""Análise dos registros: contrastes, Holm, poder, H3, estratos e rubrica de erros."""

from __future__ import annotations

import random

import pytest

from tcc_rag import analise
from tcc_rag.evaluation import stats


def _linhas(n: int = 120, semente: int = 3, split: str = "val") -> list[dict]:
    rng = random.Random(semente)
    taxa = {"C0": 0.55, "C1": 0.58, "C2": 0.70, "C3": 0.72, "C4": 0.66}
    linhas = []
    for i in range(n):
        banco = f"db{i % 4}"
        dificuldade = ("easy", "medium", "hard", "extra")[i % 4]
        facilidade = rng.random()
        for lang in ("pt", "en"):
            for cond, p in taxa.items():
                acerto = facilidade < p + (0.05 if lang == "en" else 0.0)
                hit = cond != "C0" and rng.random() < 0.6
                linhas.append({
                    "split": split, "condition": cond, "lang": lang, "example_id": f"ex{i}",
                    "db_id": banco, "dificuldade_oficial": dificuldade,
                    "ex_correct": acerto, "ex_status": "coincide" if acerto else "difere",
                    "exec_ok": True, "em_oficial": acerto and i % 2 == 0, "em_status": "coincide",
                    "structural_hit": hit, "used_fallback": False, "k_efetivo": 0 if cond == "C0" else 4,
                    "n_demos_dropped": 0, "prompt_tokens": 400 + (0 if cond == "C0" else 250),
                    "completion_tokens": 40,
                    "rascunho_prompt_tokens": 400 if cond == "C4" else None,
                    "rascunho_completion_tokens": 40 if cond == "C4" else None,
                    "prompt_demo_ids": [] if cond == "C0" else [f"d{(i + j) % 9}" for j in range(4)],
                    "generated_sql": "SELECT a FROM t WHERE b = 1",
                    "gold_sql": "SELECT a FROM t WHERE b = 2",
                })
    return linhas


def test_analise_completa_e_pareada():
    linhas = _linhas()
    estratos = {"itens": {f"ex{i}": {"estrato": "LT" if i % 5 == 0 else "SL"} for i in range(120)}}
    r = analise.analisar(linhas, estratos=estratos, semente=11, n_teste=471)
    assert r["n_exemplos"] == 120 and r["idiomas"] == ["pt", "en"]
    assert set(r["celulas"]) == {f"{c}|{l}" for c in analise.CONDICOES for l in ("pt", "en")}
    cel = r["celulas"]["C4|pt"]
    assert cel["prompt_tokens_total_medio"] == 650 + 400 and cel["n"] == 120
    # o contraste reproduz o teste de McNemar e a diferença das acurácias
    c = r["contrastes"]["C3-C2|pt"]
    assert c["papel"] == "primaria"
    assert c["diferenca_pp"] == pytest.approx(100 * (r["celulas"]["C3|pt"]["ex"] - r["celulas"]["C2|pt"]["ex"]))
    assert c["ic95"][0] <= c["diferenca_pp"] <= c["ic95"][1]
    assert set(r["familia_confirmatoria_holm"]) == {"C3-C2|pt", "C4-C2|pt"}
    assert "h2_original_c4_c0" in r["diferencas_em_diferencas"]
    assert r["literal_traduzido"]["tamanho"] == {"LT": 24, "SL": 96}
    assert r["h3"]["C3|pt"]["tabela"]["hit_correta"] + r["h3"]["C3|pt"]["tabela"]["hit_incorreta"] > 0
    assert "C0|pt" not in r["h3"]
    assert r["poder"]["C3-C2|pt"]["n_teste"] == 471
    assert len(r["itens"]["ex_correct"]["C0|pt"]) == 120
    # literais diferentes, mesma condição: categoria "literal"
    assert r["erros"]["por_celula"]["C0|pt"]["literal"] == r["erros"]["por_celula"]["C0|pt"]["total"]


def test_referencia_invalida_sai_de_tudo():
    linhas = _linhas(n=40)
    for l in linhas:
        if l["example_id"] == "ex7" and l["condition"] == "C2" and l["lang"] == "en":
            l["ex_correct"] = None
            l["ex_status"] = "referencia_invalida"
    r = analise.analisar(linhas, semente=1)
    assert r["referencias_invalidas"] == ["ex7"] and r["n_exemplos"] == 39
    assert all(c["n"] == 39 for c in r["celulas"].values())


def test_pareamento_incompleto_e_recusado():
    linhas = [l for l in _linhas(n=20) if not (l["example_id"] == "ex3" and l["condition"] == "C4")]
    with pytest.raises(ValueError, match="pareamento incompleto"):
        analise.analisar(linhas)


def test_holm():
    h = analise.holm({"a": 0.03, "b": 0.02})
    assert h["b"]["p_holm"] == pytest.approx(0.04) and h["b"]["rejeita_h0"]
    assert h["a"]["p_holm"] == pytest.approx(0.04) and h["a"]["rejeita_h0"]
    h = analise.holm({"a": 0.20, "b": 0.03})
    assert not h["b"]["rejeita_h0"] and not h["a"]["rejeita_h0"]
    assert h["b"]["p_holm"] == pytest.approx(0.06) and h["a"]["p_holm"] == pytest.approx(0.20)


def test_poder_do_mcnemar_exato_confere_com_simulacao():
    n, taxa, delta = 200, 0.20, 0.10
    exato = analise.poder_mcnemar(n, taxa, delta)
    rng = random.Random(5)
    p_a, p_b = (taxa + delta) / 2, (taxa - delta) / 2
    rejeicoes = 0
    for _ in range(1500):
        a, b = [], []
        for _ in range(n):
            u = rng.random()
            if u < p_a:
                a.append(True); b.append(False)
            elif u < p_a + p_b:
                a.append(False); b.append(True)
            else:
                a.append(True); b.append(True)
        rejeicoes += stats.mcnemar_exact(a, b).p_value <= 0.05
    assert exato == pytest.approx(rejeicoes / 1500, abs=0.04)
    # sem diferença, o poder é o tamanho do teste: no máximo o nível nominal
    assert analise.poder_mcnemar(n, taxa, 0.0) <= 0.05
    dmd = analise.diferenca_minima_detectavel(471, 0.10)
    assert dmd is not None and analise.poder_mcnemar(471, 0.10, dmd) >= 0.80


@pytest.mark.parametrize("pred,ref,erro,esperado", [
    ("SELECT x FROM t", "SELECT a FROM t", "no such column: x", "identificador"),
    ("SELEC a", "SELECT a FROM t", 'near "SELEC": syntax error', "invalida"),
    ("SELECT a FROM t JOIN u ON t.id = u.id", "SELECT a FROM t", None, "tabelas"),
    ("SELECT count(*) FROM t", "SELECT a FROM t", None, "agregacao"),
    ("SELECT a FROM t WHERE c > 1", "SELECT a FROM t WHERE b > 1", None, "filtro"),
    ("SELECT a FROM t WHERE b = 'França'", "SELECT a FROM t WHERE b = 'France'", None, "literal"),
    ("SELECT b FROM t", "SELECT a FROM t", None, "projecao"),
    ("SELECT a FROM t ORDER BY a DESC", "SELECT a FROM t ORDER BY a", None, "ordem"),
    ("SELECT T1.a FROM t AS T1", "SELECT a FROM t", None, "outras"),
])
def test_rubrica_de_erros(pred, ref, erro, esperado):
    linha = {"ex_correct": False, "ex_status": "erro_predicao" if erro else "difere",
             "exec_ok": erro is None, "exec_error": erro, "generated_sql": pred, "gold_sql": ref}
    assert analise.classificar_erro(linha) == esperado
    assert analise.classificar_erro({**linha, "ex_correct": True}) is None
