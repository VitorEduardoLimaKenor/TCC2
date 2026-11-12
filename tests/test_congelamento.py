"""Portão de sanidade e registro de congelamento: o teste só abre depois dele."""

from __future__ import annotations

import json

import pytest

from tcc_rag import congelamento
from tcc_rag.cli import _exigir_congelamento
from tcc_rag.config import ExperimentConfig


def _linhas(n: int = 12) -> list[dict]:
    linhas = []
    for i in range(n):
        for lang in ("pt", "en"):
            for cond in ("C0", "C1", "C2", "C3", "C4"):
                linhas.append({
                    "run_id": "val", "split": "val", "condition": cond, "lang": lang, "example_id": f"e{i}",
                    "partition_digest": "d", "indice_digest": "i", "template_sha256": "t",
                    "ex_avaliador": "ex", "em_avaliador": "em", "codigo_revisao": "abc", "ambiente": {"gpu": "T4"},
                    "ex_correct": i % 3 != 0, "ex_status": "coincide" if i % 3 else "difere",
                    "k_efetivo": 0 if cond == "C0" else 4, "used_fallback": False,
                    "prompt_no_limite_de_entrada": False,
                    "rascunho_prompt_tokens": 300 if cond == "C4" else None,
                })
    return linhas


def test_validacao_completa_passa_no_portao():
    c = congelamento.conferir_validacao(_linhas(), 12)
    assert c["aprovado"], c["falhas"]
    assert c["linhas"] == 120 and c["prompts_com_k_completo"] == 1.0


@pytest.mark.parametrize("estrago,trecho", [
    (lambda ls: [l for l in ls if not (l["condition"] == "C4" and l["lang"] == "en")], "ausentes"),
    (lambda ls: ls[:-1], "linhas por condição"),
    (lambda ls: [{**l, "prompt_no_limite_de_entrada": True} if l["example_id"] == "e1" else l for l in ls], "limite de entrada"),
    (lambda ls: [{**l, "ex_status": "erro_predicao", "ex_correct": False} if l["condition"] == "C2" else l for l in ls],
     "não executáveis"),
    (lambda ls: [{**l, "used_fallback": True} if l["condition"] == "C4" else l for l in ls], "ordenação inicial"),
    (lambda ls: [{**l, "rascunho_prompt_tokens": None} for l in ls], "custo da consulta preliminar"),
    (lambda ls: [{**l, "ex_correct": False, "ex_status": "difere"} if l["condition"] == "C0" else l for l in ls],
     "faixa de sanidade"),
    (lambda ls: [{**l, "k_efetivo": 2} if l["condition"] == "C3" and l["example_id"] in ("e1", "e2") else l for l in ls],
     "demonstrações"),
])
def test_portao_recusa_validacao_com_defeito(estrago, trecho):
    c = congelamento.conferir_validacao(estrago(_linhas()), 12)
    assert not c["aprovado"] and any(trecho in f for f in c["falhas"]), c["falhas"]


def test_registro_e_gravado_uma_vez_e_sela_o_teste(tmp_path, monkeypatch):
    cfg = ExperimentConfig()
    cfg.paths.work_dir = str(tmp_path / "artifacts")
    cfg.paths.questions_pt = str(tmp_path / "q.json")
    config = tmp_path / "c.yaml"
    cfg.dump(config)

    # sem registro, o teste está selado
    with pytest.raises(ValueError, match="selado"):
        _exigir_congelamento(cfg)

    # sem a análise da validação (poder), não congela
    pode, reg = congelamento.congelar(cfg, str(config), _linhas(), 12, 30, "abc123")
    assert not pode and "poder" in reg["conferencia"]["falhas"][0]

    analise = cfg.work / "analise" / f"analise-val-{cfg.config_hash}.json"
    analise.parent.mkdir(parents=True)
    analise.write_text(json.dumps({"poder": {"C3-C2|pt": {"taxa_discordancia_observada": 0.1}}}), encoding="utf-8")

    # revisão suja não congela
    pode, reg = congelamento.congelar(cfg, str(config), _linhas(), 12, 30, "abc123+deadbeef")
    assert not pode and "limpa" in reg["conferencia"]["falhas"][0]

    pode, reg = congelamento.congelar(cfg, str(config), _linhas(), 12, 30, "abc123")
    assert pode and reg["teste"]["linhas_esperadas"] == 300 and reg["validacao"]["poder"]
    gravado = (cfg.work / congelamento.ARQUIVO).read_text(encoding="utf-8")

    # retomada com a mesma revisão: confere e não regrava
    pode, _ = congelamento.congelar(cfg, str(config), _linhas(), 12, 30, "abc123")
    assert pode and (cfg.work / congelamento.ARQUIVO).read_text(encoding="utf-8") == gravado

    # outra revisão do código ou outra configuração: recusa
    pode, reg = congelamento.congelar(cfg, str(config), _linhas(), 12, 30, "zzz999")
    assert not pode and reg["_conferencia_da_retomada"]["divergencias"] == ["codigo_revisao"]

    import tcc_rag.proveniencia as prov
    monkeypatch.setattr(prov, "codigo_revisao", lambda: "abc123")
    _exigir_congelamento(cfg)  # mesma configuração e revisão: abre
    monkeypatch.setattr(prov, "codigo_revisao", lambda: "outra")
    with pytest.raises(ValueError, match="revisão do código mudou"):
        _exigir_congelamento(cfg)
    cfg.retrieval.k = 8
    monkeypatch.setattr(prov, "codigo_revisao", lambda: "abc123")
    with pytest.raises(ValueError, match="configuração mudou"):
        _exigir_congelamento(cfg)
