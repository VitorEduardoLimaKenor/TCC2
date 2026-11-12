"""Exemplo rastreável do mecanismo (F03): uma pergunta de validação, do texto à seleção.

Lê o log de uma rodada e o manifesto e reconstrói, para um item de VALIDAÇÃO em
português, o que cada condição fez: a pergunta e sua forma delexicalizada, as
demonstrações de C2 e C3, a consulta preliminar de C4, sua assinatura e a
reordenação dos N candidatos. Os escores são recalculados e conferidos com a
seleção gravada no log. A consulta de referência aparece só no campo de
avaliação: ela não participa de nenhuma seleção.

Uso:
    python scripts/exemplo_mecanismo.py <config.yaml> <log.jsonl> <saida.json> [example_id]

Sem `example_id`, escolhe o primeiro item (em ordem de identificador) em que a
delexicalização muda a pergunta e as seleções de C2, C3 e C4 são três conjuntos
diferentes, e informa quantos itens satisfazem o critério. Recusa itens que não
sejam da partição de validação.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tcc_rag.config import ExperimentConfig
from tcc_rag.data import partition as part
from tcc_rag.pipeline import build_demos, load_manifest
from tcc_rag.structure import signature as sig


def main(config: str, log: str, saida: str, example_id: str | None = None) -> int:
    cfg = ExperimentConfig.load(config)
    exemplos = load_manifest(cfg, warn=False)
    particao = {e.example_id: e.partition for e in exemplos}
    demos = {d.uid: d for d in build_demos(exemplos, partition=part.POOL)}
    linhas: dict[tuple[str, str], dict[str, dict]] = {}
    for texto in Path(log).read_text(encoding="utf-8").splitlines():
        if not texto.strip():
            continue
        r = json.loads(texto)
        if r["split"] != "val":
            raise SystemExit("o exemplo do mecanismo só pode vir da validação")
        linhas.setdefault((r["condition"], r["lang"]), {})[r["example_id"]] = r

    def conjuntos(eid: str) -> list[frozenset]:
        return [frozenset(linhas[(c, "pt")][eid]["prompt_demo_ids"]) for c in ("C2", "C3", "C4")]

    comuns = sorted(set(linhas[("C2", "pt")]) & set(linhas[("C3", "pt")]) & set(linhas[("C4", "pt")]))
    elegiveis = [e for e in comuns
                 if linhas[("C3", "pt")][e]["question_raw"] != linhas[("C3", "pt")][e]["question_delex"]
                 and len(set(conjuntos(e))) == 3 and not linhas[("C4", "pt")][e]["used_fallback"]]
    eid = example_id or (elegiveis[0] if elegiveis else None)
    if eid is None or particao.get(eid) != "val":
        raise SystemExit(f"item ausente ou fora da validação: {eid}")

    c2, c3, c4 = (linhas[(c, "pt")][eid] for c in ("C2", "C3", "C4"))
    lam = cfg.retrieval.lambda_question
    rascunho_sig = sig.signature(c4["preliminary_sql"])
    candidatos = []
    for pos, (uid, escore) in enumerate(zip(c4["retrieved_ids"], c4["retrieved_scores"]), start=1):
        d = demos[uid]
        estr = sig.similarity(rascunho_sig, d.signature)
        candidatos.append({"posicao_primeira_etapa": pos, "uid": uid, "pergunta": d.question_raw, "sql": d.sql,
                           "sim_pergunta": escore, "sim_estrutural": estr,
                           "escore": lam * escore + (1 - lam) * estr})
    ordem = sorted(range(len(candidatos)), key=lambda i: candidatos[i]["escore"], reverse=True)
    for final, i in enumerate(ordem, start=1):
        candidatos[i]["posicao_final"] = final
        candidatos[i]["selecionada"] = candidatos[i]["uid"] in c4["prompt_demo_ids"]
    recalculada = [candidatos[i]["uid"] for i in ordem[: cfg.retrieval.k]]

    def demo(uid: str) -> dict:
        d = demos[uid]
        return {"uid": uid, "pergunta": d.question_raw, "pergunta_delexicalizada": d.question_delex, "sql": d.sql}

    resultado = {
        "example_id": eid, "db_id": c3["db_id"], "particao": "val", "idioma": "pt",
        "itens_que_satisfazem_o_criterio": len(elegiveis), "escolhido_pelo_criterio": example_id is None,
        "configuracao": {"k": cfg.retrieval.k, "n_candidates": cfg.retrieval.n_candidates, "lambda_question": lam},
        "run_id": c3["run_id"], "config_hash": c3["config_hash"], "codigo_revisao": c3["codigo_revisao"],
        "pergunta": c3["question_raw"], "pergunta_delexicalizada": c3["question_delex"],
        "pergunta_en": next((r["question_raw"] for r in [linhas.get(("C3", "en"), {}).get(eid)] if r), None),
        "c2": {"demonstracoes": [demo(u) for u in c2["prompt_demo_ids"]], "gerado": c2["generated_sql"],
               "ex_correct": c2["ex_correct"], "acerto_estrutural": c2["structural_hit"]},
        "c3": {"demonstracoes": [demo(u) for u in c3["prompt_demo_ids"]], "gerado": c3["generated_sql"],
               "ex_correct": c3["ex_correct"], "acerto_estrutural": c3["structural_hit"]},
        "c4": {"consulta_preliminar": c4["preliminary_sql"], "assinatura_preliminar": rascunho_sig,
               "candidatos": candidatos, "selecao_recalculada_confere_com_o_log":
                   recalculada == list(c4["prompt_demo_ids"]),
               "demonstracoes": [demo(u) for u in c4["prompt_demo_ids"]], "gerado": c4["generated_sql"],
               "ex_correct": c4["ex_correct"], "acerto_estrutural": c4["structural_hit"]},
        "avaliacao_apenas": {"referencia_sql": c3["gold_sql"], "assinatura_referencia": sig.signature(c3["gold_sql"])},
    }
    Path(saida).write_text(json.dumps(resultado, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"item {eid} ({resultado['db_id']}); elegíveis: {len(elegiveis)}; "
          f"seleção de C4 recalculada confere: {resultado['c4']['selecao_recalculada_confere_com_o_log']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:5]))
