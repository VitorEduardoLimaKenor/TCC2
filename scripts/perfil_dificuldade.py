"""Perfil de dificuldade oficial do Spider e do recorte (figura F04, E10).

Classifica a consulta de referência de cada pergunta do Spider (train_spider e
dev) com `Evaluator.eval_hardness`, pelo mesmo adaptador usado na avaliação
(`tcc_rag.evaluation.em_oficial`), e cruza com a partição do manifesto.

Uso:
    python scripts/perfil_dificuldade.py <spider_root> <manifest.json> <saida.json>

Só lê o texto das consultas de referência e a partição: nenhuma predição,
nenhum resultado de execução.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from tcc_rag.evaluation.em_oficial import AvaliadorEM

NIVEIS = ("easy", "medium", "hard", "extra")


def main(spider_root: Path, manifesto: Path, saida: Path) -> int:
    avaliador = AvaliadorEM()
    por_id: dict[str, str | None] = {}
    banco: dict[str, str] = {}
    for arquivo in ("train_spider", "dev"):
        itens = json.loads((spider_root / f"{arquivo}.json").read_text(encoding="utf-8"))
        for i, item in enumerate(itens):
            uid = f"spider_{arquivo}_{i}"
            db = spider_root / "database" / item["db_id"] / f"{item['db_id']}.sqlite"
            try:
                por_id[uid] = avaliador.dificuldade(db, item["query"])
            except Exception:  # noqa: BLE001 - referência que o analisador oficial recusa
                por_id[uid] = None
            banco[uid] = item["db_id"]

    dados = json.loads(manifesto.read_text(encoding="utf-8"))
    particao = {e["example_id"]: e["partition"] for e in dados["examples"]}
    bancos_recorte = sorted({e["db_id"] for e in dados["examples"]})

    def contar(ids) -> dict[str, int]:
        c = Counter(por_id[i] or "sem_classificacao" for i in ids)
        return {k: c.get(k, 0) for k in (*NIVEIS, "sem_classificacao")} | {"n": sum(c.values())}

    treino = [i for i in por_id if i.startswith("spider_train_spider_")]
    dev = [i for i in por_id if i.startswith("spider_dev_")]
    grupos = {
        "spider_train": contar(treino),
        "spider_dev": contar(dev),
        "spider_train_e_dev": contar(por_id),
        "recorte": contar(particao),
        "recorte_pool": contar(i for i, p in particao.items() if p == "pool"),
        "recorte_val": contar(i for i, p in particao.items() if p == "val"),
        "recorte_test": contar(i for i, p in particao.items() if p == "test"),
        "bancos_do_recorte_todas_as_perguntas": contar(i for i in por_id if banco[i] in bancos_recorte),
    }
    resultado = {
        "avaliador": avaliador.identificador,
        "partition_digest": dados.get("partition_digest"),
        "manifesto_sha256": hashlib.sha256(manifesto.read_bytes()).hexdigest(),
        "bancos_do_recorte": bancos_recorte,
        "origem_do_recorte": {
            "train_spider": sum(1 for i in particao if i.startswith("spider_train_spider_")),
            "dev": sum(1 for i in particao if i.startswith("spider_dev_")),
        },
        "grupos": grupos,
        "por_banco_recorte": {
            b: contar(i for i in particao if banco[i] == b) for b in bancos_recorte
        },
    }
    saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for nome, g in grupos.items():
        total = g["n"] - g["sem_classificacao"]
        print(f"{nome:<40} n={g['n']:<5} " + "  ".join(
            f"{k}={g[k]} ({100 * g[k] / max(1, total):.1f}%)" for k in NIVEIS)
            + f"  sem={g['sem_classificacao']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*(Path(a) for a in sys.argv[1:4])))
