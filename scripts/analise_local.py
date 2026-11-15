"""Análise completa de um split a partir do pacote de resultados, fora do Colab.

Uso:
    python scripts/analise_local.py <pacote/artifacts> <spider_root> <split> <saida.json>

Faz o mesmo que `tcc-rag analise --split <split> --sensibilidade-distinct`
(mesma função `analise.analisar`, mesmos argumentos), mas lê os registros do
pacote baixado, e não do diretório de trabalho da configuração. Serve para dois
casos: a sessão do Colab terminou antes da etapa de análise, ou a análise
precisa ser refeita com a revisão corrente do código de análise. A geração não
é tocada: os registros são lidos como estão, e sua identidade (uma execução,
uma configuração, uma revisão de código limpa, linhas únicas, condições
pareadas) é conferida antes de qualquer cálculo.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from tcc_rag import analise
from tcc_rag.config import SandboxConfig
from tcc_rag.data.spider import load_schemas
from tcc_rag.evaluation.ex_oficial import AvaliadorEX
from tcc_rag.evaluation.sandbox import Sandbox
from tcc_rag.proveniencia import codigo_revisao

SEMENTE = 20260904  # data.seed de configs/confirmatorio.yaml


def main(art: Path, spider: Path, split: str, saida: Path) -> int:
    logs = sorted((art / "runs-v2").glob(f"*-{split}-*.jsonl"))
    if len(logs) != 1:
        raise SystemExit(f"esperado um único log de {split}; encontrados: {[p.name for p in logs]}")
    log = logs[0]
    rows = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]

    def unico(campo: str) -> str:
        valores = {r.get(campo) for r in rows}
        if len(valores) != 1:
            raise SystemExit(f"{campo} não é único no log: {sorted(map(str, valores))}")
        return next(iter(valores))

    run_id, config_hash, revisao, particao = (unico(c) for c in ("run_id", "config_hash", "codigo_revisao",
                                                                   "partition_digest"))
    if "+" in revisao:
        raise SystemExit(f"registros gerados com código modificado: {revisao}")
    if {r["split"] for r in rows} != {split}:
        raise SystemExit("o log contém linhas de outro split")
    chaves = Counter((r["condition"], r["lang"], r["example_id"]) for r in rows)
    if any(n > 1 for n in chaves.values()):
        raise SystemExit("linhas duplicadas no log")
    por_celula: dict[tuple[str, str], set[str]] = {}
    for c, l, e in chaves:
        por_celula.setdefault((c, l), set()).add(e)
    ids = list(por_celula.values())
    if any(i != ids[0] for i in ids[1:]):
        raise SystemExit("pareamento incompleto entre condições e idiomas")

    congelamento = art / "congelamento.json"
    if split == "test":
        reg = json.loads(congelamento.read_text(encoding="utf-8"))
        if reg["configuracao"]["config_hash"] != config_hash or reg["codigo_revisao"] != revisao:
            raise SystemExit("o log do teste não corresponde ao registro de congelamento")

    estratos = json.loads((art / "estratos_literais.json").read_text(encoding="utf-8"))
    manifesto = json.loads((art / "manifest.json").read_text(encoding="utf-8"))["examples"]
    familias = {e["example_id"]: e["family_id"] for e in manifesto}
    n_teste = sum(1 for e in manifesto if e["partition"] == "test")

    custo_rows: list[dict] = []
    for arquivo in sorted((art / "custo").glob(f"custo-{split}-{config_hash}-*.jsonl")):
        custo_rows += [json.loads(l) for l in arquivo.read_text(encoding="utf-8").splitlines() if l.strip()]

    schemas = load_schemas(spider)
    avaliador = AvaliadorEX(Sandbox(SandboxConfig()), keep_distinct=True)
    ex_distinct = {
        (r["condition"], r["lang"], r["example_id"]):
            avaliador.avaliar(schemas[r["db_id"]].sqlite_path, r["generated_sql"], r["gold_sql"]).correto
        for r in rows
    }

    res = analise.analisar(rows, estratos=estratos, custo_rows=custo_rows or None, semente=SEMENTE, n_teste=n_teste,
                           ex_distinct=ex_distinct, familias=familias)
    res["analise_local"] = {
        "log": log.name,
        "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
        "linhas": len(rows),
        "linhas_por_celula": {f"{c}|{l}": len(v) for (c, l), v in sorted(por_celula.items())},
        "run_id": run_id,
        "config_hash": config_hash,
        "partition_digest": particao,
        "codigo_revisao_da_geracao": revisao,
        "codigo_revisao_da_analise": codigo_revisao(),
        "medicoes_de_custo": len(custo_rows),
    }
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"split={split} linhas={len(rows)} exemplos={res['n_exemplos']} "
          f"inválidas={len(res['referencias_invalidas'])} custo={len(custo_rows)} -> {saida}")
    for chave, c in res["celulas"].items():
        print(f"  {chave:<6} EX {100 * c['ex']:5.1f}  n/exec {c['nao_executaveis']:>3}  "
              f"hit {100 * c['acerto_estrutural']:5.1f}  tok {c['prompt_tokens_total_medio']:6.0f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])))
