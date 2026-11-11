"""Medição do custo por pergunta (E06).

A rodada principal gera em lote: o tempo gravado em cada linha é o do lote
dividido pelo número de itens, e não uma latência por pergunta. Esta passagem
mede o caminho fim a fim de **uma pergunta por vez**, do jeito que o sistema
atenderia um usuário:

1. seleção das demonstrações (vetorização da pergunta, busca e, em C4, a
   geração da consulta preliminar e a reordenação);
2. montagem do prompt;
3. geração da consulta final.

Em C4 a consulta preliminar é gerada dentro da própria medição, sem o cache de
rascunhos, de modo que o custo da chamada adicional entra no tempo e nos
tokens. O cache de vetores é um diretório novo e vazio: a vetorização da
pergunta é paga em toda medição. Uma geração de aquecimento, descartada,
precede as medições para que a carga do modelo e a inicialização da GPU não
contem como custo da primeira pergunta.

Nada aqui é avaliado quanto a acerto: a passagem mede custo, e a acurácia vem
da rodada principal. As consultas geradas são gravadas apenas para auditoria.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

from .config import ExperimentConfig
from .data.normalize import extract_sql
from .data.spider import DbSchema
from .generation.llm import Backend
from .generation.prompt import build_draft_prompt, build_prompt
from .proveniencia import ambiente, codigo_revisao
from .retrieval.selectors import DemoSelector
from .schemas import Generation, RetrievalQuery, TestItem

FORMATO_CUSTO = 1


def _ja_medidos(destino: Path) -> set[tuple[str, str, str]]:
    feitos: set[tuple[str, str, str]] = set()
    if not destino.is_file():
        return feitos
    for linha in destino.read_text(encoding="utf-8").splitlines():
        if not linha.strip():
            continue
        try:
            row = json.loads(linha)
        except json.JSONDecodeError:
            continue  # linha cortada por queda de sessão: o item é medido de novo
        feitos.add((row["condition"], row["lang"], row["example_id"]))
    return feitos


def medir(
    cfg: ExperimentConfig,
    items: list[TestItem],
    selector: DemoSelector,
    backend: Backend,
    schemas: dict[str, DbSchema],
    destino: Path,
    conditions: Iterable[str],
    identidade: dict[str, Any],
    aquecer: bool = True,
) -> int:
    """Mede cada (condição, item) com lote unitário e grava uma linha por medição.

    Retomável: o que já está em `destino` é pulado. Devolve o número de linhas
    escritas nesta chamada.
    """
    g = cfg.generation
    destino.parent.mkdir(parents=True, exist_ok=True)
    feitos = _ja_medidos(destino)
    rascunho_atual: dict[str, Any] = {}

    def rascunhar(query: RetrievalQuery) -> str:
        ddl = schemas[query.db_id].ddl
        prompt = build_draft_prompt(query.question_raw, ddl, query.lang, g.max_context_tokens).text
        inicio = time.perf_counter()
        gen: Generation = backend.generate([prompt])[0]
        rascunho_atual.update(
            ms=(time.perf_counter() - inicio) * 1000,
            prompt_tokens=gen.prompt_tokens,
            completion_tokens=gen.completion_tokens,
            gpu_mem_pico_mb=gen.gpu_mem_peak_mb,
        )
        return extract_sql(gen.text)

    selector.drafter = rascunhar

    if aquecer and items:
        primeiro = items[0]
        selector.embedder.encode_one(primeiro.question_delex, kind="query")
        backend.generate([
            build_prompt(primeiro.question_raw, schemas[primeiro.db_id].ddl, [], primeiro.lang,
                         g.max_context_tokens).text
        ])

    escritas = 0
    with destino.open("a", encoding="utf-8") as saida:
        for condition in conditions:
            for item in items:
                if (condition, item.lang, item.example_id) in feitos:
                    continue
                rascunho_atual.clear()
                t0 = time.perf_counter()
                selection = selector.select(item.as_query(), condition)
                t1 = time.perf_counter()
                built = build_prompt(
                    question=item.question_raw,
                    ddl=schemas[item.db_id].ddl,
                    demos=selection.demos,
                    lang=item.lang,
                    max_context_tokens=g.max_context_tokens,
                    reserve_for_completion=g.max_new_tokens + 64,
                )
                t2 = time.perf_counter()
                gen = backend.generate([built.text])[0]
                t3 = time.perf_counter()

                rascunho_ms = rascunho_atual.get("ms")
                picos = [p for p in (gen.gpu_mem_peak_mb, rascunho_atual.get("gpu_mem_pico_mb"))
                         if p is not None]
                row = {
                    **identidade,
                    "formato_custo": FORMATO_CUSTO,
                    "codigo_revisao": codigo_revisao(),
                    "ambiente": ambiente(),
                    "backend": backend.descriptor,
                    "timestamp": time.time(),
                    "condition": condition,
                    "lang": item.lang,
                    "db_id": item.db_id,
                    "example_id": item.example_id,
                    "tamanho_lote": 1,
                    "k_efetivo": len(built.demos),
                    "used_fallback": selection.used_fallback,
                    # tempos, em milissegundos
                    "selecao_ms": round((t1 - t0) * 1000, 2),
                    "rascunho_ms": None if rascunho_ms is None else round(rascunho_ms, 2),
                    "montagem_ms": round((t2 - t1) * 1000, 2),
                    "geracao_ms": round((t3 - t2) * 1000, 2),
                    "total_ms": round((t3 - t0) * 1000, 2),
                    # tokens: geração final e, em C4, a consulta preliminar
                    "prompt_tokens": gen.prompt_tokens,
                    "completion_tokens": gen.completion_tokens,
                    "rascunho_prompt_tokens": rascunho_atual.get("prompt_tokens"),
                    "rascunho_completion_tokens": rascunho_atual.get("completion_tokens"),
                    "prompt_tokens_total": gen.prompt_tokens + (rascunho_atual.get("prompt_tokens") or 0),
                    "completion_tokens_total": gen.completion_tokens
                    + (rascunho_atual.get("completion_tokens") or 0),
                    "gpu_mem_pico_mb": max(picos) if picos else None,
                    "generated_sql": extract_sql(gen.text),
                }
                saida.write(json.dumps(row, ensure_ascii=False) + "\n")
                saida.flush()
                escritas += 1
    return escritas


def resumir(linhas: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Percentis de latência e médias de tokens por condição e idioma."""
    import numpy as np

    grupos: dict[str, list[dict[str, Any]]] = {}
    for row in linhas:
        grupos.setdefault(f"{row['condition']}|{row['lang']}", []).append(row)

    resumo: dict[str, dict[str, Any]] = {}
    for chave in sorted(grupos):
        rows = grupos[chave]
        total = np.array([r["total_ms"] for r in rows], dtype=float)
        picos = [r["gpu_mem_pico_mb"] for r in rows if r.get("gpu_mem_pico_mb") is not None]
        rascunhos = [r["rascunho_ms"] for r in rows if r.get("rascunho_ms") is not None]
        resumo[chave] = {
            "n": len(rows),
            "total_ms_p50": float(np.percentile(total, 50)),
            "total_ms_p95": float(np.percentile(total, 95)),
            "total_ms_media": float(total.mean()),
            "selecao_ms_p50": float(np.percentile([r["selecao_ms"] for r in rows], 50)),
            "geracao_ms_p50": float(np.percentile([r["geracao_ms"] for r in rows], 50)),
            "rascunho_ms_p50": float(np.percentile(rascunhos, 50)) if rascunhos else None,
            "prompt_tokens_total_media": float(np.mean([r["prompt_tokens_total"] for r in rows])),
            "completion_tokens_total_media": float(np.mean([r["completion_tokens_total"] for r in rows])),
            "gpu_mem_pico_mb_max": max(picos) if picos else None,
        }
    return resumo
