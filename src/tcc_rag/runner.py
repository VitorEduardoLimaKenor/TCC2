"""Orquestração de uma execução completa.

A ordem das operações é escolhida para o caso do experimento confirmatório, em
que ~3.000 gerações precisam caber numa sessão de GPU:

1. os rascunhos de C4 são gerados uma vez, em lote, e cacheados — eles não
   dependem de k nem de λ, então recalculá-los na varredura da validação seria
   desperdício puro;
2. cada condição monta **todos** os seus prompts antes de gerar, e gera em um
   único lote, o que permite ao vLLM manter a GPU ocupada;
3. o resultado da consulta de referência é cacheado por (banco, SQL), porque
   ele se repete em todas as cinco condições e nos dois idiomas (em
   `evaluation.ex_oficial`, que decide o acerto pelo avaliador oficial).
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

from .config import ExperimentConfig
from .data.spider import DbSchema
from .evaluation import metrics
from .evaluation.ex_oficial import AvaliadorEX
from .evaluation.sandbox import Sandbox
from .generation.llm import Backend
from .generation.prompt import build_draft_prompt, build_prompt
from .logging_io import RunLog
from .data.normalize import extract_sql
from .retrieval.selectors import DemoSelector
from .schemas import Demo, TestItem
from .structure import signature as sig


class Runner:
    def __init__(
        self,
        cfg: ExperimentConfig,
        backend: Backend,
        selector: DemoSelector,
        schemas: dict[str, DbSchema],
        log: RunLog,
        run_id: str,
        partition_digest: str = "",
        split: str = "",
    ):
        self.cfg = cfg
        self.backend = backend
        self.selector = selector
        self.schemas = schemas
        self.log = log
        self.run_id = run_id
        self.partition_digest = partition_digest
        self.split = split
        self._check_partition()
        self.avaliador = AvaliadorEX(Sandbox(cfg.sandbox))
        self._draft_cache: dict[tuple[str, str], str] = self._load_drafts()

    # -------------------------------------------------- cache de rascunhos

    @property
    def _draft_path(self) -> Path:
        """Cache em disco dos rascunhos de C4.

        O rascunho depende apenas da pergunta e do modelo — não de k, de N nem
        de λ. Sem persistência, a varredura da validação regeneraria todos eles
        a cada combinação da grade, que é o desperdício mais caro do
        experimento: nove combinações sobre cem itens custariam novecentas
        gerações que já foram feitas.
        """
        import hashlib

        chave = f"{self.cfg.generation.model}@{self.cfg.generation.revision or 'main'}"
        marca = hashlib.sha1(chave.encode("utf-8")).hexdigest()[:12]
        caminho = self.cfg.work / "cache" / f"drafts_{marca}.json"
        caminho.parent.mkdir(parents=True, exist_ok=True)
        return caminho

    def _load_drafts(self) -> dict[tuple[str, str], str]:
        import json

        if not self._draft_path.exists():
            return {}
        try:
            bruto = json.loads(self._draft_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        saida: dict[tuple[str, str], str] = {}
        for chave, sql in bruto.items():
            example_id, _, lang = chave.rpartition("|")
            if example_id and lang:
                saida[(example_id, lang)] = sql
        return saida

    def _save_drafts(self) -> None:
        import json

        payload = {f"{eid}|{lang}": sql for (eid, lang), sql in self._draft_cache.items()}
        self._draft_path.write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

    def _check_partition(self) -> None:
        """Impede retomar um log produzido sob outra partição.

        A retomada compara `(run_id, condição, idioma, example_id)`. Nada disso
        muda quando a partição muda: um mesmo exemplo pode ter saído do pool e
        entrado no teste, e as linhas antigas seriam aceitas como válidas. O
        resultado seria uma mistura de dois experimentos, sem nenhum sinal.
        """
        if not self.partition_digest:
            return
        anteriores = {
            row.get("partition_digest")
            for row in self.log.read()
            if row.get("run_id") == self.run_id
        }
        anteriores.discard(None)
        divergentes = anteriores - {self.partition_digest}
        if divergentes:
            raise RuntimeError(
                f"o log {self.log.path.name} contém linhas de outra partição "
                f"({', '.join(sorted(divergentes))}); a atual é "
                f"{self.partition_digest}.\n"
                f"Apague o arquivo ou use --run-id com outro nome. Retomar "
                f"misturaria dois experimentos."
            )

    # ------------------------------------------------------------ rascunhos

    def prepare_drafts(self, items: list[TestItem]) -> None:
        """Gera em lote as consultas preliminares usadas por C4."""
        # O seletor precisa do drafter mesmo quando o cache já está completo,
        # senão uma execução retomada quebraria ao chegar em C4.
        self.selector.drafter = lambda q: self._draft_cache.get((q.example_id, q.lang), "")

        pending = [
            item
            for item in items
            if (item.example_id, item.lang) not in self._draft_cache
        ]
        if not pending:
            return

        prompts = [
            build_draft_prompt(
                item.question_raw,
                self.schemas[item.db_id].ddl,
                item.lang,
                self.cfg.generation.max_context_tokens,
            ).text
            for item in pending
        ]

        print(f"[C4] gerando {len(pending)} rascunhos "
              f"({len(self._draft_cache)} já em cache)")
        outputs = self.backend.generate(prompts)
        for item, gen in zip(pending, outputs):
            self._draft_cache[(item.example_id, item.lang)] = extract_sql(gen.text)
        self._save_drafts()

    # ---------------------------------------------------------------- laço

    def run_condition(self, items: list[TestItem], condition: str) -> int:
        todo = [
            item
            for item in items
            if not self.log.has(self.run_id, condition, item.lang, item.example_id)
        ]
        if not todo:
            return 0

        if condition == "C4":
            self.prepare_drafts(todo)

        # ---- seleção e montagem, para todos os itens antes de gerar
        selections = []
        prompts = []
        for item in tqdm(todo, desc=f"{condition} · selecionando", leave=False):
            began = time.perf_counter()
            selection = self.selector.select(item.as_query(), condition)
            select_ms = (time.perf_counter() - began) * 1000
            built = build_prompt(
                question=item.question_raw,
                ddl=self.schemas[item.db_id].ddl,
                demos=selection.demos,
                lang=item.lang,
                max_context_tokens=self.cfg.generation.max_context_tokens,
                reserve_for_completion=self.cfg.generation.max_new_tokens + 64,
            )
            selections.append((item, selection, built, select_ms))
            prompts.append(built.text)

        # ---- geração em lote
        outputs = self.backend.generate(prompts)

        # ---- avaliação
        written = 0
        for (item, selection, built, select_ms), gen in zip(
            tqdm(selections, desc=f"{condition} · avaliando", leave=False), outputs
        ):
            predicted = extract_sql(gen.text)
            veredito = self.avaliador.avaliar(
                self.schemas[item.db_id].sqlite_path, predicted, item.gold_sql
            )
            em_correct = metrics.exact_match(predicted, item.gold_sql)
            hit = metrics.structural_hit(
                [d.signature for d in selection.demos], item.gold_sql
            )

            self.log.append(
                {
                    "run_id": self.run_id,
                    "config_hash": self.cfg.config_hash,
                    "partition_digest": self.partition_digest,
                    "split": self.split,
                    "timestamp": time.time(),
                    "condition": condition,
                    "lang": item.lang,
                    "db_id": item.db_id,
                    "example_id": item.example_id,
                    "difficulty": item.difficulty,
                    "question_raw": item.question_raw,
                    "question_delex": item.question_delex,
                    "gold_sql": item.gold_sql,
                    "gold_exec_ok": veredito.situacao != "referencia_invalida",
                    "retrieved_ids": selection.candidate_ids,
                    "retrieved_scores": [round(s, 5) for s in selection.candidate_scores],
                    "selected_demo_ids": [d.uid for d in selection.demos],
                    "preliminary_sql": selection.preliminary_sql,
                    "parse_ok": selection.parse_ok,
                    "structural_signature": selection.signature,
                    "used_fallback": selection.used_fallback,
                    "structural_hit": hit,
                    "prompt_sha256": built.sha256,
                    "n_demos_used": built.n_demos_used,
                    "n_demos_dropped": built.n_demos_dropped,
                    "prompt_tokens": gen.prompt_tokens,
                    "completion_tokens": gen.completion_tokens,
                    "raw_output": gen.text[:2000],
                    "generated_sql": predicted,
                    "exec_ok": veredito.pred_executou,
                    "exec_error": veredito.erro,
                    "exec_timeout": veredito.situacao == "tempo_predicao",
                    "ex_correct": veredito.correto,
                    "ex_status": veredito.situacao,
                    "ex_avaliador": self.avaliador.identificador,
                    "ex_ordem_importa": veredito.ordem_importa,
                    "ex_linhas_ref": veredito.linhas_ref,
                    "ex_linhas_pred": veredito.linhas_pred,
                    "ex_ambos_vazios": veredito.ambos_vazios,
                    "em_correct": em_correct,
                    "select_ms": round(select_ms, 2),
                    "latency_ms": round(gen.latency_ms, 2),
                    "backend": self.backend.descriptor,
                }
            )
            written += 1
        return written

    def run(self, items: list[TestItem], conditions: Iterable[str]) -> dict[str, int]:
        report: dict[str, int] = {}
        for condition in conditions:
            report[condition] = self.run_condition(items, condition)
        return report
