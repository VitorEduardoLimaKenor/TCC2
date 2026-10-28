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

Cada linha do log traz as métricas oficiais com o avaliador que as produziu
(`ex_*`, `em_*`, `dificuldade_oficial`) e, com nomes próprios, as medidas
aproximadas que não são oficiais (`em_canonico_proprio`, `dificuldade_aprox`).

Desde a E04, cada linha traz também a identidade da execução (`formato_log`,
`config_hash`, `partition_digest`, `split`, avaliadores e `indice_digest`), o
`item_sha256` do conteúdo do item e a proveniência (`codigo_revisao`,
`ambiente`). A retomada só pula linhas com a mesma identidade e o mesmo
conteúdo; o cache de rascunhos só devolve um rascunho com a mesma identidade
completa (`generation.rascunhos`).
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
from .evaluation.em_oficial import AvaliadorEM
from .evaluation.ex_oficial import AvaliadorEX
from .evaluation.sandbox import Sandbox
from .generation import rascunhos
from .generation.llm import Backend
from .generation.prompt import build_draft_prompt, build_prompt
from .logging_io import RunLog
from .data.normalize import extract_sql
from .pipeline import item_sha256
from .proveniencia import ambiente, codigo_revisao
from .retrieval.selectors import DemoSelector
from .schemas import Demo, RetrievalQuery, TestItem
from .structure import signature as sig

FORMATO_LOG = 2
"""Formato das linhas a partir da E04: identidade de retomada e proveniência."""

# Campos que identificam uma execução. Uma linha antiga do mesmo `run_id` com
# qualquer um deles diferente impede a retomada.
IDENTIDADE_RETOMADA = {
    "formato_log": "outro formato de log (anterior à E04?)",
    "config_hash": "outra configuração",
    "partition_digest": "outra partição",
    "split": "outro split",
    "ex_avaliador": "outro avaliador de EX",
    "em_avaliador": "outro avaliador de EM",
    "indice_digest": "outro índice de demonstrações",
}

# Marcas que substituem pergunta e DDL ao calcular o hash do template do
# prompt de rascunho: o que sobra é a instrução e os rótulos do idioma.
_MARCA_PERGUNTA = "<<PERGUNTA>>"
_MARCA_DDL = "<<DDL>>"


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
        indice_digest: str = "",
    ):
        self.cfg = cfg
        self.backend = backend
        self.selector = selector
        self.schemas = schemas
        self.log = log
        self.run_id = run_id
        self.partition_digest = partition_digest
        self.split = split
        self.indice_digest = indice_digest
        sem_tabela = sorted(db for db, s in schemas.items() if s.tables_entry is None)
        if sem_tabela:
            raise RuntimeError(
                "esquemas sem a entrada do tables.json, de onde o EM oficial tira as "
                f"chaves estrangeiras: {', '.join(sem_tabela[:5])}"
            )
        self.avaliador = AvaliadorEX(Sandbox(cfg.sandbox))
        self.avaliador_em = AvaliadorEM()
        self.identidade = {
            "formato_log": FORMATO_LOG,
            "config_hash": cfg.config_hash,
            "partition_digest": partition_digest,
            "split": split,
            "ex_avaliador": self.avaliador.identificador,
            "em_avaliador": self.avaliador_em.identificador,
            "indice_digest": indice_digest,
        }
        self._check_identidade()
        self.rascunhos = rascunhos.CacheRascunhos(cfg.work / "cache")
        self._rascunhos: dict[tuple[str, str], dict[str, Any]] = {}
        self._item_sha: dict[tuple[str, str], str] = {}

    # ------------------------------------------------------------ retomada

    def _check_identidade(self) -> None:
        """Impede retomar um log produzido sob outra identidade.

        A retomada pula um item quando `(run_id, condição, idioma, example_id)`
        já está no log. Nada disso muda quando mudam a partição, a configuração,
        o avaliador, o índice ou o texto da pergunta. Sem esta verificação, uma
        varredura nova com o `run_id` de uma antiga pularia todos os itens e
        reportaria os números antigos como novos. O texto de cada item é
        conferido à parte, em `_pendentes`.
        """
        divergencias: dict[str, set[str]] = {}
        revisoes: set[str] = set()
        for row in self.log.read():
            if row.get("run_id") != self.run_id:
                continue
            for campo in IDENTIDADE_RETOMADA:
                if row.get(campo) != self.identidade[campo]:
                    divergencias.setdefault(campo, set()).add(str(row.get(campo)))
            if row.get("codigo_revisao"):
                revisoes.add(row["codigo_revisao"])
        if divergencias:
            detalhe = "\n".join(
                f"  - {IDENTIDADE_RETOMADA[c]}: {c} gravado {', '.join(sorted(v))}; "
                f"atual {self.identidade[c]}"
                for c, v in divergencias.items()
            )
            raise RuntimeError(
                f"o log {self.log.path.name} contém linhas de outra identidade:\n{detalhe}\n"
                f"Arquive o arquivo ou use --run-id com outro nome. Retomar "
                f"misturaria dois experimentos."
            )
        outras = revisoes - {codigo_revisao()}
        if outras:
            import sys

            print(f"[retomada] AVISO: o log {self.log.path.name} tem linhas de outra revisão "
                  f"do código ({', '.join(sorted(outras))}); a atual é {codigo_revisao()}. "
                  "Cada linha registra a sua.", file=sys.stderr)

    def _sha_do_item(self, item: TestItem) -> str:
        chave = (item.example_id, item.lang)
        if chave not in self._item_sha:
            self._item_sha[chave] = item_sha256(item, self.schemas[item.db_id].ddl)
        return self._item_sha[chave]

    def _pendentes(self, items: list[TestItem], condition: str) -> list[TestItem]:
        """Itens ainda sem linha; recusa a linha gravada sobre outro conteúdo."""
        todo = []
        for item in items:
            if not self.log.has(self.run_id, condition, item.lang, item.example_id):
                todo.append(item)
                continue
            gravado = self.log.item_sha256(self.run_id, condition, item.lang, item.example_id)
            if gravado != self._sha_do_item(item):
                raise RuntimeError(
                    f"o log {self.log.path.name} tem {condition}/{item.lang}/{item.example_id} "
                    f"gravado sobre outro conteúdo (item_sha256 {gravado}; atual "
                    f"{self._sha_do_item(item)}): a pergunta, a referência ou o esquema mudou. "
                    "Use --run-id com outro nome."
                )
        return todo

    # ------------------------------------------------------------ rascunhos

    def prepare_drafts(self, items: list[TestItem]) -> None:
        """Consultas preliminares de C4: do cache, se a identidade conferir; senão, geradas.

        O rascunho não depende de k, N nem λ, e a varredura reutiliza o mesmo
        para toda a grade. A identidade (`generation.rascunhos`) cobre modelo,
        geração, template, pergunta, DDL e idioma; mudar qualquer um deles gera
        o rascunho de novo. Cada linha de C4 registra a chave e a origem.
        """
        g = self.cfg.generation
        pendentes = []
        reutilizados = 0
        for item in items:
            chave_item = (item.example_id, item.lang)
            if chave_item in self._rascunhos:
                continue
            ddl = self.schemas[item.db_id].ddl
            prompt = build_draft_prompt(item.question_raw, ddl, item.lang, g.max_context_tokens).text
            template = build_draft_prompt(_MARCA_PERGUNTA, _MARCA_DDL, item.lang, g.max_context_tokens).text
            ident = rascunhos.identidade(g, item.lang, item.question_raw, ddl, prompt, template)
            entrada = self.rascunhos.obter(rascunhos.chave(ident))
            if entrada is None:
                pendentes.append((item, ident, prompt))
            else:
                self._rascunhos[chave_item] = {**entrada, "origem": "cache"}
                reutilizados += 1

        if pendentes:
            outputs = self.backend.generate([prompt for _, _, prompt in pendentes])
            for (item, ident, _), gen in zip(pendentes, outputs):
                entrada = self.rascunhos.gravar(
                    ident, gen.text, gen.prompt_tokens, gen.completion_tokens, gen.latency_ms
                )
                self._rascunhos[(item.example_id, item.lang)] = {**entrada, "origem": "gerado"}
        if pendentes or reutilizados:
            print(f"[C4] rascunhos: {len(pendentes)} gerados, {reutilizados} do cache "
                  "com identidade conferida")
        self.selector.drafter = self._rascunho_sql

    def _rascunho_sql(self, query: RetrievalQuery) -> str:
        entrada = self._rascunhos.get((query.example_id, query.lang))
        if entrada is None:
            raise RuntimeError(f"rascunho ausente para {query.example_id}/{query.lang}: "
                               "prepare_drafts não foi chamado para este item")
        return extract_sql(entrada["texto"])

    # ---------------------------------------------------------------- laço

    def run_condition(self, items: list[TestItem], condition: str) -> int:
        todo = self._pendentes(items, condition)
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
            schema = self.schemas[item.db_id]
            rascunho = self._rascunhos.get((item.example_id, item.lang), {}) if condition == "C4" else {}
            veredito = self.avaliador.avaliar(schema.sqlite_path, predicted, item.gold_sql)
            em = self.avaliador_em.avaliar(
                schema.sqlite_path, schema.tables_entry, predicted, item.gold_sql
            )
            hit = metrics.structural_hit(
                [d.signature for d in selection.demos], item.gold_sql
            )

            self.log.append(
                {
                    "run_id": self.run_id,
                    "formato_log": FORMATO_LOG,
                    "config_hash": self.cfg.config_hash,
                    "partition_digest": self.partition_digest,
                    "split": self.split,
                    "indice_digest": self.indice_digest,
                    "item_sha256": self._sha_do_item(item),
                    "codigo_revisao": codigo_revisao(),
                    "ambiente": ambiente(),
                    "timestamp": time.time(),
                    "condition": condition,
                    "lang": item.lang,
                    "db_id": item.db_id,
                    "example_id": item.example_id,
                    "dificuldade_oficial": em.dificuldade,
                    "dificuldade_aprox": item.difficulty,
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
                    # Só em C4: de onde veio o rascunho e quanto ele custou ao ser gerado.
                    "rascunho_chave": rascunho.get("chave"),
                    "rascunho_origem": rascunho.get("origem"),
                    "rascunho_prompt_tokens": rascunho.get("prompt_tokens"),
                    "rascunho_completion_tokens": rascunho.get("completion_tokens"),
                    "rascunho_latency_ms": rascunho.get("latency_ms"),
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
                    "em_oficial": em.correto,
                    "em_status": em.situacao,
                    "em_avaliador": self.avaliador_em.identificador,
                    "em_componentes_divergentes": list(em.componentes_divergentes),
                    "em_erro": em.erro,
                    "em_canonico_proprio": metrics.em_canonico_proprio(predicted, item.gold_sql),
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
