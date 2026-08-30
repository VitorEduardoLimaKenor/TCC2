"""Seleção de demonstrações: as cinco condições experimentais (Seção 4.6.2).

Todas as condições passam por esta única função. É deliberado: se cada condição
tivesse seu próprio caminho até o prompt, elas divergiriam em detalhes que o
experimento não controla, e a comparação perderia sentido. O que varia aqui é
exclusivamente **quais** demonstrações entram; o prompt é montado depois, pelo
mesmo código, em `generation.prompt`.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Callable, Protocol

from ..config import ExperimentConfig
from ..schemas import Demo, RetrievalQuery, Selection
from ..structure import signature as sig


class Drafter(Protocol):
    """Produz a consulta preliminar de C4: uma chamada ao modelo sem
    demonstrações. Recebe a pergunta original, nunca o gabarito."""

    def __call__(self, query: RetrievalQuery) -> str: ...


class DemoSelector:
    def __init__(
        self,
        cfg: ExperimentConfig,
        index,
        embedder,
        pool: list[Demo],
        drafter: Drafter | None = None,
    ):
        self.cfg = cfg
        self.index = index
        self.embedder = embedder
        self.drafter = drafter
        self._pool_by_db_lang: dict[tuple[str, str], list[Demo]] = defaultdict(list)
        for demo in pool:
            self._pool_by_db_lang[(demo.db_id, demo.lang)].append(demo)
        self._fixed_cache: dict[tuple[str, str], list[Demo]] = {}

    # ------------------------------------------------------------ C1

    def _fixed(self, db_id: str, lang: str) -> list[Demo]:
        """C1: k exemplos do banco, sorteados uma vez com semente registrada.

        O sorteio não depende da pergunta — é esse o ponto da condição. Ela
        separa o efeito de *ver o domínio* do efeito de receber exemplos
        pertinentes à pergunta formulada.
        """
        key = (db_id, lang)
        if key not in self._fixed_cache:
            candidates = sorted(
                self._pool_by_db_lang.get(key, []), key=lambda d: d.uid
            )
            rng = random.Random(f"{self.cfg.retrieval.fixed_demo_seed}|{db_id}|{lang}")
            rng.shuffle(candidates)
            self._fixed_cache[key] = candidates[: self.cfg.retrieval.k]
        return list(self._fixed_cache[key])

    # ------------------------------------------------------- busca vetorial

    def _vector_search(
        self, text: str, query: RetrievalQuery, limit: int
    ) -> list[tuple[Demo, float]]:
        vector = self.embedder.encode_one(text, kind="query")
        return self.index.search(
            vector=vector,
            db_id=query.db_id,
            lang=query.lang,
            partition="pool",
            limit=limit,
        )

    # ------------------------------------------------------------- C4

    def _two_stage(self, query: RetrievalQuery) -> Selection:
        k = self.cfg.retrieval.k
        n = max(self.cfg.retrieval.n_candidates, k)
        lam = self.cfg.retrieval.lambda_question

        hits = self._vector_search(query.question_delex, query, limit=n)
        candidate_ids = [d.uid for d, _ in hits]
        candidate_scores = [s for _, s in hits]

        if self.drafter is None:
            raise RuntimeError("C4 exige um gerador de consulta preliminar")

        draft = self.drafter(query)
        draft_sig = sig.signature(draft)

        if draft_sig is None:
            # Desvio previsto na Seção 4.3.3: mantém a ordenação da primeira
            # etapa e registra o evento. A taxa é métrica de robustez.
            return Selection(
                demos=[d for d, _ in hits[:k]],
                candidate_ids=candidate_ids,
                candidate_scores=candidate_scores,
                preliminary_sql=draft,
                parse_ok=False,
                signature=None,
                used_fallback=True,
            )

        rescored: list[tuple[Demo, float]] = []
        for demo, question_score in hits:
            structural = sig.similarity(draft_sig, demo.signature)
            rescored.append((demo, lam * question_score + (1.0 - lam) * structural))
        rescored.sort(key=lambda pair: pair[1], reverse=True)

        return Selection(
            demos=[d for d, _ in rescored[:k]],
            candidate_ids=candidate_ids,
            candidate_scores=candidate_scores,
            preliminary_sql=draft,
            parse_ok=True,
            signature=draft_sig,
            used_fallback=False,
        )

    # ------------------------------------------------------------ despacho

    def select(self, query: RetrievalQuery, condition: str) -> Selection:
        k = self.cfg.retrieval.k

        if condition == "C0":
            return Selection()

        if condition == "C1":
            return Selection(demos=self._fixed(query.db_id, query.lang))

        if condition == "C2":
            hits = self._vector_search(query.question_raw, query, limit=k)
            return Selection(
                demos=[d for d, _ in hits],
                candidate_ids=[d.uid for d, _ in hits],
                candidate_scores=[s for _, s in hits],
            )

        if condition == "C3":
            hits = self._vector_search(query.question_delex, query, limit=k)
            return Selection(
                demos=[d for d, _ in hits],
                candidate_ids=[d.uid for d, _ in hits],
                candidate_scores=[s for _, s in hits],
            )

        if condition == "C4":
            return self._two_stage(query)

        raise ValueError(f"condição desconhecida: {condition}")
