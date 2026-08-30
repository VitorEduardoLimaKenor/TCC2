"""Varredura da grade de hiperparâmetros (Seção 4.6.3).

Fase de validação: determina o número de demonstrações $k$, o número de
candidatos $N$ da primeira etapa e o peso $\\lambda$ da reordenação. Opera
exclusivamente sobre o conjunto de validação — o teste permanece selado até o
congelamento da configuração.

A lógica vive aqui, e não na linha de comando, por dois motivos: o modelo é
carregado uma única vez para toda a grade, e o laço fica testável com
componentes falsos, sem GPU.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .config import ExperimentConfig
from .evaluation import metrics
from .logging_io import RunLog
from .retrieval.selectors import DemoSelector
from .runner import Runner
from .schemas import Demo, TestItem


@dataclass
class SweepRow:
    condicao: str
    lang: str
    k: int
    lambda_question: float
    n_candidates: int
    ex: float
    hit: float
    fallback: float
    tokens: float
    n_itens: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "condicao": self.condicao,
            "lang": self.lang,
            "k": self.k,
            "lambda": self.lambda_question,
            "n_candidates": self.n_candidates,
            "ex": self.ex,
            "hit": self.hit,
            "fallback": self.fallback,
            "tokens": self.tokens,
            "n_itens": self.n_itens,
        }


def combos(
    ks: Sequence[int], lambdas: Sequence[float], ns: Sequence[int]
) -> list[tuple[int, float, int]]:
    return [(k, lam, n) for k in ks for lam in lambdas for n in ns]


def run_sweep(
    cfg: ExperimentConfig,
    items: list[TestItem],
    pool: list[Demo],
    schemas: dict,
    index,
    embedder,
    backend,
    partition_digest: str,
    ks: Sequence[int],
    lambdas: Sequence[float],
    ns: Sequence[int],
    conditions: Iterable[str] = ("C4",),
    on_progress: Callable[[str], None] | None = None,
) -> list[SweepRow]:
    """Percorre a grade e devolve uma linha por (condição, idioma, combinação).

    Cada combinação escreve seu próprio log, identificado pelos parâmetros que
    a produziram. Isso mantém a varredura retomável e permite auditar qualquer
    ponto da grade depois.
    """
    saida: list[SweepRow] = []
    grade = combos(ks, lambdas, ns)

    for k, lam, n in grade:
        atual = copy.deepcopy(cfg)
        atual.retrieval.k = k
        atual.retrieval.lambda_question = lam
        atual.retrieval.n_candidates = n

        seletor = DemoSelector(atual, index, embedder, pool)

        for condicao in conditions:
            run_id = f"sweep-{condicao}-k{k}-l{int(round(lam * 100))}-n{n}"
            log = RunLog(cfg.work / f"runs/{run_id}.jsonl")
            Runner(
                atual, backend, seletor, schemas, log, run_id, partition_digest
            ).run(items, (condicao,))

            for chave, valor in metrics.summarize(log.rows()).items():
                cond, _, lang = chave.partition("|")
                saida.append(
                    SweepRow(
                        condicao=cond,
                        lang=lang,
                        k=k,
                        lambda_question=lam,
                        n_candidates=n,
                        ex=valor["ex"],
                        hit=valor["structural_hit"],
                        fallback=valor["fallback_rate"],
                        tokens=valor["prompt_tokens_mean"],
                        n_itens=valor["n"],
                    )
                )
            if on_progress:
                ultimo = saida[-1]
                on_progress(
                    f"  k={k} λ={lam} N={n} {condicao}: "
                    f"EX={ultimo.ex * 100:.1f}%  hit@k={ultimo.hit * 100:.1f}%"
                )

    return saida


def render(linhas: list[SweepRow]) -> str:
    """Tabela ordenada por acurácia, para leitura direta no terminal."""
    ordenadas = sorted(linhas, key=lambda l: -l.ex)
    cabecalho = (
        f"{'cond':<5}{'lang':<5}{'k':>3}{'lambda':>8}{'N':>5}"
        f"{'EX':>8}{'hit@k':>8}{'fallb':>8}{'tokens':>8}{'n':>5}"
    )
    corpo = [
        f"{l.condicao:<5}{l.lang:<5}{l.k:>3}{l.lambda_question:>8.2f}"
        f"{l.n_candidates:>5}{l.ex * 100:>7.1f}%{l.hit * 100:>7.1f}%"
        f"{l.fallback * 100:>7.1f}%{l.tokens:>8.0f}{l.n_itens:>5}"
        for l in ordenadas
    ]
    return "\n".join([cabecalho, *corpo])


def best(linhas: list[SweepRow], lang: str = "pt", condicao: str = "C4") -> SweepRow | None:
    candidatas = [l for l in linhas if l.lang == lang and l.condicao == condicao]
    return max(candidatas, key=lambda l: l.ex) if candidatas else None
