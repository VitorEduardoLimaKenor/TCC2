"""Procedimentos estatísticos (Seção 4.8).

Após a redesignação registrada na validação, C3 contra C2 em português é a
comparação primária. C4 contra C2 preserva H1 como comparação secundária. O teste
exato de McNemar é bilateral e o bootstrap mantém o pareamento e os bancos.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Sequence


@dataclass
class McNemarResult:
    n: int
    both_correct: int
    only_a: int
    only_b: int
    both_wrong: int
    diff_pp: float
    p_value: float

    @property
    def discordant(self) -> int:
        return self.only_a + self.only_b

    def render(self) -> str:
        return (
            f"n={self.n}  a&b={self.both_correct}  só A={self.only_a}  "
            f"só B={self.only_b}  nenhum={self.both_wrong}\n"
            f"discordantes={self.discordant}  Δ={self.diff_pp:+.2f} pp  p={self.p_value:.4g}"
        )


@dataclass
class BootstrapCI:
    point: float
    low: float
    high: float
    level: float = 0.95
    resamples: int = 10000

    def render(self) -> str:
        return f"{self.point:+.2f} pp  IC{int(self.level*100)}% [{self.low:+.2f}, {self.high:+.2f}]"

    @property
    def excludes_zero(self) -> bool:
        return (self.low > 0) or (self.high < 0)


def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> McNemarResult:
    """Teste exato de McNemar, bilateral.

    `a` e `b` são vetores de acerto/erro sobre **as mesmas perguntas**, na
    mesma ordem. O valor-p vem da binomial sobre os pares discordantes; o teste
    ignora, por construção, os casos em que as duas condições concordam.
    """
    if len(a) != len(b):
        raise ValueError("as duas condições precisam cobrir as mesmas perguntas")

    both = only_a = only_b = neither = 0
    for x, y in zip(a, b):
        if x and y:
            both += 1
        elif x and not y:
            only_a += 1
        elif y and not x:
            only_b += 1
        else:
            neither += 1

    n = len(a)
    discordant = only_a + only_b

    if discordant == 0:
        p_value = 1.0
    else:
        try:
            from scipy.stats import binomtest

            p_value = float(binomtest(only_a, discordant, 0.5).pvalue)
        except Exception:
            p_value = _binomial_two_sided(only_a, discordant)

    diff_pp = 100.0 * (sum(a) - sum(b)) / max(1, n)
    return McNemarResult(n, both, only_a, only_b, neither, diff_pp, p_value)


def _binomial_two_sided(successes: int, trials: int) -> float:
    """Binomial exata bilateral, para o caso de scipy indisponível."""
    def pmf(k: int) -> float:
        return math.comb(trials, k) * 0.5 ** trials

    observed = pmf(successes)
    total = sum(pmf(k) for k in range(trials + 1) if pmf(k) <= observed + 1e-12)
    return min(1.0, total)


def paired_bootstrap_diff(
    a: Sequence[bool],
    b: Sequence[bool],
    strata: Sequence[str],
    resamples: int = 10000,
    level: float = 0.95,
    seed: int = 20261031,
) -> BootstrapCI:
    """Intervalo de confiança para a diferença de acurácia entre A e B.

    A reamostragem é **pareada** — sorteia perguntas, e cada pergunta carrega os
    dois resultados — e **estratificada por banco**, de modo que a composição
    de domínios da amostra reamostrada acompanhe a do conjunto original.
    """
    if not (len(a) == len(b) == len(strata)):
        raise ValueError("vetores com tamanhos diferentes")

    n = len(a)
    if n == 0:
        return BootstrapCI(0.0, 0.0, 0.0, level, resamples)

    by_stratum: dict[str, list[int]] = {}
    for idx, key in enumerate(strata):
        by_stratum.setdefault(key, []).append(idx)

    point = 100.0 * (sum(a) - sum(b)) / n
    rng = random.Random(seed)
    diffs: list[float] = []

    for _ in range(resamples):
        acc_a = acc_b = 0
        total = 0
        for indices in by_stratum.values():
            size = len(indices)
            for _ in range(size):
                idx = indices[rng.randrange(size)]
                acc_a += int(a[idx])
                acc_b += int(b[idx])
                total += 1
        diffs.append(100.0 * (acc_a - acc_b) / max(1, total))

    diffs.sort()
    alpha = (1.0 - level) / 2.0
    low = diffs[int(alpha * (len(diffs) - 1))]
    high = diffs[int((1 - alpha) * (len(diffs) - 1))]
    return BootstrapCI(point, low, high, level, resamples)


@dataclass
class DiDResult:
    delta_pp: float
    low: float
    high: float
    pt_gain: float
    en_gain: float
    level: float = 0.95

    def render(self) -> str:
        return (
            f"ganho PT = {self.pt_gain:+.2f} pp   ganho EN = {self.en_gain:+.2f} pp\n"
            f"Δ = {self.delta_pp:+.2f} pp   IC{int(self.level*100)}% "
            f"[{self.low:+.2f}, {self.high:+.2f}]"
        )

    @property
    def conclusive(self) -> bool:
        return (self.low > 0) or (self.high < 0)


def difference_in_differences(
    pt_treated: Sequence[bool],
    pt_baseline: Sequence[bool],
    en_treated: Sequence[bool],
    en_baseline: Sequence[bool],
    strata: Sequence[str],
    resamples: int = 10000,
    level: float = 0.95,
    seed: int = 20261031,
) -> DiDResult:
    """Contraste de diferenças em diferenças (Seção 4.8).

        Δ = (EX_PT,C4 − EX_PT,C0) − (EX_EN,C4 − EX_EN,C0)

    Valores positivos indicam que o método beneficiou o português mais que o
    inglês. As quatro sequências precisam estar alinhadas ao mesmo conjunto de
    exemplos pareados — é isso que torna o contraste interpretável.
    """
    n = len(pt_treated)
    if not (len(pt_baseline) == len(en_treated) == len(en_baseline) == len(strata) == n):
        raise ValueError("as quatro condições precisam cobrir os mesmos exemplos pareados")

    def did(idx: Sequence[int]) -> tuple[float, float, float]:
        m = max(1, len(idx))
        pt = 100.0 * (sum(pt_treated[i] for i in idx) - sum(pt_baseline[i] for i in idx)) / m
        en = 100.0 * (sum(en_treated[i] for i in idx) - sum(en_baseline[i] for i in idx)) / m
        return pt - en, pt, en

    point, pt_gain, en_gain = did(range(n))

    by_stratum: dict[str, list[int]] = {}
    for idx, key in enumerate(strata):
        by_stratum.setdefault(key, []).append(idx)

    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(resamples):
        sample: list[int] = []
        for indices in by_stratum.values():
            size = len(indices)
            sample.extend(indices[rng.randrange(size)] for _ in range(size))
        draws.append(did(sample)[0])

    draws.sort()
    alpha = (1.0 - level) / 2.0
    low = draws[int(alpha * (len(draws) - 1))]
    high = draws[int((1 - alpha) * (len(draws) - 1))]
    return DiDResult(point, low, high, pt_gain, en_gain, level)


def minimum_discordant_pairs(effect_split: float = 0.70, power: float = 0.80) -> int:
    """Quantos pares discordantes são necessários para detectar uma assimetria.

    Aproximação normal para o teste de sinal. Serve à porta de decisão de
    09/10: com a taxa de discordância medida na validação, indica se o conjunto
    de teste precisa crescer **antes** de ser selado.
    """
    from math import ceil, sqrt

    try:
        from scipy.stats import norm

        z_alpha = norm.ppf(0.975)
        z_beta = norm.ppf(power)
    except Exception:
        z_alpha, z_beta = 1.959964, 0.841621

    p = effect_split
    numerator = z_alpha * 0.5 + z_beta * sqrt(p * (1 - p))
    return int(ceil((numerator / (p - 0.5)) ** 2))
