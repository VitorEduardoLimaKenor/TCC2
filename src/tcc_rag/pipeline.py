"""Construção dos objetos do pipeline a partir do manifesto."""

from __future__ import annotations

from pathlib import Path

from .config import ExperimentConfig
from .data.delex import delexicalize
from .data.normalize import referenced_tables_columns
from .data.partition import POOL, TEST, VAL
from .logging_io import read_json
from .schemas import Demo, Example, TestItem
from .structure import signature as sig


def manifest_path(cfg: ExperimentConfig) -> Path:
    return cfg.work / "manifest.json"


def environment_fingerprint() -> dict[str, str]:
    """Identifica o ambiente que produziu uma partição.

    O particionamento depende do agrupamento em famílias canônicas, que por sua
    vez depende do analisador de SQL. Se duas máquinas produzirem partições
    diferentes, a reprodutibilidade prometida na Seção 4.9 não existe — e o
    sintoma é silencioso: os totais batem, mas a composição muda. Registrar o
    ambiente faz a divergência aparecer em vez de passar despercebida.
    """
    import platform

    def versao(nome: str) -> str:
        try:
            from importlib.metadata import version

            return version(nome)
        except Exception:
            return "?"

    return {
        "python": platform.python_version(),
        "sqlglot": versao("sqlglot"),
        "platform": platform.system(),
    }


class ManifestError(RuntimeError):
    pass


def load_manifest(cfg: ExperimentConfig, warn: bool = True) -> list[Example]:
    payload = read_json(manifest_path(cfg))
    examples = [Example(**row) for row in payload["examples"]]

    # Integridade do arquivo. O manifesto costuma viver num ponto de montagem
    # de rede — o Drive do Colab — onde leitura logo após escrita pode devolver
    # conteúdo obsoleto. Recalcular o dígito e comparar com o gravado detecta
    # isso antes que uma partição errada contamine o experimento.
    gravado_digest = payload.get("partition_digest")
    atual_digest = partition_digest(examples)
    if gravado_digest and gravado_digest != atual_digest:
        raise ManifestError(
            f"manifesto inconsistente: gravado {gravado_digest}, "
            f"recalculado {atual_digest}. O arquivo pode estar truncado ou "
            f"desatualizado. Rode `tcc-rag partition` de novo."
        )

    gravado = payload.get("environment")
    if warn and gravado:
        atual = environment_fingerprint()
        difere = {
            k: (gravado.get(k), atual.get(k))
            for k in ("python", "sqlglot")
            if gravado.get(k) != atual.get(k)
        }
        if difere:
            import sys as _sys

            print(
                "[manifesto] AVISO: gerado em ambiente diferente do atual: "
                + "; ".join(f"{k}: {a} -> {b}" for k, (a, b) in difere.items())
                + "\n[manifesto] Se a partição foi refeita aqui, ela pode não ser "
                "a mesma. Use sempre o manifesto versionado.",
                file=_sys.stderr,
            )

    return examples


def partition_digest(examples: list[Example]) -> str:
    """Impressão digital da partição, insensível à ordem.

    Dois ambientes que produzam a mesma alocação chegam ao mesmo dígito. É a
    verificação que diz, em uma linha, se a partição é reprodutível.
    """
    import hashlib

    linhas = sorted(f"{e.example_id}:{e.partition}" for e in examples)
    return hashlib.sha256("\n".join(linhas).encode("utf-8")).hexdigest()[:16]


def build_demos(examples: list[Example], partition: str = POOL) -> list[Demo]:
    """Um exemplo do pool vira duas demonstrações: uma por idioma.

    Ambas carregam o mesmo `example_id` e o mesmo SQL. Assim uma pergunta em
    português recupera demonstrações em português, e a correspondente em inglês
    recupera as mesmas demonstrações em inglês — condição para que a comparação
    entre idiomas seja pareada de fato, e não apenas nas perguntas de teste.
    """
    demos: list[Demo] = []
    for ex in examples:
        if ex.partition != partition:
            continue
        signature = sig.signature(ex.sql) or {}
        tables, columns = referenced_tables_columns(ex.sql)
        for lang, question in (("pt", ex.question_pt), ("en", ex.question_en)):
            demos.append(
                Demo(
                    uid=f"{ex.example_id}::{lang}",
                    example_id=ex.example_id,
                    db_id=ex.db_id,
                    lang=lang,
                    question_raw=question,
                    question_delex=delexicalize(question),
                    sql=ex.sql,
                    signature=signature,
                    tables=tables,
                    columns=columns,
                    family_id=ex.family_id,
                    partition=ex.partition,
                    difficulty=ex.difficulty,
                )
            )
    return demos


def build_test_items(
    examples: list[Example], partition: str = TEST, languages: tuple[str, ...] = ("pt", "en")
) -> list[TestItem]:
    items: list[TestItem] = []
    for ex in examples:
        if ex.partition != partition:
            continue
        for lang in languages:
            question = ex.question_pt if lang == "pt" else ex.question_en
            items.append(
                TestItem(
                    example_id=ex.example_id,
                    db_id=ex.db_id,
                    lang=lang,
                    question_raw=question,
                    question_delex=delexicalize(question),
                    gold_sql=ex.sql,
                    difficulty=ex.difficulty,
                )
            )
    return items


__all__ = [
    "POOL",
    "TEST",
    "VAL",
    "build_demos",
    "build_test_items",
    "load_manifest",
    "manifest_path",
]
