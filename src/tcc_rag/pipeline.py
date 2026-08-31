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


def subsample(
    items: list[TestItem], limit: int, seed: int = 20261010
) -> list[TestItem]:
    """Amostra estratificada por banco, mantendo os pares de idioma juntos.

    Existe porque cortar o começo da lista **não** é amostrar. Os itens saem do
    manifesto agrupados por banco, então `items[:150]` devolve os primeiros
    bancos em ordem alfabética e ignora os demais — na validação deste trabalho,
    150 de 275 itens cobriam 9 dos 17 bancos, sem nenhuma sobreposição com o
    restante. Uma varredura de hiperparâmetros sobre esse recorte escolhe o
    parâmetro que serve àqueles nove domínios.

    A amostragem é por exemplo, não por item: se as duas línguas estiverem
    presentes, ambas entram ou nenhuma entra, sem o que a comparação bilíngue
    deixa de ser pareada.
    """
    import random

    if limit <= 0 or limit >= len(items):
        return items

    langs = {item.lang for item in items}
    por_exemplo: dict[str, list[TestItem]] = {}
    banco_de: dict[str, str] = {}
    for item in items:
        por_exemplo.setdefault(item.example_id, []).append(item)
        banco_de[item.example_id] = item.db_id

    alvo = max(1, limit // max(1, len(langs)))
    if alvo >= len(por_exemplo):
        return items

    por_banco: dict[str, list[str]] = {}
    for example_id, db_id in banco_de.items():
        por_banco.setdefault(db_id, []).append(example_id)

    total = len(por_exemplo)
    cotas: dict[str, float] = {
        db: alvo * len(ids) / total for db, ids in por_banco.items()
    }
    inteiras = {db: int(q) for db, q in cotas.items()}
    # Maior resto: distribui o que sobrou para os bancos com maior fração.
    faltam = alvo - sum(inteiras.values())
    for db in sorted(cotas, key=lambda d: -(cotas[d] - inteiras[d]))[: max(0, faltam)]:
        inteiras[db] += 1

    escolhidos: list[str] = []
    for db in sorted(por_banco):
        ids = sorted(por_banco[db])
        random.Random(f"{seed}|{db}").shuffle(ids)
        escolhidos.extend(ids[: inteiras.get(db, 0)])

    selecionados = set(escolhidos)
    return [item for item in items if item.example_id in selecionados]


def item_set_digest(items: list[TestItem]) -> str:
    """Identifica o conjunto avaliado, para que amostras distintas não
    compartilhem log."""
    import hashlib

    chave = "\n".join(sorted(f"{i.example_id}:{i.lang}" for i in items))
    return hashlib.sha256(chave.encode("utf-8")).hexdigest()[:8]


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
    "item_set_digest",
    "subsample",
    "TEST",
    "VAL",
    "build_demos",
    "build_test_items",
    "load_manifest",
    "manifest_path",
]
