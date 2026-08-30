"""Configuração do experimento.

A configuração é um único arquivo YAML. Depois da fase de validação ela é
congelada: o `config_hash` entra em cada linha do log de execução, de modo que
qualquer resultado possa ser reconduzido aos parâmetros que o produziram
(Seção 4.9 da monografia).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

Condition = Literal["C0", "C1", "C2", "C3", "C4"]
Language = Literal["pt", "en"]

ALL_CONDITIONS: tuple[Condition, ...] = ("C0", "C1", "C2", "C3", "C4")
ALL_LANGUAGES: tuple[Language, ...] = ("pt", "en")


@dataclass
class PathsConfig:
    spider_root: str = "data/spider"
    """Diretório com train_spider.json, dev.json, tables.json e database/."""
    questions_pt: str = "data/spider_pt/questions_pt.json"
    """Traduções em português, alinhadas por id ao Spider original."""
    work_dir: str = "artifacts"
    """Saídas: manifesto, partições, índices, logs, tabelas."""


@dataclass
class DataConfig:
    min_questions_per_db: int = 120
    """Bancos com menos perguntas anotadas que isto são descartados."""
    min_families_per_db: int = 25
    """Porta de decisão da Fase 0: sem famílias suficientes não há recorte."""
    pool_ratio: float = 0.60
    val_ratio: float = 0.15
    test_ratio: float = 0.25
    near_duplicate_threshold: float = 0.92
    """Similaridade de cosseno acima da qual pool e teste são considerados
    duplicatas e o item do pool é removido."""
    seed: int = 20260904


@dataclass
class EmbeddingConfig:
    model: str = "intfloat/multilingual-e5-base"
    revision: str | None = None
    device: str = "cpu"
    batch_size: int = 64
    normalize: bool = True
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    """Os modelos da família E5 exigem prefixos. Deixe vazio para BGE-m3."""


@dataclass
class QdrantConfig:
    mode: Literal["local", "server"] = "local"
    path: str = "artifacts/qdrant"
    """Modo local: diretório em disco, sem servidor. Funciona no Colab."""
    url: str = "http://localhost:6333"
    collection: str = "demos"
    distance: str = "Cosine"


@dataclass
class RetrievalConfig:
    k: int = 4
    """Número de demonstrações no prompt. Grade da validação: {1, 4, 8}."""
    n_candidates: int = 25
    """Candidatos da primeira etapa em C4. Precisa ser bem maior que k."""
    lambda_question: float = 0.5
    """Peso da similaridade da pergunta na combinação convexa de C4."""
    fixed_demo_seed: int = 20260919
    """Semente do sorteio de C1, fixada uma vez e nunca mais alterada."""


@dataclass
class GenerationConfig:
    backend: Literal["vllm", "hf", "dummy"] = "hf"
    model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    revision: str | None = None
    quantization: str | None = None
    """Ex.: "awq", "gptq", "bitsandbytes-4bit". Registrado no log."""
    dtype: str = "float16"
    """A T4 é Turing: use float16. bfloat16 exige Ampere ou mais novo."""
    max_context_tokens: int = 8192
    max_new_tokens: int = 320
    temperature: float = 0.0
    seed: int = 20261010
    batch_size: int = 16
    enable_prefix_caching: bool = True
    """Só vLLM. Os prompts de um mesmo banco compartilham o DDL inteiro."""
    gpu_memory_utilization: float = 0.90


@dataclass
class SandboxConfig:
    timeout_s: float = 15.0
    max_rows: int = 5000


@dataclass
class ExperimentConfig:
    name: str = "piloto"
    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)

    # ---------------------------------------------------------------- I/O

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls._from_dict(raw)

    @classmethod
    def _from_dict(cls, raw: dict[str, Any]) -> "ExperimentConfig":
        sections = {
            "paths": PathsConfig,
            "data": DataConfig,
            "embedding": EmbeddingConfig,
            "qdrant": QdrantConfig,
            "retrieval": RetrievalConfig,
            "generation": GenerationConfig,
            "sandbox": SandboxConfig,
        }
        kwargs: dict[str, Any] = {"name": raw.get("name", "piloto")}
        for key, klass in sections.items():
            kwargs[key] = klass(**(raw.get(key) or {}))
        return cls(**kwargs)

    def dump(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(asdict(self), sort_keys=True, allow_unicode=True),
            encoding="utf-8",
        )

    # ------------------------------------------------------------- hashing

    @property
    def config_hash(self) -> str:
        """Identidade da configuração.

        `paths` fica de fora: mover o diretório de trabalho não muda o
        experimento. Todo o resto entra.
        """
        payload = asdict(self)
        payload.pop("paths", None)
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    @property
    def work(self) -> Path:
        p = Path(self.paths.work_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p
