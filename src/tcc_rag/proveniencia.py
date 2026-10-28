"""Proveniência da execução: revisão do código e versões do ambiente.

Os logs de agosto e setembro de 2026 não registravam com que código foram
produzidos, e a revisão teve de ser inferida pelo formato dos campos (lacuna 4
da E01). A partir da E04, cada linha do log carrega a revisão do git e as
versões das bibliotecas que podem mudar uma geração, uma vetorização ou um
veredito.

Nada daqui entra no `config_hash` nem no `partition_digest`: são fatos do
ambiente, não parâmetros do experimento.
"""

from __future__ import annotations

import functools
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_RAIZ = Path(__file__).resolve().parents[2]

GERACAO = ("torch", "transformers", "accelerate", "bitsandbytes", "vllm")
VETORIZACAO = ("sentence-transformers", "transformers", "torch")
AVALIACAO = ("sqlglot", "sqlparse", "nltk", "qdrant-client")


def versao(pacote: str) -> str | None:
    try:
        return version(pacote)
    except PackageNotFoundError:
        return None


def versoes(pacotes: tuple[str, ...]) -> dict[str, str | None]:
    return {p: versao(p) for p in pacotes}


@functools.lru_cache(maxsize=1)
def codigo_revisao() -> str:
    """SHA do commit do pacote, com "+modificado" se houver mudança não commitada.

    "desconhecida" quando o pacote não vem de um clone git (por exemplo, uma
    instalação a partir de arquivo). No Colab o pacote é clonado e instalado
    em modo editável, então a revisão existe.
    """
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=_RAIZ, capture_output=True, text=True, timeout=10
        )
        if sha.returncode != 0 or not sha.stdout.strip():
            return "desconhecida"
        sujo = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=_RAIZ, capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "desconhecida"
    return sha.stdout.strip() + ("+modificado" if sujo.stdout.strip() else "")


@functools.lru_cache(maxsize=1)
def ambiente() -> dict:
    """Python, sistema, versões relevantes e, se o torch já estiver carregado, a GPU."""
    saida: dict = {
        "python": platform.python_version(),
        "sistema": platform.system(),
        "bibliotecas": versoes(tuple(dict.fromkeys(GERACAO + VETORIZACAO + AVALIACAO))),
    }
    torch = sys.modules.get("torch")
    if torch is not None:
        try:
            saida["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        except Exception:  # noqa: BLE001
            saida["gpu"] = None
    return saida
