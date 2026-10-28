"""Cache em disco das consultas preliminares de C4 (Seção 4.3.3).

O rascunho não depende de k, N nem λ, e por isso é gerado uma vez e reutilizado
por toda a varredura. Até a E04, a chave era só `modelo@revisão` mais
`(example_id, idioma)`. Com ela, trocar o texto da pergunta (por exemplo, sair
do `questions_pt.json` provisório, que copia o inglês, para a tradução real)
reaproveitaria rascunhos gerados sobre o texto antigo, sem nenhum sinal.

Agora cada entrada é guardada sob o SHA-256 da sua identidade completa, que
contém:

- toda a `GenerationConfig`: backend, modelo, revisão, quantização, dtype,
  decodificação, limites, semente e lote;
- as versões das bibliotecas de geração;
- o idioma;
- o hash do *template* do prompt de rascunho (instrução e rótulos do idioma);
- o hash da pergunta efetiva, do DDL do esquema e do prompt inteiro.

Mudar qualquer um desses fatores muda a chave, e o rascunho é gerado de novo.
A entrada guarda a saída bruta do modelo, os tokens e a latência; o SQL é
extraído na hora do uso.

Os caches no formato antigo (`drafts_*.json`, sem nada disso) nunca são lidos.
Ao abrir o cache, eles vão para `quarentena/`, com um aviso explícito.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..config import GenerationConfig
from ..logging_io import isolar_cauda_incompleta
from ..proveniencia import GERACAO, codigo_revisao, versoes

FORMATO = 2
ARQUIVO = "rascunhos-v2.jsonl"
_LEGADO = "drafts_*.json"


def _sha(texto: str) -> str:
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()


def identidade(
    cfg: GenerationConfig, lang: str, pergunta: str, ddl: str, prompt: str, template: str
) -> dict[str, Any]:
    """Tudo o que pode mudar o rascunho gerado para um item."""
    return {
        "formato": FORMATO,
        "geracao": asdict(cfg),
        "decodificacao": "gulosa, sem amostragem",
        "bibliotecas": versoes(GERACAO),
        "idioma": lang,
        "template_sha256": _sha(template),
        "pergunta_sha256": _sha(pergunta),
        "ddl_sha256": _sha(ddl),
        "prompt_sha256": _sha(prompt),
    }


def chave(ident: dict[str, Any]) -> str:
    return _sha(json.dumps(ident, sort_keys=True, ensure_ascii=False))


class CacheRascunhos:
    """Arquivo JSONL só de acréscimo; uma linha por rascunho, indexada pela chave."""

    def __init__(self, pasta: str | Path):
        self.pasta = Path(pasta)
        self.pasta.mkdir(parents=True, exist_ok=True)
        self.caminho = self.pasta / ARQUIVO
        self.em_quarentena = self._isolar_legados()
        isolar_cauda_incompleta(self.caminho)
        self._entradas: dict[str, dict[str, Any]] = {}
        if self.caminho.exists():
            for numero, linha in enumerate(self.caminho.read_text(encoding="utf-8").splitlines(), 1):
                if not linha.strip():
                    continue
                entrada = json.loads(linha)
                # Uma entrada cuja identidade não reproduz a chave foi editada ou
                # corrompida: não é usada.
                if entrada.get("chave") != chave(entrada.get("identidade") or {}):
                    print(f"[rascunhos] AVISO: {self.caminho.name}:{numero} ignorada, "
                          "identidade não confere com a chave", file=sys.stderr)
                    continue
                self._entradas[entrada["chave"]] = entrada

    def _isolar_legados(self) -> list[Path]:
        movidos = []
        for antigo in sorted(self.pasta.glob(_LEGADO)):
            destino_dir = self.pasta / "quarentena"
            destino_dir.mkdir(exist_ok=True)
            destino = destino_dir / antigo.name
            if destino.exists():
                destino = destino_dir / f"{antigo.stem}.{time.strftime('%Y%m%dT%H%M%S')}{antigo.suffix}"
            shutil.move(str(antigo), str(destino))
            destino.with_name(destino.name + ".motivo.txt").write_text(
                "Cache de rascunhos no formato anterior à E04: a chave era só modelo@revisão "
                "e (example_id, idioma), sem o texto da pergunta, o DDL, o prompt nem os "
                "parâmetros de geração. Não pode ser reaproveitado sem risco de usar "
                "rascunhos de outro texto. Movido para quarentena em "
                f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')}.\n",
                encoding="utf-8",
            )
            print(f"[rascunhos] AVISO: {antigo.name} não tem proveniência suficiente e "
                  f"não será usado; movido para {destino}", file=sys.stderr)
            movidos.append(destino)
        return movidos

    def __len__(self) -> int:
        return len(self._entradas)

    def obter(self, chave_: str) -> dict[str, Any] | None:
        return self._entradas.get(chave_)

    def gravar(self, ident: dict[str, Any], texto: str, prompt_tokens: int,
               completion_tokens: int, latency_ms: float) -> dict[str, Any]:
        entrada = {
            "chave": chave(ident),
            "identidade": ident,
            "texto": texto,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "codigo_revisao": codigo_revisao(),
            "criado_em": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with self.caminho.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entrada, ensure_ascii=False) + "\n")
        self._entradas[entrada["chave"]] = entrada
        return entrada
