"""Registro de execução: uma linha JSONL por geração.

Este arquivo é a matéria-prima do Capítulo 5. Todas as tabelas, a análise
estatística e a taxonomia de erros saem de consultas sobre ele — nenhum número
é digitado à mão.

Também é o que torna a execução retomável. Em sessão de Colab ou Kaggle, cair
na hora nove de catorze é normal; ao reiniciar, o que já existe é pulado.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def record_key(row: dict[str, Any]) -> tuple:
    return (row["run_id"], row["condition"], row["lang"], row["example_id"])


class RunLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._done: set[tuple] = set()
        if self.path.exists():
            for row in self.read():
                self._done.add(record_key(row))

    # ------------------------------------------------------------- escrita

    def append(self, row: dict[str, Any]) -> None:
        key = record_key(row)
        if key in self._done:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._done.add(key)

    def has(self, run_id: str, condition: str, lang: str, example_id: str) -> bool:
        return (run_id, condition, lang, example_id) in self._done

    @property
    def n_done(self) -> int:
        return len(self._done)

    # -------------------------------------------------------------- leitura

    def read(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Linha truncada por queda de sessão: descarta e segue.
                    continue

    def rows(self) -> list[dict[str, Any]]:
        return list(self.read())


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
