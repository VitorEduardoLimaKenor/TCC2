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


def report_rows(
    runs_dir: Path,
    *,
    config_hash: str,
    run_ids: list[str] | None = None,
    split: str | None = None,
    partition_digest: str | None = None,
) -> list[dict[str, Any]]:
    """Seleciona logs compatíveis, sem escolher por nome nem descartar duplicatas.

    Relatórios usam leitura estrita: uma linha truncada não pode desaparecer do
    denominador. Logs históricos sem split precisam ter sua proveniência
    recuperada do manifesto original antes de entrar em uma análise.
    """
    if run_ids and any(Path(name).name != name or "/" in name or "\\" in name
                       for name in run_ids):
        raise ValueError("--run-id deve conter nomes, sem diretórios")
    paths = ([runs_dir / f"{name}.jsonl" for name in run_ids] if run_ids
             else sorted(runs_dir.glob("*.jsonl")))
    filters = {"config_hash": config_hash}
    if split:
        filters["split"] = split
    if partition_digest:
        filters["partition_digest"] = partition_digest
    selected: list[dict[str, Any]] = []
    selected_paths = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"log não encontrado: {path}")
        rows = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSON inválido em {path.name}:{number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"registro inválido em {path.name}:{number}")
            rows.append(row)
        matches = [all(row.get(key) == value for key, value in filters.items())
                   for row in rows]
        if not any(matches):
            if run_ids:
                raise ValueError(f"{path.name}: log vazio ou incompatível com os filtros {filters}")
            continue
        if not all(matches):
            raise ValueError(f"{path.name}: contém registros de identidades diferentes")
        for field in ("run_id", "config_hash", "partition_digest", "split"):
            values = {row.get(field) for row in rows}
            if len(values) != 1 or not all(isinstance(v, str) and v for v in values):
                raise ValueError(f"{path.name}: {field} ausente ou misturado; "
                                 "verifique a proveniência do log original")
        if rows[0]["run_id"] != path.stem:
            raise ValueError(f"{path.name}: run_id interno difere do nome do arquivo")
        if rows[0]["split"] not in {"val", "test"}:
            raise ValueError(f"{path.name}: split inválido")
        selected_paths.append(path)
        selected.extend(rows)
    if not selected:
        raise ValueError("nenhum log corresponde aos filtros; confira configuração e --run-id")
    if not run_ids and len(selected_paths) > 1:
        raise ValueError("seleção ambígua; informe --run-id: "
                         + ", ".join(p.stem for p in selected_paths))
    for field in ("config_hash", "partition_digest", "split"):
        if len({row[field] for row in selected}) != 1:
            raise ValueError(f"não é permitido combinar logs com {field} diferentes")
    seen = set()
    databases = {}
    for row in selected:
        if any(not row.get(field) for field in ("condition", "lang", "example_id", "db_id")):
            raise ValueError("registro sem condição, idioma, exemplo ou banco")
        if not isinstance(row.get("ex_correct"), bool):
            raise ValueError("ex_correct ausente ou não booleano")
        key = (row["condition"], row["lang"], row["example_id"])
        if key in seen:
            raise ValueError(f"observação duplicada entre os logs selecionados: {key}")
        seen.add(key)
        previous = databases.setdefault(row["example_id"], row["db_id"])
        if previous != row["db_id"]:
            raise ValueError(f"bancos divergentes para {row['example_id']}")
    return selected
