"""Execução isolada das consultas (Seção 4.9).

Três barreiras, nesta ordem: a consulta precisa ser uma única instrução de
leitura; a conexão é aberta somente para leitura; e a execução acontece em
processo separado, com tempo limite imposto de fora.

A terceira barreira é a que importa na prática. Uma junção cartesiana sobre
tabelas grandes não devolve erro — ela simplesmente não termina, e sem um
processo separado ela leva o experimento junto.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ..config import SandboxConfig
from ..data.normalize import is_read_only
from ..schemas import ExecResult

_WORKER = "tcc_rag.evaluation._worker"


class Sandbox:
    def __init__(self, cfg: SandboxConfig):
        self.cfg = cfg

    def run(self, db_path: str | Path, sql: str) -> ExecResult:
        sql = (sql or "").strip()
        if not sql:
            return ExecResult(ok=False, error="consulta vazia")
        if not is_read_only(sql):
            return ExecResult(ok=False, error="rejeitada: não é uma única instrução de leitura")
        if not Path(db_path).exists():
            return ExecResult(ok=False, error=f"banco inexistente: {db_path}")

        # Sem `text=True`: a saída é lida como bytes e decodificada em UTF-8
        # explicitamente. Deixar o Python decidir a codificação faria o console
        # do Windows corromper todo acento, e o dano apareceria como erro de
        # acurácia em vez de erro de codificação.
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        try:
            proc = subprocess.run(
                [sys.executable, "-m", _WORKER, str(db_path), sql, str(self.cfg.max_rows)],
                capture_output=True,
                timeout=self.cfg.timeout_s,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return ExecResult(ok=False, error="timeout", timed_out=True)

        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")

        if proc.returncode != 0 or not stdout:
            return ExecResult(ok=False, error=(stderr or "falha no processo").strip()[:400])

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError:
            return ExecResult(ok=False, error="saída ilegível do processo de execução")

        if not payload.get("ok"):
            return ExecResult(ok=False, error=str(payload.get("error"))[:400])

        rows = [tuple(row) for row in payload.get("rows", [])]
        return ExecResult(ok=True, rows=rows)
