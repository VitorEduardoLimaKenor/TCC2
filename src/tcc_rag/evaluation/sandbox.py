"""Execução isolada das consultas (Seção 4.9).

A consulta roda em processo separado, sobre banco aberto somente para leitura,
com tempo limite imposto de fora. As barreiras contra escrita ficam no
processo-filho (`_worker`): modo somente leitura, `PRAGMA query_only` e um
autorizador que só permite leitura. O SQLite recusa mais de uma instrução.

O processo separado é a barreira que importa na prática. Uma junção cartesiana
sobre tabelas grandes não devolve erro — ela simplesmente não termina, e sem um
processo separado ela leva o experimento junto.

Não há filtro sintático prévio. Um analisador que não reconhece uma construção
válida do SQLite transformaria consulta correta em erro de execução, e o
avaliador oficial executa a consulta sem esse filtro.
"""

from __future__ import annotations

import base64
import json
import os
import subprocess
import sys
from pathlib import Path

from ..config import SandboxConfig
from ..schemas import ExecResult

_WORKER = "tcc_rag.evaluation._worker"


class Sandbox:
    def __init__(self, cfg: SandboxConfig):
        self.cfg = cfg

    def run(self, db_path: str | Path, sql: str, limite: int | None = None) -> ExecResult:
        """Executa `sql` e devolve todas as linhas, ou no máximo `limite`.

        Com limite, `truncated` indica que havia mais linhas que o limite. O
        corte nunca é silencioso: quem pede o limite decide o que o excedente
        significa.
        """
        sql = (sql or "").strip()
        if not sql:
            return ExecResult(ok=False, error="consulta vazia")
        if not Path(db_path).exists():
            return ExecResult(ok=False, error=f"banco inexistente: {db_path}")

        pedido = json.dumps(
            {"banco": str(db_path), "sql": sql, "limite": limite}, ensure_ascii=False
        ).encode("utf-8")
        # Sem `text=True`: a saída é lida como bytes e decodificada em UTF-8
        # explicitamente. Deixar o Python decidir a codificação faria o console
        # do Windows corromper todo acento, e o dano apareceria como erro de
        # acurácia em vez de erro de codificação.
        env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
        try:
            proc = subprocess.run(
                [sys.executable, "-m", _WORKER],
                input=pedido,
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

        rows = [tuple(_restaurar(v) for v in row) for row in payload.get("rows", [])]
        return ExecResult(ok=True, rows=rows, truncated=bool(payload.get("excedeu")))


def _restaurar(value):
    if isinstance(value, dict) and "$bytes" in value:
        return base64.b64decode(value["$bytes"])
    return value
