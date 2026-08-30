"""Processo-filho que executa uma consulta. Não importe isto diretamente.

Roda isolado para que uma consulta malformada ou um produto cartesiano não
derrubem nem travem o experimento. O pai impõe o tempo limite matando o
processo, o que funciona mesmo quando a consulta está presa dentro do SQLite.
"""

from __future__ import annotations

import json
import sqlite3
import sys


def main() -> int:
    db_path, sql, max_rows = sys.argv[1], sys.argv[2], int(sys.argv[3])
    try:
        uri = f"file:{db_path}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        cursor = conn.execute(sql)
        rows = cursor.fetchmany(max_rows)
        payload = {
            "ok": True,
            "rows": [[_scalar(v) for v in row] for row in rows],
        }
    except Exception as exc:  # noqa: BLE001 - qualquer erro vira resultado
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        try:
            conn.close()  # type: ignore[possibly-undefined]
        except Exception:
            pass

    # Escreve bytes UTF-8 direto no buffer. Passar por sys.stdout usaria a
    # codificação do console — no Windows, cp1252 — e trocaria cada acento por
    # U+FFFD, o que viraria falso erro de acurácia de execução em dados em
    # português.
    sys.stdout.buffer.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


def _scalar(value):
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, float):
        # Evita que 1.0 e 1 sejam considerados resultados diferentes.
        return round(value, 6)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
