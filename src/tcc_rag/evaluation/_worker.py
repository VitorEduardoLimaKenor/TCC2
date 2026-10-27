"""Processo-filho que executa uma consulta. Não importe isto diretamente.

Roda isolado para que uma consulta malformada ou um produto cartesiano não
derrubem nem travem o experimento. O pai impõe o tempo limite matando o
processo, o que funciona mesmo quando a consulta está presa dentro do SQLite.

O pedido chega pela entrada padrão, em JSON: banco, consulta e limite de linhas.
A leitura segue o avaliador oficial do Spider (`exec_eval.exec_on_db_`): texto
decodificado com `errors="ignore"` e valores sem arredondamento. A diferença é o
limite: com ele, lê no máximo `limite + 1` linhas e informa se havia mais, em
vez de cortar o resultado em silêncio.

Três barreiras impedem escrita: o banco é aberto somente para leitura,
`PRAGMA query_only` fica ligado e um autorizador nega toda ação que não seja de
leitura (inclusive ATTACH e PRAGMA). O SQLite já recusa mais de uma instrução
por chamada.
"""

from __future__ import annotations

import base64
import json
import sqlite3
import sys
from pathlib import Path

_LEITURA = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
}


def _autorizar(acao: int, *_) -> int:
    return sqlite3.SQLITE_OK if acao in _LEITURA else sqlite3.SQLITE_DENY


def main() -> int:
    pedido = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    limite = pedido.get("limite")
    conn = None
    try:
        uri = Path(pedido["banco"]).resolve().as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=5.0)
        conn.text_factory = lambda b: b.decode(errors="ignore")
        conn.execute("PRAGMA query_only = ON")
        conn.set_authorizer(_autorizar)
        cursor = conn.execute(pedido["sql"])
        if limite is None:
            rows, excedeu = cursor.fetchall(), False
        else:
            rows = cursor.fetchmany(limite + 1)
            excedeu = len(rows) > limite
            rows = rows[:limite]
        payload = {
            "ok": True,
            "excedeu": excedeu,
            "rows": [[_valor(v) for v in row] for row in rows],
        }
    except Exception as exc:  # noqa: BLE001 - qualquer erro vira resultado
        payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    finally:
        if conn is not None:
            conn.close()

    # Escreve bytes UTF-8 direto no buffer. Passar por sys.stdout usaria a
    # codificação do console — no Windows, cp1252 — e trocaria cada acento por
    # U+FFFD, o que viraria falso erro de acurácia de execução em dados em
    # português.
    sys.stdout.buffer.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
    sys.stdout.buffer.flush()
    return 0


def _valor(value):
    # BLOB não tem representação em JSON. Vai em base64 e volta como bytes no
    # pai, para que a comparação veja o mesmo valor que a oficial veria.
    if isinstance(value, (bytes, bytearray)):
        return {"$bytes": base64.b64encode(bytes(value)).decode("ascii")}
    return value


if __name__ == "__main__":
    raise SystemExit(main())
