"""Gera um `questions_pt.json` provisório copiando as perguntas em inglês.

Serve a um propósito único e imediato: a contagem de famílias por banco, que é
a porta de decisão de 04/09, **não depende da tradução**. Ela sai da consulta
SQL. Com este arquivo você roda `prepare`, `partition`, `leakage` e `index`
hoje, valida o encadeamento inteiro e descobre se o recorte é viável, sem
esperar resposta de ninguém.

O que ele NÃO serve: qualquer resultado. Com perguntas idênticas nos dois
idiomas, a comparação bilíngue mede zero por construção. Substitua pelo corpus
real antes da fase de validação.

    python scripts/stub_questions_pt.py data/spider data/spider_pt/questions_pt.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SPLITS = ("train_spider.json", "dev.json")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2

    spider_root = Path(sys.argv[1])
    destino = Path(sys.argv[2])

    saida: dict[str, str] = {}
    for split in SPLITS:
        caminho = spider_root / split
        if not caminho.exists():
            print(f"aviso: {caminho} não encontrado, pulando")
            continue
        linhas = json.loads(caminho.read_text(encoding="utf-8"))
        for idx, row in enumerate(linhas):
            saida[f"spider_{caminho.stem}_{idx}"] = row["question"]

    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(
        json.dumps(saida, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    print(f"escrito: {destino}  ({len(saida)} perguntas)")
    print()
    print("ATENÇÃO: placeholder. As perguntas 'em português' são cópias do inglês.")
    print("Use apenas para a auditoria de famílias e para validar o pipeline.")
    print("Nenhum número produzido com este arquivo pode entrar na monografia.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
