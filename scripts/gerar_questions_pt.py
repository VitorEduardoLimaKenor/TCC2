"""Gera data/spider_pt/questions_pt.json a partir da tradução de José e Cozman (2021).

Decisão D-002 do autor (28/10/2026): a fonte é `Marchanjo/spider-pt`
(Hugging Face, CC BY-SA 4.0), sem nenhuma alteração. A pergunta em PT-BR de
cada item é o campo `question` do arquivo traduzido, na mesma posição do
`train_spider.json` ou do `dev.json` original.

Os arquivos baixados são dados não confiáveis: só `json.load`. Antes de
escrever, o script confere, item a item, que `db_id` e `query` do arquivo
traduzido são idênticos aos do Spider local, e recusa qualquer desalinhamento
(tamanho, banco ou consulta). Grava também um arquivo de proveniência com
SHA-256 da fonte e do resultado.

Uso: python -I scripts/gerar_questions_pt.py <spider original> <spider-pt baixado> <destino.json>
"""

import hashlib
import json
import pathlib
import sys

FONTE = {
    "dataset": "Marchanjo/spider-pt",
    "url": "https://huggingface.co/datasets/Marchanjo/spider-pt",
    "autores": "José e Cozman (2021)",
    "licenca": "CC BY-SA 4.0",
    "decisao": "D-002 (28/10/2026)",
}


def sha(caminho: pathlib.Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def main(orig_dir: pathlib.Path, pt_dir: pathlib.Path, destino: pathlib.Path) -> int:
    saida = []
    conferencia = {}
    for nome in ("train_spider.json", "dev.json"):
        orig = json.loads((orig_dir / nome).read_text(encoding="utf-8"))
        pt = json.loads((pt_dir / nome).read_text(encoding="utf-8"))
        if len(orig) != len(pt):
            raise SystemExit(f"{nome}: {len(orig)} itens no original e {len(pt)} na tradução")
        for i, (o, p) in enumerate(zip(orig, pt)):
            if o["db_id"] != p["db_id"] or o["query"] != p["query"]:
                raise SystemExit(f"{nome}[{i}]: db_id ou query diferem do original; recusado")
            pergunta = p["question"]
            if not isinstance(pergunta, str) or not pergunta.strip():
                raise SystemExit(f"{nome}[{i}]: pergunta em PT ausente")
            saida.append({"id": f"spider_{pathlib.Path(nome).stem}_{i}", "question_pt": pergunta})
        conferencia[nome] = {"itens": len(orig), "db_id_e_query_iguais": len(orig),
                             "sha256_fonte": sha(pt_dir / nome), "sha256_original": sha(orig_dir / nome)}
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(saida, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    proveniencia = {**FONTE, "arquivos": conferencia, "itens": len(saida), "gerado": destino.name,
                    "sha256_gerado": sha(destino),
                    "formato": "lista de {id: spider_<arquivo>_<índice>, question_pt}"}
    destino.with_name(destino.stem + ".proveniencia.json").write_text(
        json.dumps(proveniencia, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(proveniencia, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*(pathlib.Path(a) for a in sys.argv[1:4])))
