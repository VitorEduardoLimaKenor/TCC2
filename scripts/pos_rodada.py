"""Do pacote de resultados da rodada às tabelas e figuras da monografia.

Uso:
    python scripts/pos_rodada.py <tcc-resultados-*.zip> <dir_evidencia> <dir_monografia>

O pacote é o zip produzido pela célula 2 de `notebooks/confirmatorio.ipynb`
(logs, análises, custo, registro de congelamento, manifesto e estratos). O
roteiro:

1. extrai o pacote em `<dir_evidencia>/pacote` e grava os SHA-256 dos arquivos;
2. gera as tabelas do Capítulo 5 em `<dir_monografia>/Textual/tabelas`;
3. gera as figuras dos resultados em `<dir_monografia>/Images` (só com o teste);
4. extrai o exemplo do mecanismo de um item de validação e gera o quadro;
5. imprime os números principais, para a redação.

Nada é digitado: tudo o que vai para a monografia sai dos arquivos do pacote.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def rodar(*args: str) -> None:
    print("$", " ".join(str(a) for a in args))
    subprocess.run([sys.executable, *map(str, args)], check=True, cwd=RAIZ)


def pct(v: float | None) -> str:
    return "  --" if v is None else f"{100 * v:5.1f}"


def resumo(nome: str, a: dict) -> None:
    print(f"\n===== {nome}: split={a['split']} n={a['n_exemplos']} inválidas={len(a['referencias_invalidas'])} =====")
    print("cond   EX pt  EX en  hit pt  n/exec pt  tok pt  EM pt")
    for c in a["condicoes"]:
        pt, en = a["celulas"].get(f"{c}|pt"), a["celulas"].get(f"{c}|en")
        print(f"{c}    {pct(pt['ex'])}  {pct(en['ex']) if en else '   --'}  {pct(pt['acerto_estrutural'])}  "
              f"{pt['nao_executaveis']:>8}  {pt['prompt_tokens_total_medio']:6.0f}  {pct(pt['em_oficial'])}")
    for chave, c in a["contrastes"].items():
        print(f"  {chave:<10} {c['diferenca_pp']:+6.2f} pp  IC95 [{c['ic95'][0]:+.2f}; {c['ic95'][1]:+.2f}]  "
              f"p={c['mcnemar']['p_value']:.4f}  disc={c['mcnemar']['only_a']}/{c['mcnemar']['only_b']}  ({c['papel']})")
    if a.get("familia_confirmatoria_holm"):
        print("  Holm:", {k: round(v["p_holm"], 4) for k, v in a["familia_confirmatoria_holm"].items()})
    for nome_did, d in (a.get("diferencas_em_diferencas") or {}).items():
        print(f"  {nome_did}: Δ={d['delta_pp']:+.2f} [{d['low']:+.2f}; {d['high']:+.2f}]  (PT {d['pt_gain']:+.2f}; EN {d['en_gain']:+.2f})")
    for chave, h in a["h3"].items():
        if chave.endswith("|pt") and h["diferenca_pp"] is not None:
            print(f"  H3 {chave}: com {pct(h['ex_com_acerto_estrutural'])} sem {pct(h['ex_sem_acerto_estrutural'])} "
                  f"dif {h['diferenca_pp']:+.1f} [{h['ic95'][0]:+.1f}; {h['ic95'][1]:+.1f}]  ajustada "
                  f"{h['diferenca_ajustada_dificuldade_pp']}")
    for chave, bloco in (a.get("poder") or {}).items():
        print(f"  poder {chave}: discordância {100 * bloco['taxa_discordancia_observada']:.1f}%  "
              f"DMD80 α=5%: {bloco['alfa_0.05']['dmd_80']}  α=2,5%: {bloco['alfa_0.025']['dmd_80']}")


def main(pacote: Path, evidencia: Path, monografia: Path) -> int:
    destino = evidencia / "pacote"
    destino.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(pacote) as z:
        z.extractall(destino)
    somas = []
    for f in sorted(p for p in destino.rglob("*") if p.is_file()):
        somas.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(destino)}")
    (evidencia / "SHA256SUMS.txt").write_text("\n".join(somas) + "\n", encoding="utf-8")
    (evidencia / "pacote-origem.txt").write_text(
        f"{pacote.name}\nsha256 {hashlib.sha256(pacote.read_bytes()).hexdigest()}\n", encoding="utf-8")

    artefatos = destino / "artifacts"
    val = sorted((artefatos / "analise").glob("analise-val-*.json"))
    teste = sorted((artefatos / "analise").glob("analise-test-*.json"))
    if not val:
        print("pacote sem análise de validação", file=sys.stderr)
        return 1
    tabelas = monografia / "Textual" / "tabelas"
    rodar("scripts/tabelas_cap5.py", val[-1], teste[-1] if teste else "-", tabelas)
    if teste:
        rodar("scripts/figuras/resultados.py", teste[-1], monografia / "Images")

    # exemplo do mecanismo: configuração local que aponta para o pacote (os caminhos não entram em nenhuma análise)
    logs_val = [p for p in sorted((artefatos / "runs-v2").glob("*.jsonl")) if "-val-" in p.name]
    if logs_val:
        cfg = (RAIZ / "configs" / "confirmatorio.yaml").read_text(encoding="utf-8").replace(
            "work_dir: /content/tcc/artifacts", f"work_dir: {artefatos}")
        local = evidencia / "confirmatorio-local.yaml"
        local.write_text(cfg, encoding="utf-8")
        exemplo = evidencia / "exemplo-mecanismo.json"
        rodar("scripts/exemplo_mecanismo.py", local, logs_val[-1], exemplo)
        rodar("scripts/tabela_exemplo.py", exemplo, tabelas / "exemplo_mecanismo.tex")

    resumo("VALIDAÇÃO", json.loads(val[-1].read_text(encoding="utf-8")))
    if teste:
        resumo("TESTE", json.loads(teste[-1].read_text(encoding="utf-8")))
    congelamento = artefatos / "congelamento.json"
    if congelamento.is_file():
        c = json.loads(congelamento.read_text(encoding="utf-8"))
        print(f"\ncongelado em {c['congelado_em_utc']}  código {c['codigo_revisao'][:12]}  config {c['configuracao']['config_hash']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*(Path(a) for a in sys.argv[1:4])))
