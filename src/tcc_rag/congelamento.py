"""Portão de sanidade da validação e registro de congelamento (E15).

O conjunto de teste só é aberto depois que este módulo grava o registro de
congelamento. O registro é produzido por máquina, a partir da validação
completa, e contém o que não pode mais mudar: a configuração, as revisões do
código e dos modelos, as identidades dos dados, do índice e dos avaliadores, o
plano de análise e o poder do teste primário calculado com a discordância
observada na validação.

Antes de gravar, o portão confere se a validação rodou como previsto. Os
critérios dizem respeito ao funcionamento do experimento — completude,
truncamento, execução, uso das demonstrações — e **não** aos contrastes entre
as condições: nenhuma decisão sobre o método depende deles. Se um critério
falha, nada é gravado e o teste não é aberto.

O registro é gravado uma única vez. Numa retomada, o que se confere é que a
configuração e a revisão do código continuam as do registro.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .config import ALL_CONDITIONS, ALL_LANGUAGES, ExperimentConfig

ARQUIVO = "congelamento.json"

LIMITES = {
    "referencias_invalidas_max": 5,
    "prompts_no_limite_de_entrada_max": 0,
    "nao_executaveis_por_celula_max": 0.15,
    "recurso_ordenacao_inicial_c4_max": 0.20,
    "ex_c0_en_min": 0.50,
    "ex_c0_en_max": 0.95,
    "k_efetivo_completo_min": 0.99,
}

PLANO_DE_ANALISE = {
    "desfecho_principal": "acurácia de execução oficial do Spider (test-suite-sql-eval @ e97acc5, keep_distinct=False)",
    "comparacao_primaria": "C3 − C2 em português brasileiro: McNemar exato bilateral; IC de 95% por bootstrap "
                           "pareado estratificado por banco (10.000 reamostras, semente da configuração)",
    "h1": "C4 − C2 em português brasileiro, mesmo procedimento; corroborada se a diferença for positiva, o IC "
          "excluir zero e o valor-p ajustado for inferior a 5%",
    "multiplicidade": "família {C3 − C2, C4 − C2} em português; correção de Holm; nível global de 5% bilateral",
    "h2": "Δ = (EX_PT,C4 − EX_PT,C0) − (EX_EN,C4 − EX_EN,C0), com IC de 95%; intervalo que contém zero é "
          "reportado como inconclusivo",
    "h3": "diferença de EX entre perguntas com e sem acerto estrutural, por condição com demonstrações, bruta e "
          "ajustada pela dificuldade oficial, com IC de 95%; associação observacional",
    "exploratorias": "C4 − C3, C2 − C1, C1 − C0, C3 − C0, todos os contrastes em inglês, EN − PT por condição, "
                     "Δ com C3 e C0, estratos de literal traduzido (pré-registro de 28/10/2026), EM oficial, "
                     "sensibilidade keep_distinct=True, rubrica automática de erros, sobreposição das demonstrações",
    "denominadores": "pergunta com referência inválida sai de todas as condições e idiomas",
    "custo": "tokens sem preenchimento do lote na rodada principal (em C4, somada a consulta preliminar); latência "
             "em passagem dedicada com lote unitário sobre 100 perguntas do teste em português, estratificadas por banco",
    "apos_a_abertura": "nenhum parâmetro é reajustado; a rodada é retomada apenas para os itens que faltam; "
                       "nenhuma tentativa é selecionada por desempenho; a análise de erros vem depois do registro "
                       "dos resultados",
}


def conferir_validacao(rows: list[dict[str, Any]], n_esperado: int) -> dict[str, Any]:
    """Critérios de funcionamento da validação. Devolve os valores e a lista de falhas."""
    falhas: list[str] = []
    celulas: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        celulas.setdefault(f"{r['condition']}|{r['lang']}", []).append(r)
    esperadas = {f"{c}|{l}" for c in ALL_CONDITIONS for l in ALL_LANGUAGES}
    faltam = sorted(esperadas - set(celulas))
    if faltam:
        falhas.append(f"condições ou idiomas ausentes: {', '.join(faltam)}")
    tamanhos = {k: len(v) for k, v in sorted(celulas.items())}
    if any(n != n_esperado for n in tamanhos.values()):
        falhas.append(f"linhas por condição e idioma diferentes de {n_esperado}: {tamanhos}")
    ids = [frozenset(r["example_id"] for r in v) for v in celulas.values()]
    if any(i != ids[0] for i in ids[1:]):
        falhas.append("pareamento incompleto: as condições não cobrem as mesmas perguntas")

    invalidas = sorted({r["example_id"] for r in rows if r.get("ex_correct") is None})
    if len(invalidas) > LIMITES["referencias_invalidas_max"]:
        falhas.append(f"{len(invalidas)} perguntas com referência inválida")
    no_limite = sum(bool(r.get("prompt_no_limite_de_entrada")) for r in rows)
    if no_limite > LIMITES["prompts_no_limite_de_entrada_max"]:
        falhas.append(f"{no_limite} prompts atingiram o limite de entrada (possível truncamento)")

    por_celula: dict[str, dict[str, Any]] = {}
    for chave, linhas in sorted(celulas.items()):
        validas = [r for r in linhas if r.get("ex_correct") is not None]
        n = max(1, len(validas))
        nao_exec = sum(r.get("ex_status") in ("erro_predicao", "tempo_predicao") for r in validas) / n
        por_celula[chave] = {
            "n": len(linhas),
            "ex": sum(bool(r["ex_correct"]) for r in validas) / n,
            "nao_executaveis": nao_exec,
            "recurso_ordenacao_inicial": sum(bool(r.get("used_fallback")) for r in linhas) / max(1, len(linhas)),
        }
        if nao_exec > LIMITES["nao_executaveis_por_celula_max"]:
            falhas.append(f"{chave}: {100 * nao_exec:.1f}% de predições não executáveis")
    for lang in ALL_LANGUAGES:
        c4 = por_celula.get(f"C4|{lang}")
        if c4 and c4["recurso_ordenacao_inicial"] > LIMITES["recurso_ordenacao_inicial_c4_max"]:
            falhas.append(f"C4|{lang}: recurso à ordenação inicial em {100 * c4['recurso_ordenacao_inicial']:.1f}%")
        sem_custo = sum(1 for r in celulas.get(f"C4|{lang}", []) if r.get("rascunho_prompt_tokens") is None)
        if sem_custo:
            falhas.append(f"C4|{lang}: {sem_custo} linhas sem o custo da consulta preliminar")
    c0_en = por_celula.get("C0|en")
    if c0_en and not (LIMITES["ex_c0_en_min"] <= c0_en["ex"] <= LIMITES["ex_c0_en_max"]):
        falhas.append(f"C0|en: EX de {100 * c0_en['ex']:.1f}% fora da faixa de sanidade")

    com_demos = [r for r in rows if r["condition"] != "C0"]
    k = max((r.get("k_efetivo") or 0 for r in com_demos), default=0)
    completos = sum((r.get("k_efetivo") or 0) == k for r in com_demos) / max(1, len(com_demos))
    if completos < LIMITES["k_efetivo_completo_min"]:
        falhas.append(f"só {100 * completos:.1f}% dos prompts de C1 a C4 receberam as {k} demonstrações")

    return {
        "aprovado": not falhas,
        "falhas": falhas,
        "limites": LIMITES,
        "linhas": len(rows),
        "linhas_por_celula": tamanhos,
        "referencias_invalidas": invalidas,
        "prompts_no_limite_de_entrada": no_limite,
        "k_nominal_observado": k,
        "prompts_com_k_completo": completos,
        "por_celula": por_celula,
    }


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def registro(cfg: ExperimentConfig, config_path: str, rows_val: list[dict[str, Any]], conferencia: dict[str, Any],
             analise_val: dict[str, Any] | None, n_teste: int, codigo_revisao: str) -> dict[str, Any]:
    from dataclasses import asdict

    primeira = rows_val[0]
    work = cfg.work
    return {
        "formato": 1,
        "congelado_em_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "codigo_revisao": codigo_revisao,
        "configuracao": {"arquivo": config_path, "config_hash": cfg.config_hash, "conteudo": asdict(cfg),
                         "arquivo_sha256": _sha(Path(config_path))},
        "modelos": {"gerador": cfg.generation.model, "gerador_revisao": cfg.generation.revision,
                    "quantizacao": cfg.generation.quantization, "vetorizador": cfg.embedding.model,
                    "vetorizador_revisao": cfg.embedding.revision},
        "dados": {
            "partition_digest": primeira.get("partition_digest"),
            "manifesto_sha256": _sha(work / "manifest.json"),
            "questions_pt_sha256": _sha(Path(cfg.paths.questions_pt)),
            "estratos_literais_sha256": _sha(work / "estratos_literais.json"),
            "indice_digest": primeira.get("indice_digest"),
            "template_sha256": primeira.get("template_sha256"),
        },
        "avaliadores": {"ex": primeira.get("ex_avaliador"), "em": primeira.get("em_avaliador")},
        "ambiente_da_validacao": primeira.get("ambiente"),
        "validacao": {
            "run_ids": sorted({r["run_id"] for r in rows_val}),
            "codigo_revisoes": sorted({r.get("codigo_revisao") or "" for r in rows_val}),
            "conferencia": conferencia,
            "analise_sha256": _sha(work / "analise" / f"analise-val-{cfg.config_hash}.json"),
            "poder": (analise_val or {}).get("poder"),
        },
        "teste": {"perguntas": n_teste, "linhas_esperadas": n_teste * len(ALL_CONDITIONS) * len(ALL_LANGUAGES)},
        "plano_de_analise": PLANO_DE_ANALISE,
        "decisoes_do_autor": "11/11/2026: adotadas as recomendações registradas para D-006, D-009 (Holm), D-010 "
                             "(sem repetir a varredura; sem a ablação E12), D-014, D-018, D-019, D-020 e D-021; "
                             "28/10/2026: C3 como método da comparação principal qualquer que seja a validação (D-003)",
    }


def congelar(cfg: ExperimentConfig, config_path: str, rows_val: list[dict[str, Any]], n_val: int, n_teste: int,
             codigo_revisao: str) -> tuple[bool, dict[str, Any]]:
    """Grava o registro se a validação passar no portão; numa retomada, só confere.

    Devolve (pode_abrir_o_teste, registro_ou_conferencia).
    """
    destino = cfg.work / ARQUIVO
    if destino.is_file():
        gravado = json.loads(destino.read_text(encoding="utf-8"))
        divergencias = []
        if gravado["configuracao"]["config_hash"] != cfg.config_hash:
            divergencias.append("config_hash")
        if gravado["codigo_revisao"] != codigo_revisao:
            divergencias.append("codigo_revisao")
        gravado["_conferencia_da_retomada"] = {"divergencias": divergencias}
        return (not divergencias, gravado)

    conferencia = conferir_validacao(rows_val, n_val)
    if not conferencia["aprovado"]:
        return (False, {"conferencia": conferencia})
    if "+" in codigo_revisao or codigo_revisao == "desconhecida":
        return (False, {"conferencia": {**conferencia, "aprovado": False,
                                         "falhas": [f"revisão do código não é limpa: {codigo_revisao}"]}})
    caminho_analise = cfg.work / "analise" / f"analise-val-{cfg.config_hash}.json"
    analise_val = json.loads(caminho_analise.read_text(encoding="utf-8")) if caminho_analise.is_file() else None
    if analise_val is None or not analise_val.get("poder"):
        return (False, {"conferencia": {**conferencia, "aprovado": False,
                                         "falhas": ["análise da validação ausente: o poder precisa estar "
                                                    "calculado antes da abertura do teste"]}})
    reg = registro(cfg, config_path, rows_val, conferencia, analise_val, n_teste, codigo_revisao)
    destino.write_text(json.dumps(reg, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return (True, reg)
