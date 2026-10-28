"""Correspondência exata e dificuldade pelo avaliador oficial do Spider (Seção 4.7).

O código vem do `evaluation.py` e do `process_sql.py` do test-suite-sql-eval, no
mesmo commit da EX (`ex_oficial`), copiados sem alteração para `_spider_oficial/`
e conferidos pelo blob git antes do uso. É o que `evaluation.py --etype match`
calcula: a "exact set match accuracy" original do Spider, que o repositório
continua reportando como referência ao lado da EX.

Correspondência exata (EM). As duas consultas são analisadas pelo `get_sql`
oficial, contra o esquema lido do próprio banco. `rebuild_sql_val` descarta os
valores literais (`DISABLE_VALUE`); `rebuild_sql_col` desliga o DISTINCT
(`DISABLE_DISTINCT`) e troca cada coluna de chave estrangeira pela coluna
canônica do seu grupo, segundo o `tables.json`. Depois,
`Evaluator.eval_exact_match` compara cláusula a cláusula, como conjuntos: a
ordem das colunas do SELECT, das condições e das tabelas do FROM não importa,
e os apelidos (T1, T2) desaparecem na análise. Os passos repetem as linhas
568–628 do `evaluation.py`: `strip()` das duas consultas, "value" → "1" na
predição e, se a predição não for analisável, comparação com a consulta vazia
oficial — o EM é falso.

Dificuldade. `Evaluator.eval_hardness` sobre a referência analisada, antes da
reconstrução, como na linha 575: easy, medium, hard ou extra.

O analisador oficial tokeniza com `nltk.word_tokenize`, que precisa do modelo
`punkt_tab`. Sem ele, a carga falha com a instrução de instalação, em vez de
cair num tokenizador diferente.

As diferenças em relação à oficial aparecem todas no veredito:

- referência que não é analisável: `referencia_nao_analisavel`, sem veredito
  (`correto=None`) e sem dificuldade. A oficial interrompe a avaliação;
- falha na comparação depois de analisadas as duas consultas:
  `erro_comparacao`, contada como erro. A oficial também seria interrompida;
- predição vazia, que o arquivo de predições oficial não representa (uma
  linha vazia separa sessões): `predicao_nao_analisavel`, como as demais.

Esta correspondência não substitui a forma canônica que define as famílias da
partição (`data.normalize`), nem é substituída por ela. A medida própria
continua disponível com outro nome, em `metrics.em_canonico_proprio`.
"""

from __future__ import annotations

import functools
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .ex_oficial import _PASTA, COMMIT, _modulo, carregar_oficial

COINCIDE = "coincide"
DIFERE = "difere"
PREDICAO_NAO_ANALISAVEL = "predicao_nao_analisavel"
REFERENCIA_NAO_ANALISAVEL = "referencia_nao_analisavel"
ERRO_COMPARACAO = "erro_comparacao"

NIVEIS = ("easy", "medium", "hard", "extra")


@dataclass(frozen=True)
class VereditoEM:
    correto: bool | None
    """None apenas com referência não analisável: o item sai do denominador do EM."""
    situacao: str
    dificuldade: str | None
    """Nível oficial da referência; None quando ela não é analisável."""
    componentes_divergentes: tuple[str, ...] = ()
    """Componentes da comparação parcial oficial com F1 diferente de 1."""
    erro: str | None = None


def _exigir_tokenizador() -> str:
    """Confere o nltk e o modelo `punkt_tab`; devolve a versão do nltk."""
    try:
        import nltk
    except ImportError as exc:
        raise RuntimeError(
            "o EM oficial do Spider usa nltk.word_tokenize: instale o nltk fixado no pyproject.toml"
        ) from exc
    try:
        nltk.data.find("tokenizers/punkt_tab/english/")
    except LookupError as exc:
        raise RuntimeError(
            "o EM oficial do Spider usa nltk.word_tokenize, que precisa do modelo punkt_tab: "
            "python -m nltk.downloader punkt_tab"
        ) from exc
    return nltk.__version__


def carregar_match(pasta: Path = _PASTA) -> tuple[ModuleType, ModuleType]:
    """Carrega `evaluation` e `process_sql` da cópia fixada, depois de conferi-la."""
    exec_eval, _ = carregar_oficial(pasta)  # confere o blob de todos os arquivos
    _exigir_tokenizador()
    process_sql = _modulo("tcc_rag_spider_oficial_process_sql", pasta / "process_sql.py")
    # evaluation.py faz `from process_sql import ...` e `from exec_eval import ...`.
    # Como em `carregar_oficial`, os nomes só existem durante a carga.
    anteriores = {nome: sys.modules.get(nome) for nome in ("process_sql", "exec_eval")}
    sys.modules["process_sql"] = process_sql
    sys.modules["exec_eval"] = exec_eval
    try:
        evaluation = _modulo("tcc_rag_spider_oficial_evaluation", pasta / "evaluation.py")
    finally:
        for nome, anterior in anteriores.items():
            if anterior is None:
                sys.modules.pop(nome, None)
            else:
                sys.modules[nome] = anterior
    return evaluation, process_sql


@functools.lru_cache(maxsize=1)
def _match() -> tuple[ModuleType, ModuleType]:
    return carregar_match()


def _consulta_vazia() -> dict[str, Any]:
    """evaluation.py, linhas 589–606: o que a oficial compara quando a predição
    não é analisável."""
    return {
        "except": None,
        "from": {"conds": [], "table_units": []},
        "groupBy": [],
        "having": [],
        "intersect": None,
        "limit": None,
        "orderBy": [],
        "select": [False, []],
        "union": None,
        "where": [],
    }


class AvaliadorEM:
    """EM e dificuldade oficiais, com esquema e chaves estrangeiras por banco."""

    def __init__(self):
        _match()  # falha já na criação se faltar o nltk ou o punkt_tab
        self._esquemas: dict[str, Any] = {}
        self._mapas: dict[str, dict[str, str]] = {}

    @property
    def identificador(self) -> str:
        import nltk

        return (
            f"spider-oficial evaluation.eval_exact_match test-suite-sql-eval@{COMMIT[:7]}; "
            f"sem valores; sem DISTINCT; chaves estrangeiras do tables.json; nltk={nltk.__version__}"
        )

    def _esquema(self, db_path: str | Path):
        """`Schema(get_schema(db))`, como na linha 573, lido uma vez por banco."""
        caminho = str(db_path)
        if caminho not in self._esquemas:
            # get_schema abre o arquivo com sqlite3.connect, que criaria um banco
            # vazio num caminho errado em vez de falhar.
            if not Path(caminho).is_file():
                raise FileNotFoundError(f"banco não encontrado: {caminho}")
            _, process_sql = _match()
            self._esquemas[caminho] = process_sql.Schema(process_sql.get_schema(caminho))
        return self._esquemas[caminho]

    def _mapa(self, tabela: dict[str, Any]) -> dict[str, str]:
        """`build_foreign_key_map` oficial sobre a entrada do banco no tables.json."""
        db_id = tabela["db_id"]
        if db_id not in self._mapas:
            evaluation, _ = _match()
            self._mapas[db_id] = evaluation.build_foreign_key_map(tabela)
        return self._mapas[db_id]

    def _analisar_referencia(self, schema, gold_sql: str):
        _, process_sql = _match()
        return process_sql.get_sql(schema, gold_sql.strip())

    def dificuldade(self, db_path: str | Path, gold_sql: str) -> str | None:
        """Nível oficial da consulta; None se o analisador oficial a recusar."""
        evaluation, _ = _match()
        schema = self._esquema(db_path)
        try:
            g_sql = self._analisar_referencia(schema, gold_sql)
        except Exception:  # noqa: BLE001
            return None
        return evaluation.Evaluator().eval_hardness(g_sql)

    def avaliar(
        self, db_path: str | Path, tabela: dict[str, Any], pred_sql: str, gold_sql: str
    ) -> VereditoEM:
        evaluation, process_sql = _match()
        schema = self._esquema(db_path)
        try:
            g_sql = self._analisar_referencia(schema, gold_sql)
        except Exception as exc:  # noqa: BLE001
            return VereditoEM(
                None, REFERENCIA_NAO_ANALISAVEL, None, erro=f"referência: {type(exc).__name__}: {exc}"
            )
        avaliador = evaluation.Evaluator()
        dificuldade = avaliador.eval_hardness(g_sql)

        erro = None
        try:
            p_sql = process_sql.get_sql(schema, (pred_sql or "").strip().replace("value", "1"))
        except Exception as exc:  # noqa: BLE001
            p_sql = _consulta_vazia()
            erro = f"predição: {type(exc).__name__}: {exc}"

        try:
            kmap = self._mapa(tabela)
            g_validas = evaluation.build_valid_col_units(g_sql["from"]["table_units"], schema)
            g_sql = evaluation.rebuild_sql_col(g_validas, evaluation.rebuild_sql_val(g_sql), kmap)
            p_validas = evaluation.build_valid_col_units(p_sql["from"]["table_units"], schema)
            p_sql = evaluation.rebuild_sql_col(p_validas, evaluation.rebuild_sql_val(p_sql), kmap)
            nota = avaliador.eval_exact_match(p_sql, g_sql)
        except Exception as exc:  # noqa: BLE001
            return VereditoEM(
                False, ERRO_COMPARACAO, dificuldade, erro=f"comparação: {type(exc).__name__}: {exc}"
            )

        divergentes = tuple(
            nome for nome, nota_parcial in avaliador.partial_scores.items() if nota_parcial["f1"] != 1
        )
        correto = bool(nota)
        if erro:
            situacao = PREDICAO_NAO_ANALISAVEL
        else:
            situacao = COINCIDE if correto else DIFERE
        return VereditoEM(correto, situacao, dificuldade, divergentes, erro)
