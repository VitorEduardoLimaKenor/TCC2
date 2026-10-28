"""Acurácia de execução pelo avaliador oficial do Spider (Seção 4.7).

O avaliador é o `exec_eval.py` do repositório test-suite-sql-eval (Zhong, Yu e
Klein, 2020), métrica oficial do Spider desde outubro de 2020. A cópia em
`_spider_oficial/` é idêntica à do commit fixado, e o blob git de cada arquivo é
conferido antes do uso: uma cópia editada faz a avaliação recusar-se a rodar.

Daqui vêm, sem reimplementação, as regras que decidem o acerto: `postprocess`,
`remove_distinct`, `replace_cur_year` e `result_eq` — multiconjunto de linhas,
tolerância à permutação de colunas, ordem das linhas só quando a referência
contém "order by" em qualquer ponto do texto. Antes delas, como no
`evaluation.py` oficial, as duas consultas passam por `strip()` e a predição tem
"value" trocado por "1".

O banco é o original, `database/<db_id>/<db_id>.sqlite`: uma instância. A test
suite destilada só existe para os bancos do desenvolvimento do Spider, e apenas
dois dos 17 bancos do recorte estão lá.

Só a execução difere da oficial, que abre o banco para escrita e cujo tempo
limite não interrompe a consulta (`asyncio.wait_for` em volta de uma chamada
síncrona). Aqui cada consulta roda no `Sandbox`. Para consultas de leitura que
terminam no prazo, o veredito é o mesmo; o resto vira situação explícita:

- referência que falha ou estoura o tempo: `referencia_invalida`, sem veredito
  (`correto=None`) — a oficial interrompe a avaliação com `assert`;
- predição que falha, escreve ou tem várias instruções: `erro_predicao`;
- predição que estoura o tempo: `tempo_predicao`, contada como erro.

A predição é lida até uma linha além da referência. Com mais linhas que ela, os
tamanhos já diferem e a oficial também a rejeitaria, então nenhum corte altera o
veredito.
"""

from __future__ import annotations

import functools
import hashlib
import importlib.util
import random
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from ..schemas import ExecResult
from .sandbox import Sandbox

COMMIT = "e97acc546ecbee8fa27fa8dbf025ef61493a876c"
REPOSITORIO = "https://github.com/taoyds/test-suite-sql-eval"
_PASTA = Path(__file__).with_name("_spider_oficial")
# Todos os arquivos da cópia são conferidos a cada carga, inclusive os do EM
# oficial (`em_oficial`), que importa `exec_eval` por meio do `evaluation.py`.
_BLOBS = {
    "exec_eval.py": "9bc34ba75eff4446223248fefb99fc429b7d23e7",
    "parse.py": "5271f5f86c9de2e0171cad8c690d7a885752f7e0",
    "evaluation.py": "fae509c6c3afe60635e0ae9971c2204be7637005",
    "process_sql.py": "fd93ea287dd80f7518c995767917a4cc1012d626",
}

COINCIDE = "coincide"
DIFERE = "difere"
ERRO_PREDICAO = "erro_predicao"
TEMPO_PREDICAO = "tempo_predicao"
REFERENCIA_INVALIDA = "referencia_invalida"


@dataclass(frozen=True)
class Veredito:
    correto: bool | None
    """None apenas com referência inválida: o item sai do denominador."""
    situacao: str
    ordem_importa: bool | None = None
    linhas_ref: int | None = None
    linhas_pred: int | None = None
    """Com mais linhas que a referência, vale `linhas_ref + 1`: o restante não é lido."""
    ambos_vazios: bool = False
    """Acerto por dois resultados vazios — regra oficial, e o caso em que uma
    única instância do banco mais produz falso positivo."""
    pred_executou: bool | None = None
    erro: str | None = None


def blob_git(caminho: Path) -> str:
    """Identificador do arquivo no git, com fim de linha normalizado para LF."""
    dados = caminho.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(dados) + dados).hexdigest()


def _modulo(nome: str, caminho: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(nome, caminho)
    modulo = importlib.util.module_from_spec(spec)
    with warnings.catch_warnings():
        # O código de 2020 usa "\s" e "\d" em strings que não são cruas.
        warnings.simplefilter("ignore", SyntaxWarning)
        spec.loader.exec_module(modulo)
    return modulo


def carregar_oficial(pasta: Path = _PASTA) -> tuple[ModuleType, ModuleType]:
    """Carrega `exec_eval` e `parse` da cópia fixada, depois de conferi-la."""
    for arquivo, esperado in _BLOBS.items():
        obtido = blob_git(pasta / arquivo)
        if obtido != esperado:
            raise RuntimeError(
                f"cópia do avaliador oficial alterada: {arquivo} tem blob {obtido}, "
                f"esperado {esperado} ({REPOSITORIO} @ {COMMIT})"
            )
    parse = _modulo("tcc_rag_spider_oficial_parse", pasta / "parse.py")
    # exec_eval.py faz `from parse import ...`. O nome fica disponível só durante
    # a carga, para não sombrear um pacote "parse" instalado no ambiente.
    anterior = sys.modules.get("parse")
    sys.modules["parse"] = parse
    try:
        exec_eval = _modulo("tcc_rag_spider_oficial_exec_eval", pasta / "exec_eval.py")
    finally:
        if anterior is None:
            sys.modules.pop("parse", None)
        else:
            sys.modules["parse"] = anterior
    return exec_eval, parse


@functools.lru_cache(maxsize=1)
def _oficial() -> tuple[ModuleType, ModuleType]:
    return carregar_oficial()


def preparar_referencia(gold_sql: str, keep_distinct: bool = False) -> str:
    """evaluation.py, linha 514 (`strip`); exec_eval.py, linhas 187–190."""
    exec_eval, parse = _oficial()
    gold = exec_eval.postprocess(gold_sql.strip())
    return gold if keep_distinct else parse.remove_distinct(gold)


def preparar_predicao(pred_sql: str, keep_distinct: bool = False) -> str:
    """evaluation.py, linhas 535 e 568–569; exec_eval.py, linhas 187–190.

    A troca de "value" por "1" vem de modelos antigos que escreviam o marcador
    "value" no lugar dos literais. Ela atinge qualquer ocorrência minúscula, até
    em nome de coluna ou dentro de texto, e é aplicada porque a oficial aplica.
    Sem `keep_distinct`, todo DISTINCT é removido, inclusive de COUNT(DISTINCT x).
    """
    exec_eval, parse = _oficial()
    pred = exec_eval.postprocess(pred_sql.strip().replace("value", "1"))
    return pred if keep_distinct else parse.remove_distinct(pred)


def comparar(ref_rows: list[tuple], pred_rows: list[tuple], ordem_importa: bool) -> bool:
    """`result_eq` oficial, com a referência no primeiro argumento, como na oficial.

    A função sorteia linhas com o `random` global para podar permutações de
    colunas. O sorteio não muda o veredito, mas mudaria o estado global de quem
    viesse depois; o estado é restaurado.
    """
    exec_eval, _ = _oficial()
    estado = random.getstate()
    try:
        return bool(exec_eval.result_eq(list(ref_rows), list(pred_rows), order_matters=ordem_importa))
    finally:
        random.setstate(estado)


class AvaliadorEX:
    """EX oficial sobre o banco original, com execução pelo `Sandbox`."""

    def __init__(self, sandbox: Sandbox, keep_distinct: bool = False):
        self.sandbox = sandbox
        self.keep_distinct = keep_distinct
        self._referencias: dict[tuple[str, str], ExecResult] = {}

    @property
    def identificador(self) -> str:
        return (
            f"spider-oficial exec_eval test-suite-sql-eval@{COMMIT[:7]}; banco original; "
            f"keep_distinct={self.keep_distinct}; plug_value=False"
        )

    def _executar_referencia(self, db_path: str | Path, gold: str) -> ExecResult:
        # O resultado da referência se repete em todas as condições e nos dois
        # idiomas; é calculado uma vez por (banco, consulta).
        chave = (str(db_path), gold)
        if chave not in self._referencias:
            exec_eval, _ = _oficial()
            self._referencias[chave] = self.sandbox.run(db_path, exec_eval.replace_cur_year(gold))
        return self._referencias[chave]

    def avaliar(self, db_path: str | Path, pred_sql: str, gold_sql: str) -> Veredito:
        exec_eval, _ = _oficial()
        try:
            gold = preparar_referencia(gold_sql, self.keep_distinct)
        except Exception as exc:  # noqa: BLE001
            return Veredito(None, REFERENCIA_INVALIDA, erro=f"referência: {type(exc).__name__}: {exc}")
        ordem = "order by" in gold.lower()  # exec_eval.py, linha 197
        ref = self._executar_referencia(db_path, gold)

        pred, erro_preparo = None, None
        if (pred_sql or "").strip():
            try:
                pred = exec_eval.replace_cur_year(preparar_predicao(pred_sql, self.keep_distinct))
            except Exception as exc:  # noqa: BLE001
                erro_preparo = f"preparo: {type(exc).__name__}: {exc}"
        else:
            erro_preparo = "consulta vazia"

        if not ref.ok:
            # Sem referência não há veredito. A predição roda mesmo assim, só para
            # registrar se executa; uma linha basta.
            executou = self.sandbox.run(db_path, pred, limite=0).ok if pred else False
            erro = "referência: " + ("timeout" if ref.timed_out else str(ref.error))
            return Veredito(None, REFERENCIA_INVALIDA, ordem, pred_executou=executou, erro=erro)

        linhas_ref = len(ref.rows or [])
        if pred is None:
            return Veredito(False, ERRO_PREDICAO, ordem, linhas_ref, pred_executou=False, erro=erro_preparo)

        res = self.sandbox.run(db_path, pred, limite=linhas_ref)
        if res.timed_out:
            return Veredito(False, TEMPO_PREDICAO, ordem, linhas_ref, pred_executou=False, erro="timeout")
        if not res.ok:
            return Veredito(False, ERRO_PREDICAO, ordem, linhas_ref, pred_executou=False, erro=res.error)
        if res.truncated:
            return Veredito(False, DIFERE, ordem, linhas_ref, linhas_ref + 1, pred_executou=True)

        pred_rows = res.rows or []
        correto = comparar(ref.rows or [], pred_rows, ordem)
        return Veredito(
            correto,
            COINCIDE if correto else DIFERE,
            ordem,
            linhas_ref,
            len(pred_rows),
            ambos_vazios=correto and linhas_ref == 0,
            pred_executou=True,
        )
