"""Registro de execução: uma linha JSONL por geração.

Este arquivo é a matéria-prima do Capítulo 5. Todas as tabelas, a análise
estatística e a taxonomia de erros saem de consultas sobre ele — nenhum número
é digitado à mão.

Também é o que torna a execução retomável. Em sessão de Colab ou Kaggle, cair
na hora nove de catorze é normal; ao reiniciar, o que já existe é pulado. Desde
a E04, a retomada só pula uma linha com a mesma identidade: o `Runner` confere a
identidade da execução e o `item_sha256` de cada item antes de pular.

Os logs no formato da E04 vão para `runs-v2/`, separados dos históricos de
`runs/`, cujos nomes e conteúdo não podem ser retomados nem misturados.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator

RUNS_DIR = "runs-v2"
"""Diretório dos logs novos, dentro de `paths.work_dir`."""

# Identidade de uma execução, conferida na retomada (`Runner`) e entre os logs
# combinados num relatório (`report_rows`). A de geração diz o que o modelo
# recebeu; a de avaliação, como a saída foi julgada. Quando só a de avaliação
# muda, não é preciso gerar de novo: basta `tcc-rag reavaliar`.
IDENTIDADE_GERACAO = {
    "formato_log": "outro formato de log (anterior à E04?)",
    "config_hash": "outra configuração",
    "partition_digest": "outra partição",
    "split": "outro split",
    "indice_digest": "outro índice de demonstrações",
    "template_sha256": "outro template do prompt final",
}
IDENTIDADE_AVALIACAO = {
    "ex_avaliador": "outro avaliador de EX",
    "em_avaliador": "outro avaliador de EM",
}
IDENTIDADE_RETOMADA = {**IDENTIDADE_GERACAO, **IDENTIDADE_AVALIACAO}

AVALIADOR_EX_HISTORICO = "proprio-anterior-a-E02 (sem ex_status nem ex_avaliador)"
"""Rótulo, só para leitura, dos logs gravados antes da EX oficial."""


def record_key(row: dict[str, Any]) -> tuple:
    return (row["run_id"], row["condition"], row["lang"], row["example_id"])


def isolar_cauda_incompleta(path: str | Path) -> Path | None:
    """Move para um arquivo de incidente a última linha sem quebra de linha.

    Uma queda no meio da escrita deixa a linha final incompleta. Se o arquivo
    fosse reaberto em modo de acréscimo, a próxima linha seria colada nela, e a
    leitura estrita dos relatórios recusaria o log inteiro. O trecho é
    preservado em `<nome>.incidente-<data>.txt`, que nenhum leitor de `*.jsonl`
    apanha.
    """
    path = Path(path)
    if not path.exists():
        return None
    dados = path.read_bytes()
    if not dados or dados.endswith(b"\n"):
        return None
    corte = dados.rfind(b"\n") + 1
    incidente = path.with_name(f"{path.name}.incidente-{time.strftime('%Y%m%dT%H%M%S')}.txt")
    incidente.write_bytes(dados[corte:])
    with path.open("r+b") as fh:
        fh.truncate(corte)
    print(f"[log] AVISO: {path.name} terminava numa linha incompleta ({len(dados) - corte} B), "
          f"provavelmente de uma queda; o trecho foi movido para {incidente.name}",
          file=sys.stderr)
    return incidente


class RunLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.incidente = isolar_cauda_incompleta(self.path)
        # chave de retomada -> item_sha256 gravado (None em logs anteriores à E04)
        self._done: dict[tuple, str | None] = {}
        if self.path.exists():
            for row in self.read():
                self._done[record_key(row)] = row.get("item_sha256")

    # ------------------------------------------------------------- escrita

    def append(self, row: dict[str, Any]) -> None:
        key = record_key(row)
        if key in self._done:
            return
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._done[key] = row.get("item_sha256")

    def has(self, run_id: str, condition: str, lang: str, example_id: str) -> bool:
        return (run_id, condition, lang, example_id) in self._done

    def item_sha256(self, run_id: str, condition: str, lang: str, example_id: str) -> str | None:
        """O `item_sha256` gravado para a chave (None se ausente ou de log antigo)."""
        return self._done.get((run_id, condition, lang, example_id))

    @property
    def n_done(self) -> int:
        return len(self._done)

    # -------------------------------------------------------------- leitura

    def read(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    # Linha truncada por queda de sessão: descarta e segue.
                    continue

    def rows(self) -> list[dict[str, Any]]:
        return list(self.read())


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def row_vazio(valor: Any) -> bool:
    return valor is None or valor == ""


def avaliador_ex(row: dict[str, Any]) -> str:
    """O avaliador de EX de uma linha, exigindo que ele esteja identificado.

    Linha com `ex_avaliador` precisa ter `ex_status` (formato da E02 em diante).
    Linha sem os dois é histórica, do avaliador próprio, e recebe um rótulo só de
    leitura. Qualquer outra combinação é recusada.
    """
    if "ex_avaliador" in row:
        if not isinstance(row["ex_avaliador"], str) or not row["ex_avaliador"] or "ex_status" not in row:
            raise ValueError("linha com ex_avaliador vazio ou sem ex_status")
        return row["ex_avaliador"]
    if "ex_status" in row:
        raise ValueError("linha com ex_status sem ex_avaliador: avaliador de EX não identificado")
    return AVALIADOR_EX_HISTORICO


def avaliador_em(row: dict[str, Any]) -> str | None:
    """O avaliador do EM oficial da linha; None em logs anteriores à E03."""
    if "em_status" not in row:
        return None
    if not isinstance(row.get("em_avaliador"), str) or not row["em_avaliador"]:
        raise ValueError("linha com em_status sem em_avaliador: avaliador de EM não identificado")
    return row["em_avaliador"]


def report_rows(
    runs_dir: Path,
    *,
    config_hash: str,
    run_ids: list[str] | None = None,
    split: str | None = None,
    partition_digest: str | None = None,
) -> list[dict[str, Any]]:
    """Seleciona logs compatíveis, sem escolher por nome nem descartar duplicatas.

    Relatórios usam leitura estrita: uma linha truncada não pode desaparecer do
    denominador. Logs históricos sem split precisam ter sua proveniência
    recuperada do manifesto original antes de entrar em uma análise.
    """
    if run_ids and any(Path(name).name != name or "/" in name or "\\" in name
                       for name in run_ids):
        raise ValueError("--run-id deve conter nomes, sem diretórios")
    paths = ([runs_dir / f"{name}.jsonl" for name in run_ids] if run_ids
             else sorted(runs_dir.glob("*.jsonl")))
    filters = {"config_hash": config_hash}
    if split:
        filters["split"] = split
    if partition_digest:
        filters["partition_digest"] = partition_digest
    selected: list[dict[str, Any]] = []
    selected_paths = []
    for path in paths:
        if not path.is_file():
            raise ValueError(f"log não encontrado: {path}")
        rows = []
        texto = path.read_text(encoding="utf-8")
        linhas_texto = texto.splitlines()
        for number, line in enumerate(linhas_texto, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                if number == len(linhas_texto) and not texto.endswith("\n"):
                    # O relatório não altera arquivos; quem repara é o RunLog.
                    raise ValueError(
                        f"JSON inválido em {path.name}:{number}, a última linha, sem quebra no "
                        "fim: provavelmente uma queda no meio da escrita. Reabra o log com o "
                        f"`tcc-rag run`/`sweep` do mesmo --run-id ({path.stem}), que move o "
                        "trecho para um arquivo de incidente e gera o item de novo."
                    ) from exc
                raise ValueError(f"JSON inválido em {path.name}:{number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"registro inválido em {path.name}:{number}")
            rows.append(row)
        matches = [all(row.get(key) == value for key, value in filters.items())
                   for row in rows]
        if not any(matches):
            if run_ids:
                raise ValueError(f"{path.name}: log vazio ou incompatível com os filtros {filters}")
            continue
        if not all(matches):
            raise ValueError(f"{path.name}: contém registros de identidades diferentes")
        for field in ("run_id", "config_hash", "partition_digest", "split"):
            values = {row.get(field) for row in rows}
            if len(values) != 1 or not all(isinstance(v, str) and v for v in values):
                raise ValueError(f"{path.name}: {field} ausente ou misturado; "
                                 "verifique a proveniência do log original")
        if rows[0]["run_id"] != path.stem:
            raise ValueError(f"{path.name}: run_id interno difere do nome do arquivo")
        if rows[0]["split"] not in {"val", "test"}:
            raise ValueError(f"{path.name}: split inválido")
        selected_paths.append(path)
        selected.extend(rows)
    if not selected:
        raise ValueError("nenhum log corresponde aos filtros; confira configuração e --run-id")
    if not run_ids and len(selected_paths) > 1:
        raise ValueError("seleção ambígua; informe --run-id: "
                         + ", ".join(p.stem for p in selected_paths))
    for field in ("config_hash", "partition_digest", "split"):
        if len({row[field] for row in selected}) != 1:
            raise ValueError(f"não é permitido combinar logs com {field} diferentes")
    # O EM oficial existe a partir da E03. Combinar linhas com e sem ele faria o
    # denominador do EM mudar de item para item sem aviso.
    if len({"em_status" in row for row in selected}) != 1:
        raise ValueError("não é permitido combinar linhas com e sem EM oficial; "
                         "reavalie os logs anteriores à E03")
    # Logs da E04 em diante: toda a identidade de geração precisa ser única e
    # preenchida entre os logs combinados, como dentro de um log na retomada.
    # Logs históricos (sem `formato_log`) só são lidos isolados.
    com_formato = {"formato_log" in row for row in selected}
    if len(com_formato) != 1:
        raise ValueError("não é permitido combinar logs históricos (sem formato_log) com logs "
                         "da E04 em diante; leia os históricos isolados, com --runs-dir")
    if com_formato == {True}:
        for field, descricao in IDENTIDADE_GERACAO.items():
            values = {json.dumps(row.get(field), sort_keys=True) for row in selected}
            if len(values) != 1 or row_vazio(selected[0].get(field)):
                raise ValueError(f"{field} ausente ou misturado entre os logs ({descricao}): "
                                 + "; ".join(sorted(values)))
        if selected[0]["split"] == "test":
            revisoes = {row.get("codigo_revisao") for row in selected}
            revisao = next(iter(revisoes))
            if len(revisoes) != 1 or not revisao or "+" in str(revisao) or revisao == "desconhecida":
                raise ValueError("o teste exige uma única revisão do código, commitada e sem "
                                 "modificações locais: " + "; ".join(sorted(map(str, revisoes))))
    # Um único avaliador de EX e de EM por seleção. O config_hash não muda
    # quando o avaliador muda, então não basta para separar duas definições de
    # EX (por exemplo, as cópias migradas da E01 e um log novo).
    for nome, extrair in (("EX", avaliador_ex), ("EM", avaliador_em)):
        valores = {extrair(row) for row in selected}
        if len(valores) != 1:
            raise ValueError(f"não é permitido combinar avaliadores de {nome} diferentes: "
                             + "; ".join(sorted(str(v) for v in valores)))
    seen = set()
    databases = {}
    for row in selected:
        if any(not row.get(field) for field in ("condition", "lang", "example_id", "db_id")):
            raise ValueError("registro sem condição, idioma, exemplo ou banco")
        if not isinstance(row.get("ex_correct"), bool) and not (
            row.get("ex_correct") is None and row.get("ex_status") == "referencia_invalida"
        ):
            raise ValueError("ex_correct ausente ou não booleano")
        if "em_status" in row and not isinstance(row.get("em_oficial"), bool) and not (
            row.get("em_oficial") is None and row["em_status"] == "referencia_nao_analisavel"
        ):
            raise ValueError("em_oficial ausente ou não booleano")
        key = (row["condition"], row["lang"], row["example_id"])
        if key in seen:
            raise ValueError(f"observação duplicada entre os logs selecionados: {key}")
        seen.add(key)
        previous = databases.setdefault(row["example_id"], row["db_id"])
        if previous != row["db_id"]:
            raise ValueError(f"bancos divergentes para {row['example_id']}")
    return selected
