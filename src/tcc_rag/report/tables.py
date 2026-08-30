"""Geração das tabelas LaTeX do Capítulo 5.

Escreve arquivos prontos para `\\input` no `.tex`. Nenhum número da monografia
é digitado à mão: se o log mudar, as tabelas mudam junto.
"""

from __future__ import annotations

from pathlib import Path

from ..evaluation import metrics

_HEADER = "%% Gerado por tcc-rag tables. Não editar à mão."

_LABELS = {
    "C0": "C0 -- Sem demonstrações",
    "C1": "C1 -- Exemplos fixos",
    "C2": "C2 -- Similaridade da pergunta",
    "C3": "C3 -- Pergunta delexicalizada",
    "C4": "C4 -- Duas etapas",
}
_ORDER = ("C0", "C1", "C2", "C3", "C4")
_DIFFICULTY = (("easy", "Fácil"), ("medium", "Médio"), ("hard", "Difícil"), ("extra", "Extra"))


def _pct(value: float) -> str:
    return f"{100 * value:.1f}".replace(".", ",")


def table_by_difficulty(summary: dict, lang: str = "pt") -> str:
    lines = [
        _HEADER,
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Acurácia de execução em português brasileiro por nível de dificuldade.}",
        "\\footnotesize",
        "\\begin{tabular}{l|c|c|c|c|c|c}",
        "\\hline",
        "\\textbf{Condição} & \\textbf{Fácil} & \\textbf{Médio} & \\textbf{Difícil} & "
        "\\textbf{Extra} & \\textbf{Total} & \\textbf{Não exec.} \\\\",
        "\\hline",
    ]
    for condition in _ORDER:
        entry = summary.get(f"{condition}|{lang}")
        if not entry:
            continue
        cells = []
        for key, _ in _DIFFICULTY:
            slot = entry["by_difficulty"].get(key)
            cells.append(_pct(slot["ex"]) if slot else "--")
        lines.append(
            f"{_LABELS[condition]} & " + " & ".join(cells)
            + f" & {_pct(entry['ex'])} & {entry['not_executable']} \\\\"
        )
        lines.append("\\hline")
    lines += [
        "\\end{tabular}",
        "\\vspace{0.1\\baselineskip}",
        "\\newline \\footnotesize \\textbf{Fonte: Autoria própria (2026).}",
        "\\label{tab:resultados_ptbr}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def table_languages(summary: dict) -> str:
    lines = [
        _HEADER,
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Acurácia de execução por idioma e diferença entre eles.}",
        "\\footnotesize",
        "\\begin{tabular}{l|c|c|c}",
        "\\hline",
        "\\textbf{Condição} & \\textbf{Inglês} & \\textbf{Português} & \\textbf{Diferença} \\\\",
        "\\hline",
    ]
    for condition in _ORDER:
        en = summary.get(f"{condition}|en")
        pt = summary.get(f"{condition}|pt")
        if not en or not pt:
            continue
        diff = 100 * (en["ex"] - pt["ex"])
        lines.append(
            f"{_LABELS[condition]} & {_pct(en['ex'])} & {_pct(pt['ex'])} & "
            f"{diff:.1f}".replace(".", ",") + " \\\\"
        )
        lines.append("\\hline")
    lines += [
        "\\end{tabular}",
        "\\vspace{0.1\\baselineskip}",
        "\\newline \\footnotesize \\textbf{Fonte: Autoria própria (2026).}",
        "\\label{tab:idiomas}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def table_cost(summary: dict, lang: str = "pt") -> str:
    baseline = summary.get(f"C0|{lang}")
    lines = [
        _HEADER,
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Acurácia, custo computacional e ganho marginal por condição.}",
        "\\footnotesize",
        "\\begin{tabular}{l|c|c|c|c|c}",
        "\\hline",
        "\\textbf{Condição} & \\textbf{EX (\\%)} & \\textbf{Tokens ent.} & "
        "\\textbf{Lat. p50 (ms)} & \\textbf{Lat. p95 (ms)} & \\textbf{Ganho marg.} \\\\",
        "\\hline",
    ]
    for condition in _ORDER:
        entry = summary.get(f"{condition}|{lang}")
        if not entry:
            continue
        if condition == "C0" or not baseline:
            marginal = "---"
        else:
            extra_tokens = entry["prompt_tokens_mean"] - baseline["prompt_tokens_mean"]
            gain = 100 * (entry["ex"] - baseline["ex"])
            marginal = (
                f"{gain / (extra_tokens / 1000):.2f}".replace(".", ",")
                if extra_tokens > 0
                else "---"
            )
        lines.append(
            f"{_LABELS[condition]} & {_pct(entry['ex'])} & "
            f"{entry['prompt_tokens_mean']:.0f} & {entry['latency_p50']:.0f} & "
            f"{entry['latency_p95']:.0f} & {marginal} \\\\"
        )
        lines.append("\\hline")
    lines += [
        "\\end{tabular}",
        "\\vspace{0.1\\baselineskip}",
        "\\newline \\footnotesize \\textbf{Fonte: Autoria própria (2026).}",
        "\\label{tab:custo}",
        "\\end{table}",
        "%% Ganho marginal: pontos percentuais de EX por mil tokens de entrada",
        "%% adicionais em relação à condição sem demonstrações.",
    ]
    return "\n".join(lines)


def table_retriever(summary: dict, lang: str = "pt") -> str:
    lines = [
        _HEADER,
        "\\begin{table}[H]",
        "\\centering",
        "\\caption{Acerto estrutural entre os $k$ primeiros resultados recuperados.}",
        "\\footnotesize",
        "\\begin{tabular}{l|c|c}",
        "\\hline",
        "\\textbf{Condição} & \\textbf{Acerto estrutural (\\%)} & "
        "\\textbf{Recurso à ordenação inicial (\\%)} \\\\",
        "\\hline",
    ]
    for condition in ("C2", "C3", "C4"):
        entry = summary.get(f"{condition}|{lang}")
        if not entry:
            continue
        fallback = _pct(entry["fallback_rate"]) if condition == "C4" else "---"
        lines.append(
            f"{_LABELS[condition]} & {_pct(entry['structural_hit'])} & {fallback} \\\\"
        )
        lines.append("\\hline")
    lines += [
        "\\end{tabular}",
        "\\vspace{0.1\\baselineskip}",
        "\\newline \\footnotesize \\textbf{Fonte: Autoria própria (2026).}",
        "\\label{tab:recuperador}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def write_all(rows: list[dict], out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary = metrics.summarize(rows)

    files = {
        "tab_resultados_ptbr.tex": table_by_difficulty(summary, "pt"),
        "tab_idiomas.tex": table_languages(summary),
        "tab_custo.tex": table_cost(summary, "pt"),
        "tab_recuperador.tex": table_retriever(summary, "pt"),
    }
    written = []
    for name, content in files.items():
        path = out / name
        path.write_text(content + "\n", encoding="utf-8")
        written.append(path)
    return written
