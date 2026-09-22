"""Regressões de proveniência e pareamento, somente com logs sintéticos."""

import json

import pytest

from tcc_rag.cli import _paired_ids, app
from tcc_rag.config import ExperimentConfig
from tcc_rag.logging_io import report_rows


def row(run_id="run", **changes):
    return {
        "run_id": run_id, "config_hash": "cfg", "partition_digest": "partition",
        "split": "val", "condition": "C2", "lang": "pt", "example_id": "e1",
        "db_id": "db1", "ex_correct": True, "em_correct": True,
        "exec_ok": True, **changes,
    }


def write_log(directory, name, rows):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )


def test_evaluate_never_selects_alphabetical_foreign_config(tmp_path):
    write_log(tmp_path, "a-smoke", [row("a-smoke", ex_correct=False)])
    write_log(tmp_path, "z-sweep", [row("z-sweep", config_hash="other")])
    selected = report_rows(tmp_path, config_hash="cfg")
    assert len(selected) == 1
    assert selected[0]["ex_correct"] is False
    with pytest.raises(ValueError, match="incompatível"):
        report_rows(tmp_path, config_hash="cfg", run_ids=["z-sweep"])


def test_repeated_runs_require_explicit_choice_and_cannot_duplicate_items(tmp_path):
    for name in ("a", "b"):
        write_log(tmp_path, name, [row(name)])
    with pytest.raises(ValueError, match="ambígua"):
        report_rows(tmp_path, config_hash="cfg")
    assert len(report_rows(tmp_path, config_hash="cfg", run_ids=["a"])) == 1
    with pytest.raises(ValueError, match="duplicada"):
        report_rows(tmp_path, config_hash="cfg", run_ids=["a", "b"])


@pytest.mark.parametrize("field,value", [("config_hash", "other"),
                                         ("partition_digest", "other"), ("split", "test")])
def test_mixed_identity_fails_instead_of_partially_filtering(tmp_path, field, value):
    write_log(tmp_path, "run", [row(), row(example_id="e2", **{field: value})])
    with pytest.raises(ValueError):
        report_rows(tmp_path, config_hash="cfg")


def test_filters_select_partition_and_split(tmp_path):
    write_log(tmp_path, "val", [row("val")])
    write_log(tmp_path, "test", [row("test", split="test")])
    write_log(tmp_path, "old", [row("old", partition_digest="old")])
    selected = report_rows(tmp_path, config_hash="cfg", split="val", partition_digest="partition")
    assert [r["run_id"] for r in selected] == ["val"]


@pytest.mark.parametrize("field", ["run_id", "partition_digest", "split", "ex_correct"])
def test_missing_provenance_or_result_is_not_accepted(tmp_path, field):
    incomplete = row()
    del incomplete[field]
    write_log(tmp_path, "run", [incomplete])
    with pytest.raises(ValueError):
        report_rows(tmp_path, config_hash="cfg")


def test_corrupt_line_does_not_disappear_from_denominator(tmp_path):
    write_log(tmp_path, "run", [row()])
    with (tmp_path / "run.jsonl").open("a", encoding="utf-8") as fh:
        fh.write('{"truncated":')
    with pytest.raises(ValueError, match="JSON inválido"):
        report_rows(tmp_path, config_hash="cfg")


def test_explicit_disjoint_condition_logs_can_be_combined(tmp_path):
    write_log(tmp_path, "c2", [row("c2")])
    write_log(tmp_path, "c3", [row("c3", condition="C3")])
    assert len(report_rows(tmp_path, config_hash="cfg", run_ids=["c2", "c3"])) == 2


def test_pairing_never_silently_drops_missing_examples():
    with pytest.raises(ValueError, match="pareamento incompleto"):
        _paired_ids({"a": True, "b": False}, {"a": True})


def test_stats_reports_c3_primary_and_preserves_original_h1(tmp_path, capsys):
    cfg = ExperimentConfig()
    cfg.paths.work_dir = str(tmp_path / "artifacts")
    config_path = tmp_path / "config.yaml"
    cfg.dump(config_path)
    rows = [row(config_hash=cfg.config_hash, condition=condition, example_id=f"e{i}",
                ex_correct=(condition == "C3" or i == 0))
            for condition in ("C0", "C2", "C3", "C4") for i in range(2)]
    write_log(cfg.work / "runs", "run", rows)
    assert app(["stats", "--config", str(config_path), "--run-id", "run"]) == 0
    output = capsys.readouterr().out
    assert "Comparação primária: C3 × C2" in output
    assert "H1 original: C4 × C2" in output
    assert "H2 não calculada" in output
    result = json.loads((cfg.work / "stats.json").read_text(encoding="utf-8"))
    assert result["comparisons"]["C3-C2|pt"]["mcnemar"]["diff_pp"] == 50.0
    assert result["selection"]["split"] == "val"


def test_stats_refuses_partial_language_pairing(tmp_path, capsys):
    cfg = ExperimentConfig()
    cfg.paths.work_dir = str(tmp_path / "artifacts")
    config_path = tmp_path / "config.yaml"
    cfg.dump(config_path)
    rows = [row(config_hash=cfg.config_hash, condition=c)
            for c in ("C0", "C2", "C3", "C4")]
    rows += [row(config_hash=cfg.config_hash, condition="C4", lang="en")]
    write_log(cfg.work / "runs", "run", rows)
    assert app(["stats", "--config", str(config_path)]) == 1
    assert "pareamento incompleto" in capsys.readouterr().err
    assert not (cfg.work / "stats.json").exists()


def test_mcnemar_known_exact_probability_and_stratified_paired_bootstrap():
    from tcc_rag.evaluation.stats import mcnemar_exact, paired_bootstrap_diff

    # Seis discordantes no mesmo sentido: binomial bilateral = 2 / 2**6.
    assert mcnemar_exact([True] * 6, [False] * 6).p_value == 0.03125
    # A diferença é constante dentro de cada banco; sua composição fica fixa.
    a, b = [True, True, False, False], [False, False, True, True]
    ci = paired_bootstrap_diff(a, b, ["db1", "db1", "db2", "db2"], resamples=100)
    assert ci.point == ci.low == ci.high == 0.0
