"""Linha de comando.

Sequência normal de uso:

    tcc-rag prepare    --config configs/dev-local.yaml
    tcc-rag partition  --config configs/dev-local.yaml
    tcc-rag leakage    --config configs/dev-local.yaml     # porta obrigatória
    tcc-rag index      --config configs/dev-local.yaml
    tcc-rag run        --config configs/dev-local.yaml --split test
    tcc-rag evaluate   --config configs/dev-local.yaml
    tcc-rag stats      --config configs/dev-local.yaml
    tcc-rag tables     --config configs/dev-local.yaml

Todos os comandos são idempotentes. `run` é retomável: o que já está no log é
pulado, o que permite atravessar quedas de sessão no Colab ou no Kaggle.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from .config import ALL_CONDITIONS, ALL_LANGUAGES, ExperimentConfig
from .data import leakage as leak
from .data import partition as part
from .data.spider import load_examples, load_schemas
from .logging_io import RunLog, write_json
from .pipeline import build_demos, build_test_items, load_manifest, manifest_path


DEFAULT_CONFIG = "configs/dev-local.yaml"


def _config_path(args) -> str:
    """Aceita --config antes ou depois do subcomando.

    `tcc-rag --config X prepare` e `tcc-rag prepare --config X` são a mesma
    coisa; a segunda forma é a natural e é a que a documentação usa.
    """
    return getattr(args, "config_sub", None) or getattr(args, "config_top", None) or DEFAULT_CONFIG


def _cfg(args) -> ExperimentConfig:
    return ExperimentConfig.load(_config_path(args))


# --------------------------------------------------------------- comandos


def cmd_prepare(args) -> int:
    cfg = _cfg(args)
    examples = load_examples(cfg.paths.spider_root, cfg.paths.questions_pt)
    print(f"exemplos com par PT/EN: {len(examples)}")

    report = part.database_report(examples)
    write_json(cfg.work / "db_report.json", report)

    eligible = part.eligible_databases(examples, cfg.data)
    print(f"bancos elegíveis (>= {cfg.data.min_questions_per_db} perguntas, "
          f">= {cfg.data.min_families_per_db} famílias): {len(eligible)}")
    for row in report[:15]:
        mark = "*" if row["db_id"] in eligible else " "
        print(f" {mark} {row['db_id']:<28} perguntas={row['n_questions']:<5} "
              f"famílias={row['n_families']}")

    if not eligible:
        print("\nPORTA 04/09 REPROVADA: nenhum banco atende ao critério.", file=sys.stderr)
        print("Reduza min_families_per_db, amplie a fonte de dados ou migre de benchmark.",
              file=sys.stderr)
        return 1

    write_json(cfg.work / "examples_raw.json", {"examples": [asdict(e) for e in examples]})
    print(f"\nescrito: {cfg.work / 'examples_raw.json'}")
    return 0


def cmd_partition(args) -> int:
    from .logging_io import read_json
    from .schemas import Example

    cfg = _cfg(args)
    payload = read_json(cfg.work / "examples_raw.json")
    examples = [Example(**row) for row in payload["examples"]]

    assigned = part.partition(examples, cfg.data)
    report = leak.audit(assigned, cfg.data.near_duplicate_threshold)

    if report.near_duplicates:
        print(f"removendo {len(report.near_duplicates)} paráfrases do pool")
        assigned = leak.drop_near_duplicates(assigned, report)
        report = leak.audit(assigned, cfg.data.near_duplicate_threshold)

    print(report.render())

    from .pipeline import environment_fingerprint, partition_digest

    digest = partition_digest(assigned)
    ambiente = environment_fingerprint()

    write_json(manifest_path(cfg), {
        "config_hash": cfg.config_hash,
        "partition_digest": digest,
        "environment": ambiente,
        "summary": part.summarize(assigned),
        "examples": [asdict(e) for e in assigned],
    })
    write_json(cfg.work / "leakage_report.json", asdict(report))
    print(f"\npartition_digest: {digest}")
    print(f"ambiente: python {ambiente['python']}, sqlglot {ambiente['sqlglot']}")
    print(f"escrito: {manifest_path(cfg)}")
    return 0 if report.clean else 1


def cmd_leakage(args) -> int:
    from .pipeline import partition_digest

    cfg = _cfg(args)
    examples = load_manifest(cfg)
    report = leak.audit(examples, cfg.data.near_duplicate_threshold)
    print(report.render())
    print(f"  partition_digest: {partition_digest(examples)}")
    return 0 if report.clean else 1


def cmd_index(args) -> int:
    cfg = _cfg(args)
    from .retrieval.embedder import Embedder
    from .retrieval.index import DemoIndex

    examples = load_manifest(cfg)
    demos = build_demos(examples, partition=part.POOL)
    if not demos:
        print("pool vazio: rode `partition` antes", file=sys.stderr)
        return 1

    embedder = Embedder(cfg.embedding, cfg.work / "cache")
    vectors = embedder.encode([d.question_delex for d in demos], kind="passage")

    index = DemoIndex(cfg.qdrant, dimension=vectors.shape[1])
    index.recreate()
    total = index.upsert(demos, vectors)
    print(f"indexadas {total} demonstrações ({len(demos)//2} exemplos × 2 idiomas)")
    for lang in ALL_LANGUAGES:
        print(f"  {lang}: {index.count(lang=lang, partition=part.POOL)}")
    index.close()
    return 0


def cmd_run(args) -> int:
    cfg = _cfg(args)
    from .generation.llm import make_backend
    from .retrieval.embedder import Embedder
    from .retrieval.index import DemoIndex
    from .retrieval.selectors import DemoSelector
    from .runner import Runner

    examples = load_manifest(cfg)
    split = args.split
    items = build_test_items(
        examples,
        partition=split,
        languages=tuple(args.langs) if args.langs else ALL_LANGUAGES,
    )
    if args.limit:
        from .pipeline import subsample
        items = subsample(items, args.limit, seed=cfg.data.seed)
    if not items:
        print(f"nenhum item na partição {split}", file=sys.stderr)
        return 1

    schemas = load_schemas(cfg.paths.spider_root)
    pool = build_demos(examples, partition=part.POOL)

    embedder = Embedder(cfg.embedding, cfg.work / "cache")
    index = DemoIndex(cfg.qdrant, dimension=embedder.dimension)
    selector = DemoSelector(cfg, index, embedder, pool)
    backend = make_backend(cfg.generation)

    from .pipeline import partition_digest

    digest = partition_digest(examples)
    run_id = args.run_id or f"{cfg.name}-{split}-{cfg.config_hash}"
    log = RunLog(cfg.work / f"runs/{run_id}.jsonl")
    print(f"run_id={run_id}  já registradas={log.n_done}  itens={len(items)}")
    print(f"partition_digest={digest}")

    try:
        runner = Runner(cfg, backend, selector, schemas, log, run_id, digest, split=split)
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        index.close()
        return 1
    conditions = tuple(args.conditions) if args.conditions else ALL_CONDITIONS
    written = runner.run(items, conditions)

    for condition, n in written.items():
        print(f"  {condition}: +{n} linhas")
    print(f"log: {log.path}")
    index.close()
    return 0


def cmd_sweep(args) -> int:
    """Varredura da fase de validacao (Secao 4.6.3).

    Carrega o modelo uma unica vez e percorre a grade. Invocar `run` em laco
    recarregaria o modelo a cada combinacao -- com um 7B sao uns dois minutos
    por combinacao, jogados fora. So toca a validacao; o teste fica selado.
    """
    cfg = _cfg(args)
    from .generation.llm import make_backend
    from .pipeline import partition_digest
    from .retrieval.embedder import Embedder
    from .retrieval.index import DemoIndex
    from .pipeline import item_set_digest, subsample
    from .sweep import render, run_sweep, best

    examples = load_manifest(cfg)
    items = build_test_items(
        examples, partition="val",
        languages=tuple(args.langs) if args.langs else ("pt",),
    )
    if args.limit:
        items = subsample(items, args.limit, seed=cfg.data.seed)
    if not items:
        print("conjunto de validacao vazio", file=sys.stderr)
        return 1

    schemas = load_schemas(cfg.paths.spider_root)
    pool = build_demos(examples, partition=part.POOL)
    embedder = Embedder(cfg.embedding, cfg.work / "cache")
    index = DemoIndex(cfg.qdrant, dimension=embedder.dimension)
    backend = make_backend(cfg.generation)

    total = len(args.k) * len(args.lambdas) * len(args.n_candidates)
    bancos = len({i.db_id for i in items})
    print(f"itens={len(items)}  bancos={bancos}  combinacoes={total}  "
          f"condicoes={' '.join(args.conditions)}")

    linhas = run_sweep(
        cfg, items, pool, schemas, index, embedder, backend,
        partition_digest(examples),
        ks=args.k, lambdas=args.lambdas, ns=args.n_candidates,
        item_digest=item_set_digest(items),
        conditions=args.conditions, on_progress=print,
    )

    write_json(cfg.work / "sweep.json", [l.as_dict() for l in linhas])
    print()
    print(render(linhas))
    melhor = best(linhas)
    if melhor:
        print()
        print(f"melhor: k={melhor.k}  lambda={melhor.lambda_question}  "
              f"N={melhor.n_candidates}  EX={melhor.ex * 100:.1f}%  "
              f"({melhor.condicao} em {melhor.lang})")
    print(f"escrito: {cfg.work / 'sweep.json'}")
    index.close()
    return 0


def cmd_evaluate(args) -> int:
    cfg = _cfg(args)
    from .evaluation import metrics

    rows = _report_rows(cfg, args)
    if not rows:
        print("log vazio", file=sys.stderr)
        return 1

    summary = metrics.summarize(rows)
    write_json(cfg.work / "summary.json", summary)
    write_json(cfg.work / "summary_selection.json", _selection(rows))

    print(f"{'cond|lang':<12} {'n':>5} {'EX':>7} {'EM':>7} {'n/exec':>7} "
          f"{'hit@k':>7} {'fallb':>7} {'tok':>8}")
    for key, value in summary.items():
        print(f"{key:<12} {value['n']:>5} {value['ex']*100:>6.1f}% "
              f"{value['em']*100:>6.1f}% {value['not_executable']:>7} "
              f"{value['structural_hit']*100:>6.1f}% {value['fallback_rate']*100:>6.1f}% "
              f"{value['prompt_tokens_mean']:>8.0f}")
    return 0


def cmd_stats(args) -> int:
    cfg = _cfg(args)
    from .evaluation import stats

    rows = _report_rows(cfg, args)
    if not rows:
        print("log vazio", file=sys.stderr)
        return 1

    def vector(condition: str, lang: str) -> tuple[list[str], dict[str, bool], dict[str, str]]:
        ids, ok, db = [], {}, {}
        for row in rows:
            # Referência inválida não tem veredito: o item sai de todas as
            # condições, e o pareamento continua exigindo os mesmos IDs.
            if row.get("ex_correct") is None:
                continue
            if row["condition"] == condition and row["lang"] == lang:
                ids.append(row["example_id"])
                ok[row["example_id"]] = bool(row.get("ex_correct"))
                db[row["example_id"]] = row["db_id"]
        return ids, ok, db

    _, c3_pt, db_map = vector("C3", "pt")
    _, c2_pt, _ = vector("C2", "pt")
    _, c4_pt, _ = vector("C4", "pt")
    report = {"selection": _selection(rows), "comparisons": {}}
    for name, treated, label in (
        ("C3-C2|pt", c3_pt, "Comparação primária: C3 × C2 em português"),
        ("C4-C2|pt", c4_pt, "H1 original: C4 × C2 em português (comparação secundária)"),
    ):
        if name.startswith("C4") and not treated:
            print("H1 não calculada: C4 em português ausente")
            continue
        shared = _paired_ids(treated, c2_pt)
        a = [treated[i] for i in shared]
        b = [c2_pt[i] for i in shared]
        strata = [db_map[i] for i in shared]
        print(f"=== {label} ===")
        mc = stats.mcnemar_exact(a, b)
        ci = stats.paired_bootstrap_diff(a, b, strata, seed=cfg.data.seed)
        print(mc.render())
        print(ci.render())
        report["comparisons"][name] = {"mcnemar": asdict(mc), "bootstrap": asdict(ci)}
    print(f"ilustração normal, não é o poder do estudo: pares discordantes "
          f"necessários para 70/30 a 80% de poder: "
          f"{stats.minimum_discordant_pairs(0.70)}")

    _, c0_pt, _ = vector("C0", "pt")
    _, c4_en, _ = vector("C4", "en")
    _, c0_en, _ = vector("C0", "en")
    if c4_en or c0_en:
        quad = _paired_ids(c4_pt, c0_pt, c4_en, c0_en)
        print("\n=== Diferença entre idiomas (diferenças em diferenças) ===")
        did = stats.difference_in_differences(
            [c4_pt[i] for i in quad],
            [c0_pt[i] for i in quad],
            [c4_en[i] for i in quad],
            [c0_en[i] for i in quad],
            [db_map[i] for i in quad],
            seed=cfg.data.seed,
        )
        print(did.render())
        report["h2_original"] = asdict(did)
        if not did.conclusive:
            print("intervalo contém zero: reportar como inconclusivo, "
                  "não como ausência de efeito")
    else:
        print("H2 não calculada: resultados em inglês ausentes")
    write_json(cfg.work / "stats.json", report)
    return 0


def cmd_tables(args) -> int:
    cfg = _cfg(args)
    from .report.tables import write_all

    rows = _report_rows(cfg, args)
    if not rows:
        print("log vazio", file=sys.stderr)
        return 1
    out = cfg.work / "tabelas"
    groups = {}
    for row in rows:
        groups.setdefault((row["condition"], row["lang"]), {})[row["example_id"]] = True
    _paired_ids(*groups.values())
    written = write_all(rows, out)
    write_json(out / "selection.json", _selection(rows))
    for path in written:
        print(f"escrito: {path}")
    return 0


def _report_rows(cfg: ExperimentConfig, args) -> list[dict]:
    from .logging_io import report_rows

    return report_rows(
        cfg.work / "runs", config_hash=args.config_hash or cfg.config_hash,
        run_ids=args.run_id, split=args.split, partition_digest=args.partition_digest,
    )


def _selection(rows: list[dict]) -> dict:
    return {
        "run_ids": sorted({row["run_id"] for row in rows}),
        **{field: rows[0][field] for field in ("config_hash", "partition_digest", "split")},
        "n_rows": len(rows),
    }


def _paired_ids(*vectors: dict) -> list[str]:
    if not vectors or not vectors[0] or any(set(v) != set(vectors[0]) for v in vectors[1:]):
        raise ValueError("pareamento incompleto: as condições/idiomas precisam cobrir "
                         "exatamente os mesmos exemplos; nenhuma interseção foi descartada")
    return sorted(vectors[0])


# ------------------------------------------------------------------ parser


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tcc-rag", description=__doc__)
    parser.add_argument("--config", dest="config_top", default=None,
                        help=f"arquivo de configuração (padrão: {DEFAULT_CONFIG})")
    sub = parser.add_subparsers(dest="command", required=True)

    for name, fn, helptext in (
        ("prepare", cmd_prepare, "carrega Spider + traduções, conta famílias por banco"),
        ("partition", cmd_partition, "aloca famílias a pool/val/test e audita vazamento"),
        ("leakage", cmd_leakage, "reexecuta a auditoria de vazamento sobre o manifesto"),
        ("index", cmd_index, "vetoriza o pool e popula o Qdrant"),
        ("run", cmd_run, "executa as condições experimentais"),
        ("sweep", cmd_sweep, "varre a grade de k, lambda e N na validação"),
        ("evaluate", cmd_evaluate, "agrega o log em métricas"),
        ("stats", cmd_stats, "McNemar, bootstrap e diferenças em diferenças"),
        ("tables", cmd_tables, "gera as tabelas LaTeX do Capítulo 5"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.set_defaults(func=fn)
        p.add_argument("--config", dest="config_sub", default=None)
        if name == "run":
            p.add_argument("--split", default="test", choices=["val", "test"])
            p.add_argument("--conditions", nargs="*", choices=list(ALL_CONDITIONS))
            p.add_argument("--langs", nargs="*", choices=list(ALL_LANGUAGES))
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--run-id", dest="run_id", default=None)
        if name == "sweep":
            p.add_argument("--split", default="val", choices=["val"])
            p.add_argument("--langs", nargs="*", choices=list(ALL_LANGUAGES))
            p.add_argument("--limit", type=int, default=150)
            p.add_argument("--conditions", nargs="*", default=["C4"],
                           choices=list(ALL_CONDITIONS))
            p.add_argument("--k", nargs="*", type=int, default=[1, 4, 8])
            p.add_argument("--lambdas", nargs="*", type=float,
                           default=[0.5, 0.7, 0.9])
            p.add_argument("--n-candidates", dest="n_candidates", nargs="*",
                           type=int, default=[25])
        if name in {"evaluate", "stats", "tables"}:
            p.add_argument("--run-id", dest="run_id", nargs="+", default=None,
                           help="logs a combinar explicitamente (sem observações duplicadas)")
            p.add_argument("--config-hash", default=None,
                           help="hash esperado; padrão: hash do arquivo --config")
            p.add_argument("--split", choices=["val", "test"], default=None)
            p.add_argument("--partition-digest", default=None)
    return parser


def app(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    caminho = _config_path(args)
    if not Path(caminho).exists():
        print(f"configuração não encontrada: {caminho}", file=sys.stderr)
        return 2
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"erro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(app())
