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
import json
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


LIMITE_PT_IGUAL_EN = 0.05


def exigir_traducao(examples) -> None:
    """Recusa gerar sobre um manifesto cujas perguntas em português copiam as em inglês.

    Todas as rodadas de agosto e setembro de 2026 usaram, sem que se notasse, um
    `questions_pt.json` provisório que reproduzia o texto em inglês (E01). Com a
    tradução de José e Cozman (2021), nenhuma pergunta do recorte coincide nos
    dois idiomas; acima de 5% de coincidência, o arquivo não é a tradução.
    """
    if not examples:
        return
    iguais = sum(1 for e in examples if e.question_pt.strip() == e.question_en.strip())
    if iguais / len(examples) > LIMITE_PT_IGUAL_EN:
        raise ValueError(
            f"{iguais} de {len(examples)} perguntas em português são idênticas às em inglês: o "
            "`questions_pt.json` deste diretório é o arquivo provisório, não a tradução. Gere-o com "
            "`scripts/gerar_questions_pt.py` e refaça `prepare` e `partition`."
        )


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
    exigir_traducao(examples)
    demos = build_demos(examples, partition=part.POOL)
    if not demos:
        print("pool vazio: rode `partition` antes", file=sys.stderr)
        return 1

    embedder = Embedder(cfg.embedding, cfg.work / "cache")
    vectors = embedder.encode([d.question_delex for d in demos], kind="passage")

    from .pipeline import identidade_indice

    index = DemoIndex(cfg.qdrant, dimension=vectors.shape[1])
    index.recreate()
    total = index.upsert(demos, vectors)
    # A referência dos vetores é gravada aqui; `run`, `sweep` e `custo` conferem
    # o índice persistido contra ela por tolerância (ver `DemoIndex`).
    identidade = identidade_indice(demos, embedder, index.gravar_referencia())
    index.gravar_identidade(identidade)
    print(f"indexadas {total} demonstrações ({len(demos)//2} exemplos × 2 idiomas)")
    print(f"identidade do índice: {identidade['digest']} ({index.identidade_path})")
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
    exigir_traducao(examples)
    split = args.split
    if split == "test":
        _exigir_congelamento(cfg)
    from .pipeline import conteudo_digest

    # O conteúdo no nome é o do split inteiro, nos dois idiomas, antes de
    # --limit e --langs: uma amostra e o split completo dividem o log, e a
    # chave (condição, idioma, exemplo) não colide. Outro texto muda o nome.
    conteudo_split = conteudo_digest(build_test_items(examples, partition=split, languages=ALL_LANGUAGES))
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
    indice_digest = _indice_conferido(index, pool, embedder)
    selector = DemoSelector(cfg, index, embedder, pool)
    backend = make_backend(cfg.generation)

    from .pipeline import partition_digest, runs_dir

    digest = partition_digest(examples)
    # O conteúdo entra no nome: outro texto de pergunta gera outro log, em vez
    # de retomar o antigo.
    run_id = args.run_id or f"{cfg.name}-{split}-{cfg.config_hash}-{conteudo_split}"
    log = RunLog(runs_dir(cfg) / f"{run_id}.jsonl")
    print(f"run_id={run_id}  já registradas={log.n_done}  itens={len(items)}")
    print(f"partition_digest={digest}  indice={indice_digest}")

    try:
        runner = Runner(cfg, backend, selector, schemas, log, run_id, digest, split=split,
                        indice_digest=indice_digest)
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
    from .pipeline import conteudo_digest, item_set_digest, subsample
    from .sweep import render, run_sweep, best

    examples = load_manifest(cfg)
    exigir_traducao(examples)
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
    indice_digest = _indice_conferido(index, pool, embedder)
    backend = make_backend(cfg.generation)

    total = len(args.k) * len(args.lambdas) * len(args.n_candidates)
    bancos = len({i.db_id for i in items})
    print(f"itens={len(items)}  bancos={bancos}  combinacoes={total}  "
          f"condicoes={' '.join(args.conditions)}")

    linhas = run_sweep(
        cfg, items, pool, schemas, index, embedder, backend,
        partition_digest(examples),
        ks=args.k, lambdas=args.lambdas, ns=args.n_candidates,
        # Amostra e conteúdo no nome: a amostra de 150 itens com o texto
        # provisório (`92419f77`, 22/09) e a mesma amostra em PT-BR real não
        # compartilham log.
        item_digest=f"{item_set_digest(items)}-{conteudo_digest(items)}",
        conditions=args.conditions, on_progress=print, indice_digest=indice_digest,
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


def cmd_custo(args) -> int:
    """Mede a latência fim a fim por pergunta, com lote unitário (E06).

    A rodada principal gera em lote e registra o tempo médio do lote; esta
    passagem mede uma pergunta por vez, inclui a consulta preliminar de C4 e
    usa um cache de vetores vazio. Não avalia acerto. É retomável.
    """
    import tempfile

    cfg = _cfg(args)
    from .custo import medir, resumir
    from .generation.llm import make_backend
    from .pipeline import item_set_digest, partition_digest, subsample
    from .retrieval.embedder import Embedder
    from .retrieval.index import DemoIndex
    from .retrieval.selectors import DemoSelector

    examples = load_manifest(cfg)
    exigir_traducao(examples)
    items = build_test_items(
        examples, partition=args.split,
        languages=tuple(args.langs) if args.langs else ("pt",),
    )
    if args.limit:
        items = subsample(items, args.limit, seed=cfg.data.seed)
    if not items:
        print(f"nenhum item na partição {args.split}", file=sys.stderr)
        return 1

    schemas = load_schemas(cfg.paths.spider_root)
    pool = build_demos(examples, partition=part.POOL)
    # Cache de vetores novo e vazio: a vetorização da pergunta entra no tempo.
    embedder = Embedder(cfg.embedding, Path(tempfile.mkdtemp(prefix="tcc-custo-vetores-")))
    index = DemoIndex(cfg.qdrant, dimension=embedder.dimension)
    indice_digest = _indice_conferido(index, pool, embedder)
    selector = DemoSelector(cfg, index, embedder, pool)
    backend = make_backend(cfg.generation)

    amostra = item_set_digest(items)
    destino = cfg.work / "custo" / f"custo-{args.split}-{cfg.config_hash}-{amostra}.jsonl"
    identidade = {
        "config_hash": cfg.config_hash,
        "partition_digest": partition_digest(examples),
        "split": args.split,
        "indice_digest": indice_digest,
        "amostra_digest": amostra,
    }
    conditions = tuple(args.conditions) if args.conditions else ALL_CONDITIONS
    print(f"itens={len(items)}  condições={' '.join(conditions)}  destino={destino}")
    escritas = medir(cfg, items, selector, backend, schemas, destino, conditions, identidade)
    index.close()

    linhas = [json.loads(l) for l in destino.read_text(encoding="utf-8").splitlines() if l.strip()]
    resumo = resumir(linhas)
    write_json(cfg.work / "custo" / f"resumo-{args.split}-{cfg.config_hash}-{amostra}.json", resumo)
    print(f"+{escritas} medições ({len(linhas)} no arquivo)")
    print(f"{'cond|lang':<10} {'n':>4} {'p50 ms':>9} {'p95 ms':>9} {'tok ent':>8} {'tok sai':>8} {'pico MiB':>9}")
    for chave, v in resumo.items():
        pico = "--" if v["gpu_mem_pico_mb_max"] is None else f"{v['gpu_mem_pico_mb_max']:.0f}"
        print(f"{chave:<10} {v['n']:>4} {v['total_ms_p50']:>9.0f} {v['total_ms_p95']:>9.0f} "
              f"{v['prompt_tokens_total_media']:>8.0f} {v['completion_tokens_total_media']:>8.0f} {pico:>9}")
    return 0


def _exigir_congelamento(cfg: ExperimentConfig) -> None:
    """O teste só é gerado sob o registro de congelamento, com a mesma configuração e revisão."""
    from .congelamento import ARQUIVO
    from .proveniencia import codigo_revisao

    caminho = cfg.work / ARQUIVO
    if not caminho.is_file():
        raise ValueError(f"o conjunto de teste está selado: falta o registro de congelamento ({caminho}). "
                         "Rode a validação completa e `tcc-rag congelar`.")
    gravado = json.loads(caminho.read_text(encoding="utf-8"))
    if gravado["configuracao"]["config_hash"] != cfg.config_hash:
        raise ValueError("a configuração mudou depois do congelamento: "
                         f"{gravado['configuracao']['config_hash']} no registro, {cfg.config_hash} agora")
    if gravado["codigo_revisao"] != codigo_revisao():
        raise ValueError("a revisão do código mudou depois do congelamento: "
                         f"{gravado['codigo_revisao']} no registro, {codigo_revisao()} agora")


def cmd_congelar(args) -> int:
    """Portão de sanidade da validação e registro de congelamento (E15).

    Confere se a validação completa rodou como previsto e, passando, grava
    `congelamento.json`, uma única vez. Sem esse registro, `run --split test`
    se recusa a começar. Os critérios são de funcionamento do experimento e não
    olham os contrastes entre as condições.
    """
    from .congelamento import ARQUIVO, congelar
    from .logging_io import report_rows
    from .pipeline import runs_dir
    from .proveniencia import codigo_revisao

    cfg = _cfg(args)
    examples = load_manifest(cfg, warn=False)
    exigir_traducao(examples)
    n_val = sum(1 for e in examples if e.partition == "val")
    n_teste = sum(1 for e in examples if e.partition == "test")
    rows = report_rows(runs_dir(cfg), config_hash=cfg.config_hash, split="val")
    if not rows:
        print("validação ausente: nada a congelar", file=sys.stderr)
        return 1
    pode, reg = congelar(cfg, _config_path(args), rows, n_val, n_teste, codigo_revisao())
    if not pode:
        conf = reg.get("conferencia") or reg.get("_conferencia_da_retomada") or {}
        print("CONGELAMENTO RECUSADO — o teste permanece selado:", file=sys.stderr)
        for falha in conf.get("falhas") or conf.get("divergencias") or ["motivo não identificado"]:
            print(f"  - {falha}", file=sys.stderr)
        return 1
    print(f"registro de congelamento: {cfg.work / ARQUIVO}")
    print(f"  congelado em {reg['congelado_em_utc']}  config_hash={reg['configuracao']['config_hash']}  "
          f"código={reg['codigo_revisao'][:12]}")
    print(f"  teste: {reg['teste']['perguntas']} perguntas, {reg['teste']['linhas_esperadas']} linhas esperadas")
    for chave, bloco in (reg["validacao"].get("poder") or {}).items():
        a5 = bloco.get("alfa_0.05", {})
        print(f"  poder {chave}: discordância {100 * bloco['taxa_discordancia_observada']:.1f}%, "
              f"menor diferença detectável (80%, alfa 5%) = {a5.get('dmd_80')}")
    return 0


def cmd_estratos(args) -> int:
    """Estrato de literal traduzido (LT/SL), pela regra pré-registrada.

    Lê o manifesto e grava `estratos_literais.json` à parte, com o SHA-256 do
    manifesto de origem. Não altera o manifesto nem a partição. Imprime só os
    tamanhos dos estratos por partição, que dependem só do texto.
    """
    import hashlib
    from collections import Counter

    from .data.estratos import literais_traduzidos

    cfg = _cfg(args)
    examples = load_manifest(cfg)
    itens = {}
    for ex in examples:
        lits = literais_traduzidos(ex.sql, ex.question_en, ex.question_pt)
        itens[ex.example_id] = {"estrato": "LT" if lits else "SL", "literais": lits,
                                "partition": ex.partition}
    contagem = Counter(f"{v['partition']}|{v['estrato']}" for v in itens.values())
    payload = {
        "regra": "orquestracao/preregistro-literais-traduzidos-2026-10-28.md; tcc_rag.data.estratos",
        "manifesto_sha256": hashlib.sha256(manifest_path(cfg).read_bytes()).hexdigest(),
        "contagem": dict(sorted(contagem.items())),
        "itens": itens,
    }
    destino = cfg.work / "estratos_literais.json"
    write_json(destino, payload)
    for parte in ("pool", "val", "test"):
        print(f"{parte:<5} LT={contagem[parte + '|LT']:<4} SL={contagem[parte + '|SL']}")
    print(f"escrito: {destino} (sha256 {hashlib.sha256(destino.read_bytes()).hexdigest()})")
    return 0


def cmd_reavaliar(args) -> int:
    """Recalcula EX e EM de um log já gerado, com os avaliadores atuais, num log novo.

    É o que fazer quando a retomada recusa um log só porque o avaliador mudou
    (outra versão, outro punkt_tab): a geração não precisa ser refeita. Roda em
    CPU. O log original não é alterado.
    """
    import hashlib

    from .runner import reavaliar

    cfg = _cfg(args)
    origem = _runs_dir(cfg, args) / f"{args.run_id}.jsonl"
    if not origem.is_file():
        print(f"log não encontrado: {origem}", file=sys.stderr)
        return 1
    linhas = RunLog(origem).rows()
    if not linhas:
        print("log vazio", file=sys.stderr)
        return 1
    schemas = load_schemas(cfg.paths.spider_root)
    from .evaluation.em_oficial import AvaliadorEM
    from .evaluation.ex_oficial import AvaliadorEX
    from .evaluation.sandbox import Sandbox

    marca = hashlib.sha256((AvaliadorEX(Sandbox(cfg.sandbox)).identificador + AvaliadorEM().identificador)
                           .encode("utf-8")).hexdigest()[:8]
    novo_id = args.novo_run_id or f"{args.run_id}-reav-{marca}"
    destino = _runs_dir(cfg, args) / f"{novo_id}.jsonl"
    if destino.exists():
        print(f"já existe: {destino}; escolha outro --novo-run-id", file=sys.stderr)
        return 1
    log = RunLog(destino)
    for linha in reavaliar(linhas, schemas, cfg.sandbox, novo_id):
        log.append(linha)
    print(f"reavaliadas {len(linhas)} linhas: {destino}")
    print(f"para retomar a geração: --run-id {novo_id}")
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
    write_json(cfg.work / "summary_selection.json", _selection(rows, _runs_dir(cfg, args)))

    # EM: oficial do Spider; "--" quando o log não o tem (anterior à E03). A
    # correspondência canônica própria aparece separada, como "EM próp".
    print(f"{'cond|lang':<12} {'n':>5} {'EX':>7} {'EM':>7} {'EM próp':>8} {'n/exec':>7} "
          f"{'hit@k':>7} {'fallb':>7} {'tok':>8}")
    for key, value in summary.items():
        em = "--" if value["em_oficial"] is None else f"{value['em_oficial']*100:.1f}%"
        print(f"{key:<12} {value['n']:>5} {value['ex']*100:>6.1f}% "
              f"{em:>7} {value['em_canonico_proprio']*100:>7.1f}% {value['not_executable']:>7} "
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
    report = {"selection": _selection(rows, _runs_dir(cfg, args)), "comparisons": {}}
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


def cmd_analise(args) -> int:
    """Análise completa de um split: contrastes, H2, H3, estratos, custo, erros e poder.

    Lê os mesmos logs que `evaluate` e `stats` (mesmas regras de seleção),
    junta os estratos de literal traduzido e as medições de `custo`, se
    existirem, e grava `analise/analise-<split>-<config_hash>.json`. Com
    `--sensibilidade-distinct`, reavalia cada predição com a variante do
    avaliador oficial que preserva DISTINCT (CPU; não altera os logs).
    """
    from . import analise

    cfg = _cfg(args)
    rows = _report_rows(cfg, args)
    if not rows:
        print("log vazio", file=sys.stderr)
        return 1
    split = rows[0]["split"]

    estratos = None
    caminho_estratos = cfg.work / "estratos_literais.json"
    if caminho_estratos.is_file():
        estratos = json.loads(caminho_estratos.read_text(encoding="utf-8"))

    custo_rows: list[dict] = []
    # Só as medições de custo do mesmo split e da mesma configuração.
    for arquivo in sorted((cfg.work / "custo").glob(f"custo-{split}-{rows[0]['config_hash']}-*.jsonl")):
        custo_rows += [json.loads(l) for l in arquivo.read_text(encoding="utf-8").splitlines() if l.strip()]

    ex_distinct = None
    if args.sensibilidade_distinct:
        from .evaluation.ex_oficial import AvaliadorEX
        from .evaluation.sandbox import Sandbox

        schemas = load_schemas(cfg.paths.spider_root)
        avaliador = AvaliadorEX(Sandbox(cfg.sandbox), keep_distinct=True)
        ex_distinct = {}
        for row in rows:
            veredito = avaliador.avaliar(schemas[row["db_id"]].sqlite_path, row["generated_sql"], row["gold_sql"])
            ex_distinct[(row["condition"], row["lang"], row["example_id"])] = veredito.correto

    from .pipeline import load_manifest as _manifesto

    exemplos = _manifesto(cfg, warn=False)
    n_teste = sum(1 for e in exemplos if e.partition == "test")
    resultado = analise.analisar(rows, estratos=estratos, custo_rows=custo_rows or None,
                                 semente=cfg.data.seed, n_teste=n_teste, ex_distinct=ex_distinct,
                                 familias={e.example_id: e.family_id for e in exemplos})
    resultado["selecao"] = _selection(rows, _runs_dir(cfg, args))
    destino = cfg.work / "analise" / f"analise-{split}-{rows[0]['config_hash']}.json"
    destino.parent.mkdir(parents=True, exist_ok=True)
    write_json(destino, resultado)

    print(f"split={split}  exemplos={resultado['n_exemplos']}  "
          f"referências inválidas={len(resultado['referencias_invalidas'])}")
    print(f"{'cond|lang':<10} {'EX':>6} {'EM':>6} {'n/exec':>6} {'hit':>6} {'tok ent':>8}")
    for chave, c in resultado["celulas"].items():
        em = "--" if c["em_oficial"] is None else f"{100 * c['em_oficial']:.1f}"
        print(f"{chave:<10} {100 * c['ex']:>6.1f} {em:>6} {c['nao_executaveis']:>6} "
              f"{100 * c['acerto_estrutural']:>6.1f} {c['prompt_tokens_total_medio']:>8.0f}")
    for chave, c in resultado["contrastes"].items():
        print(f"{chave:<12} {c['diferenca_pp']:+6.2f} pp  IC95 [{c['ic95'][0]:+.2f}, {c['ic95'][1]:+.2f}]  "
              f"p={c['mcnemar']['p_value']:.4f}  discordantes={c['discordantes']}  ({c['papel']})")
    for nome, d in (resultado.get("diferencas_em_diferencas") or {}).items():
        print(f"{nome}: Δ={d['delta_pp']:+.2f} pp  IC95 [{d['low']:+.2f}, {d['high']:+.2f}]")
    print(f"escrito: {destino}")
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
    write_json(out / "selection.json", _selection(rows, _runs_dir(cfg, args)))
    for path in written:
        print(f"escrito: {path}")
    return 0


def _indice_conferido(index, pool, embedder) -> str:
    """Recusa um índice construído com outro pool ou vetorizador; devolve o digest."""
    from .pipeline import identidade_indice

    esperada = identidade_indice(pool, embedder, index.vetores_sha256())
    gravada = index.ler_identidade()
    if gravada is None or gravada.get("digest") != esperada["digest"]:
        motivo = ("sem identidade registrada (construído antes da E04 ou interrompido)"
                  if gravada is None else "construído com outro pool ou outro vetorizador")
        raise ValueError(f"índice de demonstrações {motivo}: rode `tcc-rag index` antes "
                         f"({index.identidade_path})")
    if gravada.get("proveniencia") != esperada["proveniencia"]:
        print("[indice] AVISO: o vetorizador atual difere do que construiu o índice só em "
              "dispositivo, lote ou versões; os vetores gravados são os mesmos.", file=sys.stderr)
    return esperada["digest"]


def _runs_dir(cfg: ExperimentConfig, args) -> Path:
    from .pipeline import runs_dir

    return Path(args.runs_dir) if getattr(args, "runs_dir", None) else runs_dir(cfg)


def _report_rows(cfg: ExperimentConfig, args) -> list[dict]:
    from .logging_io import report_rows

    return report_rows(
        _runs_dir(cfg, args), config_hash=args.config_hash or cfg.config_hash,
        run_ids=args.run_id, split=args.split, partition_digest=args.partition_digest,
    )


def _selection(rows: list[dict], runs_dir: Path | None = None) -> dict:
    from .logging_io import avaliador_em, avaliador_ex

    return {
        "run_ids": sorted({row["run_id"] for row in rows}),
        **{field: rows[0][field] for field in ("config_hash", "partition_digest", "split")},
        "runs_dir": str(runs_dir) if runs_dir else None,
        "ex_avaliador": avaliador_ex(rows[0]),
        "em_avaliador": avaliador_em(rows[0]),
        "codigo_revisoes": sorted({row.get("codigo_revisao") or "nao_registrada" for row in rows}),
        **{field: rows[0].get(field) for field in ("formato_log", "indice_digest", "template_sha256")},
        "ambientes": sorted({json.dumps((row.get("ambiente") or {}).get("bibliotecas"), sort_keys=True)
                             for row in rows}),
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
        ("custo", cmd_custo, "mede a latência fim a fim por pergunta, com lote unitário"),
        ("congelar", cmd_congelar, "portão de sanidade da validação e registro de congelamento"),
        ("estratos", cmd_estratos, "estrato de literal traduzido (LT/SL), pré-registrado"),
        ("reavaliar", cmd_reavaliar, "recalcula EX e EM de um log com os avaliadores atuais"),
        ("evaluate", cmd_evaluate, "agrega o log em métricas"),
        ("stats", cmd_stats, "McNemar, bootstrap e diferenças em diferenças"),
        ("tables", cmd_tables, "gera as tabelas LaTeX do Capítulo 5"),
        ("analise", cmd_analise, "análise completa de um split (contrastes, H2, H3, custo, erros, poder)"),
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
        if name == "custo":
            p.add_argument("--split", default="val", choices=["val", "test"])
            p.add_argument("--conditions", nargs="*", choices=list(ALL_CONDITIONS))
            p.add_argument("--langs", nargs="*", choices=list(ALL_LANGUAGES))
            p.add_argument("--limit", type=int, default=100)
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
        if name == "reavaliar":
            p.add_argument("--run-id", dest="run_id", required=True)
            p.add_argument("--novo-run-id", dest="novo_run_id", default=None)
            p.add_argument("--runs-dir", default=None)
        if name == "analise":
            p.add_argument("--sensibilidade-distinct", dest="sensibilidade_distinct", action="store_true",
                           help="reavalia com a variante do avaliador que preserva DISTINCT")
        if name in {"evaluate", "stats", "tables", "analise"}:
            p.add_argument("--run-id", dest="run_id", nargs="+", default=None,
                           help="logs a combinar explicitamente (sem observações duplicadas)")
            p.add_argument("--config-hash", default=None,
                           help="hash esperado; padrão: hash do arquivo --config")
            p.add_argument("--split", choices=["val", "test"], default=None)
            p.add_argument("--partition-digest", default=None)
            p.add_argument("--runs-dir", default=None,
                           help="diretório dos logs (padrão: <work_dir>/runs-v2); use para "
                                "ler, isolados, os logs históricos de runs/")
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
