"""Stage-based ESCI build: download -> clean -> qp -> pp, per locale, resumable."""

from functools import partial
from pathlib import Path

from app.api import download_raw, load_examples, load_products
from app.api.esci import raw_path
from app.common import get_logger, load_parquet, save_parquet
from app.common.pipeline import Stage, run_pipeline, stage_status
from app.constants import esci as C
from app.dataclean import clean_examples, clean_products
from app.helpers import build_product_product, build_query_product
from app.helpers.graph import build_graph

log = get_logger(__name__)

GROUPS = ("download", "clean", "qp", "pp", "graph")


def products_path(locale: str) -> Path:
    return C.PROCESSED_DIR / C.OUT_PRODUCTS.format(locale=locale)


def examples_path(locale: str) -> Path:
    return C.INTERIM_DIR / C.OUT_EXAMPLES_CLEAN.format(locale=locale)


def qp_path(split: str, locale: str) -> Path:
    return C.PROCESSED_DIR / C.OUT_QUERY_PRODUCT.format(split=split, locale=locale)


def pp_path(locale: str) -> Path:
    return C.PROCESSED_DIR / C.OUT_PRODUCT_PRODUCT.format(
        split=C.SPLIT_TRAIN, locale=locale
    )


def queries_path(locale: str) -> Path:
    return C.PROCESSED_DIR / C.OUT_QUERIES.format(locale=locale)


def p2q_path(locale: str) -> Path:
    return C.PROCESSED_DIR / C.OUT_GRAPH_P2Q.format(locale=locale)


def q2q_path(locale: str) -> Path:
    return C.PROCESSED_DIR / C.OUT_GRAPH_Q2Q.format(locale=locale)


def _run_clean(locale: str, small: bool) -> None:
    products = clean_products(load_products(locale))
    save_parquet(products, products_path(locale))
    examples = clean_examples(load_examples(locale, small=small), products)
    save_parquet(examples, examples_path(locale))


def _run_qp(locale: str, max_negatives: int, seed: int) -> None:
    products = load_parquet(products_path(locale))
    examples = load_parquet(examples_path(locale))
    for split in (C.SPLIT_TRAIN, C.SPLIT_TEST):
        subset = examples[examples[C.COL_SPLIT] == split]
        qp = build_query_product(
            subset, products, max_negatives=max_negatives, seed=seed
        )
        save_parquet(qp, qp_path(split, locale))


def _run_pp(locale: str, cap: int, seed: int) -> None:
    products = load_parquet(products_path(locale))
    examples = load_parquet(examples_path(locale))
    train = examples[examples[C.COL_SPLIT] == C.SPLIT_TRAIN]
    pp = build_product_product(train, products, cap_per_query=cap, seed=seed)
    save_parquet(pp, pp_path(locale))


def _run_graph(locale: str, seed: int) -> None:
    examples = load_parquet(examples_path(locale))
    queries, p2q, q2q = build_graph(examples, seed=seed)
    save_parquet(queries, queries_path(locale))
    save_parquet(p2q, p2q_path(locale))
    save_parquet(q2q, q2q_path(locale))


def make_stages(
    locales: tuple[str, ...] = C.LOCALES,
    small: bool = False,
    max_negatives: int = C.MAX_NEGATIVES_PER_ROW,
    pp_cap: int = C.MAX_PP_PAIRS_PER_QUERY,
    seed: int = C.SEED,
) -> list[Stage]:
    stages = [
        Stage(
            "download",
            "download",
            download_raw,
            tuple(raw_path(f) for f in C.RAW_FILES),
        )
    ]
    for loc in locales:
        stages.append(
            Stage(
                f"clean/{loc}",
                "clean",
                partial(_run_clean, loc, small),
                (products_path(loc), examples_path(loc)),
            )
        )
        stages.append(
            Stage(
                f"qp/{loc}",
                "qp",
                partial(_run_qp, loc, max_negatives, seed),
                tuple(qp_path(s, loc) for s in (C.SPLIT_TRAIN, C.SPLIT_TEST)),
            )
        )
        stages.append(
            Stage(
                f"pp/{loc}",
                "pp",
                partial(_run_pp, loc, pp_cap, seed),
                (pp_path(loc),),
            )
        )
        stages.append(
            Stage(
                f"graph/{loc}",
                "graph",
                partial(_run_graph, loc, seed),
                (queries_path(loc), p2q_path(loc), q2q_path(loc)),
            )
        )
    return stages


def build(
    locales: tuple[str, ...] | None = None,
    small: bool = False,
    max_negatives: int = C.MAX_NEGATIVES_PER_ROW,
    pp_cap: int = C.MAX_PP_PAIRS_PER_QUERY,
    seed: int = C.SEED,
    force: bool = False,
    groups: set[str] | None = None,
) -> None:
    stages = make_stages(
        tuple(locales) if locales else C.LOCALES, small, max_negatives, pp_cap, seed
    )
    run_pipeline(stages, force=force, groups=groups)


def status(locales: tuple[str, ...] | None = None, small: bool = False) -> None:
    for name, done, outputs in stage_status(
        make_stages(tuple(locales) if locales else C.LOCALES, small)
    ):
        mark = "done" if done else "pending"
        log.info("%-12s %s", name, mark)
        for out in outputs:
            size = f"{out.stat().st_size / 1e6:.1f}MB" if out.exists() else "-"
            log.info("  %-8s %s", size, out)
