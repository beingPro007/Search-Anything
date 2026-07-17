import argparse

from app.api import download_raw
from app.constants import esci as C
from app.databuilder import GROUPS, build, status


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ESCI dataset build for encoder training"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="fetch the raw ESCI files into data/raw/")

    b = sub.add_parser("build", help="run the staged build pipeline (resumable)")
    b.add_argument("--locales", nargs="+", choices=C.LOCALES, default=None)
    b.add_argument("--small", action="store_true", help="use the reduced task-1 subset")
    b.add_argument("--stages", nargs="+", choices=GROUPS, default=None)
    b.add_argument("--force", action="store_true", help="re-run cached stages")
    b.add_argument("--max-negatives", type=int, default=C.MAX_NEGATIVES_PER_ROW)
    b.add_argument("--pp-cap", type=int, default=C.MAX_PP_PAIRS_PER_QUERY)
    b.add_argument("--seed", type=int, default=C.SEED)

    s = sub.add_parser("status", help="show which stage outputs exist")
    s.add_argument("--locales", nargs="+", choices=C.LOCALES, default=None)
    s.add_argument("--small", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "download":
        download_raw()
    elif args.command == "status":
        status(locales=args.locales, small=args.small)
    else:
        build(
            locales=args.locales,
            small=args.small,
            max_negatives=args.max_negatives,
            pp_cap=args.pp_cap,
            seed=args.seed,
            force=args.force,
            groups=set(args.stages) if args.stages else None,
        )


if __name__ == "__main__":
    main()
