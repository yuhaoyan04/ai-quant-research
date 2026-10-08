"""Run the three per-security research-data stages in a single process."""

from __future__ import annotations

import argparse

from quant_research.factors import run_factors
from quant_research.quality import run_quality
from quant_research.panel import run_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize per-security research tables")
    parser.add_argument("--missing", action="store_true", help="only create tables absent from the lake")
    args = parser.parse_args()
    print("stage=quality", flush=True)
    run_quality(skip_existing=args.missing)
    print("stage=factors", flush=True)
    run_factors(skip_existing=args.missing)
    print("stage=labels", flush=True)
    run_labels(skip_existing=args.missing)
    print("status=completed", flush=True)


if __name__ == "__main__":
    main()
