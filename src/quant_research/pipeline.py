"""CLI for non-destructive quality gates and factor generation."""

from __future__ import annotations

import argparse

from quant_research.factors import run_factors
from quant_research.factor_research import run as run_factor_research
from quant_research.panel import run_labels
from quant_research.quality import run_quality


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data-quality and factor pipeline stages")
    parser.add_argument("stage", choices=["quality", "factors", "labels", "factor-research"])
    parser.add_argument("--limit", type=int, help="process only the first N securities")
    args = parser.parse_args()
    if args.stage == "quality":
        run_quality(limit=args.limit)
    elif args.stage == "factors":
        run_factors(limit=args.limit)
    elif args.stage == "factor-research":
        run_factor_research()
    else:
        run_labels(limit=args.limit)


if __name__ == "__main__":
    main()
