"""Run the complete, fixed-protocol SECOM side-project investigation."""

import argparse
from pathlib import Path

from threadpoolctl import threadpool_limits

from secom.experiment import StudyConfig, run_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--skip-temporal", action="store_true")
    parser.add_argument(
        "--overwrite", action="store_true", help="Explicitly replace an existing run."
    )
    args = parser.parse_args()
    with threadpool_limits(limits=args.jobs):
        run_study(
            StudyConfig(
                seed=args.seed,
                folds=args.folds,
                trees=args.trees,
                jobs=args.jobs,
                bootstrap=args.bootstrap,
                temporal_check=not args.skip_temporal,
            ),
            args.data_dir,
            args.output,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
