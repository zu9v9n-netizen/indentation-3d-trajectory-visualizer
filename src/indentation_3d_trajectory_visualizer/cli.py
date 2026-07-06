from __future__ import annotations

import argparse
from pathlib import Path

from .config import VisualizationConfig
from .processing import analyze_csv, parse_column


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="indentation-3d",
        description="Create contact-corrected 3D indentation trajectory plots from CSV files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze = subparsers.add_parser("analyze", help="Analyze one CSV file")
    analyze.add_argument("input_csv", type=Path)
    analyze.add_argument("--output-root", type=Path, default=Path("outputs"))
    analyze.add_argument("--subject", default="unknown")
    analyze.add_argument("--condition", default="unknown")
    analyze.add_argument("--time-column")
    analyze.add_argument("--load-column")
    analyze.add_argument("--displacement-column")
    analyze.add_argument("--sample-interval-sec", type=float, default=0.0005)
    analyze.add_argument("--load-threshold-n", type=float, default=0.05)
    analyze.add_argument("--smoothing-window-samples", type=int, default=21)
    analyze.add_argument("--pre-margin-sec", type=float, default=2.0)
    analyze.add_argument("--post-margin-sec", type=float, default=2.0)
    analyze.add_argument("--min-active-duration-sec", type=float, default=0.2)
    analyze.add_argument("--max-gap-within-trial-sec", type=float, default=0.1)
    analyze.add_argument("--no-zero-correction", action="store_true")
    analyze.add_argument("--no-plots", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "analyze":
        config = VisualizationConfig(
            sample_interval_sec=args.sample_interval_sec,
            time_column=parse_column(args.time_column),
            load_column=parse_column(args.load_column),
            displacement_column=parse_column(args.displacement_column),
            load_threshold_n=args.load_threshold_n,
            smoothing_window_samples=args.smoothing_window_samples,
            pre_margin_sec=args.pre_margin_sec,
            post_margin_sec=args.post_margin_sec,
            min_active_duration_sec=args.min_active_duration_sec,
            max_gap_within_trial_sec=args.max_gap_within_trial_sec,
            zero_correction=not args.no_zero_correction,
            save_plots=not args.no_plots,
        )
        output_dir = analyze_csv(
            args.input_csv,
            args.output_root,
            config,
            subject=args.subject,
            condition=args.condition,
        )
        print(f"Saved analysis outputs to: {output_dir}")
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

