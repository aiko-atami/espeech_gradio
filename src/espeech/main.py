from __future__ import annotations

import argparse
import sys

import gradio as gr

from espeech.services.batch_config import load_batch_job_config
from espeech.services.batch_runner import run_batch_job
from espeech.ui.app import create_app
from espeech.ui.styles import APP_CSS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="espeech")
    subparsers = parser.add_subparsers(dest="command")

    batch_parser = subparsers.add_parser(
        "batch",
        help="Run batch synthesis from a YAML config file.",
    )
    batch_parser.add_argument(
        "--config",
        required=True,
        help="Path to the YAML batch config.",
    )
    return parser


def _run_batch_command(config_path: str) -> int:
    try:
        config = load_batch_job_config(config_path)
        result = run_batch_job(config)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Output directory: {result.output_dir}")
    print(f"Summary: {result.summary_path}")
    print(f"Results: {result.results_path}")
    if result.archive_path:
        print(f"Archive: {result.archive_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "batch":
        return _run_batch_command(args.config)

    demo = create_app()
    demo.launch(theme=gr.themes.Default(), css=APP_CSS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
