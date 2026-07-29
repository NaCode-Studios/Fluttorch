"""Command line entry point.

    fluttorch-export \
        --model mypkg.models:build_forecaster \
        --example-inputs mypkg.models:example_window \
        --backend xnnpack \
        --quantize int8-static \
        --goldens 32 \
        --out build/solar_forecast

Writes into ``--out``:

    solar_forecast.pte              runtime artifact
    solar_forecast.fluttorch.json   manifest consumed by fluttorch_gen
    goldens/                        reference inputs and outputs

The goldens are captured from the *source* model, before lowering, which is what
makes them a reference rather than a snapshot of whatever the export produced.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fluttorch-export")
    parser.add_argument("--model", required=True, help="module:factory returning an nn.Module")
    parser.add_argument("--example-inputs", required=True, help="module:factory of example inputs")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--backend", default="xnnpack", help="lowering backend")
    parser.add_argument("--quantize", default=None, help="recipe, e.g. int8-static")
    parser.add_argument("--goldens", type=int, default=16, help="reference cases to capture")
    return parser


def main(argv: list[str] | None = None) -> int:
    build_parser().parse_args(argv)
    raise NotImplementedError("see the roadmap: export toolchain")
