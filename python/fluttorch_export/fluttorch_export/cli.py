"""Command line entry point.

    fluttorch-export \
        --model mypkg.models:build_forecaster \
        --example-inputs mypkg.models:example_window \
        --goldens mypkg.models:golden_cases \
        --name solar_forecast \
        --out build/solar_forecast

Writes into ``--out``:

    solar_forecast.pte              runtime artifact
    solar_forecast.fluttorch.json   manifest consumed by fluttorch_gen
    goldens/                        reference inputs and outputs

The three are written together on purpose: a manifest paired with weights it was
not generated from is exactly what the weight hash exists to catch, and it can
only catch it if nothing produces one without the other.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

from .export import ExportError, export_model, resolve
from .manifest import ManifestError
from .quantization import RECIPES, QuantizationError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fluttorch-export",
        description="Export a PyTorch model with the contract Fluttorch reads.",
    )
    parser.add_argument("--model", required=True, help="module:factory returning an nn.Module")
    parser.add_argument("--example-inputs", required=True, help="module:factory of example inputs")
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--name", help="model name; defaults to the output directory's")
    parser.add_argument("--backend", default="xnnpack", help="lowering backend")
    parser.add_argument(
        "--goldens",
        help="module:factory yielding input tuples to capture references for. "
        "Omitted, only the example input is captured, which is a smoke test "
        "rather than coverage.",
    )
    parser.add_argument("--labels", help="module:attribute holding class labels")
    parser.add_argument(
        "--input-names",
        help="comma-separated names for the inputs. Without them the names are "
        "positional, and a generated accessor called input_0 is one nobody wants "
        "to read.",
    )
    parser.add_argument("--output-names", help="comma-separated names for the outputs")
    parser.add_argument(
        "--dynamic-batch",
        action="store_true",
        help="mark the leading dimension of every tensor dynamic",
    )
    parser.add_argument(
        "--quantize",
        help="quantization recipe: " + ", ".join(sorted(RECIPES)) + ". The name is "
        "written into the manifest and selects the tolerance the parity gate "
        "starts from, so it describes the artifact rather than labelling it.",
    )
    parser.add_argument(
        "--taps",
        help="comma-separated submodules whose outputs are captured alongside "
        "each golden, so a drift can be attributed to the layer that caused it. "
        "Off by default: an intermediate is as large as the tensor it carries.",
    )
    return parser


def _names(raw: str | None) -> list[str] | None:
    return [n.strip() for n in raw.split(",")] if raw else None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out = pathlib.Path(args.out)
    name = args.name or out.name

    try:
        result = export_model(
            model=resolve(args.model),
            example_inputs=resolve(args.example_inputs),
            out_dir=out,
            name=name,
            backend=args.backend,
            golden_inputs=resolve(args.goldens) if args.goldens else None,
            labels=resolve(args.labels) if args.labels else None,
            input_names=_names(args.input_names),
            output_names=_names(args.output_names),
            dynamic_batch=args.dynamic_batch,
            quantization=args.quantize,
            taps=_names(args.taps),
        )
    except (ExportError, ManifestError, QuantizationError) as e:
        print(f"fluttorch-export: {e}", file=sys.stderr)
        return 1

    m = result.manifest
    print(f"artifact     {result.artifact}  {result.artifact.stat().st_size} bytes")
    print(f"manifest     {result.manifest_path}")
    print(f"weight hash  {m.weight_hash}")
    print(f"inputs       {', '.join(f'{s.name}:{s.dtype}{list(s.shape)}' for s in m.inputs)}")
    print(f"outputs      {', '.join(f'{s.name}:{s.dtype}{list(s.shape)}' for s in m.outputs)}")
    print(f"quantization {m.quantization or 'none, full precision'}")
    print(f"goldens      {result.golden_count} case(s) in {result.golden_dir}")
    if m.activations:
        print(f"taps         {', '.join(s.name for s in m.activations)}")
    else:
        print(
            "taps         none. A failing gate will name the output that moved, "
            "not the layer that moved it."
        )
    if args.goldens is None:
        print(
            "\nOnly the example input was captured. One case proves the pipeline "
            "runs; it does not tell you the model survived export. Pass --goldens "
            "with inputs that represent the job.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
