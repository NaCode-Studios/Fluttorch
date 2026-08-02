"""Produces the fixture that proves two tensors are kept apart.

Every other export in this repository returns one tensor. The runtime, the gate
and the generator all handle lists throughout, and none of those lists has ever
held a second element, so the code that keeps two outputs in the right order has
been read and never executed. That is the shape of gap this project has been
bitten by before: a path written, believed, and never run.

This writes a model whose two inputs and two outputs are deliberately hard to
tell apart:

  testdata/multi_io/multi_io.fluttorch.json   ExecuTorch, portable
  testdata/multi_io_litert/multi_io.tflite    the same model through LiteRT
  testdata/multi_io_onnx/multi_io.onnx        and through ONNX Runtime

Three bundles because the ordering code is per-shim. Each binding converts its
own runtime's output list into ours, so one engine keeping the pair straight
says nothing about the other two.

## Why the shapes match

`left` and `right` are both (1, 5), and `primary` and `auxiliary` are both
(1, 3). That is the discriminating choice, and it is the whole reason this
fixture is worth committing.

A pair that differed in shape would be caught by the shape check on the way
past, which means the test would pass for a reason that has nothing to do with
ordering: any transposition would fail loudly whether or not the ordering code
was right. Matching shapes remove that safety net. `checkTensorsAgainst` still
catches a runtime that returns the pair reversed, because it compares the name
before the shape, but the generated typed API indexes `_tensors[i]` positionally
and never goes near that check, and the exporter writes the goldens in an order
nothing downstream can second-guess. Those two are what this fixture measures,
and against them a shape mismatch would have proved nothing.

So the model is asymmetric in both directions instead. Swapping the inputs
changes both outputs, because each input goes through its own projection, and
the two heads carry different weights, so swapping the outputs changes the
numbers. Nothing here is interchangeable, and the only thing that can tell the
tensors apart is being right about which is which.

    python python/fluttorch_export/scripts/export_multi_io.py

Needs `flatc` on PATH, which the ExecuTorch checkout builds:

    export PATH="$HOME/.cache/fluttorch/executorch/cmake-out-android/third-party/flatc_ep/bin:$PATH"
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fluttorch_export.export import export_model  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]

#: Both inputs this wide and both outputs this wide, so neither pair can be told
#: apart by its shape. See the module docstring: that is the point, not an
#: accident of picking round numbers.
FEATURES, WIDTH, OUTPUTS = 5, 8, 3

#: Golden windows. Four, like VoltaCast, and drawn rather than hand-written
#: because a fixture whose inputs are all zeros or all ones cannot distinguish a
#: head that ignores its argument from one that uses it.
GOLDEN_CASES = 4


class TwoHead(torch.nn.Module):
    """Two inputs and two outputs, none of the four interchangeable.

    Small enough to export in a second, and asymmetric everywhere it counts:
    separate projections for the two inputs and separate heads for the two
    outputs, so a transposition on either side moves the numbers.
    """

    def __init__(self) -> None:
        super().__init__()
        self.left_proj = torch.nn.Linear(FEATURES, WIDTH)
        self.right_proj = torch.nn.Linear(FEATURES, WIDTH)
        self.act = torch.nn.ReLU()
        self.primary = torch.nn.Linear(WIDTH, OUTPUTS)
        self.auxiliary = torch.nn.Linear(WIDTH, OUTPUTS)

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """left (B, F), right (B, F) -> primary (B, O), auxiliary (B, O)."""
        hidden = self.act(self.left_proj(left) + self.right_proj(right))
        return self.primary(hidden), self.auxiliary(hidden)


def main() -> int:
    torch.manual_seed(20260802)
    model = TwoHead().eval()

    def case() -> tuple[torch.Tensor, torch.Tensor]:
        return (torch.randn(1, FEATURES), torch.randn(1, FEATURES))

    example = case()
    goldens = [case() for _ in range(GOLDEN_CASES)]

    # The asymmetry is a claim about the weights, and a claim this fixture rests
    # on is a claim it should check rather than assume. Random initialisation
    # makes it overwhelmingly likely and not certain, and "overwhelmingly
    # likely" is how a fixture quietly stops discriminating.
    with torch.no_grad():
        primary, auxiliary = model(*example)
        swapped_primary, _ = model(example[1], example[0])
    if torch.allclose(primary, auxiliary):
        raise SystemExit("the two heads agree, so the outputs are interchangeable")
    if torch.allclose(primary, swapped_primary):
        raise SystemExit("the model ignores input order, so the inputs are interchangeable")

    for out_name, runtime, backend in (
        ("multi_io", "executorch", "portable"),
        ("multi_io_litert", "litert", "cpu"),
        ("multi_io_onnx", "onnx", "cpu"),
    ):
        out_dir = ROOT / "testdata" / out_name
        out_dir.mkdir(parents=True, exist_ok=True)
        result = export_model(
            model=model,
            example_inputs=example,
            name="multi_io",
            runtime=runtime,
            backend=backend,
            out_dir=out_dir,
            input_names=["left", "right"],
            output_names=["primary", "auxiliary"],
            golden_inputs=goldens,
        )
        m = result.manifest
        print(
            f"{runtime:11} {result.artifact.name}, "
            f"{result.artifact.stat().st_size} bytes, "
            f"{len(m.goldens)} goldens, "
            f"in {[(s.name, s.shape) for s in m.inputs]}, "
            f"out {[(s.name, s.shape) for s in m.outputs]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
