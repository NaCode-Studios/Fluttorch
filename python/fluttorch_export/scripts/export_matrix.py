"""Produces the model the parity matrix is worth running on.

The matrix has only ever measured a network of two linear layers, 4 to 8 to 3.
Its columns come out ordered the way arithmetic says they should, which is
evidence that the measurement works and no evidence at all that it is useful:
nothing in that model touches the places backends actually diverge.

This writes one that does. Convolutions, two kinds of normalisation, and a
softmax, exported once per backend this toolchain can lower for:

  testdata/matrix/<backend>/matrix.fluttorch.json

One bundle per backend rather than one shared artifact, because an ExecuTorch
export is lowered *for* a delegate. Handing the xnnpack artifact to Core ML
would measure which file was loaded rather than which delegate ran it.

## Why these layers

Each one is here because it is somewhere two delegates can legitimately
disagree, not to make the model deep.

`BatchNorm2d` in eval mode is an affine the exporter can fold into the
convolution before it. Whether a given delegate folds it is its own decision,
and the two orders do not round identically.

`GroupNorm` cannot be folded: it reduces over the group at run time, so every
delegate computes a mean and a variance and picks its own order to sum in. That
is the reduction the two-layer model never had.

`softmax` is the second, over a different axis, and it carries a max-subtract
whose ties are broken per implementation.

The convolutions themselves are the ones with the widest choice of algorithm:
direct, im2col, Winograd. They agree to within rounding and not beyond it, which
is exactly the kind of difference the gate exists to bound.

    python python/fluttorch_export/scripts/export_matrix.py

Needs `flatc` on PATH, which the ExecuTorch checkout builds:

    export PATH="$HOME/.cache/fluttorch/executorch/cmake-out-android/third-party/flatc_ep/bin:$PATH"
"""

from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fluttorch_export.export import ExportError, available_backends, export_model  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "testdata" / "matrix"

CHANNELS, SIZE = 3, 32
WIDTH, CLASSES = 8, 4

#: Eight rather than four, because the tolerance this model needs is read off
#: this suite and four numbers is not a distribution.
GOLDEN_CASES = 8

#: Every backend worth attempting. Filtered against what this toolchain can
#: actually lower for, and what fails is reported rather than dropped: a matrix
#: that quietly loses a column reads as a machine with nothing to say about it,
#: and that is a different claim from coverage.
CANDIDATES = ("portable", "xnnpack", "coreml", "mps", "mlx")


class ConvNet(torch.nn.Module):
    """Small, and made entirely of places two delegates can disagree."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(CHANNELS, WIDTH, kernel_size=3, padding=1)
        # Foldable into conv1, and whether a delegate folds it is its own call.
        self.bn = torch.nn.BatchNorm2d(WIDTH)
        self.act = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(WIDTH, WIDTH, kernel_size=3, stride=2, padding=1)
        # Not foldable: reduces over the group at run time, so every delegate
        # picks its own order to sum in.
        self.norm = torch.nn.GroupNorm(2, WIDTH)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(WIDTH, CLASSES)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        h = self.act(self.bn(self.conv1(image)))
        h = self.act(self.norm(self.conv2(h)))
        h = self.pool(h).flatten(1)
        # A second reduction, over a different axis, with a max-subtract whose
        # ties are broken per implementation.
        return torch.softmax(self.fc(h), dim=-1)


def main() -> int:
    torch.manual_seed(20260802)
    model = ConvNet().eval()

    example = torch.randn(1, CHANNELS, SIZE, SIZE)
    goldens = [(torch.randn(1, CHANNELS, SIZE, SIZE),) for _ in range(GOLDEN_CASES)]

    exportable = available_backends()
    written, refused = [], []
    for backend in CANDIDATES:
        if backend not in exportable:
            refused.append((backend, "this toolchain cannot lower for it here"))
            continue
        out_dir = OUT / backend
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = export_model(
                model=model,
                example_inputs=(example,),
                name="matrix",
                backend=backend,
                out_dir=out_dir,
                input_names=["image"],
                output_names=["probabilities"],
                golden_inputs=goldens,
                input_layouts={"image": "nchw"},
            )
        except (ExportError, RuntimeError) as e:
            refused.append((backend, f"{type(e).__name__}: {str(e).strip()[:120]}"))
            continue
        written.append(backend)
        print(
            f"{backend:9} {result.artifact.stat().st_size:>9} bytes, "
            f"{len(result.manifest.goldens)} goldens, "
            f"precision {result.manifest.precision or 'float32'}"
        )

    # And once per recipe, so the quantized rows of the tolerance table are
    # measured against the same model as the rest rather than against whatever
    # was to hand when each one was written.
    for recipe in ("int8-dynamic", "int8-static"):
        out_dir = OUT / recipe
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = export_model(
                model=model,
                example_inputs=(example,),
                name="matrix",
                backend="xnnpack",
                quantization=recipe,
                out_dir=out_dir,
                input_names=["image"],
                output_names=["probabilities"],
                golden_inputs=goldens,
                input_layouts={"image": "nchw"},
            )
        except (ExportError, RuntimeError) as e:
            refused.append((recipe, f"{type(e).__name__}: {str(e).strip()[:120]}"))
            continue
        written.append(recipe)
        print(
            f"{recipe:16} {result.artifact.stat().st_size:>9} bytes, "
            f"{len(result.manifest.goldens)} goldens, "
            f"quantization {result.manifest.quantization}"
        )

    for backend, why in refused:
        print(f"{backend:9} not written: {why}")

    if not written:
        print("no backend on this machine lowered the model, so there is no matrix")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
