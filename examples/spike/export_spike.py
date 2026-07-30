"""M2 · the end-to-end spike, export half.

A two-layer model, exported to ExecuTorch, with the manifest and one golden case
emitted beside it. Deliberately not abstracted: the point of a spike is to find
where the pipeline breaks, and an abstraction written before that is a guess.

The golden is captured from the *source* model, before lowering, which is what
makes it a reference rather than a snapshot of whatever the export produced.

    python examples/spike/export_spike.py
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.exir import to_edge_transform_and_lower

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "python" / "fluttorch_export"))

from fluttorch_export.manifest import (  # noqa: E402
    GoldenCase,
    ModelManifest,
    TensorSpec,
)

OUT = pathlib.Path(__file__).resolve().parent / "build"
IN_FEATURES, HIDDEN, OUT_FEATURES = 4, 8, 3


class TwoLayer(torch.nn.Module):
    """The smallest model that still has an interior worth attributing drift to."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(IN_FEATURES, HIDDEN)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(HIDDEN, OUT_FEATURES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def main() -> int:
    # A fixed seed so the artifact and the golden are reproducible: a spike whose
    # numbers move between runs cannot tell you whether the pipeline moved them.
    torch.manual_seed(20260730)
    model = TwoLayer().eval()

    example = torch.randn(1, IN_FEATURES)
    with torch.no_grad():
        reference = model(example)

    OUT.mkdir(parents=True, exist_ok=True)

    exported = torch.export.export(model, (example,))
    lowered = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    ).to_executorch()

    pte = OUT / "two_layer.pte"
    pte.write_bytes(lowered.buffer)
    weight_hash = "sha256:" + hashlib.sha256(lowered.buffer).hexdigest()

    # The golden: raw little-endian float32, the representation the Dart Tensor
    # wraps without copying.
    (OUT / "in_0_x.bin").write_bytes(example.numpy().astype("<f4").tobytes())
    (OUT / "out_0_y.bin").write_bytes(reference.numpy().astype("<f4").tobytes())

    manifest = ModelManifest(
        name="two_layer",
        weight_hash=weight_hash,
        inputs=(TensorSpec("x", "float32", (1, IN_FEATURES)),),
        outputs=(TensorSpec("y", "float32", (1, OUT_FEATURES)),),
        goldens=(
            GoldenCase(
                "case-0",
                ("in_0_x.bin",),
                ("out_0_y.bin",),
                "one random input, captured from the source model before lowering",
            ),
        ),
    )
    (OUT / "two_layer.fluttorch.json").write_text(manifest.to_json())

    print(f"artifact     {pte.name}  {pte.stat().st_size} bytes")
    print(f"weight hash  {weight_hash}")
    print(f"input        {example.flatten().tolist()}")
    print(f"reference    {reference.flatten().tolist()}")
    print(f"written to   {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
