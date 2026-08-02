"""Produces the fixture that proves the generated spatial steps match training.

A resize and a crop are the two preprocessing steps whose correctness cannot be
argued from the code: bilinear has a half-pixel convention, nearest does not use
it, and a centre crop with an odd margin lands on one row or the row above it.
Each of those is a choice that produces a picture either way, so the only useful
check is whether Dart lands on the same numbers torch does.

This writes both halves of that check:

  testdata/vision/vision.fluttorch.json   the manifest, declaring the layout
  testdata/vision/vision.pte              the artifact the hash is taken over
  testdata/vision/goldens/                model input to model output
  testdata/vision/preprocessing/          source frame to preprocessed tensor

The last one is the one that matters here. `source-N.bin` is a frame at a size
no model accepts, and `expected-N.bin` is what torch produced from it, so a Dart
side that disagrees is wrong about training rather than wrong about itself.

    python python/fluttorch_export/scripts/export_vision.py

Needs `flatc` on PATH, which the ExecuTorch checkout builds:

    export PATH="$HOME/.cache/fluttorch/executorch/cmake-out-android/third-party/flatc_ep/bin:$PATH"
"""

from __future__ import annotations

import pathlib
import sys

import torch
from torch.nn import functional as F

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from fluttorch_export.export import export_model  # noqa: E402
from fluttorch_export.manifest import center_crop, normalize, resize  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "testdata" / "vision"

# The model input. NCHW because that is what torch hands you, and the point of
# recording it is that NHWC would read the same bytes as a different picture.
CHANNELS, MODEL_H, MODEL_W = 3, 8, 8
# The intermediate the resize targets, cropped down to the model's extent. The
# two margins differ on purpose, and neither is even.
#
# Height leaves a margin of 5, so the offset is 2.5, and that is the case that
# discriminates: Python's round breaks the tie to the even number and gives 2,
# Dart's breaks it away from zero and gives 3. A crop one row off still looks
# like a picture.
#
# Width leaves 3, so the offset is 1.5, where the two rules happen to agree. It
# is here so a generator that handled only one axis would still be caught.
RESIZE_H, RESIZE_W = 13, 11
# Source frames, none of them a multiple of the target. The third is smaller
# than the resize target so the upsampling path is covered too.
SOURCES = [(23, 17), (12, 30), (5, 6)]

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class TinyVision(torch.nn.Module):
    """Small enough to export in a second, deep enough to have an interior."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(CHANNELS, 4, kernel_size=3, padding=1)
        self.act = torch.nn.ReLU()
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pool(self.act(self.conv(x))).flatten(1)
        return self.fc(h)


def preprocess(source: torch.Tensor) -> torch.Tensor:
    """The pipeline the manifest records, performed by torch.

    Kept in this order because the manifest records it in this order and the
    generator refuses a manifest that puts an elementwise step first. Bilinear
    resize and an affine normalize commute in exact arithmetic, which is exactly
    why the order has to be pinned rather than assumed to be harmless.
    """
    resized = F.interpolate(source, size=(RESIZE_H, RESIZE_W), mode="bilinear", align_corners=False)
    top = round((RESIZE_H - MODEL_H) / 2)
    left = round((RESIZE_W - MODEL_W) / 2)
    cropped = resized[:, :, top : top + MODEL_H, left : left + MODEL_W]
    mean = torch.tensor(MEAN).view(1, CHANNELS, 1, 1)
    std = torch.tensor(STD).view(1, CHANNELS, 1, 1)
    return (cropped - mean) / std


def main() -> int:
    torch.manual_seed(20260802)
    model = TinyVision().eval()

    example = torch.randn(1, CHANNELS, MODEL_H, MODEL_W)

    OUT.mkdir(parents=True, exist_ok=True)
    bundle = export_model(
        model=model,
        example_inputs=(example,),
        name="vision",
        out_dir=OUT,
        input_names=["image"],
        output_names=["logits"],
        golden_inputs=[(torch.randn(1, CHANNELS, MODEL_H, MODEL_W),) for _ in range(3)],
        preprocessing=[
            resize(RESIZE_H, RESIZE_W, "bilinear"),
            center_crop(MODEL_H, MODEL_W),
            normalize(MEAN, STD, axis=1),
        ],
        input_layouts={"image": "nchw"},
    )

    # The cross-language half. Written after the export so the manifest it is
    # measured against is the one on disk.
    ref = OUT / "preprocessing"
    ref.mkdir(exist_ok=True)
    for i, (h, w) in enumerate(SOURCES):
        source = torch.randn(1, CHANNELS, h, w)
        expected = preprocess(source)
        (ref / f"source-{i}.bin").write_bytes(
            source.contiguous().numpy().astype("float32").tobytes()
        )
        (ref / f"expected-{i}.bin").write_bytes(
            expected.contiguous().numpy().astype("float32").tobytes()
        )
        print(f"  preprocessing/{i}: {h}x{w} -> {tuple(expected.shape)}")

    (ref / "sources.txt").write_text(
        "\n".join(f"{h} {w}" for h, w in SOURCES) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {bundle}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
