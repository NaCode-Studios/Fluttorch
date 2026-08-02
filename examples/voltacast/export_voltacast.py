"""M29 · VoltaCast, exported and measured.

A day-ahead forecast of Italian electricity demand: one week of hourly history
in, the next 24 hours out as P10, P50 and P90 quantiles. A seq2seq Transformer,
three encoder layers over the past and two decoder layers cross-attending to it,
trained on eleven years of real data.

    python examples/voltacast/export_voltacast.py

Needs `flatc` on PATH, which an ExecuTorch checkout builds:

    export PATH="$HOME/.cache/fluttorch/executorch/cmake-out-android/third-party/flatc_ep/bin:$PATH"

## What this example proves, and where it stops

It proves the thing the project exists for on a model that can go wrong: a
Transformer with softmax attention, 340 nodes, measured against references
captured from the source model on real windows.

It writes two bundles, and the reason is the point of the whole runtime layer.
ExecuTorch lowers this model and then fails to execute it, identically under its
own Python runtime, so the failure is upstream rather than at this seam. LiteRT
carries the same model, from the same weights and the same golden windows,
through the same gate, and agrees with the notebook to within float32 rounding.
One manifest, one set of references, two engines, and the one that works is the
one the example is measured on.

ONNX Runtime is the third and is refused for a reason of ours rather than
theirs: torch.onnx moves 3.4 MB of weights into a sidecar beside the graph, and
this toolchain refuses that bundle rather than writing one whose hash covers the
graph and not the numbers. That is issue 65, and VoltaCast is exactly the size
of model that triggers it.

It stops at the model boundary, and that is a statement rather than an omission.
VoltaCast's preprocessing is not expressible in the manifest and should not be.
Fluttorch's vocabulary describes tensor-level transforms: rescale, normalize,
resize, crop, cast. VoltaCast's is feature engineering over a DataFrame, with
sines and cosines of hour, weekday and day of year, the Italian holiday calendar
including the days beside a holiday, and heating and cooling degree days against
two different base temperatures. None of that is a tensor transform, generating
Dart for it would be a second implementation of a pipeline that already exists,
and a second implementation is the drift this project is built to catch.

So the contract starts where the tensors do. The golden inputs below are real
feature windows, built by VoltaCast's own code from real data, and committed as
tensors. What the gate then proves is that the same tensors produce the same
forecast on the device as in the notebook.
"""

from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "python" / "fluttorch_export"))

from fluttorch_export.export import export_model  # noqa: E402
from model import (  # noqa: E402
    FUTURE_FEATURES,
    PAST_FEATURES,
    Scaler,
    VoltaCastTransformer,
    build_features,
)

CHECKPOINT = HERE / "checkpoint" / "transformer.pt"
DATA = HERE / "checkpoint" / "italy_load_weather.parquet"
OUT_EXECUTORCH = ROOT / "testdata" / "voltacast"
OUT_LITERT = ROOT / "testdata" / "voltacast_litert"

#: Golden windows, taken from the test split so they are hours the model never
#: trained on. Four of them, at midnight, which is the origin a day-ahead
#: forecast is actually made at.
GOLDEN_WINDOWS = 4


def main() -> int:
    # weights_only=True, because a checkpoint is data and torch.load without it
    # unpickles arbitrary Python. This one carries nothing that needs the
    # unrestricted loader, and a script in a public repository is exactly where
    # the safe form should be the one people copy.
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    context = cfg["data"]["context_hours"]
    horizon = cfg["data"]["horizon_hours"]

    model = VoltaCastTransformer(
        len(PAST_FEATURES),
        len(FUTURE_FEATURES),
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        enc_layers=cfg["model"]["enc_layers"],
        dec_layers=cfg["model"]["dec_layers"],
        ff_dim=cfg["model"]["ff_dim"],
        dropout=cfg["model"]["dropout"],
        quantiles=tuple(cfg["model"]["quantiles"]),
    )
    model.load_state_dict(ck["state_dict"])
    model.eval()

    # VoltaCast's own feature pipeline, over VoltaCast's own data. Running it
    # rather than reimplementing it is the point: a second implementation would
    # be a second thing to keep in step.
    df = pd.read_parquet(DATA)
    features = build_features(
        df,
        Scaler(**ck["scaler"]),
        ck["temp_stats"]["temp_mean"],
        ck["temp_stats"]["temp_std"],
    )
    past = features[PAST_FEATURES].to_numpy(dtype=np.float32)
    future = features[FUTURE_FEATURES].to_numpy(dtype=np.float32)

    # Midnight origins from the end of the series, which is the test split by
    # construction: the checkpoint's validation window ends 2025-06-30 and this
    # data runs past it.
    hours = features.index.tz_convert("Europe/Rome").hour.to_numpy()
    origins = [t for t in range(len(features) - horizon, context, -1) if hours[t] == 0][
        :GOLDEN_WINDOWS
    ]
    origins.reverse()
    print(f"golden origins: {[str(features.index[t]) for t in origins]}")

    def window(t: int) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.from_numpy(past[t - context : t]).unsqueeze(0),
            torch.from_numpy(future[t : t + horizon]).unsqueeze(0),
        )

    example = window(origins[0])

    goldens = [window(t) for t in origins]
    labels = [f"p{int(q * 100)}" for q in cfg["model"]["quantiles"]]

    for out_dir, runtime, backend in (
        (OUT_EXECUTORCH, "executorch", "portable"),
        (OUT_LITERT, "litert", "cpu"),
    ):
        out_dir.mkdir(parents=True, exist_ok=True)
        result = export_model(
            model=model,
            example_inputs=example,
            name="voltacast",
            runtime=runtime,
            backend=backend,
            out_dir=out_dir,
            input_names=["past", "future"],
            output_names=["quantiles"],
            golden_inputs=goldens,
            labels=labels,
        )
        m = result.manifest
        print(
            f"{runtime:11} {result.artifact.name}, "
            f"{result.artifact.stat().st_size} bytes, "
            f"{len(m.goldens)} goldens, "
            f"in {[(s.name, s.shape) for s in m.inputs]}, "
            f"out {[(s.name, s.shape) for s in m.outputs]}"
        )

    # What the forecast means in megawatts, which the tensors do not say. The
    # model works in the scaler's units, and the scaler belongs to the training
    # run rather than to the artifact.
    scaler = Scaler(**ck["scaler"])
    with torch.no_grad():
        q = model(*example)[0]
    mw = scaler.inverse(q.numpy())
    print(
        f"first window, hour 0: P10 {mw[0, 0]:.0f} MW, P50 {mw[0, 1]:.0f} MW, P90 {mw[0, 2]:.0f} MW"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
