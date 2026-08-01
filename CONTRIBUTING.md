# Contributing

Fluttorch is pre-alpha and the architecture is still moving. If you are thinking
about a substantial change, open an issue first so we can agree on the shape
before anyone writes code.

## Layout

| Path | What lives there |
| --- | --- |
| `python/fluttorch_export` | Export CLI: `torch.export` to artifact, manifest, goldens |
| `packages/fluttorch` | Runtime-agnostic core: manifest, tensor specs, drift metrics, backend interface |
| `packages/fluttorch_gen` | `build_runner` builder: manifest to typed Dart API |
| `packages/fluttorch_test` | Parity gate: golden replay, drift reports, matchers |
| `packages/fluttorch_executorch` | ExecuTorch backend |
| `examples/` | Runnable examples: the end-to-end spike and the generated typed API |

## Ground rules

The core stays runtime-agnostic. Nothing in `packages/fluttorch`, `fluttorch_gen`
or `fluttorch_test` may import a backend. A parity gate is worth the same on
ExecuTorch, LiteRT and ONNX Runtime, and the interface in `src/runtime.dart` is
the only seam.

The manifest is the single source of truth. Shapes, dtypes, preprocessing and
labels are emitted once by the exporter and consumed everywhere else. Anything
hand-written on the Dart side that duplicates them is the train/serve skew this
project exists to remove, arriving through the back door.

Backend capabilities are reported, never assumed. Whether activation taps or
deterministic execution are available is a property of the device. Code that
needs one of them asks, and degrades when the answer is no.

## Local setup

```bash
dart pub get                                   # resolves the workspace
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch executorch pytest
```

Activate the environment rather than calling `.venv/bin/python` directly. The
export shells out to `flatc` to serialise an artifact, and that binary is
installed into the environment's `bin`: without it on `PATH` every export test
fails on a missing file rather than on anything to do with the model.

```bash
dart analyze && dart test
```

## Commits and branches

`main` is protected; work on a branch and open a pull request. Keep commits
scoped to one change and write the message in the imperative mood.
