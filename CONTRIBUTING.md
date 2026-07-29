# Contributing

Fluttorch is pre-alpha and the architecture is still moving. If you are thinking
about a substantial change, open an issue first so we can agree on the shape
before anyone writes code.

## Layout

| Path | What lives there |
| --- | --- |
| `python/fluttorch_export` | Export CLI: `torch.export` → artifact, manifest, goldens |
| `packages/fluttorch` | Runtime-agnostic core: manifest, tensor specs, drift metrics, backend interface |
| `packages/fluttorch_gen` | `build_runner` builder: manifest → typed Dart API |
| `packages/fluttorch_test` | Parity gate: golden replay, drift reports, matchers |
| `packages/fluttorch_executorch` | ExecuTorch backend |
| `examples/` | Runnable examples, including the flagship forecasting model |

## Ground rules

**The core stays runtime-agnostic.** Nothing in `packages/fluttorch`,
`fluttorch_gen`, or `fluttorch_test` may import a backend. A parity gate is worth
the same on ExecuTorch, LiteRT and ONNX Runtime, and the interface in
`src/runtime.dart` is the only seam.

**The manifest is the single source of truth.** Shapes, dtypes, preprocessing and
labels are emitted once by the exporter and consumed everywhere else. Anything
hand-written on the Dart side that duplicates them is a bug waiting to happen —
that duplication is the problem this project exists to remove.

**Backend capabilities are reported, never assumed.** Whether activation taps or
deterministic execution are available is a property of the device. Code that
assumes them must degrade rather than fail.

## Local setup

```bash
dart pub get                                   # resolves the workspace
pip install -e python/fluttorch_export[dev]
```

```bash
dart analyze && dart test
```

## Commits and branches

`main` is protected; work on a branch and open a pull request. Keep commits
scoped to one change and write the message in the imperative mood.
