# Fluttorch

**Ship PyTorch models to Flutter, and prove the numbers didn't change.**

[![CI](https://github.com/NaCode-Studios/Fluttorch/actions/workflows/ci.yaml/badge.svg)](https://github.com/NaCode-Studios/Fluttorch/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/license-Apache%202.0-232B45?labelColor=0B0E17)](LICENSE)
[![Dart](https://img.shields.io/badge/Dart-3.9-0175C2?logo=dart&logoColor=white&labelColor=0B0E17)](https://dart.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white&labelColor=0B0E17)](https://pytorch.org)

Getting a model onto a phone means exporting it and usually quantizing it. Both steps change
the numbers, and nothing tells you when they change too much: the build succeeds, the app runs,
the output looks plausible, and the model is wrong. Preprocessing makes it worse — it is written
once in Python for training and once in Dart for serving, and the two drift apart on the first
refactor. Fluttorch closes both holes by making the export emit a contract, generating the Dart
side from that contract, and turning the reference outputs into a test.

```dart
// generated from the manifest the exporter emitted — not hand-written
final forecast = await SolarForecast.load();
final out = await forecast.run(features: window);   // shape- and dtype-checked

// and in your test suite
await expectParity(forecast, goldens: SolarForecast.goldens, tolerance: Tolerance.int8);
```

```
FAIL  parity/solar_forecast
      backend: xnnpack  quantization: int8-static
      output[0] "load_mw"  max |Δ| 0.118  >  tolerance 0.010
      first divergence: layer conv3
```

Fluttorch does not train, convert between formats, or serve models. It takes a model you have
already exported with `torch.export`, and it makes the boundary between Python and Dart typed
and verified.

> **Status — pre-alpha, no released version.** The public API does not exist yet; the packages in
> this repository declare the architecture and nothing more. Progress is tracked on the
> [roadmap board](https://github.com/orgs/NaCode-Studios/projects/6), and
> [`ROADMAP.md`](ROADMAP.md) is the plan of record. Pre-`1.0`, minor versions may break.

## Why Fluttorch

**The parity gate is the product.** Typed bindings are a convenience; catching silent numerical
drift is the thing nothing else does. Comparing offline, online and post-serialization predictions
is the documented recommendation everywhere, and it is a manual chore everywhere. Here it is a
matcher that fails a build.

**One contract, three consumers.** Shapes, dtypes, preprocessing, labels and the weight hash are
emitted once by the exporter and consumed by the code generator, the parity gate and the runtime.
Nothing on the Dart side restates them, so nothing can disagree with training.

**Runtime-agnostic on purpose.** ExecuTorch is the first backend, not the architecture. A parity
gate is worth exactly the same on LiteRT and ONNX Runtime, and the interface in
`packages/fluttorch/lib/src/runtime.dart` is the only seam a backend touches.

**Capabilities are reported, never assumed.** Whether a device can expose intermediate activations
or run deterministically is a property of the hardware. Code that needs either must degrade rather
than fail, and the parity gate says which backend it actually measured.

## Architecture

```
python/fluttorch_export     torch.export → artifact + signed manifest + goldens
        │
        ▼
packages/fluttorch          manifest, tensor specs, drift metrics, runtime interface
        ├── fluttorch_gen         manifest → typed Dart API (build_runner)
        ├── fluttorch_test        golden replay, parity matchers, drift reports
        └── fluttorch_executorch  first backend
```

| Package | Role |
| --- | --- |
| `fluttorch` | The contract and the seam. No backend may be imported here. |
| `fluttorch_gen` | `build_runner` builder: `*.fluttorch.json` → `*.fluttorch.dart`. |
| `fluttorch_test` | Replays goldens, measures drift, fails the build. |
| `fluttorch_executorch` | ExecuTorch backend. LiteRT and ONNX Runtime are planned. |
| `fluttorch-export` (Python) | Emits the artifact, the manifest and the goldens together. |

## Roadmap

**Next** — the export toolchain and the manifest schema, then typed codegen, then the parity gate
over final outputs. Those three make the loop usable end to end on XNNPACK.

**Later** — quantization recipes with per-layer drift attribution, which is what forces the runtime
question: attributing drift needs activation taps and deterministic execution, and no existing Dart
binding exposes either. Own `dart:ffi` binding after that, then the multi-backend parity matrix, then
LiteRT and ONNX Runtime to make the runtime-agnostic claim real rather than stated.

The full plan is [`ROADMAP.md`](ROADMAP.md); the conventions it follows are
[`ROADMAP-CONVENTIONS.md`](ROADMAP-CONVENTIONS.md), shared with
[Kdrant](https://github.com/NaCode-Studios/Kdrant) and
[Kmemo](https://github.com/NaCode-Studios/Kmemo).

## Building and testing

```bash
dart pub get                                   # resolves the workspace
dart analyze --fatal-warnings
dart format --output=none --set-exit-if-changed .
cd packages/fluttorch && dart test
```

```bash
pip install -e python/fluttorch_export
ruff check python/ && ruff format --check python/
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). The three ground rules that matter most: the core stays
runtime-agnostic, the manifest is the single source of truth, and backend capabilities are reported
rather than assumed.

## License

Apache-2.0. See [LICENSE](LICENSE).
