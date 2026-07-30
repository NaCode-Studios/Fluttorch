<p align="center">
  <img src="docs/fluttorch-hero.png" alt="Fluttorch — ship PyTorch models to Flutter, and prove the numbers didn't change" width="100%">
</p>

# Fluttorch

**Ship PyTorch models to Flutter, and prove the numbers didn't change.**

[![CI](https://github.com/NaCode-Studios/Fluttorch/actions/workflows/ci.yaml/badge.svg)](https://github.com/NaCode-Studios/Fluttorch/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/license-Apache%202.0-23201C?labelColor=100E0C)](LICENSE)
[![Dart](https://img.shields.io/badge/Dart-3.9-0553B1?logo=dart&logoColor=white&labelColor=100E0C)](https://dart.dev)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C?logo=pytorch&logoColor=white&labelColor=100E0C)](https://pytorch.org)
[![ExecuTorch](https://img.shields.io/badge/ExecuTorch-1.3-EE4C2C?labelColor=100E0C)](https://docs.pytorch.org/executorch/stable/)

Putting a model on a phone means exporting it, and usually quantizing it. Both steps change the
numbers, and nothing in the toolchain tells you when they change too much: the build succeeds, the
app runs, the output looks plausible, and the model is quietly worse than the one you evaluated.
Preprocessing compounds it — written once in Python for training and again in Dart for serving, the
two drift apart at the first refactor.

Fluttorch closes both gaps by making the export emit a contract, generating the Dart side from that
contract, and turning the reference outputs into a test that fails.

```dart
// generated from the manifest the exporter emitted — never hand-written
final forecast = await SolarForecast.load();
final out = await forecast.run(features: window);   // shape- and dtype-checked

// and in the test suite
await expectParity(
  forecast,
  goldens: SolarForecast.goldens,
  tolerance: Tolerance.startingPointFor('int8-static'),
);
```

```
FAIL  parity/solar_forecast
      backend: xnnpack  quantization: int8-static
      output[0] "load_mw"  max |Δ| 0.118  >  tolerance 0.010
      first divergence: layer conv3
```

Fluttorch does not train models, convert between formats, or serve inference. It takes a model you
exported with `torch.export` and makes the boundary between Python and Dart typed and verified.

> **Status — `0.1.0`, early development.** `fluttorch-export` works: one command produces an
> artifact, its manifest and its goldens, and the Dart side refuses an artifact that does not match
> the manifest it was handed. What does not exist yet is the typed API — that is Tier 2.
> The [board](https://github.com/orgs/NaCode-Studios/projects/6) is the plan of record and tracks it milestone by milestone. Until
> `1.0`, minor versions may break.

## Why Fluttorch

**The parity gate is the product.** Typed bindings are a convenience; catching numerical drift is the
thing nothing else does. Comparing offline, online and post-serialization predictions is the
documented advice everywhere and a manual chore everywhere. Here it is a matcher that fails a build,
and it names the tensor, the drift and the backend it measured.

**One contract, three consumers.** Shapes, dtypes, preprocessing, labels and the weight hash are
emitted once by the exporter and read by the code generator, the parity gate and the runtime. Nothing
on the Dart side restates them, so nothing can disagree with training. The runtime refuses to load an
artifact whose content hash does not match the manifest it was generated from.

**Runtime-agnostic by construction.** ExecuTorch is the first backend, not the architecture. A parity
gate is worth exactly the same on LiteRT and ONNX Runtime, and `FluttorchRuntime` is the only seam a
backend touches.

**Capabilities are reported, never assumed.** Whether a device can expose intermediate activations or
execute deterministically is a property of the hardware, not of the build. Per-layer drift
attribution depends on both, so they are declared capabilities that code degrades around — and a
report always says which backend produced it.

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
| `fluttorch_test` | Replays the goldens, measures drift, fails the build. |
| `fluttorch_executorch` | ExecuTorch backend. LiteRT and ONNX Runtime are planned. |
| `fluttorch-export` (Python) | Emits the artifact, the manifest and the goldens together. |

## Roadmap

**Shipped (`0.1.0`)** — Tier 0 and Tier 1. The manifest schema, with a canonical Python writer and a
Dart reader that reproduce the same document byte for byte down to denormals and negative zero; the
tensor and tolerance semantics the gate is built on; and `fluttorch-export`, which produces an
artifact, its manifest and its goldens in one command.

**Next** — typed codegen from the manifest, then the parity gate over final outputs. Those two make
the loop usable end to end on XNNPACK.

**Later** — quantization recipes with per-layer drift attribution. That is what forces the runtime
question: attributing drift needs activation taps and deterministic execution, and no existing Dart
binding exposes either. An own `dart:ffi` binding follows, then the multi-backend parity matrix, then
LiteRT and ONNX Runtime to make the runtime-agnostic claim measured rather than stated.

The [board](https://github.com/orgs/NaCode-Studios/projects/6) carries the milestone plan and its exit criteria, and every tier is
a [milestone](https://github.com/NaCode-Studios/Fluttorch/milestones) in this repository.

## Building and testing

```bash
dart pub get
dart analyze --fatal-warnings
dart format --output=none --set-exit-if-changed .
for p in fluttorch fluttorch_test fluttorch_gen; do (cd packages/$p && dart test); done
```

```bash
pip install -e 'python/fluttorch_export[dev]'
ruff check python/ && ruff format --check python/
PYTHONPATH=python/fluttorch_export pytest python/fluttorch_export/tests
```

The manifest suite needs neither `torch` nor `executorch`. The export half does, and
[`examples/spike/`](examples/spike/) shows the whole path end to end.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Three rules shape everything else: the core stays
runtime-agnostic, the manifest is the single source of truth, and backend capabilities are reported
rather than assumed.

Brand assets and design tokens live in [`docs/brand/`](docs/brand/).

## License

Apache-2.0. See [LICENSE](LICENSE).

## Sponsor

If Fluttorch is useful to you, consider [sponsoring NaCode Studios](https://github.com/sponsors/NaCode-Studios).
