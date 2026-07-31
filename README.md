<p align="center">
  <img src="docs/fluttorch-hero.png" alt="Fluttorch: ship PyTorch models to Flutter, and prove the numbers didn't change" width="100%">
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
Preprocessing goes wrong the other way round. Written once in Python for training and again in Dart
for serving, the two copies drift apart at the first refactor.

Fluttorch closes both gaps with one document. The exporter writes a manifest next to the artifact,
the Dart API is generated from it so that no shape or normalization constant is ever restated by
hand, and the reference outputs captured during the export replay in your test suite.

```dart
// generated from the manifest the exporter emitted, never hand-written
final forecast = await SolarForecast.load(runtime, artifact: bytes);
final out = await forecast.run(features: window);   // shape- and dtype-checked

// and in the test suite, where the tolerance comes from the recipe the
// manifest records unless you pass one you measured yourself
await expectParity(
  forecast.model,
  goldens: await DirectoryGoldenBundle.open('build/solar_forecast.fluttorch.json'),
);
```

```
FAIL  parity/case-3
      backend: xnnpack  quantization: int8-static
      output "load_mw"  max |Δ| 1.72  >  Tolerance(atol 0.1, rtol 0.1, cos ≥ 0.998)  worst at [0]: 14.0210 vs 12.3000
        2 of 4 elements (50.0%) exceed the elementwise bound
      no layer attribution: backend "xnnpack" offers no activation taps
```

Fluttorch does not train models, convert between formats, or serve inference. It takes a model you
exported with `torch.export` and makes the boundary between Python and Dart typed and verified. The
workflow that runs the gate on every pull request is in
[`docs/ci-parity-gate.md`](docs/ci-parity-gate.md).

> **Status — `0.3.0`, early development.** The loop from `torch.export` to a typed Dart API to a
> parity gate is complete: one command produces an artifact, its manifest and its goldens, the
> generated bindings make handing the model the wrong tensor a compile error, and `expectParity`
> replays the goldens and fails the build when the numbers move too far. What none of it runs on yet
> is a device: `fluttorch_executorch` has nothing behind it, so the gate is exercised against
> replayed goldens rather than against a backend, and that package is not published.
> The [board](https://github.com/orgs/NaCode-Studios/projects/6) is the plan of record and tracks it
> milestone by milestone. Until `1.0`, minor versions may break.

## Why Fluttorch

The parity gate is the reason this project exists. Typed bindings are a convenience; catching
numerical drift is the part nothing else does. Comparing offline, online and post-serialization
predictions is the documented advice everywhere and a manual chore everywhere. Here it is a matcher
that fails a build, and it names the tensor, the drift and the backend it measured.

Shapes, dtypes, preprocessing, labels and the weight hash are emitted once by the exporter and read
by three consumers: the code generator, the parity gate and the runtime. Nothing on the Dart side
restates them, so nothing there can disagree with training. The runtime also refuses to load an
artifact whose content hash is not the one its manifest recorded, which is what catches a re-export
that updated only half of the pair.

ExecuTorch is the first backend, not the architecture. A parity gate is worth exactly the same on
LiteRT and ONNX Runtime, and `FluttorchRuntime` is the only seam a backend touches. Nothing above
that seam knows which runtime is executing the model.

Capabilities are reported, never assumed. Whether a device can expose intermediate activations or
execute deterministically is a property of the hardware, not of the build. Per-layer drift
attribution depends on both, so they are declared capabilities that code degrades around, and a
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

**Shipped (`0.3.0`).** Tiers 0 to 3. The manifest schema, with a canonical Python writer and a Dart
reader that reproduce the same document byte for byte down to denormals and negative zero;
`fluttorch-export`, which produces an artifact, its manifest and its goldens in one command;
`fluttorch_gen`, which turns that manifest into an API where the compiler rejects the wrong tensor;
and the parity gate, which replays those goldens and fails the build with a report naming the
tensor, the bound it broke and the backend that ran it.

**Next.** Quantization recipes, and attribution of the drift they cause to the layer that caused it.
That is also what forces the runtime question: attributing drift needs activation taps and
deterministic execution, and the bindings the ExecuTorch backend delegates to expose neither, so the
milestone that closes the tier is a decision about when to fork rather than whether.

**Later.** An own `dart:ffi` binding with deterministic execution, then the parity matrix across
backends, then LiteRT and ONNX Runtime, which is what turns the runtime-agnostic claim into
something measured rather than stated.

The [board](https://github.com/orgs/NaCode-Studios/projects/6) carries the milestone plan and its
exit criteria, and every tier is a [milestone](https://github.com/NaCode-Studios/Fluttorch/milestones)
in this repository.

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
