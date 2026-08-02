<p align="center">
  <img src="docs/fluttorch-hero.png" alt="Fluttorch: ship PyTorch models to Flutter, and prove the numbers didn't change" width="100%">
</p>

# Fluttorch

**Ship PyTorch models to Flutter, and prove the numbers didn't change.**

[![CI](https://github.com/NaCode-Studios/Fluttorch/actions/workflows/ci.yaml/badge.svg)](https://github.com/NaCode-Studios/Fluttorch/actions/workflows/ci.yaml)
[![pub.dev](https://img.shields.io/pub/v/fluttorch?label=pub.dev&labelColor=100E0C&color=0553B1)](https://pub.dev/packages/fluttorch)
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
exported with `torch.export` and makes the boundary between Python and Dart typed and verified.

**The documentation is at [nacode-studios.github.io/Fluttorch](https://nacode-studios.github.io/Fluttorch/)**:
the [concepts](https://nacode-studios.github.io/Fluttorch/concepts.html), a
[quickstart](https://nacode-studios.github.io/Fluttorch/quickstart.html), where a
[tolerance](https://nacode-studios.github.io/Fluttorch/tolerance.html) comes from, and
[what each failure means](https://nacode-studios.github.io/Fluttorch/errors.html) including the one
that is deliberately not a failure at all.

> **Status — `0.7.0`, early development.** The loop from `torch.export` to a typed Dart API to a
> parity gate runs on hardware, reaches a phone, and now carries a model that can go wrong. VoltaCast
> is a seq2seq Transformer forecasting Italian electricity demand a day ahead, trained on eleven years
> of real data, and it holds its goldens to `5.96e-7` through LiteRT. Getting there used the runtime
> layer for the first time rather than only demonstrating it: ExecuTorch lowers that model and fails
> to execute it, so the same bundle went to a different engine, unchanged.
> The tolerances are now measured rather than assumed, and the measurement said something worth
> knowing: the two-layer model drifts up to eight times further than the convolutional one, and it is
> the simpler network. Its outputs land near `9.4` while the other ends in a softmax, and a relative
> error is measured against the output while the rounding happened on intermediates. One bound is
> still unmeasured and says so, because nothing here exports int4.
> `fluttorch_executorch` stays unpublished because pub.dev cannot carry the native half.
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

## Installation

```yaml
dependencies:
  fluttorch: ^0.7.0

dev_dependencies:
  build_runner: ^2.4.0
  fluttorch_gen: ^0.7.0
  fluttorch_test: ^0.7.0
```

The generator and the gate are `dev_dependencies`: neither runs in a shipped app, and an app that
carried the drift metrics into production would be paying for something it cannot use there.

The exporter is the Python half and is installed from this repository, since it is a command rather
than a library:

```bash
pip install 'git+https://github.com/NaCode-Studios/Fluttorch.git#subdirectory=python/fluttorch_export'
```

The three bindings execute models and are in the repository unpublished, because pub.dev cannot carry
the native half. Building one is `tool/build_native.sh` in its package.

A published archive carries a [SLSA build provenance](https://slsa.dev/) attestation when the
release workflow was able to produce one, so you can check that what you resolved was built here
rather than uploaded by someone else. `0.3.0` and `0.4.0` have none, and cannot be given one after
the fact: an attestation is signed against the run that produced the archive, and neither of those
runs made one.

```bash
gh attestation verify <archive> --repo NaCode-Studios/Fluttorch
```

## Architecture

```
python/fluttorch_export     torch.export → artifact + signed manifest + goldens
        │
        ▼
packages/fluttorch          manifest, tensor specs, drift metrics, runtime interface
        ├── fluttorch_gen         manifest → typed Dart API (build_runner)
        ├── fluttorch_test        golden replay, parity matchers, drift reports
        ├── fluttorch_executorch  the C ABI, and ExecuTorch behind it
        ├── fluttorch_onnx        the same ABI, over ONNX Runtime
        ├── fluttorch_litert      the same ABI, over LiteRT
        └── fluttorch_executorch_flutter  the native half, inside a Flutter app
```

| Package | Role |
| --- | --- |
| [`fluttorch`](https://pub.dev/packages/fluttorch) | The contract and the seam. No backend may be imported here. |
| [`fluttorch_gen`](https://pub.dev/packages/fluttorch_gen) | `build_runner` builder: `*.fluttorch.json` → `*.fluttorch.dart`. |
| [`fluttorch_test`](https://pub.dev/packages/fluttorch_test) | Replays the goldens, measures drift, fails the build. |
| `fluttorch_executorch` | ExecuTorch backend, and the `dart:ffi` client three runtimes share. Unpublished: it carries a native build. |
| `fluttorch_onnx`, `fluttorch_litert` | The same C ABI over ONNX Runtime and LiteRT. Unpublished, for the same reason. |
| `fluttorch_executorch_flutter` | Puts the native half inside a Flutter app, on Android and on iOS. |
| `fluttorch-export` (Python) | Emits the artifact, the manifest and the goldens together. |

## Roadmap

**Shipped (`0.4.0`).** Tiers 0 to 4. The manifest schema, with a canonical Python writer and a Dart
reader that reproduce the same document byte for byte down to denormals and negative zero;
`fluttorch-export`, which produces an artifact, its manifest and its goldens in one command, and
quantizes it on request; `fluttorch_gen`, which turns that manifest into an API where the compiler
rejects the wrong tensor; and the parity gate, which replays those goldens, fails the build with a
report naming the tensor, the bound it broke and the backend that ran it, and where the export
captured taps also names the earliest layer whose numbers moved.

**Shipped (`0.5.0`).** Tiers 5 and 6. Fluttorch's own `dart:ffi` binding to ExecuTorch, carrying the
four hooks no published binding exposes: a backend pinned at load, execution repeatable enough that a
tolerance measures the model rather than the hardware, activation taps, and output buffers the caller
owns. Eight backends the exporter knows and reports honestly on the machine it is asked from, with
XNNPACK, Core ML and MPS measured rather than described. One parity report across all of them, each
column answering to the bound its own manifest implies, from the recipe and the precision together.

**Shipped (`0.6.0`).** Tier 7. The binding reaches a phone: ExecuTorch and the shim cross-compile for
Android arm64 and for iOS, and `fluttorch_executorch_flutter` puts the result inside an app, as
`jniLibs` on Android and as an archive force-loaded into the binary on iOS, where an app cannot load
a dylib from its bundle. ONNX Runtime and LiteRT implement the same C ABI as ExecuTorch, which turns
the runtime-agnostic claim into something measured, and the manifest now records which engine
executes an artifact so the wrong one refuses it instead of failing somewhere inside a session.
Inference runs on a worker isolate rather than on the thread that draws.

**Shipped (`0.7.0`).** Tier 8. A tensor spec records which axes are spatial, so `resize` and
`center_crop` are generated instead of refused, and the generated crop rounds the way torchvision
does rather than the way Dart does, which on an odd margin was one row of the picture. Five failures
carry five distinct messages and each says what to do about it, with drift deliberately not among
them because it is a measurement rather than an error. VoltaCast is exported, measured and documented.
The documentation is a [site](https://nacode-studios.github.io/Fluttorch/).

**Next.** Stabilisation. An API freeze and a deprecation policy, tolerances replaced by measured ones,
and benchmarks, because parity is proven and cost is not.

**Later.** `1.0`, and after it the targets Flutter has that this runtime does not yet reach.

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

## Stability

See [STABILITY.md](STABILITY.md). Four things here carry a compatibility promise and only one of them
is a Dart API: the manifest is a document two implementations parse, the C header is an ABI three
bindings implement, the generated Dart gets committed by consumers, and the tolerances decide whether
a build is green. A deprecated API survives at least two minor releases.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Three rules shape everything else: the core stays
runtime-agnostic, the manifest is the single source of truth, and backend capabilities are reported
rather than assumed.

Measured cost of generating, loading and running a model is in
[`docs/benchmarks.md`](docs/benchmarks.md), with the tool that reproduces it.

Brand assets and design tokens live in [`docs/brand/`](docs/brand/).

## License

Apache-2.0. See [LICENSE](LICENSE).

## Sponsor

If Fluttorch is useful to you, consider [sponsoring NaCode Studios](https://github.com/sponsors/NaCode-Studios).
