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
[tolerance](https://nacode-studios.github.io/Fluttorch/tolerance.html) comes from,
[what each backend is verified to do](https://nacode-studios.github.io/Fluttorch/backend-coverage.html),
and [what each failure means](https://nacode-studios.github.io/Fluttorch/errors.html) including the
one that is deliberately not a failure at all.

> **Status — `1.0`, stable.** The API is frozen and [`STABILITY.md`](STABILITY.md) says what that
> covers. What `1.0` does not mean is that every engine runs every model: ExecuTorch lowers VoltaCast
> and then fails to execute it, identically under its own Python runtime, and the suite records it as
> a failing expectation, so the day upstream fixes it, the suite says so. The
> [board](https://github.com/orgs/NaCode-Studios/projects/6) is the plan of record.

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
  fluttorch: ^1.0.0

dev_dependencies:
  build_runner: ^2.4.0
  fluttorch_gen: ^1.0.0
  fluttorch_test: ^1.0.0
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

Every archive from `0.5.0` onward carries a [SLSA build provenance](https://slsa.dev/) attestation, so
you can check that what you resolved was built by this repository's workflow rather than uploaded by
someone else. `0.3.0` and `0.4.0` have none and cannot be given one: an attestation is signed against
the run that produced the archive.

```bash
gh attestation verify <archive> --repo NaCode-Studios/Fluttorch
```

## Usage

### Exporting a model

The exporter writes three things together, and together is the point: an artifact paired with a
manifest it was not generated from satisfies every shape and returns every number wrong.

```bash
fluttorch-export \
  --model      my_project.models:build_classifier \
  --example-inputs my_project.models:example_batch \
  --goldens    my_project.models:validation_windows \
  --out        build/classifier \
  --input-names  image \
  --output-names logits \
  --quantize   int8-dynamic
```

`--goldens` is what separates coverage from a smoke test. Without it the exporter captures the example
input alone and says so, because inventing a distribution it has no way to know would be worse than
admitting there is one case.

Name the tensors. Without `--input-names` the accessors are positional, and a generated getter called
`input_0` is one nobody wants to read.

### The generated API

`fluttorch_gen` is a `build_runner` builder: it turns `build/classifier/classifier.fluttorch.json`
into Dart beside it. Commit the result. Your CI regenerates and diffs it, which is what catches a
manifest that moved without the code being rebuilt.

```dart
final classifier = await Classifier.load(runtime, artifact: bytes);
final out = await classifier.run(image: ClassifierImage(pixels));
final scores = out.logits.values;         // Float32List, over the same memory
```

Three things there are compile errors rather than runtime surprises. Passing the wrong tensor, because
each one is its own extension type. Passing them in the wrong order, because `run` takes named
arguments. And reading an output that does not exist.

The wrapper costs nothing at run time: an extension type is the underlying `Tensor` after compilation.
A wrong length is still caught, at construction, and names the tensor:

```dart
ClassifierImage(Float32List(100));
// TensorShapeException on "image": expected 3072 values, got 100
```

### The parity gate

The goldens captured at export replay in your suite. This is the part nothing else does.

```dart
test('the quantized model still agrees with the one we evaluated', () async {
  final goldens = await DirectoryGoldenBundle.open(
    'build/classifier/classifier.fluttorch.json',
  );
  final model = await runtime.load(
    artifact: await File('build/classifier/classifier.pte').readAsBytes(),
    manifest: goldens.manifest,
  );

  await expectParity(model, goldens: goldens);
});
```

The tolerance comes from the recipe and the precision the manifest recorded, so an `int8-dynamic`
model is not held to the bound a full-precision one answers to. Pass your own when you have measured
it, which is the honest thing to do for a model whose activation ranges are nothing like the ones
these defaults were measured on:

```dart
await expectParity(model, goldens: goldens, tolerance: Tolerance(
  maxAbsolute: 2e-3,
  maxRelative: 8e-3,
  minCosine: 0.9995,
));
```

A report names the tensor, the bound it broke, the element that broke it and the backend that produced
the number, and it says all of that whether the answer is pass or fail.
[What each failure means](https://nacode-studios.github.io/Fluttorch/errors.html) reads one, and
argues why drift is a measurement here rather than an exception.

### Choosing a runtime

`FluttorchRuntime` is the only seam a backend touches. Nothing above it knows which engine is running
the model.

```dart
final runtime = ExecuTorchRuntime(NativeExecuTorchBindings.open());
// or
final runtime = OnnxRuntime.open();
final runtime = LiteRtRuntime.open();
```

The manifest records which engine an artifact was lowered for, so handing a `.pte` to ONNX Runtime is
refused at load rather than failing somewhere inside a session. That refusal is worth having: the
weight hash cannot catch it, because it is computed over whichever artifact was written.

Capabilities are asked for, never assumed:

```dart
if (model.capabilities.supportsActivationTaps) { ... }
if (model.capabilities.supportsDeterministicExecution) { ... }
```

Whether a device can expose intermediates or promise a repeatable reduction order is a property of the
hardware. Asking for determinism you cannot have throws rather than quietly running without it,
because a tolerance chosen against a promise that was silently dropped is a tolerance measuring noise.

### Preprocessing, generated rather than rewritten

Preprocessing is where the two-language problem bites hardest: written once in Python for training and
again in Dart for serving, the copies drift at the first refactor. The manifest records the steps and
the generator emits them.

```dart
final image = ClassifierImage.fromSource(frame, height: 480, width: 640);
// frame is the Float32List the camera gave you, at whatever size it gave it
```

The resize, the centre crop and the normalize come from what training recorded, including the rounding
convention. A centre crop with an odd margin lands on one row or the row above it, and both produce a
picture, which is why that one is generated rather than trusted to a second implementation.

A step this build cannot perform is named and refused rather than approximated with the nearest one it
has.

### When the weights do not fit in the graph

Above a size the exporting toolchain decides for itself, the weights leave the graph and land beside
it. The manifest names those parts and the weight hash covers them, so the pairing still covers the
numbers rather than only the structure.

```dart
final goldens = await DirectoryGoldenBundle.open('build/big/big.fluttorch.json');
final model = await runtime.load(
  artifact: await File('build/big/big.onnx').readAsBytes(),
  manifest: goldens.manifest,
  parts: await goldens.parts(),
);
```

Omit `parts` and the load throws `BundlePartMissingException` naming the file that did not arrive.
That refusal is the feature: a graph without the weights it references still parses, still declares
every shape the manifest promised, still runs, and answers from nothing.

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

**Shipped (`1.0.0`).** Export, generated API and parity replay work end to end over three backends
that share one C ABI, under tolerances measured on two models rather than assumed.
[`CHANGELOG.md`](CHANGELOG.md) has what landed in each release.

**Next.** The targets Flutter has that this runtime does not yet reach: the web backend, a signed
manifest, and parity proven where cost is not.

The [board](https://github.com/orgs/NaCode-Studios/projects/6) carries the milestone plan and its
exit criteria, and every tier is a [milestone](https://github.com/NaCode-Studios/Fluttorch/milestones)
in this repository.

## Building and testing

```bash
dart pub get
dart analyze --fatal-warnings packages/
dart format --output=none --set-exit-if-changed packages/
for p in fluttorch fluttorch_test fluttorch_gen; do (cd packages/$p && dart test); done
```

Scoped to `packages/` because `examples/` holds a Flutter app whose generated sources use Flutter-SDK
language features. Plain Dart tooling cannot parse it and reports that as a formatting failure, so it
has its own job.

```bash
pip install -e 'python/fluttorch_export[dev]'
ruff check python/ && ruff format --check python/
PYTHONPATH=python/fluttorch_export pytest python/fluttorch_export/tests
```

The manifest suite needs neither `torch` nor `executorch`. The export half does, and
[`examples/spike/`](examples/spike/) shows the whole path end to end.

The three bindings run against a real engine and are skipped everywhere the native library has not
been built, which includes CI. Each has its own `tool/build_native.sh`, and building the ExecuTorch
one takes about an hour: a suite that demanded it would be a suite nobody runs.

```bash
cd packages/fluttorch_executorch && tool/build_native.sh && dart test
```

Every published number is a command rather than a recollection. These are the ones behind
[the tolerance table](packages/fluttorch_test/lib/src/tolerance.dart) and
[`docs/benchmarks.md`](docs/benchmarks.md), and they run from `packages/fluttorch_executorch`:

```bash
dart run tool/parity_matrix.dart        # every backend this machine links, one report
dart run tool/measure_tolerances.dart   # what each recipe and precision actually costs
dart run tool/benchmark.dart            # codegen, load and per-inference cost
```

The fixtures they measure are written by `python/fluttorch_export/scripts/`, and each script's
docstring says which gap it exists to close.

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
