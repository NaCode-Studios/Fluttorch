# Fluttorch

**Train in PyTorch. Ship to Flutter. Prove the numbers didn't change.**

Fluttorch is the toolchain between a PyTorch model and a Flutter app: it generates a
typed Dart API from your exported model, and it fails your CI when the on-device
model stops agreeing with the reference.

> **Status: pre-alpha.** The public API does not exist yet. Nothing here is usable.
> Follow the [roadmap](https://github.com/orgs/NaCode-Studios/projects) for progress.

## The problem

Shipping a model to a phone means exporting and quantizing it. Both steps change the
numbers, and nothing tells you when they change too much:

- The build succeeds. The app runs. The output *looks* plausible. The model is wrong.
- Quantizing to int8 shifts activations. Mixed precision reorders `cast → multiply → sum`
  and least-significant bits accumulate over thousands of multiplies.
- Preprocessing is written twice — once in Python for training, once in Dart for
  serving — and the two drift apart on the first refactor. This is the single most
  common class of on-device ML bug, and it is invisible.

Today the recommended practice is to compare offline, online, and post-serialization
predictions *by hand*. There is no gate. In Flutter there is not even a convention.

## What Fluttorch does

**1. Export with a contract.** A Python CLI wraps `torch.export`, emits the runtime
artifact, and alongside it a signed **manifest**: input and output shapes, dtypes,
normalization constants, class labels, weight hash, and a set of golden input/output
pairs captured from the reference model.

**2. Typed bindings, generated.** A `build_runner` builder turns that manifest into a
typed Dart class. Shapes and dtypes are checked at compile time, and preprocessing is
generated from the same source of truth the model was trained with — so it cannot drift.

```dart
// generated from the manifest — not hand-written
final forecast = await SolarForecast.load();
final out = await forecast.run(features: window);   // shape-checked
```

**3. A parity gate.** The goldens become Dart tests that run on the target device, per
backend, and assert the numerics within tolerance — reporting drift per output tensor,
and (later) per layer.

```
FAIL  parity/solar_forecast
      backend: xnnpack  quantization: int8-static
      output[0] "load_mw"  max |Δ| 0.118  >  tolerance 0.010
      first divergence: layer conv3
```

Silent degradation becomes a red build.

## Architecture

Runtime-agnostic by design. The parity gate and the codegen are the product; the
inference runtime is a replaceable backend.

```
python/fluttorch_export     torch.export → artifact + manifest + goldens
        │
        ▼
packages/fluttorch          manifest, tensor specs, drift metrics, runtime interface
        ├── fluttorch_gen         manifest → typed Dart API (build_runner)
        ├── fluttorch_test        golden runner, parity matchers, drift reports
        └── fluttorch_executorch  first backend (ExecuTorch)
```

Planned backends: ExecuTorch first, then LiteRT and ONNX Runtime behind the same
interface. A parity gate is worth the same on every runtime.

## Why this doesn't exist yet

Catching numerical drift requires knowing where it comes from — quantization schemes,
mixed precision, activation ranges. Generating a typed Dart API requires `build_runner`,
`dart:ffi`, and Flutter's plugin model. The two skill sets rarely overlap, so the layer
between them was never built.

## License

Apache-2.0. See [LICENSE](LICENSE).
