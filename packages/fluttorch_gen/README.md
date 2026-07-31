# fluttorch_gen

Turns the manifest an export emitted into a Dart API, so that no shape, dtype or normalization
constant is ever restated by hand on the device side.

Preprocessing written once in Python for training and again in Dart for serving drifts apart at the
first refactor, and nothing tells you when it does. Here it is generated from the same document the
model was exported with.

```yaml
dev_dependencies:
  build_runner: ^2.4.0
  fluttorch_gen: ^0.3.0
```

Put `two_layer.fluttorch.json` next to your Dart sources and run the builder:

```bash
dart run build_runner build --delete-conflicting-outputs
```

What comes back is a class per model and a type per tensor:

```dart
final model = await TwoLayer.load(runtime, artifact: bytes);
final out = await model.run(features: TwoLayerFeatures(values));
print(out.score.values);
```

`model.run(features: TwoLayerScore(...))` does not compile, and neither does passing a bare `Tensor`.
The wrappers are extension types, so none of that costs anything at run time. A constructor refuses a
list of the wrong length where the export fixed the dimension; where a dimension is dynamic there is
`withShape` instead, deliberately wordier, because its shape is the one thing the compiler cannot
check for you.

Two recorded preprocessing steps are refused rather than generated. `resize` and `center_crop` need
to know which axes are spatial, the manifest records no tensor layout, and NCHW and NHWC would each
produce a plausible and different answer. That is a gap in the schema rather than in the generator,
and guessing there would reintroduce exactly the skew this package removes.

See the [repository README](https://github.com/NaCode-Studios/Fluttorch) for the whole pipeline.

## License

Apache-2.0.
