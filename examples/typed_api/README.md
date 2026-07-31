# Typed API example

What `fluttorch_gen` produces from a manifest, and what using it looks like.

`lib/two_layer.fluttorch.json` is the manifest the exporter wrote. Running the build turns it into
`lib/two_layer.fluttorch.dart`, which is committed here so the output can be read without running
anything.

```bash
dart run build_runner build --delete-conflicting-outputs
dart test
```

## What the generation buys

Each tensor becomes its own type, so the compiler rejects the wrong one:

```dart
final model = await TwoLayer.load(runtime, artifact: bytes);
final out = await model.run(features: TwoLayerFeatures(values));
print(out.score.values);
```

`model.run(features: TwoLayerScore(...))` does not compile, and neither does passing a bare `Tensor`.
The wrappers are extension types, so none of that costs anything at run time.

Shapes are checked where they can be. `TwoLayerFeatures(values)` refuses a list that is not exactly
four long, because the export fixed that dimension. A tensor with a dynamic dimension gets `withShape`
instead, which is deliberately wordier: its shape is the one thing the compiler cannot check for you.

`TwoLayer.load` verifies the artifact against the digest the manifest recorded, and refuses one that
does not match. That check is why nothing here takes a manifest as an argument: the generated class
already carries it.
