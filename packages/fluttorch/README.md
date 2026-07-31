# fluttorch

The contract and the seam. This package holds what every other one agrees on: the manifest an export
emits, the tensor types that satisfy it, and the interface an inference backend implements.

It depends on no backend, and adding one must not require changing anything here. Drift metrics and
the parity gate live in [`fluttorch_test`](https://pub.dev/packages/fluttorch_test), because nothing
on the inference path needs them and an app shipping a model should not carry them.

```dart
final manifest = ManifestCodec.decode(await File(path).readAsString());

// Refuses an artifact the manifest was not written for. A manifest paired with
// the wrong weights satisfies every shape and returns every number wrong.
verifyArtifact(artifact: bytes, manifest: manifest);

final model = await runtime.load(artifact: bytes, manifest: manifest);
final out = await model.run([Tensor.view(spec: manifest.inputs.single, bytes: input)]);
```

Constructing a `Tensor` is the one place the central invariant is checked, that `bytes.length` equals
the element count times the element width, so a tensor that exists is one that agrees with its
declaration. The bytes are never copied.

You will not use this package alone. It is the dependency of
[`fluttorch_gen`](https://pub.dev/packages/fluttorch_gen), which turns a manifest into a typed API,
and of [`fluttorch_test`](https://pub.dev/packages/fluttorch_test), which replays the goldens and
fails the build when the numbers move. The manifests themselves come from `fluttorch-export`, the
Python side of the repository.

See the [repository README](https://github.com/NaCode-Studios/Fluttorch) for what the project is and
what it deliberately does not do.

## License

Apache-2.0.
