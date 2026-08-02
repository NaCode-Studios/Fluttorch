# Changelog

This file records what changed in `fluttorch`, the contract package. The whole
project's record, covering every package and the Python exporter together, is in
[the repository changelog](https://github.com/NaCode-Studios/Fluttorch/blob/main/CHANGELOG.md).

## 1.0.0

- A manifest can name the parts an artifact references and cannot be loaded
  without, and `weightHash` covers them as well as the artifact. `BundlePart`,
  `bundleDigestOf` and a `parts` argument on `verifyArtifact` and on
  `FluttorchRuntime.load`. `BundlePartMissingException` names the file that did
  not arrive, because a graph loaded without the weights it references still
  parses and still answers.
- `currentSchemaVersion` rises to 2, and an export declares it only when it
  carries parts. Every other field added to this schema has been additive; this
  one is not, and the version is what stops an older reader loading structure
  with no numbers in it.
- The API is frozen. See
  [STABILITY.md](https://github.com/NaCode-Studios/Fluttorch/blob/main/STABILITY.md)
  for what that covers and how long a deprecated API survives.

## 0.7.0

- `TensorLayout`, so a spec can say which axes are spatial, and
  `DTypeUnsupportedException`, which separates a backend with no kernel for an
  element type from a caller reading bytes as the wrong one. Every exception
  carries a `remedy` beside its `message`.

## 0.6.0

- `TensorSpec` can record a layout, `nchw` or `nhwc`, so a consumer knows which
  axes are spatial. Absence is not a default: a spec that says nothing is not a
  spec that says NCHW, and asking one that does not say raises.
- Every `FluttorchException` carries a `remedy` beside its `message`, and
  `DTypeUnsupportedException` separates "this backend has no kernel for that
  element type" from "the caller read the bytes as the wrong type". The two
  were one class, and the runtimes reported the first case with the second's
  message.
- The manifest records which engine executes an artifact, and refuses to be
  loaded by another one.

## 0.5.0

- The runtime interface gained the four hooks the ExecuTorch binding implements:
  a backend pinned at load, repeatable execution, activation taps, and output
  buffers the caller owns.
- The manifest records the compute precision a delegate was lowered at.

## 0.4.0

- Quantization recipes and activation specs in the manifest.

## 0.3.0

- Drift metrics and the tolerance model the parity gate measures against.

## 0.2.0

- The manifest codec: one document, read identically by Python and Dart.

## 0.1.0

- Tensor specs, dtypes and the errors this package raises.

## 0.0.1

- First tag.
