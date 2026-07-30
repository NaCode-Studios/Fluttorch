/// The contract and the seam.
///
/// This package holds what every other one agrees on: the manifest an export
/// emits, the tensor types that satisfy it, and the interface an inference
/// backend implements. It depends on no backend, and adding one must not
/// require changing anything here.
///
/// Drift metrics and the parity gate live in `fluttorch_test`: nothing on the
/// inference path needs them, and an app shipping a model should not carry them.
library;

export 'src/dtype.dart';
export 'src/errors.dart';
export 'src/manifest.dart';
export 'src/manifest_codec.dart';
export 'src/runtime.dart';
export 'src/tensor.dart';
