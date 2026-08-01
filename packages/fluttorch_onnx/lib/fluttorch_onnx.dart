/// ONNX Runtime for Fluttorch.
///
/// A second engine behind the same seam. `src/fluttorch_onnx.cpp` implements the
/// C ABI that `fluttorch_executorch` also implements, so everything above it is
/// unchanged: the manifest is the contract, the goldens are the references, and
/// the parity gate measures this the way it measures anything else.
///
/// What differs is what a backend name means. Here it is an ONNX Runtime
/// execution provider rather than an ExecuTorch delegate, which is why the
/// manifest records the runtime and the backend as separate fields.
library;

export 'src/runtime.dart';
