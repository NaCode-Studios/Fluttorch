/// LiteRT for Fluttorch.
///
/// A third engine behind the same seam. `src/fluttorch_litert.cpp` implements
/// the C ABI that `fluttorch_executorch` and `fluttorch_onnx` also implement, so
/// everything above it is unchanged: the manifest is the contract, the goldens
/// are the references, and the parity gate measures this the way it measures
/// anything else.
///
/// What differs is what a backend name means. Here it is a LiteRT accelerator
/// rather than an ExecuTorch delegate or an ONNX Runtime provider, which is why
/// the manifest records the runtime and the backend as separate fields.
library;

export 'src/runtime.dart';
