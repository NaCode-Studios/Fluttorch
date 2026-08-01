/// ExecuTorch backend for Fluttorch.
///
/// The parity gate needs four things no published Dart binding to ExecuTorch
/// exposes: a backend pinned at load, execution repeatable enough that a
/// tolerance measures the model rather than the noise, intermediate activations,
/// and output buffers the caller owns. This package is being written for those,
/// not for ownership.
///
/// The boundary with the native library is `src/fluttorch_executorch.h`, mirrored
/// in Dart by [ExecuTorchBindings]. Everything above that seam is ordinary Dart
/// and is tested as such; below it is the one part that must be compiled against
/// ExecuTorch.
library;

export 'src/bindings.dart';
// The implementation over a real library, which every consumer needs and none
// could reach: it lived under src/ and the suites here imported it by path,
// which is a thing a test in this repository can do and an application cannot.
export 'src/ffi.dart' show FtStatus, NativeExecuTorchBindings;
export 'src/isolate_runtime.dart';
export 'src/runtime.dart';
