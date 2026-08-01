/// Delivery, not API.
///
/// This package exists so that the native half of `fluttorch_executorch` is
/// inside the application when [NativeExecuTorchBindings.open] looks for it.
/// It adds nothing a caller invokes: the binding already resolves the library
/// the way each platform requires, by name on Android and out of the process
/// image on iOS, where an app cannot load an arbitrary dylib from its bundle.
///
/// Depend on it, and use `fluttorch_executorch` as you would anywhere else.
library;

export 'package:fluttorch_executorch/fluttorch_executorch.dart';
