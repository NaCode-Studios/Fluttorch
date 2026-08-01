import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';

/// The native seam, in Dart terms.
///
/// One mirror of `src/fluttorch_executorch.h`, and the reason the runtime above
/// it can be written and tested before any native library exists. It is also the
/// seam a second implementation would use: an isolate, a mock, or a different
/// engine behind the same four capabilities.
abstract interface class ExecuTorchBindings {
  /// Backends this build was compiled with, whether or not the device runs them.
  List<String> backends();

  /// What [backend] can do here, or the preferred one when null.
  ///
  /// Throws [BackendUnavailableException] when the name is not one of
  /// [backends].
  NativeCapabilities capabilitiesOf(String? backend);

  /// Loads an artifact.
  ///
  /// [deterministic] asks for repeatable execution and fails rather than
  /// silently running without it, because a tolerance chosen against a promise
  /// that was quietly dropped is a tolerance measuring noise.
  NativeModel load({
    required Uint8List artifact,
    String? backend,
    bool deterministic = false,
  });
}

/// A model held by the native side.
abstract interface class NativeModel {
  /// The backend actually running, which is not always the one requested.
  String get backend;

  /// What this instance can do, which is a property of the loaded pair rather
  /// than of the build.
  NativeCapabilities get capabilities;

  /// Runs inference, writing into [outputs], which the caller owns.
  void run(List<Tensor> inputs, List<Tensor> outputs);

  /// Runs inference and captures the named intermediates.
  ///
  /// A layer the graph does not carry is absent from the result rather than
  /// zero-filled: the gate has to tell a layer that agreed from a layer nobody
  /// looked at, and a zero-filled tensor reads as the first.
  Map<String, Tensor> runWithTaps(
    List<Tensor> inputs,
    List<Tensor> outputs,
    List<String> layers,
  );

  void dispose();
}

/// What a backend reports about itself.
final class NativeCapabilities {
  const NativeCapabilities({
    required this.backend,
    required this.dtypes,
    this.supportsTaps = false,
    this.supportsDeterminism = false,
    this.maxTensorBytes,
  });

  final String backend;
  final Set<DType> dtypes;
  final bool supportsTaps;
  final bool supportsDeterminism;

  /// Largest single tensor, or null when the limit is not known. Null is not
  /// "unlimited": the ceiling that matters is the one nobody reported.
  final int? maxTensorBytes;

  /// The dtype bitmask the C struct carries, widened to the enum.
  ///
  /// Position, not wire name: the mask is produced by a compiled artifact that
  /// cannot ask Dart what order the enum is in, so the order is the contract.
  static Set<DType> dtypesFromMask(int mask) => {
    for (var i = 0; i < DType.values.length; i++)
      if (mask & (1 << i) != 0) DType.values[i],
  };

  /// The inverse, so a test or a second implementation can produce a mask
  /// without restating the ordering.
  static int maskOf(Iterable<DType> types) =>
      types.fold(0, (m, d) => m | (1 << d.index));

  RuntimeCapabilities toRuntime() => RuntimeCapabilities(
    backend: backend,
    dtypes: dtypes,
    supportsActivationTaps: supportsTaps,
    supportsDeterministicExecution: supportsDeterminism,
    maxTensorBytes: maxTensorBytes,
  );
}
