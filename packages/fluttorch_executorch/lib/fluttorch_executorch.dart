/// ExecuTorch backend for Fluttorch.
///
/// The first implementation of [FluttorchRuntime]. Phase one delegates to the
/// existing `executorch_flutter` bindings to reach end to end quickly; the
/// activation taps, deterministic execution and backend pinning that per-layer
/// drift attribution needs are not reachable through them, so this package is
/// expected to grow its own `dart:ffi` binding once M1 has established exactly
/// which hooks are missing.
library;

import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';

/// Runs models exported by `fluttorch-export` through ExecuTorch.
final class ExecuTorchRuntime implements FluttorchRuntime {
  @override
  Future<List<RuntimeCapabilities>> capabilities() =>
      throw UnimplementedError('M2 · end-to-end spike');

  @override
  Future<LoadedModel> load({
    required Uint8List artifact,
    required ModelManifest manifest,
    String? backend,
  }) => throw UnimplementedError('M2 · end-to-end spike');
}
