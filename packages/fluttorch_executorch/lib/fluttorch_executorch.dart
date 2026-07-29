/// ExecuTorch backend for Fluttorch.
///
/// First implementation of [FluttorchRuntime]. Phase one delegates to the
/// existing `executorch_flutter` bindings to get end-to-end quickly; the
/// activation taps and deterministic execution the parity gate needs are not
/// reachable through them, so this package is expected to grow its own
/// `dart:ffi` binding once that requirement is pinned down.
library;

import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';

class ExecuTorchRuntime implements FluttorchRuntime {
  @override
  Future<List<RuntimeCapabilities>> capabilities() =>
      throw UnimplementedError('see the roadmap: end-to-end spike');

  @override
  Future<LoadedModel> load({
    required Uint8List artifact,
    required ModelManifest manifest,
    String? backend,
  }) =>
      throw UnimplementedError('see the roadmap: end-to-end spike');
}
