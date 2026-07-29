import 'dart:typed_data';

import 'manifest.dart';
import 'tensor.dart';

/// What a backend can actually do on the current device.
///
/// Reported at runtime rather than assumed, because backend availability is a
/// property of the hardware, not of the build.
class RuntimeCapabilities {
  const RuntimeCapabilities({
    required this.backend,
    required this.dtypes,
    this.supportsActivationTaps = false,
    this.supportsDeterministicExecution = false,
  });

  /// e.g. `xnnpack`, `coreml`, `qnn`, `vulkan`, `metal`.
  final String backend;

  final Set<DType> dtypes;

  /// Whether intermediate activations can be read. Required for per-layer drift
  /// attribution; without it the parity gate can only compare final outputs.
  final bool supportsActivationTaps;

  /// Whether execution can be pinned so repeated runs are bit-identical.
  /// Without it, tolerances must absorb run-to-run noise.
  final bool supportsDeterministicExecution;
}

/// The seam between Fluttorch and whatever executes the model.
///
/// Implemented by `fluttorch_executorch` first; LiteRT and ONNX Runtime are
/// intended to sit behind the same interface. Nothing above this layer — codegen,
/// goldens, drift metrics — should need to know which runtime is in use.
abstract interface class FluttorchRuntime {
  /// Backends available on this device, in the order the runtime prefers them.
  Future<List<RuntimeCapabilities>> capabilities();

  /// Loads a model artifact, verifying it against [manifest].
  ///
  /// Throws if the artifact's content hash disagrees with
  /// [ModelManifest.weightHash], or if [backend] is unavailable.
  Future<LoadedModel> load({
    required Uint8List artifact,
    required ModelManifest manifest,
    String? backend,
  });
}

/// A model resident in memory and ready to run.
abstract interface class LoadedModel {
  ModelManifest get manifest;

  /// Backend this instance was actually loaded on, which may differ from the
  /// one requested if the runtime fell back.
  String get backend;

  /// Runs inference. Inputs must match [ModelManifest.inputs] in order,
  /// shape, and dtype; generated bindings guarantee this statically.
  Future<List<Uint8List>> run(List<Uint8List> inputs);

  /// Runs inference and additionally returns named intermediate activations.
  ///
  /// Throws [UnsupportedError] when the backend reports
  /// [RuntimeCapabilities.supportsActivationTaps] as false.
  Future<TappedRun> runWithTaps(List<Uint8List> inputs, {List<String>? layers});

  Future<void> dispose();
}

/// Outputs plus the intermediate activations captured during a run.
class TappedRun {
  const TappedRun({required this.outputs, required this.activations});

  final List<Uint8List> outputs;
  final Map<String, Uint8List> activations;
}
