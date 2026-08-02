import 'dart:typed_data';

import 'dtype.dart';
import 'errors.dart';
import 'manifest.dart';
import 'tensor.dart';

/// What a backend can actually do on the device it is running on.
///
/// Reported rather than assumed: backend availability, activation taps and
/// determinism are properties of the hardware and the driver, not of the build.
/// Code that needs one of them asks, and degrades when the answer is no.
final class RuntimeCapabilities {
  const RuntimeCapabilities({
    required this.backend,
    required this.dtypes,
    this.supportsActivationTaps = false,
    this.supportsDeterministicExecution = false,
    this.maxTensorBytes,
  });

  /// Backend identifier, e.g. `xnnpack`, `coreml`, `qnn`, `vulkan`, `metal`.
  final String backend;

  /// Element types this backend accepts.
  final Set<DType> dtypes;

  /// Whether intermediate activations can be read.
  ///
  /// Per-layer drift attribution needs this; without it the parity gate can
  /// only compare final outputs, and says so in its report rather than
  /// pretending it looked deeper.
  final bool supportsActivationTaps;

  /// Whether execution can be pinned so repeated runs are bit-identical.
  ///
  /// Without it a tolerance has to absorb run-to-run noise, which raises the
  /// floor on the drift the gate can detect.
  final bool supportsDeterministicExecution;

  /// Largest single tensor this backend will accept, when the limit is known.
  ///
  /// Null means unreported, not unlimited. The failures that actually happen in
  /// the field are a decoder refusing a resolution or an allocation exceeding
  /// what the device has, and neither can be anticipated if there is nowhere to
  /// record the ceiling.
  final int? maxTensorBytes;

  /// Whether this backend can carry every type the manifest declares.
  bool supportsAllTypesIn(ModelManifest manifest) => [
    ...manifest.inputs,
    ...manifest.outputs,
  ].every((s) => dtypes.contains(s.dtype));

  @override
  String toString() =>
      'RuntimeCapabilities($backend, ${dtypes.length} dtypes, '
      'taps: $supportsActivationTaps, '
      'deterministic: $supportsDeterministicExecution)';
}

/// The seam between Fluttorch and whatever executes the model.
///
/// Implemented by `fluttorch_executorch` first, with LiteRT and ONNX Runtime
/// intended to sit behind the same interface. Nothing above this layer — the
/// generator, the goldens, the drift metrics — knows which runtime is in use,
/// and a change that would require it to know is a change to this interface.
abstract interface class FluttorchRuntime {
  /// Backends available on this device, in the order the runtime prefers them.
  ///
  /// May be empty, which is a legitimate answer on a device with no usable
  /// accelerator and must not be mistaken for a failure to enumerate.
  Future<List<RuntimeCapabilities>> capabilities();

  /// Loads an artifact, verifying it against [manifest].
  ///
  /// Throws [ArtifactMismatchException] if the artifact's content hash
  /// disagrees with [ModelManifest.weightHash], and
  /// [BackendUnavailableException] if [backend] was named and is not there.
  /// When [backend] is null the runtime picks its preferred available one, and
  /// [LoadedModel.backend] reports what it settled on.
  ///
  /// [deterministic] asks for execution repeatable enough that two runs of the
  /// same input agree. It throws [CapabilityUnavailableException] rather than
  /// falling back, because a tolerance chosen against a promise that was quietly
  /// dropped is a tolerance measuring run-to-run noise.
  /// [parts] carries the files the artifact references and cannot be loaded
  /// without, keyed by the name [ModelManifest.parts] declares them under.
  /// Empty for a model whose weights fit inside its graph, which is nearly all
  /// of them.
  ///
  /// Throws [BundlePartMissingException] when the manifest declares a part that
  /// did not arrive, and [CapabilityUnavailableException] when the runtime
  /// cannot resolve external data at all. Neither is a fallback: a graph loaded
  /// without the weights it references still parses, still declares every shape
  /// the manifest promised, and still answers.
  Future<LoadedModel> load({
    required Uint8List artifact,
    required ModelManifest manifest,
    String? backend,
    bool deterministic = false,
    Map<String, Uint8List> parts = const {},
  });
}

/// A model resident in memory and ready to run.
abstract interface class LoadedModel {
  /// The contract this model was loaded against.
  ModelManifest get manifest;

  /// Backend this instance actually runs on, which may differ from the one
  /// requested if the runtime fell back.
  String get backend;

  /// What this instance can do.
  ///
  /// Per-instance rather than per-runtime: after a fallback, the capabilities
  /// of the backend that was asked for are not the capabilities in effect, and
  /// a report that names the wrong one is worse than no report.
  RuntimeCapabilities get capabilities;

  /// Runs inference, allocating the outputs.
  ///
  /// [inputs] must satisfy [ModelManifest.inputs] in order, name, type and
  /// shape; the seam checks it rather than trusting the caller, because the
  /// generated bindings are not the only caller. Throws
  /// [TensorShapeException] or [DTypeMismatchException] on a mismatch.
  ///
  /// Convenient, and it allocates on every call. Use [runInto] on a hot path.
  Future<List<Tensor>> run(List<Tensor> inputs);

  /// Runs inference into buffers the caller already owns.
  ///
  /// This is the path that makes repeated inference affordable: a 224×224×3
  /// `float32` input is 602 KB, and allocating and copying it on both sides of
  /// every call is the difference between usable and unusable at frame rate.
  /// [outputs] must satisfy [ModelManifest.outputs] and are written in place.
  Future<void> runInto({
    required List<Tensor> inputs,
    required List<Tensor> outputs,
  });

  /// Runs inference and captures intermediate activations.
  ///
  /// Throws [CapabilityUnavailableException] when
  /// [RuntimeCapabilities.supportsActivationTaps] is false on [capabilities].
  /// A null [layers] requests every tap the backend offers, which on a large
  /// model is a great deal of memory — name the layers when the point is one
  /// suspect op.
  Future<TappedRun> runWithTaps(List<Tensor> inputs, {List<String>? layers});

  /// Releases the native resources behind this model.
  ///
  /// Calling it twice is not an error; using the model afterwards is.
  Future<void> dispose();
}

/// Outputs plus the intermediate activations captured during a run.
final class TappedRun {
  const TappedRun({required this.outputs, required this.activations});

  final List<Tensor> outputs;

  /// Activations by layer name, in the order the graph produces them.
  ///
  /// A layer that was requested and is absent means the backend does not tap
  /// it, which the caller should report rather than treat as agreement.
  final Map<String, Tensor> activations;

  @override
  String toString() =>
      'TappedRun(${outputs.length} outputs, '
      '${activations.length} activations)';
}
