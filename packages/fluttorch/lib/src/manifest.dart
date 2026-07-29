import 'tensor.dart';

/// The contract emitted alongside an exported model.
///
/// This is the single source of truth shared by three consumers: the code
/// generator (which turns it into a typed Dart API), the parity gate (which
/// replays its goldens), and the runtime (which validates what it loads.)
class ModelManifest {
  const ModelManifest({
    required this.name,
    required this.schemaVersion,
    required this.inputs,
    required this.outputs,
    required this.weightHash,
    this.quantization,
    this.preprocessing = const [],
    this.labels,
    this.goldens = const [],
  });

  /// Identifier used to name the generated Dart class.
  final String name;

  /// Version of the manifest format itself, so old artifacts stay readable.
  final int schemaVersion;

  final List<TensorSpec> inputs;
  final List<TensorSpec> outputs;

  /// Content hash of the exported weights. The runtime refuses to load an
  /// artifact whose hash disagrees with the manifest it was generated from.
  final String weightHash;

  /// Recipe applied at export time, e.g. `int8-static`, `int4-weight-only`.
  /// Null means the model was exported in full precision.
  final String? quantization;

  /// Input transforms captured from the training pipeline, emitted so the Dart
  /// side cannot drift from the Python side.
  final List<PreprocessingStep> preprocessing;

  /// Class labels, when the model is a classifier.
  final List<String>? labels;

  /// Reference input/output pairs captured from the source model.
  final List<GoldenCase> goldens;
}

/// One input transform, replayed identically in Dart.
class PreprocessingStep {
  const PreprocessingStep({required this.kind, this.params = const {}});

  /// e.g. `normalize`, `resize`, `center_crop`, `to_float`.
  final String kind;
  final Map<String, Object?> params;
}

/// A reference input and the output the source model produced for it.
class GoldenCase {
  const GoldenCase({
    required this.id,
    required this.inputPaths,
    required this.outputPaths,
    this.description,
  });

  final String id;

  /// Paths, relative to the golden bundle, of the raw input tensors.
  final List<String> inputPaths;

  /// Paths of the expected output tensors, as produced by the source model.
  final List<String> outputPaths;

  final String? description;
}
