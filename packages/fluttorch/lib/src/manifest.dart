import 'errors.dart';
import 'tensor.dart';

/// The contract emitted alongside an exported model.
///
/// One document, three readers: the generator turns it into a typed Dart API,
/// the parity gate replays its goldens, and the runtime validates what it loads
/// against it. Nothing downstream restates any of it, which is what makes the
/// Dart side unable to disagree with the training pipeline.
final class ModelManifest {
  const ModelManifest({
    required this.name,
    required this.schemaVersion,
    required this.inputs,
    required this.outputs,
    required this.weightHash,
    this.quantization,
    this.runtime,
    this.precision,
    this.preprocessing = const [],
    this.labels,
    this.goldens = const [],
    this.activations = const [],
    this.activationHandles = const [],
  });

  /// Schema version this build writes, and the highest it can read.
  static const int currentSchemaVersion = 1;

  /// Identifier used to name the generated Dart class.
  final String name;

  /// Version of the manifest format, so an artifact stays readable after the
  /// format moves on.
  final int schemaVersion;

  /// Model inputs, in the order the graph takes them.
  final List<TensorSpec> inputs;

  /// Model outputs, in the order the graph returns them.
  final List<TensorSpec> outputs;

  /// Content hash of the exported weights.
  ///
  /// The runtime refuses an artifact whose hash disagrees, because a manifest
  /// paired with the wrong weights produces a green parity suite over a model
  /// nobody evaluated.
  final String weightHash;

  /// Recipe applied at export, e.g. `int8-static`, `int4-weight-only`. Null
  /// means full precision.
  ///
  /// Deliberately a string: the exporter may grow recipes this reader has never
  /// heard of, and a closed enum would reject artifacts it could otherwise
  /// serve. Whoever needs to act on it decides what to do with an unfamiliar
  /// value, rather than the decoder deciding for them.
  final String? quantization;

  /// Input transforms captured from the training pipeline.
  final List<PreprocessingStep> preprocessing;

  /// Class labels, when the model is a classifier.
  final List<String>? labels;

  /// Reference input and output pairs captured from the source model.
  final List<GoldenCase> goldens;

  /// Intermediate tensors the export tapped, in the order the graph produces
  /// them.
  ///
  /// The order carries the meaning: attribution reports the earliest layer whose
  /// numbers moved, and earliest is only defined against a declared sequence.
  /// Empty is the ordinary case, and it costs exactly one thing: a failing gate
  /// can name the output that was wrong but not the layer that made it wrong.
  /// The engine that executes this artifact, when it is not ExecuTorch.
  ///
  /// An artifact is a `.pte` or a `.onnx` and the two are not interchangeable,
  /// so a reader that loads one as the other gets a parse error at best.
  /// Nothing in the bytes says which, and the weight hash cannot tell them
  /// apart because it is computed over whichever one was written.
  ///
  /// Distinct from the backend, which names a delegate or provider inside a
  /// runtime. XNNPACK is an ExecuTorch delegate and CoreMLExecutionProvider is
  /// an ONNX Runtime provider, and collapsing the two axes into one field is
  /// how a bundle describes a combination that does not exist.
  ///
  /// Null means ExecuTorch, which is what every manifest written before this
  /// field meant by saying nothing.
  final String? runtime;

  /// Compute precision the delegate was lowered at, when it is not float32.
  ///
  /// Separate from [quantization] because they are separate decisions that
  /// compound. A recipe says how the weights were stored; this says what the
  /// delegate does arithmetic in, and a backend can halve the second while
  /// leaving the first alone. Core ML and MPS both do that by default.
  ///
  /// Null means float32, which is what a reader without this field assumed
  /// anyway. What it fixes is the case where that absence was untrue: an
  /// artifact lowered at float16 answered to a full-precision bound it could not
  /// hold, and the gate failed a model that was doing what it was told.
  final String? precision;

  final List<TensorSpec> activations;

  /// Where each tap lives in the lowered graph, positional with [activations].
  ///
  /// A runtime reads an intermediate by the handle the artifact carries, since
  /// submodule names do not survive lowering. Only the export sees both, so it
  /// records the correspondence here rather than leaving each runtime to guess
  /// at one.
  ///
  /// Empty when the export could not observe the layers it declared, which is
  /// what a fully delegated graph is. A reader without handles can still compare
  /// the reference activations it was given, and simply cannot ask a device for
  /// its own.
  final List<int> activationHandles;

  /// Whether any preprocessing step is unrecognised by this build.
  ///
  /// The decoder keeps such steps rather than dropping them, so a consumer can
  /// tell the difference between "no transform" and "a transform I cannot
  /// perform". Silently skipping one is exactly the train/serve skew this
  /// document exists to prevent, so the generator must refuse a manifest where
  /// this is true.
  bool get hasUnknownPreprocessing =>
      preprocessing.any((s) => s is UnknownPreprocessingStep);

  /// Input named [name].
  ///
  /// Throws [TensorShapeException] if there is none, listing what there is —
  /// a typo in a generated accessor should say what was available.
  TensorSpec inputNamed(String name) => _named(inputs, name, 'input');

  /// Output named [name].
  TensorSpec outputNamed(String name) => _named(outputs, name, 'output');

  static TensorSpec _named(List<TensorSpec> specs, String name, String role) {
    for (final s in specs) {
      if (s.name == name) return s;
    }
    throw TensorShapeException(
      'no $role named "$name"; this model declares '
      '${specs.map((s) => '"${s.name}"').join(", ")}',
    );
  }

  @override
  String toString() =>
      'ModelManifest($name, schema $schemaVersion, '
      '${inputs.length} in, ${outputs.length} out, '
      '${quantization ?? "full precision"})';
}

/// One input transform, replayed in Dart exactly as training applied it.
///
/// Sealed so the generator's handling is exhaustive: adding a transform makes
/// every unhandled `switch` a compile error, instead of a surprise in somebody
/// else's app. [UnknownPreprocessingStep] is the one escape hatch and exists so
/// an old reader can still say what it cannot do.
sealed class PreprocessingStep {
  const PreprocessingStep();

  /// Name used in the manifest.
  String get kind;
}

/// Subtract [mean] and divide by [std], per channel.
final class NormalizeStep extends PreprocessingStep {
  const NormalizeStep({required this.mean, required this.std, this.axis = -1});

  /// Per-channel means. Length must match the channel extent.
  final List<double> mean;

  /// Per-channel standard deviations. No element may be zero.
  final List<double> std;

  /// Axis the channels lie on. `-1` is the innermost.
  final int axis;

  @override
  String get kind => 'normalize';
}

/// Multiply by [factor] then add [offset] — the `uint8` to unit-range step.
final class RescaleStep extends PreprocessingStep {
  const RescaleStep({required this.factor, this.offset = 0});

  final double factor;
  final double offset;

  @override
  String get kind => 'rescale';
}

/// Resize to [height] by [width].
final class ResizeStep extends PreprocessingStep {
  const ResizeStep({
    required this.height,
    required this.width,
    this.interpolation = 'bilinear',
  });

  final int height;
  final int width;

  /// Filter name as the training pipeline used it. Reproducing the exact filter
  /// matters: bilinear and nearest disagree enough to move a prediction.
  final String interpolation;

  @override
  String get kind => 'resize';
}

/// Crop [height] by [width] from the centre.
final class CenterCropStep extends PreprocessingStep {
  const CenterCropStep({required this.height, required this.width});

  final int height;
  final int width;

  @override
  String get kind => 'center_crop';
}

/// Cast to a different element type, without rescaling.
final class CastStep extends PreprocessingStep {
  const CastStep(this.target);

  final String target;

  @override
  String get kind => 'cast';
}

/// A step this build does not recognise, preserved verbatim.
///
/// Carrying it forward is the point: a consumer that cannot perform it must
/// fail loudly, and it can only do that if the step survived decoding.
final class UnknownPreprocessingStep extends PreprocessingStep {
  const UnknownPreprocessingStep({required this.kind, required this.params});

  @override
  final String kind;

  /// Raw parameters as they appeared in the manifest.
  final Map<String, Object?> params;
}

/// A reference input and the output the source model produced for it.
///
/// Inputs and outputs are named by opaque keys, not paths: the manifest does not
/// know whether a bundle lives on disk, inside an asset archive, or in memory,
/// and on the web there is no filesystem to name. Resolution belongs to whoever
/// loads the bundle.
final class GoldenCase {
  const GoldenCase({
    required this.id,
    required this.inputKeys,
    required this.outputKeys,
    this.description,
    this.activationKeys = const [],
  });

  /// Stable identifier, used in failure output so a case can be found again.
  final String id;

  /// Keys of the input tensors, positional with [ModelManifest.inputs].
  final List<String> inputKeys;

  /// Keys of the expected output tensors, positional with
  /// [ModelManifest.outputs], as produced by the source model before lowering.
  final List<String> outputKeys;

  /// What this case is for, when it was captured deliberately rather than
  /// sampled — an edge case is worth a sentence.
  final String? description;

  /// Keys of the reference activations, positional with
  /// [ModelManifest.activations].
  ///
  /// Empty unless the export captured taps. A case carries all of them or none:
  /// a partial set would attribute a drift to the earliest layer that happened
  /// to be captured, which reads identically to the earliest layer that diverged
  /// and is a different claim.
  final List<String> activationKeys;

  @override
  String toString() => 'GoldenCase($id)';
}
