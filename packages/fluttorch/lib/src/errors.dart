import 'dtype.dart';

/// Base class for every failure Fluttorch raises deliberately.
///
/// The ways a model can fail to run are genuinely different problems with
/// different fixes: the artifact is wrong, the manifest is unreadable, the
/// buffer does not match its declaration, the device cannot do it, or the
/// numbers drifted. Sharing one exception type would collapse that distinction
/// at exactly the moment a caller needs it.
///
/// `docs/errors.md` is the taxonomy: five failures, what each one means, and
/// which of them is deliberately not an exception at all.
///
/// Every member carries a [remedy]. Saying what went wrong and not what to do
/// about it is the difference between a message a reader can act on and one
/// they can only paste into a search engine, and making it a field rather than
/// a convention means a subclass cannot quietly stop having one.
sealed class FluttorchException implements Exception {
  const FluttorchException(this.message);

  /// What went wrong, in the terms of this library's contract.
  final String message;

  /// What the reader should do about it. One sentence, imperative.
  ///
  /// Distinct from [message] on purpose: the cause and the fix are different
  /// sentences, and a caller showing this to a user usually wants only this one.
  String get remedy;

  @override
  String toString() => '$runtimeType: $message\n  → $remedy';
}

/// The artifact does not match the manifest it was loaded with.
///
/// Almost always a stale build: a model was re-exported and the manifest was
/// not, or the reverse. Left undetected it produces a green parity suite over
/// the wrong weights, which is the failure this check exists to prevent.
final class ArtifactMismatchException extends FluttorchException {
  const ArtifactMismatchException({
    required this.expectedHash,
    required this.actualHash,
  }) : super(
         'artifact hash $actualHash does not match the manifest, which '
         'declares $expectedHash — the model and its manifest come from '
         'different exports',
       );

  final String expectedHash;
  final String actualHash;

  @override
  String get remedy =>
      'Re-export the model and its manifest together. Editing either one by '
      'hand to agree with the other hides the mismatch instead of fixing it.';
}

/// A bundle arrived without a file its artifact cannot be loaded without.
///
/// Above a size the exporting toolchain decides for itself, the weights leave
/// the graph and land beside it. Both files then travel together or neither is
/// usable, and the one that goes missing is the one carrying the numbers: the
/// graph on its own still parses, still declares the right shapes, and still
/// runs, which is why this is refused loudly rather than discovered in the
/// outputs.
///
/// Usually an asset bundle or a deployment step that copied the artifact and
/// not what sits next to it.
final class BundlePartMissingException extends FluttorchException {
  BundlePartMissingException({required this.missing, required this.model})
    : super(
        missing.length == 1
            ? 'the bundle for "$model" is missing "${missing.single}", which '
                  'the artifact references and cannot be loaded without'
            : 'the bundle for "$model" is missing ${missing.length} files its '
                  'artifact references and cannot be loaded without: '
                  '${missing.join(", ")}',
      );

  /// Names of the parts that did not arrive, as the manifest declares them.
  final List<String> missing;

  /// Model the bundle belongs to, so the message names one bundle among many.
  final String model;

  @override
  String get remedy =>
      'Ship every file the export wrote beside the manifest, not the artifact '
      'alone. The manifest lists them under "parts".';
}

/// The manifest could not be read.
///
/// [field] names where the decoder stopped, so a hand-edited manifest can be
/// fixed without bisecting it.
final class ManifestFormatException extends FluttorchException {
  const ManifestFormatException(super.message, {this.field});

  /// Dotted path of the offending field, e.g. `inputs[1].dtype`.
  final String? field;

  @override
  String get remedy => field == null
      ? 'Re-export rather than repairing the document: a manifest is written '
            'by the exporter and read here, and nothing else should author one.'
      : 'Look at $field in the manifest, and re-export rather than editing it '
            'in place.';

  @override
  String toString() => field == null
      ? 'ManifestFormatException: $message\n  → $remedy'
      : 'ManifestFormatException at $field: $message\n  → $remedy';
}

/// The manifest was written by a newer schema than this build understands.
///
/// Separate from [ManifestFormatException] because the fix is different: the
/// manifest is not malformed, the reader is old.
final class ManifestVersionException extends FluttorchException {
  const ManifestVersionException({required this.found, required this.supported})
    : super(
        'manifest schema version $found is newer than the supported '
        'version $supported — upgrade the reader rather than editing the '
        'manifest',
      );

  final int found;
  final int supported;

  @override
  String get remedy =>
      'Upgrade the fluttorch package to one that understands schema $found. '
      'The manifest is not wrong, this reader is older than it.';
}

/// A buffer does not satisfy the tensor spec it was handed with.
final class TensorShapeException extends FluttorchException {
  const TensorShapeException(super.message, {this.tensorName});

  /// Tensor this concerns, when it is known.
  final String? tensorName;

  @override
  String get remedy =>
      'Size the buffer from the spec the export declared rather than from the '
      'shape the data happens to have.';

  @override
  String toString() => tensorName == null
      ? 'TensorShapeException: $message\n  → $remedy'
      : 'TensorShapeException on "$tensorName": $message\n  → $remedy';
}

/// A tensor was read as a type it does not hold.
final class DTypeMismatchException extends FluttorchException {
  DTypeMismatchException({
    required this.tensorName,
    required this.declared,
    required this.requested,
  }) : super(
         'tensor "$tensorName" holds ${declared.wireName} and was read as '
         '${requested.wireName}',
       );

  final String tensorName;
  final DType declared;
  final DType requested;

  @override
  String get remedy =>
      'Read it as ${declared.wireName}, which is what the export wrote. '
      'Reinterpreting the bytes as ${requested.wireName} yields numbers rather '
      'than an error.';
}

/// The backend cannot execute a tensor of this element type.
///
/// Distinct from [DTypeMismatchException], which is a caller reading bytes as
/// the wrong type. Nothing is being misread here: the manifest and the buffer
/// agree, and the device simply has no kernel for that type. The two were one
/// class until the runtimes reported this case as "holds float16 and was read
/// as float32", which describes a bug the caller does not have and sends them
/// looking at code that is correct.
final class DTypeUnsupportedException extends FluttorchException {
  DTypeUnsupportedException({
    required this.tensorName,
    required this.dtype,
    required this.backend,
    required this.supported,
  }) : super(
         'backend "$backend" cannot execute ${dtype.wireName}, which is what '
         '"$tensorName" declares; it handles '
         '${supported.map((d) => d.wireName).join(", ")}',
       );

  /// The tensor whose type the backend cannot carry.
  final String tensorName;

  /// The type the manifest declares for it.
  final DType dtype;

  /// The backend that was asked.
  final String backend;

  /// Every type this backend does handle.
  final List<DType> supported;

  @override
  String get remedy =>
      'Export for a backend that handles ${dtype.wireName}, or re-export the '
      'model at a type this one carries. Casting the tensor here would run and '
      'would not be the model that was measured.';
}

/// The runtime failed while doing something, and the bundle is not at fault.
///
/// The manifest parsed, the artifact matched its hash, and in most cases the
/// model had already loaded. What failed is the engine, and the useful next
/// step is the engine's own issue tracker rather than a re-export.
///
/// [status] is the shim's own code and [code] is whatever the runtime reported
/// underneath it. Both are carried as numbers rather than flattened into the
/// message, because those are the two values that mean something to somebody
/// reading the runtime's source.
///
/// This exists because the shims raised [ManifestFormatException] here, whose
/// remedy is to re-export. On a bundle that is provably correct that advice
/// sends a reader to rebuild something that was never wrong, and the real cause
/// stays unexamined.
final class RuntimeExecutionException extends FluttorchException {
  RuntimeExecutionException({
    required this.runtime,
    required this.operation,
    required this.status,
    this.code,
    this.detail,
  }) : super(
         'the $runtime runtime failed while $operation: status $status'
         '${code == null ? "" : ", error $code"}'
         '${detail == null ? "" : ", $detail"}',
       );

  /// The engine that failed: `executorch`, `onnx` or `litert`.
  final String runtime;

  /// What was being done, in the caller's terms: "running inference".
  final String operation;

  /// The shim's status code.
  final int status;

  /// The runtime's own error code, where it reported one.
  final int? code;

  /// Whatever text the runtime attached.
  final String? detail;

  @override
  String get remedy =>
      'Take status $status${code == null ? "" : " and error $code"} to the '
      '$runtime runtime, not to the export. The manifest parsed and the hash '
      'matched, so re-exporting would rebuild something that is already right.';
}

/// The device cannot do what was asked, and no fallback applies.
///
/// Carries what was requested and what is actually available, because the
/// useful next step is choosing from the second list.
final class BackendUnavailableException extends FluttorchException {
  BackendUnavailableException({
    required this.requested,
    required this.available,
  }) : super(
         available.isEmpty
             ? 'backend "$requested" is unavailable and this device reports '
                   'no usable backend at all'
             : 'backend "$requested" is unavailable; this device offers '
                   '${available.join(", ")}',
       );

  final String requested;
  final List<String> available;

  @override
  String get remedy => available.isEmpty
      ? 'Check that the native library was built and linked. A device offering '
            'no backend at all usually means the binding never loaded.'
      : 'Load with one of ${available.join(", ")}, or ask the runtime for its '
            'capabilities first and choose from what it reports.';
}

/// A capability the call depends on is absent on this backend.
///
/// Distinct from [BackendUnavailableException]: the backend loaded and runs,
/// it simply cannot do this particular thing. Per-layer drift attribution on a
/// backend without activation taps is the case that matters.
final class CapabilityUnavailableException extends FluttorchException {
  CapabilityUnavailableException({
    required this.backend,
    required this.capability,
  }) : super(
         'backend "$backend" does not support $capability; ask the model for '
         'its capabilities and degrade rather than assuming it does',
       );

  final String backend;
  final String capability;

  @override
  String get remedy =>
      'Ask the model for its capabilities and degrade, rather than assuming '
      'every backend can do this. A backend that cannot is not a broken one.';
}
