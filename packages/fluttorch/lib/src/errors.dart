import 'dtype.dart';

/// Base class for every failure Fluttorch raises deliberately.
///
/// The five ways a model can fail to run are genuinely different problems with
/// different fixes: the artifact is wrong, the manifest is unreadable, the
/// buffer does not match its declaration, the device cannot do it, or the
/// numbers drifted. Sharing one exception type would collapse that distinction
/// at exactly the moment a caller needs it.
sealed class FluttorchException implements Exception {
  const FluttorchException(this.message);

  final String message;

  @override
  String toString() => '$runtimeType: $message';
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
  String toString() => field == null
      ? 'ManifestFormatException: $message'
      : 'ManifestFormatException at $field: $message';
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
}

/// A buffer does not satisfy the tensor spec it was handed with.
final class TensorShapeException extends FluttorchException {
  const TensorShapeException(super.message, {this.tensorName});

  /// Tensor this concerns, when it is known.
  final String? tensorName;

  @override
  String toString() => tensorName == null
      ? 'TensorShapeException: $message'
      : 'TensorShapeException on "$tensorName": $message';
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
}
