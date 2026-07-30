/// The parity gate.
///
/// Replays the goldens captured at export time against a loaded model, measures
/// how far the on-device numbers moved, and fails the build when they moved too
/// far. This is where the tolerance semantics and the drift metrics live: nothing
/// on the inference path needs them, so an app shipping a model does not carry
/// them.
library;

import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';

import 'src/drift.dart';
import 'src/tolerance.dart';

export 'src/drift.dart';
export 'src/tolerance.dart';

/// The reference inputs and outputs captured at export time.
///
/// The manifest names its goldens by opaque key; this resolves those keys to
/// tensors. Implementations decide where they come from — a directory, a bundled
/// asset archive, or memory in a unit test — which is why the core does not
/// mention files.
abstract interface class GoldenBundle {
  /// Cases in the bundle, in the order the exporter captured them.
  List<GoldenCase> get cases;

  /// Resolves one key to a tensor satisfying [spec].
  ///
  /// Throws [TensorShapeException] if the stored bytes do not satisfy the spec,
  /// which means the bundle and the manifest came from different exports.
  Future<Tensor> tensor(String key, TensorSpec spec);
}

/// Replays every golden against [model] and returns one report per case.
///
/// Compares final outputs. When the backend reports activation taps, also
/// attributes the earliest divergent layer; when it does not, the report says so
/// rather than leaving the absence of an attribution to be read as agreement.
///
/// [tolerance] defaults to [Tolerance.startingPointFor] on the manifest's
/// quantization recipe. That returns null for a recipe this build does not know,
/// in which case a tolerance must be supplied — there is no default worth
/// guessing for a scheme nobody has measured.
Future<List<DriftReport>> measureParity(
  LoadedModel model, {
  required GoldenBundle goldens,
  Tolerance? tolerance,
  bool attributeLayers = true,
}) {
  throw UnimplementedError('M13 · golden replay runner');
}

/// Asserts that every golden passes, failing with a report that names the
/// tensor, the drift, the tolerance and the backend that produced it.
Future<void> expectParity(
  LoadedModel model, {
  required GoldenBundle goldens,
  Tolerance? tolerance,
}) {
  throw UnimplementedError('M15 · expectParity matcher');
}

/// Runs the same goldens across every backend the device offers.
///
/// A model that is correct on one backend and wrong on another shows up as
/// exactly that, which a single-backend run cannot reveal.
Future<Map<String, List<DriftReport>>> parityMatrix(
  FluttorchRuntime runtime, {
  required ModelArtifact artifact,
  required GoldenBundle goldens,
  Tolerance? tolerance,
}) {
  throw UnimplementedError('M24 · parity matrix across backends');
}

/// A model artifact together with the manifest it was exported with.
///
/// Paired deliberately: loading one without the other is what the weight hash
/// exists to catch, and the matrix has to reload the same artifact once per
/// backend.
final class ModelArtifact {
  const ModelArtifact({required this.bytes, required this.manifest});

  /// The runtime artifact as exported.
  final Uint8List bytes;

  /// The contract it was exported with.
  final ModelManifest manifest;
}
