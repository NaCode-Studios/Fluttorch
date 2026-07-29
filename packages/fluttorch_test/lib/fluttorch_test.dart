/// The parity gate: replays the goldens captured at export time against a
/// loaded model and fails the test when the numerics drift.
library;

import 'package:fluttorch/fluttorch.dart';

/// Replays every golden in [model]'s manifest and returns one report per case.
///
/// Compares final outputs. When the backend supports activation taps, also
/// attributes the earliest divergent layer so a failure points at an op rather
/// than at the model as a whole.
Future<List<DriftReport>> measureParity(
  LoadedModel model, {
  required GoldenBundle goldens,
  bool attributeLayers = true,
}) {
  throw UnimplementedError('see the roadmap: parity gate MVP');
}

/// Asserts that every golden passes [tolerance], failing with a report that
/// names the tensor, the drift, and where it first appeared.
Future<void> expectParity(
  LoadedModel model, {
  required GoldenBundle goldens,
  Tolerance tolerance = Tolerance.exact,
}) {
  throw UnimplementedError('see the roadmap: parity gate MVP');
}

/// Runs the same goldens across every backend available on the device and
/// returns the full matrix, so a model that is correct on XNNPACK but wrong on
/// Core ML shows up as such.
Future<Map<String, List<DriftReport>>> parityMatrix(
  FluttorchRuntime runtime, {
  required GoldenBundle goldens,
}) {
  throw UnimplementedError('see the roadmap: multi-backend parity matrix');
}

/// The reference inputs and outputs captured from the source model at export
/// time, resolved from disk or from bundled assets.
abstract interface class GoldenBundle {
  List<GoldenCase> get cases;
}
