/// How far an on-device output may stray from the reference before the parity
/// gate fails.
///
/// Absolute and relative bounds are checked per element; [minCosine] is checked
/// per tensor and catches structural divergence that elementwise bounds miss
/// when magnitudes are small.
class Tolerance {
  const Tolerance({this.maxAbsolute, this.maxRelative, this.minCosine});

  /// Sensible default for a full-precision export: tight, because nothing
  /// should have changed.
  static const exact = Tolerance(maxAbsolute: 1e-5);

  /// Starting point for int8 static quantization. Every model needs its own
  /// number — this is a placeholder that will fail loudly rather than silently.
  static const int8 = Tolerance(maxAbsolute: 1e-2, minCosine: 0.999);

  final double? maxAbsolute;
  final double? maxRelative;
  final double? minCosine;
}

/// Measured divergence for a single output tensor.
class TensorDrift {
  const TensorDrift({
    required this.tensorName,
    required this.maxAbsolute,
    required this.meanAbsolute,
    required this.cosine,
    this.worstIndex,
  });

  final String tensorName;
  final double maxAbsolute;
  final double meanAbsolute;
  final double cosine;

  /// Flat index of the element with the largest absolute difference, to make a
  /// failure locatable instead of merely reported.
  final int? worstIndex;

  bool satisfies(Tolerance t) {
    if (t.maxAbsolute != null && maxAbsolute > t.maxAbsolute!) return false;
    if (t.minCosine != null && cosine < t.minCosine!) return false;
    return true;
  }
}

/// The result of replaying one golden case on one backend.
class DriftReport {
  const DriftReport({
    required this.goldenId,
    required this.backend,
    required this.tensors,
    this.quantization,
    this.firstDivergentLayer,
  });

  final String goldenId;
  final String backend;
  final List<TensorDrift> tensors;
  final String? quantization;

  /// Earliest layer whose activations diverged, when the runtime exposes
  /// intermediate taps. Turns "the output is wrong" into "this op is wrong".
  final String? firstDivergentLayer;

  bool passes(Tolerance t) => tensors.every((d) => d.satisfies(t));
}
