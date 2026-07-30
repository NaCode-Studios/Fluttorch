/// How far an on-device output may stray from its reference before the parity
/// gate fails.
///
/// The three bounds answer different questions and none of them is sufficient
/// alone. An absolute bound is the only meaningful one near zero, where a
/// relative bound explodes. A relative bound is the only meaningful one on large
/// magnitudes, where a fixed epsilon is either vacuous or impossible. Cosine
/// catches the case both elementwise bounds miss: a tensor whose values are all
/// small enough to pass individually while pointing somewhere else entirely.
///
/// The elementwise bounds combine the way every numerical comparison library
/// combines them, rather than as two independent tests:
///
/// ```text
/// |actual - reference|  <=  maxAbsolute + maxRelative * |reference|
/// ```
///
/// Testing them separately would reject a large value for a rounding error in
/// its last bits, and accept a small value that changed sign.
final class Tolerance {
  /// Creates a tolerance from at least one bound.
  ///
  /// A tolerance with nothing configured would accept every output, which for a
  /// gate is the worst possible failure: it stays green while the model rots.
  /// That is a constructor error rather than a runtime surprise.
  Tolerance({this.maxAbsolute = 0, this.maxRelative = 0, this.minCosine})
    : assert(
        maxAbsolute > 0 || maxRelative > 0 || minCosine != null,
        'a tolerance with no bound accepts everything; set at least one',
      ),
      assert(maxAbsolute >= 0, 'maxAbsolute cannot be negative'),
      assert(maxRelative >= 0, 'maxRelative cannot be negative'),
      assert(
        minCosine == null || (minCosine >= -1 && minCosine <= 1),
        'cosine similarity lies in [-1, 1]',
      );

  /// Absolute term. Dominates near zero.
  final double maxAbsolute;

  /// Relative term, as a fraction of the reference magnitude. Dominates on
  /// large values.
  final double maxRelative;

  /// Minimum cosine similarity over the whole tensor, or null to leave the
  /// direction unconstrained.
  final double? minCosine;

  /// Whether one element is within bounds of its reference.
  ///
  /// A non-finite reference is only matched by the identical non-finite value:
  /// a NaN that appears where the reference had a number is the single most
  /// important thing a gate can catch, and arithmetic on it would swallow it.
  bool elementPasses({required double actual, required double reference}) {
    if (!reference.isFinite || !actual.isFinite) {
      return actual.isNaN ? reference.isNaN : (actual == reference);
    }
    final delta = (actual - reference).abs();
    return delta <= maxAbsolute + maxRelative * reference.abs();
  }

  /// The bound this element had to meet, for reporting a failure in the terms
  /// the tolerance was written in.
  double allowanceFor(double reference) =>
      maxAbsolute + maxRelative * reference.abs();

  /// A starting point for [recipe], or null when the recipe is unrecognised.
  ///
  /// **These are starting points, not validated thresholds.** No model has been
  /// measured against them yet, and the number that is right for a model is a
  /// property of that model's activation ranges, not of its quantization
  /// scheme. They exist so a first run fails informatively instead of requiring
  /// a guess before any evidence exists, and every one of them should be
  /// replaced by a measured value once M17 has run.
  ///
  /// Null for an unknown recipe is deliberate: inventing a default for a scheme
  /// this build has never seen would put a number on a gate that nobody chose.
  static Tolerance? startingPointFor(String? recipe) => switch (recipe) {
    // Full precision. The graph was re-ordered, not re-quantized, so anything
    // beyond accumulated float32 rounding is a real change.
    null => Tolerance(
      maxAbsolute: 1e-5,
      maxRelative: 1e-5,
      minCosine: 0.999999,
    ),

    // Weights quantized per tensor, activations left in float. The error is
    // bounded by the weight step and does not compound through activations.
    'int8-dynamic' => Tolerance(
      maxAbsolute: 5e-2,
      maxRelative: 5e-2,
      minCosine: 0.999,
    ),

    // Activations quantized too, against ranges observed during calibration, so
    // an input outside those ranges clips and the error compounds with depth.
    'int8-static' => Tolerance(
      maxAbsolute: 1e-1,
      maxRelative: 1e-1,
      minCosine: 0.998,
    ),

    // Sixteen levels per weight group. Loose by construction; cosine carries
    // most of the signal here because the elementwise bound has to be wide.
    'int4-weight-only' => Tolerance(
      maxAbsolute: 2e-1,
      maxRelative: 2e-1,
      minCosine: 0.995,
    ),

    _ => null,
  };

  /// Recipes [startingPointFor] recognises, for a diagnostic that can list them.
  static const Set<String> knownRecipes = {
    'int8-dynamic',
    'int8-static',
    'int4-weight-only',
  };

  @override
  String toString() {
    final parts = <String>[
      if (maxAbsolute > 0) 'atol ${_fmt(maxAbsolute)}',
      if (maxRelative > 0) 'rtol ${_fmt(maxRelative)}',
      if (minCosine != null) 'cos ≥ ${_fmt(minCosine!)}',
    ];
    return 'Tolerance(${parts.join(", ")})';
  }

  static String _fmt(double v) =>
      v == 0 ? '0' : (v.abs() < 1e-3 ? v.toStringAsExponential(0) : '$v');
}
