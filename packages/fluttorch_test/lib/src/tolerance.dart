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

  /// The bound for [recipe] at [precision], or null when either is unrecognised.
  ///
  /// Every number these return, apart from `int4-weight-only`, is derived from
  /// a measurement rather than chosen. The model is `testdata/matrix`: a
  /// convolutional network with a foldable `BatchNorm2d`, a `GroupNorm` that
  /// reduces at run time, and a softmax, over eight goldens. It exists because
  /// the only model this gate had ever run on was two linear layers, which has
  /// nowhere for two delegates to disagree.
  ///
  /// `packages/fluttorch_executorch/tool/measure_tolerances.dart` reproduces
  /// the measurements, and each entry below records what it saw. Re-run it when
  /// a bound is questioned rather than arguing from the number.
  ///
  /// The bounds sit above what was measured on purpose. One machine, one model
  /// and one input distribution is evidence about that combination, and a bound
  /// set exactly at the observed worst case fails the first build that differs
  /// in any of the three. Each entry says how much room it leaves and why.
  ///
  /// Null for an unknown recipe is deliberate: inventing a default for a scheme
  /// this build has never seen would put a number on a gate that nobody chose.
  ///
  /// A recipe and a precision compound and are answered together. A recipe says
  /// how the weights were stored and a precision says what the delegate does
  /// arithmetic in, so an int8 model on a half-precision GPU is wrong in both
  /// ways at once. What comes back is the looser of the two bounds on each
  /// term, because a bound tight enough for one of them fails a model that is
  /// only doing the other.
  static Tolerance? boundFor(String? recipe, {String? precision}) {
    final fromRecipe = _forRecipe(recipe);
    if (fromRecipe == null) return null;
    if (precision == null || precision == 'float32') return fromRecipe;

    final fromPrecision = _forPrecision(precision);
    if (fromPrecision == null) return null;
    return Tolerance(
      maxAbsolute: fromRecipe.maxAbsolute > fromPrecision.maxAbsolute
          ? fromRecipe.maxAbsolute
          : fromPrecision.maxAbsolute,
      maxRelative: fromRecipe.maxRelative > fromPrecision.maxRelative
          ? fromRecipe.maxRelative
          : fromPrecision.maxRelative,
      minCosine: switch ((fromRecipe.minCosine, fromPrecision.minCosine)) {
        (null, final b) => b,
        (final a, null) => a,
        (final a!, final b!) => a < b ? a : b,
      },
    );
  }

  /// What half precision costs on its own, before any recipe.
  ///
  /// Measured on `testdata/matrix`, whose Core ML and MPS exports both lower to
  /// float16: worst absolute `2.9e-4`, worst relative `1.2e-3`, worst
  /// `1 - cosine` `6.6e-8`. Core ML is the wider of the two by roughly five
  /// times, which is itself worth knowing: two delegates at the same declared
  /// precision are not the same delegate.
  ///
  /// What half precision costs on its own, before any recipe.
  ///
  /// Confirmed at `2e-2`, and the two models are why. `testdata/matrix` on Core
  /// ML shows `1.2e-3` relative and on MPS `2.2e-4`. `testdata/coreml`, which
  /// is the two-layer model, shows `1.0e-2` on the same delegate: eight times
  /// wider, on a simpler network.
  ///
  /// That gap is the thing to understand before touching this number, because
  /// it is not noise and the wider model is not the more complex one. A gate
  /// only sees outputs, but rounding happens on intermediates. The two-layer
  /// model's outputs land near `9.4`, where the absolute error inherited from
  /// its intermediates is a large fraction of nothing in particular;
  /// `testdata/matrix` ends in a softmax that pins its outputs into `[0, 1]`,
  /// where the same rounding reads as a much smaller relative number. Neither
  /// delegate is misbehaving. The two magnitudes are simply not the same
  /// magnitude.
  ///
  /// So the bound is set by the model with the large outputs, at about twice
  /// what it produces. Narrowing to what `testdata/matrix` alone would justify
  /// was tried, at `5e-3`, and it failed `testdata/coreml` immediately: a
  /// bound derived from the better-behaved of two committed fixtures is not a
  /// measured bound, it is a measurement of the fixture that was chosen.
  ///
  /// `1 - cosine` came in at `1.3e-7` at worst against a bound of `1e-4`. It
  /// stays wide because cosine is a whole-tensor statistic: a regression in a
  /// few elements moves the elementwise terms long before it moves this one, so
  /// tightening it buys nothing the other two do not already catch.
  static Tolerance? _forPrecision(String precision) => switch (precision) {
    'float16' => Tolerance(
      maxAbsolute: 1e-3,
      maxRelative: 2e-2,
      minCosine: 0.9999,
    ),
    'float32' => _forRecipe(null),
    _ => null,
  };

  static Tolerance? _forRecipe(String? recipe) => switch (recipe) {
    // Full precision. The graph was re-ordered, not re-quantized, so anything
    // beyond accumulated float32 rounding is a real change.
    //
    // Measured worst relative: 2.2e-7 on testdata/matrix through xnnpack,
    // 6.1e-7 on the two-layer model through portable. Confirmed, with about
    // sixteen times of room on the tighter term.
    //
    // The room is deliberate. Both models are shallow, float32 rounding
    // accumulates with depth, and a bound cut to what four layers happen to
    // produce would fail a deeper model that is behaving correctly.
    null => Tolerance(
      maxAbsolute: 1e-5,
      maxRelative: 1e-5,
      minCosine: 0.999999,
    ),

    // Weights quantized per tensor, activations left in float. The error is
    // bounded by the weight step and does not compound through activations.
    //
    // Measured worst relative: 3.1e-3 on testdata/matrix, 4.1e-2 on the
    // two-layer model in testdata/quantized. Widened from 5e-2, which is not a
    // margin over 4.1e-2 but a coincidence: one committed fixture sat inside it
    // by a factor of 1.2, and a build on other hardware had every chance of
    // tipping it over. A gate that fails for a reason nobody can act on is
    // worse than one set where the evidence puts it.
    //
    // The two-layer model is the wide one here for the same reason it is under
    // float16: its outputs land near 9.4 and the relative error is measured
    // against them.
    'int8-dynamic' => Tolerance(
      maxAbsolute: 1e-1,
      maxRelative: 1e-1,
      minCosine: 0.999,
    ),

    // Activations quantized too, against ranges observed during calibration, so
    // an input outside those ranges clips and the error compounds with depth.
    //
    // Measured worst relative: 6.7e-3 on testdata/matrix, which is twice what
    // the dynamic recipe costs on the same model. That is the only measurement
    // there is: nothing exports this recipe for the two-layer model, so the
    // large-magnitude case that sets every other row here is unmeasured for
    // this one.
    //
    // Held at 1e-1 for that reason rather than narrowed to fit the one model
    // that was measured. The dynamic recipe reaches 4.1e-2 on the model this
    // one has never been run against, and static quantization is strictly the
    // coarser of the two.
    'int8-static' => Tolerance(
      maxAbsolute: 2e-1,
      maxRelative: 2e-1,
      minCosine: 0.998,
    ),

    // Sixteen levels per weight group. Loose by construction; cosine carries
    // most of the signal here because the elementwise bound has to be wide.
    //
    // The one entry here that is still a starting point, and it says so rather
    // than borrowing the credibility of the rows above it. Nothing in this
    // toolchain exports int4: `quantization.RECIPES` carries int8-dynamic and
    // int8-static and no more, so there is no artifact to measure. When an
    // exporter grows the recipe, measure it here before trusting this row.
    'int4-weight-only' => Tolerance(
      maxAbsolute: 5e-1,
      maxRelative: 5e-1,
      minCosine: 0.995,
    ),

    _ => null,
  };

  /// Precisions [boundFor] recognises, for a diagnostic that can list
  /// them. Float32 is included even though a manifest writes it as absence,
  /// because a caller passing it explicitly means the same thing.
  static const Set<String> knownPrecisions = {'float16', 'float32'};

  /// Recipes [boundFor] recognises, for a diagnostic that can list them.
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
