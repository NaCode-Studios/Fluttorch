import 'dart:math' as math;

import 'package:fluttorch/fluttorch.dart';

import 'tolerance.dart';

/// Measured divergence of one output tensor from its reference.
///
/// A drift carries the [tolerance] it was measured against. Evaluating a
/// measurement under a tolerance it was not taken with is how a gate quietly
/// changes meaning, so [passes] takes no argument and there is no way to
/// reinterpret a report after the fact.
final class TensorDrift {
  const TensorDrift({
    required this.tensorName,
    required this.tolerance,
    required this.elementCount,
    required this.violatingElements,
    required this.maxAbsolute,
    required this.maxRelative,
    required this.meanAbsolute,
    required this.cosine,
    this.worstIndex,
    this.worstActual,
    this.worstReference,
  });

  final String tensorName;

  /// The bounds this was measured against.
  final Tolerance tolerance;

  final int elementCount;

  /// Elements that exceeded the combined elementwise bound.
  final int violatingElements;

  /// Largest absolute difference over all elements.
  final double maxAbsolute;

  /// Largest relative difference, over the elements whose reference is
  /// non-zero. Zero when every reference is zero, where relative error has no
  /// meaning rather than being infinite.
  final double maxRelative;

  final double meanAbsolute;

  /// Cosine similarity over the flattened tensors.
  ///
  /// Two all-zero tensors are identical, so their similarity is 1; one zero and
  /// one not have nothing in common, so theirs is 0.
  final double cosine;

  /// Flat index of the worst offending element, so a failure is locatable
  /// rather than merely reported.
  final int? worstIndex;

  /// Value found at [worstIndex].
  final double? worstActual;

  /// Value expected at [worstIndex].
  final double? worstReference;

  /// Whether every bound in [tolerance] was met.
  bool get passes {
    if (violatingElements > 0) return false;
    final minCosine = tolerance.minCosine;
    if (minCosine != null && cosine < minCosine) return false;
    return true;
  }

  /// Which bound failed, for a message that says what to change.
  String get failureReason {
    if (passes) return 'within tolerance';
    if (violatingElements > 0) {
      final pct = (violatingElements / elementCount * 100);
      return '$violatingElements of $elementCount elements '
          '(${pct.toStringAsFixed(1)}%) exceed the elementwise bound';
    }
    return 'cosine $cosine is below the required ${tolerance.minCosine}';
  }

  @override
  String toString() {
    final head =
        'output "$tensorName"  max |Δ| '
        '${maxAbsolute.toStringAsPrecision(3)}';
    if (passes) return '$head  ok';
    if (violatingElements == 0) {
      // Every element sat inside its own bound and the tensor still points
      // somewhere else. Printing the elementwise bound here would name the one
      // thing that did not fail.
      return 'output "$tensorName"  cosine '
          '${cosine.toStringAsPrecision(6)}  <  ${tolerance.minCosine} '
          'required  (max |Δ| ${maxAbsolute.toStringAsPrecision(3)})';
    }
    final where = worstIndex == null
        ? ''
        : '  worst at [$worstIndex]: '
              '${worstActual!.toStringAsPrecision(6)} vs '
              '${worstReference!.toStringAsPrecision(6)}';
    return '$head  >  $tolerance$where';
  }
}

/// The result of replaying one golden case on one backend.
final class DriftReport {
  const DriftReport({
    required this.goldenId,
    required this.backend,
    required this.tensors,
    this.quantization,
    this.firstDivergentLayer,
    this.attributionAttempted = false,
    this.attributionUnavailable,
    this.layers = const [],
  });

  final String goldenId;

  /// Backend that actually ran, not the one that was requested.
  final String backend;

  final List<TensorDrift> tensors;

  /// Recipe the artifact was exported with, from the manifest.
  final String? quantization;

  /// Earliest layer whose activations diverged.
  ///
  /// Null means either nothing diverged or nothing could be looked at.
  /// [attributionAttempted] distinguishes the two, because "we looked and found
  /// nothing" and "we could not look" are different claims.
  final String? firstDivergentLayer;

  /// Whether per-layer attribution was possible on this run.
  final bool attributionAttempted;

  /// Drift measured at each tapped layer, in the order the graph produces them.
  ///
  /// Empty when nothing was tapped. The order is what makes
  /// [firstDivergentLayer] a claim rather than a pick: it is the first entry
  /// here that failed, and everything before it agreed with the reference.
  final List<TensorDrift> layers;

  /// Why attribution was not possible, when [attributionAttempted] is false.
  ///
  /// A backend without activation taps is one reason and the default message
  /// covers it. It is not the only one: a backend can offer taps while the
  /// goldens hold no reference activations to compare them against, and a
  /// report that blamed the hardware for that would send someone looking in the
  /// wrong place.
  final String? attributionUnavailable;

  bool get passes => tensors.every((t) => t.passes);

  /// Drifts that failed, worst first.
  List<TensorDrift> get failures =>
      (tensors.where((t) => !t.passes).toList()
        ..sort((a, b) => b.maxAbsolute.compareTo(a.maxAbsolute)));

  /// The report as the gate prints it.
  String describe() {
    final b = StringBuffer()
      ..writeln('${passes ? "PASS" : "FAIL"}  parity/$goldenId')
      ..writeln(
        '      backend: $backend  '
        'quantization: ${quantization ?? "none"}',
      );
    for (final t in passes ? tensors : failures) {
      b.writeln('      $t');
      if (!t.passes && t.violatingElements > 0) {
        b.writeln('        ${t.failureReason}');
      }
    }
    if (!passes) {
      if (firstDivergentLayer != null) {
        final layer = layers.firstWhere(
          (l) => l.tensorName == firstDivergentLayer,
          orElse: () => tensors.first,
        );
        b
          ..writeln('      first divergence: layer $firstDivergentLayer')
          ..writeln(
            '        max |Δ| ${layer.maxAbsolute.toStringAsPrecision(3)} '
            'after ${layers.indexOf(layer)} layer(s) inside tolerance',
          );
      } else if (attributionAttempted) {
        b.writeln(
          '      no layer diverged: the ${layers.length} tapped layer(s) all '
          'stayed inside tolerance, so the drift accumulated below them',
        );
      } else if (!attributionAttempted) {
        b.writeln(
          '      no layer attribution: '
          '${attributionUnavailable ?? 'backend "$backend" offers no activation taps'}',
        );
      }
    }
    return b.toString();
  }

  @override
  String toString() => describe();
}

/// Measures how far [actual] strayed from [reference].
///
/// Both tensors must hold the same type and shape; a mismatch there is a defect
/// in the harness rather than drift in the model, and is reported as such.
///
/// Half-precision types have no Dart list view, so they cannot be compared until
/// widening lands with the quantization recipes in M17.
TensorDrift measureDrift({
  required Tensor actual,
  required Tensor reference,
  required Tolerance tolerance,
}) {
  if (actual.spec.dtype != reference.spec.dtype) {
    throw DTypeMismatchException(
      tensorName: reference.spec.name,
      declared: reference.spec.dtype,
      requested: actual.spec.dtype,
    );
  }
  if (!_sameShape(actual.shape, reference.shape)) {
    throw TensorShapeException(
      'actual shape ${actual.shape} differs from the reference '
      '${reference.shape}; this is a harness mismatch, not model drift',
      tensorName: reference.spec.name,
    );
  }

  final a = _asDoubles(actual);
  final r = _asDoubles(reference);
  final n = r.length;

  var violations = 0;
  var maxAbs = 0.0;
  var maxRel = 0.0;
  var sumAbs = 0.0;
  var dot = 0.0;
  var normA = 0.0;
  var normR = 0.0;
  int? worstIndex;
  var worstMargin = double.negativeInfinity;

  for (var i = 0; i < n; i++) {
    final av = a[i];
    final rv = r[i];
    final delta = (av - rv).abs();

    if (delta.isFinite) {
      sumAbs += delta;
      if (delta > maxAbs) maxAbs = delta;
      if (rv != 0) {
        final rel = delta / rv.abs();
        if (rel > maxRel) maxRel = rel;
      }
    }

    if (!tolerance.elementPasses(actual: av, reference: rv)) {
      violations++;
      // Rank by how far past its own allowance the element went, so the element
      // reported as worst is the one that violated hardest, not merely the one
      // with the largest magnitude.
      final margin = delta.isFinite
          ? delta - tolerance.allowanceFor(rv)
          : double.maxFinite;
      if (margin > worstMargin) {
        worstMargin = margin;
        worstIndex = i;
      }
    }

    if (av.isFinite && rv.isFinite) {
      dot += av * rv;
      normA += av * av;
      normR += rv * rv;
    }
  }

  return TensorDrift(
    tensorName: reference.spec.name,
    tolerance: tolerance,
    elementCount: n,
    violatingElements: violations,
    maxAbsolute: maxAbs,
    maxRelative: maxRel,
    meanAbsolute: n == 0 ? 0 : sumAbs / n,
    cosine: _cosine(dot: dot, normA: normA, normR: normR),
    worstIndex: worstIndex,
    worstActual: worstIndex == null ? null : a[worstIndex],
    worstReference: worstIndex == null ? null : r[worstIndex],
  );
}

double _cosine({
  required double dot,
  required double normA,
  required double normR,
}) {
  if (normA == 0 && normR == 0) return 1;
  if (normA == 0 || normR == 0) return 0;
  final c = dot / (math.sqrt(normA) * math.sqrt(normR));
  // Accumulated rounding can push the quotient a hair outside the valid range.
  return c.clamp(-1.0, 1.0);
}

bool _sameShape(List<int> a, List<int> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}

/// Reads a tensor as doubles, whatever it holds.
///
/// Integer outputs are widened rather than compared exactly: a quantized model
/// may legitimately land one step away, and the tolerance is what decides
/// whether one step is acceptable.
List<double> _asDoubles(Tensor t) => switch (t.spec.dtype) {
  DType.float32 => t.asFloat32List(),
  DType.float64 => t.asFloat64List(),
  DType.int8 => t.asInt8List().map((v) => v.toDouble()).toList(growable: false),
  DType.int16 =>
    t.asInt16List().map((v) => v.toDouble()).toList(growable: false),
  DType.int32 =>
    t.asInt32List().map((v) => v.toDouble()).toList(growable: false),
  DType.int64 =>
    t.asInt64List().map((v) => v.toDouble()).toList(growable: false),
  DType.uint8 || DType.boolean =>
    t.asUint8List().map((v) => v.toDouble()).toList(growable: false),
  DType.float16 || DType.bfloat16 => throw UnsupportedError(
    '${t.spec.dtype.wireName} has no Dart list view, so it cannot be compared '
    'until widening lands with the quantization recipes (M17)',
  ),
};
