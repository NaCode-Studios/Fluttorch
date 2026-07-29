import 'package:fluttorch/fluttorch.dart';
import 'package:test/test.dart';

TensorDrift _drift({
  double maxAbsolute = 0,
  double meanAbsolute = 0,
  double cosine = 1,
}) => TensorDrift(
  tensorName: 'out',
  maxAbsolute: maxAbsolute,
  meanAbsolute: meanAbsolute,
  cosine: cosine,
);

void main() {
  group('TensorDrift.satisfies', () {
    test('accepts drift inside the absolute bound', () {
      expect(
        _drift(
          maxAbsolute: 0.004,
        ).satisfies(const Tolerance(maxAbsolute: 0.01)),
        isTrue,
      );
    });

    test('rejects drift beyond the absolute bound', () {
      expect(
        _drift(
          maxAbsolute: 0.118,
        ).satisfies(const Tolerance(maxAbsolute: 0.01)),
        isFalse,
      );
    });

    test('rejects structural divergence the absolute bound would miss', () {
      // Small magnitudes keep the elementwise difference tiny while the tensor
      // points somewhere else entirely — the case cosine exists to catch.
      final drift = _drift(maxAbsolute: 0.001, cosine: 0.72);

      expect(drift.satisfies(const Tolerance(maxAbsolute: 0.01)), isTrue);
      expect(
        drift.satisfies(const Tolerance(maxAbsolute: 0.01, minCosine: 0.999)),
        isFalse,
      );
    });

    test('an unconstrained tolerance accepts anything', () {
      expect(
        _drift(maxAbsolute: 42, cosine: -1).satisfies(const Tolerance()),
        isTrue,
      );
    });
  });

  group('DriftReport.passes', () {
    test('fails when any single tensor fails', () {
      final report = DriftReport(
        goldenId: 'case-0',
        backend: 'xnnpack',
        tensors: [_drift(maxAbsolute: 0.001), _drift(maxAbsolute: 0.5)],
      );

      expect(report.passes(const Tolerance(maxAbsolute: 0.01)), isFalse);
    });

    test('passes when every tensor is within bounds', () {
      final report = DriftReport(
        goldenId: 'case-0',
        backend: 'xnnpack',
        tensors: [_drift(maxAbsolute: 0.001), _drift(maxAbsolute: 0.002)],
      );

      expect(report.passes(const Tolerance(maxAbsolute: 0.01)), isTrue);
    });
  });

  group('built-in tolerances', () {
    test('exact is tighter than int8', () {
      expect(
        Tolerance.exact.maxAbsolute,
        lessThan(Tolerance.int8.maxAbsolute!),
      );
    });

    test('int8 constrains direction as well as magnitude', () {
      expect(Tolerance.int8.minCosine, isNotNull);
    });
  });
}
