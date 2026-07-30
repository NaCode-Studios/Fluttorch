import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';
import 'package:fluttorch_test/fluttorch_test.dart';
import 'package:test/test.dart';

TensorSpec _spec(
  int n, {
  String name = 'load_mw',
  DType dtype = DType.float32,
}) => TensorSpec(name: name, dtype: dtype, shape: [n]);

Tensor _f32(List<double> values, {String name = 'load_mw'}) {
  final spec = _spec(values.length, name: name);
  final t = Tensor.zeros(spec);
  t.asFloat32List().setAll(0, values);
  return t;
}

Tensor _i32(List<int> values) {
  final spec = _spec(values.length, dtype: DType.int32);
  final t = Tensor.zeros(spec);
  t.asInt32List().setAll(0, values);
  return t;
}

void main() {
  group('measureDrift', () {
    test('identical tensors drift nowhere', () {
      final d = measureDrift(
        actual: _f32([1, 2, 3]),
        reference: _f32([1, 2, 3]),
        tolerance: Tolerance(maxAbsolute: 1e-9),
      );

      expect(d.passes, isTrue);
      expect(d.violatingElements, 0);
      expect(d.maxAbsolute, 0);
      expect(d.maxRelative, 0);
      expect(d.cosine, closeTo(1, 1e-12));
      expect(d.worstIndex, isNull);
    });

    test('it counts violations and locates the worst one', () {
      final d = measureDrift(
        actual: _f32([1, 2, 3.5, 4]),
        reference: _f32([1, 2, 3, 4]),
        tolerance: Tolerance(maxAbsolute: 0.1),
      );

      expect(d.passes, isFalse);
      expect(d.violatingElements, 1);
      expect(d.elementCount, 4);
      expect(d.worstIndex, 2);
      expect(d.worstActual, closeTo(3.5, 1e-6));
      expect(d.worstReference, closeTo(3, 1e-6));
      expect(d.maxAbsolute, closeTo(0.5, 1e-6));
    });

    test('the worst element is the one that violated hardest', () {
      // Index 0 has the larger absolute delta, index 1 the larger overspend
      // against its own allowance once the relative term is applied.
      final d = measureDrift(
        actual: _f32([100, 2]),
        reference: _f32([90, 1]),
        tolerance: Tolerance(maxRelative: 0.2),
      );

      expect(d.violatingElements, 1, reason: '100 vs 90 is within 20%');
      expect(d.worstIndex, 1);
    });

    test('mean absolute averages over every element, not just failures', () {
      final d = measureDrift(
        actual: _f32([1, 2, 3, 8]),
        reference: _f32([1, 2, 3, 4]),
        tolerance: Tolerance(maxAbsolute: 0.1),
      );
      expect(d.meanAbsolute, closeTo(1.0, 1e-6));
    });

    test('relative drift is measured only where the reference is non-zero', () {
      final d = measureDrift(
        actual: _f32([0.5, 2]),
        reference: _f32([0, 1]),
        tolerance: Tolerance(maxAbsolute: 10),
      );
      // 0.5 against a zero reference contributes no relative error.
      expect(d.maxRelative, closeTo(1.0, 1e-6));
    });
  });

  group('cosine catches what the elementwise bound cannot', () {
    test('small magnitudes pass elementwise while pointing elsewhere', () {
      final actual = _f32([0.001, -0.001, 0.001]);
      final reference = _f32([-0.001, 0.001, -0.001]);

      final elementwiseOnly = measureDrift(
        actual: actual,
        reference: reference,
        tolerance: Tolerance(maxAbsolute: 0.01),
      );
      expect(
        elementwiseOnly.passes,
        isTrue,
        reason: 'every delta is 0.002, inside the bound',
      );

      final withDirection = measureDrift(
        actual: actual,
        reference: reference,
        tolerance: Tolerance(maxAbsolute: 0.01, minCosine: 0.99),
      );
      expect(withDirection.passes, isFalse);
      expect(withDirection.cosine, closeTo(-1, 1e-6));
      expect(withDirection.failureReason, contains('cosine'));
    });

    test('two all-zero tensors are identical, not undefined', () {
      final d = measureDrift(
        actual: _f32([0, 0]),
        reference: _f32([0, 0]),
        tolerance: Tolerance(maxAbsolute: 1e-9, minCosine: 0.999),
      );
      expect(d.cosine, 1);
      expect(d.passes, isTrue);
    });

    test('one zero tensor and one not have nothing in common', () {
      final d = measureDrift(
        actual: _f32([0, 0]),
        reference: _f32([1, 1]),
        tolerance: Tolerance(maxAbsolute: 10, minCosine: 0.5),
      );
      expect(d.cosine, 0);
      expect(d.passes, isFalse);
    });

    test('cosine stays inside [-1, 1] despite accumulated rounding', () {
      final many = List.filled(4096, 0.1);
      final d = measureDrift(
        actual: _f32(many),
        reference: _f32(many),
        tolerance: Tolerance(maxAbsolute: 1e-9),
      );
      expect(d.cosine, lessThanOrEqualTo(1.0));
      expect(d.cosine, greaterThanOrEqualTo(-1.0));
    });
  });

  group('a NaN is never absorbed by a wide bound', () {
    test('a NaN where a number was expected fails', () {
      final d = measureDrift(
        actual: _f32([double.nan, 1]),
        reference: _f32([1, 1]),
        tolerance: Tolerance(maxAbsolute: 1e9),
      );
      expect(d.passes, isFalse);
      expect(d.violatingElements, 1);
      expect(d.worstIndex, 0);
    });
  });

  group('integer outputs are widened, not compared exactly', () {
    test('a one-step difference is the tolerance decision, not an error', () {
      final tight = measureDrift(
        actual: _i32([5, 7]),
        reference: _i32([5, 6]),
        tolerance: Tolerance(maxAbsolute: 0.5),
      );
      expect(tight.passes, isFalse);

      final loose = measureDrift(
        actual: _i32([5, 7]),
        reference: _i32([5, 6]),
        tolerance: Tolerance(maxAbsolute: 1),
      );
      expect(loose.passes, isTrue);
    });
  });

  group('a mismatch is a harness defect, reported as such', () {
    test('different dtypes', () {
      expect(
        () => measureDrift(
          actual: _i32([1]),
          reference: _f32([1]),
          tolerance: Tolerance(maxAbsolute: 1),
        ),
        throwsA(isA<DTypeMismatchException>()),
      );
    });

    test('different shapes say it is not model drift', () {
      expect(
        () => measureDrift(
          actual: _f32([1, 2]),
          reference: _f32([1, 2, 3]),
          tolerance: Tolerance(maxAbsolute: 1),
        ),
        throwsA(
          isA<TensorShapeException>().having(
            (e) => e.message,
            'message',
            contains('not model drift'),
          ),
        ),
      );
    });

    test('half precision says why it cannot be compared yet', () {
      const spec = TensorSpec(name: 'h', dtype: DType.float16, shape: [2]);
      final half = Tensor.view(spec: spec, bytes: Uint8List(4));
      expect(
        () => measureDrift(
          actual: half,
          reference: half,
          tolerance: Tolerance(maxAbsolute: 1),
        ),
        throwsA(
          isA<UnsupportedError>().having(
            (e) => e.message,
            'message',
            contains('M17'),
          ),
        ),
      );
    });
  });

  group('DriftReport', () {
    DriftReport report({
      required bool fail,
      String backend = 'xnnpack',
      bool taps = false,
    }) => DriftReport(
      goldenId: 'solar_forecast',
      backend: backend,
      quantization: 'int8-static',
      attributionAttempted: taps,
      firstDivergentLayer: fail && taps ? 'conv3' : null,
      tensors: [
        measureDrift(
          actual: _f32([1, fail ? 2 : 1]),
          reference: _f32([1, 1]),
          tolerance: Tolerance(maxAbsolute: 0.01),
        ),
      ],
    );

    test('a passing report says PASS and names the backend', () {
      final text = report(fail: false).describe();
      expect(text, startsWith('PASS  parity/solar_forecast'));
      expect(text, contains('backend: xnnpack'));
      expect(text, contains('int8-static'));
    });

    test('a failing report leads with the worst tensor', () {
      final r = report(fail: true);
      expect(r.passes, isFalse);
      expect(r.failures, hasLength(1));
      expect(r.describe(), startsWith('FAIL  parity/solar_forecast'));
      expect(r.describe(), contains('load_mw'));
    });

    test('with taps, it names the layer where divergence started', () {
      expect(
        report(fail: true, taps: true).describe(),
        contains('first divergence: layer conv3'),
      );
    });

    test(
      'without taps, it says it could not look rather than saying nothing',
      () {
        final text = report(fail: true).describe();
        expect(text, contains('no layer attribution'));
        expect(text, contains('offers no activation taps'));
      },
    );

    test('a report of many tensors passes only if every one does', () {
      final r = DriftReport(
        goldenId: 'c',
        backend: 'coreml',
        tensors: [
          measureDrift(
            actual: _f32([1], name: 'a'),
            reference: _f32([1], name: 'a'),
            tolerance: Tolerance(maxAbsolute: 0.01),
          ),
          measureDrift(
            actual: _f32([9], name: 'b'),
            reference: _f32([1], name: 'b'),
            tolerance: Tolerance(maxAbsolute: 0.01),
          ),
        ],
      );
      expect(r.passes, isFalse);
      expect(r.failures.single.tensorName, 'b');
    });
  });
}
