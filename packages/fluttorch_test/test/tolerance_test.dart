import 'package:fluttorch_test/fluttorch_test.dart';
import 'package:test/test.dart';

void main() {
  group('construction', () {
    test('a tolerance with no bound is refused', () {
      // The bug this guards: a gate that stays green while the model rots.
      expect(() => Tolerance(), throwsA(isA<AssertionError>()));
    });

    test('one bound is enough', () {
      expect(() => Tolerance(maxAbsolute: 1e-3), returnsNormally);
      expect(() => Tolerance(maxRelative: 1e-3), returnsNormally);
      expect(() => Tolerance(minCosine: 0.99), returnsNormally);
    });

    test('negative bounds and impossible cosines are refused', () {
      expect(() => Tolerance(maxAbsolute: -1), throwsA(isA<AssertionError>()));
      expect(() => Tolerance(maxRelative: -1), throwsA(isA<AssertionError>()));
      expect(() => Tolerance(minCosine: 1.5), throwsA(isA<AssertionError>()));
    });
  });

  group('the relative term is actually applied', () {
    // The original defect: maxRelative was declared and never read, so a
    // tolerance configured with only a relative bound accepted everything.
    final relative = Tolerance(maxRelative: 0.01);

    test('one per cent of a large reference is allowed', () {
      expect(relative.elementPasses(actual: 1005, reference: 1000), isTrue);
    });

    test('two per cent of the same reference is not', () {
      expect(relative.elementPasses(actual: 1020, reference: 1000), isFalse);
    });

    test('a relative bound alone does not accept an absolute drift', () {
      expect(relative.elementPasses(actual: 5, reference: 1), isFalse);
    });

    test('it scales with the reference rather than being fixed', () {
      expect(relative.elementPasses(actual: 0.1005, reference: 0.1), isTrue);
      expect(relative.elementPasses(actual: 0.12, reference: 0.1), isFalse);
    });
  });

  group('the two elementwise bounds combine, not compete', () {
    final t = Tolerance(maxAbsolute: 0.01, maxRelative: 0.001);

    test('the absolute term carries the comparison near zero', () {
      // A relative bound alone would reject this; 0.001 × 0 is 0.
      expect(t.elementPasses(actual: 0.005, reference: 0), isTrue);
    });

    test('the relative term carries it on large magnitudes', () {
      // An absolute bound of 0.01 alone would reject a 1.0 drift on 10000.
      expect(t.elementPasses(actual: 10001, reference: 10000), isTrue);
    });

    test('the allowance is the sum of both terms', () {
      expect(t.allowanceFor(0), 0.01);
      expect(t.allowanceFor(1000), closeTo(0.01 + 1.0, 1e-12));
    });
  });

  group('non-finite values', () {
    final t = Tolerance(maxAbsolute: 1e9);

    test('a NaN where a number was expected fails, however wide the bound', () {
      expect(t.elementPasses(actual: double.nan, reference: 1), isFalse);
    });

    test('NaN matches NaN, so a legitimately undefined output is stable', () {
      expect(
        t.elementPasses(actual: double.nan, reference: double.nan),
        isTrue,
      );
    });

    test('infinity matches only the same infinity', () {
      expect(
        t.elementPasses(actual: double.infinity, reference: double.infinity),
        isTrue,
      );
      expect(
        t.elementPasses(
          actual: double.infinity,
          reference: double.negativeInfinity,
        ),
        isFalse,
      );
      expect(t.elementPasses(actual: double.infinity, reference: 1), isFalse);
    });
  });

  group('starting points per recipe', () {
    test('full precision is the tightest', () {
      final exact = Tolerance.startingPointFor(null)!;
      final static8 = Tolerance.startingPointFor('int8-static')!;
      expect(exact.maxAbsolute, lessThan(static8.maxAbsolute));
      expect(exact.minCosine!, greaterThan(static8.minCosine!));
    });

    test('looser quantization gets a looser bound, in order', () {
      final recipes = [
        Tolerance.startingPointFor(null)!,
        Tolerance.startingPointFor('int8-dynamic')!,
        Tolerance.startingPointFor('int8-static')!,
        Tolerance.startingPointFor('int4-weight-only')!,
      ];
      for (var i = 1; i < recipes.length; i++) {
        expect(
          recipes[i].maxAbsolute,
          greaterThan(recipes[i - 1].maxAbsolute),
          reason: 'recipe $i should be looser than ${i - 1}',
        );
        expect(
          recipes[i].minCosine!,
          lessThan(recipes[i - 1].minCosine!),
          reason: 'recipe $i should constrain direction less than ${i - 1}',
        );
      }
    });

    test('every known recipe has a starting point, and vice versa', () {
      for (final r in Tolerance.knownRecipes) {
        expect(Tolerance.startingPointFor(r), isNotNull, reason: r);
      }
    });

    test('an unrecognised recipe returns null instead of inventing one', () {
      expect(Tolerance.startingPointFor('int2-magic'), isNull);
      expect(Tolerance.startingPointFor('fp8-e4m3'), isNull);
    });

    test('every starting point constrains all three, so none fails open', () {
      for (final r in [null, ...Tolerance.knownRecipes]) {
        final t = Tolerance.startingPointFor(r)!;
        expect(t.maxAbsolute, greaterThan(0), reason: '$r');
        expect(t.maxRelative, greaterThan(0), reason: '$r');
        expect(t.minCosine, isNotNull, reason: '$r');
      }
    });
  });

  test('toString reads as a bound, for a failure message', () {
    expect(
      Tolerance(
        maxAbsolute: 0.01,
        maxRelative: 0.1,
        minCosine: 0.998,
      ).toString(),
      'Tolerance(atol 0.01, rtol 0.1, cos ≥ 0.998)',
    );
  });

  group('M23 · a precision the delegate lowered at', () {
    test('float32 and absence mean the same thing', () {
      expect(
        Tolerance.startingPointFor(null, precision: 'float32')!.maxRelative,
        Tolerance.startingPointFor(null)!.maxRelative,
      );
    });

    test('float16 is looser than full precision and tighter than int8', () {
      final half = Tolerance.startingPointFor(null, precision: 'float16')!;
      final full = Tolerance.startingPointFor(null)!;
      final int8 = Tolerance.startingPointFor('int8-dynamic')!;

      expect(half.maxRelative, greaterThan(full.maxRelative));
      // The two have to stay distinguishable. A half-precision bound as wide as
      // an int8 one would mean the manifest recording the difference bought
      // nothing.
      expect(half.maxRelative, lessThan(int8.maxRelative));
    });

    test('a recipe and a precision compound rather than replace', () {
      // int8 on a half-precision GPU is wrong in both ways at once, and the
      // bound has to be at least as wide as either alone.
      final both = Tolerance.startingPointFor(
        'int8-dynamic',
        precision: 'float16',
      )!;
      final int8 = Tolerance.startingPointFor('int8-dynamic')!;
      final half = Tolerance.startingPointFor(null, precision: 'float16')!;

      expect(both.maxAbsolute, greaterThanOrEqualTo(int8.maxAbsolute));
      expect(both.maxAbsolute, greaterThanOrEqualTo(half.maxAbsolute));
      expect(both.maxRelative, greaterThanOrEqualTo(int8.maxRelative));
      expect(both.minCosine, lessThanOrEqualTo(int8.minCosine!));
    });

    test('a precision nobody measured a bound for returns null', () {
      // Same rule as an unknown recipe: inventing a number for a scheme this
      // build has never seen would put it on a gate nobody chose.
      expect(Tolerance.startingPointFor(null, precision: 'float8'), isNull);
      expect(Tolerance.startingPointFor(null, precision: 'bfloat16'), isNull);
    });
  });
}
