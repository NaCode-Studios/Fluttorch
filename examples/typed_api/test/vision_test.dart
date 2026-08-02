@TestOn('vm')
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:test/test.dart';
import 'package:typed_api_example/vision.fluttorch.dart';

/// M31 · the spatial steps are generated, and they agree with training.
///
/// A resize and a crop are the two preprocessing steps whose correctness cannot
/// be argued from reading the code. Bilinear has a half-pixel convention,
/// nearest deliberately does not use it, and a centre crop with an odd margin
/// lands on one row or the row above it. Each choice yields a picture either
/// way, so the only check worth running is against the library that trained the
/// model.
///
/// `python/fluttorch_export/scripts/export_vision.py` writes both halves:
/// `source-N.bin` is a frame at a size no model accepts, and `expected-N.bin`
/// is what torch produced from it through the pipeline this manifest records.
/// Nothing here is compared against Dart's own output.
final _fixtures = Directory('../../testdata/vision/preprocessing');

void main() {
  final sources = File('${_fixtures.path}/sources.txt')
      .readAsLinesSync()
      .where((l) => l.trim().isNotEmpty)
      .map((l) {
        final parts = l.split(' ');
        return (int.parse(parts[0]), int.parse(parts[1]));
      })
      .toList();

  Float32List read(String name) {
    final bytes = File('${_fixtures.path}/$name').readAsBytesSync();
    return Float32List.view(
      bytes.buffer.asByteData(bytes.offsetInBytes, bytes.lengthInBytes).buffer,
      bytes.offsetInBytes,
      bytes.lengthInBytes ~/ 4,
    );
  }

  group('M31 · a source frame becomes the tensor the model declared', () {
    test('the manifest records which axes are spatial', () {
      // Without this the generator refuses, and it is right to: the same bytes
      // read as NCHW and as NHWC are two different pictures.
      expect(VisionImage.spec.layout, isNotNull);
      expect(VisionImage.spec.layout!.wireName, 'nchw');
      expect(VisionImage.spec.layout!.heightAxis, 2);
      expect(VisionImage.spec.layout!.widthAxis, 3);
    });

    for (var i = 0; i < 3; i++) {
      final (h, w) = sources[i];
      test('a ${h}x$w frame lands where torch put it', () {
        final image = VisionImage.fromSource(
          read('source-$i.bin'),
          height: h,
          width: w,
        )..preprocess();

        final expected = read('expected-$i.bin');
        final actual = image.values;
        expect(actual.length, expected.length);

        // `|a - b| <= atol + rtol * |b|`, the same combined form the parity
        // gate uses, and for the same reason. A pure relative bound is
        // meaningless here because normalize subtracts a mean of 0.485 and so
        // carries values through zero, where any relative error is unbounded. A
        // pure absolute bound would instead be sized by the largest value,
        // after normalize has divided by 0.22 and multiplied every
        // disagreement by four and a half.
        //
        // The residual being measured is torch's, not this code's. Dart has no
        // float32 arithmetic: values read out of a Float32List widen to double,
        // interpolate, and narrow on the way back, while torch keeps every
        // intermediate at float32. Its epsilon is 1.2e-7, a bilinear tap is
        // three multiply-adds and normalize two more, so a few epsilons is the
        // honest ceiling and the bound below sits an order above it.
        //
        // It stays far below what this test exists to catch. A transposed axis,
        // the wrong filter, or a crop one row off moves whole pixels, and on
        // this fixture that is a disagreement of order one.
        const atol = 1e-5;
        const rtol = 1e-5;
        var worst = 0.0;
        var worstAt = -1;
        for (var j = 0; j < expected.length; j++) {
          final slack = atol + rtol * expected[j].abs();
          final excess = (actual[j] - expected[j]).abs() / slack;
          if (excess > worst) {
            worst = excess;
            worstAt = j;
          }
        }
        expect(
          worst,
          lessThan(1.0),
          reason:
              'element $worstAt missed the bound by ${worst}x: '
              'got ${actual[worstAt]}, torch said ${expected[worstAt]}',
        );
      });
    }

    test('a frame of the wrong size is refused rather than reshaped', () {
      // The size is the caller's to get right and the only one this can check.
      // Reading past the end of a short buffer would produce a picture.
      expect(
        () => VisionImage.fromSource(
          Float32List(3 * 4 * 4),
          height: 5,
          width: 5,
        ),
        throwsA(isA<Exception>()),
      );
    });

    test('upsampling is covered, not just downsampling', () {
      // The third fixture is smaller than the resize target in both axes, so it
      // exercises the branch where the half-pixel coordinate clamps at zero.
      final (h, w) = sources[2];
      expect(h < 11 && w < 11, isTrue);
    });
  });
}
