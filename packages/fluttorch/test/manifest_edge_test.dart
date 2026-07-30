@TestOn('vm')
library;

import 'dart:io';

import 'package:fluttorch/fluttorch.dart';
import 'package:test/test.dart';

/// The schema-legal documents nobody writes by hand.
///
/// M3's fixture is the happy path. This one holds the values where the two
/// languages nearly disagreed: doubles either side of the notation thresholds,
/// a denormal, negative zero, non-ASCII in a name and in a label, a rank-0
/// tensor, and a fully dynamic shape. Every one of them was a byte-for-byte
/// divergence before the exporter grew a canonical encoder.
final _fixture = File('../../testdata/manifest_edge_v1.json');

void main() {
  late String source;
  late ModelManifest manifest;

  setUpAll(() {
    source = _fixture.readAsStringSync();
    manifest = ManifestCodec.decode(source);
  });

  test('the exporter and this reader agree byte for byte on all of it', () {
    // This is the assertion that was accidentally true before: the happy-path
    // fixture happened to contain only values the two languages spell the same.
    expect(ManifestCodec.encode(manifest), source);
  });

  group('doubles survive both thresholds and the round trip', () {
    List<double> meansOf(ModelManifest m) =>
        m.preprocessing.whereType<NormalizeStep>().single.mean;

    test('values on either side of the notation boundary are exact', () {
      // 1e-5 is fixed notation in Dart and exponential in Python's default;
      // 1e16 is the reverse. Both must read back as themselves.
      expect(meansOf(manifest), [1e-5, 1e16, 1 / 3, 5e-324]);
    });

    test('the smallest denormal is not flushed to zero', () {
      expect(meansOf(manifest)[3], 5e-324);
      expect(meansOf(manifest)[3], greaterThan(0));
    });

    test('a repeating fraction keeps every bit it had', () {
      expect(meansOf(manifest)[2], 1 / 3);
    });

    test('negative zero stays negative zero', () {
      final rescale = manifest.preprocessing.whereType<RescaleStep>().single;
      expect(rescale.offset, 0.0);
      expect(rescale.offset.isNegative, isTrue, reason: '-0.0, not 0.0');
      expect(rescale.factor, 1e-7);
    });
  });

  group('shapes at the edges of what a spec can say', () {
    test('a rank-0 tensor is one element, not zero', () {
      final scalar = manifest.inputNamed('scalar');
      expect(scalar.rank, 0);
      expect(scalar.shape, isEmpty);
      expect(scalar.elementCount, 1);
      expect(scalar.isDynamic, isFalse);
    });

    test('every dimension may be dynamic at once', () {
      final ragged = manifest.inputNamed('ragged');
      expect(ragged.isDynamic, isTrue);
      expect(ragged.elementCount, isNull);
      expect(ragged.elementCountFor([3, 5]), 15);
      expect(ragged.byteLengthFor([3, 5]), 120, reason: 'int64 is 8 bytes');
    });

    test('a type with no Dart list view still describes itself', () {
      final y = manifest.outputNamed('y');
      expect(y.dtype, DType.bfloat16);
      expect(y.dtype.hasTypedListView, isFalse);
      expect(y.dtype.bytesPerElement, 2);
      expect(y.dtype.isFloatingPoint, isTrue);
    });
  });

  group('text that is legal and awkward', () {
    test('non-ASCII in a tensor name survives unescaped', () {
      expect(manifest.inputs.map((s) => s.name), contains('temperatura_°C'));
      expect(source, contains('temperatura_°C'), reason: 'raw, not \\u00b0');
    });

    test('an empty label is a label, not an absent one', () {
      expect(manifest.labels, hasLength(3));
      expect(manifest.labels!.first, isEmpty);
    });

    test('non-ASCII in a label survives too', () {
      expect(manifest.labels, contains('unicode ✓'));
    });
  });

  test('an absent description stays absent rather than becoming empty', () {
    expect(manifest.goldens.first.description, isNull);
    expect(manifest.goldens.last.description, 'with a description');
  });
}
