import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';
import 'package:test/test.dart';

const _image = TensorSpec(
  name: 'pixels',
  dtype: DType.float32,
  shape: [1, 3, 2, 2],
);
const _dynamicBatch = TensorSpec(
  name: 'tokens',
  dtype: DType.int64,
  shape: [TensorSpec.dynamicDim, 8],
);

void main() {
  group('DType', () {
    test('every member carries a width, so a buffer length is checkable', () {
      for (final d in DType.values) {
        expect(d.bytesPerElement, greaterThan(0), reason: d.name);
      }
    });

    test('wire names round-trip and are unique', () {
      final names = DType.values.map((d) => d.wireName).toList();
      expect(names.toSet(), hasLength(names.length));
      for (final d in DType.values) {
        expect(DType.tryParse(d.wireName), same(d));
      }
    });

    test('an unknown wire name is null rather than an exception', () {
      expect(DType.tryParse('float8'), isNull);
    });

    test('byteLengthFor multiplies extent by width', () {
      expect(DType.float32.byteLengthFor([1, 3, 2, 2]), 48);
      expect(DType.uint8.byteLengthFor([224, 224, 3]), 150528);
    });
  });

  group('TensorSpec', () {
    test('element count is known for a static shape', () {
      expect(_image.isDynamic, isFalse);
      expect(_image.elementCount, 12);
      expect(_image.rank, 4);
    });

    test('element count is null for a dynamic shape, not a guess', () {
      expect(_dynamicBatch.isDynamic, isTrue);
      expect(_dynamicBatch.elementCount, isNull);
    });

    test('elementCountFor resolves a dynamic dimension', () {
      expect(_dynamicBatch.elementCountFor([4, 8]), 32);
      expect(_dynamicBatch.byteLengthFor([4, 8]), 256);
    });

    test('a scalar is one element', () {
      const scalar = TensorSpec(name: 'loss', dtype: DType.float32, shape: []);
      expect(scalar.elementCount, 1);
    });

    test('accepts is exact on static dimensions', () {
      expect(_image.accepts([1, 3, 2, 2]), isTrue);
      expect(_image.accepts([1, 3, 2, 3]), isFalse);
      expect(_image.accepts([1, 3, 2]), isFalse, reason: 'wrong rank');
    });

    test('accepts any positive extent where the spec is dynamic', () {
      expect(_dynamicBatch.accepts([1, 8]), isTrue);
      expect(_dynamicBatch.accepts([512, 8]), isTrue);
      expect(_dynamicBatch.accepts([4, 9]), isFalse, reason: 'static dim');
    });

    test('resolve rejects a shape that violates the spec', () {
      expect(
        () => _image.resolve([1, 3, 2, 3]),
        throwsA(isA<TensorShapeException>()),
      );
    });

    test('resolve refuses to invent a shape for a dynamic spec', () {
      expect(
        () => _dynamicBatch.resolve(null),
        throwsA(
          isA<TensorShapeException>().having(
            (e) => e.message,
            'message',
            contains('must be supplied'),
          ),
        ),
      );
    });
  });

  group('Tensor', () {
    test('a view over a correctly sized buffer does not copy', () {
      final bytes = Uint8List(48);
      final t = Tensor.view(spec: _image, bytes: bytes);
      expect(t.bytes, same(bytes));
      expect(t.elementCount, 12);
      expect(t.shape, [1, 3, 2, 2]);
    });

    test('a wrong-sized buffer is rejected at construction', () {
      expect(
        () => Tensor.view(spec: _image, bytes: Uint8List(47)),
        throwsA(
          isA<TensorShapeException>()
              .having((e) => e.tensorName, 'tensorName', 'pixels')
              .having(
                (e) => e.message,
                'message',
                contains('needs exactly 48'),
              ),
        ),
      );
    });

    test('a misaligned buffer offset is rejected', () {
      // A float32 view cannot start on an odd byte.
      final backing = Uint8List(49);
      expect(
        () =>
            Tensor.view(spec: _image, bytes: Uint8List.sublistView(backing, 1)),
        throwsA(
          isA<TensorShapeException>().having(
            (e) => e.message,
            'message',
            contains('not aligned'),
          ),
        ),
      );
    });

    test('typed views write through to the same memory', () {
      final t = Tensor.zeros(_image);
      t.asFloat32List()[5] = 1.5;
      final again = Tensor.view(spec: _image, bytes: t.bytes);
      expect(again.asFloat32List()[5], 1.5);
    });

    test('reading a tensor as the wrong type names both types', () {
      final t = Tensor.zeros(_image);
      expect(
        () => t.asInt32List(),
        throwsA(
          isA<DTypeMismatchException>()
              .having((e) => e.declared, 'declared', DType.float32)
              .having((e) => e.requested, 'requested', DType.int32),
        ),
      );
    });

    test('bool and uint8 share a representation', () {
      const flags = TensorSpec(name: 'mask', dtype: DType.boolean, shape: [4]);
      final t = Tensor.zeros(flags);
      expect(t.bytes, hasLength(4));
      expect(() => t.asUint8List(), returnsNormally);
    });

    test('a dynamic spec needs its shape at construction', () {
      final t = Tensor.zeros(_dynamicBatch, shape: [2, 8]);
      expect(t.shape, [2, 8]);
      expect(t.bytes, hasLength(128));
      expect(
        () => Tensor.zeros(_dynamicBatch),
        throwsA(isA<TensorShapeException>()),
      );
    });

    test('the resolved shape cannot be mutated after construction', () {
      final t = Tensor.zeros(_image);
      expect(() => t.shape[0] = 9, throwsUnsupportedError);
    });
  });

  group('checkTensorsAgainst', () {
    List<Tensor> two() => [
      Tensor.zeros(_image),
      Tensor.zeros(_dynamicBatch, shape: [2, 8]),
    ];

    test('accepts a matching positional list', () {
      expect(
        () =>
            checkTensorsAgainst([_image, _dynamicBatch], two(), role: 'inputs'),
        returnsNormally,
      );
    });

    test('catches the wrong count', () {
      expect(
        () => checkTensorsAgainst([_image], two(), role: 'inputs'),
        throwsA(
          isA<TensorShapeException>().having(
            (e) => e.message,
            'message',
            contains('expected 1 inputs, got 2'),
          ),
        ),
      );
    });

    test('catches a list that is in the wrong order', () {
      expect(
        () =>
            checkTensorsAgainst([_dynamicBatch, _image], two(), role: 'inputs'),
        throwsA(
          isA<TensorShapeException>().having(
            (e) => e.message,
            'message',
            contains('out of order'),
          ),
        ),
      );
    });
  });
}
