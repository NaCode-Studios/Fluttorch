import 'package:fluttorch/fluttorch.dart';
import 'package:test/test.dart';

void main() {
  group('TensorSpec', () {
    test('reports element count for a fully static shape', () {
      const spec = TensorSpec(
        name: 'input',
        dtype: DType.float32,
        shape: [1, 3, 224, 224],
      );

      expect(spec.isDynamic, isFalse);
      expect(spec.elementCount, 150528);
    });

    test('has no element count when a dimension is dynamic', () {
      const spec = TensorSpec(
        name: 'tokens',
        dtype: DType.int64,
        shape: [1, -1],
      );

      expect(spec.isDynamic, isTrue);
      expect(spec.elementCount, isNull);
    });

    test('treats a scalar as a single element', () {
      const spec = TensorSpec(name: 'loss', dtype: DType.float32, shape: []);

      expect(spec.elementCount, 1);
    });
  });
}
