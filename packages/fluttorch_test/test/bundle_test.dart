import 'package:fluttorch/fluttorch.dart';
import 'package:fluttorch_test/fluttorch_test.dart';
import 'package:test/test.dart';

import 'support/fake_model.dart';

ModelManifest _manifest(List<TensorSpec> inputs) => ModelManifest(
  name: 'two_layer',
  schemaVersion: 1,
  weightHash: 'sha256:00',
  inputs: inputs,
  outputs: const [
    TensorSpec(name: 'score', dtype: DType.float32, shape: [1, 3]),
  ],
  goldens: const [
    GoldenCase(
      id: 'case-0',
      inputKeys: ['0/in/features.bin'],
      outputKeys: ['0/out/score.bin'],
    ),
  ],
);

const _static = TensorSpec(
  name: 'features',
  dtype: DType.float32,
  shape: [1, 4],
);

const _dynamicBatch = TensorSpec(
  name: 'features',
  dtype: DType.float32,
  shape: [TensorSpec.dynamicDim, 4],
);

void main() {
  group('M13 · resolving a golden key to a tensor', () {
    test('the cases are the manifest\'s, in the order it recorded them', () {
      final manifest = _manifest(const [_static]);
      expect(MemoryGoldenBundle(manifest, const {}).cases.map((c) => c.id), [
        'case-0',
      ]);
    });

    test('a static spec fixes the shape', () async {
      final bundle = MemoryGoldenBundle(_manifest(const [_static]), {
        '0/in/features.bin': f32Bytes([1, 2, 3, 4]),
      });

      final tensor = await bundle.tensor('0/in/features.bin', _static);

      expect(tensor.shape, [1, 4]);
      expect(tensor.asFloat32List(), [1, 2, 3, 4]);
    });

    test(
      'bytes that do not satisfy the spec are a mismatched export',
      () async {
        final bundle = MemoryGoldenBundle(_manifest(const [_static]), {
          '0/in/features.bin': f32Bytes([1, 2, 3]),
        });

        await expectLater(
          bundle.tensor('0/in/features.bin', _static),
          throwsA(
            isA<TensorShapeException>().having(
              (e) => e.message,
              'message',
              contains('needs exactly 16'),
            ),
          ),
        );
      },
    );

    test(
      'one dynamic dimension is implied by how many elements arrived',
      () async {
        final bundle = MemoryGoldenBundle(_manifest(const [_dynamicBatch]), {
          '0/in/features.bin': f32Bytes([1, 2, 3, 4, 5, 6, 7, 8]),
        });

        final tensor = await bundle.tensor('0/in/features.bin', _dynamicBatch);

        expect(tensor.shape, [
          2,
          4,
        ], reason: 'eight elements over four columns');
      },
    );

    test('an element count no extent can produce is refused', () async {
      final bundle = MemoryGoldenBundle(_manifest(const [_dynamicBatch]), {
        '0/in/features.bin': f32Bytes([1, 2, 3, 4, 5, 6]),
      });

      await expectLater(
        bundle.tensor('0/in/features.bin', _dynamicBatch),
        throwsA(
          isA<TensorShapeException>().having(
            (e) => e.message,
            'message',
            contains('no extent of the dynamic dimension'),
          ),
        ),
      );
    });

    test(
      'two dynamic dimensions have no unique answer and none is invented',
      () async {
        const twice = TensorSpec(
          name: 'features',
          dtype: DType.float32,
          shape: [TensorSpec.dynamicDim, TensorSpec.dynamicDim],
        );
        final bundle = MemoryGoldenBundle(_manifest(const [twice]), {
          '0/in/features.bin': f32Bytes([1, 2, 3, 4]),
        });

        await expectLater(
          bundle.tensor('0/in/features.bin', twice),
          throwsA(
            isA<TensorShapeException>().having(
              (e) => e.message,
              'message',
              contains('2 dimensions to infer'),
            ),
          ),
        );
      },
    );

    test('a key the bundle does not hold says what it does hold', () async {
      final bundle = MemoryGoldenBundle(_manifest(const [_static]), {
        '0/in/features.bin': f32Bytes([1, 2, 3, 4]),
      });

      await expectLater(
        bundle.tensor('9/in/features.bin', _static),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            allOf(contains('9/in/features.bin'), contains('0/in/features.bin')),
          ),
        ),
      );
    });

    test('an empty bundle says so rather than listing nothing', () async {
      final bundle = MemoryGoldenBundle(_manifest(const [_static]), const {});

      await expectLater(
        bundle.tensor('0/in/features.bin', _static),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            contains('it holds nothing'),
          ),
        ),
      );
    });
  });
}
