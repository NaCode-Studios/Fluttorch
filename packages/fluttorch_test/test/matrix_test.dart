import 'package:fluttorch/fluttorch.dart';
import 'package:fluttorch_test/fluttorch_test.dart';
import 'package:test/test.dart';

import 'support/fake_model.dart';

ModelManifest _manifest({
  String? quantization,
  List<GoldenCase> goldens = const [
    GoldenCase(
      id: 'case-0',
      inputKeys: ['0/in/features.bin'],
      outputKeys: ['0/out/load_mw.bin'],
    ),
    GoldenCase(
      id: 'case-1',
      inputKeys: ['1/in/features.bin'],
      outputKeys: ['1/out/load_mw.bin'],
    ),
  ],
}) => ModelManifest(
  name: 'solar_forecast',
  schemaVersion: 1,
  weightHash:
      'sha256:0000000000000000000000000000000000000000000000000000000000000000',
  quantization: quantization,
  inputs: const [
    TensorSpec(name: 'features', dtype: DType.float32, shape: [1, 4]),
  ],
  outputs: const [
    TensorSpec(name: 'load_mw', dtype: DType.float32, shape: [1, 3]),
  ],
  goldens: goldens,
);

MemoryGoldenBundle _bundle(ModelManifest manifest) =>
    MemoryGoldenBundle(manifest, {
      '0/in/features.bin': f32Bytes([0.1, 0.2, 0.3, 0.4]),
      '0/out/load_mw.bin': f32Bytes([12.3, 4.5, 0.75]),
      '1/in/features.bin': f32Bytes([0.5, 0.6, 0.7, 0.8]),
      '1/out/load_mw.bin': f32Bytes([31.0, 9.5, 1.25]),
    });

Future<MatrixEntry> _entry(
  String backend, {
  void Function(String, List<Tensor>)? perturb,
  String? quantization,
}) async {
  final manifest = _manifest(quantization: quantization);
  final goldens = _bundle(manifest);
  return MatrixEntry(
    model: await FakeModel.replaying(
      manifest,
      goldens,
      backend: backend,
      perturb: perturb,
    ),
    goldens: goldens,
  );
}

void main() {
  group('M24 · the parity matrix', () {
    test('it measures every golden on every backend supplied', () async {
      final matrix = await measureMatrix([
        await _entry('xnnpack'),
        await _entry('coreml'),
        await _entry('mps'),
      ]);

      expect(matrix.backends, ['xnnpack', 'coreml', 'mps']);
      expect(matrix.goldenIds, ['case-0', 'case-1']);
      // Two goldens on three backends is six measurements, and a matrix missing
      // a cell is the case the report has to be able to say something about.
      expect(matrix.reports, hasLength(6));
      for (final backend in matrix.backends) {
        for (final id in matrix.goldenIds) {
          expect(
            matrix.at(backend: backend, goldenId: id),
            isNotNull,
            reason: '$backend/$id',
          );
        }
      }
      expect(matrix.passes, isTrue);
    });

    test('one backend drifting fails the matrix and is named', () async {
      final matrix = await measureMatrix([
        await _entry('xnnpack'),
        await _entry(
          'coreml',
          // Only this backend, and only on one case, so the report has to
          // locate a single cell rather than declare the model broken.
          perturb: (id, outputs) {
            if (id == 'case-1') outputs.first.asFloat32List()[0] = 99;
          },
        ),
        await _entry('mps'),
      ]);

      expect(matrix.passes, isFalse);
      expect(matrix.at(backend: 'coreml', goldenId: 'case-1')!.passes, isFalse);
      // Everything else still agreed, which is the claim that makes the failure
      // attributable to the backend rather than to the model.
      expect(matrix.at(backend: 'coreml', goldenId: 'case-0')!.passes, isTrue);
      expect(matrix.at(backend: 'xnnpack', goldenId: 'case-1')!.passes, isTrue);
      expect(matrix.at(backend: 'mps', goldenId: 'case-1')!.passes, isTrue);

      final report = matrix.describe();
      expect(report, startsWith('FAIL'));
      expect(report, contains('coreml'));
      expect(report, contains('case-1'));
    });

    test('the report has a row per golden and a column per backend', () async {
      final matrix = await measureMatrix([
        await _entry('xnnpack'),
        await _entry('coreml'),
      ]);

      final lines = matrix.describe().trimRight().split('\n');
      expect(lines.first, startsWith('PASS'));
      // Header, then one row per golden.
      expect(lines[1], contains('xnnpack'));
      expect(lines[1], contains('coreml'));
      expect(lines[2], contains('case-0'));
      expect(lines[3], contains('case-1'));
    });

    test('each backend answers to the bound its own manifest implies', () async {
      // The point of measuring per entry rather than once: an int8 export and a
      // float32 one of the same model are not wrong by the same amount, and a
      // single bound would either excuse the second or condemn the first.
      final matrix = await measureMatrix([
        await _entry('xnnpack', quantization: 'int8-static'),
        await _entry('coreml'),
      ]);

      final quantized = matrix.at(backend: 'xnnpack', goldenId: 'case-0')!;
      final exact = matrix.at(backend: 'coreml', goldenId: 'case-0')!;
      expect(quantized.quantization, 'int8-static');
      expect(exact.quantization, isNull);
      expect(
        quantized.tensors.single.tolerance.maxAbsolute,
        greaterThan(exact.tensors.single.tolerance.maxAbsolute),
      );
    });

    test('goldens that do not line up are refused, not tabulated', () async {
      final other = _manifest(
        goldens: const [
          GoldenCase(
            id: 'case-9',
            inputKeys: ['0/in/features.bin'],
            outputKeys: ['0/out/load_mw.bin'],
          ),
        ],
      );

      await expectLater(
        measureMatrix([
          await _entry('xnnpack'),
          MatrixEntry(
            model: await FakeModel.replaying(
              other,
              _bundle(other),
              backend: 'coreml',
            ),
            goldens: _bundle(other),
          ),
        ]),
        throwsA(isA<ArgumentError>()),
      );
    });

    test('a matrix over nothing is refused rather than passing', () async {
      // The failure this package exists to prevent, in its smallest form.
      await expectLater(measureMatrix(const []), throwsA(isA<ArgumentError>()));
    });
  });
}
