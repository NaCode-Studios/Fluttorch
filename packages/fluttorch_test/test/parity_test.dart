import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';
import 'package:fluttorch_test/fluttorch_test.dart';
import 'package:test/test.dart';

import 'support/fake_model.dart';

ModelManifest _manifest({
  String? quantization = 'int8-static',
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

void main() {
  group('M13 · the golden replay runner', () {
    test('it replays every case and returns one report each', () async {
      final manifest = _manifest();
      final goldens = _bundle(manifest);
      final model = await FakeModel.replaying(manifest, goldens);

      final reports = await measureParity(model, goldens: goldens);

      expect(model.runs, 2);
      expect(reports.map((r) => r.goldenId), ['case-0', 'case-1']);
      expect(reports.every((r) => r.passes), isTrue);
    });

    test(
      'a passing report names the backend that ran and the recipe',
      () async {
        final manifest = _manifest();
        final goldens = _bundle(manifest);
        final model = await FakeModel.replaying(
          manifest,
          goldens,
          backend: 'coreml',
        );

        final text = (await measureParity(
          model,
          goldens: goldens,
        )).first.describe();

        expect(text, startsWith('PASS  parity/case-0'));
        expect(text, contains('backend: coreml'));
        expect(text, contains('quantization: int8-static'));
        expect(text, contains('output "load_mw"'));
      },
    );

    test('each case is measured against its own reference', () async {
      final manifest = _manifest();
      final goldens = _bundle(manifest);
      // The second case answers with the first case's numbers. Pairing the
      // wrong golden with the wrong output is the bug this catches, and it is
      // invisible if every case happens to hold similar values.
      final model = FakeModel(
        manifest: manifest,
        onRun: (_) async => [
          f32('load_mw', [12.3, 4.5, 0.75], shape: [1, 3]),
        ],
      );

      final reports = await measureParity(model, goldens: goldens);

      expect(reports[0].passes, isTrue);
      expect(reports[1].passes, isFalse);
      expect(reports[1].failures.single.tensorName, 'load_mw');
    });

    test(
      'the tolerance defaults to the starting point for the recipe',
      () async {
        final manifest = _manifest();
        final goldens = _bundle(manifest);
        // 0.05 is inside the int8-static starting point and outside a tolerance
        // measured for a full-precision export.
        final model = await FakeModel.replaying(
          manifest,
          goldens,
          perturb: (_, outputs) => outputs.first.asFloat32List()[0] += 0.05,
        );

        expect(
          (await measureParity(model, goldens: goldens)).every((r) => r.passes),
          isTrue,
        );
        expect(
          (await measureParity(
            model,
            goldens: goldens,
            tolerance: Tolerance(maxAbsolute: 1e-5),
          )).every((r) => r.passes),
          isFalse,
        );
      },
    );

    test(
      'an unrecognised recipe asks for a tolerance instead of inventing one',
      () async {
        final manifest = _manifest(quantization: 'int3-experimental');
        final goldens = _bundle(manifest);
        final model = await FakeModel.replaying(manifest, goldens);

        await expectLater(
          measureParity(model, goldens: goldens),
          throwsA(
            isA<ArgumentError>().having(
              (e) => e.message,
              'message',
              allOf(contains('int8-static'), contains('nobody chose')),
            ),
          ),
        );
      },
    );

    test(
      'a case that names the wrong number of tensors is a bundle mismatch',
      () async {
        final manifest = _manifest(
          goldens: const [
            GoldenCase(
              id: 'case-0',
              inputKeys: ['0/in/features.bin', '0/in/extra.bin'],
              outputKeys: ['0/out/load_mw.bin'],
            ),
          ],
        );
        final model = FakeModel(manifest: manifest, onRun: (_) async => []);

        await expectLater(
          measureParity(model, goldens: _bundle(manifest)),
          throwsA(
            isA<TensorShapeException>().having(
              (e) => e.message,
              'message',
              contains('different exports'),
            ),
          ),
        );
      },
    );

    test(
      'outputs in the wrong order are caught before they read as drift',
      () async {
        final manifest = _manifest();
        final model = FakeModel(
          manifest: manifest,
          onRun: (_) async => [
            f32('humidity', [12.3, 4.5, 0.75], shape: [1, 3]),
          ],
        );

        await expectLater(
          measureParity(model, goldens: _bundle(manifest)),
          throwsA(
            isA<TensorShapeException>().having(
              (e) => e.message,
              'message',
              contains('positional and out of order'),
            ),
          ),
        );
      },
    );

    test(
      'a backend with taps is not blamed for goldens that lack activations',
      () async {
        final manifest = _manifest();
        final goldens = _bundle(manifest);
        final model = await FakeModel.replaying(
          manifest,
          goldens,
          taps: true,
          perturb: (_, outputs) => outputs.first.asFloat32List()[0] += 5,
        );

        final text = (await measureParity(
          model,
          goldens: goldens,
        )).first.describe();

        expect(text, contains('no layer attribution'));
        expect(text, contains('final outputs only'));
        expect(text, isNot(contains('offers no activation taps')));
      },
    );

    test('a backend without taps says that is what is missing', () async {
      final manifest = _manifest();
      final goldens = _bundle(manifest);
      final model = await FakeModel.replaying(
        manifest,
        goldens,
        perturb: (_, outputs) => outputs.first.asFloat32List()[0] += 5,
      );

      expect(
        (await measureParity(model, goldens: goldens)).first.describe(),
        contains('offers no activation taps'),
      );
    });
  });

  group('M15 · expectParity', () {
    test('it returns quietly when every case is inside tolerance', () async {
      final manifest = _manifest();
      final goldens = _bundle(manifest);
      final model = await FakeModel.replaying(manifest, goldens);

      await expectParity(model, goldens: goldens);
    });

    test(
      'the failure names the tensor, the drift, the tolerance and the backend',
      () async {
        final manifest = _manifest();
        final goldens = _bundle(manifest);
        final model = await FakeModel.replaying(
          manifest,
          goldens,
          backend: 'xnnpack',
          perturb: (id, outputs) {
            if (id == 'case-1') outputs.first.asFloat32List()[2] += 0.5;
          },
        );

        await expectLater(
          expectParity(model, goldens: goldens),
          throwsA(
            isA<TestFailure>().having(
              (e) => e.message,
              'message',
              allOf([
                contains('1 of 2 golden cases'),
                contains('FAIL  parity/case-1'),
                contains('backend: xnnpack'),
                contains('output "load_mw"'),
                contains('max |Δ|'),
                contains('Tolerance(atol'),
                contains('worst at [2]'),
                isNot(contains('case-0')),
              ]),
            ),
          ),
        );
      },
    );

    test(
      'a bundle with no cases fails rather than passing over nothing',
      () async {
        final manifest = _manifest(goldens: const []);
        final model = FakeModel(manifest: manifest, onRun: (_) async => []);

        await expectLater(
          expectParity(model, goldens: MemoryGoldenBundle(manifest, const {})),
          throwsA(
            isA<TestFailure>().having(
              (e) => e.message,
              'message',
              contains('nobody evaluated'),
            ),
          ),
        );
      },
    );

    test('a drift only the cosine bound can see is reported as that', () async {
      final manifest = _manifest(
        quantization: null,
        goldens: const [
          GoldenCase(
            id: 'case-0',
            inputKeys: ['0/in/features.bin'],
            outputKeys: ['0/out/load_mw.bin'],
          ),
        ],
      );
      // Every element stays within a wide elementwise bound while the tensor
      // turns: the case both elementwise bounds are blind to.
      final model = FakeModel(
        manifest: manifest,
        onRun: (_) async => [
          f32('load_mw', [4.5, 12.3, 0.75], shape: [1, 3]),
        ],
      );

      await expectLater(
        expectParity(
          model,
          goldens: MemoryGoldenBundle(manifest, {
            '0/in/features.bin': f32Bytes([0.1, 0.2, 0.3, 0.4]),
            '0/out/load_mw.bin': f32Bytes([12.3, 4.5, 0.75]),
          }),
          tolerance: Tolerance(maxAbsolute: 100, minCosine: 0.999),
        ),
        throwsA(
          isA<TestFailure>().having(
            (e) => e.message,
            'message',
            allOf(contains('cosine'), contains('required')),
          ),
        ),
      );
    });
  });

  group('M24 · the parity matrix is not built yet', () {
    test('it says which milestone builds it', () {
      final manifest = _manifest();
      expect(
        () => parityMatrix(
          _NoRuntime(),
          artifact: ModelArtifact(bytes: Uint8List(0), manifest: manifest),
          goldens: _bundle(manifest),
        ),
        throwsA(isA<UnimplementedError>()),
      );
    });
  });
}

final class _NoRuntime implements FluttorchRuntime {
  @override
  Future<List<RuntimeCapabilities>> capabilities() async => const [];

  @override
  Future<LoadedModel> load({
    required Uint8List artifact,
    required ModelManifest manifest,
    String? backend,
  }) => throw UnimplementedError();
}
