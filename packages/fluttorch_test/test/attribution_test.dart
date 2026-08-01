import 'package:fluttorch/fluttorch.dart';
import 'package:fluttorch_test/fluttorch_test.dart';
import 'package:test/test.dart';

import 'support/fake_model.dart';

const _features = TensorSpec(
  name: 'features',
  dtype: DType.float32,
  shape: [1, 2],
);
const _score = TensorSpec(name: 'score', dtype: DType.float32, shape: [1, 2]);
const _l0 = TensorSpec(name: 'encoder.0', dtype: DType.float32, shape: [1, 2]);
const _l1 = TensorSpec(name: 'encoder.1', dtype: DType.float32, shape: [1, 2]);
const _l2 = TensorSpec(name: 'head', dtype: DType.float32, shape: [1, 2]);

ModelManifest _manifest({bool withTaps = true}) => ModelManifest(
  name: 'solar_forecast',
  schemaVersion: 1,
  weightHash: 'sha256:00',
  quantization: 'int8-static',
  inputs: const [_features],
  outputs: const [_score],
  activations: withTaps ? const [_l0, _l1, _l2] : const [],
  goldens: [
    GoldenCase(
      id: 'case-0',
      inputKeys: const ['0/in/features.bin'],
      outputKeys: const ['0/out/score.bin'],
      activationKeys: withTaps
          ? const [
              '0/act/encoder.0.bin',
              '0/act/encoder.1.bin',
              '0/act/head.bin',
            ]
          : const [],
    ),
  ],
);

MemoryGoldenBundle _bundle(ModelManifest manifest) =>
    MemoryGoldenBundle(manifest, {
      '0/in/features.bin': f32Bytes([1, 2]),
      '0/out/score.bin': f32Bytes([10, 20]),
      '0/act/encoder.0.bin': f32Bytes([1, 1]),
      '0/act/encoder.1.bin': f32Bytes([2, 2]),
      '0/act/head.bin': f32Bytes([3, 3]),
    });

/// A model whose output is wrong and whose taps are under the test's control.
FakeModel _model(
  ModelManifest manifest, {
  required Map<String, List<double>> taps,
  List<double> output = const [10, 20],
  bool supportsTaps = true,
}) {
  final model = FakeModel(
    manifest: manifest,
    taps: supportsTaps,
    onRun: (_) async => [
      f32('score', output, shape: [1, 2]),
    ],
  );
  model.taps = {
    for (final e in taps.entries) e.key: f32(e.key, e.value, shape: [1, 2]),
  };
  return model;
}

void main() {
  group('M18 · layer attribution', () {
    test(
      'it names the earliest layer that moved, not the wrong output',
      () async {
        final manifest = _manifest();
        // encoder.0 agrees, encoder.1 is where it goes wrong, head is wrong too.
        final model = _model(
          manifest,
          output: [99, 20],
          taps: {
            'encoder.0': [1, 1],
            'encoder.1': [9, 2],
            'head': [8, 3],
          },
        );

        final report = (await measureParity(
          model,
          goldens: _bundle(manifest),
        )).single;

        expect(report.passes, isFalse);
        expect(report.firstDivergentLayer, 'encoder.1');
        expect(report.attributionAttempted, isTrue);
        expect(report.layers.map((l) => l.tensorName), [
          'encoder.0',
          'encoder.1',
          'head',
        ]);
        expect(
          report.describe(),
          contains('first divergence: layer encoder.1'),
        );
        expect(
          report.describe(),
          contains('after 1 layer(s) inside tolerance'),
        );
      },
    );

    test('a drift that starts at the first layer says so', () async {
      final manifest = _manifest();
      final model = _model(
        manifest,
        output: [99, 20],
        taps: {
          'encoder.0': [50, 1],
          'encoder.1': [9, 2],
          'head': [8, 3],
        },
      );

      final report = (await measureParity(
        model,
        goldens: _bundle(manifest),
      )).single;

      expect(report.firstDivergentLayer, 'encoder.0');
    });

    test(
      'every layer agreeing is reported as such, not as no attempt',
      () async {
        final manifest = _manifest();
        // The taps match their references exactly and the output is still wrong,
        // which is a real outcome: the drift accumulated below what was tapped.
        final model = _model(
          manifest,
          output: [99, 20],
          taps: {
            'encoder.0': [1, 1],
            'encoder.1': [2, 2],
            'head': [3, 3],
          },
        );

        final report = (await measureParity(
          model,
          goldens: _bundle(manifest),
        )).single;

        expect(report.attributionAttempted, isTrue);
        expect(report.firstDivergentLayer, isNull);
        expect(report.describe(), contains('no layer diverged'));
        expect(report.describe(), contains('accumulated below them'));
      },
    );

    test(
      'a layer the backend did not tap is a hole, not an agreement',
      () async {
        final manifest = _manifest();
        final model = _model(
          manifest,
          output: [99, 20],
          taps: {
            'encoder.0': [1, 1],
            // encoder.1 missing: nothing after it can be ruled out.
            'head': [8, 3],
          },
        );

        final report = (await measureParity(
          model,
          goldens: _bundle(manifest),
        )).single;

        expect(report.attributionAttempted, isFalse);
        expect(report.firstDivergentLayer, isNull);
        expect(report.describe(), contains('no activation for "encoder.1"'));
        expect(
          report.describe(),
          contains('no layer after it can be ruled out'),
        );
      },
    );

    test(
      'a hole after the divergence does not invalidate what was found',
      () async {
        final manifest = _manifest();
        final model = _model(
          manifest,
          output: [99, 20],
          taps: {
            'encoder.0': [1, 1],
            'encoder.1': [9, 2],
            // head missing, but the earliest divergence is already established.
          },
        );

        final report = (await measureParity(
          model,
          goldens: _bundle(manifest),
        )).single;

        expect(report.firstDivergentLayer, 'encoder.1');
        expect(report.attributionAttempted, isTrue);
      },
    );

    test('a backend without taps is named as the reason', () async {
      final manifest = _manifest();
      final model = _model(
        manifest,
        supportsTaps: false,
        output: [99, 20],
        taps: const {},
      );

      final report = (await measureParity(
        model,
        goldens: _bundle(manifest),
      )).single;

      expect(report.attributionAttempted, isFalse);
      expect(report.describe(), contains('offers no activation taps'));
    });

    test(
      'an export with no taps blames the goldens rather than the device',
      () async {
        final manifest = _manifest(withTaps: false);
        final model = _model(manifest, output: [99, 20], taps: const {});

        final report = (await measureParity(
          model,
          goldens: _bundle(manifest),
        )).single;

        expect(report.attributionAttempted, isFalse);
        expect(report.describe(), contains('final outputs only'));
      },
    );
  });
}
