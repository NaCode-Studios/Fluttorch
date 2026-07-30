import 'package:fluttorch/fluttorch.dart';
import 'package:fluttorch_gen/fluttorch_gen.dart';
import 'package:test/test.dart';

ModelManifest manifest({
  List<PreprocessingStep> preprocessing = const [],
  List<TensorSpec> inputs = const [
    TensorSpec(name: 'x', dtype: DType.float32, shape: [1]),
  ],
  List<TensorSpec> outputs = const [
    TensorSpec(name: 'y', dtype: DType.float32, shape: [1]),
  ],
}) => ModelManifest(
  name: 'm',
  schemaVersion: ModelManifest.currentSchemaVersion,
  weightHash: 'sha256:00',
  inputs: inputs,
  outputs: outputs,
  preprocessing: preprocessing,
);

void main() {
  group('checkGeneratable', () {
    test('a manifest this build understands passes', () {
      expect(
        () => checkGeneratable(
          manifest(preprocessing: const [RescaleStep(factor: 1 / 255)]),
        ),
        returnsNormally,
      );
    });

    test('an unknown preprocessing step is refused, not skipped', () {
      // Skipping it would emit an API missing a transform the model was trained
      // with, and the resulting skew is invisible everywhere downstream.
      expect(
        () => checkGeneratable(
          manifest(
            preprocessing: const [
              UnknownPreprocessingStep(kind: 'mel_spectrogram', params: {}),
            ],
          ),
        ),
        throwsA(
          isA<UnsupportedManifestException>().having(
            (e) => e.reasons.single,
            'reason',
            allOf(contains('mel_spectrogram'), contains('silently drop')),
          ),
        ),
      );
    });

    test('it reports every reason at once, not the first one found', () {
      expect(
        () => checkGeneratable(
          manifest(
            inputs: const [
              TensorSpec(name: '', dtype: DType.float32, shape: [1]),
            ],
            outputs: const [],
            preprocessing: const [
              UnknownPreprocessingStep(kind: 'a', params: {}),
              UnknownPreprocessingStep(kind: 'b', params: {}),
            ],
          ),
        ),
        throwsA(
          isA<UnsupportedManifestException>().having(
            (e) => e.reasons,
            'reasons',
            hasLength(4),
          ),
        ),
      );
    });

    test('the message lists each reason on its own line', () {
      try {
        checkGeneratable(manifest(outputs: const []));
        fail('expected UnsupportedManifestException');
      } on UnsupportedManifestException catch (e) {
        expect(e.toString(), contains('· the manifest declares no outputs'));
      }
    });
  });
}
