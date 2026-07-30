@TestOn('vm')
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';
import 'package:test/test.dart';

/// A real export, produced by `fluttorch-export` and committed whole.
///
/// The manifest fixture in `manifest_codec_test.dart` is hand-written to exercise
/// every field. This one is the opposite check: whatever the exporter actually
/// emits, read back by the code that has to consume it. A contract verified only
/// against documents written to test it is a contract verified against itself.
final _dir = Directory('../../testdata/two_layer');

void main() {
  late ModelManifest manifest;
  late Uint8List artifact;

  setUpAll(() {
    expect(
      _dir.existsSync(),
      isTrue,
      reason: 'run from packages/fluttorch so the shared testdata resolves',
    );
    manifest = ManifestCodec.decode(
      File('${_dir.path}/two_layer.fluttorch.json').readAsStringSync(),
    );
    artifact = File('${_dir.path}/two_layer.pte').readAsBytesSync();
  });

  group('M6 · the artifact and its manifest come from one export', () {
    test('the committed artifact verifies against its manifest', () {
      expect(
        () => verifyArtifact(artifact: artifact, manifest: manifest),
        returnsNormally,
      );
    });

    test('the digest matches what the exporter recorded', () {
      expect(digestOf(artifact), manifest.weightHash);
      expect(manifest.weightHash, startsWith('sha256:'));
    });

    test('a single flipped byte is caught', () {
      // The whole point: an artifact that differs at all is a different model,
      // and every downstream number would be wrong while every shape still fits.
      final tampered = Uint8List.fromList(artifact)
        ..[artifact.length ~/ 2] ^= 0x01;
      expect(
        () => verifyArtifact(artifact: tampered, manifest: manifest),
        throwsA(
          isA<ArtifactMismatchException>()
              .having((e) => e.expectedHash, 'expected', manifest.weightHash)
              .having((e) => e.actualHash, 'actual', isNot(manifest.weightHash))
              .having(
                (e) => e.message,
                'message',
                contains('different exports'),
              ),
        ),
      );
    });

    test('a truncated artifact is caught', () {
      expect(
        () => verifyArtifact(
          artifact: Uint8List.sublistView(artifact, 0, artifact.length - 1),
          manifest: manifest,
        ),
        throwsA(isA<ArtifactMismatchException>()),
      );
    });

    ModelManifest withHash(String hash) => ModelManifest(
      name: manifest.name,
      schemaVersion: manifest.schemaVersion,
      weightHash: hash,
      inputs: manifest.inputs,
      outputs: manifest.outputs,
    );

    test('an unknown algorithm is a format problem, not a mismatch', () {
      // Different problem, different fix: the reader is old, the model is fine.
      expect(
        () => verifyArtifact(
          artifact: artifact,
          manifest: withHash('blake3:abc'),
        ),
        throwsA(
          isA<ManifestFormatException>().having(
            (e) => e.message,
            'message',
            contains('cannot compute'),
          ),
        ),
      );
    });

    test('a hash with no algorithm is rejected', () {
      expect(
        () =>
            verifyArtifact(artifact: artifact, manifest: withHash('deadbeef')),
        throwsA(
          isA<ManifestFormatException>().having(
            (e) => e.field,
            'field',
            'weight_hash',
          ),
        ),
      );
    });
  });

  group('M7 · the goldens satisfy the contract they were captured under', () {
    test('the exporter captured the cases it was given', () {
      expect(manifest.goldens, hasLength(4));
      expect(manifest.goldens.map((g) => g.id), [
        'case-0',
        'case-1',
        'case-2',
        'case-3',
      ]);
    });

    test('every golden names one key per declared tensor', () {
      for (final g in manifest.goldens) {
        expect(g.inputKeys, hasLength(manifest.inputs.length), reason: g.id);
        expect(g.outputKeys, hasLength(manifest.outputs.length), reason: g.id);
      }
    });

    test('every stored tensor satisfies its spec', () {
      // Constructing a Tensor is where bytes.length == elements x width is
      // checked, so this asserts the exporter wrote what it declared.
      for (final g in manifest.goldens) {
        for (final (i, key) in g.inputKeys.indexed) {
          final spec = manifest.inputs[i];
          final t = Tensor.view(
            spec: spec,
            bytes: File('${_dir.path}/goldens/$key').readAsBytesSync(),
          );
          expect(t.shape, spec.shape, reason: '${g.id} $key');
          expect(t.elementCount, spec.elementCount, reason: '${g.id} $key');
        }
        for (final (i, key) in g.outputKeys.indexed) {
          final spec = manifest.outputs[i];
          final t = Tensor.view(
            spec: spec,
            bytes: File('${_dir.path}/goldens/$key').readAsBytesSync(),
          );
          expect(t.shape, spec.shape, reason: '${g.id} $key');
        }
      }
    });

    test('the bytes are little-endian floats, not a foreign layout', () {
      // case-1 is torch.ones, so every input element must read back as exactly 1.
      final ones = manifest.goldens[1];
      final t = Tensor.view(
        spec: manifest.inputs.single,
        bytes: File(
          '${_dir.path}/goldens/${ones.inputKeys.single}',
        ).readAsBytesSync(),
      );
      expect(t.asFloat32List(), everyElement(1.0));
    });

    test('case-2 is uniformly negative, so most units are dead', () {
      final t = Tensor.view(
        spec: manifest.inputs.single,
        bytes: File(
          '${_dir.path}/goldens/${manifest.goldens[2].inputKeys.single}',
        ).readAsBytesSync(),
      );
      expect(t.asFloat32List(), everyElement(-1.0));
    });

    test('the wide-range case survived the round trip intact', () {
      // 1e3 and 1e-3 in one tensor: the case that catches a float32 written as
      // float64, or a narrowing conversion nobody meant to make.
      final t = Tensor.view(
        spec: manifest.inputs.single,
        bytes: File(
          '${_dir.path}/goldens/${manifest.goldens[3].inputKeys.single}',
        ).readAsBytesSync(),
      );
      expect(t.asFloat32List(), [1e3, -1e3, closeTo(1e-3, 1e-9), 0.0]);
    });
  });

  group('the manifest the exporter writes is one the reader understands', () {
    test('names come through as the export named them', () {
      expect(manifest.name, 'two_layer');
      expect(manifest.inputs.single.name, 'features');
      expect(manifest.outputs.single.name, 'score');
      expect(manifest.labels, ['low', 'mid', 'high']);
    });

    test('full precision is recorded as absent, not as a recipe name', () {
      expect(manifest.quantization, isNull);
    });

    test('nothing in it is beyond this build', () {
      expect(manifest.hasUnknownPreprocessing, isFalse);
      expect(manifest.schemaVersion, ModelManifest.currentSchemaVersion);
    });
  });
}
