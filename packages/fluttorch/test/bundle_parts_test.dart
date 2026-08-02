@TestOn('vm')
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';
import 'package:test/test.dart';

/// A bundle whose artifact is more than one file.
///
/// VoltaCast through `torch.onnx` is the export that produces one: 506 kB of
/// graph structure and 3.4 MB of weights in a file beside it, referenced by
/// name from within the graph.
///
/// The hash is the whole of why this is carrying rather than pretending. It is
/// computed over the artifact and every part together, in two languages that
/// never see each other run: `bundle_digest` in the exporter writes the number
/// and [bundleDigestOf] here recomputes it. A framing either side got wrong
/// would not show up as a wrong answer somewhere later, it would show up as a
/// bundle that cannot be loaded at all, so it is worth a test that says which
/// of the two moved.
final _dir = Directory('../../testdata/voltacast_onnx');

void main() {
  late ModelManifest manifest;
  late Uint8List artifact;
  late Map<String, Uint8List> parts;

  setUpAll(() async {
    manifest = ManifestCodec.decode(
      await File('${_dir.path}/voltacast.fluttorch.json').readAsString(),
    );
    artifact = await File('${_dir.path}/voltacast.onnx').readAsBytes();
    parts = {
      for (final part in manifest.parts)
        part.name: await File('${_dir.path}/${part.name}').readAsBytes(),
    };
  });

  group('a bundle that is more than one file', () {
    test('the manifest names the file the graph references', () {
      expect(manifest.parts, hasLength(1));
      expect(manifest.parts.single.name, 'voltacast.onnx.data');
      expect(manifest.parts.single.size, greaterThan(1000000));
    });

    test('it declares a version a reader without parts refuses', () {
      // Not decoration. A reader that ignored `parts` would load 506 kB of
      // structure, find every shape it expected, run, and answer from a graph
      // with no numbers in it. The version is the only thing that stops it,
      // because the decoder skips keys it does not know by design.
      expect(manifest.schemaVersion, 2);
      expect(
        manifest.schemaVersion,
        greaterThan(ModelManifest.schemaVersionWithoutParts),
      );
    });

    test('the two languages agree on what the hash covers', () {
      // The exporter wrote weight_hash with its own framing. If this recomputes
      // the same number, both sides frame the name and the length identically.
      expect(
        bundleDigestOf(artifact, [
          for (final part in manifest.parts)
            (name: part.name, bytes: parts[part.name]!),
        ]),
        manifest.weightHash,
      );
    });

    test('the graph alone does not satisfy the hash', () {
      // The assertion the issue turns on. Before parts, weight_hash was taken
      // over the artifact, so this bundle would have passed every check while
      // carrying none of the numbers. If these two ever agree, that is back.
      expect(digestOf(artifact), isNot(manifest.weightHash));
    });

    test(
      'verifying it needs the part, and says so by name when it is absent',
      () {
        expect(
          () => verifyArtifact(artifact: artifact, manifest: manifest),
          throwsA(
            isA<BundlePartMissingException>()
                .having((e) => e.missing, 'missing', ['voltacast.onnx.data'])
                .having((e) => e.model, 'model', 'voltacast')
                .having(
                  (e) => e.remedy,
                  'remedy',
                  contains('beside the manifest'),
                ),
          ),
        );
      },
    );

    test('verifying it passes once the part is there', () {
      expect(
        () => verifyArtifact(
          artifact: artifact,
          manifest: manifest,
          parts: parts,
        ),
        returnsNormally,
      );
    });

    test('a part whose bytes moved is refused, and named', () {
      final tampered = Uint8List.fromList(parts.values.single);
      tampered[tampered.length ~/ 2] ^= 0xFF;

      expect(
        () => verifyArtifact(
          artifact: artifact,
          manifest: manifest,
          parts: {'voltacast.onnx.data': tampered},
        ),
        throwsA(
          isA<ArtifactMismatchException>().having(
            (e) => e.expectedHash,
            'expectedHash',
            contains('voltacast.onnx.data'),
          ),
        ),
      );
    });

    test(
      'a part of the wrong length is reported as that, not as a bad hash',
      () {
        // Different mistakes with different fixes. A truncated copy is a
        // deployment that cut a file short; a hash mismatch is a stale export.
        expect(
          () => verifyArtifact(
            artifact: artifact,
            manifest: manifest,
            parts: {
              'voltacast.onnx.data': Uint8List.sublistView(
                parts.values.single,
                0,
                parts.values.single.length - 1,
              ),
            },
          ),
          throwsA(
            isA<ArtifactMismatchException>().having(
              (e) => e.actualHash,
              'actualHash',
              contains('bytes where'),
            ),
          ),
        );
      },
    );
  });
}
