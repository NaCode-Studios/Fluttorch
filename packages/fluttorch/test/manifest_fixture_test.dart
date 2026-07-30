@TestOn('vm')
library;

import 'dart:io';

import 'package:fluttorch/fluttorch.dart';
import 'package:test/test.dart';

/// The document the Python exporter writes and this package reads.
///
/// Both suites assert against the same file. A JSON schema alone would let the
/// two implementations drift while each remained "valid" on its own terms; a
/// fixture neither side may disagree with will not.
final _fixture = File('../../testdata/manifest_v1.json');

void main() {
  late String source;
  late ModelManifest manifest;

  setUpAll(() {
    expect(
      _fixture.existsSync(),
      isTrue,
      reason: 'run from packages/fluttorch so the shared testdata resolves',
    );
    source = _fixture.readAsStringSync();
    manifest = ManifestCodec.decode(source);
  });

  test('the exporter and this reader agree byte for byte', () {
    // The strongest form of the claim: not merely that Dart can parse what
    // Python wrote, but that re-encoding it reproduces the same document. A
    // difference here is a divergence in the contract, not in formatting.
    expect(ManifestCodec.encode(manifest), source);
  });

  test('the schema version is one this build can read', () {
    expect(
      manifest.schemaVersion,
      lessThanOrEqualTo(ModelManifest.currentSchemaVersion),
    );
  });

  test('every field the exporter wrote arrives intact', () {
    expect(manifest.name, 'solar_forecast');
    expect(manifest.weightHash, 'sha256:2f1c0a');
    expect(manifest.quantization, 'int8-static');
    expect(manifest.labels, ['low', 'high']);

    expect(manifest.inputs.map((s) => s.name), ['window', 'calendar']);
    expect(manifest.inputNamed('window').shape, [1, 168, 6]);
    expect(manifest.inputNamed('window').dtype, DType.float32);
    expect(manifest.outputNamed('load_mw').shape, [1, 24]);
  });

  test('a dynamic dimension crosses the language boundary as dynamic', () {
    final calendar = manifest.inputNamed('calendar');
    expect(calendar.isDynamic, isTrue);
    expect(calendar.shape, [TensorSpec.dynamicDim, 4]);
    expect(calendar.elementCount, isNull);
    expect(calendar.elementCountFor([3, 4]), 12);
  });

  test('every preprocessing step is recognised, in order', () {
    expect(manifest.hasUnknownPreprocessing, isFalse);
    expect(manifest.preprocessing.map((s) => s.kind), [
      'rescale',
      'normalize',
      'resize',
      'center_crop',
      'cast',
    ]);
  });

  test('the transforms that change a prediction survive exactly', () {
    final rescale = manifest.preprocessing.whereType<RescaleStep>().single;
    expect(rescale.factor, closeTo(1 / 255, 1e-15));

    final normalize = manifest.preprocessing.whereType<NormalizeStep>().single;
    expect(normalize.mean, [0.5, 0.25]);
    expect(normalize.std, [0.2, 0.1]);
    expect(normalize.axis, 2);

    // Nearest and bilinear disagree by enough to move a prediction, so the
    // filter is part of the contract rather than a rendering detail.
    final resize = manifest.preprocessing.whereType<ResizeStep>().single;
    expect(resize.interpolation, 'nearest');
    expect(resize.height, 168);
  });

  test('goldens keep their identity and their optional description', () {
    expect(manifest.goldens.map((g) => g.id), ['case-0', 'case-1']);
    final first = manifest.goldens.first;
    expect(first.inputKeys, ['in/0/window', 'in/0/calendar']);
    expect(first.outputKeys, ['out/0/load_mw']);
    expect(first.description, 'a winter evening peak');
    expect(manifest.goldens.last.description, isNull);
  });

  test('every golden names one key per declared tensor', () {
    for (final g in manifest.goldens) {
      expect(g.inputKeys, hasLength(manifest.inputs.length), reason: g.id);
      expect(g.outputKeys, hasLength(manifest.outputs.length), reason: g.id);
    }
  });
}
