import 'dart:convert';

import 'package:fluttorch/fluttorch.dart';
import 'package:test/test.dart';

final _full = ModelManifest(
  name: 'solar_forecast',
  schemaVersion: ModelManifest.currentSchemaVersion,
  weightHash: 'sha256:2f1c0a',
  quantization: 'int8-static',
  inputs: const [
    TensorSpec(name: 'window', dtype: DType.float32, shape: [1, 168, 6]),
    TensorSpec(
      name: 'calendar',
      dtype: DType.int64,
      shape: [TensorSpec.dynamicDim, 4],
    ),
  ],
  outputs: const [
    TensorSpec(name: 'load_mw', dtype: DType.float32, shape: [1, 24]),
  ],
  preprocessing: const [
    RescaleStep(factor: 1 / 255),
    NormalizeStep(mean: [0.5, 0.25], std: [0.2, 0.1], axis: 2),
    ResizeStep(height: 168, width: 6, interpolation: 'nearest'),
    CenterCropStep(height: 160, width: 6),
    CastStep('float32'),
  ],
  labels: const ['low', 'high'],
  goldens: const [
    GoldenCase(
      id: 'case-0',
      inputKeys: ['in/0/window', 'in/0/calendar'],
      outputKeys: ['out/0/load_mw'],
      description: 'a winter evening peak',
    ),
    GoldenCase(
      id: 'case-1',
      inputKeys: ['in/1/window', 'in/1/calendar'],
      outputKeys: ['out/1/load_mw'],
    ),
  ],
);

void main() {
  group('round trip', () {
    test('every field survives encode then decode', () {
      final back = ManifestCodec.decode(ManifestCodec.encode(_full));

      expect(back.name, _full.name);
      expect(back.schemaVersion, _full.schemaVersion);
      expect(back.weightHash, _full.weightHash);
      expect(back.quantization, 'int8-static');
      expect(back.labels, ['low', 'high']);

      expect(back.inputs.map((s) => s.name), ['window', 'calendar']);
      expect(back.inputs[1].shape, [TensorSpec.dynamicDim, 4]);
      expect(back.inputs[1].dtype, DType.int64);
      expect(back.outputs.single.shape, [1, 24]);

      expect(back.preprocessing.map((s) => s.kind), [
        'rescale',
        'normalize',
        'resize',
        'center_crop',
        'cast',
      ]);
      expect(back.hasUnknownPreprocessing, isFalse);

      final normalize = back.preprocessing.whereType<NormalizeStep>().single;
      expect(normalize.mean, [0.5, 0.25]);
      expect(normalize.std, [0.2, 0.1]);
      expect(normalize.axis, 2);

      expect(back.goldens.map((g) => g.id), ['case-0', 'case-1']);
      expect(back.goldens.first.inputKeys, ['in/0/window', 'in/0/calendar']);
      expect(back.goldens.first.description, 'a winter evening peak');
      expect(back.goldens.last.description, isNull);
    });

    test('encoding is stable, so a committed manifest diffs cleanly', () {
      final once = ManifestCodec.encode(_full);
      final twice = ManifestCodec.encode(ManifestCodec.decode(once));
      expect(twice, once);
    });

    test('absent optionals are omitted rather than written as null', () {
      final minimal = ModelManifest(
        name: 'm',
        schemaVersion: 1,
        weightHash: 'sha256:00',
        inputs: const [
          TensorSpec(name: 'x', dtype: DType.float32, shape: [1]),
        ],
        outputs: const [
          TensorSpec(name: 'y', dtype: DType.float32, shape: [1]),
        ],
      );
      final json = ManifestCodec.toJson(minimal);
      expect(json.containsKey('quantization'), isFalse);
      expect(json.containsKey('labels'), isFalse);
      expect(json.containsKey('goldens'), isFalse);
      expect(json.containsKey('preprocessing'), isFalse);
      expect(
        ManifestCodec.decode(ManifestCodec.encode(minimal)).quantization,
        isNull,
      );
    });
  });

  group('forward compatibility', () {
    test('an unknown preprocessing step is preserved, not dropped', () {
      final decoded = ManifestCodec.decode('''
        {
          "schema_version": 1,
          "name": "m",
          "weight_hash": "sha256:00",
          "inputs": [{"name":"x","dtype":"float32","shape":[1]}],
          "outputs": [{"name":"y","dtype":"float32","shape":[1]}],
          "preprocessing": [{"kind":"mel_spectrogram","n_fft":400,"hop":160}]
        }
      ''');

      expect(decoded.hasUnknownPreprocessing, isTrue);
      final step = decoded.preprocessing.single as UnknownPreprocessingStep;
      expect(step.kind, 'mel_spectrogram');
      expect(step.params, {'n_fft': 400, 'hop': 160});
    });

    test('an unknown step survives a round trip intact', () {
      final once = ManifestCodec.decode('''
        {
          "schema_version": 1, "name": "m", "weight_hash": "h",
          "inputs": [{"name":"x","dtype":"float32","shape":[1]}],
          "outputs": [{"name":"y","dtype":"float32","shape":[1]}],
          "preprocessing": [{"kind":"future_thing","alpha":0.5}]
        }
      ''');
      final twice = ManifestCodec.decode(ManifestCodec.encode(once));
      final step = twice.preprocessing.single as UnknownPreprocessingStep;
      expect(step.kind, 'future_thing');
      expect(step.params, {'alpha': 0.5});
    });

    test('a newer schema version is refused with both versions named', () {
      expect(
        () => ManifestCodec.decode(
          '{"schema_version": 99, "name":"m", "weight_hash":"h", '
          '"inputs":[], "outputs":[]}',
        ),
        throwsA(
          isA<ManifestVersionException>()
              .having((e) => e.found, 'found', 99)
              .having(
                (e) => e.supported,
                'supported',
                ModelManifest.currentSchemaVersion,
              ),
        ),
      );
    });
  });

  group('malformed input names the field', () {
    void expectField(String json, String field, {String? message}) {
      expect(
        () => ManifestCodec.decode(json),
        throwsA(
          isA<ManifestFormatException>()
              .having((e) => e.field, 'field', field)
              .having(
                (e) => e.message,
                'message',
                message == null ? anything : contains(message),
              ),
        ),
      );
    }

    test('a missing required field', () {
      expectField(
        '{"schema_version":1, "name":"m", "inputs":[], "outputs":[]}',
        'weight_hash',
        message: 'absent',
      );
    });

    test('an unknown dtype lists what is understood', () {
      expectField(
        '{"schema_version":1,"name":"m","weight_hash":"h",'
            '"inputs":[{"name":"x","dtype":"float8","shape":[1]}],"outputs":[]}',
        'inputs[0].dtype',
        message: 'unknown dtype "float8"',
      );
    });

    test('a non-integer dimension', () {
      expectField(
        '{"schema_version":1,"name":"m","weight_hash":"h",'
            '"inputs":[{"name":"x","dtype":"float32","shape":[1,"two"]}],'
            '"outputs":[]}',
        'inputs[0].shape[1]',
      );
    });

    test('a negative dimension that is not the dynamic marker', () {
      expectField(
        '{"schema_version":1,"name":"m","weight_hash":"h",'
            '"inputs":[{"name":"x","dtype":"float32","shape":[-3]}],"outputs":[]}',
        'inputs[0].shape[0]',
        message: 'use -1 for dynamic',
      );
    });

    test('a zero standard deviation, which would divide by zero', () {
      expectField(
        '{"schema_version":1,"name":"m","weight_hash":"h",'
            '"inputs":[{"name":"x","dtype":"float32","shape":[1]}],"outputs":[],'
            '"preprocessing":[{"kind":"normalize","mean":[0.5],"std":[0]}]}',
        'preprocessing[0].std',
        message: 'divide by zero',
      );
    });

    test('mean and std of different lengths', () {
      expectField(
        '{"schema_version":1,"name":"m","weight_hash":"h",'
            '"inputs":[{"name":"x","dtype":"float32","shape":[1]}],"outputs":[],'
            '"preprocessing":[{"kind":"normalize","mean":[0.5,0.5],"std":[1]}]}',
        'preprocessing[0]',
        message: '2 entries and std has 1',
      );
    });

    test('a duplicate golden id', () {
      expectField(
        '{"schema_version":1,"name":"m","weight_hash":"h",'
            '"inputs":[{"name":"x","dtype":"float32","shape":[1]}],"outputs":[],'
            '"goldens":[{"id":"a","inputs":[],"outputs":[]},'
            '{"id":"a","inputs":[],"outputs":[]}]}',
        'goldens[1].id',
        message: 'duplicate golden id',
      );
    });

    test('text that is not JSON at all', () {
      expect(
        () => ManifestCodec.decode('not json'),
        throwsA(isA<ManifestFormatException>()),
      );
    });

    test('valid JSON that is not an object', () {
      expect(
        () => ManifestCodec.decode('[1, 2]'),
        throwsA(isA<ManifestFormatException>()),
      );
    });
  });

  group('lookup', () {
    test('a named input is found', () {
      expect(_full.inputNamed('calendar').dtype, DType.int64);
      expect(_full.outputNamed('load_mw').shape, [1, 24]);
    });

    test('a missing name lists what is available', () {
      expect(
        () => _full.inputNamed('windwo'),
        throwsA(
          isA<TensorShapeException>().having(
            (e) => e.message,
            'message',
            allOf(contains('"window"'), contains('"calendar"')),
          ),
        ),
      );
    });
  });

  _layoutTests();
}

void _layoutTests() {
  group('The layout, which says which axes are spatial', () {
    String withLayout(Object? layout, {List<int>? shape}) => jsonEncode({
      'schema_version': 1,
      'name': 'vision',
      'weight_hash': 'sha256:${'0' * 64}',
      'inputs': [
        {
          'name': 'image',
          'dtype': 'float32',
          'shape': shape ?? [1, 3, 8, 8],
          if (layout != null) 'layout': layout,
        },
      ],
      'outputs': [
        {
          'name': 'logits',
          'dtype': 'float32',
          'shape': [1, 2],
        },
      ],
    });

    test('it survives a round trip', () {
      for (final l in TensorLayout.values) {
        final decoded = ManifestCodec.decode(withLayout(l.wireName));
        expect(decoded.inputs.single.layout, l);
        final again = ManifestCodec.decode(ManifestCodec.encode(decoded));
        expect(again.inputs.single.layout, l);
      }
    });

    test('absence is null rather than a default', () {
      // The distinction the whole field exists for. A manifest that says
      // nothing is not a manifest that says NCHW, and a reader that treated it
      // as one would resize down the wrong axes on every tabular model.
      final decoded = ManifestCodec.decode(withLayout(null));
      expect(decoded.inputs.single.layout, isNull);
      final spec =
          (ManifestCodec.toJson(decoded)['inputs'] as List).single
              as Map<String, Object?>;
      expect(spec.containsKey('layout'), isFalse);
    });

    test('an unknown layout is refused, not dropped', () {
      // Dropping it would leave a spec reading "this tensor has no spatial
      // axes", which is a different and wrong claim from "this build cannot
      // tell you which".
      expect(
        () => ManifestCodec.decode(withLayout('nhcw')),
        throwsA(
          isA<ManifestFormatException>().having(
            (e) => e.field,
            'field',
            contains('layout'),
          ),
        ),
      );
    });

    test('a layout on a shape it cannot describe is refused', () {
      expect(
        () => ManifestCodec.decode(withLayout('nchw', shape: [1, 4])),
        throwsA(
          isA<ManifestFormatException>().having(
            (e) => e.message,
            'message',
            contains('cannot say which are spatial'),
          ),
        ),
      );
    });

    test('each layout names the axes it is named after', () {
      expect(TensorLayout.nchw.channelAxis, 1);
      expect(TensorLayout.nchw.heightAxis, 2);
      expect(TensorLayout.nhwc.heightAxis, 1);
      expect(TensorLayout.nhwc.channelAxis, 3);
      for (final l in TensorLayout.values) {
        expect(l.batchAxis, 0);
      }
    });
  });
}
