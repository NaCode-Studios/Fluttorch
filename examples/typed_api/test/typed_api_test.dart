@TestOn('vm')
library;

import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';
import 'package:test/test.dart';
import 'package:typed_api_example/two_layer.fluttorch.dart';

/// A runtime that returns what it is told to.
///
/// The generated API is the thing under test here, not inference: a fake keeps
/// the assertions about types and contracts rather than about numbers, which
/// `fluttorch_test` covers on real backends.
final class FakeRuntime implements FluttorchRuntime {
  FakeRuntime(this.outputs);

  final List<Tensor> outputs;
  Uint8List? loadedArtifact;
  String? requestedBackend;

  @override
  Future<List<RuntimeCapabilities>> capabilities() async => const [
    RuntimeCapabilities(backend: 'fake', dtypes: {DType.float32}),
  ];

  @override
  Future<LoadedModel> load({
    required Uint8List artifact,
    required ModelManifest manifest,
    String? backend,
    bool deterministic = false,
  }) async {
    loadedArtifact = artifact;
    requestedBackend = backend;
    return _FakeModel(manifest, outputs);
  }
}

final class _FakeModel implements LoadedModel {
  _FakeModel(this.manifest, this._outputs);

  @override
  final ModelManifest manifest;
  final List<Tensor> _outputs;

  List<Tensor>? received;
  bool disposed = false;

  @override
  String get backend => 'fake';

  @override
  RuntimeCapabilities get capabilities =>
      const RuntimeCapabilities(backend: 'fake', dtypes: {DType.float32});

  @override
  Future<List<Tensor>> run(List<Tensor> inputs) async {
    checkTensorsAgainst(manifest.inputs, inputs, role: 'inputs');
    received = inputs;
    return _outputs;
  }

  @override
  Future<void> runInto({
    required List<Tensor> inputs,
    required List<Tensor> outputs,
  }) async => throw UnimplementedError();

  @override
  Future<TappedRun> runWithTaps(List<Tensor> inputs, {List<String>? layers}) =>
      throw CapabilityUnavailableException(
        backend: backend,
        capability: 'activation taps',
      );

  @override
  Future<void> dispose() async => disposed = true;
}

Float32List _floats(List<double> v) => Float32List.fromList(v);

Tensor _score(List<double> v) {
  final t = Tensor.zeros(TwoLayerScore.spec);
  t.asFloat32List().setAll(0, v);
  return t;
}

void main() {
  group('the generated tensor types', () {
    test('wrap values of the right length', () {
      final features = TwoLayerFeatures(_floats([1, 2, 3, 4]));
      expect(features.values, [1, 2, 3, 4]);
      expect(features.shape, [1, 4]);
      expect(TwoLayerFeatures.spec.dtype, DType.float32);
    });

    test('refuse values of the wrong length, naming the tensor', () {
      expect(
        () => TwoLayerFeatures(_floats([1, 2, 3])),
        throwsA(
          isA<TensorShapeException>()
              .having((e) => e.tensorName, 'tensorName', 'features')
              .having((e) => e.message, 'message', contains('expected 4')),
        ),
      );
    });

    test('do not copy: the view writes through to the caller\'s buffer', () {
      final buffer = _floats([0, 0, 0, 0]);
      final features = TwoLayerFeatures(buffer);
      features.values[2] = 9;
      expect(buffer[2], 9, reason: 'the wrapper is a view, not a copy');
    });

    test('wrap() accepts a tensor that already satisfies the spec', () {
      final t = Tensor.zeros(TwoLayerFeatures.spec);
      expect(() => TwoLayerFeatures.wrap(t), returnsNormally);
    });

    test('wrap() rejects one that does not', () {
      const other = TensorSpec(
        name: 'features',
        dtype: DType.float32,
        shape: [1, 5],
      );
      expect(
        () => TwoLayerFeatures.wrap(Tensor.zeros(other)),
        throwsA(isA<TensorShapeException>()),
      );
    });
  });

  group('the generated model', () {
    late FakeRuntime runtime;
    late Uint8List artifact;

    setUp(() {
      runtime = FakeRuntime([
        _score([0.1, 0.2, 0.7]),
      ]);
      // The artifact the manifest was generated for is the committed export;
      // any other bytes must be refused.
      artifact = Uint8List.fromList(
        // Not the real .pte — the point is that load() checks.
        List.filled(16, 7),
      );
    });

    test('refuses an artifact that is not the one it was generated for', () {
      expect(
        () => TwoLayer.load(runtime, artifact: artifact),
        throwsA(
          isA<ArtifactMismatchException>().having(
            (e) => e.expectedHash,
            'expected',
            TwoLayer.weightHash,
          ),
        ),
      );
    });

    test('carries the contract it was generated from', () {
      expect(TwoLayer.manifest.name, 'two_layer');
      expect(TwoLayer.manifest.inputs.single.name, 'features');
      expect(TwoLayer.weightHash, TwoLayer.manifest.weightHash);
    });

    test('the embedded manifest is the one on disk', () {
      // Re-encoding proves the embedding is the document and not a summary.
      expect(ManifestCodec.encode(TwoLayer.manifest), TwoLayer.manifestJson);
    });

    test('run passes the tensor through and types the result', () async {
      // wrap() rather than load(): this test is about the generated call shape,
      // and the artifact check has a test of its own above.
      final loaded = await runtime.load(
        artifact: Uint8List(0),
        manifest: TwoLayer.manifest,
      );
      final outputs = await TwoLayer.wrap(
        loaded,
      ).run(features: TwoLayerFeatures(_floats([1, 2, 3, 4])));

      expect(outputs.score.values, [
        closeTo(0.1, 1e-7),
        closeTo(0.2, 1e-7),
        closeTo(0.7, 1e-7),
      ]);
      expect(outputs.score.shape, [1, 3]);
      expect(TwoLayerOutputs.labels, ['low', 'mid', 'high']);
    });
  });

  test('wrap refuses a model built from a different export', () async {
    final other = ModelManifest(
      name: 'two_layer',
      schemaVersion: ModelManifest.currentSchemaVersion,
      weightHash: 'sha256:${'1' * 64}',
      inputs: TwoLayer.manifest.inputs,
      outputs: TwoLayer.manifest.outputs,
    );
    final loaded = await FakeRuntime(
      const [],
    ).load(artifact: Uint8List(0), manifest: other);
    expect(
      () => TwoLayer.wrap(loaded),
      throwsA(isA<ArtifactMismatchException>()),
    );
  });

  test('labels come from the export, in order', () {
    expect(TwoLayerOutputs.labels, hasLength(3));
    expect(TwoLayerOutputs.labels.first, 'low');
  });

  // Not testable at run time, and the point of the milestone: the following
  // does not compile, because TwoLayerScore is not a TwoLayerFeatures.
  //
  //   await model.run(features: TwoLayerScore(_floats([1, 2, 3])));
  //
  // Nor does passing a bare Tensor. That is what the wrapper types buy, and it
  // costs nothing at run time.
}
