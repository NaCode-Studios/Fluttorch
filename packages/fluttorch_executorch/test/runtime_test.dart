import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';
import 'package:fluttorch_executorch/fluttorch_executorch.dart';
import 'package:test/test.dart';

/// Bindings under the test's control.
///
/// Everything below the seam is the part that has to be compiled against
/// ExecuTorch. Standing in for it here is what lets the half above be finished
/// and proven before the half below exists, which is the reason the seam is a
/// header rather than a pile of FFI calls in the runtime.
final class FakeBindings implements ExecuTorchBindings {
  FakeBindings({
    this.available = const ['xnnpack', 'coreml'],
    this.taps = false,
    this.determinism = false,
    this.dtypes = const {DType.float32},
    this.fallbackTo,
  });

  final List<String> available;
  final bool taps;
  final bool determinism;
  final Set<DType> dtypes;

  /// What the native side settles on when it cannot honour the request.
  final String? fallbackTo;

  String? requestedBackend;
  bool? requestedDeterminism;

  @override
  List<String> backends() => available;

  @override
  NativeCapabilities capabilitiesOf(String? backend) {
    final name = backend ?? available.first;
    if (!available.contains(name)) {
      throw BackendUnavailableException(requested: name, available: available);
    }
    return NativeCapabilities(
      backend: name,
      dtypes: dtypes,
      supportsTaps: taps,
      supportsDeterminism: determinism,
      maxTensorBytes: 1 << 20,
    );
  }

  @override
  NativeModel load({
    required Uint8List artifact,
    String? backend,
    bool deterministic = false,
  }) {
    requestedBackend = backend;
    requestedDeterminism = deterministic;
    if (deterministic && !determinism) {
      throw CapabilityUnavailableException(
        backend: backend ?? available.first,
        capability: 'deterministic execution',
      );
    }
    return FakeModel(
      capabilities: capabilitiesOf(fallbackTo ?? backend),
      taps: taps,
    );
  }
}

final class FakeModel implements NativeModel {
  FakeModel({required this.capabilities, required bool taps}) : _taps = taps;

  @override
  final NativeCapabilities capabilities;

  final bool _taps;

  int runs = 0;
  bool disposed = false;

  @override
  String get backend => capabilities.backend;

  @override
  void run(List<Tensor> inputs, List<Tensor> outputs) {
    runs++;
    // Writes something recognisable so a test can prove the caller's buffer was
    // the one filled, rather than a copy the binding kept.
    outputs.first.asFloat32List().setAll(0, [1, 2, 3]);
  }

  @override
  Set<int> runWithTaps(
    List<Tensor> inputs,
    List<Tensor> outputs,
    List<Tensor> activations,
    List<int> handles,
  ) {
    run(inputs, outputs);
    if (!_taps) return const {};
    tappedHandles = handles;
    // Everything but the last is filled. A graph that does not run a requested
    // tap leaves its buffer as the caller supplied it rather than zero-filling
    // it, which would read as a layer that agreed.
    return {for (var i = 0; i < handles.length - 1; i++) i};
  }

  /// The handles the runtime resolved, so a test can prove it read them from the
  /// manifest rather than passing positions or names.
  List<int>? tappedHandles;

  @override
  void dispose() => disposed = true;
}

final _artifact = Uint8List.fromList([1, 2, 3, 4]);

ModelManifest _manifest({
  List<TensorSpec> outputs = const [
    TensorSpec(name: 'score', dtype: DType.float32, shape: [3]),
  ],
}) => ModelManifest(
  name: 'two_layer',
  schemaVersion: 1,
  weightHash: digestOf(_artifact),
  inputs: const [
    TensorSpec(name: 'features', dtype: DType.float32, shape: [2]),
  ],
  outputs: outputs,
  activations: const [
    TensorSpec(name: 'encoder.0', dtype: DType.float32, shape: [2]),
    TensorSpec(name: 'encoder.1', dtype: DType.float32, shape: [2]),
  ],
  // Where the export found each tap in the lowered graph. The second is
  // negative, which the stub reads as a layer the graph does not run.
  activationHandles: const [7, -1],
  goldens: const [
    GoldenCase(
      id: 'case-0',
      inputKeys: ['a'],
      outputKeys: ['b'],
      activationKeys: ['c', 'd'],
    ),
  ],
);

Tensor _in() => Tensor.zeros(
  const TensorSpec(name: 'features', dtype: DType.float32, shape: [2]),
);

void main() {
  group('M20 · loading', () {
    test('the artifact is checked before anything native sees it', () async {
      final bindings = FakeBindings();
      final runtime = ExecuTorchRuntime(bindings);

      await expectLater(
        runtime.load(
          artifact: Uint8List.fromList([9, 9, 9]),
          manifest: _manifest(),
        ),
        throwsA(isA<ArtifactMismatchException>()),
      );
      expect(
        bindings.requestedBackend,
        isNull,
        reason: 'nothing was loaded natively',
      );
    });

    test('a backend that does not exist is refused with the list', () async {
      final runtime = ExecuTorchRuntime(FakeBindings());

      await expectLater(
        runtime.load(
          artifact: _artifact,
          manifest: _manifest(),
          backend: 'vulkan',
        ),
        throwsA(
          isA<BackendUnavailableException>().having(
            (e) => e.message,
            'message',
            allOf(contains('vulkan'), contains('xnnpack')),
          ),
        ),
      );
    });

    test('the backend asked for is the backend pinned', () async {
      final bindings = FakeBindings();
      final model = await ExecuTorchRuntime(
        bindings,
      ).load(artifact: _artifact, manifest: _manifest(), backend: 'coreml');

      expect(bindings.requestedBackend, 'coreml');
      expect(model.backend, 'coreml');
    });

    test(
      'a fallback reports the backend that ran, not the one requested',
      () async {
        final bindings = FakeBindings(fallbackTo: 'xnnpack');
        final model = await ExecuTorchRuntime(
          bindings,
        ).load(artifact: _artifact, manifest: _manifest(), backend: 'coreml');

        expect(model.backend, 'xnnpack');
        expect(model.capabilities.backend, 'xnnpack');
      },
    );

    test('determinism is refused rather than quietly dropped', () async {
      final runtime = ExecuTorchRuntime(FakeBindings(determinism: false));

      await expectLater(
        runtime.load(
          artifact: _artifact,
          manifest: _manifest(),
          deterministic: true,
        ),
        throwsA(
          isA<CapabilityUnavailableException>().having(
            (e) => e.capability,
            'capability',
            contains('deterministic'),
          ),
        ),
      );
    });

    test('determinism is passed through when the backend has it', () async {
      final bindings = FakeBindings(determinism: true);
      await ExecuTorchRuntime(
        bindings,
      ).load(artifact: _artifact, manifest: _manifest(), deterministic: true);

      expect(bindings.requestedDeterminism, isTrue);
    });

    test('a backend that cannot carry the model types fails at load', () async {
      final runtime = ExecuTorchRuntime(
        FakeBindings(dtypes: const {DType.int8}),
      );

      await expectLater(
        runtime.load(artifact: _artifact, manifest: _manifest()),
        throwsA(isA<DTypeMismatchException>()),
      );
    });
  });

  group('M20 · running', () {
    test('runInto writes into the buffer the caller owns', () async {
      final model = await ExecuTorchRuntime(
        FakeBindings(),
      ).load(artifact: _artifact, manifest: _manifest());
      final out = Tensor.zeros(
        const TensorSpec(name: 'score', dtype: DType.float32, shape: [3]),
      );

      await model.runInto(inputs: [_in()], outputs: [out]);

      expect(out.asFloat32List(), [1, 2, 3]);
    });

    test(
      'an output whose extent is unknown is refused with the reason',
      () async {
        final model = await ExecuTorchRuntime(FakeBindings()).load(
          artifact: _artifact,
          manifest: _manifest(
            outputs: const [
              TensorSpec(
                name: 'score',
                dtype: DType.float32,
                shape: [TensorSpec.dynamicDim],
              ),
            ],
          ),
        );

        await expectLater(
          model.run([_in()]),
          throwsA(
            isA<TensorShapeException>().having(
              (e) => e.message,
              'message',
              contains('runInto'),
            ),
          ),
        );
      },
    );

    test('taps are refused on a backend without them', () async {
      final model = await ExecuTorchRuntime(
        FakeBindings(taps: false),
      ).load(artifact: _artifact, manifest: _manifest());

      await expectLater(
        model.runWithTaps([_in()]),
        throwsA(isA<CapabilityUnavailableException>()),
      );
    });

    test('a tap the graph does not carry stays absent', () async {
      final model = await ExecuTorchRuntime(
        FakeBindings(taps: true),
      ).load(artifact: _artifact, manifest: _manifest());

      final run = await model.runWithTaps([_in()]);

      expect(run.activations.keys, ['encoder.0']);
      expect(
        run.activations.containsKey('encoder.1'),
        isFalse,
        reason: 'absent, not zero-filled: the gate reads a zero as agreement',
      );
      expect(run.outputs.single.asFloat32List(), [1, 2, 3]);
    });

    test(
      'the layers requested default to the ones the export tapped',
      () async {
        final model = await ExecuTorchRuntime(
          FakeBindings(taps: true),
        ).load(artifact: _artifact, manifest: _manifest());

        final run = await model.runWithTaps([_in()]);

        expect(run.activations, isNotEmpty);
      },
    );
  });

  group('M20 · capabilities', () {
    test('every compiled backend is reported', () async {
      final caps = await ExecuTorchRuntime(FakeBindings()).capabilities();

      expect(caps.map((c) => c.backend), ['xnnpack', 'coreml']);
      expect(caps.every((c) => c.maxTensorBytes == 1 << 20), isTrue);
    });

    test('the dtype mask survives the round trip through the C struct', () {
      const types = {DType.float32, DType.int8, DType.boolean};

      expect(
        NativeCapabilities.dtypesFromMask(NativeCapabilities.maskOf(types)),
        types,
      );
    });
  });

  group('M27 · the runtime a bundle was written for', () {
    test(
      'an artifact for another engine is refused before it is loaded',
      () async {
        // A .onnx satisfies the weight hash, because the hash was computed over
        // whichever artifact was written, and then fails somewhere in ExecuTorch's
        // loader with a message about bytes.
        final manifest = _manifest().copyForRuntime('onnx');
        await expectLater(
          ExecuTorchRuntime(
            FakeBindings(),
          ).load(artifact: _artifact, manifest: manifest),
          throwsA(isA<BackendUnavailableException>()),
        );
      },
    );
  });
}

extension on ModelManifest {
  /// The same manifest, rewritten for another engine.
  ///
  /// A copyWith would be a wider API than one test needs, and the manifest is
  /// deliberately without one: it describes an export, and an export is not a
  /// thing callers edit.
  ModelManifest copyForRuntime(String runtime) => ModelManifest(
    name: name,
    schemaVersion: schemaVersion,
    weightHash: weightHash,
    inputs: inputs,
    outputs: outputs,
    runtime: runtime,
    goldens: goldens,
  );
}
