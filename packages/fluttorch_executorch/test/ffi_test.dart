@TestOn('vm')
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';
import 'package:fluttorch_executorch/fluttorch_executorch.dart';
import 'package:fluttorch_executorch/src/ffi.dart';
import 'package:test/test.dart';

/// The binding against a real shared library, compiled here.
///
/// `runtime_test.dart` proves the logic above the seam with a Dart fake. This
/// proves the seam itself, which no Dart fake can: struct field offsets, arrays
/// of strings, pointer arithmetic over tensor arrays, and whether the bytes the
/// callee wrote are the bytes Dart reads back. A wrong offset here produces
/// garbage rather than a compile error, which is exactly why it is asserted
/// against a compiler rather than reasoned about.
///
/// What it does not prove is that ExecuTorch runs a model. That needs the
/// toolchain and a device, and this file makes no claim about it.
late String _library;

Tensor _f32(String name, List<double> values) {
  final spec = TensorSpec(
    name: name,
    dtype: DType.float32,
    shape: [values.length],
  );
  final t = Tensor.zeros(spec);
  t.asFloat32List().setAll(0, values);
  return t;
}

final _artifact = Uint8List.fromList(List.generate(64, (i) => i));

ModelManifest _manifest() => ModelManifest(
  name: 'stub',
  schemaVersion: 1,
  weightHash: digestOf(_artifact),
  inputs: const [
    TensorSpec(name: 'features', dtype: DType.float32, shape: [2]),
  ],
  outputs: const [
    TensorSpec(name: 'score', dtype: DType.float32, shape: [2]),
  ],
  activations: const [
    TensorSpec(name: 'encoder.0', dtype: DType.float32, shape: [2]),
    TensorSpec(name: 'encoder.1', dtype: DType.float32, shape: [2]),
  ],
  goldens: const [
    GoldenCase(
      id: 'case-0',
      inputKeys: ['a'],
      outputKeys: ['b'],
      activationKeys: ['c', 'd'],
    ),
  ],
);

void main() {
  setUpAll(() {
    final dir = Directory.systemTemp.createTempSync('fluttorch_stub');
    _library =
        '${dir.path}/libfluttorch_stub${Platform.isMacOS ? ".dylib" : ".so"}';
    final result = Process.runSync('cc', [
      '-shared',
      '-fPIC',
      '-o',
      _library,
      'test/stub/fluttorch_stub.c',
    ]);
    expect(
      result.exitCode,
      0,
      reason: 'the stub did not compile: ${result.stderr}',
    );
  });

  group('M20 · the boundary itself', () {
    test('a list of C strings comes back as a list of Dart strings', () {
      final bindings = NativeExecuTorchBindings.open(_library);

      expect(bindings.backends(), ['xnnpack', 'coreml']);
    });

    test('every field of the capability struct reads at the right offset', () {
      final caps = NativeExecuTorchBindings.open(
        _library,
      ).capabilitiesOf('xnnpack');

      // Each of these is a different offset into the struct. A layout that
      // disagreed with the header would not fail here, it would return
      // plausible nonsense, so all five are asserted rather than one.
      expect(caps.backend, 'xnnpack');
      expect(caps.supportsTaps, isTrue);
      expect(caps.supportsDeterminism, isTrue);
      expect(caps.maxTensorBytes, 1048576);
      expect(caps.dtypes, {DType.float32, DType.int8, DType.boolean});
    });

    test('the dtype mask means the same thing in C and in Dart', () {
      final caps = NativeExecuTorchBindings.open(
        _library,
      ).capabilitiesOf('coreml');

      // The C side wrote bits 0, 4 and 9 with no way to ask Dart what its enum
      // order is. The ordering is the contract, and this is where it is checked.
      expect(
        NativeCapabilities.maskOf(caps.dtypes),
        (1 << DType.float32.index) |
            (1 << DType.int8.index) |
            (1 << DType.boolean.index),
      );
    });

    test('capabilities differ per backend rather than per build', () {
      final bindings = NativeExecuTorchBindings.open(_library);

      expect(bindings.capabilitiesOf('xnnpack').supportsTaps, isTrue);
      expect(bindings.capabilitiesOf('coreml').supportsTaps, isFalse);
    });

    test('an unknown backend arrives as the typed failure, with the list', () {
      final bindings = NativeExecuTorchBindings.open(_library);

      expect(
        () => bindings.capabilitiesOf('vulkan'),
        throwsA(
          isA<BackendUnavailableException>().having(
            (e) => e.message,
            'message',
            allOf(contains('vulkan'), contains('xnnpack')),
          ),
        ),
      );
    });

    test(
      'a truncated artifact is refused with the reason the library gave',
      () {
        final bindings = NativeExecuTorchBindings.open(_library);

        expect(
          () => bindings.load(artifact: Uint8List.fromList([1, 2])),
          throwsA(
            isA<FluttorchException>().having(
              (e) => e.message,
              'message',
              contains('shorter than any header'),
            ),
          ),
        );
      },
    );

    test('the bytes that go out are the bytes that come back', () {
      final model = NativeExecuTorchBindings.open(
        _library,
      ).load(artifact: _artifact, backend: 'xnnpack');
      final input = _f32('features', [1.5, -2.25]);
      final output = Tensor.zeros(
        const TensorSpec(name: 'score', dtype: DType.float32, shape: [2]),
      );

      model.run([input], [output]);

      // The library fills every output byte with the sum of every input byte.
      // A binding that marshalled the wrong bytes would produce a different
      // number rather than the right one by luck.
      final expected = input.bytes.fold<int>(0, (a, b) => (a + b) & 0xff);
      expect(output.bytes, everyElement(expected));
      model.dispose();
    });

    test('a tap the graph does not carry is absent, not zeroed', () {
      final model = NativeExecuTorchBindings.open(
        _library,
      ).load(artifact: _artifact, backend: 'xnnpack');
      final output = Tensor.zeros(
        const TensorSpec(name: 'score', dtype: DType.float32, shape: [2]),
      );

      final activations = model.runWithTaps(
        [
          _f32('features', [1, 2]),
        ],
        [output],
        ['encoder.0', 'encoder.1'],
      );

      expect(activations.keys, ['encoder.0']);
      expect(activations['encoder.0']!.shape, [2]);
      expect(activations['encoder.0']!.spec.dtype, DType.float32);
      model.dispose();
    });

    test('a backend without taps refuses rather than returning nothing', () {
      final model = NativeExecuTorchBindings.open(
        _library,
      ).load(artifact: _artifact, backend: 'coreml');
      final output = Tensor.zeros(
        const TensorSpec(name: 'score', dtype: DType.float32, shape: [2]),
      );

      expect(
        () => model.runWithTaps(
          [
            _f32('features', [1, 2]),
          ],
          [output],
          const ['encoder.0'],
        ),
        throwsA(isA<FluttorchException>()),
      );
      model.dispose();
    });
  });

  group('M20 · the whole stack over a real library', () {
    test('load, pin a backend, run into a caller buffer', () async {
      final runtime = ExecuTorchRuntime(
        NativeExecuTorchBindings.open(_library),
      );
      final manifest = _manifest();

      final model = await runtime.load(
        artifact: _artifact,
        manifest: manifest,
        backend: 'xnnpack',
        deterministic: true,
      );
      addTearDown(model.dispose);

      expect(model.backend, 'xnnpack');
      expect(model.capabilities.supportsDeterministicExecution, isTrue);

      final input = _f32('features', [3, 4]);
      final output = Tensor.zeros(manifest.outputs.single);
      await model.runInto(inputs: [input], outputs: [output]);

      final expected = input.bytes.fold<int>(0, (a, b) => (a + b) & 0xff);
      expect(output.bytes, everyElement(expected));
    });

    test('a wrong artifact never reaches the library', () async {
      final runtime = ExecuTorchRuntime(
        NativeExecuTorchBindings.open(_library),
      );

      await expectLater(
        runtime.load(
          artifact: Uint8List.fromList(List.filled(64, 7)),
          manifest: _manifest(),
        ),
        throwsA(isA<ArtifactMismatchException>()),
      );
    });

    test('attribution runs end to end against the compiled library', () async {
      final runtime = ExecuTorchRuntime(
        NativeExecuTorchBindings.open(_library),
      );
      final manifest = _manifest();
      final model = await runtime.load(
        artifact: _artifact,
        manifest: manifest,
        backend: 'xnnpack',
      );
      addTearDown(model.dispose);

      final run = await model.runWithTaps([
        _f32('features', [1, 2]),
      ]);

      expect(run.outputs.single.shape, [2]);
      expect(run.activations.keys, ['encoder.0']);
    });
  });
}
