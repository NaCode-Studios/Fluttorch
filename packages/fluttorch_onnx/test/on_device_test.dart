@TestOn('vm')
library;

import 'dart:io';

import 'package:fluttorch/fluttorch.dart';
import 'package:fluttorch_onnx/fluttorch_onnx.dart';
import 'package:fluttorch_test/fluttorch_test.dart';
import 'package:fluttorch_test/io.dart';
import 'package:test/test.dart';

/// The gate against ONNX Runtime, on whatever machine runs this.
///
/// Skipped where the shim has not been built, which includes CI: it needs the
/// ONNX Runtime SDK, and a suite that demanded one would be a suite nobody runs.
/// `tool/build_native.sh` builds it and `testdata/onnx/` carries an export.
///
/// What this proves that nothing else does is that the seam is a seam. The same
/// manifest, the same goldens and the same gate, over a different engine.
final _library = File(
  '.dart_tool/native/libfluttorch_onnx'
  '${Platform.isMacOS ? ".dylib" : ".so"}',
);
final _export = Directory('../../testdata/onnx');

void main() {
  if (!_library.existsSync() || !_export.existsSync()) {
    test('the ONNX gate needs tool/build_native.sh', () {}, skip: true);
    return;
  }

  late DirectoryGoldenBundle goldens;
  late LoadedModel model;
  final runtime = OnnxRuntime.open(_library.path);

  setUpAll(() async {
    goldens = await DirectoryGoldenBundle.open(
      '${_export.path}/two_layer.fluttorch.json',
    );
    model = await runtime.load(
      artifact: await File('${_export.path}/two_layer.onnx').readAsBytes(),
      manifest: goldens.manifest,
      backend: 'cpu',
    );
  });

  tearDownAll(() async => model.dispose());

  group('M27 · a model through ONNX Runtime', () {
    test('it reports the providers this build carries', () async {
      final caps = await runtime.capabilities();
      expect(caps.map((c) => c.backend), contains('cpu'));
      expect(model.backend, 'cpu');
    });

    test('the manifest names the engine, not just the provider', () {
      expect(model.manifest.runtime, 'onnx');
      // Two axes, and the gate needs both: this one decides which binding can
      // load the artifact at all, and the backend decides what runs it.
      expect(model.manifest.quantization, isNull);
    });

    test('every golden holds at the full-precision bound', () async {
      // The same references the ExecuTorch exports are measured against, byte
      // for byte, so a difference here would be the engine and nothing else.
      await expectParity(model, goldens: goldens);
    });

    test(
      'the drift is float32 rounding rather than a different model',
      () async {
        final reports = await measureParity(model, goldens: goldens);
        expect(reports, hasLength(goldens.cases.length));
        expect(reports.every((r) => r.passes), isTrue);
        expect(
          reports.every((r) => r.tensors.single.maxRelative < 1e-5),
          isTrue,
          reason: reports.map((r) => r.describe()).join(),
        );
      },
    );

    test('CPU can promise a repeatable reduction order', () {
      expect(model.capabilities.supportsDeterministicExecution, isTrue);
    });

    test('taps are refused with what would have to change', () async {
      // Absent for a structural reason rather than a missing feature: reading an
      // intermediate here means naming it as a graph output, which is a property
      // of the artifact and not of the run.
      expect(model.capabilities.supportsActivationTaps, isFalse);
      await expectLater(
        model.runWithTaps([
          await goldens.tensor(
            goldens.cases.first.inputKeys.single,
            goldens.manifest.inputs.single,
          ),
        ]),
        throwsA(isA<CapabilityUnavailableException>()),
      );
    });

    test('an ExecuTorch bundle is refused rather than parsed', () async {
      // The failure the runtime field exists to prevent, from the other side.
      final other = await DirectoryGoldenBundle.open(
        '../../testdata/quantized/two_layer.fluttorch.json',
      );
      await expectLater(
        runtime.load(
          artifact: await File(
            '../../testdata/quantized/two_layer.pte',
          ).readAsBytes(),
          manifest: other.manifest,
        ),
        throwsA(isA<BackendUnavailableException>()),
      );
    });
  });
}
