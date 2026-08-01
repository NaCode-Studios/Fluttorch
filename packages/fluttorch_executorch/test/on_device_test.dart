@TestOn('vm')
library;

import 'dart:io';

import 'package:fluttorch/fluttorch.dart';
import 'package:fluttorch_executorch/fluttorch_executorch.dart';
import 'package:fluttorch_executorch/src/ffi.dart';
import 'package:fluttorch_test/fluttorch_test.dart';
import 'package:fluttorch_test/io.dart';
import 'package:test/test.dart';

/// The gate against ExecuTorch, on whatever machine runs this.
///
/// Skipped where the native library has not been built, which includes CI:
/// building ExecuTorch takes an hour and several gigabytes, and a suite that
/// demanded it would be a suite nobody runs. `tool/build_native.sh` builds it,
/// and `testdata/quantized/` carries an export to measure.
///
/// What runs here and nowhere else is the whole claim end to end: a model
/// quantized by the exporter, loaded through this binding, executed by XNNPACK,
/// and compared against references captured from the source model before it was
/// lowered.
final _library = File(
  '.dart_tool/native/libfluttorch_executorch'
  '${Platform.isMacOS ? ".dylib" : ".so"}',
);
final _export = Directory('../../testdata/quantized');

void main() {
  if (!_library.existsSync() || !_export.existsSync()) {
    // Named rather than silent: a suite that skips without saying so is a suite
    // that has quietly stopped covering the thing it exists for.
    test('the on-device gate needs tool/build_native.sh', () {}, skip: true);
    return;
  }

  late DirectoryGoldenBundle goldens;
  late LoadedModel model;

  setUpAll(() async {
    goldens = await DirectoryGoldenBundle.open(
      '${_export.path}/two_layer.fluttorch.json',
    );
    model =
        await ExecuTorchRuntime(
          NativeExecuTorchBindings.open(_library.path),
        ).load(
          artifact: await File('${_export.path}/two_layer.pte').readAsBytes(),
          manifest: goldens.manifest,
          backend: 'xnnpack',
        );
  });

  tearDownAll(() async => model.dispose());

  group('M21 · a quantized model on this machine', () {
    test('it loads on the backend it was pinned to', () {
      expect(model.backend, 'xnnpack');
      expect(model.manifest.quantization, 'int8-dynamic');
    });

    test(
      'every golden is inside the tolerance its recipe starts from',
      () async {
        await expectParity(model, goldens: goldens);
      },
    );

    test('the drift is real, and the gate is not passing on nothing', () async {
      final reports = await measureParity(model, goldens: goldens);

      // Quantization moved the numbers. A gate reporting zero drift on an int8
      // model would be measuring something other than the model.
      expect(reports.every((r) => r.tensors.single.maxAbsolute > 0), isTrue);
      // And it is small enough to be quantization rather than a broken export.
      expect(reports.every((r) => r.tensors.single.maxAbsolute < 0.01), isTrue);
    });

    test('held to a full-precision bound, the same run fails', () async {
      // The bound is the whole product. If the recipe's tolerance passed and a
      // hundred times tighter one also passed, the gate would be decorative.
      final reports = await measureParity(
        model,
        goldens: goldens,
        tolerance: Tolerance.startingPointFor(null),
      );

      expect(reports.every((r) => !r.passes), isTrue);
      expect(
        reports.first.describe(),
        contains('exceed the elementwise bound'),
      );
    });

    test(
      'taps are reported absent rather than answered with nothing',
      () async {
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
      },
    );
  });
}
