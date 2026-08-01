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
/// and `testdata/` carries the exports to measure.
///
/// What runs here and nowhere else is the whole claim end to end: a model from
/// our exporter, loaded through this binding, executed by the delegate it was
/// lowered for, and compared against references captured from the source model
/// before it was lowered.
final _library = File(
  '.dart_tool/native/libfluttorch_executorch'
  '${Platform.isMacOS ? ".dylib" : ".so"}',
);
final _quantized = Directory('../../testdata/quantized');
final _coreml = Directory('../../testdata/coreml');

void main() {
  if (!_library.existsSync() || !_quantized.existsSync()) {
    // Named rather than silent: a suite that skips without saying so is a suite
    // that has quietly stopped covering the thing it exists for.
    test('the on-device gate needs tool/build_native.sh', () {}, skip: true);
    return;
  }

  final bindings = NativeExecuTorchBindings.open(_library.path);

  group('M21 · a quantized model on this machine', () {
    late DirectoryGoldenBundle goldens;
    late LoadedModel model;

    setUpAll(() async {
      goldens = await DirectoryGoldenBundle.open(
        '${_quantized.path}/two_layer.fluttorch.json',
      );
      model = await ExecuTorchRuntime(bindings).load(
        artifact: await File('${_quantized.path}/two_layer.pte').readAsBytes(),
        manifest: goldens.manifest,
        backend: 'xnnpack',
      );
    });

    tearDownAll(() async => model.dispose());

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

  // Which delegates this build has is a property of the ExecuTorch checkout it
  // was linked against, not of the machine, so the library is asked rather than
  // the platform. `tool/build_native.sh` links Core ML where the checkout can
  // supply it and says which of the two it did.
  if (!bindings.backends().contains('coreml') || !_coreml.existsSync()) {
    test(
      'the Core ML gate needs an ExecuTorch built with it',
      () {},
      skip: true,
    );
    return;
  }

  group('M21 · the same model through Core ML', () {
    late DirectoryGoldenBundle goldens;
    late LoadedModel model;

    setUpAll(() async {
      goldens = await DirectoryGoldenBundle.open(
        '${_coreml.path}/two_layer.fluttorch.json',
      );
      model = await ExecuTorchRuntime(bindings).load(
        artifact: await File('${_coreml.path}/two_layer.pte').readAsBytes(),
        manifest: goldens.manifest,
        backend: 'coreml',
      );
    });

    tearDownAll(() async => model.dispose());

    test('it loads on the backend it was lowered for', () {
      expect(model.backend, 'coreml');
      // Nothing quantized it, and the manifest saying so is load-bearing: the
      // exporter pins Core ML to float32 precisely because a manifest that says
      // nothing is read as full precision, and Core ML's own default is float16.
      // So the bound this answers to is the one the quantized run above fails.
      expect(model.manifest.quantization, isNull);
    });

    test('every golden holds at the full-precision bound', () async {
      // No tolerance is passed: with no recipe in the manifest the gate falls
      // back to the full-precision starting point on its own, which is the
      // bound worth holding a delegate to that only re-ordered the graph.
      await expectParity(model, goldens: goldens);
    });

    test('the delegate answered, and every case was measured', () async {
      final reports = await measureParity(model, goldens: goldens);

      expect(reports, hasLength(goldens.cases.length));
      expect(reports.every((r) => r.passes), isTrue);
      // Core ML runs in float32 here, so agreement can be exact. What would not
      // be exact is a delegate that quietly fell back to something else, and
      // the bound above is what would catch it.
      expect(
        reports.every((r) => r.tensors.single.maxAbsolute.isFinite),
        isTrue,
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
