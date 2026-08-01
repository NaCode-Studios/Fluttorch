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
final _taps = Directory('../../testdata/taps');
final _mps = Directory('../../testdata/mps');

void main() {
  if (!_library.existsSync() || !_quantized.existsSync()) {
    // Named rather than silent: a suite that skips without saying so is a suite
    // that has quietly stopped covering the thing it exists for.
    test('the on-device gate needs tool/build_native.sh', () {}, skip: true);
    return;
  }

  final bindings = NativeExecuTorchBindings.open(_library.path);

  // M24 · every backend this machine has, measured on one set of goldens.
  //
  // Built from what is present rather than from a fixed list, which is the only
  // way the matrix survives contact with a second machine: the same suite has
  // four columns here and one on a laptop that never built a delegate.
  group('M24 · the same goldens across every backend this machine has', () {
    final candidates = <String, Directory>{
      'xnnpack': _quantized,
      'portable': _taps,
      'coreml': _coreml,
      'mps': _mps,
    };

    late ParityMatrix matrix;
    final loaded = <LoadedModel>[];

    setUpAll(() async {
      final entries = <MatrixEntry>[];
      for (final entry in candidates.entries) {
        if (!entry.value.existsSync()) continue;
        if (!bindings.backends().contains(entry.key)) continue;
        final goldens = await DirectoryGoldenBundle.open(
          '${entry.value.path}/two_layer.fluttorch.json',
        );
        final model = await ExecuTorchRuntime(bindings).load(
          artifact: await File(
            '${entry.value.path}/two_layer.pte',
          ).readAsBytes(),
          manifest: goldens.manifest,
          backend: entry.key,
        );
        loaded.add(model);
        entries.add(MatrixEntry(model: model, goldens: goldens));
      }
      matrix = await measureMatrix(entries);
    });

    tearDownAll(() async {
      for (final m in loaded) {
        await m.dispose();
      }
    });

    test('it covers more than one backend, or it is not a matrix', () {
      expect(matrix.backends.length, greaterThan(1));
      expect(matrix.goldenIds, hasLength(4));
      // Every cell measured. A hole reads as agreement in a table, which is the
      // thing this whole package exists to stop.
      for (final backend in matrix.backends) {
        for (final id in matrix.goldenIds) {
          expect(
            matrix.at(backend: backend, goldenId: id),
            isNotNull,
            reason: '$backend/$id',
          );
        }
      }
    });

    test('every backend holds the bound its own export implies', () {
      expect(matrix.passes, isTrue, reason: matrix.describe());
    });

    test('the quantized column is the one that moved', () {
      // The matrix earns its place here. Three backends carry the model at full
      // precision and agree with the source to within float32; one carries it at
      // int8 and does not. A table where every column read the same would mean
      // the quantized artifact was not quantized.
      final exact = matrix.backends.where((b) => b != 'xnnpack');
      for (final id in matrix.goldenIds) {
        final quantized = matrix
            .at(backend: 'xnnpack', goldenId: id)!
            .tensors
            .single
            .maxRelative;
        for (final backend in exact) {
          final other = matrix
              .at(backend: backend, goldenId: id)!
              .tensors
              .single
              .maxRelative;
          expect(
            quantized,
            greaterThan(other),
            reason: '$id: xnnpack(int8) should move more than $backend',
          );
        }
      }
    });

    test('the report names every backend and every golden', () {
      final report = matrix.describe();
      expect(report, startsWith('PASS'));
      for (final backend in matrix.backends) {
        expect(report, contains(backend));
      }
      for (final id in matrix.goldenIds) {
        expect(report, contains(id));
      }
      printOnFailure(report);
    });
  });

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
      expect(reports.every((r) => r.tensors.single.maxRelative > 0), isTrue);
      // And it is small enough to be quantization rather than a broken export.
      //
      // Relative rather than absolute, because absolute is not comparable across
      // these goldens: one case feeds the model inputs of magnitude 1e3 and its
      // outputs are around 175, where an absolute bound of 0.01 asks for six
      // significant figures out of int8. Relative asks the same question of
      // every case.
      expect(reports.every((r) => r.tensors.single.maxRelative < 0.05), isTrue);
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

    test('the build can read intermediates, this artifact has none', () async {
      // Two different questions, and conflating them is what would make a gate
      // report agreement it never measured. The build compiles the tracer, so
      // it can read any op the runtime executes itself. This artifact is one
      // delegated partition, so it executes none, and the export declared no
      // taps for exactly that reason.
      expect(model.capabilities.supportsActivationTaps, isTrue);
      expect(model.manifest.activations, isEmpty);

      final run = await model.runWithTaps([
        await goldens.tensor(
          goldens.cases.first.inputKeys.single,
          goldens.manifest.inputs.single,
        ),
      ]);
      expect(run.activations, isEmpty);
    });
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

    test('the build can read intermediates, this artifact has none', () async {
      // Two different questions, and conflating them is what would make a gate
      // report agreement it never measured. The build compiles the tracer, so
      // it can read any op the runtime executes itself. This artifact is one
      // delegated partition, so it executes none, and the export declared no
      // taps for exactly that reason.
      expect(model.capabilities.supportsActivationTaps, isTrue);
      expect(model.manifest.activations, isEmpty);

      final run = await model.runWithTaps([
        await goldens.tensor(
          goldens.cases.first.inputKeys.single,
          goldens.manifest.inputs.single,
        ),
      ]);
      expect(run.activations, isEmpty);
    });
  });

  // Attribution needs a graph whose layers the runtime executes itself, which is
  // what `backend='portable'` exports and what no delegated artifact can be.
  if (!_taps.existsSync()) {
    test('the attribution gate needs testdata/taps', () {}, skip: true);
    return;
  }

  group('M20 · attributing a drift to the layer that caused it', () {
    late DirectoryGoldenBundle goldens;
    late LoadedModel model;

    setUpAll(() async {
      goldens = await DirectoryGoldenBundle.open(
        '${_taps.path}/two_layer.fluttorch.json',
      );
      model = await ExecuTorchRuntime(bindings).load(
        artifact: await File('${_taps.path}/two_layer.pte').readAsBytes(),
        manifest: goldens.manifest,
        backend: 'xnnpack',
      );
    });

    tearDownAll(() async => model.dispose());

    test('the export says where each tap lives in the lowered graph', () {
      expect(
        [for (final s in model.manifest.activations) s.name],
        ['fc1', 'act', 'fc2'],
      );
      // Submodule names do not survive lowering, so without these the device has
      // nothing to ask for.
      expect(model.manifest.activationHandles, hasLength(3));
    });

    test('every declared layer comes back from the device', () async {
      final run = await model.runWithTaps([
        await goldens.tensor(
          goldens.cases.first.inputKeys.single,
          goldens.manifest.inputs.single,
        ),
      ]);

      expect(run.activations.keys, containsAll(['fc1', 'act', 'fc2']));
      expect(run.activations['fc1']!.shape, [1, 8]);
      expect(run.activations['act']!.shape, [1, 8]);
      expect(run.activations['fc2']!.shape, [1, 3]);
    });

    test('the intermediates are the layers, not three copies of one', () async {
      final run = await model.runWithTaps([
        await goldens.tensor(
          goldens.cases.first.inputKeys.single,
          goldens.manifest.inputs.single,
        ),
      ]);

      final fc1 = run.activations['fc1']!.asFloat32List();
      final act = run.activations['act']!.asFloat32List();
      final fc2 = run.activations['fc2']!.asFloat32List();

      // The activation is a ReLU of the layer before it, which is a property no
      // amount of plumbing luck reproduces: every negative gone, every
      // non-negative untouched.
      for (var i = 0; i < fc1.length; i++) {
        expect(act[i], fc1[i] < 0 ? 0.0 : fc1[i], reason: 'unit $i');
      }
      // And the last tap is the output, because fc2 is the last layer.
      expect(fc2, orderedEquals(run.outputs.single.asFloat32List()));
    });

    test('each tap matches the reference the export captured', () async {
      final run = await model.runWithTaps([
        await goldens.tensor(
          goldens.cases.first.inputKeys.single,
          goldens.manifest.inputs.single,
        ),
      ]);

      // The whole point of attribution: the device's layer against the source
      // model's layer, tap by tap, so a drift names where it started.
      final case0 = goldens.cases.first;
      for (var i = 0; i < model.manifest.activations.length; i++) {
        final spec = model.manifest.activations[i];
        final reference = await goldens.tensor(case0.activationKeys[i], spec);
        final onDevice = run.activations[spec.name]!.asFloat32List();
        final expected = reference.asFloat32List();
        for (var e = 0; e < expected.length; e++) {
          expect(
            onDevice[e],
            closeTo(expected[e], 1e-5),
            reason: '${spec.name}[$e]',
          );
        }
      }
    });
  });

  // Whether this build carries MPS is a property of the ExecuTorch checkout, so
  // the library is asked. A machine that never built it gets a named skip rather
  // than a failure, which is the whole of what M23 asks of a backend.
  if (!bindings.backends().contains('mps') || !_mps.existsSync()) {
    test('the MPS gate needs an ExecuTorch built with it', () {}, skip: true);
    return;
  }

  group('M23 · the same model through MPS', () {
    late DirectoryGoldenBundle goldens;
    late LoadedModel model;

    setUpAll(() async {
      goldens = await DirectoryGoldenBundle.open(
        '${_mps.path}/two_layer.fluttorch.json',
      );
      model = await ExecuTorchRuntime(bindings).load(
        artifact: await File('${_mps.path}/two_layer.pte').readAsBytes(),
        manifest: goldens.manifest,
        backend: 'mps',
      );
    });

    tearDownAll(() async => model.dispose());

    test('it loads on the backend it was lowered for', () {
      expect(model.backend, 'mps');
    });

    test('every golden holds at the full-precision bound', () async {
      // Pinned to float32 at export for the reason Core ML is, so the bound the
      // gate picks on its own is the one this artifact can answer to.
      await expectParity(model, goldens: goldens);
    });

    test('it refuses determinism rather than promising it', () async {
      // A GPU schedules work it does not undertake to schedule the same way
      // twice. Saying so is worth more than a flag nobody can rely on.
      expect(model.capabilities.supportsDeterministicExecution, isFalse);
    });
  });
}
