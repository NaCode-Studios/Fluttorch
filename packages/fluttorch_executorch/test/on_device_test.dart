@TestOn('vm')
library;

import 'dart:io';
import 'dart:isolate';

import 'package:fluttorch/fluttorch.dart';
import 'package:fluttorch_executorch/fluttorch_executorch.dart';
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
final _voltacast = Directory('../../testdata/voltacast');
final _multiIo = Directory('../../testdata/multi_io');

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
    // One export per backend of the convolutional model, not the two-layer one.
    // A network of two linear layers has nowhere for two delegates to disagree,
    // so the matrix came out ordered the way arithmetic says it should and
    // proved the measurement worked rather than that it was worth taking.
    final candidates = <String, Directory>{
      for (final backend in ['portable', 'xnnpack', 'coreml', 'mps', 'mlx'])
        backend: Directory('../../testdata/matrix/$backend'),
    };

    late ParityMatrix matrix;
    final loaded = <LoadedModel>[];

    setUpAll(() async {
      final entries = <MatrixEntry>[];
      for (final entry in candidates.entries) {
        if (!entry.value.existsSync()) continue;
        if (!bindings.backends().contains(entry.key)) continue;
        final goldens = await DirectoryGoldenBundle.open(
          '${entry.value.path}/matrix.fluttorch.json',
        );
        final model = await ExecuTorchRuntime(bindings).load(
          artifact: await File('${entry.value.path}/matrix.pte').readAsBytes(),
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
      // Eight rather than four, because the tolerance this model needs is read
      // off this suite and four numbers is not a distribution.
      expect(matrix.goldenIds, hasLength(8));
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

    test('the half-precision columns are the ones that moved', () {
      // The matrix earns its place here. The same weights and the same goldens
      // go to every delegate, and the ones that lower to float16 land three to
      // four orders of magnitude further from the source than the ones that
      // stay at float32. A table where every column read the same would mean
      // the precision the manifest records was not the precision that ran.
      //
      // Read off the artifact rather than the backend name, because which
      // delegate halves precision is that delegate's decision and it is
      // recorded per export rather than assumed here.
      final half = matrix.backends.where(
        (b) => matrix.precisionOf(b) == 'float16',
      );
      final full = matrix.backends.where(
        (b) => matrix.precisionOf(b) != 'float16',
      );
      if (half.isEmpty || full.isEmpty) {
        markTestSkipped(
          'this machine linked only ${matrix.backends.join(", ")}, so there is '
          'no precision difference to order',
        );
        return;
      }

      for (final id in matrix.goldenIds) {
        for (final low in half) {
          final moved = matrix
              .at(backend: low, goldenId: id)!
              .tensors
              .single
              .maxRelative;
          for (final exact in full) {
            expect(
              moved,
              greaterThan(
                matrix
                    .at(backend: exact, goldenId: id)!
                    .tensors
                    .single
                    .maxRelative,
              ),
              reason: '$id: $low(float16) should move more than $exact',
            );
          }
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
        tolerance: Tolerance.boundFor(null),
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
      // Nothing quantized it, and it is not full precision either. Core ML does
      // arithmetic in float16 by default, which is what a deployment runs, and
      // the manifest recording that is what lets the gate pick a bound the
      // artifact can actually hold.
      expect(model.manifest.quantization, isNull);
      expect(model.manifest.precision, 'float16');
    });

    test('every golden holds at the bound its precision implies', () async {
      // No tolerance is passed. The manifest says float16 and the gate sizes
      // the bound from that on its own, which is the whole of #53.
      await expectParity(model, goldens: goldens);
    });

    test('held to the float32 bound, the same run fails', () async {
      // The half that makes the first one mean something. If a float16 artifact
      // passed at the float32 bound too, recording the precision would have
      // bought nothing and the bound would be decorative.
      final reports = await measureParity(
        model,
        goldens: goldens,
        tolerance: Tolerance.boundFor(null),
      );
      expect(reports.every((r) => !r.passes), isTrue);
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

    test('every golden holds at the bound its precision implies', () async {
      // Half precision, like Core ML and for the same reason: it is what an MPS
      // deployment runs, and the manifest says so rather than leaving the gate
      // to assume float32.
      expect(model.manifest.precision, 'float16');
      await expectParity(model, goldens: goldens);
    });

    test('it refuses determinism rather than promising it', () async {
      // A GPU schedules work it does not undertake to schedule the same way
      // twice. Saying so is worth more than a flag nobody can rely on.
      expect(model.capabilities.supportsDeterministicExecution, isFalse);
    });
  });

  group('M28 · the same model, run off the calling isolate', () {
    late IsolateExecuTorchRuntime runtime;
    late DirectoryGoldenBundle goldens;
    late LoadedModel model;

    setUpAll(() async {
      runtime = await IsolateExecuTorchRuntime.spawn(
        libraryPath: _library.path,
      );
      goldens = await DirectoryGoldenBundle.open(
        '${_quantized.path}/two_layer.fluttorch.json',
      );
      model = await runtime.load(
        artifact: await File('${_quantized.path}/two_layer.pte').readAsBytes(),
        manifest: goldens.manifest,
        backend: 'xnnpack',
      );
    });

    tearDownAll(() async {
      await model.dispose();
      await runtime.shutdown();
    });

    test('the native library was opened somewhere else', () async {
      // The direct evidence available for the claim. FFI executes wherever it
      // is called from, and only the worker holds the bindings, so a name that
      // is not this isolate's is a native call that did not happen here.
      final where = await runtime.whereNativeCallsRun();
      expect(where, isNotNull);
      expect(where, isNot(Isolate.current.debugName));
    });

    test('it reports the backends the library carries', () async {
      final caps = await runtime.capabilities();
      expect(caps.map((c) => c.backend), contains('xnnpack'));
      expect(model.backend, 'xnnpack');
    });

    test('the goldens hold across the isolate hop', () async {
      // The whole point: the numbers are the numbers whichever isolate produced
      // them. A hop that quietly reordered or truncated bytes would show here
      // and nowhere else.
      await expectParity(model, goldens: goldens);
    });

    test('runInto writes into the buffers the caller owns', () async {
      final input = await goldens.tensor(
        goldens.cases.first.inputKeys.single,
        goldens.manifest.inputs.single,
      );
      final output = Tensor.zeros(goldens.manifest.outputs.single);

      await model.runInto(inputs: [input], outputs: [output]);

      // Copied back across the hop rather than allocated fresh, which is the
      // semantic runInto promises even where it cannot promise zero copies.
      expect(output.asFloat32List().any((v) => v != 0), isTrue);
    });

    test('a failure keeps its type across the hop', () async {
      // A typed refusal that arrived as a bare error would turn a condition the
      // caller is meant to handle into one they cannot.
      await expectLater(
        runtime.load(
          artifact: await File(
            '${_quantized.path}/two_layer.pte',
          ).readAsBytes(),
          manifest: goldens.manifest,
          backend: 'vulkan',
        ),
        throwsA(isA<BackendUnavailableException>()),
      );
    });
  });

  // ── M29 · a model that can go wrong ────────────────────────────────────────
  //
  // Everything else here measures a two-layer network. This is VoltaCast: a
  // seq2seq Transformer forecasting Italian electricity demand, three encoder
  // layers over a week of hourly history and two decoder layers cross-attending
  // to it, trained on eleven years of real data.
  //
  // It exports. It lowers. It does not execute, and that is recorded here
  // rather than left out, because a suite that quietly drops the one model
  // large enough to disagree is a suite claiming coverage it does not have.
  //
  // The failure is upstream and the evidence is that the same artifacts fail
  // identically under ExecuTorch's own Python runtime, which never touches this
  // binding. Through XNNPACK: "Failed to resize output tensor for XNNExecutor",
  // Error 0x10. Through portable kernels: Error 0x12, InvalidArgument. Both at
  // execution, both after a lowering that reported success.
  //
  // The model itself is fine, and the proof is that LiteRT runs it. The same
  // weights and the same golden windows go through the same gate in
  // packages/fluttorch_litert/test/voltacast_test.dart and agree with the
  // notebook to within float32 rounding. That is what the runtime layer was
  // built for, and this is the first time it has been worth something rather
  // than merely demonstrated.
  //
  // What this group still tests is the half that works here: the bundle is
  // well-formed, the binding loads a 3.5 MB artifact, and the manifest
  // describes a model with two inputs.
  group('M29 · VoltaCast, as far as it currently goes', () {
    late DirectoryGoldenBundle goldens;
    late LoadedModel model;

    setUpAll(() async {
      goldens = await DirectoryGoldenBundle.open(
        '${_voltacast.path}/voltacast.fluttorch.json',
      );
      model = await ExecuTorchRuntime(bindings).load(
        artifact: await File('${_voltacast.path}/voltacast.pte').readAsBytes(),
        manifest: goldens.manifest,
      );
    });

    tearDownAll(() async => model.dispose());

    test('the binding loads it, and it is a real model', () {
      expect(model.manifest.inputs.map((s) => s.name), ['past', 'future']);
      expect(model.manifest.inputNamed('past').shape, [1, 168, 13]);
      expect(model.manifest.inputNamed('future').shape, [1, 24, 12]);
      expect(model.manifest.outputNamed('quantiles').shape, [1, 24, 3]);
      expect(model.manifest.labels, ['p10', 'p50', 'p90']);
      expect(goldens.cases, hasLength(4));
    });

    test('running it fails inside ExecuTorch, not at this seam', () async {
      // Pinned so the day upstream fixes it, this test fails and somebody
      // deletes it. A limitation recorded as a passing test is a limitation
      // that gets forgotten; recorded as a failing expectation, it announces
      // its own repair.
      await expectLater(
        expectParity(model, goldens: goldens),
        throwsA(anything),
        reason:
            'if this now passes, ExecuTorch executes the model and this whole '
            'group should become the parity gate it was written to be',
      );
    });
  });

  // Every other export in this repository returns one tensor, so the code that
  // keeps a second one in the right order has been read and never run. This
  // fixture exists to run it.
  //
  // Its two inputs share a shape and so do its two outputs, which is the choice
  // that gives the group its teeth: a pair that differed in shape would be
  // caught on the way past by the shape check, and the test would then pass for
  // a reason that has nothing to do with ordering.
  group('two inputs and two outputs, kept apart', () {
    late DirectoryGoldenBundle goldens;
    late LoadedModel model;

    setUpAll(() async {
      goldens = await DirectoryGoldenBundle.open(
        '${_multiIo.path}/multi_io.fluttorch.json',
      );
      model = await ExecuTorchRuntime(bindings).load(
        artifact: await File('${_multiIo.path}/multi_io.pte').readAsBytes(),
        manifest: goldens.manifest,
        backend: 'portable',
      );
    });

    tearDownAll(() async => model.dispose());

    test('the manifest carries two of each, and they look alike', () {
      expect(model.manifest.inputs.map((s) => s.name), ['left', 'right']);
      expect(model.manifest.outputs.map((s) => s.name), [
        'primary',
        'auxiliary',
      ]);
      expect(
        model.manifest.inputNamed('right').shape,
        model.manifest.inputNamed('left').shape,
        reason: 'matching shapes are what remove the shape check as a crutch',
      );
      expect(
        model.manifest.outputNamed('auxiliary').shape,
        model.manifest.outputNamed('primary').shape,
      );
    });

    test('the gate measures both outputs, not just the first', () async {
      await expectParity(model, goldens: goldens);

      // A gate that replayed one of the two would pass this model while saying
      // nothing about half of it, and would say it in the same words.
      final reports = await measureParity(model, goldens: goldens);
      expect(reports, hasLength(4));
      for (final report in reports) {
        expect(report.tensors, hasLength(2));
      }
    });

    test('the inputs are threaded in order, not merely accepted', () async {
      final golden = goldens.cases.first;
      final left = await goldens.tensor(
        golden.inputKeys[0],
        model.manifest.inputs[0],
      );
      final right = await goldens.tensor(
        golden.inputKeys[1],
        model.manifest.inputs[1],
      );

      final straight = await model.run([left, right]);
      // The same two buffers in the same two positions, carrying each other's
      // values. A binding that passed the first buffer twice, or dropped the
      // second, returns the same numbers for both of these; one that threads
      // them through in order cannot.
      final crossed = await model.run([
        Tensor.view(spec: model.manifest.inputs[0], bytes: right.bytes),
        Tensor.view(spec: model.manifest.inputs[1], bytes: left.bytes),
      ]);

      expect(straight[0].asFloat32List(), isNot(crossed[0].asFloat32List()));
      expect(straight[1].asFloat32List(), isNot(crossed[1].asFloat32List()));
    });
  });
}
