// What the library costs: to generate, to load, and to run.
//
//   dart run tool/benchmark.dart
//
// Three numbers, because three different claims get made about this library and
// each one has a different way of being wrong.
//
// Codegen is a build-time cost paid once per manifest, and the claim is that it
// is small enough not to think about. Load is paid once per model per process,
// and it is the one a user feels as a stall on a cold start. Run is paid on
// every inference, and the claim there is not about speed at all: `run`
// allocates its outputs and `runInto` writes into buffers the caller owns, and
// the whole reason `runInto` exists is that the difference shows up over a long
// session rather than in a single call.
//
// Latency here is the median and the 95th percentile, never the mean. A mean
// over a distribution with a long tail describes a run that never happened, and
// on a laptop with a scheduler the tail is where the interesting part is.
//
// The numbers this prints are the machine's, not the library's. Publishing them
// without the machine beside them would be publishing a number nobody can
// reproduce or argue with, which is why the header says what ran it.
import 'dart:io';

import 'package:fluttorch/fluttorch.dart';
import 'package:fluttorch_executorch/fluttorch_executorch.dart';
import 'package:fluttorch_executorch/src/ffi.dart';
import 'package:fluttorch_gen/fluttorch_gen.dart';
import 'package:fluttorch_test/io.dart';

/// Enough iterations that the median is stable and the run still ends.
const _warmup = 50;
const _iterations = 1000;

/// Bundles to measure, smallest model first so the table reads as a scale.
const _bundles = <({String dir, String model, String backend})>[
  (dir: 'taps', model: 'two_layer', backend: 'portable'),
  (dir: 'matrix/portable', model: 'matrix', backend: 'portable'),
  (dir: 'matrix/xnnpack', model: 'matrix', backend: 'xnnpack'),
  (dir: 'voltacast', model: 'voltacast', backend: 'portable'),
];

Future<void> main() async {
  final library = File(
    '.dart_tool/native/libfluttorch_executorch'
    '${Platform.isMacOS ? ".dylib" : ".so"}',
  );
  if (!library.existsSync()) {
    stderr.writeln(
      'no native library at ${library.path}. tool/build_native.sh builds it, '
      'and load and run cannot be measured without one.',
    );
    exit(2);
  }

  stdout
    ..writeln(
      '${Platform.operatingSystem} ${Platform.operatingSystemVersion}, '
      '${Platform.numberOfProcessors} cores, Dart ${Platform.version.split(" ").first}',
    )
    ..writeln('$_iterations iterations after $_warmup warmup');

  await _codegen();
  await _loadAndRun(library);
}

/// Time to turn one manifest into the Dart a consumer commits.
///
/// Measured on the manifest rather than on the builder, because build_runner's
/// own overhead is build_runner's and swamps this by orders of magnitude. What
/// is being claimed is that the emitter is not the reason a build is slow.
Future<void> _codegen() async {
  stdout.writeln('\ncodegen, per manifest');
  stdout.writeln(
    '  ${"model".padRight(14)}${"inputs".padLeft(7)}${"outputs".padLeft(8)}'
    '${"median".padLeft(11)}${"p95".padLeft(11)}',
  );

  for (final bundle in _bundles) {
    final path = '../../testdata/${bundle.dir}/${bundle.model}.fluttorch.json';
    if (!File(path).existsSync()) continue;
    final manifest = ManifestCodec.decode(await File(path).readAsString());

    final samples = <int>[];
    for (var i = 0; i < _warmup + _iterations; i++) {
      final watch = Stopwatch()..start();
      emit(manifest, sourceName: '${bundle.model}.fluttorch.json');
      watch.stop();
      if (i >= _warmup) samples.add(watch.elapsedMicroseconds);
    }

    stdout.writeln(
      '  ${bundle.model.padRight(14)}'
      '${manifest.inputs.length.toString().padLeft(7)}'
      '${manifest.outputs.length.toString().padLeft(8)}'
      '${_micros(_percentile(samples, 50)).padLeft(11)}'
      '${_micros(_percentile(samples, 95)).padLeft(11)}',
    );
  }
}

Future<void> _loadAndRun(File library) async {
  final bindings = NativeExecuTorchBindings.open(library.path);
  final linked = bindings.backends();

  stdout.writeln('\nload, per model per process');
  stdout.writeln(
    '  ${"model".padRight(14)}${"backend".padRight(10)}${"artifact".padLeft(11)}'
    '${"median".padLeft(11)}${"p95".padLeft(11)}',
  );

  final runnable = <({String label, LoadedModel model, List<Tensor> inputs})>[];

  for (final bundle in _bundles) {
    final dir = '../../testdata/${bundle.dir}';
    if (!Directory(dir).existsSync()) continue;
    if (!linked.contains(bundle.backend)) continue;

    final goldens = await DirectoryGoldenBundle.open(
      '$dir/${bundle.model}.fluttorch.json',
    );
    final artifact = await File('$dir/${bundle.model}.pte').readAsBytes();

    // Loading is slower than a single run and the sample is smaller for it:
    // a thousand loads of a 3.5 MB artifact measures the allocator more than
    // the binding.
    final samples = <int>[];
    for (var i = 0; i < 20; i++) {
      final watch = Stopwatch()..start();
      final model = await ExecuTorchRuntime(bindings).load(
        artifact: artifact,
        manifest: goldens.manifest,
        backend: bundle.backend,
      );
      watch.stop();
      samples.add(watch.elapsedMicroseconds);
      await model.dispose();
    }

    stdout.writeln(
      '  ${bundle.model.padRight(14)}${bundle.backend.padRight(10)}'
      '${_bytes(artifact.length).padLeft(11)}'
      '${_micros(_percentile(samples, 50)).padLeft(11)}'
      '${_micros(_percentile(samples, 95)).padLeft(11)}',
    );

    // Kept loaded for the run measurements below. VoltaCast is excluded there
    // rather than silently reported as zero: ExecuTorch lowers it and then
    // fails to execute it, which is recorded in the on-device suite.
    final model = await ExecuTorchRuntime(bindings).load(
      artifact: artifact,
      manifest: goldens.manifest,
      backend: bundle.backend,
    );
    final golden = goldens.cases.first;
    final inputs = <Tensor>[
      for (var i = 0; i < goldens.manifest.inputs.length; i++)
        await goldens.tensor(golden.inputKeys[i], goldens.manifest.inputs[i]),
    ];
    var runs = true;
    try {
      await model.run(inputs);
    } on Object {
      runs = false;
    }
    if (runs) {
      runnable.add((
        label: '${bundle.model}/${bundle.backend}',
        model: model,
        inputs: inputs,
      ));
    } else {
      await model.dispose();
    }
  }

  // Resident memory was measured here and is not reported. Dart exposes no
  // allocation count, and the RSS delta over a thousand runs came back negative
  // as often as positive: it measures when the collector happened to run, not
  // what the loop allocated. A column of noise in a published table is worse
  // than a column that is missing, because a reader cannot tell which it is.
  //
  // What is reported instead is the pair. `run` allocates an output tensor per
  // call and `runInto` writes into one the caller keeps, so the gap between
  // them is the cost of that allocation, measured rather than asserted.
  stdout.writeln('\nrun, per inference');
  stdout.writeln(
    '  ${"model/backend".padRight(22)}${"run".padLeft(10)}'
    '${"runInto".padLeft(10)}${"p95 run".padLeft(10)}'
    '${"p95 into".padLeft(10)}${"saved".padLeft(9)}',
  );

  for (final entry in runnable) {
    final medians = <int>[];
    final p95s = <int>[];
    for (final inPlace in [false, true]) {
      final outputs = <Tensor>[
        for (final spec in entry.model.manifest.outputs) Tensor.zeros(spec),
      ];

      for (var i = 0; i < _warmup; i++) {
        inPlace
            ? await entry.model.runInto(inputs: entry.inputs, outputs: outputs)
            : await entry.model.run(entry.inputs);
      }

      final samples = <int>[];
      for (var i = 0; i < _iterations; i++) {
        final watch = Stopwatch()..start();
        inPlace
            ? await entry.model.runInto(inputs: entry.inputs, outputs: outputs)
            : await entry.model.run(entry.inputs);
        watch.stop();
        samples.add(watch.elapsedMicroseconds);
      }
      medians.add(_percentile(samples, 50));
      p95s.add(_percentile(samples, 95));
    }

    final saved = medians[0] - medians[1];
    stdout.writeln(
      '  ${entry.label.padRight(22)}'
      '${_micros(medians[0]).padLeft(10)}${_micros(medians[1]).padLeft(10)}'
      '${_micros(p95s[0]).padLeft(10)}${_micros(p95s[1]).padLeft(10)}'
      '${_micros(saved).padLeft(9)}',
    );
    await entry.model.dispose();
  }
}

int _percentile(List<int> samples, int p) {
  final sorted = [...samples]..sort();
  return sorted[((sorted.length - 1) * p / 100).round()];
}

String _micros(int v) =>
    v >= 1000 ? '${(v / 1000).toStringAsFixed(2)} ms' : '$v us';

String _bytes(int v) {
  if (v.abs() >= 1024 * 1024) {
    return '${(v / (1024 * 1024)).toStringAsFixed(1)} MB';
  }
  if (v.abs() >= 1024) return '${(v / 1024).toStringAsFixed(1)} kB';
  return '$v B';
}
