// Measures what each recipe and precision actually costs, on a model that can
// go wrong.
//
//   dart run tool/measure_tolerances.dart
//
// Every bound in `Tolerance` was a starting point, written before any model had
// been measured and documented as such. This is what replaces the guess: the
// same convolutional network, the same eight goldens, exported once per backend
// and once per recipe, and the worst drift each one produces.
//
// It reports rather than asserts. The numbers here are evidence for the table
// in `tolerance.dart`, and a tool that failed when a measurement moved would be
// a tool nobody could use to find out that it had.
//
// Read the widest column, not the average. A tolerance has to hold the worst
// case a model produces, and the case that fails a build is by definition not
// the typical one.
import 'dart:io';

import 'package:fluttorch_executorch/fluttorch_executorch.dart';
import 'package:fluttorch_executorch/src/ffi.dart';
import 'package:fluttorch_test/fluttorch_test.dart';
import 'package:fluttorch_test/io.dart';

/// Every bundle to measure: its directory, the model in it, and the backend
/// that has to run it.
///
/// Two models rather than one, and that is the whole point of the tool. The
/// convolutional network is where delegates disagree structurally; the
/// two-layer one is where they disagree by magnitude. Its outputs land near
/// `9.4` while the convolutional model ends in a softmax that pins its own into
/// `[0, 1]`, and the relative error is measured against the output while the
/// rounding happened on intermediates. Measuring only the well-behaved one
/// produces a bound that looks rigorous and fails the other model in the same
/// repository.
///
/// The quantized recipes are lowered for xnnpack, which is the delegate the
/// quantizer targets, so they are measured there rather than wherever happened
/// to be linked.
typedef _Bundle = ({String dir, String model, String backend});

const _bundles = <_Bundle>[
  (dir: 'matrix/portable', model: 'matrix', backend: 'portable'),
  (dir: 'matrix/xnnpack', model: 'matrix', backend: 'xnnpack'),
  (dir: 'matrix/coreml', model: 'matrix', backend: 'coreml'),
  (dir: 'matrix/mps', model: 'matrix', backend: 'mps'),
  (dir: 'matrix/mlx', model: 'matrix', backend: 'mlx'),
  (dir: 'matrix/int8-dynamic', model: 'matrix', backend: 'xnnpack'),
  (dir: 'matrix/int8-static', model: 'matrix', backend: 'xnnpack'),
  (dir: 'taps', model: 'two_layer', backend: 'portable'),
  (dir: 'quantized', model: 'two_layer', backend: 'xnnpack'),
  (dir: 'coreml', model: 'two_layer', backend: 'coreml'),
  (dir: 'mps', model: 'two_layer', backend: 'mps'),
];

Future<void> main() async {
  final library = File(
    '.dart_tool/native/libfluttorch_executorch'
    '${Platform.isMacOS ? ".dylib" : ".so"}',
  );
  if (!library.existsSync()) {
    stderr.writeln(
      'no native library at ${library.path}. tool/build_native.sh builds it, '
      'and without it there is nothing to measure against.',
    );
    exit(2);
  }

  final bindings = NativeExecuTorchBindings.open(library.path);
  final linked = bindings.backends();

  stdout.writeln('measured on the exports committed under testdata/');
  stdout.writeln(
    '  ${"bundle".padRight(22)}${"recipe".padRight(14)}'
    '${"precision".padRight(11)}${"worst |Δ|".padLeft(11)}'
    '${"worst rel".padLeft(11)}${"worst 1-cos".padLeft(13)}',
  );

  final missing = <String, String>{};
  for (final bundle in _bundles) {
    final dir = Directory('../../testdata/${bundle.dir}');
    if (!dir.existsSync()) {
      missing[bundle.dir] = 'no export under ${dir.path}';
      continue;
    }
    if (!linked.contains(bundle.backend)) {
      missing[bundle.dir] = '${bundle.backend} is not linked into this build';
      continue;
    }

    final goldens = await DirectoryGoldenBundle.open(
      '${dir.path}/${bundle.model}.fluttorch.json',
    );
    final model = await ExecuTorchRuntime(bindings).load(
      artifact: await File('${dir.path}/${bundle.model}.pte').readAsBytes(),
      manifest: goldens.manifest,
      backend: bundle.backend,
    );

    // Measured against a bound wide enough that nothing fails, because the
    // point is to read the number rather than to judge it. Judging it against
    // the table it is evidence for would be circular.
    final reports = await measureParity(
      model,
      goldens: goldens,
      tolerance: Tolerance(
        maxAbsolute: double.infinity,
        maxRelative: double.infinity,
      ),
    );

    var absolute = 0.0;
    var relative = 0.0;
    var cosine = 1.0;
    for (final report in reports) {
      for (final tensor in report.tensors) {
        if (tensor.maxAbsolute > absolute) absolute = tensor.maxAbsolute;
        if (tensor.maxRelative > relative) relative = tensor.maxRelative;
        if (tensor.cosine < cosine) cosine = tensor.cosine;
      }
    }

    final m = goldens.manifest;
    stdout.writeln(
      '  ${bundle.dir.padRight(22)}${(m.quantization ?? "none").padRight(14)}'
      '${(m.precision ?? "float32").padRight(11)}'
      '${absolute.toStringAsExponential(1).padLeft(11)}'
      '${relative.toStringAsExponential(1).padLeft(11)}'
      '${(1 - cosine).toStringAsExponential(1).padLeft(13)}',
    );

    await model.dispose();
  }

  if (missing.isNotEmpty) {
    stdout.writeln('  not measured:');
    for (final entry in missing.entries) {
      stdout.writeln('    ${entry.key}: ${entry.value}');
    }
  }
}
