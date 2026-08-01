// Prints one parity report covering every backend this machine can run.
//
//   dart run tool/parity_matrix.dart
//
// The same goldens, replayed on each backend the library was linked with and
// each export that exists under testdata/, measured at the tolerance that
// export's own recipe implies. Exits non-zero when any cell fails, so it can
// stand in a script as well as be read.
//
// Backends the build lacks are listed as not run rather than omitted. A matrix
// that quietly drops a column reads as a machine that had nothing to say about
// it, and those are different claims: the first is a gap in coverage and the
// second is coverage.
import 'dart:io';

import 'package:fluttorch/fluttorch.dart';
import 'package:fluttorch_executorch/fluttorch_executorch.dart';
import 'package:fluttorch_executorch/src/ffi.dart';
import 'package:fluttorch_test/fluttorch_test.dart';
import 'package:fluttorch_test/io.dart';

/// Which export under testdata/ answers for which backend.
const _exports = <String, String>{
  'xnnpack': '../../testdata/quantized',
  'portable': '../../testdata/taps',
  'coreml': '../../testdata/coreml',
  'mps': '../../testdata/mps',
};

Future<void> main(List<String> args) async {
  final library = File(
    '.dart_tool/native/libfluttorch_executorch'
    '${Platform.isMacOS ? ".dylib" : ".so"}',
  );
  if (!library.existsSync()) {
    stderr.writeln(
      'no native library at ${library.path}. tool/build_native.sh builds it, '
      'and without it there is no device to measure.',
    );
    exit(2);
  }

  final bindings = NativeExecuTorchBindings.open(library.path);
  final linked = bindings.backends();

  final entries = <MatrixEntry>[];
  final models = <LoadedModel>[];
  final skipped = <String, String>{};

  for (final entry in _exports.entries) {
    if (!linked.contains(entry.key)) {
      skipped[entry.key] = 'not linked into this build of the binding';
      continue;
    }
    final dir = Directory(entry.value);
    if (!dir.existsSync()) {
      skipped[entry.key] = 'no export under ${entry.value}';
      continue;
    }
    final goldens = await DirectoryGoldenBundle.open(
      '${dir.path}/two_layer.fluttorch.json',
    );
    final model = await ExecuTorchRuntime(bindings).load(
      artifact: await File('${dir.path}/two_layer.pte').readAsBytes(),
      manifest: goldens.manifest,
      backend: entry.key,
    );
    models.add(model);
    entries.add(MatrixEntry(model: model, goldens: goldens));
  }

  if (entries.isEmpty) {
    stderr.writeln('no backend on this machine had an export to measure');
    exit(2);
  }

  final matrix = await measureMatrix(entries);
  stdout.writeln(matrix.describe());

  if (skipped.isNotEmpty) {
    stdout.writeln('  not run:');
    for (final s in skipped.entries) {
      stdout.writeln('    ${s.key}: ${s.value}');
    }
  }

  for (final m in models) {
    await m.dispose();
  }
  exit(matrix.passes ? 0 : 1);
}
