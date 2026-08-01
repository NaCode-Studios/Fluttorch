import 'package:fluttorch/fluttorch.dart';

import 'bundle.dart';
import 'drift.dart';
import 'parity.dart';
import 'tolerance.dart';

/// One export measured on one backend, as the matrix takes it.
///
/// The two travel together because they came from one export: a manifest names
/// the recipe the tolerance is derived from, and the bundle holds the references
/// that recipe is being judged against. Pairing them at the call site is what
/// stops a matrix comparing an artifact to somebody else's numbers.
final class MatrixEntry {
  const MatrixEntry({required this.model, required this.goldens});

  final LoadedModel model;
  final GoldenBundle goldens;
}

/// The same goldens measured across every backend a machine offers.
///
/// A single backend passing says the artifact survived one lowering. The matrix
/// is the claim worth making: that the same inputs produce the same answers
/// wherever they run, and that where they do not, the report says which backend
/// moved and by how much.
///
/// Rows are goldens and columns are backends, which is the orientation a reader
/// wants: an input that misbehaves everywhere is a property of the model, and
/// one that misbehaves on a single backend is a property of that backend.
final class ParityMatrix {
  const ParityMatrix({
    required this.backends,
    required this.goldenIds,
    required this.reports,
  });

  /// Backends measured, in the order they were supplied.
  final List<String> backends;

  /// Golden ids, in the order the export captured them.
  final List<String> goldenIds;

  /// Every measurement taken, each carrying the backend that produced it.
  final List<DriftReport> reports;

  /// The measurement for one cell, or null where that pair was not measured.
  DriftReport? at({required String backend, required String goldenId}) {
    for (final r in reports) {
      if (r.backend == backend && r.goldenId == goldenId) return r;
    }
    return null;
  }

  /// Whether every cell passed the tolerance it was measured against.
  ///
  /// An empty matrix does not pass. A gate that reports success because it had
  /// nothing to measure is the failure this package exists to prevent.
  bool get passes => reports.isNotEmpty && reports.every((r) => r.passes);

  /// The worst drift any backend showed for one golden, so a row can be read
  /// without reading every cell in it.
  double worstFor(String goldenId) {
    var worst = 0.0;
    for (final r in reports) {
      if (r.goldenId != goldenId) continue;
      for (final t in r.tensors) {
        if (t.maxRelative > worst) worst = t.maxRelative;
      }
    }
    return worst;
  }

  /// The matrix as a table, one row per golden and one column per backend.
  ///
  /// Relative drift rather than absolute, because absolute is not comparable
  /// down a column: a golden whose outputs are near 100 and one whose outputs
  /// are near 0.1 do not answer to the same number of decimal places, and a
  /// table that pretends otherwise is read as one backend being worse.
  String describe() {
    if (reports.isEmpty) {
      return 'parity matrix: nothing was measured, which is not the same as '
          'everything agreeing';
    }

    final width = goldenIds.fold(7, (w, id) => id.length > w ? id.length : w);
    final b = StringBuffer()
      ..writeln(
        '${passes ? "PASS" : "FAIL"}  parity matrix  '
        '${goldenIds.length} golden(s) across ${backends.length} backend(s)',
      )
      ..write('  ${"golden".padRight(width)}');
    for (final backend in backends) {
      b.write('  ${backend.padLeft(12)}');
    }
    b.writeln();

    for (final id in goldenIds) {
      b.write('  ${id.padRight(width)}');
      for (final backend in backends) {
        final cell = at(backend: backend, goldenId: id);
        if (cell == null) {
          // Not measured is its own outcome, distinct from measured and zero.
          b.write('  ${"-".padLeft(12)}');
          continue;
        }
        final worst = cell.tensors.fold(
          0.0,
          (w, t) => t.maxRelative > w ? t.maxRelative : w,
        );
        final mark = cell.passes ? ' ' : '!';
        b.write('  ${"${worst.toStringAsExponential(1)}$mark".padLeft(12)}');
      }
      b.writeln();
    }

    for (final backend in backends) {
      final row = reports.where((r) => r.backend == backend).toList();
      if (row.isEmpty) continue;
      b.writeln(
        '  $backend: ${row.first.quantization ?? "no quantization"}, '
        'measured against ${row.first.tensors.first.tolerance}',
      );
    }
    for (final r in reports.where((r) => !r.passes)) {
      b.writeln(r.describe());
    }
    return b.toString();
  }
}

/// Replays one set of goldens across several backends and reports all of them.
///
/// Each entry is measured at the tolerance its own manifest implies, which is
/// the only way the row means anything: an int8 export and a float32 one of the
/// same model are not wrong by the same amount and should not answer to the same
/// bound. [tolerance] overrides that for every entry, which is what a caller
/// wants when the question is how the backends compare rather than whether each
/// one meets its own recipe.
///
/// Entries whose goldens do not line up are refused rather than measured. A
/// matrix over different inputs is a table of unrelated numbers arranged to look
/// like a comparison.
Future<ParityMatrix> measureMatrix(
  Iterable<MatrixEntry> entries, {
  Tolerance? tolerance,
}) async {
  final list = entries.toList();
  if (list.isEmpty) {
    throw ArgumentError.value(
      entries,
      'entries',
      'a matrix over no backends measures nothing; supply at least one',
    );
  }

  final ids = [for (final c in list.first.goldens.cases) c.id];
  for (final entry in list.skip(1)) {
    final theirs = [for (final c in entry.goldens.cases) c.id];
    if (!_sameOrder(ids, theirs)) {
      throw ArgumentError.value(
        entry.model.backend,
        'entries',
        'carries goldens ${theirs.join(", ")} where the first entry carries '
            '${ids.join(", ")}. A matrix compares one set of inputs across '
            'backends, and these are different questions side by side',
      );
    }
  }

  final reports = <DriftReport>[];
  for (final entry in list) {
    reports.addAll(
      await measureParity(
        entry.model,
        goldens: entry.goldens,
        tolerance: tolerance,
      ),
    );
  }

  return ParityMatrix(
    backends: [for (final e in list) e.model.backend],
    goldenIds: ids,
    reports: reports,
  );
}

bool _sameOrder(List<String> a, List<String> b) {
  if (a.length != b.length) return false;
  for (var i = 0; i < a.length; i++) {
    if (a[i] != b[i]) return false;
  }
  return true;
}
