import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';
import 'package:test/test.dart' show fail;

import 'bundle.dart';
import 'drift.dart';
import 'tolerance.dart';

/// Replays every golden against [model] and returns one report per case.
///
/// Compares final outputs, which on today's backends is everything that is
/// reachable. Attributing drift to a layer needs the activations the source
/// model produced, which the manifest does not record yet, so every report says
/// that it compared outputs only rather than leaving the absence of an
/// attribution to be read as agreement.
///
/// [tolerance] defaults to [Tolerance.startingPointFor] on the manifest's
/// quantization recipe. That returns null for a recipe this build does not
/// know, in which case a tolerance must be supplied: there is no default worth
/// guessing for a scheme nobody has measured.
///
/// Measuring is not asserting. This returns the reports whether they pass or
/// fail, which is what makes a drift report usable outside a test, and
/// [expectParity] is the one that fails a build.
Future<List<DriftReport>> measureParity(
  LoadedModel model, {
  required GoldenBundle goldens,
  Tolerance? tolerance,
}) async {
  final bounds =
      tolerance ?? Tolerance.startingPointFor(model.manifest.quantization);
  if (bounds == null) {
    throw ArgumentError.value(
      model.manifest.quantization,
      'tolerance',
      'no starting point is defined for this quantization recipe; this build '
          'knows ${Tolerance.knownRecipes.join(", ")}. Supply a tolerance '
          'measured against this model rather than letting the gate run on a '
          'number nobody chose',
    );
  }

  final reports = <DriftReport>[];
  for (final golden in goldens.cases) {
    reports.add(await _replay(model, golden, goldens, bounds));
  }
  return reports;
}

/// Asserts that every golden passes, failing with a report that names the
/// tensor, the drift, the tolerance and the backend that produced it.
///
/// An empty bundle fails too. A gate that passes because it had nothing to
/// check is the failure this whole package exists to prevent, and it is
/// indistinguishable from a healthy model right up to the point where it
/// matters.
Future<void> expectParity(
  LoadedModel model, {
  required GoldenBundle goldens,
  Tolerance? tolerance,
}) async {
  if (goldens.cases.isEmpty) {
    fail(
      'parity/${model.manifest.name}: this bundle carries no golden cases, so '
      'the assertion would pass over a model nobody evaluated. Re-export with '
      'golden inputs, or remove the assertion and say why.',
    );
  }

  final reports = await measureParity(
    model,
    goldens: goldens,
    tolerance: tolerance,
  );
  final failures = reports.where((r) => !r.passes).toList();
  if (failures.isEmpty) return;

  final message = StringBuffer()
    ..writeln(
      '${failures.length} of ${reports.length} golden '
      '${reports.length == 1 ? "case" : "cases"} drifted past tolerance '
      'on backend "${model.backend}".',
    )
    ..writeln();
  for (final report in failures) {
    message.write(report.describe());
  }
  fail(message.toString());
}

/// Runs the same goldens across every backend the device offers.
///
/// A model that is correct on one backend and wrong on another shows up as
/// exactly that, which a single-backend run cannot reveal.
Future<Map<String, List<DriftReport>>> parityMatrix(
  FluttorchRuntime runtime, {
  required ModelArtifact artifact,
  required GoldenBundle goldens,
  Tolerance? tolerance,
}) {
  throw UnimplementedError('M24 · parity matrix across backends');
}

/// A model artifact together with the manifest it was exported with.
///
/// Paired deliberately: loading one without the other is what the weight hash
/// exists to catch, and the matrix has to reload the same artifact once per
/// backend.
final class ModelArtifact {
  const ModelArtifact({required this.bytes, required this.manifest});

  /// The runtime artifact as exported.
  final Uint8List bytes;

  /// The contract it was exported with.
  final ModelManifest manifest;
}

Future<DriftReport> _replay(
  LoadedModel model,
  GoldenCase golden,
  GoldenBundle goldens,
  Tolerance tolerance,
) async {
  final manifest = model.manifest;
  _checkArity(golden, 'input', golden.inputKeys.length, manifest.inputs.length);
  _checkArity(
    golden,
    'output',
    golden.outputKeys.length,
    manifest.outputs.length,
  );

  final inputs = <Tensor>[
    for (var i = 0; i < manifest.inputs.length; i++)
      await goldens.tensor(golden.inputKeys[i], manifest.inputs[i]),
  ];

  final produced = await model.run(inputs);
  // The backend is checked against the contract like any other caller. A
  // backend that returns the outputs in a different order would otherwise be
  // measured against the wrong references and reported as drift.
  checkTensorsAgainst(manifest.outputs, produced, role: 'output');

  final drifts = <TensorDrift>[];
  for (var i = 0; i < manifest.outputs.length; i++) {
    drifts.add(
      measureDrift(
        actual: produced[i],
        reference: await goldens.tensor(
          golden.outputKeys[i],
          manifest.outputs[i],
        ),
        tolerance: tolerance,
      ),
    );
  }

  return DriftReport(
    goldenId: golden.id,
    backend: model.backend,
    quantization: manifest.quantization,
    tensors: drifts,
    attributionUnavailable: model.capabilities.supportsActivationTaps
        ? 'the goldens record final outputs only'
        : null,
  );
}

void _checkArity(GoldenCase golden, String role, int found, int declared) {
  if (found == declared) return;
  throw TensorShapeException(
    'golden "${golden.id}" names $found ${role}s where the model declares '
    '$declared; the bundle and the manifest come from different exports',
  );
}
