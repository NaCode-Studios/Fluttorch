import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';
import 'package:test/test.dart' show fail;

import 'bundle.dart';
import 'drift.dart';
import 'tolerance.dart';

/// Replays every golden against [model] and returns one report per case.
///
/// Compares final outputs, and where the export captured taps and the backend
/// offers them, also walks the intermediate activations in graph order and names
/// the earliest layer whose numbers moved. Both conditions are required and
/// neither is assumed: a report that could not look says which of the two was
/// missing, rather than leaving the absence of an attribution to be read as
/// agreement.
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
      tolerance ??
      Tolerance.startingPointFor(
        model.manifest.quantization,
        precision: model.manifest.precision,
      );
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

  // Taps exist in a manifest only because someone asked for them at export, so
  // the cost of capturing them was accepted there. Running with them from the
  // start rather than re-running a failure keeps the activations and the outputs
  // from the same inference, which on a backend that cannot promise determinism
  // is the difference between attribution and a second opinion.
  final wanted =
      manifest.activations.isNotEmpty && golden.activationKeys.isNotEmpty;
  final tapped = wanted && model.capabilities.supportsActivationTaps;

  final List<Tensor> produced;
  var activations = const <String, Tensor>{};
  if (tapped) {
    final run = await model.runWithTaps(
      inputs,
      layers: [for (final spec in manifest.activations) spec.name],
    );
    produced = run.outputs;
    activations = run.activations;
  } else {
    produced = await model.run(inputs);
  }

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

  final attribution = tapped
      ? await _attribute(manifest, golden, goldens, activations, tolerance)
      : _Attribution.unavailable(
          wanted
              ? 'backend "${model.backend}" offers no activation taps'
              : 'the goldens record final outputs only',
        );

  return DriftReport(
    goldenId: golden.id,
    backend: model.backend,
    quantization: manifest.quantization,
    precision: manifest.precision,
    tensors: drifts,
    layers: attribution.layers,
    firstDivergentLayer: attribution.firstDivergent,
    attributionAttempted: attribution.attempted,
    attributionUnavailable: attribution.reason,
  );
}

/// What walking the tapped layers established, or why it could not.
final class _Attribution {
  const _Attribution({
    required this.layers,
    required this.firstDivergent,
    required this.attempted,
    this.reason,
  });

  factory _Attribution.unavailable(String reason) => _Attribution(
    layers: const [],
    firstDivergent: null,
    attempted: false,
    reason: reason,
  );

  final List<TensorDrift> layers;
  final String? firstDivergent;
  final bool attempted;
  final String? reason;
}

/// Walks the declared activations in graph order and names the first that moved.
///
/// Measured against the same tolerance as the outputs. A separate bound for
/// intermediates would be a second number nobody has measured, and the question
/// here is which layer went first rather than whether a layer is acceptable on
/// its own.
Future<_Attribution> _attribute(
  ModelManifest manifest,
  GoldenCase golden,
  GoldenBundle goldens,
  Map<String, Tensor> activations,
  Tolerance tolerance,
) async {
  final layers = <TensorDrift>[];
  String? firstDivergent;

  for (var i = 0; i < manifest.activations.length; i++) {
    final spec = manifest.activations[i];
    final actual = activations[spec.name];
    if (actual == null) {
      // A layer the backend did not tap is a hole in the sequence, and a hole
      // reads exactly like a layer that agreed. Anything found before it is
      // still the earliest divergence; anything after it is unknowable, so the
      // answer is either what we already have or nothing.
      if (firstDivergent != null) break;
      return _Attribution.unavailable(
        'the run returned no activation for "${spec.name}", so no layer after '
        'it can be ruled out',
      );
    }
    final drift = measureDrift(
      actual: actual,
      reference: await goldens.tensor(golden.activationKeys[i], spec),
      tolerance: tolerance,
    );
    layers.add(drift);
    if (!drift.passes && firstDivergent == null) firstDivergent = spec.name;
  }

  return _Attribution(
    layers: layers,
    firstDivergent: firstDivergent,
    attempted: true,
  );
}

void _checkArity(GoldenCase golden, String role, int found, int declared) {
  if (found == declared) return;
  throw TensorShapeException(
    'golden "${golden.id}" names $found ${role}s where the model declares '
    '$declared; the bundle and the manifest come from different exports',
  );
}
