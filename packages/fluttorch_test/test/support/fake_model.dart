import 'dart:convert';
import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';
import 'package:fluttorch_test/fluttorch_test.dart';

/// A [LoadedModel] whose answers the test chooses.
///
/// The gate is what is under test here, so what runs behind it has to be
/// something whose output is decided rather than measured. Against a real
/// backend every assertion below would be a statement about ExecuTorch instead.
final class FakeModel implements LoadedModel {
  FakeModel({
    required this.manifest,
    required this.onRun,
    String backend = 'xnnpack',
    bool taps = false,
  }) : backend = backend,
       capabilities = RuntimeCapabilities(
         backend: backend,
         dtypes: const {DType.float32, DType.int32},
         supportsActivationTaps: taps,
       );

  /// Answers with the outputs the export recorded, paired to the inputs that
  /// produced them.
  ///
  /// This cannot drift, so a suite that only ever ran against it would prove
  /// nothing about a device. What it does prove is the half of the gate that
  /// lives in Dart: that the runner hands each case its own inputs and measures
  /// the output against that case's reference and no other.
  static Future<FakeModel> replaying(
    ModelManifest manifest,
    GoldenBundle goldens, {
    String backend = 'xnnpack',
    bool taps = false,
    void Function(String caseId, List<Tensor> outputs)? perturb,
  }) async {
    final answers = <String, List<Tensor>>{};
    for (final golden in goldens.cases) {
      final inputs = <Tensor>[
        for (var i = 0; i < manifest.inputs.length; i++)
          await goldens.tensor(golden.inputKeys[i], manifest.inputs[i]),
      ];
      final outputs = <Tensor>[
        for (var i = 0; i < manifest.outputs.length; i++)
          await goldens.tensor(golden.outputKeys[i], manifest.outputs[i]),
      ];
      // Copied before anything is allowed to move them, so a perturbation
      // cannot edit the bundle the assertion is measured against.
      final answer = [
        for (final t in outputs)
          Tensor.view(
            spec: t.spec,
            bytes: Uint8List.fromList(t.bytes),
            shape: t.spec.isDynamic ? t.shape : null,
          ),
      ];
      perturb?.call(golden.id, answer);
      answers[_fingerprint(inputs)] = answer;
    }

    return FakeModel(
      manifest: manifest,
      backend: backend,
      taps: taps,
      onRun: (inputs) async {
        final answer = answers[_fingerprint(inputs)];
        if (answer == null) {
          throw StateError('these inputs belong to no golden case');
        }
        return answer;
      },
    );
  }

  @override
  final ModelManifest manifest;

  @override
  final String backend;

  @override
  final RuntimeCapabilities capabilities;

  /// What this model answers with.
  final Future<List<Tensor>> Function(List<Tensor> inputs) onRun;

  /// How many times [run] was called, so a test can tell a case that was
  /// skipped from one that passed.
  int runs = 0;

  @override
  Future<List<Tensor>> run(List<Tensor> inputs) async {
    checkTensorsAgainst(manifest.inputs, inputs, role: 'input');
    runs++;
    return onRun(inputs);
  }

  @override
  Future<void> runInto({
    required List<Tensor> inputs,
    required List<Tensor> outputs,
  }) => throw UnimplementedError('the gate replays through run');

  @override
  Future<TappedRun> runWithTaps(List<Tensor> inputs, {List<String>? layers}) =>
      throw CapabilityUnavailableException(
        backend: backend,
        capability: 'activation taps',
      );

  @override
  Future<void> dispose() async {}

  static String _fingerprint(List<Tensor> tensors) =>
      tensors.map((t) => base64.encode(t.bytes)).join('|');
}

/// A `float32` tensor of [values] named [name].
Tensor f32(String name, List<double> values, {List<int>? shape}) {
  final spec = TensorSpec(
    name: name,
    dtype: DType.float32,
    shape: shape ?? [values.length],
  );
  final tensor = Tensor.zeros(spec);
  tensor.asFloat32List().setAll(0, values);
  return tensor;
}

/// The bytes of a `float32` tensor, as a bundle stores them.
Uint8List f32Bytes(List<double> values) =>
    Float32List.fromList(values).buffer.asUint8List();
