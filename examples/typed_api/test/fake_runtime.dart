/// A runtime that returns what it is told to.
///
/// The generated API is the thing under test here, not inference: a fake keeps
/// the assertions about types and contracts rather than about numbers, which
/// `fluttorch_test` covers on real backends.
///
/// Shared by the suites in this package because a fake copied per suite is a
/// fake that drifts, and two of them disagreeing about what a runtime does
/// would be a difference nobody meant to introduce.
library;

import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';

final class FakeRuntime implements FluttorchRuntime {
  FakeRuntime(this.outputs);

  final List<Tensor> outputs;
  Uint8List? loadedArtifact;
  String? requestedBackend;

  @override
  Future<List<RuntimeCapabilities>> capabilities() async => const [
    RuntimeCapabilities(backend: 'fake', dtypes: {DType.float32}),
  ];

  @override
  Future<LoadedModel> load({
    required Uint8List artifact,
    required ModelManifest manifest,
    String? backend,
    bool deterministic = false,
  }) async {
    loadedArtifact = artifact;
    requestedBackend = backend;
    return _FakeModel(manifest, outputs);
  }
}

final class _FakeModel implements LoadedModel {
  _FakeModel(this.manifest, this._outputs);

  @override
  final ModelManifest manifest;
  final List<Tensor> _outputs;

  List<Tensor>? received;
  bool disposed = false;

  @override
  String get backend => 'fake';

  @override
  RuntimeCapabilities get capabilities =>
      const RuntimeCapabilities(backend: 'fake', dtypes: {DType.float32});

  @override
  Future<List<Tensor>> run(List<Tensor> inputs) async {
    checkTensorsAgainst(manifest.inputs, inputs, role: 'inputs');
    received = inputs;
    return _outputs;
  }

  @override
  Future<void> runInto({
    required List<Tensor> inputs,
    required List<Tensor> outputs,
  }) async => throw UnimplementedError();

  @override
  Future<TappedRun> runWithTaps(List<Tensor> inputs, {List<String>? layers}) =>
      throw CapabilityUnavailableException(
        backend: backend,
        capability: 'activation taps',
      );

  @override
  Future<void> dispose() async => disposed = true;
}
