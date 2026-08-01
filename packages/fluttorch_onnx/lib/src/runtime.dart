import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';
import 'package:fluttorch_executorch/fluttorch_executorch.dart';

/// Runs models exported for ONNX Runtime.
///
/// The seam below this is the same C ABI ExecuTorch answers to, so this class is
/// thin on purpose: what a second runtime changes is the engine, not the
/// contract. The manifest still says what the tensors are, the goldens are still
/// the references, and the gate still measures the difference.
///
/// It takes an [ExecuTorchBindings] because that is the Dart binding to the ABI
/// rather than a binding to ExecuTorch, despite what the name says. Two
/// implementations of one contract share one client; the name is worth fixing
/// and is not worth duplicating around.
final class OnnxRuntime implements FluttorchRuntime {
  const OnnxRuntime(this._bindings);

  final ExecuTorchBindings _bindings;

  /// Opens the ONNX shim by the name each platform gives it.
  factory OnnxRuntime.open([String? path]) =>
      OnnxRuntime(NativeExecuTorchBindings.open(path));

  @override
  Future<List<RuntimeCapabilities>> capabilities() async => [
    for (final name in _bindings.backends())
      _bindings.capabilitiesOf(name).toRuntime(),
  ];

  @override
  Future<LoadedModel> load({
    required Uint8List artifact,
    required ModelManifest manifest,
    String? backend,
    bool deterministic = false,
  }) async {
    // A .pte handed to ONNX Runtime passes the weight hash, because the hash was
    // computed over whichever artifact was written, and then fails inside the
    // session with a message about protobuf. Refused here, where the reason is
    // still legible.
    if (manifest.runtime != 'onnx') {
      throw BackendUnavailableException(
        requested: '${manifest.runtime ?? "executorch"} runtime',
        available: const ['onnx'],
      );
    }
    verifyArtifact(artifact: artifact, manifest: manifest);

    if (backend != null && !_bindings.backends().contains(backend)) {
      throw BackendUnavailableException(
        requested: backend,
        available: _bindings.backends(),
      );
    }

    final native = _bindings.load(
      artifact: artifact,
      backend: backend,
      deterministic: deterministic,
    );

    final caps = native.capabilities;
    if (!caps.toRuntime().supportsAllTypesIn(manifest)) {
      native.dispose();
      throw DTypeMismatchException(
        tensorName: manifest.name,
        declared: [
          ...manifest.inputs,
          ...manifest.outputs,
        ].firstWhere((s) => !caps.dtypes.contains(s.dtype)).dtype,
        requested: caps.dtypes.first,
      );
    }

    return _OnnxModel(native, manifest);
  }
}

final class _OnnxModel implements LoadedModel {
  _OnnxModel(this._native, this.manifest);

  final NativeModel _native;

  @override
  final ModelManifest manifest;

  @override
  String get backend => _native.backend;

  @override
  RuntimeCapabilities get capabilities => _native.capabilities.toRuntime();

  @override
  Future<List<Tensor>> run(List<Tensor> inputs) async {
    final outputs = _allocateOutputs();
    await runInto(inputs: inputs, outputs: outputs);
    return outputs;
  }

  @override
  Future<void> runInto({
    required List<Tensor> inputs,
    required List<Tensor> outputs,
  }) async {
    checkTensorsAgainst(manifest.inputs, inputs, role: 'input');
    checkTensorsAgainst(manifest.outputs, outputs, role: 'output');
    _native.run(inputs, outputs);
  }

  @override
  Future<TappedRun> runWithTaps(
    List<Tensor> inputs, {
    List<String>? layers,
  }) async {
    // Reported absent rather than approximated. ONNX Runtime reads an
    // intermediate by having it named as a graph output, which is a property of
    // the graph rather than of the run: an export that did it would be a
    // different artifact from the one the goldens describe, and a gate handed
    // the wrong tensor attributes a drift to the wrong layer.
    throw CapabilityUnavailableException(
      backend: backend,
      capability:
          'activation taps: ONNX Runtime reads intermediates by naming them as '
          'outputs, which changes the graph rather than observing it',
    );
  }

  @override
  Future<void> dispose() async => _native.dispose();

  List<Tensor> _allocateOutputs() => [
    for (final spec in manifest.outputs)
      if (spec.isDynamic)
        throw TensorShapeException(
          'output "${spec.name}" is ${spec.shape} and its extent is not known '
          'until the run that produces it. Size the buffer yourself and call '
          'runInto, which is the path this binding is built around',
          tensorName: spec.name,
        )
      else
        Tensor.zeros(spec),
  ];
}
