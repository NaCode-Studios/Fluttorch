import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';

import 'bindings.dart';

/// Runs models exported by `fluttorch-export` through ExecuTorch.
///
/// Everything above the [ExecuTorchBindings] seam is ordinary Dart and is tested
/// as such. What sits below it is the one part that has to be compiled against
/// ExecuTorch itself, which is why the seam exists at all rather than the FFI
/// calls being spread through this file.
final class ExecuTorchRuntime implements FluttorchRuntime {
  const ExecuTorchRuntime(this._bindings);

  final ExecuTorchBindings _bindings;

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
    // Before anything native touches it. An artifact paired with the wrong
    // manifest satisfies every shape and returns every number wrong, and the
    // cheapest place to catch that is before the bytes leave Dart.
    verifyArtifact(artifact: artifact, manifest: manifest);

    // And before that, whether this runtime is the one the bundle was written
    // for. A .onnx handed to ExecuTorch passes the weight hash, because the
    // hash was computed over whichever artifact was written, and then fails
    // somewhere in the loader with a message about bytes.
    if (manifest.runtime != null && manifest.runtime != 'executorch') {
      throw BackendUnavailableException(
        requested: '${manifest.runtime} runtime',
        available: const ['executorch'],
      );
    }

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

    return _ExecuTorchModel(native, manifest);
  }
}

final class _ExecuTorchModel implements LoadedModel {
  _ExecuTorchModel(this._native, this.manifest);

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
    if (!capabilities.supportsActivationTaps) {
      throw CapabilityUnavailableException(
        backend: backend,
        capability: 'activation taps',
      );
    }
    checkTensorsAgainst(manifest.inputs, inputs, role: 'input');

    // A manifest with taps but no handles is one the export could not resolve
    // against the lowered graph, which is what a delegated artifact is. Refused
    // here rather than run: every layer would come back absent, and a caller
    // cannot tell that from a model whose layers all agreed.
    if (manifest.activations.isNotEmpty && manifest.activationHandles.isEmpty) {
      throw CapabilityUnavailableException(
        backend: backend,
        capability:
            'activation taps: this export records no handle for any of its '
            '${manifest.activations.length} declared activations, so nothing '
            'on the device can be asked for one',
      );
    }

    final wanted =
        layers ?? [for (final spec in manifest.activations) spec.name];
    final selected = [
      for (final name in wanted)
        _tapIndex(name) ??
            (throw ArgumentError.value(
              name,
              'layers',
              'is not a tap this export declared; it has '
                  '${manifest.activations.map((s) => s.name).join(", ")}',
            )),
    ];

    final outputs = _allocateOutputs();
    // Sized from the contract, like the outputs and for the same reason: the
    // native side writes into buffers this side already owns.
    final buffers = [
      for (final i in selected) Tensor.zeros(manifest.activations[i]),
    ];

    final filled = _native.runWithTaps(inputs, outputs, buffers, [
      for (final i in selected) manifest.activationHandles[i],
    ]);

    return TappedRun(
      outputs: outputs,
      // Only what was filled. A buffer nobody wrote to is all zeros, and a gate
      // reads a zero as a layer that agreed.
      activations: {
        for (var slot = 0; slot < selected.length; slot++)
          if (filled.contains(slot))
            manifest.activations[selected[slot]].name: buffers[slot],
      },
    );
  }

  /// Position of the tap named [name] in the manifest, or null if it declares none.
  int? _tapIndex(String name) {
    for (var i = 0; i < manifest.activations.length; i++) {
      if (manifest.activations[i].name == name) return i;
    }
    return null;
  }

  @override
  Future<void> dispose() async => _native.dispose();

  /// Destination buffers for one run, sized from the contract.
  ///
  /// A dynamic output has no size until the run that produces it, and this
  /// binding has no way to ask for one before running: the native side writes
  /// into buffers the caller already owns, which is what makes repeated
  /// inference free of allocation and is also what makes an unknown extent
  /// unanswerable here. Such a model is refused with the reason rather than
  /// served a guess, and [runInto] takes buffers the caller sized itself.
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
