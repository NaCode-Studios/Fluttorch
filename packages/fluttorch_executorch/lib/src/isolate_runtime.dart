import 'dart:async';
import 'dart:isolate';
import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';

import 'ffi.dart';
import 'runtime.dart';

/// Runs ExecuTorch on a worker isolate, so inference does not block the caller.
///
/// Every call through the FFI seam is synchronous: `ft_run` executes on the
/// thread that called it and returns when the model is done. The Dart API above
/// it returns futures, but with [ExecuTorchRuntime] nothing suspends, so in a
/// Flutter app inference runs on the thread that draws, for as long as it takes.
/// On a two-layer model that is invisible. On a real one it is the frame budget,
/// repeatedly.
///
/// This runtime moves the whole native side onto one worker isolate: the library
/// is opened there, models are created there, and they never leave. That last
/// part is not an implementation detail. An `ft_model_t` is not safe to touch
/// from two threads at once, and a handle that crossed back to the caller would
/// be exactly that, with nothing in the type system to say so.
///
/// ## What it costs
///
/// A copy of every tensor in each direction. Isolates do not share memory, so
/// the caller-owned buffers that make [LoadedModel.runInto] cheap do not survive
/// the hop: the bytes are copied in, and the results are copied back over the
/// buffers the caller supplied.
///
/// That trade is worth stating rather than hiding. It is the right one exactly
/// when inference is long enough that blocking the UI would be visible, which is
/// also when a memcpy of the inputs is lost in the noise. It is the wrong one
/// for a model so small that the copy dominates, and such a model was not
/// blocking anything to begin with. Use [ExecuTorchRuntime] directly there, or
/// anywhere the caller is already off the platform thread.
final class IsolateExecuTorchRuntime implements FluttorchRuntime {
  IsolateExecuTorchRuntime._(this._isolate, this._commands, this._responses);

  /// Name the worker runs under, so a test can prove where the call happened.
  static const _workerName = 'fluttorch-executorch';

  final Isolate _isolate;
  final SendPort _commands;
  final ReceivePort _responses;

  final _pending = <int, Completer<Object?>>{};
  var _nextRequest = 0;
  var _closed = false;

  /// Spawns the worker and opens the native library on it.
  ///
  /// [libraryPath] is passed to [NativeExecuTorchBindings.open], so the same
  /// rules apply: null means the library is already in the process, which is
  /// what a packaged build gives, and a path is what a desktop build produces.
  ///
  /// Failing here rather than on the first run is deliberate. A library that
  /// cannot be opened is a build problem, and finding out at the first inference
  /// puts it in the middle of a frame instead of at start-up.
  static Future<IsolateExecuTorchRuntime> spawn({String? libraryPath}) async {
    final responses = ReceivePort();
    final runtime = Completer<IsolateExecuTorchRuntime>();

    final isolate = await Isolate.spawn(
      _worker,
      (responses.sendPort, libraryPath),
      errorsAreFatal: true,
      debugName: _workerName,
    );

    IsolateExecuTorchRuntime? instance;
    responses.listen((message) {
      // The first message is either the command port or the reason the worker
      // could not open the library. Everything after it is a reply.
      if (instance == null) {
        if (message is SendPort) {
          instance = IsolateExecuTorchRuntime._(isolate, message, responses);
          runtime.complete(instance);
        } else {
          responses.close();
          isolate.kill(priority: Isolate.immediate);
          runtime.completeError(
            StateError('the worker could not start: $message'),
          );
        }
        return;
      }
      instance!._deliver(message);
    });

    return runtime.future;
  }

  void _deliver(Object? message) {
    final (int id, Object? payload, String? error) =
        message as (int, Object?, String?);
    final completer = _pending.remove(id);
    if (completer == null) return;
    if (error != null) {
      completer.completeError(_rehydrate(error));
    } else {
      completer.complete(payload);
    }
  }

  Future<Object?> _send(String op, Object? payload) {
    if (_closed) {
      throw StateError('this runtime has been shut down');
    }
    final id = _nextRequest++;
    final completer = Completer<Object?>();
    _pending[id] = completer;
    _commands.send((id, op, payload));
    return completer.future;
  }

  @override
  Future<List<RuntimeCapabilities>> capabilities() async {
    final raw = await _send('capabilities', null) as List<Object?>;
    return [for (final c in raw) _capabilitiesFrom(c! as Map<String, Object?>)];
  }

  @override
  Future<LoadedModel> load({
    required Uint8List artifact,
    required ModelManifest manifest,
    String? backend,
    bool deterministic = false,
  }) async {
    // The manifest crosses as the document it already is. Sending the object
    // would work and would also make this the one place where a manifest exists
    // without having been through the codec both sides are held to.
    final handle =
        await _send('load', {
              'artifact': artifact,
              'manifest': ManifestCodec.encode(manifest),
              'backend': backend,
              'deterministic': deterministic,
            })
            as Map<String, Object?>;

    return _IsolateModel(
      runtime: this,
      id: handle['id']! as int,
      manifest: manifest,
      backend: handle['backend']! as String,
      capabilities: _capabilitiesFrom(
        handle['capabilities']! as Map<String, Object?>,
      ),
    );
  }

  /// The name of the isolate the native library was opened on.
  ///
  /// The only direct evidence available for the claim this class makes. FFI runs
  /// wherever it is called from, and only the worker holds the bindings, so an
  /// answer that differs from the caller's isolate is an answer that says the
  /// native call did not happen here.
  Future<String?> whereNativeCallsRun() async =>
      await _send('where', null) as String?;

  /// Stops the worker and releases every model still loaded on it.
  ///
  /// A model disposed by the worker shutting down is disposed the same way one
  /// disposed by hand is, so a caller that forgot does not leak the native side
  /// for the life of the process.
  Future<void> shutdown() async {
    if (_closed) return;
    _closed = true;
    try {
      await _send('shutdown', null);
    } on Object {
      // A worker that has already gone is a worker that is shut down.
    }
    _responses.close();
    _isolate.kill(priority: Isolate.beforeNextEvent);
  }
}

RuntimeCapabilities _capabilitiesFrom(Map<String, Object?> m) =>
    RuntimeCapabilities(
      backend: m['backend']! as String,
      dtypes: {
        for (final i in m['dtypes']! as List<Object?>) DType.values[i! as int],
      },
      supportsActivationTaps: m['taps']! as bool,
      supportsDeterministicExecution: m['deterministic']! as bool,
      maxTensorBytes: m['maxTensorBytes'] as int?,
    );

/// A model that lives on the worker and is addressed by handle.
final class _IsolateModel implements LoadedModel {
  _IsolateModel({
    required IsolateExecuTorchRuntime runtime,
    required int id,
    required this.manifest,
    required this.backend,
    required this.capabilities,
  }) : _runtime = runtime,
       _id = id;

  final IsolateExecuTorchRuntime _runtime;
  final int _id;

  @override
  final ModelManifest manifest;

  @override
  final String backend;

  @override
  final RuntimeCapabilities capabilities;

  @override
  Future<List<Tensor>> run(List<Tensor> inputs) async {
    final raw =
        await _runtime._send('run', {
              'model': _id,
              'inputs': [for (final t in inputs) _encode(t)],
            })
            as List<Object?>;
    return [
      for (var i = 0; i < raw.length; i++)
        _decode(raw[i]! as Map<String, Object?>, manifest.outputs[i]),
    ];
  }

  @override
  Future<void> runInto({
    required List<Tensor> inputs,
    required List<Tensor> outputs,
  }) async {
    final produced = await run(inputs);
    if (produced.length != outputs.length) {
      throw TensorShapeException(
        'the model returned ${produced.length} outputs and ${outputs.length} '
        'buffers were supplied',
        tensorName: manifest.name,
      );
    }
    // Copied back over what the caller owns, which is the semantic runInto
    // promises. What it cannot promise across an isolate is that no copy
    // happened, and the class comment says so rather than leaving it implied.
    for (var i = 0; i < outputs.length; i++) {
      outputs[i].bytes.setAll(0, produced[i].bytes);
    }
  }

  @override
  Future<TappedRun> runWithTaps(
    List<Tensor> inputs, {
    List<String>? layers,
  }) async {
    final raw =
        await _runtime._send('taps', {
              'model': _id,
              'inputs': [for (final t in inputs) _encode(t)],
              'layers': layers,
            })
            as Map<String, Object?>;

    final byName = {for (final s in manifest.activations) s.name: s};
    return TappedRun(
      outputs: [
        for (var i = 0; i < (raw['outputs']! as List<Object?>).length; i++)
          _decode(
            (raw['outputs']! as List<Object?>)[i]! as Map<String, Object?>,
            manifest.outputs[i],
          ),
      ],
      activations: {
        for (final e in (raw['activations']! as Map<Object?, Object?>).entries)
          e.key! as String: _decode(
            e.value! as Map<String, Object?>,
            byName[e.key]!,
          ),
      },
    );
  }

  @override
  Future<void> dispose() => _runtime._send('dispose', _id).then((_) {});
}

Map<String, Object?> _encode(Tensor t) => {'bytes': t.bytes, 'shape': t.shape};

Tensor _decode(Map<String, Object?> m, TensorSpec spec) => Tensor.view(
  spec: spec,
  bytes: m['bytes']! as Uint8List,
  shape: [for (final d in m['shape']! as List<Object?>) d! as int],
);

/// Rebuilds a typed failure from the worker, or reports it as it arrived.
///
/// A `BackendUnavailableException` that crossed as a string and arrived as a
/// `StateError` would turn a condition the caller is meant to handle into one
/// they cannot, so the ones the API documents are reconstructed. The rest keep
/// their message, which is more honest than inventing a type for them.
Object _rehydrate(String encoded) {
  final split = encoded.indexOf(' ');
  final kind = split < 0 ? '' : encoded.substring(0, split);
  final message = split < 0 ? encoded : encoded.substring(split + 1);
  return switch (kind) {
    'backend' => BackendUnavailableException(
      requested: message,
      available: const [],
    ),
    'capability' => CapabilityUnavailableException(
      backend: message,
      capability: 'the worker reported it unavailable',
    ),
    'shape' => TensorShapeException(message, tensorName: ''),
    _ => StateError(message),
  };
}

String _encodeError(Object error) => switch (error) {
  BackendUnavailableException(:final requested) => 'backend $requested',
  CapabilityUnavailableException(:final backend) => 'capability $backend',
  TensorShapeException(:final message) => 'shape $message',
  _ => ' $error',
};

/// The worker. Everything native happens here and nothing native leaves.
Future<void> _worker((SendPort, String?) args) async {
  final (responses, libraryPath) = args;

  final ExecuTorchRuntime runtime;
  try {
    runtime = ExecuTorchRuntime(NativeExecuTorchBindings.open(libraryPath));
  } on Object catch (e) {
    responses.send('$e');
    return;
  }

  final commands = ReceivePort();
  final models = <int, LoadedModel>{};
  var nextModel = 0;
  responses.send(commands.sendPort);

  await for (final message in commands) {
    final (int id, String op, Object? payload) =
        message as (int, String, Object?);
    try {
      final result = switch (op) {
        'capabilities' => [
          for (final c in await runtime.capabilities()) _capabilitiesTo(c),
        ],
        'load' => await () async {
          final m = payload! as Map<String, Object?>;
          final model = await runtime.load(
            artifact: m['artifact']! as Uint8List,
            manifest: ManifestCodec.decode(m['manifest']! as String),
            backend: m['backend'] as String?,
            deterministic: m['deterministic']! as bool,
          );
          final handle = nextModel++;
          models[handle] = model;
          return {
            'id': handle,
            'backend': model.backend,
            'capabilities': _capabilitiesTo(model.capabilities),
          };
        }(),
        'run' => await () async {
          final m = payload! as Map<String, Object?>;
          final model = models[m['model']! as int]!;
          final outputs = await model.run(
            _inputsFor(model.manifest, m['inputs']! as List<Object?>),
          );
          return [for (final t in outputs) _encode(t)];
        }(),
        'taps' => await () async {
          final m = payload! as Map<String, Object?>;
          final model = models[m['model']! as int]!;
          final tapped = await model.runWithTaps(
            _inputsFor(model.manifest, m['inputs']! as List<Object?>),
            layers: (m['layers'] as List<Object?>?)
                ?.map((e) => e! as String)
                .toList(),
          );
          return {
            'outputs': [for (final t in tapped.outputs) _encode(t)],
            'activations': {
              for (final e in tapped.activations.entries)
                e.key: _encode(e.value),
            },
          };
        }(),
        'where' => Isolate.current.debugName,
        'dispose' => await () async {
          await models.remove(payload! as int)?.dispose();
          return null;
        }(),
        'shutdown' => await () async {
          for (final model in models.values) {
            await model.dispose();
          }
          models.clear();
          return null;
        }(),
        _ => throw StateError('unknown command $op'),
      };
      responses.send((id, result, null));
      if (op == 'shutdown') {
        commands.close();
        return;
      }
    } on Object catch (e) {
      responses.send((id, null, _encodeError(e)));
    }
  }
}

List<Tensor> _inputsFor(ModelManifest manifest, List<Object?> raw) => [
  for (var i = 0; i < raw.length; i++)
    _decode(raw[i]! as Map<String, Object?>, manifest.inputs[i]),
];

Map<String, Object?> _capabilitiesTo(RuntimeCapabilities c) => {
  'backend': c.backend,
  'dtypes': [for (final d in c.dtypes) d.index],
  'taps': c.supportsActivationTaps,
  'deterministic': c.supportsDeterministicExecution,
  'maxTensorBytes': c.maxTensorBytes,
};
