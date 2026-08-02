import 'dart:ffi';
import 'dart:io' show Platform;
import 'dart:typed_data';

import 'package:ffi/ffi.dart';
import 'package:fluttorch/fluttorch.dart';

import 'bindings.dart';

/// The C structs from `src/fluttorch_executorch.h`, as Dart sees them.
///
/// Field order is the contract. A struct is read by offset on both sides, so
/// reordering these silently reads the wrong bytes rather than failing to
/// compile, which is why the header and this file are edited together.
final class FtCapabilities extends Struct {
  external Pointer<Utf8> backend;

  @Int32()
  external int supportsTaps;

  @Int32()
  external int supportsDeterminism;

  @Int64()
  external int maxTensorBytes;

  @Int64()
  external int dtypes;
}

final class FtTensor extends Struct {
  external Pointer<Uint8> data;

  @Int64()
  external int byteLength;

  external Pointer<Int64> shape;

  @Int32()
  external int rank;

  @Int32()
  external int dtype;
}

final class FtModel extends Opaque {}

typedef _BackendsNative =
    Int32 Function(Pointer<Pointer<Utf8>>, Int32, Pointer<Int32>);
typedef _Backends = int Function(Pointer<Pointer<Utf8>>, int, Pointer<Int32>);

typedef _CapabilitiesNative =
    Int32 Function(Pointer<Utf8>, Pointer<FtCapabilities>);
typedef _Capabilities = int Function(Pointer<Utf8>, Pointer<FtCapabilities>);

typedef _LoadNative =
    Int32 Function(
      Pointer<Uint8>,
      Int64,
      Pointer<Utf8>,
      Int32,
      Pointer<Pointer<FtModel>>,
    );
typedef _Load =
    int Function(
      Pointer<Uint8>,
      int,
      Pointer<Utf8>,
      int,
      Pointer<Pointer<FtModel>>,
    );

typedef _ModelBackendNative = Pointer<Utf8> Function(Pointer<FtModel>);
typedef _ModelBackend = Pointer<Utf8> Function(Pointer<FtModel>);

typedef _RunNative =
    Int32 Function(
      Pointer<FtModel>,
      Pointer<FtTensor>,
      Int32,
      Pointer<FtTensor>,
      Int32,
    );
typedef _Run =
    int Function(
      Pointer<FtModel>,
      Pointer<FtTensor>,
      int,
      Pointer<FtTensor>,
      int,
    );

typedef _RunWithTapsNative =
    Int32 Function(
      Pointer<FtModel>,
      Pointer<FtTensor>,
      Int32,
      Pointer<FtTensor>,
      Int32,
      Pointer<Int64>,
      Int32,
      Pointer<FtTensor>,
      Pointer<Int32>,
    );
typedef _RunWithTaps =
    int Function(
      Pointer<FtModel>,
      Pointer<FtTensor>,
      int,
      Pointer<FtTensor>,
      int,
      Pointer<Int64>,
      int,
      Pointer<FtTensor>,
      Pointer<Int32>,
    );

typedef _DisposeNative = Void Function(Pointer<FtModel>);
typedef _Dispose = void Function(Pointer<FtModel>);

typedef _LastErrorNative = Pointer<Utf8> Function();
typedef _LastError = Pointer<Utf8> Function();

/// Status codes, mirrored from the header.
abstract final class FtStatus {
  static const ok = 0;
  static const invalidArgument = 1;
  static const artifactUnreadable = 2;
  static const backendUnavailable = 3;
  static const capabilityUnavailable = 4;
  static const shapeMismatch = 5;
  static const executionFailed = 6;
  static const outOfMemory = 7;
}

/// [ExecuTorchBindings] over a real shared library.
///
/// Every call returns a status and the library never throws across the
/// boundary: an exception crossing FFI terminates the process, and a model that
/// fails to load is an ordinary Tuesday.
final class NativeExecuTorchBindings implements ExecuTorchBindings {
  NativeExecuTorchBindings(this._lib, {this.runtimeName = 'executorch'})
    : _backends = _lib.lookupFunction<_BackendsNative, _Backends>(
        'ft_backends',
      ),
      _capabilities = _lib.lookupFunction<_CapabilitiesNative, _Capabilities>(
        'ft_capabilities',
      ),
      _load = _lib.lookupFunction<_LoadNative, _Load>('ft_load'),
      _modelBackend = _lib.lookupFunction<_ModelBackendNative, _ModelBackend>(
        'ft_model_backend',
      ),
      _run = _lib.lookupFunction<_RunNative, _Run>('ft_run'),
      _runWithTaps = _lib.lookupFunction<_RunWithTapsNative, _RunWithTaps>(
        'ft_run_with_taps',
      ),
      _dispose = _lib.lookupFunction<_DisposeNative, _Dispose>('ft_dispose'),
      _lastError = _lib.lookupFunction<_LastErrorNative, _LastError>(
        'ft_last_error',
      );

  /// Opens the library by the name each platform gives it.
  factory NativeExecuTorchBindings.open([
    String? path,
    String runtimeName = 'executorch',
  ]) => NativeExecuTorchBindings(
    path != null
        ? DynamicLibrary.open(path)
        : (Platform.isIOS || Platform.isMacOS)
        ? DynamicLibrary.process()
        : DynamicLibrary.open('libfluttorch_$runtimeName.so'),
    runtimeName: runtimeName,
  );

  /// Which engine is behind this ABI, so a failure can name it.
  ///
  /// The three shims share this client, because what they share is the C ABI
  /// rather than an implementation. Without this a runtime failure could only
  /// say "the native call failed", which is true of all three.
  final String runtimeName;

  // ignore: unused_field
  final DynamicLibrary _lib;
  final _Backends _backends;
  final _Capabilities _capabilities;
  final _Load _load;
  final _ModelBackend _modelBackend;
  final _Run _run;
  final _RunWithTaps _runWithTaps;
  final _Dispose _dispose;
  final _LastError _lastError;

  static const _maxBackends = 16;

  @override
  List<String> backends() => using((arena) {
    final names = arena<Pointer<Utf8>>(_maxBackends);
    final count = arena<Int32>();
    _check(_backends(names, _maxBackends, count), 'listing backends');
    return [for (var i = 0; i < count.value; i++) names[i].toDartString()];
  });

  @override
  NativeCapabilities capabilitiesOf(String? backend) => using((arena) {
    final out = arena<FtCapabilities>();
    final name = backend == null
        ? nullptr
        : backend.toNativeUtf8(allocator: arena);
    final status = _capabilities(name.cast(), out);
    if (status == FtStatus.backendUnavailable) {
      throw BackendUnavailableException(
        requested: backend ?? '(preferred)',
        available: backends(),
      );
    }
    _check(
      status,
      'reading the capabilities of ${backend ?? "the preferred backend"}',
    );
    final c = out.ref;
    return NativeCapabilities(
      backend: c.backend.toDartString(),
      dtypes: NativeCapabilities.dtypesFromMask(c.dtypes),
      supportsTaps: c.supportsTaps != 0,
      supportsDeterminism: c.supportsDeterminism != 0,
      maxTensorBytes: c.maxTensorBytes == 0 ? null : c.maxTensorBytes,
    );
  });

  @override
  NativeModel load({
    required Uint8List artifact,
    String? backend,
    bool deterministic = false,
  }) {
    // The artifact is borrowed for the duration of the call, so it is copied
    // into native memory rather than pinned: Dart offers no way to pin a
    // typed list, and a moving collector under a pointer is a crash nobody can
    // reproduce.
    final bytes = malloc<Uint8>(artifact.length);
    final handle = malloc<Pointer<FtModel>>();
    try {
      bytes.asTypedList(artifact.length).setAll(0, artifact);
      final status = using(
        (arena) => _load(
          bytes,
          artifact.length,
          backend == null
              ? nullptr
              : backend.toNativeUtf8(allocator: arena).cast(),
          deterministic ? 1 : 0,
          handle,
        ),
      );
      if (status == FtStatus.backendUnavailable) {
        throw BackendUnavailableException(
          requested: backend ?? '(preferred)',
          available: backends(),
        );
      }
      if (status == FtStatus.capabilityUnavailable) {
        throw CapabilityUnavailableException(
          backend: backend ?? '(preferred)',
          capability: deterministic ? 'deterministic execution' : 'this load',
        );
      }
      _check(status, 'loading the artifact');
      return _NativeModel(this, handle.value);
    } finally {
      malloc
        ..free(bytes)
        ..free(handle);
    }
  }

  void _check(int status, String what) {
    if (status == FtStatus.ok) return;
    final raw = _lastError();
    final detail = raw == nullptr ? null : raw.toDartString();
    // The shim formats the engine's own code into its message, so it is pulled
    // back out rather than left inside a string. That number is what a reader
    // takes to the runtime's source, and searching a sentence for it is work
    // this can do once.
    final code = detail == null
        ? null
        : int.tryParse(
            RegExp(r'error (\d+)').firstMatch(detail)?.group(1) ?? '',
          );
    throw RuntimeExecutionException(
      runtime: runtimeName,
      operation: what,
      status: status,
      code: code,
      detail: detail,
    );
  }
}

final class _NativeModel implements NativeModel {
  _NativeModel(this._bindings, this._handle)
    : capabilities = _bindings.capabilitiesOf(
        _bindings._modelBackend(_handle).toDartString(),
      );

  final NativeExecuTorchBindings _bindings;
  final Pointer<FtModel> _handle;

  @override
  final NativeCapabilities capabilities;

  bool _disposed = false;

  @override
  String get backend => capabilities.backend;

  @override
  void run(List<Tensor> inputs, List<Tensor> outputs) => using((arena) {
    final ins = _marshal(inputs, arena);
    final outs = _marshal(outputs, arena);
    _bindings._check(
      _bindings._run(_handle, ins, inputs.length, outs, outputs.length),
      'running inference',
    );
    _writeBack(outputs, outs);
  });

  @override
  Set<int> runWithTaps(
    List<Tensor> inputs,
    List<Tensor> outputs,
    List<Tensor> activations,
    List<int> handles,
  ) => using((arena) {
    final ins = _marshal(inputs, arena);
    final outs = _marshal(outputs, arena);
    final acts = _marshal(activations, arena);

    final wanted = arena<Int64>(handles.length);
    for (var i = 0; i < handles.length; i++) {
      wanted[i] = handles[i];
    }
    final captured = arena<Int32>();

    _bindings._check(
      _bindings._runWithTaps(
        _handle,
        ins,
        inputs.length,
        outs,
        outputs.length,
        wanted,
        handles.length,
        acts,
        captured,
      ),
      'running inference with taps',
    );
    _writeBack(outputs, outs);

    // A bitmask rather than a count, because what comes back can be sparse: a
    // partially delegated graph answers for the layers it runs itself and for
    // none of the ones inside a partition, and a count could not say which.
    final filled = <int>{};
    for (var i = 0; i < handles.length; i++) {
      if (captured.value & (1 << i) != 0) filled.add(i);
    }
    _writeBack(activations, acts, only: filled);
    return filled;
  });

  @override
  void dispose() {
    if (_disposed) return;
    _disposed = true;
    _bindings._dispose(_handle);
  }

  /// Copies a list of tensors into the C struct array the ABI takes.
  ///
  /// The bytes are copied because Dart cannot hand out a stable pointer into
  /// its own heap. That copy is the cost of the boundary and not of the design:
  /// it is one memcpy per tensor per call, against an inference.
  Pointer<FtTensor> _marshal(List<Tensor> tensors, Arena arena) {
    final array = arena<FtTensor>(tensors.length);
    for (var i = 0; i < tensors.length; i++) {
      final t = tensors[i];
      final data = arena<Uint8>(t.bytes.length);
      data.asTypedList(t.bytes.length).setAll(0, t.bytes);
      final shape = arena<Int64>(t.shape.length);
      for (var d = 0; d < t.shape.length; d++) {
        shape[d] = t.shape[d];
      }
      array[i]
        ..data = data
        ..byteLength = t.bytes.length
        ..shape = shape
        ..rank = t.shape.length
        ..dtype = t.spec.dtype.index;
    }
    return array;
  }

  /// Copies what the native side wrote back into the caller's buffers.
  /// Copies the native side's writes back into the caller's tensors.
  ///
  /// [only] restricts that to the positions actually filled, which taps need:
  /// a tap the graph did not run leaves its buffer as the caller supplied it,
  /// and copying over it would turn "nobody looked" into a number.
  void _writeBack(
    List<Tensor> outputs,
    Pointer<FtTensor> written, {
    Set<int>? only,
  }) {
    for (var i = 0; i < outputs.length; i++) {
      if (only != null && !only.contains(i)) continue;
      outputs[i].bytes.setAll(
        0,
        written[i].data.asTypedList(written[i].byteLength),
      );
    }
  }
}
