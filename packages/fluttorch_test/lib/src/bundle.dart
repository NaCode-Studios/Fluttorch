import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';

/// The reference inputs and outputs captured at export time.
///
/// The manifest names its goldens by opaque key; this resolves those keys to
/// tensors. Implementations decide where they come from: a directory, a bundled
/// asset archive, or memory in a unit test. That is why the core never mentions
/// files.
abstract interface class GoldenBundle {
  /// Cases in the bundle, in the order the exporter captured them.
  List<GoldenCase> get cases;

  /// Resolves one key to a tensor satisfying [spec].
  ///
  /// Throws [TensorShapeException] if the stored bytes do not satisfy the spec,
  /// which means the bundle and the manifest came from different exports.
  Future<Tensor> tensor(String key, TensorSpec spec);
}

/// A bundle that resolves a key to the raw bytes stored under it.
///
/// The job has two halves and only one of them varies. Finding the bytes is
/// where a directory, an asset archive and a map differ; turning them into a
/// tensor is identical everywhere, and it is the half that has to reject a
/// mismatch. Implementations supply [bytesFor] and inherit the checking.
abstract base class BytesGoldenBundle implements GoldenBundle {
  const BytesGoldenBundle(this.manifest);

  /// The contract these goldens were captured against.
  ///
  /// Taken whole rather than as a list of cases, because a caller assembling
  /// cases by hand is the contract being restated, which is what this project
  /// exists to remove.
  final ModelManifest manifest;

  @override
  List<GoldenCase> get cases => manifest.goldens;

  /// The bytes stored under [key], little-endian, as the exporter wrote them.
  Future<Uint8List> bytesFor(String key);

  @override
  Future<Tensor> tensor(String key, TensorSpec spec) async {
    final bytes = await bytesFor(key);
    return Tensor.view(
      spec: spec,
      bytes: bytes,
      shape: _shapeOf(spec, bytes.length, key),
    );
  }

  /// The concrete shape of a stored tensor, or null when the spec already
  /// fixes it.
  ///
  /// One dynamic dimension is implied by how many elements arrived, which is
  /// what makes an export with `--dynamic-batch` replayable at all. Two dynamic
  /// dimensions have no unique answer, and inventing one would replay the
  /// golden at a shape the model never saw.
  List<int>? _shapeOf(TensorSpec spec, int byteLength, String key) {
    if (!spec.isDynamic) return null;

    final unknown = spec.shape.where((d) => d == TensorSpec.dynamicDim).length;
    if (unknown > 1) {
      throw TensorShapeException(
        'golden "$key" satisfies ${spec.shape}, which leaves $unknown '
        'dimensions to infer from a single byte count',
        tensorName: spec.name,
      );
    }

    final width = spec.dtype.bytesPerElement;
    if (byteLength % width != 0) {
      throw TensorShapeException(
        'golden "$key" holds $byteLength bytes, which is not a whole number of '
        '${spec.dtype.wireName} elements',
        tensorName: spec.name,
      );
    }

    final elements = byteLength ~/ width;
    final fixed = spec.shape
        .where((d) => d != TensorSpec.dynamicDim)
        .fold<int>(1, (a, b) => a * b);
    if (fixed == 0 || elements % fixed != 0) {
      throw TensorShapeException(
        'golden "$key" holds $elements elements, which no extent of the '
        'dynamic dimension in ${spec.shape} can produce',
        tensorName: spec.name,
      );
    }

    return [
      for (final d in spec.shape)
        if (d == TensorSpec.dynamicDim) elements ~/ fixed else d,
    ];
  }
}

/// Goldens held in memory, keyed exactly as the manifest names them.
///
/// What a unit test uses, and what a Flutter app builds once it has read its
/// assets. Resolving an asset key belongs to the app, which is the only party
/// that knows how its bundle was packed.
final class MemoryGoldenBundle extends BytesGoldenBundle {
  const MemoryGoldenBundle(super.manifest, this._tensors);

  final Map<String, Uint8List> _tensors;

  @override
  Future<Uint8List> bytesFor(String key) async {
    final bytes = _tensors[key];
    if (bytes == null) {
      throw StateError(
        'this bundle holds no golden "$key"; it holds '
        '${_tensors.isEmpty ? "nothing" : _tensors.keys.map((k) => '"$k"').join(", ")}',
      );
    }
    return bytes;
  }
}
