import 'dart:typed_data';

import 'dtype.dart';
import 'errors.dart';

/// The declared shape and type of one model input or output.
///
/// A dimension of [dynamicDim] is decided at run time. Everything else is fixed
/// at export, and the generated bindings turn it into a compile-time signature.
final class TensorSpec {
  const TensorSpec({
    required this.name,
    required this.dtype,
    required this.shape,
  });

  /// Marks a dimension whose extent is not known until run time.
  static const int dynamicDim = -1;

  /// Name as reported by the exported graph, used to key the generated API.
  final String name;

  /// Element type. A backend that cannot handle it fails at load rather than
  /// coercing silently.
  final DType dtype;

  /// Dimensions, innermost last. [dynamicDim] marks a dynamic extent.
  final List<int> shape;

  int get rank => shape.length;

  /// Whether any dimension is unknown until run time.
  bool get isDynamic => shape.contains(dynamicDim);

  /// Number of elements, or null when any dimension is dynamic.
  ///
  /// Prefer [elementCountFor] when a concrete shape is in hand: this getter
  /// exists for the static case and returning null is the honest answer for the
  /// other one, not a reason to guess.
  int? get elementCount => isDynamic ? null : _product(shape);

  /// Number of elements a tensor of [actualShape] holds.
  ///
  /// Throws [TensorShapeException] if [actualShape] does not satisfy this spec.
  int elementCountFor(List<int> actualShape) => _product(resolve(actualShape));

  /// Bytes a tensor of [actualShape] occupies.
  int byteLengthFor(List<int> actualShape) =>
      elementCountFor(actualShape) * dtype.bytesPerElement;

  /// Whether [actualShape] satisfies this spec: equal rank, equal static
  /// dimensions, and a positive extent wherever this spec is dynamic.
  bool accepts(List<int> actualShape) {
    if (actualShape.length != rank) return false;
    for (var i = 0; i < rank; i++) {
      final declared = shape[i];
      final actual = actualShape[i];
      if (declared == dynamicDim) {
        if (actual < 0) return false;
      } else if (declared != actual) {
        return false;
      }
    }
    return true;
  }

  /// Returns the concrete shape to use, validating it against this spec.
  ///
  /// Passing null is only valid for a fully static spec; a dynamic spec has no
  /// shape to fall back on and says so rather than inventing one.
  List<int> resolve(List<int>? actualShape) {
    if (actualShape == null) {
      if (isDynamic) {
        throw TensorShapeException(
          'spec is dynamic ($shape) so a concrete shape must be supplied',
          tensorName: name,
        );
      }
      return shape;
    }
    if (!accepts(actualShape)) {
      throw TensorShapeException(
        'shape $actualShape does not satisfy the declared $shape',
        tensorName: name,
      );
    }
    return actualShape;
  }

  static int _product(List<int> dims) => dims.fold<int>(1, (a, b) => a * b);

  @override
  String toString() => '$name: ${dtype.wireName}$shape';
}

/// A tensor: a spec, a concrete shape, and the bytes that satisfy both.
///
/// Constructing one is the only place the contract's central invariant is
/// checked — `bytes.length == elements × dtype.bytesPerElement` — so a tensor
/// that exists is a tensor that agrees with its declaration. Nothing downstream
/// has to re-derive it.
///
/// The bytes are never copied. [Tensor.view] wraps a buffer the caller already
/// owns, which is what makes it possible to reuse one allocation across many
/// inferences instead of paying a copy per call.
final class Tensor {
  Tensor._(this.spec, this.shape, this.bytes);

  /// Wraps [bytes] without copying.
  ///
  /// [shape] may be omitted only when [spec] is fully static. Throws
  /// [TensorShapeException] if the shape does not satisfy the spec, if the
  /// buffer is the wrong length, or if its offset is not aligned for the
  /// element type.
  factory Tensor.view({
    required TensorSpec spec,
    required Uint8List bytes,
    List<int>? shape,
  }) {
    final resolved = spec.resolve(shape);
    final expected = spec.byteLengthFor(resolved);
    if (bytes.length != expected) {
      throw TensorShapeException(
        'buffer is ${bytes.length} bytes; $resolved of ${spec.dtype.wireName} '
        'needs exactly $expected',
        tensorName: spec.name,
      );
    }
    final width = spec.dtype.bytesPerElement;
    if (width > 1 && bytes.offsetInBytes % width != 0) {
      throw TensorShapeException(
        'buffer starts at byte ${bytes.offsetInBytes}, which is not aligned '
        'to the $width-byte ${spec.dtype.wireName} element',
        tensorName: spec.name,
      );
    }
    return Tensor._(spec, List.unmodifiable(resolved), bytes);
  }

  /// Allocates a zeroed tensor, for use as a destination in
  /// [LoadedModel.runInto] or as a scratch buffer.
  factory Tensor.zeros(TensorSpec spec, {List<int>? shape}) {
    final resolved = spec.resolve(shape);
    return Tensor._(
      spec,
      List.unmodifiable(resolved),
      Uint8List(spec.byteLengthFor(resolved)),
    );
  }

  final TensorSpec spec;

  /// Concrete shape. Never contains [TensorSpec.dynamicDim].
  final List<int> shape;

  /// Raw little-endian bytes. The only representation available for types
  /// without a Dart list view, such as `float16`.
  final Uint8List bytes;

  int get elementCount => bytes.length ~/ spec.dtype.bytesPerElement;

  /// A `Float32List` over the same memory, without copying.
  ///
  /// Throws [DTypeMismatchException] unless this tensor holds `float32`.
  Float32List asFloat32List() =>
      _view(DType.float32, (b, o, n) => b.asFloat32List(o, n));

  /// A `Float64List` over the same memory, without copying.
  Float64List asFloat64List() =>
      _view(DType.float64, (b, o, n) => b.asFloat64List(o, n));

  /// An `Int8List` over the same memory, without copying.
  Int8List asInt8List() => _view(DType.int8, (b, o, n) => b.asInt8List(o, n));

  /// An `Int16List` over the same memory, without copying.
  Int16List asInt16List() =>
      _view(DType.int16, (b, o, n) => b.asInt16List(o, n));

  /// An `Int32List` over the same memory, without copying.
  Int32List asInt32List() =>
      _view(DType.int32, (b, o, n) => b.asInt32List(o, n));

  /// An `Int64List` over the same memory, without copying.
  Int64List asInt64List() =>
      _view(DType.int64, (b, o, n) => b.asInt64List(o, n));

  /// A `Uint8List` over the same memory. Valid for `uint8` and `bool`, which
  /// share a representation.
  Uint8List asUint8List() {
    if (spec.dtype != DType.uint8 && spec.dtype != DType.boolean) {
      throw DTypeMismatchException(
        tensorName: spec.name,
        declared: spec.dtype,
        requested: DType.uint8,
      );
    }
    return bytes;
  }

  T _view<T>(DType want, T Function(ByteBuffer, int, int) make) {
    if (spec.dtype != want) {
      throw DTypeMismatchException(
        tensorName: spec.name,
        declared: spec.dtype,
        requested: want,
      );
    }
    return make(bytes.buffer, bytes.offsetInBytes, elementCount);
  }

  @override
  String toString() =>
      'Tensor(${spec.name}: ${spec.dtype.wireName}$shape, '
      '${bytes.length} bytes)';
}

/// Validates a positional list of tensors against the specs it must satisfy.
///
/// Used at the runtime seam, where a caller hands over a list and the
/// correspondence to [ModelManifest.inputs] would otherwise be a convention
/// nothing enforces.
void checkTensorsAgainst(
  List<TensorSpec> specs,
  List<Tensor> tensors, {
  required String role,
}) {
  if (tensors.length != specs.length) {
    throw TensorShapeException(
      'expected ${specs.length} $role, got ${tensors.length}',
    );
  }
  for (var i = 0; i < specs.length; i++) {
    final spec = specs[i];
    final tensor = tensors[i];
    if (tensor.spec.name != spec.name) {
      throw TensorShapeException(
        '$role[$i] is "${tensor.spec.name}" where the model declares '
        '"${spec.name}" — the list is positional and out of order',
        tensorName: spec.name,
      );
    }
    if (tensor.spec.dtype != spec.dtype) {
      throw DTypeMismatchException(
        tensorName: spec.name,
        declared: spec.dtype,
        requested: tensor.spec.dtype,
      );
    }
    if (!spec.accepts(tensor.shape)) {
      throw TensorShapeException(
        'shape ${tensor.shape} does not satisfy the declared ${spec.shape}',
        tensorName: spec.name,
      );
    }
  }
}
