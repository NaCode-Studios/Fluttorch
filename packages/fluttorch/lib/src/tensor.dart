/// Element types Fluttorch can describe. Not every runtime supports every one;
/// a backend reports what it can handle via [FluttorchRuntime.capabilities].
enum DType {
  float32,
  float64,
  float16,
  bfloat16,
  int8,
  int16,
  int32,
  int64,
  uint8,
  bool_,
}

/// The declared shape and type of one model input or output.
///
/// A dimension of `-1` is dynamic and is not checked at compile time.
class TensorSpec {
  const TensorSpec({
    required this.name,
    required this.dtype,
    required this.shape,
  });

  /// Name as reported by the exported graph, used to key the generated API.
  final String name;

  /// Element type. A backend that cannot handle it must fail at load time
  /// rather than coerce silently.
  final DType dtype;

  /// Dimensions, innermost last. `-1` marks a dynamic dimension.
  final List<int> shape;

  /// Whether any dimension is unknown until run time.
  bool get isDynamic => shape.contains(-1);

  /// Number of elements, or `null` when any dimension is dynamic.
  int? get elementCount {
    if (isDynamic) return null;
    return shape.fold<int>(1, (a, b) => a * b);
  }
}
