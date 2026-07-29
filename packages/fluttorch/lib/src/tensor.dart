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

  final String name;
  final DType dtype;
  final List<int> shape;

  bool get isDynamic => shape.contains(-1);

  /// Number of elements, or `null` when any dimension is dynamic.
  int? get elementCount {
    if (isDynamic) return null;
    return shape.fold(1, (a, b) => a * b);
  }
}
