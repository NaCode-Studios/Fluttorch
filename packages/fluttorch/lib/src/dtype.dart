import 'dart:typed_data';

/// Element types Fluttorch can describe.
///
/// Not every runtime supports every one; a backend declares what it can handle
/// through [RuntimeCapabilities.dtypes]. Every member carries its width, because
/// the contract's central invariant — a buffer is exactly `elements × width`
/// bytes — is otherwise unverifiable, and each backend would reimplement the
/// arithmetic differently.
enum DType {
  float32('float32', 4),
  float64('float64', 8),

  /// 16-bit IEEE half. Dart has no half-precision list, so [hasTypedListView]
  /// is false and the bytes must be widened before arithmetic.
  float16('float16', 2, hasTypedListView: false),

  /// Brain float. Same story as [float16]: eight bits of exponent, no Dart view.
  bfloat16('bfloat16', 2, hasTypedListView: false),

  int8('int8', 1),
  int16('int16', 2),
  int32('int32', 4),
  int64('int64', 8),
  uint8('uint8', 1),

  /// One byte per element, zero for false, non-zero for true — matching how
  /// every runtime in scope serialises booleans.
  boolean('bool', 1);

  const DType(
    this.wireName,
    this.bytesPerElement, {
    this.hasTypedListView = true,
  });

  /// Name used in the manifest. Fixed independently of the Dart identifier, so
  /// renaming a member never invalidates an artifact already on disk.
  final String wireName;

  /// Width of one element in bytes.
  final int bytesPerElement;

  /// Whether `dart:typed_data` offers a list view over this type. When false,
  /// [Tensor.bytes] is the only representation available.
  final bool hasTypedListView;

  /// Whether values of this type carry a fraction, and therefore whether a
  /// difference between two of them is a rounding error rather than a mismatch.
  bool get isFloatingPoint =>
      this == float32 || this == float64 || this == float16 || this == bfloat16;

  static final Map<String, DType> _byWireName = {
    for (final d in values) d.wireName: d,
  };

  /// Resolves a [wireName] as written in a manifest.
  ///
  /// Returns null rather than throwing, so the manifest decoder can report the
  /// offending field and its position instead of an opaque failure.
  static DType? tryParse(String wireName) => _byWireName[wireName];

  /// Number of bytes a tensor of this type and [shape] occupies.
  int byteLengthFor(Iterable<int> shape) =>
      shape.fold<int>(1, (a, b) => a * b) * bytesPerElement;
}

/// Endianness of a tensor buffer.
///
/// Every runtime in scope is little-endian on every platform Fluttorch targets,
/// so this exists to make the assumption explicit and to give a mismatch
/// somewhere to be reported, not because big-endian is expected.
const Endian tensorEndian = Endian.little;
