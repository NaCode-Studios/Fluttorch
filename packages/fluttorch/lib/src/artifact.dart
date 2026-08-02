import 'dart:convert';
import 'dart:typed_data';

import 'package:crypto/crypto.dart';

import 'errors.dart';
import 'manifest.dart';

/// Hash algorithms a manifest may name.
///
/// A closed set on purpose: an artifact whose hash this build cannot compute is
/// one it cannot verify, and quietly loading it would defeat the check.
enum HashAlgorithm {
  sha256('sha256');

  const HashAlgorithm(this.wireName);

  /// Prefix as written in [ModelManifest.weightHash], before the colon.
  final String wireName;

  static HashAlgorithm? tryParse(String name) {
    for (final a in values) {
      if (a.wireName == name) return a;
    }
    return null;
  }
}

/// Computes the digest a manifest would record for [artifact].
///
/// Returned in the manifest's own `algorithm:hex` form, so a mismatch can be
/// printed beside the expected value without either being reformatted.
String digestOf(
  Uint8List artifact, {
  HashAlgorithm algorithm = HashAlgorithm.sha256,
}) => switch (algorithm) {
  HashAlgorithm.sha256 => '${algorithm.wireName}:${sha256.convert(artifact)}',
};

/// Collects the one digest a chunked conversion emits.
///
/// Six lines rather than a dependency on `package:convert` for its
/// `AccumulatorSink`. The core has one runtime dependency and the reason to
/// stream at all is that a bundle's parts can be hundreds of megabytes, which
/// is not a size to concatenate in memory just to hash it once.
final class _DigestSink implements Sink<Digest> {
  Digest? value;

  @override
  void add(Digest data) => value = data;

  @override
  void close() {}
}

/// The digest that covers a whole bundle, [artifact] and [parts] together.
///
/// [parts] are fed in the order the manifest declares them, each contributing
/// its name and its length as well as its bytes. Concatenation alone would be
/// ambiguous, since two different splits of the same bytes would hash the same,
/// and a hash whose meaning depends on where you cut it commits to nothing.
///
/// With no parts this is [digestOf] and nothing more, byte for byte what every
/// manifest written before parts existed declares. That is what let the field
/// arrive without re-exporting anything.
String bundleDigestOf(
  Uint8List artifact,
  List<({String name, Uint8List bytes})> parts, {
  HashAlgorithm algorithm = HashAlgorithm.sha256,
}) {
  if (parts.isEmpty) return digestOf(artifact, algorithm: algorithm);

  final output = _DigestSink();
  final input = switch (algorithm) {
    HashAlgorithm.sha256 => sha256.startChunkedConversion(output),
  };
  input.add(artifact);
  for (final part in parts) {
    final name = utf8.encode(part.name);
    input
      ..add(_length(name.length))
      ..add(name)
      ..add(_length(part.bytes.length))
      ..add(part.bytes);
  }
  input.close();
  return '${algorithm.wireName}:${output.value}';
}

/// A length as the eight little-endian bytes the exporter writes.
Uint8List _length(int value) =>
    Uint8List(8)..buffer.asByteData().setUint64(0, value, Endian.little);

/// Verifies that [artifact] is the one [manifest] was written for.
///
/// This is the cheapest check in the pipeline and the one that prevents the
/// most expensive failure: a manifest paired with weights it was not generated
/// from produces a parity suite that passes over a model nobody evaluated. The
/// shapes still line up, the goldens still load, and every number is wrong.
///
/// [parts] carries the files the manifest declares under `parts`, keyed by the
/// name it declares them under. A manifest that declares none, which is nearly
/// all of them, needs nothing here and behaves exactly as it did before the
/// field existed.
///
/// Throws [ArtifactMismatchException] when the digests differ,
/// [BundlePartMissingException] when a declared part did not arrive, and
/// [ManifestFormatException] when the manifest names a hash this build cannot
/// compute — three different problems with three different fixes, which is why
/// they are three exceptions and not one.
void verifyArtifact({
  required Uint8List artifact,
  required ModelManifest manifest,
  Map<String, Uint8List> parts = const {},
}) {
  final declared = manifest.weightHash;
  final colon = declared.indexOf(':');
  if (colon <= 0) {
    throw ManifestFormatException(
      'weight hash "$declared" is not in algorithm:hex form',
      field: 'weight_hash',
    );
  }

  final name = declared.substring(0, colon);
  final algorithm = HashAlgorithm.tryParse(name);
  if (algorithm == null) {
    throw ManifestFormatException(
      'weight hash names "$name", which this build cannot compute; it knows '
      '${HashAlgorithm.values.map((a) => a.wireName).join(", ")}',
      field: 'weight_hash',
    );
  }

  // Named before hashed. A missing part and a wrong one are different mistakes
  // with different fixes, and folding the first into a digest mismatch would
  // send a reader to re-export a model whose export was fine.
  final missing = [
    for (final part in manifest.parts)
      if (!parts.containsKey(part.name)) part.name,
  ];
  if (missing.isNotEmpty) {
    throw BundlePartMissingException(missing: missing, model: manifest.name);
  }

  final supplied = <({String name, Uint8List bytes})>[];
  for (final part in manifest.parts) {
    final bytes = parts[part.name]!;
    // Checked here so a truncated part is reported as itself rather than as a
    // bundle whose hash moved, which names the file and not the cause.
    if (bytes.length != part.size) {
      throw ArtifactMismatchException(
        expectedHash: part.hash,
        actualHash:
            'a part of ${bytes.length} bytes where "${part.name}" declares '
            '${part.size}',
      );
    }
    final actual = digestOf(bytes, algorithm: algorithm);
    if (actual != part.hash) {
      throw ArtifactMismatchException(
        expectedHash: '${part.hash} for "${part.name}"',
        actualHash: actual,
      );
    }
    supplied.add((name: part.name, bytes: bytes));
  }

  final actual = bundleDigestOf(artifact, supplied, algorithm: algorithm);
  if (actual != declared) {
    throw ArtifactMismatchException(expectedHash: declared, actualHash: actual);
  }
}
