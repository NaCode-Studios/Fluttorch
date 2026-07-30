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

/// Verifies that [artifact] is the one [manifest] was written for.
///
/// This is the cheapest check in the pipeline and the one that prevents the
/// most expensive failure: a manifest paired with weights it was not generated
/// from produces a parity suite that passes over a model nobody evaluated. The
/// shapes still line up, the goldens still load, and every number is wrong.
///
/// Throws [ArtifactMismatchException] when the digests differ, and
/// [ManifestFormatException] when the manifest names a hash this build cannot
/// compute — which is a different problem with a different fix, and must not be
/// mistaken for a mismatch.
void verifyArtifact({
  required Uint8List artifact,
  required ModelManifest manifest,
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

  final actual = digestOf(artifact, algorithm: algorithm);
  if (actual != declared) {
    throw ArtifactMismatchException(expectedHash: declared, actualHash: actual);
  }
}
