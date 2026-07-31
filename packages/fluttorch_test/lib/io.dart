/// Goldens read from the filesystem.
///
/// Separate from the main library so that `dart:io` reaches only the suites
/// that actually read files. A Flutter test loading goldens through the asset
/// bundle wants [MemoryGoldenBundle] and this import would stop it compiling
/// for the web.
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';

import 'src/bundle.dart';

/// Goldens on disk, laid out as `fluttorch-export` wrote them.
///
/// Keys are paths relative to [root], exactly as the manifest records them, so
/// a bundle is one directory plus one manifest with nothing in between to keep
/// in step.
final class DirectoryGoldenBundle extends BytesGoldenBundle {
  const DirectoryGoldenBundle(super.manifest, {required this.root});

  /// Directory holding the goldens, which the exporter writes as `goldens/`
  /// beside the artifact.
  final String root;

  /// Reads the manifest at [manifestPath] and takes its goldens from [root],
  /// which defaults to the `goldens/` directory written beside it.
  ///
  /// The pairing is the point: a manifest read from one export and goldens read
  /// from another produce a suite that measures nothing, so the common case
  /// should not require naming two paths.
  static Future<DirectoryGoldenBundle> open(
    String manifestPath, {
    String? root,
  }) async {
    final file = File(manifestPath);
    final manifest = ManifestCodec.decode(await file.readAsString());
    return DirectoryGoldenBundle(
      manifest,
      root: root ?? '${file.parent.path}/goldens',
    );
  }

  @override
  Future<Uint8List> bytesFor(String key) async {
    final file = File('$root/$key');
    if (!file.existsSync()) {
      throw StateError(
        'golden "$key" is not under "$root". The exporter writes the goldens '
        'beside the artifact, so the directory has to travel with it: check '
        'that the build fetches or commits it.',
      );
    }
    return file.readAsBytes();
  }
}
