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
  const DirectoryGoldenBundle(
    super.manifest, {
    required this.root,
    this.bundleRoot,
  });

  /// Directory holding the goldens, which the exporter writes as `goldens/`
  /// beside the artifact.
  final String root;

  /// Directory the manifest itself was read from.
  ///
  /// Separate from [root] because a bundle's parts sit beside the manifest
  /// rather than among the goldens: one is the contract and its artifact, the
  /// other is the evidence. Null when the bundle was built without opening a
  /// manifest file, in which case [parts] has nowhere to look and says so.
  final String? bundleRoot;

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
      bundleRoot: file.parent.path,
    );
  }

  /// The files the manifest declares under `parts`, read from beside it.
  ///
  /// Empty for a model whose weights fit inside its graph, which is nearly all
  /// of them, so a caller can pass this to a runtime without asking which shape
  /// of bundle it holds.
  Future<Map<String, Uint8List>> parts() async {
    if (manifest.parts.isEmpty) return const {};
    final dir = bundleRoot;
    if (dir == null) {
      throw StateError(
        'the manifest for "${manifest.name}" declares '
        '${manifest.parts.length} part(s), and this bundle was built without a '
        'directory to read them from. Open it with '
        'DirectoryGoldenBundle.open so the parts come from beside the manifest.',
      );
    }
    final read = <String, Uint8List>{};
    for (final part in manifest.parts) {
      final file = File('$dir/${part.name}');
      if (!file.existsSync()) {
        // Named rather than left to the loader. Both files travel together or
        // neither is usable, and the one that goes missing is the one carrying
        // the numbers: the graph alone still parses and still answers.
        throw BundlePartMissingException(
          missing: [part.name],
          model: manifest.name,
        );
      }
      read[part.name] = await file.readAsBytes();
    }
    return read;
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
