import 'package:fluttorch/fluttorch.dart';

/// Why a manifest cannot be turned into a typed API.
///
/// Refusing is the point. A generator that skipped what it did not understand
/// would emit an API missing a transform the model was trained with, and the
/// resulting skew is invisible at every later stage.
final class UnsupportedManifestException implements Exception {
  const UnsupportedManifestException(this.reasons);

  /// Everything wrong with the manifest, not just the first thing found — a
  /// generator that reports one problem per run is a generator run many times.
  final List<String> reasons;

  @override
  String toString() =>
      'UnsupportedManifestException:\n${reasons.map((r) => '  · $r').join('\n')}';
}

/// Checks whether [manifest] can be generated from, and throws
/// [UnsupportedManifestException] listing every reason it cannot.
///
/// Separated from generation so the exporter, the build step and a test can all
/// ask the same question without producing source.
void checkGeneratable(ModelManifest manifest) {
  final reasons = <String>[
    for (final step in manifest.preprocessing)
      if (step is UnknownPreprocessingStep)
        'preprocessing step "${step.kind}" is not understood by this build, '
            'and skipping it would silently drop a transform the model was '
            'trained with',
    if (manifest.inputs.isEmpty) 'the manifest declares no inputs',
    if (manifest.outputs.isEmpty) 'the manifest declares no outputs',
    for (final spec in [...manifest.inputs, ...manifest.outputs])
      if (spec.name.isEmpty)
        'a tensor has an empty name, so no accessor can be generated for it',
  ];
  if (reasons.isNotEmpty) throw UnsupportedManifestException(reasons);
}
