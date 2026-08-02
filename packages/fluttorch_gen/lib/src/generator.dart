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
  final spatial = manifest.preprocessing.indexed
      .where((e) => e.$2 is ResizeStep || e.$2 is CenterCropStep)
      .toList();
  final elementwise = manifest.preprocessing.indexed
      .where((e) => e.$2 is RescaleStep || e.$2 is NormalizeStep)
      .toList();

  final reasons = <String>[
    for (final step in manifest.preprocessing)
      if (step is UnknownPreprocessingStep)
        'preprocessing step "${step.kind}" is not understood by this build, '
            'and skipping it would silently drop a transform the model was '
            'trained with',
    // Resize and crop can be performed only against a declared layout.
    // Guessing would be worse than refusing: NCHW and NHWC read the same bytes
    // as different pictures, and a resize down the wrong axes returns plausible
    // numbers rather than an error.
    if (spatial.isNotEmpty && manifest.inputs.length != 1)
      'the manifest records ${spatial.length} spatial step(s) and the model '
          'takes ${manifest.inputs.length} inputs, so there is no way to tell '
          'which one they apply to',
    if (spatial.isNotEmpty && manifest.inputs.length == 1)
      if (manifest.inputs.single.layout == null)
        'preprocessing step "${spatial.first.$2.kind}" needs to know which axes '
            'are spatial, and "${manifest.inputs.single.name}" declares no '
            'layout. Record one at export: NCHW and NHWC would each produce a '
            'plausible and different answer',
    // Order is part of the contract, not a detail the generator may normalise.
    // Bilinear resize and an affine rescale commute in exact arithmetic, so
    // reordering them looks safe and is not: a cast between them, or the
    // rounding of either, moves the result by an amount nobody would look for.
    if (spatial.isNotEmpty && elementwise.isNotEmpty)
      if (spatial.last.$1 > elementwise.first.$1)
        'the recorded order applies "${elementwise.first.$2.kind}" before '
            '"${spatial.last.$2.kind}", and this generator emits spatial steps '
            'first. Re-record the pipeline with resize and center_crop ahead of '
            'the elementwise steps rather than having the generated code '
            'silently reorder what training did',
    // Named rather than approximated. Bicubic needs a four-tap kernel whose
    // coefficient torch and PIL do not agree on, and emitting bilinear in its
    // place would be a resize that runs and is not the one training used.
    for (final step in manifest.preprocessing)
      if (step is ResizeStep &&
          step.interpolation != 'bilinear' &&
          step.interpolation != 'nearest')
        'resize records the "${step.interpolation}" filter, and this generator '
            'emits bilinear and nearest. Substituting one would reproduce a '
            'resize that is not the one training applied',
    if (manifest.inputs.isEmpty) 'the manifest declares no inputs',
    if (manifest.outputs.isEmpty) 'the manifest declares no outputs',
    for (final spec in [...manifest.inputs, ...manifest.outputs])
      if (spec.name.isEmpty)
        'a tensor has an empty name, so no accessor can be generated for it',
    for (final step in manifest.preprocessing)
      if (step is NormalizeStep && manifest.inputs.length != 1)
        'normalize is recorded once but the model takes '
            '${manifest.inputs.length} inputs, so there is no way to tell which '
            'one it applies to',
  ];
  if (reasons.isNotEmpty) throw UnsupportedManifestException(reasons);
}
