import 'dart:async';

import 'package:build/build.dart';
import 'package:fluttorch/fluttorch.dart';

import 'emitter.dart';

/// The `build_runner` entry point: `*.fluttorch.json` in, `*.fluttorch.dart` out.
///
/// Thin on purpose. Everything that decides what the generated code says lives
/// in [emit], which is a pure function of the manifest and can be tested — and
/// pinned — without running a build.
Builder fluttorchBuilder(BuilderOptions options) => const _FluttorchBuilder();

class _FluttorchBuilder implements Builder {
  const _FluttorchBuilder();

  @override
  Map<String, List<String>> get buildExtensions => const {
    '.fluttorch.json': ['.fluttorch.dart'],
  };

  @override
  Future<void> build(BuildStep buildStep) async {
    final input = buildStep.inputId;
    final source = await buildStep.readAsString(input);

    final ModelManifest manifest;
    try {
      manifest = ManifestCodec.decode(source);
    } on FluttorchException catch (e) {
      // Fail the build rather than emitting nothing: a missing generated file
      // surfaces as an unresolved import somewhere else entirely.
      log.severe('${input.path}: $e');
      rethrow;
    }

    await buildStep.writeAsString(
      input.changeExtension('.dart'),
      emit(manifest, sourceName: input.pathSegments.last),
    );
  }
}
