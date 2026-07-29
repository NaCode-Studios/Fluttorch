/// Turns a model manifest into a typed Dart API.
///
/// The generated class carries the model's shapes, dtypes, labels and
/// preprocessing, so the Dart side cannot disagree with the Python side that
/// produced them.
library;

import 'package:build/build.dart';

/// Entry point for `build_runner`. Consumes `*.fluttorch.json` manifests and
/// emits `*.fluttorch.dart`.
Builder fluttorchBuilder(BuilderOptions options) =>
    throw UnimplementedError('see the roadmap: typed codegen');
