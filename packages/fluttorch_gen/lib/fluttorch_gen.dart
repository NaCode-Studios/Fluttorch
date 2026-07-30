/// Turns a model manifest into a typed Dart API.
///
/// The generated class carries the model's shapes, dtypes, labels and
/// preprocessing, so the Dart side cannot disagree with the Python side that
/// produced them.
///
/// Not wired into `build_runner` yet: a builder factory is only declared once it
/// returns a builder, because declaring one that throws turns adding this
/// package as a dependency into a build failure rather than a no-op. The factory
/// and its `build.yaml` land together in M9.
library;

export 'src/generator.dart';
