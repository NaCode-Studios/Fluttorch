/// Turns a model manifest into a typed Dart API.
///
/// The generated class carries the model's shapes, dtypes, labels and
/// preprocessing, so the Dart side cannot disagree with the Python side that
/// produced them. Each tensor becomes its own type, which is what makes handing
/// `run` the wrong one a compile error rather than a shape mismatch discovered
/// at inference.
library;

export 'src/builder.dart' show fluttorchBuilder;
export 'src/emitter.dart' show emit;
export 'src/generator.dart';
export 'src/names.dart';
