/// The parity gate.
///
/// Replays the goldens captured at export time against a loaded model, measures
/// how far the on-device numbers moved, and fails the build when they moved too
/// far. This is where the tolerance semantics and the drift metrics live:
/// nothing on the inference path needs them, so an app shipping a model does not
/// carry them.
///
/// Everything here is platform-neutral. The bundle that reads goldens from a
/// directory lives in `package:fluttorch_test/io.dart`, so that importing the
/// gate does not drag `dart:io` into a suite that runs on the web.
library;

export 'src/bundle.dart';
export 'src/drift.dart';
export 'src/matrix.dart';
export 'src/parity.dart';
export 'src/tolerance.dart';
