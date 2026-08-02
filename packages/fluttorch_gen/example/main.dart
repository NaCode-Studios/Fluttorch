// What the builder produces, and why it is not written by hand.
//
// `fluttorch_gen` turns a manifest into Dart where the compiler rejects the
// wrong tensor. Normally `build_runner` runs it; here it is called directly so
// the output can be printed.
import 'dart:convert';

import 'package:fluttorch/fluttorch.dart';
import 'package:fluttorch_gen/fluttorch_gen.dart';

void main() {
  final manifest = ManifestCodec.decode(
    jsonEncode({
      'schema_version': 1,
      'name': 'two_layer',
      'weight_hash': 'sha256:${'0' * 64}',
      'inputs': [
        {
          'name': 'features',
          'dtype': 'float32',
          'shape': [1, 4],
        },
      ],
      'outputs': [
        {
          'name': 'score',
          'dtype': 'float32',
          'shape': [1, 3],
        },
      ],
      'preprocessing': [
        {'kind': 'rescale', 'factor': 0.00392156862745098, 'offset': 0.0},
      ],
    }),
  );

  final source = emit(manifest, sourceName: 'two_layer.fluttorch.json');
  print(source);

  // The rescale above becomes arithmetic in the generated `preprocess()`. That
  // is the point: a transform training applied and the app did not is invisible
  // at every later stage, so it is generated from the same document rather than
  // reimplemented from a comment.

  // A manifest this build cannot honour is refused, with every reason at once
  // rather than the first one found.
  final unsupported = ModelManifest(
    name: 'demo',
    schemaVersion: ModelManifest.currentSchemaVersion,
    weightHash: 'sha256:${'0' * 64}',
    inputs: const [
      TensorSpec(name: 'x', dtype: DType.float32, shape: [1, 4]),
    ],
    outputs: const [
      TensorSpec(name: 'y', dtype: DType.float32, shape: [1, 2]),
    ],
    preprocessing: const [ResizeStep(height: 8, width: 8)],
  );
  try {
    emit(unsupported, sourceName: 'demo.fluttorch.json');
  } on UnsupportedManifestException catch (e) {
    print(e);
  }
}
