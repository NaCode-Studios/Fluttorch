// The contract, without a backend.
//
// `fluttorch` describes a model and says nothing about running one: reading a
// manifest, checking a buffer against what the export declared, and refusing an
// artifact that does not match. A backend package supplies the rest.
import 'dart:convert';
import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';

void main() {
  // Written by `fluttorch-export`, never by hand. Trimmed here to what this
  // example reads.
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
    }),
  );

  print('${manifest.name}: ${manifest.inputs.length} in, '
      '${manifest.outputs.length} out');

  final spec = manifest.inputNamed('features');
  print('features is ${spec.dtype.wireName}${spec.shape}');

  // A buffer is checked against the spec rather than trusted. The alternative
  // is a shape error discovered inside a delegate, where the message names
  // nothing a reader recognises.
  final values = Float32List.fromList([1, 2, 3, 4]);
  final tensor = Tensor.view(
    spec: spec,
    bytes: values.buffer.asUint8List(),
  );
  print('accepted ${tensor.shape}, ${tensor.bytes.length} bytes');

  try {
    Tensor.view(
      spec: spec,
      bytes: Float32List.fromList([1, 2]).buffer.asUint8List(),
    );
  } on FluttorchException catch (e) {
    // Every failure says what to do about it, not only what happened.
    print('refused: ${e.message}');
    print('     do: ${e.remedy}');
  }
}
