@TestOn('vm')
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';
import 'package:test/test.dart';
import 'package:voltacast_example/voltacast.fluttorch.dart';

/// M29 · the typed API for a model with more than one input.
///
/// No runtime here. Executing VoltaCast is blocked inside ExecuTorch, which the
/// on-device suite records; what this covers is everything up to that point,
/// and it is the half a consumer of the generated code actually touches.
final _bundle = Directory('../../testdata/voltacast');

void main() {
  Float32List read(String path) {
    final b = File('${_bundle.path}/$path').readAsBytesSync();
    return Float32List.view(b.buffer, b.offsetInBytes, b.lengthInBytes ~/ 4);
  }

  group('M29 · two inputs, and the compiler tells them apart', () {
    test('each input got its own type', () {
      // The reason the generator emits a type per tensor rather than taking a
      // list: past is 168 hours by 13 features and future is 24 by 12, both
      // float32, so nothing but the type stops them being swapped.
      expect(VoltacastPast.spec.shape, [1, 168, 13]);
      expect(VoltacastFuture.spec.shape, [1, 24, 12]);
      expect(VoltacastQuantiles.spec.shape, [1, 24, 3]);
      expect(VoltacastOutputs.labels, ['p10', 'p50', 'p90']);
    });

    test('a buffer of the wrong length is refused', () {
      expect(
        () => VoltacastPast(Float32List(168 * 12)),
        throwsA(isA<TensorShapeException>()),
      );
      // And the refusal says what to do, not only what happened.
      try {
        VoltacastFuture(Float32List(10));
      } on FluttorchException catch (e) {
        expect(e.remedy, isNotEmpty);
      }
    });

    test('the committed goldens satisfy the specs they were captured for', () {
      // A bundle whose tensors do not match its own manifest would fail at the
      // device with a message about memory rather than about shapes.
      final past = VoltacastPast(read('goldens/0/in/past.bin'));
      final future = VoltacastFuture(read('goldens/0/in/future.bin'));
      expect(past.tensor.shape, [1, 168, 13]);
      expect(future.tensor.shape, [1, 24, 12]);
    });

    test('the reference forecast is a prediction interval', () {
      // The head builds each quantile by adding a softplus increment to the one
      // below, so P10 <= P50 <= P90 holds for any input. Checked on the
      // captured reference, which is what the device will be measured against.
      final q = read('goldens/0/out/quantiles.bin');
      for (var h = 0; h < 24; h++) {
        expect(q[h * 3], lessThanOrEqualTo(q[h * 3 + 1]), reason: 'hour $h');
        expect(q[h * 3 + 1], lessThanOrEqualTo(q[h * 3 + 2]), reason: 'hour $h');
      }
    });

    test('the forecast is a plausible amount of electricity', () {
      // The tensors are in the training scaler's units, so this converts back
      // with the statistics the checkpoint carries. Italy draws roughly 20 to
      // 60 GW, and a number outside that would mean the scaler and the artifact
      // came from different runs.
      const mean = 32605.321413703383;
      const std = 7439.5645181078135;
      final q = read('goldens/0/out/quantiles.bin');
      for (var h = 0; h < 24; h++) {
        final mw = q[h * 3 + 1] * std + mean;
        expect(mw, inInclusiveRange(20000, 60000), reason: 'hour $h: $mw MW');
      }
    });
  });
}
