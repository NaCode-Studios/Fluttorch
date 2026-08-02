@TestOn('vm')
library;

import 'dart:io';

import 'package:fluttorch/fluttorch.dart';
import 'package:fluttorch_litert/fluttorch_litert.dart';
import 'package:fluttorch_test/fluttorch_test.dart';
import 'package:fluttorch_test/io.dart';
import 'package:test/test.dart';

/// M29 · the flagship model, on the runtime that will carry it.
///
/// VoltaCast forecasts Italian electricity demand a day ahead: a seq2seq
/// Transformer, three encoder layers over a week of hourly history and two
/// decoder layers cross-attending to them, trained on eleven years of real
/// data. Everything else measured in this repository is a two-layer network.
///
/// It is here rather than under ExecuTorch because ExecuTorch lowers it and
/// then fails to execute it, which the on-device suite records. Carrying the
/// same bundle to a different engine is what Tier 7 was built for, and this is
/// the first time that has been worth anything rather than merely demonstrated.
final _library = File(
  '.dart_tool/native/libfluttorch_litert'
  '${Platform.isMacOS ? ".dylib" : ".so"}',
);
final _export = Directory('../../testdata/voltacast_litert');

void main() {
  if (!_library.existsSync() || !_export.existsSync()) {
    test('the VoltaCast gate needs tool/build_native.sh', () {}, skip: true);
    return;
  }

  late DirectoryGoldenBundle goldens;
  late LoadedModel model;
  final runtime = LiteRtRuntime.open(_library.path);

  setUpAll(() async {
    goldens = await DirectoryGoldenBundle.open(
      '${_export.path}/voltacast.fluttorch.json',
    );
    model = await runtime.load(
      artifact: await File('${_export.path}/voltacast.tflite').readAsBytes(),
      manifest: goldens.manifest,
      backend: 'cpu',
    );
  });

  tearDownAll(() async => model.dispose());

  group('M29 · a real forecast, measured on this machine', () {
    test('two inputs, three quantiles, four real windows', () {
      expect(model.manifest.inputs.map((s) => s.name), ['past', 'future']);
      expect(model.manifest.inputNamed('past').shape, [1, 168, 13]);
      expect(model.manifest.inputNamed('future').shape, [1, 24, 12]);
      expect(model.manifest.outputNamed('quantiles').shape, [1, 24, 3]);
      expect(goldens.cases, hasLength(4));
    });

    test('every golden holds at the full-precision bound', () async {
      // The references were captured from the source model in Python, on real
      // midnight origins. What this measures is the engine: same weights, same
      // inputs, a converter and a runtime in between.
      await expectParity(model, goldens: goldens);
    });

    test('the drift is conversion rounding, not a different model', () async {
      final reports = await measureParity(model, goldens: goldens);
      expect(reports, hasLength(goldens.cases.length));
      expect(reports.every((r) => r.passes), isTrue);
      final worst = reports
          .map((r) => r.tensors.single.maxAbsolute)
          .reduce((a, b) => a > b ? a : b);
      // Stated as a number so a future run can be compared against it. On a
      // 340-node transformer through a converter, this is what agreement looks
      // like.
      print('VoltaCast worst absolute drift through LiteRT: $worst');
      expect(worst, lessThan(1e-2), reason: reports.first.describe());
    });

    test('the forecast is still a prediction interval', () async {
      // The head adds a softplus increment per quantile, so P10 <= P50 <= P90
      // holds for any input. If the converter broke that, the output would
      // still be numbers and would no longer be a forecast.
      final c = goldens.cases.first;
      final inputs = [
        for (var i = 0; i < model.manifest.inputs.length; i++)
          await goldens.tensor(c.inputKeys[i], model.manifest.inputs[i]),
      ];
      final v = (await model.run(inputs)).single.asFloat32List();
      for (var h = 0; h < 24; h++) {
        expect(v[h * 3], lessThanOrEqualTo(v[h * 3 + 1]), reason: 'hour $h');
        expect(
          v[h * 3 + 1],
          lessThanOrEqualTo(v[h * 3 + 2]),
          reason: 'hour $h',
        );
      }
    });

    test('it forecasts an amount of electricity Italy could draw', () async {
      // The tensors are in the training scaler's units. Converting back is what
      // makes the result checkable as a forecast rather than as an array: a
      // number outside this range would mean the scaler and the artifact came
      // from different runs, which no shape check would catch.
      const mean = 32605.321413703383;
      const std = 7439.5645181078135;
      final c = goldens.cases.first;
      final inputs = [
        for (var i = 0; i < model.manifest.inputs.length; i++)
          await goldens.tensor(c.inputKeys[i], model.manifest.inputs[i]),
      ];
      final v = (await model.run(inputs)).single.asFloat32List();
      for (var h = 0; h < 24; h++) {
        final mw = v[h * 3 + 1] * std + mean;
        expect(mw, inInclusiveRange(20000, 60000), reason: 'hour $h: $mw MW');
      }
    });
  });
}
