@TestOn('vm')
library;

import 'dart:io';

import 'package:fluttorch_test/fluttorch_test.dart';
import 'package:fluttorch_test/io.dart';
import 'package:test/test.dart';

import 'support/fake_model.dart';

/// The goldens a real export wrote, committed whole.
///
/// The bundles in `bundle_test.dart` are built by hand to exercise the awkward
/// cases. This one is the opposite check: the layout `fluttorch-export`
/// actually produces, read by the code that has to consume it.
const _manifestPath = '../../testdata/two_layer/two_layer.fluttorch.json';

/// The one export in this repository that returns two tensors.
///
/// Everything else returns one, so the gate's loop over `manifest.outputs` has
/// always had a single trip and the report has never had to say which of two
/// outputs moved. Replaying it here rather than only on a device is deliberate:
/// this half of the gate is Dart, it needs no engine, and a check that runs
/// only where the native libraries are built is a check that does not run.
const _multiIoPath = '../../testdata/multi_io/multi_io.fluttorch.json';

void main() {
  late DirectoryGoldenBundle goldens;

  setUpAll(() async {
    expect(
      File(_manifestPath).existsSync(),
      isTrue,
      reason:
          'run from packages/fluttorch_test so the shared testdata resolves',
    );
    goldens = await DirectoryGoldenBundle.open(_manifestPath);
  });

  group('M13 · goldens on disk', () {
    test(
      'open finds the directory the exporter writes beside the manifest',
      () {
        expect(goldens.root, endsWith('testdata/two_layer/goldens'));
        expect(goldens.cases.map((c) => c.id), [
          'case-0',
          'case-1',
          'case-2',
          'case-3',
        ]);
      },
    );

    test('a key resolves to the tensor the manifest declares', () async {
      final input = await goldens.tensor(
        goldens.cases.first.inputKeys.single,
        goldens.manifest.inputs.single,
      );

      expect(input.spec.name, 'features');
      expect(input.shape, [1, 4]);
      expect(input.asFloat32List().every((v) => v.isFinite), isTrue);
    });

    test('a golden that is not on disk says where it was looked for', () async {
      final elsewhere = DirectoryGoldenBundle(
        goldens.manifest,
        root: '../../testdata/two_layer/not-copied',
      );

      await expectLater(
        elsewhere.tensor(
          goldens.cases.first.inputKeys.single,
          goldens.manifest.inputs.single,
        ),
        throwsA(
          isA<StateError>().having(
            (e) => e.message,
            'message',
            allOf(contains('not-copied'), contains('travel with it')),
          ),
        ),
      );
    });
  });

  group('M15 · the gate over a real export', () {
    test('replaying the recorded outputs passes at full precision', () async {
      final model = await FakeModel.replaying(goldens.manifest, goldens);

      await expectParity(model, goldens: goldens);
      expect(model.runs, 4, reason: 'every captured case is replayed');
    });

    test('one drifted case fails, and the report says which one', () async {
      final model = await FakeModel.replaying(
        goldens.manifest,
        goldens,
        perturb: (id, outputs) {
          if (id == 'case-2') outputs.single.asFloat32List()[1] += 0.5;
        },
      );

      await expectLater(
        expectParity(model, goldens: goldens),
        throwsA(
          isA<TestFailure>().having(
            (e) => e.message,
            'message',
            allOf(
              contains('1 of 4 golden cases'),
              contains('FAIL  parity/case-2'),
              contains('output "score"'),
              contains('worst at [1]'),
              contains('1 of 3 elements'),
            ),
          ),
        ),
      );
    });
  });

  group('a model with two outputs', () {
    late DirectoryGoldenBundle multiIo;

    setUpAll(() async {
      multiIo = await DirectoryGoldenBundle.open(_multiIoPath);
    });

    test('both outputs are measured, not just the first', () async {
      final model = await FakeModel.replaying(multiIo.manifest, multiIo);

      await expectParity(model, goldens: multiIo);

      final reports = await measureParity(model, goldens: multiIo);
      expect(reports, hasLength(4));
      for (final report in reports) {
        expect(
          report.tensors.map((t) => t.tensorName),
          ['primary', 'auxiliary'],
          reason:
              'a loop that stopped after the first output would pass '
              'this model while measuring half of it',
        );
      }
    });

    test('drift in the second output is reported as the second', () async {
      // The whole point of the fixture: both outputs are float32[1, 3], so
      // nothing about the shapes distinguishes them and a report that named
      // "primary" here would be indistinguishable from a correct one to
      // anything except this assertion.
      final model = await FakeModel.replaying(
        multiIo.manifest,
        multiIo,
        perturb: (id, outputs) {
          if (id == 'case-1') outputs[1].asFloat32List()[2] += 0.5;
        },
      );

      await expectLater(
        expectParity(model, goldens: multiIo),
        throwsA(
          isA<TestFailure>().having(
            (e) => e.message,
            'message',
            allOf(
              contains('1 of 4 golden cases'),
              contains('FAIL  parity/case-1'),
              contains('output "auxiliary"'),
              contains('worst at [2]'),
              isNot(contains('output "primary"')),
            ),
          ),
        ),
      );
    });
  });
}
