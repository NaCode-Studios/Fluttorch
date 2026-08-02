@TestOn('vm')
library;

import 'dart:io';
import 'dart:typed_data';

import 'package:fluttorch/fluttorch.dart';
import 'package:test/test.dart';
import 'package:typed_api_example/multi_io.fluttorch.dart';

import 'fake_runtime.dart';

/// The generated API keeps two outputs apart.
///
/// Every other model in this repository returns one tensor, so until this
/// fixture the emitter had never written an index other than `_tensors[0]` and
/// the accessor that reads the second output had never been called. The runtime
/// side is measured against real engines in the on-device suites; what is left
/// over, and what only shows up here, is the generated code itself. It does not
/// go anywhere near `checkTensorsAgainst`: it indexes a list and trusts the
/// order, so nothing but this suite would notice it indexing the wrong one.
///
/// The fake is the whole point. Handing the model two tensors that are the same
/// shape and different numbers is the only way to ask which one each accessor
/// returns, and a real backend would answer with numbers that happen to be
/// right rather than with an answer about ordering.
final _artifact = File('../../testdata/multi_io/multi_io.pte');

Float32List _floats(List<double> v) => Float32List.fromList(v);

/// Same spec, different numbers. If either accessor read the wrong index this
/// is what makes it say so, because the shapes cannot.
Tensor _primary() {
  final t = Tensor.zeros(MultiIoPrimary.spec);
  t.asFloat32List().setAll(0, [1, 2, 3]);
  return t;
}

Tensor _auxiliary() {
  final t = Tensor.zeros(MultiIoAuxiliary.spec);
  t.asFloat32List().setAll(0, [7, 8, 9]);
  return t;
}

void main() {
  late Uint8List artifact;

  setUpAll(() async {
    // The real artifact, because `load` verifies it against the weight hash the
    // manifest carries and a fabricated buffer would be refused.
    artifact = await _artifact.readAsBytes();
  });

  group('two outputs through the generated API', () {
    test('each accessor returns its own tensor, not the first one', () async {
      final runtime = FakeRuntime([_primary(), _auxiliary()]);
      final model = await MultiIo.load(runtime, artifact: artifact);

      final out = await model.run(
        left: MultiIoLeft(_floats([1, 1, 1, 1, 1])),
        right: MultiIoRight(_floats([2, 2, 2, 2, 2])),
      );

      expect(out.primary.tensor.asFloat32List(), [1, 2, 3]);
      expect(
        out.auxiliary.tensor.asFloat32List(),
        [7, 8, 9],
        reason:
            'an emitter that indexed _tensors[0] twice passes every other '
            'assertion in this package',
      );
    });

    test('the positional list agrees with the named accessors', () async {
      final runtime = FakeRuntime([_primary(), _auxiliary()]);
      final model = await MultiIo.load(runtime, artifact: artifact);

      final out = await model.run(
        left: MultiIoLeft(_floats([1, 1, 1, 1, 1])),
        right: MultiIoRight(_floats([2, 2, 2, 2, 2])),
      );

      expect(out.tensors, hasLength(2));
      expect(
        out.tensors[0].asFloat32List(),
        out.primary.tensor.asFloat32List(),
      );
      expect(
        out.tensors[1].asFloat32List(),
        out.auxiliary.tensor.asFloat32List(),
      );
    });

    test('the two output types are declared apart', () {
      expect(MultiIoPrimary.spec.name, 'primary');
      expect(MultiIoAuxiliary.spec.name, 'auxiliary');
      // Identical shapes, which is what makes the ordering assertions above
      // load-bearing rather than incidentally true.
      expect(MultiIoAuxiliary.spec.shape, MultiIoPrimary.spec.shape);
    });
  });
}
