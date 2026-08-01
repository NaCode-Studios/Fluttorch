/// The parity gate, on a phone, through Fluttorch's own binding.
///
///     flutter test integration_test -d <device>
///
/// Every other number this project reports was measured on a laptop. This is the
/// one that is measured where the model is meant to run, which is the difference
/// between a library that works and a library that is believed to.
library;

import 'dart:typed_data';

import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:fluttorch/fluttorch.dart';
import 'package:fluttorch_executorch_flutter/fluttorch_executorch_flutter.dart';
import 'package:fluttorch_test/fluttorch_test.dart';
import 'package:integration_test/integration_test.dart';

Future<Uint8List> _asset(String name) async =>
    (await rootBundle.load('assets/$name')).buffer.asUint8List();

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late ModelManifest manifest;
  late Uint8List artifact;
  late MemoryGoldenBundle goldens;

  setUpAll(() async {
    manifest = ManifestCodec.decode(
      await rootBundle.loadString('assets/two_layer.fluttorch.json'),
    );
    artifact = await _asset('two_layer.pte');

    // The goldens travel as flat assets rather than as a directory, because a
    // phone has no directory to point at. The keys are the manifest's own, so
    // nothing here restates the contract.
    final bytes = <String, Uint8List>{};
    for (var i = 0; i < manifest.goldens.length; i++) {
      bytes[manifest.goldens[i].inputKeys.single] = await _asset('g${i}_in.bin');
      bytes[manifest.goldens[i].outputKeys.single] = await _asset(
        'g${i}_out.bin',
      );
    }
    goldens = MemoryGoldenBundle(manifest, bytes);
  });

  testWidgets('the artifact is the one the manifest was written for', (
    _,
  ) async {
    // Before anything native sees it. A manifest paired with the wrong weights
    // satisfies every shape and returns every number wrong.
    verifyArtifact(artifact: artifact, manifest: manifest);
  });

  testWidgets('the binding is in the app, and reports its backends', (
    _,
  ) async {
    // Resolved the way the platform wants: by name from jniLibs on Android, out
    // of the process image on iOS. If this throws, the packaging is what failed
    // rather than the model.
    final runtime = ExecuTorchRuntime(NativeExecuTorchBindings.open());
    final backends = await runtime.capabilities();
    expect(backends.map((c) => c.backend), contains('xnnpack'));
  });

  testWidgets('the goldens hold on this device', (_) async {
    final model = await ExecuTorchRuntime(NativeExecuTorchBindings.open()).load(
      artifact: artifact,
      manifest: manifest,
      backend: 'xnnpack',
    );
    addTearDown(model.dispose);

    // The claim, on the hardware. Same export, same references, same bound the
    // recipe implies, measured where it matters.
    await expectParity(model, goldens: goldens);
  });

  testWidgets('it also runs off the thread that draws', (_) async {
    final runtime = await IsolateExecuTorchRuntime.spawn();
    addTearDown(runtime.shutdown);

    final model = await runtime.load(
      artifact: artifact,
      manifest: manifest,
      backend: 'xnnpack',
    );
    addTearDown(model.dispose);

    await expectParity(model, goldens: goldens);
  });
}
