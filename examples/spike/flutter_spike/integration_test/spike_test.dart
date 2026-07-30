/// M2 · the end-to-end spike, load half.
///
/// Loads the artifact `export_spike.py` produced, runs it, and compares one
/// output against the reference captured from the source model before lowering.
///
///     flutter test integration_test -d macos
///
/// Deliberately unabstracted. The point is to find where the pipeline actually
/// breaks, and the seam this proves out is the one `fluttorch_executorch` will
/// implement — not a design invented ahead of the evidence.
library;

import 'dart:typed_data';

import 'package:executorch_flutter/executorch_flutter.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:fluttorch/fluttorch.dart';
import 'package:fluttorch_test/fluttorch_test.dart';
import 'package:integration_test/integration_test.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  late ModelManifest manifest;
  late Uint8List artifact;

  setUpAll(() async {
    manifest = ManifestCodec.decode(
      await rootBundle.loadString('assets/two_layer.fluttorch.json'),
    );
    artifact = (await rootBundle.load(
      'assets/two_layer.pte',
    )).buffer.asUint8List();
  });

  test('the artifact is the one the manifest was written for', () {
    // The check the runtime will own once it exists. A manifest paired with the
    // wrong weights produces a green suite over a model nobody evaluated, and
    // it is the cheapest thing in the whole pipeline to verify.
    final digest = 'sha256:${sha256Hex(artifact)}';
    expect(
      digest,
      manifest.weightHash,
      reason: 'the .pte and its manifest come from different exports',
    );
  });

  test('the exported model agrees with the model it was exported from', () async {
    final inputSpec = manifest.inputNamed('x');
    final outputSpec = manifest.outputNamed('y');
    final golden = manifest.goldens.single;

    final input = Tensor.view(
      spec: inputSpec,
      bytes: (await rootBundle.load(
        'assets/${golden.inputKeys.single}',
      )).buffer.asUint8List(),
    );
    final reference = Tensor.view(
      spec: outputSpec,
      bytes: (await rootBundle.load(
        'assets/${golden.outputKeys.single}',
      )).buffer.asUint8List(),
    );

    final model = await ExecuTorchModel.loadFromBytes(artifact);
    addTearDown(model.dispose);

    final outputs = await model.forward([
      TensorData(
        shape: input.shape,
        dataType: TensorType.float32,
        data: input.bytes,
      ),
    ]);

    expect(outputs, hasLength(manifest.outputs.length));
    final actual = Tensor.view(
      spec: outputSpec,
      bytes: Uint8List.fromList(outputs.single.data),
    );

    // Full precision through XNNPACK: the graph was re-ordered, not re-quantized,
    // so anything beyond accumulated float32 rounding is a real change.
    final drift = measureDrift(
      actual: actual,
      reference: reference,
      tolerance: Tolerance.startingPointFor(manifest.quantization)!,
    );

    // ignore: avoid_print
    print(
      DriftReport(
        goldenId: golden.id,
        backend: 'xnnpack',
        quantization: manifest.quantization,
        tensors: [drift],
      ).describe(),
    );

    expect(
      drift.passes,
      isTrue,
      reason: 'on-device output drifted from the reference: $drift',
    );
  });
}

/// Minimal SHA-256, so the spike does not pull a dependency to prove a point.
///
/// The real implementation belongs in the runtime at M6; this exists only to
/// show the check is cheap and worth having from the first load.
String sha256Hex(Uint8List bytes) {
  const k = <int>[
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
    0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
    0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
    0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
    0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
    0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
  ];
  var h = <int>[
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
  ];
  final len = bytes.length;
  final padded = BytesBuilder()
    ..add(bytes)
    ..addByte(0x80);
  while ((padded.length + 8) % 64 != 0) {
    padded.addByte(0);
  }
  final tail = ByteData(8)..setUint64(0, len * 8, Endian.big);
  padded.add(tail.buffer.asUint8List());
  final msg = ByteData.sublistView(padded.toBytes());

  int rotr(int x, int n) => ((x >>> n) | (x << (32 - n))) & 0xFFFFFFFF;

  for (var block = 0; block < msg.lengthInBytes; block += 64) {
    final w = List<int>.filled(64, 0);
    for (var i = 0; i < 16; i++) {
      w[i] = msg.getUint32(block + i * 4, Endian.big);
    }
    for (var i = 16; i < 64; i++) {
      final s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
      final s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) & 0xFFFFFFFF;
    }
    var (a, b, c, d) = (h[0], h[1], h[2], h[3]);
    var (e, f, g, hh) = (h[4], h[5], h[6], h[7]);
    for (var i = 0; i < 64; i++) {
      final s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      final ch = (e & f) ^ (~e & g);
      final t1 = (hh + s1 + ch + k[i] + w[i]) & 0xFFFFFFFF;
      final s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      final maj = (a & b) ^ (a & c) ^ (b & c);
      final t2 = (s0 + maj) & 0xFFFFFFFF;
      hh = g;
      g = f;
      f = e;
      e = (d + t1) & 0xFFFFFFFF;
      d = c;
      c = b;
      b = a;
      a = (t1 + t2) & 0xFFFFFFFF;
    }
    h = [
      (h[0] + a) & 0xFFFFFFFF, (h[1] + b) & 0xFFFFFFFF,
      (h[2] + c) & 0xFFFFFFFF, (h[3] + d) & 0xFFFFFFFF,
      (h[4] + e) & 0xFFFFFFFF, (h[5] + f) & 0xFFFFFFFF,
      (h[6] + g) & 0xFFFFFFFF, (h[7] + hh) & 0xFFFFFFFF,
    ];
  }
  return h.map((v) => v.toRadixString(16).padLeft(8, '0')).join();
}
