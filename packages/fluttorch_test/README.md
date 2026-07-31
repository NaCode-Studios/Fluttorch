# fluttorch_test

The parity gate. It replays the reference outputs captured when the model was exported, measures how
far the on-device numbers moved, and fails the build when they moved further than you allowed.

Exporting a model changes its numbers, and quantizing it changes them more. Nothing in the toolchain
tells you when they change too much: the build succeeds, the app runs, the output looks plausible,
and the model is quietly worse than the one you evaluated. Comparing predictions before and after
serialization is the documented advice everywhere and a manual chore everywhere.

```dart
import 'package:fluttorch_test/fluttorch_test.dart';
import 'package:fluttorch_test/io.dart';

test('the exported model still agrees with its reference', () async {
  final goldens = await DirectoryGoldenBundle.open('assets/solar.fluttorch.json');
  final model = await runtime.load(
    artifact: await File('assets/solar.pte').readAsBytes(),
    manifest: goldens.manifest,
  );

  await expectParity(model, goldens: goldens);
});
```

```
FAIL  parity/case-3
      backend: xnnpack  quantization: int8-static
      output "load_mw"  max |Δ| 1.72  >  Tolerance(atol 0.1, rtol 0.1, cos ≥ 0.998)  worst at [0]: 14.0210 vs 12.3000
        2 of 4 elements (50.0%) exceed the elementwise bound
      no layer attribution: backend "xnnpack" offers no activation taps
```

The tolerance comes from the recipe the manifest records unless you pass one you measured yourself.
Three bounds are combined because none of them is sufficient alone: an absolute bound is the only
meaningful one near zero, a relative bound is the only meaningful one on large magnitudes, and cosine
catches the tensor whose values all pass individually while pointing somewhere else.

A bundle with no cases fails. A gate that passes because it had nothing to check is
indistinguishable from a healthy model until the day it matters.

`MemoryGoldenBundle` is what a Flutter app uses, since resolving an asset key belongs to the app.
`DirectoryGoldenBundle` reads the layout the exporter writes and lives in a separate library, so that
importing the gate does not pull `dart:io` into a suite that runs on the web.

The workflow that runs this on every pull request is
[in the repository](https://github.com/NaCode-Studios/Fluttorch/blob/main/docs/ci-parity-gate.md),
along with the three conditions that decide whether a green run means anything.

## License

Apache-2.0.
