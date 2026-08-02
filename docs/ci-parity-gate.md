# Running the parity gate in CI

The gate replays the goldens the export captured and fails the build when the on-device numbers
have moved further than you allowed. This is the workflow that runs it on every pull request, and
the three things that decide whether a green run means anything.

## The workflow

```yaml
name: Parity gate

on:
  pull_request:
  push:
    branches: [main]

jobs:
  parity:
    # The gate runs wherever your backend runs. ExecuTorch through
    # executorch_flutter needs macOS 11.0 or later, which is why this is not
    # ubuntu-latest.
    runs-on: macos-14
    steps:
      - uses: actions/checkout@v7

      # The goldens are the reference every assertion is measured against, so a
      # job that cannot see them has nothing to check. Fail here rather than
      # further down with a suite that reports success over zero cases.
      - name: The goldens have to be present
        run: test -d assets/goldens

      - uses: subosito/flutter-action@v2
        with:
          channel: stable
      - run: flutter pub get
      - run: flutter test integration_test -d macos
```

A package with no Flutter dependency swaps the last three steps for `dart-lang/setup-dart`,
`dart pub get` and `dart test`, and can then run on any runner its backend supports.

## The test

On the filesystem, `DirectoryGoldenBundle` reads the layout the exporter writes, so the manifest
and its goldens are named once:

```dart
import 'dart:io';

import 'package:fluttorch_test/fluttorch_test.dart';
import 'package:fluttorch_test/io.dart';
import 'package:test/test.dart';

void main() {
  test('the exported model still agrees with its reference', () async {
    final goldens = await DirectoryGoldenBundle.open(
      'assets/solar_forecast.fluttorch.json',
    );
    final model = await runtime.load(
      artifact: await File('assets/solar_forecast.pte').readAsBytes(),
      manifest: goldens.manifest,
    );
    addTearDown(model.dispose);

    await expectParity(model, goldens: goldens);
  });
}
```

Inside a Flutter app there is no filesystem to read, and resolving an asset key is the app's job
rather than this package's. Read the assets and hand over the bytes:

```dart
final manifest = SolarForecast.manifest;
final keys = [
  for (final c in manifest.goldens) ...c.inputKeys, ...c.outputKeys,
];
final bytes = <String, Uint8List>{};
for (final key in keys) {
  final data = await rootBundle.load('assets/goldens/$key');
  // The offset and length matter: asUint8List() with no arguments hands back
  // the whole backing buffer, which is longer than the asset and fails the
  // length check the bundle performs against the spec.
  bytes[key] = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
}
final goldens = MemoryGoldenBundle(manifest, bytes);
```

## What makes a green run mean something

The goldens have to travel with the checkout. They are binary and they are the only thing standing
between a passing suite and an unevaluated model, so keep them in the repository, or fetch them in
a step that fails loudly when the fetch does. `expectParity` refuses a bundle with no cases for the
same reason: a gate that passes because it had nothing to check is indistinguishable from a healthy
model until the day it matters.

The tolerance is reviewed like code. `Tolerance.boundFor` gives a bound per quantization recipe and
precision, measured against the models in this repository and citing which. That is a default rather
than an answer about your model: the number that is right for one is a property of its activation
ranges, and a model whose outputs are large drifts further for the same rounding. Measure yours and
pass it to the gate. Widening a tolerance to make a red build green is a decision, and it belongs in
a diff someone reads.

The report names the backend it measured. A model that is correct on XNNPACK and wrong on Core ML
is a real outcome, so a run that does not say which backend produced it cannot be acted on. When
the runtime falls back, the backend named is the one that actually ran.

## What does not run here yet

This repository runs the gate's own suite on every pull request, including a replay of the export
committed under `testdata/two_layer/`. That covers the half of the gate that lives in Dart: the
pairing of each case with its own reference, the metrics, and the failure output.

It does not yet run against a device. `fluttorch_executorch` has no implementation behind it, so
the recipe above is the shape the workflow takes rather than one this repository proves on every
commit. What proves it is possible is [`examples/spike/`](../examples/spike/), which loads a real
export through `executorch_flutter` on macOS and compares it against the PyTorch reference. A
device matrix arrives with M25, tracked on the
[board](https://github.com/orgs/NaCode-Studios/projects/6).
