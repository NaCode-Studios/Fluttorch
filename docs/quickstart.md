# Quickstart

Export a model, generate its API, and run the gate.

## Export

```python
import torch
from fluttorch_export.export import export_model


class TwoLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(8, 3)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


model = TwoLayer().eval()
example = torch.randn(1, 4)

export_model(
    model=model,
    example_inputs=(example,),
    name="two_layer",
    out_dir="build",
    input_names=["features"],
    output_names=["score"],
    # References captured from the source model, before lowering. Without these
    # the bundle is a smoke test rather than coverage, and the exporter says so
    # rather than inventing a distribution it has no way to know.
    golden_inputs=[(torch.randn(1, 4),) for _ in range(4)],
)
```

Three files come out together: `two_layer.pte`, `two_layer.fluttorch.json`, and a `goldens/`
directory. They are one bundle. Moving one without the others is the failure the weight hash exists
to catch.

## Generate

Put the manifest under `lib/` and let `build_runner` do the rest.

```yaml
dev_dependencies:
  build_runner: ^2.16.0
  fluttorch_gen: ^0.6.0
```

```bash
dart run build_runner build --delete-conflicting-outputs
```

You get `two_layer.fluttorch.dart`: a type per tensor, the spec each was declared with, the manifest
carried verbatim so loading needs no separate plumbing, and any preprocessing the export recorded,
turned into arithmetic.

```dart
final features = TwoLayerFeatures(Float32List.fromList([1, 2, 3, 4]))
  ..preprocess();
```

The wrapper costs nothing at run time and makes passing the wrong tensor a compile error.

## Run the gate

```dart
final goldens = await DirectoryGoldenBundle.open('build/two_layer.fluttorch.json');
final model = await ExecuTorchRuntime(bindings).load(
  artifact: await File('build/two_layer.pte').readAsBytes(),
  manifest: goldens.manifest,
);

await expectParity(model, goldens: goldens);
```

`expectParity` fails the test. `measureParity` hands back the report instead, which is what you want
when you would rather read the number than react to it:

```dart
final reports = await measureParity(model, goldens: goldens);
print(reports.first.describe());
```

## Running this repository's own suites

The Dart suites need nothing but the SDK. The Python ones need `flatc` on `PATH`, which an ExecuTorch
checkout builds and nothing puts there for you:

```bash
export PATH="$HOME/.cache/fluttorch/executorch/cmake-out-android/third-party/flatc_ep/bin:$PATH"
```

Without it the export tests fail with `FileNotFoundError: 'flatc'` from inside a serialization step,
which names nothing you would recognise as the cause.

The on-device suite additionally needs the native library, which
`packages/fluttorch_executorch/tool/build_native.sh` builds against an ExecuTorch checkout.
`tool/prepare_executorch.sh` performs the upstream workarounds that checkout needs first. Where the
library is absent the suite skips by name rather than silently, so a green run still says what it did
not cover.
