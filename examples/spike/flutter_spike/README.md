# fluttorch_spike

The Flutter host for the M2 spike. It loads the artifact, the manifest and the golden committed under
`assets/`, runs the model through `executorch_flutter` on macOS, and compares the output against the
reference PyTorch produced before lowering.

```bash
flutter test integration_test -d macos
```

The deployment target is macOS 11.0, which `executorch_flutter` requires and a freshly generated
Flutter project does not meet. See [`../README.md`](../README.md) for what the spike measured and
what it established.
