# fluttorch_executorch_flutter

Puts the compiled ExecuTorch binding inside a Flutter app. The Dart API is in
[`fluttorch_executorch`](../fluttorch_executorch), which stays a plain Dart package: adding a Flutter
SDK dependency to it would make every consumer a Flutter consumer, including the suites that run
under `dart test` today and anything that wants the binding from a server or a command line.

So the split is not ceremony. One package is the binding; this one is the delivery.

## What it carries, and where that comes from

Nothing is committed here. The binaries a build needs are produced by
`packages/fluttorch_executorch/tool/build_native.sh`, which writes them straight into this package:

```bash
packages/fluttorch_executorch/tool/build_native.sh --android
```

That produces `android/src/main/jniLibs/arm64-v8a/libfluttorch_executorch.so`, which Gradle packages
into the APK and `DynamicLibrary.open` finds by name at run time.

A 61 MB shared object is not something to keep in git, and it is also not something to put on
pub.dev, which is why this package is `publish_to: none` for now. Distributing prebuilt binaries is a
separate problem from producing them, and solving the second first would mean shipping a package
whose contents nobody could reproduce.

## What it does not do

It adds no Dart API. `NativeExecuTorchBindings.open()` already resolves the library the way each
platform wants: by name on Android, and out of the process image on iOS, where an app cannot load an
arbitrary dylib from its bundle. This package exists so that those two lookups find something.
