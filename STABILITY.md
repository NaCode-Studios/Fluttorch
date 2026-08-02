# Stability policy

Fluttorch follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). This document says what
that means in practice: what is covered, what is not, and how long a deprecated API survives.

Four things here carry a compatibility promise and only one of them is a Dart API. The manifest is a
document two implementations parse in two languages, the C header is an ABI three bindings implement
and consumers link against as a prebuilt library, the generated Dart is code that gets committed, and
the tolerances decide whether a build is green. Each has its own rule below, because SemVer on the
package version answers none of them.

## What is public

The barrel export of each published package:

| Package | Public surface |
| --- | --- |
| `fluttorch` | `package:fluttorch/fluttorch.dart` |
| `fluttorch_gen` | `package:fluttorch_gen/fluttorch_gen.dart`, and the builder it registers |
| `fluttorch_test` | `package:fluttorch_test/fluttorch_test.dart` and `package:fluttorch_test/io.dart` |
| `fluttorch_executorch` | `package:fluttorch_executorch/fluttorch_executorch.dart` |
| `fluttorch_litert` | `package:fluttorch_litert/fluttorch_litert.dart` |
| `fluttorch_onnx` | `package:fluttorch_onnx/fluttorch_onnx.dart` |

Anything reachable only through `package:*/src/…` is internal and may change in any release without
notice. That includes `fluttorch_executorch/src/ffi.dart`, which the repository's own tools import:
depending on it from outside is a mistake this document does not protect against.

## What a version bump means

| Change | Bump |
| --- | --- |
| Behaviour a caller can observe changes incompatibly | major |
| New capability, existing callers unaffected | minor |
| Fix that brings behaviour back to what was documented | patch |

### The manifest

`schema_version` inside the manifest is versioned separately from the package, and it is the field
that decides whether an older reader may proceed. It rises when a reader that does not understand the
change would misread the document, and not for a field such a reader can ignore while continuing to do
what it did before.

The distinction is not stylistic. `activations` was additive: a reader without it compares final
outputs and says it could not look deeper, which is what it did before the field existed. `parts` was
not: a reader without it loads an artifact whose weights live in a file beside the graph, finds a
well-formed graph with no numbers in it, and answers. The decoder ignores keys it does not recognise,
deliberately, so the version is the only thing standing between that reader and a wrong answer.

Raising `schema_version` is a minor release of the packages that write it and a major one for nobody,
because a manifest declaring a version above what a build supports is refused rather than misread. An
export that carries no new field keeps declaring the version it declared before, so nothing already
written stops being loadable.

### The C ABI

`packages/fluttorch_executorch/src/fluttorch_executorch.h` is one header three bindings implement, and
a consumer may link a prebuilt library built from an earlier copy of it. Removing a function or
changing the signature or the field order of a struct is a major change. Adding a function is minor,
and the bindings look it up by name at construction, so a library that lacks it fails loudly at open
rather than at the call.

Struct field order is part of the ABI in a way that does not fail to compile. Both sides read by
offset, so reordering fields reads the wrong bytes and returns plausible numbers. Treat the header and
`ffi.dart` as one edit.

### Generated code

`fluttorch_gen` emits Dart that consumers commit and CI regenerates and diffs. A change to what it
emits therefore shows up as a diff in someone else's repository even when nothing they wrote changed.

Emitting a different API for the same manifest is a major change. Emitting the same API formatted
differently is a patch, and the generated file is compared after `dart format` for that reason.

### Numbers

The output of a computation is not covered by SemVer, and this is the answer most likely to matter.
The arithmetic belongs to the delegate that runs it, and an upstream release can reorder a reduction
without telling anyone. No version of this library can promise a bit pattern it does not produce.

What is covered is the bound. `Tolerance.boundFor` returns the default a gate uses when the caller
names none, and narrowing one turns a green build red without anything in the caller's code changing.
Narrowing a bound is a major change. Widening one is minor, and each entry records the models it was
measured against so a widening can be argued with rather than accepted. `tool/measure_tolerances.dart`
reproduces those measurements.

A caller who needs a bound that does not move should pass their own to the gate rather than inherit
this table, which is what the `tolerance` argument is for.

## Deprecation

A deprecated API survives at least two minor releases before removal, carries a `@Deprecated`
annotation naming its replacement, and is listed under `Deprecated` in [`CHANGELOG.md`](CHANGELOG.md)
in the release that deprecates it.

Two minor releases is ExecuTorch's own policy rather than a number chosen here. Fluttorch's ExecuTorch
binding cannot outlive a symbol the engine below it has removed, so promising longer than the
dependency delivers would be promising something this project does not control.

## Supported versions

The latest minor line receives fixes. There is no backport branch, and one maintainer promising more
than that would be describing an intention rather than a policy.

## Platform and toolchain support

Dart SDK `^3.9.0`. The Flutter packages need whatever Flutter version ships that SDK.

Raising a minimum SDK is a minor change, because a consumer on an older SDK keeps resolving the
previous version rather than getting a build that fails. Dropping a platform the bindings previously
ran on is major.

The native engines are not pinned by this policy. ExecuTorch, LiteRT and ONNX Runtime each move on
their own schedule, and the version a binding was built and tested against is recorded in that
package's README rather than promised here.
