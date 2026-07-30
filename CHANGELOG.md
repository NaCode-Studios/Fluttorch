# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries name the roadmap milestone they correspond to, e.g. `(M14)`, so a claim in
[`ROADMAP.md`](ROADMAP.md) can be cross-checked against what actually shipped.

## [Unreleased]

Nothing yet.

## [0.0.1] - 2026-07-30

Tier 0 — Foundations. Decisions, a contract and a set of semantics; no feature a
user can invoke yet, which is what `0.0.1` says that a minor would overstate.

### Added

- Manifest schema v1 (M3): `ManifestCodec` reads and writes it in Dart, `fluttorch_export.manifest`
  writes it in Python, and `testdata/manifest_v1.json` is re-encoded byte for byte by both, so the
  two implementations cannot drift apart while each stays independently valid.
- `Tensor` and `DType` (M3): a tensor carries its spec and its resolved shape, and constructing one
  checks that `bytes.length == elements × width`. `DType` carries that width, which had no home
  before.
- `LoadedModel.runInto` for supplying the destination buffers, so repeated inference does not
  allocate and copy on both sides of every call.
- Tolerance semantics (M4): elementwise bounds combine as `atol + rtol × |reference|`, cosine is
  checked per tensor, and `Tolerance.startingPointFor` gives a documented starting point per
  quantization recipe — returning null for a recipe it does not recognise.
- A sealed `FluttorchException` hierarchy: wrong artifact, unreadable manifest, bad buffer, absent
  backend and absent capability are five different problems and no longer share one type.
- `checkGeneratable` refuses a manifest whose preprocessing this build does not understand, rather
  than generating an API that silently omits a transform the model was trained with.

### Changed

- Preprocessing steps are a sealed hierarchy, so the generator's handling is exhaustive. An
  unrecognised step is preserved as `UnknownPreprocessingStep` rather than dropped.
- Goldens name their tensors by opaque key instead of filesystem path. The core has no business
  knowing where a bundle lives, and on the web there is no filesystem to name.
- Capabilities are reported per loaded model rather than per runtime: after a fallback, the backend
  that was asked for is not the one running.
- Drift metrics moved from `fluttorch` to `fluttorch_test`. Nothing on the inference path needs them.

### Fixed

- `Tolerance.maxRelative` was declared and never read, so a tolerance configured with only a relative
  bound accepted every output — a gate that failed open. A tolerance with no bound configured is now
  a constructor error, and a NaN where a number was expected fails however wide the bound.
- `TensorSpec.elementCount` inferred a nullable accumulator from its return type, making the
  multiplication an unchecked use of a nullable value.
- `fluttorch_gen` no longer declares a `build_runner` builder factory that throws, which turned
  adding the package as a dependency into a build failure instead of a no-op.

### Internal

- M2: the pipeline runs end to end — a two-layer model exported, loaded on macOS through
  `executorch_flutter`, and its output within 2.98e-8 of the PyTorch reference. It also established
  that `executorch_flutter` requires macOS 11.0, which a freshly generated Flutter project does not
  meet.
- M1: audited `executorch_flutter` 0.5.0. None of the four hooks the parity gate needs — activation
  taps, deterministic execution, load-time backend selection, caller-supplied output buffers — exist
  in its public API. Recorded in [`ROADMAP.md`](ROADMAP.md); the consequence is that M19 becomes a
  question of when to fork rather than whether.
- CI runs every Dart package's suite rather than only the core, and the Python manifest suite, which
  needs neither `torch` nor `executorch`.
- 109 tests: 90 Dart across three packages, 19 Python.

[Unreleased]: https://github.com/NaCode-Studios/Fluttorch/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/NaCode-Studios/Fluttorch/releases/tag/v0.0.1
