# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries name the roadmap milestone they correspond to, e.g. `(M14)`, so a claim on the
[board](https://github.com/orgs/NaCode-Studios/projects/6) can be cross-checked against what actually shipped.

## [Unreleased]

### Internal

- `release.yaml` publishes `fluttorch`, `fluttorch_gen` and `fluttorch_test` to pub.dev
  over OIDC, so no token is stored anywhere and the package pages carry the verified
  publisher line. The two dependants wait for the core version to be servable before
  they upload, because they declare it as a dependency and pub.dev refuses a package
  whose dependencies do not resolve. Creating the GitHub Release deliberately does not
  wait on any of that: it is derived from the tag and the changelog and is true whatever
  the registry did, and chaining it would mean a re-run after a partial publish fails on
  the package that already went up.
- Published archives are recorded as linked artifacts on the organisation, keyed on the
  digest pub.dev serves them under. Metadata rather than distribution: nobody resolves a
  dependency from that panel. The step cannot fail the release, because a metadata write
  that can undo a successful publish is worth less than the metadata.
- Every published archive is attested with SLSA build provenance, so it can be shown to
  have come from this repository's workflow at a named commit rather than from somebody
  who uploaded a lookalike. Attesting before publishing is not available for a Dart
  package, because the archive is built inside `dart pub publish` and never lands on the
  runner as a file, so the only digest that exists is the one the registry serves
  afterwards. The attested set is derived from which packages are publishable rather than
  listed, so a package covered the day its `publish_to` line goes. `0.3.0` has no
  attestation and cannot be given one: it was published by hand, and there is no workflow
  run to sign against.

## [0.3.0] - 2026-07-31

Tier 3, the parity gate. The goldens captured at export now replay against a loaded
model and fail the build when the numbers have moved too far.

### Added

- `measureParity` replays every golden case against a loaded model and reports what it
  measured (M13). Each case is handed its own inputs and compared against its own
  reference, and the outputs a backend returns are checked against the manifest before
  they are measured, so a backend that reorders them is reported as the defect it is
  rather than as drift.
- `GoldenBundle` has two implementations (M13). `MemoryGoldenBundle` is what a unit test
  and a Flutter app use, since resolving an asset key belongs to the app. The
  filesystem-backed `DirectoryGoldenBundle` lives in `package:fluttorch_test/io.dart`,
  so importing the gate does not drag `dart:io` into a suite that runs on the web, and
  `open` takes the manifest path and defaults to the `goldens/` directory the exporter
  writes beside it. A tensor stored against a spec with one dynamic dimension has that
  extent inferred from how many elements arrived; two dynamic dimensions are refused,
  because a golden replayed at a shape the model never saw measures nothing.
- `expectParity` fails a build with the report as its message (M15), naming the tensor,
  the drift, the bound it broke and the backend that ran. A bundle with no cases fails
  too: a gate that passes because it had nothing to check is indistinguishable from a
  healthy model until the day it matters.
- A copy-pasteable CI workflow in [`docs/ci-parity-gate.md`](docs/ci-parity-gate.md)
  (M16), with the three conditions that decide whether a green run means anything and a
  plain statement of what this repository does not yet run against a device.

### Changed

- A drift that only the cosine bound can see now says so (M14). The line used to print
  the elementwise bound, which in that case names the one thing that did not fail. A
  failing tensor also reports how many of its elements went outside the bound, beside
  the worst one.
- `DriftReport` carries why per-layer attribution is absent, rather than always blaming
  the backend for offering no activation taps. A backend can have taps while the goldens
  hold no reference activations to compare them against, and a report that blamed the
  hardware for that would send someone looking in the wrong place. Attribution itself is
  M18.

### Internal

- `release.yaml` creates the GitHub Release for a tag, with the body extracted from this
  file rather than written by hand. Tags do not populate the Releases panel, so `v0.0.1`
  and `v0.1.0` left it empty, which reads as a library that has never shipped. It calls
  `ci.yaml` rather than restating its job list, so the gate a tag passes is the one every
  pull request passes: `v0.0.1` was originally cut on a red commit because that check was
  a line in a checklist instead of a job.
- The gate is exercised against the export committed under `testdata/two_layer/`, not
  only against bundles written to test it. What stands in for a backend is a model that
  replays the recorded outputs, which cannot drift and therefore proves nothing about a
  device; what it does prove is the half of the gate that lives in Dart.
- `fluttorch`, `fluttorch_gen` and `fluttorch_test` are prepared for pub.dev: each carries
  its own licence, package page and changelog pointer, and every package in the repository
  now shares the repository's version instead of sitting at `0.0.1-dev`.
  `fluttorch_executorch` is deliberately held back, because its whole public surface throws
  `UnimplementedError` and a pub.dev page cannot be withdrawn once it exists. It is
  published in the release where it loads a model, which is M20.
- 177 Dart tests across four packages.

## [0.2.0] - 2026-07-31

Tier 2, typed bindings. A manifest now becomes a Dart API in which handing the
model the wrong tensor is a compile error.

### Added

- `fluttorch_gen` emits a typed API from a manifest (M9). Each tensor becomes its own
  extension type, so `run` cannot be called with the wrong one and the wrapper costs
  nothing at run time. The generated class embeds its manifest, so `load` verifies the
  artifact against the digest it was generated for with no separate plumbing, and
  `wrap` covers a model loaded through some other path.
- Preprocessing generated from the manifest (M10), with the constants training used
  compiled in. Rescale, normalize and cast are emitted; nothing is written by hand on
  the Dart side, which is what makes train/serve skew structurally impossible rather
  than merely discouraged.
- A checked constructor for every fixed shape, and `withShape` as the single escape
  hatch where a dimension is dynamic and the compiler cannot help (M11).
- `examples/typed_api`, showing what the generator produces and how it is used.

### Changed

- The generator formats its own output with the repository's formatter, so generated
  code needs no formatting pass and a pinned golden cannot drift from the emitter over
  line breaks alone.

### Internal

- Resize and center_crop are refused rather than generated (M10). Performing either
  means knowing which axes are spatial, and the manifest records no tensor layout, so
  NCHW and NHWC would each produce a plausible and different answer. This is a gap in
  the schema, not in the generator, and generating a guess would be worse than
  refusing.
- The generator's output is pinned and compiled (M12): the golden lives in a package CI
  analyses, so a generator that emits code which does not compile fails the build rather
  than the diff. CI also runs the builder over `examples/typed_api` and fails if the
  committed output comes back different, which catches a builder that works only in a
  test harness.
- The plan moved onto the board and `ROADMAP.md` and `ROADMAP-CONVENTIONS.md` were
  retired, so no file restates what the board owns.
- Code ownership points at the `libraries` team rather than a single account, so review
  redistributes on its own as the team changes.
- 150 Dart tests across four packages.

## [0.1.0] - 2026-07-30

Tier 1, the export contract. One command now produces an artifact, its manifest
and its goldens together, and the Dart side refuses an artifact that does not
match the manifest it was handed.

### Added

- `fluttorch-export` (M5): resolves `module:factory` references, lowers through
  XNNPACK at full precision, and writes the artifact, the manifest and the goldens in
  one command. Together, because the weight hash can only catch a mismatched pair if
  nothing produces one without the other.
- `verifyArtifact` and `digestOf` (M6): the runtime refuses an artifact whose digest
  disagrees with `ModelManifest.weightHash`. An unknown hash algorithm is reported as a
  format problem rather than a mismatch, because the fix is different.
- Golden capture from the source model (M7), before lowering, so the references describe
  the model rather than the export. Without supplied cases the exporter captures the
  example input alone and says that this is a smoke test, not coverage.
- `--input-names` and `--output-names`, because a generated accessor called `input_0`
  is one nobody wants to read.
- `--dynamic-batch`, marking the leading dimension of every tensor dynamic.

### Changed

- The Python writer emits canonical JSON instead of `json.dumps` (M8). Both languages
  write the shortest round-trip decimal and then spell it differently, switching to
  exponential notation at different magnitudes and padding the exponent differently, and
  Python escaped non-ASCII where Dart does not. Python is the only writer in the
  pipeline, so it matches the reader.

### Fixed

- The byte-for-byte round trip claimed in `0.0.1` was accidentally true: the fixture
  happened to hold only values the two languages spell identically. A normalize mean of
  `1e-5` was enough to break it. It is now true for denormals, negative zero, values on
  either side of both notation thresholds, and non-ASCII in names and labels.

### Internal

- `testdata/manifest_edge_v1.json` and `testdata/two_layer/`: a hand-written edge-case
  document, and a real export read back by the code that has to consume it. A contract
  verified only against documents written to test it is verified against itself.
- The export suite is skipped where `torch` is absent, which includes CI. That coverage
  is documented as not run there rather than implied by a green badge.
- 194 tests locally: 117 Dart, 77 Python. CI runs 117 and 56.

## [0.0.1] - 2026-07-30

Tier 0, foundations. Decisions, a contract and a set of semantics; no feature a
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
  quantization recipe, returning null for a recipe it does not recognise.
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
  bound accepted every output. That is a gate that fails open. A tolerance with no bound configured is
  now a constructor error, and a NaN where a number was expected fails however wide the bound.
- `TensorSpec.elementCount` inferred a nullable accumulator from its return type, making the
  multiplication an unchecked use of a nullable value.
- `fluttorch_gen` no longer declares a `build_runner` builder factory that throws, which turned
  adding the package as a dependency into a build failure instead of a no-op.

### Internal

- M2: the pipeline runs end to end. A two-layer model exported, loaded on macOS through
  `executorch_flutter`, and its output within 2.98e-8 of the PyTorch reference. It also established
  that `executorch_flutter` requires macOS 11.0, which a freshly generated Flutter project does not
  meet.
- M1: audited `executorch_flutter` 0.5.0. The parity gate needs four hooks, and none of them exists
  in its public API: activation taps, deterministic execution, load-time backend selection, and
  caller-supplied output buffers. Recorded on the [board](https://github.com/orgs/NaCode-Studios/projects/6);
  the consequence is that M19 becomes a question of when to fork rather than whether.
- CI runs every Dart package's suite rather than only the core, and the Python manifest suite, which
  needs neither `torch` nor `executorch`.
- 109 tests: 90 Dart across three packages, 19 Python.

[Unreleased]: https://github.com/NaCode-Studios/Fluttorch/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/NaCode-Studios/Fluttorch/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/NaCode-Studios/Fluttorch/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NaCode-Studios/Fluttorch/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/NaCode-Studios/Fluttorch/releases/tag/v0.0.1
