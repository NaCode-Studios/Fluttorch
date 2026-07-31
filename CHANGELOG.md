# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries name the roadmap milestone they correspond to, e.g. `(M14)`, so a claim on the
[board](https://github.com/orgs/NaCode-Studios/projects/6) can be cross-checked against what actually shipped.

## [Unreleased]

Nothing yet.

## [0.2.0] - 2026-07-31

Tier 2 — Typed bindings. A manifest now becomes a Dart API in which handing the
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
  means knowing which axes are spatial, and the manifest records no tensor layout —
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

Tier 1 — The export contract. One command now produces an artifact, its manifest
and its goldens together, and the Dart side refuses an artifact that does not
match the manifest it was handed.

### Added

- `fluttorch-export` (M5): resolves `module:factory` references, lowers through
  XNNPACK at full precision, and writes the artifact, the manifest and the goldens in
  one command — together, because the weight hash can only catch a mismatched pair if
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
  write the shortest round-trip decimal and then spell it differently — switching to
  exponential notation at different magnitudes, padding the exponent differently — and
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
- 194 tests locally — 117 Dart, 77 Python. CI runs 117 and 56.

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
  in its public API. Recorded on the [board](https://github.com/orgs/NaCode-Studios/projects/6); the consequence is that M19 becomes a
  question of when to fork rather than whether.
- CI runs every Dart package's suite rather than only the core, and the Python manifest suite, which
  needs neither `torch` nor `executorch`.
- 109 tests: 90 Dart across three packages, 19 Python.

[Unreleased]: https://github.com/NaCode-Studios/Fluttorch/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/NaCode-Studios/Fluttorch/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NaCode-Studios/Fluttorch/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/NaCode-Studios/Fluttorch/releases/tag/v0.0.1
