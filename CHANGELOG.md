# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries name the roadmap milestone they correspond to, e.g. `(M14)`, so a claim on the
[board](https://github.com/orgs/NaCode-Studios/projects/6) can be cross-checked against what actually shipped.

## [Unreleased]

## [0.7.0] - 2026-08-02

### Added

- A tensor spec can record its layout, `nchw` or `nhwc`, and `resize` and
  `center_crop` are generated from it. Both were understood and refused
  before, because performing either means knowing which axes are spatial and
  nothing recorded it, which stopped every image model at the door. Absence is
  not a default: a spec that says nothing is not a spec that says NCHW, and
  asking one that does not say raises rather than answering. The two steps
  change the element count, so they become a `fromSource(values, height:,
  width:)` constructor rather than joining the in-place `preprocess()` pass.
- Three new refusals in the generator, each a case where emitting something
  would be worse than emitting nothing: a resize filter this build does not
  implement is named rather than approximated with bilinear, a spatial step
  recorded after an elementwise one is refused rather than silently reordered,
  and spatial steps against a multi-input model are refused because nothing
  says which input they apply to.
- Every `FluttorchException` carries a `remedy` beside its `message` (M30). The
  cause and the fix are different sentences and a caller showing a failure to a
  user usually wants only the second, and it is a field rather than a
  convention so a member cannot quietly stop having one.
- `DTypeUnsupportedException`, for a backend with no kernel for an element type.
  It names the tensor, the type, the backend and what that backend does carry.
- VoltaCast, the flagship example (M29): a seq2seq Transformer forecasting
  Italian electricity demand a day ahead, trained on eleven years of real data,
  exported here and measured through LiteRT against references captured from the
  source model on real midnight origins. Worst absolute drift `5.96e-7`. It is
  also the first model this binding has carried with more than one input, and
  the generator handled it unchanged.
- A documentation site at
  [nacode-studios.github.io/Fluttorch](https://nacode-studios.github.io/Fluttorch/)
  (M28): concepts, a quickstart, where a tolerance comes from, and the error
  taxonomy. Written to argue rather than to enumerate, because the API is
  generated and typed and what a reader cannot get from it is why each refusal
  is worth more than the convenience it costs.
- Each published package gains a `CHANGELOG.md` and an example that runs, which
  is what pub.dev reads and what it was scoring them down for.

### Fixed

- The generated `center_crop` took the wrong row on an odd margin. Python rounds
  a tie to the even neighbour and Dart rounds away from zero, so they disagree on
  three of five half-values, and the result was still a picture. The fixture that
  caught it leaves a margin of five on one axis on purpose, and the reference
  comes from torch rather than from this code.
- The three runtimes reported "this backend has no kernel for that element type"
  with `DTypeMismatchException`, whose message reads "tensor features holds
  float32 and was read as int8". Nothing had read anything: the manifest and the
  buffer agreed and the device could not carry them. That message sends a reader
  to look at code that is correct. The suite had pinned it.

### Internal

- The ExecuTorch build installs the Python its kernel codegen imports. The four
  generators CMake invokes import `yaml`, and three of them import `torchgen`,
  which ships inside the torch wheel rather than as a package of its own. The
  workflow set up an interpreter and put nothing in it, and because the codegen
  is the last target of the build, the failure arrived at 100 per cent, an hour
  and a quarter after the omission that caused it.
- The on-device build compiles the tracer hooks. `EXECUTORCH_BUILD_DEVTOOLS` was
  absent from the workflow's configure, so the runtime it built could not read
  intermediates and the suite failed the one assertion that says so. A job that
  exercised three of the four hooks this binding exists for was measuring less
  than it appeared to. The cache key now covers the workflow as well as the
  build script, because the configure lives here and a changed one was restoring
  a tree compiled under the old flags.
- `On device` runs on demand and no longer on a weekly schedule. The first
  complete run measured that hour and a quarter on a three-core macOS runner,
  which is the largest single block of compute this repository asks for, and a
  timer measures the calendar rather than what changed. Nothing about the answer
  moves in a week where the native half did not.

## [0.6.0] - 2026-08-02

### Added

- Inference runs off the thread that draws. Every call through the FFI seam
  is synchronous, and the Dart API above it returned futures that never suspended,
  so in a Flutter app a model ran on the platform thread for as long as it took.
  `IsolateExecuTorchRuntime` moves the native side onto a worker isolate: the
  library is opened there, models are created there, and they never leave, because
  an `ft_model_t` is not safe to touch from two threads and a handle that crossed
  back would be exactly that. It costs a copy of every tensor each way, which the
  class says rather than implies.
- The binding reaches a phone. ExecuTorch and the shim cross-compile
  for `arm64-v8a` and for `arm64-apple-ios`, and `fluttorch_executorch_flutter`
  puts the result inside an app: `jniLibs` on Android, a vendored static archive
  force-loaded into the binary on iOS, where an app cannot load a dylib from its
  bundle. A model quantized to `int8-dynamic` runs on Android arm64 and produces
  drift identical to macOS in every digit reported.
- Three engines behind one seam (M26, M27). `fluttorch_onnx` and
  `fluttorch_litert` implement the same C ABI `fluttorch_executorch` implements,
  so the manifest, the goldens and the parity gate are unchanged and what a second
  and a third runtime change is the engine. Both run a model exported by this
  toolchain and hold their goldens against references byte-identical to the ones
  every ExecuTorch export is measured against.
- The manifest records which engine executes an artifact, and refuses to be
  loaded by another one. A `.pte`, a `.onnx` and a `.tflite` are not
  interchangeable, and the weight hash cannot tell them apart because it is
  computed over whichever was written, so handing one to the wrong runtime got
  past every check this project had.
- `tool/prepare_executorch.sh` performs the workarounds an ExecuTorch checkout
  needs, rather than describing them in a comment that a clean clone does not
  have: the Core ML protobuf sources, and the `flatc` that has to run on the build
  machine rather than on the phone.

### Fixed

- An ONNX export whose weights left the graph is refused rather than written.
  Above a size `torch.onnx` decides on its own, the weights move into a sidecar
  beside the artifact, and two things break at once without announcing
  themselves: the weight hash is computed over the artifact, which is then a few
  kilobytes of graph structure with the numbers outside it, and the runtime loads
  an artifact as bytes so it could not reach the sidecar anyway. A bundle that
  passes every check and carries no weights is the failure this project exists to
  prevent. The empty sidecar written for every export is removed rather than
  shipped.
- `int8-static` exports where the toolchain allows it, and says why where it does
  not. torchao introspects an operator overload torch 2.13 does not expose, before
  the model is involved, and 2.12 does. Both are reachable: `litert-torch` pins
  torch below 2.13 and `executorch` has no upper bound, and which one pip picks
  differs by platform, so a macOS checkout converts the recipe and a Linux runner
  refuses it. The suite asserts what holds on either: it converts and the manifest
  says which recipe, or it refuses and names the combination at fault.

### Internal

- The on-device workflow clones ExecuTorch under the only directory name it
  accepts. Upstream's CMakeLists refuses to configure under anything but
  `executorch`, and this cloned into `.executorch` to keep a multi-gigabyte
  checkout out of Dart's way, so the job failed to configure the first time it
  was dispatched.

## [0.5.0] - 2026-08-01

### Added

- The seam between Fluttorch and ExecuTorch is declared as a C ABI in
  `packages/fluttorch_executorch/src/fluttorch_executorch.h` (M20), and mirrored in
  Dart by `ExecuTorchBindings`. It carries the four hooks no published binding
  exposes: a backend pinned at load, execution repeatable enough that a tolerance
  measures the model rather than the noise, activation taps, and output buffers the
  caller owns.
- `ExecuTorchRuntime` is implemented on top of that seam. The artifact is verified
  before anything native sees it; a backend that does not exist is refused with the
  list; a fallback reports the backend that ran rather than the one requested;
  determinism fails rather than being quietly dropped; and a tap the graph does not
  carry stays absent instead of arriving zero-filled, because a zero reads as
  agreement.

- The native half of the binding exists and runs (M20, M21). `src/fluttorch_executorch.cpp`
  implements the ABI against ExecuTorch's `Module`, and `tool/build_native.sh` links it
  against a checkout into a shared library. On this machine a model quantized with
  `int8-dynamic` by our own exporter loads through it on XNNPACK, and all four of its
  goldens land inside the tolerance the recipe starts from, drifting between `2.4e-3`
  and `4.1e-2` relative. Held to the full-precision bound instead, the same run fails
  every case, which is what makes the first result mean something.
- The exporter lowers for Core ML as well as XNNPACK (M21), an artifact says which one it
  was lowered for, and the Core ML runtime now links and executes. A Core ML `.pte` from
  our own exporter loads through this binding on an M-series Mac, and all four of its
  goldens hold at the full-precision bound, in the same suite as the quantized XNNPACK
  case. What blocked it was one absent directory. ExecuTorch's
  `backends/apple/coreml/scripts/install_requirements.sh` runs under `set -e` and invokes
  `python`, so where only `python3` is on PATH it exits at its pip step, short of the
  commands that build coremltools' `mlmodel` target and copy the protobuf sources that
  produces into `runtime/sdk/format/`. Without that directory the delegate's SDK sources
  are dropped from the CMake target, and `libcoremldelegate.a` references
  `ETCoreMLModelAnalyzer`, `ETCoreMLModelDebugInfo` and `ModelEventLoggerImpl` without
  containing them. `tool/build_native.sh` links Core ML where the checkout can supply it,
  says which of the two it did, and records the four upstream frictions so the next
  person does not rediscover them one build at a time.
- The manifest records the compute precision a delegate was lowered at, and Core ML and
  MPS ship at the float16 they run by default. The two are one change: a recipe says how
  the weights were stored and a precision says what the delegate does arithmetic in, and
  a backend can halve the second while leaving the first alone. Without the field an
  artifact lowered at half precision answered to a full-precision bound it could not hold,
  so the gate failed a model doing exactly what it was told. Absence still means float32,
  which is what a reader without the field assumed anyway.
- The tolerance table sizes a bound from the precision as well as the recipe, and the two
  compound rather than replace: an int8 model on a half-precision GPU is wrong in both
  ways at once. Float16's starting point is `2e-2` relative rather than the small multiple
  of its `4.9e-4` epsilon that looks right. A gate sees outputs while rounding happens on
  intermediates, so feeding this project's model an input of magnitude `1e3` puts its
  intermediates where float16 spacing is about `0.5` while one output element lands near
  `9.4`, and a case well inside float16's documented accuracy still shows `1e-2` relative.
  It stays inside the `5e-2` an int8 recipe is given, which keeps the two distinguishable,
  and a float16 export held to the float32 bound still fails every case.
- The two backends differ in what they can promise, and the binding says so per backend
  rather than per build. XNNPACK on a single-threaded pool fixes the order of every
  reduction; Core ML chooses between the Neural Engine and the GPU and promises no such
  thing, so asking it for deterministic execution is refused rather than granted and
  hoped for.
- Activation taps read intermediates off the device (M20), which is the last of the four
  hooks the ABI was declared around. A tracer built into the binding keeps the layers a
  caller asked for and drops the rest, so answering for three layers does not copy the
  whole graph. On the two-layer model all three declared taps come back: the activation
  is exactly the ReLU of the layer before it, the last tap is the model's output, and
  each one matches the reference the export captured to within `1e-5`.
- The manifest records where each tap lives in the lowered graph (M20). Submodule names
  do not survive lowering, so a runtime addresses an intermediate by debug handle and by
  nothing else; only the export sees both names and handles, and `activation_handles`
  is where it writes the correspondence down. Additive, like the activations it is
  positional with.
- Taps are refused on a delegated backend rather than declared and left unanswerable. A
  delegated partition is one instruction carrying no layer inside it, so `fc1` is not
  slow to reach within an XNNPACK partition, it is absent from it. A bundle promising
  attribution the device can never deliver reports every layer missing on every run,
  which reads exactly like every layer agreeing. The refusal names the way out.
- The exporter lowers for `portable` as well as the two delegates (M20), which is the
  export that makes attribution possible: no partitioner, every operation in the
  runtime's own kernels. Slower than any delegate and not what a device would ship,
  which is the point of having it separately.
- The exporter knows eight backends and reports which of them a given machine can
  actually lower for (M23). `available_backends()` answers by lowering a one-operation
  model rather than by importing a partitioner. Metal is the reason: its partitioner
  constructs on any Mac and then fails during preprocessing, looking for a torchao dylib
  whose build flag it never names. A list built from imports would call Metal available
  and be wrong exactly where it mattered.
- A backend a machine cannot lower for is refused with the piece that is missing (M23),
  rather than with the error from three libraries down. An absent Qualcomm SDK surfaces
  upstream as an `ImportError` about a Python package nobody asked for, and Metal as a
  filename with no flag attached. Both now name the toolchain, and the original failure
  stays attached to the message.
- MPS runs (M23). A model lowered for it loads through this binding on an M-series Mac
  and all four of its goldens hold at the full-precision bound, which makes it the third
  backend measured rather than described. It refuses deterministic execution instead of
  promising it, because a GPU does not undertake to schedule work the same way twice.
- `tool/build_native.sh` links whichever delegates the checkout built and names the ones
  it did not (M23). Skipping is the designed outcome rather than a degraded one: a
  machine that never built Vulkan should get a library reporting that it cannot run
  Vulkan, not one that fails to link or claims a backend it lacks.
- `portable` is a backend the runtime reports rather than an absence a caller infers, and
  the backend an unpinned load takes is now named in one place. It used to be whichever
  entry stood first in the table, which was harmless at two entries and became a way to
  change what every unpinned load runs by editing a list.
- One report covers every backend a machine offers (M24). `measureMatrix` replays one set
  of goldens across several loaded models and returns a table, rows by golden and columns
  by backend, each cell measured at the tolerance that export's own recipe implies. An
  int8 export and a float32 one of the same model are not wrong by the same amount, and a
  single bound would either excuse the second or condemn the first. Entries whose goldens
  do not line up are refused rather than tabulated, because a matrix over different inputs
  is a set of unrelated numbers arranged to look like a comparison.
- `dart run tool/parity_matrix.dart` prints that report and exits non-zero when a cell
  fails. On this machine it covers four backends across four goldens: three carry the
  model at float32 and agree with the source to within `1e-7` relative, and the one
  carrying it at `int8-dynamic` moves by up to `4.1e-2`. A table whose columns all read
  the same would mean the quantized artifact was not quantized. Backends the build lacks
  are listed as not run rather than omitted, since an absent column and an agreeing column
  look identical once a table is printed.
- Backend claims are checked on hardware, or recorded as not checked (M25). The `Backends`
  workflow runs the export suite on a Linux runner, which keeps the two halves of a
  backend claim apart: that runner lowers for Core ML and MPS, which it could never run,
  and refuses Metal, MLX and QNN by naming the toolchain each would need. The `On device` workflow builds ExecuTorch on an
  Apple silicon runner and runs the parity matrix there, weekly and on demand rather than
  on every push, because a gate that takes the better part of an hour on every push is one
  somebody eventually routes around. `docs/backend-coverage.md` records which half of each
  backend's claim is verified where, and which are not run at all.

### Changed

- The quantized lowering imports `convert_pt2e` from `torchao` rather than from
  `torch.ao`, where the pt2e flow no longer lives. Written against the old path, it
  had never been executed: neither the development machine nor CI carried torch, so
  M17 shipped in `0.4.0` with its exit criterion claimed rather than earned.
- `FluttorchRuntime.load` takes `deterministic`. It belongs on the seam rather than
  on one backend, since it is the difference between a tolerance that measures a
  model and one that measures run-to-run noise.

### Fixed

- The goldens under `testdata/quantized/` were not the ones the sample model produces.
  They carried different inputs, random draws rather than the four cases chosen for what
  they exercise, and their reference outputs did not match the source model on those
  inputs either. The bundle was internally consistent, so the gate passed, and it was
  measuring an artifact against references no current code produces. Regenerated, and the
  four exports now share byte-identical references, which is what let the matrix compare
  them at all.
- The quantized suite's upper bound on drift is relative rather than absolute. It asked
  for `maxAbsolute < 0.01` across goldens whose outputs range from about `0.1` to about
  `175`, which is six significant figures out of int8 on the widest case and no constraint
  at all on the narrowest.
- `NativeExecuTorchBindings` binds that ABI over `dart:ffi`, and is tested against a
  C library the suite compiles and calls. A Dart fake cannot check struct field
  offsets, arrays of strings or pointer arithmetic over tensor arrays, because a
  wrong offset there returns plausible nonsense instead of failing to compile. It
  earned its keep immediately: the first run found a pointer kept past the call
  that owned it, which read freed memory and returned a backend name that looked
  like text.

- M22, migrating off the interim dependency, is closed as not planned rather than
  built. `fluttorch_executorch` was never published and its runtime always threw, and
  `executorch_flutter` was only ever used directly by the M2 spike, which is an
  example rather than an API anyone depends on. A deprecation window guards a promise
  that was never made. The argument is on the board so the milestone is not
  re-argued.


- `int8-dynamic` and `int4-weight-only` export, which is now asserted by running
  them rather than by describing them. `int8-static` does not: `torchao` introspects
  an operator overload that torch 2.13 does not expose, before the model is
  involved, and the failure surfaced as the name of a pass nobody has heard of. The
  exporter now says which toolchain combination is at fault and which recipes work
  on it, and a test pins that message so the day it is fixed upstream, the test
  fails and this entry goes.
- Tap capture is asserted against a real export: the activations are declared in
  graph order with shapes observed by running the model, every case records a key
  for each, and the files written are the length the spec declares.
- The linked artifact records are no longer skipped when the attestation fails, and
  they name the repository when the endpoint cannot work it out. It resolves which
  repository built an artifact from the artifact's attestation, so an archive without
  one has nothing to resolve and the write answers "no artifacts found" until the
  repository is named outright. Both were found by the first tag that published.
- Compiled Python is no longer committed. Nineteen `.pyc` files were tracked, each
  carrying the absolute path of the machine that built it, and `.gitignore` had no
  Python section at all, so they returned every time anyone ran the suite. They stay
  in the history, which rewriting would cost every tag and every published release to
  remove; what stops is adding more.

## [0.4.0] - 2026-08-01

Tier 4, quantization and attribution. An export can now be quantized, and a failing
gate can say which layer the numbers went wrong at rather than only which output was
wrong.

### Added

- Three quantization recipes (M17): `int8-dynamic`, `int8-static` and
  `int4-weight-only`, named in the manifest and read back on the device, where the name
  selects the tolerance the gate starts from. A static recipe refuses to export against
  fewer than two golden inputs: it fixes every activation range from what it observed,
  so calibrating on one sample produces a model that passes its own single case and
  clips everything else. The exporter cannot know what a representative set looks like
  for a model it was handed, so it asks rather than guesses.
- Reference activations in the manifest (M18). `--taps` names submodules whose outputs
  are captured alongside each golden, in the order the graph produces them, which is
  what makes "the earliest layer that diverged" a claim rather than a pick. Off by
  default, because an intermediate is as large as the tensor it carries.
- Per-layer attribution in the parity gate (M18). Where the export captured taps and the
  backend offers them, the report names the earliest layer whose activations moved and
  by how much. Both conditions are required and neither is assumed: a report that could
  not look says which of the two was missing. A layer the backend did not tap is treated
  as a hole rather than as agreement, so nothing after it is ruled out unless the
  divergence was already found before it.
- A recorded decision on the runtime (M19), on the
  [board](https://github.com/orgs/NaCode-Studios/projects/6). The four hooks M1 found
  missing are proposed to `executorch_flutter` first, and the fork starts on schedule if
  they are not merged by the time Tier 5 would begin.

### Fixed

- The publish jobs are written out rather than calling
  `dart-lang/setup-dart/.github/workflows/publish.yml`. That workflow declares
  `permissions: id-token: write` on its own job, and a permissions block in a
  called workflow narrows the set again, so whatever the caller grants, its
  checkout runs without `contents` and a private repository answers "repository
  not found". It works on public packages and cannot work here.
- The publish jobs could not check the repository out. A job-level `permissions`
  block replaces the default set rather than extending it, so declaring only
  `id-token: write` left `contents` at none, and on a private repository
  `actions/checkout` reports that as "repository not found" rather than as a
  permission error. It surfaced on the first tag that tried to publish, which is
  the first tag that could have surfaced it.

### Internal

- `release.yaml` publishes `fluttorch`, `fluttorch_gen` and `fluttorch_test` to pub.dev
  over OIDC, so no token is stored anywhere and the package pages carry the verified
  publisher line. The two dependants wait for the core version to be servable before
  they upload, because they declare it as a dependency and pub.dev refuses a package
  whose dependencies do not resolve. Creating the GitHub Release deliberately does not
  wait on any of that: it is derived from the tag and the changelog and is true whatever
  the registry did, and chaining it would mean a re-run after a partial publish fails on
  the package that already went up.
- The manifest gained `activations` and a per-golden `activations` key list. Additive, so
  the schema version does not move: a reader that does not know the fields compares final
  outputs and says it could not look deeper, which is what it did before they existed.
  Both are exercised by the shared fixture that each language re-encodes byte for byte.
- The quantization recipe names are asserted equal across the two languages. A recipe the
  exporter can apply and the gate has no tolerance for produces a model the gate cannot
  judge, and the failure is silent, so the Python suite reads `knownRecipes` out of the
  Dart source rather than trusting the two lists to be kept in step by hand.
- The lowering path for a quantized export is written and not executed. Neither this
  machine nor CI has `torch`, so the recipe table, the calibration refusal and the manifest
  round trip are covered while the quantized artifact itself is not. Running
  `pytest python/fluttorch_export/tests` where torch is installed is what would close that
  gap.
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

[Unreleased]: https://github.com/NaCode-Studios/Fluttorch/compare/v0.7.0...HEAD
[0.7.0]: https://github.com/NaCode-Studios/Fluttorch/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/NaCode-Studios/Fluttorch/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/NaCode-Studios/Fluttorch/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/NaCode-Studios/Fluttorch/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/NaCode-Studios/Fluttorch/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/NaCode-Studios/Fluttorch/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NaCode-Studios/Fluttorch/compare/v0.0.1...v0.1.0
[0.0.1]: https://github.com/NaCode-Studios/Fluttorch/releases/tag/v0.0.1
