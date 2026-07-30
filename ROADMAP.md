# Fluttorch roadmap

Where Fluttorch is going, milestone by milestone. Released versions and dates live in
[`CHANGELOG.md`](CHANGELOG.md); the short human summary lives in the
[README Roadmap section](README.md#roadmap). The structure and the shipped-state vocabulary follow
[`ROADMAP-CONVENTIONS.md`](ROADMAP-CONVENTIONS.md).

Fluttorch is pre-`1.0` and has no released version yet. Until `1.0`, minor versions may break the
public API; a `STABILITY.md` arrives with the `1.0` line.

Each tier closes with its own tag, and no tag mixes tiers — see
[`ROADMAP-CONVENTIONS.md`](ROADMAP-CONVENTIONS.md). Tier 0 cuts `0.0.1` rather than a minor, because
it delivers decisions, a schema and a set of semantics but nothing a user can yet invoke; Tier 1 is
the first release with a feature in it, and takes `0.1.0`. From there the tiers and the minors run
together up to `1.0`.

| Tier | Tag |
| --- | --- |
| 0 · Foundations | `0.0.1` |
| 1 · The export contract | `0.1.0` |
| 2 · Typed bindings | `0.2.0` |
| 3 · The parity gate | `0.3.0` |
| 4 · Quantization & attribution | `0.4.0` |
| 5 · Own runtime | `0.5.0` |
| 6 · Backend breadth | `0.6.0` |
| 7 · Runtime independence | `0.7.0` |
| 8 · Docs & example | `0.8.0` |
| 9 · Stabilisation | `0.9.0`, then `1.0.0` |

## Status — `0.1.0` (current)

Tier 1 closed on 2026-07-30. `fluttorch-export` produces an artifact, a manifest and a set of goldens in
one command, and the Dart side refuses an artifact whose digest disagrees with the manifest it was
handed.

M8 paid for itself: asserting the round trip over schema-legal documents nobody writes by hand found
that the byte-for-byte agreement claimed in `0.0.1` held only because the fixture happened to contain
values the two languages spell the same way. It is now true for the awkward ones too.

What still does not exist is a typed API. Tier 2 generates it.

## Status — `0.0.1`

Tier 0 closed on 2026-07-30. The contract exists and is enforceable: a manifest that Python writes and
Dart re-encodes byte for byte, a tensor type that checks `bytes.length == elements × width` at
construction, a tolerance whose three bounds all do something, and a spike proving a PyTorch model
reaches a Flutter app with 2.98e-8 of drift.

Nothing in it is a feature a user can invoke. That arrives with Tier 1 and the export CLI.

## Guiding principles

- **The parity gate is the product.** Typed bindings are convenience. Catching numerical drift that
  no build currently catches is the reason to exist, and every trade-off resolves in its favour.
- **One contract, three consumers.** Shapes, dtypes, preprocessing, labels and the weight hash are
  emitted once by the exporter. Anything hand-written on the Dart side that restates them is a
  defect, because it can drift from training.
- **Runtime-agnostic core.** A backend may be imported by `fluttorch_executorch` and nothing else.
  If a feature cannot be expressed through the runtime interface, the interface is wrong, not the
  feature.
- **Capabilities reported, never assumed.** Activation taps and deterministic execution are
  properties of a device. Code that wants them degrades when they are missing, and a report always
  names the backend it measured.
- **No claim without a measurement.** Backend support is verified on that backend or it is not
  claimed. Coverage that was never run is documented as not run.

## Progress

| Milestone | Status |
| --- | --- |
| M1 · Audit `executorch_flutter` | ✅ Shipped in `0.0.1`. |
| M2 · End-to-end spike | ✅ Shipped in `0.0.1`. |
| M3 · Manifest schema v1 | ✅ Shipped in `0.0.1`. |
| M4 · Tolerance semantics | ✅ Shipped in `0.0.1`. |
| M5 · Export CLI | ✅ Shipped in `0.1.0`. |
| M6 · Signed manifest with weight hash | ✅ Shipped in `0.1.0`. |
| M7 · Golden capture from the source model | ✅ Shipped in `0.1.0`. |
| M8 · Manifest round-trip tests | ✅ Shipped in `0.1.0`. |
| M9 · `build_runner` builder | Planned. |
| M10 · Generated preprocessing | Planned. |
| M11 · Compile-time shape and dtype checks | Planned. |
| M12 · Golden tests for generated code | Planned. |
| M13 · Golden replay runner | Planned. |
| M14 · Drift metrics | Planned. |
| M15 · `expectParity` matcher | Planned. |
| M16 · CI recipe for the XNNPACK gate | Planned. |
| M17 · Quantization recipes | Planned. |
| M18 · Activation taps and layer attribution | Planned. |
| M19 · Schedule the fork | Planned. |
| M20 · `dart:ffi` binding with deterministic execution | Planned. |
| M21 · Backends — XNNPACK and Core ML | Planned. |
| M22 · Migration off the interim dependency | Planned. |
| M23 · Backends — QNN, Vulkan, Metal and MLX | Planned. |
| M24 · Parity matrix across backends | Planned. |
| M25 · Real-device CI | Planned. |
| M26 · LiteRT backend | Planned. |
| M27 · ONNX Runtime backend | Planned. |
| M28 · Documentation site | Planned. |
| M29 · VoltaCast example, end to end | Planned. |
| M30 · Error taxonomy and diagnostics | Planned. |
| M31 · API freeze and deprecation policy | Planned. |
| M32 · Performance and memory benchmarks | Planned. |
| M33 · `1.0` | Planned. |

**Deferred sub-items.** On-device training and LoRA adapter hot-swap are deferred: ExecuTorch
supports training, but the demand is thin and the surface large, and neither is needed for the parity
thesis. Federated learning is declined for the same reason with less demand still. Web is deferred
until a runtime interface exists that a WASM backend can satisfy without special-casing.

## Effort legend

`S` ≈ hours to a day · `M` ≈ several days · `L` ≈ one to two weeks · `XL` ≈ multi-week.

## Tier 0 — Foundations — ✅ Shipped in `0.0.1`.

Decide the things that are expensive to change later, and prove the pipeline exists before
abstracting it.

### M1 · Audit `executorch_flutter` — `S`

**Status: ✅ Shipped in `0.0.1`.**

Delivered: all four hooks are absent from version 0.5.0. `ExecuTorchModel` exposes `modelId`,
`load` / `loadFromBytes` / `loadFromAsset`, `forward(List<TensorData>)` and `dispose`, and nothing
else. There is no layer or activation API anywhere in `lib/`; no seed or determinism control;
`executorch_manager_base.dart` contains no reference to a backend at all, so selection happens in the
build configuration and the same artifact cannot be loaded on two backends within one process; and
`forward` allocates its outputs rather than accepting a destination. The licence is MIT, so there is
no compatibility problem.

- Depend on it through Tier 3. `load` and `forward` are exactly what the export toolchain, the
  generator and final-output parity need, and reimplementing them first would spend the early weeks
  on the part where nothing is differentiated.
- M18 and M24 cannot be built on it. Per-layer attribution needs activation taps and deterministic
  execution; the parity matrix needs load-time backend selection. Two of those do not exist and the
  third is build-time only.
- **M19 therefore changes from a decision to a schedule.** The fork is required, and what remains
  open is when it lands, not whether.

### M2 · End-to-end spike — `M`

**Status: ✅ Shipped in `0.0.1`.**

Delivered: `examples/spike/` exports a two-layer model, loads it on macOS through
`executorch_flutter`, and compares one output against the reference captured before lowering —
`max |Δ| 2.98e-8` against a bound of `1e-5`, with the weight hash matching.

- macOS is the target because it needs neither a device nor a simulator.
- It established that `executorch_flutter` requires macOS 11.0, which a freshly generated Flutter
  project does not meet, and that the seam in `FluttorchRuntime` maps onto the dependency with
  nothing left over.
- It confirmed M1's gap in practice: no backend pinning, no activation, no destination buffer.

### M3 · Manifest schema v1 — `M`

**Status: ✅ Shipped in `0.0.1`.**

Delivered: schema v1 with a writer in `fluttorch_export.manifest`, a reader in `ManifestCodec`, and
`testdata/manifest_v1.json` as the artefact that binds them — Python writes it and Dart re-encodes it
byte for byte, so a divergence in either implementation fails a test instead of surfacing later as a
wrong prediction. A schema on its own would have let both sides drift while each stayed valid.

- Shapes, dtypes, preprocessing steps, labels, weight hash and golden index, versioned from the first
  release so an artifact stays readable after the format moves on.
- Every decode failure names its field as a dotted path. A newer schema version is refused with both
  versions named, because the fix there is upgrading the reader.
- An unrecognised preprocessing step is preserved rather than dropped, and the generator refuses a
  manifest containing one: tolerating it at the boundary and rejecting it at the generator is what
  catches a transform this build cannot perform.

### M4 · Tolerance semantics — `S`

**Status: ✅ Shipped in `0.0.1`.**

Delivered: the elementwise bounds combine as `atol + rtol * |reference|` rather than as two
independent tests, which is the only form that works both near zero and on large magnitudes. Cosine
is checked per tensor and catches the case both elementwise bounds miss, where every value is small
enough to pass individually while the tensor points elsewhere.

- A tolerance with no bound configured is a constructor error. It previously accepted everything,
  which for a gate is the worst available failure.
- A NaN where a number was expected fails however wide the bound; a NaN matched against a NaN passes.
- Recipe defaults are named `startingPointFor` and return null for an unrecognised recipe rather than
  inventing a threshold. **The four values are starting points, not measured thresholds**, and are to
  be replaced once M17 has produced evidence.

## Tier 1 — The export contract — ✅ Shipped in `0.1.0`.

One command produces the artifact, the manifest and the goldens together, so they cannot be
generated from different states of the model.

### M5 · Export CLI — `L`

**Status: ✅ Shipped in `0.1.0`.**

Delivered: `fluttorch-export` resolves `module:factory` references, lowers through XNNPACK at full
precision, and writes the artifact, the manifest and the goldens together.

- Together on purpose: the weight hash only catches a mismatched pair if nothing can produce one
  without the other.
- `--input-names` and `--output-names`, because a generated accessor called `input_0` is one nobody
  wants to read. `--dynamic-batch` marks the leading dimension dynamic.
- Naming a quantization recipe fails rather than exporting full precision under it, which would put a
  recipe in the manifest the artifact does not match.

- `torch.export` to a runtime artifact. XNNPACK and full precision only.
- Model and example inputs addressed as `module:factory`, so the CLI never imports a notebook.
- Reproducible output for a fixed input, because everything downstream compares against it.

### M6 · Signed manifest with weight hash — `M`

**Status: ✅ Shipped in `0.1.0`.**

Delivered: the exporter records a `sha256:` digest of the artifact, and `verifyArtifact` refuses one
that disagrees.

- A single flipped byte, a truncation, and a hash with no algorithm are all caught.
- An algorithm this build cannot compute is a `ManifestFormatException`, not a mismatch: the reader is
  old, the model is fine, and the fixes are different.

- Content hash of the exported weights recorded in the manifest, and the runtime refuses to load an
  artifact whose hash disagrees.
- This is what stops a stale model from silently passing a green parity suite.

### M7 · Golden capture from the source model — `M`

**Status: ✅ Shipped in `0.1.0`.**

Delivered: references captured from the model before lowering, written as raw little-endian bytes the
Dart `Tensor` wraps without copying.

- Without supplied cases the exporter captures the example input alone and says that one case proves
  the pipeline runs rather than that the model survived export.
- An empty golden iterable is refused rather than silently treated as none, because a broken factory
  and a deliberate omission are different things.

- Reference pairs taken from the model *before* lowering, so they are a reference rather than a
  snapshot of whatever the export happened to produce.
- Count configurable; the default is small enough to commit.

### M8 · Manifest round-trip tests — `S`

**Status: ✅ Shipped in `0.1.0`.**

Delivered: the full field set plus the schema-legal documents nobody writes by hand — a rank-0 tensor,
a fully dynamic shape, denormals, negative zero, values either side of both notation thresholds, and
non-ASCII in names and labels.

- It found that the byte-for-byte claim made in `0.0.1` was accidentally true. Both languages write
  the shortest round-trip decimal and then spell it differently, and Python escaped non-ASCII where
  Dart does not.
- Fixed by making the Python writer canonical rather than by weakening the claim: Python is the only
  writer in the pipeline, so it matches the reader.

- Python writer against Dart reader over the full field set, including the cases the schema allows
  but nobody writes by hand.

## Tier 2 — Typed bindings

The generated API is the only thing app code touches, and it is generated from the contract.

### M9 · `build_runner` builder — `L`

- Consumes `*.fluttorch.json`, emits `*.fluttorch.dart` carrying shapes, dtypes and labels.
- Loading and disposal generated alongside, so nothing about tensor layout reaches app code.

### M10 · Generated preprocessing — `L`

- Normalization, resize, crop and dtype conversion emitted from the manifest.
- This is the milestone that removes train/serve skew structurally rather than by discipline.

### M11 · Compile-time shape and dtype checks — `M`

- Static guarantees for fixed shapes; a documented escape hatch for dynamic dimensions that does not
  quietly become the common path.

### M12 · Golden tests for generated code — `S`

- The generator is itself a source of defects. Pin its output.

## Tier 3 — The parity gate

### M13 · Golden replay runner — `M`

- Loads a bundle, runs every case against a loaded model, collects raw outputs without judging them.

### M14 · Drift metrics — `M`

- Max absolute, mean absolute, cosine, and the flat index of the worst element, so a failure is
  locatable instead of merely reported.

### M15 · `expectParity` matcher — `M`

- Fails with a report naming the tensor, the drift, the tolerance and the backend measured.
- The failure message is the deliverable here; a gate nobody can read gets disabled.

### M16 · CI recipe for the XNNPACK gate — `S`

- A workflow that runs the parity suite on every pull request, copy-pasteable into a consumer repo.

## Tier 4 — Quantization and attribution

Where the interesting drift comes from, and where the runtime question gets settled with evidence.

### M17 · Quantization recipes — `L`

- int8 static, int8 dynamic, int4 weight-only in the exporter, each carrying its own default tolerance.

### M18 · Activation taps and layer attribution — `XL`

- Capture intermediates and report the earliest layer whose activations diverged.
- Turns "the output is wrong" into "this op is wrong", and is the feature that requires runtime-level
  access.

### M19 · Schedule the fork — `S`

- M1 already settled whether: the hooks M18 and M24 need are absent from the dependency, so the fork
  happens. What is open is when, and that depends on how much of Tier 4 can be delivered against
  final outputs alone.
- Re-check the dependency before committing. It is actively developed, and a hook that appears
  upstream is a hook not worth writing twice.
- Record the decision either way, including the version audited, so the next person does not repeat
  the reading.

## Tier 5 — Own runtime

Undertaken because the gate needs hooks nothing else exposes, not for ownership. M1 established that
as a fact about version 0.5.0 rather than a suspicion, so this tier is confirmed rather than
contingent.

### M20 · `dart:ffi` binding with deterministic execution — `XL`

- Backend pinning, reproducible runs, activation taps.
- Native build configuration is the real cost here, not the FFI surface.

### M21 · Backends — XNNPACK and Core ML — `L`

- The two that cover most devices and can be exercised without exotic hardware.

### M22 · Migration off the interim dependency — `M`

- A deprecation window rather than a hard cut, so early adopters are not stranded.

## Tier 6 — Backend breadth

### M23 · Backends — QNN, Vulkan, Metal and MLX — `XL`

- Each with capability reporting and graceful degradation when unavailable.

### M24 · Parity matrix across backends — `L`

- The same goldens on every available backend, so a model that is correct on XNNPACK and wrong on
  Core ML shows up as exactly that.

### M25 · Real-device CI — `L`

- QNN needs a Snapdragon, Core ML needs Apple silicon. This is the cost item that decides whether
  multi-backend support is verified or asserted.

## Tier 7 — Runtime independence

### M26 · LiteRT backend — `L`

- Proves the runtime-agnostic claim, and extends the audience past PyTorch users.

### M27 · ONNX Runtime backend — `L`

- Second proof of the same claim, and the conversion target many teams already ship.

## Tier 8 — Documentation and example

### M28 · Documentation site — `M`

- Concepts, quickstart, the tolerance guide, and an honest page on what drift does and does not mean.

### M29 · VoltaCast example, end to end — `L`

- A real energy-demand model trained in PyTorch, exported, running on device, with the parity gate
  proving the phone's forecast matches the notebook.
- A flagship built on a toy model convinces nobody, which is why this one is not a toy.

### M30 · Error taxonomy and diagnostics — `M`

- Export failure, hash mismatch, unsupported backend, unsupported dtype and genuine drift are five
  different problems and must not share one message.

## Tier 9 — Stabilisation

### M31 · API freeze and deprecation policy — `M`

- Mirror ExecuTorch's own policy: a deprecated API survives at least two minor releases.
- `STABILITY.md` lands here.

### M32 · Performance and memory benchmarks — `M`

- Codegen overhead, load time, per-run allocation. Published numbers, reproducible commands.

### M33 · `1.0` — `S`

- Frozen public API, documented backend support matrix, published to pub.dev and PyPI.
