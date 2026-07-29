# Fluttorch roadmap

Where Fluttorch is going, milestone by milestone. Released versions and dates live in
[`CHANGELOG.md`](CHANGELOG.md); the short human summary lives in the
[README Roadmap section](README.md#roadmap). This file follows
[`ROADMAP-CONVENTIONS.md`](ROADMAP-CONVENTIONS.md), the standard shared with Kdrant and Kmemo.

Fluttorch is pre-`1.0` and has no released version yet. Until `1.0`, minor versions may break the
public API; a `STABILITY.md` arrives with the `1.0` line.

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
| M1 · Audit `executorch_flutter` | Planned. |
| M2 · End-to-end spike | Planned. |
| M3 · Manifest schema v1 | Planned. |
| M4 · Tolerance semantics | Planned. |
| M5 · Export CLI | Planned. |
| M6 · Signed manifest with weight hash | Planned. |
| M7 · Golden capture from the source model | Planned. |
| M8 · Manifest round-trip tests | Planned. |
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
| M19 · Decision — fork the runtime binding | Planned. |
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

## Tier 0 — Foundations

Decide the things that are expensive to change later, and prove the pipeline exists before
abstracting it.

### M1 · Audit `executorch_flutter` — `S`

- Establish whether it exposes intermediate activations, deterministic execution, or backend pinning.
  Those three decide whether the parity gate can sit on top of it at all.
- Read the open issues and any stated direction, so the phase-one dependency is a choice rather than
  an assumption.
- Output is a written decision, not an impression.

### M2 · End-to-end spike — `M`

- Two-layer model, exported, loaded in a Flutter app, one output compared against the Python
  reference by hand.
- No abstraction, no codegen. The point is to find where the pipeline actually breaks.

### M3 · Manifest schema v1 — `M`

- Shapes, dtypes, preprocessing steps, labels, weight hash, golden index. Versioned from the first
  release so old artifacts stay readable.
- Written as a JSON schema with a Python writer and a Dart reader generated or hand-checked against it.

### M4 · Tolerance semantics — `S`

- Absolute, relative and cosine bounds, and which of them a default carries per quantization recipe.
- Cosine exists because small magnitudes let an elementwise bound pass while the tensor points
  somewhere else entirely; the defaults must reflect that.
- Getting this wrong makes the gate either useless or permanently red, so it precedes the gate.

## Tier 1 — The export contract

One command produces the artifact, the manifest and the goldens together, so they cannot be
generated from different states of the model.

### M5 · Export CLI — `L`

- `torch.export` to a runtime artifact. XNNPACK and full precision only.
- Model and example inputs addressed as `module:factory`, so the CLI never imports a notebook.
- Reproducible output for a fixed input, because everything downstream compares against it.

### M6 · Signed manifest with weight hash — `M`

- Content hash of the exported weights recorded in the manifest, and the runtime refuses to load an
  artifact whose hash disagrees.
- This is what stops a stale model from silently passing a green parity suite.

### M7 · Golden capture from the source model — `M`

- Reference pairs taken from the model *before* lowering, so they are a reference rather than a
  snapshot of whatever the export happened to produce.
- Count configurable; the default is small enough to commit.

### M8 · Manifest round-trip tests — `S`

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

### M19 · Decision — fork the runtime binding — `S`

- By this point the required hooks are known precisely. Decide with evidence whether to write an own
  `dart:ffi` binding or keep the dependency, and record the argument either way.

## Tier 5 — Own runtime

Undertaken because the gate needs hooks nothing else exposes, not for ownership.

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
