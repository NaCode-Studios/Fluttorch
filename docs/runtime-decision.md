# M19 · Whether to write our own runtime binding

Decided on 2026-08-01, at the close of Tier 4.

**The four hooks are proposed to `executorch_flutter` first. If they are not merged by the time
Tier 4 closes and Tier 5 would otherwise begin, Fluttorch writes its own `dart:ffi` binding and the
question is not reopened.**

## What is missing, and what each thing blocks

M1 audited `executorch_flutter` 0.5.0 and found four hooks absent from its public API. Two tiers of
work later, the list has not changed and each entry now has a milestone behind it rather than a
suspicion.

| Hook | What it blocks | State |
| --- | --- | --- |
| Activation taps | M18. The attribution walks the tapped layers and names the earliest that moved; without taps it reports that it could not look. | Built and tested against a fake that reports taps. No real backend reaches it. |
| Deterministic execution | The tolerance floor. Without it a bound has to absorb run-to-run noise, which raises the smallest drift the gate can detect. | Declared as a capability, reported by nothing. |
| Backend selection at load | M24, the parity matrix. A model correct on XNNPACK and wrong on Core ML is the finding, and it needs both to be selectable. | `FluttorchRuntime.load` takes a `backend` argument that no implementation honours. |
| Caller-supplied output buffers | `runInto`, the path that makes repeated inference affordable. A 224x224x3 float32 input is 602 KB, and allocating it twice per frame is the difference between usable and not. | Interface only. |

## Why upstream first

The work is the same work. `executorch_flutter` is already a `dart:ffi` binding over the same C++
API our own binding would call, so writing these hooks into it is writing the code we would write
anyway, in somebody else's repository. Accepted, Tier 5 disappears. Rejected, the patches are the
starting point of the fork and nothing is thrown away. Forking first pays the full 204 hours to
avoid asking a question that costs days.

The package is alive rather than abandoned, which is what makes the question worth asking: 0.5.0 was
published on 2026-07-25, nineteen versions exist, and it is maintained in the open at
[abdelaziz-mahdy/executorch_flutter](https://github.com/abdelaziz-mahdy/executorch_flutter). A dead
dependency would have settled this without a discussion.

## Why the deadline is not negotiable

The risk is not rejection, it is silence. An open pull request against a single-maintainer package
puts this project's schedule inside someone else's week, and a roadmap that waits indefinitely is a
roadmap that has stopped being one. Tier 5 is where the binding would be written, so that is where
the option expires: if the hooks are not merged when Tier 4 closes, the fork starts on schedule with
the patches already in hand.

## Where it probably breaks

Three of the four are mechanical. Backend selection, output buffers and a determinism flag are
arguments threaded through an FFI layer that already calls the functions underneath them.

Activation taps are not. ExecuTorch exposes intermediates through ETDump, which is a different
execution mode carrying developer-tooling machinery, and a maintainer keeping an inference package
lean has a legitimate reason to refuse it. That is the likely point of failure, and it is worth
naming in advance: if the taps are refused while the other three land, the fork is motivated by one
capability rather than by four, and it becomes a smaller and better argued piece of work than it
looks today.

## What was rejected

**Keeping the dependency and dropping the capabilities.** Cheapest, and it would leave per-layer
attribution as code no backend can exercise, the parity matrix with no way to choose a backend, and
`runInto` permanently unimplementable. Three milestones would be quietly removed from the plan
rather than delivered, and the runtime-agnostic claim would stay a claim.

**Forking immediately.** Full control, at 204 hours and the perpetual maintenance of a native
binding across platforms, to avoid an experiment that costs days and whose output is reusable either
way.
