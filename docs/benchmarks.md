# Benchmarks

Three costs, because three different claims get made about this library and each has its own way of
being wrong. Generating the typed API is a build-time cost paid once per manifest. Loading is paid
once per model per process and is what a user feels as a stall on a cold start. Running is paid on
every inference.

Reproduce with:

```sh
cd packages/fluttorch_executorch
dart run tool/benchmark.dart
```

The numbers below are one machine's. macOS 26.6, 12 cores, Dart 3.12.2, 1000 iterations after 50
warmup, median and 95th percentile. Never the mean: the distribution has a long tail because the
machine has a scheduler, and a mean over it describes a run that did not happen.

## Codegen

| Model | Inputs | Outputs | Median | p95 |
| --- | --- | --- | --- | --- |
| `two_layer` | 1 | 1 | 1.01 ms | 1.84 ms |
| `matrix` | 1 | 1 | 0.83 ms | 1.21 ms |
| `voltacast` | 2 | 1 | 0.94 ms | 1.29 ms |

About a millisecond per manifest, and flat across models that differ by three orders of magnitude in
weight size. That is expected rather than impressive: the emitter reads specs and writes text, and
never touches the artifact.

Measured on the manifest rather than through `build_runner`, whose own startup dominates this by two
orders of magnitude. What these numbers support is only that the emitter is not the reason a build is
slow. If a build feels slow, the cost is in the builder framework and not here.

## Load

| Model | Backend | Artifact | Median | p95 |
| --- | --- | --- | --- | --- |
| `two_layer` | portable | 2.4 kB | 61 us | 143 us |
| `matrix` | portable | 8.6 kB | 84 us | 132 us |
| `matrix` | xnnpack | 9.5 kB | 100 us | 203 us |
| `voltacast` | portable | 3.4 MB | 22.04 ms | 22.62 ms |

Load tracks artifact size and nothing else visible here. A model measured in kilobytes loads in tens
of microseconds; the 3.4 MB one takes 22 ms, which is about 6.5 ms per megabyte.

Twenty-two milliseconds is once per process, not once per screen. It is worth knowing before deciding
whether to load a model eagerly at startup or on the first use of the feature that needs it, and this
library does not decide that for you.

The load includes verifying the weight hash over the whole bundle, so a large artifact pays for its
own integrity check here rather than skipping it.

## Run

`run` allocates an output tensor per call. `runInto` writes into buffers the caller owns and keeps.
The gap between them is what that allocation costs.

| Model | Backend | `run` | `runInto` | p95 `run` | p95 `runInto` | Saved |
| --- | --- | --- | --- | --- | --- | --- |
| `two_layer` | portable | 6 us | 2 us | 9 us | 3 us | 4 us |
| `matrix` | portable | 251 us | 244 us | 324 us | 275 us | 7 us |
| `matrix` | xnnpack | 116 us | 112 us | 157 us | 153 us | 4 us |

The saving is roughly constant at four to seven microseconds and does not scale with the model. That
inverts the advice you might expect: `runInto` is worth reaching for on small models called at frame
rate, where it is three times faster, and is close to irrelevant on a convolutional model where the
same four microseconds is under two per cent of the call.

`voltacast` is absent from this table rather than reported as a blank. ExecuTorch lowers it and then
fails to execute it, identically under its own Python runtime, which the on-device suite records as a
failing expectation so that the day upstream fixes it, the suite says so.

Resident memory was measured and is not published. Dart exposes no allocation count, and the RSS delta
over a thousand runs came back negative as often as positive: it measures when the collector happened
to run rather than what the loop allocated. A column of noise is worse than a missing one, because a
reader cannot tell which they are looking at.
