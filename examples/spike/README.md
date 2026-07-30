# M2 · End-to-end spike

The smallest thing that proves the pipeline exists: a two-layer model trained in PyTorch, exported to
ExecuTorch, loaded in a Flutter app, and its output compared against the reference the source model
produced before lowering.

Deliberately unabstracted. A spike written through an abstraction only tests the abstraction, and the
point here was to find where the pipeline actually breaks.

## Result

```
PASS  parity/case-0
      backend: xnnpack  quantization: none
      output "y"  max |Δ| 2.98e-8  ok
```

Against `Tolerance.startingPointFor(null)`, whose absolute bound is `1e-5`. The observed drift is
three orders of magnitude inside it, which is what a full-precision export through XNNPACK should
look like: the graph was re-ordered, not re-quantized.

The weight hash also matched, which means the artifact and its manifest came from the same export —
the check that stops a green parity suite from running over a model nobody evaluated.

## Running it

The export half needs `torch` and `executorch`; the load half does not, because the artifact, the
manifest and the golden are committed under `flutter_spike/assets/`.

```bash
python examples/spike/export_spike.py                      # writes examples/spike/build/
cd examples/spike/flutter_spike && flutter test integration_test -d macos
```

macOS is the cheapest target: it needs no device and no simulator. Any other platform ExecuTorch
supports works the same way.

## What it established

**`executorch_flutter` needs macOS 11.0.** A freshly generated Flutter project targets 10.15 and
fails to build against it. `macos/Runner.xcodeproj` here is set to 11.0; any consumer will hit the
same wall, so the eventual backend package has to say so.

**The seam in `FluttorchRuntime` is the right shape.** `loadFromBytes` then `forward` maps onto
`load` and `run` with nothing left over, which is the evidence that the interface was designed
against reality rather than ahead of it.

**The gap M1 identified is real in practice, not just in the API surface.** Nothing in this spike
could pin a backend, read an intermediate activation, or hand `forward` a destination buffer. Final
outputs are all that is reachable, which is exactly what Tier 1 through Tier 3 need — and exactly
what Tier 4 does not.
