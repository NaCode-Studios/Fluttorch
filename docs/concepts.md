# Concepts

Four things, and the arguments for why each is shaped the way it is.

## The manifest

Every export writes three files at once: the artifact, a manifest, and the references the model
produced on inputs the exporter was given. The manifest is the contract. It declares the tensors by
name, type and shape, the hash of the artifact it belongs to, which engine executes it, what recipe
it was quantized with and what precision the delegate lowered to, and any transform training applied
before the model saw a number.

**One document, two readers, byte for byte.** A Python writer and a Dart reader produce the same
bytes from the same values, down to denormals and negative zero. This is checked rather than
intended, because two implementations of one format is the usual way a format quietly becomes two.

**Absence means something.** A manifest that records no quantization means full precision. One that
records no layout is not one that says NCHW: it says nothing, and a consumer that needs to know which
axes are spatial refuses instead of picking. Reading absence as a default is how a resize ends up
running down the wrong axes and returning a picture.

**The hash is over the artifact.** Not over the weights as PyTorch saw them, and not over a
serialisation of the graph: over the bytes that were written. That makes a stale pairing detectable,
which matters more than it sounds, because a parity suite run against the wrong weights is green and
green about a model nobody is shipping.

## The typed API

`fluttorch_gen` turns the manifest into Dart. Each tensor becomes its own type, so passing the model
its inputs in the wrong order is a compile error rather than a shape mismatch discovered inside a
delegate. On a model with one input that looks like ceremony. On a model with two inputs of the same
element type and different extents, it is the difference between a wrong answer and a red build.

The generated code is not written by hand and says so at the top. Everything in it comes from the
manifest, and nothing the manifest does not describe is emitted, because a generator that guessed
would be a second source of truth about the model.

**It refuses rather than approximating.** A preprocessing step this build does not understand, a
resize filter it does not implement, a spatial step recorded after an elementwise one: each is
refused with the reason, and every reason at once rather than the first one found. The refusals are
the feature. A generator that skipped what it did not understand would emit an API missing a
transform the model was trained with, and that skew is invisible at every later stage.

## The gate

The references captured at export are replayed against the model as it actually runs, and the
difference is measured. That is the whole project in one sentence, and everything else exists to make
that sentence checkable.

What makes it useful rather than decorative is that the bound is derived rather than chosen. A model
quantized to `int8-dynamic` is held to a different bound from a full-precision one, and a delegate
running at float16 is held to a different bound again, and the two compound because an int8 model on
a half-precision GPU is wrong in both ways at once. See [tolerances](tolerance.md).

Where the export captured activation taps, the report also names the earliest layer whose numbers
moved, which is the difference between knowing a model drifted and knowing where.

## The seam

Below the Dart API is one C ABI, and three runtimes implement it: ExecuTorch, ONNX Runtime and
LiteRT. What a second and a third runtime change is the engine. The manifest, the references and the
gate are unchanged, which is what turns "runtime agnostic" from a claim into something measured.

The manifest records which engine executes an artifact, and the wrong runtime refuses to load it. A
`.pte`, a `.onnx` and a `.tflite` are not interchangeable, and the weight hash cannot tell them
apart, because it is computed over whichever one was written.
