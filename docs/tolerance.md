# Tolerances

A bound is not a constant somebody picked. It is derived from what the manifest says was done to the
model, and this page is the argument for each number.

## The form

`|produced - reference| <= atol + rtol * |reference|`, per element, which is numpy's `allclose` and
is used here for the same reason: neither half works alone.

A pure relative bound is meaningless wherever the reference crosses zero, and a normalize step that
subtracts a mean guarantees it will. A pure absolute bound is sized by the largest value in the
tensor, so it excuses everything small.

## Where the number comes from

Two axes, and they compound.

**The recipe** says how the weights were stored. A full-precision model should agree with its
reference to within float32 rounding. An `int8-dynamic` model should not, and holding it to the first
bound would fail a model doing exactly what it was told.

**The precision** says what the delegate does arithmetic in, which is a different question. Core ML
and MPS run at float16 by default, so an artifact lowered for either answers to a different bound
even when nothing was quantized. Without this recorded, the gate failed models that were correct.

They compound because an int8 model on a half-precision GPU is wrong in both ways at once, so
`startingPointFor` takes the larger of the two contributions per axis rather than the first one it
finds.

## Why float16's bound is not its epsilon

Float16's machine epsilon is `4.9e-4`, and a bound of a small multiple of that is the number that
looks right. It is wrong, and the reason is worth stating because it generalises.

A gate sees outputs. Rounding happens on intermediates. Feed this project's own model an input of
magnitude `1e3` and its intermediate activations sit around `1e3` while its outputs come out near
`9.4`, because the last layer subtracts two numbers of similar size. The absolute error accumulated
at `1e3` survives the cancellation; the magnitude it is measured against does not. Divide one by the
other and the relative error at the output is two orders larger than epsilon at the intermediate.

So float16 starts at `2e-2` relative. That is not slack, it is the arithmetic.

## Starting points, not measurements

Every number the table currently carries is a starting point: chosen to be defensible and wide enough
not to fail a correct model, rather than measured across a population of models and backends.

This is stated rather than hidden because a tolerance is the one number in this project that decides
whether a build is green, and a reader deserves to know which of them were measured. Replacing them
with measured values is
[issue 60](https://github.com/NaCode-Studios/Fluttorch/issues/60), and it is open.

## Overriding one

You can pass your own, and the gate will use it:

```dart
await expectParity(model, goldens: goldens, tolerance: myTolerance);
```

Doing that is sometimes right and is worth being suspicious of. A bound loosened until the build goes
green is a bound that measures nothing, and the failure it was hiding is still there. If a model
needs a wider bound than its recipe implies, the interesting question is why, and the report names
the tensor and the backend so that question has somewhere to start.

## Reading a report

A report says which tensor moved, by how much, against which bound, and on which backend. It says
that whether the answer is pass or fail, which is deliberate: a run that passed by a hair and one
that passed comfortably are different situations, and a boolean cannot tell them apart.

Where the export captured activation taps, the report also names the earliest layer whose numbers
moved. A drift at the output tells you the model is wrong. A drift attributed to the third layer
tells you where to look.
