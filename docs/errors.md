# What went wrong, and what to do about it

Five things can go wrong between a model in PyTorch and a number in a Flutter app. They have five
different causes and five different fixes, and the reason this page exists is that four of them look
identical from inside the app: the model did not run.

Four are exceptions. The fifth is not, and that is the most important thing on this page.

## The four exceptions

Every one extends `FluttorchException`, which is `sealed`, so a `switch` over them is exhaustive and
adding one to this library is a compile error in code that handles them rather than a surprise at run
time. Every one carries a `remedy` alongside its `message`: the cause and the fix are different
sentences, and a caller showing a failure to a user usually wants only the second.

### The export did not produce a bundle

Raised in Python, by `fluttorch_export`, before anything reaches Dart. `ExportError` means the model
could not be lowered, `QuantizationError` that a recipe could not be applied, and `ManifestError`
that the manifest would have been inconsistent with the artifact beside it.

These are deliberately not Dart's problem. An export that fails writes nothing, so there is no bundle
to load and no state for the app to be confused by. The failure that would be Dart's problem is an
export that half-succeeded, and the exporter refuses rather than allowing one: this is why an ONNX
export whose weights moved into a sidecar is refused instead of written.

### The artifact and the manifest come from different exports

`ArtifactMismatchException`. The manifest declares a weight hash and the artifact handed alongside it
hashes to something else.

Almost always a stale build: the model was re-exported and the manifest was not, or the reverse. It
matters more than it looks, because the failure it prevents is silent. A parity suite run against the
wrong weights is green, and it is green about a model nobody is shipping.

Re-export both together. Editing either to agree with the other hides the mismatch.

### The backend cannot do it

Two exceptions, because two different things are absent.

`BackendUnavailableException` means the backend is not there at all. It carries the list of backends
that *are*, because choosing from that list is the useful next step. An empty list is its own signal:
a device offering no backend usually means the native library never loaded.

`CapabilityUnavailableException` means the backend loaded and runs and cannot do this particular
thing. Activation taps on a delegate that executes nothing itself is the case that comes up. A
backend without a capability is not a broken backend, and the fix is to ask the model what it can do
and degrade rather than assuming.

### The element type is wrong, in one of two ways

`DTypeMismatchException` is a caller reading bytes as a type they do not hold. Nothing is broken
except the read: the fix is to read the tensor as the type the export wrote.

`DTypeUnsupportedException` is the backend having no kernel for a type the manifest declares.
Nothing is misread at all. The manifest and the buffer agree, and the device cannot carry them.

These were one class until it was noticed that the runtimes reported the second case with the first
one's message, telling a reader their tensor "holds float16 and was read as float32" when nothing had
read anything. That message sends someone to look at code that is correct, which is worse than no
message.

### The runtime failed, and the bundle is fine

`RuntimeExecutionException`. The manifest parsed, the artifact matched its hash, the model loaded, and
the engine failed while running it. It carries the shim's status and the runtime's own error code as
numbers, because those are the two values that mean something to somebody reading that runtime's
source, and it names which of the three engines it was.

This was reported as `ManifestFormatException` until VoltaCast made it obvious: a provably correct
bundle produced a failure whose remedy was to re-export the document. That advice rebuilds something
that was never wrong and leaves the actual cause unexamined.

## The fifth is not an exception

**Drift is a measurement, not an error.**

A quantized model that disagrees with its reference by two percent has not failed. That is what
quantization does, and a binding that threw on it would be a binding that cannot ship a quantized
model. A model that disagrees by two percent when its recipe implies one part in a million has
failed, and the difference between those two sentences is a number, not a category.

So the gate returns a `ParityReport`. It says which tensor moved, by how much, against which bound,
and on which backend, and it says that whether the answer is pass or fail. `measureParity` gives you
the report. `expectParity` fails the test with it. `measureMatrix` produces one per backend a machine
offers.

Making drift an exception would collapse this. There would be no way to look at a passing run and see
that it passed by a hair, and no way to compare two backends that both passed. The report exists so a
number can be read rather than only reacted to, and where the export captured activation taps it also
names the earliest layer whose numbers moved, which is the difference between knowing a model drifted
and knowing where.

## Deciding which one you have

| The symptom | The failure |
| --- | --- |
| Nothing was written; the export command failed | `ExportError`, `QuantizationError`, `ManifestError`, in Python |
| Loading throws immediately, mentioning a hash | The bundle is stale. Re-export both halves |
| Loading throws naming a backend | That backend is absent. Read the list it carries |
| A call throws naming a capability | The backend runs and cannot do that one thing |
| Reading a tensor throws about a type | Either the read is wrong, or the device cannot carry it. The two exceptions say which |
| It loaded and then failed while running | The engine, not the bundle. Take the status and error code to that runtime |
| It ran, and the numbers are not the ones from the notebook | Not an exception. Read the `ParityReport` |
