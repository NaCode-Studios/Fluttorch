# What each backend is verified to do

A backend claim has two halves that fail independently. The export half is whether this toolchain
can lower a model for that delegate, and it is answerable on any machine with the right packages
installed. The runtime half is whether the resulting artifact produces the right numbers on the
hardware, and it is only answerable on that hardware.

This page records which half is verified where. It exists because the alternative is a README
listing eight backends, which reads as eight backends that work.

## The two questions, and where each is answered

The export half is checked by the `Backends` workflow, on a Linux runner. `available_backends()`
answers by lowering a one-operation model rather than by importing a partitioner, so the list is
what worked rather than what imported.

That runner reports `portable`, `xnnpack`, `coreml`, `mps` and `vulkan` as available, and refuses
`metal`, `mlx` and `qnn` with the toolchain each would need. Core ML and MPS being on the first list
is the clearest illustration of why this page splits the question in two: coremltools installs on
Linux and lowers there quite happily, and the artifact it produces cannot run on that machine under
any circumstances. Lowering for a backend and running on it are separate claims, and only one of
them is answerable off the hardware.

The runtime half is checked by the `On device` workflow, on an Apple silicon runner, weekly and on
demand rather than on every pull request. Building ExecuTorch is the better part of an hour, and a
gate that shape on every push is one somebody eventually routes around. Everywhere else the
on-device suite skips by name, which is what keeps a green pull request honest about what it did not
cover.

## Where the numbers come from

`dart run tool/parity_matrix.dart` prints one report: the same goldens replayed on every backend the
binding was linked with, each measured at the tolerance its own export implies. It lists the
backends it could not run and why, rather than leaving them out of the table. A column that is
absent and a column that agreed look identical once the table is printed, and they are different
claims.

On an M-series Mac with Core ML, MPS and XNNPACK linked, that report covers four backends across
four goldens. Three carry the model at float32 and agree with the source to within `1e-7` relative.
The fourth carries it at `int8-dynamic` and moves by up to `4e-2`, which is four orders of magnitude
more and is the point: a matrix whose columns all read the same would mean the quantized artifact
was not quantized.

## What is not run, and why

Metal needs torchao built with `TORCHAO_BUILD_EXPERIMENTAL_MPS=1`, which provides a dylib its own
error message names without naming the flag. QNN needs `py-cpuinfo` and Qualcomm's SDK, which
ExecuTorch does not vendor and which has no macOS build, so neither half of that claim is checked
anywhere in this repository today.

Core ML lowers and runs, but the `On device` workflow does not link its runtime: doing so needs
coremltools built and its generated protobuf sources copied into the ExecuTorch checkout, which the
header of `tool/build_native.sh` describes in full. The matrix reports Core ML as not run there and
covers it on a developer machine instead.

Vulkan and MLX lower. Neither has been run, because neither runtime is built by the configure the
on-device workflow uses.

## The rule this page follows

A backend appears in the binding's own list of backends only once it is both linked and registered,
so what a caller reads at runtime is what will load. This page is the same rule written for a
reader: nothing is listed as working on evidence weaker than a test that ran.
