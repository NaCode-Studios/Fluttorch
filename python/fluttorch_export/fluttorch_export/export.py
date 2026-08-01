"""M5 · the export pipeline.

One command produces the artifact, the manifest and the goldens together, so the
three cannot come from different states of the model. That is the whole reason
this is a single entry point rather than three scripts: a manifest paired with
weights it was not written for produces a green parity suite over a model nobody
evaluated, and the weight hash only catches that if both are written in the same
breath.

Nothing here infers a value the export did not observe. Output specs come from
running the source model, not from guessing at the graph; golden inputs come from
the caller, because a distribution the exporter invented would make the goldens a
description of `torch.randn` rather than of the model's job.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import importlib
import pathlib
from collections.abc import Callable, Iterable, Sequence
from typing import Any

import torch

from .manifest import (
    DYNAMIC_DIM,
    GoldenCase,
    ManifestError,
    ModelManifest,
    PreprocessingStep,
    TensorSpec,
)
from .quantization import Recipe, check_calibration, quantizer_for, recipe_for

#: torch dtype to the manifest's wire name. Deliberately not exhaustive over
#: torch's dtypes: a type absent here is one the contract cannot describe, and
#: failing on it is better than writing a name the Dart reader will reject.
_TORCH_DTYPES: dict[torch.dtype, str] = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
    torch.uint8: "uint8",
    torch.bool: "bool",
}


class ExportError(RuntimeError):
    """The export could not be completed, with the reason a caller can act on."""


@dataclasses.dataclass(frozen=True, slots=True)
class ExportResult:
    """What one export produced, and where it was written."""

    artifact: pathlib.Path
    manifest_path: pathlib.Path
    golden_dir: pathlib.Path
    manifest: ModelManifest

    @property
    def golden_count(self) -> int:
        return len(self.manifest.goldens)


def resolve(spec: str) -> Any:
    """Resolve a ``module:attribute`` reference and call it if it is callable.

    Addressing the model this way keeps the exporter out of the business of
    importing a notebook: the caller names something importable, and what comes
    back is whatever that name produces.
    """
    if ":" not in spec:
        raise ExportError(
            f"{spec!r} is not a module:attribute reference — write it as mypkg.models:build_model"
        )
    module_name, _, attr = spec.partition(":")
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ExportError(f"cannot import {module_name!r}: {e}") from e
    try:
        obj = getattr(module, attr)
    except AttributeError as e:
        raise ExportError(f"{module_name!r} has no attribute {attr!r}") from e
    return obj() if callable(obj) else obj


def _as_tuple(x: Any) -> tuple[torch.Tensor, ...]:
    if isinstance(x, torch.Tensor):
        return (x,)
    if isinstance(x, (list, tuple)):
        return tuple(x)
    raise ExportError(f"expected a tensor or a sequence of tensors, got {type(x).__name__}")


def _wire_dtype(t: torch.Tensor, role: str, index: int) -> str:
    try:
        return _TORCH_DTYPES[t.dtype]
    except KeyError:
        raise ExportError(
            f"{role}[{index}] has dtype {t.dtype}, which the manifest cannot "
            f"describe; supported: {', '.join(sorted(set(_TORCH_DTYPES.values())))}"
        ) from None


def _specs(
    tensors: Sequence[torch.Tensor],
    role: str,
    names: Sequence[str] | None,
    dynamic_batch: bool,
) -> tuple[TensorSpec, ...]:
    out = []
    for i, t in enumerate(tensors):
        shape = list(t.shape)
        # A batch dimension the caller declared dynamic is the one thing the
        # exporter marks: everything else is what the traced graph fixed.
        if dynamic_batch and shape:
            shape[0] = DYNAMIC_DIM
        name = names[i] if names and i < len(names) else f"{role}_{i}"
        out.append(TensorSpec(name, _wire_dtype(t, role, i), tuple(shape)))
    return tuple(out)


#: Every backend this exporter knows, in the order a caller meets them: the one
#: that delegates nothing, the two that run anywhere the toolchain is installed,
#: then the ones that need an accelerator or a vendor SDK.
BACKENDS: tuple[str, ...] = (
    "portable",
    "xnnpack",
    "coreml",
    "mps",
    "metal",
    "vulkan",
    "mlx",
    "qnn",
)

#: What a machine has to have before a backend can lower, keyed by backend.
#:
#: Written down because upstream does not say. A missing Qualcomm SDK surfaces as
#: an ImportError about a Python package nobody asked for, and Metal surfaces as a
#: dylib filename with no hint of the flag that builds it. Neither tells a caller
#: what to do, and both cost an afternoon the first time.
_TOOLCHAINS: dict[str, str] = {
    "mps": "the MPS delegate, which needs macOS and a Metal-capable device",
    "metal": (
        "torchao built with TORCHAO_BUILD_EXPERIMENTAL_MPS=1, which is what "
        "provides libtorchao_ops_mps_aten.dylib"
    ),
    "vulkan": "the Vulkan delegate and a Vulkan driver, which on macOS means MoltenVK",
    "mlx": "the MLX delegate, which needs Apple silicon",
    "qnn": (
        "py-cpuinfo and Qualcomm's QNN SDK, which executorch does not vendor and "
        "which has no macOS build"
    ),
}


def _mps_partitioner() -> Any:
    from executorch.backends.apple.mps.partition.mps_partitioner import MPSPartitioner
    from executorch.exir.backend.compile_spec_schema import CompileSpec

    # Float32, for the reason Core ML is pinned to it: the manifest has no field
    # that records a precision, so an artifact lowered at half answers to a bound
    # it cannot hold. See the note on the Core ML branch.
    return MPSPartitioner(compile_specs=[CompileSpec("use_fp16", bytes([False]))])


def _metal_partitioner() -> Any:
    from executorch.backends.apple.metal.metal_backend import MetalBackend
    from executorch.backends.apple.metal.metal_partitioner import MetalPartitioner

    return MetalPartitioner([MetalBackend.generate_method_name_compile_spec("forward")])


def _vulkan_partitioner() -> Any:
    from executorch.backends.vulkan.partitioner.vulkan_partitioner import (
        VulkanPartitioner,
    )

    return VulkanPartitioner()


def _mlx_partitioner() -> Any:
    from executorch.backends.mlx.partitioner import MLXPartitioner

    return MLXPartitioner()


def _qnn_partitioner() -> Any:
    from executorch.backends.qualcomm.partition.qnn_partitioner import QnnPartitioner

    return QnnPartitioner()


_PARTITIONERS: dict[str, Any] = {
    "mps": _mps_partitioner,
    "metal": _metal_partitioner,
    "vulkan": _vulkan_partitioner,
    "mlx": _mlx_partitioner,
    "qnn": _qnn_partitioner,
}


def _build_partitioner(backend: str) -> Any:
    """Construct a delegate's partitioner, or say what is missing.

    Degrading rather than failing is the point: a backend this machine cannot
    lower for is a fact about the machine, and a caller who is told which piece
    is absent can install it or pick another backend. An ImportError from three
    libraries down tells them neither.
    """
    try:
        return _PARTITIONERS[backend]()
    except Exception as e:  # noqa: BLE001 - re-raised with what to do about it
        raise ExportError(_missing_toolchain(backend, e)) from e


def available_backends() -> tuple[str, ...]:
    """The backends this machine can actually lower for, asked rather than assumed.

    Whether a delegate works here is a property of the installed toolchain, not
    of this package: the same checkout lowers for MLX on Apple silicon and
    refuses it everywhere else.

    Answered by lowering a one-operation model, which is slower than importing a
    partitioner and is the only thing that is true. Metal is the reason. Its
    partitioner constructs on any Mac and then fails during preprocessing,
    because what it actually needs is a torchao dylib nobody built. A list based
    on imports would name it available and be wrong exactly where it mattered.
    """
    return tuple(name for name in BACKENDS if _can_lower(name))


@functools.cache
def _can_lower(backend: str) -> bool:
    """Whether a one-operation model lowers for ``backend``, cached per process.

    Cached because the answer cannot change while the process runs and asking is
    expensive: it invokes the whole delegate.
    """

    class _OneOp(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + x

    try:
        _lower(_OneOp().eval(), (torch.ones(1, 2),), backend)
    except Exception:  # noqa: BLE001 - the question is only whether it worked
        return False
    return True


def _lower(
    model: torch.nn.Module,
    example: tuple[torch.Tensor, ...],
    backend: str,
    recipe: Recipe | None = None,
    calibration: Sequence[tuple[torch.Tensor, ...]] = (),
) -> tuple[bytes, dict[str, int]]:
    """Trace, quantize if a recipe was named, lower and serialise.

    Returns the artifact and, for each submodule the lowered graph still executes
    as ops, the debug handle of the last one it runs. That last op is what the
    submodule emits, which is the same tensor a forward hook would have caught,
    so the two sides of a tap describe one value.

    The map is empty for a delegated backend, and that is the honest answer
    rather than a gap: a delegate is one instruction, and nothing inside it is
    addressable from the runtime.
    """
    from executorch.exir import to_edge_transform_and_lower

    if backend == "portable":
        # No partitioner, so every op stays in the runtime's own kernels. Slower
        # than any delegate and not what a device would ship, which is the point:
        # this is the export you run to find out which layer moved the numbers.
        partitioner = None
    elif backend == "xnnpack":
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
            XnnpackPartitioner,
        )

        partitioner = XnnpackPartitioner()
    elif backend == "coreml":
        import coremltools as ct
        from executorch.backends.apple.coreml.compiler import CoreMLBackend
        from executorch.backends.apple.coreml.partition import CoreMLPartitioner

        # An artifact is lowered for one delegate, so which backend runs it is
        # decided here and not on the device. Recording the choice in the
        # manifest is what lets the parity matrix say which backend a number
        # came from rather than assuming.
        #
        # Float32 is pinned rather than accepted. Core ML converts to float16 by
        # default, and the manifest has no field that records it, so an artifact
        # lowered on that default answers to a full-precision bound it cannot
        # hold: on this two-layer model that reads as drift up to 9.7e-2 against
        # references the source model produced. Until the manifest can describe
        # the precision, the export stays at the one the manifest already claims
        # by saying nothing.
        partitioner = CoreMLPartitioner(
            compile_specs=CoreMLBackend.generate_compile_specs(
                compute_precision=ct.precision.FLOAT32,
            )
        )
    elif backend in _PARTITIONERS:
        partitioner = _build_partitioner(backend)
    else:
        raise ExportError(
            f"backend {backend!r} is not one this exporter knows; it lowers for "
            f"{', '.join(repr(b) for b in BACKENDS)}"
        )

    exported = torch.export.export(model, example)

    if recipe is not None:
        # torchao rather than torch.ao: the pt2e flow moved out of torch and into
        # torchao, which executorch depends on and therefore installs. Importing
        # it lazily keeps the recipe table readable without either.
        from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

        graph = prepare_pt2e(exported.module(), quantizer_for(recipe))
        # Observers only see what they are run over. A static recipe fixes every
        # activation range here, from these inputs and nothing else, which is why
        # an export with nothing representative to calibrate on is refused
        # earlier rather than quantized against its own example input.
        with torch.no_grad():
            for case in calibration:
                graph(*case)
        try:
            converted = convert_pt2e(graph)
        except Exception as e:  # noqa: BLE001 - re-raised with the reason below
            raise ExportError(_convert_failure(recipe, e)) from e
        exported = torch.export.export(converted, example)

    try:
        lowered = to_edge_transform_and_lower(
            exported, partitioner=None if partitioner is None else [partitioner]
        ).to_executorch()
    except Exception as e:  # noqa: BLE001 - re-raised with what to do about it
        # A delegate can construct and still not work. Metal's partitioner takes
        # any Mac and then fails here, looking for a torchao dylib whose build
        # flag it does not name, so the check that matters is this one and the
        # message has to come from the same place.
        if backend not in _TOOLCHAINS:
            raise
        raise ExportError(_missing_toolchain(backend, e)) from e

    return bytes(lowered.buffer), _handles_by_module(lowered)


def _missing_toolchain(backend: str, error: Exception) -> str:
    """Say which piece is absent, and leave the original reason attached.

    Degrading rather than failing is what this milestone is about: a backend this
    machine cannot lower for is a fact about the machine, and a caller told which
    piece is missing can install it or pick another backend. An error from three
    libraries down tells them neither.
    """
    return (
        f"this machine cannot lower for {backend!r}: it needs {_TOOLCHAINS[backend]}. "
        f"What it failed on was {type(error).__name__}: {str(error).strip()[:200]}. "
        f"available_backends() reports the ones it can lower for."
    )


def _handles_by_module(lowered: Any) -> dict[str, int]:
    """The debug handle of the last op each submodule still runs, by module path.

    Read off the lowered graph rather than the source model, because lowering is
    what decides whether a layer survives as ops at all. A delegated partition
    collapses to one instruction carrying no module path, so it contributes
    nothing here and the layers inside it are simply not addressable.
    """
    handles: dict[str, int] = {}
    for node in lowered.exported_program().graph_module.graph.nodes:
        handle = node.meta.get("debug_handle")
        if handle is None:
            continue
        stack = node.meta.get("nn_module_stack") or {}
        paths = [entry[0] for entry in stack.values() if entry and entry[0]]
        if not paths:
            continue
        # Later nodes overwrite earlier ones, so what survives is the last op the
        # submodule runs, which is the tensor it hands on.
        handles[paths[-1]] = int(handle)
    return handles


def _convert_failure(recipe: Recipe, error: Exception) -> str:
    """Say what a failure inside torchao's conversion actually was.

    The pass manager reports which pass raised and swallows the reason, so the
    message a caller sees names a pass they have never heard of and nothing they
    can act on. This puts the cause back and, for the one incompatibility this
    toolchain is known to hit, says which combination is at fault so nobody
    spends an afternoon on their model.
    """
    cause = str(error.__cause__ or error)
    if "has no overload name" in cause:
        return (
            f"{recipe.name} cannot be converted by the installed toolchain: "
            f"{cause}. This is torchao introspecting an operator overload that "
            f"this version of torch does not expose, and it happens before the "
            f"model is involved. torch {torch.__version__} with the torchao that "
            "executorch pins hits it on any graph containing a linear layer; "
            "int8-dynamic and int4-weight-only convert on the same toolchain."
        )
    return f"{recipe.name} could not be converted: {cause}"


def export_model(
    *,
    model: torch.nn.Module,
    example_inputs: Any,
    out_dir: pathlib.Path,
    name: str,
    backend: str = "xnnpack",
    golden_inputs: Iterable[Any] | Callable[[], Iterable[Any]] | None = None,
    preprocessing: Sequence[PreprocessingStep] = (),
    labels: Sequence[str] | None = None,
    input_names: Sequence[str] | None = None,
    output_names: Sequence[str] | None = None,
    dynamic_batch: bool = False,
    quantization: str | None = None,
    taps: Sequence[str] | None = None,
) -> ExportResult:
    """Export ``model``, and write the artifact, the manifest and the goldens.

    ``golden_inputs`` yields input tuples to capture references for. When it is
    absent only ``example_inputs`` is captured, which is a smoke test rather than
    coverage — the exporter says so rather than inventing a distribution it has
    no way to know.

    ``quantization`` names a recipe from ``quantization.RECIPES``. A static recipe
    also calibrates on those same golden inputs, which is why they are resolved
    before the model is lowered rather than after.

    ``taps`` names submodules whose outputs are captured alongside each golden, so
    the gate can attribute a drift to the layer that caused it instead of only
    reporting the output that was wrong. Off by default: an intermediate is as
    large as the tensor it carries, and on a real model taps multiply the bundle.
    """
    model = model.eval()
    example = _as_tuple(example_inputs)

    with torch.no_grad():
        reference = _as_tuple(model(*example))

    inputs = _specs(example, "input", input_names, dynamic_batch)
    outputs = _specs(reference, "output", output_names, dynamic_batch)

    cases_in = _golden_inputs(golden_inputs, example, inputs)

    recipe = recipe_for(quantization) if quantization else None
    if recipe is not None:
        check_calibration(recipe, cases_in)

    buffer, graph_handles = _lower(model, example, backend, recipe=recipe, calibration=cases_in)

    out_dir = pathlib.Path(out_dir)
    golden_dir = out_dir / "goldens"
    golden_dir.mkdir(parents=True, exist_ok=True)

    artifact = out_dir / f"{name}.pte"
    artifact.write_bytes(buffer)

    taps = tuple(taps or ())
    activations = _tap_specs(model, taps, cases_in[0], dynamic_batch)
    activation_handles = _tap_handles(taps, graph_handles, backend)

    cases = _capture_goldens(
        model=model,
        cases_in=cases_in,
        inputs=inputs,
        outputs=outputs,
        activations=activations,
        golden_dir=golden_dir,
    )

    manifest = ModelManifest(
        name=name,
        weight_hash="sha256:" + hashlib.sha256(buffer).hexdigest(),
        inputs=inputs,
        outputs=outputs,
        quantization=quantization,
        preprocessing=tuple(preprocessing),
        labels=tuple(labels) if labels is not None else None,
        goldens=cases,
        activations=activations,
        activation_handles=activation_handles,
    )

    manifest_path = out_dir / f"{name}.fluttorch.json"
    manifest_path.write_text(manifest.to_json())

    return ExportResult(artifact, manifest_path, golden_dir, manifest)


def _golden_inputs(
    golden_inputs: Iterable[Any] | Callable[[], Iterable[Any]] | None,
    example: tuple[torch.Tensor, ...],
    inputs: tuple[TensorSpec, ...],
) -> list[tuple[torch.Tensor, ...]]:
    """The input tuples to capture references for, validated against the model.

    Resolved before lowering because a static quantization recipe calibrates on
    exactly these, and calibrating on something other than what the goldens then
    measure would make the gate judge a model against inputs it was not tuned for.
    """
    if golden_inputs is None:
        cases: list[tuple[torch.Tensor, ...]] = [example]
    else:
        raw = golden_inputs() if callable(golden_inputs) else golden_inputs
        cases = [_as_tuple(c) for c in raw]
        if not cases:
            raise ExportError(
                "golden_inputs yielded nothing; omit it to capture the example "
                "input alone rather than passing an empty iterable"
            )
    for index, case in enumerate(cases):
        if len(case) != len(inputs):
            raise ExportError(
                f"golden {index} has {len(case)} inputs; the model takes {len(inputs)}"
            )
    return cases


def _tap_specs(
    model: torch.nn.Module,
    taps: tuple[str, ...],
    probe: tuple[torch.Tensor, ...],
    dynamic_batch: bool,
) -> tuple[TensorSpec, ...]:
    """Declare what each tapped submodule produces, by running the model once.

    Observed rather than inferred, for the same reason the output specs are: the
    shape a layer emits is a fact about the model and reading it off the graph
    would be a guess that happens to be right most of the time.

    The order is the order the modules run, which is what makes "the earliest
    layer that diverged" a meaningful claim on the device.
    """
    if not taps:
        return ()

    known = {name for name, _ in model.named_modules() if name}
    missing = [t for t in taps if t not in known]
    if missing:
        raise ExportError(
            f"no submodule named {', '.join(repr(m) for m in missing)}; this model "
            f"has {', '.join(sorted(known)) or 'no named submodules'}"
        )

    _, captured = _run_with_taps(model, taps, probe)
    specs = []
    for i, name in enumerate(taps):
        tensor = captured[name]
        shape = list(tensor.shape)
        if dynamic_batch and shape:
            shape[0] = DYNAMIC_DIM
        specs.append(TensorSpec(name, _wire_dtype(tensor, "activation", i), tuple(shape)))
    return tuple(specs)


def _tap_handles(
    taps: tuple[str, ...],
    graph_handles: dict[str, int],
    backend: str,
) -> tuple[int, ...]:
    """Resolve each tap to the handle the device will see it under.

    Refused rather than half-filled when the lowered graph does not run the layer
    as ops. A bundle that declares a tap the device can never answer for reports
    that layer absent on every run, which reads as a layer nobody looked at and is
    indistinguishable from one that agreed.
    """
    if not taps:
        return ()

    unobservable = [t for t in taps if t not in graph_handles]
    if unobservable:
        raise ExportError(
            f"the lowered graph does not execute {', '.join(repr(t) for t in unobservable)} "
            f"as operations, so nothing on the device can read it back. Lowering for "
            f"{backend!r} hands whole partitions to the delegate, and a delegate is one "
            f"instruction: no layer inside it is addressable. Export with "
            f"backend='portable' to attribute a drift, then lower for {backend!r} to ship it."
        )
    return tuple(graph_handles[t] for t in taps)


def _run_with_taps(
    model: torch.nn.Module,
    taps: tuple[str, ...],
    case: tuple[torch.Tensor, ...],
) -> tuple[tuple[torch.Tensor, ...], dict[str, torch.Tensor]]:
    """Run ``case`` once and return both its outputs and what each tap produced.

    One pass rather than two: a second forward for the outputs would double the
    cost of every golden on a real model, and on a model with any nondeterminism
    it would compare activations against outputs from a different run.

    A tap whose module returns something other than a tensor is refused rather
    than skipped: silently dropping it would leave a hole in the sequence the
    device side walks, and a hole reads as a layer that agreed.
    """
    captured: dict[str, torch.Tensor] = {}
    handles = []

    def hook(name: str):
        def record(_module, _args, output):
            if not isinstance(output, torch.Tensor):
                raise ExportError(
                    f"submodule {name!r} returns {type(output).__name__}, which has no "
                    "single tensor to compare against on the device"
                )
            captured[name] = output.detach()

        return record

    modules = dict(model.named_modules())
    try:
        for name in taps:
            handles.append(modules[name].register_forward_hook(hook(name)))
        with torch.no_grad():
            produced = _as_tuple(model(*case))
    finally:
        for h in handles:
            h.remove()

    absent = [t for t in taps if t not in captured]
    if absent:
        raise ExportError(
            f"submodule(s) {', '.join(repr(a) for a in absent)} did not run on this "
            "input, so there is no activation to record for them"
        )
    return produced, captured


def _capture_goldens(
    *,
    model: torch.nn.Module,
    cases_in: list[tuple[torch.Tensor, ...]],
    inputs: tuple[TensorSpec, ...],
    outputs: tuple[TensorSpec, ...],
    activations: tuple[TensorSpec, ...],
    golden_dir: pathlib.Path,
) -> tuple[GoldenCase, ...]:
    """Run the source model on each case and write the tensors beside it.

    Captured from the model **before lowering**, which is what makes these a
    reference rather than a snapshot of whatever the export happened to produce.
    """
    taps = tuple(s.name for s in activations)

    cases: list[GoldenCase] = []
    for index, case in enumerate(cases_in):
        if taps:
            produced, captured = _run_with_taps(model, taps, case)
        else:
            captured = {}
            with torch.no_grad():
                produced = _as_tuple(model(*case))

        in_keys, out_keys, act_keys = [], [], []
        for spec, tensor in zip(inputs, case, strict=True):
            key = f"{index}/in/{spec.name}.bin"
            _write_tensor(golden_dir / key, tensor, spec)
            in_keys.append(key)
        for spec, tensor in zip(outputs, produced, strict=True):
            key = f"{index}/out/{spec.name}.bin"
            _write_tensor(golden_dir / key, tensor, spec)
            out_keys.append(key)
        for spec in activations:
            key = f"{index}/act/{spec.name}.bin"
            _write_tensor(golden_dir / key, captured[spec.name], spec)
            act_keys.append(key)

        cases.append(
            GoldenCase(
                f"case-{index}",
                tuple(in_keys),
                tuple(out_keys),
                activation_keys=tuple(act_keys),
            )
        )
    return tuple(cases)


def _write_tensor(path: pathlib.Path, tensor: torch.Tensor, spec: TensorSpec) -> None:
    """Write a tensor as raw little-endian bytes.

    Little-endian and contiguous because that is what the Dart ``Tensor`` wraps
    without copying; anything else would force a conversion on the device, on a
    path where the point is that there is no conversion.
    """
    dtype = _wire_dtype(tensor, "golden", 0)
    if dtype != spec.dtype:
        raise ExportError(f"{spec.name!r} was declared {spec.dtype} but this case holds {dtype}")
    expected = spec.byte_length_for(tuple(tensor.shape))
    data = tensor.detach().contiguous().numpy()
    if data.dtype.byteorder not in ("<", "=", "|"):
        data = data.astype(data.dtype.newbyteorder("<"))
    raw = data.tobytes()
    if len(raw) != expected:
        raise ManifestError(
            f"{spec.name!r}: wrote {len(raw)} bytes where the spec needs {expected}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
