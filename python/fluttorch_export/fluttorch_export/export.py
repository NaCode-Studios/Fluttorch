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


def _lower(model: torch.nn.Module, example: tuple[torch.Tensor, ...], backend: str) -> bytes:
    """Trace, lower and serialise. Full precision only until M17."""
    if backend != "xnnpack":
        raise ExportError(
            f"backend {backend!r} is not available yet; M21 adds Core ML and "
            "M23 the rest. Only 'xnnpack' works today."
        )
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import to_edge_transform_and_lower

    exported = torch.export.export(model, example)
    lowered = to_edge_transform_and_lower(
        exported, partitioner=[XnnpackPartitioner()]
    ).to_executorch()
    return bytes(lowered.buffer)


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
) -> ExportResult:
    """Export ``model``, and write the artifact, the manifest and the goldens.

    ``golden_inputs`` yields input tuples to capture references for. When it is
    absent only ``example_inputs`` is captured, which is a smoke test rather than
    coverage — the exporter says so rather than inventing a distribution it has
    no way to know.
    """
    model = model.eval()
    example = _as_tuple(example_inputs)

    with torch.no_grad():
        reference = _as_tuple(model(*example))

    inputs = _specs(example, "input", input_names, dynamic_batch)
    outputs = _specs(reference, "output", output_names, dynamic_batch)

    buffer = _lower(model, example, backend)

    out_dir = pathlib.Path(out_dir)
    golden_dir = out_dir / "goldens"
    golden_dir.mkdir(parents=True, exist_ok=True)

    artifact = out_dir / f"{name}.pte"
    artifact.write_bytes(buffer)

    cases = _capture_goldens(
        model=model,
        example=example,
        golden_inputs=golden_inputs,
        inputs=inputs,
        outputs=outputs,
        golden_dir=golden_dir,
    )

    manifest = ModelManifest(
        name=name,
        weight_hash="sha256:" + hashlib.sha256(buffer).hexdigest(),
        inputs=inputs,
        outputs=outputs,
        preprocessing=tuple(preprocessing),
        labels=tuple(labels) if labels is not None else None,
        goldens=cases,
    )

    manifest_path = out_dir / f"{name}.fluttorch.json"
    manifest_path.write_text(manifest.to_json())

    return ExportResult(artifact, manifest_path, golden_dir, manifest)


def _capture_goldens(
    *,
    model: torch.nn.Module,
    example: tuple[torch.Tensor, ...],
    golden_inputs: Iterable[Any] | Callable[[], Iterable[Any]] | None,
    inputs: tuple[TensorSpec, ...],
    outputs: tuple[TensorSpec, ...],
    golden_dir: pathlib.Path,
) -> tuple[GoldenCase, ...]:
    """Run the source model on each case and write the tensors beside it.

    Captured from the model **before lowering**, which is what makes these a
    reference rather than a snapshot of whatever the export happened to produce.
    """
    if golden_inputs is None:
        cases_in: list[tuple[torch.Tensor, ...]] = [example]
    else:
        raw = golden_inputs() if callable(golden_inputs) else golden_inputs
        cases_in = [_as_tuple(c) for c in raw]
        if not cases_in:
            raise ExportError(
                "golden_inputs yielded nothing; omit it to capture the example "
                "input alone rather than passing an empty iterable"
            )

    cases: list[GoldenCase] = []
    for index, case in enumerate(cases_in):
        if len(case) != len(inputs):
            raise ExportError(
                f"golden {index} has {len(case)} inputs; the model takes {len(inputs)}"
            )
        with torch.no_grad():
            produced = _as_tuple(model(*case))

        in_keys, out_keys = [], []
        for spec, tensor in zip(inputs, case, strict=True):
            key = f"{index}/in/{spec.name}.bin"
            _write_tensor(golden_dir / key, tensor, spec)
            in_keys.append(key)
        for spec, tensor in zip(outputs, produced, strict=True):
            key = f"{index}/out/{spec.name}.bin"
            _write_tensor(golden_dir / key, tensor, spec)
            out_keys.append(key)

        cases.append(GoldenCase(f"case-{index}", tuple(in_keys), tuple(out_keys)))
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
