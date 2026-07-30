"""The manifest: the contract an export emits alongside its artifact.

This is the writing half. The reading half is ``ManifestCodec`` in the ``fluttorch``
Dart package, and the two are held together by a shared fixture both sides parse
in their own test suite -- a schema alone would let the implementations drift while
both remained "valid".

Nothing here derives a value the exporter did not observe. A field that cannot be
determined is absent rather than defaulted, because a default written into a
contract is indistinguishable from a measurement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

#: Schema version this module writes. Raise it only when a reader that does not
#: understand the change would misread the document, not for additive fields.
SCHEMA_VERSION = 1

#: Marks a dimension whose extent is decided at run time.
DYNAMIC_DIM = -1

#: Element types, keyed by the name written to the manifest. The value is the
#: width in bytes, which is what makes ``elements x width == len(buffer)``
#: checkable on both sides.
DTYPES: dict[str, int] = {
    "float32": 4,
    "float64": 8,
    "float16": 2,
    "bfloat16": 2,
    "int8": 1,
    "int16": 2,
    "int32": 4,
    "int64": 8,
    "uint8": 1,
    "bool": 1,
}


class ManifestError(ValueError):
    """A manifest could not be built because the inputs were inconsistent.

    Raised at construction rather than at write time: an invalid manifest that
    reaches disk is one a Dart reader has to reject, and the export is the last
    place with enough context to explain why.
    """


@dataclass(frozen=True, slots=True)
class TensorSpec:
    """The declared shape and type of one model input or output."""

    name: str
    dtype: str
    shape: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.name:
            raise ManifestError(
                "a tensor needs a name; no accessor can be generated for an empty one"
            )
        if self.dtype not in DTYPES:
            raise ManifestError(
                f"unknown dtype {self.dtype!r} for {self.name!r}; "
                f"known: {', '.join(sorted(DTYPES))}"
            )
        for i, d in enumerate(self.shape):
            if d < 0 and d != DYNAMIC_DIM:
                raise ManifestError(
                    f"{self.name!r} dimension {i} is {d}; use {DYNAMIC_DIM} for dynamic"
                )

    @property
    def is_dynamic(self) -> bool:
        return DYNAMIC_DIM in self.shape

    @property
    def element_count(self) -> int | None:
        """Elements, or None when any dimension is dynamic."""
        if self.is_dynamic:
            return None
        n = 1
        for d in self.shape:
            n *= d
        return n

    def byte_length_for(self, shape: tuple[int, ...]) -> int:
        """Bytes a tensor of ``shape`` occupies under this spec."""
        if len(shape) != len(self.shape):
            raise ManifestError(f"{self.name!r}: rank {len(shape)} does not match {self.shape}")
        n = 1
        for declared, actual in zip(self.shape, shape, strict=True):
            if declared != DYNAMIC_DIM and declared != actual:
                raise ManifestError(f"{self.name!r}: shape {shape} does not satisfy {self.shape}")
            n *= actual
        return n * DTYPES[self.dtype]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape)}


@dataclass(frozen=True, slots=True)
class PreprocessingStep:
    """One input transform, recorded so the Dart side replays it identically.

    ``kind`` and its parameters are written flat, which is what lets an older
    reader preserve a step it does not recognise instead of dropping it.
    """

    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, **self.params}


def normalize(mean: list[float], std: list[float], axis: int = -1) -> PreprocessingStep:
    """Subtract ``mean`` and divide by ``std``, per channel."""
    if len(mean) != len(std):
        raise ManifestError(f"normalize: {len(mean)} means against {len(std)} deviations")
    if any(s == 0 for s in std):
        raise ManifestError("normalize: a standard deviation of zero divides by zero at inference")
    return PreprocessingStep("normalize", {"mean": list(mean), "std": list(std), "axis": axis})


def rescale(factor: float, offset: float = 0.0) -> PreprocessingStep:
    """Multiply by ``factor`` then add ``offset``."""
    return PreprocessingStep("rescale", {"factor": factor, "offset": offset})


def resize(
    height: int, width: int, interpolation: Literal["bilinear", "nearest", "bicubic"] = "bilinear"
) -> PreprocessingStep:
    """Resize to ``height`` by ``width`` with the filter training used.

    The filter is part of the contract: bilinear and nearest disagree by enough
    to move a prediction, so reproducing the wrong one is silent skew.
    """
    return PreprocessingStep(
        "resize", {"height": height, "width": width, "interpolation": interpolation}
    )


def center_crop(height: int, width: int) -> PreprocessingStep:
    """Crop ``height`` by ``width`` from the centre."""
    return PreprocessingStep("center_crop", {"height": height, "width": width})


def cast(target: str) -> PreprocessingStep:
    """Cast to another element type without rescaling."""
    if target not in DTYPES:
        raise ManifestError(f"cast: unknown target {target!r}")
    return PreprocessingStep("cast", {"target": target})


@dataclass(frozen=True, slots=True)
class GoldenCase:
    """A reference input and the output the source model produced for it.

    Tensors are named by opaque key, not path: the Dart side may resolve them
    from a directory, a bundled asset archive or memory, and on the web there is
    no filesystem to name.
    """

    id: str
    input_keys: tuple[str, ...]
    output_keys: tuple[str, ...]
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "inputs": list(self.input_keys),
            "outputs": list(self.output_keys),
        }
        if self.description is not None:
            d["description"] = self.description
        return d


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """The contract emitted alongside an exported model."""

    name: str
    weight_hash: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    quantization: str | None = None
    preprocessing: tuple[PreprocessingStep, ...] = ()
    labels: tuple[str, ...] | None = None
    goldens: tuple[GoldenCase, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.inputs:
            raise ManifestError(f"{self.name!r} declares no inputs")
        if not self.outputs:
            raise ManifestError(f"{self.name!r} declares no outputs")
        for role, specs in (("input", self.inputs), ("output", self.outputs)):
            names = [s.name for s in specs]
            duplicates = {n for n in names if names.count(n) > 1}
            if duplicates:
                raise ManifestError(f"duplicate {role} name(s): {', '.join(sorted(duplicates))}")
        ids = [g.id for g in self.goldens]
        dup_ids = {i for i in ids if ids.count(i) > 1}
        if dup_ids:
            raise ManifestError(
                f"duplicate golden id(s) {', '.join(sorted(dup_ids))}; "
                "ids appear in failure output and must identify one case"
            )
        for g in self.goldens:
            if len(g.input_keys) != len(self.inputs):
                raise ManifestError(
                    f"golden {g.id!r} names {len(g.input_keys)} inputs; "
                    f"the model takes {len(self.inputs)}"
                )
            if len(g.output_keys) != len(self.outputs):
                raise ManifestError(
                    f"golden {g.id!r} names {len(g.output_keys)} outputs; "
                    f"the model returns {len(self.outputs)}"
                )

    def to_dict(self) -> dict[str, Any]:
        """The manifest as the JSON document a Dart reader consumes.

        Absent optionals are omitted rather than written as null, so an
        unquantized model and one quantized with an unnamed recipe cannot be
        confused for each other.
        """
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "name": self.name,
            "weight_hash": self.weight_hash,
        }
        if self.quantization is not None:
            d["quantization"] = self.quantization
        d["inputs"] = [s.to_dict() for s in self.inputs]
        d["outputs"] = [s.to_dict() for s in self.outputs]
        if self.preprocessing:
            d["preprocessing"] = [s.to_dict() for s in self.preprocessing]
        if self.labels is not None:
            d["labels"] = list(self.labels)
        if self.goldens:
            d["goldens"] = [g.to_dict() for g in self.goldens]
        return d

    def to_json(self) -> str:
        """Two-space indented JSON with a trailing newline.

        Byte-identical to what the Dart encoder produces for the same manifest,
        which is what lets a committed manifest diff cleanly regardless of which
        side last wrote it.
        """
        return json.dumps(self.to_dict(), indent=2) + "\n"
