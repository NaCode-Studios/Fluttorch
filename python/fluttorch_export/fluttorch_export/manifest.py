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

import decimal
import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

#: Schema version this module writes. Raise it only when a reader that does not
#: understand the change would misread the document, not for additive fields.
SCHEMA_VERSION = 1

#: What a manifest declares once it carries parts beside the artifact.
#:
#: Every other field added to this schema has been additive, meaning a reader
#: that did not know it carried on doing what it did before. ``parts`` is the
#: first that is not: a reader without it loads the artifact alone, and for a
#: model whose weights live beside the graph that is a graph with no numbers in
#: it, which loads and runs and answers nonsense. The decoder ignores keys it
#: does not recognise, deliberately, so the version is the only thing standing
#: between an old reader and that.
#:
#: It rises only for a manifest that actually has parts. Every export written so
#: far stays version 1 and nothing has to be re-exported to stay readable.
SCHEMA_VERSION_WITH_PARTS = 2

#: Marks a dimension whose extent is decided at run time.
DYNAMIC_DIM = -1

#: Engines that can execute an artifact. Absent means "executorch", which is what
#: every manifest written before this field meant by saying nothing.
#:
#: Separate from the backend, which names a delegate or provider *within* a
#: runtime. XNNPACK is an ExecuTorch delegate and CoreMLExecutionProvider is an
#: ONNX Runtime provider, and collapsing the two axes into one field is how a
#: bundle ends up describing a combination that does not exist.
RUNTIMES: frozenset[str] = frozenset({"executorch", "litert", "onnx"})

#: Compute precisions a delegate can be lowered at, and that the gate can size a
#: bound for. Float32 is written as absence rather than as a name, so a manifest
#: from before the field existed keeps meaning what it always meant.
PRECISIONS: frozenset[str] = frozenset({"float16", "float32"})

#: Axis orders a rank-4 tensor can declare, so a reader knows which axes are
#: spatial. Absence is not a default: it means the manifest says nothing, and a
#: reader must refuse to resize or crop rather than pick one, because NCHW and
#: NHWC each produce a plausible and different answer from the same bytes.
#:
#: Only rank 4 carries a layout. A rank-2 table has no height, and a name for
#: its axes would be a claim nobody could check.
LAYOUTS: frozenset[str] = frozenset({"nchw", "nhwc"})

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
    layout: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ManifestError(
                "a tensor needs a name; no accessor can be generated for an empty one"
            )
        if self.layout is not None:
            if self.layout not in LAYOUTS:
                raise ManifestError(
                    f"unknown layout {self.layout!r} for {self.name!r}; "
                    f"known: {', '.join(sorted(LAYOUTS))}"
                )
            if len(self.shape) != 4:
                raise ManifestError(
                    f"{self.name!r} declares layout {self.layout!r} and has rank "
                    f"{len(self.shape)}; a layout names four axes, so it only "
                    "describes a rank-4 tensor"
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

    @property
    def spatial_axes(self) -> tuple[int, int]:
        """The (height, width) axis indices this layout names.

        Raises when no layout is declared, rather than returning a default: the
        caller is asking which axes are spatial, and the honest answer to that
        question on a manifest that does not say is that it does not say.
        """
        if self.layout is None:
            raise ManifestError(
                f"{self.name!r} declares no layout, so which axes are spatial is "
                "unknown; record one at export to make resize and center_crop "
                "generatable"
            )
        return (2, 3) if self.layout == "nchw" else (1, 2)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "dtype": self.dtype, "shape": list(self.shape)}
        if self.layout is not None:
            d["layout"] = self.layout
        return d


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
    activation_keys: tuple[str, ...] = ()
    """Reference activations, positional with ``ModelManifest.activations``.

    Absent unless the export captured taps. A gate can compare final outputs
    without them; attributing a drift to the layer that caused it cannot, because
    there is nothing to compare an on-device activation against.
    """

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": self.id,
            "inputs": list(self.input_keys),
            "outputs": list(self.output_keys),
        }
        if self.description is not None:
            d["description"] = self.description
        if self.activation_keys:
            d["activations"] = list(self.activation_keys)
        return d


@dataclass(frozen=True, slots=True)
class BundlePart:
    """One file of an artifact that is more than one file.

    Above a size the exporting toolchain decides on its own, the weights leave
    the graph and land beside it. The graph then references them by file name,
    and a loader handed only the graph has the structure and none of the
    numbers.

    ``name`` is that reference, not a path. It is the string the artifact itself
    carries, so the loader has to hand the bytes back under exactly it for the
    session to resolve them, and a part renamed on disk is a part the graph can
    no longer find.
    """

    name: str
    size: int
    hash: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ManifestError("a bundle part with no name cannot be resolved")
        if "/" in self.name or "\\" in self.name:
            # A part is named, not located. Letting a separator through would
            # make a manifest able to say "read this file over here", which is
            # a bundle that reads outside itself.
            raise ManifestError(
                f"part {self.name!r} looks like a path; a part is named relative "
                "to the manifest and cannot point outside the bundle"
            )
        if self.size < 0:
            raise ManifestError(f"part {self.name!r} declares a negative size")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "size": self.size, "hash": self.hash}


def bundle_digest(artifact: bytes, parts: Sequence[tuple[str, bytes]] = ()) -> str:
    """The hash that covers a whole bundle, artifact and parts together.

    With no parts this is the digest of the artifact and nothing else, which is
    what every manifest written so far declares. That is deliberate: a
    single-file bundle keeps the number it already had, so adding this function
    re-exports nothing.

    With parts, each one contributes its name and its length as well as its
    bytes. Concatenation alone would be ambiguous, since two different splits of
    the same bytes would hash the same, and a hash whose meaning depends on how
    you happened to divide it is not a commitment to anything.
    """
    digest = hashlib.sha256()
    digest.update(artifact)
    for name, data in parts:
        encoded = name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(data)
    return "sha256:" + digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ModelManifest:
    """The contract emitted alongside an exported model."""

    name: str
    weight_hash: str
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    quantization: str | None = None
    runtime: str | None = None
    """The engine that executes this artifact, when it is not ExecuTorch.

    An artifact is a `.pte` or a `.onnx` and the two are not interchangeable, so
    a reader that loads one as the other gets a parse error at best. Nothing in
    the bytes says which, and the weight hash cannot tell them apart because it
    is computed over whichever one was written.

    Absent means ExecuTorch, so a manifest from before this field keeps meaning
    what it meant.
    """
    precision: str | None = None
    """The compute precision the delegate was lowered at, when it is not float32.

    Separate from ``quantization`` because they are separate decisions that
    compound. A recipe says how the weights were stored; this says what the
    delegate does arithmetic in, and a backend can halve the second while leaving
    the first alone. Core ML and MPS both do exactly that by default.

    Absent means float32, which is what a reader without this field assumed
    anyway, so the field is additive. What it fixes is the case where the
    absence was a lie: an artifact lowered at float16 answered to a
    full-precision bound it could not hold, and the gate failed a model that was
    doing what it was told.
    """
    preprocessing: tuple[PreprocessingStep, ...] = ()
    labels: tuple[str, ...] | None = None
    goldens: tuple[GoldenCase, ...] = ()
    activations: tuple[TensorSpec, ...] = ()
    """Intermediate tensors the export tapped, in the order the graph produces them.

    The order is the whole point: attribution reports the earliest layer whose
    numbers moved, and earliest is only meaningful against a declared sequence.

    Additive, so the schema version does not move. A reader that does not know
    this field compares final outputs and says it could not look deeper, which is
    what it did before the field existed.
    """
    activation_handles: tuple[int, ...] = ()
    """Where each tap lives in the lowered graph, positional with ``activations``.

    A device reads intermediates by debug handle, because that is the only name
    the artifact carries: submodule names do not survive lowering. Resolving one
    to the other is the export's job, since only the export sees both.

    Empty when the export could not observe the layers it was asked about, which
    is what a fully delegated graph is. Also additive: a reader without it can
    still compare the activations it was given against the goldens, and simply
    has no way to ask the device for its own.
    """
    parts: tuple[BundlePart, ...] = ()
    """Files the artifact references and cannot be loaded without.

    Empty for every model whose weights fit inside its graph, which is most of
    them, and that is the case the rest of this contract was written for.

    Not additive, unlike everything above it: see ``SCHEMA_VERSION_WITH_PARTS``
    for why a manifest carrying these declares a version an older reader
    refuses. ``weight_hash`` covers the parts as well as the artifact, so the
    pairing this contract rests on still covers the numbers and not merely the
    structure.
    """
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        # Raised here rather than left to the caller because a manifest that
        # carries parts and claims version 1 is precisely the document this
        # version exists to prevent, and one built by hand is as dangerous as
        # one built by the exporter.
        if self.parts and self.schema_version < SCHEMA_VERSION_WITH_PARTS:
            object.__setattr__(self, "schema_version", SCHEMA_VERSION_WITH_PARTS)
        seen: set[str] = set()
        for part in self.parts:
            if part.name in seen:
                raise ManifestError(
                    f"{self.name!r} declares the part {part.name!r} twice, and "
                    "the graph references it once"
                )
            seen.add(part.name)
        if not self.inputs:
            raise ManifestError(f"{self.name!r} declares no inputs")
        if not self.outputs:
            raise ManifestError(f"{self.name!r} declares no outputs")
        for role, specs in (
            ("input", self.inputs),
            ("output", self.outputs),
            ("activation", self.activations),
        ):
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
            # Either a case carries every declared activation or none of them. A
            # partial set would attribute drift to the earliest layer that
            # happens to have been captured, which reads identically to the
            # earliest layer that diverged and is a different claim.
            if g.activation_keys and len(g.activation_keys) != len(self.activations):
                raise ManifestError(
                    f"golden {g.id!r} names {len(g.activation_keys)} activations; "
                    f"the export declares {len(self.activations)}"
                )
        if self.activations and not any(g.activation_keys for g in self.goldens):
            raise ManifestError(
                f"{self.name!r} declares {len(self.activations)} activation(s) that no "
                "golden case records; a tap nothing was captured for cannot be compared"
            )
        if self.runtime is not None and self.runtime not in RUNTIMES:
            raise ManifestError(
                f"{self.name!r} names runtime {self.runtime!r}; this build knows "
                f"{', '.join(sorted(RUNTIMES))}. A runtime nobody can load is a "
                "bundle that describes an artifact no reader will execute"
            )
        if self.precision is not None and self.precision not in PRECISIONS:
            raise ManifestError(
                f"{self.name!r} declares precision {self.precision!r}; this build "
                f"knows {', '.join(sorted(PRECISIONS))}. A precision nobody has "
                "measured a bound for would leave the gate with no bound to pick"
            )
        # Positional, so a partial list would silently address the wrong layer:
        # handle[1] answering for activations[2] reads as a clean comparison of
        # two unrelated tensors.
        if self.activation_handles and len(self.activation_handles) != len(self.activations):
            raise ManifestError(
                f"{self.name!r} carries {len(self.activation_handles)} activation handle(s) "
                f"for {len(self.activations)} activation(s); they are positional"
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
        if self.parts:
            d["parts"] = [p.to_dict() for p in self.parts]
        if self.runtime is not None:
            d["runtime"] = self.runtime
        if self.quantization is not None:
            d["quantization"] = self.quantization
        if self.precision is not None:
            d["precision"] = self.precision
        d["inputs"] = [s.to_dict() for s in self.inputs]
        d["outputs"] = [s.to_dict() for s in self.outputs]
        if self.preprocessing:
            d["preprocessing"] = [s.to_dict() for s in self.preprocessing]
        if self.activations:
            d["activations"] = [s.to_dict() for s in self.activations]
        if self.activation_handles:
            d["activation_handles"] = list(self.activation_handles)
        if self.labels is not None:
            d["labels"] = list(self.labels)
        if self.goldens:
            d["goldens"] = [g.to_dict() for g in self.goldens]
        return d

    def to_json(self) -> str:
        """Two-space indented JSON with a trailing newline.

        Byte-identical to what the Dart reader re-encodes for the same manifest,
        including how a double is spelled and how non-ASCII is written — see
        the canonical encoder below for why that does not come for free.
        """
        return _canonical_json(self.to_dict()) + "\n"


# ── canonical JSON ────────────────────────────────────────────────────────────
#
# The manifest is written here and read by ``ManifestCodec`` in Dart, so the two
# have to agree on how a document is spelled, not merely on what it means.
# ``json.dumps`` does not: it escapes non-ASCII, and it switches a float to
# exponential notation at different magnitudes than Dart does and pads the
# exponent to two digits where Dart does not. Left alone, a normalize mean of
# 1e-5 makes the two sides disagree byte for byte while both stay valid.
#
# Python is the only writer in the pipeline, so it is Python that matches the
# reader's spelling rather than the reverse.

_ESCAPES = {
    '"': '\\"',
    "\\": "\\\\",
    "\b": "\\b",
    "\t": "\\t",
    "\n": "\\n",
    "\f": "\\f",
    "\r": "\\r",
}


def _json_string(value: str) -> str:
    out = ['"']
    for ch in value:
        esc = _ESCAPES.get(ch)
        if esc is not None:
            out.append(esc)
        elif ch < " ":
            out.append(f"\\u{ord(ch):04x}")
        else:
            # Non-ASCII stays raw, which is what Dart emits.
            out.append(ch)
    out.append('"')
    return "".join(out)


def _json_double(value: float) -> str:
    """Render a double the way Dart's ``double.toString`` does.

    Fixed notation for ``1e-6 <= |v| < 1e21`` and for zero; exponential outside
    it, with an unpadded, always-signed exponent. Integral values in the fixed
    range carry a trailing ``.0``.
    """
    if value != value or value in (float("inf"), float("-inf")):
        raise ManifestError(
            f"{value!r} cannot appear in a manifest: JSON has no way to write it, "
            "and a contract that cannot be read back is not a contract"
        )
    if value == 0.0:
        return "-0.0" if math.copysign(1.0, value) < 0 else "0.0"

    d = decimal.Decimal(repr(value))  # repr gives the shortest round-trip digits
    sign, digits, exponent = d.as_tuple()
    adjusted = len(digits) + int(exponent) - 1  # power of ten of the leading digit
    if -7 < adjusted < 21:
        text = format(abs(d), "f")
        if "." not in text:
            text += ".0"
    else:
        mantissa = "".join(str(x) for x in digits)
        if len(mantissa) > 1:
            mantissa = f"{mantissa[0]}.{mantissa[1:]}"
        text = f"{mantissa}e{'+' if adjusted >= 0 else '-'}{abs(adjusted)}"
    return ("-" if sign else "") + text


def _canonical_json(value: Any, depth: int = 0) -> str:
    pad, inner = "  " * depth, "  " * (depth + 1)
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, str):
        return _json_string(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return _json_double(value)
    if isinstance(value, (list, tuple)):
        if not value:
            return "[]"
        items = ",\n".join(inner + _canonical_json(v, depth + 1) for v in value)
        return "[\n" + items + "\n" + pad + "]"
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = ",\n".join(
            f"{inner}{_json_string(str(k))}: {_canonical_json(v, depth + 1)}"
            for k, v in value.items()
        )
        return "{\n" + items + "\n" + pad + "}"
    raise ManifestError(f"{type(value).__name__} has no JSON representation")
