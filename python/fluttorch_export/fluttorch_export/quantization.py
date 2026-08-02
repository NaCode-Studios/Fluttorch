"""M17 · the quantization recipes an export can apply.

A recipe is named in the manifest and read back on the device, where it selects
the tolerance the parity gate starts from. That is the whole reason the name is
part of the contract rather than a flag: the numbers a quantized model is allowed
to move by are a property of how it was quantized, and a gate that does not know
the recipe is a gate holding an int4 model to a full-precision bound.

Nothing here invents a scheme. Each recipe is the XNNPACK quantizer configured
one way, and the names match what ``Tolerance.boundFor`` recognises on
the Dart side; a recipe added here without a tolerance there produces an export
the gate cannot judge.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from typing import Any


class QuantizationError(RuntimeError):
    """A recipe could not be applied, with the reason a caller can act on."""


@dataclasses.dataclass(frozen=True, slots=True)
class Recipe:
    """One way of quantizing a model, and what applying it requires."""

    name: str

    #: Whether activation ranges are decided per inference rather than fixed at
    #: export. Dynamic needs no calibration and cannot clip an unseen input;
    #: static is faster on device and clips whatever the calibration set missed.
    is_dynamic: bool

    #: Weight range. int8 is the full signed byte; int4 is the same machinery
    #: held to sixteen levels, which is why it is a range rather than a dtype.
    weight_qmin: int
    weight_qmax: int

    summary: str

    @property
    def needs_calibration(self) -> bool:
        """Whether observed inputs are required to fix the activation ranges."""
        return not self.is_dynamic


#: The recipes this exporter can apply. Keys are what the manifest carries.
RECIPES: dict[str, Recipe] = {
    "int8-dynamic": Recipe(
        name="int8-dynamic",
        is_dynamic=True,
        weight_qmin=-128,
        weight_qmax=127,
        summary="weights per channel to int8, activation ranges decided per call",
    ),
    "int8-static": Recipe(
        name="int8-static",
        is_dynamic=False,
        weight_qmin=-128,
        weight_qmax=127,
        summary="weights and activations to int8, activation ranges fixed by calibration",
    ),
    "int4-weight-only": Recipe(
        name="int4-weight-only",
        is_dynamic=True,
        weight_qmin=-8,
        weight_qmax=7,
        summary="weights per channel to sixteen levels, activations left in float",
    ),
}


def recipe_for(name: str) -> Recipe:
    """The recipe called ``name``.

    Raises [QuantizationError] naming the known recipes, because the failure a
    caller actually hits is a typo and the useful next step is the list.
    """
    try:
        return RECIPES[name]
    except KeyError:
        raise QuantizationError(
            f"unknown quantization recipe {name!r}; this build applies {', '.join(sorted(RECIPES))}"
        ) from None


def quantizer_for(recipe: Recipe) -> Any:
    """Build the XNNPACK quantizer that applies ``recipe``.

    Imported here rather than at module scope so that reading the recipe table,
    which the manifest tests do, costs neither torch nor executorch.
    """
    try:
        from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (
            XNNPACKQuantizer,
            get_symmetric_quantization_config,
        )
    except ImportError as e:
        raise QuantizationError(
            f"quantizing needs executorch's XNNPACK quantizer, which did not import: {e}"
        ) from e

    config = get_symmetric_quantization_config(
        is_per_channel=True,
        is_dynamic=recipe.is_dynamic,
        weight_qmin=recipe.weight_qmin,
        weight_qmax=recipe.weight_qmax,
    )
    return XNNPACKQuantizer().set_global(config)


def check_calibration(recipe: Recipe, cases: Sequence[Any]) -> None:
    """Refuse a static recipe with nothing to calibrate against.

    Calibrating on the single example input fixes every activation range from one
    sample, and every input outside those ranges then clips. The model still
    exports, the parity gate still passes on that one case, and the failure
    appears in production on the second input. Refusing is the only honest answer
    the exporter can give, because it cannot know what a representative set looks
    like for a model it was handed.
    """
    if not recipe.needs_calibration:
        return
    if len(cases) < 2:
        raise QuantizationError(
            f"{recipe.name} fixes activation ranges from observed inputs, and this "
            f"export has {len(cases)} case to observe. Pass --goldens with inputs "
            "that represent the job, or use int8-dynamic, which decides the ranges "
            "per call and cannot clip an input it never saw."
        )
