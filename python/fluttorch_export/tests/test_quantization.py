"""M17 · the recipe table, and the invariant that binds it to the Dart side.

Runs everywhere, including CI: reading a recipe costs neither torch nor
executorch, which is the reason ``quantizer_for`` imports them inside the call
rather than at module scope. What cannot run here is the lowering itself, and no
test in this file pretends otherwise.
"""

from __future__ import annotations

import pathlib
import re

import pytest
from fluttorch_export.quantization import (
    RECIPES,
    QuantizationError,
    check_calibration,
    recipe_for,
)

TOLERANCE_DART = (
    pathlib.Path(__file__).resolve().parents[3]
    / "packages"
    / "fluttorch_test"
    / "lib"
    / "src"
    / "tolerance.dart"
)


class TestTheTable:
    def test_every_recipe_is_reachable_by_the_name_it_carries(self) -> None:
        for name, recipe in RECIPES.items():
            assert recipe.name == name
            assert recipe_for(name) is recipe

    def test_an_unknown_recipe_lists_the_known_ones(self) -> None:
        with pytest.raises(QuantizationError, match="int8-static"):
            recipe_for("int3-experimental")

    def test_only_a_static_recipe_needs_calibration(self) -> None:
        assert RECIPES["int8-static"].needs_calibration
        assert not RECIPES["int8-dynamic"].needs_calibration
        assert not RECIPES["int4-weight-only"].needs_calibration

    def test_int4_is_sixteen_levels_and_int8_is_the_full_byte(self) -> None:
        int4 = RECIPES["int4-weight-only"]
        assert int4.weight_qmax - int4.weight_qmin + 1 == 16
        int8 = RECIPES["int8-dynamic"]
        assert int8.weight_qmax - int8.weight_qmin + 1 == 256


class TestCalibration:
    def test_a_static_recipe_refuses_a_single_case(self) -> None:
        # One case fixes every activation range from one sample. The export
        # succeeds, the gate passes on that case, and the second input clips.
        with pytest.raises(QuantizationError, match="represent the job"):
            check_calibration(RECIPES["int8-static"], [object()])

    def test_a_static_recipe_accepts_a_set(self) -> None:
        check_calibration(RECIPES["int8-static"], [object(), object()])

    def test_a_dynamic_recipe_needs_nothing(self) -> None:
        check_calibration(RECIPES["int8-dynamic"], [])


class TestBothSidesKnowTheSameRecipes:
    """A recipe the exporter can apply and the gate has no tolerance for is a
    model the gate cannot judge, and the failure is silent: the manifest names a
    scheme, ``startingPointFor`` returns null, and the suite demands a tolerance
    nobody has measured. The two lists are asserted equal rather than kept in
    step by hand."""

    def test_the_dart_tolerance_recognises_exactly_these_recipes(self) -> None:
        source = TOLERANCE_DART.read_text()
        block = re.search(r"knownRecipes\s*=\s*\{(.*?)\}", source, re.DOTALL)
        assert block, f"knownRecipes not found in {TOLERANCE_DART}"
        dart_names = set(re.findall(r"'([^']+)'", block.group(1)))
        assert dart_names == set(RECIPES)
