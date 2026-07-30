"""The canonical encoder, and why it is not ``json.dumps``.

Python and Dart both write the shortest round-trip decimal for a double, and then
spell it differently: Python switches to exponential below ``1e-4`` and at
``1e16`` with a two-digit exponent, Dart below ``1e-6`` and at ``1e21`` with an
unpadded one. Python also escapes non-ASCII by default and Dart does not.

Either alone would make two valid documents differ byte for byte. Python is the
only writer in the pipeline, so it is the one that matches the reader.
"""

from __future__ import annotations

import math

import pytest
from fluttorch_export.manifest import (
    ManifestError,
    _canonical_json,
    _json_double,
    _json_string,
)


class TestDoubles:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            # Dart keeps fixed notation down to 1e-6 and up to just under 1e21.
            (1e-4, "0.0001"),
            (1e-5, "0.00001"),
            (1e-6, "0.000001"),
            (1e-7, "1e-7"),
            (1e-8, "1e-8"),
            (1e19, "10000000000000000000.0"),
            (1e20, "100000000000000000000.0"),
            (1e21, "1e+21"),
            (1e22, "1e+22"),
            # An exponent is signed and never zero-padded.
            (1.25e-7, "1.25e-7"),
            (1.25e21, "1.25e+21"),
            (1e100, "1e+100"),
            # Integral values inside the fixed range carry a trailing .0.
            (3.0, "3.0"),
            (-3.5, "-3.5"),
            (0.0, "0.0"),
            (-0.0, "-0.0"),
            # Shortest round-trip digits, not a rounded rendering.
            (0.1 + 0.2, "0.30000000000000004"),
            (1 / 3, "0.3333333333333333"),
            (1.7976931348623157e308, "1.7976931348623157e+308"),
            (5e-324, "5e-324"),
        ],
    )
    def test_matches_dart_double_to_string(self, value: float, expected: str) -> None:
        assert _json_double(value) == expected

    def test_every_double_round_trips_through_its_rendering(self) -> None:
        for value in (1e-7, 1e21, 0.1 + 0.2, 1 / 3, 5e-324, -2.5e-11, 1e16):
            assert float(_json_double(value)) == value

    @pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
    def test_a_value_json_cannot_write_is_refused(self, value: float) -> None:
        # Writing NaN would produce a document no conforming reader accepts, and
        # a contract that cannot be read back is not a contract.
        with pytest.raises(ManifestError, match="cannot appear in a manifest"):
            _json_double(value)


class TestStrings:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("plain", '"plain"'),
            ('quote "', '"quote \\""'),
            ("back \\", '"back \\\\"'),
            ("tab \t", '"tab \\t"'),
            ("nl \n", '"nl \\n"'),
            ("\r\f\b", '"\\r\\f\\b"'),
            ("\x01", '"\\u0001"'),
        ],
    )
    def test_escapes_match_dart(self, value: str, expected: str) -> None:
        assert _json_string(value) == expected

    def test_non_ascii_is_written_raw(self) -> None:
        # json.dumps would emit ° here; Dart emits the character.
        assert _json_string("°C ✓") == '"°C ✓"'


class TestStructure:
    def test_empty_collections_stay_on_one_line(self) -> None:
        assert _canonical_json({"a": {}, "b": []}) == '{\n  "a": {},\n  "b": []\n}'

    def test_nesting_indents_two_spaces_per_level(self) -> None:
        assert _canonical_json({"a": [1, {"b": True}]}) == (
            '{\n  "a": [\n    1,\n    {\n      "b": true\n    }\n  ]\n}'
        )

    def test_literals(self) -> None:
        assert _canonical_json({"t": True, "f": False, "n": None}) == (
            '{\n  "t": true,\n  "f": false,\n  "n": null\n}'
        )

    def test_integers_are_not_rendered_as_doubles(self) -> None:
        # A shape of [1, 24] must not become [1.0, 24.0].
        assert _canonical_json([1, -1, 9007199254740991]) == (
            "[\n  1,\n  -1,\n  9007199254740991\n]"
        )

    def test_a_type_with_no_representation_is_refused(self) -> None:
        with pytest.raises(ManifestError, match="no JSON representation"):
            _canonical_json({"a": {1, 2}})
