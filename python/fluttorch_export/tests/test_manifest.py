"""The writing half of the contract, and the fixture that binds it to the reader.

``testdata/manifest_v1.json`` is parsed by this suite and by the Dart suite in
``packages/fluttorch``. A schema alone would let the two implementations drift
while both stayed "valid"; a document both sides must reproduce byte for byte
will not.
"""

from __future__ import annotations

import json
import pathlib

import pytest
from fluttorch_export.manifest import (
    DYNAMIC_DIM,
    SCHEMA_VERSION,
    GoldenCase,
    ManifestError,
    ModelManifest,
    TensorSpec,
    cast,
    center_crop,
    normalize,
    rescale,
    resize,
)

FIXTURE = pathlib.Path(__file__).resolve().parents[3] / "testdata" / "manifest_v1.json"


def canonical() -> ModelManifest:
    """The manifest both suites assert against.

    Deliberately exercises every field and every branch a reader has: a dynamic
    dimension, an omitted optional, every preprocessing step, and a golden with
    and without a description.
    """
    return ModelManifest(
        name="solar_forecast",
        weight_hash="sha256:2f1c0a",
        quantization="int8-static",
        inputs=(
            TensorSpec("window", "float32", (1, 168, 6)),
            TensorSpec("calendar", "int64", (DYNAMIC_DIM, 4)),
        ),
        outputs=(TensorSpec("load_mw", "float32", (1, 24)),),
        preprocessing=(
            rescale(0.00392156862745098),
            normalize([0.5, 0.25], [0.2, 0.1], axis=2),
            resize(168, 6, "nearest"),
            center_crop(160, 6),
            cast("float32"),
        ),
        labels=("low", "high"),
        goldens=(
            GoldenCase(
                "case-0",
                ("in/0/window", "in/0/calendar"),
                ("out/0/load_mw",),
                "a winter evening peak",
            ),
            GoldenCase("case-1", ("in/1/window", "in/1/calendar"), ("out/1/load_mw",)),
        ),
    )


class TestFixture:
    def test_the_committed_fixture_is_what_this_writer_produces(self) -> None:
        assert FIXTURE.read_text() == canonical().to_json(), (
            "the writer and the committed fixture disagree; regenerate it and "
            "check the diff is intended"
        )

    def test_the_fixture_declares_the_current_schema(self) -> None:
        assert json.loads(FIXTURE.read_text())["schema_version"] == SCHEMA_VERSION


class TestEncoding:
    def test_absent_optionals_are_omitted_not_null(self) -> None:
        d = ModelManifest(
            name="m",
            weight_hash="h",
            inputs=(TensorSpec("x", "float32", (1,)),),
            outputs=(TensorSpec("y", "float32", (1,)),),
        ).to_dict()
        for key in ("quantization", "labels", "goldens", "preprocessing"):
            assert key not in d

    def test_dynamic_dimensions_survive_as_the_marker(self) -> None:
        assert canonical().to_dict()["inputs"][1]["shape"] == [DYNAMIC_DIM, 4]

    def test_json_ends_with_a_newline(self) -> None:
        assert canonical().to_json().endswith("}\n")


class TestTensorSpec:
    def test_element_count_is_none_when_dynamic(self) -> None:
        assert TensorSpec("x", "float32", (DYNAMIC_DIM, 4)).element_count is None
        assert TensorSpec("x", "float32", (2, 4)).element_count == 8

    def test_byte_length_resolves_a_dynamic_dimension(self) -> None:
        assert TensorSpec("x", "int64", (DYNAMIC_DIM, 8)).byte_length_for((4, 8)) == 256

    def test_byte_length_rejects_a_shape_that_violates_the_spec(self) -> None:
        with pytest.raises(ManifestError, match="does not satisfy"):
            TensorSpec("x", "float32", (1, 3)).byte_length_for((1, 4))

    @pytest.mark.parametrize(
        ("name", "dtype", "shape", "match"),
        [
            ("", "float32", (), "needs a name"),
            ("x", "float8", (), "unknown dtype"),
            ("x", "float32", (-3,), "use -1 for dynamic"),
        ],
    )
    def test_invalid_specs_are_refused_at_construction(
        self, name: str, dtype: str, shape: tuple[int, ...], match: str
    ) -> None:
        with pytest.raises(ManifestError, match=match):
            TensorSpec(name, dtype, shape)


class TestPreprocessing:
    def test_a_zero_deviation_is_refused_before_it_reaches_disk(self) -> None:
        with pytest.raises(ManifestError, match="divides by zero"):
            normalize([0.5], [0.0])

    def test_mismatched_mean_and_std_lengths_are_refused(self) -> None:
        with pytest.raises(ManifestError, match="2 means against 1"):
            normalize([0.5, 0.5], [1.0])

    def test_the_resize_filter_is_recorded_because_it_changes_the_answer(self) -> None:
        assert resize(8, 8, "nearest").params["interpolation"] == "nearest"
        assert resize(8, 8).params["interpolation"] == "bilinear"

    def test_cast_rejects_an_unknown_target(self) -> None:
        with pytest.raises(ManifestError, match="unknown target"):
            cast("float8")


class TestManifestConsistency:
    IN = (TensorSpec("x", "float32", (1,)),)
    OUT = (TensorSpec("y", "float32", (1,)),)

    def test_a_model_with_no_inputs_is_refused(self) -> None:
        with pytest.raises(ManifestError, match="no inputs"):
            ModelManifest(name="m", weight_hash="h", inputs=(), outputs=self.OUT)

    def test_duplicate_tensor_names_are_refused(self) -> None:
        with pytest.raises(ManifestError, match="duplicate input"):
            ModelManifest(
                name="m",
                weight_hash="h",
                inputs=(
                    TensorSpec("x", "float32", (1,)),
                    TensorSpec("x", "int32", (1,)),
                ),
                outputs=self.OUT,
            )

    def test_duplicate_golden_ids_are_refused(self) -> None:
        with pytest.raises(ManifestError, match="duplicate golden id"):
            ModelManifest(
                name="m",
                weight_hash="h",
                inputs=self.IN,
                outputs=self.OUT,
                goldens=(
                    GoldenCase("a", ("i",), ("o",)),
                    GoldenCase("a", ("i",), ("o",)),
                ),
            )

    def test_a_golden_must_name_one_key_per_tensor(self) -> None:
        with pytest.raises(ManifestError, match="names 2 inputs"):
            ModelManifest(
                name="m",
                weight_hash="h",
                inputs=self.IN,
                outputs=self.OUT,
                goldens=(GoldenCase("a", ("i", "j"), ("o",)),),
            )
