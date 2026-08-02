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
        activations=(
            TensorSpec("encoder.0", "float32", (1, 168, 32)),
            TensorSpec("encoder.2", "float32", (DYNAMIC_DIM, 84, 32)),
        ),
        goldens=(
            GoldenCase(
                "case-0",
                ("in/0/window", "in/0/calendar"),
                ("out/0/load_mw",),
                "a winter evening peak",
                activation_keys=("act/0/encoder.0", "act/0/encoder.2"),
            ),
            GoldenCase(
                "case-1",
                ("in/1/window", "in/1/calendar"),
                ("out/1/load_mw",),
                activation_keys=("act/1/encoder.0", "act/1/encoder.2"),
            ),
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


class TestActivations:
    """M18 · the taps a gate needs to say which layer moved first."""

    def test_a_case_may_carry_none(self) -> None:
        m = ModelManifest(
            name="m",
            weight_hash="sha256:00",
            inputs=(TensorSpec("x", "float32", (1,)),),
            outputs=(TensorSpec("y", "float32", (1,)),),
            goldens=(GoldenCase("case-0", ("a",), ("b",)),),
        )
        assert m.to_dict().get("activations") is None
        assert "activations" not in m.to_dict()["goldens"][0]

    def test_a_partial_set_is_refused(self) -> None:
        # Attributing to the earliest captured layer reads exactly like
        # attributing to the earliest diverging one, and is a different claim.
        with pytest.raises(ManifestError, match="names 1 activations"):
            ModelManifest(
                name="m",
                weight_hash="sha256:00",
                inputs=(TensorSpec("x", "float32", (1,)),),
                outputs=(TensorSpec("y", "float32", (1,)),),
                activations=(
                    TensorSpec("l0", "float32", (1,)),
                    TensorSpec("l1", "float32", (1,)),
                ),
                goldens=(GoldenCase("case-0", ("a",), ("b",), activation_keys=("k",)),),
            )

    def test_a_declared_tap_nothing_captured_is_refused(self) -> None:
        with pytest.raises(ManifestError, match="no golden case records"):
            ModelManifest(
                name="m",
                weight_hash="sha256:00",
                inputs=(TensorSpec("x", "float32", (1,)),),
                outputs=(TensorSpec("y", "float32", (1,)),),
                activations=(TensorSpec("l0", "float32", (1,)),),
                goldens=(GoldenCase("case-0", ("a",), ("b",)),),
            )

    def test_two_taps_cannot_share_a_name(self) -> None:
        with pytest.raises(ManifestError, match="duplicate activation"):
            ModelManifest(
                name="m",
                weight_hash="sha256:00",
                inputs=(TensorSpec("x", "float32", (1,)),),
                outputs=(TensorSpec("y", "float32", (1,)),),
                activations=(
                    TensorSpec("l0", "float32", (1,)),
                    TensorSpec("l0", "float32", (2,)),
                ),
                goldens=(GoldenCase("case-0", ("a",), ("b",), activation_keys=("k", "j")),),
            )


class TestLayout:
    """M31 · which axes are spatial, recorded rather than inferred."""

    def test_a_layout_round_trips_through_the_document(self) -> None:
        spec = TensorSpec("image", "float32", (1, 3, 8, 8), layout="nchw")
        assert spec.to_dict()["layout"] == "nchw"

    def test_absence_is_absence_rather_than_a_default(self) -> None:
        # A manifest that says nothing is not one that says NCHW. Writing a
        # default would make every tabular model claim to have spatial axes.
        spec = TensorSpec("features", "float32", (1, 4))
        assert "layout" not in spec.to_dict()
        assert spec.layout is None

    def test_an_unknown_layout_names_the_ones_there_are(self) -> None:
        with pytest.raises(ManifestError, match="known: nchw, nhwc"):
            TensorSpec("image", "float32", (1, 3, 8, 8), layout="nhcw")

    def test_a_layout_only_describes_four_axes(self) -> None:
        with pytest.raises(ManifestError, match="rank 2"):
            TensorSpec("features", "float32", (1, 4), layout="nchw")

    def test_it_says_which_axes_it_named(self) -> None:
        assert TensorSpec("i", "float32", (1, 3, 8, 8), layout="nchw").spatial_axes == (2, 3)
        assert TensorSpec("i", "float32", (1, 8, 8, 3), layout="nhwc").spatial_axes == (1, 2)

    def test_asking_a_manifest_that_does_not_say_raises(self) -> None:
        # Rather than returning a default, which is the failure the field
        # exists to prevent: a resize down the wrong axes returns a picture.
        with pytest.raises(ManifestError, match="declares no layout"):
            TensorSpec("features", "float32", (1, 4)).spatial_axes
