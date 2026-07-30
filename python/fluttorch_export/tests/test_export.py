"""The export pipeline.

Skipped where ``torch`` is absent, which includes CI: installing torch and
executorch there costs several hundred megabytes per run to exercise a path the
manifest suite already covers up to the point where lowering begins. That means
this coverage is **not run in CI**, and saying so is better than a badge that
implies otherwise — the export half is verified locally and on any runner that
has the toolchain.
"""

from __future__ import annotations

import pathlib

import pytest

torch = pytest.importorskip("torch", reason="the export half needs the torch toolchain")
pytest.importorskip("executorch", reason="the export half needs executorch")

from fluttorch_export.export import (  # noqa: E402
    ExportError,
    export_model,
    resolve,
)
from fluttorch_export.manifest import ManifestError  # noqa: E402

from . import sample_model  # noqa: E402


class TestResolve:
    def test_a_factory_is_called(self) -> None:
        assert isinstance(resolve("tests.sample_model:build"), torch.nn.Module)

    def test_a_plain_attribute_is_returned_as_is(self) -> None:
        assert resolve("tests.sample_model:LABELS") == ["low", "mid", "high"]

    @pytest.mark.parametrize(
        ("spec", "match"),
        [
            ("tests.sample_model", "module:attribute"),
            ("nope.not_a_module:x", "cannot import"),
            ("tests.sample_model:missing", "has no attribute"),
        ],
    )
    def test_a_bad_reference_says_which_part_is_wrong(self, spec: str, match: str) -> None:
        with pytest.raises(ExportError, match=match):
            resolve(spec)


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory):
    out = tmp_path_factory.mktemp("export")
    return export_model(
        model=sample_model.build(),
        example_inputs=sample_model.example_inputs(),
        out_dir=out,
        name="two_layer",
        golden_inputs=sample_model.golden_cases,
        labels=sample_model.LABELS,
        input_names=["features"],
        output_names=["score"],
    )


class TestExport:
    def test_the_three_artifacts_are_written_together(self, exported) -> None:
        # Together, because a manifest paired with weights it was not generated
        # from is exactly what the weight hash exists to catch.
        assert exported.artifact.exists()
        assert exported.manifest_path.exists()
        assert exported.golden_dir.is_dir()

    def test_the_hash_is_of_the_artifact_that_was_written(self, exported) -> None:
        import hashlib

        digest = hashlib.sha256(exported.artifact.read_bytes()).hexdigest()
        assert exported.manifest.weight_hash == f"sha256:{digest}"

    def test_specs_carry_the_names_the_caller_gave(self, exported) -> None:
        assert [s.name for s in exported.manifest.inputs] == ["features"]
        assert [s.name for s in exported.manifest.outputs] == ["score"]
        assert exported.manifest.inputs[0].shape == (1, sample_model.IN_FEATURES)
        assert exported.manifest.outputs[0].shape == (1, sample_model.OUT_FEATURES)

    def test_output_specs_come_from_running_the_model(self, exported) -> None:
        # Not from reading the graph: the shape a model actually returns is the
        # only thing that cannot be wrong.
        assert exported.manifest.outputs[0].dtype == "float32"

    def test_full_precision_is_recorded_as_absent(self, exported) -> None:
        assert exported.manifest.quantization is None

    def test_the_export_is_reproducible(self, tmp_path: pathlib.Path, exported) -> None:
        again = export_model(
            model=sample_model.build(),
            example_inputs=sample_model.example_inputs(),
            out_dir=tmp_path,
            name="two_layer",
            input_names=["features"],
            output_names=["score"],
        )
        assert again.manifest.weight_hash == exported.manifest.weight_hash

    def test_a_dynamic_batch_is_marked_when_asked_for(self, tmp_path: pathlib.Path) -> None:
        result = export_model(
            model=sample_model.build(),
            example_inputs=sample_model.example_inputs(),
            out_dir=tmp_path,
            name="dyn",
            dynamic_batch=True,
        )
        assert result.manifest.inputs[0].shape[0] == -1
        assert result.manifest.inputs[0].is_dynamic


class TestGoldens:
    def test_one_case_per_input_the_caller_supplied(self, exported) -> None:
        assert exported.golden_count == len(sample_model.golden_cases())
        assert [g.id for g in exported.manifest.goldens] == [
            "case-0",
            "case-1",
            "case-2",
            "case-3",
        ]

    def test_each_case_names_one_key_per_tensor(self, exported) -> None:
        for g in exported.manifest.goldens:
            assert len(g.input_keys) == len(exported.manifest.inputs)
            assert len(g.output_keys) == len(exported.manifest.outputs)

    def test_the_bytes_are_the_length_the_spec_declares(self, exported) -> None:
        spec = exported.manifest.inputs[0]
        for g in exported.manifest.goldens:
            written = (exported.golden_dir / g.input_keys[0]).read_bytes()
            assert len(written) == spec.byte_length_for(spec.shape)

    def test_references_come_from_the_source_model(self, exported) -> None:
        # Captured before lowering, which is what makes them a reference rather
        # than a snapshot of whatever the export produced.
        import struct

        model = sample_model.build()
        for index, case in enumerate(sample_model.golden_cases()):
            with torch.no_grad():
                expected = model(case).flatten().tolist()
            raw = (exported.golden_dir / f"{index}/out/score.bin").read_bytes()
            stored = list(struct.unpack(f"<{len(expected)}f", raw))
            assert stored == pytest.approx(expected, abs=0, rel=0)

    def test_without_golden_inputs_only_the_example_is_captured(
        self, tmp_path: pathlib.Path
    ) -> None:
        result = export_model(
            model=sample_model.build(),
            example_inputs=sample_model.example_inputs(),
            out_dir=tmp_path,
            name="smoke",
        )
        assert result.golden_count == 1

    def test_an_empty_golden_iterable_is_refused(self, tmp_path: pathlib.Path) -> None:
        # Passing nothing and passing an empty list mean different things, and
        # silently treating the second as the first would hide a broken factory.
        with pytest.raises(ExportError, match="omit it"):
            export_model(
                model=sample_model.build(),
                example_inputs=sample_model.example_inputs(),
                out_dir=tmp_path,
                name="empty",
                golden_inputs=[],
            )

    def test_a_case_with_the_wrong_arity_is_refused(self, tmp_path: pathlib.Path) -> None:
        with pytest.raises(ExportError, match="the model takes 1"):
            export_model(
                model=sample_model.build(),
                example_inputs=sample_model.example_inputs(),
                out_dir=tmp_path,
                name="arity",
                golden_inputs=[[torch.zeros(1, 4), torch.zeros(1, 4)]],
            )


class TestRefusals:
    def test_an_unavailable_backend_says_which_milestone_adds_it(
        self, tmp_path: pathlib.Path
    ) -> None:
        with pytest.raises(ExportError, match="not available yet"):
            export_model(
                model=sample_model.build(),
                example_inputs=sample_model.example_inputs(),
                out_dir=tmp_path,
                name="coreml",
                backend="coreml",
            )

    def test_an_undescribable_dtype_is_refused(self, tmp_path: pathlib.Path) -> None:
        class Complex(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x

        with pytest.raises((ExportError, ManifestError), match="cannot|unknown"):
            export_model(
                model=Complex(),
                example_inputs=torch.ones(2, dtype=torch.complex64),
                out_dir=tmp_path,
                name="cplx",
            )
