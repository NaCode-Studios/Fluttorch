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
    BACKENDS,
    ExportError,
    available_backends,
    export_model,
    resolve,
)
from fluttorch_export.manifest import ManifestError  # noqa: E402
from fluttorch_export.quantization import QuantizationError  # noqa: E402

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


class TestBackends:
    """M23 · every backend says what it can do, and what it would need."""

    def test_an_unknown_backend_lists_the_ones_this_exporter_knows(
        self, tmp_path: pathlib.Path
    ) -> None:
        with pytest.raises(ExportError, match="xnnpack"):
            export_model(
                model=sample_model.build(),
                example_inputs=sample_model.example_inputs(),
                out_dir=tmp_path,
                name="tpu",
                backend="tpu",
            )

    def test_available_backends_is_a_subset_that_always_has_the_two_that_need_nothing(
        self,
    ) -> None:
        available = available_backends()
        assert set(available) <= set(BACKENDS)
        # These delegate to nothing and to a CPU library respectively, so a
        # machine that can run this suite at all can lower for both.
        assert {"portable", "xnnpack"} <= set(available)

    def test_a_backend_this_machine_lacks_says_what_it_would_need(
        self, tmp_path: pathlib.Path
    ) -> None:
        missing = [b for b in BACKENDS if b not in available_backends()]
        if not missing:
            pytest.skip("this machine lowers for every backend the exporter knows")

        # Named rather than generic. An ImportError from three libraries down
        # says a package nobody asked for is absent; this says which piece of
        # which toolchain, so a caller can install it or pick another backend.
        with pytest.raises(ExportError, match="needs"):
            export_model(
                model=sample_model.build(),
                example_inputs=sample_model.example_inputs(),
                out_dir=tmp_path,
                name="absent",
                backend=missing[0],
            )

    def test_every_backend_this_machine_has_lowers_the_sample_model(
        self, tmp_path: pathlib.Path
    ) -> None:
        # Parameterised over the machine rather than over a fixed list, which is
        # what degrading rather than failing means here: the same suite is green
        # on a laptop with four delegates and on CI with one.
        for backend in available_backends():
            result = export_model(
                model=sample_model.build(),
                example_inputs=sample_model.example_inputs(),
                out_dir=tmp_path / backend,
                name="two_layer",
                backend=backend,
                golden_inputs=sample_model.golden_cases(),
            )
            assert result.artifact.stat().st_size > 0, backend
            assert result.golden_count == len(sample_model.golden_cases()), backend


class TestRuntimes:
    """M27 · an artifact says which engine executes it."""

    def test_an_onnx_export_says_so_and_writes_a_onnx(self, tmp_path) -> None:
        pytest.importorskip("onnxscript", reason="the ONNX export needs onnxscript")
        result = export_model(
            model=sample_model.build(),
            example_inputs=sample_model.example_inputs(),
            out_dir=tmp_path / "onnx",
            name="two_layer",
            runtime="onnx",
            golden_inputs=sample_model.golden_cases(),
            input_names=["features"],
            output_names=["score"],
        )

        assert result.artifact.suffix == ".onnx"
        assert result.manifest.runtime == "onnx"
        assert result.manifest.to_dict()["runtime"] == "onnx"
        assert result.golden_count == len(sample_model.golden_cases())

    def test_an_executorch_export_says_nothing_about_its_runtime(self, tmp_path) -> None:
        # Absence is how ExecuTorch is written, so a manifest from before the
        # field keeps meaning what it meant.
        result = export_model(
            model=sample_model.build(),
            example_inputs=sample_model.example_inputs(),
            out_dir=tmp_path / "et",
            name="two_layer",
            golden_inputs=sample_model.golden_cases(),
        )

        assert result.artifact.suffix == ".pte"
        assert result.manifest.runtime is None
        assert "runtime" not in result.manifest.to_dict()

    def test_onnx_refuses_a_recipe_that_describes_another_runtime(self, tmp_path) -> None:
        # The recipes are built on ExecuTorch's XNNPACK quantizer, so naming one
        # for an ONNX export would put a word in the manifest that describes a
        # lowering nobody performed.
        with pytest.raises(ExportError, match="XNNPACK"):
            export_model(
                model=sample_model.build(),
                example_inputs=sample_model.example_inputs(),
                out_dir=tmp_path / "q",
                name="two_layer",
                runtime="onnx",
                golden_inputs=sample_model.golden_cases(),
                quantization="int8-dynamic",
            )

    def test_onnx_refuses_taps_it_has_no_handles_for(self, tmp_path) -> None:
        with pytest.raises(ExportError, match="portable"):
            export_model(
                model=sample_model.build(),
                example_inputs=sample_model.example_inputs(),
                out_dir=tmp_path / "t",
                name="two_layer",
                runtime="onnx",
                golden_inputs=sample_model.golden_cases(),
                taps=["fc1"],
            )

    def test_an_unknown_runtime_lists_the_ones_it_writes_for(self, tmp_path) -> None:
        with pytest.raises(ExportError, match="executorch"):
            export_model(
                model=sample_model.build(),
                example_inputs=sample_model.example_inputs(),
                out_dir=tmp_path / "n",
                name="two_layer",
                runtime="tflite",
            )


class TestRefusals:
    def test_core_ml_lowers_and_the_manifest_says_so(self, tmp_path: pathlib.Path) -> None:
        # An artifact is lowered for one delegate, so which backend ran a number
        # is a property of the export and not of the device it landed on.
        result = export_model(
            model=sample_model.build(),
            example_inputs=sample_model.example_inputs(),
            out_dir=tmp_path / "coreml",
            name="two_layer",
            backend="coreml",
            golden_inputs=sample_model.golden_cases(),
        )

        assert result.artifact.stat().st_size > 0
        assert result.golden_count == len(sample_model.golden_cases())

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


class TestQuantization:
    """M17 · the recipes, run rather than described.

    Skipped where torch is absent, which includes CI. What these assert is that a
    recipe produces an artifact and that the manifest says which one, not that the
    numbers survived it: that is the parity gate's job, on the device.
    """

    @pytest.mark.parametrize("recipe", ["int8-dynamic", "int4-weight-only"])
    def test_a_recipe_exports_and_is_recorded(self, tmp_path, recipe: str) -> None:
        result = export_model(
            model=sample_model.build(),
            example_inputs=sample_model.example_inputs(),
            out_dir=tmp_path / recipe,
            name="two_layer",
            golden_inputs=sample_model.golden_cases(),
            quantization=recipe,
        )

        assert result.manifest.quantization == recipe
        assert result.artifact.stat().st_size > 0
        # The name is read back on the device to pick a tolerance, so an artifact
        # whose manifest forgot it is one the gate cannot judge.
        assert f'"quantization": "{recipe}"' in result.manifest_path.read_text()

    def test_int8_static_either_exports_or_names_the_toolchain(self, tmp_path) -> None:
        # Whether this recipe converts is a property of the torch a machine
        # resolved, not of this repository. torchao introspects an operator
        # overload that torch 2.13 does not expose, before the model is involved,
        # and 2.12 does. Both versions are reachable: litert-torch pins below 2.13
        # and executorch has no upper bound, and which one pip picks differs by
        # platform, so a macOS checkout converts it and a Linux runner does not.
        #
        # Asserting either outcome universally is asserting something about one
        # machine. What holds everywhere is that it converts and says so, or
        # refuses and names the combination at fault: a bare crash or a manifest
        # that forgot the recipe fails this either way.
        try:
            result = export_model(
                model=sample_model.build(),
                example_inputs=sample_model.example_inputs(),
                out_dir=tmp_path / "static",
                name="two_layer",
                golden_inputs=sample_model.golden_cases(),
                quantization="int8-static",
            )
        except ExportError as e:
            assert "cannot be converted by the installed toolchain" in str(e)
            assert "torch" in str(e)
            return

        assert result.manifest.quantization == "int8-static"
        assert result.artifact.stat().st_size > 0

    def test_a_static_recipe_refuses_a_single_case(self, tmp_path) -> None:
        with pytest.raises(QuantizationError, match="represent the job"):
            export_model(
                model=sample_model.build(),
                example_inputs=sample_model.example_inputs(),
                out_dir=tmp_path / "one",
                name="two_layer",
                quantization="int8-static",
            )

    def test_an_unknown_recipe_is_refused_before_anything_runs(self, tmp_path) -> None:
        with pytest.raises(QuantizationError, match="int8-static"):
            export_model(
                model=sample_model.build(),
                example_inputs=sample_model.example_inputs(),
                out_dir=tmp_path / "nope",
                name="two_layer",
                quantization="int3-experimental",
            )


class TestTaps:
    """M18 · the reference activations a gate needs to attribute a drift."""

    def test_taps_are_declared_and_captured_per_case(self, tmp_path) -> None:
        result = export_model(
            model=sample_model.build(),
            example_inputs=sample_model.example_inputs(),
            out_dir=tmp_path / "taps",
            name="two_layer",
            backend="portable",
            golden_inputs=sample_model.golden_cases(),
            taps=["fc1", "act", "fc2"],
        )

        manifest = result.manifest
        # Declared in the order the graph runs them, which is what makes
        # "the earliest layer that diverged" mean anything on the device.
        assert [s.name for s in manifest.activations] == ["fc1", "act", "fc2"]
        # Shapes are observed by running the model, not read off the graph.
        assert [s.shape for s in manifest.activations] == [(1, 8), (1, 8), (1, 3)]

        for case in manifest.goldens:
            assert len(case.activation_keys) == 3
            for spec, key in zip(manifest.activations, case.activation_keys, strict=True):
                written = result.golden_dir / key
                assert written.exists()
                assert written.stat().st_size == spec.byte_length_for(spec.shape)

    def test_each_tap_carries_the_handle_the_device_reads_it_by(self, tmp_path) -> None:
        # Submodule names do not survive lowering, so the bundle has to say where
        # each tap ended up or the device has nothing to ask for.
        result = export_model(
            model=sample_model.build(),
            example_inputs=sample_model.example_inputs(),
            out_dir=tmp_path / "handles",
            name="two_layer",
            backend="portable",
            golden_inputs=sample_model.golden_cases(),
            taps=["fc1", "act", "fc2"],
        )

        handles = result.manifest.activation_handles
        assert len(handles) == len(result.manifest.activations)
        # Distinct, and ascending in the order the graph runs them: a repeat would
        # mean two layers were resolved to one tensor.
        assert len(set(handles)) == 3
        assert list(handles) == sorted(handles)

    def test_a_half_precision_backend_records_that_it_is(self, tmp_path) -> None:
        # The manifest saying nothing used to mean float32, and Core ML lowering
        # at float16 made that untrue: the artifact answered to a bound it could
        # not hold and the gate failed a model doing what it was told.
        if "coreml" not in available_backends():
            pytest.skip("this machine cannot lower for Core ML")

        result = export_model(
            model=sample_model.build(),
            example_inputs=sample_model.example_inputs(),
            out_dir=tmp_path / "half",
            name="two_layer",
            backend="coreml",
            golden_inputs=sample_model.golden_cases(),
        )

        assert result.manifest.precision == "float16"
        assert result.manifest.quantization is None
        assert result.manifest.to_dict()["precision"] == "float16"

    def test_a_full_precision_backend_says_nothing_about_precision(self, tmp_path) -> None:
        # Absence is how float32 is written, so a manifest from before the field
        # existed keeps meaning what it always meant.
        result = export_model(
            model=sample_model.build(),
            example_inputs=sample_model.example_inputs(),
            out_dir=tmp_path / "full",
            name="two_layer",
            backend="xnnpack",
            golden_inputs=sample_model.golden_cases(),
        )

        assert result.manifest.precision is None
        assert "precision" not in result.manifest.to_dict()

    def test_a_delegated_export_refuses_taps_it_could_never_answer(self, tmp_path) -> None:
        # The export side alone would succeed. It is the device side that cannot,
        # and a bundle promising attribution nobody can deliver is worse than one
        # that says it does not do attribution.
        with pytest.raises(ExportError, match="portable"):
            export_model(
                model=sample_model.build(),
                example_inputs=sample_model.example_inputs(),
                out_dir=tmp_path / "delegated",
                name="two_layer",
                backend="xnnpack",
                golden_inputs=sample_model.golden_cases(),
                taps=["fc1", "act", "fc2"],
            )

    def test_an_unknown_submodule_lists_the_ones_there_are(self, tmp_path) -> None:
        with pytest.raises(ExportError, match="fc1"):
            export_model(
                model=sample_model.build(),
                example_inputs=sample_model.example_inputs(),
                out_dir=tmp_path / "nope",
                name="two_layer",
                taps=["encoder.0"],
            )

    def test_without_taps_nothing_is_declared(self, tmp_path) -> None:
        result = export_model(
            model=sample_model.build(),
            example_inputs=sample_model.example_inputs(),
            out_dir=tmp_path / "none",
            name="two_layer",
        )

        assert result.manifest.activations == ()
        assert all(not c.activation_keys for c in result.manifest.goldens)
