# Changelog

This file records what changed in `fluttorch_test`, the parity gate. The whole
project's record is in
[the repository changelog](https://github.com/NaCode-Studios/Fluttorch/blob/main/CHANGELOG.md).

## 1.0.0

- `Tolerance.startingPointFor` is now `Tolerance.boundFor`, and every bound
  except `int4-weight-only` comes from a measurement rather than a defensible
  guess. Each entry cites the models it was measured against, and
  `tool/measure_tolerances.dart` in the ExecuTorch package reproduces them.
- `int8-dynamic` widens from `5e-2` to `1e-1`, because it measured `4.1e-2` and
  a factor of 1.2 is a coincidence rather than a margin. `int8-static` and
  `int4-weight-only` widen to keep the recipes ordered.
- `ParityMatrix.precisionOf`, so a report can order its columns by what the
  artifact was lowered at rather than by the backend's name.
- `DirectoryGoldenBundle.parts()` reads the files a manifest declares beside it,
  and `bundleRoot` says where they were read from.

## 0.7.0

- No API change. Released with the rest of the project.

## 0.6.0

- No API change. Released with the rest of the project.

## 0.5.0

- `measureMatrix` replays one set of goldens across every backend a machine
  offers, each column measured at the bound its own manifest implies.
- Tolerances are sized from the recipe and the precision together, because an
  int8 model on a half-precision GPU is wrong in both ways at once.

## 0.4.0

- Drift attribution: where the export captured activation taps, a report names
  the earliest layer whose numbers moved.

## 0.3.0

- `measureParity` and `expectParity`, and the reports they produce.

## 0.2.0

- Golden bundles, read from a directory.

## 0.1.0

- First matchers.

## 0.0.1

- First tag.
