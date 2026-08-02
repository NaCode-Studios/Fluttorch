# Changelog

This file records what changed in `fluttorch_test`, the parity gate. The whole
project's record is in
[the repository changelog](https://github.com/NaCode-Studios/Fluttorch/blob/main/CHANGELOG.md).

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
