# Fluttorch

Ship a PyTorch model to Flutter and find out when it stops agreeing with the notebook.

Fluttorch is a Dart binding to on-device ML runtimes, built around one claim: an artifact, the
manifest that describes it and the references it was measured against are written together, and
anything that reads one reads all three. A model exported here produces a typed Dart API where the
compiler rejects the wrong tensor, and a gate that replays the references and fails the build when
the numbers move further than the recipe allows.

## Start here

- **[Concepts](concepts.md)**. The manifest, the gate, and why they are one thing rather than two.
- **[Quickstart](quickstart.md)**. Export a model, generate the API, run the gate.
- **[Tolerances](tolerance.md)**. Where a bound comes from, and why it is not a constant.
- **[Errors](errors.md)**. The five ways this fails, and the one that is deliberately not an error.
- **[Backend coverage](backend-coverage.md)**. What each backend is verified to do, and where.
- **[The parity gate in CI](ci-parity-gate.md)**. The workflow, and the recipe to copy.

## What it deliberately does not do

It does not train, convert or optimise a model. It does not hide a runtime behind a portable
abstraction. It does not guess: where the manifest does not say something, the code refuses rather
than assuming, and most of this documentation is about where those refusals are and why each one is
worth more than the convenience it costs.

## Status

Early development. Until `1.0`, minor versions may break. The
[board](https://github.com/orgs/NaCode-Studios/projects/6) is the plan of record, and the
[changelog](https://github.com/NaCode-Studios/Fluttorch/blob/main/CHANGELOG.md) is what shipped.
