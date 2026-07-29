# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Entries name the roadmap milestone they correspond to, e.g. `(M14)`, so a claim in
[`ROADMAP.md`](ROADMAP.md) can be cross-checked against what actually shipped.

## [Unreleased]

### Added

- Workspace scaffold: the `fluttorch` core, the `fluttorch_gen` and `fluttorch_test` packages, the
  `fluttorch_executorch` backend, and the `fluttorch-export` Python distribution.
- The runtime interface, tensor specs, manifest types and drift metrics that the rest of the project
  is built against. No behaviour yet — these declare the contract.

### Internal

- CI over both toolchains: `dart analyze --fatal-warnings`, `dart format`, `dart test`, and `ruff`.
- Unit tests for element-count arithmetic over static and dynamic shapes, and for tolerance
  evaluation including the small-magnitude case that cosine exists to catch.

[Unreleased]: https://github.com/NaCode-Studios/Fluttorch/commits/main
