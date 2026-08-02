# Changelog

This file records what changed in `fluttorch_gen`, the `build_runner` builder.
The whole project's record is in
[the repository changelog](https://github.com/NaCode-Studios/Fluttorch/blob/main/CHANGELOG.md).

## 0.7.0

- `resize` and `center_crop` are generated from the layout the manifest
  records, as a `fromSource` constructor rather than an in-place pass. The
  generated crop rounds the way torchvision does, which is not the way Dart
  does, and on an odd margin the difference was one row.

## 0.6.0

- `resize` and `center_crop` are generated, from the layout the manifest now
  records. They become a `fromSource` constructor rather than joining the
  in-place `preprocess()` pass, because they change the number of elements.
- Three new refusals, each a case where generating something would have been
  worse than generating nothing: a resize filter this builder does not
  implement, a spatial step recorded after an elementwise one, and spatial steps
  on a model with more than one input.
- `build` moves to `^4.0.0`. The generated output is byte-identical across the
  bump.

## 0.5.0

- Aliases, and the generated model class that loads its own manifest.

## 0.4.0

- Quantized manifests generate the same typed API as full-precision ones.

## 0.3.0

- Generated preprocessing: rescale, normalize and cast, from the manifest.

## 0.2.0

- One extension type per tensor, so passing the wrong one is a compile error.

## 0.1.0

- First builder: a manifest becomes a typed Dart API.

## 0.0.1

- First tag.
