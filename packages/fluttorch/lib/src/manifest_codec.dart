import 'dart:convert';

import 'dtype.dart';
import 'errors.dart';
import 'manifest.dart';
import 'tensor.dart';

/// Reads and writes [ModelManifest] as JSON.
///
/// The exporter writes it and this reads it, so the two must agree exactly; the
/// round-trip is covered by tests on both sides. Every failure names the field
/// it stopped at, because the alternative — bisecting a hand-edited manifest
/// against an opaque error — is how a contract stops being trusted.
abstract final class ManifestCodec {
  /// Decodes a manifest from its JSON text.
  static ModelManifest decode(String source) {
    final Object? raw;
    try {
      raw = jsonDecode(source);
    } on FormatException catch (e) {
      throw ManifestFormatException('not valid JSON: ${e.message}');
    }
    if (raw is! Map<String, Object?>) {
      throw const ManifestFormatException(
        'expected a JSON object at the top level',
      );
    }
    return fromJson(raw);
  }

  /// Decodes a manifest from an already-parsed JSON object.
  static ModelManifest fromJson(Map<String, Object?> json) {
    final version = _int(json, 'schema_version');
    if (version > ModelManifest.currentSchemaVersion) {
      throw ManifestVersionException(
        found: version,
        supported: ModelManifest.currentSchemaVersion,
      );
    }
    if (version < 1) {
      throw const ManifestFormatException(
        'schema version must be 1 or greater',
        field: 'schema_version',
      );
    }

    return ModelManifest(
      name: _string(json, 'name'),
      schemaVersion: version,
      weightHash: _string(json, 'weight_hash'),
      quantization: _optionalString(json, 'quantization'),
      runtime: _optionalString(json, 'runtime'),
      precision: _optionalString(json, 'precision'),
      inputs: _specs(json, 'inputs'),
      outputs: _specs(json, 'outputs'),
      preprocessing: _preprocessing(json),
      labels: _optionalStringList(json, 'labels'),
      goldens: _goldens(json),
      activations: json.containsKey('activations')
          ? _specs(json, 'activations')
          : const [],
      activationHandles: json.containsKey('activation_handles')
          ? _intList(json, 'activation_handles')
          : const [],
      parts: _parts(json),
    );
  }

  /// Encodes a manifest as JSON text.
  ///
  /// Written with sorted keys and two-space indentation so a manifest that is
  /// committed produces a readable diff when the model changes.
  static String encode(ModelManifest manifest) =>
      '${const JsonEncoder.withIndent('  ').convert(toJson(manifest))}\n';

  /// Converts a manifest to a JSON-compatible map.
  ///
  /// Absent optional fields are omitted rather than written as null, so an
  /// unquantized model and one quantized with an unnamed recipe cannot be
  /// confused for each other.
  static Map<String, Object?> toJson(ModelManifest m) => {
    'schema_version': m.schemaVersion,
    'name': m.name,
    'weight_hash': m.weightHash,
    if (m.parts.isNotEmpty)
      'parts': [
        for (final p in m.parts)
          {'name': p.name, 'size': p.size, 'hash': p.hash},
      ],
    if (m.runtime != null) 'runtime': m.runtime,
    if (m.quantization != null) 'quantization': m.quantization,
    if (m.precision != null) 'precision': m.precision,
    'inputs': m.inputs.map(_specToJson).toList(),
    'outputs': m.outputs.map(_specToJson).toList(),
    if (m.preprocessing.isNotEmpty)
      'preprocessing': m.preprocessing.map(_stepToJson).toList(),
    if (m.activations.isNotEmpty)
      'activations': m.activations.map(_specToJson).toList(),
    if (m.activationHandles.isNotEmpty)
      'activation_handles': m.activationHandles,
    if (m.labels != null) 'labels': m.labels,
    if (m.goldens.isNotEmpty) 'goldens': m.goldens.map(_goldenToJson).toList(),
  };

  // ── specs ───────────────────────────────────────────────────────────────────

  static Map<String, Object?> _specToJson(TensorSpec s) => {
    'name': s.name,
    'dtype': s.dtype.wireName,
    'shape': s.shape,
    if (s.layout != null) 'layout': s.layout!.wireName,
  };

  static List<TensorSpec> _specs(Map<String, Object?> json, String key) {
    final list = _list(json, key);
    return List.unmodifiable([
      for (var i = 0; i < list.length; i++) _spec(list[i], '$key[$i]'),
    ]);
  }

  static TensorSpec _spec(Object? raw, String path) {
    final map = _object(raw, path);
    final wire = _string(map, 'dtype', path: path);
    final dtype = DType.tryParse(wire);
    if (dtype == null) {
      throw ManifestFormatException(
        'unknown dtype "$wire"; this build understands '
        '${DType.values.map((d) => d.wireName).join(", ")}',
        field: '$path.dtype',
      );
    }
    final shape = _list(map, 'shape', path: path);
    final dims = <int>[];
    for (var i = 0; i < shape.length; i++) {
      final d = shape[i];
      if (d is! int) {
        throw ManifestFormatException(
          'dimension must be an integer, got ${d.runtimeType}',
          field: '$path.shape[$i]',
        );
      }
      if (d < 0 && d != TensorSpec.dynamicDim) {
        throw ManifestFormatException(
          'dimension $d is negative; use ${TensorSpec.dynamicDim} for dynamic',
          field: '$path.shape[$i]',
        );
      }
      dims.add(d);
    }
    return TensorSpec(
      name: _string(map, 'name', path: path),
      dtype: dtype,
      shape: List.unmodifiable(dims),
      layout: _layout(map, dims, path),
    );
  }

  /// Reads the optional layout, rejecting one this build does not know and one
  /// that cannot describe the shape it sits on.
  ///
  /// An unknown layout is refused rather than dropped. Dropping it would leave
  /// a spec that reads as "this tensor has no spatial axes", which is a
  /// different and wrong statement from "this build cannot tell you which".
  static TensorLayout? _layout(
    Map<String, Object?> map,
    List<int> dims,
    String path,
  ) {
    if (!map.containsKey('layout')) return null;
    final wire = _string(map, 'layout', path: path);
    final layout = TensorLayout.tryParse(wire);
    if (layout == null) {
      throw ManifestFormatException(
        'unknown layout "$wire"; this build understands '
        '${TensorLayout.values.map((l) => l.wireName).join(", ")}',
        field: '$path.layout',
      );
    }
    if (dims.length != TensorLayout.rank) {
      throw ManifestFormatException(
        'layout "$wire" names ${TensorLayout.rank} axes but the shape has '
        '${dims.length}, so it cannot say which are spatial',
        field: '$path.layout',
      );
    }
    return layout;
  }

  // ── preprocessing ───────────────────────────────────────────────────────────

  static Map<String, Object?> _stepToJson(PreprocessingStep s) => switch (s) {
    NormalizeStep(:final mean, :final std, :final axis) => {
      'kind': s.kind,
      'mean': mean,
      'std': std,
      'axis': axis,
    },
    RescaleStep(:final factor, :final offset) => {
      'kind': s.kind,
      'factor': factor,
      'offset': offset,
    },
    ResizeStep(:final height, :final width, :final interpolation) => {
      'kind': s.kind,
      'height': height,
      'width': width,
      'interpolation': interpolation,
    },
    CenterCropStep(:final height, :final width) => {
      'kind': s.kind,
      'height': height,
      'width': width,
    },
    CastStep(:final target) => {'kind': s.kind, 'target': target},
    UnknownPreprocessingStep(:final params) => {'kind': s.kind, ...params},
  };

  static List<PreprocessingStep> _preprocessing(Map<String, Object?> json) {
    if (!json.containsKey('preprocessing')) return const [];
    final list = _list(json, 'preprocessing');
    return List.unmodifiable([
      for (var i = 0; i < list.length; i++) _step(list[i], 'preprocessing[$i]'),
    ]);
  }

  static PreprocessingStep _step(Object? raw, String path) {
    final map = _object(raw, path);
    final kind = _string(map, 'kind', path: path);
    switch (kind) {
      case 'normalize':
        final mean = _doubles(map, 'mean', path: path);
        final std = _doubles(map, 'std', path: path);
        if (std.any((v) => v == 0)) {
          throw ManifestFormatException(
            'a standard deviation of zero would divide by zero at inference',
            field: '$path.std',
          );
        }
        if (mean.length != std.length) {
          throw ManifestFormatException(
            'mean has ${mean.length} entries and std has ${std.length}',
            field: path,
          );
        }
        return NormalizeStep(
          mean: mean,
          std: std,
          axis: map.containsKey('axis') ? _int(map, 'axis', path: path) : -1,
        );
      case 'rescale':
        return RescaleStep(
          factor: _double(map, 'factor', path: path),
          offset: map.containsKey('offset')
              ? _double(map, 'offset', path: path)
              : 0,
        );
      case 'resize':
        return ResizeStep(
          height: _int(map, 'height', path: path),
          width: _int(map, 'width', path: path),
          interpolation: map.containsKey('interpolation')
              ? _string(map, 'interpolation', path: path)
              : 'bilinear',
        );
      case 'center_crop':
        return CenterCropStep(
          height: _int(map, 'height', path: path),
          width: _int(map, 'width', path: path),
        );
      case 'cast':
        return CastStep(_string(map, 'target', path: path));
      default:
        // Preserved rather than dropped: a consumer must be able to tell a
        // transform it cannot perform from no transform at all.
        return UnknownPreprocessingStep(
          kind: kind,
          params: Map.unmodifiable(
            Map<String, Object?>.of(map)..remove('kind'),
          ),
        );
    }
  }

  // ── goldens ─────────────────────────────────────────────────────────────────

  static Map<String, Object?> _goldenToJson(GoldenCase g) => {
    'id': g.id,
    'inputs': g.inputKeys,
    'outputs': g.outputKeys,
    if (g.description != null) 'description': g.description,
    if (g.activationKeys.isNotEmpty) 'activations': g.activationKeys,
  };

  static List<BundlePart> _parts(Map<String, Object?> json) {
    if (!json.containsKey('parts')) return const [];
    final list = _list(json, 'parts');
    final seen = <String>{};
    final parts = <BundlePart>[];
    for (var i = 0; i < list.length; i++) {
      final path = 'parts[$i]';
      final map = _object(list[i], path);
      final name = _string(map, 'name', path: path);
      if (name.isEmpty) {
        throw ManifestFormatException(
          'a part with no name cannot be resolved',
          field: '$path.name',
        );
      }
      // A part is named, not located. A separator would let a manifest say
      // "read this file over there", which is a bundle that reads outside
      // itself, and the name has to match what the graph references anyway.
      if (name.contains('/') || name.contains(r'\')) {
        throw ManifestFormatException(
          'part "$name" looks like a path; a part is named relative to the '
          'manifest and cannot point outside the bundle',
          field: '$path.name',
        );
      }
      if (!seen.add(name)) {
        throw ManifestFormatException(
          'duplicate part "$name"; the graph references it once',
          field: '$path.name',
        );
      }
      final size = _int(map, 'size', path: path);
      if (size < 0) {
        throw ManifestFormatException(
          'part "$name" declares a negative size',
          field: '$path.size',
        );
      }
      parts.add(
        BundlePart(
          name: name,
          size: size,
          hash: _string(map, 'hash', path: path),
        ),
      );
    }
    return List.unmodifiable(parts);
  }

  static List<GoldenCase> _goldens(Map<String, Object?> json) {
    if (!json.containsKey('goldens')) return const [];
    final list = _list(json, 'goldens');
    final seen = <String>{};
    final cases = <GoldenCase>[];
    for (var i = 0; i < list.length; i++) {
      final path = 'goldens[$i]';
      final map = _object(list[i], path);
      final id = _string(map, 'id', path: path);
      if (!seen.add(id)) {
        throw ManifestFormatException(
          'duplicate golden id "$id"; ids appear in failure output and must '
          'identify one case',
          field: '$path.id',
        );
      }
      cases.add(
        GoldenCase(
          id: id,
          inputKeys: _stringList(map, 'inputs', path: path),
          outputKeys: _stringList(map, 'outputs', path: path),
          description: _optionalString(map, 'description'),
          activationKeys: map.containsKey('activations')
              ? _stringList(map, 'activations', path: path)
              : const [],
        ),
      );
    }
    return List.unmodifiable(cases);
  }

  // ── typed field access ──────────────────────────────────────────────────────

  static String _at(String? path, String key) =>
      path == null ? key : '$path.$key';

  static Never _missing(String field) =>
      throw ManifestFormatException('required field is absent', field: field);

  static Never _wrong(String field, String want, Object? got) =>
      throw ManifestFormatException(
        'expected $want, got ${got.runtimeType}',
        field: field,
      );

  static Map<String, Object?> _object(Object? raw, String field) =>
      raw is Map<String, Object?> ? raw : _wrong(field, 'an object', raw);

  static String _string(Map<String, Object?> m, String key, {String? path}) {
    final field = _at(path, key);
    final v = m[key] ?? _missing(field);
    return v is String ? v : _wrong(field, 'a string', v);
  }

  static String? _optionalString(Map<String, Object?> m, String key) {
    final v = m[key];
    return v == null ? null : (v is String ? v : _wrong(key, 'a string', v));
  }

  static int _int(Map<String, Object?> m, String key, {String? path}) {
    final field = _at(path, key);
    final v = m[key] ?? _missing(field);
    return v is int ? v : _wrong(field, 'an integer', v);
  }

  static double _double(Map<String, Object?> m, String key, {String? path}) {
    final field = _at(path, key);
    final v = m[key] ?? _missing(field);
    // JSON writes 1.0 as 1, so an int is a valid double here.
    return v is num ? v.toDouble() : _wrong(field, 'a number', v);
  }

  static List<Object?> _list(
    Map<String, Object?> m,
    String key, {
    String? path,
  }) {
    final field = _at(path, key);
    final v = m[key] ?? _missing(field);
    return v is List<Object?> ? v : _wrong(field, 'an array', v);
  }

  static List<double> _doubles(
    Map<String, Object?> m,
    String key, {
    String? path,
  }) {
    final field = _at(path, key);
    final list = _list(m, key, path: path);
    return List.unmodifiable([
      for (var i = 0; i < list.length; i++)
        if (list[i] case final num n)
          n.toDouble()
        else
          _wrong('$field[$i]', 'a number', list[i]),
    ]);
  }

  static List<String> _stringList(
    Map<String, Object?> m,
    String key, {
    String? path,
  }) {
    final field = _at(path, key);
    final list = _list(m, key, path: path);
    return List.unmodifiable([
      for (var i = 0; i < list.length; i++)
        if (list[i] case final String s)
          s
        else
          _wrong('$field[$i]', 'a string', list[i]),
    ]);
  }

  static List<String>? _optionalStringList(
    Map<String, Object?> m,
    String key,
  ) => m.containsKey(key) ? _stringList(m, key) : null;

  static List<int> _intList(
    Map<String, Object?> m,
    String key, {
    String? path,
  }) {
    final field = _at(path, key);
    final list = _list(m, key, path: path);
    return List.unmodifiable([
      for (var i = 0; i < list.length; i++)
        if (list[i] case final int n)
          n
        else
          _wrong('$field[$i]', 'an integer', list[i]),
    ]);
  }
}
