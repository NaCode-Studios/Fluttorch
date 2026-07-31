/// Turning manifest names into Dart identifiers.
///
/// A model is named by whoever exported it and a tensor by whoever wrote the
/// graph, so neither is guaranteed to be a legal identifier. Rather than
/// mangling silently — which produces two tensors that generate the same
/// accessor and a build failure nobody can trace back — anything unusable is
/// reported with the name that caused it.
library;

final _wordBoundary = RegExp(r'[^A-Za-z0-9]+');
final _leadingDigit = RegExp(r'^[0-9]');

/// Dart keywords that cannot be used as an identifier even where the grammar
/// would otherwise allow one.
const _reserved = {
  'abstract',
  'as',
  'assert',
  'async',
  'await',
  'break',
  'case',
  'catch',
  'class',
  'const',
  'continue',
  'covariant',
  'default',
  'deferred',
  'do',
  'dynamic',
  'else',
  'enum',
  'export',
  'extends',
  'extension',
  'external',
  'factory',
  'false',
  'final',
  'finally',
  'for',
  'function',
  'get',
  'hide',
  'if',
  'implements',
  'import',
  'in',
  'interface',
  'is',
  'late',
  'library',
  'mixin',
  'new',
  'null',
  'on',
  'operator',
  'part',
  'required',
  'rethrow',
  'return',
  'sealed',
  'set',
  'show',
  'static',
  'super',
  'switch',
  'sync',
  'this',
  'throw',
  'true',
  'try',
  'typedef',
  'var',
  'void',
  'when',
  'with',
  'while',
  'yield',
};

/// Why a name cannot become an identifier.
final class UnusableNameException implements Exception {
  const UnusableNameException(this.name, this.reason);

  final String name;
  final String reason;

  @override
  String toString() => 'UnusableNameException: "$name" $reason';
}

/// `solar_forecast` becomes `SolarForecast`.
String pascalCase(String name) => _parts(name).map(_capitalise).join();

/// `load_mw` becomes `loadMw`; `class` becomes `class_`.
String camelCase(String name) {
  final parts = _parts(name);
  final joined =
      parts.first.toLowerCase() + parts.skip(1).map(_capitalise).join();
  return _reserved.contains(joined) ? '${joined}_' : joined;
}

List<String> _parts(String name) {
  final parts = name
      .split(_wordBoundary)
      .where((p) => p.isNotEmpty)
      .toList(growable: false);
  if (parts.isEmpty) {
    throw UnusableNameException(
      name,
      'contains no letters or digits, so there is nothing to build an '
      'identifier from',
    );
  }
  if (_leadingDigit.hasMatch(parts.first)) {
    throw UnusableNameException(
      name,
      'starts with a digit, which Dart does not allow; rename it in the export',
    );
  }
  return parts;
}

String _capitalise(String s) =>
    s[0].toUpperCase() + (s.length > 1 ? s.substring(1) : '');

/// Checks that a set of names produces distinct identifiers.
///
/// `load_mw` and `load.mw` are different tensors and the same accessor, which
/// would otherwise surface as a duplicate-member error in generated code that
/// nobody can trace back to the manifest.
void checkDistinct(Iterable<String> names, String role) {
  final seen = <String, String>{};
  for (final name in names) {
    final identifier = camelCase(name);
    final previous = seen[identifier];
    if (previous != null) {
      throw UnusableNameException(
        name,
        'and "$previous" both become "$identifier"; two $role cannot share an '
        'accessor',
      );
    }
    seen[identifier] = name;
  }
}
