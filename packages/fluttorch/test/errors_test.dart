import 'package:fluttorch/fluttorch.dart';
import 'package:test/test.dart';

/// M30 · the failures are told apart, and each says what to do.
///
/// The taxonomy is in `docs/errors.md`. What is checked here is the part a
/// document cannot hold up on its own: that the types are actually distinct,
/// and that no member can quietly stop carrying a remedy.
void main() {
  final all = <FluttorchException>[
    const ArtifactMismatchException(
      expectedHash: 'sha256:aaa',
      actualHash: 'sha256:bbb',
    ),
    const ManifestFormatException('not an object', field: 'inputs[1].dtype'),
    const ManifestFormatException('not an object'),
    const ManifestVersionException(found: 9, supported: 1),
    const TensorShapeException('rank 2 does not match [1, 3, 8, 8]'),
    DTypeMismatchException(
      tensorName: 'image',
      declared: DType.float16,
      requested: DType.float32,
    ),
    DTypeUnsupportedException(
      tensorName: 'image',
      dtype: DType.float16,
      backend: 'xnnpack',
      supported: const [DType.float32, DType.int8],
    ),
    BackendUnavailableException(requested: 'qnn', available: const ['xnnpack']),
    BackendUnavailableException(requested: 'qnn', available: const []),
    CapabilityUnavailableException(
      backend: 'coreml',
      capability: 'activation taps',
    ),
  ];

  group('M30 · every failure says what to do about it', () {
    test('each carries a remedy, and it is not the message again', () {
      for (final e in all) {
        expect(e.remedy, isNotEmpty, reason: '${e.runtimeType} has no remedy');
        expect(
          e.remedy,
          isNot(equals(e.message)),
          reason: '${e.runtimeType} restates its message instead of advising',
        );
        // Imperative, which is what makes it a remedy rather than a second
        // description of the same fact.
        expect(
          e.remedy[0],
          e.remedy[0].toUpperCase(),
          reason: '${e.runtimeType} remedy should read as an instruction',
        );
      }
    });

    test('toString carries both halves', () {
      for (final e in all) {
        final s = e.toString();
        expect(s, contains(e.message));
        expect(s, contains(e.remedy));
      }
    });

    test('the remedy names the thing the caller has to act on', () {
      expect(
        BackendUnavailableException(
          requested: 'qnn',
          available: const ['xnnpack', 'coreml'],
        ).remedy,
        allOf(contains('xnnpack'), contains('coreml')),
      );
      // An empty list is a different situation with a different fix, and
      // suggesting the caller choose from nothing would be useless.
      expect(
        BackendUnavailableException(
          requested: 'qnn',
          available: const [],
        ).remedy,
        contains('never loaded'),
      );
      expect(
        const ManifestFormatException('bad', field: 'inputs[1].dtype').remedy,
        contains('inputs[1].dtype'),
      );
    });
  });

  group('M30 · the five failures are five types', () {
    test('a wrong read and an unsupported type are not the same failure', () {
      // They were one class, and the runtimes used the wrong one: a reader was
      // told their tensor "holds float16 and was read as float32" when nothing
      // had read anything. That sends them to look at correct code.
      final misread = DTypeMismatchException(
        tensorName: 'image',
        declared: DType.float16,
        requested: DType.float32,
      );
      final unsupported = DTypeUnsupportedException(
        tensorName: 'image',
        dtype: DType.float16,
        backend: 'xnnpack',
        supported: const [DType.float32],
      );
      expect(misread, isNot(isA<DTypeUnsupportedException>()));
      expect(unsupported, isNot(isA<DTypeMismatchException>()));
      expect(misread.message, contains('was read as'));
      expect(unsupported.message, contains('cannot execute'));
      expect(unsupported.message, contains('xnnpack'));
    });

    test('an absent backend and an absent capability are not the same', () {
      // One did not load. The other loaded and cannot do this one thing, which
      // is a working backend rather than a broken one.
      expect(
        BackendUnavailableException(requested: 'qnn', available: const []),
        isNot(isA<CapabilityUnavailableException>()),
      );
    });

    test('a switch over them is exhaustive', () {
      // The point of sealing the hierarchy: adding a failure to this library
      // is a compile error where they are handled, not a surprise at run time.
      String name(FluttorchException e) => switch (e) {
        ArtifactMismatchException() => 'stale bundle',
        ManifestFormatException() => 'unreadable manifest',
        ManifestVersionException() => 'old reader',
        TensorShapeException() => 'wrong buffer',
        DTypeMismatchException() => 'wrong read',
        DTypeUnsupportedException() => 'device cannot carry it',
        BackendUnavailableException() => 'no such backend',
        CapabilityUnavailableException() => 'backend cannot do it',
      };
      expect(all.map(name).toSet(), hasLength(8));
    });
  });

  group('M30 · drift is deliberately not one of them', () {
    test('no exception in the hierarchy describes drift', () {
      // Drift is a measurement. A quantized model that disagrees by two per
      // cent has not failed, it has quantized, and a binding that threw on it
      // could not ship a quantized model. The gate returns a report so the
      // number can be read rather than only reacted to.
      for (final e in all) {
        expect(
          e.message.toLowerCase(),
          isNot(contains('drift')),
          reason: '${e.runtimeType} should not be describing a measurement',
        );
      }
    });
  });
}
