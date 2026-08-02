// The gate, run against a model that is not real.
//
// `fluttorch_test` replays the goldens an export captured and measures how far
// the answer moved. Normally the model comes from a backend package; here it is
// a stand-in, so the example shows what a report says rather than needing a
// native build.
import 'dart:typed_data';

import 'package:fluttorch_test/fluttorch_test.dart';

void main() {
  // A bound is not a constant. It is sized from the recipe the model was
  // quantized with and the precision the delegate lowered to, because an int8
  // model on a half-precision GPU is wrong in both ways at once.
  // Nullable on purpose. A recipe this build does not know has no bound, and
  // inventing one would be a gate measuring against a number nobody chose.
  final full = Tolerance.boundFor(null)!;
  final quantized = Tolerance.boundFor('int8-dynamic')!;
  final half = Tolerance.boundFor(null, precision: 'float16')!;

  print('full precision : rtol ${full.maxRelative}, atol ${full.maxAbsolute}');
  print(
    'int8-dynamic   : rtol ${quantized.maxRelative}, '
    'atol ${quantized.maxAbsolute}',
  );
  print('float16        : rtol ${half.maxRelative}, atol ${half.maxAbsolute}');
  print(
    'an unknown recipe has no bound: '
    '${Tolerance.boundFor('int3-imaginary')}',
  );

  // Measuring one tensor against its reference. `measureParity` does this over
  // a whole golden bundle and a loaded model; the arithmetic underneath is the
  // same, and it reports rather than throws.
  final reference = Float32List.fromList([1.0, 2.0, 3.0]);
  final produced = Float32List.fromList([1.0001, 2.0003, 2.9994]);

  var worstAbsolute = 0.0;
  var worstRelative = 0.0;
  for (var i = 0; i < reference.length; i++) {
    final d = (produced[i] - reference[i]).abs();
    if (d > worstAbsolute) worstAbsolute = d;
    final r = d / reference[i].abs();
    if (r > worstRelative) worstRelative = r;
  }

  print('worst absolute $worstAbsolute, worst relative $worstRelative');
  print(
    'inside the int8 bound: '
    '${worstAbsolute <= quantized.maxAbsolute + quantized.maxRelative * 3}',
  );

  // Drift is a measurement, not an error. A quantized model that moves by two
  // per cent has not failed, it has quantized. `expectParity` is what turns a
  // report into a failing test, and it does that only against the bound the
  // manifest implies.
}
