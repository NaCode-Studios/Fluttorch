import 'package:flutter/material.dart';

/// A window for the integration test to run in.
///
/// The evidence this example exists for is in `integration_test/`, not here. A
/// screen that ran a model and printed a number would be a second place the
/// claim is made, and it would be the one nobody checks.
void main() => runApp(const OnDeviceApp());

class OnDeviceApp extends StatelessWidget {
  const OnDeviceApp({super.key});

  @override
  Widget build(BuildContext context) => const MaterialApp(
    home: Scaffold(body: Center(child: Text('Run the integration test.'))),
  );
}
