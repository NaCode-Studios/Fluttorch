package it.nacodestudios.fluttorch_executorch

import io.flutter.embedding.engine.plugins.FlutterPlugin

/**
 * Registers nothing and is required anyway.
 *
 * The binding talks to the native library over dart:ffi, so no method channel
 * crosses this class and no call ever reaches it. What it buys is the plugin
 * shape: Flutter packages `android/src/main/jniLibs` into the APK for a plugin
 * and not for a plain package, and a plugin needs a class to name.
 */
class FluttorchExecuTorchPlugin : FlutterPlugin {
    override fun onAttachedToEngine(binding: FlutterPlugin.FlutterPluginBinding) {}

    override fun onDetachedFromEngine(binding: FlutterPlugin.FlutterPluginBinding) {}
}
