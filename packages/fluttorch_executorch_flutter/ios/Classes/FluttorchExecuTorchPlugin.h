#import <Flutter/Flutter.h>

// Registers nothing, and is required anyway.
//
// The binding talks to the archive over dart:ffi, so no method channel crosses
// this class. What it buys is the plugin shape: CocoaPods installs the pod and
// links the vendored archive for a plugin, and a plugin needs a class to name.
@interface FluttorchExecuTorchPlugin : NSObject <FlutterPlugin>
@end
