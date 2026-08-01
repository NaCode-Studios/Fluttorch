#
# Puts the binding inside the application binary, which on iOS is the only place
# it can be.
#
# An iOS app cannot dlopen an arbitrary dylib from its bundle, so
# NativeExecuTorchBindings.open() resolves through DynamicLibrary.process() and
# the symbols have to already be there. That makes this a static library rather
# than a framework, and it makes the -force_load below load-bearing rather than
# an optimisation: a backend registers itself from a static initialiser, and a
# linker drops those from an archive nobody references by symbol. Without it the
# app links, launches, and reports no backend at all.
#
Pod::Spec.new do |s|
  s.name             = 'fluttorch_executorch_flutter'
  s.version          = '0.6.0'
  s.summary          = 'Ships the compiled ExecuTorch binding into an iOS app.'
  s.description      = <<-DESC
The Dart API lives in fluttorch_executorch. This pod carries the archive that
API talks to, built by packages/fluttorch_executorch/tool/build_native.sh --ios.
                       DESC
  s.homepage         = 'https://github.com/NaCode-Studios/Fluttorch'
  s.license          = { :file => '../../../LICENSE' }
  s.author           = { 'NaCode Studios' => 'info@nacodestudios.it' }
  s.source           = { :path => '.' }
  s.source_files     = 'Classes/**/*'
  s.dependency 'Flutter'
  s.platform = :ios, '17.0'

  s.vendored_libraries = 'Libraries/libfluttorch_executorch.a'
  s.static_framework = true

  # Accelerate is XNNPACK's, and the two Apple delegates need their own. They are
  # listed unconditionally because a framework that goes unused costs nothing at
  # run time, while a missing one is a link error on somebody else's machine.
  s.frameworks = 'Accelerate', 'CoreML', 'Metal', 'MetalPerformanceShaders',
                 'MetalPerformanceShadersGraph'
  s.libraries = 'sqlite3', 'c++'

  s.pod_target_xcconfig = {
    'DEFINES_MODULE' => 'YES',
    'EXCLUDED_ARCHS[sdk=iphonesimulator*]' => 'i386'
  }
  s.user_target_xcconfig = {
    'OTHER_LDFLAGS' => '-force_load "${PODS_ROOT}/../.symlinks/plugins/fluttorch_executorch_flutter/ios/Libraries/libfluttorch_executorch.a"'
  }
end
