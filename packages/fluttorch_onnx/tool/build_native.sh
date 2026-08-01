#!/usr/bin/env bash
# Builds the ONNX Runtime half of the binding.
#
#   tool/build_native.sh [path-to-onnxruntime-sdk]
#
# The SDK is the official release archive, which carries both the headers and
# the library. The wheel on PyPI ships only the library, so a build against it
# cannot compile anything:
#
#   V=1.28.0
#   curl -sSL -o ort.tgz \
#     "https://github.com/microsoft/onnxruntime/releases/download/v$V/onnxruntime-osx-arm64-$V.tgz"
#   tar xzf ort.tgz && mv "onnxruntime-osx-arm64-$V" ~/.cache/fluttorch/onnxruntime
#
# The shim implements the same header fluttorch_executorch implements, on
# purpose. A second runtime is a second implementation of one seam.
set -euo pipefail

ORT="${1:-$HOME/.cache/fluttorch/onnxruntime}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

[ -f "$ORT/include/onnxruntime_c_api.h" ] || {
  echo "no ONNX Runtime SDK at $ORT. The header of this script has the command" >&2
  echo "that fetches one." >&2
  exit 1
}

case "$(uname -s)" in
  Darwin) LIB=libfluttorch_onnx.dylib; SHARED=(-dynamiclib) ;;
  *)      LIB=libfluttorch_onnx.so;    SHARED=(-shared -fPIC) ;;
esac

DEFINES=()
# Core ML is a provider the Apple builds of ONNX Runtime carry. Asked of the
# library rather than assumed from the platform, for the same reason every other
# backend here is: a build that reports a provider it does not have fails at
# load rather than at link.
if [ "$(uname -s)" = Darwin ] &&
   nm -gU "$ORT/lib/libonnxruntime.dylib" 2>/dev/null |
     grep CoreML >/dev/null; then
  DEFINES+=(-DFLUTTORCH_ONNX_WITH_COREML)
  echo "linking the Core ML provider"
else
  echo "no Core ML provider in this SDK"
fi

mkdir -p "$HERE/.dart_tool/native"

c++ -std=c++17 -O2 -Wall -fPIC -c "$HERE/src/fluttorch_onnx.cpp" \
  -o "$HERE/.dart_tool/native/shim.o" \
  -I "$ORT/include" ${DEFINES[@]+"${DEFINES[@]}"}

# rpath rather than a copy: the SDK is a checkout like ExecuTorch's, and a
# library duplicated into .dart_tool is one that goes stale silently.
c++ -std=c++17 "${SHARED[@]}" -o "$HERE/.dart_tool/native/$LIB" \
  "$HERE/.dart_tool/native/shim.o" \
  -L "$ORT/lib" -lonnxruntime -Wl,-rpath,"$ORT/lib"

rm -f "$HERE/.dart_tool/native/shim.o"
echo "built $HERE/.dart_tool/native/$LIB"
