#!/usr/bin/env bash
# Builds the LiteRT half of the binding.
#
#   tool/build_native.sh [path-to-litert-cc-sdk]
#
# Two pieces from two places, because neither ships both. The C headers are in
# the SDK archive attached to a LiteRT release; the library is in the
# ai-edge-litert wheel, which ships no headers at all:
#
#   V=2.1.6
#   curl -sSL -o sdk.zip \
#     "https://github.com/google-ai-edge/LiteRT/releases/download/v$V/litert_cc_sdk.zip"
#   unzip -q sdk.zip -d ~/.cache/fluttorch/litert
#
# Keep the two versions equal. They are an ABI pair, and a mismatch is a struct
# layout disagreement rather than a link error.
set -euo pipefail

SDK="${1:-$HOME/.cache/fluttorch/litert/litert_cc_sdk}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

[ -f "$SDK/litert/c/litert_common.h" ] || {
  echo "no LiteRT C SDK at $SDK. The header of this script has the command that" >&2
  echo "fetches one." >&2
  exit 1
}

# The library the wheel carries, found through Python rather than guessed: a
# virtual environment can be anywhere and the wheel names the file by version.
LIB_DIR="$(python -c 'import ai_edge_litert, pathlib; print(pathlib.Path(ai_edge_litert.__file__).parent)' 2>/dev/null || true)"
[ -n "$LIB_DIR" ] && [ -f "$LIB_DIR/libLiteRt.dylib" ] || {
  echo "no libLiteRt in the environment. pip install ai-edge-litert, and" >&2
  echo "activate the environment so this script can ask Python where it went." >&2
  exit 1
}

# build_config.h is shipped as a CMake template, and the SDK expects to be
# consumed through CMake. Nothing is disabled, to match the wheel's library:
# these toggles gate code in the headers, so a build that disables what the
# library kept is a struct layout disagreement waiting to happen.
CONFIG="$SDK/litert/build_common/build_config.h"
if [ ! -f "$CONFIG" ]; then
  sed -e 's/#cmakedefine01 \(LITERT_BUILD_CONFIG_DISABLE_[A-Z]*\)/#define \1 0/' \
    "$CONFIG.in" > "$CONFIG"
  echo "generated build_config.h from the SDK's template"
fi

case "$(uname -s)" in
  Darwin) LIB=libfluttorch_litert.dylib; SHARED=(-dynamiclib) ;;
  *)      LIB=libfluttorch_litert.so;    SHARED=(-shared -fPIC) ;;
esac

mkdir -p "$HERE/.dart_tool/native"

c++ -std=c++17 -O2 -Wall -fPIC -c "$HERE/src/fluttorch_litert.cpp" \
  -o "$HERE/.dart_tool/native/shim.o" -I "$SDK"

# rpath rather than a copy, as the ONNX build does: the library belongs to the
# environment, and a duplicate in .dart_tool is one that goes stale silently.
c++ -std=c++17 "${SHARED[@]}" -o "$HERE/.dart_tool/native/$LIB" \
  "$HERE/.dart_tool/native/shim.o" \
  -L "$LIB_DIR" -lLiteRt -Wl,-rpath,"$LIB_DIR"

rm -f "$HERE/.dart_tool/native/shim.o"
echo "built $HERE/.dart_tool/native/$LIB"
