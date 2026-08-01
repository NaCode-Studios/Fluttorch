#!/usr/bin/env bash
# Builds the native half of the binding against a checkout of ExecuTorch.
#
#   tool/build_native.sh [path-to-executorch]
#
# Writes .dart_tool/native/libfluttorch_executorch.{dylib,so}, which is what the
# integration test loads. Nothing else in the repository needs it: the unit
# suites run against a C stub the test compiles itself, so a contributor without
# ExecuTorch still gets a green build and a red one still means something.
#
# The ExecuTorch checkout must be configured and built first:
#
#   cmake -B cmake-out -DCMAKE_BUILD_TYPE=Release \
#     -DEXECUTORCH_BUILD_XNNPACK=ON \
#     -DEXECUTORCH_BUILD_EXTENSION_MODULE=ON \
#     -DEXECUTORCH_BUILD_EXTENSION_NAMED_DATA_MAP=ON \
#     -DEXECUTORCH_BUILD_EXTENSION_TENSOR=ON \
#     -DEXECUTORCH_BUILD_EXTENSION_DATA_LOADER=ON \
#     -DEXECUTORCH_BUILD_EXTENSION_FLAT_TENSOR=ON
#   cmake --build cmake-out -j
#
# Core ML is linked when the checkout carries it and skipped when it does not, so
# a contributor who has only built XNNPACK still gets a library. Getting it means
# adding -DEXECUTORCH_BUILD_COREML=ON -DEXECUTORCH_BUILD_DEVTOOLS=ON to the
# configure above, and four steps upstream documents nowhere.
#
# The coremltools submodule is referenced by CMakeLists and absent from
# .gitmodules, so it arrives only through
# backends/apple/coreml/scripts/install_requirements.sh. That script runs under
# `set -e` and invokes `python`, so where only `python3` is on PATH it dies at its
# pip step, short of the three commands that matter: configure coremltools, build
# its mlmodel target, copy the protobuf sources that produces into
# backends/apple/coreml/runtime/sdk/format/. Activating the environment supplies
# the `python` it wants. Its vendored protobuf then needs
# -DCMAKE_POLICY_VERSION_MINIMUM=3.5 under CMake 4 and -Dprotobuf_BUILD_TESTS=OFF
# for a googletest it does not ship.
#
# That format/ directory is the whole thing. Without it the delegate's SDK sources
# are dropped from the target and libcoremldelegate.a references
# ETCoreMLModelAnalyzer, ETCoreMLModelDebugInfo and ModelEventLoggerImpl without
# containing them, which reads like the EXECUTORCH_BUILD_DEVTOOLS branch failing
# to fire and is not: the branch runs, and hands it sources that are not there.
set -euo pipefail

ET="${1:-$HOME/.cache/fluttorch/executorch}"
OUT="$ET/cmake-out"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

[ -f "$OUT/libexecutorch.a" ] || {
  echo "no build under $OUT. Configure and build ExecuTorch first; the header of" >&2
  echo "this script carries the exact cmake invocation." >&2
  exit 1
}

case "$(uname -s)" in
  Darwin) LIB=libfluttorch_executorch.dylib; SHARED=(-dynamiclib -framework Accelerate -framework Foundation) ;;
  *)      LIB=libfluttorch_executorch.so;    SHARED=(-shared -fPIC) ;;
esac

mkdir -p "$HERE/.dart_tool/native"

# Whether this checkout can supply Core ML, asked of the archive rather than of
# the cache. A build configured without devtools produces a libcoremldelegate.a
# that links only against sources it does not contain, so the question worth
# asking is not whether the file is there but whether it carries its SDK half.
COREML="$OUT/backends/apple/coreml"
DEFINES=(-DC10_USING_CUSTOM_GENERATED_MACROS)
BACKENDS=(-Wl,-force_load,"$OUT/backends/xnnpack/libxnnpack_backend.a")

# Activation taps need the tracer hooks compiled into ExecuTorch itself, which
# -DEXECUTORCH_BUILD_DEVTOOLS=ON is what turns on. The hooks are #ifdef'd in the
# executor, so a shim compiled with the flag against a runtime built without it
# would collect nothing and report every layer absent.
#
# Asked of the flags the runtime was compiled with rather than of its symbols,
# because what the flag controls is a call inside an existing function: the
# tracer hooks are header inlines and no symbol appears or disappears with them.
ETCORE_FLAGS="$OUT/CMakeFiles/executorch_core.dir/flags.make"
if [ -f "$ETCORE_FLAGS" ] &&
   grep ET_EVENT_TRACER_ENABLED "$ETCORE_FLAGS" >/dev/null; then
  DEFINES+=(-DFLUTTORCH_WITH_TAPS -DET_EVENT_TRACER_ENABLED)
  echo "linking activation taps"
else
  echo "no event tracer in $OUT; taps will report absent"
fi

# grep reads to the end rather than stopping at the first hit: under pipefail a
# `grep -q` that exits early leaves nm killed by SIGPIPE, and the test reports no
# Core ML on a checkout that has it.
if [ "$(uname -s)" = "Darwin" ] && [ -f "$COREML/libcoremldelegate.a" ] &&
   nm -g --defined-only "$COREML/libcoremldelegate.a" 2>/dev/null |
     grep ETCoreMLModelAnalyzer >/dev/null; then
  DEFINES+=(-DFLUTTORCH_WITH_COREML)
  BACKENDS+=(
    -Wl,-force_load,"$COREML/libcoremldelegate.a"
    "$COREML/libcoreml_util.a"
    "$COREML/libcoreml_inmemoryfs.a"
    "$COREML/third-party/coremltools/deps/protobuf/cmake/libprotobuf-lite.a"
  )
  SHARED+=(-framework CoreML -lsqlite3)
  echo "linking Core ML"
else
  echo "no Core ML under $OUT; building XNNPACK alone"
fi

# The parent of the checkout, because ExecuTorch's own headers include each other
# as executorch/... and the checkout directory is that "executorch".
c++ -std=c++17 -O2 -c "$HERE/src/fluttorch_executorch.cpp" -o "$HERE/.dart_tool/native/shim.o" \
  -I "$HERE/src" -I "$(dirname "$ET")" -I "$ET/runtime/core/portable_type/c10" \
  "${DEFINES[@]}"

# A backend registers itself from a static initialiser, which a linker drops from
# a static library nobody references by symbol. force_load is what keeps it, and
# without it the library loads and reports no backend at all.
c++ -std=c++17 "${SHARED[@]}" -o "$HERE/.dart_tool/native/$LIB" "$HERE/.dart_tool/native/shim.o" \
  "${BACKENDS[@]}" \
  -Wl,-force_load,"$OUT/kernels/portable/libportable_ops_lib.a" \
  "$OUT/libexecutorch.a" "$OUT/libexecutorch_core.a" \
  "$OUT/extension/module/libextension_module_static.a" \
  "$OUT/extension/data_loader/libextension_data_loader.a" \
  "$OUT/extension/tensor/libextension_tensor.a" \
  "$OUT/extension/flat_tensor/libextension_flat_tensor.a" \
  "$OUT/extension/named_data_map/libextension_named_data_map.a" \
  "$OUT/extension/threadpool/libextension_threadpool.a" \
  "$OUT/kernels/portable/libportable_kernels.a" \
  "$OUT/kernels/portable/cpu/util/libkernels_util_all_deps.a" \
  "$OUT/backends/xnnpack/third-party/XNNPACK/libXNNPACK.a" \
  "$OUT/backends/xnnpack/third-party/XNNPACK/libxnnpack-microkernels-prod.a" \
  "$OUT/backends/xnnpack/third-party/cpuinfo/libcpuinfo.a" \
  "$OUT/backends/xnnpack/third-party/pthreadpool/libpthreadpool.a" \
  "$OUT/kleidiai/libkleidiai.a" \
  "$OUT/third-party/flatcc_ep/lib/libflatccrt.a"

rm -f "$HERE/.dart_tool/native/shim.o"
echo "built $HERE/.dart_tool/native/$LIB"
