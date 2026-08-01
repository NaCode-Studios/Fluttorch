#!/usr/bin/env bash
# Builds the native half of the binding against a checkout of ExecuTorch.
#
#   tool/build_native.sh [path-to-executorch]
#   tool/build_native.sh --android [path-to-executorch]
#
# The second cross-compiles for arm64-v8a against a checkout configured with the
# NDK's toolchain file, and writes a .so an APK can carry. It reads
# cmake-out-android rather than cmake-out, because a host build and a device
# build cannot share an output directory and a script that let them would produce
# a library for the wrong architecture without saying so.
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
# Each further delegate is one more flag on that configure, and each is optional:
# -DEXECUTORCH_BUILD_MPS=ON, -DEXECUTORCH_BUILD_METAL=ON,
# -DEXECUTORCH_BUILD_VULKAN=ON, -DEXECUTORCH_BUILD_MLX=ON,
# -DEXECUTORCH_BUILD_QNN=ON. This script links whichever ones the checkout ended
# up with and says so, one line per backend, so a library that turns out to lack
# one says which rather than leaving it to be discovered at load.
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

TARGET=host
if [ "${1:-}" = "--android" ]; then TARGET=android; shift; fi

ET="${1:-$HOME/.cache/fluttorch/executorch}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

# Forcing a static archive to be linked whole is spelled differently by the two
# linkers, and a backend registers itself from a static initialiser that neither
# keeps otherwise. Getting this wrong produces a library that loads and reports
# no backend, which is the failure that looks like a missing model.
#
# It sets an array rather than printing one because macOS ships bash 3.2, where
# readarray does not exist: this script runs on the machine that has the NDK, and
# that is a Mac.
if [ "$TARGET" = android ]; then
  OUT="$ET/cmake-out-android"
  NDK="${ANDROID_NDK:-$(ls -d "$HOME/Library/Android/sdk/ndk"/* 2>/dev/null | sort -V | tail -1)}"
  [ -n "$NDK" ] && [ -d "$NDK" ] || {
    echo "no NDK found; set ANDROID_NDK to one" >&2
    exit 1
  }
  CXX_BIN="$NDK/toolchains/llvm/prebuilt/$(uname -s | tr '[:upper:]' '[:lower:]')-x86_64/bin/aarch64-linux-android26-clang++"
  force_load() { FORCE=(-Wl,--whole-archive "$1" -Wl,--no-whole-archive); }
else
  OUT="$ET/cmake-out"
  CXX_BIN="${CXX:-c++}"
  force_load() { FORCE=(-Wl,-force_load,"$1"); }
fi

[ -f "$OUT/libexecutorch.a" ] || {
  echo "no build under $OUT. Configure and build ExecuTorch first; the header of" >&2
  echo "this script carries the exact cmake invocation." >&2
  exit 1
}

if [ "$TARGET" = android ]; then
  LIB=libfluttorch_executorch.so
  # --no-undefined because a shared object on Android links happily with symbols
  # it cannot resolve and fails at dlopen instead, on the device, in front of a
  # user. -llog is what ExecuTorch's platform layer logs through.
  #
  # -static-libstdc++ puts the C++ runtime inside this library. The NDK's default
  # is the shared one, which then has to be in the APK beside us, and an app that
  # has no other native code has no reason to carry it: without it dlopen fails
  # looking for libc++_shared.so, which reads as our library being missing.
  SHARED=(-shared -fPIC -Wl,--no-undefined -static-libstdc++ -llog)
elif [ "$(uname -s)" = Darwin ]; then
  LIB=libfluttorch_executorch.dylib
  SHARED=(-dynamiclib -framework Accelerate -framework Foundation)
else
  LIB=libfluttorch_executorch.so
  SHARED=(-shared -fPIC)
fi

# Where the library has to land for the target to find it. The host build feeds
# the integration test, which loads it by path. The Android build feeds an APK,
# and Gradle packages jniLibs only for a plugin, which is what
# fluttorch_executorch_flutter is for.
if [ "$TARGET" = android ]; then
  DEST="$HERE/../fluttorch_executorch_flutter/android/src/main/jniLibs/arm64-v8a"
else
  DEST="$HERE/.dart_tool/native"
fi
mkdir -p "$DEST"

# Whether this checkout can supply Core ML, asked of the archive rather than of
# the cache. A build configured without devtools produces a libcoremldelegate.a
# that links only against sources it does not contain, so the question worth
# asking is not whether the file is there but whether it carries its SDK half.
COREML="$OUT/backends/apple/coreml"
DEFINES=(-DC10_USING_CUSTOM_GENERATED_MACROS)
force_load "$OUT/backends/xnnpack/libxnnpack_backend.a"
BACKENDS=("${FORCE[@]}")

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
  force_load "$COREML/libcoremldelegate.a"
  BACKENDS+=(
    "${FORCE[@]}"
    "$COREML/libcoreml_util.a"
    "$COREML/libcoreml_inmemoryfs.a"
    "$COREML/third-party/coremltools/deps/protobuf/cmake/libprotobuf-lite.a"
  )
  SHARED+=(-framework CoreML -lsqlite3)
  echo "linking Core ML"
else
  echo "no Core ML under $OUT"
fi

# The rest of the delegates, each linked when the checkout built it and skipped
# when it did not. Skipping is the designed outcome and not a degraded one: a
# machine that never built Vulkan should produce a library that says it cannot
# run Vulkan, rather than one that fails to link or claims a backend it lacks.
#
# The frameworks each needs are its own. MPS and Metal draw on Apple's, and both
# are absent everywhere else, which the Darwin test above already governs.
link_backend() {
  local name="$1" archive="$2" define="$3"
  shift 3
  if [ ! -f "$archive" ]; then
    echo "no $name under $OUT"
    return
  fi
  DEFINES+=("$define")
  force_load "$archive"
  BACKENDS+=("${FORCE[@]}")
  if [ "$#" -gt 0 ]; then SHARED+=("$@"); fi
  echo "linking $name"
}

if [ "$(uname -s)" = "Darwin" ]; then
  link_backend MPS "$OUT/backends/apple/mps/libmpsdelegate.a" \
    -DFLUTTORCH_WITH_MPS -framework MetalPerformanceShaders -framework MetalPerformanceShadersGraph -framework Metal
  link_backend Metal "$OUT/backends/apple/metal/libmetal_backend.a" \
    -DFLUTTORCH_WITH_METAL -framework Metal
fi
link_backend Vulkan "$OUT/backends/vulkan/libvulkan_backend.a" -DFLUTTORCH_WITH_VULKAN
link_backend MLX "$OUT/backends/mlx/libmlx_backend.a" -DFLUTTORCH_WITH_MLX
link_backend QNN "$OUT/backends/qualcomm/libqnn_executorch_backend.a" -DFLUTTORCH_WITH_QNN

# The parent of the checkout, because ExecuTorch's own headers include each other
# as executorch/... and the checkout directory is that "executorch".
"$CXX_BIN" -std=c++17 -O2 -fPIC -c "$HERE/src/fluttorch_executorch.cpp" -o "$DEST/shim.o" \
  -I "$HERE/src" -I "$(dirname "$ET")" -I "$ET/runtime/core/portable_type/c10" \
  "${DEFINES[@]}"

# A backend registers itself from a static initialiser, which a linker drops from
# a static library nobody references by symbol. force_load is what keeps it, and
# without it the library loads and reports no backend at all.
force_load "$OUT/kernels/portable/libportable_ops_lib.a"
OPS=("${FORCE[@]}")

# flatcc is built by an ExternalProject that does not cross-compile, so the
# archive under a device build is a host one the device linker skips with a
# warning. Nothing in these targets references it, and asking for it anyway
# would trade a real warning for a confusing one.
FLATCC=()
[ "$TARGET" = host ] && FLATCC=("$OUT/third-party/flatcc_ep/lib/libflatccrt.a")

"$CXX_BIN" -std=c++17 "${SHARED[@]}" -o "$DEST/$LIB" "$DEST/shim.o" \
  "${BACKENDS[@]}" \
  "${OPS[@]}" \
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
  "$OUT/kleidiai/libkleidiai.a" ${FLATCC[@]+"${FLATCC[@]}"}

rm -f "$DEST/shim.o"
echo "built $DEST/$LIB ($TARGET)"
