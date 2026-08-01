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
# Core ML is not linked, and the reason is upstream rather than a decision here.
# Building it in ExecuTorch 1.3.0 takes four steps the documentation does not
# mention: the coremltools submodule is referenced by CMakeLists and absent from
# .gitmodules, so it arrives only through
# backends/apple/coreml/scripts/install_requirements.sh, which itself invokes
# `python` and fails where only `python3` is on PATH; the protobuf it vendors
# needs -DCMAKE_POLICY_VERSION_MINIMUM=3.5 under CMake 4 and
# -Dprotobuf_BUILD_TESTS=OFF for a googletest it does not ship. After all four,
# libcoremldelegate.a still references ETCoreMLModelAnalyzer,
# ETCoreMLModelDebugInfo and ModelEventLoggerImpl without containing them:
# EXECUTORCH_BUILD_DEVTOOLS=ON is in the cache and the generated build.make for
# that target carries no SDK source at all. The archive cannot be linked into a
# shared library until that is fixed or those sources are compiled by hand.
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

# The parent of the checkout, because ExecuTorch's own headers include each other
# as executorch/... and the checkout directory is that "executorch".
c++ -std=c++17 -O2 -c "$HERE/src/fluttorch_executorch.cpp" -o "$HERE/.dart_tool/native/shim.o" \
  -I "$HERE/src" -I "$(dirname "$ET")" -I "$ET/runtime/core/portable_type/c10" \
  -DC10_USING_CUSTOM_GENERATED_MACROS

# A backend registers itself from a static initialiser, which a linker drops from
# a static library nobody references by symbol. force_load is what keeps it, and
# without it the library loads and reports no backend at all.
c++ -std=c++17 "${SHARED[@]}" -o "$HERE/.dart_tool/native/$LIB" "$HERE/.dart_tool/native/shim.o" \
  -Wl,-force_load,"$OUT/backends/xnnpack/libxnnpack_backend.a" \
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
