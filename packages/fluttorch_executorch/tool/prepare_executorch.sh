#!/usr/bin/env bash
# Makes a checkout of ExecuTorch buildable for the targets this binding needs.
#
#   tool/prepare_executorch.sh [path-to-executorch]
#
# Everything here works around something upstream, and every one of them was
# found the same way: a build that fails with a message pointing somewhere else.
# They are scripted rather than described because a workaround that lives in a
# comment is a workaround the next clone does not have.
#
# Idempotent. Run it again after updating the checkout, and it will say what it
# had to do and what was already done.
set -euo pipefail

ET="${1:-$HOME/.cache/fluttorch/executorch}"
[ -d "$ET" ] || { echo "no ExecuTorch checkout at $ET" >&2; exit 1; }

changed=0

# ── 1. flatc has to run on the build machine ─────────────────────────────────
#
# flatc is the FlatBuffers compiler, a tool the build runs on the host to
# generate headers. On the iOS path it comes out compiled for iOS, macOS kills
# it with SIGKILL, and what a reader sees is a schema target failing with no
# message at all.
#
# Upstream knows the hazard. third-party/CMakeLists.txt builds flatc through an
# ExternalProject "to force it target the host", unsetting both the toolchain
# file and, for iOS, CMAKE_OSX_SYSROOT. Unsetting the sysroot is the part that
# does not work: Xcode exports SDKROOT=iphoneos to its script phases, and an
# empty CMAKE_OSX_SYSROOT falls back to exactly that.
#
# Measured both ways with SDKROOT set: empty gives `platform IOS`, and
# `macosx` gives `platform MACOS`.
THIRD_PARTY="$ET/third-party/CMakeLists.txt"
if grep 'Fluttorch: pin flatc to the host' "$THIRD_PARTY" >/dev/null 2>&1; then
  echo "flatc: already pinned to the host"
else
  python3 - "$THIRD_PARTY" <<'PY'
import pathlib, sys
p = pathlib.Path(sys.argv[1])
s = p.read_text()
anchor = "    -DCMAKE_OSX_DEPLOYMENT_TARGET:STRING=${CMAKE_OSX_DEPLOYMENT_TARGET}"
# The line appears twice, once for flatbuffers_ep and once for flatcc_ep. Only
# the first builds flatc, so the patch is scoped to that block rather than
# applied to whichever occurrence comes first in the file.
start = s.find("ExternalProject_Add(\n  flatbuffers_ep")
if start < 0 or s.count(anchor) < 1:
    raise SystemExit(
        "third-party/CMakeLists.txt does not look like the version this patch "
        "was written against; check whether upstream has fixed it"
    )
at = s.find(anchor, start)
if at < 0:
    raise SystemExit("flatbuffers_ep no longer carries the line this patches")
head, tail = s[:at], s[at:]
s = head + tail.replace(anchor, anchor + """
    # Fluttorch: pin flatc to the host. Unsetting the sysroot above does not
    # reach it under the Xcode generator, because Xcode exports SDKROOT to its
    # script phases and an empty CMAKE_OSX_SYSROOT resolves to that.
    -DCMAKE_OSX_SYSROOT=macosx
    -DCMAKE_SYSTEM_NAME=Darwin""", 1)
p.write_text(s)
PY
  echo "flatc: pinned to the host"
  changed=1
fi

# ── 2. Core ML needs protobuf sources nobody generated ───────────────────────
#
# backends/apple/coreml/runtime/sdk/format/ holds 33 generated protobuf sources
# the delegate compiles. They are written by
# backends/apple/coreml/scripts/install_requirements.sh, which runs under set -e
# and invokes `python`, so where only python3 is on PATH it exits at its pip
# step and never reaches them. Without the directory the delegate's SDK sources
# are dropped from the CMake target, and libcoremldelegate.a ends up referencing
# ETCoreMLModelAnalyzer, ETCoreMLModelDebugInfo and ModelEventLoggerImpl without
# containing them.
#
# Skipped where coremltools is absent: this repository builds Core ML on the
# machines that have it and reports it as not run everywhere else.
COREML="$ET/backends/apple/coreml"
CT="$COREML/third-party/coremltools"
if [ ! -d "$CT" ]; then
  echo "Core ML: no coremltools checkout, skipping"
elif [ -d "$COREML/runtime/sdk/format" ] &&
     [ "$(find "$COREML/runtime/sdk/format" -name '*.pb.cc' | wc -l | tr -d ' ')" -gt 0 ]; then
  echo "Core ML: protobuf sources already in place"
else
  command -v python >/dev/null 2>&1 || {
    echo "Core ML: needs a 'python' on PATH; activate the environment first" >&2
    exit 1
  }
  echo "Core ML: building coremltools' mlmodel target"
  SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
  export SDKROOT
  export CC="$(xcrun --find clang)" CXX="$(xcrun --find clang++)"
  export CFLAGS="-isysroot $SDKROOT" CXXFLAGS="-isysroot $SDKROOT"
  # -DCMAKE_POLICY_VERSION_MINIMUM for the protobuf it vendors under CMake 4,
  # and -Dprotobuf_BUILD_TESTS=OFF for a googletest it does not ship.
  cmake -S "$CT" -B "$CT/build" \
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -Dprotobuf_BUILD_TESTS=OFF >/dev/null
  cmake --build "$CT/build" --parallel --target mlmodel >/dev/null
  rm -rf "$COREML/runtime/sdk/format/"
  mkdir -p "$COREML/runtime/sdk/format/"
  cp -rf "$CT/build/mlmodel/format/" "$COREML/runtime/sdk/format/"
  echo "Core ML: copied $(find "$COREML/runtime/sdk/format" -name '*.pb.cc' | wc -l | tr -d ' ') protobuf sources"
  changed=1
fi

if [ "$changed" -eq 1 ]; then
  echo
  echo "The checkout changed. Reconfigure before building:"
  echo "  cmake -B <build-dir> -S $ET ..."
fi
