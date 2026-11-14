#!/bin/bash
# Build a patched MuJoCo python wheel that adds `Simulate.mouse` — the cursor state
# mjstudio polls to make its overlay panels mouse-draggable. The MuJoCo core library is
# NOT rebuilt: it is linked from your already-installed `mujoco` package, so the patched
# `_simulate` extension is ABI-compatible with the rest of that install.
#
# Usage:
#   scripts/build_patched_mujoco.sh [MUJOCO_VERSION]
#
# Requirements (in the active venv / on PATH):
#   - a working `python` with: pip, build, cmake, ninja   (uv venvs: `uv pip install pip build cmake ninja`)
#   - a C++ toolchain (Xcode CLT on macOS)
#   - network access (the build FetchContent's glfw/Eigen/lodepng/pybind11 from GitHub)
#
# Output (in mjstudio/dist/):
#   - mujoco_core-<ver>-...whl          the full patched wheel
#   - _simulate.cpython-*.so            just the patched extension (drop-in over an install)
#
# Sync with upstream MuJoCo: bump the submodule (or MUJOCO_VERSION) to a new tag, then
#   `git -C third_party/mujoco apply --3way patches/expose-sim-mouse.patch`
# and regenerate the patch if it drifted.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # mjstudio/
VER="${1:-$(cat "$HERE/patches/MUJOCO_VERSION")}"
PATCH="$HERE/patches/expose-sim-mouse.patch"

# CMake chokes on spaces in paths, so do all work in a space-free temp dir.
WORK="$(mktemp -d)"; trap 'rm -rf "$WORK"' EXIT
echo ">> workdir=$WORK  mujoco=$VER"

# 1. Source: prefer the vendored submodule; otherwise clone the pinned tag.
if [[ -f "$HERE/third_party/mujoco/python/mujoco/simulate.cc" ]]; then
  cp -R "$HERE/third_party/mujoco" "$WORK/mujoco"
else
  echo ">> submodule missing; cloning $VER"
  git clone --depth 1 --branch "$VER" https://github.com/google-deepmind/mujoco.git "$WORK/mujoco"
fi
SRC="$WORK/mujoco"

# 2. Apply the patch (skip cleanly if it is already in the tree).
if git -C "$SRC" apply --check "$PATCH" 2>/dev/null; then
  git -C "$SRC" apply "$PATCH"; echo ">> patch applied"
else
  echo ">> patch not applicable (already applied?), continuing"
fi

# 3. MUJOCO_PATH: copy the installed mujoco core lib+headers+plugins to a space-free dir.
PKG="$(python -c 'import mujoco,os;print(os.path.dirname(mujoco.__file__))')"
MJP="$WORK/mjpath"; mkdir -p "$MJP/include" "$MJP/plugin"
cp "$PKG"/libmujoco*.dylib "$MJP/" 2>/dev/null || cp "$PKG"/libmujoco*.so* "$MJP/" 2>/dev/null || true
cp -R "$PKG/include/mujoco" "$MJP/include/mujoco"
cp "$PKG/plugin/"* "$MJP/plugin/" 2>/dev/null || true
export MUJOCO_PATH="$MJP" MUJOCO_PLUGIN_PATH="$MJP/plugin"
export CMAKE_BUILD_PARALLEL_LEVEL="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"

# 4. Build the sdist (bundles the patched simulate.cc), then the wheel.
( cd "$SRC/python" && bash make_sdist.sh )
SDIST="$(ls "$SRC/python/dist/"*.tar.gz | head -1)"
OUT="$HERE/dist"; mkdir -p "$OUT"
pip wheel "$SDIST" -w "$OUT" --no-deps --no-build-isolation

# 5. Also extract the single patched _simulate extension for drop-in reuse.
WHL="$(ls "$OUT"/mujoco_core-*.whl | head -1)"
python - "$WHL" "$OUT" <<'PY'
import os, sys, zipfile
whl, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(whl) as z:
    for n in z.namelist():
        if os.path.basename(n).startswith("_simulate") and n.endswith(".so"):
            src = z.open(n)
            dst = os.path.join(out, os.path.basename(n))
            open(dst, "wb").write(src.read())
            print("extracted", dst)
PY
echo ">> done -> $OUT"
