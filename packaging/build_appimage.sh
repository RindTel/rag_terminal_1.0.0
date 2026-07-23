#!/usr/bin/env bash
# Build a QwenRAG AppImage from the PyInstaller onedir (dist/QwenRAG/).
#
#   bash packaging/build_appimage.sh [output.AppImage]
#
# Produces a single double-clickable file (users still `chmod +x` it once).
# Runs in CI on ubuntu-latest; not validated on the dev box (no appimagetool).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/QwenRAG-linux-x86_64.AppImage}"
APPDIR="$ROOT/AppDir"

[ -d "$ROOT/dist/QwenRAG" ] || { echo "dist/QwenRAG missing — run PyInstaller first"; exit 1; }

rm -rf "$APPDIR"
mkdir -p "$APPDIR/usr/bin"
cp -r "$ROOT/dist/QwenRAG/." "$APPDIR/usr/bin/"

# Launcher: resolve the AppImage's own dir and exec the bundled binary.
cat > "$APPDIR/AppRun" <<'EOF'
#!/bin/bash
HERE="$(dirname "$(readlink -f "$0")")"
exec "$HERE/usr/bin/QwenRAG" "$@"
EOF
chmod +x "$APPDIR/AppRun"

cat > "$APPDIR/QwenRAG.desktop" <<'EOF'
[Desktop Entry]
Type=Application
Name=QwenRAG
Comment=Local document chat — fully offline
Exec=QwenRAG
Icon=qwenrag
Categories=Office;Utility;
Terminal=false
EOF

# Placeholder icon (1x1 PNG). Replace packaging/qwenrag.png with a real icon later.
if [ -f "$ROOT/packaging/qwenrag.png" ]; then
  cp "$ROOT/packaging/qwenrag.png" "$APPDIR/qwenrag.png"
else
  python3 - "$APPDIR/qwenrag.png" <<'PY'
import sys, base64
open(sys.argv[1], "wb").write(base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="))
PY
fi

# appimagetool (no FUSE needed with --appimage-extract-and-run).
TOOL="/tmp/appimagetool-x86_64.AppImage"
if [ ! -x "$TOOL" ]; then
  curl -sSL -o "$TOOL" \
    https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage
  chmod +x "$TOOL"
fi

ARCH=x86_64 "$TOOL" --appimage-extract-and-run "$APPDIR" "$OUT"
echo "AppImage -> $OUT"
