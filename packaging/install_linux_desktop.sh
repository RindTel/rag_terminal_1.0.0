#!/usr/bin/env bash
# Install the built QwenRAG onedir as a desktop app for the CURRENT user — it then
# shows up in rofi / your app menu and launches on click. No sudo, no webkit2gtk:
# clicking starts the bundled server and opens it in an app-mode browser window
# (Chromium/Brave/Chrome `--app=`), falling back to your default browser.
#
#   bash packaging/install_linux_desktop.sh [path-to-dist/QwenRAG]
#
# Re-run any time to update the installed copy.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${1:-$(cd "$HERE/.." && pwd)/dist/QwenRAG}"
[ -x "$SRC/QwenRAG" ] || { echo "Bundle not found at $SRC — run PyInstaller first."; exit 1; }

OPT="$HOME/.local/opt/QwenRAG"
APPS="$HOME/.local/share/applications"
ICONS="$HOME/.local/share/icons"
BIN="$HOME/.local/bin"
mkdir -p "$HOME/.local/opt" "$APPS" "$ICONS" "$BIN"

echo "Installing bundle -> $OPT (this copies ~2.6 GB) ..."
if [ "$SRC" != "$OPT" ]; then
  rm -rf "$OPT"
  cp -r "$SRC" "$OPT"
fi

# Launcher: singleton server on a fixed loopback port, opened in an app-mode window.
cat > "$OPT/qwenrag-launch.sh" <<'EOF'
#!/usr/bin/env bash
APP="$HOME/.local/opt/QwenRAG/QwenRAG"
PORT="${QWENRAG_PORT:-8531}"
URL="http://127.0.0.1:$PORT"
LOG="$HOME/.local/share/QwenRAG/server.log"
mkdir -p "$(dirname "$LOG")"
# Start the server only if it isn't already up (repeated clicks reuse it).
if ! curl -sf "$URL/_stcore/health" >/dev/null 2>&1; then
  QWENRAG_STREAMLIT_WORKER=1 QWENRAG_PORT="$PORT" setsid "$APP" >"$LOG" 2>&1 &
  for _ in $(seq 1 90); do
    curl -sf "$URL/_stcore/health" >/dev/null 2>&1 && break
    sleep 1
  done
fi
# Prefer a chromeless app window; fall back to the default browser.
for b in chromium chromium-browser google-chrome-stable google-chrome brave brave-browser vivaldi-stable microsoft-edge; do
  command -v "$b" >/dev/null 2>&1 && exec "$b" --app="$URL" --class=QwenRAG
done
exec xdg-open "$URL"
EOF
chmod +x "$OPT/qwenrag-launch.sh"
ln -sf "$OPT/qwenrag-launch.sh" "$BIN/qwenrag"

[ -f "$HERE/qwenrag.png" ] && cp "$HERE/qwenrag.png" "$ICONS/qwenrag.png"

cat > "$APPS/qwenrag.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=QwenRAG
GenericName=Local Document Chat
Comment=Chat with your documents — fully offline
Exec=$OPT/qwenrag-launch.sh
Icon=$ICONS/qwenrag.png
Terminal=false
StartupWMClass=QwenRAG
Categories=Utility;
Keywords=rag;llm;documents;chat;ai;offline;
EOF

command -v update-desktop-database >/dev/null 2>&1 && update-desktop-database "$APPS" 2>/dev/null || true
echo "Done. Open rofi (drun) or your app menu and search 'QwenRAG'. CLI: 'qwenrag'."
