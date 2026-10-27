#!/usr/bin/env bash
# The pip OpenCV wheel ships a Qt5 highgui backend but no fonts, so every
# cv2.imshow() prints "QFontDatabase: Cannot find font directory ...".
# Point that directory at the system DejaVu fonts to silence it.
#
# Re-run this after any `uv sync` that reinstalls opencv (the .venv is not
# tracked in git, so the symlinks go away with it).
set -euo pipefail

cv2_dir="$(uv run python -c 'import cv2, os; print(os.path.dirname(cv2.__file__))')"
font_dir="$cv2_dir/qt/fonts"
src="/usr/share/fonts/truetype/dejavu"

if [[ ! -d "$src" ]]; then
    echo "DejaVu fonts not found at $src — install with: sudo apt install fonts-dejavu-core" >&2
    exit 1
fi

mkdir -p "$font_dir"
ln -sf "$src"/*.ttf "$font_dir"/
echo "linked $(ls "$font_dir" | wc -l) fonts into $font_dir"
