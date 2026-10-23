#!/bin/bash
# Build the v2 paper outside the Hebrew path (tectonic fails on it), check the page limit,
# and copy the PDF back to paper/main.pdf.
# Usage (Git Bash): bash scripts/build_paper.sh
# OK iff it prints "References heading on page 5 at y=73" (main text = exactly 4 pages).
TECTONIC=${TECTONIC:-/c/Users/riden/tools/tectonic/tectonic.exe}
FITZ_PY=${FITZ_PY:-/c/Users/riden/AppData/Local/Programs/Python/Python311/python.exe}  # has PyMuPDF (fitz)
BUILD=${BUILD:-/c/Users/riden/tools/paper_build_v2}
SRC="$(cd "$(dirname "$0")/../paper" && pwd)"
rm -rf "$BUILD" && cp -r "$SRC" "$BUILD" && cd "$BUILD" || exit 1
"$TECTONIC" -X compile main.tex -Z continue-on-errors --keep-logs 2>&1 | grep -E "^error|warning: .*(undefined|Overfull)|^!" | head -20
grep -E "^!|Overfull|undefined" main.log | head -20
"$FITZ_PY" - <<'PY'
import fitz
d = fitz.open("main.pdf")
print("pages", len(d))
for i, p in enumerate(d):
    p.get_pixmap(dpi=110).save(f"p{i+1}.png")
    for b in p.get_text("blocks"):
        if b[4].startswith("References"):
            print(f"References heading on page {i+1} at y={b[1]:.0f} of {p.rect.height:.0f}")
PY
cp main.pdf "$SRC/main.pdf" && echo "copied to $SRC/main.pdf (page renders: $BUILD/p*.png)"
