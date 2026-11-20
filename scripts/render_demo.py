"""Run real CLI commands and draw their output as a terminal-style SVG for the README.
Run: python scripts/render_demo.py   (writes docs/demo.svg)"""
import os
import subprocess
import sys
from pathlib import Path
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parent.parent
COMMANDS = [
    ["suggest", "my dear", "--genre", "mystery", "--count", "3"],
    ["suggest", "the old house at the end of the", "-g", "horror", "--explain", "--count", "2"],
    ["compare", "the door opened slowly and", "--length", "14"],
]
lines = []
for args in COMMANDS:
    shown = " ".join(f'"{a}"' if " " in a else a for a in args)
    lines.append(("cmd", f"$ python -m app.cli {shown}"))
    out = subprocess.run([sys.executable, "-m", "app.cli", *args], cwd=ROOT, capture_output=True, text=True,
                         check=True, env={**os.environ, "PYTHONIOENCODING": "utf-8"}).stdout
    lines += [("out", l) for l in out.rstrip("\n").splitlines()]
    lines.append(("out", ""))
lines.pop()

CW, LH, PAD, TOP = 8.4, 19, 18, 44
width = int(PAD * 2 + CW * max(len(t) for _, t in lines)) + 10
height = TOP + LH * len(lines) + PAD
svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
       f'<rect width="{width}" height="{height}" rx="10" fill="#1e1e2e"/>',
       '<circle cx="20" cy="18" r="6" fill="#f38ba8"/><circle cx="40" cy="18" r="6" fill="#f9e2af"/>'
       '<circle cx="60" cy="18" r="6" fill="#a6e3a1"/>',
       '<g font-family="ui-monospace, SFMono-Regular, Menlo, Consolas, monospace" font-size="14" xml:space="preserve">']
for i, (kind, text) in enumerate(lines):
    color = "#89b4fa" if kind == "cmd" else ("#a6adc8" if text.strip().startswith("e.g.") else "#cdd6f4")
    svg.append(f'<text x="{PAD}" y="{TOP + i * LH}" fill="{color}">{escape(text)}</text>')
svg += ["</g>", "</svg>"]
(ROOT / "docs").mkdir(exist_ok=True)
(ROOT / "docs/demo.svg").write_text("\n".join(svg) + "\n", encoding="utf-8")
print(f"docs/demo.svg: {len(lines)} lines, {width}x{height}")
