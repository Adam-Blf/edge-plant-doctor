"""Render the PWA icons from the Reicon "Leaf" glyph (Outline weight) with Playwright.

Set CHROME_PATH to a Chromium executable when Playwright's own browser is not installed.
"""

import os
import re
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
GLYPH_JS = ROOT / "app" / "node_modules" / "reicon" / "icons" / "Leaf.js"
OUT = ROOT / "app" / "public" / "icons"
BACKGROUND, INK = "#1f6b3a", "#f3f0e6"
TARGETS = {
    "apple-touch-icon.png": 180,
    "icon-192.png": 192,
    "icon-512.png": 512,
    "icon-maskable-512.png": 512,
}


def outline_paths() -> str:
    match = re.search(r"O:\s*`([^`]+)`", GLYPH_JS.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError("Outline weight not found in Leaf.js")
    return match.group(1)


def page(size: int, paths: str) -> str:
    glyph = round(size * 0.56)
    return f"""<body style="margin:0;width:{size}px;height:{size}px;background:{BACKGROUND};
display:grid;place-items:center"><svg xmlns="http://www.w3.org/2000/svg" width="{glyph}"
height="{glyph}" viewBox="0 0 24 24" fill="none" style="color:{INK}">{paths}</svg></body>"""


def main() -> None:
    paths = outline_paths()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=os.environ.get("CHROME_PATH"))
        for name, size in TARGETS.items():
            tab = browser.new_page(viewport={"width": size, "height": size})
            tab.set_content(page(size, paths))
            tab.screenshot(path=str(OUT / name))
            tab.close()
        browser.close()


if __name__ == "__main__":
    main()
