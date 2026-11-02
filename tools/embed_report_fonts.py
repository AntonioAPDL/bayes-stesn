#!/usr/bin/env python3
"""Re-render a verified report PDF with embedded fonts, without refitting.

The saved HTML fixes figure order/titles and embeds each PNG. Every PNG must
match an already inventoried local file. Only the PDF and its inventory entry
change; simulations, forecasts, scores, figures, and movies remain untouched.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import tempfile


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FigureOrder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_title = False
        self.title = []
        self.figures = []

    def handle_starttag(self, tag, attrs):
        if tag == "h2":
            self.in_title = True
            self.title = []
        if tag == "img":
            value = dict(attrs).get("src", "")
            prefix = "data:image/png;base64,"
            if not value.startswith(prefix):
                raise ValueError("Report figures must be embedded PNGs, never external URLs")
            payload = base64.b64decode(value[len(prefix):], validate=True)
            self.figures.append(("".join(self.title), hashlib.sha256(payload).hexdigest()))

    def handle_endtag(self, tag):
        if tag == "h2":
            self.in_title = False

    def handle_data(self, data):
        if self.in_title:
            self.title.append(data)


def embed_fonts(directory):
    directory = Path(directory).resolve()
    receipt_path = directory / "report_manifest.json"
    receipt = json.loads(receipt_path.read_text())
    if not receipt.get("render_only"):
        raise ValueError("An existing saved-artifact report is required")
    indexed = {}
    seen = set()
    for entry in receipt["files"]:
        name = Path(entry["path"])
        path = directory / name
        if name.is_absolute() or ".." in name.parts or path.is_symlink() or str(name) in seen:
            raise ValueError("Unsafe report inventory path")
        seen.add(str(name))
        if not path.resolve().is_relative_to(directory):
            raise ValueError("Report inventory leaves its directory")
        if path.stat().st_size != entry["bytes"] or digest(path) != entry["sha256"]:
            raise ValueError(f"Saved report payload changed: {name}")
        if path.suffix == ".png":
            indexed.setdefault(entry["sha256"], path)
    pdf_entry = next((entry for entry in receipt["files"]
                      if entry["path"] == "benchmark_report.pdf"), None)
    if pdf_entry is None:
        raise ValueError("The original inventory did not contain the report PDF")
    parser = FigureOrder()
    parser.feed((directory / "report.html").read_text())
    if not parser.figures:
        raise ValueError("No verified report figures were found")
    figures = [(title, indexed[value]) for title, value in parser.figures]
    summary = json.loads((directory / "summary.json").read_text())
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if receipt["run_identity"] != manifest["identity"] or summary["manifest_identity"] != manifest["identity"]:
        raise ValueError("Report input identities do not agree")

    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "bstesn-mpl"))
    import matplotlib
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from bayes_stesn.benchmark.report import _pdf_report
    from bayes_stesn.benchmark.runner import write_json

    fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    families = {
        "Helvetica": ("DejaVuSans", "Oblique", "BoldOblique"),
        "Courier": ("DejaVuSansMono", "Oblique", "BoldOblique"),
        "Times": ("DejaVuSerif", "Italic", "BoldItalic"),
    }
    used = []
    for family, (stem, italic, bold_italic) in families.items():
        normal_name = "Times-Roman" if family == "Times" else family
        aliases = (normal_name, family + "-Bold", family + "-Italic", family + "-BoldItalic")
        suffixes = ("", "-Bold", "-" + italic, "-" + bold_italic)
        for alias, suffix in zip(aliases, suffixes):
            path = fonts / (stem + suffix + ".ttf")
            pdfmetrics.registerFont(TTFont(alias, str(path)))
            used.append({"alias": alias, "file": path.name, "sha256": digest(path)})
        pdfmetrics.registerFontFamily(normal_name, normal=aliases[0], bold=aliases[1],
                                      italic=aliases[2], boldItalic=aliases[3])

    target = directory / "benchmark_report.pdf"
    temporary = directory / ".benchmark_report_embedded.partial.pdf"
    try:
        _pdf_report(temporary, summary, manifest, figures)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)
    pdf_entry.update(bytes=target.stat().st_size, sha256=digest(target))
    receipt["typography"] = {
        "embedded": True,
        "font_source": "Matplotlib's bundled DejaVu TrueType fonts",
        "matplotlib_version": matplotlib.__version__,
        "fonts": used,
        "figure_order_source": "Verified saved HTML and inventoried PNGs",
        "numerical_artifacts_changed": False,
        "reproduce": "python tools/embed_report_fonts.py " + str(directory),
    }
    write_json(receipt_path, receipt)
    return {"pdf": str(target), "figures": len(figures), "fonts_embedded": True,
            "run_identity": receipt["run_identity"], "bytes": target.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    print(json.dumps(embed_fonts(args.report), indent=2))


if __name__ == "__main__":
    main()
