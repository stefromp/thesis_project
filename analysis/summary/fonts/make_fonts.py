#!/usr/bin/env python3
"""
Builds the thesis figure fonts (ThesisCMSerif-*.ttf) from Computer Modern Unicode (CMU).

    python3 analysis/summary/fonts/make_fonts.py <path to cm-unicode/fonts/otf>
    # cm-unicode from CTAN: https://mirrors.ctan.org/fonts/cm-unicode.zip

Why: matplotlib embeds fonts in PDF as TrueType (pdf.fonttype 42) and cannot do so correctly
for CMU's CFF-flavoured .otf files. This converts the three faces the figures use to TrueType
outlines (cubic -> quadratic, fontTools cu2qu), subsets them to Latin, Greek and the math /
punctuation symbols the figures print, and renames the family: CMU is under the SIL Open Font
Licence with the Reserved Font Family Name "Computer Modern Unicode fonts", so a modified
version must carry another name (OFL.txt, condition 3). Faces:
    cmunrm (CMU Serif Roman)          -> ThesisCMSerif-Regular.ttf
    cmunbx (CMU Serif Bold Extended)  -> ThesisCMSerif-Bold.ttf     (= LaTeX's \\textbf)
    cmunti (CMU Serif Italic)         -> ThesisCMSerif-Italic.ttf
"""
from __future__ import annotations

import os
import sys

from fontTools import subset
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont, newTable

HERE = os.path.dirname(os.path.abspath(__file__))
FAMILY = "Thesis CM Serif"
FACES = {"cmunrm": ("Regular", 400), "cmunbx": ("Bold", 700), "cmunti": ("Italic", 400)}
UNICODES = (list(range(0x20, 0x7F)) + list(range(0xA0, 0x180)) + list(range(0x370, 0x400))
            + list(range(0x2000, 0x2070)) + list(range(0x2070, 0x20A0)) + list(range(0x2190, 0x2200))
            + list(range(0x2200, 0x2300)))


def to_truetype(f: TTFont) -> None:
    gs, order = f.getGlyphSet(), f.getGlyphOrder()
    glyf = newTable("glyf")
    glyf.glyphOrder, glyf.glyphs = order, {}
    for g in order:
        pen = TTGlyphPen(gs)
        gs[g].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyf.glyphs[g] = pen.glyph()
    f["glyf"], f["loca"] = glyf, newTable("loca")
    m = f["maxp"]
    m.tableVersion = 0x00010000
    for k in ("maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs", "maxStackElements",
              "maxSizeOfInstructions", "maxComponentElements", "maxComponentDepth"):
        setattr(m, k, 0)
    m.maxZones = 1
    del f["CFF "]
    if "VORG" in f:
        del f["VORG"]
    f["head"].glyphDataFormat = 0
    f["post"].formatType, f["post"].extraNames, f["post"].mapping = 2.0, [], {}
    f.sfntVersion = "\x00\x01\x00\x00"


def rename(f: TTFont, style: str, weight: int) -> None:
    ps = FAMILY.replace(" ", "") + "-" + style
    names = {1: FAMILY if style != "Bold" else FAMILY, 2: style, 3: f"{ps};derived from CMU", 4: f"{FAMILY} {style}",
             6: ps, 16: FAMILY, 17: style}
    tbl = f["name"]
    tbl.names = [n for n in tbl.names if n.nameID not in names]
    for nid, s in names.items():
        tbl.setName(s, nid, 3, 1, 0x409)
        tbl.setName(s, nid, 1, 0, 0)
    f["OS/2"].usWeightClass = weight
    f["OS/2"].usWidthClass = 5                      # normal width, so matplotlib matches "bold"
    f["OS/2"].fsSelection = (0x20 if style == "Bold" else 0x01 if style == "Italic" else 0x40)
    f["head"].macStyle = {"Bold": 1, "Italic": 2}.get(style, 0)


def main(src: str) -> None:
    for name, (style, weight) in FACES.items():
        f = TTFont(os.path.join(src, f"{name}.otf"))
        opts = subset.Options()
        opts.name_IDs, opts.name_languages, opts.notdef_outline, opts.layout_features = ["*"], ["*"], True, ["*"]
        s = subset.Subsetter(opts)
        s.populate(unicodes=UNICODES)
        s.subset(f)
        to_truetype(f)
        rename(f, style, weight)
        out = os.path.join(HERE, f"ThesisCMSerif-{style}.ttf")
        f.save(out)
        print(f"{name}.otf -> {os.path.basename(out)} ({os.path.getsize(out) // 1024} KiB)")


if __name__ == "__main__":
    main(sys.argv[1])
