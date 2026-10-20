# Thesis figure fonts

`ThesisCMSerif-{Regular,Bold,Italic}.ttf` are Computer Modern Unicode (CMU Serif Roman, Bold
Extended = LaTeX's `\textbf`, Italic) from CTAN's `cm-unicode` package, converted from CFF to
TrueType outlines, subset to Latin, Greek and the punctuation / math symbols the figures use, and
renamed. They exist because matplotlib embeds fonts in PDF as TrueType (`pdf.fonttype 42`) and
cannot embed CMU's CFF-flavoured `.otf` files correctly, and because no LaTeX engine is installed on
the machine that builds the figures (with `latex` on the PATH, `thesis_style.py` uses `text.usetex`
instead).

Licence: SIL Open Font License 1.1 (`OFL.txt`, copied from cm-unicode). CMU reserves the font family
name "Computer Modern Unicode fonts", so the modified fonts carry the name "Thesis CM Serif"
(OFL condition 3).

Rebuild: download https://mirrors.ctan.org/fonts/cm-unicode.zip, unzip it, then

    python3 analysis/summary/fonts/make_fonts.py <unzipped>/cm-unicode/fonts/otf
