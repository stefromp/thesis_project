"""
House style of the thesis figures (A4, 11 pt, text width 15.9 cm = 6.27 in). Every figure
script imports this module and saves through save(), which:
  * fits the saved (tight) width to the size preset, so \\includegraphics[width=\\textwidth] is 1:1
    for FULL (and 1:1 at width=0.486\\textwidth for HALF);
  * checks every rendered string: no code identifiers (adaptive_jenks, entmax15, topk5, ...), no
    raw underscores outside math, datasets capitalised, nothing below 6 pt;
  * writes <name>.pdf (fonts embedded as TrueType, pdf.fonttype 42), <name>.png (300 dpi) and
    <name>.caption.tex (the caption, for the thesis; figures carry no caption of their own).

Fonts: LaTeX (text.usetex) when a `latex` binary is found, otherwise Computer Modern through
the bundled fonts/ThesisCMSerif-*.ttf (CMU Serif converted to TrueType, OFL; see
fonts/make_fonts.py) with mathtext "cm", falling back to CMU Serif / Latin Modern Roman /
DejaVu Serif where those are installed.
"""
from __future__ import annotations

import colorsys
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.text import Text  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE / "figures"            # save() reads this at call time (tests redirect it)
FONT_DIR = HERE / "fonts"

# ---------------------------------------------------------------------------
# Sizes (inches) and type sizes (pt) at final scale
# ---------------------------------------------------------------------------
FULL, HALF, MAX_H = 6.27, 3.05, 8.5
FS_LABEL, FS_TICK, FS_LEGEND, FS_CELL, FS_SUB, FS_MIN = 9, 8, 8, 6.5, 8, 6
LW_DATA, LW_AXIS = 1.2, 0.6
GRID = "#e6e6e6"
INK, INK2 = "#1a1a1a", "#555555"       # text, secondary text

# ---------------------------------------------------------------------------
# Palette and names (identical in every figure)
# ---------------------------------------------------------------------------
RULES = ["fixed", "jenks", "entmax15", "dense"]
RULE_NAME = {"fixed": "Fixed top-k", "jenks": "Jenks", "entmax15": "entmax-1.5", "dense": r"Dense ($k = N-1$)"}
RULE_COLOR = {"fixed": "#6b6b6b", "jenks": "#1f6fd1", "entmax15": "#e8662e", "dense": "#111111"}
RULE_MARKER = {"fixed": "s", "jenks": "o", "entmax15": "^", "dense": "D"}
BLOCK_LS = ["-", "--", ":"]
DATASETS = ["adult", "beijing", "default", "diabetes", "magic", "shoppers"]
DS_NAME = {d: d.capitalize() for d in DATASETS + ["news"]}
DS_SHORT = {"adult": "AD", "beijing": "BE", "default": "DE", "diabetes": "DI", "magic": "MA", "shoppers": "SH"}
# Okabe-Ito without black and yellow: colour-blind safe, fixed mapping
DS_COLOR = {"adult": "#0072B2", "beijing": "#D55E00", "default": "#009E73", "diabetes": "#CC79A7",
            "magic": "#56B4E9", "shoppers": "#E69F00"}
SEQ_CMAP = "Purples"
DIV_CMAP = "RdBu"                      # low = red, high = blue: use with "higher = better"
DIV_CMAP_R = "RdBu_r"                  # low = blue: use with "lower = better" (error differences)


def rule_label(arm: str) -> str:
    return RULE_NAME[arm]


def ds_label(ds: str) -> str:
    return DS_NAME[ds]


# ---------------------------------------------------------------------------
# rcParams
# ---------------------------------------------------------------------------
USETEX = False


def setup() -> None:
    """Apply the house style (idempotent)."""
    global USETEX
    for p in sorted(FONT_DIR.glob("*.ttf")):
        if str(p) not in {f.fname for f in fm.fontManager.ttflist}:
            fm.fontManager.addfont(str(p))
    USETEX = shutil.which("latex") is not None
    rc = {
        "font.family": "serif",
        "font.serif": ["Thesis CM Serif", "CMU Serif", "Latin Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "axes.unicode_minus": True,
        "font.size": FS_TICK, "axes.titlesize": FS_LABEL, "axes.labelsize": FS_LABEL,
        "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEGEND,
        "legend.title_fontsize": FS_LEGEND, "figure.titlesize": FS_LABEL,
        "axes.titleweight": "normal", "axes.titlelocation": "left", "axes.titlepad": 4,
        "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": INK2, "axes.titlecolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelcolor": INK, "ytick.labelcolor": INK,
        "lines.linewidth": LW_DATA, "lines.markersize": 4.5, "lines.markeredgewidth": 0.5,
        "lines.markeredgecolor": "white", "patch.linewidth": 0.5,
        "axes.linewidth": LW_AXIS, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "xtick.direction": "out", "ytick.direction": "out", "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": LW_AXIS, "ytick.major.width": LW_AXIS, "xtick.minor.visible": False,
        "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2,
        "legend.frameon": False, "legend.handlelength": 1.8, "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2, "legend.borderaxespad": 0.3,
        "errorbar.capsize": 1.5, "hatch.linewidth": 0.4,
        "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.04, "figure.constrained_layout.wspace": 0.04,
        "text.usetex": USETEX,
    }
    if USETEX:
        rc["text.latex.preamble"] = (r"\usepackage{amsmath}\usepackage[utf8]{inputenc}\usepackage{newunicodechar}"
                                     r"\newunicodechar{−}{\ensuremath{-}}\newunicodechar{≈}{\ensuremath{\approx}}"
                                     r"\newunicodechar{τ}{\ensuremath{\tau}}\newunicodechar{φ}{\ensuremath{\varphi}}"
                                     r"\newunicodechar{α}{\ensuremath{\alpha}}\newunicodechar{β}{\ensuremath{\beta}}"
                                     r"\newunicodechar{·}{\ensuremath{\cdot}}\newunicodechar{–}{--}")
    plt.rcParams.update(rc)


setup()


def figure(width: float, height: float, **kw):
    assert width in (FULL, HALF), width
    assert height <= MAX_H, height
    return plt.figure(figsize=(width, height), layout="constrained", **kw)


# ---------------------------------------------------------------------------
# Small layout helpers
# ---------------------------------------------------------------------------
def panel_label(ax, letter: str, x_pt: float = -2) -> None:
    """Bold (a), (b), ... at the top-left of a panel, outside the axes, on the title's baseline."""
    y_pt = getattr(ax, "_thesis_title_pad", plt.rcParams["axes.titlepad"])
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(x_pt, y_pt), textcoords="offset points",
                ha="right", va="baseline", fontsize=FS_LABEL, fontweight="bold", color=INK, annotation_clip=False)


def title2(ax, main: str, sub: str | None = None) -> None:
    """Panel title, with an optional second, smaller grey line under it."""
    pad = plt.rcParams["axes.titlepad"] + (FS_SUB * 1.3 * (sub.count("\n") + 1) if sub is not None else 0)
    ax.set_title(main, loc="left", pad=pad)
    ax._thesis_title_pad = pad
    if sub is not None:
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 3), textcoords="offset points",
                    ha="left", va="bottom", fontsize=FS_SUB, color=INK2, annotation_clip=False, linespacing=1.15)


def mnum(v: float, nd: int, sign: bool = False) -> str:
    """A number for use INSIDE $...$: unary minus without binary-operator spacing."""
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", "{-}").replace("+", "{+}")


def tnum(v: float, nd: int, sign: bool = False) -> str:
    """A number in running text: true minus sign (U+2212)."""
    s = f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}"
    return s.replace("-", "−")


def text_color_on(rgb) -> str:
    """White on dark fills, ink on light ones (relative luminance)."""
    r, g, b = to_rgb(rgb)
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else INK


def desaturate(rgb, s: float = 0.35, toward_white: float = 0.45):
    h, l_, sat = colorsys.rgb_to_hls(*to_rgb(rgb))
    r, g, b = colorsys.hls_to_rgb(h, l_, sat * s)
    return tuple(c + (1 - c) * toward_white for c in (r, g, b))


def style_axes(ax, grid: str | None = "y") -> None:
    ax.grid(False)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Rendered-text checks
# ---------------------------------------------------------------------------
_MATH = re.compile(r"\$[^$]*\$")
FORBIDDEN = [
    (re.compile(r"adaptive_|entmax15|\btopk\d*|top_k|tabdiff_|_tabdiff|\bjenks\b|c2st_xgb|wnmis|kept_fraction|"
                r"_mean\b|_std\b|\bsample_|\bseed\d"), "code identifier"),
    (re.compile(r"\b(adult|beijing|default|diabetes|magic|shoppers|news)\b"), "dataset name not capitalised"),
    (re.compile(r"_"), "underscore outside math"),
]


def rendered_texts(fig) -> list[Text]:
    fig.canvas.draw()
    return [t for t in fig.findobj(Text) if t.get_visible() and t.get_text().strip()]


def check_texts(fig, allow: frozenset | set = frozenset()) -> list[str]:
    """Problems in the strings a figure renders (empty list = clean). `allow`: exact strings that are
    data labels (a dataset's own column names, e.g. BILL_AMT1), exempt from the naming rules."""
    bad = []
    for t in rendered_texts(fig):
        s = t.get_text()
        if t.get_fontsize() < FS_MIN - 1e-9:
            bad.append(f"font {t.get_fontsize():.2f} pt < {FS_MIN} pt: {s!r}")
        if s in allow:
            continue
        plain = _MATH.sub("", s)
        for rx, why in FORBIDDEN:
            if rx.search(plain):
                bad.append(f"{why}: {s!r}")
    return bad


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------
def _fit_width(fig, target: float) -> tuple[float, float]:
    """Adjust the figure width until its tight bbox (+ pad) is `target` inches wide; returns the
    saved size (width, height) in inches."""
    pad = plt.rcParams["savefig.pad_inches"]
    for _ in range(12):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        w = bb.width + 2 * pad
        if abs(w - target) < 0.0015:
            return round(w, 3), round(bb.height + 2 * pad, 3)
        W, H = fig.get_size_inches()
        fig.set_size_inches(W + (target - w), H)
    raise RuntimeError(f"could not fit the figure to {target} in (now {w:.3f} in)")


SAVED: dict[str, tuple[float, float]] = {}


def save(fig, name: str, caption: str, width: float = FULL, allow=frozenset()) -> tuple[float, float]:
    """Check, fit and write <name>.pdf / .png / .caption.tex; returns the saved size (in)."""
    bad = check_texts(fig, allow)
    nonascii = sorted({c for c in caption if ord(c) > 127})
    if nonascii:                                       # pdflatex-safe captions: write math as $...$
        bad.append(f"caption has non-ASCII characters {nonascii}")
    if (caption.count("$") - caption.count(r"\$")) % 2 or re.search(r"(?<!\\)%", caption):
        bad.append("caption has unbalanced $ or an unescaped %")
    if bad:
        raise ValueError(f"{name}: rendered text problems:\n  " + "\n  ".join(bad))
    size = _fit_width(fig, width)
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)
    # the fitted box, passed explicitly: PDF and PNG get exactly this size (the PDF backend's own
    # "tight" box can differ from Agg's by a few hundredths of an inch)
    bb = fig.get_tightbbox(fig.canvas.get_renderer()).padded(plt.rcParams["savefig.pad_inches"])
    fig.savefig(out / f"{name}.pdf", bbox_inches=bb, metadata={"CreationDate": None})   # reproducible bytes
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches=bb)
    (out / f"{name}.caption.tex").write_text(" ".join(caption.split()) + "\n")
    assert size[1] <= MAX_H + 1e-6, (name, size)
    plt.close(fig)
    SAVED[name] = size
    return size


# ---------------------------------------------------------------------------
# Contact sheet and LaTeX preview
# ---------------------------------------------------------------------------
_PLAIN = [(r"\%", "%"), ("~", " "), (r"\,", " "), ("---", "—"), ("--", "–"), (r"\tau", "τ"), (r"\approx", "≈"),
          (r"\leq", "≤"), (r"\le", "≤"), (r"\pm", "±"), (r"\alpha", "α"), (r"\beta", "β"), (r"\varphi", "φ"),
          (r"\phi", "φ"), (r"\cdot", "·"), (r"\times", "×"), (r"\to", "→"), (r"\ge", "≥"), ("^2", "²"), ("_+", "₊"),
          (r"\_", "_"), (r"\&", "&")]


def _caption_plain(tex: str) -> str:
    """Plain-text rendering of a caption for the contact sheet (no math parsing)."""
    s = re.sub(r"\\(emph|textit|textbf|mathrm|text)\{([^}]*)\}", r"\2", tex)
    for a, b in _PLAIN:
        s = s.replace(a, b)
    s = re.sub(r"\\[a-zA-Z]+", "", s).replace("$", "").replace("{", "").replace("}", "")
    return s


def contact_sheet(names: list[str], path: Path | None = None) -> Path:
    """All figures, one per A4 page at true size (PNG), caption underneath."""
    from matplotlib.backends.backend_pdf import PdfPages
    import matplotlib.image as mpimg
    out = Path(OUT_DIR)
    path = path or out / "_contact_sheet.pdf"
    A4 = (8.27, 11.69)
    with PdfPages(path, metadata={"CreationDate": None}) as pdf:
        for n in names:
            png = out / f"{n}.png"
            if not png.exists():
                continue
            img = mpimg.imread(png)
            h_in, w_in = img.shape[0] / 300, img.shape[1] / 300
            fig = plt.figure(figsize=A4, layout=None)
            left, top = (A4[0] - w_in) / 2, 0.9
            ax = fig.add_axes([left / A4[0], 1 - (top + h_in) / A4[1], w_in / A4[0], h_in / A4[1]])
            ax.imshow(img, interpolation="none")
            ax.set_axis_off()
            fig.text(left / A4[0], 1 - 0.55 / A4[1], f"{n}   ({w_in:.2f} × {h_in:.2f} in)", fontsize=9,
                     fontweight="bold", va="bottom")
            cap = (out / f"{n}.caption.tex").read_text() if (out / f"{n}.caption.tex").exists() else ""
            fig.text(left / A4[0], 1 - (top + h_in + 0.2) / A4[1], _caption_plain(cap), fontsize=8, va="top",
                     wrap=True, usetex=False, parse_math=False)
            pdf.savefig(fig)
            plt.close(fig)
    return path


def latex_preview(names: list[str], path: Path | None = None) -> Path:
    out = Path(OUT_DIR)
    path = path or out / "_preview.tex"
    body = []
    for n in names:
        if (out / f"{n}.pdf").exists():
            w = r"\textwidth" if SAVED.get(n, (FULL,))[0] > HALF + 0.5 else r"0.486\textwidth"
            body += [r"\begin{figure}[p]", r"  \centering", rf"  \includegraphics[width={w}]{{{n}.pdf}}",
                     rf"  \caption[{n.replace('_', ' ')}]{{\input{{{n}.caption.tex}}}}", rf"  \label{{fig:{n}}}",
                     r"\end{figure}", ""]
    path.write_text("\n".join([
        "% Preview of the thesis figures at their final size (A4, 11pt, text width 15.9 cm = 6.27 in).",
        "% Compile in analysis/summary/figures/: pdflatex _preview.tex",
        r"\documentclass[11pt,a4paper]{report}",
        r"\usepackage[a4paper,textwidth=15.9cm,vmargin=2.5cm]{geometry}",
        r"\usepackage{graphicx,amsmath,lmodern}",
        r"\usepackage[T1]{fontenc}",
        r"\begin{document}", ""] + body + [r"\end{document}", ""]))
    return path
