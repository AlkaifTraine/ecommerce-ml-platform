"""Render a Markdown document to a formatted PDF.

Written for docs/PROJECT_REFERENCE.md, which is heavy on tables and fenced code
blocks, so a naive text dump would be unreadable. Handles headings, paragraphs,
pipe tables, fenced code, blockquotes, bullet and numbered lists, horizontal
rules, and inline bold / italic / code.

Usage:
    python -m scripts.md_to_pdf docs/PROJECT_REFERENCE.md docs/PROJECT_REFERENCE.pdf
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5b6570")
RULE = colors.HexColor("#d5dae0")
CODE_BG = colors.HexColor("#f5f6f8")
HEAD_BG = colors.HexColor("#eef1f5")
ACCENT = colors.HexColor("#1f4e79")


def build_styles() -> dict:
    base = getSampleStyleSheet()
    s = {}
    s["title"] = ParagraphStyle(
        "title", parent=base["Title"], fontName="Helvetica-Bold",
        fontSize=20, leading=25, textColor=ACCENT, spaceAfter=4,
    )
    s["subtitle"] = ParagraphStyle(
        "subtitle", parent=base["Normal"], fontName="Helvetica",
        fontSize=9.5, leading=13, textColor=MUTED, spaceAfter=14,
    )
    s["h1"] = ParagraphStyle(
        "h1", parent=base["Heading1"], fontName="Helvetica-Bold",
        fontSize=15, leading=19, textColor=ACCENT, spaceBefore=16, spaceAfter=7,
    )
    s["h2"] = ParagraphStyle(
        "h2", parent=base["Heading2"], fontName="Helvetica-Bold",
        fontSize=12, leading=16, textColor=INK, spaceBefore=12, spaceAfter=5,
    )
    s["h3"] = ParagraphStyle(
        "h3", parent=base["Heading3"], fontName="Helvetica-Bold",
        fontSize=10.5, leading=14, textColor=INK, spaceBefore=9, spaceAfter=4,
    )
    s["body"] = ParagraphStyle(
        "body", parent=base["Normal"], fontName="Helvetica",
        fontSize=9.2, leading=13.4, textColor=INK, alignment=TA_LEFT, spaceAfter=6,
    )
    s["bullet"] = ParagraphStyle(
        "bullet", parent=s["body"], leftIndent=12, bulletIndent=3, spaceAfter=3,
    )
    s["quote"] = ParagraphStyle(
        "quote", parent=s["body"], leftIndent=12, textColor=MUTED,
        borderPadding=(0, 0, 0, 6), spaceBefore=4, spaceAfter=8,
    )
    s["code"] = ParagraphStyle(
        "code", parent=base["Code"], fontName="Courier",
        fontSize=7.4, leading=9.4, textColor=INK,
        backColor=CODE_BG, borderPadding=(5, 5, 5, 5), spaceBefore=4, spaceAfter=8,
    )
    s["cell"] = ParagraphStyle(
        "cell", parent=base["Normal"], fontName="Helvetica",
        fontSize=7.4, leading=9.6, textColor=INK,
    )
    s["cellhead"] = ParagraphStyle(
        "cellhead", parent=s["cell"], fontName="Helvetica-Bold", textColor=ACCENT,
    )
    return s


# --------------------------------------------------------------------------
# inline markup
# --------------------------------------------------------------------------
def inline(text: str) -> str:
    """Markdown inline -> reportlab mini-HTML. Escapes XML first."""
    out = escape(text)
    # `code` before bold/italic so underscores inside code are untouched
    out = re.sub(
        r"`([^`]+)`",
        r'<font face="Courier" size="8" backColor="#f0f1f4">\1</font>',
        out,
    )
    out = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", out)
    # single * italics, but not the leftovers of ** handled above
    out = re.sub(r"(?<!\*)\*([^*\n]+)\*(?!\*)", r"<i>\1</i>", out)
    # markdown links -> text (underlined), the URL is rarely useful in print
    out = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<u>\1</u>", out)
    return out


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def is_divider(line: str) -> bool:
    cells = split_row(line)
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", c) for c in cells)


def col_widths(rows: list[list[str]], avail: float) -> list[float]:
    """Proportional to the longest cell per column, with a floor."""
    n = len(rows[0])
    longest = [1] * n
    for r in rows:
        for i, c in enumerate(r[:n]):
            longest[i] = max(longest[i], len(c))
    total = sum(longest)
    floor = avail * 0.07
    w = [max(floor, avail * (x / total)) for x in longest]
    scale = avail / sum(w)
    return [x * scale for x in w]


# --------------------------------------------------------------------------
# document assembly
# --------------------------------------------------------------------------
def convert(md_path: Path, pdf_path: Path, title: str, subtitle: str) -> None:
    s = build_styles()
    lines = md_path.read_text(encoding="utf-8").splitlines()

    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4,
        leftMargin=16 * mm, rightMargin=14 * mm,
        topMargin=15 * mm, bottomMargin=15 * mm,
        title=title, author="ecommerce-ml-platform",
    )
    avail = doc.width
    story: list = [Paragraph(escape(title), s["title"]),
                   Paragraph(escape(subtitle), s["subtitle"]),
                   HRFlowable(width="100%", color=RULE, spaceAfter=10)]

    i = 0
    first_h1 = True
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # fenced code -------------------------------------------------------
        if stripped.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            body = escape("\n".join(buf)) or " "
            story.append(Paragraph(body.replace("\n", "<br/>").replace(" ", "&nbsp;"), s["code"]))
            continue

        # table -------------------------------------------------------------
        if stripped.startswith("|") and i + 1 < len(lines) and is_divider(lines[i + 1]):
            header = split_row(stripped)
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(split_row(lines[i].strip()))
                i += 1
            ncol = len(header)
            norm = [header] + [(r + [""] * ncol)[:ncol] for r in rows]
            data = [
                [Paragraph(inline(c), s["cellhead"] if ri == 0 else s["cell"]) for c in row]
                for ri, row in enumerate(norm)
            ]
            t = Table(data, colWidths=col_widths(norm, avail), repeatRows=1, hAlign="LEFT")
            t.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), HEAD_BG),
                ("GRID", (0, 0), (-1, -1), 0.4, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fafbfc")]),
            ]))
            story.extend([Spacer(1, 3), t, Spacer(1, 8)])
            continue

        # headings ----------------------------------------------------------
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            level, text = len(m.group(1)), m.group(2)
            if level == 1:
                if not first_h1:
                    story.append(PageBreak())
                first_h1 = False
                story.append(Paragraph(inline(text), s["h1"]))
            elif level == 2:
                story.append(KeepTogether([Paragraph(inline(text), s["h1"])]))
            elif level == 3:
                story.append(KeepTogether([Paragraph(inline(text), s["h2"])]))
            else:
                story.append(Paragraph(inline(text), s["h3"]))
            i += 1
            continue

        # horizontal rule ---------------------------------------------------
        if re.fullmatch(r"-{3,}|\*{3,}|_{3,}", stripped):
            story.append(HRFlowable(width="100%", color=RULE, spaceBefore=6, spaceAfter=8))
            i += 1
            continue

        # blockquote --------------------------------------------------------
        if stripped.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            story.append(Paragraph(inline(" ".join(buf)), s["quote"]))
            continue

        # bullets -----------------------------------------------------------
        if re.match(r"^[-*]\s+", stripped):
            while i < len(lines) and re.match(r"^\s*[-*]\s+", lines[i]):
                text = re.sub(r"^\s*[-*]\s+", "", lines[i])
                story.append(Paragraph(inline(text), s["bullet"], bulletText="•"))
                i += 1
            story.append(Spacer(1, 4))
            continue

        # numbered ----------------------------------------------------------
        if re.match(r"^\d+\.\s+", stripped):
            while i < len(lines) and re.match(r"^\s*\d+\.\s+", lines[i]):
                mm_ = re.match(r"^\s*(\d+)\.\s+(.*)$", lines[i])
                story.append(Paragraph(inline(mm_.group(2)), s["bullet"],
                                       bulletText=f"{mm_.group(1)}."))
                i += 1
            story.append(Spacer(1, 4))
            continue

        # blank ---------------------------------------------------------------
        if not stripped:
            i += 1
            continue

        # paragraph (gather until blank / structural line) ---------------------
        buf = []
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt or nxt.startswith(("#", "|", "```", ">", "---"))
                    or re.match(r"^([-*]|\d+\.)\s+", nxt)):
                break
            buf.append(nxt)
            i += 1
        if buf:
            story.append(Paragraph(inline(" ".join(buf)), s["body"]))

    def footer(canvas, doc_):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(16 * mm, 9 * mm, title)
        canvas.drawRightString(A4[0] - 14 * mm, 9 * mm, f"page {doc_.page}")
        canvas.setStrokeColor(RULE)
        canvas.line(16 * mm, 12 * mm, A4[0] - 14 * mm, 12 * mm)
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source")
    ap.add_argument("target")
    ap.add_argument("--title", default="Real-Time Purchase-Intent Platform")
    ap.add_argument("--subtitle",
                    default="Complete Project Reference - "
                            "github.com/AlkaifTraine/ecommerce-ml-platform")
    args = ap.parse_args()

    src, dst = Path(args.source), Path(args.target)
    dst.parent.mkdir(parents=True, exist_ok=True)
    convert(src, dst, args.title, args.subtitle)
    print(f"written {dst}  ({dst.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
