"""Render the final Markdown research report to a portable CJK PDF."""

from __future__ import annotations

import argparse
import html
import re
from pathlib import Path

from quant_research.ingest import ROOT


DEFAULT_INPUT = ROOT / "reports" / "final_research_report.md"
DEFAULT_OUTPUT = ROOT / "reports" / "final_report.pdf"
FONT_CANDIDATES = (
    Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
    Path("/usr/share/fonts/truetype/unifont/unifont.ttf"),
    Path("C:/Windows/Fonts/msyh.ttc"),
    Path("C:/Windows/Fonts/simhei.ttf"),
)


def markdown_blocks(text: str) -> list[tuple[str, object]]:
    """Parse the small Markdown subset used by the final report."""
    lines = text.splitlines()
    blocks: list[tuple[str, object]] = []
    index = 0
    while index < len(lines):
        line = lines[index].rstrip()
        if not line:
            index += 1
            continue
        if line.startswith("!["):
            blocks.append(("image_note", line))
            index += 1
            continue
        if line.startswith("|"):
            table = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table.append([cell.strip() for cell in lines[index].strip().strip("|").split("|")])
                index += 1
            if len(table) >= 2 and all(re.fullmatch(r":?-{3,}:?", cell) for cell in table[1]):
                table.pop(1)
            blocks.append(("table", table))
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", line)
        if heading:
            blocks.append((f"h{len(heading.group(1))}", heading.group(2)))
        elif re.match(r"^[-*]\s+", line):
            blocks.append(("bullet", re.sub(r"^[-*]\s+", "", line)))
        elif re.match(r"^\d+\.\s+", line):
            blocks.append(("bullet", re.sub(r"^\d+\.\s+", "", line)))
        else:
            paragraph = [line]
            index += 1
            while index < len(lines):
                following = lines[index].rstrip()
                if (not following or following.startswith(("#", "|", "![", "- ", "* "))
                        or re.match(r"^\d+\.\s+", following)):
                    break
                paragraph.append(following)
                index += 1
            blocks.append(("paragraph", " ".join(paragraph)))
            continue
        index += 1
    return blocks


def _inline(text: str) -> str:
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`(.+?)`", r"<font color='#374151'>\1</font>", escaped)
    escaped = re.sub(r"\[(.+?)\]\((.+?)\)", r"<u>\1</u> (\2)", escaped)
    return escaped


def _font_path() -> Path:
    for path in FONT_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError("No CJK font found; install WenQuanYi Zen Hei or Microsoft YaHei")


def build_pdf(source: Path = DEFAULT_INPUT, destination: Path = DEFAULT_OUTPUT) -> Path:
    try:
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (
            BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, Table, TableStyle,
        )
    except ImportError as exc:
        raise RuntimeError("Install the project dependencies to generate PDF: pip install -e .") from exc

    font_path = _font_path()
    try:
        pdfmetrics.registerFont(TTFont("QuantCJK", str(font_path), subfontIndex=0))
    except TypeError:
        pdfmetrics.registerFont(TTFont("QuantCJK", str(font_path)))
    page_width, page_height = A4
    left = right = 18 * mm
    top, bottom = 18 * mm, 17 * mm
    frame = Frame(left, bottom, page_width - left - right, page_height - top - bottom,
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)

    def decorate(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("QuantCJK", 8)
        canvas.setFillColor(colors.HexColor("#6B7280"))
        canvas.drawString(left, 9 * mm, "A 股 PIT 横截面多因子研究")
        canvas.drawRightString(page_width - right, 9 * mm, f"{doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(
        str(destination), pagesize=A4, leftMargin=left, rightMargin=right,
        topMargin=top, bottomMargin=bottom,
        title="A股横截面多因子选股：点时数据、风险中性化与可成交归因",
        author="ai-quant-research",
    )
    doc.addPageTemplates([PageTemplate(id="main", frames=[frame], onPage=decorate)])
    base = getSampleStyleSheet()
    styles = {
        "h1": ParagraphStyle("H1CJK", parent=base["Title"], fontName="QuantCJK",
                              fontSize=20, leading=27, textColor=colors.HexColor("#0F3D63"),
                              spaceAfter=12, alignment=TA_CENTER, wordWrap="CJK"),
        "h2": ParagraphStyle("H2CJK", parent=base["Heading2"], fontName="QuantCJK",
                              fontSize=14, leading=19, textColor=colors.HexColor("#145A86"),
                              spaceBefore=10, spaceAfter=6, wordWrap="CJK"),
        "h3": ParagraphStyle("H3CJK", parent=base["Heading3"], fontName="QuantCJK",
                              fontSize=11.5, leading=16, textColor=colors.HexColor("#1F5875"),
                              spaceBefore=7, spaceAfter=4, wordWrap="CJK"),
        "body": ParagraphStyle("BodyCJK", parent=base["BodyText"], fontName="QuantCJK",
                                fontSize=9.2, leading=14.2, textColor=colors.HexColor("#1F2937"),
                                spaceAfter=5, wordWrap="CJK"),
        "bullet": ParagraphStyle("BulletCJK", parent=base["BodyText"], fontName="QuantCJK",
                                  fontSize=9.2, leading=14, leftIndent=12, firstLineIndent=-7,
                                  bulletIndent=4, spaceAfter=3, wordWrap="CJK"),
        "small": ParagraphStyle("SmallCJK", parent=base["BodyText"], fontName="QuantCJK",
                                 fontSize=7.2, leading=9.2, wordWrap="CJK"),
    }
    story = []
    for kind, content in markdown_blocks(source.read_text(encoding="utf-8")):
        if kind == "table":
            rows = [[Paragraph(_inline(cell), styles["small"]) for cell in row]
                    for row in content]
            widths = [(page_width - left - right) / max(len(rows[0]), 1)] * len(rows[0])
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCEAF3")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0F3D63")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 3),
                ("RIGHTPADDING", (0, 0), (-1, -1), 3),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]))
            story.extend([table, Spacer(1, 5)])
        elif kind == "bullet":
            story.append(Paragraph(_inline(str(content)), styles["bullet"], bulletText="•"))
        elif kind == "image_note":
            label = re.match(r"!\[(.*?)\]", str(content))
            story.append(Paragraph(
                f"[图表见仓库 HTML/SVG：{_inline(label.group(1) if label else 'figure')}]",
                styles["small"],
            ))
        elif kind.startswith("h"):
            level = min(int(kind[1:]), 3)
            story.append(Paragraph(_inline(str(content)), styles[f"h{level}"]))
        else:
            story.append(Paragraph(_inline(str(content)), styles["body"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    doc.build(story)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the final research report as PDF")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(build_pdf(args.input, args.output))


if __name__ == "__main__":
    main()
