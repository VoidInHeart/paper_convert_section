from __future__ import annotations

import re
import statistics
from pathlib import Path

from paper_review_system.models import PageInfo, PaperBlock, PaperDocument, build_doc_id
from paper_review_system.parser.reading_order import ReadingOrderResolver


class PDFParser:
    """Parse PDF pages into a document-level IR with block coordinates."""

    TEXT_REPLACEMENTS = {
        "\u00a0": " ",
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2015": "-",
        "\u2212": "-",
        "鈥?": "-",
        "鈥�": "\"",
        "鈥": "\"",
        "鉁?": "[Y]",
        "脳": "[N]",
    }

    def __init__(self) -> None:
        self.reading_order = ReadingOrderResolver()

    def parse(self, pdf_path: str | Path) -> PaperDocument:
        path = Path(pdf_path)
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is required to parse PDF files. Install dependencies with `pip install -e .`."
            ) from exc

        source_file = str(path.resolve())
        doc_id = build_doc_id(source_file)
        pdf = fitz.open(source_file)
        pages: list[PageInfo] = []
        block_records: list[dict[str, object]] = []

        for page_index, page in enumerate(pdf, start=1):
            pages.append(PageInfo(page=page_index, width=float(page.rect.width), height=float(page.rect.height)))
            text_dict = page.get_text("dict")
            text_blocks = [item for item in text_dict.get("blocks", []) if item.get("type") == 0]
            body_size = self._estimate_body_font_size(text_blocks)

            for block_index, raw_block in enumerate(text_blocks, start=1):
                text = self._flatten_block_text(raw_block).strip()
                if not text:
                    continue
                bbox = [round(float(value), 2) for value in raw_block.get("bbox", (0, 0, 0, 0))]
                font_size = self._block_font_size(raw_block)
                block_type, level, role = self._classify_block(
                    text=text,
                    font_size=font_size,
                    body_size=body_size,
                    page_number=page_index,
                    block_index=block_index,
                )
                block_records.append(
                    {
                        "block_id": f"blk_{len(block_records) + 1:06d}",
                        "page": page_index,
                        "bbox": bbox,
                        "type": block_type,
                        "text": text,
                        "level": level,
                        "font_size": font_size,
                        "role": role,
                    }
                )

        pdf.close()
        blocks = [PaperBlock(**record) for record in block_records]
        blocks = self.reading_order.order_blocks(blocks, pages)
        metadata = {"parser": "pymupdf", "block_count": len(blocks)}
        return PaperDocument(doc_id=doc_id, source_file=source_file, pages=pages, blocks=blocks, metadata=metadata)

    @classmethod
    def _flatten_block_text(cls, raw_block: dict) -> str:
        parts: list[str] = []
        for line in raw_block.get("lines", []):
            line_parts: list[str] = []
            for span in line.get("spans", []):
                span_text = str(span.get("text", "")).strip()
                if span_text:
                    line_parts.append(span_text)
            if line_parts:
                parts.append(" ".join(line_parts))
        return cls._normalize_text("\n".join(parts))

    @staticmethod
    def _block_font_size(raw_block: dict) -> float:
        sizes: list[float] = []
        for line in raw_block.get("lines", []):
            for span in line.get("spans", []):
                size = span.get("size")
                if isinstance(size, (int, float)):
                    sizes.append(float(size))
        if not sizes:
            return 0.0
        return round(max(sizes), 2)

    @staticmethod
    def _estimate_body_font_size(text_blocks: list[dict]) -> float:
        sizes: list[float] = []
        for block in text_blocks:
            block_size = PDFParser._block_font_size(block)
            text = PDFParser._flatten_block_text(block).strip()
            if text and len(text) > 20 and block_size > 0:
                sizes.append(block_size)
        if not sizes:
            return 11.0
        return round(statistics.median(sizes), 2)

    @staticmethod
    def _classify_block(
        text: str,
        font_size: float,
        body_size: float,
        page_number: int,
        block_index: int,
    ) -> tuple[str, int | None, str | None]:
        stripped = text.strip()
        lines = [line for line in stripped.splitlines() if line.strip()]
        single_line = len(lines) == 1
        is_title = page_number == 1 and block_index <= 3 and font_size >= body_size * 1.5

        if PDFParser._looks_like_header_footer(stripped):
            return "metadata", None, "header_footer"
        if PDFParser._looks_like_caption(stripped):
            return "caption", None, "caption"

        # looks_like_heading = single_line and (
        #     font_size >= body_size * 1.18
        #     or stripped.lower() in {"abstract", "introduction", "references", "conclusion"}
        #     or stripped.startswith(("摘要", "引言", "结论", "参考文献", "附录"))
        # )

        #防止出现单行误识别为标题：只把长度小于100的单行识别为标题
        looks_like_heading = single_line and (
            font_size >= body_size * 1.18
            and len(stripped) < 100  
            and not re.search(r"Trained on", stripped, re.IGNORECASE)
            and stripped.lower() not in {"abstract", "introduction", "references", "conclusion"}
            and not stripped.startswith(("摘要", "引言", "结论", "参考文献", "附录"))
        )
        if is_title:
            return "heading", 1, "title"
        if looks_like_heading:
            level = 2
            if stripped.startswith(tuple(str(i) for i in range(1, 10))):
                level = min(stripped.count(".") + 2, 6)
            return "heading", level, "section_heading"
        if PDFParser._looks_like_table(stripped):
            return "table", None, "table_like"
        if PDFParser._looks_like_formula(stripped):
            return "formula", None, "formula_like"
        return "paragraph", None, "body"

    @staticmethod
    def _looks_like_table(text: str) -> bool:
        return ("|" in text) or ("\t" in text) or (text.count("  ") >= 3 and any(ch.isdigit() for ch in text))

    @staticmethod
    def _looks_like_formula(text: str) -> bool:
        operators = {"=", "+", "-", "∑", "∫", "√", "≤", "≥"}
        return len(text) < 120 and sum(ch in operators for ch in text) >= 2

    @staticmethod
    def _looks_like_caption(text: str) -> bool:
        compact = re.sub(r"\s+", " ", text).strip()
        return bool(
            re.match(r"^(figure|fig\.|table)\s*\d+[.:]?\s", compact, re.IGNORECASE)
            or re.match(r"^(图|表)\s*\d+[.:：]?\s*", compact)
        )

    @staticmethod
    def _looks_like_header_footer(text: str) -> bool:
        compact = re.sub(r"\s+", " ", text).strip()
        lower = compact.lower()
        return bool(
            "arxiv:" in lower
            or re.fullmatch(r"\d{1,3}", compact)
            or re.fullmatch(r"page\s+\d{1,3}", lower)
        )

    @classmethod
    def _normalize_text(cls, text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        for source, target in cls.TEXT_REPLACEMENTS.items():
            normalized = normalized.replace(source, target)
        normalized = re.sub(r"(?<=[A-Za-z])-\n(?=[A-Za-z])", "", normalized)
        normalized = re.sub(r"(?<![:.!?])\n(?!\n)", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized
