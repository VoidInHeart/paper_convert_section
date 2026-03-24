from __future__ import annotations

from paper_review_system.models import PaperBlock


class MarkdownRenderer:
    """Render cleaned blocks into a readable markdown document."""

    def render(self, clean_blocks: list[PaperBlock]) -> str:
        lines: list[str] = []
        for block in clean_blocks:
            if block.is_noise:
                continue
            text = block.text.strip()
            if not text:
                continue
            if block.type == "heading":
                level = min(block.level or 2, 6)
                lines.append(f"{'#' * level} {text}")
                lines.append("")
                continue
            if block.type == "caption":
                lines.append(f"> {text}")
                lines.append("")
                continue
            if block.type == "table":
                lines.extend(self._render_table(block))
                lines.append("")
                continue
            if block.type == "formula":
                lines.append("```math")
                lines.extend(text.splitlines())
                lines.append("```")
                lines.append("")
                continue
            lines.append(text.replace("\n", " "))
            lines.append("")
        return "\n".join(lines).strip() + "\n"

    def _render_table(self, block: PaperBlock) -> list[str]:
        headers = list(block.table_headers or [])
        rows = [list(row) for row in (block.table_rows or [])]
        lines: list[str] = []
        if block.table_caption:
            lines.append(f"> {block.table_caption}")
            lines.append("")
        if not rows:
            lines.extend(["```text", *block.text.splitlines(), "```"])
            return lines

        col_count = max(len(headers), *(len(row) for row in rows))
        if not headers:
            headers = [f"Column {index}" for index in range(1, col_count + 1)]
        headers = self._pad_row(headers, col_count)
        normalized_rows = [self._pad_row(row, col_count) for row in rows]

        lines.extend([
            "| " + " | ".join(self._escape_cell(cell) for cell in headers) + " |",
            "| " + " | ".join("---" for _ in range(col_count)) + " |",
        ])
        for row in normalized_rows:
            lines.append("| " + " | ".join(self._escape_cell(cell) for cell in row) + " |")
        return lines

    @staticmethod
    def _pad_row(row: list[str], size: int) -> list[str]:
        padded = row[:size]
        if len(padded) < size:
            padded.extend([""] * (size - len(padded)))
        return padded

    @staticmethod
    def _escape_cell(text: str) -> str:
        return text.replace("|", "\\|").replace("\n", "<br>")
