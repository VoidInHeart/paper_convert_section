from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from paper_review_system.models import PaperBlock


@dataclass(slots=True)
class TableCandidate:
    page: int
    bbox: list[float]
    headers: list[str]
    rows: list[list[str]]
    caption: str | None = None

    @property
    def col_count(self) -> int:
        return len(self.headers) if self.headers else max((len(row) for row in self.rows), default=0)


class TableStructureRestorer:
    """Recover structured tables from PDF pages and inject them into the block stream."""

    def restore(self, pdf_path: str | Path, clean_blocks: list[PaperBlock]) -> list[PaperBlock]:
        path = Path(pdf_path)
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError(
                "PyMuPDF is required to restore table structure. Install dependencies with `pip install -e .`."
            ) from exc

        updated_blocks = [self._clone_block(block) for block in clean_blocks]
        page_to_blocks: dict[int, list[PaperBlock]] = {}
        for block in updated_blocks:
            page_to_blocks.setdefault(block.page, []).append(block)

        pdf = fitz.open(str(path.resolve()))
        candidates: list[TableCandidate] = []
        for page_number, page in enumerate(pdf, start=1):
            try:
                detected_tables = page.find_tables().tables
            except Exception:
                continue
            for detected in detected_tables:
                payload = self._extract_table_payload(detected)
                if payload is None:
                    continue
                candidates.append(
                    TableCandidate(
                        page=page_number,
                        bbox=[round(float(value), 2) for value in detected.bbox],
                        headers=payload["headers"],
                        rows=payload["rows"],
                    )
                )
        pdf.close()

        merged_candidates = self._merge_candidates(candidates)
        self._bind_captions(merged_candidates, page_to_blocks)

        generated_tables: list[PaperBlock] = []
        table_index = 0
        for candidate in merged_candidates:
            table_index += 1
            generated_tables.append(
                PaperBlock(
                    block_id=f"tbl_{candidate.page:03d}_{table_index:03d}",
                    page=candidate.page,
                    bbox=list(candidate.bbox),
                    type="table",
                    text=self._table_text(candidate.headers, candidate.rows),
                    is_noise=False,
                    source="table_reconstructor",
                    role="reconstructed_table",
                    table_headers=list(candidate.headers),
                    table_rows=[list(row) for row in candidate.rows],
                    table_caption=candidate.caption,
                )
            )
            self._mark_overlapping_blocks(page_to_blocks.get(candidate.page, []), candidate.bbox)

        combined = updated_blocks + generated_tables
        return sorted(combined, key=self._block_sort_key)

    def _extract_table_payload(self, detected_table: object) -> dict[str, list] | None:
        rows = detected_table.extract()
        normalized_rows = [self._normalize_row(row) for row in rows]
        normalized_rows = [row for row in normalized_rows if any(cell for cell in row)]
        if not normalized_rows:
            return None

        row_count = len(normalized_rows)
        col_count = max(len(row) for row in normalized_rows)
        nonempty_count = sum(1 for row in normalized_rows for cell in row if cell)
        filled_rows = sum(1 for row in normalized_rows if sum(1 for cell in row if cell) >= 2)
        char_count = sum(len(cell) for row in normalized_rows for cell in row)
        largest_cell = max((len(cell) for row in normalized_rows for cell in row), default=0)
        fill_ratio = nonempty_count / max(1, row_count * col_count)

        if col_count < 2 or row_count < 2:
            return None
        if nonempty_count < 4 or fill_ratio < 0.35:
            return None
        if filled_rows == 0:
            return None
        if char_count and largest_cell / char_count > 0.72 and nonempty_count <= 4:
            return None

        headers = self._normalize_headers(getattr(getattr(detected_table, "header", None), "names", None), col_count)
        expanded_rows = self._expand_rows(normalized_rows, col_count)
        if headers is None:
            headers = [f"Column {index}" for index in range(1, col_count + 1)]
        return {"headers": headers, "rows": expanded_rows}

    def _merge_candidates(self, candidates: list[TableCandidate]) -> list[TableCandidate]:
        merged: list[TableCandidate] = []
        for candidate in sorted(candidates, key=self._candidate_sort_key):
            if not merged:
                merged.append(candidate)
                continue
            previous = merged[-1]
            if self._should_merge(previous, candidate):
                previous.headers = self._pick_better_headers(previous.headers, candidate.headers)
                previous.rows.extend(candidate.rows)
                previous.bbox = [
                    min(previous.bbox[0], candidate.bbox[0]),
                    min(previous.bbox[1], candidate.bbox[1]),
                    max(previous.bbox[2], candidate.bbox[2]),
                    max(previous.bbox[3], candidate.bbox[3]),
                ]
                continue
            merged.append(candidate)
        return merged

    def _should_merge(self, current: TableCandidate, next_candidate: TableCandidate) -> bool:
        if current.col_count != next_candidate.col_count:
            return False
        if not self._x_overlap_ratio(current.bbox, next_candidate.bbox) >= 0.85:
            return False

        current_headers = self._normalized_header_signature(current.headers)
        next_headers = self._normalized_header_signature(next_candidate.headers)
        headers_compatible = (
            current_headers == next_headers
            or self._is_generic_headers(current.headers)
            or self._is_generic_headers(next_candidate.headers)
        )
        if not headers_compatible:
            return False

        same_page = current.page == next_candidate.page
        if same_page:
            vertical_gap = next_candidate.bbox[1] - current.bbox[3]
            return 0 <= vertical_gap <= 90

        consecutive_page = next_candidate.page == current.page + 1
        near_bottom = current.bbox[3] >= 680
        near_top = next_candidate.bbox[1] <= 180
        return consecutive_page and near_bottom and near_top

    def _bind_captions(self, candidates: list[TableCandidate], page_to_blocks: dict[int, list[PaperBlock]]) -> None:
        used_caption_ids: set[str] = set()
        for candidate in candidates:
            page_blocks = page_to_blocks.get(candidate.page, [])
            caption_block = self._find_table_caption(page_blocks, candidate.bbox, used_caption_ids)
            if caption_block is None:
                continue
            candidate.caption = caption_block.text
            caption_block.is_noise = True
            caption_block.role = "table_caption_bound"
            used_caption_ids.add(caption_block.block_id)

    @staticmethod
    def _normalize_row(row: list[str | None]) -> list[str]:
        cells: list[str] = []
        for cell in row:
            text = "" if cell is None else str(cell)
            text = text.replace("\r\n", "\n").replace("\r", "\n")
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r"\n{2,}", "\n", text)
            text = text.strip()
            cells.append(text)
        return cells

    def _normalize_headers(self, headers: list[str | None] | None, col_count: int) -> list[str] | None:
        if not headers:
            return None
        normalized = self._normalize_row(list(headers))
        if len(normalized) < col_count:
            normalized.extend([""] * (col_count - len(normalized)))
        normalized = normalized[:col_count]
        if sum(1 for cell in normalized if cell) < max(1, col_count // 2):
            return None
        if any(self._looks_numeric(cell) for cell in normalized if cell):
            return None
        if any(len(cell) > 60 for cell in normalized if cell):
            return None
        return [cell or f"Column {index}" for index, cell in enumerate(normalized, start=1)]

    def _expand_rows(self, rows: list[list[str]], col_count: int) -> list[list[str]]:
        expanded: list[list[str]] = []
        for row in rows:
            padded = row + [""] * (col_count - len(row))
            split_cells = [self._split_cell_lines(cell) for cell in padded]
            multi_line_counts = [len(lines) for lines in split_cells if len(lines) > 1]
            if multi_line_counts and len(multi_line_counts) >= 2 and len(set(multi_line_counts)) == 1:
                line_count = multi_line_counts[0]
                for index in range(line_count):
                    expanded.append([lines[index] if index < len(lines) else "" for lines in split_cells])
                continue
            expanded.append([cell.replace("\n", " <br> ") for cell in padded])
        return [row for row in expanded if any(cell for cell in row)]

    @staticmethod
    def _split_cell_lines(cell: str) -> list[str]:
        if not cell:
            return [""]
        parts = [re.sub(r"\s+", " ", item).strip() for item in cell.split("\n")]
        parts = [item for item in parts if item]
        return parts or [""]

    @staticmethod
    def _looks_numeric(text: str) -> bool:
        compact = re.sub(r"[\s,%.$]", "", text)
        return bool(compact) and compact.replace("-", "", 1).isdigit()

    def _find_table_caption(
        self,
        blocks: list[PaperBlock],
        bbox: list[float],
        used_caption_ids: set[str],
    ) -> PaperBlock | None:
        table_center = (bbox[0] + bbox[2]) / 2
        candidates: list[tuple[float, PaperBlock]] = []
        for block in blocks:
            if block.block_id in used_caption_ids or block.is_noise:
                continue
            if block.type != "caption" or not self._is_table_caption(block.text):
                continue
            if self._x_overlap_ratio(block.bbox, bbox) < 0.45 and not (bbox[0] <= (block.bbox[0] + block.bbox[2]) / 2 <= bbox[2]):
                continue

            if block.bbox[3] <= bbox[1]:
                distance = bbox[1] - block.bbox[3]
                if distance <= 140:
                    candidates.append((distance + abs(((block.bbox[0] + block.bbox[2]) / 2) - table_center) * 0.05, block))
            elif block.bbox[1] >= bbox[3]:
                distance = block.bbox[1] - bbox[3]
                if distance <= 120:
                    candidates.append((distance + abs(((block.bbox[0] + block.bbox[2]) / 2) - table_center) * 0.05, block))

        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    @staticmethod
    def _is_table_caption(text: str) -> bool:
        compact = re.sub(r"\s+", " ", text).strip()
        return bool(re.match(r"^(table)\s*\d+", compact, re.IGNORECASE) or re.match(r"^(表)\s*\d+", compact))

    def _mark_overlapping_blocks(self, page_blocks: list[PaperBlock], table_bbox: list[float]) -> None:
        for block in page_blocks:
            if block.type in {"caption", "heading", "metadata"}:
                continue
            if block.role in {"title", "section_heading", "header_footer", "table_caption_bound"}:
                continue
            if self._overlap_ratio(block.bbox, table_bbox) >= 0.35:
                block.is_noise = True
                block.role = "table_source"

    @staticmethod
    def _overlap_ratio(block_bbox: list[float], table_bbox: list[float]) -> float:
        left = max(block_bbox[0], table_bbox[0])
        top = max(block_bbox[1], table_bbox[1])
        right = min(block_bbox[2], table_bbox[2])
        bottom = min(block_bbox[3], table_bbox[3])
        if right <= left or bottom <= top:
            return 0.0
        intersection = (right - left) * (bottom - top)
        block_area = max(1.0, (block_bbox[2] - block_bbox[0]) * (block_bbox[3] - block_bbox[1]))
        return intersection / block_area

    @staticmethod
    def _x_overlap_ratio(first_bbox: list[float], second_bbox: list[float]) -> float:
        left = max(first_bbox[0], second_bbox[0])
        right = min(first_bbox[2], second_bbox[2])
        if right <= left:
            return 0.0
        overlap = right - left
        width = max(1.0, min(first_bbox[2] - first_bbox[0], second_bbox[2] - second_bbox[0]))
        return overlap / width

    @staticmethod
    def _table_text(headers: list[str], rows: list[list[str]]) -> str:
        parts = ["\t".join(headers)]
        for row in rows:
            parts.append("\t".join(row))
        return "\n".join(parts)

    @staticmethod
    def _pick_better_headers(first: list[str], second: list[str]) -> list[str]:
        first_score = TableStructureRestorer._header_quality(first)
        second_score = TableStructureRestorer._header_quality(second)
        return list(second if second_score > first_score else first)

    @staticmethod
    def _header_quality(headers: list[str]) -> tuple[int, int]:
        informative = sum(1 for cell in headers if cell and not re.fullmatch(r"column \d+", cell.lower()))
        total_len = sum(len(cell) for cell in headers if cell)
        return informative, total_len

    @staticmethod
    def _is_generic_headers(headers: list[str]) -> bool:
        return all(re.fullmatch(r"column \d+", cell.lower()) for cell in headers if cell)

    @staticmethod
    def _normalized_header_signature(headers: list[str]) -> tuple[str, ...]:
        normalized: list[str] = []
        for cell in headers:
            compact = re.sub(r"\s+", " ", cell).strip().lower()
            if re.fullmatch(r"column \d+", compact):
                compact = ""
            normalized.append(compact)
        return tuple(normalized)

    @staticmethod
    def _clone_block(block: PaperBlock) -> PaperBlock:
        return PaperBlock(
            block_id=block.block_id,
            page=block.page,
            bbox=list(block.bbox),
            type=block.type,
            text=block.text,
            level=block.level,
            is_noise=block.is_noise,
            font_size=block.font_size,
            source=block.source,
            role=block.role,
            table_headers=list(block.table_headers) if block.table_headers else None,
            table_rows=[list(row) for row in block.table_rows] if block.table_rows else None,
            table_caption=block.table_caption,
        )

    @staticmethod
    def _candidate_sort_key(candidate: TableCandidate) -> tuple[int, float, float]:
        return candidate.page, round(candidate.bbox[1], 2), round(candidate.bbox[0], 2)

    @staticmethod
    def _block_sort_key(block: PaperBlock) -> tuple[int, float, float, str]:
        return (block.page, round(block.bbox[1], 2), round(block.bbox[0], 2), block.block_id)
