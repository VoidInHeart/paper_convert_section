from __future__ import annotations

from collections import defaultdict

from paper_review_system.models import PageInfo, PaperBlock


class ReadingOrderResolver:
    """Restore reading order for single-column and double-column PDF layouts."""

    def order_blocks(self, blocks: list[PaperBlock], pages: list[PageInfo]) -> list[PaperBlock]:
        page_map = {page.page: page for page in pages}
        page_to_blocks: dict[int, list[PaperBlock]] = defaultdict(list)
        for block in blocks:
            page_to_blocks[block.page].append(block)

        ordered: list[PaperBlock] = []
        for page_number in sorted(page_to_blocks):
            page_info = page_map.get(page_number)
            page_blocks = page_to_blocks[page_number]
            if page_info is None:
                ordered.extend(sorted(page_blocks, key=self._fallback_sort_key))
                continue
            ordered.extend(self._order_page_blocks(page_blocks, page_info))
        return ordered

    def _order_page_blocks(self, blocks: list[PaperBlock], page: PageInfo) -> list[PaperBlock]:
        split_x = self._detect_split_x(blocks, page.width)
        if split_x is None:
            return sorted(blocks, key=self._fallback_sort_key)

        left_column: list[PaperBlock] = []
        right_column: list[PaperBlock] = []
        top_full_width: list[PaperBlock] = []
        middle_full_width: list[PaperBlock] = []
        bottom_full_width: list[PaperBlock] = []

        column_blocks = [block for block in blocks if self._classify_region(block, split_x, page.width) in {"left", "right"}]
        body_column_blocks = self._body_column_candidates(blocks, page.width)
        if not column_blocks:
            return sorted(blocks, key=self._fallback_sort_key)

        active_column_blocks = body_column_blocks or column_blocks
        left_top = min((block.bbox[1] for block in active_column_blocks if self._classify_region(block, split_x, page.width) == "left"), default=None)
        right_top = min((block.bbox[1] for block in active_column_blocks if self._classify_region(block, split_x, page.width) == "right"), default=None)
        left_bottom = max((block.bbox[3] for block in active_column_blocks if self._classify_region(block, split_x, page.width) == "left"), default=None)
        right_bottom = max((block.bbox[3] for block in active_column_blocks if self._classify_region(block, split_x, page.width) == "right"), default=None)
        column_top = min(value for value in [left_top, right_top] if value is not None)
        column_bottom = max(value for value in [left_bottom, right_bottom] if value is not None)

        for block in blocks:
            region = self._classify_region(block, split_x, page.width)
            if block.bbox[3] <= column_top + 24:
                top_full_width.append(block)
            elif region == "left":
                left_column.append(block)
            elif region == "right":
                right_column.append(block)
            else:
                if block.bbox[1] >= column_bottom - 24:
                    bottom_full_width.append(block)
                else:
                    middle_full_width.append(block)

        ordered: list[PaperBlock] = []
        ordered.extend(sorted(top_full_width, key=self._fallback_sort_key))
        ordered.extend(sorted(left_column, key=self._fallback_sort_key))
        ordered.extend(sorted(right_column, key=self._fallback_sort_key))
        ordered.extend(sorted(middle_full_width + bottom_full_width, key=self._fallback_sort_key))
        return ordered

    def _detect_split_x(self, blocks: list[PaperBlock], page_width: float) -> float | None:
        candidate_sets = [
            self._body_column_candidates(blocks, page_width),
            self._column_candidates(blocks, page_width, strict=True),
            self._column_candidates(blocks, page_width, strict=False),
        ]
        for candidates in candidate_sets:
            split_x = self._split_from_candidates(candidates, page_width)
            if split_x is not None:
                return split_x
        return None

    def _body_column_candidates(self, blocks: list[PaperBlock], page_width: float) -> list[PaperBlock]:
        candidates: list[PaperBlock] = []
        for block in blocks:
            if block.is_noise or block.type != "paragraph":
                continue
            width = block.bbox[2] - block.bbox[0]
            if width < page_width * 0.28 or width > page_width * 0.52:
                continue
            if len(block.text.strip()) < 120:
                continue
            candidates.append(block)
        return candidates

    def _column_candidates(self, blocks: list[PaperBlock], page_width: float, strict: bool) -> list[PaperBlock]:
        candidates: list[PaperBlock] = []
        min_width = page_width * (0.24 if strict else 0.18)
        max_width = page_width * (0.52 if strict else 0.62)
        min_text_length = 40 if strict else 18

        for block in blocks:
            if block.is_noise or block.type == "metadata":
                continue
            width = block.bbox[2] - block.bbox[0]
            if width < min_width or width > max_width:
                continue
            if strict:
                if block.type not in {"paragraph", "heading", "table", "caption"}:
                    continue
                if block.type == "paragraph" and len(block.text.strip()) < min_text_length:
                    continue
            else:
                if len(block.text.strip()) < min_text_length and block.type != "heading":
                    continue
            candidates.append(block)
        return candidates

    def _split_from_candidates(self, candidates: list[PaperBlock], page_width: float) -> float | None:
        if len(candidates) < 2:
            return None

        x_positions = sorted(block.bbox[0] for block in candidates)
        gaps = [(x_positions[index + 1] - x_positions[index], index) for index in range(len(x_positions) - 1)]
        large_gaps = [(gap, index) for gap, index in gaps if gap >= page_width * 0.14]
        if not large_gaps:
            return None

        gap, index = max(large_gaps, key=lambda item: item[0])
        split_x = x_positions[index] + gap / 2

        left_count = sum(1 for block in candidates if ((block.bbox[0] + block.bbox[2]) / 2) < split_x)
        right_count = sum(1 for block in candidates if ((block.bbox[0] + block.bbox[2]) / 2) >= split_x)
        if left_count < 1 or right_count < 1:
            return None
        return split_x

    @staticmethod
    def _classify_region(block: PaperBlock, split_x: float, page_width: float) -> str:
        left, _, right, _ = block.bbox
        width = right - left
        center = (left + right) / 2
        margin = page_width * 0.03
        spans_split = left <= split_x - margin and right >= split_x + margin
        if width >= page_width * 0.72 or (spans_split and width >= page_width * 0.55):
            return "full"
        return "left" if center < split_x else "right"

    @staticmethod
    def _fallback_sort_key(block: PaperBlock) -> tuple[float, float, str]:
        return (round(block.bbox[1], 2), round(block.bbox[0], 2), block.block_id)
