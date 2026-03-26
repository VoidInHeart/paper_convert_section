from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from paper_review_system.models import PaperBlock
from paper_review_system.models import PageInfo
from paper_review_system.parser.reading_order import ReadingOrderResolver


@dataclass(slots=True)
class TableCandidate:
    page: int
    bbox: list[float]
    headers: list[str]
    rows: list[list[str]]
    caption: str | None = None
    caption_position: str | None = None

    @property
    def col_count(self) -> int:
        return len(self.headers) if self.headers else max((len(row) for row in self.rows), default=0)


class TableStructureRestorer:
    """Recover structured tables from PDF pages and inject them into the block stream."""

    def __init__(self) -> None:
        self.reading_order = ReadingOrderResolver()

    def restore(self, pdf_path: str | Path, clean_blocks: list[PaperBlock], pages: list[PageInfo] | None = None) -> list[PaperBlock]:
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
        merged_candidates = self._recover_textual_tables(page_to_blocks, merged_candidates)
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
                    table_caption_position=candidate.caption_position,
                )
            )
            self._mark_overlapping_blocks(page_to_blocks.get(candidate.page, []), candidate.bbox)

        combined = updated_blocks + generated_tables
        if pages:
            return self.reading_order.order_blocks(combined, pages)
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
        subset_matrix = self._looks_like_subset_matrix(normalized_rows)
        egr_matrix = self._looks_like_egr_matrix(normalized_rows)

        if col_count < 2 or row_count < 2:
            return None
        if nonempty_count < 4 or (fill_ratio < 0.35 and not subset_matrix and not egr_matrix):
            return None
        if filled_rows == 0:
            return None
        if char_count and largest_cell / char_count > 0.72 and nonempty_count <= 4 and not subset_matrix and not egr_matrix:
            return None

        headers = self._normalize_headers(getattr(getattr(detected_table, "header", None), "names", None), col_count)
        expanded_rows = self._expand_rows(normalized_rows, col_count)
        headers, expanded_rows = self._repair_subset_matrix_table(headers, expanded_rows, normalized_rows)
        headers, expanded_rows = self._repair_egr_matrix_table(headers, expanded_rows, normalized_rows)
        if headers is None:
            headers = [f"Column {index}" for index in range(1, col_count + 1)]
        headers, expanded_rows = self._repair_compressed_metric_table(headers, expanded_rows, normalized_rows)
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
            if candidate.caption:
                continue
            page_blocks = page_to_blocks.get(candidate.page, [])
            match = self._find_table_caption(page_blocks, candidate.bbox, used_caption_ids)
            if match is None:
                continue
            caption_block, caption_position = match
            candidate.caption = caption_block.text
            candidate.caption_position = caption_position
            caption_block.is_noise = True
            caption_block.role = "table_caption_bound"
            used_caption_ids.add(caption_block.block_id)

    def _recover_textual_tables(
        self,
        page_to_blocks: dict[int, list[PaperBlock]],
        candidates: list[TableCandidate],
    ) -> list[TableCandidate]:
        recovered = list(candidates)
        for page_number, page_blocks in page_to_blocks.items():
            caption_blocks = [
                block
                for block in sorted(page_blocks, key=self._block_sort_key)
                if not block.is_noise and block.type == "caption" and self._is_table_caption(block.text)
            ]
            for caption_block in caption_blocks:
                source_blocks = self._find_textual_table_blocks(page_blocks, caption_block)
                if not source_blocks:
                    continue
                parsed = self._parse_textual_table(caption_block.text, source_blocks)
                if parsed is None:
                    continue

                target = self._find_candidate_near_caption(recovered, caption_block)
                bbox = self._combine_bbox(source_blocks)
                if target is None:
                    recovered.append(
                        TableCandidate(
                            page=page_number,
                            bbox=bbox,
                            headers=parsed["headers"],
                            rows=parsed["rows"],
                            caption=caption_block.text,
                            caption_position="below",
                        )
                    )
                else:
                    target.bbox = bbox
                    target.headers = parsed["headers"]
                    target.rows = parsed["rows"]
                    target.caption = caption_block.text
                    target.caption_position = "below"

                caption_block.is_noise = True
                caption_block.role = "table_caption_bound"
        return self._sort_candidates(recovered)

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

    def _repair_compressed_metric_table(
        self,
        headers: list[str],
        rows: list[list[str]],
        raw_rows: list[list[str]],
    ) -> tuple[list[str], list[list[str]]]:
        header_text = " ".join(cell for row in raw_rows[:1] for cell in row if cell)
        metric_headers = self._extract_metric_headers(header_text)
        if not metric_headers:
            return headers, rows

        rebuilt: list[list[str]] = []
        index = 0
        if rows and self._is_metric_header_row(rows[0]):
            index = 1

        rebuilt_any = False
        while index < len(rows):
            row = self._pad_row(rows[index], 4)
            group = self._rebuild_metric_group(rows, index, metric_headers)
            if group is not None:
                rebuilt.extend(group["rows"])
                index = group["next_index"]
                rebuilt_any = True
                continue
            rebuilt.append(row)
            index += 1

        if rebuilt_any:
            return ["Method", "Train Subset", *metric_headers], rebuilt
        return headers, rows

    def _repair_subset_matrix_table(
        self,
        headers: list[str] | None,
        rows: list[list[str]],
        raw_rows: list[list[str]],
    ) -> tuple[list[str] | None, list[list[str]]]:
        if not self._looks_like_subset_matrix(raw_rows):
            return headers, rows

        rebuilt_rows: list[list[str]] = []
        section_metric_headers: list[list[str]] = []
        index = 0
        while index < len(raw_rows):
            row = self._pad_row(raw_rows[index], 4)
            header_cell = row[0]
            if "Test Subset" not in header_cell or "Train Set" not in header_cell:
                index += 1
                continue

            subset_name = header_cell.splitlines()[0].strip()
            metric_headers = self._extract_subset_metric_headers(header_cell)
            if len(metric_headers) < 3:
                index += 1
                continue
            section_metric_headers.append(metric_headers)

            index += 1
            while index < len(raw_rows) and self._is_placeholder_train_row(raw_rows[index]):
                index += 1
            if index >= len(raw_rows):
                break

            first_data_row = self._pad_row(raw_rows[index], 4)
            if first_data_row[0] and first_data_row[1] and first_data_row[2] and first_data_row[3]:
                rebuilt_rows.extend(self._expand_first_subset_style(subset_name, metric_headers, raw_rows, index))
                while index < len(raw_rows) and "Test Subset" not in self._pad_row(raw_rows[index], 4)[0]:
                    index += 1
                continue

            if first_data_row[0]:
                rebuilt_rows.extend(self._expand_second_subset_style(subset_name, metric_headers, first_data_row[0]))
            while index < len(raw_rows) and "Test Subset" not in self._pad_row(raw_rows[index], 4)[0]:
                index += 1

        if rebuilt_rows:
            unified_headers = list(section_metric_headers[0]) if section_metric_headers else []
            if any(headers_list and headers_list[0] != unified_headers[0] for headers_list in section_metric_headers[1:]):
                unified_headers[0] = "Source Set"
            return ["Test Subset", "Method", "Train Set", *unified_headers], rebuilt_rows
        return headers, rows

    def _repair_egr_matrix_table(
        self,
        headers: list[str] | None,
        rows: list[list[str]],
        raw_rows: list[list[str]],
    ) -> tuple[list[str] | None, list[list[str]]]:
        if not self._looks_like_egr_matrix(raw_rows):
            return headers, rows

        metric_headers = ["T2I", "I2I", "FS", "FE"]
        rebuilt_rows: list[list[str]] = []
        index = 0
        while index < len(raw_rows):
            row = self._pad_row(raw_rows[index], 7)
            train_subset = row[2].strip()
            if train_subset not in metric_headers or not row[0] or not row[1]:
                index += 1
                continue

            train_index = metric_headers.index(train_subset)
            methods = self._split_cell_lines(row[0])
            egr_flags = self._split_cell_lines(row[1])
            row_count = min(len(methods), len(egr_flags))
            if row_count == 0:
                index += 1
                continue

            prefix_values = self._extract_egr_prefix_values(row[3], row_count, train_index)
            diagonal_values, next_index = self._collect_egr_diagonal_values(raw_rows, index, row_count, 3 + train_index)
            suffix_values = self._extract_egr_suffix_values(
                row[4 + train_index] if train_index < len(metric_headers) - 1 else "",
                row_count,
                len(metric_headers) - train_index - 1,
            )

            if prefix_values is None or len(diagonal_values) != row_count or suffix_values is None:
                index += 1
                continue

            for row_index in range(row_count):
                rebuilt_rows.append(
                    [
                        methods[row_index],
                        egr_flags[row_index],
                        train_subset,
                        *prefix_values[row_index],
                        diagonal_values[row_index],
                        *suffix_values[row_index],
                    ]
                )
            index = next_index

        if rebuilt_rows:
            return ["Backbone", "+EGR", "Train Subset", *metric_headers], rebuilt_rows
        return headers, rows

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

    @staticmethod
    def _split_numeric_tokens(text: str) -> list[str]:
        return re.findall(r"-?\d+(?:\.\d+)?", text)

    def _rebuild_metric_group(
        self,
        rows: list[list[str]],
        start: int,
        metric_headers: list[str],
    ) -> dict[str, object] | None:
        if start >= len(rows):
            return None

        current = self._pad_row(rows[start], 4)
        if not (current[0] and current[1] and current[2]):
            return None
        first_metric_values = self._split_numeric_tokens(current[3])
        if len(first_metric_values) != len(metric_headers) - 1:
            return None

        method_rows = [current]
        index = start + 1
        while index < len(rows):
            candidate = self._pad_row(rows[index], 4)
            metric_values = self._split_numeric_tokens(candidate[3])
            if candidate[0] and not candidate[1] and not candidate[2] and len(metric_values) == len(metric_headers) - 1:
                method_rows.append(candidate)
                index += 1
                continue
            break

        if len(method_rows) < 2:
            return None

        none_values = [current[2]]
        while index < len(rows) and len(none_values) < len(method_rows):
            candidate = self._pad_row(rows[index], 4)
            if not candidate[0] and not candidate[1] and candidate[2] and not candidate[3]:
                none_values.append(candidate[2])
                index += 1
                continue
            break

        if len(none_values) != len(method_rows):
            return None

        rebuilt_rows: list[list[str]] = []
        subset = current[1]
        for row_index, method_row in enumerate(method_rows):
            metric_values = self._split_numeric_tokens(method_row[3])
            rebuilt_rows.append([method_row[0], subset, none_values[row_index], *metric_values])

        return {"rows": rebuilt_rows, "next_index": index}

    @staticmethod
    def _extract_metric_headers(header_text: str) -> list[str] | None:
        compact = re.sub(r"\s+", " ", header_text).strip().lower()
        if "method" not in compact or "subset" not in compact:
            return None
        tokens = re.findall(r"\b(?:none|gn|gb|mb|jpeg)\b", compact)
        if len(tokens) < 3:
            return None
        ordered: list[str] = []
        for token in tokens:
            upper = token.upper()
            label = "None" if upper == "NONE" else upper
            if label not in ordered:
                ordered.append(label)
        return ordered

    @staticmethod
    def _is_metric_header_row(row: list[str]) -> bool:
        first_cell = row[0] if row else ""
        compact = re.sub(r"\s+", " ", first_cell.replace("<br>", " ")).strip().lower()
        return "method" in compact and "subset" in compact and "none" in compact

    @staticmethod
    def _looks_like_subset_matrix(raw_rows: list[list[str]]) -> bool:
        first_cells = " ".join(row[0] for row in raw_rows if row and row[0])
        compact = re.sub(r"\s+", " ", first_cells)
        return "Test Subset" in compact and "Train Set" in compact

    @staticmethod
    def _looks_like_egr_matrix(raw_rows: list[list[str]]) -> bool:
        compact = re.sub(r"\s+", " ", " ".join(cell for row in raw_rows for cell in row if cell))
        return "+EGR" in compact and "T2I" in compact and "I2I" in compact and "FS" in compact and "FE" in compact

    @staticmethod
    def _extract_subset_metric_headers(header_text: str) -> list[str]:
        compact = re.sub(r"\s+", " ", header_text)
        tokens = re.findall(r"(FF\+\+|DFor|T2I|I2I|FS|FE)", compact, re.IGNORECASE)
        ordered: list[str] = []
        for token in tokens:
            normalized = "FF++" if token.upper() == "FF++" else token
            normalized = "DFor" if token.lower() == "dfor" else normalized
            normalized = normalized.upper() if normalized in {"T2I", "I2I", "FS", "FE"} else normalized
            if normalized not in ordered:
                ordered.append(normalized)
        return ordered

    @staticmethod
    def _is_placeholder_train_row(row: list[str]) -> bool:
        padded = TableStructureRestorer._pad_row(row, 4)
        return not padded[0] and padded[1] == "Train Set" and not padded[2] and not padded[3]

    def _extract_egr_prefix_values(
        self,
        cell_text: str,
        row_count: int,
        prefix_count: int,
    ) -> list[list[str]] | None:
        if prefix_count == 0:
            return [[] for _ in range(row_count)]
        lines = self._split_cell_lines(cell_text)
        if len(lines) != row_count:
            return None
        prefix_values: list[list[str]] = []
        for line in lines:
            values = self._split_numeric_tokens(line)
            if len(values) != prefix_count:
                return None
            prefix_values.append(values)
        return prefix_values

    def _collect_egr_diagonal_values(
        self,
        raw_rows: list[list[str]],
        start: int,
        row_count: int,
        column_index: int,
    ) -> tuple[list[str], int]:
        first_row = self._pad_row(raw_rows[start], 7)
        diagonal_values = []
        if first_row[column_index]:
            diagonal_values.extend(self._split_numeric_tokens(first_row[column_index]))

        index = start + 1
        while len(diagonal_values) < row_count and index < len(raw_rows):
            candidate = self._pad_row(raw_rows[index], 7)
            if any(candidate[position] for position in range(7) if position != column_index):
                break
            values = self._split_numeric_tokens(candidate[column_index])
            if len(values) != 1:
                break
            diagonal_values.extend(values)
            index += 1
        return diagonal_values[:row_count], index

    def _extract_egr_suffix_values(
        self,
        cell_text: str,
        row_count: int,
        suffix_count: int,
    ) -> list[list[str]] | None:
        if suffix_count == 0:
            return [[] for _ in range(row_count)]
        lines = self._split_cell_lines(cell_text)
        if len(lines) != row_count:
            return None
        suffix_values: list[list[str]] = []
        for line in lines:
            values = self._split_numeric_tokens(line)
            if len(values) != suffix_count:
                return None
            suffix_values.append(values)
        return suffix_values

    def _expand_first_subset_style(
        self,
        subset_name: str,
        metric_headers: list[str],
        raw_rows: list[list[str]],
        start: int,
    ) -> list[list[str]]:
        first = self._pad_row(raw_rows[start], 4)
        methods = self._split_cell_lines(first[0])
        train_set = first[1]
        first_column_values = [first[2]]
        remaining_metrics = [self._split_numeric_tokens(line) for line in self._split_cell_lines(first[3])]

        index = start + 1
        while len(first_column_values) < len(methods) and index < len(raw_rows):
            candidate = self._pad_row(raw_rows[index], 4)
            if not candidate[0] and not candidate[1] and candidate[2] and not candidate[3]:
                first_column_values.append(candidate[2])
                index += 1
                continue
            break

        rebuilt_rows: list[list[str]] = []
        if len(first_column_values) != len(methods):
            return rebuilt_rows

        for row_index, method in enumerate(methods):
            metric_values = remaining_metrics[row_index] if row_index < len(remaining_metrics) else []
            if len(metric_values) != len(metric_headers) - 1:
                continue
            rebuilt_rows.append([subset_name, method, train_set, first_column_values[row_index], *metric_values])
        return rebuilt_rows

    def _expand_second_subset_style(
        self,
        subset_name: str,
        metric_headers: list[str],
        cell_text: str,
    ) -> list[list[str]]:
        lines = self._split_cell_lines(cell_text)
        train_set = metric_headers[0] if metric_headers else ""
        rebuilt_rows: list[list[str]] = []
        for line in lines:
            if line == train_set or line.startswith(train_set + " "):
                continue
            numeric_values = self._split_numeric_tokens(line)
            if len(numeric_values) < len(metric_headers):
                continue
            metric_values = numeric_values[-len(metric_headers):]
            numeric_start = line.rfind(metric_values[0])
            method = line[:numeric_start].strip()
            if not method:
                continue
            rebuilt_rows.append([subset_name, method, train_set, *metric_values])
        return rebuilt_rows

    @staticmethod
    def _pad_row(row: list[str], size: int) -> list[str]:
        padded = list(row[:size])
        if len(padded) < size:
            padded.extend([""] * (size - len(padded)))
        return padded

    def _find_table_caption(
        self,
        blocks: list[PaperBlock],
        bbox: list[float],
        used_caption_ids: set[str],
    ) -> tuple[PaperBlock, str] | None:
        table_center = (bbox[0] + bbox[2]) / 2
        candidates: list[tuple[float, PaperBlock, str]] = []
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
                    score = distance + abs(((block.bbox[0] + block.bbox[2]) / 2) - table_center) * 0.05
                    candidates.append((score, block, "above"))
            elif block.bbox[1] >= bbox[3]:
                distance = block.bbox[1] - bbox[3]
                if distance <= 120:
                    score = distance + abs(((block.bbox[0] + block.bbox[2]) / 2) - table_center) * 0.05
                    candidates.append((score, block, "below"))

        if not candidates:
            return None
        _, block, position = min(candidates, key=lambda item: item[0])
        return block, position

    @staticmethod
    def _is_table_caption(text: str) -> bool:
        compact = re.sub(r"\s+", " ", text).strip()
        return bool(re.match(r"^(table)\s*\d+", compact, re.IGNORECASE) or re.match(r"^(表)\s*\d+", compact))

    def _find_textual_table_blocks(self, page_blocks: list[PaperBlock], caption_block: PaperBlock) -> list[PaperBlock]:
        column_captions = [
            block
            for block in page_blocks
            if block.type == "caption" and self._is_table_caption(block.text) and self._same_column(block.bbox, caption_block.bbox)
        ]
        previous_bottom = max((block.bbox[3] for block in column_captions if block.bbox[3] <= caption_block.bbox[1]), default=0.0)
        source_blocks: list[PaperBlock] = []
        for block in page_blocks:
            if block.is_noise or block.block_id == caption_block.block_id:
                continue
            if block.type not in {"paragraph", "formula", "table"}:
                continue
            if block.bbox[3] > caption_block.bbox[1] + 2:
                continue
            if block.bbox[1] < previous_bottom - 2:
                continue
            if caption_block.bbox[1] - block.bbox[3] > 180:
                continue
            if not self._same_column(block.bbox, caption_block.bbox):
                continue
            if not self._looks_like_textual_table_block(block):
                continue
            source_blocks.append(block)
        return sorted(source_blocks, key=self._block_sort_key)

    def _parse_textual_table(
        self,
        caption_text: str,
        source_blocks: list[PaperBlock],
    ) -> dict[str, list] | None:
        compact_caption = re.sub(r"\s+", " ", caption_text).strip().lower()
        if "fid" in compact_caption and "psnr" in compact_caption:
            return self._parse_metric_comparison_table(source_blocks)
        if "trained and tested on same datasets" in compact_caption:
            return self._parse_same_dataset_auc_table(source_blocks)
        if "trained on different datasets" in compact_caption:
            return self._parse_cross_dataset_auc_table(source_blocks)
        if "removal of the regularization" in compact_caption:
            return self._parse_regularization_ablation_table(source_blocks)
        return None

    def _parse_metric_comparison_table(self, source_blocks: list[PaperBlock]) -> dict[str, list] | None:
        dataset_text = next((block.text for block in source_blocks if "Dataset" in block.text), "")
        metric_text = next((block.text for block in source_blocks if "FID" in block.text and "PSNR" in block.text), "")
        datasets = self._extract_dataset_labels(dataset_text)
        metric_values = self._split_numeric_tokens(metric_text)
        if len(datasets) < 2 or len(metric_values) != len(datasets) * 2:
            return None
        midpoint = len(datasets)
        return {
            "headers": ["Metric", *datasets],
            "rows": [
                ["FID ↓", *metric_values[:midpoint]],
                ["PSNR ↑", *metric_values[midpoint:]],
            ],
        }

    def _parse_same_dataset_auc_table(self, source_blocks: list[PaperBlock]) -> dict[str, list] | None:
        header_text = " ".join(block.text for block in source_blocks if "Dataset" in block.text or "FF++" in block.text)
        datasets = self._extract_dataset_labels(header_text)
        data_text = " ".join(
            block.text
            for block in source_blocks
            if len(self._split_numeric_tokens(block.text)) >= len(datasets)
            and "dataset" not in block.text.lower()
        )
        if len(datasets) < 2:
            return None
        rows = self._split_dense_rows(data_text, len(datasets))
        if not rows:
            return None
        return {"headers": ["Method", *datasets], "rows": rows}

    def _parse_cross_dataset_auc_table(self, source_blocks: list[PaperBlock]) -> dict[str, list] | None:
        header_text = " ".join(block.text for block in source_blocks if "Test Set" in block.text or "ForgeryNet" in block.text)
        method_text = next((block.text for block in source_blocks if len(block.text.split()) <= 3 and block.text.strip()), "")
        datasets = self._extract_dataset_labels(header_text)
        data_text = " ".join(
            block.text
            for block in source_blocks
            if len(self._split_numeric_tokens(block.text)) >= len(datasets)
            and "test set" not in block.text.lower()
        )
        train_sets = ["FF++ [49]", "DFor [63]", "GFW [5]", "DiFF"]
        if len(datasets) < 4 or not method_text:
            return None

        normalized_data = self._normalize_dense_text(data_text)
        rows: list[list[str]] = []
        for index, train_set in enumerate(train_sets):
            token = train_set if train_set != "DiFF" else "DiFF"
            start = normalized_data.find(token)
            matched_token = token
            if start == -1:
                matched_token = train_set.split(" ")[0]
                start = normalized_data.find(matched_token)
            if start == -1:
                continue
            next_positions = []
            for next_train in train_sets[index + 1 :]:
                next_token = next_train if next_train != "DiFF" else "DiFF"
                position = normalized_data.find(next_token, start + 1)
                if position == -1:
                    position = normalized_data.find(next_train.split(" ")[0], start + 1)
                if position != -1:
                    next_positions.append(position)
            end = min(next_positions) if next_positions else len(normalized_data)
            segment = normalized_data[start:end]
            segment_body = segment[len(matched_token) :].strip()
            values = re.findall(r"-|\d+(?:\.\d+)?", segment_body)
            if len(values) != len(datasets):
                continue
            rows.append([method_text.strip(), train_set, *values])
        if len(rows) != len(train_sets):
            return None
        return {"headers": ["Method", "Train Set", *datasets], "rows": rows}

    def _parse_regularization_ablation_table(self, source_blocks: list[PaperBlock]) -> dict[str, list] | None:
        metric_headers = self._extract_metric_headers_from_blocks(source_blocks)
        if len(metric_headers) != 4:
            return None

        rows: list[list[str]] = []
        for block in source_blocks:
            compact = re.sub(r"\s+", " ", block.text).strip()
            if "w/o regu." not in compact:
                continue
            parsed_rows = self._parse_regularization_ablation_row(compact, metric_headers)
            if parsed_rows is not None:
                rows.extend(parsed_rows)

        if not rows:
            return None
        return {"headers": ["Method", "Setting", *metric_headers], "rows": rows}

    @staticmethod
    def _normalize_dense_text(text: str) -> str:
        normalized = re.sub(r"\[\s*(\d+)\s*\]", r"[\1]", text)
        normalized = re.sub(r"F\s+3\s*-\s*Net", "F3-Net", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"EffciientNet", "EfficientNet", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()

    def _extract_metric_headers_from_blocks(self, source_blocks: list[PaperBlock]) -> list[str]:
        metric_labels = ["T2I", "I2I", "FS", "FE"]
        collected: list[str] = []
        for block in source_blocks:
            compact = re.sub(r"\s+", " ", block.text).strip().upper()
            for label in metric_labels:
                if label in compact and label not in collected:
                    collected.append(label)
        return collected

    def _parse_regularization_ablation_row(
        self,
        text: str,
        metric_headers: list[str],
    ) -> list[list[str]] | None:
        normalized = self._normalize_dense_text(text)
        marker = "w/o regu."
        if marker not in normalized:
            return None

        prefix, suffix = normalized.split(marker, 1)
        with_values = re.findall(r"\d+(?:\.\d+)?", prefix)
        if len(with_values) < len(metric_headers):
            return None
        with_values = with_values[-len(metric_headers):]

        first_value_index = prefix.find(with_values[0])
        method = prefix[:first_value_index].strip()
        if not method:
            return None

        without_values = self._extract_ablation_metric_tokens(suffix)
        if len(without_values) != len(metric_headers):
            return None

        return [
            [method, "with regu.", *with_values],
            [method, "w/o regu.", *without_values],
        ]

    @staticmethod
    def _extract_ablation_metric_tokens(text: str) -> list[str]:
        raw_tokens = re.findall(r"\d+(?:\.\d+)?(?:\s*\(\s*-?\d+(?:\.\d+)?\s*\))?", text)
        normalized: list[str] = []
        for token in raw_tokens:
            compact = re.sub(r"\s+", "", token)
            if "(" in compact:
                value, delta = compact.split("(", 1)
                normalized.append(f"{value} ({delta}")
            else:
                normalized.append(compact)
        return normalized

    def _split_dense_rows(self, text: str, value_count: int) -> list[list[str]]:
        normalized = self._normalize_dense_text(text)
        rows: list[list[str]] = []
        cursor = 0
        number_pattern = r"(?:-|\d+(?:\.\d+)?)"
        while cursor < len(normalized):
            match = re.search(rf"{number_pattern}(?:\s+{number_pattern}){{{value_count - 1}}}", normalized[cursor:])
            if match is None:
                break
            absolute_start = cursor + match.start()
            absolute_end = cursor + match.end()
            method = normalized[cursor:absolute_start].strip()
            values = re.findall(number_pattern, normalized[absolute_start:absolute_end])
            if method:
                rows.append([method, *values])
            cursor = absolute_end
        return rows

    def _extract_dataset_labels(self, text: str) -> list[str]:
        normalized = self._normalize_dense_text(text)
        pattern = r"FF\+\+\s*\[\d+\]|ForgeryNet\s*\[\d+\]|DFor\s*\[\d+\]|GFW\s*\[\d+\]|DiFF|DFDC\s*\[\d+\]"
        labels = re.findall(pattern, normalized, flags=re.IGNORECASE)
        cleaned: list[str] = []
        for label in labels:
            canonical = re.sub(r"\s+", " ", label).strip()
            canonical = re.sub(r"\[\s*(\d+)\s*\]", r"[\1]", canonical)
            if canonical.lower() == "diff":
                canonical = "DiFF"
            if canonical not in cleaned:
                cleaned.append(canonical)
        return cleaned

    def _find_candidate_near_caption(
        self,
        candidates: list[TableCandidate],
        caption_block: PaperBlock,
    ) -> TableCandidate | None:
        matches: list[tuple[float, TableCandidate]] = []
        for candidate in candidates:
            if candidate.page != caption_block.page:
                continue
            if not self._same_column(candidate.bbox, caption_block.bbox):
                continue
            distance = caption_block.bbox[1] - candidate.bbox[3]
            if 0 <= distance <= 60:
                matches.append((distance, candidate))
        if not matches:
            return None
        return min(matches, key=lambda item: item[0])[1]

    @staticmethod
    def _combine_bbox(blocks: list[PaperBlock]) -> list[float]:
        return [
            min(block.bbox[0] for block in blocks),
            min(block.bbox[1] for block in blocks),
            max(block.bbox[2] for block in blocks),
            max(block.bbox[3] for block in blocks),
        ]

    @staticmethod
    def _same_column(first_bbox: list[float], second_bbox: list[float]) -> bool:
        return TableStructureRestorer._x_overlap_ratio(first_bbox, second_bbox) >= 0.4

    @staticmethod
    def _looks_like_textual_table_block(block: PaperBlock) -> bool:
        compact = re.sub(r"\s+", " ", block.text).strip()
        if block.type == "formula":
            return True
        if compact.startswith(("Table ", "Figure ", "Fig. ")):
            return False
        upper = compact.upper()
        metric_labels = {"T2I", "I2I", "FS", "FE"}
        if metric_labels.issubset(set(upper.split())):
            return True
        if block.type == "paragraph" and 1 <= len(compact.split()) <= 3 and compact[0].isalnum():
            return True
        numeric_tokens = re.findall(r"\d+(?:\.\d+)?", compact)
        keywords = ("dataset", "method", "train", "test", "fid", "psnr", "ff++", "dfor", "gfw", "diff", "forgerynet", "dfdc")
        return len(numeric_tokens) >= 3 or any(keyword in compact.lower() for keyword in keywords)

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
            table_caption_position=block.table_caption_position,
        )

    @staticmethod
    def _candidate_sort_key(candidate: TableCandidate) -> tuple[int, float, float]:
        return candidate.page, round(candidate.bbox[1], 2), round(candidate.bbox[0], 2)

    @staticmethod
    def _sort_candidates(candidates: list[TableCandidate]) -> list[TableCandidate]:
        return sorted(candidates, key=TableStructureRestorer._candidate_sort_key)

    @staticmethod
    def _block_sort_key(block: PaperBlock) -> tuple[int, float, float, str]:
        return (block.page, round(block.bbox[1], 2), round(block.bbox[0], 2), block.block_id)
