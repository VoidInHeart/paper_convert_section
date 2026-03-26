from __future__ import annotations

import unittest

from paper_review_system.models import PaperBlock
from paper_review_system.models import PageInfo
from paper_review_system.parser.markdown_renderer import MarkdownRenderer
from paper_review_system.parser.pdf_parser import PDFParser
from paper_review_system.parser.reading_order import ReadingOrderResolver
from paper_review_system.parser.table_reconstructor import TableCandidate, TableStructureRestorer
from paper_review_system.rules.grammar_rules import GrammarRuleChecker
from paper_review_system.rules.violation_id import ViolationIdAllocator


class MarkdownRendererTest(unittest.TestCase):
    def test_render_skips_noise_and_formats_headings(self) -> None:
        blocks = [
            PaperBlock(block_id="1", page=1, bbox=[0, 0, 1, 1], type="heading", text="摘要", level=2),
            PaperBlock(block_id="2", page=1, bbox=[0, 0, 1, 1], type="paragraph", text="这是正文。"),
            PaperBlock(block_id="3", page=1, bbox=[0, 0, 1, 1], type="paragraph", text="页码", is_noise=True),
        ]
        markdown = MarkdownRenderer().render(blocks)
        self.assertIn("## 摘要", markdown)
        self.assertIn("这是正文。", markdown)
        self.assertNotIn("页码", markdown)

    def test_render_formats_captions_as_quotes(self) -> None:
        blocks = [PaperBlock(block_id="1", page=1, bbox=[0, 0, 1, 1], type="caption", text="Figure 1. Demo caption")]
        markdown = MarkdownRenderer().render(blocks)
        self.assertIn("> Figure 1. Demo caption", markdown)

    def test_pdf_text_normalization_merges_hyphenated_lines(self) -> None:
        text = "diffu-\nsion 鈥?driven"
        normalized = PDFParser._normalize_text(text)
        self.assertEqual(normalized, "diffusion -driven")

    def test_grammar_checker_skips_caption_like_blocks(self) -> None:
        blocks = [
            PaperBlock(
                block_id="1",
                page=1,
                bbox=[0, 0, 1, 1],
                type="paragraph",
                text="Figure 1. " + ("caption " * 60),
                role="caption",
            )
        ]
        violations = GrammarRuleChecker().check(blocks, ViolationIdAllocator())
        self.assertEqual(violations, [])

    def test_render_outputs_markdown_table(self) -> None:
        block = PaperBlock(
            block_id="tbl_1",
            page=1,
            bbox=[0, 0, 10, 10],
            type="table",
            text="Method\tScore\nA\t91",
            table_headers=["Method", "Score"],
            table_rows=[["A", "91"], ["B", "88"]],
            table_caption="Table 1. Demo",
        )
        markdown = MarkdownRenderer().render([block])
        self.assertIn("> Table 1. Demo", markdown)
        self.assertIn("| Method | Score |", markdown)
        self.assertIn("| A | 91 |", markdown)

    def test_table_reconstructor_extracts_headers_and_expands_rows(self) -> None:
        class FakeHeader:
            names = ["Method", "Score"]

        class FakeDetectedTable:
            header = FakeHeader()

            def extract(self) -> list[list[str | None]]:
                return [["A\nB", "91\n88"], ["C", "77"]]

        payload = TableStructureRestorer()._extract_table_payload(FakeDetectedTable())
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["headers"], ["Method", "Score"])

    def test_chinese_space_fix_and_indent(self) -> None:
        blocks = [
            PaperBlock(block_id="1", page=1, bbox=[0, 0, 1, 1], type="paragraph", text="分 类 号 学号\n今 天 天 气 好"),
        ]
        markdown = MarkdownRenderer().render(blocks)
        self.assertIn("　分类号学号", markdown)
        self.assertIn("　今天天气好", markdown)

    def test_add_page_marker_for_each_page(self) -> None:
        blocks = [
            PaperBlock(block_id="1", page=1, bbox=[0, 0, 1, 1], type="paragraph", text="A"),
            PaperBlock(block_id="2", page=2, bbox=[0, 0, 1, 1], type="paragraph", text="B"),
        ]
        markdown = MarkdownRenderer().render(blocks)
        self.assertIn("【page 1】", markdown)
        self.assertIn("【page 2】", markdown)

    def test_preserve_paragraph_line_breaks(self) -> None:
        blocks = [
            PaperBlock(block_id="1", page=1, bbox=[0, 0, 1, 1], type="paragraph", text="x\ny"),
            PaperBlock(block_id="2", page=1, bbox=[0, 0, 1, 1], type="paragraph", text="z"),
        ]
        markdown = MarkdownRenderer().render(blocks)
        self.assertIn("　x\n　y", markdown)
        self.assertIn("　z", markdown)

    def test_table_reconstructor_merges_adjacent_candidates(self) -> None:
        restorer = TableStructureRestorer()
        merged = restorer._merge_candidates(
            [
                TableCandidate(page=4, bbox=[300, 90, 450, 160], headers=["Method", "#Images"], rows=[["A", "10"]]),
                TableCandidate(page=4, bbox=[300, 185, 450, 230], headers=["Column 1", "Column 2"], rows=[["B", "12"]]),
            ]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].headers, ["Method", "#Images"])
        self.assertEqual(merged[0].rows, [["A", "10"], ["B", "12"]])

    def test_table_reconstructor_binds_caption_above_or_below(self) -> None:
        restorer = TableStructureRestorer()
        candidate = TableCandidate(page=1, bbox=[100, 100, 300, 220], headers=["Method", "Score"], rows=[["A", "1"]])
        blocks = {
            1: [
                PaperBlock(block_id="cap1", page=1, bbox=[110, 225, 290, 240], type="caption", text="Table 2. Demo table"),
                PaperBlock(block_id="cap2", page=1, bbox=[110, 20, 290, 40], type="caption", text="Figure 1. Ignore me"),
            ]
        }
        restorer._bind_captions([candidate], blocks)
        self.assertEqual(candidate.caption, "Table 2. Demo table")
        self.assertTrue(blocks[1][0].is_noise)
        self.assertEqual(blocks[1][0].role, "table_caption_bound")

    def test_table_reconstructor_repairs_compressed_metric_table(self) -> None:
        restorer = TableStructureRestorer()
        headers = ["Column 1", "Column 2", "Column 3", "Column 4"]
        rows = [
            ["Processing Method <br> Method Train <br> Subset None GN GB MB JPEG", "", "", ""],
            ["Xception", "T2I", "59.52", "47.65 15.02 56.59 58.69"],
            ["F3-Net", "", "", "48.04 74.67 71.68 74.61"],
            ["EfficientNet", "", "", "40.09 53.62 65.35 54.98"],
            ["DIRE", "", "", "34.07 32.78 41.36 40.99"],
            ["", "", "76.08", ""],
            ["", "", "67.69", ""],
            ["", "", "66.28", ""],
        ]
        repaired_headers, repaired_rows = restorer._repair_compressed_metric_table(headers, rows, [["Processing Method", "Method", "Train", "Subset None GN GB MB JPEG"]])
        self.assertEqual(repaired_headers, ["Method", "Train Subset", "None", "GN", "GB", "MB", "JPEG"])
        self.assertEqual(repaired_rows[0], ["Xception", "T2I", "59.52", "47.65", "15.02", "56.59", "58.69"])
        self.assertEqual(repaired_rows[1], ["F3-Net", "T2I", "76.08", "48.04", "74.67", "71.68", "74.61"])

    def test_reading_order_prefers_left_column_before_right_column(self) -> None:
        resolver = ReadingOrderResolver()
        blocks = [
            PaperBlock(block_id="right_top", page=1, bbox=[310, 80, 540, 140], type="paragraph", text="right top " * 20),
            PaperBlock(block_id="left_low", page=1, bbox=[50, 220, 280, 320], type="paragraph", text="left low " * 20),
            PaperBlock(block_id="left_top", page=1, bbox=[50, 90, 280, 160], type="paragraph", text="left top " * 20),
            PaperBlock(block_id="right_low", page=1, bbox=[310, 230, 540, 300], type="paragraph", text="right low " * 20),
        ]
        ordered = resolver.order_blocks(blocks, [PageInfo(page=1, width=595, height=842)])
        self.assertEqual([block.block_id for block in ordered], ["left_top", "left_low", "right_top", "right_low"])

    def test_long_multiline_text_is_not_heading(self) -> None:
        block_type, level, role = PDFParser._classify_block(
            text="This is a long paragraph that should remain body text even if the font size is slightly larger than normal and spans multiple lines.",
            raw_text="This is a long paragraph\nthat should remain body text\neven if the font size is slightly larger.",
            font_size=12.5,
            body_size=10.0,
            page_number=7,
            block_index=5,
            line_count=3,
        )
        self.assertEqual((block_type, level, role), ("paragraph", None, "body"))


if __name__ == "__main__":
    unittest.main()
