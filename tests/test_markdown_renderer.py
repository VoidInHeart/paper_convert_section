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

    def test_render_places_below_caption_after_table(self) -> None:
        block = PaperBlock(
            block_id="tbl_1",
            page=1,
            bbox=[0, 0, 10, 10],
            type="table",
            text="Method\tScore\nA\t91",
            table_headers=["Method", "Score"],
            table_rows=[["A", "91"]],
            table_caption="Table 2. Below",
            table_caption_position="below",
        )
        markdown = MarkdownRenderer().render([block])
        self.assertLess(markdown.find("| Method | Score |"), markdown.find("> Table 2. Below"))

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
        self.assertEqual(candidate.caption_position, "below")
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

    def test_table_reconstructor_repairs_subset_matrix_table(self) -> None:
        restorer = TableStructureRestorer()
        raw_rows = [
            ["Deepfake Test Subset\nMethod Train Set FF++ T2I I2I FS FE", "", "", ""],
            ["", "Train Set", "", ""],
            ["Xception† [49]\nF3-Net† [43]\nEfficientNet† [58]\nDIRE‡ [63]", "FF++ [49]", "98.12", "62.43 56.83 85.97 58.64\n66.87 67.64 81.01 60.60\n74.12 57.27 82.11 57.20\n44.22 64.64 84.98 57.72"],
            ["", "", "98.89", ""],
            ["", "", "98.51", ""],
            ["", "", "99.43", ""],
            ["General Diffusion Test Subset\nMethod Train Set DFor T2I I2I FS FE", "", "", ""],
            ["Xception† [49] 99.98 20.52 30.92 69.42 37.89\nF3-Net† [43] 99.99 43.88 60.58 52.39 47.06\nDFor [63]\nEfficientNet† [58] 98.99 27.23 44.79 61.25 30.86\nDIRE‡ [63] 98.80 36.37 34.83 36.28 39.92", "", "", ""],
        ]
        repaired_headers, repaired_rows = restorer._repair_subset_matrix_table(None, [], raw_rows)
        self.assertEqual(repaired_headers, ["Test Subset", "Method", "Train Set", "Source Set", "T2I", "I2I", "FS", "FE"])
        self.assertEqual(repaired_rows[0], ["Deepfake Test Subset", "Xception† [49]", "FF++ [49]", "98.12", "62.43", "56.83", "85.97", "58.64"])
        self.assertEqual(repaired_rows[4], ["General Diffusion Test Subset", "Xception† [49]", "DFor", "99.98", "20.52", "30.92", "69.42", "37.89"])

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

    def test_reading_order_keeps_column_tables_out_of_top_full_width_bucket(self) -> None:
        resolver = ReadingOrderResolver()
        blocks = [
            PaperBlock(block_id="left_body", page=1, bbox=[50, 350, 280, 430], type="paragraph", text="left body " * 30),
            PaperBlock(block_id="right_body", page=1, bbox=[310, 340, 540, 420], type="paragraph", text="right body " * 30),
            PaperBlock(block_id="left_table_source", page=1, bbox=[56, 90, 278, 150], type="paragraph", text="dataset 1 2 3 4 5"),
            PaperBlock(block_id="right_table", page=1, bbox=[315, 72, 545, 208], type="table", text="a\tb", table_headers=["A", "B"], table_rows=[["1", "2"]]),
        ]
        ordered = resolver.order_blocks(blocks, [PageInfo(page=1, width=595, height=842)])
        self.assertEqual([block.block_id for block in ordered], ["left_table_source", "left_body", "right_table", "right_body"])

    def test_parse_textual_table_4_style(self) -> None:
        restorer = TableStructureRestorer()
        blocks = [
            PaperBlock(block_id="1", page=1, bbox=[0, 0, 1, 1], type="formula", text="Dataset FF++ [ 49 ] ForgeryNet [ 18 ] DFor [ 63 ] GFW [ 5 ] DiFF"),
            PaperBlock(block_id="2", page=1, bbox=[0, 0, 1, 1], type="paragraph", text="FID ↓ 33.87 36.94 31.79 39.35 25.75 PSNR ↑ 18.47 18.98 19.17 19.14 19.95"),
        ]
        parsed = restorer._parse_textual_table("Table 4. FID and PSNR comparison across various datasets.", blocks)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["headers"], ["Metric", "FF++ [49]", "ForgeryNet [18]", "DFor [63]", "GFW [5]", "DiFF"])
        self.assertEqual(parsed["rows"][0], ["FID ↓", "33.87", "36.94", "31.79", "39.35", "25.75"])

    def test_parse_textual_table_5_style(self) -> None:
        restorer = TableStructureRestorer()
        blocks = [
            PaperBlock(block_id="1", page=1, bbox=[0, 0, 1, 1], type="paragraph", text="Method Dataset"),
            PaperBlock(block_id="2", page=1, bbox=[0, 0, 1, 1], type="formula", text="FF++ [ 49 ] GFW [ 5 ] DiFF"),
            PaperBlock(block_id="3", page=1, bbox=[0, 0, 1, 1], type="paragraph", text="Xception 98.12 99.72 93.87 F 3 -Net 98.89 99.17 98.47 EfficientNet 98.51 97.58 94.34 DIRE 99.43 99.59 96.35"),
        ]
        parsed = restorer._parse_textual_table("Table 5. AUC (%) of detectors trained and tested on same datasets.", blocks)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["headers"], ["Method", "FF++ [49]", "GFW [5]", "DiFF"])
        self.assertEqual(parsed["rows"][1], ["F3-Net", "98.89", "99.17", "98.47"])

    def test_parse_textual_table_6_style(self) -> None:
        restorer = TableStructureRestorer()
        blocks = [
            PaperBlock(block_id="1", page=1, bbox=[0, 0, 1, 1], type="paragraph", text="Method Train Test Set"),
            PaperBlock(block_id="2", page=1, bbox=[0, 0, 1, 1], type="formula", text="Set FF++ [ 49 ] DFor [ 63 ] GFW [ 5 ] DiFF DFDC [ 13 ] ForgeryNet [ 18 ]"),
            PaperBlock(block_id="3", page=1, bbox=[0, 0, 1, 1], type="paragraph", text="Xception"),
            PaperBlock(block_id="4", page=1, bbox=[0, 0, 1, 1], type="paragraph", text="FF++ - 40.65 43.42 65.96 63.97 50.56 DFor 55.21 - 52.30 75.67 56.35 38.06 GFW 53.37 45.81 - 74.87 51.43 62.75 DiFF 65.33 55.30 63.50 - 67.10 65.78"),
        ]
        parsed = restorer._parse_textual_table("Table 6. AUC (%) of detectors trained on different datasets.", blocks)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["headers"], ["Method", "Train Set", "FF++ [49]", "DFor [63]", "GFW [5]", "DiFF", "DFDC [13]", "ForgeryNet [18]"])
        self.assertEqual(parsed["rows"][0], ["Xception", "FF++ [49]", "-", "40.65", "43.42", "65.96", "63.97", "50.56"])

    def test_find_candidate_near_caption_prefers_nearest_table(self) -> None:
        restorer = TableStructureRestorer()
        caption = PaperBlock(block_id="cap", page=1, bbox=[60, 220, 260, 230], type="caption", text="Table 2. Demo")
        candidates = [
            TableCandidate(page=1, bbox=[60, 80, 260, 100], headers=["A"], rows=[["1"]]),
            TableCandidate(page=1, bbox=[60, 170, 260, 205], headers=["B"], rows=[["2"]]),
        ]
        target = restorer._find_candidate_near_caption(candidates, caption)
        self.assertIsNotNone(target)
        assert target is not None
        self.assertEqual(target.headers, ["B"])

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
