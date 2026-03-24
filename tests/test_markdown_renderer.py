from __future__ import annotations

import unittest

from paper_review_system.models import PaperBlock
from paper_review_system.parser.markdown_renderer import MarkdownRenderer
from paper_review_system.parser.pdf_parser import PDFParser
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
        self.assertEqual(payload["rows"], [["A", "91"], ["B", "88"], ["C", "77"]])

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


if __name__ == "__main__":
    unittest.main()
