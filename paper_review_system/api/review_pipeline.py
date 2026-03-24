from __future__ import annotations

from pathlib import Path

from paper_review_system.io_utils import ensure_directory, write_json, write_text
from paper_review_system.logic.analyzer import LogicAnalyzer
from paper_review_system.models import EvidenceBundle, PaperDocument
from paper_review_system.parser.anchor_builder import AnchorBuilder
from paper_review_system.parser.markdown_renderer import MarkdownRenderer
from paper_review_system.parser.noise_cleaner import NoiseCleaner
from paper_review_system.parser.pdf_parser import PDFParser
from paper_review_system.parser.section_builder import SectionTreeBuilder
from paper_review_system.parser.table_reconstructor import TableStructureRestorer
from paper_review_system.report.assembler import ReportAssembler
from paper_review_system.retrieval.planner import ImprovementPlanner
from paper_review_system.rules.engine import RuleEngine


class ReviewPipeline:
    """End-to-end pipeline for conversion and review report generation."""

    def __init__(self) -> None:
        self.pdf_parser = PDFParser()
        self.noise_cleaner = NoiseCleaner()
        self.table_reconstructor = TableStructureRestorer()
        self.section_builder = SectionTreeBuilder()
        self.anchor_builder = AnchorBuilder()
        self.markdown_renderer = MarkdownRenderer()
        self.logic_analyzer = LogicAnalyzer()
        self.rule_engine = RuleEngine()
        self.improvement_planner = ImprovementPlanner()
        self.report_assembler = ReportAssembler()

    def convert_pdf(self, pdf_path: str | Path, output_dir: str | Path) -> dict[str, str]:
        output_root = ensure_directory(Path(output_dir))
        document = self.pdf_parser.parse(pdf_path)
        evidence = self._build_evidence(document)
        return self._write_conversion_artifacts(output_root, document, evidence)

    def review_pdf(self, pdf_path: str | Path, output_dir: str | Path) -> dict[str, str]:
        output_root = ensure_directory(Path(output_dir))
        document = self.pdf_parser.parse(pdf_path)
        evidence = self._build_evidence(document)
        convert_result = self._write_conversion_artifacts(output_root, document, evidence)
        logic_analysis = self.logic_analyzer.analyze(evidence.clean_blocks, evidence.anchors)
        rule_violations = self.rule_engine.scan(evidence.clean_blocks)
        improvement_plan = self.improvement_planner.plan(logic_analysis, rule_violations)
        report = self.report_assembler.assemble(logic_analysis, rule_violations, improvement_plan)

        review_report_path = output_root / "review_report.json"
        write_json(review_report_path, report.to_dict())

        convert_result["review_report_path"] = str(review_report_path.resolve())
        return convert_result

    def _build_evidence(self, document: PaperDocument) -> EvidenceBundle:
        clean_blocks = self.noise_cleaner.clean(document)
        clean_blocks = self.table_reconstructor.restore(document.source_file, clean_blocks, document.pages)
        section_tree = self.section_builder.build(clean_blocks)
        anchors = self.anchor_builder.build(clean_blocks, section_tree)
        return EvidenceBundle(
            doc_id=document.doc_id,
            anchors=anchors,
            clean_blocks=clean_blocks,
            raw_blocks=document.blocks,
            section_tree=section_tree,
        )

    def _write_conversion_artifacts(
        self,
        output_root: Path,
        document: PaperDocument,
        evidence: EvidenceBundle,
    ) -> dict[str, str]:
        markdown = self.markdown_renderer.render(evidence.clean_blocks)
        document_ir_path = output_root / "document_ir.json"
        evidence_ir_path = output_root / "evidence_ir.json"
        markdown_path = output_root / "paper.md"

        write_json(document_ir_path, document.to_dict())
        write_json(evidence_ir_path, evidence.to_dict())
        write_text(markdown_path, markdown)

        return {
            "document_ir_path": str(document_ir_path.resolve()),
            "evidence_ir_path": str(evidence_ir_path.resolve()),
            "markdown_path": str(markdown_path.resolve()),
        }
