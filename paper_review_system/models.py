from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def _drop_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _drop_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [_drop_none(item) for item in value]
    return value


@dataclass(slots=True)
class PageInfo:
    page: int
    width: float
    height: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PaperBlock:
    block_id: str
    page: int
    bbox: list[float]
    type: str
    text: str
    level: int | None = None
    is_noise: bool = False
    font_size: float | None = None
    source: str = "pdf"
    role: str | None = None
    table_headers: list[str] | None = None
    table_rows: list[list[str]] | None = None
    table_caption: str | None = None
    table_caption_position: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass(slots=True)
class SectionNode:
    section_id: str
    title: str
    level: int
    page_start: int
    page_end: int
    block_ids: list[str] = field(default_factory=list)
    parent_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass(slots=True)
class PaperAnchor:
    anchor_id: str
    block_id: str
    page: int
    bbox: list[float]
    text: str
    section_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass(slots=True)
class PaperDocument:
    doc_id: str
    source_file: str
    pages: list[PageInfo]
    blocks: list[PaperBlock]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "source_file": self.source_file,
            "pages": [page.to_dict() for page in self.pages],
            "blocks": [block.to_dict() for block in self.blocks],
            "metadata": _drop_none(self.metadata),
        }


@dataclass(slots=True)
class EvidenceBundle:
    doc_id: str
    anchors: list[PaperAnchor]
    clean_blocks: list[PaperBlock]
    raw_blocks: list[PaperBlock]
    section_tree: list[SectionNode]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "anchors": [anchor.to_dict() for anchor in self.anchors],
            "clean_blocks": [block.to_dict() for block in self.clean_blocks],
            "raw_blocks": [block.to_dict() for block in self.raw_blocks],
            "section_tree": [node.to_dict() for node in self.section_tree],
        }


@dataclass(slots=True)
class LogicDetail:
    logical_node: str
    severity: str
    analysis: str
    evidence_links: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_node": self.logical_node,
            "severity": self.severity,
            "analysis": self.analysis,
            "evidence_links": self.evidence_links or [],
        }


@dataclass(slots=True)
class CoreArgumentConsistency:
    is_consistent: bool
    conflict_summary: str
    details: list[LogicDetail] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_consistent": self.is_consistent,
            "conflict_summary": self.conflict_summary,
            "details": [detail.to_dict() for detail in self.details],
        }


@dataclass(slots=True)
class ReasoningDepth:
    assessment: str
    suggestion: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class LogicAnalysis:
    academic_integrity_score: int
    core_argument_consistency: CoreArgumentConsistency
    reasoning_depth: ReasoningDepth

    def to_dict(self) -> dict[str, Any]:
        return {
            "academic_integrity_score": self.academic_integrity_score,
            "core_argument_consistency": self.core_argument_consistency.to_dict(),
            "reasoning_depth": self.reasoning_depth.to_dict(),
        }


@dataclass(slots=True)
class Violation:
    id: str
    category: str
    title: str
    location: str
    description: str
    fix_type: str
    original_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return _drop_none(asdict(self))


@dataclass(slots=True)
class SemanticRefinement:
    target_ref: str
    action: str
    content: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExpertSample:
    topic: str
    source_paper: str
    value_proposition: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ImprovementPlan:
    semantic_refinement: list[SemanticRefinement] = field(default_factory=list)
    expert_samples: list[ExpertSample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_refinement": [item.to_dict() for item in self.semantic_refinement],
            "expert_samples": [item.to_dict() for item in self.expert_samples],
        }


@dataclass(slots=True)
class ReviewReport:
    project_metadata: dict[str, Any]
    logic_analysis: LogicAnalysis
    rule_violations: list[Violation]
    improvement_plan: ImprovementPlan

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_metadata": _drop_none(self.project_metadata),
            "logic_analysis": self.logic_analysis.to_dict(),
            "rule_violations": [item.to_dict() for item in self.rule_violations],
            "improvement_plan": self.improvement_plan.to_dict(),
        }


def build_doc_id(source_file: str) -> str:
    stem = Path(source_file).stem
    safe_stem = "".join(ch if ch.isalnum() else "_" for ch in stem).strip("_") or "paper"
    return safe_stem.lower()
