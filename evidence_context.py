"""Bounded context-complete source-span construction.

Inputs are one authorized source document/chunk.  The module has no access to
conversation history, planner output, tenant lookup, or other sources, so it
cannot accidentally promote any of them into evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import re
from typing import Any, Mapping, Optional

from answer_units import GROUPED_DECIMAL_LITERAL_PATTERN, AtomicFactSpan, atomic_fact_spans
from proposition_quality import assess_proposition, balanced_delimiters
from structural_evidence import (
    StructuralEvidenceType,
    StructuralEvidenceUnit,
    atomic_structural_evidence_unit,
)


# Keep every expanded span eligible for a verbatim citation. These bounds
# intentionally match the quote verifier's hard per-citation limits.
MAX_CONTEXT_WORDS = 80
MAX_CONTEXT_CHARS = 800

_BREADCRUMB_RE = re.compile(r"\A(?:[ \t]*\[[^\]\r\n]{1,500}\][ \t]*(?:\r?\n|$))+")
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?׃])(?:[ \t\r\n]+)(?=\S)")
_HEBREW_RE = re.compile(r"[\u0590-\u05ff]")
_LETTER_RE = re.compile(r"[A-Za-z\u05d0-\u05ea]")
_NUMBER_RE = re.compile(rf"(?:{GROUPED_DECIMAL_LITERAL_PATTERN}|\d+(?:[.,]\d+)?)")
_PERCENT_RE = re.compile(r"(?:%|אחוז(?:ים)?|\bpercent(?:age)?\b)", re.IGNORECASE)
_POPULATION_RE = re.compile(
    r"(?:חברות|עובדים|משיבים|משתתפים|ארגונים|אנשים|תלמידים|לקוחות|"
    r"אוכלוסי(?:ה|ית)|מדגם|קבוצה|מועמדים|משרות|מעסיקים|"
    r"\b(?:companies|workers|employees|employers|respondents|participants|organizations|"
    r"people|students|customers|population|sample|group|candidates|roles)\b)",
    re.IGNORECASE,
)
_UNRESOLVED_RE = re.compile(
    r"(?:"
    r"מרבית\s+החברות|רוב\s+החברות|(?:^|\s)החברות(?:\s|$)|מהן|(?:^|\s)הן(?:\s|$)|"
    r"(?:^|\s)(?:אלה|אלו|זו|זה|אותם|שלהן)(?:\s|$)|הנתון|הממצא|"
    r"הסקר\s+(?:הנוכחי|הקודם)|בתחום\s+זה|בענף\s+זה|בקבוצה\s+זו|"
    r"\bthe\s+companies\b|\bmost\s+of\s+them\b|\b(?:they|these|those)\b|"
    r"\bthis\s+(?:result|finding)\b|\bthe\s+(?:current|previous)\s+survey\b|"
    r"\bin\s+this\s+(?:sector|group)\b"
    r")",
    re.IGNORECASE,
)
_DISCOURSE_START_RE = re.compile(
    r"^\s*[\(\[\{),;:–—-]*\s*(?:"
    r"ו\s+(?=\S)|ו(?:ירידה|עלייה|גידול|קיטון|בשנת)|"
    r"כאמור|בנוסף|כמו\s+כן|לעומת\s+זאת|בהשוואה\s+לכך|"
    r"and\b|but\b|while\b|however|additionally|also|"
    r"in\s+contrast|compared\s+with\s+that)\b",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"(?:לעומת|בהשוואה|בין|קיטון|ירידה|גידול|עליי?ה|"
    r"\b(?:compared|versus|between|decrease|decline|increase|growth|"
    r"fell|grew)\b)",
    re.IGNORECASE,
)
_DELTA_RE = re.compile(
    r"(?:"
    r"(?:ירד|עלה|פחת|גדל|קיטון|ירידה|גידול|עליי?ה)\s+(?:ב|של)\s*-?\d|"
    r"\b(?:fell|grew|increased?|decreased?|declined?|change[ds]?)\s+"
    r"(?:by|of)\s+-?\d"
    r")",
    re.IGNORECASE,
)
_PRIMARY_RATE_THEN_QUALITATIVE_COMPARISON_RE = re.compile(
    r"(?:%|אחוז(?:ים)?|\bpercent(?:age)?\b).{0,180}"
    r"(?:לעומת|בהשוואה|קיטון|ירידה|גידול|עליי?ה|"
    r"\b(?:compared|increase|decrease|decline|growth)\b)",
    re.IGNORECASE,
)
_PREDICATE_RE = re.compile(
    r"(?:הוא|היא|הם|הן|יש|אין|היה|היו|מעסיק|משויכ|עוסק|נמצא|מראה|"
    r"מצביע|מהווה|כולל|עלה|ירד|גדל|פחת|חל|נדרש|צפוי|"
    r"\b(?:is|are|was|were|has|have|employs?|belongs?|found|shows?|"
    r"indicates?|represents?|includes?|increased?|decreased?|changed?)\b)",
    re.IGNORECASE,
)
_HEADING_MARKDOWN_RE = re.compile(r"^\s*#{1,6}\s+")
_BULLET_RE = re.compile(r"^\s*[•▪◦*-]\s+")
_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")


# One authoritative schema for every textual relation whose predicate alone
# cannot prove the requested claim.  Extractors and the claim gate consume the
# same mapping; a downstream route must not reinterpret an incomplete fact as
# a complete one.
RELATION_ARGUMENT_SCHEMAS: Mapping[str, tuple[str, ...]] = {
    "caused_by": ("effect", "reason"),
    "explained_as": ("explained_event", "explanation"),
    "purpose_of": ("purpose_target", "purpose"),
    "defined_as": ("defined_entity", "definition"),
    "performed_by": ("procedure_target", "method"),
    "excluded_from": ("excluded_entity", "excluded_scope"),
    "exception_to": ("excluded_entity", "excluded_scope"),
    "included_in": ("included_entity", "included_scope"),
    "funded_by": ("funding_target", "funder"),
    "published_by": ("publication_target", "publisher"),
    "authored_by": ("authored_target", "author"),
}

RELATION_ARGUMENT_ROLE_DESCRIPTIONS: Mapping[str, str] = {
    "effect": "the event or condition whose cause is stated",
    "reason": "the cause or reason for the effect",
    "explained_event": "the event or condition being explained",
    "explanation": "the interpretation or explanation of that event",
    "purpose_target": "the entity, program, study, or process whose purpose is stated",
    "purpose": "the goal, intended use, or action served by the purpose target",
    "defined_entity": "the term or entity being defined",
    "definition": "the meaning assigned to the defined entity",
    "procedure_target": "the population, measure, or process whose method is stated",
    "method": "the method used for the procedure target",
    "excluded_entity": "the entity that is excluded or omitted",
    "excluded_scope": "the set, sample, or scope from which the entity is excluded",
    "included_entity": "the entity that is included",
    "included_scope": "the set, sample, or scope in which the entity is included",
    "funding_target": "the report, study, or activity whose funding is explicitly stated",
    "funder": "the explicitly stated funding organization or person",
    "publication_target": "the document whose publication is explicitly stated",
    "publisher": "the explicitly stated publisher",
    "authored_target": "the document whose authorship is explicitly stated",
    "author": "the explicitly stated author",
}


@dataclass(frozen=True)
class FieldProvenance:
    """Exact evidence origin for a local or structurally inherited field."""

    field: str
    value: str
    provenance_type: str
    source_container: str
    source_span: tuple[int, int]
    source_sha256: str
    confidence: float
    revisability: str = "revisable"
    source_quote: str = ""
    derivation_type: str = "literal"
    structural_unit_id: str = ""
    structural_group_id: str = ""
    structural_role: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "value": self.value,
            "provenance_type": self.provenance_type,
            "source_container": self.source_container,
            "source_span": list(self.source_span),
            "source_sha256": self.source_sha256,
            "confidence": self.confidence,
            "revisability": self.revisability,
            "source_quote": self.source_quote,
            "derivation_type": self.derivation_type,
            "structural_unit_id": self.structural_unit_id,
            "structural_group_id": self.structural_group_id,
            "structural_role": self.structural_role,
        }


@dataclass(frozen=True)
class RelationArgument:
    """One source-grounded role in a relational proposition."""

    role: str
    value: str
    provenance: FieldProvenance

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "value": self.value,
            "provenance": self.provenance.as_dict(),
        }


class EvidenceSemanticValidationError(ValueError):
    """Stable validation failure from the bounded evidence resolver."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = str(reason_code)
        super().__init__(self.reason_code)


@dataclass(frozen=True)
class EvidenceSemanticResolverResult:
    """One bounded resolver outcome plus safe transport/protocol diagnostics."""

    resolution: Optional[Mapping[str, Any]]
    diagnostics: Mapping[str, Any]
    reason_code: str

    @classmethod
    def not_started(
        cls,
        *,
        reason_code: str,
        call_id: str,
    ) -> "EvidenceSemanticResolverResult":
        return cls(
            resolution=None,
            diagnostics={
                "resolver_call_started": False,
                "resolver_call_id": str(call_id),
                "resolver_failure_stage": "admission",
                "resolver_transport_status": "not_started",
                "resolver_parse_status": "not_attempted",
                "resolver_schema_status": "not_attempted",
                "resolver_quote_status": "not_attempted",
                "resolver_fact_status": "not_attempted",
            },
            reason_code=str(reason_code),
        )


@dataclass(frozen=True)
class EvidenceContext:
    """Immutable, document-derived semantic context for one evidence unit.

    Request slots are intentionally absent. Compatibility is evaluated only
    after this object has been constructed from the source proposition and its
    bounded structural labels.
    """

    evidence_unit_id: str
    candidate_id: str
    source_id: str
    chunk_id: str
    page: Optional[int]
    structural_type: str
    local_proposition: str
    proposition_span: tuple[int, int]
    proposition_sha256: str
    section_title: str = ""
    section_provenance: Optional[FieldProvenance] = None
    table_chart_title: str = ""
    title_provenance: Optional[FieldProvenance] = None
    row_header: str = ""
    row_header_provenance: Optional[FieldProvenance] = None
    column_header: str = ""
    column_header_provenance: Optional[FieldProvenance] = None
    category: str = ""
    category_provenance: Optional[FieldProvenance] = None
    period: str = ""
    period_provenance: Optional[FieldProvenance] = None
    population: str = ""
    population_provenance: Optional[FieldProvenance] = None
    evidence_subject: str = ""
    evidence_predicate: str = ""
    evidence_target_entity: str = ""
    evidence_unit: str = ""
    evidence_numeric_role: str = ""
    evidence_polarity: str = "neutral"
    semantic_provenance: tuple[FieldProvenance, ...] = ()
    relation_arguments: tuple[RelationArgument, ...] = ()
    model_assisted: bool = False
    structural_evidence_unit: Optional[StructuralEvidenceUnit] = field(
        default=None, repr=False, compare=False
    )
    # This is an audit-only input.  Runtime consumers must use
    # ``structural_type`` which is canonicalized from the structural unit in
    # ``__post_init__``.  Retaining the supplied value makes a bad hand-off
    # observable without creating a second active type authority.
    structural_type_input: str = ""
    structural_validation_errors: tuple[str, ...] = ()
    governing_context: str = ""
    governing_context_provenance: Optional[FieldProvenance] = None
    source_candidate_text: str = field(default="", repr=False, compare=False)

    def __post_init__(self) -> None:
        start, end = self.proposition_span
        if start < 0 or end < start:
            raise ValueError("invalid EvidenceContext proposition span")
        expected = hashlib.sha256(
            self.local_proposition.encode("utf-8")
        ).hexdigest()
        if self.proposition_sha256 != expected:
            raise ValueError("EvidenceContext proposition hash mismatch")
        for provenance in self.all_provenance:
            if not provenance.source_sha256 or provenance.source_span[0] < 0:
                raise ValueError("invalid EvidenceContext field provenance")
        structural_errors = list(self.structural_validation_errors)
        unit = self.structural_evidence_unit
        if unit is not None:
            canonical_type = unit.structural_type.value
            supplied_type = str(
                self.structural_type_input or self.structural_type or ""
            )
            # ``body`` and ``paragraph`` were generic pre-canonical adapters,
            # not assertions about a more specific source structure.  They
            # are intentionally treated as unspecified at this boundary.
            if supplied_type.casefold() not in {"", "body", "paragraph"} and (
                supplied_type != canonical_type
            ):
                structural_errors.append("structural_type_transition_mismatch")
            object.__setattr__(self, "structural_type", canonical_type)
            object.__setattr__(self, "structural_type_input", supplied_type)
        structural_errors.extend(structural_binding_validation_errors(self))
        structural_errors = tuple(dict.fromkeys(structural_errors))
        object.__setattr__(self, "structural_validation_errors", structural_errors)

    @property
    def structural_binding_valid(self) -> bool:
        return not self.structural_validation_errors

    @property
    def all_provenance(self) -> tuple[FieldProvenance, ...]:
        structural = tuple(item for item in (
            self.section_provenance,
            self.governing_context_provenance,
            self.title_provenance,
            self.row_header_provenance,
            self.column_header_provenance,
            self.category_provenance,
            self.period_provenance,
            self.population_provenance,
        ) if item is not None)
        return (
            structural
            + self.semantic_provenance
            + tuple(item.provenance for item in self.relation_arguments)
        )

    def provenance_for(self, field: str) -> Optional[FieldProvenance]:
        return next((item for item in self.all_provenance if item.field == field), None)

    def source_container_text(self, provenance: FieldProvenance) -> str:
        """Resolve a provenance container without consulting request state."""
        if provenance.source_container == "local_proposition":
            return self.local_proposition
        if provenance.source_container == "source_candidate":
            return self.source_candidate_text
        structural = {
            "section_title": self.section_title,
            "governing_context": self.governing_context,
            "table_chart_title": self.table_chart_title,
            "row_header": self.row_header,
            "column_header": self.column_header,
            "category": self.category,
            "period": self.period,
            "population": self.population,
        }
        if provenance.source_container.startswith("structural_labels."):
            return structural.get(
                provenance.source_container.rsplit(".", 1)[-1], ""
            )
        if provenance.source_container.startswith("metadata."):
            metadata_key = provenance.source_container.rsplit(".", 1)[-1]
            aliases = {
                "section": "section_title",
                "heading": "section_title",
                "table_title": "table_chart_title",
                "chart_title": "table_chart_title",
                "figure_title": "table_chart_title",
                "title": "table_chart_title",
                "table_row_header": "row_header",
                "table_column_header": "column_header",
                "value_header": "column_header",
                "facet_label": "category",
                "series_label": "category",
                "year": "period",
                "column_period": "period",
                "scope_population": "population",
                "sample": "population",
            }
            return structural.get(aliases.get(metadata_key, metadata_key), "")
        return ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "evidence_unit_id": self.evidence_unit_id,
            "candidate_id": self.candidate_id,
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "page": self.page,
            "structural_type": self.structural_type,
            "structural_type_authority": (
                "structural_evidence_unit"
                if self.structural_evidence_unit is not None
                else "canonical_source_representation"
            ),
            "structural_type_transition": {
                "input": self.structural_type_input or self.structural_type,
                "canonical": self.structural_type,
                "status": (
                    "rejected_mismatch"
                    if "structural_type_transition_mismatch"
                    in self.structural_validation_errors
                    else "canonical"
                ),
            },
            "local_proposition": self.local_proposition,
            "proposition_span": list(self.proposition_span),
            "proposition_sha256": self.proposition_sha256,
            "section_title": self.section_title,
            "governing_context": self.governing_context,
            "table_chart_title": self.table_chart_title,
            "row_header": self.row_header,
            "column_header": self.column_header,
            "category": self.category,
            "period": self.period,
            "population": self.population,
            "evidence_subject": self.evidence_subject,
            "evidence_predicate": self.evidence_predicate,
            "evidence_target_entity": self.evidence_target_entity,
            "evidence_unit": self.evidence_unit,
            "evidence_numeric_role": self.evidence_numeric_role,
            "evidence_polarity": self.evidence_polarity,
            "relation_arguments": [
                item.as_dict() for item in self.relation_arguments
            ],
            "field_provenance": [item.as_dict() for item in self.all_provenance],
            "model_assisted": self.model_assisted,
            "structural_evidence_unit": (
                self.structural_evidence_unit.safe_summary()
                if self.structural_evidence_unit else None
            ),
            "structural_binding_valid": self.structural_binding_valid,
            "structural_validation_errors": list(
                self.structural_validation_errors
            ),
            "relation_validation_errors": list(
                relation_argument_validation_errors(self)
            ),
        }


@dataclass(frozen=True)
class NumericBindingProvenance:
    """Compact source-local proof connecting one scalar to its semantic role."""

    value_span: tuple[int, int]
    binding_method: str
    source_span_sha256: str
    semantic_label_span: Optional[tuple[int, int]] = None
    metric_subject_span: Optional[tuple[int, int]] = None
    period_span: Optional[tuple[int, int]] = None
    facet_category_span: Optional[tuple[int, int]] = None
    scalar_role_span: Optional[tuple[int, int]] = None
    structural_role: str = ""
    scalar_role: str = ""
    quantity_kind: str = ""
    statistical_role: str = ""
    aggregation_role: str = ""
    statistical_role_cue_span: Optional[tuple[int, int]] = None
    aggregation_role_cue_span: Optional[tuple[int, int]] = None

    def __post_init__(self) -> None:
        for span in (
            self.value_span,
            self.semantic_label_span,
            self.metric_subject_span,
            self.period_span,
            self.facet_category_span,
            self.scalar_role_span,
            self.statistical_role_cue_span,
            self.aggregation_role_cue_span,
        ):
            if span is not None and (span[0] < 0 or span[1] <= span[0]):
                raise ValueError("invalid numeric binding source span")
        if not self.binding_method or not self.source_span_sha256:
            raise ValueError("numeric binding provenance is incomplete")

    def as_dict(self) -> dict[str, Any]:
        return {
            "value_span": list(self.value_span),
            "semantic_label_span": (
                list(self.semantic_label_span)
                if self.semantic_label_span is not None else None
            ),
            "metric_subject_span": (
                list(self.metric_subject_span)
                if self.metric_subject_span is not None else None
            ),
            "period_span": (
                list(self.period_span) if self.period_span is not None else None
            ),
            "facet_category_span": (
                list(self.facet_category_span)
                if self.facet_category_span is not None else None
            ),
            "scalar_role_span": (
                list(self.scalar_role_span)
                if self.scalar_role_span is not None else None
            ),
            "structural_role": self.structural_role,
            "scalar_role": self.scalar_role,
            "quantity_kind": self.quantity_kind,
            "statistical_role": self.statistical_role,
            "aggregation_role": self.aggregation_role,
            "statistical_role_cue_span": (
                list(self.statistical_role_cue_span)
                if self.statistical_role_cue_span is not None else None
            ),
            "aggregation_role_cue_span": (
                list(self.aggregation_role_cue_span)
                if self.aggregation_role_cue_span is not None else None
            ),
            "binding_method": self.binding_method,
            "source_span_sha256": self.source_span_sha256,
        }


@dataclass(frozen=True)
class EvidenceAtom:
    """A fact-bearing child span of one bounded evidence context.

    Atoms never contain request slots.  Every role is copied from the source
    context (or from a literal numeric token within it), so an unmet request
    cannot be made to look supported by filling a fact field from the request.
    """

    atom_id: str
    evidence_unit_id: str
    candidate_id: str
    source_id: str
    chunk_id: str
    page: Optional[int]
    structural_type: str
    source_span: tuple[int, int]
    text: str
    text_sha256: str
    atom_kind: str
    subject: str = ""
    predicate: str = ""
    target_entity: str = ""
    value: str = ""
    unit: str = ""
    period: str = ""
    population: str = ""
    facet: str = ""
    scalar_role: str = ""
    quantity_kind: str = ""
    statistical_role: str = ""
    aggregation_role: str = ""
    polarity: str = "neutral"
    relation_arguments: tuple[RelationArgument, ...] = ()
    source_evidence_unit_ids: tuple[str, ...] = ()
    numeric_binding: Optional[NumericBindingProvenance] = None

    def __post_init__(self) -> None:
        start, end = self.source_span
        if start < 0 or end <= start:
            raise ValueError("invalid EvidenceAtom source span")
        if self.text_sha256 != hashlib.sha256(self.text.encode("utf-8")).hexdigest():
            raise ValueError("EvidenceAtom text hash mismatch")
        if self.value and self.value.casefold() not in self.text.casefold():
            raise ValueError("EvidenceAtom value is not literal source text")
        if not self.source_evidence_unit_ids:
            object.__setattr__(
                self, "source_evidence_unit_ids", (self.evidence_unit_id,)
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "atom_id": self.atom_id,
            "evidence_unit_id": self.evidence_unit_id,
            "candidate_id": self.candidate_id,
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "page": self.page,
            "structural_type": self.structural_type,
            "source_span": list(self.source_span),
            "text_sha256": self.text_sha256,
            "atom_kind": self.atom_kind,
            "subject": self.subject,
            "predicate": self.predicate,
            "target_entity": self.target_entity,
            "value": self.value,
            "unit": self.unit,
            "period": self.period,
            "population": self.population,
            "facet": self.facet,
            "scalar_role": self.scalar_role,
            "quantity_kind": self.quantity_kind,
            "statistical_role": self.statistical_role,
            "aggregation_role": self.aggregation_role,
            "polarity": self.polarity,
            "relation_arguments": [item.as_dict() for item in self.relation_arguments],
            "source_evidence_unit_ids": list(self.source_evidence_unit_ids),
            "numeric_binding": (
                self.numeric_binding.as_dict()
                if self.numeric_binding is not None else None
            ),
        }


def evidence_atoms_for_context(
    context: EvidenceContext,
    *,
    value: str = "",
    unit: str = "",
    numeric_role: str = "",
    scalar_role: str = "",
    quantity_kind: str = "",
    statistical_role: str = "",
    aggregation_role: str = "",
    atom_kind: str = "source_atomic",
    numeric_binding: Optional[NumericBindingProvenance] = None,
) -> tuple[EvidenceAtom, ...]:
    """Atomize a complete source proposition into local, citeable spans.

    ``atomic_fact_spans`` keeps multi-value lists separated.  A scalar caller
    may pass its observed literal value to select only the atom containing it.
    The function deliberately has no requirement argument.
    """
    spans = atomic_fact_spans(context.local_proposition)
    if not spans:
        return ()
    output: list[EvidenceAtom] = []
    literal_value = str(value or "").strip()
    for index, span in enumerate(spans, start=1):
        # A scalar must match a complete literal. Substring selection can
        # mistake an endpoint such as 24 for a different atom's year 2024.
        if literal_value and not re.search(
            rf"(?<![\w.,]){re.escape(literal_value)}(?![\w.,])", span.text, re.I
        ):
            continue
        absolute_start = context.proposition_span[0] + span.start
        absolute_end = context.proposition_span[0] + span.end
        payload = "\x1f".join((
            context.evidence_unit_id,
            str(absolute_start), str(absolute_end), span.text,
            literal_value, str(unit or ""), str(numeric_role or ""),
            str(scalar_role or ""),
            str(quantity_kind or ""),
            str(statistical_role or ""),
            str(aggregation_role or ""),
        ))
        output.append(EvidenceAtom(
            atom_id="atom-" + hashlib.sha256(
                payload.encode("utf-8")
            ).hexdigest()[:24],
            evidence_unit_id=context.evidence_unit_id,
            candidate_id=context.candidate_id,
            source_id=context.source_id,
            chunk_id=context.chunk_id,
            page=context.page,
            structural_type=context.structural_type,
            source_span=(absolute_start, absolute_end),
            text=span.text,
            text_sha256=hashlib.sha256(span.text.encode("utf-8")).hexdigest(),
            atom_kind=(
                "numeric" if literal_value else
                "multi_value" if span.statistic_count else atom_kind
            ),
            subject=context.evidence_subject,
            predicate=context.evidence_predicate,
            target_entity=context.evidence_target_entity,
            value=literal_value,
            unit=str(unit or context.evidence_unit or ""),
            period=context.period,
            population=context.population,
            facet=context.category or context.row_header,
            scalar_role=str(scalar_role or ""),
            quantity_kind=str(quantity_kind or ""),
            statistical_role=str(statistical_role or ""),
            aggregation_role=str(aggregation_role or ""),
            polarity=context.evidence_polarity,
            relation_arguments=context.relation_arguments,
            numeric_binding=numeric_binding,
        ))
    return tuple(output)


def relation_argument_validation_errors(
    context: EvidenceContext,
    *,
    predicate: str = "",
) -> tuple[str, ...]:
    """Validate mandatory relation roles and their exact source provenance."""
    relation = str(predicate or context.evidence_predicate or "")
    required_roles = RELATION_ARGUMENT_SCHEMAS.get(relation, ())
    if not required_roles:
        return ()
    arguments = {item.role: item for item in context.relation_arguments}
    reasons: list[str] = []
    for role in required_roles:
        argument = arguments.get(role)
        if argument is None or not str(argument.value or "").strip():
            reasons.append(f"relation_argument_missing:{role}")
            continue
        provenance = argument.provenance
        container = context.source_container_text(provenance)
        start, end = provenance.source_span
        if not container or start < 0 or end <= start or end > len(container):
            reasons.append(f"relation_argument_provenance_invalid:{role}")
            continue
        source_quote = container[start:end]
        if (
            provenance.source_quote != source_quote
            or provenance.source_sha256
            != hashlib.sha256(source_quote.encode("utf-8")).hexdigest()
        ):
            reasons.append(f"relation_argument_provenance_invalid:{role}")
            continue
        normalized_value = " ".join(argument.value.casefold().split())
        normalized_quote = " ".join(source_quote.casefold().split())
        if normalized_value not in normalized_quote:
            reasons.append(f"relation_argument_not_literal:{role}")
    return tuple(dict.fromkeys(reasons))


def structural_binding_validation_errors(
    context: EvidenceContext,
) -> tuple[str, ...]:
    """Validate source grouping independently of literal/semantic validity."""
    unit = context.structural_evidence_unit
    if unit is None:
        return ("structural_parent_missing",)
    reasons: list[str] = list(unit.validation_errors)
    if context.governing_context or context.governing_context_provenance is not None:
        provenance = context.governing_context_provenance
        valid = provenance is not None and provenance.source_container == "source_candidate"
        if valid:
            start, end = provenance.source_span
            quote = context.source_candidate_text[start:end]
            valid = bool(
                0 <= start < end <= len(context.source_candidate_text)
                and provenance.value == context.governing_context
                and provenance.source_quote == quote == context.governing_context
                and provenance.source_sha256 == hashlib.sha256(quote.encode("utf-8")).hexdigest()
                and provenance.structural_unit_id == unit.unit_id
                and provenance.structural_group_id == (unit.structural_group_id or unit.unit_id)
                and unit.source_spans
                and end <= min(span.start for span in unit.source_spans)
            )
        if not valid:
            reasons.append("governing_context_provenance_invalid")
    if context.structural_type != unit.structural_type.value:
        reasons.append("structural_type_transition_mismatch")
    if (
        context.evidence_unit_id != unit.unit_id
        or context.candidate_id != unit.candidate_id
        or context.source_id != unit.source_id
        or context.chunk_id != unit.chunk_id
    ):
        reasons.append("structural_parent_missing")
    proposition = context.local_proposition
    unit_text = unit.local_text
    local_position = unit_text.find(proposition)
    if local_position < 0:
        reasons.append("proposition_boundary_unresolved")
    start, end = context.proposition_span
    if (
        local_position >= 0
        and not (
            0 <= start <= end <= len(unit_text)
            and unit_text[start:end] == proposition
        )
    ):
        reasons.append("proposition_boundary_unresolved")

    material_fields = {
        "evidence_subject", "evidence_target_entity", "evidence_period",
        "evidence_population", "evidence_category",
    }
    material_provenance = tuple(
        item for item in context.all_provenance
        if item.field in material_fields or item.field.startswith("relation_")
    )
    expected_group = unit.structural_group_id or unit.unit_id
    mismatched = [
        item for item in material_provenance
        if (
            item.structural_unit_id != unit.unit_id
            or item.structural_group_id != expected_group
        )
    ]
    if mismatched:
        if unit.structural_type is StructuralEvidenceType.TABLE_ROW:
            reasons.append("cross_row_binding_rejected")
        elif unit.structural_type is StructuralEvidenceType.LIST_ITEM:
            reasons.append("cross_item_binding_rejected")
        elif unit.structural_type is StructuralEvidenceType.CHART_CATEGORY_RECORD:
            reasons.append("cross_category_binding_rejected")
        else:
            reasons.append("structural_argument_mismatch")
    if unit.structural_type is StructuralEvidenceType.TABLE_ROW:
        if not unit.row_id or not unit.row_label:
            reasons.append("table_row_boundary_unresolved")
        literal_roles = {
            item.role for item in unit.field_bindings
            if item.structural_group_id == expected_group
        }
        if context.evidence_subject and "entity" not in literal_roles:
            reasons.append("structural_value_unbound")
    elif unit.structural_type is StructuralEvidenceType.LIST_ITEM:
        if not unit.list_id or not unit.item_id:
            reasons.append("list_item_boundary_unresolved")
    elif unit.structural_type is StructuralEvidenceType.CHART_CATEGORY_RECORD:
        if not unit.category_id or not unit.row_label:
            reasons.append("chart_category_boundary_unresolved")
    return tuple(dict.fromkeys(reasons))


_ASSISTED_SEMANTIC_FIELDS = frozenset({
    "subject", "predicate", "target_entity", "unit", "numeric_role",
    "polarity", "period", "population", "category",
})
_ASSISTED_RELATION_ARGUMENT_FIELDS = frozenset(
    role for roles in RELATION_ARGUMENT_SCHEMAS.values() for role in roles
)
_ASSISTED_ALL_FIELDS = (
    _ASSISTED_SEMANTIC_FIELDS | _ASSISTED_RELATION_ARGUMENT_FIELDS
)
_ASSISTED_PREDICATES = frozenset({
    "caused_by", "changed_between", "compared_with", "defined_as",
    "employs_workers", "excluded_from", "explained_as", "has_attribute",
    "has_count", "has_members", "has_percentage", "included_in",
    "exception_to", "performed_by", "purpose_of", "raises_capital", "recruits_employees",
    "stated_as",
}) | frozenset(RELATION_ARGUMENT_SCHEMAS)
_ASSISTED_NUMERIC_ROLES = frozenset({
    "category_value", "count", "percentage", "requested_answer_value",
})
_ASSISTED_POLARITIES = frozenset({"negative", "neutral", "positive"})


def model_assisted_semantic_input(context: EvidenceContext) -> dict[str, Any]:
    """Return the complete and only payload allowed to reach the resolver."""
    existing_arguments = {item.role for item in context.relation_arguments}
    relation_roles = RELATION_ARGUMENT_SCHEMAS.get(
        context.evidence_predicate, ()
    )
    if relation_roles:
        requested_fields = [
            role for role in relation_roles if role not in existing_arguments
        ]
    else:
        requested_fields = [
            name for name, value in (
                ("subject", context.evidence_subject),
                ("predicate", context.evidence_predicate),
                ("target_entity", context.evidence_target_entity),
            )
            if not value
        ]
        if context.evidence_numeric_role:
            requested_fields.extend(
                name for name, value in (
                    ("unit", context.evidence_unit),
                    ("period", context.period),
                    ("population", context.population),
                )
                if not value
            )
    # The runtime still authorizes only requested fields plus the mandatory
    # roles for the predicate actually returned.  The model must nevertheless
    # see the finite role vocabulary up front so it can emit a complete
    # relational fact in one generation.
    initially_allowed_fields = tuple(dict.fromkeys(requested_fields))
    relation_role_fields = tuple(dict.fromkeys(
        role
        for roles in RELATION_ARGUMENT_SCHEMAS.values()
        for role in roles
    ))
    return {
        "local_proposition": context.local_proposition,
        "structural_labels": {
            "section_title": context.section_title,
            "table_chart_title": context.table_chart_title,
            "row_header": context.row_header,
            "column_header": context.column_header,
            "category": context.category,
            "period": context.period,
            "population": context.population,
        },
        "requested_fields": list(initially_allowed_fields),
        "allowed_fields": list(dict.fromkeys(
            (*initially_allowed_fields, *relation_role_fields)
        )),
        "allowed_predicates": sorted(_ASSISTED_PREDICATES),
        "relation_argument_schemas": {
            relation: list(roles)
            for relation, roles in RELATION_ARGUMENT_SCHEMAS.items()
        },
        "requested_field_instructions": {
            name: RELATION_ARGUMENT_ROLE_DESCRIPTIONS.get(
                name,
                "resolve this evidence-semantic field from the named source container",
            )
            for name in initially_allowed_fields
        },
        "allowed_numeric_roles": sorted(_ASSISTED_NUMERIC_ROLES),
        "allowed_polarities": sorted(_ASSISTED_POLARITIES),
        "field_object_required_keys": [
            "value", "source_container", "source_quote", "confidence",
        ],
        "response_schema": {
            "status": "resolved or uncertain",
            "fields": {
                "<allowed_field>": {
                    "value": "literal or allowed semantic label",
                    "source_container": "local_proposition or structural_labels.<name>",
                    "source_quote": "exact unique quote copied from source_container",
                    "confidence": "number from 0.65 through 1.0 when resolved",
                }
            },
            "uncertain": False,
            "relation_rule": (
                "When predicate resolves to a relation in relation_argument_schemas, "
                "return every mandatory argument role for that relation in the same "
                "fields object."
            ),
        },
    }


def apply_model_assisted_semantics(
    context: EvidenceContext,
    resolution: Mapping[str, Any],
) -> EvidenceContext:
    """Validate quote-based evidence-only semantics and compute spans locally."""
    if str(resolution.get("status") or "").casefold() == "uncertain" or bool(
        resolution.get("uncertain")
    ):
        raise EvidenceSemanticValidationError("resolver_uncertain")
    raw_fields = resolution.get("fields")
    if isinstance(raw_fields, list):
        normalized_fields: dict[str, Any] = {}
        for item in raw_fields:
            if not isinstance(item, Mapping) or not item.get("name"):
                raise EvidenceSemanticValidationError(
                    "resolver_field_not_structured"
                )
            normalized_fields[str(item["name"])] = item
        raw_fields = normalized_fields
    if not isinstance(raw_fields, Mapping) or not raw_fields:
        raise EvidenceSemanticValidationError("resolver_fields_missing")
    allowed_fields = set(
        model_assisted_semantic_input(context)["requested_fields"]
    )
    returned_predicate = raw_fields.get("predicate")
    if isinstance(returned_predicate, Mapping):
        allowed_fields.update(RELATION_ARGUMENT_SCHEMAS.get(
            str(returned_predicate.get("value") or ""), ()
        ))
    if not set(map(str, raw_fields)) <= allowed_fields:
        raise EvidenceSemanticValidationError("resolver_field_not_allowed")

    containers = {
        "local_proposition": context.local_proposition,
        "structural_labels.section_title": context.section_title,
        "structural_labels.table_chart_title": context.table_chart_title,
        "structural_labels.row_header": context.row_header,
        "structural_labels.column_header": context.column_header,
        "structural_labels.category": context.category,
        "structural_labels.period": context.period,
        "structural_labels.population": context.population,
    }
    existing = {
        "subject": context.evidence_subject,
        "predicate": context.evidence_predicate,
        "target_entity": context.evidence_target_entity,
        "unit": context.evidence_unit,
        "numeric_role": context.evidence_numeric_role,
        "polarity": context.evidence_polarity,
        "period": context.period,
        "population": context.population,
        "category": context.category,
    }
    accepted = dict(existing)
    records: list[FieldProvenance] = []
    relation_arguments = {
        item.role: item for item in context.relation_arguments
    }
    literal_fields = {
        "subject", "target_entity", "unit", "period", "population", "category",
    }
    accepted_change_count = 0
    for raw_name, raw_spec in raw_fields.items():
        name = str(raw_name)
        if not isinstance(raw_spec, Mapping):
            raise EvidenceSemanticValidationError("resolver_field_not_structured")
        value = str(raw_spec.get("value") or "").strip()
        container_name = str(raw_spec.get("source_container") or "")
        source_quote = str(
            raw_spec.get("source_quote")
            or raw_spec.get("exact_source_quote")
            or ""
        )
        confidence = float(raw_spec.get("confidence") or 0.0)
        source_text = containers.get(container_name)
        if not value or source_text is None or not source_text:
            raise EvidenceSemanticValidationError("resolver_source_container_invalid")
        if not source_quote:
            raise EvidenceSemanticValidationError("resolver_source_quote_missing")
        start = source_text.find(source_quote)
        if start < 0:
            raise EvidenceSemanticValidationError("resolver_source_quote_absent")
        if source_text.find(source_quote, start + 1) >= 0:
            raise EvidenceSemanticValidationError("resolver_source_quote_ambiguous")
        end = start + len(source_quote)
        source_slice = source_quote
        expected_hash = hashlib.sha256(source_slice.encode("utf-8")).hexdigest()
        if confidence < 0.65 or confidence > 1.0:
            raise EvidenceSemanticValidationError("resolver_confidence_insufficient")
        if name == "predicate" and value not in _ASSISTED_PREDICATES:
            raise EvidenceSemanticValidationError("resolver_predicate_invalid")
        if name == "numeric_role" and value not in _ASSISTED_NUMERIC_ROLES:
            raise EvidenceSemanticValidationError("resolver_numeric_role_invalid")
        if name == "polarity" and value not in _ASSISTED_POLARITIES:
            raise EvidenceSemanticValidationError("resolver_polarity_invalid")
        if name in literal_fields:
            normalized_value = " ".join(value.casefold().split())
            normalized_slice = " ".join(source_slice.casefold().split())
            if normalized_value not in normalized_slice:
                raise EvidenceSemanticValidationError(
                    "resolver_literal_absent_from_quote"
                )
        if name in _ASSISTED_RELATION_ARGUMENT_FIELDS:
            normalized_value = " ".join(value.casefold().split())
            normalized_slice = " ".join(source_slice.casefold().split())
            if normalized_value not in normalized_slice:
                raise EvidenceSemanticValidationError(
                    "resolver_relation_argument_absent_from_quote"
                )
        current = str(existing.get(name) or "")
        if current and current != "neutral" and current.casefold() != value.casefold():
            existing_provenance = (
                context.provenance_for(f"evidence_{name}")
                or context.provenance_for(name)
            )
            if (
                existing_provenance is None
                or existing_provenance.revisability
                != "deterministic_revisable"
                or existing_provenance.confidence >= 0.95
            ):
                raise EvidenceSemanticValidationError("resolver_override_forbidden")
        provenance = FieldProvenance(
            field=f"evidence_{name}",
            value=value,
            provenance_type="bounded_model_exact_quote",
            source_container=container_name,
            source_span=(start, end),
            source_sha256=expected_hash,
            confidence=confidence,
            revisability="model_revisable",
            source_quote=source_quote,
            derivation_type=(
                "bounded_model_semantic_label"
                if name in {"predicate", "numeric_role", "polarity"}
                else "bounded_model_literal"
            ),
            structural_unit_id=(
                context.structural_evidence_unit.unit_id
                if context.structural_evidence_unit else ""
            ),
            structural_group_id=(
                context.structural_evidence_unit.structural_group_id
                if context.structural_evidence_unit else ""
            ),
            structural_role=name,
        )
        if name in _ASSISTED_RELATION_ARGUMENT_FIELDS:
            prior = relation_arguments.get(name)
            if prior is not None and prior.value.casefold() != value.casefold() and (
                prior.provenance.revisability != "deterministic_revisable"
                or prior.provenance.confidence >= 0.95
            ):
                raise EvidenceSemanticValidationError("resolver_override_forbidden")
            relation_arguments[name] = RelationArgument(
                role=name,
                value=value,
                provenance=provenance,
            )
            accepted_change_count += 1
        else:
            accepted[name] = value
            records.append(provenance)
            accepted_change_count += 1

    if not accepted_change_count:
        raise EvidenceSemanticValidationError("resolver_no_usable_fields")
    replaced_fields = {item.field for item in records}
    semantic_provenance = tuple(
        item for item in context.semantic_provenance
        if item.field not in replaced_fields
    ) + tuple(records)
    return replace(
        context,
        category=accepted["category"],
        period=accepted["period"],
        population=accepted["population"],
        evidence_subject=accepted["subject"],
        evidence_predicate=accepted["predicate"],
        evidence_target_entity=accepted["target_entity"],
        evidence_unit=accepted["unit"],
        evidence_numeric_role=accepted["numeric_role"],
        evidence_polarity=accepted["polarity"],
        semantic_provenance=semantic_provenance,
        relation_arguments=tuple(relation_arguments.values()),
        model_assisted=True,
    )


def _field_provenance(
    field: str,
    value: Any,
    provenance_type: str,
    source_container: str,
    source_text: Any,
    *,
    confidence: float,
    revisability: str = "revisable",
    source_span: Optional[tuple[int, int]] = None,
    value_must_be_literal: bool = True,
    derivation_type: str = "literal",
    structural_unit_id: str = "",
    structural_group_id: str = "",
    structural_role: str = "",
) -> Optional[FieldProvenance]:
    normalized_value = str(value or "").strip()
    text = str(source_text or "")
    if not normalized_value or not text:
        return None
    span = source_span
    if span is None:
        if value_must_be_literal:
            position = text.casefold().find(normalized_value.casefold())
            if position < 0:
                return None
            span = (position, position + len(normalized_value))
        else:
            span = (0, len(text))
    start, end = span
    if start < 0 or end > len(text) or end <= start:
        return None
    source_quote = text[start:end]
    if value_must_be_literal and (
        " ".join(normalized_value.casefold().split())
        not in " ".join(source_quote.casefold().split())
    ):
        return None
    return FieldProvenance(
        field=field,
        value=normalized_value,
        provenance_type=provenance_type,
        source_container=source_container,
        source_span=(start, end),
        source_sha256=hashlib.sha256(source_quote.encode("utf-8")).hexdigest(),
        confidence=max(0.0, min(1.0, float(confidence))),
        revisability=str(revisability or "revisable"),
        source_quote=source_quote,
        derivation_type=str(derivation_type or "literal"),
        structural_unit_id=str(structural_unit_id or ""),
        structural_group_id=str(structural_group_id or ""),
        structural_role=str(structural_role or field),
    )


def build_evidence_context(
    *,
    evidence_unit_id: str,
    candidate_id: str,
    source_id: str,
    chunk_id: str,
    page: Optional[int],
    structural_type: str,
    local_proposition: str,
    proposition_span: tuple[int, int],
    metadata: Optional[Mapping[str, Any]] = None,
    evidence_semantics: Optional[Mapping[str, Any]] = None,
    semantic_sources: Optional[Mapping[str, Mapping[str, Any]]] = None,
    relation_arguments: Optional[Mapping[str, Any]] = None,
    model_assisted: bool = False,
    structural_evidence_unit: Optional[StructuralEvidenceUnit] = None,
) -> EvidenceContext:
    """Construct one context solely from evidence and bounded labels."""
    metadata = dict(metadata or {})
    semantics = dict(evidence_semantics or {})
    sources = dict(semantic_sources or {})
    raw_relation_arguments = dict(relation_arguments or {})
    proposition = str(local_proposition or "")
    if structural_evidence_unit is None:
        structural_evidence_unit = atomic_structural_evidence_unit(
            proposition,
            source_id=str(source_id),
            candidate_id=str(candidate_id),
            chunk_id=str(chunk_id),
            page=page,
            unit_id=str(evidence_unit_id),
        )
    canonical_structural_type = structural_evidence_unit.structural_type.value
    supplied_structural_type = str(structural_type or "")
    transition_errors: tuple[str, ...] = ()
    if supplied_structural_type.casefold() not in {"", "body", "paragraph"} and (
        supplied_structural_type != canonical_structural_type
    ):
        transition_errors = ("structural_type_transition_mismatch",)
    structural_unit_id = structural_evidence_unit.unit_id
    structural_group_id = (
        structural_evidence_unit.structural_group_id or structural_unit_id
    )

    def metadata_value(*keys: str) -> tuple[str, str]:
        for key in keys:
            value = str(metadata.get(key) or "").strip()
            if value:
                return value, key
        return "", ""

    def structural(field: str, *keys: str, provenance_type: str) -> tuple[str, Optional[FieldProvenance]]:
        value, key = metadata_value(*keys)
        provenance = _field_provenance(
            field,
            value,
            provenance_type,
            f"metadata.{key}" if key else "",
            value,
            confidence=0.96,
            revisability="fixed_source_structure",
            source_span=(0, len(value)) if value else None,
            structural_unit_id=structural_unit_id,
            structural_group_id=structural_group_id,
            structural_role=field,
        )
        return (value if provenance else ""), provenance

    section, section_provenance = structural(
        "section_title", "section_title", "section", "heading",
        provenance_type="governing_section",
    )
    title, title_provenance = structural(
        "table_chart_title", "table_title", "chart_title", "figure_title", "title",
        provenance_type="table_chart_title",
    )
    row, row_provenance = structural(
        "row_header", "row_header", "table_row_header",
        provenance_type="row_header",
    )
    column, column_provenance = structural(
        "column_header", "column_header", "table_column_header", "value_header",
        provenance_type="column_header",
    )
    category, category_provenance = structural(
        "category", "category", "facet_label", "series_label",
        provenance_type="row_or_chart_category",
    )
    period, period_provenance = structural(
        "period", "period", "year", "column_period",
        provenance_type="column_or_structural_period",
    )
    population, population_provenance = structural(
        "population", "population", "scope_population", "sample",
        provenance_type="governing_population",
    )

    semantic_records: list[FieldProvenance] = []
    accepted_semantics: dict[str, str] = {}
    for field_name in (
        "subject", "predicate", "target_entity", "unit", "numeric_role", "polarity",
        "period", "population", "category",
    ):
        value = str(semantics.get(field_name) or "").strip()
        if not value:
            continue
        source = dict(sources.get(field_name) or {})
        source_text = str(source.get("source_text") or proposition)
        record = _field_provenance(
            f"evidence_{field_name}",
            value,
            str(source.get("provenance_type") or "same_proposition"),
            str(source.get("source_container") or "local_proposition"),
            source_text,
            confidence=float(source.get("confidence") or 0.9),
            revisability=str(
                source.get("revisability") or "deterministic_revisable"
            ),
            source_span=(
                tuple(source["source_span"])
                if source.get("source_span") is not None else None
            ),
            value_must_be_literal=field_name not in {
                "predicate", "unit", "numeric_role", "polarity"
            },
            derivation_type=(
                str(source.get("derivation_type") or "deterministic_semantic_label")
                if field_name in {"predicate", "unit", "numeric_role", "polarity"}
                else "literal"
            ),
            structural_unit_id=str(
                source.get("structural_unit_id") or structural_unit_id
            ),
            structural_group_id=str(
                source.get("structural_group_id") or structural_group_id
            ),
            structural_role=str(
                source.get("structural_role") or field_name
            ),
        )
        if record is None:
            continue
        semantic_records.append(record)
        accepted_semantics[field_name] = value

    def semantic_record(field_name: str) -> Optional[FieldProvenance]:
        return next((
            item for item in semantic_records
            if item.field == f"evidence_{field_name}"
        ), None)

    accepted_arguments: list[RelationArgument] = []
    for role, raw_value in raw_relation_arguments.items():
        value = str(raw_value or "").strip()
        source = dict(sources.get(str(role)) or {})
        source_text = str(source.get("source_text") or proposition)
        record = _field_provenance(
            f"relation_{role}",
            value,
            str(source.get("provenance_type") or "same_proposition_argument"),
            str(source.get("source_container") or "local_proposition"),
            source_text,
            confidence=float(source.get("confidence") or 0.9),
            revisability=str(
                source.get("revisability") or "deterministic_revisable"
            ),
            source_span=(
                tuple(source["source_span"])
                if source.get("source_span") is not None else None
            ),
            value_must_be_literal=True,
            derivation_type="deterministic_literal_argument",
            structural_unit_id=str(
                source.get("structural_unit_id") or structural_unit_id
            ),
            structural_group_id=str(
                source.get("structural_group_id") or structural_group_id
            ),
            structural_role=str(source.get("structural_role") or role),
        )
        if record is not None:
            accepted_arguments.append(RelationArgument(
                role=str(role), value=value, provenance=record
            ))

    return EvidenceContext(
        evidence_unit_id=str(evidence_unit_id),
        candidate_id=str(candidate_id),
        source_id=str(source_id),
        chunk_id=str(chunk_id),
        page=page,
        structural_type=canonical_structural_type,
        local_proposition=proposition,
        proposition_span=tuple(proposition_span),
        proposition_sha256=hashlib.sha256(proposition.encode("utf-8")).hexdigest(),
        section_title=section,
        section_provenance=section_provenance,
        table_chart_title=title,
        title_provenance=title_provenance,
        row_header=row,
        row_header_provenance=row_provenance,
        column_header=column,
        column_header_provenance=column_provenance,
        category=category or row or accepted_semantics.get("category", ""),
        category_provenance=(
            category_provenance or row_provenance or semantic_record("category")
        ),
        period=period or accepted_semantics.get("period", ""),
        period_provenance=period_provenance or semantic_record("period"),
        population=population or accepted_semantics.get("population", ""),
        population_provenance=population_provenance or semantic_record("population"),
        evidence_subject=accepted_semantics.get("subject", ""),
        evidence_predicate=accepted_semantics.get("predicate", ""),
        evidence_target_entity=accepted_semantics.get("target_entity", ""),
        evidence_unit=accepted_semantics.get("unit", ""),
        evidence_numeric_role=accepted_semantics.get("numeric_role", ""),
        evidence_polarity=accepted_semantics.get("polarity", "neutral"),
        semantic_provenance=tuple(semantic_records),
        relation_arguments=tuple(accepted_arguments),
        model_assisted=bool(model_assisted),
        structural_evidence_unit=structural_evidence_unit,
        structural_type_input=supplied_structural_type,
        structural_validation_errors=transition_errors,
    )


@dataclass(frozen=True)
class StructuralSpan:
    start: int
    end: int
    text: str
    kind: str
    parent_index: Optional[int] = None


@dataclass(frozen=True)
class ContextDependency:
    reason_codes: tuple[str, ...]
    explicit_subject: bool
    complete_predicate: bool
    percentage_population: bool
    comparison_complete: bool

    @property
    def complete(self) -> bool:
        return (
            not self.reason_codes
            and self.explicit_subject
            and self.complete_predicate
            and self.percentage_population
            and self.comparison_complete
        )


@dataclass(frozen=True)
class ContextEvidenceSpan:
    focus_text: str
    evidence_text: str
    focus_start: int
    focus_end: int
    evidence_start: int
    evidence_end: int
    dependency_codes: tuple[str, ...]
    expansion_type: str
    sentence_count: int
    complete: bool


def _trimmed_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _looks_heading(value: str) -> bool:
    text = value.strip()
    words = text.split()
    if _HEADING_MARKDOWN_RE.match(text):
        return True
    return bool(
        1 <= len(words) <= 10
        and len(text) <= 120
        and not text.endswith((".", "?", "!", "׃"))
        and not _PREDICATE_RE.search(text)
        and not _PERCENT_RE.search(text)
        and len(_LETTER_RE.findall(text)) >= 4
    )


def structural_spans(value: str) -> tuple[StructuralSpan, ...]:
    """Parse headings, sentences, bullets, and table rows with source offsets."""
    text = str(value or "")
    content_start = 0
    breadcrumb = _BREADCRUMB_RE.match(text)
    if breadcrumb:
        content_start = breadcrumb.end()
    blocks: list[tuple[int, int]] = []
    cursor = content_start
    for match in re.finditer(r"\n[ \t]*\n+", text[content_start:]):
        boundary_start = content_start + match.start()
        boundary_end = content_start + match.end()
        start, end = _trimmed_bounds(text, cursor, boundary_start)
        if start < end:
            blocks.append((start, end))
        cursor = boundary_end
    start, end = _trimmed_bounds(text, cursor, len(text))
    if start < end:
        blocks.append((start, end))

    spans: list[StructuralSpan] = []
    nearest_heading: Optional[int] = None
    nearest_table_header: Optional[int] = None
    for block_start, block_end in blocks:
        block = text[block_start:block_end]
        lines = list(re.finditer(r"[^\n]+", block))
        if lines and all(
            _BULLET_RE.match(line.group(0)) or not line.group(0).strip()
            for line in lines
        ):
            for line in lines:
                line_start = block_start + line.start()
                line_end = block_start + line.end()
                clean_start, clean_end = _trimmed_bounds(text, line_start, line_end)
                if clean_start < clean_end:
                    spans.append(StructuralSpan(
                        clean_start,
                        clean_end,
                        text[clean_start:clean_end],
                        "bullet",
                        nearest_heading,
                    ))
            continue
        if lines and any(_TABLE_RE.match(line.group(0)) for line in lines):
            for line_index, line in enumerate(lines):
                raw_line = line.group(0)
                if _TABLE_SEPARATOR_RE.match(raw_line):
                    continue
                line_start = block_start + line.start()
                line_end = block_start + line.end()
                kind = "table_header" if line_index == 0 else "table_row"
                parent = None if kind == "table_header" else nearest_table_header
                spans.append(StructuralSpan(
                    line_start,
                    line_end,
                    text[line_start:line_end],
                    kind,
                    parent,
                ))
                if kind == "table_header":
                    nearest_table_header = len(spans) - 1
            continue
        if _looks_heading(block):
            spans.append(StructuralSpan(
                block_start,
                block_end,
                block,
                "heading",
                nearest_heading,
            ))
            nearest_heading = len(spans) - 1
            continue

        sentence_start = block_start
        for match in _SENTENCE_BREAK_RE.finditer(text, block_start, block_end):
            sentence_end = match.start()
            clean_start, clean_end = _trimmed_bounds(
                text,
                sentence_start,
                sentence_end,
            )
            if clean_start < clean_end:
                spans.append(StructuralSpan(
                    clean_start,
                    clean_end,
                    text[clean_start:clean_end],
                    "sentence",
                    nearest_heading,
                ))
            sentence_start = match.end()
        clean_start, clean_end = _trimmed_bounds(text, sentence_start, block_end)
        if clean_start < clean_end:
            spans.append(StructuralSpan(
                clean_start,
                clean_end,
                text[clean_start:clean_end],
                "sentence",
                nearest_heading,
            ))
    return tuple(spans)


def source_atomic_units(value: str) -> tuple[AtomicFactSpan, ...]:
    output: list[AtomicFactSpan] = []
    for span in structural_spans(value):
        if span.kind in {"heading", "table_header"}:
            continue
        output.extend(atomic_fact_spans(span.text, absolute_start=span.start))
    return tuple(output)


def analyze_context_dependency(value: str) -> ContextDependency:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    proposition = assess_proposition(text)
    reasons: list[str] = []
    unresolved = bool(_UNRESOLVED_RE.search(text))
    if unresolved:
        reasons.append("unresolved_reference")
    if _DISCOURSE_START_RE.search(text):
        reasons.append("discourse_dependency")
    if not proposition.sentence_start:
        reasons.append("mid_clause_start")
    if not balanced_delimiters(text):
        reasons.append("unbalanced_delimiters")
    explicit_subject = bool(
        len(_LETTER_RE.findall(text)) >= 4
        and not re.match(
            r"^\s*(?:הן|הם|אלה|אלו|זו|זה|they|these|those|this|that)\b",
            text,
            re.IGNORECASE,
        )
    )
    if not explicit_subject:
        reasons.append("missing_explicit_subject")
    complete_predicate = bool(
        _PREDICATE_RE.search(text)
        or (_PERCENT_RE.search(text) and len(text.split()) >= 2)
    )
    if not complete_predicate:
        reasons.append("incomplete_predicate")
    percentage_population = not _PERCENT_RE.search(text) or bool(_POPULATION_RE.search(text))
    if not percentage_population:
        reasons.append("percentage_population_missing")
    years = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", text))
    non_year_numbers = [
        match.group(0)
        for match in _NUMBER_RE.finditer(text)
        if match.group(0) not in years
    ]
    percentages = re.findall(
        r"\d+(?:[.,]\d+)?\s*(?:%|אחוז(?:ים)?|\bpercent(?:age)?\b)",
        text,
        re.IGNORECASE,
    )
    comparison_complete = (
        not _COMPARISON_RE.search(text)
        or len(non_year_numbers) >= 2
        or (len(years) >= 2 and len(percentages) >= 2)
        or bool(_DELTA_RE.search(text))
        or bool(_PRIMARY_RATE_THEN_QUALITATIVE_COMPARISON_RE.search(text))
    )
    if not comparison_complete:
        reasons.append("comparison_endpoint_missing")
    return ContextDependency(
        reason_codes=tuple(dict.fromkeys(reasons)),
        explicit_subject=explicit_subject,
        complete_predicate=complete_predicate,
        percentage_population=percentage_population,
        comparison_complete=comparison_complete,
    )


def _specific_population_context(value: str) -> bool:
    text = re.sub(r"\s+", " ", value or "").strip()
    if not _POPULATION_RE.search(text):
        return False
    meaningful = [
        token for token in re.findall(r"[A-Za-z\u05d0-\u05ea]{2,}", text)
        if token.casefold() not in {
            "the", "companies", "workers", "employees", "group",
            "החברות", "חברות", "העובדים", "עובדים", "קבוצה",
        }
    ]
    return len(meaningful) >= 2


def _resolved_by_context(
    focus: str,
    context_before: str,
    dependency: ContextDependency,
) -> bool:
    if not dependency.reason_codes:
        return True
    before = re.sub(r"\s+", " ", context_before or "").strip()
    if not before:
        return False
    for reason in dependency.reason_codes:
        if reason in {"unresolved_reference", "percentage_population_missing"}:
            if not _specific_population_context(before):
                return False
        elif reason == "discourse_dependency":
            if len(_LETTER_RE.findall(before)) < 4:
                return False
        elif reason == "missing_explicit_subject":
            if not _specific_population_context(before):
                return False
        elif reason == "incomplete_predicate":
            if not (_PREDICATE_RE.search(focus) or _PERCENT_RE.search(focus)):
                return False
        elif reason == "comparison_endpoint_missing":
            if len(_NUMBER_RE.findall(f"{before} {focus}")) < 2:
                return False
        elif reason == "mid_clause_start":
            if len(_LETTER_RE.findall(before)) < 4:
                return False
        elif reason == "unbalanced_delimiters":
            if not balanced_delimiters(f"{before} {focus}"):
                return False
    return True


def _bounded(document: str, start: int, end: int) -> bool:
    value = document[start:end]
    return (
        0 < len(value) <= MAX_CONTEXT_CHARS
        and len(value.split()) <= MAX_CONTEXT_WORDS
    )


def construct_context_span(
    document: str,
    focus_start: int,
    focus_end: int,
) -> ContextEvidenceSpan:
    """Build the smallest bounded contiguous span resolving focus dependencies."""
    text = str(document or "")
    start, end = _trimmed_bounds(text, focus_start, focus_end)
    focus = text[start:end]
    dependency = analyze_context_dependency(focus)
    structures = structural_spans(text)
    containing_index = next(
        (
            index for index, span in enumerate(structures)
            if span.start <= start < span.end
        ),
        None,
    )
    containing = (
        structures[containing_index]
        if containing_index is not None
        else None
    )
    if (
        dependency.complete
        and containing is not None
        and containing.kind != "table_row"
        and _bounded(text, start, end)
    ):
        return ContextEvidenceSpan(
            focus, focus, start, end, start, end,
            dependency.reason_codes, "none", 1, True,
        )
    if containing_index is None:
        return ContextEvidenceSpan(
            focus, focus, start, end, start, end,
            dependency.reason_codes, "context_structure_missing", 1, False,
        )
    containing = structures[containing_index]
    candidates: list[tuple[int, int, str, int]] = []
    if containing.start < start:
        candidates.append((
            containing.start,
            end,
            "within_sentence_prefix",
            1,
        ))
    if containing.parent_index is not None:
        parent = structures[containing.parent_index]
        expansion = {
            "bullet": "bullet_parent_heading",
            "table_row": "table_heading_row",
        }.get(containing.kind, "heading_plus_sentence")
        candidates.append((
            parent.start,
            end,
            expansion,
            2,
        ))
    if containing_index > 0:
        previous = structures[containing_index - 1]
        if previous.kind != "table_header":
            candidates.append((
                previous.start,
                end,
                (
                    "heading_plus_sentence"
                    if previous.kind == "heading"
                    else "preceding_sentence"
                ),
                2,
            ))
    if containing_index + 1 < len(structures):
        following = structures[containing_index + 1]
        if following.kind == "sentence":
            candidates.append((
                start,
                following.end,
                "following_sentence",
                2,
            ))

    seen: set[tuple[int, int]] = set()
    for evidence_start, evidence_end, expansion, count in candidates:
        bounds = (evidence_start, evidence_end)
        if bounds in seen or not _bounded(text, evidence_start, evidence_end):
            continue
        seen.add(bounds)
        before = text[evidence_start:start]
        if not _resolved_by_context(focus, before, dependency):
            continue
        evidence = re.sub(r"\s+", " ", text[evidence_start:evidence_end]).strip()
        return ContextEvidenceSpan(
            focus_text=re.sub(r"\s+", " ", focus).strip(),
            evidence_text=evidence,
            focus_start=start,
            focus_end=end,
            evidence_start=evidence_start,
            evidence_end=evidence_end,
            dependency_codes=dependency.reason_codes,
            expansion_type=expansion,
            sentence_count=count,
            complete=True,
        )
    return ContextEvidenceSpan(
        focus_text=re.sub(r"\s+", " ", focus).strip(),
        evidence_text=re.sub(r"\s+", " ", focus).strip(),
        focus_start=start,
        focus_end=end,
        evidence_start=start,
        evidence_end=end,
        dependency_codes=dependency.reason_codes,
        expansion_type="context_unresolved",
        sentence_count=1,
        complete=False,
    )
