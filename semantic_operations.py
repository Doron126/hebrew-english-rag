"""Domain-independent semantic request operations and deterministic executors.

The module deliberately contains no tenant, document, page, question, or
acceptance-data constants.  It turns a request into an operator-shaped frame
and executes only operations whose result can be derived from bounded,
source-linked evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import math
import os
import re
import unicodedata
from typing import Any, Iterable, Mapping, Optional, Sequence

from evidence_context import (
    EvidenceContext,
    RelationArgument,
    build_evidence_context,
    structural_binding_validation_errors,
)
from evidence_quotes import opaque_evidence_chunk_id
from semantic_contracts import (
    EDUCATION_TERM_PATTERN,
    candidate_id as semantic_candidate_id,
    classify_quantity_semantics,
    classify_scalar_role,
    normalize_aggregation_role,
    normalize_quantity_kind,
    normalize_scalar_role,
    normalize_statistical_role,
    quantity_semantics_from_scalar_role,
    scalar_role_from_quantity_semantics,
)
from eval_trace import current as current_eval_trace
from structural_evidence import (
    StructuralEvidenceType,
    StructuralEvidenceUnit,
    bounded_adjacent_structural_unit,
    canonical_structural_entry,
)


MAX_FRAME_VALUES = 12
MAX_FRAME_TEXT = 240
MAX_OPERATOR_FACTS = 256

_BIDI_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")
_TOKEN_RE = re.compile(r"[A-Za-z0-9\u05d0-\u05ea]+")
_PERIOD_RE = re.compile(
    r"(?<![\w])(?:19|20)\d{2}(?![\w])|"
    r"(?<![\w])(?:Q[1-4]|H[12])[- /]?(?:19|20)?\d{2}(?![\w])",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(
    r"(?P<approx>כ(?:־|-)?|~|approximately\s+|about\s+)?"
    r"(?P<value>\d{1,3}(?:,\d{3})+|\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>%|percent(?:age)?|MW|GW|kW|MWh|GWh|"
    r"million|billion|thousand|אלף|מיליון|מיליארד|"
    r"עובדים|משרות|חברות|יחידות|לקוחות|רשומות|שקלים|דולר(?:ים)?|"
    r"employees?|workers?|jobs?|positions?|companies|units?|customers?|"
    r"accounts?|records?|facilities|stations?|sites?)?",
    re.IGNORECASE,
)


class RequestOperator(str, Enum):
    DIRECT_LOOKUP = "DIRECT_LOOKUP"
    DEFINITION = "DEFINITION"
    COUNT = "COUNT"
    LIST = "LIST"
    LIST_WITH_ATTRIBUTES = "LIST_WITH_ATTRIBUTES"
    COMPARE = "COMPARE"
    CHANGE_OVER_TIME = "CHANGE_OVER_TIME"
    ARGMAX = "ARGMAX"
    ARGMIN = "ARGMIN"
    FILTER = "FILTER"
    EXCLUSION = "EXCLUSION"
    EXCEPTION = "EXCEPTION"
    CAUSE = "CAUSE"
    EXPLANATION = "EXPLANATION"
    SUMMARY = "SUMMARY"
    PROCEDURE = "PROCEDURE"
    VERIFY_PREVIOUS = "VERIFY_PREVIOUS"
    CORRECT_PREVIOUS = "CORRECT_PREVIOUS"
    EXPAND_PREVIOUS = "EXPAND_PREVIOUS"
    ANOTHER_RESULT = "ANOTHER_RESULT"
    SOURCE_SCOPED_QUERY = "SOURCE_SCOPED_QUERY"
    DOCUMENT_QUIZ = "DOCUMENT_QUIZ"


class SemanticPredicate(str, Enum):
    VALUE = "value"
    MAXIMUM = "maximum"
    MINIMUM = "minimum"
    NOT_INCLUDED = "not_included"
    NOT_MEASURED = "not_measured"
    NOT_AVAILABLE = "not_available"
    ABSENT = "absent"
    OUT_OF_SCOPE = "out_of_scope"
    UNDERREPRESENTED = "underrepresented"
    ZERO = "zero"
    COMPARED_WITH = "compared_with"
    VERIFIED = "verified"


@dataclass(frozen=True)
class SemanticRequirement:
    id: str
    kind: str
    metric: str = ""
    period: str = ""
    entity: str = ""
    unit: str = ""
    qualifier: str = ""
    mandatory: bool = True

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "metric": self.metric,
            "period": self.period,
            "entity": self.entity,
            "unit": self.unit,
            "qualifier": self.qualifier,
            "mandatory": self.mandatory,
        }


@dataclass(frozen=True)
class SemanticFrame:
    operation: RequestOperator
    normalized_request: str
    entities: tuple[str, ...] = ()
    metrics: tuple[str, ...] = ()
    periods: tuple[str, ...] = ()
    qualifiers: tuple[str, ...] = ()
    requirements: tuple[SemanticRequirement, ...] = ()
    confidence: float = 0.0
    deterministic: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "normalized_request": self.normalized_request,
            "entities": list(self.entities),
            "metrics": list(self.metrics),
            "periods": list(self.periods),
            "qualifiers": list(self.qualifiers),
            "requirements": [item.as_dict() for item in self.requirements],
            "confidence": round(self.confidence, 3),
            "deterministic": self.deterministic,
        }


@dataclass(frozen=True)
class SemanticFact:
    entity: str
    predicate: str
    metric: str
    value: str
    unit: str
    period: str
    population: str
    direction: str
    negated: bool
    qualifiers: tuple[str, ...]
    evidence_id: str
    source_claim: str
    hit_index: int = -1
    evidence_subject: str = ""
    evidence_predicate: str = ""
    evidence_target_entity: str = ""
    evidence_population: str = ""
    evidence_period: str = ""
    evidence_unit: str = ""
    evidence_numeric_role: str = ""
    scalar_role: str = ""
    quantity_kind: str = ""
    statistical_role: str = ""
    aggregation_role: str = ""
    evidence_span: tuple[int, int] = (0, 0)
    evidence_facet: str = ""
    fact_id: str = ""
    source_fact_id: str = ""
    candidate_id: str = ""
    source_evidence_unit_ids: tuple[str, ...] = ()
    source_candidate_ids: tuple[str, ...] = ()
    source_typed_fact_ids: tuple[str, ...] = ()
    operator_derivation_id: str = ""
    derivation_description: str = ""
    origin_debug_id: str = ""
    evidence_context: Optional[EvidenceContext] = field(
        default=None, repr=False, compare=False
    )
    relation_arguments: tuple[RelationArgument, ...] = ()
    structural_binding_valid: bool = True
    structural_validation_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        scalar_role = self.scalar_role
        quantity_semantics = None
        if not scalar_role and not any((
            self.quantity_kind, self.statistical_role, self.aggregation_role,
        )):
            source_text = str(
                getattr(self.evidence_context, "local_proposition", "")
                or self.source_claim
                or ""
            )
            normalized_value = re.sub(r"[,_\s]", "", self.value or "")
            for match in _NUMBER_RE.finditer(source_text):
                candidate_value = re.sub(
                    r"[,_\s]", "", match.group("value") or "",
                )
                if candidate_value != normalized_value:
                    continue
                quantity_semantics = classify_quantity_semantics(
                    source_text,
                    match.start("value"),
                    match.end("value"),
                    unit=(self.evidence_unit or self.unit),
                    structural_role=(
                        "category"
                        if getattr(self.evidence_context, "category", "")
                        else ""
                    ),
                )
                scalar_role = quantity_semantics.scalar_role
                break
        legacy = quantity_semantics or quantity_semantics_from_scalar_role(
            scalar_role or self.evidence_numeric_role
        )
        quantity_kind = normalize_quantity_kind(
            self.quantity_kind or legacy.quantity_kind
        )
        statistical_role = normalize_statistical_role(
            self.statistical_role or legacy.statistical_role
        )
        aggregation_role = normalize_aggregation_role(
            self.aggregation_role or legacy.aggregation_role
        )
        object.__setattr__(self, "quantity_kind", quantity_kind)
        object.__setattr__(self, "statistical_role", statistical_role)
        object.__setattr__(self, "aggregation_role", aggregation_role)
        object.__setattr__(self, "scalar_role", scalar_role_from_quantity_semantics(
            quantity_kind, statistical_role, aggregation_role,
        ))
        if not self.fact_id:
            payload = "\x1f".join((
                self.evidence_id, self.entity, self.predicate, self.metric,
                self.value, self.unit, self.period, str(self.hit_index),
                self.operator_derivation_id,
            ))
            object.__setattr__(
                self,
                "fact_id",
                "fact-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24],
            )
        if not self.source_fact_id:
            object.__setattr__(self, "source_fact_id", self.fact_id)
        if not self.source_evidence_unit_ids:
            object.__setattr__(self, "source_evidence_unit_ids", (self.evidence_id,))
        if not self.source_candidate_ids and self.candidate_id:
            object.__setattr__(self, "source_candidate_ids", (self.candidate_id,))
        structural_errors = tuple(self.structural_validation_errors)
        if self.evidence_context is not None:
            structural_errors += structural_binding_validation_errors(
                self.evidence_context
            )
        else:
            structural_errors += ("structural_parent_missing",)
        structural_errors = tuple(dict.fromkeys(structural_errors))
        object.__setattr__(self, "structural_validation_errors", structural_errors)
        object.__setattr__(self, "structural_binding_valid", not structural_errors)

    def as_dict(self, *, include_source_claim: bool = False) -> dict[str, Any]:
        value = {
            "entity": self.entity,
            "predicate": self.predicate,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "period": self.period,
            "population": self.population,
            "direction": self.direction,
            "negated": self.negated,
            "qualifiers": list(self.qualifiers),
            "evidence_id": self.evidence_id,
            "evidence_subject": self.evidence_subject,
            "evidence_predicate": self.evidence_predicate,
            "evidence_target_entity": self.evidence_target_entity,
            "evidence_population": self.evidence_population,
            "evidence_period": self.evidence_period,
            "evidence_unit": self.evidence_unit,
            "evidence_numeric_role": self.evidence_numeric_role,
            "scalar_role": self.scalar_role,
            "quantity_kind": self.quantity_kind,
            "statistical_role": self.statistical_role,
            "aggregation_role": self.aggregation_role,
            "evidence_span": list(self.evidence_span),
            "evidence_facet": self.evidence_facet,
            "fact_id": self.fact_id,
            "source_fact_id": self.source_fact_id,
            "candidate_id": self.candidate_id,
            "source_evidence_unit_ids": list(self.source_evidence_unit_ids),
            "source_candidate_ids": list(self.source_candidate_ids),
            "source_typed_fact_ids": list(self.source_typed_fact_ids),
            "operator_derivation_id": self.operator_derivation_id,
            "derivation_description": self.derivation_description,
            "origin_debug_id": self.origin_debug_id,
            "evidence_context": (
                self.evidence_context.as_dict() if self.evidence_context else None
            ),
            "relation_arguments": [
                item.as_dict() for item in self.relation_arguments
            ],
            "structural_binding_valid": self.structural_binding_valid,
            "structural_validation_errors": list(
                self.structural_validation_errors
            ),
        }
        if include_source_claim:
            value["source_claim"] = self.source_claim[:MAX_FRAME_TEXT]
        return value

    def as_prompt_dict(self) -> dict[str, Any]:
        """Expose an operator source fact through the shared fact lifecycle.

        Operators retain their deterministic semantics, but their evidence is
        serialized in the same compact source-fact shape as ordinary facts.
        The synthetic atom is an auditable view of this already source-bound
        operator input; it never creates new evidence or changes execution.
        """
        value = self.as_dict(include_source_claim=True)
        source_text = str(self.source_claim or "")
        start, end = tuple(self.evidence_span or (0, 0))
        if end <= start:
            start, end = 0, max(1, len(source_text))
        atom_id = "operator-atom-" + hashlib.sha256(
            self.fact_id.encode("utf-8")
        ).hexdigest()[:24]
        text_sha256 = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
        value.update({
            "source_fact_id": self.source_fact_id,
            "fact_type": "operator_source_fact",
            "subject": self.entity,
            "structural_type": (
                self.evidence_context.structural_type
                if self.evidence_context is not None else "operator_evidence"
            ),
            "evidence_span_sha256": text_sha256,
            "evidence_atom_id": atom_id,
            "evidence_atom": {
                "atom_id": atom_id,
                "evidence_unit_id": self.evidence_id,
                "candidate_id": self.candidate_id,
                "source_span": [start, end],
                "text_sha256": text_sha256,
                "atom_kind": "operator_evidence",
                "subject": self.evidence_subject or self.entity,
                "predicate": self.evidence_predicate or self.predicate,
                "target_entity": self.evidence_target_entity,
                "value": self.value,
                "unit": self.evidence_unit or self.unit,
                "period": self.evidence_period or self.period,
                "population": self.evidence_population or self.population,
                "facet": self.evidence_facet,
                "scalar_role": self.scalar_role,
                "quantity_kind": self.quantity_kind,
                "statistical_role": self.statistical_role,
                "aggregation_role": self.aggregation_role,
                "polarity": "negative" if self.negated else "neutral",
            },
            "source_semantic_fields": {
                "subject": self.evidence_subject or self.entity,
                "predicate": self.evidence_predicate or self.predicate,
                "target_entity": self.evidence_target_entity,
                "population": self.evidence_population or self.population,
                "period": self.evidence_period or self.period,
                "unit": self.evidence_unit or self.unit,
                "numeric_role": self.evidence_numeric_role,
                "scalar_role": self.scalar_role,
                "quantity_kind": self.quantity_kind,
                "statistical_role": self.statistical_role,
                "aggregation_role": self.aggregation_role,
                "polarity": "negative" if self.negated else "neutral",
            },
        })
        return value


@dataclass(frozen=True)
class OperatorExecution:
    operation: RequestOperator
    accepted: bool
    facts: tuple[SemanticFact, ...]
    result: Optional[SemanticFact]
    evidence_hit_indexes: tuple[int, ...]
    reason_code: str
    scan_complete: bool
    result_attributes: tuple[SemanticFact, ...] = ()
    candidate_order: tuple[SemanticFact, ...] = ()

    def safe_summary(self) -> dict[str, Any]:
        return {
            "operation": self.operation.value,
            "accepted": self.accepted,
            "fact_count": len(self.facts),
            "facts": [fact.as_dict() for fact in self.facts[:16]],
            "result": self.result.as_dict() if self.result else None,
            "evidence_hit_indexes": list(self.evidence_hit_indexes),
            "reason_code": self.reason_code,
            "scan_complete": self.scan_complete,
            "result_attributes": [fact.as_dict() for fact in self.result_attributes],
            "candidate_order": [fact.as_dict() for fact in self.candidate_order[:32]],
        }


_VERIFY_RE = re.compile(
    r"(?:\b(?:recheck|re-check|verify|validate|confirm)\b|"
    r"(?:בדוק|תבדוק|בדקי|אמת|אמתו|וודא|ודא).{0,36}(?:שוב|מחדש|את\s+(?:זה|התשובה))|"
    r"(?:שוב|מחדש).{0,24}(?:בדוק|תבדוק|אמת|וודא|ודא))",
    re.IGNORECASE,
)
_CORRECT_RE = re.compile(
    r"(?:\bcorrect\s+(?:that|the\s+(?:(?:previous|prior)\s+)?(?:answer|result))\b|"
    r"(?:תקן|תתקן|תקני).{0,36}(?:את\s+)?(?:זה|זאת|אותו|אותה|התשובה|התוצאה))",
    re.IGNORECASE,
)
_EXPAND_RE = re.compile(
    r"(?:\b(?:expand|elaborate)\b|(?:תוכל\s+|תוכלי\s+)?"
    r"(?:פרט|תפרט|להרחיב|הרחב|תרחיב).{0,24}(?:עוד|יותר|על|$))",
    re.IGNORECASE,
)
_ANOTHER_RE = re.compile(
    r"(?:\banother\s+(?:result|one|fact|example)\b|"
    r"(?:עוד|נוסף|אחר).{0,16}(?:תוצאה|אחד|עובדה|דוגמה|ממצא)|"
    r"הבא(?:ה)?\s+אחרי(?:ו|ה))",
    re.IGNORECASE,
)
_ARGMAX_RE = re.compile(
    r"(?:\b(?:largest|highest|greatest|maximum|most)\b|"
    r"(?:גדול|גבוה|רב|מרבי|מקסימלי)(?:ה|ה ביותר|\s+ביותר)|"
    r"הכי\s+(?:גדול|גבוה|רב))",
    re.IGNORECASE,
)
_ARGMIN_RE = re.compile(
    r"(?:\b(?:smallest|lowest|least|minimum)\b|"
    r"(?:קטן|נמוך|מועט|מזערי)(?:ה|ה ביותר|\s+ביותר)|"
    r"הכי\s+(?:קטן|נמוך|מועט))",
    re.IGNORECASE,
)
_EXCLUSION_RE = re.compile(
    r"(?:\b(?:excluded|omitted|not\s+included|did\s+not\s+include|"
    r"wasn['’]?t\s+included|outside\s+the\s+scope)\b|"
    r"(?<![\u05d0-\u05ea])(?:לא|אינ(?:ו|ה|ם|ן)|איננו|אינה)"
    r"(?![\u05d0-\u05ea]).{0,28}(?:נכלל|נכללה|נכללו|כולל|כוללת)|"
    r"מחוץ\s+ל(?:אוכלוסייה|תחום|אומדן|היקף))",
    re.IGNORECASE,
)
_EXCEPTION_RE = re.compile(
    r"(?:\b(?:except|exception|other\s+than)\b|(?:למעט|פרט\s+ל|חריג))",
    re.IGNORECASE,
)
_DEFINITION_RE = re.compile(
    r"(?:^\s*(?:מהו|מהי|מהם|מהן)\s+(?:הגדר|המונח|מושג)|"
    r"\b(?:define|definition|what\s+does\s+.+\s+mean)\b)",
    re.IGNORECASE,
)
_COUNT_RE = re.compile(
    r"(?:^\s*כמה\b(?!\s+(?:כסף|שקלים|דולרים|אירו|התקבל|שולם))|\bhow\s+many\b|\b(?:count|number)\s+of\b)",
    re.IGNORECASE,
)
_LIST_RE = re.compile(
    r"(?:^\s*(?:רשום|רשמי|מנה|ציין|אילו|מהם|מהן)\b|"
    r"^\s*(?:list|name|enumerate|which\s+are|what\s+are)\b)",
    re.IGNORECASE,
)
_ATTRIBUTE_RE = re.compile(
    r"(?:לכל|עבור\s+כל|אחריות|מאפיינ|תפקידו|חוב(?:ה|ות)|פונקצי|"
    r"\b(?:for\s+each|attribute|responsibilit|characteristic|dut(?:y|ies)|"
    r"function|accountabilit|with\s+their)\b)",
    re.IGNORECASE,
)
_CHANGE_RE = re.compile(
    r"(?:\b(?:change|changed|trend|increase|increased|decrease|decreased|"
    r"grew|fell|over\s+time)\b|"
    r"השתנ|מגמ|עלייה|יריד|גדל|קטן|עלה|ירד)",
    re.IGNORECASE,
)
_COMPARE_RE = re.compile(
    r"(?:\b(?:compare|comparison|versus|vs\.?)\b|השוו|השווה|השוואה|לעומת|בין.+לבין)",
    re.IGNORECASE,
)
_FILTER_RE = re.compile(
    r"(?:\b(?:filter|only\s+those|where|meeting|exceed(?:s|ed|ing)?|"
    r"greater\s+than|more\s+than|above|over|less\s+than|below|under|"
    r"at\s+least|at\s+most|before|after)\b|"
    r"(?:סנן|רק\s+(?:אלה|אלו|מי)|שעומד|(?:עלה|עלתה|עולה|עבר|עברה)\s+על|"
    r"מעל|יותר\s+מ|פחות\s+מ|מתחת|לפחות|לכל\s+היותר|לפני|אחרי))",
    re.IGNORECASE,
)
_CAUSE_RE = re.compile(
    r"(?:^\s*(?:למה|מדוע|why)\b|\b(?:cause|caused|because|why\s+did)\b|"
    r"(?:explain|explains|explained|מסביר|מסבירה).{0,20}(?:why|מדוע|למה)|"
    r"מה\s+גרם|הסיבה\s+ל)",
    re.IGNORECASE,
)
_EXPLANATION_RE = re.compile(
    r"(?:\b(?:explain|explanation|what\s+does\s+this\s+mean)\b|"
    r"(?:הסבר|תסביר|מה\s+המשמעות))",
    re.IGNORECASE,
)
_SUMMARY_RE = re.compile(
    r"(?:\b(?:summary|summari[sz]e|overview|takeaways)\b|(?:סכם|סכמי|תקציר|תמצית))",
    re.IGNORECASE,
)
_PROCEDURE_RE = re.compile(
    r"(?:\b(?:how\s+to|procedure|steps\s+to)\b|(?:איך|כיצד).{0,24}(?:לבצע|לעשות|לפתוח|להוריד))",
    re.IGNORECASE,
)
_QUIZ_RE = re.compile(
    r"(?:\b(?:quiz|test)\s+me\b|(?:בחן|תבחן|בחני|תבחני).{0,18}(?:אותי|מהמסמך|על\s+המסמך))",
    re.IGNORECASE,
)


def normalize_semantic_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = _BIDI_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _unique(values: Iterable[str], *, maximum: int = MAX_FRAME_VALUES) -> tuple[str, ...]:
    output: list[str] = []
    for raw in values:
        value = normalize_semantic_text(raw).strip(" ,;:.?!")[:MAX_FRAME_TEXT]
        if value and value.casefold() not in {item.casefold() for item in output}:
            output.append(value)
        if len(output) >= maximum:
            break
    return tuple(output)


def _extract_periods(value: str) -> tuple[str, ...]:
    return _unique(match.group(0) for match in _PERIOD_RE.finditer(value))


def _extract_entities(value: str) -> tuple[str, ...]:
    quoted = re.findall(r"[\"“”'׳]([^\"“”'׳]{2,100})[\"“”'׳]", value)
    acronyms = re.findall(r"(?<![\w])(?:[A-Z][A-Za-z0-9.+/-]{1,30})(?![\w])", value)
    return _unique((*quoted, *acronyms))


def _extract_metrics(value: str) -> tuple[str, ...]:
    lowered = value.casefold()
    families: list[str] = []
    family_patterns = (
        ("demand", r"(?:ביקוש|\bdemand\b)"),
        ("share", r"(?:שיעור|אחוז|חלק|נתח|\b(?:share|rate|percent(?:age)?)\b)"),
        ("experience", r"(?:ניסיון|וותק|\bexperience\b|\btenure\b)"),
        ("education", EDUCATION_TERM_PATTERN),
        ("revenue", r"(?:הכנס|\brevenue\b)"),
        ("profit", r"(?:רווח|\bprofit\b)"),
        ("margin", r"(?:מרווח|שולי\s+רווח|\bmargin\b)"),
        ("capacity", r"(?:קיבולת|הספק|\bcapacity\b)"),
        ("cost", r"(?:עלות|מחיר|\b(?:cost|price)\b)"),
        ("count", r"(?:מספר|כמות|היקפ|\b(?:count|number|extent)\b)"),
        ("role", r"(?:תפקיד|עיסוק|סוגי?\s+משרה|\b(?:role|occupation|job\s+type)\b)"),
        ("education_level", r"(?:"+EDUCATION_TERM_PATTERN+r"|\bqualification\b)"),
        ("experience_band", r"(?:ניסיון|וותק|\b(?:experience|tenure|seniority)\b)"),
        ("employee_count", r"(?:מעסיק|מעסיקה|מעסיקות|מועסק|מספר\s+העובדים|\b(?:employ|employs|employed|headcount|workforce)\b)"),
        ("employee_recruitment", r"(?:(?:גייס|גיוס).{0,24}(?:עובד|מומח|כוח\s*אדם)|\b(?:recruit|hire).{0,24}(?:employee|worker|staff))"),
        ("capital_raised", r"(?:(?:גייס|גיוס).{0,24}(?:כסף|הון|מימון)|\b(?:raise|fundrais).{0,24}(?:capital|money|fund))"),
    )
    for family, pattern in family_patterns:
        if re.search(pattern, lowered, re.IGNORECASE):
            families.append(family)
    lower_band = bool(re.search(
        r"(?:עד\s+\S+|פחות\s+מ|מתחת\s+ל|\b(?:up\s+to|under|less\s+than|below)\b)",
        lowered,
        re.IGNORECASE,
    ))
    higher_band = bool(re.search(
        r"(?:מעל|יותר\s+מ|לפחות|\b(?:over|above|more\s+than|at\s+least)\b)",
        lowered,
        re.IGNORECASE,
    ))
    if lower_band and higher_band:
        base = "_".join(families) or "band"
        return (f"{base}:lower_band", f"{base}:higher_band")
    patterns = (
        r"(?:שיעור|אחוז|חלק|נתח|מספר|כמות|אומדן|ביקוש|הכנסה|רווח|קיבולת|עלות)"
        r"(?:\s+[\u05d0-\u05ea\"'׳/-]+){0,5}",
        r"\b(?:share|rate|percentage|count|number|estimate|demand|revenue|profit|"
        r"margin|capacity|cost)(?:\s+[A-Za-z0-9'/-]+){0,5}\b",
    )
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(match.group(0) for match in re.finditer(pattern, value, re.IGNORECASE))
    return _unique(matches)


def _requirements(
    operation: RequestOperator,
    metrics: Sequence[str],
    periods: Sequence[str],
) -> tuple[SemanticRequirement, ...]:
    metric = metrics[0] if metrics else ""
    if operation in {RequestOperator.ARGMAX, RequestOperator.ARGMIN}:
        return (
            SemanticRequirement("candidate_set", "complete_candidate_set", metric=metric),
            SemanticRequirement("category_values", "category_value_bindings", metric=metric),
            SemanticRequirement("compatible_scope", "compatible_period_unit_population", metric=metric),
            SemanticRequirement("winner_context", "winning_value_and_set_context", metric=metric),
        )
    if operation in {RequestOperator.EXCLUSION, RequestOperator.EXCEPTION}:
        return (
            SemanticRequirement("scope_subject", "subject_scope", metric=metric),
            SemanticRequirement("negative_predicate", "negative_relation", metric=metric),
            SemanticRequirement("excluded_entity", "bound_excluded_entity", metric=metric),
            SemanticRequirement("qualifier", "preserved_scope_qualifier", metric=metric),
        )
    if operation in {RequestOperator.COMPARE, RequestOperator.CHANGE_OVER_TIME}:
        requested_periods = tuple(periods) or ("endpoint_a", "endpoint_b")
        output: list[SemanticRequirement] = []
        for metric_index, metric_value in enumerate(metrics or (metric,), start=1):
            for period in requested_periods:
                output.append(SemanticRequirement(
                    f"metric_{metric_index}_{period}",
                    "comparison_endpoint",
                    metric=metric_value,
                    period=period,
                ))
        return tuple(output)
    if operation is RequestOperator.VERIFY_PREVIOUS:
        return (
            SemanticRequirement("previous_question", "previous_factual_question"),
            SemanticRequirement("alternate_retrieval", "alternate_evidence_retrieval"),
            SemanticRequirement("fact_comparison", "previous_and_new_typed_fact_comparison"),
            SemanticRequirement("direct_citations", "direct_supporting_citations"),
        )
    return (SemanticRequirement("fact", operation.value.casefold(), metric=metric),)


def infer_semantic_frame(
    query: str,
    *,
    has_previous: bool = False,
    selected_source: bool = False,
) -> SemanticFrame:
    """Return a bounded deterministic frame when the operation is explicit.

    Low-confidence requests remain DIRECT_LOOKUP candidates so the caller can
    invoke its single local planning call.  High-confidence branches describe
    semantic relations rather than any domain-specific answer.
    """
    value = normalize_semantic_text(query)
    periods = _extract_periods(value)
    entities = _extract_entities(value)
    metrics = _extract_metrics(value)
    qualifiers = _unique(
        item
        for item, pattern in (
            ("approximate", r"(?:כ(?:־|-)?\s*\d|~\s*\d|\b(?:approximately|about)\s+\d)"),
            (
                "negated",
                r"(?:\b(?:not|no|without|excluded)\b|"
                r"(?<![\u05d0-\u05ea])(?:לא|אין|אינו|אינה|אינם|אינן)(?![\u05d0-\u05ea]))",
            ),
            ("current", r"(?:בעת\s+(?:הסקר|המדידה)|כיום|\bcurrent(?:ly)?\b)"),
            ("population_scoped", r"(?:אוכלוסי|מדגם|קבוצה|\b(?:population|sample|group)\b)"),
        )
        if re.search(pattern, value, re.IGNORECASE)
    )

    operation = RequestOperator.DIRECT_LOOKUP
    confidence = 0.56
    explicit_multi_item_list = bool(
        _LIST_RE.search(value)
        and re.search(
            r"(?<![\w])(?:[2-9]|two|three|four|five|several|multiple|"
            r"שניים|שתיים|שלוש|שלושה|ארבע|ארבעה|חמש|חמישה)(?![\w])",
            value,
            re.IGNORECASE,
        )
    )
    if has_previous and _VERIFY_RE.search(value):
        operation, confidence = RequestOperator.VERIFY_PREVIOUS, 0.99
    elif has_previous and _CORRECT_RE.search(value):
        operation, confidence = RequestOperator.CORRECT_PREVIOUS, 0.98
    elif has_previous and _EXPAND_RE.search(value):
        operation, confidence = RequestOperator.EXPAND_PREVIOUS, 0.96
    elif has_previous and _ANOTHER_RE.search(value):
        operation, confidence = RequestOperator.ANOTHER_RESULT, 0.96
    elif _QUIZ_RE.search(value):
        operation, confidence = RequestOperator.DOCUMENT_QUIZ, 0.98
    elif _ARGMAX_RE.search(value) and not explicit_multi_item_list:
        operation, confidence = RequestOperator.ARGMAX, 0.97
    elif _ARGMIN_RE.search(value):
        operation, confidence = RequestOperator.ARGMIN, 0.97
    elif _CAUSE_RE.search(value):
        operation, confidence = RequestOperator.CAUSE, 0.95
    elif _EXCLUSION_RE.search(value):
        operation, confidence = RequestOperator.EXCLUSION, 0.97
    elif _EXCEPTION_RE.search(value):
        operation, confidence = RequestOperator.EXCEPTION, 0.96
    elif (
        (_CHANGE_RE.search(value) and (len(periods) >= 2 or _COMPARE_RE.search(value)))
        or (_COMPARE_RE.search(value) and len(periods) >= 2)
    ):
        operation, confidence = RequestOperator.CHANGE_OVER_TIME, 0.96
    elif _COMPARE_RE.search(value):
        operation, confidence = RequestOperator.COMPARE, 0.96
    elif _FILTER_RE.search(value):
        operation, confidence = RequestOperator.FILTER, 0.94
    elif _DEFINITION_RE.search(value):
        operation, confidence = RequestOperator.DEFINITION, 0.94
    elif _COUNT_RE.search(value):
        operation, confidence = RequestOperator.COUNT, 0.94
    elif _LIST_RE.search(value) and _ATTRIBUTE_RE.search(value):
        operation, confidence = RequestOperator.LIST_WITH_ATTRIBUTES, 0.92
    elif _LIST_RE.search(value):
        operation, confidence = RequestOperator.LIST, 0.91
    elif _EXPLANATION_RE.search(value):
        operation, confidence = RequestOperator.EXPLANATION, 0.92
    elif _SUMMARY_RE.search(value):
        operation, confidence = RequestOperator.SUMMARY, 0.94
    elif _PROCEDURE_RE.search(value):
        operation, confidence = RequestOperator.PROCEDURE, 0.91
    elif selected_source:
        operation, confidence = RequestOperator.SOURCE_SCOPED_QUERY, 0.95

    return SemanticFrame(
        operation=operation,
        normalized_request=value,
        entities=entities,
        metrics=metrics,
        periods=periods,
        qualifiers=qualifiers,
        requirements=_requirements(operation, metrics, periods),
        confidence=confidence,
        deterministic=confidence >= 0.84,
    )


def semantic_frame_for_operation(
    query: str,
    operation: RequestOperator,
    *,
    confidence: float = 0.80,
    deterministic: bool = False,
) -> SemanticFrame:
    """Build the same bounded frame when one local planner chose the operator."""
    value = normalize_semantic_text(query)
    metrics = _extract_metrics(value)
    periods = _extract_periods(value)
    base = infer_semantic_frame(query)
    return SemanticFrame(
        operation=operation,
        normalized_request=value,
        entities=_extract_entities(value),
        metrics=metrics,
        periods=periods,
        qualifiers=base.qualifiers,
        requirements=_requirements(operation, metrics, periods),
        confidence=confidence,
        deterministic=deterministic,
    )


_REJECTED_STRUCTURAL_TYPES = frozenset({
    "heading", "caption", "table_of_contents", "list_of_figures",
    "list_of_tables", "header_footer", "page_number", "formula", "malformed",
})


def _hit_parts(
    hit: Mapping[str, Any],
    index: int,
) -> tuple[str, Mapping[str, Any], str, str, str]:
    metadata = hit.get("metadata") if isinstance(hit, Mapping) else {}
    if not isinstance(metadata, Mapping):
        metadata = {}
    text = normalize_semantic_text(hit.get("document") if isinstance(hit, Mapping) else "")
    chunk_id = str(
        metadata.get("chunk_id")
        or metadata.get("evidence_chunk_id")
        or hit.get("id")
        or f"hit-{index}"
    )
    source_id = str(metadata.get("source_id") or "")
    try:
        page = int(metadata.get("page")) if metadata.get("page") is not None else None
    except (TypeError, ValueError, OverflowError):
        page = None
    evidence_id = (
        opaque_evidence_chunk_id(source_id, chunk_id, page)
        if source_id and chunk_id
        else "evidence-" + hashlib.sha256(
            f"{chunk_id}\x1f{text}".encode("utf-8")
        ).hexdigest()[:24]
    )
    return text, metadata, evidence_id, semantic_candidate_id(hit), chunk_id


def _candidate_mode_enabled() -> bool:
    return os.getenv("PROJECT_X_CANDIDATE_MODE", "0").strip().casefold() in {
        "1", "true", "yes", "on",
    }


def _operator_structural_units(
    hit: Mapping[str, Any],
    index: int,
) -> tuple[tuple[str, Mapping[str, Any], str, str, str, StructuralEvidenceUnit], ...]:
    """Return the canonical source-only units consumed by every operator."""
    _normalized_text, metadata, _chunk_evidence_id, candidate_id, chunk_id = _hit_parts(
        hit, index
    )
    text = str(hit.get("document") or "")
    source_id = str(metadata.get("source_id") or "")
    if not source_id:
        # Production candidates must never manufacture source identity.  The
        # bounded fallback exists only for historical pure-unit fixtures.
        if _candidate_mode_enabled():
            return ()
        source_id = "fixture-source-" + hashlib.sha256(
            chunk_id.encode("utf-8")
        ).hexdigest()[:16]
    try:
        page = int(metadata.get("page")) if metadata.get("page") is not None else None
    except (TypeError, ValueError, OverflowError):
        page = None
    entry = canonical_structural_entry(
        text,
        source_id=source_id,
        candidate_id=candidate_id,
        chunk_id=chunk_id,
        page=page,
        metadata=metadata,
    )
    recorder = current_eval_trace()
    if recorder is not None:
        recorder.record_structural_decomposition(
            candidate_id,
            status="units_created" if entry.structural_units else "rejected",
            reason=(
                "structural_units_created" if entry.structural_units
                else entry.rejection_reason
            ),
            units=entry.structural_units,
            entry=entry,
        )
    units = entry.structural_units
    output = []
    for unit in units:
        unit_metadata = dict(metadata)
        # Chunk-level row/category fields may describe a neighboring record.
        # Only the unit-local label is allowed to enter operator semantics.
        for key in ("row_header", "category", "column_header", "value_header"):
            unit_metadata.pop(key, None)
        if unit.row_label:
            unit_metadata["row_header"] = unit.row_label
            unit_metadata["category"] = unit.row_label
        output.append((
            unit.local_text,
            unit_metadata,
            unit.unit_id,
            candidate_id,
            chunk_id,
            unit,
        ))
    return tuple(output)


def _operator_evidence_context(
    *,
    text: str,
    metadata: Mapping[str, Any],
    evidence_id: str,
    candidate_id: str,
    chunk_id: str,
    proposition: str,
    span: tuple[int, int],
    subject: str,
    predicate: str,
    target_entity: str,
    population: str,
    period: str,
    unit: str,
    numeric_role: str,
    polarity: str,
    relation_arguments: Optional[Mapping[str, str]] = None,
    structural_evidence_unit: Optional[StructuralEvidenceUnit] = None,
) -> EvidenceContext:
    try:
        page = int(metadata.get("page")) if metadata.get("page") is not None else None
    except (TypeError, ValueError, OverflowError):
        page = None
    semantics = {
        "subject": subject,
        "predicate": predicate,
        "target_entity": target_entity,
        "population": population,
        "period": period,
        "unit": unit,
        "numeric_role": numeric_role,
        "polarity": polarity,
        "category": subject,
    }
    whole = (0, len(proposition))
    literal_fields = {"subject", "target_entity", "population", "period", "category"}
    metadata_containers = (
        "row_header", "category", "table_title", "chart_title",
        "section_title", "section", "title", "population", "year",
    )
    sources: dict[str, dict[str, Any]] = {}
    for name, value in semantics.items():
        if not value:
            continue
        normalized_value = str(value)
        position = proposition.casefold().find(normalized_value.casefold())
        source_container = "local_proposition"
        source_text = proposition
        source_span = (
            (position, position + len(normalized_value))
            if position >= 0 else whole
        )
        if name in literal_fields and position < 0:
            structural_match = next((
                (key, str(metadata.get(key) or ""))
                for key in metadata_containers
                if str(metadata.get(key) or "")
                and normalized_value.casefold()
                in str(metadata.get(key) or "").casefold()
            ), None)
            if structural_match is not None:
                key, source_text = structural_match
                structural_position = source_text.casefold().find(
                    normalized_value.casefold()
                )
                source_container = f"metadata.{key}"
                source_span = (
                    structural_position,
                    structural_position + len(normalized_value),
                )
        sources[name] = {
            "provenance_type": (
                "deterministic_structural_label"
                if source_container.startswith("metadata.")
                else "deterministic_same_proposition"
            ),
            "source_container": source_container,
            "source_text": source_text,
            "source_span": source_span,
            "confidence": 0.9,
        }
    for role, value in dict(relation_arguments or {}).items():
        position = proposition.casefold().find(str(value).casefold())
        sources[str(role)] = {
            "provenance_type": "deterministic_operator_relation_argument",
            "source_container": "local_proposition",
            "source_text": proposition,
            "source_span": (
                (position, position + len(str(value)))
                if position >= 0 else whole
            ),
            "confidence": 0.9,
            "revisability": "deterministic_revisable",
            "derivation_type": "deterministic_literal_argument",
        }
    return build_evidence_context(
        evidence_unit_id=evidence_id,
        candidate_id=candidate_id,
        source_id=(
            structural_evidence_unit.source_id
            if structural_evidence_unit is not None
            else str(metadata.get("source_id") or "")
        ),
        chunk_id=chunk_id,
        page=page,
        structural_type=(
            structural_evidence_unit.structural_type.value
            if structural_evidence_unit is not None
            else "body"
        ),
        local_proposition=proposition,
        proposition_span=span,
        metadata=metadata,
        evidence_semantics=semantics,
        semantic_sources=sources,
        relation_arguments=relation_arguments,
        structural_evidence_unit=structural_evidence_unit,
    )


def _eligible_hit(metadata: Mapping[str, Any]) -> bool:
    structural = str(
        metadata.get("structural_type")
        or metadata.get("source_unit_type")
        or metadata.get("unit_type")
        or "body"
    ).casefold()
    return structural not in _REJECTED_STRUCTURAL_TYPES


def _numeric(value: str) -> Optional[float]:
    try:
        number = float(str(value).replace(",", ""))
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _frame_metric(frame: SemanticFrame) -> str:
    """Return a concrete request slot; never an anonymous placeholder."""
    if frame.metrics:
        return frame.metrics[0]
    if frame.entities:
        return frame.entities[0]
    return frame.normalized_request[:MAX_FRAME_TEXT]


def _unit_family(value: str) -> str:
    unit = normalize_semantic_text(value).casefold()
    if unit in {"%", "percent", "percentage"}:
        return "percentage"
    if unit in {"mw", "gw", "kw", "mwh", "gwh"}:
        return "capacity"
    if unit in {"million", "billion", "thousand", "אלף", "מיליון", "מיליארד", "שקלים", "דולר", "דולרים"}:
        return "financial"
    if unit in {
        "עובדים", "משרות", "חברות", "יחידות", "לקוחות",
        "רשומות",
        "employee", "employees", "worker", "workers", "job", "jobs",
        "position", "positions", "company", "companies", "unit", "units",
        "customer", "customers",
        "account", "accounts", "record", "records", "facility", "facilities",
        "station", "stations", "site", "sites",
    }:
        return "count"
    return unit or "number"


def _evidence_population(value: str) -> str:
    match = re.search(
        r"\b(?P<en>(?:[A-Za-z0-9'-]+\s+){0,3}(?:population|sample|cohort|subgroup))\b|"
        r"(?P<he>(?:אוכלוסיית?|מדגם|תת.?קבוצה)\s+[\u05d0-\u05ea\"'׳/-]+(?:\s+[\u05d0-\u05ea\"'׳/-]+){0,3})",
        value,
        re.IGNORECASE,
    )
    return normalize_semantic_text(match.group("en") or match.group("he")) if match else ""


def _evidence_facet(value: str) -> str:
    if re.search(r"\b(?:experience|seniority|tenure)\b|(?:ניסיון|וותק|בכירות)", value, re.I):
        return "experience_band"
    if re.search(r"\b(?:education|degree|academic|graduate)\b|(?:השכלה|תואר|אקדמי|בוגר)", value, re.I):
        return "education_level"
    if re.search(r"\b(?:role|position|occupation|job)\b|(?:תפקיד|משרה|מקצוע)", value, re.I):
        return "role"
    return ""


def _clean_entity(value: str) -> str:
    text = normalize_semantic_text(value)
    text = re.sub(
        r"^(?:הוא|היא|הם|הן|is|was|were|for|of|ל|עבור|מסוג)\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s+(?:מתוך|מכלל|of\s+the\s+total).*$", "", text, flags=re.IGNORECASE)
    return text.strip(" \t,;:.()[]{}\"'׳–—-")[:120]


def _category_boundary_valid(entity: str) -> bool:
    """Reject flattened adjacent labels unless a boundary is unambiguous."""
    value = normalize_semantic_text(entity).strip()
    if not value or len(value.split()) > 12 or "|" in value:
        return False
    if "/" not in value:
        return True
    # Hebrew gender inflection such as "מנהל/ת פרויקט" is one lexical role;
    # two full labels separated by a slash are structurally ambiguous.
    return bool(re.match(
        r"^[\u05d0-\u05ea]{2,}/[\u05d0-\u05ea](?:\s+[\u05d0-\u05ea\"'׳-]+){0,4}$",
        value,
    ))


def _entity_is_total(entity: str) -> bool:
    normalized = normalize_semantic_text(entity)
    return bool(re.match(
        r"^(?:סך\s*ה?כול|כלל|סה.?כ|total|overall|all)(?:\b|\s|:|-)",
        normalized,
        re.IGNORECASE,
    )) or bool(re.search(
        r"(?:אומדן|ביקוש|התפלגות|estimate|demand|distribution).{0,80}"
        r"(?:לפי|by\s+(?:type|category|role))",
        normalized,
        re.IGNORECASE,
    ))


def _entity_is_structural_noise(entity: str) -> bool:
    value = normalize_semantic_text(entity)
    return bool(
        re.search(r"(?:\.pdf\b|\bpage\s+\d|\bfigure\s+\d|\btable\s+\d)", value, re.IGNORECASE)
        or (value.count("_") >= 1 and value.count("-") >= 3)
    )


def _local_period(text: str, start: int, end: int, frame: SemanticFrame) -> Optional[str]:
    prefix_start = max(0, start - 160)
    prefix = text[prefix_start:start]
    boundary = max(prefix.rfind("."), prefix.rfind("!"), prefix.rfind("?"), prefix.rfind("׃"))
    if boundary >= 0:
        prefix = prefix[boundary + 1:]
    suffix = text[end:min(len(text), end + 180)]
    suffix_boundaries = [
        position for position in (
            suffix.find("."), suffix.find("!"), suffix.find("?"), suffix.find("׃")
        )
        if position >= 0
    ]
    if suffix_boundaries:
        suffix = suffix[:min(suffix_boundaries) + 1]
    local = f"{prefix} {text[start:end]} {suffix}"
    periods = _extract_periods(local)
    requested = set(frame.periods)
    if periods:
        compatible = [period for period in periods if not requested or period in requested]
        if not compatible:
            return None
        return compatible[0]
    return ""


def _extremum_pairs(text: str) -> list[tuple[str, re.Match[str], str]]:
    """Extract category/value pairs only from explicit local bindings."""
    pairs: list[tuple[str, re.Match[str], str]] = []
    # Relation-first extraction keeps a percentage bound to its category even
    # when the source also gives a parenthetical count before "is/was".  It is
    # generic to prose statistics and selects the closest compatible measure,
    # while excluding a nearby year from the value candidates.
    relation_then_entity = re.compile(
        r"(?:\b(?:is|was|were|accounted\s+for)\b|(?:הוא|היא)\s+ל?)"
        r"\s*[)(\[\]\"'׳]*\s*-?\s*"
        r"(?P<entity>[A-Za-z\u05d0-\u05ea][A-Za-z0-9\u05d0-\u05ea /+_'׳\"-]{1,180}?)"
        r"(?=\s*(?:\(|\[|,|;|\.|%|$|ו-?\s*\d|[-–—]\s))",
        re.IGNORECASE,
    )
    period_spans = tuple(match.span() for match in _PERIOD_RE.finditer(text))
    for relation in relation_then_entity.finditer(text):
        prefix_start = max(0, relation.start() - 220)
        local_numbers = [
            match
            for match in _NUMBER_RE.finditer(text, prefix_start, relation.start())
            if not any(
                match.start() < end and match.end() > start
                for start, end in period_spans
            )
        ]
        if not local_numbers:
            continue
        percentages = [
            match for match in local_numbers
            if _unit_family(match.group("unit") or "") == "percentage"
        ]
        chosen = (percentages or local_numbers)[-1]
        intervening = text[chosen.end():relation.start()]
        if len(intervening) > 170 or intervening.count(".") > 1:
            continue
        entity = _clean_entity(relation.group("entity"))
        if entity and _category_boundary_valid(entity):
            pairs.append((
                entity,
                chosen,
                text[chosen.start():relation.end()],
            ))

    value_then_entity = re.compile(
        r"(?<![\w])(?!(?:19|20)\d{2}(?![\w]))"
        + _NUMBER_RE.pattern
        + r"(?P<middle>[^.!?%\d]{0,96}?)"
        + r"(?:\b(?:is|was|were|for|accounted\s+for)\b|"
        + r"(?:הוא|היא|הם|הן)\s+ל?)\s*-?\s*"
        + r"(?P<entity>[A-Za-z\u05d0-\u05ea][A-Za-z0-9\u05d0-\u05ea /+_'׳\"-]{1,100}?)"
        + r"(?=\s*(?:\(|\[|,|;|\.|%|$|ו-?\s*\d))",
        re.IGNORECASE,
    )
    for match in value_then_entity.finditer(text):
        entity = _clean_entity(match.group("entity"))
        if entity and _category_boundary_valid(entity):
            pairs.append((entity, match, match.group(0)))

    row_pattern = re.compile(
        r"(?:^|[\n;])\s*"
        r"(?P<entity>[A-Za-z\u05d0-\u05ea][A-Za-z0-9\u05d0-\u05ea /+_'׳\"-]{1,100}?)"
        r"\s*(?:\||:|–|—|\t|\s+(?:was|is|עמד(?:ה)?\s+על|היווה|היוותה)\s+)\s*"
        + r"(?<![\w])(?!(?:19|20)\d{2}(?![\w]))"
        + _NUMBER_RE.pattern,
        re.IGNORECASE,
    )
    for match in row_pattern.finditer(text):
        entity = _clean_entity(match.group("entity"))
        if entity and _category_boundary_valid(entity):
            pairs.append((entity, match, match.group(0)))

    # Narrative distributions often express a small category as
    # "demand for <category> (less than 1%)".  Preserve the explicit
    # category/value binding and comparator instead of dropping that row from
    # an otherwise complete matrix.
    for number in _NUMBER_RE.finditer(text):
        prefix = text[max(0, number.start() - 180):number.start()]
        comparator = re.search(
            r"(?:\(\s*(?:(?:less\s+than|under|below|פחות\s+מ)\s*)?-?|"
            r"(?:less\s+than|under|below|פחות\s+מ)\s*-?)\s*$",
            prefix,
            re.IGNORECASE,
        )
        if comparator is None:
            continue
        entity_source = prefix[:comparator.start()]
        relation = re.search(
            r"(?:\b(?:demand|share|rate|count)\b.{0,64}?\b(?:for|of)\s+|"
            r"(?:ביקוש|שיעור|חלק|מספר).{0,64}?(?<![\u05d0-\u05ea])ל(?=[\u05d0-\u05ea]))"
            r"(?P<entity>[A-Za-z\u05d0-\u05ea][A-Za-z0-9\u05d0-\u05ea /+_'׳\"-]{1,80}?)\s*$",
            entity_source,
            re.IGNORECASE,
        )
        if relation is None:
            continue
        entity = _clean_entity(relation.group("entity"))
        if entity and _category_boundary_valid(entity) and not any(
            prior_entity.casefold() == entity.casefold()
            and prior_match.start() == number.start()
            for prior_entity, prior_match, _claim in pairs
        ):
            pairs.append((
                entity,
                number,
                text[relation.start():number.end()],
            ))
    return pairs


def _extremum_claim_window(
    text: str,
    entity: str,
    match: re.Match[str],
) -> tuple[str, int, int]:
    entity_position = text.find(entity, max(0, match.end() - 24))
    if entity_position < 0:
        entity_position = match.end()
    start = max(0, match.start() - 80)
    end = min(len(text), entity_position + len(entity) + 220)
    for boundary in ".!?׃\n":
        left = text.rfind(boundary, start, match.start())
        if left >= 0:
            start = max(start, left + 1)
        right = text.find(boundary, entity_position + len(entity), end)
        if right >= 0:
            end = min(end, right + 1)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return text[start:end], start, end


def _extremum_attribute_facts(
    *,
    text: str,
    entity: str,
    primary_match: re.Match[str],
    frame: SemanticFrame,
    evidence_id: str,
    candidate_id: str,
    chunk_id: str,
    metadata: Mapping[str, Any],
    hit_index: int,
    period: str,
    evidence_facet: str,
    structural_evidence_unit: StructuralEvidenceUnit,
) -> tuple[SemanticFact, ...]:
    claim, claim_start, claim_end = _extremum_claim_window(
        text, entity, primary_match
    )
    entity_position = text.find(
        entity,
        max(0, primary_match.start("value") - 180),
        min(len(text), primary_match.end("value") + 240),
    )
    if entity_position >= 0:
        anchor_start = min(primary_match.start("value"), entity_position)
        anchor_end = max(primary_match.end("value"), entity_position + len(entity))
        left_boundaries = [
            text.rfind(boundary, 0, anchor_start)
            for boundary in (";", ".", "!", "?", "׃", "\n")
        ]
        claim_start = max(left_boundaries) + 1
        right_boundaries = [
            position for position in (
                text.find(boundary, anchor_end)
                for boundary in (";", ".", "!", "?", "׃", "\n")
            ) if position >= 0
        ]
        claim_end = min(right_boundaries) if right_boundaries else len(text)
        next_percentage = next((
            match for match in _NUMBER_RE.finditer(
                text, primary_match.end("value"), claim_end
            )
            if _unit_family(match.group("unit") or "") == "percentage"
        ), None)
        if next_percentage is not None and next_percentage.start() > entity_position:
            claim_end = next_percentage.start()
        while claim_start < claim_end and text[claim_start].isspace():
            claim_start += 1
        while claim_end > claim_start and text[claim_end - 1].isspace():
            claim_end -= 1
        claim = text[claim_start:claim_end]
    output: list[SemanticFact] = []
    primary_value = primary_match.group("value").replace(",", "")
    primary_family = _unit_family(primary_match.group("unit") or "")
    attribute_numbers: tuple[re.Match[str], ...] = ()
    if entity_position >= 0:
        if primary_match.end("value") <= entity_position:
            # Common distribution form: ``33% ... (20 positions) was X``.
            # Only numbers between this category's primary value and label
            # can be attributes of X. Looking back to the sentence start can
            # silently copy the preceding row's count into later categories.
            attribute_ranges = (
                (primary_match.end("value"), entity_position),
                (entity_position + len(entity), claim_end),
            )
        else:
            # Label-first prose can place the attribute on either side of the
            # primary value: ``X: 33% with 20 positions`` or
            # ``X had 20 positions, or 33%``.
            attribute_ranges = (
                (primary_match.end("value"), claim_end),
                (entity_position + len(entity), primary_match.start("value")),
            )
        for attribute_start, attribute_end in attribute_ranges:
            local_counts = tuple(
                match for match in _NUMBER_RE.finditer(
                    text, attribute_start, attribute_end
                )
                if _unit_family(match.group("unit") or "") == "count"
            )
            if local_counts:
                attribute_numbers = local_counts
                break
    for match in attribute_numbers:
        raw_value = match.group("value")
        unit = match.group("unit") or ""
        family = _unit_family(unit)
        if (
            raw_value.replace(",", "") == primary_value
            and family == primary_family
        ):
            continue
        if family != "count":
            continue
        population = _evidence_population(claim)
        context = _operator_evidence_context(
            text=text,
            metadata=metadata,
            evidence_id=evidence_id,
            candidate_id=candidate_id,
            chunk_id=chunk_id,
            proposition=claim,
            span=(claim_start, claim_end),
            subject=entity,
            predicate="has_count",
            target_entity=entity,
            population=population,
            period=period,
            unit=unit or "number",
            numeric_role="count",
            polarity="neutral",
            structural_evidence_unit=structural_evidence_unit,
        )
        fact = SemanticFact(
            entity=entity,
            predicate=SemanticPredicate.VALUE.value,
            metric="count",
            value=raw_value,
            unit=unit,
            period=period,
            population=context.population,
            direction="",
            negated=False,
            qualifiers=("winner_attribute",),
            evidence_id=evidence_id,
            source_claim=claim,
            hit_index=hit_index,
            evidence_subject=entity,
            evidence_predicate="has_count",
            evidence_target_entity=entity,
            evidence_population=context.population,
            evidence_period=period,
            evidence_unit=unit or "number",
            evidence_numeric_role="count",
            evidence_span=(claim_start, claim_end),
            evidence_facet=evidence_facet,
            candidate_id=candidate_id,
            origin_debug_id=f"operator-hit-{hit_index}",
            evidence_context=context,
        )
        if fact.structural_binding_valid:
            output.append(fact)
        break
    return tuple(output)


def _structural_extremum_facts(
    *,
    text: str,
    metadata: Mapping[str, Any],
    evidence_id: str,
    candidate_id: str,
    chunk_id: str,
    structural_evidence_unit: StructuralEvidenceUnit,
    hit_index: int,
    period: str,
    evidence_facet: str,
    requested_percentage: bool,
) -> tuple[SemanticFact, ...]:
    """Bind an extremum row from its source-derived entity/value fields."""
    if structural_evidence_unit.structural_type not in {
        StructuralEvidenceType.TABLE_ROW,
        StructuralEvidenceType.CHART_CATEGORY_RECORD,
    } or not structural_evidence_unit.row_label:
        return ()
    numeric_fields: list[tuple[Any, re.Match[str], str]] = []
    for binding in structural_evidence_unit.field_bindings:
        if binding.role != "value":
            continue
        start, end = binding.source_span
        match = _NUMBER_RE.search(
            text,
            max(0, int(start)),
            min(len(text), max(int(end), int(start)) + 40),
        )
        if match is None or match.start("value") != int(start):
            continue
        family = _unit_family(match.group("unit") or "")
        column = normalize_semantic_text(binding.column_role).casefold()
        if family == "number" and re.search(
            r"(?:שיעור|אחוז|חלק|נתח|percent|percentage|share|rate)",
            column,
            re.IGNORECASE,
        ):
            family = "percentage"
        elif family == "number" and re.search(
            r"(?:מספר|כמות|רשומ|count|number|record)",
            column,
            re.IGNORECASE,
        ):
            family = "count"
        numeric_fields.append((binding, match, family))
    if not numeric_fields:
        return ()
    primary_index = next((
        index for index, (_binding, _match, family) in enumerate(numeric_fields)
        if family == ("percentage" if requested_percentage else "count")
    ), 0)
    entity = structural_evidence_unit.row_label
    output: list[SemanticFact] = []
    for index, (_binding, match, family) in enumerate(numeric_fields):
        if family not in {"percentage", "count"}:
            continue
        raw_unit = match.group("unit") or (
            "%" if family == "percentage" else "number"
        )
        predicate = (
            "has_percentage" if family == "percentage" else "has_count"
        )
        context = _operator_evidence_context(
            text=text,
            metadata=metadata,
            evidence_id=evidence_id,
            candidate_id=candidate_id,
            chunk_id=chunk_id,
            proposition=text,
            span=(0, len(text)),
            subject=entity,
            predicate=predicate,
            target_entity=entity,
            population=_evidence_population(text),
            period=period,
            unit=raw_unit,
            numeric_role=(
                "percentage" if family == "percentage" else "count"
            ),
            polarity="neutral",
            structural_evidence_unit=structural_evidence_unit,
        )
        fact = SemanticFact(
            entity=entity,
            predicate=SemanticPredicate.VALUE.value,
            metric=family,
            value=match.group("value"),
            unit=raw_unit,
            period=period,
            population=context.population,
            direction="",
            negated=False,
            qualifiers=("winner_attribute",) if index != primary_index else (),
            evidence_id=evidence_id,
            source_claim=text,
            hit_index=hit_index,
            evidence_subject=entity,
            evidence_predicate=predicate,
            evidence_target_entity=entity,
            evidence_population=context.population,
            evidence_period=period,
            evidence_unit=raw_unit,
            evidence_numeric_role=(
                "percentage" if family == "percentage" else "count"
            ),
            evidence_span=(match.start(), match.end()),
            evidence_facet=evidence_facet,
            candidate_id=candidate_id,
            origin_debug_id=f"operator-hit-{hit_index}",
            evidence_context=context,
        )
        if fact.structural_binding_valid:
            output.append(fact)
    if primary_index >= len(output):
        return ()
    if primary_index:
        output.insert(0, output.pop(primary_index))
    return tuple(output)


def execute_extremum(
    frame: SemanticFrame,
    hits: Sequence[Mapping[str, Any]],
    *,
    scan_complete: bool,
    execution_contract: Any = None,
) -> OperatorExecution:
    if execution_contract is None and os.getenv(
        "PROJECT_X_CANDIDATE_MODE", "0"
    ).strip().casefold() in {"1", "true", "yes", "on"}:
        raise RuntimeError("extremum execution requires RequestExecutionContract")
    if frame.operation not in {RequestOperator.ARGMAX, RequestOperator.ARGMIN}:
        raise ValueError("execute_extremum requires ARGMAX or ARGMIN")
    requested_facets = {
        str(getattr(item, "facet", "") or "")
        for item in getattr(execution_contract, "requirements", ())
        if getattr(item, "facet", "")
    }
    requested_percentage = bool(re.search(
        r"(?:%|אחוז|שיעור|חלק|נתח|percent|percentage|share|rate)",
        " ".join(frame.metrics) + " " + frame.normalized_request,
        re.IGNORECASE,
    ))
    facts: list[SemanticFact] = []
    seen: set[tuple[str, str, str]] = set()
    structural_inputs = (
        (hit_index, *unit)
        for hit_index, hit in enumerate(hits)
        for unit in _operator_structural_units(hit, hit_index)
    )
    for (
        hit_index, text, metadata, evidence_id, candidate_id, chunk_id,
        structural_evidence_unit,
    ) in structural_inputs:
        if not text or not _eligible_hit(metadata):
            continue
        evidence_facet = _evidence_facet(
            " ".join((
                str(metadata.get("section") or metadata.get("title") or ""),
                text,
            ))
        )
        if requested_facets and evidence_facet not in requested_facets:
            continue
        period = _local_period(text, 0, len(text), frame)
        if period is not None:
            structural_facts = _structural_extremum_facts(
                text=text,
                metadata=metadata,
                evidence_id=evidence_id,
                candidate_id=candidate_id,
                chunk_id=chunk_id,
                structural_evidence_unit=structural_evidence_unit,
                hit_index=hit_index,
                period=period,
                evidence_facet=evidence_facet,
                requested_percentage=requested_percentage,
            )
            if structural_facts:
                primary_fact = structural_facts[0]
                key = (
                    primary_fact.entity.casefold(),
                    _unit_family(primary_fact.unit),
                    primary_fact.period,
                )
                if key not in seen:
                    seen.add(key)
                    facts.extend(structural_facts)
                continue
        for entity, match, claim in _extremum_pairs(text):
            raw_value = match.group("value")
            unit = match.group("unit") or ""
            period = _local_period(text, match.start(), match.end(), frame)
            if (
                period is None
                or _entity_is_total(entity)
                or _entity_is_structural_noise(entity)
            ):
                continue
            key = (entity.casefold(), _unit_family(unit), period)
            if key in seen:
                continue
            seen.add(key)
            local_prefix = text[max(0, match.start() - 28):match.start()]
            qualifiers = tuple(
                qualifier for qualifier, present in (
                    ("approximate", bool(match.group("approx"))),
                    ("upper_bound", bool(re.search(
                        r"(?:\bless\s+than\b|\bunder\b|פחות\s+מ)\s*-?\s*$",
                        local_prefix,
                        re.IGNORECASE,
                    ))),
                ) if present
            )
            complete_claim, _claim_start, _claim_end = _extremum_claim_window(
                text, entity, match
            )
            evidence_predicate = (
                "has_percentage"
                if _unit_family(unit) == "percentage" else "has_count"
            )
            population = _evidence_population(complete_claim or claim)
            context = _operator_evidence_context(
                text=text,
                metadata=metadata,
                evidence_id=evidence_id,
                candidate_id=candidate_id,
                chunk_id=chunk_id,
                proposition=complete_claim or claim,
                span=(_claim_start, _claim_end),
                subject=entity,
                predicate=evidence_predicate,
                target_entity=entity,
                population=population,
                period=period,
                unit=unit or "number",
                numeric_role=(
                    "percentage" if _unit_family(unit) == "percentage" else "count"
                ),
                polarity="neutral",
                structural_evidence_unit=structural_evidence_unit,
            )
            primary_fact = SemanticFact(
                entity=entity,
                predicate=SemanticPredicate.VALUE.value,
                metric=_unit_family(unit or "number"),
                value=raw_value,
                unit=unit or "number",
                period=period,
                population=context.population,
                direction="",
                negated=False,
                qualifiers=qualifiers,
                evidence_id=evidence_id,
                source_claim=complete_claim or claim,
                hit_index=hit_index,
                evidence_subject=entity,
                evidence_predicate=context.evidence_predicate,
                evidence_target_entity=entity,
                evidence_population=context.population,
                evidence_period=period,
                evidence_unit=unit or "number",
                evidence_numeric_role=(
                    "percentage"
                    if _unit_family(unit) == "percentage" else "count"
                ),
                evidence_span=(_claim_start, _claim_end),
                evidence_facet=evidence_facet,
                candidate_id=candidate_id,
                origin_debug_id=f"operator-hit-{hit_index}",
                evidence_context=context,
            )
            if not primary_fact.structural_binding_valid:
                continue
            facts.append(primary_fact)
            facts.extend(_extremum_attribute_facts(
                text=text,
                entity=entity,
                primary_match=match,
                frame=frame,
                evidence_id=evidence_id,
                candidate_id=candidate_id,
                chunk_id=chunk_id,
                metadata=metadata,
                hit_index=hit_index,
                period=period,
                evidence_facet=evidence_facet,
                structural_evidence_unit=structural_evidence_unit,
            ))
            if len(facts) >= MAX_OPERATOR_FACTS:
                break

    groups: dict[tuple[str, str, str, str], list[SemanticFact]] = {}
    for fact in facts:
        groups.setdefault((
            fact.metric,
            _unit_family(fact.unit),
            fact.period,
            fact.population,
        ), []).append(fact)
    compatible_groups = [
        (key, values)
        for key, values in groups.items()
        if len({fact.entity.casefold() for fact in values}) >= 2
    ]
    compositional_percentage = bool(re.search(
        r"(?:\b(?:share|part|portion)\b|(?:חלק|נתח)\s+|"
        r"שיעור.{0,28}ביקוש|(?:הגדול(?:ה)?|הקטנ(?:ה)?)\s+ביותר\s+בביקוש)",
        frame.normalized_request,
        re.IGNORECASE,
    ))
    if requested_percentage and compositional_percentage:
        compatible_groups = [
            (key, values) for key, values in compatible_groups
            if key[1] != "percentage" or 95.0 <= sum(
                number for number in (_numeric(fact.value) for fact in values)
                if number is not None
            ) <= 105.0
        ]
    compatible_groups.sort(
        key=lambda item: (
            int(requested_percentage and item[0][1] == "percentage"),
            len({fact.entity.casefold() for fact in item[1]}),
        ),
        reverse=True,
    )
    if not scan_complete:
        return OperatorExecution(frame.operation, False, tuple(facts), None, (), "candidate_scan_incomplete", False)
    if not compatible_groups:
        return OperatorExecution(frame.operation, False, tuple(facts), None, (), "complete_compatible_candidate_set_missing", True)
    selected_facts = compatible_groups[0][1]
    valued = [(fact, _numeric(fact.value)) for fact in selected_facts]
    valued = [(fact, number) for fact, number in valued if number is not None]
    if len(valued) < 2:
        return OperatorExecution(frame.operation, False, tuple(selected_facts), None, (), "candidate_values_invalid", True)
    winner = (max if frame.operation is RequestOperator.ARGMAX else min)(valued, key=lambda item: item[1])[0]
    derivation_payload = "\x1f".join((
        frame.operation.value,
        str(getattr(execution_contract, "contract_id", "")),
        "|".join(sorted(fact.fact_id for fact in selected_facts)),
        winner.fact_id,
    ))
    derivation_id = "derivation-" + hashlib.sha256(
        derivation_payload.encode("utf-8")
    ).hexdigest()[:24]
    source_evidence_ids = tuple(dict.fromkeys(
        fact.evidence_id for fact in selected_facts
    ))
    source_candidate_ids = tuple(dict.fromkeys(
        fact.candidate_id for fact in selected_facts if fact.candidate_id
    ))
    source_fact_ids = tuple(dict.fromkeys(
        fact.fact_id for fact in selected_facts
    ))
    result = replace(
        winner,
        fact_id="",
        predicate=(
            SemanticPredicate.MAXIMUM.value
            if frame.operation is RequestOperator.ARGMAX
            else SemanticPredicate.MINIMUM.value
        ),
        source_evidence_unit_ids=source_evidence_ids,
        source_candidate_ids=source_candidate_ids,
        source_typed_fact_ids=source_fact_ids,
        operator_derivation_id=derivation_id,
        derivation_description=(
            f"{frame.operation.value} over {len(selected_facts)} compatible "
            "category values with common metric, unit, period, and population"
        ),
    )
    winner_attributes = [result]
    seen_attributes = {(_unit_family(result.unit), result.value)}
    for fact in facts:
        family = _unit_family(fact.unit)
        key = (family, fact.value)
        if (
            fact.entity.casefold() == winner.entity.casefold()
            and fact.hit_index == winner.hit_index
            and family in {"percentage", "count"}
            and key not in seen_attributes
        ):
            seen_attributes.add(key)
            winner_attributes.append(replace(
                fact,
                fact_id="",
                source_typed_fact_ids=(fact.fact_id,),
                operator_derivation_id=derivation_id,
                derivation_description=(
                    "Requested winner attribute bound to the winning category "
                    "and its source typed fact"
                ),
            ))
    requested_count = bool(re.search(
        r"(?:\b(?:count|number)\b|(?:מספר|כמות|היקפ))",
        frame.normalized_request,
        re.IGNORECASE,
    ))
    has_percentage_attribute = any(
        _unit_family(fact.unit) == "percentage" for fact in winner_attributes
    )
    has_count_attribute = any(
        _unit_family(fact.unit) == "count" for fact in winner_attributes
    )
    if (
        requested_percentage and not has_percentage_attribute
        or requested_count and not has_count_attribute
    ):
        return OperatorExecution(
            frame.operation, False, tuple(selected_facts), None, (),
            "requested_winner_attributes_missing", True,
        )
    candidate_order = tuple(
        fact for fact, _number in sorted(
            valued,
            key=lambda item: item[1],
            reverse=frame.operation is RequestOperator.ARGMAX,
        )
    )
    selected_entities = {fact.entity.casefold() for fact in selected_facts}
    selected_hit_indexes = {fact.hit_index for fact in selected_facts}
    accepted_facts = tuple(
        fact for fact in facts
        if fact.entity.casefold() in selected_entities
        and fact.hit_index in selected_hit_indexes
        and fact.period == winner.period
    )
    evidence_indexes = tuple(dict.fromkeys(
        fact.hit_index for fact in accepted_facts if fact.hit_index >= 0
    ))
    return OperatorExecution(
        frame.operation,
        True,
        accepted_facts,
        result,
        evidence_indexes,
        "none",
        True,
        result_attributes=tuple(winner_attributes),
        candidate_order=candidate_order,
    )


def _filter_comparison(operator: str, value: float, threshold: float) -> bool:
    return {
        "greater_than": value > threshold,
        "greater_than_or_equal": value >= threshold,
        "less_than": value < threshold,
        "less_than_or_equal": value <= threshold,
        "before": value < threshold,
        "after": value > threshold,
    }.get(operator, False)


def execute_filter_rows(
    frame: SemanticFrame,
    hits: Sequence[Mapping[str, Any]],
    *,
    scan_complete: bool,
    execution_contract: Any = None,
) -> OperatorExecution:
    """Evaluate a typed threshold and return evidence-bound period/value rows."""
    if execution_contract is None:
        raise RuntimeError("filter execution requires RequestExecutionContract")
    if frame.operation is not RequestOperator.FILTER:
        raise ValueError("execute_filter_rows requires FILTER")
    conditions = tuple(getattr(execution_contract, "filter_conditions", ()) or ())
    if len(conditions) != 1:
        return OperatorExecution(
            frame.operation, False, (), None, (),
            "filter_condition_unresolved", scan_complete,
        )
    condition = conditions[0]
    threshold = _numeric(str(getattr(condition, "threshold", "")))
    if threshold is None:
        return OperatorExecution(
            frame.operation, False, (), None, (),
            "filter_threshold_invalid", scan_complete,
        )
    requested_metric_tokens = set(_TOKEN_RE.findall(
        normalize_semantic_text(str(getattr(condition, "metric", ""))).casefold()
    ))
    expected_family = _unit_family(str(getattr(condition, "unit", "") or ""))
    source_facts: list[SemanticFact] = []
    matching: list[SemanticFact] = []
    structural_inputs = (
        (hit_index, *unit)
        for hit_index, hit in enumerate(hits)
        for unit in _operator_structural_units(hit, hit_index)
    )
    for (
        hit_index, text, metadata, evidence_id, candidate_id, chunk_id,
        structural_evidence_unit,
    ) in structural_inputs:
        if not text or not _eligible_hit(metadata):
            continue
        cursor = 0
        for raw_segment in re.split(r"(?<=[.!?׃])\s+|\n+|;\s*", text):
            start = text.find(raw_segment, cursor)
            if start < 0:
                start = text.find(raw_segment)
            start = max(0, start)
            cursor = start + len(raw_segment)
            end = start + len(raw_segment)
            while start < end and text[start].isspace():
                start += 1
            while end > start and text[end - 1].isspace():
                end -= 1
            proposition = text[start:end]
            if not proposition:
                continue
            periods = _extract_periods(proposition)
            if len(periods) != 1:
                continue
            period_spans = tuple(match.span() for match in _PERIOD_RE.finditer(proposition))
            candidates = [
                match for match in _NUMBER_RE.finditer(proposition)
                if not any(
                    match.start() < end and match.end() > begin
                    for begin, end in period_spans
                )
                and (
                    not getattr(condition, "unit", "")
                    or _unit_family(match.group("unit") or "") == expected_family
                )
            ]
            if len(candidates) != 1:
                continue
            match = candidates[0]
            numeric_value = _numeric(match.group("value"))
            if numeric_value is None:
                continue
            structural_label = normalize_semantic_text(next((
                str(metadata.get(key) or "")
                for key in (
                    "row_header", "category", "table_title", "chart_title",
                    "section_title", "section", "title",
                )
                if str(metadata.get(key) or "")
            ), ""))
            evidence_subject = structural_label or normalize_semantic_text(
                _PERIOD_RE.sub(" ", _NUMBER_RE.sub(" ", proposition))
            ).strip(" ,;:.|-–—")
            evidence_tokens = set(_TOKEN_RE.findall(
                normalize_semantic_text(f"{evidence_subject} {proposition}").casefold()
            ))
            if requested_metric_tokens and not requested_metric_tokens & evidence_tokens:
                continue
            unit = str(match.group("unit") or getattr(condition, "unit", "") or "number")
            numeric_role = (
                "percentage" if _unit_family(unit) == "percentage" else "count"
                if _unit_family(unit) == "count" else "requested_answer_value"
            )
            evidence_predicate = (
                "has_percentage" if numeric_role == "percentage"
                else "has_count" if numeric_role == "count"
                else "stated_as"
            )
            span = (start, end)
            context = _operator_evidence_context(
                text=text,
                metadata=metadata,
                evidence_id=evidence_id,
                candidate_id=candidate_id,
                chunk_id=chunk_id,
                proposition=proposition,
                span=span,
                subject=evidence_subject,
                predicate=evidence_predicate,
                target_entity=evidence_subject,
                population=_evidence_population(proposition),
                period=periods[0],
                unit=unit,
                numeric_role=numeric_role,
                polarity="neutral",
                structural_evidence_unit=structural_evidence_unit,
            )
            fact = SemanticFact(
                entity=evidence_subject,
                predicate=SemanticPredicate.VALUE.value,
                metric=normalize_semantic_text(
                    str(getattr(condition, "metric", "") or evidence_subject)
                ),
                value=match.group("value").replace(",", ""),
                unit=unit,
                period=context.period,
                population=context.population,
                direction="",
                negated=False,
                qualifiers=("filter_input",),
                evidence_id=evidence_id,
                source_claim=proposition,
                hit_index=hit_index,
                evidence_subject=context.evidence_subject,
                evidence_predicate=context.evidence_predicate,
                evidence_target_entity=context.evidence_target_entity,
                evidence_population=context.population,
                evidence_period=context.period,
                evidence_unit=context.evidence_unit,
                evidence_numeric_role=context.evidence_numeric_role,
                evidence_span=span,
                candidate_id=candidate_id,
                origin_debug_id=f"operator-hit-{hit_index}",
                evidence_context=context,
            )
            if not fact.structural_binding_valid:
                continue
            source_facts.append(fact)
            if _filter_comparison(condition.operator, numeric_value, threshold):
                matching.append(fact)
    if not scan_complete:
        return OperatorExecution(
            frame.operation, False, tuple(source_facts), None, (),
            "candidate_scan_incomplete", False,
        )
    if not matching:
        return OperatorExecution(
            frame.operation, False, tuple(source_facts), None, (),
            "matching_filter_rows_missing", True,
        )
    matching.sort(key=lambda fact: fact.period)
    derivation_id = "derivation-" + hashlib.sha256("\x1f".join((
        frame.operation.value,
        str(getattr(execution_contract, "contract_id", "")),
        condition.operator,
        condition.threshold,
        "|".join(fact.fact_id for fact in matching),
    )).encode("utf-8")).hexdigest()[:24]
    source_evidence_ids = tuple(dict.fromkeys(fact.evidence_id for fact in matching))
    source_candidate_ids = tuple(dict.fromkeys(
        fact.candidate_id for fact in matching if fact.candidate_id
    ))
    source_fact_ids = tuple(fact.fact_id for fact in matching)
    derived_rows = tuple(
        replace(
            fact,
            fact_id="",
            predicate="filter_rows",
            qualifiers=("filter_result_row",),
            source_evidence_unit_ids=source_evidence_ids,
            source_candidate_ids=source_candidate_ids,
            source_typed_fact_ids=(fact.fact_id,),
            operator_derivation_id=derivation_id,
            derivation_description=(
                f"Row value satisfies {condition.operator} threshold "
                f"{condition.threshold} {condition.unit}".strip()
            ),
        )
        for fact in matching
    )
    return OperatorExecution(
        frame.operation,
        True,
        tuple(source_facts),
        derived_rows[0],
        tuple(dict.fromkeys(fact.hit_index for fact in matching)),
        "none",
        True,
        result_attributes=derived_rows,
        candidate_order=derived_rows,
    )


_NEGATIVE_RELATIONS = (
    (SemanticPredicate.UNDERREPRESENTED.value, re.compile(
        r"(?:\b(?:underrepresented|under-represented|underrepresentation\s+of)\b|"
        r"(?:י|יי)צוג\s+חסר(?:\s+של)?)",
        re.IGNORECASE,
    )),
    (SemanticPredicate.OUT_OF_SCOPE.value, re.compile(
        r"(?:\b(?:outside|out\s+of)\s+(?:the\s+)?scope\b|"
        r"מחוץ\s+ל(?:תחום|אוכלוסי|אומדן))",
        re.IGNORECASE,
    )),
    (SemanticPredicate.ABSENT.value, re.compile(
        r"(?:\b(?:was|were|is|are)\s+absent\b|\babsence\s+of\b|"
        r"(?<![\u05d0-\u05ea])נעדר(?:ה|ו)?(?:\s+מן|\s+מה))",
        re.IGNORECASE,
    )),
    (SemanticPredicate.NOT_INCLUDED.value, re.compile(
        r"(?:\b(?:does\s+not|did\s+not|do\s+not|not)\s+include(?:d)?\b|"
        r"\b(?:excluded|omitted|wasn['’]?t\s+included)\b|"
        r"(?<![\u05d0-\u05ea])(?:לא|אינ(?:ו|ה|ם|ן)|איננו|אינה)"
        r"(?![\u05d0-\u05ea])\s+(?:גם\s+)?(?:כוללת|כולל|כללו|"
        r"נכללו|נכללה|נכלל)(?![\u05d0-\u05ea]))",
        re.IGNORECASE,
    )),
    (SemanticPredicate.NOT_MEASURED.value, re.compile(
        r"(?:\b(?:not|wasn['’]?|weren['’]?t)\s+(?:measured|estimated|surveyed)\b|"
        r"(?<![\u05d0-\u05ea])(?:לא|אינ(?:ו|ה|ם|ן))(?![\u05d0-\u05ea])"
        r"\s+(?:נמדד|נמדדה|נמדדו|נאמד|נאמדה|נסקר|נסקרה))",
        re.IGNORECASE,
    )),
    (SemanticPredicate.NOT_AVAILABLE.value, re.compile(
        r"(?:\b(?:not|un)available\b|(?<![\u05d0-\u05ea])"
        r"(?:לא|אינ(?:ו|ה|ם|ן))(?![\u05d0-\u05ea])\s+(?:זמין|זמינה|קיים|קיימת))",
        re.IGNORECASE,
    )),
)


def _sentences(text: str) -> list[str]:
    return [
        normalize_semantic_text(item)
        for item in re.split(r"(?<=[.!?׃])\s+|\n{2,}", text)
        if len(normalize_semantic_text(item)) >= 6
    ]


def _query_numbers(value: str) -> tuple[str, ...]:
    return _unique(match.group("value").replace(",", "") for match in _NUMBER_RE.finditer(value))


def _contains_number(text: str, compact: str) -> bool:
    return any(match.group("value").replace(",", "") == compact for match in _NUMBER_RE.finditer(text))


def _excluded_entity(sentence: str, relation: re.Match[str]) -> str:
    relation_text = normalize_semantic_text(relation.group(0)).casefold()
    passive_relation = bool(re.search(
        r"\b(?:excluded|omitted|absent|outside|unavailable|"
        r"wasn['’]?t\s+included|not\s+(?:included|measured|estimated|surveyed))\b|"
        r"(?:לא\s+(?:נכלל|נכללה|נכללו|נמדד|נמדדה|נמדדו|נאמד|נאמדה|נסקר|נסקרה)|"
        r"מחוץ\s+ל|נעדר|אינו\s+זמין|אינה\s+זמינה)",
        relation_text,
        re.IGNORECASE,
    ))
    if passive_relation and relation.start() > 0:
        prefix = sentence[:relation.start()]
        prefix = re.sub(
            r"(?:\b(?:was|were|is|are|has\s+been|had\s+been)\b|"
            r"(?:היה|הייתה|היו))\s*$",
            " ",
            prefix,
            flags=re.IGNORECASE,
        )
        prefix = re.sub(r"^\s*(?:the|a|an)\s+", "", prefix, flags=re.I)
        candidate = _clean_entity(prefix)
        if candidate:
            return candidate
    tail = sentence[relation.end():]
    tail = re.sub(
        r"^(?:\s+(?:גם|את|של|the|any|of|from|in|ב|את\s+ה))\s+",
        " ", tail, flags=re.IGNORECASE,
    )
    entity = re.split(
        r"(?:(?:,|;|\.)\s+|\s+\(|\s+(?:because|since|כאשר|משום|מפני|אך)\s+|"
        r"\s+(?:המדגם|the\s+sample)\s+(?:כולל|included|contains)\b)",
        tail,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    return _clean_entity(entity)


def execute_exclusion(
    frame: SemanticFrame,
    hits: Sequence[Mapping[str, Any]],
    *,
    scan_complete: bool,
    execution_contract: Any = None,
) -> OperatorExecution:
    if execution_contract is None and os.getenv(
        "PROJECT_X_CANDIDATE_MODE", "0"
    ).strip().casefold() in {"1", "true", "yes", "on"}:
        raise RuntimeError("exclusion execution requires RequestExecutionContract")
    if frame.operation not in {RequestOperator.EXCLUSION, RequestOperator.EXCEPTION}:
        raise ValueError("execute_exclusion requires EXCLUSION or EXCEPTION")
    anchors = _query_numbers(frame.normalized_request)
    underrepresentation_requested = bool(re.search(
        r"(?:\b(?:underrepresented|under-represented|underrepresentation)\b|"
        r"(?:י|יי)צוג\s+חסר)",
        frame.normalized_request,
        re.IGNORECASE,
    ))
    ranked: list[tuple[int, int, SemanticFact]] = []
    structural_inputs: list[tuple[
        int, str, Mapping[str, Any], str, str, str, StructuralEvidenceUnit
    ]] = []
    for hit_index, hit in enumerate(hits):
        raw_source_text = str(hit.get("document") or "")
        records = _operator_structural_units(hit, hit_index)
        for unit_index, record in enumerate(records):
            (
                text, metadata, evidence_id, candidate_id, chunk_id,
                structural_evidence_unit,
            ) = record
            if not text or not _eligible_hit(metadata):
                continue
            if anchors and not any(
                _contains_number(text, anchor) for anchor in anchors
            ) and unit_index > 0:
                previous = records[unit_index - 1][5]
                if any(
                    _contains_number(previous.local_text, anchor)
                    for anchor in anchors
                ):
                    composite = bounded_adjacent_structural_unit(
                        raw_source_text,
                        (previous, structural_evidence_unit),
                    )
                    if composite is not None:
                        structural_evidence_unit = composite
                        text = composite.local_text
                        evidence_id = composite.unit_id
            structural_inputs.append((
                hit_index, text, metadata, evidence_id, candidate_id,
                chunk_id, structural_evidence_unit,
            ))
    for (
        hit_index, text, metadata, evidence_id, candidate_id, chunk_id,
        structural_evidence_unit,
    ) in structural_inputs:
        sentences = _sentences(text)
        anchor_positions = [
            match.start()
            for match in _NUMBER_RE.finditer(text)
            if match.group("value").replace(",", "") in anchors
        ]
        sentence_cursor = 0
        for sentence_index, sentence in enumerate(sentences):
            sentence_start = text.find(sentence, sentence_cursor)
            if sentence_start < 0:
                sentence_start = text.find(sentence)
            sentence_start = max(0, sentence_start)
            sentence_cursor = sentence_start + len(sentence)
            for predicate, pattern in _NEGATIVE_RELATIONS:
                relation = pattern.search(sentence)
                if relation is None:
                    continue
                if (
                    underrepresentation_requested
                    and predicate != SemanticPredicate.UNDERREPRESENTED.value
                ):
                    continue
                structural_entity = normalize_semantic_text(
                    structural_evidence_unit.row_label
                    or next((
                        binding.value
                        for binding in structural_evidence_unit.field_bindings
                        if binding.role == "entity"
                    ), "")
                )
                entity = structural_entity or _excluded_entity(
                    sentence, relation
                )
                if len(entity) < 2:
                    continue
                subject_matches = list(re.finditer(
                    r"(?:נתון|אומדן|estimate|figure|population|אוכלוסי)",
                    sentence[:relation.start()],
                    re.IGNORECASE,
                ))
                claim_start = subject_matches[-1].start() if subject_matches else 0
                # Markdown ingestion can serialize a governing structural
                # label as ``[source > section] proposition``.  The sentence
                # normalizer deliberately drops the leading ``[``, so using
                # the whole normalized sentence as the evidence proposition
                # leaves a stray ``]`` and makes an otherwise exact quote
                # structurally invalid.  Start after only that recognized
                # structural prefix; ordinary bracketed prose is preserved.
                structural_close = sentence.rfind("]", 0, relation.start())
                if (
                    structural_close >= claim_start
                    and ">" in sentence[claim_start:structural_close]
                ):
                    claim_start = structural_close + 1
                entity_position = sentence.find(entity, relation.end())
                claim_end = (
                    entity_position + len(entity)
                    if entity_position >= 0
                    else len(sentence)
                )
                direct_claim = normalize_semantic_text(
                    sentence[claim_start:claim_end]
                ).strip(" ,;:.()[]{}")
                context = sentence
                if anchor_positions and not any(
                    _contains_number(sentence, anchor) for anchor in anchors
                ):
                    preceding = [
                        item for item in sentences[max(0, sentence_index - 2):sentence_index]
                        if any(_contains_number(item, anchor) for anchor in anchors)
                    ]
                    if preceding:
                        context = f"{preceding[-1]} {sentence}"
                evidence_scope = _evidence_population(context)
                if not evidence_scope and anchors:
                    anchor_match = next((
                        match for match in _NUMBER_RE.finditer(context)
                        if match.group("value").replace(",", "") in anchors
                    ), None)
                    if anchor_match is not None:
                        evidence_scope = normalize_semantic_text(
                            anchor_match.group(0)
                        )
                if (
                    structural_evidence_unit.segmentation_method
                    == "bounded_adjacent_reconstruction"
                    or structural_evidence_unit.structural_type in {
                        StructuralEvidenceType.TABLE_ROW,
                        StructuralEvidenceType.LIST_ITEM,
                        StructuralEvidenceType.CHART_CATEGORY_RECORD,
                    }
                ):
                    context = text
                    claim_text = text
                    claim_span = (0, len(text))
                else:
                    claim_text = direct_claim or sentence
                    exact_start = text.find(claim_text, sentence_start)
                    if exact_start < 0:
                        claim_text = text[
                            sentence_start:sentence_start + len(sentence)
                        ]
                        exact_start = sentence_start
                    claim_span = (
                        exact_start,
                        exact_start + len(claim_text),
                    )
                evidence_period = (
                    _extract_periods(claim_text)[0]
                    if _extract_periods(claim_text) else ""
                )
                entity_after_relation = sentence.find(
                    entity, relation.end()
                ) >= 0
                if entity_after_relation:
                    excluded_scope = normalize_semantic_text(
                        sentence[claim_start:relation.start()]
                    ).strip(" ,;:.()[]{}")
                    excluded_scope = re.sub(
                        r"\s+(?:does|did|do)\s*$", "", excluded_scope,
                        flags=re.IGNORECASE,
                    ).strip()
                else:
                    scope_tail = sentence[relation.end():claim_end]
                    scope_tail = re.sub(
                        r"^\s*(?:from|in|of|מ|ב|מן|מה)\s*", "", scope_tail,
                        flags=re.IGNORECASE,
                    )
                    excluded_scope = normalize_semantic_text(
                        scope_tail
                    ).strip(" ,;:.()[]{}")
                if not excluded_scope:
                    excluded_scope = evidence_scope
                relation_arguments = {
                    "excluded_entity": entity,
                    "excluded_scope": excluded_scope,
                }
                evidence_context = _operator_evidence_context(
                    text=text,
                    metadata=metadata,
                    evidence_id=evidence_id,
                    candidate_id=candidate_id,
                    chunk_id=chunk_id,
                    proposition=claim_text,
                    span=claim_span,
                    subject=entity,
                    predicate="excluded_from",
                    target_entity=entity,
                    population=evidence_scope,
                    period=evidence_period,
                    unit="text",
                    numeric_role="text_proposition",
                    polarity="negative",
                    relation_arguments=relation_arguments,
                    structural_evidence_unit=structural_evidence_unit,
                )
                fact = SemanticFact(
                    entity=entity,
                    predicate=predicate,
                    metric=predicate,
                    value="",
                    unit="",
                    period=evidence_context.period,
                    population=evidence_context.population,
                    direction="",
                    negated=True,
                    qualifiers=("negated",),
                    evidence_id=evidence_id,
                    source_claim=claim_text,
                    hit_index=hit_index,
                    evidence_subject=entity,
                    evidence_predicate="excluded_from",
                    evidence_target_entity=entity,
                    evidence_population=evidence_context.population,
                    evidence_period=evidence_context.period,
                    evidence_unit="text",
                    evidence_numeric_role="text_proposition",
                    evidence_span=claim_span,
                    candidate_id=candidate_id,
                    origin_debug_id=f"operator-hit-{hit_index}",
                    evidence_context=evidence_context,
                    relation_arguments=evidence_context.relation_arguments,
                )
                if not fact.structural_binding_valid:
                    continue
                direct_subject = int(bool(re.search(
                    r"(?:^|[.!?׃]\s+)\s*(?:נתון|אומדן|estimate|figure|population|אוכלוסי)",
                    sentence,
                    re.IGNORECASE,
                )))
                relation_position = sentence_start + relation.start()
                anchor_proximity = (
                    max(0, 4000 - min(
                        abs(relation_position - position)
                        for position in anchor_positions
                    ))
                    if anchor_positions
                    else 0
                )
                ranked.append((
                    direct_subject * 10_000 + anchor_proximity,
                    -len(entity),
                    fact,
                ))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    facts = tuple(item[2] for item in ranked[:MAX_OPERATOR_FACTS])
    if not scan_complete:
        return OperatorExecution(frame.operation, False, facts, None, (), "candidate_scan_incomplete", False)
    if not facts:
        return OperatorExecution(frame.operation, False, (), None, (), "negative_relation_missing", True)
    source = facts[0]
    derivation_payload = "\x1f".join((
        frame.operation.value,
        str(getattr(execution_contract, "contract_id", "")),
        source.fact_id,
    ))
    derivation_id = "derivation-" + hashlib.sha256(
        derivation_payload.encode("utf-8")
    ).hexdigest()[:24]
    result = replace(
        source,
        fact_id="",
        source_typed_fact_ids=(source.fact_id,),
        operator_derivation_id=derivation_id,
        derivation_description=(
            f"{frame.operation.value} selected an explicit negative relation "
            "from one evidence-derived source fact"
        ),
    )
    return OperatorExecution(
        frame.operation,
        True,
        facts,
        result,
        (result.hit_index,),
        "none",
        True,
    )


def material_fact_signature(value: str) -> tuple[tuple[str, ...], tuple[str, ...], bool]:
    """Return a bounded fact signature for previous-answer verification."""
    text = normalize_semantic_text(re.sub(r"\[\d{1,3}\]", " ", value)).casefold()
    periods = _extract_periods(text)
    period_spans = tuple(match.span() for match in _PERIOD_RE.finditer(text))
    numbers = tuple(sorted(_unique(
        f"{match.group('value').replace(',', '')}:{_unit_family(match.group('unit') or '')}"
        for match in _NUMBER_RE.finditer(text)
        if not any(
            match.start() < end and match.end() > start
            for start, end in period_spans
        )
    )))
    negated = bool(re.search(r"(?:\b(?:not|no|without)\b|(?:לא|אין|אינ(?:ו|ה|ם|ן)))", text))
    return numbers, periods, negated


def previous_answer_confirmed(previous_answer: str, new_answer: str) -> bool:
    old_numbers, old_periods, old_negated = material_fact_signature(previous_answer)
    new_numbers, new_periods, new_negated = material_fact_signature(new_answer)
    if old_numbers:
        return old_numbers == new_numbers and old_periods == new_periods and old_negated == new_negated
    old_terms = {
        token for token in _TOKEN_RE.findall(normalize_semantic_text(previous_answer).casefold())
        if len(token) >= 3
    }
    new_terms = {
        token for token in _TOKEN_RE.findall(normalize_semantic_text(new_answer).casefold())
        if len(token) >= 3
    }
    overlap = len(old_terms & new_terms) / max(1, min(len(old_terms), len(new_terms)))
    return overlap >= 0.72 and old_negated == new_negated


def render_operator_answer(
    execution: OperatorExecution,
    *,
    language: str,
    citation_ids: Sequence[str],
    bound_facts: Sequence[Any] = (),
) -> str:
    # Rendering is deliberately unable to inspect the operator's hidden
    # matrix/result values.  Only facts already linked to canonical output
    # requirements may cross this boundary.
    facts = tuple(bound_facts)
    if not execution.accepted or not facts:
        return ""
    markers = " ".join(f"[{value}]" for value in citation_ids)
    if execution.operation in {RequestOperator.ARGMAX, RequestOperator.ARGMIN}:
        fact = next((
            item for item in facts
            if str(getattr(item, "requirement_id", "")).endswith("-winner")
        ), facts[0])
        attributes = facts
        percentage_fact = next(
            (
                item for item in attributes
                if str(getattr(item, "numeric_role", "")) == "percentage"
            ),
            None,
        )
        count_fact = next(
            (
                item for item in attributes
                if str(getattr(item, "numeric_role", "")) == "count"
            ),
            None,
        )
        requested_percentage = percentage_fact is not None
        requested_count = bool(count_fact)
        rendered_values: list[str] = []
        if requested_percentage and percentage_fact is not None:
            rendered_percentage = (
                f"{percentage_fact.value}{percentage_fact.unit}"
                if percentage_fact.unit == "%"
                else f"{percentage_fact.value} {percentage_fact.unit}".strip()
            )
            if "upper_bound" in percentage_fact.qualifiers:
                rendered_percentage = (
                    f"פחות מ-{rendered_percentage}"
                    if language == "he"
                    else f"less than {rendered_percentage}"
                )
            rendered_values.append(rendered_percentage)
        if requested_count and count_fact is not None:
            rendered_values.append(
                f"{count_fact.value} {count_fact.unit}".strip()
            )
        if not rendered_values:
            value_text = str(getattr(fact, "value", ""))
            unit_text = str(getattr(fact, "unit", ""))
            if not value_text:
                return ""
            rendered_values.append(
                f"{value_text}{unit_text}"
                if unit_text == "%"
                else f"{value_text} {unit_text}".strip()
            )
        value = " ו-".join(rendered_values) if language == "he" else " and ".join(rendered_values)
        if language == "he":
            direction = "הגדול ביותר" if execution.operation is RequestOperator.ARGMAX else "הקטן ביותר"
            return f"הקטגוריה בעלת הערך {direction} היא {getattr(fact, 'subject', '')}, עם {value}. {markers}".strip()
        direction = "highest" if execution.operation is RequestOperator.ARGMAX else "lowest"
        return f"The category with the {direction} value is {getattr(fact, 'subject', '')}, at {value}. {markers}".strip()
    if execution.operation in {RequestOperator.EXCLUSION, RequestOperator.EXCEPTION}:
        fact = facts[0]
        if language == "he":
            relation = {
                SemanticPredicate.NOT_MEASURED.value: "לא נמדדה",
                SemanticPredicate.NOT_AVAILABLE.value: "לא הייתה זמינה",
                SemanticPredicate.ABSENT.value: "נעדרה",
                SemanticPredicate.OUT_OF_SCOPE.value: "הייתה מחוץ לתחום",
                SemanticPredicate.UNDERREPRESENTED.value: "יוצגה בחסר",
            }.get(str(getattr(fact, "predicate", "")), "לא נכללה")
            return f"הקבוצה {getattr(fact, 'subject', '')} {relation} באומדן ובתחום המתואר במקור. {markers}".strip()
        relation = {
            SemanticPredicate.NOT_MEASURED.value: "was not measured",
            SemanticPredicate.NOT_AVAILABLE.value: "was not available",
            SemanticPredicate.ABSENT.value: "was absent",
            SemanticPredicate.OUT_OF_SCOPE.value: "was outside the scope",
            SemanticPredicate.UNDERREPRESENTED.value: "was underrepresented",
        }.get(str(getattr(fact, "predicate", "")), "was not included")
        return f"The group {getattr(fact, 'subject', '')} {relation} in the estimate and source scope. {markers}".strip()
    if execution.operation is RequestOperator.FILTER:
        rows = next((
            tuple(getattr(fact, "filter_rows", ()) or ())
            for fact in facts if getattr(fact, "filter_rows", ())
        ), ())
        if not rows:
            return ""
        rendered = [
            (
                f"{period}: {value}{unit}"
                if unit == "%" else f"{period}: {value} {unit}".strip()
            )
            for period, value, unit in rows
        ]
        joined = "; ".join(rendered)
        if language == "he":
            return f"השורות העומדות בתנאי הן: {joined}. {markers}".strip()
        return f"The rows meeting the condition are: {joined}. {markers}".strip()
    return ""
