"""Adaptive per-subquestion retrieval and explicit evidence coverage.

The module contains no tenant- or document-specific facts. It accepts a
tenant-safe retriever and produces traceable evidence units and coverage
requirements before answer generation.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field, replace
from enum import Enum
from types import SimpleNamespace
import hashlib
import json
import math
import os
import re
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

from evidence_quotes import (
    complete_evidence_sentences,
    normalize_evidence_whitespace,
    opaque_evidence_chunk_id,
)
from evidence_context import (
    EvidenceAtom,
    EvidenceContext,
    FieldProvenance,
    EvidenceSemanticResolverResult,
    EvidenceSemanticValidationError,
    NumericBindingProvenance,
    RELATION_ARGUMENT_SCHEMAS,
    RelationArgument,
    apply_model_assisted_semantics,
    build_evidence_context,
    evidence_atoms_for_context,
    relation_argument_validation_errors,
    structural_binding_validation_errors,
)
from answer_units import GROUPED_DECIMAL_LITERAL_PATTERN, atomic_fact_spans
from structural_evidence import (
    StructuralEvidenceType,
    StructuralEvidenceUnit,
    account_scalar_spans,
    canonical_structural_entry,
    bounded_adjacent_structural_unit,
    ordered_list_item_text,
)
from source_authorization import SourceAuthorization
from proposition_quality import balanced_delimiters
from request_understanding import (
    AnswerShape,
    QuestionType,
    RequestIntent,
    RequestPlan,
    Subquestion,
    normalize_request_text,
)
from eval_trace import current as current_eval_trace
from semantic_contracts import (
    RequestExecutionContract,
    assess_candidate,
    RequirementConstraintState,
    RequirementSatisfactionMode,
    AggregationRole,
    QuantityKind,
    QuantitySemantics,
    StatisticalRole,
    canonical_entity_equivalent,
    candidate_id as semantic_candidate_id,
    classify_numeric_role,
    classify_quantity_semantics,
    classify_scalar_role,
    experience_band_compatible,
    experience_band_facets,
    source_fact_facets,
    source_population_scope,
    source_population_matches_scope,
    source_facet_compatible,
    source_endpoint_digest,
    compatible_source_endpoints,
    source_endpoint_order,
    DOCUMENT_ROLE_PREDICATES,
    document_role_target_compatible,
    experience_scope_populations,
    explicit_metric_concepts,
    infer_request_scalar_role,
    infer_request_quantity_semantics,
    numeric_fact_binding_errors,
    normalize_scalar_role,
    normalize_aggregation_role,
    normalize_quantity_kind,
    normalize_statistical_role,
    periods_semantically_compatible,
    population_scope_compatibility,
    requirement_dimension_constraint,
    proposition_relation_supported,
    scalar_role_has_local_cue,
    scalar_role_compatibility,
    quantity_dimension_compatibility,
    quantity_semantics_from_scalar_role,
    scalar_role_from_quantity_semantics,
    _source_metric_label_establishes_cardinality,
    segment_text_propositions,
    select_semantic_candidates,
    text_fact_binding_errors,
)


MAX_QUERY_REWRITES = 6
MAX_CHANNEL_RESULTS = 24
MAX_FUSED_RESULTS = 32
MAX_EVIDENCE_PER_SUBQUESTION = 12
MAX_VECTOR_RESULTS_PER_SUBQUESTION = 8
MAX_VECTOR_CANDIDATES_PER_SUBQUESTION = 16
MAX_VECTOR_RERANK_PER_SUBQUESTION = 8
MAX_STRUCTURAL_ANCHORS = 4
MAX_STRUCTURAL_RECORDS = 128
MAX_STRUCTURAL_SCAN = 256
MAX_STRUCTURAL_SIBLINGS = 8
MAX_MULTI_HOPS = 1
MAX_ALLOWED_CLAIMS_PER_UNIT = 12
MAX_ALLOWED_CLAIM_CHARS = 900

_TOKEN_RE = re.compile(r"[A-Za-z0-9\u05d0-\u05ea]{2,}")
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_BULLET_RE = re.compile(r"(?m)^\s*(?:[-*•▪◦]|\d{1,2}[.)])\s+(.+)$")
_ROW_SEPARATOR_RE = re.compile(r"(?:\s+[-–—]\s+|:\s+)")
_BREADCRUMB_RE = re.compile(
    r"\A(?:[ \t]*\[[^\]\r\n]{1,500}\][ \t]*(?:\r?\n|$))+"
)
_SENTENCE_OR_BULLET_BREAK_RE = re.compile(
    r"(?<=[.!?׃])\s+(?=\S)|\s+(?=[•▪◦]\s*)"
)
_DOTTED_LEADER_RE = re.compile(r"(?:\.{4,}|…{2,})")
_NAVIGATION_HEADING_RE = re.compile(
    r"(?:\b(?:table\s+of\s+contents|contents|list\s+of\s+"
    r"(?:figures|tables|charts))\b|"
    r"(?:תוכן\s+עניינים|רשימת\s+(?:איורים|טבלאות|תרשימים)))",
    re.IGNORECASE,
)
_CAPTION_START_RE = re.compile(
    r"^\s*(?:\d{1,3}\s+)?(?:figure|fig\.?|table|chart|איור|טבלה|תרשים)"
    r"\s*\d{0,3}\b",
    re.IGNORECASE,
)
_SOURCE_CAPTION_RE = re.compile(r"^\s*(?:source|מקור)\s*:", re.IGNORECASE)
_REFERENCE_ENTRY_START_RE = re.compile(
    r"(?:https?://|\bdoi\.org/|"
    r"^\s*\d{1,3}\s+[A-Z][A-Za-z'’-]+,\s*[A-Z](?:\.|,))",
    re.IGNORECASE,
)
_HEADING_PREFIX_RE = re.compile(r"^\s*#{1,6}\s+")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$|(?:\S+\s{2,}){2,}\S+")
_FORMULA_TOKEN_RE = re.compile(
    r"(?:χ|χχ|[=<>]\s*[A-Za-zα-ωΑ-Ω]|\bp\s*[<=>]|\bpp\s*[<=>])",
    re.IGNORECASE,
)
_PREDICATE_RE = re.compile(
    r"(?:\b(?:is|are|was|were|stood|amounted|estimated|reported|represented|"
    r"increased|decreased|rose|fell|grew|declined|changed|remained)\b|"
    r"(?:הוא|היא|היו|היה|הייתה|עמד|עמדה|נאמד|נאמדה|מהווה|היווה|"
    r"עלה|עלתה|ירד|ירדה|גדל|גדלה|פחת|פחתה|נמצא|נמצאה|חל))",
    re.IGNORECASE,
)
_CURRENT_TIME_RE = re.compile(
    r"(?:\b(?:current|present|latest)\b|"
    r"\bthis\s+(?:survey|period|year|quarter)\b|"
    r"(?:[הב]סקר\s+הנוכחי|בעת\s+(?:ביצוע\s+)?הסקר|השנה\s+הנוכחית|"
    r"התקופה\s+הנוכחית|נוכחי(?:ת|ים|ות)?))",
    re.IGNORECASE,
)
_HISTORICAL_TIME_RE = re.compile(
    r"(?:\b(?:previous|prior|earlier|baseline)\s+"
    r"(?:survey|period|year|quarter)\b|"
    r"(?:[הב]סקר\s+(?:הקודם|הקודמת)|בתקופה\s+הקודמת|בשנה\s+הקודמת))",
    re.IGNORECASE,
)
_VALUE_REQUEST_RE = re.compile(
    r"(?:"
    r"^\s*(?:ו)?כמה\b|"
    r"^\s*(?:מהו?|מהי|מה\s+(?:היה|הייתה|היו))\b.{0,180}"
    r"(?:אומדן|מספר|כמות|שיעור|אחוז|ערך|חסם|רמת\s+ביטחון)|"
    r"^\s*(?:באיזו|איזו)\s+רמת\s+ביטחון\b|"
    r"^\s*(?:and\s+)?(?:how\s+many|how\s+much)\b|"
    r"^\s*(?:what|which|at\s+what)\b.{0,180}"
    r"(?:estimate|number|count|amount|rate|percentage|percent|value|bound|"
    r"confidence\s+level)"
    r")",
    re.IGNORECASE,
)
_APPROX_RE = re.compile(
    r"(?:\b(?:approximately|about|around|roughly|nearly|almost)\b|"
    r"(?:בקירוב|בערך|כמעט|משוער)|"
    r"(?<![\u05d0-\u05ea])כ\s*[-־]?\s*$)",
    re.IGNORECASE,
)
_COUNT_UNIT_RE = re.compile(
    r"(?:\b(?:employees?|workers?|positions?|jobs?|vacancies|people|persons?|"
    r"units?|cases?|alerts?|respondents?|customers?|users?|companies|organizations|"
    r"sites?|labels?|inspections?|nodes?|records?|items?|dollars?|euros?|shekels?)\b|"
    r"(?:עובדים?|משרות?|מועסקים?|אנשים?|יחידות?|מקרים?|משיבים?|לקוחות?|"
    r"משתמשים?|חברות?|ארגונים?|שקלים?))",
    re.IGNORECASE,
)
_TOTAL_METRIC_RE = re.compile(
    r"(?:\b(?:total|overall|count|number|volume|amount)\b|"
    r"(?:סך\s*הכול|סה[\"״']?כ|כולל|כלל|מספר|כמות))",
    re.IGNORECASE,
)
_TOTAL_AGGREGATION_CUE_RE = re.compile(
    r"(?:\b(?:total|overall|all|number\s+of|count\s+of)\b|"
    r"(?:(?:מספר|כמות)\s+(?:ה)?[\u05d0-\u05ea]+)|"
    r"(?:סה.?כ|סך\s+הכל|כלל))",
    re.IGNORECASE,
)
_COUNT_METRIC_RE = re.compile(
    r"(?:\b(?:count|number|positions?|jobs?|vacancies|employees?|workers?|"
    r"people|persons?|units?|cases?|alerts?|respondents?|customers?|users?|demand|"
    r"companies|organizations)\b|"
    r"(?:מספר|כמות|משרות?|עובדים?|מועסקים?|אנשים?|יחידות?|מקרים?|"
    r"משיבים?|לקוחות?|משתמשים?|ביקוש|חברות?|ארגונים?))",
    re.IGNORECASE,
)
_DISTRIBUTION_DIMENSION_RE = re.compile(
    r"(?:\b(?:distribution|profile|mix|bands?|ranges?|age\s+groups?)\b|"
    r"(?:התפלגות|פרופיל|תמהיל|טווחים|קבוצות\s+גיל))",
    re.IGNORECASE,
)
_HIGHER_BAND_RE = re.compile(
    r"(?:\b(?:higher|upper|above|over|more\s+than|at\s+least|experienced|senior)\b|"
    r"(?:מעל|יותר\s+מ|לפחות|גבוה(?:ה|ים|ות)?|מנוס(?:ה|ים|ות)|"
    r"בכיר(?:ה|ים|ות)?))",
    re.IGNORECASE,
)
_LOWER_BAND_RE = re.compile(
    r"(?:\b(?:lower|below|under|less\s+than|up\s+to|at\s+most|"
    r"entry[- ]level|early[- ]career|junior)\b|"
    r"(?:עד|פחות\s+מ|לכל\s+היותר|נמו(?:ך|כה|כים|כות)|"
    r"בתחילת\s+דרכ|מתחיל(?:ה|ים|ות)|זוטר(?:ה|ים|ות)?))",
    re.IGNORECASE,
)
_CHANGE_DIRECTION_RE = re.compile(
    r"(?:"
    r"\b(?:increase|increased|rise|rose|growth|grew|decrease|decreased|"
    r"decline|declined|fall|fell|reduction)\b|"
    r"(?:עלייה|עליה|עלה|עלתה|גידול|גדל|גדלה|ירידה|ירד|ירדה|קיטון|פחת|פחתה)"
    r")",
    re.IGNORECASE,
)
_TIME_UNIT_RE = re.compile(
    r"\b(?:years?|months?|weeks?|days?|hours?|minutes?|seconds?)\b|"
    r"(?:שנות?|שנים|חודשים?|שבועות?|ימים?|שעות?|דקות?|שניות?)",
    re.IGNORECASE,
)
_EXPERIENCE_FACET_RE = re.compile(
    r"(?:\b(?:experience|seniority|tenure|years?\s+of\s+experience)\b|"
    r"(?:ניסיון|וותק|בכירות))",
    re.IGNORECASE,
)
_EDUCATION_FACET_RE = re.compile(
    r"(?:\b(?:education|degree|degrees|academic|graduate|graduates)\b|"
    r"(?:השכלה|תואר|תארים|בוגר|בוגרי|אקדמי))",
    re.IGNORECASE,
)
_FORECAST_FACET_RE = re.compile(
    r"(?:\b(?:forecast|projection|projected|future|outlook)\b|"
    r"(?:תחזית|צפי|עתידי|בעתיד))",
    re.IGNORECASE,
)
_SCALAR_MEASURE_MODIFIERS = (
    (
        "lower_bound",
        re.compile(
            r"(?:\b(?:lower\s+bound|confidence\s+interval\s+floor)\b|"
            r"ה?(?:חסם|גבול)\s+ה?תחתון)",
            re.IGNORECASE,
        ),
    ),
    (
        "upper_bound",
        re.compile(
            r"(?:\b(?:upper\s+bound|confidence\s+interval\s+ceiling)\b|"
            r"ה?(?:חסם|גבול)\s+ה?עליון)",
            re.IGNORECASE,
        ),
    ),
    (
        "confidence",
        re.compile(
            r"(?:\bconfidence\s+(?:level|interval)\b|רמת\s+ב[י]?טחון)",
            re.IGNORECASE,
        ),
    ),
    (
        "median",
        re.compile(r"(?:\bmedian\b|חציון)", re.IGNORECASE),
    ),
    (
        "mean",
        re.compile(r"(?:\b(?:mean|average)\b|ממוצע)", re.IGNORECASE),
    ),
)
_METRIC_RELATION_RE = re.compile(
    r"(?:"
    r"[:=]|"
    r"\b(?:is|are|was|were|stood|amounted|represented|represents|"
    r"for|of)\b|"
    r"(?:הוא|היא|היו|היה|הייתה|עמד|עמדה|מהווה|נאמד|נאמדה|"
    r"ל(?=[\u05d0-\u05ea]))"
    r")",
    re.IGNORECASE,
)
_ENUMERATION_CUE_RE = re.compile(
    r"[,;]|(?:\b(?:and|or|others?|including|such\s+as)\b|"
    r"(?:ו(?:אחרים|אחרות)|אחרים|אחרות|כולל|כגון))",
    re.IGNORECASE,
)
_CLAUSE_BOUNDARY_RE = re.compile(
    r"[.;׃]|,\s*(?:\b(?:when|while|whereas|which)\b|כאשר|ואילו)|"
    r"\b(?:subsequently|afterwards)\b|"
    r"(?:בהמשך|לאחר\s+מכן)(?=\s+(?:התבקש|נשאל))",
    re.IGNORECASE,
)
_REVERSED_ADJACENT_DELIMITER_RE = re.compile(r"\)\s*\(")
_MALFORMED_ADJACENT_GROUPED_NUMBERS_RE = re.compile(
    r"\d{1,3}(?:,\d{3})+,\d{1,3}(?:,\d{3})+"
)
_UNLABELLED_PARALLEL_NUMERIC_RE = re.compile(
    r"(?:"
    r":\s*[-+]?\d+(?:[.,]\d+)?(?:\s*(?:%|[A-Za-z\u05d0-\u05ea]+))?"
    r"\s+(?:and|or|או)\s+[-+]?\d+(?:[.,]\d+)?|"
    r"(?<![\w])(?!\d{4}\b)[-+]?\d+(?:[.,]\d+)?\s+(?:and|or|או)\s+"
    r"[-+]?\d+(?:[.,]\d+)?\s*(?:%|[A-Za-z\u05d0-\u05ea]+)\b"
    r")",
    re.IGNORECASE,
)
_SUBGROUP_SCOPE_RE = re.compile(
    r"(?:"
    r"\b(?:among|within)\b|"
    r"\bfor\s+(?:the\s+)?(?:group|segment|subset)\b|"
    r"\b(?:companies|organizations|workers|employees|respondents|people)"
    r"(?:\s+\w+){0,3}\s+(?:with|who|that|employ(?:s|ed|ing))\b|"
    r"(?:בקבוצת|בקרב|בפלח|בתת[־-]?קבוצה|"
    r"חברות(?:\s+[\u05d0-\u05ea]+){0,3}\s+ש?מעסיקות|"
    r"עובדים\s+בעלי|משיבים\s+ש)"
    r")",
    re.IGNORECASE,
)
_NONRESPONSE_CATEGORY_RE = re.compile(
    r"(?:\b(?:did\s+not\s+(?:answer|respond)|no\s+(?:answer|response))\b|"
    r"לא\s+השיבו)",
    re.IGNORECASE,
)
_POSITIVE_RESPONSE_METRIC_RE = re.compile(
    r"(?:\b(?:reported|indicated|stated|said|answered\s+yes)\b|"
    r"(?:ציינו|השיבו|אמרו)\s+ש|שקיימת)",
    re.IGNORECASE,
)
_NEGATED_RESPONSE_METRIC_RE = re.compile(
    r"(?:\b(?:did\s+not|didn't|not)\s+(?:report|indicate|state|say|answer)\b|"
    r"לא\s+(?:ציינו|השיבו|אמרו))",
    re.IGNORECASE,
)
_ANSWER_OPTION_RE = re.compile(
    r"(?<!\w)(?:(?P<positive>yes|כן)|(?P<negative>no|לא))(?!\w)",
    re.IGNORECASE,
)
_GENERIC_METRIC_TERMS = frozenset({
    "amount", "at", "average", "bound", "change", "changed", "confidence",
    "count", "current", "demand", "during",
    "employees", "estimate", "estimated", "in", "jobs", "mean", "number",
    "level", "lower", "mean", "median", "percent", "percentage",
    "positions", "present", "rate", "survey", "total", "upper", "value",
    "was", "were", "workers",
    "אומדן", "אחוז", "אחוזים", "בעת", "ביקוש", "במספר", "הביקוש", "היה",
    "הייתה", "היתה", "היו", "הנוכחי", "הסקר", "החסם", "התחתון",
    "העליון", "כמות", "כלל", "מספר", "ממוצע", "נאמד", "נוכחי", "עובד",
    "עובדים", "ערך", "סקר", "רמת", "ביטחון", "בטחון", "חסם", "תחתון",
    "עליון", "שיעור",
})
_NUMBER_WITH_OPTIONAL_UNIT_RE = re.compile(
    r"(?<![\w])(?P<value>"
    rf"(?:{GROUPED_DECIMAL_LITERAL_PATTERN}|\d+(?:\.\d+)?)"
    rf"(?:\s*[-–]\s*(?:{GROUPED_DECIMAL_LITERAL_PATTERN}|\d+(?:\.\d+)?))?"
    r")"
    r"\s*(?P<percent>%|percent(?:age)?|אחוז(?:ים)?)?",
    re.IGNORECASE,
)
_HEBREW_PREFIXES = frozenset("בכלמהוש")
_GENERIC_STRUCTURED_ITEM_IDENTITIES = frozenset({
    "another",
    "example",
    "examples",
    "for example",
    "for instance",
    "item",
    "other",
    "role",
    "such as",
    "אחר",
    "אחרת",
    "דוגמא",
    "דוגמה",
    "כגון",
    "כמו למשל",
    "לדוגמה",
    "למשל",
    "פריט",
    "תפקיד",
})

_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "between", "both", "by",
    "did", "do", "does", "each", "for", "from", "give", "how", "in", "is",
    "list", "main", "of", "on", "or", "the", "their", "to", "what", "which",
    "with", "according", "document", "report",
    "או", "איזה", "אילו", "איך", "בין", "גם", "האם", "הוא", "היא", "הן",
    "המסמך", "המחקר", "העיקרית", "וכן", "כל", "כמה", "כיצד", "לפי", "מה",
    "מהם", "מי", "מן", "מבחינת", "של", "תן", "לכל", "אחד",
})

# Deterministic, non-factual terminology expansion. Groups add search
# alternatives only; they are never exposed to generation as source evidence.
_SYNONYM_GROUPS = (
    ("worker", "workers", "employee", "employees", "עובד", "עובדים"),
    (
        "artificial intelligence", "ai", "machine intelligence",
        "בינה מלאכותית", "אינטליגנציה מלאכותית",
    ),
    (
        "demand", "positions", "jobs", "vacancies", "workforce", "staffing",
        "ביקוש", "דרישה", "משרות", "תפקידים", "כוח אדם", "עובדים",
    ),
    (
        "role", "roles", "job", "jobs", "title", "titles", "position",
        "תפקיד", "תפקידים", "משרה", "משרות", "סוג תפקיד",
    ),
    (
        "occupation", "occupations", "profession", "professions",
        "עיסוק", "עיסוקים", "מקצוע", "מקצועות",
    ),
    (
        "responsibility", "responsibilities", "function", "functions",
        "duty", "duties", "accountability", "אחריות", "תפקוד", "פונקציה",
        "חובה", "תחום אחריות",
    ),
    (
        "experience", "years of experience", "seniority", "tenure",
        "ניסיון", "שנות ניסיון", "וותק", "בכירות",
    ),
    (
        "help", "assist", "support",
        "לסייע", "לעזור", "לתמוך",
    ),
    (
        "active", "operating", "operate", "operates",
        "פעיל", "פעילה", "פעילים", "פעילות",
        "פועל", "פועלת", "פועלים", "פועלות",
    ),
)
_LIST_CATEGORY_SYNONYM_GROUPS = (
    (
        "role", "roles", "job", "jobs", "title", "titles", "position",
        "positions", "תפקיד", "תפקידים", "משרה", "משרות", "סוג תפקיד",
    ),
    (
        "occupation", "occupations", "profession", "professions",
        "עיסוק", "עיסוקים", "מקצוע", "מקצועות",
    ),
)
_PURPOSE_ACTION_QUERY_RE = re.compile(
    r"(?:\b(?:help|assist|support)\b|(?:לסייע|לעזור|לתמוך))",
    re.IGNORECASE,
)
_PURPOSE_ACTION_BOILERPLATE_RE = re.compile(
    r"(?<![\w])(?:"
    r"should|supposed|intended|the|use|using|usage|results?|do|doing|"
    r"אמור|אמורה|אמורים|אמורות|השימוש|שימוש|בתוצאות|תוצאות|"
    r"לעשות|ביצוע"
    r")(?![\w])",
    re.IGNORECASE,
)


class RetrievalChannel(str, Enum):
    VECTOR = "vector"
    LEXICAL = "lexical"
    EXACT = "exact"
    CONJUNCTIVE = "conjunctive"
    ADJACENT = "adjacent"
    SECTION = "section"
    PARENT_LIST = "parent_list"
    MULTI_HOP = "multi_hop"


class StructuralUnitType(str, Enum):
    BODY = "body"
    HEADING = "heading"
    BULLET = "bullet"
    LIST = "list"
    TABLE_ROW = "table_row"
    CHART_DATA = "chart_data"
    CAPTION = "caption"
    TABLE_OF_CONTENTS = "table_of_contents"
    LIST_OF_FIGURES = "list_of_figures"
    LIST_OF_TABLES = "list_of_tables"
    HEADER_FOOTER = "header_footer"
    PAGE_NUMBER = "page_number"
    FORMULA = "formula"
    MALFORMED = "malformed"


class CoverageStatus(str, Enum):
    COVERED = "covered"
    PARTIAL = "partial"
    MISSING = "missing"


class AbstentionReason(str, Enum):
    NONE = "none"
    NO_AUTHORIZED_EVIDENCE = "no_authorized_evidence"
    PARTIAL_EVIDENCE_ONLY = "partial_evidence_only"
    REQUESTED_COUNT_NOT_COVERED = "requested_count_not_covered"
    MISSING_ATTRIBUTE_COVERAGE = "missing_attribute_coverage"
    REFERENCE_RESOLUTION_FAILED = "reference_resolution_failed"
    RETRIEVAL_TIMEOUT = "retrieval_timeout"
    PLANNER_SCHEMA_FAILURE = "planner_schema_failure"
    GROUNDING_FAILURE = "grounding_failure"
    SOURCE_INTEGRITY_FAILURE = "source_integrity_failure"
    NO_DISTINCT_RESULT_REMAINING = "no_distinct_result_remaining"
    CONTEXT_BUDGET_EXCEEDED = "context_budget_exceeded"


@dataclass(frozen=True)
class QueryRewrite:
    text: str
    category: str


@dataclass(frozen=True)
class EvidenceUnit:
    id: str
    subquestion_ids: tuple[str, ...]
    source_id: str
    page: Optional[int]
    section_fingerprint: str
    canonical_text: str
    allowed_claims: tuple[str, ...]
    channels: tuple[RetrievalChannel, ...]
    score: Optional[float]
    score_kind: Optional[str]
    structural_type: str
    raw_hit: Mapping[str, Any] = field(repr=False, compare=False)
    candidate_id: str = ""
    chunk_id: str = ""
    evidence_context: Optional[EvidenceContext] = field(
        default=None, repr=False, compare=False
    )
    structural_evidence_unit: Optional[StructuralEvidenceUnit] = field(
        default=None, repr=False, compare=False
    )
    source_evidence_unit_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.source_evidence_unit_ids:
            object.__setattr__(self, "source_evidence_unit_ids", (self.id,))
        unit = self.structural_evidence_unit
        if unit is None:
            return
        # A structural evidence unit has already passed the source boundary;
        # it is therefore the one authority for the unit type everywhere
        # downstream of retrieval.
        object.__setattr__(self, "structural_type", unit.structural_type.value)

    def safe_summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subquestion_ids": list(self.subquestion_ids),
            "source_id": self.source_id,
            "page": self.page,
            "channels": [value.value for value in self.channels],
            "structural_type": self.structural_type,
            "candidate_id": self.candidate_id,
            "chunk_id": self.chunk_id,
            "source_evidence_unit_ids": list(self.source_evidence_unit_ids),
            "evidence_context": (
                self.evidence_context.as_dict() if self.evidence_context else None
            ),
            "structural_evidence_unit": (
                self.structural_evidence_unit.safe_summary()
                if self.structural_evidence_unit else None
            ),
        }


@dataclass(frozen=True)
class TypedFact:
    # Requirement ownership is assigned only after a source fact has been
    # constructed.  A blank value therefore represents a canonical
    # SOURCE_FACT, rather than an incomplete requirement-shaped placeholder.
    requirement_id: str
    subquestion_id: str
    subject: str
    metric: str
    facet: str
    period: str
    time_role: str
    value: str
    unit: str
    direction: str
    qualifiers: tuple[str, ...]
    evidence_id: str
    source_claim: str = field(repr=False, compare=False)
    # The exact proposition remains the claim/citation source.  This bounded
    # parent claim is source-only semantic context for later compatibility.
    source_container_claim: str = field(default="", repr=False, compare=False)
    requirement_ids: tuple[str, ...] = ()
    origin_requirement_id: str = ""
    numeric_role: str = "requested_answer_value"
    # Legacy projection; new source compatibility uses the dimensions below.
    scalar_role: str = "not_applicable"
    quantity_kind: str = "unspecified"
    statistical_role: str = "unspecified"
    aggregation_role: str = "unspecified"
    fact_id: str = ""
    structural_type: str = "body"
    row_column_bound: bool = True
    row_header: str = ""
    column_header: str = ""
    evidence_span: tuple[int, int] = (0, 0)
    evidence_span_sha256: str = ""
    population: str = ""
    association_quality: int = field(default=0, repr=False, compare=False)
    evidence_subject: str = ""
    evidence_predicate: str = ""
    evidence_target_entity: str = ""
    evidence_population: str = ""
    evidence_period: str = ""
    evidence_unit: str = ""
    evidence_numeric_role: str = ""
    evidence_polarity: str = "neutral"
    candidate_id: str = ""
    source_evidence_unit_ids: tuple[str, ...] = ()
    source_candidate_ids: tuple[str, ...] = ()
    source_typed_fact_ids: tuple[str, ...] = ()
    source_typed_fact_sha256s: tuple[str, ...] = ()
    operator_derivation_id: str = ""
    derivation_description: str = ""
    evidence_context: Optional[EvidenceContext] = field(
        default=None, repr=False, compare=False
    )
    model_assisted_semantics: bool = False
    structural_binding_valid: bool = True
    structural_validation_errors: tuple[str, ...] = ()
    evidence_atom: Optional[EvidenceAtom] = field(
        default=None, repr=False, compare=False
    )
    numeric_binding: Optional[NumericBindingProvenance] = field(
        default=None, repr=False, compare=False
    )
    fact_type: str = "numeric"
    source_fact_id: str = ""
    # This is the source-local occurrence identity for a SOURCE_FACT.  It is
    # intentionally distinct from the structural evidence unit that happened
    # to expose the occurrence: one bounded source binding can be observed
    # through several structural views without becoming several facts.
    source_fact_occurrence_id: str = ""
    # Stable, source-local support for ``source_fact_occurrence_id``.  Unlike
    # ``evidence_id`` and ``source_evidence_unit_ids``, this is not the
    # structural view that first observed the fact.  It makes the
    # authoritative provenance inspectable without allowing first-observer
    # order to define it.
    source_fact_occurrence_provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        requirement_ids = tuple(dict.fromkeys(
            self.requirement_ids
            if self.requirement_ids
            else ((self.requirement_id,) if self.requirement_id else ())
        ))
        object.__setattr__(self, "requirement_ids", requirement_ids)
        legacy = quantity_semantics_from_scalar_role(
            self.scalar_role or self.numeric_role
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
                (
                    self.source_fact_occurrence_id
                    if self.source_fact_occurrence_id else self.evidence_id
                ),
                (
                    ""
                    if self.source_fact_occurrence_id else
                    self.source_evidence_unit_ids and "|".join(
                        self.source_evidence_unit_ids
                    ) or ""
                ), self.subject, self.metric, self.facet, self.period,
                self.value, self.unit, self.evidence_predicate,
                self.evidence_target_entity, self.evidence_population,
                self.evidence_polarity, self.scalar_role,
                self.quantity_kind, self.statistical_role,
                self.aggregation_role,
                self.evidence_span_sha256,
            ))
            object.__setattr__(
                self,
                "fact_id",
                "fact-" + hashlib.sha256(
                    payload.encode("utf-8")
                ).hexdigest()[:24],
            )
        if not self.source_fact_id:
            object.__setattr__(self, "source_fact_id", self.fact_id)
        if not self.source_evidence_unit_ids:
            object.__setattr__(self, "source_evidence_unit_ids", (self.evidence_id,))
        if self.evidence_atom is not None:
            if self.evidence_atom.evidence_unit_id != self.evidence_id:
                structural_errors = tuple(self.structural_validation_errors) + (
                    "evidence_atom_unit_mismatch",
                )
                object.__setattr__(self, "structural_validation_errors", structural_errors)
            atom_scalar_role = normalize_scalar_role(
                getattr(self.evidence_atom, "scalar_role", "")
            )
            if (
                atom_scalar_role != "not_applicable"
                and atom_scalar_role != self.scalar_role
            ):
                structural_errors = tuple(self.structural_validation_errors) + (
                    "evidence_atom_scalar_role_mismatch",
                )
                object.__setattr__(self, "structural_validation_errors", structural_errors)
            for field_name, error_code in (
                ("quantity_kind", "evidence_atom_quantity_kind_mismatch"),
                ("statistical_role", "evidence_atom_statistical_role_mismatch"),
                ("aggregation_role", "evidence_atom_aggregation_role_mismatch"),
            ):
                atom_value = str(getattr(self.evidence_atom, field_name, "") or "")
                fact_value = str(getattr(self, field_name, "") or "")
                if atom_value and atom_value != fact_value:
                    structural_errors = tuple(self.structural_validation_errors) + (
                        error_code,
                    )
                    object.__setattr__(self, "structural_validation_errors", structural_errors)
            atom_binding = self.evidence_atom.numeric_binding
            if self.numeric_binding is None and atom_binding is not None:
                object.__setattr__(self, "numeric_binding", atom_binding)
            elif (
                self.numeric_binding is not None
                and atom_binding is not None
                and self.numeric_binding != atom_binding
            ):
                structural_errors = tuple(self.structural_validation_errors) + (
                    "numeric_binding_provenance_mismatch",
                )
                object.__setattr__(self, "structural_validation_errors", structural_errors)
            if (
                self.numeric_binding is not None
                and self.numeric_binding.scalar_role
                and normalize_scalar_role(self.numeric_binding.scalar_role)
                != self.scalar_role
            ):
                structural_errors = tuple(self.structural_validation_errors) + (
                    "numeric_binding_scalar_role_mismatch",
                )
                object.__setattr__(self, "structural_validation_errors", structural_errors)
            if self.numeric_binding is not None:
                for field_name, error_code in (
                    ("quantity_kind", "numeric_binding_quantity_kind_mismatch"),
                    ("statistical_role", "numeric_binding_statistical_role_mismatch"),
                    ("aggregation_role", "numeric_binding_aggregation_role_mismatch"),
                ):
                    bound_value = str(getattr(self.numeric_binding, field_name, "") or "")
                    fact_value = str(getattr(self, field_name, "") or "")
                    if bound_value and bound_value != fact_value:
                        structural_errors = tuple(self.structural_validation_errors) + (
                            error_code,
                        )
                        object.__setattr__(self, "structural_validation_errors", structural_errors)
        if not self.source_candidate_ids and self.candidate_id:
            object.__setattr__(self, "source_candidate_ids", (self.candidate_id,))
        structural_errors = tuple(self.structural_validation_errors)
        if self.evidence_context is not None:
            context_type = self.evidence_context.structural_type
            supplied_type = str(self.structural_type or "")
            if supplied_type.casefold() not in {"", "body", "paragraph"} and (
                supplied_type != context_type
            ):
                structural_errors += ("structural_type_transition_mismatch",)
            object.__setattr__(self, "structural_type", context_type)
            structural_errors += structural_binding_validation_errors(
                self.evidence_context
            )
        elif self.structural_type in {
            item.value for item in StructuralEvidenceType
        }:
            structural_errors += ("structural_parent_missing",)
        structural_errors = tuple(dict.fromkeys(structural_errors))
        object.__setattr__(self, "structural_validation_errors", structural_errors)
        object.__setattr__(self, "structural_binding_valid", not structural_errors)

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "requirement_ids": list(self.requirement_ids),
            "origin_requirement_id": self.origin_requirement_id,
            "fact_id": self.fact_id,
            "source_fact_id": self.source_fact_id,
            "source_fact_occurrence_id": self.source_fact_occurrence_id,
            "source_fact_occurrence_provenance": list(
                self.source_fact_occurrence_provenance
            ),
            "subquestion_id": self.subquestion_id,
            "subject": self.subject,
            "metric": self.metric,
            "facet": self.facet,
            "period": self.period,
            "time_role": self.time_role,
            "value": self.value,
            "unit": self.unit,
            "numeric_role": self.numeric_role,
            "scalar_role": self.scalar_role,
            "quantity_kind": self.quantity_kind,
            "statistical_role": self.statistical_role,
            "aggregation_role": self.aggregation_role,
            "structural_type": self.structural_type,
            "row_column_bound": self.row_column_bound,
            "row_header": self.row_header,
            "column_header": self.column_header,
            "evidence_span": list(self.evidence_span),
            "evidence_span_sha256": self.evidence_span_sha256,
            "population": self.population,
            "direction": self.direction,
            "qualifiers": list(self.qualifiers),
            "evidence_id": self.evidence_id,
            "evidence_subject": self.evidence_subject,
            "evidence_predicate": self.evidence_predicate,
            "evidence_target_entity": self.evidence_target_entity,
            "evidence_population": self.evidence_population,
            "evidence_period": self.evidence_period,
            "evidence_unit": self.evidence_unit,
            "evidence_numeric_role": self.evidence_numeric_role,
            "evidence_polarity": self.evidence_polarity,
            "candidate_id": self.candidate_id,
            "source_evidence_unit_ids": list(self.source_evidence_unit_ids),
            "source_candidate_ids": list(self.source_candidate_ids),
            "source_typed_fact_ids": list(self.source_typed_fact_ids),
            "source_typed_fact_sha256s": list(self.source_typed_fact_sha256s),
            "source_facets": list(source_fact_facets(self)),
            "operator_derivation_id": self.operator_derivation_id,
            "derivation_description": self.derivation_description,
            "evidence_context": (
                self.evidence_context.as_dict() if self.evidence_context else None
            ),
            "model_assisted_semantics": self.model_assisted_semantics,
            "fact_type": self.fact_type,
            "evidence_atom_id": self.evidence_atom.atom_id if self.evidence_atom else "",
            "evidence_atom": (
                self.evidence_atom.as_dict() if self.evidence_atom else None
            ),
            "numeric_binding_provenance": (
                self.numeric_binding.as_dict()
                if self.numeric_binding is not None else None
            ),
            "fact_constructor": (
                "derived_from_compatible_source_facts"
                if self.fact_type.startswith("derived_")
                else "source_derived_numeric"
            ),
            "source_semantic_fields": {
                "subject": self.evidence_subject,
                "predicate": self.evidence_predicate,
                "target_entity": self.evidence_target_entity,
                "population": self.evidence_population,
                "period": self.evidence_period,
                "unit": self.evidence_unit,
                "numeric_role": self.evidence_numeric_role,
                "scalar_role": self.scalar_role,
                "quantity_kind": self.quantity_kind,
                "statistical_role": self.statistical_role,
                "aggregation_role": self.aggregation_role,
                "polarity": self.evidence_polarity,
            },
            "structural_binding_valid": self.structural_binding_valid,
            "structural_validation_errors": list(
                self.structural_validation_errors
            ),
        }


@dataclass(frozen=True)
class TextPropositionFact:
    requirement_id: str
    requirement_ids: tuple[str, ...]
    origin_requirement_id: str
    subquestion_id: str
    subject: str
    predicate: str
    object_text: str
    qualifiers: tuple[str, ...]
    evidence_id: str
    source_claim: str = field(repr=False, compare=False)
    fact_id: str = ""
    fact_type: str = "text_proposition"
    structural_type: str = "body"
    row_column_bound: bool = True
    proposition_start: int = 0
    proposition_end: int = 0
    proposition_sha256: str = ""
    evidence_subject: str = ""
    evidence_predicate: str = ""
    evidence_target_entity: str = ""
    evidence_population: str = ""
    evidence_period: str = ""
    evidence_unit: str = ""
    evidence_numeric_role: str = "text_proposition"
    evidence_polarity: str = "neutral"
    evidence_span: tuple[int, int] = (0, 0)
    candidate_id: str = ""
    source_evidence_unit_ids: tuple[str, ...] = ()
    source_candidate_ids: tuple[str, ...] = ()
    source_typed_fact_ids: tuple[str, ...] = ()
    operator_derivation_id: str = ""
    derivation_description: str = ""
    evidence_context: Optional[EvidenceContext] = field(
        default=None, repr=False, compare=False
    )
    relation_arguments: tuple[RelationArgument, ...] = ()
    model_assisted_semantics: bool = False
    canonical_valid: bool = True
    canonical_validation_errors: tuple[str, ...] = ()
    evidence_atom: Optional[EvidenceAtom] = field(
        default=None, repr=False, compare=False
    )
    source_fact_id: str = ""
    source_fact_occurrence_id: str = ""
    source_fact_occurrence_provenance: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        ids = tuple(dict.fromkeys(
            self.requirement_ids
            if self.requirement_ids
            else ((self.requirement_id,) if self.requirement_id else ())
        ))
        object.__setattr__(self, "requirement_ids", ids)
        if not self.fact_id:
            payload = "\x1f".join((
                (
                    self.source_fact_occurrence_id
                    if self.source_fact_occurrence_id else self.evidence_id
                ),
                (
                    ""
                    if self.source_fact_occurrence_id else
                    "|".join(self.source_evidence_unit_ids)
                ),
                self.object_text, self.proposition_sha256,
            ))
            object.__setattr__(
                self, "fact_id",
                "fact-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24],
            )
        if not self.source_fact_id:
            object.__setattr__(self, "source_fact_id", self.fact_id)
        if not self.source_evidence_unit_ids:
            object.__setattr__(self, "source_evidence_unit_ids", (self.evidence_id,))
        if not self.source_candidate_ids and self.candidate_id:
            object.__setattr__(self, "source_candidate_ids", (self.candidate_id,))
        validation_errors = tuple(self.canonical_validation_errors)
        if self.evidence_atom is not None and (
            self.evidence_atom.evidence_unit_id != self.evidence_id
            or self.evidence_atom.text not in self.object_text
        ):
            validation_errors += ("evidence_atom_source_mismatch",)
        if self.evidence_context is not None:
            context = self.evidence_context
            supplied_type = str(self.structural_type or "")
            if supplied_type.casefold() not in {"", "body", "paragraph"} and (
                supplied_type != context.structural_type
            ):
                validation_errors += ("structural_type_transition_mismatch",)
            object.__setattr__(self, "structural_type", context.structural_type)
            # These are evidence facts, not requirement-shaped placeholders.
            # The output fact must therefore retain the source-derived
            # proposition and semantic relation exactly as its context did.
            if self.object_text != context.local_proposition:
                validation_errors += ("source_proposition_mismatch",)
            for field_name, context_value, error_code in (
                ("subject", context.evidence_subject, "source_fact_subject_mismatch"),
                ("predicate", context.evidence_predicate, "source_fact_predicate_mismatch"),
            ):
                supplied = str(getattr(self, field_name) or "")
                if context_value and supplied and supplied != context_value:
                    validation_errors += (error_code,)
            for field_name, context_value, error_code in (
                ("evidence_subject", context.evidence_subject, "source_semantic_subject_mismatch"),
                ("evidence_predicate", context.evidence_predicate, "source_semantic_predicate_mismatch"),
                ("evidence_target_entity", context.evidence_target_entity, "source_semantic_target_mismatch"),
                ("evidence_population", context.population, "source_semantic_population_mismatch"),
                ("evidence_period", context.period, "source_semantic_period_mismatch"),
            ):
                supplied = str(getattr(self, field_name) or "")
                if not supplied:
                    object.__setattr__(self, field_name, context_value)
                elif supplied != context_value:
                    relation_values = {
                        str(argument.value or "")
                        for argument in context.relation_arguments
                    }
                    # Some relation predicates encode the target as their
                    # first source-bound argument.  It remains source-derived
                    # even when the generic context target slot is empty.
                    if not (not context_value and supplied in relation_values):
                        validation_errors += (error_code,)
        relation = str(self.evidence_predicate or self.predicate or "")
        if relation in RELATION_ARGUMENT_SCHEMAS:
            fact_arguments = {item.role: item for item in self.relation_arguments}
            validation_errors += tuple(
                f"relation_argument_missing:{role}"
                for role in RELATION_ARGUMENT_SCHEMAS[relation]
                if role not in fact_arguments
            )
            if self.evidence_context is None:
                if not validation_errors:
                    validation_errors = ("relation_argument_provenance_missing",)
            else:
                validation_errors += relation_argument_validation_errors(
                    self.evidence_context,
                    predicate=relation,
                )
                context_arguments = {
                    item.role: item for item in self.evidence_context.relation_arguments
                }
                for role, argument in fact_arguments.items():
                    context_argument = context_arguments.get(role)
                    if (
                        context_argument is None
                        or context_argument.value != argument.value
                        or context_argument.provenance != argument.provenance
                    ):
                        validation_errors += (
                            f"relation_argument_context_mismatch:{role}",
                        )
        if self.evidence_context is not None:
            validation_errors += structural_binding_validation_errors(
                self.evidence_context
            )
        elif self.structural_type in {
            item.value for item in StructuralEvidenceType
        }:
            validation_errors += ("structural_parent_missing",)
        validation_errors = tuple(dict.fromkeys(validation_errors))
        object.__setattr__(self, "canonical_validation_errors", validation_errors)
        object.__setattr__(self, "canonical_valid", not validation_errors)

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "source_fact_id": self.source_fact_id,
            "source_fact_occurrence_id": self.source_fact_occurrence_id,
            "source_fact_occurrence_provenance": list(
                self.source_fact_occurrence_provenance
            ),
            "fact_type": self.fact_type,
            "requirement_id": self.requirement_id,
            "requirement_ids": list(self.requirement_ids),
            "origin_requirement_id": self.origin_requirement_id,
            "subquestion_id": self.subquestion_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object_text": self.object_text,
            "qualifiers": list(self.qualifiers),
            "evidence_id": self.evidence_id,
            "structural_type": self.structural_type,
            "row_column_bound": self.row_column_bound,
            "proposition_start": self.proposition_start,
            "proposition_end": self.proposition_end,
            "proposition_sha256": self.proposition_sha256,
            "evidence_subject": self.evidence_subject,
            "evidence_predicate": self.evidence_predicate,
            "evidence_target_entity": self.evidence_target_entity,
            "evidence_population": self.evidence_population,
            "evidence_period": self.evidence_period,
            "evidence_unit": self.evidence_unit,
            "evidence_numeric_role": self.evidence_numeric_role,
            "evidence_polarity": self.evidence_polarity,
            "evidence_span": list(self.evidence_span),
            "candidate_id": self.candidate_id,
            "source_evidence_unit_ids": list(self.source_evidence_unit_ids),
            "source_candidate_ids": list(self.source_candidate_ids),
            "source_typed_fact_ids": list(self.source_typed_fact_ids),
            "operator_derivation_id": self.operator_derivation_id,
            "derivation_description": self.derivation_description,
            "evidence_context": (
                self.evidence_context.as_dict() if self.evidence_context else None
            ),
            "relation_arguments": [
                item.as_dict() for item in self.relation_arguments
            ],
            "model_assisted_semantics": self.model_assisted_semantics,
            "evidence_atom_id": self.evidence_atom.atom_id if self.evidence_atom else "",
            "evidence_atom": (
                self.evidence_atom.as_dict() if self.evidence_atom else None
            ),
            "fact_constructor": "source_derived_text_relation",
            "source_semantic_fields": {
                "subject": self.evidence_subject,
                "predicate": self.evidence_predicate,
                "target_entity": self.evidence_target_entity,
                "population": self.evidence_population,
                "period": self.evidence_period,
                "unit": self.evidence_unit,
                "numeric_role": self.evidence_numeric_role,
                "polarity": self.evidence_polarity,
            },
            "canonical_valid": self.canonical_valid,
            "canonical_validation_errors": list(
                self.canonical_validation_errors
            ),
            "structural_binding_valid": not any(
                reason in {
                    "proposition_boundary_unresolved",
                    "table_row_boundary_unresolved",
                    "table_column_role_unresolved",
                    "cross_row_binding_rejected",
                    "cross_item_binding_rejected",
                    "cross_category_binding_rejected",
                    "structural_parent_missing",
                    "structural_binding_ambiguous",
                    "structural_argument_mismatch",
                    "structural_value_unbound",
                }
                for reason in self.canonical_validation_errors
            ),
        }


def _source_fact_occurrence_location(
    unit: EvidenceUnit,
    atom: EvidenceAtom,
) -> tuple[str, ...]:
    """Return a source-local anchor for one atom without using request state.

    Structural views can have different unit IDs while covering the same raw
    source characters.  Prefer the atom's mapped raw span, then an exact
    atom/source-span match.  If neither is provable, retain the structural
    view as a fail-closed anchor rather than merging on semantic text alone.
    """
    structural = unit.structural_evidence_unit
    source_text = str(unit.raw_hit.get("document") or "")
    identity = (
        str(unit.source_id or atom.source_id or ""),
        str(unit.chunk_id or atom.chunk_id or ""),
        str(unit.page if unit.page is not None else atom.page or ""),
    )
    if structural is None or not structural.source_spans or not source_text:
        return (*identity, "structural_view", str(unit.id or atom.evidence_unit_id))
    spans = tuple(sorted(
        structural.source_spans, key=lambda item: (item.start, item.end)
    ))
    source_start, source_end = spans[0].start, spans[-1].end
    atom_start, atom_end = atom.source_span
    if (
        0 <= source_start < source_end <= len(source_text)
        and source_text[source_start:source_end] == structural.local_text
        and 0 <= atom_start < atom_end <= len(structural.local_text)
    ):
        absolute_start = source_start + atom_start
        absolute_end = source_start + atom_end
        return (
            *identity,
            "atom_source_span",
            str(absolute_start),
            str(absolute_end),
            hashlib.sha256(
                source_text[absolute_start:absolute_end].encode("utf-8")
            ).hexdigest(),
        )
    exact_spans = tuple(
        item for item in spans if item.sha256 == atom.text_sha256
    )
    if len(exact_spans) == 1:
        span = exact_spans[0]
        return (
            *identity,
            "exact_structural_source_span",
            str(span.start), str(span.end), span.sha256,
        )
    literal = str(atom.value or "")
    if literal:
        literal_spans: list[tuple[int, int]] = []
        pattern = re.compile(r"(?<![0-9A-Za-z])" + re.escape(literal) + r"(?![0-9A-Za-z])")
        for span in spans:
            if not (0 <= span.start < span.end <= len(source_text)):
                continue
            literal_spans.extend(
                (span.start + match.start(), span.start + match.end())
                for match in pattern.finditer(source_text[span.start:span.end])
            )
        if len(literal_spans) == 1:
            start, end = literal_spans[0]
            return (
                *identity,
                "unique_literal_source_span",
                str(start), str(end),
                hashlib.sha256(source_text[start:end].encode("utf-8")).hexdigest(),
            )
    # An unresolved cross-view anchor is not an excuse to collapse evidence.
    return (*identity, "structural_view", str(unit.id or atom.evidence_unit_id))


def _source_fact_occurrence_identity(
    unit: EvidenceUnit,
    fact: TypedFact | TextPropositionFact,
) -> str:
    """Derive a request-independent SOURCE_FACT occurrence identity.

    Identity combines a proved source location with the complete source-side
    semantic binding.  Neither semantic equality nor an evidence-unit ID is
    sufficient by itself: the former could collapse independent statements,
    and the latter would duplicate one statement reached through parent,
    child, or bounded-composite views.
    """
    atom = fact.evidence_atom
    if atom is None:
        return ""
    relation_arguments = tuple(sorted(
        (
            str(item.role or ""),
            str(item.value or ""),
        )
        for item in getattr(fact, "relation_arguments", ())
    ))
    semantic_binding = {
        "fact_type": str(getattr(fact, "fact_type", "") or "numeric"),
        "subject": str(getattr(fact, "subject", "") or ""),
        "metric": str(getattr(fact, "metric", "") or ""),
        "facet": str(getattr(fact, "facet", "") or ""),
        "period": str(getattr(fact, "period", "") or ""),
        "value": str(getattr(fact, "value", "") or ""),
        "unit": str(getattr(fact, "unit", "") or ""),
        "object_text": str(getattr(fact, "object_text", "") or ""),
        "predicate": str(getattr(fact, "evidence_predicate", "") or getattr(fact, "predicate", "") or ""),
        "target_entity": str(getattr(fact, "evidence_target_entity", "") or ""),
        "population": str(getattr(fact, "evidence_population", "") or ""),
        "polarity": str(getattr(fact, "evidence_polarity", "") or ""),
        "quantity_kind": str(getattr(fact, "quantity_kind", "") or ""),
        "statistical_role": str(getattr(fact, "statistical_role", "") or ""),
        "aggregation_role": str(getattr(fact, "aggregation_role", "") or ""),
        "direction": str(getattr(fact, "direction", "") or ""),
        "qualifiers": tuple(str(item) for item in getattr(fact, "qualifiers", ())),
        "relation_arguments": relation_arguments,
    }
    source_location = tuple(
        str(item) for item in fact.source_fact_occurrence_provenance
    ) or _source_fact_occurrence_location(unit, atom)
    payload = json.dumps(
        {
            "source_location": source_location,
            "semantic_binding": semantic_binding,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "sfo-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _bind_source_fact_occurrence(
    unit: EvidenceUnit,
    fact: TypedFact | TextPropositionFact,
) -> TypedFact | TextPropositionFact:
    """Bind a SOURCE_FACT to stable source support before registration.

    ``evidence_id`` remains the structural observation used by downstream
    citation selection.  It is deliberately not authoritative source
    provenance: parent, child, and bounded views can expose the same source
    occurrence in a different order.  The explicit source-local provenance
    below is the stable authority used for identity and audit.
    """
    atom = fact.evidence_atom
    if atom is None:
        return fact
    bound = replace(
        fact,
        fact_id="",
        source_fact_id="",
        source_fact_occurrence_id="",
        source_fact_occurrence_provenance=_source_fact_occurrence_location(
            unit, atom,
        ),
    )
    return replace(
        bound,
        fact_id="",
        source_fact_id="",
        source_fact_occurrence_id=_source_fact_occurrence_identity(unit, bound),
    )


def _register_source_fact_occurrence(
    fact: TypedFact | TextPropositionFact,
    registry: Optional[dict[str, TypedFact | TextPropositionFact]],
) -> tuple[TypedFact | TextPropositionFact, bool]:
    """Return the single source fact emitted for this execution occurrence.

    The registry deliberately exists only for one evidence-coverage execution.
    A later request re-observes the same source fact identity in its own trace;
    it is not silently removed by process-global state.
    """
    occurrence_id = fact.source_fact_occurrence_id
    if registry is None or not occurrence_id:
        return fact, True
    existing = registry.get(occurrence_id)
    if existing is None:
        registry[occurrence_id] = fact
        return fact, True
    return existing, False


def _source_fact_semantic_completeness_errors(
    fact: TypedFact | TextPropositionFact,
) -> tuple[str, ...]:
    """Validate only the semantic roles established by one source fact.

    This boundary intentionally receives no requirement.  A source fact is
    either complete enough to preserve the proposition the evidence actually
    establishes, or it is rejected here with source-local reasons.  Request
    compatibility remains a separate, strictly downstream decision.
    """
    reasons: list[str] = []
    if isinstance(fact, TypedFact):
        if not str(fact.value or "").strip():
            reasons.append("source_value_unresolved")
        if not str(fact.evidence_subject or "").strip():
            reasons.append("source_subject_unresolved")
        if not str(fact.metric or "").strip():
            reasons.append("source_metric_unresolved")
        if not str(fact.evidence_predicate or "").strip():
            reasons.append("source_predicate_unresolved")
        if not str(fact.evidence_unit or "").strip():
            reasons.append("source_unit_unresolved")
        if not str(fact.evidence_span_sha256 or "").strip():
            reasons.append("source_span_unresolved")
        binding = fact.numeric_binding
        if binding is None:
            reasons.append("source_numeric_binding_provenance_missing")
        else:
            if (
                fact.statistical_role != StatisticalRole.UNSPECIFIED.value
                and (
                    binding.statistical_role != fact.statistical_role
                    or binding.statistical_role_cue_span is None
                )
            ):
                reasons.append("source_statistical_role_cue_unresolved")
            if (
                fact.aggregation_role
                not in {AggregationRole.NONE.value, AggregationRole.UNSPECIFIED.value}
                and (
                    binding.aggregation_role != fact.aggregation_role
                    or (
                        binding.aggregation_role_cue_span is None
                        and not binding.structural_role
                    )
                )
            ):
                reasons.append("source_aggregation_role_cue_unresolved")
        source_claim = str(fact.source_claim or "")
        source_container = str(fact.source_container_claim or source_claim)
        normalized_subject = normalize_request_text(
            fact.evidence_subject or fact.subject or ""
        )
        continuation = re.search(
            r"\bof\s+which\b|(?:מתוכ(?:ו|ן)|מהם|מהן)",
            source_claim,
            re.IGNORECASE,
        )
        if continuation is not None:
            # A continuation is meaningful only when the atom retained the
            # governed source subject.  A scalar labelled merely "of which"
            # is not a complete source proposition.
            if (
                not normalized_subject
                or re.fullmatch(r"(?:of\s+which|which|מתוכ(?:ו|ן)|מהם|מהן)",
                                normalized_subject, re.IGNORECASE)
            ):
                reasons.append("source_continuation_parent_anchor_unresolved")
            parent_prefix = source_container[:source_container.casefold().find(
                continuation.group(0).casefold()
            )]
            if not re.search(r"[A-Za-z\u05d0-\u05ea]", parent_prefix):
                reasons.append("source_continuation_parent_anchor_unresolved")
        confidence_signaled = (
            fact.scalar_role == "confidence_level"
            or fact.numeric_role == "confidence_level"
            or bool(re.search(r"\bconfidence\b|ביטחון", source_claim, re.I))
        )
        if confidence_signaled and not re.search(
            r"\bconfidence\b|ביטחון", source_claim, re.IGNORECASE
        ):
            reasons.append("source_confidence_relation_unresolved")
        share_signaled = (
            str(fact.evidence_unit or fact.unit or "").casefold()
            in {"%", "percent", "percentage", "אחוז"}
            and bool(re.search(
                r"\b(?:share|percentage|percent|rate|portion|represented|"
                r"accounted\s+for)\b|(?:שיעור|נתח|אחוז)",
                source_claim,
                re.IGNORECASE,
            ))
        )
        if share_signaled and not (
            normalized_subject
            or normalize_request_text(fact.evidence_target_entity or "")
            or normalize_request_text(fact.facet or "")
        ):
            reasons.append("source_share_category_unresolved")
        # Completeness for a scalar is determined by its own bounded atom
        # proposition.  A wider source-only container may provide governing
        # context, but values in a neighbouring proposition are not a local
        # ambiguity for this fact.
        source_text = fact.source_claim
        if (
            _UNLABELLED_PARALLEL_NUMERIC_RE.search(source_text)
            and not (
                fact.numeric_binding is not None
                and fact.numeric_binding.binding_method
                in {
                    "local_label_value",
                    "local_period_value",
                    "local_unit_role",
                    "local_scalar_role",
                    "local_response_category",
                    "structural_row_column",
                }
            )
        ):
            reasons.extend((
                "source_multivalue_role_unresolved",
                "source_numeric_binding_ambiguous",
            ))
        if len(tuple(_YEAR_RE.finditer(source_text))) > 1 and not fact.evidence_period:
            reasons.append("source_period_unresolved")
        if _EVIDENCE_POPULATION_RE.search(source_text) and not fact.evidence_population:
            reasons.append("source_population_unresolved")
        facet_signaled = bool(
            _EXPERIENCE_FACET_RE.search(source_text)
            or _EDUCATION_FACET_RE.search(source_text)
        )
        if facet_signaled and not fact.facet:
            reasons.append("source_facet_unresolved")
        if not fact.structural_binding_valid:
            reasons.extend(fact.structural_validation_errors)
    else:
        if not str(fact.object_text or "").strip():
            reasons.append("source_proposition_unresolved")
        if not str(fact.evidence_subject or fact.subject or "").strip():
            reasons.append("source_subject_unresolved")
        predicate = str(fact.evidence_predicate or fact.predicate or "").strip()
        if not predicate:
            reasons.append("source_predicate_unresolved")
        required_roles = RELATION_ARGUMENT_SCHEMAS.get(predicate, ())
        arguments = {
            str(item.role or ""): str(item.value or "")
            for item in fact.relation_arguments
        }
        for role in required_roles:
            if not arguments.get(role, "").strip():
                reasons.append(f"relation_argument_missing:{role}")
        if required_roles:
            reasons.extend(relation_argument_validation_errors(
                fact.evidence_context,
                predicate=predicate,
            ) if fact.evidence_context is not None else (
                "relation_argument_provenance_missing",
            ))
        if not str(fact.proposition_sha256 or "").strip():
            reasons.append("source_span_unresolved")
        reasons.extend(fact.canonical_validation_errors)
    return tuple(dict.fromkeys(reasons))


@dataclass(frozen=True)
class CoverageRequirement:
    id: str
    subquestion_id: str
    kind: str
    label: str
    status: CoverageStatus
    evidence_ids: tuple[str, ...]
    metric: str = ""
    facet: str = ""
    period: str = ""
    time_role: str = ""
    expected_unit: str = ""
    requires_value: bool = False
    requires_qualifier: bool = False
    origin_id: str = ""
    requirement_ids: tuple[str, ...] = ()
    population: str = ""
    scalar_role: str = "not_applicable"
    quantity_kind: str = "unspecified"
    statistical_role: str = "unspecified"
    aggregation_role: str = "unspecified"
    satisfaction_mode: str = RequirementSatisfactionMode.SOURCE_FACT.value


@dataclass(frozen=True)
class FactAdmissibilityResult:
    """One canonical source-fact admission decision and its compact proof.

    A compatibility pass is not a lexical score.  It is legal only when each
    request-constrained dimension has a source-derived value, compatible
    decision, and provenance.  ``scalar_role`` deliberately does not occur
    here: it remains a serialization projection, while admission consumes the
    independent quantity semantics.
    """

    reason_codes: tuple[str, ...]
    dimension_decisions: Mapping[str, str]
    dimension_proof: Mapping[str, Mapping[str, Any]]
    first_material_mismatch: str = ""

    @property
    def compatible(self) -> bool:
        return not self.reason_codes


@dataclass(frozen=True)
class ListCoverageRow:
    item: str
    attributes: Mapping[str, str]
    item_evidence_ids: tuple[str, ...]
    attribute_evidence_ids: Mapping[str, tuple[str, ...]]
    complete: bool


@dataclass(frozen=True)
class EvidenceCoverage:
    requirements: tuple[CoverageRequirement, ...]
    items: tuple[ListCoverageRow, ...]
    requested_count: int
    covered_count: int
    complete: bool
    coverage_ratio: float
    reason_code: AbstentionReason
    typed_facts: tuple[TypedFact | TextPropositionFact, ...] = ()
    targeted_repair_attempted: bool = False
    targeted_repair_requirement_ids: tuple[str, ...] = ()

    def safe_summary(self) -> dict[str, Any]:
        return {
            "requirements": [
                {
                    "id": item.id,
                    "status": item.status.value,
                    "evidence_ids": list(item.evidence_ids),
                }
                for item in self.requirements
            ],
            "requested_count": self.requested_count,
            "covered_count": self.covered_count,
            "complete": self.complete,
            "coverage_ratio": self.coverage_ratio,
            "reason_code": self.reason_code.value,
            "typed_fact_count": len(self.typed_facts),
            "targeted_repair_attempted": self.targeted_repair_attempted,
            "targeted_repair_requirement_ids": list(
                self.targeted_repair_requirement_ids
            ),
        }


@dataclass(frozen=True)
class StructuredRowCandidate:
    item: str
    attributes: Mapping[str, str]
    item_evidence_ids: tuple[str, ...]
    attribute_evidence_ids: Mapping[str, tuple[str, ...]]


@dataclass(frozen=True)
class SubquestionRetrieval:
    subquestion_id: str
    query_count: int
    vector_candidates: int
    lexical_candidates: int
    exact_candidates: int
    fused_candidates: int
    selected_evidence_ids: tuple[str, ...]
    query_rewrite_ms: float
    vector_ms: float
    lexical_ms: float
    exact_ms: float
    fusion_ms: float
    structural_expansion_ms: float
    multi_hop_ms: float


@dataclass(frozen=True)
class AdaptiveRetrievalResult:
    evidence_units: tuple[EvidenceUnit, ...]
    by_subquestion: Mapping[str, tuple[str, ...]]
    coverage: EvidenceCoverage
    diagnostics: tuple[SubquestionRetrieval, ...]
    retrieval_timed_out: bool
    integrity_failures: int


def _normalized_terms(value: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for raw in _TOKEN_RE.findall(normalize_request_text(value).casefold()):
        if raw in _STOPWORDS:
            continue
        tokens.append(raw)
        if (
            len(raw) >= 4
            and re.fullmatch(r"[\u05d0-\u05ea]+", raw)
            and raw[0] in _HEBREW_PREFIXES
        ):
            stripped = raw[1:]
            if len(stripped) >= 2 and stripped not in _STOPWORDS:
                tokens.append(stripped)
    return tuple(dict.fromkeys(tokens))


def _term_overlap(left: str, right: str) -> float:
    left_terms = set(_normalized_terms(left))
    right_terms = set(_normalized_terms(right))
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(1, len(left_terms))


def _semantic_term_overlap(left: str, right: str) -> float:
    """Match bounded terminology groups without treating rewrites as evidence."""
    left_terms = set(_normalized_terms(left))
    right_terms = set(_normalized_terms(right))
    if not left_terms or not right_terms:
        return 0.0
    matched = set(left_terms & right_terms)
    normalized_right = normalize_request_text(right).casefold()
    for group in _SYNONYM_GROUPS:
        normalized_group = tuple(term.casefold() for term in group)
        if not any(
            term in left_terms or term in normalize_request_text(left).casefold()
            for term in normalized_group
        ):
            continue
        if any(term in normalized_right for term in normalized_group):
            matched.update(
                term for term in left_terms
                if any(
                    term == candidate
                    or term in candidate.split()
                    or candidate in term
                    for candidate in normalized_group
                )
            )
    overlap = len(matched) / max(1, len(left_terms))
    left_hebrew = len(re.findall(r"[\u05d0-\u05ea]", left))
    left_latin = len(re.findall(r"[A-Za-z]", left))
    right_hebrew = len(re.findall(r"[\u05d0-\u05ea]", right))
    right_latin = len(re.findall(r"[A-Za-z]", right))
    left_hebrew_dominant = left_hebrew >= max(2, left_latin * 2)
    left_latin_dominant = left_latin >= max(2, left_hebrew * 2)
    right_hebrew_dominant = right_hebrew >= max(2, right_latin * 2)
    right_latin_dominant = right_latin >= max(2, right_hebrew * 2)
    clearly_cross_language = (
        left_hebrew_dominant
        and right_latin_dominant
    ) or (
        left_latin_dominant
        and right_hebrew_dominant
    )
    if clearly_cross_language:
        shared_concepts = sum(
            any(
                term.casefold() in normalize_request_text(left).casefold()
                for term in group
            )
            and any(
                term.casefold() in normalized_right
                for term in group
            )
            for group in _SYNONYM_GROUPS
        )
        if shared_concepts:
            # The terms inside one bilingual concept are alternatives, not
            # independent query requirements. A source-side synonym therefore
            # establishes topical coverage without requiring impossible
            # literal overlap across scripts. Endpoint coverage remains a
            # separate exact check below.
            overlap = max(overlap, min(1.0, 0.5 + 0.15 * (shared_concepts - 1)))
    return overlap


_RELATIONAL_EVENT_FRAMING_RE = re.compile(
    r"\b(?:increase|increased|increasing|rise|rose|rising|growth|grew|"
    r"decrease|decreased|decreasing|fall|fell|falling|decline|declined|"
    r"change|changed|because|reason|cause|caused|explain|explains|explained|"
    r"report|reports|reported|why|what|how|whether|that)\b|"
    r"(?:עלייה|עליה|עלה|עלתה|גדל|גדלה|ירידה|ירד|ירדה|פחת|פחתה|"
    r"שינוי|השתנה|בגלל|בשל|עקב|סיבה|גרם|מסביר|מסבירה|דוח)",
    re.IGNORECASE,
)


def _relation_event_compatible(requested: str, evidence: str) -> bool:
    requested_direction = _direction_in_claim(requested)
    evidence_direction = _direction_in_claim(evidence)
    if (
        requested_direction
        and evidence_direction
        and requested_direction != evidence_direction
    ):
        return False
    requested_core = _RELATIONAL_EVENT_FRAMING_RE.sub(" ", requested)
    evidence_core = _RELATIONAL_EVENT_FRAMING_RE.sub(" ", evidence)
    return max(
        _term_overlap(requested_core, evidence_core),
        _semantic_term_overlap(requested_core, evidence_core),
    ) >= 0.45


def _contains_normalized_phrase(needle: str, haystack: str) -> bool:
    left = " ".join(_normalized_terms(needle))
    right = " ".join(_normalized_terms(haystack))
    return bool(left) and left in right


_GENERIC_POPULATION_TERMS = frozenset({
    "among", "within", "across", "in", "of", "the", "population",
    "sample", "subgroup", "segment", "cohort", "group", "region",
    "companies", "company", "workers", "worker", "employees", "employee",
    "customers", "customer", "sites", "site", "projects", "project",
    "בקרב", "מבין", "מתוך", "אוכלוסייה", "אוכלוסיית", "מדגם",
    "קבוצה", "תת", "חברות", "חברה", "עובדים", "עובד", "לקוחות",
    "לקוח", "אתרים", "אתר", "פרויקטים", "פרויקט", "אזור",
})


def _population_scope_compatible(required: str, observed_clause: str) -> bool:
    """Compatibility adapter for source-derived candidate text only."""
    return population_scope_compatibility(required, observed_clause)[0]


def build_query_rewrites(
    subquestion: Subquestion,
    execution_contract: Optional[RequestExecutionContract] = None,
) -> tuple[QueryRewrite, ...]:
    """Create bounded morphology, bilingual, and terminology rewrites."""
    if execution_contract is None and os.getenv(
        "PROJECT_X_CANDIDATE_MODE", "0"
    ).strip().casefold() in {"1", "true", "yes", "on"}:
        raise RuntimeError("query generation requires RequestExecutionContract")
    primary = normalize_request_text(subquestion.query)
    output: list[QueryRewrite] = [QueryRewrite(primary, "primary")]
    seen = {primary.casefold()}

    def add(text: str, category: str) -> None:
        text = normalize_request_text(text)
        key = text.casefold()
        if text and key not in seen and len(output) < MAX_QUERY_REWRITES:
            seen.add(key)
            output.append(QueryRewrite(text, category))

    terms = _normalized_terms(primary)
    if terms:
        add(" ".join(terms), "normalized_morphology")

    normalized_primary = primary.casefold()
    for group in _SYNONYM_GROUPS:
        if any(term.casefold() in normalized_primary for term in group):
            matched = [
                term for term in group
                if term.casefold() not in normalized_primary
            ]
            add(
                f"{primary} {' '.join(matched[:6])}",
                "terminology_expansion",
            )

    if subquestion.subject and subquestion.subject != primary:
        add(subquestion.subject, "subject_focus")
    for dimension in subquestion.required_dimensions:
        add(
            f"{subquestion.subject} {dimension}",
            "dimension_focus",
        )
    for attribute in subquestion.required_attributes:
        add(
            f"{subquestion.subject} {attribute}",
            "attribute_focus",
        )
    return tuple(output)


def _hit_key(hit: Mapping[str, Any]) -> tuple[str, ...]:
    metadata = hit.get("metadata") or {}
    chunk_id = metadata.get("chunk_id") or hit.get("id")
    if chunk_id:
        return ("chunk", str(chunk_id))
    return (
        "fallback",
        str(metadata.get("source_id") or ""),
        str(metadata.get("page") or ""),
        str(metadata.get("chunk_index") or ""),
        normalize_evidence_whitespace(hit.get("document") or "")[:160],
    )


def reciprocal_rank_fuse(
    rankings: Sequence[tuple[RetrievalChannel, Sequence[Mapping[str, Any]]]],
    *,
    rank_constant: int = 60,
) -> list[dict[str, Any]]:
    scores: dict[tuple[str, ...], float] = defaultdict(float)
    hits: dict[tuple[str, ...], dict[str, Any]] = {}
    channels: dict[tuple[str, ...], set[RetrievalChannel]] = defaultdict(set)
    channel_ranks: dict[tuple[str, ...], dict[str, int]] = defaultdict(dict)
    best_position: dict[tuple[str, ...], int] = {}
    for channel, ranking in rankings:
        for position, raw_hit in enumerate(ranking, start=1):
            key = _hit_key(raw_hit)
            scores[key] += 1.0 / (rank_constant + position)
            channels[key].add(channel)
            channel_ranks[key][channel.value] = min(
                channel_ranks[key].get(channel.value, position), position
            )
            for existing_name, existing_rank in dict(
                raw_hit.get("_channel_ranks") or {}
            ).items():
                try:
                    rank_value = max(1, int(existing_rank))
                except (TypeError, ValueError):
                    continue
                channel_ranks[key][str(existing_name)] = min(
                    channel_ranks[key].get(str(existing_name), rank_value),
                    rank_value,
                )
            for existing_channel in raw_hit.get("_channels") or ():
                try:
                    channels[key].add(
                        existing_channel
                        if isinstance(existing_channel, RetrievalChannel)
                        else RetrievalChannel(str(existing_channel))
                    )
                except ValueError:
                    continue
            best_position[key] = min(best_position.get(key, position), position)
            if key not in hits:
                hits[key] = dict(raw_hit)
    ordered = sorted(
        hits,
        key=lambda key: (-scores[key], best_position[key], key),
    )
    output: list[dict[str, Any]] = []
    for key in ordered[:MAX_FUSED_RESULTS]:
        hit = hits[key]
        hit["_fusion_score"] = scores[key]
        hit["_channels"] = tuple(sorted(channels[key], key=lambda value: value.value))
        hit["_channel_ranks"] = dict(sorted(channel_ranks[key].items()))
        hit["_best_retrieval_rank"] = best_position[key]
        output.append(hit)
    return output


def _select_exact_anchors(subquestion: Subquestion) -> tuple[str, ...]:
    value = normalize_request_text(subquestion.query)
    anchors: list[str] = []
    for quoted in re.findall(r"[\"“”]([^\"“”]{2,80})[\"“”]", value):
        anchors.append(quoted)
    anchors.extend(_YEAR_RE.findall(value))
    anchors.extend(
        token
        for token in _normalized_terms(subquestion.subject)
        if len(token) >= 6 or token.isupper()
    )
    return tuple(dict.fromkeys(anchors))[:2]


def _select_conjunctive_groups(
    subquestion: Subquestion,
) -> tuple[tuple[tuple[str, ...], ...], ...]:
    terms = [
        token for token in _normalized_terms(subquestion.subject)
        if len(token) >= 3
    ][:4]
    if len(terms) < 2:
        return ()
    return ((tuple((term,) for term in terms[:4])),)


def _source_value(hit: Mapping[str, Any]) -> str:
    metadata = hit.get("metadata") or {}
    return str(
        metadata.get("source")
        or metadata.get("source_relative_path")
        or metadata.get("path")
        or ""
    )


def _record_index(hit: Mapping[str, Any]) -> Optional[int]:
    metadata = hit.get("metadata") or {}
    for key in ("chunk_index", "sentence_start"):
        value = metadata.get(key)
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            continue
    return None


def _section_value(hit: Mapping[str, Any]) -> str:
    metadata = hit.get("metadata") or {}
    return normalize_request_text(
        metadata.get("section")
        or metadata.get("title")
        or metadata.get("section_slug")
        or ""
    )


def _c45_load_diagnostics():
    """Contain initialization before requests; no lazy import wait at the edge."""
    try:
        import c45_diagnostics
        return c45_diagnostics
    except Exception:
        return None


_C45_DIAGNOSTICS = _c45_load_diagnostics()


def _c45_observe(witness_family_id):
    """Bounded diagnostic containment; no application evaluation belongs here."""
    try:
        _C45_DIAGNOSTICS.observe(witness_family_id)
    except Exception:
        pass


def classify_structural_unit(
    value: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> StructuralUnitType:
    """Classify one source span before it can become factual evidence."""
    text = normalize_evidence_whitespace(value)
    lowered = text.casefold()
    letters = re.findall(r"[A-Za-z\u05d0-\u05ea]", text)
    numbers = _NUMBER_WITH_OPTIONAL_UNIT_RE.findall(text)
    if not text:
        _c45_application_value = StructuralUnitType.MALFORMED
        _c45_observe('C44E2E2_FAMILY_09_118762662a9e')
        return _c45_application_value
    if re.fullmatch(r"\d{1,4}", text):
        _c45_application_value = StructuralUnitType.PAGE_NUMBER
        _c45_observe('C44E2E2_FAMILY_09_118762662a9e')
        return _c45_application_value
    if not letters and not re.search(r"\d", text):
        _c45_application_value = StructuralUnitType.MALFORMED
        _c45_observe('C44E2E2_FAMILY_09_118762662a9e')
        return _c45_application_value
    if text.startswith("[") and re.search(r">\s*Page\s+\d+\]?", text):
        _c45_application_value = StructuralUnitType.HEADER_FOOTER
        _c45_observe('C44E2E2_FAMILY_09_118762662a9e')
        return _c45_application_value
    if _NAVIGATION_HEADING_RE.search(text) or _DOTTED_LEADER_RE.search(text):
        if re.search(r"(?:list\s+of\s+tables|רשימת\s+טבלאות)", text, re.I):
            _c45_application_value = StructuralUnitType.LIST_OF_TABLES
            _c45_observe('C44E2E2_FAMILY_09_118762662a9e')
            return _c45_application_value
        if re.search(
            r"(?:list\s+of\s+(?:figures|charts)|רשימת\s+(?:איורים|תרשימים))",
            text,
            re.I,
        ) or re.search(r"(?:figure|fig\.?|chart|איור|תרשים)\s*\d+", text, re.I):
            _c45_application_value = StructuralUnitType.LIST_OF_FIGURES
            _c45_observe('C44E2E2_FAMILY_09_118762662a9e')
            return _c45_application_value
        _c45_application_value = StructuralUnitType.TABLE_OF_CONTENTS
        _c45_observe('C44E2E2_FAMILY_09_118762662a9e')
        return _c45_application_value
    formula_tokens = len(_FORMULA_TOKEN_RE.findall(text))
    if formula_tokens and (
        formula_tokens >= 2
        or len(letters) < 18
        or len(re.findall(r"[=<>χχ]", text)) >= 4
    ):
        _c45_application_value = StructuralUnitType.FORMULA
        _c45_observe('C44E2E2_FAMILY_09_118762662a9e')
        return _c45_application_value
    if (
        len(text) > 24
        and len(letters) < 6
        and len(re.findall(r"[^\w\s]", text)) > len(letters)
    ):
        _c45_application_value = StructuralUnitType.MALFORMED
        _c45_observe('C44E2E2_FAMILY_09_118762662a9e')
        return _c45_application_value
    if _SOURCE_CAPTION_RE.match(text):
        _c45_application_value = StructuralUnitType.CAPTION
        _c45_observe('C44E2E2_FAMILY_09_118762662a9e')
        return _c45_application_value
    if _CAPTION_START_RE.match(text):
        caption_body = _CAPTION_START_RE.sub("", text, count=1)
        material_numbers = [
            match.group("value")
            for match in _NUMBER_WITH_OPTIONAL_UNIT_RE.finditer(caption_body)
            if not _YEAR_RE.fullmatch(match.group("value"))
        ]
        quantitative = len(material_numbers) >= 2 and bool(
            _PREDICATE_RE.search(text) or _COUNT_UNIT_RE.search(text)
        )
        _c45_application_value = (
            StructuralUnitType.CHART_DATA
            if quantitative
            else StructuralUnitType.CAPTION
        )
        _c45_observe('C44E2E2_FAMILY_09_118762662a9e')
        return _c45_application_value
    if (metadata or {}).get("has_table_context") or _TABLE_ROW_RE.search(value):
        _c45_application_value = StructuralUnitType.TABLE_ROW
        _c45_observe('C44E2E2_FAMILY_09_118762662a9e')
        return _c45_application_value
    if re.match(r"^\s*[•▪◦*-]\s*", value):
        _c45_application_value = StructuralUnitType.BULLET
        _c45_observe('C44E2E2_FAMILY_09_118762662a9e')
        return _c45_application_value
    if len(_BULLET_RE.findall(value)) >= 2:
        _c45_application_value = StructuralUnitType.LIST
        _c45_observe('C44E2E2_FAMILY_09_118762662a9e')
        return _c45_application_value
    if _HEADING_PREFIX_RE.match(text) or (
        len(text.split()) <= 8
        and not numbers
        and not _PREDICATE_RE.search(text)
        and not re.search(r"[.!?׃]\s*$", text)
    ):
        _c45_application_value = StructuralUnitType.HEADING
        _c45_observe('C44E2E2_FAMILY_09_118762662a9e')
        return _c45_application_value
    _c45_application_value = StructuralUnitType.BODY
    _c45_observe('C44E2E2_FAMILY_09_118762662a9e')
    return _c45_application_value


_FACTUAL_STRUCTURAL_TYPES = frozenset({
    StructuralUnitType.BODY,
    StructuralUnitType.BULLET,
    StructuralUnitType.LIST,
    StructuralUnitType.TABLE_ROW,
    StructuralUnitType.CHART_DATA,
})


def _source_claim_candidates(text: str) -> tuple[str, ...]:
    """Recover complete layout-fragmented spans without skipping source text."""
    raw = _BREADCRUMB_RE.sub("", str(text or ""), count=1)
    normalized = normalize_evidence_whitespace(raw)
    normalized = re.sub(r"^\d{1,3}\s+(?=[A-Za-z\u05d0-\u05ea])", "", normalized)
    normalized = re.sub(
        r"(\bכ)\s*[-–]?\s*\.\s*(?=\d+(?:[.,]\d+)?\s*%)",
        r"\1-",
        normalized,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(
        r"\b(approx(?:imately)?)\.\s*(?=\d)",
        r"\1 ",
        normalized,
        flags=re.IGNORECASE,
    )
    candidates: list[str] = list(_reconstructed_bullet_claims(raw))
    candidates.extend(_SENTENCE_OR_BULLET_BREAK_RE.split(normalized))
    candidates.extend(complete_evidence_sentences(raw))
    candidates.extend(_BULLET_RE.findall(raw))
    bounded_fragments: list[str] = []
    for candidate in candidates:
        candidate = normalize_evidence_whitespace(candidate)
        for delimiter in re.finditer(r"[()]", candidate):
            prefix = candidate[:delimiter.start()].rstrip(" ,;:-")
            if (
                24 <= len(prefix) <= MAX_ALLOWED_CLAIM_CHARS
                and balanced_delimiters(prefix)
                and _PREDICATE_RE.search(prefix)
                and _NUMBER_WITH_OPTIONAL_UNIT_RE.search(prefix)
                and (
                    _COUNT_UNIT_RE.search(prefix)
                    or re.search(r"%|\bpercent(?:age)?\b|אחוז", prefix, re.I)
                )
            ):
                bounded_fragments.append(prefix)
            suffix = candidate[delimiter.end():].lstrip(" ,;:-")
            if (
                24 <= len(suffix) <= MAX_ALLOWED_CLAIM_CHARS
                and balanced_delimiters(suffix)
                and _PREDICATE_RE.search(suffix)
                and _NUMBER_WITH_OPTIONAL_UNIT_RE.search(suffix)
                and (
                    _COUNT_UNIT_RE.search(suffix)
                    or re.search(r"%|\bpercent(?:age)?\b|אחוז", suffix, re.I)
                )
            ):
                bounded_fragments.append(suffix)
    candidates = bounded_fragments + candidates
    output: list[str] = []
    for candidate in candidates:
        candidate = normalize_evidence_whitespace(candidate)
        candidate = re.sub(r"^[•▪◦*-]\s*", "", candidate)
        if (
            12 <= len(candidate) <= MAX_ALLOWED_CLAIM_CHARS
            and candidate not in output
        ):
            output.append(candidate)
    return tuple(output)


def _reconstructed_bullet_claims(text: str) -> tuple[str, ...]:
    """Join layout-wrapped bullet records without merging adjacent sections."""
    value = re.sub(r"\s*([•▪◦])\s*", r"\n\1 ", str(text or ""))
    lines = [
        normalize_evidence_whitespace(line)
        for line in value.splitlines()
    ]
    output: list[str] = []
    current: list[str] | None = None

    def flush() -> None:
        nonlocal current
        if current:
            claim = normalize_evidence_whitespace(" ".join(current))
            if 12 <= len(claim) <= MAX_ALLOWED_CLAIM_CHARS:
                output.append(claim)
        current = None

    for index, line in enumerate(lines):
        if not line:
            continue
        marker = re.match(
            r"^(?:[-*•▪◦]|\d{1,2}[.)])(?:\s+(.*))?$",
            line,
        )
        if marker:
            flush()
            body = normalize_evidence_whitespace(marker.group(1) or "")
            current = [body] if body else []
            continue
        if current is None:
            continue
        following = next(
            (candidate for candidate in lines[index + 1:] if candidate),
            "",
        )
        is_section_heading = (
            bool(current)
            and bool(re.search(r"[.!?׃]\s*$", current[-1]))
            and 1 <= len(_normalized_terms(line)) <= 8
            and not re.search(r"[.!?׃]\s*$", line)
            and bool(re.match(r"^(?:[-*•▪◦]|\d{1,2}[.)])(?:\s|$)", following))
        )
        if is_section_heading:
            flush()
            continue
        current.append(line)
    flush()
    return tuple(output)


def _allowed_claims(
    text: str,
    metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[str, ...]:
    claims = [
        claim
        for claim in _source_claim_candidates(text)
        if classify_structural_unit(claim, metadata) in _FACTUAL_STRUCTURAL_TYPES
    ]
    return tuple(claims[:MAX_ALLOWED_CLAIMS_PER_UNIT])


def _structural_type(
    hit: Mapping[str, Any],
    allowed_claims: Sequence[str] = (),
) -> str:
    metadata = hit.get("metadata") or {}
    types = [
        classify_structural_unit(claim, metadata)
        for claim in allowed_claims
    ]
    if StructuralUnitType.TABLE_ROW in types:
        return StructuralUnitType.TABLE_ROW.value
    if StructuralUnitType.CHART_DATA in types:
        return StructuralUnitType.CHART_DATA.value
    if StructuralUnitType.BULLET in types or metadata.get("has_bullet_context"):
        return StructuralUnitType.BULLET.value
    if StructuralUnitType.LIST in types:
        return StructuralUnitType.LIST.value
    return StructuralUnitType.BODY.value


def _normalize_structural_evidence_text(value: Any) -> str:
    """Normalize each source line while preserving list and table boundaries."""
    lines = [
        normalize_evidence_whitespace(line)
        for line in str(value or "").splitlines()
    ]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def _safe_int(value: Any) -> Optional[int]:
    try:
        output = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return output if output > 0 else None


def _safe_float(value: Any) -> Optional[float]:
    try:
        output = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return output if math.isfinite(output) else None


def _governing_heading_for_unit(unit: EvidenceUnit) -> tuple[str, tuple[int, int]]:
    """Find only an explicit preceding heading in this same stored chunk."""
    structural = unit.structural_evidence_unit
    raw = str(unit.raw_hit.get("document") or "")
    if structural is None or not structural.source_spans:
        return "", (0, 0)
    start = min(span.start for span in structural.source_spans)
    prefix = raw[max(0, start - 2048):start]
    headings = list(re.finditer(r"(?m)^[ \t]*(?P<heading>[^\n:]{4,160}:)[ \t]*$", prefix))
    if not headings:
        return "", (0, 0)
    match = headings[-1]
    heading = match.group("heading")
    offset = max(0, start - 2048)
    return heading, (offset + match.start("heading"), offset + match.end("heading"))


def _source_semantic_literal_span(proposition: str, value: str) -> tuple[int, int]:
    position = proposition.casefold().find(value.casefold())
    if position >= 0:
        return position, position + len(value)
    # Semantic whitespace normalization must not turn a field-owned value
    # quote into the entire proposition. Keep its original bytes and offsets.
    tokens = value.split()
    matches = list(re.finditer(r"\s+".join(re.escape(token) for token in tokens), proposition, re.I)) if tokens else []
    return matches[0].span() if len(matches) == 1 else (0, len(proposition))


def _evidence_context_for_proposition(
    unit: EvidenceUnit,
    proposition: str,
    start: int,
    end: int,
    semantics: Mapping[str, Any],
    *,
    evidence_unit_value: str = "",
    numeric_role: str = "",
) -> EvidenceContext:
    """Build evidence semantics before applying a request requirement."""
    metadata = dict(unit.raw_hit.get("metadata") or {})
    structural_unit = unit.structural_evidence_unit
    context_start, context_end = (
        (start, end)
        if structural_unit is not None
        else (0, len(proposition))
    )
    if structural_unit is not None:
        if structural_unit.structural_type in {
            StructuralEvidenceType.TABLE_ROW,
            StructuralEvidenceType.CHART_CATEGORY_RECORD,
        }:
            # Row/category identity comes only from the bounded structural
            # unit. A chunk-level row label may belong to a neighboring row.
            for key in (
                "row_header", "table_row_header", "category", "facet_label",
                "series_label",
            ):
                metadata.pop(key, None)
            if structural_unit.row_label:
                metadata["row_header"] = structural_unit.row_label
                metadata["category"] = structural_unit.row_label
    relation_arguments = dict(semantics.get("relation_arguments") or {})
    evidence_semantics = {
        "subject": str(semantics.get("subject") or ""),
        "predicate": str(semantics.get("predicate") or ""),
        "target_entity": str(semantics.get("target_entity") or ""),
        "unit": str(evidence_unit_value or semantics.get("unit") or ""),
        "numeric_role": str(numeric_role or semantics.get("numeric_role") or ""),
        "polarity": str(semantics.get("polarity") or "neutral"),
        "period": str(semantics.get("period") or ""),
        "population": str(semantics.get("population") or ""),
        "category": str(semantics.get("category") or ""),
    }
    whole_span = (0, len(proposition))
    semantic_sources: dict[str, dict[str, Any]] = {
        field_name: {
            "provenance_type": "deterministic_same_proposition",
            "source_container": "local_proposition",
            "source_text": proposition,
            "source_span": _source_semantic_literal_span(proposition, str(value)) if str(value) else whole_span,
            "confidence": 0.9,
            "revisability": "deterministic_revisable",
            "derivation_type": (
                "deterministic_semantic_label"
                if field_name in {"predicate", "unit", "numeric_role", "polarity"}
                else "literal_extraction"
            ),
        }
        for field_name, value in evidence_semantics.items()
        if value
    }
    semantic_period = str(evidence_semantics.get("period") or "")
    if (
        semantic_period in {"current", "historical"}
        and semantic_period.casefold() not in proposition.casefold()
    ):
        cue_pattern = (
            _CURRENT_TIME_RE
            if semantic_period == "current" else _HISTORICAL_TIME_RE
        )
        cue = cue_pattern.search(unit.canonical_text)
        if cue is not None:
            literal = next((
                token for token in re.findall(
                    r"[A-Za-z\u05d0-\u05ea]+", cue.group(0)
                )
                if (
                    _CURRENT_TIME_RE.fullmatch(token)
                    if semantic_period == "current"
                    else _HISTORICAL_TIME_RE.fullmatch(token)
                )
            ), cue.group(0))
            literal_start = unit.canonical_text.casefold().find(
                literal.casefold(), cue.start(), cue.end()
            )
            semantic_sources["period"] = {
                "provenance_type": "governing_time_context",
                "source_container": "structural_parent_context",
                "source_text": unit.canonical_text,
                "source_span": (
                    literal_start, literal_start + len(literal)
                ),
                "confidence": 0.94,
                "revisability": "fixed_source_structure",
                "derivation_type": "literal",
            }
    semantic_population = str(evidence_semantics.get("population") or "")
    if (
        semantic_population
        and semantic_population.casefold() not in proposition.casefold()
    ):
        population_position = unit.canonical_text.casefold().find(
            semantic_population.casefold()
        )
        if population_position >= 0:
            semantic_sources["population"] = {
                "provenance_type": "governing_scope_context",
                "source_container": "structural_parent_context",
                "source_text": unit.canonical_text,
                "source_span": (
                    population_position,
                    population_position + len(semantic_population),
                ),
                "confidence": 0.94,
                "revisability": "fixed_source_structure",
                "derivation_type": "literal",
            }
    for role, argument in relation_arguments.items():
        position = proposition.casefold().find(str(argument).casefold())
        semantic_sources[str(role)] = {
            "provenance_type": "deterministic_relation_argument",
            "source_container": "local_proposition",
            "source_text": proposition,
            "source_span": (
                (position, position + len(str(argument)))
                if position >= 0 else whole_span
            ),
            "confidence": 0.88,
            "revisability": "deterministic_revisable",
            "derivation_type": "deterministic_literal_argument",
        }
    context = build_evidence_context(
        evidence_unit_id=unit.id,
        candidate_id=unit.candidate_id,
        source_id=unit.source_id,
        chunk_id=unit.chunk_id,
        page=unit.page,
        # Pre-C9 compatibility callers may supply only a coarse EvidenceUnit
        # label.  Their bounded proposition is wrapped here as a canonical
        # source unit, so the coarse label must not become a competing type
        # authority.  Live retrieval always carries the canonical unit.
        structural_type=(
            unit.structural_type if structural_unit is not None else "body"
        ),
        local_proposition=proposition,
        proposition_span=(context_start, context_end),
        metadata=metadata,
        evidence_semantics=evidence_semantics,
        semantic_sources=semantic_sources,
        relation_arguments=relation_arguments,
        structural_evidence_unit=structural_unit,
    )
    context = replace(context, source_candidate_text=str(unit.raw_hit.get("document") or ""))
    if structural_unit is not None:
        heading, span = _governing_heading_for_unit(unit)
        if heading:
            context = replace(
                context,
                governing_context=heading,
                governing_context_provenance=FieldProvenance(
                    field="governing_context", value=heading,
                    provenance_type="governing_source_heading",
                    source_container="source_candidate", source_span=span,
                    source_sha256=hashlib.sha256(heading.encode("utf-8")).hexdigest(),
                    confidence=1.0, revisability="fixed_source_structure",
                    source_quote=heading, structural_unit_id=structural_unit.unit_id,
                    structural_group_id=structural_unit.structural_group_id or structural_unit.unit_id,
                    structural_role="governing_heading",
                ),
            )
    return context


EvidenceSemanticResolver = Callable[
    [EvidenceContext],
    Optional[Mapping[str, Any] | EvidenceSemanticResolverResult],
]


def _maybe_resolve_evidence_semantics(
    unit: EvidenceUnit,
    context: EvidenceContext,
    resolver: Optional[EvidenceSemanticResolver],
    attempted_unit_ids: set[str],
) -> Optional[EvidenceContext]:
    """Run at most one evidence-only resolver call for this evidence unit."""
    if resolver is None or unit.id in attempted_unit_ids:
        _c45_application_value = None
        _c45_observe('C44E2E2_FAMILY_08_ffcc7a246f0b')
        return _c45_application_value
    # Retrieval has already selected this authorized unit. Require a bounded
    # proposition and either a retrieval score/channel or structural label so
    # arbitrary nearby text can never trigger semantic promotion.
    structurally_grounded = bool(
        context.section_title
        or context.table_chart_title
        or context.row_header
        or context.column_header
    )
    if not (
        4 <= len(context.local_proposition) <= MAX_ALLOWED_CLAIM_CHARS
        and unit.candidate_id
        and (unit.score is not None or unit.channels or structurally_grounded)
    ):
        _c45_application_value = None
        _c45_observe('C44E2E2_FAMILY_08_ffcc7a246f0b')
        return _c45_application_value
    attempted_unit_ids.add(unit.id)
    recorder = current_eval_trace()
    diagnostics: dict[str, Any] = {}
    try:
        outcome = resolver(context)
        if isinstance(outcome, EvidenceSemanticResolverResult):
            diagnostics = dict(outcome.diagnostics)
            if outcome.resolution is None:
                if recorder is not None and hasattr(
                    recorder, "record_model_assisted_semantic_attempt"
                ):
                    recorder.record_model_assisted_semantic_attempt(
                        evidence_id=unit.id,
                        accepted=False,
                        reason_code=outcome.reason_code,
                        diagnostics=diagnostics,
                    )
                _c45_application_value = None
                _c45_observe('C44E2E2_FAMILY_08_ffcc7a246f0b')
                return _c45_application_value
            resolution = outcome.resolution
        else:
            resolution = outcome
        if not isinstance(resolution, Mapping):
            raise EvidenceSemanticValidationError("resolver_no_mapping")
        resolved = apply_model_assisted_semantics(context, resolution)
        relation_errors = relation_argument_validation_errors(resolved)
        if relation_errors:
            raise EvidenceSemanticValidationError(relation_errors[0])
    except Exception as exc:
        reason_code = (
            exc.reason_code
            if isinstance(exc, EvidenceSemanticValidationError)
            else f"resolver_internal_error:{type(exc).__name__}"
        )
        diagnostics.update({
            "resolver_failure_stage": (
                "quote_validation"
                if "quote" in reason_code or "literal" in reason_code
                else "fact_validation"
                if reason_code.startswith("relation_argument_")
                else "schema_validation"
                if reason_code.startswith("resolver_")
                else "runtime"
            ),
            "resolver_schema_status": "rejected",
            "resolver_quote_status": (
                "rejected" if "quote" in reason_code or "literal" in reason_code
                else diagnostics.get("resolver_quote_status", "not_reached")
            ),
            "resolver_fact_status": (
                "rejected" if reason_code.startswith("relation_argument_")
                else diagnostics.get("resolver_fact_status", "not_reached")
            ),
        })
        if recorder is not None and hasattr(
            recorder, "record_model_assisted_semantic_attempt"
        ):
            recorder.record_model_assisted_semantic_attempt(
                evidence_id=unit.id,
                accepted=False,
                reason_code=reason_code,
                diagnostics=diagnostics,
            )
        _c45_application_value = None
        _c45_observe('C44E2E2_FAMILY_08_ffcc7a246f0b')
        return _c45_application_value
    if recorder is not None and hasattr(
        recorder, "record_model_assisted_semantic_attempt"
    ):
        recorder.record_model_assisted_semantic_attempt(
            evidence_id=unit.id,
            accepted=True,
            reason_code="exact_quotes_uniquely_located",
            diagnostics={
                **diagnostics,
                "resolver_schema_status": "accepted",
                "resolver_quote_status": "accepted",
                "resolver_fact_status": "accepted",
                "resolver_failure_stage": "none",
            },
        )
    _c45_application_value = resolved
    _c45_observe('C44E2E2_FAMILY_08_ffcc7a246f0b')
    return _c45_application_value


def _units_from_hit(
    hit: Mapping[str, Any],
    *,
    subquestion_id: str,
    effective_tenant: str,
    integrity_validator: Callable[[Mapping[str, Any]], bool],
    source_authorizer: Optional[Callable[[Mapping[str, Any], str, str], SourceAuthorization]] = None,
) -> tuple[EvidenceUnit, ...]:
    metadata = dict(hit.get("metadata") or {})
    stored_tenant = metadata.get("tenant_id")
    candidate_identity = semantic_candidate_id(hit)
    recorder = current_eval_trace()
    if stored_tenant is not None and str(stored_tenant) != effective_tenant:
        if recorder is not None:
            recorder.record_source_authorization(candidate_identity, {
                "source_authorization_status": "source_tenant_mismatch",
                "source_authorization_reason": "tenant_mismatch",
            })
        return ()
    raw_text = str(hit.get("document") or "")
    authorization = (
        source_authorizer(metadata, raw_text, str(hit.get("id") or candidate_identity))
        if source_authorizer else None
    )
    if authorization is not None:
        if recorder is not None:
            recorder.record_source_authorization(candidate_identity, authorization.as_dict())
        if not authorization.authorized:
            return ()
    elif not integrity_validator(metadata):
        if recorder is not None:
            recorder.record_source_authorization(candidate_identity, {
                "source_authorization_status": "source_authorization_rejected",
                "source_authorization_reason": "legacy_integrity_validator_rejected",
            })
        return ()
    source_id = str(metadata.get("source_id") or "")
    chunk_id = str(metadata.get("chunk_id") or hit.get("id") or "")
    page = _safe_int(metadata.get("page"))
    if not source_id or not chunk_id or not raw_text.strip():
        if recorder is not None:
            recorder.record_structural_decomposition(
                candidate_identity, status="rejected", reason="candidate_identity_or_text_missing"
            )
        return ()
    # Enter through the source-only boundary before a coarse source-type hint.
    # A page breadcrumb is decoration; it cannot classify unrelated body text.
    structural_entry = canonical_structural_entry(
        raw_text,
        source_id=source_id,
        candidate_id=candidate_identity,
        chunk_id=chunk_id,
        page=page,
        metadata=metadata,
    )
    structural_units, scalar_span_audit = account_scalar_spans(
        raw_text, structural_entry,
    )
    if structural_units != structural_entry.structural_units:
        structural_entry = replace(
            structural_entry, structural_units=structural_units,
        )
    if recorder is not None:
        recorder.record_structural_decomposition(
            candidate_identity,
            status=(
                "units_created" if structural_entry.structural_units else "rejected"
            ),
            reason=(
                "structural_units_created" if structural_entry.structural_units
                else structural_entry.rejection_reason
            ),
            units=structural_entry.structural_units,
            entry=structural_entry,
            scalar_span_audit=scalar_span_audit,
        )
    if not structural_entry.accepted:
        return ()
    section = _section_value(hit)
    section_fingerprint = opaque_evidence_chunk_id(
        source_id,
        section or chunk_id,
        page,
    )
    channels = hit.get("_channels") or (RetrievalChannel.VECTOR,)
    parsed_channels: list[RetrievalChannel] = []
    for channel in channels:
        try:
            parsed = (
                channel
                if isinstance(channel, RetrievalChannel)
                else RetrievalChannel(str(channel))
            )
        except ValueError:
            continue
        if parsed not in parsed_channels:
            parsed_channels.append(parsed)
    structural_units = structural_entry.structural_units
    output: list[EvidenceUnit] = []
    for structural_unit in structural_units:
        text = structural_unit.local_text
        allowed_claims = (text,)
        provisional = EvidenceUnit(
            id=structural_unit.unit_id,
            subquestion_ids=(subquestion_id,),
            source_id=source_id,
            page=page,
            section_fingerprint=section_fingerprint,
            canonical_text=text,
            allowed_claims=allowed_claims,
            channels=tuple(parsed_channels),
            score=_safe_float(
                hit.get("_fusion_score")
                if hit.get("_fusion_score") is not None
                else hit.get("score")
            ),
            score_kind=str(hit.get("score_kind") or "") or None,
            # The bounded structural unit is the only active type authority.
            # A whole-chunk classifier may remain in entry diagnostics, but
            # must never relabel a list/prose/table/chart unit here.
            structural_type=structural_unit.structural_type.value,
            raw_hit=hit,
            candidate_id=candidate_identity,
            chunk_id=chunk_id,
            structural_evidence_unit=structural_unit,
        )
        semantics = _text_evidence_semantics(
            text,
            row_header=structural_unit.row_label,
        )
        context = _evidence_context_for_proposition(
            provisional, text, 0, len(text), semantics
        )
        if not context.structural_binding_valid:
            continue
        output.append(replace(provisional, evidence_context=context))
    return tuple(output)


def _unit_from_hit(
    hit: Mapping[str, Any],
    *,
    subquestion_id: str,
    effective_tenant: str,
    integrity_validator: Callable[[Mapping[str, Any]], bool],
) -> Optional[EvidenceUnit]:
    """Compatibility helper returning the first canonical structural unit."""
    units = _units_from_hit(
        hit,
        subquestion_id=subquestion_id,
        effective_tenant=effective_tenant,
        integrity_validator=integrity_validator,
    )
    return units[0] if units else None


def _merge_units(units: Iterable[EvidenceUnit]) -> tuple[EvidenceUnit, ...]:
    merged: dict[str, EvidenceUnit] = {}
    for unit in units:
        existing = merged.get(unit.id)
        if existing is None:
            merged[unit.id] = unit
            continue
        merged[unit.id] = EvidenceUnit(
            id=unit.id,
            subquestion_ids=tuple(dict.fromkeys(
                existing.subquestion_ids + unit.subquestion_ids
            )),
            source_id=existing.source_id,
            page=existing.page,
            section_fingerprint=existing.section_fingerprint,
            canonical_text=existing.canonical_text,
            allowed_claims=existing.allowed_claims,
            channels=tuple(dict.fromkeys(existing.channels + unit.channels)),
            score=max(
                value for value in (existing.score, unit.score)
                if value is not None
            ) if existing.score is not None or unit.score is not None else None,
            score_kind=existing.score_kind or unit.score_kind,
            structural_type=(
                existing.structural_type
                if existing.structural_type != "paragraph"
                else unit.structural_type
            ),
            raw_hit=existing.raw_hit,
            candidate_id=existing.candidate_id or unit.candidate_id,
            chunk_id=existing.chunk_id or unit.chunk_id,
            evidence_context=existing.evidence_context or unit.evidence_context,
            structural_evidence_unit=(
                existing.structural_evidence_unit
                or unit.structural_evidence_unit
            ),
            source_evidence_unit_ids=tuple(dict.fromkeys(
                existing.source_evidence_unit_ids + unit.source_evidence_unit_ids
            )),
        )
    return tuple(merged.values())


def _bounded_adjacent_evidence_units(
    units: Sequence[EvidenceUnit],
) -> tuple[EvidenceUnit, ...]:
    """Recover only source-contiguous split propositions for fact extraction.

    This is deliberately downstream of structural entry and only composes two
    canonical units whose original spans are adjacent in the same authorized
    chunk.  It never reintroduces a whole retrieved chunk as evidence.
    """
    grouped: dict[tuple[str, str, str, Optional[int]], list[EvidenceUnit]] = {}
    for unit in units:
        structural = unit.structural_evidence_unit
        if structural is None or not unit.raw_hit:
            continue
        grouped.setdefault(
            (unit.source_id, unit.candidate_id, unit.chunk_id, unit.page), []
        ).append(unit)
    recovered: list[EvidenceUnit] = []
    existing_ids = {unit.id for unit in units}
    for members in grouped.values():
        ordered = sorted(
            members,
            key=lambda item: min(
                (span.start for span in item.structural_evidence_unit.source_spans),
                default=0,
            ),
        )
        for left, right in zip(ordered, ordered[1:]):
            left_structural = left.structural_evidence_unit
            right_structural = right.structural_evidence_unit
            if left_structural is None or right_structural is None:
                continue
            raw_text = str(left.raw_hit.get("document") or "")
            if not raw_text or raw_text != str(right.raw_hit.get("document") or ""):
                continue
            structural = bounded_adjacent_structural_unit(
                raw_text, (left_structural, right_structural)
            )
            if structural is None or structural.unit_id in existing_ids:
                continue
            provisional = EvidenceUnit(
                id=structural.unit_id,
                subquestion_ids=tuple(dict.fromkeys(
                    left.subquestion_ids + right.subquestion_ids
                )),
                source_id=left.source_id,
                page=left.page,
                section_fingerprint=left.section_fingerprint,
                canonical_text=structural.local_text,
                allowed_claims=(structural.local_text,),
                channels=tuple(dict.fromkeys(left.channels + right.channels)),
                score=max(
                    value for value in (left.score, right.score)
                    if value is not None
                ) if left.score is not None or right.score is not None else None,
                score_kind=left.score_kind or right.score_kind,
                structural_type=structural.structural_type.value,
                raw_hit=left.raw_hit,
                candidate_id=left.candidate_id,
                chunk_id=left.chunk_id,
                structural_evidence_unit=structural,
                source_evidence_unit_ids=tuple(dict.fromkeys(
                    (*left.source_evidence_unit_ids, *right.source_evidence_unit_ids)
                )),
            )
            context = _evidence_context_for_proposition(
                provisional,
                structural.local_text,
                0,
                len(structural.local_text),
                _text_evidence_semantics(structural.local_text),
            )
            if context.structural_binding_valid:
                recovered.append(replace(provisional, evidence_context=context))
                existing_ids.add(structural.unit_id)
    return tuple(recovered)


def _structural_expand(
    retriever: Any,
    selected: Sequence[Mapping[str, Any]],
    *,
    profile: RequestIntent,
) -> tuple[list[dict[str, Any]], float, float]:
    started = time.monotonic()
    output: list[dict[str, Any]] = []
    scans: dict[str, list[dict[str, Any]]] = {}
    for anchor in selected[:MAX_STRUCTURAL_ANCHORS]:
        source = _source_value(anchor)
        if not source:
            continue
        if source not in scans:
            try:
                records, complete = retriever.scan_records(
                    limit=MAX_STRUCTURAL_RECORDS,
                    scan_limit=MAX_STRUCTURAL_SCAN,
                    where={"source": source},
                )
            except Exception:
                records, complete = [], False
            scans[source] = list(records) if complete else []
        records = scans[source]
        if not records:
            continue
        anchor_index = _record_index(anchor)
        section = _section_value(anchor)
        candidates: list[tuple[RetrievalChannel, Mapping[str, Any]]] = []
        for record in records:
            index = _record_index(record)
            record_section = _section_value(record)
            if (
                anchor_index is not None
                and index is not None
                and abs(index - anchor_index) <= 2
            ):
                candidates.append((RetrievalChannel.ADJACENT, record))
            if section and record_section == section:
                channel = (
                    RetrievalChannel.PARENT_LIST
                    if _structural_type(record) == "bullet_or_list"
                    else RetrievalChannel.SECTION
                )
                candidates.append((channel, record))
        for channel, record in candidates[:MAX_STRUCTURAL_SIBLINGS]:
            expanded = dict(record)
            expanded["_channels"] = (channel,)
            output.append(expanded)
    expansion_ms = (time.monotonic() - started) * 1000.0

    hop_started = time.monotonic()
    multi_hop: list[dict[str, Any]] = []
    if profile in {
        RequestIntent.LIST_WITH_ATTRIBUTES,
        RequestIntent.MULTI_PART_COMPARISON,
        RequestIntent.EXPLANATION,
        RequestIntent.CROSS_SOURCE_SYNTHESIS,
    }:
        # One traceable hop is the already bounded same-section/adjacent set.
        for hit in output[:MAX_STRUCTURAL_SIBLINGS]:
            hopped = dict(hit)
            channels = tuple(hit.get("_channels") or ())
            hopped["_channels"] = tuple(dict.fromkeys(
                channels + (RetrievalChannel.MULTI_HOP,)
            ))
            multi_hop.append(hopped)
    multi_hop_ms = (time.monotonic() - hop_started) * 1000.0
    return output + multi_hop, expansion_ms, multi_hop_ms


def _facet_families(value: str) -> frozenset[str]:
    families: set[str] = set()
    if _EXPERIENCE_FACET_RE.search(value):
        families.add("experience")
    if _EDUCATION_FACET_RE.search(value):
        families.add("education")
    if _FORECAST_FACET_RE.search(value):
        families.add("forecast")
    return frozenset(families)


def _comparison_endpoint_specs(
    subquestion: Subquestion,
) -> tuple[tuple[str, str], ...]:
    periods = tuple(
        value
        for value in subquestion.required_dimensions
        if _YEAR_RE.fullmatch(str(value))
    )
    if len(periods) >= 2:
        return tuple(
            (
                period,
                (
                    "historical"
                    if index == 0
                    else "current"
                    if index == len(periods) - 1
                    else "intermediate"
                ),
            )
            for index, period in enumerate(periods)
        )
    return (("", "historical"), ("", "current"))


def _canonical_requirement_specs(
    legacy_specs: Sequence[CoverageRequirement],
    execution_contract: RequestExecutionContract,
) -> tuple[CoverageRequirement, ...]:
    """Map origin retrieval slots one-to-one onto canonical req-* IDs."""
    unused = set(range(len(legacy_specs)))
    output: list[CoverageRequirement] = []
    for requirement in execution_contract.requirements:
        candidates: list[tuple[int, int]] = []
        for index in unused:
            spec = legacy_specs[index]
            score = 0
            if spec.kind == requirement.kind:
                score += 12
            if (
                spec.kind == "comparison_endpoint"
                and requirement.kind == "comparison_endpoint"
            ):
                score += 10
            if spec.period and spec.period in requirement.periods:
                score += 8
            if spec.expected_unit and any(
                spec.expected_unit.casefold() in str(unit).casefold()
                or str(unit).casefold() in spec.expected_unit.casefold()
                for unit in requirement.units
            ):
                score += 5
            requested_text = " ".join((
                requirement.subject,
                requirement.predicate,
                " ".join(requirement.objects_or_metrics),
            ))
            score += round(6 * max(
                _term_overlap(spec.metric or spec.label, requested_text),
                _semantic_term_overlap(spec.metric or spec.label, requested_text),
            ))
            clause_number = re.search(r"(\d+)$", requirement.clause_id)
            subquestion_number = re.search(r"(\d+)", spec.subquestion_id)
            if (
                clause_number
                and subquestion_number
                and clause_number.group(1) == subquestion_number.group(1)
            ):
                score += 4
            candidates.append((score, index))
        selected = max(candidates, default=(-1, -1))[1]
        if selected >= 0:
            unused.discard(selected)
            origin = legacy_specs[selected]
        else:
            origin = CoverageRequirement(
                id=requirement.id,
                subquestion_id=requirement.clause_id.replace("clause-", "sq"),
                kind=requirement.kind,
                label=requirement.subject or requirement.predicate,
                status=CoverageStatus.MISSING,
                evidence_ids=(),
            )
        requested_metric = (
            requirement.objects_or_metrics[0]
            if requirement.objects_or_metrics
            else ""
        )
        # ``count`` and ``percentage`` describe the answer shape, not the
        # requested predicate.  Keep the concrete subject for evidence
        # association so a sentence such as "27 sites reported outages" does
        # not need to contain the literal word "count".
        concrete_target = (
            requirement.counted_entity or requirement.target_entity_type
        )
        metric = concrete_target or (
            # Preserve the source-language request subject for a scalar
            # lookup.  The contract's normalized metric remains authoritative
            # for compatibility, but this adapter needs the literal request
            # phrasing to bridge cross-language evidence without inventing a
            # second semantic interpretation.
            requirement.subject
            if requirement.kind == "lookup"
            and requirement.value_type == "numeric"
            and requirement.subject
            else requirement.subject or origin.metric
            if requested_metric in {"count", "percentage"}
            else requested_metric or requirement.subject or origin.metric
        )
        expected_unit = (
            requirement.units[0]
            if requirement.units
            # A numeric lookup without an explicitly requested unit must not
            # inherit a legacy count default from its route-shaped plan.
            else ""
            if requirement.value_type in {"numeric", "currency", "period"}
            else origin.expected_unit
        )
        numeric_kinds = {
            "count", "percentage", "comparison_endpoint",
            "category_value_bindings", "winner", "winner_attribute",
            "common_metric_unit_period_population",
        }
        value_bearing_types = {
            "numeric", "count", "percentage", "currency", "period",
        }
        output.append(replace(
            origin,
            id=requirement.id,
            origin_id=origin.origin_id or origin.id,
            requirement_ids=(requirement.id,),
            kind=requirement.kind,
            label=(metric or requirement.predicate or origin.label)[:120],
            metric=metric,
            period=(requirement.periods[0] if requirement.periods else origin.period),
            facet=(requirement.facet if requirement.facet.startswith("experience_band:") else origin.facet),
            requires_qualifier=(origin.requires_qualifier or requirement.facet.startswith("experience_band:")),
            expected_unit=str(expected_unit or ""),
            population=(
                requirement.scope_population
                or (
                    requirement.populations[0]
                    if requirement.populations else origin.population
                )
            ),
            # The immutable contract carries output type independently of its
            # execution route.  A lookup can therefore require a numeric
            # fact; treating all lookup requirements as propositions would
            # bypass the existing typed-value and scope gates.
            requires_value=(
                requirement.satisfaction_mode
                is RequirementSatisfactionMode.SOURCE_FACT
                and (
                    requirement.kind in numeric_kinds
                    or requirement.value_type in value_bearing_types
                )
            ),
            satisfaction_mode=requirement.satisfaction_mode.value,
            scalar_role=normalize_scalar_role(
                getattr(requirement, "scalar_role", "")
            ),
            quantity_kind=normalize_quantity_kind(
                getattr(requirement, "quantity_kind", "")
            ),
            statistical_role=normalize_statistical_role(
                getattr(requirement, "statistical_role", "")
            ),
            aggregation_role=normalize_aggregation_role(
                getattr(requirement, "aggregation_role", "")
            ),
        ))
    return tuple(output)


def derive_evidence_requirements(
    plan: RequestPlan,
    execution_contract: Optional[RequestExecutionContract] = None,
) -> tuple[CoverageRequirement, ...]:
    """Translate a request plan into mandatory evidence slots before retrieval."""
    if plan.intent is RequestIntent.LIST_WITH_ATTRIBUTES:
        return ()
    output: list[CoverageRequirement] = []
    for subquestion in plan.subquestions:
        is_comparison = subquestion.question_type is QuestionType.COMPARISON
        if not is_comparison:
            dimensions = subquestion.required_dimensions or ("",)
            for index, dimension in enumerate(dimensions, start=1):
                dimension_key = str(dimension).casefold()
                value_requested = bool(
                    subquestion.question_type is QuestionType.COUNT
                    or dimension
                    or _VALUE_REQUEST_RE.search(subquestion.query)
                )
                expected_unit = (
                    "percentage"
                    if dimension_key == "percentage"
                    or re.search(
                        r"(?:%|\bpercent(?:age)?\b|"
                        r"\b(?:confidence\s+level|rate|share)\b|"
                        r"אחוז|שיעור|נתח|רמת\s+ביטחון)",
                        str(dimension or subquestion.query),
                        re.IGNORECASE,
                    )
                    else "count"
                    if dimension_key == "count"
                    or subquestion.question_type is QuestionType.COUNT
                    or _COUNT_METRIC_RE.search(subquestion.subject)
                    else "numeric"
                )
                # Request quantity cues must come from this independently
                # parsed request unit.  A direct fact clause can express a
                # count as "how many" even when request understanding has
                # not named the route COUNT.  Conversely, an explicit output
                # dimension is its own request unit: carrying a sibling
                # ``percentage`` slot into a ``number`` slot would make the
                # latter incorrectly require a percentage.  Neither form
                # reads the original compound query, so adjacent clauses
                # cannot leak aggregation cues across the boundary.
                quantity_request_text = str(dimension or subquestion.query)
                quantity_semantics = infer_request_quantity_semantics(
                    quantity_request_text,
                    expected_unit=expected_unit,
                    has_facet=False,
                )
                if (
                    expected_unit == "numeric"
                    and quantity_semantics.quantity_kind
                    == QuantityKind.COUNT.value
                ):
                    expected_unit = "count"
                time_role = (
                    "current"
                    if _CURRENT_TIME_RE.search(subquestion.query)
                    else "historical"
                    if _HISTORICAL_TIME_RE.search(subquestion.query)
                    else ""
                )
                output.append(CoverageRequirement(
                    id=(
                        f"{subquestion.id}_{dimension}"
                        if dimension
                        else subquestion.id
                    ),
                    subquestion_id=subquestion.id,
                    kind=subquestion.question_type.value.casefold(),
                    label=str(dimension or subquestion.subject)[:120],
                    status=CoverageStatus.MISSING,
                    evidence_ids=(),
                    metric=subquestion.subject,
                    facet=(
                        dimension_key
                        if dimension_key in {"count", "percentage"}
                        else ""
                    ),
                    time_role=time_role,
                    expected_unit=expected_unit,
                    requires_value=value_requested,
                    scalar_role=infer_request_scalar_role(
                        str(dimension or subquestion.query),
                        expected_unit=expected_unit,
                        has_facet=bool(dimension),
                    ),
                    quantity_kind=quantity_semantics.quantity_kind,
                    statistical_role=quantity_semantics.statistical_role,
                    aggregation_role=quantity_semantics.aggregation_role,
                ))
            continue

        experience_dimension = bool(
            _EXPERIENCE_FACET_RE.search(subquestion.subject)
        )
        scalar_experience_level = bool(re.search(
            r"\bexperience\s+levels?\b|רמות\s+(?:ה)?ניסיון",
            subquestion.subject,
            re.IGNORECASE,
        ))
        distribution = bool(
            _DISTRIBUTION_DIMENSION_RE.search(subquestion.subject)
            or (
                _LOWER_BAND_RE.search(subquestion.subject)
                and _HIGHER_BAND_RE.search(subquestion.subject)
            )
            or (
                experience_dimension
                and not scalar_experience_level
                and plan.intent is RequestIntent.MULTI_PART_COMPARISON
            )
        )
        facets = (
            ("lower_band", "higher_band")
            if distribution
            else ("metric",)
        )
        expected_unit = (
            "percentage"
            if distribution
            or re.search(
                r"(?:%|\b(?:percent(?:age)?|rate|share)\b|אחוז|שיעור|נתח)",
                subquestion.subject,
                re.IGNORECASE,
            )
            else "count"
            if _COUNT_METRIC_RE.search(subquestion.subject)
            else "numeric"
        )
        for facet in facets:
            quantity_semantics = infer_request_quantity_semantics(
                subquestion.subject,
                expected_unit=expected_unit,
                has_facet=facet != "metric",
            )
            for endpoint_index, (period, time_role) in enumerate(
                _comparison_endpoint_specs(subquestion),
                start=1,
            ):
                endpoint_label = period or time_role
                output.append(CoverageRequirement(
                    id=(
                        f"{subquestion.id}_{facet}_{endpoint_label}"
                    ),
                    subquestion_id=subquestion.id,
                    kind="comparison_endpoint",
                    label=endpoint_label[:120],
                    status=CoverageStatus.MISSING,
                    evidence_ids=(),
                    metric=subquestion.subject,
                    facet=facet,
                    period=period,
                    time_role=time_role,
                    expected_unit=expected_unit,
                    requires_value=True,
                    requires_qualifier=distribution,
                    scalar_role=infer_request_scalar_role(
                        subquestion.query,
                        expected_unit=expected_unit,
                        has_facet=facet != "metric",
                    ),
                    quantity_kind=quantity_semantics.quantity_kind,
                    statistical_role=quantity_semantics.statistical_role,
                    aggregation_role=quantity_semantics.aggregation_role,
                ))
    legacy = tuple(output)
    return (
        _canonical_requirement_specs(legacy, execution_contract)
        if execution_contract is not None
        else legacy
    )


def _requirement_retrieval_tasks(
    plan: RequestPlan,
    requirements: Sequence[CoverageRequirement],
) -> tuple[tuple[str, Subquestion], ...]:
    """Create one bounded retrieval task per metric/facet requirement group."""
    output: list[tuple[str, Subquestion]] = []
    for subquestion in plan.subquestions:
        matching_requirements = tuple(
            requirement for requirement in requirements
            if requirement.subquestion_id == subquestion.id
        )
        facets = tuple(dict.fromkeys(
            requirement.facet
            for requirement in requirements
            if requirement.subquestion_id == subquestion.id
            and requirement.requires_value
        ))
        if not facets:
            output.append((
                matching_requirements[0].id
                if matching_requirements
                and matching_requirements[0].id.startswith("req-")
                else subquestion.id,
                subquestion,
            ))
            continue
        for facet in facets:
            facet_band = facet.split(":")[1] if facet.startswith("experience_band:") else facet
            if facet_band == "higher_band":
                focus = (
                    "higher upper more than experienced senior "
                    "מעל יותר מנוסים בכירים"
                )
            elif facet_band == "lower_band":
                focus = (
                    "lower less than up to entry level "
                    "עד פחות מתחילים זוטרים"
                )
            elif facet == "count":
                focus = (
                    "count number employees workers positions "
                    "מספר כמות עובדים משרות"
                )
            elif facet == "percentage":
                focus = "percentage percent share rate אחוז שיעור נתח"
            else:
                focus = "total overall value סך הכול כלל ערך"
            facet_requirements = tuple(
                requirement for requirement in matching_requirements
                if requirement.facet == facet
            )
            output.append((
                facet_requirements[0].id
                if facet_requirements
                and facet_requirements[0].id.startswith("req-")
                else subquestion.id,
                replace(
                    subquestion,
                    id=f"{subquestion.id}:{facet}",
                    query=normalize_request_text(
                        f"{subquestion.query} Focus requirement: {focus}"
                    )[:360],
                ),
            ))
    return tuple(output)


def _aggregate_retrieval_diagnostics(
    diagnostics: Sequence[SubquestionRetrieval],
    plan: RequestPlan,
) -> list[SubquestionRetrieval]:
    output: list[SubquestionRetrieval] = []
    for subquestion in plan.subquestions:
        group = [
            item
            for item in diagnostics
            if item.subquestion_id == subquestion.id
        ]
        if not group:
            continue
        output.append(SubquestionRetrieval(
            subquestion_id=subquestion.id,
            query_count=sum(item.query_count for item in group),
            vector_candidates=sum(item.vector_candidates for item in group),
            lexical_candidates=sum(item.lexical_candidates for item in group),
            exact_candidates=sum(item.exact_candidates for item in group),
            fused_candidates=sum(item.fused_candidates for item in group),
            selected_evidence_ids=tuple(dict.fromkeys(
                evidence_id
                for item in group
                for evidence_id in item.selected_evidence_ids
            )),
            query_rewrite_ms=sum(item.query_rewrite_ms for item in group),
            vector_ms=sum(item.vector_ms for item in group),
            lexical_ms=sum(item.lexical_ms for item in group),
            exact_ms=sum(item.exact_ms for item in group),
            fusion_ms=sum(item.fusion_ms for item in group),
            structural_expansion_ms=sum(
                item.structural_expansion_ms for item in group
            ),
            multi_hop_ms=sum(item.multi_hop_ms for item in group),
        ))
    return output


def _metric_compatible(
    subject: str,
    claim: str,
    *,
    expected_unit: str,
    require_specific_focus: bool = True,
    owned_subject: str = "",
    owned_facet: str = "",
    source_metric: str = "",
) -> bool:
    requested_families = _facet_families(subject)
    if owned_facet.startswith("experience_band:"):
        # The caller has independently proved this precise canonical facet.
        # Its family is not a second, metric-owned equality constraint.
        requested_families = requested_families | {"experience"}
        if not set(explicit_metric_concepts(subject)).issubset(
            explicit_metric_concepts(source_metric)
        ):
            return False
    claim_families = _facet_families(claim)
    # A forecast marker can be an independently required canonical subject
    # rather than part of the metric (for example, ``forecast`` + ``demand``).
    # It remains strict: this exception is available only after that source
    # subject has independently matched, never from lexical overlap alone.
    owned_subject_families = _facet_families(owned_subject)
    if (
        "forecast" in claim_families
        and "forecast" not in requested_families
        and "forecast" not in owned_subject_families
    ):
        return False
    if (
        claim_families & {"education", "experience"}
        and not (claim_families & requested_families)
    ):
        return False
    requested_modifiers = {
        name
        for name, pattern in _SCALAR_MEASURE_MODIFIERS
        if pattern.search(subject)
    }
    claim_modifiers = {
        name
        for name, pattern in _SCALAR_MEASURE_MODIFIERS
        if pattern.search(claim)
    }
    if requested_modifiers and not requested_modifiers.issubset(
        claim_modifiers
    ):
        return False
    if requested_families and not (requested_families & claim_families):
        return False
    if (
        "experience" in requested_families
        and "education" in claim_families
        and "experience" not in claim_families
    ):
        return False
    subject_overlap = max(
        _term_overlap(subject, claim),
        _semantic_term_overlap(subject, claim),
    )
    # Count targets are commonly expressed as either workers or employees.
    # Reuse the contract's bounded canonical entity family rather than
    # treating a broader employment/demand synonym as metric evidence.
    canonical_count_entity_match = (
        expected_unit == "count"
        and any(
            canonical_entity_equivalent(left, right)
            for left in _normalized_terms(subject)
            for right in _normalized_terms(claim)
        )
    )
    if canonical_count_entity_match:
        subject_overlap = max(subject_overlap, 0.60)
    subject_count_family = _count_unit_family(subject)
    population_match = bool(
        expected_unit == "count"
        and subject_count_family
        and _count_unit_family(claim) == subject_count_family
        and _TOTAL_METRIC_RE.search(claim)
        and subject_overlap >= 0.25
    )
    if subject_overlap < 0.25 and not population_match:
        return False
    subject_hebrew = len(re.findall(r"[\u05d0-\u05ea]", subject))
    claim_hebrew = len(re.findall(r"[\u05d0-\u05ea]", claim))
    subject_latin = len(re.findall(r"[A-Za-z]", subject))
    claim_latin = len(re.findall(r"[A-Za-z]", claim))
    subject_hebrew_dominant = subject_hebrew >= max(2, subject_latin * 2)
    claim_hebrew_dominant = claim_hebrew >= max(2, claim_latin * 2)
    subject_latin_dominant = subject_latin >= max(2, subject_hebrew * 2)
    claim_latin_dominant = claim_latin >= max(2, claim_hebrew * 2)
    same_script = (
        subject_hebrew_dominant
        and claim_hebrew_dominant
        or subject_latin_dominant
        and claim_latin_dominant
    )
    focus_groups = _metric_focus_groups(subject)
    if (
        require_specific_focus
        and
        same_script
        and len(focus_groups) >= 2
        and not _metric_focus_spans(subject, claim)
        and not population_match
    ):
        return False
    if (
        _TOTAL_METRIC_RE.search(subject)
        and not _TOTAL_METRIC_RE.search(claim)
        and (
            _facet_families(claim)
            or re.search(r"%|\bpercent(?:age)?\b|אחוז", claim, re.I)
        )
    ):
        return False
    if expected_unit == "percentage" and not re.search(
        r"%|\bpercent(?:age)?\b|אחוז",
        claim,
        re.IGNORECASE,
    ):
        return False
    return True


def _metric_focus_groups(value: str) -> tuple[frozenset[str], ...]:
    """Return ordered content-token variants for a population or role."""
    groups: list[frozenset[str]] = []
    for raw in _TOKEN_RE.findall(normalize_request_text(value).casefold()):
        if _YEAR_RE.fullmatch(raw):
            continue
        variants = {raw}
        if raw.isascii() and len(raw) >= 5:
            variants.add(raw[:-1] if raw.endswith("s") else f"{raw}s")
        if (
            len(raw) >= 4
            and re.fullmatch(r"[\u05d0-\u05ea]+", raw)
            and raw[0] in _HEBREW_PREFIXES
        ):
            variants.add(raw[1:])
        if variants & (_GENERIC_METRIC_TERMS | _STOPWORDS):
            continue
        if all(len(item) < 3 for item in variants):
            continue
        group = frozenset(variants)
        if not groups or group != groups[-1]:
            groups.append(group)
    return tuple(groups)


def _metric_focus_terms(value: str) -> frozenset[str]:
    return frozenset(
        term
        for group in _metric_focus_groups(value)
        for term in group
    )


def _metric_focus_matches(
    metric: str,
    region: str,
) -> tuple[tuple[int, int, int, int], ...]:
    focus_groups = _metric_focus_groups(metric)
    if not focus_groups:
        return ()
    tokens: list[tuple[int, int, frozenset[str]]] = []
    for match in _TOKEN_RE.finditer(region):
        token = match.group(0).casefold()
        variants = {token}
        if token.isascii() and len(token) >= 5:
            variants.add(
                token[:-1] if token.endswith("s") else f"{token}s"
            )
        if (
            len(token) >= 4
            and re.fullmatch(r"[\u05d0-\u05ea]+", token)
            and token[0] in _HEBREW_PREFIXES
        ):
            variants.add(token[1:])
        if (
            _YEAR_RE.fullmatch(token)
            or variants & (_GENERIC_METRIC_TERMS | _STOPWORDS)
        ):
            continue
        tokens.append((match.start(), match.end(), frozenset(variants)))

    matches: list[tuple[int, int, int, int]] = []
    if len(focus_groups) == 1:
        return tuple(
            (start, end, 1, 0)
            for start, end, variants in tokens
            if variants & focus_groups[0]
        )
    for width in range(len(focus_groups), 1, -1):
        for group_start in range(len(focus_groups) - width + 1):
            group_window = focus_groups[group_start:group_start + width]
            for token_start in range(len(tokens) - width + 1):
                token_window = tokens[token_start:token_start + width]
                if all(
                    variants & group
                    for (_start, _end, variants), group in zip(
                        token_window,
                        group_window,
                    )
                ):
                    matches.append(
                        (
                            token_window[0][0],
                            token_window[-1][1],
                            width,
                            group_start,
                        )
                    )
        if matches:
            break
    return tuple(matches)


def _metric_focus_spans(
    metric: str,
    region: str,
) -> tuple[tuple[int, int], ...]:
    return tuple(
        (start, end)
        for start, end, _width, _group_start in _metric_focus_matches(
            metric,
            region,
        )
    )


def _inside_enumeration(
    region: str,
    focus_start: int,
    candidate_end: int,
) -> bool:
    opening = region.rfind("(", candidate_end, focus_start + 1)
    if opening < candidate_end:
        return False
    closing = region.find(")", opening + 1)
    if closing < 0 or focus_start >= closing:
        return False
    return bool(_ENUMERATION_CUE_RE.search(region[opening + 1:closing]))


def _candidate_clause(
    region: str,
    candidate_match: re.Match[str],
) -> tuple[str, int]:
    boundaries = list(_CLAUSE_BOUNDARY_RE.finditer(region))
    start = max(
        (match.end() for match in boundaries if match.end() <= candidate_match.start()),
        default=0,
    )
    end = min(
        (match.start() for match in boundaries if match.start() >= candidate_match.end()),
        default=len(region),
    )
    return region[start:end], start


def _candidate_in_unrequested_subgroup(
    requirement: CoverageRequirement,
    region: str,
    candidate: tuple[re.Match[str], str, str],
    *,
    leading_context: str = "",
) -> bool:
    # A source-literal population that satisfies an explicit request scope is
    # not an accidental subgroup.  The old aggregate-safety guard predates
    # first-class scope and would otherwise reject the exact scoped fact
    # before typed compatibility can evaluate it.
    if requirement.population and _population_scope_compatible(
        requirement.population,
        _evidence_population(region) or _evidence_population(leading_context),
    ):
        return False
    if (
        requirement.facet in {"lower_band", "higher_band"}
        or _facet_families(requirement.metric) & {"education", "experience"}
        or _SUBGROUP_SCOPE_RE.search(requirement.metric)
    ):
        return False
    clause, clause_start = _candidate_clause(region, candidate[0])
    relative_start = candidate[0].start() - clause_start
    leading = clause[max(0, relative_start - 48):relative_start]
    contextual_leading = normalize_evidence_whitespace(
        f"{leading_context} {leading}"
    )[-72:]
    answer_options = list(_ANSWER_OPTION_RE.finditer(contextual_leading))
    return bool(
        _SUBGROUP_SCOPE_RE.search(clause)
        or _NONRESPONSE_CATEGORY_RE.search(contextual_leading)
        or (
            _POSITIVE_RESPONSE_METRIC_RE.search(requirement.metric)
            and not _NEGATED_RESPONSE_METRIC_RE.search(requirement.metric)
            and answer_options
            and answer_options[-1].group("negative")
        )
    )


def _candidate_metric_association(
    requirement: CoverageRequirement,
    region: str,
    candidate: tuple[re.Match[str], str, str],
) -> int:
    """Score direct modifier/population-to-value association in one claim."""
    candidate_match = candidate[0]
    score = 0
    requested_count_family = _count_unit_family(requirement.metric)
    clause, clause_start = _candidate_clause(region, candidate_match)
    clause_end = clause_start + len(clause)

    if (
        requirement.expected_unit == "count"
        and requested_count_family
        and _count_unit_family(candidate[2]) == requested_count_family
        and not _SUBGROUP_SCOPE_RE.search(requirement.metric)
    ):
        relative_start = candidate_match.start() - clause_start
        leading = clause[max(0, relative_start - 72):relative_start]
        if (
            _TOTAL_METRIC_RE.search(leading)
            and _count_unit_family(candidate[2])
        ):
            score += 5

    def directly_related(between: str) -> bool:
        return bool(
            len(between) <= 56
            and len(_TOKEN_RE.findall(between)) <= 5
            and not _numeric_candidates(
                between,
                requirement.expected_unit,
            )
            and not _CLAUSE_BOUNDARY_RE.search(between)
            and _METRIC_RELATION_RE.search(between)
        )

    for name, pattern in _SCALAR_MEASURE_MODIFIERS:
        if not pattern.search(requirement.metric):
            continue
        matches = [
            match
            for match in pattern.finditer(region)
            if clause_start <= match.start() and match.end() <= clause_end
        ]
        if not matches:
            return -8
        modifier = min(
            matches,
            key=lambda item: min(
                abs(candidate_match.start() - item.end()),
                abs(item.start() - candidate_match.end()),
            ),
        )
        between = (
            region[modifier.end():candidate_match.start()]
            if modifier.end() <= candidate_match.start()
            else region[candidate_match.end():modifier.start()]
            if candidate_match.end() <= modifier.start()
            else ""
        )
        distance = min(
            abs(candidate_match.start() - modifier.end()),
            abs(modifier.start() - candidate_match.end()),
        )
        score += (
            8
            if modifier.end() <= candidate_match.start()
            and directly_related(between)
            else 4
            if distance <= 48
            else 1
            if distance <= 120
            else -4
        )

    all_focus_matches = _metric_focus_matches(requirement.metric, region)
    if not all_focus_matches:
        return score
    focus_matches = [
        match
        for match in all_focus_matches
        if clause_start <= match[0] and match[1] <= clause_end
    ]
    if not focus_matches:
        return score - 8
    focus_start, focus_end, _focus_width, focus_group_start = min(
        focus_matches,
        key=lambda match: min(
            abs(candidate_match.start() - match[1]),
            abs(match[0] - candidate_match.end()),
        ),
    )
    if candidate_match.end() <= focus_start:
        between = region[candidate_match.end():focus_start]
    elif focus_end <= candidate_match.start():
        between = region[focus_end:candidate_match.start()]
    else:
        between = ""
    distance = min(
        abs(candidate_match.start() - focus_end),
        abs(focus_start - candidate_match.end()),
    )
    if directly_related(between):
        score += 5
    elif distance <= 20:
        score += 2
    elif distance <= 96:
        score += 1
    if (
        candidate_match.end() <= focus_start
        and _COUNT_UNIT_RE.search(between)
    ):
        score += 1
    score += min(3, focus_group_start)
    if _inside_enumeration(region, focus_start, candidate_match.end()):
        score -= 7
    return score


def _facet_region(value: str, facet: str) -> str:
    if facet not in {"lower_band", "higher_band"}:
        return value
    target = _LOWER_BAND_RE if facet == "lower_band" else _HIGHER_BAND_RE
    opposing = _HIGHER_BAND_RE if facet == "lower_band" else _LOWER_BAND_RE
    target_matches = list(target.finditer(value))
    if not target_matches:
        return ""
    anchor = target_matches[0]
    directions = list(_CHANGE_DIRECTION_RE.finditer(value))
    nearby_before = [
        match.start()
        for match in directions
        if anchor.start() - 96 <= match.start() <= anchor.start()
    ]
    nearby_after = [
        match.start()
        for match in directions
        if anchor.start() < match.start() <= anchor.end() + 64
    ]
    start = (
        anchor.start()
        if nearby_after
        else max(nearby_before)
        if nearby_before
        else max(0, anchor.start() - 48)
    )
    opposing_after = [
        match.start()
        for match in opposing.finditer(value)
        if match.start() > anchor.start()
    ]
    end = min(opposing_after) if opposing_after else len(value)
    region = value[start:end]
    if opposing_after:
        # The opposing band label is often introduced by a coordinator.  That
        # coordinator belongs to neither bounded proposition and must not make
        # the first source claim end with an incomplete "while the" fragment.
        region = re.sub(
            r"(?:,\s*)?(?:while|whereas|and)\s+(?:the\s+)?$|"
            r"(?:,\s*)?(?:בעוד\s+ש?|ואילו|ו)\s*$",
            "",
            region,
            flags=re.IGNORECASE,
        ).rstrip()
    return region


def _band_qualifier(value: str, facet: str) -> str:
    if facet == "lower_band":
        inclusive = re.search(
            r"(?:up\s+to|at\s+most)\s+"
            r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
            r"(?:\s+years?(?:\s+of\s+experience)?)?|"
            r"(?:עד|לכל\s+היותר)\s*"
            r"(?:\d+|אחת|שנה|שנתיים|שתיים|שתי|שלוש|ארבע|חמש|שש|"
            r"שבע|שמונה|תשע|עשר)(?:\s+(?:שנים|שנות\s+ניסיון))?",
            value,
            re.IGNORECASE,
        )
        if inclusive:
            return normalize_evidence_whitespace(inclusive.group(0))
    patterns = (
        (
            r"(?:more\s+than|over|above|at\s+least)\s+"
            r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
            r"(?:\s+years?(?:\s+of\s+experience)?)?",
            r"(?:מעל|יותר\s+מ|לפחות)\s*"
            r"(?:\d+|אחת|שנה|שנתיים|שתיים|שתי|שלוש|ארבע|חמש|שש|שבע|שמונה|תשע|עשר)"
            r"(?:\s+(?:שנים|שנות\s+ניסיון))?",
        )
        if facet == "higher_band"
        else (
            r"(?:up\s+to|at\s+most|less\s+than|below|under)\s+"
            r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten)"
            r"(?:\s+years?(?:\s+of\s+experience)?)?",
            r"(?:עד|פחות\s+מ|לכל\s+היותר)\s*"
            r"(?:\d+|אחת|שנה|שנתיים|שתיים|שתי|שלוש|ארבע|חמש|שש|שבע|שמונה|תשע|עשר)"
            r"(?:\s+(?:שנים|שנות\s+ניסיון))?",
        )
    )
    for pattern in patterns:
        match = re.search(pattern, value, re.IGNORECASE)
        if match:
            return normalize_evidence_whitespace(match.group(0))
    return facet


def _numeric_candidates(
    value: str,
    expected_unit: str,
) -> list[tuple[re.Match[str], str, str]]:
    output: list[tuple[re.Match[str], str, str]] = []
    for match in _NUMBER_WITH_OPTIONAL_UNIT_RE.finditer(value):
        raw_value = normalize_evidence_whitespace(match.group("value"))
        if _YEAR_RE.fullmatch(raw_value):
            continue
        label_prefix = value[max(0, match.start() - 20):match.start()]
        label_suffix = value[match.end():match.end() + 4]
        if (
            re.search(
                r"(?:\b(?:table|figure|chart|appendix|chapter)|"
                r"(?:טבלה|איור|תרשים|נספח|פרק))\s*$",
                label_prefix,
                re.IGNORECASE,
            )
            and re.match(r"\s*[.:]", label_suffix)
        ):
            continue
        percent = bool(match.group("percent"))
        nearby = value[max(0, match.start() - 28):match.end() + 32]
        immediate = value[max(0, match.start() - 12):match.end() + 16]
        if expected_unit == "percentage" and not percent:
            continue
        if expected_unit == "count" and (
            percent
            or _TIME_UNIT_RE.search(immediate)
            or not _COUNT_UNIT_RE.search(nearby)
        ):
            continue
        # Bind a count to the lexical unit immediately following that number
        # before considering a preceding noun.  Otherwise prose such as
        # "the service unit recruited 32 workers" incorrectly attaches 32 to
        # "unit" and then rejects the real workforce fact.
        following_text = value[match.end():match.end() + 32]
        following_text = re.split(r"[.;׃,\n]", following_text, maxsplit=1)[0]
        following_match = re.match(
            rf"\s*(?P<unit>{_COUNT_UNIT_RE.pattern})",
            following_text,
            re.IGNORECASE,
        )
        following_matches = list(_COUNT_UNIT_RE.finditer(following_text))
        preceding_text = value[max(0, match.start() - 28):match.start()]
        preceding_matches = list(_COUNT_UNIT_RE.finditer(preceding_text))
        preceding_match = next((
            item for item in reversed(preceding_matches)
            if re.fullmatch(
                r"\s*(?::|[—–-])?\s*",
                preceding_text[item.end():],
            )
        ), None)
        unit_match = (
            following_match
            if following_match is not None else
            following_matches[0] if following_matches else
            preceding_match
        )
        duration_match = re.match(rf'\s*(?P<unit>{_TIME_UNIT_RE.pattern})',following_text,re.I)
        from financial_semantics import currency_for_span
        currency = currency_for_span(value, match.start("value"), match.end("value"))
        unit_label = "%" if percent else (currency or (duration_match.group('unit') if duration_match else (
            (
                unit_match.group("unit")
                if following_match is not None else unit_match.group(0)
            )
            if unit_match else "number"
        )))
        output.append((match, raw_value.replace(" ", ""), unit_label))
    # A coordinated period series may state its count unit only once, for
    # example ``40 positions in 2021 and 52 in 2023``.  The second scalar is
    # still a count because the first unit is in the same bounded proposition
    # and the intervening text is solely a date/series connector.  Do not
    # carry a unit over punctuation or free narrative text.
    for index, (match, raw_value, unit_label) in enumerate(output):
        if str(unit_label or "").casefold() not in {"", "number", "numeric"}:
            continue
        for previous_index in range(index - 1, -1, -1):
            previous_match, _previous_value, previous_unit = output[previous_index]
            if str(previous_unit or "").casefold() in {"", "number", "numeric"}:
                continue
            connector = value[previous_match.end():match.start()]
            if len(connector) > 80 or re.search(r"[.;׃\n]", connector):
                break
            # The numeric regex deliberately keeps a following count unit
            # outside the value match.  For a coordinated source-local series
            # (``40 positions in 2021 and 52 in 2023``), consume that already
            # bound unit before validating the connector.  This is not a
            # noun/unit shortcut: a prior scalar in the same proposition has
            # already established the unit association.
            inherited_connector = re.sub(
                rf"^\s*{re.escape(str(previous_unit))}\b",
                "",
                connector,
                count=1,
                flags=re.IGNORECASE,
            )
            if re.fullmatch(
                r"\s*(?:(?:in|during|at|בשנת|בשנה|ב[־-])\s*"
                r"(?:19|20)\d{2}\s*)?"
                r"(?:,?\s*(?:and|to|through|until|ו[־-]?|עד)\s*)?",
                inherited_connector,
                re.IGNORECASE,
            ) and inherited_connector.strip():
                output[index] = (match, raw_value, previous_unit)
            break
    return output


def _count_unit_family(value: str) -> str:
    if re.search(
        r"\b(?:companies|organizations)\b|(?:חברות?|ארגונים?)",
        value,
        re.IGNORECASE,
    ):
        return "organization"
    if re.search(
        r"\b(?:employees?|workers?|positions?|jobs?|vacancies|people|persons?)\b|"
        r"(?:עובדים?|משרות?|מועסקים?|אנשים?)",
        value,
        re.IGNORECASE,
    ):
        return "workforce"
    return ""


def _direction_in_claim(value: str) -> str:
    if re.search(
        r"\b(?:increase|increased|rose|grew|growth)\b|"
        r"(?:עלייה|עליה|עלה|עלתה|גידול|גדל|גדלה)",
        value,
        re.IGNORECASE,
    ):
        return "increase"
    if re.search(
        r"\b(?:decrease|decreased|decline|declined|fell|reduction)\b|"
        r"(?:ירידה|ירד|ירדה|קיטון|פחת|פחתה)",
        value,
        re.IGNORECASE,
    ):
        return "decrease"
    return ""


_EVIDENCE_NEGATION_RE = re.compile(
    r"\b(?:was|were|is|are|had)\s+not\s+(?:approximately\s+|about\s+)?\d|"
    r"(?:לא\s+(?:היה|הייתה|היו)|אינו|אינה)\s+(?:כ[־-]?\s*)?\d|"
    r"\b(?:excluded|omitted|not\s+included|does\s+not\s+include|"
    r"do\s+not\s+include|without|absent)\b|"
    r"(?:לא\s+נכלל|לא\s+נכללה|לא\s+נכללו|הוחרג|הוחרגה|"
    r"אינו\s+כולל|אינה\s+כוללת|אינם\s+כוללים|אינן\s+כוללות|ללא)",
    re.IGNORECASE,
)
_EVIDENCE_POPULATION_RE = re.compile(
    r"\b(?P<en>(?:[A-Za-z0-9'-]+\s+){0,3}(?:population|sample|cohort|subgroup))\b|"
    r"(?P<he>(?:אוכלוסיית?|מדגם|תת.?קבוצה)\s+[\u05d0-\u05ea\"'׳/-]+(?:\s+[\u05d0-\u05ea\"'׳/-]+){0,3})",
    re.IGNORECASE,
)


def _source_purpose_arguments(value: str) -> dict[str, str]:
    """Bind an explicit purpose frame without promoting an inner relation.

    The entire complement remains the purpose argument, including a rejected
    alternative, conditions and exceptions. No request supplies its subject.
    """
    match = re.fullmatch(
        r"(?:the\s+)?(?:purpose|objective|aim)\s+of\s+(?P<target>.+?)\s+"
        r"(?:is|was)\s+(?P<purpose>.+?)\s*[.!]?",
        value.strip(), re.I,
    )
    if match:
        target = match["target"]
        purpose = match["purpose"]
        # A purpose object can itself be excluded from a document. That is
        # an exclusion relation, not a statement of the owner's objective.
        if re.match(r"(?:excluded|omitted|not\s+included)\b",purpose,re.I):
            return {}
    else:
        # Specifications often express the same source-local relation as a
        # labelled requirement ("purpose of X — Y") rather than a copular
        # sentence.  Both arguments must remain literal in this proposition.
        labelled = re.fullmatch(
            r"ה?מטרת\s+(?:של\s+)?(?P<target>.+?)\s*[–—:]+\s*"
            r"(?P<purpose>.+?)\s*[.!]?",
            value.strip(), re.I,
        )
        if labelled:
            target = normalize_request_text(labelled["target"]).strip(" ,;:.?!")
            purpose = normalize_request_text(labelled["purpose"]).strip(" ;:.?!")
            if target and purpose and not re.match(
                r"(?:הוחרג|הוחרגה|לא\s+נכלל|לא\s+נכללה)(?:\s|$)",
                purpose,
            ):
                return {"purpose_target": target, "purpose": purpose}
        match = re.fullmatch(
            r"ה?מטר(?:תו|תה|תם|תן|ת|ה)\s+(?:של\s+)?(?P<target>.+?)\s+"
            r"(?P<copula>היא|הייתה|הוא|היה|אינה|אינו|איננה|איננו|אינן|אינם)\s+"
            r"(?P<purpose>.+?)\s*[.!]?",
            value.strip(), re.I,
        )
        if match is None:
            return {}
        target = match["target"]
        purpose = match["purpose"]
        if match["copula"].startswith("אינ"):
            purpose = match["copula"]+" "+purpose
    target = normalize_request_text(target).strip(" ,;:.?!")
    purpose = normalize_request_text(purpose).strip(" ;:.?!")
    if not target or target.casefold() in {"of", "של"} or not purpose:
        return {}
    return {"purpose_target":target, "purpose":purpose}


def _evidence_predicate(value: str, *, numeric_target: str = "", unit: str = "") -> str:
    # An explicit outer purpose governs even when its complement mentions
    # inclusion, recruitment or a causal/purpose connective of its own.
    if _source_purpose_arguments(value):
        return "purpose_of"
    # A source-local exclusion/filter relation governs the scalar before any
    # incidental workforce, recruitment, or capital vocabulary.  Otherwise a
    # number inside an exclusion criterion can be relabelled as the measured
    # outcome merely because the excluded entity is described by such a verb.
    if _EVIDENCE_NEGATION_RE.search(value):
        return "excluded_from"
    for predicate in DOCUMENT_ROLE_PREDICATES:
        if _relation_arguments(value, predicate):
            return predicate
    if re.search(
        r"\b(?:recruit(?:s|ed|ing)?|hire[sd]?)\b|"
        r"\bhiring\s+(?:workers?|employees?|staff)\b|(?:גייס|גייסה|מגייס)",
        value,
        re.I,
    ):
        return "recruits_employees"
    if re.search(r"\b(?:employs?|employed|workforce)\b|(?:מעסיק|מעסיקה|העסיק)", value, re.I):
        return "employs_workers"
    if re.search(
        r"\b(?:underrepresented|under-represented)\b|(?:תת[־-]?ייצוג)",
        value,
        re.IGNORECASE,
    ):
        return "underrepresented"
    if re.search(
        r"\b(?:corrected|correcting|remedied|adjusted)\b|(?:תוקן|תוקנה|תוקנו)",
        value,
        re.IGNORECASE,
    ):
        return "corrected_by"
    if re.search(r"\b(?:raised|raises?|fundraising|capital\s+raised)\b|(?:גייס(?:ה)?\s+הון)", value, re.I):
        return "raises_capital"
    if re.search(
        r"\b(?:included|includes?|contains?)\b|"
        r"(?<![\u05d0-\u05ea])(?:נכלל|נכללה|נכללו|כולל|כוללת)"
        r"(?![\u05d0-\u05ea])",
        value,
        re.I,
    ):
        return "included_in"
    if re.search(r"\b(?:because|due\s+to|reason|in\s+order\s+to|so\s+that)\b|"
                 r"(?<!\w)(?:בשל|עקב|מפני|משום|כדי|הסיבה)(?!\w)", value, re.I):
        return "caused_by"
    if re.search(r"\bto\s+(?:assist|help)\b", value, re.I):
        return "purpose_of"
    if re.search(r"\b(?:purpose|objective|aim|intended\s+to|designed\s+to|created\s+to|used\s+to)\b|(?:מטר(?:ה|ת)|נועד|נועדה|שימש|שימשה)", value, re.I):
        return "purpose_of"
    if re.search(r"\b(?:means?|defined\s+as|refers?\s+to)\b|(?:מוגדר|מוגדרת|פירושו|משמעותו)", value, re.I):
        return "defined_as"
    if re.search(
        r"\b(?:handles?|manages?|responsible\s+for)\b|"
        r"(?:מטפל(?:ת|ים|ות)?\s+ב|אחראי(?:ת|ים|ות)?\s+על)",
        value,
        re.IGNORECASE,
    ):
        return "performed_by"
    if re.search(
        r"\b(?:identified|determined|measured|selected|constructed|derived|"
        r"performed|implemented|conducted|calculated)\s+"
        r"(?:by|using|through)\b|"
        r"(?:זוהה|זוהתה|נקבע|נקבעה|נמדד|נמדדה|נבחר|נבחרה|נבנה|נבנתה|"
        r"נגזר|נגזרה|בוצע|בוצעה|יושם|יושמה|חושב|חושבה)\s+"
        r"(?:באמצעות|על\s+ידי)",
        value,
        re.IGNORECASE,
    ):
        return "performed_by"
    if re.search(r"\b(?:increase|increased|decrease|decreased|rose|fell|changed|grew|declined)\b|(?:עלה|עלתה|ירד|ירדה|השתנה|שינוי)", value, re.I):
        return "changed_between"
    # A numeric unit is bound to the local atom.  A neighbouring percentage
    # must not turn a worker/organization count into a percentage fact.
    if unit == "%":
        return "has_percentage" if numeric_target else ""
    return "has_count" if numeric_target else ""


def _numeric_evidence_target(
    region: str,
    match: re.Match[str],
    unit_label: str,
    *,
    row_header: str = "",
) -> str:
    from financial_semantics import normalize_currency
    if row_header:
        return row_header
    # A positive from/to source proposition states its measured subject once.
    # Both operand values retain that owner; a following unit is not a new
    # metric, and the change predicate is not a category label.
    change = re.match(
        r'\s*(?P<subject>[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea\x27׳-]*'
        r'(?:\s+[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea\x27׳-]*){0,14})\s+'
        r'(?:increased|decreased|rose|fell|changed|grew|declined)\s+from\s+'
        r'(?:(?:approximately|about|around|roughly|nearly)\s+)?(?=[-+]?\d)',region,re.I)
    if (change and change.end() <= match.start()
            and not re.search(r'\b(?:not|no|never|if|unless|because|denies|denied)\b',change['subject'],re.I)):
        return normalize_request_text(change['subject'])
    if (unit_label and not normalize_currency(unit_label)
            and unit_label.casefold() not in {"number", "numeric", "%", "percent", "percentage"}):
        suffix = region[match.end():match.end() + 100]
        # Preserve a literal modifier that is contiguous with the source
        # count unit (``17 audit labels``).  The unit alone is not the whole
        # source entity and cannot safely stand in for a requested counted
        # entity such as ``audit labels``.
        nominal_phrase = re.match(
            r"\s*(?P<target>(?:"
            r"[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*\s+){0,2}"
            rf"{re.escape(unit_label)})\b",
            suffix,
            re.IGNORECASE,
        )
        if nominal_phrase is not None:
            target = normalize_request_text(nominal_phrase.group("target"))
            return target
        qualified_unit = re.match(
            rf"\s*{re.escape(unit_label)}\s+"
            r"(?!(?:during|within|over|in|for|among|of|that|which|"
            r"במהלך|בתוך|בקרב|של)\b)"
            r"(?P<descriptor>[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*"
            r"(?:\s+[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*){0,2})"
            r"(?=\s+(?:ו?[-־]?\s*\d)|[,.;:]|$)",
            suffix,
            re.IGNORECASE,
        )
        if qualified_unit:
            return normalize_request_text(
                f"{unit_label} {qualified_unit.group('descriptor')}"
            )
        return normalize_request_text(unit_label)
    suffix = region[match.end():match.end() + 90]
    following = re.match(
        r"\s*(?P<target>[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*"
        r"(?:\s+[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*){0,3})"
        r"(?=\s+(?:were|was|are|is|that|which|used|reported|included|excluded|to|for|in|during|within|over|"
        r"היו|היה|הייתה|שימשו|נכללו|דיווחו)\b|[,.;:]|$)",
        suffix,
        re.IGNORECASE,
    )
    if following:
        return normalize_request_text(following.group("target"))
    # A coordinated percentage series can name its denominator after the
    # first endpoint.  Reuse only a unique, explicit partitive subject inside
    # this already bounded proposition, rather than widening to its container.
    partitive_targets = tuple(dict.fromkeys(
        normalize_request_text(item.group("target"))
        for item in re.finditer(
            r"%\s+(?P<target>(?:מכלל|of\s+(?:the\s+)?(?:total|all))\s+"
            r"[A-Za-z\u05d0-\u05ea'׳-]+(?:\s+[A-Za-z\u05d0-\u05ea'׳-]+){0,3}?)"
            r"\s+(?:was|were|is|are|היה|הייתה|היו)\b", region, re.I,
        )
    ))
    if unit_label == "%" and len(partitive_targets) == 1:
        return partitive_targets[0]
    prefix = region[max(0, match.start() - 100):match.start()]
    preceding = re.search(
        r"(?P<target>[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*"
        r"(?:\s+[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*){0,3})"
        r"\s+(?:was|were|is|are|stood\s+at|amounted\s+to|numbered|"
        r"היה|הייתה|היו|עמד(?:ה)?\s+על)\s*$",
        prefix,
        re.IGNORECASE,
    )
    if preceding:
        return normalize_request_text(preceding.group("target"))
    # Parallel endpoint prose often states a shared subject once: "The rate
    # was 11% in 2022 and 19% in 2025."  Reuse that evidence-side subject for
    # the later endpoint only when the same bounded proposition itself
    # supplies it.  A repeated category-specific subject immediately before a
    # value has already won in the branch above.
    shared = re.match(
        r"\s*(?P<target>.+?)\s+(?:was|were|is|are|stood\s+at|amounted\s+to|numbered|"
        r"היה|הייתה|היו|עמד(?:ה)?\s+על)\s+"
        r"[-+]?\d",
        region,
        re.IGNORECASE,
    )
    return normalize_request_text(shared.group("target")) if shared else ""


def _numeric_local_proposition(
    value: str,
    match: re.Match[str],
) -> tuple[str, int, int, re.Match[str]]:
    # Balanced parenthetical comparisons can carry separate category/period
    # series in a single sentence.  Keep the governing lead-in for that
    # parenthetical, but never borrow another sibling's band or scalar.
    boundary_start, boundary_end = 0, len(value)
    if balanced_delimiters(value):
        stack: list[int] = []
        pairs: list[tuple[int, int]] = []
        for position, character in enumerate(value):
            if character == "(":
                stack.append(position)
            elif character == ")" and stack:
                pairs.append((stack.pop(), position + 1))
        for opening, closing in sorted(pairs, key=lambda pair: pair[1] - pair[0]):
            if not opening < match.start() < match.end() <= closing:
                continue
            previous = max((end for start, end in pairs if end <= opening), default=0)
            region = value[previous:closing]
            if (len(set(_YEAR_RE.findall(region))) >= 2
                    and len(_numeric_candidates(value[opening:closing], "percentage")) >= 2
                    and _PREDICATE_RE.search(region)):
                boundary_start, boundary_end = previous, closing
                break
    delimiters = [
        item for item in re.finditer(
            r"[.;׃]|,\s+|\b(?:while|whereas)\b", value, re.IGNORECASE
        )
        if boundary_start <= item.start() < boundary_end and not (
            item.group(0).startswith(",")
            and re.search(r"(?:19|20)\d{2}\s*$", value[:item.start()])
        )
    ]
    start = max(
        (item.end() for item in delimiters if item.end() <= match.start()),
        default=boundary_start,
    )
    end = min(
        (item.start() for item in delimiters if item.start() >= match.end()),
        default=boundary_end,
    )
    raw = value[start:end]
    leading = len(raw) - len(raw.lstrip())
    trailing = len(raw.rstrip())
    start += leading
    end = start + max(0, trailing - leading)
    proposition = value[start:end]
    local_start = max(0, match.start() - start)
    local_end = local_start + (match.end() - match.start())
    local_match = _NUMBER_WITH_OPTIONAL_UNIT_RE.search(
        proposition,
        max(0, local_start - 2),
        min(len(proposition), local_end + 32),
    )
    if local_match is None:
        return value, 0, len(value), match
    return proposition, start, end, local_match


def _source_numeric_value_category(
    proposition: str,
    numeric_match: re.Match[str],
    unit_label: str,
) -> str:
    """Return one literal category bound to a value in a parallel source list.

    The binding is deliberately local: a label must occur immediately after
    the value (optionally after its unit) and before the next scalar or the
    end of the proposition.  It does not consult a request and therefore
    cannot select a desired value from an otherwise ambiguous list.
    """
    if len(_numeric_candidates(proposition, "")) < 2:
        return ""
    if re.search(
        r"\d+(?:[.,]\d+)?\s+(?:and|or|או)\s*$",
        proposition[:numeric_match.start()],
        re.IGNORECASE,
    ):
        # In ``14 and 19 vessels`` the trailing noun is a shared unit, not
        # a category that distinguishes the second scalar from the first.
        return ""
    suffix = proposition[numeric_match.end():]
    if unit_label and unit_label.casefold() not in {
        "", "number", "numeric", "%", "percent", "percentage",
    }:
        unit_match = re.match(rf"\s*{re.escape(unit_label)}\b", suffix, re.I)
        if unit_match is not None:
            suffix = suffix[unit_match.end():]
    label_match = re.match(
        r"\s*(?:(?:for|among|within)\s+)?"
        r"(?P<label>[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*"
        r"(?:\s+[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*){0,2})"
        r"(?=\s*(?:[,;.!?׃]|\band\b|\bאו\b|$))",
        suffix,
        re.IGNORECASE,
    )
    if label_match is None:
        return ""
    category = normalize_request_text(label_match.group("label")).strip(" ,;:.?!")
    if not category or _YEAR_RE.search(category):
        return ""
    if re.search(
        r"\b(?:in|during|at|and|or)\b|(?:בשנת|בשנה|ב[־-])",
        category,
        re.IGNORECASE,
    ):
        return ""
    return category


@dataclass(frozen=True)
class _SourceLocalNumericBinding:
    """One deterministic, source-only association for a numeric literal."""

    provenance: NumericBindingProvenance
    category: str = ""
    period: str = ""
    scalar_role: str = "not_applicable"
    quantity_kind: str = "unspecified"
    statistical_role: str = "unspecified"
    aggregation_role: str = "unspecified"
    disambiguates_multiple_values: bool = False


def _source_local_numeric_binding(
    proposition: str,
    numeric_match: re.Match[str],
    *,
    absolute_proposition_start: int,
    source_span_sha256: str,
    unit_label: str = "",
    row_header: str = "",
    column_header: str = "",
    scalar_role: str = "not_applicable",
    quantity_semantics: Optional[QuantitySemantics] = None,
) -> _SourceLocalNumericBinding:
    """Prove a numeric association from its own bounded source structure.

    The helper intentionally receives no request/requirement state.  In a
    multi-value proposition it admits a scalar only when a local label,
    period label, or validated table role makes that scalar non-interchangeable
    with its neighbours.
    """
    value_span = (
        absolute_proposition_start + numeric_match.start(),
        absolute_proposition_start + numeric_match.end(),
    )
    candidates = _numeric_candidates(proposition, "")
    multi_value = len(candidates) > 1
    candidate_units = {
        str(candidate_unit or "number").casefold()
        for _candidate_match, _candidate_value, candidate_unit in candidates
    }
    distinct_unit_roles = (
        multi_value
        and len(candidate_units) > 1
        and "number" not in candidate_units
        and "numeric" not in candidate_units
    )
    candidate_scalar_roles = {
        classify_scalar_role(
            proposition,
            candidate_match.start(),
            candidate_match.end(),
            unit=candidate_unit,
            structural_role=(
                f"row:{row_header}|column:{column_header}"
                if row_header and column_header else ""
            ),
        )
        for candidate_match, _candidate_value, candidate_unit in candidates
    }
    quantity_semantics = quantity_semantics or quantity_semantics_from_scalar_role(
        scalar_role,
    )
    scalar_role = quantity_semantics.scalar_role
    distinct_scalar_roles = (
        multi_value
        and scalar_role != "not_applicable"
        and scalar_role_has_local_cue(
            proposition,
            numeric_match.start(),
            numeric_match.end(),
            role=scalar_role,
            unit=unit_label,
            structural_role=(
                f"row:{row_header}|column:{column_header}"
                if row_header and column_header else ""
            ),
        )
        and len(candidate_scalar_roles) > 1
    )
    prefix = proposition[:numeric_match.start()]
    suffix = proposition[numeric_match.end():]

    period = ""
    period_span: Optional[tuple[int, int]] = None
    for period_match in _YEAR_RE.finditer(prefix):
        between = prefix[period_match.end():]
        if re.fullmatch(
            r"\s*(?:[:(—–-])?\s*(?:כ\s*[־-]?\s*|~\s*|approximately\s+|about\s+)?",
            between, re.I,
        ):
            period = period_match.group(0)
            period_span = (
                absolute_proposition_start + period_match.start(),
                absolute_proposition_start + period_match.end(),
            )
    if not period:
        trailing_period = re.match(
            r"\s*(?:%|[A-Za-z\u05d0-\u05ea]+)?\s*"
            r"(?:in|during|at|בשנת|בשנה|ב[־-])?\s*"
            r"(?P<period>(?:19|20)\d{2})\b",
            suffix,
            re.IGNORECASE,
        )
        if trailing_period is not None:
            period = trailing_period.group("period")
            period_start = numeric_match.end() + trailing_period.start("period")
            period_span = (
                absolute_proposition_start + period_start,
                absolute_proposition_start + period_start + len(period),
            )

    category = ""
    category_span: Optional[tuple[int, int]] = None
    # A heading/list label immediately before a scalar.  At the beginning of
    # a proposition require an explicit separator, while after a list
    # delimiter a bare label/value pair is sufficient.
    before_label = re.search(
        r"(?:^|[,:;\n])\s*(?P<label>"
        r"[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳ -]{0,80}?)"
        r"\s*(?:(?P<separator>[:—–-])\s*)?$",
        prefix,
        re.IGNORECASE,
    )
    if before_label is not None:
        starts_at_boundary = before_label.start() == 0
        bare_item_label = (
            starts_at_boundary
            and not before_label.group("separator")
            and not re.search(
                r"\b(?:is|are|was|were|recruit(?:s|ed|ing)?|"
                r"employ(?:s|ed|ing)?|reported|handled|recorded|"
                r"process(?:es|ed|ing)?|handled|"
                r"represents?|amounted|stood)\b",
                before_label.group("label"),
                re.IGNORECASE,
            )
        )
        explicit_local_label = bool(before_label.group("separator"))
        label_is_narrative = bool(_CHANGE_DIRECTION_RE.search(before_label.group('label')) or re.search(
            r"\b(?:is|are|was|were|used|use[sd]?|failed|handled|"
            r"process(?:es|ed|ing)?|recruit(?:s|ed|ing)?|"
            r"employ(?:s|ed|ing)?|reported|recorded)\b",
            before_label.group("label"),
            re.IGNORECASE,
        ))
        if (
            (multi_value or explicit_local_label)
            and not label_is_narrative
            and (not starts_at_boundary or explicit_local_label or bare_item_label)
        ):
            raw_category = normalize_request_text(before_label.group("label"))
            if raw_category and not _YEAR_RE.search(raw_category):
                category = raw_category
                label_start = before_label.start("label")
                category_span = (
                    absolute_proposition_start + label_start,
                    absolute_proposition_start + label_start + len(
                        before_label.group("label")
                    ),
                )
    unlabelled_shared_suffix = bool(re.search(
        r"\d+(?:[.,]\d+)?(?:\s+(?:%|[A-Za-z\u05d0-\u05ea]+)){0,2}"
        r"\s+(?:and|or|או)\s*$",
        prefix,
        re.IGNORECASE,
    ))
    if multi_value and not category and not unlabelled_shared_suffix:
        suffix_for_label = suffix
        suffix_offset = 0
        if unit_label and unit_label.casefold() not in {
            "", "number", "numeric", "%", "percent", "percentage",
        }:
            unit_match = re.match(
                rf"\s*{re.escape(unit_label)}\b", suffix_for_label, re.I
            )
            if unit_match is not None:
                suffix_offset = unit_match.end()
                suffix_for_label = suffix_for_label[suffix_offset:]
        elif unit_label.casefold() in {"", "number", "numeric"}:
            symbolic_unit = re.match(r"\s*[A-Z]{1,6}\b", suffix_for_label)
            if symbolic_unit is not None:
                suffix_offset = symbolic_unit.end()
                suffix_for_label = suffix_for_label[suffix_offset:]
        after_label = re.match(
            r"\s*(?:(?:for|among|within)\s+)?"
            r"(?P<label>[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*"
            r"(?:\s+[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*){0,2})"
            r"(?=\s*(?:[,;.!?׃]|\band\b|\bאו\b|"
            r"(?:in|during|at|בשנת|בשנה|ב[־-])\s+(?:19|20)\d{2}\b|$))",
            suffix_for_label,
            re.IGNORECASE,
        )
        if after_label is not None:
            raw_category = normalize_request_text(after_label.group("label"))
            repeated_shared_role = bool(
                raw_category
                and re.search(
                    rf"\b{re.escape(raw_category)}\b", prefix,
                    re.IGNORECASE,
                )
            )
            if (
                raw_category
                and not _YEAR_RE.search(raw_category)
                and not _TIME_UNIT_RE.search(raw_category)
                and not re.fullmatch(r"(?:and|or|או|ו)", raw_category, re.IGNORECASE)
                and not repeated_shared_role
            ):
                category = raw_category
                label_start = (
                    numeric_match.end()
                    + suffix_offset
                    + after_label.start("label")
                )
                category_span = (
                    absolute_proposition_start + label_start,
                    absolute_proposition_start + label_start + len(
                        after_label.group("label")
                    ),
                )

    structural_role = ""
    if row_header and column_header:
        structural_role = f"row:{row_header}|column:{column_header}"
    elif distinct_unit_roles:
        structural_role = f"unit:{unit_label or 'number'}"
    if structural_role:
        method = (
            "structural_row_column"
            if row_header and column_header else "local_unit_role"
        )
    elif distinct_scalar_roles:
        method = "local_scalar_role"
    elif category:
        method = "local_label_value"
    elif period:
        method = "local_period_value"
    else:
        method = "single_value_proposition"
    return _SourceLocalNumericBinding(
        provenance=NumericBindingProvenance(
            value_span=value_span,
            semantic_label_span=category_span,
            period_span=period_span,
            binding_method=method,
            structural_role=structural_role,
            scalar_role=scalar_role,
            quantity_kind=quantity_semantics.quantity_kind,
            statistical_role=quantity_semantics.statistical_role,
            aggregation_role=quantity_semantics.aggregation_role,
            statistical_role_cue_span=(
                (
                    absolute_proposition_start
                    + quantity_semantics.statistical_role_cue_span[0],
                    absolute_proposition_start
                    + quantity_semantics.statistical_role_cue_span[1],
                )
                if quantity_semantics.statistical_role_cue_span is not None else None
            ),
            aggregation_role_cue_span=(
                (
                    absolute_proposition_start
                    + quantity_semantics.aggregation_role_cue_span[0],
                    absolute_proposition_start
                    + quantity_semantics.aggregation_role_cue_span[1],
                )
                if quantity_semantics.aggregation_role_cue_span is not None else None
            ),
            source_span_sha256=source_span_sha256,
        ),
        category=category,
        period=period,
        scalar_role=scalar_role,
        quantity_kind=quantity_semantics.quantity_kind,
        statistical_role=quantity_semantics.statistical_role,
        aggregation_role=quantity_semantics.aggregation_role,
        disambiguates_multiple_values=(
            not multi_value or bool(
                structural_role or category or period or distinct_scalar_roles
            )
        ),
    )


def _evidence_subject(
    value: str,
    target: str,
    *,
    row_header: str = "",
    numeric_match: Optional[re.Match[str]] = None,
) -> str:
    if row_header:
        return row_header
    if numeric_match is not None:
        # A source-owned numeric experience cutoff is part of the category,
        # not a new measured subject. The word-only subject patterns below
        # otherwise restart after "5.0" at "years of experience". Recover
        # only a literal subject before its property verb, within this same
        # bounded proposition, and require the category proof in that subject.
        category_subject = re.match(
            r"\s*(?P<subject>[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*"
            r"(?:\s+(?:[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*|\d+(?:\.\d+)?)){0,14})"
            r"\s+(?:is|are|was|were|stood\s+at|amounted\s+to|represented|"
            r"constituted|accounted\s+for|increased|decreased|rose|fell|changed|"
            r"הוא|היא|היו|היה|הייתה|עמד(?:ה)?\s+על|עלה|עלתה|ירד|ירדה)\b",
            value, re.IGNORECASE,
        )
        if (category_subject is not None
                and category_subject.end() <= numeric_match.start()
                and re.search(r"\d", category_subject.group("subject"))
                and any(len(facet.split(":")) >= 4 for facet in experience_band_facets(category_subject.group("subject")))):
            return normalize_request_text(category_subject.group("subject"))
        # A change proposition supplies its subject once, before both endpoint
        # values.  Prefer that source-local binding before looking only at the
        # short prefix of a later endpoint.
        shared_change = re.match(
            r"\s*(?P<subject>[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*"
            r"(?:\s+[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*){0,14})"
            r"\s+(?:increased|decreased|rose|fell|changed|grew|declined|"
            r"עלה|עלתה|עלו|ירד|ירדה|ירדו|השתנה|השתנתה)\b",
            value,
            re.IGNORECASE,
        )
        if shared_change:
            return normalize_request_text(shared_change.group("subject"))
        shared_period_series = re.match(
            r"\s*(?P<subject>[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*"
            r"(?:\s+[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*){0,14})"
            r"\s+(?:is|are|was|were|stood\s+at|amounted\s+to|numbered|"
            r"הוא|היא|היו|היה|הייתה|עמד(?:ה)?\s+על)\s+"
            r"(?:19|20)\d{2}\s*:",
            value,
            re.IGNORECASE,
        )
        if shared_period_series:
            return normalize_request_text(shared_period_series.group("subject"))
        # A parenthetical count can quantify an explicitly named total scope
        # after a percentage (``of all active firms (2,158 firms)``).  The
        # percentage is not the count's subject.  Preserve the literal
        # governing scope from the same proposition before considering the
        # broader percentage relation below.
        parenthetical_total_scope = re.search(
            r"(?:\b(?:of\s+all|all|total)\b|מכלל)\s+"
            r"(?P<subject>[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳ -]{0,160}?)"
            r"\s*\(\s*$",
            value[:numeric_match.start()],
            re.IGNORECASE,
        )
        if parenthetical_total_scope:
            return normalize_request_text(
                parenthetical_total_scope.group("subject")
            )
        # A count inside a parenthetical can be the value of an immediately
        # preceding percentage/share proposition.  The local unit noun
        # ("workers") is not its semantic subject; retain the parent
        # proposition that governs the parenthetical value.
        nested_parent = re.search(
            r"(?P<subject>[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳ -]{0,120}?)"
            r"\s+(?:is|are|was|were|represented|representing|"
            r"constituted|accounted\s+for|מהווה|היה|הייתה)\b"
            r"[^()]{0,72}\(\s*$",
            value[:numeric_match.start()],
            re.IGNORECASE,
        )
        if nested_parent:
            return normalize_request_text(nested_parent.group("subject"))
        prefix = value[max(0, numeric_match.start() - 150):numeric_match.start()]
        bound = re.search(
            r"(?P<subject>[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*"
            r"(?:\s+[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*){0,14})"
            r"\s+(?:is|are|was|were|stood\s+at|amounted\s+to|numbered|"
            r"increased|decreased|rose|fell|changed|grew|declined|"
            r"הוא|היא|היו|היה|הייתה|עמד(?:ה)?\s+על|עלה|עלתה|עלו|"
            r"ירד|ירדה|ירדו|השתנה|השתנתה)\s+"
            r"(?:(?:from|to)\s+|מ-?\s*|ל-?\s*)?"
            r"(?:approximately|approx\.?|about|around|nearly|roughly|בקירוב|כ-?)?\s*$",
            prefix,
            re.IGNORECASE,
        )
        if bound:
            return normalize_request_text(bound.group("subject"))
        # A compact endpoint series names its measured subject once: for
        # example, ``response time was 18 minutes in one year and a later
        # value in the next``.  The later scalar has no new local subject, but the same
        # bounded source proposition supplies one.  This must happen after
        # the immediately preceding binding above, so an explicit later
        # ``west`` association still takes precedence.
        shared_endpoint = re.match(
            r"\s*(?P<subject>[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*"
            r"(?:\s+[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*){0,14})"
            r"\s+(?:is|are|was|were|stood\s+at|amounted\s+to|numbered|"
            r"הוא|היא|היו|היה|הייתה|עמד(?:ה)?\s+על)\s+[-+]?\d",
            value,
            re.IGNORECASE,
        )
        if shared_endpoint:
            return normalize_request_text(shared_endpoint.group("subject"))
        represented = re.match(
            r"\s*(?P<subject>[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*"
            r"(?:\s+[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*){0,14})"
            r"\s+(?:represented|representing|constituted|accounted\s+for)\b",
            value,
            re.IGNORECASE,
        )
        if represented:
            return normalize_request_text(represented.group("subject"))
    if target:
        return target
    cleaned = _YEAR_RE.sub(" ", _NUMBER_WITH_OPTIONAL_UNIT_RE.sub(" ", value))
    cleaned = normalize_request_text(cleaned).strip(" ,;:.?!")
    return " ".join(cleaned.split()[:12])


def _source_relation_subject(
    value: str,
    numeric_match: re.Match[str],
    predicate: str,
) -> str:
    """Return a source-stated actor for a counted employment proposition.

    The counted people are the target of an employment/recruitment relation,
    not automatically the actor that owns the fact.  This stays entirely
    source-local: it accepts either an explicit actor before the relation or,
    for a pronoun-led continuation, an explicit ``for <actor>`` phrase in the
    same bounded proposition.  It never consults request semantics.
    """
    action = (
        r"(?:employ(?:s|ed|ing)?|recruit(?:s|ed|ing)?|hire(?:s|d|ing)?|"
        r"process(?:es|ed|ing)?|handle(?:s|d|ing)?|manage(?:s|d|ing)?|"
        r"record(?:s|ed|ing)?|report(?:s|ed|ing)?|identify(?:s|ied|ing)?|"
        r"use(?:s|d|ing)?|contain(?:s|ed|ing)?|include(?:s|d|ing)?|"
        r"has|have|had)"
    )
    prefix = value[:numeric_match.start()]
    direct = re.search(
        rf"(?P<subject>[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*"
        rf"(?:\s+[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*){{0,5}})"
        rf"\s+{action}\s*$",
        prefix,
        re.IGNORECASE,
    )
    if direct is not None:
        direct_subject = normalize_request_text(
            direct.group("subject")
        ).strip(" ,;:.?!")
        if direct_subject.casefold() not in {
            "it", "they", "he", "she", "הוא", "היא", "הם", "הן",
        }:
            return direct_subject

    # A trailing ``for <actor>`` role is specific to employment/recruitment
    # propositions.  Do not generalize it to arbitrary verbs, where ``for``
    # can describe an audience or purpose rather than the source subject.
    if predicate not in {"employs_workers", "recruits_employees"}:
        return ""
    pronoun_continuation = re.match(
        rf"\s*(?:it|they|he|she|הוא|היא|הם|הן)\s+{action}\b",
        value,
        re.IGNORECASE,
    )
    if pronoun_continuation is None:
        return ""
    suffix = value[numeric_match.end():]
    contextual_actor = re.match(
        r"\s*(?:[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*\s+){0,3}"
        r"for\s+(?:the\s+)?(?P<subject>[A-Za-z\u05d0-\u05ea]"
        r"[A-Za-z\u05d0-\u05ea'׳-]*(?:\s+[A-Za-z\u05d0-\u05ea]"
        r"[A-Za-z\u05d0-\u05ea'׳-]*){0,4})\s*$",
        suffix,
        re.IGNORECASE,
    )
    return normalize_request_text(
        contextual_actor.group("subject") if contextual_actor is not None else ""
    ).strip(" ,;:.?!")


def _evidence_population(
    value: str, *, value_span: Optional[tuple[int, int]] = None,
) -> str:
    # Population ownership comes from this proposition, never the request or
    # an unrelated organization mention elsewhere in the source chunk.
    local_scope = source_population_scope(value, value_span=value_span)
    if local_scope:
        return local_scope
    if experience_band_facets(value):
        scopes = experience_scope_populations(value)
        if len(scopes) == 1:
            return scopes[0]
    match = _EVIDENCE_POPULATION_RE.search(value)
    if match:
        population = normalize_request_text(match.group("en") or match.group("he"))
        return re.sub(
            r"^(?:identify|identifies|identified|cover|covers|covered|represent|represents)\s+(?:the\s+)?",
            "",
            population,
            flags=re.IGNORECASE,
        )
    # Scope also appears in a governing prepositional clause without an
    # explicit population noun. Preserve that literal source phrase so a
    # broad or unknown fact cannot be promoted into a narrower request scope.
    contextual = re.search(
        r"\b(?:among|within|across)\s+(?P<en>[^,.;:]{2,180})(?=[,.;:]|$)|"
        r"(?:בקרב|מבין|מתוך)\s+(?P<he>[^,.;:]{2,180})(?=[,.;:]|$)",
        value,
        re.IGNORECASE,
    )
    return normalize_request_text(
        contextual.group("en") or contextual.group("he")
    ) if contextual else ""


def _source_response_category(
    value: str,
    numeric_match: re.Match[str],
    *,
    leading_context: str = "",
) -> str:
    """Return a nearby source-stated response category for one scalar.

    Flattened table text can contain several otherwise identical percentages.
    A literal yes/no/nonresponse label immediately governing one value is a
    source binding, not a request interpretation.  It is used to keep such
    atom-local values distinct; an unlabelled scalar deliberately remains
    unclassified.
    """
    prefix = normalize_evidence_whitespace(
        f"{leading_context} {value[:numeric_match.start()]}"
    )[-64:]
    categories: list[tuple[int, str]] = [
        (match.start(), normalize_request_text(
            match.group("en") or match.group("he") or ""
        ).casefold())
        for match in re.finditer(
        r"\b(?P<en>yes|no|nonresponse|did\s+not\s+(?:answer|respond))\b|"
        r"\b(?P<he>לא\s+השיבו|כן|לא)\b",
        prefix,
        re.IGNORECASE,
    )
    ]
    # An affirmative source proposition need not repeat a table's ``yes``
    # label.  Bind it only when the source locally says that respondents
    # affirmed existence, and let that nearer signal supersede an earlier
    # nonresponse value in flattened table text.  No request term is used.
    categories.extend(
        (match.start(), "yes")
        for match in re.finditer(
            r"\b(?:answered|responded|reported|indicated|stated)\s+"
            r"(?:that\s+)?(?:there\s+(?:is|are)|an?\s+existing)\b|"
            r"(?:השיבו|ציינו|אמרו)\s+ש?(?:קיים|קיימת|קיימים|קיימות)\b",
            prefix,
            re.IGNORECASE,
        )
    )
    if not categories:
        return ""
    raw = max(categories, key=lambda item: item[0])[1]
    if raw in {"yes", "כן"}:
        return "response_yes"
    if raw in {"no", "לא"}:
        return "response_no"
    return "response_nonresponse"


def _evidence_period(value: str, selected_period: str = "") -> str:
    if selected_period and re.search(
        rf"(?<!\d){re.escape(selected_period)}(?!\d)", value
    ):
        return selected_period
    years = tuple(dict.fromkeys(_YEAR_RE.findall(value)))
    if len(years) == 1:
        return years[0]
    if _CURRENT_TIME_RE.search(value):
        return "current"
    if _HISTORICAL_TIME_RE.search(value):
        return "historical"
    return ""


def _relation_arguments(value: str, predicate: str) -> dict[str, str]:
    """Extract typed relational roles from one bounded proposition."""
    if predicate == "purpose_of":
        purpose = _source_purpose_arguments(value)
        if purpose:
            return purpose
        # A visible purpose frame that failed the anchored parser must not
        # become the left side of the legacy generic ``X to Y`` fallback.
        # In particular an outer condition belongs to the whole proposition.
        if re.search(r"\b(?:purpose|objective|aim)\s+of\b|"
                     r"(?<!\w)ה?מטר(?:תו|תה|תם|תן|ת|ה)\s", value, re.I):
            return {}
    if predicate in DOCUMENT_ROLE_PREDICATES:
        from semantic_contracts import document_role_arguments
        return document_role_arguments(value, predicate)
    if predicate == "performed_by":
        active_responsibility = re.search(
            r"^\s*[-*•▪◦]?\s*(?P<agent>.+?)\s*(?:—|–|-|:)\s*"
            r"(?:handles?|manages?|is\s+responsible\s+for|"
            r"מטפל(?:ת|ים|ות)?\s+ב|אחראי(?:ת|ים|ות)?\s+על)\s*"
            r"(?P<target>.+?)\s*[.;]?\s*$",
            value,
            re.IGNORECASE,
        )
        if active_responsibility:
            return {
                "procedure_target": normalize_request_text(
                    active_responsibility.group("target")
                ).strip(" ,;:.?!"),
                "method": normalize_request_text(
                    active_responsibility.group("agent")
                ).strip(" ,;:.?!"),
            }
    patterns: dict[str, tuple[re.Pattern[str], tuple[str, str]]] = {
        "caused_by": (
            re.compile(
                r"^(?P<left>.+?)\s+(?:because|due\s+to|as\s+a\s+result\s+of|"
                r"since|בשל|עקב|מפני|משום|מאחר\s+ש)\s+(?P<right>.+)$",
                re.IGNORECASE,
            ),
            ("effect", "reason"),
        ),
        "explained_as": (
            re.compile(
                r"^(?P<left>.+?)\s+(?:is|was|are|were)?\s*"
                r"(?:explained|interpreted|accounted\s+for)\s+(?:by|as)\s+"
                r"(?P<right>.+)$|^(?P<left_he>.+?)\s+(?:מוסבר|מוסברת|"
                r"מתפרש|מתפרשת)\s+(?:ב|כ)?(?P<right_he>.+)$",
                re.IGNORECASE,
            ),
            ("explained_event", "explanation"),
        ),
        "purpose_of": (
            re.compile(
                r"^(?:the\s+)?purpose\s+of\s+(?P<left>.+?)\s+(?:is|was)\s+"
                r"(?P<right>.+)$|^(?P<left_alt>.+?)\s+(?:(?:was|is)\s+"
                r"(?:intended|designed|created)\s+to\s+|to\s+)"
                r"(?P<right_alt>.+)$|"
                r"^מטר(?:ת|ה)\s+(?P<left_he>.+?)\s+(?:היא|הייתה)\s+"
                r"(?P<right_he>.+)$|^(?P<left_he_alt>.+?)\s+נועד(?:ה)?\s+"
                r"(?:ל|כדי\s+ל)(?P<right_he_alt>.+)$",
                re.IGNORECASE,
            ),
            ("purpose_target", "purpose"),
        ),
        "defined_as": (
            re.compile(
                r"^(?P<left>.+?)\s+(?:means?|is\s+defined\s+as|refers?\s+to)\s+"
                r"(?P<right>.+)$|^(?P<left_he>.+?)\s+(?:מוגדר(?:ת)?\s+כ|"
                r"פירושו|משמעותו)\s*(?P<right_he>.+)$",
                re.IGNORECASE,
            ),
            ("defined_entity", "definition"),
        ),
        "performed_by": (
            re.compile(
                r"^(?P<left>.+?)\s+(?:was|is)\s+(?:identified|determined|"
                r"measured|selected|constructed|derived|performed|implemented|"
                r"conducted|calculated)\s+(?:by|using|through)\s+(?P<right>.+)$|"
                r"^(?P<left_he>.+?)\s+(?:זוהה|זוהתה|נקבע|נקבעה|נמדד|נמדדה|"
                r"נבחר|נבחרה|נבנה|נבנתה|נגזר|נגזרה|בוצע|בוצעה|יושם|יושמה|חושב|חושבה)\s+"
                r"(?:באמצעות|על\s+ידי)\s+(?P<right_he>.+)$",
                re.IGNORECASE,
            ),
            ("procedure_target", "method"),
        ),
        "excluded_from": (
            re.compile(
                r"^(?P<left>.+?)\s+(?:was|were|is|are)?\s*"
                r"(?:excluded|omitted|not\s+included)\s+(?:from|in)\s+"
                r"(?P<right>.+)$|^(?P<left_he>.+?)\s+(?:לא\s+נכלל(?:ה|ו)?|"
                r"הוחרג(?:ה)?)\s+(?:מ|ב)(?P<right_he>.+)$",
                re.IGNORECASE,
            ),
            ("excluded_entity", "excluded_scope"),
        ),
        "included_in": (
            re.compile(
                r"^(?P<left>.+?)\s+(?:was|were|is|are)?\s*included\s+in\s+"
                r"(?P<right>.+)$|^(?P<left_he>.+?)\s+נכלל(?:ה|ו)?\s+ב"
                r"(?P<right_he>.+)$",
                re.IGNORECASE,
            ),
            ("included_entity", "included_scope"),
        ),
    }
    spec = patterns.get(predicate)
    if predicate == "exception_to":
        spec = patterns.get("excluded_from")
    if spec is None:
        return {}
    relation_text = value
    if "|" in relation_text:
        prefix, suffix = relation_text.rsplit("|", 1)
        if len(prefix.strip()) <= 120 and suffix.strip():
            relation_text = suffix.strip()
    match = spec[0].search(relation_text)
    if match is None:
        return {}
    groups = match.groupdict()
    left = next((groups.get(name) for name in (
        "left", "left_alt", "left_he", "left_he_alt"
    ) if groups.get(name)), "")
    right = next((groups.get(name) for name in (
        "right", "right_alt", "right_he", "right_he_alt"
    ) if groups.get(name)), "")
    roles = spec[1]
    def clean_argument(raw: str) -> str:
        normalized = normalize_request_text(raw).strip(" ,;:.?!")
        if "|" in normalized:
            normalized = normalized.rsplit("|", 1)[-1].strip(" ,;:.?!")
        return normalized

    return {
        roles[0]: clean_argument(left),
        roles[1]: clean_argument(right),
    } if left and right else {}


def _evidence_fact_compatibility_errors(
    requirement: Any,
    *,
    subject: str,
    predicate: str,
    target_entity: str,
    population: str,
    relation_arguments: Sequence[Any] = (),
) -> tuple[str, ...]:
    reasons: list[str] = []
    arguments = {
        str(getattr(item, "role", "") or ""):
        str(getattr(item, "value", "") or "")
        for item in relation_arguments
    }
    if not subject:
        reasons.append("evidence_subject_unresolved")
    if not predicate:
        reasons.append("evidence_predicate_unresolved")
    requested_target = normalize_request_text(
        getattr(requirement, "counted_entity", "")
        or getattr(requirement, "target_entity_type", "")
        or ""
    )
    if requested_target:
        relation_target_role = (
            RELATION_ARGUMENT_SCHEMAS.get(predicate, ("",))[0]
            if RELATION_ARGUMENT_SCHEMAS.get(predicate) else ""
        )
        observed_target = target_entity or arguments.get(
            relation_target_role, ""
        )
        if not observed_target:
            reasons.append("evidence_target_entity_unresolved")
        else:
            target_terms = {
                token.casefold()
                for token in _normalized_terms(requested_target)
                if token.casefold() not in {
                    "the", "a", "an", "of", "count", "number", "total",
                    "number", "מספר", "סך", "של",
                }
            }
            observed_terms = {
                token.casefold() for token in _normalized_terms(observed_target)
            }
            literal_coverage = (
                len(target_terms & observed_terms) / len(target_terms)
                if target_terms else 1.0
            )
            semantic_coverage = _semantic_term_overlap(
                requested_target, observed_target
            )
            if (
                literal_coverage < 0.75
                and semantic_coverage < 0.60
                and not canonical_entity_equivalent(
                    requested_target, observed_target,
                )
            ):
                reasons.append("evidence_target_entity_incompatible")
    expected_predicate = str(getattr(requirement, "predicate", "") or "")
    compatible_predicates = {
        "stated_as": {
            "stated_as", "has_attribute", "has_members", "performed_by",
        },
        "explained_as": {"explained_as", "caused_by", "purpose_of"},
        "method_of": {"performed_by"},
        "has_count": {"has_count", "included_in", "employs_workers", "recruits_employees"},
        "included_in": {"included_in", "has_count"},
        "maximum_of": {"has_count", "has_percentage"},
        "minimum_of": {"has_count", "has_percentage"},
        "compared_with": {"has_count", "has_percentage", "changed_between"},
        "changed_between": {"has_count", "has_percentage", "changed_between"},
        "employs_workers": {"employs_workers"},
        "recruits_employees": {"recruits_employees"},
        "raises_capital": {"raises_capital"},
        "excluded_from": {"excluded_from", "exception_to"},
    }
    if expected_predicate and predicate not in compatible_predicates.get(
        expected_predicate, {expected_predicate}
    ):
        reasons.append("evidence_predicate_incompatible")
    required_roles = RELATION_ARGUMENT_SCHEMAS.get(predicate, ())
    for role in required_roles:
        if not arguments.get(role):
            reasons.append(f"relation_argument_missing:{role}")
    kind = str(getattr(requirement, "kind", "") or "")
    concrete_metrics = tuple(
        normalize_request_text(item)
        for item in getattr(requirement, "objects_or_metrics", ()) or ()
        if normalize_request_text(item).casefold() not in {
            "", "count", "percentage", "scalar", "text proposition",
        }
    )
    if concrete_metrics and kind not in {
        "cause_or_reason", "explanation", "purpose", "intended_use",
        "definition", "procedure", "negative_relation",
        "positive_inclusion_or_count", "winner", "winner_attribute",
    }:
        observed_semantics = " ".join((subject, target_entity, *arguments.values()))
        source_language_subject = normalize_request_text(
            getattr(requirement, "subject", "") or ""
        )
        source_language_metric_match = bool(
            source_language_subject
            and max(
                _term_overlap(source_language_subject, observed_semantics),
                _semantic_term_overlap(
                    source_language_subject,
                    observed_semantics,
                ),
            ) >= 0.34
        )
        metric_aliases = {
            "employee_count": "employees workers workforce employs employment עובדים מעסיק",
            "employee_recruitment": "recruits recruited recruitment hires hired employees workers גיוס גייס עובדים",
            "capital_raised": "capital funding finance raised fundraising הון מימון גייס",
            "industry_sector": "industry sector market תעשייה ענף סקטור",
        }
        if not source_language_metric_match and all(
            max(
                _term_overlap(metric_aliases.get(metric, metric), observed_semantics),
                _semantic_term_overlap(metric_aliases.get(metric, metric), observed_semantics),
            ) <= 0
            for metric in concrete_metrics
        ):
            reasons.append("evidence_metric_incompatible")
    requested_population = str(getattr(requirement, "scope_population", "") or "")
    if requested_population:
        scope_ok, scope_reason = population_scope_compatibility(
            requested_population, population
        )
        if not scope_ok:
            reasons.append(scope_reason)
    requested_relation_target = normalize_request_text(
        getattr(requirement, "relation_target", "") or ""
    )
    if requested_relation_target:
        requested_role = {
            "cause_or_reason": (
                "effect" if predicate == "caused_by" else "explained_event"
            ),
            "explanation": (
                "effect" if predicate == "caused_by" else "explained_event"
            ),
            "purpose": "purpose_target",
            "intended_use": "purpose_target",
            "definition": "defined_entity",
            "procedure": "procedure_target",
            "negative_relation": "excluded_scope",
            "positive_inclusion_or_count": "included_scope",
        }.get(kind, "")
        argument = arguments.get(requested_role, "") if requested_role else ""
        if requested_role and not argument:
            reasons.append("relational_argument_unresolved")
        elif requested_role and (
            not _relation_event_compatible(
                requested_relation_target,
                argument,
            )
            if kind in {"cause_or_reason", "explanation"}
            else max(
                _term_overlap(requested_relation_target, argument),
                _semantic_term_overlap(requested_relation_target, argument),
            ) < 0.45
        ):
            reasons.append("relational_argument_incompatible")
    return tuple(dict.fromkeys(reasons))


def _nominal_attribute_semantics(proposition: str) -> Optional[dict[str, Any]]:
    """Read a literal label followed by an explicit technical identifier tail."""
    value = normalize_request_text(proposition)
    if not (re.match(r"^[•▪◦*-]\s+", value) or ":" in value):
        return None
    value = re.sub(r"^[•▪◦*-]\s+", "", value).strip()
    match = re.fullmatch(
        r"(?P<label>[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea '\u05f3-]{1,100}?)"
        r"\s*:?\s+(?P<identifier>[A-Z][A-Za-z0-9]*(?:[-/_.][A-Za-z0-9]+)*"
        r"(?:\s+[A-Z][A-Za-z0-9]*(?:[-/_.][A-Za-z0-9]+)*){0,4})[.]?",
        value,
    )
    if match is None:
        return None
    identifier = match.group("identifier")
    model_token = identifier.split()[0]
    if not (re.fullmatch(r"[A-Z]{2,}", model_token) or re.search(r"[0-9/_.-]", model_token)):
        return None
    label = match.group("label").strip(" :-")
    if re.search(r"\b(?:not|never|must|shall|will|can|install|remove|avoid|requires?|"
                 r"explains?|causes?|includes?|excludes?|is|are|was|were|has|have)\b|"
                 r"(?<![\u05d0-\u05ea])(?:לא|ללא|אין|אסור|נדרש)(?![\u05d0-\u05ea])", label, re.I):
        return None
    return {
        "subject": label, "predicate": "has_attribute",
        "target_entity": identifier,
        "population": "", "period": "", "polarity": "neutral",
        "relation_arguments": {}, "nominal_attribute": True,
    }


def _text_evidence_semantics(
    proposition: str,
    *,
    row_header: str = "",
) -> dict[str, Any]:
    nominal = _nominal_attribute_semantics(proposition)
    if nominal is not None:
        return nominal
    value = normalize_request_text(proposition)
    predicate = _evidence_predicate(value)
    if not predicate and re.search(
        r"\b(?:explains?|explained|interprets?|interpreted|"
        r"accounted\s+for|indicates?|shows?|states?|reports?)\b|"
        r"(?:מסביר|מסבירה|מוסבר|מוסברת|מתפרש|מתפרשת|מצביע|מראה|מציין|מדווח)",
        value,
        re.IGNORECASE,
    ):
        predicate = "explained_as"
    if not predicate and re.search(
        r"\b(?:performed|implemented|conducted|carried\s+out)\b|"
        r"(?:בוצע|בוצעה|יושם|יושמה)",
        value,
        re.IGNORECASE,
    ):
        predicate = "performed_by"
    if (
        not predicate
        and _NUMBER_WITH_OPTIONAL_UNIT_RE.search(value)
        and re.search(
            r"\b(?:is|are|was|were|stood\s+at|amounted\s+to|total(?:ed|s)?)\b|"
            r"(?:הוא|היא|היה|הייתה|היו|עמד(?:ה)?\s+על|הסתכ(?:ם|מה)\s+ב)",
            value,
            re.IGNORECASE,
        )
    ):
        predicate = "stated_as"
    relation_arguments = _relation_arguments(value, predicate)
    target = row_header
    if not target:
        match = re.match(
            r"^\s*(?P<target>.+?)\s+(?:is|are|was|were|has|have|"
            r"stood\s+at|amounted\s+to|numbered|total(?:ed|s)?|"
            r"הוא|היא|היו|היה|הייתה|עמד(?:ה)?\s+על|מנה|מנתה|"
            r"הסתכ(?:ם|מה)\s+ב|נכלל|נכללה|נכללו|הוחרג|הוחרגה)\b",
            value,
            re.IGNORECASE,
        )
        if match:
            target = normalize_request_text(match.group("target")).strip(" ,;:.?!")
    subject = _evidence_subject(value, target, row_header=row_header)
    # Discourse wrappers (for example, "the report explains that ...") can
    # leave the generic subject parser empty after numeric cleanup.  A typed
    # relation already supplies a stronger evidence-derived subject: its
    # first mandatory argument is the entity/event the proposition is about.
    # Use that literal argument rather than borrowing a target from the
    # request or escalating an otherwise complete fact to the resolver.
    primary_role = next(iter(RELATION_ARGUMENT_SCHEMAS.get(predicate, ())), "")
    primary_argument = relation_arguments.get(primary_role, "")
    if (
        predicate == "performed_by"
        and relation_arguments.get("method")
        and re.search(
            r"\b(?:handles?|manages?|responsible\s+for)\b|"
            r"(?:מטפל(?:ת|ים|ות)?\s+ב|אחראי(?:ת|ים|ות)?\s+על)",
            value,
            re.IGNORECASE,
        )
    ):
        # In an active responsibility assertion ("team — handles task"),
        # the requested entity is the literal agent.  Keep the task in the
        # typed relation argument so metric compatibility is still checked
        # against this same bounded item.
        subject = relation_arguments["method"]
        target = relation_arguments["method"]
    elif primary_argument:
        subject = primary_argument
        target = primary_argument
    return {
        "subject": subject,
        "predicate": predicate,
        "target_entity": target,
        "population": _evidence_population(value),
        "period": _evidence_period(value),
        "polarity": "negative" if _EVIDENCE_NEGATION_RE.search(value) else "neutral",
        "relation_arguments": relation_arguments,
    }


def _text_fact_propositions(source: str) -> tuple[tuple[str, int, int], ...]:
    """Keep explicit nominal fields separate from their following prose."""
    nominal_spans = [
        match.span("paragraph")
        for match in re.finditer(r"(?ms)(?:^|\n\s*\n)(?P<paragraph>\S[^\n]*(?:\n(?!\s*\n)[^\n]*)*)", source)
        if _nominal_attribute_semantics(match.group("paragraph")) is not None
    ]
    if not nominal_spans:
        return tuple(segment_text_propositions(source))
    result = []
    cursor = 0
    for start, end in nominal_spans:
        result.extend((text, cursor + left, cursor + right)
                      for text, left, right in segment_text_propositions(source[cursor:start]))
        result.append((source[start:end], start, end))
        cursor = end
    result.extend((text, cursor + left, cursor + right)
                  for text, left, right in segment_text_propositions(source[cursor:]))
    return tuple(result)


def _period_candidate_quality(
    value: str,
    period_match: re.Match[str],
    candidate_match: re.Match[str],
) -> int:
    if candidate_match.end() <= period_match.start():
        between = value[candidate_match.end():period_match.start()]
        if re.fullmatch(
            r"\s*(?:in|during|בשנת|בשנה|ב־|ב-)\s*",
            between,
            re.IGNORECASE,
        ):
            return 8
        if (
            _COUNT_UNIT_RE.search(between)
            and re.search(
                r"(?:\bduring\b|בעת|במהלך)\s*$",
                between,
                re.IGNORECASE,
            )
        ):
            return 8
        return 4
    between = value[period_match.end():candidate_match.start()]
    if re.search(
        r"[,;]\s*(?:compared(?:\s+with|\s+to)?|versus|vs\.?|while|whereas|"
        r"לעומת|בהשוואה\s+ל)",
        between,
        re.IGNORECASE,
    ):
        return 0
    preceding = value[max(0, period_match.start() - 16):period_match.start()]
    if re.match(
        r"\s*(?:to|until|through|עד|לכ[־-]?|ל[־-]?)\b",
        between,
        re.IGNORECASE,
    ):
        return 2
    if re.search(
        r"(?:\bin\s+|\bduring\s+|בשנת\s*|בשנה\s*|ב[־-]\s*)$",
        preceding,
        re.IGNORECASE,
    ):
        return 8
    return 6


def _time_candidate_rank(
    requirement: CoverageRequirement,
    region: str,
    cue: re.Match[str],
    candidate: tuple[re.Match[str], str, str],
) -> tuple[int, int, int, int]:
    metric_association = _candidate_metric_association(
        requirement,
        region,
        candidate,
    )
    period_association = _period_candidate_quality(
        region,
        cue,
        candidate[0],
    )
    distance = min(
        abs(candidate[0].start() - cue.end()),
        abs(cue.start() - candidate[0].end()),
    )
    modifier_specific = any(
        pattern.search(requirement.metric)
        for _name, pattern in _SCALAR_MEASURE_MODIFIERS
    )
    return (
        metric_association if modifier_specific else period_association,
        period_association if modifier_specific else -distance,
        -distance if modifier_specific else metric_association,
        -candidate[0].start(),
    )


def _legacy_typed_fact_for_requirement(
    unit: EvidenceUnit,
    requirement: CoverageRequirement,
    execution_contract: Optional[RequestExecutionContract] = None,
    *,
    evidence_semantic_resolver: Optional[EvidenceSemanticResolver] = None,
    assisted_attempted_unit_ids: Optional[set[str]] = None,
    record_trace: bool = True,
) -> Optional[TypedFact]:
    if execution_contract is not None and requirement.id not in execution_contract.requirement_ids:
        raise RuntimeError("fact extraction received noncanonical requirement ID")
    if execution_contract is None and os.getenv(
        "PROJECT_X_CANDIDATE_MODE", "0"
    ).strip().casefold() in {"1", "true", "yes", "on"}:
        raise RuntimeError("fact extraction requires RequestExecutionContract")
    canonical_requirement = next((
        item for item in getattr(execution_contract, "requirements", ())
        if item.id == requirement.id
    ), None)
    attempted_unit_ids = (
        assisted_attempted_unit_ids
        if assisted_attempted_unit_ids is not None else set()
    )

    def terminal(outcome: str, *reason_codes: str, fact: Optional[TypedFact] = None) -> None:
        recorder = current_eval_trace() if record_trace else None
        if recorder is not None and hasattr(recorder, "record_fact_terminal_outcome"):
            recorder.record_fact_terminal_outcome(
                requirement_id=requirement.id,
                evidence_id=unit.id,
                outcome=outcome,
                reason_codes=reason_codes,
                fact_id=(fact.fact_id if fact is not None else ""),
                atom_id=(
                    fact.evidence_atom.atom_id
                    if fact is not None and fact.evidence_atom is not None else ""
                ),
            )

    if (
        canonical_requirement is not None
        and canonical_requirement.satisfaction_mode
        is not RequirementSatisfactionMode.SOURCE_FACT
    ):
        terminal(
            "NOT_APPLICABLE",
            "requirement_satisfaction_mode_not_source_fact",
        )
        return None

    def reject(*reason_codes: str) -> None:
        recorder = current_eval_trace() if record_trace else None
        if recorder is not None and hasattr(recorder, "record_fact_rejection"):
            recorder.record_fact_rejection(
                requirement_id=requirement.id,
                evidence_id=unit.id,
                reason_codes=reason_codes,
            )

    if unit.structural_type in {
        StructuralUnitType.HEADING.value,
        StructuralUnitType.CAPTION.value,
        StructuralUnitType.TABLE_OF_CONTENTS.value,
        StructuralUnitType.LIST_OF_FIGURES.value,
        StructuralUnitType.LIST_OF_TABLES.value,
        StructuralUnitType.HEADER_FOOTER.value,
        StructuralUnitType.PAGE_NUMBER.value,
        StructuralUnitType.FORMULA.value,
        StructuralUnitType.MALFORMED.value,
    }:
        reject("structural_type_ineligible")
        terminal("REJECTED", "structural_type_ineligible")
        return None
    metadata = dict(unit.raw_hit.get("metadata") or {})
    row_header = normalize_request_text(
        (
            unit.structural_evidence_unit.row_label
            if unit.structural_evidence_unit is not None else ""
        )
        or metadata.get("row_header")
        or metadata.get("table_row_header")
        or metadata.get("category")
        or ""
    )
    column_header = normalize_request_text(
        metadata.get("column_header")
        or metadata.get("table_column_header")
        or metadata.get("value_header")
        or (
            " ".join(unit.structural_evidence_unit.column_roles[1:])
            if unit.structural_evidence_unit is not None
            and len(unit.structural_evidence_unit.column_roles) > 1
            else ""
        )
        or ""
    )
    if (
        requirement.id.startswith("req-")
        and unit.structural_type in {
            StructuralUnitType.TABLE_ROW.value,
            StructuralUnitType.CHART_DATA.value,
            StructuralEvidenceType.TABLE_ROW.value,
            StructuralEvidenceType.CHART_CATEGORY_RECORD.value,
        }
    ):
        # A numeric table/chart fact is valid only when both axes are bound.
        # A single visible number is not evidence of its row/column identity.
        if not row_header or not column_header:
            reject(
                *(code for code, value in (
                    ("row_header_unbound", row_header),
                    ("column_header_unbound", column_header),
                ) if not value)
            )
            terminal("REJECTED", "table_axis_unbound")
            return None
        if row_header and column_header:
            header_context = f"{row_header} {column_header}"
            if max(
                _term_overlap(requirement.metric, header_context),
                _semantic_term_overlap(requirement.metric, header_context),
            ) <= 0:
                reject("table_axis_metric_incompatible")
                terminal("REJECTED", "table_axis_metric_incompatible")
                return None
    best: Optional[TypedFact] = None
    for claim in unit.allowed_claims:
        if _MALFORMED_ADJACENT_GROUPED_NUMBERS_RE.search(claim):
            continue
        if not _metric_compatible(
            requirement.metric,
            claim,
            expected_unit=requirement.expected_unit,
            require_specific_focus=(
                requirement.kind != "comparison_endpoint"
                and not (
                    canonical_requirement is not None
                    and canonical_requirement.kind == "lookup"
                    and canonical_requirement.value_type == "numeric"
                )
            ),
        ):
            continue
        claim_position = unit.canonical_text.find(claim)
        if claim_position >= 0 and requirement.kind == "direct_fact":
            local_context = unit.canonical_text[
                max(0, claim_position - 180):
                min(
                    len(unit.canonical_text),
                    claim_position + len(claim) + 180,
                )
            ]
            requested_families = _facet_families(requirement.metric)
            contextual_families = _facet_families(local_context)
            if (
                contextual_families & {"education", "experience"}
                and not (
                    contextual_families & requested_families
                )
            ):
                continue
        region = _facet_region(claim, requirement.facet)
        if not region:
            continue
        candidates = _numeric_candidates(region, requirement.expected_unit)
        requested_count_family = (
            _count_unit_family(requirement.metric)
            if requirement.expected_unit == "count"
            else ""
        )
        if requested_count_family:
            candidates = [
                candidate
                for candidate in candidates
                if _count_unit_family(candidate[2])
                == requested_count_family
            ]
        region_position = claim.find(region)
        candidates = [
            candidate
            for candidate in candidates
            if not _candidate_in_unrequested_subgroup(
                requirement,
                region,
                candidate,
                leading_context=(
                    unit.canonical_text[
                        max(
                            0,
                            claim_position
                            + max(0, region_position)
                            + candidate[0].start()
                            - 72,
                        ):
                        claim_position
                        + max(0, region_position)
                        + candidate[0].start()
                    ]
                    if claim_position >= 0
                    else ""
                ),
            )
        ]
        candidates = [
            candidate for candidate in candidates
            if (
                classify_numeric_role(
                    region,
                    candidate[0].start(),
                    candidate[0].end(),
                    unit=candidate[2],
                    relation=requirement.kind,
                ) not in {"filter_threshold", "entity_identifier", "document_identifier"}
                and (
                    classify_numeric_role(
                        region,
                        candidate[0].start(),
                        candidate[0].end(),
                        unit=candidate[2],
                        relation=requirement.kind,
                    ) != "confidence_level"
                    or bool(re.search(
                        r"confidence|ביטחון",
                        f"{requirement.metric} {requirement.label}",
                        re.IGNORECASE,
                    ))
                )
                and (
                    not requirement.population
                    or _population_scope_compatible(
                        requirement.population,
                        _evidence_population(
                            _candidate_clause(region, candidate[0])[0]
                        ),
                    )
                )
            )
        ]
        if not candidates:
            continue
        period = requirement.period
        quality = 0
        selected: Optional[tuple[re.Match[str], str, str]] = None
        if period:
            period_matches = list(re.finditer(
                rf"(?<!\d){re.escape(period)}(?!\d)",
                region,
            ))
            ranked: list[
                tuple[
                    int,
                    int,
                    int,
                    int,
                    tuple[re.Match[str], str, str],
                ]
            ] = []
            for period_match in period_matches:
                for candidate in candidates:
                    match = candidate[0]
                    distance = min(
                        abs(match.start() - period_match.end()),
                        abs(period_match.start() - match.end()),
                    )
                    if distance > 140:
                        continue
                    association = _period_candidate_quality(
                        region,
                        period_match,
                        match,
                    )
                    ranked.append((
                        -association,
                        distance,
                        -_candidate_metric_association(
                            requirement,
                            region,
                            candidate,
                        ),
                        match.start(),
                        candidate,
                    ))
            if ranked:
                (
                    association_rank,
                    _distance,
                    metric_rank,
                    _position,
                    selected,
                ) = min(ranked)
                quality = -association_rank - metric_rank
            elif (
                requirement.time_role == "current"
                and _CURRENT_TIME_RE.search(region)
            ):
                cue = list(_CURRENT_TIME_RE.finditer(region))[-1]
                selected = max(
                    candidates,
                    key=lambda candidate: _time_candidate_rank(
                        requirement,
                        region,
                        cue,
                        candidate,
                    ),
                )
                compact = re.sub(r"\D", "", selected[1])
                quality = 4 + int(
                    bool(compact) and not compact.endswith("00")
                ) + _candidate_metric_association(
                    requirement,
                    region,
                    selected,
                )
            elif (
                requirement.time_role == "historical"
                and _HISTORICAL_TIME_RE.search(region)
            ):
                cue = next(_HISTORICAL_TIME_RE.finditer(region))
                selected = max(
                    candidates,
                    key=lambda candidate: _time_candidate_rank(
                        requirement,
                        region,
                        cue,
                        candidate,
                    ),
                )
                quality = 4 + _candidate_metric_association(
                    requirement,
                    region,
                    selected,
                )
            elif (
                requirement.time_role == "current"
                and not _YEAR_RE.search(region)
                and not _HISTORICAL_TIME_RE.search(region)
                and not _FORECAST_FACET_RE.search(region)
                and (
                    _YEAR_RE.search(unit.canonical_text)
                    or _HISTORICAL_TIME_RE.search(unit.canonical_text)
                )
            ):
                selected = max(
                    candidates,
                    key=lambda candidate: (
                        _candidate_metric_association(
                            requirement,
                            region,
                            candidate,
                        ),
                        -candidate[0].start(),
                    ),
                )
                association = _candidate_metric_association(
                    requirement,
                    region,
                    selected,
                )
                if (
                    association >= 2
                    or (
                        len(candidates) == 1
                        and max(
                            _term_overlap(requirement.metric, region),
                            _semantic_term_overlap(
                                requirement.metric,
                                region,
                            ),
                        ) >= 0.30
                    )
                ):
                    period = requirement.period or "current"
                    quality = 3 + association
                else:
                    selected = None
        else:
            associated: list[
                tuple[
                    int,
                    int,
                    int,
                    int,
                    str,
                    tuple[re.Match[str], str, str],
                ]
            ] = []
            for year_match in _YEAR_RE.finditer(region):
                for candidate in candidates:
                    distance = min(
                        abs(candidate[0].start() - year_match.end()),
                        abs(year_match.start() - candidate[0].end()),
                    )
                    if distance <= 140:
                        associated.append((
                            -_period_candidate_quality(
                                region,
                                year_match,
                                candidate[0],
                            ),
                            distance,
                            -_candidate_metric_association(
                                requirement,
                                region,
                                candidate,
                            ),
                            candidate[0].start(),
                            year_match.group(0),
                            candidate,
                        ))
            years = sorted({item[4] for item in associated})
            role_cue_present = bool(
                (
                    requirement.time_role == "current"
                    and _CURRENT_TIME_RE.search(region)
                )
                or (
                    requirement.time_role == "historical"
                    and _HISTORICAL_TIME_RE.search(region)
                )
            )
            chosen_year = (
                years[0]
                if years
                and requirement.time_role == "historical"
                and role_cue_present
                else years[-1]
                if years
                and requirement.time_role == "current"
                and role_cue_present
                else ""
            )
            choices = [
                item for item in associated if item[4] == chosen_year
            ]
            if choices:
                (
                    quality_rank,
                    _distance,
                    metric_rank,
                    _position,
                    period,
                    selected,
                ) = min(choices)
                quality = -quality_rank - metric_rank
            elif (
                requirement.time_role == "current"
                and _CURRENT_TIME_RE.search(region)
            ) or (
                requirement.time_role == "historical"
                and _HISTORICAL_TIME_RE.search(region)
            ):
                cue_pattern = (
                    _CURRENT_TIME_RE
                    if requirement.time_role == "current"
                    else _HISTORICAL_TIME_RE
                )
                cue_matches = list(cue_pattern.finditer(region))
                cue = (
                    cue_matches[-1]
                    if requirement.time_role == "current"
                    else cue_matches[0]
                )
                selected = max(
                    candidates,
                    key=lambda candidate: _time_candidate_rank(
                        requirement,
                        region,
                        cue,
                        candidate,
                    ),
                )
                period = requirement.time_role
                compact = re.sub(r"\D", "", selected[1])
                quality = 4 + int(
                    requirement.time_role == "current"
                    and bool(compact)
                    and not compact.endswith("00")
                ) + _candidate_metric_association(
                    requirement,
                    region,
                    selected,
                )
            elif (
                requirement.time_role == "current"
                and _HISTORICAL_TIME_RE.search(region)
            ):
                historical_cue = next(
                    _HISTORICAL_TIME_RE.finditer(region)
                )
                current_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate[0].end() <= historical_cue.start()
                ]
                if current_candidates:
                    selected = max(
                        current_candidates,
                        key=lambda candidate: (
                            _candidate_metric_association(
                                requirement,
                                region,
                                candidate,
                            ),
                            candidate[0].start(),
                        ),
                    )
                    period = "current"
                    quality = 4 + _candidate_metric_association(
                        requirement,
                        region,
                        selected,
                    )
            elif (
                requirement.time_role == "current"
                and not _YEAR_RE.search(region)
                and not _HISTORICAL_TIME_RE.search(region)
                and not _FORECAST_FACET_RE.search(region)
                and (
                    _YEAR_RE.search(unit.canonical_text)
                    or _HISTORICAL_TIME_RE.search(unit.canonical_text)
                )
            ):
                selected = max(
                    candidates,
                    key=lambda candidate: (
                        _candidate_metric_association(
                            requirement,
                            region,
                            candidate,
                        ),
                        -candidate[0].start(),
                    ),
                )
                association = _candidate_metric_association(
                    requirement,
                    region,
                    selected,
                )
                if (
                    association >= 2
                    or (
                        len(candidates) == 1
                        and max(
                            _term_overlap(requirement.metric, region),
                            _semantic_term_overlap(
                                requirement.metric,
                                region,
                            ),
                        ) >= 0.30
                    )
                ):
                    period = "current"
                    quality = 3 + association
                else:
                    selected = None
            elif (
                requirement.time_role == "current"
                and not _YEAR_RE.search(region)
                and not _HISTORICAL_TIME_RE.search(region)
                and not _FORECAST_FACET_RE.search(region)
            ):
                selected = max(
                    candidates,
                    key=lambda candidate: (
                        _candidate_metric_association(
                            requirement,
                            region,
                            candidate,
                        ),
                        -candidate[0].start(),
                    ),
                )
                association = _candidate_metric_association(
                    requirement,
                    region,
                    selected,
                )
                if association >= 4 and _CURRENT_TIME_RE.search(
                    unit.canonical_text
                ):
                    period = "current"
                    quality = 3 + association
                else:
                    selected = None
            elif not requirement.time_role and len(candidates) > 1:
                scored = [
                    (
                        _candidate_metric_association(
                            requirement, region, candidate
                        ),
                        candidate,
                    )
                    for candidate in candidates
                ]
                plausible = [item for item in scored if item[0] >= 2]
                best_score = max((item[0] for item in plausible), default=-1)
                best_candidates = [
                    item for item in plausible if item[0] == best_score
                ]
                second_score = max(
                    (item[0] for item in plausible if item[0] < best_score),
                    default=-1,
                )
                explicitly_bound: list[
                    tuple[int, tuple[re.Match[str], str, str]]
                ] = []
                requested_terms = {
                    term for term in _normalized_terms(requirement.metric)
                    if term not in {
                        "count", "number", "total", "companies", "company",
                        "חברות", "חברה", "מספר", "כמה", "מהן", "מתוכן",
                    }
                }
                predicate_bound: list[
                    tuple[int, tuple[re.Match[str], str, str]]
                ] = []
                target_bound: list[
                    tuple[int, tuple[re.Match[str], str, str]]
                ] = []
                requested_evidence_predicate = _evidence_predicate(
                    requirement.metric,
                    numeric_target=requirement.metric,
                    unit=requirement.expected_unit,
                )
                concrete_predicate_required = bool(
                    canonical_requirement is not None
                    and canonical_requirement.predicate not in {
                        "has_count", "has_percentage", "stated_as",
                        "compared_with", "changed_between",
                    }
                )
                predicate_candidates = (
                    scored if concrete_predicate_required else plausible
                )
                for item in predicate_candidates:
                    candidate = item[1]
                    local, _local_start, _local_end, local_match = (
                        _numeric_local_proposition(region, candidate[0])
                    )
                    if requested_terms & set(_normalized_terms(local)):
                        explicitly_bound.append(item)
                    local_target = _numeric_evidence_target(
                        local,
                        local_match,
                        candidate[2],
                        row_header=row_header,
                    )
                    if max(
                        _term_overlap(requirement.metric, local_target),
                        _semantic_term_overlap(requirement.metric, local_target),
                    ) >= 0.5:
                        target_bound.append(item)
                    if concrete_predicate_required:
                        (
                            local_proposition,
                            _local_start,
                            _local_end,
                            local_match,
                        ) = _numeric_local_proposition(region, candidate[0])
                        local_target = _numeric_evidence_target(
                            local_proposition,
                            local_match,
                            candidate[2],
                            row_header=row_header,
                        )
                        local_subject = _evidence_subject(
                            local_proposition,
                            local_target,
                            row_header=row_header,
                            numeric_match=local_match,
                        )
                        local_predicate = _evidence_predicate(
                            local_proposition,
                            numeric_target=local_target,
                            unit=candidate[2],
                        )
                        if not _evidence_fact_compatibility_errors(
                            canonical_requirement,
                            subject=local_subject,
                            predicate=local_predicate,
                            target_entity=local_target,
                            population=_evidence_population(local_proposition),
                        ):
                            predicate_bound.append(item)
                    elif requested_evidence_predicate not in {
                        "", "has_count", "has_percentage"
                    } and _evidence_predicate(
                        local,
                        numeric_target=local_target,
                        unit=candidate[2],
                    ) == requested_evidence_predicate:
                        predicate_bound.append(item)
                if (
                    len(predicate_bound) == 1
                    and (
                        concrete_predicate_required
                        or predicate_bound[0][0] >= 2
                    )
                ):
                    association, selected = predicate_bound[0]
                    period = ""
                    quality = 3 + association
                elif len(target_bound) == 1 and target_bound[0][0] >= 2:
                    association, selected = target_bound[0]
                    period = ""
                    quality = 3 + association
                elif (
                    len(explicitly_bound) == 1
                    and best_score >= 3
                    and best_score > second_score
                ):
                    association, selected = explicitly_bound[0]
                    period = ""
                    quality = 3 + association
                else:
                    reject("numeric_binding_ambiguous")
                    selected = None
            elif not requirement.time_role and len(candidates) == 1:
                selected = candidates[0]
                period = ""
                quality = 3 + _candidate_metric_association(
                    requirement,
                    region,
                    selected,
                )
        if selected is None:
            continue
        if (
            not requirement.period
            and
            _metric_focus_spans(requirement.metric, region)
            and _candidate_metric_association(
                requirement,
                region,
                selected,
            ) < 2
        ):
            continue
        match, raw_value, unit_label = selected
        numeric_role = classify_numeric_role(
            region,
            match.start(),
            match.end(),
            unit=unit_label,
            relation=requirement.kind,
        )
        if numeric_role in {
            "filter_threshold", "period", "table_header", "unrelated_neighbor",
            "entity_identifier", "document_identifier",
        }:
            continue
        if numeric_role == "confidence_level" and not re.search(
            r"confidence|ביטחון",
            f"{requirement.metric} {requirement.label}",
            re.IGNORECASE,
        ):
            continue
        prefix = region[max(0, match.start() - 24):match.start()]
        qualifiers: list[str] = []
        if _APPROX_RE.search(prefix):
            qualifiers.append("approximate")
        if requirement.facet in {"lower_band", "higher_band"}:
            qualifiers.append(_band_qualifier(region, requirement.facet))
        (
            evidence_region,
            evidence_region_start,
            evidence_region_end,
            evidence_match,
        ) = _numeric_local_proposition(region, match)
        evidence_target = _numeric_evidence_target(
            evidence_region,
            evidence_match,
            unit_label,
            row_header=row_header,
        )
        evidence_subject = _evidence_subject(
            evidence_region,
            evidence_target,
            row_header=row_header,
            numeric_match=evidence_match,
        )
        evidence_predicate = _evidence_predicate(
            evidence_region, numeric_target=evidence_target, unit=unit_label
        )
        evidence_population = _evidence_population(
            evidence_region, value_span=evidence_match.span(),
        )
        evidence_period = _evidence_period(evidence_region, period)
        if (
            not evidence_period
            and requirement.time_role in {"current", "historical"}
            and (
                _CURRENT_TIME_RE.search(unit.canonical_text)
                or _HISTORICAL_TIME_RE.search(unit.canonical_text)
            )
        ):
            evidence_period = requirement.time_role
        absolute_span = (
            max(0, unit.canonical_text.find(claim)) + evidence_region_start,
            max(0, unit.canonical_text.find(claim)) + evidence_region_end,
        )
        evidence_polarity = (
            "negative" if _EVIDENCE_NEGATION_RE.search(evidence_region)
            else "positive" if evidence_predicate == "included_in"
            else "neutral"
        )
        numeric_context = _evidence_context_for_proposition(
            unit,
            evidence_region,
            absolute_span[0],
            absolute_span[1],
            {
                "subject": evidence_subject,
                "predicate": evidence_predicate,
                "target_entity": evidence_target,
                "population": evidence_population,
                "period": evidence_period,
                "category": row_header,
                "polarity": evidence_polarity,
            },
            evidence_unit_value=unit_label,
            numeric_role=numeric_role,
        )
        atoms = evidence_atoms_for_context(
            numeric_context,
            value=raw_value,
            unit=unit_label,
            numeric_role=numeric_role,
        )
        if len(atoms) != 1:
            reject("evidence_atom_unresolved")
            continue
        atom = replace(
            atoms[0], source_evidence_unit_ids=unit.source_evidence_unit_ids
        )
        recorder = current_eval_trace() if record_trace else None
        if recorder is not None and hasattr(recorder, "record_evidence_atom"):
            recorder.record_evidence_atom(atom)
        semantic_errors = (
            _evidence_fact_compatibility_errors(
                canonical_requirement,
                subject=numeric_context.evidence_subject,
                predicate=numeric_context.evidence_predicate,
                target_entity=numeric_context.evidence_target_entity,
                population=numeric_context.population,
            )
            if canonical_requirement is not None else ()
        )
        if semantic_errors and any(
            reason.endswith("_unresolved") for reason in semantic_errors
        ):
            resolved_context = _maybe_resolve_evidence_semantics(
                unit,
                numeric_context,
                evidence_semantic_resolver,
                attempted_unit_ids,
            )
            if resolved_context is not None:
                numeric_context = resolved_context
                semantic_errors = _evidence_fact_compatibility_errors(
                    canonical_requirement,
                    subject=numeric_context.evidence_subject,
                    predicate=numeric_context.evidence_predicate,
                    target_entity=numeric_context.evidence_target_entity,
                    population=numeric_context.population,
                )
        if semantic_errors:
            reject(*semantic_errors)
            continue
        structural_errors = structural_binding_validation_errors(
            numeric_context
        )
        if structural_errors:
            reject(*structural_errors)
            continue
        evidence_context = f"{row_header} {column_header} {evidence_region}"
        evidence_facet = (
            "lower_band" if _LOWER_BAND_RE.search(evidence_context)
            else "higher_band" if _HIGHER_BAND_RE.search(evidence_context)
            else "experience_band" if _EXPERIENCE_FACET_RE.search(evidence_context)
            else "education_level" if _EDUCATION_FACET_RE.search(evidence_context)
            else "role" if re.search(r"\b(?:role|position|job)\b|(?:תפקיד|משרה)", evidence_context, re.I)
            else ""
        )
        fact = TypedFact(
            requirement_id=requirement.id,
            requirement_ids=(requirement.id,),
            origin_requirement_id=requirement.origin_id,
            subquestion_id=requirement.subquestion_id,
            subject=numeric_context.evidence_subject,
            metric=(
                numeric_context.evidence_subject
                or numeric_context.evidence_target_entity
                or numeric_context.evidence_predicate
            ),
            facet=evidence_facet,
            period=numeric_context.period,
            time_role=(
                "current" if evidence_period == "current" or _CURRENT_TIME_RE.search(evidence_region)
                else "historical" if evidence_period == "historical" or _HISTORICAL_TIME_RE.search(evidence_region)
                else ""
            ),
            value=raw_value,
            unit=unit_label,
            numeric_role=numeric_role,
            # The fact inherits the type already canonicalized by its source
            # context.  A legacy EvidenceUnit may not carry a structural unit
            # at all, so its old coarse label must never re-enter here and
            # create a false type-transition rejection.
            structural_type=numeric_context.structural_type,
            population=numeric_context.population,
            row_column_bound=(
                unit.structural_type not in {
                    StructuralUnitType.TABLE_ROW.value,
                    StructuralUnitType.CHART_DATA.value,
                    StructuralEvidenceType.TABLE_ROW.value,
                    StructuralEvidenceType.CHART_CATEGORY_RECORD.value,
                }
                or bool(row_header and column_header)
            ),
            row_header=row_header,
            column_header=column_header,
            evidence_span=absolute_span,
            evidence_span_sha256=hashlib.sha256(
                evidence_region.encode("utf-8")
            ).hexdigest(),
            direction=_direction_in_claim(evidence_region),
            qualifiers=tuple(dict.fromkeys(qualifiers)),
            evidence_id=unit.id,
            source_claim=evidence_region,
            evidence_subject=numeric_context.evidence_subject,
            evidence_predicate=numeric_context.evidence_predicate,
            evidence_target_entity=numeric_context.evidence_target_entity,
            evidence_population=numeric_context.population,
            evidence_period=numeric_context.period,
            evidence_unit=numeric_context.evidence_unit,
            evidence_numeric_role=numeric_context.evidence_numeric_role,
            evidence_polarity=numeric_context.evidence_polarity,
            candidate_id=unit.candidate_id,
            source_evidence_unit_ids=unit.source_evidence_unit_ids,
            evidence_context=numeric_context,
            model_assisted_semantics=numeric_context.model_assisted,
            evidence_atom=atom,
            association_quality=(
                quality
                - (6 if not balanced_delimiters(claim) else 0)
                - (
                    8
                    if not requirement.period
                    and not requirement.time_role
                    and _YEAR_RE.search(claim)
                    else 0
                )
                - (
                    4
                    if _REVERSED_ADJACENT_DELIMITER_RE.search(claim)
                    else 0
                )
            ),
        )
        binding_errors = (
            numeric_fact_binding_errors(canonical_requirement, fact)
            if canonical_requirement is not None else ()
        )
        if binding_errors:
            reject(*binding_errors)
            continue
        if not fact.structural_binding_valid:
            reject(*fact.structural_validation_errors)
            continue
        if best is None or fact.association_quality > best.association_quality:
            best = fact
    if best is None:
        terminal("REJECTED", "no_compatible_source_fact")
    else:
        terminal("FACT_CREATED", fact=best)
    return best


def _source_period_for_numeric(
    proposition: str,
    numeric_match: re.Match[str],
    *,
    source_context: str = "",
) -> str:
    """Bind a numeric value to its own explicit source period, if any."""
    directly_bound = [
        period_match.group(0)
        for period_match in _YEAR_RE.finditer(proposition)
        if period_match.start() >= numeric_match.end()
        and re.fullmatch(
            r"\s*(?:[%A-Za-z\u05d0-\u05ea]+\s+)?"
            r"(?:in|during|at|בשנת|בשנה|ב[־-])\s*",
            proposition[numeric_match.end():period_match.start()],
            re.IGNORECASE,
        )
    ]
    if len(directly_bound) == 1:
        return directly_bound[0]
    if len(directly_bound) > 1:
        return ""
    candidates: list[tuple[int, int, str]] = []
    for period_match in _YEAR_RE.finditer(proposition):
        distance = min(
            abs(numeric_match.start() - period_match.end()),
            abs(period_match.start() - numeric_match.end()),
        )
        if distance > 140:
            continue
        candidates.append((
            -_period_candidate_quality(proposition, period_match, numeric_match),
            distance,
            period_match.group(0),
        ))
    if not candidates:
        period = _evidence_period(proposition) or _evidence_period(source_context)
        if period:
            return period
        if re.search(
            r"\b(?:is|are|currently|presently)\b|(?:נוכחי(?:ת|ים|ות)?)",
            f"{proposition} {source_context}",
            re.IGNORECASE,
        ):
            return "current"
        return ""
    candidates.sort()
    best_score, best_distance, best_period = candidates[0]
    tied_periods = {
        period for score, distance, period in candidates
        if (score, distance) == (best_score, best_distance)
    }
    return best_period if len(tied_periods) == 1 else ""


def _structured_scalar_owner(unit: EvidenceUnit, claim_start: int, match: re.Match[str]):
    """Find the one explicit field owning this scalar in its original row."""
    structural = unit.structural_evidence_unit
    start = claim_start + match.start("value")
    end = claim_start + match.end("value")
    owners = [binding for binding in structural.field_bindings
              if binding.role == "value" and binding.field_span is not None
              and binding.source_span[0] <= start < end <= binding.source_span[1]]
    if len(owners) != 1:
        return None
    owner = owners[0]
    from document_query import role_bound_identifier_spans
    # The structured producer already proves this scalar's exact label and
    # value owner. Remove only its worksheet-letter decoration when reading
    # an identifier role; retain the original field and spans as provenance.
    label = re.sub(r"\s*\([A-Z]{1,3}\)\s*$", "", owner.column_role)
    owned_value = str(owner.value or "").strip()
    if any(identifier.identifier == owned_value.casefold()
           and identifier.identifier_span[0] >= len(label)
           for identifier in role_bound_identifier_spans(label + " " + owned_value)):
        return None
    field_start, field_end = owner.field_span
    field_text = unit.canonical_text[field_start:field_end]
    local_candidates = [candidate for candidate in _numeric_candidates(field_text, "")
                        if field_start + candidate[0].start("value") == start
                        and field_start + candidate[0].end("value") == end]
    if not local_candidates and _YEAR_RE.fullmatch(match.group("value")):
        # An explicit quantity/price cell can legitimately equal a year-like
        # integer. Only an actual time column makes that scalar a period.
        if re.search(r"\b(?:year|date)\b|(?:שנה|תאריך)", owner.column_role, re.IGNORECASE):
            return None
        local_candidates = [(match, match.group("value"), "number")]
    if len(local_candidates) != 1:
        return None
    unit_label = local_candidates[0][2]
    suffix = field_text[end - field_start:].strip()
    if unit_label == "number" and re.fullmatch(r"[A-Za-z\u05d0-\u05ea'׳\"״]{1,24}", suffix):
        unit_label = suffix
    if unit_label == "number":
        header_unit = re.search(r"\((kg|g|mm|cm|m|l|ml|kw|w)\)", owner.column_role)
        if header_unit:
            unit_label = header_unit.group(1)
    return owner, field_text, field_start - claim_start, unit_label


def _field_quantity_semantics(text, match, *, unit_label, row_header, column_header,
                             source_metric_label, source_container="", owner=None):
    field_offset = owner[2] if owner else 0
    source = owner[1] if owner else text
    semantics = classify_quantity_semantics(
        source,
        (match.start("value") if owner else match.start()) - field_offset,
        (match.end("value") if owner else match.end()) - field_offset,
        unit=unit_label,
        structural_role=f"row:{row_header}|column:{column_header}" if row_header and column_header else "",
        source_container="" if owner else source_container,
        source_metric_label=source_metric_label,
    )
    if not owner:
        return semantics
    if semantics.quantity_kind == QuantityKind.PERIOD.value:
        semantics = replace(semantics, quantity_kind=(
            QuantityKind.COUNT.value
            if _source_metric_label_establishes_cardinality(source_metric_label)
            else QuantityKind.NUMBER.value
        ))
    if (semantics.quantity_kind == QuantityKind.COUNT.value
            and unit_label.casefold() not in {"", "number", "numeric", "pcs", "pieces", "ea"}
            and not _COUNT_UNIT_RE.fullmatch(unit_label)):
        # A quantity header alone cannot turn a mass/distance/unknown unit
        # into a discrete count. Keep that literal unit, without guessing.
        semantics = replace(semantics, quantity_kind=QuantityKind.NUMBER.value)
    return replace(semantics, **{
        name: ((span[0] + field_offset, span[1] + field_offset) if span else None)
        for name in ("statistical_role_cue_span", "aggregation_role_cue_span")
        for span in (getattr(semantics, name),)
    })


def _source_numeric_facts_for_unit(
    unit: EvidenceUnit,
    *,
    source_fact_occurrence_registry: Optional[
        dict[str, TypedFact | TextPropositionFact]
    ] = None,
) -> tuple[tuple[TypedFact, ...], tuple[str, ...]]:
    """Construct canonical numeric SOURCE_FACTs from one bounded source unit.

    This boundary intentionally has no request or requirement parameter.  It
    may reject only missing source bindings, structural defects, or a genuinely
    ambiguous atom.  Request compatibility is evaluated only after the source
    fact exists and has a stable, source-derived identity.
    """
    ineligible = {
        StructuralUnitType.HEADING.value,
        StructuralUnitType.CAPTION.value,
        StructuralUnitType.TABLE_OF_CONTENTS.value,
        StructuralUnitType.LIST_OF_FIGURES.value,
        StructuralUnitType.LIST_OF_TABLES.value,
        StructuralUnitType.HEADER_FOOTER.value,
        StructuralUnitType.PAGE_NUMBER.value,
        StructuralUnitType.FORMULA.value,
        StructuralUnitType.MALFORMED.value,
    }
    if unit.structural_type in ineligible:
        recorder = current_eval_trace()
        if recorder is not None and hasattr(
            recorder, "record_evidence_atom_terminal_outcome"
        ):
            recorder.record_evidence_atom_terminal_outcome(
                evidence_id=unit.id,
                candidate_id=unit.candidate_id,
                outcome="REJECTED",
                reason_codes=("structural_type_ineligible",),
                binding_status="structural_type_ineligible",
            )
        return (), ("structural_type_ineligible",)
    metadata = dict(unit.raw_hit.get("metadata") or {})
    structural_unit = unit.structural_evidence_unit
    structured_fields = bool(structural_unit is not None and
                             structural_unit.segmentation_method == "ingested_structured_record")
    flattened_chart_record = bool(
        structural_unit is not None
        and structural_unit.derivation_type
        == "explicit_source_chart_value_category_pair"
    )
    chart_metric = next(
        (
            binding.value
            for binding in (structural_unit.field_bindings if structural_unit else ())
            if binding.role == "entity" and binding.column_role == "metric"
        ),
        "",
    )
    chart_category = next(
        (
            binding.value
            for binding in (structural_unit.field_bindings if structural_unit else ())
            if binding.role == "category" and binding.column_role == "category"
        ),
        "",
    )
    if flattened_chart_record and (not chart_metric or not chart_category):
        recorder = current_eval_trace()
        if recorder is not None and hasattr(
            recorder, "record_evidence_atom_terminal_outcome"
        ):
            recorder.record_evidence_atom_terminal_outcome(
                evidence_id=unit.id,
                candidate_id=unit.candidate_id,
                outcome="REJECTED",
                reason_codes=("chart_value_category_binding_unresolved",),
                binding_status="atom_boundary_unresolved",
            )
        return (), ("chart_value_category_binding_unresolved",)
    row_header = normalize_request_text(
        (
            structural_unit.row_label
            if structural_unit is not None and not flattened_chart_record else ""
        )
        or metadata.get("row_header")
        or metadata.get("table_row_header")
        or metadata.get("category")
        or ""
    )
    column_header = normalize_request_text(
        chart_metric if flattened_chart_record else metadata.get("column_header")
        or metadata.get("table_column_header")
        or metadata.get("value_header")
        or (
            " ".join(structural_unit.column_roles[1:])
            if structural_unit is not None
            and len(structural_unit.column_roles) > 1
            else ""
        )
        or ""
    )
    source_metric_label = chart_metric if flattened_chart_record else column_header
    table_like = (
        structural_unit is not None
        and not flattened_chart_record
        and unit.structural_type in {
        StructuralUnitType.TABLE_ROW.value,
        StructuralUnitType.CHART_DATA.value,
        StructuralEvidenceType.TABLE_ROW.value,
        StructuralEvidenceType.CHART_CATEGORY_RECORD.value,
        }
    )
    if table_like and (not row_header or not column_header):
        reasons = tuple(
            code for code, value in (
                ("row_header_unbound", row_header),
                ("column_header_unbound", column_header),
            ) if not value
        )
        recorder = current_eval_trace()
        if recorder is not None and hasattr(
            recorder, "record_evidence_atom_terminal_outcome"
        ):
            recorder.record_evidence_atom_terminal_outcome(
                evidence_id=unit.id,
                candidate_id=unit.candidate_id,
                outcome="REJECTED",
                reason_codes=reasons,
                binding_status="atom_boundary_unresolved",
            )
        return (), reasons

    facts: list[TypedFact] = []
    construction_reasons: list[str] = []
    seen_atoms: set[str] = set()
    structural_reference_spans = tuple(
        binding.source_span
        for binding in (structural_unit.field_bindings if structural_unit else ())
        if (
            binding.role == "structural_reference"
            and binding.column_role == "non_factual_ordinal"
        )
    )
    source_claims = tuple(unit.allowed_claims)
    for claim_index, claim in enumerate(source_claims):
        # Structural extraction can split a governing proposition from an
        # immediately following bound fragment.  Keep only that adjacent
        # source parent for a demonstrably continuative fragment; it never
        # crosses a structural unit or incorporates request semantics.
        bounded_parent = ""
        if claim_index and re.match(
            r"\s*(?:the\s+)?(?:lower|upper)\s+bound\b|"
            r"\s*(?:of\s+which|מתוכ(?:ו|ן)|מהם|מהן)\b",
            claim,
            re.IGNORECASE,
        ):
            candidate_parent = str(source_claims[claim_index - 1] or "").strip()
            if candidate_parent and candidate_parent in unit.canonical_text:
                bounded_parent = candidate_parent
        source_container_claim = " ".join(
            item for item in (bounded_parent, claim) if item
        )
        if not structured_fields and _MALFORMED_ADJACENT_GROUPED_NUMBERS_RE.search(claim):
            construction_reasons.append("numeric_source_span_malformed")
            recorder = current_eval_trace()
            if recorder is not None and hasattr(
                recorder, "record_evidence_atom_terminal_outcome"
            ):
                recorder.record_evidence_atom_terminal_outcome(
                    evidence_id=unit.id,
                    candidate_id=unit.candidate_id,
                    outcome="REJECTED",
                    reason_codes=("numeric_source_span_malformed",),
                    binding_status="atom_boundary_unresolved",
                )
            continue
        if not structured_fields and re.search(
            r"\b\d+(?:[.,]\d+)?\s*(?:or|או)\s*\d+(?:[.,]\d+)?\b",
            claim,
            re.IGNORECASE,
        ):
            construction_reasons.append("source_numeric_binding_ambiguous")
            recorder = current_eval_trace()
            if recorder is not None and hasattr(
                recorder, "record_evidence_atom_terminal_outcome"
            ):
                recorder.record_evidence_atom_terminal_outcome(
                    evidence_id=unit.id,
                    candidate_id=unit.candidate_id,
                    outcome="REJECTED",
                    reason_codes=("source_numeric_binding_ambiguous",),
                    binding_status="no_local_role",
                )
            continue
        claim_position = unit.canonical_text.find(claim)
        numeric_candidates = _numeric_candidates(claim, "")
        if structured_fields:
            numeric_candidates = [
                (item, normalize_evidence_whitespace(item.group("value")).replace(" ", ""), "number")
                for item in _NUMBER_WITH_OPTIONAL_UNIT_RE.finditer(claim)
            ]
        for match, raw_value, unit_label in numeric_candidates:
            owner = _structured_scalar_owner(unit, max(0, claim_position), match) if structured_fields else None
            if structured_fields:
                if owner is None:
                    continue
                column_header = owner[0].column_role
                source_metric_label = column_header
                unit_label = owner[3]
            claim_span = (claim_position + match.start(), claim_position + match.end())
            if any(
                start < claim_span[1] and claim_span[0] < end
                for start, end in structural_reference_spans
            ):
                construction_reasons.append("source_structural_reference_ineligible")
                continue
            numeric_role = classify_numeric_role(
                owner[1] if owner else claim,
                match.start() - (owner[2] if owner else 0),
                match.end() - (owner[2] if owner else 0), unit=unit_label,
                relation="source_fact",
            )
            if numeric_role in {"entity_identifier", "document_identifier"}:
                # Identity tokens remain available to nominal/text facts.
                # They must not become a measured scalar merely because an
                # adjacent source field carries a count or statistical role.
                construction_reasons.append(
                    f"source_numeric_role_ineligible:{numeric_role}"
                )
                continue
            if owner and numeric_role == "period":
                numeric_role = "requested_answer_value"
            quantity_semantics = _field_quantity_semantics(
                claim, match, unit_label=unit_label, row_header=row_header,
                column_header=column_header, owner=owner,
                source_metric_label=source_metric_label,
            )
            scalar_role = quantity_semantics.scalar_role
            # Compatibility consumes ``scalar_role``.  Keep the legacy
            # numeric projection coherent for older renderers and validators
            # instead of allowing a nearby confidence phrase to relabel a
            # point estimate as a confidence value.
            numeric_role = {
                "confidence_level": "confidence_level",
                "lower_bound": "lower_bound",
                "upper_bound": "requested_answer_value",
                "filter_threshold": "filter_threshold",
                "percentage_share": "percentage",
                "category_count": "category_value",
                "period_value": "period",
            }.get(scalar_role, (
                "requested_answer_value"
                if numeric_role in {"confidence_level", "lower_bound"}
                else numeric_role
            ))
            if numeric_role in {
                "period", "table_header", "unrelated_neighbor",
            }:
                construction_reasons.append(
                    f"source_numeric_role_ineligible:{numeric_role}"
                )
                continue
            (
                evidence_region,
                evidence_region_start,
                evidence_region_end,
                evidence_match,
            ) = _numeric_local_proposition(claim, match)
            if structured_fields:
                # Keep the contiguous row for source identity and citation,
                # while scalar semantics come solely from the owning field.
                evidence_region, evidence_region_start, evidence_region_end, evidence_match = claim, 0, len(claim), match
            evidence_target = _numeric_evidence_target(
                evidence_region, evidence_match, unit_label,
                row_header=row_header,
            )
            if re.fullmatch(r"[A-Z]{1,6}", evidence_target or ""):
                # A compact engineering/scientific unit can immediately
                # follow a value without being the subject it measures.
                # Preserve it as source text but do not promote it to a
                # semantic target.
                evidence_target = ""
            evidence_subject = _evidence_subject(
                evidence_region, evidence_target, row_header=row_header,
                numeric_match=evidence_match,
            )
            evidence_predicate = _evidence_predicate(
                evidence_region, numeric_target=evidence_target, unit=unit_label,
            )
            generic_local_subject = bool(re.fullmatch(
                r"(?:workers?|employees?|companies|organizations?|firms?|"
                r"respondents?|people|jobs?|חברות|עובדים|משרות|ארגונים)",
                normalize_request_text(evidence_subject or evidence_target),
                re.IGNORECASE,
            ))
            continuation_fragment = bool(re.match(
                r"\s*(?:of\s+which|מתוכ(?:ו|ן)|מהם|מהן)\b",
                evidence_region,
                re.IGNORECASE,
            ))
            # Punctuation introduced by PDF/OCR extraction can isolate a
            # scalar (for example, after ``approx.``).  Recover only the
            # enclosing authorized claim, never request text or a different
            # evidence unit, when the local fragment cannot name its fact.
            if (
                continuation_fragment
                or (
                    generic_local_subject
                    and bool(re.search(r"\([^()]*$", claim[:match.start()]))
                )
                or (
                    generic_local_subject
                    and not _numeric_candidates(claim[:match.start()], "")
                    and bool(re.search(r"[A-Za-z\u05d0-\u05ea]", claim[:match.start()]))
                )
                or not evidence_subject
                or not evidence_predicate
            ):
                evidence_region = claim
                evidence_region_start = 0
                evidence_region_end = len(claim)
                evidence_match = match
                evidence_target = _numeric_evidence_target(
                    evidence_region, evidence_match, unit_label,
                    row_header=row_header,
                )
                if re.fullmatch(r"[A-Z]{1,6}", evidence_target or ""):
                    evidence_target = ""
                evidence_subject = _evidence_subject(
                    evidence_region,
                    evidence_target,
                    row_header=row_header,
                    numeric_match=evidence_match,
                )
                evidence_predicate = _evidence_predicate(
                    evidence_region,
                    numeric_target=evidence_target,
                    unit=unit_label,
                )
            # Percentage prose frequently names the measured subject before
            # the scalar ("response rate was 8.4%").  The subject is then the
            # source-local target for the numeric relation; this is not a
            # request-derived substitution.  Re-evaluate the numeric predicate
            # only after that source binding is available.
            if not evidence_target and evidence_subject:
                evidence_target = evidence_subject
                evidence_predicate = _evidence_predicate(
                    evidence_region,
                    numeric_target=evidence_target,
                    unit=unit_label,
                )
            relation_subject = _source_relation_subject(
                evidence_region, evidence_match, evidence_predicate,
            )
            if relation_subject:
                evidence_subject = relation_subject
            if experience_band_facets(evidence_region) and evidence_subject:
                # The adjacent experience duration describes the category,
                # not the measured percentage's target/metric. Its bounded
                # source subject owns the target for every endpoint in this
                # local series; no request field participates in this binding.
                evidence_target = evidence_subject
            absolute_span = (
                max(0, claim_position) + evidence_region_start,
                max(0, claim_position) + evidence_region_end,
            )
            # Bind roles to the same bounded source proposition that will be
            # atomized and cited.  The surrounding claim may contain an
            # adjacent but unrelated numeric cue.
            quantity_semantics = _field_quantity_semantics(
                evidence_region, evidence_match, unit_label=unit_label,
                row_header=row_header, column_header=column_header, owner=owner,
                source_container=source_container_claim,
                source_metric_label=source_metric_label,
            )
            scalar_role = quantity_semantics.scalar_role
            source_span_sha256 = hashlib.sha256(
                evidence_region.encode("utf-8")
            ).hexdigest()
            numeric_binding = _source_local_numeric_binding(
                evidence_region,
                evidence_match,
                absolute_proposition_start=absolute_span[0],
                source_span_sha256=source_span_sha256,
                unit_label=unit_label,
                row_header=row_header,
                column_header=column_header,
                scalar_role=scalar_role,
                quantity_semantics=quantity_semantics,
            )
            if owner:
                numeric_binding = replace(
                    numeric_binding, category="", period="",
                    provenance=replace(
                        numeric_binding.provenance,
                        value_span=(max(0, claim_position) + match.start("value"),
                                    max(0, claim_position) + match.end("value")),
                        semantic_label_span=owner[0].label_span,
                        period_span=None,
                    ),
                )
            # A locally stated experience band is a source-side category
            # binding for every coordinated endpoint in that proposition.
            # It is not inferred from a requested facet: the cue must occur
            # in the atom's own bounded evidence region.
            experience_band_cue = None
            if not structured_fields and _EXPERIENCE_FACET_RE.search(evidence_region):
                experience_band_cue = (
                    _LOWER_BAND_RE.search(evidence_region)
                    or _HIGHER_BAND_RE.search(evidence_region)
                )
            if (
                experience_band_cue is not None
                and numeric_binding.aggregation_role in {
                    AggregationRole.NONE.value,
                    AggregationRole.UNSPECIFIED.value,
                }
                and numeric_binding.statistical_role
                == StatisticalRole.UNSPECIFIED.value
            ):
                projected_role = scalar_role_from_quantity_semantics(
                    numeric_binding.quantity_kind,
                    numeric_binding.statistical_role,
                    AggregationRole.CATEGORY.value,
                )
                numeric_binding = replace(
                    numeric_binding,
                    scalar_role=projected_role,
                    aggregation_role=AggregationRole.CATEGORY.value,
                    provenance=replace(
                        numeric_binding.provenance,
                        scalar_role=projected_role,
                        aggregation_role=AggregationRole.CATEGORY.value,
                        aggregation_role_cue_span=(
                            max(0, claim_position)
                            + evidence_region_start
                            + experience_band_cue.start(),
                            max(0, claim_position)
                            + evidence_region_start
                            + experience_band_cue.end(),
                        ),
                    ),
                )
            # A period/value atom may begin after the governing total/share
            # label in the same source proposition.  Preserve that bounded
            # source-local aggregation cue without widening to a chunk or
            # consulting request semantics.
            if not structured_fields and numeric_binding.aggregation_role in {
                AggregationRole.NONE.value,
                AggregationRole.UNSPECIFIED.value,
            } and numeric_binding.statistical_role == StatisticalRole.UNSPECIFIED.value:
                total_cue = _TOTAL_AGGREGATION_CUE_RE.search(evidence_region)
                share_cue = re.search(
                    r"\b(?:share|percentage|percent|rate|portion|represented|"
                    r"accounted\s+for)\b|(?:שיעור|נתח|אחוז)",
                    evidence_region, re.IGNORECASE,
                )
                aggregation_role = (
                    AggregationRole.SHARE.value if share_cue else
                    AggregationRole.TOTAL.value if total_cue else ""
                )
                aggregation_cue = share_cue or total_cue
                if aggregation_role and aggregation_cue is not None:
                    projected_role = scalar_role_from_quantity_semantics(
                        numeric_binding.quantity_kind,
                        numeric_binding.statistical_role,
                        aggregation_role,
                    )
                    numeric_binding = replace(
                        numeric_binding,
                        scalar_role=projected_role,
                        aggregation_role=aggregation_role,
                        provenance=replace(
                            numeric_binding.provenance,
                            scalar_role=projected_role,
                            aggregation_role=aggregation_role,
                        aggregation_role_cue_span=(
                            max(0, claim_position)
                            + evidence_region_start
                            + aggregation_cue.start(),
                            max(0, claim_position)
                            + evidence_region_start
                            + aggregation_cue.end(),
                        ),
                        ),
                    )
            # A flattened response table can bind a percentage and count to
            # the same literal response label. That category is source-local
            # evidence for this scalar, not a request-side interpretation.
            response_category = "" if structured_fields else _source_response_category(
                evidence_region,
                evidence_match,
                leading_context=(
                    unit.canonical_text[max(0, claim_position - 80):claim_position]
                    if claim_position >= 0 else ""
                ),
            )
            if (
                response_category
                and numeric_binding.provenance.binding_method
                == "single_value_proposition"
            ):
                numeric_binding = replace(
                    numeric_binding,
                    provenance=replace(
                        numeric_binding.provenance,
                        binding_method="local_response_category",
                    ),
                )
            # A bounded atom can be the final fragment of a larger list.  Its
            # own local text may therefore look scalar even though the
            # authorized source claim contains several values.  Admit that
            # scalar only when one source-local role (category, period, or
            # structural axis) proves it distinct from its siblings.
            claim_numeric_candidates = _numeric_candidates(claim, "")
            if len(claim_numeric_candidates) > 1:
                first_numeric = claim_numeric_candidates[0][0]
                last_numeric = claim_numeric_candidates[-1][0]
                claim_has_multiple_values = not bool(re.search(
                    r"[;.!?׃]",
                    claim[first_numeric.end():last_numeric.start()],
                ))
            else:
                claim_has_multiple_values = False
            claim_distinct_units = len({
                str(candidate_unit or "number").casefold()
                for _candidate_match, _candidate_value, candidate_unit
                in claim_numeric_candidates
            }) > 1
            claim_scalar_roles = {
                classify_scalar_role(
                    claim,
                    candidate_match.start(),
                    candidate_match.end(),
                    unit=candidate_unit,
                    structural_role=(
                        f"row:{row_header}|column:{column_header}"
                        if row_header and column_header else ""
                    ),
                )
                for candidate_match, _candidate_value, candidate_unit
                in claim_numeric_candidates
            }
            claim_distinct_scalar_roles = (
                len(claim_scalar_roles) > 1
                and scalar_role_has_local_cue(
                    claim,
                    match.start(),
                    match.end(),
                    role=scalar_role,
                    unit=unit_label,
                    structural_role=(
                        f"row:{row_header}|column:{column_header}"
                        if row_header and column_header else ""
                    ),
                )
            )
            if (
                claim_distinct_scalar_roles
                and numeric_binding.provenance.binding_method
                == "single_value_proposition"
            ):
                numeric_binding = replace(
                    numeric_binding,
                    provenance=replace(
                        numeric_binding.provenance,
                        binding_method="local_scalar_role",
                    ),
                )
            continuation_total_binding = bool(re.match(
                r"\s*,?\s*(?:of\s+which|מתוכ(?:ו|ן)|מהם|מהן)\b",
                claim[match.end():],
                re.IGNORECASE,
            ))
            explicit_total_binding = bool(re.search(
                r"(?:\bnumber\s+of\b|מספר\s+)"
                r"[A-Za-z\u05d0-\u05ea'׳ -]{1,96}"
                r"(?:\bis\b|\bare\b|הוא|היא)\s*$",
                claim[:match.start()],
                re.IGNORECASE,
            ))
            claim_local_role_bound = bool(
                numeric_binding.category
                or numeric_binding.period
                or numeric_binding.provenance.structural_role
                or numeric_binding.provenance.binding_method in {
                    "local_scalar_role", "local_response_category",
                }
                or claim_distinct_units
                or continuation_total_binding
                or explicit_total_binding
            )
            if (
                not (
                    numeric_binding.disambiguates_multiple_values
                    or response_category
                )
                or (claim_has_multiple_values and not claim_local_role_bound)
            ):
                construction_reasons.extend((
                    "source_multivalue_role_unresolved",
                    "source_numeric_binding_ambiguous",
                ))
                recorder = current_eval_trace()
                if recorder is not None and hasattr(
                    recorder, "record_evidence_atom_terminal_outcome"
                ):
                    recorder.record_evidence_atom_terminal_outcome(
                        evidence_id=unit.id,
                        candidate_id=unit.candidate_id,
                        outcome="REJECTED",
                        reason_codes=(
                            "source_multivalue_role_unresolved",
                            "source_numeric_binding_ambiguous",
                        ),
                        binding_status="no_local_role",
                    )
                continue
            container_region_start = source_container_claim.find(evidence_region)
            evidence_population = "" if structured_fields else (
                _evidence_population(evidence_region, value_span=evidence_match.span())
                or (_evidence_population(source_container_claim, value_span=(
                    container_region_start + evidence_match.start(),
                    container_region_start + evidence_match.end(),
                )) if container_region_start >= 0 else "")
            )
            evidence_period = "" if structured_fields else (
                chart_category if flattened_chart_record else (
                    numeric_binding.period
                    or _source_period_for_numeric(
                        evidence_region,
                        evidence_match,
                        source_context=claim,
                    )
                )
            )
            source_value_category = "" if structured_fields else numeric_binding.category
            if (
                source_value_category
                and re.fullmatch(
                    r"(?:of\s+which|which|מתוכ(?:ו|ן)|מהם|מהן)",
                    normalize_request_text(source_value_category),
                    re.IGNORECASE,
                )
            ):
                category_suffix = evidence_region[evidence_match.end():]
                if unit_label:
                    consumed_unit = re.match(
                        rf"\s*{re.escape(unit_label)}\b", category_suffix,
                        re.IGNORECASE,
                    )
                    if consumed_unit is not None:
                        category_suffix = category_suffix[consumed_unit.end():]
                category_match = re.match(
                    r"\s*(?P<label>[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳ -]{0,80}?)"
                    r"(?=\s*(?:[,.;׃]|\b(?:and|or)\b|ו-?\s*[-+]?\d|$))",
                    category_suffix,
                    re.IGNORECASE,
                )
                if category_match is not None:
                    source_value_category = normalize_request_text(
                        category_match.group("label")
                    )
            if source_value_category:
                # A local phrase such as "lower bound" or "confidence" is a
                # statistical cue, not a category label for the same scalar.
                if numeric_binding.statistical_role != StatisticalRole.UNSPECIFIED.value:
                    source_value_category = ""
            if source_value_category:
                # A parallel source list can bind each scalar to its own
                # literal category.  Keep that category in the source fact
                # and atom rather than allowing a shared lead-in to make the
                # values interchangeable later.
                evidence_target = source_value_category
                evidence_subject = source_value_category
                evidence_predicate = _evidence_predicate(
                    evidence_region,
                    numeric_target=evidence_target,
                    unit=unit_label,
                )
                if numeric_binding.aggregation_role in {
                    AggregationRole.NONE.value,
                    AggregationRole.UNSPECIFIED.value,
                }:
                    numeric_binding = replace(
                        numeric_binding,
                        scalar_role=scalar_role_from_quantity_semantics(
                            numeric_binding.quantity_kind,
                            numeric_binding.statistical_role,
                            AggregationRole.CATEGORY.value,
                        ),
                        aggregation_role=AggregationRole.CATEGORY.value,
                        provenance=replace(
                            numeric_binding.provenance,
                            scalar_role=scalar_role_from_quantity_semantics(
                                numeric_binding.quantity_kind,
                                numeric_binding.statistical_role,
                                AggregationRole.CATEGORY.value,
                            ),
                            aggregation_role=AggregationRole.CATEGORY.value,
                            aggregation_role_cue_span=(
                                numeric_binding.provenance.semantic_label_span
                            ),
                        ),
                    )
            response_category = "" if structured_fields else _source_response_category(
                evidence_region,
                evidence_match,
                # An allowed claim can be a bounded suffix reconstructed from
                # its own structural unit.  Preserve an immediately governing
                # source response label instead of letting that suffix create
                # a duplicate unclassified value.
                leading_context=(
                    unit.canonical_text[max(0, claim_position - 80):claim_position]
                    if claim_position >= 0 else ""
                ),
            )
            subject_literal = evidence_subject or evidence_target
            subject_position = evidence_region.casefold().find(
                subject_literal.casefold()
            ) if subject_literal else -1
            metric_subject_span = (
                (
                    absolute_span[0] + subject_position,
                    absolute_span[0] + subject_position + len(subject_literal),
                ) if subject_position >= 0 else None
            )
            numeric_binding = replace(
                numeric_binding,
                provenance=replace(
                    numeric_binding.provenance,
                    metric_subject_span=metric_subject_span,
                ),
            )
            evidence_polarity = (
                "negative" if _EVIDENCE_NEGATION_RE.search(evidence_region)
                else "positive" if evidence_predicate == "included_in"
                else "neutral"
            )
            context = _evidence_context_for_proposition(
                unit,
                evidence_region,
                absolute_span[0],
                absolute_span[1],
                {
                    "subject": evidence_subject,
                    "predicate": evidence_predicate,
                    "target_entity": evidence_target,
                    "population": evidence_population,
                    "period": evidence_period,
                    "category": (
                        response_category
                        or source_value_category
                        or chart_category
                        or row_header
                    ),
                    "polarity": evidence_polarity,
                },
                evidence_unit_value=unit_label,
                numeric_role=numeric_role,
            )
            atoms = evidence_atoms_for_context(
                context, value=raw_value, unit=unit_label,
                numeric_role=numeric_role,
                scalar_role=numeric_binding.scalar_role,
                quantity_kind=numeric_binding.quantity_kind,
                statistical_role=numeric_binding.statistical_role,
                aggregation_role=numeric_binding.aggregation_role,
                numeric_binding=numeric_binding.provenance,
            )
            if len(atoms) != 1:
                construction_reasons.append("evidence_atom_unresolved")
                recorder = current_eval_trace()
                if recorder is not None and hasattr(
                    recorder, "record_evidence_atom_terminal_outcome"
                ):
                    recorder.record_evidence_atom_terminal_outcome(
                        evidence_id=unit.id,
                        candidate_id=unit.candidate_id,
                        outcome="REJECTED",
                        reason_codes=("evidence_atom_unresolved",),
                        binding_status="atom_boundary_unresolved",
                    )
                continue
            atom = replace(
                atoms[0], source_evidence_unit_ids=unit.source_evidence_unit_ids
            )
            if atom.atom_id in seen_atoms:
                continue
            seen_atoms.add(atom.atom_id)
            recorder = current_eval_trace()
            if recorder is not None and hasattr(recorder, "record_evidence_atom"):
                recorder.record_evidence_atom(atom)
            evidence_context = column_header if structured_fields else f"{row_header} {column_header} " + (
                source_container_claim if bounded_parent else evidence_region
            )
            evidence_facet = (
                "lower_band" if _LOWER_BAND_RE.search(evidence_context)
                else "higher_band" if _HIGHER_BAND_RE.search(evidence_context)
                else "experience_band" if _EXPERIENCE_FACET_RE.search(evidence_context)
                else "education_level" if _EDUCATION_FACET_RE.search(evidence_context)
                else "role" if re.search(
                    r"\b(?:role|position|job)\b|(?:תפקיד|משרה)",
                    evidence_context, re.I,
                ) else ""
            )
            fact = TypedFact(
                requirement_id="",
                requirement_ids=(),
                origin_requirement_id="",
                subquestion_id="",
                subject=evidence_subject,
                # The fact actor/subject and the measured quantity are distinct
                # semantic dimensions in a relational proposition.  A count of
                # workers recruited by a unit must not make that unit both the
                # source subject and the metric merely because it is nearby.
                metric=(column_header if structured_fields else (
                    " ".join(explicit_metric_concepts(evidence_subject)) or evidence_subject
                    if experience_band_facets(evidence_region)
                    else evidence_target or evidence_predicate)),
                facet=response_category or evidence_facet,
                period=evidence_period,
                time_role=(
                    "current" if evidence_period == "current"
                    or _CURRENT_TIME_RE.search(source_container_claim)
                    else "historical" if evidence_period == "historical"
                    or _HISTORICAL_TIME_RE.search(source_container_claim) else ""
                ),
                value=raw_value,
                unit=unit_label,
                numeric_role=numeric_role,
                scalar_role=numeric_binding.scalar_role,
                quantity_kind=numeric_binding.quantity_kind,
                statistical_role=numeric_binding.statistical_role,
                aggregation_role=numeric_binding.aggregation_role,
                structural_type=context.structural_type,
                population=evidence_population,
                row_column_bound=(not table_like or bool(row_header and column_header)),
                row_header=row_header,
                column_header=column_header,
                evidence_span=absolute_span,
                evidence_span_sha256=source_span_sha256,
                direction=_direction_in_claim(evidence_region),
                qualifiers=(
                    ("approximate",) if _APPROX_RE.search(
                        evidence_region[max(0, evidence_match.start() - 24):
                                        evidence_match.start()]
                    ) else ()
                ),
                evidence_id=unit.id,
                # Keep the claim/citation binding at the atom-local span.  A
                # separate source-only container retains a governing lead-in
                # for compatibility without widening the quote.
                source_claim=evidence_region,
                source_container_claim=source_container_claim,
                evidence_subject=evidence_subject,
                evidence_predicate=evidence_predicate,
                evidence_target_entity=evidence_target,
                evidence_population=evidence_population,
                evidence_period=evidence_period,
                evidence_unit=context.evidence_unit,
                evidence_numeric_role=context.evidence_numeric_role,
                evidence_polarity=evidence_polarity,
                candidate_id=unit.candidate_id,
                source_evidence_unit_ids=unit.source_evidence_unit_ids,
                evidence_context=context,
                evidence_atom=atom,
                numeric_binding=atom.numeric_binding,
                # This ordering is source-local evidence quality only.  It
                # never reads a requirement, so selection can prefer a
                # well-formed atom without reviving the legacy request-side
                # candidate parser.
                association_quality=(
                    1
                    - (6 if not balanced_delimiters(claim) else 0)
                    - (4 if _REVERSED_ADJACENT_DELIMITER_RE.search(claim) else 0)
                ),
            )
            fact = _bind_source_fact_occurrence(unit, fact)
            errors = list(_source_fact_semantic_completeness_errors(fact))
            if errors:
                construction_reasons.extend(errors)
                if recorder is not None and hasattr(
                    recorder, "record_source_fact_construction"
                ):
                    recorder.record_source_fact_construction(
                        fact, outcome="REJECTED", reason_codes=errors,
                    )
                if recorder is not None and hasattr(
                    recorder, "record_evidence_atom_terminal_outcome"
                ):
                    recorder.record_evidence_atom_terminal_outcome(
                        evidence_id=unit.id,
                        candidate_id=unit.candidate_id,
                        atom_id=atom.atom_id,
                        outcome="REJECTED",
                        reason_codes=errors,
                        binding_status="atom_role_incomplete_for_source_fact",
                    )
                continue
            facts.append(fact)
    # Multiple values in one local proposition may be legitimate only when
    # their own source-derived category/target bindings distinguish them.  A
    # request is not allowed to choose among otherwise identical values.
    grouped: dict[tuple[str, ...], list[TypedFact]] = defaultdict(list)
    for fact in facts:
        grouped[(
            fact.evidence_span_sha256,
            fact.evidence_subject,
            fact.evidence_predicate,
            fact.evidence_target_entity,
            fact.evidence_period,
            str(getattr(fact.evidence_atom, "facet", "") or fact.facet),
            fact.unit,
            fact.quantity_kind,
            fact.statistical_role,
            fact.aggregation_role,
            fact.column_header,
        )].append(fact)
    ambiguous_ids = {
        item.fact_id
        for group in grouped.values()
        if len({item.value for item in group}) > 1
        for item in group
    }
    if ambiguous_ids:
        construction_reasons.append("source_numeric_binding_ambiguous")
    unlabelled_parallel_spans = {
        fact.evidence_span_sha256
        for fact in facts
        if (
            _UNLABELLED_PARALLEL_NUMERIC_RE.search(
                fact.source_container_claim or fact.source_claim
            )
            and not (
                fact.numeric_binding is not None
                and fact.numeric_binding.binding_method
                in {
                    "local_label_value",
                    "local_period_value",
                    "local_unit_role",
                    "local_scalar_role",
                    "local_response_category",
                    "structural_row_column",
                }
            )
        )
    }
    if unlabelled_parallel_spans:
        construction_reasons.append("source_numeric_binding_ambiguous")
    # Independently proven local bindings must survive an unrelated scalar in
    # the same container that remains incomplete.  The incomplete scalar is
    # already recorded and rejected above; widening that failure into a
    # different label/value or period/value pair would discard source facts
    # whose associations are explicitly established.
    incomplete_multi_value_spans: set[str] = set()
    if incomplete_multi_value_spans:
        construction_reasons.append("source_numeric_binding_ambiguous")
    retained: list[TypedFact] = []
    for fact in facts:
        if (
            fact.fact_id in ambiguous_ids
            or fact.evidence_span_sha256 in unlabelled_parallel_spans
            or fact.evidence_span_sha256 in incomplete_multi_value_spans
        ):
            if recorder is not None and hasattr(
                recorder, "record_source_fact_construction"
            ):
                recorder.record_source_fact_construction(
                    fact,
                    outcome="REJECTED",
                    reason_codes=("source_numeric_binding_ambiguous",),
                )
            if recorder is not None and hasattr(
                recorder, "record_evidence_atom_terminal_outcome"
            ):
                recorder.record_evidence_atom_terminal_outcome(
                    evidence_id=unit.id,
                    candidate_id=unit.candidate_id,
                    atom_id=(fact.evidence_atom.atom_id if fact.evidence_atom else ""),
                    outcome="REJECTED",
                    reason_codes=("source_numeric_binding_ambiguous",),
                    binding_status="atom_role_conflict",
                )
            continue
        registered, emitted = _register_source_fact_occurrence(
            fact, source_fact_occurrence_registry,
        )
        if not isinstance(registered, TypedFact):
            raise RuntimeError("numeric source fact occurrence changed fact family")
        retained.append(registered)
        if recorder is not None and hasattr(
            recorder, "record_source_fact_occurrence_observation"
        ):
            recorder.record_source_fact_occurrence_observation(
                observed_fact=fact,
                authoritative_fact=registered,
                outcome=(
                    "EMITTED" if emitted else
                    "REOBSERVED_SAME_STRUCTURAL_VIEW"
                    if fact.evidence_id == registered.evidence_id else
                    "REOBSERVED_EQUIVALENT_STRUCTURAL_VIEW"
                ),
            )
        if emitted:
            if recorder is not None and hasattr(
                recorder, "record_source_fact_construction"
            ):
                recorder.record_source_fact_construction(
                    registered, outcome="CREATED",
                )
        if recorder is not None and hasattr(
            recorder, "record_evidence_atom_terminal_outcome"
        ):
            recorder.record_evidence_atom_terminal_outcome(
                evidence_id=unit.id,
                candidate_id=unit.candidate_id,
                atom_id=(fact.evidence_atom.atom_id if fact.evidence_atom else ""),
                outcome=(
                    "CONSUMED_BY_SOURCE_FACT"
                    if emitted else "OBSERVED_BY_EXISTING_SOURCE_FACT"
                ),
                binding_status=(
                    "atom_local_binding_consumed"
                    if emitted else "atom_local_binding_reobserved"
                ),
            )
    return tuple(retained), tuple(dict.fromkeys(construction_reasons))


def _base_source_fact_compatibility_errors(
    requirement: CoverageRequirement,
    canonical_requirement: Any,
    fact: TypedFact,
) -> tuple[str, ...]:
    """Existing field-level checks used by canonical admissibility.

    This function deliberately does not decide admission on its own.  C19
    wraps these checks with explicit required/source/provenance decisions so a
    blank source field can never masquerade as an unconstrained request field.
    """
    reasons: list[str] = []
    # Period is an immutable atom-local binding.  It must be evaluated even
    # for legacy callers that have not yet supplied an execution contract;
    # otherwise the first compatible source fact can silently stand in for a
    # different requested endpoint.
    observed_period = str(
        getattr(fact, "evidence_period", "") or getattr(fact, "period", "") or ""
    )
    observed_time_role = str(getattr(fact, "time_role", "") or "")
    period_ok, _period_decision = periods_semantically_compatible(
        requirement.period,
        observed_period,
        required_time_role=requirement.time_role,
        observed_time_role=observed_time_role,
    )
    if requirement.period and observed_period and not period_ok:
        reasons.append("evidence_period_incompatible")
    if not requirement.period and requirement.time_role == "current" and (
        observed_period == "historical"
        or (
            bool(_YEAR_RE.fullmatch(observed_period))
            and observed_time_role != "current"
        )
    ):
        reasons.append("evidence_period_incompatible")
    if not requirement.period and requirement.time_role == "historical" and (
        observed_period == "current" or observed_time_role == "current"
    ):
        reasons.append("evidence_period_incompatible")
    observed_unit = str(
        getattr(fact, "evidence_unit", "") or getattr(fact, "unit", "") or ""
    ).casefold()
    if requirement.expected_unit == "percentage" and observed_unit not in {
        "%", "percent", "percentage", "אחוז",
    }:
        reasons.append("evidence_unit_incompatible")
    if (
        requirement.facet in {"lower_band", "higher_band"}
        and getattr(fact, "facet", getattr(fact, "evidence_facet", ""))
        in {"lower_band", "higher_band"}
        and requirement.facet != getattr(
            fact, "facet", getattr(fact, "evidence_facet", "")
        )
    ):
        reasons.append("evidence_facet_incompatible")
    if requirement.expected_unit == "count" and (
        observed_unit in {"%", "percent", "percentage", "אחוז"}
        or getattr(fact, "scalar_role", "") == "confidence_level"
    ):
        reasons.append("evidence_unit_incompatible")
    required_dimensions = (
        (
            "quantity_kind",
            getattr(canonical_requirement, "quantity_kind", "")
            if canonical_requirement is not None else (
                # Legacy plans encode count/percentage as output labels, not
                # a canonical source-semantic constraint.  Their existing
                # unit and role guards below remain in force, but they must
                # not relabel a source NUMBER as COUNT merely to match an
                # old route-shaped requirement.
                "" if requirement.expected_unit in {"count", "percentage"}
                else requirement.quantity_kind
            ),
            getattr(fact, "quantity_kind", ""),
            "quantity_kind_incompatible",
        ),
        (
            "statistical_role",
            getattr(canonical_requirement, "statistical_role", "")
            if canonical_requirement is not None else requirement.statistical_role,
            getattr(fact, "statistical_role", ""),
            "statistical_role_incompatible",
        ),
        (
            "aggregation_role",
            getattr(canonical_requirement, "aggregation_role", "")
            if canonical_requirement is not None else requirement.aggregation_role,
            getattr(fact, "aggregation_role", ""),
            "aggregation_role_incompatible",
        ),
    )
    # Canonical flows are strict only for dimensions explicitly requested.
    # Legacy plan-only callers preserve their historical safety distinctions
    # without receiving an invented point-estimate requirement.
    for dimension, required, observed, reason in required_dimensions:
        matches, _decision = quantity_dimension_compatibility(
            required, observed, dimension=dimension,
        )
        if not matches:
            reasons.extend((reason, "wrong_numeric_role"))
    legacy_required = normalize_scalar_role(
        getattr(canonical_requirement, "scalar_role", "") or requirement.scalar_role
    )
    legacy_ok, _legacy_decision = scalar_role_compatibility(
        legacy_required, getattr(fact, "scalar_role", ""),
    )
    if (
        canonical_requirement is not None
        and legacy_required != "not_applicable"
        and not legacy_ok
        and any(reason in {
            "quantity_kind_incompatible",
            "statistical_role_incompatible",
            "aggregation_role_incompatible",
        } for reason in reasons)
    ):
        # ``scalar_role`` is retained solely as an audit projection.  It can
        # annotate a genuine independent-dimension mismatch, but must never
        # turn an otherwise compatible fact into a rejection (in particular
        # on a legacy plan-only caller that has no canonical dimensions).
        reasons.append("scalar_role_incompatible")
    # Response-category identity belongs to the source fact.  Apply the
    # request's positive/negative response condition only here, after the
    # fact has been preserved.  This is the same safety distinction used by
    # candidate selection, now expressed against a canonical fact rather than
    # by silently dropping its atom.
    if (
        getattr(fact, "facet", getattr(fact, "evidence_facet", ""))
        in {"response_no", "response_nonresponse"}
        and _POSITIVE_RESPONSE_METRIC_RE.search(requirement.metric)
        and not _NEGATED_RESPONSE_METRIC_RE.search(requirement.metric)
    ):
        reasons.append("requirement_response_category_incompatible")
    # Scope is an independent requirement slot.  It must be checked against
    # source population below, not fused into the lexical metric admission.
    # Otherwise a source fact can lose a correctly bound scoped population
    # simply because its local numeric proposition omits the surrounding
    # scope phrase.
    compatibility_metric = normalize_request_text(requirement.metric)
    scope_phrase = normalize_request_text(
        requirement.population
        or getattr(canonical_requirement, "scope_population", "")
        or ""
    )
    if scope_phrase:
        compatibility_metric = normalize_request_text(
            compatibility_metric.replace(scope_phrase, " ")
        )
    if not compatibility_metric or compatibility_metric.casefold() in {
        "count", "number", "scalar", "value",
    }:
        compatibility_metric = " ".join(
            str(value) for value in getattr(
                canonical_requirement, "objects_or_metrics", ()
            ) if str(value)
        ) or requirement.metric
    if not _metric_compatible(
        compatibility_metric,
        getattr(fact, "source_container_claim", "")
        or getattr(fact, "source_claim", ""),
        expected_unit=requirement.expected_unit,
        require_specific_focus=(
            requirement.kind != "comparison_endpoint"
            and not (
                canonical_requirement is not None
                and canonical_requirement.kind == "lookup"
                and canonical_requirement.value_type == "numeric"
            )
        ),
    ):
        reasons.append("requirement_metric_incompatible")
    # A complete source claim may mention the requested role incidentally
    # after the value has already been bound to another role.  Once a source
    # fact has a concrete multi-token subject, compare that subject directly
    # rather than allowing a later incidental mention in the container claim
    # to satisfy the request.  Generic count targets such as ``workers`` have
    # no such focus terms and remain governed by the ordinary metric check.
    request_focus = _metric_focus_terms(compatibility_metric)
    source_subject_focus = _metric_focus_terms(
        getattr(fact, "evidence_subject", "")
    )
    source_claim = str(
        getattr(fact, "source_container_claim", "")
        or getattr(fact, "source_claim", "") or ""
    )
    source_value = str(getattr(fact, "value", "") or "").casefold()
    source_value_position = source_claim.casefold().find(source_value)
    source_prefix = (
        source_claim[:source_value_position]
        if source_value and source_value_position >= 0 else ""
    )
    if (
        canonical_requirement is not None
        and
        len(request_focus) >= 2
        and len(source_subject_focus) >= 2
        and len(request_focus & source_subject_focus) < 2
        and _metric_focus_spans(
            compatibility_metric,
            getattr(fact, "source_container_claim", "")
            or getattr(fact, "source_claim", ""),
        )
        and not _metric_focus_spans(
            compatibility_metric, getattr(fact, "evidence_subject", "")
        )
    ):
        reasons.append("requirement_subject_incompatible")
    elif (
        canonical_requirement is None
        and len(request_focus) >= 2
        and _metric_focus_spans(
            compatibility_metric,
            getattr(fact, "source_container_claim", "")
            or getattr(fact, "source_claim", ""),
        )
        and not _metric_focus_spans(
            compatibility_metric,
            " ".join(filter(None, (
                getattr(fact, "evidence_subject", ""),
                getattr(fact, "evidence_target_entity", ""),
            ))),
        )
        and not _metric_focus_spans(compatibility_metric, source_prefix)
    ):
        # Legacy plans may still use source-local ownership to reject an
        # incidental mention. Unlike the canonical branch, a source target
        # (for example, companies contained by a registry) or a governing
        # prefix before the scalar is an equally valid owner because the
        # legacy plan has no typed subject slot. A later mention remains
        # insufficient and cannot bind a different scalar retrospectively.
        reasons.append("requirement_subject_incompatible")
    if canonical_requirement is not None:
        reasons.extend(_evidence_fact_compatibility_errors(
            canonical_requirement,
            subject=getattr(fact, "evidence_subject", ""),
            predicate=getattr(fact, "evidence_predicate", ""),
            target_entity=getattr(fact, "evidence_target_entity", ""),
            population=getattr(fact, "evidence_population", ""),
        ))
        reasons.extend(numeric_fact_binding_errors(canonical_requirement, fact))
    if requirement.population and not _population_scope_compatible(
        requirement.population, getattr(fact, "evidence_population", ""),
    ):
        reasons.append("evidence_population_incompatible")
    return tuple(dict.fromkeys(reasons))


_ADMISSIBILITY_DIMENSIONS = (
    "value",
    "quantity_kind",
    "statistical_role",
    "aggregation_role",
    "subject",
    "metric",
    "predicate",
    "period",
    "population",
    "facet",
    "unit",
    "polarity",
    "target_entity",
    "relation_arguments",
    "fact_role",
    "structural_type",
    "proposition_completeness",
    "financial_role",
    "document_scope",
    "source_location",
    "attribute_context",
    "ordered_list_owner",
)


_TEXTUAL_SOURCE_REQUIREMENT_KINDS = frozenset({
    "purpose", "intended_use", "definition", "procedure",
    "cause_or_reason", "explanation", "negative_relation",
    "positive_inclusion_or_count", "comparison_significance", "list",
    "list_attributes",
})


def _requirement_needs_bound_numeric(
    requirement: CoverageRequirement,
    canonical_requirement: Any,
) -> bool:
    """Whether the request actually requires a scalar rather than a relation."""
    canonical_kind = str(getattr(canonical_requirement, "kind", "") or "")
    canonical_value_type = str(
        getattr(canonical_requirement, "value_type", "") or ""
    )
    if (
        canonical_kind in _TEXTUAL_SOURCE_REQUIREMENT_KINDS
        or canonical_value_type in {"text", "text_proposition", "list"}
    ):
        return False
    quantity_kind = normalize_quantity_kind(
        getattr(canonical_requirement, "quantity_kind", "") or requirement.quantity_kind
    )
    statistical_role = normalize_statistical_role(
        getattr(canonical_requirement, "statistical_role", "") or requirement.statistical_role
    )
    aggregation_role = normalize_aggregation_role(
        getattr(canonical_requirement, "aggregation_role", "") or requirement.aggregation_role
    )
    return bool(
        requirement.requires_value
        or requirement.expected_unit in {"count", "percentage"}
        or quantity_kind != QuantityKind.UNSPECIFIED.value
        or statistical_role != StatisticalRole.UNSPECIFIED.value
        or aggregation_role not in {
            AggregationRole.NONE.value, AggregationRole.UNSPECIFIED.value,
        }
        or (
            str(getattr(canonical_requirement, "value_type", "") or "") == "numeric"
            and canonical_kind in {
                "lookup", "count", "percentage", "comparison_endpoint",
                "filtered_row_value", "filtered_row_period", "winner",
                "winner_attribute",
            }
        )
    )


def _canonical_requirement_value(
    requirement: CoverageRequirement,
    canonical_requirement: Any,
    dimension: str,
) -> str:
    """Return a request-owned dimension without inventing source semantics."""
    canonical = canonical_requirement
    if canonical is not None and hasattr(canonical, "dimension_constraints"):
        constraint = requirement_dimension_constraint(canonical, dimension)
        return (
            constraint.value
            if constraint.state is RequirementConstraintState.REQUIRED
            else ""
        )
    if dimension == "value":
        return "required" if _requirement_needs_bound_numeric(
            requirement, canonical,
        ) else ""
    if dimension == "quantity_kind":
        if not _requirement_needs_bound_numeric(requirement, canonical):
            return ""
        if canonical is None and requirement.expected_unit in {
            "count", "percentage",
        }:
            # A legacy requested output label has no independently compiled
            # quantity constraint.  The source fact retains its own typed
            # quantity semantics; legacy-safe unit checks stay in the base
            # adapter rather than becoming a second semantic authority.
            return ""
        value = str(getattr(canonical, "quantity_kind", "") or requirement.quantity_kind)
        value = normalize_quantity_kind(value)
        if value == QuantityKind.UNSPECIFIED.value:
            if requirement.expected_unit == "count":
                return QuantityKind.COUNT.value
            if requirement.expected_unit == "percentage":
                return QuantityKind.PERCENTAGE.value
        return "" if value == QuantityKind.UNSPECIFIED.value else value
    if dimension == "statistical_role":
        if not _requirement_needs_bound_numeric(requirement, canonical):
            return ""
        value = normalize_statistical_role(
            getattr(canonical, "statistical_role", "") or requirement.statistical_role
        )
        return "" if value == StatisticalRole.UNSPECIFIED.value else value
    if dimension == "aggregation_role":
        if not _requirement_needs_bound_numeric(requirement, canonical):
            return ""
        value = normalize_aggregation_role(
            getattr(canonical, "aggregation_role", "") or requirement.aggregation_role
        )
        return "" if value in {
            AggregationRole.NONE.value, AggregationRole.UNSPECIFIED.value,
        } else value
    if dimension == "subject":
        return str(getattr(canonical, "subject", "") or "")
    if dimension == "metric":
        values = tuple(getattr(canonical, "objects_or_metrics", ()) or ())
        return " ".join(str(item) for item in values if str(item).strip()) or str(
            requirement.metric or ""
        )
    if dimension == "predicate":
        return str(getattr(canonical, "predicate", "") or "")
    if dimension == "period":
        values = tuple(getattr(canonical, "periods", ()) or ())
        if (
            str(getattr(canonical, "value_type", "") or "")
            in {"text", "text_proposition"}
            and len(values) > 1
        ):
            # A relational textual proposition can prove a two-period claim
            # only by containing both endpoints; it is not a scalar bound to
            # either endpoint alone.
            return "|".join(str(value) for value in values if str(value))
        return str(values[0] if values else requirement.period or "") or str(
            requirement.time_role or ""
        )
    if dimension == "population":
        values = tuple(getattr(canonical, "populations", ()) or ())
        return str(
            getattr(canonical, "scope_population", "") or (
                values[0] if values else requirement.population
            ) or ""
        )
    if dimension == "facet":
        # ``CoverageRequirement.facet`` has older planning labels such as
        # ``metric``.  They describe how the request was decomposed, not a
        # source-side category, and therefore cannot become a new admission
        # constraint.  A concrete canonical facet remains authoritative;
        # retain only the older source-level facets for legacy callers.
        canonical_facet = str(getattr(canonical, "facet", "") or "").strip()
        if canonical_facet:
            return canonical_facet
        legacy_facet = str(requirement.facet or "").strip()
        return "" if legacy_facet.casefold() in {
            "", "metric", "value", "scalar", "numeric", "count", "percentage",
        } else legacy_facet
    if dimension == "unit":
        values = tuple(getattr(canonical, "units", ()) or ())
        unit = str(values[0] if values else requirement.expected_unit or "")
        if canonical is None and unit.casefold() in {"count", "percentage"}:
            return ""
        return unit if (
            _requirement_needs_bound_numeric(requirement, canonical)
            and unit.casefold() not in {"", "numeric", "value"}
        ) else ""
    if dimension == "polarity":
        value = str(getattr(canonical, "polarity", "") or "neutral")
        return "" if value == "neutral" else value
    if dimension == "target_entity":
        return str(
            getattr(canonical, "counted_entity", "")
            or getattr(canonical, "target_entity_type", "")
            or ""
        )
    if dimension == "relation_arguments":
        return str(
            getattr(canonical, "relation_target", "")
            or (
                getattr(canonical, "predicate", "")
                if getattr(canonical, "predicate", "") in RELATION_ARGUMENT_SCHEMAS
                else ""
            )
            or ""
        )
    if dimension == "fact_role":
        return "measured_outcome" if _requirement_needs_bound_numeric(
            requirement, canonical,
        ) else ""
    if dimension in {"structural_type", "proposition_completeness"}:
        return "required"
    return ""


def _fact_relation_argument_value(
    fact: TypedFact | TextPropositionFact,
    canonical_requirement: Any = None,
) -> str:
    """Return a source-local relation binding for one admissibility check.

    Typed relation predicates retain their role-labelled arguments.  A request
    may also explicitly name a reporting relation while its numeric source
    fact is represented as ``has_count``/``has_percentage``.  In that case the
    bounded source proposition is the existing native authority proving that
    the response complement belongs to this actor and scalar; a surrounding
    chunk must never satisfy that relation on its behalf.
    """
    arguments = "|".join(
        f"{getattr(argument, 'role', '')}:{getattr(argument, 'value', '')}"
        for argument in (getattr(fact, "relation_arguments", ()) or ())
        if getattr(argument, "value", "")
    )
    required_predicate = ""
    if canonical_requirement is not None:
        required_predicate = str(
            requirement_dimension_constraint(
                canonical_requirement, "predicate",
            ).value or ""
        )
    if arguments or required_predicate in RELATION_ARGUMENT_SCHEMAS:
        if required_predicate in DOCUMENT_ROLE_PREDICATES:
            role = RELATION_ARGUMENT_SCHEMAS[required_predicate][0]
            return next((argument.value for argument in getattr(fact, "relation_arguments", ())
                         if argument.role == role), "")
        if required_predicate == "purpose_of":
            # This dimension constrains the requested target, while the
            # complete purpose output remains an independently required role.
            return next((argument.value for argument in getattr(fact, "relation_arguments", ())
                         if argument.role == "purpose_target"), "")
        return arguments
    return str(getattr(fact, "source_claim", "") or "")


def _fact_dimension_value(
    fact: TypedFact | TextPropositionFact,
    dimension: str,
    canonical_requirement: Any = None,
) -> str:
    """Return the source-derived value for one admissibility dimension."""
    if dimension == "value":
        return str(getattr(fact, "value", "") or "")
    if dimension in {"quantity_kind", "statistical_role", "aggregation_role"}:
        return str(getattr(fact, dimension, "") or "")
    if dimension == "financial_role":
        from financial_semantics import monetary_role, normalize_currency
        return monetary_role(str(getattr(fact, "source_claim", "") or "")) if normalize_currency(getattr(fact, "unit", "")) else ""
    if dimension == "document_scope":
        from financial_semantics import source_document_label
        return source_document_label(fact)
    if dimension == "source_location":
        from document_query import fact_source_locations
        location = fact_source_locations(fact)
        return json.dumps(location, sort_keys=True) if location else ""
    if dimension == "attribute_context":
        context = getattr(fact, "evidence_context", None)
        return " ".join((str(getattr(context, "governing_context", "") or ""), str(getattr(fact, "source_claim", "") or ""))).strip()
    if dimension == "ordered_list_owner":
        from ordered_list_evidence import source_ordered_list_owner
        context = getattr(fact, "evidence_context", None)
        return source_ordered_list_owner(
            getattr(context, "structural_evidence_unit", None),
            str(getattr(context, "source_candidate_text", "") or ""),
        )
    if dimension == "subject":
        return str(getattr(fact, "evidence_subject", "") or getattr(fact, "subject", "") or "")
    if dimension == "metric":
        return str(
            getattr(fact, "metric", "")
            or getattr(fact, "evidence_target_entity", "")
            or getattr(fact, "evidence_subject", "")
            or ""
        )
    if dimension == "predicate":
        return str(getattr(fact, "evidence_predicate", "") or getattr(fact, "predicate", "") or "")
    if dimension == "period":
        observed = str(
            getattr(fact, "evidence_period", "")
            or getattr(fact, "period", "") or ""
        )
        if not observed and isinstance(fact, TextPropositionFact):
            source_periods = tuple(dict.fromkeys(_YEAR_RE.findall(
                str(getattr(fact, "source_claim", "") or "")
            )))
            if source_periods:
                return "|".join(source_periods)
        return observed
    if dimension == "population":
        return str(getattr(fact, "evidence_population", "") or getattr(fact, "population", "") or "")
    if dimension == "facet":
        return "|".join(source_fact_facets(fact))
    if dimension == "unit":
        return str(getattr(fact, "evidence_unit", "") or getattr(fact, "unit", "") or "")
    if dimension == "polarity":
        return str(getattr(fact, "evidence_polarity", "") or "neutral")
    if dimension == "target_entity":
        return str(getattr(fact, "evidence_target_entity", "") or "")
    if dimension == "relation_arguments":
        return _fact_relation_argument_value(fact, canonical_requirement)
    if dimension == "fact_role":
        predicate = _fact_dimension_value(fact, "predicate")
        source = str(getattr(fact, "source_claim", "") or "")
        if predicate in {"excluded_from", "exception_to"} or str(
            getattr(fact, "evidence_polarity", "neutral")
        ) == "negative":
            return "exclusion_condition"
        if str(getattr(fact, "statistical_role", "")) == (
            StatisticalRole.FILTER_THRESHOLD.value
        ):
            return "filter_condition"
        if re.search(
            r"\b(?:filter|eligib(?:le|ility)|threshold|cut[ -]?off)\b|"
            r"(?:סף|תנאי|סינון)", source, re.IGNORECASE,
        ):
            return "filter_condition"
        return "measured_outcome"
    if dimension == "structural_type":
        return str(getattr(fact, "structural_type", "") or "")
    if dimension == "proposition_completeness":
        if isinstance(fact, (TypedFact, TextPropositionFact)):
            return "complete" if not _source_fact_semantic_completeness_errors(fact) else ""
        # Operator facts are already source-derived semantic records, but
        # older operator adapters do not subclass either direct SOURCE_FACT
        # class.  Record their missing atom/provenance through the ordinary
        # proof path rather than raising or auto-admitting them.
        return "complete" if all(
            str(getattr(fact, field, "") or "").strip()
            for field in ("source_claim", "value", "evidence_subject", "evidence_predicate")
        ) else ""
    return ""


def _fact_dimension_provenance(
    fact: TypedFact | TextPropositionFact,
    dimension: str,
) -> Mapping[str, Any]:
    """Return compact source-only provenance for a compatibility proof row."""
    context = getattr(fact, "evidence_context", None)
    field_names = {
        "subject": "evidence_subject",
        "predicate": "evidence_predicate",
        "target_entity": "evidence_target_entity",
        "period": "evidence_period",
        "population": "evidence_population",
        "unit": "evidence_unit",
        "polarity": "evidence_polarity",
        "document_scope": "table_chart_title",
        "attribute_context": "governing_context",
    }
    provenance = (
        context.provenance_for(field_names[dimension])
        if context is not None and dimension in field_names else None
    )
    if dimension == "ordered_list_owner" and context is not None:
        from structural_evidence import source_ordered_list_owner_binding
        structural = context.structural_evidence_unit
        binding = source_ordered_list_owner_binding(
            structural, context.source_candidate_text,
        ) if structural is not None else None
        if binding is not None:
            owner, span = binding
            provenance = FieldProvenance(
                field="ordered_list_owner", value=owner,
                provenance_type="explicit_source_organization_subject",
                source_container="source_candidate", source_span=(span.start, span.end),
                source_sha256=span.sha256, confidence=1.0,
                revisability="fixed_source_structure", source_quote=owner,
                derivation_type="literal_organization_subject",
                structural_unit_id=structural.unit_id,
                structural_group_id=structural.structural_group_id or structural.unit_id,
                structural_role="organization_owner",
            )
    numeric_binding = getattr(fact, "numeric_binding", None)
    binding_payload = (
        numeric_binding.as_dict()
        if numeric_binding is not None and dimension in {
            "value", "quantity_kind", "statistical_role", "aggregation_role",
            "fact_role",
        } else {}
    )
    return {
        "source_fact_id": str(getattr(fact, "source_fact_id", "") or getattr(fact, "fact_id", "")),
        "source_id": str(getattr(context, "source_id", "") or ""),
        "financial_role_proof": ({
            "source_container": "local_proposition",
            "source_quote": str(getattr(fact, "source_claim", "") or ""),
            "source_span": list(getattr(fact, "evidence_span", ()) or ()),
        } if dimension == "financial_role" else {}),
        "evidence_id": str(getattr(fact, "evidence_id", "") or ""),
        "atom_id": str(getattr(getattr(fact, "evidence_atom", None), "atom_id", "") or ""),
        "source_span_sha256": str(
            getattr(fact, "evidence_span_sha256", "")
            or getattr(fact, "proposition_sha256", "")
            or ""
        ),
        "field_provenance": provenance.as_dict() if provenance is not None else {},
        "numeric_binding": binding_payload,
    }


def _admissibility_reason_dimension(reason: str) -> str:
    """Map a stable compatibility reason to its first material dimension."""
    mappings = (
        ("ordered_list_owner", ("ordered_list_owner",)),
        ("document_scope", ("ordered_list_document_scope",)),
        ("financial_role", ("financial_row_role",)),
        ("document_scope", ("financial_document_scope",)),
        ("unit", ("financial_currency",)),
        ("quantity_kind", ("quantity_kind",)),
        ("statistical_role", ("statistical_role",)),
        ("aggregation_role", ("aggregation_role",)),
        ("subject", ("subject", "evidence_subject", "requirement_subject")),
        ("metric", ("metric", "requirement_metric")),
        ("predicate", ("predicate", "evidence_predicate", "proposition_relation")),
        ("period", ("period", "evidence_period")),
        ("population", ("population", "evidence_population", "scope")),
        ("facet", ("facet", "evidence_facet")),
        ("unit", ("unit", "evidence_unit")),
        ("polarity", ("polarity", "evidence_polarity", "response_category")),
        ("target_entity", ("target_entity", "evidence_target_entity")),
        ("relation_arguments", ("relation_argument", "relational_argument")),
        ("fact_role", ("fact_role", "filter_condition", "exclusion_condition")),
        ("structural_type", ("structural_", "row_header", "column_header", "cross_")),
        ("proposition_completeness", ("source_", "evidence_span")),
    )
    return next((dimension for dimension, prefixes in mappings if any(
        reason == prefix or reason.startswith(prefix) for prefix in prefixes
    )), "")


def _period_proof_compatible(required: str, observed: str) -> bool:
    """Require every request period to occur in the source-derived proof."""
    required_periods = tuple(
        value.strip() for value in str(required or "").split("|") if value.strip()
    )
    observed_periods = tuple(
        value.strip() for value in str(observed or "").split("|") if value.strip()
    )
    if not required_periods or not observed_periods:
        return not required_periods
    return all(any(
        periods_semantically_compatible(requested, source)[0]
        for source in observed_periods
    ) for requested in required_periods)


def _facet_proof_compatible(required: str, observed: str) -> bool:
    """Compare an explicitly requested source facet without route aliases."""
    normalized_required = normalize_request_text(required).replace(" ", "_")
    normalized_observed = normalize_request_text(observed).replace(" ", "_")
    observed_facets = normalized_observed.split("|")
    return bool(normalized_required and source_facet_compatible(normalized_required, observed_facets))


def _is_typed_numeric_requirement(
    requirement: CoverageRequirement,
    canonical_requirement: Any,
) -> bool:
    """Whether a text proposition needs a bound scalar SOURCE_FACT."""
    return _requirement_needs_bound_numeric(requirement, canonical_requirement)


def _source_predicate_compatible(required: str, observed: str) -> bool:
    """Compare source predicates without reusing route/output aliases."""
    compatible = {
        "has_count": {"has_count", "included_in", "employs_workers", "recruits_employees"},
        "has_percentage": {"has_percentage", "has_count"},
        "compared_with": {"has_count", "has_percentage", "changed_between"},
        "changed_between": {"has_count", "has_percentage", "changed_between"},
        "maximum_of": {"has_count", "has_percentage"},
        "minimum_of": {"has_count", "has_percentage"},
        "stated_as": {"stated_as", "has_count", "has_percentage", "employs_workers", "recruits_employees", "raises_capital"},
        "included_in": {"included_in", "has_count"},
        "excluded_from": {"excluded_from", "exception_to"},
        "exception_to": {"exception_to", "excluded_from"},
        "purpose_of": {"purpose_of"},
        "defined_as": {"defined_as", "stated_as"},
        "caused_by": {"caused_by", "explained_as"},
        "explained_as": {"explained_as", "caused_by"},
        "method_of": {"method_of", "performed_by"},
    }
    return observed in compatible.get(required, {required})


def _source_entity_compatible(required: str, observed: str) -> bool:
    """Compare source-backed entities without promoting topical similarity.

    Source propositions often inflect the event in the question (``slow`` /
    ``slowed``).  Accept only a complete English inflection match in addition
    to the existing entity and term comparisons; arbitrary prefixes remain
    incompatible.
    """
    if (
        canonical_entity_equivalent(required, observed)
        or _semantic_term_overlap(required, observed) >= 0.60
        or _term_overlap(required, observed) >= 0.60
    ):
        return True
    required_terms = set(_normalized_terms(required))
    observed_terms = set(_normalized_terms(observed))
    if not required_terms or not observed_terms:
        return False

    def inflection_stem(term: str) -> str:
        if not term.isascii() or len(term) < 4:
            return term
        if term.endswith("ies") and len(term) > 4:
            return f"{term[:-3]}y"
        for suffix in ("ing", "ed", "es", "s"):
            if term.endswith(suffix) and len(term) > len(suffix) + 2:
                return term[:-len(suffix)]
        return term

    return all(
        any(
            inflection_stem(required_term) == inflection_stem(observed_term)
            for observed_term in observed_terms
        )
        for required_term in required_terms
    )


def canonical_fact_admissibility(
    requirement: CoverageRequirement,
    canonical_requirement: Any,
    fact: TypedFact | TextPropositionFact,
) -> FactAdmissibilityResult:
    """Make the only admission decision for a source fact and requirement.

    Source construction is immutable and request independent.  This function
    solely compares the completed source fact with the canonical requirement;
    neither its proof values nor its failure codes can introduce a request
    value into the source fact.
    """
    from semantic_contracts import scoped_document_target_compatible

    compiler_owned_constraints = bool(
        canonical_requirement is not None
        and hasattr(canonical_requirement, "dimension_constraints")
    )
    if compiler_owned_constraints:
        # C20 constraints are the sole source-admission authority.  The
        # older helpers still serve legacy callers, but must not re-harden a
        # route/output field after compilation has marked it NOT_REQUIRED or
        # SCOPE_ONLY.
        reasons: list[str] = []
        if not isinstance(fact, TypedFact) and _is_typed_numeric_requirement(
            requirement, canonical_requirement,
        ):
            reasons.append("typed_numeric_requirement_requires_bound_scalar")
    elif isinstance(fact, TypedFact) or (
        not isinstance(fact, TextPropositionFact)
        and hasattr(fact, "value")
    ):
        reasons = list(_base_source_fact_compatibility_errors(
            requirement, canonical_requirement, fact,
        ))
    else:
        reasons = list(_base_source_text_fact_compatibility_errors(
            canonical_requirement, fact,
        ))
        if _is_typed_numeric_requirement(requirement, canonical_requirement):
            reasons.append("typed_numeric_requirement_requires_bound_scalar")

    from financial_semantics import financial_fact_errors
    reasons.extend(financial_fact_errors(
        _canonical_requirement_value(requirement, canonical_requirement, "financial_role"),
        _canonical_requirement_value(requirement, canonical_requirement, "document_scope"),
        _canonical_requirement_value(requirement, canonical_requirement, "unit"),
        fact,
    ))
    required = {
        dimension: _canonical_requirement_value(
            requirement, canonical_requirement, dimension,
        )
        for dimension in _ADMISSIBILITY_DIMENSIONS
    }
    observed = {
        dimension: _fact_dimension_value(
            fact, dimension, canonical_requirement,
        )
        for dimension in _ADMISSIBILITY_DIMENSIONS
    }
    # Keep source-compatibility reasons immutable while producing the proof.
    # The proof itself adds ``source_<dimension>_unproven`` diagnostics.  If a
    # later dimension were to inspect that newly generated diagnostic, the
    # generic ``source_`` prefix could turn one unknown field into a false
    # proposition-completeness mismatch.
    base_reasons = tuple(reasons)
    proof: dict[str, Mapping[str, Any]] = {}
    decisions: dict[str, str] = {}

    for dimension in _ADMISSIBILITY_DIMENSIONS:
        required_value = required[dimension]
        source_value = observed[dimension]
        dimension_reasons = [
            reason for reason in base_reasons
            if _admissibility_reason_dimension(reason) == dimension
        ]
        if not required_value:
            decision = "NOT_REQUIRED"
        elif not source_value or source_value in {
            QuantityKind.UNSPECIFIED.value,
            StatisticalRole.UNSPECIFIED.value,
            AggregationRole.UNSPECIFIED.value,
        }:
            reason = f"source_{dimension}_unproven"
            if reason not in reasons:
                reasons.append(reason)
            decision = (
                "SOURCE_UNSPECIFIED"
                if dimension in {
                    "quantity_kind", "statistical_role", "aggregation_role",
                } else "NOT_PROVEN"
            )
        elif dimension == "financial_role" and source_value != required_value:
            reasons.append("financial_row_role_incompatible")
            decision = "MISMATCH"
        elif dimension == "source_location":
            from document_query import location_fact_errors
            location_errors = location_fact_errors(required_value, fact)
            decision = "MISMATCH" if location_errors else "MATCH"
            reasons.extend(location_errors)
        elif dimension == "attribute_context":
            from document_query import attribute_context_errors
            context_errors = attribute_context_errors(required_value, fact)
            decision = "MISMATCH" if context_errors else "MATCH"
            reasons.extend(context_errors)
        elif dimension == "ordered_list_owner":
            from ordered_list_evidence import ordered_list_owner_errors
            owner_errors = ordered_list_owner_errors(canonical_requirement, fact)
            decision = "MISMATCH" if owner_errors else "MATCH"
            reasons.extend(owner_errors)
        elif dimension == "document_scope":
            from document_query import document_fact_errors
            try:
                scope_schema = json.loads(required_value)
            except (TypeError, ValueError):
                scope_schema = None
            if isinstance(scope_schema, dict) and scope_schema.get("schema") == "project-x-named-document-v1":
                scoped_errors = document_fact_errors(required_value, fact)
                decision = "MISMATCH" if scoped_errors else "MATCH"
                reasons.extend(scoped_errors)
            else:
                from financial_semantics import document_scope_matches
                decision = "MATCH" if document_scope_matches(required_value, source_value) else "MISMATCH"
                if decision != "MATCH":
                    reasons.append("financial_document_scope_incompatible")
        elif dimension == "subject" and required.get("predicate") == "has_attribute" and canonical_requirement is not None:
            from semantic_contracts import technical_attribute_label_errors
            label_errors = technical_attribute_label_errors(canonical_requirement, fact)
            decision = "MISMATCH" if label_errors else "MATCH"
            reasons.extend(label_errors)
        elif dimension == "fact_role" and source_value != required_value:
            reasons.append("source_fact_role_incompatible")
            decision = "MISMATCH"
        elif dimension in {
            "quantity_kind", "statistical_role", "aggregation_role",
        } and not quantity_dimension_compatibility(
            required_value, source_value, dimension=dimension,
        )[0]:
            reasons.extend((
                f"requirement_{dimension}_incompatible",
                f"{dimension}_incompatible",
            ))
            legacy_required = normalize_scalar_role(
                getattr(canonical_requirement, "scalar_role", "")
                or requirement.scalar_role
            )
            if (
                legacy_required != "not_applicable"
                and not scalar_role_compatibility(
                    legacy_required,
                    getattr(fact, "scalar_role", ""),
                )[0]
            ):
                reasons.append("scalar_role_incompatible")
            decision = "MISMATCH"
        elif dimension == "period" and not _period_proof_compatible(
            required_value, source_value,
        ):
            reasons.append("evidence_period_incompatible")
            decision = "MISMATCH"
        elif dimension == "facet" and not _facet_proof_compatible(
            required_value, source_value,
        ):
            reasons.append("evidence_facet_incompatible")
            decision = "MISMATCH"
        elif dimension in {"structural_type", "proposition_completeness"}:
            decision = "MATCH"
        elif dimension == "unit":
            normalized_required_unit = normalize_request_text(
                required_value,
            ).casefold()
            normalized_source_units = {
                normalize_request_text(value).casefold()
                for value in source_value.split("|") if value
            }
            # ``count`` and ``percentage`` are abstract requested output
            # types, not literal source-unit tokens.  Their independently
            # checked quantity kind owns that requirement; compare a concrete
            # requested unit only to source text.  This preserves a source
            # unit such as ``organizations`` without treating it as unequal
            # to the requested count type.
            unit_matches = (
                normalized_required_unit == "count"
                and observed.get("quantity_kind") == QuantityKind.COUNT.value
            ) or (
                normalized_required_unit == "percentage"
                and observed.get("quantity_kind")
                == QuantityKind.PERCENTAGE.value
            ) or normalized_required_unit in normalized_source_units
            if not unit_matches:
                reasons.append("evidence_unit_incompatible")
                decision = "MISMATCH"
            else:
                decision = "MATCH"
        elif dimension == "polarity" and required_value != source_value:
            reasons.append("evidence_polarity_incompatible")
            decision = "MISMATCH"
        elif dimension == "population":
            population_matches = _population_scope_compatible(required_value, source_value)
            if re.match(r"(?:at|in|within)\s+(?:the\s+)?[A-Z]|(?:בחברת|בחברה|בארגון|אצל)\s+", required_value):
                # A source-defined local alias may prove a named request, but
                # literal generic overlap cannot supply the requested owner.
                population_matches = source_population_matches_scope(required_value, fact)
            if not population_matches:
                reasons.append("evidence_population_incompatible")
                decision = "MISMATCH"
            else:
                decision = "MATCH"
        elif dimension == "relation_arguments" and required.get("predicate") in DOCUMENT_ROLE_PREDICATES and not document_role_target_compatible(required_value, source_value):
            reasons.append("document_role_target_incompatible")
            decision = "MISMATCH"
        elif dimension == "relation_arguments" and required.get("predicate") in DOCUMENT_ROLE_PREDICATES:
            decision = "MATCH"
        elif dimension in {"subject", "target_entity", "relation_arguments"} and not any(
            _source_entity_compatible(required_value, candidate)
            for candidate in (
                source_value,
                str(getattr(fact, "evidence_subject", "") or "")
                if dimension == "target_entity" else "",
                # A governing section is a source-derived structural context,
                # not a request-side repair.  Combining it with the bounded
                # proposition preserves an entity named by the heading while
                # still requiring its metric/value in that same source fact.
                " ".join(filter(None, (
                    source_value,
                    str(getattr(getattr(fact, "evidence_context", None), "section_title", "") or ""),
                )))
                if dimension == "subject"
                and getattr(getattr(fact, "evidence_context", None), "provenance_for", lambda _field: None)("section_title") is not None
                else "",
            ) if candidate
        ) and not (
            dimension in {"target_entity", "relation_arguments"}
            and scoped_document_target_compatible(
                canonical_requirement, fact, required_value, source_value,
            )
        ):
            reasons.append(f"evidence_{dimension}_incompatible")
            decision = "MISMATCH"
        elif dimension == "metric" and not _metric_compatible(
            required_value,
            # A fact's metric must be present in that fact's bounded
            # proposition.  The larger container is retrieval context only;
            # using it here could combine a request metric from one
            # proposition with a scalar from another.
            str(getattr(fact, "source_claim", "") or source_value),
            expected_unit=(
                "count" if required.get("quantity_kind") == QuantityKind.COUNT.value
                else "percentage" if required.get("quantity_kind") == QuantityKind.PERCENTAGE.value
                else ""
            ),
            require_specific_focus=False,
            owned_subject=(
                required.get("subject", "")
                if decisions.get("subject") == "MATCH" else ""
            ),
            owned_facet=(
                required.get("facet", "")
                if _facet_proof_compatible(required.get("facet", ""), observed.get("facet", ""))
                else ""
            ),
            source_metric=" ".join(filter(None, (
                source_value,
                str(getattr(fact, "evidence_subject", "") or ""),
            ))),
        ):
            reasons.extend((
                "evidence_metric_incompatible",
                "requirement_metric_incompatible",
            ))
            decision = "MISMATCH"
        elif dimension == "predicate" and not _source_predicate_compatible(
            required_value, source_value,
        ):
            reasons.append("evidence_predicate_incompatible")
            decision = "MISMATCH"
        elif dimension_reasons:
            decision = "NOT_PROVEN" if any(
                token in reason for reason in dimension_reasons
                for token in ("unresolved", "unbound", "missing", "provenance")
            ) else "MISMATCH"
        else:
            decision = "MATCH"
        decisions[dimension] = decision
        proof[dimension] = {
            "requirement_value": required_value,
            "source_value": source_value,
            "decision": decision,
            "provenance": _fact_dimension_provenance(fact, dimension),
        }
        if compiler_owned_constraints:
            constraint = requirement_dimension_constraint(
                canonical_requirement, dimension,
            )
            proof[dimension] = {
                **proof[dimension],
                "requirement_state": constraint.state.value,
                "requirement_origin": constraint.origin.value,
                "request_span": constraint.request_span,
                "requirement_rule": constraint.rule,
                "requirement_justification": constraint.justification,
                "semantic_owner": constraint.semantic_owner,
                "hard_constraint_owner": constraint.hard_constraint_owner,
                "enforcement_path": constraint.enforcement_path,
                "enforcement_equivalence_status": (
                    constraint.enforcement_equivalence_status.value
                ),
            }

    # A compatibility pass must be mechanically impossible while a required
    # dimension is unknown, unproven, or mismatched.  Base compatibility
    # errors remain authoritative too, so this adds proof rather than a
    # competing selector.
    for dimension, decision in decisions.items():
        if required[dimension] and decision != "MATCH":
            reasons.append(f"admissibility_{dimension}_{decision.casefold()}")
    reasons = list(dict.fromkeys(reasons))
    first = next((
        dimension for dimension in _ADMISSIBILITY_DIMENSIONS
        if required[dimension] and decisions[dimension] != "MATCH"
    ), "")
    if not first:
        first = next((
            _admissibility_reason_dimension(reason)
            for reason in reasons if _admissibility_reason_dimension(reason)
        ), "")
    decisions["scalar_role"] = "NOT_REQUIRED"
    decisions["final"] = "COMPATIBLE" if not reasons else "INCOMPATIBLE"
    _c45_application_value = FactAdmissibilityResult(
        reason_codes=tuple(reasons),
        dimension_decisions=decisions,
        dimension_proof=proof,
        first_material_mismatch=first,
    )
    _c45_observe('C44E2E2_FAMILY_07_4e0f40837437')
    return _c45_application_value


def coverage_requirement_from_canonical_requirement(
    canonical_requirement: Any,
) -> CoverageRequirement:
    """Adapt one frozen request requirement for the shared admission gate.

    Operators use this adapter only to audit their source facts through the
    same admissibility authority as direct coverage.  It derives no source
    fields and is intentionally not an alternative operator selector.
    """
    units = tuple(getattr(canonical_requirement, "units", ()) or ())
    quantity_kind = normalize_quantity_kind(
        getattr(canonical_requirement, "quantity_kind", "")
    )
    expected_unit = str(units[0] if units else "")
    if quantity_kind == QuantityKind.COUNT.value:
        expected_unit = "count"
    elif quantity_kind == QuantityKind.PERCENTAGE.value:
        expected_unit = "percentage"
    return CoverageRequirement(
        id=str(getattr(canonical_requirement, "id", "") or ""),
        subquestion_id=str(getattr(canonical_requirement, "clause_id", "") or ""),
        kind=str(getattr(canonical_requirement, "kind", "") or ""),
        label=str(getattr(canonical_requirement, "subject", "") or ""),
        status=CoverageStatus.MISSING,
        evidence_ids=(),
        metric=" ".join(
            str(item) for item in (
                getattr(canonical_requirement, "objects_or_metrics", ()) or ()
            ) if str(item).strip()
        ) or str(getattr(canonical_requirement, "subject", "") or ""),
        facet=str(getattr(canonical_requirement, "facet", "") or ""),
        period=str(next(iter(
            getattr(canonical_requirement, "periods", ()) or ()
        ), "") or ""),
        expected_unit=expected_unit,
        requires_value=(
            str(getattr(canonical_requirement, "value_type", "") or "")
            == "numeric"
        ),
        population=str(getattr(canonical_requirement, "scope_population", "") or ""),
        quantity_kind=quantity_kind,
        statistical_role=normalize_statistical_role(
            getattr(canonical_requirement, "statistical_role", "")
        ),
        aggregation_role=normalize_aggregation_role(
            getattr(canonical_requirement, "aggregation_role", "")
        ),
        satisfaction_mode=str(
            getattr(getattr(canonical_requirement, "satisfaction_mode", ""), "value", "")
            or getattr(canonical_requirement, "satisfaction_mode", "")
            or RequirementSatisfactionMode.SOURCE_FACT.value
        ),
    )


def _source_fact_compatibility_errors(
    requirement: CoverageRequirement,
    canonical_requirement: Any,
    fact: TypedFact,
) -> tuple[str, ...]:
    """Compatibility adapter retained for older callers; admission is canonical."""
    return canonical_fact_admissibility(
        requirement, canonical_requirement, fact,
    ).reason_codes


def _source_fact_compatibility_evaluation(
    requirement: CoverageRequirement,
    canonical_requirement: Any,
    fact: TypedFact,
) -> tuple[tuple[str, ...], dict[str, str], str]:
    """Compatibility adapter backed by the sole canonical admission result."""
    result = canonical_fact_admissibility(
        requirement, canonical_requirement, fact,
    )
    return (
        result.reason_codes,
        dict(result.dimension_decisions),
        result.first_material_mismatch,
    )


def _source_fact_requirement_quality(
    requirement: CoverageRequirement,
    fact: TypedFact,
) -> int:
    """Rank compatible immutable facts without reparsing a candidate.

    This consumes only slots and the atom-local proposition already preserved
    on ``fact``.  It is intentionally unable to create, modify, or replace a
    source binding; compatibility remains the authority for admission.
    """
    quality = int(fact.association_quality)
    request_terms = _metric_focus_terms(requirement.metric)
    source_terms = _metric_focus_terms(" ".join((
        fact.evidence_subject,
        fact.evidence_target_entity,
        fact.facet,
    )))
    quality += 4 * len(request_terms & source_terms)
    if _metric_focus_spans(requirement.metric, fact.source_claim):
        quality += 10
    requested_predicate = _evidence_predicate(requirement.metric)
    if requested_predicate and requested_predicate == fact.evidence_predicate:
        # Legacy plans do not carry canonical requirement predicates, but an
        # explicitly named action in the request can still rank already
        # compatible SOURCE_FACTs.  This does not alter construction or make
        # an incompatible action admissible.
        quality += 12
    if requirement.period and requirement.period == fact.evidence_period:
        quality += 6
    if requirement.time_role and requirement.time_role == fact.time_role:
        quality += 4
    if not requirement.period and not requirement.time_role and _YEAR_RE.search(
        fact.source_claim
    ):
        quality -= 8
    if (
        requirement.expected_unit == "count"
        and _TOTAL_METRIC_RE.search(requirement.metric)
        and _TOTAL_METRIC_RE.search(fact.source_claim)
    ):
        quality += 5
    if (
        _SUBGROUP_SCOPE_RE.search(fact.source_container_claim or fact.source_claim)
        and not _SUBGROUP_SCOPE_RE.search(requirement.metric)
        and not (_facet_families(requirement.metric) & {"education", "experience"})
    ):
        quality -= 6
    return quality


def _typed_fact_for_requirement(
    unit: EvidenceUnit,
    requirement: CoverageRequirement,
    execution_contract: Optional[RequestExecutionContract] = None,
    *,
    evidence_semantic_resolver: Optional[EvidenceSemanticResolver] = None,
    assisted_attempted_unit_ids: Optional[set[str]] = None,
    source_fact_cache: Optional[
        dict[str, tuple[tuple[TypedFact, ...], tuple[str, ...]]]
    ] = None,
    source_fact_occurrence_registry: Optional[
        dict[str, TypedFact | TextPropositionFact]
    ] = None,
) -> Optional[TypedFact]:
    """Bind an already-constructed SOURCE_FACT to one canonical requirement."""
    if execution_contract is not None and requirement.id not in execution_contract.requirement_ids:
        raise RuntimeError("fact extraction received noncanonical requirement ID")
    if execution_contract is None and os.getenv(
        "PROJECT_X_CANDIDATE_MODE", "0"
    ).strip().casefold() in {"1", "true", "yes", "on"}:
        raise RuntimeError("fact extraction requires RequestExecutionContract")
    canonical_requirement = next((
        item for item in getattr(execution_contract, "requirements", ())
        if item.id == requirement.id
    ), None)
    recorder = current_eval_trace()

    def terminal(outcome: str, *reason_codes: str, fact: Optional[TypedFact] = None) -> None:
        if recorder is not None and hasattr(recorder, "record_fact_terminal_outcome"):
            recorder.record_fact_terminal_outcome(
                requirement_id=requirement.id,
                evidence_id=unit.id,
                outcome=outcome,
                reason_codes=reason_codes,
                fact_id=(fact.fact_id if fact is not None else ""),
                source_fact_id=(fact.source_fact_id if fact is not None else ""),
                atom_id=(
                    fact.evidence_atom.atom_id
                    if fact is not None and fact.evidence_atom is not None else ""
                ),
            )

    if (
        canonical_requirement is not None
        and canonical_requirement.satisfaction_mode
        is not RequirementSatisfactionMode.SOURCE_FACT
    ):
        terminal("NOT_APPLICABLE", "requirement_satisfaction_mode_not_source_fact")
        return None
    cache = source_fact_cache if source_fact_cache is not None else {}
    source_facts, construction_reasons = cache.get(unit.id, ((), ()))
    if unit.id not in cache:
        source_facts, construction_reasons = _source_numeric_facts_for_unit(
            unit,
            source_fact_occurrence_registry=source_fact_occurrence_registry,
        )
        cache[unit.id] = (source_facts, construction_reasons)
    if not source_facts:
        reasons = construction_reasons or ("source_fact_not_constructed",)
        if recorder is not None and hasattr(recorder, "record_fact_rejection"):
            recorder.record_fact_rejection(
                requirement_id=requirement.id,
                evidence_id=unit.id,
                reason_codes=reasons,
            )
        terminal("REJECTED", *reasons)
        return None

    compatible: list[TypedFact] = []
    incompatibilities: list[str] = []
    for source_fact in source_facts:
        if canonical_requirement is None:
            # Non-candidate legacy callers predate immutable request
            # contracts.  Preserve their established base compatibility
            # behavior rather than pretending that plan labels are canonical
            # source constraints.  Candidate/runtime flows are required to
            # supply a contract above and always use the C19 authority.
            errors = _base_source_fact_compatibility_errors(
                requirement, None, source_fact,
            )
            dimension_decisions = {
                "final": "COMPATIBLE" if not errors else "INCOMPATIBLE",
            }
            admissibility_proof: Mapping[str, Mapping[str, Any]] = {}
            first_material_mismatch = next((
                _admissibility_reason_dimension(reason)
                for reason in errors if _admissibility_reason_dimension(reason)
            ), "")
        else:
            admissibility = canonical_fact_admissibility(
                requirement, canonical_requirement, source_fact,
            )
            errors = admissibility.reason_codes
            dimension_decisions = admissibility.dimension_decisions
            admissibility_proof = admissibility.dimension_proof
            first_material_mismatch = admissibility.first_material_mismatch
        if recorder is not None and hasattr(recorder, "record_fact_compatibility"):
            recorder.record_fact_compatibility(
                requirement_id=requirement.id,
                evidence_id=unit.id,
                source_fact=source_fact,
                compatible=not errors,
                reason_codes=errors,
                dimension_decisions=dimension_decisions,
                admissibility_proof=admissibility_proof,
                first_material_mismatch=first_material_mismatch,
            )
        if errors:
            incompatibilities.extend(errors)
            if recorder is not None and hasattr(recorder, "record_fact_rejection"):
                recorder.record_fact_rejection(
                    requirement_id=requirement.id,
                    evidence_id=unit.id,
                    reason_codes=errors,
                    source_fact_id=source_fact.source_fact_id,
                    atom_id=(
                        source_fact.evidence_atom.atom_id
                        if source_fact.evidence_atom is not None else ""
                    ),
                )
            terminal("INCOMPATIBLE", *errors, fact=source_fact)
            continue
        compatible.append(replace(
            source_fact,
            requirement_id=requirement.id,
            requirement_ids=(requirement.id,),
            origin_requirement_id=requirement.origin_id,
            subquestion_id=requirement.subquestion_id,
            association_quality=_source_fact_requirement_quality(
                requirement, source_fact,
            ),
        ))
    if not compatible:
        if not incompatibilities:
            terminal(
                "INCOMPATIBLE",
                "source_fact_has_no_requirement_compatibility",
                fact=source_facts[0],
            )
        return None
    if requirement.kind == "comparison_endpoint" and _comparison_endpoint_conflict(compatible):
        for fact in compatible:
            terminal("INCOMPATIBLE", "source_comparison_endpoint_conflict", fact=fact)
        return None
    if _canonical_requirement_value(requirement, canonical_requirement, "financial_role"):
        from financial_semantics import conflicting_monetary_amounts
        if conflicting_monetary_amounts(compatible):
            for fact in compatible:
                terminal("INCOMPATIBLE", "financial_document_amount_conflict", fact=fact)
            return None
    best = max(
        compatible,
        key=lambda fact: (
            fact.association_quality,
        ),
    )
    for fact in compatible:
        if fact.source_fact_id != best.source_fact_id:
            if recorder is not None and hasattr(
                recorder, "record_fact_compatibility_selection"
            ):
                recorder.record_fact_compatibility_selection(
                    requirement_id=requirement.id,
                    evidence_id=unit.id,
                    source_fact_id=fact.source_fact_id,
                    atom_id=(
                        fact.evidence_atom.atom_id
                        if fact.evidence_atom is not None else ""
                    ),
                    selected=False,
                )
            terminal(
                "COMPATIBLE_ALTERNATIVE_NOT_SELECTED",
                "better_local_atom_binding_selected",
                fact=fact,
            )
    # The best fact for this unit remains pending.  Only the caller that sees
    # every compatible unit may mark it selected_for_canonical and assign its
    # final terminal lineage.
    return best


def _source_text_fact_for_proposition(
    unit: EvidenceUnit,
    proposition: str,
    start: int,
    end: int,
    *,
    evidence_semantic_resolver: Optional[EvidenceSemanticResolver] = None,
    assisted_attempted_unit_ids: Optional[set[str]] = None,
    source_fact_occurrence_registry: Optional[
        dict[str, TypedFact | TextPropositionFact]
    ] = None,
) -> tuple[Optional[TextPropositionFact], tuple[str, ...]]:
    """Build one relation SOURCE_FACT without consulting a requirement."""
    semantics = _text_evidence_semantics(proposition)
    context = _evidence_context_for_proposition(
        unit, proposition, start, end, semantics, numeric_role="text_proposition",
    )
    # A bounded evidence resolver is still source-only at this boundary.  It
    # may repair an incomplete proposition before a canonical SOURCE_FACT is
    # constructed, but receives neither a requirement nor a request slot.
    attempted = (
        assisted_attempted_unit_ids
        if assisted_attempted_unit_ids is not None else set()
    )
    if (
        not context.evidence_subject
        or not context.evidence_predicate
        or not context.evidence_target_entity
        or relation_argument_validation_errors(context)
    ):
        resolved_context = _maybe_resolve_evidence_semantics(
            unit,
            context,
            evidence_semantic_resolver,
            attempted,
        )
        if resolved_context is not None:
            context = resolved_context
    atoms = evidence_atoms_for_context(context)
    if not atoms:
        _c45_application_value = None, ("evidence_atom_unresolved",)
        _c45_observe('C44E2E2_FAMILY_10_999f7411d1fa')
        return _c45_application_value
    atom = replace(
        atoms[0], source_evidence_unit_ids=unit.source_evidence_unit_ids,
    )
    recorder = current_eval_trace()
    if recorder is not None and hasattr(recorder, "record_evidence_atom"):
        recorder.record_evidence_atom(atom)
    fact = TextPropositionFact(
        requirement_id="",
        requirement_ids=(),
        origin_requirement_id="",
        subquestion_id="",
        subject=context.evidence_subject,
        predicate=context.evidence_predicate,
        object_text=proposition,
        qualifiers=(),
        evidence_id=unit.id,
        source_claim=proposition,
        structural_type=context.structural_type,
        proposition_start=start,
        proposition_end=end,
        proposition_sha256=hashlib.sha256(proposition.encode("utf-8")).hexdigest(),
        evidence_subject=context.evidence_subject,
        evidence_predicate=context.evidence_predicate,
        evidence_target_entity=context.evidence_target_entity,
        evidence_population=context.population,
        evidence_period=context.period,
        evidence_polarity=context.evidence_polarity,
        evidence_span=(start, end),
        candidate_id=unit.candidate_id,
        source_evidence_unit_ids=unit.source_evidence_unit_ids,
        evidence_context=context,
        relation_arguments=context.relation_arguments,
        model_assisted_semantics=context.model_assisted,
        evidence_atom=atom,
    )
    fact = _bind_source_fact_occurrence(unit, fact)
    errors = _source_fact_semantic_completeness_errors(fact)
    if errors:
        if recorder is not None and hasattr(recorder, "record_source_fact_construction"):
            recorder.record_source_fact_construction(
                fact, outcome="REJECTED", reason_codes=errors,
            )
        if recorder is not None and hasattr(
            recorder, "record_evidence_atom_terminal_outcome"
        ):
            recorder.record_evidence_atom_terminal_outcome(
                evidence_id=unit.id,
                candidate_id=unit.candidate_id,
                atom_id=atom.atom_id,
                outcome="REJECTED",
                reason_codes=errors,
                binding_status="atom_role_incomplete_for_source_fact",
            )
        _c45_application_value = None, errors
        _c45_observe('C44E2E2_FAMILY_10_999f7411d1fa')
        return _c45_application_value
    registered, emitted = _register_source_fact_occurrence(
        fact, source_fact_occurrence_registry,
    )
    if not isinstance(registered, TextPropositionFact):
        raise RuntimeError("text source fact occurrence changed fact family")
    if recorder is not None and hasattr(
        recorder, "record_source_fact_occurrence_observation"
    ):
        recorder.record_source_fact_occurrence_observation(
            observed_fact=fact,
            authoritative_fact=registered,
            outcome=(
                "EMITTED" if emitted else
                "REOBSERVED_SAME_STRUCTURAL_VIEW"
                if fact.evidence_id == registered.evidence_id else
                "REOBSERVED_EQUIVALENT_STRUCTURAL_VIEW"
            ),
        )
    if emitted and recorder is not None and hasattr(
        recorder, "record_source_fact_construction"
    ):
        recorder.record_source_fact_construction(registered, outcome="CREATED")
    if recorder is not None and hasattr(
        recorder, "record_evidence_atom_terminal_outcome"
    ):
        recorder.record_evidence_atom_terminal_outcome(
            evidence_id=unit.id,
            candidate_id=unit.candidate_id,
            atom_id=atom.atom_id,
            outcome=(
                "CONSUMED_BY_SOURCE_FACT"
                if emitted else "OBSERVED_BY_EXISTING_SOURCE_FACT"
            ),
            binding_status=(
                "atom_local_binding_consumed"
                if emitted else "atom_local_binding_reobserved"
            ),
        )
    _c45_application_value = registered, ()
    _c45_observe('C44E2E2_FAMILY_10_999f7411d1fa')
    return _c45_application_value


def _base_source_text_fact_compatibility_errors(
    requirement: Any,
    fact: TextPropositionFact,
) -> tuple[str, ...]:
    """Keep relation/requirement checks downstream of source construction."""
    reasons: list[str] = []
    reasons.extend(_evidence_fact_compatibility_errors(
        requirement,
        subject=fact.evidence_subject,
        predicate=fact.evidence_predicate,
        target_entity=fact.evidence_target_entity,
        population=fact.evidence_population,
        relation_arguments=fact.relation_arguments,
    ))
    reasons.extend(text_fact_binding_errors(requirement, fact))
    return tuple(dict.fromkeys(reasons))


def _source_text_fact_compatibility_errors(
    requirement: CoverageRequirement,
    canonical_requirement: Any,
    fact: TextPropositionFact,
) -> tuple[str, ...]:
    """Text compatibility adapter backed by canonical fact admissibility."""
    return canonical_fact_admissibility(
        requirement, canonical_requirement, fact,
    ).reason_codes


def _auxiliary_text_candidate_eligible(
    unit: EvidenceUnit,
    proposition: str,
    canonical_requirement: Any,
    execution_contract: Optional[RequestExecutionContract],
) -> bool:
    """Spend helper work only on a locally relevant unresolved proposition."""
    if execution_contract is None or canonical_requirement is None:
        return False
    from document_query import document_fact_errors, attribute_context_errors
    scope = requirement_dimension_constraint(canonical_requirement, "document_scope")
    if document_fact_errors(scope.value if scope is not None else "", unit):
        return False
    room = requirement_dimension_constraint(canonical_requirement, "attribute_context")
    if attribute_context_errors(room.value if room is not None else "", SimpleNamespace(
        evidence_context=unit.evidence_context, source_claim=proposition,
    )):
        return False
    hit = dict(unit.raw_hit)
    hit["document"] = proposition
    semantics = _text_evidence_semantics(proposition)
    decision = assess_candidate(
        hit, replace(execution_contract, requirements=(canonical_requirement,)),
        effective_tenant=str((hit.get("metadata") or {}).get("tenant_id") or ""),
    )
    if canonical_requirement.id not in decision.compatible_requirement_ids or decision.rejected:
        # Missing source predicate is precisely what an optional resolver can
        # establish. Every other candidate constraint must already pass.
        if not (
            not semantics["predicate"]
            and set(decision.rejection_reason_codes) == {"predicate_incompatible"}
            and decision.subject_compatibility == "compatible"
            and decision.metric_compatibility == "compatible"
        ):
            return False
    # A helper may complete missing fields, not revise a source relation
    # already known to contradict the requested relation or target.
    known_errors = _evidence_fact_compatibility_errors(
        canonical_requirement,
        subject=semantics["subject"], predicate=semantics["predicate"],
        target_entity=semantics["target_entity"], population=semantics["population"],
    )
    return not any(
        reason == "evidence_predicate_incompatible" and semantics["predicate"]
        or reason == "evidence_target_entity_incompatible" and semantics["target_entity"]
        or reason == "evidence_metric_incompatible" and semantics["subject"]
        for reason in known_errors
    )


def eligible_structure_extraction_units(
    plan: RequestPlan,
    units: Sequence[EvidenceUnit],
    execution_contract: Optional[RequestExecutionContract],
) -> tuple[EvidenceUnit, ...]:
    """Do not re-extract complete literal lists or rows to fill absent facts."""
    if execution_contract is None or plan.intent not in {RequestIntent.LIST, RequestIntent.LIST_WITH_ATTRIBUTES}:
        return ()
    requirements = tuple(requirement for requirement in execution_contract.requirements
                         if requirement.kind in {"list", "list_attributes"})
    eligible = []
    for unit in units:
        context = unit.evidence_context
        if (context is None or not context.structural_binding_valid
                or unit.structural_type not in {"prose_proposition", "other_bounded_unit"}):
            continue
        for proposition, _start, _end in _text_fact_propositions(unit.canonical_text):
            if not 4 <= len(proposition) <= MAX_ALLOWED_CLAIM_CHARS:
                continue
            semantics = _text_evidence_semantics(proposition)
            if all(semantics.get(name) for name in ("subject", "predicate", "target_entity")):
                continue
            if any(_auxiliary_text_candidate_eligible(unit, proposition, requirement, execution_contract)
                   for requirement in requirements):
                eligible.append(unit)
                break
    return tuple(eligible)


def _evidence_matches_requirement(
    unit: EvidenceUnit,
    *,
    subject: str,
    dimension: Optional[str] = None,
) -> bool:
    text = unit.canonical_text
    subject_focus = normalize_request_text(re.sub(
        r"(?<![\w])(?:19|20)\d{2}(?![\w])|"
        r"(?<![\w])(?:Q[1-4]|H[12])[- /]?(?:19|20)?\d{2}(?![\w])",
        " ",
        subject,
        flags=re.IGNORECASE,
    ))
    if _PURPOSE_ACTION_QUERY_RE.search(subject_focus):
        subject_focus = normalize_request_text(
            _PURPOSE_ACTION_BOILERPLATE_RE.sub(" ", subject_focus)
        )
    subject_overlap = max(
        _term_overlap(subject_focus, text),
        _semantic_term_overlap(subject_focus, text),
    )
    subject_terms = _normalized_terms(subject_focus)
    subject_ok = (
        subject_overlap >= 0.34
        or (
            len(subject_terms) == 1
            and subject_overlap > 0
        )
    )
    if not subject_ok:
        return False
    if dimension is None:
        return True
    return _contains_normalized_phrase(dimension, text) or dimension in text


def _dimension_diverse_hits(
    hits: Sequence[Mapping[str, Any]],
    subquestion: Subquestion,
) -> list[Mapping[str, Any]]:
    """Keep the highest-ranked hit for each required dimension before fill."""
    output: list[Mapping[str, Any]] = []
    seen: set[tuple[str, ...]] = set()

    def add(hit: Mapping[str, Any]) -> None:
        key = _hit_key(hit)
        if key not in seen:
            seen.add(key)
            output.append(hit)

    for dimension in subquestion.required_dimensions:
        for hit in hits:
            if _contains_normalized_phrase(
                dimension,
                str(hit.get("document") or ""),
            ):
                add(hit)
                break
    for hit in hits:
        add(hit)
    return output


def _validated_row(
    candidate: StructuredRowCandidate,
    *,
    units_by_id: Mapping[str, EvidenceUnit],
    requested_attributes: Sequence[str],
    requested_subject: str,
    require_semantic_class: bool = False,
) -> Optional[ListCoverageRow]:
    if not valid_structured_item_identity(candidate.item):
        return None
    item_ids = tuple(
        evidence_id
        for evidence_id in candidate.item_evidence_ids
        if evidence_id in units_by_id
    )
    if not item_ids:
        return None
    if not any(
        _term_overlap(candidate.item, units_by_id[evidence_id].canonical_text)
        >= 0.60
        for evidence_id in item_ids
    ):
        return None
    normalized_subject = normalize_request_text(requested_subject).casefold()
    semantic_context = normalize_request_text(" ".join(
        part
        for evidence_id in item_ids
        for part in (
            units_by_id[evidence_id].canonical_text,
            str((units_by_id[evidence_id].raw_hit.get("metadata") or {}).get("section") or ""),
            str((units_by_id[evidence_id].raw_hit.get("metadata") or {}).get("section_title") or ""),
            str((units_by_id[evidence_id].raw_hit.get("metadata") or {}).get("table_title") or ""),
            str((units_by_id[evidence_id].raw_hit.get("metadata") or {}).get("chart_title") or ""),
        )
    ))
    if require_semantic_class and normalized_subject and max(
        _term_overlap(normalized_subject, semantic_context),
        _semantic_term_overlap(normalized_subject, semantic_context),
    ) <= 0:
        recorder = current_eval_trace()
        if recorder is not None and hasattr(recorder, "record_fact_rejection"):
            for evidence_id in item_ids:
                recorder.record_fact_rejection(
                    requirement_id="list_semantic_membership",
                    evidence_id=evidence_id,
                    reason_codes=("list_item_semantic_class_incompatible",),
                )
        return None
    category_groups = [
        group
        for group in _LIST_CATEGORY_SYNONYM_GROUPS
        if any(
            term.casefold() in normalized_subject
            for term in group
        )
    ]
    if category_groups:
        category_context = normalize_request_text(" ".join(
            part
            for evidence_id in item_ids
            for part in (
                units_by_id[evidence_id].canonical_text,
                str(
                    (
                        units_by_id[evidence_id].raw_hit.get("metadata")
                        or {}
                    ).get("section")
                    or ""
                ),
                units_by_id[evidence_id].section_fingerprint,
            )
        )).casefold()
        if not all(
            any(term.casefold() in category_context for term in group)
            for group in category_groups
        ):
            return None
    attributes: dict[str, str] = {}
    attribute_ids: dict[str, tuple[str, ...]] = {}
    for requested in requested_attributes:
        value = normalize_request_text(candidate.attributes.get(requested, ""))
        ids = tuple(
            evidence_id
            for evidence_id in candidate.attribute_evidence_ids.get(requested, ())
            if evidence_id in units_by_id
        )
        if not value or not ids:
            continue
        if not any(
            _term_overlap(value, units_by_id[evidence_id].canonical_text) >= 0.38
            for evidence_id in ids
        ):
            continue
        # Preserve relationship binding at the source structure boundary.  A
        # section fingerprint is deliberately insufficient: two neighboring
        # list items often share it while describing different entities.
        shared = set(item_ids) & set(ids)
        structurally_linked = any(
            units_by_id[left].structural_evidence_unit is not None
            and units_by_id[right].structural_evidence_unit is not None
            and units_by_id[left].structural_evidence_unit.structural_group_id
            == units_by_id[right].structural_evidence_unit.structural_group_id
            for left in item_ids
            for right in ids
        )
        if not shared and not structurally_linked:
            continue
        attributes[requested] = value
        attribute_ids[requested] = ids
    complete = all(
        attribute in attributes for attribute in requested_attributes
    )
    return ListCoverageRow(
        item=normalize_request_text(candidate.item),
        attributes=attributes,
        item_evidence_ids=item_ids,
        attribute_evidence_ids=attribute_ids,
        complete=complete,
    )


def extract_deterministic_row_candidates(
    units: Sequence[EvidenceUnit],
    *,
    requested_attributes: Sequence[str],
) -> tuple[StructuredRowCandidate, ...]:
    """Extract explicit `item — attribute` rows without model inference."""
    output: list[StructuredRowCandidate] = []
    for unit in units:
        if not requested_attributes:
            inline_parts = re.split(
                r"\s*[•▪◦]\s*",
                unit.canonical_text,
            )
            bullet_items = [
                normalize_evidence_whitespace(item).strip(" .,:;")
                for item in inline_parts[1:]
                if normalize_evidence_whitespace(item).strip(" .,:;")
            ]
            if len(inline_parts) >= 2 and ":" in inline_parts[0]:
                first_item = normalize_evidence_whitespace(
                    inline_parts[0].rsplit(":", 1)[-1]
                ).strip(" .,:;")
                if first_item:
                    bullet_items.insert(0, first_item)
            for item in bullet_items:
                if (
                    valid_structured_item_identity(item)
                    and len(_normalized_terms(item)) >= 2
                ):
                    output.append(StructuredRowCandidate(
                        item=item,
                        attributes={},
                        item_evidence_ids=(unit.id,),
                        attribute_evidence_ids={},
                    ))
            if unit.structural_type in {
                StructuralUnitType.BULLET.value,
                StructuralUnitType.LIST.value,
            }:
                for item in unit.allowed_claims:
                    if _REFERENCE_ENTRY_START_RE.search(item):
                        break
                    item = normalize_evidence_whitespace(item).strip(" .,:;")
                    if (
                        valid_structured_item_identity(item)
                        and len(_normalized_terms(item)) >= 2
                    ):
                        output.append(StructuredRowCandidate(
                            item=item,
                            attributes={},
                            item_evidence_ids=(unit.id,),
                            attribute_evidence_ids={},
                        ))
        raw_lines = _BULLET_RE.findall(unit.canonical_text)
        for claim in unit.allowed_claims:
            if _REFERENCE_ENTRY_START_RE.search(claim):
                break
            raw_lines.append(claim)
        normalized_unit = "\n".join(
            normalize_evidence_whitespace(line)
            for line in unit.canonical_text.splitlines()
            if normalize_evidence_whitespace(line)
        )
        layout_rows: list[tuple[int, str, str]] = []
        for match in re.finditer(
            r"[•▪◦]\s*(?P<item>[^•]{2,100}?)\)\s*"
            r"(?P<description>.+?)"
            r"(?=\n|\s+(?:[A-Za-z][A-Za-z0-9_-]{1,40}\s+)?[•▪◦]|$)",
            normalized_unit,
        ):
            layout_rows.append((
                match.start(),
                match.group("item").strip(" .,:;()"),
                match.group("description").strip(" .,:;"),
            ))
        for match in re.finditer(
            r"(?P<item>[A-Za-z][A-Za-z0-9_-]{1,60})\s*[•▪◦]\s*:\s*"
            r"(?P<description>.+?)(?=\n|\s+[•▪◦]|$)",
            normalized_unit,
        ):
            layout_rows.append((
                match.start(),
                match.group("item").strip(" .,:;()"),
                match.group("description").strip(" .,:;"),
            ))
        for match in re.finditer(
            r"(?m)^\s*(?P<item>[^•:\n)]{2,100}?)\)\s*"
            r"(?P<description>[^\n]{4,})$",
            normalized_unit,
        ):
            layout_rows.append((
                match.start(),
                match.group("item").strip(" .,:;()"),
                match.group("description").strip(" .,:;"),
            ))
        seen_layout_items: set[str] = set()
        for _position, item, description in sorted(layout_rows):
            item_key = normalize_request_text(item).casefold()
            if item_key in seen_layout_items:
                continue
            if not (
                valid_structured_item_identity(item)
                and 1 <= len(_normalized_terms(item)) <= 12
                and len(_normalized_terms(description)) >= 2
            ):
                continue
            seen_layout_items.add(item_key)
            output.append(StructuredRowCandidate(
                item=item,
                attributes={
                    attribute: description
                    for attribute in requested_attributes
                },
                item_evidence_ids=(unit.id,),
                attribute_evidence_ids={
                    attribute: (unit.id,)
                    for attribute in requested_attributes
                },
            ))
        for line in raw_lines:
            parts = _ROW_SEPARATOR_RE.split(
                normalize_evidence_whitespace(line),
                maxsplit=1,
            )
            if len(parts) != 2:
                continue
            item, description = (part.strip(" .,:;") for part in parts)
            if not (
                valid_structured_item_identity(item)
                and 1 <= len(_normalized_terms(item)) <= 12
                and len(_normalized_terms(description)) >= 2
            ):
                continue
            attributes = {
                attribute: description
                for attribute in requested_attributes
            }
            ids = {
                attribute: (unit.id,)
                for attribute in requested_attributes
            }
            output.append(StructuredRowCandidate(
                item=item,
                attributes=attributes,
                item_evidence_ids=(unit.id,),
                attribute_evidence_ids=ids,
            ))
    return tuple(output)


def valid_structured_item_identity(value: str) -> bool:
    """Reject discourse markers and category placeholders as list items."""
    raw_normalized = normalize_request_text(value).casefold()
    normalized = raw_normalized
    normalized = re.sub(r"^[\W_]+|[\W_]+$", "", normalized)
    return bool(
        normalized
        and not re.search(r"(?:https?://|www\.|[\w.+-]+@[\w.-]+)", normalized)
        and "%" not in raw_normalized
        and normalized not in _GENERIC_STRUCTURED_ITEM_IDENTITIES
        and re.search(r"[A-Za-z\u05d0-\u05ea]", normalized)
    )


def _numeric_fact_value(value: str) -> Optional[float]:
    cleaned = re.sub(r"[^0-9.\-]", "", str(value or "").replace(",", ""))
    if not cleaned or cleaned.count(".") > 1 or cleaned.count("-") > 1:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _comparison_endpoint_conflict(facts: Sequence[TypedFact]) -> bool:
    """A ranking preference cannot adjudicate different authorized values."""
    return len({(_numeric_fact_value(fact.value), fact.unit) for fact in facts}) > 1


def _compatible_derivation_endpoints(
    left: TypedFact,
    right: TypedFact,
) -> tuple[bool, tuple[str, ...]]:
    """Require source facts to describe one quantity before deriving change."""
    reasons = compatible_source_endpoints(left, right)
    return (not reasons, reasons)


def _derived_direction_fact(
    requirement: Any,
    spec: CoverageRequirement,
    endpoint_facts: Sequence[TypedFact],
) -> tuple[Optional[TypedFact], tuple[str, ...]]:
    """Build a direction only from two compatible, source-bound endpoints."""
    endpoints = list(endpoint_facts)
    if len(endpoints) != 2:
        return None, ("derived_endpoint_count_incomplete",)
    endpoints.sort(key=source_endpoint_order)
    left, right = endpoints
    compatible, reasons = _compatible_derivation_endpoints(left, right)
    if not compatible:
        return None, reasons
    left_value = _numeric_fact_value(left.value)
    right_value = _numeric_fact_value(right.value)
    assert left_value is not None and right_value is not None
    direction = (
        "increase" if right_value > left_value
        else "decrease" if right_value < left_value
        else "stable"
    )
    source_ids = tuple(dict.fromkeys(
        (*left.source_evidence_unit_ids, *right.source_evidence_unit_ids)
    ))
    candidate_ids = tuple(dict.fromkeys(
        (*left.source_candidate_ids, *right.source_candidate_ids)
    ))
    fact = TypedFact(
        requirement_id=spec.id,
        requirement_ids=(spec.id,),
        origin_requirement_id=spec.origin_id,
        subquestion_id=spec.subquestion_id,
        subject=left.evidence_subject,
        metric=left.metric,
        facet=left.facet,
        period=f"{left.period}->{right.period}",
        time_role="derived",
        value=direction,
        unit="derived_direction",
        direction=direction,
        qualifiers=tuple(dict.fromkeys((*left.qualifiers, *right.qualifiers))),
        evidence_id=left.evidence_id,
        # This is a computation with endpoint provenance, not a source quote
        # or a numeric evidence atom. Each endpoint retains its own atom.
        source_claim="",
        structural_type="derived_direction",
        population=left.population,
        evidence_subject=left.evidence_subject,
        evidence_predicate=left.evidence_predicate,
        evidence_target_entity=left.evidence_target_entity,
        evidence_population=left.evidence_population,
        evidence_period=f"{left.period}->{right.period}",
        evidence_unit="derived_direction",
        evidence_numeric_role="derived_direction",
        evidence_polarity=left.evidence_polarity,
        candidate_id=left.candidate_id,
        source_evidence_unit_ids=source_ids,
        source_candidate_ids=candidate_ids,
        source_typed_fact_ids=(left.fact_id, right.fact_id),
        source_typed_fact_sha256s=(source_endpoint_digest(left), source_endpoint_digest(right)),
        derivation_description="direction_from_two_compatible_source_endpoints",
        fact_type="derived_direction",
    )
    return fact, ()


def _ordered_list_facts_for_requirement(
    requirement: Any,
    units: Sequence[EvidenceUnit],
) -> tuple[TextPropositionFact, ...]:
    """Select one complete source list before applying request ownership."""
    from ordered_list_evidence import eligible_ordered_list_groups, ordered_list_fact_errors
    groups = eligible_ordered_list_groups(requirement, units)
    if len(groups) != 1:
        return ()
    facts = []
    for unit in groups[0]:
        structural = unit.structural_evidence_unit
        assert structural is not None
        label = ordered_list_item_text(structural)
        start = unit.canonical_text.find(label)
        if start < 0:
            return ()
        end = start + len(label)
        context = _evidence_context_for_proposition(
            unit, label, start, end,
            {"subject": label, "predicate": "has_members", "target_entity": label,
             "polarity": "neutral"},
        )
        raw = str(unit.raw_hit.get("document") or "")
        governing_span = structural.list_context_span
        assert governing_span is not None
        governing = raw[governing_span.start:governing_span.end]
        context = replace(context, governing_context=governing,
                          governing_context_provenance=FieldProvenance(
                              field="governing_context", value=governing,
                              provenance_type="governing_source_heading",
                              source_container="source_candidate",
                              source_span=(governing_span.start, governing_span.end),
                              source_sha256=governing_span.sha256, confidence=1.0,
                              revisability="fixed_source_structure", source_quote=governing,
                              structural_unit_id=structural.unit_id,
                              structural_group_id=structural.structural_group_id or structural.unit_id,
                              structural_role="ordered_list_governing_context"))
        fact = TextPropositionFact(
            requirement_id=requirement.id, requirement_ids=(requirement.id,),
            origin_requirement_id=f"item_{structural.list_ordinal}",
            subquestion_id=requirement.clause_id.replace("clause-", "sq"),
            subject=label, predicate="has_members", object_text=label, qualifiers=(),
            evidence_id=unit.id, source_claim=unit.canonical_text,
            structural_type=unit.structural_type, evidence_span=(start, end),
            proposition_start=start, proposition_end=end,
            proposition_sha256=hashlib.sha256(label.encode()).hexdigest(),
            evidence_subject=context.evidence_subject,
            evidence_predicate=context.evidence_predicate,
            evidence_target_entity=context.evidence_target_entity,
            candidate_id=unit.candidate_id, evidence_context=context,
        )
        if ordered_list_fact_errors(requirement, fact):
            return ()
        facts.append(fact)
    return tuple(facts)


def build_evidence_coverage(
    plan: RequestPlan,
    units: Sequence[EvidenceUnit],
    *,
    structured_rows: Sequence[StructuredRowCandidate] = (),
    requirement_specs: Sequence[CoverageRequirement] = (),
    execution_contract: Optional[RequestExecutionContract] = None,
    evidence_semantic_resolver: Optional[EvidenceSemanticResolver] = None,
    _assisted_attempted_unit_ids: Optional[set[str]] = None,
) -> EvidenceCoverage:
    source_units = tuple(units)
    recovered_units = _bounded_adjacent_evidence_units(source_units)
    units = (*source_units, *recovered_units)
    units_by_id = {unit.id: unit for unit in units}
    requirements: list[CoverageRequirement] = []
    rows: list[ListCoverageRow] = []
    typed_facts: list[TypedFact | TextPropositionFact] = []
    assisted_attempted_unit_ids = (
        _assisted_attempted_unit_ids
        if _assisted_attempted_unit_ids is not None else set()
    )

    canonical_lists = tuple(
        item for item in getattr(execution_contract, "requirements", ())
        if item.kind in {"list", "list_attributes"}
    )
    list_requirement_ids = {item.id for item in canonical_lists}
    list_plan = plan.intent in {
        RequestIntent.LIST,
        RequestIntent.LIST_WITH_ATTRIBUTES,
    }
    list_requested = bool(canonical_lists) if execution_contract is not None else list_plan
    list_budget = max((item.cardinality for item in canonical_lists), default=plan.requested_items)
    if list_requested:
        requested_subject = next(
            (
                subquestion.subject
                for subquestion in plan.subquestions
                if subquestion.question_type is QuestionType.ITEM_DISCOVERY
            ),
            canonical_lists[0].subject if canonical_lists and not list_plan else "",
        )
        requested_attributes = (
            plan.requested_attributes
            if plan.intent is RequestIntent.LIST_WITH_ATTRIBUTES
            else ()
        )
        ordered_facts_by_requirement = {
            item.id: _ordered_list_facts_for_requirement(item, units)
            for item in getattr(execution_contract, "requirements", ())
            if item.kind == "list"
        }
        complete_ordered_facts = next(
            (facts for facts in ordered_facts_by_requirement.values() if facts), ()
        )
        candidates = tuple(structured_rows) or extract_deterministic_row_candidates(
            units,
            requested_attributes=requested_attributes,
        )
        seen_items: list[str] = []
        if complete_ordered_facts:
            rows.extend(ListCoverageRow(
                item=fact.object_text, attributes={}, item_evidence_ids=(fact.evidence_id,),
                attribute_evidence_ids={}, complete=True,
            ) for fact in complete_ordered_facts)
        for candidate in candidates:
            if complete_ordered_facts:
                break
            row = _validated_row(
                candidate,
                units_by_id=units_by_id,
                requested_attributes=requested_attributes,
                requested_subject=requested_subject,
                require_semantic_class=execution_contract is not None,
            )
            if row is None or any(
                _term_overlap(row.item, existing) >= 0.85
                and _term_overlap(existing, row.item) >= 0.85
                for existing in seen_items
            ):
                continue
            seen_items.append(row.item)
            rows.append(row)
            if len(rows) >= list_budget:
                break
        if execution_contract is not None:
            for item in execution_contract.requirements:
                if item.kind not in {"list", "list_attributes"}:
                    continue
                is_attribute = item.kind == "list_attributes"
                ordered_facts = ordered_facts_by_requirement.get(item.id, ())
                from semantic_contracts import requirement_dimension_value
                ordered_required = bool(requirement_dimension_value(item, "ordered_list_heading"))
                evidence_ids = tuple(dict.fromkeys(
                    evidence_id
                    for row in rows
                    for evidence_id in (
                        tuple(
                            evidence
                            for values in row.attribute_evidence_ids.values()
                            for evidence in values
                        )
                        if is_attribute else row.item_evidence_ids
                    )
                ))
                enough = len(rows) >= item.cardinality and (
                    not is_attribute
                    or all(row.complete for row in rows[:item.cardinality])
                )
                if ordered_required:
                    enough = bool(ordered_facts)
                    evidence_ids = tuple(fact.evidence_id for fact in ordered_facts)
                requirements.append(CoverageRequirement(
                    id=item.id,
                    subquestion_id=item.clause_id.replace("clause-", "sq"),
                    kind=item.kind,
                    label=item.subject or item.predicate,
                    status=(
                        CoverageStatus.COVERED
                        if enough else CoverageStatus.MISSING
                    ),
                    evidence_ids=evidence_ids,
                    origin_id="list_rows",
                    requirement_ids=(item.id,),
                ))
                if ordered_required:
                    typed_facts.extend(ordered_facts)
                    continue
                for row_index, row in enumerate(rows[:item.cardinality], start=1):
                    row_evidence_ids = (
                        tuple(
                            evidence
                            for values in row.attribute_evidence_ids.values()
                            for evidence in values
                        )
                        if is_attribute else row.item_evidence_ids
                    )
                    if not row_evidence_ids:
                        continue
                    object_text = (
                        "; ".join(
                            f"{name}: {value}"
                            for name, value in row.attributes.items()
                        )
                        if is_attribute else row.item
                    )
                    row_unit = units_by_id[row_evidence_ids[0]]
                    source_claim = row_unit.canonical_text
                    row_context = _evidence_context_for_proposition(
                        row_unit,
                        source_claim,
                        0,
                        len(source_claim),
                        {
                            "subject": row.item,
                            "predicate": (
                                "has_attribute" if is_attribute else "has_members"
                            ),
                            "target_entity": row.item,
                            "category": requested_subject,
                            "polarity": "neutral",
                        },
                    )
                    fact = TextPropositionFact(
                        requirement_id=item.id,
                        requirement_ids=(item.id,),
                        origin_requirement_id=f"item_{row_index}",
                        subquestion_id=item.clause_id.replace("clause-", "sq"),
                        subject=row.item,
                        predicate=("has_attribute" if is_attribute else "has_members"),
                        object_text=object_text,
                        qualifiers=tuple(row.attributes) if is_attribute else (),
                        evidence_id=row_evidence_ids[0],
                        source_claim=source_claim,
                        structural_type=row_unit.structural_type,
                        evidence_subject=row_context.evidence_subject,
                        evidence_predicate=row_context.evidence_predicate,
                        evidence_target_entity=row_context.evidence_target_entity,
                        evidence_span=(0, len(source_claim)),
                        candidate_id=row_unit.candidate_id,
                        evidence_context=row_context,
                    )
                    if fact.canonical_valid:
                        typed_facts.append(fact)
                    else:
                        recorder = current_eval_trace()
                        if recorder is not None and hasattr(
                            recorder, "record_fact_rejection"
                        ):
                            recorder.record_fact_rejection(
                                requirement_id=item.id,
                                evidence_id=row_evidence_ids[0],
                                reason_codes=fact.canonical_validation_errors,
                            )
        else:
            for index in range(plan.requested_items):
                row = rows[index] if index < len(rows) else None
                requirements.append(CoverageRequirement(
                    id=f"req-legacy-item-{index + 1}", subquestion_id="sq1",
                    kind="list_item", label=f"item_{index + 1}",
                    status=(CoverageStatus.COVERED if row else CoverageStatus.MISSING),
                    evidence_ids=row.item_evidence_ids if row else (),
                    origin_id=f"item_{index + 1}",
                    requirement_ids=(f"req-legacy-item-{index + 1}",),
                ))
                for attribute in requested_attributes:
                    evidence_ids = row.attribute_evidence_ids.get(attribute, ()) if row else ()
                    requirements.append(CoverageRequirement(
                        id=f"req-legacy-item-{index + 1}-attribute", subquestion_id="sq2",
                        kind="list_attribute", label=attribute,
                        status=(CoverageStatus.COVERED if evidence_ids else CoverageStatus.MISSING),
                        evidence_ids=tuple(evidence_ids),
                        origin_id=f"item_{index + 1}_{attribute}",
                        requirement_ids=(f"req-legacy-item-{index + 1}-attribute",),
                    ))
    # A global LIST plan does not own the other canonical clauses, and a
    # factual plan does not erase an independently compiled list clause.
    # Keep the existing source caches and assisted-attempt registry shared
    # while dispatching each canonical requirement exactly once.
    if not list_requested or any(
        item.id not in list_requirement_ids
        for item in getattr(execution_contract, "requirements", ())
    ):
        specs = tuple(
            item for item in (tuple(requirement_specs) or derive_evidence_requirements(
                plan, execution_contract
            ))
            if item.id not in list_requirement_ids
        )
        unit_order = {
            unit.id: index for index, unit in enumerate(units)
        }
        canonical_by_id = {
            item.id: item
            for item in getattr(execution_contract, "requirements", ())
        }
        source_fact_cache: dict[
            str, tuple[tuple[TypedFact, ...], tuple[str, ...]]
        ] = {}
        # SOURCE_FACT construction is idempotent within this execution.  It
        # is intentionally not process-global: another request must retain a
        # visible observation of the same source-local identity in its trace.
        source_fact_occurrence_registry: dict[
            str, TypedFact | TextPropositionFact
        ] = {}
        # Construct all deterministic text source facts before optional model
        # work. This cache is execution-local and contains source-only facts;
        # canonical compatibility is still evaluated separately per request.
        text_source_cache = {}
        if any(not spec.requires_value and spec.satisfaction_mode == RequirementSatisfactionMode.SOURCE_FACT.value
               for spec in specs):
            for unit in units:
                text_source_cache[unit.id] = tuple(
                    (text, start, end, *_source_text_fact_for_proposition(
                        unit, text, start, end,
                        source_fact_occurrence_registry=source_fact_occurrence_registry,
                    ))
                    for text, start, end in _text_fact_propositions(unit.canonical_text)
                )
        deferred_derived: list[tuple[CoverageRequirement, Any]] = []
        deferred_composite: list[tuple[CoverageRequirement, Any]] = []

        def finalize_selected_source_fact(
            requirement: CoverageRequirement,
            fact: TypedFact | TextPropositionFact,
            *,
            selected: bool,
            reason_code: str = "",
        ) -> None:
            """Close the global selection lifecycle for one compatible fact.

            Per-unit extraction can identify only compatible candidates.  The
            requirement loop is the first point that sees every candidate, so
            it is the sole owner of selected_for_canonical and its terminal
            disposition.
            """
            recorder = current_eval_trace()
            if recorder is None:
                return
            atom_id = (
                fact.evidence_atom.atom_id
                if getattr(fact, "evidence_atom", None) is not None else ""
            )
            if hasattr(recorder, "record_fact_compatibility_selection"):
                recorder.record_fact_compatibility_selection(
                    requirement_id=requirement.id,
                    evidence_id=fact.evidence_id,
                    source_fact_id=fact.source_fact_id,
                    atom_id=atom_id,
                    selected=selected,
                )
            if hasattr(recorder, "record_fact_terminal_outcome"):
                recorder.record_fact_terminal_outcome(
                    requirement_id=requirement.id,
                    evidence_id=fact.evidence_id,
                    outcome=(
                        "EMITTED_DIRECT" if selected
                        else "COMPATIBLE_ALTERNATIVE_NOT_SELECTED"
                    ),
                    reason_codes=(
                        () if selected else (
                            reason_code or "better_compatible_fact_selected",
                        )
                    ),
                    fact_id=fact.fact_id,
                    source_fact_id=fact.source_fact_id,
                    atom_id=atom_id,
                )
            if hasattr(recorder, "record_source_fact_reobservation_terminal"):
                recorder.record_source_fact_reobservation_terminal(
                    requirement_id=requirement.id,
                    source_fact_id=fact.source_fact_id or fact.fact_id,
                )
        for spec in specs:
            canonical_requirement = canonical_by_id.get(spec.id)
            satisfaction_mode = (
                canonical_requirement.satisfaction_mode
                if canonical_requirement is not None
                else RequirementSatisfactionMode(spec.satisfaction_mode)
            )
            if satisfaction_mode is RequirementSatisfactionMode.REQUEST_CONDITION:
                requirements.append(replace(
                    spec,
                    status=CoverageStatus.COVERED,
                    evidence_ids=(),
                    satisfaction_mode=satisfaction_mode.value,
                ))
                recorder = current_eval_trace()
                if recorder is not None and hasattr(
                    recorder, "record_fact_terminal_outcome"
                ):
                    recorder.record_fact_terminal_outcome(
                        requirement_id=spec.id,
                        evidence_id="",
                        outcome="REQUEST_CONDITION",
                        reason_codes=("request_condition_validated_without_source_extraction",),
                    )
                continue
            if satisfaction_mode is RequirementSatisfactionMode.DERIVED_FACT:
                deferred_derived.append((spec, canonical_requirement))
                continue
            if satisfaction_mode is RequirementSatisfactionMode.COMPOSITE_REQUIREMENT:
                deferred_composite.append((spec, canonical_requirement))
                continue
            if spec.requires_value:
                matched_facts = [
                    fact
                    for unit in units
                    if (
                        spec.subquestion_id in unit.subquestion_ids
                        or spec.id in unit.subquestion_ids
                    )
                    for fact in (
                        _typed_fact_for_requirement(
                            unit,
                            spec,
                            execution_contract,
                            evidence_semantic_resolver=evidence_semantic_resolver,
                            assisted_attempted_unit_ids=assisted_attempted_unit_ids,
                            source_fact_cache=source_fact_cache,
                            source_fact_occurrence_registry=(
                                source_fact_occurrence_registry
                            ),
                        ),
                    )
                    if fact is not None
                ]
                matched_facts.sort(key=lambda fact: (
                    -fact.association_quality,
                    unit_order.get(fact.evidence_id, len(unit_order)),
                    fact.evidence_id,
                ))
                # The occurrence registry can expose the same authoritative
                # SOURCE_FACT through several structural observations.  Those
                # observations remain in the trace, but one fact cannot be
                # selected and then displaced as its own alternative.
                authoritative_matches: list[TypedFact] = []
                selected_source_fact_ids: set[str] = set()
                for fact in matched_facts:
                    source_fact_id = fact.source_fact_id or fact.fact_id
                    if source_fact_id in selected_source_fact_ids:
                        continue
                    selected_source_fact_ids.add(source_fact_id)
                    authoritative_matches.append(fact)
                conflict_reason = ""
                if spec.kind == "comparison_endpoint" and _comparison_endpoint_conflict(authoritative_matches):
                    conflict_reason = "source_comparison_endpoint_conflict"
                elif _canonical_requirement_value(spec, canonical_requirement, "financial_role"):
                    from financial_semantics import conflicting_monetary_amounts
                    # A compound source unit may itself be ambiguous and
                    # return no preferred fact. Its independently compatible
                    # source facts still prevent another view from selecting
                    # just one of the contradictory payable amounts.
                    compatible_source_facts = [
                        fact for source_facts, _reasons in source_fact_cache.values()
                        for fact in source_facts
                        if not canonical_fact_admissibility(spec, canonical_requirement, fact).reason_codes
                    ]
                    if conflicting_monetary_amounts(compatible_source_facts):
                        conflict_reason = "financial_document_amount_conflict"
                if conflict_reason:
                    recorder = current_eval_trace()
                    for fact in authoritative_matches:
                        if recorder is not None:
                            recorder.record_fact_rejection(
                                requirement_id=spec.id,
                                evidence_id=fact.evidence_id,
                                reason_codes=(conflict_reason,),
                                source_fact_id=fact.source_fact_id,
                            )
                            recorder.record_fact_terminal_outcome(
                                requirement_id=spec.id, evidence_id=fact.evidence_id,
                                outcome="INCOMPATIBLE", reason_codes=(conflict_reason,),
                                fact_id=fact.fact_id, source_fact_id=fact.source_fact_id,
                            )
                    authoritative_matches = []
                evidence_ids = tuple(dict.fromkeys(
                    fact.evidence_id for fact in authoritative_matches
                ))
                requirements.append(replace(
                    spec,
                    status=(
                        CoverageStatus.COVERED
                        if evidence_ids
                        else CoverageStatus.MISSING
                    ),
                    evidence_ids=evidence_ids,
                ))
                if authoritative_matches:
                    selected_fact = authoritative_matches[0]
                    finalize_selected_source_fact(spec, selected_fact, selected=True)
                    for alternative in authoritative_matches[1:]:
                        finalize_selected_source_fact(
                            spec,
                            alternative,
                            selected=False,
                        )
                    typed_facts.append(selected_fact)
                continue
            proposition_matches: list[tuple[EvidenceUnit, TextPropositionFact]] = []
            unresolved_propositions = []
            for unit in units:
                if not (
                    spec.subquestion_id in unit.subquestion_ids
                    or spec.id in unit.subquestion_ids
                ):
                    continue
                selected: Optional[TextPropositionFact] = None
                compatible_candidates: list[TextPropositionFact] = []
                unit_reasons: list[str] = []
                for text, start, end, source_fact, construction_errors in text_source_cache.get(unit.id, ()):
                    if source_fact is None:
                        unit_reasons.extend(construction_errors)
                        if construction_errors and all(
                            reason in {"source_subject_unresolved", "source_predicate_unresolved",
                                       "source_target_entity_unresolved", "evidence_atom_unresolved"}
                            or reason.startswith("relation_argument_missing:")
                            for reason in construction_errors
                        ):
                            unresolved_propositions.append((unit, text, start, end))
                        continue
                    if canonical_requirement is None:
                        # Legacy plan-only text flows historically had no
                        # typed requirement to compare here.  They remain
                        # outside the candidate contract path; do not turn
                        # their decomposition labels into C19 constraints.
                        errors = ()
                        dimension_decisions = {"final": "COMPATIBLE"}
                        admissibility_proof: Mapping[str, Mapping[str, Any]] = {}
                        first_material_mismatch = ""
                    else:
                        admissibility = canonical_fact_admissibility(
                            spec, canonical_requirement, source_fact,
                        )
                        errors = admissibility.reason_codes
                        dimension_decisions = admissibility.dimension_decisions
                        admissibility_proof = admissibility.dimension_proof
                        first_material_mismatch = (
                            admissibility.first_material_mismatch
                        )
                    recorder = current_eval_trace()
                    if recorder is not None and hasattr(recorder, "record_fact_compatibility"):
                        recorder.record_fact_compatibility(
                            requirement_id=spec.id,
                            evidence_id=unit.id,
                            source_fact=source_fact,
                            compatible=not errors,
                            reason_codes=errors,
                            dimension_decisions=dimension_decisions,
                            admissibility_proof=admissibility_proof,
                            first_material_mismatch=first_material_mismatch,
                        )
                    if errors:
                        unit_reasons.extend(errors)
                        if recorder is not None and hasattr(recorder, "record_fact_rejection"):
                            recorder.record_fact_rejection(
                                requirement_id=spec.id,
                                evidence_id=unit.id,
                                source_fact_id=source_fact.source_fact_id,
                                atom_id=(
                                    source_fact.evidence_atom.atom_id
                                    if source_fact.evidence_atom is not None else ""
                                ),
                                reason_codes=errors,
                            )
                        if hasattr(recorder, "record_fact_terminal_outcome"):
                            recorder.record_fact_terminal_outcome(
                                requirement_id=spec.id,
                                evidence_id=unit.id,
                                outcome="INCOMPATIBLE",
                                reason_codes=errors,
                                fact_id=source_fact.fact_id,
                                source_fact_id=source_fact.source_fact_id,
                                atom_id=(
                                    source_fact.evidence_atom.atom_id
                                    if source_fact.evidence_atom is not None else ""
                                ),
                            )
                        continue
                    compatible_candidates.append(replace(
                        source_fact,
                        requirement_id=spec.id,
                        requirement_ids=(spec.id,),
                        origin_requirement_id=spec.origin_id,
                        subquestion_id=spec.subquestion_id,
                    ))
                if compatible_candidates:
                    selected = compatible_candidates[0]
                    recorder = current_eval_trace()
                    for candidate in compatible_candidates:
                        if candidate.source_fact_id == selected.source_fact_id:
                            continue
                        if recorder is not None and hasattr(
                            recorder, "record_fact_compatibility_selection"
                        ):
                            recorder.record_fact_compatibility_selection(
                                requirement_id=spec.id,
                                evidence_id=unit.id,
                                source_fact_id=candidate.source_fact_id,
                                atom_id=(
                                    candidate.evidence_atom.atom_id
                                if candidate.evidence_atom is not None else ""
                                ),
                                selected=False,
                            )
                        if recorder is not None and hasattr(
                            recorder, "record_fact_terminal_outcome"
                        ):
                            recorder.record_fact_terminal_outcome(
                                requirement_id=spec.id,
                                evidence_id=unit.id,
                                outcome="COMPATIBLE_ALTERNATIVE_NOT_SELECTED",
                                reason_codes=("better_local_proposition_selected",),
                                fact_id=candidate.fact_id,
                                source_fact_id=candidate.source_fact_id,
                                atom_id=(
                                    candidate.evidence_atom.atom_id
                                    if candidate.evidence_atom is not None else ""
                                ),
                            )
                if selected is None:
                    recorder = current_eval_trace()
                    if recorder is not None and hasattr(
                        recorder, "record_fact_rejection"
                    ):
                        recorder.record_fact_rejection(
                            requirement_id=spec.id,
                            evidence_id=unit.id,
                            reason_codes=tuple(dict.fromkeys(
                                unit_reasons or (
                                    "source_text_fact_has_no_requirement_compatibility",
                                )
                            )),
                        )
                    if hasattr(recorder, "record_fact_terminal_outcome"):
                        recorder.record_fact_terminal_outcome(
                            requirement_id=spec.id,
                            evidence_id=unit.id,
                            outcome="INCOMPATIBLE" if unit_reasons else "REJECTED",
                            reason_codes=tuple(dict.fromkeys(
                                unit_reasons or (
                                    "source_text_fact_not_constructed",
                                )
                            )),
                        )
                    continue
                proposition_matches.append((unit, selected))
            if not proposition_matches and evidence_semantic_resolver is not None:
                for unit, text, start, end in unresolved_propositions:
                    if unit.id in assisted_attempted_unit_ids or not _auxiliary_text_candidate_eligible(
                        unit, text, canonical_requirement, execution_contract,
                    ):
                        continue
                    source_fact, _errors = _source_text_fact_for_proposition(
                        unit, text, start, end,
                        evidence_semantic_resolver=evidence_semantic_resolver,
                        assisted_attempted_unit_ids=assisted_attempted_unit_ids,
                        source_fact_occurrence_registry=source_fact_occurrence_registry,
                    )
                    if source_fact is None:
                        continue
                    admissibility = canonical_fact_admissibility(spec, canonical_requirement, source_fact)
                    recorder = current_eval_trace()
                    if recorder is not None and hasattr(recorder, "record_fact_compatibility"):
                        recorder.record_fact_compatibility(
                            requirement_id=spec.id, evidence_id=unit.id, source_fact=source_fact,
                            compatible=not admissibility.reason_codes,
                            reason_codes=admissibility.reason_codes,
                            dimension_decisions=admissibility.dimension_decisions,
                            admissibility_proof=admissibility.dimension_proof,
                            first_material_mismatch=admissibility.first_material_mismatch,
                        )
                    if admissibility.compatible:
                        proposition_matches.append((unit, replace(
                            source_fact, requirement_id=spec.id, requirement_ids=(spec.id,),
                            origin_requirement_id=spec.origin_id, subquestion_id=spec.subquestion_id,
                        )))
                        break
            matching = tuple(
                dict.fromkeys(item[0].id for item in proposition_matches)
            )
            requirements.append(replace(
                spec,
                status=(
                    CoverageStatus.COVERED
                    if matching
                    else CoverageStatus.MISSING
                ),
                evidence_ids=matching,
            ))
            if proposition_matches:
                authoritative_proposition_matches: list[
                    tuple[EvidenceUnit, TextPropositionFact]
                ] = []
                selected_source_fact_ids: set[str] = set()
                for match_unit, match_fact in proposition_matches:
                    source_fact_id = match_fact.source_fact_id or match_fact.fact_id
                    if source_fact_id in selected_source_fact_ids:
                        continue
                    selected_source_fact_ids.add(source_fact_id)
                    authoritative_proposition_matches.append(
                        (match_unit, match_fact)
                    )
                _matching_unit, fact = authoritative_proposition_matches[0]
                finalize_selected_source_fact(spec, fact, selected=True)
                for _unit, alternative in authoritative_proposition_matches[1:]:
                    finalize_selected_source_fact(spec, alternative, selected=False)
                typed_facts.append(fact)

        requirement_rows = {item.id: item for item in requirements}
        facts_by_requirement: dict[str, list[TypedFact]] = defaultdict(list)
        for fact in typed_facts:
            if not isinstance(fact, TypedFact) or fact.fact_type != "numeric":
                continue
            for requirement_id in fact.requirement_ids:
                facts_by_requirement[requirement_id].append(fact)
        for spec, canonical_requirement in deferred_derived:
            linked_ids = tuple(
                getattr(canonical_requirement, "linked_requirement_ids", ()) or ()
            )
            endpoint_facts = [
                facts_by_requirement[requirement_id][0]
                for requirement_id in linked_ids
                if facts_by_requirement.get(requirement_id)
            ]
            fact, reasons = _derived_direction_fact(
                canonical_requirement, spec, endpoint_facts
            )
            recorder = current_eval_trace()
            if fact is None:
                requirements.append(replace(
                    spec, status=CoverageStatus.MISSING, evidence_ids=(),
                    satisfaction_mode=RequirementSatisfactionMode.DERIVED_FACT.value,
                ))
                if recorder is not None and hasattr(
                    recorder, "record_fact_terminal_outcome"
                ):
                    recorder.record_fact_terminal_outcome(
                        requirement_id=spec.id,
                        evidence_id="",
                        outcome="DERIVED_REQUIREMENT",
                        reason_codes=reasons,
                    )
                continue
            typed_facts.append(fact)
            requirements.append(replace(
                spec,
                status=CoverageStatus.COVERED,
                evidence_ids=tuple(dict.fromkeys(fact.source_evidence_unit_ids)),
                satisfaction_mode=RequirementSatisfactionMode.DERIVED_FACT.value,
            ))
            if recorder is not None and hasattr(
                recorder, "record_fact_terminal_outcome"
            ):
                recorder.record_fact_terminal_outcome(
                    requirement_id=spec.id,
                    evidence_id=fact.evidence_id,
                    outcome="DERIVED_REQUIREMENT",
                    reason_codes=("derived_from_compatible_source_endpoints",),
                    fact_id=fact.fact_id,
                    atom_id=(fact.evidence_atom.atom_id if fact.evidence_atom else ""),
                )
        requirement_rows = {item.id: item for item in requirements}
        for spec, canonical_requirement in deferred_composite:
            linked_ids = tuple(
                getattr(canonical_requirement, "linked_requirement_ids", ()) or ()
            )
            covered = bool(linked_ids) and all(
                requirement_rows.get(requirement_id, CoverageRequirement(
                    id="", subquestion_id="", kind="", label="",
                    status=CoverageStatus.MISSING, evidence_ids=(),
                )).status is CoverageStatus.COVERED
                for requirement_id in linked_ids
            )
            requirements.append(replace(
                spec,
                status=(CoverageStatus.COVERED if covered else CoverageStatus.MISSING),
                evidence_ids=tuple(dict.fromkeys(
                    evidence_id
                    for requirement_id in linked_ids
                    for evidence_id in requirement_rows.get(
                        requirement_id,
                        CoverageRequirement(
                            id="", subquestion_id="", kind="", label="",
                            status=CoverageStatus.MISSING, evidence_ids=(),
                        ),
                    ).evidence_ids
                )),
                satisfaction_mode=RequirementSatisfactionMode.COMPOSITE_REQUIREMENT.value,
            ))
            recorder = current_eval_trace()
            if recorder is not None and hasattr(
                recorder, "record_fact_terminal_outcome"
            ):
                recorder.record_fact_terminal_outcome(
                    requirement_id=spec.id,
                    evidence_id="",
                    outcome="NOT_APPLICABLE",
                    reason_codes=(
                        "composite_children_covered"
                        if covered else "composite_child_requirement_missing",
                    ),
                )

        explicit_periods = sorted({
            getattr(fact, "period", "")
            for fact in typed_facts
            if _YEAR_RE.fullmatch(getattr(fact, "period", ""))
        })
        if len(explicit_periods) >= 2:
            typed_facts = [
                replace(
                    fact,
                    period=(
                        explicit_periods[0]
                        if fact.period == "historical"
                        else explicit_periods[-1]
                        if fact.period == "current"
                        else fact.period
                    ),
                ) if hasattr(fact, "period") else fact
                for fact in typed_facts
            ]

    if execution_contract is not None:
        canonical_mandatory = tuple(
            item for item in execution_contract.requirements if item.mandatory
        )
        canonical_by_id = {item.id: item for item in canonical_mandatory}
        counts: dict[str, int] = defaultdict(int)
        for item in requirements:
            counts[item.id] += 1
        duplicate_ids = tuple(sorted(
            requirement_id for requirement_id, count in counts.items()
            if count != 1 and requirement_id in canonical_by_id
        ))
        if duplicate_ids:
            raise RuntimeError(
                "duplicate canonical mandatory coverage rows: "
                + ",".join(duplicate_ids)
            )
        emitted = {item.id for item in requirements}
        explicit_operator_precoverage = {
            "category_set", "category_value_bindings",
            "common_metric_unit_period_population", "filter_condition",
        }
        for canonical in canonical_mandatory:
            if canonical.id in emitted:
                continue
            precovered = canonical.kind in explicit_operator_precoverage
            requirements.append(CoverageRequirement(
                id=canonical.id,
                subquestion_id=canonical.clause_id.replace("clause-", "sq"),
                kind=canonical.kind,
                label=canonical.subject or canonical.predicate,
                status=(
                    CoverageStatus.COVERED if precovered else CoverageStatus.MISSING
                ),
                evidence_ids=(),
                origin_id=(
                    "documented_operator_precoverage" if precovered
                    else "canonical_missing_fact"
                ),
                requirement_ids=(canonical.id,),
            ))
        fact_requirement_ids = {
            requirement_id
            for fact in typed_facts
            for requirement_id in fact.requirement_ids
        }
        canonical_answer_producing = {
            item.id for item in canonical_mandatory if item.answer_producing
        }
        requirements = [
            replace(item, status=CoverageStatus.MISSING, evidence_ids=())
            if (
                item.id in canonical_answer_producing
                and item.status is CoverageStatus.COVERED
                and item.id not in fact_requirement_ids
            )
            else item
            for item in requirements
            if item.id in canonical_by_id
        ]
        final_ids = [item.id for item in requirements]
        expected_ids = [item.id for item in canonical_mandatory]
        if len(final_ids) != len(set(final_ids)) or set(final_ids) != set(expected_ids):
            raise RuntimeError("canonical mandatory coverage row invariant failed")
        order = {requirement_id: index for index, requirement_id in enumerate(expected_ids)}
        requirements.sort(key=lambda item: order[item.id])

    covered_requirements = sum(
        item.status is CoverageStatus.COVERED for item in requirements
    )
    ratio = (
        covered_requirements / len(requirements)
        if requirements
        else (1.0 if not plan.requires_citations else 0.0)
    )
    covered_rows = sum(row.complete for row in rows)
    if list_requested and list_plan:
        complete = (
            len(rows) == list_budget
            and covered_rows == list_budget
            and all(
                item.status is CoverageStatus.COVERED
                for item in requirements
            )
        )
        if not units:
            reason = AbstentionReason.NO_AUTHORIZED_EVIDENCE
        elif len(rows) < list_budget:
            reason = AbstentionReason.REQUESTED_COUNT_NOT_COVERED
        elif (
            plan.intent is RequestIntent.LIST_WITH_ATTRIBUTES
            and covered_rows < list_budget
        ):
            reason = AbstentionReason.MISSING_ATTRIBUTE_COVERAGE
        elif not complete:
            reason = (
                AbstentionReason.PARTIAL_EVIDENCE_ONLY
                if covered_requirements and plan.allow_partial_answer
                else AbstentionReason.NO_AUTHORIZED_EVIDENCE
            )
        else:
            reason = AbstentionReason.NONE
    else:
        complete = bool(requirements) and covered_requirements == len(requirements)
        if not units:
            reason = AbstentionReason.NO_AUTHORIZED_EVIDENCE
        elif complete:
            reason = AbstentionReason.NONE
        elif covered_requirements and plan.allow_partial_answer:
            reason = AbstentionReason.PARTIAL_EVIDENCE_ONLY
        else:
            reason = AbstentionReason.NO_AUTHORIZED_EVIDENCE
    if execution_contract is not None:
        canonical_ids = execution_contract.requirement_ids
        emitted_ids = {item.id for item in requirements}
        fact_requirement_ids = {
            requirement_id
            for fact in typed_facts
            for requirement_id in fact.requirement_ids
        }
        mandatory_ids = {
            item.id for item in execution_contract.requirements if item.mandatory
        }
        if (
            emitted_ids != mandatory_ids
            or not fact_requirement_ids <= canonical_ids
            or (complete and ratio < 1.0)
        ):
            raise RuntimeError("noncanonical requirement ID emitted by evidence coverage")
    _c45_application_value = EvidenceCoverage(
        requirements=tuple(requirements),
        items=tuple(rows),
        requested_count=list_budget if list_requested else plan.requested_items,
        covered_count=(
            covered_rows
            if list_requested and list_plan
            else covered_requirements
        ),
        complete=complete,
        coverage_ratio=ratio,
        reason_code=reason,
        typed_facts=tuple(typed_facts),
    )
    _c45_observe('C44E2E2_FAMILY_06_4206ccaa33c2')
    return _c45_application_value


def repair_missing_requirement_coverage(
    plan: RequestPlan,
    units: Sequence[EvidenceUnit],
    coverage: EvidenceCoverage,
    *,
    requirement_specs: Sequence[CoverageRequirement],
    execution_contract: Optional[RequestExecutionContract],
    evidence_semantic_resolver: Optional[EvidenceSemanticResolver] = None,
    assisted_attempted_unit_ids: Optional[set[str]] = None,
) -> EvidenceCoverage:
    """Perform one extraction-only repair scoped to immutable missing reqs."""
    missing_ids = tuple(
        item.id for item in coverage.requirements
        if item.status is not CoverageStatus.COVERED
    )
    if not missing_ids or not coverage.typed_facts:
        return coverage
    missing_specs = tuple(
        item for item in requirement_specs if item.id in set(missing_ids)
    )
    if not missing_specs:
        return coverage
    fragments: list[EvidenceUnit] = []
    for unit in units:
        seen: set[str] = set()
        for claim in unit.allowed_claims:
            candidates = list(segment_text_propositions(claim))
            candidates.extend(
                (normalize_request_text(part), 0, len(part))
                for part in re.split(r"\s*;\s*", claim)
                if normalize_request_text(part)
            )
            for proposition, _start, _end in candidates:
                proposition = normalize_request_text(proposition)
                if proposition in seen or proposition == normalize_request_text(claim):
                    continue
                seen.add(proposition)
                fragments.append(replace(
                    unit,
                    canonical_text=proposition,
                    allowed_claims=(proposition,),
                ))
    if not fragments:
        return replace(
            coverage,
            targeted_repair_attempted=True,
            targeted_repair_requirement_ids=missing_ids,
        )
    repaired = build_evidence_coverage(
        plan,
        fragments,
        requirement_specs=missing_specs,
        execution_contract=execution_contract,
        evidence_semantic_resolver=evidence_semantic_resolver,
        _assisted_attempted_unit_ids=assisted_attempted_unit_ids,
    )
    repaired_by_id = {item.id: item for item in repaired.requirements}
    requirements = tuple(
        repaired_by_id.get(item.id, item)
        if repaired_by_id.get(item.id, item).status is CoverageStatus.COVERED
        else item
        for item in coverage.requirements
    )
    existing_fact_ids = {
        getattr(item, "fact_id", "") for item in coverage.typed_facts
    }
    facts = coverage.typed_facts + tuple(
        item for item in repaired.typed_facts
        if getattr(item, "fact_id", "") not in existing_fact_ids
    )
    covered_count = sum(
        item.status is CoverageStatus.COVERED for item in requirements
    )
    complete = bool(requirements) and covered_count == len(requirements)
    return EvidenceCoverage(
        requirements=requirements,
        items=coverage.items,
        requested_count=coverage.requested_count,
        covered_count=covered_count,
        complete=complete,
        coverage_ratio=(covered_count / len(requirements) if requirements else 0.0),
        reason_code=(AbstentionReason.NONE if complete else coverage.reason_code),
        typed_facts=facts,
        targeted_repair_attempted=True,
        targeted_repair_requirement_ids=missing_ids,
    )


class AdaptiveRetriever:
    """Run bounded hybrid retrieval independently for every subquestion."""

    def __init__(
        self,
        retriever: Any,
        *,
        effective_tenant: str,
        where: Optional[dict[str, str]],
        integrity_validator: Callable[[Mapping[str, Any]], bool],
        source_authorizer: Optional[Callable[[Mapping[str, Any], str, str], SourceAuthorization]] = None,
        avoid_evidence_chunk_ids: Iterable[str] = (),
        avoid_source_page_pairs: Iterable[tuple[str, int]] = (),
        evidence_semantic_resolver: Optional[EvidenceSemanticResolver] = None,
    ) -> None:
        self.retriever = retriever
        self.effective_tenant = effective_tenant
        self.where = where
        self.integrity_validator = integrity_validator
        self.source_authorizer = source_authorizer
        self.avoid_evidence_chunk_ids = frozenset(avoid_evidence_chunk_ids)
        self.avoid_source_page_pairs = frozenset(avoid_source_page_pairs)
        self.evidence_semantic_resolver = evidence_semantic_resolver

    def _is_excluded(self, hit: Mapping[str, Any]) -> bool:
        metadata = hit.get("metadata") or {}
        source_id = str(metadata.get("source_id") or "")
        page = _safe_int(metadata.get("page"))
        public_id = opaque_evidence_chunk_id(
            source_id,
            str(metadata.get("chunk_id") or hit.get("id") or ""),
            page,
        ) if source_id else ""
        raw_chunk = str(metadata.get("chunk_id") or "")
        return (
            public_id in self.avoid_evidence_chunk_ids
            or raw_chunk in self.avoid_evidence_chunk_ids
            or (
                bool(source_id)
                and page is not None
                and (source_id, page) in self.avoid_source_page_pairs
            )
        )

    def retrieve(
        self,
        plan: RequestPlan,
        *,
        execution_contract: Optional[RequestExecutionContract] = None,
        structured_rows: Sequence[StructuredRowCandidate] = (),
    ) -> AdaptiveRetrievalResult:
        all_units: list[EvidenceUnit] = []
        diagnostics: list[SubquestionRetrieval] = []
        retrieval_timed_out = False
        integrity_failures = 0
        if execution_contract is None and os.getenv(
            "PROJECT_X_CANDIDATE_MODE", "0"
        ).strip().casefold() in {"1", "true", "yes", "on"}:
            raise RuntimeError("AdaptiveRetriever requires RequestExecutionContract")
        requirement_specs = derive_evidence_requirements(plan, execution_contract)
        retrieval_tasks = _requirement_retrieval_tasks(
            plan,
            requirement_specs,
        )

        for original_subquestion_id, subquestion in retrieval_tasks:
            unit_subquestion_id = next((
                requirement.subquestion_id
                for requirement in requirement_specs
                if requirement.id == original_subquestion_id
            ), subquestion.id.split(":", 1)[0])
            trace_recorder = current_eval_trace()
            if trace_recorder is not None:
                trace_recorder.current_requirement_id = original_subquestion_id
            rewrite_started = time.monotonic()
            rewrites = build_query_rewrites(subquestion, execution_contract)
            query_rewrite_ms = (
                time.monotonic() - rewrite_started
            ) * 1000.0
            vector_started = time.monotonic()
            try:
                bounded_search = getattr(
                    self.retriever,
                    "bounded_search",
                    None,
                )
                if callable(bounded_search):
                    vector_hits = list(bounded_search(
                        rewrites[0].text,
                        k=MAX_VECTOR_RESULTS_PER_SUBQUESTION,
                        where=self.where,
                        candidate_limit=(
                            MAX_VECTOR_CANDIDATES_PER_SUBQUESTION
                        ),
                        rerank_limit=MAX_VECTOR_RERANK_PER_SUBQUESTION,
                    ) or [])
                else:
                    vector_hits = list(self.retriever.search(
                        rewrites[0].text,
                        k=MAX_CHANNEL_RESULTS,
                        where=self.where,
                    ) or [])
            except TimeoutError:
                vector_hits = []
                retrieval_timed_out = True
            except Exception:
                vector_hits = []
            vector_ms = (time.monotonic() - vector_started) * 1000.0

            lexical_started = time.monotonic()
            lexical_rankings: list[Mapping[str, Any]] = []
            for rewrite in rewrites[:3]:
                try:
                    lexical_rankings.extend(
                        self.retriever.keyword_search(
                            rewrite.text,
                            k=MAX_CHANNEL_RESULTS,
                            min_score=1,
                            where=self.where,
                        ) or []
                    )
                except TimeoutError:
                    retrieval_timed_out = True
                except Exception:
                    continue
            lexical_ms = (time.monotonic() - lexical_started) * 1000.0

            exact_started = time.monotonic()
            exact_hits: list[Mapping[str, Any]] = []
            for anchor in _select_exact_anchors(subquestion):
                try:
                    exact_hits.extend(
                        self.retriever.exact_search(
                            anchor,
                            k=MAX_CHANNEL_RESULTS,
                            where=self.where,
                        ) or []
                    )
                except TimeoutError:
                    retrieval_timed_out = True
                except Exception:
                    continue
            conjunctive_hits: list[Mapping[str, Any]] = []
            for groups in _select_conjunctive_groups(subquestion):
                try:
                    conjunctive_hits.extend(
                        self.retriever.conjunctive_search(
                            [list(group) for group in groups],
                            k=MAX_CHANNEL_RESULTS,
                            where=self.where,
                        ) or []
                    )
                except TimeoutError:
                    retrieval_timed_out = True
                except Exception:
                    continue
            exact_ms = (time.monotonic() - exact_started) * 1000.0

            fusion_started = time.monotonic()
            fused = reciprocal_rank_fuse([
                (RetrievalChannel.VECTOR, vector_hits),
                (RetrievalChannel.LEXICAL, lexical_rankings),
                (RetrievalChannel.EXACT, exact_hits),
                (RetrievalChannel.CONJUNCTIVE, conjunctive_hits),
            ])
            fused = [hit for hit in fused if not self._is_excluded(hit)]
            fusion_ms = (time.monotonic() - fusion_started) * 1000.0

            structural_hits, structural_ms, multi_hop_ms = _structural_expand(
                self.retriever,
                fused,
                profile=plan.intent,
            )
            selected_hits = reciprocal_rank_fuse([
                (RetrievalChannel.VECTOR, fused),
                (RetrievalChannel.SECTION, structural_hits),
            ])
            selected_hits = _dimension_diverse_hits(
                selected_hits,
                subquestion,
            )
            if execution_contract is not None:
                selected_hits, semantic_decisions = select_semantic_candidates(
                    selected_hits,
                    execution_contract,
                    effective_tenant=self.effective_tenant,
                    limit=MAX_EVIDENCE_PER_SUBQUESTION,
                )
            else:
                semantic_decisions = ()
            integrity_failures += sum(
                decision.source_tenant_integrity == "failed"
                for decision in semantic_decisions
            )
            selected_units: list[EvidenceUnit] = []
            selected_candidate_ids: list[str] = []
            for hit in selected_hits:
                if self._is_excluded(hit):
                    continue
                hit_units = _units_from_hit(
                    hit,
                    subquestion_id=unit_subquestion_id,
                    effective_tenant=self.effective_tenant,
                    integrity_validator=self.integrity_validator,
                    source_authorizer=self.source_authorizer,
                )
                if not hit_units:
                    # Phase 2C8 records the terminal authorization or
                    # decomposition outcome at the candidate boundary.  A
                    # missing unit is not, by itself, an integrity failure.
                    continue
                # Candidate selection has already applied its bounded
                # per-subquestion limit.  Do not let the number of structural
                # units in an earlier candidate suppress source construction
                # for a later selected candidate: that would make a
                # request-independent structural lifecycle appear selected in
                # the trace while never entering the canonical source boundary.
                # Each candidate is still structurally bounded by
                # ``MAX_STRUCTURAL_UNITS_PER_CHUNK``; this changes neither
                # retrieval depth nor model/resolver budgets.
                selected_units.extend(hit_units)
                selected_candidate_ids.append(semantic_candidate_id(hit))
            if trace_recorder is not None:
                trace_recorder.record_candidate_decisions(
                    semantic_decisions,
                    selected_candidate_ids=selected_candidate_ids,
                    selection_stage="requirement_scoped_fact_extraction",
                )
            all_units.extend(selected_units)
            diagnostics.append(SubquestionRetrieval(
                subquestion_id=unit_subquestion_id,
                query_count=len(rewrites),
                vector_candidates=len(vector_hits),
                lexical_candidates=len(lexical_rankings),
                exact_candidates=len(exact_hits) + len(conjunctive_hits),
                fused_candidates=len(fused),
                selected_evidence_ids=tuple(
                    unit.id for unit in selected_units
                ),
                query_rewrite_ms=query_rewrite_ms,
                vector_ms=vector_ms,
                lexical_ms=lexical_ms,
                exact_ms=exact_ms,
                fusion_ms=fusion_ms,
                structural_expansion_ms=structural_ms,
                multi_hop_ms=multi_hop_ms,
            ))
            if trace_recorder is not None:
                trace_recorder.current_requirement_id = None

        diagnostics = _aggregate_retrieval_diagnostics(
            diagnostics,
            plan,
        )
        units = _merge_units(all_units)
        by_subquestion = {
            subquestion.id: tuple(
                unit.id
                for unit in units
                if subquestion.id in unit.subquestion_ids
            )
            for subquestion in plan.subquestions
        }
        assisted_attempted_unit_ids: set[str] = set()
        coverage = build_evidence_coverage(
            plan,
            units,
            structured_rows=structured_rows,
            requirement_specs=requirement_specs,
            execution_contract=execution_contract,
            evidence_semantic_resolver=self.evidence_semantic_resolver,
            _assisted_attempted_unit_ids=assisted_attempted_unit_ids,
        )
        if 0 < coverage.covered_count < len(coverage.requirements):
            coverage = repair_missing_requirement_coverage(
                plan,
                units,
                coverage,
                requirement_specs=requirement_specs,
                execution_contract=execution_contract,
                evidence_semantic_resolver=self.evidence_semantic_resolver,
                assisted_attempted_unit_ids=assisted_attempted_unit_ids,
            )
        if retrieval_timed_out and not units:
            coverage = EvidenceCoverage(
                requirements=coverage.requirements,
                items=coverage.items,
                requested_count=coverage.requested_count,
                covered_count=coverage.covered_count,
                complete=False,
                coverage_ratio=coverage.coverage_ratio,
                reason_code=AbstentionReason.RETRIEVAL_TIMEOUT,
            )
        elif integrity_failures and not units:
            coverage = EvidenceCoverage(
                requirements=coverage.requirements,
                items=coverage.items,
                requested_count=coverage.requested_count,
                covered_count=coverage.covered_count,
                complete=False,
                coverage_ratio=coverage.coverage_ratio,
                reason_code=AbstentionReason.SOURCE_INTEGRITY_FAILURE,
            )
        result = AdaptiveRetrievalResult(
            evidence_units=units,
            by_subquestion=by_subquestion,
            coverage=coverage,
            diagnostics=tuple(diagnostics),
            retrieval_timed_out=retrieval_timed_out,
            integrity_failures=integrity_failures,
        )
        recorder = current_eval_trace()
        if recorder is not None:
            recorder.record_adaptive_result(result)
        _c45_application_value = result
        _c45_observe('C44E2E2_FAMILY_05_462e07fc25b7')
        return _c45_application_value
