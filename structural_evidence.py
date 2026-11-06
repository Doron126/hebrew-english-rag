"""Canonical source-only structural evidence decomposition.

This module owns the boundary from one retrieved source chunk to bounded,
auditable structural evidence units.  It deliberately has no access to the
request, semantic contract, tenant lookup, or answer expectations.  Ordinary
and operator fact constructors consume the same units and validation result.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import re
from typing import Any, Mapping, Optional, Sequence


from answer_units import GROUPED_DECIMAL_LITERAL_PATTERN


MAX_STRUCTURAL_UNIT_CHARS = 900
MAX_STRUCTURAL_UNITS_PER_CHUNK = 48

# The ingest pipeline prefixes PDF-shaped chunks with source breadcrumbs such
# as ``[report.pdf > Page 17]``.  This is source decoration, not source body.
# Keep the grammar intentionally bounded and tied to a locator: arbitrary
# bracketed prose must never be removed as if it were decoration.
_SOURCE_DECORATION_LINE_RE = re.compile(
    r"[ \t]*\[(?P<label>[^\]\r\n]{1,500}?)\s*>\s*"
    r"(?P<locator>(?:page|p\.?|עמוד|slide|שקופית)\s+\d{1,6})\][ \t]*(?:\r?\n|$)",
    re.IGNORECASE,
)
_BULLET_START_RE = re.compile(r"(?m)^\s*(?:[-*•▪◦]|\d{1,2}[.)])\s+")
_ORDERED_MARKER_ONLY_RE = re.compile(r"(?:\((?P<parenthesized>\d{1,3})\)|(?P<ordinal>\d{1,3})[.)]?)")
_ORDERED_INLINE_ITEM_RE = re.compile(
    r"(?:\((?P<parenthesized>\d{1,3})\)|(?P<ordinal>\d{1,3})[.)])"
    r"[ \t]+(?P<label>\S.*)"
)
_WORD_ORDINAL_VALUES = {
    "first": 1, "second": 2, "third": 3, "fourth": 4,
    "ראשון": 1, "ראשונה": 1, "שני": 2, "שניה": 2, "שנייה": 2,
    "שלישי": 3, "שלישית": 3, "רביעי": 4, "רביעית": 4,
}
_COLLECTION_SINGULAR = {
    "axes": "axis", "dimensions": "dimension", "sections": "section",
    "topics": "topic", "areas": "area", "themes": "theme", "parts": "part",
    "categories": "category", "strands": "strand",
    "צירים": "ציר", "ממדים": "ממד", "מימדים": "מימד", "תחומים": "תחום",
    "נושאים": "נושא", "חלקים": "חלק", "פרקים": "פרק", "היבטים": "היבט",
    "קטגוריות": "קטגוריה",
}
_SOURCE_ORGANIZATION_RE = re.compile(
    r"(?P<subject>[^.!?:;,\r\n]+?)\s+(?:is|was)\s+"
    r"(?:organized|organised|structured|divided)\s+(?:around|into)\s+"
    r"(?P<count_en>two|three|2|3)\s+(?:(?:interlocking|interrelated|main|major)\s+)?"
    r"(?P<collection_en>axes|dimensions|sections|topics|areas|themes|parts|categories|strands)"
    r"\s*[.:]?|"
    r"(?P<subject_he>[^.!?:;,\r\n]+?)\s+(?:מאורגן|מאורגנת|מחולק|מחולקת|בנוי|בנויה)\s+"
    r"(?:סביב|לפי|ל)\s*(?P<count_he>שני|שתי|שלושה|שלוש|2|3)\s+"
    r"(?P<collection_he>צירים|ממדים|מימדים|תחומים|נושאים|חלקים|פרקים|היבטים|קטגוריות)"
    r"(?:\s+(?:שלובים|משולבים|מרכזיים|עיקריים))?\s*[.:]?",
    re.I,
)
_ORGANIZATION_SUBJECT_QUALIFIER_RE = re.compile(
    r"(?<!\w)(?:not|no|never|neither|nor|without|if|unless|except|only|whether|"
    r"may|might|could|would|should|must|can|shall|perhaps|possibly|probably|apparently|"
    r"allegedly|proposed|planned|intended|expected|supposed|provided|assuming|"
    r"conditional(?:ly)?|after|before|once|upon|when|whenever|"
    r"claims?|claimed|says|said|suggests?|suggested|believed|assumed|"
    r"ו?ש?(?:אינו|אינה|אינם|אינן|איננו|איננה|אין|לא|לאו|ללא|בלי|טרם)|"
    r"ו?ש?(?:אם|אלא|כאשר|האם|בתנאי|ובלבד|בכפוף|בהנחה|במידה|במידת|ככל|רק|"
    r"לאחר|אחרי|לפני|ייתכן|יתכן|אפשר|עשוי|עשויה|עשויים|עשויות|"
    r"עלול|עלולה|עלולים|עלולות|אמור|אמורה|אמורים|אמורות|יכול|יכולה|יכולים|יכולות|צריך|צריכה|"
    r"חייב|חייבת|מומלץ|ראוי|ניתן|לכאורה|אולי|כנראה|מתוכנן|מתוכננת|"
    r"צפוי|צפויה|מוצע|מוצעת|מיועד|מיועדת|יהיה|תהיה|יהיו|נטען|נאמר|דווח))"
    r"(?!\w)", re.I,
)
_WORD_ORDINAL_ITEM_RE = re.compile(
    r"(?:(?:the\s+)?(?P<en>first|second|third|fourth)"
    r"(?:\s+(?P<en_role>axis|dimension|section|topic|area|theme|part|category|strand))?|"
    r"ו?(?:ה?(?P<he_role>ציר|ממד|מימד|תחום|נושא|חלק|פרק|היבט|קטגוריה)\s+)?"
    r"ה?(?P<he>ראשון|ראשונה|שני|שניה|שנייה|שלישי|שלישית|רביעי|רביעית))"
    r"(?P<separator>\s*[, :–—-]\s*|\s+)(?P<label>\S.*)",
    re.I | re.S,
)
_WORD_ORDINAL_PREDICATE_RE = re.compile(
    r"(?:addresses?|examines?|concerns?|focuses?|discusses?|presents?|considers?|"
    r"describes?|explores?|covers?|deals?\s+with|"
    r"עוסק(?:ת)?|בוחן|בוחנת|מצביע(?:ה)?|מתמקד(?:ת)?|מציג(?:ה)?|"
    r"מתאר(?:ת)?|נוגע(?:ת)?|דן|דנה)(?!\w)", re.I,
)
_INLINE_BULLET_MARKER_RE = re.compile(r"[•▪◦]\s*(?::\s*)?(?=\S)")
_REVERSED_BULLET_MARKER_RE = re.compile(
    r"(?P<label>[A-Za-z\u05d0-\u05ea][A-Za-z0-9\u05d0-\u05ea _-]{1,80})"
    r"\s+[•▪◦]\s*:\s*(?=\S)"
)
_TRAILING_BULLET_MARKER_RE = re.compile(
    r"(?m)^\s*(?P<label>[A-Za-z\u05d0-\u05ea]"
    r"[A-Za-z0-9\u05d0-\u05ea _-]{1,80}?)\s+[•▪◦]\s*$"
)
_STRUCTURED_RECORD_HEADER_RE = re.compile(
    r"(?im)^\s*(?P<kind>Spreadsheet row|Delimited row|Table row|Log event)\s+"
    r"(?P<number>\d{1,9})\s*:\s*$"
)
_STRUCTURED_FIELD_RE = re.compile(
    r"(?m)^\s*[-*•▪◦]\s+(?P<label>.{1,160}?):\s*(?P<value>.*?)\s*$"
)
_PIPE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_PIPE_SEPARATOR_RE = re.compile(
    r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$"
)
_SENTENCE_END_RE = re.compile(r"[.!?׃…][\"'\u05f4\u2019\u201d)\]}]*\s*$")
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?׃…])(?:[ \t\r\n]+)(?=\S)")
_VALUE_RE = re.compile(
    rf"(?<![\w])(?:{GROUPED_DECIMAL_LITERAL_PATTERN}|\d+(?:[.,]\d+)?)\s*"
    r"(?:%|אחוז(?:ים)?|percent(?:age)?|[kmgt]?w|עובדים|משרות|יחידות|"
    r"workers?|employees?|positions?|units?)?",
    re.IGNORECASE,
)
_STATUS_RE = re.compile(
    r"(?:"
    r"\b(?:status\s+)?(?:negative|normal|positive|excluded|included|"
    r"underrepresented|overrepresented|unavailable|absent)\b|"
    r"(?:סטטוס\s+)?(?:שלילי|שלילית|תקין|תקינה|חיובי|חיובית|"
    r"מוחרג|מוחרגת|נכלל|נכללה|ייצוג\s+חסר|ייצוג\s+יתר|לא\s+זמין)"
    r")",
    re.IGNORECASE,
)
_PREDICATE_RE = re.compile(
    r"(?:"
    r"\b(?:is|are|was|were|has|have|stood|amounted|estimated|reported|"
    r"represented|included|excluded|increased|decreased|rose|fell|counted|"
    r"designed|defined|means|because|due\s+to|funded|financed|sponsored|"
    r"published|authored|wrote)\b|"
    r"(?:הוא|היא|היו|היה|הייתה|עמד|עמדה|נאמד|נאמדה|מהווה|"
    r"נכלל|נכללה|לא\s+נכלל|הוחרג|עלה|עלתה|ירד|ירדה|מנה|מנתה|מטרת|"
    r"פירוש|בגלל|בשל|עקב|בוצע|בוצעה|נערך|נערכה|מומן|מומנה|מימן|מימנה|"
    r"פורסם|פורסמה|פרסם|פרסמה|נכתב|נכתבה|כתב|כתבה)"
    r")",
    re.IGNORECASE,
)
_HEADING_PREFIX_RE = re.compile(r"^\s*#{1,6}\s+")
_SECTION_HEADING_RE = re.compile(
    r"^\s*(?:chapter|section|part|פרק|סעיף)\s+\d{1,4}(?:\s*[-–—:.]\s*.+)?\s*$",
    re.IGNORECASE,
)
_LETTER_RE = re.compile(r"[A-Za-z\u05d0-\u05ea]")
_TEXTUAL_PROPOSITION_RE = re.compile(
    r"(?:"
    r"\b(?:is|are|was|were|means|defined|defines|aims?|aimed|explains?|"
    r"causes?|because|due\s+to|uses?|performed|compares?|consists|serves|"
    r"supports?|requires?|provides?)\b|"
    r"(?:מטרת|מוגדר(?:ת|ים|ות)?|מוסבר(?:ת|ים|ות)?|מתבצע(?:ת|ים|ות)?|"
    r"נועד(?:ת|ים|ות)?|בוחנ(?:ת|ים|ות)?|משוו(?:ה|ים|ות)|באמצעות|כדי|"
    r"בגלל|בשל|עקב|משום)"
    r")",
    re.IGNORECASE,
)
_NON_FACTUAL_FRAGMENT_RE = re.compile(
    r"^(?:"
    r"(?:table\s+of\s+contents|contents|list\s+of\s+"
    r"(?:tables|figures|charts)|home|next|previous|page\s+\d+)"
    r"|(?:תוכן\s+עניינים|רשימת\s+(?:טבלאות|איורים|תרשימים)|עמוד\s+\d+)"
    r")$",
    re.IGNORECASE,
)
_NAVIGATION_FRAGMENT_RE = re.compile(
    r"(?:\b(?:table\s+of\s+contents|contents|list\s+of\s+"
    r"(?:tables|figures|charts))\b|"
    r"(?:תוכן\s+עניינים|רשימת\s+(?:טבלאות|איורים|תרשימים)))"
    r".*(?:\.{4,}|…{2,})",
    re.IGNORECASE,
)
_CAPTION_ONLY_RE = re.compile(
    r"^\s*(?:\d{1,3}\s+)?(?:figure|fig\.?|table|chart|איור|טבלה|תרשים)"
    r"\s*\d{0,3}\b",
    re.IGNORECASE,
)
_STRUCTURAL_REFERENCE_ORDINAL_RE = re.compile(
    r"(?:figure|fig\.?|table|chart|איור|טבלה|תרשים)\s*"
    r"(?P<ordinal>\d{1,3})\b",
    re.IGNORECASE,
)
_SERIES_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_SERIES_VALUE_RE = re.compile(rf"(?<![\w])(?:{GROUPED_DECIMAL_LITERAL_PATTERN}|\d{{1,3}})(?![\w])")
_SOURCE_ATTRIBUTION_RE = re.compile(
    r"\(?\s*\d{1,3}\s*:\s*(?:source|מקור)\b|\b(?:source|מקור)\s*:",
    re.IGNORECASE,
)


class StructuralEvidenceType(str, Enum):
    PROSE_PROPOSITION = "prose_proposition"
    TABLE_ROW = "table_row"
    TABLE_CELL_GROUP = "table_cell_group"
    LIST_ITEM = "list_item"
    CHART_CATEGORY_RECORD = "chart_category_record"
    HEADING_CONTEXT = "heading_context"
    OTHER_BOUNDED_UNIT = "other_bounded_unit"


@dataclass(frozen=True)
class StructuralSourceSpan:
    start: int
    end: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class StructuralFieldBinding:
    role: str
    value: str
    source_span: tuple[int, int]
    structural_group_id: str
    column_role: str = ""
    field_span: Optional[tuple[int, int]] = None
    label_span: Optional[tuple[int, int]] = None
    column_letter: str = ""
    cell_coordinate: str = ""
    worksheet_row: Optional[int] = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "value": self.value,
            "source_span": list(self.source_span),
            "structural_group_id": self.structural_group_id,
            "column_role": self.column_role,
            "field_span": list(self.field_span) if self.field_span else None,
            "label_span": list(self.label_span) if self.label_span else None,
            "column_letter": self.column_letter or None,
            "cell_coordinate": self.cell_coordinate or None,
            "worksheet_row": self.worksheet_row,
        }


@dataclass(frozen=True)
class StructuralEvidenceUnit:
    unit_id: str
    source_id: str
    candidate_id: str
    chunk_id: str
    page: Optional[int]
    source_container: str
    source_spans: tuple[StructuralSourceSpan, ...]
    structural_type: StructuralEvidenceType
    local_text: str
    local_text_sha256: str
    semantic_boundaries: tuple[int, int]
    parent_structural_id: str = ""
    structural_group_id: str = ""
    row_id: str = ""
    row_label: str = ""
    list_id: str = ""
    item_id: str = ""
    category_id: str = ""
    column_roles: tuple[str, ...] = ()
    field_bindings: tuple[StructuralFieldBinding, ...] = ()
    segmentation_method: str = ""
    derivation_type: str = "source_bounded"
    structural_confidence: float = 1.0
    validation_errors: tuple[str, ...] = ()
    workbook: str = ""
    sheet: str = ""
    worksheet_row: Optional[int] = None
    header_row: Optional[int] = None
    worksheet_row_span: Optional[StructuralSourceSpan] = None
    list_ordinal: Optional[int] = None
    list_heading: str = ""
    list_heading_span: Optional[StructuralSourceSpan] = None
    list_context_span: Optional[StructuralSourceSpan] = None
    slide: Optional[int] = None

    @property
    def structurally_valid(self) -> bool:
        return not self.validation_errors

    def safe_summary(self) -> dict[str, Any]:
        return {
            "structural_unit_id": self.unit_id,
            "structural_unit_type": self.structural_type.value,
            "source_id": self.source_id,
            "candidate_id": self.candidate_id,
            "chunk_id": self.chunk_id,
            "page": self.page,
            "source_container": self.source_container,
            "source_spans": [item.as_dict() for item in self.source_spans],
            "parent_structural_id": self.parent_structural_id or None,
            "structural_group_id": self.structural_group_id or None,
            "row_id": self.row_id or None,
            "row_label": self.row_label or None,
            "list_id": self.list_id or None,
            "item_id": self.item_id or None,
            "category_id": self.category_id or None,
            "column_roles": list(self.column_roles),
            "field_bindings": [item.as_dict() for item in self.field_bindings],
            "workbook": self.workbook or None,
            "sheet": self.sheet or None,
            "worksheet_row": self.worksheet_row,
            "header_row": self.header_row,
            "worksheet_row_span": self.worksheet_row_span.as_dict() if self.worksheet_row_span else None,
            "list_ordinal": self.list_ordinal,
            "list_heading": self.list_heading or None,
            "list_heading_span": self.list_heading_span.as_dict() if self.list_heading_span else None,
            "list_context_span": self.list_context_span.as_dict() if self.list_context_span else None,
            "slide": self.slide,
            "segmentation_method": self.segmentation_method,
            "derivation_type": self.derivation_type,
            "structural_confidence": round(self.structural_confidence, 4),
            "semantic_proposition_hash": self.local_text_sha256,
            "semantic_proposition_preview": _bounded_preview(self.local_text),
            "structural_validation_result": (
                "valid" if self.structurally_valid else "rejected"
            ),
            "structural_rejection_reasons": list(self.validation_errors),
        }


@dataclass(frozen=True)
class StructuralEntryResult:
    """The single source-decoration-to-structural-unit boundary.

    Spans always refer to the original retrieved chunk.  ``factual_body_text``
    is deliberately retained only in memory for downstream coarse hints; the
    trace summary records spans and bounded metadata, never the full chunk.
    """

    status: str
    source_id: str
    candidate_id: str
    source_decoration_detected: bool
    source_decoration_kind: str
    source_decoration_span: Optional[StructuralSourceSpan]
    factual_body_span: Optional[StructuralSourceSpan]
    factual_body_text: str
    coarse_structural_type: str
    canonical_decomposition_attempted: bool
    structural_units: tuple[StructuralEvidenceUnit, ...]
    rejection_reason: str = ""

    @property
    def accepted(self) -> bool:
        return self.status == "accepted"

    def trace_summary(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "structural_entry_status": self.status,
            "source_decoration_detected": self.source_decoration_detected,
            "source_decoration_kind": self.source_decoration_kind or None,
            "source_decoration_span": (
                self.source_decoration_span.as_dict()
                if self.source_decoration_span else None
            ),
            "body_span": (
                self.factual_body_span.as_dict()
                if self.factual_body_span else None
            ),
            "body_length": len(self.factual_body_text),
            "coarse_structural_type": self.coarse_structural_type,
            "canonical_decomposition_attempted": self.canonical_decomposition_attempted,
            "decomposition_status": (
                "units_created" if self.structural_units else "rejected"
            ),
            "decomposition_reason": (
                "structural_units_created" if self.structural_units
                else self.rejection_reason
            ),
            "candidate_lifecycle_terminal_state": (
                "structural_units_created" if self.structural_units
                else "structural_rejected"
            ),
        }


@dataclass(frozen=True)
class ScalarSpanAudit:
    """One bounded, source-only accounting decision for a scalar literal."""

    source_span: tuple[int, int]
    source_span_sha256: str
    status: str
    reason_code: str
    structural_unit_id: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_span": list(self.source_span),
            "source_span_sha256": self.source_span_sha256,
            "status": self.status,
            "reason_code": self.reason_code,
            "structural_unit_id": self.structural_unit_id or None,
        }


def _bounded_preview(value: str, maximum: int = 240) -> str:
    normalized = " ".join(str(value or "").split())
    return normalized if len(normalized) <= maximum else normalized[:maximum]


def _trimmed_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _balanced_grouping(text: str) -> bool:
    """Reject candidate boundaries that cut through a source grouping."""
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for character in text:
        if character in "([{":
            stack.append(character)
        elif character in pairs:
            if not stack or stack.pop() != pairs[character]:
                return False
    return not stack


def _unit_identity(
    candidate_id: str,
    structural_type: StructuralEvidenceType,
    start: int,
    end: int,
    text: str,
) -> str:
    payload = "\x1f".join((
        candidate_id,
        structural_type.value,
        str(start),
        str(end),
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
    ))
    return "seu-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _parent_identity(candidate_id: str, kind: str, ordinal: int) -> str:
    payload = f"{candidate_id}\x1f{kind}\x1f{ordinal}"
    return "struct-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _row_label(text: str) -> tuple[str, tuple[int, int]]:
    raw = text.strip().strip("|")
    prefix_match = re.match(
        r"\s*(?:[-*•▪◦]|\d{1,2}[.)])?\s*(?P<label>.+?)"
        r"(?=\s+(?:\d|[-+]?[.,]\d)|\s+" + _STATUS_RE.pattern + r"|\s*\||$)",
        raw,
        re.IGNORECASE,
    )
    label = ""
    if prefix_match:
        label = " ".join(prefix_match.group("label").strip(" |:;-–—").split())
    if not label:
        first_cell = raw.split("|", 1)[0]
        first_value = _VALUE_RE.search(first_cell)
        label = " ".join(
            first_cell[:first_value.start() if first_value else len(first_cell)]
            .strip(" |:;-–—").split()
        )
    if not label or len(_LETTER_RE.findall(label)) < 2:
        return "", (0, 0)
    relative = text.casefold().find(label.casefold())
    return label, (
        (relative, relative + len(label)) if relative >= 0 else (0, 0)
    )


def _column_names(header: str) -> tuple[str, ...]:
    cells = [" ".join(item.split()) for item in header.strip().strip("|").split("|")]
    return tuple(item for item in cells if item)


def _structured_record_layout(text: str) -> tuple[str, tuple[str, ...]]:
    """Return a literal entity label and field roles for one ingested row."""
    fields = tuple(_STRUCTURED_FIELD_RE.finditer(text))
    column_roles = tuple(
        " ".join(match.group("label").split()) for match in fields
    )
    identity_fields = tuple(match for match in fields if re.search(
        r"\b(?:(?:product|item)\s*(?:name|code)|model|sku|part\s*(?:number|no\.?))\b|"
        r"(?:שם\s*(?:המוצר|מוצר|הפריט|פריט)|דגם|מק[\"״']?ט)",
        match.group("label"), re.IGNORECASE,
    ))
    for match in (*identity_fields, *fields):
        value = " ".join(match.group("value").strip().split())
        if (
            value
            and (match in identity_fields or len(_LETTER_RE.findall(value)) >= 2)
            and not _STATUS_RE.fullmatch(value)
            and (match in identity_fields or not _VALUE_RE.fullmatch(value))
        ):
            return value, column_roles
    return "", column_roles


def _structured_field_bindings(
    text: str, *, group_id: str, worksheet_row: Optional[int],
) -> tuple[StructuralFieldBinding, ...]:
    """Bind scalars to their explicit labeled field, never numeric ordinal."""
    fields = tuple(_STRUCTURED_FIELD_RE.finditer(text))
    owners = []
    for field in fields:
        label = " ".join(field.group("label").split())
        letter = re.search(r"\(([A-Z]{1,3})\)\s*$", label)
        owners.append((field, label, letter.group(1).upper() if letter else ""))
    output = []
    for field, label, letter in owners:
        common = dict(
            structural_group_id=group_id, column_role=label,
            field_span=field.span(), label_span=field.span("label"),
            column_letter=letter,
            cell_coordinate=f"{letter}{worksheet_row}" if letter and worksheet_row else "",
            worksheet_row=worksheet_row,
        )
        output.append(StructuralFieldBinding(
            role="field", value=field.group("value"),
            source_span=field.span("value"), **common,
        ))
        base_label = re.sub(r"\s*\([A-Za-z]{1,3}\)\s*$", "", label).casefold()
        if sum(re.sub(r"\s*\([A-Za-z]{1,3}\)\s*$", "", other_label).casefold() == base_label
               or bool(letter and other_letter == letter)
               for _, other_label, other_letter in owners) != 1:
            continue
        # Product identifiers and descriptive specifications are text fields,
        # even when their content contains or consists entirely of digits.
        if re.search(
            r"\b(?:description|model|sku|serial|identifier|product\s*(?:name|code)|"
            r"part\s*(?:number|no\.?))\b|(?:תיאור|תאור|דגם|מק[\"״']?ט|קוד|מספר\s*סידורי)",
            label, re.IGNORECASE,
        ):
            continue
        value = field.group("value")
        value_offset = field.start("value")
        cached = re.search(r";\s*Cached value:\s*(?P<scalar>[^;]+)$", value, re.IGNORECASE)
        if cached:
            value_offset += cached.start("scalar")
            value = cached.group("scalar")
        numeric = tuple(_VALUE_RE.finditer(value))
        if len(numeric) != 1:
            continue
        scalar = numeric[0]
        prefix, suffix = value[:scalar.start()], value[scalar.end():]
        if not re.fullmatch(r"[\s₪$€£]*", prefix) or not re.fullmatch(
            r"\s*(?:[₪$€£]|[A-Za-z\u05d0-\u05ea'׳\"״]{1,24})?\s*", suffix,
        ):
            continue
        output.append(StructuralFieldBinding(
            role="value", value=scalar.group(0).strip(),
            source_span=(value_offset + scalar.start(), value_offset + scalar.end()),
            **common,
        ))
    return tuple(output)


def _field_bindings(
    text: str,
    *,
    group_id: str,
    label: str,
    label_span: tuple[int, int],
    column_roles: Sequence[str] = (),
    worksheet_row: Optional[int] = None,
) -> tuple[StructuralFieldBinding, ...]:
    output: list[StructuralFieldBinding] = []
    if label and label_span != (0, 0):
        output.append(StructuralFieldBinding(
            role="entity",
            value=label,
            source_span=label_span,
            structural_group_id=group_id,
            column_role=(column_roles[0] if column_roles else "row_label"),
        ))
    structured = bool(column_roles and _STRUCTURED_FIELD_RE.search(text))
    if structured:
        owned_fields = _structured_field_bindings(text, group_id=group_id, worksheet_row=worksheet_row)
        if output and output[0].role == "entity":
            entity_owner = next((item for item in owned_fields if item.role == "field"
                                 and item.source_span[0] <= label_span[0] < label_span[1] <= item.source_span[1]), None)
            if entity_owner:
                output[0] = replace(entity_owner, role="entity", value=label, source_span=label_span)
        output.extend(owned_fields)
    for index, match in enumerate(() if structured else _VALUE_RE.finditer(text), start=1):
        value = match.group(0).strip()
        if not value:
            continue
        output.append(StructuralFieldBinding(
            role="value",
            value=value,
            source_span=match.span(),
            structural_group_id=group_id,
            column_role=(
                column_roles[index]
                if index < len(column_roles) else f"value_{index}"
            ),
        ))
    # A figure/table/chart ordinal identifies source layout; it is not a
    # measured scalar.  Retain its exact source-local span so fact construction
    # can exclude the literal through structural provenance, rather than from
    # request semantics or an ad-hoc text rewrite.
    for match in _STRUCTURAL_REFERENCE_ORDINAL_RE.finditer(text):
        output.append(StructuralFieldBinding(
            role="structural_reference",
            value=match.group("ordinal"),
            source_span=match.span("ordinal"),
            structural_group_id=group_id,
            column_role="non_factual_ordinal",
        ))
    for match in _STATUS_RE.finditer(text):
        output.append(StructuralFieldBinding(
            role="status",
            value=match.group(0).strip(),
            source_span=match.span(),
            structural_group_id=group_id,
            column_role=(column_roles[-1] if column_roles else "status"),
        ))
    return tuple(output)


def _validation_errors(
    structural_type: StructuralEvidenceType,
    local_text: str,
    source_spans: Sequence[StructuralSourceSpan],
    *,
    row_id: str = "",
    row_label: str = "",
    list_id: str = "",
    item_id: str = "",
    category_id: str = "",
    group_id: str = "",
    fields: Sequence[StructuralFieldBinding] = (),
) -> tuple[str, ...]:
    reasons: list[str] = []
    if not local_text or len(local_text) > MAX_STRUCTURAL_UNIT_CHARS:
        reasons.append("proposition_boundary_unresolved")
    if not source_spans or any(item.end <= item.start for item in source_spans):
        reasons.append("structural_parent_missing")
    if structural_type is StructuralEvidenceType.TABLE_ROW:
        if not row_id or not row_label:
            reasons.append("table_row_boundary_unresolved")
        if not group_id:
            reasons.append("structural_parent_missing")
    elif structural_type is StructuralEvidenceType.LIST_ITEM:
        if not list_id or not item_id:
            reasons.append("list_item_boundary_unresolved")
    elif structural_type is StructuralEvidenceType.CHART_CATEGORY_RECORD:
        if not category_id or not row_label:
            reasons.append("chart_category_boundary_unresolved")
    if group_id and any(
        item.structural_group_id != group_id for item in fields
    ):
        reasons.append("structural_argument_mismatch")
    return tuple(dict.fromkeys(reasons))


def _make_unit(
    *,
    source_text: str,
    start: int,
    end: int,
    structural_type: StructuralEvidenceType,
    source_id: str,
    candidate_id: str,
    chunk_id: str,
    page: Optional[int],
    source_container: str,
    parent_id: str = "",
    row_id: str = "",
    row_label: str = "",
    list_id: str = "",
    item_id: str = "",
    category_id: str = "",
    column_roles: Sequence[str] = (),
    segmentation_method: str,
    derivation_type: str,
    confidence: float,
    workbook: str = "",
    sheet: str = "",
    worksheet_row: Optional[int] = None,
    header_row: Optional[int] = None,
) -> StructuralEvidenceUnit:
    start, end = _trimmed_bounds(source_text, start, end)
    local_text = source_text[start:end]
    unit_id = _unit_identity(
        candidate_id, structural_type, start, end, local_text
    )
    group_id = row_id or item_id or category_id or unit_id
    label, label_span = (
        (row_label, (local_text.casefold().find(row_label.casefold()),
                     local_text.casefold().find(row_label.casefold()) + len(row_label)))
        if row_label and local_text.casefold().find(row_label.casefold()) >= 0
        else _row_label(local_text)
        if structural_type in {
            StructuralEvidenceType.TABLE_ROW,
            StructuralEvidenceType.CHART_CATEGORY_RECORD,
        }
        else ("", (0, 0))
    )
    if not row_label:
        row_label = label
    source_span = StructuralSourceSpan(
        start,
        end,
        hashlib.sha256(local_text.encode("utf-8")).hexdigest(),
    )
    fields = _field_bindings(
        local_text,
        group_id=group_id,
        label=row_label,
        label_span=label_span,
        column_roles=column_roles,
        worksheet_row=worksheet_row,
    )
    errors = _validation_errors(
        structural_type,
        local_text,
        (source_span,),
        row_id=row_id,
        row_label=row_label,
        list_id=list_id,
        item_id=item_id,
        category_id=category_id,
        group_id=group_id,
        fields=fields,
    )
    return StructuralEvidenceUnit(
        unit_id=unit_id,
        source_id=source_id,
        candidate_id=candidate_id,
        chunk_id=chunk_id,
        page=page,
        source_container=source_container,
        source_spans=(source_span,),
        structural_type=structural_type,
        local_text=local_text,
        local_text_sha256=source_span.sha256,
        semantic_boundaries=(0, len(local_text)),
        parent_structural_id=parent_id,
        structural_group_id=group_id,
        row_id=row_id,
        row_label=row_label,
        list_id=list_id,
        item_id=item_id,
        category_id=category_id,
        column_roles=tuple(column_roles),
        field_bindings=fields,
        segmentation_method=segmentation_method,
        derivation_type=derivation_type,
        structural_confidence=max(0.0, min(1.0, confidence)),
        validation_errors=errors,
        workbook=workbook,
        sheet=sheet,
        worksheet_row=worksheet_row,
        header_row=header_row,
    )


def _make_composite_chart_category_unit(
    *,
    source_text: str,
    metric_span: tuple[int, int],
    value_span: tuple[int, int],
    category_span: tuple[int, int],
    source_id: str,
    candidate_id: str,
    chunk_id: str,
    page: Optional[int],
    source_container: str,
    parent_id: str,
    ordinal: int,
) -> StructuralEvidenceUnit:
    """Build one explicitly composite chart record from proven local spans.

    Flattened PDF charts often serialize all values followed by all categories.
    A value/category record is therefore not contiguous, but it is still
    source-local when a bounded cardinality-preserving sequence proves the
    pairing.  The rendered text names the composition explicitly and every
    component retains its original source span.
    """
    fragments = (metric_span, value_span, category_span)
    values = [source_text[start:end].strip() for start, end in fragments]
    metric, value, category = values
    local_text = " … ".join(values)
    unit_id = _unit_identity(
        candidate_id,
        StructuralEvidenceType.CHART_CATEGORY_RECORD,
        metric_span[0],
        category_span[1],
        local_text,
    )
    category_id = _parent_identity(candidate_id, "category", ordinal)
    local_value_start = len(metric) + 3
    local_category_start = local_value_start + len(value) + 3
    source_spans = tuple(
        StructuralSourceSpan(
            start, end,
            hashlib.sha256(source_text[start:end].encode("utf-8")).hexdigest(),
        )
        for start, end in fragments
    )
    fields = (
        StructuralFieldBinding(
            role="entity", value=metric, source_span=(0, len(metric)),
            structural_group_id=category_id, column_role="metric",
        ),
        StructuralFieldBinding(
            role="value", value=value,
            source_span=(local_value_start, local_value_start + len(value)),
            structural_group_id=category_id, column_role="value",
        ),
        StructuralFieldBinding(
            role="category", value=category,
            source_span=(
                local_category_start,
                local_category_start + len(category),
            ),
            structural_group_id=category_id, column_role="category",
        ),
    )
    errors = _validation_errors(
        StructuralEvidenceType.CHART_CATEGORY_RECORD,
        local_text,
        source_spans,
        row_label=category,
        category_id=category_id,
        group_id=category_id,
        fields=fields,
    )
    return StructuralEvidenceUnit(
        unit_id=unit_id,
        source_id=source_id,
        candidate_id=candidate_id,
        chunk_id=chunk_id,
        page=page,
        source_container=source_container,
        source_spans=source_spans,
        structural_type=StructuralEvidenceType.CHART_CATEGORY_RECORD,
        local_text=local_text,
        local_text_sha256=hashlib.sha256(local_text.encode("utf-8")).hexdigest(),
        semantic_boundaries=(0, len(local_text)),
        parent_structural_id=parent_id,
        structural_group_id=category_id,
        category_id=category_id,
        row_label=category,
        column_roles=("category", "value"),
        field_bindings=fields,
        segmentation_method="bounded_flattened_chart_series",
        derivation_type="explicit_source_chart_value_category_pair",
        structural_confidence=1.0,
        validation_errors=errors,
    )


def _contiguous_match_groups(
    matches: Sequence[re.Match[str]],
    source_text: str,
) -> tuple[tuple[re.Match[str], ...], ...]:
    groups: list[list[re.Match[str]]] = []
    for match in matches:
        if not groups or source_text[groups[-1][-1].end():match.start()].strip():
            groups.append([match])
        else:
            groups[-1].append(match)
    return tuple(tuple(group) for group in groups)


def _caption_metric_span(
    source_text: str,
    *,
    body_start: int,
    before: int,
) -> Optional[tuple[int, int]]:
    """Return a bounded metric label from a caption line, when explicit."""
    for line_start, line_end in _line_spans(source_text, body_start):
        if line_start >= before:
            break
        line = source_text[line_start:line_end]
        match = _CAPTION_ONLY_RE.match(line)
        if match is None:
            continue
        start = line_start + match.end()
        end = min(line_end, before)
        source_marker = _SOURCE_ATTRIBUTION_RE.search(source_text, start, end)
        if source_marker is not None:
            end = source_marker.start()
        start, end = _trimmed_bounds(source_text, start, end)
        if (
            start < end
            and len(_LETTER_RE.findall(source_text[start:end])) >= 8
        ):
            return start, end
    return None


def _flattened_chart_series_units(
    source_text: str,
    *,
    source_id: str,
    candidate_id: str,
    chunk_id: str,
    page: Optional[int],
    source_container: str,
    body_start: int,
) -> tuple[StructuralEvidenceUnit, ...]:
    """Recover a chart only when source order proves every value/category pair.

    This deliberately rejects an unequal, interrupted, or unlabeled series.
    It is structural parsing only: metric text, scalar literals, and categories
    are all copied from the source and no request terminology is consulted.
    """
    body = source_text[body_start:]
    if not any(_CAPTION_ONLY_RE.match(source_text[start:end]) for start, end in _line_spans(source_text, body_start)):
        return ()
    years = _contiguous_match_groups(
        tuple(_SERIES_YEAR_RE.finditer(body)), body)
    if not years:
        return ()
    year_group = years[-1]
    if not 2 <= len(year_group) <= MAX_STRUCTURAL_UNITS_PER_CHUNK:
        return ()
    year_start = body_start + year_group[0].start()
    year_end = body_start + year_group[-1].end()
    trailing = source_text[year_end:]
    if re.search(r"\d", trailing) or len(trailing.strip()) > 48:
        return ()
    values = _contiguous_match_groups(
        tuple(_SERIES_VALUE_RE.finditer(source_text, body_start, year_start)),
        source_text,
    )
    if not values:
        return ()
    value_group = values[-1]
    if len(value_group) != len(year_group):
        return ()
    if source_text[value_group[-1].end():year_start].strip():
        return ()
    metric_span = _caption_metric_span(
        source_text, body_start=body_start, before=value_group[0].start(),
    )
    if metric_span is None:
        return ()
    parent = _parent_identity(candidate_id, "chart", 1)
    units = tuple(
        _make_composite_chart_category_unit(
            source_text=source_text,
            metric_span=metric_span,
            value_span=(value.start(), value.end()),
            category_span=(
                body_start + year.start(), body_start + year.end(),
            ),
            source_id=source_id,
            candidate_id=candidate_id,
            chunk_id=chunk_id,
            page=page,
            source_container=source_container,
            parent_id=parent,
            ordinal=ordinal,
        )
        for ordinal, (value, year) in enumerate(zip(value_group, year_group), start=1)
    )
    return tuple(unit for unit in units if unit.structurally_valid)


def _caption_has_source_local_data_shape(
    body: str,
    units: Sequence[StructuralEvidenceUnit],
) -> bool:
    """Discriminate data-bearing caption bodies from title-only decoration."""
    if any(
        unit.segmentation_method == "bounded_flattened_chart_series"
        for unit in units
    ):
        return True
    list_items = [
        unit for unit in units
        if unit.structural_type is StructuralEvidenceType.LIST_ITEM
    ]
    if len(list_items) >= 2 and all(
        len(_LETTER_RE.findall(unit.local_text)) >= 2 for unit in list_items
    ):
        return True
    percent_scalars = re.findall(r"\d+(?:[.,]\d+)?\s*(?:%|אחוז(?:ים)?)", body)
    years = _SERIES_YEAR_RE.findall(body)
    return (
        len(percent_scalars) >= 2
        and len(years) >= 2
        and any(
            unit.structural_type is StructuralEvidenceType.PROSE_PROPOSITION
            for unit in units
        )
    )


def _line_spans(text: str, start: int = 0) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    for match in re.finditer(r"[^\r\n]+", text[start:]):
        begin, end = _trimmed_bounds(
            text,
            start + match.start(),
            start + match.end(),
        )
        if begin < end:
            output.append((begin, end))
    return output


def _delimited_record_spans(text: str, start: int) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    cursor = start
    for match in re.finditer(r"\s*[;；]\s*", text[start:]):
        end = start + match.start()
        begin, bounded_end = _trimmed_bounds(text, cursor, end)
        if begin < bounded_end:
            output.append((begin, bounded_end))
        cursor = start + match.end()
    begin, end = _trimmed_bounds(text, cursor, len(text))
    if begin < end:
        output.append((begin, end))
    return output


def _status_record_spans(text: str, start: int) -> list[tuple[int, int]]:
    matches = list(_STATUS_RE.finditer(text, start))
    if len(matches) < 2:
        return []
    output: list[tuple[int, int]] = []
    cursor = start
    for match in matches:
        begin, end = _trimmed_bounds(text, cursor, match.end())
        if begin < end:
            output.append((begin, end))
        cursor = match.end()
    tail_start, tail_end = _trimmed_bounds(text, cursor, len(text))
    if tail_start < tail_end:
        # A trailing fragment without a status is structurally ambiguous and
        # must not be attached to the preceding row.
        return []
    return output


def _looks_heading(text: str) -> bool:
    value = " ".join(text.split())
    return bool(
        _HEADING_PREFIX_RE.match(value)
        or _SECTION_HEADING_RE.match(value)
        or (
            1 <= len(value.split()) <= 9
            and len(value) <= 120
            and not _VALUE_RE.search(value)
            and not _STATUS_RE.search(value)
            and not _PREDICATE_RE.search(value)
            and not _SENTENCE_END_RE.search(value)
            and len(_LETTER_RE.findall(value)) >= 4
        )
    )


def _factual_line(text: str) -> bool:
    return bool(
        len(_LETTER_RE.findall(text)) >= 3
        and (_VALUE_RE.search(text) or _STATUS_RE.search(text) or _PREDICATE_RE.search(text))
    )


def _bounded_coherent_proposition(text: str) -> bool:
    """Decide proposition *shape* without deciding request compatibility.

    Numeric/status propositions remain eligible exactly as before.  Textual
    propositions are admitted only when their local source boundary is clear
    and they contain a bounded statement form; headings and navigation remain
    context only.  Relation matching is intentionally downstream.
    """
    value = " ".join(str(text or "").split())
    if not value or len(value) > MAX_STRUCTURAL_UNIT_CHARS:
        return False
    if len(_LETTER_RE.findall(value)) < 5 or _looks_heading(value):
        return False
    if _NON_FACTUAL_FRAGMENT_RE.fullmatch(value.rstrip(".!?׃…")):
        return False
    if _factual_line(value):
        return True
    words = re.findall(r"[A-Za-z\u05d0-\u05ea]+", value)
    if len(words) < 3 or not _balanced_grouping(value):
        return False
    return bool(_SENTENCE_END_RE.search(value) or _TEXTUAL_PROPOSITION_RE.search(value))


def _source_decoration_bounds(text: str) -> tuple[Optional[tuple[int, int]], str]:
    """Return only leading, grammar-proven ingest decoration bounds."""
    cursor = 0
    end = 0
    presentation = False
    while cursor < len(text):
        match = _SOURCE_DECORATION_LINE_RE.match(text, cursor)
        if not match:
            break
        end = match.end()
        presentation = presentation or bool(re.match(
            r"(?:slide|שקופית)\b", match.group("locator"), re.IGNORECASE,
        ))
        cursor = end
    if not end:
        return None, ""
    return (0, end), (
        "presentation_slide_breadcrumb" if presentation else "pdf_page_breadcrumb"
    )


def _coarse_structural_hint(metadata: Mapping[str, Any], body: str) -> str:
    hint = str(
        metadata.get("structural_type")
        or metadata.get("source_unit_type")
        or metadata.get("unit_type")
        or ""
    ).casefold()
    if hint:
        return hint
    if _PIPE_ROW_RE.search(body):
        return "table"
    if _BULLET_START_RE.search(body):
        return "list"
    return "body"


def _header_footer_only_body(body: str) -> bool:
    lines = [" ".join(line.split()) for line in body.splitlines() if line.strip()]
    # A heading can provide harmless context for a following proposition.  It
    # is header/footer-only only when it is the entire bounded body.
    if len(lines) != 1:
        return False
    value = lines[0]
    if not value:
        return False
    return _looks_heading(value) or _NON_FACTUAL_FRAGMENT_RE.fullmatch(
        value.rstrip(".!?׃…")
    ) is not None


def _unsupported_source_structure(body: str) -> bool:
    """Reject only source-only navigation/caption forms, never fact prose."""
    value = " ".join(body.split())
    if _NAVIGATION_FRAGMENT_RE.search(value):
        return True
    if re.match(
        r"^(?:spreadsheet row|delimited row|table row|log event)\s+\d{1,9}\s*:",
        value,
        re.IGNORECASE,
    ):
        return False
    return bool(
        _CAPTION_ONLY_RE.match(value)
        and not _PREDICATE_RE.search(value)
        and not _STATUS_RE.search(value)
    )


def _predicate_clause_spans(
    source_text: str,
    start: int,
    end: int,
) -> list[tuple[int, int]]:
    """Recover repeated proposition boundaries when punctuation was lost."""
    local = source_text[start:end].rstrip()
    predicates = list(_PREDICATE_RE.finditer(local))
    if len(predicates) < 2:
        return []
    boundaries = [0]
    for previous, current in zip(predicates, predicates[1:]):
        values = [
            match for match in _VALUE_RE.finditer(local, previous.end(), current.start())
            if match.group(0).strip()
        ]
        if not values:
            return []
        trailing = local[values[-1].end():current.start()]
        if not re.sub(r"[\s,;:()\[\]{}\-–—]", "", trailing):
            # ``55 MW was South Port`` is a value-before-predicate row in a
            # narrative distribution.  The value belongs to the upcoming
            # predicate, so splitting after it would create two wrong units.
            return []
        boundary = values[-1].end()
        while boundary < len(local) and local[boundary].isspace():
            boundary += 1
        if boundary <= boundaries[-1] or boundary >= current.start():
            return []
        boundaries.append(boundary)
    boundaries.append(len(local))
    spans = [
        _trimmed_bounds(source_text, start + left, start + right)
        for left, right in zip(boundaries, boundaries[1:])
    ]
    return spans if all(
        begin < bounded_end
        and _balanced_grouping(source_text[begin:bounded_end])
        and _factual_line(source_text[begin:bounded_end])
        for begin, bounded_end in spans
    ) else []


def _list_item_end_before_trailing_heading(
    source_text: str,
    start: int,
    end: int,
) -> int:
    """Exclude a blank-line heading between two explicit list items."""
    local = source_text[start:end].rstrip()
    paragraphs = list(re.finditer(
        r"\S(?:.*?\S)?(?=(?:\r?\n){2,}|\Z)",
        local,
        re.DOTALL,
    ))
    if len(paragraphs) < 2:
        return end
    trailing = paragraphs[-1]
    trailing_text = trailing.group(0).strip()
    if _looks_heading(trailing_text) and not _BULLET_START_RE.match(
        trailing_text
    ):
        return start + trailing.start()
    return end


def ordered_list_item_text(unit: StructuralEvidenceUnit) -> str:
    """Return the literal item body from an explicitly numbered source unit."""
    if (
        unit.structural_type is not StructuralEvidenceType.LIST_ITEM
        or type(unit.list_ordinal) is not int
    ):
        return ""
    first_line = unit.local_text.splitlines()[0]
    inline = _ORDERED_INLINE_ITEM_RE.fullmatch(first_line)
    if inline:
        return unit.local_text[inline.start("label"):].strip()
    if _ORDERED_MARKER_ONLY_RE.fullmatch(first_line):
        return unit.local_text[len(first_line):].strip()
    word = _WORD_ORDINAL_ITEM_RE.fullmatch(unit.local_text)
    if word:
        return word["label"].strip()
    return ""


def _source_sentence_spans(text: str, start: int = 0) -> list[tuple[int, int]]:
    """Retain original paragraph/sentence boundaries, including wrapped lines."""
    spans = []
    for paragraph in re.finditer(r"\S(?:.*?\S)?(?=(?:\r?\n){2,}|\Z)", text[start:], re.S):
        left, right = start + paragraph.start(), start + paragraph.end()
        cursor = left
        for boundary in _SENTENCE_BREAK_RE.finditer(text, left, right):
            spans.append(_trimmed_bounds(text, cursor, boundary.start()))
            cursor = boundary.end()
        spans.append(_trimmed_bounds(text, cursor, right))
    return [(left, right) for left, right in spans if left < right]


def word_ordinal_list_spans(source_text: str, body_start: int = 0):
    """Prove a declared two/three-item textual sequence from original spans.

    Every item is one complete source sentence/paragraph. An intervening peer,
    missing/reordered ordinal, or continuation after the final sentence in its
    paragraph makes the sequence unavailable rather than silently truncated.
    """
    sentences = _source_sentence_spans(source_text, body_start)
    groups = []
    for index, (heading_start, heading_end) in enumerate(sentences):
        introduction = _SOURCE_ORGANIZATION_RE.fullmatch(source_text[heading_start:heading_end])
        if introduction is None:
            continue
        subject = introduction["subject"] or introduction["subject_he"]
        # The free-form subject is not an assertion parser: a Hebrew negator
        # or an outer condition/modal can otherwise be swallowed into it.
        # Such qualification is not represented by positive has_members facts.
        if _ORGANIZATION_SUBJECT_QUALIFIER_RE.search(subject):
            continue
        count = {"two": 2, "three": 3, "שני": 2, "שתי": 2,
                 "שלושה": 3, "שלוש": 3, "2": 2, "3": 3}[
                     (introduction["count_en"] or introduction["count_he"]).casefold()]
        collection = (introduction["collection_en"] or introduction["collection_he"]).casefold()
        entries = []
        for offset in range(1, count + 1):
            if index + offset >= len(sentences):
                break
            start, end = sentences[index + offset]
            marker = _WORD_ORDINAL_ITEM_RE.fullmatch(source_text[start:end])
            if marker is None:
                break
            ordinal = _WORD_ORDINAL_VALUES[(marker["en"] or marker["he"]).casefold()]
            role = marker["en_role"] or marker["he_role"]
            if ordinal != offset or role and role.casefold() != _COLLECTION_SINGULAR[collection]:
                break
            # Without an explicit role or enumerating punctuation, a pronoun
            # ordinal must directly govern a predicate, not a peer owner noun.
            if (not role and not marker["separator"].strip()
                    and not _WORD_ORDINAL_PREDICATE_RE.match(marker["label"])):
                break
            label_start = start + marker.start("label")
            if not _LETTER_RE.search(source_text[label_start:end]) or end - start > MAX_STRUCTURAL_UNIT_CHARS:
                break
            entries.append((ordinal, start, end, label_start, end))
        if len(entries) != count:
            continue
        # Do not omit a further ordinal, condition, exception or other sentence
        # continuing the final item's stored paragraph.
        next_index = index + count + 1
        if next_index < len(sentences):
            next_start, next_end = sentences[next_index]
            gap = source_text[entries[-1][2]:next_start]
            if (_WORD_ORDINAL_ITEM_RE.fullmatch(source_text[next_start:next_end])
                    or not re.search(r"(?:\r?\n){2,}", gap)):
                continue
        groups.append((heading_start, heading_end, tuple(entries)))
    return tuple(groups)


def source_ordered_list_owner_binding(
    structural: StructuralEvidenceUnit, source_text: str,
) -> Optional[tuple[str, StructuralSourceSpan]]:
    """Recover the literal governing subject from one proved organization list.

    The subject is never taken from item text, surrounding topic mentions,
    source metadata or a request. Its span belongs to the actual introduction.
    """
    heading = structural.list_heading_span
    context = structural.list_context_span
    if (structural.structural_type is not StructuralEvidenceType.LIST_ITEM
            or len(structural.source_spans) != 1 or heading is None or context is None):
        return None
    item = structural.source_spans[0]
    for start, end, entries in word_ordinal_list_spans(source_text):
        if (heading.start != start or heading.end != end
                or source_text[start:end] != structural.list_heading
                or heading.sha256 != hashlib.sha256(source_text[start:end].encode()).hexdigest()
                or context.start != start or context.end != entries[0][1]
                or context.sha256 != hashlib.sha256(source_text[start:context.end].encode()).hexdigest()
                or not any(ordinal == structural.list_ordinal and left == item.start and right == item.end
                           for ordinal, left, right, _label_start, _label_end in entries)):
            continue
        introduction = _SOURCE_ORGANIZATION_RE.fullmatch(source_text[start:end])
        assert introduction is not None
        group = "subject" if introduction["subject"] is not None else "subject_he"
        left, right = start + introduction.start(group), start + introduction.end(group)
        owner = source_text[left:right]
        return owner, StructuralSourceSpan(left, right, hashlib.sha256(owner.encode()).hexdigest())
    return None


def _ordered_list_source_units(
    source_text: str,
    *,
    source_id: str,
    candidate_id: str,
    chunk_id: str,
    page: Optional[int],
    source_container: str,
    metadata: Mapping[str, Any],
    body_start: int,
) -> tuple[StructuralEvidenceUnit, ...]:
    """Preserve consecutive printed ordinals and their source-local labels.

    Presentation extraction can put a number and its label in separate text
    paragraphs. Two or more consecutive markers with intervening whitespace
    prove the list boundary; short nominal labels do not need to be sentences.
    A dangling marker, numerical-only split label, or interrupted numbering
    never supplies an invented item or a certified complete sequence.
    """
    lines = _line_spans(source_text, body_start)
    # ordinal, item start/end, label start/end; a missing label remains visible
    # in its run so a dangling final marker cannot silently become complete.
    entries: list[tuple[int, int, int, Optional[int], Optional[int]]] = []
    for index, (start, end) in enumerate(lines):
        line = source_text[start:end]
        inline = _ORDERED_INLINE_ITEM_RE.fullmatch(line)
        marker = _ORDERED_MARKER_ONLY_RE.fullmatch(line)
        if not inline and not marker:
            continue
        match = inline or marker
        assert match is not None
        ordinal = int(match.group("parenthesized") or match.group("ordinal"))
        label_start: Optional[int] = None
        label_end: Optional[int] = None
        if inline:
            label_start, label_end = start + inline.start("label"), end
        elif index + 1 < len(lines):
            next_start, next_end = lines[index + 1]
            label = source_text[next_start:next_end]
            if (
                not _ORDERED_MARKER_ONLY_RE.fullmatch(label)
                and not _ORDERED_INLINE_ITEM_RE.fullmatch(label)
                and _LETTER_RE.search(label)
            ):
                label_start, label_end = next_start, next_end
        if label_start is not None and label_end is not None:
            last_index = index + (0 if inline else 1)
            # Continue only inside the same stored paragraph. A blank line is
            # a source boundary and cannot attach an unrelated heading/prose.
            while last_index + 1 < len(lines):
                following_start, following_end = lines[last_index + 1]
                following = source_text[following_start:following_end]
                gap = source_text[label_end:following_start]
                if (
                    len(re.findall(r"\r?\n", gap)) != 1
                    or _ORDERED_MARKER_ONLY_RE.fullmatch(following)
                    or _ORDERED_INLINE_ITEM_RE.fullmatch(following)
                    or _BULLET_START_RE.match(following)
                ):
                    break
                label_end = following_end
                last_index += 1
            end = label_end
        entries.append((ordinal, start, end, label_start, label_end))
    groups: list[list[tuple[int, int, int, Optional[int], Optional[int]]]] = []
    current: list[tuple[int, int, int, Optional[int], Optional[int]]] = []
    for entry in entries:
        if current and source_text[current[-1][2]:entry[1]].strip():
            groups.append(current)
            current = []
        current.append(entry)
    if current:
        groups.append(current)
    groups = [group for group in groups if len(group) >= 2
              and all(right[0] == left[0] + 1 for left, right in zip(group, group[1:]))
              and all(
        item[3] is not None and item[4] is not None
        and item[2] - item[1] <= MAX_STRUCTURAL_UNIT_CHARS
        for item in group
    )]
    word_groups = word_ordinal_list_spans(source_text, body_start)
    word_headings = {items[0][1]: (start, end) for start, end, items in word_groups}
    groups.extend(list(items) for _start, _end, items in word_groups)
    groups.sort(key=lambda group: group[0][1])
    if not groups or sum(len(group) for group in groups) > MAX_STRUCTURAL_UNITS_PER_CHUNK:
        return ()

    slide_values: list[int] = []
    for key in ("slide", "slide_number"):
        value = metadata.get(key)
        if value is None or value == "":
            continue
        if type(value) is int and value > 0:
            slide_values.append(value)
        elif isinstance(value, str) and re.fullmatch(r"[1-9]\d{0,5}", value):
            slide_values.append(int(value))
        else:
            return ()
    decoration = _SOURCE_DECORATION_LINE_RE.match(source_text)
    if decoration:
        locator = re.fullmatch(
            r"(?:slide|שקופית)\s+(\d{1,6})", decoration.group("locator"), re.IGNORECASE,
        )
        if locator:
            slide_values.append(int(locator.group(1)))
    if len(set(slide_values)) > 1:
        return ()
    slide = slide_values[0] if slide_values else None

    units: list[StructuralEvidenceUnit] = []
    consumed = [(entry[1], entry[2]) for group in groups for entry in group]
    for group_index, group in enumerate(groups, start=1):
        list_id = _parent_identity(candidate_id, "ordered-list", group_index)
        heading_bounds = word_headings.get(group[0][1]) or next((
            (start, end) for start, end in reversed(lines)
            if end <= group[0][1]
            and not any(left <= start < right for left, right in consumed)
            and _looks_heading(source_text[start:end])
        ), None)
        heading = source_text[heading_bounds[0]:heading_bounds[1]] if heading_bounds else ""
        heading_span = (
            StructuralSourceSpan(*heading_bounds, hashlib.sha256(heading.encode("utf-8")).hexdigest())
            if heading_bounds else None
        )
        context_span = (
            StructuralSourceSpan(
                heading_bounds[0], group[0][1],
                hashlib.sha256(source_text[heading_bounds[0]:group[0][1]].encode("utf-8")).hexdigest(),
            ) if heading_bounds else None
        )
        for ordinal, start, end, _label_start, _label_end in group:
            item_id = _parent_identity(list_id, "item", ordinal)
            unit = _make_unit(
                source_text=source_text, start=start, end=end,
                structural_type=StructuralEvidenceType.LIST_ITEM,
                source_id=source_id, candidate_id=candidate_id, chunk_id=chunk_id,
                page=page, source_container=source_container, parent_id=list_id,
                list_id=list_id, item_id=item_id,
                segmentation_method="source_ordered_list",
                derivation_type="explicit_source_ordered_list", confidence=1.0,
            )
            units.append(replace(unit, list_ordinal=ordinal, list_heading=heading,
                                 list_heading_span=heading_span,
                                 list_context_span=context_span, slide=slide))
    # Retain pre-existing bounded factual prose around a list; headings remain
    # context on their items rather than becoming unsupported factual units.
    prose_spans = _source_sentence_spans(source_text, body_start) if word_groups else lines
    excluded = [*consumed, *((start, end) for start, end, _items in word_groups)]
    for start, end in prose_spans:
        if not any(start < right and left < end for left, right in excluded) and _bounded_coherent_proposition(source_text[start:end]):
            units.append(_make_unit(
                source_text=source_text, start=start, end=end,
                structural_type=StructuralEvidenceType.PROSE_PROPOSITION,
                source_id=source_id, candidate_id=candidate_id, chunk_id=chunk_id,
                page=page, source_container=source_container,
                segmentation_method="layout_line_proposition",
                derivation_type="bounded_source_proposition", confidence=0.94,
            ))
    return tuple(sorted((unit for unit in units if unit.structurally_valid),
                        key=lambda unit: unit.source_spans[0].start))


def _decompose_factual_body(
    source_text: str,
    *,
    source_id: str,
    candidate_id: str,
    chunk_id: str,
    page: Optional[int],
    metadata: Mapping[str, Any],
    body_start: int,
) -> tuple[StructuralEvidenceUnit, ...]:
    """Decompose a canonical factual body without request semantics."""
    if not source_text.strip() or not source_id or not candidate_id or not chunk_id:
        return ()
    content_start = body_start
    source_container = f"candidate:{candidate_id}:chunk:{chunk_id}"
    structural_hint = str(
        metadata.get("structural_type")
        or metadata.get("source_unit_type")
        or metadata.get("unit_type")
        or ""
    ).casefold()
    table_hint = bool(
        metadata.get("has_table_context")
        or structural_hint in {"table", "table_row", "table_cell_group"}
    )
    chart_hint = bool(
        metadata.get("chart_title")
        or structural_hint in {"chart", "chart_data", "chart_category_record"}
    )
    units: list[StructuralEvidenceUnit] = []

    line_spans = _line_spans(source_text, content_start)
    pipe_lines = [
        bounds for bounds in line_spans
        if _PIPE_ROW_RE.match(source_text[bounds[0]:bounds[1]])
        and not _PIPE_SEPARATOR_RE.match(source_text[bounds[0]:bounds[1]])
    ]
    if len(pipe_lines) >= 2:
        header_text = source_text[pipe_lines[0][0]:pipe_lines[0][1]]
        columns = _column_names(header_text)
        parent = _parent_identity(candidate_id, "table", 1)
        for ordinal, (start, end) in enumerate(pipe_lines[1:], start=1):
            row_id = _parent_identity(candidate_id, "row", ordinal)
            units.append(_make_unit(
                source_text=source_text,
                start=start,
                end=end,
                structural_type=StructuralEvidenceType.TABLE_ROW,
                source_id=source_id,
                candidate_id=candidate_id,
                chunk_id=chunk_id,
                page=page,
                source_container=source_container,
                parent_id=parent,
                row_id=row_id,
                column_roles=columns,
                segmentation_method="explicit_pipe_row",
                derivation_type="explicit_source_table",
                confidence=1.0,
            ))
        return tuple(unit for unit in units if unit.structurally_valid)

    # Ingestion deliberately repeats table headers beside every value and
    # emits blocks headed "Table row N:" (and equivalent structured-record
    # markers).  Those bullet-looking fields are one logical row, not sibling
    # list items, so reconstruct the row before generic bullet handling.
    record_headers = list(
        _STRUCTURED_RECORD_HEADER_RE.finditer(source_text, content_start)
    )
    if record_headers:
        parent = _parent_identity(candidate_id, "structured-records", 1)
        for ordinal, header in enumerate(record_headers, start=1):
            start = header.end()
            end = (
                record_headers[ordinal].start()
                if ordinal < len(record_headers)
                else len(source_text)
            )
            start, end = _trimmed_bounds(source_text, start, end)
            if start >= end:
                continue
            local_text = source_text[start:end]
            row_label, columns = _structured_record_layout(local_text)
            row_id = _parent_identity(candidate_id, "row", ordinal)
            spreadsheet = header.group("kind").casefold() == "spreadsheet row"
            row_number = int(header.group("number")) if spreadsheet else None
            explicit_header_row = metadata.get("header_row", metadata.get("header_row_number"))
            header_row = (
                int(explicit_header_row)
                if isinstance(explicit_header_row, (int, str))
                and not isinstance(explicit_header_row, bool)
                and str(explicit_header_row).isdigit() and int(explicit_header_row) > 0
                else None
            )
            unit = _make_unit(
                source_text=source_text,
                start=start,
                end=end,
                structural_type=StructuralEvidenceType.TABLE_ROW,
                source_id=source_id,
                candidate_id=candidate_id,
                chunk_id=chunk_id,
                page=page,
                source_container=source_container,
                parent_id=parent,
                row_id=row_id,
                row_label=row_label,
                column_roles=columns,
                segmentation_method="ingested_structured_record",
                derivation_type="logical_row_from_repeated_fields",
                confidence=0.99,
                workbook=str(metadata.get("source_relative_path") or metadata.get("title") or "") if spreadsheet else "",
                sheet=str(metadata.get("sheet") or "") if spreadsheet else "",
                worksheet_row=row_number,
                header_row=header_row if spreadsheet else None,
            )
            if spreadsheet:
                unit = replace(unit, worksheet_row_span=StructuralSourceSpan(
                    start=header.start("number"), end=header.end("number"),
                    sha256=hashlib.sha256(header.group("number").encode("utf-8")).hexdigest(),
                ))
            units.append(unit)
        return tuple(unit for unit in units if unit.structurally_valid)

    ordered_list = (
        _ordered_list_source_units(
            source_text, source_id=source_id, candidate_id=candidate_id,
            chunk_id=chunk_id, page=page, source_container=source_container,
            metadata=metadata, body_start=content_start,
        ) if not table_hint and not chart_hint else ()
    )
    if ordered_list:
        return ordered_list

    flattened_chart = _flattened_chart_series_units(
        source_text,
        source_id=source_id,
        candidate_id=candidate_id,
        chunk_id=chunk_id,
        page=page,
        source_container=source_container,
        body_start=content_start,
    )
    if flattened_chart:
        return flattened_chart

    bullet_start_offsets = {
        match.start()
        for match in _BULLET_START_RE.finditer(source_text, content_start)
    }
    # Some PDF/layout extractors reverse a compact item marker as
    # ``Role • : responsibility``.  Keep its local label with its own item;
    # this is a bounded source layout form, not semantic reconstruction.
    reversed_markers = [
        *_REVERSED_BULLET_MARKER_RE.finditer(source_text, content_start),
        *_TRAILING_BULLET_MARKER_RE.finditer(source_text, content_start),
    ]
    for marker in reversed_markers:
        bullet_start_offsets.add(marker.start("label"))
    for marker in _INLINE_BULLET_MARKER_RE.finditer(source_text, content_start):
        if not any(
            reversed_marker.start() <= marker.start() < reversed_marker.end()
            for reversed_marker in reversed_markers
        ):
            bullet_start_offsets.add(marker.start())
    bullet_starts = sorted(bullet_start_offsets)
    if bullet_starts:
        list_id = _parent_identity(candidate_id, "list", 1)
        for ordinal, start in enumerate(bullet_starts, start=1):
            end = (
                bullet_starts[ordinal]
                if ordinal < len(bullet_starts)
                else len(source_text)
            )
            if ordinal < len(bullet_starts):
                end = _list_item_end_before_trailing_heading(
                    source_text, start, end
                )
            item_id = _parent_identity(candidate_id, "item", ordinal)
            units.append(_make_unit(
                source_text=source_text,
                start=start,
                end=end,
                structural_type=StructuralEvidenceType.LIST_ITEM,
                source_id=source_id,
                candidate_id=candidate_id,
                chunk_id=chunk_id,
                page=page,
                source_container=source_container,
                parent_id=list_id,
                list_id=list_id,
                item_id=item_id,
                segmentation_method="explicit_list_marker",
                derivation_type="explicit_source_list",
                confidence=1.0,
            ))
        return tuple(unit for unit in units if unit.structurally_valid)

    record_spans: list[tuple[int, int]] = []
    segmentation_method = ""
    derivation_type = ""
    confidence = 0.0
    if table_hint or chart_hint:
        factual_lines = [
            bounds for bounds in line_spans
            if _factual_line(source_text[bounds[0]:bounds[1]])
        ]
        if len(factual_lines) >= 2:
            record_spans = factual_lines
            segmentation_method = "layout_line_record"
            derivation_type = "logical_row_from_layout_lines"
            confidence = 0.94
        else:
            delimited = [
                bounds for bounds in _delimited_record_spans(source_text, content_start)
                if _factual_line(source_text[bounds[0]:bounds[1]])
            ]
            if len(delimited) >= 2:
                record_spans = delimited
                segmentation_method = "delimited_flat_record"
                derivation_type = "logical_row_from_delimiters"
                confidence = 0.9
            else:
                statuses = _status_record_spans(source_text, content_start)
                if len(statuses) >= 2:
                    record_spans = statuses
                    segmentation_method = "terminal_status_pattern"
                    derivation_type = "logical_row_from_repeated_status"
                    confidence = 0.82
    if record_spans:
        parent_kind = "chart" if chart_hint else "table"
        parent = _parent_identity(candidate_id, parent_kind, 1)
        columns = tuple(
            str(metadata.get(key) or "")
            for key in (
                "row_header", "column_header", "value_header", "status_header"
            )
            if str(metadata.get(key) or "")
        )
        for ordinal, (start, end) in enumerate(record_spans, start=1):
            if chart_hint:
                category_id = _parent_identity(candidate_id, "category", ordinal)
                unit_type = StructuralEvidenceType.CHART_CATEGORY_RECORD
                row_id = ""
            else:
                row_id = _parent_identity(candidate_id, "row", ordinal)
                category_id = ""
                unit_type = StructuralEvidenceType.TABLE_ROW
            units.append(_make_unit(
                source_text=source_text,
                start=start,
                end=end,
                structural_type=unit_type,
                source_id=source_id,
                candidate_id=candidate_id,
                chunk_id=chunk_id,
                page=page,
                source_container=source_container,
                parent_id=parent,
                row_id=row_id,
                category_id=category_id,
                column_roles=columns,
                segmentation_method=segmentation_method,
                derivation_type=derivation_type,
                confidence=confidence,
            ))
        return tuple(unit for unit in units if unit.structurally_valid)

    nearest_heading = ""
    prose_candidates: list[tuple[int, int, StructuralEvidenceType, str, float]] = []
    for line_start, line_end in line_spans:
        line = source_text[line_start:line_end]
        if _looks_heading(line):
            heading_id = _parent_identity(candidate_id, "heading", len(prose_candidates) + 1)
            nearest_heading = heading_id
            prose_candidates.append((
                line_start, line_end, StructuralEvidenceType.HEADING_CONTEXT,
                "bounded_heading", 0.98,
            ))
            continue
        punctuation_lost = (
            []
            if _SENTENCE_BREAK_RE.search(source_text, line_start, line_end)
            else _predicate_clause_spans(source_text, line_start, line_end)
        )
        if punctuation_lost:
            prose_candidates.extend((
                start,
                end,
                StructuralEvidenceType.PROSE_PROPOSITION,
                "repeated_predicate_boundary",
                0.84,
            ) for start, end in punctuation_lost)
            continue
        cursor = line_start
        for match in _SENTENCE_BREAK_RE.finditer(source_text, line_start, line_end):
            start, end = _trimmed_bounds(source_text, cursor, match.start())
            if start < end:
                prose_candidates.append((
                    start, end, StructuralEvidenceType.PROSE_PROPOSITION,
                    "punctuation_sentence", 0.96,
                ))
            cursor = match.end()
        start, end = _trimmed_bounds(source_text, cursor, line_end)
        if start < end:
            prose_candidates.append((
                start, end, StructuralEvidenceType.PROSE_PROPOSITION,
                "layout_line_proposition", 0.88 if not _SENTENCE_END_RE.search(source_text[start:end]) else 0.94,
            ))
    for ordinal, (start, end, unit_type, method, item_confidence) in enumerate(
        prose_candidates, start=1
    ):
        parent = nearest_heading if unit_type is not StructuralEvidenceType.HEADING_CONTEXT else ""
        units.append(_make_unit(
            source_text=source_text,
            start=start,
            end=end,
            structural_type=unit_type,
            source_id=source_id,
            candidate_id=candidate_id,
            chunk_id=chunk_id,
            page=page,
            source_container=source_container,
            parent_id=parent,
            segmentation_method=method,
            derivation_type="bounded_source_proposition",
            confidence=item_confidence,
        ))
        if len(units) >= MAX_STRUCTURAL_UNITS_PER_CHUNK:
            break
    propositions = tuple(
        unit for unit in units
        if unit.structurally_valid
        and unit.structural_type is not StructuralEvidenceType.HEADING_CONTEXT
        and _bounded_coherent_proposition(unit.local_text)
    )
    return propositions


def canonical_structural_entry(
    text: str,
    *,
    source_id: str,
    candidate_id: str,
    chunk_id: str,
    page: Optional[int],
    metadata: Optional[Mapping[str, Any]] = None,
) -> StructuralEntryResult:
    """Enter every source candidate through decoration/body separation.

    This is intentionally source-only.  It neither knows the request nor
    applies semantic predicates, so ordinary, operator, and model-assisted
    evidence must observe the exact same bounded units and original spans.
    """
    source_text = str(text or "")
    normalized_metadata = dict(metadata or {})
    if not source_text.strip() or not source_id or not candidate_id or not chunk_id:
        return StructuralEntryResult(
            status="rejected", source_id=source_id, candidate_id=candidate_id,
            source_decoration_detected=False, source_decoration_kind="",
            source_decoration_span=None, factual_body_span=None,
            factual_body_text="", coarse_structural_type="unknown",
            canonical_decomposition_attempted=False, structural_units=(),
            rejection_reason="candidate_identity_or_text_missing",
        )
    decoration_bounds, decoration_kind = _source_decoration_bounds(source_text)
    decoration_span = (
        StructuralSourceSpan(
            start=decoration_bounds[0], end=decoration_bounds[1],
            sha256=hashlib.sha256(
                source_text[decoration_bounds[0]:decoration_bounds[1]].encode("utf-8")
            ).hexdigest(),
        )
        if decoration_bounds else None
    )
    body_start, body_end = _trimmed_bounds(
        source_text, decoration_bounds[1] if decoration_bounds else 0, len(source_text)
    )
    if body_start >= body_end:
        return StructuralEntryResult(
            status="rejected", source_id=source_id, candidate_id=candidate_id,
            source_decoration_detected=bool(decoration_bounds),
            source_decoration_kind=decoration_kind,
            source_decoration_span=decoration_span, factual_body_span=None,
            factual_body_text="", coarse_structural_type="header_footer",
            canonical_decomposition_attempted=False, structural_units=(),
            rejection_reason=(
                "source_decoration_only" if decoration_bounds
                else "empty_body_after_decoration"
            ),
        )
    body = source_text[body_start:body_end]
    body_span = StructuralSourceSpan(
        start=body_start, end=body_end,
        sha256=hashlib.sha256(body.encode("utf-8")).hexdigest(),
    )
    coarse_hint = _coarse_structural_hint(normalized_metadata, body)
    if _header_footer_only_body(body):
        return StructuralEntryResult(
            status="rejected", source_id=source_id, candidate_id=candidate_id,
            source_decoration_detected=bool(decoration_bounds),
            source_decoration_kind=decoration_kind,
            source_decoration_span=decoration_span, factual_body_span=body_span,
            factual_body_text=body, coarse_structural_type="header_footer",
            canonical_decomposition_attempted=False, structural_units=(),
            rejection_reason="header_footer_only",
        )
    caption_candidate = _unsupported_source_structure(body)
    units = _decompose_factual_body(
        source_text,
        source_id=source_id,
        candidate_id=candidate_id,
        chunk_id=chunk_id,
        page=page,
        metadata=normalized_metadata,
        body_start=body_start,
    )
    # A source heading such as "Figure 4" remains non-factual decoration.
    # It may be admitted only when the *same bounded source body* proves a
    # data structure independently of every request: an explicit list, a
    # paired percentage/time proposition, or a cardinality-preserving chart
    # value/category series.  Never promote caption text by topical overlap.
    if caption_candidate and not _caption_has_source_local_data_shape(body, units):
        return StructuralEntryResult(
            status="rejected", source_id=source_id, candidate_id=candidate_id,
            source_decoration_detected=bool(decoration_bounds),
            source_decoration_kind=decoration_kind,
            source_decoration_span=decoration_span, factual_body_span=body_span,
            factual_body_text=body, coarse_structural_type=coarse_hint,
            canonical_decomposition_attempted=bool(units), structural_units=(),
            rejection_reason="unsupported_structural_type",
        )
    if units:
        return StructuralEntryResult(
            status="accepted", source_id=source_id, candidate_id=candidate_id,
            source_decoration_detected=bool(decoration_bounds),
            source_decoration_kind=decoration_kind,
            source_decoration_span=decoration_span, factual_body_span=body_span,
            factual_body_text=body, coarse_structural_type=coarse_hint,
            canonical_decomposition_attempted=True, structural_units=units,
        )
    if len(body) > MAX_STRUCTURAL_UNIT_CHARS:
        reason = "structural_unit_too_large"
    elif len(_line_spans(source_text, body_start)) > MAX_STRUCTURAL_UNITS_PER_CHUNK:
        reason = "bounded_composite_recovery_failed"
    elif _SENTENCE_END_RE.search(body) or _TEXTUAL_PROPOSITION_RE.search(body):
        reason = "proposition_boundary_unresolved"
    else:
        reason = "structural_decomposition_empty"
    return StructuralEntryResult(
        status="rejected", source_id=source_id, candidate_id=candidate_id,
        source_decoration_detected=bool(decoration_bounds),
        source_decoration_kind=decoration_kind,
        source_decoration_span=decoration_span, factual_body_span=body_span,
        factual_body_text=body, coarse_structural_type=coarse_hint,
        canonical_decomposition_attempted=True, structural_units=(),
        rejection_reason=reason,
    )


def decompose_structural_evidence(
    text: str,
    *,
    source_id: str,
    candidate_id: str,
    chunk_id: str,
    page: Optional[int],
    metadata: Optional[Mapping[str, Any]] = None,
) -> tuple[StructuralEvidenceUnit, ...]:
    """Compatibility view of the canonical structural-entry result."""
    return canonical_structural_entry(
        text,
        source_id=source_id,
        candidate_id=candidate_id,
        chunk_id=chunk_id,
        page=page,
        metadata=metadata,
    ).structural_units


def structural_unit_for_span(
    units: Sequence[StructuralEvidenceUnit],
    start: int,
    end: int,
) -> Optional[StructuralEvidenceUnit]:
    compatible = [
        unit for unit in units
        if any(span.start <= start and end <= span.end for span in unit.source_spans)
    ]
    if not compatible:
        return None
    compatible.sort(key=lambda unit: sum(
        span.end - span.start for span in unit.source_spans
    ))
    return compatible[0]


def bounded_adjacent_structural_unit(
    source_text: str,
    units: Sequence[StructuralEvidenceUnit],
) -> Optional[StructuralEvidenceUnit]:
    """Compose adjacent prose spans without crossing row/item/category groups."""
    members = tuple(units)
    if len(members) < 2:
        return members[0] if members else None
    identity = {
        (item.source_id, item.candidate_id, item.chunk_id, item.page)
        for item in members
    }
    if len(identity) != 1:
        return None
    structural_groups = {
        item.structural_group_id
        for item in members
        if item.structural_type in {
            StructuralEvidenceType.TABLE_ROW,
            StructuralEvidenceType.LIST_ITEM,
            StructuralEvidenceType.CHART_CATEGORY_RECORD,
        }
    }
    if len(structural_groups) > 1:
        return None
    spans = sorted(
        (span for item in members for span in item.source_spans),
        key=lambda item: (item.start, item.end),
    )
    if not spans:
        return None
    for left, right in zip(spans, spans[1:]):
        gap = source_text[left.end:right.start]
        if right.start < left.end or len(gap) > 12 or gap.strip(" \t\r\n"):
            return None
    start, end = spans[0].start, spans[-1].end
    if end - start > MAX_STRUCTURAL_UNIT_CHARS:
        return None
    source_id, candidate_id, chunk_id, page = next(iter(identity))
    result = _make_unit(
        source_text=source_text,
        start=start,
        end=end,
        structural_type=StructuralEvidenceType.OTHER_BOUNDED_UNIT,
        source_id=source_id,
        candidate_id=candidate_id,
        chunk_id=chunk_id,
        page=page,
        source_container=members[0].source_container,
        parent_id=(
            members[0].parent_structural_id
            if len({item.parent_structural_id for item in members}) == 1
            else ""
        ),
        segmentation_method="bounded_adjacent_reconstruction",
        derivation_type="explicit_bounded_composite",
        confidence=min(item.structural_confidence for item in members),
    )
    return replace(result, source_spans=tuple(spans))


def account_scalar_spans(
    source_text: str,
    entry: StructuralEntryResult,
) -> tuple[tuple[StructuralEvidenceUnit, ...], tuple[ScalarSpanAudit, ...]]:
    """Ensure every factual-body scalar has a bounded structural outcome.

    Canonical decomposition remains the primary authority. A scalar absent
    from every unit is recovered only as its own sentence-bounded source span;
    no request data, whole chunk, or unrelated neighbouring unit is merged.
    When that bounded recovery cannot be proven, the scalar receives an
    explicit rejection audit instead of silently disappearing.
    """
    if not entry.accepted or entry.factual_body_span is None:
        return entry.structural_units, ()
    body_span = entry.factual_body_span
    body = entry.factual_body_text
    units = list(entry.structural_units)
    audits: list[ScalarSpanAudit] = []
    existing = {unit.unit_id for unit in units}
    source = str(source_text or "")
    if not body or source[body_span.start:body_span.end] != body:
        return tuple(units), ()

    # A period before a numeric token is not automatically a sentence break:
    # source text such as "approx. 420" keeps its governing proposition.
    boundaries = [0]
    for match in re.finditer(
        r"(?<=[!?])\s+|(?<=\.)\s+(?=[A-Z\u05d0-\u05ea])|\n{2,}",
        body,
    ):
        boundaries.append(match.end())
    boundaries.append(len(body))
    boundaries = sorted(set(boundaries))

    for match in _VALUE_RE.finditer(body):
        value_start = body_span.start + match.start()
        value_end = body_span.start + match.end()
        value_hash = hashlib.sha256(
            source[value_start:value_end].encode("utf-8")
        ).hexdigest()
        containing = next((
            unit for unit in units
            if any(
                span.start <= value_start and value_end <= span.end
                for span in unit.source_spans
            )
        ), None)
        if containing is not None:
            audits.append(ScalarSpanAudit(
                (value_start, value_end), value_hash, "ACCOUNTED",
                "structural_evidence_unit", containing.unit_id,
            ))
            continue
        local_start = max(
            (boundary for boundary in boundaries if boundary <= match.start()),
            default=0,
        )
        local_end = min(
            (boundary for boundary in boundaries if boundary >= match.end()),
            default=len(body),
        )
        absolute_start = body_span.start + local_start
        absolute_end = body_span.start + local_end
        local_text = source[absolute_start:absolute_end].strip()
        leading = len(source[absolute_start:absolute_end]) - len(
            source[absolute_start:absolute_end].lstrip()
        )
        absolute_start += leading
        # A caption ordinal or note can share a paragraph with a flattened
        # chart.  Recovering that unrelated scalar as one prose sentence
        # would swallow independently proven chart value/category records and
        # recreate their values without their structural pairing.  Reject the
        # caption scalar explicitly instead; it remains accounted evidence and
        # the chart records keep their original, narrower authority.
        overlaps_structured_record = any(
            unit.structural_type in {
                StructuralEvidenceType.TABLE_ROW,
                StructuralEvidenceType.LIST_ITEM,
                StructuralEvidenceType.CHART_CATEGORY_RECORD,
            }
            and any(
                absolute_start < span.end and span.start < absolute_end
                for span in unit.source_spans
            )
            for unit in units
        )
        if overlaps_structured_record:
            audits.append(ScalarSpanAudit(
                (value_start, value_end), value_hash, "REJECTED",
                "scalar_recovery_overlaps_structured_record",
            ))
            continue
        if (
            not local_text
            or absolute_end <= absolute_start
            or absolute_end - absolute_start > MAX_STRUCTURAL_UNIT_CHARS
            or not _LETTER_RE.search(local_text)
        ):
            audits.append(ScalarSpanAudit(
                (value_start, value_end), value_hash, "REJECTED",
                "scalar_span_bounded_recovery_unavailable",
            ))
            continue
        recovered = _make_unit(
            source_text=source,
            start=absolute_start,
            end=absolute_end,
            structural_type=StructuralEvidenceType.OTHER_BOUNDED_UNIT,
            source_id=entry.source_id,
            candidate_id=entry.candidate_id,
            chunk_id=next((unit.chunk_id for unit in units if unit.chunk_id), ""),
            page=next((unit.page for unit in units), None),
            source_container=(
                units[0].source_container if units
                else f"candidate:{entry.candidate_id}"
            ),
            segmentation_method="lossless_scalar_sentence_recovery",
            derivation_type="explicit_scalar_span_recovery",
            confidence=1.0,
        )
        if recovered.unit_id not in existing:
            units.append(recovered)
            existing.add(recovered.unit_id)
        audits.append(ScalarSpanAudit(
            (value_start, value_end), value_hash, "RECOVERED",
            "bounded_source_sentence_recovery", recovered.unit_id,
        ))
    return tuple(units), tuple(audits)


def atomic_structural_evidence_unit(
    text: str,
    *,
    source_id: str,
    candidate_id: str,
    chunk_id: str,
    page: Optional[int],
    structural_type: StructuralEvidenceType = (
        StructuralEvidenceType.OTHER_BOUNDED_UNIT
    ),
    segmentation_method: str = "declared_atomic_source_unit",
    unit_id: str = "",
) -> StructuralEvidenceUnit:
    """Wrap a caller-proven atomic source unit without inventing layout data."""
    value = str(text or "")
    result = _make_unit(
        source_text=value,
        start=0,
        end=len(value),
        structural_type=structural_type,
        source_id=source_id,
        candidate_id=candidate_id,
        chunk_id=chunk_id,
        page=page,
        source_container=f"candidate:{candidate_id}:chunk:{chunk_id}",
        segmentation_method=segmentation_method,
        derivation_type="already_atomic_source_unit",
        confidence=1.0,
    )
    if not unit_id or unit_id == result.unit_id:
        return result
    fields = tuple(replace(
        item,
        structural_group_id=unit_id,
    ) for item in result.field_bindings)
    return replace(
        result,
        unit_id=str(unit_id),
        structural_group_id=str(unit_id),
        field_bindings=fields,
    )
