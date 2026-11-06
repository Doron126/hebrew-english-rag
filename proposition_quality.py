"""Typed factual-proposition validation and deterministic realization.

The functions in this module accept source evidence only.  Questions, planner
output, and conversation history may be used by callers to choose a candidate,
but they are never passed here as factual fields.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Optional

from request_performance import timed_stage


PROPOSITION_SCHEMA_REVISION = "proposition-v2"

_HEBREW_RE = re.compile(r"[\u0590-\u05ff]")
_LETTER_RE = re.compile(r"[A-Za-z\u05d0-\u05ea]")
_WORD_RE = re.compile(r"[A-Za-z\u05d0-\u05ea]{2,}")
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_NUMBER_RE = re.compile(r"(?<![\w])(\d{1,3}(?:,\d{3})+|\d+(?:[.,]\d+)?)(?![\w])")
_PERCENT_VALUE_RE = re.compile(
    r"(?P<approx>כ(?:\s*[-־])?\s*|כמעט\s+|בערך\s+|"
    r"approximately\s+|about\s+|around\s+)?"
    r"(?P<value>\d+(?:[.,]\d+)?)\s*(?:%|\bpercent(?:age)?\b)",
    re.IGNORECASE,
)
_COUNT_VALUE_RE = re.compile(
    r"(?P<approx>כ(?:\s*[-־])?\s*|כמעט\s+|בערך\s+|"
    r"approximately\s+|about\s+|around\s+)?"
    r"(?P<value>\d{1,3}(?:,\d{3})+|\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)
_POPULATION_RE = re.compile(
    r"(?:"
    r"עובדים|מועמדים|משרות|חברות|ארגונים|משיבים|משתתפים|לקוחות|"
    r"אוכלוסי(?:ה|ית)|מדגם|קבוצה|מעסיקים|תלמידים|אנשים|"
    r"\bworkers?\b|\bemployees?\b|\bcandidates?\b|\broles?\b|"
    r"\bcompanies\b|\borganizations\b|\brespondents\b|"
    r"\bparticipants\b|\bcustomers\b|\bpopulation\b|\bsample\b|"
    r"\bgroups?\b|\bemployers?\b|\bstudents?\b|\bpeople\b"
    r")",
    re.IGNORECASE,
)
_METRIC_RE = re.compile(
    r"(?:"
    r"ביקוש|שיעור|חלק|אחוז|מספר|כמות|היקף|נתח|ממוצע|חציון|"
    r"\bdemand\b|\brate\b|\bshare\b|\bpercentage\b|\bpercent\b|"
    r"\bnumber\b|\bcount\b|\bvolume\b|\baverage\b|\bmedian\b"
    r")",
    re.IGNORECASE,
)
_PREDICATE_RE = re.compile(
    r"(?:"
    r"הוא|היא|הם|הן|היה|הייתה|היו|מהווה|עמד|עמדה|נמצא|נמצאה|"
    r"עלה|עלתה|ירד|ירדה|גדל|גדלה|פחת|פחתה|חל|מראה|מצביע|"
    r"בוצע|בוצעה|בוצעו|נכלל|נכללה|נכללו|כלל|כללה|כללו|"
    r"תוקן|תוקנה|תוקנו|שימש|שימשה|שימשו|"
    r"\bis\b|\bare\b|\bwas\b|\bwere\b|\brepresents?\b|"
    r"\bstood\b|\bfound\b|\breached\b|\breported\b|\brequires?\b|\bshows?\b|"
    r"\bindicates?\b|\bincreased?\b|\bdecreased?\b|\bfell\b|\bgrew\b|"
    r"\bchanged?\b|\bcorrected\b|\bincluded\b|\bused\b"
    r")",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"(?:"
    r"בין|לעומת|בהשוואה|ירד|ירדה|עלה|עלתה|פחת|פחתה|גדל|גדלה|"
    r"קיטון|ירידה|גידול|עלייה|"
    r"\bbetween\b|\bcompared\b|\bversus\b|\bfrom\b.{0,80}\bto\b|"
    r"\bincrease\b|\bdecrease\b|\bdecline\b|\bgrowth\b|"
    r"\bfell\b|\bgrew\b"
    r")",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"(?:\bnot\b|\bno\b|\bnever\b|\bwithout\b|לא|אין|אינו|אינה|ללא|בלי|טרם)",
    re.IGNORECASE,
)
_CONTINUATION_START_RE = re.compile(
    r"^\s*[\(\[\{),;:–—-]*\s*(?:"
    r"ו\s+(?=\S)|ו(?:ירידה|עלייה|גידול|קיטון|בשנת)|"
    r"ובשנת|בשנת|ואילו|לעומת|בעוד|אך|אבל|כאשר|מתוכם|מהם|מהן|"
    r"and\b|but\b|while\b|and\s+in|whereas|compared\s+with"
    r")\b",
    re.IGNORECASE,
)
_SENTENCE_FINAL_RE = re.compile(r"[.!?׃…][\"'\u05f4\u2019\u201d)\]}]*\s*$")
_CITATION_RE = re.compile(r"\s*\[\d{1,3}\]\s*$")
_CLAUSE_BREAK_RE = re.compile(
    r"\s*(?:[,;:]|\b(?:לעומת|ואילו|בעוד|whereas|compared\s+with)\b)\s*",
    re.IGNORECASE,
)
_QUALITATIVE_BASELINE_RE = re.compile(
    r"(?:"
    r"(?:ירידה|עלייה|גידול|קיטון).{0,100}(?:הסקר|התקופה)\s+(?:הקודמ|האחרונ)|"
    r"(?:increase|decrease|decline|growth|fell|grew).{0,100}"
    r"(?:prior|previous|last)\s+(?:survey|period|year)"
    r")",
    re.IGNORECASE,
)
_BOUNDED_DELTA_RE = re.compile(
    r"(?:"
    r"(?:ירד|עלה|פחת|גדל|קיטון|ירידה|גידול|עליי?ה)\s+(?:ב|של)\s*-?\d|"
    r"\b(?:fell|grew|increased?|decreased?|declined?)\s+by\s+-?\d"
    r")",
    re.IGNORECASE,
)
_NEW_COMPARISON_CLAUSE_RE = re.compile(
    r"(?:"
    r"ו\s*(?:ירידה|עלייה|גידול|קיטון)\b|"
    r"(?:מצד\s+שני|בעוד\s+ש|ואילו)\s+"
    r"(?=[^,;:.]{0,100}(?:שיעור|ביקוש|חלק|אחוז|עובדים|חברות))|"
    r"\b(?:and|whereas|while|but)\s+"
    r"(?=[^,;:.]{0,100}(?:increase|decrease|decline|growth|rate|share|"
    r"workers?|employees?|companies))"
    r")",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PropositionAssessment:
    complete: bool
    reason_codes: tuple[str, ...]
    explicit_subject: bool
    complete_predicate: bool
    population_present: bool
    metric_present: bool
    comparison_complete: bool
    sentence_start: bool
    sentence_end: bool
    malformed_repetition: bool


@dataclass(frozen=True)
class StructuredFact:
    fact_type: Literal[
        "count",
        "percentage",
        "one_period",
        "comparison",
        "multi_period",
        "categorical",
        "trend",
        "recommendation",
        "supported_limitation",
    ]
    subject: str
    metric: str
    population: str
    period_a: Optional[str] = None
    value_a: Optional[str] = None
    period_b: Optional[str] = None
    value_b: Optional[str] = None
    direction: Optional[Literal["increase", "decrease", "no_change"]] = None
    approximation_a: bool = False
    approximation_b: bool = False
    unit: Optional[str] = None
    negated: bool = False
    evidence_span_ids: tuple[str, ...] = ()
    evidence_text: str = ""
    statement: str = ""
    source_page_span: tuple[str, ...] = ()
    field_evidence: tuple[tuple[str, str], ...] = ()


def normalize_proposition_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _without_citation(value: str) -> str:
    return _CITATION_RE.sub("", normalize_proposition_text(value))


def _malformed_repetition(value: str) -> bool:
    tokens = [
        token.casefold()
        for token in re.findall(r"[A-Za-z\u05d0-\u05ea]{2,}|\d+(?:[.,]\d+)?", value)
    ]
    if len(tokens) < 4:
        return False
    # Repeated adjacent phrases are a reliable signal for outputs equivalent
    # to "demand for workers was for workers"; ordinary repeated stop words
    # alone do not trip this check.
    for width in (2, 3, 4):
        for index in range(0, len(tokens) - (width * 2) + 1):
            first = tokens[index:index + width]
            second = tokens[index + width:index + (width * 2)]
            if first == second and any(len(token) >= 4 for token in first):
                return True
    for index, token in enumerate(tokens[:-1]):
        if len(token) >= 4 and tokens[index + 1] == token:
            return True
    return False


def balanced_delimiters(value: str) -> bool:
    """Validate ordered (), [], and {} pairs without inspecting source content."""
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    for character in str(value or ""):
        if character in "([{":
            stack.append(character)
        elif character in pairs:
            if not stack or stack.pop() != pairs[character]:
                return False
    return not stack


def assess_proposition(value: str) -> PropositionAssessment:
    text = _without_citation(value)
    reasons: list[str] = []
    sentence_start = bool(text) and not _CONTINUATION_START_RE.match(text)
    sentence_end = bool(text) and bool(_SENTENCE_FINAL_RE.search(text))
    explicit_subject = bool(
        sentence_start
        and len(_LETTER_RE.findall(text)) >= 4
        and len(_WORD_RE.findall(text)) >= 2
    )
    metric_present = bool(_METRIC_RE.search(text))
    population_present = bool(_POPULATION_RE.search(text))
    percentage_values = list(_PERCENT_VALUE_RE.finditer(text))
    complete_predicate = bool(
        _PREDICATE_RE.search(text)
        or (metric_present and percentage_values)
    )
    comparison_requested = bool(
        _COMPARISON_RE.search(text)
        or re.search(r"\b(?:ואילו|whereas)\b", text, re.IGNORECASE)
        or len(_YEAR_RE.findall(text)) >= 2
        or len(percentage_values) >= 2
    )
    non_year_numbers = [
        match
        for match in _NUMBER_RE.finditer(text)
        if not _YEAR_RE.fullmatch(match.group(0))
    ]
    comparison_complete = not comparison_requested or bool(
        (
            len(set(_YEAR_RE.findall(text))) >= 2
            and len(percentage_values) >= 2
        )
        or (not percentage_values and len(non_year_numbers) >= 2)
        or bool(_QUALITATIVE_BASELINE_RE.search(text))
        or bool(_BOUNDED_DELTA_RE.search(text))
    )
    malformed = _malformed_repetition(text)
    delimiters_balanced = balanced_delimiters(text)

    if not text:
        reasons.append("empty")
    if not sentence_start:
        reasons.append("mid_clause_start")
    if not sentence_end:
        reasons.append("mid_clause_end")
    if not explicit_subject:
        reasons.append("missing_explicit_subject")
    if not complete_predicate:
        reasons.append("incomplete_predicate")
    if percentage_values and not population_present:
        reasons.append("percentage_population_missing")
    if percentage_values and not metric_present:
        reasons.append("percentage_metric_missing")
    if not comparison_complete:
        if len(set(_YEAR_RE.findall(text))) < 2:
            reasons.append("comparison_period_missing")
        if len(percentage_values) < 2:
            reasons.append("comparison_value_missing")
    if malformed:
        reasons.append("malformed_repetition")
    if not delimiters_balanced:
        reasons.append("unbalanced_delimiters")
    return PropositionAssessment(
        complete=not reasons,
        reason_codes=tuple(dict.fromkeys(reasons)),
        explicit_subject=explicit_subject,
        complete_predicate=complete_predicate,
        population_present=population_present,
        metric_present=metric_present,
        comparison_complete=comparison_complete,
        sentence_start=sentence_start,
        sentence_end=sentence_end,
        malformed_repetition=malformed,
    )


def _normalized_number(value: str) -> str:
    return str(value or "").replace(",", "").strip()


def _numeric_value(value: str) -> Optional[float]:
    try:
        return float(_normalized_number(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _period_value_pairs(
    text: str,
) -> list[tuple[str, str, bool, int, int]]:
    """Pair one period and one percentage in the smallest source clause."""
    pairs: list[tuple[str, str, bool, int, int]] = []
    clause_start = 0
    clauses: list[tuple[int, int]] = []
    for match in _CLAUSE_BREAK_RE.finditer(text):
        if match.start() > clause_start:
            clauses.append((clause_start, match.start()))
        clause_start = match.end()
    clauses.append((clause_start, len(text)))

    for start, end in clauses:
        clause = text[start:end]
        years = list(_YEAR_RE.finditer(clause))
        values = list(_PERCENT_VALUE_RE.finditer(clause))
        if len(years) != 1 or len(values) != 1:
            continue
        value = values[0]
        pairs.append((
            years[0].group(0),
            value.group("value"),
            bool(value.group("approx")),
            start + min(years[0].start(), value.start()),
            start + max(years[0].end(), value.end()),
        ))
    if len(pairs) >= 2:
        return pairs

    # PDF reconstruction may remove clause punctuation while preserving order.
    # Pair each percentage with the nearest unused year under a strict bounded
    # distance, then retain source order.
    years = list(_YEAR_RE.finditer(text))
    values = list(_PERCENT_VALUE_RE.finditer(text))
    used_years: set[int] = set()
    fallback: list[tuple[str, str, bool, int, int]] = []
    for value in values:
        candidates = [
            (abs(year.start() - value.start()), index, year)
            for index, year in enumerate(years)
            if index not in used_years
            and abs(year.start() - value.start()) <= 90
        ]
        if not candidates:
            continue
        _distance, index, year = min(candidates)
        used_years.add(index)
        fallback.append((
            year.group(0),
            value.group("value"),
            bool(value.group("approx")),
            min(year.start(), value.start()),
            max(year.end(), value.end()),
        ))
    return sorted(fallback, key=lambda item: item[3])


def _subject_prefix(text: str, first_fact_start: int) -> str:
    prefix = text[:first_fact_start].strip(" ,;:–—-")
    prefix = re.sub(
        r"^(?:לפי\s+(?:המסמך|הדוח)[,:]?\s*|according\s+to\s+the\s+(?:document|report)[,:]?\s*)",
        "",
        prefix,
        flags=re.IGNORECASE,
    ).strip()
    # Keep the complete evidence phrase, but exclude a trailing comparison
    # connective that belongs to the first endpoint clause.
    prefix = re.sub(
        r"\s+(?:בין|בשנת|in|during)\s*$",
        "",
        prefix,
        flags=re.IGNORECASE,
    ).strip()
    prefix = re.sub(
        r"\s+(?:"
        r"עמד(?:ה|ו)?\s+על|היה|הייתה|היו|מהווה|"
        r"stood\s+at|was|were|represented"
        r")\s*$",
        "",
        prefix,
        flags=re.IGNORECASE,
    ).strip()
    return prefix


def _comparison_fact_from_pair(
    text: str,
    first: tuple[str, str, bool, int, int],
    second: tuple[str, str, bool, int, int],
    *,
    local_start: int,
    evidence_span_ids: tuple[str, ...],
    inherited_metric: str = "",
) -> Optional[StructuredFact]:
    """Build one local comparison from two adjacent period/value clauses."""
    if first[0] == second[0]:
        return None
    between_endpoints = text[first[4]:second[3]]
    if _NEW_COMPARISON_CLAUSE_RE.search(between_endpoints):
        return None
    first_value = _numeric_value(first[1])
    second_value = _numeric_value(second[1])
    if first_value is None or second_value is None:
        return None
    if second_value > first_value:
        direction: Literal["increase", "decrease", "no_change"] = "increase"
    elif second_value < first_value:
        direction = "decrease"
    else:
        direction = "no_change"

    bounded_start = max(0, min(int(local_start), first[3]))
    local_end = max(first[4], second[4])
    local_text = text[bounded_start:local_end].strip(" \t%,;:–—-")
    local_first_start = max(0, first[3] - bounded_start)
    subject = _subject_prefix(local_text, local_first_start)
    subject = re.sub(
        r"^(?:"
        r"[\(\[\{)\]}\u05f4\u2019\u201d%.,;:\s]+|"
        r"(?:ו?\s*לעומת(?:\s+זאת)?|ואילו|בעוד|אך|אבל)\s+|"
        r"ו\s*(?=(?:ירידה|עלייה|גידול|קיטון)\b)|"
        r"(?:and|whereas|while|but|however)\s+"
        r")+",
        "",
        subject,
        flags=re.IGNORECASE,
    ).strip(" \t,;:–—-")
    subject = re.sub(
        r"\s+(?:בהשוואה|לעומת|בין|compared\s+(?:with|to)|versus|between)\b.*$",
        "",
        subject,
        flags=re.IGNORECASE,
    ).strip(" \t,;:–—-")
    first_metric = _METRIC_RE.search(subject)
    if (
        first_metric
        and first_metric.start() > 0
        and _COMPARISON_RE.search(subject[:first_metric.start()])
    ):
        subject = subject[first_metric.start():].strip(" \t,;:–—-")
    subject = re.sub(
        r"^(?:"
        r"ו?\s*(?:ירידה|עלייה|גידול|קיטון)\s+(?:ב|של\s+)?|"
        r"(?:a\s+)?(?:decrease|increase|decline|growth)\s+(?:in|of)\s+"
        r")",
        "",
        subject,
        flags=re.IGNORECASE,
    ).strip(" \t,;:–—-")
    if (
        subject
        and not _METRIC_RE.search(subject)
        and _POPULATION_RE.search(subject)
        and inherited_metric
    ):
        population_start = _POPULATION_RE.search(subject)
        assert population_start is not None
        population_phrase = subject[population_start.start():].strip(
            " \t,;:–—-()[]{}"
        )
        connector = " בקרב " if _HEBREW_RE.search(subject) else " among "
        subject = f"{inherited_metric}{connector}{population_phrase}".strip()
    if not subject or not _METRIC_RE.search(subject) or not _POPULATION_RE.search(subject):
        subject = local_text[:max(first[4] - bounded_start, 0)].strip(
            " \t,;:–—-"
        )
    if not subject or not _METRIC_RE.search(subject) or not _POPULATION_RE.search(subject):
        return None
    metric_match = _METRIC_RE.search(subject)
    population_match = _POPULATION_RE.search(subject)
    return StructuredFact(
        fact_type="comparison",
        subject=subject,
        metric=metric_match.group(0) if metric_match else "",
        population=population_match.group(0) if population_match else "",
        period_a=first[0],
        value_a=first[1],
        period_b=second[0],
        value_b=second[1],
        direction=direction,
        approximation_a=first[2],
        approximation_b=second[2],
        unit="percent",
        negated=bool(_NEGATION_RE.search(local_text)),
        evidence_span_ids=tuple(evidence_span_ids),
        evidence_text=normalize_proposition_text(local_text),
        field_evidence=tuple(
            (field, span_id)
            for field in (
                "subject",
                "metric",
                "population",
                "period_a",
                "value_a",
                "period_b",
                "value_b",
                "direction",
                "approximation_a",
                "approximation_b",
                "unit",
            )
            for span_id in evidence_span_ids
        ),
    )


@timed_stage("structured_fact_extraction_ms")
def extract_structured_comparisons(
    value: str,
    *,
    evidence_span_ids: tuple[str, ...] = (),
) -> tuple[StructuredFact, ...]:
    """Extract each non-overlapping comparison from a compound source span.

    A single source sentence can state several populations using the same two
    periods. Pairing adjacent endpoint clauses preserves those independent
    propositions instead of silently keeping only the first one.
    """
    text = _without_citation(value)
    assessment = assess_proposition(text)
    pairs = _period_value_pairs(text)
    if (
        len(pairs) < 2
        or len(pairs) % 2
        or not assessment.explicit_subject
        or not assessment.population_present
        or not assessment.metric_present
        or assessment.malformed_repetition
        or not balanced_delimiters(text)
    ):
        return ()

    inherited_metric_match = _METRIC_RE.search(text)
    inherited_metric = (
        inherited_metric_match.group(0)
        if inherited_metric_match is not None
        else ""
    )
    output: list[StructuredFact] = []
    for pair_index in range(0, len(pairs) - 1, 2):
        first, second = pairs[pair_index], pairs[pair_index + 1]
        local_start = pairs[pair_index - 1][4] if pair_index else 0
        fact = _comparison_fact_from_pair(
            text,
            first,
            second,
            local_start=local_start,
            evidence_span_ids=evidence_span_ids,
            inherited_metric=inherited_metric,
        )
        if fact is not None and structured_fact_is_complete(fact):
            output.append(fact)
    return tuple(output) if len(output) * 2 == len(pairs) else ()


def extract_structured_comparison(
    value: str,
    *,
    evidence_span_ids: tuple[str, ...] = (),
) -> Optional[StructuredFact]:
    facts = extract_structured_comparisons(
        value,
        evidence_span_ids=evidence_span_ids,
    )
    return facts[0] if facts else None


def comparison_endpoint_pair_count(value: str) -> int:
    """Return the number of source period/value pairs consumed by comparisons."""
    return len(_period_value_pairs(_without_citation(value)))


def structured_fact_reason_codes(fact: Optional[StructuredFact]) -> tuple[str, ...]:
    """Return closed completeness failures for one typed source fact."""
    if fact is None:
        return ("typed_fact_missing",)
    reasons: list[str] = []
    if not fact.subject:
        reasons.append("typed_subject_missing")
    if fact.fact_type in {"comparison", "multi_period", "trend"} and not fact.metric:
        reasons.append("typed_metric_missing")
    if fact.fact_type in {"comparison", "multi_period", "percentage", "one_period"} and not fact.population:
        reasons.append("typed_population_missing")
    if fact.fact_type in {"count", "percentage", "one_period"}:
        if fact.value_a in {None, ""}:
            reasons.append("typed_value_a_missing")
        if not fact.unit:
            reasons.append("typed_unit_missing")
    if fact.fact_type in {"percentage", "one_period"} and not fact.metric:
        reasons.append("typed_metric_missing")
    if fact.fact_type == "comparison":
        for field_name in ("period_a", "value_a", "period_b", "value_b", "direction", "unit"):
            if getattr(fact, field_name) in {None, ""}:
                reasons.append(f"typed_{field_name}_missing")
        if (
            fact.value_a not in {None, ""}
            and fact.value_b not in {None, ""}
            and fact.direction in {"increase", "decrease", "no_change"}
        ):
            first = _numeric_value(str(fact.value_a))
            second = _numeric_value(str(fact.value_b))
            expected = (
                "increase"
                if first is not None and second is not None and second > first
                else "decrease"
                if first is not None and second is not None and second < first
                else "no_change"
            )
            if first is None or second is None or fact.direction != expected:
                reasons.append("typed_direction_mismatch")
    if fact.fact_type in {
        "categorical",
        "recommendation",
        "supported_limitation",
    }:
        if not fact.statement:
            reasons.append("typed_statement_missing")
        elif not assess_proposition(fact.statement).complete:
            reasons.append("typed_statement_incomplete")
    if not fact.evidence_text:
        reasons.append("typed_evidence_missing")
    return tuple(dict.fromkeys(reasons))


def structured_fact_is_complete(fact: Optional[StructuredFact]) -> bool:
    return not structured_fact_reason_codes(fact)


_RECOMMENDATION_RE = re.compile(
    r"(?:מומלץ|המלצה|נדרש|יש\s+צורך|ראוי|"
    r"\b(?:recommend(?:ed|ation)?|should|need(?:ed)?|required)\b)",
    re.IGNORECASE,
)
_LIMITATION_RE = re.compile(
    r"(?:מגבלה|הסתייגות|עם\s+זאת|אולם|למרות|"
    r"\b(?:limitation|caveat|however|although|despite)\b)",
    re.IGNORECASE,
)


def _fact_provenance(
    fields: tuple[str, ...],
    evidence_span_ids: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    return tuple(
        (field, span_id)
        for field in fields
        for span_id in evidence_span_ids
    )


def extract_structured_facts(
    value: str,
    *,
    evidence_span_ids: tuple[str, ...] = (),
) -> tuple[StructuredFact, ...]:
    """Convert one complete source proposition into typed, source-only facts."""
    text = _without_citation(value)
    comparisons = extract_structured_comparisons(
        text,
        evidence_span_ids=evidence_span_ids,
    )
    if comparisons:
        return comparisons
    raw_percentages = list(_PERCENT_VALUE_RE.finditer(text))
    if (
        len(raw_percentages) > 1
        or (
            _COMPARISON_RE.search(text)
            and (
                len(set(_YEAR_RE.findall(text))) > 1
                or len(_NUMBER_RE.findall(text)) > 2
            )
        )
    ):
        # Never downgrade a multi-endpoint proposition after comparison
        # extraction failed. The caller must expand or try another fact.
        return ()
    assessment = assess_proposition(text)
    if not assessment.complete:
        return ()

    metric_match = _METRIC_RE.search(text)
    population_match = _POPULATION_RE.search(text)
    percentages = list(_PERCENT_VALUE_RE.finditer(text))
    years = _YEAR_RE.findall(text)
    numeric_values = [
        match.group(1)
        for match in _NUMBER_RE.finditer(text)
        if not _YEAR_RE.fullmatch(match.group(0))
    ]
    if percentages:
        value_match = percentages[0]
        fact = StructuredFact(
            fact_type="percentage" if len(years) <= 1 else "one_period",
            subject=text.rstrip(" .!?׃…"),
            metric=metric_match.group(0) if metric_match else "",
            population=population_match.group(0) if population_match else "",
            period_a=years[0] if years else None,
            value_a=value_match.group("value"),
            approximation_a=bool(value_match.group("approx")),
            unit="percent",
            evidence_span_ids=tuple(evidence_span_ids),
            evidence_text=text,
            statement=text,
            field_evidence=_fact_provenance(
                ("subject", "metric", "population", "period_a", "value_a", "unit"),
                tuple(evidence_span_ids),
            ),
        )
        return (fact,) if structured_fact_is_complete(fact) else ()

    fact_type: Literal[
        "count",
        "percentage",
        "one_period",
        "comparison",
        "multi_period",
        "categorical",
        "trend",
        "recommendation",
        "supported_limitation",
    ]
    if _RECOMMENDATION_RE.search(text):
        fact_type = "recommendation"
    elif _LIMITATION_RE.search(text):
        fact_type = "supported_limitation"
    elif numeric_values:
        fact_type = "count"
    elif _COMPARISON_RE.search(text):
        fact_type = "trend"
    else:
        fact_type = "categorical"
    direction: Optional[Literal["increase", "decrease", "no_change"]] = None
    if re.search(r"(?:עליי?ה|גידול|עלה|גדל|\b(?:increase|growth|grew)\b)", text, re.I):
        direction = "increase"
    elif re.search(r"(?:ירידה|קיטון|ירד|פחת|\b(?:decrease|decline|fell)\b)", text, re.I):
        direction = "decrease"
    fact = StructuredFact(
        fact_type=fact_type,
        subject=text.rstrip(" .!?׃…"),
        metric=metric_match.group(0) if metric_match else "",
        population=population_match.group(0) if population_match else "",
        period_a=years[0] if years else None,
        value_a=numeric_values[0] if numeric_values else None,
        direction=direction,
        unit="count" if numeric_values else None,
        evidence_span_ids=tuple(evidence_span_ids),
        evidence_text=text,
        statement=text,
        field_evidence=_fact_provenance(
            ("subject", "statement"),
            tuple(evidence_span_ids),
        ),
    )
    return (fact,) if structured_fact_is_complete(fact) else ()


def extract_contextual_structured_facts(
    focus: str,
    evidence: str,
    *,
    evidence_span_ids: tuple[str, ...] = (),
) -> tuple[StructuredFact, ...]:
    """Resolve a bounded percentage reference from preceding source context."""
    focus_text = _without_citation(focus)
    evidence_text = _without_citation(evidence)
    if not focus_text or focus_text == evidence_text:
        return ()
    percentages = list(_PERCENT_VALUE_RE.finditer(focus_text))
    if len(percentages) != 1:
        return ()
    value_match = percentages[0]

    english_population = re.search(
        r"\b(?:population|sample)\s+(?:is|was|includes?)\s+"
        r"(?P<population>[^.!?]{3,160})[.!?]",
        evidence_text,
        re.IGNORECASE,
    )
    english_predicate = re.search(
        r"\)\s*(?P<predicate>(?:employ|have|report|use|require|include|"
        r"represent|show)[^.!?]{2,180})[.!?]",
        focus_text,
        re.IGNORECASE,
    )
    statement = ""
    population = ""
    metric = ""
    if english_population and english_predicate:
        population = english_population.group("population").strip(" ,;:")
        predicate = english_predicate.group("predicate").strip()
        statement = (
            f"The percentage of {population} that {predicate} was "
            f"{value_match.group('value')} percent."
        )
        metric = "percentage"
    else:
        hebrew_population = re.search(
            r"(?:אוכלוסיית|קבוצת)\s+(?P<population>[^.!?׃]{3,160})[.!?׃]",
            evidence_text,
        )
        hebrew_predicate = re.search(
            r"\)\s*(?P<predicate>(?:מעסיק|מדווח|משתמש|דורש|כולל)"
            r"[^.!?׃]{2,180})[.!?׃]",
            focus_text,
        )
        if hebrew_population and hebrew_predicate:
            population = hebrew_population.group("population").strip(" ,;:")
            predicate = hebrew_predicate.group("predicate").strip()
            statement = (
                f"שיעור {population} ש{predicate} עמד על "
                f"{value_match.group('value')}%."
            )
            metric = "שיעור"
    if not statement or not assess_proposition(statement).complete:
        return ()
    fact = StructuredFact(
        fact_type="percentage",
        subject=statement.rstrip("."),
        metric=metric,
        population=population,
        value_a=value_match.group("value"),
        approximation_a=bool(value_match.group("approx")),
        unit="percent",
        evidence_span_ids=tuple(evidence_span_ids),
        evidence_text=evidence_text,
        statement=statement,
        field_evidence=_fact_provenance(
            ("subject", "metric", "population", "value_a", "unit"),
            tuple(evidence_span_ids),
        ),
    )
    return (fact,) if structured_fact_is_complete(fact) else ()


def _hebrew_value(value: str, approximate: bool, preposition: str) -> str:
    if approximate:
        return f"{preposition}כ־{value}%"
    return f"{preposition}־{value}%"


def _english_value(value: str, approximate: bool) -> str:
    return f"approximately {value}%" if approximate else f"{value}%"


@timed_stage("answer_generation_ms")
def realize_structured_fact(
    fact: StructuredFact,
    *,
    language: Literal["he", "en"],
    citation_id: str,
) -> Optional[str]:
    """Render one validated fact without adding any factual field."""
    if not structured_fact_is_complete(fact):
        return None
    if fact.fact_type != "comparison":
        statement = normalize_proposition_text(fact.statement)
        if not statement or not assess_proposition(statement).complete:
            return None
        if statement[-1:] not in {".", "!", "?", "׃", "…"}:
            statement = f"{statement}."
        return normalize_proposition_text(f"{statement} [{citation_id}]")
    if (
        not fact.subject
        or not fact.period_a
        or not fact.period_b
        or fact.value_a is None
        or fact.value_b is None
        or fact.direction not in {"increase", "decrease", "no_change"}
        or fact.unit != "percent"
    ):
        return None
    subject = fact.subject.rstrip(" ,;:.-")
    if language == "he":
        direction = {
            "increase": "עלה",
            "decrease": "ירד",
            "no_change": "לא השתנה",
        }[fact.direction]
        answer = (
            f"{subject} {direction} "
            f"{_hebrew_value(fact.value_a, fact.approximation_a, 'מ')} "
            f"בשנת {fact.period_a} "
            f"{_hebrew_value(fact.value_b, fact.approximation_b, 'ל')} "
            f"בשנת {fact.period_b}. [{citation_id}]"
        )
    else:
        direction = {
            "increase": "increased",
            "decrease": "decreased",
            "no_change": "did not change",
        }[fact.direction]
        answer = (
            f"{subject} {direction} from "
            f"{_english_value(fact.value_a, fact.approximation_a)} "
            f"in {fact.period_a} to "
            f"{_english_value(fact.value_b, fact.approximation_b)} "
            f"in {fact.period_b}. [{citation_id}]"
        )
    return normalize_proposition_text(answer)


@timed_stage("answer_generation_ms")
def realize_structured_endpoint(
    fact: StructuredFact,
    *,
    period: str,
    language: Literal["he", "en"],
    citation_id: str,
) -> Optional[str]:
    """Render one requested endpoint from a validated comparison fact."""
    if fact.fact_type != "comparison" or not fact.subject or fact.unit != "percent":
        return None
    if period == fact.period_a:
        value = fact.value_a
        approximate = fact.approximation_a
    elif period == fact.period_b:
        value = fact.value_b
        approximate = fact.approximation_b
    else:
        return None
    if value is None:
        return None
    subject = fact.subject.rstrip(" ,;:.-")
    if language == "he":
        rendered_value = f"כ־{value}%" if approximate else f"{value}%"
        answer = (
            f"{subject} עמד על {rendered_value} בשנת {period}. "
            f"[{citation_id}]"
        )
    else:
        answer = (
            f"{subject} was {_english_value(value, approximate)} "
            f"in {period}. [{citation_id}]"
        )
    return (
        normalize_proposition_text(answer)
        if assess_proposition(answer).complete
        else None
    )


def requested_unit_is_complete(value: str, requested_items: int) -> bool:
    text = normalize_proposition_text(value)
    if requested_items != 1:
        return bool(text)
    prose = _CITATION_RE.sub("", text).strip()
    sentences = [
        item for item in re.split(r"(?<=[.!?׃…])\s+(?=\S)", prose)
        if item.strip()
    ]
    return len(sentences) == 1 and assess_proposition(text).complete
