"""Bounded semantic critic and authoritative deterministic answer gate."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re
from typing import Literal, Optional, Sequence

from proposition_quality import (
    StructuredFact,
    assess_proposition,
    balanced_delimiters,
    normalize_proposition_text,
    structured_fact_is_complete,
)
from request_performance import timed_stage


SEMANTIC_QUALITY_SCHEMA_VERSION = 1
logger = logging.getLogger(__name__)

_CITATION_RE = re.compile(r"\[(?P<id>\d{1,3})\]")
_LEADING_LIST_RE = re.compile(r"^\s*\d{1,2}[.)]\s+")
_LEADING_FRAGMENT_RE = re.compile(
    r"^\s*(?:"
    r"[\(\[\{),;:–—-]|"
    r"ו(?:\s+|ירידה\b|עלייה\b|גידול\b|קיטון\b|בשנת\b)|"
    r"אבל\b|אך\b|ואילו\b|לעומת\b|בעוד\b|כאשר\b|"
    r"and\b|but\b|while\b|whereas\b|however\b|compared\b"
    r")",
    re.IGNORECASE,
)
_UNRESOLVED_START_RE = re.compile(
    r"^\s*(?:הם|הן|אלה|אלו|זו|זה|אותם|מהם|מהן|"
    r"they|these|those|this|that|them)\b",
    re.IGNORECASE,
)
_NUMBER_RE = re.compile(r"(?<![\w])\d+(?:[.,]\d+)?(?![\w])")
_HEBREW_RE = re.compile(r"[\u05d0-\u05ea]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_QUOTED_VALUE_RE = re.compile(
    r'(?:"[^"\n]{1,240}"|“[^”\n]{1,240}”|״[^״\n]{1,240}״)'
)
_HEBREW_EXTRACTION_SPACE_RE = re.compile(
    r"(?:"
    r"(?<![\u05d0-\u05ea])ו\s+(?:ירידה|עלייה|גידול|קיטון)\b|"
    r"(?<![\u05d0-\u05ea])ב\s+(?:שיעור|ביקוש|חלק|אחוז)\b"
    r")"
)
_MALFORMED_IDENTITY_RE = re.compile(
    r"(?:"
    r"(ביקוש\s+ל[\u05d0-\u05ea\s]{0,80})\s+היה\s+ל\1|"
    r"\b([A-Za-z][A-Za-z -]{2,60})\s+was\s+for\s+\2\b"
    r")",
    re.IGNORECASE,
)
_INCREASE_RE = re.compile(
    r"(?:עלה|עלתה|עלייה|גידול|גדל|גדלה|\b(?:increase|increased|growth|grew)\b)",
    re.IGNORECASE,
)
_DECREASE_RE = re.compile(
    r"(?:ירד|ירדה|ירידה|קיטון|פחת|פחתה|\b(?:decrease|decreased|decline|fell)\b)",
    re.IGNORECASE,
)
_NO_CHANGE_RE = re.compile(
    r"(?:לא\s+השתנ|ללא\s+שינוי|\b(?:no\s+change|did\s+not\s+change|unchanged)\b)",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"(?:\b(?:not|no|never|without|cannot)\b|"
    r"(?:^|\s)(?:לא|אין|ללא|בלי|אינו|אינה)(?=\s|$))",
    re.IGNORECASE,
)
_APPROX_RE = re.compile(
    r"(?:כ\s*[-־]?\s*|כמעט\s+|בערך\s+|approximately\s+|about\s+|around\s+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SemanticQualityReport:
    schema_version: int
    is_complete_sentence: bool
    is_independently_understandable: bool
    is_natural_in_requested_language: bool
    starts_mid_clause: bool
    has_unresolved_reference: bool
    has_balanced_delimiters: bool
    has_explicit_subject: bool
    has_complete_predicate: bool
    preserves_typed_fact: bool
    contains_all_required_endpoints: bool
    introduces_unsupported_fact: bool
    meaning_matches_evidence: bool
    quality_reason_codes: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        return not self.quality_reason_codes


@dataclass(frozen=True)
class FinalGateResult:
    accepted: bool
    reason_codes: tuple[str, ...]
    critic_reports: tuple[SemanticQualityReport, ...]
    answer_unit_count: int


def _without_citations(value: str) -> str:
    return normalize_proposition_text(_CITATION_RE.sub("", value or ""))


def _normalized_numbers(value: str) -> set[str]:
    return {
        match.group(0).replace(",", "")
        for match in _NUMBER_RE.finditer(_CITATION_RE.sub("", value or ""))
    }


def _language_matches(value: str, language: Literal["he", "en"]) -> bool:
    text = _QUOTED_VALUE_RE.sub(" ", _CITATION_RE.sub("", value or ""))
    hebrew = len(_HEBREW_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    if language == "he":
        return hebrew >= 6 and hebrew >= latin * 0.15
    return latin >= 6 and (hebrew <= 4 or latin >= hebrew * 2.0)


def _direction_matches(value: str, fact: StructuredFact) -> bool:
    if fact.fact_type != "comparison":
        return True
    matches = {
        "increase": bool(_INCREASE_RE.search(value)),
        "decrease": bool(_DECREASE_RE.search(value)),
        "no_change": bool(_NO_CHANGE_RE.search(value)),
    }
    return bool(matches.get(str(fact.direction), False)) and sum(matches.values()) == 1


def _approximation_preserved(value: str, fact: StructuredFact) -> bool:
    if fact.fact_type != "comparison":
        return True
    for fact_value, required in (
        (fact.value_a, fact.approximation_a),
        (fact.value_b, fact.approximation_b),
    ):
        if not fact_value:
            return False
        match = re.search(
            rf"{re.escape(str(fact_value))}\s*%",
            value,
            re.IGNORECASE,
        )
        if not match:
            return False
        prefix = value[max(0, match.start() - 32):match.start()]
        approximation_matches = list(_APPROX_RE.finditer(prefix))
        has_approximation = bool(
            approximation_matches
            and len(prefix) - approximation_matches[-1].end() <= 8
        )
        if bool(required) != has_approximation:
            return False
    return True


def _negation_preserved(value: str, fact: StructuredFact) -> bool:
    expected = bool(
        fact.negated
        or _NEGATION_RE.search(fact.statement or fact.evidence_text)
    )
    return bool(_NEGATION_RE.search(value or "")) == expected


def _fact_fields_in_text(
    value: str,
    fact: StructuredFact,
    *,
    language: Optional[Literal["he", "en"]] = None,
    cross_language_semantics_verified: bool = False,
) -> bool:
    normalized = normalize_proposition_text(value).casefold()
    for required in (
        fact.period_a,
        fact.value_a,
        fact.period_b if fact.fact_type == "comparison" else None,
        fact.value_b if fact.fact_type == "comparison" else None,
    ):
        if required and str(required).casefold() not in normalized:
            return False
    fact_semantics = f"{fact.metric} {fact.population}"
    cross_language_output = bool(
        language
        and (
            language == "en"
            and bool(_HEBREW_RE.search(fact_semantics))
            and not _HEBREW_RE.search(value)
            or language == "he"
            and bool(_LATIN_RE.search(fact_semantics))
            and not _LATIN_RE.search(value)
        )
    )
    cross_language_fields_authorized = bool(
        cross_language_output
        and (
            fact.fact_type == "comparison"
            or cross_language_semantics_verified
        )
    )
    if not cross_language_fields_authorized:
        for required in (fact.metric, fact.population):
            if (
                required
                and normalize_proposition_text(required).casefold() not in normalized
            ):
                return False
    return (
        _direction_matches(value, fact)
        and _approximation_preserved(value, fact)
        and _negation_preserved(value, fact)
    )


def _fact_fields_in_evidence(fact: StructuredFact) -> bool:
    evidence = normalize_proposition_text(fact.evidence_text).casefold()
    for required in (
        fact.period_a,
        fact.value_a,
        fact.period_b if fact.fact_type == "comparison" else None,
        fact.value_b if fact.fact_type == "comparison" else None,
    ):
        if required and str(required).casefold() not in evidence:
            return False
    return structured_fact_is_complete(fact)


def _unsupported_numbers(value: str, fact: StructuredFact) -> set[str]:
    answer_numbers = _normalized_numbers(value)
    evidence_numbers = _normalized_numbers(fact.evidence_text)
    return answer_numbers - evidence_numbers


def canonical_quote_is_readable(value: str) -> bool:
    text = normalize_proposition_text(value)
    return bool(
        text
        and balanced_delimiters(text)
        and not _LEADING_FRAGMENT_RE.match(text)
        and not _HEBREW_EXTRACTION_SPACE_RE.search(text)
    )


@timed_stage("semantic_critic_ms")
def semantic_quality_critic(
    answer: str,
    fact: StructuredFact,
    *,
    language: Literal["he", "en"],
    cross_language_semantics_verified: bool = False,
) -> SemanticQualityReport:
    """Return closed structured quality output; never rewrite or add facts."""
    prose = _LEADING_LIST_RE.sub("", _without_citations(answer))
    assessment = assess_proposition(prose)
    starts_mid_clause = bool(_LEADING_FRAGMENT_RE.match(prose))
    unresolved = bool(_UNRESOLVED_START_RE.match(prose))
    balanced = balanced_delimiters(prose) and prose.count('"') % 2 == 0
    preserves = _fact_fields_in_text(
        prose,
        fact,
        language=language,
        cross_language_semantics_verified=cross_language_semantics_verified,
    )
    endpoints = preserves if fact.fact_type == "comparison" else bool(prose)
    unsupported = bool(_unsupported_numbers(prose, fact))
    meaning_matches = _fact_fields_in_evidence(fact) and _direction_matches(prose, fact)
    language_matches = _language_matches(prose, language)
    extraction_space = bool(_HEBREW_EXTRACTION_SPACE_RE.search(prose))
    malformed_identity = bool(_MALFORMED_IDENTITY_RE.search(prose))
    natural_language = bool(
        language_matches
        and not extraction_space
        and not malformed_identity
        and not assessment.malformed_repetition
    )
    if not natural_language:
        logger.info(
            "SEMANTIC_NATURAL_LANGUAGE accepted=false language=%s "
            "language_match=%s extraction_space=%s malformed_identity=%s "
            "malformed_repetition=%s",
            language,
            language_matches,
            extraction_space,
            malformed_identity,
            assessment.malformed_repetition,
        )
    independently_understandable = bool(
        assessment.explicit_subject
        and assessment.complete_predicate
        and not starts_mid_clause
        and not unresolved
    )
    reasons: list[str] = []
    checks = (
        (assessment.sentence_end, "critic_incomplete_sentence"),
        (independently_understandable, "critic_context_dependent"),
        (natural_language, "critic_unnatural_language"),
        (not starts_mid_clause, "critic_mid_clause_start"),
        (not unresolved, "critic_unresolved_reference"),
        (balanced, "critic_unbalanced_delimiters"),
        (assessment.explicit_subject, "critic_subject_missing"),
        (assessment.complete_predicate, "critic_predicate_missing"),
        (preserves, "critic_typed_fact_not_preserved"),
        (endpoints, "critic_endpoint_missing"),
        (not unsupported, "critic_unsupported_number"),
        (meaning_matches, "critic_evidence_mismatch"),
    )
    for passed, reason in checks:
        if not passed:
            reasons.append(reason)
    return SemanticQualityReport(
        schema_version=SEMANTIC_QUALITY_SCHEMA_VERSION,
        is_complete_sentence=assessment.sentence_end,
        is_independently_understandable=independently_understandable,
        is_natural_in_requested_language=natural_language,
        starts_mid_clause=starts_mid_clause,
        has_unresolved_reference=unresolved,
        has_balanced_delimiters=balanced,
        has_explicit_subject=assessment.explicit_subject,
        has_complete_predicate=assessment.complete_predicate,
        preserves_typed_fact=preserves,
        contains_all_required_endpoints=endpoints,
        introduces_unsupported_fact=unsupported,
        meaning_matches_evidence=meaning_matches,
        quality_reason_codes=tuple(dict.fromkeys(reasons)),
    )


def _answer_units(value: str) -> tuple[str, ...]:
    lines = [line.strip() for line in str(value or "").splitlines() if line.strip()]
    if len(lines) > 1:
        return tuple(lines)
    text = str(value or "").strip()
    return (text,) if text else ()


@timed_stage("deterministic_final_gate_ms")
def deterministic_final_gate(
    answer: str,
    facts: Sequence[StructuredFact],
    *,
    language: Literal["he", "en"],
    requested_items: int,
    citation_authorized: Sequence[bool],
    quote_verified: Sequence[bool],
    quote_readable: Sequence[bool],
    critic_reports: Optional[Sequence[SemanticQualityReport]] = None,
    cross_language_semantics_verified: Optional[Sequence[bool]] = None,
) -> FinalGateResult:
    """Fail closed immediately before serialization."""
    units = _answer_units(answer)
    reasons: list[str] = []
    if len(units) != requested_items or len(facts) != requested_items:
        reasons.append("final_item_count_mismatch")
    if len(citation_authorized) != requested_items or not all(citation_authorized):
        reasons.append("final_authorized_source_missing")
    if len(quote_verified) != requested_items or not all(quote_verified):
        reasons.append("final_verified_quote_missing")
    if len(quote_readable) != requested_items or not all(quote_readable):
        reasons.append("final_quote_not_readable")
    if critic_reports is not None and len(critic_reports) != len(facts):
        reasons.append("final_critic_report_mismatch")
    if (
        cross_language_semantics_verified is not None
        and len(cross_language_semantics_verified) != len(facts)
    ):
        reasons.append("final_cross_language_verification_mismatch")

    reports: list[SemanticQualityReport] = []
    for index, (unit, fact) in enumerate(zip(units, facts), start=1):
        report = (
            critic_reports[index - 1]
            if critic_reports is not None and index <= len(critic_reports)
            else semantic_quality_critic(
                unit,
                fact,
                language=language,
                cross_language_semantics_verified=bool(
                    cross_language_semantics_verified is not None
                    and index <= len(cross_language_semantics_verified)
                    and cross_language_semantics_verified[index - 1]
                ),
            )
        )
        reports.append(report)
        reasons.extend(report.quality_reason_codes)
        citation_ids = [int(match.group("id")) for match in _CITATION_RE.finditer(unit)]
        if citation_ids != [index]:
            reasons.append("final_citation_mismatch")
        prose = _LEADING_LIST_RE.sub("", _without_citations(unit))
        if _LEADING_FRAGMENT_RE.match(prose):
            reasons.append("final_leading_fragment")
        if _HEBREW_EXTRACTION_SPACE_RE.search(prose):
            reasons.append("final_intra_word_space")
        if _MALFORMED_IDENTITY_RE.search(prose):
            reasons.append("final_malformed_identity")
        if _unsupported_numbers(prose, fact):
            reasons.append("final_unsupported_number")
        if not _fact_fields_in_text(
            prose,
            fact,
            language=language,
            cross_language_semantics_verified=bool(
                cross_language_semantics_verified is not None
                and index <= len(cross_language_semantics_verified)
                and cross_language_semantics_verified[index - 1]
            ),
        ):
            reasons.append("final_typed_fact_not_preserved")
        assessment = assess_proposition(prose)
        if (
            not assessment.sentence_end
            or not assessment.explicit_subject
            or not assessment.complete_predicate
        ):
            reasons.append("final_incomplete_proposition")
        if not balanced_delimiters(prose):
            reasons.append("final_unbalanced_delimiters")
    unique = tuple(dict.fromkeys(reasons))
    return FinalGateResult(
        accepted=not unique,
        reason_codes=unique,
        critic_reports=tuple(reports),
        answer_unit_count=len(units),
    )
