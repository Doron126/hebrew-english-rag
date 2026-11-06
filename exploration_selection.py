"""Deterministic, tenant-safe evidence selection for corpus exploration."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import re
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

try:
    from .config_env import normalize_search_text
    from .evidence_quotes import (
        normalize_evidence_whitespace,
        opaque_evidence_chunk_id,
        quote_is_contiguous,
    )
    from .source_access import location_metadata, source_metadata_complete
    from .canonical_text import has_visual_order_artifact
    from .evidence_context import construct_context_span, source_atomic_units
    from .answer_units import AtomicFactSpan
    from .proposition_quality import (
        StructuredFact,
        assess_proposition,
        balanced_delimiters,
        extract_contextual_structured_facts,
        extract_structured_facts,
        structured_fact_is_complete,
    )
    from .source_structure import complete_proposition_spans, reconstruct_soft_blocks
except ImportError:
    from config_env import normalize_search_text
    from evidence_quotes import (
        normalize_evidence_whitespace,
        opaque_evidence_chunk_id,
        quote_is_contiguous,
    )
    from source_access import location_metadata, source_metadata_complete
    from canonical_text import has_visual_order_artifact
    from evidence_context import construct_context_span, source_atomic_units
    from answer_units import AtomicFactSpan
    from proposition_quality import (
        StructuredFact,
        assess_proposition,
        balanced_delimiters,
        extract_contextual_structured_facts,
        extract_structured_facts,
        structured_fact_is_complete,
    )
    from source_structure import complete_proposition_spans, reconstruct_soft_blocks


MAX_EXPLORATION_SCAN_CHUNKS = 128
MAX_EXPLORATION_SCAN_RECORDS = 256
MAX_EXPLORATION_CANDIDATES = 256
MAX_EXPLORATION_ITEMS = 5
MIN_CANDIDATE_WORDS = 8
MAX_CANDIDATE_WORDS = 80
MAX_CANDIDATE_CHARS = 800
MIN_SALIENCE_SCORE = 5.0
CONTEXT_DEPENDENCY_SALIENCE_PENALTY = 1.5

_WORD_RE = re.compile(r"[A-Za-z0-9\u05d0-\u05ea]+")
_NUMBER_RE = re.compile(
    r"(?<![\w])(?:\d{1,3}(?:[,\u066c]\d{3})+|\d+(?:[.,]\d+)?)(?![\w])"
)
_PERCENT_RE = re.compile(r"(?:%|אחוז|אחוזים|\bpercent(?:age)?\b)", re.IGNORECASE)
_CHANGE_RE = re.compile(
    r"(?:קיטון|ירידה|גידול|עליי?ה|השתנ|פחת|גדל|עלה|ירד|לעומת|"
    r"\b(?:increase|decrease|decline|growth|grew|fell|changed?|compared|versus)\b)",
    re.IGNORECASE,
)
_FINDING_RE = re.compile(
    r"(?:ממצא|נמצא|עולה\s+כי|המחקר\s+מראה|המסקנה|תוצאה|"
    r"\b(?:finding|found|shows?|indicates?|conclusion|result)\b)",
    re.IGNORECASE,
)
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
_CAUSAL_RE = re.compile(
    r"(?:בשל|מפני|משום|כתוצאה|לכן|"
    r"\b(?:because|therefore|resulting|due\s+to)\b)",
    re.IGNORECASE,
)
_SALIENT_SECTION_RE = re.compile(
    r"(?:ממצאים|מסקנות|סיכום|המלצות|תוצאות|דיון|"
    r"\b(?:findings|conclusions?|executive\s+summary|recommendations?|results?|discussion)\b)",
    re.IGNORECASE,
)
_REFERENCE_SECTION_RE = re.compile(
    r"(?:מקורות|ביבליוגרפ|הערות\s+שוליים|"
    r"\b(?:references|bibliography|works\s+cited|footnotes?)\b)",
    re.IGNORECASE,
)
_REFERENCE_TEXT_RE = re.compile(
    r"(?:https?://|www\.|doi\s*:|isbn\s*:|"
    r"\bet\s+al\.?\b|\bvol\.?\s*\d+|\bpp?\.?\s*\d+)",
    re.IGNORECASE,
)
_NAVIGATION_RE = re.compile(
    r"(?:תוכן\s+עניינים|רשימת\s+(?:טבלאות|איורים)|"
    r"\b(?:table\s+of\s+contents|list\s+of\s+(?:tables|figures))\b)",
    re.IGNORECASE,
)
_GENERIC_RE = re.compile(
    r"^(?:המסמך\s+עוסק|הדוח\s+עוסק|מידע\s+נוסף|"
    r"(?:this\s+document|the\s+report)\s+(?:discusses|contains)|"
    r"for\s+more\s+information)",
    re.IGNORECASE,
)
_VERBISH_RE = re.compile(
    r"(?:הוא|היא|הם|הן|יש|אין|היה|היו|עומד|עמד|נמצא|נדר|מראה|מצביע|"
    r"חל|עלה|ירד|גדל|פחת|כולל|מהווה|צפוי|"
    r"\b(?:is|are|was|were|has|have|had|shows?|indicates?|includes?|"
    r"represents?|increased?|decreased?|changed?|requires?|found)\b)",
    re.IGNORECASE,
)
_PATH_RE = re.compile(
    r"(?:/(?:home|srv|var/lib|opt|mnt)/|[A-Za-z]:[\\/]|\\\\[^\\\s]+\\)",
    re.IGNORECASE,
)
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "for", "in",
    "on", "at", "by", "with", "and", "or", "that", "this", "from", "as",
    "הוא", "היא", "הם", "הן", "זה", "זו", "זאת", "של", "על", "עם", "את",
    "גם", "או", "לפי", "מן", "מתוך", "כי",
})


@dataclass(frozen=True)
class ExplorationCandidate:
    text: str
    hit_index: int
    sentence_index: int
    score: float
    fact_fingerprint: str
    evidence_chunk_id: str
    source_id: str
    page: Optional[int]
    section: str
    tie_key: tuple[str, int, int, int, str]
    hit: Mapping[str, Any]
    evidence_text: str
    context_dependency_codes: tuple[str, ...]
    context_expansion_type: str
    evidence_sentence_count: int
    atomic_statistic_count: int
    structured_fact: Optional[StructuredFact] = None
    proposition_start: int = 0
    proposition_end: int = 0
    sentence_start: bool = False
    sentence_end: bool = False
    paragraph_id: str = ""
    clause_spans: tuple[tuple[int, int], ...] = ()
    coordination_group_id: str = ""
    heading_context: bool = False
    bullet_context: bool = False
    table_context: bool = False
    page_start: Optional[int] = None
    page_end: Optional[int] = None
    previous_authorized_unit: Optional[tuple[int, int]] = None
    next_authorized_unit: Optional[tuple[int, int]] = None
    origin_mapping_revision: str = ""
    origin_audit_key: str = ""


@dataclass(frozen=True)
class ExplorationSelection:
    candidates_examined: int
    eligible_candidates: int
    selected: tuple[ExplorationCandidate, ...]
    reason_code: str
    context_dependencies_detected: int = 0
    context_expansions: int = 0
    canonical_rejections: int = 0
    proposition_rejections: int = 0
    typed_fact_rejections: int = 0
    coordination_reconstructions: int = 0
    ranked_candidates: tuple[ExplorationCandidate, ...] = ()


def _metadata(hit: Mapping[str, Any]) -> Mapping[str, Any]:
    value = hit.get("metadata") if isinstance(hit, Mapping) else None
    return value if isinstance(value, Mapping) else {}


def _canonical_tokens(value: str) -> tuple[str, ...]:
    normalized = normalize_search_text(value or "")
    tokens = []
    for raw in _WORD_RE.findall(normalized):
        token = raw.casefold()
        if token in _STOPWORDS or len(token) < 2:
            continue
        tokens.append(token)
    return tuple(tokens)


def fact_fingerprint(value: str) -> str:
    """Return a stable semantic-ish fingerprint without retaining source text."""
    normalized_numbers = _NUMBER_RE.sub(
        lambda match: match.group(0).replace(",", "").replace("\u066c", ""),
        normalize_search_text(value or "").casefold(),
    )
    material = "\x1f".join(_canonical_tokens(normalized_numbers))
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def _jaccard(left: Iterable[str], right: Iterable[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def _candidate_near_recent(text: str, recent_texts: Sequence[str]) -> bool:
    tokens = _canonical_tokens(text)
    if len(tokens) < 4:
        return False
    return any(
        _jaccard(tokens, _canonical_tokens(recent)) >= 0.52
        for recent in recent_texts[:8]
    )


def _fact_focus_score(fact: StructuredFact, focus: str) -> float:
    """Match one typed fact to the source clause that contributed its fields."""
    focus_tokens = _canonical_tokens(focus)
    evidence_tokens = _canonical_tokens(fact.evidence_text)
    score = _jaccard(focus_tokens, evidence_tokens)
    for marker in (fact.population, fact.metric):
        marker_tokens = set(_canonical_tokens(marker))
        if marker_tokens and marker_tokens.issubset(set(focus_tokens)):
            score += 0.35
    for marker in (fact.period_a, fact.value_a, fact.period_b, fact.value_b):
        if marker and str(marker) in focus:
            score += 0.2
    return score


def _matching_structured_facts(
    focus: str,
    evidence: str,
    *,
    evidence_span_id: str,
) -> tuple[StructuredFact, ...]:
    facts = tuple(
        fact
        for fact in extract_structured_facts(
            evidence,
            evidence_span_ids=(evidence_span_id,),
        )
        if structured_fact_is_complete(fact)
    )
    if not facts:
        facts = tuple(
            fact
            for fact in extract_contextual_structured_facts(
                focus,
                evidence,
                evidence_span_ids=(evidence_span_id,),
            )
            if structured_fact_is_complete(fact)
        )
    if len(facts) <= 1:
        return facts
    ranked = sorted(
        facts,
        key=lambda fact: (-_fact_focus_score(fact, focus), fact.evidence_text),
    )
    best = _fact_focus_score(ranked[0], focus)
    return tuple(
        fact
        for fact in ranked
        if _fact_focus_score(fact, focus) >= max(0.25, best - 0.05)
    ) or (ranked[0],)


def _balanced_delimiters(value: str) -> bool:
    pairs = (("(", ")"), ("[", "]"), ("{", "}"))
    return all(value.count(left) == value.count(right) for left, right in pairs)


def _looks_like_title_or_fragment(
    text: str,
    metadata: Mapping[str, Any],
) -> bool:
    value = normalize_evidence_whitespace(text)
    words = _WORD_RE.findall(value)
    if (
        not value
        or len(words) < MIN_CANDIDATE_WORDS
        or len(words) > MAX_CANDIDATE_WORDS
        or len(value) > MAX_CANDIDATE_CHARS
        or value.endswith((",", ";", ":", "-", "–", "—"))
        or not _balanced_delimiters(value)
        or _PATH_RE.search(value)
        or _NAVIGATION_RE.search(value)
        or _REFERENCE_TEXT_RE.search(value)
    ):
        return True
    section = str(metadata.get("section") or "")
    title = str(metadata.get("title") or "")
    if _REFERENCE_SECTION_RE.search(section):
        return True
    normalized = normalize_search_text(value).casefold()
    if normalized and normalized in {
        normalize_search_text(section).casefold(),
        normalize_search_text(title).casefold(),
    }:
        return True
    letters = len(re.findall(r"[A-Za-z\u05d0-\u05ea]", value))
    if letters < 20:
        return True
    # Short headings often have title casing or no finite predicate. A number,
    # finding/change cue, or verb-like form is required for a standalone fact.
    if not (
        _NUMBER_RE.search(value)
        or _FINDING_RE.search(value)
        or _CHANGE_RE.search(value)
        or _RECOMMENDATION_RE.search(value)
        or _VERBISH_RE.search(value)
    ):
        return True
    return False


def _salience_score(text: str, metadata: Mapping[str, Any]) -> float:
    words = _WORD_RE.findall(text)
    score = 0.0
    if 12 <= len(words) <= 55:
        score += 2.0
    elif MIN_CANDIDATE_WORDS <= len(words) <= MAX_CANDIDATE_WORDS:
        score += 1.0
    if _NUMBER_RE.search(text):
        score += 3.0
    if _PERCENT_RE.search(text):
        score += 1.5
    if _CHANGE_RE.search(text):
        score += 2.5
    if _FINDING_RE.search(text):
        score += 2.0
    if _RECOMMENDATION_RE.search(text):
        score += 1.5
    if _LIMITATION_RE.search(text):
        score += 1.0
    if _CAUSAL_RE.search(text):
        score += 1.0
    section = str(metadata.get("section") or "")
    if _SALIENT_SECTION_RE.search(section):
        score += 1.5
    location = location_metadata(metadata)
    if location.get("page") is not None or location.get("sheet"):
        score += 1.0
    if text.endswith((".", "!", "?", "׃")):
        score += 0.5
    if _GENERIC_RE.search(text):
        score -= 3.0
    return score


def _hit_chunk_id(hit: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    for value in (
        metadata.get("chunk_id"),
        hit.get("id"),
        metadata.get("chunk_index"),
    ):
        if value is not None and str(value):
            return str(value)
    return ""


def _int_value(value: Any, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return parsed


def select_exploration_evidence(
    hits: Sequence[Mapping[str, Any]],
    *,
    effective_tenant: str,
    requested_items: int = 1,
    avoid_evidence_chunk_ids: Iterable[str] = (),
    avoid_source_page_pairs: Iterable[tuple[str, int]] = (),
    recent_assistant_texts: Sequence[str] = (),
    integrity_validator: Optional[Callable[[Mapping[str, Any]], bool]] = None,
    _protected_fact_signature: Optional[Mapping[str, Any]] = None,
) -> ExplorationSelection:
    """Select stable salient evidence or fail closed with a reason code."""
    try:
        item_limit = max(1, min(int(requested_items), MAX_EXPLORATION_ITEMS))
    except (TypeError, ValueError, OverflowError):
        item_limit = 1
    avoid_chunks = frozenset(str(value) for value in avoid_evidence_chunk_ids)
    avoid_pages = frozenset(
        (str(source), int(page))
        for source, page in avoid_source_page_pairs
        if source
    )
    integrity_cache: dict[tuple[Any, ...], bool] = {}
    examined = 0
    eligible: list[ExplorationCandidate] = []
    seen_fingerprints: set[str] = set()
    context_dependencies_detected = 0
    context_expansions = 0
    canonical_rejections = 0
    proposition_rejections = 0
    typed_fact_rejections = 0
    coordination_reconstructions = 0

    for hit_index, raw_hit in enumerate(list(hits or [])[:MAX_EXPLORATION_SCAN_CHUNKS]):
        if not isinstance(raw_hit, Mapping):
            continue
        metadata = _metadata(raw_hit)
        if (
            str(metadata.get("tenant_id") or "") != effective_tenant
            or not source_metadata_complete(metadata, effective_tenant)
        ):
            continue
        source_id = str(metadata.get("source_id") or "")
        if not source_id:
            continue
        integrity_key = (
            source_id,
            metadata.get("source_size"),
            metadata.get("source_mtime_ns"),
            metadata.get("source_sha256"),
        )
        if integrity_validator is not None:
            integrity_ok = integrity_cache.get(integrity_key)
            if integrity_ok is None:
                try:
                    integrity_ok = bool(integrity_validator(metadata))
                except Exception:
                    integrity_ok = False
                integrity_cache[integrity_key] = integrity_ok
            if not integrity_ok:
                continue

        document = str(raw_hit.get("document") or "")
        if not document:
            continue
        if has_visual_order_artifact(document):
            canonical_rejections += 1
            continue
        document, _reconstruction_reasons = reconstruct_soft_blocks(document)
        candidate_hit = dict(raw_hit)
        candidate_hit["document"] = document
        location = location_metadata(metadata)
        page = location.get("page")
        if not (page or location.get("sheet") or location.get("section")):
            continue
        chunk_id = _hit_chunk_id(raw_hit, metadata)
        if not chunk_id:
            continue
        public_chunk_id = opaque_evidence_chunk_id(source_id, chunk_id, page)
        if public_chunk_id in avoid_chunks:
            continue

        units = list(source_atomic_units(document))
        proposition_by_text: dict[str, Any] = {}
        proposition_spans = list(complete_proposition_spans(document))
        known_unit_texts = {
            normalize_evidence_whitespace(unit.text).casefold()
            for unit in units
        }
        for proposition in proposition_spans:
            proposition_text = normalize_evidence_whitespace(proposition.text)
            proposition_by_text[proposition_text.casefold()] = proposition
            if proposition_text.casefold() in known_unit_texts:
                continue
            units.append(AtomicFactSpan(
                text=proposition_text,
                start=proposition.start,
                end=proposition.end,
                statistic_count=len(_NUMBER_RE.findall(proposition_text)),
                unit_kind="complete_proposition",
            ))
            known_unit_texts.add(proposition_text.casefold())

        for sentence_index, unit in enumerate(units):
            if examined >= MAX_EXPLORATION_CANDIDATES:
                break
            examined += 1
            text = normalize_evidence_whitespace(unit.text)
            context = construct_context_span(
                document,
                unit.start,
                unit.end,
            )
            if context.dependency_codes:
                context_dependencies_detected += 1
            evidence = normalize_evidence_whitespace(context.evidence_text)
            proposition = proposition_by_text.get(text.casefold())
            if proposition is None:
                proposition = next(
                    (
                        span
                        for span in proposition_spans
                        if span.start <= unit.start and unit.end <= span.end
                    ),
                    None,
                )
            if not context.complete:
                proposition_rejections += 1
                continue
            facts = _matching_structured_facts(
                text,
                evidence,
                evidence_span_id=public_chunk_id,
            )
            if _protected_fact_signature is not None:
                permitted_keys = {
                    "fact_type",
                    "period_a",
                    "value_a",
                    "period_b",
                    "value_b",
                    "direction",
                    "approximation_a",
                    "approximation_b",
                    "unit",
                }
                if (
                    not _protected_fact_signature
                    or set(_protected_fact_signature) - permitted_keys
                ):
                    facts = ()
                else:
                    facts = tuple(
                        fact
                        for fact in facts
                        if all(
                            getattr(fact, key, None) == expected
                            for key, expected in _protected_fact_signature.items()
                        )
                    )
            if not facts:
                typed_fact_rejections += 1
                continue
            evidence_assessment = assess_proposition(evidence)
            typed_termination = bool(
                facts
                and all(fact.fact_type == "comparison" for fact in facts)
                and evidence_assessment.sentence_start
                and not evidence_assessment.malformed_repetition
            )
            contextual_resolution = bool(
                context.expansion_type != "none"
                and context.complete
                and balanced_delimiters(evidence)
                and all(structured_fact_is_complete(fact) for fact in facts)
            )
            if (
                not (
                    evidence_assessment.complete
                    or typed_termination
                    or contextual_resolution
                )
                or _looks_like_title_or_fragment(evidence, metadata)
                or not quote_is_contiguous(text, document)
                or not quote_is_contiguous(evidence, document)
                or _candidate_near_recent(text, recent_assistant_texts)
            ):
                proposition_rejections += 1
                continue
            if context.expansion_type != "none":
                context_expansions += 1
            for fact_index, structured_fact in enumerate(facts):
                fact_text = normalize_evidence_whitespace(
                    (
                        text
                        if (
                            context.expansion_type != "none"
                            and structured_fact.fact_type != "comparison"
                            and structured_fact.evidence_text == evidence
                        )
                        else structured_fact.evidence_text or text
                    )
                )
                fingerprint = fact_fingerprint(
                    "\x1f".join((
                        structured_fact.fact_type,
                        structured_fact.subject,
                        str(structured_fact.period_a or ""),
                        str(structured_fact.value_a or ""),
                        str(structured_fact.period_b or ""),
                        str(structured_fact.value_b or ""),
                    ))
                )
                if fingerprint in seen_fingerprints:
                    continue
                score = _salience_score(fact_text, metadata)
                if context.dependency_codes:
                    score -= CONTEXT_DEPENDENCY_SALIENCE_PENALTY
                if (source_id, page) in avoid_pages:
                    score -= 1.5
                if not math.isfinite(score) or score < MIN_SALIENCE_SCORE:
                    continue
                seen_fingerprints.add(fingerprint)
                chunk_index = _int_value(metadata.get("chunk_index"), 1_000_000_000)
                page_key = page if isinstance(page, int) else 1_000_000_000
                tie_key = (
                    source_id,
                    page_key,
                    chunk_index,
                    sentence_index * 10 + fact_index,
                    fingerprint,
                )
                coordination_group_id = str(
                    getattr(proposition, "coordination_group_id", "") or ""
                )
                if coordination_group_id:
                    coordination_reconstructions += 1
                eligible.append(ExplorationCandidate(
                    text=fact_text,
                    hit_index=hit_index,
                    sentence_index=sentence_index,
                    score=score,
                    fact_fingerprint=fingerprint,
                    evidence_chunk_id=public_chunk_id,
                    source_id=source_id,
                    page=page,
                    section=str(metadata.get("section") or ""),
                    tie_key=tie_key,
                    hit=candidate_hit,
                    evidence_text=evidence,
                    context_dependency_codes=context.dependency_codes,
                    context_expansion_type=(
                        "coordination_group"
                        if coordination_group_id
                        else context.expansion_type
                    ),
                    evidence_sentence_count=max(
                        context.sentence_count,
                        int(getattr(proposition, "unit_count", 1) or 1),
                    ),
                    atomic_statistic_count=unit.statistic_count,
                    structured_fact=structured_fact,
                    proposition_start=int(
                        getattr(proposition, "start", context.evidence_start)
                    ),
                    proposition_end=int(
                        getattr(proposition, "end", context.evidence_end)
                    ),
                    sentence_start=bool(
                        getattr(proposition, "sentence_start", evidence_assessment.sentence_start)
                    ),
                    sentence_end=bool(
                        getattr(proposition, "sentence_end", evidence_assessment.sentence_end)
                    ),
                    paragraph_id=str(getattr(proposition, "paragraph_id", "") or ""),
                    clause_spans=tuple(
                        getattr(proposition, "clause_spans", ()) or ()
                    ),
                    coordination_group_id=coordination_group_id,
                    heading_context=bool(location.get("section")),
                    bullet_context=bool(re.match(r"^\s*[•▪◦*-]\s+", evidence)),
                    table_context=bool(re.match(r"^\s*\|", evidence)),
                    page_start=page if isinstance(page, int) else None,
                    page_end=page if isinstance(page, int) else None,
                    previous_authorized_unit=getattr(
                        proposition,
                        "previous_authorized_unit",
                        None,
                    ),
                    next_authorized_unit=getattr(
                        proposition,
                        "next_authorized_unit",
                        None,
                    ),
                    origin_mapping_revision=str(
                        metadata.get("canonical_mapping_revision") or ""
                    ),
                    origin_audit_key=str(
                        metadata.get("canonical_audit_key") or ""
                    ),
                ))
        if examined >= MAX_EXPLORATION_CANDIDATES:
            break

    eligible.sort(key=lambda candidate: (-candidate.score, candidate.tie_key))
    selected: list[ExplorationCandidate] = []
    ranked_distinct: list[ExplorationCandidate] = []
    selected_tokens: list[tuple[str, ...]] = []
    selected_chunks: set[str] = set()
    for candidate in eligible:
        if candidate.evidence_chunk_id in selected_chunks:
            continue
        tokens = _canonical_tokens(candidate.text)
        if any(_jaccard(tokens, previous) >= 0.62 for previous in selected_tokens):
            continue
        ranked_distinct.append(candidate)
        selected_tokens.append(tokens)
        selected_chunks.add(candidate.evidence_chunk_id)
        if len(selected) < item_limit:
            selected.append(candidate)

    if selected:
        reason = "salient_evidence_selected"
    elif not hits:
        reason = "exploration_scope_empty"
    elif examined == 0:
        reason = "exploration_no_complete_sentences"
    elif not eligible:
        reason = "exploration_no_eligible_facts"
    else:
        reason = "exploration_no_distinct_facts"
    return ExplorationSelection(
        candidates_examined=examined,
        eligible_candidates=len(eligible),
        selected=tuple(selected),
        reason_code=reason,
        context_dependencies_detected=context_dependencies_detected,
        context_expansions=context_expansions,
        canonical_rejections=canonical_rejections,
        proposition_rejections=proposition_rejections,
        typed_fact_rejections=typed_fact_rejections,
        coordination_reconstructions=coordination_reconstructions,
        ranked_candidates=tuple(ranked_distinct),
    )
