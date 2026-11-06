"""Deterministic requested-cardinality enforcement for grounded answers."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence

try:
    from .answer_units import atomic_fact_spans
except ImportError:
    from answer_units import atomic_fact_spans


MAX_REQUESTED_UNITS = 8

_CITATION_RE = re.compile(r"\[(\d{1,3})\]")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_SEGMENT_RE = re.compile(
    r"(?:\r?\n)+|(?<=[.!?׃])\s+(?=(?:[-*•]|\d+[.)])?\s*[A-Za-z\u0590-\u05ff])"
)
_TERM_RE = re.compile(r"[A-Za-z0-9\u0590-\u05ff]{2,}")
_NUMBER_RE = re.compile(
    r"(?<![\w])(?:\d{1,3}(?:,\d{3})+|\d+(?:[.,]\d+)?)(?:\s*%)?(?![\w])"
)
_PREDICATE_RE = re.compile(
    r"(?:הוא|היא|הם|הן|יש|אין|היה|היו|מעסיק|נמצא|מראה|מצביע|"
    r"מהווה|כולל|עלה|ירד|גדל|הגדיל|פחת|הפחית|חל|נדרש|צפוי|"
    r"\b(?:is|are|was|were|has|have|employs?|found|shows?|indicates?|"
    r"represents?|includes?|increased?|decreased?|reduced?|fell|grew|"
    r"requires?)\b)",
    re.IGNORECASE,
)
_INDEPENDENT_CONNECTIVE_RE = re.compile(
    r"\s+(?:and|while|whereas)\s+|\s+(?=ו[\u05d0-\u05ea]{2,})",
    re.IGNORECASE,
)
_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "for", "in",
    "on", "at", "by", "with", "and", "or", "that", "this", "from", "as",
    "הוא", "היא", "הם", "הן", "זה", "זו", "זאת", "של", "על", "עם", "את",
    "גם", "או", "לפי", "מן", "מתוך", "כי",
})


@dataclass(frozen=True)
class PrincipalAnswerUnit:
    text: str
    citation_ids: tuple[int, ...]
    fingerprint: str
    order: int


@dataclass(frozen=True)
class CardinalityResult:
    answer: str
    generated_units: int
    retained_units: int
    discarded_units: int
    correction_mode: str
    complete: bool
    retained_citation_ids: tuple[int, ...]


def _normalized_terms(value: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in (
            match.group(0).casefold()
            for match in _TERM_RE.finditer(value or "")
        )
        if token not in _STOPWORDS
    )


def _fingerprint(value: str) -> str:
    terms = _normalized_terms(_CITATION_RE.sub("", value or ""))
    numbers = tuple(
        match.group(0).replace(" ", "")
        for match in _NUMBER_RE.finditer(_CITATION_RE.sub("", value or ""))
    )
    return "\x1f".join((*terms, *numbers))


def _independent_qualitative_clauses(value: str) -> list[str]:
    """Split a conjunction only when both sides contain full predicates."""
    matches = list(_INDEPENDENT_CONNECTIVE_RE.finditer(value or ""))
    if not matches:
        return [value]
    output: list[str] = []
    start = 0
    for match in matches:
        left = value[start:match.start()].strip(" \t,;")
        right = value[match.end():].strip(" \t,;")
        if _PREDICATE_RE.search(left) and _PREDICATE_RE.search(right):
            if left:
                output.append(left)
            start = match.end()
    remainder = value[start:].strip(" \t,;")
    if remainder:
        output.append(remainder)
    return output or [value]


def segment_principal_answer_units(answer: str) -> tuple[PrincipalAnswerUnit, ...]:
    """Segment principal facts while preserving citation ownership.

    A two-endpoint comparison remains one unit through ``atomic_fact_spans``.
    Multi-statistic distributions and independently predicated conjunctions
    become separate units even when generation placed them in one sentence.
    """
    # A model may put markers immediately after terminal punctuation. Move
    # those markers immediately before the punctuation only for segmentation;
    # factual text and marker ownership remain unchanged.
    value = re.sub(
        r"([.!?׃])\s*((?:\[\d{1,3}\]\s*)+)(?=[A-Za-z\u0590-\u05ff])",
        lambda match: f" {match.group(2).strip()}{match.group(1)} ",
        str(answer or ""),
    )
    units: list[PrincipalAnswerUnit] = []
    order = 0
    for raw_segment in _SEGMENT_RE.split(value):
        segment = _LIST_PREFIX_RE.sub("", raw_segment or "").strip()
        if not segment:
            continue
        citation_ids = tuple(dict.fromkeys(
            int(match.group(1)) for match in _CITATION_RE.finditer(segment)
        ))
        body = _CITATION_RE.sub("", segment).strip()
        if not _TERM_RE.search(body):
            continue
        atomic = atomic_fact_spans(body)
        pieces = [item.text for item in atomic] if atomic else [body]
        if len(pieces) == 1 and not _NUMBER_RE.search(pieces[0]):
            pieces = _independent_qualitative_clauses(pieces[0])
        for piece in pieces:
            text = re.sub(r"\s+", " ", piece).strip(" \t,;")
            if not _TERM_RE.search(text):
                continue
            units.append(PrincipalAnswerUnit(
                text=text,
                citation_ids=citation_ids,
                fingerprint=_fingerprint(text),
                order=order,
            ))
            order += 1
    return tuple(units)


def principal_answer_unit_count(answer: str) -> int:
    return len(segment_principal_answer_units(answer))


def _match_score(unit: PrincipalAnswerUnit, focus: str, citation_id: int) -> float:
    unit_terms = set(_normalized_terms(unit.text))
    focus_terms = set(_normalized_terms(focus))
    overlap = (
        len(unit_terms & focus_terms) / max(1, len(unit_terms | focus_terms))
    )
    unit_numbers = set(_NUMBER_RE.findall(unit.text))
    focus_numbers = set(_NUMBER_RE.findall(focus or ""))
    number_score = (
        len(unit_numbers & focus_numbers) / max(1, len(focus_numbers))
        if focus_numbers
        else 0.0
    )
    citation_score = 4.0 if citation_id in unit.citation_ids else -4.0
    return citation_score + (number_score * 3.0) + overlap


def _render_selected(
    selected: Sequence[tuple[PrincipalAnswerUnit, int]],
) -> str:
    lines: list[str] = []
    multiple = len(selected) > 1
    for output_index, (unit, citation_id) in enumerate(selected, start=1):
        text = unit.text.rstrip()
        if not re.search(r"[.!?׃]$", text):
            text += "."
        line = f"{text} [{citation_id}]"
        lines.append(f"{output_index}. {line}" if multiple else line)
    return "\n".join(lines)


def enforce_requested_cardinality(
    answer: str,
    requested_items: int,
    *,
    permitted_focuses: Iterable[str] = (),
) -> CardinalityResult:
    """Retain exactly one grounded principal unit per permitted proposition."""
    try:
        requested = max(1, min(int(requested_items), MAX_REQUESTED_UNITS))
    except (TypeError, ValueError, OverflowError):
        requested = 1
    units = list(segment_principal_answer_units(answer))
    focuses = tuple(str(value or "").strip() for value in permitted_focuses)
    if focuses:
        focuses = focuses[:requested]

    selected: list[tuple[PrincipalAnswerUnit, int]] = []
    used_orders: set[int] = set()
    used_fingerprints: set[str] = set()
    targets = focuses if len(focuses) == requested else tuple("" for _ in range(requested))
    for target_index, focus in enumerate(targets, start=1):
        candidates = [
            unit for unit in units
            if unit.order not in used_orders
            and unit.fingerprint
            and unit.fingerprint not in used_fingerprints
            and (
                target_index in unit.citation_ids
                if focuses
                else bool(unit.citation_ids)
            )
        ]
        if not candidates:
            break
        chosen = max(
            candidates,
            key=lambda unit: (
                _match_score(unit, focus, target_index),
                -unit.order,
            ),
        )
        selected.append((chosen, target_index if focuses else chosen.citation_ids[0]))
        used_orders.add(chosen.order)
        used_fingerprints.add(chosen.fingerprint)

    complete = len(selected) == requested
    generated = len(units)
    if not complete:
        return CardinalityResult(
            answer="",
            generated_units=generated,
            retained_units=len(selected),
            discarded_units=max(0, generated - len(selected)),
            correction_mode="insufficient_supported_units",
            complete=False,
            retained_citation_ids=tuple(item[1] for item in selected),
        )

    no_change = (
        generated == requested
        and all(unit.order == index for index, (unit, _citation) in enumerate(selected))
    )
    rendered = answer.strip() if no_change else _render_selected(selected)
    mode = "none" if no_change else "deterministic_reduction"
    return CardinalityResult(
        answer=rendered,
        generated_units=generated,
        retained_units=len(selected),
        discarded_units=max(0, generated - len(selected)),
        correction_mode=mode,
        complete=True,
        retained_citation_ids=tuple(item[1] for item in selected),
    )
