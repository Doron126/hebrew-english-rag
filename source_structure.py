"""Source-bound sentence reconstruction across PDF blocks and index records."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any, Iterable, Mapping, Optional, Sequence

from proposition_quality import (
    assess_proposition,
    balanced_delimiters,
    comparison_endpoint_pair_count,
    extract_structured_comparisons,
    normalize_proposition_text,
)
from request_performance import timed_stage


SOURCE_STRUCTURE_REVISION = "source-structure-v2"

_BREADCRUMB_RE = re.compile(r"\A(?:[ \t]*\[[^\]\r\n]{1,500}\][ \t]*(?:\r?\n|$))+")
_BLOCK_BREAK_RE = re.compile(r"\n[ \t]*\n+")
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?׃…])(?:[ \t\r\n]+)(?=\S)")
_UNIT_BREAK_RE = re.compile(
    r"(?:(?<=[.!?׃…])(?:[ \t\r\n]+)(?=\S)|\n[ \t]*\n+)"
)
_SENTENCE_FINAL_RE = re.compile(r"[.!?׃…][\"'\u05f4\u2019\u201d)\]}]*\s*$")
_CONTINUATION_RE = re.compile(
    r"^\s*[\(\[\{),;:–—-]*\s*(?:"
    r"ו\s+(?=\S)|ו(?:ירידה|עלייה|גידול|קיטון|בשנת)|"
    r"ובשנת|ואילו|לעומת|בעוד|אך|אבל|כאשר|מתוכם|מהם|מהן|"
    r"and\b|but\b|while\b|and\s+in|whereas|compared\s+with"
    r")\b",
    re.IGNORECASE,
)
_BULLET_RE = re.compile(r"^\s*(?:[•▪◦*-]|\d+[.)])\s+")
_TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*$")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+")
_PERCENT_RE = re.compile(r"\d+(?:[.,]\d+)?\s*%")
_YEAR_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")
_LETTER_RE = re.compile(r"[A-Za-z\u05d0-\u05ea]")
_PUNCTUATION_ONLY_RE = re.compile(r"^[\s.,:;!?׃…()\[\]{}\-–—]+$")


@dataclass(frozen=True)
class SentenceSpan:
    start: int
    end: int
    text: str
    sentence_start: bool
    sentence_end: bool


@dataclass(frozen=True)
class PropositionSpan:
    start: int
    end: int
    text: str
    unit_count: int
    reason_codes: tuple[str, ...]
    sentence_start: bool = True
    sentence_end: bool = True
    paragraph_id: str = ""
    clause_spans: tuple[tuple[int, int], ...] = ()
    coordination_group_id: str = ""
    previous_authorized_unit: Optional[tuple[int, int]] = None
    next_authorized_unit: Optional[tuple[int, int]] = None


@dataclass(frozen=True)
class StructuralRecord:
    record_id: str
    source_id: str
    chunk_index: int
    page: Optional[int]
    text: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ReconstructedRecord:
    anchor_record_id: str
    text: str
    record_ids: tuple[str, ...]
    page_range: tuple[int, ...]
    cross_record_continuation: bool
    cross_page_continuation: bool
    sentence_spans: tuple[SentenceSpan, ...]
    reason_codes: tuple[str, ...]
    revision: str = SOURCE_STRUCTURE_REVISION


def strip_breadcrumb(value: str) -> str:
    return _BREADCRUMB_RE.sub("", str(value or ""), count=1).strip()


def _looks_heading(value: str) -> bool:
    text = value.strip()
    words = text.split()
    if _CONTINUATION_RE.match(text):
        return False
    return bool(
        _HEADING_RE.match(text)
        or (
            1 <= len(words) <= 3
            and 5 <= len(text) <= 120
            and not _SENTENCE_FINAL_RE.search(text)
            and not _PERCENT_RE.search(text)
            and len(_LETTER_RE.findall(text)) >= 4
            and (
                text.endswith(":")
                or not re.search(r"\d", text)
            )
        )
    )


def _unbalanced_delimiters(value: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for character in value:
        if character in pairs.values():
            stack.append(character)
        elif character in pairs:
            if not stack or stack.pop() != pairs[character]:
                return True
    return bool(stack)


def _semantic_boundary(value: str) -> bool:
    text = value.strip()
    return bool(
        not text
        or _BULLET_RE.match(text)
        or _TABLE_RE.match(text)
        or _TABLE_SEPARATOR_RE.match(text)
        or _looks_heading(text)
    )


def boundary_join_reason(previous: str, following: str) -> Optional[str]:
    """Return a safe reason for joining two adjacent source blocks."""
    left = normalize_proposition_text(previous)
    right = normalize_proposition_text(following)
    if _semantic_boundary(left) or _semantic_boundary(right):
        return None
    if _CONTINUATION_RE.match(right):
        return "continuation_marker"
    if _unbalanced_delimiters(left):
        return "unbalanced_delimiter"
    if _SENTENCE_FINAL_RE.search(left):
        return None

    left_years = len(_YEAR_RE.findall(left))
    right_years = len(_YEAR_RE.findall(right))
    left_values = len(_PERCENT_RE.findall(left))
    right_values = len(_PERCENT_RE.findall(right))
    if (
        left_years >= 1
        and right_years >= 1
        and left_values >= 1
        and right_values >= 1
    ):
        return "split_year_value_comparison"

    left_assessment = assess_proposition(left)
    combined_assessment = assess_proposition(f"{left} {right}")
    resolvable_reasons = {
        "mid_clause_end",
        "comparison_period_missing",
        "comparison_value_missing",
        "percentage_population_missing",
        "percentage_metric_missing",
        "incomplete_predicate",
    }
    if (
        set(left_assessment.reason_codes) & resolvable_reasons
        and len(combined_assessment.reason_codes) < len(left_assessment.reason_codes)
        and combined_assessment.sentence_end
    ):
        return "proposition_completion"
    return None


@timed_stage("sentence_reconstruction_ms")
def reconstruct_soft_blocks(value: str) -> tuple[str, tuple[str, ...]]:
    text = str(value or "").strip()
    if not text:
        return "", ()
    blocks = [
        match.group(0).strip()
        for match in re.finditer(r".+?(?=\n[ \t]*\n+|\Z)", text, re.DOTALL)
        if match.group(0).strip()
    ]
    if not blocks:
        return text, ()
    output = blocks[0]
    reasons: list[str] = []
    for block in blocks[1:]:
        previous = output.rsplit("\n\n", 1)[-1]
        reason = boundary_join_reason(previous, block)
        if reason:
            output = f"{output.rstrip()} {block.lstrip()}"
            reasons.append(reason)
        else:
            output = f"{output.rstrip()}\n\n{block.lstrip()}"
    return output, tuple(reasons)


def sentence_spans(value: str) -> tuple[SentenceSpan, ...]:
    text = str(value or "")
    spans: list[SentenceSpan] = []
    cursor = 0
    for match in _UNIT_BREAK_RE.finditer(text):
        start = cursor
        end = match.start()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            candidate = text[start:end]
            spans.append(SentenceSpan(
                start,
                end,
                candidate,
                not bool(_CONTINUATION_RE.match(candidate)),
                bool(_SENTENCE_FINAL_RE.search(candidate)),
            ))
        cursor = match.end()
    start = cursor
    end = len(text)
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    if start < end:
        candidate = text[start:end]
        spans.append(SentenceSpan(
            start,
            end,
            candidate,
            not bool(_CONTINUATION_RE.match(candidate)),
            bool(_SENTENCE_FINAL_RE.search(candidate)),
        ))
    return tuple(spans)


@timed_stage("sentence_reconstruction_ms")
def complete_proposition_spans(
    value: str,
    *,
    maximum_units: int = 12,
    maximum_chars: int = 900,
) -> tuple[PropositionSpan, ...]:
    """Return the smallest complete factual windows in source order.

    PDF layout blocks are not semantic sentence boundaries. This bounded
    search may join adjacent units only when the combined source text passes
    the proposition validator. It never crosses a bullet, heading, table, or
    other explicit structural boundary.
    """
    reconstructed, reconstruction_reasons = reconstruct_soft_blocks(value)
    units = sentence_spans(reconstructed)
    output: list[PropositionSpan] = []
    unit_limit = max(1, min(int(maximum_units), 12))
    char_limit = max(80, min(int(maximum_chars), 2_000))

    for start_index, first in enumerate(units):
        if _semantic_boundary(first.text):
            continue
        for end_index in range(start_index, min(len(units), start_index + unit_limit)):
            unit = units[end_index]
            if end_index > start_index and _semantic_boundary(unit.text):
                break
            start = first.start
            end = unit.end
            candidate = normalize_proposition_text(reconstructed[start:end])
            if len(candidate) > char_limit:
                break
            assessment = assess_proposition(candidate)
            endpoint_pairs = comparison_endpoint_pair_count(candidate)
            typed_comparisons = extract_structured_comparisons(candidate)
            typed_complete = bool(
                endpoint_pairs >= 2
                and endpoint_pairs % 2 == 0
                and len(typed_comparisons) * 2 == endpoint_pairs
            )
            typed_termination = bool(
                typed_complete
                and assessment.sentence_start
                and not assessment.malformed_repetition
                and balanced_delimiters(candidate)
            )
            if not (assessment.complete or typed_termination):
                continue
            local_clause_breaks = [
                match.start()
                for match in re.finditer(
                    r"\s+(?:ו(?:אילו|ירידה|עלייה|גידול|קיטון)|"
                    r"מצד\s+שני|בעוד\s+ש|לעומת|whereas|while|but|and)\b",
                    candidate,
                    re.IGNORECASE,
                )
            ]
            clause_bounds = [0, *local_clause_breaks, len(candidate)]
            clause_spans = tuple(
                (start + clause_bounds[index], start + clause_bounds[index + 1])
                for index in range(len(clause_bounds) - 1)
                if clause_bounds[index] < clause_bounds[index + 1]
            )
            coordination_group_id = (
                hashlib.sha256(
                    f"{start}:{end}:{candidate}".encode("utf-8")
                ).hexdigest()[:24]
                if len(clause_spans) > 1 or len(typed_comparisons) > 1
                else ""
            )
            output.append(PropositionSpan(
                start=start,
                end=end,
                text=candidate,
                unit_count=end_index - start_index + 1,
                reason_codes=tuple(reconstruction_reasons) + (
                    ("typed_comparison_termination",)
                    if not assessment.sentence_end
                    else ()
                ) + (
                    ("coordination_group_reconstructed",)
                    if coordination_group_id
                    else ()
                ),
                sentence_start=assessment.sentence_start,
                sentence_end=assessment.sentence_end or typed_termination,
                paragraph_id=hashlib.sha256(
                    f"{start}:{end}".encode("ascii")
                ).hexdigest()[:20],
                clause_spans=clause_spans or ((start, end),),
                coordination_group_id=coordination_group_id,
                previous_authorized_unit=(
                    (units[start_index - 1].start, units[start_index - 1].end)
                    if start_index > 0
                    else None
                ),
                next_authorized_unit=(
                    (units[end_index + 1].start, units[end_index + 1].end)
                    if end_index + 1 < len(units)
                    else None
                ),
            ))
            break
    return tuple(output)


def complete_sentence_for_focus(
    value: str,
    focus: str,
) -> Optional[SentenceSpan]:
    reconstructed, _reasons = reconstruct_soft_blocks(value)
    needle = normalize_proposition_text(focus)
    if not needle:
        return None
    normalized = normalize_proposition_text(reconstructed)
    normalized_position = normalized.find(needle)
    if normalized_position < 0:
        return None
    # Whitespace normalization changes offsets.  Find the same focus using a
    # bounded flexible-space expression so the returned source span is exact.
    pattern = re.compile(
        r"\s+".join(re.escape(part) for part in needle.split())
    )
    match = pattern.search(reconstructed)
    if not match:
        return None
    for span in sentence_spans(reconstructed):
        if span.start <= match.start() and match.end() <= span.end:
            assessment = assess_proposition(span.text)
            if (
                span.sentence_start
                and span.sentence_end
                and assessment.explicit_subject
                and assessment.complete_predicate
            ):
                return span
            return None
    return None


def _maximum_overlap(left: str, right: str, limit: int = 512) -> int:
    left_value = left.rstrip()
    right_value = right.lstrip()
    maximum = min(len(left_value), len(right_value), max(0, int(limit)))
    for size in range(maximum, 15, -1):
        if left_value[-size:] == right_value[:size]:
            return size
    return 0


def _merge_record_text(left: str, right: str) -> str:
    overlap = _maximum_overlap(left, right)
    if overlap:
        return f"{left.rstrip()}{right.lstrip()[overlap:]}"
    return f"{left.rstrip()}\n\n{right.lstrip()}"


def structural_records(hits: Iterable[Mapping[str, Any]]) -> tuple[StructuralRecord, ...]:
    output: list[StructuralRecord] = []
    for raw in hits:
        if not isinstance(raw, Mapping):
            continue
        metadata = raw.get("metadata")
        if not isinstance(metadata, Mapping):
            metadata = {}
        record_id = str(
            metadata.get("chunk_id")
            or raw.get("id")
            or ""
        )
        source_id = str(metadata.get("source_id") or "")
        try:
            chunk_index = int(metadata.get("chunk_index"))
        except (TypeError, ValueError, OverflowError):
            continue
        page_value = metadata.get("page")
        page = (
            int(page_value)
            if isinstance(page_value, int) and not isinstance(page_value, bool)
            else None
        )
        text = strip_breadcrumb(str(raw.get("document") or ""))
        if not record_id or not source_id or not text:
            continue
        output.append(StructuralRecord(
            record_id,
            source_id,
            chunk_index,
            page,
            text,
            metadata,
        ))
    return tuple(sorted(
        output,
        key=lambda record: (record.source_id, record.chunk_index, record.record_id),
    ))


def reconstruct_record(
    records: Sequence[StructuralRecord],
    anchor_record_id: str,
    *,
    maximum_records: int = 3,
) -> Optional[ReconstructedRecord]:
    anchor_index = next(
        (
            index for index, record in enumerate(records)
            if record.record_id == anchor_record_id
        ),
        None,
    )
    if anchor_index is None:
        return None
    anchor = records[anchor_index]
    selected = [anchor]
    reasons: list[str] = []

    if anchor_index > 0 and len(selected) < maximum_records:
        previous = records[anchor_index - 1]
        if (
            previous.source_id == anchor.source_id
            and previous.chunk_index + 1 == anchor.chunk_index
        ):
            previous_tail = previous.text.rsplit("\n\n", 1)[-1]
            current_head = anchor.text.split("\n\n", 1)[0]
            reason = boundary_join_reason(previous_tail, current_head)
            if reason:
                selected.insert(0, previous)
                reasons.append(f"previous:{reason}")

    if anchor_index + 1 < len(records) and len(selected) < maximum_records:
        following = records[anchor_index + 1]
        if (
            following.source_id == anchor.source_id
            and anchor.chunk_index + 1 == following.chunk_index
        ):
            current_tail = anchor.text.rsplit("\n\n", 1)[-1]
            following_head = following.text.split("\n\n", 1)[0]
            reason = boundary_join_reason(current_tail, following_head)
            if reason:
                selected.append(following)
                reasons.append(f"next:{reason}")

    combined = selected[0].text
    for record in selected[1:]:
        combined = _merge_record_text(combined, record.text)
    reconstructed, internal_reasons = reconstruct_soft_blocks(combined)
    reasons.extend(f"internal:{reason}" for reason in internal_reasons)
    pages = tuple(dict.fromkeys(
        record.page for record in selected if record.page is not None
    ))
    return ReconstructedRecord(
        anchor_record_id=anchor.record_id,
        text=reconstructed,
        record_ids=tuple(record.record_id for record in selected),
        page_range=pages,
        cross_record_continuation=len(selected) > 1,
        cross_page_continuation=len(set(pages)) > 1,
        sentence_spans=sentence_spans(reconstructed),
        reason_codes=tuple(reasons),
    )


def reconstruct_hit_documents(
    hits: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Return hit copies with authorized same-source structure reconstructed."""
    records = structural_records(hits)
    by_id = {record.record_id: record for record in records}
    output: list[dict[str, Any]] = []
    for raw in hits:
        candidate = dict(raw)
        metadata = dict(candidate.get("metadata") or {})
        record_id = str(metadata.get("chunk_id") or candidate.get("id") or "")
        record = by_id.get(record_id)
        reconstruction = reconstruct_record(records, record_id) if record else None
        if reconstruction:
            candidate["document"] = reconstruction.text
            metadata["_source_structure_revision"] = reconstruction.revision
            metadata["_reconstructed_record_ids"] = reconstruction.record_ids
            metadata["_page_range"] = reconstruction.page_range
            metadata["_cross_record_continuation"] = (
                reconstruction.cross_record_continuation
            )
            metadata["_cross_page_continuation"] = (
                reconstruction.cross_page_continuation
            )
            metadata["_sentence_start"] = all(
                span.sentence_start for span in reconstruction.sentence_spans
            )
            metadata["_sentence_end"] = all(
                span.sentence_end for span in reconstruction.sentence_spans
            )
        candidate["metadata"] = metadata
        output.append(candidate)
    return output
