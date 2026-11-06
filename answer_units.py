"""Deterministic atomic factual units shared by selection and enforcement."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

# One lexical scalar may contain both grouped thousands and a decimal part.
# Keep this leaf fragment shared by structural/source/contract consumers.
GROUPED_DECIMAL_LITERAL_PATTERN = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"


_NUMBER_RE = re.compile(
    rf"(?<![\w])(?:{GROUPED_DECIMAL_LITERAL_PATTERN}|\d+(?:[.,]\d+)?)(?:\s*%)?(?![\w])"
)
_PERCENT_WORD_RE = re.compile(
    r"(?:%|אחוז(?:ים)?|\bpercent(?:age)?\b)",
    re.IGNORECASE,
)
_COMPARISON_RE = re.compile(
    r"(?:לעומת|בהשוואה|בין|מ-?\s*\d.+ל-?\s*\d|קיטון|ירידה|גידול|עליי?ה|"
    r"\b(?:compared|versus|between|from\s+[^.;]{0,80}\s+to|"
    r"decrease|decline|increase|growth|fell|grew)\b)",
    re.IGNORECASE,
)
_CONNECTIVE_RE = re.compile(
    r"\s+(?P<word>ו[\u05d0-\u05ea]{2,}|\b(?:and|while|whereas)\b)\s+",
    re.IGNORECASE,
)
_CITATION_RE = re.compile(r"\[\d{1,3}\]")
_LEADING_LIST_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_LETTER_RE = re.compile(r"[A-Za-z\u05d0-\u05ea]")


@dataclass(frozen=True)
class AtomicFactSpan:
    text: str
    start: int
    end: int
    statistic_count: int
    unit_kind: str


def _trimmed_span(value: str, start: int, end: int) -> tuple[int, int]:
    while start < end and value[start].isspace():
        start += 1
    while end > start and value[end - 1].isspace():
        end -= 1
    return start, end


def _statistics(value: str) -> list[re.Match[str]]:
    return list(_NUMBER_RE.finditer(_CITATION_RE.sub("", value or "")))


def _factlike(value: str) -> bool:
    return bool(_LETTER_RE.search(value or "") and _statistics(value))


def _top_level_split(value: str, delimiters: Iterable[str]) -> list[tuple[int, int]]:
    delimiter_set = set(delimiters)
    stack: list[str] = []
    pairs = {")": "(", "]": "[", "}": "{"}
    spans: list[tuple[int, int]] = []
    start = 0
    for index, character in enumerate(value):
        if character in "([{":
            stack.append(character)
            continue
        if character in ")]}":
            if stack and stack[-1] == pairs[character]:
                stack.pop()
            continue
        if stack or character not in delimiter_set:
            continue
        if (
            character == ","
            and index > 0
            and index + 1 < len(value)
            and value[index - 1].isdigit()
            and value[index + 1].isdigit()
        ):
            continue
        spans.append((start, index))
        start = index + 1
    spans.append((start, len(value)))
    return spans


def _split_connective(value: str, start: int, end: int) -> list[tuple[int, int]]:
    segment = value[start:end]
    output: list[tuple[int, int]] = []
    cursor = 0
    for match in _CONNECTIVE_RE.finditer(segment):
        left = segment[cursor:match.start()]
        right = segment[match.start():]
        if _factlike(left) and _factlike(right):
            output.append((start + cursor, start + match.start()))
            cursor = match.start()
    output.append((start + cursor, end))
    return output


def _distribution_spans(value: str) -> list[tuple[int, int]]:
    colon_positions = [
        index for index, character in enumerate(value)
        if character == ":" and len(_top_level_split(value[:index], ":")) == 1
    ]
    distribution_colon = next(
        (
            index for index in colon_positions
            if len(_statistics(value[index + 1:])) >= 2
        ),
        None,
    )
    if distribution_colon is None:
        base = _top_level_split(value, ";")
        output: list[tuple[int, int]] = []
        for start, end in base:
            output.extend(_split_connective(value, start, end))
        return output

    output: list[tuple[int, int]] = []
    prefix = value[:distribution_colon]
    prefix_spans = _split_connective(value, 0, distribution_colon)
    output.extend(
        span for span in prefix_spans
        if _factlike(value[span[0]:span[1]])
    )
    list_start = distribution_colon + 1
    for local_start, local_end in _top_level_split(value[list_start:], ",;"):
        start = list_start + local_start
        end = list_start + local_end
        for span in _split_connective(value, start, end):
            if _factlike(value[span[0]:span[1]]):
                output.append(span)
    return output


def atomic_fact_spans(value: str, *, absolute_start: int = 0) -> tuple[AtomicFactSpan, ...]:
    """Split a source sentence into contiguous principal factual units."""
    sentence = str(value or "")
    if not sentence.strip():
        return ()
    statistics = _statistics(sentence)
    if not statistics:
        start, end = _trimmed_span(sentence, 0, len(sentence))
        return (
            AtomicFactSpan(
                text=sentence[start:end],
                start=absolute_start + start,
                end=absolute_start + end,
                statistic_count=0,
                unit_kind="qualitative",
            ),
        ) if _LETTER_RE.search(sentence[start:end]) else ()

    if (
        len(statistics) <= 2
        and _COMPARISON_RE.search(sentence)
        and not (
            len(_PERCENT_WORD_RE.findall(sentence)) >= 2
            and ":" in sentence
        )
    ):
        start, end = _trimmed_span(sentence, 0, len(sentence))
        return (AtomicFactSpan(
            text=sentence[start:end],
            start=absolute_start + start,
            end=absolute_start + end,
            statistic_count=len(statistics),
            unit_kind="comparison",
        ),)

    if len(statistics) == 1:
        start, end = _trimmed_span(sentence, 0, len(sentence))
        return (AtomicFactSpan(
            text=sentence[start:end],
            start=absolute_start + start,
            end=absolute_start + end,
            statistic_count=1,
            unit_kind="numeric",
        ),)

    raw_spans = _distribution_spans(sentence)
    output: list[AtomicFactSpan] = []
    for raw_start, raw_end in raw_spans:
        start, end = _trimmed_span(sentence, raw_start, raw_end)
        if start >= end:
            continue
        text = sentence[start:end]
        count = len(_statistics(text))
        if not count or not _LETTER_RE.search(text):
            continue
        output.append(AtomicFactSpan(
            text=text,
            start=absolute_start + start,
            end=absolute_start + end,
            statistic_count=count,
            unit_kind="numeric",
        ))
    if output:
        return tuple(output)
    start, end = _trimmed_span(sentence, 0, len(sentence))
    return (AtomicFactSpan(
        text=sentence[start:end],
        start=absolute_start + start,
        end=absolute_start + end,
        statistic_count=len(statistics),
        unit_kind="numeric",
    ),)


def principal_fact_count(value: str) -> int:
    """Count source-like principal units; comparisons keep both endpoints."""
    text = _LEADING_LIST_RE.sub("", str(value or "")).strip()
    if not text:
        return 0
    lines = [line for line in text.splitlines() if line.strip()]
    count = 0
    for line in lines:
        sentence_parts = [
            part for part in re.split(r"(?<=[.!?׃])\s+(?=\S)", line)
            if part.strip()
        ]
        for sentence in sentence_parts or [line]:
            count += max(1, len(atomic_fact_spans(sentence)))
    return count
