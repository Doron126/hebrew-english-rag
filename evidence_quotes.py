"""Deterministic, source-bound evidence quotation selection.

The language model never participates in this module.  Callers provide the
exact retrieved evidence text and may supply the same deterministic support
predicate used to accept the grounded claim. Returned text is one contiguous
source span, or two independently bounded source spans in source order when a
claim needs both local context and a distant value statement.
"""

from __future__ import annotations

from dataclasses import dataclass
import base64
import hashlib
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

try:
    from .config_env import (
        hebrew_token_variants,
        normalize_search_text,
    )
    from .source_access import (
        location_metadata,
        open_source_file,
        source_metadata_complete,
        source_sha256_matches_metadata,
        source_stat_matches_metadata,
    )
    from .request_performance import timed_stage
except ImportError:
    from config_env import (
        hebrew_token_variants,
        normalize_search_text,
    )
    from source_access import (
        location_metadata,
        open_source_file,
        source_metadata_complete,
        source_sha256_matches_metadata,
        source_stat_matches_metadata,
    )
    from request_performance import timed_stage


PREFERRED_QUOTE_MIN_WORDS = 15
PREFERRED_QUOTE_MAX_WORDS = 50
HARD_QUOTE_MAX_WORDS = 80
HARD_QUOTE_MAX_CHARS = 800
ANSWER_QUOTE_MAX_WORDS = 240
ANSWER_QUOTE_MAX_CHARS = 2400

_BOUNDARY_ELLIPSIS = "…"
_BREADCRUMB_RE = re.compile(r"\A(?:[ \t]*\[[^\]\r\n]{1,500}\][ \t]*(?:\r?\n|$))+")
_BLOCK_BREAK_RE = re.compile(
    r"(?:\r?\n[ \t]*\r?\n+)|(?=\r?\n[ \t]*[•▪◦*-][ \t]+)"
)
_SENTENCE_BREAK_RE = re.compile(r"(?<=[.!?׃])(?:[ \t\r\n]+)(?=\S)")
_LEADING_BULLET_RE = re.compile(r"\A[ \t\r\n]*[•▪◦*-][ \t]*")
_WORD_RE = re.compile(r"\S+")
_TERM_RE = re.compile(r"[A-Za-z0-9\u0590-\u05FF]{2,}")
_NUMERIC_TOKEN_RE = re.compile(r"(?<!\d)\d[\d,]*(?:\.\d+)?")
_CITATION_RE = re.compile(r"\[\d{1,3}\]")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")
_PATH_DISCLOSURE_RE = re.compile(
    r"(?:/(?:home|srv|var/lib|opt|mnt)/|"
    r"(?<![A-Za-z])[A-Za-z]:[\\/]|\\\\[^\\\s]+\\)",
    re.IGNORECASE,
)
_NEGATION_TERMS = {
    "not", "no", "never", "without", "cannot",
    "לא", "אין", "אינו", "אינה", "ללא", "בלי", "טרם",
}
_LOW_VALUE_TERMS = {
    "the", "a", "an", "is", "are", "was", "were", "of", "to", "for", "in",
    "on", "at", "by", "with", "and", "or", "that", "this", "according",
    "source", "sources", "regarding",
    "הוא", "היא", "הם", "הן", "זה", "זו", "זאת", "של", "על", "עם", "לפי",
    "המקור", "המקורות", "גם", "או", "את", "בנושא",
}


@dataclass(frozen=True)
class EvidenceQuote:
    text: str
    language: str
    truncated: bool
    word_count: int


@dataclass(frozen=True)
class _Span:
    start: int
    end: int
    sentence_count: int
    block_index: int


def normalize_evidence_whitespace(value: str) -> str:
    """Normalize only whitespace for display and substring verification."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def quote_core(value: str) -> str:
    """Remove only selector-added boundary ellipses from a displayed quote."""
    core = str(value or "").strip()
    if core.startswith(_BOUNDARY_ELLIPSIS):
        core = core[len(_BOUNDARY_ELLIPSIS):].lstrip()
    if core.endswith(_BOUNDARY_ELLIPSIS):
        core = core[:-len(_BOUNDARY_ELLIPSIS)].rstrip()
    return core


def quote_is_contiguous(value: str, evidence_text: str) -> bool:
    """Prove one span, or two ordered spans joined by a selector ellipsis."""
    normalized_quote = normalize_evidence_whitespace(quote_core(value))
    normalized_evidence = normalize_evidence_whitespace(evidence_text)
    parts = [
        part.strip()
        for part in re.split(r"\s+…\s+", normalized_quote)
        if part.strip()
    ]
    if not parts or len(parts) > 2:
        return False
    cursor = 0
    for part in parts:
        position = normalized_evidence.find(part, cursor)
        if position < 0:
            return False
        cursor = position + len(part)
    return True


def workbook_field_excerpt_matches(value: str, fields: Iterable[str]) -> bool:
    """Check a displayed workbook field or explicitly marked field prefix.

    The excerpt is a navigation/display aid; the complete authenticated row
    remains the evidence for all claim checks. A clipped field must begin at
    the field boundary, end at a word boundary, and visibly carry an ellipsis.
    """
    display = str(value or '').strip()
    core = normalize_evidence_whitespace(quote_core(display))
    if not core or not quote_has_balanced_delimiters(display):
        return False
    for field in fields:
        source = normalize_evidence_whitespace(field)
        if source == core:
            return True
        if (display.endswith(_BOUNDARY_ELLIPSIS) and len(core) >= 40
                and len(core) <= 320 and source.startswith(core)
                and len(source) > len(core) and source[len(core)].isspace()):
            return True
    return False


def quote_has_balanced_delimiters(value: str) -> bool:
    """Reject source spans cut through parentheses or other paired delimiters."""
    pairs = {")": "(", "]": "[", "}": "{"}
    for part in re.split(r"\s+…\s+", quote_core(value)):
        stack: list[str] = []
        for character in part:
            if character in pairs.values():
                stack.append(character)
            elif character in pairs:
                if not stack or stack.pop() != pairs[character]:
                    return False
        if stack:
            return False
    return True


def evidence_quote_language(value: str) -> str:
    """Return a conservative language label without translating the quote."""
    hebrew = len(re.findall(r"[\u05D0-\u05EA]", value or ""))
    latin = len(re.findall(r"[A-Za-z]", value or ""))
    if hebrew >= 2 and latin >= 2:
        return "mixed"
    if hebrew >= 2:
        return "he"
    if latin >= 2:
        return "en"
    return "und"


def evidence_source_matches(
    metadata: Any,
    *,
    effective_tenant: str,
    citation_source_id: Optional[str],
    citation_page: Optional[int],
) -> bool:
    """Bind a quote to the citation's tenant, opaque source, and page."""
    if not source_metadata_complete(metadata, effective_tenant):
        return False
    if str(metadata.get("tenant_id") or "") != effective_tenant:
        return False
    if not citation_source_id or metadata.get("source_id") != citation_source_id:
        return False
    indexed_page = location_metadata(metadata).get("page")
    return indexed_page == citation_page


@timed_stage("source_integrity_ms")
def verify_evidence_source_integrity(
    document_root: str | Path,
    tenant_id: str,
    metadata: Any,
) -> bool:
    """Securely open and fully verify the exact indexed source generation."""
    if not source_metadata_complete(metadata, tenant_id):
        return False
    source_handle = None
    try:
        _relative, source_handle, source_stat = open_source_file(
            document_root,
            tenant_id,
            str(metadata.get("source_relative_path") or ""),
        )
        return (
            source_stat_matches_metadata(metadata, source_stat)
            and source_sha256_matches_metadata(
                metadata,
                source_handle,
                enabled=True,
            )
        )
    except (OSError, RuntimeError, TypeError, ValueError):
        return False
    finally:
        if source_handle is not None:
            source_handle.close()


def opaque_evidence_chunk_id(
    source_id: str,
    chunk_id: str,
    page: Optional[int],
) -> str:
    """Return a bounded opaque public identifier, never a storage path."""
    material = f"{source_id}\0{chunk_id}\0{page or ''}".encode("utf-8")
    digest = hashlib.sha256(material).digest()[:18]
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _trimmed_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    bullet = _LEADING_BULLET_RE.match(text[start:end])
    if bullet:
        start += bullet.end()
    while start < end and text[start].isspace():
        start += 1
    return start, end


def _sentence_spans(text: str) -> list[_Span]:
    content_start = 0
    breadcrumb = _BREADCRUMB_RE.match(text)
    if breadcrumb:
        content_start = breadcrumb.end()

    spans: list[_Span] = []
    block_start = content_start
    block_index = 0
    block_ranges: list[tuple[int, int, int]] = []
    for match in _BLOCK_BREAK_RE.finditer(text, content_start):
        block_ranges.append((block_start, match.start(), block_index))
        block_start = match.end() if match.end() > match.start() else match.start() + 1
        block_index += 1
    block_ranges.append((block_start, len(text), block_index))

    for raw_start, raw_end, current_block in block_ranges:
        start, end = _trimmed_bounds(text, raw_start, raw_end)
        if start >= end:
            continue
        sentence_start = start
        block_sentence_spans: list[tuple[int, int]] = []
        for match in _SENTENCE_BREAK_RE.finditer(text, start, end):
            sentence_end = match.start()
            clean_start, clean_end = _trimmed_bounds(
                text,
                sentence_start,
                sentence_end,
            )
            if clean_start < clean_end:
                block_sentence_spans.append((clean_start, clean_end))
            sentence_start = match.end()
        clean_start, clean_end = _trimmed_bounds(text, sentence_start, end)
        if clean_start < clean_end:
            block_sentence_spans.append((clean_start, clean_end))

        for sentence_start, sentence_end in block_sentence_spans:
            spans.append(_Span(
                sentence_start,
                sentence_end,
                1,
                current_block,
            ))
        for index in range(len(block_sentence_spans) - 1):
            pair_start = block_sentence_spans[index][0]
            pair_end = block_sentence_spans[index + 1][1]
            spans.append(_Span(pair_start, pair_end, 2, current_block))

    # Paragraph and bullet boundaries are meaningful, but bounded context can
    # still span two immediately adjacent source units. Keep the raw interval
    # contiguous; never skip an intervening block or sentence.
    single_spans = sorted(
        (span for span in spans if span.sentence_count == 1),
        key=lambda span: (span.start, span.end),
    )
    existing_pairs = {
        (span.start, span.end)
        for span in spans
        if span.sentence_count == 2
    }
    for first, second in zip(single_spans, single_spans[1:]):
        pair_key = (first.start, second.end)
        if first.block_index == second.block_index or pair_key in existing_pairs:
            continue
        spans.append(_Span(first.start, second.end, 2, first.block_index))
    # A parenthetical/table context can legitimately cross more than one unit.
    # Search only small adjacent windows; downstream hard bounds still apply.
    existing_ranges = {(span.start, span.end) for span in spans}
    for window_size in (3, 4):
        for index in range(len(single_spans) - window_size + 1):
            first = single_spans[index]
            last = single_spans[index + window_size - 1]
            span_key = (first.start, last.end)
            if span_key in existing_ranges:
                continue
            spans.append(
                _Span(first.start, last.end, window_size, first.block_index)
            )
            existing_ranges.add(span_key)
    return spans


def complete_evidence_sentences(value: str) -> list[str]:
    """Return de-duplicated complete sentence/bullet candidates in source order."""
    text = str(value or "")
    output: list[str] = []
    seen: set[str] = set()
    for span in _sentence_spans(text):
        if span.sentence_count != 1:
            continue
        candidate = normalize_evidence_whitespace(text[span.start:span.end])
        if not candidate or _looks_like_boilerplate(candidate):
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        output.append(candidate)
    return output


def _claim_variants(claims: Iterable[str]) -> list[str]:
    variants: list[str] = []
    for claim in claims:
        value = _LIST_PREFIX_RE.sub("", _CITATION_RE.sub("", str(claim or ""))).strip()
        if not value:
            continue
        variants.append(value)
        if ":" in value:
            prefix, tail = value.split(":", 1)
            if re.search(r"(?:\bregarding\b|בנושא)", prefix, re.IGNORECASE) and tail.strip():
                variants.append(tail.strip())
    return list(dict.fromkeys(variants))


def _term_groups(value: str) -> list[set[str]]:
    groups: list[set[str]] = []
    for raw in _TERM_RE.findall(normalize_search_text(value or "")):
        token = raw.casefold()
        if len(token) < 2:
            continue
        if all("\u0590" <= character <= "\u05ff" for character in token):
            variants = {
                candidate.casefold()
                for candidate in hebrew_token_variants(token)
                if len(candidate) >= 2
            }
        else:
            variants = {token}
        if not variants:
            continue
        if token in _LOW_VALUE_TERMS and token not in _NEGATION_TERMS:
            continue
        groups.append(variants)
    return groups


def _coverage(candidate: str, claim: str) -> tuple[int, int]:
    candidate_tokens = {
        token.casefold()
        for token in _TERM_RE.findall(normalize_search_text(candidate or ""))
    }
    matched = 0
    groups = _term_groups(claim)
    for group in groups:
        if group & candidate_tokens:
            matched += 1
    return matched, len(groups)


def _looks_like_boilerplate(value: str) -> bool:
    text = normalize_evidence_whitespace(value)
    if not text or _PATH_DISCLOSURE_RE.search(text):
        return True
    if re.fullmatch(r"[\d\s./|_-]+", text):
        return True
    if re.fullmatch(r"(?:page|עמוד|עמ׳)\s+\d{1,7}", text, re.IGNORECASE):
        return True
    if text.casefold() in {
        "table of contents", "list of figures", "list of tables",
        "תוכן עניינים", "רשימת איורים", "רשימת טבלאות",
    }:
        return True
    letters = len(re.findall(r"[A-Za-z\u05D0-\u05EA]", text))
    return letters < 8


def _word_spans(value: str) -> list[re.Match[str]]:
    return list(_WORD_RE.finditer(value))


def _numeric_tokens(value: str) -> set[str]:
    return {
        match.group(0).replace(",", "")
        for match in _NUMERIC_TOKEN_RE.finditer(value or "")
    }


def _support_count(
    candidate: str,
    claims: list[str],
    support_checker: Optional[Callable[[str, str], bool]],
) -> int:
    supported = 0
    for claim in claims:
        if support_checker is not None:
            try:
                if support_checker(claim, candidate):
                    supported += 1
            except (TypeError, ValueError):
                continue
        else:
            matched, total = _coverage(candidate, claim)
            required = 1 if total <= 1 else min(2, total)
            if matched >= required:
                supported += 1
    return supported


def _bounded_span(
    text: str,
    span: _Span,
    claims: list[str],
    support_checker: Optional[Callable[[str, str], bool]],
    *,
    allow_boundary_trim: bool = True,
) -> Optional[tuple[str, bool, int]]:
    raw = text[span.start:span.end]
    words = _word_spans(raw)
    if not words:
        return None
    claim_numbers = {
        number
        for claim in claims
        for number in _numeric_tokens(claim)
    }
    refine_numeric_boundaries = bool(
        allow_boundary_trim
        and (_numeric_tokens(raw) - claim_numbers)
    )
    if (
        not refine_numeric_boundaries
        and len(words) <= (
            HARD_QUOTE_MAX_WORDS if not allow_boundary_trim
            else PREFERRED_QUOTE_MAX_WORDS
        )
        and len(raw) <= HARD_QUOTE_MAX_CHARS
    ):
        normalized = normalize_evidence_whitespace(raw)
        if quote_has_balanced_delimiters(normalized):
            return normalized, False, _support_count(
                normalized,
                claims,
                support_checker,
            )

    claim_terms: set[str] = set()
    for claim in claims:
        for group in _term_groups(claim):
            claim_terms.update(group)
    anchor_indices: list[int] = []
    for index, match in enumerate(words):
        word_terms: set[str] = set()
        for group in _term_groups(match.group(0)):
            word_terms.update(group)
        if word_terms & claim_terms:
            anchor_indices.append(index)
    if not anchor_indices:
        anchor_indices = [len(words) // 2]

    best: Optional[
        tuple[tuple[int, int, int, int, int, int], str, bool, int]
    ] = None
    max_window = min(HARD_QUOTE_MAX_WORDS, len(words))
    ordinary_window_sizes = dict.fromkeys((
        *range(
            min(PREFERRED_QUOTE_MIN_WORDS, max_window),
            min(PREFERRED_QUOTE_MAX_WORDS, max_window) + 1,
            5,
        ),
        min(PREFERRED_QUOTE_MAX_WORDS, max_window),
        max_window,
        *range(min(6, max_window), min(14, max_window) + 1, 2),
    ))
    window_sizes = (
        range(1, max_window + 1)
        if refine_numeric_boundaries
        else ordinary_window_sizes
    )
    for window_size in window_sizes:
        if window_size < 1:
            continue
        starts = (
            set(range(0, len(words) - window_size + 1))
            if refine_numeric_boundaries
            else {
                max(0, min(len(words) - window_size, anchor - (window_size // 2)))
                for anchor in anchor_indices
            }
        )
        if len(words) <= window_size:
            starts.add(0)
        for word_start in sorted(starts):
            word_end = word_start + window_size
            start = words[word_start].start()
            end = words[word_end - 1].end()
            window = raw[start:end]
            variants: list[tuple[str, bool, bool]] = [
                (window, False, False)
            ]
            if allow_boundary_trim:
                for delimiter in re.finditer(r"[\[\](){}]", window):
                    variants.append(
                        (window[delimiter.end():], True, False)
                    )
                    variants.append(
                        (window[:delimiter.start()], False, True)
                    )
            for variant, boundary_start, boundary_end in variants:
                core = normalize_evidence_whitespace(variant)
                if (
                    not core
                    or len(core) > HARD_QUOTE_MAX_CHARS
                    or not quote_has_balanced_delimiters(core)
                ):
                    continue
                support_count = _support_count(
                    core,
                    claims,
                    support_checker,
                )
                if support_count < 1:
                    continue
                truncated_start = word_start > 0 or boundary_start
                truncated_end = word_end < len(words) or boundary_end
                display = core
                if truncated_start:
                    display = _BOUNDARY_ELLIPSIS + display
                if truncated_end:
                    display += _BOUNDARY_ELLIPSIS
                rank = (
                    -support_count,
                    len(claim_numbers - _numeric_tokens(core)),
                    len(_numeric_tokens(core) - claim_numbers),
                    int(boundary_start or boundary_end),
                    (
                        window_size
                        if refine_numeric_boundaries
                        else abs(PREFERRED_QUOTE_MAX_WORDS - window_size)
                    ),
                    word_start,
                )
                candidate = (
                    rank,
                    display,
                    truncated_start or truncated_end,
                    support_count,
                )
                if best is None or candidate[0] < best[0]:
                    best = candidate
        if (
            not refine_numeric_boundaries
            and best is not None
            and best[3] == len(claims)
        ):
            break
    if best is None:
        return None
    return best[1], best[2], best[3]


def _bounded_composite_span(
    source: str,
    claims: list[str],
    support_checker: Optional[Callable[[str, str], bool]],
) -> Optional[tuple[str, int]]:
    """Join two short, balanced, ordered spans only when together supported."""
    breadcrumb = _BREADCRUMB_RE.match(source)
    content_start = breadcrumb.end() if breadcrumb else 0
    words = [
        match
        for match in _word_spans(source)
        if match.start() >= content_start
    ]
    if len(words) < 4:
        return None
    claim_terms: set[str] = set()
    for claim in claims:
        for group in _term_groups(claim):
            claim_terms.update(group)
    anchor_indices: list[int] = []
    for index, match in enumerate(words):
        word_terms = {
            term
            for group in _term_groups(match.group(0))
            for term in group
        }
        if word_terms & claim_terms:
            anchor_indices.append(index)
    anchor_indices = list(dict.fromkeys(anchor_indices))
    if len(anchor_indices) > 32:
        last = len(anchor_indices) - 1
        anchor_indices = [
            anchor_indices[round(index * last / 31)]
            for index in range(32)
        ]
    if len(anchor_indices) < 2:
        return None

    windows: dict[tuple[int, int], tuple[int, int, str, int]] = {}
    maximum = min(36, HARD_QUOTE_MAX_WORDS // 2, len(words))
    minimum = min(8, maximum)
    for anchor in anchor_indices:
        for size in range(minimum, maximum + 1, 4):
            earliest = max(0, anchor - size + 1)
            latest = min(anchor, len(words) - size)
            starts = {earliest, latest, max(earliest, min(latest, anchor - size // 2))}
            for word_start in starts:
                word_end = word_start + size
                start = words[word_start].start()
                end = words[word_end - 1].end()
                core = normalize_evidence_whitespace(source[start:end])
                if (
                    not core
                    or len(core) > HARD_QUOTE_MAX_CHARS // 2
                    or _looks_like_boilerplate(core)
                    or not quote_has_balanced_delimiters(core)
                ):
                    continue
                windows[(start, end)] = (start, end, core, size)

    ordered = sorted(windows.values(), key=lambda item: (item[0], item[1]))
    best: Optional[tuple[tuple[int, int, int], str, int]] = None
    for index, first in enumerate(ordered):
        for second in ordered[index + 1:]:
            if first[1] >= second[0]:
                continue
            word_count = first[3] + second[3] + 1
            if word_count > HARD_QUOTE_MAX_WORDS:
                continue
            display = f"{first[2]} {_BOUNDARY_ELLIPSIS} {second[2]}"
            if len(display) > HARD_QUOTE_MAX_CHARS:
                continue
            support_count = _support_count(
                display,
                claims,
                support_checker,
            )
            if support_count < 1:
                continue
            rank = (
                -support_count,
                word_count,
                second[0] - first[1],
            )
            candidate = (rank, display, support_count)
            if best is None or candidate[0] < best[0]:
                best = candidate
        if best is not None and best[2] == len(claims):
            break
    if best is None:
        return None
    return best[1], best[2]


def select_evidence_quote(
    evidence_text: str,
    claims: Iterable[str],
    *,
    support_checker: Optional[Callable[[str, str], bool]] = None,
    prefer_context: bool = False,
    minimum_context_chars: int = 0,
    allow_composite: bool = False,
) -> Optional[EvidenceQuote]:
    """Select and prove one bounded, contiguous, claim-relevant quote."""
    source = str(evidence_text or "")
    if not source.strip() or _PATH_DISCLOSURE_RE.search(source):
        return None
    claim_values = _claim_variants(claims)
    if not claim_values:
        return None

    longest_claim = max(
        (len(normalize_evidence_whitespace(value)) for value in claim_values),
        default=0,
    )
    context_target = max(longest_claim, max(0, int(minimum_context_chars)))
    ranked: list[tuple[tuple[int, int, int, int, int, int], str, bool]] = []
    for span in _sentence_spans(source):
        raw = source[span.start:span.end]
        if _looks_like_boilerplate(raw):
            continue
        bounded = _bounded_span(
            source,
            span,
            claim_values,
            support_checker,
            allow_boundary_trim=not prefer_context,
        )
        if bounded is None:
            continue
        display, truncated, support_count = bounded
        if support_count < 1 or not quote_is_contiguous(display, source):
            continue
        if not quote_has_balanced_delimiters(display):
            continue
        words = len(_word_spans(quote_core(display)))
        if words > HARD_QUOTE_MAX_WORDS or len(quote_core(display)) > HARD_QUOTE_MAX_CHARS:
            continue
        coverage = max((_coverage(display, claim)[0] for claim in claim_values), default=0)
        preferred_penalty = (
            0
            if PREFERRED_QUOTE_MIN_WORDS <= words <= PREFERRED_QUOTE_MAX_WORDS
            else min(
                abs(words - PREFERRED_QUOTE_MIN_WORDS),
                abs(words - PREFERRED_QUOTE_MAX_WORDS),
            )
        )
        context_penalty = (
            0
            if not prefer_context or len(quote_core(display)) > context_target
            else 1
        )
        rank = (
            -support_count,
            context_penalty,
            span.sentence_count,
            preferred_penalty,
            -coverage,
            words,
        )
        ranked.append((rank, display, truncated))

    if not ranked:
        # A chunk without usable sentence boundaries is still safe when one
        # bounded token window independently passes the support predicate.
        fallback_span = _Span(0, len(source), 3, 0)
        bounded = _bounded_span(
            source,
            fallback_span,
            claim_values,
            support_checker,
            allow_boundary_trim=not prefer_context,
        )
        if bounded is None:
            composite = (
                _bounded_composite_span(
                    source,
                    claim_values,
                    support_checker,
                )
                if allow_composite
                else None
            )
            if composite is None:
                return None
            display, support_count = composite
            truncated = True
        else:
            display, truncated, support_count = bounded
        if (
            support_count < 1
            or _looks_like_boilerplate(display)
            or not quote_is_contiguous(display, source)
            or not quote_has_balanced_delimiters(display)
        ):
            return None
        fallback_context_penalty = (
            0
            if not prefer_context or len(quote_core(display)) > context_target
            else 1
        )
        ranked.append(
            (
                (
                    0,
                    fallback_context_penalty,
                    3,
                    1,
                    0,
                    len(_word_spans(display)),
                ),
                display,
                truncated,
            )
        )

    _rank, selected, truncated = min(ranked, key=lambda item: item[0])
    core = quote_core(selected)
    word_count = len(_word_spans(core))
    if (
        not selected
        or _looks_like_boilerplate(selected)
        or word_count > HARD_QUOTE_MAX_WORDS
        or len(core) > HARD_QUOTE_MAX_CHARS
        or not quote_is_contiguous(selected, source)
        or not quote_has_balanced_delimiters(selected)
    ):
        return None
    return EvidenceQuote(
        text=selected,
        language=evidence_quote_language(core),
        truncated=truncated,
        word_count=word_count,
    )


def select_context_complete_quote(
    evidence_text: str,
    claims: Iterable[str],
    *,
    support_checker: Optional[Callable[[str, str], bool]] = None,
) -> Optional[EvidenceQuote]:
    """Verify a pre-bounded context span without shortening its context."""
    source = str(evidence_text or "")
    core = normalize_evidence_whitespace(source)
    claim_values = _claim_variants(claims)
    word_count = len(_word_spans(core))
    if (
        not core
        or not claim_values
        or word_count > HARD_QUOTE_MAX_WORDS
        or len(core) > HARD_QUOTE_MAX_CHARS
        or not quote_is_contiguous(core, source)
        or not quote_has_balanced_delimiters(core)
    ):
        return None
    checker = support_checker or _default_support_checker
    if not any(checker(claim, core) for claim in claim_values):
        return None
    return EvidenceQuote(
        text=core,
        language=evidence_quote_language(core),
        truncated=False,
        word_count=word_count,
    )
