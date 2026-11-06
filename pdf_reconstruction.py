"""Conservative coordinate-aware reconstruction for born-digital PDF text.

PDF text extraction exposes positioned glyph/word runs, not paragraphs.  This
module keeps a raw representation for audit and builds a separate canonical
representation for retrieval and display.  A boundary is removed only when
layout and token-origin evidence make an intra-word split conclusive.  Visual
line boundaries are converted to natural paragraph flow only when the two
lines are layout-compatible and do not cross a semantic boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
import statistics
import unicodedata
from typing import Iterable, Mapping, Optional, Sequence


PDF_RECONSTRUCTION_REVISION = "pdf-coordinates-v2"

_HEBREW_RE = re.compile(r"[\u0590-\u05ff]")
_HEBREW_TOKEN_RE = re.compile(r"[\u0590-\u05ff]+")
_LATIN_TOKEN_RE = re.compile(r"[A-Za-z]+")
_CONTINUATION_RE = re.compile(
    r"^\s*(?:"
    r"ובשנת|ואילו|לעומת|בעוד|אך|כאשר|מתוכם|מהם|מהן|"
    r"and\s+in|whereas|compared\s+with"
    r")\b",
    re.IGNORECASE,
)
_SENTENCE_FINAL_RE = re.compile(r"[.!?׃…][\"'\u05f4\u2019\u201d)\]}]*\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[•▪◦*-]|\d+[.)])\s+")
_TABLEISH_RE = re.compile(r"(?:\|.{0,200}\||\t)")
_HEADING_PUNCT_RE = re.compile(r"[.!?׃…,:;]\s*$")
_SEMANTIC_PUNCTUATION = frozenset(",.;:!?%׃־-–—()[]{}")


def _finite(value: object, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return result if math.isfinite(result) else default


def _script(value: str) -> str:
    text = str(value or "")
    if _HEBREW_TOKEN_RE.fullmatch(text):
        return "he"
    if _LATIN_TOKEN_RE.fullmatch(text):
        return "en"
    if text and all(character.isdigit() for character in text):
        return "number"
    return "mixed"


def _direction(value: str) -> str:
    return "rtl" if _HEBREW_RE.search(value or "") else "ltr"


@dataclass(frozen=True)
class PositionedToken:
    page: int
    block: int
    line: int
    word: int
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    font: str = ""
    size: float = 0.0
    direction: str = "auto"
    origin_token: Optional[str] = None
    ocr_confidence: Optional[float] = None

    @property
    def width(self) -> float:
        return max(0.0, self.x1 - self.x0)

    @property
    def height(self) -> float:
        return max(0.1, self.y1 - self.y0)


@dataclass(frozen=True)
class ReconstructedLine:
    page: int
    block: int
    line: int
    raw_text: str
    text: str
    x0: float
    y0: float
    x1: float
    y1: float
    direction: str
    font: str
    size: float
    origin_offsets: tuple[Optional[int], ...]
    removed_boundaries: tuple[int, ...]

    @property
    def height(self) -> float:
        return max(0.1, self.y1 - self.y0)


@dataclass(frozen=True)
class ReconstructionEdit:
    raw_start: int
    raw_end: int
    canonical_start: int
    canonical_end: int
    reason_code: str


@dataclass(frozen=True)
class ReconstructedPage:
    page: int
    raw_text: str
    text: str
    origin_offsets: tuple[Optional[int], ...]
    edits: tuple[ReconstructionEdit, ...]
    lines: tuple[ReconstructedLine, ...]
    paragraph_spans: tuple[tuple[int, int], ...]
    revision: str = PDF_RECONSTRUCTION_REVISION
    line_fragment_merge_count: int = 0


def _visual_gap(first: PositionedToken, second: PositionedToken) -> float:
    if first.direction == "rtl":
        return first.x0 - second.x1
    return second.x0 - first.x1


def _same_baseline(first: PositionedToken, second: PositionedToken) -> bool:
    height = max(first.height, second.height)
    return (
        abs(first.y0 - second.y0) <= height * 0.18
        and abs(first.y1 - second.y1) <= height * 0.22
    )


def _compatible_font(first: PositionedToken, second: PositionedToken) -> bool:
    if first.font and second.font and first.font != second.font:
        return False
    if first.size > 0 and second.size > 0:
        return abs(first.size - second.size) <= max(first.size, second.size) * 0.08
    return True


def _wordlike_pair(first: PositionedToken, second: PositionedToken) -> bool:
    first_script = _script(first.text)
    second_script = _script(second.text)
    return first_script == second_script and first_script in {"he", "en"}


def should_remove_token_boundary(
    first: PositionedToken,
    second: PositionedToken,
    *,
    typical_word_gap: float,
) -> bool:
    """Return true only for a conclusive intra-word extraction boundary."""
    if (
        first.page != second.page
        or first.block != second.block
        or first.direction != second.direction
        or not _wordlike_pair(first, second)
        or not _same_baseline(first, second)
        or not _compatible_font(first, second)
    ):
        return False
    if (
        first.ocr_confidence is not None
        and first.ocr_confidence < 0.92
    ) or (
        second.ocr_confidence is not None
        and second.ocr_confidence < 0.92
    ):
        return False
    gap = _visual_gap(first, second)
    height = max(first.height, second.height)
    if gap < -height * 0.12:
        return False
    gap_ratio = max(0.0, gap) / height
    same_origin = bool(
        first.origin_token
        and second.origin_token
        and first.origin_token == second.origin_token
    )
    # A normal PDF word-space is commonly 0.12–0.25 of the line height.  Keep
    # a generous safety margin below that distribution.  Same-origin glyph
    # runs may use the slightly wider bound because the producer explicitly
    # says both fragments came from one source token.
    ratio_limit = 0.085 if same_origin else 0.055
    if gap_ratio > ratio_limit:
        return False
    if (
        not same_origin
        and typical_word_gap > 0
        and max(0.0, gap) > typical_word_gap * 0.42
    ):
        return False
    return True


def _ordered_tokens(tokens: Sequence[PositionedToken], *, preserve_ltr_runs: bool = False) -> list[PositionedToken]:
    direction = _direction("".join(token.text for token in tokens))
    ordered = sorted(
        tokens,
        key=lambda token: (token.x0, token.word),
        reverse=direction == "rtl",
    )
    if preserve_ltr_runs and direction == "rtl":
        # RTL paragraph order must not reverse the words of an embedded LTR
        # name, measurement or reference. No characters inside a token move.
        start = 0
        while start < len(ordered):
            if _HEBREW_RE.search(ordered[start].text):
                start += 1
                continue
            end = start + 1
            while end < len(ordered) and not _HEBREW_RE.search(ordered[end].text):
                end += 1
            if any(re.search(r"[A-Za-z0-9]", t.text) for t in ordered[start:end]):
                ordered[start:end] = sorted(ordered[start:end], key=lambda t: (t.x0, t.word))
            start = end
    return [
        PositionedToken(
            **{
                **token.__dict__,
                "direction": direction,
            }
        )
        for token in ordered
    ]


def reconstruct_line(tokens: Sequence[PositionedToken], *, preserve_ltr_runs: bool = False) -> ReconstructedLine:
    if not tokens:
        raise ValueError("a reconstructed line requires at least one token")
    ordered = _ordered_tokens(tokens, preserve_ltr_runs=preserve_ltr_runs)
    positive_gaps = [
        _visual_gap(first, second)
        for first, second in zip(ordered, ordered[1:])
        if _wordlike_pair(first, second) and _visual_gap(first, second) > 0
    ]
    typical_gap = statistics.median(positive_gaps) if positive_gaps else 0.0

    raw_parts: list[str] = []
    canonical_parts: list[str] = []
    origins: list[Optional[int]] = []
    removed: list[int] = []
    raw_cursor = 0
    for index, token in enumerate(ordered):
        if index:
            raw_parts.append(" ")
            raw_boundary = raw_cursor
            raw_cursor += 1
            previous = ordered[index - 1]
            remove = should_remove_token_boundary(
                previous,
                token,
                typical_word_gap=typical_gap,
            )
            if remove:
                removed.append(raw_boundary)
            else:
                canonical_parts.append(" ")
                origins.append(raw_boundary)
        raw_parts.append(token.text)
        canonical_parts.append(token.text)
        origins.extend(range(raw_cursor, raw_cursor + len(token.text)))
        raw_cursor += len(token.text)

    raw_text = "".join(raw_parts)
    text = "".join(canonical_parts)
    first = ordered[0]
    fonts = [token.font for token in ordered if token.font]
    sizes = [token.size for token in ordered if token.size > 0]
    return ReconstructedLine(
        page=first.page,
        block=first.block,
        line=first.line,
        raw_text=raw_text,
        text=text,
        x0=min(token.x0 for token in ordered),
        y0=min(token.y0 for token in ordered),
        x1=max(token.x1 for token in ordered),
        y1=max(token.y1 for token in ordered),
        direction=first.direction,
        font=statistics.mode(fonts) if fonts else "",
        size=statistics.median(sizes) if sizes else 0.0,
        origin_offsets=tuple(origins),
        removed_boundaries=tuple(removed),
    )


def _group_geometry(
    tokens: Sequence[PositionedToken],
) -> tuple[float, float, float, float, float]:
    return (
        min(token.x0 for token in tokens),
        min(token.y0 for token in tokens),
        max(token.x1 for token in tokens),
        max(token.y1 for token in tokens),
        max(token.height for token in tokens),
    )


def _same_visual_baseline_fragments(
    first: Sequence[PositionedToken],
    second: Sequence[PositionedToken],
) -> bool:
    """Prove two extractor line records are one visual baseline fragment."""
    if not first or not second:
        return False
    left = first[0]
    right = second[0]
    if left.page != right.page or left.block != right.block:
        return False
    if _direction("".join(token.text for token in first)) != _direction(
        "".join(token.text for token in second)
    ):
        return False
    if not all(_compatible_font(a, b) for a in first for b in second):
        return False
    x0_a, y0_a, x1_a, y1_a, height_a = _group_geometry(first)
    x0_b, y0_b, x1_b, y1_b, height_b = _group_geometry(second)
    height = max(height_a, height_b)
    if (
        abs(y0_a - y0_b) > height * 0.18
        or abs(y1_a - y1_b) > height * 0.22
    ):
        return False
    overlap = max(0.0, min(x1_a, x1_b) - max(x0_a, x0_b))
    if overlap > height * 0.12:
        return False
    horizontal_gap = max(0.0, max(x0_a, x0_b) - min(x1_a, x1_b))
    return horizontal_gap <= height * 0.75


def _coalesce_visual_line_fragments(
    groups: Sequence[Sequence[PositionedToken]],
) -> tuple[list[list[PositionedToken]], int]:
    """Merge only geometry-proven fragments before token boundary decisions."""
    output: list[list[PositionedToken]] = []
    merges = 0
    for raw_group in groups:
        group = list(raw_group)
        compatible_index = next(
            (
                index
                for index, existing in enumerate(output)
                if _same_visual_baseline_fragments(existing, group)
            ),
            None,
        )
        if compatible_index is None:
            output.append(group)
            continue
        output[compatible_index].extend(group)
        merges += 1
    return output, merges


def _looks_heading(line: ReconstructedLine, median_size: float) -> bool:
    value = line.text.strip()
    words = value.split()
    size_is_heading = bool(
        median_size > 0 and line.size > median_size * 1.16
    )
    return bool(
        value
        and len(words) <= 12
        and not _HEADING_PUNCT_RE.search(value)
        and size_is_heading
    )


def _same_column(first: ReconstructedLine, second: ReconstructedLine) -> bool:
    overlap = max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    narrow = max(1.0, min(first.x1 - first.x0, second.x1 - second.x0))
    return overlap / narrow >= 0.55


def should_soft_join_lines(
    first: ReconstructedLine,
    second: ReconstructedLine,
    *,
    median_size: float,
) -> bool:
    """Decide whether one visual line boundary is non-semantic."""
    if first.page != second.page:
        return False
    if first.direction != second.direction:
        return False
    if _SENTENCE_FINAL_RE.search(first.text):
        return False
    if _BULLET_RE.match(first.text) or _BULLET_RE.match(second.text):
        return False
    if _TABLEISH_RE.search(first.text) or _TABLEISH_RE.search(second.text):
        return False
    if _looks_heading(first, median_size) or _looks_heading(second, median_size):
        return False
    if not _same_column(first, second):
        return False
    if first.font and second.font and first.font != second.font:
        return False
    if first.size > 0 and second.size > 0:
        if abs(first.size - second.size) > max(first.size, second.size) * 0.10:
            return False
    vertical_gap = second.y0 - first.y1
    height = max(first.height, second.height)
    if vertical_gap < -height * 0.35:
        return False
    # Continuation markers are strong grammatical evidence, while ordinary
    # prose requires a tighter geometric relationship.
    limit = 1.45 if _CONTINUATION_RE.match(second.text) else 0.95
    return vertical_gap <= height * limit


def _same_block_flow_join(
    first: ReconstructedLine,
    second: ReconstructedLine,
    *,
    median_size: float,
) -> bool:
    """Keep ordinary lines from one producer block in natural text flow."""
    if _BULLET_RE.match(first.text) or _BULLET_RE.match(second.text):
        return False
    if _TABLEISH_RE.search(first.text) or _TABLEISH_RE.search(second.text):
        return False
    if _looks_heading(first, median_size) or _looks_heading(second, median_size):
        return False
    return first.direction == second.direction


def _line_sort_key(line: ReconstructedLine) -> tuple[float, float, int, int]:
    # PyMuPDF block/line order is the stable reading-order primary key.  The
    # coordinates make ties deterministic without reordering columns.
    return (float(line.block), float(line.line), int(line.y0 * 100), int(line.x0 * 100))


def reconstruct_page(
    tokens: Iterable[PositionedToken],
    *, preserve_ltr_runs: bool = False,
) -> ReconstructedPage:
    grouped: dict[tuple[int, int], list[PositionedToken]] = {}
    page_number: Optional[int] = None
    for token in tokens:
        page_number = token.page if page_number is None else page_number
        if token.page != page_number:
            raise ValueError("reconstruct_page accepts tokens from one page")
        if not token.text:
            continue
        grouped.setdefault((token.block, token.line), []).append(token)
    if page_number is None:
        return ReconstructedPage(0, "", "", (), (), (), ())

    raw_groups = [
        grouped[key]
        for key in sorted(grouped)
    ]
    visual_groups, line_fragment_merges = _coalesce_visual_line_fragments(
        raw_groups
    )
    lines = sorted(
        (reconstruct_line(values, preserve_ltr_runs=preserve_ltr_runs) for values in visual_groups),
        key=_line_sort_key,
    )
    median_size = statistics.median(
        [line.size for line in lines if line.size > 0]
    ) if any(line.size > 0 for line in lines) else 0.0

    raw_parts: list[str] = []
    canonical_parts: list[str] = []
    origins: list[Optional[int]] = []
    edits: list[ReconstructionEdit] = []
    paragraph_spans: list[tuple[int, int]] = []
    raw_cursor = 0
    canonical_cursor = 0
    paragraph_start = 0

    for index, line in enumerate(lines):
        if index:
            previous = lines[index - 1]
            same_block = previous.block == line.block
            raw_separator = " " if same_block else "\n\n"
            raw_parts.append(raw_separator)
            separator_start = raw_cursor
            raw_cursor += len(raw_separator)
            soft = (
                _same_block_flow_join(
                    previous,
                    line,
                    median_size=median_size,
                )
                if same_block
                else should_soft_join_lines(
                    previous,
                    line,
                    median_size=median_size,
                )
            )
            canonical_separator = " " if soft else "\n\n"
            canonical_parts.append(canonical_separator)
            origins.extend(
                separator_start + min(offset, len(raw_separator) - 1)
                for offset in range(len(canonical_separator))
            )
            if canonical_separator != raw_separator:
                reason_code = (
                    "pdf_soft_visual_line_wrap"
                    if soft
                    else "pdf_structural_boundary_restored"
                )
                edits.append(ReconstructionEdit(
                    raw_start=separator_start,
                    raw_end=raw_cursor,
                    canonical_start=canonical_cursor,
                    canonical_end=canonical_cursor + len(canonical_separator),
                    reason_code=reason_code,
                ))
            if not soft:
                paragraph_spans.append((paragraph_start, canonical_cursor))
                paragraph_start = canonical_cursor + len(canonical_separator)
            canonical_cursor += len(canonical_separator)

        line_raw_start = raw_cursor
        raw_parts.append(line.raw_text)
        canonical_parts.append(line.text)
        origins.extend(
            line_raw_start + offset if offset is not None else None
            for offset in line.origin_offsets
        )
        for boundary in line.removed_boundaries:
            edits.append(ReconstructionEdit(
                raw_start=line_raw_start + boundary,
                raw_end=line_raw_start + boundary + 1,
                canonical_start=canonical_cursor,
                canonical_end=canonical_cursor,
                reason_code="pdf_conclusive_intra_word_boundary",
            ))
        raw_cursor += len(line.raw_text)
        canonical_cursor += len(line.text)

    if canonical_cursor > paragraph_start:
        paragraph_spans.append((paragraph_start, canonical_cursor))
    paragraph_spans = [
        (start, end)
        for start, end in paragraph_spans
        if start < end and "".join(canonical_parts)[start:end].strip()
    ]
    return ReconstructedPage(
        page=page_number,
        raw_text="".join(raw_parts).strip(),
        text="".join(canonical_parts).strip(),
        origin_offsets=tuple(origins),
        edits=tuple(edits),
        lines=tuple(lines),
        paragraph_spans=tuple(paragraph_spans),
        line_fragment_merge_count=line_fragment_merges,
    )


def positioned_tokens_from_pymupdf(
    page_number: int,
    words: Iterable[Sequence[object]],
    styles: Mapping[tuple[int, int], tuple[str, float]] | None = None,
) -> tuple[PositionedToken, ...]:
    """Convert the stable PyMuPDF words tuple into validated positioned tokens."""
    style_map = styles or {}
    output: list[PositionedToken] = []
    for raw in words:
        if len(raw) < 8:
            continue
        x0, y0, x1, y1, text, block, line, word = raw[:8]
        value = unicodedata.normalize("NFC", str(text or "")).strip()
        if not value:
            continue
        try:
            block_number = int(block)
            line_number = int(line)
            word_number = int(word)
        except (TypeError, ValueError, OverflowError):
            continue
        font, size = style_map.get((block_number, line_number), ("", 0.0))
        output.append(PositionedToken(
            page=max(1, int(page_number)),
            block=block_number,
            line=line_number,
            word=word_number,
            text=value,
            x0=_finite(x0),
            y0=_finite(y0),
            x1=_finite(x1),
            y1=_finite(y1),
            font=str(font or ""),
            size=max(0.0, _finite(size)),
            direction=_direction(value),
        ))
    return tuple(output)


def pymupdf_line_styles(raw_dict: object) -> dict[tuple[int, int], tuple[str, float]]:
    """Read only non-sensitive font/size structure from a PyMuPDF dict."""
    if not isinstance(raw_dict, dict):
        return {}
    styles: dict[tuple[int, int], tuple[str, float]] = {}
    for block_index, block in enumerate(raw_dict.get("blocks") or []):
        if not isinstance(block, dict):
            continue
        block_number = int(block.get("number", block_index))
        for line_index, line in enumerate(block.get("lines") or []):
            if not isinstance(line, dict):
                continue
            spans = [
                span for span in (line.get("spans") or [])
                if isinstance(span, dict)
            ]
            if not spans:
                continue
            span = max(
                spans,
                key=lambda item: len(str(item.get("text") or "")),
            )
            styles[(block_number, line_index)] = (
                str(span.get("font") or ""),
                max(0.0, _finite(span.get("size"))),
            )
    return styles
