"""Canonical logical text for extraction, retrieval, grounding, and display.

The source bytes and the extractor's raw output remain audit material.  This
module produces a separate logical-order representation and an origin map.  It
never reverses a Hebrew string: the only character reordering is a bounded,
structurally invalid closing/opening delimiter pair around mixed Hebrew/LTR
content emitted in visual order by PDF extractors.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import unicodedata
from typing import Iterable, Optional, Sequence


CANONICAL_TEXT_REVISION = "logical-text-v3"
RAW_MAPPING_REVISION = "origin-offsets-v1"
MAX_VISUAL_ORDER_PAIR_CHARS = 180

_HEBREW_RE = re.compile(r"[\u0590-\u05ff]")
_LTR_ANCHOR_RE = re.compile(r"[A-Za-z0-9]")
_BIDI_CONTROL_RE = re.compile(r"[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]")
_HORIZONTAL_SPACE_RE = re.compile(r"(?: {2,}|\t+)")
_SINGLE_LINE_WRAP_RE = re.compile(r"(?<!\n)\n(?!\n)")
_PERCENT_SPACE_RE = re.compile(r"(?<=\d)[ \t]+(?=%)")
_NUMBER_SEPARATOR_RE = re.compile(r"(?<=\d)[ \t]+([,./])[ \t]+(?=\d)")
_HEBREW_NUMERIC_DASH_RE = re.compile(
    r"(?<![\u0590-\u05ff])([\u05d0-\u05ea])[ \t]+([-‐‑‒–—―\u05be])[ \t]+(?=\d)"
)
_SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \t]+([,.;:!?])")
_SPACE_AFTER_OPEN_RE = re.compile(r"([\(\[\{])[ \t]+")
_SPACE_BEFORE_CLOSE_RE = re.compile(r"[ \t]+([\)\]\}])")

_DELIMITER_PAIRS = (("[", "]"), ("{", "}"), ("(", ")"))
_TRAILING_PUNCT_RE = re.compile(r"(?s)^(.*?)([ \t]*)([,.;:!?])([ \t]*)$")


@dataclass(frozen=True)
class CanonicalEdit:
    raw_start: int
    raw_end: int
    canonical_start: int
    canonical_end: int
    reason_code: str


@dataclass(frozen=True)
class CanonicalText:
    raw_text: str
    text: str
    origin_offsets: tuple[Optional[int], ...]
    edits: tuple[CanonicalEdit, ...]
    revision: str = CANONICAL_TEXT_REVISION
    mapping_revision: str = RAW_MAPPING_REVISION

    @property
    def applied(self) -> bool:
        return self.raw_text != self.text

    @property
    def raw_sha256(self) -> str:
        return hashlib.sha256(self.raw_text.encode("utf-8")).hexdigest()

    @property
    def canonical_sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def raw_spans_for(self, canonical_start: int, canonical_end: int) -> tuple[tuple[int, int], ...]:
        """Map a canonical half-open span to ordered raw extraction spans."""
        start = max(0, min(int(canonical_start), len(self.origin_offsets)))
        end = max(start, min(int(canonical_end), len(self.origin_offsets)))
        offsets = [
            value for value in self.origin_offsets[start:end]
            if isinstance(value, int) and 0 <= value < len(self.raw_text)
        ]
        if not offsets:
            return ()
        spans: list[tuple[int, int]] = []
        span_start = offsets[0]
        previous = offsets[0]
        for value in offsets[1:]:
            if value == previous + 1:
                previous = value
                continue
            spans.append((span_start, previous + 1))
            span_start = previous = value
        spans.append((span_start, previous + 1))
        return tuple(spans)

    def metadata(self) -> dict[str, object]:
        return {
            "canonicalization_revision": self.revision,
            "canonical_mapping_revision": self.mapping_revision,
            "canonicalization_applied": self.applied,
            "canonical_edit_count": len(self.edits),
            "raw_extracted_sha256": self.raw_sha256,
            "canonical_extracted_sha256": self.canonical_sha256,
        }


@dataclass(frozen=True)
class _PendingEdit:
    raw_start: int
    raw_end: int
    reason_code: str


_Item = tuple[str, Optional[int]]


def _items_text(items: Sequence[_Item]) -> str:
    return "".join(character for character, _origin in items)


def _raw_bounds(items: Sequence[_Item]) -> tuple[int, int]:
    offsets = [
        origin for _character, origin in items
        if isinstance(origin, int)
    ]
    if not offsets:
        return 0, 0
    return min(offsets), max(offsets) + 1


def _recorded_replace(
    items: list[_Item],
    pattern: re.Pattern[str],
    replacement,
    reason_code: str,
    pending: list[_PendingEdit],
) -> list[_Item]:
    text = _items_text(items)
    output: list[_Item] = []
    cursor = 0
    for match in pattern.finditer(text):
        if match.start() < cursor:
            continue
        original = items[match.start():match.end()]
        next_items = replacement(match, original)
        if next_items is None:
            continue
        if next_items == original:
            continue
        output.extend(items[cursor:match.start()])
        output.extend(next_items)
        raw_start, raw_end = _raw_bounds(original)
        pending.append(_PendingEdit(raw_start, raw_end, reason_code))
        cursor = match.end()
    if cursor == 0:
        return items
    output.extend(items[cursor:])
    return output


def _normalize_compatibility(value: str) -> list[_Item]:
    items: list[_Item] = []
    for raw_index, raw_character in enumerate(str(value or "")):
        compatibility = unicodedata.normalize("NFKC", raw_character)
        for character in compatibility:
            if character in {"\ufeff", "\u00ad", "\u200b", "\x00"}:
                continue
            if _BIDI_CONTROL_RE.fullmatch(character):
                continue
            if character in {"\u2028"}:
                character = "\n"
            elif character in {"\u2029"}:
                for replacement in "\n\n":
                    items.append((replacement, raw_index))
                continue
            elif character in {"\u00a0", "\u202f"}:
                character = " "
            if character == "\r":
                character = "\n"
            if character in "\n\t" or unicodedata.category(character) not in {"Cc", "Cs"}:
                items.append((character, raw_index))
    return items


def _repair_reversed_delimiters(
    items: list[_Item],
    pending: list[_PendingEdit],
) -> list[_Item]:
    for opening, closing in _DELIMITER_PAIRS:
        text = _items_text(items)
        candidate_spans: list[tuple[int, int, int]] = []
        for line_match in re.finditer(r"[^\n]*", text):
            line = line_match.group(0)
            if not line:
                continue
            stack: list[int] = []
            unmatched_closings: set[int] = set()
            for local_index, character in enumerate(line):
                if character == opening:
                    stack.append(local_index)
                elif character == closing:
                    if stack:
                        stack.pop()
                    else:
                        unmatched_closings.add(local_index)
            unmatched_openings = set(stack)
            local_pattern = re.compile(
                re.escape(closing)
                + rf"(?P<body>[^{re.escape(opening + closing)}\\n]"
                + "{1,"
                + str(MAX_VISUAL_ORDER_PAIR_CHARS)
                + "}?)"
                + re.escape(opening)
            )
            for candidate in local_pattern.finditer(line):
                closing_index = candidate.start()
                opening_index = candidate.end() - 1
                body = candidate.group("body")
                stripped = body.strip()
                if not stripped or stripped[0] in ",.;:!?":
                    continue
                structurally_unmatched = (
                    closing_index in unmatched_closings
                    and opening_index in unmatched_openings
                )
                previous_open = line.rfind(opening, 0, closing_index)
                previous_close = line.rfind(closing, 0, closing_index)
                locally_unmatched = previous_open <= previous_close
                # Visual-order extractors can emit two reversed pairs as
                # ``) first ( ... ) second (``. A conventional stack makes
                # the middle opening/closing look matched even though the
                # outer closing and opening remain unmatched. Accept the
                # second pair only when those unmatched outer witnesses exist;
                # this does not match valid sequential ``(first) ... (second)``
                # groups.
                crossing_visual_chain = (
                    opening_index in unmatched_openings
                    and any(
                        unmatched_closing < closing_index
                        for unmatched_closing in unmatched_closings
                    )
                )
                numeric_start = bool(re.match(
                    r"(?:\d|[A-Za-z]|[\u05d0-\u05ea][ \t]*[-\u05be])",
                    stripped,
                ))
                nested_content = any(
                    left in body and right in body
                    for left, right in _DELIMITER_PAIRS
                    if (left, right) != (opening, closing)
                )
                if not (
                    structurally_unmatched
                    or locally_unmatched
                    or crossing_visual_chain
                    or numeric_start
                    or nested_content
                ):
                    continue
                local_end = opening_index + 1
                numeric_tail = re.match(
                    r"[ \t]*(?:כ(?:[ \t]*[-־])?[ \t]*)?"
                    r"\d+(?:[.,]\d+)?[ \t]*%",
                    line[local_end:],
                )
                if numeric_tail:
                    local_end += numeric_tail.end()
                candidate_spans.append((
                    line_match.start() + closing_index,
                    line_match.start() + local_end,
                    line_match.start() + opening_index,
                ))

        for start, end, opening_position in reversed(candidate_spans):
            original = items[start:end]
            opening_offset = opening_position - start
            opening_item = original[opening_offset]
            prefix_items = list(original[1:opening_offset])
            tail_items = list(original[opening_offset + 1:])
            punctuation_items: list[_Item] = []
            if tail_items:
                prefix_text = _items_text(prefix_items)
                prefix_trailing = _TRAILING_PUNCT_RE.match(prefix_text)
                if prefix_trailing:
                    punctuation_items = prefix_items[
                        prefix_trailing.start(3):prefix_trailing.end(3)
                    ]
                    prefix_items = prefix_items[:prefix_trailing.end(1)]
                if (
                    prefix_items
                    and tail_items
                    and _items_text(prefix_items).rstrip()[-1:].isalnum()
                    and _items_text(tail_items).lstrip()[:1].isalnum()
                ):
                    prefix_items.append((" ", opening_item[1]))
            body_items = [*prefix_items, *tail_items]
            body = _items_text(body_items)
            if not _LTR_ANCHOR_RE.search(body):
                continue
            current_text = _items_text(items)
            line_start = current_text.rfind("\n", 0, start) + 1
            line_end = current_text.find("\n", end)
            if line_end < 0:
                line_end = len(items)
            line = _items_text(items[line_start:line_end])
            if len(_HEBREW_RE.findall(line)) < 3:
                continue

            body_text = _items_text(body_items)
            trailing = _TRAILING_PUNCT_RE.match(body_text)
            if trailing and not punctuation_items:
                core_end = trailing.end(1)
                punctuation_start = trailing.start(3)
                punctuation_end = trailing.end(3)
                punctuation_items = body_items[punctuation_start:punctuation_end]
                body_items = body_items[:core_end]
            replacement = [
                (opening, original[0][1]),
                *body_items,
                (closing, opening_item[1]),
                *punctuation_items,
            ]
            raw_start, raw_end = _raw_bounds(original)
            pending.append(_PendingEdit(
                raw_start,
                raw_end,
                "pdf_visual_order_delimiter_pair",
            ))
            items[start:end] = replacement
    return items


def _regex_sub_items(
    items: list[_Item],
    pattern: re.Pattern[str],
    replacement_text,
    reason_code: str,
    pending: list[_PendingEdit],
) -> list[_Item]:
    def replacement(match: re.Match[str], original: list[_Item]) -> list[_Item]:
        rendered = (
            replacement_text(match)
            if callable(replacement_text)
            else match.expand(replacement_text)
        )
        origins = [origin for _character, origin in original if origin is not None]
        fallback = origins[0] if origins else None
        output: list[_Item] = []
        search_cursor = 0
        original_text = _items_text(original)
        for character in rendered:
            found = original_text.find(character, search_cursor)
            if found < 0:
                found = original_text.find(character)
            if found >= 0:
                output.append((character, original[found][1]))
                search_cursor = found + 1
            else:
                output.append((character, fallback))
        return output

    local_pending: list[_PendingEdit] = []
    output = _recorded_replace(
        items,
        pattern,
        replacement,
        reason_code,
        local_pending,
    )
    if local_pending:
        pending.append(_PendingEdit(
            min(edit.raw_start for edit in local_pending),
            max(edit.raw_end for edit in local_pending),
            reason_code,
        ))
    return output


def _repair_proven_pdf_block_wraps(
    items: list[_Item],
    pending: list[_PendingEdit],
) -> list[_Item]:
    """Join only double-newline boundaries proven to continue one sentence."""
    try:
        from .source_structure import boundary_join_reason
    except ImportError:
        from source_structure import boundary_join_reason

    text = _items_text(items)
    boundaries = list(re.finditer(r"\n[ \t]*\n+", text))
    if not boundaries:
        return items
    output = list(items)
    for match in reversed(boundaries):
        previous_start = text.rfind("\n\n", 0, match.start())
        previous = text[previous_start + 2 if previous_start >= 0 else 0:match.start()]
        following_end_match = re.search(r"\n[ \t]*\n+", text[match.end():])
        following_end = (
            match.end() + following_end_match.start()
            if following_end_match
            else len(text)
        )
        following = text[match.end():following_end]
        reason = boundary_join_reason(previous, following)
        if not reason:
            continue
        original = output[match.start():match.end()]
        origins = [
            origin for _character, origin in original
            if isinstance(origin, int)
        ]
        replacement_origin = origins[0] if origins else None
        raw_start, raw_end = _raw_bounds(original)
        output[match.start():match.end()] = [(" ", replacement_origin)]
        pending.append(_PendingEdit(
            raw_start,
            raw_end,
            f"pdf_proven_soft_block_wrap:{reason}",
        ))
    return output


def _finalize(
    raw_text: str,
    items: list[_Item],
    pending: Iterable[_PendingEdit],
) -> CanonicalText:
    text = _items_text(items).strip()
    leading = len(_items_text(items)) - len(_items_text(items).lstrip())
    trailing_end = leading + len(text)
    items = items[leading:trailing_end]
    origins = tuple(origin for _character, origin in items)
    edits: list[CanonicalEdit] = []
    for edit in pending:
        positions = [
            index for index, origin in enumerate(origins)
            if isinstance(origin, int) and edit.raw_start <= origin < edit.raw_end
        ]
        canonical_start = min(positions) if positions else 0
        canonical_end = max(positions) + 1 if positions else canonical_start
        candidate = CanonicalEdit(
            raw_start=edit.raw_start,
            raw_end=edit.raw_end,
            canonical_start=canonical_start,
            canonical_end=canonical_end,
            reason_code=edit.reason_code,
        )
        if candidate not in edits:
            edits.append(candidate)
    return CanonicalText(
        raw_text=raw_text,
        text=text,
        origin_offsets=origins,
        edits=tuple(edits),
    )


def canonicalize_source_text(
    value: str,
    *,
    repair_visual_bidi: bool = False,
    normalize_pdf_spacing: bool = False,
    join_pdf_line_wraps: bool = False,
) -> CanonicalText:
    """Return canonical logical text and a deterministic raw-origin mapping."""
    raw_text = str(value or "")
    pending: list[_PendingEdit] = []
    items = _normalize_compatibility(raw_text)
    if _items_text(items) != raw_text:
        pending.append(_PendingEdit(0, len(raw_text), "unicode_compatibility"))

    if join_pdf_line_wraps:
        for _pass in range(8):
            repaired = _repair_proven_pdf_block_wraps(items, pending)
            if repaired == items:
                break
            items = repaired
        items = _regex_sub_items(
            items,
            _SINGLE_LINE_WRAP_RE,
            " ",
            "pdf_nonsemantic_line_wrap",
            pending,
        )

    if repair_visual_bidi:
        items = _repair_reversed_delimiters(items, pending)

    if normalize_pdf_spacing:
        transformations = (
            (_HORIZONTAL_SPACE_RE, " ", "compatible_horizontal_whitespace"),
            (_PERCENT_SPACE_RE, "", "pdf_percent_spacing"),
            (_NUMBER_SEPARATOR_RE, r"\1", "pdf_numeric_separator_spacing"),
            (
                _HEBREW_NUMERIC_DASH_RE,
                lambda match: f"{match.group(1)}{match.group(2)}",
                "pdf_hebrew_numeric_dash_spacing",
            ),
            (_SPACE_AFTER_OPEN_RE, r"\1", "pdf_open_delimiter_spacing"),
            (_SPACE_BEFORE_CLOSE_RE, r"\1", "pdf_close_delimiter_spacing"),
            (_SPACE_BEFORE_PUNCT_RE, r"\1", "pdf_punctuation_spacing"),
        )
        for pattern, replacement, reason in transformations:
            items = _regex_sub_items(items, pattern, replacement, reason, pending)

    return _finalize(raw_text, items, pending)


def canonicalize_reconstructed_source(
    raw_text: str,
    reconstructed_text: str,
    reconstructed_origin_offsets: Sequence[Optional[int]],
    reconstruction_edits: Iterable[CanonicalEdit],
    *,
    repair_visual_bidi: bool = False,
    normalize_pdf_spacing: bool = False,
    join_pdf_line_wraps: bool = False,
) -> CanonicalText:
    """Canonicalize reconstructed text while retaining raw extraction origins.

    ``reconstructed_origin_offsets`` maps every reconstructed character to the
    raw extractor representation.  This composes that map with the normal
    logical-text transformations so restricted audit can always trace a
    displayed character back through coordinate reconstruction.
    """
    raw_value = str(raw_text or "")
    reconstructed_value = str(reconstructed_text or "")
    base_origins = tuple(reconstructed_origin_offsets)
    if len(base_origins) != len(reconstructed_value):
        raise ValueError("reconstructed origin map must align with reconstructed text")
    logical = canonicalize_source_text(
        reconstructed_value,
        repair_visual_bidi=repair_visual_bidi,
        normalize_pdf_spacing=normalize_pdf_spacing,
        join_pdf_line_wraps=join_pdf_line_wraps,
    )
    composed: list[Optional[int]] = []
    for origin in logical.origin_offsets:
        if isinstance(origin, int) and 0 <= origin < len(base_origins):
            raw_origin = base_origins[origin]
            composed.append(
                raw_origin
                if isinstance(raw_origin, int) and 0 <= raw_origin < len(raw_value)
                else None
            )
        else:
            composed.append(None)

    edits: list[CanonicalEdit] = []
    base_edit_values = list(reconstruction_edits)
    for edit in base_edit_values:
        positions = [
            index
            for index, origin in enumerate(composed)
            if (
                isinstance(origin, int)
                and edit.raw_start <= origin < edit.raw_end
            )
        ]
        candidate = CanonicalEdit(
            raw_start=max(0, min(int(edit.raw_start), len(raw_value))),
            raw_end=max(0, min(int(edit.raw_end), len(raw_value))),
            canonical_start=min(positions) if positions else 0,
            canonical_end=max(positions) + 1 if positions else 0,
            reason_code=str(edit.reason_code),
        )
        if candidate not in edits:
            edits.append(candidate)
    for edit in logical.edits:
        raw_origins = [
            base_origins[index]
            for index in range(
                max(0, edit.raw_start),
                min(len(base_origins), edit.raw_end),
            )
            if isinstance(base_origins[index], int)
        ]
        raw_start = min(raw_origins) if raw_origins else 0
        raw_end = max(raw_origins) + 1 if raw_origins else raw_start
        candidate = CanonicalEdit(
            raw_start=raw_start,
            raw_end=raw_end,
            canonical_start=edit.canonical_start,
            canonical_end=edit.canonical_end,
            reason_code=edit.reason_code,
        )
        if candidate not in edits:
            edits.append(candidate)
    return CanonicalText(
        raw_text=raw_value,
        text=logical.text,
        origin_offsets=tuple(composed),
        edits=tuple(edits),
    )


def has_visual_order_artifact(value: str) -> bool:
    """Return true only for the bounded reversed mixed-direction structure."""
    probe = canonicalize_source_text(
        value,
        repair_visual_bidi=True,
        normalize_pdf_spacing=False,
        join_pdf_line_wraps=False,
    )
    return any(
        edit.reason_code == "pdf_visual_order_delimiter_pair"
        for edit in probe.edits
    )


def _origin_runs(offsets: Sequence[Optional[int]]) -> list[list[Optional[int]]]:
    runs: list[list[Optional[int]]] = []
    if not offsets:
        return runs
    start = 0
    first = offsets[0]
    previous = first
    for index, value in enumerate(offsets[1:], start=1):
        contiguous = (
            isinstance(previous, int)
            and isinstance(value, int)
            and value == previous + 1
        )
        repeated_none = previous is None and value is None
        if contiguous or repeated_none:
            previous = value
            continue
        runs.append([start, index, first, previous])
        start = index
        first = previous = value
    runs.append([start, len(offsets), first, previous])
    return runs


def extraction_audit_id(result: CanonicalText, source_sha256: str) -> str:
    """Identify the complete extracted generation, including provenance mapping."""
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("invalid audit source checksum")
    generation = {
        "raw_sha256": result.raw_sha256,
        "canonical_sha256": result.canonical_sha256,
        "mapping_revision": result.mapping_revision,
        "origin_runs": _origin_runs(result.origin_offsets),
        "edits": [edit.__dict__ for edit in result.edits],
    }
    digest = hashlib.sha256(json.dumps(
        generation, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return f"{source_sha256}-{result.revision}-{digest}"


def persist_extraction_audit(
    result: CanonicalText,
    *,
    audit_root: str | Path,
    tenant_id: str,
    source_id: str,
    source_sha256: str,
) -> tuple[str, str]:
    """Persist one immutable, service-private raw/canonical audit generation."""
    if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9_-]{0,62}[A-Za-z0-9])?", tenant_id):
        raise ValueError("invalid audit tenant")
    if not re.fullmatch(r"[A-Za-z0-9_-]{43}", source_id):
        raise ValueError("invalid audit source")
    if not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("invalid audit source checksum")

    root = Path(audit_root)
    directory = root / tenant_id / source_id
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(directory, 0o700)
    audit_id = extraction_audit_id(result, source_sha256)
    destination = directory / f"{audit_id}.json.gz"
    payload = {
        "schema_version": 1,
        "audit_id": audit_id,
        "tenant_id": tenant_id,
        "source_id": source_id,
        "source_sha256": source_sha256,
        "canonicalization_revision": result.revision,
        "mapping_revision": result.mapping_revision,
        "raw_sha256": result.raw_sha256,
        "canonical_sha256": result.canonical_sha256,
        "raw_text": result.raw_text,
        "canonical_text": result.text,
        "origin_runs": _origin_runs(result.origin_offsets),
        "edits": [edit.__dict__ for edit in result.edits],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload_sha256 = hashlib.sha256(encoded).hexdigest()
    def verify_existing():
        try:
            with gzip.open(destination, "rb") as handle:
                previous = handle.read()
        except (OSError, EOFError) as error:
            raise ValueError("existing extraction audit is unreadable") from error
        if previous != encoded:
            raise ValueError("existing extraction audit generation differs")

    if destination.exists():
        verify_existing()
        return audit_id, payload_sha256

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".canonical-audit-",
        suffix=".tmp",
        dir=directory,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as raw_handle:
            with gzip.GzipFile(
                fileobj=raw_handle,
                mode="wb",
                filename="",
                mtime=0,
            ) as compressed:
                compressed.write(encoded)
            raw_handle.flush()
            os.fsync(raw_handle.fileno())
        try:
            os.link(temporary_name, destination)
        except FileExistsError:
            verify_existing()
        finally:
            os.unlink(temporary_name)
    except Exception:
        try:
            os.close(descriptor)
        except OSError:
            pass
        try:
            os.unlink(temporary_name)
        except OSError:
            pass
        raise
    return audit_id, payload_sha256
