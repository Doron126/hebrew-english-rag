# rag/splitter.py
import os
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

HEADER_RE = re.compile(
    r"^[ \t]{0,3}(#{1,6})[ \t]+(.+?)(?:[ \t]+#+[ \t]*)?$",
    re.MULTILINE,
)
_FENCE_RE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    """Read a bounded integer without making document ingestion fragile."""
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))

_TARGET_CHARS = _env_int("CHUNK_SIZE", 1200, 256, 8000)
_CHUNK_OVERLAP_CHARS = _env_int(
    "CHUNK_OVERLAP",
    120,
    0,
    max(0, _TARGET_CHARS // 3),
)
_HARD_MAX_CHARS = _env_int(
    "CHUNK_HARD_MAX_CHARS",
    max(1800, _TARGET_CHARS + _CHUNK_OVERLAP_CHARS),
    _TARGET_CHARS + _CHUNK_OVERLAP_CHARS,
    12000,
)
_MAX_BREADCRUMB_CHARS = 512
_BREADCRUMB_BUDGET = max(
    32,
    min(
        _MAX_BREADCRUMB_CHARS,
        _TARGET_CHARS // 2,
        _HARD_MAX_CHARS - 64,
    ),
)

# ---------- cleaning helpers ----------
_FRONT_MATTER_RE = re.compile(
    r"^\s*(?:---|\+\+\+)\s*[\s\S]*?(?:---|\+\+\+)\s*", re.MULTILINE
)
# Hugo shortcodes {{% ... %}} or {{< ... >}} – single or multi-line
_HUGO_SC_RE = re.compile(r"^\s*\{\{[%<].*?[%>]\}\}\s*$", re.MULTILINE)

def strip_front_matter(text: str) -> str:
    return _FRONT_MATTER_RE.sub("", text or "", count=1)

def strip_hugo_shortcodes(text: str) -> str:
    return _HUGO_SC_RE.sub("", text or "")

_HTML_TAG_RE = re.compile(r"</?(?:div|span|section|article)[^>]*>", re.IGNORECASE)


def _strip_html_wrappers_outside_fences(text: str) -> str:
    output: List[str] = []
    fence_char = ""
    fence_length = 0
    for line in (text or "").splitlines(keepends=True):
        match = _FENCE_RE.match(line)
        if match:
            marker = match.group(1)
            if not fence_char:
                fence_char, fence_length = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length:
                fence_char, fence_length = "", 0
            output.append(line)
        elif fence_char:
            output.append(line)
        else:
            output.append(_HTML_TAG_RE.sub("", line))
    return "".join(output)


def _strip_hugo_shortcodes_outside_fences(text: str) -> str:
    output: List[str] = []
    fence_char = ""
    fence_length = 0
    for line in (text or "").splitlines(keepends=True):
        match = _FENCE_RE.match(line)
        if match:
            marker = match.group(1)
            if not fence_char:
                fence_char, fence_length = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length:
                fence_char, fence_length = "", 0
            output.append(line)
        elif fence_char:
            output.append(line)
        else:
            output.append(_HUGO_SC_RE.sub("", line))
    return "".join(output)

def clean_markdown(text: str) -> str:
    t = strip_front_matter(text or "")
    t = _strip_hugo_shortcodes_outside_fences(t)
    t = _strip_html_wrappers_outside_fences(t)
    # collapse excessive blank lines
    t = re.sub(r"\n{3,}", "\n\n", t).strip()
    return t

# ---------- utilities ----------
def _slug(s: str, max_len: int = 80) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:max_len] or "section"

def _iter_headers(text: str) -> List[Tuple[int, str, int]]:
    """Return headings outside fenced code plus a sentinel end marker."""
    out: List[Tuple[int, str, int]] = []
    offset = 0
    fence_char = ""
    fence_length = 0

    for line in (text or "").splitlines(keepends=True):
        if fence_char:
            closing_fence = re.match(
                rf"^[ \t]{{0,3}}{re.escape(fence_char)}{{{fence_length},}}"
                r"[ \t]*(?:\r?\n)?$",
                line,
            )
            if closing_fence:
                fence_char = ""
                fence_length = 0
            offset += len(line)
            continue

        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            info_string = line[fence.end():].rstrip("\r\n")
            # CommonMark forbids a backtick in a backtick fence's info string.
            # Treat such a line as ordinary content; otherwise one malformed
            # example can hide every real heading that follows it.
            if marker[0] == "`" and "`" in info_string:
                offset += len(line)
                continue
            fence_char = marker[0]
            fence_length = len(marker)
            offset += len(line)
            continue

        match = HEADER_RE.match(line)
        if match:
            out.append((len(match.group(1)), match.group(2).strip(), offset))
        offset += len(line)

    out.append((1, "__END__", len(text)))
    return out

def _ellipsize_middle(value: str, max_chars: int) -> str:
    """Keep recognizable beginning/end identity inside a strict character cap."""
    value = value or ""
    if len(value) <= max_chars:
        return value
    if max_chars <= 0:
        return ""
    if max_chars == 1:
        return "\u2026"

    available = max_chars - 1
    left = (available + 1) // 2
    right = available - left
    suffix = value[-right:] if right else ""
    return f"{value[:left]}\u2026{suffix}"

def _bounded_breadcrumb(
    file_title: str,
    section_titles: List[str],
    max_chars: int = _BREADCRUMB_BUDGET,
) -> str:
    """Render a useful breadcrumb without letting navigation dwarf content.

    Normal breadcrumbs are returned byte-for-byte unchanged.  For pathological
    paths, retain a middle-elided file identity and as much of the deepest
    section path as fits.  The unabridged values remain available in metadata.
    """
    sections = [str(title) for title in section_titles if title and title != "__END__"]
    file_identity = str(file_title or "")
    full = " > ".join([file_identity] + sections)
    max_chars = max(1, min(int(max_chars), _HARD_MAX_CHARS - 4))
    if len(full) <= max_chars:
        return full

    file_budget = min(max_chars, max(12, max_chars // 3))
    rendered_file = _ellipsize_middle(file_identity, file_budget)
    if not sections:
        return _ellipsize_middle(rendered_file, max_chars)

    separator = " > "
    tail_budget = max_chars - len(rendered_file) - len(separator)
    if tail_budget <= 0:
        return _ellipsize_middle(rendered_file, max_chars)
    # If ancestors must be omitted, reserve room for the explicit ellipsis.
    marker_cost = len("\u2026") + len(separator)
    current_budget = max(1, tail_budget - (marker_cost if len(sections) > 1 else 0))
    rendered_tail = [_ellipsize_middle(sections[-1], current_budget)]
    remaining = tail_budget - len(rendered_tail[0])

    omitted = False
    for parent in reversed(sections[:-1]):
        cost = len(separator) + len(parent)
        if cost <= remaining:
            rendered_tail.insert(0, parent)
            remaining -= cost
        else:
            omitted = True
            break

    if omitted and marker_cost <= remaining:
        rendered_tail.insert(0, "\u2026")

    result = separator.join([rendered_file] + rendered_tail)
    return result[:max_chars]

def _dedup_near_duplicates(
    chunks: List[str],
    scopes: Optional[List[str]] = None,
) -> List[bool]:
    """Collapse only canonically identical text in the same semantic scope.

    Rendered section chunks begin with a breadcrumb.  It is deliberately part
    of the identity: ``Alpha > Status: enabled`` and ``Beta > Status: enabled``
    are separate evidence, even though their bodies match. Removing one makes
    entity- or section-scoped questions impossible to answer reliably. The
    complete body also participates, so shared boilerplate never hides changed
    numbers, dates, exceptions, negations, or later paragraphs.

    Canonicalization is intentionally conservative.  It normalizes equivalent
    Unicode composition and whitespace introduced by line wrapping, while
    preserving case and punctuation because either can carry meaning.
    """
    if scopes is not None and len(scopes) != len(chunks):
        raise ValueError("scopes must align with chunks")

    seen = set()
    keep = [True] * len(chunks)
    for i, c in enumerate(chunks):
        canonical = unicodedata.normalize("NFC", (c or "").strip())
        canonical = re.sub(r"\s+", " ", canonical)
        scope = scopes[i] if scopes is not None else ""
        identity = (unicodedata.normalize("NFC", scope), canonical)
        if identity in seen:
            keep[i] = False
        else:
            seen.add(identity)
    return keep

# ---------- fallback chunkers ----------
def _paragraphs(text: str) -> List[str]:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    return parts

def _split_oversized_paragraph(paragraph: str, hard_max: int) -> List[str]:
    """Split one paragraph without ever returning a piece above ``hard_max``.

    Sentence endings are preferred, then any whitespace, with a character-level
    split as the final fallback for minified text or long identifiers.
    """
    if hard_max < 1:
        raise ValueError("hard_max must be at least 1")

    remaining = (paragraph or "").strip()
    pieces: List[str] = []
    while len(remaining) > hard_max:
        window = remaining[: hard_max + 1]
        minimum_useful_cut = max(1, hard_max // 2)

        sentence_cuts = [
            m.end()
            for m in re.finditer(r"[.!?\u2026][\"'\u2019\u201d\u00bb)\]]*\s+", window)
            if minimum_useful_cut <= m.end() <= hard_max
        ]
        if sentence_cuts:
            cut = sentence_cuts[-1]
        else:
            whitespace_cuts = [
                window.rfind(char, minimum_useful_cut, hard_max + 1)
                for char in ("\n", "\t", " ")
            ]
            cut = max(whitespace_cuts)
            if cut < minimum_useful_cut:
                cut = hard_max

        piece = remaining[:cut].rstrip()
        if not piece:  # Defensive progress guarantee for whitespace-only edges.
            piece = remaining[:hard_max]
            cut = hard_max
        pieces.append(piece)
        remaining = remaining[cut:].lstrip()

    if remaining:
        pieces.append(remaining)
    return pieces

def _pack_paragraphs(
    paras: List[str],
    target_chars: int = _TARGET_CHARS,
    hard_max: int = _HARD_MAX_CHARS,
) -> List[str]:
    if hard_max < 1:
        raise ValueError("hard_max must be at least 1")
    target_chars = max(1, min(target_chars, hard_max))

    # Split oversized paragraphs near the configured target.  The previous
    # implementation split only at the hard ceiling and then kept packing up to
    # that ceiling, so CHUNK_SIZE did not actually control normal chunk size.
    bounded_paras: List[str] = []
    for para in paras:
        bounded_paras.extend(_split_oversized_paragraph(para, target_chars))

    chunks: List[str] = []
    cur: List[str] = []
    cur_len = 0
    for p in bounded_paras:
        plen = len(p)
        if not cur:
            cur.append(p); cur_len = plen
            continue
        if cur_len + 2 + plen <= target_chars:
            cur.append(p); cur_len += 2 + plen
        else:
            chunks.append("\n\n".join(cur))
            cur = [p]; cur_len = plen
    if cur:
        chunks.append("\n\n".join(cur))
    return chunks


def _overlap_tail(value: str, max_chars: int) -> str:
    """Return a readable, bounded tail without usually starting mid-token."""
    value = (value or "").strip()
    if max_chars <= 0 or not value:
        return ""
    if len(value) <= max_chars:
        return value

    tail = value[-max_chars:]
    boundary = re.search(r"\s+", tail)
    if boundary and boundary.end() < len(tail):
        readable = tail[boundary.end():].strip()
        # Do not discard most of a small overlap merely to find whitespace.
        if len(readable) >= max(16, max_chars // 2):
            return readable
    return tail.strip()


def _with_overlap(parts: List[str], overlap_chars: int, hard_max: int) -> List[str]:
    """Repeat a small tail across adjacent chunks in one logical section."""
    if overlap_chars <= 0 or len(parts) < 2:
        return list(parts)

    output = [parts[0]]
    previous_original = parts[0]
    for current in parts[1:]:
        capacity = max(0, hard_max - len(current) - 2)
        overlap = _overlap_tail(previous_original, min(overlap_chars, capacity))
        output.append(f"{overlap}\n\n{current}" if overlap else current)
        previous_original = current
    return output

# ---------- main API ----------
def split_with_breadcrumbs(text: str, file_title: str) -> Tuple[List[str], List[Dict]]:
    """
    Preferred: split by Markdown headers. If none, fall back to paragraph packing.
    Adds a breadcrumb line at the top of each chunk.
    Returns (chunks, metadatas).
    """
    text = clean_markdown(text)
    if not text:
        return [], []

    # 1) Header-based splitting (preferred)
    headers = _iter_headers(text)
    chunks: List[str] = []
    metas: List[Dict] = []

    if len(headers) > 1:  # at least one header
        stack: List[str] = []
        sections: List[Tuple[str, str, Dict]] = []

        # Markdown commonly begins with an abstract, warning, or executive
        # summary before its first heading.  Treat that region as a real
        # Overview section instead of silently dropping potentially decisive
        # evidence.  It enters the same deduplication and bounded subdivision
        # pipeline as every headed section below.
        first_header_start = headers[0][2]
        preamble = text[:first_header_start].strip()
        if preamble:
            sections.append((
                _bounded_breadcrumb(file_title, ["Overview"]),
                preamble,
                {
                    "title": file_title,
                    "section": "Overview",
                    "section_slug": "overview",
                    "content_kind": "preamble",
                },
            ))

        for idx in range(len(headers) - 1):
            level, title, start = headers[idx]
            _, _, nxt = headers[idx + 1]

            # maintain breadcrumb stack
            while len(stack) >= level:
                if stack:
                    stack.pop()
                else:
                    break
            stack.append(title)

            body = text[start:nxt].strip()
            # remove the leading header line once
            body_stripped = HEADER_RE.sub("", body, count=1).strip()
            # if there is no body under the header, keep the header title as minimal content
            if not body_stripped:
                body_stripped = title

            breadcrumb = _bounded_breadcrumb(file_title, stack)
            sections.append((
                breadcrumb,
                body_stripped,
                {
                    "title": file_title,
                    "section": " / ".join(stack),
                    "section_slug": "/".join(_slug(t) for t in stack if t != "__END__"),
                },
            ))

        # Deduplicate complete logical sections before subdividing them.  This
        # preserves every ordered part of one long section, even when its parts
        # share the same breadcrumb or start with similar boilerplate.
        # Use the unabridged title/section metadata as the deduplication scope.
        # Rendered breadcrumbs may be middle-elided and are display only; two
        # distinct long paths can legitimately render to the same string.
        section_bodies = [body for _, body, _ in sections]
        section_scopes = [
            "\u0000".join(
                str(meta.get(key) or "")
                for key in ("title", "section", "content_kind")
            )
            for _, _, meta in sections
        ]
        keep = _dedup_near_duplicates(section_bodies, section_scopes)
        for (breadcrumb, body, meta), should_keep in zip(sections, keep):
            if not should_keep:
                continue

            prefix = f"[{breadcrumb}]\n\n"
            body_hard_max = max(1, _HARD_MAX_CHARS - len(prefix))
            body_target = max(1, min(body_hard_max, _TARGET_CHARS - len(prefix)))
            bodies = (
                [body]
                if len(body) <= body_target
                else _pack_paragraphs(
                    _paragraphs(body),
                    target_chars=body_target,
                    hard_max=body_hard_max,
                )
            )
            bodies = _with_overlap(
                bodies,
                min(_CHUNK_OVERLAP_CHARS, max(0, body_hard_max - body_target)),
                body_hard_max,
            )
            for part in bodies:
                chunks.append(f"{prefix}{part}".strip())
                metas.append(dict(meta))

        if chunks:
            return chunks, metas

    # 2) Fallback: paragraph/fixed-length splitting (headerless docs)
    paras = _paragraphs(text)
    if not paras:
        return [], []

    # Filter logical paragraphs before packing so hard-split pieces originating
    # in one paragraph cannot be mistaken for duplicates of each other.
    para_keep = _dedup_near_duplicates(paras)
    paras = [p for p, k in zip(paras, para_keep) if k]

    breadcrumb = _bounded_breadcrumb(file_title, ["Overview"])
    prefix = f"[{breadcrumb}]\n\n"
    body_hard_max = max(1, _HARD_MAX_CHARS - len(prefix))
    body_target = max(1, min(body_hard_max, _TARGET_CHARS - len(prefix)))
    packed = _pack_paragraphs(
        paras,
        target_chars=body_target,
        hard_max=body_hard_max,
    )
    packed = _with_overlap(
        packed,
        min(_CHUNK_OVERLAP_CHARS, max(0, body_hard_max - body_target)),
        body_hard_max,
    )

    out_chunks: List[str] = []
    out_meta: List[Dict] = []
    for i, body in enumerate(packed):
        combined = f"{prefix}{body}".strip()
        out_chunks.append(combined)
        out_meta.append({
            "title": file_title,
            "section": "Overview",
            "section_slug": "overview",
            "chunk_index": i,
        })

    return out_chunks, out_meta
