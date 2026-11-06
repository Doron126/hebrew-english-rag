"""Opaque source identity and safe tenant-document resolution helpers.

Only tenant-relative paths are stored in Chroma.  Browser-facing routes use a
random, non-guessable source ID and resolve the path again beneath the
operator-configured document root after tenant authorization.
"""

from __future__ import annotations

import mimetypes
import hashlib
import hmac
import os
from pathlib import Path, PurePosixPath
import re
import secrets
import stat
import unicodedata
from urllib.parse import quote, unquote
from typing import Any, BinaryIO, Dict, Iterable, Optional, Tuple


SOURCE_LINK_SCHEMA_VERSION = 2
SOURCE_ID_RE = re.compile(r"^[A-Za-z0-9_-]{43}$")

_PAGE_RE = re.compile(r"(?:^|\s/\s)Page\s+([1-9][0-9]{0,6})(?:\s/\s|$)", re.I)
_SHEET_RE = re.compile(r"(?:^|\s/\s)Sheet:\s*([^/]{1,255}?)(?:\s/\s|$)", re.I)
_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

_INLINE_IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}
_INLINE_TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".log", ".csv", ".tsv"}
_ACTIVE_SUFFIXES = {
    ".html", ".htm", ".xhtml", ".svg", ".js", ".mjs", ".cjs",
    ".sh", ".bash", ".zsh", ".fish", ".ps1", ".psm1", ".bat",
    ".cmd", ".com", ".exe", ".msi", ".py", ".pl", ".php", ".rb",
}
_OFFICE_TYPES = {
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


class SourceAccessError(ValueError):
    """A source identifier or stored source path failed closed."""


def new_source_id() -> str:
    """Return a 256-bit URL-safe opaque ID suitable for persistent metadata."""
    value = secrets.token_urlsafe(32)
    if not SOURCE_ID_RE.fullmatch(value):  # pragma: no cover - stdlib invariant
        raise RuntimeError("generated source ID did not match the expected format")
    return value


def validate_source_id(value: str) -> str:
    source_id = str(value or "").strip()
    if not SOURCE_ID_RE.fullmatch(source_id):
        raise SourceAccessError("invalid source identifier")
    return source_id


def existing_source_id(records: Iterable[Tuple[str, Dict[str, Any]]]) -> Optional[str]:
    """Reuse one coherent prior ID so ordinary re-ingestion keeps links stable."""
    identifiers = {
        str((metadata or {}).get("source_id") or "")
        for _record_id, metadata in records
        if SOURCE_ID_RE.fullmatch(str((metadata or {}).get("source_id") or ""))
    }
    return next(iter(identifiers)) if len(identifiers) == 1 else None


def guessed_mime_type(source_relative_path: str) -> str:
    suffix = PurePosixPath(str(source_relative_path or "")).suffix.casefold()
    if suffix in _OFFICE_TYPES:
        return _OFFICE_TYPES[suffix]
    guessed, _encoding = mimetypes.guess_type(str(source_relative_path or ""), strict=False)
    return guessed or "application/octet-stream"


def location_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Extract explicit source locations from splitter metadata."""
    section = str((metadata or {}).get("section") or "").strip()
    page_value = (metadata or {}).get("page")
    page: Optional[int] = None
    if isinstance(page_value, int) and not isinstance(page_value, bool) and page_value > 0:
        page = page_value
    elif section:
        match = _PAGE_RE.search(section)
        if match:
            page = int(match.group(1))

    sheet = str((metadata or {}).get("sheet") or "").strip()
    if not sheet and section:
        match = _SHEET_RE.search(section)
        if match:
            sheet = match.group(1).strip()

    slide_value = (metadata or {}).get("slide")
    slide = slide_value if isinstance(slide_value, int) and not isinstance(slide_value, bool) and slide_value > 0 else None
    if slide is None and section:
        match = re.search(r"(?:^| / )Slide ([1-9][0-9]{0,6})(?:\D|$)", section, re.I)
        if match:
            slide = int(match.group(1))

    return {
        "page": page,
        "slide": slide,
        "sheet": sheet or None,
        "section": section or None,
    }


def source_metadata_complete(metadata: Any, tenant_id: str) -> bool:
    """Return whether a Chroma row can safely back an authenticated link."""
    if not isinstance(metadata, dict):
        return False
    try:
        source_id = metadata.get("source_id")
        relative_value = metadata.get("source_relative_path")
        if not isinstance(source_id, str) or not isinstance(relative_value, str):
            return False
        validate_source_id(source_id)
        relative = _validate_relative_path(relative_value)
    except SourceAccessError:
        return False

    display_name = metadata.get("display_name")
    mime_type = metadata.get("mime_type")
    source_sha256 = metadata.get("source_sha256")
    source_size = metadata.get("source_size")
    source_mtime_ns = metadata.get("source_mtime_ns")
    return (
        metadata.get("source_link_schema_version") == SOURCE_LINK_SCHEMA_VERSION
        and metadata.get("tenant_id") == tenant_id
        and isinstance(display_name, str)
        and display_name == relative.name
        and isinstance(mime_type, str)
        and mime_type == guessed_mime_type(relative.as_posix())
        and isinstance(source_sha256, str)
        and _SHA256_RE.fullmatch(source_sha256) is not None
        and isinstance(source_size, int)
        and not isinstance(source_size, bool)
        and source_size >= 0
        and isinstance(source_mtime_ns, int)
        and not isinstance(source_mtime_ns, bool)
        and source_mtime_ns >= 0
    )


def _unsafe_path_shape(value: str) -> bool:
    if not value or "\\" in value or _CONTROL_RE.search(value):
        return True
    if value.startswith(("/", "//")) or _DRIVE_RE.match(value):
        return True
    path = PurePosixPath(value)
    return path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts)


def _validate_relative_path(value: str) -> PurePosixPath:
    raw = unicodedata.normalize("NFC", str(value or "").strip())
    if _unsafe_path_shape(raw):
        raise SourceAccessError("invalid source path")

    # Stored paths are not URL paths. Reject encoded traversal/absolute forms
    # as well, including double encoding, rather than normalizing them.
    decoded = raw
    for _ in range(3):
        next_value = unquote(decoded)
        if next_value == decoded:
            break
        decoded = next_value
        if _unsafe_path_shape(decoded):
            raise SourceAccessError("invalid encoded source path")
    return PurePosixPath(raw)


def _validate_tenant_component(value: str) -> str:
    """Require one normalized path component even after API authorization."""
    raw = unicodedata.normalize("NFC", str(value or "").strip())
    if _unsafe_path_shape(raw):
        raise SourceAccessError("invalid tenant path")
    path = PurePosixPath(raw)
    if len(path.parts) != 1:
        raise SourceAccessError("invalid tenant path")
    return raw


def resolve_source_file(
    document_root: str | Path,
    tenant_id: str,
    source_relative_path: str,
) -> Path:
    """Resolve an indexed source beneath ``document_root/tenant_id``.

    ``Path.resolve(strict=True)`` applies to the root, tenant directory, and
    final file.  A symlink at any level that escapes either allowed root is
    therefore rejected before the file is returned.
    """
    relative = _validate_relative_path(source_relative_path)
    tenant_component = _validate_tenant_component(tenant_id)
    try:
        root = Path(document_root).expanduser().resolve(strict=True)
        tenant_root = (root / tenant_component).resolve(strict=True)
        tenant_root.relative_to(root)
        candidate = tenant_root.joinpath(*relative.parts).resolve(strict=True)
        candidate.relative_to(tenant_root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise SourceAccessError("source is unavailable") from exc
    if not candidate.is_file():
        raise SourceAccessError("source is unavailable")
    return candidate


def open_source_file(
    document_root: str | Path,
    tenant_id: str,
    source_relative_path: str,
) -> Tuple[PurePosixPath, BinaryIO, os.stat_result]:
    """Open a regular source without a pathname check/open race.

    Linux directory file descriptors anchor every lookup. ``O_NOFOLLOW`` is
    applied to the tenant, each intermediate directory, and the final file, so
    a symlink swap cannot redirect the opened descriptor outside the authorized
    tenant root after validation.
    """
    relative = _validate_relative_path(source_relative_path)
    tenant_component = _validate_tenant_component(tenant_id)
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise SourceAccessError("safe source opening is unavailable")

    try:
        root = Path(document_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise SourceAccessError("source is unavailable")
    except (OSError, RuntimeError, ValueError) as exc:
        raise SourceAccessError("source is unavailable") from exc

    close_on_exec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | close_on_exec
    file_flags = (
        os.O_RDONLY
        | os.O_NOFOLLOW
        | close_on_exec
        | getattr(os, "O_NONBLOCK", 0)
    )
    directory_fds = []
    file_fd: Optional[int] = None
    try:
        current_fd = os.open(root, directory_flags)
        directory_fds.append(current_fd)
        for component in (tenant_component, *relative.parts[:-1]):
            current_fd = os.open(component, directory_flags, dir_fd=current_fd)
            directory_fds.append(current_fd)

        file_fd = os.open(relative.parts[-1], file_flags, dir_fd=current_fd)
        file_stat = os.fstat(file_fd)
        if not stat.S_ISREG(file_stat.st_mode):
            raise SourceAccessError("source is unavailable")
        source_handle = os.fdopen(file_fd, "rb", closefd=True)
        file_fd = None
        return relative, source_handle, file_stat
    except (OSError, RuntimeError, ValueError) as exc:
        raise SourceAccessError("source is unavailable") from exc
    finally:
        if file_fd is not None:
            try:
                os.close(file_fd)
            except OSError:
                pass
        for directory_fd in reversed(directory_fds):
            try:
                os.close(directory_fd)
            except OSError:
                pass


def source_stat_matches_metadata(metadata: Any, file_stat: os.stat_result) -> bool:
    """Bind the opened descriptor to the exact file generation that was indexed."""
    if not isinstance(metadata, dict) or not stat.S_ISREG(file_stat.st_mode):
        return False
    source_size = metadata.get("source_size")
    source_mtime_ns = metadata.get("source_mtime_ns")
    return (
        isinstance(source_size, int)
        and not isinstance(source_size, bool)
        and source_size >= 0
        and isinstance(source_mtime_ns, int)
        and not isinstance(source_mtime_ns, bool)
        and source_mtime_ns >= 0
        and source_size == file_stat.st_size
        and source_mtime_ns == file_stat.st_mtime_ns
    )


def source_sha256_matches_metadata(
    metadata: Any,
    source_handle: BinaryIO,
    *,
    enabled: bool,
    chunk_size: int = 1024 * 1024,
) -> bool:
    """Verify the indexed digest using the already-authorized descriptor.

    The descriptor is always rewound before returning, including on malformed
    metadata, read errors, and mismatches.  Callers can therefore stream the
    same descriptor without reopening an attacker-controlled pathname.
    """
    try:
        source_handle.seek(0)
        if not enabled:
            return True
        expected = metadata.get("source_sha256") if isinstance(metadata, dict) else None
        if not isinstance(expected, str) or _SHA256_RE.fullmatch(expected) is None:
            return False
        digest = hashlib.sha256()
        while True:
            chunk = source_handle.read(max(1, int(chunk_size)))
            if not chunk:
                break
            digest.update(chunk)
        return hmac.compare_digest(digest.hexdigest(), expected)
    except (OSError, ValueError, TypeError):
        return False
    finally:
        try:
            source_handle.seek(0)
        except (OSError, ValueError):
            pass


def safe_display_name(value: str, fallback: str) -> str:
    """Return a bounded header-safe Unicode basename without path material."""
    candidate = unicodedata.normalize("NFC", str(value or "")).replace("\\", "/")
    candidate = candidate.rsplit("/", 1)[-1].strip()
    candidate = _CONTROL_RE.sub("", candidate)
    if not candidate or candidate in {".", ".."}:
        candidate = unicodedata.normalize("NFC", str(fallback or "document"))
        candidate = candidate.replace("\\", "/").rsplit("/", 1)[-1].strip()
        candidate = _CONTROL_RE.sub("", candidate) or "document"
    return candidate[:240]


def response_media_policy(path: Path) -> Tuple[str, str]:
    """Return ``(media_type, disposition)`` for same-origin source delivery."""
    suffix = path.suffix.casefold()
    if suffix == ".pdf":
        return "application/pdf", "inline"
    if suffix in _INLINE_IMAGE_TYPES:
        return _INLINE_IMAGE_TYPES[suffix], "inline"
    if suffix in _INLINE_TEXT_SUFFIXES:
        # Preserve the original bytes, including Windows-1255 and UTF-16,
        # without falsely labelling them UTF-8 or rendering active-looking
        # text in the application's same-origin browsing context.
        return "application/octet-stream", "attachment"
    if suffix in _ACTIVE_SUFFIXES:
        return "application/octet-stream", "attachment"
    if suffix in _OFFICE_TYPES:
        return _OFFICE_TYPES[suffix], "attachment"
    return "application/octet-stream", "attachment"


def content_disposition_value(disposition: str, filename: str) -> str:
    """Build an RFC 5987-compatible header with a conservative ASCII fallback."""
    kind = "inline" if disposition == "inline" else "attachment"
    safe_name = safe_display_name(filename, "document")
    ascii_name = unicodedata.normalize("NFKD", safe_name).encode("ascii", "ignore").decode("ascii")
    ascii_name = re.sub(r"[^A-Za-z0-9._ -]+", "_", ascii_name).strip(" .") or "document"
    ascii_name = ascii_name.replace("\\", "_").replace('"', "_")[:120]
    encoded = quote(safe_name.encode("utf-8"), safe="!#$&+-.^_`|~")
    return f"{kind}; filename=\"{ascii_name}\"; filename*=UTF-8''{encoded}"


def parse_single_byte_range(value: str, size: int) -> Optional[Tuple[int, int]]:
    """Parse one HTTP byte range and return inclusive ``(start, end)``."""
    raw = str(value or "").strip()
    if not raw:
        return None
    if size < 0 or not raw.startswith("bytes=") or "," in raw:
        raise SourceAccessError("invalid byte range")
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", raw)
    if not match or (not match.group(1) and not match.group(2)):
        raise SourceAccessError("invalid byte range")
    start_text, end_text = match.groups()
    if start_text:
        start = int(start_text)
        end = int(end_text) if end_text else size - 1
    else:
        suffix_length = int(end_text)
        if suffix_length <= 0:
            raise SourceAccessError("invalid byte range")
        start = max(0, size - suffix_length)
        end = size - 1
    if size == 0 or start < 0 or start >= size or end < start:
        raise SourceAccessError("unsatisfiable byte range")
    return start, min(end, size - 1)

def docx_source_body(document_root, tenant_id, metadata):
    """Read authenticated original body with the ingestion parser, no ingestion.

    Tables and content controls retain source order. Supplemental Office parts
    are explicitly reported as partial rather than silently called complete.
    """
    import io, zipfile
    from ingest import _docx_render_container, _docx_styles, _parse_xml_part, _read_zip_part
    relative, handle, before = open_source_file(document_root, tenant_id, metadata.get('source_relative_path', ''))
    with handle:
        if relative.suffix.casefold() != '.docx' or not source_stat_matches_metadata(metadata, before):
            raise SourceAccessError('original source generation changed')
        if before.st_size > 32 * 1024 * 1024:
            return '', False
        payload = handle.read(32 * 1024 * 1024 + 1)
        if hashlib.sha256(payload).hexdigest() != metadata.get('source_sha256'):
            raise SourceAccessError('original source checksum changed')
        after = os.fstat(handle.fileno())
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise SourceAccessError('original source changed during reading')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos = archive.infolist()
        if (len(infos) > 4096 or len({info.filename for info in infos}) != len(infos)
                or sum(info.file_size for info in infos) > 64 * 1024 * 1024):
            raise SourceAccessError('unsupported original Office archive')
        styles = _docx_styles(archive)
        root = _parse_xml_part(_read_zip_part(archive, 'word/document.xml', required=True), 'word/document.xml')
        body = root.find('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}body')
        if body is None:
            return '', False
        text = _docx_render_container(body, styles)
        supplemental = any(re.fullmatch(r'word/(?:header\d+|footer\d+|footnotes|endnotes)\.xml', name) for name in archive.namelist())
        return text, not supplemental


def pdf_source_pages(document_root, tenant_id, metadata, pages, *, include_geometry=False):
    """Read bounded native PDF pages from the authenticated source generation.

    This changes no indexed records. Returned text is deterministic native
    text with the existing logical-order normalization; images and drawings
    are never rendered or interpreted. Callers retain the real PDF page.
    """
    if type(include_geometry) is not bool:
        raise SourceAccessError('native PDF geometry option is invalid')
    if (not source_metadata_complete(metadata, tenant_id)
            or str(metadata.get('mime_type') or '').casefold() != 'application/pdf'):
        raise SourceAccessError('native PDF source identity is unavailable')
    if (not isinstance(pages, (tuple, list, set)) or not pages or len(pages) > 8
            or any(type(page) is not int or page < 1 for page in pages)):
        raise SourceAccessError('native PDF page request is invalid')
    requested = tuple(sorted(set(pages)))
    relative, handle, before = open_source_file(document_root, tenant_id,
                                               metadata.get('source_relative_path', ''))
    with handle:
        if relative.suffix.casefold() != '.pdf' or not source_stat_matches_metadata(metadata, before):
            raise SourceAccessError('original PDF generation changed')
        if before.st_size > 32 * 1024 * 1024:
            raise SourceAccessError('original PDF exceeds native reading bound')
        payload = handle.read(32 * 1024 * 1024 + 1)
        if hashlib.sha256(payload).hexdigest() != metadata.get('source_sha256'):
            raise SourceAccessError('original PDF checksum changed')
        after = os.fstat(handle.fileno())
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise SourceAccessError('original PDF changed during reading')
    from canonical_text import canonicalize_source_text
    try:
        import fitz
    except ImportError as exc:
        raise SourceAccessError('native PDF text reader is unavailable') from exc
    result = []
    total_chars = 0
    try:
        with fitz.open(stream=payload, filetype='pdf') as document:
            if document.needs_pass:
                raise SourceAccessError('original PDF requires a password')
            for number in requested:
                if number > document.page_count:
                    continue
                # Preserve the PDF's declared content order. Coordinate sorting can
                # reverse mixed RTL/LTR role and identifier runs before normalization.
                native_page = document.load_page(number-1)
                raw = native_page.get_text('text', sort=False)
                text = canonicalize_source_text(raw, repair_visual_bidi=True,
                    normalize_pdf_spacing=True, join_pdf_line_wraps=False).text
                total_chars += len(raw) + len(text)
                if total_chars > 320_000:
                    raise SourceAccessError('native PDF page text exceeds reading bound')
                result.append(dict(page=number, text=text,
                    raw_sha256=hashlib.sha256(raw.encode('utf-8')).hexdigest(),
                    text_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
                    page_count=document.page_count))
                if include_geometry:
                    # Native text/rules only. No raster interpretation or index
                    # mutation; the authenticated source and page bounds above
                    # also govern the monetary association view.
                    words = tuple(tuple(word) for word in native_page.get_text('words', sort=False))
                    drawings = native_page.get_drawings()
                    if len(words) > 4000 or len(drawings) > 12000:
                        raise SourceAccessError('native PDF geometry exceeds reading bound')
                    vectors = set()
                    for drawing in drawings:
                        for item in drawing.get('items', ()):
                            if item[0] == 're':
                                vectors.add(('rect', *(float(value) for value in item[1])))
                            elif item[0] == 'l':
                                first, last = item[1:3]
                                if abs(float(first.y) - float(last.y)) < .05:
                                    vectors.add(('line', min(float(first.x), float(last.x)),
                                        float(first.y), max(float(first.x), float(last.x)), float(last.y)))
                            if len(vectors) > 20000:
                                raise SourceAccessError('native PDF geometry exceeds reading bound')
                    blocks = tuple(tuple(block) for block in native_page.get_text('blocks', sort=False)
                        if len(block) == 7 and block[6] == 0)
                    if len(blocks) > 4000:
                        raise SourceAccessError('native PDF geometry exceeds reading bound')
                    result[-1].update(words=words, native_vectors=tuple(sorted(vectors)), native_blocks=blocks,
                        width=float(native_page.rect.width), height=float(native_page.rect.height))
    except SourceAccessError:
        raise
    except (RuntimeError, ValueError) as exc:
        raise SourceAccessError('original PDF native text is unavailable') from exc
    if not result:
        raise SourceAccessError('native PDF page is outside the source')
    return tuple(result)


def docx_source_paragraphs(document_root, tenant_id, metadata):
    """Read original paragraph positions after authenticating the indexed file.

    This is citation location lookup only; it changes no indexed content.
    Empty paragraphs retain their original ordinal. Unsafe/oversized XML fails
    closed instead of guessing a paragraph position from flattened chunks.
    """
    import hashlib, io, zipfile
    import xml.etree.ElementTree as ET
    relative, handle, before = open_source_file(document_root, tenant_id, metadata.get("source_relative_path", ""))
    with handle:
        if relative.suffix.casefold() != ".docx":
            return ()
        if not source_stat_matches_metadata(metadata, before):
            raise ValueError("original source generation changed")
        if before.st_size > 32 * 1024 * 1024:
            return ()
        payload = handle.read(32 * 1024 * 1024 + 1)
        if hashlib.sha256(payload).hexdigest() != metadata.get("source_sha256"):
            raise ValueError("original source checksum changed")
        after = os.fstat(handle.fileno())
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError("original source changed during paragraph lookup")
    try:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            info = archive.getinfo("word/document.xml")
            if info.file_size > 8 * 1024 * 1024:
                return ()
            xml = archive.read(info)
        if b"<!DOCTYPE" in xml or b"<!ENTITY" in xml:
            return ()
        root = ET.fromstring(xml)
        namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        body = root.find("w:body", namespace)
        if body is None:
            return ()
        paragraphs = []
        for paragraph in body.findall("w:p", namespace):
            paragraphs.append("".join(text.text or "" for text in paragraph.findall(".//w:t", namespace)))
            if len(paragraphs) > 100_000:
                return ()
        return tuple(paragraphs)
    except (KeyError, ValueError, zipfile.BadZipFile, ET.ParseError):
        return ()
