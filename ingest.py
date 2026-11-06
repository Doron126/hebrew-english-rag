# rag/ingest.py
import argparse
from bisect import bisect_right
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timedelta
from html.parser import HTMLParser
import hashlib
import io
import json
import math
import os
import posixpath
import re
import shutil
import subprocess
import struct
import sys
import tempfile
import time
import unicodedata
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional, Union
import xml.etree.ElementTree as ET

try:
    from .config_env import (
        DEFAULT_EMBED_MODEL,
        embedding_model_kwargs,
        EMBEDDING_FORMAT_REVISION,
        format_embedding_document,
        load_project_environment,
        validate_tenant_id,
    )
    from .source_access import (
        SOURCE_LINK_SCHEMA_VERSION,
        existing_source_id,
        guessed_mime_type,
        location_metadata,
        new_source_id,
        source_metadata_complete,
    )
    from .canonical_text import (
        CANONICAL_TEXT_REVISION,
        CanonicalEdit,
        CanonicalText,
        canonicalize_reconstructed_source,
        canonicalize_source_text,
        extraction_audit_id,
        persist_extraction_audit,
    )
    from .pdf_reconstruction import (
        PDF_RECONSTRUCTION_REVISION,
        positioned_tokens_from_pymupdf,
        pymupdf_line_styles,
        reconstruct_page,
    )
    from .proposition_quality import PROPOSITION_SCHEMA_REVISION
    from .source_structure import SOURCE_STRUCTURE_REVISION, sentence_spans
except ImportError:
    from config_env import (
        DEFAULT_EMBED_MODEL,
        embedding_model_kwargs,
        EMBEDDING_FORMAT_REVISION,
        format_embedding_document,
        load_project_environment,
        validate_tenant_id,
    )
    from source_access import (
        SOURCE_LINK_SCHEMA_VERSION,
        existing_source_id,
        guessed_mime_type,
        location_metadata,
        new_source_id,
        source_metadata_complete,
    )
    from canonical_text import (
        CANONICAL_TEXT_REVISION,
        CanonicalEdit,
        CanonicalText,
        canonicalize_reconstructed_source,
        canonicalize_source_text,
        extraction_audit_id,
        persist_extraction_audit,
    )
    from pdf_reconstruction import (
        PDF_RECONSTRUCTION_REVISION,
        positioned_tokens_from_pymupdf,
        pymupdf_line_styles,
        reconstruct_page,
    )
    from proposition_quality import PROPOSITION_SCHEMA_REVISION
    from source_structure import SOURCE_STRUCTURE_REVISION, sentence_spans

try:
    from .pdf_layout import PDF_LAYOUT_REVISION, PDF_PACKING_REVISION, PDF_NATIVE_ROW_ASSOCIATION_REVISION, reconstruct_document as reconstruct_pdf_layout, paragraph_packing_groups, packing_group_metadata
except ImportError:
    from pdf_layout import PDF_LAYOUT_REVISION, PDF_PACKING_REVISION, PDF_NATIVE_ROW_ASSOCIATION_REVISION, reconstruct_document as reconstruct_pdf_layout, paragraph_packing_groups, packing_group_metadata

load_project_environment(Path(__file__).resolve().parent)

try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

try:
    from PIL import Image
    import pytesseract
except ImportError:
    Image = None
    pytesseract = None


from chromadb import PersistentClient
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

# Use the shared splitter/cleaner so behavior is consistent everywhere.
try:
    from .splitter import (
        _CHUNK_OVERLAP_CHARS,
        _HARD_MAX_CHARS,
        _TARGET_CHARS,
        split_with_breadcrumbs,
        clean_markdown,
    )
except ImportError:
    try:
        # Preserve compatibility with the original rag/ package layout.
        from rag.splitter import (
            _CHUNK_OVERLAP_CHARS,
            _HARD_MAX_CHARS,
            _TARGET_CHARS,
            split_with_breadcrumbs,
            clean_markdown,
        )
    except ImportError:
        # The distributed demo snapshot keeps all modules at repository root.
        from splitter import (
            _CHUNK_OVERLAP_CHARS,
            _HARD_MAX_CHARS,
            _TARGET_CHARS,
            split_with_breadcrumbs,
            clean_markdown,
        )

# ---------- Defaults from env ----------
CHROMA_PATH = os.getenv("CHROMA_PATH", "storage/chroma")
BASE_COLLECTION = os.getenv("RAG_COLLECTION", "company-knowledge")
TENANT_STRATEGY = os.getenv("TENANT_STRATEGY", "collection")  # "collection" | "metadata"
EMBED_MODEL = os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "cpu").strip() or "cpu"

# Bump this whenever extraction, splitting, or embedding-document formatting
# changes in a way that makes previously stored chunks stale.  It is persisted
# in every chunk and is part of the incremental-index validity check.
INGEST_SCHEMA_VERSION = 13
STRUCTURAL_ADJACENCY_REVISION = "same-source-record-v2"


# The cache exists only during one offline ingestion process.  It retains the
# raw coordinate extraction and its deterministic origin map long enough for
# `_chunks_for_file_with_trace` to compose logical-text-v3 audit provenance.
_PDF_RECONSTRUCTION_TRACE_CACHE: Dict[
    str,
    Tuple[str, str, Tuple[Optional[int], ...], Tuple[CanonicalEdit, ...], Dict[str, Any]],
] = {}


_SOURCE_EXTRACTION_METHODS: Dict[str, str] = {}
PDF_NATIVE_ONLY_REVISION = "native-text-no-ocr-v1"


class IngestExtractionError(RuntimeError):
    """Raised when a source could not be extracted safely.

    An extraction failure is deliberately different from a genuinely empty
    text file.  Callers may prune old chunks for the latter, but must retain the
    last known-good index for the former.
    """


class IngestNoTextError(IngestExtractionError):
    """Extraction completed safely but found no textual content."""


def _validated_tenant_strategy(value: Optional[str] = None) -> str:
    strategy = (TENANT_STRATEGY if value is None else str(value)).strip().lower()
    if strategy not in {"collection", "metadata"}:
        raise ValueError(
            f"Invalid TENANT_STRATEGY={strategy!r}; expected 'collection' or 'metadata'"
        )
    return strategy


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int, *, minimum: int = 0, maximum: Optional[int] = None) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    value = max(minimum, value)
    if maximum is not None:
        value = min(maximum, value)
    return value


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        value = float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, value)


def _ocr_language() -> str:
    """Return the configured Tesseract languages, favoring Hebrew + English."""
    language = os.getenv("OCR_LANGUAGE", "heb+eng").strip() or "heb+eng"
    # Tesseract language packs use short ASCII identifiers joined with `+`.
    # Reject malformed/oversized values instead of forwarding arbitrary input
    # to the OCR subprocess wrapper.
    if not re.fullmatch(r"[A-Za-z0-9_.+-]{1,128}", language):
        return "heb+eng"
    return language


_PPTX_EXTRACTOR_REVISION = 'presentation-order-tables-v2'


def _pipeline_fingerprint(
    embed_model: str, *, source_suffix: str = "", pdf_native_only: bool = False,
) -> str:
    """Fingerprint every setting that can materially change stored chunks.

    A source-byte hash alone is insufficient: changing chunk boundaries, OCR,
    extraction order, or embedding prefixes must force a fresh generation.
    """
    payload = {
        "ingest_schema": INGEST_SCHEMA_VERSION,
        "source_link_schema": SOURCE_LINK_SCHEMA_VERSION,
        "chunk_size": int(_TARGET_CHARS),
        "chunk_overlap": int(_CHUNK_OVERLAP_CHARS),
        "chunk_hard_max": int(_HARD_MAX_CHARS),
        "pdf_extractor": os.getenv("PDF_EXTRACTOR", "auto").strip().lower(),
        "pdf_ocr_fallback": _env_flag("PDF_OCR_FALLBACK", True),
        "pdf_ocr_min_chars": _env_int("PDF_OCR_MIN_CHARS", 80),
        "pdf_ocr_min_words": _env_int("PDF_OCR_MIN_WORDS", 10),
        "pdf_ocr_min_quality": _env_float("PDF_OCR_MIN_QUALITY", 12.0),
        "pdf_ocr_dpi": _env_int("PDF_OCR_DPI", 220, minimum=72, maximum=600),
        "pdf_ocr_max_pages": _env_int("PDF_OCR_MAX_PAGES", 100),
        "ocr_language": _ocr_language(),
        "ocr_max_pixels": _env_int(
            "OCR_MAX_IMAGE_PIXELS",
            40_000_000,
            minimum=1_000_000,
            maximum=200_000_000,
        ),
        "ocr_timeout_seconds": _env_int(
            "OCR_TIMEOUT_SECONDS", 60, minimum=0, maximum=3600
        ),
        "ocr_max_file_bytes": _env_int(
            "OCR_MAX_FILE_BYTES",
            100 * 1024 * 1024,
            minimum=1,
            maximum=2 * 1024 * 1024 * 1024,
        ),
        "max_text_file_bytes": _env_int(
            "INGEST_MAX_TEXT_FILE_BYTES",
            64 * 1024 * 1024,
            minimum=1,
            maximum=2 * 1024 * 1024 * 1024,
        ),
        "max_structured_file_bytes": _env_int(
            "INGEST_MAX_STRUCTURED_FILE_BYTES",
            512 * 1024 * 1024,
            minimum=1,
            maximum=4 * 1024 * 1024 * 1024,
        ),
        "max_archive_entries": _env_int(
            "INGEST_MAX_ARCHIVE_ENTRIES", 10000, minimum=1, maximum=100000
        ),
        "max_archive_uncompressed_bytes": _env_int(
            "INGEST_MAX_ARCHIVE_UNCOMPRESSED_BYTES",
            512 * 1024 * 1024,
            minimum=1,
            maximum=4 * 1024 * 1024 * 1024,
        ),
        "max_xml_part_bytes": _env_int(
            "INGEST_MAX_XML_PART_BYTES",
            64 * 1024 * 1024,
            minimum=1,
            maximum=512 * 1024 * 1024,
        ),
        "max_extracted_text_chars": _env_int(
            "INGEST_MAX_EXTRACTED_TEXT_CHARS",
            64 * 1024 * 1024,
            minimum=1,
            maximum=2 * 1024 * 1024 * 1024,
        ),
        "xlsx_max_cells": _env_int(
            "XLSX_MAX_CELLS", 2_000_000, minimum=1, maximum=10_000_000
        ),
        "xlsx_rows_per_section": _env_int(
            "XLSX_ROWS_PER_SECTION", 25, minimum=1, maximum=1000
        ),
        "xlsx_header_scan_rows": _env_int(
            "XLSX_HEADER_SCAN_ROWS", 20, minimum=1, maximum=100
        ),
        "csv_header_mode": os.getenv("CSV_HEADER_MODE", "auto").strip().lower(),
        "csv_header_scan_rows": _env_int(
            "CSV_HEADER_SCAN_ROWS", 10, minimum=1, maximum=100
        ),
        "csv_max_rows": _env_int(
            "CSV_MAX_ROWS", 1_000_000, minimum=1, maximum=10_000_000
        ),
        "csv_max_columns": _env_int(
            "CSV_MAX_COLUMNS", 10_000, minimum=1, maximum=100_000
        ),
        "csv_max_field_chars": _env_int(
            "CSV_MAX_FIELD_CHARS", 1_000_000, minimum=1, maximum=16_000_000
        ),
        "log_max_events": _env_int(
            "LOG_MAX_EVENTS", 1_000_000, minimum=1, maximum=10_000_000
        ),
        "log_events_per_section": _env_int(
            "LOG_EVENTS_PER_SECTION", 50, minimum=1, maximum=1000
        ),
        "max_chunks_per_file": _env_int(
            "INGEST_MAX_CHUNKS_PER_FILE", 20000, minimum=1, maximum=1000000
        ),
        "ps_text_extractor": os.getenv("PS_TEXT_EXTRACTOR", "auto").strip().lower(),
        "ps_max_file_bytes": _env_int(
            "INGEST_MAX_PS_FILE_BYTES",
            128 * 1024 * 1024,
            minimum=1,
            maximum=2 * 1024 * 1024 * 1024,
        ),
        "ps_timeout_seconds": _env_int(
            "PS_EXTRACT_TIMEOUT_SECONDS", 60, minimum=1, maximum=3600
        ),
        "ps_max_output_bytes": _env_int(
            "PS_MAX_OUTPUT_BYTES",
            64 * 1024 * 1024,
            minimum=1,
            maximum=512 * 1024 * 1024,
        ),
        "text_decoder_revision": "unicode-hebrew-v6",
        "canonical_text_revision": CANONICAL_TEXT_REVISION,
        "pdf_reconstruction_revision": PDF_RECONSTRUCTION_REVISION,
        "pdf_layout_revision": PDF_LAYOUT_REVISION,
        "pdf_packing_revision": PDF_PACKING_REVISION,
        "pdf_page_manifest_revision": "pdf-file-pages-v1",
        "source_structure_revision": SOURCE_STRUCTURE_REVISION,
        "proposition_schema_revision": PROPOSITION_SCHEMA_REVISION,
        "structured_extractor_revision": "office-html-tabular-code-v5",
        "pptx_extractor_revision": (_PPTX_EXTRACTOR_REVISION
                                    if source_suffix.lower() == ".pptx"
                                    else "presentation-order-tables-v1"),
        "ghostscript_available": any(
            shutil.which(name) for name in ("ps2ascii", "gs", "gswin64c", "gswin32c")
        ),
        "embed_model": str(embed_model),
        "embedding_format_revision": EMBEDDING_FORMAT_REVISION,
        "embedding_compute_dtype": embedding_model_kwargs(EMBED_DEVICE).get("dtype", "model-default"),
        "pypdf_available": pypdf is not None,
        "pymupdf_available": fitz is not None,
        "tesseract_wrapper_available": pytesseract is not None,
    }
    # Extraction-only changes invalidate the affected format, preserving
    # completed embeddings for unrelated source formats during incremental work.
    if source_suffix.lower() == ".pdf":
        payload["pdf_native_row_association_revision"] = PDF_NATIVE_ROW_ASSOCIATION_REVISION
        if pdf_native_only:
            payload["pdf_native_only_revision"] = PDF_NATIVE_ONLY_REVISION
    if source_suffix.lower() == ".docx":
        payload["docx_image_ocr_revision"] = _DOCX_IMAGE_OCR_REVISION
        payload["docx_ocr_max_images"] = _env_int("DOCX_OCR_MAX_IMAGES", 16, minimum=1, maximum=128)
        payload["emf_max_records"] = _env_int("EMF_MAX_RECORDS", 100000, minimum=1, maximum=1000000)
        payload["emf_max_text_chars"] = _env_int("EMF_MAX_TEXT_CHARS", 1000000, minimum=1, maximum=16000000)
        payload["docx_ocr_max_total_image_bytes"] = _env_int(
            "DOCX_OCR_MAX_TOTAL_IMAGE_BYTES", 64 * 1024 * 1024,
            minimum=1, maximum=512 * 1024 * 1024,
        )
        payload["docx_ocr_max_total_image_pixels"] = _env_int(
            "DOCX_OCR_MAX_TOTAL_IMAGE_PIXELS", 80_000_000,
            minimum=1, maximum=400_000_000,
        )
    encoded = json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalize_extracted_text(value: str) -> str:
    """Normalize extractor artifacts without rewriting factual content."""
    text = unicodedata.normalize("NFC", value or "")
    text = (
        text.replace("\ufeff", "")
        .replace("\u00ad", "")
        .replace("\x00", "")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n\n")
        .replace("\u00a0", " ")
        .replace("\u202f", " ")
    )
    # Directional controls can visually reorder identifiers, numbers, and
    # punctuation without changing the underlying string. Hebrew letters still
    # retain their natural RTL behavior after these invisible controls are
    # removed, while mixed Hebrew/English facts become stable for retrieval.
    text = re.sub(r"[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]", "", text)
    text = text.replace("\u200b", "")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove non-printing controls that poison tokenization while preserving
    # tabs/newlines and all visible bidi/Hebrew characters.
    text = "".join(
        ch
        for ch in text
        if ch in "\n\t" or unicodedata.category(ch) not in {"Cc", "Cs"}
    )
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip()


def _decoded_text_score(value: str) -> float:
    """Rank legacy text decodings, heavily penalizing binary/mojibake output."""
    if not value:
        return float("-inf")
    length = max(1, len(value))
    printable = sum(ch.isprintable() or ch in "\n\r\t" for ch in value) / length
    controls = sum(
        unicodedata.category(ch) in {"Cc", "Cs"} and ch not in "\n\r\t"
        for ch in value
    ) / length
    replacements = value.count("\ufffd") / length
    mojibake = len(re.findall(r"(?:Ã.|Â.|×.|Ø.)", value)) / length
    words = len(re.findall(r"[^\W_]+", value, flags=re.UNICODE))
    return (
        printable * 100.0
        + min(words, 1000) * 0.02
        - controls * 500.0
        - replacements * 1000.0
        - mojibake * 250.0
    )


def _decode_text_bytes(raw: bytes) -> str:
    """Decode UTF and common Hebrew legacy encodings without silent loss."""
    if not raw:
        return ""

    # Binary content masquerading as a supported text extension must not be
    # indexed as garbage or replace a previous good version.
    nul_ratio = raw.count(b"\x00") / len(raw)
    has_utf16_bom = raw.startswith((b"\xff\xfe", b"\xfe\xff"))
    bomless_unicode: List[str] = []
    if not has_utf16_bom and len(raw) >= 4 and len(raw) % 2 == 0:
        even_bytes = raw[0::2]
        odd_bytes = raw[1::2]
        # UTF-16 text in the Latin/Hebrew ranges has a consistently small
        # high byte (00 for Latin, 05 for Hebrew). This detects BOM-less files
        # without treating arbitrary NUL-heavy binary data as Unicode text.
        even_low = sum(byte <= 0x20 for byte in even_bytes) / len(even_bytes)
        odd_low = sum(byte <= 0x20 for byte in odd_bytes) / len(odd_bytes)
        if odd_low >= 0.60 and even_low < 0.60:
            bomless_unicode.append("utf-16-le")
        if even_low >= 0.60 and odd_low < 0.60:
            bomless_unicode.append("utf-16-be")
    if nul_ratio > 0.15 and not has_utf16_bom and not bomless_unicode:
        raise IngestExtractionError("source appears to be binary (too many NUL bytes)")

    if raw.startswith(b"\xef\xbb\xbf"):
        candidates = ["utf-8-sig"]
    elif raw.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        candidates = ["utf-32"]
    elif has_utf16_bom:
        candidates = ["utf-16"]
    else:
        candidates = ["utf-8"] + bomless_unicode + ["cp1255", "cp1252", "latin-1"]

    decoded: List[Tuple[float, str, str]] = []
    for encoding in candidates:
        try:
            value = raw.decode(encoding, errors="strict")
        except (LookupError, UnicodeDecodeError):
            continue
        score = _decoded_text_score(value)
        # When UTF-8 is valid, it is unambiguous and should always win.
        if encoding in {
            "utf-8", "utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "utf-32"
        }:
            score += 1000.0
        # cp1255 is the common source of otherwise-lost Hebrew corpora. Reward
        # it only when it actually produces a substantive Hebrew signal.
        if encoding == "cp1255":
            hebrew_letters = len(re.findall(r"[\u05d0-\u05ea]", value))
            if hebrew_letters >= 4:
                score += min(100.0, hebrew_letters / max(1, len(value)) * 200.0)
        decoded.append((score, encoding, value))

    if not decoded:
        raise IngestExtractionError("source text encoding could not be decoded safely")
    score, encoding, best = max(decoded, key=lambda item: item[0])
    if not math.isfinite(score) or score < 50.0:
        raise IngestExtractionError(
            f"decoded source quality is too low (best encoding: {encoding})"
        )
    return _normalize_extracted_text(best)


def _file_snapshot(path: Path) -> Tuple[int, int]:
    stat = path.stat()
    return int(stat.st_size), int(stat.st_mtime_ns)


def _source_fingerprint(path: Path) -> Tuple[str, Tuple[int, int]]:
    """Hash source bytes and return the exact size/mtime snapshot hashed."""
    before = _file_snapshot(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = _file_snapshot(path)
    if before != after:
        raise IngestExtractionError(f"source changed while hashing: {path}")
    return digest.hexdigest(), after


def _text_language(value: str) -> str:
    """Return a compact, retrieval-friendly language hint for chunk metadata."""
    # Do not let English extractor scaffolding or an ASCII filename relabel a
    # Hebrew fact. Actual headings, spreadsheet headers, and document content
    # remain in the probe, so genuinely bilingual chunks are still "mixed".
    probe = value or ""
    probe = re.sub(r"(?m)^\[[^\n]{0,1024}\]\s*$", "", probe)
    probe = re.sub(
        r"(?im)^#{1,6}\s+(?:page\s+\d+|document body|header\s*\d+|"
        r"footer\s*\d+|footnotes|endnotes|postscript metadata|postscript text|"
        r"rows?\s+\d+(?:-\d+)?|delimited data|log events|records|"
        r"script (?:body|preamble))\s*$",
        "",
        probe,
    )
    probe = re.sub(
        r"(?im)^(?:spreadsheet(?: context)? row|table row|delimited row|log event)"
        r"\s+\d+:\s*",
        "",
        probe,
    )
    probe = re.sub(r"(?im)^\s*-?\s*column\s+[A-Z]{1,3}:\s*", "", probe)
    hebrew = len(re.findall(r"[\u05d0-\u05ea]", probe))
    latin = len(re.findall(r"[A-Za-z]", probe))
    if hebrew and latin:
        minority = min(hebrew, latin) / max(hebrew, latin)
        return "mixed" if minority >= 0.08 else ("he" if hebrew > latin else "en")
    if hebrew:
        return "he"
    if latin:
        return "en"
    return "unknown"

DEFAULT_INCLUDE = (
    ".md",
    ".mdx",
    ".txt",
    ".pdf",
    ".docx",
    ".xlsx",
    ".pptx",
    ".html",
    ".htm",
    ".ps",
    ".ps1",
    ".svg",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".sh",
    ".log",
    ".yaml",
    ".yml",
    ".go",
    ".mod",
    ".work",
    ".sum",
    ".json",
    ".csv",
    ".tsv",
)

_MEDIA_SUFFIXES = {".pdf", ".svg", ".png", ".jpg", ".jpeg", ".webp"}
_FAIL_CLOSED_EXTRACT_SUFFIXES = _MEDIA_SUFFIXES | {".docx", ".xlsx", ".pptx", ".ps"}


# ---------- Helpers ----------
def _is_rtl_text(s: str) -> bool:
    import unicodedata
    rtl = 0
    total = 0
    for ch in s:
        if ch.isspace():
            continue
        bidi = unicodedata.bidirectional(ch)
        if bidi in ("R", "AL", "AN"):
            rtl += 1
        total += 1
    return total > 0 and (rtl / total) >= 0.30


def _text_quality_score(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return float("-inf")
    tokens = s.split()
    wc = len(tokens)
    if wc == 0:
        return float("-inf")

    avg_len = sum(len(t) for t in tokens) / wc
    long_alpha = sum(1 for t in tokens if t.isalpha() and len(t) >= 20)
    space_ratio = s.count(" ") / max(1, len(s))
    nl_ratio = s.count("\n") / max(1, len(s))

    return (wc * 1.0) + (space_ratio * 50.0) + (nl_ratio * 20.0) - (avg_len * 0.5) - (long_alpha * 3.0)


def _pdf_text_needs_ocr(text: str) -> bool:
    """Conservatively identify empty or visibly broken PDF text extraction."""
    text = (text or "").strip()
    if not text:
        return True

    visible_chars = sum(1 for ch in text if ch.isalnum())
    words = re.findall(r"[^\W_]+|\d+", text, flags=re.UNICODE)
    if visible_chars < _env_int("PDF_OCR_MIN_CHARS", 80):
        return True
    if len(words) < _env_int("PDF_OCR_MIN_WORDS", 10):
        return True

    nonspace_chars = sum(1 for ch in text if not ch.isspace())
    if visible_chars / max(1, nonspace_chars) < 0.35:
        return True
    alpha_words = [word for word in words if word.isalpha()]
    if len(alpha_words) >= 10:
        isolated_ratio = sum(len(word) == 1 for word in alpha_words) / len(alpha_words)
        if isolated_ratio >= 0.55:
            return True

    broken_chars = sum(
        1
        for ch in text
        if ch == "\ufffd" or (ord(ch) < 32 and ch not in "\n\r\t")
    )
    if broken_chars / max(1, len(text)) > 0.02:
        return True

    return _text_quality_score(text) < _env_float("PDF_OCR_MIN_QUALITY", 12.0)


def _prefer_ocr_text(extracted_text: str, ocr_text: str) -> str:
    """Use OCR only when it provides better substantive text."""
    extracted = _normalize_extracted_text(extracted_text)
    ocr = _normalize_extracted_text(ocr_text)
    # Two-character Hebrew facts such as "כן"/"לא" are substantive, while a
    # single OCR glyph is too weak to replace another extraction.
    if sum(1 for ch in ocr if ch.isalnum()) < 2:
        return extracted
    if not extracted:
        return ocr
    if _text_quality_score(ocr) > _text_quality_score(extracted):
        return ocr
    return extracted


def _ocr_image_to_string(image: Any, language: str) -> str:
    """Run Tesseract with a hard timeout when the installed wrapper supports it."""
    timeout = _env_int("OCR_TIMEOUT_SECONDS", 60, minimum=0, maximum=3600)
    kwargs: Dict[str, Any] = {"lang": language}
    if timeout:
        kwargs["timeout"] = timeout
    try:
        return pytesseract.image_to_string(image, **kwargs) or ""
    except TypeError:
        # Older pytesseract releases do not expose ``timeout``. Keep backward
        # compatibility, while modern deployments remain bounded by default.
        kwargs.pop("timeout", None)
        return pytesseract.image_to_string(image, **kwargs) or ""


def _safe_pdf_ocr_dpi(page: Any, requested_dpi: int) -> int:
    """Lower render DPI before allocation when a page would exceed pixel limits."""
    max_pixels = _env_int(
        "OCR_MAX_IMAGE_PIXELS",
        40_000_000,
        minimum=1_000_000,
        maximum=200_000_000,
    )
    rect = getattr(page, "rect", None)
    width = float(getattr(rect, "width", 0.0) or 0.0)
    height = float(getattr(rect, "height", 0.0) or 0.0)
    if width <= 0 or height <= 0:
        return requested_dpi
    estimated = width * height * (requested_dpi / 72.0) ** 2
    if estimated <= max_pixels:
        return requested_dpi
    safe = int(72.0 * math.sqrt(max_pixels / (width * height)))
    return max(72, min(requested_dpi, safe))


def _image_dimensions(image: Any) -> Tuple[int, int]:
    size = getattr(image, "size", None)
    if isinstance(size, (tuple, list)) and len(size) == 2:
        try:
            return max(0, int(size[0])), max(0, int(size[1]))
        except (TypeError, ValueError):
            pass
    return 0, 0


def _page_section(page_number: int, text: str) -> str:
    """Keep page identity beside extracted text for chunking and citations."""
    normalized = _normalize_extracted_text(text)
    if not normalized:
        return ""
    return f"# Page {page_number}\n\n{normalized}"


def _read_pdf_text_pymupdf_words(path: Path) -> str:
    if fitz is None:
        return ""

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise IngestExtractionError(
            f"PyMuPDF failed to open PDF {path}: {e}"
        ) from e

    pages_out: List[str] = []
    raw_pages_out: List[str] = []
    document_origins: List[Optional[int]] = []
    document_edits: List[CanonicalEdit] = []
    raw_cursor = 0
    canonical_cursor = 0
    reconstruction_reason_counts: Dict[str, int] = {}
    line_fragment_merge_count = 0
    page_errors: List[str] = []
    try:
        for page_index, page in enumerate(doc):
            try:
                words = page.get_text("words") or []
                style_dict = page.get_text("dict") or {}
            except Exception as e:
                page_errors.append(f"page {page_index + 1}: {e}")
                continue

            if not words:
                continue
            positioned = positioned_tokens_from_pymupdf(
                page_index + 1,
                words,
                pymupdf_line_styles(style_dict),
            )
            reconstructed = reconstruct_page(positioned)
            if not reconstructed.text:
                continue
            line_fragment_merge_count += reconstructed.line_fragment_merge_count
            prefix = f"# Page {page_index + 1}\n\n"
            raw_section = prefix + reconstructed.raw_text
            canonical_section = prefix + reconstructed.text
            if pages_out:
                raw_pages_out.append("\n\n")
                pages_out.append("\n\n")
                document_origins.extend((raw_cursor, raw_cursor + 1))
                raw_cursor += 2
                canonical_cursor += 2
            raw_pages_out.append(raw_section)
            pages_out.append(canonical_section)
            document_origins.extend(
                range(raw_cursor, raw_cursor + len(prefix))
            )
            document_origins.extend(
                raw_cursor + len(prefix) + int(origin)
                if isinstance(origin, int)
                else None
                for origin in reconstructed.origin_offsets
            )
            for edit in reconstructed.edits:
                reconstruction_reason_counts[edit.reason_code] = (
                    reconstruction_reason_counts.get(edit.reason_code, 0) + 1
                )
                document_edits.append(CanonicalEdit(
                    raw_start=raw_cursor + len(prefix) + edit.raw_start,
                    raw_end=raw_cursor + len(prefix) + edit.raw_end,
                    canonical_start=(
                        canonical_cursor + len(prefix) + edit.canonical_start
                    ),
                    canonical_end=(
                        canonical_cursor + len(prefix) + edit.canonical_end
                    ),
                    reason_code=edit.reason_code,
                ))
            raw_cursor += len(raw_section)
            canonical_cursor += len(canonical_section)
    finally:
        try:
            doc.close()
        except Exception:
            pass

    if page_errors:
        raise IngestExtractionError(
            f"PyMuPDF produced a partial PDF extraction for {path}: "
            + "; ".join(page_errors[:3])
        )

    raw_document = "".join(raw_pages_out)
    canonical_document = "".join(pages_out)
    normalized = _normalize_extracted_text(canonical_document)
    if normalized != canonical_document or len(document_origins) != len(canonical_document):
        # A producer-specific control character required an additional generic
        # normalization pass.  Keep serving the safe text but do not publish a
        # misaligned origin map; normal logical canonicalization will retain
        # its own complete audit instead.
        _PDF_RECONSTRUCTION_TRACE_CACHE.pop(str(path.resolve()), None)
        return normalized
    _PDF_RECONSTRUCTION_TRACE_CACHE[str(path.resolve())] = (
        raw_document,
        canonical_document,
        tuple(document_origins),
        tuple(document_edits),
        {
            "pdf_reconstruction_revision": PDF_RECONSTRUCTION_REVISION,
            "pdf_reconstruction_edit_count": len(document_edits),
            "pdf_intra_word_repair_count": reconstruction_reason_counts.get(
                "pdf_conclusive_intra_word_boundary",
                0,
            ),
            "pdf_soft_wrap_repair_count": reconstruction_reason_counts.get(
                "pdf_soft_visual_line_wrap",
                0,
            ),
            "pdf_line_fragment_merge_count": line_fragment_merge_count,
        },
    )
    return canonical_document


def _read_pdf_text_pypdf(path: Path, mode: str = "layout") -> str:
    if pypdf is None:
        return ""
    try:
        reader = pypdf.PdfReader(str(path))
    except Exception as e:
        raise IngestExtractionError(
            f"pypdf failed to open PDF {path}: {e}"
        ) from e

    pages_text: List[str] = []
    page_errors: List[str] = []
    for page_index, page in enumerate(reader.pages):
        try:
            if mode == "layout":
                try:
                    txt = page.extract_text(extraction_mode="layout") or ""
                except TypeError:
                    txt = page.extract_text() or ""
            else:
                txt = page.extract_text() or ""
        except Exception as e:
            page_errors.append(f"page {page_index + 1}: {e}")
            continue
        section = _page_section(page_index + 1, txt)
        if section:
            pages_text.append(section)

    if page_errors:
        raise IngestExtractionError(
            f"pypdf produced a partial PDF extraction for {path}: "
            + "; ".join(page_errors[:3])
        )

    return _normalize_extracted_text("\n\n".join(pages_text))


def _pdf_page_bodies(value: str) -> Dict[int, str]:
    """Parse the page markers emitted by every built-in PDF extractor."""
    text = _normalize_extracted_text(value)
    matches = list(re.finditer(r"(?m)^# Page\s+(\d+)\s*$", text))
    pages: Dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = text[match.end():end].strip()
        page_number = int(match.group(1))
        if page_number > 0 and body:
            pages[page_number] = body
    return pages


def _pdf_page_body_spans(value: str) -> Dict[int, Tuple[int, int, str]]:
    """Return page-body offsets without losing their document origin."""
    text = str(value or "")
    matches = list(re.finditer(r"(?m)^# Page\s+(\d+)\s*$", text))
    spans: Dict[int, Tuple[int, int, str]] = {}
    for index, match in enumerate(matches):
        segment_end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(text)
        )
        start = match.end()
        while start < segment_end and text[start].isspace():
            start += 1
        end = segment_end
        while end > start and text[end - 1].isspace():
            end -= 1
        page_number = int(match.group(1))
        if page_number > 0 and end > start:
            spans[page_number] = (start, end, text[start:end])
    return spans


def _install_hybrid_pdf_reconstruction_trace(
    path: Path,
    final_text: str,
    coordinate_trace: Optional[
        Tuple[
            str,
            str,
            Tuple[Optional[int], ...],
            Tuple[CanonicalEdit, ...],
            Dict[str, Any],
        ]
    ],
) -> str:
    """Retain coordinate origins for selected pages in a mixed PDF extraction.

    Auto extraction may combine layout text, coordinate-reconstructed digital
    pages, and OCR-only pages.  The fallback pages cannot truthfully claim
    glyph-coordinate provenance, but they must not erase the coordinate map
    for pages that were selected byte-for-byte from the reconstruction.
    """
    normalized = _normalize_extracted_text(final_text)
    if not coordinate_trace:
        return normalized
    (
        coordinate_raw,
        coordinate_text,
        coordinate_origins,
        coordinate_edits,
        coordinate_metadata,
    ) = coordinate_trace
    if len(coordinate_origins) != len(coordinate_text):
        return normalized
    if normalized == coordinate_text:
        _PDF_RECONSTRUCTION_TRACE_CACHE[str(path.resolve())] = coordinate_trace
        return normalized

    final_spans = _pdf_page_body_spans(normalized)
    coordinate_spans = _pdf_page_body_spans(coordinate_text)
    separator = "\n\n"
    fallback_raw_start = len(coordinate_raw) + len(separator)
    origins: List[Optional[int]] = [
        fallback_raw_start + index
        for index in range(len(normalized))
    ]
    coordinate_pages: List[int] = []
    for page_number, (final_start, final_end, final_body) in final_spans.items():
        coordinate_span = coordinate_spans.get(page_number)
        if coordinate_span is None:
            continue
        coordinate_start, coordinate_end, coordinate_body = coordinate_span
        if (
            final_body != coordinate_body
            or final_end - final_start != coordinate_end - coordinate_start
        ):
            continue
        origins[final_start:final_end] = coordinate_origins[
            coordinate_start:coordinate_end
        ]
        coordinate_pages.append(page_number)

    if not coordinate_pages:
        return normalized
    used_coordinate_origins = sorted({
        int(origin)
        for origin in origins
        if isinstance(origin, int) and 0 <= origin < len(coordinate_raw)
    })
    relevant_edits: List[CanonicalEdit] = []
    for edit in coordinate_edits:
        position = bisect_right(used_coordinate_origins, edit.raw_start - 1)
        if (
            position < len(used_coordinate_origins)
            and used_coordinate_origins[position] < edit.raw_end
        ):
            relevant_edits.append(edit)

    metadata = dict(coordinate_metadata)
    metadata.update({
        "pdf_hybrid_provenance_revision": "coordinate-fallback-v1",
        "pdf_reconstruction_edit_count": len(relevant_edits),
        "pdf_intra_word_repair_count": sum(
            edit.reason_code == "pdf_conclusive_intra_word_boundary"
            for edit in relevant_edits
        ),
        "pdf_soft_wrap_repair_count": sum(
            edit.reason_code == "pdf_soft_visual_line_wrap"
            for edit in relevant_edits
        ),
        "pdf_coordinate_provenance_page_count": len(coordinate_pages),
        "pdf_fallback_provenance_page_count": max(
            0,
            len(final_spans) - len(coordinate_pages),
        ),
    })
    _PDF_RECONSTRUCTION_TRACE_CACHE[str(path.resolve())] = (
        coordinate_raw + separator + normalized,
        normalized,
        tuple(origins),
        tuple(relevant_edits),
        metadata,
    )
    return normalized


def _pdf_total_pages(path: Path, extracted_pages: Dict[int, str]) -> int:
    """Best-effort page count used only to find selectively scanned pages."""
    fallback = max(extracted_pages, default=0)
    if pypdf is not None:
        try:
            return max(fallback, len(pypdf.PdfReader(str(path)).pages))
        except Exception:
            pass
    if fitz is not None:
        try:
            document = fitz.open(str(path))
            try:
                return max(fallback, len(document))
            finally:
                document.close()
        except Exception:
            pass
    return fallback


def _merge_pdf_pages(
    extracted: str,
    ocr_text: str,
    total_pages: int,
) -> str:
    extracted_pages = _pdf_page_bodies(extracted)
    ocr_pages = _pdf_page_bodies(ocr_text)
    if not extracted_pages and not ocr_pages:
        return _prefer_ocr_text(extracted, ocr_text)
    last_page = max(total_pages, max(extracted_pages, default=0), max(ocr_pages, default=0))
    sections: List[str] = []
    for page_number in range(1, last_page + 1):
        selected = _prefer_ocr_text(
            extracted_pages.get(page_number, ""),
            ocr_pages.get(page_number, ""),
        )
        section = _page_section(page_number, selected)
        if section:
            sections.append(section)
    return _normalize_extracted_text("\n\n".join(sections))


def _pdf_hebrew_layout_artifacts(value: str) -> bool:
    """Detect layout extraction that expanded normal Hebrew text with spaces."""
    text = str(value or "")
    hebrew_tokens = re.findall(r"[\u0590-\u05ff]+", text)
    hebrew_letters = sum(len(token) for token in hebrew_tokens)
    if hebrew_letters < 40 or not hebrew_tokens:
        return False
    spaces = len(re.findall(r"[ \t]", text))
    space_ratio = spaces / max(1, spaces + hebrew_letters)
    singleton_ratio = sum(len(token) == 1 for token in hebrew_tokens) / len(
        hebrew_tokens
    )
    return space_ratio >= 0.38 or singleton_ratio >= 0.18


def _merge_pdf_layout_pages(
    primary: str,
    coordinate: str,
    total_pages: int,
) -> str:
    """Prefer coordinate text only when it conclusively repairs layout text."""
    primary_pages = _pdf_page_bodies(primary)
    coordinate_pages = _pdf_page_bodies(coordinate)
    last_page = max(
        total_pages,
        max(primary_pages, default=0),
        max(coordinate_pages, default=0),
    )
    sections: List[str] = []
    for page_number in range(1, last_page + 1):
        primary_body = primary_pages.get(page_number, "")
        coordinate_body = coordinate_pages.get(page_number, "")
        coordinate_repairs_layout = bool(
            coordinate_body
            and not _pdf_text_needs_ocr(coordinate_body)
            and _pdf_hebrew_layout_artifacts(primary_body)
            and not _pdf_hebrew_layout_artifacts(coordinate_body)
        )
        selected = (
            coordinate_body
            if coordinate_repairs_layout
            else _prefer_ocr_text(primary_body, coordinate_body)
        )
        section = _page_section(page_number, selected)
        if section:
            sections.append(section)
    return _normalize_extracted_text("\n\n".join(sections))


def _native_pdf_pages_with_coordinate_loss(primary: str, coordinate: str) -> set[int]:
    """Locate layout pages where coordinate extraction has lost source text.

    Comparing alphanumeric character counts is independent of Hebrew visual
    order. The conservative threshold avoids replacing minor layout repairs.
    """
    primary_pages = _pdf_page_bodies(primary)
    coordinate_pages = _pdf_page_bodies(coordinate)
    result = set()
    for number, body in primary_pages.items():
        other = coordinate_pages.get(number, "")
        count = sum(ch.isalnum() for ch in body)
        other_count = sum(ch.isalnum() for ch in other)
        if (count >= 100 and _pdf_hebrew_layout_artifacts(body)
                and other_count < count * 0.7):
            result.add(number)
    return result


def _restore_native_pdf_pages(extracted: str, primary: str, plain: str,
                              pages: set[int]) -> str:
    """Restore a page only when plain extraction conserves its layout text."""
    selected = _pdf_page_bodies(extracted)
    layout_pages = _pdf_page_bodies(primary)
    plain_pages = _pdf_page_bodies(plain)
    for number in pages:
        layout = layout_pages.get(number, "")
        body = plain_pages.get(number, "")
        from collections import Counter
        layout_chars = Counter(ch for ch in layout if ch.isalnum())
        plain_chars = Counter(ch for ch in body if ch.isalnum())
        layout_count = sum(layout_chars.values())
        common_count = sum((layout_chars & plain_chars).values())
        if not (layout_count and common_count >= layout_count * 0.9
                and not _pdf_text_needs_ocr(body)
                and not _pdf_hebrew_layout_artifacts(body)):
            raise IngestExtractionError(
                f"native PDF page {number} loses text in coordinate extraction "
                "and has no complete plain-text fallback"
            )
        selected[number] = body
    return _normalize_extracted_text("\n\n".join(
        _page_section(number, selected[number]) for number in sorted(selected)
    ))


def _read_pdf_text_ocr(
    path: Path,
    page_numbers: Optional[set[int]] = None,
) -> str:
    """OCR a scanned PDF with optional PyMuPDF, Pillow, and pytesseract.

    Missing optional packages return an empty fallback so a complete digital
    extraction can still be used. Once OCR begins, any page failure or page cap
    fails loudly: partial OCR must never replace a complete older generation.
    """
    if fitz is None or Image is None or pytesseract is None:
        print(
            f"[INGEST] WARNING: PDF OCR unavailable for {path}; "
            "install PyMuPDF, Pillow, pytesseract, Tesseract, and its language data.",
            file=sys.stderr,
        )
        return ""

    try:
        doc = fitz.open(str(path))
    except Exception as e:
        raise IngestExtractionError(
            f"failed to open PDF for OCR {path}: {e}"
        ) from e

    dpi = _env_int("PDF_OCR_DPI", 220, minimum=72, maximum=600)
    max_pages = _env_int("PDF_OCR_MAX_PAGES", 100)
    language = _ocr_language()
    pages_text: List[str] = []
    selected_pages = (
        {int(page) for page in page_numbers if int(page) > 0}
        if page_numbers is not None
        else None
    )
    pages_processed = 0

    try:
        for page_index, page in enumerate(doc):
            page_number = page_index + 1
            if selected_pages is not None and page_number not in selected_pages:
                continue
            if max_pages and pages_processed >= max_pages:
                raise IngestExtractionError(
                    f"PDF OCR would be partial for {path}: extraction requires more than "
                    f"PDF_OCR_MAX_PAGES={max_pages} pages; raise the limit or set it to 0"
                )
            pages_processed += 1

            try:
                page_dpi = _safe_pdf_ocr_dpi(page, dpi)
                try:
                    pixmap = page.get_pixmap(dpi=page_dpi, alpha=False)
                except TypeError:
                    scale = page_dpi / 72.0
                    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                image_bytes = pixmap.tobytes("png")
                with Image.open(io.BytesIO(image_bytes)) as image:
                    width, height = _image_dimensions(image)
                    max_pixels = _env_int(
                        "OCR_MAX_IMAGE_PIXELS",
                        40_000_000,
                        minimum=1_000_000,
                        maximum=200_000_000,
                    )
                    if width and height and width * height > max_pixels:
                        raise IngestExtractionError(
                            f"rendered OCR page exceeds {max_pixels} pixels"
                        )
                    page_text = _ocr_image_to_string(image, language)
            except Exception as e:
                raise IngestExtractionError(
                    f"PDF OCR failed on page {page_number} of {path} "
                    f"using language {language!r}: {e}"
                ) from e

            section = _page_section(page_number, page_text)
            if section:
                pages_text.append(section)
    finally:
        try:
            doc.close()
        except Exception:
            pass

    return _normalize_extracted_text("\n\n".join(pages_text))

def _read_pdf_text(path: Union[str, Path], *, native_only: bool = False) -> str:
    path = Path(path)
    if native_only:
        # Sparse technical labels are valid native evidence even when a prose
        # quality heuristic requests OCR. Explicit opt-in keeps the existing
        # coordinate reconstruction and page provenance without rasterization.
        _PDF_RECONSTRUCTION_TRACE_CACHE.pop(str(path.resolve()), None)
        extracted = _read_pdf_text_pymupdf_words(path)
        if not extracted.strip():
            extracted = _read_pdf_text_pypdf(path, mode="plain")
        if not extracted.strip():
            raise IngestNoTextError(f"native-only PDF extraction found no text: {path}")
        return extracted
    extractor = os.getenv("PDF_EXTRACTOR", "auto").strip().lower()
    extraction_errors: List[IngestExtractionError] = []
    coordinate_trace = None

    # Hebrew-friendly extractor. In auto mode, use the fast layout extractor
    # first and consult PyMuPDF's word coordinates only when the first result
    # looks broken. This materially improves Hebrew RTL PDFs without doubling
    # work for healthy digital documents.
    if extractor in {"pymupdf_words", "mupdf_words", "fitz_words"}:
        extracted = _read_pdf_text_pymupdf_words(path)
        coordinate_trace = _PDF_RECONSTRUCTION_TRACE_CACHE.get(str(path.resolve()))
    else:
        if extractor == "auto":
            _PDF_RECONSTRUCTION_TRACE_CACHE.pop(str(path.resolve()), None)
            try:
                extracted = _read_pdf_text_pypdf(path, mode="layout")
            except IngestExtractionError as e:
                extraction_errors.append(e)
                extracted = ""
            extracted_pages = _pdf_page_bodies(extracted)
            total_pages = (
                _pdf_total_pages(path, extracted_pages) if extracted_pages else 0
            )
            weak_pages = {
                page_number
                for page_number in range(1, total_pages + 1)
                if _pdf_text_needs_ocr(extracted_pages.get(page_number, ""))
            }
            layout_artifact_pages = {
                page_number
                for page_number, body in extracted_pages.items()
                if _pdf_hebrew_layout_artifacts(body)
            }
            if _pdf_text_needs_ocr(extracted) or weak_pages or layout_artifact_pages:
                try:
                    alternate = _read_pdf_text_pymupdf_words(path)
                except IngestExtractionError as e:
                    extraction_errors.append(e)
                    alternate = ""
                coordinate_trace = _PDF_RECONSTRUCTION_TRACE_CACHE.get(
                    str(path.resolve())
                )
                alternate_pages = _pdf_page_bodies(alternate)
                if extracted_pages and alternate_pages:
                    primary = extracted
                    lossy_pages = _native_pdf_pages_with_coordinate_loss(primary, alternate)
                    extracted = _merge_pdf_layout_pages(
                        primary,
                        alternate,
                        max(total_pages, max(alternate_pages, default=0)),
                    )
                    if lossy_pages:
                        try:
                            plain = _read_pdf_text_pypdf(path, mode="plain")
                        except IngestExtractionError as e:
                            extraction_errors.append(e)
                            plain = ""
                        extracted = _restore_native_pdf_pages(
                            extracted, primary, plain, lossy_pages,
                        )
                    _PDF_RECONSTRUCTION_TRACE_CACHE.pop(str(path.resolve()), None)
                else:
                    extracted = _prefer_ocr_text(extracted, alternate)
                    if extracted != alternate:
                        _PDF_RECONSTRUCTION_TRACE_CACHE.pop(str(path.resolve()), None)
            extractor = "auto_selected"

        if extractor == "auto_selected":
            pass
        elif extractor == "pypdf_layout":
            extracted = _read_pdf_text_pypdf(path, mode="layout")
        elif extractor == "pypdf_plain":
            extracted = _read_pdf_text_pypdf(path, mode="plain")
        elif extractor == "pypdf":
            # optional: allow "pypdf" as alias
            extracted = _read_pdf_text_pypdf(path, mode="layout")
        else:
            raise ValueError(f"Unknown PDF_EXTRACTOR={extractor}")

    if _env_flag("PDF_OCR_FALLBACK", True):
        extracted_pages = _pdf_page_bodies(extracted)
        if extracted_pages:
            total_pages = _pdf_total_pages(path, extracted_pages)
            pages_needing_ocr = {
                page_number
                for page_number in range(1, total_pages + 1)
                if _pdf_text_needs_ocr(extracted_pages.get(page_number, ""))
            }
            if pages_needing_ocr:
                ocr_text = _read_pdf_text_ocr(path, pages_needing_ocr)
                merged = _merge_pdf_pages(extracted, ocr_text, total_pages)
                if ocr_text.strip() and merged != _normalize_extracted_text(extracted):
                    _PDF_RECONSTRUCTION_TRACE_CACHE.pop(str(path.resolve()), None)
                    print(
                        f"[INGEST] PDF OCR improved {len(pages_needing_ocr)} page(s) "
                        f"for {path} (language={_ocr_language()})."
                    )
                if merged:
                    return _install_hybrid_pdf_reconstruction_trace(
                        path,
                        merged,
                        coordinate_trace,
                    )
            normalized = _normalize_extracted_text(extracted)
            if normalized:
                return _install_hybrid_pdf_reconstruction_trace(
                    path,
                    normalized,
                    coordinate_trace,
                )

    if _env_flag("PDF_OCR_FALLBACK", True) and _pdf_text_needs_ocr(extracted):
        ocr_text = _read_pdf_text_ocr(path)
        selected = _prefer_ocr_text(extracted, ocr_text)
        if ocr_text.strip() and selected == ocr_text.strip():
            _PDF_RECONSTRUCTION_TRACE_CACHE.pop(str(path.resolve()), None)
            print(
                f"[INGEST] PDF OCR fallback selected for {path} "
                f"(language={_ocr_language()})."
            )
        selected = _normalize_extracted_text(selected)
        if selected:
            return _install_hybrid_pdf_reconstruction_trace(
                path,
                selected,
                coordinate_trace,
            )
    normalized = _normalize_extracted_text(extracted)
    if not normalized and extraction_errors:
        raise IngestExtractionError(
            f"all PDF extraction methods failed for {path}: "
            + " | ".join(str(error) for error in extraction_errors)
        )
    return _install_hybrid_pdf_reconstruction_trace(
        path,
        normalized,
        coordinate_trace,
    )


_WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_OFFICE_REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
_PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"


def _validated_zip(path: Path, expected_format: str) -> zipfile.ZipFile:
    """Open an Office archive after bounded zip-bomb and encryption checks."""
    max_file_bytes = _env_int(
        "INGEST_MAX_STRUCTURED_FILE_BYTES",
        512 * 1024 * 1024,
        minimum=1,
        maximum=4 * 1024 * 1024 * 1024,
    )
    try:
        if path.stat().st_size > max_file_bytes:
            raise IngestExtractionError(
                f"{expected_format} exceeds INGEST_MAX_STRUCTURED_FILE_BYTES "
                f"({max_file_bytes}): {path}"
            )
        archive = zipfile.ZipFile(path)
    except IngestExtractionError:
        raise
    except (OSError, zipfile.BadZipFile) as error:
        raise IngestExtractionError(
            f"failed to open {expected_format} archive {path}: {error}"
        ) from error

    try:
        entries = archive.infolist()
        max_entries = _env_int(
            "INGEST_MAX_ARCHIVE_ENTRIES", 10000, minimum=1, maximum=100000
        )
        if len(entries) > max_entries:
            raise IngestExtractionError(
                f"{expected_format} archive exceeds INGEST_MAX_ARCHIVE_ENTRIES="
                f"{max_entries}: {path}"
            )
        if any(info.flag_bits & 0x1 for info in entries):
            raise IngestExtractionError(
                f"encrypted {expected_format} archives are not supported: {path}"
            )
        total_uncompressed = sum(max(0, int(info.file_size)) for info in entries)
        max_uncompressed = _env_int(
            "INGEST_MAX_ARCHIVE_UNCOMPRESSED_BYTES",
            512 * 1024 * 1024,
            minimum=1,
            maximum=4 * 1024 * 1024 * 1024,
        )
        if total_uncompressed > max_uncompressed:
            raise IngestExtractionError(
                f"{expected_format} archive exceeds "
                f"INGEST_MAX_ARCHIVE_UNCOMPRESSED_BYTES={max_uncompressed}: {path}"
            )
        return archive
    except Exception:
        archive.close()
        raise


def _read_zip_part(
    archive: zipfile.ZipFile,
    name: str,
    *,
    required: bool = False,
) -> bytes:
    try:
        info = archive.getinfo(name)
    except KeyError:
        if required:
            raise IngestExtractionError(f"required Office XML part is missing: {name}")
        return b""
    max_part = _env_int(
        "INGEST_MAX_XML_PART_BYTES",
        64 * 1024 * 1024,
        minimum=1,
        maximum=512 * 1024 * 1024,
    )
    if info.file_size > max_part:
        raise IngestExtractionError(
            f"Office XML part exceeds INGEST_MAX_XML_PART_BYTES={max_part}: {name}"
        )
    try:
        with archive.open(info, "r") as handle:
            payload = handle.read(max_part + 1)
    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
        raise IngestExtractionError(f"failed to read Office XML part {name}: {error}") from error
    if len(payload) > max_part:
        raise IngestExtractionError(
            f"Office XML part exceeds INGEST_MAX_XML_PART_BYTES={max_part}: {name}"
        )
    return payload


def _parse_xml_part(payload: bytes, name: str) -> ET.Element:
    if not payload:
        raise IngestExtractionError(f"Office XML part is empty: {name}")
    if re.search(br"<!\s*(?:DOCTYPE|ENTITY)\b", payload, flags=re.IGNORECASE):
        raise IngestExtractionError(f"unsafe XML declaration in Office part: {name}")
    try:
        return ET.fromstring(payload)
    except ET.ParseError as error:
        raise IngestExtractionError(f"malformed Office XML part {name}: {error}") from error


def _markdown_cell(value: str) -> str:
    compact = re.sub(r"\s+", " ", _normalize_extracted_text(value)).strip()
    return compact.replace("|", "\\|")


def _markdown_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    header = [_markdown_cell(cell) for cell in normalized[0]]
    body = [[_markdown_cell(cell) for cell in row] for row in normalized[1:]]
    rendered = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    rendered.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(rendered)


def _structured_table_records(
    rows: List[List[str]], *, force_header: bool = False
) -> str:
    """Render table rows as self-contained facts for reliable retrieval.

    Markdown tables are readable, but once a large table is chunked a data row
    can become separated from its header. Repeating the column label beside
    every value preserves the meaning of each row in both Hebrew and English.
    """
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    first = [_markdown_cell(cell) for cell in normalized[0]]
    nonempty = [cell for cell in first if cell]
    unique = len(set(nonempty)) == len(nonempty)
    textual = sum(
        bool(re.search(r"[^\W\d_]", cell, flags=re.UNICODE)) for cell in nonempty
    )
    has_header = force_header or (
        len(normalized) >= 2
        and len(nonempty) >= 2
        and unique
        and textual / len(nonempty) >= 0.6
    )
    labels = [
        (first[index] if has_header and first[index] else f"Column {_xlsx_column_name(index + 1)}")
        for index in range(width)
    ]
    data = normalized[1:] if has_header else normalized
    if has_header and not data:
        return "Table columns: " + "; ".join(labels)

    records: List[str] = []
    for row_index, row in enumerate(data, start=2 if has_header else 1):
        facts = [
            f"- {labels[index]}: {_markdown_cell(value)}"
            for index, value in enumerate(row)
            if _markdown_cell(value)
        ]
        if facts:
            records.append(f"Table row {row_index}:\n" + "\n".join(facts))
    return "\n\n".join(records)


def _docx_styles(archive: zipfile.ZipFile) -> Dict[str, int]:
    payload = _read_zip_part(archive, "word/styles.xml")
    if not payload:
        return {}
    root = _parse_xml_part(payload, "word/styles.xml")
    headings: Dict[str, int] = {}
    for style in root.findall(f".//{{{_WORD_NS}}}style"):
        if style.get(f"{{{_WORD_NS}}}type") not in {None, "paragraph"}:
            continue
        style_id = style.get(f"{{{_WORD_NS}}}styleId") or ""
        name_node = style.find(f"{{{_WORD_NS}}}name")
        name = name_node.get(f"{{{_WORD_NS}}}val") if name_node is not None else ""
        probe = f"{style_id} {name}".strip()
        match = re.search(r"(?:heading|כותרת)\s*([1-6])", probe, flags=re.IGNORECASE)
        if match:
            headings[style_id] = int(match.group(1))
        elif re.search(r"(?:^|\s)title(?:$|\s)", probe, flags=re.IGNORECASE):
            headings[style_id] = 1
    return headings


def _docx_paragraph_text(paragraph: ET.Element) -> str:
    parts: List[str] = []
    image_labels: List[str] = []
    def visit(node: ET.Element) -> None:
        local = node.tag.rsplit("}", 1)[-1]
        if local in {"del", "moveFrom"}:
            return
        if local == "t" and node.text:
            parts.append(node.text)
        elif local == "tab":
            parts.append("\t")
        elif local in {"br", "cr"}:
            parts.append("\n")
        elif local == "noBreakHyphen":
            parts.append("-")
        elif local == "docPr":
            label = (node.get("descr") or node.get("title") or "").strip()
            if label:
                image_labels.append(label)
        for child in list(node):
            visit(child)

    visit(paragraph)
    value = "".join(parts).strip()
    if image_labels:
        images = " ".join(f"[Image: {label}]" for label in image_labels)
        value = f"{value} {images}".strip()
    return _normalize_extracted_text(value)


def _docx_render_table(table: ET.Element) -> str:
    rows: List[List[str]] = []
    for row in table.findall(f"./{{{_WORD_NS}}}tr"):
        cells: List[str] = []
        for cell in row.findall(f"./{{{_WORD_NS}}}tc"):
            paragraphs = [
                _docx_paragraph_text(paragraph)
                for paragraph in cell.findall(f".//{{{_WORD_NS}}}p")
            ]
            cells.append(" / ".join(value for value in paragraphs if value))
        if any(cell for cell in cells):
            rows.append(cells)
    return _structured_table_records(rows)


def _docx_render_container(
    container: ET.Element,
    heading_styles: Dict[str, int],
) -> str:
    blocks: List[str] = []
    children = list(container)
    for child in children:
        local = child.tag.rsplit("}", 1)[-1]
        if local == "p":
            text = _docx_paragraph_text(child)
            if not text:
                continue
            style_node = child.find(f"./{{{_WORD_NS}}}pPr/{{{_WORD_NS}}}pStyle")
            style_id = (
                style_node.get(f"{{{_WORD_NS}}}val") if style_node is not None else ""
            ) or ""
            level = heading_styles.get(style_id)
            numbered = child.find(f"./{{{_WORD_NS}}}pPr/{{{_WORD_NS}}}numPr") is not None
            if level:
                blocks.append(f"{'#' * level} {text}")
            elif numbered:
                blocks.append(f"- {text}")
            else:
                blocks.append(text)
        elif local == "tbl":
            rendered = _docx_render_table(child)
            if rendered:
                blocks.append(rendered)
        elif local in {"sdt", "sdtContent", "customXml", "smartTag", "ins", "moveTo"}:
            # Modern Word templates frequently wrap complete sections in
            # structured-document tags (content controls). Recurse through
            # content-bearing wrappers in document order, while deliberately
            # excluding deletion/move-from revision containers.
            rendered = _docx_render_container(child, heading_styles)
            if rendered:
                blocks.append(rendered)
    return "\n\n".join(blocks)



_DOCX_IMAGE_OCR_REVISION = "empty-native-embedded-image-v1"


def _docx_embedded_image_parts(archive, sources):
    """Resolve only images actually referenced by the parsed DOCX sections."""
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise IngestExtractionError("DOCX has ambiguous duplicate archive members")
    images = []
    seen = set()
    for source_name, source_root in sources:
        references = []
        for node in source_root.iter():
            local = node.tag.rsplit("}", 1)[-1]
            if local == "blip":
                embedded = node.get(f"{{{_OFFICE_REL_NS}}}embed")
                linked = node.get(f"{{{_OFFICE_REL_NS}}}link")
                if linked and not embedded:
                    raise IngestExtractionError("linked DOCX images cannot be extracted offline")
                if embedded:
                    references.append(embedded)
            elif local == "imagedata":
                relationship = node.get(f"{{{_OFFICE_REL_NS}}}id")
                if relationship:
                    references.append(relationship)
        if not references:
            continue
        parent, leaf = posixpath.split(source_name)
        relations_name = posixpath.join(parent, "_rels", leaf + ".rels")
        relations = _parse_xml_part(
            _read_zip_part(archive, relations_name, required=True), relations_name
        )
        targets = {}
        for node in relations.findall(f"./{{{_PACKAGE_REL_NS}}}Relationship"):
            identity = node.get("Id") or ""
            if not identity or identity in targets:
                raise IngestExtractionError("ambiguous DOCX image relationship")
            targets[identity] = node
        for reference in references:
            relation = targets.get(reference)
            if relation is None or not (relation.get("Type") or "").endswith("/image"):
                raise IngestExtractionError("DOCX embedded image relationship is missing or invalid")
            if relation.get("TargetMode", "").lower() == "external":
                raise IngestExtractionError("external DOCX image targets are not supported")
            target = relation.get("Target") or ""
            if not target or chr(92) in target or chr(0) in target:
                raise IngestExtractionError("unsafe DOCX image target")
            normalized = posixpath.normpath(
                target.lstrip("/") if target.startswith("/") else posixpath.join(parent, target)
            )
            if not re.fullmatch(r"word/media/[^/]+", normalized):
                raise IngestExtractionError("unsafe DOCX image target")
            if normalized not in seen:
                seen.add(normalized)
                images.append(normalized)
    return images



def _emf_unicode_text(payload: bytes) -> str:
    """Read bounded Unicode EMF text records; never interpret glyph indices.

    Record layout follows Microsoft's MS-EMF EMR_EXTTEXTOUTW/EmrText format.
    Unknown text-output variants fail explicitly instead of omitting content.
    References define rows only for an untransformed, unrotated MM_TEXT file.
    https://learn.microsoft.com/en-us/openspecs/windows_protocols/ms-emf/a59a79ac-328e-492d-a34d-e02727af6edf
    """
    if len(payload) < 88:
        raise IngestExtractionError("EMF header is truncated")
    kind, header_size = struct.unpack_from("<II", payload)
    signature, version, declared_bytes, declared_records = struct.unpack_from("<IIII", payload, 40)
    if (kind != 1 or header_size < 88 or header_size % 4
            or header_size > len(payload) or signature != 0x464D4520
            or version != 0x10000 or declared_bytes != len(payload)):
        raise IngestExtractionError("invalid EMF header, signature, or byte count")
    bounds = struct.unpack_from("<iiii", payload, 8)
    if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
        raise IngestExtractionError("invalid EMF device bounds")
    maximum_records = _env_int("EMF_MAX_RECORDS", 100000, minimum=1, maximum=1000000)
    maximum_chars = _env_int("EMF_MAX_TEXT_CHARS", 1000000, minimum=1, maximum=16000000)
    if not 2 <= declared_records <= maximum_records:
        raise IngestExtractionError("EMF exceeds EMF_MAX_RECORDS or has invalid record count")
    records = []
    cursor = count = characters = 0
    geometry_safe = True
    eof = False
    while cursor < len(payload):
        if cursor + 8 > len(payload):
            raise IngestExtractionError("EMF record header is truncated")
        record_type, size = struct.unpack_from("<II", payload, cursor)
        if not 1 <= record_type <= 122:
            raise IngestExtractionError("unsupported EMF record type")
        if size < 8 or size % 4 or cursor + size > len(payload):
            raise IngestExtractionError("EMF record size escapes its container")
        count += 1
        if count > maximum_records:
            raise IngestExtractionError("EMF exceeds EMF_MAX_RECORDS")
        if count > 1 and record_type == 1:
            raise IngestExtractionError("EMF contains multiple headers")
        if record_type in {83, 96, 97, 108}:
            raise IngestExtractionError("unsupported EMF text-output variant")
        if record_type in {9, 11, 12, 17, 31, 32, 35, 36}:
            geometry_safe = False
        elif record_type == 10:
            if size != 16:
                raise IngestExtractionError("invalid EMF window-origin record")
            if struct.unpack_from("<ii", payload, cursor + 8) != (0, 0):
                geometry_safe = False
        elif record_type == 22:
            if size != 12:
                raise IngestExtractionError("invalid EMF text-alignment record")
            if struct.unpack_from("<I", payload, cursor + 8)[0] & 1:
                geometry_safe = False
        elif record_type == 82:
            if size < 104:
                raise IngestExtractionError("EMF font record is truncated")
            if any(struct.unpack_from("<ii", payload, cursor + 20)):
                geometry_safe = False
        if record_type == 84:
            if size < 76:
                raise IngestExtractionError("EMF Unicode text record is truncated")
            graphics_mode, scale_x, scale_y = struct.unpack_from("<Iff", payload, cursor + 24)
            if (graphics_mode not in {1, 2}
                    or not all(math.isfinite(v) for v in (scale_x, scale_y))
                    or (graphics_mode == 1 and min(scale_x, scale_y) <= 0)):
                raise IngestExtractionError("invalid EMF text graphics mode or scale")
            x, y, units, string_offset, options = struct.unpack_from("<iiIII", payload, cursor + 36)
            if options & 0x10:
                raise IngestExtractionError("EMF glyph-index text cannot be decoded as Unicode")
            if options & ~0x3C86:
                raise IngestExtractionError("unsupported EMF Unicode text options")
            if units > maximum_chars - characters:
                raise IngestExtractionError("EMF exceeds EMF_MAX_TEXT_CHARS")
            end = string_offset + units * 2
            if string_offset < 76 or string_offset % 2 or end > size:
                raise IngestExtractionError("EMF Unicode string offset escapes its record")
            dx_offset = struct.unpack_from("<I", payload, cursor + 72)[0]
            if dx_offset:
                dx_end = dx_offset + units * (8 if options & 0x2000 else 4)
                if (dx_offset < 76 or dx_offset % 4 or dx_end > size
                        or (units and not (dx_end <= string_offset or end <= dx_offset))):
                    raise IngestExtractionError("EMF character-spacing array escapes or overlaps text")
            try:
                text = payload[cursor + string_offset:cursor + end].decode("utf-16-le", errors="strict")
            except UnicodeDecodeError as error:
                raise IngestExtractionError("EMF text is not valid UTF-16LE") from error
            characters += units
            text = _normalize_extracted_text(text)
            if text.strip():
                position_safe = bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]
                records.append((count, x, y, text, position_safe))
        if record_type == 14:
            if size < 20 or cursor + size != len(payload):
                raise IngestExtractionError("EMF EOF record is invalid or not terminal")
            eof = True
        cursor += size
    if not eof or count != declared_records:
        raise IngestExtractionError("EMF EOF or declared record count mismatch")
    sections = []
    index = 0
    while index < len(records):
        record, x, y, text, position_safe = records[index]
        group = [(record, x, y, text, position_safe)]
        index += 1
        if geometry_safe and position_safe:
            while index < len(records) and records[index][2] == y and records[index][4]:
                group.append(records[index])
                index += 1
            # Preserve original draw-record order. Coordinate anchors provide
            # attribution, without inventing column labels or reading order.
            body = " | ".join(f"[x={item[1]}] {item[3]}" for item in group)
            sections.append(f"Vector row y={y}: {body}")
        else:
            sections.append(f"Vector text record {record}: {text}")
    return _normalize_extracted_text("\n\n".join(sections))


def _docx_embedded_image_text(archive, sources, path):
    """Bounded OCR fallback for a DOCX with no native paragraph/table text."""
    parts = _docx_embedded_image_parts(archive, sources)
    if not parts:
        return ""
    maximum_images = _env_int("DOCX_OCR_MAX_IMAGES", 16, minimum=1, maximum=128)
    maximum_bytes = _env_int(
        "DOCX_OCR_MAX_TOTAL_IMAGE_BYTES", 64 * 1024 * 1024,
        minimum=1, maximum=512 * 1024 * 1024,
    )
    maximum_pixels = _env_int(
        "DOCX_OCR_MAX_TOTAL_IMAGE_PIXELS", 80_000_000,
        minimum=1, maximum=400_000_000,
    )
    if len(parts) > maximum_images:
        raise IngestExtractionError("DOCX exceeds DOCX_OCR_MAX_IMAGES")
    total_bytes = total_pixels = 0
    sections = []
    methods = set()
    for index, name in enumerate(parts, 1):
        info = archive.getinfo(name)
        per_image_bytes = _env_int(
            "OCR_MAX_FILE_BYTES", 100 * 1024 * 1024,
            minimum=1, maximum=2 * 1024 * 1024 * 1024,
        )
        total_bytes += info.file_size
        if info.file_size > per_image_bytes or total_bytes > maximum_bytes:
            raise IngestExtractionError("DOCX embedded images exceed byte limits")
        payload = _read_zip_part(archive, name, required=True)
        if Path(name).suffix.lower() == ".emf":
            text = _emf_unicode_text(payload)
            methods.add("emf-unicode")
            if text.strip():
                sections.append(f"# Embedded image {index}: {Path(name).name}\n\n{text}")
            continue
        if Image is None or pytesseract is None:
            raise IngestExtractionError("DOCX embedded-image OCR requires Pillow and pytesseract")
        try:
            with Image.open(io.BytesIO(payload)) as image:
                width, height = _image_dimensions(image)
                per_image_pixels = _env_int(
                    "OCR_MAX_IMAGE_PIXELS", 40_000_000,
                    minimum=1_000_000, maximum=200_000_000,
                )
                pixels = width * height
                total_pixels += pixels
                if pixels <= 0 or pixels > per_image_pixels or total_pixels > maximum_pixels:
                    raise IngestExtractionError("DOCX embedded images exceed pixel limits")
                text = _normalize_extracted_text(_ocr_image_to_string(image, _ocr_language()))
            methods.add("image-ocr")
        except IngestExtractionError:
            raise
        except Exception as error:
            raise IngestExtractionError(f"DOCX embedded image extraction failed for {name}: {error}") from error
        if text.strip():
            sections.append(f"# Embedded image {index}: {Path(name).name}\n\n{text}")
    if sections:
        _SOURCE_EXTRACTION_METHODS[str(path.resolve())] = "docx-embedded-" + "-and-".join(sorted(methods))
    return _normalize_extracted_text("\n\n".join(sections))



def _docx_has_native_text(root):
    if root.tag.rsplit("}", 1)[-1] in {"del", "moveFrom"}:
        return False
    if root.tag == f"{{{_WORD_NS}}}t" and (root.text or "").strip():
        return True
    return any(_docx_has_native_text(child) for child in root)


def _read_docx_text(path: Path) -> str:
    _SOURCE_EXTRACTION_METHODS[str(path.resolve())] = "docx-office-xml"
    archive = _validated_zip(path, "DOCX")
    try:
        heading_styles = _docx_styles(archive)
        document_name = "word/document.xml"
        document_root = _parse_xml_part(
            _read_zip_part(archive, document_name, required=True),
            document_name,
        )
        body = document_root.find(f".//{{{_WORD_NS}}}body")
        if body is None:
            raise IngestExtractionError(f"DOCX body is missing: {path}")
        sources = [(document_name, document_root)]
        sections: List[str] = []
        main = _docx_render_container(body, heading_styles)
        if main:
            sections.append(f"# Document body\n\n{main}")

        supplemental_parts = [
            name
            for name in archive.namelist()
            if re.fullmatch(r"word/(?:header|footer)\d+\.xml", name)
        ]
        supplemental_parts.extend(
            name for name in ("word/footnotes.xml", "word/endnotes.xml")
            if name in archive.namelist()
        )
        for name in sorted(set(supplemental_parts)):
            root = _parse_xml_part(_read_zip_part(archive, name), name)
            sources.append((name, root))
            label = Path(name).stem.replace("header", "Header ").replace("footer", "Footer ")
            if name.endswith(("footnotes.xml", "endnotes.xml")):
                note_tag = "footnote" if name.endswith("footnotes.xml") else "endnote"
                note_blocks: List[str] = []
                for note in root.findall(f".//{{{_WORD_NS}}}{note_tag}"):
                    note_id = note.get(f"{{{_WORD_NS}}}id") or ""
                    if note_id.startswith("-"):
                        continue
                    rendered = _docx_render_container(note, heading_styles)
                    if rendered:
                        note_blocks.append(f"## {note_tag.title()} {note_id}\n\n{rendered}")
                rendered_part = "\n\n".join(note_blocks)
                label = note_tag.title() + "s"
            else:
                rendered_part = _docx_render_container(root, heading_styles)
            if rendered_part:
                sections.append(f"# {label.strip()}\n\n{rendered_part}")
        extracted = _normalize_extracted_text("\n\n".join(sections))
        if extracted.strip() and any(_docx_has_native_text(root) for _name, root in sources):
            return extracted
        return _docx_embedded_image_text(archive, sources, path)
    finally:
        archive.close()



_DRAWING_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
_PRESENTATION_NS = "http://schemas.openxmlformats.org/presentationml/2006/main"


def _pptx_paragraph_text(paragraph: ET.Element) -> str:
    parts: List[str] = []
    for node in paragraph.iter():
        if node.tag == f"{{{_DRAWING_NS}}}t":
            parts.append(node.text or "")
        elif node.tag == f"{{{_DRAWING_NS}}}br":
            parts.append("\n")
    return _normalize_extracted_text("".join(parts))


def _pptx_ordered_shapes(container: ET.Element) -> List[ET.Element]:
    """Repair only a uniquely established wrapped line displaced in z-order.

    PowerPoint's shape tree is generally the best available authoring order:
    sorting an entire slide by coordinates would separate multi-column lists
    or roadmap labels from their numbers. A very narrow exception is a later
    text box that geometrically continues a preceding same-column line, with
    every intervening text box clearly below it. Ambiguity keeps source order.
    """
    children = list(container)
    # Group-local transforms can include scale, rotation and flips; their
    # coordinates need a full transform model before implying reading order.
    if container.tag == f"{{{_PRESENTATION_NS}}}grpSp":
        return children
    shape_tag = f"{{{_PRESENTATION_NS}}}sp"
    draw = f"{{{_DRAWING_NS}}}"

    def details(child):
        if child.tag != shape_tag:
            return None
        xfrm = child.find(f"./{{{_PRESENTATION_NS}}}spPr/{draw}xfrm")
        if xfrm is None or xfrm.get('rot', '0') != '0' or any(
                xfrm.get(key, '0') not in ('0', 'false') for key in ('flipH', 'flipV')):
            return None
        offset, extent = xfrm.find(f'{draw}off'), xfrm.find(f'{draw}ext')
        try:
            x, y = int(offset.get('x')), int(offset.get('y'))
            width, height = int(extent.get('cx')), int(extent.get('cy'))
        except (AttributeError, TypeError, ValueError):
            return None
        if width <= 0 or height <= 0:
            return None
        body_properties = child.find(f"./{{{_PRESENTATION_NS}}}txBody/{draw}bodyPr")
        if body_properties is not None and body_properties.get('vert', 'horz') != 'horz':
            return None
        paragraphs = [_pptx_paragraph_text(item).strip() for item in child.findall(
            f"./{{{_PRESENTATION_NS}}}txBody/{draw}p")]
        paragraphs = [item for item in paragraphs if item]
        if len(paragraphs) != 1 or '\n' in paragraphs[0]:
            return None
        return x, y, width, height, paragraphs[0]

    info = [details(child) for child in children]
    proposed = []
    for tail_index, tail in enumerate(info):
        if tail is None or not re.search(r'[.!?。！？]\s*$', tail[4]) or len(tail[4]) > 200:
            continue
        options = []
        for head_index in range(tail_index):
            head = info[head_index]
            if (head is None or head[:1] != tail[:1] or head[2] != tail[2]
                    or re.search(r'[.!?。！？:;]\s*$', head[4])
                    or head[4].strip() in {'•', '*', '-', '–', '—'}):
                continue
            if (tail[1] <= head[1] or abs(tail[1] - (head[1] + head[3])) > head[3] // 3
                    or abs(tail[3] - head[3]) > max(tail[3], head[3]) // 3):
                continue
            # Overlapping same-column candidates make the continuation
            # relationship ambiguous even when only one appears earlier in
            # the XML tree. Do not choose one based on z-order.
            if any(other is not None and index not in (head_index, tail_index)
                   and other[0] == head[0] and other[2] == head[2]
                   and (abs(other[1] - head[1]) <= min(other[3], head[3]) // 3
                        or abs(other[1] - tail[1]) <= min(other[3], tail[3]) // 3)
                   for index, other in enumerate(info)):
                continue
            between = children[head_index + 1:tail_index]
            # Do not cross a nested group or a table: their relative reading
            # order may encode a different layout that coordinates cannot prove.
            if any(child.tag in {f"{{{_PRESENTATION_NS}}}grpSp",
                                 f"{{{_PRESENTATION_NS}}}graphicFrame"}
                   for child in between):
                continue
            if any(info[index] is None and any((node.text or '').strip()
                        for node in children[index].iter(f'{draw}t'))
                   for index in range(head_index + 1, tail_index)):
                continue
            intervening = [info[index] for index in range(head_index + 1, tail_index)
                           if info[index] is not None]
            if not intervening or any(item[1] < tail[1] + tail[3] for item in intervening):
                continue
            def same_reading_column(item):
                if item[4].strip() in {'•', '*', '-', '–', '—'}:
                    # Bullet glyphs are often their own narrow text boxes on
                    # either edge of a left-to-right or right-to-left column.
                    return min(abs(item[0] - (head[0] + head[2])),
                               abs((item[0] + item[2]) - head[0])) <= 2 * head[3]
                return (item[0] == head[0]
                        and abs(item[2] - head[2]) <= head[2] // 5)
            if any(not same_reading_column(item) for item in intervening):
                continue
            options.append(head_index)
        if len(options) == 1:
            proposed.append((options[0], tail_index))
    # Two potential continuations for one line are not enough evidence to
    # reorder either; likewise a chain would need a richer layout model.
    heads = [head for head, _ in proposed]
    tails = [tail for _, tail in proposed]
    pairs = [(head, tail) for head, tail in proposed
             if heads.count(head) == 1 and head not in tails and tail not in heads]
    ordered = children[:]
    for head, tail in pairs:
        item = children[tail]
        ordered.remove(item)
        ordered.insert(ordered.index(children[head]) + 1, item)
    return ordered


def _pptx_render_shapes(container: ET.Element) -> str:
    blocks: List[str] = []
    for child in _pptx_ordered_shapes(container):
        if child.tag == f"{{{_PRESENTATION_NS}}}sp":
            paragraphs = [
                _pptx_paragraph_text(node)
                for node in child.findall(
                    f"./{{{_PRESENTATION_NS}}}txBody/{{{_DRAWING_NS}}}p"
                )
            ]
            blocks.extend(value for value in paragraphs if value.strip())
        elif child.tag == f"{{{_PRESENTATION_NS}}}graphicFrame":
            for table in child.findall(f".//{{{_DRAWING_NS}}}tbl"):
                rows = [
                    [
                        " / ".join(
                            text for paragraph in cell.findall(f".//{{{_DRAWING_NS}}}p")
                            if (text := _pptx_paragraph_text(paragraph)).strip()
                        )
                        for cell in row.findall(f"./{{{_DRAWING_NS}}}tc")
                    ]
                    for row in table.findall(f"./{{{_DRAWING_NS}}}tr")
                ]
                rendered = _structured_table_records(rows)
                if rendered:
                    blocks.append(rendered)
        elif child.tag == f"{{{_PRESENTATION_NS}}}grpSp":
            grouped = _pptx_render_shapes(child)
            if grouped:
                blocks.append(grouped)
    return "\n\n".join(blocks)


def _read_pptx_text(path: Path) -> str:
    """Read slide text/tables in presentation order without extracting members."""
    archive = _validated_zip(path, "PPTX")
    try:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise IngestExtractionError("PPTX has ambiguous duplicate archive members")
        presentation_name = "ppt/presentation.xml"
        presentation = _parse_xml_part(
            _read_zip_part(archive, presentation_name, required=True), presentation_name
        )
        relations_name = "ppt/_rels/presentation.xml.rels"
        relations = _parse_xml_part(
            _read_zip_part(archive, relations_name, required=True), relations_name
        )
        relationships: Dict[str, str] = {}
        for node in relations.findall(f"./{{{_PACKAGE_REL_NS}}}Relationship"):
            if not (node.get("Type") or "").endswith("/slide"):
                continue
            if node.get("TargetMode", "").lower() == "external":
                raise IngestExtractionError("external PPTX slide targets are not supported")
            relationship_id = node.get("Id") or ""
            target = node.get("Target") or ""
            if not relationship_id or relationship_id in relationships:
                raise IngestExtractionError("ambiguous PPTX slide relationship")
            if "\\" in target or "\x00" in target:
                raise IngestExtractionError(f"unsafe PPTX slide target: {target!r}")
            normalized = posixpath.normpath(
                target.lstrip("/") if target.startswith("/") else posixpath.join("ppt", target)
            )
            if not re.fullmatch(r"ppt/slides/[^/]+\.xml", normalized):
                raise IngestExtractionError(f"unsafe PPTX slide target: {target!r}")
            relationships[relationship_id] = normalized
        sections: List[str] = []
        seen_slides = set()
        for number, slide_id in enumerate(
            presentation.findall(
                f"./{{{_PRESENTATION_NS}}}sldIdLst/{{{_PRESENTATION_NS}}}sldId"
            ), 1
        ):
            relationship_id = slide_id.get(f"{{{_OFFICE_REL_NS}}}id") or ""
            slide_name = relationships.get(relationship_id)
            if not slide_name or slide_name in seen_slides:
                raise IngestExtractionError("missing or duplicate PPTX slide relationship")
            seen_slides.add(slide_name)
            slide = _parse_xml_part(
                _read_zip_part(archive, slide_name, required=True), slide_name
            )
            shapes = slide.find(
                f"./{{{_PRESENTATION_NS}}}cSld/{{{_PRESENTATION_NS}}}spTree"
            )
            if shapes is None:
                raise IngestExtractionError(f"PPTX slide shape tree missing: {slide_name}")
            rendered = _pptx_render_shapes(shapes)
            if rendered.strip():
                hidden = " (hidden)" if slide.get("show") == "0" else ""
                sections.append(f"# Slide {number}{hidden}\n\n{rendered}")
        return _normalize_extracted_text("\n\n".join(sections))
    finally:
        archive.close()


def _extraction_method(path: Path, *, pdf_native_only: bool = False) -> str:
    suffix = path.suffix.lower()
    if suffix == ".docx" and str(path.resolve()) in _SOURCE_EXTRACTION_METHODS:
        return _SOURCE_EXTRACTION_METHODS[str(path.resolve())]
    if suffix == ".pdf":
        if pdf_native_only:
            return "pdf-native-only"
        return "pdf-native-with-bounded-ocr" if _env_flag("PDF_OCR_FALLBACK", True) else "pdf-native"
    if suffix in {".jpg", ".jpeg", ".png", ".webp"}:
        return "tesseract-" + _ocr_language()
    if suffix in {".docx", ".xlsx", ".pptx"}:
        return suffix[1:] + "-office-xml"
    return suffix.lstrip(".") + "-text"

def _xlsx_shared_strings(archive: zipfile.ZipFile) -> List[str]:
    payload = _read_zip_part(archive, "xl/sharedStrings.xml")
    if not payload:
        return []
    root = _parse_xml_part(payload, "xl/sharedStrings.xml")
    return [
        _normalize_extracted_text("".join(node.text or "" for node in item.iter() if node.tag.endswith("}t")))
        for item in root.findall(f".//{{{_SHEET_NS}}}si")
    ]


def _xlsx_style_formats(archive: zipfile.ZipFile) -> List[Tuple[int, str]]:
    payload = _read_zip_part(archive, "xl/styles.xml")
    if not payload:
        return []
    root = _parse_xml_part(payload, "xl/styles.xml")
    custom = {
        int(node.get("numFmtId") or 0): node.get("formatCode") or ""
        for node in root.findall(f".//{{{_SHEET_NS}}}numFmt")
    }
    cell_xfs = root.find(f".//{{{_SHEET_NS}}}cellXfs")
    if cell_xfs is None:
        return []
    return [
        (int(node.get("numFmtId") or 0), custom.get(int(node.get("numFmtId") or 0), ""))
        for node in cell_xfs.findall(f"./{{{_SHEET_NS}}}xf")
    ]


def _xlsx_is_date_format(num_fmt_id: int, format_code: str) -> bool:
    if num_fmt_id in set(range(14, 23)) | set(range(27, 37)) | {45, 46, 47} | set(range(50, 59)):
        return True
    cleaned = re.sub(r'"[^"]*"|\\.|\[[^\]]*\]', "", format_code or "").lower()
    return bool(re.search(r"[dyhs]", cleaned) or re.search(r"m+[/:-]", cleaned))


def _xlsx_excel_datetime(value: str, date_1904: bool) -> str:
    try:
        serial = float(value)
    except (TypeError, ValueError):
        return value
    if date_1904:
        base = datetime(1904, 1, 1)
    else:
        # Excel intentionally emulates Lotus 1-2-3's nonexistent 1900-02-29.
        # Serial 1 is 1900-01-01, while serials 60+ need the compatibility-day
        # offset. A single 1899-12-30 base is wrong for dates before serial 60.
        base = datetime(1899, 12, 31) if serial < 60 else datetime(1899, 12, 30)
    converted = base + timedelta(days=serial)
    if converted.time() == datetime.min.time():
        return converted.date().isoformat()
    if converted.date() == base.date() and 0 <= serial < 1:
        return converted.time().isoformat(timespec="seconds")
    return converted.isoformat(sep=" ", timespec="seconds")


def _xlsx_cell_value(
    cell: ET.Element,
    shared_strings: List[str],
    style_formats: List[Tuple[int, str]],
    date_1904: bool,
) -> str:
    cell_type = cell.get("t") or "n"
    style_index = int(cell.get("s") or 0)
    formula = cell.find(f"./{{{_SHEET_NS}}}f")
    raw_value_node = cell.find(f"./{{{_SHEET_NS}}}v")
    raw_value = raw_value_node.text if raw_value_node is not None and raw_value_node.text else ""

    if cell_type == "inlineStr":
        value = "".join(
            node.text or ""
            for node in cell.findall(f".//{{{_SHEET_NS}}}is//{{{_SHEET_NS}}}t")
        )
    elif cell_type == "s":
        try:
            value = shared_strings[int(raw_value)]
        except (ValueError, IndexError):
            raise IngestExtractionError(f"XLSX shared-string index is invalid: {raw_value!r}")
    elif cell_type == "b":
        value = "TRUE" if raw_value == "1" else "FALSE"
    elif cell_type == "e":
        value = f"Error: {raw_value or 'unknown'}"
    elif cell_type in {"str", "d"}:
        value = raw_value
    else:
        value = raw_value
        if value and style_index < len(style_formats):
            num_fmt_id, format_code = style_formats[style_index]
            if _xlsx_is_date_format(num_fmt_id, format_code):
                value = _xlsx_excel_datetime(value, date_1904)

    value = _normalize_extracted_text(value)
    if formula is not None:
        expression = (formula.text or "").strip()
        formula_text = f"Formula: ={expression}" if expression else "Formula: [shared formula]"
        if value:
            return f"{formula_text}; Cached value: {value}"
        return formula_text
    return value


def _xlsx_column_index(reference: str) -> int:
    match = re.match(r"([A-Za-z]{1,3})", reference or "")
    if not match:
        return 0
    index = 0
    for char in match.group(1).upper():
        index = index * 26 + ord(char) - ord("A") + 1
    return index


def _xlsx_column_name(index: int) -> str:
    if index < 1:
        return "A"
    name = ""
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(ord("A") + remainder) + name
    return name


def _xlsx_cell_coordinates(reference: str) -> Optional[Tuple[int, int]]:
    match = re.fullmatch(r"\$?([A-Za-z]{1,3})\$?([1-9][0-9]{0,6})", reference or "")
    if not match:
        return None
    column = _xlsx_column_index(match.group(1))
    row = int(match.group(2))
    return row, column


def _xlsx_sheet_rows(
    root: ET.Element,
    shared_strings: List[str],
    style_formats: List[Tuple[int, str]],
    date_1904: bool,
) -> List[Tuple[int, Dict[int, str]]]:
    row_values: Dict[int, Dict[int, str]] = {}
    cell_count = 0
    max_cells = _env_int("XLSX_MAX_CELLS", 2_000_000, minimum=1, maximum=10_000_000)
    for fallback_row, row in enumerate(root.findall(f".//{{{_SHEET_NS}}}sheetData/{{{_SHEET_NS}}}row"), 1):
        row_number = int(row.get("r") or fallback_row)
        values: Dict[int, str] = {}
        fallback_column = 0
        for cell in row.findall(f"./{{{_SHEET_NS}}}c"):
            fallback_column += 1
            column = _xlsx_column_index(cell.get("r") or "") or fallback_column
            value = _xlsx_cell_value(cell, shared_strings, style_formats, date_1904)
            if value:
                values[column] = value
            cell_count += 1
            if cell_count > max_cells:
                raise IngestExtractionError(
                    f"XLSX worksheet exceeds XLSX_MAX_CELLS={max_cells}"
                )
        if values:
            row_values[row_number] = values

    # Expand merged values into otherwise-empty cells. This is especially
    # important for vertically merged categories: without propagation, every
    # row after the first loses the entity it belongs to once rows are chunked.
    for merged in root.findall(f".//{{{_SHEET_NS}}}mergeCells/{{{_SHEET_NS}}}mergeCell"):
        reference = (merged.get("ref") or "").strip()
        bounds = reference.split(":", 1)
        start = _xlsx_cell_coordinates(bounds[0]) if bounds else None
        end = _xlsx_cell_coordinates(bounds[-1]) if bounds else None
        if start is None or end is None:
            raise IngestExtractionError(f"invalid XLSX merged-cell range: {reference!r}")
        start_row, start_column = start
        end_row, end_column = end
        if end_row < start_row or end_column < start_column:
            raise IngestExtractionError(f"reversed XLSX merged-cell range: {reference!r}")
        anchor = row_values.get(start_row, {}).get(start_column, "")
        if not anchor:
            continue
        expansion = (end_row - start_row + 1) * (end_column - start_column + 1)
        cell_count += expansion - 1
        if cell_count > max_cells:
            raise IngestExtractionError(
                f"XLSX worksheet exceeds XLSX_MAX_CELLS={max_cells} after merged-cell expansion"
            )
        for row_number in range(start_row, end_row + 1):
            values = row_values.setdefault(row_number, {})
            for column in range(start_column, end_column + 1):
                values.setdefault(column, anchor)

    return sorted(
        ((row_number, values) for row_number, values in row_values.items() if values),
        key=lambda item: item[0],
    )


def _xlsx_header_index(
    rows: List[Tuple[int, Dict[int, str]]], columns: List[int]
) -> Optional[int]:
    """Find a likely header after optional report titles/notes."""
    if len(rows) < 2:
        return None
    scan_rows = _env_int("XLSX_HEADER_SCAN_ROWS", 20, minimum=1, maximum=100)
    candidates: List[Tuple[float, int]] = []
    for index, (_row_number, row) in enumerate(rows[: min(scan_rows, len(rows) - 1)]):
        values = [_markdown_cell(row.get(column, "")) for column in columns]
        nonempty = [value for value in values if value]
        if len(nonempty) < 2 or len(set(nonempty)) < 2:
            continue
        textual = sum(
            bool(re.search(r"[^\W\d_]", value, flags=re.UNICODE)) for value in nonempty
        )
        textual_ratio = textual / len(nonempty)
        if textual_ratio < 0.60:
            continue
        density = len(nonempty) / max(1, len(columns))
        # Prefer a fuller row; use the earlier row as a deterministic tie-break.
        candidates.append((len(nonempty) * 10.0 + density + textual_ratio, index))
    if not candidates:
        return None
    best_score = max(score for score, _index in candidates)
    return min(index for score, index in candidates if score == best_score)


def _xlsx_header_row(rows: List[Tuple[int, Dict[int, str]]], columns: List[int]) -> bool:
    """Backward-compatible predicate used by older integrations/tests."""
    return _xlsx_header_index(rows, columns) == 0


def _xlsx_render_sheet(name: str, state: str, rows: List[Tuple[int, Dict[int, str]]]) -> str:
    if not rows:
        return ""
    columns = sorted({column for _row_number, values in rows for column in values})
    output = [f"# Sheet: {name}"]
    if state and state != "visible":
        output.append(f"Sheet state: {state}")

    header_index = _xlsx_header_index(rows, columns)
    has_header = header_index is not None
    header_values = rows[header_index][1] if header_index is not None else {}
    preamble_rows = rows[:header_index] if header_index is not None else []
    data_rows = rows[header_index + 1:] if header_index is not None else rows
    header_labels = {
        column: (
            f"{_markdown_cell(header_values[column])} ({_xlsx_column_name(column)})"
            if header_values.get(column)
            else _xlsx_column_name(column)
        )
        for column in columns
    }

    output.append(
        "Columns: "
        + "; ".join(
            f"{_xlsx_column_name(column)} = {header_labels[column]}" for column in columns
        )
    )

    if preamble_rows:
        output.append("## Sheet context")
        context_records: List[str] = []
        for row_number, values in preamble_rows:
            seen_values: set[str] = set()
            facts: List[str] = []
            for column in sorted(values):
                value = _markdown_cell(values[column])
                if not value or value in seen_values:
                    continue
                seen_values.add(value)
                facts.append(f"- Cell {_xlsx_column_name(column)}{row_number}: {value}")
            if facts:
                context_records.append(
                    f"Spreadsheet context row {row_number}:\n" + "\n".join(facts)
                )
        if context_records:
            output.append("\n\n".join(context_records))

    # Repeat labels on every record. This is intentionally more verbose than a
    # visual grid: a row remains an unambiguous fact after embedding, chunking,
    # reranking, and bilingual question matching.
    batch_size = _env_int("XLSX_ROWS_PER_SECTION", 25, minimum=1, maximum=1000)
    if not data_rows:
        return "\n\n".join(output)
    for start in range(0, len(data_rows), batch_size):
        batch = data_rows[start:start + batch_size]
        first_row, last_row = batch[0][0], batch[-1][0]
        output.append(
            f"## Rows {first_row}-{last_row}"
            if first_row != last_row
            else f"## Row {first_row}"
        )
        row_records: List[str] = []
        for row_number, values in batch:
            facts = [
                f"- {header_labels.get(column, _xlsx_column_name(column))}: "
                f"{_markdown_cell(values[column])}"
                for column in sorted(values)
                if _markdown_cell(values[column])
            ]
            if facts:
                row_records.append(f"Spreadsheet row {row_number}:\n" + "\n".join(facts))
        if row_records:
            output.append("\n\n".join(row_records))
    return "\n\n".join(output)


def _read_xlsx_text(path: Path) -> str:
    archive = _validated_zip(path, "XLSX")
    try:
        workbook_name = "xl/workbook.xml"
        workbook_root = _parse_xml_part(
            _read_zip_part(archive, workbook_name, required=True), workbook_name
        )
        relationships_name = "xl/_rels/workbook.xml.rels"
        relationships_root = _parse_xml_part(
            _read_zip_part(archive, relationships_name, required=True), relationships_name
        )
        relationships = {
            node.get("Id") or "": node.get("Target") or ""
            for node in relationships_root.findall(f".//{{{_PACKAGE_REL_NS}}}Relationship")
        }
        workbook_properties = workbook_root.find(f"./{{{_SHEET_NS}}}workbookPr")
        date_1904 = bool(
            workbook_properties is not None
            and (workbook_properties.get("date1904") or "").lower() in {"1", "true"}
        )
        shared_strings = _xlsx_shared_strings(archive)
        style_formats = _xlsx_style_formats(archive)
        sections: List[str] = []
        for sheet in workbook_root.findall(f".//{{{_SHEET_NS}}}sheets/{{{_SHEET_NS}}}sheet"):
            name = sheet.get("name") or "Untitled sheet"
            state = sheet.get("state") or "visible"
            relationship_id = sheet.get(f"{{{_OFFICE_REL_NS}}}id") or ""
            target = relationships.get(relationship_id)
            if not target:
                raise IngestExtractionError(
                    f"XLSX worksheet relationship is missing for sheet {name!r}"
                )
            normalized_target = target.lstrip("/")
            if not normalized_target.startswith("xl/"):
                normalized_target = posixpath.normpath(posixpath.join("xl", normalized_target))
            if normalized_target.startswith("../"):
                raise IngestExtractionError(f"unsafe XLSX worksheet target: {target}")
            sheet_root = _parse_xml_part(
                _read_zip_part(archive, normalized_target, required=True),
                normalized_target,
            )
            rows = _xlsx_sheet_rows(
                sheet_root, shared_strings, style_formats, date_1904
            )
            rendered = _xlsx_render_sheet(name, state, rows)
            if rendered:
                sections.append(rendered)
        return _normalize_extracted_text("\n\n".join(sections))
    finally:
        archive.close()


class _SemanticHTMLParser(HTMLParser):
    """Convert visible HTML structure to retrieval-friendly Markdown text."""

    _SKIP_TAGS = {"script", "style", "noscript", "template", "svg", "canvas"}
    _VOID_TAGS = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    _CAPTURE_TAGS = {"title", "p", "li", "dt", "dd", "pre", "blockquote"} | {
        f"h{level}" for level in range(1, 7)
    }
    _BOUNDARY_TAGS = {"article", "aside", "body", "div", "footer", "header", "main", "section"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: List[str] = []
        self.title = ""
        self.description = ""
        self._skip_depth = 0
        self._active_tag = ""
        self._active_parts: List[str] = []
        self._loose_parts: List[str] = []
        self._table_rows: Optional[List[List[str]]] = None
        self._table_row: Optional[Dict[int, str]] = None
        self._table_cell: Optional[List[str]] = None
        self._table_next_column = 0
        self._table_cell_colspan = 1
        self._table_cell_rowspan = 1
        self._table_rowspans: Dict[int, Tuple[int, str]] = {}
        self._table_has_header_cells = False
        self._table_current_row_has_th = False
        self._table_current_row_has_td = False
        self._table_caption_parts: Optional[List[str]] = None
        self._table_caption = ""

    @staticmethod
    def _attrs(values: List[Tuple[str, Optional[str]]]) -> Dict[str, str]:
        return {str(key).lower(): str(value or "") for key, value in values}

    def _append_text(self, value: str) -> None:
        if not value:
            return
        if self._table_caption_parts is not None:
            self._table_caption_parts.append(value)
        elif self._table_cell is not None:
            self._table_cell.append(value)
        elif self._active_tag:
            self._active_parts.append(value)
        else:
            self._loose_parts.append(value)

    def _flush_loose(self) -> None:
        text = _normalize_extracted_text(" ".join(self._loose_parts))
        self._loose_parts = []
        if text:
            self.blocks.append(text)

    def _finish_capture(self) -> None:
        tag = self._active_tag
        text = _normalize_extracted_text("".join(self._active_parts))
        self._active_tag = ""
        self._active_parts = []
        if not text:
            return
        if tag == "title":
            self.title = text
        elif tag.startswith("h") and tag[1:].isdigit():
            self.blocks.append(f"{'#' * int(tag[1:])} {text}")
        elif tag == "li":
            self.blocks.append(f"- {text}")
        elif tag == "dt":
            self.blocks.append(f"**{text}**")
        elif tag == "pre":
            self.blocks.append(f"```text\n{text}\n```")
        elif tag == "blockquote":
            self.blocks.append("\n".join(f"> {line}" for line in text.splitlines()))
        else:
            self.blocks.append(text)

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        values = self._attrs(attrs)
        if self._skip_depth:
            if tag not in self._VOID_TAGS:
                self._skip_depth += 1
            return
        hidden = (
            tag in self._SKIP_TAGS
            or "hidden" in values
            or values.get("aria-hidden", "").lower() == "true"
            or "display:none" in values.get("style", "").replace(" ", "").lower()
            or "visibility:hidden" in values.get("style", "").replace(" ", "").lower()
        )
        if hidden:
            if tag not in self._VOID_TAGS:
                self._skip_depth = 1
            return
        if tag == "meta":
            name = values.get("name", "").lower()
            if name == "description" and values.get("content"):
                self.description = values["content"].strip()
            return
        if tag == "img" and values.get("alt"):
            self._append_text(f" [Image: {values['alt'].strip()}] ")
            return
        if tag == "br":
            self._append_text("\n")
            return
        if tag == "table":
            if self._active_tag:
                self._finish_capture()
            self._flush_loose()
            self._table_rows = []
            self._table_caption = ""
            self._table_rowspans = {}
            self._table_has_header_cells = False
            return
        if self._table_rows is not None:
            if tag == "caption":
                self._table_caption_parts = []
            elif tag == "tr":
                self._table_row = {}
                self._table_next_column = 0
                self._table_current_row_has_th = False
                self._table_current_row_has_td = False
                for column, (remaining, value) in list(self._table_rowspans.items()):
                    self._table_row[column] = value
                    if remaining <= 1:
                        del self._table_rowspans[column]
                    else:
                        self._table_rowspans[column] = (remaining - 1, value)
            elif tag in {"th", "td"}:
                self._table_cell = []
                self._table_cell_colspan = max(
                    1, min(1000, int(values.get("colspan", "1")) if values.get("colspan", "1").isdigit() else 1)
                )
                self._table_cell_rowspan = max(
                    1, min(100000, int(values.get("rowspan", "1")) if values.get("rowspan", "1").isdigit() else 1)
                )
                if tag == "th":
                    self._table_current_row_has_th = True
                else:
                    self._table_current_row_has_td = True
            return
        if tag in self._CAPTURE_TAGS and not self._active_tag:
            self._flush_loose()
            self._active_tag = tag
            self._active_parts = []
        elif tag in self._BOUNDARY_TAGS and not self._active_tag:
            self._flush_loose()

    def handle_startendtag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        before = self._skip_depth
        self.handle_starttag(tag, attrs)
        if self._skip_depth > before and tag.lower() not in self._VOID_TAGS:
            self._skip_depth -= 1

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._skip_depth:
            self._skip_depth -= 1
            return
        if self._table_rows is not None:
            if tag == "caption" and self._table_caption_parts is not None:
                self._table_caption = _normalize_extracted_text(
                    "".join(self._table_caption_parts)
                )
                self._table_caption_parts = None
            elif tag in {"th", "td"} and self._table_cell is not None:
                value = _normalize_extracted_text("".join(self._table_cell))
                if self._table_row is None:
                    self._table_row = {}
                while self._table_next_column in self._table_row:
                    self._table_next_column += 1
                for offset in range(self._table_cell_colspan):
                    column = self._table_next_column + offset
                    self._table_row[column] = value
                    if self._table_cell_rowspan > 1:
                        self._table_rowspans[column] = (
                            self._table_cell_rowspan - 1,
                            value,
                        )
                self._table_next_column += self._table_cell_colspan
                self._table_cell = None
            elif tag == "tr" and self._table_row is not None:
                if any(self._table_row.values()):
                    if (
                        not self._table_rows
                        and self._table_current_row_has_th
                        and not self._table_current_row_has_td
                    ):
                        self._table_has_header_cells = True
                    width = max(self._table_row, default=-1) + 1
                    self._table_rows.append(
                        [self._table_row.get(column, "") for column in range(width)]
                    )
                self._table_row = None
            elif tag == "table":
                rendered = _structured_table_records(
                    self._table_rows,
                    force_header=self._table_has_header_cells,
                )
                if rendered:
                    if self._table_caption:
                        rendered = f"Table: {self._table_caption}\n\n{rendered}"
                    self.blocks.append(rendered)
                self._table_rows = None
                self._table_row = None
                self._table_cell = None
                self._table_caption_parts = None
                self._table_caption = ""
                self._table_rowspans = {}
                self._table_has_header_cells = False
            return
        if self._active_tag and tag == self._active_tag:
            self._finish_capture()
        elif tag in self._BOUNDARY_TAGS and not self._active_tag:
            self._flush_loose()

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._append_text(data)

    def rendered_text(self) -> str:
        if self._active_tag:
            self._finish_capture()
        self._flush_loose()
        prefix: List[str] = []
        if self.title:
            prefix.append(f"# Page title: {self.title}")
        if self.description:
            prefix.append(f"Page description: {self.description}")
        return _normalize_extracted_text("\n\n".join(prefix + self.blocks))


def _decode_html_bytes(raw: bytes) -> str:
    sniff = raw[:8192].decode("latin-1", errors="ignore")
    declared = re.search(
        r"charset\s*=\s*[\"']?\s*([A-Za-z0-9._-]{1,40})",
        sniff,
        flags=re.IGNORECASE,
    )
    if declared:
        encoding = declared.group(1)
        try:
            decoded = raw.decode(encoding, errors="strict")
        except (LookupError, UnicodeDecodeError):
            decoded = ""
        if decoded and _decoded_text_score(decoded) >= 50.0:
            return _normalize_extracted_text(decoded)
    return _decode_text_bytes(raw)


def _read_html_text(path: Path) -> str:
    max_bytes = _env_int(
        "INGEST_MAX_TEXT_FILE_BYTES",
        64 * 1024 * 1024,
        minimum=1,
        maximum=2 * 1024 * 1024 * 1024,
    )
    if path.stat().st_size > max_bytes:
        raise IngestExtractionError(
            f"HTML source exceeds INGEST_MAX_TEXT_FILE_BYTES ({max_bytes}): {path}"
        )
    try:
        source = _decode_html_bytes(path.read_bytes())
        parser = _SemanticHTMLParser()
        parser.feed(source)
        parser.close()
        return parser.rendered_text()
    except IngestExtractionError:
        raise
    except Exception as error:
        raise IngestExtractionError(f"failed to parse HTML source {path}: {error}") from error


_PS_SHOW_OPERATOR = re.compile(
    br"(?<![A-Za-z])(?:show|ashow|widthshow|awidthshow|kshow|xshow|yshow|xyshow)(?![A-Za-z])"
)


def _decode_postscript_string(raw: bytes) -> str:
    if not raw:
        return ""
    if raw.startswith((b"\xfe\xff", b"\xff\xfe")):
        try:
            return _normalize_extracted_text(raw.decode("utf-16"))
        except UnicodeDecodeError:
            pass
    try:
        return _normalize_extracted_text(raw.decode("utf-8", errors="strict"))
    except UnicodeDecodeError:
        pass
    candidates: List[Tuple[float, str]] = []
    for encoding in ("cp1255", "cp1252"):
        try:
            decoded = raw.decode(encoding, errors="strict")
        except UnicodeDecodeError:
            continue
        score = _decoded_text_score(decoded)
        if encoding == "cp1255" and re.search(r"[\u05d0-\u05ea]", decoded):
            score += 40.0
        candidates.append((score, decoded))
    return _normalize_extracted_text(max(candidates, default=(0.0, ""))[1])


def _decode_postscript_output(raw: bytes) -> str:
    """Decode extractor output while retaining form-feed page boundaries."""
    page_payloads = raw.split(b"\x0c")
    if len(page_payloads) <= 1:
        return _decode_text_bytes(raw)
    sections: List[str] = []
    page_number = 0
    for payload in page_payloads:
        if not payload.strip():
            continue
        page_number += 1
        decoded = _decode_text_bytes(payload)
        section = _page_section(page_number, decoded)
        if section:
            sections.append(section)
    return _normalize_extracted_text("\n\n".join(sections))


def _parse_postscript_literal(raw: bytes, start: int) -> Tuple[int, bytes]:
    depth = 1
    index = start + 1
    output = bytearray()
    escapes = {ord("n"): b"\n", ord("r"): b"\r", ord("t"): b"\t", ord("b"): b"\b", ord("f"): b"\f"}
    while index < len(raw) and depth:
        value = raw[index]
        if value == ord("\\"):
            index += 1
            if index >= len(raw):
                break
            escaped = raw[index]
            if escaped in escapes:
                output.extend(escapes[escaped])
            elif escaped in b"\r\n":
                if escaped == ord("\r") and index + 1 < len(raw) and raw[index + 1] == ord("\n"):
                    index += 1
            elif ord("0") <= escaped <= ord("7"):
                digits = bytes([escaped])
                for _ in range(2):
                    if index + 1 < len(raw) and ord("0") <= raw[index + 1] <= ord("7"):
                        index += 1
                        digits += bytes([raw[index]])
                    else:
                        break
                output.append(int(digits, 8) & 0xFF)
            else:
                output.append(escaped)
        elif value == ord("("):
            depth += 1
            output.append(value)
        elif value == ord(")"):
            depth -= 1
            if depth:
                output.append(value)
        else:
            output.append(value)
        index += 1
    if depth:
        raise IngestExtractionError("unterminated PostScript string literal")
    return index, bytes(output)


def _read_postscript_text_internal(raw: bytes) -> str:
    records: List[Tuple[int, int, str]] = []
    dsc_pages = [
        match.start()
        for match in re.finditer(br"^%%Page:\s+", raw, flags=re.MULTILINE)
    ]
    show_pages = [
        match.start()
        for match in re.finditer(
            br"(?<![A-Za-z])showpage(?![A-Za-z])", raw
        )
    ]
    index = 0
    while index < len(raw):
        value = raw[index]
        parsed = b""
        end = index + 1
        if value == ord("("):
            end, parsed = _parse_postscript_literal(raw, index)
        elif value == ord("<") and index + 1 < len(raw) and raw[index + 1] != ord("<"):
            close = raw.find(b">", index + 1)
            if close >= 0:
                candidate = re.sub(br"\s+", b"", raw[index + 1:close])
                if candidate and re.fullmatch(br"[0-9A-Fa-f]+", candidate):
                    if len(candidate) % 2:
                        candidate += b"0"
                    parsed = bytes.fromhex(candidate.decode("ascii"))
                    end = close + 1
        if parsed:
            lookahead = raw[end:min(len(raw), end + 256)]
            if _PS_SHOW_OPERATOR.search(lookahead):
                decoded = _decode_postscript_string(parsed)
                if decoded and re.search(r"[\w\u0590-\u05ff]", decoded, flags=re.UNICODE):
                    if dsc_pages:
                        page_number = max(1, bisect_right(dsc_pages, index))
                    else:
                        page_number = bisect_right(show_pages, index) + 1
                    records.append((page_number, raw.count(b"\n", 0, index), decoded))
            index = end
        else:
            index += 1

    grouped: Dict[int, List[str]] = {}
    last_page_number: Optional[int] = None
    last_line_number: Optional[int] = None
    for page_number, line_number, text in records:
        page_lines = grouped.setdefault(page_number, [])
        if page_lines and last_page_number == page_number and last_line_number == line_number:
            page_lines[-1] = f"{page_lines[-1]} {text}"
        else:
            page_lines.append(text)
        last_page_number = page_number
        last_line_number = line_number

    metadata: List[str] = []
    for match in re.finditer(br"^%%(Title|Subject|Keywords):\s*(.+)$", raw, flags=re.MULTILINE):
        label = match.group(1).decode("ascii", errors="ignore")
        value = _decode_postscript_string(match.group(2).strip())
        if value:
            metadata.append(f"{label}: {value}")
    sections: List[str] = []
    if metadata:
        sections.append("# PostScript metadata\n\n" + "\n".join(metadata))
    for page_number in sorted(grouped):
        page_text = "\n".join(grouped[page_number])
        section = _page_section(page_number, page_text)
        if section:
            sections.append(section)
    return _normalize_extracted_text("\n\n".join(sections))


def _postscript_command(path: Path, extractor: str) -> Optional[List[str]]:
    candidates: List[Tuple[str, Optional[str]]] = []
    if extractor in {"auto", "ghostscript"}:
        candidates.extend(("ghostscript", shutil.which(name)) for name in ("gs", "gswin64c", "gswin32c"))
    if extractor in {"auto", "ps2ascii"}:
        candidates.append(("ps2ascii", shutil.which("ps2ascii")))
    for kind, executable in candidates:
        if not executable:
            continue
        if kind == "ps2ascii":
            return [executable, str(path)]
        return [
            executable,
            "-dSAFER",
            "-dBATCH",
            "-dNOPAUSE",
            "-q",
            "-sDEVICE=txtwrite",
            "-sOutputFile=-",
            str(path),
        ]
    return None


def _read_postscript_text(path: Path) -> str:
    max_file_bytes = _env_int(
        "INGEST_MAX_PS_FILE_BYTES",
        128 * 1024 * 1024,
        minimum=1,
        maximum=2 * 1024 * 1024 * 1024,
    )
    if path.stat().st_size > max_file_bytes:
        raise IngestExtractionError(
            f"PostScript source exceeds INGEST_MAX_PS_FILE_BYTES ({max_file_bytes}): {path}"
        )
    extractor = os.getenv("PS_TEXT_EXTRACTOR", "auto").strip().lower()
    if extractor not in {"auto", "internal", "ghostscript", "ps2ascii"}:
        raise ValueError(f"Unknown PS_TEXT_EXTRACTOR={extractor}")
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise IngestExtractionError(
            f"failed to read PostScript source {path}: {error}"
        ) from error
    if extractor != "internal":
        command = _postscript_command(path, extractor)
        if command:
            timeout = _env_int("PS_EXTRACT_TIMEOUT_SECONDS", 60, minimum=1, maximum=3600)
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                completed = subprocess.run(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout,
                    check=False,
                    creationflags=creation_flags,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                if extractor != "auto":
                    raise IngestExtractionError(
                        f"PostScript extractor failed for {path}: {error}"
                    ) from error
            else:
                max_output = _env_int(
                    "PS_MAX_OUTPUT_BYTES",
                    64 * 1024 * 1024,
                    minimum=1,
                    maximum=512 * 1024 * 1024,
                )
                if len(completed.stdout) > max_output:
                    raise IngestExtractionError(
                        f"PostScript output exceeds PS_MAX_OUTPUT_BYTES={max_output}: {path}"
                    )
                if completed.returncode == 0 and completed.stdout:
                    extracted = _decode_postscript_output(completed.stdout)
                    if extracted:
                        return extracted
                elif extractor != "auto":
                    detail = completed.stderr[:500].decode("utf-8", errors="replace")
                    raise IngestExtractionError(
                        f"PostScript extractor returned {completed.returncode}: {detail}"
                    )
        elif extractor != "auto":
            raise IngestExtractionError(
                f"requested PostScript extractor is not installed: {extractor}"
            )
    return _read_postscript_text_internal(raw)


def _read_svg_text(path: Path) -> str:
    """
    SVG is XML; extract all text nodes.
    """
    try:
        tree = ET.parse(path)
        root = tree.getroot()
        texts: List[str] = []
        for elem in root.iter():
            if elem.text and elem.text.strip():
                texts.append(elem.text.strip())
        return "\n".join(texts)
    except Exception as e:
        print(f"[INGEST] WARNING: Failed to parse SVG {path}: {e}", file=sys.stderr)
        return ""


def _read_image_text(path: Path) -> str:
    """
    OCR for PNG (and optionally JPG/JPEG if you want).
    Requires Pillow + pytesseract + system tesseract.
    """
    if Image is None or pytesseract is None:
        raise IngestExtractionError(
            f"image OCR unavailable for {path}: Pillow/pytesseract is not installed"
        )
    try:
        max_bytes = _env_int(
            "OCR_MAX_FILE_BYTES",
            100 * 1024 * 1024,
            minimum=1,
            maximum=2 * 1024 * 1024 * 1024,
        )
        if path.stat().st_size > max_bytes:
            raise IngestExtractionError(
                f"image exceeds OCR_MAX_FILE_BYTES ({max_bytes})"
            )
        with Image.open(path) as img:
            width, height = _image_dimensions(img)
            max_pixels = _env_int(
                "OCR_MAX_IMAGE_PIXELS",
                40_000_000,
                minimum=1_000_000,
                maximum=200_000_000,
            )
            if width and height and width * height > max_pixels:
                raise IngestExtractionError(
                    f"image exceeds OCR_MAX_IMAGE_PIXELS ({max_pixels})"
                )
            text = _ocr_image_to_string(img, _ocr_language())
        return _normalize_extracted_text(text)
    except IngestExtractionError:
        raise
    except Exception as e:
        raise IngestExtractionError(f"image OCR failed for {path}: {e}") from e


def _preserve_markdown_front_matter(value: str) -> str:
    """Turn YAML/TOML front matter into searchable document metadata.

    The shared Markdown cleaner intentionally removes front matter. For source
    ingestion, fields such as owner, product, date, and Hebrew titles are often
    answerable facts, so expose them as an ordinary section before cleaning.
    """
    text = _normalize_extracted_text(value)
    lines = text.splitlines()
    if not lines or lines[0].strip() not in {"---", "+++"}:
        return text
    delimiter = lines[0].strip()
    closing = next(
        (index for index in range(1, min(len(lines), 10002)) if lines[index].strip() == delimiter),
        None,
    )
    if closing is None:
        return text
    metadata = "\n".join(lines[1:closing]).strip()
    body = "\n".join(lines[closing + 1:]).strip()
    sections: List[str] = []
    if metadata:
        sections.append("# Document metadata\n\n" + metadata)
    if body:
        sections.append(body)
    return _normalize_extracted_text("\n\n".join(sections))


def _markdown_table_cells(line: str) -> List[str]:
    value = line.strip()
    if value.startswith("|"):
        value = value[1:]
    if value.endswith("|") and not value.endswith("\\|"):
        value = value[:-1]
    return [
        _normalize_extracted_text(cell.replace("\\|", "|")).strip()
        for cell in re.split(r"(?<!\\)\|", value)
    ]


def _expand_markdown_tables(value: str) -> str:
    """Keep every Markdown table row paired with its bilingual headers."""
    lines = (value or "").splitlines()
    output: List[str] = []
    index = 0
    fence_char = ""
    fence_length = 0
    while index < len(lines):
        line = lines[index]
        fence = re.match(r"^[ \t]{0,3}(`{3,}|~{3,})", line)
        if fence:
            marker = fence.group(1)
            if not fence_char:
                fence_char, fence_length = marker[0], len(marker)
            elif marker[0] == fence_char and len(marker) >= fence_length:
                fence_char, fence_length = "", 0
            output.append(line)
            index += 1
            continue
        if fence_char or index + 1 >= len(lines) or "|" not in line:
            output.append(line)
            index += 1
            continue

        header = _markdown_table_cells(line)
        separator = _markdown_table_cells(lines[index + 1])
        valid_separator = (
            len(header) >= 2
            and len(separator) == len(header)
            and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in separator)
        )
        if not valid_separator:
            output.append(line)
            index += 1
            continue

        table_rows = [header]
        cursor = index + 2
        while cursor < len(lines) and lines[cursor].strip() and "|" in lines[cursor]:
            row = _markdown_table_cells(lines[cursor])
            if len(row) < len(header):
                row.extend([""] * (len(header) - len(row)))
            table_rows.append(row[: len(header)])
            cursor += 1
        rendered = _structured_table_records(table_rows, force_header=True)
        if rendered:
            output.extend(rendered.splitlines())
        index = cursor
    return _normalize_extracted_text("\n".join(output))


def _read_bounded_text_source(path: Path) -> str:
    max_bytes = _env_int(
        "INGEST_MAX_TEXT_FILE_BYTES",
        64 * 1024 * 1024,
        minimum=1,
        maximum=2 * 1024 * 1024 * 1024,
    )
    try:
        size = path.stat().st_size
        if size > max_bytes:
            raise IngestExtractionError(
                f"text source exceeds INGEST_MAX_TEXT_FILE_BYTES ({max_bytes}): {path}"
            )
        return _decode_text_bytes(path.read_bytes())
    except IngestExtractionError:
        raise
    except OSError as error:
        raise IngestExtractionError(f"failed to read source {path}: {error}") from error


def _delimited_header_index(
    rows: List[Tuple[int, List[str]]],
    mode: str,
) -> Optional[int]:
    if mode == "absent" or not rows:
        return None
    if mode == "present":
        return 0
    if mode != "auto":
        raise ValueError(
            f"Unknown CSV_HEADER_MODE={mode!r}; expected auto, present, or absent"
        )
    if len(rows) < 2:
        return None
    scan_rows = _env_int("CSV_HEADER_SCAN_ROWS", 10, minimum=1, maximum=100)
    width = max(len(row) for _line, row in rows)
    candidates: List[Tuple[float, int]] = []
    for index, (_line_number, row) in enumerate(rows[: min(scan_rows, len(rows) - 1)]):
        values = [_markdown_cell(row[column]) if column < len(row) else "" for column in range(width)]
        nonempty = [value for value in values if value]
        if len(nonempty) < 2 or len(set(nonempty)) < 2:
            continue
        textual = sum(
            bool(re.search(r"[^\W\d_]", value, flags=re.UNICODE)) for value in nonempty
        )
        textual_ratio = textual / len(nonempty)
        if textual_ratio < 0.60:
            continue
        density = len(nonempty) / max(1, width)
        candidates.append((len(nonempty) * 10.0 + density + textual_ratio, index))
    if not candidates:
        return None
    best_score = max(score for score, _index in candidates)
    return min(index for score, index in candidates if score == best_score)


def _render_delimited_rows(
    rows: List[Tuple[int, List[str]]],
    *,
    delimiter: str,
    header_index: Optional[int],
) -> str:
    if not rows:
        return ""
    width = max(len(row) for _line, row in rows)
    header = rows[header_index][1] if header_index is not None else []
    labels = [
        (
            f"{_markdown_cell(header[column])} ({_xlsx_column_name(column + 1)})"
            if column < len(header) and _markdown_cell(header[column])
            else f"Column {_xlsx_column_name(column + 1)}"
        )
        for column in range(width)
    ]
    delimiter_names = {",": "comma", ";": "semicolon", "\t": "tab", "|": "pipe"}
    output = [
        "# Delimited data",
        f"Delimiter: {delimiter_names.get(delimiter, repr(delimiter))}",
        "Columns: " + "; ".join(labels),
    ]

    if header_index is not None and header_index > 0:
        output.append("## File context")
        context: List[str] = []
        for line_number, row in rows[:header_index]:
            facts = [
                f"- Cell {_xlsx_column_name(column + 1)}: {_markdown_cell(value)}"
                for column, value in enumerate(row)
                if _markdown_cell(value)
            ]
            if facts:
                context.append(
                    f"Delimited context row {line_number}:\n" + "\n".join(facts)
                )
        if context:
            output.append("\n\n".join(context))

    data_start = header_index + 1 if header_index is not None else 0
    data_rows = rows[data_start:]
    if data_rows:
        output.append("## Records")
    records: List[str] = []
    for line_number, row in data_rows:
        facts = [
            f"- {labels[column]}: {_markdown_cell(value)}"
            for column, value in enumerate(row)
            if _markdown_cell(value)
        ]
        if facts:
            records.append(f"Delimited row {line_number}:\n" + "\n".join(facts))
    if records:
        output.append("\n\n".join(records))
    return _normalize_extracted_text("\n\n".join(output))


def _read_delimited_text(path: Path) -> str:
    mode = os.getenv("CSV_HEADER_MODE", "auto").strip().lower()
    if mode not in {"auto", "present", "absent"}:
        raise ValueError(
            f"Unknown CSV_HEADER_MODE={mode!r}; expected auto, present, or absent"
        )
    source = _read_bounded_text_source(path)
    if not source.strip():
        return ""
    sample = source[:65536]
    fallback_delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = fallback_delimiter
        dialect = type(
            "BoundedDialect",
            (csv.Dialect,),
            {
                "delimiter": delimiter,
                "quotechar": '"',
                "escapechar": None,
                "doublequote": True,
                "skipinitialspace": False,
                "lineterminator": "\n",
                "quoting": csv.QUOTE_MINIMAL,
            },
        )

    max_rows = _env_int("CSV_MAX_ROWS", 1_000_000, minimum=1, maximum=10_000_000)
    max_columns = _env_int("CSV_MAX_COLUMNS", 10_000, minimum=1, maximum=100_000)
    max_field_chars = _env_int(
        "CSV_MAX_FIELD_CHARS", 1_000_000, minimum=1, maximum=16_000_000
    )
    previous_field_limit = csv.field_size_limit()
    csv.field_size_limit(max_field_chars)
    rows: List[Tuple[int, List[str]]] = []
    try:
        reader = csv.reader(io.StringIO(source, newline=""), dialect=dialect, strict=True)
        while True:
            start_line = reader.line_num + 1
            try:
                row = next(reader)
            except StopIteration:
                break
            except csv.Error as error:
                raise IngestExtractionError(
                    f"malformed delimited source {path} near line {reader.line_num}: {error}"
                ) from error
            if len(row) > max_columns:
                raise IngestExtractionError(
                    f"delimited source exceeds CSV_MAX_COLUMNS={max_columns}: {path}"
                )
            normalized = [_normalize_extracted_text(value) for value in row]
            if not any(normalized):
                continue
            rows.append((start_line, normalized))
            if len(rows) > max_rows:
                raise IngestExtractionError(
                    f"delimited source exceeds CSV_MAX_ROWS={max_rows}: {path}"
                )
    finally:
        csv.field_size_limit(previous_field_limit)

    header_index = _delimited_header_index(rows, mode)
    return _render_delimited_rows(
        rows,
        delimiter=delimiter,
        header_index=header_index,
    )


def _flatten_log_json(
    value: Any,
    *,
    prefix: str = "",
    depth: int = 0,
    output: Optional[List[Tuple[str, str]]] = None,
) -> List[Tuple[str, str]]:
    fields = output if output is not None else []
    if len(fields) >= 256:
        return fields
    if isinstance(value, dict) and depth < 8:
        for key, child in value.items():
            label = f"{prefix}.{key}" if prefix else str(key)
            _flatten_log_json(child, prefix=label, depth=depth + 1, output=fields)
            if len(fields) >= 256:
                break
    elif isinstance(value, list):
        fields.append((prefix or "value", json.dumps(value, ensure_ascii=False)))
    else:
        fields.append((prefix or "value", str(value)))
    return fields


_LOG_TIMESTAMP_RE = re.compile(
    r"^\s*(?P<timestamp>(?:\d{4}-\d{2}-\d{2}[T ][0-9:.+\-Z]+)|"
    r"(?:\d{2}/\d{2}/\d{4}[ T][0-9:]+)|"
    r"(?:[A-Z][a-z]{2}\s+\d{1,2}\s+[0-9:]{8}))"
)
_LOG_LEVEL_RE = re.compile(
    r"(?<![A-Za-z])(?P<level>TRACE|DEBUG|INFO|NOTICE|WARN|WARNING|ERROR|FATAL|CRITICAL)(?![A-Za-z])",
    flags=re.IGNORECASE,
)


def _render_log_events(events: List[List[Tuple[str, str]]]) -> str:
    if not events:
        return ""
    batch_size = _env_int("LOG_EVENTS_PER_SECTION", 50, minimum=1, maximum=1000)
    output = ["# Log events"]
    for start in range(0, len(events), batch_size):
        batch = events[start:start + batch_size]
        first, last = start + 1, start + len(batch)
        output.append(
            f"## Events {first}-{last}" if first != last else f"## Event {first}"
        )
        rendered: List[str] = []
        for offset, fields in enumerate(batch, start=first):
            facts = [
                f"- {_markdown_cell(label)}: {_markdown_cell(value)}"
                for label, value in fields
                if _markdown_cell(label) and _markdown_cell(value)
            ]
            if facts:
                rendered.append(f"Log event {offset}:\n" + "\n".join(facts))
        if rendered:
            output.append("\n\n".join(rendered))
    return _normalize_extracted_text("\n\n".join(output))


def _read_log_text(path: Path) -> str:
    source = _read_bounded_text_source(path)
    lines = source.splitlines()
    nonempty = [(index + 1, line) for index, line in enumerate(lines) if line.strip()]
    if not nonempty:
        return ""
    max_events = _env_int("LOG_MAX_EVENTS", 1_000_000, minimum=1, maximum=10_000_000)

    json_events: List[List[Tuple[str, str]]] = []
    jsonl = True
    for line_number, line in nonempty:
        try:
            parsed = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            jsonl = False
            break
        if not isinstance(parsed, dict):
            jsonl = False
            break
        fields = [("Source line", str(line_number))] + _flatten_log_json(parsed)
        json_events.append(fields)
        if len(json_events) > max_events:
            raise IngestExtractionError(
                f"log source exceeds LOG_MAX_EVENTS={max_events}: {path}"
            )
    if jsonl:
        return _render_log_events(json_events)

    grouped: List[Tuple[int, List[str]]] = []
    for line_number, line in nonempty:
        starts_event = bool(_LOG_TIMESTAMP_RE.search(line) or _LOG_LEVEL_RE.search(line[:160]))
        continuation = bool(
            line[:1].isspace()
            or re.match(r"\s*(?:at\s|Caused by:|Traceback\s|File\s+\")", line)
        )
        if grouped and (continuation or not starts_event):
            # Unprefixed, unindented lines are usually separate records unless
            # the previous record visibly introduced a multiline detail.
            previous = grouped[-1][1]
            if continuation or previous[-1].rstrip().endswith((":", "\\")):
                previous.append(line)
                continue
        grouped.append((line_number, [line]))
        if len(grouped) > max_events:
            raise IngestExtractionError(
                f"log source exceeds LOG_MAX_EVENTS={max_events}: {path}"
            )

    events: List[List[Tuple[str, str]]] = []
    for line_number, event_lines in grouped:
        first_line = event_lines[0]
        fields: List[Tuple[str, str]] = [("Source line", str(line_number))]
        timestamp = _LOG_TIMESTAMP_RE.search(first_line)
        level = _LOG_LEVEL_RE.search(first_line[:240])
        if timestamp:
            fields.append(("Timestamp", timestamp.group("timestamp")))
        if level:
            fields.append(("Level", level.group("level").upper()))
        for match in re.finditer(
            r"(?<!\S)(?P<key>[\w.\-\u0590-\u05ff]{1,80})=(?P<value>\"[^\"]*\"|'[^']*'|[^\s]+)",
            first_line,
            flags=re.UNICODE,
        ):
            fields.append((match.group("key"), match.group("value").strip("\"'")))
        fields.append(("Message", "\n".join(event_lines)))
        events.append(fields)
    return _render_log_events(events)


def _script_code_fence(value: str) -> str:
    longest = max((len(match.group(0)) for match in re.finditer(r"`+", value)), default=0)
    return "`" * max(3, longest + 1)


def _script_function_starts(value: str, suffix: str) -> List[Tuple[int, str]]:
    starts: List[Tuple[int, str]] = []
    offset = 0
    in_ps_block_comment = False
    if suffix == ".ps1":
        pattern = re.compile(
            r"^\s*(?:function|filter|workflow)\s+"
            r"(?:(?:global|script|local|private):)?(?P<name>[A-Za-z_][\w.:-]*)",
            flags=re.IGNORECASE,
        )
    else:
        pattern = re.compile(
            r"^\s*(?:function\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)\s*"
            r"(?:\(\s*\))?\s*\{"
        )
    for line in value.splitlines(keepends=True):
        stripped = line.lstrip()
        if suffix == ".ps1":
            if in_ps_block_comment:
                if "#>" in line:
                    in_ps_block_comment = False
                offset += len(line)
                continue
            if stripped.startswith("<#"):
                if "#>" not in stripped[2:]:
                    in_ps_block_comment = True
                offset += len(line)
                continue
        if not stripped.startswith("#"):
            match = pattern.match(line)
            if match:
                starts.append((offset, match.group("name")))
        offset += len(line)
    return starts


def _read_script_text(path: Path) -> str:
    source = _read_bounded_text_source(path)
    if not source.strip():
        return ""
    suffix = path.suffix.lower()
    language = "powershell" if suffix == ".ps1" else "bash"
    display_language = "PowerShell" if suffix == ".ps1" else "Shell"
    starts = _script_function_starts(source, suffix)
    output = [f"# Script: {path.name}", f"Script language: {display_language}"]

    def append_code_section(title: str, code: str, function_name: str = "") -> None:
        cleaned_code = code.strip()
        if not cleaned_code:
            return
        output.append(f"## {title}")
        if function_name:
            output.append(f"Function name: {function_name}")
        fence = _script_code_fence(cleaned_code)
        output.append(f"{fence}{language}\n{cleaned_code}\n{fence}")

    if not starts:
        append_code_section("Script body", source)
    else:
        append_code_section("Script preamble", source[: starts[0][0]])
        for index, (start, name) in enumerate(starts):
            end = starts[index + 1][0] if index + 1 < len(starts) else len(source)
            append_code_section(f"Function: {name}", source[start:end], name)
    return _normalize_extracted_text("\n\n".join(output))


def _read_file_text(path: Path, *, pdf_native_only: bool = False) -> str:
    suffix = path.suffix.lower()

    # Special formats first
    if suffix == ".pdf":
        if pdf_native_only:
            return _read_pdf_text(path, native_only=True)
        return _read_pdf_text(path)
    if suffix == ".docx":
        return _read_docx_text(path)
    if suffix == ".xlsx":
        return _read_xlsx_text(path)
    if suffix == ".pptx":
        return _read_pptx_text(path)
    if suffix in {".html", ".htm"}:
        return _read_html_text(path)
    if suffix in {".csv", ".tsv"}:
        return _read_delimited_text(path)
    if suffix == ".log":
        return _read_log_text(path)
    if suffix in {".ps1", ".sh"}:
        return _read_script_text(path)
    if suffix == ".ps":
        return _read_postscript_text(path)
    if suffix == ".svg":
        return _read_svg_text(path)
    if suffix in {".png", ".jpg", ".jpeg", ".webp"}:
        return _read_image_text(path)

    # Default: treat as text file (this covers .md, .txt, .yaml, .go, and
    # files with no extension as long as they are actually text).
    decoded = _read_bounded_text_source(path)
    if suffix in {".md", ".mdx"}:
        return _expand_markdown_tables(_preserve_markdown_front_matter(decoded))
    return decoded



def _pdf_file_page_manifest(path: Path) -> Dict[str, Any]:
    """Describe every physical file page, including pages with little/no text.

    Native text/image counts describe extractability, not the semantic content
    of charts. The source file remains the authority for embedded graphics.
    """
    pages: List[Dict[str, Any]] = []
    try:
        if fitz is not None:
            with fitz.open(str(path)) as document:
                for index, page in enumerate(document):
                    pages.append({
                        "page": index + 1,
                        "native_text_chars": len(page.get_text("text") or ""),
                        "embedded_image_count": len(page.get_images()),
                    })
        elif pypdf is not None:
            reader = pypdf.PdfReader(str(path))
            for index, page in enumerate(reader.pages):
                pages.append({
                    "page": index + 1,
                    "native_text_chars": len(page.extract_text() or ""),
                    "embedded_image_count": len(page.images),
                })
        else:
            raise RuntimeError("no PDF page manifest reader is available")
    except Exception as error:
        raise IngestExtractionError(
            f"PDF file-page manifest failed for {path}: {error}"
        ) from error
    if not pages:
        raise IngestExtractionError(f"PDF contains no file pages: {path}")
    return {
        "pdf_page_manifest_revision": "pdf-file-pages-v1",
        "source_page_count": len(pages),
        "pdf_page_manifest_json": json.dumps(pages, separators=(",", ":")),
        "pdf_embedded_images_semantically_extracted": False,
        "pdf_text_only_evidence_limitations": (
            "Embedded figures and chart images are preserved in the original "
            "PDF but their values are not semantically extracted by this text "
            "pipeline. Narrative text and captions are separate evidence."
        ),
    }


def _normalize_rel_path(p: Path, root: Path) -> str:
    """
    Return a stable, normalized, POSIX-style path relative to docs root.
    Example: ceo/index.md
    """
    rel = unicodedata.normalize("NFC", p.relative_to(root).as_posix())
    rel = re.sub(r"/{2,}", "/", rel)
    rel = re.sub(r"^\./", "", rel)
    return rel


def _safe_meta(meta: Dict, source_rel: str) -> Dict:
    """
    Clean metadata so it's JSON-serializable and attach a canonical 'source'
    and 'filename'. 'source' MUST stay as the full relative path (e.g. ceo/index.md).
    """
    cleaned: Dict[str, object] = {}
    for k, v in (meta or {}).items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            cleaned[k] = v
        else:
            cleaned[k] = json.dumps(v, ensure_ascii=False)

    # Attach canonical path-based source
    cleaned["source"] = source_rel
    # Reserved identity fields always come from the filesystem, never from a
    # splitter/plugin metadata payload.
    cleaned["filename"] = Path(source_rel).name
    cleaned["path"] = source_rel
    cleaned["source_extension"] = Path(source_rel).suffix.lower()
    cleaned["document_format"] = (
        Path(source_rel).suffix.lower().lstrip(".") or "text"
    )
    return cleaned


def _chunks_for_file_with_trace(
    path: Path,
    docs_dir: Path,
    *,
    pdf_native_only: bool = False,
) -> Tuple[List[str], List[Dict], CanonicalText]:
    """
    Read and clean a file, then split into chunks with breadcrumbs.
    The splitter is expected to return (chunks, metas) where each meta is
    chunk-local (titles, headings, etc.). We attach file-level metadata later.
    """
    raw = (
        _read_file_text(path, pdf_native_only=True)
        if pdf_native_only else _read_file_text(path)
    )
    layout_document = None
    if raw.strip() and path.suffix.lower() == ".pdf" and not pdf_native_only and os.getenv("PDF_EXTRACTOR", "auto").strip().lower() == "auto":
        try:
            layout_document = reconstruct_pdf_layout(path, raw, fitz)
        except Exception as error:
            raise IngestExtractionError(f"positioned PDF layout extraction failed for {path}: {error}") from error
    extraction_suffixes = {
        ".pdf", ".ps", ".png", ".jpg", ".jpeg", ".webp",
    }
    extracted_layout = path.suffix.lower() in extraction_suffixes
    reconstruction_trace = (
        _PDF_RECONSTRUCTION_TRACE_CACHE.get(str(path.resolve()))
        if path.suffix.lower() == ".pdf"
        else None
    )
    if layout_document is not None:
        canonical = layout_document.canonical
        reconstruction_metadata = {
            "pdf_layout_revision": PDF_LAYOUT_REVISION,
            "pdf_layout_extractor": "pymupdf-positioned-words",
            "pdf_layout_raw_representation": "ordered-coordinate-flows-and-legacy-pages",
        }
    elif reconstruction_trace and reconstruction_trace[1] == raw:
        (
            extractor_raw,
            reconstructed,
            reconstructed_origins,
            reconstruction_edits,
            reconstruction_metadata,
        ) = reconstruction_trace
        canonical = canonicalize_reconstructed_source(
            extractor_raw,
            reconstructed,
            reconstructed_origins,
            reconstruction_edits,
            repair_visual_bidi=True,
            normalize_pdf_spacing=True,
            join_pdf_line_wraps=True,
        )
    else:
        reconstruction_metadata = {}
        canonical = canonicalize_source_text(
            raw,
            repair_visual_bidi=extracted_layout,
            normalize_pdf_spacing=extracted_layout,
            join_pdf_line_wraps=extracted_layout,
        )
    raw = canonical.text
    max_extracted_chars = _env_int(
        "INGEST_MAX_EXTRACTED_TEXT_CHARS",
        64 * 1024 * 1024,
        minimum=1,
        maximum=2 * 1024 * 1024 * 1024,
    )
    if len(raw) > max_extracted_chars:
        raise IngestExtractionError(
            f"extracted text exceeds INGEST_MAX_EXTRACTED_TEXT_CHARS="
            f"{max_extracted_chars}: {path}"
        )
    if (
        path.suffix.lower() in _FAIL_CLOSED_EXTRACT_SUFFIXES
        and path.stat().st_size > 0
        and not raw.strip()
    ):
        raise IngestNoTextError(
            f"non-empty {path.suffix.lower()} source produced no extractable text: {path}"
        )
    cleaned = clean_markdown(raw)
    # Short answers such as Hebrew "כן"/"לא" are complete facts. Only truly
    # empty content is an empty generation.
    if not cleaned or not cleaned.strip():
        return [], [], canonical

    # NOTE: splitter signature is (text, file_title)
    source_rel = _normalize_rel_path(path, docs_dir)
    if layout_document is None:
        chunks, metas = split_with_breadcrumbs(cleaned, file_title=source_rel)
    else:
        chunks, metas = [], []
        for group in paragraph_packing_groups(layout_document):
            # Packing retains each literal source paragraph and no ownership
            # label is transferred between independently reconstructed flows.
            members = group["members"]
            body = canonical.text[group["canonical_start"]:group["canonical_end"]]
            page_title = f"Page {group['page']}"
            heading = members[0]["heading"] if len(members) == 1 else ""
            section_title = page_title + (" / " + heading if heading else "")
            parts, local_metas = split_with_breadcrumbs(
                f"# {section_title}\n\n{body}", file_title=source_rel,
            )
            provenance = packing_group_metadata(group, canonical)
            for part, meta in zip(parts, local_metas):
                meta.update(provenance)
                chunks.append(part)
                metas.append(meta)
        for index, meta in enumerate(metas):
            flow_id = meta.get("pdf_flow_id")
            meta["previous_same_pdf_flow"] = bool(flow_id) and index > 0 and metas[index - 1].get("pdf_flow_id") == flow_id
            meta["next_same_pdf_flow"] = bool(flow_id) and index + 1 < len(metas) and metas[index + 1].get("pdf_flow_id") == flow_id
            meta["previous_same_pdf_packing_group"] = index > 0 and metas[index - 1]["pdf_packing_group_id"] == meta["pdf_packing_group_id"]
            meta["next_same_pdf_packing_group"] = index + 1 < len(metas) and metas[index + 1]["pdf_packing_group_id"] == meta["pdf_packing_group_id"]
    chunks = [_normalize_extracted_text(chunk) for chunk in (chunks or [])]
    metas = list(metas or [])
    if len(chunks) != len(metas):
        raise IngestExtractionError(
            f"splitter returned mismatched chunks/metadatas for {source_rel}: "
            f"{len(chunks)} != {len(metas)}"
        )
    if any(not chunk.strip() for chunk in chunks):
        raise IngestExtractionError(
            f"splitter returned an empty chunk for {source_rel}"
        )
    canonical_metadata = canonical.metadata()
    pdf_manifest = (
        _pdf_file_page_manifest(path) if path.suffix.lower() == ".pdf" else {}
    )
    manifest_pages = {
        entry["page"]: entry
        for entry in json.loads(pdf_manifest.get("pdf_page_manifest_json", "[]"))
    }
    for chunk, meta in zip(chunks, metas):
        meta.update(pdf_manifest)
        meta["extraction_method"] = _extraction_method(path, pdf_native_only=pdf_native_only)
        if path.suffix.lower() == ".docx" and meta["extraction_method"].startswith("docx-embedded-"):
            image_match = re.match(r"Embedded image ([1-9][0-9]*): (.+)", str(meta.get("section") or ""))
            if image_match:
                meta["embedded_image_index"] = int(image_match.group(1))
                meta["embedded_image_name"] = image_match.group(2)
        if path.suffix.lower() == ".pptx":
            slide_match = re.search(r"(?:^| / )Slide ([1-9][0-9]*)(?:\D|$)", str(meta.get("section") or ""))
            if slide_match:
                meta["slide"] = int(slide_match.group(1))
        if pdf_manifest:
            meta["pdf_layout_revision"] = PDF_LAYOUT_REVISION
            meta["pdf_native_row_association_revision"] = PDF_NATIVE_ROW_ASSOCIATION_REVISION
            meta["pdf_layout_evaluated"] = not pdf_native_only and os.getenv("PDF_EXTRACTOR", "auto").strip().lower() == "auto"
            meta["pdf_layout_mode"] = (
                "positioned-coordinate-flows" if layout_document is not None
                else "single-column-compatible" if meta["pdf_layout_evaluated"]
                else "explicit-extractor"
            )
            if pdf_native_only:
                meta["pdf_native_only_revision"] = PDF_NATIVE_ONLY_REVISION
                meta["pdf_ocr_attempted"] = False
                meta["pdf_text_only_evidence_limitations"] = (
                    "Only existing native PDF text is indexed. OCR, drawing "
                    "geometry, and graphical relationships were not extracted."
                )
        page = location_metadata(meta).get("page")
        if page is not None:
            if pdf_manifest and page not in manifest_pages:
                raise IngestExtractionError(
                    f"PDF chunk page is outside the file-page manifest: {path}"
                )
            meta["page"] = page
            if page in manifest_pages:
                meta["pdf_page_embedded_image_count"] = manifest_pages[page]["embedded_image_count"]
        meta.update(canonical_metadata)
        meta.update(reconstruction_metadata)
        spans = sentence_spans(chunk)
        meta["source_structure_revision"] = SOURCE_STRUCTURE_REVISION
        meta["proposition_schema_revision"] = PROPOSITION_SCHEMA_REVISION
        meta["paragraph_count"] = len([
            block
            for block in re.split(r"\n[ \t]*\n+", chunk)
            if block.strip()
        ])
        meta["sentence_count"] = len(spans)
        meta["sentence_start"] = bool(spans) and spans[0].sentence_start
        meta["sentence_end"] = bool(spans) and spans[-1].sentence_end
        meta["sentence_spans_json"] = json.dumps(
            [
                {
                    "start": span.start,
                    "end": span.end,
                    "sentence_start": span.sentence_start,
                    "sentence_end": span.sentence_end,
                }
                for span in spans
            ],
            separators=(",", ":"),
        )
        meta["has_bullet_context"] = bool(
            re.search(r"(?m)^\s*[•▪◦*-]\s+", chunk)
        )
        meta["has_table_context"] = bool(
            re.search(r"(?m)^\s*\|.*\|\s*$", chunk)
        )
    return chunks, metas, canonical


_CANONICAL_TRACE_CACHE: Dict[str, CanonicalText] = {}


def _chunks_for_file(
    path: Path, docs_dir: Path, *, pdf_native_only: bool = False,
) -> Tuple[List[str], List[Dict]]:
    """Backward-compatible chunk helper; main ingestion retains the trace."""
    chunks, metas, canonical = (
        _chunks_for_file_with_trace(path, docs_dir, pdf_native_only=True)
        if pdf_native_only else _chunks_for_file_with_trace(path, docs_dir)
    )
    _CANONICAL_TRACE_CACHE[str(path.resolve())] = canonical
    return chunks, metas


def _iter_files(docs_dir: Path, include_exts: Tuple[str, ...]) -> Iterable[Path]:
    # Deterministic traversal keeps logs, dry-runs, and failure recovery
    # reproducible across operating systems.
    for p in sorted(docs_dir.rglob("*"), key=lambda value: value.as_posix().casefold()):
        if not p.is_file():
            continue

        suffix = p.suffix.lower()

        # Normal case: extension matches
        if suffix in include_exts:
            yield p
            continue

        # Special case: include files with NO extension if "" is in include_exts
        if suffix == "" and "" in include_exts:
            yield p


def _source_matches_include(source_rel: str, include_exts: Tuple[str, ...]) -> bool:
    suffix = Path(source_rel).suffix.lower()
    return suffix in include_exts or (not suffix and "" in include_exts)


def _selected_source_files(
    docs_dir: Path, include_exts: Tuple[str, ...], requested_sources: Iterable[str],
) -> List[Path]:
    """Resolve exact source identities without scanning unrelated documents."""
    selected: List[Path] = []
    seen = set()
    for source in requested_sources:
        source = str(source)
        parts = source.split("/")
        if (
            not source or "\\" in source or "\x00" in source
            or source.startswith("/") or re.match(r"^[A-Za-z]:", source)
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError("--source requires an exact POSIX path relative to --docs-dir")
        candidate = docs_dir
        for part in parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ValueError(f"--source must not traverse a symlink: {source}")
        if not candidate.is_file():
            raise ValueError(f"--source is missing or not a regular file: {source}")
        candidate.resolve(strict=True).relative_to(docs_dir)
        identity = _normalize_rel_path(candidate, docs_dir)
        if identity in seen:
            raise ValueError(f"duplicate --source identity: {source}")
        if not _source_matches_include(identity, include_exts):
            raise ValueError(f"--source extension is excluded by --include: {source}")
        seen.add(identity)
        selected.append(candidate)
    return selected


def _get_or_create_collection(client: PersistentClient, name: str) -> Collection:
    get_or_create = getattr(client, "get_or_create_collection", None)
    if callable(get_or_create):
        # Modern Chroma exposes an atomic API. Its failures must propagate;
        # treating a connection/permission error as "not found" is unsafe.
        return get_or_create(name=name, metadata={"hnsw:space": "cosine"})
    try:
        return client.get_collection(name=name)
    except Exception:
        return client.create_collection(name=name, metadata={"hnsw:space": "cosine"})


def _collection_for_tenant(
    client: PersistentClient,
    tenant_id: Optional[str],
) -> Tuple[Collection, Optional[Dict]]:
    """
    Either use one collection per tenant (TENANT_STRATEGY=collection)
    or a shared collection with tenant_id in metadata (TENANT_STRATEGY=metadata).
    """
    strategy = _validated_tenant_strategy()
    if strategy == "collection":
        if not tenant_id:
            raise ValueError("tenant_id is required when TENANT_STRATEGY=collection")
        coll_name = f"{BASE_COLLECTION}_{tenant_id}"
        return _get_or_create_collection(client, coll_name), None
    else:
        coll = _get_or_create_collection(client, BASE_COLLECTION)
        where = {"tenant_id": tenant_id} if tenant_id else None
        return coll, where


def _embedder(model_name: str) -> SentenceTransformer:
    model = SentenceTransformer(
        model_name, device=EMBED_DEVICE, model_kwargs=embedding_model_kwargs(EMBED_DEVICE),
    )
    print(
        f"[INGEST] Embedding runtime device={getattr(model, 'device', EMBED_DEVICE)} "
        f"dtype={getattr(model, 'dtype', 'unknown')}"
    )
    # Keep the exact CLI-selected contract available to the batch encoder.
    model._rag_model_name = model_name
    return model


def _coerce_embedding_vector(value: Any, *, expected_dim: Optional[int]) -> List[float]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("embedder returned an empty or non-vector embedding")
    try:
        vector = [float(component) for component in value]
    except (TypeError, ValueError) as e:
        raise ValueError("embedder returned a non-numeric embedding") from e
    if not all(math.isfinite(component) for component in vector):
        raise ValueError("embedder returned NaN or infinite embedding values")
    if not any(abs(component) > 1e-12 for component in vector):
        raise ValueError("embedder returned an all-zero embedding")
    if expected_dim is not None and len(vector) != expected_dim:
        raise ValueError(
            f"inconsistent embedding dimensions: {len(vector)} != {expected_dim}"
        )
    return vector


def _encode_documents(
    docs: List[str],
    embedder: SentenceTransformer,
    *,
    batch_size: Optional[int] = None,
) -> List[List[float]]:
    """Encode all documents and validate every vector before any index write."""
    if not docs:
        return []
    size = batch_size or _env_int(
        "EMBED_BATCH_SIZE", 64, minimum=1, maximum=4096
    )
    model_name = getattr(embedder, "_rag_model_name", EMBED_MODEL)
    if not isinstance(model_name, str) or not model_name.strip():
        model_name = EMBED_MODEL
    vectors: List[List[float]] = []
    expected_dim: Optional[int] = None
    for start in range(0, len(docs), size):
        raw_docs = docs[start:start + size]
        if any(not isinstance(doc, str) or not doc.strip() for doc in raw_docs):
            raise ValueError("refusing to embed an empty or non-string chunk")
        embedding_docs = [
            format_embedding_document(doc, model_name) for doc in raw_docs
        ]
        encoded = embedder.encode(
            embedding_docs,
            batch_size=min(size, len(embedding_docs)),
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        encoded = list(encoded)
        if len(encoded) != len(raw_docs):
            raise ValueError(
                f"embedder output count mismatch: {len(encoded)} != {len(raw_docs)}"
            )
        for raw_vector in encoded:
            vector = _coerce_embedding_vector(
                raw_vector,
                expected_dim=expected_dim,
            )
            if expected_dim is None:
                expected_dim = len(vector)
            vectors.append(vector)
    return vectors


def _validate_upsert_payload(docs: List[str], metas: List[Dict], ids: List[str]) -> None:
    if not (len(docs) == len(metas) == len(ids)):
        raise ValueError(
            f"Length mismatch: docs={len(docs)}, metas={len(metas)}, ids={len(ids)}"
        )
    if len(set(ids)) != len(ids):
        raise ValueError("refusing to upsert duplicate chunk IDs")
    for index, (doc, meta, record_id) in enumerate(zip(docs, metas, ids)):
        if not isinstance(doc, str) or not doc.strip():
            raise ValueError(f"chunk {index} is empty or not text")
        if not isinstance(meta, dict):
            raise ValueError(f"metadata {index} is not a dictionary")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"chunk ID {index} is empty or not text")


def _upsert_precomputed(
    coll: Collection,
    docs: List[str],
    metas: List[Dict],
    ids: List[str],
    embeddings: List[List[float]],
) -> None:
    _validate_upsert_payload(docs, metas, ids)
    if len(embeddings) != len(docs):
        raise ValueError(
            f"embedding count mismatch: {len(embeddings)} != {len(docs)}"
        )
    coll.upsert(
        documents=docs,
        metadatas=metas,
        ids=ids,
        embeddings=embeddings,
    )


def add_embeddings_and_upsert(
    coll: Collection,
    docs: List[str],
    metas: List[Dict],
    ids: List[str],
    embedder: SentenceTransformer,
):
    """
    Encode and upsert a batch of chunks.

    IMPORTANT: We DO NOT touch 'source' here anymore. The caller must have
    already attached the correct path-based 'source' (e.g. 'ceo/index.md')
    when building metas. This keeps explicit path queries in sync with ingest.
    """
    if not docs:
        return

    _validate_upsert_payload(docs, metas, ids)
    vecs = _encode_documents(docs, embedder)
    _upsert_precomputed(coll, docs, metas, ids, vecs)


def _chunk_id(
    tenant_id: str,
    source_rel: str,
    chunk_index: int,
    *,
    tenant_strategy: Optional[str] = None,
) -> str:
    """Return a stable chunk ID without cross-tenant collisions.

    Collection-per-tenant deployments keep their historical IDs. Shared
    metadata collections must include the tenant because Chroma IDs are unique
    across the whole collection, not within a metadata filter.
    """
    strategy = _validated_tenant_strategy(tenant_strategy)
    suffix = f"{source_rel}::chunk::{int(chunk_index):06d}"
    if strategy == "metadata":
        # Length-prefixed raw components make this representation injective.
        # The old ``tenant::{tenant}::{source}`` form collided when either
        # component itself contained ``::``. Exact-ID generation checks make
        # all legacy ambiguous IDs stale on the next successful file update.
        tenant = str(tenant_id)
        source = str(source_rel)
        return (
            f"v2::t{len(tenant)}:{tenant}::s{len(source)}:{source}"
            f"::chunk::{int(chunk_index):06d}"
        )
    return suffix


def _tenant_where(
    tenant_id: str,
    *,
    tenant_strategy: Optional[str] = None,
) -> Optional[Dict]:
    strategy = _validated_tenant_strategy(tenant_strategy)
    if strategy == "metadata":
        return {"tenant_id": {"$eq": tenant_id}}
    return None


def _tenant_source_where(
    tenant_id: str,
    source_rel: str,
    *,
    tenant_strategy: Optional[str] = None,
) -> Dict:
    strategy = _validated_tenant_strategy(tenant_strategy)
    if strategy == "metadata":
        return {
            "$and": [
                {"tenant_id": {"$eq": tenant_id}},
                {"source": {"$eq": source_rel}},
            ]
        }
    return {"source": {"$eq": source_rel}}


def _collection_records(coll: Collection, where: Optional[Dict]) -> Tuple[List[str], List[Dict]]:
    """Read IDs and metadata using a bounded caller-supplied tenant filter."""
    kwargs = {"include": ["metadatas"]}
    if where is not None:
        kwargs["where"] = where
    result = coll.get(**kwargs) or {}
    ids = list(result.get("ids") or [])
    metas = list(result.get("metadatas") or [])
    if len(metas) < len(ids):
        metas.extend({} for _ in range(len(ids) - len(metas)))
    return ids, metas


def _indexed_sources(
    coll: Collection,
    tenant_id: str,
    *,
    tenant_strategy: Optional[str] = None,
) -> Dict[str, List[Tuple[str, Dict]]]:
    """Load current tenant metadata once for O(1) incremental checks per file."""
    strategy = _validated_tenant_strategy(tenant_strategy)
    ids, metas = _collection_records(
        coll,
        _tenant_where(tenant_id, tenant_strategy=strategy),
    )
    grouped: Dict[str, List[Tuple[str, Dict]]] = {}
    for record_id, raw_meta in zip(ids, metas):
        meta = raw_meta or {}
        source = meta.get("source")
        if not isinstance(source, str) or not source:
            continue
        if not _record_belongs_to_scope(
            meta,
            tenant_id,
            tenant_strategy=strategy,
        ):
            continue
        grouped.setdefault(source, []).append((record_id, meta))
    return grouped


def _source_index_is_current(
    records: Iterable[Tuple[str, Dict]],
    tenant_id: str,
    source_rel: str,
    source_sha256: str,
    embed_model: str,
    *,
    tenant_strategy: Optional[str] = None,
    pipeline_fingerprint: Optional[str] = None,
) -> bool:
    """Accept a cached source only when its complete chunk generation is intact."""
    rows = list(records)
    if not rows:
        return False
    expected_pipeline = pipeline_fingerprint or _pipeline_fingerprint(
        embed_model, source_suffix=Path(source_rel).suffix.lower(),
    )
    expected_count = len(rows)
    indices = set()
    for record_id, meta in rows:
        if not isinstance(meta, dict):
            return False
        if meta.get("source") != source_rel:
            return False
        if meta.get("source_sha256") != source_sha256:
            return False
        if meta.get("embed_model") != embed_model:
            return False
        if meta.get("ingest_schema_version") != INGEST_SCHEMA_VERSION:
            return False
        if not source_metadata_complete(meta, tenant_id):
            return False
        if meta.get("pipeline_fingerprint") != expected_pipeline:
            return False
        chunk_count = meta.get("chunk_count")
        if (
            not isinstance(chunk_count, int)
            or isinstance(chunk_count, bool)
            or chunk_count != expected_count
        ):
            return False
        index = meta.get("chunk_index")
        if not isinstance(index, int) or isinstance(index, bool):
            return False
        if record_id != _chunk_id(
            tenant_id,
            source_rel,
            index,
            tenant_strategy=tenant_strategy,
        ):
            return False
        indices.add(index)
    return indices == set(range(expected_count))


def _source_snapshot(
    coll: Collection,
    tenant_id: str,
    source_rel: str,
    *,
    tenant_strategy: Optional[str] = None,
) -> Dict[str, List[Any]]:
    """Capture a complete source snapshot suitable for rollback."""
    result = coll.get(
        where=_tenant_source_where(
            tenant_id,
            source_rel,
            tenant_strategy=tenant_strategy,
        ),
        include=["documents", "metadatas", "embeddings"],
    ) or {}
    ids = list(result.get("ids") or [])
    if not ids:
        return {"ids": [], "documents": [], "metadatas": [], "embeddings": []}

    documents = list(result.get("documents") or [])
    metadatas = list(result.get("metadatas") or [])
    raw_embeddings = result.get("embeddings")
    if hasattr(raw_embeddings, "tolist"):
        raw_embeddings = raw_embeddings.tolist()
    embeddings = list(raw_embeddings or [])
    if not (len(ids) == len(documents) == len(metadatas) == len(embeddings)):
        raise RuntimeError(
            f"cannot safely update {source_rel}: Chroma did not return a complete rollback snapshot"
        )
    if any(not isinstance(document, str) for document in documents):
        raise RuntimeError(
            f"cannot safely update {source_rel}: previous documents are unavailable for rollback"
        )
    if any(not isinstance(meta, dict) for meta in metadatas):
        raise RuntimeError(
            f"cannot safely update {source_rel}: previous metadata is unavailable for rollback"
        )
    for meta in metadatas:
        if not _record_belongs_to_scope(
            meta,
            tenant_id,
            source_rel=source_rel,
            tenant_strategy=tenant_strategy,
        ):
            raise RuntimeError(
                f"cannot safely update {source_rel}: backend returned an out-of-scope record"
            )
    safe_embeddings: List[List[float]] = []
    expected_dim: Optional[int] = None
    for value in embeddings:
        vector = _coerce_embedding_vector(value, expected_dim=expected_dim)
        if expected_dim is None:
            expected_dim = len(vector)
        safe_embeddings.append(vector)
    return {
        "ids": ids,
        "documents": documents,
        "metadatas": metadatas,
        "embeddings": safe_embeddings,
    }


def _rollback_source_update(
    coll: Collection,
    snapshot: Dict[str, List[Any]],
    attempted_ids: Iterable[str],
) -> None:
    """Restore the exact pre-update source after a partial multi-batch write."""
    old_ids = set(snapshot["ids"])
    newly_created = [record_id for record_id in attempted_ids if record_id not in old_ids]
    errors: List[str] = []
    if newly_created:
        try:
            coll.delete(ids=newly_created)
        except Exception as e:
            errors.append(f"failed to delete staged IDs: {e}")
    if snapshot["ids"]:
        try:
            coll.upsert(
                documents=snapshot["documents"],
                metadatas=snapshot["metadatas"],
                ids=snapshot["ids"],
                embeddings=snapshot["embeddings"],
            )
        except Exception as e:
            errors.append(f"failed to restore previous chunks: {e}")
    if errors:
        raise RuntimeError("; ".join(errors))


def _record_belongs_to_scope(
    meta: Dict,
    tenant_id: str,
    *,
    source_rel: Optional[str] = None,
    tenant_strategy: Optional[str] = None,
) -> bool:
    """Defensively verify backend-filtered records before destructive work."""
    strategy = _validated_tenant_strategy(tenant_strategy)
    meta = meta or {}
    if strategy == "metadata" and meta.get("tenant_id") != tenant_id:
        return False
    if source_rel is not None and meta.get("source") != source_rel:
        return False
    return True


def _remove_stale_source_ids(
    coll: Collection,
    tenant_id: str,
    source_rel: str,
    desired_ids: Iterable[str],
    *,
    dry_run: bool = False,
    tenant_strategy: Optional[str] = None,
) -> List[str]:
    """Delete obsolete chunks only after a replacement file was fully upserted."""
    if dry_run:
        return []

    strategy = _validated_tenant_strategy(tenant_strategy)
    desired = set(desired_ids)
    ids, metas = _collection_records(
        coll,
        _tenant_source_where(
            tenant_id,
            source_rel,
            tenant_strategy=strategy,
        ),
    )
    stale = [
        record_id
        for record_id, meta in zip(ids, metas)
        if record_id not in desired
        and _record_belongs_to_scope(
            meta,
            tenant_id,
            source_rel=source_rel,
            tenant_strategy=strategy,
        )
    ]
    if stale:
        # Delete exact IDs rather than repeating a broad metadata deletion.
        coll.delete(ids=stale)
    return stale


def _upsert_file_and_prune_stale(
    coll: Collection,
    docs: List[str],
    metas: List[Dict],
    ids: List[str],
    embedder: SentenceTransformer,
    tenant_id: str,
    source_rel: str,
    *,
    batch_size: int = 512,
    dry_run: bool = False,
    tenant_strategy: Optional[str] = None,
) -> List[str]:
    """Upsert one complete file, then remove only that file's stale IDs.

    Cleanup deliberately occurs after every batch succeeds. If parsing,
    embedding, or an upsert fails, the exception propagates and no stale IDs
    are deleted.
    """
    if dry_run:
        return []
    _validate_upsert_payload(docs, metas, ids)

    # Embedding is the most failure-prone and expensive step. Complete and
    # validate it before touching the last known-good collection state.
    vectors = _encode_documents(docs, embedder)
    snapshot = _source_snapshot(
        coll,
        tenant_id,
        source_rel,
        tenant_strategy=tenant_strategy,
    )
    if snapshot["embeddings"] and vectors:
        old_dim = len(snapshot["embeddings"][0])
        new_dim = len(vectors[0])
        if old_dim != new_dim:
            raise ValueError(
                f"embedding dimension changed for {source_rel}: {old_dim} -> {new_dim}; "
                "use a new collection/CHROMA path for an embedding-dimension migration"
            )

    size = max(1, int(batch_size))
    write_attempted = False
    cleanup_started = False
    try:
        for start in range(0, len(docs), size):
            end = start + size
            # A storage/backend exception does not prove that the write was
            # rejected before commit (for example, a lost response after the
            # server committed). Mark the attempt before entering Chroma so
            # even the first raised call is rolled back conservatively.
            write_attempted = True
            _upsert_precomputed(
                coll,
                docs[start:end],
                metas[start:end],
                ids[start:end],
                vectors[start:end],
            )
        cleanup_started = True
        removed = _remove_stale_source_ids(
            coll,
            tenant_id,
            source_rel,
            ids,
            tenant_strategy=tenant_strategy,
        )
        return removed
    except Exception as original_error:
        if write_attempted or cleanup_started:
            try:
                _rollback_source_update(coll, snapshot, ids)
            except Exception as rollback_error:
                raise RuntimeError(
                    f"source update failed for {source_rel}: {original_error}; "
                    f"rollback also failed: {rollback_error}"
                ) from original_error
        raise


def _sync_deleted_sources(
    coll: Collection,
    tenant_id: str,
    current_sources: Iterable[str],
    *,
    dry_run: bool = False,
    tenant_strategy: Optional[str] = None,
) -> List[str]:
    """Remove chunks for deleted files, scoped to one successfully indexed tenant."""
    if dry_run:
        return []

    strategy = _validated_tenant_strategy(tenant_strategy)
    current = set(current_sources)
    ids, metas = _collection_records(
        coll,
        _tenant_where(tenant_id, tenant_strategy=strategy),
    )
    stale = []
    for record_id, meta in zip(ids, metas):
        meta = meta or {}
        source = meta.get("source")
        # Unknown/legacy records without a source are retained: sync must be
        # conservative rather than guessing about destructive ownership.
        if not source or source in current:
            continue
        if not _record_belongs_to_scope(
            meta,
            tenant_id,
            tenant_strategy=strategy,
        ):
            continue
        stale.append(record_id)

    if stale:
        coll.delete(ids=stale)
    return stale


def _clear_tenant_data(
    client: PersistentClient,
    coll: Collection,
    tenant_id: str,
    *,
    dry_run: bool = False,
    tenant_strategy: Optional[str] = None,
) -> Collection:
    """Clear one tenant and fail closed on any scoped-delete error."""
    if dry_run:
        return coll

    strategy = _validated_tenant_strategy(tenant_strategy)
    if strategy == "collection":
        # This collection belongs only to this tenant.
        collection_name = coll.name
        client.delete_collection(name=collection_name)
        return _get_or_create_collection(client, collection_name)

    # Never fall back to deleting the shared collection. Any failure here is
    # allowed to propagate so operators can repair the scoped operation safely.
    coll.delete(where=_tenant_where(tenant_id, tenant_strategy=strategy))
    return coll


def _clear_except_ids(
    coll: Collection,
    tenant_id: str,
    desired_ids: Iterable[str],
    *,
    dry_run: bool = False,
    tenant_strategy: Optional[str] = None,
) -> List[str]:
    """Finish a safe clear by deleting only records absent from the new build.

    Unlike deleting the collection before parsing, this reconciliation runs
    only after every included file has been extracted, embedded, and upserted.
    A mid-run failure therefore leaves complete per-file generations available.
    """
    if dry_run:
        return []
    strategy = _validated_tenant_strategy(tenant_strategy)
    desired = set(desired_ids)
    ids, metas = _collection_records(
        coll,
        _tenant_where(tenant_id, tenant_strategy=strategy),
    )
    stale = [
        record_id
        for record_id, meta in zip(ids, metas)
        if record_id not in desired
        and _record_belongs_to_scope(
            meta,
            tenant_id,
            tenant_strategy=strategy,
        )
    ]
    if stale:
        coll.delete(ids=stale)
    return stale


# ---------- CLI ----------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAG: ingest documents into Chroma.")
    p.add_argument(
        "--docs-dir",
        type=str,
        default=os.getenv("DOCS_PATH", "data/docs"),
        help="Root folder of docs to ingest (default: env DOCS_PATH or data/docs).",
    )
    p.add_argument(
        "--tenant",
        type=str,
        required=True,
        help="Tenant identifier chosen by you (e.g., my-tenant).",
    )
    p.add_argument(
        "--embedder",
        type=str,
        default=EMBED_MODEL,
        help=f"SentenceTransformer name (default: EMBED_MODEL env or {EMBED_MODEL}).",
    )
    p.add_argument(
        "--include",
        type=str,
        nargs="*",
        default=list(DEFAULT_INCLUDE),
        help=f"File extensions to include (default: {DEFAULT_INCLUDE}).",
    )
    p.add_argument(
        "--source", action="append", metavar="RELATIVE_PATH",
        help="Process only this exact source-relative path; repeat for multiple files. Incompatible with --clear/--sync.",
    )
    p.add_argument(
        "--pdf-native-only", action="store_true",
        help="Index existing native PDF text without OCR or rasterization. Requires explicit --source selection; preserves the default PDF pipeline for other imports.",
    )
    p.add_argument(
        "--clear",
        action="store_true",
        help=(
            "Safely rebuild this tenant and remove obsolete records only after "
            "all current files were indexed successfully."
        ),
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and split, but do not write to Chroma.",
    )
    p.add_argument(
        "--sync",
        action="store_true",
        help=(
            "After every current file is indexed successfully, remove this "
            "tenant's indexed chunks for files no longer present under --docs-dir."
        ),
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-extract and re-embed unchanged files instead of using content hashes.",
    )
    p.add_argument(
        "--allow-empty-sync",
        action="store_true",
        help=(
            "Allow --clear/--sync to delete indexed sources when no included "
            "files were found. Without this explicit opt-in, an empty/unmounted "
            "docs folder fails closed."
        ),
    )
    p.add_argument(
        "--continue-on-error", action="store_true",
        help="Preserve per-source failures, continue remaining files, and exit 2 on a partial import.",
    )
    p.add_argument(
        "--report-json", type=str,
        help="Write an atomic incremental document-status report to this external path.",
    )
    return p


def _write_ingest_report(path: Optional[Path], payload: Dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def main():
    started_at = time.perf_counter()
    args = build_arg_parser().parse_args()
    docs_dir = Path(args.docs_dir).resolve()
    if not docs_dir.is_dir():
        print(f"[INGEST] Docs dir is missing or not a directory: {docs_dir}", file=sys.stderr)
        sys.exit(1)

    tenant_id = validate_tenant_id(args.tenant)
    tenant_strategy = _validated_tenant_strategy()

        # Support a special token "*" to mean "files with no extension"
    include_exts_list = []
    for ext in args.include:
        if ext == "*" or ext == "<noext>":
            # Represent "no extension" internally as empty string
            include_exts_list.append("")
        else:
            include_exts_list.append(
                (ext if ext.startswith(".") else f".{ext}").lower()
            )
    include_exts = tuple(dict.fromkeys(include_exts_list))

    requested_sources = list(getattr(args, "source", None) or [])
    pdf_native_only = bool(getattr(args, "pdf_native_only", False))
    if requested_sources and (args.clear or args.sync):
        raise ValueError("--source cannot be combined with --clear or --sync")
    if pdf_native_only and not requested_sources:
        raise ValueError("--pdf-native-only requires explicit --source selection")
    source_files = (
        _selected_source_files(docs_dir, include_exts, requested_sources)
        if requested_sources else list(_iter_files(docs_dir, include_exts))
    )
    allow_empty_sync = bool(getattr(args, "allow_empty_sync", False))
    if (
        not source_files
        and (bool(args.clear) or bool(args.sync))
        and not bool(args.dry_run)
        and not allow_empty_sync
    ):
        raise RuntimeError(
            "--clear/--sync found no included source files; refusing to delete "
            "the tenant index. Verify the docs mount/include filter or pass "
            "--allow-empty-sync explicitly."
        )


    print(f"[INGEST] CHROMA_PATH={CHROMA_PATH}")
    print(f"[INGEST] RAG_COLLECTION={BASE_COLLECTION}")
    print(f"[INGEST] TENANT_STRATEGY={tenant_strategy}")
    print(f"[INGEST] TENANT={tenant_id}")
    print(f"[INGEST] EMBEDDER={args.embedder}")
    print(f"[INGEST] DOCS_DIR={docs_dir}")
    print(f"[INGEST] INCLUDE_EXTS={include_exts}")

    client = None
    coll = None
    embedder = None
    if args.dry_run:
        print("[INGEST] Dry run: Chroma will not be opened or modified.")
    else:
        client = PersistentClient(path=CHROMA_PATH)
        coll, _ = _collection_for_tenant(client, tenant_id)

        if args.clear:
            print(
                f"[INGEST] Safe rebuild for tenant={tenant_id}; obsolete "
                "records will be removed only after all files succeed."
            )

        embedder = _embedder(args.embedder)

    total_files = 0
    batch_size = _env_int(
        "CHROMA_BATCH_SIZE", 512, minimum=1, maximum=5000
    )
    hash_workers = _env_int(
        "INGEST_HASH_WORKERS", min(4, os.cpu_count() or 1), minimum=1, maximum=16
    )
    max_chunks_per_file = _env_int(
        "INGEST_MAX_CHUNKS_PER_FILE", 20000, minimum=1, maximum=1000000
    )
    def fingerprint_or_error(path: Path):
        try:
            return _source_fingerprint(path)
        except Exception as error:
            return error

    with ThreadPoolExecutor(max_workers=hash_workers) as executor:
        source_fingerprints = dict(zip(
            source_files, executor.map(fingerprint_or_error, source_files),
        ))

    current_sources = set()
    desired_ids = set()
    indexed_files = 0
    skipped_files = 0
    empty_files = 0
    indexed_chunks = 0
    removed_chunks = 0
    skip_unchanged = (
        not args.dry_run
        and not args.clear
        and not bool(getattr(args, "force", False))
        and _env_flag("INGEST_SKIP_UNCHANGED", True)
    )
    known_index = _indexed_sources(coll, tenant_id) if not args.dry_run else {}
    indexed = known_index if skip_unchanged else {}

    continue_on_error = bool(getattr(args, "continue_on_error", False))
    report_argument = getattr(args, "report_json", None)
    report_path = Path(report_argument).expanduser().resolve() if report_argument else None
    report: Dict[str, Any] = {
        "tenant": tenant_id, "docs_dir": str(docs_dir),
        "dry_run": bool(args.dry_run), "complete": False,
        "discovered_files": len(source_files), "documents": [],
    }
    if requested_sources:
        report["selected_sources"] = requested_sources
        report["pdf_native_only"] = pdf_native_only
    failed_files = 0
    _write_ingest_report(report_path, report)
    for file_path in source_files:
        total_files += 1
        source_result: Dict[str, Any] = {
            "source": _normalize_rel_path(file_path, docs_dir),
            "extension": file_path.suffix.lower(), "status": "processing",
            "extraction_method": _extraction_method(file_path, pdf_native_only=pdf_native_only), "chunks": 0,
        }
        report["documents"].append(source_result)
        _write_ingest_report(report_path, report)
        try:

            source_rel = _normalize_rel_path(file_path, docs_dir)  # e.g. "ceo/index.md"
            pipeline_fingerprint = _pipeline_fingerprint(
                args.embedder, source_suffix=file_path.suffix.lower(),
                pdf_native_only=pdf_native_only,
            )
            current_sources.add(source_rel)
            fingerprint = source_fingerprints[file_path]
            if isinstance(fingerprint, Exception):
                raise fingerprint
            source_sha256, source_snapshot = fingerprint
            source_result["sha256"] = source_sha256
            source_result["size"] = source_snapshot[0]
            if skip_unchanged and _source_index_is_current(
                indexed.get(source_rel, []),
                tenant_id,
                source_rel,
                source_sha256,
                args.embedder,
                pipeline_fingerprint=pipeline_fingerprint,
            ):
                print(f"[INGEST] Unchanged, kept existing chunks: {source_rel}")
                skipped_files += 1
                source_result.update(
                    status="unchanged", chunks=len(indexed.get(source_rel, [])),
                    source_id=existing_source_id(indexed.get(source_rel, [])),
                )
                stored_methods = {
                    meta.get("extraction_method") for _record_id, meta in indexed.get(source_rel, [])
                    if isinstance(meta.get("extraction_method"), str) and meta.get("extraction_method")
                }
                if len(stored_methods) == 1:
                    source_result["extraction_method"] = next(iter(stored_methods))
                continue

            chunks, metas = (
                _chunks_for_file(file_path, docs_dir, pdf_native_only=True)
                if pdf_native_only else _chunks_for_file(file_path, docs_dir)
            )
            source_result["extraction_method"] = _extraction_method(file_path, pdf_native_only=pdf_native_only)
            source_result["extracted_chunks"] = len(chunks)
            source_result["extracted_characters"] = sum(len(chunk) for chunk in chunks)
            canonical_trace = _CANONICAL_TRACE_CACHE.pop(
                str(file_path.resolve()),
                None,
            )
            if canonical_trace is None:
                canonical_trace = canonicalize_source_text("\n\n".join(chunks))
            if _file_snapshot(file_path) != source_snapshot:
                raise IngestExtractionError(
                    f"source changed while being extracted: {file_path}"
                )
            if len(chunks) > max_chunks_per_file:
                raise IngestExtractionError(
                    f"source produced {len(chunks)} chunks, exceeding "
                    f"INGEST_MAX_CHUNKS_PER_FILE={max_chunks_per_file}: {file_path}"
                )

            if not chunks:
                removed: List[str] = []
                if not args.dry_run:
                    # A successfully decoded empty text file represents
                    # an empty new generation. Remove its previous chunks. Media
                    # extraction failures raise earlier and never reach this path.
                    removed = _upsert_file_and_prune_stale(
                        coll,
                        [],
                        [],
                        [],
                        embedder,
                        tenant_id,
                        source_rel,
                        batch_size=batch_size,
                    )
                print(
                    f"[INGEST] Empty source: {source_rel}; "
                    f"removed {len(removed)} previous chunks"
                )
                source_result.update(status="no_extractable_text", chunks=0)
                empty_files += 1
                removed_chunks += len(removed)
                continue

            # Attach consistent metadata for each chunk from this file.
            stable_source_id = (
                existing_source_id(known_index.get(source_rel, []))
                or new_source_id()
            )
            canonical_audit_key = extraction_audit_id(canonical_trace, source_sha256)
            file_metas: List[Dict] = []
            for i, m in enumerate(metas):
                meta = _safe_meta(m or {}, source_rel)
                meta["path"] = source_rel  # alias for source
                meta["chunk_index"] = i
                meta["chunk_count"] = len(chunks)
                meta["previous_chunk_index"] = i - 1 if i > 0 else -1
                meta["next_chunk_index"] = i + 1 if i + 1 < len(chunks) else -1
                meta["structural_adjacency_revision"] = STRUCTURAL_ADJACENCY_REVISION
                meta["source_sha256"] = source_sha256
                meta["source_size"] = source_snapshot[0]
                meta["source_mtime_ns"] = source_snapshot[1]
                meta["ingest_schema_version"] = INGEST_SCHEMA_VERSION
                meta["source_link_schema_version"] = SOURCE_LINK_SCHEMA_VERSION
                meta["pipeline_fingerprint"] = pipeline_fingerprint
                meta["embedding_format_revision"] = EMBEDDING_FORMAT_REVISION
                meta["embed_model"] = args.embedder
                meta["language"] = _text_language(chunks[i])
                meta["chunk_chars"] = len(chunks[i])
                # Browser links are always resolved through this opaque ID.  The
                # path remains tenant-relative and is never itself a public URL.
                meta["source_id"] = stable_source_id
                meta["tenant_id"] = tenant_id
                meta["source_relative_path"] = source_rel
                meta["display_name"] = Path(source_rel).name
                meta["mime_type"] = guessed_mime_type(source_rel)
                meta["source_checksum"] = source_sha256
                meta["canonical_audit_key"] = canonical_audit_key
                for key, value in location_metadata(meta).items():
                    if value is not None:
                        meta[key] = value
                file_metas.append(meta)

            file_ids = [
                _chunk_id(tenant_id, source_rel, i)
                for i in range(len(chunks))
            ]
            for index, meta in enumerate(file_metas):
                meta["previous_record_id"] = (
                    file_ids[index - 1] if index > 0 else ""
                )
                meta["next_record_id"] = (
                    file_ids[index + 1] if index + 1 < len(file_ids) else ""
                )
            desired_ids.update(file_ids)

            if not args.dry_run:
                if file_path.suffix.lower() in {
                    ".pdf", ".ps", ".png", ".jpg", ".jpeg", ".webp",
                }:
                    audit_root = os.getenv(
                        "INGEST_EXTRACTION_AUDIT_DIR",
                        str(Path(CHROMA_PATH).parent / "extraction-audit"),
                    )
                    audit_id, _audit_payload_sha256 = persist_extraction_audit(
                        canonical_trace,
                        audit_root=audit_root,
                        tenant_id=tenant_id,
                        source_id=stable_source_id,
                        source_sha256=source_sha256,
                    )
                    if audit_id != canonical_audit_key:
                        raise IngestExtractionError(
                            "canonical extraction audit identity mismatch"
                        )
                removed = _upsert_file_and_prune_stale(
                    coll,
                    chunks,
                    file_metas,
                    file_ids,
                    embedder,
                    tenant_id,
                    source_rel,
                    batch_size=batch_size,
                )
                if removed:
                    removed_chunks += len(removed)
                    print(
                        f"[INGEST] Removed {len(removed)} stale chunks "
                        f"for {source_rel}"
                    )

            print(f"[INGEST] Prepared {len(chunks)} chunks from {source_rel}")
            indexed_files += 1
            indexed_chunks += len(chunks)
            source_result.update(
                status="prepared" if args.dry_run else "indexed",
                chunks=len(chunks), source_id=stable_source_id,
            )
        except Exception as error:
            failed_files += 1
            source_result.update(
                status="no_extractable_text" if isinstance(error, IngestNoTextError) else "failed",
                error=f"{type(error).__name__}: {error}",
            )
            print(
                f"[INGEST] FAILED {source_result['source']}: {type(error).__name__}: {error}",
                file=sys.stderr,
            )
            if not continue_on_error:
                raise
        finally:
            _CANONICAL_TRACE_CACHE.pop(str(file_path.resolve()), None)
            _PDF_RECONSTRUCTION_TRACE_CACHE.pop(str(file_path.resolve()), None)
            _SOURCE_EXTRACTION_METHODS.pop(str(file_path.resolve()), None)
            _write_ingest_report(report_path, report)

    # A partial import keeps successful per-source transactions, but never
    # authorizes global reconciliation that could delete unprocessed data.
    if failed_files and (args.clear or args.sync):
        print("[INGEST] Partial import: global rebuild/sync deletion was skipped.")
    elif args.clear and not args.dry_run:
        removed = _clear_except_ids(coll, tenant_id, desired_ids)
        removed_chunks += len(removed)
        print(
            f"[INGEST] Safe rebuild removed {len(removed)} obsolete tenant records."
        )
    elif args.sync and not args.dry_run:
        if not current_sources and not allow_empty_sync:
            raise RuntimeError(
                "--sync found no included source files; refusing to delete the "
                "tenant index. Verify the docs mount/include filter or pass "
                "--allow-empty-sync explicitly."
            )
        # A scoped --include is an indexing filter, not deletion authority for
        # other extensions. Protect previously indexed excluded sources even
        # if they are currently absent from disk.
        sync_sources = set(current_sources)
        current_index = _indexed_sources(coll, tenant_id)
        sync_sources.update(
            source
            for source in current_index
            if not _source_matches_include(source, include_exts)
        )
        removed = _sync_deleted_sources(coll, tenant_id, sync_sources)
        removed_chunks += len(removed)
        print(f"[INGEST] Sync removed {len(removed)} chunks for deleted files.")
    elif args.clear:
        print("[INGEST] Dry run: skipped safe-rebuild reconciliation.")
    elif args.sync:
        print("[INGEST] Dry run: skipped deleted-file synchronization.")

    report["complete"] = True
    report["failed_files"] = failed_files
    report["elapsed_sec"] = round(time.perf_counter() - started_at, 3)
    _write_ingest_report(report_path, report)
    print(f"[INGEST] Done. Processed files: {total_files}")
    print("[INGEST] SUMMARY " + json.dumps({
        "tenant": tenant_id,
        "processed_files": total_files,
        "indexed_files": indexed_files,
        "skipped_files": skipped_files,
        "empty_files": empty_files,
        "failed_files": failed_files,
        "indexed_chunks": indexed_chunks,
        "removed_chunks": removed_chunks,
        "hash_workers": hash_workers,
        "elapsed_sec": round(time.perf_counter() - started_at, 3),
        "dry_run": bool(args.dry_run),
    }, sort_keys=True))
    if args.dry_run:
        print("[INGEST] Dry run finished (no writes).")
    if failed_files:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
