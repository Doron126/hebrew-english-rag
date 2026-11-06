"""Metadata-only inventory for an already authorized tenant collection."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import quote

try:
    from .source_access import open_source_file, source_metadata_complete, source_stat_matches_metadata
except ImportError:
    from source_access import open_source_file, source_metadata_complete, source_stat_matches_metadata


class InventoryIntegrityError(ValueError):
    """The active index and its original files do not form one coherent corpus."""


class NamedDocumentAmbiguity(InventoryIntegrityError):
    """Several authorized catalog identities match the requested title."""
    def __init__(self, names):
        self.names = tuple(names)
        super().__init__('ambiguous named document')


def inventory_documents(retriever, document_root: str | Path) -> dict:
    """Read no document bodies from Chroma and fail closed on foreign rows.

    The caller must authorize ``retriever.tenant_id`` before invoking this helper.
    PDF page counts come from the actual original file, independently of which
    pages contain enough text to produce a searchable chunk.
    """
    tenant = retriever.tenant_id
    grouped: dict[str, dict] = {}
    seen_ids: set[str] = set()
    offset = 0
    while True:
        kwargs = {"include": ["metadatas"], "offset": offset, "limit": 2048}
        if retriever.where is not None:
            kwargs["where"] = retriever.where
        result = retriever.collection.get(**kwargs)
        ids = result.get("ids") or []
        metas = result.get("metadatas") or []
        if len(ids) != len(metas):
            raise InventoryIntegrityError("incomplete inventory metadata")
        if not ids:
            break
        for chunk_id, meta in zip(ids, metas):
            if chunk_id in seen_ids:
                raise InventoryIntegrityError("duplicate inventory chunk")
            seen_ids.add(chunk_id)
            if not source_metadata_complete(meta, tenant):
                raise InventoryIntegrityError("incompatible or foreign source metadata")
            source_id = meta["source_id"]
            identity = tuple(meta[key] for key in (
                "display_name", "source_relative_path", "source_sha256",
                "source_size", "source_mtime_ns", "mime_type",
            ))
            item = grouped.setdefault(source_id, {
                "identity": identity, "metadata": meta, "chunk_count": 0,
                "indexed_pages": set(), "indexed_slides": set(), "indexed_sheets": set(),
            })
            if item["identity"] != identity:
                raise InventoryIntegrityError("inconsistent source identity")
            item["chunk_count"] += 1
            page = meta.get("page")
            if isinstance(page, int) and not isinstance(page, bool) and page > 0:
                item["indexed_pages"].add(page)
            slide = meta.get("slide")
            if isinstance(slide, int) and not isinstance(slide, bool) and slide > 0:
                item["indexed_slides"].add(slide)
            sheet = meta.get("sheet")
            if isinstance(sheet, str) and sheet.strip():
                item["indexed_sheets"].add(sheet)
        offset += len(ids)
        if len(ids) < 2048:
            break
        if offset >= 1_000_000:
            raise InventoryIntegrityError("inventory exceeds supported bound")
    documents = []
    for source_id, item in grouped.items():
        meta = item["metadata"]
        relative, handle, before = open_source_file(document_root, tenant, meta["source_relative_path"])
        with handle:
            if not source_stat_matches_metadata(meta, before):
                raise InventoryIntegrityError("source file generation does not match index")
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != meta["source_sha256"]:
                raise InventoryIntegrityError("source checksum does not match index")
            pdf_page_count = None
            if relative.suffix.casefold() == ".pdf":
                import fitz
                handle.seek(0)
                with fitz.open(stream=handle.read(), filetype="pdf") as pdf:
                    pdf_page_count = len(pdf)
                if any(page > pdf_page_count for page in item["indexed_pages"]):
                    raise InventoryIntegrityError("indexed PDF page is out of range")
            after = os.fstat(handle.fileno())
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise InventoryIntegrityError("source changed during inventory")
        documents.append({
            "source_id": source_id,
            "display_name": meta["display_name"],
            "source_relative_path": meta["source_relative_path"],
            "source_sha256": digest,
            "source_size": after.st_size,
            "mime_type": meta["mime_type"],
            "chunk_count": item["chunk_count"],
            "indexed_pages": sorted(item["indexed_pages"]),
            "indexed_slides": sorted(item["indexed_slides"]),
            "indexed_sheets": sorted(item["indexed_sheets"]),
            "pdf_page_count": pdf_page_count,
            "open_url": f"/api/sources/{quote(source_id)}?tenant={quote(tenant)}",
            "text_extraction_limitations": (
                "Text retrieval includes extracted prose and captions. Image-only chart values are unavailable unless explicitly extracted and validated."
                if pdf_page_count is not None else None
            ),
        })
    documents.sort(key=lambda item: (item["source_relative_path"], item["source_id"]))
    return {
        "tenant_id": tenant,
        "document_count": len(documents),
        "chunk_count": len(seen_ids),
        "documents": documents,
    }

def resolve_named_document_metadata(retriever, reference, *, document_root=None, distinguish_ambiguity=False):
    """Resolve a unique explicit document name inside the existing tenant.

    Identity is selected from fresh bounded metadata. A presentation's first
    indexed chunk may supply its literal title when the filename is unrelated.
    Original file integrity remains the caller's existing authorization gate.
    """
    from copy import deepcopy
    from config_env import validate_tenant_id
    from document_query import DocumentReference, reference_matches
    tenant = validate_tenant_id(retriever.tenant_id)
    if not isinstance(reference, DocumentReference) or reference.kind not in {"document", "spreadsheet", "presentation"}:
        raise InventoryIntegrityError("invalid named-source reference")
    if not isinstance(reference.title, str) or not 1 < len(reference.title) <= 160:
        raise InventoryIntegrityError("invalid named-source reference")
    root = Path(document_root) if document_root is not None else None
    if root is None or not root.is_absolute() or not root.is_dir():
        raise InventoryIntegrityError("invalid named-source document root")
    scope_filter = deepcopy(retriever.where)
    identity_keys = ("tenant_id", "source_id", "source_link_schema_version", "display_name",
                     "source_relative_path", "source_sha256", "source_size", "source_mtime_ns", "mime_type", "title")

    def identity(meta):
        if not source_metadata_complete(meta, tenant) or ("title" in meta and not isinstance(meta["title"], str)):
            raise InventoryIntegrityError("incompatible named-source metadata")
        return tuple(meta.get(key) for key in identity_keys)

    def page(*, limit, offset, include, where):
        kwargs = {"include": include, "limit": limit, "offset": offset}
        if where is not None:
            kwargs["where"] = deepcopy(where)
        try:
            result = retriever.collection.get(**kwargs)
        except Exception as exc:
            raise InventoryIntegrityError("named-source storage unavailable") from exc
        if not isinstance(result, dict):
            raise InventoryIntegrityError("incomplete named-source metadata")
        values = tuple(result.get(key) for key in ("ids", *include))
        if any(not isinstance(value, list) for value in values) or len(values[0]) > limit:
            raise InventoryIntegrityError("incomplete named-source metadata")
        if any(len(value) != len(values[0]) for value in values[1:]):
            raise InventoryIntegrityError("incomplete named-source metadata")
        if any(not isinstance(value, str) or not value for value in values[0]):
            raise InventoryIntegrityError("invalid named-source record identity")
        return values

    grouped, indexed_rows = {}, {}
    for offset in range(0, 100_000, 512):
        ids, metas = page(limit=512, offset=offset, include=["metadatas"], where=scope_filter)
        for chunk_id, meta in zip(ids, metas):
            current_identity = identity(meta)
            if chunk_id in indexed_rows:
                raise InventoryIntegrityError("duplicate named-source record")
            indexed_rows[chunk_id] = dict(meta)
            identifier = meta["source_id"]
            old = grouped.setdefault(identifier, dict(meta))
            if identity(old) != current_identity:
                raise InventoryIntegrityError("ambiguous named-source identity")
        if len(ids) < 512:
            break
    else:
        raise InventoryIntegrityError("named-source metadata scan incomplete")
    matches = {identifier for identifier, meta in grouped.items()
               if (reference_matches(reference, meta["source_relative_path"], meta["mime_type"])
                   or (reference.kind == 'document' and meta.get('title')
                       and reference_matches(reference, meta['title'], meta['mime_type'])))}
    explicit_extension = Path(reference.title).suffix.casefold() in {".pptx", ".ppt", ".odp"}
    if reference.kind == "presentation" and not explicit_extension:
        # Only first chunks of presentation sources are eligible title evidence.
        # Every row is post-checked even if an adapter ignores the filter.
        where = {"$and": [scope_filter, {"chunk_index": 0}]} if scope_filter is not None else {"chunk_index": 0}
        seen_title_ids = set()
        for offset in range(0, 2048, 64):
            ids, docs, metas = page(limit=64, offset=offset, include=["documents", "metadatas"], where=where)
            for chunk_id, meta, text in zip(ids, metas, docs):
                current_identity = identity(meta)
                if chunk_id in seen_title_ids:
                    raise InventoryIntegrityError("duplicate source-title record")
                seen_title_ids.add(chunk_id)
                identifier = meta["source_id"]
                indexed = indexed_rows.get(chunk_id)
                if (indexed is None or identifier not in grouped or identity(indexed) != current_identity
                        or any(type(indexed.get(key)) is not type(meta.get(key)) or indexed.get(key) != meta.get(key)
                               for key in ("chunk_index", "slide"))):
                    raise InventoryIntegrityError("changed source-title identity")
                if type(meta.get("chunk_index")) is not int or meta["chunk_index"] != 0:
                    continue
                if type(meta.get("slide")) is not int or meta["slide"] != 1:
                    continue
                if text is not None and not isinstance(text, str):
                    raise InventoryIntegrityError("invalid source-title text")
                lines = [line.strip() for line in (text or "")[:4096].splitlines()
                         if line.strip() and not line.lstrip().startswith("[")]
                heading = lines[0] if lines and len(lines[0]) <= 240 else ""
                if reference_matches(reference, meta["source_relative_path"], meta["mime_type"], heading):
                    matches.add(identifier)
            if len(ids) < 64:
                break
        else:
            raise InventoryIntegrityError("source-title scan incomplete")
    if len(matches) > 1 and distinguish_ambiguity:
        raise NamedDocumentAmbiguity(sorted(grouped[identifier]['source_relative_path'] for identifier in matches))
    if len(matches) != 1:
        return None
    selected = grouped[next(iter(matches))]
    try:
        relative, handle, before = open_source_file(root, tenant, selected["source_relative_path"])
        with handle:
            if relative.as_posix() != selected["source_relative_path"] or not source_stat_matches_metadata(selected, before):
                raise InventoryIntegrityError("stale named-source generation")
            after = os.fstat(handle.fileno())
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                raise InventoryIntegrityError("named source changed during selection")
    except (OSError, ValueError) as exc:
        raise InventoryIntegrityError("named-source original unavailable or stale") from exc
    # This is metadata selection, not a substitute for the caller's mandatory
    # independent source-file hash authorization.
    return dict(selected)
