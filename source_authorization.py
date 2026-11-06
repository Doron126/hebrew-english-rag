"""One auditable authorization boundary for retrieved source candidates.

This module deliberately knows nothing about request semantics or structural
segmentation.  It establishes that a retrieved candidate belongs to the active
tenant and to the exact indexed source generation before any fact path may use
its text.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

from source_access import (
    SourceAccessError,
    open_source_file,
    source_metadata_complete,
    source_sha256_matches_metadata,
    source_stat_matches_metadata,
)


@dataclass(frozen=True)
class SourceAuthorization:
    status: str
    reason: str
    source_id: str = ""
    chunk_id: str = ""
    source_relative_path: str = ""
    expected_source_hash: str = ""
    actual_source_hash: str = ""
    source_path_status: str = "not_checked"
    source_stat_status: str = "not_checked"

    @property
    def authorized(self) -> bool:
        return self.status == "source_authorized"

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_authorization_status": self.status,
            "source_authorization_reason": self.reason,
            "source_id": self.source_id,
            "chunk_id": self.chunk_id,
            "source_relative_path": self.source_relative_path,
            "expected_source_hash": self.expected_source_hash,
            "actual_source_hash": self.actual_source_hash,
            "actual_source_hash_match": self.authorized,
            "source_path_status": self.source_path_status,
            "source_stat_status": self.source_stat_status,
        }


def authorize_source_candidate(
    document_root: str | Path,
    tenant_id: str,
    metadata: Mapping[str, Any],
    *,
    candidate_text: str = "",
    candidate_id: str = "",
) -> SourceAuthorization:
    """Authorize one selected candidate with a precise terminal outcome."""
    metadata = dict(metadata or {})
    source_id = str(metadata.get("source_id") or "")
    # Established Chroma rows can carry the opaque chunk identity in the
    # record ID rather than metadata.  Accept only that retrieved identity;
    # never manufacture a chunk identifier.
    chunk_id = str(metadata.get("chunk_id") or candidate_id or "")
    relative = str(metadata.get("source_relative_path") or "")
    expected_hash = str(metadata.get("source_sha256") or "")
    common = dict(source_id=source_id, chunk_id=chunk_id,
                  source_relative_path=relative, expected_source_hash=expected_hash)
    if str(metadata.get("tenant_id") or "") != str(tenant_id or ""):
        return SourceAuthorization("source_tenant_mismatch", "tenant_mismatch", **common)
    if not source_id:
        return SourceAuthorization("source_identity_missing", "source_identity_missing", **common)
    if not chunk_id:
        return SourceAuthorization("chunk_identity_missing", "chunk_identity_missing", **common)
    if not relative:
        return SourceAuthorization("source_relative_path_missing", "source_relative_path_missing", **common)
    # Classify unsafe stored path shapes before metadata completeness folds
    # them into a generic failure.  ``open_source_file`` still enforces the
    # authoritative path policy and protects against races/symlinks.
    normalized_parts = relative.replace("\\", "/").split("/")
    if relative.startswith(("/", "\\")) or ".." in normalized_parts:
        return SourceAuthorization("source_path_escape_rejected", "source_path_escape_rejected", **common)
    root = Path(document_root).expanduser()
    if not str(document_root or ""):
        return SourceAuthorization("docs_base_dir_missing", "docs_base_dir_missing", **common)
    if not root.is_absolute() or not root.is_dir():
        return SourceAuthorization("docs_base_dir_invalid", "docs_base_dir_invalid", **common)
    if not source_metadata_complete(metadata, tenant_id):
        return SourceAuthorization("source_metadata_inconsistent", "source_metadata_inconsistent", **common)
    if not str(candidate_text or "").strip():
        return SourceAuthorization("candidate_text_missing", "candidate_text_missing", **common)
    handle = None
    try:
        _relative, handle, source_stat = open_source_file(root, tenant_id, relative)
    except SourceAccessError as exc:
        # source_access intentionally does not disclose filesystem details.
        # Preserve the path-shape distinction before falling back to absence.
        reason = "source_file_not_found"
        return SourceAuthorization(reason, reason, source_path_status="rejected", **common)
    except OSError:
        return SourceAuthorization("source_open_failed", "source_open_failed", source_path_status="open_failed", **common)
    try:
        if not source_stat_matches_metadata(metadata, source_stat):
            size_matches = metadata.get("source_size") == source_stat.st_size
            reason = "source_stat_mismatch" if size_matches else "source_size_mismatch"
            return SourceAuthorization(reason, reason, source_path_status="opened",
                                       source_stat_status="mismatch", **common)
        if not expected_hash:
            return SourceAuthorization("source_sha256_missing", "source_sha256_missing",
                                       source_path_status="opened", source_stat_status="matched", **common)
        digest = hashlib.sha256()
        handle.seek(0)
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        actual = digest.hexdigest()
        if not source_sha256_matches_metadata(metadata, handle, enabled=True):
            return SourceAuthorization("source_sha256_mismatch", "source_sha256_mismatch",
                                       actual_source_hash=actual, source_path_status="opened",
                                       source_stat_status="matched", **common)
        return SourceAuthorization("source_authorized", "source_authorized",
                                   actual_source_hash=actual, source_path_status="opened",
                                   source_stat_status="matched", **common)
    finally:
        if handle is not None:
            handle.close()
