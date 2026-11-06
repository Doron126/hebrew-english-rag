"""Bounded query-time native text recovery for already-authorized PDF sources.

The indexed scan remains the coverage witness. Native page rows are a separate
representation of the same source generation, selected by an exact indexed
entity identity and independently reread by the final quote consumer.
"""
from dataclasses import dataclass
import hashlib


@dataclass(frozen=True)
class NativePDFRecovery:
    native_rows: tuple[dict, ...] = ()
    reasons: tuple[str, ...] = ()


def recover_entity_pdf_rows(grouped, query, document_root, tenant):
    from entity_sections import (entity_reference, _source_matches,
                                 _select_entity_sections, _BREADCRUMB)
    from document_query import role_bound_identifier_spans
    from source_access import SourceAccessError, pdf_source_pages, source_metadata_complete
    from semantic_answer_quality import canonical_quote_is_readable
    reference = entity_reference(query)
    if reference is None:
        return NativePDFRecovery()
    indexed, _complete, _reason = _select_entity_sections(grouped, reference, query=query)
    if indexed and all(canonical_quote_is_readable(hit['document']) for hit in indexed):
        return NativePDFRecovery()
    reason = 'indexed_entity_evidence_missing' if not indexed else 'indexed_entity_quote_unreadable'
    recovered = []
    reasons = []
    used_pages = 0
    for _source, source_rows in sorted(grouped.items()):
        rows = sorted(source_rows, key=lambda row: ((row.get('metadata') or {}).get('chunk_index', 0),
                                                   str(row.get('id') or '')))
        if not rows or not _source_matches(reference, rows):
            continue
        metadata = dict(rows[0].get('metadata') or {})
        if str(metadata.get('mime_type') or '').casefold() != 'application/pdf':
            continue
        identity_keys = ('source_id','tenant_id','source_sha256','source_relative_path',
                         'source_size','source_mtime_ns','mime_type')
        if (not source_metadata_complete(metadata, tenant)
                or any(any(row['metadata'].get(key) != metadata.get(key) for key in identity_keys)
                       for row in rows)):
            reasons.append('native_pdf_source_identity_inconsistent')
            continue
        joined = ''
        positions = []
        for row in rows:
            text = str(row.get('document') or '')
            crumb = _BREADCRUMB.match(text)
            body = text[crumb.end():] if crumb else text
            positions.append((len(joined),len(joined)+len(body),row))
            joined += body+'\n'
        anchors = [item for item in role_bound_identifier_spans(joined)
                   if item.kind == reference.kind and item.identifier == reference.identifier]
        anchor_rows = [row for a,b,row in positions if any(
            a < item.entity_span[1] and item.entity_span[0] < b for item in anchors)]
        if (not anchor_rows or any(type(row['metadata'].get('page')) is not int
                                   or row['metadata']['page'] < 1 for row in anchor_rows)):
            continue
        anchor_pages = tuple(sorted({row['metadata']['page'] for row in anchor_rows}))
        requested = tuple(sorted({page for anchor in anchor_pages
                                  for page in (anchor-1,anchor,anchor+1) if page > 0}))
        if requested != tuple(range(requested[0],requested[-1]+1)):
            reasons.append('native_pdf_disjoint_page_window')
            continue
        if len(requested)+used_pages > 8:
            reasons.append('native_pdf_request_page_bound')
            continue
        try:
            pages = pdf_source_pages(document_root, tenant, metadata, requested)
        except (OSError, SourceAccessError):
            reasons.append('native_pdf_source_unavailable')
            continue
        actual_requested = tuple(page['page'] for page in pages)
        if not set(anchor_pages) <= set(actual_requested):
            reasons.append('native_pdf_anchor_page_outside_source')
            continue
        native_text = '\n'.join(page['text'] for page in pages)
        if not any(item.kind == reference.kind and item.identifier == reference.identifier
                   for item in role_bound_identifier_spans(native_text)):
            reasons.append('native_pdf_entity_identity_not_found')
            continue
        used_pages += len(pages)
        anchor_ids = tuple(str(row.get('id') or row['metadata'].get('chunk_id') or '')
                           for row in anchor_rows)
        for page in pages:
            meta = dict(metadata)
            meta.pop('chunk_count', None)
            identifier = f"{metadata['source_id']}::native-pdf::{page['page']:06d}::{page['text_sha256'][:16]}"
            meta.update(page=page['page'], section=f"Page {page['page']}",
                chunk_id=identifier, chunk_index=page['page']-1,
                _entity_representation='native_pdf',
                _native_pdf_page_text_sha256=page['text_sha256'],
                _native_pdf_raw_text_sha256=page['raw_sha256'],
                _native_pdf_page_count=page['page_count'],
                _native_pdf_anchor_pages=anchor_pages,
                _native_pdf_anchor_chunk_ids=anchor_ids,
                _native_pdf_requested_pages=actual_requested,
                _native_pdf_recovery_reason=reason)
            recovered.append(dict(id=identifier,document=page['text'],metadata=meta,
                                  score_kind='native_pdf_page'))
        reasons.append(reason)
    return NativePDFRecovery(tuple(recovered),tuple(dict.fromkeys(reasons)))
