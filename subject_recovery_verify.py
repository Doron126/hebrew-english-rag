"""Independent qualification of a failed monetary representation read.

The first reader records candidate-bound outcomes. This check reauthorizes
each exact candidate and repeats native reads and parsing against the original
source generation before a document-only operation may be reconsidered.
"""
from source_authorization import authorize_source_candidate
import source_access
from source_access import SourceAccessError, guessed_mime_type
from monetary_native_pdf import parse_monetary_page
from monetary_questions import (MonetaryEvidence, MonetaryReadOutcome, candidate_ids,
                                MAX_MONETARY_PAGES, MAX_MONETARY_SOURCES)
from financial_semantics import monetary_question


def independently_verified_unsupported(evidence, query, tenant, document_root):
    if (not isinstance(evidence, MonetaryEvidence) or not evidence.witness
            or not evidence.witness.storage_complete or evidence.passages
            or evidence.reason != 'monetary_source_representation_unsupported'
            or evidence.tenant_id != tenant):
        return False
    grouped = {}
    for row in evidence.witness.rows:
        meta = row.get('metadata') or {}
        if meta.get('tenant_id') != tenant or not meta.get('source_id'):
            return False
        grouped.setdefault(meta['source_id'], []).append(row)
    request = monetary_question(query)
    if request is None:
        return False
    candidates = candidate_ids(grouped, request)
    outcomes = evidence.read_outcomes
    if (not candidates or len(candidates) > MAX_MONETARY_SOURCES
            or len(outcomes) != len(candidates)
            or any(not isinstance(item, MonetaryReadOutcome) for item in outcomes)
            or tuple(item.source_id for item in outcomes) != candidates):
        return False
    used = 0
    for outcome in outcomes:
        rows = grouped.get(outcome.source_id, ())
        if not rows:
            return False
        metadata = dict(rows[0]['metadata'])
        identity_keys = ('tenant_id','source_id','source_sha256','source_relative_path',
                         'source_size','source_mtime_ns','mime_type')
        if (outcome.source_sha256 != metadata.get('source_sha256')
                or any(any(row['metadata'].get(key) != metadata.get(key)
                           for key in identity_keys) for row in rows)):
            return False
        authorization = authorize_source_candidate(
            document_root, tenant, metadata, candidate_text=str(rows[0].get('document') or ''),
            candidate_id=str(rows[0].get('id') or ''))
        if not authorization.authorized:
            return False
        mime = str(metadata.get('mime_type') or '')
        if outcome.state == 'non_pdf':
            if (mime == 'application/pdf' or outcome.requested_pages or outcome.page_hashes
                    or guessed_mime_type(metadata.get('source_relative_path', '')) != mime):
                return False
            continue
        if outcome.state != 'unsupported_pdf' or mime != 'application/pdf':
            return False
        pages = tuple(sorted({row['metadata'].get('page') for row in rows
                              if type(row['metadata'].get('page')) is int}))
        if not pages or len(pages) + used > MAX_MONETARY_PAGES:
            return False
        requested = tuple(sorted({page for anchor in pages
                                  for page in (anchor-1,anchor,anchor+1) if page > 0}))
        if len(requested) > MAX_MONETARY_PAGES or requested != outcome.requested_pages:
            return False
        try:
            recovered = source_access.pdf_source_pages(document_root, tenant, metadata,
                                                       requested, include_geometry=True)
        except (OSError, ValueError, SourceAccessError):
            return False
        used += len(recovered)
        hashes = tuple((page['page'],page['raw_sha256'],page['text_sha256'])
                       for page in recovered)
        counts = {page.get('page_count') for page in recovered}
        if (not recovered or len(counts) != 1
                or any(type(count) is not int or count < 1 for count in counts)):
            return False
        page_count = next(iter(counts))
        actual = tuple(page['page'] for page in recovered)
        expected = tuple(page for page in requested if page <= page_count)
        if (used > MAX_MONETARY_PAGES or actual != expected
                or not set(pages).issubset(actual) or hashes != outcome.page_hashes
                or any(parse_monetary_page(page) is not None for page in recovered)):
            return False
    return True
