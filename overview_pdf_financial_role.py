"""Strict, original-PDF financial roles for a bounded source overview."""
from __future__ import annotations
import re
from financial_documents import (DOCUMENT_KIND_PATTERN, canonical_document_kind,
                                 role_matches, primary_document_identities, document_identities)
from financial_semantics import _PAYMENT_METHOD, canonical_payment_method
from monetary_native_pdf import (_PAYMENT_TABLE, native_cell_literal_spans,
                                 parse_monetary_page)
from monetary_questions import _recorded_payments
from monetary_presentation import material_monetary_conditions

_TYPE = re.compile(
    r'^(?:(?:המסמך|המקור|החומר)\s+(?:הוא|הינו|הינה)\s+|'
    r'(?:the\s+)?(?:document|source|material)\s+is\s+(?:an?\s+)?)'
    r'(?P<kind>'+DOCUMENT_KIND_PATTERN+r')\s*[.!?]?$', re.I)
_ITEM = re.compile(
    r'^(?:(?:המסמך|המקור|החומר)\s+(?:מפרט|מציין|מתאר)\s+|'
    r'(?:the\s+)?(?:document|source|material)\s+(?:lists|describes|states)\s+)'
    r'(?P<item>.+?)\s*[.!?]?$', re.I)
_PAYMENT = re.compile(
    r'^(?:(?:המסמך|המקור|החומר)\s+(?:נוגע|מתייחס)\s+לתשלום\s+'
    r'ש(?:בוצע|התקבל|שולם)\s+באמצעות\s+|'
    r'(?:תועד|נרשם)\s+תשלום\s+ש(?:בוצע|התקבל|שולם)\s+באמצעות\s+|'
    r'(?:the\s+)?(?:document|source|material)\s+(?:describes|records|concerns)\s+'
    r'(?:a\s+)?payment\s+(?:made|received|completed)\s+(?:by|via|through)\s+)'
    r'(?P<method>'+_PAYMENT_METHOD.pattern+r')\s*[.!?]?$', re.I)


def _normalized(value):
    return ' '.join(str(value).split())


def _supported_by_native_role(body, passage, page, record):
    if passage.metadata.get('page') != record.page or passage.metadata.get('page') != page['page']:
        return False
    spans = native_cell_literal_spans(page, record.document_identity_word_ids)
    if len(spans) != 1:
        return False
    title = page['text'][spans[0][0]:spans[0][1]]
    native_id = primary_document_identities(title)
    if (len(native_id) != 1 or native_id[0].span != (0, len(title))
            or native_id[0].number != record.document_number
            or not role_matches(native_id[0].kind, record.document_kind)):
        return False
    type_claim = _TYPE.fullmatch(body.strip())
    if type_claim:
        target = (native_id[0].kind, record.document_number)
        indexed = [identity for identity in document_identities(passage.source_claim)
                   if (identity.kind, identity.number) == target]
        page_ids = [identity for identity in document_identities(passage.text)
                    if (identity.kind, identity.number) == target]
        requested = canonical_document_kind(type_claim.group('kind'))
        if len(indexed) != 1 or len(page_ids) != 1 or not role_matches(record.document_kind, requested):
            return False
        # The canonical role intentionally groups tax and ordinary invoices;
        # a claimed tax modifier needs its own source-identity proof.
        # Canonical financial kinds intentionally collapse some invoice
        # subtypes. A specific modifier must be present in this very title,
        # both in the native identity and the unique cited indexed identity.
        modifiers = (
            re.compile(r'\btax\s+invoice\b|(?<![א-ת])חשבונית\s+מס(?![א-ת])', re.I),
            re.compile(r'(?<![א-ת])חשבונית\s+עסקה(?![א-ת])', re.I),
        )
        indexed_title = passage.source_claim[slice(*indexed[0].span)]
        for modifier in modifiers:
            if modifier.search(type_claim.group('kind')) and (
                    not modifier.search(title) or not modifier.search(indexed_title)):
                return False
        return True
    item_claim = _ITEM.fullmatch(body.strip())
    if item_claim:
        from monetary_questions import _unsafe_payment_item_excerpt
        wanted = _normalized(item_claim.group('item')).rstrip('.!?')
        matching = [item for item in record.items
                    if _normalized(item.description).rstrip('.!?') == wanted]
        if len(matching) != 1 or not wanted or _unsafe_payment_item_excerpt(wanted):
            return False
        item = matching[0]
        if material_monetary_conditions(record, item):
            return False
        full = _normalized(passage.text)
        cited = _normalized(passage.source_claim)
        if full.count(wanted) != 1 or cited.count(wanted) != 1:
            return False
        spans = native_cell_literal_spans(page, item.description_word_ids)
        return bool(spans and all(_normalized(page['text'][a:b]) in cited for a,b in spans))
    payment_claim = _PAYMENT.fullmatch(body.strip())
    if not payment_claim:
        return False
    requested = canonical_payment_method(payment_claim.group('method'))
    if not requested or not _PAYMENT_TABLE.search(passage.source_claim):
        return False
    # An unbound material condition may apply to this payment. Until its
    # ownership is independently proved, do not state occurrence.
    if material_monetary_conditions(record):
        return False
    recorded = _recorded_payments(record, passage.source_id, passage.tenant_id)
    matches = [p for p in record.payments
        if canonical_payment_method(p.method) == requested
        and any(p.amount == q.amount and p.currency == q.currency
                and p.payment_date == q.date
                and canonical_payment_method(p.method) == canonical_payment_method(q.method)
                for q in recorded)]
    if len(matches) != 1:
        return False
    # Anchor all scalar cells to the *same* reconstructed native row in this
    # citation, not merely to identical values elsewhere on the page.
    heading = _PAYMENT_TABLE.search(matches[0].context_text)
    row = _normalized(matches[0].context_text[heading.end():]) if heading else ''
    cited = _normalized(passage.source_claim)
    full_page = _normalized(passage.text)
    cited_heading = _PAYMENT_TABLE.search(cited)
    if (not row or not cited_heading or cited.count(row) != 1 or full_page.count(row) != 1
            or cited_heading.end() > cited.index(row)):
        return False
    for field in (matches[0].amount_word_ids, matches[0].date_word_ids, matches[0].method_word_ids):
        pieces = native_cell_literal_spans(page, field)
        if not pieces or any(_normalized(page['text'][a:b]) not in _normalized(passage.source_claim)
                             for a,b in pieces):
            return False
    return True


def _verified_native(body, ids, evidence):
    """Authenticate the scoped PDF and obtain its original monetary record."""
    from chat_api import DOCS_BASE_DIR
    from document_reading import (SourceSummaryEvidence, numbered_document_overview_request,
                                  _verify_original_pdf_summary, validate_summary_evidence)
    from source_access import pdf_source_pages
    from overview_semantic import _verified
    if (not isinstance(evidence, SourceSummaryEvidence) or not evidence.overview_synthesis
            or not numbered_document_overview_request(evidence.question)
            or str(DOCS_BASE_DIR) != evidence.pdf_document_root
            or not evidence.pdf_tenant or not evidence.witness
            or not validate_summary_evidence(evidence, evidence.question)
            or len(ids) != 1 or evidence.quote_eligible_ids is None
            or ids[0] not in evidence.quote_eligible_ids):
        return None
    matches = [p for p in evidence.passages if p.citation_id == ids[0]]
    if len(matches) != 1:
        return None
    passage = matches[0]
    if (passage.tenant_id != evidence.pdf_tenant
            or passage.metadata.get('mime_type') != 'application/pdf'
            or type(passage.metadata.get('page')) is not int):
        return None
    # A reviewed exact claim cannot evade a negative, unresolved or invalid
    # verdict merely because another deterministic role path is available.
    key = (body, tuple(ids))
    indices = [i for i, candidate in enumerate(evidence.overview_claims) if candidate == key]
    if indices:
        verdicts = _verified(evidence.overview_question, evidence, evidence.overview_claims,
                             evidence.overview_tenant, evidence.overview_review)['claims']
        if any(verdicts[i][1] != 'supported' for i in indices):
            return None
    if (_verify_original_pdf_summary(evidence.witness.rows, DOCS_BASE_DIR, evidence.pdf_tenant)
            != evidence.pdf_page_hashes or not evidence.pdf_page_hashes):
        return None
    pages = pdf_source_pages(DOCS_BASE_DIR, evidence.pdf_tenant, passage.metadata,
                             (passage.metadata['page'],), include_geometry=True)
    if len(pages) != 1:
        return None
    record = parse_monetary_page(pages[0])
    return (passage, pages[0], record) if record else None


def overview_pdf_role_supported(body, ids, evidence):
    """Re-open and reparse originals; never let review authorize the proof."""
    try:
        if not isinstance(body, str) or not (
                _TYPE.fullmatch(body.strip()) or _PAYMENT.fullmatch(body.strip())
                or _ITEM.fullmatch(body.strip())):
            return False
        native = _verified_native(body, ids, evidence)
        return bool(native and _supported_by_native_role(body, *native))
    except Exception:
        return False


def overview_pdf_item_recovery(body, ids, evidence, hebrew):
    """Return a full uniquely matching original item, or no recovery.

    Only a supported one-topic model claim may nominate a source-owned item;
    nomination cannot borrow another row, condition, property or owner.
    """
    try:
        from overview_semantic import claim_supported, _topic_items
        if (not claim_supported(evidence.overview_question, evidence, evidence.overview_claims,
                                evidence.overview_tenant, evidence.overview_review, body, ids)):
            return ''
        topics = _topic_items(body)
        if len(topics) != 1:
            return ''
        native = _verified_native(body, ids, evidence)
        if not native:
            return ''
        passage, page, record = native
        topic = topics[0].casefold()
        items = [item for item in record.items if topic in
                 {word.casefold() for word in re.findall(r'[A-Za-z]+|[א-ת]+',item.description)}]
        if len(items) != 1:
            return ''
        description = _normalized(items[0].description)
        # Do not append incidental amounts, model numbers, or identifiers
        # merely to recover the subject of an overview.
        if re.search(r'[0-9₪$€£%]', description):
            return ''
        claim = ('המסמך מפרט ' if hebrew else 'The document lists ') + description
        return claim if _supported_by_native_role(claim, passage, page, record) else ''
    except Exception:
        return ''
