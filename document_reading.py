"""Coverage-aware reading for the existing request/evidence/acceptance pipeline.

Catalog identity, source authorization and citation verification remain owned by
their existing boundaries. This adds a literal passage representation: a list or
paragraph need not pretend to be a scalar proposition to reach the common gate.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import json
import re

MAX_SCAN_RECORDS = 12000
MAX_SCAN_CHARS = 8_000_000
MAX_SOURCE_CHARS = 160_000
MAX_PUBLIC_CHARS = 2100
MAX_PUBLIC_WORDS = 210


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def contents_reference(query):
    from document_query import named_document_reference
    from entity_sections import entity_reference
    if entity_reference(query):
        return None
    if not re.search(r'^(?:מה\s+(?:כתוב|מופיע|נכתב|יש)\s+ב|(?:הצג|הציגו|קרא|הקרא|פרט)\s+|(?:please\s+)?(?:read|show|reproduce)\s+|what\s+(?:is\s+(?:written\s+)?in|does)\s+)', query.strip(), re.I):
        return None
    # "What does [title] say?" is a contents request. A scoped fact request
    # such as "what does [title] say about revenue?" stays on normal synthesis.
    if re.search(r'\babout\b|לגבי|אודות|(?<!\w)על\s+', query, re.I):
        return None
    return named_document_reference(query)


def named_document_request(query):
    """Separate one declared source identity from its literal request body."""
    from document_query import named_document_reference
    reference = named_document_reference(query)
    if reference is None or not reference.title or reference.kind == "ambiguous":
        return None
    title = re.escape(reference.title)
    introduction = (r'(?:(?:\b(?:in|from|according\s+to)\s+)?(?:the\s+)?'
        r'(?:document|file|presentation|workbook|spreadsheet)\s+|'
        r'(?<!\w)(?:במסמך|מהמסמך|בקובץ|מהקובץ|במצגת|מהמצגת|בחוברת)\s+)')
    pattern = re.compile(r'(?P<intro>'+introduction+r')?'
        r'(?P<quote>["“״\'])?'+title+r'(?(quote)["”״\'])',re.I)
    matches = tuple(pattern.finditer(query))
    if len(matches) != 1:
        return None
    match = matches[0]
    if not (match['intro'] or match['quote'] or re.search(
            r'\.(?:pdf|docx?|pptx?|xlsx?|csv|tsv|txt|md|odt|ods|odp)$',reference.title,re.I)):
        return None
    content = (query[:match.start()]+' '+query[match.end():]).strip(' ,;:')
    return reference,content


def numbered_document_overview_request(query):
    """Bind a descriptive whole-document question to a literal numbered identity."""
    # A numbered financial document is an identity, not a request for its
    # amount. Admit only an unqualified descriptive question; its exact
    # document kind and number are checked again during source resolution.
    from financial_documents import document_identities
    from financial_semantics import financial_request, financial_query, monetary_question
    from document_query import DocumentReference
    value = str(query or '').strip()
    identities = document_identities(value)
    if (len(identities) == 1 and financial_request(value) is None
            and financial_query(value) is None and monetary_question(value) is None):
        identity = identities[0]
        before, after = value[:identity.span[0]], value[identity.span[1]:]
        title = value[slice(*identity.span)].strip()
        hebrew = bool(re.fullmatch(r'מה\s+מתואר\s*', before)
                      and title.startswith('ב'))
        english = bool(re.fullmatch(
            r'what\s+is\s+described\s+in\s+(?:the\s+)?', before, re.I))
        if (hebrew or english) and not after.strip(' ?!.,;:'):
            from financial_documents import primary_document_identities
            title = title[1:] if hebrew else title
            primary = primary_document_identities(title)
            if (len(primary) == 1 and
                    (primary[0].kind, primary[0].number) == (identity.kind, identity.number)):
                return DocumentReference('document', title), ''
    return None


def selected_source_summary_request(query):
    """Recognize a complete summary of a deictic selected source only."""
    value = str(query or '').strip()
    return bool(re.fullmatch(
        r'(?:סכם|סכמי|סכמו)\s+(?:את\s+)?(?:המסמך|הקובץ)\s+(?:הזה|הזאת)|'
        r'(?:please\s+)?summari[sz]e\s+(?:this|the\s+selected)\s+(?:document|file)',
        value.rstrip(' ?!.'), re.I))


def independently_verified_overview_request(query):
    from broad_material_discovery import parse_broad_material_request
    return bool(numbered_document_overview_request(query) or selected_source_summary_request(query)
                or parse_broad_material_request(query))


def named_summary_request(query):
    """Admit only anchored summary commands over a literal declared source."""
    from typed_summary import presentation_summary_reference
    from broad_material_discovery import parse_broad_material_request
    broad = parse_broad_material_request(query)
    if broad:
        from document_query import DocumentReference
        return DocumentReference('discovered_material', broad.subject), ''
    if selected_source_summary_request(query):
        from document_query import DocumentReference
        return DocumentReference('selected_source', ''), ''
    topical = presentation_summary_reference(query)
    if topical:
        return topical, ''
    numbered = numbered_document_overview_request(query)
    if numbered:
        return numbered
    request = named_document_request(query)
    if request is None:
        return None
    reference,content = request
    # A source-qualified question for the main points requests the same
    # whole-source summary as an imperative, without adding a source topic.
    if re.fullmatch(r'(?:מה|מהם|מהן|אילו)\s+(?:עיקרי\s+(?:הדברים|התוכן|המסמך|הקובץ)|'
            r'הנקודות\s+(?:העיקריות|המרכזיות)|הנושאים\s+(?:העיקריים|המרכזיים))\s*[?!.]*|'
            r'what\s+(?:are|is)\s+(?:the\s+)?(?:main|key)\s+(?:points?|themes?|ideas?)\s*[?!.]*',
            content, re.I):
        return reference, ''
    command = re.fullmatch(r'(?:please\s+)?(?:summari[sz]e|describe|'
        r'give\s+(?:a\s+)?(?:brief\s+)?summary\s+of|'
        r'(?<!\w)(?:סכם|סכמי|סכמו|תאר|תארי|תארו)(?!\w))\s*(?P<body>[\s\S]*)',content,re.I)
    return (reference,command['body'].strip(' ,;:')) if command else None


def unqualified_whole_summary(reference, body):
    """Distinguish a whole-source request from an owner/field summary."""
    if reference is None:
        return False
    if getattr(reference, 'kind', '') == 'presentation_topic':
        return True
    value = str(body or '').strip(' \t\r\n.,;:?!')
    return bool(re.fullmatch(r'(?:(?:את|the)\s+)?(?:ה?מסמך|ה?קובץ|document|file)?', value, re.I))


def operation(query, plan=None):
    from contact_intent import fast_contact_intent, validate_contact_interpretation
    proposed_person = getattr(plan, 'contact_person', '') if plan is not None else ''
    contact = (validate_contact_interpretation(query, proposed_person)
               if proposed_person else fast_contact_intent(query))
    if contact:
        return 'contact_lookup', contact
    from subject_intent import subject_request, SubjectRequest
    from financial_documents import document_type_listing
    # An explicit summary of a presentation is a source-reading operation,
    # even when an earlier conversation or a planner proposed a financial
    # operation. Its topic identifies the source, not a section to truncate.
    from typed_summary import presentation_summary_reference
    topical_summary = presentation_summary_reference(query)
    if topical_summary:
        return 'document_summary', topical_summary
    listing = getattr(plan, 'document_type_interpretation', None)
    if listing is not None:
        from financial_documents import validate_semantic_document_listing
        checked = validate_semantic_document_listing(query, {'operation': 'document_listing', **asdict(listing)})
        if checked == listing:
            return 'document_type_listing', listing
    listing = document_type_listing(query)
    if listing:
        return 'document_type_listing', listing
    planned_kind = getattr(plan, 'subject_operation', '')
    planned_subject = getattr(plan, 'subject_text', '')
    if planned_kind and planned_subject:
        return planned_kind, SubjectRequest(planned_kind, planned_subject)
    planned_money = getattr(plan, 'monetary_interpretation', None)
    if planned_money is not None:
        return 'financial_records', planned_money
    spec = subject_request(query)
    if spec:
        return spec.kind, spec
    from entity_sections import entity_reference
    from document_query import document_topic_reference
    from financial_documents import explicit_document_request
    document = explicit_document_request(query)
    if document:
        return 'document_reference', document
    topic = document_topic_reference(query)
    entity = entity_reference(query)
    if entity and not topic:
        return 'entity_sections', entity
    reference = contents_reference(query)
    # Explicit document reading owns its named identity even when its title
    # happens to contain a financial type and month.
    if not topic and reference and re.search(r'^(?:מה\s+(?:כתוב|מופיע|נכתב|יש)|קרא|הקרא|(?:please\s+)?(?:read|reproduce)|what\s+(?:is|does))|(?:תוכן\s+המסמך|\bcontents\s+of\b|\b(?:the\s+)?document\b)', query.strip(), re.I):
        return 'document_contents', reference
    summary = named_summary_request(query)
    if summary:
        return 'document_summary', summary[0]
    from financial_semantics import financial_query
    financial = financial_query(query)
    if financial:
        return 'financial_records', financial
    from financial_semantics import monetary_question
    monetary = monetary_question(query)
    if monetary:
        return 'financial_records', monetary
    if topic:
        return 'document_topic', topic
    reference = contents_reference(query)
    if reference:
        return 'document_contents', reference
    return None


def retained_summary_binding(contract):
    """Read the trusted server-owned prior source generation, if present."""
    prefix = 'retained_summary_source:'
    values = [item[len(prefix):] for item in contract.qualifiers if item.startswith(prefix)]
    if not values:
        return None
    if len(values) != 1:
        raise ValueError('invalid retained summary provenance')
    try:
        value = json.loads(values[0])
    except (ValueError, TypeError) as exc:
        raise ValueError('invalid retained summary provenance') from exc
    if (not isinstance(value, list) or len(value) != 3 or
            not re.fullmatch(r'[A-Za-z0-9_-]{43}', value[0] if isinstance(value[0], str) else '') or
            not re.fullmatch(r'[0-9a-f]{64}', value[1] if isinstance(value[1], str) else '') or
            not re.fullmatch(r'[0-9a-fA-F]{64}', value[2] if isinstance(value[2], str) else '')):
        raise ValueError('invalid retained summary provenance')
    return tuple(value)


def compile_reading_contract(query, plan, source_scope):
    spec = operation(query, plan)
    if not spec:
        return None
    from semantic_contracts import (RequestExecutionContract, TypedRequirement,
        CanonicalClause, SemanticValidationRecord)
    from request_understanding import explicit_requested_items
    kind, listing_spec = spec
    if kind == 'document_summary' and getattr(listing_spec, 'kind', '') == 'selected_source':
        from request_understanding import RequestScope
        if getattr(plan, 'scope', None) is not RequestScope.SELECTED_SOURCE or not getattr(plan, 'source_id', None):
            return None
    limit = explicit_requested_items(query)
    relation = 'summary' if kind in {'entity_sections', 'document_topic', 'document_summary', 'subject_work', 'project_overview'} else 'lookup' if kind in {'document_contents', 'document_reference', 'contact_lookup'} else 'list'
    subject = (getattr(spec[1], 'person', '') if kind == 'contact_lookup' else
               getattr(spec[1], 'subject', '') if kind in {'subject_documents', 'subject_work', 'project_overview'} else '')
    reference_current = str(getattr(plan, 'subject_reference_current', '') or '')
    reference_prior = str(getattr(plan, 'subject_reference_prior', '') or '')
    if bool(reference_current) != bool(reference_prior):
        raise ValueError('incomplete subject reference provenance')
    provenance = (('subject_reference_current:' + reference_current,
                   'subject_reference_prior:' + reference_prior)
                  if reference_current else ())
    retained = getattr(plan, 'retained_summary_provenance', ())
    if retained:
        if (kind != 'document_summary' or not selected_source_summary_request(query)
                or not isinstance(retained, tuple) or len(retained) != 3
                or retained[0] != getattr(plan, 'source_id', None)
                or not isinstance(retained[1], str) or not retained[1]
                or not re.fullmatch(r'[0-9a-fA-F]{64}', retained[2] if isinstance(retained[2], str) else '')):
            raise ValueError('invalid retained summary provenance')
        provenance += ('retained_summary_source:' + json.dumps((retained[0], digest(retained[1]), retained[2]), separators=(',', ':')),)
    payload = {'request': query, 'operation': kind, 'subject': subject, 'limit': limit,
               'source_scope': tuple(source_scope), 'source_id': getattr(plan, 'source_id', None),
               'subject_reference': provenance}
    identity = digest(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    periods = (spec[1].period,) if kind == 'financial_records' and spec[1].period else ()
    temporal = ('date_basis:'+spec[1].date_basis,) if kind == 'financial_records' and spec[1].date_basis else ()
    req = TypedRequirement(id='reading-1', clause_id='clause-1', kind=kind,
        subject=subject, predicate=kind, cardinality=limit or 0,
        periods=periods, qualifiers=temporal,
        output_slot=kind, value_type='source_passage', completeness='coverage_declared')
    clause = CanonicalClause('clause-1', query, relation, 'list', '', kind,
                             requested_cardinality=limit or 0)
    validation = SemanticValidationRecord(kind, (), False, 0, True, identity,
        (), (), (kind,), ('source_identity', 'coverage', 'literal_span'))
    scope = tuple(source_scope)
    source_id = getattr(plan, 'source_id', None)
    if source_id:
        scope = tuple(dict.fromkeys((*scope, source_id)))
    contract = RequestExecutionContract(contract_id='contract-'+identity[:24],
        contract_sha256=identity, schema_version='request-execution-contract-1.5.0',
        resolved_user_request=query, normalized_request=query,
        conversation_operation=getattr(getattr(plan, 'conversation_operation', None), 'value', 'NONE'),
        execution_route=('SUMMARY' if kind == 'project_overview' else 'DIRECT_LOOKUP'), support_status='supported', source_scope=scope,
        answer_scope=('overview' if kind == 'project_overview' or
                      (kind == 'document_summary' and (getattr(plan, 'answer_scope', '') == 'overview' or selected_source_summary_request(query)))
                      else 'targeted'), semantic_relation=relation, answer_shape='list', subjects=(subject,) if subject else (), predicates=(kind,),
        objects_or_metrics=(), requested_values=(), units=(), periods=periods, populations=(),
        polarity='neutral', qualifiers=('answer_language:'+getattr(getattr(plan, 'answer_language', None), 'value', 'en'), *temporal, *provenance,
            *((('document_listing_kind:'+listing_spec.kind,) if listing_spec.kind else ()) + ('document_listing_request:'+json.dumps({'operation':'document_listing', **asdict(listing_spec)}, ensure_ascii=False, sort_keys=True, separators=(',', ':')),) if kind == 'document_type_listing' and getattr(plan, 'document_type_interpretation', None) else (('document_listing_kind:'+listing_spec.kind,) if kind == 'document_type_listing' else ()))), requested_cardinality=limit or 0,
        completeness='coverage_declared', clauses=(clause,), requirements=(req,),
        semantic_validation=validation)
    identity = contract_digest(contract)
    return replace(contract, contract_id='contract-'+identity[:24], contract_sha256=identity)


def contract_digest(contract):
    value = contract.as_dict()
    value.pop('contract_id', None)
    value.pop('contract_sha256', None)
    return digest(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')))


def _technical_plan_binding(plan):
    from dataclasses import asdict
    return json.dumps(asdict(plan), ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def compile_technical_contract(query, plan, technical_plan, source_scope=()):
    """Bind the validated technical interpretation to the common contract."""
    from semantic_contracts import (RequestExecutionContract, TypedRequirement,
                                    CanonicalClause, SemanticValidationRecord)
    if technical_plan.question != query:
        raise ValueError('technical question mismatch')
    binding = _technical_plan_binding(technical_plan)
    kind = 'technical_specification'
    req = TypedRequirement(id='reading-1', clause_id='clause-1', kind=kind,
        subject=technical_plan.target, predicate=kind, cardinality=len(technical_plan.requirements),
        output_slot=kind, value_type='source_passage', completeness='all_requested_clauses')
    clause = CanonicalClause('clause-1', query, 'lookup', 'scalar', technical_plan.target, kind)
    validation = SemanticValidationRecord(kind, (), False, 0, True, digest(binding),
        (), (), (kind,), ('source_identity', 'coverage', 'literal_span'))
    scope = tuple(dict.fromkeys(source_scope))
    contract = RequestExecutionContract(contract_id='', contract_sha256='',
        schema_version='request-execution-contract-1.5.0', resolved_user_request=query,
        normalized_request=query,
        conversation_operation=getattr(getattr(plan, 'conversation_operation', None), 'value', 'NONE'),
        execution_route='DIRECT_LOOKUP', support_status='supported', source_scope=scope,
        semantic_relation='lookup', answer_shape='scalar', subjects=(technical_plan.target,),
        predicates=(kind,), objects_or_metrics=(), requested_values=(), units=(), periods=(),
        populations=(), polarity='neutral',
        qualifiers=('answer_language:'+getattr(getattr(plan, 'answer_language', None), 'value', 'en'),
                    'technical_plan:'+binding),
        requested_cardinality=len(technical_plan.requirements), completeness='all_requested_clauses',
        clauses=(clause,), requirements=(req,), semantic_validation=validation)
    identity = contract_digest(contract)
    return replace(contract, contract_id='contract-'+identity[:24], contract_sha256=identity)


@dataclass(frozen=True)
class SourcePassage:
    fact_id: str
    evidence_id: str
    requirement_id: str
    source_id: str
    tenant_id: str
    source_hash: str
    text: str
    span: tuple[int, int]
    span_hash: str
    source_claim: str
    citation_id: str
    metadata: dict
    fact_type: str = 'source_passage'
    value: str = ''
    unit: str = ''
    representation: str = 'indexed'


@dataclass(frozen=True)
class ScanWitness:
    rows: tuple[dict, ...]
    storage_complete: bool
    original_body: str = ''
    original_complete: bool = True


@dataclass(frozen=True)
class ReadingEvidence:
    passages: tuple[SourcePassage, ...]
    complete: bool
    examined_documents: int
    examined_chunks: int
    reason: str = ''
    witness: ScanWitness | None = None


@dataclass(frozen=True)
class SourceSummaryEvidence(ReadingEvidence):
    reference: object | None = None
    question: str = ''
    summary_request: str = ''
    summary_topic: str = ''
    summary_relation: str = 'identity'
    selection_complete: bool = True
    quote_eligible_ids: tuple[str, ...] | None = None
    overview_synthesis: bool = False
    overview_question: str = ''
    overview_claims: tuple = ()
    overview_review: object = None
    overview_tenant: str = ''
    pdf_document_root: str = ''
    pdf_tenant: str = ''
    pdf_page_hashes: tuple = ()


@dataclass(frozen=True)
class TechnicalReadingEvidence(ReadingEvidence):
    technical_plan: object = None
    sections: tuple = ()
    answer_json: str = ''
    review_json: str = ''
    document_root: str = ''
    tenant_id: str = ''
    allowed_source_ids: tuple[str, ...] = ()
    competing_source_ids: tuple[str, ...] = ()
    answer_language: str = 'en'
    selection_complete: bool = True


def _technical_decision(plan, sections, answer_json, review_json, allowed, competing):
    from technical_validation import validate_technical_answer, semantic_validator_from_response
    initial = validate_technical_answer(plan, answer_json, sections,
        allowed_source_ids=allowed, competing_source_ids=competing)
    if initial.reason != 'semantic_verification_required':
        raise ValueError(initial.reason)
    reviewer = semantic_validator_from_response(review_json, plan, initial.claims, sections)
    decision = validate_technical_answer(plan, answer_json, sections,
        allowed_source_ids=allowed, competing_source_ids=competing, semantic_validator=reviewer)
    if not decision.accepted:
        raise ValueError(decision.reason)
    return decision


def _technical_passages(decision, sections):
    by_id = {str(i): item for i, item in enumerate(sections, 1)}
    grouped = {}
    for claim in decision.claims:
        grouped.setdefault(claim.citation_id, []).append(claim.source_quote)
    result = []
    for citation_id, quotes in grouped.items():
        section = by_id[citation_id].section
        offsets = [(section.text.index(quote), section.text.index(quote)+len(quote)) for quote in quotes]
        start = min(pair[0] for pair in offsets)
        end = max(pair[1] for pair in offsets)
        claim = section.text[start:end]
        result.append(make_passage(section.text, claim, dict(section.metadata), section.chunk_id, citation_id))
    return tuple(result)


def make_technical_reading_evidence(technical_plan, sections, grouped, storage_complete,
                                    raw_answer, raw_review, *, document_root, tenant,
                                    allowed_source_ids=(), competing_source_ids=(), answer_language=''):
    """Bind model-reviewed claims to exact indexed section and source spans."""
    def frozen_json(value):
        from technical_validation import _object
        return json.dumps(_object(value),
                          ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    answer_json, review_json = frozen_json(raw_answer), frozen_json(raw_review)
    sections = tuple(sections)
    allowed, competing = tuple(allowed_source_ids), tuple(competing_source_ids)
    decision = _technical_decision(technical_plan, sections, answer_json, review_json, allowed, competing)
    language = str(answer_language or ('he' if re.search(r'[\u05d0-\u05ea]',technical_plan.question) else 'en')).casefold()
    if language not in {'he','en'}:
        raise ValueError('invalid_answer_language')
    facts = _technical_passages(decision, sections)
    rows = tuple(row for source_rows in grouped.values() for row in source_rows)
    evidence = TechnicalReadingEvidence(tuple(facts), bool(storage_complete), len(grouped),
        len(rows), '', ScanWitness(rows, bool(storage_complete)), technical_plan, sections,
        answer_json, review_json, str(document_root), tenant, allowed, competing,
        language, not bool(decision.missing))
    from technical_validation import render_technical_answer
    answer = render_technical_answer(technical_plan, decision, language)
    return evidence, answer


def validate_technical_reading_evidence(evidence, contract, answer):
    """Recompute source hierarchy, claim checks, and rendered text at release."""
    from numbered_sections import numbered_source_sections
    if not isinstance(evidence, TechnicalReadingEvidence) or not evidence.witness:
        return False
    if ('technical_plan:'+_technical_plan_binding(evidence.technical_plan) not in contract.qualifiers
            or evidence.technical_plan.question != contract.normalized_request
            or 'answer_language:'+evidence.answer_language not in contract.qualifiers):
        return False
    grouped = {}
    for row in evidence.witness.rows:
        grouped.setdefault(row.get('metadata', {}).get('source_id'), []).append(row)
    used_sources = {item.section.metadata.get('source_id') for item in evidence.sections}
    try:
        rebuilt = tuple(item for source in used_sources for item in numbered_source_sections(
            grouped.get(source, ()), tenant=evidence.tenant_id, document_root=evidence.document_root))
        if any(item not in rebuilt for item in evidence.sections):
            return False
        decision = _technical_decision(evidence.technical_plan, evidence.sections,
            evidence.answer_json, evidence.review_json, evidence.allowed_source_ids,
            evidence.competing_source_ids)
        from technical_validation import render_technical_answer
        expected = render_technical_answer(evidence.technical_plan, decision, evidence.answer_language)
        expected_facts = _technical_passages(decision, evidence.sections)
        return bool(evidence.complete and evidence.selection_complete == (not bool(decision.missing))
                    and answer == expected and evidence.passages == expected_facts)
    except (ValueError, TypeError, KeyError):
        return False


def passage_valid(fact):
    from source_access import source_metadata_complete
    if not isinstance(fact, SourcePassage) or not source_metadata_complete(fact.metadata, fact.tenant_id):
        return False
    if not isinstance(fact.span, (tuple, list)) or len(fact.span) != 2:
        return False
    a, b = fact.span
    return bool(type(a) is int and type(b) is int and 0 <= a < b <= len(fact.text)
                and fact.text[a:b] == fact.source_claim
                and digest(fact.source_claim) == fact.span_hash
                and fact.source_id == fact.metadata['source_id']
                and fact.source_hash == fact.metadata['source_sha256']
                and fact.evidence_id and fact.fact_id and fact.citation_id)


def scan_authorized(retriever, tenant, document_root, *, where=None,
                    record_limit=MAX_SCAN_RECORDS, char_limit=MAX_SCAN_CHARS):
    """Paginate existing storage; never fill the model context with the corpus."""
    from source_authorization import authorize_source_candidate
    hits, complete = retriever.scan_records(limit=record_limit, scan_limit=min(record_limit + 1, 1_000_000), where=where)
    grouped = {}
    count = 0
    size = 0
    authorization = {}
    for hit in hits:
        text = str(hit.get('document') or '')
        meta = dict(hit.get('metadata') or {})
        source = meta.get('source_id', '')
        size += len(text)
        if char_limit is not None and size > char_limit:
            complete = False
            break
        # A storage adapter ignoring its filter must never introduce evidence.
        if meta.get('tenant_id') != tenant or (where and meta.get('source_relative_path') != where['source']):
            complete = False
            continue
        key = (source, meta.get('source_sha256'), meta.get('source_size'), meta.get('source_mtime_ns'))
        if key not in authorization:
            authorization[key] = authorize_source_candidate(document_root, tenant, meta,
                candidate_text=text, candidate_id=str(hit.get('id') or '')).authorized
        if not authorization[key]:
            complete = False
            continue
        from source_access import source_metadata_complete
        if not source_metadata_complete(meta, tenant):
            complete = False
            continue
        grouped.setdefault(source, []).append({**hit, 'metadata': meta})
        count += 1
    for rows in grouped.values():
        rows.sort(key=lambda hit: (hit['metadata'].get('chunk_index', 0), str(hit.get('id', ''))))
    return grouped, complete, count


def source_body(rows):
    # Splitter breadcrumbs are metadata, never business dates or source prose.
    parts = []
    previous = ''
    for hit in rows:
        body = re.sub(r'^\[[^\]\n]+\]\s*\n+', '', hit['document']).strip()
        if not body or body == previous:
            continue
        # Adjacent chunk overlap is copied text, not a second financial event.
        overlap = next((n for n in range(min(len(previous), len(body)), 19, -1)
                        if previous.endswith(body[:n])), 0)
        parts.append(body[overlap:].lstrip() if overlap else body)
        previous = body
    return '\n\n'.join(parts)


def make_passage(text, claim, metadata, chunk_id, citation_id, *, source_start=None):
    start = text.index(claim) if source_start is None else source_start
    if (type(start) is not int or start < 0 or not claim
            or text[start:start+len(claim)] != claim):
        raise ValueError('source passage offset does not match source bytes')
    return SourcePassage('passage-'+digest(text+str(citation_id))[:24],
        str(chunk_id), 'reading-1', metadata['source_id'], metadata['tenant_id'],
        metadata['source_sha256'], text, (start, start+len(claim)), digest(claim),
        claim, str(citation_id), dict(metadata))


def bounded_excerpt(text, chars, words):
    tokens = list(re.finditer(r'\S+', text))
    end = min(len(text), chars)
    if len(tokens) > words:
        end = min(end, tokens[max(0, words-1)].end() if words else 0)
    if end < len(text):
        # Prefer complete source paragraphs; retain an honest excerpt when
        # one paragraph itself exceeds the public quotation bound.
        boundary = text.rfind('\n', 0, end)
        if boundary > end//2:
            end = boundary
    return text[:end]


def passage_segments(text):
    start = 0
    count = 0
    for token in re.finditer(r'\S+(?:\s+|$)', text):
        if count and (token.end()-start > 570 or count >= 70):
            yield text[start:token.start()]
            start = token.start()
            count = 0
        count += 1
    if start < len(text):
        yield text[start:]


def summary_sentence_boundaries(text):
    """Conservative sentence and explicit sibling-bullet source boundaries."""
    boundaries = []
    for match in re.finditer(r'(?<=[.!])\s+', text):
        start, end = match.span()
        if end >= len(text):
            continue
        previous = re.search(r'(\S+)$', text[:start])
        word = previous.group(1).casefold() if previous else ''
        if re.fullmatch(r'\d+\.', word) or word in {
            'mr.', 'mrs.', 'ms.', 'dr.', 'prof.', 'sr.', 'jr.', 'etc.', 'e.g.', 'i.e.'
        }:
            continue
        boundaries.append(end)
    # The slide extractor preserves bullet markers as their own lines. Each
    # subsequent marker starts a new source item; wrapped lines stay inside
    # their item. Keep the heading with its first bullet and never detach an
    # unfinished condition, parent heading, or comma continuation.
    markers = list(re.finditer(r'(?m)^[ \t]*[•*](?=[ \t\r\n]|$)[ \t]*(?:\r?\n)?', text))
    for previous, current in zip(markers, markers[1:]):
        item = text[previous.end():current.start()].strip()
        if not item or item.endswith((':', ',', '–', '—', '-')):
            continue
        condition = re.match(r'(?i)^(?:(?:only\s+)?(?:if|when|until)|unless|'
            r'provided\s+that|subject\s+to|as\s+long\s+as|without|except)\b|'
            r'^(?:רק\s+|אלא\s+)?(?:אם|כאשר|עד|ללא|בלי|בכפוף|למעט|בתנאי)(?!\w)', item)
        if condition:
            if ',' not in item or len(item.split(',', 1)[1].split()) < 2:
                continue
        if re.search(r'(?i)(?:\b(?:if|unless|except|without|when)|'
                     r'(?<!\w)(?:אם|כאשר|ללא|למעט))(?:\s+\S+){0,1}\s*$', item):
            continue
        boundaries.append(current.start())
    return sorted(set(boundaries))


def summary_passage_segments(text):
    """Keep complete sentence units inside the existing per-quote limits."""
    from evidence_quotes import HARD_QUOTE_MAX_CHARS, HARD_QUOTE_MAX_WORDS, _word_spans
    boundaries = [0, *summary_sentence_boundaries(text), len(text)]
    start, end = 0, 0
    for left, right in zip(boundaries, boundaries[1:]):
        unit = text[left:right]
        if len(unit) > HARD_QUOTE_MAX_CHARS or len(_word_spans(unit)) > HARD_QUOTE_MAX_WORDS:
            raise ValueError('summary_sentence_exceeds_quote_bound')
        candidate = text[start:right]
        if end > start and (len(candidate) > HARD_QUOTE_MAX_CHARS
                            or len(_word_spans(candidate)) > HARD_QUOTE_MAX_WORDS):
            yield text[start:end]
            start = left
        end = right
    if end > start:
        yield text[start:end]


def summary_pdf_record_segments(text, rows):
    """Partition an indexed record-like PDF page only at authenticated flow boundaries.

    The full reconstructed page remains the text of every resulting passage.
    This fallback never splits an indexed record or a packing group and refuses
    ambiguous continuation/qualification boundaries. It is used only for a
    whole numbered-document overview after ordinary sentence splitting fails.
    """
    from evidence_quotes import HARD_QUOTE_MAX_CHARS, HARD_QUOTE_MAX_WORDS, _word_spans
    if len(rows) < 5 or source_body(rows) != text:
        raise ValueError('summary_pdf_layout_unavailable')
    indexed = []
    previous = ''
    for row in rows:
        meta = row.get('metadata') or {}
        if (meta.get('mime_type') != 'application/pdf' or
                not isinstance(meta.get('pdf_flow_id'), str) or
                not isinstance(meta.get('pdf_packing_group_id'), str) or
                meta.get('pdf_packing_revision') != 'pdf-paragraph-groups-v1' or
                not isinstance(meta.get('chunk_index'), int)):
            raise ValueError('summary_pdf_layout_unavailable')
        original = re.sub(r'^\[[^\]\n]+\]\s*\n+', '', str(row.get('document') or '')).strip()
        if not original or original == previous:
            previous = original
            continue
        overlap = next((n for n in range(min(len(previous), len(original)), 19, -1)
                        if previous.endswith(original[:n])), 0)
        piece = original[overlap:].lstrip() if overlap else original
        if not piece:
            raise ValueError('summary_pdf_layout_unavailable')
        indexed.append((piece, meta))
        previous = original
    if not indexed or '\n\n'.join(piece for piece, _ in indexed) != text:
        raise ValueError('summary_pdf_layout_unavailable')
    if any(right['chunk_index'] != left['chunk_index'] + 1
           for (_, left), (_, right) in zip(indexed, indexed[1:])):
        raise ValueError('summary_pdf_layout_unavailable')
    if len({m.get('page') for _, m in indexed}) != 1 or len({m.get('source_id') for _, m in indexed}) != 1:
        raise ValueError('summary_pdf_layout_unavailable')
    # Dense short labels and several numeric-bearing records establish a
    # form-like layout. Merely splitting arbitrary long prose into indexed
    # chunks is insufficient authority for this fallback.
    labels = sum(1 for value, _ in indexed if len(value.split()) <= 3
                 and re.search(r'[A-Za-zא-ת]', value) and not re.search(r'\d', value))
    values = sum(1 for value, _ in indexed if re.search(r'\d', value))
    if labels < 4 or values < 2:
        raise ValueError('summary_pdf_layout_unavailable')

    # A split before/after linked packing or flow pieces would change their
    # owner. Similarly, ordinary wrapped text and material conditional
    # continuations cannot be detached from their governing statement.
    leading_continuation = re.compile(
        r'^(?:and|or|but|if|unless|except|when|while|only|provided|subject\s+to|'
        r'without|with|including|excluding|therefore|however|'
        r'(?:ו|ש)?(?:אם|כאשר|אלא|למעט|ללא|בלי|בכפוף|בתנאי|רק)|וכן|או|אבל)(?:\s|$)', re.I)
    trailing_continuation = re.compile(
        r'(?:[,;:–—-]|\b(?:and|or|if|unless|except|when|subject\s+to)|'
        r'(?<!\w)(?:אם|כאשר|אלא|למעט|ללא|בכפוף|או|וכן))\s*$', re.I)
    leading_value = re.compile(r'^(?:[₪$€£]|\d)')
    leading_condition = re.compile(r'^(?:(?:ו|ש)?(?:אחריות|תנאי)|warranty|condition)(?:\s|$)', re.I)
    def boundary_safe(left, lm, right, rm):
        if (lm.get('pdf_packing_group_id') == rm.get('pdf_packing_group_id')
                or lm.get('pdf_flow_id') == rm.get('pdf_flow_id')
                or lm.get('next_same_pdf_flow') is True or rm.get('previous_same_pdf_flow') is True
                or lm.get('next_same_pdf_packing_group') is True
                or rm.get('previous_same_pdf_packing_group') is True):
            return False
        l, r = left.strip(), right.strip()
        if (not l or not r or len(l.split()) <= 3 or trailing_continuation.search(l)
                or leading_continuation.match(r) or leading_value.match(r)
                or leading_condition.match(r)):
            return False
        return True

    # Keep the entire source text and source order; no content is dropped.
    offsets, cursor = [0], 0
    for index, (piece, _) in enumerate(indexed):
        cursor += len(piece)
        if index + 1 < len(indexed):
            cursor += 2
            offsets.append(cursor)
    offsets.append(len(text))
    segments = []
    start_index = 0
    while start_index < len(indexed):
        last_fit, safe = start_index, None
        for end_index in range(start_index + 1, len(indexed) + 1):
            segment = text[offsets[start_index]:offsets[end_index]]
            if len(segment) > HARD_QUOTE_MAX_CHARS or len(_word_spans(segment)) > HARD_QUOTE_MAX_WORDS:
                break
            last_fit = end_index
            if end_index == len(indexed) or boundary_safe(
                    indexed[end_index-1][0], indexed[end_index-1][1],
                    indexed[end_index][0], indexed[end_index][1]):
                safe = end_index
        if safe is None or safe <= start_index:
            raise ValueError('summary_pdf_layout_unavailable')
        segments.append(text[offsets[start_index]:offsets[safe]])
        start_index = safe
    if ''.join(segments) != text:
        raise ValueError('summary_pdf_layout_unavailable')
    return tuple(segments)


def collect_contents(retriever, tenant, document_root, where, *, summary_request=None):
    grouped, complete, count = scan_authorized(retriever, tenant, document_root, where=where)
    if len(grouped) != 1:
        return ReadingEvidence((), False, len(grouped), count, 'source_missing_or_ambiguous')
    rows = next(iter(grouped.values()))
    storage_complete = complete
    text = source_body(rows)
    meta = dict(rows[0]['metadata'])
    # Read the authenticated original paragraphs for short DOCX sources. This
    # retains unusual words, empty-line positions, headings and trailing items.
    original_body = ''
    original_complete = True
    if meta['mime_type'].endswith('wordprocessingml.document'):
        from source_access import docx_source_body
        original_body, original_complete = docx_source_body(document_root, tenant, meta)
        if original_body:
            text = original_body
        complete = complete and original_complete
    if summary_request is not None:
        summary = _summary_from_rows(rows, storage_complete, count, original_body,
                                     original_complete, summary_request)
        if (meta.get('mime_type') == 'application/pdf'
                and independently_verified_overview_request(summary_request[0])):
            page_hashes = _verify_original_pdf_summary(rows, document_root, tenant)
            if not page_hashes:
                return replace(summary, passages=(), complete=False, selection_complete=False,
                    reason='summary_original_pdf_mismatch',
                    pdf_document_root=str(document_root), pdf_tenant=str(tenant))
            summary = replace(summary, pdf_document_root=str(document_root),
                              pdf_tenant=str(tenant), pdf_page_hashes=page_hashes)
        return summary
    return _contents_from_rows(rows, complete, count, original_body, original_complete)


def _contents_from_rows(rows, complete, count, original_body='', original_complete=True,
                        *, selected_text=None, source_topic='',
                        public_chars=MAX_PUBLIC_CHARS, public_words=MAX_PUBLIC_WORDS,
                        preserve_summary_sentences=False, layout_record_summary=False):
    """Rebuild the bounded literal selection from its complete source witness."""
    text = (original_body or source_body(rows)) if selected_text is None else selected_text
    meta = dict(rows[0]['metadata'])
    if source_topic:
        meta['_summary_topic'] = source_topic
    if not text:
        return ReadingEvidence((), False, 1, count, 'source_exceeds_reading_bound')
    # Bound public quotations without pretending this excerpt is a full read.
    excerpt = bounded_excerpt(text, public_chars, public_words)
    if excerpt != text:
        text = excerpt
        complete = False
    passages = []
    sections = [(text, meta, rows[0].get('id') or meta.get('chunk_id'))]
    location_key = ('page' if meta['mime_type'] == 'application/pdf' else 'slide'
                    if meta['mime_type'].endswith('presentationml.presentation') else None)
    if location_key:
        pages = {}
        for row in rows:
            pages.setdefault(row['metadata'].get(location_key), []).append(row)
        sections = [(source_body(page_rows), dict(page_rows[0]['metadata']), page_rows[0].get('id')) for page_rows in pages.values()]
        if selected_text is not None:
            sections = [(selected_text,dict(metadata,_summary_topic=source_topic),chunk_id)
                        for body,metadata,chunk_id in sections if selected_text in body]
            if len(sections) != 1:
                return ReadingEvidence((),False,1,count,'summary_topic_not_location_bound',
                    ScanWitness(tuple(rows),bool(complete),original_body,original_complete))
    # Split at actual source boundaries to respect existing quote limits.
    remaining = public_chars
    words_remaining = public_words
    for body, metadata, chunk_id in sections:
        excerpt = bounded_excerpt(body, remaining, words_remaining)
        if excerpt != body:
            body = excerpt
            complete = False
        if not body:
            complete = False
            break
        remaining -= len(body)
        words_remaining -= len(body.split())
        try:
            if preserve_summary_sentences:
                try:
                    segments = tuple(summary_passage_segments(body))
                except ValueError:
                    if not (layout_record_summary and location_key == 'page'):
                        raise
                    segments = summary_pdf_record_segments(body, pages.get(metadata.get('page'), ()))
            else:
                segments = tuple(passage_segments(body))
        except ValueError:
            return ReadingEvidence((), False, 1, count, 'summary_sentence_exceeds_quote_bound',
                ScanWitness(tuple(rows), bool(complete), original_body, original_complete))
        source_cursor = 0
        for segment in segments:
            position = body.find(segment, source_cursor)
            if position < 0:
                raise ValueError('source passage segment is not in source order')
            source_cursor = position + len(segment)
            if not segment.strip():
                continue
            metadata['section'] = 'Document body' if meta['mime_type'].endswith('wordprocessingml.document') else (metadata.get('section') or 'Indexed source text')
            passage = make_passage(body, segment, metadata, chunk_id, len(passages)+1,
                                   source_start=position)
            if original_body:
                passage = replace(passage, representation='docx_original')
            passages.append(passage)
    return ReadingEvidence(tuple(passages), complete, 1, count, '' if complete else 'partial_source_excerpt', ScanWitness(tuple(rows), bool(complete), original_body, original_complete))


def collect_summary(retriever, tenant, document_root, query, where):
    request = named_summary_request(query)
    if request is None:
        raise ValueError('named summary request is not bound')
    return collect_contents(retriever,tenant,document_root,where,
                            summary_request=(query,*request))


_SUMMARY_ROLE = (r'project|company|program|programme|site|system|team|department|'
                 r'פרויקט|פרוייקט|מיזם|חברה|אתר|מערכת|צוות|מחלקה')


def _summary_role_key(value):
    """Canonicalize only closed role/field grammar, never an owner's name."""
    value = value.casefold().strip()
    value = re.sub(r'\s+name$|^שם\s+ה?', '', value)
    return {'פרויקט':'project','פרוייקט':'project','חברה':'company','programme':'program',
        'אתר':'site','מערכת':'system','צוות':'team','מחלקה':'department'}.get(value,value)


def _summary_owner_request(body):
    """Recognize a bounded named owner, retaining its literal name and role."""
    pattern = (r'(?<!\w)(?:[בלמכ]ה?)?(?P<role>'+_SUMMARY_ROLE+r')\s+(?:(?P<genitive>of|של)\s+)?'
        r'(?P<name>[^:;,?!\n]{1,100})(?=[:;,?!\n]|$)')
    from request_understanding import _quoted_request_spans
    quoted = _quoted_request_spans(body)
    owners, name_spans, unresolved = [], [], []
    # Lookahead retains later explicit owners even inside the unknown tail of
    # a skipped interrogative ("which site in Project Cedar ...").
    for match in re.finditer('(?='+pattern+')',body,re.I):
        if any(a <= match.start() < b for a,b in name_spans):
            continue
        enclosing = next(((a,b) for a,b in quoted if a <= match.start() < b),None)
        if enclosing and body[enclosing[0]+1:match.start()].strip():
            continue
        relation = ('project_customer' if match['genitive'] and _summary_role_key(match['role'])=='project'
                    else 'identity')
        literal = re.match(r'''(?:"(?P<double>[^"\n]{1,100})"|“(?P<curly>[^”\n]{1,100})”|'''
            r'''״(?P<hebrew>[^״\n]{1,100})״|'(?P<single>[^'\n]{1,100})')''',body[match.start('name'):])
        literal_name = (body[match.start('name'):enclosing[1]-1].strip() if enclosing else
            next((value for value in literal.groupdict().values() if value is not None),None) if literal else None)
        literal_stop = enclosing[1]-1 if enclosing else match.start('name')+literal.end() if literal else None
        quoted_name = match['name'].lstrip()[:1] in {'"','“','״',"'"}
        prefix = body[:match.start()].rstrip()
        interrogative = re.search(r'(?<!\w)(?:(?:what|which)(?:\s+(?:(?:is|was|are|were|does|do|did|has|have|had)\s+)?the)?|'
            r'ו?[בלמכ]?(?:איזה|איזו|אילו)|ו?(?:מהו|מהי|מהם|מהן|מי))$',prefix,re.I)
        tail = match['name'].strip()
        attribute_tail = re.match(r'^(?:is|are|was|were|does|do|did|has|have|had)(?![\w./-])|'
            r'^(?:the|this|that)\s+(?:proposal|offer|document|report|file|presentation)\s+'
            r'(?:concerns?|refers?|relates?|describes?|mentions?|covers?|is|was)(?![\w./-])|'
            r'^(?:described|mentioned|listed|specified|stated)\s+(?:in|by)\s+(?:the|this|that)\s+'
            r'(?:proposal|offer|document|report|file|presentation)(?!\w)|'
            r'^(?:and\s+(?:the\s+)?(?:work|services?|tasks?|activities)\s+)?'
            r'(?:include|includes|involve|involves|contain|contains|provide|provides|cover|covers)(?!\w)|'
            r'^(?:(?:for|in|of|at)\s+(?:the\s+)?|[בלמכ]ה?)(?:'+_SUMMARY_ROLE+r')\s+\S|'
            r'^(?:(?:הוא|היא|הם|הן)\s+)?(?:מתייחס(?:ת|ים|ות)?|מתואר(?:ת|ים|ות)?|'
            r'מופיע(?:ה|ים|ות)?|מצוין(?:ת|ים|ות)?)(?:\s+|$)',tail,re.I)
        field_question = (re.search(r'(?<!\w)ו?מה\s+(?:כלל|כללה|כללו|כולל|כוללת|כוללים|כוללות)\s*$',prefix)
            and re.match(r'^(?:ו?(?:מה|מהו|מהי|מהם|מהן|אילו)|ו?השירותים)(?!\w)',tail))
        if literal_name is None and not quoted_name and relation != 'project_customer':
            if field_question or interrogative and attribute_tail:
                continue
            if interrogative:
                unresolved.append((match['role'],'',relation))
                continue
        name = literal_name if literal_name is not None else match['name'].strip(' ."״“”')
        # Quoted owner names are literal identity, not interrogative syntax.
        # Unknown unquoted instruction tails remain unresolved, never global.
        if not name or len(name)>100 or literal_name is None and (quoted_name
                or re.search(r'\b(?:and|when|what|which|how)\b|(?<!\w)(?:ומה|ומתי|וכיצד)(?!\w)',name,re.I)):
            unresolved.append((match['role'],'',relation))
            continue
        stop = literal_stop if literal_stop is not None else match.end('name')
        name_spans.append((match.start('name'),stop))
        owners.append((match['role'],name,relation))
    if unresolved:
        return unresolved[0]
    if not owners:
        return '', '', 'identity'
    if len({(_summary_role_key(role),' '.join(name.split()),relation) for role,name,relation in owners}) != 1:
        return owners[0][0],'',owners[0][2]
    return owners[0]


def _summary_topic_request(body):
    return _summary_owner_request(body)[:2]


def _summary_header_role(label):
    """Recognize source-owned identity and commissioning-role field grammar."""
    key = _summary_role_key(label)
    if key in {_summary_role_key(role) for role in _SUMMARY_ROLE.split('|')}:
        return key
    customer = (r'(?:(?:name|identity)\s+of\s+(?:the\s+)?|(?:project|work|services?)\s+)?'
        r'(?:(?:commissioning|ordering|requesting)\s+(?:body|entity|organization|organisation|party)|'
        r'customer|client|commissioner)(?:\s+(?:name|for\s+(?:the\s+)?(?:project|work|services?)))?')
    customer_he = (r'(?:שם\s+(?:של\s+)?)?(?:ה?(?:גוף|גורם)\s+)?'
        r'(?:ה?מזמין|ה?לקוח)(?:\s+(?:את\s+)?ה?(?:עבודה|עבודות|שירות|שירותים|פרויקט|פרוייקט))?')
    return 'customer' if re.fullmatch(customer+'|'+customer_he,label.strip(),re.I) else ''


def _summary_section(text,role,topic,relation='identity'):
    """Bind a named topic to one literal paragraph section or table record."""
    if not role:
        return text, ''
    if not topic:
        return '', ''
    from document_query import _identity_words
    from source_structure import _looks_heading
    wanted = _identity_words(topic)
    def same(value):
        return _identity_words(value) == wanted
    def named_start(value,*,content_fields=False):
        literal = re.match(re.escape(topic)+r'(?!\w)',value,re.I)
        if not literal:
            return None
        following = value[literal.end():].lstrip()
        if content_fields:
            # A repeated owner can introduce another content field. This
            # recognition only triggers ambiguity; it never broadens which
            # initial source name may establish the requested owner.
            following = re.sub(r'^(?:work|services?|period|years?|dates?|עבודות|עבודה|'
                r'שירותים|שירות|תקופה|שנים|שנה|תאריך)\s+','',following,flags=re.I)
        # The name must end at a real heading boundary or source predicate.
        # A longer proper name such as Cedar Grove is a different owner.
        predicate = (r'(?:is|are|was|were|has|have|had|operated|operates|ran|started|began|'
            r'ended|completed|appeared|appears|provided|provides|included|includes|contains|supports|'
            r'פעל|פעלה|פעלו|היה|הייתה|היו|בוצע|בוצעה|החל|החלה|הסתיים|הסתיימה|'
            r'כלל|כללה|כולל|כוללת|סיפק|סיפקה)(?!\w)')
        return literal if not following or following[0] in ':;,.!?–—-' or re.match(predicate,following,re.I) else None
    blocks = list(re.finditer(r'\S[\s\S]*?(?=\n[ \t]*\n|\Z)',text))
    admitted = []
    source_role = r'(?:the\s+|ה)?(?:'+_SUMMARY_ROLE+r')\s+(?:(?:of|של)\s+)?'
    owner = re.compile(r'^\s*#*\s*'+source_role,re.I)
    for index,block in enumerate(blocks):
        value = block.group().strip()
        table = bool(re.match(r'^Table row \d+:\s*\n',value))
        if table:
            anchors = []
            fields = list(re.finditer(r'^[-*•]\s+(?P<label>[^:\n]+):\s*(?P<value>[^\n]+)$',value,re.M))
            wanted_role = 'customer' if relation=='project_customer' else _summary_role_key(role)
            primary = [match for match in fields if _summary_header_role(match['label'])==wanted_role]
            if relation=='project_customer' and not any(re.search(
                    r'\b(?:project|work|services?|experience)\b|(?<!\w)ה?(?:פרויקט|פרוייקט|עבודה|עבודות|שירות|שירותים|ניסיון)(?!\w)',
                    match['label'],re.I) for match in fields):
                continue
            # Role identity is mandatory. A generic Name/contact field cannot
            # overrule the project's identity or establish a commissioning role.
            names = []
            for match in primary:
                literal = match['value'].strip()
                prefix = owner.match(literal)
                name = literal[prefix.end():] if prefix else literal
                prefix_role = re.search(_SUMMARY_ROLE,prefix.group(),re.I).group() if prefix else ''
                normalized_name = (name if relation=='identity' and prefix
                    and _summary_role_key(prefix_role)==_summary_role_key(role) else literal)
                names.append(_identity_words(normalized_name))
                if same(literal):
                    anchors.append(literal)
                elif (relation=='identity' and prefix
                        and _summary_role_key(prefix_role)==_summary_role_key(role) and same(name)):
                    # Keep the actual name span, not a request-supplied role
                    # or a globally stemmed spelling, as the source owner.
                    anchors.append(name)
            if len(set(names))>1:
                if wanted in names:
                    return '', ''
                continue
            if anchors:
                admitted.append((block.start(),block.end(),anchors[0]))
            continue
        head = owner.match(value)
        if not head:
            # A bare matching name is not proof of the requested role or of
            # a project-to-customer relationship.
            continue
        else:
            source_kind = re.search(_SUMMARY_ROLE,head.group(),re.I).group()
            if _summary_role_key(source_kind) != _summary_role_key(role):
                continue
            if _summary_role_key(role)=='project' and bool(re.search(r'\bof\b|(?<!\w)של(?!\w)',head.group(),re.I)) != (relation=='project_customer'):
                continue
            literal = named_start(value[head.end():])
            if not literal:
                continue
            anchor = literal.group()
            # A peer introduction in the same paragraph is a real ownership
            # boundary too. Do not grant this owner's context to its facts.
            peers = list(re.finditer(r'(?:[.!?;,]\s*|\s+(?:and|ו)\s*|\s+ו(?='+source_role+r'))'
                +source_role,value,re.I))
            if peers:
                if any(named_start(value[peer.end():],content_fields=True) for peer in peers):
                    # A repeated requested owner is an additional matching
                    # section, not proof that the first prefix was complete.
                    return '', ''
                admitted.append((block.start(),block.start()+peers[0].start(),anchor))
                continue
        stop = block.end()
        # A new peer owner, top-level heading or complete table record owns
        # its own facts. It cannot borrow this section's topic context.
        for following in blocks[index+1:]:
            child = following.group().strip()
            reference_intro = (r'(?<![\w./-])(?:(?:for|of|by|at|in|to|from)\s+|'
                r'(?:עבור|של)\s+|[לבמ](?='+source_role+r'))')
            references = [item for item in re.finditer(reference_intro+source_role,child,re.I)
                if _summary_role_key(re.search(_SUMMARY_ROLE,item.group(),re.I).group())
                   == _summary_role_key(role)]
            if any(not named_start(child[item.end():]) for item in references):
                break
            child_field = bool(re.fullmatch(r'#*\s*(?:work|services?|work\s+and\s+services|'
                r'period|years?|dates?|עבודות|עבודה|שירותים|שירות|עבודות\s+ושירותים|'
                r'תקופה|שנים|שנה|תאריך)\s*:?',child,re.I))
            if (owner.match(child) or re.match(r'^Table row \d+:',child)
                    or _looks_heading(child) and not (references or child_field)):
                break
            stop = following.end()
        admitted.append((block.start(),stop,anchor))
    if len(admitted) != 1:
        return '', ''
    start,stop,anchor = admitted[0]
    return text[start:stop],anchor


def _summary_from_rows(rows,storage_complete,count,original_body,original_complete,request):
    query,reference,body = request
    role,topic,relation = _summary_owner_request(body)
    full_text = original_body or source_body(rows)
    selected,source_topic = _summary_section(full_text,role,topic,relation)
    if getattr(reference, 'kind', '') == 'discovered_material':
        # The owner and topic witness proposes only a bounded literal passage.
        # Do not summarize the rest of a source that may contain other topics,
        # versions or configurations. The same selection is rebuilt by the
        # final summary validator from its full original-verified witness.
        from broad_material_discovery import (parse_broad_material_request,
            original_matches_broad_material, select_broad_material_excerpt)
        requested = parse_broad_material_request(query)
        match = original_matches_broad_material(requested, full_text)
        pages = {}
        for row in rows:
            pages.setdefault(row['metadata'].get('page'), []).append(row)
        matching_pages = [source_body(group) for group in pages.values()
                          if match and match.topic_witness in source_body(group)]
        selected = (select_broad_material_excerpt(requested, matching_pages[0], match,
                    full_text=full_text)
                    if len(matching_pages) == 1 else '')
        source_topic = match.topic_witness if match else ''
    whole_source = unqualified_whole_summary(reference, body)
    evidence = _contents_from_rows(rows,bool(storage_complete and original_complete),count,
        original_body,original_complete,selected_text=selected if role or getattr(reference, 'kind', '') == 'discovered_material' else None,source_topic=source_topic,
        public_chars=MAX_SCAN_CHARS if whole_source else MAX_PUBLIC_CHARS,
        public_words=MAX_SCAN_CHARS if whole_source else MAX_PUBLIC_WORDS,
        preserve_summary_sentences=whole_source,
        layout_record_summary=bool(whole_source and independently_verified_overview_request(query)))
    if not selected:
        evidence = replace(evidence,reason='summary_topic_not_found')
    # This witness remains the full authorized source, including peer owners
    # and any global prefix omitted before applying the unchanged public bound.
    evidence = replace(evidence,witness=ScanWitness(tuple(rows),bool(storage_complete),
                       original_body,original_complete))
    return SourceSummaryEvidence(**evidence.__dict__,reference=reference,question=query,
        summary_request=body,summary_topic=topic,summary_relation=relation,selection_complete=evidence.complete)



def _verify_original_pdf_summary(rows, document_root, tenant):
    """Rebuild PDF page evidence from the authenticated original, not index text."""
    import os
    from pathlib import Path
    from source_access import (open_source_file, source_stat_matches_metadata,
                               pdf_source_pages)
    from ingest import _read_pdf_text_pymupdf_words, _pdf_page_bodies
    from pdf_layout import (reconstruct_document, paragraph_packing_groups,
                            packing_group_metadata, PDF_LAYOUT_REVISION)
    from splitter import split_with_breadcrumbs
    import fitz
    try:
        if not rows or not isinstance(tenant, str) or not tenant or not document_root:
            return ()
        meta = rows[0]['metadata']
        keys = ('source_id','tenant_id','source_sha256','source_relative_path',
                'source_size','source_mtime_ns','mime_type')
        if (meta.get('tenant_id') != tenant or meta.get('mime_type') != 'application/pdf'
                or any(any(row.get('metadata', {}).get(key) != meta.get(key) for key in keys)
                       for row in rows)):
            return ()
        pages = tuple(sorted({row['metadata'].get('page') for row in rows}))
        if not pages or len(pages) > 8 or any(type(page) is not int or page < 1 for page in pages):
            return ()
        native = pdf_source_pages(document_root, tenant, meta, pages)
        if tuple(page['page'] for page in native) != pages:
            return ()
        page_count = native[0].get('page_count') if native else None
        if (type(page_count) is not int or not 1 <= page_count <= 8
                or pages != tuple(range(1, page_count + 1))
                or any(page.get('page_count') != page_count for page in native)
                or any(row['metadata'].get('source_page_count') != page_count for row in rows)):
            return ()
        relative, handle, before = open_source_file(document_root, tenant, meta['source_relative_path'])
        with handle:
            if relative.suffix.casefold() != '.pdf' or not source_stat_matches_metadata(meta, before):
                return ()
            if before.st_size > 32 * 1024 * 1024:
                return ()
            raw_bytes = handle.read(32 * 1024 * 1024 + 1)
            if hashlib.sha256(raw_bytes).hexdigest() != meta.get('source_sha256'):
                return ()
            # Both extractions open the already-authorized descriptor, never
            # an index-provided path; legacy text comes from this same PDF.
            descriptor_path = Path('/proc/self/fd/' + str(handle.fileno()))
            legacy = _read_pdf_text_pymupdf_words(descriptor_path)
            layout = reconstruct_document(descriptor_path, legacy, fitz)
            compatible = None
            if {row['metadata'].get('pdf_layout_mode') for row in rows} == {'single-column-compatible'}:
                # Re-run the same native extraction/canonicalization/splitting
                # producer on the authenticated descriptor. A direct native-page
                # comparison is not equivalent: the indexed producer performs
                # Hebrew reconstruction and may reorder page furniture.
                from ingest import (_read_pdf_text, _PDF_RECONSTRUCTION_TRACE_CACHE,
                                    clean_markdown, _normalize_extracted_text,
                                    location_metadata)
                from canonical_text import (canonicalize_source_text,
                                            canonicalize_reconstructed_source)
                if layout is not None:
                    return ()
                raw = _read_pdf_text(descriptor_path)
                trace = _PDF_RECONSTRUCTION_TRACE_CACHE.get(str(descriptor_path.resolve()))
                if trace and trace[1] == raw:
                    canonical = canonicalize_reconstructed_source(
                        trace[0], trace[1], trace[2], trace[3],
                        repair_visual_bidi=True, normalize_pdf_spacing=True,
                        join_pdf_line_wraps=True)
                else:
                    canonical = canonicalize_source_text(
                        raw, repair_visual_bidi=True, normalize_pdf_spacing=True,
                        join_pdf_line_wraps=True)
                chunks, locations = split_with_breadcrumbs(
                    clean_markdown(canonical.text), file_title=meta['source_relative_path'])
                compatible = (tuple(_normalize_extracted_text(chunk) for chunk in chunks),
                              tuple(location_metadata(item) for item in locations),
                              canonical.metadata())
            after = os.fstat(handle.fileno())
            identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
            if identity(before) != identity(after):
                return ()
        groups = {}
        for row in rows:
            groups.setdefault(row['metadata']['page'], []).append(row)
        modes = {row['metadata'].get('pdf_layout_mode') for row in rows}
        if len(modes) != 1:
            return ()
        if modes == {'positioned-coordinate-flows'}:
            if not layout or any(row['metadata'].get('pdf_layout_revision') != PDF_LAYOUT_REVISION
                                 for row in rows):
                return ()
            # Recreate the actual positioned producer's ordered chunks and
            # packing/flow metadata from the original PDF, including owner
            # headings. The index is never used to choose direction or fallback.
            expected_chunks = []
            for group in paragraph_packing_groups(layout):
                if group['page'] not in pages:
                    continue
                members = group['members']
                content = layout.canonical.text[group['canonical_start']:group['canonical_end']]
                heading = members[0]['heading'] if len(members) == 1 else ''
                section = 'Page ' + str(group['page']) + (' / ' + heading if heading else '')
                chunks, _ = split_with_breadcrumbs('# ' + section + '\n\n' + content,
                                                   file_title=meta['source_relative_path'])
                provenance = packing_group_metadata(group, layout.canonical)
                expected_chunks.extend((chunk, provenance) for chunk in chunks)
            if len(expected_chunks) != len(rows):
                return ()
            for row, (chunk, provenance) in zip(rows, expected_chunks):
                if row.get('document') != chunk or any(
                        row['metadata'].get(key) != value for key, value in provenance.items()):
                    return ()
            source_pages = _pdf_page_bodies(layout.canonical.text)
            if any(source_body(groups[page]) != source_pages.get(page, '') for page in pages):
                return ()
        elif modes == {'single-column-compatible'}:
            if (not compatible or any(row['metadata'].get('pdf_layout_revision') != PDF_LAYOUT_REVISION
                                      for row in rows)):
                return ()
            chunks, locations, canonical_metadata = compatible
            if (len(chunks) != len(rows) or len(locations) != len(rows)
                    or any(row.get('document') != chunk for row, chunk in zip(rows, chunks))
                    or any(row['metadata'].get('chunk_index') != index
                           or row['metadata'].get('page') != location.get('page')
                           or row['metadata'].get('section') != location.get('section')
                           for index, (row, location) in enumerate(zip(rows, locations)))
                    or any(any(row['metadata'].get(key) != value
                               for key, value in canonical_metadata.items()) for row in rows)):
                return ()
        else:
            # Legacy PDFs have no positioned witness: require their indexed
            # page to equal the independently authenticated native page after
            # whitespace normalization. No semantic approximation is allowed.
            if any(' '.join(source_body(groups[p['page']]).split()) != ' '.join(p['text'].split())
                   for p in native):
                return ()
        return tuple((page, digest(source_body(groups[page]))) for page in pages)
    except Exception:
        return ()


def validate_summary_evidence(evidence, query):
    """Recompute selection and request scope before accepting summary prose."""
    request = named_summary_request(query)
    witness = getattr(evidence, 'witness', None)
    if (not isinstance(evidence, SourceSummaryEvidence) or not request or not witness
            or evidence.question != query or (evidence.reference, evidence.summary_request) != request
            or evidence.selection_complete != evidence.complete):
        return False
    if (witness.rows and witness.rows[0]['metadata'].get('mime_type') == 'application/pdf'
            and independently_verified_overview_request(query)):
        groups = {}
        for row in witness.rows:
            groups.setdefault(row['metadata'].get('page'), []).append(row)
        expected_pdf = tuple((page, digest(source_body(rows))) for page, rows in sorted(groups.items()))
        if (not evidence.pdf_document_root or not evidence.pdf_tenant
                or evidence.pdf_page_hashes != expected_pdf):
            return False
    eligible = evidence.quote_eligible_ids
    if eligible is not None and (not isinstance(eligible,tuple)
            or any(not isinstance(value,str) or value not in {p.citation_id for p in evidence.passages}
                   for value in eligible)
            or len(eligible) != len(set(eligible))):
        return False
    grouped = {}
    for row in witness.rows:
        grouped.setdefault(row.get('metadata',{}).get('source_id'),[]).append(row)
    if len(grouped) != 1 or None in grouped:
        return False
    rows = next(iter(grouped.values()))
    expected = _summary_from_rows(rows,witness.storage_complete,len(rows),
        witness.original_body,witness.original_complete,(query,*request))
    from document_query import reference_matches
    from typed_summary import presentation_topic_matches
    if request[0].kind == 'discovered_material':
        from broad_material_discovery import parse_broad_material_request, original_matches_broad_material
        reference_valid = bool(original_matches_broad_material(parse_broad_material_request(query), source_body(rows)))
    else:
        reference_valid = (True if request[0].kind == 'selected_source' else presentation_topic_matches(request[0], rows)
            if request[0].kind == 'presentation_topic' else
            all(reference_matches(request[0],p.metadata['source_relative_path'],p.metadata['mime_type'])
                or reference_matches(request[0],p.metadata.get('title',''),p.metadata['mime_type'])
                for p in evidence.passages))
    return bool(evidence.summary_topic == expected.summary_topic
        and evidence.summary_relation == expected.summary_relation
        and evidence.passages == expected.passages and evidence.complete == expected.complete
        and evidence.reason == expected.reason
        and evidence.examined_documents == expected.examined_documents
        and evidence.examined_chunks == expected.examined_chunks
        and reference_valid)


def render_contents(evidence, hebrew):
    lead = ('תוכן המסמך, כפי שנכתב במקור:' if hebrew else 'Document contents, as written in the source:')
    body = '\n\n'.join(p.source_claim.strip() + ' ['+p.citation_id+']' for p in evidence.passages)
    if not evidence.complete:
        lead += ('\nזהו קטע חלקי; המסמך לא הוצג במלואו.' if hebrew else '\nThis is a partial excerpt; the full document is not reproduced.')
    return lead+'\n\n'+body


@dataclass(frozen=True)
class DocumentReferenceEvidence(ReadingEvidence):
    native_rows: tuple[dict, ...] = ()
    tenant_id: str = ''
    candidate_reads_complete: bool = True


def _reference_candidate_ids(grouped, request):
    # Digits are only a discovery hint. Primary ownership is checked from the
    # authenticated original, never established by a filename or incidental hit.
    number = re.compile(r'(?<!\d)'+re.escape(request.number)+r'(?!\d)')
    return tuple(source for source, rows in grouped.items() if number.search(
        source_body(rows)+'\n'+str(rows[0]['metadata'].get('source_relative_path', ''))))


def _document_reference_passages(grouped, native_rows, query):
    from financial_documents import explicit_document_request, primary_document_identities, role_matches, issuer_span
    request = explicit_document_request(query)
    if request is None:
        return ()
    passages = []
    for source in _reference_candidate_ids(grouped, request):
        rows = grouped[source]
        if rows[0]['metadata'].get('mime_type') == 'application/pdf':
            originals = tuple(row for row in native_rows if row['metadata']['source_id'] == source)
            if len(originals) != 1:
                continue
            row = originals[0]
            text = row['document']
            representation = 'native_pdf'
        else:
            row = rows[0]
            text = source_body(rows)
            representation = 'indexed'
        identities = primary_document_identities(text)
        # Conflicting primary titles are not silently reduced to the wanted one.
        if len(identities) != 1:
            continue
        identity = identities[0]
        if identity.number != request.number or not role_matches(identity.kind, request.kind):
            continue
        start, end = identity.span
        if request.issuer:
            issuer = issuer_span(text, request.issuer)
            if issuer is None:
                continue
            start, end = min(start, issuer[0]), max(end, issuer[1])
            if end-start > 780:
                continue
        claim = text[start:end]
        passages.append(replace(make_passage(text, claim, row['metadata'], row['id'], len(passages)+1),
            representation=representation))
    return tuple(passages)


def collect_document_reference(retriever, tenant, document_root, query, where=None):
    from financial_documents import explicit_document_request
    from source_access import pdf_source_pages, SourceAccessError
    request = explicit_document_request(query)
    grouped, complete, count = scan_authorized(retriever, tenant, document_root, where=where)
    witness = ScanWitness(tuple(row for rows in grouped.values() for row in rows), complete)
    evidence = DocumentReferenceEvidence((), False, len(grouped), count,
        'document_identity_not_found', witness, (), tenant)
    if request is None:
        return evidence
    candidates = _reference_candidate_ids(grouped, request)
    # Reuse the existing bounded native-source allowance, with one title page
    # per proposed source. This does not expand model context or retrieval depth.
    if len(candidates) > 8:
        return replace(evidence, reason='document_identity_candidates_exceed_bound',
            candidate_reads_complete=False)
    native = []
    read_complete = True
    for source in candidates:
        metadata = dict(grouped[source][0]['metadata'])
        if metadata.get('mime_type') != 'application/pdf':
            continue
        try:
            pages = pdf_source_pages(document_root, tenant, metadata, (1,))
        except (OSError, ValueError, SourceAccessError):
            if not _listing_relationship_excludes_source(rows, request, query):
                read_complete = False
            continue
        for page in pages:
            identifier = f"{source}::native-document-title::{page['page']:06d}::{page['text_sha256'][:16]}"
            meta = dict(metadata)
            meta.update(page=page['page'], section=f"Page {page['page']}", chunk_id=identifier,
                _entity_representation='native_pdf', _native_pdf_page_text_sha256=page['text_sha256'],
                _native_pdf_raw_text_sha256=page['raw_sha256'], _native_pdf_page_count=page['page_count'],
                _native_pdf_anchor_pages=(1,), _native_pdf_requested_pages=(page['page'],))
            native.append(dict(id=identifier, document=page['text'], metadata=meta))
    passages = _document_reference_passages(grouped, native, query)
    finished = bool(complete and read_complete)
    return replace(evidence, passages=passages, native_rows=tuple(native),
        complete=finished, candidate_reads_complete=read_complete,
        reason=('document_identity_ambiguous' if len(passages) > 1 else
                'document_identity_partial' if passages and not read_complete else '' if passages else
                'document_identity_not_found' if finished else 'document_identity_unavailable'))


def validate_document_reference(evidence, query):
    if not isinstance(evidence, DocumentReferenceEvidence) or not evidence.witness:
        return False
    grouped = {}
    for row in evidence.witness.rows:
        grouped.setdefault(row['metadata']['source_id'], []).append(row)
    expected = _document_reference_passages(grouped, evidence.native_rows, query)
    from financial_documents import explicit_document_request
    request = explicit_document_request(query)
    candidates = _reference_candidate_ids(grouped, request) if request else ()
    expected_pdf = {source for source in candidates
                    if grouped[source][0]['metadata'].get('mime_type') == 'application/pdf'}
    native_sources = tuple(row['metadata']['source_id'] for row in evidence.native_rows)
    reads_valid = (set(native_sources) <= expected_pdf and len(native_sources) == len(set(native_sources))
        and (not evidence.candidate_reads_complete or set(native_sources) == expected_pdf))
    return bool(expected and evidence.passages == expected and reads_valid
        and evidence.complete == bool(evidence.witness.storage_complete and evidence.candidate_reads_complete)
        and all(f.tenant_id == evidence.tenant_id for f in expected))


def render_document_reference(evidence, query, hebrew):
    from financial_documents import primary_document_identities, document_kind_label
    if not evidence.passages:
        return ('לא נמצא מסמך בעל סוג ומספר תואמים במקורות שניתן היה לאמת.' if hebrew else
                'No document with that primary type and number was verified in the sources examined.')
    lines = []
    if len(evidence.passages) > 1:
        lines.append('נמצאו כמה מסמכים מתאימים; בחרו במקור הרצוי:' if hebrew else
                     'Several documents match; choose the intended source:')
    for passage in evidence.passages:
        identity, = primary_document_identities(passage.text)
        label = document_kind_label(identity.kind, hebrew)
        name = passage.metadata['source_relative_path']
        lines.append(f'{label} {identity.number} — {name}. [{passage.citation_id}]')
    if len(evidence.passages) == 1:
        lines.append('אפשר לפתוח את המסמך דרך המקור המצורף.' if hebrew else
                     'Open the document using the attached source.')
    if not evidence.candidate_reads_complete:
        lines.append('לא ניתן היה לאמת את כל המקורות המועמדים.' if hebrew else
                     'Some candidate sources could not be verified.')
    return '\n'.join(lines)


def _reading_answer_hebrew(contract, evidence):
    """Recompute workbook response language from the canonical request.

    Generic lookup contracts do not carry a language qualifier. The request
    planner uses this same deterministic policy; the canonical request is
    digest-bound, while evidence supplied by a producer is not authority for
    choosing the response language.
    """
    from workbook_project import WorkbookProjectEvidence
    if isinstance(evidence, WorkbookProjectEvidence):
        from request_understanding import AnswerLanguage, detect_answer_language
        return detect_answer_language(contract.normalized_request) is AnswerLanguage.HE
    return 'answer_language:he' in contract.qualifiers
def build_reading_graph(contract, evidence, answer):
    from semantic_contracts import ClaimGraph, ClaimNode, CitationBinding
    facts = evidence.passages
    if isinstance(evidence, TechnicalReadingEvidence):
        try:
            decision = _technical_decision(evidence.technical_plan, evidence.sections,
                evidence.answer_json, evidence.review_json, evidence.allowed_source_ids,
                evidence.competing_source_ids)
        except (ValueError, TypeError):
            return ClaimGraph(contract.contract_id,contract.contract_sha256,(),(),(),('reading-1',),False,False)
        by_citation = {fact.citation_id: fact for fact in facts}
        claims, bindings = [], []
        for index, item in enumerate(decision.claims, 1):
            fact = by_citation.get(item.citation_id)
            if fact is None:
                return ClaimGraph(contract.contract_id,contract.contract_sha256,(),(),(),('reading-1',),False,False)
            claim = ClaimNode('reading-claim-'+str(index), item.answer+' ['+item.citation_id+']',
                ('reading-1',), (fact.fact_id,), (fact.evidence_id,), (fact.citation_id,), True)
            claims.append(claim)
            bindings.append(CitationBinding(fact.citation_id,claim.claim_id,(fact.fact_id,),fact.evidence_id,True))
        return ClaimGraph(contract.contract_id,contract.contract_sha256,tuple(claims),tuple(bindings),
                          ('reading-1',),(),True,True)
    if facts and all(str(getattr(fact,'value','')).startswith('monetary_role:') for fact in facts):
        # Each answer clause is bound only to the exact role snippets named
        # immediately after it. A claim may require several source fields.
        by_id={fact.citation_id:fact for fact in facts}
        claims=[];bindings=[];cursor=0
        for marker in re.finditer(r'(?:\[\d{1,3}\](?:\s+|$))+',answer):
            ids=tuple(re.findall(r'\[(\d{1,3})\]',marker.group()))
            if not ids or any(identifier not in by_id for identifier in ids):
                return ClaimGraph(contract.contract_id,contract.contract_sha256,(),(),(),('reading-1',),False,False)
            clause=answer[cursor:marker.start()].strip()
            if not clause:
                return ClaimGraph(contract.contract_id,contract.contract_sha256,(),(),(),('reading-1',),False,False)
            linked=tuple(by_id[identifier] for identifier in ids)
            claim=ClaimNode('reading-claim-'+str(len(claims)+1),clause,('reading-1',),
                tuple(f.fact_id for f in linked),tuple(f.evidence_id for f in linked),ids,True)
            claims.append(claim)
            bindings.extend(CitationBinding(f.citation_id,claim.claim_id,(f.fact_id,),f.evidence_id,True)
                            for f in linked)
            cursor=marker.end()
        if not claims or {f.citation_id for f in facts}!={c.citation_id for c in bindings}:
            return ClaimGraph(contract.contract_id,contract.contract_sha256,(),(),(),('reading-1',),False,False)
        return ClaimGraph(contract.contract_id,contract.contract_sha256,tuple(claims),tuple(bindings),
                          ('reading-1',),(),True,True)
    from workbook_project import WorkbookProjectEvidence
    if (contract.requirements[0].kind in {'entity_sections', 'document_topic', 'document_summary'}
            or isinstance(evidence, WorkbookProjectEvidence)):
        # Selection remains independently reproducible from the full witness;
        # the answer graph binds only passages actually used in this answer.
        used = set(re.findall(r'\[(\d{1,3})\]', answer))
        facts = tuple(f for f in facts if f.citation_id in used)
        from passage_synthesis import assess_section_answer
        decision = assess_section_answer(answer,evidence,_reading_answer_hebrew(contract,evidence))
        if decision.accepted and decision.bindings:
            by_citation = {fact.citation_id:fact for fact in facts}
            claims,bindings = [],[]
            for index,(text,citation_ids) in enumerate(decision.bindings,1):
                linked = tuple(by_citation[value] for value in citation_ids)
                claim = ClaimNode('reading-claim-'+str(index),text,('reading-1',),
                    tuple(f.fact_id for f in linked),tuple(f.evidence_id for f in linked),
                    citation_ids,True)
                claims.append(claim)
                bindings.extend(CitationBinding(f.citation_id,claim.claim_id,(f.fact_id,),f.evidence_id,True)
                    for f in linked)
            return ClaimGraph(contract.contract_id,contract.contract_sha256,tuple(claims),tuple(bindings),
                ('reading-1',),(),True,True)
    claim = ClaimNode('reading-claim', answer, ('reading-1',), tuple(p.fact_id for p in facts),
        tuple(p.evidence_id for p in facts), tuple(p.citation_id for p in facts), bool(facts))
    return ClaimGraph(contract.contract_id, contract.contract_sha256, (claim,),
        tuple(CitationBinding(p.citation_id, claim.claim_id, (p.fact_id,), p.evidence_id, True) for p in facts),
        ('reading-1',) if facts else (), () if facts else ('reading-1',), bool(facts), True)


def valid_subject_reference_provenance(contract):
    """Recompute a deictic document subject from bound user-side text."""
    current_refs = [item[len('subject_reference_current:'):] for item in contract.qualifiers
                    if item.startswith('subject_reference_current:')]
    prior_refs = [item[len('subject_reference_prior:'):] for item in contract.qualifiers
                  if item.startswith('subject_reference_prior:')]
    if not current_refs and not prior_refs:
        return contract.conversation_operation != 'ATTRIBUTE_FOCUS'
    if (len(current_refs) != 1 or len(prior_refs) != 1
            or contract.conversation_operation != 'ATTRIBUTE_FOCUS'):
        return False
    from subject_intent import resolve_document_subject_reference
    resolution = resolve_document_subject_reference(current_refs[0], prior_refs[0])
    return bool(resolution and resolution == (contract.normalized_request,
                                               contract.requirements[0].subject))


def reading_acceptance(contract, answer, graph, facts, citations, *, structural_safety):
    """Common gate extension for literal passages and checked ledger analysis.

    Recompute spans, source identities, date selection, duplicate accounting,
    arithmetic and the released rendering. Prior PASS booleans are not proof.
    """
    from semantic_contracts import AcceptanceDecision
    evidence = getattr(graph, 'reading_evidence', None)
    # Evidence is stored on the immutable facts' container by the caller via
    # request-local graph context; no serialization/prose can supply it.
    if evidence is None:
        # ClaimGraph is frozen; its typed extension is a separate subclass.
        evidence = getattr(graph, 'evidence', None)
    from workbook_project import WorkbookProjectEvidence, validate_workbook_project
    workbook_information = isinstance(evidence, WorkbookProjectEvidence)
    valid = bool(evidence and facts and tuple(facts) == evidence.passages and all(passage_valid(p) for p in facts))
    witness = evidence.witness if evidence else None
    grouped = {}
    if witness:
        for row in witness.rows:
            grouped.setdefault(row['metadata']['source_id'], []).append(row)
    valid = bool(valid and witness and len({str(row.get('id')) for row in witness.rows}) == len(witness.rows)
        and evidence.examined_chunks == len(witness.rows) and evidence.examined_documents == len(grouped)
        and (not evidence.complete or witness.storage_complete))
    if valid:
        for fact in facts:
            own = grouped.get(fact.source_id, ())
            identity_keys = ('source_id', 'tenant_id', 'source_sha256', 'source_relative_path', 'source_size', 'source_mtime_ns', 'mime_type')
            if not own or any(any(row['metadata'].get(key) != fact.metadata.get(key) for key in identity_keys) for row in own):
                valid = False
            own = [row for row in own if fact.metadata.get('page') is None or row['metadata'].get('page') == fact.metadata.get('page')]
            if workbook_information and fact.representation == 'workbook_original':
                # Workbook projections are reconstructed from the original
                # cells, not copied from indexed text. The workbook validator
                # below independently reopens and checks their structure.
                canonical = ' '.join(fact.text.split())
            elif fact.representation == 'native_pdf':
                # The indexed census remains the authorization/coverage
                # witness. Native pages are a separate selection witness;
                # the quote gate independently reopens those complete pages.
                native = [row for row in getattr(evidence, 'native_rows', ())
                          if row.get('metadata', {}).get('source_id') == fact.source_id
                          and row.get('metadata', {}).get('page') == fact.metadata.get('page')]
                if (contract.requirements[0].kind not in {'entity_sections', 'document_topic', 'financial_records', 'document_reference', 'document_type_listing', 'subject_work'} or not native
                        or any(any(row['metadata'].get(key) != fact.metadata.get(key)
                                   for key in identity_keys) for row in native)):
                    valid = False
                canonical = ' '.join(source_body(native).split())
            elif isinstance(evidence, TechnicalReadingEvidence):
                canonical = ' '.join(fact.text.split()) if any(
                    ' '.join(fact.text.split()) in ' '.join(str(row.get('document') or '').split())
                    for row in own) else ''
            else:
                canonical = ' '.join((witness.original_body if fact.representation == 'docx_original' else source_body(own)).split())
            if not canonical or ' '.join(fact.text.split()) not in canonical:
                valid = False
        if contract.requirements[0].kind in {'document_contents', 'document_summary'}:
            from document_query import reference_matches
            reference = (named_summary_request(contract.normalized_request)[0]
                         if contract.requirements[0].kind == 'document_summary'
                         else contents_reference(contract.normalized_request))
            if reference and reference.kind == 'selected_source':
                ids = {item for item in contract.source_scope if re.fullmatch(r'[A-Za-z0-9_-]{43}', str(item))}
                try:
                    retained = retained_summary_binding(contract)
                except (ValueError, TypeError):
                    retained = None
                    valid = False
                if len(ids) != 1 or any(f.source_id not in ids for f in facts):
                    valid = False
                if retained and (retained[0] not in ids or any(
                        (row.get('metadata', {}).get('source_id'),
                         digest(row.get('metadata', {}).get('source_relative_path') or ''),
                         row.get('metadata', {}).get('source_sha256')) != retained
                        for row in witness.rows)):
                    valid = False
            elif reference and reference.kind == 'discovered_material':
                from broad_material_discovery import parse_broad_material_request, original_matches_broad_material
                if (len(grouped) != 1 or not original_matches_broad_material(
                        parse_broad_material_request(contract.normalized_request), source_body(next(iter(grouped.values()))))):
                    valid = False
            elif reference and reference.kind == 'presentation_topic':
                from typed_summary import presentation_topic_matches
                if len(grouped) != 1 or not presentation_topic_matches(reference, next(iter(grouped.values()))):
                    valid = False
            elif reference and any(not (reference_matches(reference, f.metadata['source_relative_path'], f.metadata['mime_type'])
                or reference_matches(reference, f.metadata.get('title', ''), f.metadata['mime_type'])) for f in facts):
                valid = False
    from monetary_questions import MonetaryEvidence, validate_monetary_evidence
    from subject_native_work import (SubjectWorkNativeEvidence, SubjectWorkCombinedEvidence,
                                     validate_subject_work_native, validate_subject_work_combined)
    if valid and contract.requirements[0].kind == 'contact_lookup':
        from contact_evidence import ContactEvidence, validate_contact
        from chat_api import DOCS_BASE_DIR
        from chat_api import _CONVERSATION_CONTEXT
        context = getattr(_CONVERSATION_CONTEXT, 'value', None)
        ids = {item for item in contract.source_scope if re.fullmatch(r'[A-Za-z0-9_-]{43}', str(item))}
        valid = bool(isinstance(evidence, ContactEvidence) and isinstance(context, dict)
                     and context.get('tenant_id') == evidence.tenant
                     and evidence.document_root == str(DOCS_BASE_DIR)
                     and all(f.tenant_id == evidence.tenant for f in facts)
                     and all(row['metadata'].get('tenant_id') == evidence.tenant
                             and (not ids or row['metadata'].get('source_id') in ids)
                             for row in witness.rows)
                     and validate_contact(evidence, contract, answer))
    elif valid and workbook_information:
        valid = (contract.requirements[0].kind in {'lookup', 'project_overview'}
                 and evidence.subject == contract.requirements[0].subject
                 # Only the bound request contract may enable the overview
                 # interpretation; generated prose cannot select its gate.
                 and (not evidence.overview_synthesis
                      or getattr(contract, 'answer_scope', '') == 'overview')
                 and all(f.representation == 'workbook_original' for f in facts)
                 and validate_workbook_project(evidence, evidence.subject,
                                               evidence.document_root, evidence.tenant_id))
        if valid and any((evidence.narrow_scope, evidence.narrow_target, evidence.narrow_property)):
            from workbook_narrow import validate_workbook_narrow_answer
            valid = validate_workbook_narrow_answer(evidence, contract, answer)
    elif valid and isinstance(evidence, TechnicalReadingEvidence):
        valid = validate_technical_reading_evidence(evidence, contract, answer)
    elif valid and isinstance(evidence, SubjectWorkCombinedEvidence):
        from subject_work_verify import (validate_native_context_witnesses,
                                         validate_partial_context_answer)
        ordinary=evidence.ordinary_component
        ordinary_passages=evidence.passages[len(evidence.native_component.passages):]
        represented={passage.source_id for passage in ordinary.passages}
        valid = (validate_subject_work_combined(evidence,contract)
            and validate_native_context_witnesses(evidence.native_component,contract,answer,
                                                  accounted_omissions=represented)
            and validate_partial_context_answer(contract,ordinary,answer,ordinary_passages,
                accounted_omissions={passage.source_id for passage in evidence.native_component.passages}))
    elif valid and isinstance(evidence, SubjectWorkNativeEvidence):
        from subject_work_verify import validate_native_context_witnesses
        valid = (validate_subject_work_native(evidence, contract)
                 and validate_native_context_witnesses(evidence, contract, answer))
    elif valid and isinstance(evidence, MonetaryEvidence):
        valid = validate_monetary_evidence(evidence, contract.normalized_request)
        if valid and all(str(getattr(fact,'value','')).startswith('monetary_role:') for fact in facts):
            from monetary_role_verify import validate_monetary_role_witnesses
            valid = validate_monetary_role_witnesses(evidence,contract,answer,graph)
    elif valid and isinstance(evidence, DocumentTypeListingEvidence):
        listing_qualifiers = tuple(x for x in contract.qualifiers if x.startswith('document_listing_kind:'))
        kind_qualifier = ('document_listing_kind:'+evidence.requested_kind,) if evidence.requested_kind else ()
        page_qualifiers = tuple(x for x in contract.qualifiers if x.startswith('document_listing_page:'))
        expected_page = (f'document_listing_page:{evidence.page_offset}:{evidence.catalog_sha256}',)
        proposal_qualifiers = tuple(x for x in contract.qualifiers if x.startswith('document_listing_request:'))
        expected_proposal = ('document_listing_request:'+evidence.requested_proposal,) if evidence.requested_proposal else ()
        valid = (listing_qualifiers == kind_qualifier and proposal_qualifiers == expected_proposal
                 and validate_document_type_listing(evidence, contract.normalized_request)
                 and (page_qualifiers == expected_page if evidence.page_offset else not page_qualifiers))
    elif valid and contract.requirements[0].kind == 'conversation_document_reference':
        # Each reference must be represented once by the same currently
        # authorized source identity bound into this request's contract.
        ids = tuple(f.source_id for f in facts)
        valid = (ids == tuple(contract.source_scope) and len(ids) == len(set(ids))
                 and evidence.complete and witness.storage_complete
                 and all(f.metadata.get('display_name') for f in facts))
    elif valid and isinstance(evidence, DocumentReferenceEvidence):
        valid = validate_document_reference(evidence, contract.normalized_request)
    elif valid and isinstance(evidence, FinancialEvidence):
        rebuilt_sources, bodies_complete = sources_from_rows(grouped)
        expected_ids = selected_financial_ids(evidence, contract.normalized_request)
        quoted_ids = {p.source_id for p in facts}
        records = {record.source_id: record for record in financial_records(evidence)}
        # A citation to one field cannot vouch for all fields later rendered
        # from that source. Rebuild and require the exact quote groups used
        # by the producer, including the header and each monetary/date role.
        quotes_bound = all(metadata['source_id'] in records
            and bool(financial_record_quotes(text, records[metadata['source_id']]))
            and all(any(quote in fact.source_claim for fact in facts
            if fact.source_id == metadata['source_id'])
            for quote in financial_record_quotes(text, records[metadata['source_id']]))
            for text, metadata, _chunk_id in evidence.sources if metadata['source_id'] in quoted_ids)
        valid = (evidence.sources == rebuilt_sources
            and quoted_ids <= expected_ids and quotes_bound
            and evidence.complete == bool(witness.storage_complete and bodies_complete
                and expected_ids <= quoted_ids))
    elif valid and contract.requirements[0].kind in {'subject_documents', 'subject_work'}:
        from subject_reading import validate_subject
        valid = validate_subject(evidence, contract.requirements[0].subject,
                                 contract.requirements[0].kind,
                                 limit=contract.requirements[0].cardinality or 0,
                                 query=contract.normalized_request)
        if valid and contract.requirements[0].kind == 'subject_work':
            from subject_work_verify import validate_subject_work_contract,validate_partial_context_answer
            valid = (validate_subject_work_contract(contract, evidence)
                     and validate_partial_context_answer(contract,evidence,answer))
        elif valid and contract.requirements[0].kind == 'subject_documents':
            from subject_document_verify import validate_subject_document_contract
            valid = validate_subject_document_contract(contract, evidence)
    elif valid and contract.requirements[0].kind in {'entity_sections', 'document_topic'}:
        from entity_sections import validate_entity_evidence
        valid = validate_entity_evidence(evidence, contract.normalized_request)
    elif valid and contract.requirements[0].kind == 'document_summary':
        valid = validate_summary_evidence(evidence,contract.normalized_request)
        if (valid and facts and facts[0].metadata.get('mime_type') == 'application/pdf'
                and independently_verified_overview_request(contract.normalized_request)):
            from chat_api import DOCS_BASE_DIR
            valid = bool(str(DOCS_BASE_DIR) == evidence.pdf_document_root
                and all(fact.tenant_id == evidence.pdf_tenant for fact in facts)
                and _verify_original_pdf_summary(evidence.witness.rows, DOCS_BASE_DIR,
                    evidence.pdf_tenant) == evidence.pdf_page_hashes)
    elif valid:
        expected_text = witness.original_body or source_body(next(iter(grouped.values())))
        reproduced = ' '.join(''.join(f.source_claim for f in facts).split())
        valid = evidence.complete == bool(witness.storage_complete and witness.original_complete and reproduced == ' '.join(expected_text.split()))
    scope = {item for item in contract.source_scope if re.fullmatch(r'[A-Za-z0-9_-]{43}', str(item))}
    valid = bool(valid and (not scope or all(f.source_id in scope for f in facts))
        and len({row['metadata'].get('tenant_id') for row in witness.rows}) == 1
        and (not isinstance(evidence, FinancialEvidence) or all(f.tenant_id == evidence.tenant_id for f in facts)))
    spec = operation(contract.normalized_request)
    if contract.requirements[0].kind == 'contact_lookup':
        from contact_intent import validate_contact_interpretation
        contact = validate_contact_interpretation(contract.normalized_request,
                                                  contract.requirements[0].subject)
        spec = ('contact_lookup', contact) if contact else None
    if contract.requirements[0].kind == 'conversation_document_reference':
        spec = ('conversation_document_reference', None)
    if isinstance(evidence, DocumentTypeListingEvidence):
        interpreted = listing_request(contract.normalized_request, evidence.requested_kind, evidence.requested_proposal)
        if interpreted:
            spec = ('document_type_listing', interpreted)
    if isinstance(evidence, TechnicalReadingEvidence):
        spec = ('technical_specification', None)
    if workbook_information:
        from subject_intent import project_information_intent, validate_project_overview_subject
        if contract.requirements[0].kind == 'project_overview':
            proposal = validate_project_overview_subject(contract.normalized_request,
                                                         contract.requirements[0].subject)
        else:
            proposal = project_information_intent(contract.normalized_request,
                                                   contract.requirements[0].subject)
        if evidence.narrow_scope:
            from workbook_narrow import validate_workbook_narrow_answer
            spec = ('lookup', None) if validate_workbook_narrow_answer(evidence, contract, answer) else None
        else:
            spec = (contract.requirements[0].kind, proposal) if proposal else None
    if isinstance(evidence, MonetaryEvidence) and spec is None:
        from monetary_questions import evidence_monetary_request
        interpreted = evidence_monetary_request(evidence, contract.normalized_request)
        if interpreted is not None:
            spec = ('financial_records', interpreted)
    if contract.requirements[0].kind in {'subject_documents', 'subject_work'}:
        from subject_intent import validated_subject_proposal
        proposal = validated_subject_proposal(contract.normalized_request,
            contract.requirements[0].kind, contract.requirements[0].subject)
        ordinary = bool(proposal and (spec is None or
            (spec[0] == proposal.kind and spec[1].subject == proposal.subject)))
        # Recovery from a mistaken financial route is permitted only for a
        # complete, deterministic document-discovery subject. The consumer
        # independently checks the literal subject and unchanged filters.
        recovered = None
        if (not ordinary and spec and spec[0] == 'financial_records'
                and contract.requirements[0].kind == 'subject_documents'):
            from subject_intent import subject_request
            recovered = subject_request(contract.normalized_request, allow_monetary=True)
            if (not recovered or recovered.kind != 'subject_documents'
                    or recovered.subject != contract.requirements[0].subject):
                recovered = None
        accepted = proposal if ordinary else recovered
        spec = (accepted.kind, accepted) if accepted else None
    if (contract.requirements[0].kind == 'subject_documents'
            and not valid_subject_reference_provenance(contract)):
        spec = None
    graph_bound = bool(graph.contract_id == contract.contract_id and graph.contract_sha256 == contract.contract_sha256
        and contract.contract_sha256 == contract_digest(contract)
        and spec and spec[0] == contract.requirements[0].kind)
    supplied = {str(c.id): c for c in citations}
    used = set(re.findall(r'\[(\d{1,3})\]', answer))
    answer_facts = (tuple(f for f in facts if f.citation_id in used)
                    if workbook_information or contract.requirements[0].kind in {'entity_sections', 'document_topic', 'document_summary'} else facts)
    linked = valid and set(supplied) == {f.citation_id for f in answer_facts}
    for fact in answer_facts:
        citation = supplied.get(fact.citation_id)
        from evidence_quotes import normalize_evidence_whitespace, quote_core
        displayed = normalize_evidence_whitespace(quote_core(citation.evidence_quote or '')) if citation else ''
        source_claim = normalize_evidence_whitespace(fact.source_claim)
        quote_bound = bool(displayed and source_claim in displayed)
        if workbook_information and getattr(fact, 'representation', '') == 'workbook_original':
            # The quote consumer has independently re-opened and verified the
            # complete row. Its card may show a whole field or a visibly
            # truncated field prefix; neither is used to prove the claim.
            from workbook_project import _fields
            from evidence_quotes import workbook_field_excerpt_matches
            fields = [value for _, _, value, _, _ in _fields(fact.source_claim)]
            quote_bound = quote_bound or bool(displayed and
                workbook_field_excerpt_matches(citation.evidence_quote, fields) and
                (not str(citation.evidence_quote).strip().endswith('…') or
                 citation.evidence_quote_truncated))
        linked = bool(linked and citation and citation.source_id == fact.source_id
            and citation.source_relative_path == fact.metadata['source_relative_path']
            and citation.citation_lifecycle == 'verified'
            and fact.fact_id in citation._verified_fact_ids
            and citation._evidence_text == fact.text
            and fact.source_claim.strip() in citation._evidence_text
            and quote_bound)
    hebrew = _reading_answer_hebrew(contract,evidence)
    if workbook_information:
        language_qualifiers = tuple(x for x in contract.qualifiers if x.startswith('answer_language:'))
        expected_language = 'answer_language:he' if hebrew else 'answer_language:en'
        valid = bool(valid and (not language_qualifiers or language_qualifiers == (expected_language,)))
        if evidence.semantic_claims or evidence.semantic_review:
            from workbook_semantic_review import workbook_review_matches_question
            valid = bool(valid and workbook_review_matches_question(evidence, contract.normalized_request))
    if getattr(evidence, 'overview_review', None):
        from overview_semantic import review_matches
        valid = bool(valid and evidence.overview_question == contract.normalized_request
            and review_matches(contract.normalized_request, evidence, evidence.overview_claims,
                               evidence.overview_tenant, evidence.overview_review))
    kind = contract.requirements[0].kind
    if kind == 'contact_lookup':
        from contact_evidence import render_contact
        expected = render_contact(evidence) if evidence else ''
    else:
        expected = (render_contents(evidence, hebrew) if evidence and kind == 'document_contents'
        else render_financial(evidence, contract.normalized_request, hebrew) if evidence and kind == 'financial_records'
        else render_document_reference(evidence, contract.normalized_request, hebrew) if evidence and kind == 'document_reference'
        else render_document_type_listing(evidence, contract.normalized_request, hebrew) if evidence and kind == 'document_type_listing' else '')
    if evidence and kind == 'conversation_document_reference':
        from conversation_documents import render_references
        expected = render_references(evidence, hebrew)
    if kind == 'technical_specification' and isinstance(evidence, TechnicalReadingEvidence):
        rendering_supported = validate_technical_reading_evidence(evidence, contract, answer)
    elif kind in {'subject_documents', 'subject_work'} and evidence:
        from subject_reading import render_subject_documents, render_subject_work
        if kind == 'subject_work' and isinstance(evidence, SubjectWorkCombinedEvidence):
            from subject_native_work import render_subject_work_combined
            expected = render_subject_work_combined(evidence,hebrew)
        elif kind == 'subject_work' and isinstance(evidence, SubjectWorkNativeEvidence):
            from subject_native_work import render_subject_work_native
            expected = render_subject_work_native(evidence, hebrew)
        else:
            expected = (render_subject_documents(evidence, hebrew) if kind == 'subject_documents'
                        else render_subject_work(evidence, hebrew))
        rendering_supported = answer == expected and bool(expected)
    elif (kind in {'entity_sections', 'document_topic', 'document_summary'} or workbook_information) and evidence:
        from passage_synthesis import section_answer_supported
        rendering_supported = section_answer_supported(answer, evidence, hebrew)
    else:
        rendering_supported = bool(expected and answer == expected)
    expected_graph = build_reading_graph(contract,evidence,answer) if rendering_supported else None
    rendered = bool(expected_graph and graph.claims == expected_graph.claims
        and graph.citation_bindings == expected_graph.citation_bindings)
    parts = dict(mandatory_requirement_coverage=valid, claim_fact_linkage=bool(graph_bound and rendered),
        fact_evidence_linkage=valid, claim_citation_linkage=bool(linked),
        contextual_complete=bool(valid and rendered), standalone_complete=True,
        factual_bindings=valid, structural_safety=bool(structural_safety), rendered_values_bound=rendered)
    reasons = tuple(key for key, value in parts.items() if not value)
    return AcceptanceDecision(accepted=not reasons, reason_codes=reasons, **parts)


def graph_with_evidence(contract, evidence, answer):
    from semantic_contracts import ClaimGraph
    @dataclass(frozen=True)
    class ReadingGraph(ClaimGraph):
        reading_evidence: ReadingEvidence | None = None
    graph = build_reading_graph(contract, evidence, answer)
    return ReadingGraph(**graph.__dict__, reading_evidence=evidence)


@dataclass(frozen=True)
class FinancialEvidence(ReadingEvidence):
    # Ordered authenticated source bodies, not model-produced ledger entries.
    sources: tuple[tuple[str, dict, str], ...] = ()
    tenant_id: str = ''


def financial_records(evidence):
    from financial_semantics import extract_financial_record
    records = []
    for text, metadata, _chunk_id in evidence.sources:
        record = extract_financial_record(text, source_id=metadata['source_id'],
            title=metadata['display_name'], tenant_id=metadata['tenant_id'],
            page=metadata.get('page'), content_hash=metadata['source_sha256'])
        if record:
            records.append(record)
    return tuple(records)


def sources_from_rows(grouped):
    sources = []
    complete = True
    for rows in grouped.values():
        text = source_body(rows)
        if len(text) > MAX_SOURCE_CHARS:
            complete = False
            continue
        metadata = dict(rows[0]['metadata'])
        pages = {row['metadata'].get('page') for row in rows}
        if len(pages) != 1:
            metadata.pop('page', None)
            metadata['section'] = 'Indexed source text'
        sources.append((text, metadata, str(rows[0].get('id') or metadata.get('chunk_id') or '')))
    return tuple(sources), complete


def collect_financial(retriever, tenant, document_root, query, where=None):
    from financial_semantics import monetary_question
    if monetary_question(query) is not None:
        from monetary_questions import collect_monetary
        return collect_monetary(retriever, tenant, document_root, query, where)
    from financial_semantics import financial_query, financial_record_matches, aggregate_payments
    grouped, complete, count = scan_authorized(retriever, tenant, document_root, where=where)
    scan_complete = complete
    sources, bodies_complete = sources_from_rows(grouped)
    complete = complete and bodies_complete
    witness = ScanWitness(tuple(row for rows in grouped.values() for row in rows), scan_complete)
    evidence = FinancialEvidence((), complete, len(grouped), count, '', witness, tuple(sources), tenant)
    selected_ids = selected_financial_ids(evidence, query)
    request = financial_query(query)
    records = financial_records(evidence)
    passages = []
    quoted_chars = 0
    for text, metadata, chunk_id in sources:
        if metadata['source_id'] not in selected_ids:
            continue
        record = next(record for record in records if record.source_id == metadata['source_id'])
        claims = financial_record_quotes(text, record)
        # Long unrelated prose between a header and payment row must not
        # discard an otherwise supported record. Quote its actual fields in
        # bounded passages, preserving all facts used in the rendering.
        if not claims or quoted_chars+sum(map(len, claims)) > MAX_PUBLIC_CHARS:
            complete = False
            continue
        for claim in claims:
            passages.append(make_passage(text, claim, metadata, chunk_id, len(passages)+1))
            quoted_chars += len(claim)
    return replace(evidence, passages=tuple(passages), complete=complete,
                   reason='' if complete else 'financial_coverage_partial')


def financial_record_quotes(text, record):
    """Quote each rendered source role without quoting intervening prose."""
    from financial_semantics import _FINANCIAL_TYPE, _CANCELLED
    kind = _FINANCIAL_TYPE.search(text)
    if not kind:
        return ()
    # The identifier immediately following the type is part of the header,
    # rather than an amount or a title-derived document number.
    end = kind.end()
    if record.document_number:
        number = re.search(r'(?<!\d)'+re.escape(record.document_number)+r'(?!\d)', text[end:end+65])
        if number:
            end += number.end()
    spans = [(kind.start(), end)]
    spans.extend(field.span for field in (record.issue_date, record.subtotal, record.vat,
        record.total, *record.service_periods, *record.payments) if field)
    if record.cancelled:
        spans.extend(match.span() for match in _CANCELLED.finditer(text))
    merged = []
    for start, end in sorted(spans):
        joined = text[merged[-1][0]:max(end, merged[-1][1])] if merged else ''
        if merged and (start <= merged[-1][1] or len(joined) <= 780 and len(joined.split()) <= 78):
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    claims = tuple(text[start:end] for start, end in merged)
    # Ambiguous/oversized individual fields remain excluded, while ordinary
    # records no longer depend on their unrelated header-to-footer distance.
    return claims if all(len(claim) <= 780 and len(claim.split()) <= 78 for claim in claims) else ()


def financial_coverage(evidence, query):
    """Request-local diagnostics; citations are distinct from search coverage."""
    from financial_semantics import financial_query, financial_record_matches, aggregate_payments, monetary_question
    from monetary_questions import MonetaryEvidence
    if isinstance(evidence, MonetaryEvidence):
        from monetary_questions import monetary_coverage
        return monetary_coverage(evidence, query)
    request = financial_query(query)
    records = financial_records(evidence)
    selected = selected_financial_ids(evidence, query)
    matching = ({record.source_id for record, _payment in aggregate_payments(
        records, request, tenant_id=evidence.tenant_id).included}
        if request.operation == 'payments' else
        {record.source_id for record in records if financial_record_matches(record, request)})
    cited = {passage.source_id for passage in evidence.passages}
    excluded = len(_group_witness(evidence)) - len(evidence.sources)
    return dict(authorized_sources_scanned=evidence.examined_documents,
        indexed_chunks_examined=evidence.examined_chunks,
        financial_sources_examined=len(evidence.sources),
        sources_excluded_from_financial_extraction=excluded,
        financial_records_parsed=len(records),
        matching_records=len(matching), related_records=len(selected-matching),
        selected_records=len(cited),
        matching_records_without_selected_evidence=len(matching-cited),
        selected_records_without_evidence=len(selected-cited),
        selected_evidence_passages=len(evidence.passages),
        storage_scan_complete=bool(evidence.witness and evidence.witness.storage_complete),
        financial_coverage_complete=evidence.complete)


def selected_financial_ids(evidence, query):
    from financial_semantics import financial_query, financial_record_matches, aggregate_payments
    request = financial_query(query)
    records = financial_records(evidence)
    if request.operation == 'payments':
        aggregate = aggregate_payments(records, request, tenant_id=evidence.tenant_id)
        selected_ids = {record.source_id for record, _payment in aggregate.included}
        # Related service/issue records provide a useful, explicitly separate
        # explanation when the payment month does not match.
        from dataclasses import replace as change
        related = change(request, operation='documents', date_basis='service_or_issue')
        selected_ids.update(record.source_id for record in records if financial_record_matches(record, related)
            and not any((payment.date.year, payment.date.month) == (request.year, request.month) for payment in record.payments))
    else:
        selected_ids = {record.source_id for record in records if financial_record_matches(record, request)}
    from request_understanding import explicit_requested_items
    limit = explicit_requested_items(query)
    if limit and request.operation == 'documents':
        selected_ids = {record.source_id for record in records if record.source_id in selected_ids}
        selected_ids = set([record.source_id for record in records if record.source_id in selected_ids][:limit])
    return selected_ids


def render_financial(evidence, query, hebrew):
    from financial_semantics import financial_query, financial_record_matches, aggregate_payments, monetary_question
    from monetary_questions import MonetaryEvidence
    if isinstance(evidence, MonetaryEvidence):
        from monetary_questions import render_monetary
        return render_monetary(evidence, query, hebrew)
    request = financial_query(query)
    if not request or not isinstance(evidence, FinancialEvidence):
        return ''
    # Reparse actual source text at the release gate; never accept cached
    # extraction values or a model's arithmetic as the canonical ledger.
    records = financial_records(evidence)
    cited = {}
    for passage in evidence.passages:
        cited.setdefault(passage.source_id, []).append(passage.citation_id)
    markers_for = lambda source: ' '.join('['+item+']' for item in cited[source])
    visible = tuple(record for record in records if record.source_id in cited)
    lines = []
    roles_he = {'service': 'תקופת פעילות', 'issue': 'הנפקה', 'payment': 'תשלום'}
    if request.operation == 'payments':
        aggregation = aggregate_payments(visible, request, tenant_id=evidence.tenant_id)
        if aggregation.included:
            totals = '; '.join(f'{amount:,.2f} {currency}' for currency, amount in aggregation.totals)
            markers = ' '.join(dict.fromkeys(markers_for(record.source_id) for record, _payment in aggregation.included))
            lines.append((f'סך התשלומים המתועדים שאומתו לחודש {request.period}: {totals}. {markers}' if hebrew else f'Verified documented payments dated {request.period}: {totals}. {markers}'))
            for record, payment in aggregation.included:
                lines.append(f'- {record.title}: {payment.date.isoformat()}, {payment.amount:,.2f} {payment.currency} {markers_for(record.source_id)}')
        elif aggregate_payments(records, request, tenant_id=evidence.tenant_id).included:
            lines.append((f'נמצאו רשומות תשלום המתוארכות לחודש {request.period}, אך לא ניתן היה לצרף ראיות מספיקות להצגת סכום מאומת.' if hebrew else f'Payment records dated {request.period} were found, but sufficient supporting evidence could not be attached to present a verified total.'))
        else:
            lines.append((f'לא נמצא תשלום מאומת המתוארך לחודש {request.period} במסמכים שנבדקו.' if hebrew else f'No verified payment dated {request.period} was found in the documents examined.'))
        related_query = replace(request, operation='documents', date_basis='service_or_issue')
        related = [record for record in visible if financial_record_matches(record, related_query)
                   and not any((payment.date.year, payment.date.month) == (request.year, request.month) for payment in record.payments)]
        if related:
            lines.append('מסמכים קשורים לחודש הפעילות או ההנפקה, בנפרד ממועד קבלת הכסף:' if hebrew else 'Related service or issue records, separate from the payment month:')
        rows = related
    else:
        rows = [record for record in visible if financial_record_matches(record, request)]
        basis = ('תקופת פעילות או חודש הנפקה; התאריכים מובחנים להלן' if hebrew else 'service period or issue month, distinguished below') if request.date_basis == 'service_or_issue' else roles_he.get(request.date_basis, request.date_basis) if hebrew else request.date_basis
        if rows:
            lines.append((f'מסמכים לחודש {request.period}, לפי {basis}:' if hebrew else f'Documents for {request.period}, interpreted by {basis}:'))
        elif selected_financial_ids(evidence, query):
            lines.append((f'נמצאו רשומות קשורות לחודש {request.period}, אך לא ניתן היה לצרף ראיות מספיקות להצגתן.' if hebrew else f'Related records for {request.period} were found, but sufficient supporting evidence could not be attached to display them.'))
        else:
            lines.append((f'לא נמצאו מסמכים תואמים לחודש {request.period} במקורות שנבדקו, לפי {basis}.' if hebrew else f'No matching documents for {request.period} were found in the sources examined, interpreted by {basis}.'))
    from request_understanding import explicit_requested_items
    limit = explicit_requested_items(query)
    if limit:
        rows = rows[:limit]
    for record in rows:
        issue = record.issue_date.value.isoformat() if record.issue_date else ('לא מאומת' if hebrew else 'unproven')
        service = ', '.join(f'{field.value[0]:04d}-{field.value[1]:02d}' for field in record.service_periods)
        payment = '; '.join(f'{item.date.isoformat()}: {item.amount:,.2f} {item.currency}' for item in record.payments)
        parts = [record.title, ('הנפקה: ' if hebrew else 'issued: ')+issue]
        if record.document_number:
            parts.insert(1, ('מספר מסמך: ' if hebrew else 'document number: ')+record.document_number)
        if service:
            parts.append(('פעילות: ' if hebrew else 'service: ')+service)
        if payment:
            parts.append(('תשלום: ' if hebrew else 'payment: ')+payment)
        if record.total:
            parts.append(('סך המסמך: ' if hebrew else 'document total: ')+f'{record.total.value:,.2f} {record.currency}')
        if record.cancelled or record.document_kind == 'credit':
            parts.append('זיכוי/ביטול; אינו הוכחת תקבול או החזר' if hebrew else 'credit/cancellation; does not establish receipt or refund')
        lines.append('- '+'; '.join(parts)+' '+markers_for(record.source_id))
    # Coverage counts belong to diagnostics. A single material qualification
    # follows the supported answer; no-execution or no-match is never a zero
    # real-world cash-receipt claim.
    if not evidence.complete:
        lines.append(('הבדיקה מוגבלת לראיות שניתן היה לעבד ולהציג; הרשימה או הסכום אינם ממצים.' if hebrew else 'This is limited to evidence that could be processed and presented; the list or total is not exhaustive.'))
    elif request.operation == 'payments' and not aggregation.included:
        lines.append(('היעדר תשלום מתועד אינו מוכיח שלא התקבל כסף בפועל.' if hebrew else 'The absence of a documented payment does not establish that no money was received.'))
    return '\n'.join(lines)


def _group_witness(evidence):
    grouped = {}
    for row in evidence.witness.rows if evidence.witness else ():
        grouped.setdefault(row['metadata']['source_id'], []).append(row)
    return grouped
@dataclass(frozen=True)
class DocumentTypeListingEvidence(ReadingEvidence):
    native_rows: tuple[dict, ...] = ()
    tenant_id: str = ''
    candidate_reads_complete: bool = True
    fallback_sources: tuple[str, ...] = ()
    display_complete: bool = True
    page_offset: int = 0
    catalog_sha256: str = ''
    requested_kind: str = ''
    requested_proposal: str = ''
    semantic_candidates: tuple = ()
    semantic_batch: object = None
    semantic_gap: bool = False
    semantic_failure: bool = False


# Keep each page within the browser's response-size and citation-marker bounds.
MAX_DOCUMENT_TYPE_RESULTS = 50


def listing_request(query, requested_kind='', requested_proposal=''):
    from financial_documents import document_type_listing, validate_semantic_document_listing
    if requested_proposal:
        try:
            proposal = json.loads(requested_proposal)
        except (ValueError, TypeError):
            return None
        result = validate_semantic_document_listing(query, proposal)
        return result if result and result.kind == requested_kind else None
    direct = document_type_listing(query)
    if direct:
        return direct if not requested_kind or direct.kind == requested_kind else None
    if requested_kind:
        return validate_semantic_document_listing(query,
            {'operation': 'document_listing', 'kind': requested_kind})
    return None


def _native_listing_title(rows):
    """Select an unambiguous primary title on either of the first two pages."""
    from financial_documents import primary_document_classifications
    by_page = {row['metadata'].get('page'): row for row in rows
               if row['metadata'].get('page') in (1, 2)}
    if 1 not in by_page:
        return None, False
    first = primary_document_classifications(by_page[1]['document'])
    second = primary_document_classifications(by_page[2]['document']) if 2 in by_page else ()
    if len(first) > 1 or len(second) > 1:
        return None, True
    if first and second:
        left, right = first[0], second[0]
        if (left.kind, left.number) != (right.kind, right.number):
            return None, True
    if first:
        return by_page[1], False
    if second:
        return by_page[2], False
    return by_page[1], False


def _native_listing_groups(native_rows):
    grouped = {}
    for row in native_rows:
        grouped.setdefault(row['metadata']['source_id'], []).append(row)
    return grouped


def _complete_quote_total(rows):
    """Prove a unique total over every page of one bounded native PDF."""
    from financial_documents import quote_total_literal
    if not rows:
        return None
    counts = {row.get('metadata', {}).get('_native_pdf_page_count') for row in rows}
    if len(counts) != 1:
        return None
    count, = counts
    if type(count) is not int or not 1 <= count <= 8:
        return None
    by_page = {row.get('metadata', {}).get('page'): row for row in rows}
    if len(by_page) != len(rows) or set(by_page) != set(range(1, count+1)):
        return None
    ordered = [by_page[number] for number in range(1, count+1)]
    source_ids = {row.get('metadata', {}).get('source_id') for row in ordered}
    if len(source_ids) != 1:
        return None
    full = '\n'.join(row.get('document', '') for row in ordered)
    literal = quote_total_literal(full)
    if not literal:
        return None
    matches = [row for row in ordered if literal in row.get('document', '')]
    return (matches[0], literal) if len(matches) == 1 else None


def _document_type_listing_passages(grouped, native_rows, query, requested_kind='', requested_proposal='',
                                    semantic_receipts=()):
    from financial_documents import (document_type_listing, primary_document_classifications,
                                     listing_kind_matches, listing_filter_matches, _listing_named_match)
    request = listing_request(query, requested_kind, requested_proposal)
    if request is None:
        return ()
    native_by_source = _native_listing_groups(native_rows)
    semantic_by_source = {}
    for receipt in semantic_receipts:
        semantic_by_source.setdefault(receipt.candidate.source_id, receipt)
    passages = []
    for source, rows in grouped.items():
        topic_match = None
        if request.topic_groups:
            from topic_discovery import find_topic_match
            topic_match = find_topic_match(rows, request.topic_groups)
            if topic_match is None and source in semantic_by_source:
                from topic_discovery import TopicMatch
                candidate = semantic_by_source[source].candidate
                indices = [index for index, row in enumerate(rows) if str(row.get('id') or '') == candidate.chunk_id]
                if len(indices) == 1:
                    topic_match = TopicMatch(indices[0], candidate.start, candidate.end, candidate.excerpt)
            if topic_match is None:
                continue
        if not request.kind:
            if topic_match is None:
                continue
            row = rows[topic_match.row_index]
            if listing_filter_matches(row['document'], replace(request, topic='', topic_groups=()), row['metadata']) is not True:
                continue
            text = source_body((row,))
            passages.append(make_passage(text, topic_match.excerpt, row['metadata'], row['id'], len(passages)+1))
            continue
        if rows[0]['metadata'].get('mime_type') == 'application/pdf':
            row, conflict = _native_listing_title(native_by_source.get(source, ()))
            if row is None or conflict:
                continue
            text, representation = row['document'], 'native_pdf'
        else:
            row, text, representation = rows[0], source_body(rows), 'indexed'
        identities = primary_document_classifications(text)
        if len(identities) != 1:
            continue
        identity = identities[0]
        from financial_documents import (quotation_amount_topic, quotation_topic_variants,
                                         quote_header_topic_matches, quote_total_literal)
        quote_amount = bool(quotation_amount_topic(query))
        if not listing_kind_matches(identity.kind, request.kind):
            continue
        if quote_amount:
            if not quote_header_topic_matches(text, identity, quotation_topic_variants(query)):
                continue
        elif listing_filter_matches(text, replace(request, topic='', topic_groups=()) if topic_match else request, row['metadata']) is not True:
            continue
        claim = text[slice(*identity.span)]
        # A source-level quote total is a distinct field, not an item price.
        # Keep only an unambiguous, explicitly labeled literal in that same
        # authenticated source; selection is recomputed at the consumer.
        if quote_amount:
            if representation == 'native_pdf':
                total = _complete_quote_total(native_by_source.get(source, ()))
                if not total:
                    continue
                row, claim = total
                text = row['document']
            else:
                claim = quote_total_literal(text)
                if not claim:
                    continue
        # A filtered listing's citation should show why the source is related
        # to the requested person or topic, not just repeat its document title.
        # Classification and all filters remain independently checked above.
        if topic_match and not quote_amount:
            row = rows[topic_match.row_index]
            text = source_body((row,))
            claim = topic_match.excerpt
            representation = 'indexed'
        elif request.topic and not quote_amount:
            body = re.split(r'(?im)^\s*(?:footer|contact\s+(?:details|information)|פרטי\s+קשר|יצירת\s+קשר)\s*[:\-]?\s*$', text, maxsplit=1)[0]
            for line in body.splitlines():
                if (line.strip() and not re.search(r'(?i)(?:footer|טלפון|פקס|דוא"ל|כתובת|www\.|https?://|@)', line)
                        and _listing_named_match([line], request.topic)):
                    claim = line.strip()
                    break
        elif request.customer or request.project:
            pattern = (r'(?im)^[ \t]*[:\-]?[ \t]*(?:customer|client|bill\s+to|לקוח|לקוחה|לכבוד)[ \t]*[:\-]?[ \t]*(?:\n[ \t]*)?[^\n]+'
                       if request.customer else r'(?im)^\s*(?:project|פרויקט)\s*[:\-]\s*[^\n]+')
            relation = request.customer or request.project
            for match in re.finditer(pattern, text):
                if _listing_named_match([match.group()], relation):
                    claim = match.group().strip()
                    break
        passages.append(replace(make_passage(text, claim, row['metadata'], row['id'],
                                             len(passages)+1), representation=representation))
    return tuple(passages)



def _listing_index_complete(rows):
    if not rows:
        return False
    n = rows[0]['metadata'].get('chunk_count')
    return bool(type(n) is int and n > 0 and len(rows) == n
                and all(row['metadata'].get('chunk_count') == n for row in rows)
                and {row['metadata'].get('chunk_index') for row in rows} == set(range(n))
                and all(str(row.get('document') or '').strip() for row in rows))


def _listing_relationship_excludes_source(rows, request, query=""):
    """Exclude only complete indexed source bodies missing an explicit relation.

    Missing chunks, empty content and image-only PDF pages cannot prove absence.
    The check uses the whole indexed source, not the top retrieval hits or name.
    """
    from financial_documents import _listing_named_match
    if not rows or not request:
        return False
    terms = tuple(x for x in (request.customer, request.project, request.topic) if x)
    if not terms:
        return False
    if not _listing_index_complete(rows):
        return False
    metadata = rows[0]['metadata']
    if metadata.get('mime_type') == 'application/pdf':
        try:
            manifest = json.loads(metadata.get('pdf_page_manifest_json') or 'null')
        except (TypeError, ValueError):
            return False
        if (not isinstance(manifest, list) or not manifest
                or any(not isinstance(page, dict)
                       or (not page.get('native_text_chars') and page.get('embedded_image_count'))
                       for page in manifest)):
            return False
    body = source_body(rows)
    if request.topic_groups:
        from topic_discovery import find_topic_match
        return bool(body and find_topic_match(rows, request.topic_groups) is None)
    from financial_documents import quotation_topic_variants
    variants = quotation_topic_variants(query)
    if variants and request.topic and terms == (request.topic,):
        return bool(body and not any(_listing_named_match([body], term) for term in variants))
    return bool(body and any(_listing_named_match([body], term) is False for term in terms))

def _listing_source_unresolved(grouped, native_rows, fallback_sources, query, requested_kind='', requested_proposal='', *, ignore_missing=False):
    from financial_documents import (document_type_listing, primary_document_classifications,
        listing_kind_matches, listing_filter_matches, possible_document_type_mention)
    request = listing_request(query, requested_kind, requested_proposal)
    if not request:
        return True
    native_map = _native_listing_groups(native_rows)
    for source, rows in grouped.items():
        if (request.customer or request.project or request.topic) and not _listing_index_complete(rows):
            return True
        if _listing_relationship_excludes_source(rows, request, query):
            continue
        if not request.kind:
            from topic_discovery import find_topic_match
            if request.topic_groups and find_topic_match(rows, request.topic_groups) is None:
                return True
            continue
        if rows[0]['metadata'].get('mime_type') == 'application/pdf':
            row, conflict = _native_listing_title(native_map.get(source, ()))
            if conflict:
                return True
            if row is None:
                if source not in fallback_sources and not ignore_missing:
                    return True
                continue
            body = row['document']
        else:
            body = source_body(rows)
            if not body.strip():
                return True
        identities = primary_document_classifications(body)
        indexed_identities = primary_document_classifications(source_body(rows))
        if identities and indexed_identities:
            selected = {(item.kind, item.number) for item in identities}
            if any((item.kind, item.number) not in selected for item in indexed_identities):
                return True
        if not identities and possible_document_type_mention(source_body(rows), request):
            scoped = any((request.customer, request.project, request.topic, request.collection))
            dated = any((request.year, request.date, request.date_start, request.date_end, request.period))
            if not dated or scoped:
                return True
            from financial_documents import possible_primary_document_type_heading
            if possible_primary_document_type_heading(source_body(rows), request):
                return True
        if len(identities) > 1 and any(listing_kind_matches(identity.kind, request.kind) for identity in identities):
            return True
        from financial_documents import quotation_topic_variants, quote_header_topic_matches, quote_total_literal
        variants = quotation_topic_variants(query)
        if variants and len(identities) == 1 and listing_kind_matches(identities[0].kind, request.kind):
            from financial_documents import _listing_named_match
            header_matches = quote_header_topic_matches(body, identities[0], variants)
            if header_matches:
                quote_total = (_complete_quote_total(native_map.get(source, ()))
                    if rows[0]['metadata'].get('mime_type') == 'application/pdf' else quote_total_literal(body))
                if not quote_total:
                    return True
            elif any(_listing_named_match([source_body(rows)], term) for term in variants):
                # A subject or project mentioned only after the title cannot
                # prove an issuer/recipient relation or establish absence.
                return True
        if (len(identities) == 1 and listing_kind_matches(identities[0].kind, request.kind)
                and listing_filter_matches(body, replace(request, topic='', topic_groups=()) if request.topic_groups else request, rows[0]['metadata']) is None):
            return True
    return False


def listing_catalog_digest(grouped):
    """Bind a continuation to every authorized source generation and indexed row."""
    rows = [(source, tuple((row['metadata'].get('source_sha256'),
                            row['metadata'].get('source_relative_path'),
                            str(row.get('id') or '')) for row in hits))
            for source, hits in sorted(grouped.items())]
    return digest(json.dumps(rows, ensure_ascii=False, separators=(',', ':')))


def listing_result_digest(grouped, passages, semantic_candidates=(), semantic_batch=None, semantic_gap=False,
                          semantic_failure=False):
    membership = tuple((fact.source_id, fact.source_hash) for fact in passages)
    base = listing_catalog_digest(grouped)+json.dumps(membership, separators=(',', ':'))
    if semantic_candidates or semantic_batch is not None or semantic_gap or semantic_failure:
        base += json.dumps({'candidates': [asdict(item) for item in semantic_candidates],
                            'batch': asdict(semantic_batch) if semantic_batch is not None else None,
                            'gap': semantic_gap, 'failure': semantic_failure}, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    return digest(base)


def _valid_semantic_batch(grouped, tenant, request, excluded, candidates, batch, gap, failure, *, diagnostics=None):
    """Check model review belongs to this exact authorized scan and topic."""
    from topic_semantic import (MAX_CANDIDATES, SemanticBatchResult,
                                validate_candidate, validate_receipt)

    def reject(reason):
        if isinstance(diagnostics, list):
            diagnostics.append(reason)
        return False

    if (type(gap) is not bool or type(failure) is not bool or (failure and not gap)
            or not isinstance(candidates, tuple) or len(candidates) > MAX_CANDIDATES):
        return reject('batch_flags_or_candidate_shape')
    if not candidates:
        return True if batch is None else reject('batch_without_candidates')
    if (not request or not request.topic_groups or not isinstance(batch, SemanticBatchResult)
            or not batch.valid_response):
        return reject('batch_missing_or_invalid_decisions')
    if not all(validate_candidate(candidate, grouped, tenant)
               and candidate.source_id not in excluded for candidate in candidates):
        return reject('batch_candidate_source_binding')
    ids = [candidate.candidate_id for candidate in candidates]
    if len(ids) != len(set(ids)):
        return reject('batch_duplicate_candidates')
    if not isinstance(batch.receipts, tuple) or not isinstance(batch.rejected_ids, tuple) or not isinstance(batch.unresolved_ids, tuple):
        return reject('batch_result_shape')
    if not all(validate_receipt(receipt, grouped, tenant, request.topic_groups)
               for receipt in batch.receipts):
        return reject('batch_receipt_support_binding')
    accepted = [receipt.candidate.candidate_id for receipt in batch.receipts]
    if any(receipt.candidate not in candidates for receipt in batch.receipts):
        return reject('batch_foreign_receipt')
    accounted = accepted + list(batch.rejected_ids) + list(batch.unresolved_ids)
    if not (all(isinstance(item, str) for item in accounted)
            and len(accounted) == len(set(accounted)) and set(accounted) == set(ids)):
        return reject('batch_accounting_mismatch')
    if batch.unresolved_ids and not gap:
        return reject('batch_unresolved_without_gap')
    if isinstance(diagnostics, list):
        diagnostics.append('valid')
    return True


def indexed_ocr_alternate_title(rows, request):
    """Use only page-one OCR to affirm a different primary document type."""
    from financial_documents import (primary_document_classifications,
        alternate_primary_document_classification, role_matches)
    if not rows or not request:
        return False
    if any(row['metadata'].get('extraction_method') != 'pdf-native-with-bounded-ocr'
           for row in rows):
        return False
    page_one = [row for row in rows if row['metadata'].get('page') == 1]
    if not page_one:
        return False
    text = source_body(page_one)
    if not text.strip():
        return False
    # A financial heading anywhere in the authenticated indexed source is
    # conflicting; do not reduce it to an unrelated OCR title.
    all_financial = primary_document_classifications(source_body(rows))
    if all_financial:
        page_financial = primary_document_classifications(text)
        return (len(page_financial) == len(all_financial) == 1
                and page_financial[0].kind == all_financial[0].kind
                and page_financial[0].number == all_financial[0].number
                and not role_matches(page_financial[0].kind, request.kind))
    return bool(alternate_primary_document_classification(text))


def collect_document_type_listing(retriever, tenant, document_root, query, where=None, *,
                                  page_offset=0, expected_catalog='', requested_kind='', requested_proposal='',
                                  semantic_assessor=None):
    """Classify each authorized source by its primary title, not incidental mentions."""
    from source_access import pdf_source_pages, SourceAccessError
    grouped, complete, count = scan_authorized(retriever, tenant, document_root, where=where,
                                               record_limit=100_000, char_limit=None)
    grouped = dict(sorted(grouped.items()))
    witness = ScanWitness(tuple(row for rows in grouped.values() for row in rows), complete)
    native, fallback_sources, read_complete = [], [], True
    request = listing_request(query, requested_kind, requested_proposal)
    semantic_candidates, semantic_batch, semantic_gap, semantic_failure = (), None, False, False
    if semantic_assessor is not None and request and request.topic_groups:
        from topic_discovery import find_topic_matches
        excluded = frozenset(find_topic_matches(grouped, request.topic_groups))
        batch_diagnostics = []
        try:
            semantic_candidates, semantic_batch, semantic_gap, semantic_failure = semantic_assessor(grouped, request, excluded)
            if not _valid_semantic_batch(grouped, tenant, request, excluded,
                                         semantic_candidates, semantic_batch, semantic_gap, semantic_failure,
                                         diagnostics=batch_diagnostics):
                raise ValueError('invalid semantic review')
        except Exception as exc:
            # Content-free diagnostics distinguish an assessor exception from
            # the existing source-binding/accounting rejection.
            import logging
            from request_performance import current as current_request_performance
            performance = current_request_performance()
            logging.getLogger(__name__).info('DOCUMENT_DISCOVERY_SEMANTIC_COLLECTOR %s', json.dumps({
                'request_id': getattr(performance, 'request_id', 'unavailable'),
                'reason': batch_diagnostics[-1] if batch_diagnostics else 'assessor_exception',
                'exception_type': type(exc).__name__,
            }, sort_keys=True))
            # A failed optional review cannot erase independently proven hits.
            semantic_candidates, semantic_batch, semantic_gap, semantic_failure = (), None, True, True
    for source, rows in grouped.items():
        metadata = dict(rows[0]['metadata'])
        if metadata.get('mime_type') != 'application/pdf' or (request and not request.kind):
            continue
        try:
            pages = pdf_source_pages(document_root, tenant, metadata, (1,))
            if len(pages) != 1:
                raise ValueError('missing original title page')
            first_blank = not str(pages[0].get('text') or '').strip()
            requested = (1, 2) if pages[0]['page_count'] > 1 else (1,)
            if len(requested) > 1:
                pages = pdf_source_pages(document_root, tenant, metadata, requested)
                if len(pages) != len(requested):
                    raise ValueError('missing title context page')
            candidates = [dict(document=page['text'], metadata={'page':page['page']}) for page in pages]
            selected, conflict = _native_listing_title(candidates)
            if first_blank and (not selected or selected['metadata']['page'] != 2 or conflict):
                # A blank cover does not preclude a readable page-two title.
                # If none is established, only affirmative existing OCR can
                # classify a different primary type.
                request = listing_request(query, requested_kind, requested_proposal)
                if indexed_ocr_alternate_title(rows, request):
                    fallback_sources.append(source)
                elif not _listing_relationship_excludes_source(rows, request, query):
                    read_complete = False
                continue
            anchor = 2 if selected and not conflict and selected['metadata']['page'] == 2 else 1
            from financial_documents import (quotation_topic_variants, primary_document_classifications,
                                              quote_header_topic_matches, role_matches)
            variants = quotation_topic_variants(query)
            title = primary_document_classifications(selected['document']) if selected and not conflict else ()
            entire_quote = bool(variants and len(title) == 1 and role_matches(title[0].kind, 'quote')
                                and quote_header_topic_matches(selected['document'], title[0], variants))
            if entire_quote:
                page_count = pages[0]['page_count']
                if page_count > 8:
                    raise ValueError('quote page census exceeds native reading bound')
                anchors = tuple(range(1, page_count+1))
                requested = anchors
            else:
                anchors = (anchor,)
                requested = tuple(p for p in (anchor-1, anchor, anchor+1)
                                  if 1 <= p <= pages[0]['page_count'])
            if len(requested) != len(pages):
                pages = pdf_source_pages(document_root, tenant, metadata, requested)
                if len(pages) != len(requested):
                    raise ValueError('missing title context page')
        except (OSError, ValueError, SourceAccessError):
            if not _listing_relationship_excludes_source(rows, request, query):
                read_complete = False
            continue
        for page in pages:
            identifier = f"{source}::native-document-title::{page['page']:06d}::{page['text_sha256'][:16]}"
            meta = dict(metadata)
            meta.update(page=page['page'], section=f"Page {page['page']}", chunk_id=identifier,
                _entity_representation='native_pdf', _native_pdf_page_text_sha256=page['text_sha256'],
                _native_pdf_raw_text_sha256=page['raw_sha256'], _native_pdf_page_count=page['page_count'],
                _native_pdf_anchor_pages=anchors, _native_pdf_requested_pages=requested)
            native.append(dict(id=identifier, document=page['text'], metadata=meta))
    read_complete = bool(read_complete and not _listing_source_unresolved(
        grouped, native, fallback_sources, query, requested_kind, requested_proposal))
    receipts = semantic_batch.receipts if semantic_batch is not None else ()
    all_passages = _document_type_listing_passages(grouped, native, query, requested_kind, requested_proposal, receipts)
    from financial_documents import listing_result_limit
    limit = listing_result_limit(query)
    if limit:
        all_passages = all_passages[:limit]
    catalog_sha = listing_result_digest(grouped, all_passages, semantic_candidates, semantic_batch, semantic_gap,
                                        semantic_failure)
    if page_offset and expected_catalog != catalog_sha:
        return DocumentTypeListingEvidence((), False, len(grouped), count,
            'document_type_catalog_changed', witness, tuple(native), tenant, read_complete,
            tuple(fallback_sources), False, page_offset, catalog_sha, requested_kind, requested_proposal,
            semantic_candidates, semantic_batch, semantic_gap, semantic_failure)
    display_complete = page_offset + MAX_DOCUMENT_TYPE_RESULTS >= len(all_passages)
    passages = tuple(replace(fact, citation_id=str(index+1)) for index, fact in
                     enumerate(all_passages[page_offset:page_offset+MAX_DOCUMENT_TYPE_RESULTS]))
    finished = bool(complete and read_complete and display_complete and not semantic_gap)
    reason = ('' if finished else 'document_type_display_limit' if not display_complete
              else 'document_type_scan_incomplete')
    return DocumentTypeListingEvidence(passages, finished, len(grouped), count,
        reason, witness, tuple(native), tenant, read_complete, tuple(fallback_sources),
        display_complete, page_offset, catalog_sha, requested_kind, requested_proposal,
        semantic_candidates, semantic_batch, semantic_gap, semantic_failure)


def validate_document_type_listing(evidence, query):
    if not isinstance(evidence, DocumentTypeListingEvidence) or not evidence.witness:
        return False
    grouped = {}
    for row in evidence.witness.rows:
        grouped.setdefault(row['metadata']['source_id'], []).append(row)
    request = listing_request(query, evidence.requested_kind, evidence.requested_proposal)
    from topic_discovery import find_topic_matches
    excluded = frozenset(find_topic_matches(grouped, request.topic_groups)) if request and request.topic_groups else frozenset()
    if not _valid_semantic_batch(grouped, evidence.tenant_id, request, excluded,
                                 evidence.semantic_candidates, evidence.semantic_batch,
                                 evidence.semantic_gap, evidence.semantic_failure):
        return False
    receipts = evidence.semantic_batch.receipts if evidence.semantic_batch is not None else ()
    all_expected = _document_type_listing_passages(grouped, evidence.native_rows, query, evidence.requested_kind,
                                                   evidence.requested_proposal, receipts)
    from financial_documents import listing_result_limit
    limit = listing_result_limit(query)
    if limit:
        all_expected = all_expected[:limit]
    expected = tuple(replace(fact, citation_id=str(index+1)) for index, fact in
                     enumerate(all_expected[evidence.page_offset:evidence.page_offset+MAX_DOCUMENT_TYPE_RESULTS]))
    pdf_sources = {source for source, rows in grouped.items()
                   if rows[0]['metadata'].get('mime_type') == 'application/pdf'}
    native_sources = tuple(row['metadata']['source_id'] for row in evidence.native_rows if row['metadata'].get('page') == 1)
    from financial_documents import (document_type_listing, primary_document_classifications,
                                     role_matches, possible_document_type_mention)
    if request and not request.kind:
        pdf_sources = set()
    excluded_sources = {source for source in pdf_sources
                        if _listing_relationship_excludes_source(grouped[source], request, query)}
    fallback_valid = bool(request) and all(
        indexed_ocr_alternate_title(grouped[source], request)
        for source in evidence.fallback_sources if source in grouped)
    return bool(expected and fallback_valid and evidence.passages == expected
        and set(native_sources) <= pdf_sources and len(native_sources) == len(set(native_sources))
        and set(evidence.fallback_sources) <= pdf_sources - set(native_sources)
        and len(evidence.fallback_sources) == len(set(evidence.fallback_sources))
        and (not evidence.candidate_reads_complete or
             (set(native_sources) | set(evidence.fallback_sources) | excluded_sources == pdf_sources
              and not _listing_source_unresolved(grouped, evidence.native_rows,
                                                  evidence.fallback_sources, query, evidence.requested_kind, evidence.requested_proposal)))
        and type(evidence.page_offset) is int and evidence.page_offset >= 0
        and evidence.catalog_sha256 == listing_result_digest(grouped, all_expected, evidence.semantic_candidates,
                                                              evidence.semantic_batch, evidence.semantic_gap,
                                                              evidence.semantic_failure)
        and evidence.display_complete == (evidence.page_offset + MAX_DOCUMENT_TYPE_RESULTS >= len(all_expected))
        and evidence.complete == bool(evidence.witness.storage_complete and evidence.candidate_reads_complete
                                      and evidence.display_complete and not evidence.semantic_gap)
        and all(f.tenant_id == evidence.tenant_id for f in expected))


def render_document_type_listing(evidence, query, hebrew):
    from financial_documents import primary_document_classifications, document_kind_label, listing_issue_date
    if not evidence.passages and evidence.complete:
        request = listing_request(query, evidence.requested_kind, evidence.requested_proposal)
        if request and request.topic_groups:
            return ('לא נמצאו התאמות לביטויי הנושא שנבדקו; ייתכן שמסמכים בניסוח אחר לא אותרו.' if hebrew else
                    'No matches to the checked topic phrases were found; documents using different wording may have been missed.')
        if request and request.period:
            return (f'לא נמצאו מסמכים תואמים לחודש {request.period} בסריקה שהושלמה.' if hebrew else
                    f'No matching documents for {request.period} were found in the completed scan.')
        return ('לא נמצאו מסמכים תואמים בסריקה שהושלמה.' if hebrew else
                'No matching documents were found in the completed scan.')
    if not evidence.passages:
        lines = [('לא ניתן לקבוע אם קיימים מסמכים מתאימים.' if hebrew else
                  'No absence can be established.')]
    else:
        lines = [('המשך המסמכים התואמים:' if hebrew else 'More matching documents:')
                 if evidence.page_offset else
                 ('המסמכים התואמים שנמצאו:' if hebrew else 'Matching documents:')]
    request = listing_request(query, evidence.requested_kind, evidence.requested_proposal)
    if request and request.period and lines:
        lines[0] += ' ' + request.period
    grouped = _group_witness(evidence)
    native = _native_listing_groups(evidence.native_rows)
    for passage in evidence.passages:
        from financial_documents import quotation_amount_topic
        source_rows = grouped.get(passage.source_id, ())
        if source_rows and source_rows[0]['metadata'].get('mime_type') == 'application/pdf' and request and request.kind:
            title_row, conflict = _native_listing_title(native.get(passage.source_id, ()))
            title_text = title_row['document'] if title_row and not conflict else ''
        else:
            title_text = source_body(source_rows) if source_rows else passage.text
        identities = primary_document_classifications(title_text)
        identity = identities[0] if len(identities) == 1 else None
        if request and request.kind and identity is None:
            return ''
        label = document_kind_label(identity.kind if identity else '', hebrew)
        name = passage.metadata['source_relative_path']
        number = ' '+identity.number if identity and identity.number else ''
        issued = listing_issue_date(title_text) if request and (request.year or request.date or request.date_start or request.date_end) else None
        date_display = f' — {issued.strftime("%d/%m/%Y")}' if issued else ''
        from financial_documents import quotation_amount_topic, quote_total_literal
        total = quote_total_literal(passage.text) if quotation_amount_topic(query) else ''
        if total:
            lines.append(f'{label}{number} — {name}: {total}. [{passage.citation_id}]')
        else:
            relevance = f': {passage.source_claim}' if request and request.topic_groups else ''
            lines.append(f'{label}{number} — {name}{date_display}{relevance}. [{passage.citation_id}]')
    if request and request.topic_groups and evidence.semantic_batch is None:
        lines.append('החיפוש בדק ביטויי נושא במקור; מסמכים בניסוח אחר עשויים שלא להופיע.'
                     if hebrew else 'The search checked topic phrases in source text; documents using different wording may be missing.')
    if evidence.semantic_failure:
        lines.append('בדיקת מועמדים נוספים נכשלה עקב תקלה טכנית; ייתכן שקיימים מסמכים מתאימים נוספים.'
                     if hebrew else 'A technical error prevented review of additional candidates; more matching documents may exist.')
    elif evidence.semantic_gap:
        lines.append('בדיקת מועמדים בעלי ניסוח שונה לא הושלמה; ייתכן שקיימים מסמכים מתאימים נוספים.'
                     if hebrew else 'Review of candidates using different wording was incomplete; additional matching documents may exist.')
    if evidence.semantic_batch is not None and evidence.semantic_batch.unresolved_ids:
        lines.append('לא ניתן היה לאמת את ההתאמה לנושא בחלק מהמועמדים שנבדקו.'
                     if hebrew else 'The topical relevance of some reviewed candidates could not be verified.')
    if not evidence.display_complete:
        lines.append('הרשימה ארוכה ממגבלת התצוגה; כתבו ״הצג את ההמשך״ כדי לראות את יתר המסמכים שנמצאו.'
                     if hebrew else 'The list exceeds the display limit; ask “show the next page” to see the remaining matches.')
    if not evidence.witness.storage_complete:
        lines.append('לא ניתן היה לסרוק את כל רשומות המקורות; ייתכן שקיימים מסמכים נוספים.'
                     if hebrew else 'Source enumeration did not finish; additional documents may exist.')
    if not evidence.candidate_reads_complete:
        grouped = {}
        for row in evidence.witness.rows:
            grouped.setdefault(row['metadata']['source_id'], []).append(row)
        request = listing_request(query, evidence.requested_kind, evidence.requested_proposal)
        if request and not request.kind and request.topic_groups:
            # Topic discovery reads indexed source content, not PDF title
            # pages. Explain the actual content-coverage gap rather than
            # claiming that a document type or title could not be determined.
            unresolved = {source: rows for source, rows in grouped.items()
                if _listing_source_unresolved({source: rows}, evidence.native_rows, evidence.fallback_sources,
                                              query, evidence.requested_kind, evidence.requested_proposal,
                                              ignore_missing=True)}
            incomplete = {source for source, rows in unresolved.items() if not _listing_index_complete(rows)}
            scanned = set()
            for source, rows in unresolved.items():
                metadata = rows[0]['metadata']
                if metadata.get('mime_type') != 'application/pdf':
                    continue
                try:
                    manifest = json.loads(metadata.get('pdf_page_manifest_json') or 'null')
                except (TypeError, ValueError):
                    manifest = None
                if (isinstance(manifest, list) and any(isinstance(page, dict)
                        and not page.get('native_text_chars') and page.get('embedded_image_count')
                        for page in manifest)):
                    scanned.add(source)
            if incomplete:
                lines.append('לא ניתן היה לבדוק את כל הטקסט המאונדקס בחלק מהמסמכים.'
                             if hebrew else 'Some indexed document text could not be checked completely.')
            if scanned:
                lines.append('בחלק ממסמכי ה-PDF יש עמודים סרוקים שזיהוי הטקסט שלהם עשוי להיות חלקי.'
                             if hebrew else 'Some PDF pages are scanned and their recognized text may be incomplete.')
            if set(unresolved) - incomplete - scanned or not unresolved:
                lines.append('לא ניתן היה לוודא את כיסוי הטקסט בחלק מהמקורות.'
                             if hebrew else 'Text coverage could not be verified for some sources.')
        else:
            pdfs = {source for source, rows in grouped.items()
                    if rows[0]['metadata'].get('mime_type') == 'application/pdf'}
            native = {row['metadata']['source_id'] for row in evidence.native_rows
                      if row['metadata'].get('page') == 1}
            excluded = {source for source in pdfs if _listing_relationship_excludes_source(grouped[source], request, query)}
            if pdfs - native - set(evidence.fallback_sources) - excluded:
                lines.append('לא ניתן היה לקרוא או לאמת כותרת של מקור אחד או יותר; ייתכן שקיימים מסמכים נוספים.'
                             if hebrew else 'One or more source titles could not be read or verified; additional documents may exist.')
            if _listing_source_unresolved(grouped, evidence.native_rows, evidence.fallback_sources,
                                          query, evidence.requested_kind, evidence.requested_proposal, ignore_missing=True):
                lines.append('לא ניתן היה לקבוע את סוג המסמך או לאמת את המסננים בחלק מהמקורות.'
                             if hebrew else 'The document type or requested filters could not be verified for some sources.')
    return '\n'.join(lines)
