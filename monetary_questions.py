"""Description-led monetary reading inside the existing financial reading route.

Indexed records discover authorized sources. Native source fields retain row
ownership; the common quote consumer independently reopens every selected page.
No amount, entity, document identity or tax rate comes from a request fixture.
"""
from dataclasses import dataclass, replace
import re

from document_reading import (FinancialEvidence, ScanWitness, scan_authorized,
    source_body, sources_from_rows, make_passage, digest)

MAX_MONETARY_PAGES = 8
MAX_MONETARY_SOURCES = 4


@dataclass(frozen=True)
class MonetaryReadOutcome:
    source_id: str
    source_sha256: str
    state: str
    requested_pages: tuple = ()
    page_hashes: tuple = ()


@dataclass(frozen=True)
class MonetaryEvidence(FinancialEvidence):
    native_rows: tuple = ()
    geometry_pages: tuple = ()  # (source id, authenticated page payload)
    read_outcomes: tuple[MonetaryReadOutcome, ...] = ()
    # A request-local model adjudication may select a literal source-owned
    # description as the retrieval anchor.  It cannot change the requested
    # operation, monetary role, method, date, or other explicit filters.
    semantic_request: object = None
    semantic_source_id: str = ''
    interpreted_request: object = None
    interpreted_subject: str = ''
    # Coverage of the selected request, separate from tenant-wide parsing.
    answer_coverage_complete: bool | None = None


def evidence_monetary_request(evidence, query):
    from financial_semantics import monetary_question
    original = monetary_question(query)
    if getattr(evidence, 'interpreted_request', None) is not None:
        from financial_semantics import validated_semantic_monetary_question
        interpreted = evidence.interpreted_request
        original = validated_semantic_monetary_question(
            query, getattr(interpreted, 'operation', ''), evidence.interpreted_subject)
        if original != interpreted:
            return None
    proposed = getattr(evidence, 'semantic_request', None)
    if proposed is None:
        return original
    if original is None or type(proposed) is not type(original):
        return None
    if replace(proposed, description=original.description, terms=original.terms) != original:
        return None
    source = getattr(evidence, 'semantic_source_id', '')
    grouped = {}
    for row in evidence.witness.rows if evidence.witness else ():
        grouped.setdefault(row['metadata']['source_id'], []).append(row)
    if (not source or source not in grouped or not proposed.description or
            proposed.terms != tuple(dict.fromkeys(_words(proposed.description))) or
            candidate_ids(grouped, proposed) != (source,)):
        return None
    # A PDF line can be split in indexed reading order by a neighboring price
    # cell. The complete description must still be a native parsed item from
    # this source; later verification reopens and checks that page separately.
    if not any(sid == source and any(item.description == proposed.description for item in record.items)
               for sid, record in parsed_pages(evidence)):
        return None
    from semantic_monetary_guard import independently_matches_monetary_request
    if not independently_matches_monetary_request(original, source, grouped):
        return None
    return proposed


def _words(value):
    return tuple(re.findall(r'[A-Za-z0-9\u05d0-\u05ea]+', str(value).casefold()))


_GRAMMATICAL_HEADS = frozenset({'פרויקט','לקוח','לקוחה','אתר','מוקד','מרכז','מועצה',
    'אזורית','טיפול','תקלה','תיקון','התקנה','אספקה','החלפה','מערכת','שירות','עבודה',
    'טכנאי','טכנאית','טכני','יעוץ','ייעוץ'})
_WORK_UMBRELLAS = frozenset({'טיפול','שירות','עבודה'})
_WORK_DETAILS = frozenset({'תקלה','תיקון','התקנה','אספקה','החלפה','טכנאי','טכנאית'})
_LOCATION_HEADS = _GRAMMATICAL_HEADS - {'מערכת','אזורית'}
# Hebrew articles and conjunctions attach to ordinary equipment and service
# nouns as well as work/site heads. Proper names remain excluded: removing a
# letter from an arbitrary name could select a different owner.
_TECHNICAL_LABELS = frozenset({'מולטימדיה','מצלמה','מיקרופון','בקר','מסך','מקרן','רמקול','תל'})


def _variants(word):
    # Only general work/site labels admit clitic normalization. A proper name
    # such as כרמל must never silently become the different identity רמל.
    variants={word}
    for prefix in ('ו','ב','ה','כ','ל','מ','ש','וה','בה','לה','מה','של'):
        if word.startswith(prefix) and word[len(prefix):] in _GRAMMATICAL_HEADS | _TECHNICAL_LABELS:
            variants.add(word[len(prefix):])
    # Common work nouns in Hebrew construct form: replacement/installation
    # of an object. Restrict this to known work nouns, never arbitrary names.
    for candidate in tuple(variants):
        for prefix in ('', 'ו', 'ה', 'ב', 'ל', 'מ', 'ש'):
            stem=candidate[len(prefix):] if candidate.startswith(prefix) else ''
            if stem.endswith('ת') and stem[:-1]+'ה' in _WORK_DETAILS:
                variants.update((stem,stem[:-1]+'ה'))
    # Hebrew full/defective spelling of the common consulting work noun;
    # proper names do not receive arbitrary letter insertion or removal.
    if variants & {'יעוץ','ייעוץ'}:
        variants.update(('יעוץ','ייעוץ'))
    return variants


def _word_variants(words):
    output=[]
    for index,word in enumerate(words):
        variants=_variants(word)
        # Locative ב is licensed by the preceding work/site noun. This is
        # not blanket removal of name initials (כרמל/רמל, הראל/ראל, etc.).
        if (index and len(word)>=4 and word.startswith('ב')
                and _variants(words[index-1]) & _LOCATION_HEADS):
            variants.add(word[1:])
        output.append(variants)
    return tuple(output)


def matches_terms(terms, text):
    observed = _word_variants(_words(text))
    wanted = _word_variants(tuple(terms))
    # An umbrella such as "treatment of a fault" describes the operation;
    # the specific work and its owner must still match the actual source.
    detailed=any(variants & _WORK_DETAILS for variants in wanted)
    required=tuple(variants for variants in wanted
                   if not (detailed and variants & _WORK_UMBRELLAS))
    return bool(required) and all(any(term & variants for variants in observed)
                                 for term in required)


def _explicit_matches(request, rows, record=None):
    reference=request.explicit_source
    if reference is None:return True
    from financial_semantics import FinancialDocumentReference,document_references
    if isinstance(reference,FinancialDocumentReference):
        from financial_documents import role_matches
        if record is not None:
            return role_matches(record.document_kind,reference.kind) and reference.number==record.document_number
        # Index fragments and filenames propose candidates only. The reopened
        # native page below must independently prove the primary identity.
        return any(role_matches(item.kind,reference.kind) and item.number==reference.number
                   for item in document_references(source_body(rows)+' '+
                       str(rows[0]['metadata'].get('source_relative_path',''))))
    from document_query import reference_matches
    metadata=rows[0]['metadata']
    return any(reference_matches(reference,str(metadata.get(key,'')),metadata.get('mime_type',''))
               for key in ('source_relative_path','title','display_name'))


def candidate_ids(grouped, request, *, recipient_subject=''):
    from quotation_recipient_witness import indexed_recipient_candidate
    return tuple(source for source,rows in grouped.items()
        if _explicit_matches(request,rows) and (not request.terms or
            matches_terms(request.terms, source_body(rows)+' '+str(rows[0]['metadata'].get('title',''))
                          +' '+str(rows[0]['metadata'].get('source_relative_path',''))) or
            (recipient_subject and request.operation == 'subject_work' and
             indexed_recipient_candidate(recipient_subject, source_body(rows)))))


def parsed_pages(evidence):
    from monetary_native_pdf import parse_monetary_page
    output = []
    for source, payload in evidence.geometry_pages:
        record = parse_monetary_page(payload)
        if record is not None:
            output.append((source, record))
    return tuple(output)


def selected_items(evidence, request, *, recipient_subject=''):
    grouped = {}
    for row in evidence.witness.rows:
        grouped.setdefault(row['metadata']['source_id'], []).append(row)
    selected = []
    for source, record in parsed_pages(evidence):
        rows = grouped.get(source, ())
        if not rows:
            continue
        if not _explicit_matches(request,rows,record):continue
        if request.document_kind:
            from financial_documents import role_matches
            if not role_matches(record.document_kind, request.document_kind):
                continue
        context = record.service_context_text
        for index,item in enumerate(record.items):
            from quotation_recipient_witness import propose_quotation_recipient
            page = next((p for sid,p in evidence.geometry_pages
                         if sid == source and p.get('page') == record.page), None)
            recipient = (propose_quotation_recipient(page, recipient_subject, source)
                         if recipient_subject and request.operation == 'subject_work' else None)
            if ((recipient and recipient.item_index == index) or
                (not request.terms and request.explicit_source is not None) or
                (matches_terms(request.terms, context+' '+item.description)
                 and any(matches_terms((term,), item.description) for term in request.terms))):
                selected.append((source, record, index, item))
    return tuple(selected)


def _context_labels(record,request,item):
    missing=tuple(term for term in request.terms if not matches_terms((term,),item.description))
    labels=[]
    for line in record.source_text.splitlines():
        if (line.strip() and len(line)<=180 and missing
            and any(matches_terms((term,),line) for term in missing)
            and matches_terms(_words(line),record.service_context_text)):
            labels.append(line.strip())
    return tuple(dict.fromkeys(labels))


def _passages_for_native(row, record, start_id, request):
    """Only exact source substrings become quotes, including price conditions."""
    text = row['document']
    start,end = record.quote_span
    for item in record.items:
        for label in _context_labels(record,request,item):
            position=text.find(label)
            if position>=0:start=min(start,position)
    from financial_semantics import _PAYMENT_HEADING
    payment=_PAYMENT_HEADING.search(text)
    if payment and payment.start()>=start:
        end=len(text)
    if not (0 <= start < end <= len(text)):
        return ()
    claim = text[start:end]
    # Split only on actual line boundaries. All of the selected source region
    # remains quoted, including qualifiers and any unrequested sibling items.
    pieces=[]; chunk=''
    for line in claim.splitlines(keepends=True):
        if len(line)>780 or len(line.split())>78:
            return ()
        if chunk and (len(chunk+line)>780 or len((chunk+line).split())>78):
            pieces.append(chunk);chunk=''
        chunk+=line
    if chunk:pieces.append(chunk)
    if len(pieces)>6:
        return ()
    return tuple(replace(make_passage(text, part, row['metadata'], row['id'], start_id+i),
                         representation='native_pdf') for i,part in enumerate(pieces))


def _bounded_payment_context_label(label, missing):
    # Keep only an exact labelled service owner. An adjacent field may be
    # excluded only when its complete remainder is a recognizable contact
    # field; any extra prose could be a condition and is not discarded.
    from monetary_presentation import _FIELD_BOUNDARY,_CONTACT_VALUE
    boundary=_FIELD_BOUNDARY.search(label)
    owner=label
    if boundary:
        suffix=label[boundary.start():]
        field=_FIELD_BOUNDARY.match(suffix)
        if not field or not _CONTACT_VALUE.fullmatch(suffix[field.end():].strip()):
            return ''
        owner=label[:boundary.start()].rstrip()
    match=re.match(r'(?i)^\s*(?:customer|service(?:\s+(?:site|location))?|project|site|'
                   r'work(?:\s+site)?|לקוח(?:ה)?|אתר(?:\s+שירות)?|פרויקט|'
                   r'מקום\s+ביצוע|תיאור\s+עבודה)\s*:\s*',owner)
    if not match:
        return ''
    actual=_word_variants(_words(owner[match.end():]))
    wanted=_word_variants(tuple(missing))
    if (not actual or not wanted or
            not all(any(observed & expected for expected in wanted) for observed in actual) or
            not all(any(observed & expected for observed in actual) for expected in wanted)):
        return ''
    return owner


def _bounded_unit_service_heading(label, missing, item, record):
    """Bind a unit heading to one item when its service terms agree."""
    if len(record.items) != 1 or not missing:
        return ''
    if not all(re.fullmatch(r'(?i)שעות|שעה|hours?|hrs?', term) for term in missing):
        return ''
    match = re.fullmatch(
        r'(?i)\s*(?:הצעת\s+מחיר|quotation|quote|estimate)\s+'
        r'(?:שעות|שעה|hours?|hrs?)\s+(.+?)\s*', str(label))
    if not match or not re.fullmatch(r'[A-Za-z\u05d0-\u05ea\s]+', match.group(1)):
        return ''
    terms = _words(match.group(1))
    return str(label).strip() if terms and matches_terms(terms, item.description) else ''


def _unsafe_payment_condition_excerpt(value):
    from monetary_presentation import _CONTACT_VALUE
    return bool(_CONTACT_VALUE.search(str(value)) or
                re.search(r'(?i)\b(?:bank\s+)?account(?:\s*(?:number|no\.?|#|:))?\s*[:#-]?\s*\d|מס(?:פר)?\s*חשבון',str(value)))


def _unsafe_payment_item_excerpt(value):
    from monetary_presentation import _CONTACT_VALUE
    text=str(value)
    if re.search(r'(?i)\b(?:bank\s+)?account(?:\s*(?:number|no\.?|#|:))?\s*[:#-]?\s*\d|מס(?:פר)?\s*חשבון',text):
        return True
    for match in _CONTACT_VALUE.finditer(text):
        token=match.group()
        prefix=text[max(0,match.start()-24):match.start()]
        if ('@' in token or re.match(r'(?i)(?:https?://|www\.)',token) or
                re.search(r'(?i)(?:\b(?:tel|telephone|phone|call|contact)|טלפון|נייד)\s*[:.\-]?\s*$',prefix)):
            return True
        if re.search(r'(?i)(?:(?:\b(?:model|serial|sku)|דגם)\s*[:#-]?\s*(?:[A-Za-z]{1,8}-)?|'
                     r'(?:USD|EUR|ILS|₪|[$€£])\s*)$',prefix):
            continue
        return True
    return False


def _explicit_other_item_condition(text, record, selected_indices):
    # One uniquely identified other item is the grammatical subject of a
    # complete, narrowly bounded subject-to proposition. This does not prove
    # that the selected item is unconditional.
    candidate=str(text).strip()
    if not candidate or '\n' in candidate:
        return ''
    owners=[]
    for index,item in enumerate(record.items):
        if index in selected_indices:continue
        name=' '.join(item.description.split())
        if not name or sum(' '.join(other.description.split()).casefold()==name.casefold()
                           for other in record.items)!=1:
            continue
        word=r'[^\W\d_]+(?:[-\u2019\'][^\W\d_]+)?'
        pattern=(r'(?i)^'+re.escape(name)+r'\s+(?:is\s+)?subject\s+to\s+'+
                 word+r'(?:\s+'+word+r')?[.!?]$')
        if re.fullmatch(pattern,candidate):
            owners.append(index)
    return candidate if len(owners)==1 else ''


def _monetary_context_inventory(record, selected, request, payload=None):
    # Reuse the subject-work source-span and scope checks. Nomination does not
    # establish item ownership; all other unclassified context remains an
    # explicit source-level uncertainty.
    from subject_native_work import (_context_spans,_nominated_qualification,
        _display_context_span,_context_scope,_ordinary_title_block)
    from monetary_presentation import material_monetary_conditions
    spans=_context_spans(record)
    if spans is None:
        return None
    shown={' '.join(value.split()) for _i,item in selected
           for value in material_monetary_conditions(record,item)}
    roles=[]
    uncertain=False
    item_indices={index for index,_item in selected}
    for ordinal,(_start,_end,block) in enumerate(spans):
        if _ordinary_title_block(block,record):
            continue
        from subject_reading import _instruction_like
        if _instruction_like(str(block)):
            if _nominated_qualification(block):return None
            uncertain=True
            continue
        if not _nominated_qualification(block):
            uncertain=True
            continue
        residual=str(block)
        removed=False
        for condition in shown:
            pattern=r'\s+'.join(re.escape(piece) for piece in condition.split())
            hits=list(re.finditer(pattern,residual)) if pattern else []
            if len(hits)==1:
                residual=residual[:hits[0].start()]+' '+residual[hits[0].end():]
                removed=True
        if removed and not _nominated_qualification(residual):
            uncertain=True
            continue
        if removed:
            other=_explicit_other_item_condition(residual,record,item_indices)
            if other and not _unsafe_payment_condition_excerpt(other) and record.source_text.count(other)==1:
                # Cite the exact other-item condition without asserting
                # whether it also affects the requested item.
                start=record.source_text.find(other)
                roles.append((f'source_elsewhere_context:{ordinal}',start,start+len(other),other))
                uncertain=True
                continue
        scope,_proof=_context_scope(block,record,item_indices)
        if scope=='elsewhere':
            continue
        display=_display_context_span(block,record)
        if not display or _unsafe_payment_condition_excerpt(display):
            return None
        if ' '.join(display.split()) in shown:
            continue
        if record.source_text.count(display)!=1:
            return None
        start=record.source_text.find(display)
        # A common native RTL layout extracts a numeric run before its Hebrew
        # label. Cite each literal run and label the visual-order reading as a
        # reconstruction, not as a verbatim source quotation.
        if re.match(r'^\s*(?:[+\-]\s+)?[0-9]+\s+(?:[+\-]\s+)?', display) and re.search(r'[֐-׿]', display):
            from monetary_native_pdf import native_rtl_numeric_context
            reconstructed=native_rtl_numeric_context(payload, display)
            if reconstructed is None:
                # Keep independently verified prices, while clearly limiting
                # the source qualification whose visual order is ambiguous.
                labels=[m for m in re.finditer(r'[֐-׿]+(?:\s+[֐-׿]+)*',display)
                        if record.source_text.count(m.group())==1]
                if len(labels)!=1:
                    return None
                a=start+labels[0].start()
                b=start+labels[0].end()
                roles.append((f'source_context_limited:{ordinal}',a,b,record.source_text[a:b]))
            else:
                _reading, literal_spans=reconstructed
                for a,b in literal_spans:
                    roles.append((f'source_reconstructed_context:{ordinal}',a,b,record.source_text[a:b]))
        else:
            roles.append((f'source_context:{ordinal}',start,start+len(display),display))
    if (len(roles)>4 or sum(len(role[3]) for role in roles)>1200 or
            any(len(role[3])>780 or len(role[3].split())>78 for role in roles)):
        return None
    return tuple(roles),uncertain


def _monetary_role_inventory(record, page, selected, request, source='', tenant='', query='', include_identity_date=False):
    # Source-bound role inventory for monetary reading. The independent
    # consumer checks role coverage, original offsets and rendered claim links.
    from monetary_native_pdf import native_cell_literal_spans
    from monetary_presentation import material_monetary_conditions, requested_source_details
    from financial_semantics import (canonical_payment_method,_BUSINESS_DATE,
                                     _PAYMENT_METHOD,_money_matches,_CURRENCY,_CANCELLED)
    if request.operation not in {'payment_status','description_price'} or not selected:
        return ()
    roles=[('identity',record.document_identity_word_ids)]
    extras=[]
    def literal(role,value,containers=None):
        pattern=r'\s+'.join(re.escape(piece) for piece in str(value).split())
        found=list(re.finditer(pattern,page['text'])) if pattern else []
        if len(found)!=1 or (containers is not None and not any(found[0].group() in c for c in containers)):
            return False
        m=found[0]
        end=m.end()
        if ':condition:' in role and page['text'][end:end+1]==';':
            end+=1
        extras.append((role,m.start(),end,page['text'][m.start():end]))
        return True
    if include_identity_date and record.issue_date:
        if not literal('identity_date',record.issue_date.strftime('%d/%m/%Y')):
            return ()
    if len(record.items)==1:
        roles.extend((('vat_label',record.vat_label_word_ids),
                      ('tax',record.tax.word_ids),('payable',record.payable.word_ids)))
    elif record.payable:
        roles.append(('payable',record.payable.word_ids))
    for index,item in selected:
        if _unsafe_payment_item_excerpt(item.description):
            return ()
        unit_heading = False
        if request.terms and not matches_terms(request.terms,item.description):
            missing=tuple(term for term in request.terms if not matches_terms((term,),item.description))
            labels=_context_labels(record,request,item)
            unit_heading = any(_bounded_unit_service_heading(label,missing,item,record) for label in labels)
            bounded=[(_bounded_payment_context_label(label,missing) or
                      _bounded_unit_service_heading(label,missing,item,record)) for label in labels]
            if not bounded or any(not label for label in bounded):
                return ()
            if any(not literal(f'item:{index}:context:{n}',label)
                   for n,label in enumerate(bounded)):
                return ()
        for n,condition in enumerate(material_monetary_conditions(record,item)):
            if _unsafe_payment_condition_excerpt(condition):
                return ()
            if not literal(f'item:{index}:condition:{n}',condition,record.qualifiers):
                return ()
        roles.append((f'item:{index}:description',item.description_word_ids))
        if request.role_requested != 'unit_price':
            roles.append((f'item:{index}:line_total',item.line_total_word_ids))
        if 'quantity' in request.requested_fields or request.role_requested=='unit_price' or unit_heading:
            roles.append((f'item:{index}:quantity',item.quantity_word_ids))
        if request.role_requested=='unit_price' or unit_heading:
            roles.append((f'item:{index}:unit_price',item.unit_price_word_ids))
    for field,value in requested_source_details(record,request.requested_fields):
        if value is None:
            continue
        literal_value=value
        if field=='issue_date':
            literal_value=record.issue_date.strftime('%d/%m/%Y')
        if not literal('detail:'+field,literal_value):
            return ()
    payment_requested=(request.operation=='payment_status' or
        bool(re.search(r'\bpayment\b|(?<!\w)ה?תשלום(?!\w)',query,re.I)))
    show_payments=bool(payment_requested or record.document_kind in {'quote','quotation','estimate'})
    if show_payments and len(record.items)==1 and record.payment_table_present and not _CANCELLED.search(record.source_text):
        for index,payment in enumerate(record.payments):
            if request.payment_method and canonical_payment_method(payment.method)!=request.payment_method:
                continue
            if (payment.payment_date is None or _PAYMENT_PENDING.search(payment.context_text) or
                    (record.document_kind not in {'receipt','invoice_receipt'} and
                     not _PAYMENT_OCCURRED.search(payment.context_text))):
                continue
            if record.document_kind not in {'receipt','invoice_receipt'}:
                occurrence=_PAYMENT_OCCURRED.search(payment.context_text)
                if not occurrence or not literal(f'payment:{index}:occurrence',occurrence.group()):
                    return ()
            roles.extend(((f'payment:{index}:amount',payment.amount_word_ids),
                          (f'payment:{index}:date',payment.date_word_ids),
                          (f'payment:{index}:method',payment.method_word_ids)))
    elif show_payments and len(record.items)==1 and not record.payment_table_present:
        recorded=_recorded_payments(record,source,tenant)
        for index,payment in enumerate(recorded):
            if request.payment_method and canonical_payment_method(payment.method)!=request.payment_method:
                continue
            candidates=[]
            cursor=0
            for line in page['text'].splitlines(keepends=True):
                amounts=_money_matches(line)
                dates=list(_BUSINESS_DATE.finditer(line))
                methods=list(_PAYMENT_METHOD.finditer(line))
                if (len(amounts)==1 and len(dates)==1 and len(methods)<=1 and
                        amounts[0][1]==payment.amount and amounts[0][2]==payment.currency and
                        (methods[0].group() if methods else '')==payment.method):
                    from financial_semantics import _date_value
                    if _date_value(dates[0])==payment.date:
                        candidates.append((cursor,amounts[0][0],dates[0],methods[0] if methods else None))
                cursor+=len(line)
            if len(candidates)!=1:
                return ()
            cursor,amount,date,method=candidates[0]
            line_end=page['text'].find('\n',cursor)
            line=page['text'][cursor:line_end if line_end>=0 else len(page['text'])]
            if record.document_kind not in {'receipt','invoice_receipt'}:
                from financial_semantics import _PAYMENT_HEADING
                occurrence=_PAYMENT_OCCURRED.search(line)
                if occurrence:
                    a,b=cursor+occurrence.start(),cursor+occurrence.end()
                    extras.append((f'payment:{index}:occurrence',a,b,page['text'][a:b]))
                else:
                    heading=_PAYMENT_HEADING.search(page['text'])
                    if not heading or heading.start()>=cursor:
                        return ()
                    occurrence=_PAYMENT_OCCURRED.search(heading.group())
                    if not occurrence or not literal(f'payment:{index}:occurrence',occurrence.group()):
                        return ()
            currencies=[currency for currency in _CURRENCY.finditer(line)
                if ((currency.end()<=amount.start() and not line[currency.end():amount.start()].strip())
                    or (currency.start()>=amount.end() and not line[amount.end():currency.start()].strip()))]
            if len(currencies)!=1:
                return ()
            currency=currencies[0]
            a,b=cursor+min(amount.start(),currency.start()),cursor+max(amount.end(),currency.end())
            extras.append((f'payment:{index}:amount',a,b,page['text'][a:b]))
            for label,match in (('date',date),('method',method)):
                if match is None:continue
                a,b=cursor+match.start(),cursor+match.end()
                extras.append((f'payment:{index}:{label}',a,b,page['text'][a:b]))
    context=_monetary_context_inventory(record,selected,request,page)
    if context is None:
        return ()
    context_roles,_uncertain=context
    output=[]
    for role,ids in roles:
        spans=native_cell_literal_spans(page,ids)
        if not spans:
            return ()
        if role=='identity':
            from financial_documents import primary_document_identities,role_matches
            if len(spans)!=1:
                return ()
            a,b=spans[0]
            literal_title=page['text'][a:b]
            identities=primary_document_identities(literal_title)
            if (len(identities)!=1 or identities[0].span!=(0,len(literal_title)) or
                    identities[0].number!=record.document_number or
                    not role_matches(identities[0].kind,record.document_kind)):
                return ()
        output.extend((role,a,b,page['text'][a:b]) for a,b in spans)
    return tuple(output+extras+list(context_roles))


def _monetary_role_passages(row, record, payload, selected, request, start_id, query='', include_identity_date=False):
    inventory=_monetary_role_inventory(record,payload,selected,request,
                                      row['metadata']['source_id'],row['metadata']['tenant_id'],query,
                                      include_identity_date)
    if not inventory:
        return ()
    output=[]
    for role,a,b,claim in inventory:
        metadata=dict(row['metadata'],_monetary_role=role,_monetary_source_span=(a,b))
        passage=make_passage(claim,claim,metadata,row['id'],start_id+len(output))
        output.append(replace(passage,value='monetary_role:'+role,representation='native_pdf'))
    return tuple(output)


def collect_monetary(retriever, tenant, document_root, query, where=None, *, request_override=None, _authorized_scan=None, recipient_subject=''):
    from financial_semantics import monetary_question
    from source_access import pdf_source_pages, SourceAccessError
    from monetary_native_pdf import parse_monetary_page
    request=request_override if request_override is not None else monetary_question(query)
    if _authorized_scan is None:
        grouped,complete,count=scan_authorized(retriever,tenant,document_root,where=where)
        witness=ScanWitness(tuple(row for rows in grouped.values() for row in rows),complete)
    else:
        # Reuse only the already authorized request-local census. Reject any
        # cross-tenant, duplicate or scope-inconsistent row before selection.
        from source_access import source_metadata_complete
        if not isinstance(_authorized_scan,ScanWitness):
            raise ValueError('invalid authorized scan')
        grouped={}
        seen=set()
        for row in _authorized_scan.rows:
            metadata=row.get('metadata',{})
            identifier=str(row.get('id') or '')
            if (not identifier or identifier in seen or
                    not source_metadata_complete(metadata,tenant) or
                    (where and metadata.get('source_relative_path') != where['source'])):
                raise ValueError('invalid authorized scan census')
            seen.add(identifier)
            grouped.setdefault(metadata['source_id'],[]).append(row)
        for rows in grouped.values():
            rows.sort(key=lambda hit:(hit['metadata'].get('chunk_index',0),str(hit.get('id',''))))
        complete=_authorized_scan.storage_complete
        count=len(seen)
        witness=_authorized_scan
    sources,bodies_complete=sources_from_rows(grouped)
    evidence=MonetaryEvidence((),False,len(grouped),count,'monetary_evidence_not_found',
        witness,sources,tenant)
    if request is None:return evidence
    if request.unsupported_payment_method_filter:
        return replace(evidence,reason='monetary_method_scope_unsupported')
    candidates=candidate_ids(grouped,request,recipient_subject=recipient_subject)
    if len(candidates)>MAX_MONETARY_SOURCES:
        return replace(evidence,reason='monetary_sources_ambiguous')
    native=[];payloads=[];outcomes=[];used=0;failed=False
    for source in candidates:
        rows=grouped[source];metadata=dict(rows[0]['metadata'])
        if metadata.get('mime_type')!='application/pdf':
            outcomes.append(MonetaryReadOutcome(source, metadata.get('source_sha256',''), 'non_pdf'))
            failed=True;continue
        pages=tuple(sorted({row['metadata'].get('page') for row in rows
                            if type(row['metadata'].get('page')) is int}))
        if not pages or len(pages)+used>MAX_MONETARY_PAGES:
            outcomes.append(MonetaryReadOutcome(source, metadata.get('source_sha256',''), 'bounds'))
            failed=True;continue
        # Every selected page plus its immediate neighbors forms the same
        # bounded native window used by the independent final quote consumer.
        requested=tuple(sorted({p for anchor in pages for p in (anchor-1,anchor,anchor+1) if p>0}))
        if len(requested)>MAX_MONETARY_PAGES:
            outcomes.append(MonetaryReadOutcome(source, metadata.get('source_sha256',''), 'bounds'))
            failed=True;continue
        try:recovered=pdf_source_pages(document_root,tenant,metadata,requested,include_geometry=True)
        except (OSError,ValueError,SourceAccessError):
            outcomes.append(MonetaryReadOutcome(source, metadata.get('source_sha256',''), 'access_or_integrity'))
            failed=True;continue
        used+=len(recovered)
        if used>MAX_MONETARY_PAGES:
            outcomes.append(MonetaryReadOutcome(source, metadata.get('source_sha256',''), 'bounds'))
            failed=True;break
        actual=tuple(page['page'] for page in recovered)
        parsed=tuple(parse_monetary_page(payload) for payload in recovered)
        state=('invalid_read' if not recovered else
               'unsupported_pdf' if all(record is None for record in parsed) else
               'parsed_pdf' if all(record is not None for record in parsed) else 'mixed')
        hashes=tuple((payload['page'],payload['raw_sha256'],payload['text_sha256'])
                     for payload in recovered)
        outcomes.append(MonetaryReadOutcome(source, metadata.get('source_sha256',''), state, requested, hashes))
        for payload,record in zip(recovered,parsed):
            if record is None:
                failed=True;continue
            identifier=f"{source}::native-monetary::{payload['page']:06d}::{payload['text_sha256'][:16]}"
            meta=dict(metadata)
            meta.update(page=payload['page'],section=f"Page {payload['page']}",chunk_id=identifier,
                _entity_representation='native_pdf',_native_pdf_page_text_sha256=payload['text_sha256'],
                _native_pdf_raw_text_sha256=payload['raw_sha256'],_native_pdf_page_count=payload['page_count'],
                _native_pdf_anchor_pages=pages,_native_pdf_requested_pages=actual,
                _native_pdf_geometry_sha256=record.geometry_sha256)
            native.append(dict(id=identifier,document=payload['text'],metadata=meta))
            payloads.append((source,payload))
    evidence=replace(evidence,native_rows=tuple(native),geometry_pages=tuple(payloads),read_outcomes=tuple(outcomes))
    selected=selected_items(evidence,request,recipient_subject=recipient_subject)
    passages=[]
    role_words=role_chars=0
    role_budget_limited=False
    from evidence_quotes import (ANSWER_QUOTE_MAX_WORDS,ANSWER_QUOTE_MAX_CHARS,
                                 HARD_QUOTE_MAX_WORDS,HARD_QUOTE_MAX_CHARS)
    for row in native:
        matching=[record for source,record,_,_item in selected
                  if source==row['metadata']['source_id'] and record.page==row['metadata']['page']]
        if matching:
            record=matching[0]
            if request.operation in {'payment_status','description_price'}:
                page=next((payload for sid,payload in payloads
                           if sid==row['metadata']['source_id'] and payload['page']==record.page),None)
                items=[(i,item) for sid,r,i,item in selected
                       if sid==row['metadata']['source_id'] and r.page==record.page]
                additions=_monetary_role_passages(row,record,page,items,request,len(passages)+1,query,
                    len({sid for sid,_r,_i,_it in selected})>1) if page else ()
            else:
                additions=_passages_for_native(row,record,len(passages)+1,request)
            if request.operation in {'payment_status','description_price'} and additions:
                added_words=sum(len(p.source_claim.split()) for p in additions)
                added_chars=sum(len(p.source_claim) for p in additions)
                if (role_words+added_words>ANSWER_QUOTE_MAX_WORDS or
                        role_chars+added_chars>ANSWER_QUOTE_MAX_CHARS or
                        any(len(p.source_claim.split())>HARD_QUOTE_MAX_WORDS or
                            len(p.source_claim)>HARD_QUOTE_MAX_CHARS for p in additions)):
                    additions=()
                    role_budget_limited=True
                else:
                    role_words+=added_words
                    role_chars+=added_chars
            if not additions:failed=True
            passages.extend(additions)
    # The answer is explicitly scoped to the records actually examined.
    # A tenant-wide storage or body parsing shortfall is diagnostic, while a
    # failed matching candidate or missing selected item is a material gap.
    answer_coverage_complete=bool(not failed and selected
                  and {source for source,_,_,_ in selected}==set(candidates))
    complete=bool(complete and answer_coverage_complete and bodies_complete)
    from monetary_presentation import material_monetary_conditions
    unsafe_item=any(_unsafe_payment_item_excerpt(item.description) for _source,_record,_index,item in selected)
    unsafe_condition=any(_unsafe_payment_condition_excerpt(condition)
        for _source,record,_index,item in selected
        for condition in material_monetary_conditions(record,item))
    context_problem=(request.operation in {'payment_status','description_price'} and any(
        _monetary_context_inventory(record,[(i,it) for sid,rec,i,it in selected
            if sid==source and rec.page==record.page],request,
            next((payload for sid,payload in payloads if sid==source and payload['page']==record.page),None)) is None
        for source,record,_index,_item in selected))
    reason=('' if complete else 'monetary_evidence_partial' if passages else
            'monetary_qualification_excerpt_unsafe' if unsafe_condition else
            'monetary_item_excerpt_unsafe' if unsafe_item else
            'monetary_context_excerpt_unavailable' if context_problem or role_budget_limited else
            'monetary_source_representation_unsupported' if candidates and failed else
            'monetary_service_not_verified' if candidates else 'monetary_evidence_not_found')
    return replace(evidence,passages=tuple(passages),complete=complete,reason=reason,
                   answer_coverage_complete=answer_coverage_complete)


def validate_monetary_evidence(evidence,query, *, request_override=None, recipient_subject=''):
    from financial_semantics import monetary_question
    if not isinstance(evidence,MonetaryEvidence) or not evidence.witness:return False
    request=request_override if request_override is not None else evidence_monetary_request(evidence,query)
    if request is None or request.unsupported_payment_method_filter:return False
    grouped={}
    for row in evidence.witness.rows:
        grouped.setdefault(row['metadata']['source_id'],[]).append(row)
    rebuilt,bodies_complete=sources_from_rows(grouped)
    if evidence.sources!=rebuilt:return False
    candidates=set(candidate_ids(grouped,request,recipient_subject=recipient_subject))
    if len(candidates)>MAX_MONETARY_SOURCES:return False
    if len(evidence.geometry_pages)!=len(evidence.native_rows):return False
    expected=[]
    role_words=role_chars=0
    from evidence_quotes import (ANSWER_QUOTE_MAX_WORDS,ANSWER_QUOTE_MAX_CHARS,
                                 HARD_QUOTE_MAX_WORDS,HARD_QUOTE_MAX_CHARS)
    from monetary_native_pdf import parse_monetary_page
    for (source,payload),row in zip(evidence.geometry_pages,evidence.native_rows):
        record=parse_monetary_page(payload)
        if (record is None or source not in candidates or source!=row['metadata']['source_id']
            or payload['text']!=row['document'] or payload['page']!=row['metadata']['page']
            or record.geometry_sha256!=row['metadata'].get('_native_pdf_geometry_sha256')):return False
    selected=selected_items(evidence,request,recipient_subject=recipient_subject)
    for row in evidence.native_rows:
        records=[record for source,record,_,_ in selected
                 if source==row['metadata']['source_id'] and record.page==row['metadata']['page']]
        if records:
            record=records[0]
            if request.operation in {'payment_status','description_price'}:
                page=next((payload for sid,payload in evidence.geometry_pages
                           if sid==row['metadata']['source_id'] and payload['page']==record.page),None)
                items=[(i,item) for sid,r,i,item in selected
                       if sid==row['metadata']['source_id'] and r.page==record.page]
                additions=_monetary_role_passages(row,record,page,items,request,len(expected)+1,query,
                    len({sid for sid,_r,_i,_it in selected})>1) if page else ()
            else:
                additions=_passages_for_native(row,record,len(expected)+1,request)
            if request.operation in {'payment_status','description_price'} and additions:
                added_words=sum(len(p.source_claim.split()) for p in additions)
                added_chars=sum(len(p.source_claim) for p in additions)
                if (role_words+added_words>ANSWER_QUOTE_MAX_WORDS or
                        role_chars+added_chars>ANSWER_QUOTE_MAX_CHARS or
                        any(len(p.source_claim.split())>HARD_QUOTE_MAX_WORDS or
                            len(p.source_claim)>HARD_QUOTE_MAX_CHARS for p in additions)):
                    additions=()
                else:
                    role_words+=added_words
                    role_chars+=added_chars
            expected.extend(additions)
    if tuple(expected)!=evidence.passages:return False
    if evidence.complete and (not bodies_complete or not evidence.witness.storage_complete
        or {source for source,_,_,_ in selected}!=candidates):return False
    return bool(selected and expected)


def monetary_coverage(evidence,query):
    selected=selected_items(evidence,evidence_monetary_request(evidence,query))
    return dict(authorized_sources_scanned=evidence.examined_documents,
        indexed_chunks_examined=evidence.examined_chunks,
        selected_records=len({source for source,_,_,_ in selected}),
        selected_items=len(selected),selected_evidence_passages=len(evidence.passages),
        financial_coverage_complete=evidence.complete)


def _amount(value,currency):
    return f'{value:,.2f} {currency}'


_PAYMENT_OCCURRED = re.compile(r'\b(?:payment\s+(?:received|completed|confirmed)|received\s+(?:a\s+)?payment|confirmed\s+paid|amount\s+paid|paid\s+(?:on|by))\b|\bpaid\s*:|(?<![\u05d0-\u05ea])(?:שולם|שולמו|התקבל|התקבלו|נפרע|נפרעו)(?![\u05d0-\u05ea])',re.I)
_PAYMENT_PENDING = re.compile(r'\b(?:due|scheduled|planned|conditional|condition|pending|awaiting|expected|if|unless|provisional|unpaid|not\s+(?:yet\s+)?(?:paid|received)|to\s+be\s+paid|will\s+(?:be\s+)?pa(?:y|id)|subject\s+to)\b|(?:לא|טרם)\s+(?:שולם|שולמו|התקבל|נפרע)|(?<![\u05d0-\u05ea])אם(?![\u05d0-\u05ea])|לתשלום|ישולם|ישולמו|מתוכנ|מותנ|בכפוף',re.I)


@dataclass(frozen=True)
class _RecordedPayment:
    amount: object
    currency: str
    date: object
    method: str


def _recorded_payments(record,source,tenant):
    from financial_semantics import extract_financial_record,_PAYMENT_HEADING,_CANCELLED
    from financial_documents import canonical_document_kind
    kind=canonical_document_kind(record.document_kind)
    # Neither an amount due nor a credit note establishes settlement/refund.
    if kind=='credit' or _CANCELLED.search(record.source_text):return ()
    if record.payment_table_present:
        return tuple(_RecordedPayment(payment.amount,payment.currency,payment.payment_date,payment.method)
            for payment in record.payments if payment.payment_date is not None
            and not _PAYMENT_PENDING.search(payment.context_text)
            and (kind in {'receipt','invoice_receipt'} or _PAYMENT_OCCURRED.search(payment.context_text)))
    financial=extract_financial_record(record.source_text,source_id=source,
        title='',tenant_id=tenant,page=record.page,include_quotations=True)
    if not financial:return ()
    completed=[]
    for payment in financial.payments:
        a,b=payment.span
        # Read the actual source row and its governing heading, rather than
        # promoting a transfer method/date or an amount due into occurrence.
        line_start=record.source_text.rfind('\n',0,a)+1
        line_end=record.source_text.find('\n',b)
        line_end=len(record.source_text) if line_end<0 else line_end
        row=record.source_text[line_start:line_end]
        heading=_PAYMENT_HEADING.search(record.source_text)
        context_start=max(heading.start() if heading else 0,line_start-120)
        before=record.source_text[context_start:line_start]
        context=before+row
        if _PAYMENT_PENDING.search(context):continue
        if financial.document_kind not in {'receipt','invoice_receipt'} and not _PAYMENT_OCCURRED.search(context):continue
        # A nearby previous payment may use a different method. Only bind a
        # method when its own physical source line also contains this
        # payment's independently extracted amount and date.
        from financial_semantics import _PAYMENT_METHOD,_BUSINESS_DATE,_money_matches
        date_start,date_end=payment.date_span
        owner_start=record.source_text.rfind('\n',0,date_start)+1
        owner_end=record.source_text.find('\n',date_end)
        owner_end=len(record.source_text) if owner_end<0 else owner_end
        owner_line=record.source_text[owner_start:owner_end]
        amounts=_money_matches(owner_line)
        dates=list(_BUSINESS_DATE.finditer(owner_line))
        if (len(amounts)!=1 or len(dates)!=1 or
                (owner_start+dates[0].start(),owner_start+dates[0].end()) != (date_start,date_end)):
            continue
        methods=list(_PAYMENT_METHOD.finditer(owner_line))
        method=methods[0].group() if len(methods)==1 else ''
        completed.append(_RecordedPayment(amounts[0][1],amounts[0][2],payment.date,method))
    return tuple(completed)


def render_monetary(evidence,query,hebrew):
    from financial_semantics import monetary_question,extract_financial_record,canonical_payment_method
    from monetary_presentation import material_monetary_conditions, requested_source_details
    request=evidence_monetary_request(evidence,query)
    if request is None:
        return ''
    requested_fields=request.requested_fields
    all_selected=selected_items(evidence,request)
    cited={}
    for p in evidence.passages:cited.setdefault(p.source_id,[]).append(p.citation_id)
    role_mode=bool(evidence.passages and all(str(p.value).startswith('monetary_role:') for p in evidence.passages))
    represented={(p.source_id,p.metadata.get('page')) for p in evidence.passages}
    selected=tuple(item for item in all_selected if item[0] in cited and
                   (not role_mode or (item[0],item[1].page) in represented))
    if not selected:
        if evidence.reason=='monetary_sources_ambiguous':
            return ('נמצאו כמה מקורות אפשריים. נא לציין את הפרויקט או את סוג המסמך ומספרו.' if hebrew else
                    'Several sources may match. Please specify the project or the document type and number.')
        if evidence.reason=='monetary_context_excerpt_unavailable':
            return ('המקור כולל מידע רלוונטי שלא ניתן לצטט במלואו בבטחה; לכן לא ניתן לאמת את הסכום המבוקש.' if hebrew else
                    'Relevant source context cannot be quoted completely and safely, so the requested amount cannot be verified.')
        if evidence.reason=='monetary_item_excerpt_unsafe':
            return ('שדה במקור כולל מידע שלא ניתן להציגו בבטחה; לכן לא ניתן לאמת את הסכום המבוקש.' if hebrew else
                    'A source field could not be quoted safely, so the requested amount cannot be verified.')
        if evidence.reason=='monetary_qualification_excerpt_unsafe':
            return ('נמצא במקור תנאי מהותי שלא ניתן להציגו בבטחה; לכן לא ניתן לאמת את הסכום המבוקש.' if hebrew else
                    'A material source qualification could not be quoted safely, so the requested amount cannot be verified.')
        if evidence.reason=='monetary_source_representation_unsupported':
            return ('נמצא מקור שעשוי להתאים, אך לא ניתן היה לאמת בבטחה את השדות הכספיים שלו. אפשר לבקש לפתוח את המסמך לפי סוגו ומספרו.' if hebrew else
                    'A possible source was found, but its monetary fields could not be verified safely. You can request the document by type and number.')
        return 'לא נמצאו ראיות כספיות מתאימות במקורות שנבדקו.' if hebrew else 'No matching monetary evidence was found in the sources examined.'
    lines=[]
    sources={source for source,_,_,_ in selected}
    if len(sources)>1:
        lines.append('נמצאו כמה מסמכים מתאימים; הסכומים מוצגים בנפרד:' if hebrew else 'Several matching documents were found; their prices are shown separately:')
    context_shown=set()
    for source,record,index,item in selected:
        markers=' '.join('['+cid+']' for cid in cited[source])
        quote=record.document_kind in {'quote','quotation','estimate'}
        kind=('מחיר מוצע' if quote else 'סכום במסמך') if hebrew else ('Quoted price' if quote else 'Documented price')
        identity=str(record.document_number)
        has_identity_date=any(p.source_id==source and p.metadata.get('page')==record.page and
                              p.value=='monetary_role:identity_date' for p in evidence.passages)
        if record.issue_date and (not role_mode or has_identity_date):identity+='; '+str(record.issue_date)
        parts=[]
        part_roles=[]
        def push(text,*roles):
            parts.append(text)
            part_roles.append(roles)
        def annotate(text,roles):
            if not role_mode:
                return text
            wanted=[]
            for passage in evidence.passages:
                if passage.source_id!=source or passage.metadata.get('page')!=record.page:
                    continue
                value=str(passage.value)[len('monetary_role:'):]
                if any(value==role or (role.endswith('*') and value.startswith(role[:-1])) for role in roles):
                    wanted.append('['+passage.citation_id+']')
            return text+(' '+' '.join(wanted) if wanted else '')
        recorded=_recorded_payments(record,source,evidence.tenant_id)
        if request.payment_method:
            recorded=tuple(payment for payment in recorded
                           if canonical_payment_method(payment.method)==request.payment_method)
        payments_proven=bool(recorded and len(record.items)==1)
        payment_requested=(request.operation=='payment_status' or
            bool(re.search(r'\bpayment\b|(?<!\w)ה?תשלום(?!\w)',query,re.I)))
        if request.operation=='payment_status' or (payment_requested and payments_proven):
            # Existing payment evidence parser requires a recorded payment row,
            # method/date and amount; amount due never becomes a payment.
            if payments_proven:
                payments='; '.join(_amount(p.amount,p.currency)+' ('+
                    (p.method+'; ' if p.method else '')+p.date.isoformat()+')' for p in recorded)
                push(('תשלומים מתועדים: ' if hebrew else 'Recorded payments: ')+payments+'.','payment:*')
            elif request.payment_method:
                push('לא נמצא תשלום מאומת באמצעי התשלום המבוקש במקור זה.' if hebrew else 'There is no verified payment using the requested method in this source.')
            else:
                push('לא נמצאה הוכחה לתשלום שבוצע בפועל במקור זה.' if hebrew else 'This source does not establish an actual payment.')
        single=len(record.items)==1
        price=item.line_total
        role=request.role_requested
        if role=='unit_price':
            push(('מחיר יחידה: ' if hebrew else 'Unit price: ')+_amount(item.unit_price,item.currency)+(' לפני מע״מ.' if hebrew else ' before VAT.'),f'item:{index}:unit_price')
        elif role=='payable_total' and single and record.payable:
            push(kind+(' כולל מע״מ: ' if hebrew else ' including VAT: ')+_amount(record.payable.amount,record.payable.currency)+'.','payable')
        elif role=='tax_amount' and single and record.tax:
            push(('סכום המע״מ: ' if hebrew else 'VAT amount: ')+_amount(record.tax.amount,record.tax.currency)+'.','tax','vat_label')
        else:
            push(kind+': '+_amount(price,item.currency)+(' לפני מע״מ.' if hebrew else ' before VAT.'),f'item:{index}:line_total')
        if not quote:
            from financial_documents import document_kind_label
            parts[0]=document_kind_label(record.document_kind,hebrew)+': '+parts[0]
        if role in {'payable_total','tax_amount'} and not single:
            push('המקור אינו מציין בנפרד את הסכום המבוקש כולל מע״מ או את מס הפריט הזה.' if hebrew else 'The source does not separately state the requested VAT or VAT-inclusive amount for this item.')
        if single:
            if role in {'payable_total','tax_amount'}:
                push(('סכום הפריט לפני מע״מ: ' if hebrew else 'Item amount before VAT: ')+_amount(item.line_total,item.currency)+'.',f'item:{index}:line_total')
            if record.payable and role!='payable_total':
                push(('כולל מע״מ: ' if hebrew else 'Including VAT: ')+_amount(record.payable.amount,record.payable.currency)+'.','payable')
            if record.tax and role!='tax_amount':
                rate=(' ('+str(record.vat_rate)+'%)') if record.vat_rate is not None else ''
                push(('מע״מ' if hebrew else 'VAT')+rate+': '+_amount(record.tax.amount,record.tax.currency)+'.','tax','vat_label')
        elif record.payable:
            push(('סך המסמך כולו, עבור כל הפריטים: ' if hebrew else 'Whole-document total for all items: ')+_amount(record.payable.amount,record.payable.currency)+'.','payable')
        push(('העבודה המתומחרת יחד: ' if hebrew else 'The work priced together: ')+item.description+'.',f'item:{index}:description',f'item:{index}:context:*')
        missing=tuple(term for term in request.terms if not matches_terms((term,),item.description))
        unit_heading=any(_bounded_unit_service_heading(label,missing,item,record)
                         for label in _context_labels(record,request,item))
        if item.quantity is not None and ('quantity' in requested_fields or role=='unit_price' or unit_heading):
            push(('כמות שעות: ' if unit_heading and hebrew else 'Hours: ' if unit_heading else
                  'כמות: ' if hebrew else 'Quantity: ')+str(item.quantity)+'.',f'item:{index}:quantity')
        if unit_heading and item.unit_price is not None and role!='unit_price':
            push(('מחיר לשעה: ' if hebrew else 'Price per hour: ')+_amount(item.unit_price,item.currency)+'.',
                 f'item:{index}:unit_price')
        conditions=material_monetary_conditions(record,item)
        if conditions:
            price_bound=all(re.search(r'(?i)\b(?:price|cost|fee|amount|quote|quotation)\b|מחיר|עלות|הצעה',condition)
                            for condition in conditions)
            if role_mode and not (request.operation=='description_price' and price_bound):
                lead=('מידע במקור שהקשרו לתשלום ולעבודה אינו ברור: ' if hebrew else
                      'Source text whose relation to the payment and work is unclear: ')
            else:
                lead='תנאים למחיר: ' if hebrew else 'Price conditions: '
            push(lead+'; '.join(conditions),f'item:{index}:condition:*')
        labels={
            'issue_date': ('תאריך המסמך', 'Document date'),
            'supplier_phone': ('טלפון הספק', 'Supplier phone'),
            'supplier_email': ('דוא״ל הספק', 'Supplier email'),
            'supplier_website': ('אתר הספק', 'Supplier website'),
        }
        for field,value in requested_source_details(record,requested_fields):
            label=labels[field][0 if hebrew else 1]
            push(label+': '+(value if value is not None else
                ('לא זוהה בוודאות במקור.' if hebrew else 'not established safely by this source.')),'detail:'+field)
        page_key=(source,record.page)
        if role_mode and page_key not in context_shown:
            context_shown.add(page_key)
            page_items=[(i,it) for sid,rec,i,it in selected if sid==source and rec.page==record.page]
            payload=next((page for sid,page in evidence.geometry_pages
                          if sid==source and page['page']==record.page),None)
            context=_monetary_context_inventory(record,page_items,request,payload)
            if context is not None:
                context_roles,uncertain=context
                reconstructed_seen=set()
                for context_role,_a,_b,context_text in context_roles:
                    if context_role.startswith('source_elsewhere_context:'):
                        # Keep the source-local peer qualification available
                        # in the citation, without presenting it as applying
                        # to the requested item.
                        for part_index,roles in enumerate(part_roles):
                            if f'item:{index}:description' in roles:
                                part_roles[part_index]=roles+(context_role,)
                                break
                        continue
                    if context_role.startswith('source_context_limited:'):
                        push(('במקור יש תנאי שלא ניתן לשחזר בבטחה את סדר מילותיו.' if hebrew else
                              'A source qualification cannot be reconstructed safely in its original order.'),
                             context_role)
                        continue
                    if context_role.startswith('source_reconstructed_context:'):
                        if context_role in reconstructed_seen:
                            continue
                        reconstructed_seen.add(context_role)
                        from subject_native_work import _context_spans, _display_context_span
                        from monetary_native_pdf import native_rtl_numeric_context
                        ordinal=int(context_role.rsplit(':',1)[1])
                        block=_context_spans(record)[ordinal][2]
                        result=native_rtl_numeric_context(payload,_display_context_span(block,record))
                        if result is None:
                            continue
                        lead=('שחזור לפי מיקום הטקסט במקור: ' if hebrew else
                              'Reconstructed from the source layout: ')
                        push(lead+result[0],context_role)
                        continue
                    push(('מידע במקור שהקשרו לתשלום ולעבודה אינו ברור: ' if hebrew else
                          'Source text whose relation to the payment and work is unclear: ')+context_text,
                         context_role)
        if quote and not payment_requested:
            if payments_proven:
                payments='; '.join(_amount(p.amount,p.currency)+' ('+p.date.isoformat()+')' for p in recorded)
                push(('בנפרד מהמחיר המוצע, תשלומים מתועדים: ' if hebrew else 'Separate from the quoted price, recorded payments: ')+payments+'.','payment:*')
        # A source identifier/date distinguishes competing quotations; a
        # single cited amount does not need unrequested document metadata.
        if len(sources)>1 or (role_mode and has_identity_date):
            push(('מסמך ' if hebrew else 'Document ')+identity+'.','identity','identity_date')
        if role_mode:
            if parts:
                identity_index=1 if quote and (request.operation=='payment_status' or
                    (payment_requested and payments_proven)) and len(parts)>1 else 0
                part_roles[identity_index]=part_roles[identity_index]+('identity',)
            parts=[annotate(text,roles) for text,roles in zip(parts,part_roles)]
        else:
            parts.append(markers)
        lines.append(' '.join(parts))
    # Tenant-wide extraction coverage is diagnostic for a source-specific
    # amount. A failed matching source/page can change the interpretation and
    # still warrants a specific qualification.
    grouped={}
    for row in evidence.witness.rows:
        grouped.setdefault(row['metadata']['source_id'],[]).append(row)
    unmatched=set(candidate_ids(grouped,request))-sources
    verified_pages={(source,record.page) for source,record in parsed_pages(evidence)}
    expected_pages={(row['metadata']['source_id'],page) for row in evidence.native_rows
        for page in row['metadata'].get('_native_pdf_requested_pages',())}
    if unmatched:
        lines.append('מחירים במקורות מתאימים נוספים לא אומתו; הסכומים המוצגים הם מהמקורות המצוטטים.' if hebrew else 'Prices in other matching sources could not be verified; the amounts shown are from the cited sources.')
    elif expected_pages-verified_pages:
        lines.append('תנאים בעמודים נוספים של המקור לא אומתו; הסכום המוצג נתמך בעמוד המצוטט.' if hebrew else 'Conditions on other source pages could not be verified; the amount shown is supported by the cited page.')
    if role_mode and {(source,record.page) for source,record,_,_ in all_selected}-represented:
        lines.append('שדות או תנאים במקורות תואמים נוספים לא אומתו.' if hebrew else
                     'Other matching source fields or qualifications could not be verified.')
    return '\n'.join(lines)
