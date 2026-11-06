"""Independent acceptance checks for subject-work coverage and ownership.

This verifier reads contract slots and witnessed source rows directly; it does
not call the subject selector or use its coverage verdict.
"""
import re
import unicodedata
from pathlib import PurePosixPath


_LOCATION_HEADS = frozenset(('פרויקט', 'לקוח', 'לקוחה', 'אתר', 'מוקד', 'מרכז',
                             'מועצה', 'טיפול', 'תקלה', 'תיקון', 'התקנה',
                             'אספקה', 'החלפה', 'שירות', 'עבודה', 'טכנאי', 'טכנאית'))
_COMMON = _LOCATION_HEADS | frozenset(('בקר', 'מסך', 'מקרן', 'רמקול', 'מצלמה',
                                      'מיקרופון', 'מולטימדיה', 'מערכת', 'אזורית', 'תל'))
_STOP = frozenset(('the', 'a', 'an', 'at', 'in', 'of', 'for', 'and', 'or',
                   'של', 'על', 'את', 'או'))


def _tokens(text):
    return tuple(t.casefold() for t in re.findall(
        r'[A-Za-z0-9\u05d0-\u05ea]+', unicodedata.normalize('NFKC', str(text or '')))
        if t.casefold() not in _STOP)


def _forms(word):
    variants = {word}
    for prefix in ('ו', 'ב', 'ה', 'כ', 'ל', 'מ', 'ש', 'וה', 'בה', 'לה', 'מה', 'של'):
        if word.startswith(prefix) and word[len(prefix):] in _COMMON:
            variants.add(word[len(prefix):])
    # Independently recognize Hebrew construct forms of known work nouns.
    # Proper names are not candidates for this bounded inflection.
    work_nouns = {'החלפה','התקנה','אספקה','תקלה','עבודה'}
    for candidate in tuple(variants):
        for prefix in ('', 'ו', 'ה', 'ב', 'ל', 'מ', 'ש'):
            stem=candidate[len(prefix):] if candidate.startswith(prefix) else ''
            if stem.endswith('ת') and stem[:-1]+'ה' in work_nouns:
                variants.update((stem,stem[:-1]+'ה'))
    return variants


def _word_forms(words):
    result = []
    for index, word in enumerate(words):
        forms = _forms(word)
        # A locative prefix is licensed only after a general work/site head.
        # Unknown names never lose an initial letter in other contexts.
        if (index and len(word) >= 4 and word.startswith('ב')
                and _forms(words[index - 1]) & _LOCATION_HEADS):
            forms.add(word[1:])
        result.append(forms)
    return tuple(result)


def _includes(wanted, text):
    need = _word_forms(_tokens(wanted))
    actual = _word_forms(_tokens(text))
    return bool(need) and all(any(term & item for item in actual) for term in need)


def _item_head(claim):
    product = re.search(r'(?:מוצר|product)\s*\([^)]*\)\s*:\s*([^\n]+)', claim, re.I)
    value = product.group(1).strip() if product else str(claim).strip()
    return re.sub(r'^(?:התקנה\s+של|אספקה\s+של|installation\s+of|supply\s+of)\s+',
                  '', value, flags=re.I)


def _item_matches(slot, claim):
    head = _item_head(claim)
    first = re.match(r'^[\s\-*•]*([A-Za-z0-9\u05d0-\u05ea]+)', head)
    terms = _tokens(slot)
    return bool(first and terms and _includes(slot, head)
                and any(_forms(term) & _forms(first.group(1).casefold()) for term in terms))


def _labels(text, labels):
    pattern = r'(?im)^[^\S\r\n]*(?:' + labels + r')[^\S\r\n]*(?:[:\-–—][^\S\r\n]*|\r?\n[^\S\r\n]*)([^\r\n]+)'
    return re.findall(pattern, str(text or ''))


def _owner_bound(owner, slots, passage, rows):
    if not owner:
        return True
    owner_rows = [r for r in rows if r.get('id') == passage.evidence_id]
    if len(owner_rows) != 1:
        return False
    row = owner_rows[0]
    text = str(row.get('document') or '')
    all_header = '\n'.join(str(r.get('document') or '') for r in rows)[:2000]
    suppliers = _labels(all_header, r'ספק|supplier|issuer|מנפיק')
    customers = _labels(all_header, r'לקוח|customer|client')
    if any(_includes(owner, item) for item in suppliers):
        return False
    if any(not _includes(owner, item) for item in customers):
        return False
    # Literal text in the selected passage is strongest; a retrieval chunk
    # containing several spreadsheet rows cannot transfer an owner from a
    # different logical row to this item.
    if _includes(owner, passage.source_claim):
        return True
    metadata = row.get('metadata') or {}
    mime = str(metadata.get('mime_type') or '')
    if 'spreadsheet' in mime or 'excel' in mime:
        return False
    # A complete source filename naming both owner and every requested item is
    # a topic identity hint. Parent directories alone are not business owners.
    name = PurePosixPath(str(metadata.get('source_relative_path') or '')).name
    body_rows = [r for r in rows if not re.search(r'(?i)^(?:header|footer)\b',
                 str((r.get('metadata') or {}).get('section') or ''))]
    if (len(body_rows) == 1 and body_rows[0].get('id') == passage.evidence_id
            and _includes(owner, name) and all(_includes(slot, name) for slot in slots)):
        return True
    # An explicit project/customer/quotation header can scope a single
    # non-spreadsheet source chunk, but arbitrary mentions cannot.
    if len(rows) != 1:
        return False
    for line in text[:2000].splitlines():
        if re.search(r'(?i)(?:ספק|supplier|issuer|מנפיק)\s*[:\-–—]?', line):
            continue
        if (_includes(owner, line) and (_includes(line, owner) or re.search(
                r'(?i)לקוח|customer|client|פרויקט|project|אתר|site|מיקום|location|מוקד|מרכז|מועצה|הצעת\s+מחיר|quotation',
                line))):
            return True
    return False


def validate_subject_work_contract(contract, evidence):
    """Verify requested items and owner directly against witness passages."""
    from subject_intent import subject_work_slots
    requirement = contract.requirements[0]
    # Reopen original source for PDF literals instead of accepting a producer
    # assertion that flattened text preserved item/amount ownership.
    if evidence and any((getattr(p, 'metadata', {}) or {}).get('mime_type') == 'application/pdf'
                        and p.value != 'subject_document_identity' for p in evidence.passages):
        from chat_api import DOCS_BASE_DIR
        from subject_work_unruled import verify_unruled_passages
        if not verify_unruled_passages(evidence, evidence.tenant_id, DOCS_BASE_DIR):
            return False
    if getattr(evidence,'partial_projection',False):
        # Reopen each emitted original independently at acceptance. An indexed
        # excerpt by itself is not source proof, even for the partial route.
        from chat_api import DOCS_BASE_DIR
        from subject_reading import _partial_original_body, PartialContextWitness
        import hashlib
        originals={}
        independently_witnessed={}
        for passage in evidence.passages:
            if passage.source_id not in originals:
                originals[passage.source_id]=_partial_original_body(
                    DOCS_BASE_DIR,evidence.tenant_id,passage.metadata)
            body=originals[passage.source_id]
            if not body or passage.text not in body or passage.source_claim not in body:
                return False
        from subject_native_work import _nominated_qualification
        from subject_reading import _instruction_like, _irrelevant_contact
        for source,body in originals.items():
            work=[p.source_claim for p in evidence.passages if p.source_id==source
                  and p.value != 'subject_partial_context']
            context=[(p.source_claim,p.metadata.get('_partial_context_scope'))
                     for p in evidence.passages if p.source_id==source
                     and p.value == 'subject_partial_context']
            # Independently validate both model inventories against the full
            # reopened units and the selected source work claims. Neither an
            # unsupported scope nor a model-only administrative label can
            # suppress a source block.
            from types import SimpleNamespace
            receipt_entries=getattr(evidence,'partial_context_semantics',())
            if (not isinstance(receipt_entries,tuple)
                    or any(not isinstance(entry,tuple) or len(entry)!=3
                           or not isinstance(entry[0],str) for entry in receipt_entries)
                    or len({entry[0] for entry in receipt_entries})!=len(receipt_entries)
                    or {entry[0] for entry in receipt_entries}!=set(originals)):
                return False
            source_receipt=next(entry for entry in receipt_entries if entry[0]==source)
            semantic_wrapper=SimpleNamespace(context_semantics=((source,None,source_receipt[1],source_receipt[2]),))
            if any(re.match(r'(?i)^\s*(?:only\s+for|רק\s+עבור)\s+[^:\n]+[:：]',item)
                   or re.fullmatch(r'(?i)\s*[^\n:]{1,120}\s+(?:is|are)\s+'
                       r'(?:distinct|different|separate)\s+from\s+[^\n:]{1,120}[.]?\s*',item)
                   for item in work):
                return False
            rows=[r for r in evidence.witness.rows if r.get('metadata',{}).get('source_id')==source]
            known=[]
            unknown=False
            blocked=False
            # Reconstruct issuer field boundaries from original text at
            # acceptance; no producer-supplied list controls what is omitted.
            from financial_documents import issuer_span
            field_pattern=(r'(?i)\s*(?:issuer|supplier|issued\s+by|מנפיק|ספק)'
                           r'\s*[:\-]\s*([^\W\d_]+)\s*')
            units=[]
            for paragraph in (part.strip() for part in re.split(r'\n\s*\n',body) if part.strip()):
                lines=paragraph.splitlines(keepends=True)
                fields=[re.fullmatch(field_pattern,line.strip(),re.UNICODE) for line in lines]
                if not any(re.match(r'(?i)^\s*(?:issuer|supplier|publisher|vendor|author|producer|issued\s+by|מנפיק|ספק|מחבר)\s*[:\-]',line)
                           for line in lines):
                    units.append(paragraph)
                    continue
                pending=[]
                for line,field in zip(lines,fields):
                    boundary=re.match(r'(?i)^\s*(?:issuer|supplier|publisher|vendor|author|producer|issued\s+by|מנפיק|ספק|מחבר)\s*[:\-]',line)
                    if boundary:
                        if ''.join(pending).strip():
                            units.append(''.join(pending).strip())
                        units.append(line.strip())
                        pending=[]
                    else:
                        pending.append(line)
                if ''.join(pending).strip():
                    units.append(''.join(pending).strip())
            def publisher_value_support(value):
                wanted=' '.join(str(value).split()).casefold()
                if not wanted or '\n' in str(value):
                    return None
                matches=[match.span() for match in re.finditer(
                    r'(?im)^(?:issuer|supplier|publisher|vendor|author|producer|issued\s+by|מנפיק|ספק|מחבר)\s*[:\-]\s*([^\n;:]+)$',
                    body) if ' '.join(match.group(1).split()).casefold()==wanted]
                return matches[0] if len(matches)==1 else None

            def publisher_label_support(value):
                field=re.fullmatch(
                    r'(?i)(?:issuer|supplier|publisher|vendor|author|producer|issued\s+by|מנפיק|ספק|מחבר)\s*[:\-]\s*([^\n;:]{1,120})',
                    str(value).strip())
                if not field:
                    return None
                matches=[match.span() for match in re.finditer(
                    r'(?im)^[ \t]*'+re.escape(field.group(1).strip())+r'[ \t]*$',body)]
                return matches[0] if len(matches)==1 else None

            def explicit_scope(value, block_start):
                match=re.fullmatch(
                    r'(?i)\s*(?:only\s+for|רק\s+עבור)\s+(?P<target>[^:\n]{1,120})'
                    r'\s*[:：]\s*(?P<tail>[^\n]+)\s*',str(value))
                address_scope=False
                if not match:
                    match=re.fullmatch(
                        r'(?i)\s*(?:at|in)\s+(?P<target>[^,\n:]{1,100})'
                        r'\s*,\s*(?P<tail>[^\n]+)\s*',str(value))
                    address_scope=bool(match)
                if not match:
                    return None
                grammar={'the','a','an','at','in','of','for','and','or','של','על','את','או'}
                def terms(text):
                    return tuple(dict.fromkeys(w for w in re.findall(
                        r'[A-Za-z0-9\u05d0-\u05ea]+',str(text).casefold()) if w not in grammar))
                target=match.group('target').strip()
                own=terms(requirement.subject)
                other=terms(target)
                if not own or not other:
                    return None
                tail=match.group('tail')
                if re.search(r'\b(?:except|excluding|other|others|all|both|each|every|also|'
                             r'including|elsewhere|otherwise|not|unless)\b|למעט|פרט\s+ל|'
                             r'חוץ\s+מ|אחר|אחרים|כולם|כולן|כל\s+ה|גם|וכן',tail,re.I):
                    return None
                if ' '.join(str(requirement.subject).split()).casefold()==' '.join(target.split()).casefold():
                    start=block_start+match.start('target')+match.group('target').find(target)
                    return ('subject',start,start+len(target))
                if any(re.search(r'(?<!\w)'+re.escape(term)+r'(?!\w)',tail,re.I)
                       for term in own if term not in other):
                    return None
                if address_scope:
                    def location(name):
                        pattern=(r'(?im)\b(?:at|in)\s+'+re.escape(name)+
                                 r'\s*,\s*(?P<address>\d+[^\n,:]{2,100})\s*:')
                        return list(re.finditer(pattern,body))
                    selected_locations=location(str(requirement.subject).strip())
                    peer_locations=location(target)
                    if len(selected_locations)!=1 or len(peer_locations)!=1:
                        return None
                    selected_loc,peer_loc=selected_locations[0],peer_locations[0]
                    left=' '.join(selected_loc.group('address').casefold().split())
                    right=' '.join(peer_loc.group('address').casefold().split())
                    if left==right or re.search(
                        r'(?i)\b(?:same\s+(?:site|location|place|item)|'
                        r'another\s+name|also\s+called|alias|same\s+as)\b|'
                        r'אותו\s+(?:מקום|אתר)|שם\s+נוסף',body):
                        return None
                    support=(min(selected_loc.start(),peer_loc.start()),
                             max(selected_loc.end(),peer_loc.end()))
                else:
                    pattern=(r'(?im)^[ \t]*'+re.escape(target)+r'\s+(?:is|are)\s+'
                             r'(?:distinct|different|separate)\s+from\s+'
                             +re.escape(str(requirement.subject).strip())+r'[.]?[ \t]*$')
                    relations=list(re.finditer(pattern,body))
                    if len(relations)!=1:
                        return None
                    support=relations[0].span()
                support_text=body[support[0]:support[1]]
                for unit in units:
                    if (unit.strip() in {str(value).strip(),support_text.strip()} or
                            (address_scope and unit.strip() and unit.strip() in support_text)):
                        continue
                    if (re.search(r'(?<!\w)'+re.escape(target)+r'(?!\w)',unit,re.I)
                            and re.search(r'(?<!\w)'+re.escape(str(requirement.subject).strip())
                                          +r'(?!\w)',unit,re.I)):
                        return None
                return ('elsewhere',*support)

            # Independently build the exact source-unit candidate inventory:
            # the selected work claims precede every remaining source block.
            # Candidate membership alone does not prove a separate item.
            inventory=[(claim,True,None) for claim in work]
            inventory.extend((block,False,i) for i,block in enumerate(units)
                             if not any(block==item or block in item for item in work))
            decisions=_checked_semantic_page(semantic_wrapper,source,None,units,len(inventory),set(range(len(work))))
            if decisions is False or decisions is None or (source_receipt[2] and not source_receipt[1]):
                return False
            selected_blocks={'s'+str(i) for i,unit in enumerate(units)
                             if any(unit==item or unit in item for item in work)}
            distinct_peer_work={}
            distinct_peer_notes=set()
            for ordinal,unit in enumerate(units):
                decision=decisions.get(ordinal)
                if (not decision or decision[:2]!=('condition','elsewhere')
                        or body.count(unit)!=1):
                    continue
                candidate_scope=explicit_scope(unit,body.find(unit))
                if not candidate_scope or candidate_scope[0]!='elsewhere':
                    continue
                refs=set(decision[2])
                if not refs.intersection(selected_blocks):
                    continue
                for item_index,(_value,is_selected,block_index) in enumerate(inventory):
                    if (is_selected or block_index is None or 'i'+str(item_index) not in refs
                            or 's'+str(block_index) not in refs):
                        continue
                    peer=decisions.get(block_index)
                    if peer and peer[:2]==('work','elsewhere'):
                        distinct_peer_work[block_index]=candidate_scope
                        distinct_peer_notes.add(ordinal)
            cursor=0
            for ordinal,block in enumerate(units):
                start=body.find(block,cursor)
                if start < 0:
                    return False
                end=start+len(block)
                cursor=end
                normalized=' '.join(block.split())
                disposition,support='uncertain','unclassified_source_text'
                scope='unresolved'
                scope_info=explicit_scope(block,start)
                support_start,support_end=start,end
                field=re.fullmatch(field_pattern,block,re.UNICODE)
                heading=re.fullmatch(
                    r'(?i)(?:(?:publisher|issuer|supplier|vendor|author|producer)\s+'
                    r'(?:information|details|profile)|(?:reference|document|source)\s+'
                    r'(?:appendix|index|register)|(?:פרטי|מידע\s+על)\s+'
                    r'(?:ה?מנפיק|ה?ספק|ה?מחבר)|נספח\s+(?:הפניות|מקורות))\s*[:：]?',
                    normalized.rstrip('.'))
                contact=re.fullmatch(
                    r'(?i)(?:(?:e-?mail|phone|telephone|טלפון)\s*[:\-]\s*)?(?:'
                    r'[^@\s]+@[^@\s]+\.[A-Za-z]{2,}|(?:https?://|www\.)[^\s]+|'
                    r'[+0-9][0-9 ()\-]{6,})',block.strip())
                reference=re.fullmatch(
                    r'(?i)(?:reference|ref|אסמכתא)\s*[:#\-]\s*'
                    r'[A-Za-z0-9][A-Za-z0-9./\-]*[.]?',normalized)
                if any(block==item or block in item for item in work):
                    disposition,support='work','selected_work_span'
                elif ordinal in distinct_peer_work:
                    disposition,support='other','explicit_distinct_peer_work'
                    scope='elsewhere'
                    _,support_start,support_end=distinct_peer_work[ordinal]
                elif _instruction_like(block):
                    unknown=True
                elif field and issuer_span(body,field.group(1)) is not None:
                    disposition,support='other','source_issuer_field'
                elif heading:
                    disposition,support='other','administrative_heading'
                elif contact:
                    disposition,support='other','contact_field'
                elif publisher_label_support(block) is not None:
                    disposition,support='other','source_publisher_field'
                    support_start,support_end=publisher_label_support(block)
                elif publisher_value_support(block) is not None:
                    disposition,support='other','source_publisher_value'
                    support_start,support_end=publisher_value_support(block)
                elif reference:
                    disposition,support='other','reference_identifier'
                else:
                    unsafe=(_instruction_like(block) or _irrelevant_contact(block)
                            or bool(re.search(r'(?im)^\s*(?:issuer|supplier|publisher|vendor|author|producer|'
                                              r'מנפיק|ספק|מחבר)\s*[:\-]',block)))
                    recognized=_nominated_qualification(block)
                    decision=decisions.get(ordinal)
                    semantic_role=decision[0] if decision else None
                    # Model-only work labels do not establish that a source
                    # block belongs in the requested work description.
                    semantic_work=False
                    peer_qualification=bool(re.fullmatch(
                        r'(?i)\s*(?:only\s+for|רק\s+עבור)\s+[^:\n]{1,120}'
                        r'\s*[:：]\s*[^\n]+\s*',block))
                    indexed=any(block in row.get('document','') for row in rows)
                    if not unsafe and (recognized or peer_qualification or
                                       semantic_role=='condition' or semantic_work) and indexed:
                        disposition='condition' if recognized or semantic_role=='condition' else 'possible'
                        support=('source_qualification' if recognized else
                                 'unresolved_peer_qualification' if peer_qualification else
                                 'semantic_condition' if semantic_role=='condition' else 'semantic_work')
                        if scope_info and scope_info[0]=='subject':
                            scope='subject'
                            support='explicit_selected_subject_scope'
                            _,support_start,support_end=scope_info
                        elif (ordinal in distinct_peer_notes and scope_info
                              and scope_info[0]=='elsewhere'):
                            scope='elsewhere'
                            disposition='other'
                            support='explicit_distinct_subject_scope'
                            _,support_start,support_end=scope_info
                        if scope!='elsewhere':
                            known.append((block,disposition,scope))
                        if scope=='unresolved':
                            unknown=True
                    else:
                        unknown=True
                        if recognized and unsafe:
                            blocked=True
                witness=PartialContextWitness(source,ordinal,start,end,
                    hashlib.sha256(body[start:end].encode()).hexdigest(),disposition,support,
                    support_start,support_end,
                    hashlib.sha256(body[support_start:support_end].encode()).hexdigest(),scope)
                key=(source,ordinal)
                if key in independently_witnessed:
                    return False
                independently_witnessed[key]=witness
            def exceeds(candidates):
                return (len(candidates)>4 or sum(len(block) for block,_,_ in candidates)>1200
                        or any(len(block)>780 or len(block.split())>78 for block,_,_ in candidates))
            overflow=blocked or exceeds(known)
            if (source in evidence.partial_overflow_sources) != overflow:
                return False
            if (source in evidence.partial_uncertain_sources) != unknown:
                return False
            if overflow:
                ordered=[entry for entry in known if entry[1]=='condition']+[
                    entry for entry in known if entry[1]=='possible']
                expected=[]
                for entry in ordered:
                    if not exceeds(expected+[entry]):
                        expected.append(entry)
            else:
                expected=known
            if context != [(block,scope) for block,_,scope in expected]:
                return False
        reported={(w.source_id,w.ordinal):w for w in evidence.partial_context_witnesses}
        if len(reported) != len(evidence.partial_context_witnesses) or reported != independently_witnessed:
            return False
    slots = subject_work_slots(contract.normalized_request, requirement.subject)
    if slots is None or not evidence or not evidence.witness:
        return False
    rows_by_source = {}
    for row in evidence.witness.rows:
        metadata = row.get('metadata') or {}
        if metadata.get('tenant_id') != evidence.tenant_id:
            return False
        rows_by_source.setdefault(metadata.get('source_id'), []).append(row)
    passages_by_source = {}
    conditions = []
    for passage in evidence.passages:
        if passage.value == 'subject_partial_context' and getattr(evidence,'partial_projection',False):
            continue
        if passage.value == 'subject_material_condition':
            conditions.append(passage)
            continue
        if passage.value == 'subject_document_identity':
            continue
        passages_by_source.setdefault(passage.source_id, []).append(passage)
    for condition in conditions:
        text = condition.source_claim.strip()
        if not re.fullmatch(
            r'(?:[*•-]\s*)?(?:(?:נדרש|נדרשת|חובה|יש\s+צורך)\s+(?:לכל|עבור\s+כל)\s+(?:מוצר|פריט|יחידה)|'
            r'(?:required|must)\s+(?:for\s+)?(?:each|every)\s+(?:item|product|unit))[^\n]{0,140}',
            text, re.I):
            return False
        if not any(p.source_id == condition.source_id
                   and p.evidence_id == condition.evidence_id
                   for p in passages_by_source.get(condition.source_id, ())):
            return False
    if not passages_by_source:
        return False
    for source, passages in passages_by_source.items():
        rows = rows_by_source.get(source, ())
        if not rows:
            return False
        if slots.content:
            used = set()
            for slot in slots.content:
                candidates = [p for p in passages
                    if p.fact_id not in used and _item_matches(slot, p.source_claim)
                    and _owner_bound(slots.owner, slots.content, p, rows)]
                if not candidates:
                    return False
                used.add(candidates[0].fact_id)
            # Every emitted item passage must itself be a requested item,
            # rather than an unrelated passage with a coincidental word.
            if any(not any(_item_matches(slot, p.source_claim) for slot in slots.content)
                   for p in passages):
                return False
        elif slots.owner:
            if any(not _includes(slots.owner, p.source_claim) for p in passages):
                return False
    return True


def validate_partial_context_answer(contract, evidence, answer, passages=None, *,
                                    accounted_omissions=None):
    """Check actual released text and coverage against reopened source spans."""
    if not getattr(evidence,'partial_projection',False):
        return True
    if not validate_subject_work_contract(contract,evidence):
        return False
    original=evidence.passages
    released=tuple(passages) if passages is not None else original
    if len(original)!=len(released):
        return False
    hebrew='answer_language:he' in getattr(contract,'qualifiers',())
    locations=[]
    for original_passage,passage in zip(original,released):
        if (original_passage.source_id != passage.source_id
                or original_passage.source_claim != passage.source_claim
                or original_passage.value != passage.value):
            return False
        if passage.value=='subject_partial_context':
            if passage.metadata.get('_partial_context_scope')=='subject':
                lead=('במקור מצוין לגבי הנושא המצוין להלן: ' if hebrew else
                      'The source states for the subject named below: ')
            elif passage.metadata.get('_partial_context_scope')=='document':
                lead=('במסמך מצוין: ' if hebrew else 'The document states: ')
            else:
                lead=('מידע במקור שהקשרו לעבודה אינו ברור: ' if hebrew else
                      'Source text whose relation to this work is unclear: ')
        else:
            lead=''
        literal=lead+passage.source_claim.strip()+' ['+passage.citation_id+']'
        matches=list(re.finditer(re.escape(literal)+r'(?=\n|$)',answer))
        if len(matches)!=1:
            return False
        locations.append((passage.source_id,matches[0].start(),matches[0].end()))
    partial_sources=set(evidence.partial_overflow_sources) | {
        w.source_id for w in evidence.partial_context_witnesses if w.disposition=='uncertain'}
    limitation=('במקור יש טקסט נוסף שסיווגו אינו ברור; תיאור העבודה הוא קטע חלקי ואינו קובע מהם תנאי העבודה, אם אושרה או אם בוצעה.' if hebrew else
        'The source contains additional unclassified text; the work description is only a partial excerpt and does not establish the conditions, authorization, or completion of the work.')
    for source in partial_sources:
        own=[(start,end) for sid,start,end in locations if sid==source]
        if not own:
            return False
        last_end=max(end for _,end in own)
        next_source=min((start for sid,start,_ in locations if sid!=source and start>last_end),
                        default=len(answer))
        if limitation not in answer[last_end:next_source]:
            return False
    omitted=set(evidence.partial_omitted_sources)
    accounted=set(accounted_omissions or ())
    if not accounted <= omitted:
        return False
    if omitted-accounted:
        disclaimer=('פרטים ממקורות תואמים נוספים לא נכללו בתשובה.' if hebrew else
                    'Details from other matching sources are not included in this answer.')
        if disclaimer not in answer:
            return False
    return True


def _checked_semantic_page(evidence, source, page, spans, item_count, selected_indices):
    """Validate two independently produced semantic inventories against source IDs.

    Authentication and exact source reopening remain the caller's responsibility.
    Agreement is a semantic check, not deterministic proof of meaning.
    """
    entries=getattr(evidence,'context_semantics',())
    matches=[entry for entry in entries
             if isinstance(entry,tuple) and len(entry)==4
             and entry[0]==source and entry[1]==page]
    if not matches:
        return None
    if len(matches)!=1:
        return False
    _,_,proposal,checked=matches[0]
    if not isinstance(proposal,tuple) or not isinstance(checked,tuple):
        return False
    def inventory(rows):
        if not rows:
            return {}
        output={}
        for row in rows:
            if not isinstance(row,tuple) or len(row)!=4:
                return None
            ordinal,role,scope,support=row
            if (type(ordinal) is not int or ordinal<0 or ordinal>=len(spans)
                    or ordinal in output or role not in {'condition','administrative','work','uncertain'}
                    or scope not in {'item','document','elsewhere','unresolved'}
                    or not isinstance(support,tuple) or not all(type(ref) is str for ref in support)
                    or len(set(support))!=len(support)):
                return None
            for ref in support:
                if type(ref) is not str or not re.fullmatch(r'[si]\d+',ref):
                    return None
                index=int(ref[1:])
                if (ref[0]=='s' and index>=len(spans)) or (ref[0]=='i' and index>=item_count):
                    return None
            if len(support)>12 or (role!='uncertain' and 's'+str(ordinal) not in support):
                return None
            if role=='uncertain' and scope!='unresolved':
                return None
            items={int(ref[1:]) for ref in support if ref.startswith('i')}
            if role=='administrative' and (scope in {'item','elsewhere'} or items):
                return None
            if role in {'condition','work'} and scope=='item' and (not items or not items<=selected_indices):
                return None
            if role in {'condition','work'} and scope=='elsewhere' and (not items or bool(items & selected_indices)):
                return None
            output[ordinal]=(role,scope,support)
        if set(output)!=set(range(len(spans))):
            return None
        if len({v[0] for v in output.values()})==1:
            homogeneous=rows[0][1]
            if homogeneous=='administrative' or (len(spans)>1 and homogeneous in {'condition','work'}):
                return None
        return output
    left,right=inventory(proposal),inventory(checked)
    if left is None or right is None:
        return False
    return {i:left[i] if i in left and i in right and left[i]==right[i] else None
            for i in range(len(spans))}


def validate_native_context_witnesses(evidence, contract, answer, *, accounted_omissions=None):
    """Independently reopen context and verify claim-scoped source witnesses."""
    import hashlib
    from chat_api import DOCS_BASE_DIR
    from monetary_native_pdf import parse_monetary_page
    from monetary_questions import selected_items
    from source_access import pdf_source_pages, SourceAccessError
    from subject_native_work import (work_request, subject_selected_items, recipient_subject, context_units, _context_spans, _semantic_inventory_spans,
        _nominated_qualification, _ordinary_title_block, _REFUND_ONLY,
        _CONDITION_CUE, _independent_context_risk, _context_overflow)
    if not evidence or not evidence.witness:
        return False
    # Recompute requested item ownership from the reopened descriptions.
    # A peer item sharing only a generic subject word remains context.
    selected = subject_selected_items(evidence)
    records = {(source,record.page): record for source,record,_,_ in selected}
    reviews = {(source,page): dict(roles) for source,page,roles in evidence.context_reviews}
    if len(reviews) != len(evidence.context_reviews) or set(reviews) != set(records):
        return False
    witnesses = {(w.source_id,w.page,w.ordinal):w for w in evidence.context_witnesses}
    if len(witnesses) != len(evidence.context_witnesses):
        return False
    needed = set()
    limited = False
    partial_context = False
    overflow = False
    hebrew = 'answer_language:he' in contract.qualifiers
    for (source,page), _reported_record in records.items():
        rows = [row for row in evidence.native_rows if row.get('metadata',{}).get('source_id') == source
                and row.get('metadata',{}).get('page') == page]
        if len(rows) != 1:
            return False
        row = rows[0]
        try:
            pages = pdf_source_pages(DOCS_BASE_DIR, evidence.tenant_id, row['metadata'], (page,),
                                     include_geometry=True)
        except (OSError,ValueError,SourceAccessError):
            return False
        if len(pages) != 1 or pages[0]['text'] != row['document']:
            return False
        record = parse_monetary_page(pages[0])
        if (not record or record.geometry_sha256 != row['metadata'].get('_native_pdf_geometry_sha256')
                or pages[0]['raw_sha256'] != row['metadata'].get('_native_pdf_raw_text_sha256')
                or pages[0]['text_sha256'] != row['metadata'].get('_native_pdf_page_text_sha256')):
            return False
        from quotation_recipient_witness import verify_quotation_recipient
        recipient_witnesses = [w for w in evidence.recipient_witnesses
                               if (w.source_id,w.page)==(source,page)]
        recipient_only = [i for sid,rec,i,item in selected if sid==source and rec.page==page
                          and not _includes(evidence.subject,item.description)]
        if recipient_only:
            if (not recipient_subject(evidence) or len(recipient_witnesses)!=1
                    or recipient_only != [recipient_witnesses[0].item_index]
                    or not verify_quotation_recipient(pages[0],evidence.subject,source,recipient_witnesses[0])):
                return False
        spans = _context_spans(record)
        units = context_units(record)
        roles = reviews[(source,page)]
        if spans is None or units is None or set(roles) != set(range(len(spans))):
            return False
        # Verify that every occurrence of every parser-retained context unit
        # is covered, including repeated identical phrases.
        for unit in units:
            if unit and unit not in record.source_text:
                return False
            cursor = 0
            while unit:
                start = record.source_text.find(unit, cursor)
                if start < 0:
                    break
                end = start + len(unit)
                if not any(a <= start and end <= b for a,b,_ in spans):
                    return False
                cursor = start + 1
        selected_indices = {i for sid,rec,i,_ in selected if sid == source and rec.page == page}
        inventory=_semantic_inventory_spans(record,spans)
        semantic=_checked_semantic_page(evidence,source,page,inventory,len(record.items),selected_indices)
        if semantic is False:
            return False

        # Recompute every administrative exclusion from the reopened source.
        # The complete authenticated span, rather than a model label or a
        # partial heading match, is the support for each disposition.
        from financial_documents import issuer_span
        from semantic_contracts import parse_document_role_propositions
        from subject_reading import _instruction_like, _irrelevant_contact
        def publisher_value_support(value):
            wanted = " ".join(str(value).split()).casefold()
            if not wanted or "\n" in str(value):
                return None
            matches = [match.span() for match in re.finditer(
                r'(?im)^(?:issuer|supplier|publisher|vendor|author|producer|issued\s+by|מנפיק|ספק|מחבר)\s*[:\-]\s*([^\n;:]+)$',
                record.source_text)
                if " ".join(match.group(1).split()).casefold() == wanted]
            return matches[0] if len(matches) == 1 else None

        def publisher_label_support(value):
            field = re.fullmatch(
                r'(?i)(?:issuer|supplier|publisher|vendor|author|producer|issued\s+by|מנפיק|ספק|מחבר)\s*[:\-]\s*([^\n;:]{1,120})',
                str(value).strip())
            if not field:
                return None
            actor = field.group(1).strip()
            matches = [match.span() for match in re.finditer(
                r'(?im)^[ \t]*' + re.escape(actor) + r'[ \t]*$', record.source_text)]
            return matches[0] if len(matches) == 1 else None

        def publisher_proposition_support(value):
            propositions,errors=parse_document_role_propositions(str(value).strip())
            if errors or len(propositions)!=1:
                return None
            prop=propositions[0]
            actor=prop.actor.strip()
            if (prop.predicate!='published_by' or prop.polarity!='positive'
                    or prop.target.casefold() not in {'this quotation','this quote','this estimate'}
                    or not re.fullmatch(r'[^\W\d_]+',actor,re.UNICODE)
                    or issuer_span(record.source_text,actor) is None):
                return None
            fields=[match.span() for match in re.finditer(
                r'(?im)^(?:issuer|supplier|issued\s+by|מנפיק|ספק)[ \t]*[:\-][ \t]*'
                +re.escape(actor)+r'[ \t]*$',record.source_text)]
            return fields[0] if len(fields)==1 else None

        def source_role_reason(value):
            raw = str(value).strip()
            normalized = " ".join(raw.split())
            if _ordinary_title_block(raw, record):
                return "document_identity"
            identity = re.fullmatch(
                r'(?i)(?:issuer|supplier|issued\s+by|מנפיק|ספק)\s*[:\-]\s*([^\W\d_]+)',
                raw, re.UNICODE)
            publisher = bool(identity and issuer_span(record.source_text, identity.group(1)) is not None)
            if record.document_kind in {'quote','quotation','estimate'}:
                for unit in units:
                    propositions, errors = parse_document_role_propositions(unit)
                    if errors or len(propositions) != 1:
                        continue
                    prop = propositions[0]
                    actor = prop.actor.strip()
                    if (prop.predicate != 'published_by' or prop.polarity != 'positive'
                            or prop.target.casefold() not in {'this quotation','this quote','this estimate'}
                            or not re.fullmatch(r'[^\W\d_]+', actor, re.UNICODE)
                            or issuer_span(record.source_text, actor) is None):
                        continue
                    fields = [field for field in units if re.fullmatch(
                        r'(?i)(?:issuer|supplier|issued\s+by|מנפיק|ספק)\s*[:\-]\s*' +
                        re.escape(actor), str(field).strip())]
                    if len(fields) == 1 and raw in (str(unit).strip(), str(fields[0]).strip()):
                        publisher = True
            if publisher:
                return "source_publisher"
            if _REFUND_ONLY.fullmatch(normalized):
                return "refund_only"
            heading = re.fullmatch(
                r'(?i)(?:(?:publisher|issuer|supplier|vendor|author|producer)\s+'
                r'(?:information|details|profile)|(?:reference|document|source)\s+'
                r'(?:appendix|index|register)|(?:פרטי|מידע\s+על)\s+'
                r'(?:ה?מנפיק|ה?ספק|ה?מחבר)|נספח\s+(?:הפניות|מקורות))\s*[:：]?',
                normalized)
            if heading:
                return "administrative_heading"
            if re.fullmatch(r'(?i)(?:payment\s+(?:details|information)|פרטי\s+תשלומים?)\s*[:：]?',normalized):
                return "financial_field_heading"
            contact = re.fullmatch(
                r'(?i)(?:(?:e-?mail|phone|telephone|טלפון)\s*[:\-]\s*)?(?:'
                r'[^@\s]+@[^@\s]+\.[A-Za-z]{2,}|(?:https?://|www\.)[^\s]+|'
                r'[+0-9][0-9 ()\-]{6,})', raw)
            if contact:
                return "contact_field"
            if publisher_label_support(raw) is not None:
                return "source_publisher_field"
            if publisher_value_support(raw) is not None:
                return "source_publisher_value"
            if (normalized.casefold() == " ".join(evidence.subject.split()).casefold()
                    and any(normalized.casefold() in " ".join(item.description.split()).casefold()
                            for item in record.items if item.description)):
                return "repeated_subject"
            return ""

        # Re-evaluate safe residual notes directly from the reopened
        # source. A role-looking fragment without a whole-field witness stays
        # unresolved/partial; it is not silently declared administrative.
        def safe_source_note(value):
            raw = str(value).strip()
            if not raw or _instruction_like(raw):
                return False
            if _irrelevant_contact(raw):
                contact_field = re.compile(
                    r'(?i)^(?:(?:e-?mail|phone|telephone|טלפון)\s*[:\-]\s*)?(?:'
                    r'[^@\s]+@[^@\s]+\.[A-Za-z]{2,}|(?:https?://|www\.)[^\s]+|'
                    r'[+0-9][0-9 ()\-]{6,})$')
                lines = [line.strip() for line in raw.splitlines() if line.strip()]
                remainder = [line for line in lines if not contact_field.fullmatch(line)]
                if (len(lines) <= 1 or len(remainder) != 1
                        or _irrelevant_contact(remainder[0]) or _instruction_like(remainder[0])):
                    return False
                raw = remainder[0]
            elif re.search(r'(?i)(?:issuer|supplier|publisher|vendor|author|producer|'
                           r'issued\s+by|מנפיק|ספק|מחבר)\s*[:\-]',raw):
                return False
            if re.match(r'(?i)^\s*(?:publisher|issuer|supplier|vendor|author|producer|reference)\s+'
                        r'(?:details?|information|profile|appendix|index|register)\b', raw):
                return False
            if not re.fullmatch(r"[^\W\d_]+(?:\s+[^\W\d_]+){1,3}[.!]?", raw, re.UNICODE):
                return False
            words=re.findall(r'[^\W\d_]+',raw,re.UNICODE)
            if all(re.fullmatch(r'[\u05d0-\u05ea]+',word) for word in words):
                return len(words)==2 or (len(words)<=4 and 'של' in words[1:-1])
            if not all(re.fullmatch(r'[A-Za-z]+',word) for word in words):
                return False
            return not (len(words[0])>1 and words[0].isupper()) and all(
                word.islower() for word in words[1:])

        page_passages = [p for p in evidence.passages
                         if p.source_id == source and p.metadata.get("page") == page]
        titles = [p.citation_id for p in page_passages if p.value == "subject_work_native_title"]
        recipients = [p for p in page_passages if p.value == "subject_work_native_recipient"]
        if recipient_only:
            witness = recipient_witnesses[0]
            if (len(recipients)!=1 or recipients[0].representation!='native_pdf'
                    or recipients[0].evidence_id!=row['id']
                    or recipients[0].text!=pages[0]['text']
                    or recipients[0].source_claim!=witness.recipient_block):
                return False
        elif recipients:
            return False
        descriptions = [p for p in page_passages if p.value == "subject_work_native_description"]
        from document_reading import source_body
        from monetary_native_pdf import native_cell_literal_spans
        from pdf_reconstruction import positioned_tokens_from_pymupdf, reconstruct_page
        from canonical_text import canonicalize_source_text
        tokens = positioned_tokens_from_pymupdf(page, pages[0]["words"])
        cell_spans = {}
        # The source-first interpretation includes every native item, not
        # just selected items. Authenticate each description cell so a peer
        # reference cannot be supplied by the producer without source proof.
        for index in range(len(record.items)):
            item = record.items[index]
            native_spans = native_cell_literal_spans(pages[0], item.description_word_ids)
            if not native_spans:
                return False
            cell_spans[index] = native_spans
            word_ids = set(item.description_word_ids)
            words = tuple(token for token in tokens
                          if (page, token.block, token.line, token.word) in word_ids)
            if len(words) != len(word_ids):
                return False
            raw_cell = reconstruct_page(words, preserve_ltr_runs=True).text
            logical_cell = canonicalize_source_text(raw_cell, repair_visual_bidi=True,
                normalize_pdf_spacing=True, join_pdf_line_wraps=False).text.strip()
            if " ".join(logical_cell.split()) != " ".join(item.description.split()):
                return False
        for passage in descriptions:
            index = passage.metadata.get("_native_work_item_index")
            if type(index) is not int or index not in selected_indices or not 0 <= index < len(record.items):
                return False
            if (not isinstance(passage.span, tuple) or len(passage.span) != 2
                    or any(type(n) is not int for n in passage.span)):
                return False
            start, end = passage.span
            if (start < 0 or end <= start or end > len(passage.text)
                    or passage.text[start:end] != passage.source_claim
                    or passage.span_hash != hashlib.sha256(passage.source_claim.encode()).hexdigest()):
                return False
            if passage.evidence_id == row["id"]:
                if (passage.representation != "native_pdf" or passage.text != pages[0]["text"]
                        or not any(a <= start and end <= b for a,b in cell_spans[index])):
                    return False
            else:
                indexed_rows = [r for r in evidence.witness.rows if r.get("id") == passage.evidence_id
                                and r.get("metadata",{}).get("source_id") == source
                                and r.get("metadata",{}).get("page") == page
                                and r.get("metadata",{}).get("tenant_id") == evidence.tenant_id]
                if (len(indexed_rows) != 1 or passage.representation != "indexed"
                        or source_body(indexed_rows) != passage.text
                        or passage.text.count(passage.source_claim) != 1
                        or pages[0]["text"].count(passage.source_claim) > 1):
                    return False
            # Reopened native item ownership, rather than a producer label or
            # a shared source identity, must support every description excerpt.
            normalized_claim = " ".join(passage.source_claim.split())
            normalized_item = " ".join(record.items[index].description.split())
            if not normalized_claim or normalized_claim not in normalized_item:
                return False
            # Identical text in another native item cannot independently
            # establish which item supplied the indexed occurrence.
            if any(other != index and normalized_claim in " ".join(item.description.split())
                   for other,item in enumerate(record.items)):
                return False
        item_line_positions = []
        answer_lines = answer.splitlines()
        for index in selected_indices:
            own = [p.citation_id for p in descriptions
                   if p.metadata.get("_native_work_item_index") == index]
            if not own:
                return False
            own_claims = [p.source_claim for p in descriptions
                          if p.metadata.get("_native_work_item_index") == index]
            if " ".join(" ".join(own_claims).split()) != " ".join(record.items[index].description.split()):
                return False
            description = record.items[index].description
            if index in recipient_only:
                recipient = recipient_witnesses[0].recipient
                if hebrew and re.match(r'^ה(?:מכללה|אוניברסיטה|חברה|מועצה|עירייה|מרכז|בנק)(?:\s|$)', recipient):
                    recipient = recipient[1:]
                lead = ("בהצעת המחיר הממוענת ל" + recipient + " מתוארת העבודה הבאה: " if hebrew else
                        "The quotation addressed to " + recipient + " describes the following proposed work: ")
                prefix = lead + description + " "
            elif record.document_kind in {"quote", "quotation", "estimate"}:
                lead = ("בהצעת המחיר מתוארת העבודה הבאה: " if hebrew else
                        "The quotation describes the following proposed work: ")
                prefix = lead + description + " "
            elif record.document_kind == 'invoice_receipt':
                lead = ("בחשבונית מס / קבלה מופיע סעיף המתייחס ל" if hebrew else
                        "The invoice / receipt includes an item concerning ")
                punctuation = '' if description.rstrip().endswith(('.', '!', '?')) else '.'
                prefix = lead + description + punctuation + " "
            else:
                lead = "לפי המסמך: " if hebrew else "According to the document: "
                prefix = lead + description + " "
            expected_markers = set(titles + [p.citation_id for p in recipients] + own)
            matching = []
            for line_number,line in enumerate(answer_lines):
                if not line.startswith(prefix):
                    continue
                tail = line[len(prefix):]
                found = re.findall(r"\[(\d+)\]", tail)
                if found and not re.sub(r"\[\d+\]|\s+", "", tail):
                    matching.append((set(found), line_number))
            positions = [pos for markers,pos in matching if markers == expected_markers]
            if len(positions) != 1:
                return False
            item_line_positions.extend(positions)
        # Reconstruct bounded source occurrences independently of the
        # submitted answer and require each selected exact span below.
        from subject_native_work import _display_context_span
        candidates=[]
        for ordinal,(_start,_end,text) in enumerate(spans):
            witness=witnesses.get((source,page,ordinal))
            if (roles[ordinal] in {'condition','possible'} and witness
                    and witness.scope != 'elsewhere'):
                candidates.append((ordinal,roles[ordinal],_display_context_span(
                    text,record,semantic_checked=(witness.support=='semantic_checked'))))
        conditions=[entry for entry in candidates if entry[1]=='condition']
        def exceeds(entries):
            return (any(value is None for _,_,value in entries) or len(entries)>4
                or sum(len(value) for _,_,value in entries if value is not None)>1200
                or any(len(value)>780 or len(value.split())>78 for _,_,value in entries
                       if value is not None))
        condition_overflow=exceeds(conditions)
        page_overflow=exceeds(candidates)
        ordered=conditions+[entry for entry in candidates if entry[1]=='possible']
        selected_candidates=[]
        for entry in ordered:
            if not exceeds(selected_candidates+[entry]):
                selected_candidates.append(entry)
        selected_ordinals={entry[0] for entry in selected_candidates}
        first_item_line = min(item_line_positions)
        work_leads = ("בהצעת המחיר מתוארת העבודה הבאה: " if hebrew else
                      "The quotation describes the following proposed work: ",
                      "לפי המסמך: " if hebrew else "According to the document: ",
                      "בחשבונית מס / קבלה מופיע סעיף המתייחס ל" if hebrew else
                      "The invoice / receipt includes an item concerning ",
                      "בהצעת המחיר הממוענת ל" if hebrew else "The quotation addressed to ")
        end_line = next((i for i in range(first_item_line + 1, len(answer_lines))
                         if any(answer_lines[i].startswith(lead) for lead in work_leads)),
                        len(answer_lines))
        context_answer = chr(10).join(answer_lines[first_item_line + 1:end_line])
        def source_condition_scope(text):
            # Recompute the source wording independently of model decisions.
            normalized=' '.join(text.split())
            document_match=re.match(r'(?i)^\s*(?:for\s+all\s+items|all\s+items|'
                r'לכל\s+הפריטים|כל\s+הפריטים)\s*[:：]\s*(.+)$',normalized)
            if document_match:
                tail=document_match.group(1)
                if (not re.search(r'\b(?:except|excluding|only|other|others|not|unless)\b|'
                                  r'למעט|פרט\s+ל|חוץ\s+מ|רק|לא|אחר',tail,re.I)
                        and not any(' '.join(item.description.split()).casefold() in tail.casefold()
                                    for item in record.items if item.description.strip())):
                    return 'document'
            explicit=[]
            for item_index,item in enumerate(record.items):
                name=' '.join(item.description.split())
                if not name:
                    continue
                match=re.match(r'(?i)^\s*(?P<exclusive>only\s+for\s+|רק\s+עבור\s+|'
                    r'for\s+|עבור\s+)'+re.escape(name)+r'\s*[:：]\s*(?P<tail>.+)$',normalized)
                if not match:
                    continue
                tail=match.group('tail')
                if re.search(r'\b(?:except|excluding|other|others|all|both|each|every|also|'
                             r'including|elsewhere|otherwise)\b|למעט|פרט\s+ל|חוץ\s+מ|'
                             r'אחר|אחרים|כולם|כולן|כל\s+ה|גם|וכן',tail,re.I):
                    continue
                if any(j!=item_index and ' '.join(other.description.split()).casefold() in tail.casefold()
                       for j,other in enumerate(record.items) if other.description.strip()):
                    continue
                exclusive=bool(re.match(r'(?i)^(?:only\s+for|רק\s+עבור)',match.group('exclusive')))
                if item_index in selected_indices or exclusive:
                    explicit.append(item_index)
            if len(explicit)==1:
                return 'item' if explicit[0] in selected_indices else 'elsewhere'
            return 'unresolved'
        # Independently evaluate whether both passes covered the complete
        # context. A work label is safe only for a source-backed heading;
        # otherwise it may hide an alias or another material relationship.
        checked=bool(isinstance(semantic,dict) and len(semantic)==len(inventory)
            and all(decision and decision[0]!='uncertain' and decision[1]!='unresolved'
                    and (i>=len(spans) or decision[0]!='work' or source_role_reason(spans[i][2]))
                    for i,decision in semantic.items()))
        if checked:
            for index in range(len(record.items)):
                decision=semantic[len(spans)+index]
                expected='item' if index in selected_indices else 'elsewhere'
                if decision[:2]!=('work',expected) or 'i'+str(index) not in decision[2]:
                    checked=False
                    break
        for ordinal,(start,end,text) in enumerate(spans):
            key = (source,page,ordinal)
            needed.add(key)
            w = witnesses.get(key)
            role = roles[ordinal]
            if (w is None or w.start != start or w.end != end
                    or w.text_sha256 != hashlib.sha256(text.encode()).hexdigest()):
                return False
            if role == 'other' and w.support == 'source_publisher_value':
                support=publisher_value_support(text)
            elif role == 'other' and w.support == 'source_publisher_field':
                support=publisher_label_support(text)
            elif role == 'other' and w.support == 'source_publisher':
                support=publisher_proposition_support(text) or (start,end)
            else:
                support=(start,end)
            if (support is None or (w.support_start,w.support_end) != support
                    or w.support_sha256 != hashlib.sha256(record.source_text[support[0]:support[1]].encode()).hexdigest()):
                return False
            if w.support == 'semantic_checked':
                decision=semantic.get(ordinal) if isinstance(semantic,dict) else None
                if decision is None:
                    return False
                semantic_role,semantic_scope,_support=decision
                if semantic_role!='condition':
                    return False
                mapped=('condition' if semantic_role=='condition' else
                        ('other' if semantic_scope=='elsewhere' else 'possible') if semantic_role=='work' else
                        'other' if semantic_role=='administrative' else '')
                expected_scope='other' if mapped=='other' else semantic_scope
                if semantic_role=='condition':
                    if semantic_scope in {'item','elsewhere'}:
                        source_scope=source_condition_scope(text)
                        if semantic_scope=='item' and source_scope=='item':
                            expected_scope='item'
                        elif semantic_scope=='elsewhere' and source_scope=='elsewhere' and checked:
                            expected_scope='elsewhere'
                        else:
                            expected_scope='unresolved'
                    elif semantic_scope=='document':
                        expected_scope='document' if checked else 'unresolved'
                    elif semantic_scope!='item' or source_condition_scope(text)!='item':
                        expected_scope='unresolved'
                if not mapped or role!=mapped or w.scope!=expected_scope:
                    return False
            else:
                if role == 'other':
                    reason = source_role_reason(text)
                    if not reason or (w.scope,w.support) != ('other',reason):
                        return False
                else:
                    # Only an exact, explicit whole-item reference can support an
                    # item or other-item association. Otherwise retain source scope.
                    explicit = []
                    normalized = ' '.join(text.split())
                    for item_index,item in enumerate(record.items):
                        name = ' '.join(item.description.split())
                        if not name:
                            continue
                        match = re.match(r'(?i)^\s*(?P<exclusive>only\s+for\s+|רק\s+עבור\s+|'
                            r'for\s+|עבור\s+)' + re.escape(name) + r'\s*[:：]\s*(?P<tail>.+)$', normalized)
                        if not match:
                            continue
                        tail = match.group('tail')
                        if re.search(r'\b(?:except|excluding|other|others|all|both|each|every|also|'
                                     r'including|elsewhere|otherwise)\b|למעט|פרט\s+ל|חוץ\s+מ|'
                                     r'אחר|אחרים|כולם|כולן|כל\s+ה|גם|וכן', tail, re.I):
                            continue
                        if any(j != item_index and
                               ' '.join(other.description.split()).casefold() in tail.casefold()
                               for j,other in enumerate(record.items) if other.description.strip()):
                            continue
                        exclusive = bool(re.match(r'(?i)^(?:only\s+for|רק\s+עבור)',match.group('exclusive')))
                        if item_index in selected_indices or exclusive:
                            explicit.append(item_index)
                    document_match = re.match(r'(?i)^\s*(?:for\s+all\s+items|all\s+items|'
                        r'לכל\s+הפריטים|כל\s+הפריטים)\s*[:：]\s*(.+)$', normalized)
                    document_scope = False
                    if document_match:
                        document_tail = document_match.group(1)
                        document_scope = not re.search(
                            r'\b(?:except|excluding|only|other|others|not|unless)\b|'
                            r'למעט|פרט\s+ל|חוץ\s+מ|רק|לא|אחר', document_tail, re.I)
                        document_scope = document_scope and not any(
                            ' '.join(item.description.split()).casefold() in document_tail.casefold()
                            for item in record.items if item.description.strip())
                    expected_scope = ('document' if document_scope else
                        (('item' if explicit[0] in selected_indices else 'elsewhere')
                         if len(explicit) == 1 else 'unresolved'))
                    expected_support = ('explicit_all_items_scope' if expected_scope == 'document'
                        else 'explicit_full_item_reference' if expected_scope == 'item'
                        else 'explicit_full_other_item_reference' if expected_scope == 'elsewhere'
                        else 'no_unique_explicit_item_reference')
                    if role=='condition' and expected_scope=='elsewhere':
                        expected_scope='unresolved'
                        expected_support='unresolved_other_item_scope'
                    if (w.scope,w.support) != (expected_scope,expected_support):
                        return False
                    if _nominated_qualification(text) and role != 'condition':
                        return False
                    if role == 'possible' and not (_independent_context_risk(text,evidence,source,record)
                        or expected_scope == 'item' or safe_source_note(text)):
                        return False
                    if role == 'uncertain' and (_nominated_qualification(text)
                            or (semantic is None and (_independent_context_risk(text,evidence,source,record)
                                                       or safe_source_note(text)))):
                        return False
                    if role not in {'condition','possible','uncertain'}:
                        return False
            if role != 'other':
                if (role in {'condition','possible'} and expected_scope != 'elsewhere'
                        and ordinal in selected_ordinals):
                    # Independently rederive any bounded mixed issuer sentence
                    # from the original source; no producer-provided projection
                    # is accepted as evidence of a separable field.
                    from subject_reading import _irrelevant_contact
                    raw=text.strip()
                    lines=raw.splitlines()
                    issuer_pattern=(r'(?i)\s*(?:issuer|supplier|issued\s+by|מנפיק|ספק)'
                                    r'\s*[:\-]\s*([^\W\d_]+)\s*')
                    fields=[re.fullmatch(issuer_pattern,line,re.UNICODE) for line in lines]
                    if _irrelevant_contact(raw):
                        # Independently require every discarded line to be a
                        # complete contact field and retain one exact source
                        # qualification line. Never split a mixed line.
                        from subject_reading import _instruction_like
                        contact_field = re.compile(
                            r"(?i)^(?:(?:e-?mail|phone|telephone|טלפון)\s*[:\-]\s*)?(?:"
                            r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}|(?:https?://|www\.)[^\s]+|"
                            r"[+0-9][0-9 ()\-]{6,})$")
                        source_lines = [line.strip() for line in lines if line.strip()]
                        remainder = [line for line in source_lines if not contact_field.fullmatch(line)]
                        if (len(source_lines) <= 1 or len(remainder) != 1
                                or _irrelevant_contact(remainder[0])
                                or _instruction_like(remainder[0])):
                            return False
                        candidate = remainder[0]
                    elif any(fields):
                        found=[m for m in fields if m]
                        if len(found)!=1 or len(lines)!=2:
                            return False
                        actor=found[0].group(1)
                        if issuer_span(record.source_text,actor) is None:
                            return False
                        candidate=next(line.strip() for line,field in zip(lines,fields) if not field)
                        if w.support=='semantic_checked':
                            if (not candidate or _irrelevant_contact(candidate) or _instruction_like(candidate)
                                    or re.search(r'(?i)(?:issuer|supplier|publisher|vendor|author|producer|'
                                                 r'issued\s+by|מנפיק|ספק|מחבר)\s*[:\-]',candidate)):
                                return False
                        else:
                            matches=0
                            for split in (m.start() for m in re.finditer(r'\s+',candidate)):
                                prefix,suffix=candidate[:split].strip(),candidate[split:].strip()
                                if not suffix or not _nominated_qualification(suffix):
                                    continue
                                props,errors=parse_document_role_propositions(prefix)
                                if (not errors and len(props)==1 and props[0].predicate=='published_by'
                                        and props[0].polarity=='positive'
                                        and props[0].actor.casefold()==actor.casefold()
                                        and props[0].target.casefold() in {'this quotation','this quote','this estimate'}):
                                    matches+=1
                            if matches != 1:
                                return False
                    else:
                        if re.search(r'(?i)(?:issuer|supplier|publisher|vendor|author|producer|'
                                     r'issued\s+by|מנפיק|ספק|מחבר)\s*[:\-]',raw):
                            return False
                        candidate=raw
                    if expected_scope == 'item':
                        lead = ('במקור מצוין לגבי הפריט המצוין להלן: ' if hebrew else
                                'The source states for the item named below: ')
                    elif expected_scope == 'document':
                        if w.support=='explicit_all_items_scope':
                            lead = ('במקור מצוין לגבי כל הפריטים: ' if hebrew else
                                    'The source states for all items: ')
                        else:
                            lead = ('במסמך מצוין: ' if hebrew else 'The document states: ')
                    else:
                        lead = ('מידע במקור שהקשרו לעבודה אינו ברור: ' if hebrew else
                                'Source text whose relation to this work is unclear: ')
                    context_ids = {p.citation_id for p in page_passages
                                   if p.value == "subject_work_native_context"
                                   and p.source_claim == candidate}
                    if not context_ids:
                        return False
                    pattern = re.escape(lead + candidate) + "[ \\t]+((?:\\[\\d+\\][ \\t]*)+)(?:\\n|$)"
                    cited_sets = [set(re.findall(r"\[(\d+)\]", match.group(1)))
                                  for match in re.finditer(pattern, context_answer)]
                    if context_ids not in cited_sets:
                        return False
                if page_overflow:
                    if condition_overflow:
                        overflow = True
                    else:
                        partial_context = True
                elif role == 'uncertain':
                    # Re-evaluate potentially material context from source.
                    material_note=re.search(
                        r'(?i)\b(?:proof|evidence|insurance|certification|documentation|'
                        r'coverage|training|paperwork|access|warranty|guarantee|'
                        r'cancellation|reversal|refund)\b|אסמכתא|ביטוח|הסמכה|'
                        r'תיעוד|אחריות|ביטול|זיכוי',text)
                    passive_role=re.search(r'(?i)\b[A-Za-z]+(?:ed|en)\s+by\b',text)
                    unbound_value=(re.fullmatch(r'[A-Za-zא-ת]{2,80}',text.strip())
                        and re.search(r'(?im)^\s*(?:issuer|supplier|publisher|vendor|author|producer|'
                            r'issued\s+by|מנפיק|ספק|מחבר)\s*[:\-]',record.source_text))
                    if (material_note or passive_role or unbound_value
                            or _independent_context_risk(text,evidence,source,record)
                            or safe_source_note(text)):
                        partial_context = True
                elif role == 'possible' and w.scope == 'unresolved':
                    limited = True
    if set(witnesses) != needed:
        return False
    if evidence.partial_omitted_sources:
        from monetary_questions import candidate_ids
        from subject_native_work import subject_source_coverage
        grouped={}
        for row in evidence.witness.rows:
            grouped.setdefault(row['metadata']['source_id'],[]).append(row)
        selected_sources={source for source,_,_,_ in selected}
        omitted=tuple(source for source in candidate_ids(grouped,work_request(evidence.subject),
                                                        recipient_subject=recipient_subject(evidence))
                      if source not in selected_sources)
        if (evidence.partial_omitted_sources != omitted or
                not subject_source_coverage(evidence,selected_pdf_only=True)):
            return False
        if accounted_omissions is None:
            disclaimer=('הקשר של מקורות תואמים נוספים לנושא המבוקש אינו ברור; פרטיהם אינם נכללים בתשובה.' if hebrew else
                        'The relationship of other matching sources to the requested subject is unclear; their details are not included.')
            if disclaimer not in answer:
                return False
        else:
            accounted=set(accounted_omissions)
            if not accounted <= set(omitted):
                return False
            if set(omitted)-accounted:
                disclaimer=('פרטים ממקורות תואמים נוספים לא נכללו בתשובה.' if hebrew else
                            'Details from other matching sources are not included in this answer.')
                if disclaimer not in answer:
                    return False
    if overflow:
        constraint = ('במקור מופיעים סייגים שלא ניתן להציג במלואם כאן; תיאור העבודה הוא קטע חלקי ואינו קובע מהם תנאי העבודה, אם אושרה או אם בוצעה.' if hebrew else
                      'The source contains qualifications that cannot all be shown here; the work description is only a partial excerpt and does not establish the conditions, authorization, or completion of the work.')
        if constraint not in answer:
            return False
    if partial_context:
        constraint = ('במקור יש טקסט נוסף שסיווגו אינו ברור; תיאור העבודה הוא קטע חלקי ואינו קובע מהם תנאי העבודה, אם אושרה או אם בוצעה.' if hebrew else
                      'The source contains additional unclassified text; the work description is only a partial excerpt and does not establish the conditions, authorization, or completion of the work.')
        if constraint not in answer:
            return False
    elif limited:
        constraint = ('במקור יש טקסט נוסף שסיווגו אינו ברור; קטע זה אינו קובע מהם תנאי העבודה, אם אושרה או אם בוצעה.' if hebrew else
                      'The source contains additional unclassified text; this excerpt does not establish the conditions, authorization, or completion of the work.')
        if constraint not in answer:
            return False
    return True
