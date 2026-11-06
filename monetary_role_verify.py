"""Independent acceptance of source-owned monetary role excerpts."""
import re

from financial_semantics import monetary_question, canonical_payment_method, _CANCELLED
from monetary_native_pdf import parse_monetary_page, native_cell_literal_spans
from monetary_presentation import material_monetary_conditions, requested_source_details
from monetary_questions import selected_items, evidence_monetary_request, _PAYMENT_PENDING, _PAYMENT_OCCURRED


def validate_monetary_role_witnesses(evidence, contract, answer, graph):
    request=evidence_monetary_request(evidence,contract.normalized_request)
    if not request or request.operation not in {'payment_status','description_price'} or request.unsupported_payment_method_filter:
        return False
    # This validator currently admits only native ruled payment rows with all
    # fields in their selected item; unsupported layouts must not fall back to
    # broad source excerpts.
    selected=selected_items(evidence,request)
    selected_keys={(source,record.page) for source,record,_,_ in selected}
    represented={(p.source_id,p.metadata.get('page')) for p in evidence.passages}
    if not represented or not represented<=selected_keys:
        return False
    missing=selected_keys-represented
    if missing and (evidence.complete or not any(text in answer for text in (
            'Other matching source fields or qualifications could not be verified.',
            'שדות או תנאים במקורות תואמים נוספים לא אומתו.'))):
        return False
    groups={}
    for source,record,index,item in selected:
        if (source,record.page) in represented:
            groups.setdefault((source,record.page),[]).append((index,item))
    if not groups:
        return False
    rows={(row['metadata']['source_id'],row['metadata']['page']):row
          for row in evidence.native_rows}
    if len(rows)!=len(evidence.native_rows):
        return False
    from chat_api import DOCS_BASE_DIR
    from source_access import pdf_source_pages,SourceAccessError
    from monetary_questions import matches_terms
    expected={}
    checked_records={}
    checked_events={}
    checked_contexts={}
    try:
        for key,items in groups.items():
            source,page=key
            row=rows.get(key)
            if row is None:return False
            originals=pdf_source_pages(DOCS_BASE_DIR,evidence.tenant_id,row['metadata'],
                                       (page,),include_geometry=True)
            if len(originals)!=1:return False
            original=originals[0]
            payloads=[payload for sid,payload in evidence.geometry_pages
                      if sid==source and payload.get('page')==page]
            if len(payloads)!=1 or payloads[0]!=original:return False
            record=parse_monetary_page(original)
            if not record:return False
            checked_records[key]=record
            roles={'identity':record.document_identity_word_ids}
            if len({sid for sid,_rec,_idx,_item in selected})>1 and record.issue_date:
                date_literal=record.issue_date.strftime('%d/%m/%Y')
                found=list(re.finditer(re.escape(date_literal),original['text']))
                if len(found)!=1:return False
                expected[(source,page,'identity_date')]=((found[0].start(),found[0].end()),)
            if len(record.items)==1:
                roles.update(vat_label=record.vat_label_word_ids,tax=record.tax.word_ids,
                             payable=record.payable.word_ids)
            elif record.payable:
                roles['payable']=record.payable.word_ids
            for index,item in items:
                if index>=len(record.items) or item!=record.items[index]:
                    return False
                unit_heading = False
                if request.terms and not matches_terms(request.terms,item.description):
                    from monetary_questions import _words,_word_variants
                    from monetary_presentation import _FIELD_BOUNDARY,_CONTACT_VALUE
                    missing=tuple(term for term in request.terms if not matches_terms((term,),item.description))
                    labels=[]
                    for line in record.source_text.splitlines():
                        if (line.strip() and len(line)<=180 and missing and
                            any(matches_terms((term,),line) for term in missing) and
                            matches_terms(_words(line),record.service_context_text)):
                            labels.append(line.strip())
                    labels=tuple(dict.fromkeys(labels))
                    if not labels:return False
                    for n,label in enumerate(labels):
                        contact=_FIELD_BOUNDARY.search(label)
                        owner=label
                        if contact:
                            suffix=label[contact.start():]
                            field=_FIELD_BOUNDARY.match(suffix)
                            if not field or not _CONTACT_VALUE.fullmatch(suffix[field.end():].strip()):
                                return False
                            owner=label[:contact.start()].rstrip()
                        prefix=re.match(r'(?i)^\s*(?:customer|service(?:\s+(?:site|location))?|project|site|'
                            r'work(?:\s+site)?|לקוח(?:ה)?|אתר(?:\s+שירות)?|פרויקט|'
                            r'מקום\s+ביצוע|תיאור\s+עבודה)\s*:\s*',owner)
                        if prefix:
                            present=_word_variants(_words(owner[prefix.end():]))
                            wanted=_word_variants(tuple(missing))
                            if (not present or not wanted or
                                not all(any(a & b for b in wanted) for a in present) or
                                not all(any(a & b for a in present) for b in wanted)):
                                return False
                        else:
                            heading=re.fullmatch(
                                r'(?i)\s*(?:הצעת\s+מחיר|quotation|quote|estimate)\s+'
                                r'(?:שעות|שעה|hours?|hrs?)\s+(.+?)\s*',owner)
                            if (len(record.items)!=1 or not heading or
                                    not all(re.fullmatch(r'(?i)שעות|שעה|hours?|hrs?',term) for term in missing) or
                                    not re.fullmatch(r'[A-Za-z\u05d0-\u05ea\s]+',heading.group(1)) or
                                    not matches_terms(_words(heading.group(1)),item.description)):
                                return False
                            unit_heading = True
                        found=list(re.finditer(re.escape(owner),original['text']))
                        if len(found)!=1:return False
                        expected[(source,page,f'item:{index}:context:{n}')]=((found[0].start(),found[0].end()),)
                # Recheck the exact source-owned description for contact or
                # account content without trusting the producer's role label.
                from monetary_presentation import _CONTACT_VALUE
                if re.search(r'(?i)\b(?:bank\s+)?account(?:\s*(?:number|no\.?|#|:))?\s*[:#-]?\s*\d|מס(?:פר)?\s*חשבון',item.description):
                    return False
                for contact in _CONTACT_VALUE.finditer(item.description):
                    token=contact.group()
                    before=item.description[max(0,contact.start()-24):contact.start()]
                    if ('@' in token or re.match(r'(?i)(?:https?://|www\.)',token) or
                        re.search(r'(?i)(?:\b(?:tel|telephone|phone|call|contact)|טלפון|נייד)\s*[:.\-]?\s*$',before)):
                        return False
                    if not re.search(r'(?i)(?:(?:\b(?:model|serial|sku)|דגם)\s*[:#-]?\s*(?:[A-Za-z]{1,8}-)?|'
                                     r'(?:USD|EUR|ILS|₪|[$€£])\s*)$',before):
                        return False
                roles[f'item:{index}:description']=item.description_word_ids
                if request.role_requested!='unit_price':
                    roles[f'item:{index}:line_total']=item.line_total_word_ids
                if 'quantity' in request.requested_fields or request.role_requested=='unit_price' or unit_heading:
                    roles[f'item:{index}:quantity']=item.quantity_word_ids
                if request.role_requested=='unit_price' or unit_heading:
                    roles[f'item:{index}:unit_price']=item.unit_price_word_ids
                for n,condition in enumerate(material_monetary_conditions(record,item)):
                    from monetary_presentation import _CONTACT_VALUE
                    if (_CONTACT_VALUE.search(condition) or
                        re.search(r'(?i)\b(?:bank\s+)?account(?:\s*(?:number|no\.?|#|:))?\s*[:#-]?\s*\d|מס(?:פר)?\s*חשבון',condition)):
                        return False
                    pattern=r'\s+'.join(re.escape(piece) for piece in condition.split())
                    found=list(re.finditer(pattern,original['text'])) if pattern else []
                    if (len(found)!=1 or not any(found[0].group() in block for block in record.qualifiers)):
                        return False
                    end=found[0].end()
                    if original['text'][end:end+1]==';':end+=1
                    expected[(source,page,f'item:{index}:condition:{n}')]=((found[0].start(),end),)
            for field,value in requested_source_details(record,request.requested_fields):
                if value is None:continue
                literal=record.issue_date.strftime('%d/%m/%Y') if field=='issue_date' else value
                found=list(re.finditer(re.escape(literal),original['text']))
                if len(found)!=1:return False
                expected[(source,page,'detail:'+field)]=((found[0].start(),found[0].end()),)
            checked_events[key]={}
            payment_requested=(request.operation=='payment_status' or
                bool(re.search(r'\bpayment\b|(?<!\w)ה?תשלום(?!\w)',contract.normalized_request,re.I)))
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
                        if not occurrence:return False
                        found=list(re.finditer(re.escape(occurrence.group()),original['text']))
                        if len(found)!=1:return False
                        expected[(source,page,f'payment:{index}:occurrence')]=((found[0].start(),found[0].end()),)
                    checked_events[key][index]=(payment.amount,payment.currency,payment.payment_date,payment.method)
                    roles[f'payment:{index}:amount']=payment.amount_word_ids
                    roles[f'payment:{index}:date']=payment.date_word_ids
                    roles[f'payment:{index}:method']=payment.method_word_ids
            elif show_payments and len(record.items)==1 and not record.payment_table_present:
                # Independently rebuild each fallback event from a single
                # source line; never carry a neighboring row's amount/method.
                from financial_semantics import (extract_financial_record,_PAYMENT_HEADING,
                    _BUSINESS_DATE,_PAYMENT_METHOD,_money_matches,_CURRENCY,_date_value)
                from financial_documents import canonical_document_kind
                financial=extract_financial_record(record.source_text,source_id=source,
                    title='',tenant_id=evidence.tenant_id,page=record.page,include_quotations=True)
                events=[]
                if financial and canonical_document_kind(record.document_kind)!='credit' and not _CANCELLED.search(record.source_text):
                    for payment in financial.payments:
                        a,b=payment.span
                        line_start=record.source_text.rfind('\n',0,a)+1
                        line_end=record.source_text.find('\n',b)
                        line_end=len(record.source_text) if line_end<0 else line_end
                        heading=_PAYMENT_HEADING.search(record.source_text)
                        context_start=max(heading.start() if heading else 0,line_start-120)
                        context=record.source_text[context_start:line_end]
                        if (_PAYMENT_PENDING.search(context) or
                            (financial.document_kind not in {'receipt','invoice_receipt'} and
                             not _PAYMENT_OCCURRED.search(context))):continue
                        date_start,date_end=payment.date_span
                        first=record.source_text.rfind('\n',0,date_start)+1
                        last=record.source_text.find('\n',date_end)
                        last=len(record.source_text) if last<0 else last
                        line=record.source_text[first:last]
                        amounts=_money_matches(line);dates=list(_BUSINESS_DATE.finditer(line))
                        methods=list(_PAYMENT_METHOD.finditer(line))
                        if (len(amounts)!=1 or len(dates)!=1 or len(methods)>1 or
                            (first+dates[0].start(),first+dates[0].end())!=(date_start,date_end)):
                            continue
                        currencies=[c for c in _CURRENCY.finditer(line)
                            if ((c.end()<=amounts[0][0].start() and not line[c.end():amounts[0][0].start()].strip())
                                or (c.start()>=amounts[0][0].end() and not line[amounts[0][0].end():c.start()].strip()))]
                        if len(currencies)!=1:continue
                        event=(amounts[0][1],amounts[0][2],_date_value(dates[0]),methods[0].group() if methods else '')
                        amt=amounts[0][0];currency=currencies[0]
                        spans={'amount':(first+min(amt.start(),currency.start()),first+max(amt.end(),currency.end())),
                               'date':(first+dates[0].start(),first+dates[0].end())}
                        if methods:
                            spans['method']=(first+methods[0].start(),first+methods[0].end())
                        events.append((event,spans))
                # Indices include filtered-out methods, matching the original
                # source event order rather than renumbering a filtered list.
                for index,(event,spans) in enumerate(events):
                    if request.payment_method and canonical_payment_method(event[3])!=request.payment_method:continue
                    if record.document_kind not in {'receipt','invoice_receipt'}:
                        first=spans['date'][0]
                        line_start=record.source_text.rfind('\n',0,first)+1
                        line_end=record.source_text.find('\n',first)
                        line_end=len(record.source_text) if line_end<0 else line_end
                        line=record.source_text[line_start:line_end]
                        occurrence=_PAYMENT_OCCURRED.search(line)
                        if occurrence:
                            span=(line_start+occurrence.start(),line_start+occurrence.end())
                        else:
                            heading=_PAYMENT_HEADING.search(record.source_text)
                            occurrence=_PAYMENT_OCCURRED.search(heading.group()) if heading else None
                            if not occurrence or heading.start()>=line_start:return False
                            found=list(re.finditer(re.escape(occurrence.group()),record.source_text))
                            if len(found)!=1:return False
                            span=(found[0].start(),found[0].end())
                        expected[(source,page,f'payment:{index}:occurrence')]=(span,)
                    checked_events[key][index]=event
                    for label,span in spans.items():
                        expected[(source,page,f'payment:{index}:{label}')]=(span,)
            # Rebuild all source context occurrences from the reopened page.
            # Qualifier nomination is only a reason to quote; unclassified
            # material remains an explicit uncertainty, never proof of absence.
            from subject_native_work import (_context_spans,_nominated_qualification,
                _ordinary_title_block,_display_context_span)
            from subject_reading import _irrelevant_contact
            spans=_context_spans(record)
            if spans is None:return False
            shown={' '.join(value.split()) for _i,item in items
                   for value in material_monetary_conditions(record,item)}
            context_roles=[];uncertain=False
            selected_indices={i for i,_item in items}
            for ordinal,(_first,_last,block) in enumerate(spans):
                if _ordinary_title_block(block,record):continue
                from subject_reading import _instruction_like
                if _instruction_like(str(block)):
                    if _nominated_qualification(block):return False
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
                    candidate=residual.strip()
                    owners=[]
                    if candidate and '\n' not in candidate:
                        for j,other in enumerate(record.items):
                            if j in selected_indices:continue
                            name=' '.join(other.description.split())
                            if (not name or sum(' '.join(item.description.split()).casefold()==name.casefold()
                                                for item in record.items)!=1):
                                continue
                            word=r'[^\W\d_]+(?:[-\u2019\'][^\W\d_]+)?'
                            pattern=(r'(?i)^'+re.escape(name)+r'\s+(?:is\s+)?subject\s+to\s+'+
                                     word+r'(?:\s+'+word+r')?[.!?]$')
                            if re.fullmatch(pattern,candidate):owners.append(j)
                    from monetary_presentation import _CONTACT_VALUE
                    if (len(owners)==1 and not _CONTACT_VALUE.search(candidate) and
                        not re.search(r'(?i)\b(?:bank\s+)?account(?:\s*(?:number|no\.?|#|:))?\s*[:#-]?\s*\d|מס(?:פר)?\s*חשבון',candidate) and
                            record.source_text.count(candidate)==1):
                        pos=record.source_text.find(candidate)
                        role=f'source_elsewhere_context:{ordinal}'
                        expected[(source,page,role)]=((pos,pos+len(candidate)),)
                        context_roles.append((role,candidate))
                        uncertain=True
                        continue
                # Recheck the only scope that permits omission: an explicit,
                # exclusive, noncontradictory complete reference to another
                # item. All other associations remain source-level uncertain.
                normalized=' '.join(str(block).split())
                elsewhere=[]
                for j,other in enumerate(record.items):
                    name=' '.join(other.description.split())
                    if not name:continue
                    match=re.match(r'(?i)^\s*(?:only\s+for\s+|רק\s+עבור\s+)'
                        +re.escape(name)+r'\s*[:：]\s*(?P<tail>.+)$',normalized)
                    if not match or j in selected_indices:continue
                    tail=match.group('tail')
                    if re.search(r'\b(?:except|excluding|other|others|all|both|each|every|also|'
                                 r'including|elsewhere|otherwise)\b|למעט|פרט\s+ל|חוץ\s+מ|'
                                 r'אחר|אחרים|כולם|כולן|כל\s+ה|גם|וכן',tail,re.I):
                        continue
                    if any(k!=j and ' '.join(other_item.description.split()).casefold() in tail.casefold()
                           for k,other_item in enumerate(record.items) if other_item.description.strip()):
                        continue
                    elsewhere.append(j)
                if len(elsewhere)==1:continue
                display=_display_context_span(block,record)
                if not display or _irrelevant_contact(display):return False
                from monetary_presentation import _CONTACT_VALUE
                if (_CONTACT_VALUE.search(display) or
                    re.search(r'(?i)\b(?:bank\s+)?account(?:\s*(?:number|no\.?|#|:))?\s*[:#-]?\s*\d|מס(?:פר)?\s*חשבון',display)):
                    return False
                if ' '.join(display.split()) in shown:continue
                if record.source_text.count(display)!=1:return False
                pos=record.source_text.find(display)
                if re.match(r'^\s*(?:[+\-]\s+)?[0-9]+\s+(?:[+\-]\s+)?',display) and re.search(r'[֐-׿]',display):
                    from monetary_native_pdf import native_rtl_numeric_context
                    reconstructed=native_rtl_numeric_context(original,display)
                    if reconstructed is None:
                        labels=[m for m in re.finditer(r'[֐-׿]+(?:\s+[֐-׿]+)*',display)
                                if record.source_text.count(m.group())==1]
                        if len(labels)!=1:return False
                        a=pos+labels[0].start()
                        b=pos+labels[0].end()
                        role=f'source_context_limited:{ordinal}'
                        expected[(source,page,role)]=((a,b),)
                        context_roles.append((role,display))
                    else:
                        reading,literal_spans=reconstructed
                        role=f'source_reconstructed_context:{ordinal}'
                        expected[(source,page,role)]=literal_spans
                        context_roles.append((role,reading))
                else:
                    role=f'source_context:{ordinal}'
                    expected[(source,page,role)]=((pos,pos+len(display)),)
                    context_roles.append((role,display))
            if (len(context_roles)>4 or sum(len(text) for _,text in context_roles)>1200 or
                    any(len(text)>780 or len(text.split())>78 for _,text in context_roles)):
                return False
            checked_contexts[key]=(dict(context_roles),uncertain)
            for role,ids in roles.items():
                spans=native_cell_literal_spans(original,ids)
                if not spans:return False
                expected[(source,page,role)]=tuple(spans)
            from financial_documents import primary_document_identities,role_matches
            identity=expected.get((source,page,'identity'))
            if not identity or len(identity)!=1:return False
            a,b=identity[0]
            identities=primary_document_identities(original['text'][a:b])
            if (len(identities)!=1 or identities[0].span!=(0,b-a) or
                    identities[0].number!=record.document_number or
                    not role_matches(identities[0].kind,record.document_kind)):
                return False
    except (SourceAccessError,OSError,ValueError,KeyError,TypeError):
        return False
    observed={}
    for passage in evidence.passages:
        role=str(passage.value).removeprefix('monetary_role:')
        key=(passage.source_id,passage.metadata.get('page'),role)
        span=passage.metadata.get('_monetary_source_span')
        if (key not in expected or not isinstance(span,(tuple,list)) or len(span)!=2 or
                any(type(value) is not int for value in span)):
            return False
        payload=next((p for sid,p in evidence.geometry_pages
                      if sid==key[0] and p.get('page')==key[1]),None)
        if (not payload or not 0<=span[0]<span[1]<=len(payload['text']) or
                passage.text!=passage.source_claim or payload['text'][slice(*span)]!=passage.source_claim):
            return False
        observed.setdefault(key,[]).append(tuple(span))
    if set(observed)!=set(expected) or any(tuple(observed[k])!=expected[k] for k in expected):
        return False
    # Each claim is supported by a source-local bundle of field witnesses.
    # Verify both the bundle type and the literal source-derived values in the
    # displayed claim; a payment event requires all three same-row atoms.
    by_id={p.citation_id:p for p in evidence.passages}
    if len(by_id)!=len(evidence.passages):return False
    covered=set()
    for claim in graph.claims:
        if not claim.citation_ids or any(cid not in by_id for cid in claim.citation_ids):
            return False
        linked=[by_id[cid] for cid in claim.citation_ids]
        keys={(p.source_id,p.metadata.get('page')) for p in linked}
        if len(keys)!=1:return False
        record=checked_records.get(next(iter(keys)))
        if not record:return False
        names={p.metadata.get('_monetary_role') for p in linked}
        if any(p.value!='monetary_role:'+str(p.metadata.get('_monetary_role')) for p in linked):
            return False
        if tuple(p.fact_id for p in linked)!=claim.fact_ids:return False
        text=' '.join(claim.text.split())
        def amount(value,currency):
            return f'{value:,.2f} {currency}'
        remaining=set(names)
        if 'identity' in remaining:
            from financial_documents import document_kind_label
            labels=[document_kind_label(record.document_kind,lang) for lang in (False,True)]
            labels.extend(('Document '+record.document_number,'מסמך '+record.document_number))
            if record.document_kind in {'quote','quotation','estimate'}:
                labels.extend(('Quoted price','מחיר מוצע'))
            uncertain_text=(
                'The source includes additional context whose relationship to the payment and work is unclear.',
                'במקור יש מידע נוסף שהקשרו לתשלום ולעבודה אינו ברור.')
            if not any(label in text for label in labels):
                implicit_field=(request.operation=='description_price' and
                    request.role_requested in {'tax_amount','unit_price'} and
                    any(name=='tax' or str(name).endswith(':unit_price') for name in names))
                if not implicit_field and (text not in uncertain_text or not checked_contexts[next(iter(keys))][1]):
                    return False
            remaining.remove('identity')
        payments={int(m.group(1)) for name in remaining if (m:=re.fullmatch(r'payment:(\d+):(?:amount|date|method|occurrence)',str(name)))}
        for index in payments:
            event=checked_events[next(iter(keys))].get(index)
            if not event:return False
            needed={f'payment:{index}:amount',f'payment:{index}:date'}
            if event[3]:needed.add(f'payment:{index}:method')
            if record.document_kind not in {'receipt','invoice_receipt'}:
                needed.add(f'payment:{index}:occurrence')
            if not needed<=remaining:return False
            if (amount(event[0],event[1]) not in text or event[2].isoformat() not in text or event[3] not in text):
                return False
            remaining-=needed
        if 'identity_date' in remaining:
            if not record.issue_date or record.issue_date.isoformat() not in text:return False
            remaining.remove('identity_date')
        if remaining=={'tax','vat_label'}:
            if amount(record.tax.amount,record.tax.currency) not in text:return False
            remaining.clear()
        if remaining=={'payable'}:
            if amount(record.payable.amount,record.payable.currency) not in text:return False
            remaining.clear()
        for name in tuple(remaining):
            match=re.fullmatch(r'item:(\d+):(description|line_total|unit_price|quantity)',str(name))
            if not match:continue
            index=int(match.group(1))
            if index>=len(record.items):return False
            item=record.items[index]
            field=match.group(2)
            value={'description':item.description,'line_total':amount(item.line_total,item.currency),
                   'unit_price':amount(item.unit_price,item.currency),'quantity':str(item.quantity)}[field]
            if value not in text:return False
            remaining.remove(name)
        for name in tuple(remaining):
            if str(name).startswith('source_elsewhere_context:'):
                context=checked_contexts[next(iter(keys))][0].get(name)
                source,page=next(iter(keys))
                selected_descriptions={
                    f'item:{index}:description' for sid,record_value,index,_item in selected
                    if sid==source and record_value.page==page}
                # The sibling passage supplies citation context, never an
                # assertion that the sibling's condition applies here. It
                # must accompany a separately verified selected-item claim.
                if (context is None or not selected_descriptions.intersection(names)
                        or ' '.join(context.split()) in text):
                    return False
                remaining.remove(name)
                continue
            if str(name).startswith('source_context_limited:'):
                context=checked_contexts[next(iter(keys))][0].get(name)
                if context is None or text not in answer or not any(phrase in text for phrase in (
                        'A source qualification cannot be reconstructed safely in its original order',
                        'תנאי שלא ניתן לשחזר בבטחה את סדר מילותיו')):
                    return False
                remaining.remove(name)
                continue
            if str(name).startswith('source_reconstructed_context:'):
                context=checked_contexts[next(iter(keys))][0].get(name)
                if (context is None or text not in answer or ' '.join(context.split()) not in text or
                        not any(phrase in text for phrase in (
                            'Reconstructed from the source layout',
                            'שחזור לפי מיקום הטקסט במקור'))):
                    return False
                remaining.remove(name)
                continue
            if str(name).startswith('source_context:'):
                context=checked_contexts[next(iter(keys))][0].get(name)
                if context is None or ' '.join(context.split()) not in text:return False
                remaining.remove(name)
                continue
            match=re.fullmatch(r'item:(\d+):context:(\d+)',str(name))
            if match:
                item=record.items[int(match.group(1))]
                if item.description not in text:return False
                remaining.remove(name)
                continue
            match=re.fullmatch(r'item:(\d+):condition:(\d+)',str(name))
            if match:
                item=record.items[int(match.group(1))]
                conditions=material_monetary_conditions(record,item)
                index=int(match.group(2))
                if index>=len(conditions) or conditions[index] not in text:return False
                remaining.remove(name)
                continue
            if str(name).startswith('detail:'):
                field=str(name)[len('detail:'):]
                details=dict(requested_source_details(record,request.requested_fields))
                if field not in details or details[field] is None or details[field] not in text:return False
                remaining.remove(name)
        if remaining:return False
        covered.update(claim.citation_ids)
    if covered!=set(by_id):return False
    # Unclassified text alone is not a missing answer requirement.
    return True
