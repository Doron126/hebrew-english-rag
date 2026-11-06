"""Independent source-block relevance check for document discovery."""
import re
from pathlib import PurePosixPath

from subject_work_verify import _includes, _tokens, _word_forms


def _filename_proves_subject(subject, name):
    """Independently bind a whole subject to one title proposition."""
    wanted = _word_forms(_tokens(subject))
    if not wanted:
        return False
    stem = PurePosixPath(str(name)).stem
    pieces = re.split(
        r'(?<!\d),(?!\d)|(?<=\s)[-–—](?=\s)|[;!?]|'
        r'\b(?:while|whereas|but|however|although|and|or)\b|'
        r'(?<!\w)(?:בעוד|ואילו|אבל|אך)(?!\w)|'
        r'(?<!\w)ו(?=(?:[בלמ])?(?:ה)?(?:תקל(?:ה|ת)|תיקון|התקנ(?:ה|ת)|טיפול|עבוד(?:ה|ת)|שירות|החלפ(?:ה|ת)|'
        r'מוקד|מרכז|אתר|פרויקט|מועצה|מתקן|בניין|בית|מחלקה|חדר|כיתה|אולם|קומה|'
        r'מצלמה|מיקרופון|בקר|מסך|מקרן|רמקול|מערכת)(?!\w))',
        stem, flags=re.I)
    for piece in pieces:
        if re.search(r'\b(?:no|not|without|except)\b|(?<!\w)(?:לא|ללא|למעט)(?!\w)', piece, re.I):
            continue
        observed = _word_forms(_tokens(piece))
        if any(all(wanted[i] & observed[start+i] for i in range(len(wanted)))
               for start in range(len(observed)-len(wanted)+1)):
            return True
    return False


def _native_item_fragment(fragment, description):
    # Recompute ownership independently and reject partial native source words.
    import unicodedata
    def normalized(value):
        return ' '.join(unicodedata.normalize('NFKC',value).casefold().split())
    fragment=normalized(fragment)
    if not fragment:
        return False
    description=normalized(description)
    numeric_start=bool(re.match(r'^[^\w\s]*\d',fragment))
    numeric_end=bool(re.search(r'\d[^\w\s]*$',fragment))
    for match in re.finditer(re.escape(fragment),description):
        before=description[match.start()-1] if match.start() else ''
        after=description[match.end()] if match.end()<len(description) else ''
        def interferes(char, edge, numeric):
            if not char or not (edge.isalnum() or edge=='_' or numeric):
                return False
            return (char.isalnum() or char=='_' or
                    (numeric and unicodedata.category(char)[0] in 'PS'))
        if not interferes(before,fragment[0],numeric_start) and not interferes(after,fragment[-1],numeric_end):
            return True
    return False


def _independent_coordinated_item(subject, query, fragment, item):
    # Independently check ordered source words and the location census in the
    # reopened item. This validates a co-mention, not the action's location.
    from subject_intent import subject_work_slots
    slots=subject_work_slots(query,subject)
    if not slots or not slots.owner or len(slots.content)!=1:
        return False
    owners=_word_forms(_tokens(slots.owner))
    actions=_word_forms(_tokens(slots.content[0]))
    tokens=_tokens(fragment)
    observed=_word_forms(tokens)
    places={'מוקד','מרכז','אתר','פרויקט','מועצה','station','site','project','center','centre'}
    if len(owners)<2 or len(actions)<2 or not owners[0] & places:
        return False
    if re.search(r'(?i)\b(?:no|not|without|except|but|although|whereas)\b|'
                 r'(?<!\w)(?:לא|ללא|למעט|אך|אבל|בעוד)(?!\w)|[;!?\n]',item):
        return False
    item_tokens=tuple(t.casefold() for t in re.findall(r'[A-Za-z0-9\u05d0-\u05ea]+',item))
    item_forms=_word_forms(item_tokens)
    if sum(bool(group & places) for group in item_forms) != 1:
        return False
    owner_starts=[i for i in range(len(item_tokens)-len(owners)+1)
                  if all(owners[k] & item_forms[i+k] for k in range(len(owners)))]
    if len(owner_starts)!=1:
        return False
    prefix=item_forms[:owner_starts[0]]
    if len(prefix)>3 or any(not v & {'טיפול','תקלה','תיקון','שירות','עבודה'} for v in prefix):
        return False
    starts=[i for i in range(len(item_tokens)-len(actions)+1)
            if all(actions[k] & item_forms[i+k] for k in range(len(actions)))]
    if len(starts)!=1:
        return False
    tail=item_tokens[starts[0]+len(actions):]
    equipment={'בקר','מסך','מגע','מקרן','רמקול','מצלמה','מיקרופון'}
    if tail and not (len(tail)>=4 and 'מערכת' in _word_forms((tail[0],))[0]
            and re.fullmatch(r'[a-z0-9]+',tail[1]) and tail[2]=='ל'
            and re.fullmatch(r'[a-z0-9]+',tail[3])
            and all(_word_forms((word,))[0] & equipment for word in tail[4:])):
        return False
    for offset in range(len(tokens)-len(owners)-len(actions)+1):
        if not all(owners[k] & observed[offset+k] for k in range(len(owners))):
            continue
        end=offset+len(owners)
        if end<len(tokens) and tokens[end]=='ו':
            end+=1
        elif end>=len(tokens) or not tokens[end].startswith('ו'):
            continue
        if end+len(actions)<=len(tokens) and all(actions[k] & observed[end+k] for k in range(len(actions))):
            return True
    return False


def validate_subject_document_contract(contract, evidence):
    """Recheck each displayed source against its own authorized witness.

    Source-wide words, unrelated sentences and parent-directory labels cannot
    be combined to attribute a subject to a document.
    """
    if not evidence or not evidence.witness or not evidence.passages:
        return False
    subject = contract.requirements[0].subject
    if not subject:
        return False
    rows = {}
    for row in evidence.witness.rows:
        meta = row.get('metadata') or {}
        if meta.get('tenant_id') != evidence.tenant_id:
            return False
        rows.setdefault(meta.get('source_id'), []).append(row)
    for passage in evidence.passages:
        source_rows = rows.get(passage.source_id, ())
        own = [r for r in source_rows if r.get('id') == passage.evidence_id]
        if len(own) != 1 or passage.tenant_id != evidence.tenant_id:
            return False
        row_metadata=own[0].get('metadata') or {}
        passage_metadata=getattr(passage,'metadata',None) or {}
        row_mime=row_metadata.get('mime_type')
        passage_mime=passage_metadata.get('mime_type')
        marker=str(getattr(passage,'value',''))
        if row_mime == 'application/pdf' or passage_mime == 'application/pdf':
            allowed={'subject_document_native_item','subject_document_native_context',
                     'subject_document_native_unruled','subject_document_native_coordinated'}
            if row_mime != 'application/pdf' or passage_mime != row_mime or marker not in allowed:
                return False
            keys=('source_id','tenant_id','source_sha256','source_relative_path',
                  'source_size','source_mtime_ns','mime_type','page')
            if any(row_metadata.get(key) != passage_metadata.get(key) for key in keys):
                return False
        # The normal acceptance/quote gate also verifies the source identity,
        # bytes and citation; this check independently binds relevance.
        coordinated_local=False
        if str(getattr(passage,'value','')).startswith('subject_document_native_'):
            from chat_api import DOCS_BASE_DIR
            from monetary_native_pdf import parse_monetary_page
            from source_access import SourceAccessError,pdf_source_pages
            from subject_work_unruled import verify_unruled_passages
            if passage.text != passage.source_claim:
                return False
            if passage.value == 'subject_document_native_unruled':
                from types import SimpleNamespace
                if not verify_unruled_passages(SimpleNamespace(witness=evidence.witness,
                        passages=(passage,)),evidence.tenant_id,DOCS_BASE_DIR):
                    return False
            else:
                page=passage.metadata.get('page')
                try:
                    pages=pdf_source_pages(DOCS_BASE_DIR,evidence.tenant_id,passage.metadata,
                                           (page,),include_geometry=True)
                except (OSError,ValueError,SourceAccessError):
                    return False
                record=parse_monetary_page(pages[0]) if len(pages)==1 else None
                if not record or passage.source_claim not in record.source_text:
                    return False
                item_count=sum(_native_item_fragment(passage.source_claim,item.description)
                               for item in record.items)
                if passage.value == 'subject_document_native_item':
                    if item_count != 1:
                        return False
                elif passage.value == 'subject_document_native_coordinated':
                    owners=[item for item in record.items
                            if _native_item_fragment(passage.source_claim,item.description)]
                    if (len(owners)!=1 or not _independent_coordinated_item(
                            subject,getattr(evidence,'query',''),passage.source_claim,owners[0].description)):
                        return False
                    coordinated_local=True
                elif passage.value == 'subject_document_native_context':
                    if item_count or sum(passage.source_claim in block for block in record.qualifiers) != 1:
                        return False
                else:
                    return False
        original = str(own[0].get('document') or '')
        if ' '.join(passage.source_claim.split()) not in ' '.join(original.split()):
            return False
        # A retrieval block may contain several propositions. Do not let
        # a place in one conjunct acquire a work term from another. Numeric
        # grouping commas remain intact; common contrast/conjunction markers
        # delimit claims independently of model or producer selection.
        pieces = re.split(
            r'\n\s*\n+|(?<=[.!?;])\s+|(?<!\d),(?!\d)|'
            r'\b(?:while|whereas|but|although|and|or)\b|'
            r'(?<!\w)(?:בעוד|ואילו|אך|אבל)(?!\w)|'
            r'(?<!\w)ו(?=(?:[בלמ])?(?:ה)?(?:תקל(?:ה|ת)|תיקון|התקנ(?:ה|ת)|טיפול|עבוד(?:ה|ת)|שירות|החלפ(?:ה|ת)|'
            r'מוקד|מרכז|אתר|פרויקט|מועצה|מתקן|בניין|בית|מחלקה|חדר|כיתה|אולם|קומה|'
            r'מצלמה|מיקרופון|בקר|מסך|מקרן|רמקול|מערכת)(?!\w))',
            passage.source_claim, flags=re.I)
        local = coordinated_local or any(_includes(subject, piece) for piece in pieces if piece.strip())
        name = PurePosixPath(str((own[0].get('metadata') or {}).get(
            'source_relative_path') or '')).name
        if not local and not _filename_proves_subject(subject, name):
            return False
    return True
