"""Source-bound evidence that a quotation was addressed to a named recipient.

This witness does not identify a work site or prove the work was performed.
Callers must obtain the page from an authenticated original and independently
reopen it for final verification. Ambiguous layouts and multi-item quotes fail.
"""
from dataclasses import dataclass
import re
import unicodedata

_LABEL = re.compile(r'(?i)(?<!\w)(?:לכבוד|prepared\s+for|customer|client)(?!\w)')
_NAME = re.compile(r'^[\w\u05d0-\u05ea][\w\u05d0-\u05ea\s.\-"׳״]*$', re.UNICODE)
_ARTICLE_ROLES = frozenset({'מכללה','אוניברסיטה','חברה','מועצה','עירייה','מרכז','בנק'})


@dataclass(frozen=True)
class QuotationRecipientWitness:
    source_id: str
    page: int
    source_hash: str
    geometry_sha256: str
    recipient: str
    recipient_block: str
    item_index: int
    description: str
    document_number: str


def _name_words(value):
    return tuple(re.findall(r'[\w\u05d0-\u05ea]+', unicodedata.normalize('NFKC', value).casefold()))


def _same_recipient(requested, actual):
    wanted, seen = _name_words(requested), _name_words(actual)
    if not wanted or len(wanted) != len(seen):
        return False
    # Hebrew definite article only for an organizational role, never a name.
    def forms(word):
        variants = {word}
        if word.startswith('ה') and word[1:] in _ARTICLE_ROLES:
            variants.add(word[1:])
        return variants
    return all(forms(a) & forms(b) for a,b in zip(wanted,seen))


def indexed_recipient_candidate(subject, text):
    """Propose only; the authenticated native recipient still owns identity."""
    wanted = _name_words(subject)
    words = _name_words(str(text or ''))
    return bool(wanted and any(_same_recipient(subject, ' '.join(words[i:i+len(wanted)]))
                               for i in range(len(words)-len(wanted)+1)))


def _block_recipient(block):
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    labels = [(i, list(_LABEL.finditer(line))) for i,line in enumerate(lines)]
    labels = [(i,m) for i,m in labels if m]
    if len(labels) != 1 or len(labels[0][1]) != 1:
        return None
    index, matches = labels[0]
    match = matches[0]
    before = lines[index][:match.start()].strip(' :：')
    after = lines[index][match.end():].strip(' :：')
    if before or (after and len(lines) != 1):
        return None
    if after:
        name = after
    elif index == 0 and len(lines) == 2:
        name = lines[1].strip()
    else:
        return None
    return name if _NAME.fullmatch(name) and not re.search(r'\d',name) else None


def _has_unbound_location(description):
    # A document addressee cannot establish ownership of a different named
    # work site. Unknown locatives are treated as ambiguous, even if benign.
    return bool(re.search(r'(?i)(?<!\w)(?:at|for|in|of)\s+\w+|(?<!\w)(?:עבור|ללקוח|ללקוחה|של|אצל|מטעם)\s+\S+|(?<!\w)[בל][א-ת]{3,}', description))


def propose_quotation_recipient(page, subject, source_id):
    from monetary_native_pdf import parse_monetary_page
    if not isinstance(page,dict) or not isinstance(subject,str) or not subject.strip() or not source_id:
        return None
    blocks = page.get('native_blocks', ())
    if not isinstance(blocks,tuple):
        return None
    candidates = [block for block in blocks if len(block) == 7 and isinstance(block[4],str)
                  and _LABEL.search(block[4])]
    # Any second recipient/customer marker is ambiguous even if it agrees.
    if len(candidates) != 1 or sum(len(_LABEL.findall(b[4])) for b in candidates) != 1:
        return None
    block = candidates[0]
    recipient = _block_recipient(block[4])
    record = parse_monetary_page(page)
    if (not recipient or not _same_recipient(subject,recipient) or not record
            or page.get('page_count') != 1 or record.document_kind != 'quotation' or len(record.items) != 1
            or not record.items[0].description or _has_unbound_location(record.items[0].description)
            or not record.document_number):
        return None
    # The recipient block must exist verbatim on the same independently
    # reconstructed page as the quotation table.
    exact = block[4].strip()
    if exact not in page.get('text',''):
        return None
    source_hash = page.get('source_sha256') or page.get('raw_sha256')
    if not source_hash or not record.geometry_sha256:
        return None
    return QuotationRecipientWitness(str(source_id),page['page'],source_hash,
        record.geometry_sha256,recipient,exact,0,record.items[0].description,record.document_number)


def verify_quotation_recipient(page, subject, source_id, witness):
    """Reopen/reparse, then independently recompute owner and item binding."""
    from monetary_native_pdf import parse_monetary_page
    if not isinstance(witness, QuotationRecipientWitness) or not isinstance(page, dict):
        return False
    record = parse_monetary_page(page)
    if (not record or page.get('page_count') != 1 or record.document_kind != 'quotation'
            or len(record.items) != 1 or witness.source_id != source_id
            or witness.page != page.get('page')
            or witness.source_hash != (page.get('source_sha256') or page.get('raw_sha256'))
            or witness.geometry_sha256 != record.geometry_sha256
            or witness.item_index != 0 or witness.description != record.items[0].description
            or witness.document_number != record.document_number):
        return False
    # Independent fail-closed reading of item references: a quotation recipient
    # does not establish that work for another named beneficiary belongs to it.
    item = unicodedata.normalize('NFKC', record.items[0].description)
    tokens = re.findall(r'[A-Za-zא-ת]+', item.casefold())
    if (any(token in {'at','for','in','of','עבור','ללקוח','ללקוחה','של','אצל','מטעם'}
            for token in tokens)
            or any(len(token) >= 4 and token.startswith(('ב','ל')) and
                   re.fullmatch(r'[א-ת]+',token) for token in tokens)):
        return False
    # Re-derive the single explicit addressee from raw native blocks without
    # calling the producer's block/name parser. Labels and their role remain
    # part of the proof; a contact or supplier occurrence is never substituted.
    labelled = []
    for block in page.get('native_blocks', ()):
        if not isinstance(block, tuple) or len(block) != 7 or not isinstance(block[4], str):
            return False
        text = unicodedata.normalize('NFKC', block[4]).strip()
        matches = list(re.finditer(r'(?i)(?<!\w)(?:לכבוד|customer|client|prepared\s+for)(?!\w)',text))
        if matches:
            labelled.append((text,matches))
    if len(labelled) != 1 or len(labelled[0][1]) != 1:
        return False
    block, matches = labelled[0]
    match = matches[0]
    if block[:match.start()].strip(' :：'):
        return False
    candidate = block[match.end():].strip(' :：\n\r\t')
    if ('\n' in candidate or not candidate or re.search(r'\d',candidate)
            or not re.fullmatch(r'[\wא-ת][\wא-ת\s.\-"׳״]*',candidate)
            or block != unicodedata.normalize('NFKC',witness.recipient_block)
            or block not in unicodedata.normalize('NFKC',page.get('text',''))
            or candidate != unicodedata.normalize('NFKC',witness.recipient)):
        return False
    def tokens_of(value):
        return re.findall(r'[\wא-ת]+',unicodedata.normalize('NFKC',value).casefold())
    wanted, actual = tokens_of(subject), tokens_of(candidate)
    if not wanted or len(wanted) != len(actual):
        return False
    roles = {'מכללה','אוניברסיטה','חברה','מועצה','עירייה','מרכז','בנק'}
    for expected,found in zip(wanted,actual):
        if expected == found:
            continue
        if expected in roles and found == 'ה'+expected:
            continue
        if found in roles and expected == 'ה'+found:
            continue
        return False
    return True
