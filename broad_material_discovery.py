"""Conservative parsing and native-text witnesses for broad material questions.

These helpers propose a scope only. Callers must obtain ``text`` from an
authenticated original in the current tenant and retain its source binding.
"""
from dataclasses import dataclass
from contextlib import contextmanager
from contextvars import ContextVar
import re
import unicodedata


@dataclass(frozen=True)
class BroadMaterialRequest:
    subject: str
    topic: str
    owner: str


@dataclass(frozen=True)
class BroadMaterialMatch:
    owner_witness: str
    topic_witness: str


_WORD = re.compile(r"[\w\u05d0-\u05ea]+", re.UNICODE)
_STOP = frozenset({'של', 'על', 'את', 'עם', 'the', 'a', 'an', 'of', 'in', 'at', 'for', 'and'})
_CONSTRAINT = re.compile(
    r'(?i)(?<!\w)(?:כל|כולן|כולם|רשימ[הת]|כמה|מחיר|מחירים|עלות|סכום|'
    r'תאריך|השווה|השוואה|מפרט|בדיוק|[בול]?ה?פרו[יו]*קט|מסמך\s+הזה|המסמך\s+הזה|'
    r'all|every|complete|list|how\s+many|price|cost|compare|exact|project|this\s+document)(?!\w)|\d')
_HEBREW = re.compile(
    r'^(?:מה\s+(?:כתוב|מתואר|נאמר|ידוע|יש)|איזה\s+מידע\s+(?:יש|קיים)|'
    r'מה\s+אפשר\s+(?:לומר|ללמוד)|ספר(?:י|ו)?\s+לי)\s+(?:על|לגבי|אודות)\s+'
    r'(?P<topic>.+?)\s+ב(?P<owner>[א-ת][^?!.;,]*)$', re.I)
_ENGLISH = re.compile(
    r'^(?:what\s+(?:is\s+(?:written|said|available)|do\s+the\s+materials?\s+say)|'
    r'what\s+information\s+is\s+available)\s+(?:about|on)\s+'
    r'(?P<topic>.+?)\s+(?:at|for|in)\s+(?P<owner>[^?!.;,]+)$', re.I)


_routing_disabled = ContextVar('broad_material_routing_disabled', default=False)


@contextmanager
def disable_broad_material_routing():
    """Temporarily retain legacy routing for this context, including on error."""
    token = _routing_disabled.set(True)
    try:
        yield
    finally:
        _routing_disabled.reset(token)


def parse_broad_material_request(query):
    if _routing_disabled.get():
        return None
    value = unicodedata.normalize('NFKC', str(query or '')).strip(' \t\r\n?!.;,')
    if not value or _CONSTRAINT.search(value):
        return None
    match = _HEBREW.fullmatch(value) or _ENGLISH.fullmatch(value)
    if not match:
        return None
    topic = match.group('topic').strip()
    owner = match.group('owner').strip()
    if not _terms(topic) or len(_terms(owner)) < 2 or len(topic) > 100 or len(owner) > 100:
        return None
    return BroadMaterialRequest(subject=topic + ' / ' + owner, topic=topic, owner=owner)


def _terms(value):
    return tuple(word.casefold() for word in _WORD.findall(unicodedata.normalize('NFKC', str(value)))
                 if word.casefold() not in _STOP)


def _forms(word):
    """Small spelling set, not a semantic synonym or fuzzy-name matcher."""
    forms = {word}
    if re.fullmatch(r'[א-ת]{4,}', word):
        stems = {word}
        if len(word) >= 5 and word[0] in 'ובל':
            stems.add(word[1:])
        for stem in stems:
            forms.add(stem)
            if stem.endswith('ות') and len(stem) >= 5:
                forms.add(stem[:-2])
            if stem.endswith(('ה', 'ת')):
                forms.add(stem[:-1])
    elif re.fullmatch(r'[a-z]{4,}', word) and word.endswith('s'):
        forms.add(word[:-1])
    return frozenset(form for form in forms if len(form) >= 3)


def _ordered_match(needle, text):
    wanted, actual = _terms(needle), _terms(text)
    if not wanted or len(wanted) > len(actual):
        return False
    return any(all(_forms(w) & _forms(actual[start + i])
                   for i, w in enumerate(wanted))
               for start in range(len(actual) - len(wanted) + 1))


def _terminal_topic_match(needle, text):
    """Require the witnessed topic to end its subject statement.

    Only a numeric room/record identifier may follow. This deliberately refuses
    unparsed relations such as a different owner introduced after the topic.
    """
    wanted = _terms(needle)
    actual = tuple(word.casefold() for word in _WORD.findall(text))
    # Permit only ordinary framing before the topic. Arbitrary preceding
    # names could reassign the topic to a different organization.
    safe_prefix = {'הנדון', 'נושא', 'הצעה', 'הצעת', 'מחיר', 'עבור', 'הנני',
                   'מתכבדת', 'מתכבד', 'להגיש', 'מערכת', 'מולטימדיה',
                   'subject', 'proposal', 'quotation', 'for', 'about', 'regarding', 'a', 'the'}
    for start in range(len(actual) - len(wanted) + 1):
        if all(_forms(word) & _forms(actual[start + i]) for i, word in enumerate(wanted)):
            return (all(word in safe_prefix or (word.startswith('ל') and word[1:] in safe_prefix)
                        for word in actual[:start])
                    and all(word.isdigit() for word in actual[start + len(wanted):]))
    return False


# An addressee/customer is a source-owned relationship; a supplier, email,
# letterhead or contact mention alone is not. A whole-line heading is allowed
# only when it exactly names the requested owner.
_OWNER_LINE = re.compile(r'^\s*(?:לכבוד\s*[:：]?|עבור\s*[:：]|לקוח(?:ה)?\s*[:：]|מזמין\s*[:：]|'
                         r'(?:customer|client|prepared\s+for|to)\s*[:：])\s*(.+?)\s*$', re.I)
_NON_TOPIC_LINE = re.compile(r'^\s*(?:לידי|ספק|טל(?:פון)?|דוא[״"\']?ל|'
                             r'contact|supplier|telephone|e-?mail)\s*[:：]', re.I)
_SUBJECT_LINE = re.compile(r'(?i)(?:^\s*(?:הנדון|נושא|subject|regarding)\s*[:：]|'
                           r'הצעה|הצעת\s+מחיר|להגיש\s+הצעת|proposal|quotation)')
_ORG_HEAD = re.compile(r'^(?:[בלו])?(?:אוניברסיט[הת]|מועצ[הת]|עיריי[הת]|'
                       r'מכלל[הת]|חבר[הת]|משרד|מרכז|university|council|college)$', re.I)
_ORG_END = frozenset({'עבור', 'לגבי', 'בנושא', 'על', 'של', 'עם', 'for', 'about', 'regarding'})


def _conflicting_org(request, text):
    words = tuple(word.casefold() for word in _WORD.findall(text))
    wanted = _terms(request.owner)
    for index, word in enumerate(words):
        if not _ORG_HEAD.fullmatch(word):
            continue
        end = index + 1
        while end < len(words) and words[end] not in _ORG_END and end - index < 5:
            end += 1
        observed = tuple(word for word in words[index:end] if word not in _STOP)
        if len(observed) != len(wanted):
            return True
        if not all(_forms(left) & _forms(right)
                   for left, right in zip(wanted, observed)):
            return True
    return False


def _unbound_location(request, text):
    """Reject an additional attached location whose identity is not bound.

    This is intentionally conservative: an unfamiliar ב-prefixed expression
    following the topic may be locative, so it cannot safely be discarded.
    """
    words = _terms(text)
    topic = _terms(request.topic)
    owner = _terms(request.owner)
    head = _forms(topic[0])
    starts = [i for i, word in enumerate(words) if head & _forms(word)]
    if not starts:
        return False
    for word in words[min(starts):]:
        if not (word.startswith('ב') and re.fullmatch(r'[א-ת]{4,}', word)):
            continue
        forms = _forms(word)
        if not any(forms & _forms(value) for value in topic + owner):
            return True
    return False


def original_matches_broad_material(request, text):
    """Return exact native witnesses, or None when identity/topic is unclear.

    Only the introduction is considered, and all topic terms must occur in a
    single line or adjacent two-line window. Owner markers that name a
    different addressee make the candidate ambiguous and are rejected.
    """
    if not isinstance(request, BroadMaterialRequest) or not isinstance(text, str):
        return None
    lines = [line.strip() for line in unicodedata.normalize('NFKC', text[:10000]).splitlines()
             if line.strip()][:25]
    owners = []
    for line in lines:
        match = _OWNER_LINE.match(line)
        if match:
            owners.append((line, match.group(1)))
        elif _terms(line) == _terms(request.owner):
            owners.append((line, line))
    if not owners or any(_terms(request.owner) != _terms(value) for _, value in owners):
        return None
    subjects = []
    for index, line in enumerate(lines):
        if _OWNER_LINE.match(line) or _NON_TOPIC_LINE.match(line):
            continue
        if not _SUBJECT_LINE.search(line):
            continue
        windows = [line]
        if index + 1 < len(lines) and not (_OWNER_LINE.match(lines[index + 1]) or
                                           _NON_TOPIC_LINE.match(lines[index + 1]) or
                                           _SUBJECT_LINE.search(lines[index + 1])):
            windows.append(line + ' ' + lines[index + 1])
        subjects.append(windows)
    if any(_conflicting_org(request, windows[0]) or _unbound_location(request, windows[0])
           for windows in subjects):
        return None
    # An explicit subject heading must at least refer to the requested topic's
    # head; an unrelated heading cannot borrow an incidental later mention.
    topic_head = _terms(request.topic)[0]
    for windows in subjects:
        if re.match(r'(?i)^\s*(?:הנדון|נושא|subject)\s*[:：]', windows[0]):
            if not any(_ordered_match(topic_head, window) for window in windows):
                return None
    for windows in subjects:
        for window in windows:
            if (not _conflicting_org(request, window) and not _unbound_location(request, window)
                    and _terminal_topic_match(request.topic, window)):
                return BroadMaterialMatch(owner_witness=owners[0][0], topic_witness=window)
    return None


def select_broad_material_excerpt(request, page_text, match, full_text):
    """Select an original contiguous introduction and first fully bounded row.

    ``page_text`` must be one independently verified source page. Unknown
    layouts, ownership, row boundaries and qualifications use the introduction
    only; no inferred or rearranged text is returned.
    """
    if not isinstance(request, BroadMaterialRequest) or not isinstance(match, BroadMaterialMatch):
        return ''
    intro = match.topic_witness
    if (not isinstance(page_text, str) or not isinstance(full_text, str)
            or not intro or intro not in page_text or page_text not in full_text):
        return ''
    # An item can be governed by a later document-wide condition. Do not
    # detach it from an explicit limitation that this narrow selector cannot
    # reliably bind to a row. The verified introduction remains usable.
    global_condition = re.compile(
        r'(?i)(?:כל\s+(?:הפריטים|פריט|הציוד|הרכיבים)|כל\s+המחירים|'
        r'ההצעה\s+(?:מותנית|כפופה|תלויה)|הפריטים\s+(?:מותנים|אופציונליים)|'
        r'ההצעה[^\n.]{0,100}(?:רק\s+לאחר\s+אישור|בכפוף\s+לאישור)|'
        r'כפופ(?:ה|ים|ות)?\s+ל|בהתאם\s+לאישור|בכפוף\s+לאישור|'
        r'(?:all|each)\s+(?:listed\s+)?(?:items?|products?|equipment)|'
        r'(?:items|products|equipment)\s+(?:are\s+)?optional|'
        r'(?:proposal|quotation|offer)\s+(?:is\s+)?subject\s+to|'
        r'only\s+(?:if|after)\s+(?:the\s+)?(?:client|customer)\s+approv)')
    if global_condition.search(full_text):
        return intro
    start = page_text.find(intro)
    intro_end = start + len(intro)
    attached = re.search(r'(\d{1,6})\s*[.]?\s*$', intro)
    if attached:
        number = attached.group(1)
        cursor = intro_end
    else:
        number_line = re.match(r'\s*[.]?\s*(\d{1,6})\s*[.]?(?=\s|$)', page_text[intro_end:])
        if not number_line:
            return intro
        number = number_line.group(1)
        cursor = intro_end + number_line.end()

    # A title independently binds this exact identifier to the requested topic.
    heading = re.search(r'(?m)^\s*(?:הנדון|נושא|subject)\s*[:：]\s*([^\n]+)$',
                        page_text[:start], re.I)
    if not heading or not _ordered_match(_terms(request.topic)[0], heading.group(1)):
        return intro
    if number not in re.findall(r'(?<!\d)\d{1,6}(?!\d)', heading.group(1)):
        return intro

    tail = page_text[cursor:]
    lines = list(re.finditer(r'(?m)^[^\n]*$', tail))
    nonempty = [line for line in lines if line.group().strip()]
    if not nonempty:
        return intro
    if nonempty[0].group().strip() in {'להלן ההצעה', 'להלן ההצעה:', 'the proposal follows', 'the proposal follows:'}:
        nonempty = nonempty[1:]
    if len(nonempty) < 5:
        return intro
    section_words = _terms(nonempty[0].group())
    head = _terms(request.topic)[0]
    if (len(section_words) not in {2, 3} or not (_forms(section_words[0]) & _forms(head))
            or section_words[-1] != number or
            (len(section_words) == 3 and section_words[1] not in {'גדולה', 'גדול', 'קטנה', 'קטן', 'large', 'small'})):
        return intro
    header = ' '.join(_terms(nonempty[1].group()))
    if not (all(word in header for word in ('סדורי', 'מוצר', 'כמות', 'מחיר')) or
            all(word in header for word in ('item', 'description', 'quantity', 'price'))):
        return intro
    if nonempty[2].group().strip() != '1':
        return intro
    second = next((i for i in range(3, len(nonempty)) if nonempty[i].group().strip() == '2'), None)
    if second is None or second < 5:
        return intro
    item = nonempty[3:second]
    numeric = re.fullmatch(r'1\s+[\d,]+(?:\.\d+)?(?:\s+[\d,]+(?:\.\d+)?)+',
                           item[-1].group().strip())
    if not numeric or not any(re.search(r'[A-Za-zא-ת]', line.group()) for line in item[:-1]):
        return intro
    # Detached conditions, peer sections or list markers cannot be attached to
    # the first item without a separate structural witness.
    unsafe = re.compile(r'(?i)(?:לא\s+כולל|אינ[הו]\s+כולל|למעט|בתנאי|אם\s|ייתכן|'
                        r'exclud|except|subject\s+to|conditional|^\s*\d+\s*$)')
    if any(unsafe.search(line.group()) for line in item[:-1]):
        return intro
    end = cursor + item[-1].end()
    # Keep the original subject heading with the selected item. It is part of
    # the same verified page and independently binds the quoted proposal and
    # its source-local section identifier; no text outside this span is cited.
    return page_text[heading.start():end].lstrip()


def exact_broad_material_candidates(evidence, question):
    """Propose exact, separately reviewable source heading and first item.

    These are candidates, not accepted answers. The ordinary review, original
    quote gate and final acceptance independently check each proposed string.
    """
    from document_reading import validate_summary_evidence
    if (getattr(getattr(evidence, 'reference', None), 'kind', '') != 'discovered_material'
            or not validate_summary_evidence(evidence, question)
            or len(evidence.passages) != 1):
        return ()
    passage = evidence.passages[0]
    if passage.citation_id not in (evidence.quote_eligible_ids or ()):
        return ()
    text = passage.source_claim
    heading = re.search(r'(?m)^[ \t]*(?:הנדון|נושא|subject)\s*[:：][^\n]+', text, re.I)
    row = re.search(r'(?m)^[ \t]*1[ \t]*\n[ \t\n]*(?P<item>[^\r\n]+)\n[ \t\n]*'
                    r'1[ \t]+[\d,]+(?:\.\d+)?(?:[ \t]+[\d,]+(?:\.\d+)?)+[ \t]*$', text)
    if not heading or not row:
        # A narrative source can still expose its exact proposal introduction;
        # the same source-wide review and final quote checks remain mandatory.
        literal = text.strip()
        if (len(literal) <= 300 and '\n' not in literal and '[' not in literal
                and ']' not in literal
                and re.search(r'(?i)הצעת\s+מחיר|quotation|\bproposal\b|\bquote\b', literal)):
            return ((literal, (passage.citation_id,)),)
        return ()
    values = (heading.group().strip(), row.group('item').strip())
    # An isolated product field is meaningful here only if the separately
    # reviewed heading preserves the document's proposed-work status.
    if not re.search(r'(?i)הצעת\s+מחיר|quotation|\bproposal\b|\bquote\b', values[0]):
        return ()
    if any(not value or len(value) > 300 or '[' in value or ']' in value
           or not re.search(r'[A-Za-zא-ת]', value) for value in values):
        return ()
    return tuple((value, (passage.citation_id,)) for value in values)
