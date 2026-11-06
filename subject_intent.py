"""Bounded subject requests for the shared source-reading contract.

Only the operation and the user's own subject words are interpreted here.
Source identity, evidence and authorization remain separate.
"""
from dataclasses import dataclass
import re
import unicodedata

@dataclass(frozen=True)
class SubjectRequest:
    kind: str
    subject: str

@dataclass(frozen=True)
class OrdinalWorkbookSection:
    subject: str
    document: str
    owner: str
    section: str
    ordinal: int


def ordinal_workbook_section_subject(subject):
    """Parse an explicit document, owner, section and ordinal as one bound request.

    This does not establish that the source exists, or that its structure
    contains multiple matching sections. Those are source-owned decisions.
    """
    value = unicodedata.normalize('NFKC', str(subject or '')).strip(' ?!.;,')
    match = re.fullmatch(
        r'(?P<document>[א-תA-Za-z][א-תA-Za-z\s]{2,70}?\s+\d{2,12})\s+'
        r'(?:ל|של\s+)(?P<owner>[א-תA-Za-z][א-תA-Za-z\s]{2,80}?)\s+(?:עבור|לגבי)\s+'
        r'(?P<section>[א-תA-Za-z0-9][א-תA-Za-z0-9\s-]{2,80}?)\s+'
        r'ב(?:תצורה|חלופה|גרסה)\s+ה(?P<ordinal>ראשונה|ראשון|שנייה|שני|שלישית|שלישי)',
        value, re.I)
    if not match:
        return None
    ordinal = {'ראשונה': 1, 'ראשון': 1, 'שנייה': 2, 'שני': 2,
               'שלישית': 3, 'שלישי': 3}[match.group('ordinal')]
    return OrdinalWorkbookSection(value, match.group('document').strip(),
        match.group('owner').strip(), match.group('section').strip(), ordinal)


def ordinal_workbook_section_request(query):
    value = unicodedata.normalize('NFKC', str(query or '')).strip(' ?!.;,')
    match = re.fullmatch(
        r'(?:מה\s+עיקרי|(?:תן|תני|תנו)\s+(?:לי\s+)?סקירה(?:\s+כללית)?\s+של)\s+(?P<subject>.+)',
        value, re.I)
    return ordinal_workbook_section_subject(match.group('subject')) if match else None

_DOCUMENT = r'(?:ה?(?:מסמך|מסמכים|קובץ|קבצים)|documents?|files?|records?)'
_ABOUT = r'(?:על|לגבי|אודות|about|concerning|regarding)'
_FIND = r'(?:תן|תני|תנו|תביא|תביאי|תביאו|תיתן|תתני|הצג|הציגי|הראו|הראה|מצא|מצאי|מצאו|show|find|locate|give|bring|list)'
_ACTION = (rf'(?:(?:(?:could|can|would|will)\s+you\s+)?{_FIND}|'
           rf'(?:i|we)\s+(?:would\s+like|want|need)(?:\s+to\s+{_FIND})?|'
           r'(?:אני|אנחנו)\s+(?:רוצה|רוצים|רוצות|מבקש|מבקשת)(?:\s+(?:לקבל|לראות|למצוא|להציג))?|'
           r'אפשר\s+(?:לקבל|לראות|למצוא|להציג))')
_PREFIX = r'(?:לי\s+|me\s+)?(?:את\s+|the\s+)?'
_COUNT = r'(?:\d+|one|two|three|four|five|six|seven|eight|אחד|אחת|שני|שתי|שניים|שתיים|שלושה|שלוש|ארבעה|ארבע|חמישה|חמש|שישה|שש|שבעה|שבע|שמונה)'
# Commands are recognized compositionally, not by any particular named source.
_DOC_REQUEST = re.compile(
    rf'^(?:please\s+|נא\s+)?{_ACTION}\s+{_PREFIX}(?:{_COUNT}\s+)?{_DOCUMENT}'
    rf'(?:\s+(?:או|and|or)\s+{_DOCUMENT})?\s+{_ABOUT}\s+(?P<subject>.+)$', re.I)
# Questions asking which document bears a relationship to a topic retain
# the entire literal topic tail, just like the imperative discovery route.
_DOC_RELATION_REQUEST = re.compile(
    rf'^(?:איזה|איזו|אילו|which|what)\s+{_DOCUMENT}\s+'
    r'(?:(?:ש|שה)?מתייחס(?:ת|ים|ות)?\s*ל|'
    r'(?:ש|שה)?קשור(?:ה|ים|ות)?\s*ל|'
    r'(?:relates?|refers?)\s+to\s+|(?:is|are)\s+related\s+to\s+)'
    r'(?P<subject>.+)$', re.I)
_WORK_REQUESTS = (
    re.compile(r'^(?:מה\s+(?:הוצע|מוצע|תוכנן|מתוכנן)|what\s+(?:was\s+proposed|is\s+proposed|was\s+planned))\s+(?:לגבי|אודות|על|about|regarding|for)\s+(?P<subject>.+)$',re.I),
    re.compile(r'^(?:מה\s+(?:היה|קרה|נעשה)|what\s+(?:happened|was\s+done))\s+(?:ב|in\s+|at\s+)(?P<subject>.+)$',re.I),
    re.compile(r'^(?:איזה|אילו|what|which)\s+(?:עבודות|עבודה|טיפול|work|works|services?)\s+(?:(?:היה|היו|נעשה|נעשו|was|were|was\s+done)\s+)?(?:ב|in\s+|at\s+)(?P<subject>.+)$',re.I),
)

def client_work_request(query):
    """Recognize an unqualified request for documented work at a named owner."""
    value = unicodedata.normalize('NFKC', str(query or '')).strip(' ?!.;,')
    match = re.fullmatch(
        r'(?:(?:מה\s+(?:כתוב|נכתב|מתואר|מופיע|ידוע|יש)|'
        r'מה\s+(?:אפשר|ניתן)\s+ללמוד)\s+(?:על|לגבי|אודות)|'
        r'(?:ספר|ספרי|ספרו)\s+(?:לי\s+)?(?:על|לגבי|אודות)|'
        r'(?:תן|תני|תנו)\s+(?:לי\s+)?סקירה(?:\s+כללית)?\s+(?:על|לגבי|אודות|של))'
        r'\s+(?:ה?עבודה|עבודות|ה?שירות)\s+(?:ב|עבור\s+)(?P<owner>.+)',
        value, re.I)
    if (not match or _has_unrepresented_constraints(value)
            or _overview_has_unrepresented_restriction(value, allow_framing=True)):
        return None
    owner = match.group('owner').strip()
    # A free-form trailing condition or fact is not a client name. Keep the
    # complete owner literal and leave qualified requests to their own route.
    if (len(owner) > 100 or not re.fullmatch(r'[A-Za-zא-ת][A-Za-zא-ת\s\-\"׳״\']*', owner)
            or re.search(r'(?<!\w)(?:כל|כולן|כולם|רשימה|בלבד|ללא|בלי|של|עם|לגבי|בנושא|בתחום|עבור|מספר|טלפון|נייד|מחיר|עלות|תשלום|ציוד|מתי|כמה|למה|איפה|כיצד|לפי|ליד|לפני|אחרי|בשנה|בחודש|בתאריך|אשר)(?!\w)', owner)
            or re.search(r'(?<!\w)ש(?:ה|בוצע|בוצעה|בוצעו|תוכנן|תוכננה|הוצע|הוצעה|מופיע|מופיעה|מתואר|מתוארת)[א-ת]*(?!\w)', owner)):
        return None
    from technical_validation import derive_requested_slots
    if derive_requested_slots(value):
        return None
    return SubjectRequest('subject_work', owner)

# Require a broad, explicitly selective framing and keep the complete
# literal source-independent subject. Counted, exhaustive and qualified
# enumerations do not match this grammar.
_MATERIAL_THEME_PATTERNS = (
    r'(?:אילו|מהם|מהן)\s+(?:ה)?(?:נושאים|היבטים|סוגיות|תחומים)'
    r'(?:\s+ו(?:נושאים|היבטים|סוגיות|תחומים|תפקידים|פונקציות))?'
    r'\s+(?:ה)?(?:עיקריים|מרכזיים)\s+'
    r'(?:מתוארים|מופיעים|נדונים|נזכרים)\s+ב(?:חומר|מידע)\s+'
    r'(?:של|על|לגבי|אודות)\s+(?P<subject>.+)',
    r'(?:what|which)\s+(?:are\s+)?(?:the\s+)?(?:main|key|major)\s+'
    r'(?:themes|topics|aspects)(?:\s+and\s+(?:themes|topics|aspects|functions))?'
    r'\s+(?:are\s+)?(?:described|discussed|covered)\s+in\s+'
    r'(?:the\s+)?(?:material|content)\s+(?:about|concerning|for|of)\s+'
    r'(?P<subject>.+)',
)


def _material_theme_request(query):
    value = unicodedata.normalize('NFKC', str(query or '')).strip(' ?!.;,')
    return next((match for pattern in _MATERIAL_THEME_PATTERNS
                 if (match := re.fullmatch(pattern, value, re.I))), None)


def information_subject(query):
    """Extract a requested topic without treating request grammar as evidence."""
    value = unicodedata.normalize('NFKC', str(query or '')).strip().strip(' ?!.')
    patterns = (
        # An unquantified request for principal themes in material about a
        # project is a selective overview rather than a fixed-size list.
        *_MATERIAL_THEME_PATTERNS,
        r'(?:(?:באופן\s+כללי|בקצרה)\s*,?\s*)?במה\s+עוסק(?:ת)?\s+(?:החומר|המידע)\s+(?:על|לגבי|אודות)\s+(?P<subject>.+)',
        r'(?:מה\s+(?:כתוב|נכתב|מופיע|מתואר|ידוע)\s+(?:על|לגבי|אודות)|'
        r'מה\s+(?:כתוב|נכתב|מופיע|מתואר)\s+במסמכים\s+(?:על|לגבי|אודות)|'
        r'מה\s+(?:אפשר|ניתן)\s+ללמוד\s+(?:על|לגבי|אודות)|'
        r'(?:תן|תני|תנו)\s+(?:לי\s+)?סקירה(?:\s+כללית)?\s+(?:על|לגבי|אודות|של)|'
        r'(?:ספר|ספרי|ספרו)\s+(?:לי\s+)?(?:על|לגבי|אודות)|'
        r'what\s+(?:is\s+(?:written|stated|said|known)|do\s+(?:the\s+)?'
        r'(?:sources|documents)\s+say)\s+(?:about|concerning|regarding)|'
        r'(?:tell\s+me|explain)\s+about)\s+(?P<subject>.+)',
        r'(?:מה\s+כולל|(?:תאר|תארי|תארו|הסבר|הסבירי|הסבירו)\s+(?:את\s+)?|'
        r'(?:describe|explain)\s+)(?P<subject>.+)',
    )
    for pattern in patterns:
        match = re.fullmatch(pattern, value, re.I)
        if match:
            return match.group('subject').strip(' ?!.;,')
    return ''


def project_location_anchor(subject):
    """Return the literal location of a project for candidate retrieval only.

    The caller must retain the complete subject and independently validate
    that a candidate answers it. A location match is not project evidence.
    """
    value = unicodedata.normalize('NFKC', str(subject or '')).strip(' ?!.;,')
    match = re.fullmatch(
        r'(?:ה?פרויקט\s+(?:ב(?=(?:בית|מרכז|מוקד|אתר|מועצה|קמפוס)\b)|'
        r'ב\s+|של\s+)?|(?:the\s+)?project\s+(?:(?:at|in|for|of)\s+)?)'
        r'(?P<location>.+)', value, re.I)
    if not match:
        return ''
    location = match.group('location').strip(' ?!.;,')
    return location if 2 <= len(location) <= 80 else ''


def project_location_candidates(subject):
    """Propose literal and possible Hebrew locative owners without deciding identity.

    An attached Hebrew ב after a definite project noun can be a preposition
    or part of a name. Both readings must remain candidates until source
    identity resolves them; callers must not merge competing owners.
    """
    value = unicodedata.normalize('NFKC', str(subject or '')).strip(' ?!.;,')
    match = re.fullmatch(r'הפרויקט\s+(?P<owner>ב[^\s]+(?:\s+.*)?)', value)
    anchor = project_location_anchor(value)
    if not match:
        return (anchor,) if anchor else ()
    literal = match.group('owner').strip(' ?!.;,')
    locative = literal[1:].strip()
    return tuple(dict.fromkeys(item for item in (literal, locative) if len(item) >= 2))


def project_information_intent(query, subject):
    """Recognize an unrestricted project-information action and its anchor.

    The anchor proposes candidates only. The complete request subject stays
    authoritative, and caller-owned source scope still governs access.
    """
    requested = information_subject(query)
    if not requested:
        return None
    # A number immediately after the project noun can be its literal name.
    # Mask only this syntactically delimited identifier while checking for
    # constraints; the original subject remains authoritative downstream.
    # Other numbers, dates, amounts and counts are never masked.
    numeric_name = re.match(
        r'^(?:ה?פרויקט|(?:the\s+)?project)\s+(?P<name>\d{2,6})'
        r'(?=$|\s+(?:ב\S*|של|at|in|for|of)\s*)', requested, re.I)
    constraint_query = unicodedata.normalize('NFKC', str(query or ''))
    if numeric_name:
        start = constraint_query.find(requested)
        if start >= 0:
            a, b = numeric_name.span('name')
            constraint_query = constraint_query[:start+a] + 'IDENTIFIER' + constraint_query[start+b:]
    if _has_unrepresented_constraints(constraint_query):
        return None
    # A named project may be followed by a restriction on what is asked.
    # Leave these questions to the normal targeted-question path.
    if re.search(r"(?<!\w)(?:לגבי|בנוגע|בהקשר|מבחינת|אודות|שכולל|שכוללת|שכוללים|שכוללות|שיש|עם|about|regarding|concerning|specifically|with|containing)(?!\w)", requested, re.I):
        return None
    if unicodedata.normalize('NFKC', str(subject or '')).strip(' ?!.;,').casefold() != requested.casefold():
        return None
    anchor = project_location_anchor(requested)
    # A general question about material on an exact subject can use the same
    # scoped source search even without adding an invented project noun.
    material_about = bool(re.fullmatch(
        r'(?:(?:באופן\s+כללי|בקצרה)\s*,?\s*)?במה\s+עוסק(?:ת)?\s+'
        r'(?:החומר|המידע)\s+(?:על|לגבי|אודות)\s+.+',
        unicodedata.normalize('NFKC', str(query or '')).strip(' ?!.;,'), re.I))
    return SubjectRequest('project_information', requested) if anchor or material_about else None


def overview_interpretation_candidate(query):
    """Propose semantic interpretation for broad work/planning language.

    This is only a planner trigger, never an overview decision. It does not
    decide based on a project name, source type, or a previous answer.
    """
    value = unicodedata.normalize('NFKC', str(query or ''))
    if _overview_has_unrepresented_restriction(value):
        return False
    return bool(re.search(
        r'(?i)(?<!\w)(?:[והבלמכ]*)(?:עיקר\w*|תכנו\w*|עבוד\w*|'
        r'סקיר\w*|פרט\w*|תיאו\w*|overview|main|planning|works?)(?!\w)', value))


def _overview_has_unrepresented_restriction(value, *, allow_framing=False):
    # An unfiltered overview cannot carry exclusions or source references by
    # merely copying a subject suffix. Explicitly parsed general-overview
    # grammar may use words such as "regarding" as introductory framing.
    value = str(value or '')
    if re.search(
        r'(?i)\b(?:only|solely|except|excluding|exclude|without)\b|'
        r'(?<!\w)(?:רק|בלבד|למעט|מלבד|חוץ|ללא|בלי|בתחום)(?!\w)|'
        r'(?<!\w)(?:מהמקור|במקור|למקור|אותו\s+מקור|המקור\s+(?:ההוא|הזה|הנבחר))(?!\w)|'
        r'\b(?:from|in|of)\s+(?:(?:the\s+)?(?:same|previous|selected|that|this))\s+(?:source|document|file)\b', value):
        return True
    return bool(not allow_framing and re.search(
        r'(?i)\b(?:regarding|concerning)\b|(?<!\w)(?:לגבי|בנוגע|מבחינת)(?!\w)', value))


def validate_project_overview_subject(query, subject):
    """Bind an accepted overview subject to the complete current request.

    A planner may propose an action but may not add a source or drop a
    recognizable property or restriction. The named subject must be an
    exact suffix, so trailing qualifiers cannot silently disappear.
    """
    value = unicodedata.normalize('NFKC', str(query or '')).strip(' ?!.;,')
    subject = unicodedata.normalize('NFKC', str(subject or '')).strip(' ?!.;,')
    if (len(subject) < 2 or len(subject) > 160 or
            not value.casefold().endswith(subject.casefold()) or
            not re.search(r'[A-Za-z\u05d0-\u05ea]', subject)):
        return False
    scoped = ordinal_workbook_section_request(value)
    if scoped:
        return subject.casefold() == scoped.subject.casefold()
    deterministic = bool(project_information_intent(value, subject))
    if not deterministic:
        # A copied suffix alone may be only the last word of a multiword
        # owner. Require the whole owner introduced by a source-independent
        # relation, rather than permitting a planner to shorten it.
        owner = re.search(r'(?i)(?<!\w)(?:עבור|for|at|in)\s+(?P<owner>.+)$', value)
        if owner:
            owners = (owner.group('owner').strip(' ?!.;,'),)
        else:
            # Hebrew commonly attaches the locative preposition ב to the
            # site name. Preserve the full owner whether the copied span
            # includes the prefix or starts immediately after it.
            attached = re.search(r'(?<!\w)(?P<locative>ב(?P<owner>[א-ת]{2,}(?:\s+[^?!.;,]+)*))$', value)
            owners = ((attached.group('locative'), attached.group('owner')) if attached else ())
        if not any(subject.casefold() == item.casefold() for item in owners):
            return False
    if ((not deterministic and _has_unrepresented_constraints(value)) or
            _overview_has_unrepresented_restriction(value, allow_framing=deterministic)):
        return False
    from technical_validation import derive_requested_slots
    # The optional word for functions belongs to the general overview
    # framing; preserve any technical slot named in the actual subject.
    # This narrow grammar never matches a count or an exhaustive request.
    material_themes = _material_theme_request(value)
    slot_query = subject if material_themes else value
    if derive_requested_slots(slot_query):
        return False
    from financial_documents import explicit_document_request
    if explicit_document_request(value):
        return False
    return bool(deterministic or overview_interpretation_candidate(value))


def _has_unrepresented_constraints(query, *, allow_monetary=False):
    """Do not turn a selective request into an unrestricted subject scan."""
    from financial_semantics import monetary_question, financial_query
    from financial_documents import explicit_document_request
    value = str(query or '')
    from request_understanding import explicit_requested_items
    count = explicit_requested_items(value)
    counted = re.sub(rf'(?<!\w){_COUNT}(?=\s+{_DOCUMENT}(?!\w))', '', value, count=1, flags=re.I) if count and 1 <= count <= 8 else value
    return bool((not allow_monetary and monetary_question(value)) or financial_query(value)
        or explicit_document_request(value)
        or re.search(r'(?<!\w)(?:רק|בלבד|only|exclusively)(?!\w)', value, re.I)
        or re.search(r'\.(?:pdf|docx?|pptx?|xlsx?|csv|txt|md)(?!\w)',value,re.I)
        or re.search(r'(?<!\w)(?:במסמך|מהמסמך|בקובץ|מהקובץ)(?!\w)', value)
        or re.search(r'\b(?:in|from|within)\s+(?:(?:this|that|selected)|the\s+same)\s+(?:document|file|source)\b', value,re.I)
        or re.search(r'(?<!\d)(?:19|20)\d{2}(?!\d)', value)
        or re.search(r'(?<!\d)\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?(?!\d)',value)
        or re.search(r'(?<!\w)(?:ינואר|פברואר|מרץ|אפריל|מאי|יוני|יולי|אוגוסט|ספטמבר|אוקטובר|נובמבר|דצמבר|january|february|march|april|may|june|july|august|september|october|november|december)(?!\w)', value, re.I)
        or re.search(r'(?<!\w)\d+(?!\w)',counted)
        or (count is not None and not 1 <= count <= 8))

def subject_request(query, *, allow_monetary=False):
    value = unicodedata.normalize('NFKC', str(query or '')).strip().strip('?! .')
    # Explicit identities and explicit document restrictions retain their
    # existing resolution/authorization path.
    from financial_documents import explicit_document_request
    if _has_unrepresented_constraints(value, allow_monetary=allow_monetary):
        return None
    client_work = client_work_request(value)
    if client_work:
        return client_work
    match = _DOC_REQUEST.fullmatch(value) or _DOC_RELATION_REQUEST.fullmatch(value)
    if match:
        kind = 'subject_documents'
    else:
        match = next((m for p in _WORK_REQUESTS if (m := p.fullmatch(value))), None)
        kind = 'subject_work'
    if not match:
        return None
    subject = match.group('subject').strip(' ?!.;,')
    if not subject or len(subject)>160 or not re.search(r'[A-Za-z\u05d0-\u05ea]', subject):
        return None
    if re.search(r'(?<!\w)(?:רק|only)(?!\w)', subject, re.I):
        return None
    return SubjectRequest(kind, subject)

_DEICTIC_WORK_HEAD = r'(?:fault|repair|work|service|installation|job|ה?תקלה|ה?תיקון|ה?עבודה|ה?טיפול|ה?שירות|ה?התקנה)'
_DEICTIC_WORK = re.compile(
    rf'^(?:(?:this|that|the\s+same)\s+{_DEICTIC_WORK_HEAD}|'
    rf'{_DEICTIC_WORK_HEAD}\s+(?:הזה|הזאת|הזו|ההיא|ההוא)|'
    rf'(?:אותה|אותו)\s+{_DEICTIC_WORK_HEAD})$', re.I)
_WORK_TERM = re.compile(
    r'\b(?:fault|repair|work|service|installation|job|technician|maintenance)\b|'
    r'(?<!\w)(?:ה|ל|ב|מ|ו)?(?:תקלה|תיקון|עבודה|טיפול|שירות|התקנה|טכנאי|טכנאית)(?!\w)',
    re.I)


def is_deictic_work_subject(subject):
    return bool(_DEICTIC_WORK.fullmatch(unicodedata.normalize('NFKC', str(subject or '')).strip()))


def _work_family(word):
    """Interpret only closed work nouns; relationships are never inferred."""
    value = unicodedata.normalize('NFKC', str(word or '')).casefold()
    if re.fullmatch(r'fault|(?:ה|ל|ב|מ|ו)?תקלה', value):
        return 'fault'
    if re.fullmatch(r'repair|(?:ה|ל|ב|מ|ו)?תיקון', value):
        return 'repair'
    if re.fullmatch(r'installation|(?:ה|ל|ב|מ|ו)?התקנה', value):
        return 'installation'
    if re.fullmatch(r'work|service|job|technician|maintenance|'
                    r'(?:ה|ל|ב|מ|ו)?(?:עבודה|טיפול|שירות|טכנאי|טכנאית)', value):
        return 'general'
    return ''


def _reference_head(subject):
    head = re.sub(r'^(?:this|that|the\s+same|אותה|אותו)\s+', '',
                  subject, flags=re.I)
    return re.sub(r'\s+(?:הזה|הזאת|הזו|ההיא|ההוא)$', '', head, flags=re.I)


def resolve_document_subject_reference(current, previous):
    """Resolve a closed work reference from one prior user request only.

    No assistant response, source body, model output, or new factual value is
    used. Ambiguous or unrepresentable prior owners fail closed.
    """
    from request_understanding import normalize_request_text
    from financial_semantics import monetary_question
    current = normalize_request_text(current)
    previous = normalize_request_text(previous)
    if not current or not previous or len(current) > 1200 or len(previous) > 1200:
        return None
    spec = subject_request(current)
    if not spec or spec.kind != 'subject_documents' or not is_deictic_work_subject(spec.subject):
        return None
    # Several work subjects cannot be collapsed into one implicit referent.
    if re.search(r'\b(?:and|or|versus|compare|between)\b|(?<!\w)(?:או|לעומת|השוואה|בין|ו)(?!\w)|'
                 r'(?<!\w)ו(?:ב|ל)?(?:מוקד|מרכז|אתר|פרויקט|מועצה)(?=\s)|[,;]',
                 previous, re.I):
        return None
    head = _reference_head(spec.subject)
    requested_family = _work_family(head)
    if not requested_family:
        return None
    prior = subject_request(previous)
    if prior:
        owner = prior.subject
    else:
        monetary = monetary_question(previous)
        if not monetary or monetary.explicit_source is not None:
            return None
        owner = monetary.description
    owner = normalize_request_text(owner).strip(' ?!.;,')
    if not owner or len(owner) > 160 or is_deictic_work_subject(owner):
        return None
    if not _WORK_TERM.search(owner):
        # A broad, user-named location and the current user's literal work
        # head can form a search subject without asserting an event or using
        # assistant prose. Multiple possible documents remain separate.
        broad = bool(prior and prior.kind == 'subject_work' and re.match(
            r'^(?:מה\s+(?:היה|קרה|נעשה)|what\s+(?:happened|was\s+done)|'
            r'(?:איזה|אילו|what|which)\s+(?:עבודות|עבודה|טיפול|work|works|services?))',
            previous, re.I))
        if not broad:
            return None
        owner = normalize_request_text(head + ' ' + owner)
    prior_families = {_work_family(match.group()) for match in _WORK_TERM.finditer(owner)}
    prior_families.discard('')
    # A broad reference ("that service") may name an identified operation.
    # Specific fault/repair/installation heads cannot replace one another:
    # the current head must have appeared in the prior user-owned subject.
    if requested_family != 'general' and requested_family not in prior_families:
        return None
    words = re.findall(r'[A-Za-z0-9\u05d0-\u05ea]+', owner)
    # Require a specific work and a distinct identifying anchor, not only
    # "the repair" or another open-ended descriptive phrase.
    if len(words) < 2 or re.search(r'\b(?:and|or)\b|(?<!\w)או(?!\w)', owner, re.I):
        return None
    from monetary_questions import _GRAMMATICAL_HEADS, _TECHNICAL_LABELS, _variants
    generic = (_GRAMMATICAL_HEADS | _TECHNICAL_LABELS | {
        'fault', 'repair', 'work', 'service', 'installation', 'job', 'technician',
        'maintenance', 'station', 'project', 'site', 'center', 'centre',
        'the', 'a', 'an', 'for', 'of', 'at', 'in', 'on', 'של', 'על', 'את',
    })
    anchors = [word for word in words if not (_variants(word.casefold()) & generic)]
    if not anchors:
        return None
    # A second name-like Hebrew word beginning in ו can be either a
    # conjunction or part of a proper name. Do not remove the letter or
    # silently combine owners; ask for clarification instead.
    if len(anchors) > 1 and any(re.fullmatch(r'ו[\u05d0-\u05ea]{2,}', word)
                                for word in anchors[1:]):
        return None
    start = current.rfind(spec.subject)
    if start < 0 or current.find(spec.subject) != start:
        return None
    resolved_query = current[:start] + owner + current[start + len(spec.subject):]
    resolved = subject_request(resolved_query)
    if not resolved or resolved.kind != 'subject_documents' or resolved.subject != owner:
        return None
    return resolved_query, owner


@dataclass(frozen=True)
class SubjectWorkSlots:
    """Request-owned content groups and an optional explicit location."""
    content: tuple[str, ...]
    owner: str


def subject_work_slots(query, subject):
    """Split only supported, explicit grammatical boundaries; never infer facts.

    Broad work questions use the whole subject as their owner. Descriptive
    questions may supply coordinated content and an explicitly introduced
    location. Unsupported or ambiguous conjunctions fail closed.
    """
    value = unicodedata.normalize('NFKC', str(subject or '')).strip(' ?!.;,')
    if not value or len(value) > 160:
        return None
    from monetary_questions import _variants
    # Hebrew prefixes are admitted only for generic location heads, never
    # by stripping an arbitrary initial letter from a proper name.
    boundaries = list(re.finditer(
        r'(?<!\w)(?:at|in|for)\s+|'
        r'(?<!\w)ב(?=(?:פרויקט|מוקד|מרכז|אתר|מועצה|תל)(?:\s|$))',
        value, re.I))
    if len(boundaries) > 1:
        return None
    if boundaries:
        boundary = boundaries[0]
        content_text = value[:boundary.start()].strip(' ,')
        owner = value[boundary.end():].strip(' ,')
        if not content_text or not owner:
            return None
    else:
        proposal_question = bool(re.match(
            r'^(?:מה\s+(?:הוצע|מוצע|תוכנן|מתוכנן)|what\s+(?:was\s+proposed|is\s+proposed|was\s+planned))',
            str(query or '').strip(), re.I))
        if not proposal_question:
            return SubjectWorkSlots((), value)
        content_text, owner = value, ''

    # Only a lexical conjunction with a recognized common noun can split
    # Hebrew clitics. Unknown leading ו could be part of a proper name.
    groups, current = [], []
    for word in re.findall(r'[A-Za-z0-9\u05d0-\u05ea]+', content_text):
        if word.casefold() == 'and' or word == 'ו':
            if not current:
                return None
            groups.append(' '.join(current)); current = []
        elif word.startswith('ו') and current:
            if len(word) <= 2 or word[1:] not in _variants(word):
                return None
            groups.append(' '.join(current)); current = [word]
        else:
            current.append(word)
    if not current:
        return None
    groups.append(' '.join(current))
    return SubjectWorkSlots(tuple(groups), owner)


def _independently_delimited_subject(query):
    """Find the whole topical tail; never let a model discard its qualifiers."""
    value = unicodedata.normalize('NFKC', str(query or '')).strip().strip(' ?!.;,')
    direct = subject_request(value)
    if direct:
        return direct.subject
    # A relation word delimits the complete user-owned topic. Hebrew ל may be
    # attached to its first noun; only that grammatical prefix is excluded.
    connector = re.compile(
        r'(?<!\w)(?:about|concerning|regarding|for|discuss(?:es|ing)?|'
        r'relevant\s+to|related\s+to|על|לגבי|אודות|עבור|בגין)\s+'
        r'|(?<!\w)(?:ש?מתייחס(?:ים|ות)?|(?:ה|ש|שה)?קשור(?:ים|ות)?|בנוגע)\s*ל',
        re.I)
    matches = list(connector.finditer(value))
    if len(matches) != 1:
        return ''
    tail = value[matches[0].end():].strip(' ?!.;,')
    return tail if tail and len(tail) <= 160 else ''


def validated_subject_proposal(query, kind, subject, *, allow_monetary=False):
    """Validate a complete literal subject against user-owned grammar."""
    if kind not in {'subject_documents','subject_work'} or not isinstance(subject,str):
        return None
    value = unicodedata.normalize('NFKC',str(query or '')).strip()
    direct = subject_request(value, allow_monetary=allow_monetary)
    if direct and direct.kind != kind:
        return None
    # Explicit discovery wording controls the requested result even when the
    # planner proposes a structurally valid but incompatible operation.
    document_noun_request = re.match(
        rf'^(?:(?:please|נא)\s+)?(?:{_DOCUMENT}\b|'
        rf'(?:what|which|אילו|איזה)\s+{_DOCUMENT}\b|'
        rf'{_FIND}\s+{_PREFIX}(?:{_COUNT}\s+)?{_DOCUMENT}\b)',
        value, re.I)
    if document_noun_request and kind != 'subject_documents':
        return None
    work_noun_request = re.match(
        r'^(?:(?:the|ה)\s+)?(?:(?:what|which|איזה|אילו)\s+)?'
        r'(?:work|works|services?|ה?עבודות|ה?עבודה|ה?שירותים)\b'
        r'\s+(?:about|concerning|regarding|related\s+to|על|לגבי|אודות)(?!\w)',
        value, re.I)
    if work_noun_request and kind != 'subject_work':
        return None
    subject = unicodedata.normalize('NFKC',subject).strip(' ?!.;,')
    if not subject or len(subject)>160 or subject.casefold() not in value.casefold():
        return None
    delimited = _independently_delimited_subject(value)
    # A general work/history request may introduce the owner through a
    # locative attached to the work noun.  The planner still owns the
    # operation; this check merely ensures it copied the complete owner and
    # did not discard a specific kind of work or another qualifier.
    owner_only = False
    if kind == 'subject_work' and not direct:
        plain = value.strip(' ?!.;,')
        work_owner = re.search(
            r'(?i)(?<!\w)(?:ה?עבודה|עבודות|ה?שירות|work|works|services?)'
            r'(?:\s+(?:ה?מתועד(?:ת|ות|ים)?|ה?מתואר(?:ת|ות|ים)?|'
            r'documented|recorded|described))?'
            r'\s+(?:ב|(?:at|in|for)\s+)(?P<owner>.+)$', plain)
        if work_owner:
            # Only framing words may precede a general work noun. A concrete
            # work type or modifier must remain part of the selected subject.
            framing = {'מה','כתוב','מופיע','נכתב','יש','על','לגבי','אודות',
                       'היסטוריה','תיעוד','סיכום','פירוט','מידע','של',
                       'what','is','written','about','history','summary',
                       'information','of','the','סכם','סכמי','סכמו','את','נא',
                       'summarize','summarise','documented','recorded','described'}
            prefix = re.findall(r'[A-Za-z\u05d0-\u05ea]+', plain[:work_owner.start()].casefold())
            owner = work_owner.group('owner').strip(' ?!.;,')
            locative = plain[work_owner.start('owner') - 1:].strip(' ?!.;,')
            variants = (owner, locative, work_owner.group().strip(' ?!.;,'))
            owner_only = (all(word in framing for word in prefix) and
                          any(subject.casefold() == item.casefold() for item in variants))
            if owner_only:
                # All three variants identify the same literal owner; the
                # general work noun remains represented by subject_work.
                subject = owner
    if not owner_only and (not delimited or subject.casefold() != delimited.casefold()):
        return None
    if re.search(r'(?<!\w)(?:זה|זאת|הזה|הזאת|אותו|אותה|this|that|it)(?!\w)',
                 subject, re.I):
        return None
    if _has_unrepresented_constraints(value, allow_monetary=allow_monetary):
        return None
    if not re.search(r'[A-Za-z\u05d0-\u05ea]',subject):
        return None
    return SubjectRequest(kind,subject)
