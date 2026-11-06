"""Question-owned display choices over complete authenticated money evidence.

This module never changes source text, passages or quote verification. Retained
context is evidence; only requested fields and material price conditions belong
in the ordinary displayed answer.
"""
import re


_FIELDS = {
    'quantity': r'\b(?:quantity|how\s+many\s+units?|number\s+of\s+units?)\b|(?<!\w)כמות(?!\w)',
    'issue_date': r'\b(?:(?:document|issue|quotation|quote|invoice)\s+date|date\s+(?:of\s+)?(?:the\s+)?(?:document|quotation|quote|invoice))\b|תאריך\s+(?:ה?(?:מסמך|הצעה|חשבונית)|הנפקה)',
    'supplier_phone': r'\b(?:(?:supplier|vendor|provider)[\s\x27’]*(?:s\s+)?(?:phone|telephone)(?:\s+number)?|(?:phone|telephone)(?:\s+number)?\s+(?:of|for)\s+(?:the\s+)?(?:supplier|vendor|provider))\b|(?:מספר\s+)?(?:ה?טלפון)\s+(?:של\s+)?ה?ספק',
    'supplier_email': r'\b(?:(?:supplier|vendor|provider)[\s\x27’]*(?:s\s+)?(?:e-?mail)(?:\s+address)?|(?:e-?mail)(?:\s+address)?\s+(?:of|for)\s+(?:the\s+)?(?:supplier|vendor|provider))\b|(?:כתובת\s+)?(?:ה?דוא[״\"]?ל|ה?אימייל)\s+(?:של\s+)?ה?ספק',
    'supplier_website': r'\b(?:(?:supplier|vendor|provider)[\s\x27’]*(?:s\s+)?(?:website|web\s+address)|(?:website|web\s+address)\s+(?:of|for)\s+(?:the\s+)?(?:supplier|vendor|provider))\b|(?:אתר|כתובת\s+האתר)\s+(?:של\s+)?ה?ספק',
}
_FIELD_PATTERNS = {key: re.compile(pattern, re.I) for key, pattern in _FIELDS.items()}


def requested_monetary_fields(query):
    return tuple(key for key, pattern in _FIELD_PATTERNS.items() if pattern.search(query))


def without_requested_fields(query):
    """Remove recognized output fields, never service or project identity words."""
    for pattern in _FIELD_PATTERNS.values():
        query = pattern.sub(' ', query)
    return query


_SUBJECT = re.compile(r'\b(?:price|amount|cost|rate|fee|quote|quotation|estimate|offer|payment|discount|charge|work|service|installation|transport|delivery|quantity|unit|vat|tax|repair|inspection|approval|authorization|cabling|purchase|warranty|guarantee)\b|מחיר|סכום|עלות|תשלום|מע\s*[״\"]?\s*מ|הנחה|הצעה|עבודה|שירות|התקנה|הובלה|אספקה|כמות|יחידה|תיקון|בדיקה|אישור|טיפול|אחריות', re.I)
_CONDITION = re.compile(r'\b(?:except|exclud(?:es?|ed|ing)|not\s+included|subject\s+to|provisional|estimated|conditional|conditioned|if\s+approved|unless|minimum|per\s+(?:unit|item|hour|day|visit|met(?:er|re)|square\s+met(?:er|re))|extra|charged\s+separately|separately\s+charged|additional\s+charge|requires?|required|must|partial|deposit|balance|due|pending|valid|approval|authorization)\b|למעט|בכפוף|מותנ|לא\s+כולל|אינו\s+כולל|(?:אינו|אינה|אינם|אינן|לא)\s+כלול(?:ה|ים|ות)?|זמני|אומד|מקדמ|חלקי|יתרה|מינימ|ליחידה|למטר|למ[״\"]ר|לשעה|ליום|לפריט|בתוספת|בנפרד|נדרש|דרוש|טעון|בתנאי|תוקף|תק[פף][הת]?|אישור', re.I)
_BARE_CONDITION = re.compile(r'^(?:subject\s+to|except|excluding|not\s+included|minimum\s+(?:quantity|order)|valid\s+(?:for|until)|vat\s+(?:excluded|included)|בכפוף|למעט|לא\s+כולל|מינימום)', re.I)
_PRICE_SCOPE = re.compile(r'\b(?:price|quote|discount|payment|amount|rate|fee)\b.{0,70}\b(?:only|after|before|if)\b|\b(?:work|service|installation|repair|delivery)\s+(?:(?:may|will|can)\s+)?(?:begins?|starts?|available|provided|performed)\s+(?:only\s+)?(?:after|before|following|on|with|upon)\b|\b(?:before|including|with|without)\s+(?:vat|tax)\b|(?:המחיר|העבודה|השירות|התקנה|הובלה).{0,55}(?:בלבד|לאחר|לפני|מותנ)', re.I)
_OPEN_CONDITION = re.compile(r'(?:subject\s+to|after|before|excluding|except|unless|and|or|בכפוף\s+ל?|למעט|לאחר|לפני|בתנאי|כולל|[,:])\s*$', re.I)
_FIELD_BOUNDARY = re.compile(r'(?<!\w)(?:phone|telephone|tel|e-?mail|website|contact|address|registration|טלפון|כתובת|דוא[״\"]?ל|אימייל)\s*(?::|(?=\+?\d|https?://|www\.|[\w.+-]+@))', re.I)
_CONTACT_VALUE = re.compile(r'\S+@\S+|(?:https?://|www\.)\S+|\+?\d[\d ()-]{6,}\d', re.I)
_CONTINUATION = re.compile(r'^(?:and|or|of|for|by|with|without|including|plus|to|the\s+(?:site|owner)|של|לרבות|וכן|ו(?=[\u05d0-\u05ea]))\b', re.I)


def _material(value):
    return bool((_SUBJECT.search(value) and _CONDITION.search(value))
        or _BARE_CONDITION.search(value) or _PRICE_SCOPE.search(value))


def material_monetary_conditions(record, selected_item=None):
    """Select literal condition clauses, including detached continuations.

    A clause qualifies by its price/work meaning, never by its page position.
    Explicit field boundaries keep an adjacent contact field out of a condition;
    the same contact stays available to an explicit request below.
    """
    pieces = []
    for block in record.qualifiers:
        for line in str(block).splitlines():
            starts = [match.start() for match in _FIELD_BOUNDARY.finditer(line)]
            if starts:
                boundaries = sorted(set([0, *starts, len(line)]))
                fragments = [line[a:b] for a, b in zip(boundaries, boundaries[1:])]
            else:
                fragments = [line]
            for fragment in fragments:
                pieces.extend(value.strip() for value in re.split(
                    r'(?<=;)|\s+\|\s+|(?<=[.!?])\s+(?=[A-Z\u05d0-\u05ea])', fragment) if value.strip())
    selected = []
    index = 0
    while index < len(pieces):
        clause = pieces[index]
        index += 1
        if not _material(clause):
            continue
        while index < len(pieces) and not re.search(r'[.!?;]\s*$', clause):
            following = pieces[index]
            # Explicit structured fields are independent of the condition.
            if _FIELD_BOUNDARY.match(following) or (_CONTACT_VALUE.search(following) and not _material(following)):
                break
            if not (_material(following) or _OPEN_CONDITION.search(clause)
                    or _CONTINUATION.search(following)
                    or (following[:1].islower() and _SUBJECT.search(following))):
                break
            clause += ' ' + following
            index += 1
        # A condition repeated in an item's own description is already shown
        # in the priced scope; retain its evidence without displaying it twice.
        normalized = ' '.join(clause.split()).rstrip(';')
        # A footer can explicitly name another table row. Full description
        # matching is safe here; a generic exclusion such as "transport is
        # extra" remains a document-level condition rather than guessing an
        # owner from one shared service word.
        owners = []
        words = tuple(re.findall(r'[A-Za-z0-9\u05d0-\u05ea]+', normalized.casefold()))
        for item in record.items:
            description = tuple(re.findall(r'[A-Za-z0-9\u05d0-\u05ea]+', item.description.casefold()))
            if len(description)>=2 and any(words[start:start+len(description)]==description
                    for start in range(len(words)-len(description)+1)):
                owners.append(item)
        if selected_item is not None and owners and selected_item not in owners:
            continue
        if normalized and not any(normalized in ' '.join(item.description.split()) for item in record.items):
            selected.append(normalized)
    return tuple(dict.fromkeys(selected))


def requested_source_details(record, fields):
    """Read explicitly requested labelled source fields or report uncertainty.

    Contacts require a supplier role on their own line or governing block. A
    recipient's number is never silently promoted into the supplier's number.
    """
    output = []
    lines = record.source_text.splitlines()
    supplier = re.compile(r'\b(?:supplier|vendor|provider)\b|(?<!\w)ספק(?!\w)', re.I)
    recipient = re.compile(r'\b(?:customer|recipient|client|bill\s+to|ship\s+to)\b|לכבוד|לקוח|נמען', re.I)
    value_patterns = {
        'supplier_phone': re.compile(r'(?:phone|telephone|tel|טלפון)\s*:?\s*(\+?\d[\d ()-]{6,}\d)', re.I),
        'supplier_email': re.compile(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}'),
        'supplier_website': re.compile(r'(?:https?://|www\.)[\w./?&=%+#~-]+', re.I),
    }
    supplier_context = False
    candidates = {field: [] for field in fields if field in value_patterns}
    for line in lines:
        if recipient.search(line):
            supplier_context = False
        if supplier.search(line):
            supplier_context = True
        if not supplier_context:
            continue
        for field in candidates:
            for match in value_patterns[field].finditer(line):
                candidates[field].append(match[1] if field == 'supplier_phone' else match[0])
    for field in fields:
        if field == 'issue_date':
            output.append((field, record.issue_date.isoformat() if record.issue_date else None))
        elif field in candidates:
            values = tuple(dict.fromkeys(candidates[field]))
            output.append((field, values[0] if len(values) == 1 else None))
    return tuple(output)
