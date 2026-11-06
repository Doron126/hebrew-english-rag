"""Shared financial document roles and literal primary identities.

Request references propose a source. Only a numbered source title establishes
its primary identity; related references and arbitrary matching digits do not.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import re


DOCUMENT_KIND_PATTERN = (
    r'(?:tax\s+invoice\s*/\s*receipt|invoice\s*/\s*receipt|receipt\s*/\s*invoice|'
    r'חשבונית(?:\s+מס)?\s*/?\s*קבלה|קבלה\s*/\s*חשבונית(?:\s+מס)?|'
    r'credit\s+note|חשבונית\s+זיכוי|quotation|quote|estimate|הצעת\s+מחיר|'
    r'tax\s+invoice|חשבונית\s+(?:מס|עסקה)|invoice|receipt|חשבונית|קבלה)'
)
NUMBER_LABEL_PATTERN = r'(?:number|no\.?|מספר|מס[\'׳.]?)'
DOCUMENT_NUMBER_PATTERN = r'[0-9]{1,12}(?![0-9]|[.,][0-9]|\s*[%₪$€£])'
DOCUMENT_REFERENCE_RE = re.compile(
    r'(?<!\w)(?:ו?[בלמכ]?ה?)?(?P<kind>'+DOCUMENT_KIND_PATTERN+r')(?!\w)\s*'
    r'(?:'+NUMBER_LABEL_PATTERN+r'\s*)?[#:]?\s*'
    r'(?P<number>'+DOCUMENT_NUMBER_PATTERN+r')', re.I)


@dataclass(frozen=True)
class DocumentIdentity:
    kind: str
    number: str
    span: tuple[int, int]
    number_span: tuple[int, int]
    issuer: str = ''


def canonical_document_kind(value):
    value = str(value or '').casefold()
    if value in {'invoice_receipt', 'quotation', 'credit', 'quote'}:
        return 'quote' if value == 'quotation' else value
    if re.search(r'credit|זיכוי', value):
        return 'credit'
    invoice = bool(re.search(r'invoice|חשבוני(?:ת|ות)', value))
    receipt = bool(re.search(r'receipt|קבל(?:ה|ות)', value))
    if invoice and receipt:
        return 'invoice_receipt'
    if invoice:
        return 'invoice'
    if receipt:
        return 'receipt'
    return 'quote' if re.search(r'quotation|quote|estimate|הצע(?:ת|ות)\s+ה?מחיר', value) else ''


def document_roles(kind):
    kind = canonical_document_kind(kind)
    return frozenset({'invoice', 'receipt'}) if kind == 'invoice_receipt' else frozenset({kind}) if kind else frozenset()


def role_matches(source_kind, requested_kind):
    wanted = document_roles(requested_kind)
    return bool(wanted and wanted <= document_roles(source_kind))


def listing_kind_matches(source_kind, requested_kind):
    """List the invoice family with its actual subtype, including credits.

    This discovery policy does not change financial-record role matching.
    """
    actual = canonical_document_kind(source_kind)
    wanted = canonical_document_kind(requested_kind)
    return bool(actual and (not wanted or role_matches(actual, wanted)
                            or (wanted == 'invoice' and actual == 'credit')))


def document_identities(text):
    """Return request references with original spelling and exact spans."""
    return tuple(DocumentIdentity(canonical_document_kind(match['kind']), match['number'],
        match.span(), match.span('number')) for match in DOCUMENT_REFERENCE_RE.finditer(str(text or '')))


# Primary title IDs may be alphanumeric. Unlike a loose document mention,
# they must occupy an entire heading and include a digit.
_PRIMARY_NUMBER = r'(?:[A-Za-z](?=[A-Za-z0-9-]{1,20}(?![A-Za-z0-9-]))(?=[A-Za-z0-9-]*\d)[A-Za-z0-9-]+|'+DOCUMENT_NUMBER_PATTERN+r')'
_PRIMARY_TITLE = re.compile(
    r'(?:(?P<kind>'+DOCUMENT_KIND_PATTERN+r')\s*'
    r'(?:'+NUMBER_LABEL_PATTERN+r'\s*)?[#:]?\s*(?P<number>'+_PRIMARY_NUMBER+r')|'
    r'(?P<reverse_number>'+_PRIMARY_NUMBER+r')\s*[#:]?\s*(?P<reverse_kind>'+DOCUMENT_KIND_PATTERN+r')'
    r'(?:\s+'+NUMBER_LABEL_PATTERN+r')?)'
    r'(?:\s+(?:original|copy|מקור|העתק))?\s*[.:]?', re.I)
_BODY_HEADING = re.compile(
    r'(?:description|service\s+description|items?|line\s+items?|sub[ -]?total|total\s+due|'
    r'payment\s+(?:details|history|received)|פירוט|תיאור|תאור|פרטי\s+תשלומים|פירוט\s+תשלומים)\s*:?', re.I)
_RELATED_CONTEXT = re.compile(
    r'(?:related|reference|referenced|according\s+to|for\s+(?:account|transaction|order)|'
    r'אסמכתא|מסמך\s+קשור|עבור\s+(?:חשבון|הזמנה)|בהתאם\s+ל)\b', re.I)


def primary_document_identities(text):
    """Read only complete numbered title lines, in either visual order.

    A wrapped title/number pair is accepted. Footer metadata, references to an
    order/account/other document and a service heading are not primary titles.
    Repeated copies of the same title are one identity, preserving its first
    source span. Distinct titles stay distinct so the caller can reject them.
    """
    value = str(text or '')
    lines = tuple(re.finditer(r'[^\n]+', value))
    found = []
    seen = set()
    for index, line in enumerate(lines):
        if _BODY_HEADING.fullmatch(line.group().strip()):
            break
        if index and _RELATED_CONTEXT.search(lines[index-1].group()):
            continue
        options = [(line.start(), line.end())]
        if index+1 < len(lines) and not value[line.end():lines[index+1].start()].strip():
            options.append((line.start(), lines[index+1].end()))
        for start, end in options:
            body = value[start:end]
            left = len(body)-len(body.lstrip())
            body = body.strip()
            match = _PRIMARY_TITLE.fullmatch(body)
            if not match:
                continue
            kind = canonical_document_kind(match['kind'] or match['reverse_kind'])
            group = 'number' if match['number'] else 'reverse_number'
            number = match[group]
            key = (kind, number)
            if key not in seen:
                a, b = match.span(group)
                found.append(DocumentIdentity(kind, number, (start+left, start+left+len(body)),
                    (start+left+a, start+left+b)))
                seen.add(key)
            break
    return tuple(found)


def explicit_document_request(query):
    """An anchored find/open command requests a source, rather than a scalar."""
    value = str(query or '').strip()
    references = document_identities(value)
    if len(references) != 1:
        return None
    reference = references[0]
    before, after = value[:reference.span[0]], value[reference.span[1]:]
    if not re.fullmatch(
        r'(?:(?:please\s+)?(?:show|display|find|open|bring|give|locate)(?:\s+me)?\s+(?:the\s+)?|'
        r'(?:הצג|הציגי|הציגו|תציג|תציגי|הראה|הראי|תראה|תראי|מצא|מצאי|תמצא|תמצאי|פתח|פתחי|תפתח|תפתחי|הבא|הביאי|תביא|תביאי|תן|תני)\s+(?:לי\s+)?(?:את\s+)?)',
        before, re.I):
        return None
    remainder = after.strip(' .?!,;:')
    if not remainder:
        return reference
    issuer = re.fullmatch(r'(?:from|by|issued\s+by|של|מאת|מטעם)\s+(.{1,120})', remainder, re.I)
    if not issuer or not re.fullmatch(r'[\w\s.&\-\'"״]+', issuer[1]):
        return None
    return replace(reference, issuer=issuer[1].strip())


def issuer_span(text, issuer):
    """Only an explicitly labelled source issuer disambiguates an issuer cue."""
    if not issuer:
        return None
    normalize = lambda value: tuple(re.findall(r'[^\W_]+', value.casefold(), re.UNICODE))
    wanted = normalize(issuer)
    for match in re.finditer(r'(?im)^(?:issuer|supplier|issued\s+by|מנפיק|ספק)\s*[:\-]\s*([^\n]+)$', text):
        observed = normalize(match[1])
        if wanted and any(observed[a:a+len(wanted)] == wanted for a in range(len(observed)-len(wanted)+1)):
            return match.span()
    return None


def document_kind_label(kind, hebrew=False):
    labels = {
        'quote': ('הצעת מחיר', 'quotation'),
        'invoice': ('חשבונית', 'invoice'),
        'receipt': ('קבלה', 'receipt'),
        'invoice_receipt': ('חשבונית מס / קבלה', 'invoice / receipt'),
        'credit': ('חשבונית זיכוי', 'credit note'),
    }
    return labels.get(canonical_document_kind(kind), ('מסמך', 'document'))[0 if hebrew else 1]
@dataclass(frozen=True)
class DocumentTypeListing:
    """An unnumbered document category and its explicit restrictions."""
    kind: str
    customer: str = ''
    year: str = ''
    project: str = ''
    collection: str = ''
    date: str = ''
    topic: str = ''
    period: str = ''
    date_basis: str = ''
    topic_groups: tuple[tuple[str, ...], ...] = ()
    date_start: str = ''
    date_end: str = ''


_LIST_TYPES = (
    r'(?:חשבוניות?\s+ה?זיכוי|חשבונית\s+ה?זיכוי|'
    r'credit\s+(?:notes?|invoices?)|invoices?\s+credits?|'
    r'הצע(?:ת|ות)\s+ה?מחיר|quotations?|quotes?|estimates?|'
    r'חשבוני(?:ת|ות)\s+ה?מס\s*/?\s*קבל(?:ה|ות)|'
    r'tax\s+invoices?\s*/\s*receipts?|invoices?\s*/\s*receipts?|'
    r'חשבוני(?:ת|ות)\s+ה?מס|tax\s+invoices?|'
    r'חשבוניות|חשבונית|invoices?|קבלות|קבלה|receipts?)')
_LIST_COMMAND = (
    r'(?:תביא(?:י|ו)?|הבא|הביאי|הביאו|תציג(?:י|ו)?|הצג|הציגי|הציגו|'
    r'תראה|תראי|הראה|הראו|מצא|מצאי|מצאו|תמצא|תמצאי|רשום|רשמי|תן|תני|'
    r'(?:please\s+)?(?:show|find|list|give|bring|display))')
_LIST_PREFIX = rf'(?:{_LIST_COMMAND}\s+(?:(?:לי|me)\s+)?(?:(?:את|the)\s+)?|(?:אילו|איזה|איזו|which|what)\s+)(?:(?:רשימה\s+של|רשימת|(?:a\s+)?list\s+of)\s+)?'
_LIST_QUALIFIER = (
    r'(?:שאתה\s+מוצא|שאת\s+מוצאת|ש(?:קיימות|קיימים|קיימת|קיים)|'
    r'הקיימות|הקיימים|שיש\s+ברשותך|יש(?:\s+(?:במסמכים|ברשותך))?|'
    r'נמצא(?:ות|ים|ת)?\s+במסמכים|you\s+(?:can\s+)?find|available)'
)
_TOPIC_RELATION_START = r'(?:(?:ש|ה)?קשור(?:ה|ות|ים)?|(?:ש|ה)?נוגע(?:ת|ות|ים)?|(?:ש|ה)?מתייחס(?:ת|ות|ים)?)\s+ל'
_RELATION_START = (r'(?:' + _TOPIC_RELATION_START +
                   r'|(?:(?:ש|ה)?מיועד(?:ת|ות|ים)?|(?:ש|ה)?שייכ(?:ת|ות|ים)?)\s+ל)')
_FILTER_START = (
    r'(?:של\s+(?:הלקוח\s+)?|עבור\s+(?:הלקוח\s+)?|ללקוח\s+|'
    r'for\s+(?:customer\s+)?|' + _RELATION_START + r'|(?:ב|ל)?פרויקט\s+|(?:in|for)\s+project\s+|'
    r'(?:ב|מ)?אוסף\s+|(?:in|from)\s+(?:the\s+)?collection\s+|'
    r'(?:מ(?:שנת|שנה)|בשנת|בשנה|של)\s*(?=\d{4}(?:\s|$))|(?:בשנת|in\s+)(?=\d)|'
    r'(?:בתאריך|מתאריך|on\s+|dated\s+))'
)
_NAMED_VALUE = r"[\w\s.&\-'\"׳״]+?"


def _listing_tail_filters(tail):
    """Consume recognized restrictions completely; never ignore leftover words."""
    fields = {'customer': '', 'year': '', 'project': '', 'collection': '', 'date': '', 'topic': ''}
    remaining = tail.strip()
    while remaining:
        remaining = re.sub(r'^(?:(?:וגם|and)\s+|ו(?=(?:ב|ל)?פרויקט|(?:ב|מ)?אוסף))', '', remaining, flags=re.I)
        suffix = re.match(r'^' + _LIST_QUALIFIER + r'(?:\s+|$)', remaining, re.I)
        if suffix:
            remaining = remaining[suffix.end():].strip()
            continue
        date = re.match(
            r'^(?:(?:מ(?:שנת|שנה)|בשנת|בשנה|של)\s*|(?:in|of|from)\s+(?:the\s+year\s+)?)'
            r'(?P<year>\d{4})(?=\s|$)', remaining, re.I)
        exact_date = re.match(
            r'^(?:בתאריך|מתאריך|on\s+|dated\s+)\s*'
            r'(?P<date>\d{1,2}[./-]\d{1,2}[./-]\d{4}|\d{4}-\d{2}-\d{2})(?=\s|$)',
            remaining, re.I)
        if date or exact_date:
            match = date or exact_date
            key = 'year' if date else 'date'
            if fields[key]:
                return None
            fields[key] = match[key]
            remaining = remaining[match.end():].strip()
            continue
        named = re.match(
            r'^(?:(?P<project>(?:ב|ל)?פרויקט\s+|(?:in|for)\s+project\s+)|'
            r'(?P<customer>של\s+(?:הלקוח\s+)?|עבור\s+(?:הלקוח\s+)?|ללקוח\s+|'
            r'for\s+(?:customer\s+)?)|'
            r'(?P<topic>'+_RELATION_START+r')|'
            r'(?P<collection>(?:ב|מ)?אוסף\s+|(?:in|from)\s+(?:the\s+)?collection\s+))'
            rf'(?P<value>{_NAMED_VALUE})(?=\s+(?:{_FILTER_START}|{_LIST_QUALIFIER})|$)',
            remaining, re.I)
        if not named:
            return None
        key = next(k for k in ('customer', 'project', 'collection', 'topic') if named[k])
        value = named['value'].strip()
        # These words introduce conditions rather than a customer/project name.
        if (not value or fields[key] or re.search(
                r'(?:\b(?:with|where|after|before|between|from|dated|project|collection)\b|'
                r'(?:^|\s)(?:שכולל|שכוללת|שכוללים|ששולמו|שטרם|עם|ללא|אחרי|לפני|בין|בפרויקט|באוסף)(?:\s|$))',
                value, re.I)):
            return None
        fields[key] = value
        remaining = remaining[named.end():].strip()
    return fields


def document_type_listing(query):
    """Recognize category discovery without consuming numbered or content queries."""
    value = str(query or '').strip().strip(' ?!.,')
    from financial_semantics import financial_query, _NAMED_MONTH, _NUMERIC_MONTH
    monthly = financial_query(value)
    if monthly and monthly.operation == 'documents':
        months = list(_NAMED_MONTH.finditer(value)) + list(_NUMERIC_MONTH.finditer(value))
        if len(months) == 1:
            before = value[:months[0].start()].strip()
            after = value[months[0].end():].strip(' ?!.,')
            simple = re.fullmatch(
                rf'(?:{_LIST_PREFIX}(?:(?:כל|all)\s+)?(?:ה)?{_LIST_TYPES}\s+(?:for|in|of|של|עבור|ב)|'
                rf'(?:מהן|מהם)\s+(?:ה)?{_LIST_TYPES}\s+(?:של|עבור|ב))\s*', before, re.I)
            if simple and not after:
                return DocumentTypeListing(monthly.document_kind, period=monthly.period,
                                           date_basis=monthly.date_basis)
    match = re.fullmatch(
        rf'^{_LIST_PREFIX}(?:(?:כל|all|\d{{1,3}})\s+)?(?:ה)?(?P<kind>{_LIST_TYPES})(?P<tail>.*)$',
        value, re.I)
    if match:
        kind_text = match['kind']
    else:
        # Noun phrases may place the document and subtype in either order.
        match = re.fullmatch(
            rf'^{_LIST_PREFIX}(?:(?:כל|all)\s+)?'
            rf'(?:מסמכי\s+(?:ה)?(?P<subtype>זיכוי|קבלה|מס|עסקה)\s+מסוג\s+'
            rf'(?P<head>חשבונית|קבלה)|מסמכי\s+(?:ה)?(?P<kind>{_LIST_TYPES}))'
            rf'(?P<tail>.*)$', value, re.I)
        if not match:
            return None
        kind_text = match.groupdict().get('kind') or (match['head'] + ' ' + match['subtype'])
    kind = canonical_document_kind(kind_text)
    if not kind:
        return None
    fields = _listing_tail_filters(match['tail'])
    return DocumentTypeListing(kind=kind, **fields) if fields is not None else None



def _listing_result_count_match(query):
    """Locate an explicit leading document count without consuming IDs or dates."""
    value = str(query or '')
    return re.match(
        rf'^\s*{_LIST_PREFIX}(?:first\s+)?(?P<limit>\d{{1,3}})\s+'
        rf'(?:ה)?(?:{_LIST_TYPES}|מסמכים|מסמך|קבצים|קובץ|documents?|files?|records?|sources?)(?!\w)',
        value, re.I)


def listing_result_limit(query):
    """Count only an explicit leading document-list size."""
    match = _listing_result_count_match(query)
    return int(match['limit']) if match and int(match['limit']) > 0 else 0

def _listing_named_match(labels, requested):
    wanted = tuple(re.findall(r'[^\W_]+', requested.casefold(), re.UNICODE))
    if not labels:
        return None
    # Reuse the monetary reader's bounded Hebrew normalization: known work
    # nouns can carry grammatical prefixes, and a locative prefix on a name
    # is interpreted only after an explicit work/site noun. Never remove an
    # arbitrary first letter from a standalone proper name.
    from monetary_questions import _word_variants
    wanted_variants = _word_variants(wanted)
    for label in labels:
        actual = tuple(re.findall(r'[^\W_]+', str(label).casefold(), re.UNICODE))
        if wanted:
            actual_variants = _word_variants(actual)
            for i in range(len(actual) - len(wanted) + 1):
                window = actual_variants[i:i + len(wanted)]
                if all(left & right for left, right in zip(window, wanted_variants)):
                    return True
    return False


_LISTING_DATE = re.compile(r'(?<!\d)(?:\d{4}-\d{2}-\d{2}|\d{1,2}[./-]\d{1,2}[./-]\d{4})(?!\d)')
_DATE_ROLE_PATTERNS = {
    'issue': r'(?:issue\s+date|invoice\s+date|date\s+(?:of\s+)?issue|תאריך\s+(?:הנפקה|הפקה|המסמך|חשבונית))',
    'payment': r'(?:payment\s+date|date\s+(?:of\s+)?payment|תאריך\s+(?:ה)?תשלום|מועד\s+(?:ה)?תשלום)',
    'due': r'(?:due\s+date|date\s+due|תאריך\s+(?:ה)?פירעון|מועד\s+(?:ה)?פירעון)',
    'service': r'(?:service\s+date|date\s+(?:of\s+)?service|תאריך\s+(?:ה)?שירות)',
}


def listing_query_date_basis(query):
    """Return the one explicitly named date role, or an ambiguity marker."""
    found = [role for role, pattern in _DATE_ROLE_PATTERNS.items()
             if re.search(pattern, str(query or ''), re.I)]
    return found[0] if len(found) == 1 else 'ambiguous' if found else ''


def _parse_listing_dates(literals):
    from datetime import date as calendar_date
    parsed = set()
    for literal in literals:
        parts = [int(part) for part in re.split(r'[./-]', literal)]
        try:
            parsed.add(calendar_date(*parts) if len(literal.split('-')[0]) == 4
                       else calendar_date(parts[2], parts[1], parts[0]))
        except ValueError:
            continue
    return next(iter(parsed)) if len(parsed) == 1 else None


def parse_listing_request_date(value):
    """Parse a complete user-supplied calendar date (numeric or named month)."""
    from datetime import date as calendar_date
    value = str(value or '').strip()
    if _LISTING_DATE.fullmatch(value):
        return _parse_listing_dates([value])
    from financial_semantics import _MONTH_MAP, _MONTH_WORD
    match = re.fullmatch(rf'(\d{{1,2}})\s+(?:ב|ל)?({_MONTH_WORD})\.?\s*,?\s*(\d{{4}})', value, re.I)
    if not match:
        return None
    try:
        return calendar_date(int(match[3]), _MONTH_MAP[match[2].casefold()], int(match[1]))
    except ValueError:
        return None


def listing_issue_date(text):
    """Return an unambiguous source-owned issue date, without requiring amounts.

    Generic dates in payment, service or due-date lines are not issue dates;
    filesystem and filename dates never establish the document's date.
    """
    value = str(text or '')
    labelled = re.findall(
        r'(?im)^[ \t]*(?:issue\s+date|invoice\s+date|document\s+date|date\s+issued|date|'
        r'תאריך(?:\s+המסמך|\s+הפקה|\s+הנפקה|\s+חשבונית)?)'
        r'[ \t]*[:\-]?[ \t]*(' + _LISTING_DATE.pattern + r')', value)
    literals = list(labelled)
    if not literals:
        titles = primary_document_classifications(value)
        if len(titles) == 1:
            before = value[:titles[0].span[0]].splitlines()
            adjacent = [before[-1].strip()] if before else []
            literals = [item for item in adjacent if _LISTING_DATE.fullmatch(item)]
    return _parse_listing_dates(literals)


def listing_restriction_date(text, basis=''):
    if basis in ('', 'issue'):
        return listing_issue_date(text)
    pattern = _DATE_ROLE_PATTERNS.get(basis)
    if not pattern:
        return None
    dates = re.findall(r'(?im)^[ \t]*' + pattern + r'[ \t]*[:\-]?[ \t]*('
                       + _LISTING_DATE.pattern + r')', str(text or ''))
    return _parse_listing_dates(dates)


def listing_filter_matches(text, request, metadata=None):
    """Match source-backed explicit restrictions; unknown evidence is not exclusion."""
    checks = []
    if request.year or request.date or request.date_start or request.date_end:
        from datetime import date as calendar_date
        actual = listing_restriction_date(text, request.date_basis)
        if actual is None:
            checks.append(None)
        else:
            if request.year:
                checks.append(str(actual.year) == request.year)
            if request.date:
                try:
                    parts = [int(part) for part in re.split(r'[./-]', request.date)]
                    expected = (calendar_date(*parts) if len(str(parts[0])) == 4
                                else calendar_date(parts[2], parts[1], parts[0]))
                    checks.append(actual == expected)
                except (ValueError, TypeError):
                    checks.append(None)
            if request.date_start or request.date_end:
                start = parse_listing_request_date(request.date_start)
                end = parse_listing_request_date(request.date_end)
                checks.append(start <= actual <= end if start and end and start <= end else None)
    if request.period:
        from financial_semantics import extract_financial_record, FinancialQuery, financial_record_matches
        try:
            year, month = (int(part) for part in request.period.split('-'))
            record = extract_financial_record(text, source_id='listing', title='', tenant_id='listing')
            if record is None:
                checks.append(None)
            else:
                temporal = FinancialQuery('documents', request.date_basis, year, month, request.kind)
                roles = financial_record_matches(record, temporal)
                available = record.issue_date or record.service_periods
                checks.append(bool(roles) if available else None)
        except (TypeError, ValueError):
            checks.append(None)
    if request.customer:
        labels = re.findall(r'(?im)^[ \t]*[:\-]?[ \t]*(?:customer|client|bill\s+to|לקוח|לקוחה|לכבוד)[ \t]*[:\-]?[ \t]+([^\n]+)', text)
        labels += re.findall(r'(?im)^[ \t]*[:\-]?[ \t]*לכבוד[ \t]*[:\-]?[ \t]*\n[ \t]*([^\n]+)', text)
        checks.append(_listing_named_match(labels, request.customer))
    if request.project:
        labels = re.findall(r'(?im)^(?:project|פרויקט)\s*[:\-]\s*([^\n]+)', text)
        checks.append(_listing_named_match(labels, request.project))
    if request.topic:
        # A marked footer/contact section and contact lines cannot establish
        # the relationship; inspect headers and substantive body context.
        body = re.split(r'(?im)^\s*(?:footer|contact\s+(?:details|information)|פרטי\s+קשר|יצירת\s+קשר)\s*[:\-]?\s*$', text, maxsplit=1)[0]
        relevant = [line for line in body.splitlines()
                    if not re.search(r'(?i)(?:footer|טלפון|פקס|דוא\"ל|כתובת|www\.|https?://|@)', line)]
        checks.append(_listing_named_match(relevant, request.topic))
    if request.collection:
        meta = metadata or {}
        labels = [meta[key] for key in ('collection', 'collection_name')
                  if isinstance(meta.get(key), str) and meta[key].strip()]
        checks.append(_listing_named_match(labels, request.collection))
    if False in checks:
        return False
    return None if None in checks else True





def semantic_listing_candidate(query):
    """Offer the planner an unnumbered category, independently of phrasing."""
    value = str(query or '').strip()
    direct = document_type_listing(value)
    if direct and direct.period:
        # This fully parsed, explicitly structured month request already
        # carries the period and date basis understood by the listing reader.
        return False
    if not value or document_identities(value):
        return False
    if len(list(re.finditer(_LIST_TYPES, value, re.I))) == 1:
        return True
    # General document discovery can name a topic without naming a type.
    # Require both a discovery command and a relationship cue; questions
    # about the subject itself remain with the normal answer pipeline.
    interrogative_discovery = (re.search(r'(?i)(?<!\w)(?:which|what|אילו|איזה|איזו|מהן|מהם)(?!\w)', value)
        and re.search(r'(?i)(?<!\w)(?:documents?|files?|materials?|sources?|records?|'
                      r'מסמכ\w*|קבצי\w*|המסמכ\w*|הקבצי\w*|חומר|החומר|מקורות|תיעוד)(?!\w)', value))
    return bool((re.search(_LIST_COMMAND, value, re.I) or interrogative_discovery)
                and re.search(r'(?i:\b(?:about|related\s+to|regarding|on|describ(?:e|es)|discuss(?:es)?|cover(?:s)?)\b|'
                              r'(?<!\w)(?:לגבי|אודות|בנושא|שקשור\w*|קשור\w*|שנוגע\w*|'
                              r'שעוסק\w*|עוסק\w*|מתאר\w*|המתאר\w*|העוסק\w*|דנ(?:ים|ות|ה))(?!\w))', value)
                and not re.search(r'(?i:\b(?:summari[sz]e|summary|explain)\b|'
                                  r'(?<!\w)(?:סכם|סכמי|סיכום|הסבר|הסבירי)(?!\w))', value))


def explicit_category_financial_intent(query):
    """A category's own price word is not a request for a monetary fact."""
    remainder = re.sub(_LIST_TYPES, ' ', str(query or ''), flags=re.I)
    # A named date role constrains a document listing; it does not by itself
    # ask for a payment amount. Other monetary words remain untouched.
    for pattern in _DATE_ROLE_PATTERNS.values():
        remainder = re.sub(pattern, ' ', remainder, flags=re.I)
    return bool(re.search(
        r'(?i)\b(?:price|cost|amount|total|payment|paid|pay|compare|comparison|'
        r'expensive|cheaper|more|less|above|below)\b|'
        r'(?<!\w)(?:מחיר|המחיר|מחירן|מחירם|עלות|העלות|סכום|הסכום|סך|'
        r'תשלום|התשלום|שולם|שולמו|שילם|כמה|השוואה|השווה|להשוות|'
        r'יקר|יקרה|זול|זולה|מעל|מתחת)(?!\w)|[₪$€£%]', remainder))


def quotation_amount_topic(query):
    """Extract a literal topic from an explicit question for one quote's price."""
    value = str(query or '').strip()
    match = re.fullmatch(r'(?:מה|מהו|מהי)\s+(?:(?:היה|הייתה|היתה|יהיה|תהיה)\s+)?'
        r'(?:ה)?הצעת\s+(?:ה)?מחיר\s+(?:ל(?P<attached>[\u05d0-\u05ea]+'
        r'(?:\s+[\u05d0-\u05ea]+){0,3})|עבור\s+(?P<separate>[\u05d0-\u05ea]+'
        r'(?:\s+[\u05d0-\u05ea]+){0,3}))\s*[?!.]*', value)
    return (match.group('attached') or match.group('separate')) if match else ''


def quotation_topic_variants(query):
    """Retain both readings of an attached Hebrew preposition in a name."""
    topic = quotation_amount_topic(query)
    if not topic:
        return ()
    value = str(query or '')
    if re.search(r'(?<!\w)ל'+re.escape(topic)+r'\s*[?!.]*$', value):
        return topic, 'ל'+topic
    return (topic,)


def quote_header_topic_matches(text, identity, topics):
    """Bind a quote topic to an issuer or recipient in its title header."""
    header = str(text or '')[:identity.span[0]]
    lines = [line.strip() for line in header.splitlines() if line.strip()]
    if len(lines) > 40:
        return False
    for index, line in enumerate(lines):
        if re.search(r'(?i)(?:www\.|https?://|@|טלפון|פקס|telephone|phone)', line):
            continue
        preceding = lines[index-1] if index else ''
        role = re.search(r'(?i)(?:issuer|supplier|customer|client|bill\s+to|מנפיק|ספק|לקוח|לקוחה|לכבוד)', line+' '+preceding)
        company_heading = (index < 6 and re.search(
            r'(?i)(?:חברה|חברת|בע\s*["״]?\s*מ|\b(?:company|corporation|inc|ltd)\b)', line))
        if (company_heading or role) and any(_listing_named_match([line], topic) for topic in topics):
            return True
    return False


def quote_total_literal(text):
    """Find one complete explicitly labeled NIS document-total line.

    Match a whole source line, with currency, amount and the total-price
    label adjacent. Never calculate, select a line-item price, or resolve
    conflicting totals by choosing one.
    """
    currency = r'(?:שח|ש["״]ח|₪|ILS|NIS)'
    amount = r'(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d{2})?'
    label = r'ס["״]?ה["״]?כ\s+מחיר'
    pattern = re.compile(r'^\s*(?:'+currency+r'\s*'+amount+r'\s*'+label+r'|'
                         +label+r'\s*'+amount+r'\s*'+currency+r')\s*$', re.I | re.M)
    value = str(text or '')
    matches = tuple(pattern.finditer(value))
    # Another explicitly labeled overall total, including a VAT-qualified
    # amount, makes the requested monetary scope uncertain. Do not ignore it.
    alternative = re.compile(r'(?i)(?:ס["״]?ה["״]?כ\s*(?:כולל|לתשלום|לפני|אחרי)|'
        r'(?:מחיר|סכום)\s+כולל\s+(?:מע\s*["״]?מ|מס)|'
        r'\btotal\s+(?:including|excluding|with|without|due)\b)')
    # Extraction can wrap the competing amount onto the following line or
    # page. The label alone makes monetary scope unresolved.
    if any(alternative.search(line) for line in value.splitlines()):
        return ''
    return matches[0].group().strip() if len(matches) == 1 else ''


def validate_semantic_document_listing(query, proposal):
    """Bind planner fields to literal user text; never invent or drop restrictions."""
    if not isinstance(proposal, dict) or proposal.get('operation') != 'document_listing':
        return None
    if proposal.get('kind') not in {'', 'credit', 'invoice', 'invoice_receipt', 'receipt', 'quote'}:
        return None
    if not semantic_listing_candidate(query) or explicit_category_financial_intent(query):
        return None
    matched = re.search(_LIST_TYPES, str(query), re.I)
    if matched:
        if canonical_document_kind(matched.group()) != proposal['kind']:
            return None
    elif proposal['kind']:
        return None
    fields = ('customer', 'project', 'topic', 'collection', 'year', 'date')
    if set(proposal) == {'operation', 'kind'}:
        # Legacy interpretation is valid only for an unrestricted category query.
        if not matched:
            return None
        suffix = str(query)[matched.end():].strip(' ?!.,')
        if suffix and not re.fullmatch(_LIST_QUALIFIER, suffix, re.I):
            return None
        prefix = str(query)[:matched.start()]
        count_match = _listing_result_count_match(str(query))
        if count_match and int(count_match['limit']) > 0:
            start, end = count_match.span('limit')
            prefix = prefix[:start] + ' ' + prefix[end:]
        if re.search(r'\d|(?i:\b(?:for|from|customer|client|project|collection|without|except|where|with)\b|(?<!\w)(?:עבור|לקוח|לקוחה|פרויקט|אוסף|ללא|למעט|של)(?!\w))', prefix):
            return None
        return DocumentTypeListing(proposal['kind'])
    keys = set(proposal)
    base_keys = {'operation', 'kind', *fields}
    if not base_keys <= keys or keys - base_keys - {'period', 'date_basis', 'topic_groups', 'date_start', 'date_end'}:
        return None
    if proposal.get('period', ''):
        return None
    values = {}
    for field in fields:
        value = proposal[field]
        if not isinstance(value, str) or len(value) > 120 or value != value.strip():
            return None
        if value and value.casefold() not in str(query).casefold():
            return None
        values[field] = value
    for field in ('date_start', 'date_end'):
        value = proposal.get(field, '')
        if not isinstance(value, str) or len(value) > 120 or value != value.strip():
            return None
        if value and value.casefold() not in str(query).casefold():
            return None
        values[field] = value
    if values['date_start'] or values['date_end']:
        start = parse_listing_request_date(values['date_start'])
        end = parse_listing_request_date(values['date_end'])
        if (not start or not end or start > end
                or not re.search(r'(?i:\bbetween\b|(?<!\w)בין(?!\w))', str(query))):
            return None
    basis = listing_query_date_basis(query)
    if basis == 'ambiguous':
        return None
    proposed_basis = proposal.get('date_basis', '')
    if not isinstance(proposed_basis, str) or proposed_basis not in ('', 'issue', 'payment', 'due', 'service'):
        return None
    if proposed_basis and proposed_basis != (basis or 'issue'):
        return None
    if basis and not (values['year'] or values['date'] or values['date_start']):
        return None
    groups = proposal.get('topic_groups', ())
    if not isinstance(groups, (tuple, list)) or len(groups) > 6:
        return None
    normalized_groups = []
    for group in groups:
        if not isinstance(group, (tuple, list)) or not 1 <= len(group) <= 8:
            return None
        if any(not isinstance(term, str) or not term.strip() or term != term.strip()
               or len(term) > 100 for term in group):
            return None
        # Every semantic concept has a literal anchor in the requested topic;
        # expansions cannot add a separate invented restriction.
        if not any(len(term) >= 2 and term.casefold() in values['topic'].casefold() for term in group):
            return None
        normalized_groups.append(tuple(group))
    if groups:
        # Don't let a planner silently omit a qualification from the topic.
        anchors = [term.casefold() for group in groups for term in group
                   if term.casefold() in values['topic'].casefold()]
        words = re.findall(r'[^\W_]+', values['topic'].casefold(), re.UNICODE)
        ignored = {'או', 'וגם', 'וכן', 'של', 'the', 'and', 'or', 'of', 'for', 'a', 'an'}
        if any(word not in ignored and not any(
                re.search(r'(?<!\w)' + re.escape(word) + r'(?!\w)', anchor)
                for anchor in anchors) for word in words):
            return None
    # A general relationship is broader than the role of recipient. Reject a
    # model proposal that silently narrows it to the customer field, unless the
    # user independently and explicitly asked for that role.
    if (values['customer'] and not values['topic']
            and re.search(_TOPIC_RELATION_START, str(query), re.I)
            and not re.search(r'(?i:\b(?:customer|client|recipient)\b|(?<!\w)(?:לקוח|לקוחה|ללקוח|נמען)(?!\w))', str(query))):
        return None
    if not proposal['kind'] and (not values['topic'] or not normalized_groups):
        return None
    if not any(values.values()):
        return validate_semantic_document_listing(query, {'operation': 'document_listing', 'kind': proposal['kind']})
    # A model is not authorized to silently drop numbers or comparisons.
    remaining = str(query)
    count_match = _listing_result_count_match(remaining)
    if count_match and int(count_match['limit']) > 0:
        start, end = count_match.span('limit')
        remaining = remaining[:start] + ' ' + remaining[end:]
    for value in values.values():
        if value:
            remaining = re.sub(re.escape(value), ' ', remaining, count=1, flags=re.I)
    if basis:
        remaining = re.sub(_DATE_ROLE_PATTERNS[basis], ' ', remaining, flags=re.I)
    if values['date_start'] and values['date_end']:
        remaining = re.sub(r'(?i:\bbetween\b|(?<!\w)בין(?!\w))', ' ', remaining)
    if re.search(r'\d|(?i:\b(?:without|except|after|before|between|over|under|paid|payment|amount|price|total)\b|(?<!\w)(?:ללא|למעט|אחרי|לפני|בין|מעל|מתחת|ששולמו|שטרם|סכום|מחירם|סך)(?!\w))', remaining):
        return None
    if values['year'] and not re.fullmatch(r'\d{4}', values['year']):
        return None
    if values['date'] and not re.fullmatch(r'\d{1,4}[./-]\d{1,2}[./-]\d{1,4}', values['date']):
        return None
    # Explicit typed restrictions cannot be silently collapsed into another field.
    for pattern, field in ((r'(?i:\bproject\b|(?<!\w)(?:פרויקט|בפרויקט)(?!\w))', 'project'),
                           (r'(?i:\bcollection\b|(?<!\w)(?:אוסף|באוסף)(?!\w))', 'collection'),
                           (r'(?i:\b(?:customer|client)\b|(?<!\w)(?:לקוח|לקוחה|ללקוח)(?!\w))', 'customer')):
        if re.search(pattern, remaining) and not values[field]:
            return None
    # A year introduced by "of/from/של" is not a customer or topic.
    relation_remaining = str(query)
    if basis:
        relation_remaining = re.sub(
            r'(?i)(?:(?<!\w)(?:for|on|by|of|לפי|על\s+פי|ל|ב)\s+)?'
            + _DATE_ROLE_PATTERNS[basis], ' ', relation_remaining, count=1)
    if values['year']:
        relation_remaining = re.sub(
            r'(?i)(?<!\w)(?:של|of|from)\s+' + re.escape(values['year']) + r'(?!\d)',
            ' ', relation_remaining, count=1)
    for value in values.values():
        if value:
            relation_remaining = re.sub(re.escape(value), ' ', relation_remaining, count=1, flags=re.I)
    # Require a relation field when the request contains a relationship cue.
    if re.search(r'(?i:\b(?:for|from|customer|client|project|collection|about|related)\b|(?<!\w)(?:עבור|לקוח|לקוחה|פרויקט|אוסף|לגבי|אודות|של|שקשור\w*)(?!\w))', relation_remaining) and not any(values[k] for k in ('customer','project','topic','collection')):
        return None
    return DocumentTypeListing(proposal['kind'], **values, date_basis=basis,
                               topic_groups=tuple(normalized_groups))


def primary_document_classifications(text):
    """Recognize complete primary headings, including unnumbered headings.

    A title embedded in a prose sentence is never a classification. Numbered
    and conflicting primary titles preserve the existing identity semantics.
    """
    numbered = primary_document_identities(text)
    value = str(text or '')
    found = list(numbered)
    for line in re.finditer(r'[^\n]+', value):
        body = line.group().strip()
        if _BODY_HEADING.fullmatch(body):
            break
        match = re.fullmatch(r'(?:(?:הנדון|נושא|subject)\s*[:\-]\s*)?(?P<kind>'+DOCUMENT_KIND_PATTERN+r')(?:\s+(?:original|copy|מקור|העתק))?(?:\s+(?:עבור|לצורך|בנושא|for|regarding)\s+[^\n]+)?\s*[.:]?', body, re.I)
        if match:
            start = line.start()+len(line.group())-len(line.group().lstrip())
            if any(item.span[0] <= start < item.span[1] for item in numbered):
                continue
            kind = canonical_document_kind(match['kind'])
            # A descriptive subheading such as "invoice for ..." is not a
            # second primary title when its role is already established by a
            # numbered compound title. Distinct document kinds still conflict.
            if not any(role_matches(item.kind, kind) for item in found):
                found.append(DocumentIdentity(kind, '', (start, start+len(body)), (start, start)))
    return tuple(sorted(found, key=lambda item: item.span[0]))


def possible_document_type_mention(text, request):
    """A kind mention without a primary classification needs further reading."""
    return any(listing_kind_matches(canonical_document_kind(match.group()), request.kind)
               for match in re.finditer(DOCUMENT_KIND_PATTERN, str(text or ''), re.I))


def possible_primary_document_type_heading(text, request):
    """Keep malformed title-like financial lines uncertain, not prose mentions.

    This only assesses whether a source might have a primary title; it never
    establishes a document identity or treats a filename as evidence.
    """
    header = re.compile(
        r'(?i)^\s*(?:(?:הנדון|נושא|subject)\s*[:\-]\s*)?'
        r'(?:(?:original|copy|מקור|העתק)\s+)?'
        r'(?:[A-Za-z0-9][A-Za-z0-9/-]*\s+)?(?:ה)?'
        r'(?P<kind>' + DOCUMENT_KIND_PATTERN + r')(?!\w)(?:\s|[:#./\-–—]|$)')
    for raw in str(text or '').splitlines():
        line = raw.strip()
        if not line:
            continue
        words = re.findall(r'[^\W_]+', line, re.UNICODE)
        if len(words) > 3 and re.search(r'[.!?]\s*$', line):
            continue
        match = header.match(line)
        if match and listing_kind_matches(match.group('kind'), request.kind):
            return True
    return False


def alternate_primary_document_classification(text):
    """Recognize affirmative nonfinancial document headings in OCR title text.

    Only a clear title in the opening material is evidence; absence of a
    financial phrase is never used as proof. Conflicting financial headings or
    multiple alternate titles remain unresolved.
    """
    value = str(text or '')
    if primary_document_classifications(value):
        return ''
    opening = value[:1800]
    kinds = []
    headings = (
        ('tender_invitation',
         r'(?m)^(?:(?:[^\n.!?]{1,100}\s+)?הזמנה\s+(?:מס(?:פר)?|מט[\u0027\u05f3.]?)\s*[\d/.-]+\s+)?'
         r'הזמנה\s+להציע\s+הצעות\s+(?:למתן|לקבלת|לביצוע)'
         r'|^\s*invitation\s+to\s+(?:tender|bid)(?:\s*[:\-].*|\s*$)'),
        ('business_registration',
         r'(?m)^\s*(?:העתק\s+)?תעודת\s+עוסק\s+מורשה\s*$'
         r'|^\s*(?:(?:business|tax)\s+registration\s+certificate|certificate\s+of\s+incorporation)\s*$'),
    )
    for kind, pattern in headings:
        if re.search(pattern, opening, re.I):
            kinds.append(kind)
    return kinds[0] if len(kinds) == 1 else ''
