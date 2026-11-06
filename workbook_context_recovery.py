"""Recover one exact workbook field after a source-bound context preamble."""
import re
import unicodedata


_ITEM_FIELD = re.compile(
    r'(?:product(?:\s+name)?|item(?:\s+name)?|description|details|specification|'
    r'equipment|device|work|service|מוצר|פריט|ציוד|מפרט|עבודה|שירות|'
    r'תיאור(?:\s+(?:כללי|מפורט))?|תאור(?:\s+(?:כללי|מפורט))?)', re.I)
_SCALAR_FIELD = re.compile(
    r'(?:qty|quantity|serial(?:\s+number)?|כמות|מספר\s+סדורי|מס.?\s*סדורי)', re.I)
_PRICE_FIELD = re.compile(
    r'(?:price|unit\s+price|total\s+price|מחיר(?:\s+יחידה)?|'
    r'סה.?כ\s+(?:מחיר|בש.?ח))', re.I)
_NUMBER = r'[+\-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
_SCALAR = re.compile(r'\s*(?:[₪$€£]\s*' + _NUMBER + r'|' + _NUMBER +
                     r'\s*[₪$€£]|' + _NUMBER + r')\s*')
_FORMULA = re.compile(
    r'(?is)^formula:\s*(?:\[shared formula\]|=[A-Z0-9+*/(). -]+);\s*'
    r'cached\s+value:\s*' + _NUMBER + r'\s*$')
_PREAMBLE = re.compile(
    r'^\s*(?:המסמך|הפרויקט|הגיליון)\s+(?:מפרט|מציג|כולל)\s+'
    r'ציוד\s+עבור\s+(.+?)\s+(?:הכולל|הכוללת|כולל|כוללת)\s*$', re.I)


def _context_is_bound(context, metadata):
    """Require the whole owner and section selected by the ordinal producer."""
    from subject_intent import ordinal_workbook_section_subject
    from workbook_project import _ordinal_declared_owner
    subject = metadata.get('_workbook_subject', '')
    request = ordinal_workbook_section_subject(subject)
    headings = tuple(metadata.get('_workbook_heading', ()))
    if (not request or metadata.get('_workbook_scope') != request.ordinal
            or not 1 <= len(headings) <= 8
            or any(not isinstance(value, str) or len(value) > 500 for value in headings)):
        return False
    def normalize(value):
        return ' '.join(unicodedata.normalize('NFKC', value).casefold().split())
    if not any(normalize(heading) == normalize(request.section) for heading in headings):
        return False
    owners = [_ordinal_declared_owner(heading, request, metadata) for heading in headings
              if normalize(heading) != normalize(request.section)]
    owners = [owner for owner in owners if owner is not None]
    if not owners or any(normalize(owner) != normalize(request.owner) for owner in owners):
        return False
    # The preposition is permitted only on the complete bound owner; it may
    # not turn a suffix of an institution name into an asserted location.
    expected = normalize(request.section + ' ב' + request.owner)
    return normalize(context) == expected


def _recovery_row_proof(identities, evidence, hebrew, assess, *,
                        require_quantity=False, allow_designated_qualification=False):
    """Bind one source row and supply its exact sibling-field accounting."""
    from decimal import Decimal, InvalidOperation
    from workbook_project import _fields, workbook_overview_total
    if not hebrew or not getattr(evidence, 'overview_synthesis', False):
        return None
    ids = tuple(identities)
    by_id = {p.citation_id: p for p in evidence.passages}
    eligible = getattr(evidence, 'quote_eligible_ids', None)
    if (len(ids) != 1 or len(by_id) != len(evidence.passages) or ids[0] not in by_id
            or (eligible is not None and ids[0] not in eligible)
            or not evidence.complete or not evidence.selection_complete
            or len({p.source_id for p in evidence.passages}) != 1):
        return None
    row = by_id[ids[0]]
    metadata = row.metadata
    if (not row.source_id or not row.tenant_id or not metadata.get('sheet')
            or metadata.get('_workbook_scope') is None):
        return None
    fields = _fields(row.source_claim)
    # A single original worksheet row cannot have two distinct cells at the
    # same column. A duplicate may be embedded text mimicking a field marker;
    # never guess which occurrence owns a qualification.
    if len({column for _, column, _, _, _ in fields}) != len(fields):
        return None
    def proven_total():
        # The ordinal producer derives this marker only after checking every
        # line's quantity × price against its cached amount and the complete
        # sum. The FINAL path independently rebuilds those original rows.
        try:
            total = workbook_overview_total(evidence, hebrew)
        except (InvalidOperation, TypeError, ValueError):
            return False
        if not total or len(total[1]) != 1 or total[1][0] not in by_id:
            return False
        total_id = total[1][0]
        if eligible is not None and total_id not in eligible:
            return False
        total_row = by_id[total_id]
        return ((total_row.source_id, total_row.tenant_id, total_row.metadata.get('sheet'),
                 total_row.metadata.get('_workbook_scope')) ==
                (row.source_id, row.tenant_id, metadata.get('sheet'), metadata.get('_workbook_scope')))
    quantities = [value.strip() for label, _, value, _, _ in fields
                  if re.fullmatch(r'(?:qty|quantity|count|כמות)', label.strip(), re.I)]
    if require_quantity and not quantities:
        return None
    if quantities:
        try:
            if (len(quantities) != 1 or not re.fullmatch(_NUMBER, quantities[0])
                    or not Decimal(quantities[0].replace(',', '')).is_finite()
                    or Decimal(quantities[0].replace(',', '')) <= 0):
                return None
        except InvalidOperation:
            return None
    designated = metadata.get('_workbook_general_qualification_fields', ())
    qualification_ids = metadata.get('_workbook_general_qualification_ids', ())
    if allow_designated_qualification and (
            not isinstance(designated, (tuple, list)) or not isinstance(qualification_ids, (tuple, list))):
        return None

    def accounted(other_label, other):
        if ((_SCALAR_FIELD.fullmatch(other_label.strip())
             or (_PRICE_FIELD.fullmatch(other_label.strip()) and proven_total()))
                and (_SCALAR.fullmatch(other.strip()) or _FORMULA.fullmatch(other.strip()))):
            return True
        return (allow_designated_qualification and ids[0] in qualification_ids
                and any(isinstance(item, str) and ' '.join(item.split()) == ' '.join(other.split())
                        for item in designated)
                and assess(other.strip() + ' [' + ids[0] + ']', evidence, hebrew,
                           check_coverage=False).accepted)
    return row, fields, accounted


def recover_context_owned_field(body, identities, evidence, hebrew, assess):
    """Release one exact source field, not the rejected contextual claim.

    Every context phrase comes from the cited row's authenticated headings.
    Unknown row cells make recovery unavailable. The caller retains the
    original rejected claim as unresolved.
    """
    ids = tuple(identities)
    proof = _recovery_row_proof(ids, evidence, hebrew, assess)
    if not proof:
        return ()
    row, fields, accounted = proof
    metadata = row.metadata
    headings = tuple(metadata.get('_workbook_heading', ()))
    if not headings:
        return ()
    text = body.strip()
    matches = set()
    for field_index, (label, column, value, _, _) in enumerate(fields):
        value = value.strip()
        if not _ITEM_FIELD.fullmatch(label.strip()) or not value or not text.endswith(value):
            continue
        prefix = text[:-len(value)]
        if not prefix or not prefix[-1].isspace():
            continue
        parsed = _PREAMBLE.fullmatch(prefix)
        if not parsed or not _context_is_bound(parsed.group(1), metadata):
            continue
        # A different non-scalar field can contain an unrecognized material
        # condition or item. Do not leave it behind through this recovery.
        if any(other_index != field_index
               and not accounted(other_label, other)
               for other_index, (other_label, other_column, other, _, _) in enumerate(fields)
               if other.strip()):
            continue
        if assess(value + ' [' + ids[0] + ']', evidence, hebrew,
                  check_coverage=False).accepted:
            matches.add(value)
    return ((matches.pop(), ids),) if len(matches) == 1 else ()



def recover_exact_item_with_context(body, identities, evidence, hebrew, assess):
    """Recover exact item/qualification fields from a strictly bounded proposal.

    A complete source item followed by an explicit quantity suffix, or a
    complete item followed by its exact designated section qualification, is
    split into independently assessed source fields. The generated quantity or
    relationship is never copied, and the caller retains the original
    rejected claim as unresolved.
    """
    from decimal import Decimal, InvalidOperation
    from workbook_project import WorkbookProjectEvidence

    if not isinstance(evidence, WorkbookProjectEvidence) or not evidence.overview_synthesis:
        return ()
    ids = tuple(identities)
    if len(ids) != 1:
        return ()
    proof = _recovery_row_proof(ids, evidence, hebrew, assess,
                                require_quantity=True, allow_designated_qualification=True)
    if not proof:
        return ()
    row, fields, accounted = proof
    metadata = row.metadata
    designated = metadata.get('_workbook_general_qualification_fields', ())
    qualification_ids = metadata.get('_workbook_general_qualification_ids', ())
    if (not isinstance(designated, (tuple, list))
            or not isinstance(qualification_ids, (tuple, list))):
        return ()
    def norm_space(value):
        return ' '.join(str(value).split())
    designated_values = tuple(norm_space(value) for value in designated)
    if len(set(designated_values)) != len(designated_values):
        return ()
    if designated_values and (len(designated_values) != 1 or ids[0] not in qualification_ids
            or sum(norm_space(value) == designated_values[0] for _, _, value, _, _ in fields) != 1):
        return ()
    items = [(index, value.strip()) for index, (label, _, value, _, _) in enumerate(fields)
             if _ITEM_FIELD.fullmatch(label.strip()) and value.strip()
             and norm_space(value) not in designated_values]
    quantities = [value.strip() for label, _, value, _, _ in fields
                  if re.fullmatch(r'(?:qty|quantity|count|כמות)', label.strip(), re.I)]
    if len(items) != 1 or len(quantities) != 1:
        return ()
    item_index, item = items[0]
    quantity = quantities[0]
    if any(index != item_index and not accounted(label, value)
           for index, (label, _, value, _, _) in enumerate(fields) if value.strip()):
        return ()
    # Preserve all characters and words; normalize only spacing around commas
    # and duplicate adjacent comma delimiters introduced by source extraction.
    def punctuation(value):
        value = norm_space(value)
        value = re.sub(r'\s*,(?:\s*,)+\s*', ', ', value)
        return re.sub(r'\s*,\s*', ', ', value).strip()
    opening = re.sub(
        r'^(?:הפרויקט|הציוד|המסמך|הגיליון|the\s+project|the\s+equipment|'
        r'the\s+document|the\s+worksheet|the\s+workbook)\s+'
        r'(?:כולל|כוללת|כוללים|כוללות|מפרט|מציין|includes?|contains?|lists?)\s+',
        '', str(body).strip(), flags=re.I)
    selected = []
    match = re.fullmatch(r'(?P<item>.+?)\s+(?:בכמות|כמות|quantity)\s+'
                         r'(?P<quantity>\d+(?:\.\d+)?)\s*[.!?]?', opening, re.I)
    if match and punctuation(match.group('item')) == punctuation(item):
        try:
            if Decimal(match.group('quantity')) != Decimal(quantity.replace(',', '')):
                return ()
        except InvalidOperation:
            return ()
        selected = [item]
    elif (ids[0] in qualification_ids and len(designated_values) == 1):
        note = designated_values[0]
        # "כאשר" is not accepted as a newly proven causal relationship: it
        # only identifies two exact source fields, which are released apart.
        proposed = punctuation(opening)
        if proposed == punctuation(item + ', כאשר ' + note):
            selected = [item, next(value.strip() for _, _, value, _, _ in fields
                                   if norm_space(value) == note)]
    if not selected:
        return ()
    result = []
    for value in selected:
        decision = assess(value + ' [' + ids[0] + ']', evidence, hebrew,
                          check_coverage=False)
        if not decision.accepted:
            return ()
        result.append((value, ids))
    return tuple(result)
