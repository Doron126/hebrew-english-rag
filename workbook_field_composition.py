"""Conservative field-role proof for one workbook overview claim.

This helper proves only an explicit product/description/quantity envelope.
It is not a general parser for descriptions, a semantic judge, or an
alternative to original-source, qualification, numeric and final gates.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re

_NAME = re.compile(r'(?:product(?:\s+name)?|item(?:\s+name)?|שם\s+מוצר|שם\s+פריט|מוצר|פריט)', re.I)
_DESCRIPTION = re.compile(r'(?:description|details|specification|תיאור(?:\s+מפורט)?|תאור(?:\s+מפורט)?|מפרט)', re.I)
_QUANTITY = re.compile(r'(?:quantity|qty|count|כמות)', re.I)
_PRICE = re.compile(r'(?:unit\s+price|total\s+price|line\s+total|price\s+per\s+unit|מחיר\s+יחידה|מחיר\s+כולל|סה.?כ\s+מחיר)', re.I)
_IDENTIFIER = re.compile(r'[A-Za-z0-9][A-Za-z0-9._/-]{1,80}')
_NUMBER = re.compile(r'(?:0|[1-9][0-9]*)(?:\.[0-9]+)?')
# A labeled, textual finish/color suffix is independently identifiable as a
# simple item attribute. It may be omitted without moving a number, condition
# or a second item's property into the name. No free-text clipping is allowed.
_NOMINAL_WITH_ATTRIBUTE = re.compile(
    r'^(?P<nominal>[^()]+?)\s*\((?:colou?r|finish|צבע|גימור)\s*:\s*'
    r'[A-Za-zא-ת]+\)$', re.I)
_FORMULA = re.compile(r'Formula:\s*(?:\[[^\]\n]{1,80}\]|=[^\n;]{1,180});\s*Cached value:\s*[+\-]?\d+(?:[.,]\d+)*', re.I)

# Do not infer that a nested item's property belongs to the primary item.
# Full, atomic nominal descriptions are useful without a general dependency
# parser. Clause and relation markers are deliberately rejected rather than
# clipped, even where their meaning might sometimes be harmless.
_AMBIGUOUS = re.compile(
    r'\b(?:with|without|including|includes?|contains?|containing|for|of|'
    r'and|or|optional|example|such\s+as|if|unless|when|provided|'
    r'available|subject|only|except|warranty|not|no|requires?|'
    r'connected|controls|on|attached|mounted|plus|together|alongside|'
    r'installed|completed|delivered|purchased|paid|'
    r'approved|price|cost|payment|tax)\b|'
    r'(?<![א-ת])(?:עם|ללא|בלי|כולל|כוללת|כוללים|כוללות|או|אם|כאשר|'
    r'בתנאי|בכפוף|בתוספת|בנוסף|לצד|מחובר|מחוברת|'
    r'אופציונלי|לדוגמה|למשל|כדוגמת|אחריות|נדרש|מותנה|לא|רק|'
    r'הותקן|נרכש|שולם|מחיר|עלות|תשלום)(?![א-ת])|'
    # Hebrew frequently attaches the conjunction directly to the next word.
    # Without a dependency parser, even a single extra item can make its
    # quantity, identifier or numeric property ambiguous. Fail closed.
    r'(?<![א-ת])ו[א-ת]{2,}(?![א-ת])|[,;\n]', re.I)


def _normal(value):
    return ' '.join(str(value).split()).casefold()


def _numeric_price(value):
    value = value.strip()
    return bool(re.fullmatch(r'[+\-]?\d+(?:[.,]\d+)*', value) or _FORMULA.fullmatch(value))


def _role_fields(passage):
    """Return unique safe table roles, never values from an unknown column."""
    from workbook_project import _fields, _QUALIFIER
    roles = {'name': [], 'description': [], 'quantity': []}
    fields = _fields(passage.source_claim)
    if not fields or len({column for _, column, _, _, _ in fields}) != len(fields):
        return None
    for label, _column, value, _start, _end in fields:
        label, value = label.strip(), value.strip()
        if not value:
            return None
        if _NAME.fullmatch(label):
            roles['name'].append(value)
        elif _DESCRIPTION.fullmatch(label):
            roles['description'].append(value)
        elif _QUANTITY.fullmatch(label):
            roles['quantity'].append(value)
        elif _PRICE.fullmatch(label) and _numeric_price(value):
            continue
        else:
            # Unknown notes, identifiers or conditional price cells may
            # qualify this row. They cannot be silently stripped.
            return None
    if any(len(roles[role]) != 1 for role in roles):
        return None
    name, description, quantity = (roles[key][0] for key in ('name', 'description', 'quantity'))
    if (not _IDENTIFIER.fullmatch(name) or not re.search(r'[A-Za-z]', name)
            or not re.search(r'\d', name)):
        return None
    if not _NUMBER.fullmatch(quantity):
        return None
    try:
        if Decimal(quantity) <= 0:
            return None
    except InvalidOperation:
        return None
    if (len(description) > 300 or len(description.split()) < 2
            or _AMBIGUOUS.search(description) or _QUALIFIER.search(description)):
        return None
    return name, description, quantity


def workbook_field_composition_supported(body, identities, evidence):
    """Prove a complete simple item envelope from three source-owned fields.

    Exactly one eligible row and unique recognized table roles are required.
    The accepted wording contains a literal product identifier, a literal
    complete atomic description and the exact positive quantity. No field
    snippets, invented noun, action, property or relationship are admitted.
    The caller must still run all normal source/qualification/number/graph
    checks, including any negative semantic verdict already bound to a claim.
    """
    from workbook_project import WorkbookProjectEvidence

    if not isinstance(evidence, WorkbookProjectEvidence) or not evidence.overview_synthesis:
        return False
    ids = tuple(identities)
    passages = tuple(evidence.passages)
    by_id = {p.citation_id: p for p in passages}
    if (len(ids) != 1 or len(by_id) != len(passages) or ids[0] not in by_id
            or (evidence.quote_eligible_ids is not None and ids[0] not in evidence.quote_eligible_ids)):
        return False
    row = by_id[ids[0]]
    metadata = getattr(row, 'metadata', {})
    if (not getattr(row, 'source_id', None) or not getattr(row, 'tenant_id', None)
            or not metadata.get('sheet') or metadata.get('_workbook_scope') is None):
        return False
    if evidence.tenant_id and evidence.tenant_id != row.tenant_id:
        return False
    roles = _role_fields(row)
    if not roles:
        return False
    name, description, quantity = roles
    # An identifier repeated in the same selected source and sheet does not
    # uniquely bind a primary row, even if one of the repeated rows would fit.
    for other in passages:
        if other is row or other.source_id != row.source_id or other.tenant_id != row.tenant_id:
            continue
        other_meta = getattr(other, 'metadata', {})
        if (other_meta.get('sheet'), other_meta.get('_workbook_scope')) != (
                metadata.get('sheet'), metadata.get('_workbook_scope')):
            continue
        from workbook_project import _fields
        if any(_NAME.fullmatch(label.strip()) and _normal(value) == _normal(name)
               for label, _column, value, _start, _end in _fields(other.source_claim)):
            return False
    descriptions = [description]
    nominal = _NOMINAL_WITH_ATTRIBUTE.fullmatch(description)
    if nominal:
        head = nominal.group('nominal').strip()
        # The source explicitly scopes a single, nonnumeric labeled attribute
        # to this item; no words inside the nominal phrase can be skipped.
        if len(head.split()) >= 2 and not re.search(r'\d', description):
            descriptions.append(head)
    candidates = []
    for phrase in descriptions:
        candidates.extend((
            f'המסמך מפרט {quantity} יחידות של {name}, שהם {phrase}',
            f'המסמך מפרט {quantity} יחידות של {name}, שהן {phrase}',
            f'המסמך מפרט {name}: {phrase} (כמות: {quantity})',
            f'the document lists {quantity} units of {name}, which are {phrase}',
            f'the document lists {quantity} units of {name}: {phrase}',
            f'the document lists {name}: {phrase} (quantity: {quantity})',
        ))
    return _normal(body).rstrip('.!?') in {_normal(candidate).rstrip('.!?') for candidate in candidates}
