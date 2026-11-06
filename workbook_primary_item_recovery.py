"""Recover a whole source description after a uniquely anchored overview claim.

The generated claim locates a row only. No generated property is released;
the complete original field must independently pass the shared final assessor.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re

_NAME = re.compile(r'(?:product(?:\s+name)?|item(?:\s+name)?|מוצר|פריט)', re.I)
_DESCRIPTION = re.compile(r'(?:description|details|specification|תיאור(?:\s+מפורט)?|תאור(?:\s+מפורט)?|מפרט)', re.I)
_QUANTITY = re.compile(r'(?:qty|quantity|count|כמות)', re.I)
_IDENTIFIER = re.compile(r'(?=[A-Za-z0-9._/-]*[A-Za-z])(?=[A-Za-z0-9._/-]*\d)[A-Za-z0-9][A-Za-z0-9._/-]{1,80}')
_WORD = re.compile(r'[A-Za-z0-9]+(?:[./-][A-Za-z0-9]+)*|[א-ת]+')
_BOUNDARY = re.compile(r'\b(?:with|without|including|includes?|and|or|for|of|connected)\b|'
                       r'(?<![א-ת])(?:עם|ללא|בלי|כולל|כוללת|או|מחובר|מחוברת)(?![א-ת])|'
                       r'(?<![א-ת])ו(?=[א-ת]{2,})', re.I)
_INTRO = re.compile(
    r'^(?:הפרויקט|הציוד|המסמך|הגיליון|the\s+project|the\s+equipment|'
    r'the\s+document|the\s+worksheet|the\s+workbook)\s+'
    r'(?:כולל|כוללת|כוללים|כוללות|מפרט|מציין|includes?|contains?|lists?)\s+', re.I)
_NUMBER = re.compile(r'\d+(?:[.,]\d+)?')


def recover_primary_source_field(body, identities, evidence, hebrew, assess):
    """Return at most one complete field with its independently checked citation.

    A unique primary product identifier, a literal leading source item phrase,
    a source-proven positive quantity and complete sibling-field accounting
    identify the row. The candidate wording itself remains rejected/unresolved.
    """
    from workbook_project import WorkbookProjectEvidence, _fields
    from workbook_context_recovery import (_recovery_row_proof, _ITEM_FIELD,
                                           _PRICE_FIELD, _SCALAR, _FORMULA)
    from passage_synthesis import _workbook_word_forms

    if (not isinstance(evidence, WorkbookProjectEvidence)
            or not getattr(evidence, 'overview_synthesis', False)):
        return ()
    ids = tuple(identities)
    if len(ids) != 1:
        return ()
    proof = _recovery_row_proof(ids, evidence, hebrew, assess, require_quantity=True)
    if not proof:
        return ()
    row, fields, accounted = proof
    names = [value.strip() for label, _, value, _, _ in fields if _NAME.fullmatch(label.strip())]
    descriptions = [value.strip() for label, _, value, _, _ in fields
                    if _DESCRIPTION.fullmatch(label.strip())]
    quantities = [value.strip() for label, _, value, _, _ in fields
                  if _QUANTITY.fullmatch(label.strip())]
    if (len(names) != 1 or len(descriptions) != 1 or len(quantities) != 1
            or not _IDENTIFIER.fullmatch(names[0])):
        return ()
    # Recognized scalar price cells may be present beside the item. They are
    # classified, never emitted or used as a total or a property of an item.
    def accounted_sibling(label, value):
        if _PRICE_FIELD.fullmatch(label.strip()):
            return bool(_SCALAR.fullmatch(value.strip()) or _FORMULA.fullmatch(value.strip()))
        return accounted(label, value)
    if any(not (_NAME.fullmatch(label.strip()) or _DESCRIPTION.fullmatch(label.strip())
                or accounted_sibling(label, value))
           for label, _, value, _, _ in fields if value.strip()):
        return ()
    name, description, quantity = names[0], descriptions[0], quantities[0]
    if len(description) > 300 or '\n' in description:
        return ()

    identifier = re.compile(r'(?<![\w./-])' + re.escape(name) + r'(?![\w./-])', re.I)
    owners = {p.citation_id for p in evidence.passages
              for label, _, value, _, _ in _fields(p.source_claim)
              if _ITEM_FIELD.fullmatch(label.strip()) and identifier.search(value)}
    if owners != {ids[0]}:
        return ()

    text = _INTRO.sub('', str(body).strip())
    count = re.match(r'^(\d+(?:\.\d+)?)\s+', text)
    if count:
        try:
            if Decimal(count.group(1)) != Decimal(quantity.replace(',', '')):
                return ()
        except InvalidOperation:
            return ()
        text = text[count.end():]

    # The model identifier follows the primary head with an explicit model
    # marker. A code after an accessory or an extended identifier is not an
    # independently established association.
    match = re.fullmatch(r'(?P<head>.+?)\s+(?:מדגם|דגם|מסוג|model)\s+'
                         + re.escape(name) + r'\s*[.!?]?', text, re.I)
    if not match:
        return ()
    head = match.group('head')
    if _BOUNDARY.search(head):
        return ()
    wanted, observed = _WORD.findall(head), _WORD.findall(description)
    if not 2 <= len(wanted) <= 12 or len(wanted) > len(observed):
        return ()
    if not all(_workbook_word_forms(a) & _workbook_word_forms(b)
               for a, b in zip(wanted, observed)):
        return ()
    # Do not jump past a punctuation boundary in the original description:
    # the entire primary head must lie in the first source clause.
    first_clause = re.split(r'[,;\n]', description, maxsplit=1)[0]
    if len(_WORD.findall(first_clause)) < len(wanted):
        return ()
    result = assess(description + ' [' + ids[0] + ']', evidence, hebrew,
                    check_coverage=False)
    if not result.accepted:
        return ()
    return ((description, ids),)

