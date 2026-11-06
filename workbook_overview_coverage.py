"""Estimate overview coverage from the evidence actually bound to accepted claims.

This is a source-structural estimate, not a semantic proof. It never validates
a claim: callers must first apply the normal source and fact checks. It looks
for breadth at the overview level, rather than requiring every listed item.
In particular, translated text cannot be checked for thematic meaning by this
estimator; sufficiently substantive, broadly bound translated text is a
signal, not proof that the translations or coverage are correct.
"""
from __future__ import annotations

import re
import math

from workbook_project import _fields, _PRICE, _QUANTITY, _LINK_LABEL, _OWNER_LABEL

# Grammatical or table boilerplate only; this list does not classify equipment.
_COMMON = frozenset(('the a an of for and or with from in to on includes including '
    'supply installation unit project item product description general detailed '
    'הפרויקט הפריטים תיאור תאור מפורט כללי כולל כוללת כולליה וכן של את עם או '
    'אספקה התקנה דגם מקט יחידה מוצר פריט').split())
_WORD = re.compile(r'[A-Za-z]+|[א-ת]+')
_SIGNATURE_TOKEN = re.compile(r'[A-Za-z0-9]+(?:[-./][A-Za-z0-9]+)*|[א-ת]+')
_NOTE_LABEL = re.compile(r'\b(?:notes?|remarks?|comments?)\b|הער(?:ה|ות)', re.I)
_ITEM_LABEL = re.compile(r'\b(?:product(?:\s+name)?|items?|equipment|devices?|parts?|'
    r'description|details|specification|scope|work|service)\b|'
    r'(?:תי?אור|פריט|מוצר|ציוד|עבודה|שירות|מפרט)', re.I)


def _terms(text, limit=12):
    terms = []
    for match in _WORD.finditer(text.casefold()):
        word = match.group()
        # Ignore isolated fragments of model codes (letters touching digits).
        before = text[match.start()-1:match.start()]
        after = text[match.end():match.end()+1]
        if before.isdigit() or after.isdigit() or len(word) < 3:
            continue
        if word not in _COMMON:
            terms.append(word)
    return tuple(terms[:limit])


def _content_values(passage):
    raw = passage.source_claim
    if not raw.startswith('Spreadsheet row '):
        return ()  # headings/currency context are not inventory rows
    fields = _fields(raw)
    descriptions = []
    other = []
    for label, _column, value, _, _ in fields:
        if (_PRICE.search(label) or _QUANTITY.search(label) or _LINK_LABEL.search(label)
                or _OWNER_LABEL.search(label) or _NOTE_LABEL.search(label)
                or 'Formula:' in value):
            continue
        if re.search(r'\b(?:description|details|specification)\b|תיאור|תאור|מפרט', label, re.I):
            descriptions.append(value)
        elif _ITEM_LABEL.search(label):
            other.append(value)
        elif not re.fullmatch(r'[A-Z]{1,3}', label):
            # Unknown labeled text may be material, but it is not a known
            # inventory field. It cannot establish overview completeness.
            return None
        else:
            # A column without a schema label is ambiguous rather than proof
            # of either an item or an unrelated note.
            return None
    # The description generally states the item; a product-code column by
    # itself does not provide coverage for an otherwise omitted description.
    values = descriptions or other
    return tuple(values)


def _signature(values):
    """Use a set of description words, preserving polarity/qualification words.

    Numeric and mixed alphanumeric identifiers are omitted only for grouping
    variants in an overview. Factual and numeric validation is independent.
    """
    return frozenset(token.casefold() for value in values
                     for token in _SIGNATURE_TOKEN.findall(value)
                     if not any(char.isdigit() for char in token))


def _groups(passages):
    # Merge descriptions with the same normalized words in any order.
    # Digits/model-code variants can differ; qualifiers and item roles cannot.
    groups = []
    for passage in passages:
        values = _content_values(passage)
        if values is None:
            return None
        terms = _terms(' '.join(values))
        signature = _signature(values)
        if not terms:
            continue
        matching = [i for i, group in enumerate(groups)
                    if signature == group['signature']]
        if not matching:
            groups.append({'terms': [terms], 'ids': {passage.citation_id},
                           'signature': signature})
            continue
        first = matching[0]
        groups[first]['terms'].append(terms)
        groups[first]['ids'].add(passage.citation_id)
        for index in reversed(matching[1:]):
            groups[first]['terms'].extend(groups[index]['terms'])
            groups[first]['ids'].update(groups[index]['ids'])
            del groups[index]
    return groups


def _forms(word):
    """Conservative singular/plural and Hebrew prefix forms, not translation."""
    from passage_synthesis import _workbook_word_forms
    forms = set(_workbook_word_forms(word))
    if re.fullmatch(r'[א-ת]{5,}', word) and word[0] in 'והבלכמ':
        forms.update(_workbook_word_forms(word[1:]))
    if re.fullmatch(r'[a-z]{4,}s', word):
        forms.add(word[:-1])
    return forms


def _binding_mentions_group(group, groups, bindings):
    # A reference is not itself a coverage claim. A distinguishing term can
    # indicate a topic, as can two shared source-derived terms: the latter lets
    # related rows be described together without requiring each variant's own
    # word. A single common word and a citation list are insufficient.
    own = set().union(*(_forms(word) for row in group['terms'] for word in row))
    other = set().union(*(_forms(word) for candidate in groups if candidate is not group
                          for row in candidate['terms'] for word in row))
    distinctive = own - other
    for text, ids in bindings:
        if group['ids'].isdisjoint(ids):
            continue
        words = _terms(re.sub(r'\[\d+\]', '', text), limit=None)
        if any(_forms(word) & distinctive for word in words):
            return True
        answer_forms = set().union(*(_forms(word) for word in words))
        if any(sum(bool(_forms(word) & answer_forms) for word in set(row)) >= 2
               for row in group['terms']):
            return True
    return False


def _translated_breadth(groups, bindings, passages):
    """Estimate broad translated content without pretending to translate it.

    Require substantive answer text, multiple bound groups, and differing
    dominant scripts. A citation list, a terse installation clause, or a
    number alone cannot establish breadth. This is only an estimate: it cannot
    detect omissions or mistakes within a fluent translated paragraph.
    """
    by_id = {p.citation_id: p for p in passages}
    relevant = [(text, set(ids) & set(by_id)) for text, ids in bindings
                if set(ids) & set(by_id)]
    ids = set().union(*(ids for _text, ids in relevant))
    referenced = [group for group in groups if not group['ids'].isdisjoint(ids)]
    if len(referenced) < min(2, len(groups)):
        return 0
    text = ' '.join(text for text, _ids in relevant)
    words = _terms(re.sub(r'\[\d+\]', '', text), limit=None)
    if len(set(words)) < 5:
        return 0
    source = ' '.join(' '.join(_content_values(by_id[i]) or ()) for i in ids)
    source_he = len(re.findall(r'[א-ת]', source))
    source_en = len(re.findall(r'[A-Za-z]', source))
    text_he = len(re.findall(r'[א-ת]', text))
    text_en = len(re.findall(r'[A-Za-z]', text))
    if (source_en > source_he * 2 and text_he > text_en * 2 or
            source_he > source_en * 2 and text_en > text_he * 2):
        return len(referenced)
    return 0


def workbook_overview_coverage(evidence, bindings):
    """Estimate whether supported bindings cover the source's major scope.

    Each selected source/sheet/owned section with substantive rows must show
    several supported topics where available. A bounded breadth threshold
    permits related rows and incidental details to be omitted. Lexical matches
    and, for different scripts, substantive bound prose are imperfect signals;
    neither proves that the major themes were actually described.
    Source read completeness, rejected material claims, qualifications, and
    truncation remain the caller's independent responsibilities.
    """
    passages = tuple(getattr(evidence, 'passages', ()) or ())
    by_id = {p.citation_id: p for p in passages}
    represented = {identity for _text, ids in bindings for identity in ids}
    if not represented or not represented <= set(by_id):
        return False
    sections = {}
    for passage in passages:
        key = (passage.source_id, passage.metadata.get('sheet'),
               passage.metadata.get('_workbook_scope'))
        sections.setdefault(key, []).append(passage)
    substantive = [(_groups(section), section) for section in sections.values()]
    if any(group is None for group, _section in substantive):
        return False
    substantive = [(group, section) for group, section in substantive if group]
    if not substantive:
        return False
    for groups, section in substantive:
        covered = sum(_binding_mentions_group(group, groups, bindings) for group in groups)
        covered = max(covered, _translated_breadth(groups, bindings, section))
        required = min(4, max(min(2, len(groups)), math.ceil(len(groups) / 2)))
        if covered < required:
            return False
    # A proven aggregate cost has a different role from line-item prices.
    # When available it is part of the requested source overview.
    from workbook_project import workbook_overview_total
    total = workbook_overview_total(evidence, False)
    if total is not None and not set(total[1]) <= represented:
        return False
    return True
