"""Fail closed for explicit same-row exclusions from a workbook's scope."""
import re


_EXPLICIT_EXCLUSION = re.compile(
    r'\b(?:outside\s+(?:the\s+)?scope|out[\s-]+of[\s-]+(?:the[\s-]+)?scope|'
    r'excluded\s+from\s+(?:the\s+)?scope|not\s+(?:within|in|included\s+in)\s+'
    r'(?:the\s+)?scope)\b|'
    r'(?<![א-ת])(?:מחוץ\s+(?:לתחום|להיקף|לתכולת|לתכולה)|'
    r'לא\s+(?:נכלל|נכללת|נכללים|נכללות)\s+(?:בתחום|בהיקף|בתכולה))(?![א-ת])',
    re.I)
_SCOPE_LABEL = re.compile(r'^(?:scope|work\s+scope|תכולה|היקף|תחום)$', re.I)
_SCOPE_VALUE = re.compile(
    r'^(?:excluded|not\s+included|outside|out[\s-]+of[\s-]+scope|'
    r'לא\s+(?:נכלל|נכללת|נכללים|נכללות)|מחוץ\s+לתחום)[.!]?$', re.I)


def _exclusion_fields(passage):
    from workbook_project import _fields
    return tuple(value for label, _column, value, _start, _end in _fields(passage.source_claim)
                 if _EXPLICIT_EXCLUSION.search(value) or
                 (_SCOPE_LABEL.fullmatch(label.strip()) and _SCOPE_VALUE.fullmatch(value.strip())))


def excluded_rows_safe_in_excerpt(evidence, units):
    """Every displayed fragment of an excluded row retains its own exclusion."""
    passages = {p.citation_id: p for p in evidence.passages}
    if len(passages) != len(evidence.passages):
        return False
    for unit in units:
        for identity in unit.citation_ids:
            if identity not in passages:
                return False
            text = ' '.join(unit.text.split()).casefold()
            if any(' '.join(value.split()).casefold() not in text
                   for value in _exclusion_fields(passages[identity])):
                return False
    return True


def excluded_row_allows_claim(body, identities, passages):
    """Do not turn an excluded row into included work.

    A configuration-wide note independently designated by the producer can
    still be quoted exactly. It has its own meaning and does not adopt the
    adjacent row's exclusion. Other claims using the excluded row fail
    closed; no scope or cross-row ownership is inferred.
    """
    from workbook_project import _fields

    for identity in identities:
        passage = passages[identity]
        fields = _fields(passage.source_claim)
        if not _exclusion_fields(passage):
            continue
        designated = passage.metadata.get('_workbook_general_qualification_fields', ())
        if (len(identities) == 1 and isinstance(designated, (tuple, list)) and
                any(' '.join(body.split()) == ' '.join(str(note).split()) and
                    any(' '.join(value.split()) == ' '.join(str(note).split())
                        for _label, _column, value, _start, _end in fields)
                    for note in designated)):
            continue
        return False
    return True
