"""Independent, conservative identity checks for semantic source proposals.

The model may suggest that an indexed source has the same meaning as a
request, but it cannot on its own discard a named owner, a specific item, or
an incident.  This guard reads the request and the authorized census again.
"""
from difflib import SequenceMatcher

from document_reading import source_body
from monetary_questions import _words, _word_variants, _variants, _GRAMMATICAL_HEADS


_INCIDENT_HEADS = frozenset({
    'תקלה', 'תיקון', 'התקנה', 'אספקה', 'החלפה',
    'fault', 'repair', 'installation', 'supply', 'replacement',
})
_ROLE_HEADS = frozenset({'טכנאי', 'טכנאית', 'technician'})
_SITE_HEADS = frozenset({'מוקד', 'מרכז', 'אתר', 'פרויקט', 'מועצה',
                         'project', 'site', 'station', 'center', 'centre'})
_GENERAL_HEADS = _GRAMMATICAL_HEADS | _ROLE_HEADS | frozenset({
    'fault', 'repair', 'installation', 'supply', 'replacement',
    'work', 'service', 'job', 'project', 'site', 'station', 'center', 'centre',
})


def _forms(text):
    return _word_variants(_words(text))


def _exact(forms, actual):
    return any(forms & observed for observed in actual)


def _close_words(forms, actual):
    result = set()
    for word in forms:
        if len(word) < 5:
            continue
        for variants in actual:
            for other in variants:
                if len(other) < 5 or word[0] != other[0]:
                    continue
                if SequenceMatcher(None, word, other).ratio() >= 0.84:
                    result.add(other)
    return result


def relevant_monetary_window(body, request, *, size=500):
    """Bound model context around request-owned entity and incident terms."""
    text = str(body or '')
    wanted = _word_variants(tuple(getattr(request, 'terms', ()) or ()))
    if not text or not wanted:
        return ''
    windows = [(offset, text[offset:offset + size])
               for offset in range(0, len(text), max(1, size // 2))]
    def score(entry):
        offset, window = entry
        actual = _forms(window)
        matched = sum(1 for term in wanted if _exact(term, actual) or _close_words(term, actual))
        # Later equally relevant context retains nearby payment/qualification
        # fields rather than spending the budget on repeated letterheads.
        return matched, offset
    return max(windows, key=score)[1]


def independently_matches_monetary_request(request, source, grouped):
    """Check all specific terms and prevent one-edit name collisions.

    A single spelling variation requires another exact identifying anchor.
    Competing spellings in the authorized corpus cause abstention.  A role
    such as technician may be implicit only if another explicit incident
    term is independently present; a separate labor request is excluded by
    the caller before this guard is reached.
    """
    if not request or source not in grouped:
        return False
    actual = _forms(source_body(grouped[source]))
    terms = _word_variants(tuple(request.terms))
    incidents = [forms for forms in terms if forms & _INCIDENT_HEADS]
    roles = [forms for forms in terms if forms & _ROLE_HEADS]
    sites = [forms for forms in terms if forms & _SITE_HEADS]
    if incidents and not all(_exact(forms, actual) for forms in incidents):
        return False
    if sites and not all(_exact(forms, actual) for forms in sites):
        return False
    if roles and not incidents and not all(_exact(forms, actual) for forms in roles):
        return False
    identifiers = [forms for forms in terms if not forms & _GENERAL_HEADS]
    if not identifiers:
        return False
    exact = [forms for forms in identifiers if _exact(forms, actual)]
    missing = [forms for forms in identifiers if not _exact(forms, actual)]
    if not missing:
        return True
    if len(missing) != 1 or not exact:
        return False
    close = _close_words(missing[0], actual)
    if len(close) != 1:
        return False
    # An alternative spelling with the same exact anchors can name a distinct
    # owner. Do not choose between them from edit similarity alone.
    for other_source, rows in grouped.items():
        if other_source == source:
            continue
        other = _forms(source_body(rows))
        if all(_exact(forms, other) for forms in exact):
            alternatives = _close_words(missing[0], other)
            if alternatives and not alternatives <= close:
                return False
    return True
