"""Conservative device-category normalization for source ownership checks.

These terms express general equipment categories, never document names,
projects, requirements, values, or answers. Unknown and ambiguous terms
remain unknown; semantic review and source-local numeric binding are still
required after category matching.
"""
import re
import unicodedata

_WORDS = re.compile(r'[A-Za-z\u05d0-\u05ea]+')
_FINAL = str.maketrans('ךםןףץ', 'כמנפצ')
_FAMILIES = {
    'display': ('מסכ', 'מסכי', 'צג', 'צגי', 'screen', 'monitor', 'display'),
    'controller': ('בקר', 'בקרי', 'controller'),
    'camera': ('מצלמה', 'מצלמ', 'מצלמות', 'camera'),
    'sensor': ('חיישנ', 'חיישני', 'גלאי', 'sensor', 'detector'),
    'projector': ('מקרנ', 'מקרני', 'projector'),
    'speaker': ('רמקול', 'רמקולי', 'speaker', 'loudspeaker'),
    'microphone': ('מיקרופונ', 'מיקרופוני', 'microphone'),
    'server': ('שרת', 'שרתי', 'server'),
    'router': ('נתב', 'נתבי', 'router'),
    'switch': ('מתג', 'מתגי', 'switch'),
    'keyboard': ('מקלדת', 'מקלדות', 'keyboard'),
    'door': ('דלת', 'דלתות', 'door'),
    'gate': ('שער', 'שערי', 'gate'),
    'lock': ('מנעול', 'מנעולי', 'lock'),
    'pump': ('משאבה', 'משאבות', 'pump'),
    'valve': ('שסתומ', 'שסתומי', 'valve'),
    'battery': ('סוללה', 'סוללות', 'battery'),
    'cable': ('כבל', 'כבלי', 'cable'),
    'antenna': ('אנטנה', 'אנטנות', 'antenna'),
    'motor': ('מנוע', 'מנועי', 'motor'),
}
_LOOKUP = {term.translate(_FINAL): family for family, words in _FAMILIES.items() for term in words}


def _forms(word):
    word = unicodedata.normalize('NFKC', word.casefold()).translate(_FINAL)
    values = {word}
    if re.fullmatch(r'[\u05d0-\u05ea]+', word):
        if len(word) >= 4 and word[0] in 'והבלכמ':
            values.add(word[1:])
        if len(word) >= 5 and word[:2] in {'וה','בה','לה','מה','כה'}:
            values.add(word[2:])
        for item in tuple(values):
            if len(item) >= 4 and item.endswith(('ימ','ות')):
                values.add(item[:-2])
    else:
        if len(word) >= 4 and word.endswith('s'):
            values.add(word[:-1])
    return values


def equipment_category_mentions(text, *, context_terms=()):
    """Return exact-token family mentions and their original character spans.

    A flattened PDF can concatenate a category and a named context. Split such
    a token only when its suffix exactly matches an independently supplied
    context term; arbitrary prefix guesses are never category evidence.
    """
    suffixes = {form for term in context_terms for form in _forms(str(term)) if len(form) >= 3}
    result = []
    for match in _WORDS.finditer(str(text or '')):
        forms = _forms(match.group())
        prefixes = set()
        for value in forms:
            for suffix in suffixes:
                if value.endswith(suffix) and len(value) > len(suffix):
                    prefixes.add(value[:-len(suffix)])
        for prefix in prefixes:
            forms.update(_forms(prefix))
        families = {_LOOKUP[value] for value in forms if value in _LOOKUP}
        result.extend((match.start(), match.end(), family) for family in sorted(families))
    return tuple(result)


def equipment_categories(text, *, context_terms=()):
    return frozenset(family for _, _, family in equipment_category_mentions(
        text, context_terms=context_terms))
