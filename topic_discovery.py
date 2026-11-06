"""Source-owned, bounded lexical witnesses for composed topic constraints.

The caller supplies interpreted concepts: each group is required, while its
phrases are alternatives. This module neither infers synonyms nor treats a
filename, URL, or catalogue entry as proof of topical content.
"""
from dataclasses import dataclass
import re


_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_CONTACT = re.compile(r"@|https?://|www\.|\b(?:e-?mail|telephone|phone)\b|טלפון|דוא[\"׳״']?ל", re.I)
_BOUNDARY = re.compile(r"(?:\r?\n){2,}|(?<=[.!?;])\s+|\r?\n")
_HEBREW = re.compile(r"^[\u05d0-\u05ea]+$")
_PREFIXES = ('ו', 'ב', 'ל', 'ה', 'מ', 'ש', 'וה', 'בה', 'לה', 'מה')
_HEADING = re.compile(r'^(?:subject|heading|title|regarding|הנדון|נושא|כותרת)\s*[:：\-]', re.I)
_FOOTER = re.compile(r'^(?:footer|contact(?:\s+(?:details|information))?|פרטי\s+קשר|יצירת\s+קשר)\s*[:：\-]?$', re.I)
_BREADCRUMB = re.compile(r'^\[[^\]\n]+\]$')
_RELATION_WORDS = frozenset({'for', 'of', 'to', 'by', 'in', 'with', 'של', 'עבור', 'למען', 'לצורך', 'לשם'})
_CLAUSE_WORDS = frozenset({'and', 'or', 'but', 'however', 'whereas', 'although', 'while',
                           'not', 'without', 'except', 'וגם', 'וכן', 'או', 'ו', 'אבל',
                           'אך', 'בעוד', 'לא', 'ללא', 'בלי', 'למעט'})


@dataclass(frozen=True)
class TopicMatch:
    row_index: int
    start: int
    end: int
    excerpt: str


def _tokens(value):
    return tuple(match.group().casefold() for match in _WORD.finditer(value))


def _word_matches(observed, wanted):
    if observed == wanted:
        return True
    # Permit a grammatical prefix only on a substantial Hebrew topic word.
    # Acronyms and all non-Hebrew tokens still require complete word equality.
    return bool(len(wanted) >= 4 and _HEBREW.fullmatch(wanted)
                and any(observed == prefix + wanted for prefix in _PREFIXES))


def _phrase_matches(text, phrase, *, heading_continuation=False):
    wanted = _tokens(phrase)
    words = tuple(_WORD.finditer(text))
    if not wanted:
        return False

    def matches(index, expected, *, continuation=False, position=0):
        observed = words[index].group().casefold()
        # An attached conjunction introduces a separate item, not an
        # alternative spelling of a later word in one topic phrase.
        if continuation and observed != expected and observed == 'ו' + expected:
            return False
        if _word_matches(observed, expected):
            return True
        # A Hebrew construct plural can refer to one instance in a source
        # heading. Limit this normalization to a compound phrase with prior
        # context and an immediately following, exact plural noun. Standalone
        # names, abbreviations and short words never receive this expansion.
        if (position < 1 or position + 1 >= len(wanted) or index + 1 >= len(words)
                or not _HEBREW.fullmatch(expected) or not _HEBREW.fullmatch(observed)
                or not expected.endswith('י') or len(expected) < 4):
            return False
        following = wanted[position + 1]
        if (not _HEBREW.fullmatch(following) or len(following) < 4
                or not following.endswith(('ים', 'ות'))
                or words[index + 1].group().casefold() != following):
            return False
        stems = {expected[:-1]}
        if expected[0] in 'בלמה' and len(expected) >= 5:
            stems.add(expected[1:-1])
        stems = {stem for stem in stems if len(stem) >= 3}
        return observed in stems

    def extend(index, position, budget):
        if position == len(wanted):
            return True
        for following in range(index + 1, min(len(words), index + budget + 2)):
            skipped = following - index - 1
            if skipped > budget:
                continue
            gap = text[words[index].end():words[following].start()]
            if re.search(r'[,;.!?]', gap) or (not heading_continuation and re.search(r'[\r\n]', gap)):
                continue
            between = tuple(word.group().casefold() for word in words[index+1:following])
            if any(word in _CLAUSE_WORDS or (len(word) > 2 and word.startswith('ו')) for word in between):
                continue
            normal_match = matches(following, wanted[position], continuation=True, position=position)
            # A requested locative prefix can be established by a source's
            # explicit site head immediately before the same place name. The
            # shared role grammar licenses this only in that context, never by
            # removing a letter from an isolated name.
            locative_site = False
            if skipped == 1:
                from monetary_questions import (_LOCATION_HEADS, _WORK_DETAILS,
                    _WORK_UMBRELLAS, _word_variants)
                site_heads = (_LOCATION_HEADS - _WORK_DETAILS - _WORK_UMBRELLAS
                    - {'לקוח', 'לקוחה', 'טכנאי', 'טכנאית', 'טכני', 'יעוץ', 'ייעוץ'})
                observed = words[following].group().casefold()
                requested = wanted[position]
                head_forms = _word_variants((between[0],))[0]
                locative_site = bool(head_forms & site_heads and requested.startswith('ב')
                    and len(observed) >= 3 and _HEBREW.fullmatch(observed)
                    and requested[1:] == observed
                    and observed in _word_variants((between[0], requested))[1])
            if not normal_match and not locative_site:
                continue
            # A small insertion is accepted only when an explicit generic
            # relation joins the requested words; raw co-occurrence is not
            # proof that two unrelated nouns describe the same topic.
            if skipped and not locative_site and not any(word in _RELATION_WORDS for word in between):
                continue
            if extend(following, position + 1, budget - skipped):
                return True
        return False

    return any(matches(i, wanted[0]) and extend(i, 1, 2) for i in range(len(words)))


def find_topic_match(rows, concept_groups, *, max_chars=600, max_words=85):
    """Return the first bounded literal body witness, or None.

    ``rows`` is an ordered sequence of indexed row dictionaries with a
    ``document`` string. ``concept_groups`` is a nonempty sequence of
    nonempty sequences of alternative nonempty phrases. All groups must be
    witnessed within the same sentence or source line, or within a labelled
    heading and its immediately adjacent short subheading. This avoids joining
    independent body clauses, larger gaps, and footer or contact boundaries.
    The result's offsets are into that row's exact original document.
    """
    if (not concept_groups or isinstance(concept_groups, (str, bytes))
            or any(not group or isinstance(group, (str, bytes))
                   or any(not isinstance(phrase, str) or not _tokens(phrase)
                          for phrase in group) for group in concept_groups)):
        return None
    for row_index, row in enumerate(rows):
        text = row.get('document')
        if not isinstance(text, str):
            continue
        start = 0
        for boundary in (*_BOUNDARY.finditer(text), None):
            end = boundary.start() if boundary else len(text)
            fragment = text[start:end]
            left = len(fragment) - len(fragment.lstrip())
            right = len(fragment.rstrip())
            excerpt = fragment[left:right]
            if (excerpt and len(excerpt) <= max_chars and len(excerpt.split()) <= max_words
                    and not _CONTACT.search(excerpt) and not _BREADCRUMB.fullmatch(excerpt)):
                if all(any(_phrase_matches(excerpt, phrase) for phrase in group)
                       for group in concept_groups):
                    absolute = start + left
                    return TopicMatch(row_index, absolute, absolute + len(excerpt), excerpt)
            if boundary is not None:
                start = boundary.end()
        # Some source-native headings wrap a subject and its subheading over
        # two lines. Only a labelled heading followed immediately by a short,
        # punctuation-free heading fragment permits joining. Do not join body
        # paragraphs or cross larger gaps, sentence ends, or contact blocks.
        # Indexed Word paragraphs are separated by a single blank line.
        lines = list(re.finditer(r'[^\r\n]+(?:\r?\n|$)', text))
        for first, second in zip(lines, lines[1:]):
            left_line = first.group().rstrip('\r\n').strip()
            right_line = second.group().rstrip('\r\n').strip()
            gap = text[first.end():second.start()]
            if (gap not in ('', '\n', '\r\n') or not _HEADING.match(left_line)
                    or not right_line or len(left_line) > 150 or len(right_line) > 90
                    or len(right_line.split()) > 6 or re.search(r'[.!?;]', left_line + right_line)
                    or _CONTACT.search(left_line + ' ' + right_line)
                    or _FOOTER.fullmatch(right_line)):
                continue
            start = first.start() + len(first.group()) - len(first.group().lstrip())
            end = second.start() + len(second.group().rstrip())
            excerpt = text[start:end]
            if (len(excerpt) <= max_chars and len(excerpt.split()) <= max_words
                    and all(any(_phrase_matches(excerpt, phrase, heading_continuation=True) for phrase in group)
                            for group in concept_groups)):
                return TopicMatch(row_index, start, end, excerpt)
    return None


def find_topic_matches(grouped_rows, concept_groups):
    """Choose at most one witness per source without a result-count cap."""
    found = {}
    for source_id, rows in grouped_rows.items():
        match = find_topic_match(rows, concept_groups)
        if match is not None:
            found[source_id] = match
    return found
