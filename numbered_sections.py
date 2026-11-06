"""Source-bound numbered specification sections for focused retrieval.

The input must be rows obtained from the index, not caller-supplied prose.
Each row's tenant and source bytes are verified before its text is used.  The
returned spans are literal substrings of one row and carry their original
page and chunk identity.  Ancestors never come from another source or from
an unobserved gap in the index sequence.
"""
from dataclasses import dataclass
from difflib import SequenceMatcher
import math
import re
import unicodedata
from typing import Any, Mapping, Sequence

from source_authorization import authorize_source_candidate
from source_access import source_metadata_complete


# A trailing dot distinguishes section labels from decimal measurements and
# dates. Boundaries also support PDF extractions that flatten line breaks.
_NUMBER = re.compile(r"(?<![\w.\-/])(?P<number>\d{1,2}(?:\.\d{1,3}){0,5})\.(?=\s|[\u05d0-\u05eaA-Za-z])"
                     r"|(?m:^[ \t]*\.[ \t]*(?P<reversed>\d{1,2})[ \t]*(?=\n|[\u05d0-\u05eaA-Za-z]))")
_BREADCRUMB = re.compile(r"\A(?:\s*\[[^\]\n]{1,500}\]\s*)+")


@dataclass(frozen=True)
class SectionSpan:
    number: str
    text: str
    start: int
    end: int
    chunk_id: str
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class OwnedSection:
    section: SectionSpan
    ancestors: tuple[SectionSpan, ...]


@dataclass(frozen=True)
class SectionSelection:
    sections: tuple[OwnedSection, ...]
    ambiguous: bool = False
    reason: str = ""


_WORDS = re.compile(r"[A-Za-z\u05d0-\u05ea0-9]+")
_SCOPE_GENERIC = frozenset(("project", "site", "campus", "document", "file",
                            "פרויקט", "פרוייקט", "קמפוס", "מסמך", "קובץ"))
_LEADING_SCOPE_GRAMMAR = frozenset(("של", "את", "לפי", "of", "the", "in", "from", "for"))
_SCOPE_DESCRIPTORS = frozenset(('מערכת','מערכות','תשתית','תשתיות','מפרט','תוכנית','תכנית',
                                'מסמך','קובץ','system','systems','infrastructure','specification',
                                'design','document','file'))


def _terms(text: str) -> tuple[str, ...]:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"(?<=[A-Za-z\u05d0-\u05ea])(?=\d)|(?<=\d)(?=[A-Za-z\u05d0-\u05ea])", " ", text)
    return tuple(t.casefold() for t in _WORDS.findall(text))


def _related(left: str, right: str) -> bool:
    if left == right:
        return True
    # Hebrew articles/prepositions and PDF word concatenation may obscure a
    # literal anchor. This is recall/ranking only, never claim validation.
    def forms(word):
        word = word.translate(str.maketrans('ךםןףץ', 'כמנפצ'))
        out = {word}
        if len(word) >= 4 and word[0] in "הבולמכ":
            out.add(word[1:])
        if len(word) >= 5 and word[:2] in {"וה", "בה", "לה", "מה", "כה"}:
            out.add(word[2:])
        for item in tuple(out):
            if len(item) >= 5 and item.endswith(('ימ', 'ות')):
                out.add(item[:-2])
        return {x for x in out if len(x) >= 3}
    return any(a == b or (len(a) >= 3 and len(b) >= 3 and (a in b or b in a))
               or (len(a) >= 4 and len(b) >= 4 and a[:3] == b[:3]
                   and SequenceMatcher(None, a, b).ratio() >= .8)
               for a in forms(left) for b in forms(right))


def _matches(needles, haystack) -> int:
    return sum(any(_related(term, value) for value in haystack) for term in needles)


def _scope_match(term: str, tokens: tuple[str, ...]) -> bool:
    # Names constrain source identity. Do not use fuzzy or substring matching
    # here: similarly named projects must not silently share evidence.
    forms = {term}
    if len(term) >= 4 and term[0] in "הבולמכ":
        forms.add(term[1:])
    return any(token in forms or (len(token) >= 4 and token[0] in "הבולמכ"
                                  and token[1:] in forms) for token in tokens)


def _scope_parts(raw: str):
    """Separate an explicitly introduced project from preceding context.

    Without a project/document marker, retain every scope term: an arbitrary
    subsystem phrase is not proof that some suffix is a project name. Two
    distinct marked names are ambiguous and cannot be narrowed to the latter.
    """
    terms = list(_terms(raw))
    while terms and terms[0] in _LEADING_SCOPE_GRAMMAR:
        terms.pop(0)
    def marker(term):
        forms = {term}
        if len(term) >= 4 and term[0] in "הבולמכ":
            forms.add(term[1:])
        if len(term) >= 5 and term[:2] in {"וה", "בה", "לה", "מה", "כה"}:
            forms.add(term[2:])
        return bool(forms & _SCOPE_GENERIC)
    marks = [index for index, term in enumerate(terms) if marker(term)]
    if marks:
        for left, right in zip(marks, marks[1:]):
            if any(term not in _LEADING_SCOPE_GRAMMAR for term in terms[left+1:right]):
                return (), (), ()
        position = marks[-1]
        context = tuple(term for term in terms[:marks[0]]
                        if term not in _LEADING_SCOPE_GRAMMAR)
        identity = terms[position+1:]
        while identity and identity[0] in _LEADING_SCOPE_GRAMMAR:
            identity.pop(0)
        cover = tuple(terms[position:])
    else:
        context, identity, cover = (), terms, tuple(terms)
    return tuple(identity), tuple(context), cover


def technical_source_scope_matches(plan: Any, metadata: Mapping[str, Any], rows=()) -> bool:
    """Match scope to source path or a literal same-source cover phrase.

    Rows must be from the already-authorized index census. A random mention on
    a later page cannot promote a different project's document into scope.
    """
    raw = getattr(plan, "scope", "")
    if not raw:
        return True
    scope, _context, full = _scope_parts(raw)
    path = _terms(str(metadata.get("source_relative_path") or ""))
    if not scope:
        return False
    if all(_scope_match(term, path) for term in scope):
        return True
    # A file title can omit the possessive connector between a generic
    # descriptive noun phrase and its owner. Keep every content word on both
    # sides mandatory; never infer a project from an unmarked suffix alone.
    separators = [i for i, term in enumerate(scope) if term in {'של', 'of'}]
    def descriptor(term):
        candidates = {term}
        if len(term) >= 4 and term[0] in 'הבולמכ':
            candidates.add(term[1:])
        return bool(candidates & _SCOPE_DESCRIPTORS)
    if (len(separators) == 1 and 0 < separators[0] < len(scope)-1
            and any(descriptor(term) for term in scope[:separators[0]])
            and all(_scope_match(term, path) for i, term in enumerate(scope)
                    if i != separators[0])):
        return True
    identity_keys = ("source_id", "tenant_id", "source_sha256", "source_relative_path")
    for row in rows:
        meta = row.get("metadata") or {}
        if meta.get("page") != 1 or any(meta.get(k) != metadata.get(k) for k in identity_keys):
            continue
        cover = _terms(str(row.get("document") or ""))
        if any(all(_scope_match(term, (value,)) for term, value in zip(full, cover[i:i+len(full)]))
               for i in range(len(cover) - len(full) + 1)):
            return True
    return False


def _navigation(section: OwnedSection) -> bool:
    meta, body = section.section.metadata, section.section.text
    role = str(meta.get("structural_type") or meta.get("source_unit_type") or "").casefold()
    return (role in {"toc", "table_of_contents", "navigation"}
            or bool(re.search(r"\.{5,}", body))
            or bool(re.search(r"(?:תוכן\s+עניינים|table\s+of\s+contents)", body, re.I)))


_TARGET_LOCATION = re.compile(
    r'(?i)(?<!\w)(?:ב(?:חדר|אולם|קומה|מבנה|בניין|אתר|מתחם|מעבדה|משרד|עמדה|מתקן)|'
    r'(?:in|at|inside)\s+(?:the\s+)?(?:[a-z-]+\s+){0,3}'
    r'(?:room|hall|floor|building|site|laboratory|office|station))\b')


def split_technical_target(target: str) -> tuple[str, str]:
    """Separate a locative qualifier from the equipment's owning identity."""
    match = _TARGET_LOCATION.search(target or '')
    if not match or not target[:match.start()].strip():
        return target or '', ''
    return target[:match.start()].strip(), target[match.start():].strip()


def _requested_category_terms(plan: Any) -> tuple[str, ...]:
    # A requirement may contain both a property and the equipment category
    # that owns it ("size of the panels"). Keep the latter as a ranking cue.
    from technical_validation import _PROPERTY_PATTERNS
    result = []
    for requirement in getattr(plan, 'requirements', ()):
        text = str(getattr(requirement, 'text', '') or '')
        for pattern in _PROPERTY_PATTERNS.values():
            text = re.sub(pattern, ' ', text, flags=re.I)
        result.extend(term for term in _terms(text)
                      if len(term) >= 3 and term not in {'the','for','and','של','עבור'})
    return tuple(dict.fromkeys(result))


def technical_scope_device_context(plan: Any) -> tuple[str, ...]:
    """Retain an explicit user-side device phrase before a project name.

    This is a candidate/owner anchor, not an assertion that a device noun and
    a source noun are synonyms. A semantic review must still establish that.
    """
    if getattr(plan, 'scope_origin', '') != 'current':
        return ()
    from technical_validation import _reference_topic_forms
    _identity, context, _cover = _scope_parts(getattr(plan, 'scope', ''))
    terms = tuple(term for term in context if _reference_topic_forms(term)
                  and term not in _SCOPE_DESCRIPTORS
                  and not (len(term) >= 4 and term[0] in 'הבולמכ'
                           and term[1:] in _SCOPE_DESCRIPTORS))
    return terms if len(terms) >= 2 else ()


def rank_technical_sections(plan: Any, sections: Sequence[OwnedSection], *,
                            max_sections_per_source: int = 8,
                            max_sources: int = 4, source_rows=None) -> SectionSelection:
    """Rank source-owned section candidates without choosing a project by default.

    Explicit scope is matched against the source path; scope conflicts fail
    closed. Unscoped matches from different source identities are preserved
    for the caller to compare or ask for clarification. This function ranks
    retrieval candidates only: it does not assert or validate answer facts.
    """
    core, location = split_technical_target(getattr(plan, "target", ""))
    target = _terms(core)
    location_terms = _terms(location)
    categories = _requested_category_terms(plan)
    scope_device = technical_scope_device_context(plan)
    scope, scope_context, _cover = _scope_parts(getattr(plan, "scope", ""))
    requirements = tuple(_terms(getattr(r, "text", ""))
                         for r in getattr(plan, "requirements", ()))
    if not target or (getattr(plan, "scope", "") and not scope):
        return SectionSelection((), reason="scope_or_target_unresolved")
    ranked: dict[str, list[tuple]] = {}
    for position, item in enumerate(sections):
        if _navigation(item):
            continue
        meta = item.section.metadata
        source = str(meta.get("source_id") or "")
        if not source or not technical_source_scope_matches(plan, meta, (source_rows or {}).get(source, ())):
            continue
        owner_text = " ".join(x.text for x in (*item.ancestors, item.section))
        owner_terms = _terms(owner_text)
        target_matches = _matches(target, owner_terms)
        # A user may use a synonym for the equipment while explicitly naming
        # the device in the pre-project scope. Preserve candidates matching
        # that exact device phrase for the later semantic ownership review.
        effective_target = target
        if not target_matches:
            if not scope_device or _matches(scope_device, owner_terms) != len(scope_device):
                continue
            effective_target = scope_device
            target_matches = len(scope_device)
        local_terms = _terms(item.section.text)
        local_matches = _matches(effective_target, local_terms)
        opening_matches = _matches(effective_target, _terms(item.section.text[:80]))
        category_matches = _matches(categories, _terms(item.section.text[:120]))
        location_matches = _matches(location_terms, local_terms)
        owned_location_matches = _matches(location_terms, owner_terms)
        numeric = bool(re.search(r'(?<!\d)\d+(?:[.,]\d+)?',
                                 re.sub(r'^\s*\d+(?:\.\d+)*\.', '', item.section.text)))
        parent_matches = (_matches(effective_target, _terms(item.ancestors[-1].text))
                          if item.ancestors else 0)
        property_counts = tuple(_matches(req, local_terms) for req in requirements)
        property_matches = sum(property_counts)
        context_matches = _matches(scope_context, owner_terms)
        score = (target_matches * 20 + local_matches * 10 + parent_matches * 25
                 + property_matches * 12 + context_matches * 8 + owned_location_matches * 15)
        ranked.setdefault(source, []).append((score, position, item, property_counts, target_matches,
                                            local_matches, opening_matches, category_matches,
                                            location_matches, numeric, owned_location_matches))
    if len(ranked) > max_sources:
        return SectionSelection((), ambiguous=True, reason="multiple_matching_sources")
    selected = []
    for source in sorted(ranked):
        # Device terms have different discriminating power within each
        # authorized source. A ubiquitous term such as "interface" or
        # "system" should not crowd out a more specific device modifier.
        # This is candidate ranking only; ownership is checked separately.
        source_rows_ranked = ranked[source]
        owner_terms = [ _terms(' '.join(span.text for span in
                         (*row[2].ancestors, row[2].section)))
                        for row in source_rows_ranked ]
        weights = [1.0 + math.log((len(owner_terms) + 1) /
                   (1 + sum(bool(_matches((term,), words)) for words in owner_terms)))
                   for term in target]
        local_words = [_terms(row[2].section.text) for row in source_rows_ranked]
        property_weights = [[1.0 + math.log((len(local_words) + 1) /
                             (1 + sum(bool(_matches((term,), words)) for words in local_words)))
                             for term in requirement] for requirement in requirements]
        weighted = []
        for row, words in zip(source_rows_ranked, owner_terms):
            specificity = sum(weight for term, weight in zip(target, weights)
                              if _matches((term,), words))
            parent = (_terms(row[2].ancestors[-1].text) if row[2].ancestors else ())
            parent_specificity = sum(weight for term, weight in zip(target, weights)
                                     if _matches((term,), parent))
            weighted.append((*row, specificity, parent_specificity))
        rows = sorted(weighted, key=lambda item: (-(item[0] + 20 * item[11]), item[1]))
        chosen = []
        limit = max(1, max_sections_per_source)
        # Retain the strongest locally named device even if a value is given
        # without repeating the property's name (e.g. a heading followed by
        # a bare measurement). Do not rely on a property keyword alone.
        anchor = max(rows, key=lambda row: (row[6], row[7], row[9], row[10], row[5], row[4], row[0], -row[1]))
        chosen.append(anchor)
        # Preserve the same-source passage that ties the equipment to an
        # explicitly requested location, when the value's own heading omits
        # it. The final fact gate must independently check this relationship.
        if location_terms and len(chosen) < limit:
            locs = [row for row in rows if row[5] == len(target) and row[8] == len(location_terms)]
            if locs:
                location_witness = max(locs, key=lambda row: (row[0], -row[1]))
                if location_witness not in chosen:
                    chosen.append(location_witness)
        # Each requested property gets an opportunity before the shared
        # candidate budget, so a locally named property in a child section is
        # not crowded out by generic repeated device headings.
        for index in range(len(requirements)):
            matches = [row for row in rows if row[3][index]]
            if matches:
                def property_rank(row):
                    local = _terms(row[2].section.text)
                    specificity = sum(weight for term, weight in
                                      zip(requirements[index], property_weights[index])
                                      if _matches((term,), local))
                    return (row[11] + row[12] + 2 * specificity,
                            row[9], row[3][index], row[0], -row[1])
                best = max(matches, key=property_rank)
                if best not in chosen and len(chosen) < limit:
                    chosen.append(best)
        for row in rows:
            if len(chosen) >= limit:
                break
            if row not in chosen:
                chosen.append(row)
        selected.extend(item[2] for item in chosen)
    return SectionSelection(tuple(selected), reason="matched" if selected else "no_matching_sections")


def numbered_source_sections(rows: Sequence[Mapping[str, Any]], *, tenant: str,
                             document_root: str) -> tuple[OwnedSection, ...]:
    """Build contiguous hierarchy for one independently authorized source.

    Raise ValueError for mixed identities, invalid locations, duplicate index
    positions, or failed authorization; silently mixing documents would make
    category ownership and citations unsound. Index gaps reset context.
    """
    if not rows:
        return ()
    prepared = []
    identity = None
    authorized = False
    for row in rows:
        meta = dict(row.get("metadata") or {})
        text = str(row.get("document") or "")
        chunk = str(meta.get("chunk_id") or row.get("id") or "")
        key = tuple(meta.get(k) for k in ("tenant_id", "source_id", "source_sha256",
                                           "source_relative_path", "source_size", "source_mtime_ns"))
        index, page = meta.get("chunk_index"), meta.get("page")
        if (not chunk or not text.strip() or not source_metadata_complete(meta, tenant)
                or type(index) is not int or index < 0
                or type(page) is not int or page < 1):
            raise ValueError("invalid indexed section location")
        if identity is None:
            identity = key
        elif key != identity:
            raise ValueError("mixed source identity")
        if not authorized:
            authorization = authorize_source_candidate(document_root, tenant, meta,
                                                         candidate_text=text, candidate_id=chunk)
            if not authorization.authorized:
                raise ValueError("source authorization failed: " + authorization.reason)
            authorized = True
        prepared.append((index, chunk, text, meta))
    prepared.sort(key=lambda item: item[0])
    if len({item[0] for item in prepared}) != len(prepared):
        raise ValueError("duplicate chunk index")

    headings: dict[tuple[str, ...], SectionSpan] = {}
    previous = None
    previous_text = ""
    active: OwnedSection | None = None
    previous_page = None
    page_context = None
    result = []
    for index, chunk, text, meta in prepared:
        if previous is not None and index != previous + 1:
            headings.clear()
            previous_text = ""
            active = None
            page_context = None
            previous_page = None
        previous = index
        first_chunk_on_page = meta['page'] != previous_page
        if first_chunk_on_page:
            page_context = None
        previous_page = meta['page']
        start = _BREADCRUMB.match(text)
        offset = start.end() if start else 0
        matches = list(_NUMBER.finditer(text, offset))
        prefix_end = matches[0].start() if matches else len(text)
        prefix = text[offset:prefix_end].strip()
        # A short, non-sentential page opening can be a literal source title
        # or location heading. Carry it only within that same page, never from
        # a different source or page; the model still has to verify its role.
        lines = [line.strip() for line in prefix.splitlines() if line.strip()]
        if (first_chunk_on_page and page_context is None and 1 <= len(lines) <= 5 and len(prefix) <= 180
                and not re.search(r"(?:https?://|www\.|@|\d{4,}|[.!?])", prefix, re.I)
                and all(len(line) <= 80 for line in lines)):
            a = text.find(prefix, offset, prefix_end)
            if a >= 0:
                page_context = SectionSpan('', prefix, a, a + len(prefix), chunk, meta)
        # A record that starts in the middle of a section can inherit its
        # owner only when a substantial exact indexed overlap proves the
        # continuation. A page change alone is not proof of ownership.
        if active and prefix:
            overlap = next((n for n in range(min(len(previous_text), len(prefix)), 23, -1)
                            if previous_text.endswith(prefix[:n])), 0)
            # Repeated page headers/footers often overlap adjacent chunks but
            # cannot establish section continuity. Suppress an overlap also
            # observed in a third indexed chunk from the same source.
            repeated = overlap and sum(prefix[:overlap] in row[2] for row in prepared) > 2
            page_header = bool(re.match(r"\d{1,4}\s*\n", prefix))
            if overlap and not repeated and not page_header:
                a = text.find(prefix, offset, prefix_end)
                span = SectionSpan(active.section.number, prefix, a, a + len(prefix), chunk, meta)
                result.append(OwnedSection(span, active.ancestors))
        for i, match in enumerate(matches):
            number = match.group("number") or match.group("reversed")
            path = tuple(number.split("."))
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            a, b = match.start(), end
            while b > a and text[b - 1].isspace():
                b -= 1
            if b <= match.end():
                continue
            # A sibling or ancestor ends the previous branch. This prevents a
            # later equipment category from inheriting its neighbor's owner.
            headings = {p: h for p, h in headings.items()
                        if len(p) < len(path) and path[:len(p)] == p}
            span = SectionSpan(number, text[a:b], a, b, chunk, meta)
            ancestors = ((page_context,) if page_context is not None else ()) + tuple(
                headings[p] for n in range(1, len(path)) if (p := path[:n]) in headings)
            active = OwnedSection(span, ancestors)
            result.append(active)
            headings[path] = span
        if not matches and not (active and prefix and overlap and not repeated and not page_header):
            # For an unnumbered document, retain bounded literal paragraphs as
            # candidates; do not invent a heading or combine neighboring text.
            for block in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text[offset:]):
                a, b = offset + block.start(), offset + block.end()
                body = text[a:b].strip()
                if len(_terms(body)) < 3 or len(body) > 1800:
                    continue
                a = text.find(body, a, b)
                span = SectionSpan('', body, a, a+len(body), chunk, meta)
                result.append(OwnedSection(span, ()))
        previous_text = text.rstrip()
    return tuple(result)
