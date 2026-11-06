"""Conservative direct measurements from authenticated workbook item fields."""
from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r'[A-Za-z]+|[א-ת]+')
_SIZE = re.compile(r'(?<!\w)(?:ו?ה?)?(?:גודל|מידות|קוטר|אלכסון|רוחב|גובה|אורך)(?!\w)|'
                   r'\b(?:size|dimensions?|diagonal|diameter|width|height|length)\b', re.I)
_MEASUREMENT = re.compile(
    r'(?<![\w.])\d+(?:\.\d+)?\s*(?:["″]|inch(?:es)?\b|in\b|'
    r'אינץ[׳\']?|אינצ[׳\']ים|mm\b|cm\b|millimet(?:er|re)s?\b|'
    r'centimet(?:er|re)s?\b|מילימטר(?:ים)?(?!\w)|סנטימטר(?:ים)?(?!\w))', re.I)
_PREFIX = re.compile(r'^\s*(?:(?:size|diagonal|diameter|width|height|length|'
                     r'גודל|בגודל|אלכסון|באלכסון|קוטר|בקוטר|רוחב|ברוחב|'
                     r'גובה|בגובה|אורך|באורך)\s*[:\-]?\s*)?', re.I)
_DIMENSION_ROLE = {
    'diagonal': re.compile(r'\bdiagonal\b|(?<![א-ת])(?:ב)?אלכסון(?![א-ת])', re.I),
    'diameter': re.compile(r'\bdiameter\b|(?<![א-ת])(?:ב)?קוטר(?![א-ת])', re.I),
    'width': re.compile(r'\b(?:width|wide)\b|(?<![א-ת])(?:ב)?רוחב(?![א-ת])', re.I),
    'height': re.compile(r'\b(?:height|high|tall)\b|(?<![א-ת])(?:ב)?גובה(?![א-ת])', re.I),
    'length': re.compile(r'\b(?:length|long)\b|(?<![א-ת])(?:ב)?אורך(?![א-ת])', re.I),
}
_REQUEST_GRAMMAR = frozenset(('what which is are the a an of in at for on to '
    'please tell me show give does do have has its and מה כמה איזה איזו מהם מהו '
    'האם הוא היא זה זאת של את ב ל ו בפרויקט שלפרויקט').split())


def _forms(word):
    from passage_synthesis import _workbook_word_forms
    value = word.casefold()
    result = set(_workbook_word_forms(value))
    if re.fullmatch(r'[א-ת]{4,}', value) and value.startswith('ה'):
        result.update(_workbook_word_forms(value[1:]))
    return result


def select_workbook_narrow_fact(evidence, target, requested_property):
    """Return one original item field when its size is directly adjacent.

    This is a narrow deterministic path, not a general semantic parser: a
    unique measured item must be named in source order immediately before a
    single length measurement. Ambiguous rows, other properties, missing
    evidence, and measurements attached to another item are declined.
    """
    from workbook_project import _fields, _QUALIFIER
    if not evidence.complete or not getattr(evidence, 'selection_complete', True):
        return None
    if not _SIZE.fullmatch(str(requested_property).strip()):
        return None
    requested = str(requested_property).strip()
    # A plural dimensions request calls for more than one measurement; this
    # single-value path cannot establish it.
    if re.fullmatch(r'(?:dimensions|מידות|המידות)', requested, re.I):
        return None
    specific_role = next((name for name, pattern in _DIMENSION_ROLE.items()
                          if pattern.search(requested.lstrip('ה'))), None)
    wanted = _TOKEN.findall(str(target))
    if not wanted or len(wanted) > 8:
        return None
    passages = tuple(evidence.passages)
    if len({p.source_id for p in passages}) != 1:
        return None
    matches = []
    for passage in passages:
        fields = _fields(passage.source_claim)
        if passage.metadata.get('_workbook_general_qualification_ids'):
            continue
        for label, _, value, _, _ in fields:
            if not re.search(r'\b(?:product|item|description|details|specification|equipment|device)\b|'
                             r'(?:פריט|מוצר|תי?אור|ציוד|מפרט)', label, re.I):
                continue
            # A different field's qualification cannot be lost by returning
            # only this field, even when it identifies the correct item.
            if any(_QUALIFIER.search(other) for _, _, other, _, _ in fields if other != value):
                continue
            measures = list(_MEASUREMENT.finditer(value))
            if len(measures) != 1:
                continue
            tokens = list(_TOKEN.finditer(value))
            for start in range(len(tokens) - len(wanted) + 1):
                selected = tokens[start:start + len(wanted)]
                if not all(_forms(a) & _forms(b.group()) for a, b in zip(wanted, selected)):
                    continue
                suffix = value[selected[-1].end():]
                prefix = _PREFIX.match(suffix)
                measure = _MEASUREMENT.match(suffix, prefix.end()) if prefix else None
                if measure and specific_role:
                    # A specific dimension is supported only by a marker
                    # attached directly to this measurement, either before
                    # it ("width 9 inches") or just after ("9 inches wide").
                    after = suffix[measure.end():]
                    trailing = re.match(r'^\s*(?:in\s+)?(?:width|wide|height|high|tall|'
                                        r'length|long|diagonal|diameter|ברוחב|רוחב|'
                                        r'בגובה|גובה|באורך|אורך|באלכסון|אלכסון|בקוטר|קוטר)(?!\w)',
                                        after, re.I)
                    role_text = prefix.group() + (trailing.group() if trailing else '')
                    roles = {name for name, pattern in _DIMENSION_ROLE.items() if pattern.search(role_text)}
                    if roles != {specific_role}:
                        continue
                if measure:
                    matches.append((value.strip(), passage.citation_id))
                    break
    return matches[0] if len(matches) == 1 else None


def workbook_narrow_request_matches(request, scope, target, requested_property):
    """Require the planned scope, item and property to cover the request."""
    scoped = scoped_multiplier_question(request)
    if scoped:
        return (scoped.scope, scoped.target, scoped.property) == (scope, target, requested_property)
    from technical_validation import _literal
    if (not _SIZE.fullmatch(str(requested_property).strip())
            or not all((scope, target, requested_property))
            or not all(_literal(value, request) for value in (scope, target, requested_property))):
        return False
    # The planner's literal spans cannot silently omit a distinguishing item
    # or project word. Remove the three spans and permit only request grammar;
    # unknown leftover material takes the normal technical path instead.
    remainder = request
    for value in (scope, target, requested_property):
        remainder = remainder.replace(value, ' ', 1)
    return not any(word.casefold() not in _REQUEST_GRAMMAR for word in _TOKEN.findall(remainder))


def validate_workbook_narrow_answer(evidence, contract, answer):
    """Recompute the released fact using the current contract and originals."""
    if (getattr(contract, 'answer_scope', '') == 'overview' or
            not workbook_narrow_request_matches(contract.normalized_request,
                evidence.narrow_scope, evidence.narrow_target, evidence.narrow_property)):
        return False
    if scoped_multiplier_question(contract.normalized_request):
        from request_understanding import AnswerLanguage, detect_answer_language
        fact = select_multiplier_ambiguity(evidence, evidence.narrow_target, evidence.narrow_property)
        hebrew = detect_answer_language(contract.normalized_request) is AnswerLanguage.HE
        expected = render_multiplier_ambiguity(fact, evidence, hebrew) if fact else ''
        return bool(expected and answer.strip() == expected)
    fact = select_workbook_narrow_fact(evidence, evidence.narrow_target, evidence.narrow_property)
    return bool(fact and answer.strip() == fact[0] + ' [' + fact[1] + ']')


@dataclass(frozen=True)
class ScopedMultiplierQuestion:
    scope: str
    target: str
    property: str


@dataclass(frozen=True)
class MultiplierAmbiguity:
    citation_id: str
    first: str
    second: str
    qualification_ids: tuple[str, ...]
    source_label: str
    source_column: str
    source_field: str
    row_context: tuple[tuple[str, str, str], ...]


def scoped_multiplier_question(query):
    """Bind a literal item, one property and a complete ordinal workbook scope."""
    from subject_intent import ordinal_workbook_section_subject
    value = str(query or '').strip().strip(' ?!.')
    match = re.fullmatch(
        r'(?:מה|מהו|מהי)\s+(?P<property>[א-תA-Za-z]{2,25})\s+של\s+'
        r'(?P<target>[א-תA-Za-z0-9][א-תA-Za-z0-9 -]{1,60}?)\s+ב'
        r'(?P<scope>.+)', value, re.I)
    if not match or not ordinal_workbook_section_subject(match.group('scope')):
        return None
    return ScopedMultiplierQuestion(match.group('scope'), match.group('target').strip(),
                                    match.group('property').strip())


def multiplier_property_candidate(requested_property):
    """Cheaply exclude established dimensions, money and conditions."""
    from workbook_project import _QUALIFIER
    from workbook_semantic_review import _MONEY
    prop = str(requested_property or '').strip()
    if not re.fullmatch(r'[א-תA-Za-z]{2,25}', prop):
        return False
    variants = (prop, prop[1:]) if len(prop) > 3 and prop.startswith('ה') else (prop,)
    return not any(_SIZE.fullmatch(value) or _MONEY.search(value) or _QUALIFIER.search(value)
                   for value in variants)


def select_multiplier_ambiguity(evidence, target, requested_property, *,
                                veto_only=False, citation_id=None):
    """Find two exact multiplier expressions in one identified source field.

    This does not determine that they share a property or owner. A full
    re-anchor or a repeated distinguishing token bounds the second literal;
    a visible same-script owner mismatch fails closed.
    """
    from decimal import Decimal, InvalidOperation
    from workbook_project import _fields, _QUALIFIER
    from workbook_semantic_review import _STRICT_CONDITION, _EXAMPLE_MARKER, _STATUS
    from evidence_quotes import HARD_QUOTE_MAX_CHARS, HARD_QUOTE_MAX_WORDS
    if (not veto_only and (not evidence.complete or not evidence.selection_complete
            or len({p.source_id for p in evidence.passages}) != 1)):
        return None
    target = str(target or '').strip()
    prop = str(requested_property or '').strip()
    if not target or not multiplier_property_candidate(prop):
        return None
    property_forms = {prop.casefold()}
    if len(prop) >= 4 and prop.startswith('ה'):
        property_forms.add(prop[1:].casefold())
    target_pattern = re.compile(r'(?<![\w./-])' + re.escape(target) + r'(?![\w./-])', re.I)
    multiplier = re.compile(
        r'(?<![\w./-])(?:\*\s*(?P<left>[0-9]+(?:\.[0-9]+)?)|'
        r'(?P<right>[0-9]+(?:\.[0-9]+)?)\s*[x×](?![\w.]))', re.I)
    candidates = []
    for row in evidence.passages:
        if citation_id is not None and row.citation_id != citation_id:
            continue
        fields = _fields(row.source_claim)
        # One workbook row has at most one cell per column. A duplicate
        # parsed coordinate is ambiguous source structure, not a second
        # value that may be discarded or relabeled as independent context.
        columns = [column for _, column, _, _, _ in fields]
        if len(columns) != len(set(columns)):
            continue
        quantities = [v.strip() for label, _, v, _, _ in fields
                      if re.fullmatch(r'(?:qty|quantity|count|כמות)', label.strip(), re.I)]
        try:
            if len(quantities) != 1 or not Decimal(quantities[0]).is_finite() or Decimal(quantities[0]) <= 0:
                continue
        except InvalidOperation:
            continue
        for label, column, value, _, _ in fields:
            if not re.fullmatch(r'(?:product(?:\s+name)?|item(?:\s+name)?|description|'
                                r'details|specification|מוצר|פריט|תיאור|תאור)', label.strip(), re.I):
                continue
            if (len(value) > HARD_QUOTE_MAX_CHARS or len(value.split()) > HARD_QUOTE_MAX_WORDS
                    or (not veto_only and (_QUALIFIER.search(value) or _STRICT_CONDITION.search(value)
                    or _EXAMPLE_MARKER.search(value) or _STATUS.search(value.translate(
                        str.maketrans({'ם':'מ','ן':'נ','ך':'כ','ף':'פ','ץ':'צ'})))))):
                continue
            designated = tuple(row.metadata.get('_workbook_general_qualification_fields', ()))
            if not veto_only and any(other != value and other not in designated and
                   (_QUALIFIER.search(other) or _STRICT_CONDITION.search(other)
                    or _EXAMPLE_MARKER.search(other))
                   for _, _, other, _, _ in fields):
                continue
            matches = list(multiplier.finditer(value))
            if len(matches) != 2:
                continue
            a, b = matches
            if Decimal(a.group('left') or a.group('right')) == Decimal(b.group('left') or b.group('right')):
                continue
            # A model/serial label immediately before a multiplier-shaped
            # token identifies it as an identifier, not a measurement.
            if re.search(r'(?i)(?<!\w)(?:model|sku|serial|part\s+number|'
                         r'דגם|מדגם|מספר\s+דגם)\s*[:#-]?\s*$', value[:b.start()]):
                continue
            first_owner = target_pattern.search(value[:a.start()])
            between = value[a.end():b.start()]
            second_owner = target_pattern.search(between)
            second_start = a.end() + second_owner.start() if second_owner else None
            if not first_owner:
                continue
            if not second_owner:
                # A repeated exact distinguishing token can locate a second
                # literal expression without asserting it has the same owner.
                # Where the adjacent owner words use the same script, reject
                # a visibly different owner instead of treating it as an alias.
                words = re.findall(r'[A-Za-z]+|[א-ת]+', target)
                if len(words) < 2 or len(words[-1]) < 3:
                    continue
                anchor = re.compile(r'(?<![\w./-])' + re.escape(words[-1]) + r'(?![\w./-])', re.I)
                anchors = list(anchor.finditer(between))
                if not anchors:
                    continue
                found = anchors[-1]
                before = re.search(r'([A-Za-z]+|[א-ת]+)\s*$', between[:found.start()])
                after = re.match(r'\s*([A-Za-z]+|[א-ת]+)', between[found.end():])
                owner_words = words[:-1]
                same_script = [neighbor.group(1) for neighbor in (before, after) if neighbor and
                    any(bool(re.fullmatch(r'[א-ת]+', neighbor.group(1))) ==
                        bool(re.fullmatch(r'[א-ת]+', owner)) for owner in owner_words)]
                if same_script and not any(
                        _forms(neighbor) & _forms(owner)
                        for neighbor in same_script for owner in owner_words):
                    continue
                # Include the immediately adjacent source word, so a
                # potentially separate owner is not hidden in the quotation.
                second_start = a.end() + (before.start() if before else found.start())
            before = value[:a.start()]
            prop_matches = [m for m in re.finditer(r'[A-Za-z]+|[א-ת]+', before)
                            if m.group().casefold() in property_forms]
            if not prop_matches:
                continue
            property_match = prop_matches[-1]
            if not re.fullmatch(r'\s*', before[property_match.end():]):
                continue
            first = value[first_owner.start():a.end()].strip()
            # Include a short exact following label; no invented role or unit.
            tail = re.match(r'(?:\s+[A-Za-z][A-Za-z-]*){1,3}', value[b.end():])
            if not tail:
                continue
            second = value[second_start:b.end() + tail.end()].strip()
            qualifiers = tuple(row.metadata.get('_workbook_general_qualification_ids', ()))
            # The selected row may carry material details after either
            # multiplier, or in a separate free-text cell. Preserve complete
            # source fields; a substring around a number cannot prove that an
            # unrecognized trailing requirement is incidental.
            scalar_role = re.compile(
                r'(?:serial(?:\s+number)?|מספר\s+סדורי|מס.?\s*סדורי|'
                r'unit\s+price|price|מחיר(?:\s+יחידה)?|line\s+total|'
                r'סה.?כ\s+בש.?ח)', re.I)
            number_text = r'[+\-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?'
            numeric = re.compile(r'^\s*(?:(?:[₪$€£]\s*)?' + number_text + r'|'
                                 + number_text + r'\s*[₪$€£])\s*$')
            cached_formula = re.compile(
                r'(?is)^formula:\s*(?:\[shared formula\]|=[A-Z0-9+*/(). -]+);\s*'
                r'cached\s+value:\s*' + number_text + r'\s*$')
            ingestion_metadata = re.compile(r'(?i)(?<!\w)(?:formula|cached\s+value):')
            context = []
            safe = True
            for other_label, other_column, other, _, _ in fields:
                other = other.strip()
                if not other or (other_label, other_column) == (label, column):
                    continue
                if (scalar_role.fullmatch(other_label.strip())
                        and (numeric.fullmatch(other) or cached_formula.fullmatch(other))):
                    continue
                # Unknown formula/cache wrappers are not literal workbook
                # prose. Do not quote or silently discard them.
                if ingestion_metadata.search(other):
                    safe = False
                    break
                if (len(other_label) > HARD_QUOTE_MAX_CHARS or len(other_label.split()) > HARD_QUOTE_MAX_WORDS
                        or len(other) > HARD_QUOTE_MAX_CHARS or len(other.split()) > HARD_QUOTE_MAX_WORDS
                        or re.search(r'\[\d{1,3}\]', other_label + ' ' + other_column + ' ' + other)):
                    safe = False
                    break
                if (other_label, other_column, other) not in context:
                    context.append((other_label, other_column, other))
            if not safe or re.search(r'\[\d{1,3}\]', value) or ingestion_metadata.search(value):
                continue
            complete_row = ' '.join([label, column, value, first, second] +
                                    [part for field in context for part in field])
            if (len(complete_row) > HARD_QUOTE_MAX_CHARS
                    or len(complete_row.split()) > HARD_QUOTE_MAX_WORDS):
                continue
            candidates.append(MultiplierAmbiguity(row.citation_id, first, second, qualifiers,
                                                 label.strip(), column.strip(), value.strip(), tuple(context)))
    # Identical item fields in the same row do not create a second numeric
    # pair. The first selection retains every other labeled field in context.
    groups = {(c.citation_id, c.first, c.second, c.qualification_ids, c.source_field)
              for c in candidates}
    return candidates[0] if len(groups) == 1 else None


def render_multiplier_ambiguity(fact, evidence, hebrew):
    """Render exact source spans with an explicitly conditional comparison."""
    from evidence_quotes import HARD_QUOTE_MAX_CHARS, HARD_QUOTE_MAX_WORDS
    by_id = {p.citation_id: p for p in evidence.passages}
    if not hebrew or fact.citation_id not in by_id:
        return ''
    answer = (f'בשורה מופיע {fact.source_label} ({fact.source_column}): ״{fact.source_field}״. '
              f'אם הנתונים מתייחסים לאותו מאפיין של אותו רכיב, הערכים '
              f'״{fact.first}״ ו־״{fact.second}״ שונים; '
              f'מהקטעים המצוטטים לבדם לא ניתן לקבוע אם כך הדבר. [{fact.citation_id}]')
    for label, column, value in fact.row_context:
        answer += f'\n\n{label} ({column}): ״{value}״ [{fact.citation_id}]'
    for identity in fact.qualification_ids:
        passage = by_id.get(identity)
        if passage is None:
            return ''
        fields = tuple(passage.metadata.get('_workbook_general_qualification_fields', ()))
        if not fields:
            return ''
        for value in fields:
            if (not isinstance(value, str) or len(value) > HARD_QUOTE_MAX_CHARS
                    or len(value.split()) > HARD_QUOTE_MAX_WORDS
                    or re.search(r'\[\d{1,3}\]', value)):
                return ''
            answer += f'\n\n{value} [{identity}]'
    return answer


def render_overview_multiplier_disclosure(fact, hebrew):
    """Present one complete source row field and its uncertain comparison.

    The returned body deliberately has no citation marker; the caller must
    attach the selected row's marker. Its exact bytes are recomputed during
    final assessment, including every retained same-row context field.
    """
    if not hebrew or not fact:
        return ''
    context = ''.join(f'; {label} ({column}): ״{value}״' for label, column, value in fact.row_context)
    return (f'בשורה מופיע {fact.source_label} ({fact.source_column}): ״{fact.source_field}״{context}; '
            f'אם הנתונים מתייחסים לאותו מאפיין של אותו רכיב, הערכים '
            f'״{fact.first}״ ו־״{fact.second}״ שונים; '
            'מהקטעים המצוטטים לבדם לא ניתן לקבוע אם כך הדבר')


def overview_multiplier_ambiguity(evidence, identity, *, veto_only=False):
    """Identify a source-bound multiplier pair without guessing property ownership.

    Only a literal first two-word item head and the literal word immediately
    before the first multiplier propose the existing narrow proof. The proof
    is accepted only if it resolves uniquely to this same cited row.
    """
    from workbook_project import _fields
    row = next((p for p in evidence.passages if p.citation_id == identity), None)
    if row is None:
        return None
    multiplier = re.compile(
        r'(?<![\w./-])(?:\*\s*(?P<left>[0-9]+(?:\.[0-9]+)?)|'
        r'(?P<right>[0-9]+(?:\.[0-9]+)?)\s*[x×](?![\w.]))', re.I)
    found = []
    for label, _, value, _, _ in _fields(row.source_claim):
        if not re.fullmatch(r'(?:product(?:\s+name)?|item(?:\s+name)?|description|'
                            r'details|specification|מוצר|פריט|תיאור|תאור)', label.strip(), re.I):
            continue
        head = re.match(r'^\s*([A-Za-zא-ת]+\s+[A-Za-zא-ת]+)(?=\s|$)', value)
        matches = list(multiplier.finditer(value))
        if not head or len(matches) != 2:
            continue
        property_word = re.search(r'([A-Za-zא-ת]+)\s*$', value[:matches[0].start()])
        if not property_word:
            continue
        target = head.group(1)
        prop = property_word.group(1)
        fact = select_multiplier_ambiguity(evidence, target, prop, veto_only=veto_only,
                                           citation_id=identity if veto_only else None)
        if fact and fact.citation_id == identity:
            found.append((fact, target, prop))
    unique = list(dict.fromkeys(found))
    return unique[0] if len(unique) == 1 else None


def _cited_multiplier_values(identities, evidence):
    """Collect multiplier literals independently within each cited field."""
    from decimal import Decimal, InvalidOperation
    from workbook_project import _fields
    multiplier = re.compile(
        r'(?<![\w./-])(?:\*\s*(?P<left>[0-9]+(?:\.[0-9]+)?)|'
        r'(?P<right>[0-9]+(?:\.[0-9]+)?)\s*[x×](?![\w.]))', re.I)
    def number(value):
        try:
            return Decimal(value)
        except InvalidOperation:
            return None
    by_id = {p.citation_id: p for p in evidence.passages}
    result = []
    for identity in identities:
        row = by_id.get(identity)
        if row is None:
            continue
        for _, _, value, _, _ in _fields(row.source_claim):
            numbers = {number(m.group('left') or m.group('right'))
                       for m in multiplier.finditer(value)}
            numbers.discard(None)
            if len(numbers) >= 2:
                result.append(frozenset(numbers))
    return tuple(result)


def overview_cited_field_has_ambiguous_multipliers(identities, evidence):
    """Conservatively detect ambiguity even without a usable item selector."""
    return bool(_cited_multiplier_values(identities, evidence))


def overview_multiplier_claim_has_undisclosed_value(body, identities, evidence):
    """Veto a matching source-local value without asserting its meaning."""
    from decimal import Decimal, InvalidOperation
    multiplier = re.compile(
        r'(?<![\w./-])(?:\*\s*(?P<left>[0-9]+(?:\.[0-9]+)?)|'
        r'(?P<right>[0-9]+(?:\.[0-9]+)?)\s*[x×](?![\w.]))', re.I)
    standalone = re.compile(r'(?<![\w./-])([0-9]+(?:\.[0-9]+)?)(?![\w./-])')
    def number(value):
        try:
            return Decimal(value)
        except InvalidOperation:
            return None
    text = str(body or '')
    claimed = {number(m.group(1)) for m in standalone.finditer(text)}
    claimed.update(number(m.group('left') or m.group('right')) for m in multiplier.finditer(text))
    claimed.discard(None)
    return bool(claimed and any(numbers & claimed
                                for numbers in _cited_multiplier_values(identities, evidence)))
