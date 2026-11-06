"""Conservative recovery of original workbook fields from a rejected group.

Each returned item is an unchanged field and its own source reference.  This
does not prove or retain the rejected sentence's asserted relationships.
"""
import re


def grouped_numeric_claim_supported(body, identities, evidence):
    """Check the rejected original's number ownership independently of salvage."""
    from passage_synthesis import _overview_numbered_supported
    by_id = {p.citation_id: p for p in evidence.passages}
    ids = tuple(identities)
    return bool(ids and len(ids) == len(set(ids)) and len(by_id) == len(evidence.passages)
                and all(i in by_id for i in ids)
                and _overview_numbered_supported(body, [by_id[i] for i in ids]))


def grouped_recovery_constraints_supported(body, identities, evidence, recovered=()):
    """Do not call a rejected status/numeric claim fully recovered by IDs."""
    from workbook_semantic_review import _STATUS
    from workbook_project import _fields
    normalized = body.translate(str.maketrans({'ם':'מ','ן':'נ','ך':'כ','ף':'פ','ץ':'צ'}))
    by_id = {p.citation_id: p for p in evidence.passages}
    # A selected prefix is deliberately less than the complete source field;
    # do not assert that all details in the rejected wording were recovered.
    full_fields = all(len(ids) == 1 and ids[0] in by_id and
        any(' '.join(value.split()) == ' '.join(text.split())
            for _, _, value, _, _ in _fields(by_id[ids[0]].source_claim))
        for text, ids in recovered)
    return (full_fields and not _STATUS.search(normalized)
            and grouped_numeric_claim_supported(body, identities, evidence))


def _without_source_context(body, rows, money_pattern):
    """Mask exact shared source-heading text for recovery checks only.

    Heading text is not a device specification. This does not validate the
    original claim, and every released field still faces the ordinary gate.
    """
    scopes = {(p.source_id, p.tenant_id, p.metadata.get('sheet'),
               p.metadata.get('_workbook_scope')) for p in rows}
    headings = [tuple(p.metadata.get('_workbook_heading', ())) for p in rows]
    if len(scopes) != 1 or not headings or any(h != headings[0] for h in headings):
        return body
    masked = body
    for heading in headings[0]:
        if not isinstance(heading, str) or len(heading.split()) < 2:
            continue
        # Exact source-owned heading, never an isolated matching number.
        masked = re.sub(r'(?<!\w)' + re.escape(heading) + r'(?!\w)',
                        lambda m: ' ' * len(m.group()), masked, flags=re.I)
        # A document-type prefix may appear without its longer owner heading.
        # Only a literal initial phrase containing the monetary word qualifies;
        # a directly following number remains a monetary assertion.
        match = re.match(r'^\s*(\S+\s+\S+)', heading)
        if match and money_pattern.search(match.group(1)):
            prefix = match.group(1)
            pattern = r'(?<!\w)(?:[וכלבהמ])?' + re.escape(prefix) + r'(?!\w)(?!\s*\d)'
            masked = re.sub(pattern, lambda m: ' ' * len(m.group()), masked, flags=re.I)
    return masked


def _primary_identifier_witness(body, row, head, evidence, hebrew, assess):
    """Prove an identifier directly attached to the source's primary head.

    This identifies a source row only; it does not validate the rejected
    model wording or any generated measurement.  A later accessory's name or
    code cannot be promoted to the main item's identifier by co-occurrence.
    """
    from workbook_project import WorkbookProjectEvidence, _fields
    from workbook_context_recovery import _recovery_row_proof
    if not isinstance(evidence, WorkbookProjectEvidence) or not evidence.overview_synthesis:
        return False
    proof = _recovery_row_proof((row.citation_id,), evidence, hebrew, assess,
                                require_quantity=True)
    if not proof:
        return False
    proven_row, fields, accounted = proof
    if proven_row is not row:
        return False
    item_role = re.compile(r'(?:product(?:\s+name)?|item(?:\s+name)?|'
                           r'description|details|specification|מוצר|פריט|תיאור|תאור)', re.I)
    items = [(label, value) for label, _, value, _, _ in fields
             if item_role.fullmatch(label.strip()) and value.strip()]
    if len(items) != 1:
        return False
    if any(not item_role.fullmatch(label.strip()) and not accounted(label, value)
           for label, _, value, _, _ in fields if value.strip()):
        return False
    words = head.split()
    if len(words) != 2:
        return False
    # The source itself must place a mixed alphanumeric identifier immediately
    # after the primary item head.  A code appearing after another item or an
    # accessory phrase has no source-proven primary attachment.
    pattern = (r'^\s*' + re.escape(head) + r'\s+'
               r'(?P<identifier>(?=[A-Za-z0-9./-]*[A-Za-z])'
               r'(?=[A-Za-z0-9./-]*\d)[A-Za-z0-9][A-Za-z0-9./-]{1,60})'
               r'(?![\w./-])')
    found = re.match(pattern, items[0][1], re.I)
    if not found:
        return False
    identifier = found.group('identifier')
    literal = re.compile(r'(?<![\w./-])' + re.escape(identifier) + r'(?![\w./-])', re.I)
    # The same identifier in any other selected item field is ambiguous.
    occurrences = [(passage.citation_id, value) for passage in evidence.passages
        for label, _, value, _, _ in _fields(passage.source_claim)
        if item_role.fullmatch(label.strip()) and literal.search(value)]
    if len(occurrences) != 1 or occurrences[0][0] != row.citation_id:
        return False
    opening = re.sub(
        r'^(?:הפרויקט|הציוד|המסמך|הגיליון|the\s+project|the\s+equipment|'
        r'the\s+document|the\s+worksheet|the\s+workbook)\s+'
        r'(?:כולל|כוללת|כוללים|כוללות|מפרט|מציין|includes?|contains?|lists?)\s+',
        '', body.strip(), flags=re.I)
    # The generated identifier must attach directly to the same two source
    # head words. Mentions after an accessory or another item do not count.
    return bool(re.match(r'^' + re.escape(words[0]) + r'\s+' + re.escape(identifier) +
                         r'\s+' + re.escape(words[1]) + r'(?![\w./-])', opening, re.I))


def _source_head_list_recovery(body, identity, evidence, hebrew, assess):
    """Recover a unique source field whose explicit list head is proposed.

    No generated detail is copied. The field, its row ownership, quantity,
    conditions and final validation remain independent source authority; the
    rejected generated wording is still unresolved in the caller.
    """
    from workbook_project import WorkbookProjectEvidence, _fields, _QUALIFIER
    from workbook_context_recovery import (_recovery_row_proof, _PRICE_FIELD,
                                           _SCALAR, _FORMULA)
    from passage_synthesis import _PROTECTED, _SOURCE_CONDITIONAL
    from workbook_semantic_review import _EXAMPLE_MARKER, _STRICT_CONDITION
    from evidence_quotes import HARD_QUOTE_MAX_CHARS, HARD_QUOTE_MAX_WORDS
    if not isinstance(evidence, WorkbookProjectEvidence) or not evidence.overview_synthesis:
        return ()
    proof = _recovery_row_proof((identity,), evidence, hebrew, assess,
                                require_quantity=True)
    if not proof:
        return ()
    row, fields, accounted = proof
    item_role = re.compile(r'(?:product(?:\s+name)?|item(?:\s+name)?|description|'
        r'details|specification|equipment|device|work|service|מוצר|פריט|'
        r'תיאור(?:\s+(?:כללי|מפורט))?|תאור(?:\s+(?:כללי|מפורט))?|ציוד|מפרט|עבודה|שירות)', re.I)
    items = [value.strip() for label, _, value, _, _ in fields
             if item_role.fullmatch(label.strip()) and value.strip()]
    def accounted_sibling(label, value):
        if accounted(label, value):
            return True
        # A syntactically complete, explicitly labelled price cell can be
        # ignored when recovering a nonfinancial item field. This does not
        # assert its amount or prove a total, and unknown/conditional price
        # wording remains a veto.
        return bool(_PRICE_FIELD.fullmatch(label.strip()) and
                    (_SCALAR.fullmatch(value.strip()) or _FORMULA.fullmatch(value.strip())))
    if (len(items) != 1 or any(not item_role.fullmatch(label.strip()) and
            not accounted_sibling(label, value)
            for label, _, value, _, _ in fields if value.strip())):
        return ()
    value = items[0]
    if len(value) > HARD_QUOTE_MAX_CHARS or len(value.split()) > HARD_QUOTE_MAX_WORDS:
        return ()
    # The explicit colon marks a source-owned item/list boundary. A generic
    # first word elsewhere in a field is not enough to establish ownership.
    source_head = re.match(
        r'^\s*(?P<head>[A-Za-zא-ת]{3,}(?:\s+[A-Za-zא-ת]{3,}){0,2})\s+'
        r'(?:כולל(?:ת|ים|ות)?|includes?|including)\s*:\s*\S', value, re.I)
    if not source_head:
        return ()
    head = source_head.group('head')
    opening = re.sub(
        r'^(?:הפרויקט|הציוד|המסמך|הגיליון|the\s+project|the\s+equipment|'
        r'the\s+document|the\s+worksheet|the\s+workbook)\s+'
        r'(?:כולל|כוללת|כוללים|כוללות|מפרט|מציין|includes?|contains?|lists?)\s+',
        '', body.strip(), flags=re.I)
    if not re.match(r'^' + re.escape(head) + r'\s+'
                    r'(?:[השו]?כולל(?:ת|ים|ות)?|includes?|including)(?![A-Za-zא-ת])',
                    opening, re.I):
        return ()
    # A repeated source head is not a unique primary-item witness.
    owners = {passage.citation_id for passage in evidence.passages
              for label, _, other, _, _ in _fields(passage.source_claim)
              if item_role.fullmatch(label.strip()) and re.match(
                  r'^\s*' + re.escape(head) + r'\s+'
                  r'(?:כולל(?:ת|ים|ות)?|includes?|including)\s*:', other, re.I)}
    if owners != {identity}:
        return ()
    # An invented condition, example, number or model may not anchor a
    # replacement. The source field is retained in full even when safe.
    for marker in (_QUALIFIER, _PROTECTED, _SOURCE_CONDITIONAL,
                   _EXAMPLE_MARKER, _STRICT_CONDITION):
        asserted = {m.group().casefold() for m in marker.finditer(body)}
        actual = {m.group().casefold() for m in marker.finditer(value)}
        if not asserted <= actual:
            return ()
    number = re.compile(r'\d+(?:[.,]\d+)*')
    if not set(number.findall(body)) <= set(number.findall(value)):
        return ()
    identifier = re.compile(r'(?<![A-Za-z0-9])(?=[A-Za-z0-9/-]*[A-Za-z])'
                            r'(?=[A-Za-z0-9/-]*\d)[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)+'
                            r'(?![A-Za-z0-9])', re.I)
    if not {m.group().casefold() for m in identifier.finditer(body)} <= {
            m.group().casefold() for m in identifier.finditer(value)}:
        return ()
    decision = assess(value + ' [' + identity + ']', evidence, hebrew, check_coverage=False)
    return ((value, (identity,)),) if decision.accepted else ()


def recover_grouped_source_claims(body, identities, evidence, hebrew, assess):
    """Return independently assessed original fields relevant to a failed group.

    The caller must have resolved the overview operation from the request
    contract. Unknown or unavailable IDs and numbers absent from the cited
    source rows make recovery unavailable. No generated text is used as a
    source; numeric ownership of the original claim is reported separately.
    """
    from passage_synthesis import _FORMS, _PROTECTED, _workbook_word_forms
    from workbook_project import _fields, _QUALIFIER
    from workbook_semantic_review import (
        _EXAMPLE_MARKER, _MONEY, _STATUS, _STRICT_CONDITION,
    )
    from evidence_quotes import HARD_QUOTE_MAX_CHARS, HARD_QUOTE_MAX_WORDS
    prohibited = re.compile(r'\b(?:prohibited|forbidden)\b', re.I)
    condition_patterns = (_QUALIFIER, _STRICT_CONDITION, _EXAMPLE_MARKER,
                          _PROTECTED, prohibited)

    ids = tuple(identities)
    passages = {p.citation_id: p for p in evidence.passages}
    eligible = getattr(evidence, 'quote_eligible_ids', None)
    if (not getattr(evidence, 'overview_synthesis', False) or not ids
            or len(ids) != len(set(ids)) or len(passages) != len(evidence.passages)
            or any(i not in passages for i in ids)
            or (eligible is not None and any(i not in eligible for i in ids))):
        return ()
    rows = tuple(passages[i] for i in ids)
    owners = {(p.source_id, p.tenant_id, p.metadata.get('sheet')) for p in rows}
    if len(owners) != 1 or not all(next(iter(owners))):
        return ()
    recovery_body = _without_source_context(body, rows, _MONEY)
    # Hebrew prepositions can attach to a monetary word. Check the existing
    # money guard after removing only a grammatical one-letter prefix.
    prefixed_money = any(word[0] in 'ובכלמהש' and _MONEY.fullmatch(word[1:])
                         for word in re.findall(r'[א-ת]+', recovery_body) if len(word) > 1)
    if _MONEY.search(recovery_body) or prefixed_money:
        return ()
    normalize = str.maketrans({'ם':'מ','ן':'נ','ך':'כ','ף':'פ','ץ':'צ'})
    status_markers = {m.group().casefold() for m in _STATUS.finditer(body.translate(normalize))}
    source_statuses = {m.group().casefold() for row in rows for _, _, value, _, _ in _fields(row.source_claim)
                       for m in _STATUS.finditer(value.translate(normalize))}
    if not status_markers <= source_statuses:
        return ()

    # When syntax prevents number ownership from being established, retain
    # only source fields. Reject entirely if a proposed number does not even
    # occur in a non-price field of the cited source rows. The caller still
    # records the original claim as unresolved unless the independent numeric
    # assessment succeeds.
    if not grouped_numeric_claim_supported(body, ids, evidence):
        number = re.compile(r'\d+(?:[.,]\d+)*')
        proposed = set(number.findall(recovery_body))
        available = {match.group() for row in rows for label, _, value, _, _ in _fields(row.source_claim)
                     if not re.search(r'\b(?:price|cost|total|amount)\b|'
                                      r'(?:מחיר|עלות|סכום|סה.?כ)', label, re.I)
                     for match in number.finditer(value)}
        if not proposed <= available:
            return ()

    # An item head alone can hide an unmarked source condition. If this
    # rejected claim identifies the item, recover only the complete exact
    # source disclosure and explicit conditional comparison, independently
    # revalidated by the final gate. The original claim remains unresolved.
    from workbook_narrow import overview_multiplier_ambiguity, render_overview_multiplier_disclosure
    def undisputed_head(identity, row):
        selected = overview_multiplier_ambiguity(evidence, identity)
        if not selected:
            return None
        fact, head, _ = selected
        # An incidental nested accessory does not identify the claimed item.
        # Allow only the same neutral source-list introduction used by the
        # singleton source-description recovery below.
        exact_head = re.match(
                r'^\s*(?:(?:הפרויקט|הציוד|המסמך|הגיליון|the\s+project|the\s+equipment|'
                r'the\s+document|the\s+worksheet|the\s+workbook)\s+'
                r'(?:כולל|כוללת|כוללים|כוללות|מפרט|מציין|includes?|contains?|lists?)\s+)?'
                + re.escape(head) + r'(?![\w./-])', recovery_body, re.I)
        if not exact_head and not _primary_identifier_witness(
                recovery_body, row, head, evidence, hebrew, assess):
            # Accept a plural form only when its first item and exact second
            # head word match the source and the count is proved by that
            # same row's single positive quantity field.
            from decimal import Decimal, InvalidOperation
            opening = re.match(
                r'^\s*(?:(?:הפרויקט|הציוד|המסמך|הגיליון|ההצעה|the\s+project|'
                r'the\s+equipment|the\s+document|the\s+worksheet|the\s+workbook|'
                r'the\s+offer)\s+(?:כולל|כוללת|כוללים|כוללות|מפרט|מציין|'
                r'includes?|contains?|lists?)\s+)?'
                r'(?P<count>[A-Za-zא-ת]+)\s+(?P<item>[A-Za-zא-ת]+)\s+'
                r'(?P<identifier>[A-Za-zא-ת]+)(?![\w./-])', recovery_body, re.I)
            head_words = head.split()
            if not opening or len(head_words) != 2:
                return None
            claimed_item, source_item = opening.group('item').casefold(), head_words[0].casefold()
            same_item = (claimed_item == source_item or
                bool(re.fullmatch(r'[א-ת]{3,}ת', source_item) and
                     claimed_item == source_item[:-1] + 'ות'))
            if not same_item or opening.group('identifier').casefold() != head_words[1].casefold():
                return None
            count = _FORMS.get(opening.group('count').casefold(), '')
            quantities = [value.strip() for label, _, value, _, _ in _fields(row.source_claim)
                          if re.fullmatch(r'(?:qty|quantity|count|כמות)', label.strip(), re.I)]
            try:
                if (not re.fullmatch(r'\d+', str(count)) or len(quantities) != 1
                        or Decimal(quantities[0]) != Decimal(count) or Decimal(count) <= 0):
                    return None
            except InvalidOperation:
                return None
        for _, _, value, _, _ in _fields(row.source_claim):
            if (any(pattern.search(value) for pattern in condition_patterns)
                    or _STATUS.search(value.translate(normalize))):
                return None
        rendered = render_overview_multiplier_disclosure(fact, hebrew)
        if not rendered:
            return None
        if assess(rendered + ' [' + identity + ']', evidence, hebrew,
                  check_coverage=False).accepted:
            return rendered
        return None

    if len(rows) == 1:
        head = undisputed_head(ids[0], rows[0])
        if head:
            return ((head, (ids[0],)),)
        from workbook_context_recovery import recover_exact_item_with_context
        exact = recover_exact_item_with_context(body, ids, evidence, hebrew, assess)
        if exact:
            return exact
        # The identifier/head witness can only release an independently
        # assessed source field. Original rejected wording stays unresolved.
        from workbook_primary_item_recovery import recover_primary_source_field
        primary = recover_primary_source_field(body, ids, evidence, hebrew, assess)
        if primary:
            return primary

    if len(rows) == 1:
        from workbook_list_recovery import recover_source_list
        recovered = recover_source_list(body, ids, evidence, hebrew, assess)
        if recovered:
            return recovered

        recovered = _source_head_list_recovery(body, ids[0], evidence, hebrew, assess)
        if recovered:
            return recovered

        # A literal, unique product field identifies the row; a matching
        # leading description identifies the item the model was describing.
        # No generated property is retained: the complete source description
        # must independently pass the normal final assessment. The caller
        # keeps the rejected model sentence unresolved.
        row = rows[0]
        product_label = re.compile(r'(?:product(?:\s+name)?|item(?:\s+name)?|פריט|מוצר)', re.I)
        detail_label = re.compile(r'(?:description|details|specification|'
                                  r'תיאור(?:\s+מפורט)?|תאור(?:\s+מפורט)?|מפרט)', re.I)
        fields = _fields(row.source_claim)
        products = [value.strip() for label, _, value, _, _ in fields
                    if product_label.fullmatch(label.strip()) and value.strip()]
        if len(products) == 1:
            product = products[0]
            # A period followed by another identifier character remains an
            # extension (e.g. CODE.2). Permit a following period only at a
            # genuine sentence/closing boundary, never before more token text.
            witness = re.compile(r'(?<![\w./-])' + re.escape(product) +
                                 r'(?![\w/-])(?!\.(?![\s"”׳״)\]}]|$))', re.I)
            owners = {p.citation_id for p in evidence.passages
                      for label, _, value, _, _ in _fields(p.source_claim)
                      if product_label.fullmatch(label.strip()) and witness.search(value)}
            quantities = [value.strip() for label, _, value, _, _ in fields
                          if re.fullmatch(r'(?:qty|quantity|count|כמות)', label.strip(), re.I)]
            from decimal import Decimal, InvalidOperation
            try:
                quantity_ok = all(Decimal(value).is_finite() and Decimal(value) > 0
                                  for value in quantities)
            except InvalidOperation:
                quantity_ok = False
            neutral_head = re.sub(
                r'^(?:הפרויקט|הציוד|המסמך|הגיליון|the\s+project|the\s+equipment|'
                r'the\s+document|the\s+worksheet|the\s+workbook)\s+'
                r'(?:כולל|כוללת|כוללים|כוללות|מפרט|מציין|includes?|contains?|lists?)\s+',
                '', body.strip(), flags=re.I)
            tokens = lambda value: re.findall(r'[A-Za-z0-9]+|[א-ת]+', value.casefold())
            head = tokens(neutral_head)
            matches = []
            if (owners == {row.citation_id} and witness.search(body) and quantity_ok):
                for label, _, value, _, _ in fields:
                    words = tokens(value)
                    if (not detail_label.fullmatch(label.strip()) or len(words) < 2
                            or head[:2] != words[:2]
                            or len(value) > HARD_QUOTE_MAX_CHARS
                            or len(value.split()) > HARD_QUOTE_MAX_WORDS):
                        continue
                    # The product must attach directly to the item's opening
                    # words, not to a later nested accessory. Model-introduction
                    # terms are neutral grammar, not evidence of a property.
                    head_spans = list(re.finditer(r'[A-Za-z0-9]+|[א-ת]+', neutral_head))
                    if len(head_spans) < 2:
                        continue
                    following = neutral_head[head_spans[1].end():]
                    product_match = witness.search(following)
                    if (not product_match or not re.fullmatch(
                            r'\s*(?:(?:model|דגם|מדגם|מסוג)\s+)?',
                            following[:product_match.start()], flags=re.I)):
                        continue
                    if assess(value + ' [' + ids[0] + ']', evidence, hebrew,
                              check_coverage=False).accepted:
                        matches.append(value)
            if len(matches) == 1:
                return ((matches[0], (ids[0],)),)

    token = re.compile(r'[A-Za-z]+|[א-ת]+')
    neutral = frozenset(('the a an and or with includes include contains document source '
        'workbook worksheet project equipment items item system listed includes optional excluded '
        'המסמך המקור הגיליון הפרויקט הציוד הפריטים מופיע מופיעה מופיעים מופיעות '
        'מפרט מציין כוללת כולל כוללים כוללות וכן וגם את של מערכת עם על ידי לצד אופציונלי אופציונלית '
        'לשינוי').split())

    def forms(word):
        result = set(_workbook_word_forms(word))
        if re.fullmatch(r'[א-ת]{4,}', word) and word[0] in 'והבלכמ':
            result.update(_workbook_word_forms(word[1:]))
        # Hebrew feminine/construct singular and plural often differ only in
        # their ending. Use this solely to locate a candidate original field;
        # the entire source-owned field is independently assessed below.
        result.update(form[:-1] for form in tuple(result)
                      if re.fullmatch(r'[א-ת]{5,}', form) and form.endswith('ת'))
        return result

    def words(value):
        return [w.casefold() for w in token.findall(value)
                if len(w) >= 2 and w.casefold() not in neutral]

    def concise_field(value, row_conditions, identity):
        """Use a complete leading item phrase before an incidental capability."""
        boundary = re.search(
            r'(?<![A-Za-zא-ת])(?:תמיכה|כולל(?:ת|ים|ות)?|supports?|includes?|including)'
            r'(?![A-Za-zא-ת])', value, re.I)
        if not boundary:
            return value
        prefix = value[:boundary.start()].strip(' \t,;:-')
        omitted = value[boundary.start():]
        # The suffix may qualify the entire item, carry an example or status,
        # or state a count of items. Do not discard any of those.
        protected = re.compile(
            r'\b(?:if|when|provided|unless|only|optional|excluded|except|not|without|'
            r'planned|proposed|intended|existing|example|illustrative|side|units?|items?|pieces?|qty|quantity)\b|'
            r'(?<![א-ת])[וש]?(?:אם|כאשר|במידה|בלבד|למעט|ללא|בלי|לא|כגון|לדוגמא|אופציונלי(?:ת|ים|ות)?|'
            r'בכפוף|כפוף|מותנה|מתוכנן(?:ת|ים|ות)?|מוצע(?:ת|ים|ות)?|'
            r'קיים(?:ת|ים|ות)?|צד|כמות|יחידות|יחידה|פריטים|פריט)(?![א-ת])', re.I)
        if (len(words(prefix)) < 2 or not any(any(forms(w) & forms(s) for s in words(prefix))
                                             for w in wanted)
                # A later number may qualify or contradict the earlier
                # numeric assertion. Preserve the full field instead of
                # selecting one value from an ambiguous source description.
                or (re.search(r'\d', prefix) and re.search(r'\d', omitted))
                or protected.search(omitted) or _QUALIFIER.search(omitted)
                or _STRICT_CONDITION.search(omitted) or _EXAMPLE_MARKER.search(omitted)
                or _PROTECTED.search(omitted) or prohibited.search(omitted)
                or _STATUS.search(omitted.translate(normalize)) or _MONEY.search(omitted)
                or prefix.count('"') % 2 or prefix.count('“') != prefix.count('”')
                or re.search(r'(?:\b(?:of|for|with|and|or)|(?<![א-ת])(?:של|עם|על|או))$', prefix, re.I)):
            return value
        prefix_conditions = {m.group().casefold() for pattern in
            condition_patterns for m in pattern.finditer(prefix)}
        if not row_conditions <= prefix_conditions:
            return value
        decision = assess(prefix + ' [' + identity + ']', evidence, hebrew, check_coverage=False)
        return prefix if decision.accepted else value

    wanted = words(recovery_body)
    if not wanted:
        return ()
    composite = re.compile(r'(?<![A-Za-z0-9])(?=[A-Za-z0-9/-]*[A-Za-z])'
                           r'(?=[A-Za-z0-9/-]*\d)[A-Za-z0-9]+(?:[-/][A-Za-z0-9]+)+'
                           r'(?![A-Za-z0-9])', re.I)
    wanted_identifiers = {m.group().casefold() for m in composite.finditer(body)}
    proposed_conditions = tuple(m.group().casefold() for pattern in
        condition_patterns for m in pattern.finditer(body))
    results = []
    for identity, row in zip(ids, rows):
        all_fields = _fields(row.source_claim)
        fields = [(label, value.strip()) for label, _, value, _, _ in all_fields
                  if value.strip() and re.fullmatch(
                      r'(?:product(?:\s+name)?|item(?:\s+name)?|description|details|'
                      r'specification|equipment|device|work|service|'
                      r'פריט|מוצר|תיאור(?:\s+(?:כללי|מפורט))?|'
                      r'תאור(?:\s+(?:כללי|מפורט))?|ציוד|מפרט|עבודה|שירות)', label, re.I)]
        if not fields:
            continue
        # Unclassified row cells can carry an item's condition or state even
        # though they cannot safely serve as a standalone item description.
        # Do not silently erase such context by presenting a neighboring field.
        recognized = {label for label, _ in fields}
        if any(label not in recognized and (
                _QUALIFIER.search(value) or _STRICT_CONDITION.search(value)
                or _PROTECTED.search(value) or prohibited.search(value)
                or _EXAMPLE_MARKER.search(value)
                or re.search(r'\bincluded\b|(?<![א-ת])כלול(?:ה|ים|ות)?(?![א-ת])', value, re.I))
               for label, _, value, _, _ in all_fields):
            continue
        # This source-bound disclosure has a stricter item and quantity proof
        # than the generic field score and must not be downgraded to a head.
        disclosure = undisputed_head(identity, row)
        if disclosure:
            results.append((disclosure, (identity,)))
            continue
        # A qualifier elsewhere in the row cannot be lost by selecting the
        # shorter item-name field. Prefer a complete field carrying every
        # source qualification, or decline this row.
        row_conditions = {m.group().casefold() for _, value in fields for pattern in
                          condition_patterns
                          for m in pattern.finditer(value)}
        # A short item-name field can identify its qualified description,
        # even when the latter adds an example or a different-language term.
        item_anchor = any(
            re.fullmatch(r'(?:product(?:\s+name)?|item(?:\s+name)?|equipment|device|'
                         r'תיאור\s+כללי|תאור\s+כללי|פריט|מוצר|ציוד)', label, re.I)
            and bool(words(value))
            and all(any(forms(w) & forms(s) for s in wanted) for w in words(value))
            for label, value in fields)
        candidates = []
        for label, value in fields:
            # Preserve complete source fields, including late conditions.
            # A display excerpt is independently bounded by the quote gate.
            if len(value) > HARD_QUOTE_MAX_CHARS or len(value.split()) > HARD_QUOTE_MAX_WORDS:
                continue
            if len(ids) == 1:
                # A rejected one-row assertion cannot use this broad group
                # fallback to replace a wrong item, numeric property, example
                # or optionality. Recover only an exact source-owned
                # impersonal notice whose qualifying action is also present
                # in the proposed claim; its surrounding context remains
                # separately unresolved by the caller.
                notice = re.match(r'^\s*(יש\s+[א-ת]+)(?![א-ת])', value)
                if not notice or not re.search(
                        r'(?<![א-ת])' + re.escape(notice.group(1)) + r'(?![א-ת])', body):
                    continue
            source = words(value)
            score = sum(any(forms(w) & forms(s) for w in wanted) for s in set(source))
            conditions = {m.group().casefold() for pattern in
                          condition_patterns
                          for m in pattern.finditer(value)}
            if not row_conditions <= conditions:
                continue
            identifier_anchor = bool(wanted_identifiers &
                {m.group().casefold() for m in composite.finditer(value)})
            qualified_detail = bool(item_anchor and row_conditions and conditions
                and re.fullmatch(r'(?:description|details|specification|'
                                 r'תיאור(?:\s+מפורט)?|תאור(?:\s+מפורט)?|מפרט)', label, re.I))
            # Shared modifiers alone cannot identify a different source item.
            # A model ID or a complete anchored same-row item name can supply
            # identity when the detailed field begins in another language.
            local_script = [word for word in source if re.fullmatch(r'[א-ת]+', word)]
            head = local_script[0] if local_script and any(re.search(r'[א-ת]', w) for w in wanted) else (source[0] if source else '')
            head_anchor = bool(head and any(forms(head) & forms(w) for w in wanted))
            if not (head_anchor or item_anchor or identifier_anchor):
                continue
            if ((not score and not qualified_detail and not identifier_anchor) or
                    (len(source) > 1 and score < 2 and not item_anchor
                     and not identifier_anchor)):
                continue
            # Conditions asserted in the model fragment cannot silently
            # disappear. Only evaluate those that also occur in this row.
            relevant = set(proposed_conditions) & row_conditions
            if not relevant <= conditions:
                continue
            candidates.append((score, label, value))
        if not candidates:
            continue
        for score in sorted({score for score, _, _ in candidates}, reverse=True):
            matches = {value for candidate_score, _, value in candidates if candidate_score == score}
            if len(matches) != 1:
                # A weaker match cannot resolve ambiguity among stronger
                # source fields; it may refer to a different item altogether.
                break
            value = matches.pop()
            decision = assess(value + ' [' + identity + ']', evidence, hebrew, check_coverage=False)
            if decision.accepted:
                results.append((concise_field(value, row_conditions, identity), (identity,)))
                break
    return tuple(results)
