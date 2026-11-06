"""Conservative recovery of a complete source-owned coordinated list."""
import re


def _same_list(source, claim):
    claim = re.sub(r'^(?:המסמך|הפרויקט|הציוד|הגיליון)\s+'
                   r'(?:מפרט|מציין|כולל|כוללת|כוללים|כוללות)\s+', '', claim.strip(), count=1)
    marker = re.compile(r'(?<![א-ת])(?:הכוללת|הכולל|כוללת|כולל)(?![א-ת])')
    sm, cm = list(marker.finditer(source)), list(marker.finditer(claim))
    if len(sm) != 1 or len(cm) != 1:
        return False

    def work(value):
        parts = re.split(r'\s*,+\s*|\s+ו(?=[א-ת])', value.strip())
        return tuple(' '.join(p.split()) for p in parts if p.strip())

    sw, cw = work(source[:sm[0].start()]), work(claim[:cm[0].start()])
    if len(sw) < 3 or sw != cw or any(len(p) < 2 for p in sw):
        return False
    # Exactly two bare coordinated words under the same literal relation.
    # The Hebrew lamed before the positive numeral is handled only here.
    template = (r'\s*(?P<head>[א-ת]+)\s+(?P<one>[א-ת]+)\s+ו(?P<two>[א-ת]+)'
                r'\s+%s(?P<suffix>\d+\s+.+?)\s*')
    st = re.fullmatch(template % '', source[sm[0].end():])
    ct = re.fullmatch(template % 'ל-', claim[cm[0].end():])
    return bool(st and ct and st.group('head') == ct.group('head')
                and sorted((st.group('one'), st.group('two'))) ==
                    sorted((ct.group('one'), ct.group('two')))
                and st.group('one') != st.group('two')
                and ' '.join(st.group('suffix').split()) == ' '.join(ct.group('suffix').split()))


def recover_source_list(body, identities, evidence, hebrew, assess):
    """Return an exact field; the original rejected claim stays unresolved."""
    from workbook_context_recovery import _ITEM_FIELD, _recovery_row_proof
    from evidence_quotes import HARD_QUOTE_MAX_CHARS, HARD_QUOTE_MAX_WORDS
    ids = tuple(identities)
    proof = _recovery_row_proof(ids, evidence, hebrew, assess, require_quantity=True,
                                allow_designated_qualification=True)
    if not proof:
        return ()
    row, fields, accounted = proof
    matches = []
    for index, (label, _, value, _, _) in enumerate(fields):
        value = value.strip()
        if (not _ITEM_FIELD.fullmatch(label.strip()) or not value
                or len(value) > HARD_QUOTE_MAX_CHARS or len(value.split()) > HARD_QUOTE_MAX_WORDS
                or not _same_list(value, body)):
            continue
        if any(i != index and other.strip() and not accounted(other_label, other)
               for i, (other_label, _, other, _, _) in enumerate(fields)):
            continue
        if assess(value + ' [' + ids[0] + ']', evidence, hebrew, check_coverage=False).accepted:
            matches.append(value)
    return ((matches[0], ids),) if len(matches) == 1 else ()
