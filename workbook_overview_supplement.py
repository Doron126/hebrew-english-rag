"""Bounded, source-scoped supplement for workbook project overviews.

Grouping ranks candidates only. The original immutable passages and the shared
section assessor remain the sole factual/citation authority.
"""
from __future__ import annotations

from dataclasses import replace
from decimal import Decimal, InvalidOperation
import re


def _grouping_view(passage):
    """Exclude only already-recognized scalar metadata from a grouping view.

    Unknown fields remain present so the existing grouping code fails closed.
    This view is never used to validate a claim, quote, or source identity.
    """
    from workbook_project import _fields, _QUANTITY
    from workbook_context_recovery import _SCALAR_FIELD, _PRICE_FIELD, _SCALAR, _FORMULA
    fields = _fields(passage.source_claim)
    if not fields or len({column for _, column, _, _, _ in fields}) != len(fields):
        return None
    kept = []
    for label, column, value, _, _ in fields:
        label = label.strip()
        if _PRICE_FIELD.fullmatch(label):
            if not (_SCALAR.fullmatch(value) or _FORMULA.fullmatch(value)):
                return None
            continue
        if _SCALAR_FIELD.fullmatch(label) and not _QUANTITY.fullmatch(label):
            if not _SCALAR.fullmatch(value):
                return None
            continue
        kept.append((label, column, value))
    if not kept:
        return None
    # _groups needs only a row label and labelled fields. This copy is a
    # non-authoritative selection view, never a replacement source claim.
    view = 'Spreadsheet row 1:\n' + '\n'.join(
        '- ' + label + ' (' + column + '): ' + value
        for label, column, value in kept)
    return replace(passage, source_claim=view)


def _summary_row(passage):
    """Keep aggregate/financial totals on the existing total-proof path."""
    from workbook_project import _fields
    from workbook_context_recovery import _ITEM_FIELD
    items = [value.strip() for label, _, value, _, _ in _fields(passage.source_claim)
             if _ITEM_FIELD.fullmatch(label.strip()) and value.strip()]
    return bool(items and all(re.match(r'^(?:total|subtotal|סה.?כ)(?:\W|$)', value, re.I)
                              for value in items))


def _safe_literal_candidates(passage):
    """Yield exact eligible source fields; never join item/property roles."""
    from workbook_project import _fields
    from workbook_context_recovery import _ITEM_FIELD, _SCALAR_FIELD, _NUMBER
    fields = _fields(passage.source_claim)
    if not fields or len({column for _, column, _, _, _ in fields}) != len(fields):
        return ()
    quantities = [value.strip() for label, _, value, _, _ in fields
                  if re.fullmatch(r'(?:qty|quantity|count|כמות)', label.strip(), re.I)]
    if len(quantities) > 1:
        return ()
    if quantities:
        if not re.fullmatch(_NUMBER, quantities[0]):
            return ()
        try:
            quantity = Decimal(quantities[0].replace(',', ''))
            if not quantity.is_finite() or quantity <= 0:
                return ()
        except InvalidOperation:
            return ()
    candidates = []
    for label, _column, value, _, _ in fields:
        value = value.strip()
        if (not _ITEM_FIELD.fullmatch(label.strip()) or not value
                or len(value) > 300 or '\n' in value):
            continue
        # Price and count fields cannot act as standalone items.
        if _SCALAR_FIELD.fullmatch(label.strip()):
            continue
        candidates.append((label.strip(), value))
    # A description explains an item more often than a short product code.
    description = re.compile(
        r'(?:description|details|specification|תיאור(?:\s+מפורט)?|'
        r'תאור(?:\s+מפורט)?|מפרט)', re.I)
    descriptions = [item for item in candidates if description.fullmatch(item[0])]
    if len(descriptions) > 1 or len(candidates) - len(descriptions) > 1:
        return ()
    if descriptions:
        candidates = descriptions
    return tuple(value for _label, value in candidates)


def supplement_overview(evidence, bindings, hebrew, assess, *, excluded_ids=()):
    """Return source-exact, independently assessed gaps as (claim, IDs).

    No extra model call or semantic review is performed. This is a bounded
    supplemental selection, never a completeness or priority proof.
    """
    from workbook_project import WorkbookProjectEvidence, _spread_positions
    from workbook_overview_coverage import _groups, _binding_mentions_group
    if not (isinstance(evidence, WorkbookProjectEvidence)
            and evidence.overview_synthesis
            and evidence.complete and evidence.selection_complete):
        return ()
    passages = tuple(evidence.passages)
    by_id = {p.citation_id: p for p in passages}
    eligible = evidence.quote_eligible_ids
    if (len(by_id) != len(passages) or not isinstance(eligible, (list, tuple))
            or len(set(eligible)) != len(eligible) or not set(eligible) <= set(by_id)):
        return ()
    selected = set(eligible)
    if not isinstance(excluded_ids, (list, tuple, set, frozenset)) or any(
            not isinstance(identity, str) for identity in excluded_ids):
        return ()
    excluded = set(excluded_ids)
    sections = {}
    for passage in passages:
        if passage.citation_id not in selected:
            continue
        meta = passage.metadata
        key = (passage.source_id, passage.tenant_id, meta.get('sheet'),
               meta.get('_workbook_scope'))
        if (not all(key[:3]) or key[3] is None
                or passage.tenant_id != evidence.tenant_id):
            continue
        if _summary_row(passage):
            continue
        view = _grouping_view(passage)
        # An unknown field disqualifies this passage, without inventing its
        # meaning or hiding it from the original coverage calculation.
        if view is None or _groups((view,)) is None:
            continue
        sections.setdefault(key, []).append(view)
    gaps = []
    for scope in sections.values():
        groups = _groups(scope)
        if groups is None:
            continue
        for group in groups:
            if not _binding_mentions_group(group, groups, bindings):
                gaps.append(group)
    if not gaps:
        return ()
    # Interleave positions to avoid exhausting the budget at the start of a
    # section. The cap applies only to supplemental fragments, not the final
    # answer, and does not imply the omitted groups are unimportant.
    order = tuple(dict.fromkeys(_spread_positions(len(gaps))))
    limit = min(4, len(gaps))
    additions = []
    seen = set()
    for position in order:
        group = gaps[position]
        # Use original passage order, never set iteration, for reproducibility.
        for original in passages:
            identity = original.citation_id
            if identity not in group['ids'] or identity in seen or identity in excluded:
                continue
            for value in _safe_literal_candidates(original):
                claim = value + ' [' + identity + ']'
                decision = assess(claim, evidence, hebrew, check_coverage=False)
                if (decision.accepted and decision.bindings == ((claim, (identity,)),)):
                    additions.append((claim, (identity,)))
                    seen.add(identity)
                    break
            if identity in seen:
                break
        if len(additions) >= limit:
            break
    return tuple(additions)


def round_trip_overview_paragraphs(claims, evidence, hebrew, assess):
    """Group verified claim lines only if the original bindings round-trip."""
    original = '\n\n'.join(claims)
    if not getattr(evidence, 'overview_synthesis', False) or len(claims) < 2:
        return original
    before = assess(original, evidence, hebrew, check_coverage=False)
    if not before.accepted:
        return original
    # The application renders soft line breaks as spaces. Only group claims
    # that already contain a sentence boundary adjacent to their citations;
    # otherwise grouping creates an unreadable run-on even when citations bind.
    terminal = re.compile(
        r'(?:[.!?]\s*(?:\[\d{1,3}\]\s*)+|(?:\[\d{1,3}\]\s*)+[.!?])$')
    if any(not terminal.search(claim.strip()) for claim in claims):
        return original
    # A Markdown soft line break keeps the independent claim bytes and citation
    # markers intact; no new factual connective or sentence is introduced.
    candidate = '\n\n'.join('\n'.join(claims[i:i + 3])
                            for i in range(0, len(claims), 3))
    after = assess(candidate, evidence, hebrew, check_coverage=False)
    return candidate if after.accepted and after.bindings == before.bindings else original
