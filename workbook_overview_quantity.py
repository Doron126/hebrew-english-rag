"""Quantity guard for source-owned workbook overview claims.

An explicitly zero, invalid, or multiply declared row quantity cannot establish
that an item was selected.  This is intentionally scoped to overview answers;
targeted questions may instead ask what an original zero means.
"""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re

_QUANTITY = re.compile(r'(?:qty|quantity|count|כמות)', re.I)
_NUMBER = re.compile(r'[+\-]?(?:\d+(?:\.\d+)?|\.\d+|\d{1,3}(?:,\d{3})+(?:\.\d+)?)')


def _quantity_allows_item(passage):
    """Missing quantity is unknown; explicit quantity must be unique and > 0."""
    from workbook_project import _fields
    fields = _fields(passage.source_claim)
    quantities = [(column, value.strip()) for label, column, value, _, _ in fields
                  if _QUANTITY.fullmatch(label.strip())]
    if not quantities:
        return True
    if len(quantities) != 1 or len({column for _, column, _, _, _ in fields}) != len(fields):
        return False
    value = quantities[0][1]
    if not _NUMBER.fullmatch(value):
        return False
    try:
        number = Decimal(value.replace(',', ''))
    except InvalidOperation:
        return False
    return number.is_finite() and number > 0


def _designated_qualification(body, identity, row):
    """An exact independent section qualification can survive its host row."""
    from workbook_project import _fields
    metadata = getattr(row, 'metadata', {})
    identities = metadata.get('_workbook_general_qualification_ids', ())
    designated = metadata.get('_workbook_general_qualification_fields', ())
    if not isinstance(identities, (tuple, list)) or not isinstance(designated, (tuple, list)):
        return False
    if identity not in identities:
        return False
    norm = lambda text: ' '.join(str(text).split())
    actual_fields = {norm(value) for _, _, value, _, _ in _fields(row.source_claim)}
    normalized = norm(body)
    return bool(normalized and normalized in actual_fields and
                any(normalized == norm(value) for value in designated))


def overview_quantity_allows_claim(body, identities, evidence, hebrew):
    """Protect overview item claims while preserving proven totals/conditions."""
    from workbook_project import (WorkbookProjectEvidence, workbook_overview_total,
                                  workbook_total_tax_limitation)
    if not isinstance(evidence, WorkbookProjectEvidence) or not evidence.overview_synthesis:
        return True
    ids = tuple(identities)
    by_id = {p.citation_id: p for p in evidence.passages}
    if not ids or len(set(ids)) != len(ids) or len(by_id) != len(evidence.passages):
        return False
    if any(identity not in by_id for identity in ids):
        return False
    if all(_quantity_allows_item(by_id[identity]) for identity in ids):
        return True
    # These totals are authoritative only when recomputed by the producer
    # from this evidence; matching source numbers or copied prose is not enough.
    for total in (workbook_overview_total(evidence, hebrew),
                  workbook_total_tax_limitation(evidence, hebrew)):
        if total and body == total[0] and ids == total[1]:
            return True
    return (len(ids) == 1 and _designated_qualification(body, ids[0], by_id[ids[0]]))


def overview_quantity_allows_excerpt(evidence):
    """Literal excerpt fast path must not list zero-quantity items as selected."""
    from workbook_project import WorkbookProjectEvidence
    if not isinstance(evidence, WorkbookProjectEvidence) or not evidence.overview_synthesis:
        return True
    return all(_quantity_allows_item(passage) for passage in evidence.passages)
