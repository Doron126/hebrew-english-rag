"""Tenant-scoped, source-revalidated evidence for explicit mobile lookups."""
from __future__ import annotations

from dataclasses import dataclass, replace
import re
import zipfile
from xml.etree import ElementTree

from document_reading import ReadingEvidence, ScanWitness, make_passage, scan_authorized, source_body
from contact_lookup import authenticated_docx_mobile_cells, mobile_request


@dataclass(frozen=True)
class ContactEvidence(ReadingEvidence):
    tenant: str = ''
    document_root: str = ''
    question: str = ''
    number: str = ''
    person: str = ''


def _norm(value):
    return ' '.join(str(value or '').split())


def _identity(value):
    value = str(value or '').strip()
    return '' if '\n' in value or '/' in value else _norm(value)


def _organization(value):
    # Preserve all parenthetical boundaries and source-owned words.
    value = _identity(value)
    value = re.sub(r'\(\s*', '(', value)
    value = re.sub(r'\s*\)', ')', value)
    return _norm(value)


def _mentions(group, person):
    if person in _norm(source_body(group)):
        return True
    # Splitters may put the boundary inside a word. Join just adjacent chunk
    # edges (after removing breadcrumbs), including any actual overlap. A
    # false candidate is safe: the authenticated original must resolve it.
    bodies = [re.sub(r'^\[[^\]\n]+\]\s*\n+', '', str(row.get('document') or ''))
              for row in group]
    for left, right in zip(bodies, bodies[1:]):
        overlap = next((n for n in range(min(len(left), len(right)), 0, -1)
                        if left.endswith(right[:n])), 0)
        # Equal characters at a boundary do not themselves prove overlap:
        # they can be consecutive characters in the underlying name. Check
        # both interpretations; original-source validation resolves matches.
        for start in {0, overlap}:
            joined = left[-len(person) * 2:] + right[start:start + len(person) * 2]
            if person in _norm(joined):
                return True
    return False


def _resolve(rows, complete, tenant, root, question, person=None):
    """Reopen every indexed source mentioning the person; uncertain means no answer."""
    from contact_pdf import pdf_contact_rows
    from source_access import docx_source_body
    from contact_intent import validate_contact_interpretation
    person = person or (mobile_request(question).person if mobile_request(question) else '')
    if not validate_contact_interpretation(question, person) or not complete:
        return None
    grouped = {}
    for row in rows:
        meta = row.get('metadata') or {}
        if meta.get('tenant_id') != tenant:
            return None
        grouped.setdefault(meta.get('source_id'), []).append(row)
    candidates = [group for group in grouped.values() if _mentions(group, person)]
    found = []
    for group in candidates:
        meta = group[0]['metadata']
        if any(any(item['metadata'].get(k) != meta.get(k) for k in
                   ('source_id', 'tenant_id', 'source_sha256', 'source_relative_path',
                    'source_size', 'source_mtime_ns', 'mime_type')) for item in group):
            return None
        try:
            mime = meta.get('mime_type')
            if mime == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                native, resolved = authenticated_docx_mobile_cells(root, tenant, meta, person)
                if not resolved or not native:
                    return None
                body, original_complete = docx_source_body(root, tenant, meta)
                if not original_complete:
                    return None
                for cell in native:
                    found.append((cell.value, (_organization(cell.organization), _identity(cell.role)),
                                  group, cell, body))
            elif mime == 'application/pdf':
                native = pdf_contact_rows(root, tenant, meta, person)
                if not native:
                    return None
                for cell in native:
                    found.append((cell.value, (_organization(cell.organization), _identity(cell.role)),
                                  group, cell, None))
            elif mime == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                from contact_xlsx import xlsx_has_no_phone_evidence
                if not xlsx_has_no_phone_evidence(root, tenant, meta, person):
                    return None
            else:
                return None
        except (OSError, ValueError, RuntimeError, TypeError, KeyError,
                zipfile.BadZipFile, ElementTree.ParseError):
            return None
    if not found or len({item[0] for item in found}) != 1:
        return None
    # Paragraph slots alone cannot establish visual association. Each such
    # DOCX proposal needs an independently authenticated PDF table with the
    # same entire name/number population, their ordering, roles and organization.
    checked = []
    for item in found:
        value, identity, group, cell, body = item
        if getattr(cell, 'layout_dependent', False):
            compatible = [other for other in found if getattr(other[3], 'layout_group', False)
                and identity[0] and identity[0] == other[1][0]
                and cell.group_names == other[3].group_names
                and cell.group_values == other[3].group_values
                and cell.group_roles and cell.group_roles == other[3].group_roles
                and all(cell.group_roles)]
            if not compatible or len({_identity(other[3].role) for other in compatible}) != 1:
                return None
            identity = (identity[0], _identity(compatible[0][3].role))
        checked.append((value, identity, group, cell, body))
    found = checked
    # Independently represented copies must share uniquely headed organization
    # and role fields. Bare identical phone strings do not establish identity.
    for i, left in enumerate(found):
        for right in found[i + 1:]:
            if not all(left[1]) or left[1] != right[1]:
                return None
    available = [item for item in found if item[4] is not None]
    if not available:
        return None
    number, _context_values, group, _cell, body = available[0]
    # Cite a literal short contiguous span in the native DOCX rendering. The
    # header/column association is independently checked above from native XML.
    blocks = re.finditer(r'(?ms)^Table row \d+:\n.*?(?=\n\n|\Z)', body)
    spans = []
    for block in blocks:
        lines = list(re.finditer(r'(?m)^- [^\n]+$', block.group()))
        def contains_cell(line, target):
            return target in [part.strip() for part in line.group().split(': ', 1)[-1].split(' / ')]
        name_lines = [line for line in lines if contains_cell(line, person)]
        phone_lines = [line for line in lines if contains_cell(line, number)]
        if len(name_lines) == len(phone_lines) == 1:
            start = block.start() + min(name_lines[0].start(), phone_lines[0].start())
            end = block.start() + max(name_lines[0].end(), phone_lines[0].end())
            claim = body[start:end]
            if len(claim) <= 800 and len(claim.split()) <= 80:
                spans.append((start, claim))
    if len(spans) != 1:
        return None
    start, claim = spans[0]
    row = group[0]
    fact = make_passage(body, claim, row['metadata'], row.get('id'), 1, source_start=start)
    return replace(fact, representation='docx_original'), number, body


def collect_contact(retriever, tenant, root, question, *, person=None, where=None):
    person = person or (mobile_request(question).person if mobile_request(question) else '')
    grouped, complete, count = scan_authorized(retriever, tenant, root, where=where)
    rows = tuple(row for group in grouped.values() for row in group)
    result = _resolve(rows, complete, tenant, root, question, person)
    witness = ScanWitness(rows, complete, result[2] if result else '', bool(result))
    return ContactEvidence((result[0],) if result else (), bool(result), len(grouped), count,
        '' if result else 'contact_unavailable_or_ambiguous', witness,
        tenant, str(root), question, result[1] if result else '', person)


def render_contact(evidence):
    return f'הטלפון הנייד של {evidence.person} הוא {evidence.number}. [1]' if evidence.person and evidence.number else ''


def validate_contact(evidence, contract, answer):
    from contact_intent import validate_contact_interpretation
    if (not isinstance(evidence, ContactEvidence) or not evidence.witness
            or evidence.question != contract.normalized_request
            or not validate_contact_interpretation(evidence.question, evidence.person)
            or contract.requirements[0].subject != evidence.person):
        return False
    result = _resolve(evidence.witness.rows, evidence.witness.storage_complete,
                      evidence.tenant, evidence.document_root, evidence.question, evidence.person)
    return bool(result and evidence.complete and evidence.passages == (result[0],)
                and evidence.number == result[1] and evidence.witness.original_body == result[2]
                and answer == render_contact(evidence))
