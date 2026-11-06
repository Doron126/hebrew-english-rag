"""Authenticated, row-bound evidence for project information in workbooks.

Workbook paths propose a source; explicit project headings and row-level
project fields take precedence.  Every released row is reconstructed from the
same authenticated original through the ingestion workbook reader.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
import json
import os
import re

from document_reading import ReadingEvidence, ScanWitness, make_passage, scan_authorized
from source_access import (SourceAccessError, open_source_file, source_metadata_complete,
                           source_sha256_matches_metadata, source_stat_matches_metadata)

MAX_PROJECT_ROWS = 80
MAX_PROJECT_CHARS = 30000
MAX_PROMPT_CHARS = 5000
MAX_PROMPT_ROWS = 24
_ROW = re.compile(r'(?m)^Spreadsheet row ([1-9][0-9]*):\n')
_SHEET = re.compile(r'(?m)^# Sheet: ([^\n]+)$')
_FIELD = re.compile(r'(?m)^- (?:(?P<label>[^\n:]+?) \((?P<column>[A-Z]{1,3})\)|(?P<bare_column>[A-Z]{1,3})): (?P<value>[^\n]*(?:\n(?!- )[^\n]+)*)')
_OWNER = re.compile(r'^(?:project(?:\s+(?:name|site))?|site|client|customer|שם\s+(?:ה)?פרויקט|פרויקט|הפרויקט|אתר|לקוח)\s*[:\-–—]\s*(.+)$', re.I)
_OWNER_LABEL = re.compile(r'^(?:project(?:\s+(?:name|site))?|site|client|customer|שם\s+פרויקט|פרויקט|הפרויקט|אתר|לקוח)$', re.I)
_ITEM = re.compile(r'\b(?:product(?:\s+name)?|item(?:\s+name)?|equipment|device|part(?:\s+name)?|description|details|specification|scope|work|service|notes?|remarks?)\b|(?:תי?אור|פריט|מוצר|ציוד|עבודה|שירות|הערות|מפרט)', re.I)
_LINK_LABEL = re.compile(r'\b(?:links?|urls?|hyperlinks?)\b|קישור', re.I)
_PRICE = re.compile(r'^(?:unit\s+price|price\s+per\s+unit|total\s+price|line\s+total|מחיר\s+יחידה|מחיר\s+כולל|סה.?כ\s+מחיר)$', re.I)
_TOTAL_PRICE = re.compile(r'^(?:total\s+price|line\s+total|מחיר\s+כולל|סה.?כ\s+מחיר)$', re.I)
# A free-standing note about a particular subject (for example travel
# costs) does not establish the currency of the table below it. Accept only
# an unqualified, table-wide price/cost label with an explicit currency.
_TABLE_CURRENCY_PREFIX = re.compile(
    r'(?:(?:עלות|עלויות|מחיר|מחירים|העלות|העלויות|המחיר|המחירים|סכום|הסכום)'
    r'(?:\s+(?:שלנו|הכולל))?\s*ב|'
    r'(?:(?:our|all)\s+)?(?:costs?|prices?|amounts?)(?:\s+(?:are|is))?\s+in)\s*', re.I)
_VAT = re.compile(
    r'(?<![A-Za-z])v[.\s]*a[.\s]*t\.?(?![A-Za-z])|'
    r'\bvalue[ -]added\s+tax(?:es)?\b|'
    r'(?<!\w)מ\s*[.\"׳״\x27]?\s*ע\s*[.\"׳״\x27]?\s*מ(?!\w)|'
    r'מס\s+ערך\s+מוסף', re.I)
_TAX = re.compile(r'\btax(?:es|ation)?\b|(?<!\w)[והבלכמ]?(?:מס|מסים|מיסים|מיסוי)(?!\w)', re.I)
_CURRENCY = re.compile(r'(?:₪|\b(?:ILS|NIS|shekels?|USD|EUR|GBP|dollars?|euros?|pounds?)\b|שקלים|שקל|דולרים|דולר|אירו|יורו|ליש[\"״]?ט)', re.I)
_QUANTITY = re.compile(r'\b(?:qty|quantity|count|units?)\b|(?:כמות|יחידות)', re.I)
# A section-wide condition must name the entire set as the grammatical
# subject, or explicitly say that a condition applies to the entire set.
# Merely mentioning every item as the object of an accessory note is not a
# condition on those items.
_UNIVERSAL_EN = (r'(?:all\s+(?:(?:listed|the)\s+)?(?:items?|products?|devices?|sections?|services?|equipment|rows?)|'
                 r'(?:each|every)\s+(?:item|product|device|section|service|equipment|row))')
_UNIVERSAL_HE = r'כל\s+(?:הסעיפים|הפריטים|המוצרים|המכשירים|הציוד|העבודות|השירותים)'
_GENERAL_QUALIFIER = re.compile(
    r'^\s*(?:' + _UNIVERSAL_EN + r'\b|' + _UNIVERSAL_HE + r'(?!\w))|'
    r'\bwarranty\s+(?:is\s+)?(?:applies\s+to|covers)\s+' + _UNIVERSAL_EN + r'\b', re.I)
_QUALIFIER = re.compile(r'\b(?:optional|unless|only|except|required|warranty|not|without|subject\s+to|conditional\s+(?:on|upon)|provided\s+that|pending\s+approval|upon\s+approval)\b|'
                        + _UNIVERSAL_EN + r'\b|'
                        r'(?<!\w)(?:אופציונלי(?:י?ם|ות|ת)?|למעט|בכפוף|כפוף|כפופה|כפופים|מותנה|מותנית|מותנים|בהתקיים|לאחר\s+אישור|אחריות|'
                        + _UNIVERSAL_HE + r'|יש\s+לבחון|חובה|ללא|בלי|לא|רק|אם)(?!\w)', re.I)
_OWNER_CONFLICT = re.compile(r'\b(?:not|without|except|neither|nor|and|or)\b|'
                             r'(?<!\w)(?:לא|ללא|בלי|למעט|או|וגם)(?!\w)', re.I)

@dataclass(frozen=True)
class WorkbookProjectEvidence(ReadingEvidence):
    subject: str = ''
    document_root: str = ''
    tenant_id: str = ''
    where_source: str = ''
    selection_complete: bool = True
    quote_eligible_ids: tuple[str, ...] | None = None
    semantic_claims: tuple = ()
    semantic_review: str = ''
    semantic_parse_rejections: tuple = ()
    overview_synthesis: bool = False
    overview_question: str = ''
    overview_claims: tuple = ()
    overview_review: object = None
    overview_tenant: str = ''
    overview_raw_topic_draft: str = ''
    # For a targeted technical fact, selection uses an explicit project scope
    # while the immutable request subject still retains the item and property.
    narrow_scope: str = ''
    narrow_target: str = ''
    narrow_property: str = ''

    @property
    def vat_mentioned_in_extracted_text(self):
        # This reports only a search of reconstructed workbook text, not VAT
        # inclusion, exclusion, or the content of drawings/embedded objects.
        if not self.complete or not self.passages:
            return None
        values = [p.metadata.get('_workbook_vat_mentioned_in_extracted_text')
                  for p in self.passages]
        return any(values) if all(type(value) is bool for value in values) else None

    @property
    def tax_mentioned_in_extracted_text(self):
        # Generic tax wording can qualify a total without identifying VAT.
        if not self.complete or not self.passages:
            return None
        values = [p.metadata.get('_workbook_tax_mentioned_in_extracted_text')
                  for p in self.passages]
        return any(values) if all(type(value) is bool for value in values) else None


def _candidates(subject):
    from subject_intent import project_location_candidates
    return project_location_candidates(subject) or (str(subject),)


def _matching_candidates(subject, text, *, owner=False):
    from monetary_questions import _word_variants, _words
    from subject_reading import _terms
    value = str(text or '')
    if owner and _OWNER_CONFLICT.search(value):
        return frozenset()
    grammar = {'the', 'a', 'an', 'at', 'in', 'of', 'for', 'and', 'or', 'של', 'על', 'את', 'או'}
    # Preserve occurrences and their order. Deduplicating the entire source
    # would make a later complete name disappear after an earlier prefix.
    observed = _word_variants(tuple(word for word in _words(value) if word not in grammar))
    result = set()
    for i, candidate in enumerate(_candidates(subject)):
        wanted = _word_variants(_terms(candidate))
        if not wanted or len(wanted) > len(observed):
            continue
        positions = (0,) if owner else range(len(observed)-len(wanted)+1)
        if any(all(wanted[j] & observed[start+j] for j in range(len(wanted)))
               for start in positions):
            result.add(i)
    return frozenset(result)


def _match(subject, text):
    return bool(_matching_candidates(subject, text))


def _common_owners(subject, owners):
    candidates = set(range(len(_candidates(subject))))
    for owner in owners:
        candidates.intersection_update(_matching_candidates(subject, owner, owner=True))
    return frozenset(candidates)


def _owner_identities(owners):
    """Preserve the whole explicit identity, not just matching query words."""
    from subject_reading import _terms
    return frozenset(' '.join(_terms(owner)) for owner in owners if _terms(owner))
def _original_workbook_layout(path):
    """Read merged sheet ranges from the same validated original XLSX."""
    import posixpath
    from ingest import (_validated_zip, _parse_xml_part, _read_zip_part,
        _SHEET_NS, _OFFICE_REL_NS, _PACKAGE_REL_NS, _xlsx_cell_coordinates,
        IngestExtractionError)
    archive = _validated_zip(path, 'XLSX')
    try:
        workbook = _parse_xml_part(_read_zip_part(archive, 'xl/workbook.xml', required=True), 'xl/workbook.xml')
        rel_name = 'xl/_rels/workbook.xml.rels'
        rels = _parse_xml_part(_read_zip_part(archive, rel_name, required=True), rel_name)
        relationships = {node.get('Id') or '': node.get('Target') or ''
                         for node in rels.findall(f'.//{{{_PACKAGE_REL_NS}}}Relationship')}
        layout = {}
        for sheet in workbook.findall(f'.//{{{_SHEET_NS}}}sheets/{{{_SHEET_NS}}}sheet'):
            name = sheet.get('name') or 'Untitled sheet'
            if name in layout:
                raise IngestExtractionError('duplicate XLSX sheet name')
            target = relationships.get(sheet.get(f'{{{_OFFICE_REL_NS}}}id') or '')
            if not target:
                raise IngestExtractionError('missing worksheet relationship')
            normalized = target.lstrip('/')
            if not normalized.startswith('xl/'):
                normalized = posixpath.join('xl', normalized)
            normalized = posixpath.normpath(normalized)
            if not re.fullmatch(r'xl/worksheets/[^/]+\.xml', normalized):
                raise IngestExtractionError('unsafe XLSX worksheet target')
            root = _parse_xml_part(_read_zip_part(archive, normalized, required=True), normalized)
            ranges = []
            for cell in root.findall(f'.//{{{_SHEET_NS}}}mergeCells/{{{_SHEET_NS}}}mergeCell'):
                bounds = (cell.get('ref') or '').split(':', 1)
                first = _xlsx_cell_coordinates(bounds[0]) if bounds else None
                last = _xlsx_cell_coordinates(bounds[-1]) if bounds else None
                if not first or not last or last[0] < first[0] or last[1] < first[1]:
                    raise IngestExtractionError('invalid XLSX merged-cell range')
                ranges.append((first[0], first[1], last[0], last[1]))
            layout[name] = tuple(ranges)
        return layout
    finally:
        archive.close()


def _original_workbook(document_root, tenant, metadata, *, include_layout=False):
    """Read the indexed workbook generation using its held source descriptor."""
    from ingest import _read_xlsx_text, IngestExtractionError
    if not source_metadata_complete(metadata, tenant):
        raise SourceAccessError('incomplete workbook identity')
    relative, handle, before = open_source_file(document_root, tenant, metadata['source_relative_path'])
    with handle:
        if relative.suffix.casefold() != '.xlsx' or not source_stat_matches_metadata(metadata, before):
            raise SourceAccessError('workbook source generation changed')
        if not source_sha256_matches_metadata(metadata, handle, enabled=True):
            raise SourceAccessError('workbook checksum changed')
        # The open descriptor remains held while the ingestion ZIP reader
        # opens this descriptor-backed path; no tenant pathname is reopened.
        try:
            source_path = Path('/proc/self/fd') / str(handle.fileno())
            text = _read_xlsx_text(source_path)
            layout = _original_workbook_layout(source_path) if include_layout else None
        except (OSError, ValueError, IngestExtractionError) as exc:
            raise SourceAccessError('workbook extraction failed') from exc
        after = os.fstat(handle.fileno())
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) !=
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                or not source_sha256_matches_metadata(metadata, handle, enabled=True)):
            raise SourceAccessError('workbook changed while reading')
        return (text, layout) if include_layout else text


def _sheets(text):
    starts = list(_SHEET.finditer(text))
    for i, start in enumerate(starts):
        end = starts[i+1].start() if i+1 < len(starts) else len(text)
        yield start.group(1).strip(), text[start.end():end].strip()


def _marker(value):
    match = _OWNER.fullmatch(value.strip())
    return match.group(1).strip() if match else ''


def _fields(row_text):
    return tuple(((m.group('label') or m.group('bare_column')).strip(),
                  m.group('column') or m.group('bare_column'),
                  m.group('value').strip(), m.start(), m.end())
                 for m in _FIELD.finditer(row_text))


def _heading_entries(sheet_body):
    # Only explicit project labels are ownership evidence. Ordinary body
    # mentions do not acquire ownership of neighboring rows.
    before = _ROW.search(sheet_body)
    head = sheet_body[:before.start()] if before else sheet_body
    entries = []
    for line in head.splitlines():
        value = re.sub(r'^- Cell [A-Z]+[0-9]+:\s*', '', line).strip()
        owner = _marker(value)
        if not owner:
            # Common source title form: a declared document type followed by
            # a separated project name. Do not treat arbitrary notes as
            # project ownership merely because they mention a location.
            title = re.fullmatch(r'(?:כתב\s+כמויות|תכנון|project\s+(?:plan|scope)|bill\s+of\s+quantities)\b.{0,120}?\s+[\-–—]\s+(.+)', value, re.I)
            owner = title.group(1).strip() if title else ''
        if owner:
            entries.append((owner, value))
    return tuple(entries)


def _heading_owners(sheet_body):
    return tuple(owner for owner, _ in _heading_entries(sheet_body))


def _owner_columns(sheet_body):
    match = re.search(r'(?m)^Columns:\s*(.+)$', sheet_body)
    result = set()
    for part in match.group(1).split(';') if match else ():
        item = re.fullmatch(r'\s*([A-Z]{1,3})\s*=\s*(.*?)\s*\([A-Z]{1,3}\)\s*', part)
        if item and _OWNER_LABEL.fullmatch(item.group(2)):
            result.add(item.group(1))
    return frozenset(result)
def _row_owner(fields):
    owners = []
    for label, _, value, _, _ in fields:
        if _OWNER_LABEL.fullmatch(label):
            owners.append(value)
        elif len({entry[2] for entry in fields}) == 1:
            # Ingestion expands merged cells; repeated identical section
            # labels still describe one explicit source heading.
            marker = _marker(value)
            if marker and marker not in owners:
                owners.append(marker)
    return tuple(owners)


def _row_projection(row_text, fields):
    selected = [i for i, (label, _, _, _, _) in enumerate(fields)
                if not _LINK_LABEL.search(label) and
                (_ITEM.search(label) or _QUANTITY.fullmatch(label.strip()))]
    if not selected or not any(_ITEM.search(fields[i][0]) and fields[i][2] for i in selected):
        return ''
    # Repeated table headers are structure, not equipment. Summary lines are
    # not individual items and must not become project scope claims.
    if len(fields) >= 2 and sum(label.casefold() == value.casefold()
                                 for label, _, value, _, _ in fields) >= 2:
        return ''
    item_values = [fields[i][2] for i in selected if _ITEM.search(fields[i][0])]
    if item_values and all(re.fullmatch(r'(?:total|subtotal|סה.?כ|סה.?כ\s+מחיר)',
                                        value.strip(), re.I) for value in item_values):
        return ''
    # Keep a literal contiguous row prefix. Include explicitly labeled unit
    # and line prices, but stop before unrelated comparison columns.
    last = max(selected)
    for label, _, _, _, _ in fields[:last+1]:
        # Prices can legitimately precede quantity or description. Retain
        # only positively identified unit/line-price roles; an unknown
        # financial column cannot acquire a guessed relationship to the row.
        if (re.search(r'\b(?:price|cost|total|amount)\b|(?:מחיר|עלות|סכום|סה.?כ)', label, re.I)
                and not _PRICE.fullmatch(label)):
            return ''
    while last + 1 < len(fields) and _PRICE.fullmatch(fields[last+1][0]):
        last += 1
    return row_text[:fields[last][4]].rstrip()


def _spread_positions(length):
    """Return a bounded source-order-independent spread over a section."""
    if length <= 0:
        return ()
    seen, result = set(), []
    for denominator in range(1, min(length, MAX_PROJECT_ROWS) + 1):
        for numerator in range(denominator + 1):
            position = round((length - 1) * numerator / denominator)
            if position not in seen:
                seen.add(position)
                result.append(position)
                if len(result) >= min(length, MAX_PROJECT_ROWS):
                    return tuple(result)
    return tuple(result)


def _bounded_owned_sheet_plan(sheet_parts, subject, path, fallback):
    """Budget the existing row cap fairly across source-owned sheets.

    The source text is already extracted. This census does not retrieve more
    documents or expand the prompt; it only changes which bounded original
    rows can be represented. Ownership transitions remain checked later.
    """
    plans = []
    for sheet, body in sheet_parts:
        if re.search(r'(?m)^Sheet state:\s*(?!visible\s*$)', body):
            continue
        markers = list(_ROW.finditer(body))
        if not markers:
            continue
        headings = _heading_owners(body)
        explicit = bool(headings and _common_owners(subject, headings))
        sheet_owner = _marker(sheet)
        explicit |= bool(sheet_owner and _matching_candidates(subject, sheet_owner, owner=True))
        explicit |= bool(fallback and _match(subject, PurePosixPath(path).name))
        mandatory = set()
        for i, marker in enumerate(markers):
            stop = markers[i+1].start() if i+1 < len(markers) else len(body)
            fields = _fields(body[marker.start():stop])
            owners = _row_owner(fields)
            explicit |= bool(owners and _common_owners(subject, owners))
            # Section-wide terms and standalone qualifying notes cannot be
            # dropped while retaining apparently unqualified nearby facts.
            if any(_GENERAL_QUALIFIER.search(value) for _, _, value, _, _ in fields):
                mandatory.add(i)
            elif len(fields) == 1 and _QUALIFIER.search(fields[0][2]):
                mandatory.add(i)
        if not explicit:
            continue
        preamble = body[:markers[0].start()]
        contexts = sum(1 for m in re.finditer(
            r'(?m)^Spreadsheet context row ([1-9][0-9]*):\n(- Cell ([A-Z]+)([1-9][0-9]*): ([^\n]+))', preamble)
            if m.group(1) == m.group(4) and
            (_CURRENCY.search(m.group(5)) or _VAT.search(m.group(5)) or _TAX.search(m.group(5))))
        plans.append({'sheet': sheet, 'count': len(markers), 'mandatory': mandatory,
                      'contexts': contexts, 'quota': 0})
    remaining = max(0, MAX_PROJECT_ROWS - min(MAX_PROJECT_ROWS, sum(x['contexts'] for x in plans)))
    while remaining:
        progress = False
        for item in plans:
            if remaining and item['quota'] < min(item['count'], MAX_PROJECT_ROWS):
                item['quota'] += 1
                remaining -= 1
                progress = True
        if not progress:
            break
    result = {}
    for item in plans:
        mandatory, quota = item['mandatory'], item['quota']
        if len(mandatory) > quota:
            result[item['sheet']] = None
            continue
        chosen = set(mandatory)
        for position in _spread_positions(item['count']):
            if len(chosen) >= quota:
                break
            chosen.add(position)
        result[item['sheet']] = frozenset(chosen)
    return result


def _source_passages(metadata, subject, text, citation_start):
    sheet_parts = list(_sheets(text))
    if not sheet_parts:
        return (), False, False, frozenset()
    path = str(metadata['source_relative_path'])
    path_match = _match(subject, PurePosixPath(path).name)
    # A filename can only supply workbook-level identity when the workbook has
    # one sheet and no explicit conflicting body identity. Multiple sheets
    # need a matching sheet or project heading for each selected section.
    all_head_owners = [owner for _, body in sheet_parts for owner in _heading_owners(body)]
    # Inspect all explicit row owners before enabling the filename fallback.
    # A later section heading must prevent an earlier unrelated row from
    # inheriting a filename-derived project identity.
    explicit_rows = []
    for _, body in sheet_parts:
        markers = list(_ROW.finditer(body))
        for i, marker in enumerate(markers):
            end = markers[i+1].start() if i+1 < len(markers) else len(body)
            explicit_rows.extend(_row_owner(_fields(body[marker.start():end])))
    fallback = len(sheet_parts) == 1 and path_match and not all_head_owners and not explicit_rows
    sheet_plan = _bounded_owned_sheet_plan(sheet_parts, subject, path, fallback)
    passages = []
    incomplete = False
    unclear_coverage = False
    matched_identity = False
    matched_owners = set()
    blocked_scopes = set()
    for sheet, body in sheet_parts:
        if re.search(r'(?m)^Sheet state:\s*(?!visible\s*$)', body):
            continue
        head_entries = _heading_entries(body)
        head_owners = tuple(owner for owner, _ in head_entries)
        sheet_owner = _marker(sheet)
        active = (_common_owners(subject, head_owners) if head_owners
                  else _matching_candidates(subject, sheet_owner, owner=True) if sheet_owner
                  else _matching_candidates(subject, PurePosixPath(path).name) if fallback
                  else frozenset())
        active_heading = tuple(value for owner, value in head_entries
                               if _common_owners(subject, (owner,))) if active else ()
        matched_identity = matched_identity or bool(active)
        if active:
            matched_owners.update(_owner_identities(head_owners or ((sheet_owner,) if sheet_owner else ())))
        rows = list(_ROW.finditer(body))
        planned = sheet_plan.get(sheet)
        if sheet in sheet_plan and planned is None:
            incomplete = True
            continue
        owner_columns = _owner_columns(body)
        first_row = _ROW.search(body)
        preamble = body[:first_row.start()] if first_row else body
        # A separate contextual qualification is not a row fact. Unless it
        # can be included with a proven cell reference, exclude this section
        # rather than detach that qualification from its governed items.
        context_values = re.findall(r'(?m)^- Cell [A-Z]+[0-9]+:\s*(.+)$', preamble)
        if active and any(_QUALIFIER.search(value) for value in context_values if not _marker(value)):
            blocked_scopes.add((sheet, 0))
            unclear_coverage = True
        known_section_owner = bool(head_owners or sheet_owner or fallback)
        section_number = 0
        if active and not any(_QUALIFIER.search(value) for value in context_values if not _marker(value)):
            for context in re.finditer(r'(?m)^Spreadsheet context row ([1-9][0-9]*):\n(- Cell ([A-Z]+)([1-9][0-9]*): ([^\n]+))', preamble):
                value = context.group(5)
                if (context.group(1) != context.group(4) or
                        not (_CURRENCY.search(value) or _VAT.search(value) or _TAX.search(value))):
                    continue
                claim = context.group(0)
                cell = context.group(3) + context.group(4)
                m = dict(metadata)
                m.update(sheet=sheet, row=int(context.group(1)), section=f'Sheet: {sheet} / {cell}',
                         _workbook_range=cell, _workbook_subject=subject, _workbook_scope=0,
                         _workbook_heading=active_heading,
                         _workbook_owner_candidates=tuple(sorted(active)))
                chunk_id = f"{metadata['source_id']}::workbook::{sheet}::{cell}"
                passage = make_passage(claim, claim, m, chunk_id, str(citation_start+len(passages)))
                passages.append(replace(passage, representation='workbook_original'))
        for i, marker in enumerate(rows):
            stop = rows[i+1].start() if i+1 < len(rows) else len(body)
            row_text = body[marker.start():stop].strip()
            # Drop the next rendering section header while preserving an exact
            # source-prefix used to validate the projected cell text.
            row_text = re.split(r'\n\n## (?:Rows? )', row_text, maxsplit=1)[0].rstrip()
            fields = _fields(row_text)
            owners = _row_owner(fields)
            # A standalone heading remains active until the next heading. A
            # project column belongs only to its row; it cannot silently lend
            # its identity to a later row with an empty owner cell. Explicit
            # merged cells are expanded by the source reader for each row.
            heading = bool(owners and len({entry[2] for entry in fields}) == 1
                           and any(_marker(entry[2]) for entry in fields))
            if heading:
                active = _common_owners(subject, owners)
                active_heading = tuple(value for _, _, value, _, _ in fields
                                       if _marker(value) and _common_owners(subject, (_marker(value),))) if active else ()
                matched_identity = matched_identity or bool(active)
                if active:
                    matched_owners.update(_owner_identities(owners))
                known_section_owner = True
                section_number += 1
                continue
            # If a table declares an owner column, its omission cannot borrow
            # the prior heading. Merged owners are explicitly expanded by the
            # original XLSX reader and therefore still appear in this row.
            missing_row_owner = bool(owner_columns and not any(
                column in owner_columns and _OWNER_LABEL.fullmatch(label)
                for label, column, _, _, _ in fields))
            row_active = (frozenset() if missing_row_owner else
                          _common_owners(subject, owners) if owners else active)
            if not row_active:
                if fields and (missing_row_owner or not known_section_owner and not owners):
                    unclear_coverage = True
                continue
            matched_identity = True
            matched_owners.update(_owner_identities(owners))
            if planned is not None and i not in planned:
                incomplete = True
                continue
            claim = _row_projection(row_text, fields)
            # A qualifier outside the evidence projection might apply to the
            # entire section or its row. Never retain nearby unqualified facts
            # when the relevant source condition cannot be represented.
            omitted_qualifier = any(
                _QUALIFIER.search(label + ' ' + value)
                for label, _, value, start, end in fields
                if not claim or end > len(claim))
            if omitted_qualifier:
                blocked_scopes.add((sheet, section_number))
                unclear_coverage = True
                continue
            if not claim and len(fields) == 1:
                # A cached total is usable only if the actual SUM formula
                # covers a contiguous set of already accepted rows from this
                # same owned section and the summed column is retained in
                # every one. A last or largest number alone is not a total.
                label, column, value, _, _ = fields[0]
                total = re.fullmatch(r'Formula: =SUM\(([A-Z]{1,3})([1-9][0-9]*):([A-Z]{1,3})([1-9][0-9]*)\); Cached value: ([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))', value, re.I)
                if total and _TOTAL_PRICE.fullmatch(label) and total.group(1).upper() == column == total.group(3).upper():
                    first, last = int(total.group(2)), int(total.group(4))
                    section_rows = [p for p in passages if p.metadata.get('sheet') == sheet
                                    and p.metadata.get('_workbook_scope') == section_number
                                    and first <= p.metadata.get('row', 0) <= last]
                    if (first <= last < int(marker.group(1)) and last - first < MAX_PROJECT_ROWS and
                            {p.metadata['row'] for p in section_rows} == set(range(first, last+1)) and
                            all(any(f[1] == column for f in _fields(p.source_claim)) for p in section_rows)):
                        claim = row_text
            if not claim:
                continue
            if len(passages) >= MAX_PROJECT_ROWS or sum(len(p.text) for p in passages)+len(claim) > MAX_PROJECT_CHARS:
                incomplete = True
                continue
            row = int(marker.group(1))
            used = _fields(claim)
            if not used:
                continue
            columns = [x[1] for x in used]
            cell_range = f'{columns[0]}{row}:{columns[-1]}{row}' if len(columns)>1 else f'{columns[0]}{row}'
            m = dict(metadata)
            m.update(sheet=sheet, row=row, section=f'Sheet: {sheet} / {cell_range}',
                     _workbook_range=cell_range, _workbook_subject=subject, _workbook_scope=section_number,
                     _workbook_heading=() if owners else active_heading,
                     _workbook_owner_candidates=tuple(sorted(row_active)))
            chunk_id = f"{metadata['source_id']}::workbook::{sheet}::{cell_range}"
            passage = make_passage(claim, claim, m, chunk_id, str(citation_start+len(passages)))
            passages.append(replace(passage, representation='workbook_original'))
    passages = [p for p in passages
                if (p.metadata.get('sheet'), p.metadata.get('_workbook_scope')) not in blocked_scopes]
    # Explicit universal wording is the only basis for requiring a separate
    # qualification in every answer using this source-owned section.
    general = {}
    general_fields = {}
    for passage in passages:
        scope = (passage.metadata.get('sheet'), passage.metadata.get('_workbook_scope'))
        values = tuple(value for _, _, value, _, _ in _fields(passage.source_claim)
                       if _GENERAL_QUALIFIER.search(value))
        if values:
            general.setdefault(scope, []).append(passage.citation_id)
            general_fields[passage.citation_id] = values
    bound = []
    for passage in passages:
        metadata = dict(passage.metadata)
        scope = (metadata.get('sheet'), metadata.get('_workbook_scope'))
        metadata['_workbook_general_qualification_ids'] = tuple(general.get(scope, ()))
        metadata['_workbook_general_qualification_fields'] = general_fields.get(passage.citation_id, ())
        bound.append(replace(passage, metadata=metadata))
    conflict = bool(path_match and not matched_identity and (all_head_owners or explicit_rows))
    return tuple(bound), not (incomplete or unclear_coverage), conflict, frozenset(matched_owners)


def compact_workbook_passage(passage):
    """Prompt-only view; the full original claim remains the citation authority.

    Row labels come from the actual workbook; prices remain in the retained
    source claim. For an overview the row descriptions and quantities carry
    scope, while a separately proven overall total keeps its exact source
    formula and cached value. This does not alter evidence or citations.
    """
    fields = _fields(passage.source_claim)
    if len(fields) <= 1:
        return passage.source_claim
    values = [f'{label} ({column}): {value}' for label, column, value, _, _ in fields
              if not _PRICE.fullmatch(label)]
    if not values:
        return passage.source_claim
    return f"Spreadsheet row {passage.metadata.get('row')}: " + '; '.join(values)


def workbook_prompt_lines(passages, *, _selection_view=False):
    """Represent ordered owned rows with their authenticated source context.

    Context is repeated when a group changes, so it cannot be accidentally
    attributed to a different sheet or project section. The bracketed IDs
    belong to rows only; headings are context, not independent citations.
    """
    passages = tuple(passages)
    # A single context already identifies its source unambiguously; avoid
    # adding a source alias when it cannot deduplicate any later context.
    keys = {(p.source_id, p.metadata.get('sheet'), p.metadata.get('_workbook_scope'),
             tuple(p.metadata.get('_workbook_heading', ()))) for p in passages}
    use_alias = not _selection_view and len(keys) > 1
    lines, previous, sources = [], None, {}
    for passage in passages:
        metadata = passage.metadata
        heading = tuple(metadata.get('_workbook_heading', ()))
        key = (passage.source_id, metadata.get('sheet'),
               metadata.get('_workbook_scope'), heading)
        if key != previous:
            if not use_alias:
                # A single source context needs no alias. The same format is
                # also used solely for stable source-selection sizing.
                context = {'file': metadata.get('display_name') or '',
                           'sheet': metadata.get('sheet') or '', 'heading': list(heading)}
            else:
                # The request-local alias follows the authenticated source
                # ID, not its filename: distinct workbooks can share a name.
                # Repeat the alias on each section, including returns.
                first_source = passage.source_id not in sources
                if first_source:
                    sources[passage.source_id] = 'S' + str(len(sources) + 1)
                context = {'source': sources[passage.source_id]}
                if first_source:
                    context['file'] = metadata.get('display_name') or ''
                context.update(sheet=metadata.get('sheet') or '', heading=list(heading))
            lines.append('Source context: ' + json.dumps(context, ensure_ascii=False, separators=(',', ':')))
            previous = key
        lines.append('[' + passage.citation_id + '] ' + compact_workbook_passage(passage))
    return lines


def _prompt_size(passages):
    return len('\n'.join(workbook_prompt_lines(passages)))


def _projection_size(passages):
    """Stable selection heuristic; not the transmitted prompt or token count."""
    return len('\n'.join(workbook_prompt_lines(passages, _selection_view=True)))


def _prompt_item_group(passage):
    """Use the workbook's own item label to spread a bounded section sample.

    This is a selection heuristic, not a claim that the labels exhaust every
    category or prove semantic equivalence. Full row evidence is retained.
    """
    fields = _fields(passage.source_claim)
    preferred = [value for label, _, value, _, _ in fields if re.fullmatch(
        r'(?:product(?:\s+name)?|item(?:\s+name)?|equipment|device|'
        r'תיאור\s+כללי|תאור\s+כללי|פריט|מוצר|ציוד)', label.strip(), re.I)]
    if not preferred:
        preferred = [value for label, _, value, _, _ in fields if re.fullmatch(
            r'(?:description|details|specification|תיאור(?:\s+מפורט)?|'
            r'תאור(?:\s+מפורט)?|מפרט)', label.strip(), re.I)]
    for value in preferred:
        if _GENERAL_QUALIFIER.search(value) or _QUALIFIER.fullmatch(value.strip()):
            continue
        words = re.findall(r'[A-Za-z]+|[א-ת]+', value.casefold())
        if not words:
            continue
        # System/equipment are structural prefixes; retain the following word
        # to keep distinct workbook sections represented within a sheet.
        if words[0] in {'system', 'equipment', 'מערכת', 'המערכת', 'ציוד', 'הציוד'} and len(words) > 1:
            return tuple(words[:2])
        return (words[0],)
    return ()


def _prompt_projection(passages):
    """Bound one model request and represent each authenticated section.

    Prioritize literal qualifications, then distribute remaining space over
    all known source/sheet/section groups. The size check matches compact_workbook_passage, while every selected
    item retains its full original source claim for validation and citation.
    """
    passages = tuple(passages)
    if (len(passages) <= MAX_PROMPT_ROWS and _projection_size(passages) <= MAX_PROMPT_CHARS
            and _prompt_size(passages) <= MAX_PROMPT_CHARS):
        return passages, True
    from collections import OrderedDict
    groups = OrderedDict()
    for index, passage in enumerate(passages):
        key = (passage.source_id, passage.metadata.get('sheet'), passage.metadata.get('_workbook_scope'))
        groups.setdefault(key, []).append(index)
    selected = set()
    def projected(indices):
        return tuple(p for i, p in enumerate(passages) if i in indices)
    def choose(index):
        candidate = selected | {index}
        if (index not in selected and len(candidate) <= MAX_PROMPT_ROWS
                and _projection_size(projected(candidate)) <= MAX_PROMPT_CHARS
                and _prompt_size(projected(candidate)) <= MAX_PROMPT_CHARS):
            selected.add(index)
    allowed_groups = []
    for indices in groups.values():
        general = {identity for i in indices for identity in
                   passages[i].metadata.get('_workbook_general_qualification_ids', ())}
        qualified = [i for i in indices if passages[i].citation_id in general or
                     (len(_fields(passages[i].source_claim)) == 1 and
                      _QUALIFIER.search(passages[i].source_claim))]
        # Reserve section-wide terms and standalone qualifying notes. A
        # qualifier attached to one item remains intact when that item is
        # selected; it does not require selecting every other optional row.
        reserved = selected | set(qualified)
        if (len(reserved) > MAX_PROMPT_ROWS or _projection_size(projected(reserved)) > MAX_PROMPT_CHARS
                or _prompt_size(projected(reserved)) > MAX_PROMPT_CHARS):
            continue
        for index in qualified:
            choose(index)
        allowed_groups.append(indices)
    # Interleave sections and first distribute each section's budget across
    # its own distinct source item labels, before choosing repeated labels.
    # Keep the spread order within each phase and never treat this heuristic
    # as evidence that unselected content was covered.
    pending = []
    for indices in allowed_groups:
        length = len(indices)
        positions = []
        for position in (0, length-1, round((length-1)/3), round(2*(length-1)/3)):
            if position not in positions:
                positions.append(position)
        positions.extend(i for i in _spread_positions(length) if i not in positions)
        first, rest, signatures = [], [], set()
        for position in positions:
            index = indices[position]
            signature = _prompt_item_group(passages[index])
            if index in selected:
                continue
            if signature and signature not in signatures:
                signatures.add(signature)
                first.append(index)
            else:
                rest.append(index)
        pending.append(first + rest)
    while any(pending) and len(selected) < MAX_PROMPT_ROWS:
        for values in pending:
            if values:
                choose(values.pop(0))
    return tuple(p for i, p in enumerate(passages) if i in selected), False
def _ordinal_section_total(rows):
    """Prove an explicitly currency- and VAT-labelled sum from source rows."""
    from decimal import Decimal, InvalidOperation
    # An original sheet row cannot contain two cells at one coordinate.
    # Embedded text that resembles a second field must not let the first
    # cached value stand in for the actual line amount.
    if any(len({column for _, column, _, _, _ in fields}) != len(fields)
           for _, _, fields in rows):
        return None
    by_row = {row: fields for row, _, fields in rows}
    def total_labels(fields):
        return [value for label, _, value, _, _ in fields
                if _ITEM.search(label) and re.match(
                    r'^\s*ס[ה][\x22\x27\u05f3\u05f4]?כ(?:\s|$)', value)]
    total_rows = [row for row, _, fields in rows if total_labels(fields)]
    if len(total_rows) != 1:
        return None
    def numeric(value):
        value = value.strip()
        cached = re.fullmatch(r'Formula: =([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)); Cached value: ([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))', value)
        if cached:
            if Decimal(cached.group(1)) != Decimal(cached.group(2)):
                raise InvalidOperation
            return Decimal(cached.group(2))
        return Decimal(value)
    for row, _claim, fields in rows:
        labels = total_labels(fields)
        if len(labels) != 1:
            continue
        label = labels[0]
        # Explicit exclusion is part of the source's amount role. Other tax
        # wording is left for an ordinary source-cited answer.
        if not (re.search(r'(?<!\w)לא\s+כולל(?!\w)', label) and _VAT.search(label)):
            continue
        for field_label, column, value, _, _ in fields:
            if not re.search(r'(?:בש[\x22\x27\u05f3\u05f4]?ח|₪|שקלים)', field_label, re.I):
                continue
            match = re.fullmatch(
                r'Formula: =SUM\(([A-Z]{1,3})([1-9][0-9]*):([A-Z]{1,3})([1-9][0-9]*)\); Cached value: ([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))', value, re.I)
            if not match or match.group(1).upper() != column or match.group(3).upper() != column:
                continue
            first, last = int(match.group(2)), int(match.group(4))
            if first > last or last >= row or last-first >= MAX_PROJECT_ROWS or not all(n in by_row for n in range(first,last+1)):
                continue
            item_rows = {n for n, _, values in rows if n != row
                         and not total_labels(values)
                         and any(_ITEM.search(f[0]) for f in values)
                         and not (len(values) >= 2 and sum(f[0].casefold() == f[2].casefold()
                                                         for f in values) >= 2)}
            if item_rows != set(range(first, last+1)):
                continue
            try:
                values = []
                for n in range(first,last+1):
                    line = by_row[n]
                    quantities = [f[2] for f in line if re.fullmatch(
                        r'(?:qty|quantity|count|כמות)', f[0].strip(), re.I)]
                    prices = [f[2] for f in line if re.fullmatch(
                        r'(?:unit\s+price|price|מחיר(?:\s+יחידה)?)', f[0].strip(), re.I)]
                    if len(quantities) != 1 or len(prices) != 1:
                        raise InvalidOperation
                    quantity, price = numeric(quantities[0]), numeric(prices[0])
                    if not quantity.is_finite() or not price.is_finite() or quantity < 0 or price < 0:
                        raise InvalidOperation
                    cell = next(f[2] for f in line if f[1] == column)
                    cached = re.fullmatch(r'Formula: .*?; Cached value: ([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))', cell)
                    amount = Decimal(cached.group(1) if cached else cell)
                    if amount.quantize(Decimal('.01')) != (quantity * price).quantize(Decimal('.01')):
                        raise InvalidOperation
                    values.append(amount)
                total = Decimal(match.group(5))
                if sum(values, Decimal(0)).quantize(Decimal('.01')) != total.quantize(Decimal('.01')):
                    continue
            except (InvalidOperation, StopIteration):
                continue
            return row, str(total.quantize(Decimal('.01'))), label
    return None


def _ordinal_declared_owner(value, request, metadata):
    """Read an owner from an explicit label or the selected document title."""
    import unicodedata
    def same_heading(left, right):
        return (' '.join(unicodedata.normalize('NFKC', left).casefold().split()) ==
                ' '.join(unicodedata.normalize('NFKC', right).casefold().split()))
    marked = _marker(value)
    if marked:
        return marked
    document_words = re.findall(r'[A-Za-z0-9]+|[א-ת]+', request.document.casefold())[:-1]
    source_words = list(re.finditer(r'[A-Za-z0-9]+|[א-ת]+', value.casefold()))
    if not document_words or len(source_words) < len(document_words):
        return None
    for wanted, observed in zip(document_words, source_words):
        actual = observed.group()
        if wanted != actual and not (len(wanted) > 3 and wanted.startswith('ה') and wanted[1:] == actual):
            return None
    owner = value[source_words[len(document_words)-1].end():].strip(' :')
    parts = re.split(r'\s*[-–—]\s*', owner, maxsplit=1)
    if len(parts) == 2:
        parents = PurePosixPath(metadata.get('source_relative_path', '')).parts[:-1]
        if not any(same_heading(part, request.owner) for part in parents):
            return ''
        owner = parts[0].strip()
    return owner.strip()


def _ordinal_section_source(metadata, text, layout, request, subject):
    """Select a repeated named section in authenticated original sheet order.

    An ordinal is meaningful only when at least two separated instances of the
    same heading exist. Consecutive duplicate heading rows (including merged
    cell expansion) are one heading. Other repeated-cell headings terminate
    the selected section, preventing it from borrowing a later room's rows.
    """
    import unicodedata
    def same_heading(left, right):
        return ' '.join(unicodedata.normalize('NFKC', left).casefold().split()) ==                ' '.join(unicodedata.normalize('NFKC', right).casefold().split())

    def declared_owner(value):
        # A declared project/customer label is direct owner evidence. A
        # document title can also declare an owner immediately after the
        # requested document-type words; a dash-delimited location is a
        # separate title qualifier, not part of that owner.
        return _ordinal_declared_owner(value, request, metadata)

    candidates = []
    for sheet, body in _sheets(text):
        if re.search(r'(?m)^Sheet state:\s*(?!visible\s*$)', body):
            continue
        first_row = _ROW.search(body)
        preamble = body[:first_row.start()] if first_row else body
        contexts = [(int(m.group(1)), m.group(2).strip()) for m in re.finditer(
            r'(?m)^- Cell [A-Z]+([1-9][0-9]*): ([^\n]+)', preamble)]
        # Every explicit owner declaration in the sheet context must agree;
        # scattered body text and pathname alone never establish ownership.
        owners = [owner for _, value in contexts
                  if (owner := declared_owner(value)) is not None]
        if not owners or any(not same_heading(owner, request.owner) for owner in owners):
            continue
        from ingest import _xlsx_column_index
        header = re.search(r'(?m)^Columns:\s*(.+)$', body)
        columns = []
        for part in header.group(1).split(';') if header else ():
            item = re.fullmatch(r'\s*([A-Z]{1,3})\s*=\s*(.*?)\s*\([A-Z]{1,3}\)\s*', part)
            if item and re.fullmatch(
                    r'(?:product(?:\s+name)?|item(?:\s+name)?|description|equipment|'
                    r'qty|quantity|count|מוצר|פריט|ציוד|תיאור|תאור|כמות)',
                    item.group(2).strip(), re.I):
                columns.append(_xlsx_column_index(item.group(1)))
        if len(columns) < 2:
            continue
        def merged_heading(row):
            return any(r1 <= row <= r2 and c1 <= min(columns) and c2 >= max(columns)
                       for r1, c1, r2, c2 in layout.get(sheet, ()))
        row_matches = list(_ROW.finditer(body))
        row_data = []
        headings = [(number, value) for number, value in contexts
                    if same_heading(value, request.section) and merged_heading(number)]
        for i, marker in enumerate(row_matches):
            end = row_matches[i+1].start() if i+1 < len(row_matches) else len(body)
            row_text = re.split(r'\n\n## (?:Rows? )', body[marker.start():end].strip(), maxsplit=1)[0].rstrip()
            fields = _fields(row_text)
            row = int(marker.group(1))
            row_data.append((row, row_text, fields))
            values = [value for _, _, value, _, _ in fields]
            if (len(values) >= 2 and merged_heading(row)
                    and all(same_heading(values[0], value) for value in values)):
                headings.append((row, values[0]))
        groups = []
        for number, value in sorted(headings):
            if groups and number <= groups[-1][-1][0] + 1 and same_heading(value, groups[-1][0][1]):
                groups[-1].append((number, value))
            else:
                groups.append([(number, value)])
        matches = [g for g in groups if same_heading(g[0][1], request.section)]
        if len(matches) < 2 or len(matches) < request.ordinal:
            continue
        chosen = matches[request.ordinal-1]
        start = chosen[-1][0] + 1
        following = [g[0][0] for g in groups if g[0][0] > chosen[-1][0]]
        stop = min(following) if following else 10**9
        selected = [(row, row_text, fields) for row,row_text,fields in row_data
                    if start <= row < stop]
        # Do not infer a section from headings alone. A row in the section
        # must declare an item-bearing field, and unidentified single-field
        # records are not silently absorbed or dropped.
        if not selected or not any(any(_ITEM.search(f[0]) for f in fields)
                                   for _,_,fields in selected):
            continue
        owner_context = tuple(dict.fromkeys(value for _, value in contexts
                                            if declared_owner(value) is not None))
        candidates.append((sheet, selected, owner_context))
    if len(candidates) != 1:
        return (), False, len(candidates) > 1
    sheet, selected, owner_context = candidates[0]
    total = _ordinal_section_total(selected)
    passages = []
    qualification_fields = {}
    has_quantity_column = any(re.fullmatch(r'(?:qty|quantity|count|כמות)', f[0].strip(), re.I)
                              for _, _, fields in selected for f in fields)
    def configuration_notes(fields):
        return tuple(value for label, _, value, _, _ in fields
                     if re.match(r'^(?:בתצורה|בחלופה|בגרסה)\s+זו\b', value)
                     or _GENERAL_QUALIFIER.search(value))
    for row, claim, fields in selected:
        if not fields:
            continue
        # A lone cell inside a table can be an unrecognized section heading
        # or a free-standing condition. Neither may borrow the prior owner.
        if len(fields) == 1:
            return (), False, False
        # Repeated table headers are not notes or item facts, including
        # when one of their columns is itself headed Notes.
        if len(fields) >= 2 and sum(label.casefold() == value.casefold()
                                     for label, _, value, _, _ in fields) >= 2:
            continue
        notes = configuration_notes(fields)
        ancillary = [value for label, _, value, _, _ in fields
                     if re.fullmatch(r'(?:[A-Z]{1,3}|notes?|remarks?|הערות?)', label.strip(), re.I)
                     and value.strip()]
        # An unscoped side note may qualify the item or the whole section.
        # Do not discard it or guess which other rows it governs.
        if any(value not in notes for value in ancillary):
            return (), False, False
        if not any(_ITEM.search(label) for label, _, _, _, _ in fields):
            # Unidentified adjacent text may be a section-wide condition.
            # Refuse to invent its scope or silently discard it.
            return (), False, False
        quantities = [value.strip() for label, _, value, _, _ in fields
                      if re.fullmatch(r'(?:qty|quantity|count|כמות)', label.strip(), re.I)]
        is_total_row = any(_ITEM.search(label) and re.match(
            r'^\s*ס[ה][\x22\x27\u05f3\u05f4]?כ(?:\s|$)', value)
            for label, _, value, _, _ in fields)
        if has_quantity_column and not is_total_row:
            from decimal import Decimal, InvalidOperation
            if len(quantities) != 1:
                return (), False, False
            try:
                quantity = Decimal(quantities[0])
                if not quantity.is_finite() or quantity < 0:
                    return (), False, False
            except InvalidOperation:
                return (), False, False
            # Explicit zero is not an inclusion. Never silently discard a
            # configuration-wide note that happens to be stored in this row.
            if not quantity:
                if (configuration_notes(fields) or any(
                        value.strip() for label, _, value, _, _ in fields
                        if re.fullmatch(r'(?:[A-Z]{1,3}|notes?|remarks?|הערות?)', label.strip(), re.I))):
                    return (), False, False
                continue
        elif len(quantities) > 1:
            return (), False, False
        if len(passages) >= MAX_PROJECT_ROWS or sum(len(p.text) for p in passages)+len(claim) > MAX_PROJECT_CHARS:
            return (), False, False
        columns = [field[1] for field in fields]
        cell_range = f'{columns[0]}{row}:{columns[-1]}{row}' if len(columns)>1 else f'{columns[0]}{row}'
        m = dict(metadata)
        m.update(sheet=sheet, row=row, section=f'Sheet: {sheet} / {cell_range}',
                 _workbook_range=cell_range, _workbook_subject=subject,
                 _workbook_scope=request.ordinal,
                 _workbook_heading=(*owner_context, request.section),
                 _workbook_owner_candidates=(0,),
                 _workbook_general_qualification_ids=(),
                 _workbook_general_qualification_fields=())
        chunk_id = f"{metadata['source_id']}::workbook::{sheet}::{cell_range}"
        identity = str(len(passages)+1)
        if total and total[0] == row:
            m['_ordinal_section_total'] = total[1:]
        notes = configuration_notes(fields)
        if notes:
            qualification_fields[identity] = notes
        passages.append(replace(make_passage(claim, claim, m, chunk_id, identity),
                                representation='workbook_original'))
    ids = tuple(qualification_fields)
    passages = [replace(p, metadata={**p.metadata,
        '_workbook_general_qualification_ids': ids,
        '_workbook_general_qualification_fields': qualification_fields.get(p.citation_id, ())})
        for p in passages]
    return tuple(passages), bool(passages), False


def _build_ordinal_section(grouped, subject, document_root, tenant, request):
    # Match the requested literal document identity against the filename,
    # allowing only the grammatical definite article on Hebrew title words.
    document = re.findall(r'[A-Za-z0-9]+|[א-ת]+', request.document.casefold())
    def filename_matches(value):
        words = re.findall(r'[A-Za-z0-9]+|[א-ת]+', str(value).casefold())
        return any(all(a == b or (len(a) > 3 and a.startswith('ה') and a[1:] == b)
                       for a, b in zip(document, words[start:start+len(document)]))
                   for start in range(len(words)-len(document)+1))
    matches = []
    for source in sorted(grouped):
        rows = grouped[source]
        if not rows:
            continue
        metadata = rows[0]['metadata']
        if (str(metadata.get('mime_type') or '') !=
                'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                or not filename_matches(PurePosixPath(metadata.get('source_relative_path', '')).name)):
            continue
        try:
            text, layout = _original_workbook(document_root, tenant, metadata, include_layout=True)
            selected, complete, ambiguous = _ordinal_section_source(metadata, text, layout, request, subject)
        except (SourceAccessError, ValueError, TypeError, KeyError):
            return (), False, False
        if ambiguous:
            return (), False, True
        if selected:
            vat_mentioned, tax_mentioned = bool(_VAT.search(text)), bool(_TAX.search(text))
            currency_markers = tuple(m.group(0).casefold() for m in _CURRENCY.finditer(text))
            selected = tuple(replace(p, metadata={**p.metadata,
                '_workbook_vat_mentioned_in_extracted_text': vat_mentioned,
                '_workbook_tax_mentioned_in_extracted_text': tax_mentioned,
                '_workbook_currency_markers': currency_markers}) for p in selected)
            matches.append((selected, complete))
    if len(matches) > 1:
        return (), False, True
    if not matches:
        return (), False, False
    selected, complete = matches[0]
    projected, projection_complete = _prompt_projection(selected)
    return projected, bool(complete and projection_complete), False


def _build(grouped, subject, document_root, tenant):
    from subject_intent import ordinal_workbook_section_subject
    ordinal = ordinal_workbook_section_subject(subject)
    if ordinal:
        return _build_ordinal_section(grouped, subject, document_root, tenant, ordinal)
    passages, complete, owner_ambiguous = [], True, False
    explicit_owners = set()
    for source in sorted(grouped):
        rows = grouped[source]
        if not rows:
            continue
        metadata = rows[0]['metadata']
        if str(metadata.get('mime_type') or '') != 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            continue
        # Filename/sheet/body-index text are candidate hints only. The original
        # is reopened and identity is checked separately before inclusion.
        if not (_match(subject, metadata.get('source_relative_path', '')) or
                any(_match(subject, row.get('document', '')) for row in rows)):
            continue
        try:
            text = _original_workbook(document_root, tenant, metadata)
            selected, source_complete, source_ambiguous, source_owners = _source_passages(metadata, subject, text, len(passages)+1)
            vat_mentioned = bool(_VAT.search(text))
            currency_markers = tuple(match.group(0).casefold() for match in _CURRENCY.finditer(text))
            tax_mentioned = bool(_TAX.search(text))
            selected = tuple(replace(p, metadata={**p.metadata,
                '_workbook_vat_mentioned_in_extracted_text': vat_mentioned,
                '_workbook_tax_mentioned_in_extracted_text': tax_mentioned,
                '_workbook_currency_markers': currency_markers}) for p in selected)
        except (SourceAccessError, ValueError, TypeError, KeyError):
            complete = False
            continue
        owner_ambiguous = owner_ambiguous or source_ambiguous
        explicit_owners.update(source_owners)
        passages.extend(selected)
        complete = complete and source_complete
    # Alternative interpretations of a prefixed Hebrew name must not cause
    # distinct project owners to be combined as though they were one.
    common = set(range(len(_candidates(subject))))
    for passage in passages:
        common.intersection_update(passage.metadata.get('_workbook_owner_candidates', ()))
    ambiguous = owner_ambiguous or len(explicit_owners) > 1 or bool(passages and not common)
    if ambiguous:
        return (), complete, True
    projected, projection_complete = _prompt_projection(passages)
    return projected, bool(complete and projection_complete), False


def _table_currency_context(source_claim):
    """Return the currency token of an unqualified table-wide price label."""
    match = re.fullmatch(r'Spreadsheet context row [1-9][0-9]*:\n- Cell [A-Z]+[1-9][0-9]*: ([^\n]+)', source_claim)
    if not match:
        return ''
    label = match.group(1).strip()
    currencies = list(_CURRENCY.finditer(label))
    if len(currencies) != 1:
        return ''
    currency = currencies[0]
    if (not _TABLE_CURRENCY_PREFIX.fullmatch(label[:currency.start()].strip()) or
            label[currency.end():].strip(' .:')):
        return ''
    return currency.group(0).casefold()


def _workbook_overview_total(evidence, hebrew, *, allow_tax=False):
    """Return a deterministic total statement and its source IDs, or None.

    This intentionally accepts only a whole, single, owned section with one
    currency context and one SUM covering every item row. No inference is made
    from a large/last number or from an ambiguous currency hint.
    """
    from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
    if not isinstance(evidence, WorkbookProjectEvidence) or not evidence.complete:
        return None
    # A scoped table can carry its own exact total role and tax exclusion in
    # the total row. The collector re-reads and verifies its complete SUM and
    # cached row amounts from the authenticated source on every validation.
    from subject_intent import ordinal_workbook_section_subject
    if ordinal_workbook_section_subject(evidence.subject):
        totals = [(p, p.metadata.get('_ordinal_section_total')) for p in evidence.passages
                  if p.metadata.get('_ordinal_section_total')]
        if len(totals) != 1:
            return None
        passage, (amount, label) = totals[0]
        if hebrew:
            text = f'הסכום תחת הכותרת ״{label}״ הוא {Decimal(amount):,.2f} ₪.'
        else:
            text = f'The amount under the heading “{label}” is {Decimal(amount):,.2f} ₪.'
        return text, (passage.citation_id,)
    # A VAT mention may qualify the amount. Until its exact role is bound to
    # this total, do not emit an amount that silently drops that qualification.
    # Unknown text coverage is likewise insufficient to assert VAT absence.
    vat = evidence.vat_mentioned_in_extracted_text
    tax = evidence.tax_mentioned_in_extracted_text
    if vat is None or tax is None or (not allow_tax and (vat or tax)):
        return None
    ps = evidence.passages
    if not ps or len({(p.source_id, p.metadata.get('sheet'), p.metadata.get('_workbook_scope')) for p in ps}) != 1:
        return None
    contexts = [p for p in ps if p.source_claim.startswith('Spreadsheet context row ') and _CURRENCY.search(p.source_claim)]
    totals = []
    for passage in ps:
        fields = _fields(passage.source_claim)
        if len(fields) != 1:
            continue
        label, column, value, _, _ = fields[0]
        match = re.fullmatch(r'Formula: =SUM\(([A-Z]{1,3})([1-9][0-9]*):([A-Z]{1,3})([1-9][0-9]*)\); Cached value: ([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))', value, re.I)
        if match and _TOTAL_PRICE.fullmatch(label) and match.group(1).upper() == column == match.group(3).upper():
            totals.append((passage, match))
    if len(contexts) != 1 or len(totals) != 1:
        return None
    context, (total, match) = contexts[0], totals[0]
    markers = total.metadata.get('_workbook_currency_markers')
    if not isinstance(markers, tuple) or len(markers) != 1:
        return None
    marker = markers[0]
    currencies = {
        '₪':'₪', 'ils':'₪', 'nis':'₪', 'shekel':'₪', 'shekels':'₪', 'שקל':'₪', 'שקלים':'₪',
        'usd':'$', 'dollar':'$', 'dollars':'$', 'דולר':'$', 'דולרים':'$',
        'eur':'€', 'euro':'€', 'euros':'€', 'אירו':'€', 'יורו':'€',
        'gbp':'£', 'pound':'£', 'pounds':'£', 'ליש״ט':'£', 'ליש"ט':'£', 'לישט':'£',
    }
    currency = currencies.get(marker)
    if not currency or _table_currency_context(context.source_claim) != marker:
        return None
    first, last = int(match.group(2)), int(match.group(4))
    items = [p for p in ps if p is not total and
             not p.source_claim.startswith('Spreadsheet context row ')]
    if (last < first or last - first >= MAX_PROJECT_ROWS or
            {p.metadata.get('row') for p in items} != set(range(first, last + 1)) or
            any(not any(f[1] == match.group(1).upper() for f in _fields(p.source_claim)) for p in items)):
        return None
    try:
        amount = Decimal(match.group(5)).quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    except InvalidOperation:
        return None
    # Check the stated cache against the recorded line amounts. Never replace
    # the workbook total with our calculation; disclose a material difference.
    calculated = Decimal(0)
    try:
        for item in items:
            value = next(f[2] for f in _fields(item.source_claim) if f[1] == match.group(1).upper())
            cached = re.fullmatch(r'Formula: .*?; Cached value: ([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))', value)
            calculated += Decimal(cached.group(1) if cached else value)
        calculated = calculated.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    except (InvalidOperation, StopIteration):
        return None
    formatted = f'{amount:,.2f}'
    mismatch = amount != calculated
    # Preserve the workbook's own cost/price ownership label, not merely its currency.
    label = context.source_claim.rsplit(': ', 1)[-1].strip()
    if hebrew:
        sentence = f'הסכום הכולל תחת הכותרת ״{label}״ בגיליון הוא כ־{formatted} {currency}.'
        if mismatch:
            sentence += f' סכימת סכומי השורות השמורים נותנת כ־{calculated:,.2f} {currency}.'
        if not vat and not tax:
            sentence += ' בנתונים הזמינים בגיליון לא צוין אופן הטיפול במע״מ.'
    else:
        sentence = f'The worksheet total under the heading “{label}” is approximately {currency}{formatted}.'
        if mismatch:
            sentence += f' The sum of the stored line amounts is approximately {currency}{calculated:,.2f}.'
        if not vat and not tax:
            sentence += ' The available worksheet data does not state the VAT treatment.'
    return sentence, ((total.citation_id, context.citation_id) +
                      tuple(p.citation_id for p in items) if mismatch else
                      (total.citation_id, context.citation_id))


def workbook_overview_total(evidence, hebrew):
    return _workbook_overview_total(evidence, hebrew)


def workbook_total_tax_limitation(evidence, hebrew):
    """Return a source-bound, nonnumeric notice for a tax-qualified total.

    Only when the total otherwise passes the same source checks and an actual
    tax mention is among the authenticated passages can we explain this
    limitation. A whole-workbook scan flag by itself is not a citable note.
    """
    if not isinstance(evidence, WorkbookProjectEvidence):
        return None
    if (evidence.vat_mentioned_in_extracted_text is not True and
            evidence.tax_mentioned_in_extracted_text is not True):
        return None
    total = _workbook_overview_total(evidence, hebrew, allow_tax=True)
    if not total:
        return None
    by_id = {p.citation_id: p for p in evidence.passages}
    total_passage = by_id.get(total[1][0])
    notes = [p for p in evidence.passages if p is not total_passage
             and (_VAT.search(p.source_claim) or _TAX.search(p.source_claim))]
    if not notes:
        return None
    # _workbook_overview_total has already required every passage to belong
    # to the same authenticated source, sheet, and owned section.
    note = notes[0]
    sentence = ('הגיליון מציג סכום כולל ומזכיר מס; הקשר בין אזכור המס לסכום לא אומת.'
                if hebrew else
                'The worksheet shows a total and mentions tax; the connection between the tax mention and the total is not established.')
    return sentence, (total_passage.citation_id, note.citation_id)


def collect_workbook_project(retriever, tenant, document_root, subject, where=None):
    grouped, scanned, count = scan_authorized(retriever, tenant, document_root, where=where)
    passages, selected, ambiguous = _build(grouped, subject, document_root, tenant)
    witness = ScanWitness(tuple(row for rows in grouped.values() for row in rows), scanned)
    return WorkbookProjectEvidence(passages, bool(scanned and selected), len(grouped), count,
        '' if passages else 'workbook_project_ambiguous' if ambiguous else 'workbook_project_not_found', witness, str(subject), str(document_root),
        str(tenant), str((where or {}).get('source') or ''), bool(selected))


def validate_workbook_project(evidence, subject, document_root, tenant):
    if (not isinstance(evidence, WorkbookProjectEvidence) or not evidence.witness
            or evidence.subject != subject or evidence.document_root != str(document_root)
            or evidence.tenant_id != tenant):
        return False
    grouped = {}
    for row in evidence.witness.rows:
        meta = row.get('metadata') or {}
        if (meta.get('tenant_id') != tenant or (evidence.where_source and
                meta.get('source_relative_path') != evidence.where_source)):
            return False
        grouped.setdefault(meta.get('source_id'), []).append(row)
    try:
        rebuilt, selection_complete, ambiguous = _build(
            grouped, evidence.narrow_scope or subject, document_root, tenant)
        return (not ambiguous and rebuilt == evidence.passages and evidence.selection_complete == selection_complete
                and evidence.complete == bool(evidence.witness.storage_complete and selection_complete))
    except (ValueError, TypeError, KeyError, SourceAccessError):
        return False






















