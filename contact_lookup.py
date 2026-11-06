"""Conservative recognition and native table reading for mobile lookups.

This module does not authorize sources or release answers.  Callers must use
the tenant-scoped index and verify each source generation before reading it.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import zipfile
from xml.etree import ElementTree as ET


@dataclass(frozen=True)
class MobileRequest:
    person: str


_HE_NAME = r"[\u05d0-\u05ea][\u05d0-\u05ea\u05f3\u05f4'\-]*(?:\s+[\u05d0-\u05ea][\u05d0-\u05ea\u05f3\u05f4'\-]*){1,3}"
_FIELD = r"(?:(?:מספר\s+)?(?:ה?טלפון\s+)?(?:הנייד|נייד)|(?:מספר\s+)?ה?טלפון\s+הנייד)"
_QUESTION = re.compile(
    rf"^(?:(?:תן|תני|תנו)\s+לי\s+(?:את\s+)?|(?:מה|מהו|מהי)\s+)"
    rf"{_FIELD}\s+של\s+(?P<person>{_HE_NAME})[?.!\s]*$"
)


def mobile_request(question: str) -> MobileRequest | None:
    """Accept only an explicit mobile field and a complete named person."""
    match = _QUESTION.fullmatch(str(question or '').strip())
    return MobileRequest(' '.join(match.group('person').split())) if match else None


@dataclass(frozen=True)
class MobileCell:
    person: str
    value: str
    table_index: int
    header_row: int
    data_row: int
    person_column: int
    mobile_column: int
    row_context: tuple[str, ...]
    organization: str = ''
    role: str = ''
    layout_dependent: bool = False
    group_names: tuple[str, ...] = ()
    group_values: tuple[str, ...] = ()
    group_roles: tuple[str, ...] = ()


def authenticated_docx_mobile_cells(document_root, tenant, metadata, person):
    """Reopen and authenticate the current source generation before parsing."""
    import hashlib
    import io
    import os
    from source_access import (open_source_file, source_metadata_complete,
                               source_stat_matches_metadata)
    if not source_metadata_complete(metadata, tenant):
        raise ValueError('invalid DOCX source metadata')
    relative, handle, before = open_source_file(document_root, tenant, metadata['source_relative_path'])
    with handle:
        if (relative.suffix.casefold() != '.docx' or before.st_size > 32 * 1024 * 1024
                or not source_stat_matches_metadata(metadata, before)):
            raise ValueError('DOCX generation changed')
        payload = handle.read(32 * 1024 * 1024 + 1)
        after = os.fstat(handle.fileno())
        if (len(payload) != before.st_size or hashlib.sha256(payload).hexdigest() != metadata['source_sha256']
                or (before.st_ino, before.st_size, before.st_mtime_ns) !=
                   (after.st_ino, after.st_size, after.st_mtime_ns)):
            raise ValueError('DOCX generation changed during read')
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        infos = archive.infolist()
        if (len(infos) > 4096 or len({i.filename for i in infos}) != len(infos)
                or sum(i.file_size for i in infos) > 64 * 1024 * 1024
                or any(re.fullmatch(r'word/(?:header\d+|footer\d+|footnotes|endnotes)\.xml', i.filename)
                       for i in infos)):
            raise ValueError('DOCX archive cannot be fully examined')
    return docx_mobile_cells(io.BytesIO(payload), person)


_W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
_PERSON_HEADERS = {'שם ושם משפחה', 'שם מלא', 'שם איש הקשר', 'שם איש קשר'}
_MOBILE_HEADERS = {'טלפון נייד', 'מספר טלפון נייד', 'מספר נייד', 'נייד'}
_ORG_HEADERS = {'שם גוף מזמין העבודה', 'שם הארגון', 'ארגון', 'חברה'}
_ROLE_HEADERS = {'תפקיד', 'תפקיד איש הקשר'}
_PHONE = re.compile(r'\+?[0-9][0-9 -]{5,20}[0-9]')


def _cell_text(cell: ET.Element) -> str:
    # A paragraph boundary is significant: two people in one cell cannot
    # be treated as one identity or paired with two phone values.
    from ingest import _docx_paragraph_text
    paragraphs = cell.findall(_W + 'p')
    return '\n'.join(_docx_paragraph_text(p) for p in paragraphs).strip()


def _paragraph_slots(cell):
    """Propose explicit paragraph slots; never prove a visual association."""
    from ingest import _docx_paragraph_text
    paragraphs = cell.findall(_W + 'p')
    if not paragraphs or len(paragraphs) > 100:
        return None
    forbidden = {'br', 'cr', 'tab', 'drawing', 'pict', 'object', 'fldChar', 'instrText'}
    for paragraph in paragraphs:
        if any(node.tag.rsplit('}', 1)[-1] in forbidden for node in paragraph.iter()):
            return None
    return tuple(_docx_paragraph_text(p).strip() for p in paragraphs)


def _simple_cell(cell: ET.Element) -> bool:
    props = cell.find(_W + 'tcPr')
    return (cell.find(_W + 'tbl') is None and
            (props is None or all(props.find(_W + field) is None
                                  for field in ('gridSpan', 'vMerge', 'hMerge'))))


def _plain_row_geometry(row: ET.Element, columns: tuple[int, ...]) -> bool:
    props = row.find(_W + 'trPr')
    if props is not None and any(props.find(_W + name) is not None
                                 for name in ('gridBefore', 'gridAfter')):
        return False
    cells = row.findall(_W + 'tc')
    # A span before either relevant cell changes its grid position even if
    # those two cells themselves are unmerged. Vertical merges before them do
    # not change horizontal positions.
    for cell in cells[:max(columns) + 1]:
        cell_props = cell.find(_W + 'tcPr')
        if cell_props is not None and any(cell_props.find(_W + name) is not None
                                          for name in ('gridSpan', 'hMerge')):
            return False
    return all(index < len(cells) and _simple_cell(cells[index]) for index in columns)


def docx_mobile_cells(handle, person: str) -> tuple[tuple[MobileCell, ...], bool]:
    """Return exact single-valued matches and whether all mentions were resolved.

    Header and data cells must have unambiguous native grid positions.  A
    mention outside such a row is deliberately unresolved, not a negative.
    This is a structural parser only: source authentication is the caller's
    responsibility.
    """
    from ingest import _read_zip_part, _parse_xml_part, _docx_paragraph_text, IngestExtractionError
    person = ' '.join(str(person or '').split())
    if not person:
        return (), False
    try:
        with zipfile.ZipFile(handle) as archive:
            payload = _read_zip_part(archive, 'word/document.xml')
        if not payload:
            return (), False
        root = _parse_xml_part(payload, 'word/document.xml')
    except (OSError, ValueError, zipfile.BadZipFile, ET.ParseError, IngestExtractionError):
        return (), False
    matches = []
    resolved_nodes: set[int] = set()
    for ti, table in enumerate(root.iter(_W + 'tbl')):
        rows = table.findall(_W + 'tr')
        for hi, header in enumerate(rows):
            cells = header.findall(_W + 'tc')
            labels = [' '.join(_cell_text(c).split()) for c in cells]
            names = [i for i, label in enumerate(labels) if label in _PERSON_HEADERS]
            mobiles = [i for i, label in enumerate(labels) if label in _MOBILE_HEADERS]
            if len(names) != 1 or len(mobiles) != 1 or names[0] == mobiles[0]:
                continue
            name_i, mobile_i = names[0], mobiles[0]
            if not _plain_row_geometry(header, (name_i, mobile_i)):
                continue
            roles = [i for i, label in enumerate(labels) if label in _ROLE_HEADERS]
            orgs = [i for i, label in enumerate(labels) if label in _ORG_HEADERS]
            if orgs and (len(orgs) != 1 or not _plain_row_geometry(header, (orgs[0],))):
                orgs = []
            # A two-level header may vertically merge an organization label
            # from the immediately preceding row into this header row.
            if not orgs and hi:
                previous = rows[hi - 1].findall(_W + 'tc')
                previous_labels = [' '.join(_cell_text(c).split()) for c in previous]
                indices = [i for i, label in enumerate(previous_labels) if label in _ORG_HEADERS]
                if len(indices) == 1 and indices[0] < len(cells):
                    oi = indices[0]
                    prior_props = rows[hi - 1].find(_W + 'trPr')
                    prior_aligned = (prior_props is None or all(
                        prior_props.find(_W + name) is None for name in ('gridBefore', 'gridAfter')))
                    before_props = previous[oi].find(_W + 'tcPr')
                    now_props = cells[oi].find(_W + 'tcPr')
                    before_merge = before_props.find(_W + 'vMerge') if before_props is not None else None
                    now_merge = now_props.find(_W + 'vMerge') if now_props is not None else None
                    earlier = previous[:oi + 1]
                    horizontal_safe = all(c.find(_W + 'tcPr') is None or
                        all(c.find(_W + 'tcPr').find(_W + x) is None for x in ('gridSpan', 'hMerge'))
                        for c in earlier)
                    if (prior_aligned and horizontal_safe and before_merge is not None and now_merge is not None
                            and before_merge.get(_W + 'val') == 'restart'
                            and now_merge.get(_W + 'val') in (None, 'continue')):
                        orgs = [oi]
            for ri in range(hi + 1, len(rows)):
                row = rows[ri]
                data = row.findall(_W + 'tc')
                values = [' '.join(_cell_text(cell).split()) for cell in data]
                # A subsequent header begins a different schema, including a
                # partial or changed header we cannot interpret safely.
                if any(value in _PERSON_HEADERS | _MOBILE_HEADERS for value in values):
                    break
                if not _plain_row_geometry(row, (name_i, mobile_i)):
                    continue
                raw_name, raw_value = _cell_text(data[name_i]), _cell_text(data[mobile_i])
                nameslots = _paragraph_slots(data[name_i])
                valueslots = _paragraph_slots(data[mobile_i])
                if nameslots and valueslots and len(nameslots) == len(valueslots):
                    occupied = tuple(i for i, name in enumerate(nameslots) if name)
                    phone_occupied = tuple(i for i, value in enumerate(valueslots) if value)
                    # Paragraph equality proposes a grouping only; the caller
                    # must independently prove it with authenticated layout.
                    if (len(occupied) >= 2 and occupied == phone_occupied
                            and all(b - a >= 3 for a, b in zip(occupied, occupied[1:]))
                            and nameslots.count(person) == 1
                            and all(_PHONE.fullmatch(valueslots[i]) and
                                    7 <= sum(ch.isdigit() for ch in valueslots[i]) <= 15
                                    for i in occupied)):
                        target = nameslots.index(person)
                        if all(not re.search(r'[/|;]', nameslots[i]) for i in occupied):
                            org = (_cell_text(data[orgs[0]]) if len(orgs) == 1 and
                                   _plain_row_geometry(row, (orgs[0],)) else '')
                            role_slots = (_paragraph_slots(data[roles[0]]) if len(roles) == 1 and
                                          _plain_row_geometry(row, (roles[0],)) else None)
                            group_roles = tuple(value for value in (role_slots or ()) if value)
                            context = tuple(' '.join(_cell_text(cell).split()) for index, cell in enumerate(data)
                                            if index not in {name_i, mobile_i} and _cell_text(cell).strip())
                            matches.append(MobileCell(person, valueslots[target], ti, hi, ri, name_i,
                                mobile_i, context, org, '', True,
                                tuple(nameslots[i] for i in occupied),
                                tuple(valueslots[i] for i in occupied), group_roles))
                            resolved_nodes.add(id(data[name_i].findall(_W + 'p')[target]))
                            continue
                if (raw_name != person or '\n' in raw_value or not _PHONE.fullmatch(raw_value)
                        or not 7 <= sum(ch.isdigit() for ch in raw_value) <= 15):
                    continue
                context = tuple(' '.join(_cell_text(cell).split())
                                for index, cell in enumerate(data)
                                if index not in {name_i, mobile_i} and _cell_text(cell).strip())
                role = (_cell_text(data[roles[0]]) if len(roles) == 1 and
                        _plain_row_geometry(header, (roles[0],)) and
                        _plain_row_geometry(row, (roles[0],)) else '')
                org = (_cell_text(data[orgs[0]]) if len(orgs) == 1 and
                       _plain_row_geometry(row, (orgs[0],)) else '')
                matches.append(MobileCell(person, raw_value, ti, hi, ri, name_i, mobile_i, context, org, role))
                resolved_nodes.update(id(n) for n in data[name_i].findall(_W + 'p'))
    # If the full target occurs anywhere except a proven exact name cell, the
    # document may contain another person/value or an unsupported structure.
    unresolved = any(person in _docx_paragraph_text(paragraph)
                     and id(paragraph) not in resolved_nodes
                     for paragraph in root.iter(_W + 'p'))
    return tuple(matches), not unresolved
