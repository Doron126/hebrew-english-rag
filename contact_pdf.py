"""Conservative native PDF table association for explicitly named contacts.

This is an evidence reader, not a corpus search or an answer renderer.  The
caller must apply it to all relevant tenant-authorized candidate sources and
handle failure as unresolved evidence.
"""
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass

from source_access import (SourceAccessError, open_source_file,
                           source_metadata_complete, source_stat_matches_metadata)

_NAMES = {"שם ושם משפחה", "שם מלא", "שם איש הקשר", "שם איש קשר"}
_ORGANIZATIONS = {"שם גוף מזמין העבודה", "ארגון", "שם הארגון", "שם החברה", "מעסיק"}
_ROLES = {"תפקיד"}
_MOBILES = {"טלפון נייד", "מספר טלפון נייד", "מספר נייד", "נייד"}
@dataclass(frozen=True)
class PdfContactRow:
    person: str
    value: str
    page: int
    table_index: int
    header_row: int
    data_row: int
    person_column: int
    mobile_column: int
    cells: tuple[str | None, ...]
    organization: str = ""
    role: str = ""
    layout_group: bool = False
    group_names: tuple[str, ...] = ()
    group_values: tuple[str, ...] = ()
    group_roles: tuple[str, ...] = ()


_PHONE = re.compile(r"\+?[0-9][0-9 -]{5,20}[0-9]")


def _norm(value):
    return " ".join(str(value or "").split())


def _cell(page, rectangle):
    """Extract only complete native words from one unambiguous cell."""
    if rectangle is None:
        raise ValueError("merged table cell")
    x0, y0, x1, y1 = map(float, rectangle)
    if not (0 <= x0 < x1 <= page.rect.width and 0 <= y0 < y1 <= page.rect.height):
        raise ValueError("invalid table cell")
    # fitz clipping may include a word merely intersecting a boundary. Refuse
    # boundary-crossing words rather than attribute them to either cell.
    for word in page.get_text("words", sort=False):
        a, b, c, d = map(float, word[:4])
        if min(c, x1) - max(a, x0) > 0.05 and min(d, y1) - max(b, y0) > 0.05:
            if a < x0 - .5 or b < y0 - .5 or c > x1 + .5 or d > y1 + .5:
                raise ValueError("text crosses table cell boundary")
    return _norm(page.get_text("text", clip=rectangle, sort=False))


def _disjoint(a, b):
    return (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _lines(page, rect):
    """Complete native text lines and their original page coordinates."""
    _cell(page, rect)
    lines = []
    for block in page.get_text("dict", clip=rect, sort=False).get("blocks", ()):
        for line in block.get("lines", ()):
            text = _norm("".join(span.get("text", "") for span in line.get("spans", ())))
            if not text:
                continue
            box = tuple(float(x) for x in line.get("bbox", ()))
            if len(box) != 4 or box[1] >= box[3] or box[1] < rect[1] - .5 or box[3] > rect[3] + .5:
                raise ValueError("invalid native text line")
            lines.append((box, text))
    return tuple(sorted(lines, key=lambda item: (item[0][1], item[0][0])))


def _layout_groups(page, names_rect, numbers_rect, role_rect):
    """Prove a one-to-one name/number mapping by actual shared baselines."""
    names, numbers = _lines(page, names_rect), _lines(page, numbers_rect)
    if not 2 <= len(names) <= 20 or len(names) != len(numbers):
        raise ValueError("incomplete contact layout")
    if any(re.search(r"[/|;]", text) for _, text in names):
        raise ValueError("multiple identities on one line")
    if any(not _PHONE.fullmatch(text) or not 7 <= sum(ch.isdigit() for ch in text) <= 15
           for _, text in numbers):
        raise ValueError("multiple or invalid numbers on a line")
    if len({text for _, text in names}) != len(names):
        raise ValueError("duplicate contact identity")
    if any(names[i][0][3] >= names[i + 1][0][1] for i in range(len(names) - 1)):
        raise ValueError("overlapping contact names")
    aligned = []
    used = set()
    for box, _ in names:
        hits = [i for i, (other, _) in enumerate(numbers)
                if abs(box[1] - other[1]) <= 1 and abs(box[3] - other[3]) <= 1]
        if len(hits) != 1 or hits[0] in used:
            raise ValueError("contact lines do not have unique aligned values")
        used.add(hits[0])
        aligned.append(numbers[hits[0]][1])
    # Roles are optional but, if present, each line must fall into exactly
    # one non-overlapping vertical group anchored by the name baselines.
    roles = ["" for _ in names]
    if role_rect is not None:
        for box, text in _lines(page, role_rect):
            groups = [i for i, (anchor, _) in enumerate(names)
                      if box[1] >= anchor[1] - 1 and
                      (i + 1 == len(names) or box[3] < names[i + 1][0][1] - .5)]
            if len(groups) != 1 or re.search(r"[/|;]", text):
                raise ValueError("ambiguous role layout")
            i = groups[0]
            roles[i] = _norm(roles[i] + " " + text)
    return tuple(text for _, text in names), tuple(aligned), tuple(roles)


def _organization_with_native_punctuation(page, rect, value):
    """Repair an RTL closing bracket only when glyph positions prove it."""
    match = re.search(r'\)([^\s()]+)', value)
    if not match or value.count('(') != 1 or value.count(')') != 1:
        return value
    words = page.get_text('words', clip=rect, sort=False)
    closing = [w for w in words if w[4] == ')']
    if len(closing) != 1:
        return value
    final = [w for w in words if w[4] == match.group(1)
             and abs(closing[0][1] - w[1]) <= 1 and abs(closing[0][3] - w[3]) <= 1
             and -.5 <= w[0] - closing[0][2] <= 1]
    if len(final) != 1:
        return value
    return value[:match.start()] + match.group(1) + ')' + value[match.end():]


def pdf_contact_rows(document_root, tenant_id, metadata, person):
    """Return source-supported exact mobile rows, or raise on uncertainty.

    Authenticates the original PDF using the indexed generation identity. Every
    mention of the full name must be accounted for by a uniquely headed,
    unmerged same-row name/mobile cell pair. Values remain strings verbatim.
    """
    person = _norm(person)
    if not person or not source_metadata_complete(metadata, tenant_id):
        raise ValueError("missing source or person identity")
    if str(metadata.get("mime_type") or "").casefold() != "application/pdf":
        raise ValueError("not a PDF source")
    try:
        relative, handle, before = open_source_file(document_root, tenant_id, metadata.get("source_relative_path", ""))
        with handle:
            if relative.suffix.casefold() != ".pdf" or not source_stat_matches_metadata(metadata, before):
                raise ValueError("PDF generation changed")
            if before.st_size > 32 * 1024 * 1024:
                raise ValueError("PDF exceeds size bound")
            payload = handle.read(32 * 1024 * 1024 + 1)
            after = os.fstat(handle.fileno())
            if (len(payload) != before.st_size or hashlib.sha256(payload).hexdigest() != metadata.get("source_sha256")
                    or (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns)):
                raise ValueError("PDF source integrity mismatch")
        import fitz
        values = []
        occurrences = 0
        resolved = 0
        with fitz.open(stream=payload, filetype="pdf") as doc:
            if doc.needs_pass or len(doc) > 200:
                raise ValueError("unsupported PDF")
            for page_number, page in enumerate(doc, start=1):
                raw = page.get_text("text", sort=False)
                if len(raw) > 160000:
                    raise ValueError("PDF page exceeds text bound")
                mentions = _norm(raw).count(person)
                if not mentions:
                    continue
                occurrences += mentions
                words = page.get_text("words", sort=False)
                if len(words) > 4000 or len(page.get_drawings()) > 12000:
                    raise ValueError("PDF geometry exceeds bound")
                tables = page.find_tables().tables
                if len(tables) > 40:
                    raise ValueError("too many PDF tables")
                for table_index, table in enumerate(tables):
                    if table.row_count > 500 or table.col_count > 40:
                        raise ValueError("PDF table exceeds bound")
                    for hi, header in enumerate(table.rows):
                        # Unrelated cells may be merged; only inspect simple
                        # candidate cells, and later require an exact pair.
                        labels = []
                        for rect in header.cells:
                            try:
                                labels.append(_cell(page, rect))
                            except ValueError:
                                labels.append(None)
                        ni = [i for i, x in enumerate(labels) if x in _NAMES]
                        mi = [i for i, x in enumerate(labels) if x in _MOBILES]
                        if len(ni) != 1 or len(mi) != 1 or ni[0] == mi[0]:
                            continue
                        n, m = ni[0], mi[0]
                        # Organization may occupy a vertically merged cell in
                        # the immediately preceding parent header. Only a
                        # unique recognized role in these two header rows is
                        # accepted, with its data column checked below.
                        context_headers = []
                        for source_hi in range(max(0, hi - 1), hi + 1):
                            for ci, rect in enumerate(table.rows[source_hi].cells):
                                try:
                                    label = _cell(page, rect)
                                except ValueError:
                                    continue
                                if label in _ORGANIZATIONS or label in _ROLES:
                                    context_headers.append((label, ci, rect))
                        org_headers = [(i, rect) for label, i, rect in context_headers if label in _ORGANIZATIONS]
                        role_headers = [(i, rect) for label, i, rect in context_headers if label in _ROLES]
                        hn, hm = header.cells[n], header.cells[m]
                        if not _disjoint(hn, hm):
                            raise ValueError("overlapping header cells")
                        for row_index, row in enumerate(table.rows[hi + 1:], start=hi + 1):
                            # Never carry a previous header across another
                            # recognized header with different column roles.
                            row_labels = []
                            for rect in row.cells:
                                try:
                                    row_labels.append(_cell(page, rect))
                                except ValueError:
                                    row_labels.append(None)
                            if any(label is None or label in _NAMES or label in _MOBILES for label in row_labels):
                                break
                            if max(n, m) >= len(row.cells):
                                continue
                            rn, rm = row.cells[n], row.cells[m]
                            if rn is None or rm is None:
                                continue
                            try:
                                name = _cell(page, rn)
                            except ValueError:
                                continue
                            if person not in name:
                                continue
                            if not _disjoint(rn, rm):
                                raise ValueError("overlapping data cells")
                            # Require each data cell to use the same horizontal
                            # boundaries as its own header cell; no merged or
                            # misaligned table row can transfer a field.
                            if any(abs(float(a) - float(b)) > 1 for a, b in
                                   ((rn[0], hn[0]), (rn[2], hn[2]), (rm[0], hm[0]), (rm[2], hm[2]))):
                                raise ValueError("misaligned table columns")
                            multi = name != person
                            group_names = group_values = group_roles = ()
                            if multi:
                                role_rect = None
                                if len(role_headers) == 1:
                                    role_index, role_header_rect = role_headers[0]
                                    if role_index < len(row.cells) and row.cells[role_index] is not None:
                                        candidate = row.cells[role_index]
                                        if abs(candidate[0] - role_header_rect[0]) <= 1 and abs(candidate[2] - role_header_rect[2]) <= 1:
                                            role_rect = candidate
                                group_names, group_values, group_roles = _layout_groups(page, rn, rm, role_rect)
                                if group_names.count(person) != 1:
                                    raise ValueError("contact identity not unique")
                                value = group_values[group_names.index(person)]
                            else:
                                value = _cell(page, rm)
                                if not _PHONE.fullmatch(value) or not 7 <= sum(ch.isdigit() for ch in value) <= 15:
                                    raise ValueError("missing or ambiguous mobile value")
                            def context_value(headers):
                                if len(headers) != 1:
                                    return ""
                                index, hrect = headers[0]
                                if index >= len(row.cells) or row.cells[index] is None:
                                    return ""
                                rect = row.cells[index]
                                if (abs(rect[0] - hrect[0]) > 1 or abs(rect[2] - hrect[2]) > 1
                                        or not _disjoint(rect, rn) or not _disjoint(rect, rm)):
                                    return ""
                                raw_context = page.get_text("text", clip=rect, sort=False).strip()
                                # Multiple identity qualifiers in one cell do
                                # not establish which person they describe.
                                if re.search(r"[/|;]|\n\s*\n", raw_context):
                                    return ""
                                return row_labels[index] or ""
                            organization, role = context_value(org_headers), context_value(role_headers)
                            if organization and len(org_headers) == 1:
                                organization = _organization_with_native_punctuation(
                                    page, row.cells[org_headers[0][0]], organization)
                            if multi:
                                role = group_roles[group_names.index(person)]
                            values.append(PdfContactRow(person, value, page_number, table_index, hi, row_index, n, m, tuple(row_labels), organization, role, multi, group_names, group_values, group_roles))
                            resolved += 1
        if not occurrences or occurrences != resolved:
            raise ValueError("person mention not uniquely resolved")
        return tuple(values)
    except (SourceAccessError, OSError, RuntimeError, ImportError, TypeError, AttributeError) as exc:
        raise ValueError("PDF source unavailable or unsupported") from exc
