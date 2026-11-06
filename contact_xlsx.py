"""Conservative negative contact evidence from an authenticated XLSX.

This reader never establishes a contact fact.  It only permits an indexed
workbook mentioning a person to be excluded as a competing phone source when
all supported native cell contents have been inspected and contain no phone
clue. Unsupported workbook structures are uncertainty.
"""
from __future__ import annotations

import hashlib
import ast
import io
import os
import posixpath
import re
import zipfile
from decimal import Decimal, DecimalException, localcontext
from xml.etree import ElementTree as ET

from source_access import open_source_file, source_metadata_complete, source_stat_matches_metadata
from ingest import (_parse_xml_part, _read_zip_part, _xlsx_shared_strings,
                    _xlsx_style_formats, _xlsx_sheet_rows, _xlsx_column_index,
                    _xlsx_column_name, _xlsx_cell_coordinates,
                    IngestExtractionError)

_S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
_R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
_P = "{http://schemas.openxmlformats.org/package/2006/relationships}"
_CLUE = re.compile(r"נייד|טלפון|טלפ|פלאפון|סלול|סלולר|mobile|telephone|\bphone\b|(?:\+?\d[\d ()-]{5,}\d)", re.I)
_ALLOWED = re.compile(r"(?:\[Content_Types\]\.xml|_rels/\.rels|docProps/[^/]+\.xml|xl/(?:workbook\.xml|_rels/workbook\.xml\.rels|sharedStrings\.xml|styles\.xml|calcChain\.xml|theme/[^/]+\.xml|persons/[^/]+\.xml|worksheets/sheet[0-9]+\.xml))")


def _safe_formula(value):
    # Arithmetic over fixed cells is permitted solely to inspect its cached
    # value. String construction, external links and unknown functions are not.
    value = re.sub(r"(?i)\bSUM\b", "", value)
    value = re.sub(r"\$?[A-Za-z]{1,3}\$?[0-9]+", "", value)
    return not re.search(r"[^0-9\s+*/:().,\-]", value)


_REF = re.compile(r'(\$?)([A-Za-z]{1,3})(\$?)([0-9]+)')


def _formula_cache_verified(sheet):
    """Recompute bounded numeric arithmetic, including shared relative refs."""
    cells = {}
    anchors = {}
    for cell in sheet.findall('.//' + _S + 'sheetData/' + _S + 'row/' + _S + 'c'):
        ref = cell.get('r') or ''
        if not re.fullmatch(r'[A-Z]{1,3}[1-9][0-9]*', ref) or ref in cells:
            return False
        cells[ref] = cell
        formula = cell.find(_S + 'f')
        if formula is not None and formula.get('si') is not None and (formula.text or '').strip():
            if formula.get('si') in anchors:
                return False
            anchors[formula.get('si')] = (ref, formula.text.strip())
    cache, visiting = {}, set()

    def evaluate(ref):
        if ref in cache:
            return cache[ref]
        if ref in visiting:
            raise ValueError('formula cycle')
        cell = cells.get(ref)
        if cell is None:
            return Decimal(0)
        visiting.add(ref)
        if (cell.get('t') or 'n') not in {'n', 'b'}:
            raise ValueError('nonnumeric formula reference')
        node = cell.find(_S + 'v')
        raw = (node.text or '') if node is not None else ''
        formula = cell.find(_S + 'f')
        if not raw and formula is None:
            visiting.remove(ref)
            cache[ref] = Decimal(0)
            return cache[ref]
        if not raw:
            raise ValueError('missing numeric value')
        cached = Decimal(raw)
        if not cached.is_finite() or abs(cached) > Decimal('1e20'):
            raise ValueError('numeric value exceeds bound')
        result = cached
        if formula is not None:
            expression = (formula.text or '').strip()
            dr = dc = 0
            if not expression:
                anchor_ref, expression = anchors[formula.get('si')]
                current_row, current_col = _xlsx_cell_coordinates(ref)
                anchor_row, anchor_col = _xlsx_cell_coordinates(anchor_ref)
                dr, dc = current_row - anchor_row, current_col - anchor_col

            def moved(match):
                abs_col, letters, abs_row, row = match.groups()
                col = _xlsx_column_index(letters + '1') + (0 if abs_col else dc)
                number = int(row) + (0 if abs_row else dr)
                if not (1 <= col <= 16384 and 1 <= number <= 1048576):
                    raise ValueError('shifted cell reference out of bounds')
                return _xlsx_column_name(col) + str(number)
            expression = _REF.sub(moved, expression)

            def expand_sum(match):
                a, b = match.groups()
                r1, c1 = _xlsx_cell_coordinates(a)
                r2, c2 = _xlsx_cell_coordinates(b)
                if r2 < r1 or c2 < c1 or (r2-r1+1)*(c2-c1+1) > 10000:
                    raise ValueError('invalid formula range')
                return '(' + '+'.join(_xlsx_column_name(c)+str(r)
                    for r in range(r1, r2+1) for c in range(c1, c2+1)) + ')'
            expression = re.sub(r'(?i)SUM\(([A-Z]+[0-9]+):([A-Z]+[0-9]+)\)', expand_sum, expression)
            expression = _REF.sub(lambda m: str(evaluate(m.group(2) + m.group(4))), expression)
            tree = ast.parse(expression, mode='eval')

            def arithmetic(node):
                if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                    return Decimal(str(node.value))
                if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
                    val = arithmetic(node.operand)
                    return val if isinstance(node.op, ast.UAdd) else -val
                if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                    a, b = arithmetic(node.left), arithmetic(node.right)
                    if isinstance(node.op, ast.Add): return a + b
                    if isinstance(node.op, ast.Sub): return a - b
                    if isinstance(node.op, ast.Mult): return a * b
                    return a / b
                raise ValueError('unsupported formula')
            with localcontext() as context:
                context.prec = 28
                result = arithmetic(tree.body)
            if not result.is_finite() or result != cached:
                raise ValueError('formula cache mismatch')
        visiting.remove(ref)
        cache[ref] = result
        return result

    try:
        for ref, cell in cells.items():
            if cell.find(_S + 'f') is not None:
                evaluate(ref)
        return True
    except (ValueError, TypeError, KeyError, DecimalException, ZeroDivisionError, SyntaxError, RecursionError):
        return False


def xlsx_has_no_phone_evidence(document_root, tenant, metadata, person):
    if not person or not source_metadata_complete(metadata, tenant):
        return False
    try:
        rel, handle, before = open_source_file(document_root, tenant, metadata["source_relative_path"])
        with handle:
            if (rel.suffix.casefold() != ".xlsx" or before.st_size > 32 * 1024 * 1024
                    or not source_stat_matches_metadata(metadata, before)):
                return False
            payload = handle.read(32 * 1024 * 1024 + 1)
            after = os.fstat(handle.fileno())
        if (len(payload) != before.st_size or hashlib.sha256(payload).hexdigest() != metadata["source_sha256"]
                or (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns)):
            return False
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if (len(infos) > 4096 or len(set(names)) != len(names)
                    or sum(item.file_size for item in infos) > 64 * 1024 * 1024
                    or any(not _ALLOWED.fullmatch(name) for name in names)):
                return False
            root = _parse_xml_part(_read_zip_part(archive, "xl/workbook.xml", required=True), "xl/workbook.xml")
            rels = _parse_xml_part(_read_zip_part(archive, "xl/_rels/workbook.xml.rels", required=True), "xl/_rels/workbook.xml.rels")
            targets = {}
            for relation in rels.findall(_P + "Relationship"):
                if (relation.get("TargetMode") or "").casefold() == "external":
                    return False
                ident, target = relation.get("Id"), relation.get("Target")
                if not ident or not target or ident in targets:
                    return False
                targets[ident] = target
            declared = []
            for sheet in root.findall(".//" + _S + "sheets/" + _S + "sheet"):
                target = targets.get(sheet.get(_R + "id"))
                if not target:
                    return False
                target = target.lstrip("/")
                if not target.startswith("xl/"):
                    target = posixpath.normpath(posixpath.join("xl", target))
                if not re.fullmatch(r"xl/worksheets/sheet[0-9]+\.xml", target):
                    return False
                declared.append(target)
            sheet_files = [name for name in names if re.fullmatch(r"xl/worksheets/sheet[0-9]+\.xml", name)]
            if not declared or len(set(declared)) != len(declared) or set(declared) != set(sheet_files):
                return False
            shared = _xlsx_shared_strings(archive)
            if any(_CLUE.search(value) for value in shared):
                return False
            styles = _xlsx_style_formats(archive)
            formats = [fmt for _id, fmt in styles]
            if 'xl/styles.xml' in names:
                style_root = _parse_xml_part(_read_zip_part(archive, 'xl/styles.xml', required=True), 'xl/styles.xml')
                formats.extend(node.get('formatCode') or '' for node in style_root.iter()
                               if node.tag == _S + 'numFmt')
            for fmt in formats:
                # A custom format can inject digits absent from stored values.
                if (_CLUE.search(fmt) or len(re.findall(r'[0-9#?]', fmt)) >= 7):
                    return False
            all_text = []
            count = 0
            for path in declared:
                sheet = _parse_xml_part(_read_zip_part(archive, path, required=True), path)
                # Data beyond ordinary cell values requires a separate reader.
                if any(node.tag.rsplit("}", 1)[-1] in {"drawing", "legacyDrawing", "oleObjects", "controls", "extLst"} for node in sheet.iter()):
                    return False
                native_rows = _xlsx_sheet_rows(sheet, shared, styles, False)
                if not native_rows:
                    return False
                if not _formula_cache_verified(sheet):
                    return False
                # Only the first data row immediately under an authenticated
                # header can use an order-number exception. Never extrapolate
                # the meaning to later rows or inferred unlabeled columns.
                header_row, headers = native_rows[0]
                order_columns = {col for col, value in headers.items() if value == "מספר הזמנה"}
                first_data = native_rows[1][0] if len(native_rows) > 1 else -1
                relevant_merges = []
                native_cells = {cell.get('r'): cell for cell in sheet.findall('.//' + _S + 'sheetData/' + _S + 'row/' + _S + 'c')}
                for merge in sheet.findall('.//' + _S + 'mergeCells/' + _S + 'mergeCell'):
                    refs = (merge.get('ref') or '').split(':')
                    if len(refs) != 2:
                        return False
                    a, b = (_xlsx_cell_coordinates(ref) for ref in refs)
                    if not a or not b:
                        return False
                    r1, c1 = a
                    r2, c2 = b
                    if r1 > r2 or c1 > c2:
                        return False
                    if not (set(range(c1, c2 + 1)) & order_columns and r1 <= first_data and r2 >= header_row):
                        continue
                    if r1 != r2 or r1 not in {header_row, first_data}:
                        return False
                    if r1 == first_data and not set(range(c1, c2 + 1)) <= order_columns:
                        return False
                    if any(not (c2 < x[1] or x[3] < c1 or r2 < x[0] or x[2] < r1) for x in relevant_merges):
                        return False
                    relevant_merges.append((r1, c1, r2, c2))
                    for col in range(c1 + 1, c2 + 1):
                        other = native_cells.get(_xlsx_column_name(col) + str(r1))
                        if other is not None and any(child.tag in {_S + 'v', _S + 'f', _S + 'is'} for child in other):
                            return False
                anchors = {}
                formulas = []
                for cell in sheet.findall(".//" + _S + "sheetData/" + _S + "row/" + _S + "c"):
                    count += 1
                    if count > 100000:
                        return False
                    kind = cell.get("t") or "n"
                    if kind not in {"s", "inlineStr", "n", "b", "str", "d"}:
                        return False
                    v = cell.find(_S + "v")
                    value = (v.text or "") if v is not None else ""
                    if kind == "s":
                        if not value.isdigit() or int(value) >= len(shared):
                            return False
                        value = shared[int(value)]
                    elif kind == "inlineStr":
                        value = "".join(node.text or "" for node in cell.findall(".//" + _S + "is//" + _S + "t"))
                    if kind == "n" and re.fullmatch(r"[+-]?[0-9.]+[eE][+-]?[0-9]+", value):
                        return False
                    if _CLUE.search(value):
                        ref = cell.get("r") or ""
                        col = _xlsx_column_index(ref)
                        rowmatch = re.fullmatch(r"[A-Z]+([0-9]+)", ref)
                        is_order = (kind == "n" and value.isdigit() and rowmatch is not None
                                    and int(rowmatch.group(1)) == first_data == header_row + 1
                                    and col in order_columns)
                        if not is_order:
                            return False
                    all_text.append(value)
                    formula = cell.find(_S + "f")
                    if formula is not None:
                        if not value or kind not in {"n", "b"}:
                            return False
                        expression = (formula.text or "").strip()
                        if expression and not _safe_formula(expression):
                            return False
                        shared_id = formula.get("si")
                        if expression and shared_id is not None:
                            if shared_id in anchors:
                                return False
                            anchors[shared_id] = expression
                        formulas.append((expression, shared_id, formula.get("t")))
                if any(not expression and (kind != "shared" or ident not in anchors)
                       for expression, ident, kind in formulas):
                    return False
            return any(person in value for value in all_text)
    except (OSError, ValueError, TypeError, KeyError, IndexError, NotImplementedError, ET.ParseError, zipfile.BadZipFile, IngestExtractionError):
        return False
