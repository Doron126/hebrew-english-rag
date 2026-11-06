"""Positioned PDF section flows, before lossy whitespace normalization.

Only comparable, vertically overlapping text columns activate this path. Small
marginal notes do not. Source words/blocks remain auditable; no offsets into a
separate flattened extractor string are claimed.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import json, math, re, statistics, unicodedata
from typing import Any
try:
    from .canonical_text import CanonicalEdit, CanonicalText, canonicalize_reconstructed_source, canonicalize_source_text
    from .pdf_reconstruction import positioned_tokens_from_pymupdf, pymupdf_line_styles, reconstruct_page
except ImportError:
    from canonical_text import CanonicalEdit, CanonicalText, canonicalize_reconstructed_source, canonicalize_source_text
    from pdf_reconstruction import positioned_tokens_from_pymupdf, pymupdf_line_styles, reconstruct_page
PDF_LAYOUT_REVISION = 'pdf-section-flows-v2'
PDF_LAYOUT_CANONICAL_REVISION = 'logical-text-v3-pdf-flows-v3'
PDF_PACKING_REVISION = 'pdf-paragraph-groups-v1'
PDF_NATIVE_ROW_ASSOCIATION_REVISION = 'native-line-label-scalar-v1'

@dataclass(frozen=True)
class LayoutFlow:
    page: int
    tokens: tuple
    heading: str
    blocks: tuple
    ambiguous: bool = False
    continuations: tuple = ()
    packing_geometry: dict | None = None

@dataclass(frozen=True)
class LayoutDocument:
    canonical: CanonicalText
    flows: tuple[dict, ...]

def _box(block):
    return tuple((float(x) for x in block['bbox']))

def _text(block):
    return ' '.join((s.get('text', '') for l in block.get('lines', []) for s in l.get('spans', []))).strip()

def _size(block):
    values = [(float(s.get('size', 0)), len(s.get('text', '').strip())) for l in block.get('lines', []) for s in l.get('spans', []) if s.get('text', '').strip()]
    return statistics.median([size for size, n in values for _ in range(max(1, n))]) if values else 0

def _columns(blocks, width, height):
    """Find comparable text columns and report whether continuity is well supported. A visible gutter with insufficient evidence is preserved as ambiguous blocks."""
    sizes = [_size(b) for b in blocks for _ in range(max(1, min(300, len(_text(b)))))]
    body_size = statistics.median(sizes) if sizes else 0
    def heading_only(block):
        # A source heading can span the gutter without being a body column.
        # Mixed heading+body blocks still contribute their real body extent.
        lines = [line for line in block.get('lines', []) if any(span.get('text', '').strip() for span in line.get('spans', []))]
        return len(lines) == 1 and bool(_visual_heading(block, body_size))
    major = [b for b in blocks if not heading_only(b) and width * 0.22 <= _box(b)[2] - _box(b)[0] <= width * 0.64 and len(_text(b)) >= 45]
    groups = []
    for b in sorted(major, key=lambda b: _box(b)[0]):
        x0, y0, x1, y1 = _box(b)
        if groups and x0 < groups[-1][1] - min(x1 - x0, groups[-1][1] - groups[-1][0]) * 0.2:
            groups[-1][1] = max(groups[-1][1], x1)
            groups[-1][2].append(b)
        else:
            groups.append([x0, x1, [b]])
    if not 2 <= len(groups) <= 3:
        return None
    widths = [g[1] - g[0] for g in groups]
    if min(widths) < max(widths) * 0.65:
        return None
    confident = all((len(g[2]) >= 2 and sum((len(_text(b)) for b in g[2])) >= 180 for g in groups))
    for left, right in zip(groups, groups[1:]):
        if right[0] - left[1] < max(10, width * 0.018):
            return None
        top = max(min((_box(b)[1] for b in left[2])), min((_box(b)[1] for b in right[2])))
        bottom = min(max((_box(b)[3] for b in left[2])), max((_box(b)[3] for b in right[2])))
        if bottom - top < height * 0.16:
            confident = False
    return ([(g[0], g[1]) for g in groups], confident)

def _visual_heading(block, body_size):
    """Recognize a source heading from font size or an all-bold first line."""
    lines = block.get('lines', [])
    if not lines:
        return ''
    first = lines[0]
    text = ''.join((s.get('text', '') for s in first.get('spans', []))).strip()
    size = max([float(s.get('size', 0)) for s in first.get('spans', [])] or [0])
    spans = [s for s in first.get('spans', []) if s.get('text', '').strip()]
    bold = bool(spans) and all((int(s.get('flags', 0)) & 16 or re.search('bold|(?:^|[-_])bd(?:$|[-_])', str(s.get('font', '')), re.I) for s in spans))
    return text if text and len(text.split()) <= 16 and (size >= body_size * 1.075 or bold) and (not re.search('[.!?;]$', text)) else ''

def _lines(block):
    return [line for line in block.get('lines', []) if any(span.get('text', '').strip() for span in line.get('spans', []))]


def _line_size(line):
    sizes = [float(span.get('size', 0)) for span in line.get('spans', []) for _ in range(len(span.get('text', '').strip()))]
    return statistics.median(sizes) if sizes else 0


def _line_bold(line):
    spans = [span for span in line.get('spans', []) if span.get('text', '').strip()]
    return bool(spans) and all(int(span.get('flags', 0)) & 16 or re.search('bold|(?:^|[-_])bd(?:$|[-_])', str(span.get('font', '')), re.I) for span in spans)


def _contrasts(first, body_lines):
    if not body_lines:
        return False
    size = statistics.median([_line_size(line) for line in body_lines])
    return (_line_size(first) >= size * 1.075 and size > 0) or (_line_bold(first) and not any(_line_bold(line) for line in body_lines))


def _heading(block, body_size, following=None):
    """An ownership anchor needs local body contrast, not a page-wide median."""
    lines = _lines(block)
    if not lines:
        return ''
    first = lines[0]
    text = ''.join(span.get('text', '') for span in first.get('spans', [])).strip()
    if not text or len(text.split()) > 16 or re.search('[.!?;]$', text):
        return ''
    if len(lines) > 1:
        return text if _contrasts(first, lines[1:]) else ''
    if following is None:
        return ''
    next_lines = _lines(following)
    if len(next_lines) < 2 or _contrasts(next_lines[0], next_lines[1:]):
        return ''
    a, b = _box(block), _box(following)
    overlap = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(1, a[3] - a[1], next_lines[0]['bbox'][3] - next_lines[0]['bbox'][1])
    if overlap < min(a[2] - a[0], b[2] - b[0]) * .55 or not -height * .2 <= b[1] - a[3] <= height * 1.6:
        return ''
    return text if _contrasts(first, next_lines) else ''


def _unfinished(block):
    value = _text(block).rstrip().rstrip('"' + "')]}”’")
    return bool(value) and not value.endswith(('.', '!', '?', '׃', '…'))


def _paragraph_link(first, second, first_column, second_column, columns, body_size, first_in_column):
    """Record geometric evidence for one unfinished paragraph, not a new owner."""
    if first_column == second_column:
        # Distinct same-column PDF blocks remain distinct. Within-block wraps
        # are reconstructed separately; matching fonts do not establish owner.
        return False
    if int(second.get('number', -1)) != int(first.get('number', -1)) + 1 or not _unfinished(first):
        return False
    left, right = _lines(first)[-1], _lines(second)[0]
    a, b = left['bbox'], right['bbox']
    first_fonts = {span.get('font', '') for span in left.get('spans', []) if span.get('text', '').strip()}
    second_fonts = {span.get('font', '') for span in right.get('spans', []) if span.get('text', '').strip()}
    if first_fonts != second_fonts or not first_fonts or abs(_line_size(left) - _line_size(right)) > body_size * .1:
        return False
    if a[2] - a[0] < (columns[first_column][1] - columns[first_column][0]) * .8 or b[2] - b[0] < (columns[second_column][1] - columns[second_column][0]) * .8:
        return False
    return first_in_column



def _scalar_cell(tokens):
    """Recognize a literal numeric value, optionally with currency/percent."""
    value = ''.join(token.text for token in tokens)
    return bool(value) and any(char.isdecimal() for char in value) and all(
        char.isdecimal() or char.isspace() or char in ',.+-−()%‰'
        or unicodedata.category(char) == 'Sc'
        for char in value
    )


def _native_scalar_rows(groups, cell_for):
    """Associate a label/value row only when native and cell geometry agree.

    Separate text columns, independent native lines, repeated/multiple scalar
    columns and table headers remain isolated. The result remains an ambiguous
    unowned flow; this association cannot confer a heading on other content.
    """
    by_line = {}
    for key, selected in groups.items():
        if key not in cell_for or not selected:
            continue
        native = {(token.page, token.block, token.line) for token in selected}
        if len(native) == 1:
            by_line.setdefault(next(iter(native)), []).append(key)
    pairs = {}
    for native, keys in by_line.items():
        if len(keys) != 2:
            continue
        first, second = (groups[key] for key in keys)
        scalar = (_scalar_cell(first), _scalar_cell(second))
        if scalar[0] == scalar[1]:
            continue
        label = second if scalar[0] else first
        if not any(char.isalpha() for token in label for char in token.text):
            continue
        selected = first + second
        height = max(token.height for token in selected)
        if not math.isfinite(height) or height <= 0:
            continue
        # Every member must share a physical baseline, not just an extractor ID.
        if (max(token.y0 for token in selected) - min(token.y0 for token in selected) > height * .2
                or max(token.y1 for token in selected) - min(token.y1 for token in selected) > height * .2):
            continue
        fonts = {token.font for token in selected if token.font}
        sizes = [token.size for token in selected if token.size > 0]
        if len(fonts) > 1 or (sizes and max(sizes) - min(sizes) > height * .15):
            continue
        left, right = sorted((cell_for[key] for key in keys), key=lambda box: box[0])
        tolerance = max(1.0, height * .1)
        if (abs(left[1] - right[1]) > tolerance
                or abs(left[3] - right[3]) > tolerance
                or abs(left[2] - right[0]) > tolerance):
            continue
        for key in keys:
            pairs[key] = tuple(keys)
    return pairs


def positioned_page_flows(words, style_dict, page, width, height, *, direction='auto', table_cells=()):
    """Keep independent sections, table cells and uncertain regions separate. Column continuation records geometric cues, never a semantic guarantee."""
    blocks = [b for b in style_dict.get('blocks', []) if b.get('type') == 0 and _text(b)]
    result = _columns(blocks, width, height)
    if not result:
        return None
    columns, confident = result
    tokens = positioned_tokens_from_pymupdf(page, words, pymupdf_line_styles(style_dict))
    by_block = {}
    for t in tokens:
        by_block.setdefault(t.block, []).append(t)
    blocks = [b for b in blocks if int(b.get('number', -1)) in by_block]
    chars = ''.join((_text(b) for b in blocks))
    rtl = direction == 'rtl' or (direction == 'auto' and len(re.findall('[\\u0590-\\u05ff]', chars)) > len(re.findall('[A-Za-z]', chars)))
    sizes = [_size(b) for b in blocks for _ in range(max(1, min(300, len(_text(b)))))]
    body = statistics.median(sizes) if sizes else 10
    if table_cells:
        cells = sorted(set((tuple((float(v) for v in box)) for box in table_cells if box)))
        groups = {}
        cell_for = {}
        for t in tokens:
            x = (t.x0 + t.x1) / 2
            y = (t.y0 + t.y1) / 2
            candidates = [(i, b) for i, b in enumerate(cells) if b[0] <= x <= b[2] and b[1] <= y <= b[3]]
            match = min(candidates, key=lambda pair: (pair[1][2] - pair[1][0]) * (pair[1][3] - pair[1][1])) if candidates else None
            key = ('cell', match[0]) if match else ('block', t.block)
            groups.setdefault(key, []).append(t)
            if match:
                cell_for[key] = match[1]
        output = []
        block_map = {int(b['number']): b for b in blocks}
        associated = _native_scalar_rows(groups, cell_for)
        emitted = set()
        for key in groups:
            if key in emitted:
                continue
            member_keys = associated.get(key, (key,))
            emitted.update(member_keys)
            selected = [token for member in member_keys for token in groups[member]]
            ids = sorted({t.block for t in selected})
            mapping = {n: i for i, n in enumerate(ids)}
            provenance = []
            for member in member_keys:
                for n in sorted({token.block for token in groups[member]}):
                    item = {
                        'block': n, 'bbox': list(_box(block_map[n])),
                        'word_ids': [[t.line, t.word] for t in groups[member] if t.block == n],
                    }
                    if member in cell_for:
                        item['cell_bbox'] = list(cell_for[member])
                    if len(member_keys) > 1:
                        item['native_row_association_revision'] = PDF_NATIVE_ROW_ASSOCIATION_REVISION
                    provenance.append(item)
            ordered = tuple((replace(t, block=mapping[t.block]) for t in selected))
            output.append(LayoutFlow(page, ordered, '' if key in cell_for else _heading(block_map[ids[0]], body), tuple(provenance), True))
        if sum((len(f.tokens) for f in output)) != len(tokens):
            raise ValueError('PDF table word conservation failed')
        return sorted(output, key=lambda f: (min((t.y0 for t in f.tokens)), -min((t.x0 for t in f.tokens)) if rtl else min((t.x0 for t in f.tokens))))
    if not confident:
        output = []
        for rank, b in enumerate(sorted(blocks, key=lambda b: (_box(b)[1], -_box(b)[0] if rtl else _box(b)[0]))):
            n = int(b.get('number', -1))
            selected = tuple((replace(t, block=0) for t in by_block[n]))
            output.append(LayoutFlow(page, selected, _heading(b, body), ({'block': n, 'bbox': list(_box(b)), 'reading_order': rank},), True))
        if sum((len(f.tokens) for f in output)) != len(tokens):
            raise ValueError('PDF layout word conservation failed')
        return output
    gutters = [(a[1] + b[0]) / 2 for a, b in zip(columns, columns[1:])]
    spanning = []
    regular = []
    for b in blocks:
        x0, y0, x1, y1 = _box(b)
        if any((x0 + 10 < g < x1 - 10 for g in gutters)):
            spanning.append(b)
        else:
            regular.append(b)
    spanning.sort(key=lambda b: (_box(b)[1], _box(b)[0]))
    bands = [[] for _ in range(len(spanning) + 1)]
    for b in regular:
        cy = (_box(b)[1] + _box(b)[3]) / 2
        band = sum((cy > (_box(s)[1] + _box(s)[3]) / 2 for s in spanning))
        bands[band].append(b)
    result = []
    reading_rank = {}
    memberships = {}
    rank = 0

    def make(bs, heading='', ambiguous=False, continuations=()):
        selected = []
        provenance = []
        for order, b in enumerate(bs):
            num = int(b.get('number', -1))
            provenance.append({'block': num, 'bbox': list(_box(b)), 'reading_order': reading_rank.get(num, 0)})
            selected.extend((replace(t, block=order) for t in by_block[num]))
        if selected:
            geometry = None
            identities = {memberships.get(int(b.get('number', -1))) for b in bs}
            if not ambiguous and len(identities) == 1 and None not in identities:
                band, column = next(iter(identities))
                bbox = [min(t.x0 for t in selected), min(t.y0 for t in selected), max(t.x1 for t in selected), max(t.y1 for t in selected)]
                line_height = statistics.median(t.height for t in selected)
                if all(math.isfinite(value) for value in [*bbox, line_height]) and line_height > 0:
                    geometry = {'page': page, 'column': column, 'band': band, 'confident': True, 'direction': 'rtl' if rtl else 'ltr', 'bbox': bbox, 'line_height': line_height, 'reading_edge': bbox[2] if rtl else bbox[0]}
            result.append(LayoutFlow(page, tuple(selected), heading, tuple(provenance), ambiguous, tuple(continuations), geometry))
    for band_index, band in enumerate(bands):
        buckets = [[] for _ in columns]
        for b in band:
            center = (_box(b)[0] + _box(b)[2]) / 2
            idx = min(range(len(columns)), key=lambda i: abs(center - sum(columns[i]) / 2))
            buckets[idx].append(b)
            memberships[int(b.get('number', -1))] = (band_index, idx)
        active = []
        title = ''
        last_body = None
        previous_column = None
        continuations = []
        ambiguous = False
        pending_body = None

        def flush():
            nonlocal active, title, last_body, previous_column, continuations, ambiguous, pending_body
            if active:
                make(active, title, ambiguous, continuations=continuations)
            active, title, last_body, previous_column = [], '', None, None
            continuations, ambiguous, pending_body = [], False, None
        for col in sorted(range(len(columns)), reverse=rtl):
            ordered = sorted(buckets[col], key=lambda b: (_box(b)[1], _box(b)[0]))
            for bi, b in enumerate(ordered):
                number = int(b.get('number', -1))
                reading_rank[number] = rank
                rank += 1
                following = ordered[bi + 1] if bi + 1 < len(ordered) else None
                if following is not None and _size(following) < body * .9:
                    following = None
                heading = _heading(b, body, following)
                visual_candidate = _visual_heading(b, body)
                if _size(b) < body * .9:
                    make([b], '', True)
                    continue
                linked = False
                kind = ''
                if active and not heading:
                    if previous_column == col and pending_body == number:
                        linked, kind = True, 'contrasted-source-heading-and-its-body'
                    elif _paragraph_link(last_body, b, previous_column, col, columns, body, bi == 0):
                        linked, kind = True, 'geometric-unfinished-paragraph'
                if active and not linked:
                    flush()
                if heading:
                    title = heading
                if linked:
                    continuations.append({'kind': kind, 'from_block': int(last_body['number']), 'to_block': number, 'from_column': previous_column, 'to_column': col, 'evidence': 'local contrasted heading/body or adjacent aligned full-width unfinished paragraph with matching font/size', 'semantic_guarantee': False})
                elif visual_candidate and not heading:
                    ambiguous = True
                active.append(b)
                last_body, previous_column = b, col
                pending_body = int(following['number']) if heading and len(_lines(b)) == 1 and following is not None else None
        flush()
        if band_index < len(spanning):
            b = spanning[band_index]
            reading_rank[int(b.get('number', -1))] = rank
            rank += 1
            # A possible spanning title is bounded to this source block; it
            # cannot confer ownership over independent columns below it.
            label = _visual_heading(b, body) if len(_lines(b)) == 1 else ''
            make([b], label, True)
    expected = sorted(((t.block, t.line, t.word, t.text) for t in tokens))
    actual = []
    for f in result:
        for b in f.blocks:
            actual.extend(((t.block, t.line, t.word, t.text) for t in by_block[b['block']]))
    if sorted(actual) != expected:
        raise ValueError('PDF layout word conservation failed')
    return sorted(result, key=lambda f: min((b['reading_order'] for b in f.blocks)))

def reconstruct_document(path, legacy_text, fitz_module):
    """Return a new truthful coordinate-flow generation, or preserve legacy bytes."""
    if fitz_module is None or not all((hasattr(fitz_module, x) for x in ('open', 'TEXTFLAGS_DICT', 'TEXT_PRESERVE_IMAGES'))):
        return None
    doc = fitz_module.open(str(path))
    page_layouts = {}
    direction = 'rtl' if len(re.findall('[\\u0590-\\u05ff]', legacy_text)) > len(re.findall('[A-Za-z]', legacy_text)) else 'ltr'
    try:
        for i, page in enumerate(doc):
            if not hasattr(page, 'get_text'):
                return None
            words = page.get_text('words') or []
            styles = page.get_text('dict', flags=fitz_module.TEXTFLAGS_DICT & ~fitz_module.TEXT_PRESERVE_IMAGES) or {}
            if not words:
                continue
            table_cells = []
            if hasattr(page, 'find_tables') and _columns([b for b in styles.get('blocks', []) if b.get('type') == 0 and _text(b)], float(page.rect.width), float(page.rect.height)):
                for table in page.find_tables().tables:
                    table_cells.extend((cell for cell in table.cells if cell))
            flows = positioned_page_flows(words, styles, i + 1, float(page.rect.width), float(page.rect.height), direction=direction, table_cells=table_cells)
            if flows:
                page_layouts[i + 1] = flows
    finally:
        doc.close()
    if not page_layouts:
        return None
    markers = list(re.finditer('(?m)^# Page\\s+(\\d+)\\s*$', legacy_text))
    pages = {int(m.group(1)): legacy_text[m.end():markers[n + 1].start() if n + 1 < len(markers) else len(legacy_text)].strip() for n, m in enumerate(markers)}
    if set(page_layouts) - set(pages):
        raise ValueError('positioned page missing from extraction')
    raw_parts = []
    text_parts = []
    origins = []
    edits = []
    flow_records = []
    raw_cursor = 0
    cursor = 0

    def append(part):
        nonlocal raw_cursor, cursor
        raw_parts.append(part.raw_text)
        text_parts.append(part.text)
        origins.extend((raw_cursor + x if x is not None else None for x in part.origin_offsets))
        edits.extend((CanonicalEdit(e.raw_start + raw_cursor, e.raw_end + raw_cursor, e.canonical_start + cursor, e.canonical_end + cursor, e.reason_code) for e in part.edits))
        raw_cursor += len(part.raw_text)
        cursor += len(part.text)
    for page, body in sorted(pages.items()):
        if raw_parts:
            append(CanonicalText('\n\n', '\n\n', (None, None), ()))
        prefix = f'# Page {page}\n\n'
        append(CanonicalText(prefix, prefix, (None,) * len(prefix), ()))
        # A page with severe word loss cannot safely replace the complete
        # native fallback just because a different page has column flows.
        flows = page_layouts.get(page, ())
        legacy_chars = sum(ch.isalnum() for ch in body)
        flow_chars = sum(ch.isalnum() for f in flows for token in f.tokens for ch in token.text)
        if not flows or (legacy_chars >= 100 and flow_chars < legacy_chars * .7):
            part = canonicalize_source_text(body, repair_visual_bidi=True, normalize_pdf_spacing=True, join_pdf_line_wraps=True)
            start = cursor
            append(part)
            flow_records.append({'page': page, 'heading': '', 'canonical_start': start, 'canonical_end': cursor, 'blocks': [], 'layout_changed': False, 'ambiguous': False})
            continue
        for f in page_layouts[page]:
            if text_parts and (not text_parts[-1].endswith('\n\n')):
                append(CanonicalText('\n\n', '\n\n', (None, None), ()))
            reconstructed = reconstruct_page(f.tokens, preserve_ltr_runs=True)
            part = canonicalize_reconstructed_source(reconstructed.raw_text, reconstructed.text, reconstructed.origin_offsets, tuple((CanonicalEdit(**e.__dict__) for e in reconstructed.edits)), repair_visual_bidi=True, normalize_pdf_spacing=True, join_pdf_line_wraps=False)
            start = cursor
            append(part)
            flow_records.append({'page': page, 'heading': f.heading, 'canonical_start': start, 'canonical_end': cursor, 'blocks': list(f.blocks), 'column_continuations': [item for item in f.continuations if item['from_column'] != item['to_column']], 'continuations': list(f.continuations), 'packing_geometry': f.packing_geometry, 'layout_changed': True, 'ambiguous': f.ambiguous})
    canonical = CanonicalText(''.join(raw_parts), ''.join(text_parts), tuple(origins), tuple(edits), revision=PDF_LAYOUT_CANONICAL_REVISION)
    for index, f in enumerate(flow_records):
        f['flow_id'] = f"page-{f['page']}-flow-{index + 1}"
        f['source_flow_index'] = index
        f['raw_spans'] = canonical.raw_spans_for(f['canonical_start'], f['canonical_end'])
    return LayoutDocument(canonical, tuple(flow_records))


def _packing_geometry(flow):
    geometry = flow.get('packing_geometry')
    if not isinstance(geometry, dict) or geometry.get('confident') is not True:
        return None
    if flow.get('heading') or flow.get('ambiguous') or flow.get('column_continuations'):
        return None
    if any(type(geometry.get(key)) is not int for key in ('page', 'column', 'band')):
        return None
    if geometry['page'] != flow.get('page') or geometry.get('direction') not in ('ltr', 'rtl'):
        return None
    box = geometry.get('bbox')
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    values = [*box, geometry.get('line_height'), geometry.get('reading_edge')]
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values) or geometry['line_height'] <= 0:
        return None
    if box[0] >= box[2] or box[1] >= box[3]:
        return None
    return geometry


def paragraph_packing_groups(document):
    """Pack source paragraphs, never owners or sentences; canonical data is read-only.

    Only consecutive unowned/unambiguous flows in one explicitly confident
    positioned column/band are eligible. Existing paragraph separators remain
    literal. The shared splitter, not this helper, sets chunk size and overlap.
    """
    groups = []
    for flow in document.flows:
        group = groups[-1] if groups else None
        previous = group['members'][-1] if group else None
        first_geometry = _packing_geometry(previous) if previous else None
        geometry = _packing_geometry(flow)
        join = False
        if first_geometry is not None and geometry is not None:
            height = max(first_geometry['line_height'], geometry['line_height'])
            gap = geometry['bbox'][1] - first_geometry['bbox'][3]
            join = (
                all(first_geometry[key] == geometry[key] for key in ('page', 'column', 'band', 'direction'))
                and flow['source_flow_index'] == previous['source_flow_index'] + 1
                and document.canonical.text[previous['canonical_end']:flow['canonical_start']] == '\n\n'
                and -height * .2 <= gap <= height * 2
                and abs(first_geometry['reading_edge'] - geometry['reading_edge']) <= height * 2
            )
        if join:
            group['members'].append(flow)
            group['canonical_end'] = flow['canonical_end']
        else:
            groups.append({'id': f"page-{flow['page']}-packing-{len(groups) + 1}", 'page': flow['page'], 'members': [flow], 'canonical_start': flow['canonical_start'], 'canonical_end': flow['canonical_end']})
    return groups


def packing_group_metadata(group, canonical):
    """Describe full-group provenance, without claiming per-chunk contribution."""
    members = group['members']
    blocks = [block for flow in members for block in flow['blocks']]
    geometry = members[0].get('packing_geometry') if len(members) == 1 else dict(members[0]['packing_geometry'])
    if len(members) > 1:
        boxes = [flow['packing_geometry']['bbox'] for flow in members]
        geometry['bbox'] = [min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes)]
    metadata = {
        'page': group['page'],
        'pdf_packing_revision': PDF_PACKING_REVISION,
        'pdf_packing_group_id': group['id'],
        'pdf_packing_group_member_count': len(members),
        'pdf_packing_group_flow_ids_json': json.dumps([flow['flow_id'] for flow in members], separators=(',', ':')),
        'pdf_packing_group_member_spans_json': json.dumps(members, separators=(',', ':')),
        'pdf_packing_group_canonical_start': group['canonical_start'],
        'pdf_packing_group_canonical_end': group['canonical_end'],
        'pdf_packing_group_raw_spans_json': json.dumps(canonical.raw_spans_for(group['canonical_start'], group['canonical_end']), separators=(',', ':')),
        'pdf_packing_group_blocks_json': json.dumps(blocks, separators=(',', ':')),
        'pdf_packing_group_geometry_json': json.dumps(geometry, separators=(',', ':')),
        'pdf_packing_offset_contract': 'full packing-group bounds and ordered member-flow ranges; not exact contributing spans of this chunk; overlap may repeat source tails',
        'pdf_layout_changed': any(flow['layout_changed'] for flow in members),
        'pdf_layout_ambiguous': any(flow['ambiguous'] for flow in members),
    }
    if len(members) == 1:
        flow = members[0]
        metadata.update({
            'pdf_flow_id': flow['flow_id'], 'pdf_flow_heading': flow['heading'],
            'pdf_flow_canonical_start': flow['canonical_start'], 'pdf_flow_canonical_end': flow['canonical_end'],
            'pdf_flow_blocks_json': json.dumps(flow['blocks'], separators=(',', ':')),
            'pdf_flow_column_continuations_json': json.dumps(flow.get('column_continuations', []), separators=(',', ':')),
            'pdf_flow_continuations_json': json.dumps(flow.get('continuations', []), separators=(',', ':')),
            'pdf_flow_raw_spans_json': json.dumps(flow['raw_spans'], separators=(',', ':')),
            'pdf_flow_offset_contract': 'flow bounds in reordered coordinate generation, not chunk offsets or old layout offsets',
        })
    return metadata


def validate_pdf_packing_metadata(metadata):
    """Validate provenance structure only; this never judges semantic support.

    Legacy/non-layout metadata without packing fields remains valid. Ranges are
    explicitly full-group/flow bounds, not invented exact chunk contribution.
    """
    if not any(str(key).startswith('pdf_packing_') for key in metadata):
        return None
    required = ('pdf_packing_revision', 'pdf_packing_group_id', 'pdf_packing_group_member_count', 'pdf_packing_group_flow_ids_json', 'pdf_packing_group_member_spans_json', 'pdf_packing_group_canonical_start', 'pdf_packing_group_canonical_end', 'pdf_packing_group_raw_spans_json', 'pdf_packing_group_blocks_json', 'pdf_packing_group_geometry_json', 'pdf_packing_offset_contract')
    if any(key not in metadata for key in required):
        raise ValueError('missing PDF packing provenance field')
    if metadata['pdf_packing_revision'] != PDF_PACKING_REVISION:
        raise ValueError('unsupported PDF packing revision')
    if not isinstance(metadata['pdf_packing_group_id'], str) or not metadata['pdf_packing_group_id']:
        raise ValueError('invalid PDF packing group identity')
    if type(metadata.get('page')) is not int or metadata['page'] < 1:
        raise ValueError('invalid PDF packing page')
    count = metadata['pdf_packing_group_member_count']
    start, end = metadata['pdf_packing_group_canonical_start'], metadata['pdf_packing_group_canonical_end']
    if type(count) is not int or count < 1 or type(start) is not int or type(end) is not int or not 0 <= start < end:
        raise ValueError('invalid PDF packing count or bounds')
    try:
        ids = json.loads(metadata['pdf_packing_group_flow_ids_json'])
        members = json.loads(metadata['pdf_packing_group_member_spans_json'])
        spans = json.loads(metadata['pdf_packing_group_raw_spans_json'])
        blocks = json.loads(metadata['pdf_packing_group_blocks_json'])
        geometry = json.loads(metadata['pdf_packing_group_geometry_json'])
    except (TypeError, ValueError) as error:
        raise ValueError('invalid PDF packing JSON provenance') from error
    if not isinstance(ids, list) or not isinstance(members, list) or len(ids) != count or len(members) != count or any(not isinstance(item, str) or not item for item in ids) or len(set(ids)) != count:
        raise ValueError('PDF packing member count/identity mismatch')
    for index, member in enumerate(members):
        if not isinstance(member, dict) or member.get('flow_id') != ids[index] or member.get('page') != metadata.get('page'):
            raise ValueError('PDF packing member order/page mismatch')
        if type(member.get('page')) is not int or not isinstance(member.get('heading'), str) or type(member.get('ambiguous')) is not bool:
            raise ValueError('invalid PDF packing member types')
        a, b = member.get('canonical_start'), member.get('canonical_end')
        if type(a) is not int or type(b) is not int or not start <= a < b <= end or (index and a < members[index - 1]['canonical_end']):
            raise ValueError('PDF packing member bounds mismatch')
        if type(member.get('source_flow_index')) is not int or (index and member['source_flow_index'] != members[index - 1]['source_flow_index'] + 1):
            raise ValueError('PDF packing members are not consecutive source flows')
        if not isinstance(member.get('blocks'), list):
            raise ValueError('invalid PDF packing member blocks')
    if members[0]['canonical_start'] != start or members[-1]['canonical_end'] != end or blocks != [block for member in members for block in member['blocks']]:
        raise ValueError('PDF packing group/member provenance mismatch')
    for block in blocks:
        if not isinstance(block, dict) or type(block.get('block')) is not int or block['block'] < 0 or not isinstance(block.get('bbox'), list) or len(block['bbox']) != 4 or any(type(value) not in (int, float) or not math.isfinite(value) for value in block['bbox']):
            raise ValueError('invalid PDF packing block provenance')
    for bounds in [spans, *[member.get('raw_spans') for member in members]]:
        if not isinstance(bounds, list) or any(not isinstance(pair, list) or len(pair) != 2 or any(type(value) is not int for value in pair) or not 0 <= pair[0] < pair[1] for pair in bounds):
            raise ValueError('invalid PDF packing raw spans')
    if not isinstance(metadata['pdf_packing_offset_contract'], str) or not metadata['pdf_packing_offset_contract'].startswith('full packing-group bounds'):
        raise ValueError('PDF packing offset scope missing')
    if count > 1:
        geometries = [_packing_geometry(member) for member in members]
        if any(item is None for item in geometries) or not isinstance(geometry, dict):
            raise ValueError('compound PDF packing lacks confident geometry')
        if geometry.get('confident') is not True or any(geometry.get(key) != geometries[0][key] for key in ('page', 'column', 'band', 'direction')) or any(any(item[key] != geometries[0][key] for key in ('page', 'column', 'band', 'direction')) for item in geometries):
            raise ValueError('compound PDF packing crosses physical region')
        boxes = [item['bbox'] for item in geometries]
        expected_box = [min(box[0] for box in boxes), min(box[1] for box in boxes), max(box[2] for box in boxes), max(box[3] for box in boxes)]
        if geometry.get('bbox') != expected_box:
            raise ValueError('PDF packing group geometry/member mismatch')
        if any(key in metadata for key in ('pdf_flow_id', 'pdf_flow_heading', 'pdf_flow_canonical_start', 'pdf_flow_canonical_end', 'pdf_flow_blocks_json', 'pdf_flow_raw_spans_json')):
            raise ValueError('compound PDF packing masquerades as one flow')
    elif metadata.get('pdf_flow_id') != ids[0] or metadata.get('pdf_flow_canonical_start') != start or metadata.get('pdf_flow_canonical_end') != end:
        raise ValueError('single-flow PDF packing identity mismatch')
    return {'group_id': metadata['pdf_packing_group_id'], 'members': members, 'canonical_start': start, 'canonical_end': end, 'geometry': geometry}
