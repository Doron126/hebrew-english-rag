"""Bounded native table associations for already authenticated monetary pages.

This is a query-time view, never an alternative ingestion format. Native rules
and labelled cells own each amount; text order and number ordinals do not.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import hashlib
import json
import math
import re
import unicodedata

from canonical_text import canonicalize_source_text
from financial_semantics import _CURRENCY, _decimal_literal, _money_matches, monetary_role
from pdf_reconstruction import positioned_tokens_from_pymupdf, reconstruct_page


@dataclass(frozen=True)
class NativeMonetaryField:
    amount: Decimal
    currency: str
    word_ids: tuple


@dataclass(frozen=True)
class NativeMonetaryItem:
    description: str
    quantity: Decimal
    unit_price: Decimal
    line_total: Decimal
    currency: str
    word_ids: tuple
    item_code: str = ''
    description_word_ids: tuple = ()
    quantity_word_ids: tuple = ()
    unit_price_word_ids: tuple = ()
    line_total_word_ids: tuple = ()


@dataclass(frozen=True)
class NativeMonetaryPayment:
    amount: Decimal
    currency: str
    payment_date: date | None
    method: str
    word_ids: tuple
    context_text: str
    span: tuple[int, int]
    amount_word_ids: tuple = ()
    date_word_ids: tuple = ()
    method_word_ids: tuple = ()


@dataclass(frozen=True)
class NativeMonetaryPage:
    source_text: str
    quote_span: tuple[int, int]
    geometry_sha256: str
    page: int
    document_kind: str
    document_number: str
    issue_date: date | None
    items: tuple[NativeMonetaryItem, ...]
    subtotal: NativeMonetaryField
    tax: NativeMonetaryField
    payable: NativeMonetaryField
    vat_rate: Decimal
    qualifiers: tuple[str, ...]
    context_text: str
    payments: tuple[NativeMonetaryPayment, ...] = ()
    payment_table_present: bool = False
    service_context_text: str = ''
    vat_label_word_ids: tuple = ()
    document_identity_word_ids: tuple = ()


_HEADER = {
    'description': r'description|details|service|פירוט|תיאור',
    'quantity': r'quantity|qty\.?|כמות',
    'item_code': r'item(?:\s+code)?|code|sku|מק\s*["״]?\s*ט',
    'unit_price': r'unit\s+price|price|מחיר(?:\s+יחידה)?',
    'line_total': r'line\s+total|total|amount|סה\s*["״]?\s*כ',
}
_KIND = re.compile(r'\binvoice\s*/\s*receipt\b|\b(?:quotation|quote|estimate|invoice|receipt)\b|הצעת\s+מחיר|חשבונית(?:\s+מס)?(?:\s*/?\s*קבלה)?|קבלה', re.I)
_QUALIFICATION = re.compile(r'\b(?:may|except|excluding|excluded|provisional|conditional|subject\s+to|not\s+included|unless|estimated|conditions?|notes?)\b|בכפוף|למעט|לא\s+כולל|אינו\s+כולל|בתנאי|מותנ|זמני|ייתכן|הער(?:ה|ות)|ט\s*[.״"]?\s*ל\s*[.״"]?\s*ח', re.I)
_PAYMENT_HEADER = {
    'amount': r'amount|paid\s+amount|סכום',
    'date': r'date|payment\s+date|תאריך(?:\s+תשלום)?',
    'description': r'description|details|פירוט|תיאור',
    'method': r'(?:payment\s+)?method|אמצעי\s+תשלום',
}
_PAYMENT_TABLE = re.compile(r'\b(?:payment\s+details|(?:recorded|received)\s+payments|payments?(?:\s+received)?)\b|פרטי\s+תשלומים|תשלומים\s+שהתקבלו', re.I)
_PARTY_CONTEXT = re.compile(r'^\s*:?(?:bill\s+to|invoice\s+to|sold\s+to|recipient|addressee|issuer|supplier|vendor|from|לכבוד|נמען|ספק|מאת|מנפיק)\s*(?::|$)', re.I)
_SERVICE_CONTEXT = re.compile(r'^\s*(?:service(?:\s+(?:site|location))?|project|site|work(?:\s+site)?|customer|לקוח(?:ה)?|אתר(?:\s+שירות)?|פרויקט|מקום\s+ביצוע|תיאור\s+עבודה)\s*:', re.I)


def parse_monetary_page(page):
    """Return immutable cell ownership, or None for an unsupported/unsafe page.

    The caller obtains ``page`` with source_access.pdf_source_pages geometry
    enabled. Acceptance must reopen the same source and independently reparse;
    a supplied geometry hash alone is not an authorization or evidence proof.
    """
    if not isinstance(page, dict):
        return None
    words, vectors = page.get('words'), page.get('native_vectors')
    native_blocks = page.get('native_blocks', ())
    source_text, number = page.get('text'), page.get('page')
    if (not isinstance(source_text, str) or not source_text or len(source_text) > 160000
            or type(number) is not int or number < 1 or type(page.get('page_count')) is not int
            or page['page_count'] < number or not isinstance(words, tuple) or not 1 <= len(words) <= 4000
            or not isinstance(vectors, tuple) or len(vectors) > 20000):
        return None
    if (not isinstance(native_blocks, tuple) or len(native_blocks) > 4000
            or any(not isinstance(block, tuple) or len(block) != 7 or not isinstance(block[4], str)
                or any(type(block[i]) not in (int, float) or not math.isfinite(block[i]) for i in range(4))
                or type(block[5]) is not int or block[6] != 0 for block in native_blocks)):
        return None
    if any(not isinstance(w, tuple) or len(w) != 8 or not isinstance(w[4], str)
           or any(type(w[i]) is not int for i in (5, 6, 7))
           or any(type(w[i]) not in (int, float) or not math.isfinite(w[i]) for i in (0, 1, 2, 3))
           or w[0] >= w[2] or w[1] >= w[3] for w in words):
        return None
    if len({(w[5], w[6], w[7]) for w in words}) != len(words):
        return None
    if any(any(char.isnumeric() for char in word[4])
            and unicodedata.normalize('NFKC', word[4]) != unicodedata.normalize('NFC', word[4]) for word in words):
        return None
    if any(not isinstance(v, tuple) or len(v) != 5 or v[0] not in {'rect', 'line'}
           or any(type(x) not in (int, float) or not math.isfinite(x) for x in v[1:])
           or v[1] >= v[3] or (v[0] == 'rect' and v[2] >= v[4]) for v in vectors):
        return None
    tokens = positioned_tokens_from_pymupdf(number, words)

    def text(selected):
        if not selected:
            return ''
        raw = reconstruct_page(tuple(selected), preserve_ltr_runs=True).text
        return ' '.join(canonicalize_source_text(raw, repair_visual_bidi=True,
            normalize_pdf_spacing=True, join_pdf_line_wraps=False).text.split())

    def inside(token, box):
        return box[0] - .05 <= token.x0 and token.x1 <= box[2] + .05 and box[1] - .05 <= token.y0 and token.y1 <= box[3] + .05

    def selected(box):
        touched = tuple(t for t in tokens if t.x0 < box[2] and box[0] < t.x1 and t.y0 < box[3] and box[1] < t.y1)
        return touched if all(inside(t, box) for t in touched) else None

    def ids(selected):
        return tuple((number, t.block, t.line, t.word) for t in selected)

    def money(selected):
        value = text(selected)
        matches = _money_matches(value)
        if len(matches) != 1:
            return None
        match, amount, currency = matches[0]
        remainder = _CURRENCY.sub('', value[:match.start()] + value[match.end():]).strip()
        return NativeMonetaryField(amount, currency, ids(selected)) if not remainder else None

    bands = {}
    for vector in vectors:
        if vector[0] == 'rect':
            bands.setdefault((round(vector[2], 2), round(vector[4], 2)), set()).add(tuple(vector[1:]))
    headers = []
    for group in bands.values():
        boxes = sorted(group)
        if not 4 <= len(boxes) <= 5 or any(abs(a[2] - b[0]) > 1.5 for a, b in zip(boxes, boxes[1:])):
            continue
        labels = [selected(box) for box in boxes]
        if any(not label for label in labels):
            continue
        roles = [next((role for role, pattern in _HEADER.items() if re.fullmatch(pattern, text(label), re.I)), '') for label in labels]
        if (not set(roles) - set(_HEADER) and '' not in roles and len(set(roles)) == len(roles)
                and {'description', 'quantity', 'unit_price', 'line_total'} <= set(roles)):
            headers.append((boxes, roles))
    if len(headers) != 1:
        return None
    boxes, roles = headers[0]
    columns = dict(zip(roles, boxes))
    scalar_indices = sorted((roles.index('unit_price'), roles.index('line_total')))
    if scalar_indices not in ([0, 1], [len(boxes)-2, len(boxes)-1]):
        return None
    scalar_boxes = [boxes[i] for i in scalar_indices]
    header_top, body_top = boxes[0][1], boxes[0][3]
    table_left, table_right = boxes[0][0], boxes[-1][2]

    def matching_rule(y, box):
        return any(v[0] == 'line' and abs(v[2] - y) < .05 and abs(v[4] - y) < .05
            and abs(v[1] - box[0]) <= 1 and abs(v[3] - box[2]) <= 1 for v in vectors)

    ys = sorted({v[2] for v in vectors if v[0] == 'line' and v[2] > body_top})
    footer_tops = []
    for group in bands.values():
        candidate = sorted(group)
        if len(candidate) != 2 or candidate[0][1] <= body_top:
            continue
        if any(abs(a[j] - b[j]) > 1.5 for a, b in zip(candidate, scalar_boxes) for j in (0, 2)):
            continue
        if any(monetary_role(text(selected(box) or ())) == 'payable_total' for box in candidate):
            footer_tops.append(candidate[0][1])
    if len(footer_tops) != 1:
        return None
    body_boundaries = [y for y in ys if all(matching_rule(y, box) for box in boxes)
        or matching_rule(y, (table_left, y, table_right, y))]
    body_boundaries = [y for y in body_boundaries if y < footer_tops[0]]
    if not body_boundaries:
        return None
    items = []
    for bottom in body_boundaries:
        cells = {role: selected((box[0], body_top, box[2], bottom)) for role, box in columns.items()}
        if any(value is None for value in cells.values()):
            return None
        if not cells['description'] and not cells['quantity'] and not cells.get('item_code'):
            break  # A full-width summary rule is not another item row.
        if any(not value for value in cells.values()):
            return None
        unit, total = money(cells['unit_price']), money(cells['line_total'])
        quantity_text = text(cells['quantity'])
        quantity = _decimal_literal(quantity_text)
        if (unit is None or total is None or unit.currency != total.currency
                or quantity is None or quantity <= 0 or not re.fullmatch(r'\d+(?:[.,]\d+)?', quantity_text)):
            return None
        descriptor = cells['description'] + cells.get('item_code', ())
        heights = [t.y1-t.y0 for t in descriptor]
        # Small raised note/identity glyphs cannot be silently folded into IDs.
        if (heights and max(heights) > min(heights) * 1.5
                or any(unicodedata.normalize('NFKC', t.text) != unicodedata.normalize('NFC', t.text)
                    and any(char.isnumeric() for char in t.text) for t in descriptor)
                or any(re.search(r'[*†‡]|[\u2070-\u209f]', t.text) for t in descriptor)):
            return None
        covered = {id(t) for cell in cells.values() for t in cell}
        if any(body_top < t.y1 and t.y0 < bottom and table_left < t.x1 and t.x0 < table_right
                and id(t) not in covered for t in tokens):
            return None
        items.append(NativeMonetaryItem(text(cells['description']), quantity, unit.amount,
            total.amount, total.currency, ids(tuple(t for cell in cells.values() for t in cell)),
            text(cells.get('item_code', ())), ids(cells['description']), ids(cells['quantity']),
            unit.word_ids, total.word_ids))
        body_top = bottom
        if len(items) > 40:
            return None
    if not items:
        return None

    footers = []
    for group in bands.values():
        footer_boxes = sorted(group)
        if len(footer_boxes) != 2 or footer_boxes[0][1] <= body_top:
            continue
        if any(abs(a[j] - b[j]) > 1.5 for a, b in zip(footer_boxes, scalar_boxes) for j in (0, 2)):
            continue
        cells = [selected(box) for box in footer_boxes]
        if any(not cell for cell in cells):
            continue
        values = [money(cell) for cell in cells]
        if sum(value is not None for value in values) != 1:
            continue
        index = next(i for i, value in enumerate(values) if value is not None)
        if monetary_role(text(cells[1-index])) == 'payable_total':
            footers.append((footer_boxes, values[index]))
    if len(footers) != 1:
        return None
    footer_boxes, payable = footers[0]
    footer_top, footer_end = footer_boxes[0][1], footer_boxes[0][3]
    summary_boundaries = [y for y in ys if body_top < y < footer_top
        and all(matching_rule(y, box) for box in scalar_boxes)]
    if len(summary_boundaries) != 1:
        return None
    summary = []
    for top, bottom in zip([body_top] + summary_boundaries, summary_boundaries + [footer_top]):
        cells = [selected((box[0], top, box[2], bottom)) for box in scalar_boxes]
        if any(not cell for cell in cells):
            return None
        values = [money(cell) for cell in cells]
        if sum(value is not None for value in values) != 1:
            return None
        index = next(i for i, value in enumerate(values) if value is not None)
        summary.append((text(cells[1-index]), values[index], ids(cells[1-index])))
    subtotal_label, subtotal, _subtotal_label_ids = summary[0]
    tax_label, tax, vat_label_ids = summary[1]
    if (not re.fullmatch(r'(?:sub\s*total|net\s*(?:amount|total)|total\s+before\s+(?:vat|tax)|סה\s*["״]?\s*כ(?:\s+לפני\s+מע\s*["״]?\s*מ)?)', subtotal_label, re.I)
            or not (monetary_role(tax_label) == 'tax_amount'
                or re.fullmatch(r'(?:VAT|tax|מע\s*["״]?\s*מ)\s*\d+(?:[.,]\d+)?\s*%', tax_label, re.I))):
        return None
    rates = re.findall(r'(?<![\d.])\d+(?:[.,]\d+)?(?=\s*%)', tax_label)
    if len(rates) != 1:
        return None
    vat_rate = _decimal_literal(rates[0])
    currencies = {item.currency for item in items} | {field.currency for field in (subtotal, tax, payable)}
    if len(currencies) != 1 or vat_rate is None or vat_rate > 100:
        return None
    # Summary rows may not hide an additional service outside the scalar cells.
    if any(body_top < t.y1 and t.y0 < footer_end and not any(
            inside(t, (box[0], body_top, box[2], footer_end)) for box in scalar_boxes) for t in tokens):
        return None

    context_tokens = tuple(t for t in tokens if t.y1 <= header_top)
    context_text = text(context_tokens)
    from financial_documents import primary_document_identities
    native_lines = {}
    for token in context_tokens:
        native_lines.setdefault((token.block, token.line), []).append(token)
    identities = set()
    title_lines = set()
    for key, line in native_lines.items():
        line_identities = primary_document_identities(text(line))
        identities.update((identity.kind, identity.number) for identity in line_identities)
        if line_identities:
            title_lines.add(key)
    if len(identities) != 1:
        return None
    document_kind, document_number = next(iter(identities))
    document_kind = {'quote': 'quotation', 'credit': 'credit_note'}.get(document_kind, document_kind)
    # Party names are not service-site ownership. Keep the full context as
    # evidence, but discovery/selection can use only post-title service context.
    # Party ownership continues through adjacent native blocks: PDF extraction
    # block boundaries do not establish a new semantic role. An explicit
    # service/site/project label or primary title establishes the next role.
    title_top = min(t.y0 for key in title_lines for t in native_lines[key])
    service_tokens = []
    party = False
    ordered_lines = sorted(native_lines.items(), key=lambda item:
        (min(t.y0 for t in item[1]), min(t.x0 for t in item[1])))
    for key, line in ordered_lines:
        line_text = text(line)
        if key in title_lines:
            party = False
        elif _PARTY_CONTEXT.search(line_text):
            party = True
        elif _SERVICE_CONTEXT.search(line_text):
            party = False
        if not party and min(t.y0 for t in line) >= title_top - .05:
            service_tokens.extend(line)
    service_context_text = text(tuple(service_tokens))
    dates = set()
    for match in re.finditer(r'(?<!\d)(\d{1,2})/(\d{1,2})/(\d{4})(?!\d)', context_text):
        try:
            dates.add(date(int(match[3]), int(match[2]), int(match[1])))
        except ValueError:
            return None
    if len(dates) > 1:
        return None

    # Payment rows have their own labeled column ownership. Native extraction
    # order may merge dates, bank details and methods from different columns;
    # repeated totals below the table are summaries, never additional payments.
    payments = []
    payment_headers = []
    payment_table_present = False
    for group in bands.values():
        payment_boxes = sorted(group)
        if (not 3 <= len(payment_boxes) <= 4 or payment_boxes[0][1] <= footer_end
                or any(abs(a[2] - b[0]) > 1.5 for a, b in zip(payment_boxes, payment_boxes[1:]))):
            continue
        cells = [selected(box) for box in payment_boxes]
        if any(not cell for cell in cells):
            continue
        payment_roles = [next((role for role, pattern in _PAYMENT_HEADER.items()
            if re.fullmatch(pattern, text(cell), re.I)), '') for cell in cells]
        heading = text(tuple(t for t in tokens
            if footer_end <= t.y0 and t.y1 <= payment_boxes[0][1]))
        if _PAYMENT_TABLE.search(heading):
            payment_table_present = True
        if ('' not in payment_roles and len(set(payment_roles)) == len(payment_roles)
                and {'amount', 'date', 'method'} <= set(payment_roles)
                and _PAYMENT_TABLE.search(heading)):
            payment_headers.append((payment_boxes, payment_roles, heading))
    if len(payment_headers) == 1:
        payment_boxes, payment_roles, heading = payment_headers[0]
        payment_columns = dict(zip(payment_roles, payment_boxes))
        payment_top = payment_boxes[0][3]
        payment_left, payment_right = payment_boxes[0][0], payment_boxes[-1][2]
        summary_tops = []
        for group in bands.values():
            summary_boxes = sorted(group)
            if len(summary_boxes) != 2 or summary_boxes[0][1] <= payment_top:
                continue
            cells = [selected(box) for box in summary_boxes]
            if any(not cell for cell in cells):
                continue
            summary_values = [money(cell) for cell in cells]
            if sum(value is not None for value in summary_values) != 1:
                continue
            label_index = next(i for i, value in enumerate(summary_values) if value is None)
            if re.fullmatch(r'total(?:\s+paid)?|סה\s*["״]?\s*כ', text(cells[label_index]), re.I):
                summary_tops.append(summary_boxes[0][1])
        payment_limit = min(summary_tops, default=float('inf'))
        payment_bottoms = sorted({v[2] for v in vectors if v[0] == 'line'
            and payment_top < v[2] < payment_limit and (
                all(matching_rule(v[2], box) for box in payment_boxes)
                or matching_rule(v[2], (payment_left, v[2], payment_right, v[2])))})
        for bottom in payment_bottoms:
            cells = {role: selected((box[0], payment_top, box[2], bottom))
                for role, box in payment_columns.items()}
            if (any(value is None for value in cells.values())
                    or any(not cells[role] for role in ('amount', 'date', 'method'))):
                payments = []
                break
            payment_amount = money(cells['amount'])
            payment_date = None
            date_text = text(cells['date'])
            match = re.fullmatch(r'(\d{1,2})/(\d{1,2})/(\d{4})', date_text)
            iso = re.fullmatch(r'(\d{4})-(\d{2})-(\d{2})', date_text)
            try:
                if match:
                    payment_date = date(int(match[3]), int(match[2]), int(match[1]))
                elif iso:
                    payment_date = date(int(iso[1]), int(iso[2]), int(iso[3]))
            except ValueError:
                pass
            row_tokens = tuple(t for cell in cells.values() for t in cell)
            covered = {id(t) for t in row_tokens}
            if (payment_amount is None or payment_date is None
                    or any(payment_top < t.y1 and t.y0 < bottom
                        and payment_left < t.x1 and t.x0 < payment_right
                        and id(t) not in covered for t in tokens)):
                payments = []
                break
            spans = []
            for block_id in {t.block for t in row_tokens}:
                matching = [block for block in native_blocks if block[5] == block_id]
                if len(matching) != 1:
                    spans = []
                    break
                block_text = canonicalize_source_text(matching[0][4], repair_visual_bidi=True,
                    normalize_pdf_spacing=True, join_pdf_line_wraps=False).text.strip()
                offset = source_text.find(block_text)
                if not block_text or offset < 0:
                    spans = []
                    break
                spans.append((offset, offset + len(block_text)))
            if not spans:
                payments = []
                break
            payments.append(NativeMonetaryPayment(payment_amount.amount, payment_amount.currency,
                payment_date, text(cells['method']), ids(row_tokens), heading + ' ' + text(row_tokens),
                (min(a for a, _b in spans), max(b for _a, b in spans)),
                ids(cells['amount']),ids(cells['date']),ids(cells['method'])))
            payment_top = bottom
            if len(payments) > 40:
                payments = []
                break

    # Literal qualifications anywhere on the authenticated page remain visible.
    # Footnote markers/raised glyphs are rejected above rather than guessed.
    source_lines = source_text.splitlines(keepends=True)
    offsets, offset = [], 0
    for line in source_lines:
        offsets.append((offset, offset + len(line), line))
        offset += len(line)
    qualification_starts = [a for a, _b, line in offsets if _QUALIFICATION.search(line)]
    # A condition can wrap onto lines that carry no qualifier keyword. Keep
    # the entire remaining literal region, including detached continuations;
    # the renderer may not reduce a condition to its trigger word.
    qualification_spans = []
    if qualification_starts:
        qualification_spans.append((min(qualification_starts), len(source_text)))
    # Unknown wording can also carry a material condition. Preserve every
    # native context block above the header and below the total, including
    # continuations and ordinary metadata, without deciding what prose means. A raw
    # extraction can put these blocks first, so source-order tail slicing
    # alone would silently lose them.
    trailing_ids = {t.block for t in tokens if t.y0 >= footer_end - .05}
    context_ids = {t.block for t in context_tokens}
    note_ids = trailing_ids | context_ids
    if note_ids and not native_blocks:
        return None
    for block_id in note_ids:
        matching = [block for block in native_blocks if block[5] == block_id]
        if len(matching) != 1:
            return None
        block_text = canonicalize_source_text(matching[0][4], repair_visual_bidi=True,
            normalize_pdf_spacing=True, join_pdf_line_wraps=False).text.strip()
        if not block_text:
            return None
        start_at = source_text.find(block_text)
        if start_at < 0:
            return None
        # Repeated literal notes are one displayed note, while every native
        # occurrence remains inside the citation region and geometry proof.
        while start_at >= 0:
            qualification_spans.append((start_at, start_at + len(block_text)))
            start_at = source_text.find(block_text, start_at + len(block_text))
    qualification_spans = sorted(set(qualification_spans))
    # Preserve only literal substrings; overlapping regions remain a single
    # source region rather than repeating or selectively truncating a note.
    merged = []
    for first, last in qualification_spans:
        if merged and first <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(last, merged[-1][1]))
        else:
            merged.append((first, last))
    qualifiers = tuple(dict.fromkeys(source_text[first:last] for first, last in merged))
    header_lines = [(a, b) for a, b, line in offsets if _KIND.search(line)]
    # Native content order can concatenate a value and its following RTL
    # label. Currency tokens still locate the complete literal source line;
    # flattened monetary-token regexes must not truncate the payable citation.
    money_lines = [(a, b) for a, b, line in offsets if _CURRENCY.search(line)]
    start = header_lines[-1][0] if header_lines else 0
    end = max((b for a, b in money_lines), default=len(source_text))
    for a, b, line in offsets:
        if _QUALIFICATION.search(line):
            start, end = min(start, a), len(source_text)
    if merged:
        start, end = min(start, merged[0][0]), max(end, merged[-1][1])
    # Include preceding issue date where declared, without inventing a date.
    previous = [item for item in offsets if item[1] <= start]
    if previous and re.search(r'\d{1,2}/\d{1,2}/\d{4}', previous[-1][2]):
        start = previous[-1][0]
    if start >= end:
        start, end = 0, len(source_text)
    geometry = {'revision': 'monetary-native-cells-v4', 'page': number, 'words': words,
        'vectors': vectors, 'blocks': native_blocks, 'text_sha256': hashlib.sha256(source_text.encode()).hexdigest()}
    signature = hashlib.sha256(json.dumps(geometry, ensure_ascii=False, sort_keys=True,
        separators=(',', ':')).encode()).hexdigest()
    return NativeMonetaryPage(source_text, (start, end), signature, number, document_kind,
        document_number, next(iter(dates)) if dates else None, tuple(items), subtotal,
        tax, payable, vat_rate, qualifiers, context_text, tuple(payments), payment_table_present,
        service_context_text, vat_label_ids,
        ids(tuple(t for key in sorted(title_lines) for t in native_lines[key])))


def native_cell_literal_spans(page, word_ids):
    # Align every native word of each involved block to its exact source
    # characters. Omitted words break spans, so adjacent account fields cannot
    # be incorporated into a field quote.
    if (not isinstance(page,dict) or not isinstance(word_ids,tuple) or not word_ids or
            any(not isinstance(key,tuple) or len(key)!=4 or
                any(type(value) is not int for value in key) for key in word_ids) or
            len(set(word_ids))!=len(word_ids)):
        return ()
    source=page.get('text')
    words=page.get('words',())
    blocks=page.get('native_blocks',())
    page_number=page.get('page')
    if not isinstance(source,str) or not isinstance(words,tuple) or not isinstance(blocks,tuple):
        return ()
    wanted=set(word_ids)
    if any(not isinstance(key,tuple) or len(key)!=4 or key[0]!=page_number or
           any(type(value) is not int for value in key) for key in wanted):
        return ()
    result=[]
    observed=set()
    for block_id in sorted({key[1] for key in wanted}):
        matching=[block for block in blocks if block[5]==block_id]
        if len(matching)!=1:
            return ()
        block_text=canonicalize_source_text(matching[0][4],repair_visual_bidi=True,
            normalize_pdf_spacing=True,join_pdf_line_wraps=False).text.strip()
        if not block_text or source.count(block_text)!=1:
            return ()
        offset=source.find(block_text)
        raw=[(c,i) for i,c in enumerate(block_text) if not c.isspace()]
        block_words=sorted((word for word in words if word[5]==block_id),
                           key=lambda word:(word[6],word[7]))
        expected=[]
        for word in block_words:
            value=canonicalize_source_text(word[4],repair_visual_bidi=True,
                normalize_pdf_spacing=True,join_pdf_line_wraps=False).text
            expected.append(''.join(char for char in value if not char.isspace()))
        if not block_words or any(not value for value in expected):
            return ()
        if ''.join(char for char,_ in raw) != ''.join(expected):
            return ()
        cursor=0
        active=None
        for word,value in zip(block_words,expected):
            key=(page_number,word[5],word[6],word[7])
            first=offset+raw[cursor][1]
            cursor+=len(value)
            last=offset+raw[cursor-1][1]+1
            if key in wanted:
                observed.add(key)
                if active is None:
                    active=[first,last]
                else:
                    active[1]=last
            elif active is not None:
                result.append(tuple(active))
                active=None
        if active is not None:
            result.append(tuple(active))
    if observed != wanted:
        return ()
    return tuple(sorted(result))


def recover_monetary_pdf(metadata, document_root, tenant, pages=(1,)):
    from source_access import pdf_source_pages
    return tuple(record for page in pdf_source_pages(document_root, tenant, metadata, pages,
        include_geometry=True) if (record := parse_monetary_page(page)) is not None)


def native_rtl_numeric_context(page, literal):
    """Return a separately citable, one-line visual reconstruction, or None.

    This deliberately handles only a whole unique source block composed of
    Hebrew words and one standalone number with optional plus/minus.  A
    visually ambiguous line is not interpreted by this helper.
    """
    if not isinstance(page, dict) or not isinstance(literal, str) or not literal.strip():
        return None
    text = page.get('text')
    blocks = page.get('native_blocks', ())
    words = page.get('words', ())
    number = page.get('page')
    if not isinstance(text, str) or text.count(literal) != 1:
        return None
    candidates = []
    for block in blocks:
        block_text = canonicalize_source_text(block[4], repair_visual_bidi=True,
            normalize_pdf_spacing=True, join_pdf_line_wraps=False).text.strip()
        if block_text == literal.strip():
            candidates.append(block[5])
    if len(candidates) != 1:
        return None
    block_id = candidates[0]
    selected = [word for word in words if word[5] == block_id]
    if not selected or len({word[6] for word in selected}) != 1:
        return None
    if any(not re.fullmatch(r'[֐-׿]+|[0-9]+|[+\-]', str(w[4])) for w in selected):
        return None
    hebrew = [w for w in selected if re.fullmatch(r'[֐-׿]+', w[4])]
    numeric = [w for w in selected if re.fullmatch(r'[0-9]+', w[4])]
    operators = [w for w in selected if w[4] in {'+', '-'}]
    if len(hebrew) < 2 or len(numeric) != 1 or len(operators) > 1 or len(selected) > 16:
        return None
    visual = sorted(selected, key=lambda w: -w[0])
    if operators:
        operator=operators[0]
        number_word=numeric[0]
        if abs(visual.index(operator)-visual.index(number_word)) != 1:
            return None
        gap=max(operator[0]-number_word[2],number_word[0]-operator[2],0)
        if gap > 2*max(operator[3]-operator[1],number_word[3]-number_word[1]):
            return None
    if any(visual[i][0] < visual[i+1][2] for i in range(len(visual)-1)):
        return None
    if max(w[1] for w in selected)-min(w[1] for w in selected) > 2:
        return None
    reconstructed = reconstruct_page(positioned_tokens_from_pymupdf(number, selected),
                                     preserve_ltr_runs=False)
    if len(reconstructed.lines) != 1 or reconstructed.lines[0].removed_boundaries:
        return None
    reading = ' '.join(w[4] for w in visual)
    if reconstructed.text != reading or reading == ' '.join(literal.split()):
        return None
    # The entire block must align exactly with authenticated source characters.
    all_ids = tuple((number, w[5], w[6], w[7]) for w in selected)
    full = native_cell_literal_spans(page, all_ids)
    if len(full) != 1 or text[full[0][0]:full[0][1]].strip() != literal.strip():
        return None
    # Source-order runs are separate literal quotes; no reconstructed text is
    # ever represented as a contiguous verbatim citation.
    runs = []
    ordered = sorted(selected, key=lambda w: (w[6], w[7]))
    # The existing quote gate cannot faithfully retain a leading sign in a
    # standalone excerpt. Preserve the price and explicitly limit the term.
    if ordered[0][4] in {'+', '-'}:
        return None
    for word in ordered:
        kind = 'he' if word in hebrew else 'numeric'
        if not runs or runs[-1][0] != kind:
            runs.append((kind, []))
        runs[-1][1].append((number, word[5], word[6], word[7]))
    if len(runs) != 2:
        return None
    spans = []
    for _kind, ids in runs:
        found = native_cell_literal_spans(page, tuple(ids))
        if len(found) != 1:
            return None
        spans.extend(found)
    return reading, tuple(spans)
