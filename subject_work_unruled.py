"""Conservative source-backed verification for literal unruled PDF items."""
import re
import unicodedata

from canonical_text import canonicalize_source_text
from source_access import pdf_source_pages, SourceAccessError


def _tokens(text):
    text = unicodedata.normalize('NFKC', str(text)).casefold()
    # Numeric tokens are indivisible: 12 30 must never become 1 230.
    parts = re.findall(r'\d+(?:[,.]\d+)*|[^\W\d_]+|_|[^\w\s]', text, re.UNICODE)
    result = []
    i = 0
    while i < len(parts):
        if (i + 1 < len(parts) and parts[i] in '₪$€£'
                and re.fullmatch(r'\d+(?:[,.]\d+)*', parts[i+1])):
            result.append('money:' + parts[i+1] + ':' + parts[i]); i += 2
        elif (i + 1 < len(parts) and re.fullmatch(r'\d+(?:[,.]\d+)*', parts[i])
              and parts[i+1] in '₪$€£'):
            result.append('money:' + parts[i] + ':' + parts[i+1]); i += 2
        else:
            result.append(parts[i]); i += 1
    return tuple(result)


def _same_tokens(left, right):
    a, b = _tokens(left), _tokens(right)
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            i += 1; j += 1
        # Hebrew clitics can split in PDF text; this never merges numbers.
        elif (i+1 < len(a) and a[i] in {'ב','כ','ל','מ','ו','ה','ש'}
              and re.fullmatch(r'[\u05d0-\u05ea]+', a[i+1]) and a[i]+a[i+1] == b[j]):
            i += 2; j += 1
        elif (j+1 < len(b) and b[j] in {'ב','כ','ל','מ','ו','ה','ש'}
              and re.fullmatch(r'[\u05d0-\u05ea]+', b[j+1]) and b[j]+b[j+1] == a[i]):
            i += 1; j += 2
        else:
            return False
    return i == len(a) and j == len(b)


def verify_unruled_passages(evidence, tenant, document_root):
    grouped = {}
    for row in evidence.witness.rows:
        grouped.setdefault(row['metadata']['source_id'], []).append(row)
    relevant = [p for p in evidence.passages
                if p.metadata.get('mime_type') == 'application/pdf'
                and p.value != 'subject_document_identity']
    if not relevant:
        return True
    cache = {}
    for passage in relevant:
        source = passage.source_id
        page = passage.metadata.get('page')
        rows = grouped.get(source, ())
        if not rows or type(page) is not int:
            return False
        if (source, page) not in cache:
            try:
                pages = pdf_source_pages(document_root, tenant, rows[0]['metadata'], (page,),
                                         include_geometry=True)
            except (OSError, ValueError, SourceAccessError):
                return False
            if len(pages) != 1:
                return False
            payload = pages[0]
            bands = {}
            for v in payload.get('native_vectors', ()):
                if v[0] == 'rect':
                    bands.setdefault((round(v[2], 1), round(v[4], 1)), []).append(v)
            for cells in bands.values():
                cells.sort(key=lambda x:x[1])
                if len(cells) >= 3 and sum(abs(a[3]-b[1]) <= 1.5 for a,b in zip(cells,cells[1:])) >= 2:
                    return False
            cache[(source, page)] = payload
        payload = cache[(source, page)]
        wanted = passage.source_claim.strip()
        if not wanted:
            return False
        blocks = []
        for block in payload.get('native_blocks', ()):
            text = canonicalize_source_text(block[4], repair_visual_bidi=True,
                normalize_pdf_spacing=True, join_pdf_line_wraps=False).text
            if _same_tokens(text, wanted):
                blocks.append((block, text))
        if len(blocks) != 1:
            return False
        block, text = blocks[0]
        if passage.value != 'subject_material_condition':
            # Numeric item descriptions are safe to preserve only as complete
            # labeled source prose, not as an inferred unlabeled column.
            if re.search(r'\d', text):
                # One explicit terminal price field is retained verbatim; it
                # cannot occur inside the equipment description.
                flat = ' '.join(text.split())
                price = re.search(
                    r'(?<!\w)(?:עלות|מחיר|price|cost)\s*[:–—-]?\s*'
                    r'(?:[₪$€£]\s*)?\d[\d,.]*\s*(?:[₪$€£])?'
                    r'(?:\s*(?:ליחידה|per\s+(?:unit|item)))?\s*[.!]?\s*$',
                    flat, re.I)
                if not price:
                    return False
                prefix = flat[:price.start()]
                for scalar in re.finditer(r'(?<![\w])\d[\d,.]*(?![\w])', prefix):
                    before, after = prefix[:scalar.start()], prefix[scalar.end():]
                    # A source can state model numbers and dimensions without
                    # turning them into prices. Other free-standing values
                    # have unknown ownership and must not be interposed.
                    if not (re.search(r'(?:\bmodel|דגם)\s*[:–—-]?\s*$', before, re.I)
                            or re.match(r'\s*(?:mm|cm|met(?:er|re)s?|feet|ft|מטר|סנטימטר|מ\"מ)(?!\w)', after, re.I)):
                        return False
            words = [w for w in payload.get('words', ()) if w[5] == block[5]]
            if not words:
                return False
            height = max(w[3]-w[1] for w in words)
            if max(w[1] for w in words)-min(w[1] for w in words) > height * 1.5:
                return False
            # A source extraction block can contain multiple visual columns.
            # Require a continuous physical line at the source's own font scale
            # instead of assuming that adjacent extracted words are neighbors.
            ordered = sorted(words, key=lambda w: w[0])
            if any(right[0]-left[2] > height * 1.5
                   for left,right in zip(ordered, ordered[1:])):
                return False
    return True
