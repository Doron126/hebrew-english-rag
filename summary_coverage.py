"""Conservative, source-derived sufficient proof of whole-summary coverage.

This does not decide factual support. The caller must first validate source
identity, original bytes, the released claims and their citation bindings.
Exhaustive literal source-unit coverage is sufficient for completeness; its
absence is not proof that a more abstract summary is semantically incomplete.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
import unicodedata

_WORD = re.compile(r'\ufff8[0-9a-f]{6}\ufff9|[+-]?\d+(?:[.,]\d+)*|'
                   r'[^\W\d_]+(?:[\w/-]*[\w])?|[%₪$€°<>≤≥=()]', re.UNICODE)
_MARKER = re.compile(r'\[(\d{1,3})\]')
# A narrowly recognized conventional invocation, not a class of acronyms.
_PREFATORY_INVOCATION = re.compile(r'^בס["״\']ד[.!]?$')


@dataclass(frozen=True)
class SummaryCoverage:
    complete: bool
    reason: str
    source_units: int
    covered_units: int
    source_digest: str
    answer_digest: str


def _words(value):
    text = unicodedata.normalize('NFKC', str(value or '')).casefold()
    # Preserve arbitrary numeric operators and separators, while ignoring
    # ordinary list/heading punctuation. Canonicalize equivalent minus/range
    # dashes; retain multiplication, dates, signs and every other symbol.
    text = text.replace('\N{MINUS SIGN}', '-')
    def operator(match):
        value = match.group('internal') or match.group('prefix') or match.group('suffix')
        value = '-' if value in '-–—' else value
        return ' \ufff8' + format(ord(value), '06x') + '\ufff9 '
    numeric_operator = (r'(?<=\d)\s*(?P<internal>[^\w\s.,]|[xX])\s*(?=\d)|'
                        r'(?<![\w])(?P<prefix>[^\w\s.,])(?=\d)|'
                        r'(?<=\d)(?P<suffix>[^\w\s.,])(?=\s|$)')
    text = re.sub(numeric_operator, operator, text)
    # Symbols may themselves encode status, including checkmarks, crosses,
    # arrows and mathematical signs. Do not treat them as decorative merely
    # because they are not adjacent to a number.
    text = ''.join((' \ufff8' + format(ord(char), '06x') + '\ufff9 ')
                   if unicodedata.category(char).startswith('S') else char for char in text)
    return tuple(_WORD.findall(text))


def _digest(value):
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def _source_lines(evidence):
    """Yield source lines with their literal citation and stable order."""
    result = []
    for fact in evidence.passages:
        for line in fact.source_claim.splitlines():
            line = line.strip()
            if line and _words(line):
                result.append((fact.citation_id, line))
    # This invocation is non-content only as the isolated first original
    # paragraph with following source text. It is not removed in a heading,
    # table row, later location, or any other context.
    if result and evidence.passages and evidence.passages[0].span[0] == 0:
        first = evidence.passages[0]
        marker = result[0][1]
        raw = str(first.text or '')
        if (_PREFATORY_INVOCATION.fullmatch(marker)
                and re.match(r'^\s*'+re.escape(marker)+r'[ \t]*\r?\n[ \t]*\r?\n(?=\S)', raw)):
            result.pop(0)
    return tuple(result)


def _cited_claims(answer):
    from chat_api import _split_grounded_claim_segments
    result = []
    for segment in _split_grounded_claim_segments(answer):
        ids = tuple(dict.fromkeys(_MARKER.findall(segment)))
        words = _words(_MARKER.sub('', segment))
        if ids and words:
            result.append((ids, words))
    return tuple(result)


def assess_summary_coverage(evidence, answer):
    """Recompute an exhaustive-source witness from the released answer.

    Claims must already have passed independent factual and citation gates.
    Every material line must occur as an ordered token span in an actually
    released claim carrying that passage's citation. Matches are consumed so
    one repeated output cannot prove multiple repeated source units.
    """
    from document_reading import SourceSummaryEvidence, passage_valid, unqualified_whole_summary
    answer = str(answer or '')
    answer_hash = _digest(answer)
    if (not isinstance(evidence, SourceSummaryEvidence)
            or not unqualified_whole_summary(evidence.reference, evidence.summary_request)
            or not evidence.complete or not evidence.selection_complete
            or not evidence.passages or not all(passage_valid(p) for p in evidence.passages)):
        return SummaryCoverage(False, 'incomplete_source', 0, 0, '', answer_hash)
    ids = tuple(p.citation_id for p in evidence.passages)
    eligible = evidence.quote_eligible_ids
    if (len(ids) != len(set(ids)) or not isinstance(eligible, tuple)
            or len(eligible) != len(ids) or len(eligible) != len(set(eligible))
            or set(eligible) != set(ids)):
        return SummaryCoverage(False, 'unavailable_source_quote', 0, 0, '', answer_hash)
    lines = _source_lines(evidence)
    source_hash = _digest('\n'.join(p.source_id+'|'+p.source_hash+'|'+str(p.span)+'|'+p.span_hash
                                    for p in evidence.passages))
    if not lines:
        return SummaryCoverage(False, 'no_source_units', 0, 0, source_hash, answer_hash)
    claims = _cited_claims(answer)
    used = [set() for _ in claims]
    covered = 0
    for citation_id, line in lines:
        required = _words(line)
        found = False
        for index, (claim_ids, words) in enumerate(claims):
            if citation_id not in claim_ids or len(required) > len(words):
                continue
            for start in range(len(words)-len(required)+1):
                positions = set(range(start, start+len(required)))
                if not positions.intersection(used[index]) and words[start:start+len(required)] == required:
                    used[index].update(positions)
                    covered += 1
                    found = True
                    break
            if found:
                break
    return SummaryCoverage(covered == len(lines), 'exhaustive_source_units' if covered == len(lines)
                           else 'unproved_source_coverage', len(lines), covered, source_hash, answer_hash)
