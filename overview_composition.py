"""Reversible presentation of independently reviewed overview topics.

This helper establishes no semantic support: the shared assessor and original
source gate validate each atomic claim and its own citation after expansion.
"""
from __future__ import annotations

import re

_HE_CLAIM = re.compile(r"המסמך עוסק ב([א-ת]+)\.((?: \[\d{1,3}\])+)$")
_EN_CLAIM = re.compile(r"The document concerns ([A-Za-z]+)\.((?: \[\d{1,3}\])+)$")
_HE_RENDER = re.compile(r"המסמך עוסק ב([א-ת]+)((?: \[\d{1,3}\])+)((?:, ב[א-ת]+(?: \[\d{1,3}\])+)* וב[א-ת]+(?: \[\d{1,3}\])+)\.$")
_EN_RENDER = re.compile(r"The document concerns ([A-Za-z]+)((?: \[\d{1,3}\])+)((?:, [A-Za-z]+(?: \[\d{1,3}\])+)* and [A-Za-z]+(?: \[\d{1,3}\])+)\.$")


def expand_compound(part, evidence):
    """Recover the exact atomic claims represented by the fixed grammar."""
    if not getattr(evidence, 'overview_raw_topic_draft', ''):
        return None
    match = _HE_RENDER.fullmatch(part)
    if match:
        pairs = [(match.group(1), match.group(2))]
        pairs.extend(re.findall(r'(?:, ב| וב)([א-ת]+)((?: \[\d{1,3}\])+)', match.group(3)))
        claims = tuple('המסמך עוסק ב' + word + '.' + markers for word, markers in pairs)
    else:
        match = _EN_RENDER.fullmatch(part)
        if not match:
            return None
        pairs = [(match.group(1), match.group(2))]
        pairs.extend(re.findall(r'(?:, | and )([A-Za-z]+)((?: \[\d{1,3}\])+)', match.group(3)))
        claims = tuple('The document concerns ' + word + '.' + markers for word, markers in pairs)
    if not 2 <= len(claims) <= 3 or len(set(claims)) != len(claims):
        return None
    return claims


def _compose(claims, hebrew):
    pattern = _HE_CLAIM if hebrew else _EN_CLAIM
    parts = [pattern.fullmatch(claim) for claim in claims]
    if any(part is None for part in parts):
        return None
    items = [part.group(1) + part.group(2) for part in parts]
    if hebrew:
        return 'המסמך עוסק ב' + items[0] + ''.join(', ב' + item for item in items[1:-1]) + ' וב' + items[-1] + '.'
    return 'The document concerns ' + items[0] + ''.join(', ' + item for item in items[1:-1]) + ' and ' + items[-1] + '.'


def compose_tagged_topics(claims, evidence, hebrew, assess):
    """Compose atomic topics only if both renderings have identical bindings."""
    original = '\n\n'.join(claims)
    if not getattr(evidence, 'overview_raw_topic_draft', '') or len(claims) < 2:
        return original
    before = assess(original, evidence, hebrew, check_coverage=False)
    if not before.accepted:
        return original
    blocks = []
    index = 0
    pattern = _HE_CLAIM if hebrew else _EN_CLAIM
    while index < len(claims):
        end = index
        while end < len(claims) and end - index < 3 and pattern.fullmatch(claims[end]):
            end += 1
        block = _compose(claims[index:end], hebrew) if end - index >= 2 else None
        if block and expand_compound(block, evidence) == tuple(claims[index:end]):
            blocks.append(block)
            index = end
        else:
            blocks.append(claims[index])
            index += 1
    candidate = '\n\n'.join(blocks)
    after = assess(candidate, evidence, hebrew, check_coverage=False)
    return candidate if after.accepted and after.bindings == before.bindings else original
