"""Resolve a topic-qualified presentation without turning it into a filename.

The index supplies candidate text, while scan_authorized owns tenant and source
integrity.  The same predicate is used again when accepting the evidence.
"""
from __future__ import annotations

import re
import unicodedata


def presentation_summary_reference(query):
    from document_query import DocumentReference
    value = str(query or '').strip()
    pattern = (
        r'^(?:(?:אפשר|נא|please)\s+)?(?:תסכם|תסכמ(?:י|ו)|סכמ(?:י|ו)|סכם|לסכם|תמצת|summari[sz]e)\s+'
        r'(?:(?:לי|me)\s+)?(?:(?:את|the)\s+)?'
        r'(?:ה?מצגת|presentation|slide\s+deck)\s+'
        r'(?:על|בנושא|אודות|לגבי|שעוסקת\s+ב|העוסקת\s+ב|שמדברת\s+על|about|on)\s*(?P<topic>[^?;\n]+?)\s*[?.!]*$'
    )
    match = re.fullmatch(pattern, value, re.I)
    if not match:
        return None
    topic = match['topic'].strip(' ,:')
    if not topic or len(topic) > 160:
        return None
    return DocumentReference('presentation_topic', topic)


def _tokens(value):
    value = unicodedata.normalize('NFKC', str(value or '')).casefold()
    return re.findall(r'[a-z0-9]+|[\u05d0-\u05ea]+', value)


def _forms(word):
    result = {word}
    if len(word) > 4 and word[0] in 'והבלמכש':
        result.add(word[1:])
    return result


def presentation_topic_matches(reference, rows):
    if reference.kind != 'presentation_topic' or not rows:
        return False
    meta = rows[0].get('metadata') or {}
    if meta.get('mime_type') not in {
        'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        'application/vnd.ms-powerpoint',
        'application/vnd.oasis.opendocument.presentation',
    }:
        return False
    wanted = _tokens(reference.title)
    if not wanted:
        return False
    # Rows are already grouped under one authenticated source identity. Topic
    # words may be split across slides/chunks or appear in its indexed title;
    # never combine terms from separate documents to manufacture a match.
    observed = set(_tokens(meta.get('title') or ''))
    for row in rows:
        observed.update(_tokens(row.get('document') or ''))
    forms = set().union(*(_forms(word) for word in observed)) if observed else set()
    return all(_forms(word) & forms for word in wanted)


def resolve_presentation_topic(retriever, tenant, document_root, reference, where=None):
    from document_reading import scan_authorized
    grouped, complete, _ = scan_authorized(retriever, tenant, document_root, where=where)
    if not complete:
        return 'incomplete', ()
    matches = tuple(rows[0]['metadata'] for rows in grouped.values()
                    if presentation_topic_matches(reference, rows))
    return ('found' if len(matches) == 1 else 'ambiguous' if matches else 'missing'), matches
