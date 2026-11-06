"""Bound local-model formatting to the existing literal summary interface.

This module offers source-owned output choices, not evidence authorization or
answer acceptance. Every selected claim still passes the existing summary
parser, source reconstruction, citation checks and final factual assessor.
"""
from __future__ import annotations

import json
import re

from document_reading import passage_valid, summary_sentence_boundaries
from passage_synthesis import is_whole_named_source_summary


MAX_SUMMARY_CLAIMS = 16
MAX_SOURCE_CHOICES = 256
MAX_CHOICE_CHARS = 1200
MAX_SCHEMA_BYTES = 192 * 1024

LITERAL_SUMMARY_INSTRUCTION = (
    "Select a concise set of main facts from across the complete source. "
    "When the source has enough material, select at least four distinct facts "
    "covering its different main topics before adding detail about one topic. "
    "Return the required claims array, with one supplied citation ID and one "
    "exact source sentence or complete bullet per item. Put separate source "
    "sentences in separate items, even when they share a citation ID. Do not "
    "combine sentences, paraphrase, translate, or change source punctuation "
    "or internal whitespace. Preserve complete conditions and exceptions. "
    "Choose only facts that answer the question; do not repeat a selected item. "
    "Use at most 16 claims."
)


def literal_summary_response_schema(evidence):
    """Build a bounded grammar over exact, separately cited source units.

    Only the ordinary whole named-source summary envelope is supported. The
    caller must opt into this helper for the portable runtime after obtaining
    quote eligibility. Oversized or malformed evidence raises ValueError;
    source choices are never truncated to fit and no fallback is generated.
    """
    if (getattr(evidence, "overview_synthesis", False)
            or not is_whole_named_source_summary(evidence, require_complete=False)):
        raise ValueError("local_summary_requires_literal_named_source")
    passages = tuple(evidence.passages)
    eligible = evidence.quote_eligible_ids
    identifiers = tuple(passage.citation_id for passage in passages)
    if (not isinstance(eligible, tuple) or not eligible
            or any(not isinstance(ref, str) or not re.fullmatch(r"\d{1,3}", ref)
                   for ref in identifiers)
            or len(set(identifiers)) != len(identifiers)
            or any(not isinstance(ref, str) or ref not in identifiers for ref in eligible)
            or len(set(eligible)) != len(eligible)):
        raise ValueError("local_summary_invalid_eligible_sources")
    selected = [passage for passage in passages if passage.citation_id in eligible]
    if len(selected) > MAX_SOURCE_CHOICES:
        raise ValueError("local_summary_source_choice_limit_exceeded")
    if any(not passage_valid(passage) for passage in selected):
        raise ValueError("local_summary_invalid_source_span")
    owners = {(passage.tenant_id, passage.source_id, passage.source_hash,
               passage.metadata.get("source_relative_path")) for passage in selected}
    if len(owners) != 1 or any(not value for value in next(iter(owners))):
        raise ValueError("local_summary_mixed_source_ownership")

    alternatives = []
    total_choices = 0
    for passage in selected:
        source = passage.source_claim
        # Reuse the exact sentence/bullet boundaries used to build and split
        # whole-source summary evidence. Trimming edge whitespace retains a
        # contiguous original substring; internal bytes are never rewritten.
        boundaries = [0, *summary_sentence_boundaries(source), len(source)]
        choices = list(dict.fromkeys(source[left:right].strip()
            for left, right in zip(boundaries, boundaries[1:])
            if source[left:right].strip()))
        if not choices or any(not re.search(r"\w", value) for value in choices):
            raise ValueError("local_summary_has_no_source_sentences")
        if any(len(value) > MAX_CHOICE_CHARS for value in choices):
            raise ValueError("local_summary_source_sentence_limit_exceeded")
        total_choices += len(choices)
        if total_choices > MAX_SOURCE_CHOICES:
            raise ValueError("local_summary_source_choice_limit_exceeded")
        alternatives.append({
            "type": "array", "minItems": 2, "maxItems": 2,
            "items": [{"const": passage.citation_id},
                      {"type": "string", "enum": choices}],
            "additionalItems": False,
        })
    schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {"claims": {
            "type": "array", "minItems": min(4, total_choices), "maxItems": MAX_SUMMARY_CLAIMS,
            "items": {"oneOf": alternatives},
        }},
        "required": ["claims"], "additionalProperties": False,
    }
    if len(json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode("utf-8")) > MAX_SCHEMA_BYTES:
        raise ValueError("local_summary_schema_byte_limit_exceeded")
    return schema
