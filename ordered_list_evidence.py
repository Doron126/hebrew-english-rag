"""Source-owned ordered-list grouping and consumer checks.

Printed ordinals establish membership. Request terms only select an already
bounded source list; they never supply labels, order, or governing context.
"""
from __future__ import annotations

import hashlib
import json
import re

from document_query import _document_identity
from structural_evidence import (
    StructuralEvidenceType, canonical_structural_entry, ordered_list_item_text,
    word_ordinal_list_spans, source_ordered_list_owner_binding,
)


def _dimension(requirement, name):
    from semantic_contracts import requirement_dimension_value
    return requirement_dimension_value(requirement, name)


def heading_topic_matches(wanted, observed):
    """Match one literal heading phrase with bounded article agreement.

    Hebrew initial ה can belong to a proper name. A single article difference
    is therefore supported only by the ordinary construct role מפת דרך
    (roadmap). Otherwise permit only a consecutive, same-direction definite
    noun/adjective pair with visible feminine agreement. Unknown constructions
    abstain; neither proper-name letters nor word order disappear.
    """
    wanted_role, wanted = _document_identity(wanted)
    observed_role, observed = _document_identity(observed)
    if wanted_role != observed_role:
        return False
    if wanted[:1] == ("the",):
        wanted = wanted[1:]
    if observed[:1] == ("the",):
        observed = observed[1:]
    if not wanted:
        return False
    for start in range(len(observed) - len(wanted) + 1):
        window = observed[start:start + len(wanted)]
        differences = [index for index, (left, right) in enumerate(zip(wanted, window)) if left != right]
        if not differences:
            return True
        bases = []
        sides = []
        for index in differences:
            left, right = wanted[index], window[index]
            if not re.fullmatch(r"[\u05d0-\u05ea]+", left + right):
                break
            if left.startswith("ה") and left[1:] == right:
                bases.append(right)
                sides.append("wanted")
            elif right.startswith("ה") and right[1:] == left:
                bases.append(left)
                sides.append("observed")
            else:
                break
        else:
            if len(differences) == 1:
                index = differences[0]
                if index > 0 and wanted[index - 1] == window[index - 1] == "מפת" and bases == ["דרך"]:
                    return True
            elif (len(set(sides)) == 1
                  and differences == list(range(differences[0], differences[-1] + 1))
                  and bases[0].endswith("ה")
                  and all(base.endswith(("ית", "יות")) for base in bases[1:])):
                return True
    return False


def _terms_present(wanted, observed):
    return heading_topic_matches(wanted, observed)


def _span_matches(raw, span):
    return bool(span is not None and type(span.start) is int and type(span.end) is int
                and 0 <= span.start < span.end <= len(raw)
                and hashlib.sha256(raw[span.start:span.end].encode()).hexdigest() == span.sha256)


def ordered_list_source_errors(structural, raw, *, source_id="", candidate_id=""):
    if structural is None or structural.structural_type is not StructuralEvidenceType.LIST_ITEM:
        return ("ordered_list_source_unit_missing",)
    if (type(structural.list_ordinal) is not int or structural.list_ordinal < 1
            or not structural.list_id or not structural.item_id
            or not structural.list_heading or not ordered_list_item_text(structural)):
        return ("ordered_list_identity_missing",)
    if source_id and structural.source_id != source_id or candidate_id and structural.candidate_id != candidate_id:
        return ("ordered_list_source_identity_mismatch",)
    printed = re.fullmatch(r"(?:\((\d{1,3})\)|(\d{1,3})[.)]?)(?:[ \t]+\S.*)?",
                           structural.local_text.splitlines()[0])
    word_sequence = None
    if printed is None:
        word_sequence = next((
            (heading_start, heading_end, entries)
            for heading_start, heading_end, entries in word_ordinal_list_spans(raw)
            if any(start == structural.source_spans[0].start
                   and end == structural.source_spans[0].end
                   and ordinal == structural.list_ordinal
                   for ordinal, start, end, _label_start, _label_end in entries)
        ), None) if len(structural.source_spans) == 1 else None
    if ((printed is None and word_sequence is None)
            or printed is not None and int(printed.group(1) or printed.group(2)) != structural.list_ordinal):
        return ("ordered_list_printed_ordinal_mismatch",)
    spans = structural.source_spans
    heading = structural.list_heading_span
    governing = structural.list_context_span
    if (len(spans) != 1 or not _span_matches(raw, spans[0])
            or raw[spans[0].start:spans[0].end] != structural.local_text
            or not _span_matches(raw, heading) or not _span_matches(raw, governing)
            or raw[heading.start:heading.end] != structural.list_heading
            or not heading.start == governing.start < heading.end <= governing.end <= spans[0].start):
        return ("ordered_list_source_context_unproved",)
    # Every item binds the same pre-list context ending at the first printed
    # marker. Previous item labels can never lend the requested topic.
    if word_sequence is not None:
        # Re-enter the original candidate, independently of the claimed list
        # identity/heading/item. A literal ordinal alone is insufficient.
        reconstructed = canonical_structural_entry(
            raw, source_id=structural.source_id, candidate_id=structural.candidate_id,
            chunk_id=structural.chunk_id, page=structural.page,
            metadata={"slide": structural.slide} if structural.slide is not None else {},
        )
        fields = ("unit_id", "list_id", "item_id", "list_ordinal", "local_text",
                  "source_spans", "list_heading", "list_heading_span", "list_context_span")
        if not any(all(getattr(unit, name) == getattr(structural, name) for name in fields)
                   for unit in reconstructed.structural_units):
            return ("ordered_list_source_sequence_unproved",)
    else:
        first_line = raw[governing.end:].splitlines()[0]
        if not re.fullmatch(r"(?:1[.)]?|\(1\))(?:[ \t]+\S.*)?", first_line):
            return ("ordered_list_first_marker_unproved",)
    if re.search(r"(?mi)^\s*\[[^\]\r\n]+\s*>\s*(?:slide|שקופית)\s+\d+\]\s*$",
                 raw[governing.start:governing.end]):
        return ("ordered_list_cross_slide_context",)
    return ()


def source_ordered_list_owner(structural, raw):
    binding = source_ordered_list_owner_binding(structural, raw) if structural is not None else None
    return binding[0] if binding is not None else ""


def ordered_list_owner_errors(requirement, evidence):
    """Match a requested actor to the actual source organization subject."""
    wanted = _dimension(requirement, "ordered_list_owner")
    context = getattr(evidence, "evidence_context", None)
    structural = (getattr(evidence, "structural_evidence_unit", None)
                  or getattr(context, "structural_evidence_unit", None))
    raw_hit = getattr(evidence, "raw_hit", None)
    raw = str(raw_hit.get("document") or "") if raw_hit is not None else str(
        getattr(context, "source_candidate_text", "") or "")
    observed = source_ordered_list_owner(structural, raw)
    # Legacy numeric lists remain governed by their existing heading/topic
    # contract. A textual organization list must have an explicit actor slot.
    if not wanted and not observed:
        return ()
    if ordered_list_source_errors(structural, raw):
        return ("ordered_list_owner_source_unproved",)
    if not wanted:
        return ("ordered_list_owner_required",)
    if not observed:
        return ("ordered_list_owner_unproved",)
    key = lambda value: " ".join(value.split())
    unresolved = {"it", "he", "she", "they", "this", "that", "הוא", "היא", "הם", "הן", "זה", "זו", "אלה"}
    if any(key(value).casefold() in unresolved for value in (wanted, observed)):
        return ("ordered_list_owner_unproved",)
    scope = _dimension(requirement, "document_scope")
    from document_query import document_fact_errors
    if scope and document_fact_errors(scope, evidence):
        return ("ordered_list_document_scope_incompatible",)
    # Preserve every material owner token, case and identifier punctuation.
    if key(wanted) == key(observed):
        return ()
    from semantic_contracts import generic_document_reference, document_role_target_compatible
    if scope and generic_document_reference(wanted) and generic_document_reference(observed):
        try:
            declared = json.loads(scope)
        except (ValueError, TypeError):
            declared = None
        if (isinstance(declared, dict) and declared.get("schema") == "project-x-named-document-v1"
                and not document_fact_errors(scope, evidence)
                and document_role_target_compatible(wanted, observed)):
            return ()
    return ("ordered_list_owner_incompatible",)


def ordered_list_fact_errors(requirement, fact):
    heading = _dimension(requirement, "ordered_list_heading")
    if not heading:
        return ()
    context = getattr(fact, "evidence_context", None)
    structural = getattr(context, "structural_evidence_unit", None)
    raw = str(getattr(context, "source_candidate_text", "") or "")
    errors = list(ordered_list_source_errors(
        structural, raw, source_id=str(getattr(context, "source_id", "") or ""),
        candidate_id=str(getattr(context, "candidate_id", "") or "")))
    if errors:
        return tuple(errors)
    errors.extend(ordered_list_owner_errors(requirement, fact))
    if not getattr(fact, "canonical_valid", False) or not getattr(context, "structural_binding_valid", False):
        errors.append("ordered_list_fact_binding_invalid")
    if str(getattr(fact, "object_text", "") or "") != ordered_list_item_text(structural):
        errors.append("ordered_list_literal_item_mismatch")
    if not _terms_present(heading, structural.list_heading):
        errors.append("ordered_list_heading_incompatible")
    topic = _dimension(requirement, "ordered_list_topic")
    span = structural.list_context_span
    if topic and not _terms_present(topic, raw[span.start:span.end]):
        errors.append("ordered_list_topic_incompatible")
    return tuple(errors)


def ordered_list_group_key(structural):
    return (structural.source_id, structural.candidate_id, structural.chunk_id,
            structural.list_id, structural.slide)


def eligible_ordered_list_groups(requirement, units):
    """Yield complete source lists, never a ranked combination of partial lists."""
    heading = _dimension(requirement, "ordered_list_heading")
    if not heading or requirement.kind != "list":
        return ()
    groups = {}
    for unit in units:
        structural = unit.structural_evidence_unit
        raw = str(unit.raw_hit.get("document") or "")
        if ordered_list_source_errors(structural, raw, source_id=unit.source_id, candidate_id=unit.candidate_id):
            continue
        if ordered_list_owner_errors(requirement, unit):
            continue
        groups.setdefault(ordered_list_group_key(structural), []).append(unit)
    eligible = []
    for group in groups.values():
        group.sort(key=lambda unit: unit.structural_evidence_unit.list_ordinal)
        if ([unit.structural_evidence_unit.list_ordinal for unit in group]
                != list(range(1, requirement.cardinality + 1))):
            continue
        first = group[0].structural_evidence_unit
        raw = str(group[0].raw_hit.get("document") or "")
        if (len({(unit.structural_evidence_unit.list_heading_span,
                  unit.structural_evidence_unit.list_context_span) for unit in group}) != 1
                or not _terms_present(heading, first.list_heading)):
            continue
        topic = _dimension(requirement, "ordered_list_topic")
        span = first.list_context_span
        if topic and not _terms_present(topic, raw[span.start:span.end]):
            continue
        eligible.append(tuple(group))
    return tuple(eligible)


def rendered_ordered_list_errors(answer, linked_facts, requirements):
    """Require all linked literal labels once, in their printed source order."""
    errors = []
    answer = str(answer or "")
    for requirement in requirements:
        if not _dimension(requirement, "ordered_list_heading"):
            continue
        facts = [fact for fact in linked_facts if requirement.id in getattr(fact, "requirement_ids", ())]
        if len(facts) != requirement.cardinality or any(ordered_list_fact_errors(requirement, fact) for fact in facts):
            errors.append("rendered_ordered_list_membership_unproved")
            continue
        structures = [fact.evidence_context.structural_evidence_unit for fact in facts]
        if (len({ordered_list_group_key(unit) for unit in structures}) != 1
                or sorted(unit.list_ordinal for unit in structures) != list(range(1, requirement.cardinality + 1))):
            errors.append("rendered_ordered_list_source_sequence_incomplete")
            continue
        ordered = sorted(facts, key=lambda fact: fact.evidence_context.structural_evidence_unit.list_ordinal)
        printed_items = list(re.finditer(
            r"(?m)^[ \t]*(?:[-*][ \t]+)?(?:\*\*)?(?:\((?P<parenthesized>\d{1,3})\)|(?P<ordinal>\d{1,3})[.)])(?:\*\*)?[ \t]+(?P<label>[^\n]+)",
            answer))
        if printed_items:
            if [int(match.group("parenthesized") or match.group("ordinal")) for match in printed_items] != list(range(1, requirement.cardinality + 1)):
                errors.append("rendered_ordered_list_cardinality_incompatible")
                continue
            labels = [re.sub(r"\s+", " ", match.group("label")).casefold() for match in printed_items]
            if any(not re.search(r"(?<!\w)" + re.escape(re.sub(r"\s+", " ", fact.object_text).casefold()) + r"(?!\w)", label)
                   for fact, label in zip(ordered, labels)):
                errors.append("rendered_ordered_list_order_incompatible")
            continue
        normalized = re.sub(r"\s+", " ", answer).casefold()
        positions = []
        for fact in ordered:
            label = re.sub(r"\s+", " ", fact.object_text).casefold()
            matches = list(re.finditer(r"(?<!\w)" + re.escape(label) + r"(?!\w)", normalized))
            if len(matches) != 1:
                errors.append("rendered_ordered_list_item_missing_or_repeated")
            else:
                positions.append(matches[0].start())
        if _dimension(requirement, "ordered_list_order") == "source_order" and positions != sorted(positions):
            errors.append("rendered_ordered_list_order_incompatible")
    return tuple(dict.fromkeys(errors))
