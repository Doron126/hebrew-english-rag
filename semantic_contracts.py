"""Route-independent semantic contracts, evidence compatibility, and gates.

The runtime has several internal routing vocabularies.  This module is the
shared contract below those vocabularies: requests, evidence, and answer
claims are compared through typed slots and required output, never by route
enum identity.  It intentionally contains no tenant, document, page,
benchmark-question, or expected-answer constants.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import re
import unicodedata
from enum import Enum
from typing import Any, Iterable, Mapping, Optional, Sequence

from answer_units import GROUPED_DECIMAL_LITERAL_PATTERN

from evidence_context import (
    RELATION_ARGUMENT_SCHEMAS,
    RelationArgument,
    relation_argument_validation_errors,
    structural_binding_validation_errors,
)


MAX_CONTRACT_TEXT = 480
MAX_CONTRACT_ITEMS = 64
_TEXTUAL_EVIDENCE_VALUE_TYPES = frozenset({"text_proposition", "list"})
DOCUMENT_ROLE_PREDICATES = frozenset({"funded_by", "published_by", "authored_by"})


def document_role_query(value: str) -> tuple[str, str]:
    """Parse an explicit role question; output actors are not target entities."""
    patterns = (
        ("funded_by", r"(?:who\s+(?:funded|financed|sponsored)|מי\s+מימ(?:ן|נה|נו))\s+(?:את\s+)?(?P<target>.+)"),
        ("published_by", r"(?:who\s+published|מי\s+פרס(?:ם|מה|מו))\s+(?:את\s+)?(?P<target>.+)"),
        ("authored_by", r"(?:who\s+(?:wrote|authored)|מי\s+כתב(?:ה|ו)?)\s+(?:את\s+)?(?P<target>.+)"),
    )
    text = normalize_semantic_text(value).strip(" ,;:.?!")
    for predicate, pattern in patterns:
        match = re.fullmatch(pattern, text, re.I)
        if match:
            return predicate, match.group("target").strip()
    passive = re.fullmatch(r"by\s+whom\s+was\s+(?P<target>.+?)\s+(?P<verb>funded|financed|published|written|authored)", text, re.I)
    if passive:
        return {"funded":"funded_by","financed":"funded_by","published":"published_by",
                "written":"authored_by","authored":"authored_by"}[passive.group("verb").lower()], passive.group("target")
    return "", ""


def document_role_target_compatible(required: str, observed: str) -> bool:
    """Resolve grammatical articles/deictics while retaining the target name."""
    def key(value: str) -> str:
        value = normalize_semantic_text(value).casefold().strip(" ,;:.?!")
        value = re.sub(r"^(?:the|a|an|this|that)\s+|^את\s+", "", value)
        value = re.sub(r"\s+(?:זה|זו)$", "", value)
        words = value.split()
        if words and len(words[0]) >= 4 and re.fullmatch(r"ה[\u05d0-\u05ea]+", words[0]):
            words[0] = words[0][1:]
        return " ".join(words)
    return bool(key(required) and key(required) == key(observed))


def generic_document_reference(value: str) -> bool:
    """Closed grammatical document references, without proper-name aliases."""
    return bool(re.fullmatch(
        r"(?:(?:(?:the|a|an|this|that)\s+)?(?:document|report|study|file|presentation)|"
        r"ה?(?:מסמך|דוח|מחקר|קובץ|מצגת)(?:\s+(?:זה|זו))?)",
        normalize_semantic_text(value).strip(" ,;:.?!"), re.I,
    ))


def scoped_document_target_compatible(
    requirement: Any, fact: Any, required: str, observed: str,
) -> bool:
    """Resolve only generic purpose targets under a proved named source.

    This supplies grammatical reference agreement, never a source fact or a
    proper-name alias. All other role, proposition and provenance checks stay
    with the existing caller.
    """
    if (getattr(requirement, "kind", "") not in {"purpose", "intended_use"}
            or getattr(fact, "evidence_predicate", "") != "purpose_of"):
        return False
    if not all(generic_document_reference(value) for value in (required, observed)):
        return False
    scope = requirement_dimension_value(requirement, "document_scope")
    if not scope:
        return False
    try:
        declared = json.loads(scope)
    except (ValueError, TypeError):
        return False
    if not isinstance(declared, dict) or declared.get("schema") != "project-x-named-document-v1":
        return False
    from document_query import document_fact_errors
    if document_fact_errors(scope, fact):
        return False
    return document_role_target_compatible(required, observed)


@dataclass(frozen=True)
class DocumentRoleProposition:
    """One complete asserted relationship, independent of a request slot."""

    predicate: str
    actor: str
    target: str
    polarity: str = "positive"


_DOCUMENT_ROLE_VERBS = {
    "funded_by": (r"funded|financed|sponsored", r"funded|financed|sponsored",
                  r"מומן|מומנה|מומנו", r"מימן|מימנה|מימנו"),
    "published_by": (r"published|issued", r"published|issued",
                     r"פורסם|פורסמה|פורסמו", r"פרסם|פרסמה|פרסמו"),
    "authored_by": (r"written|authored", r"wrote|authored",
                   r"נכתב|נכתבה|נכתבו", r"כתב|כתבה|כתבו"),
}
_DOCUMENT_ROLE_NOUNS = {
    "funded_by": r"funder|financier|sponsor",
    "published_by": r"publisher",
    "authored_by": r"author|writer",
}


def _document_role_body(value: str) -> str:
    value = normalize_semantic_text(_CLAIM_CITATION_RE.sub(" ", value or ""))
    return re.sub(r"^[•▪◦*\-]\s*", "", value).strip(" .;:?!")


def _document_role_single(value: str) -> Optional[DocumentRoleProposition]:
    """Parse only a complete bounded proposition, including all modifiers."""
    body = _document_role_body(value)
    # Residual subordinate clauses and scope/polarity modifiers must be
    # parsed separately. A conjunction inside an exact source-owned entity
    # is permitted; additional role predicates are rejected below, and an
    # unknown suffix cannot match the admitted actor/target identity.
    if re.search(r"[,;()\n]|\b(?:but|who|which|that|not|never|only|partly|"
                 r"partially|allegedly|apparently|possibly|exclusively)\b|"
                 r"(?:^|\s)(?:לא|רק|אולי|בלבד|חלקית)(?:\s|$)", body, re.I):
        return None
    for predicate, (passive, active, passive_he, active_he) in _DOCUMENT_ROLE_VERBS.items():
        patterns = [
            rf"(?P<target>.+?)\s+(?:was|is|were|are|has\s+been|had\s+been)\s+(?:{passive})\s+by\s+(?P<actor>.+)",
            rf"(?P<actor>.+?)\s+(?:{active})\s+(?P<target>.+)",
            rf"(?P<target>.+?)\s+(?:{passive_he})\s+(?:על\s+ידי|בידי)\s+(?P<actor>.+)",
            rf"(?P<actor>.+?)\s+(?:{active_he})\s+את\s+(?P<target>.+)",
            rf"(?P<actor>.+?)\s+(?:is|was)\s+(?:the\s+)?(?:{_DOCUMENT_ROLE_NOUNS[predicate]})\s+of\s+(?P<target>.+)",
        ]
        if predicate == "funded_by":
            patterns += [
                r"(?P<target>.+?)\s+(?:בוצע|בוצעה|נערך|נערכה)\s+במימון\s+(?P<actor>.+)",
                r"(?P<actor>.+?)\s+(?:provided|supplied)\s+(?:the\s+)?funding\s+for\s+(?P<target>.+)",
                r"(?:the\s+)?funding\s+for\s+(?P<target>.+?)\s+(?:was|is)\s+(?:provided|supplied)\s+by\s+(?P<actor>.+)",
                r"(?P<target>.+?)\s+received\s+funding\s+from\s+(?P<actor>.+)",
                r"(?:המימון|מימון)\s+(?:של|עבור|ל)\s*(?P<target>.+?)\s+(?:ניתן|סופק)\s+(?:על\s+ידי|בידי)\s+(?P<actor>.+)",
            ]
        for pattern in patterns:
            match = re.fullmatch(pattern, body, re.I)
            if match:
                actor = normalize_semantic_text(match.group("actor")).strip(" .;:?!")
                target = normalize_semantic_text(match.group("target")).strip(" .;:?!")
                # A second verbal predicate inside an argument is an
                # incomplete parse, even without punctuation/conjunction.
                verb_tokens = "|".join(v for variants in _DOCUMENT_ROLE_VERBS.values() for v in variants)
                if actor and target and not re.search(rf"\b(?:{verb_tokens})\b", actor + " " + target, re.I):
                    return DocumentRoleProposition(predicate, actor, target)
    return None



_DOCUMENT_ROLE_INCOMPLETE_ERRORS = frozenset(
    "document_role_assertion_incomplete_" + predicate
    for predicate in DOCUMENT_ROLE_PREDICATES
)


def _incomplete_document_role_predicate(value: str) -> Optional[str]:
    """Recognize role syntax lacking a proved target, never a source fact.

    These diagnostics activate rejection; no target or actor relationship is
    recovered from the request. A role noun in a nonassertive object, such as
    funding gaps, is not a proposition.
    """
    body = _document_role_body(value)
    for predicate, (passive, active, passive_he, active_he) in _DOCUMENT_ROLE_VERBS.items():
        patterns = (
            rf"(?:(?:was|is|were|are|has\s+been|had\s+been)\s+)?(?:{passive})\s+by\s+.+",
            rf"(?:{passive_he})\s+(?:על\s+ידי|בידי)\s+.+",
            rf"(?:(?:its|their|his|her|the)\s+)?(?:{_DOCUMENT_ROLE_NOUNS[predicate]})\s+(?:is|was|are|were)\s+.+",
            rf".+?\s+(?:{active})",
            rf".+?\s+(?:{active_he})",
        )
        if any(re.fullmatch(pattern, body, re.I) for pattern in patterns):
            return predicate
    return None


def parse_document_role_propositions(value: str) -> tuple[tuple[DocumentRoleProposition, ...], tuple[str, ...]]:
    """Positively parse every clause; unknown/partial assertion is failure.

    The bounded grammar supports voice changes and source-backed role
    paraphrases.  A successful prefix never licenses an unparsed suffix.
    No actor, target or role is supplied from the question.
    """
    body = _document_role_body(value)
    if not body:
        return (), ("document_role_assertion_empty",)
    clauses = tuple(item.strip() for item in re.split(r"[.!?׃;](?:\s+|$)|\n+", body) if item.strip())
    propositions: list[DocumentRoleProposition] = []
    parse_errors: list[str] = []
    for clause in clauses:
        parsed = _document_role_single(clause)
        if parsed:
            propositions.append(parsed)
            continue
        # Shared-subject coordinated predicates: both get independently
        # checked against source facts.  This is grammar, not support.
        active_verbs = "|".join(variants[1] for variants in _DOCUMENT_ROLE_VERBS.values())
        coordinated = re.fullmatch(
            rf"(?P<actor>.+?)\s+(?P<verbs>(?:{active_verbs})(?:\s+and\s+(?:{active_verbs}))+?)\s+(?P<target>.+)", clause, re.I,
        )
        expansions: list[str] = []
        if coordinated:
            expansions = [f"{coordinated.group('actor')} {verb} {coordinated.group('target')}"
                          for verb in re.split(r"\s+and\s+", coordinated.group("verbs"), flags=re.I)]
        else:
            # Full coordinated clauses, with an explicitly grammatical
            # shared actor when the second clause starts with a role verb.
            parts = re.split(r"\s+and\s+|\s+ו(?=(?:מימן|פרסם|כתב)\s)", clause, flags=re.I)
            if len(parts) > 1:
                first = _document_role_single(parts[0])
                if first:
                    expansions = [parts[0]]
                    for part in parts[1:]:
                        if re.match(rf"(?:{active_verbs})\s+", part, re.I):
                            part = f"{first.actor} {part}"
                        expansions.append(part)
                else:
                    # A non-role prefix must not hide a later material role
                    # assertion. Preserve positive parses for activation;
                    # the unrecognized clause still makes a role-context
                    # answer incomplete and cannot acquire source support.
                    expansions = parts
            # Explicit actor relative/appositional role clauses.
            relative = re.fullmatch(r"(?P<actor>.+?),\s*(?:who|which)\s+(?P<extra>.+?),\s*(?P<main>.+)", clause, re.I)
            if relative:
                expansions = [f"{relative.group('actor')} {relative.group('extra')}",
                              f"{relative.group('actor')} {relative.group('main')}"]
            apposition = re.fullmatch(r"(?P<actor>.+?),\s*(?:the\s+)?(?P<role>funder|financier|sponsor|publisher|author|writer)\s+of\s+(?P<target>.+?),\s*(?P<main>.+)", clause, re.I)
            if apposition:
                expansions = [f"{apposition.group('actor')} is the {apposition.group('role')} of {apposition.group('target')}",
                              f"{apposition.group('actor')} {apposition.group('main')}"]
        if not expansions:
            incomplete_predicate = _incomplete_document_role_predicate(clause)
            parse_errors.append("document_role_assertion_incomplete_" + incomplete_predicate
                                if incomplete_predicate else "document_role_assertion_unrecognized_or_incomplete")
        for part in expansions:
            # Coordination may contain a relative/appositional assertion.
            # Reuse the same whole-clause grammar, retaining each asserted
            # role and every incomplete-fragment error at arbitrary nesting.
            nested_propositions, nested_errors = parse_document_role_propositions(part)
            propositions.extend(nested_propositions)
            parse_errors.extend(nested_errors)
    if not propositions:
        parse_errors.append("document_role_assertion_unrecognized_or_incomplete")
    return tuple(propositions), tuple(dict.fromkeys(parse_errors))


def document_role_proposition_compatible(asserted: DocumentRoleProposition, source: DocumentRoleProposition) -> bool:
    """Exact actor identity and source-owned target/role/polarity agreement."""
    return bool(asserted.predicate == source.predicate
        and normalize_semantic_text(asserted.actor).casefold() == normalize_semantic_text(source.actor).casefold()
        and document_role_target_compatible(asserted.target, source.target)
        and asserted.polarity == source.polarity)


def document_role_arguments(value: str, predicate: str) -> dict[str, str]:
    """Source construction and rendered support share one role grammar."""
    propositions, errors = parse_document_role_propositions(value)
    if errors or len(propositions) != 1 or propositions[0].predicate != predicate:
        return {}
    proposition = propositions[0]
    target_role, actor_role = RELATION_ARGUMENT_SCHEMAS[predicate]
    return {target_role: proposition.target, actor_role: proposition.actor}


def rendered_document_role_errors(
    answer: str,
    typed_facts: Sequence[Any],
    requirements: Sequence[Any],
    *,
    cited_evidence_ids: Optional[Iterable[str]] = None,
    allow_actor_only: bool = True,
) -> tuple[str, ...]:
    """Bind the actual rendered propositions to admitted cited source roles.

    Actor-only answers use only the unambiguous admitted request context.
    Every other answer must positively state supported actor/target/role;
    absence of a recognized conflicting predicate is never proof.
    """
    requirement_map = {item.id: item for item in requirements}
    role_requirements = tuple(item for item in requirements if item.predicate in DOCUMENT_ROLE_PREDICATES)
    role_facts = tuple(fact for fact in typed_facts
        if str(getattr(fact, "evidence_predicate", "")) in DOCUMENT_ROLE_PREDICATES)
    if not role_requirements and not role_facts:
        propositions, parse_errors = parse_document_role_propositions(answer)
        requires_role_support = bool(propositions or _DOCUMENT_ROLE_INCOMPLETE_ERRORS.intersection(parse_errors))
        return (("rendered_document_role_admitted_support_missing",) if requires_role_support else ())
    cited = None if cited_evidence_ids is None else set(cited_evidence_ids)
    admitted = []
    for fact in role_facts:
        if cited is not None and getattr(fact, "evidence_id", "") not in cited:
            continue
        requirement = requirement_map.get(getattr(fact, "requirement_id", ""))
        if (requirement is None or text_fact_binding_errors(requirement, fact)
                or document_role_source_errors(fact)):
            continue
        predicate = str(getattr(fact, "evidence_predicate", ""))
        arguments = {item.role: item.value for item in getattr(fact, "relation_arguments", ())}
        target_role, actor_role = RELATION_ARGUMENT_SCHEMAS[predicate]
        admitted.append(DocumentRoleProposition(predicate, arguments[actor_role], arguments[target_role]))
    if not admitted:
        return ("rendered_document_role_admitted_support_missing",)
    body = _document_role_body(answer)
    if allow_actor_only and role_requirements and len({(item.predicate, normalize_semantic_text(item.target).casefold()) for item in admitted}) == 1:
        if any(body.casefold() == normalize_semantic_text(item.actor).casefold() for item in admitted):
            return ()
    propositions, errors = parse_document_role_propositions(body)
    if errors:
        return errors
    for proposition in propositions:
        if not any(document_role_proposition_compatible(proposition, source) for source in admitted):
            return ("rendered_document_role_not_supported",)
    return ()


def document_role_actor_answer_supported(answer: str, typed_facts: Sequence[Any], requirements: Sequence[Any]) -> bool:
    """Recognize only an exact, proved concise actor in explicit role context."""
    propositions, _errors = parse_document_role_propositions(answer)
    return bool(not propositions
        and any(item.predicate in DOCUMENT_ROLE_PREDICATES for item in requirements)
        and not rendered_document_role_errors(answer, typed_facts, requirements))


class RequirementSatisfactionMode(str, Enum):
    """The only ways a canonical requirement may become covered.

    This is intentionally independent of request routes.  In particular, a
    direction requested by a comparison is not a proposition that should be
    searched for in a source span: it is derived only after its endpoint facts
    are independently validated.
    """

    SOURCE_FACT = "SOURCE_FACT"
    DERIVED_FACT = "DERIVED_FACT"
    REQUEST_CONDITION = "REQUEST_CONDITION"
    COMPOSITE_REQUIREMENT = "COMPOSITE_REQUIREMENT"


class RequirementConstraintState(str, Enum):
    """Whether one request dimension constrains a source fact.

    This is kept on ``TypedRequirement`` rather than in a compatibility-side
    adapter: compilation is the one authority that decides whether a request
    semantic is evidence-bearing, scope-only, or unresolved.
    """

    REQUIRED = "REQUIRED"
    NOT_REQUIRED = "NOT_REQUIRED"
    SCOPE_ONLY = "SCOPE_ONLY"
    UNKNOWN = "UNKNOWN"


class RequirementConstraintOrigin(str, Enum):
    """Auditable provenance for a compiled requirement dimension."""

    EXPLICIT_REQUEST_SEMANTIC = "EXPLICIT_REQUEST_SEMANTIC"
    OPERATION_REQUIRED = "OPERATION_REQUIRED"
    OUTPUT_TYPE_REQUIRED = "OUTPUT_TYPE_REQUIRED"
    DISCOURSE_SCOPE = "DISCOURSE_SCOPE"
    DERIVED_BUT_PROVEN = "DERIVED_BUT_PROVEN"
    UNJUSTIFIED = "UNJUSTIFIED"
    UNKNOWN = "UNKNOWN"


class EnforcementEquivalenceStatus(str, Enum):
    """Whether a non-independent hard constraint has a complete proof."""

    SELF_ENFORCED = "SELF_ENFORCED"
    DISCRIMINATIVE_EQUIVALENCE_PROVEN = "DISCRIMINATIVE_EQUIVALENCE_PROVEN"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNPROVEN = "UNPROVEN"


_REQUIREMENT_CONSTRAINT_DIMENSIONS = (
    "value",
    "quantity_kind",
    "statistical_role",
    "aggregation_role",
    "subject",
    "metric",
    "predicate",
    "period",
    "population",
    "facet",
    "unit",
    "polarity",
    "target_entity",
    "relation_arguments",
    "fact_role",
    "structural_type",
    "proposition_completeness",
    "financial_role",
    "document_scope",
    "source_location",
    "attribute_context",
    "ordered_list_heading",
    "ordered_list_topic",
    "ordered_list_owner",
    "ordered_list_order",
)

_SEMANTIC_CATEGORY_BY_DIMENSION = {
    "financial_role": "METRIC_OWNED",
    "document_scope": "DOCUMENT_IDENTITY_OWNED",
    "source_location": "STRUCTURAL_LOCATION_OWNED",
    "attribute_context": "SOURCE_CONTEXT_OWNED",
    "ordered_list_heading": "LIST_IDENTITY_OWNED",
    "ordered_list_topic": "LIST_CONTEXT_OWNED",
    "ordered_list_owner": "CORE_SUBJECT_IDENTITY",
    "ordered_list_order": "OUTPUT_SHAPE_ONLY",
    "subject": "CORE_SUBJECT_IDENTITY",
    "predicate": "PREDICATE_OWNED",
    "target_entity": "TARGET_OWNED",
    "relation_arguments": "RELATION_ARGUMENT_OWNED",
    "metric": "METRIC_OWNED",
    "population": "POPULATION_OWNED",
    "facet": "FACET_OWNED",
    "period": "PERIOD_OWNED",
    "aggregation_role": "OUTPUT_SHAPE_ONLY",
    "quantity_kind": "OUTPUT_SHAPE_ONLY",
    "statistical_role": "OUTPUT_SHAPE_ONLY",
    "value": "OUTPUT_SHAPE_ONLY",
    "fact_role": "COMPOSITIONALLY_REQUIRED",
    "structural_type": "COMPOSITIONALLY_REQUIRED",
    "proposition_completeness": "COMPOSITIONALLY_REQUIRED",
}


@dataclass(frozen=True)
class RequirementDimensionConstraint:
    """One compiler-owned semantic constraint with request provenance."""

    dimension: str
    state: RequirementConstraintState | str
    value: str = ""
    origin: RequirementConstraintOrigin | str = RequirementConstraintOrigin.UNKNOWN
    request_span: str = ""
    rule: str = ""
    justification: str = ""
    # C23 keeps semantic accounting separate from the equality constraint.  A
    # meaningful request distinction can be enforced by another canonical
    # dimension without becoming a duplicate subject equality.
    semantic_owner: str = ""
    hard_constraint_owner: str = ""
    enforcement_path: str = ""
    enforcement_equivalence_status: EnforcementEquivalenceStatus | str = (
        EnforcementEquivalenceStatus.NOT_APPLICABLE
    )

    def __post_init__(self) -> None:
        if self.dimension not in _REQUIREMENT_CONSTRAINT_DIMENSIONS:
            raise ValueError("unknown requirement constraint dimension")
        state = RequirementConstraintState(self.state)
        origin = RequirementConstraintOrigin(self.origin)
        equivalence = EnforcementEquivalenceStatus(
            self.enforcement_equivalence_status
        )
        value = str(self.value or "")
        if state is RequirementConstraintState.REQUIRED:
            if not value:
                raise ValueError("required constraint is missing a value")
            if origin in {
                RequirementConstraintOrigin.UNJUSTIFIED,
                RequirementConstraintOrigin.UNKNOWN,
            }:
                raise ValueError("required constraint lacks semantic provenance")
            if not (self.request_span or self.rule or self.justification):
                raise ValueError("required constraint lacks justification")
        object.__setattr__(self, "state", state)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "semantic_owner", self.semantic_owner or self.dimension)
        object.__setattr__(
            self, "hard_constraint_owner",
            self.hard_constraint_owner or (
                self.dimension if state is RequirementConstraintState.REQUIRED else ""
            ),
        )
        object.__setattr__(
            self, "enforcement_path",
            self.enforcement_path or (
                self.dimension if state is RequirementConstraintState.REQUIRED else "none"
            ),
        )
        object.__setattr__(
            self, "enforcement_equivalence_status",
            equivalence if equivalence is not EnforcementEquivalenceStatus.NOT_APPLICABLE
            else (
                EnforcementEquivalenceStatus.SELF_ENFORCED
                if state is RequirementConstraintState.REQUIRED
                else EnforcementEquivalenceStatus.NOT_APPLICABLE
            ),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "dimension": self.dimension,
            "state": self.state.value,
            "value": self.value,
            "origin": self.origin.value,
            "request_span": self.request_span,
            "rule": self.rule,
            "justification": self.justification,
            "semantic_owner": self.semantic_owner,
            "hard_constraint_owner": self.hard_constraint_owner,
            "enforcement_path": self.enforcement_path,
            "enforcement_equivalence_status": self.enforcement_equivalence_status.value,
        }


@dataclass(frozen=True)
class RequirementSemanticAccounting:
    """One request distinction and its single canonical enforcement path.

    ``dimension_constraints`` say what currently compares against a source
    fact.  This record additionally preserves meaningful semantics that have
    a proven, discriminatorily equivalent owner elsewhere.  It is compiler
    output only; source facts never receive request-derived semantics.
    """

    semantic_distinction: str
    semantic_category: str
    request_span: str
    semantic_owner: str
    hard_constraint_owner: str
    enforcement_path: str
    enforcement_equivalence_status: EnforcementEquivalenceStatus | str
    independently_meaningful: bool = True
    independently_discriminative: bool = True
    proof: str = ""

    def __post_init__(self) -> None:
        status = EnforcementEquivalenceStatus(
            self.enforcement_equivalence_status
        )
        if not self.semantic_distinction or not self.semantic_owner:
            raise ValueError("semantic accounting requires distinction and owner")
        if not self.enforcement_path:
            raise ValueError("semantic accounting requires enforcement path")
        if (
            not self.hard_constraint_owner
            and status is not EnforcementEquivalenceStatus.DISCRIMINATIVE_EQUIVALENCE_PROVEN
        ):
            raise ValueError("non-independent semantic accounting lacks equivalence proof")
        object.__setattr__(self, "enforcement_equivalence_status", status)

    def as_dict(self) -> dict[str, Any]:
        return {
            "semantic_distinction": self.semantic_distinction,
            "semantic_category": self.semantic_category,
            "request_span": self.request_span,
            "semantic_owner": self.semantic_owner,
            "hard_constraint_owner": self.hard_constraint_owner,
            "enforcement_path": self.enforcement_path,
            "enforcement_equivalence_status": self.enforcement_equivalence_status.value,
            "independently_meaningful": self.independently_meaningful,
            "independently_discriminative": self.independently_discriminative,
            "proof": self.proof,
        }


class ScalarSemanticRole(str, Enum):
    """The source-established semantic job of one numeric literal.

    ``numeric_role`` predates the fact boundary and is retained in a few
    presentation adapters as a coarse unit/shape hint.  This vocabulary is
    the canonical authority for whether a numeric source fact can satisfy a
    numeric request.  It is deliberately independent of request routes and
    never derives a source role from a requirement.
    """

    POINT_ESTIMATE = "point_estimate"
    LOWER_BOUND = "lower_bound"
    UPPER_BOUND = "upper_bound"
    CONFIDENCE_LEVEL = "confidence_level"
    FILTER_THRESHOLD = "filter_threshold"
    TOTAL_COUNT = "total_count"
    CATEGORY_COUNT = "category_count"
    PERCENTAGE_SHARE = "percentage_share"
    RANGE_BOUND = "range_bound"
    PERIOD_VALUE = "period_value"
    NOT_APPLICABLE = "not_applicable"


class QuantityKind(str, Enum):
    """The literal shape of a source quantity, independent of its meaning."""

    COUNT = "count"
    PERCENTAGE = "percentage"
    NUMBER = "number"
    PERIOD = "period"
    UNSPECIFIED = "unspecified"


class StatisticalRole(str, Enum):
    """The statistical job established by source-local wording for one value."""

    POINT_ESTIMATE = "point_estimate"
    LOWER_BOUND = "lower_bound"
    UPPER_BOUND = "upper_bound"
    CONFIDENCE_LEVEL = "confidence_level"
    FILTER_THRESHOLD = "filter_threshold"
    RANGE_BOUND = "range_bound"
    OBSERVED_VALUE = "observed_value"
    UNSPECIFIED = "unspecified"


class AggregationRole(str, Enum):
    """The aggregation job established by the same source-local proposition."""

    TOTAL = "total"
    CATEGORY = "category"
    SHARE = "share"
    NONE = "none"
    UNSPECIFIED = "unspecified"


@dataclass(frozen=True)
class QuantitySemantics:
    """Orthogonal, source-bound semantics for one scalar literal.

    ``scalar_role`` remains a derived compatibility projection for historical
    adapters.  It must never be used as the authority for new source-fact or
    requirement compatibility decisions.
    """

    quantity_kind: str = QuantityKind.UNSPECIFIED.value
    statistical_role: str = StatisticalRole.UNSPECIFIED.value
    aggregation_role: str = AggregationRole.UNSPECIFIED.value
    statistical_role_cue_span: Optional[tuple[int, int]] = None
    aggregation_role_cue_span: Optional[tuple[int, int]] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "quantity_kind", normalize_quantity_kind(
            self.quantity_kind
        ))
        object.__setattr__(self, "statistical_role", normalize_statistical_role(
            self.statistical_role
        ))
        object.__setattr__(self, "aggregation_role", normalize_aggregation_role(
            self.aggregation_role
        ))

    @property
    def scalar_role(self) -> str:
        return scalar_role_from_quantity_semantics(
            self.quantity_kind, self.statistical_role, self.aggregation_role,
        )


_SCALAR_ROLE_ALIASES = {
    "": ScalarSemanticRole.NOT_APPLICABLE.value,
    "requested_answer_value": ScalarSemanticRole.POINT_ESTIMATE.value,
    "count": ScalarSemanticRole.POINT_ESTIMATE.value,
    "category_value": ScalarSemanticRole.CATEGORY_COUNT.value,
    "percentage": ScalarSemanticRole.PERCENTAGE_SHARE.value,
    "period": ScalarSemanticRole.PERIOD_VALUE.value,
    "table_header": ScalarSemanticRole.NOT_APPLICABLE.value,
    "unrelated_neighbor": ScalarSemanticRole.NOT_APPLICABLE.value,
}

_QUANTITY_KIND_ALIASES = {
    "": QuantityKind.UNSPECIFIED.value,
    "numeric": QuantityKind.NUMBER.value,
    "requested_answer_value": QuantityKind.NUMBER.value,
    "category_value": QuantityKind.COUNT.value,
    "percentage_share": QuantityKind.PERCENTAGE.value,
    "percent": QuantityKind.PERCENTAGE.value,
}
_STATISTICAL_ROLE_ALIASES = {
    "": StatisticalRole.UNSPECIFIED.value,
    "not_applicable": StatisticalRole.UNSPECIFIED.value,
    "period_value": StatisticalRole.UNSPECIFIED.value,
}
_AGGREGATION_ROLE_ALIASES = {
    "": AggregationRole.UNSPECIFIED.value,
    "not_applicable": AggregationRole.UNSPECIFIED.value,
    "percentage_share": AggregationRole.SHARE.value,
    "total_count": AggregationRole.TOTAL.value,
    "category_count": AggregationRole.CATEGORY.value,
}


def normalize_quantity_kind(value: str | QuantityKind | None) -> str:
    raw = str(getattr(value, "value", value) or "").strip().casefold()
    if raw in {item.value for item in QuantityKind}:
        return raw
    return _QUANTITY_KIND_ALIASES.get(raw, QuantityKind.UNSPECIFIED.value)


def normalize_statistical_role(value: str | StatisticalRole | None) -> str:
    raw = str(getattr(value, "value", value) or "").strip().casefold()
    if raw in {item.value for item in StatisticalRole}:
        return raw
    return _STATISTICAL_ROLE_ALIASES.get(raw, StatisticalRole.UNSPECIFIED.value)


def normalize_aggregation_role(value: str | AggregationRole | None) -> str:
    raw = str(getattr(value, "value", value) or "").strip().casefold()
    if raw in {item.value for item in AggregationRole}:
        return raw
    return _AGGREGATION_ROLE_ALIASES.get(raw, AggregationRole.UNSPECIFIED.value)


def quantity_semantics_from_scalar_role(
    scalar_role: str | ScalarSemanticRole | None,
    *,
    quantity_kind: str | QuantityKind | None = None,
) -> QuantitySemantics:
    """Lift one legacy scalar role into independent quantity dimensions."""
    role = normalize_scalar_role(scalar_role)
    kind = normalize_quantity_kind(quantity_kind)
    statistical = StatisticalRole.UNSPECIFIED.value
    aggregation = AggregationRole.UNSPECIFIED.value
    if role in {
        ScalarSemanticRole.POINT_ESTIMATE.value,
        ScalarSemanticRole.LOWER_BOUND.value,
        ScalarSemanticRole.UPPER_BOUND.value,
        ScalarSemanticRole.CONFIDENCE_LEVEL.value,
        ScalarSemanticRole.FILTER_THRESHOLD.value,
        ScalarSemanticRole.RANGE_BOUND.value,
    }:
        statistical = role
    elif role == ScalarSemanticRole.TOTAL_COUNT.value:
        aggregation = AggregationRole.TOTAL.value
    elif role == ScalarSemanticRole.CATEGORY_COUNT.value:
        aggregation = AggregationRole.CATEGORY.value
    elif role == ScalarSemanticRole.PERCENTAGE_SHARE.value:
        aggregation = AggregationRole.SHARE.value
        if kind == QuantityKind.UNSPECIFIED.value:
            kind = QuantityKind.PERCENTAGE.value
    elif role == ScalarSemanticRole.PERIOD_VALUE.value:
        kind = QuantityKind.PERIOD.value
    if kind == QuantityKind.UNSPECIFIED.value and role not in {
        ScalarSemanticRole.NOT_APPLICABLE.value,
        ScalarSemanticRole.PERIOD_VALUE.value,
    }:
        kind = QuantityKind.NUMBER.value
    return QuantitySemantics(kind, statistical, aggregation)


def scalar_role_from_quantity_semantics(
    quantity_kind: str | QuantityKind | None,
    statistical_role: str | StatisticalRole | None,
    aggregation_role: str | AggregationRole | None,
) -> str:
    """Project orthogonal semantics for legacy serialization only."""
    statistical = normalize_statistical_role(statistical_role)
    aggregation = normalize_aggregation_role(aggregation_role)
    kind = normalize_quantity_kind(quantity_kind)
    if statistical in {
        StatisticalRole.LOWER_BOUND.value,
        StatisticalRole.UPPER_BOUND.value,
        StatisticalRole.CONFIDENCE_LEVEL.value,
        StatisticalRole.FILTER_THRESHOLD.value,
        StatisticalRole.RANGE_BOUND.value,
    }:
        return statistical
    if aggregation == AggregationRole.SHARE.value or (
        kind == QuantityKind.PERCENTAGE.value
        and aggregation == AggregationRole.CATEGORY.value
    ):
        return ScalarSemanticRole.PERCENTAGE_SHARE.value
    if aggregation == AggregationRole.TOTAL.value:
        return ScalarSemanticRole.TOTAL_COUNT.value
    if aggregation == AggregationRole.CATEGORY.value:
        return ScalarSemanticRole.CATEGORY_COUNT.value
    if statistical == StatisticalRole.POINT_ESTIMATE.value:
        return ScalarSemanticRole.POINT_ESTIMATE.value
    if kind == QuantityKind.PERIOD.value:
        return ScalarSemanticRole.PERIOD_VALUE.value
    return ScalarSemanticRole.NOT_APPLICABLE.value


def quantity_dimension_compatibility(
    required: str,
    observed: str,
    *,
    dimension: str,
) -> tuple[bool, str]:
    """Compare one explicit dimension, failing closed only when requested."""
    normalizers = {
        "quantity_kind": normalize_quantity_kind,
        "statistical_role": normalize_statistical_role,
        "aggregation_role": normalize_aggregation_role,
    }
    unspecified = {
        "quantity_kind": QuantityKind.UNSPECIFIED.value,
        "statistical_role": StatisticalRole.UNSPECIFIED.value,
        "aggregation_role": AggregationRole.UNSPECIFIED.value,
    }
    normalizer = normalizers[dimension]
    expected = normalizer(required)
    actual = normalizer(observed)
    if expected == unspecified[dimension]:
        return True, "NOT_REQUIRED"
    if actual == unspecified[dimension]:
        return False, "SOURCE_ROLE_UNSPECIFIED"
    return (True, "MATCH") if expected == actual else (False, "MISMATCH")


def normalize_scalar_role(value: str | ScalarSemanticRole | None) -> str:
    """Return the one canonical scalar-role value for source or request data."""
    raw = str(getattr(value, "value", value) or "").strip().casefold()
    if raw in {item.value for item in ScalarSemanticRole}:
        return raw
    return _SCALAR_ROLE_ALIASES.get(raw, ScalarSemanticRole.NOT_APPLICABLE.value)


def scalar_role_compatibility(
    required: str | ScalarSemanticRole | None,
    observed: str | ScalarSemanticRole | None,
) -> tuple[bool, str]:
    """Compare canonical scalar roles without falling back to numeric shape.

    A total is a sound point estimate when the request has not required a
    narrower total role.  All other material roles stay distinct: in
    particular, a lower bound, confidence level, and threshold never satisfy
    a point estimate.
    """
    expected = normalize_scalar_role(required)
    actual = normalize_scalar_role(observed)
    if expected == ScalarSemanticRole.NOT_APPLICABLE.value:
        return True, "NOT_REQUIRED"
    if expected == actual:
        return True, "MATCH"
    if (
        expected == ScalarSemanticRole.POINT_ESTIMATE.value
        and actual == ScalarSemanticRole.TOTAL_COUNT.value
    ):
        return True, "MATCH_EQUIVALENT_TOTAL_ESTIMATE"
    return False, "MISMATCH"


def normalize_semantic_period(value: str | None) -> str:
    """Normalize only source forms whose period identity is explicit."""
    raw = normalize_semantic_text(value or "").casefold()
    academic = re.fullmatch(r"(?:academic\s+year\s+)?((?:19|20)\d{2})[/-](\d{2}|(?:19|20)\d{2})", raw)
    if academic:
        start = int(academic.group(1))
        tail = academic.group(2)
        end = int(tail) if len(tail) == 4 else (start // 100) * 100 + int(tail)
        if end == start + 1:
            return f"academic-year:{start}-{end}"
    return raw


def periods_semantically_compatible(
    required: str | None,
    observed: str | None,
    *,
    required_time_role: str = "",
    observed_time_role: str = "",
) -> tuple[bool, str]:
    """Compare source-derived periods without assigning request years to facts."""
    wanted = normalize_semantic_period(required)
    actual = normalize_semantic_period(observed)
    if not wanted:
        if required_time_role and observed_time_role:
            return (
                required_time_role == observed_time_role,
                "MATCH_TIME_ROLE" if required_time_role == observed_time_role else "MISMATCH_TIME_ROLE",
            )
        return True, "NOT_REQUIRED"
    if wanted == actual:
        return True, "MATCH"
    # A request can designate its endpoint as the current reporting period
    # while the source names it only as ``current``.  This admits the source
    # label for a current-role requirement; downstream rendering still uses
    # the source period and therefore cannot assert the request's calendar
    # year.  It never equates current with a historical source period.
    if (
        required_time_role == "current"
        and actual == "current"
        and observed_time_role in {"", "current"}
    ):
        return True, "MATCH_CURRENT_ROLE"
    if wanted == "current" and required_time_role == "current" and (
        actual == "current" or observed_time_role == "current"
    ):
        return True, "MATCH_CURRENT_CONTEXT"
    return False, "MISMATCH"


def requirement_satisfaction_mode(kind: str) -> RequirementSatisfactionMode:
    """Classify a requirement without consulting evidence or route enums."""
    normalized = str(kind or "").casefold()
    if normalized in {"comparison_direction"}:
        return RequirementSatisfactionMode.DERIVED_FACT
    if normalized in {"filter_condition"}:
        return RequirementSatisfactionMode.REQUEST_CONDITION
    if normalized in {
        "category_set", "category_value_bindings",
        "common_metric_unit_period_population", "winner", "winner_attribute",
        "list", "list_attributes",
    }:
        return RequirementSatisfactionMode.COMPOSITE_REQUIREMENT
    return RequirementSatisfactionMode.SOURCE_FACT

_BIDI_RE = re.compile(r"[\u200e\u200f\u202a-\u202e\u2066-\u2069]")
_TOKEN_RE = re.compile(r"[A-Za-z0-9\u05d0-\u05ea]{2,}")
_NUMBER_RE = re.compile(
    rf"(?<![\w])(?P<value>{GROUPED_DECIMAL_LITERAL_PATTERN}|\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>%|percent(?:age)?|MW|GW|kW|MWh|GWh|km|m|kg|"
    r"million|billion|thousand|אלף|מיליון|מיליארד|"
    r"עובדים|משרות|חברות|ארגונים|יחידות|לקוחות|שקלים|דולר(?:ים)?)?",
    re.IGNORECASE,
)
_PERIOD_RE = re.compile(
    r"(?<![\w])(?:19|20)\d{2}(?:[/-]\d{2,4})?(?![\w])|"
    r"(?<![\w])(?:Q[1-4]|H[12])[- /]?(?:19|20)?\d{2}(?![\w])",
    re.IGNORECASE,
)
_CLAIM_CITATION_RE = re.compile(r"\[(?P<id>\d{1,3})\]")

_STOPWORDS = frozenset({
    "the", "a", "an", "of", "to", "for", "and", "or", "in", "on",
    "at", "by", "from", "with", "what", "which", "who", "why", "how",
    "was", "were", "is", "are", "did", "does", "report", "document",
    "מה", "מהו", "מהי", "מי", "מדוע", "למה", "כיצד", "איך", "של", "את",
    "על", "עם", "לפי", "בדוח", "הדוח", "המחקר", "בין", "לבין", "או", "גם",
})
_GENERIC_POPULATION_TERMS = frozenset({
    "among", "within", "across", "in", "of", "the", "region",
    "population", "sample", "group", "subgroup", "segment", "cohort",
    "companies", "company", "workers", "employees", "customers", "sites",
    "projects", "בקרב", "מבין", "מתוך", "אזור", "אוכלוסייה", "אוכלוסיית", "מדגם", "קבוצה", "חברות", "חברה",
    "עובדים", "לקוחות", "אתרים", "פרויקטים",
})

_NEGATION_RE = re.compile(
    r"(?:\b(?:not|no|without|excluded|omitted|absent)\b|"
    r"(?<![\u05d0-\u05ea])(?:לא|אין|אינו|אינה|אינם|אינן|ללא)(?![\u05d0-\u05ea]))",
    re.IGNORECASE,
)
_POSITIVE_INCLUDE_RE = re.compile(
    r"(?:\b(?:include|included|contains?|comprised)\b|"
    r"(?<![\u05d0-\u05ea])(?:נכלל|נכללה|נכללו|כלל|הכיל|הכילה)"
    r"(?![\u05d0-\u05ea]))",
    re.IGNORECASE,
)
_NEGATIVE_INCLUDE_RE = re.compile(
    r"(?:\b(?:excluded|omitted|not\s+included|did\s+not\s+include|"
    r"outside\s+(?:the\s+)?scope|underrepresented|under-represented)\b|"
    r"(?<![\u05d0-\u05ea])(?:לא|אינ(?:ו|ה|ם|ן)|איננו|אינה)"
    r"(?![\u05d0-\u05ea]).{0,30}"
    r"(?:נכלל|נכללה|נכללו|כלול|כלולה|כלולים|כלולות|כולל|כוללת)|מחוץ\s+ל(?:תחום|אוכלוסי|אומדן)|"
    r"(?:י|יי)צוג\s+חסר)",
    re.IGNORECASE,
)
_CAUSE_RE = re.compile(
    r"(?:^\s*(?:why|מדוע|למה)\b|\b(?:cause|reason|because|due\s+to|"
    r"as\s+a\s+result\s+of)\b|"
    r"(?:explain|explains|explained|מסביר|מסבירה).{0,20}(?:why|מדוע|למה)|"
    r"מה\s+גרם|הסיבה|בשל|עקב|מפני|משום)",
    re.IGNORECASE,
)
_CAUSE_EVIDENCE_RE = re.compile(
    r"(?:\b(?:because|since|due\s+to|reason|therefore|so\s+that|"
    r"in\s+order\s+to|as\s+a\s+result|accordingly|to\s+(?:correct|"
    r"address|reduce|prevent))\b|"
    r"(?:בשל|עקב|מפני|משום|מאחר|כדי|על\s+מנת|הסיבה|לכן|"
    r"בהתאם\s+לכך|לתיקון|לצמצום|למניעת))",
    re.IGNORECASE,
)
_PURPOSE_RE = re.compile(
    r"(?:\b(?:purpose|objective|goal|aim|intended\s+to|in\s+order\s+to)\b|"
    r"(?:מטר(?:ה|ת)|לשם\s+מה|נועד|נועדה|כדי|על\s+מנת))",
    re.IGNORECASE,
)
_DEFINITION_RE = re.compile(
    r"(?:\b(?:define|defined|definition|means?|responsib(?:le|ility)|role\s+of)\b|"
    r"(?:מגדיר|מוגדר|הגדר|פירוש|משמעות|תפקיד|אחריות|כולל))",
    re.IGNORECASE,
)
_CHANGE_RE = re.compile(
    r"(?:\b(?:change|changed|trend|increase[ds]?|decrease[ds]?|rose|fell|grew|"
    r"declined|over\s+time)\b|(?:שינוי|השתנ|מגמ|עלייה|ירידה|עלה|עלתה|ירד|ירדה|גדל|פחת))",
    re.IGNORECASE,
)
_COMPARE_RE = re.compile(
    r"(?:\b(?:compare|comparison|versus|vs\.?|than)\b|"
    r"\b(?:difference|gap)\s+between\b|"
    r"\b(?:more|less|fewer)\b|"
    r"(?:השוו|השווה|השוואה|לעומת|בין.{0,80}לבין|יותר|(?<!ל)פחות))",
    re.IGNORECASE,
)
_ARGMAX_RE = re.compile(
    r"(?:\b(?:largest|highest|greatest|maximum|most)\b|"
    r"(?:הגדול(?:ה|ים|ות)?|הגבוה(?:ה|ים|ות)?|המרבי(?:ת|ים|ות)?|"
    r"הרב(?:ה|ים|ות)?)\s+ביותר|הכי\s+(?:גדול(?:ה)?|גבוה(?:ה)?|רב(?:ה)?))",
    re.IGNORECASE,
)
_ARGMIN_RE = re.compile(
    r"(?:\b(?:smallest|lowest|minimum)\b|(?<!at\s)\bleast\b|"
    r"(?:הקטן(?:ה|ים|ות)?|הנמוך(?:ה|ים|ות)?|ה המזערי(?:ת|ים|ות)?|"
    r"המועט(?:ה|ים|ות)?)\s+ביותר|הכי\s+(?:קטן(?:ה)?|נמוך(?:ה)?|מעט))",
    re.IGNORECASE,
)
_THRESHOLD_CONTEXT_RE = re.compile(
    r"(?:\b(?:more\s+than|over|above|at\s+least|at\s+most|under|below|"
    r"fewer\s+than|up\s+to|minimum|maximum)\s*$|"
    r"(?:מעל|יותר\s+מ|לפחות|לכל\s+היותר|פחות\s+מ|עד)\s*$)",
    re.IGNORECASE,
)
_CONFIDENCE_CONTEXT_RE = re.compile(
    r"(?:\bconfidence(?:\s+level)?\b|רמת\s+ביטחון|ביטחון)", re.IGNORECASE
)
_LOWER_BOUND_CONTEXT_RE = re.compile(
    r"(?:\blower\s+bound\b|\bminimum\s+estimate\b|חסם\s+תחתון|אומדן\s+מזערי)",
    re.IGNORECASE,
)
_UPPER_BOUND_CONTEXT_RE = re.compile(
    r"(?:\bupper\s+bound\b|\bmaximum\s+estimate\b|חסם\s+עליון|אומדן\s+מרבי)",
    re.IGNORECASE,
)
_RANGE_BOUND_CONTEXT_RE = re.compile(
    r"(?:\b(?:range|between)\b|(?:טווח|בין))", re.IGNORECASE,
)
_POINT_ESTIMATE_CONTEXT_RE = re.compile(
    r"(?:\b(?:estimate|estimated|estimated\s+at|estimated\s+to\s+be|"
    r"was|were|is|are|stood\s+at|amounted\s+to|numbered)\b|"
    r"(?:אומדן|הוערך|עמד(?:ה)?\s+על|היה|היו|הייתה))",
    re.IGNORECASE,
)
_TOTAL_COUNT_CONTEXT_RE = re.compile(
    r"(?:\b(?:total|overall|all|number\s+of|count\s+of)\b|"
    r"(?:(?:מספר|כמות)\s+(?:ה)?[\u05d0-\u05ea]+)|"
    r"(?:סה.?כ|סך\s+הכל|כלל))",
    re.IGNORECASE,
)
_CATEGORY_COUNT_CONTEXT_RE = re.compile(
    r"(?:\b(?:category|segment|role|position|band|group)\b|"
    r"(?:קטגוריה|פלח|תפקיד|קבוצה|רמה))",
    re.IGNORECASE,
)
_COUNT_UNITS = frozenset({
    "count", "employee", "employees", "worker", "workers", "facilities",
    "position", "positions", "job", "jobs", "company", "companies",
    "organization", "organizations", "unit", "units", "case", "cases",
    "alert", "alerts",
    "respondent", "respondents", "person", "people", "user", "users",
    "customer", "customers", "site", "sites", "label", "labels",
    "inspection", "inspections", "node", "nodes", "record", "records",
    "item", "items", "עובדים", "משרות", "חברות", "ארגונים",
    "יחידות", "מקרים", "משיבים", "אנשים", "משתמשים", "לקוחות",
})
_UNIT_EQUIVALENCE = {
    "employee": ("employee", "worker", "עובד"),
    "employees": ("employee", "worker", "עובד"),
    "worker": ("employee", "worker", "עובד"),
    "workers": ("employee", "worker", "עובד"),
    "company": ("company", "companies", "חבר"),
    "companies": ("company", "companies", "חבר"),
    "organization": ("organization", "organisation", "ארגו"),
    "organizations": ("organization", "organisation", "ארגו"),
    "facility": ("facility", "facilities", "site", "מתקנ", "אתר"),
    "facilities": ("facility", "facilities", "site", "מתקנ", "אתר"),
    "sites": ("facility", "facilities", "site", "מתקנ", "אתר"),
}
NUMERIC_VALUE_ROLES = frozenset({
    "requested_answer_value", "filter_threshold", "period",
    "confidence_level", "lower_bound", "category_value", "count",
    "percentage", "table_header", "unrelated_neighbor",
})
_SOURCE_CARDINALITY_PREDICATE_RE = re.compile(
    r"\b(?:employ(?:s|ed|ing)?|recruit(?:s|ed|ing)?|include(?:s|d|ing)?|"
    r"contain(?:s|ed|ing)?|record(?:s|ed|ing)?|report(?:s|ed|ing)?|"
    r"reach(?:es|ed|ing)?|process(?:es|ed|ing)?|handle(?:s|d|ing)?|"
    r"manage(?:s|d|ing)?|identify(?:s|ied|ing)?|use(?:s|d|ing)?|"
    r"there\s+(?:are|were))\s*$|"
    r"(?:מעסיק(?:ה|ים|ות)?|גייס(?:ה|ו|ים|ות)?|נכלל(?:ה|ו|ים|ות)?|"
    r"כולל(?:ת|ים|ות)?|דיווח(?:ה|ו|ים|ות)?)\s*$",
    re.IGNORECASE,
)
_SOURCE_CARDINALITY_SUFFIX_RE = re.compile(
    r"^\s*(?:[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*\s+){0,3}"
    r"(?:are|were|is|was|employed|recruited|included|responded|"
    r"reported|processed|recruits?|employs?)\b|"
    r"^\s*(?:[\u05d0-\u05ea][\u05d0-\u05ea'׳-]*\s+){0,3}"
    r"(?:הועסק(?:ו|ה)?|גויס(?:ו|ה)?|מגייס(?:ת|ים|ות)?|"
    r"נכלל(?:ו|ה)?|השיב(?:ו|ה|ים|ות)?)\b",
    re.IGNORECASE,
)
_SOURCE_CARDINALITY_NOUN_RELATION_RE = re.compile(
    r"\b(?:number|count)\s+of\s+"
    r"(?:[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*\s+){1,10}?"
    r"(?:is|are|was|were|stood\s+at|amounted\s+to|numbered)\s*$|"
    r"(?:מספר|כמות)\s+(?:[\u05d0-\u05ea][\u05d0-\u05ea\u0022\u0027׳-]*\s+){1,10}?"
    r"(?:הוא|היא|היו|היה|הייתה|עמד(?:ה)?\s+על)\s*$",
    re.IGNORECASE,
)
_SOURCE_CARDINALITY_ENTITY_RELATION_RE = re.compile(
    r"\b(?:employees?|workers?|positions?|jobs?|companies|organizations?|"
    r"units?|cases?|alerts?|respondents?|people|users?|customers?|sites?|"
    r"labels?|inspections?|nodes?|records?|items?)\b"
    r"(?:\s+[A-Za-z][A-Za-z'’-]*){0,4}?\s+"
    r"(?:is|are|was|were|stood\s+at|amounted\s+to|numbered|"
    r"increased|decreased|rose|fell|changed|grew|declined)"
    r"(?:\s+(?:from|to))?\s*$|"
    r"(?:עובדים|משרות|חברות|ארגונים|יחידות|מקרים|משיבים|אנשים|"
    r"משתמשים|לקוחות)"
    r"(?:\s+[\u05d0-\u05ea\u0022\u0027׳-]+){0,4}?\s+"
    r"(?:הוא|היא|היו|היה|הייתה|עמד(?:ה)?\s+על|עלה|עלתה|עלו|"
    r"ירד|ירדה|ירדו|השתנה|השתנתה|פחת|פחתה)\s*$",
    re.IGNORECASE,
)
_SOURCE_CARDINALITY_SERIES_RE = re.compile(
    r"\b(?:employees?|workers?|positions?|jobs?|companies|organizations?|"
    r"units?|cases?|alerts?|respondents?|people|users?|customers?|sites?|"
    r"labels?|inspections?|nodes?|records?|items?)\b"
    r"(?:\s+[A-Za-z][A-Za-z'’-]*){0,4}?\s+"
    r"(?:is|are|was|were|stood\s+at|amounted\s+to|numbered|"
    r"increased|decreased|rose|fell|changed|grew|declined)"
    r"[^.;\n]*\d|"
    r"(?:עובדים|משרות|חברות|ארגונים|יחידות|מקרים|משיבים|אנשים|"
    r"משתמשים|לקוחות)"
    r"(?:\s+[\u05d0-\u05ea\u0022\u0027׳-]+){0,4}?\s+"
    r"(?:הוא|היא|היו|היה|הייתה|עמד(?:ה)?\s+על|עלה|עלתה|עלו|"
    r"ירד|ירדה|ירדו|השתנה|השתנתה|פחת|פחתה)"
    r"[^.;\n]*\d",
    re.IGNORECASE,
)
_SOURCE_CARDINALITY_CONTINUATION_RE = re.compile(
    r"(?:\bof\s+which\b|מתוכ(?:ו|ן|ם)?)\s*$", re.IGNORECASE,
)
_SOURCE_CARDINALITY_PARENT_CUE_RE = re.compile(
    r"\b(?:number|count|contain(?:s|ed|ing)?)\b|(?:מספר|כמות|כולל(?:ת|ים|ות)?|כלל)",
    re.IGNORECASE,
)


def _source_metric_label_establishes_cardinality(label: str) -> bool:
    """Return whether a source-bound metric label explicitly names |S|.

    ``number/count of <population>`` (and its Hebrew equivalent) is a
    grammatical source relation, not an entity whitelist, integer-shape
    heuristic, request interpretation, or chart-type shortcut.  Callers must
    provide only the scalar's own structural metric binding.
    """
    normalized = " ".join(str(label or "").split()).strip()
    # Explicit spreadsheet headers own their scalar through a source field
    # binding. A bare quantity header is a cardinality declaration; price or
    # description labels are never promoted by a neighboring quantity cell.
    field_label = re.sub(r"\s*\([A-Z]{1,3}\)\s*$", "", normalized).strip()
    if re.fullmatch(r"(?:quantity|qty\.?|כמות)", field_label, re.IGNORECASE):
        return True
    english = re.match(
        r"^(?:number|count)\s+of\s+(?P<population>.+)$",
        normalized,
        re.IGNORECASE,
    )
    hebrew = re.match(r"^(?:מספר|כמות)\s+(?P<population>.+)$", normalized)
    match = english or hebrew
    if match is None:
        return False
    population = match.group("population").strip()
    return bool(
        population
        and re.search(r"[A-Za-z\u05d0-\u05ea]", population)
        and not re.fullmatch(r"[\d\s.,%]+", population)
    )


_SCALAR_PROPOSITION_DELIMITER_RE = re.compile(r"[;:׃]|\.(?!\d)|(?<!\d)\.")


def _scalar_proposition_bounds(source: str, start: int, end: int) -> tuple[int, int]:
    """Keep a scalar's decimal literals inside its own source proposition."""
    left, right = 0, len(source)
    for delimiter in _SCALAR_PROPOSITION_DELIMITER_RE.finditer(source):
        if delimiter.end() <= start:
            left = delimiter.end()
        elif delimiter.start() >= end:
            right = delimiter.start()
            break
    return left, right


def classify_quantity_semantics(
    text: str,
    start: int,
    end: int,
    *,
    unit: str = "",
    structural_role: str = "",
    source_container: str = "",
    source_metric_label: str = "",
) -> QuantitySemantics:
    """Classify one scalar only from its own bounded source proposition.

    A numeric literal with no local source cue stays statistically and
    aggregationally unspecified.  In particular, number-ness does not imply
    ``point_estimate`` and a percent sign does not imply ``share``.
    """
    source = text or ""
    if _numeric_identifier_role(source, start, end):
        # A role-bound designation is identity, never a measured scalar. A
        # route or nearby count predicate cannot supply numerical meaning.
        return QuantitySemantics()
    proposition_start, proposition_end = _scalar_proposition_bounds(source, start, end)
    before = source[max(proposition_start, start - 72):start]
    after = source[end:min(proposition_end, end + 72)]
    # Scalar role labels cannot leak across adjacent propositions.  A
    # semicolon-delimited lower-bound statement, for example, must not turn a
    # preceding estimate into a lower bound merely because both live in the
    # same structural evidence unit.
    local_before = before
    local_after = after
    around = f"{local_before}{source[start:end]}{local_after}"
    def _span(match: re.Match[str], *, group: str | int = 0) -> tuple[int, int]:
        return (match.start(group), match.end(group))

    normalized_unit = normalize_semantic_text(unit).casefold()
    percentage_literal = unit == "%" or "%" in source[start:end + 1]
    continuation = _SOURCE_CARDINALITY_CONTINUATION_RE.search(local_before)
    continuation_parent = (
        local_before[:continuation.start()] if continuation is not None else ""
    )
    # Numeric atomization can trim a bounded ``of which`` fragment away from
    # its parent statement.  Recover only the preceding text in the same
    # authorized source container, and only when no proposition delimiter was
    # crossed.  This is source-local cardinality evidence; callers never pass
    # request text here.
    if source_container:
        source_position = source_container.rfind(source)
        if source_position >= 0:
            container_before = source_container[:source_position + start]
            container_continuation = re.search(
                r"(?P<parent>[^.;:׃]{0,160})"
                r"(?:\bof\s+which\b|מתוכ(?:ו|ן|ם)?)"
                r"(?P<children>[^.;:׃]{0,160})$",
                container_before,
                re.IGNORECASE,
            )
            if container_continuation is not None:
                continuation = continuation or container_continuation
                continuation_parent = container_continuation.group("parent")
    continuation_proves_cardinality = bool(
        continuation is not None
        and normalized_unit in _COUNT_UNITS
        and re.search(r"\d", continuation_parent)
        and _SOURCE_CARDINALITY_PARENT_CUE_RE.search(continuation_parent)
    )
    metric_label_proves_cardinality = _source_metric_label_establishes_cardinality(
        source_metric_label
    )
    same_scalar_cardinality = bool(
        (
            _SOURCE_CARDINALITY_PREDICATE_RE.search(local_before)
            and re.match(r"\s*[A-Za-z\u05d0-\u05ea]", local_after)
        )
        or _SOURCE_CARDINALITY_NOUN_RELATION_RE.search(local_before)
        or _SOURCE_CARDINALITY_ENTITY_RELATION_RE.search(local_before)
        # A coordinated endpoint inherits a count classification only when
        # the same bounded proposition has already stated a count entity and
        # its cardinality relation.  This does not promote a bare unit or a
        # neighbouring proposition to COUNT.
        or _SOURCE_CARDINALITY_SERIES_RE.search(local_before)
        or _SOURCE_CARDINALITY_SUFFIX_RE.search(local_after)
        # A bounded "of which" continuation may name a child category after
        # the parent cardinality relation.  The child is a COUNT only when its
        # own unit is a discrete entity and the same source proposition proves
        # a numeric cardinal parent; a bare entity unit remains NUMBER.
        or continuation_proves_cardinality
        # Flattened source records may preserve a metric, value, and category
        # in separate spans.  A same-record metric can still establish that
        # its bound value denotes |S| without a finite prose predicate.
        or metric_label_proves_cardinality
    )
    quantity_kind = (
        QuantityKind.PERCENTAGE.value
        if percentage_literal or normalized_unit in {"percent", "percentage", "אחוז"}
        else QuantityKind.COUNT.value
        if same_scalar_cardinality
        else QuantityKind.NUMBER.value
    )
    for match in _PERIOD_RE.finditer(source):
        if start < match.end() and end > match.start():
            return QuantitySemantics(QuantityKind.PERIOD.value)
    for match in _FILTER_CONDITION_RE.finditer(source):
        threshold_start, threshold_end = match.span("threshold")
        if start < threshold_end and end > threshold_start:
            return QuantitySemantics(
                quantity_kind, StatisticalRole.FILTER_THRESHOLD.value,
                statistical_role_cue_span=_span(match, group="operator"),
            )
    threshold = _THRESHOLD_CONTEXT_RE.search(local_before)
    if threshold:
        return QuantitySemantics(
            quantity_kind, StatisticalRole.FILTER_THRESHOLD.value,
            statistical_role_cue_span=(
                max(0, start - len(before)) + threshold.start(),
                max(0, start - len(before)) + threshold.end(),
            ),
        )
    confidence = _CONFIDENCE_CONTEXT_RE.search(local_before[-72:])
    trailing_confidence = re.match(
        r"\s*(?:confidence(?:\s+level)?|רמת\s+ביטחון|ביטחון)\b",
        local_after, re.IGNORECASE,
    )
    if percentage_literal and (confidence or trailing_confidence):
        cue = (
            (start - len(before) + max(0, len(before) - 72) + confidence.start(),
             start - len(before) + max(0, len(before) - 72) + confidence.end())
            if confidence else (end + trailing_confidence.start(), end + trailing_confidence.end())
        )
        return QuantitySemantics(
            quantity_kind, StatisticalRole.CONFIDENCE_LEVEL.value,
            statistical_role_cue_span=cue,
        )
    lower_before = _LOWER_BOUND_CONTEXT_RE.search(local_before[-56:])
    lower_after = _LOWER_BOUND_CONTEXT_RE.search(local_after[:24])
    if lower_before or lower_after:
        lower = lower_before or lower_after
        cue_base = start - min(56, len(before)) if lower_before else end
        return QuantitySemantics(
            quantity_kind, StatisticalRole.LOWER_BOUND.value,
            statistical_role_cue_span=(cue_base + lower.start(), cue_base + lower.end()),
        )
    upper_before = _UPPER_BOUND_CONTEXT_RE.search(local_before[-56:])
    upper_after = _UPPER_BOUND_CONTEXT_RE.search(local_after[:24])
    if upper_before or upper_after:
        upper = upper_before or upper_after
        cue_base = start - min(56, len(before)) if upper_before else end
        return QuantitySemantics(
            quantity_kind, StatisticalRole.UPPER_BOUND.value,
            statistical_role_cue_span=(cue_base + upper.start(), cue_base + upper.end()),
        )
    if _RANGE_BOUND_CONTEXT_RE.search(around) and re.search(
        r"(?:\bbetween\b[^.;:]{0,72}\band\b|\bfrom\b[^.;:]{0,72}\bto\b|"
        r"(?:טווח|בין)[^.;:]{0,72}(?:ו|עד))",
        around,
        re.IGNORECASE,
    ):
        cue = _RANGE_BOUND_CONTEXT_RE.search(around)
        return QuantitySemantics(
            quantity_kind, StatisticalRole.RANGE_BOUND.value,
            statistical_role_cue_span=(
                max(0, start - len(local_before)) + cue.start(),
                max(0, start - len(local_before)) + cue.end(),
            ),
        )
    point = _POINT_ESTIMATE_CONTEXT_RE.search(local_before[-64:])
    statistical_role = StatisticalRole.UNSPECIFIED.value
    statistical_span = None
    if point:
        statistical_role = StatisticalRole.POINT_ESTIMATE.value
        point_base = start - min(64, len(local_before))
        statistical_span = (point_base + point.start(), point_base + point.end())

    aggregation_role = AggregationRole.NONE.value
    aggregation_span = None
    share = re.search(
        r"\b(?:share|percentage|percent|rate|portion|represented|accounted\s+for)\b|"
        r"(?:שיעור|נתח|אחוז)", around, re.IGNORECASE,
    )
    total = _TOTAL_COUNT_CONTEXT_RE.search(around)
    category = _CATEGORY_COUNT_CONTEXT_RE.search(around)
    partitive_share = re.search(
        r"(?:מכלל|מתוך\s+(?:כלל|סך))|\bof\s+(?:the\s+)?(?:total|all)\b", source, re.I,
    ) if percentage_literal else None
    shared_share = re.match(
        r"\s*(?:the\s+)?(?P<share>share|percentage|rate|proportion)\b"
        r".{1,160}\b(?:was|were|is|are)\b",
        source[proposition_start:proposition_end], re.I,
    ) if percentage_literal else None
    if shared_share:
        aggregation_role = AggregationRole.SHARE.value
        aggregation_span = tuple(proposition_start + offset for offset in shared_share.span("share"))
    elif partitive_share:
        aggregation_role = AggregationRole.SHARE.value
        aggregation_span = partitive_share.span()
    elif share:
        aggregation_role = AggregationRole.SHARE.value
        aggregation_span = (
            max(0, start - len(local_before)) + share.start(),
            max(0, start - len(local_before)) + share.end(),
        )
    elif total:
        aggregation_role = AggregationRole.TOTAL.value
        aggregation_span = (
            max(0, start - len(local_before)) + total.start(),
            max(0, start - len(local_before)) + total.end(),
        )
    elif structural_role or category:
        aggregation_role = AggregationRole.CATEGORY.value
        if category:
            aggregation_span = (
                max(0, start - len(local_before)) + category.start(),
                max(0, start - len(local_before)) + category.end(),
            )
    return QuantitySemantics(
        quantity_kind, statistical_role, aggregation_role,
        statistical_role_cue_span=statistical_span,
        aggregation_role_cue_span=aggregation_span,
    )


def classify_scalar_role(
    text: str,
    start: int,
    end: int,
    *,
    unit: str = "",
    structural_role: str = "",
) -> str:
    """Return the legacy projection of :func:`classify_quantity_semantics`."""
    return classify_quantity_semantics(
        text, start, end, unit=unit, structural_role=structural_role,
    ).scalar_role


def scalar_role_has_local_cue(
    text: str,
    start: int,
    end: int,
    *,
    role: str | ScalarSemanticRole,
    unit: str = "",
    structural_role: str = "",
) -> bool:
    """Require an actual source cue before a role resolves a value list."""
    source = text or ""
    normalized = normalize_scalar_role(role)
    proposition_start, proposition_end = _scalar_proposition_bounds(source, start, end)
    before = source[max(proposition_start, start - 72):start]
    after = source[end:min(proposition_end, end + 48)]
    around = f"{before} {after}"
    if normalized == ScalarSemanticRole.POINT_ESTIMATE.value:
        return bool(_POINT_ESTIMATE_CONTEXT_RE.search(before[-64:]))
    if normalized == ScalarSemanticRole.LOWER_BOUND.value:
        return bool(_LOWER_BOUND_CONTEXT_RE.search(around))
    if normalized == ScalarSemanticRole.UPPER_BOUND.value:
        return bool(_UPPER_BOUND_CONTEXT_RE.search(around))
    if normalized == ScalarSemanticRole.CONFIDENCE_LEVEL.value:
        return bool(
            (unit == "%" or "%" in source[start:end + 1])
            and _CONFIDENCE_CONTEXT_RE.search(around)
        )
    if normalized == ScalarSemanticRole.FILTER_THRESHOLD.value:
        return bool(_THRESHOLD_CONTEXT_RE.search(before))
    if normalized == ScalarSemanticRole.TOTAL_COUNT.value:
        return bool(_TOTAL_COUNT_CONTEXT_RE.search(around))
    if normalized == ScalarSemanticRole.PERCENTAGE_SHARE.value:
        return bool(re.search(
            r"\b(?:share|percentage|percent|rate|portion|represented|"
            r"accounted\s+for)\b|(?:שיעור|נתח|אחוז)",
            around,
            re.IGNORECASE,
        ))
    if normalized == ScalarSemanticRole.CATEGORY_COUNT.value:
        return bool(structural_role or _CATEGORY_COUNT_CONTEXT_RE.search(around))
    if normalized == ScalarSemanticRole.RANGE_BOUND.value:
        return bool(_RANGE_BOUND_CONTEXT_RE.search(around))
    if normalized == ScalarSemanticRole.PERIOD_VALUE.value:
        return True
    return False


def infer_request_scalar_role(
    text: str,
    *,
    expected_unit: str = "",
    has_facet: bool = False,
) -> str:
    """Classify the scalar output requested by one request-local clause.

    This is intentionally separate from :func:`classify_scalar_role`: request
    role inference is only for requirements, while source facts retain roles
    that are established in their own evidence spans.
    """
    request = normalize_semantic_text(text)
    normalized_unit = normalize_semantic_text(expected_unit).casefold()
    if (
        normalized_unit in {"%", "percent", "percentage", "אחוז"}
        and _CONFIDENCE_CONTEXT_RE.search(request)
    ):
        return ScalarSemanticRole.CONFIDENCE_LEVEL.value
    if _LOWER_BOUND_CONTEXT_RE.search(request):
        return ScalarSemanticRole.LOWER_BOUND.value
    if _UPPER_BOUND_CONTEXT_RE.search(request):
        return ScalarSemanticRole.UPPER_BOUND.value
    if _CONFIDENCE_CONTEXT_RE.search(request):
        return ScalarSemanticRole.CONFIDENCE_LEVEL.value
    if _THRESHOLD_CONTEXT_RE.search(request):
        return ScalarSemanticRole.FILTER_THRESHOLD.value
    explicit_period_pair = bool(re.search(
        r"(?:19|20)\d{2}\s*(?:and|to|[-–—])\s*(?:19|20)\d{2}",
        request,
        re.IGNORECASE,
    ))
    if (
        not explicit_period_pair
        and _RANGE_BOUND_CONTEXT_RE.search(request)
        and re.search(
        r"\b(?:range|between|from)\b|(?:טווח|בין)", request, re.IGNORECASE,
        )
    ):
        return ScalarSemanticRole.RANGE_BOUND.value
    if (
        normalized_unit in {"%", "percent", "percentage", "אחוז"}
        or _PERCENT_RE.search(request)
    ):
        return ScalarSemanticRole.PERCENTAGE_SHARE.value
    if _TOTAL_COUNT_CONTEXT_RE.search(request):
        return ScalarSemanticRole.TOTAL_COUNT.value
    if has_facet or _CATEGORY_COUNT_CONTEXT_RE.search(request):
        return ScalarSemanticRole.CATEGORY_COUNT.value
    return ScalarSemanticRole.POINT_ESTIMATE.value


def infer_request_quantity_semantics(
    text: str,
    *,
    expected_unit: str = "",
    has_facet: bool = False,
) -> QuantitySemantics:
    """Compile only quantity dimensions the request explicitly constrains.

    This is request typing, not source classification.  A generic numeric or
    comparison request therefore never imports a point-estimate requirement
    just because an older route calls the requested value a scalar.
    """
    request = normalize_semantic_text(text)
    normalized_unit = normalize_semantic_text(expected_unit).casefold()
    quantity_kind = (
        QuantityKind.PERCENTAGE.value
        if normalized_unit in {"%", "percent", "percentage", "אחוז"}
        or bool(_PERCENT_RE.search(request))
        else QuantityKind.COUNT.value
        if normalized_unit == "count" or _request_has_count_cue(request)
        else QuantityKind.UNSPECIFIED.value
    )
    statistical_role = StatisticalRole.UNSPECIFIED.value
    # A contextual lower/upper-bound phrase can accompany a distinct
    # confidence-level subquestion.  The explicitly requested confidence
    # role is therefore more specific than that surrounding context.
    if _CONFIDENCE_CONTEXT_RE.search(request):
        statistical_role = StatisticalRole.CONFIDENCE_LEVEL.value
    elif _LOWER_BOUND_CONTEXT_RE.search(request):
        statistical_role = StatisticalRole.LOWER_BOUND.value
    elif _UPPER_BOUND_CONTEXT_RE.search(request):
        statistical_role = StatisticalRole.UPPER_BOUND.value
    elif _THRESHOLD_CONTEXT_RE.search(request):
        statistical_role = StatisticalRole.FILTER_THRESHOLD.value
    elif (
        _RANGE_BOUND_CONTEXT_RE.search(request)
        and not re.search(r"(?:19|20)\d{2}\s*(?:and|to|[-–—])\s*(?:19|20)\d{2}", request, re.I)
    ):
        statistical_role = StatisticalRole.RANGE_BOUND.value
    elif re.search(
        r"\b(?:point\s+estimate|estimate(?:d)?(?:\s+(?:value|demand|count|total))?)\b|"
        r"(?:אומדן\s+נקודתי)",
        request, re.I,
    ):
        statistical_role = StatisticalRole.POINT_ESTIMATE.value
    if statistical_role == StatisticalRole.CONFIDENCE_LEVEL.value:
        quantity_kind = QuantityKind.PERCENTAGE.value

    aggregation_role = AggregationRole.UNSPECIFIED.value
    if _TOTAL_COUNT_CONTEXT_RE.search(request):
        aggregation_role = AggregationRole.TOTAL.value
    elif re.search(r"\b(?:share|portion|proportion)\b|(?:נתח|חלק|שיעור)", request, re.I):
        aggregation_role = AggregationRole.SHARE.value
    elif has_facet or _CATEGORY_COUNT_CONTEXT_RE.search(request):
        aggregation_role = AggregationRole.CATEGORY.value
    return QuantitySemantics(quantity_kind, statistical_role, aggregation_role)


def _request_scalar_role(clause: "CanonicalClause") -> str:
    """Compile a requested scalar role from request language only."""
    request = " ".join((
        clause.text,
        clause.subject,
        clause.predicate,
        *clause.objects_or_metrics,
        *clause.requested_output_slots,
    ))
    return infer_request_scalar_role(
        request,
        expected_unit=(
            "percentage"
            if "percentage" in clause.requested_output_slots
            or _PERCENT_RE.search(request)
            else ""
        ),
        has_facet=bool(clause.requested_facets),
    )


def _request_quantity_semantics(clause: "CanonicalClause") -> QuantitySemantics:
    request = " ".join((
        clause.text, clause.subject, clause.predicate,
        *clause.objects_or_metrics, *clause.requested_output_slots,
    ))
    semantics = infer_request_quantity_semantics(
        request,
        expected_unit=(
            "percentage" if "percentage" in clause.requested_output_slots
            or _PERCENT_RE.search(request) else ""
        ),
        has_facet=bool(clause.requested_facets),
    )
    # A comparison's endpoints are operator structure, not scalar
    # lower/upper/range roles.  Only a scalar range explicitly requested as
    # an answer may constrain a source fact that way.
    if (
        clause.semantic_relation in {"compare", "change_over_time"}
        and semantics.statistical_role == StatisticalRole.RANGE_BOUND.value
    ):
        semantics = replace(
            semantics, statistical_role=StatisticalRole.UNSPECIFIED.value,
        )
    # Inclusion language and an "all" population scope are not a total
    # aggregation cue.  A source total is required only for explicit total
    # metric wording in this same request clause.
    if (
        semantics.aggregation_role == AggregationRole.TOTAL.value
        and not _EXPLICIT_TOTAL_REQUEST_RE.search(clause.text)
    ):
        semantics = replace(
            semantics, aggregation_role=AggregationRole.UNSPECIFIED.value,
        )
    return semantics
_COUNT_RE = re.compile(
    r"(?:^\s*(?:כמה|בכמה)\b(?!\s+(?:כסף|שקלים|דולרים|אירו|התקבל|שולם))|\bhow\s+many\b|\b(?:count|number)\s+of\b|"
    r"(?:מספר|כמות|היקף))",
    re.IGNORECASE,
)
_COUNT_SLOT_RE = re.compile(
    r"(?:\b(?:count|number|quantity|total)\b|"
    r"(?<![\u05d0-\u05ea])(?:מספר|כמות|סה.?כ)(?![\u05d0-\u05ea])|היקפ)",
    re.IGNORECASE,
)


def _request_has_count_cue(value: str, *, include_slot: bool = False) -> bool:
    """Separate a requested cardinality from an entity's number label."""
    from document_query import mask_role_bound_identifiers
    text = mask_role_bound_identifiers(value)
    return bool(_COUNT_RE.search(text) or include_slot and _COUNT_SLOT_RE.search(text))
_PERCENT_RE = re.compile(
    r"(?:%|\b(?:percent(?:age)?|share|rate|proportion)\b|"
    r"(?:אחוז|שיעור|חלק|נתח))",
    re.IGNORECASE,
)
_LIST_RE = re.compile(
    r"(?:^\s*(?:list|name|which|what\s+are|רשום|מנה|ציין|אילו|מהם|מהן)\b|"
    r"(?:רשימה|list\s+of))",
    re.IGNORECASE,
)
_FILTER_RE = re.compile(
    r"(?:\b(?:filter|only\s+those|where|meeting|above|below|under|over)\b|"
    r"(?:סנן|רק\s+(?:אלה|אלו)|מעל|מתחת|פחות\s+מ|יותר\s+מ|שעבר))",
    re.IGNORECASE,
)
_FILTER_CONDITION_RE = re.compile(
    r"(?P<operator>\b(?:exceed(?:s|ed|ing)?|pass(?:es|ed|ing)?|greater\s+than|more\s+than|"
    r"above|over|less\s+than|fewer\s+than|below|under|at\s+least|"
    r"at\s+most|before|after)\b|"
    r"(?:עלה|עלתה|עולה|עולות|עבר|עברה)"
    r"(?:\s+[A-Za-z\u05d0-\u05ea\"'׳/%.-]+){0,8}\s+(?:על|את)|"
    r"מעל|יותר\s+מ|"
    r"פחות\s+מ|מתחת|לפחות|לכל\s+היותר|לפני|אחרי)\s*"
    rf"(?P<threshold>{GROUPED_DECIMAL_LITERAL_PATTERN}|\d+(?:[.,]\d+)?)\s*"
    r"(?P<unit>%|percent(?:age)?|MW|GW|kW|MWh|GWh|km|m|kg|"
    r"million|billion|thousand|אלף|מיליון|מיליארד|"
    r"עובדים|משרות|חברות|ארגונים|יחידות|לקוחות|שקלים|דולר(?:ים)?)?",
    re.IGNORECASE,
)
_PROCEDURE_RE = re.compile(
    r"(?:^\s*how\b(?!\s+many\b).{0,96}\b(?:identify|identified|determine|"
    r"determined|calculate|calculated|measure|measured|perform|performed|"
    r"select|selected|construct|constructed|derive|derived|implement|implemented)\b|"
    r"^\s*(?:כיצד|איך)\b.{0,96}(?:זוה|אותר|נבחר|נקבע|חושב|נמדד|בוצע|"
    r"יושם|הוגדר|נבנה))",
    re.IGNORECASE,
)
_OUTER_EXPLANATION_RE = re.compile(
    r"(?:\b(?:explain|explains|explained|interpret|interprets|interpreted)\b"
    r".{0,80}\b(?:why|reason)\b|"
    r"(?:מסביר|מסבירה|הסביר|הסבירה|מפרש|מפרשת).{0,80}(?:מדוע|למה|סיבה))",
    re.IGNORECASE,
)
_EMPLOYS_RE = re.compile(
    r"(?:\b(?:employs?|employed|headcount|workforce)\b|"
    r"(?:מעסיק|מעסיקה|מעסיקות|מועסק|מועסקים|מספר\s+העובדים))",
    re.IGNORECASE,
)
_RECRUITS_EMPLOYEES_RE = re.compile(
    r"(?:\b(?:recruit(?:s|ed|ing)?|hir(?:e|es|ed|ing))\b.{0,24}"
    r"(?:employees?|workers?|staff|people)|"
    r"\b(?:employees?|workers?|staff|people)\b.{0,48}"
    r"(?:recruit(?:s|ed|ing)?|hir(?:e|es|ed|ing))\b|"
    r"(?:גייס|גייסה|גייסו|גייסות|גיוס).{0,24}(?:עובד|מומח|כוח\s*אדם))",
    re.IGNORECASE,
)
_RAISES_CAPITAL_RE = re.compile(
    r"(?:\b(?:rais(?:e|es|ed|ing)|fundrais(?:e|es|ed|ing))\b.{0,24}"
    r"(?:capital|money|funds?|financing)|"
    r"\bfundrais(?:e|es|ed|ing)\b|"
    r"(?:גייס|גייסה|גייסו|גיוס).{0,24}(?:כסף|הון|מימון))",
    re.IGNORECASE,
)

EDUCATION_TERM_PATTERN = (
    r'(?:\b(?:education|degree)\b|(?<!\w)(?:ו?[בלכ]?ה?תואר|ו?[בלמכ]?ה?השכלה)(?!\w))'
)

_FACET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("role", re.compile(r"(?:\b(?:role|occupation|job\s+type)\b|(?:תפקיד|עיסוק|סוגי?\s+משרה))", re.I)),
    ("education_level", re.compile(r"(?:"+EDUCATION_TERM_PATTERN+r"|\bqualification\b|(?<!\w)ו?[בלמכ]?ה?הכשרה(?!\w))", re.I)),
    ("experience_band", re.compile(r"(?:\b(?:experience|tenure|seniority)\b|(?:ניסיון|וותק|בכירות))", re.I)),
    ("population_subgroup", re.compile(r"(?:\b(?:subgroup|segment|cohort)\b|(?:תת.?קבוצה|פלח|קבוצה))", re.I)),
    ("total", re.compile(r"(?:\b(?:total|overall|all)\b|(?:סך\s*הכול|סה.?כ)|(?<![\u05d0-\u05ea])(?:כולל|כלל)(?![\u05d0-\u05ea]))", re.I)),
)


def _requested_facets(value: str, relation: str) -> tuple[str, ...]:
    facets = tuple(name for name, pattern in _FACET_PATTERNS if pattern.search(value))
    # Narrative experience is a summary topic. A requested experience band
    # requires an actual bound or an explicit level/group/tenure dimension;
    # it cannot be manufactured from a project's history or a source title.
    if relation == 'summary' and 'experience_band' in facets:
        explicit = experience_band_facets(value) or re.search(
            r'\b(?:tenure|seniority)\b|\bexperience\s+(?:levels?|bands?|ranges?|groups?)\b|'
            r'\b(?:levels?|bands?|ranges?)\s+of\s+experience\b|'
            r'(?<!\w)(?:וותק|ותק|בכירות)(?!\w)|'
            r'(?<!\w)(?:רמת|רמה|רמות|טווחי|קבוצות)\s+(?:של\s+)?(?:ה)?ניסיון(?!\w)',value,re.I)
        if not explicit:
            facets = tuple(name for name in facets if name != 'experience_band')
    return facets

_EXPERIENCE_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    "אפס": "0", "אחת": "1", "אחד": "1", "שתי": "2", "שתיים": "2", "שניים": "2",
    "שלוש": "3", "שלושה": "3", "ארבע": "4", "חמש": "5", "שש": "6",
    "שבע": "7", "שמונה": "8", "תשע": "9", "עשר": "10",
}
_EXPERIENCE_BOUND_RE = re.compile(
    r"(?P<operator>up\s+to|at\s+most|less\s+than|below|under|"
    r"more\s+than|over|above|at\s+least|עד|לכל\s+היותר|פחות\s+מ|"
    r"מתחת\s+ל|מעל|יותר\s+מ|לפחות)\s*"
    r"(?P<number>\d+(?:\.\d+)?|" + "|".join(_EXPERIENCE_NUMBER_WORDS) + r")"
    r"\s+(?P<unit>years?|months?|weeks?|days?|שנים|שנות|חודשים|חודשי|שבועות|ימים|ימי)"
    r"(?:\s+(?:of\s+)?(?:experience|ניסיון))?",
    re.IGNORECASE,
)
_EXPERIENCE_ZERO_RANGE_RE = re.compile(
    r"(?<![\w\u05d0-\u05ea])(?P<zero>zero|0|אפס)(?![\w\u05d0-\u05ea])"
    r"\s+(?P<connector>through|to|until|עד)\s*"
    r"(?P<number>\d+(?:\.\d+)?|" + "|".join(_EXPERIENCE_NUMBER_WORDS) + r")"
    r"\s+(?P<unit>years?|months?|weeks?|days?|שנים|שנות|חודשים|חודשי|שבועות|ימים|ימי)"
    r"(?:\s+(?:of\s+)?(?:experience|ניסיון))?",
    re.IGNORECASE,
)
_ENTRY_EXPERIENCE_RE = re.compile(
    r"\b(?:entry[ -]level|early[ -]career|junior)\b|"
    r"(?:בתחילת\s+דרכ[םן]|מתחיל(?:ים|ות)|זוטר(?:ים|ות))", re.I,
)
_SENIOR_EXPERIENCE_RE = re.compile(
    r"\b(?:experienced|senior)\s+(?:workers?|employees?)\b|"
    r"(?:מנוס(?:ים|ות)|בכיר(?:ים|ות))", re.I,
)


def _experience_bound_owned(value: str, match: re.Match[str]) -> bool:
    """A duration must name experience, rather than borrow a remote mention."""
    start, end = match.start("operator"), match.end()
    if end < len(value) and re.match(r"[A-Za-z\u05d0-\u05ea]", value[end]):
        return False
    bound = value[start:end]
    if re.search(r"\bexperience\b|ניסיון", bound, re.I):
        return True
    after = value[end:end + 80]
    owner = re.match(
        r"\s+(?:(?:of|של)\s+)?"
        r"(?:(?:professional|prior|relevant|work|מקצועי|קודם|רלוונטי|תעסוקתי)\s+)?"
        r"(?:experience|tenure|seniority|ניסיון|וותק)\b", after, re.I,
    )
    if owner:
        return True
    # An explicit other owner (years of education, שנות הכשרה, etc.)
    # cannot be reassigned even when an experience label occurs nearby.
    if re.match(r"\s+(?:of|של)\b", after, re.I) or bound.endswith("שנות"):
        return False
    prefix = value[max(0, start - 80):start]
    return bool(re.search(
        r"(?:\b(?:experience|tenure|seniority)\b|ניסיון|וותק)"
        r"(?:\s+(?:professional|prior|relevant|work|מקצועי|קודם|רלוונטי|תעסוקתי))?"
        r"\s*(?:of|של|:)?\s*\(?\s*$", prefix, re.I,
    ))


def _experience_zero_range_owned(value: str, match: re.Match[str]) -> bool:
    """Require the zero-based range to be explicitly owned by experience."""
    start, end = match.span()
    bound = value[start:end]
    if re.search(r"\bexperience\b|ניסיון", bound, re.I):
        return True
    after = value[end:end + 80]
    if re.match(r"\s+(?:of|של)\s+(?:education|training|השכלה|הכשרה)\b", after, re.I):
        return False
    if re.match(
        r"\s+(?:(?:of|של)\s+)?"
        r"(?:(?:required|professional|prior|relevant|work|מקצועי|קודם|רלוונטי|תעסוקתי)\s+)?"
        r"(?:experience|tenure|seniority|ניסיון|וותק)\b",
        after,
        re.I,
    ):
        return True
    prefix = value[max(0, start - 100):start]
    return bool(re.search(
        r"(?:\b(?:experience|tenure|seniority)\b|ניסיון|וותק)"
        r"(?:\s+(?:professional|prior|relevant|work|required|מקצועי|קודם|רלוונטי|תעסוקתי))?"
        r"\s*(?:is|was|are|were|of|של|:)?\s*\(?\s*$",
        prefix,
        re.I,
    ))


def _experience_zero_range_is_inclusive(
    value: str,
    match: re.Match[str],
) -> bool:
    connector = normalize_semantic_text(match.group("connector")).casefold()
    if connector in {"through", "עד"}:
        return True
    return bool(re.match(
        r"\s*(?:[,;]\s*)?(?:\(\s*)?(?:inclusive|incl\.?|כולל)\b",
        value[match.end():match.end() + 32],
        re.I,
    ))


def canonical_decimal_literal(value: str) -> str:
    """Canonical decimal spelling without rounding or changing its value."""
    match = re.fullmatch(r"(?P<sign>[-+]?)(?P<whole>\d+)(?:\.(?P<fraction>\d+))?", value)
    if match is None:
        return value
    whole = match.group("whole")
    fraction = match.group("fraction") or ""
    whole = whole.lstrip("0") or "0"
    fraction = fraction.rstrip("0")
    return match.group("sign") + whole + ("." + fraction if fraction else "")


def _experience_facet_key(value: str) -> tuple[str, ...]:
    parts = value.split(":")
    if len(parts) not in {2, 4, 5} or parts[0] != "experience_band" or parts[1] not in {"lower_band", "higher_band"}:
        return ()
    if len(parts) == 2:
        return tuple(parts)
    operators = {"lt", "lte"} if parts[1] == "lower_band" else {"gt", "gte"}
    if (parts[2] not in operators or not re.fullmatch(r"\d+(?:\.\d+)?", parts[3])
            or (len(parts) == 5 and parts[4] not in {"years", "months", "weeks", "days"})):
        return ()
    return (*parts[:3], canonical_decimal_literal(parts[3]), parts[4] if len(parts) == 5 else "years")


def experience_band_bounds(value: str) -> tuple[tuple[str, tuple[int, int]], ...]:
    """Explicit owned facets and the literal spans of their threshold values."""
    output = []
    for match in _EXPERIENCE_BOUND_RE.finditer(value):
        if not _experience_bound_owned(value, match):
            continue
        operator = normalize_semantic_text(match.group("operator")).casefold()
        comparison = (
            "lte" if operator in {"up to", "at most", "עד", "לכל היותר"}
            else "gte" if operator in {"at least", "לפחות"}
            else "lt" if operator in {"less than", "below", "under", "פחות מ", "מתחת ל"}
            else "gt"
        )
        raw_number = match.group("number").casefold()
        number = canonical_decimal_literal(_EXPERIENCE_NUMBER_WORDS.get(raw_number, raw_number))
        band = "lower_band" if comparison in {"lt", "lte"} else "higher_band"
        raw_unit = match.group("unit").casefold()
        unit = ("years" if raw_unit in {"year", "years", "שנים", "שנות"}
                else "months" if raw_unit in {"month", "months", "חודשים", "חודשי"}
                else "weeks" if raw_unit in {"week", "weeks", "שבועות"} else "days")
        # The established four-part form denotes years. Other durations have
        # an explicit unit and cannot satisfy a year-bound request.
        facet = f"experience_band:{band}:{comparison}:{number}" + (f":{unit}" if unit != "years" else "")
        output.append((facet, match.span("number")))
    for match in _EXPERIENCE_ZERO_RANGE_RE.finditer(value):
        if not _experience_zero_range_is_inclusive(value, match):
            continue
        if not _experience_zero_range_owned(value, match):
            continue
        raw_number = match.group("number").casefold()
        number = canonical_decimal_literal(
            _EXPERIENCE_NUMBER_WORDS.get(raw_number, raw_number)
        )
        raw_unit = match.group("unit").casefold()
        unit = ("years" if raw_unit in {"year", "years", "שנים", "שנות"}
                else "months" if raw_unit in {"month", "months", "חודשים", "חודשי"}
                else "weeks" if raw_unit in {"week", "weeks", "שבועות"} else "days")
        facet = "experience_band:lower_band:lte:" + number
        if unit != "years":
            facet += f":{unit}"
        output.append((facet, match.span("number")))
    return tuple(output)


def experience_band_facets(value: str) -> tuple[str, ...]:
    """Normalize explicit source categories without inventing career cutoffs."""
    output = [facet for facet, _span in experience_band_bounds(value)]
    if not output:
        if _ENTRY_EXPERIENCE_RE.search(value):
            output.append("experience_band:lower_band")
        if _SENIOR_EXPERIENCE_RE.search(value):
            output.append("experience_band:higher_band")
    return _unique(output)


def experience_band_compatible(required: str, observed: Sequence[str]) -> bool:
    """Match a typed experience band, retaining operator, cutoff and unit.

    A family request has no invented cutoff. A precise request retains its
    operator and threshold and needs that exact source-owned representation.
    """
    parts = _experience_facet_key(required)
    if not parts:
        return False
    return any(
        key == parts or (len(parts) == 2 and len(key) == 5 and key[:2] == parts)
        for candidate in observed
        if (key := _experience_facet_key(candidate))
    )


def source_fact_facets(fact: Any) -> tuple[str, ...]:
    """The source-owned facet contract used by admission and final consumers.

    Only the fact's bounded source claim may prove a numeric experience bound.
    A broad category is retained for rendering. Multiple operators are allowed
    only when the same local category explicitly states them at one threshold;
    another series or duration cannot lend its bound to this fact.
    """
    facet = str(getattr(fact, "evidence_facet", "") or getattr(fact, "facet", "") or "")
    bands = experience_band_facets(str(getattr(fact, "source_claim", "") or ""))
    family = facet.split(":")[1] if facet.startswith("experience_band:") else facet
    if family in {"lower_band", "higher_band"}:
        if not bands or any(band.split(":")[1] != family for band in bands):
            return (family,)
        thresholds = {(parts[3], parts[4] if len(parts) == 5 else "years")
                      for band in bands if len(parts := band.split(":")) >= 4}
        if len(thresholds) > 1:
            return (family,)
        return _unique((family, *bands))
    return tuple(value for value in facet.split("|") if value)


def source_facet_compatible(required: str, observed: Sequence[str]) -> bool:
    return (experience_band_compatible(required, observed)
            if required.startswith("experience_band:") else required in observed)


def source_endpoint_digest(fact: Any) -> str:
    """Bind a derivation to the complete producer snapshot, not only its ID."""
    payload = asdict(fact)
    payload["source_facets"] = list(source_fact_facets(fact))
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
        separators=(",", ":")).encode("utf-8")).hexdigest()


def source_endpoint_order(fact: Any) -> tuple[int, str, str]:
    """The source period order used for direction construction and checks."""
    return (
        int(fact.period) if re.fullmatch(r"(?:19|20)\d{2}", fact.period)
        else 0 if fact.time_role == "historical" else 9999,
        fact.time_role, fact.fact_id,
    )


_POPULATION_DIRECTION_BOUNDARY = (
    r"(?:increased?|decreased?|declined?|rose|fell|grew|remained|stayed|did\s+not\s+change)"
    r"(?=\s+(?:from|to|by|between|in|during|approximately|about|stable|unchanged)\b|\s+[-+]?\d|$)|"
    r"(?:עלה|עלתה|ירד|ירדה|גדל|גדלה|פחת|פחתה|לא\s+השתנה)"
    r"(?=\s+(?:מ|ל|ב|כ|בין|בשנת|מכ|לכ)[־-]?|$)"
)


_POPULATION_SCOPE_ANCHORS = (
    r"\b(?:at|within|in|among|across|of\s+(?:the\s+)?(?:total|all))\s+(?=[A-Za-z])|"
    r"(?<![\u05d0-\u05ea])(?:בחברת|בחברה|בארגון|בארגונית|אצל)\s+|"
    r"(?<![\u05d0-\u05ea])(?:בקרב|מבין|מתוך|מכלל)\s+|"
    r"(?<![\u05d0-\u05ea])ב(?=חברת\s+)"
)


_POPULATION_CANONICAL_DECLARATION = re.compile(
    r"\((?i:canonical\s+entity\s+identifier)\s+"
    r"(?P<identifier>[A-Za-z][A-Za-z0-9_-]{0,63}(?:/[A-Za-z][A-Za-z0-9_-]{0,63})+)\)"
)


def _without_population_identity_declaration(scope: str) -> str:
    """Remove the source declaration only for local label/scope matching."""
    return normalize_semantic_text(_POPULATION_CANONICAL_DECLARATION.sub("", scope))


def source_population_scope(value: str, *, value_span: Optional[tuple[int, int]] = None) -> str:
    """Extract a literal, proposition-local measured population restriction.

    A company attached to the measured workforce is a scope owner; a company
    named only as a funder, publisher or author is not. The returned phrase is
    kept literal so the evidence context can authenticate its source span.
    """
    text = value or ""
    if value_span is not None:
        start, finish = value_span
        if not 0 <= start < finish <= len(text):
            return ""
        boundary = 0
        for coordination in re.finditer(r"\b(?:and|while|whereas)\b|(?:בעוד|ואילו)\s+", text[:start], re.I):
            preceding = text[:coordination.start()]
            prior_value = any(match.group("unit") for match in _NUMBER_RE.finditer(preceding)) or bool(re.search(
                r"\b(?:was|were|is|are)\s+(?:approximately\s+|about\s+)?\d|"
                r"(?:היה|הייתה|היו|הוא|היא)\s+(?:כ[-־]?\s*)?\d", preceding, re.I,
            ))
            remainder = text[coordination.end():]
            owned_scope = source_population_scope(remainder)
            scope_position = remainder.casefold().find(owned_scope.casefold()) if owned_scope else -1
            if prior_value and 0 <= scope_position < start - coordination.end():
                boundary = coordination.end()
        if boundary:
            text = text[boundary:].lstrip()
    owner = re.search(
        r"\b(?P<scope>(?:(?:[A-Z][A-Za-z0-9'’-]*\s+){0,4}[A-Z][A-Za-z0-9’'-]*|"
        r"[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea0-9 '’׳/-]{0,160}"
        r"(?:\([^,.;:!?]*\)\s*)?\((?i:canonical\s+entity\s+identifier)\s+"
        r"[A-Za-z][A-Za-z0-9_/-]*\))['’]s\s+"
        r"(?:(?:share|percentage|rate|proportion)\s+of\s+)?"
        r"(?:demand|workforce|headcount|employees|workers|revenue|profit|sales)\b[^,.;:!?]*)", text,
    )
    if owner:
        phrase = re.split(r"\s+(?:" + _POPULATION_DIRECTION_BOUNDARY + r"|was|were|is|are|in\s+(?:19|20)\d{2}|during\s+|"
                          r"between\s+(?:19|20)\d{2}|from\s+(?:19|20)\d{2})\b", owner.group("scope"),
                          maxsplit=1, flags=re.I)[0]
        phrase = phrase.strip()
        while phrase.endswith(")") and phrase.count(")") > phrase.count("("):
            phrase = phrase[:-1].rstrip()
        return phrase
    anchors = re.finditer(_POPULATION_SCOPE_ANCHORS, text, re.I)
    for anchor in anchors:
        if re.match(r"(?:least|most)\s+(?:\d+(?:\.\d+)?|" + "|".join(_EXPERIENCE_NUMBER_WORDS) + r")\b", text[anchor.end():], re.I):
            continue
        prefix = text[:anchor.start()]
        # Publication location, research setting and funding provenance do
        # not identify the measured population.
        if re.search(r"\b(?:published|funded|financed|sponsored|authored|written|issued)"
                     r"[^,;:.]{0,48}$|(?:פורסם|פורסמה|מומן|מומנה|נכתב|נכתבה)[^,;:.]{0,48}$", prefix, re.I):
            continue
        suffix = text[anchor.end():]
        end = re.search(
            r"[,;:.!?]|\s+(?:" + _POPULATION_DIRECTION_BOUNDARY + r"|was|were|is|are|stood|amounted|accounted|"
            r"represented|constituted|in\s+(?:19|20)\d{2}|during\s+|"
            r"between\s+(?:19|20)\d{2}|from\s+(?:19|20)\d{2}|"
            r"היה|הייתה|היו|הוא|היא|עמד|עמדה|בשנת|בשנים|בשנה)\b|"
            r"\s+(?=[-+]?\d+(?:\.\d+)?\s*%)", suffix, re.I,
        )
        phrase = text[anchor.start():anchor.end() + (end.start() if end else len(suffix))].strip()
        while phrase.endswith(")") and phrase.count(")") > phrase.count("("):
            phrase = phrase[:-1].rstrip()
        owner = text[anchor.end():anchor.end() + (end.start() if end else len(suffix))].strip()
        if not owner or re.fullmatch(r"(?:the\s+)?(?:past|present|future|total|all)|(?:עבר|הווה|עתיד)", owner, re.I):
            continue
        # A locative phrase is relevant only when it governs a measured
        # entity/quantity. Do not promote arbitrary document organization
        # mentions into population semantics.
        measured_context = prefix
        if not prefix.strip() and end and suffix[end.start():].startswith(","):
            remainder = re.split(r"[.;:!?]", suffix[end.start() + 1:], maxsplit=1)[0]
            if (re.search(r"\b(?:was|were|is|are)\b|(?:היה|הייתה|היו|הוא|היא)", remainder, re.I)
                    and re.search(r"\b(?:was|were|is|are)\s+"
                                  r"(?:(?:approximately|about|around|roughly)\s+)?[-+]?\d+(?:[.,]\d+)?|"
                                  r"(?:היה|הייתה|היו|הוא|היא)\s+"
                                  r"(?:(?:בערך|בקירוב)\s+|כ[־-]?\s*)?[-+]?\d+(?:[.,]\d+)?", remainder, re.I)):
                measured_context = remainder
        if not re.search(
            r"\b(?:demand|workers?|employees?|workforce|customers?|respondents?|"
            r"participants?|households?|vacancies|jobs?|sample|cohort|population|"
            r"share|percentage|rate|revenue|profit|sales|headcount)\b|"
            r"(?:ביקוש|עובד|מועסק|משרות|לקוחות|משיבים|משתתפים|מדגם|אוכלוס|שיעור|הכנס|רווח)", measured_context, re.I,
        ):
            continue
        return phrase
    # An explicit denominator/cohort is also a population restriction.
    denominator = re.search(
        r"\b(?:among|within|across|of\s+(?:the\s+)?(?:total|all))\s+"
        r"(?P<scope>[^,.;:!?]{2,180})|(?:בקרב|מבין|מתוך|מכלל)\s+"
        r"(?P<scope_he>[^,.;:!?]{2,180})", text, re.I,
    )
    if denominator:
        phrase = denominator.group(0)
        phrase = re.split(r"\s+(?:" + _POPULATION_DIRECTION_BOUNDARY + r"|was|were|is|are|in\s+(?:19|20)\d{2}|"
                          r"היה|הייתה|היו|הוא|היא|בשנת|בשנים)\b", phrase, maxsplit=1, flags=re.I)[0]
        return phrase.strip()
    return ""


def _aggregate_population_identity(value: str) -> str:
    """Recognize a complete generic denominator, retaining its quantity."""
    text = normalize_semantic_text(value).casefold().strip(" ()")
    english = re.fullmatch(r"of\s+(?:the\s+)?(?:all|total)\s+(?P<quantity>"
                           r"demand\s+for\s+(?:workers|employees|jobs)|"
                           r"workers|employees|jobs)", text)
    hebrew = re.fullmatch(r"(?:מכלל|מתוך\s+(?:כלל|סך))\s+(?P<quantity>"
                          r"ה?ביקוש\s+ל(?:ה?עובדים|ה?משרות)|ה?עובדים|ה?משרות)", text)
    if not english and not hebrew:
        return ""
    quantity = (english or hebrew).group("quantity")
    demand = bool(re.search(r"\bdemand\b|ביקוש", quantity))
    workers = bool(re.search(r"\b(?:workers|employees)\b|עובדים", quantity))
    return ("demand_" if demand else "") + ("workers" if workers else "jobs")


def render_source_population_scope(scope: str, language: str) -> str:
    """Translate only a fully resolved generic denominator; retain names."""
    identity = _aggregate_population_identity(scope)
    vocabulary = {
        "workers": ("of all workers", "מכלל העובדים"),
        "jobs": ("of all jobs", "מכלל המשרות"),
        "demand_workers": ("of all demand for workers", "מכלל הביקוש לעובדים"),
        "demand_jobs": ("of all demand for jobs", "מכלל הביקוש למשרות"),
    }
    target = str(language).casefold()
    if identity not in vocabulary or target not in {"en", "english", "he", "hebrew"}:
        return scope
    return vocabulary[identity][0 if target in {"en", "english"} else 1]


def _population_scope_key(value: str) -> str:
    if aggregate := _aggregate_population_identity(value):
        return "total_denominator:" + aggregate
    value = _experience_comparison_subject(_without_population_identity_declaration(value))
    value = re.sub(r"\bfor\s+(?:the\s+)?(?:workers|employees)\b", "", value, flags=re.I)
    value = normalize_semantic_text(value).casefold().strip(" ,;:.?!")
    value = re.sub(r"^(?:at|in|within|among|across)\s+(?:the\s+)?", "", value)
    value = re.sub(r"^the\s+", "", value)
    value = re.sub(r"^(?:בחברת|בחברה|בארגון|אצל|בקרב|מבין|מתוך)\s+", "", value)
    return re.sub(r"\s+", " ", value).strip(" ()")


def _population_scope_definition(scope: str) -> Optional[re.Match[str]]:
    """Return the established, literal local alias-definition grammar."""
    scope = _without_population_identity_declaration(scope)
    definition = re.fullmatch(
        r"(?P<name>.+?)\s*\((?:(?:also\s+known\s+as|also\s+called|aka|hereafter|"
        r"להלן|המכונה)\s+)?(?P<alias>[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea0-9 '׳-]{0,64})\)"
        r"(?P<qualifier>(?:\s+(?:among|within|across|of|בקרב|מתוך|מכלל)\s+.+)?)", scope, re.I,
    )
    if definition and (
        re.search(r"\((?:also\s+known\s+as|also\s+called|aka|hereafter|להלן|המכונה)\s+", scope, re.I)
        or re.fullmatch(r"[A-Z][A-Z0-9]{1,9}", definition.group("alias"))
    ):
        return definition
    return None


def _population_scope_aliases(scope: str) -> frozenset[str]:
    """Literal rendering labels; these are not by themselves entity identity."""
    if definition := _population_scope_definition(scope):
        qualifier = definition.group("qualifier")
        keys = {_population_scope_key(definition.group("name") + qualifier),
                _population_scope_key(definition.group("alias") + qualifier)}
    else:
        keys = {_population_scope_key(scope)}
    return frozenset(key for key in keys if key)


class PopulationIdentityKind(str, Enum):
    RESOLVED_NAMED = "resolved_named"
    RESOLVED_AGGREGATE = "resolved_aggregate"
    LOCALLY_RESOLVED_REFERENCE = "locally_resolved_reference"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class PopulationIdentifierDeclaration:
    """One literal declaration and the source argument that owns it.

    Spans are relative to the parsed scope until its source provenance is
    authenticated; source proofs rebase them to the structural evidence unit.
    Participant declarations are retained for audit, never as owner bridges.
    """
    identifier: str
    owner: str
    owner_span: tuple[int, int]
    declaration_span: tuple[int, int]
    measured_owner: bool = False


@dataclass(frozen=True)
class PopulationIdentityProof:
    """A source-local identity decision, separate from numerical compatibility.

    Keys are emitted only for an affirmative naming/domain construction. The
    source binding is retained for audit and for bounded reference resolution;
    source IDs, hashes and structural IDs never themselves create identity.
    """
    kind: PopulationIdentityKind = PopulationIdentityKind.UNRESOLVED
    keys: frozenset[str] = frozenset()
    named_referents: frozenset[str] = frozenset()
    definition: bool = False
    source_basis: tuple[str, ...] = ()
    source_id: str = ""
    structural_unit_id: str = ""
    canonical_referents: frozenset[str] = frozenset()
    # Label/coreference keys are local. Only these separately authenticated
    # declarations establish identity between independent source documents.
    cross_document_keys: frozenset[str] = frozenset()
    # Unit-local exact source spans, parallel to source_basis.
    source_basis_spans: tuple[tuple[int, int], ...] = ()
    # Exact source-container coordinates when the adapter emitted one lossless
    # source span. These join overlapping authenticated structural adapters.
    source_basis_origin_spans: tuple[tuple[str, int, int], ...] = ()
    measured_owner: str = ""
    measured_owner_span: Optional[tuple[int, int]] = None
    identity_declarations: tuple[PopulationIdentifierDeclaration, ...] = ()
    # Source-owned labels/aliases for rendering or request selection. Audit
    # subspans for other actors must never acquire this separate authority.
    local_reference_keys: frozenset[str] = frozenset()


_POPULATION_PROPER_TOKEN = r"[A-Z][a-z][A-Za-z0-9'’׳-]*"
_POPULATION_NAMED_FORM = r"(?:Company|Corporation|Corp|Ltd|Limited|Inc|LLC|PLC|Works|Labs)"
_POPULATION_COLLECTIVE = r"(?:workers|employees|jobs|customers|respondents|participants|households)"
_POPULATION_COHORT = r"(?:(?:all|total|permanent|temporary|salaried|full-time|part-time)\s+)?" + _POPULATION_COLLECTIVE


def _population_anonymous_name(value: str) -> bool:
    """An explicit anonymity/deictic assertion vetoes naming syntax."""
    return bool(re.search(r"\b(?:the|a|an|this|that|these|those|their|its|our|your|"
                          r"unidentified|unnamed|unknown|ambiguous|unspecified|same|other|another|relevant|"
                          r"northern|southern|eastern|western|regional|local|nearby|surrounding)\b", value, re.I))


def _named_population_head(value: str) -> bool:
    """Locate a supported local owner label; never prove global identity."""
    name = normalize_semantic_text(value).strip()
    if _population_anonymous_name(name):
        return False  # Determiner-led descriptions are not naming syntax.
    if re.fullmatch(_POPULATION_NAMED_FORM, name):
        return False  # A type label alone supplies no entity name.
    if re.fullmatch(_POPULATION_PROPER_TOKEN, name):
        return True  # A bare proper noun in an owner position, e.g. at Cedar.
    if re.fullmatch(rf"(?:{_POPULATION_PROPER_TOKEN}\s+){{1,4}}{_POPULATION_NAMED_FORM}", name):
        return True
    if re.fullmatch(rf"(?:Company|Corporation)\s+{_POPULATION_PROPER_TOKEN}(?:\s+{_POPULATION_PROPER_TOKEN}){{0,3}}", name):
        return True
    return False


def _typed_population_name(value: str) -> bool:
    """A typed entity designation can disambiguate a bare alias name."""
    name = re.sub(r"^(?i:at|in|within)\s+", "", normalize_semantic_text(value))
    return bool(re.fullmatch(rf"(?:{_POPULATION_PROPER_TOKEN}\s+){{1,4}}{_POPULATION_NAMED_FORM}", name)
                or re.fullmatch(rf"(?:Company|Corporation)\s+{_POPULATION_PROPER_TOKEN}(?:\s+{_POPULATION_PROPER_TOKEN}){{0,3}}", name)
                or re.fullmatch(r"(?:בחברת|חברת)\s+[\u05d0-\u05ea][\u05d0-\u05ea '׳-]*", name))


def _population_owner_parts(scope: str) -> tuple[str, str]:
    """Locate a measured owner and retain its independently owned qualifier."""
    text = _experience_comparison_subject(_without_population_identity_declaration(scope))
    text = re.sub(r"\bfor\s+(?:the\s+)?(?:workers|employees)\b", "", text, flags=re.I)
    text = normalize_semantic_text(text).strip(" ,;:.?!")
    if definition := _population_scope_definition(text):
        text = definition.group("name") + definition.group("qualifier")
    owner = re.sub(r"^(?i:at|in|within|בחברת|חברת)\s+", "", text)
    possessive = re.fullmatch(
        r"(?P<name>.+?)['’]s\s+(?:(?:share|percentage|rate|proportion)\s+of\s+)?"
        r"(?:demand|workforce|headcount|employees|workers|revenue|profit|sales)"
        r"(?:\s+for\s+(?:workers|employees))?"
        r"(?P<qualifier>(?:\s+(?:among|within|across|in|of)\s+.+)?)", owner, re.I,
    )
    if possessive:
        return possessive.group("name"), possessive.group("qualifier").strip()
    pieces = re.split(r"\s+(?=(?i:among|within|across|in|of|and|or)\s)", owner, maxsplit=1)
    return pieces[0], pieces[1] if len(pieces) == 2 else ""


def _population_qualifier_key(qualifier: str) -> Optional[str]:
    if not qualifier:
        return ""
    if re.fullmatch(r"(?i:in\s+(?:(?:region|country|city)\s+)?)"
                    r"[A-Za-z][A-Za-z0-9'’-]*(?:\s+[A-Za-z][A-Za-z0-9'’-]*){0,3}", qualifier):
        return _population_scope_key(qualifier)
    if _positive_population_scope(qualifier).kind is PopulationIdentityKind.RESOLVED_AGGREGATE:
        return _population_scope_key(qualifier)
    return None


def _population_name_argument(text: str, start: int) -> Optional[tuple[int, int]]:
    """Positively close one supported source entity designation.

    A typed or bare owner has one source name atom; an organization-type marker
    closes that designation before any later participant. An undelimited series
    of words cannot establish whether it is a compound name or several actors.
    Case and the absence of a recognized connector do not resolve that ambiguity.
    """
    atom = r"[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea0-9'’׳/-]*"
    kind = rf"(?i:{_POPULATION_NAMED_FORM})"
    tail = text[start:]
    typed = re.match(rf"(?!{kind}\b){atom}\s+{kind}(?=\s|\(|['’]s|$)", tail)
    if typed:
        return start, start + typed.end()
    type_first = re.match(rf"{kind}\s+(?P<name>{atom})(?=\s|\(|['’]s|$)", tail)
    if type_first:
        # A prefix type has no following type terminator to close a compound
        # name. Only its atomic designation is unambiguous in this source form.
        return start, start + type_first.end()
    bare = re.match(atom, tail)
    return (start, start + bare.end()) if bare else None


def _population_argument_annotations(
    scope: str, owner_span: tuple[int, int], *, measured_owner: bool,
) -> tuple[int, tuple[PopulationIdentifierDeclaration, ...]]:
    """Read only annotations in this already closed entity argument."""
    position = owner_span[1]
    declarations = []
    alias_seen = False
    while True:
        whitespace = re.match(r"\s*", scope[position:])
        candidate = position + whitespace.end()
        declaration = _POPULATION_CANONICAL_DECLARATION.match(scope, candidate)
        if declaration is not None:
            declarations.append(PopulationIdentifierDeclaration(
                declaration.group("identifier"), scope[slice(*owner_span)],
                owner_span, declaration.span(), measured_owner,
            ))
            position = declaration.end()
            continue
        if not alias_seen and scope[candidate:candidate + 1] == "(":
            close = scope.find(")", candidate + 1)
            # An alias definition belongs to the closed owner. It never
            # extends that owner across an intervening participant argument.
            if close >= 0 and _population_scope_definition(scope[owner_span[0]:close + 1]) is not None:
                position = close + 1
                alias_seen = True
                continue
        return candidate, tuple(declarations)


def _population_role_arguments(
    scope: str, start: int,
) -> Optional[tuple[PopulationIdentifierDeclaration, ...]]:
    """Parse separately designated participants using the existing role schema.

    The role noun positively marks the following closed argument as a document
    participant. Its connective does not decide canonical owner attachment.
    All remaining text must belong to these bounded participant constructions.
    """
    role_nouns = "|".join(_DOCUMENT_ROLE_NOUNS.values())
    declarations = []
    position = start
    while position < len(scope):
        role = re.match(rf"\s*(?:[A-Za-z-]+\s+)?(?:the\s+)?(?:{role_nouns})\s+", scope[position:], re.I)
        if role is None:
            return None
        owner_span = _population_name_argument(scope, position + role.end())
        if owner_span is None:
            return None
        position, attached = _population_argument_annotations(scope, owner_span, measured_owner=False)
        if len(attached) > 1:
            return None
        declarations.extend(attached)
    return tuple(declarations)


def _canonical_population_scope(scope: str) -> PopulationIdentityProof:
    """Select a declaration only from the positively closed measured owner.

    Syntactic selectors use this parse without claiming authentication. Source
    proofs additionally bind every selected owner/declaration span to the actual
    numeric fact's authenticated population container. No participant tag can
    substitute for a declaration inside the measured-owner argument.
    """
    locative = re.match(r"(?i:at|in|within|בחברת|חברת)\s+", scope)
    start = locative.end() if locative else 0
    owner_span = _population_name_argument(scope, start)
    if owner_span is None:
        return PopulationIdentityProof()
    owner = scope[slice(*owner_span)]
    end, owner_declarations = _population_argument_annotations(scope, owner_span, measured_owner=True)
    remainder = scope[end:]
    qualifier = ""
    if not locative:
        possessive = re.fullmatch(
            r"['’]s\s+(?:(?:share|percentage|rate|proportion)\s+of\s+)?"
            r"(?:demand|workforce|headcount|employees|workers|revenue|profit|sales)"
            r"(?:\s+for\s+(?:workers|employees))?"
            r"(?P<qualifier>(?:\s+(?:among|within|across|in|of)\s+.+)?)",
            _experience_comparison_subject(remainder), re.I,
        )
        if possessive is None:
            return PopulationIdentityProof()
        qualifier = possessive.group("qualifier").strip()
        participants = ()
    elif not remainder or _population_qualifier_key(remainder) is not None:
        qualifier = remainder
        participants = ()
    else:
        participants = _population_role_arguments(scope, end)
    declarations = list(owner_declarations)
    if participants is not None:
        declarations.extend(participants)
    known_spans = {item.declaration_span for item in declarations}
    # Retain unmatched declarations as unbound evidence, rather than silently
    # attributing them to the nearest owner or an entire pre-tag string.
    declarations.extend(PopulationIdentifierDeclaration(
        match.group("identifier"), "", (-1, -1), match.span(), False,
    ) for match in _POPULATION_CANONICAL_DECLARATION.finditer(scope)
        if match.span() not in known_spans)
    audit = dict(measured_owner=owner, measured_owner_span=owner_span,
                 identity_declarations=tuple(declarations))
    qualifier_key = _population_qualifier_key(qualifier)
    if (len(owner_declarations) != 1 or participants is None or qualifier_key is None
            or any(not declaration.owner for declaration in declarations)):
        return PopulationIdentityProof(**audit)
    declaration = owner_declarations[0]
    # Only the closed owner and its independently supported population
    # restriction supply labels; other actors do not become owner aliases.
    owner_scope = _without_population_identity_declaration(scope[:end])
    literal = normalize_semantic_text(owner_scope + (" " + qualifier if qualifier else ""))
    label_proof = _positive_population_scope(literal)
    identity = "entity_identifier:" + declaration.identifier
    bridge = identity + ("|population:" + qualifier_key if qualifier_key else "")
    return PopulationIdentityProof(
        PopulationIdentityKind.RESOLVED_NAMED,
        label_proof.keys or _population_scope_aliases(literal),
        label_proof.named_referents or frozenset({_population_scope_key(owner)}), True,
        canonical_referents=frozenset({identity}), cross_document_keys=frozenset({bridge}),
        local_reference_keys=_population_scope_aliases(literal),
        **audit,
    )


def _positive_population_scope(scope: str) -> PopulationIdentityProof:
    """Classify local owner/domain grammar; no label supplies a global bridge.

    This parser is deliberately closed: a new generic owner vocabulary does
    not require a new rejection expression. Modifiers can restrict a resolved
    owner/domain, but cannot provide a missing owner or domain.
    """
    if _aggregate_population_identity(scope):
        return PopulationIdentityProof(PopulationIdentityKind.RESOLVED_AGGREGATE,
                                       frozenset({_population_scope_key(scope)}))
    text = _experience_comparison_subject(scope)
    text = re.sub(r"\bfor\s+(?:the\s+)?(?:workers|employees)\b", "", text, flags=re.I)
    text = normalize_semantic_text(text).strip(" ,;:.?!")
    if not text:
        return PopulationIdentityProof()
    if _aggregate_population_identity(text):
        return PopulationIdentityProof(PopulationIdentityKind.RESOLVED_AGGREGATE,
                                       frozenset({_population_scope_key(text)}))
    if definition := _population_scope_definition(text):
        primary = _positive_population_scope(definition.group("name") + definition.group("qualifier"))
        if primary.kind is not PopulationIdentityKind.RESOLVED_NAMED:
            return PopulationIdentityProof()
        alias = _positive_population_scope(definition.group("alias") + definition.group("qualifier"))
        # A generic alias is a local rendering label, not a cross-source key.
        local_alias = bool(re.search(r"\((?:hereafter|להלן)\s+", text, re.I))
        alias_keys = alias.keys if alias.kind is PopulationIdentityKind.RESOLVED_NAMED and not local_alias else frozenset()
        alias_names = alias.named_referents if alias_keys else frozenset()
        # A bare shared alias such as Atlas does not override its explicitly
        # typed referent. Supplemental aliases of the same identified company
        # do not, in turn, create a contradictory canonical identity.
        canonical = primary.named_referents
        if (alias_names and not _typed_population_name(definition.group("name"))
                and _typed_population_name(definition.group("alias"))):
            canonical = alias_names
        return PopulationIdentityProof(primary.kind, primary.keys | alias_keys,
                                       primary.named_referents | alias_names, True,
                                       canonical_referents=canonical)
    collective = re.fullmatch(rf"(?:among|within|across)\s+(?:the\s+)?{_POPULATION_COHORT}", text, re.I)
    if collective:
        return PopulationIdentityProof(PopulationIdentityKind.RESOLVED_AGGREGATE,
                                       frozenset({_population_scope_key(text)}))
    domain_scope = re.fullmatch(r"(?i:in|within|across)\s+(?i:region|country|city|industry|sector)\s+"
                               r"(?P<label>[A-Za-z][A-Za-z0-9'’-]*)", text)
    if domain_scope and not _population_anonymous_name(domain_scope.group("label")):
        return PopulationIdentityProof(PopulationIdentityKind.RESOLVED_AGGREGATE,
                                       frozenset({_population_scope_key(text)}))
    bounded = re.fullmatch(
        rf"(?i:(?:among|within|across)\s+(?:companies|firms|corporations|organizations)\s+(?:in|within|of)\s+)"
        rf"(?:(?i:region|country|city|industry|sector)\s+(?P<label>[A-Za-z][A-Za-z0-9'’-]*)|"
        rf"(?P<domain>{_POPULATION_PROPER_TOKEN}(?:\s+{_POPULATION_PROPER_TOKEN}){{0,3}}))", text,
    )
    if bounded and not _population_anonymous_name(bounded.group("label") or bounded.group("domain")):
        return PopulationIdentityProof(PopulationIdentityKind.RESOLVED_AGGREGATE,
                                       frozenset({_population_scope_key(text)}))
    # Hebrew explicitly marks a company name with the construct חברת. The
    # definite/deictic forms חברת החברה / חברת זו do not assert a name.
    hebrew = re.fullmatch(r"(?:(?:בחברת|חברת)\s+)(?P<name>[\u05d0-\u05ea][\u05d0-\u05ea'׳-]*(?:\s+[\u05d0-\u05ea][\u05d0-\u05ea'׳-]*){0,3})", text)
    if hebrew and not re.search(r"\b(?:ה?חברה|ה?ארגון|זו|זה|הזו|הזה|אותה|אותו|"
                                r"אלמוני|אלמונית|האלמוני|האלמונית|לא\s+מזוהה|בלתי\s+מזוהה|לא\s+ידוע|לא\s+ידועה)\b", hebrew.group("name")):
        key = _population_scope_key(text)
        return PopulationIdentityProof(PopulationIdentityKind.RESOLVED_NAMED,
                                       frozenset({key}), frozenset({key}))
    owner = re.sub(r"^(?i:at|in|within)\s+", "", text)
    # A possessive construction locates its workforce-owner label.
    possessive = re.fullmatch(
        r"(?P<name>.+?)['’]s\s+(?:(?:share|percentage|rate|proportion)\s+of\s+)?"
        r"(?:demand|workforce|headcount|employees|workers|revenue|profit|sales)"
        r"(?:\s+for\s+(?:workers|employees))?"
        r"(?P<qualifier>(?:\s+(?:among|within|across|in|of)\s+.+)?)", owner,
    )
    if possessive and _named_population_head(possessive.group("name")):
        # The possessive locates a local owner label. Its
        # measured quantity stays in the fact contract, and its own scope
        # qualifiers must still independently satisfy the population grammar.
        return _positive_population_scope(possessive.group("name") + possessive.group("qualifier"))
    pieces = re.split(r"\s+(?=(?i:among|within|across|in|of|and|or)\s)", owner, maxsplit=1)
    if not _named_population_head(pieces[0]):
        return PopulationIdentityProof()
    if len(pieces) == 2:
        qualifier = pieces[1]
        qualified_domain = re.fullmatch(r"(?i:in\s+(?:(?:region|country|city)\s+)?)"
                                         r"(?P<name>[A-Za-z][A-Za-z0-9'’-]*(?:\s+[A-Za-z][A-Za-z0-9'’-]*){0,3})", qualifier)
        if qualified_domain and _population_anonymous_name(qualified_domain.group("name")):
            return PopulationIdentityProof()
        if (not qualified_domain
                and _positive_population_scope(qualifier).kind is not PopulationIdentityKind.RESOLVED_AGGREGATE):
            return PopulationIdentityProof()
    key = _population_scope_key(text)
    return PopulationIdentityProof(PopulationIdentityKind.RESOLVED_NAMED,
                                   frozenset({key}), frozenset({_population_scope_key(pieces[0])}))


def _unresolved_population_reference(scope: str) -> bool:
    """No affirmative identity construction means unresolved, by default."""
    return bool(scope and _positive_population_scope(scope).kind is PopulationIdentityKind.UNRESOLVED)


def _resolved_population_scope_keys(scope: str) -> frozenset[str]:
    """Only affirmative naming/domain grammar can supply identity keys."""
    return _positive_population_scope(scope).keys


def _unrestricted_population_domain(claim: str) -> str:
    """Recognize a complete unrestricted aggregate proposition.

    Merely finding a demand phrase cannot prove that an unknown owner is
    absent. This closed sentence grammar admits the supported scalar/series
    construction after independently owned experience facets are removed.
    Values and periods do not enter the identity key or their compatibility.
    """
    text = _experience_comparison_subject(claim)
    english_subject = r"demand\s+for\s+(?P<kind>workers|employees|jobs)"
    hebrew_subject = r"(?:שיעור|חלק)\s+(?:של\s+)?ה?ביקוש\s+ל(?P<kind_he>ה?עובדים|ה?משרות)"
    scalar = (r"(?:(?:approximately|about|around|roughly|בערך|בקירוב)\s+|כ[-־]?\s*)?"
              r"[-+]?\d+(?:[.,]\d+)?\s*(?:%|percent|אחוז(?:ים)?)\s+"
              r"(?:in|during|בשנת|ב[-־]?)\s*(?:19|20)\d{2}")
    predicate = r"(?:was|were|is|are|היה|הייתה|היו|הוא|היא)"
    source = re.fullmatch(rf"(?:{english_subject}|{hebrew_subject})\s+{predicate}\s+{scalar}"
                          rf"(?:\s+(?:and|ו[-־]?)\s*{scalar})?", text, re.I)
    if source is None:
        return ""
    quantity = source.group("kind") or source.group("kind_he")
    return "jobs" if re.fullmatch(r"jobs|ה?משרות", quantity, re.I) else "workers"


def _attached_demand_denominator_proof(
    scope: str, claim: str, population_span: tuple[int, int],
) -> PopulationIdentityProof:
    """Resolve a short denominator from its own complete copular relation.

    The caller authenticates the local proposition and the retained population
    span. A domain elsewhere in the structural unit cannot supply this proof.
    Experience facets may be removed; independent owners and other modifiers
    remain and must satisfy the closed relation grammar.
    """
    if not re.fullmatch(r"(?:מכלל|מתוך\s+(?:כלל|סך))\s+ה?ביקוש",
                        normalize_semantic_text(scope)):
        return PopulationIdentityProof()
    start, end = population_span
    prefix = _experience_comparison_subject(claim[:start])
    suffix = _experience_comparison_subject(claim[end:])
    domain = r"ה?עובדים|ה?משרות"
    scalar = r"(?:(?:בערך|בקירוב)\s+|כ[-־]?\s*)?[-+]?\d+(?:[.,]\d+)?\s*%"
    year = r"בשנת\s+(?:19|20)\d{2}"
    heading = (rf"(?:(?:ו?(?:ירידה|עלייה|עליה|גידול|קיטון|שינוי)\s+ב|"
               rf"(?:שיעור|חלק)\s+(?:של\s+)?)ה?ביקוש\s+ל(?P<heading_domain>{domain})"
               rf"(?:\s+בהשוואה\s+ל(?:סקר|מחקר)\s+הקודם)?\s*)?")
    preceding = re.fullmatch(rf"{heading}\(?\s*{year}\s+{scalar}", prefix)
    attached = re.fullmatch(
        rf"(?:היה|הייתה|היו|הוא|היא)\s+ל(?P<domain>{domain})"
        rf"(?:\s+ו{year}\s+{scalar})?\s*\)?", suffix,
    )
    if preceding is None or attached is None:
        return PopulationIdentityProof()
    quantity = "jobs" if "משרות" in attached.group("domain") else "workers"
    declared_domain = preceding.group("heading_domain")
    if declared_domain and ("jobs" if "משרות" in declared_domain else "workers") != quantity:
        return PopulationIdentityProof()
    return PopulationIdentityProof(
        PopulationIdentityKind.RESOLVED_AGGREGATE,
        frozenset({"total_denominator:demand_" + quantity}),
        source_basis=(claim[start:end], claim[start:]),
    )


def _population_source_binding(fact: Any) -> Optional[tuple[Any, Any, Any, tuple[int, int]]]:
    """Authenticate a literal population even when its global identity is unknown."""
    scope = str(getattr(fact, "evidence_population", "") or "")
    claim = str(getattr(fact, "source_claim", "") or "")
    context = getattr(fact, "evidence_context", None)
    unit = getattr(context, "structural_evidence_unit", None)
    provenance = context.provenance_for("evidence_population") if context is not None else None
    if (not scope or context is None or unit is None or provenance is None
            or structural_binding_validation_errors(context)
            or context.local_proposition != claim
            or context.proposition_sha256 != hashlib.sha256(claim.encode("utf-8")).hexdigest()
            or context.population != scope or provenance.value != scope
            or unit.local_text_sha256 != hashlib.sha256(unit.local_text.encode("utf-8")).hexdigest()
            or provenance.source_sha256 != hashlib.sha256(provenance.source_quote.encode("utf-8")).hexdigest()
            or normalize_semantic_text(scope).casefold() not in normalize_semantic_text(provenance.source_quote).casefold()):
        return None
    container = (context.local_proposition if provenance.source_container == "local_proposition"
                 else unit.local_text if provenance.source_container == "structural_parent_context" else "")
    start, end = provenance.source_span
    if (not container or not 0 <= start < end <= len(container)
            or container[start:end] != provenance.source_quote
            or provenance.structural_unit_id != unit.unit_id
            or provenance.structural_group_id != (unit.structural_group_id or unit.unit_id)):
        return None
    origin = context.proposition_span[0] if provenance.source_container == "local_proposition" else 0
    quote_span = (origin + start, origin + end)
    if not (0 <= quote_span[0] < quote_span[1] <= len(unit.local_text)
            and unit.local_text[quote_span[0]:quote_span[1]] == provenance.source_quote):
        return None
    # A wider authenticated field quote does not widen the governing owner.
    # Locate its unique exact literal scope before deriving identity or joins.
    if provenance.source_quote.count(scope) != 1:
        return None
    scope_position = provenance.source_quote.index(scope)
    span = (quote_span[0] + scope_position, quote_span[0] + scope_position + len(scope))
    return context, unit, provenance, span


def _population_basis_origin_spans(unit: Any, spans: tuple[tuple[int, int], ...]) -> tuple[tuple[str, int, int], ...]:
    """Map only a proven lossless source span, without joining source fragments."""
    source_spans = tuple(unit.source_spans)
    if len(source_spans) != 1:
        return ()
    source_span = source_spans[0]
    if (source_span.end - source_span.start != len(unit.local_text)
            or source_span.sha256 != unit.local_text_sha256):
        return ()
    return tuple((unit.source_container, source_span.start + start, source_span.start + end)
                 for start, end in spans)


def source_population_identity_proof(fact: Any) -> PopulationIdentityProof:
    """Authenticate local referents separately from cross-document identity."""
    scope = str(getattr(fact, "evidence_population", "") or "")
    proof = _positive_population_scope(scope)
    context = getattr(fact, "evidence_context", None)
    unit = getattr(context, "structural_evidence_unit", None)
    provenance = context.provenance_for("evidence_population") if context is not None else None
    if not scope:
        claim = str(getattr(fact, "source_claim", "") or "")
        if (context is None or unit is None or structural_binding_validation_errors(context)
                or context.population or context.local_proposition != claim
                or unit.local_text_sha256 != hashlib.sha256(unit.local_text.encode("utf-8")).hexdigest()
                or claim not in unit.local_text):
            return PopulationIdentityProof()
        quantity = _unrestricted_population_domain(claim)
        if not quantity:
            return PopulationIdentityProof()
        position = context.proposition_span[0]
        return PopulationIdentityProof(PopulationIdentityKind.RESOLVED_AGGREGATE,
                                       frozenset({"unrestricted_demand:" + quantity}),
                                       source_basis=(claim,), source_id=context.source_id,
                                       structural_unit_id=unit.unit_id,
                                       source_basis_spans=((position, position + len(claim)),))
    binding = _population_source_binding(fact)
    if binding is None or source_fact_population_errors(fact):
        return PopulationIdentityProof()
    context, unit, provenance, basis_span = binding
    origin = context.proposition_span[0] if provenance.source_container == "local_proposition" else 0
    start, end = basis_span[0] - origin, basis_span[1] - origin

    def bound(result: PopulationIdentityProof, extra: tuple[str, ...] = (),
              extra_spans: tuple[tuple[int, int], ...] = (),
              proof_origin: Optional[int] = None) -> PopulationIdentityProof:
        offset = basis_span[0] if proof_origin is None else proof_origin
        def rebase(span: tuple[int, int]) -> tuple[int, int]:
            return (offset + span[0], offset + span[1]) if span[0] >= 0 else span
        measured_span = (rebase(result.measured_owner_span)
                         if result.measured_owner_span is not None else None)
        declarations = tuple(replace(declaration,
            owner_span=rebase(declaration.owner_span),
            declaration_span=rebase(declaration.declaration_span))
            for declaration in result.identity_declarations)
        return replace(result, source_basis=(scope, *extra),
                       source_basis_spans=(basis_span, *extra_spans),
                       source_basis_origin_spans=_population_basis_origin_spans(unit, (basis_span, *extra_spans)),
                       measured_owner_span=measured_span, identity_declarations=declarations,
                       source_id=context.source_id, structural_unit_id=unit.unit_id)

    claim = str(getattr(fact, "source_claim", "") or "")
    if provenance.source_container == "local_proposition" and context.local_proposition == claim:
        attached = _attached_demand_denominator_proof(scope, claim, (start, end))
        if attached.keys:
            spans = tuple((origin + claim.index(literal), origin + claim.index(literal) + len(literal))
                          for literal in attached.source_basis)
            return replace(attached, source_id=context.source_id, structural_unit_id=unit.unit_id,
                           source_basis_spans=spans,
                           source_basis_origin_spans=_population_basis_origin_spans(unit, spans))
    canonical = _canonical_population_scope(scope)
    if canonical.identity_declarations:
        # Every proof-bearing subspan must be inside the same authenticated
        # numeric population scope. Select the parsed owner tag, never .search's
        # first arbitrary declaration, and preserve other actors separately.
        proof_spans = []
        if canonical.measured_owner_span is not None:
            proof_spans.append(canonical.measured_owner_span)
        for declaration in canonical.identity_declarations:
            if declaration.owner_span[0] >= 0:
                proof_spans.append(declaration.owner_span)
            proof_spans.append(declaration.declaration_span)
        proof_spans = tuple(dict.fromkeys(proof_spans))
        if any(not 0 <= begin < finish <= len(scope) for begin, finish in proof_spans):
            return PopulationIdentityProof()
        literals = tuple(scope[begin:finish] for begin, finish in proof_spans)
        absolute_spans = tuple((basis_span[0] + begin, basis_span[0] + finish)
                               for begin, finish in proof_spans)
        if any(unit.local_text[begin:finish] != literal
               for literal, (begin, finish) in zip(literals, absolute_spans)):
            return PopulationIdentityProof()
        return bound(canonical, literals, absolute_spans)
    # Syntax identifies only a bounded local label. It cannot establish an
    # entity bridge merely by appearing in another authenticated document.
    if proof.kind is PopulationIdentityKind.RESOLVED_NAMED:
        proof = replace(proof, kind=PopulationIdentityKind.LOCALLY_RESOLVED_REFERENCE)
    source_container = str(getattr(fact, "source_container_claim", "") or "")
    if (not source_container or source_container not in unit.local_text
            or _population_scope_definition(scope) is not None):
        return bound(proof) if proof.keys else replace(proof, source_id=context.source_id,
                                                        structural_unit_id=unit.unit_id)
    container_position = unit.local_text.find(source_container)
    current_key = _population_scope_key(scope)
    definitions = []
    for anchor in re.finditer(_POPULATION_SCOPE_ANCHORS, source_container, re.I):
        candidate_scope = source_population_scope(source_container[anchor.start():])
        candidate_definition = _population_scope_definition(candidate_scope)
        if candidate_definition is not None:
            alias_key = _population_scope_key(candidate_definition.group("alias"))
            if alias_key and (current_key == alias_key or current_key.startswith(alias_key + " ")):
                candidate_position = source_container.find(candidate_scope, anchor.start())
                if candidate_position >= 0:
                    definitions.append((candidate_scope, candidate_definition, alias_key, candidate_position))
    if not definitions:
        return bound(proof) if proof.keys else replace(proof, source_id=context.source_id,
                                                        structural_unit_id=unit.unit_id)
    owners = set()
    for candidate_scope, definition, _, _ in definitions:
        established = _canonical_population_scope(candidate_scope)
        primary = _positive_population_scope(definition.group("name"))
        if not established.cross_document_keys and primary.kind is not PopulationIdentityKind.RESOLVED_NAMED:
            return PopulationIdentityProof()
        owners.add(tuple(sorted(established.canonical_referents or primary.named_referents)))
    if len(owners) != 1:
        return PopulationIdentityProof()
    defining_scope, definition, alias_key, definition_position = definitions[0]
    name = definition.group("name")
    qualifier = current_key[len(alias_key):].strip()
    qualifier_key = _population_qualifier_key(qualifier)
    if qualifier_key is None:
        return PopulationIdentityProof()
    resolved = _positive_population_scope(name + (" " + qualifier if qualifier else ""))
    defining_proof = _canonical_population_scope(defining_scope)
    if not resolved.keys and not defining_proof.cross_document_keys:
        return PopulationIdentityProof()
    if not resolved.keys:
        resolved = replace(resolved, keys=_population_scope_aliases(name + (" " + qualifier if qualifier else "")))
    if not re.search(r"\((?:hereafter|להלן)\s+", defining_scope, re.I):
        resolved = replace(resolved, keys=resolved.keys | proof.keys)
    if not defining_proof.cross_document_keys:
        defining_proof = _positive_population_scope(defining_scope)
    bridge_keys = frozenset(identity + ("|population:" + qualifier_key if qualifier_key else "")
                            for identity in defining_proof.canonical_referents
                            if defining_proof.cross_document_keys)
    definition_span = (container_position + definition_position,
                       container_position + definition_position + len(defining_scope))
    if unit.local_text[slice(*definition_span)] != defining_scope:
        return PopulationIdentityProof()
    declaration_spans = []
    if defining_proof.measured_owner_span is not None:
        declaration_spans.append(defining_proof.measured_owner_span)
    for declaration in defining_proof.identity_declarations:
        if declaration.owner_span[0] >= 0:
            declaration_spans.append(declaration.owner_span)
        declaration_spans.append(declaration.declaration_span)
    declaration_spans = tuple(dict.fromkeys(declaration_spans))
    if any(not 0 <= begin < finish <= len(defining_scope)
           for begin, finish in declaration_spans):
        return PopulationIdentityProof()
    declaration_literals = tuple(defining_scope[begin:finish]
                                 for begin, finish in declaration_spans)
    absolute_declaration_spans = tuple((definition_span[0] + begin, definition_span[0] + finish)
                                      for begin, finish in declaration_spans)
    if any(unit.local_text[begin:finish] != literal for literal, (begin, finish)
           in zip(declaration_literals, absolute_declaration_spans)):
        return PopulationIdentityProof()
    return bound(replace(resolved, kind=PopulationIdentityKind.LOCALLY_RESOLVED_REFERENCE,
                         definition=True, named_referents=defining_proof.named_referents,
                         canonical_referents=defining_proof.canonical_referents,
                         cross_document_keys=bridge_keys,
                         measured_owner=defining_proof.measured_owner,
                         measured_owner_span=defining_proof.measured_owner_span,
                         identity_declarations=defining_proof.identity_declarations,
                         local_reference_keys=(defining_proof.local_reference_keys
                                               | _population_scope_aliases(scope)
                                               if defining_proof.cross_document_keys else frozenset())),
                 (defining_scope, *declaration_literals),
                 (definition_span, *absolute_declaration_spans), proof_origin=definition_span[0])


def source_population_identity_keys(fact: Any) -> frozenset[str]:
    """Authenticated local rendering/referent keys; never a global bridge."""
    return source_population_identity_proof(fact).keys


def _population_proof_owner_labels(proof: PopulationIdentityProof) -> frozenset[str]:
    """Select owner spellings only from already authenticated source labels."""
    return frozenset(_population_scope_key(_population_owner_parts(label)[0])
                     for label in proof.keys | proof.local_reference_keys | proof.named_referents
                     if label)


def source_population_matches_scope(required_scope: str, fact: Any) -> bool:
    """A named request must match an authenticated source identity or alias."""
    definition = _population_scope_definition(required_scope)
    required_base = (definition.group("name") + definition.group("qualifier")
                     if definition is not None else required_scope)
    # A requested identifier selects identity but never supplies its proof.
    if _POPULATION_CANONICAL_DECLARATION.search(required_scope):
        requested = _canonical_population_scope(required_scope)
        established = source_population_identity_proof(fact)
        return bool(requested.cross_document_keys & established.cross_document_keys
                    and requested.keys & established.keys
                    and _population_scope_key(requested.measured_owner)
                    in _population_proof_owner_labels(established))
    # A source-local request may select a literal or authenticated alias. This
    # choice does not create the global bridge required by a later trend.
    proof = source_population_identity_proof(fact)
    if not proof.keys:
        return False
    labels = set(proof.keys | proof.local_reference_keys)
    if not proof.identity_declarations:
        # Existing bounded local definitions remain selectable. Canonical
        # proofs instead carry labels from their parsed owner argument; their
        # audit basis also includes participant spans and is not alias authority.
        for literal in (str(getattr(fact, "evidence_population", "") or ""), *proof.source_basis):
            labels.update(_population_scope_aliases(literal))
    return bool(required_base and _population_scope_key(required_base) in labels)


def source_fact_population_errors(fact: Any) -> tuple[str, ...]:
    """Recompute local scope and require its admitted context to retain it."""
    scope = str(getattr(fact, "evidence_population", "") or "")
    context = getattr(fact, "evidence_context", None)
    claim = str(getattr(fact, "source_claim", "") or "")
    binding = getattr(fact, "numeric_binding", None)
    span = tuple(getattr(fact, "evidence_span", ()) or ())
    local_value_span = None
    if binding is not None and len(span) == 2:
        local_value_span = (binding.value_span[0] - span[0], binding.value_span[1] - span[0])
    local_scope = source_population_scope(claim, value_span=local_value_span)
    reasons = []
    if local_scope and not scope:
        reasons.append("source_population_scope_unbound")
    elif local_scope and normalize_semantic_text(local_scope).casefold() != normalize_semantic_text(scope).casefold():
        reasons.append("source_population_scope_mismatch")
    if scope and (str(getattr(fact, "population", "") or "") != scope
                  or _population_source_binding(fact) is None):
        reasons.append("source_population_context_unbound")
    return _unique(reasons)


def source_populations_compatible(left: Any, right: Any) -> bool:
    """An independent-document trend requires an affirmative identity bridge."""
    if source_fact_population_errors(left) or source_fact_population_errors(right):
        return False
    a, b = source_population_identity_proof(left), source_population_identity_proof(right)
    if not a.keys or not b.keys:
        return False
    if (a.kind is PopulationIdentityKind.RESOLVED_AGGREGATE
            or b.kind is PopulationIdentityKind.RESOLVED_AGGREGATE):
        return (a.kind is b.kind and bool(a.keys & b.keys))
    if a.cross_document_keys or b.cross_document_keys:
        return bool(a.cross_document_keys & b.cross_document_keys)
    if not a.source_id or a.source_id != b.source_id or not a.keys & b.keys:
        return False
    # Overlapping lossless adapters may expose the same bounded definition in
    # different structural units. Join exact original source coordinates,
    # never an equal spelling, neighboring statement or document-wide label.
    shared_origin = (set(zip(a.source_basis, a.source_basis_origin_spans))
                     & set(zip(b.source_basis, b.source_basis_origin_spans)))
    if shared_origin:
        return True
    if not a.structural_unit_id or a.structural_unit_id != b.structural_unit_id:
        return False
    shared_basis = set(zip(a.source_basis, a.source_basis_spans)) & set(zip(b.source_basis, b.source_basis_spans))
    return bool(shared_basis)


def rendered_population_scope_errors(answer: str, linked_facts: Sequence[Any]) -> tuple[str, ...]:
    """Material scope must remain attached to the released numeric claims."""
    reasons = []
    numeric_facts = tuple(fact for fact in linked_facts
                         if str(getattr(fact, "fact_type", "numeric")) in {"numeric", "derived_direction"})
    source_scopes = tuple(str(getattr(fact, "evidence_population", "") or "")
                          for fact in numeric_facts if getattr(fact, "evidence_population", ""))
    if not source_scopes:
        if source_population_scope(answer) and numeric_facts:
            reasons.append("rendered_population_scope_invented")
        return _unique(reasons)
    resolved_scopes = tuple(
        (str(getattr(fact, "evidence_population", "") or ""), source_population_identity_proof(fact))
        for fact in numeric_facts if getattr(fact, "evidence_population", "")
    )
    # A rendered aggregate cannot discard the source's company/cohort owner.
    # Check each material clause so a company mentioned elsewhere in the
    # answer cannot lend its identity to an unrelated numerical assertion.
    for clause in re.split(r"[.!?׃;](?:\s+|$)|\n+", _CLAIM_CITATION_RE.sub(" ", answer or "")):
        if not _numeric_values(clause) and not rendered_direction_mentioned(clause):
            continue
        rendered_scope = source_population_scope(clause)
        definition = _population_scope_definition(rendered_scope)
        primary_scope = (definition.group("name") + definition.group("qualifier")
                         if definition is not None else rendered_scope)
        rendered_keys = _positive_population_scope(primary_scope).keys
        rendered_proof = (_canonical_population_scope(rendered_scope)
                          if _POPULATION_CANONICAL_DECLARATION.search(rendered_scope) else None)
        claimed_aliases = (_population_scope_aliases(rendered_scope)
                           if definition is not None else frozenset())
        if rendered_proof is not None:
            claimed_aliases = frozenset()
            if rendered_proof.measured_owner_span is not None:
                owner_end, _ = _population_argument_annotations(
                    rendered_scope, rendered_proof.measured_owner_span, measured_owner=True)
                if _population_scope_definition(rendered_scope[:owner_end]) is not None:
                    claimed_aliases = rendered_proof.local_reference_keys
        matched = False
        for scope, proof in resolved_scopes:
            if not proof.keys:
                continue
            # Only authenticated source labels define owner/alias relationships.
            # Canonical audit spans may contain separately owned participants.
            labels = proof.keys | (proof.local_reference_keys if proof.identity_declarations
                                   else _population_scope_aliases(scope))
            if rendered_proof is not None:
                owner_matches = bool(
                    rendered_proof.cross_document_keys & proof.cross_document_keys
                    and _population_scope_key(rendered_proof.measured_owner)
                    in _population_proof_owner_labels(proof)
                    and rendered_proof.keys & labels)
            else:
                owner_matches = bool(rendered_keys & proof.keys
                                     or _population_scope_key(primary_scope) in labels)
            # Output definitions are claims, never new source authority. Check
            # both labels, retaining their restrictions under this same proof.
            if owner_matches and claimed_aliases <= labels:
                matched = True
                break
        if not rendered_scope or not matched:
            reasons.append("rendered_population_scope_unbound")
    return _unique(reasons)


def compatible_source_endpoints(left: Any, right: Any) -> tuple[str, ...]:
    """A direction can compare only one source-owned quantity and scope."""
    reasons = []
    for endpoint in (left, right):
        if (getattr(endpoint, "fact_type", "") != "numeric"
                or not getattr(endpoint, "structural_binding_valid", False)
                or getattr(endpoint, "evidence_atom", None) is None
                or getattr(endpoint, "numeric_binding", None) is None):
            reasons.append("derived_endpoint_source_binding_invalid")
            continue
        atom = endpoint.evidence_atom
        context = endpoint.evidence_context
        if (context is None or structural_binding_validation_errors(context)
                or atom.evidence_unit_id != endpoint.evidence_id
                or atom.candidate_id != endpoint.candidate_id
                or atom.value != endpoint.value or atom.unit != endpoint.unit
                or atom.period != endpoint.period
                or endpoint.evidence_period != endpoint.period
                or atom.numeric_binding != endpoint.numeric_binding):
            reasons.append("derived_endpoint_source_binding_invalid")
    for endpoint in (left, right):
        reasons.extend(source_fact_population_errors(endpoint))
    if not source_populations_compatible(left, right):
        proofs = (source_population_identity_proof(left), source_population_identity_proof(right))
        if any(not proof.keys for proof in proofs) or (
                proofs[0].source_id != proofs[1].source_id
                and any(proof.kind is not PopulationIdentityKind.RESOLVED_AGGREGATE
                        and not proof.cross_document_keys for proof in proofs)):
            reasons.append("source_population_identity_unresolved")
        reasons.append("derived_endpoint_population_incompatible")
    if not left.evidence_subject or not right.evidence_subject or not (
        canonical_entity_equivalent(left.evidence_subject, right.evidence_subject)
        or _overlap(left.evidence_subject, right.evidence_subject) >= 0.50
    ):
        reasons.append("derived_endpoint_subject_incompatible")
    for name in ("metric", "evidence_predicate", "unit",
                 "quantity_kind", "aggregation_role", "evidence_polarity"):
        if getattr(left, name, "") != getattr(right, name, ""):
            reasons.append(f"derived_endpoint_{name}_incompatible")
    if not left.unit or source_fact_facets(left) != source_fact_facets(right):
        reasons.append("derived_endpoint_facet_or_unit_incompatible")
    # A source scalar observation need not assert a statistical estimator.
    # It may be compared with a point estimate, retaining its original role
    # and approximation. Bounds, confidence levels and filter thresholds do
    # not establish ordered endpoint observations, even when their roles match.
    if any(endpoint.statistical_role not in {
        StatisticalRole.UNSPECIFIED.value, StatisticalRole.POINT_ESTIMATE.value,
        StatisticalRole.OBSERVED_VALUE.value,
    } for endpoint in (left, right)):
        reasons.append("derived_endpoint_statistical_role_incompatible")
    if not left.period or not right.period or left.period == right.period:
        reasons.append("derived_endpoint_period_incompatible")
    for endpoint in (left, right):
        if not re.fullmatch(r"[-+]?\d+(?:\.\d+)?", str(endpoint.value).replace(",", "")):
            reasons.append("derived_endpoint_value_unresolved")
    return _unique(reasons)


def derived_direction_source_errors(fact: Any, typed_facts: Sequence[Any]) -> tuple[str, ...]:
    """Resolve and recompute a derived direction from exact source snapshots."""
    ids = tuple(getattr(fact, "source_typed_fact_ids", ()) or ())
    digests = tuple(getattr(fact, "source_typed_fact_sha256s", ()) or ())
    if len(ids) != 2 or len(set(ids)) != 2 or len(digests) != 2:
        return ("derived_fact_provenance_incomplete",)
    matches = [[value for value in typed_facts if getattr(value, "fact_id", "") == identifier] for identifier in ids]
    if any(len(values) != 1 for values in matches):
        return ("derived_endpoint_identity_unresolved",)
    left, right = (values[0] for values in matches)
    reasons = list(compatible_source_endpoints(left, right))
    if source_endpoint_order(left) >= source_endpoint_order(right):
        reasons.append("derived_endpoint_order_inconsistent")
    if any(source_endpoint_digest(endpoint) != digest for endpoint,digest in zip((left,right),digests)):
        reasons.append("derived_endpoint_snapshot_mismatch")
    if fact.evidence_atom is not None or fact.numeric_binding is not None or fact.source_claim:
        reasons.append("derived_fact_fabricated_source_atom")
    if not fact.structural_binding_valid or fact.structural_type != "derived_direction":
        reasons.append("derived_fact_structure_invalid")
    source_ids = set(left.source_evidence_unit_ids) | set(right.source_evidence_unit_ids)
    if set(fact.source_evidence_unit_ids) != source_ids or fact.evidence_id not in source_ids:
        reasons.append("derived_fact_source_evidence_mismatch")
    if set(fact.source_candidate_ids) != set(left.source_candidate_ids) | set(right.source_candidate_ids):
        reasons.append("derived_fact_source_candidate_mismatch")
    if (fact.facet != left.facet or fact.metric != left.metric
            or fact.evidence_subject != left.evidence_subject
            or fact.population != left.population or fact.evidence_population != left.evidence_population
            or fact.period != f"{left.period}->{right.period}"):
        reasons.append("derived_fact_scope_mismatch")
    if not reasons:
        a,b = (float(str(endpoint.value).replace(",", "")) for endpoint in (left,right))
        expected = "increase" if b > a else "decrease" if b < a else "stable"
        if fact.direction != expected or fact.value != expected:
            reasons.append("derived_direction_inconsistent")
    return _unique(reasons)


def _experience_comparison_subject(value: str) -> str:
    """Remove comparison grammar and facet-owned spans from its subject."""
    text = re.sub(
        r"^\s*(?:according\s+to\s+(?:the\s+)?(?:document|report|study)|"
        r"לפי\s+(?:המסמך|הדוח|המחקר))\s*,?\s*", "", value, flags=re.I,
    )
    text = re.sub(r"^\s*(?:how\s+(?:did|has|does)|(?:כיצד|איך)\s+השתנ(?:ה|ו|תה))\s+", "", text, flags=re.I)
    text = re.sub(r"\bchange(?:d)?\b", "", text, flags=re.I)
    text = re.sub(r"^\s*(?:(?:the\s+)?(?:share|percentage|rate|proportion)\s+of\s+|(?:חלקו|שיעורו|חלק|שיעור)\s+של\s+)", "", text, flags=re.I)
    # The duration and its operator have a facet owner.  Retain the workers
    # and any independent company/region/population qualifier in the subject.
    for match in reversed(list(_EXPERIENCE_BOUND_RE.finditer(text))):
        if not _experience_bound_owned(text, match):
            continue
        start, end = match.span()
        introducer = re.search(r"(?:\b(?:with|having)\b|(?:בעלי|עם))\s+(?:(?:experience|ניסיון)\s+(?:of\s+|של\s+)?)?$", text[:start], re.I)
        if introducer:
            start = introducer.start()
        text = text[:start] + " " + text[end:]
    text = _ENTRY_EXPERIENCE_RE.sub(" ", text)
    text = _SENIOR_EXPERIENCE_RE.sub(lambda m: re.sub(r"^(?:experienced|senior)\s+", "", m.group(), flags=re.I) if re.search(r"[A-Za-z]", m.group()) else " ", text)
    text = re.sub(r"\bby\s+(?:experience|seniority)\s+levels?\b|לפי\s+(?:רמות\s+)?ניסיון", " ", text, flags=re.I)
    period = r"(?:19|20)\d{2}"
    text = re.sub(rf"(?:\b(?:between|from)\s+|בין\s+){period}\s+(?:and|to|ו[־-]?|ל[־-]?)\s*{period}", " ", text, flags=re.I)
    return normalize_semantic_text(text).strip(" ,;:.?!")

_METRIC_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("employee_count", _EMPLOYS_RE),
    ("employee_recruitment", _RECRUITS_EMPLOYEES_RE),
    ("capital_raised", _RAISES_CAPITAL_RE),
    ("percentage", _PERCENT_RE),
    ("count", _COUNT_RE),
    ("demand", re.compile(r"(?:\bdemand\b|ביקוש)", re.I)),
    ("revenue", re.compile(r"(?:\brevenue\b|הכנס)", re.I)),
    ("profit", re.compile(r"(?:\bprofit\b|רווח)", re.I)),
    ("cost", re.compile(r"(?:\b(?:cost|price)\b|עלות|מחיר)", re.I)),
    ("capacity", re.compile(r"(?:\bcapacity\b|קיבולת|הספק)", re.I)),
    ("duration", re.compile(r"(?:\b(?:duration|time|latency)\b|משך|זמן)", re.I)),
    ("salary", re.compile(r"(?:\b(?:salary|wage|compensation|pay)\b|(?:שכר|משכורת|תגמול))", re.I)),
    ("industry_sector", re.compile(r"(?:\b(?:industry|sector)\b|(?:תעשיי|ענף|סקטור))", re.I)),
)

_NAVIGATION_TYPES = frozenset({
    "table_of_contents", "list_of_figures", "list_of_tables", "header_footer",
    "page_number", "navigation", "malformed", "formula",
})


def normalize_semantic_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = _BIDI_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _enum(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _unique(values: Iterable[Any], maximum: int = MAX_CONTRACT_ITEMS) -> tuple[str, ...]:
    output: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = normalize_semantic_text(raw).strip(" ,;:.?!")[:MAX_CONTRACT_TEXT]
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            output.append(value)
        if len(output) >= maximum:
            break
    return tuple(output)


def _terms(value: Any) -> frozenset[str]:
    output: set[str] = set()
    for raw in _TOKEN_RE.findall(normalize_semantic_text(value).casefold()):
        if raw in _STOPWORDS:
            continue
        output.add(raw)
        if re.fullmatch(r"[a-z]+", raw) and len(raw) >= 5 and raw.endswith("s"):
            output.add(raw[:-1])
        if len(raw) >= 4 and re.fullmatch(r"[\u05d0-\u05ea]+", raw) and raw[0] in "ובכלמהש":
            stripped = raw[1:]
            if len(stripped) >= 2 and stripped not in _STOPWORDS:
                output.add(stripped)
    return frozenset(output)


def canonical_entity_equivalent(left: Any, right: Any) -> bool:
    """Return whether two atomic entity labels share a canonical unit family.

    This is deliberately narrower than topical or metric similarity.  It
    accepts only a complete label from the same established unit family (for
    example, ``workers`` and ``employees``), so it cannot turn a headcount
    into a demand, vacancy, company, or facility count.
    """
    left_terms = _terms(left)
    right_terms = _terms(right)
    if not left_terms or not right_terms:
        return False
    for family in _UNIT_EQUIVALENCE.values():
        def in_family(term: str) -> bool:
            return any(
                term == equivalent
                or (len(equivalent) >= 3 and term.startswith(equivalent))
                for equivalent in family
            )
        if all(in_family(term) for term in left_terms) and all(
            in_family(term) for term in right_terms
        ):
            return True
    return False


def _overlap(left: Any, right: Any) -> float:
    left_terms = _terms(left)
    right_terms = _terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms & right_terms) / max(1, min(len(left_terms), len(right_terms)))


def population_scope_compatibility(
    required_population: str,
    evidence_population: str,
) -> tuple[bool, str]:
    """Evaluate a scope relation without allowing broad facts to narrow.

    A request-side population is a constraint, not topical context.  A source
    population can add detail, but it must state every distinguishing requested
    term.  Missing or generic-only source scope is therefore not evidence for
    an explicit narrower request.
    """
    required_terms = {
        term for term in _terms(required_population)
        if term not in _GENERIC_POPULATION_TERMS
        and not (
            len(term) >= 3 and term[0] in "ובכלמהש"
            and term[1:] in _GENERIC_POPULATION_TERMS
        )
    }
    if not normalize_semantic_text(required_population):
        return True, "scope_not_requested"
    observed_terms = set(_terms(evidence_population))
    if not observed_terms:
        return False, "evidence_scope_unresolved"
    observed_distinguishing = observed_terms - _GENERIC_POPULATION_TERMS
    if required_terms:
        if required_terms <= observed_terms:
            return True, "scope_exact_or_narrower"
        if not observed_distinguishing:
            return False, "evidence_scope_broad_or_unknown"
        return False, "evidence_scope_incompatible"
    if _overlap(required_population, evidence_population) >= 0.18:
        return True, "scope_generic_compatible"
    return False, "evidence_scope_incompatible"


def _population_compatible(population: str, evidence: str) -> bool:
    """Compatibility adapter for legacy call sites; preserves directionality."""
    return population_scope_compatibility(population, evidence)[0]


def candidate_id(hit: Mapping[str, Any]) -> str:
    metadata = hit.get("metadata") or {}
    content_hash = hashlib.sha256(
        str(hit.get("document") or "").encode("utf-8")
    ).hexdigest()
    identity = "\x1f".join(str(item or "") for item in (
        metadata.get("source_id"), metadata.get("chunk_id") or hit.get("id"),
        metadata.get("page"), content_hash,
    ))
    return "cand-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]


def _numeric_identifier_role(text: str, start: int, end: int) -> str:
    from document_query import role_bound_identifier_spans
    from financial_semantics import document_reference_spans
    if any(start < b and a < end for a, b in document_reference_spans(text or '')):
        return 'document_identifier'
    for identifier in role_bound_identifier_spans(text):
        a, b = identifier.identifier_span
        if start < b and a < end:
            return ('document_identifier' if identifier.kind in {'invoice', 'receipt', 'document'}
                    else 'entity_identifier')
    return ''


def classify_numeric_role(
    text: str,
    start: int,
    end: int,
    *,
    unit: str = "",
    relation: str = "",
    structural_role: str = "",
) -> str:
    """Classify a numeric mention without treating proximity as entailment."""
    if structural_role in {"table_header", "unrelated_neighbor"}:
        return structural_role
    identity_role = _numeric_identifier_role(text, start, end)
    if identity_role:
        return identity_role
    for match in _PERIOD_RE.finditer(text or ""):
        if start < match.end() and end > match.start():
            return "period"
    for match in _FILTER_CONDITION_RE.finditer(text or ""):
        threshold_start, threshold_end = match.span("threshold")
        if start < threshold_end and end > threshold_start:
            return "filter_threshold"
    before = (text or "")[max(0, start - 56):start]
    around = (text or "")[max(0, start - 72):min(len(text or ""), end + 72)]
    after = (text or "")[end:min(len(text or ""), end + 40)]
    if _THRESHOLD_CONTEXT_RE.search(before):
        return "filter_threshold"
    if (unit == "%" or "%" in (text or "")[start:end + 1]) and _CONFIDENCE_CONTEXT_RE.search(after):
        return "confidence_level"
    if _LOWER_BOUND_CONTEXT_RE.search(before) and re.search(
        r"(?:\b(?:was|is|at)\b|(?:היה|הייתה|היו|עמד|עמדה))[^;:.]{0,32}$",
        before,
        re.IGNORECASE,
    ):
        return "lower_bound"
    if _CONFIDENCE_CONTEXT_RE.search(around):
        return "confidence_level"
    if _LOWER_BOUND_CONTEXT_RE.search(around):
        return "lower_bound"
    normalized_unit = normalize_semantic_text(unit).casefold()
    if unit == "%" or normalized_unit in {"percent", "percentage", "אחוז"}:
        return "percentage"
    if relation == "count" or normalized_unit in _COUNT_UNITS:
        return "count"
    if relation in {"argmax", "argmin", "compare", "change_over_time"}:
        return "category_value"
    return "requested_answer_value"


def numeric_mentions(text: str, *, relation: str = "") -> tuple[NumericMention, ...]:
    output: list[NumericMention] = []
    for match in _NUMBER_RE.finditer(text or ""):
        unit = str(match.group("unit") or "")
        output.append(NumericMention(
            value=str(match.group("value") or "").replace(",", ""),
            unit=unit,
            role=classify_numeric_role(
                text,
                match.start(),
                match.end(),
                unit=unit,
                relation=relation,
            ),
            text=match.group(0),
        ))
    return tuple(output)


@dataclass(frozen=True)
class ComparisonEndpoint:
    """A real comparison dimension; never a synthetic period placeholder."""

    id: str
    endpoint_type: str
    value: str

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "endpoint_type": self.endpoint_type,
            "value": self.value,
        }


@dataclass(frozen=True)
class FilterCondition:
    """A condition used to select answer rows; its threshold is not an answer."""

    metric: str
    operator: str
    threshold: str
    unit: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "metric": self.metric,
            "operator": self.operator,
            "threshold": self.threshold,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class ScopeConstraint:
    """A condition that constrains the event/scope, not returned rows."""

    metric: str
    operator: str
    threshold: str
    unit: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "metric": self.metric,
            "operator": self.operator,
            "threshold": self.threshold,
            "unit": self.unit,
        }


@dataclass(frozen=True)
class EmbeddedEvent:
    """An event mentioned inside a request for another answer relation."""

    relation: str
    metric: str = ""
    period_a: str = ""
    period_b: str = ""
    direction: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "relation": self.relation,
            "metric": self.metric,
            "period_a": self.period_a,
            "period_b": self.period_b,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class CanonicalClause:
    id: str
    text: str
    semantic_relation: str
    answer_shape: str
    subject: str
    predicate: str
    objects_or_metrics: tuple[str, ...] = ()
    requested_values: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    periods: tuple[str, ...] = ()
    populations: tuple[str, ...] = ()
    polarity: str = "neutral"
    qualifiers: tuple[str, ...] = ()
    requested_cardinality: int = 1
    completeness: str = "claim"
    numeric_mentions: tuple["NumericMention", ...] = ()
    comparison_endpoints: tuple[ComparisonEndpoint, ...] = ()
    requested_facets: tuple[str, ...] = ()
    requested_output_slots: tuple[str, ...] = ()
    target_entity_type: str = ""
    counted_entity: str = ""
    requested_attribute: str = ""
    comparison_subject: str = ""
    comparison_reference: str = ""
    scope_population: str = ""
    relation_target: str = ""
    answer_relation: str = ""
    embedded_event: Optional[EmbeddedEvent] = None
    filter_condition: Optional[FilterCondition] = None
    scope_constraint: Optional[ScopeConstraint] = None
    output_schema: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "text": self.text,
            "semantic_relation": self.semantic_relation,
            "answer_shape": self.answer_shape, "subject": self.subject,
            "predicate": self.predicate,
            "objects_or_metrics": list(self.objects_or_metrics),
            "requested_values": list(self.requested_values),
            "units": list(self.units), "periods": list(self.periods),
            "populations": list(self.populations), "polarity": self.polarity,
            "qualifiers": list(self.qualifiers),
            "requested_cardinality": self.requested_cardinality,
            "completeness": self.completeness,
            "numeric_mentions": [item.as_dict() for item in self.numeric_mentions],
            "comparison_endpoints": [
                item.as_dict() for item in self.comparison_endpoints
            ],
            "requested_facets": list(self.requested_facets),
            "requested_output_slots": list(self.requested_output_slots),
            "target_entity_type": self.target_entity_type,
            "counted_entity": self.counted_entity,
            "requested_attribute": self.requested_attribute,
            "comparison_subject": self.comparison_subject,
            "comparison_reference": self.comparison_reference,
            "scope_population": self.scope_population,
            "relation_target": self.relation_target,
            "answer_relation": self.answer_relation or self.semantic_relation,
            "embedded_event": (
                self.embedded_event.as_dict() if self.embedded_event else None
            ),
            "filter_condition": (
                self.filter_condition.as_dict() if self.filter_condition else None
            ),
            "scope_constraint": (
                self.scope_constraint.as_dict() if self.scope_constraint else None
            ),
            "output_schema": list(self.output_schema),
        }


@dataclass(frozen=True)
class NumericMention:
    """A number and its semantic role in the resolved user request."""

    value: str
    unit: str
    role: str
    text: str

    def as_dict(self) -> dict[str, str]:
        return {
            "value": self.value,
            "unit": self.unit,
            "role": self.role,
            "text": self.text,
        }


@dataclass(frozen=True)
class TypedRequirement:
    id: str
    clause_id: str
    kind: str
    subject: str
    predicate: str
    objects_or_metrics: tuple[str, ...] = ()
    requested_values: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    periods: tuple[str, ...] = ()
    populations: tuple[str, ...] = ()
    polarity: str = "neutral"
    qualifiers: tuple[str, ...] = ()
    cardinality: int = 1
    completeness: str = "complete_proposition"
    linked_requirement_ids: tuple[str, ...] = ()
    mandatory: bool = True
    endpoint_type: str = ""
    endpoint_value: str = ""
    facet: str = ""
    output_slot: str = ""
    value_type: str = "numeric"
    answer_producing: bool = True
    target_entity_type: str = ""
    counted_entity: str = ""
    requested_attribute: str = ""
    comparison_subject: str = ""
    comparison_reference: str = ""
    scope_population: str = ""
    relation_target: str = ""
    answer_relation: str = ""
    filter_operator: str = ""
    filter_threshold: str = ""
    filter_unit: str = ""
    scope_operator: str = ""
    scope_threshold: str = ""
    scope_unit: str = ""
    # ``scalar_role`` is a legacy projection.  New compatibility uses the
    # three dimensions below and leaves unrequested dimensions unspecified.
    scalar_role: str = ScalarSemanticRole.NOT_APPLICABLE.value
    quantity_kind: str = QuantityKind.UNSPECIFIED.value
    statistical_role: str = StatisticalRole.UNSPECIFIED.value
    aggregation_role: str = AggregationRole.UNSPECIFIED.value
    # A canonical C18 compiler can intentionally leave every quantity
    # dimension open.  This flag prevents a legacy scalar projection from
    # silently reintroducing a constraint at that boundary.
    quantity_semantics_authoritative: bool = False
    satisfaction_mode: RequirementSatisfactionMode = (
        RequirementSatisfactionMode.SOURCE_FACT
    )
    dimension_constraints: tuple[RequirementDimensionConstraint, ...] = ()
    semantic_accounting: tuple[RequirementSemanticAccounting, ...] = ()

    def __post_init__(self) -> None:
        declared = self.satisfaction_mode
        if isinstance(declared, str):
            try:
                declared = RequirementSatisfactionMode(declared)
            except ValueError:
                declared = RequirementSatisfactionMode.SOURCE_FACT
        # Requirement construction owns this classification.  A caller may
        # only use the canonical classification; accepting a route-shaped
        # override would let a request condition enter source extraction.
        canonical = requirement_satisfaction_mode(self.kind)
        object.__setattr__(self, "satisfaction_mode", canonical)
        legacy = quantity_semantics_from_scalar_role(self.scalar_role)
        quantity_kind = normalize_quantity_kind(self.quantity_kind)
        statistical_role = normalize_statistical_role(self.statistical_role)
        aggregation_role = normalize_aggregation_role(self.aggregation_role)
        legacy_only = (
            quantity_kind == QuantityKind.UNSPECIFIED.value
            and statistical_role == StatisticalRole.UNSPECIFIED.value
            and aggregation_role == AggregationRole.UNSPECIFIED.value
        )
        if legacy_only and not self.quantity_semantics_authoritative:
            quantity_kind = legacy.quantity_kind
            statistical_role = legacy.statistical_role
            aggregation_role = legacy.aggregation_role
        object.__setattr__(self, "quantity_kind", quantity_kind)
        object.__setattr__(self, "statistical_role", statistical_role)
        object.__setattr__(self, "aggregation_role", aggregation_role)
        object.__setattr__(self, "scalar_role", scalar_role_from_quantity_semantics(
            quantity_kind, statistical_role, aggregation_role,
        ))
        supplied = tuple(self.dimension_constraints or ())
        constraints = _normalize_requirement_constraints(
            self,
            supplied,
        )
        object.__setattr__(self, "dimension_constraints", constraints)
        supplied_accounting = tuple(self.semantic_accounting or ())
        if supplied_accounting:
            accounting = supplied_accounting
        else:
            accounting = tuple(
                RequirementSemanticAccounting(
                    semantic_distinction=constraint.value,
                    semantic_category=_SEMANTIC_CATEGORY_BY_DIMENSION.get(
                        constraint.dimension, "OTHER",
                    ),
                    request_span=constraint.request_span,
                    semantic_owner=constraint.semantic_owner,
                    hard_constraint_owner=constraint.hard_constraint_owner,
                    enforcement_path=constraint.enforcement_path,
                    enforcement_equivalence_status=(
                        constraint.enforcement_equivalence_status
                    ),
                    independently_meaningful=True,
                    independently_discriminative=True,
                    proof=constraint.justification,
                )
                for constraint in constraints
                if constraint.state is RequirementConstraintState.REQUIRED
            )
        object.__setattr__(self, "semantic_accounting", accounting)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "clause_id": self.clause_id, "kind": self.kind,
            "subject": self.subject, "predicate": self.predicate,
            "objects_or_metrics": list(self.objects_or_metrics),
            "requested_values": list(self.requested_values),
            "units": list(self.units), "periods": list(self.periods),
            "populations": list(self.populations), "polarity": self.polarity,
            "qualifiers": list(self.qualifiers), "cardinality": self.cardinality,
            "completeness": self.completeness,
            "linked_requirement_ids": list(self.linked_requirement_ids),
            "mandatory": self.mandatory,
            "endpoint_type": self.endpoint_type,
            "endpoint_value": self.endpoint_value,
            "facet": self.facet,
            "output_slot": self.output_slot,
            "value_type": self.value_type,
            "answer_producing": self.answer_producing,
            "target_entity_type": self.target_entity_type,
            "counted_entity": self.counted_entity,
            "requested_attribute": self.requested_attribute,
            "comparison_subject": self.comparison_subject,
            "comparison_reference": self.comparison_reference,
            "scope_population": self.scope_population,
            "relation_target": self.relation_target,
            "answer_relation": self.answer_relation,
            "filter_operator": self.filter_operator,
            "filter_threshold": self.filter_threshold,
            "filter_unit": self.filter_unit,
            "scope_operator": self.scope_operator,
            "scope_threshold": self.scope_threshold,
            "scope_unit": self.scope_unit,
            "scalar_role": self.scalar_role,
            "quantity_kind": self.quantity_kind,
            "statistical_role": self.statistical_role,
            "aggregation_role": self.aggregation_role,
            "quantity_semantics_authoritative": (
                self.quantity_semantics_authoritative
            ),
            "satisfaction_mode": self.satisfaction_mode.value,
            "dimension_constraints": [
                item.as_dict() for item in self.dimension_constraints
            ],
            "semantic_accounting": [
                item.as_dict() for item in self.semantic_accounting
            ],
        }


def _legacy_constraint_values(
    requirement: TypedRequirement,
) -> Mapping[str, str]:
    """Project direct legacy construction into explicit, justified fields.

    Runtime contracts do not use this fallback: C20 compiles their full
    request-span provenance below.  It preserves test and compatibility
    callers that construct a ``TypedRequirement`` directly with explicit
    semantic fields, without permitting an unspecified field to harden later.
    """
    numeric = (
        requirement.value_type in {"numeric", "currency", "period", "count", "percentage"}
        or requirement.kind in {
            "count", "percentage", "comparison_endpoint",
            "category_value_bindings", "winner", "winner_attribute",
            "filtered_row_value",
        }
    )
    return {
        "value": "required" if numeric else "",
        "quantity_kind": (
            requirement.quantity_kind
            if numeric and requirement.quantity_kind != QuantityKind.UNSPECIFIED.value
            else ""
        ),
        "statistical_role": (
            requirement.statistical_role
            if numeric and requirement.statistical_role != StatisticalRole.UNSPECIFIED.value
            else ""
        ),
        "aggregation_role": (
            requirement.aggregation_role
            if numeric and requirement.aggregation_role not in {
                AggregationRole.NONE.value, AggregationRole.UNSPECIFIED.value,
            } else ""
        ),
        "subject": requirement.subject,
        "metric": " ".join(
            item for item in requirement.objects_or_metrics
            if item not in {"count", "percentage", "number", "scalar"}
        ),
        "predicate": requirement.predicate,
        "period": "|".join(requirement.periods),
        "population": requirement.scope_population or "|".join(requirement.populations),
        "facet": requirement.facet,
        "unit": "|".join(requirement.units) if numeric else "",
        "polarity": requirement.polarity if requirement.polarity != "neutral" else "",
        "target_entity": requirement.counted_entity or requirement.target_entity_type,
        "relation_arguments": requirement.relation_target,
        "fact_role": "measured_outcome" if numeric else "",
        "structural_type": "required",
        "proposition_completeness": "required",
    }


def _normalize_requirement_constraints(
    requirement: TypedRequirement,
    supplied: Sequence[RequirementDimensionConstraint | Mapping[str, Any]],
) -> tuple[RequirementDimensionConstraint, ...]:
    by_dimension: dict[str, RequirementDimensionConstraint] = {}
    for item in supplied:
        constraint = (
            item if isinstance(item, RequirementDimensionConstraint)
            else RequirementDimensionConstraint(**dict(item))
        )
        if constraint.dimension in by_dimension:
            raise ValueError("duplicate requirement constraint dimension")
        by_dimension[constraint.dimension] = constraint
    if not supplied:
        for dimension, value in _legacy_constraint_values(requirement).items():
            by_dimension[dimension] = RequirementDimensionConstraint(
                dimension=dimension,
                state=(
                    RequirementConstraintState.REQUIRED
                    if value else RequirementConstraintState.NOT_REQUIRED
                ),
                value=value,
                origin=(
                    RequirementConstraintOrigin.DERIVED_BUT_PROVEN
                    if value else RequirementConstraintOrigin.UNKNOWN
                ),
                rule="explicit_typed_requirement_field",
                justification=(
                    "direct TypedRequirement constructor supplied the field"
                    if value else "no direct field constrains this dimension"
                ),
            )
    for dimension in _REQUIREMENT_CONSTRAINT_DIMENSIONS:
        by_dimension.setdefault(
            dimension,
            RequirementDimensionConstraint(
                dimension=dimension,
                state=RequirementConstraintState.NOT_REQUIRED,
                origin=RequirementConstraintOrigin.UNKNOWN,
                rule="compiler_absence",
                justification="no request semantic constrains this dimension",
            ),
        )
    return tuple(by_dimension[dimension] for dimension in _REQUIREMENT_CONSTRAINT_DIMENSIONS)


def requirement_dimension_constraint(
    requirement: TypedRequirement,
    dimension: str,
) -> RequirementDimensionConstraint:
    """Return the canonical per-dimension state for one requirement."""
    for constraint in requirement.dimension_constraints:
        if constraint.dimension == dimension:
            return constraint
    raise ValueError("missing requirement constraint dimension")


def requirement_dimension_is_required(
    requirement: TypedRequirement,
    dimension: str,
) -> bool:
    return requirement_dimension_constraint(
        requirement, dimension,
    ).state is RequirementConstraintState.REQUIRED


def requirement_dimension_value(
    requirement: TypedRequirement,
    dimension: str,
) -> str:
    """Return a value only when compilation made it a source constraint."""
    constraint = requirement_dimension_constraint(requirement, dimension)
    return (
        constraint.value
        if constraint.state is RequirementConstraintState.REQUIRED else ""
    )


@dataclass
class SemanticContractDraft:
    """The sole mutable request representation before deterministic freeze."""

    normalized_request: str
    draft_operation: str
    clauses: list[CanonicalClause]
    endpoint_hints: tuple[str, ...] = ()
    violations: list[str] = field(default_factory=list)
    normalization_applied: bool = False
    normalization_count: int = 0


@dataclass(frozen=True)
class SemanticCompilationResult:
    """Bounded result of semantic compilation; uncertainty is data, not 500."""

    status: str
    draft: SemanticContractDraft
    violations: tuple[str, ...]
    contract: Optional["RequestExecutionContract"] = None
    reason_codes: tuple[str, ...] = ()
    model_resolution_attempted: bool = False
    planner_operation: str = ""
    deterministic_operation: str = ""
    deterministic_semantic_agreement: Optional[bool] = None
    pre_freeze_disagreement: bool = False
    reconciled: bool = False

    @property
    def valid(self) -> bool:
        return self.contract is not None and self.status in {"valid", "repaired"}

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "draft_operation": self.draft.draft_operation,
            "normalized_request": self.draft.normalized_request,
            "violations": list(self.violations),
            "reason_codes": list(self.reason_codes),
            "model_resolution_attempted": self.model_resolution_attempted,
            "planner_operation": self.planner_operation,
            "deterministic_operation": self.deterministic_operation,
            "deterministic_semantic_agreement": self.deterministic_semantic_agreement,
            "pre_freeze_disagreement": self.pre_freeze_disagreement,
            "reconciled": self.reconciled,
            "contract_id": self.contract.contract_id if self.contract else None,
        }


class SemanticCompilationError(ValueError):
    """Compatibility exception carrying the safe compilation result."""

    def __init__(self, result: SemanticCompilationResult):
        super().__init__("semantic_compilation_unresolved")
        self.result = result


@dataclass(frozen=True)
class SemanticValidationRecord:
    draft_operation: str
    violations: tuple[str, ...]
    normalization_applied: bool
    normalization_count: int
    valid: bool
    final_semantic_signature: str
    endpoint_types: tuple[str, ...]
    requested_facets: tuple[str, ...]
    requested_output_slots: tuple[str, ...]
    semantic_role_checks: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "draft_operation": self.draft_operation,
            "violations": list(self.violations),
            "normalization_applied": self.normalization_applied,
            "normalization_count": self.normalization_count,
            "valid": self.valid,
            "final_semantic_signature": self.final_semantic_signature,
            "endpoint_types": list(self.endpoint_types),
            "requested_facets": list(self.requested_facets),
            "requested_output_slots": list(self.requested_output_slots),
            "semantic_role_checks": list(self.semantic_role_checks),
        }


@dataclass(frozen=True)
class RequestExecutionContract:
    contract_id: str
    contract_sha256: str
    schema_version: str
    resolved_user_request: str
    normalized_request: str
    conversation_operation: str
    execution_route: str
    support_status: str
    source_scope: tuple[str, ...]
    semantic_relation: str
    answer_shape: str
    subjects: tuple[str, ...]
    predicates: tuple[str, ...]
    objects_or_metrics: tuple[str, ...]
    requested_values: tuple[str, ...]
    units: tuple[str, ...]
    periods: tuple[str, ...]
    populations: tuple[str, ...]
    polarity: str
    qualifiers: tuple[str, ...]
    requested_cardinality: int
    completeness: str
    clauses: tuple[CanonicalClause, ...]
    requirements: tuple[TypedRequirement, ...]
    semantic_validation: SemanticValidationRecord
    target_entity_types: tuple[str, ...] = ()
    counted_entities: tuple[str, ...] = ()
    requested_attributes: tuple[str, ...] = ()
    comparison_subjects: tuple[str, ...] = ()
    comparison_references: tuple[str, ...] = ()
    scope_populations: tuple[str, ...] = ()
    answer_relations: tuple[str, ...] = ()
    embedded_events: tuple[EmbeddedEvent, ...] = ()
    filter_conditions: tuple[FilterCondition, ...] = ()
    scope_constraints: tuple[ScopeConstraint, ...] = ()
    answer_scope: str = "targeted"

    @property
    def requirement_ids(self) -> frozenset[str]:
        return frozenset(item.id for item in self.requirements)

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "contract_sha256": self.contract_sha256,
            "schema_version": self.schema_version,
            "resolved_user_request": self.resolved_user_request,
            "normalized_request": self.normalized_request,
            "conversation_operation": self.conversation_operation,
            "execution_route": self.execution_route,
            "support_status": self.support_status,
            "source_scope": list(self.source_scope),
            "semantic_relation": self.semantic_relation,
            "answer_shape": self.answer_shape,
            "answer_scope": self.answer_scope,
            "subjects": list(self.subjects), "predicates": list(self.predicates),
            "objects_or_metrics": list(self.objects_or_metrics),
            "requested_values": list(self.requested_values),
            "units": list(self.units), "periods": list(self.periods),
            "populations": list(self.populations), "polarity": self.polarity,
            "qualifiers": list(self.qualifiers),
            "requested_cardinality": self.requested_cardinality,
            "completeness": self.completeness,
            "clauses": [item.as_dict() for item in self.clauses],
            "requirements": [item.as_dict() for item in self.requirements],
            "semantic_validation": self.semantic_validation.as_dict(),
            "target_entity_types": list(self.target_entity_types),
            "counted_entities": list(self.counted_entities),
            "requested_attributes": list(self.requested_attributes),
            "comparison_subjects": list(self.comparison_subjects),
            "comparison_references": list(self.comparison_references),
            "scope_populations": list(self.scope_populations),
            "answer_relations": list(self.answer_relations),
            "embedded_events": [item.as_dict() for item in self.embedded_events],
            "filter_conditions": [item.as_dict() for item in self.filter_conditions],
            "scope_constraints": [item.as_dict() for item in self.scope_constraints],
        }


# Compatibility name for callers that only need the semantic slots. Runtime
# construction uses ``build_request_execution_contract`` exactly once.
CanonicalSemanticContract = RequestExecutionContract


@dataclass(frozen=True)
class CandidateDecision:
    candidate_id: str
    compatible_requirement_ids: tuple[str, ...]
    selected: bool
    rejected: bool
    rejection_reason_codes: tuple[str, ...]
    subject_compatibility: str
    predicate_compatibility: str
    metric_compatibility: str
    period_compatibility: str
    population_compatibility: str
    polarity_compatibility: str
    unit_compatibility: str
    structural_type_compatibility: str
    proposition_completeness: str
    source_tenant_integrity: str
    preference_score: int
    proposition_support_score: int
    retrieval_score: float
    retrieval_channels: tuple[str, ...]
    retrieval_channel_ranks: tuple[tuple[str, int], ...]
    best_retrieval_rank: int
    selection_disposition: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "retrieved": True, "eligibility_evaluated": True,
            "compatible_requirements": list(self.compatible_requirement_ids),
            "requirement_compatible": bool(self.compatible_requirement_ids),
            "selected_for_fact_extraction": self.selected,
            "selected_for_synthesis": False,
            "produced_fact": False,
            "selected_for_evidence_bundle": False,
            "supplied_to_generation": False,
            "used_in_claim": False,
            "rejected": self.rejected,
            "rejection_reason_codes": list(self.rejection_reason_codes),
            "selection_disposition": self.selection_disposition,
            "subject_compatibility": self.subject_compatibility,
            "predicate_compatibility": self.predicate_compatibility,
            "metric_compatibility": self.metric_compatibility,
            "period_compatibility": self.period_compatibility,
            "population_compatibility": self.population_compatibility,
            "polarity_compatibility": self.polarity_compatibility,
            "unit_compatibility": self.unit_compatibility,
            "structural_type_compatibility": self.structural_type_compatibility,
            "proposition_completeness": self.proposition_completeness,
            "source_tenant_integrity": self.source_tenant_integrity,
            "preference_score": self.preference_score,
            "proposition_support_score": self.proposition_support_score,
            "retrieval_score": self.retrieval_score,
            "retrieval_channels": list(self.retrieval_channels),
            "retrieval_channel_ranks": dict(self.retrieval_channel_ranks),
            "best_retrieval_rank": self.best_retrieval_rank,
            "selected_for_citation": False,
            "omitted_after_synthesis": False,
            "removed_by_final_gate": False,
        }


@dataclass(frozen=True)
class FieldEntailment:
    requirement_id: str
    claim_index: Optional[int]
    citation_ids: tuple[str, ...]
    subject: bool
    predicate: bool
    metric: bool
    value: bool
    unit: bool
    period: bool
    population: bool
    polarity: bool
    qualifier: bool
    passed: bool
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "claim_index": self.claim_index,
            "citation_ids": list(self.citation_ids),
            "subject_check": self.subject, "predicate_check": self.predicate,
            "metric_check": self.metric, "value_check": self.value,
            "unit_check": self.unit, "period_check": self.period,
            "population_check": self.population,
            "polarity_check": self.polarity,
            "qualifier_check": self.qualifier, "passed": self.passed,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class ResponseContractResult:
    passed: bool
    relevance_passed: bool
    citation_entailment_passed: bool
    claims: tuple[str, ...]
    requirements: tuple[FieldEntailment, ...]
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "relevance_passed": self.relevance_passed,
            "citation_entailment_passed": self.citation_entailment_passed,
            "claim_count": len(self.claims),
            "requirements": [item.as_dict() for item in self.requirements],
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class ClaimNode:
    claim_id: str
    text: str
    requirement_ids: tuple[str, ...]
    fact_ids: tuple[str, ...]
    evidence_unit_ids: tuple[str, ...]
    citation_ids: tuple[str, ...]
    supported: bool
    reason_codes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim_sha256": hashlib.sha256(self.text.encode("utf-8")).hexdigest(),
            "claim_preview": self.text[:240],
            "requirement_ids": list(self.requirement_ids),
            "fact_ids": list(self.fact_ids),
            "evidence_unit_ids": list(self.evidence_unit_ids),
            "citation_ids": list(self.citation_ids),
            "supported": self.supported,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class CitationBinding:
    citation_id: str
    claim_id: str
    fact_ids: tuple[str, ...]
    evidence_unit_id: str
    direct: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "claim_id": self.claim_id,
            "fact_ids": list(self.fact_ids),
            "evidence_unit_id": self.evidence_unit_id,
            "direct": self.direct,
        }


@dataclass(frozen=True)
class FactProvenanceNode:
    fact_id: str
    evidence_unit_id: str
    candidate_id: str
    source_evidence_unit_ids: tuple[str, ...]
    source_candidate_ids: tuple[str, ...]
    source_typed_fact_ids: tuple[str, ...]
    operator_derivation_id: str
    derivation_description: str
    evidence_context: Mapping[str, Any]
    evidence_atom_id: str = ""

    @property
    def operator_provenance_complete(self) -> bool:
        if not self.operator_derivation_id:
            return True
        return bool(
            self.evidence_unit_id
            and not self.evidence_unit_id.startswith("operator-hit-")
            and self.candidate_id
            and self.source_evidence_unit_ids
            and self.source_candidate_ids
            and self.source_typed_fact_ids
            and self.derivation_description
            and self.evidence_context
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "evidence_unit_id": self.evidence_unit_id,
            "candidate_id": self.candidate_id,
            "source_evidence_unit_ids": list(self.source_evidence_unit_ids),
            "source_candidate_ids": list(self.source_candidate_ids),
            "source_typed_fact_ids": list(self.source_typed_fact_ids),
            "operator_derivation_id": self.operator_derivation_id,
            "derivation_description": self.derivation_description,
            "evidence_context": dict(self.evidence_context),
            "evidence_atom_id": self.evidence_atom_id,
            "operator_provenance_complete": self.operator_provenance_complete,
        }


@dataclass(frozen=True)
class ClaimGraph:
    contract_id: str
    contract_sha256: str
    claims: tuple[ClaimNode, ...]
    citation_bindings: tuple[CitationBinding, ...]
    covered_requirement_ids: tuple[str, ...]
    missing_requirement_ids: tuple[str, ...]
    contextual_factual_correctness: bool
    standalone_completeness: bool
    fact_provenance: tuple[FactProvenanceNode, ...] = ()
    deterministic_repair_used: bool = False

    @property
    def passed(self) -> bool:
        return (
            self.contextual_factual_correctness
            and not self.missing_requirement_ids
            and all(item.supported for item in self.claims)
            and all(item.direct for item in self.citation_bindings)
            and all(item.operator_provenance_complete for item in self.fact_provenance)
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "contract_id": self.contract_id,
            "contract_sha256": self.contract_sha256,
            "passed": self.passed,
            "contextual_factual_correctness": self.contextual_factual_correctness,
            "standalone_completeness": self.standalone_completeness,
            "deterministic_repair_used": self.deterministic_repair_used,
            "covered_requirement_ids": list(self.covered_requirement_ids),
            "missing_requirement_ids": list(self.missing_requirement_ids),
            "claims": [item.as_dict() for item in self.claims],
            "citation_bindings": [item.as_dict() for item in self.citation_bindings],
            "fact_provenance": [item.as_dict() for item in self.fact_provenance],
        }


_PURPOSE_PROPOSITION_RE = re.compile(
    r"\b(?:purpose|objective|goal|aim(?:ed|s)?|intended|designed|created)\b|"
    r"(?:מטר(?:ה|ת)|נועד|נועדה|נועדו|יעד|תכלית|כדי|לשם)",
    re.IGNORECASE,
)
_DEFINITION_PROPOSITION_RE = re.compile(
    r"\b(?:is|are)\s+(?:defined|responsible)|\bmeans?\b|\brefers?\s+to\b|"
    r"\b(?:role|responsibilit(?:y|ies))\b|"
    r"(?:מוגדר|מוגדרת|פירוש|משמעות|אחראי|אחראית|תפקיד|כולל(?:ת|ים|ות)?)",
    re.IGNORECASE,
)
_CAUSE_PROPOSITION_RE = re.compile(
    r"\b(?:because|due\s+to|owing\s+to|caused\s+by|reason(?:s)?\s+(?:was|were|is|are))\b|"
    r"(?:בגלל|בשל|עקב|מפני|כתוצאה\s+מ|הסיבה|נבע|נבעה|לא\s+ניתן.{0,80}(?:מפאת|בשל|עקב))",
    re.IGNORECASE,
)
_EXPLANATION_PROPOSITION_RE = re.compile(
    r"\b(?:because|therefore|(?:is|was|are|were)\s+(?:explained|interpreted)\s+(?:by|as)|"
    r"accounted\s+for\s+by|this\s+(?:means|indicates|suggests)|"
    r"no\s+(?:reason|explanation)\s+(?:was|is)\s+(?:given|provided))\b|"
    r"(?:מוסבר|מוסברת|מתפרש|מתפרשת|כלומר|משמעות(?:ו|ה)?|מכאן|לכן|"
    r"הדבר\s+מצביע|לא\s+(?:ניתן|נמסר).{0,48}הסבר)",
    re.IGNORECASE,
)


def segment_text_propositions(value: str) -> tuple[tuple[str, int, int], ...]:
    """Return bounded contiguous propositions with exact source offsets."""
    text = str(value or "")
    output: list[tuple[str, int, int]] = []
    for match in re.finditer(
        r"(?:^|(?<=[.!?׃])\s+|(?=^\s*(?:[-*•▪◦]|\d{1,2}[.)])\s+))"
        r"(?P<body>[^\n]{4,900}?(?:[.!?׃](?=\s|$)|$))",
        text,
        re.MULTILINE,
    ):
        body = normalize_semantic_text(match.group("body"))
        if body:
            start = match.start("body")
            end = match.end("body")
            output.append((body, start, end))
        if len(output) >= 24:
            break
    if not output:
        normalized = normalize_semantic_text(text)
        if normalized:
            output.append((normalized[:900], 0, min(len(text), 900)))
    return tuple(output)


def proposition_relation_supported(
    requirement: TypedRequirement,
    proposition: str,
) -> bool:
    """Require the proposition itself to express the requested relation."""
    kind = requirement.kind
    value = normalize_semantic_text(proposition)
    if not value:
        return False
    if kind in {"list", "list_attributes", "summary", "procedure"}:
        return True
    if kind in {"purpose", "intended_use"}:
        return bool(_PURPOSE_PROPOSITION_RE.search(value))
    if kind == "definition":
        return bool(_DEFINITION_PROPOSITION_RE.search(value))
    if kind == "cause_or_reason":
        return bool(_CAUSE_PROPOSITION_RE.search(value))
    if kind == "explanation":
        return bool(
            _EXPLANATION_PROPOSITION_RE.search(value)
            or _CAUSE_PROPOSITION_RE.search(value)
        )
    if kind == "negative_relation":
        return bool(_NEGATION_RE.search(value))
    return True


def document_role_source_errors(fact: Any) -> tuple[str, ...]:
    """Prove a source-owned role independently of the rendered predicate."""
    predicate = str(getattr(fact, "evidence_predicate", "") or "")
    if predicate not in DOCUMENT_ROLE_PREDICATES:
        return ("document_role_predicate_incompatible",)
    arguments = {str(getattr(item, "role", "")): str(getattr(item, "value", ""))
                 for item in getattr(fact, "relation_arguments", ()) or ()}
    target_role, output_role = RELATION_ARGUMENT_SCHEMAS[predicate]
    reasons = []
    source_arguments = document_role_arguments(str(getattr(fact, "source_claim", "") or ""), predicate)
    if (not source_arguments.get(target_role) or not source_arguments.get(output_role)
            or not document_role_target_compatible(source_arguments.get(target_role, ""), arguments.get(target_role, ""))
            or normalize_semantic_text(source_arguments.get(output_role, "")) != normalize_semantic_text(arguments.get(output_role, ""))):
        reasons.append("document_role_source_relation_not_proved")
    context = getattr(fact, "evidence_context", None)
    if context is None:
        reasons.append("document_role_provenance_missing")
    else:
        reasons.extend(structural_binding_validation_errors(context))
        reasons.extend(relation_argument_validation_errors(context, predicate=predicate))
    proposition = str(getattr(fact, "object_text", "") or getattr(fact, "source_claim", ""))
    if _NEGATION_RE.search(proposition) or getattr(fact, "evidence_polarity", "") == "negative":
        reasons.append("document_role_affirmation_missing")
    return _unique(reasons)


def technical_attribute_label_errors(requirement: TypedRequirement, fact: Any) -> tuple[str, ...]:
    """Preserve every frozen equipment-role word in its source-owned label."""
    if requirement_dimension_value(requirement, "predicate") != "has_attribute":
        return ()
    from document_query import _words
    def equipment_terms(value):
        return {"ניהול" if term == "לניהול" else term for term in _words(value)}
    wanted = equipment_terms(requirement_dimension_value(requirement, "subject"))
    observed = equipment_terms(str(getattr(fact, "evidence_subject", "") or getattr(fact, "subject", "") or ""))
    return () if wanted and wanted <= observed else ("technical_attribute_subject_incompatible",)


def text_fact_binding_errors(
    requirement: TypedRequirement,
    fact: Any,
) -> tuple[str, ...]:
    """Validate proposition relation and its requested relational target."""
    proposition = str(
        getattr(fact, "object_text", "")
        or getattr(fact, "source_claim", "")
    )
    predicate = str(getattr(fact, "evidence_predicate", "") or "")
    kind = requirement.kind
    arguments = {
        str(getattr(item, "role", "") or ""):
        str(getattr(item, "value", "") or "")
        for item in getattr(fact, "relation_arguments", ()) or ()
    }
    role_pairs = {
        ("cause_or_reason", "caused_by"): (("effect", "reason"),),
        ("explanation", "caused_by"): (("effect", "reason"),),
        ("explanation", "explained_as"): (
            ("explained_event", "explanation"),
        ),
        ("purpose", "purpose_of"): (("purpose_target", "purpose"),),
        ("intended_use", "purpose_of"): (("purpose_target", "purpose"),),
        ("definition", "defined_as"): (("defined_entity", "definition"),),
        ("procedure", "performed_by"): (("procedure_target", "method"),),
        ("negative_relation", "excluded_from"): (
            ("excluded_scope", "excluded_entity"),
        ),
        ("negative_relation", "exception_to"): (
            ("excluded_scope", "excluded_entity"),
        ),
        ("positive_inclusion_or_count", "included_in"): (
            ("included_scope", "included_entity"),
        ),
    }.get((kind, predicate), ())
    canonical_errors = tuple(
        getattr(fact, "canonical_validation_errors", ()) or ()
    )
    from document_query import document_fact_errors, attribute_context_errors
    reasons: list[str] = list(document_fact_errors(requirement_dimension_value(requirement, "document_scope"), fact))
    reasons.extend(attribute_context_errors(requirement_dimension_value(requirement, "attribute_context"), fact))
    reasons.extend(technical_attribute_label_errors(requirement, fact))
    if requirement.kind == "list" and requirement_dimension_value(requirement, "ordered_list_heading"):
        from ordered_list_evidence import ordered_list_fact_errors
        return _unique((*reasons, *ordered_list_fact_errors(requirement, fact)))
    if canonical_errors or getattr(fact, "canonical_valid", True) is False:
        reasons.append("canonical_relation_fact_invalid")
        reasons.extend(canonical_errors)
    if requirement.predicate in DOCUMENT_ROLE_PREDICATES:
        if predicate != requirement.predicate:
            reasons.append("document_role_predicate_incompatible")
        target_role, output_role = RELATION_ARGUMENT_SCHEMAS[requirement.predicate]
        reasons.extend(document_role_source_errors(fact))
        if not arguments.get(output_role) or not document_role_target_compatible(
            requirement.relation_target or requirement.subject, arguments.get(target_role, "")
        ):
            reasons.append("document_role_target_incompatible")
        return _unique(reasons)
    typed_relation_supported = any(
        arguments.get(target_role) and arguments.get(output_role)
        for target_role, output_role in role_pairs
    )
    if (
        not typed_relation_supported
        and not proposition_relation_supported(requirement, proposition)
    ):
        reasons.append("proposition_relation_not_supported")
    requested_target = normalize_semantic_text(
        getattr(requirement, "relation_target", "") or ""
    )
    if not requested_target:
        return tuple(reasons)
    role = role_pairs[0][0] if role_pairs else {
        "cause_or_reason": (
            "effect" if predicate == "caused_by" else "explained_event"
        ),
        "explanation": (
            "effect" if predicate == "caused_by" else "explained_event"
        ),
        "purpose": "purpose_target",
        "intended_use": "purpose_target",
        "definition": "defined_entity",
        "procedure": "procedure_target",
        "negative_relation": "excluded_scope",
        "positive_inclusion_or_count": "included_scope",
    }.get(kind, "")
    if not role:
        return tuple(reasons)
    value = arguments.get(role, "")
    # Inclusion/exclusion scope may be expressed by a local relation argument
    # plus its provenance-bound governing section (for example, a row says
    # "excluded from the estimate" under a named inspection section).  Keep
    # the typed argument literal, but use the recorded structural context when
    # checking compatibility with the complete requested scope.
    comparison_value = value
    if kind in {"negative_relation", "positive_inclusion_or_count"}:
        evidence_context = getattr(fact, "evidence_context", None)
        contextual_scope = " ".join(
            str(getattr(evidence_context, field, "") or "")
            for field in ("section_title", "table_chart_title")
        )
        if contextual_scope:
            comparison_value = normalize_semantic_text(
                f"{value} {contextual_scope}"
            )
    requested_direction = (
        "increase" if re.search(
            r"\b(?:rise|rose|rising|increase|increased|growth|grew)\b|"
            r"(?:עלייה|עליה|עלה|עלתה|גדל|גדלה)",
            requested_target,
            re.IGNORECASE,
        ) else "decrease" if re.search(
            r"\b(?:fall|fell|falling|decrease|decreased|decline|declined)\b|"
            r"(?:ירידה|ירד|ירדה|פחת|פחתה)",
            requested_target,
            re.IGNORECASE,
        ) else ""
    )
    evidence_direction = (
        "increase" if re.search(
            r"\b(?:rise|rose|rising|increase|increased|growth|grew)\b|"
            r"(?:עלייה|עליה|עלה|עלתה|גדל|גדלה)",
            value,
            re.IGNORECASE,
        ) else "decrease" if re.search(
            r"\b(?:fall|fell|falling|decrease|decreased|decline|declined)\b|"
            r"(?:ירידה|ירד|ירדה|פחת|פחתה)",
            value,
            re.IGNORECASE,
        ) else ""
    )
    relational_event_framing = re.compile(
        r"\b(?:increase|increased|increasing|rise|rose|rising|growth|grew|"
        r"decrease|decreased|decreasing|fall|fell|falling|decline|declined|"
        r"change|changed|because|reason|cause|caused|explain|explains|explained|"
        r"report|reports|reported|why|what|how|whether|that)\b|"
        r"(?:עלייה|עליה|עלה|עלתה|גדל|גדלה|ירידה|ירד|ירדה|פחת|פחתה|"
        r"שינוי|השתנה|בגלל|בשל|עקב|סיבה|גרם|מסביר|מסבירה|דוח)",
        re.IGNORECASE,
    )
    requested_comparison = requested_target
    evidence_comparison = value
    if kind in {"cause_or_reason", "explanation"}:
        requested_comparison = relational_event_framing.sub(
            " ", requested_target
        )
        evidence_comparison = relational_event_framing.sub(" ", value)
    generic_relation_target_terms = {
        "index", "metric", "rate", "output", "demand", "pressure", "event",
        "change", "program", "study", "process", "method", "sample", "dataset",
        "data", "report", "system", "cohort", "population", "entity", "scope",
        "מדד", "שיעור", "תפוקה", "ביקוש", "לחץ", "אירוע", "שינוי", "תוכנית",
        "מחקר", "תהליך", "שיטה", "מדגם", "נתונים", "דוח", "מערכת", "אוכלוסייה",
    }
    requested_distinctive = (
        set(_terms(requested_comparison)) - generic_relation_target_terms
    )
    if kind in {"negative_relation", "positive_inclusion_or_count"}:
        evidence_comparison = comparison_value
    evidence_terms = set(_terms(evidence_comparison))
    distinctive_overlap = (
        len(requested_distinctive & evidence_terms)
        / len(requested_distinctive)
        if requested_distinctive else 1.0
    )
    if not value:
        reasons.append("relational_argument_unresolved")
    elif ((
        requested_direction
        and evidence_direction
        and requested_direction != evidence_direction
    ) or distinctive_overlap < 0.50 or _overlap(
        requested_comparison, evidence_comparison
    ) < 0.45) and not scoped_document_target_compatible(
        requirement, fact, requested_comparison, evidence_comparison
    ):
        reasons.append("relational_argument_incompatible")
    return tuple(dict.fromkeys(reasons))


def numeric_fact_binding_errors(
    requirement: TypedRequirement,
    fact: Any,
    *,
    typed_facts: Sequence[Any] = (),
    requirements: Sequence[TypedRequirement] = (),
    authorized_evidence_ids: Optional[Iterable[str]] = None,
) -> tuple[str, ...]:
    """Return field-level semantic role failures for a proposed numeric fact."""
    reasons: list[str] = []
    if str(getattr(fact, "fact_type", "") or "") == "numeric":
        reasons.extend(source_fact_population_errors(fact))
    compiler_owned = bool(getattr(requirement, "dimension_constraints", ()))

    def constrained(dimension: str, fallback: str = "") -> str:
        if compiler_owned:
            return requirement_dimension_value(requirement, dimension)
        return fallback

    if requirement.satisfaction_mode is RequirementSatisfactionMode.DERIVED_FACT:
        if str(getattr(fact, "fact_type", "") or "") != "derived_direction":
            return ("derived_requirement_missing_derived_fact",)
        if str(getattr(fact, "direction", "") or "") not in {
            "increase", "decrease", "stable",
        }:
            return ("derived_direction_unresolved",)
        reasons.extend(derived_direction_source_errors(fact, typed_facts))
        if reasons:
            return _unique(reasons)
        required = {item.id:item for item in requirements}
        linked = tuple(requirement.linked_requirement_ids)
        if len(linked) != 2 or len(set(linked)) != 2 or authorized_evidence_ids is None:
            return ("derived_endpoint_admission_unresolved",)
        endpoints = {item.fact_id:item for item in typed_facts}
        admitted_ids = set()
        for identifier in fact.source_typed_fact_ids:
            endpoint = endpoints[identifier]
            matches = [required[slot] for slot in linked if slot in required and slot in endpoint.requirement_ids]
            if len(matches) != 1 or matches[0].kind != "comparison_endpoint":
                reasons.append("derived_endpoint_requirement_mismatch")
                continue
            endpoint_requirement = matches[0]
            # Reuse the same canonical admission as source coverage. An
            # endpoint reference cannot bypass a rejected source dimension.
            from enterprise_retrieval import (
                canonical_fact_admissibility,
                coverage_requirement_from_canonical_requirement,
            )
            admission = canonical_fact_admissibility(
                coverage_requirement_from_canonical_requirement(endpoint_requirement),
                endpoint_requirement, endpoint,
            )
            if admission.reason_codes or numeric_fact_binding_errors(endpoint_requirement, endpoint):
                reasons.append("derived_endpoint_canonical_admission_failed")
            if not set(endpoint.source_evidence_unit_ids) <= set(authorized_evidence_ids):
                reasons.append("derived_endpoint_not_authorized")
            admitted_ids.add(endpoint_requirement.id)
        if admitted_ids != set(linked):
            reasons.append("derived_endpoint_requirement_incomplete")
        return _unique(reasons)
    from document_query import document_fact_errors, location_fact_errors
    reasons.extend(document_fact_errors(constrained("document_scope"), fact))
    reasons.extend(location_fact_errors(constrained("source_location"), fact))
    from financial_semantics import financial_fact_errors
    reasons.extend(financial_fact_errors(
        constrained("financial_role"), constrained("document_scope"), constrained("unit"), fact,
    ))
    role = str(getattr(fact, "numeric_role", "requested_answer_value") or "")
    if role in {'document_identifier', 'entity_identifier'} or str(
        getattr(fact, 'evidence_numeric_role', '') or '') in {'document_identifier', 'entity_identifier'}:
        reasons.append('identifier_not_scalar_answer')
    unit = normalize_semantic_text(getattr(fact, "unit", "")).casefold()
    value = str(getattr(fact, "value", "") or "")
    evidence_subject = normalize_semantic_text(
        getattr(fact, "evidence_subject", "") or ""
    )
    evidence_predicate = str(
        getattr(fact, "evidence_predicate", "") or ""
    )
    evidence_target = normalize_semantic_text(
        getattr(fact, "evidence_target_entity", "") or ""
    )
    evidence_unit = normalize_semantic_text(
        getattr(fact, "evidence_unit", "") or ""
    ).casefold()
    evidence_span = tuple(getattr(fact, "evidence_span", ()) or ())
    evidence_period = normalize_semantic_text(
        getattr(fact, "evidence_period", "") or ""
    )
    evidence_population = normalize_semantic_text(
        getattr(fact, "evidence_population", "") or ""
    )
    if requirement.kind == "filtered_row_period":
        fact_period = normalize_semantic_text(getattr(fact, "period", "") or "")
        if not value or value != fact_period or evidence_period != fact_period:
            reasons.append("filtered_row_period_unbound")
        if not evidence_subject:
            reasons.append("evidence_subject_unresolved")
        if not evidence_predicate:
            reasons.append("evidence_predicate_unresolved")
        if len(evidence_span) != 2 or int(evidence_span[1]) <= int(evidence_span[0]):
            reasons.append("evidence_span_unbound")
        return _unique(reasons)
    if requirement.value_type in _TEXTUAL_EVIDENCE_VALUE_TYPES:
        return ("proposition_relation_not_supported",)
    if not value and requirement.value_type not in {"direction", "scope", "matrix", "category"}:
        reasons.append("answer_value_missing")
    if not evidence_subject:
        reasons.append("evidence_subject_unresolved")
    if not evidence_predicate:
        reasons.append("evidence_predicate_unresolved")
    # An explicitly requested unit must bind exactly.  A scalar numeric
    # request may deliberately leave its unit open, in which case the source
    # still has to bind a concrete unit/value role but need not equal an
    # invented legacy default.
    if not evidence_unit or (unit and evidence_unit != unit):
        reasons.append("evidence_unit_unbound")
    if len(evidence_span) != 2 or int(evidence_span[1]) <= int(evidence_span[0]):
        reasons.append("evidence_span_unbound")
    # A route can legitimately describe many answers as a "count".  Count is
    # an answer shape, not a source metric: require the source-derived fact to
    # retain at least one concrete request subject term when such a term was
    # supplied.  This prevents, for example, a company count from satisfying
    # a thesis count merely because both have a number and period.
    generic_subject_terms = {
        "what", "which", "how", "many", "count", "number", "total",
        "of", "the", "a", "an", "was", "were", "is", "are", "in",
        "for", "with", "and", "or", "companies", "company", "workers",
        "employees", "organizations", "organization", "חברות", "חברה",
        "עובדים", "ארגונים", "ארגון", "כמה", "מספר", "סך", "של",
    }
    constrained_subject = constrained("subject", requirement.subject)
    requested_subject_terms = {
        token.casefold()
        for token in _TOKEN_RE.findall(
            normalize_semantic_text(constrained_subject or "")
        )
        if token.casefold() not in generic_subject_terms
    }
    observed_semantics = " ".join((
        evidence_subject,
        evidence_target,
        str(getattr(fact, "source_claim", "") or ""),
    ))
    observed_terms = {
        token.casefold() for token in _TOKEN_RE.findall(observed_semantics)
    }
    if (
        requested_subject_terms
        and not (requested_subject_terms & observed_terms)
        and _overlap(constrained_subject, observed_semantics) <= 0
    ):
        reasons.append("evidence_subject_metric_incompatible")
    requested_target = normalize_semantic_text(constrained(
        "target_entity", requirement.counted_entity or requirement.target_entity_type,
    ))
    observed_facet = str(
        getattr(fact, "evidence_facet", "")
        or getattr(fact, "facet", "") or ""
    )
    if requested_target:
        if requirement.kind in {"winner", "winner_attribute"}:
            if requirement.facet:
                if not observed_facet or _overlap(
                    requirement.facet.replace("_", " "),
                    observed_facet.replace("_", " "),
                ) <= 0:
                    reasons.append("evidence_target_facet_incompatible")
            elif not evidence_target or _overlap(
                requested_target, evidence_target
            ) <= 0:
                reasons.append("evidence_target_entity_incompatible")
        elif requirement.kind not in {"filtered_row_value"} and (
            not evidence_target
            or (
                _overlap(requested_target, evidence_target) <= 0
                and not canonical_entity_equivalent(
                    requested_target, evidence_target,
                )
            )
        ):
            reasons.append("evidence_target_entity_incompatible")
    expected_predicates = {
        "maximum_of": {"has_count", "has_percentage"},
        "minimum_of": {"has_count", "has_percentage"},
        "has_count": {"has_count", "included_in", "employs_workers", "recruits_employees"},
        "compared_with": {"has_count", "has_percentage", "changed_between"},
        "changed_between": {"has_count", "has_percentage", "changed_between"},
        "stated_as": {"has_count", "has_percentage", "employs_workers", "recruits_employees", "raises_capital"},
        "satisfies_filter": {"stated_as", "has_count", "has_percentage"},
    }
    constrained_predicate = constrained("predicate", requirement.predicate)
    if constrained_predicate and evidence_predicate not in expected_predicates.get(
        constrained_predicate, {constrained_predicate}
    ):
        reasons.append("evidence_predicate_incompatible")
    constrained_periods = tuple(
        item for item in constrained("period", "|".join(requirement.periods)).split("|") if item
    )
    if constrained_periods and not any(
        _period_compatible((period,), evidence_period)
        for period in constrained_periods
    ):
        reasons.append("evidence_period_incompatible")
    constrained_population = constrained(
        "population", requirement.scope_population,
    )
    if constrained_population:
        scope_ok, scope_reason = population_scope_compatibility(
            constrained_population, evidence_population
        )
        if re.match(r"(?:at|in|within)\s+(?:the\s+)?[A-Z]|(?:בחברת|בחברה|בארגון|אצל)\s+", constrained_population):
            scope_ok = source_population_matches_scope(constrained_population, fact)
            scope_reason = "scope_exact_or_source_alias" if scope_ok else "evidence_scope_incompatible"
        if not scope_ok:
            reasons.append(scope_reason)
    legacy_observed = quantity_semantics_from_scalar_role(
        getattr(fact, "scalar_role", "") or role,
    )
    observed_quantity_kind = normalize_quantity_kind(
        getattr(fact, "quantity_kind", "")
    )
    # Compatibility adapters from earlier phases may carry a typed source
    # unit/numeric role without the new C18 fields.  Recover only that
    # source-side kind; never infer a missing role from a requirement.
    if observed_quantity_kind == QuantityKind.UNSPECIFIED.value:
        evidence_numeric_role = str(
            getattr(fact, "evidence_numeric_role", "") or role
        ).casefold()
        if unit in {"%", "percent", "percentage", "אחוז"}:
            observed_quantity_kind = QuantityKind.PERCENTAGE.value
        elif (
            unit in _COUNT_UNITS
            or evidence_unit in _COUNT_UNITS
            or evidence_numeric_role in {"count", "category_value"}
        ):
            observed_quantity_kind = QuantityKind.COUNT.value
        else:
            observed_quantity_kind = legacy_observed.quantity_kind
    observed_statistical_role = normalize_statistical_role(
        getattr(fact, "statistical_role", "") or legacy_observed.statistical_role
    )
    observed_aggregation_role = normalize_aggregation_role(
        getattr(fact, "aggregation_role", "") or legacy_observed.aggregation_role
    )
    # A winner is derived from a complete compatible matrix.  Every
    # source-backed quantity requirement instead compares each independently
    # requested dimension, with no point-estimate/total shortcut.
    if requirement.kind != "winner":
        for dimension, required, observed, reason in (
            ("quantity_kind", constrained("quantity_kind", requirement.quantity_kind), observed_quantity_kind,
             "quantity_kind_incompatible"),
            ("statistical_role", constrained("statistical_role", requirement.statistical_role),
             observed_statistical_role, "statistical_role_incompatible"),
            ("aggregation_role", constrained("aggregation_role", requirement.aggregation_role),
             observed_aggregation_role, "aggregation_role_incompatible"),
        ):
            matches, _decision = quantity_dimension_compatibility(
                required, observed, dimension=dimension,
            )
            if not matches:
                reasons.extend((reason, "wrong_numeric_role"))
    if requirement.output_slot == "count" or requirement.kind == "count":
        if observed_quantity_kind != QuantityKind.COUNT.value:
            reasons.append("wrong_numeric_role")
        if unit in {"%", "percent", "percentage", "אחוז"}:
            reasons.append("wrong_unit")
    if requirement.output_slot == "percentage" or requirement.kind == "percentage":
        if observed_quantity_kind != QuantityKind.PERCENTAGE.value:
            reasons.append("wrong_numeric_role")
        if unit not in {"%", "percent", "percentage", "אחוז"}:
            reasons.append("wrong_unit")
    if requirement.kind == "filtered_row_value":
        requested_metric = " ".join(requirement.objects_or_metrics)
        if requested_metric and _overlap(requested_metric, evidence_subject) <= 0:
            reasons.append("evidence_subject_metric_incompatible")
    structural_type = str(getattr(fact, "structural_type", "body") or "")
    if structural_type in {"table_row", "chart_data"} and not bool(
        getattr(fact, "row_column_bound", False)
    ):
        reasons.extend(("row_header_unbound", "column_header_unbound"))
    if requirement.endpoint_type == "period":
        if not _period_compatible((requirement.endpoint_value,), str(getattr(fact, "period", "") or "")):
            reasons.append("wrong_period")
    if requirement.endpoint_type == "population":
        if not _population_compatible(
            requirement.endpoint_value,
            str(getattr(fact, "population", "") or getattr(fact, "source_claim", "")),
        ):
            reasons.append("wrong_population")
    constrained_facet = constrained("facet", requirement.facet)
    if constrained_facet.startswith("experience_band:"):
        requested_metrics = set(explicit_metric_concepts(constrained("metric", " ".join(requirement.objects_or_metrics))))
        source_metrics = set(explicit_metric_concepts(str(getattr(fact, "metric", "") or "")))
        if requested_metrics and not requested_metrics <= source_metrics:
            reasons.append("evidence_subject_metric_incompatible")
    if constrained_facet and not source_facet_compatible(constrained_facet, source_fact_facets(fact)):
        reasons.append("wrong_facet")
    return _unique(reasons)


def _c45_load_diagnostics():
    """Contain initialization before requests; no lazy import wait at the edge."""
    try:
        import c45_diagnostics
        return c45_diagnostics
    except Exception:
        return None


_C45_DIAGNOSTICS = _c45_load_diagnostics()


def _c45_observe(witness_family_id):
    """Bounded diagnostic containment; no application evaluation belongs here."""
    try:
        _C45_DIAGNOSTICS.observe(witness_family_id)
    except Exception:
        pass


def build_claim_graph(
    contract: RequestExecutionContract,
    claim_bindings: Sequence[Any],
    typed_facts: Sequence[Any],
    evidence_to_citation: Mapping[str, str],
    *,
    standalone_complete: bool = True,
    precovered_requirement_ids: Iterable[str] = (),
) -> ClaimGraph:
    """Bind pre-rendered claims through facts to evidence and citations.

    No lexical overlap, citation-marker presence, or answer non-emptiness is
    used as proof.  A claim receives multiple requirements only when distinct
    typed facts in its cited evidence explicitly carry those requirements.
    """
    canonical_ids = contract.requirement_ids
    facts_by_evidence: dict[str, list[Any]] = {}
    for fact in typed_facts:
        fact_requirement_ids = tuple(
            getattr(fact, "requirement_ids", ())
            or (getattr(fact, "requirement_id", ""),)
        )
        if not set(fact_requirement_ids) <= canonical_ids:
            raise RuntimeError("claim graph received noncanonical fact requirement ID")
        evidence_ids = tuple(dict.fromkeys(
            str(value)
            for value in (
                getattr(fact, "evidence_id", ""),
                *(getattr(fact, "source_evidence_unit_ids", ()) or ()),
            )
            if value
        ))
        for evidence_id in evidence_ids:
            facts_by_evidence.setdefault(evidence_id, []).append(fact)

    def claim_supports_fact(
        claim_text: str,
        fact: Any,
        requirement: TypedRequirement,
        candidate_facts: Sequence[Any],
        cited_evidence_ids: Sequence[str],
        selected_fact_ids: set[str],
    ) -> bool:
        body = normalize_semantic_text(
            _CLAIM_CITATION_RE.sub(" ", claim_text or "")
        )
        if not body:
            return False
        fact_type = str(getattr(fact, "fact_type", "numeric") or "numeric")
        population_facts = (fact,)
        if fact_type == "numeric":
            # A comparison's displayed owner/alias may be defined by either
            # contributing endpoint. Use only its actual claim-cited pair,
            # after independently validating the existing canonical bridge,
            # scope and derivation snapshots. Surface aliases never join facts.
            for comparison in candidate_facts:
                if str(getattr(comparison, "fact_type", "")) != "derived_direction":
                    continue
                endpoint_ids = tuple(getattr(comparison, "source_typed_fact_ids", ()) or ())
                if getattr(fact, "fact_id", "") not in endpoint_ids:
                    continue
                if selected_fact_ids and not {comparison.fact_id, *endpoint_ids} <= selected_fact_ids:
                    continue
                endpoints = tuple({item.fact_id: item for item in candidate_facts
                                   if getattr(item, "fact_type", "") == "numeric"
                                   and item.fact_id in endpoint_ids}.values())
                if (len(endpoint_ids) == 2 and len(endpoints) == 2
                        and rendered_direction_supports_endpoints(claim_text, comparison.direction, endpoints)
                        and not derived_direction_source_errors(comparison, typed_facts)):
                    population_facts = endpoints
                    break
        if fact_type == "derived_direction":
            # A computed direction carries endpoint identities, not a source
            # quote/context. Resolve the actual endpoints for scope evidence;
            # the derivation contract below still validates both bindings.
            endpoint_ids = tuple(getattr(fact, "source_typed_fact_ids", ()) or ())
            population_facts = tuple(item for item in typed_facts
                                     if getattr(item, "fact_id", "") in endpoint_ids)
            if len(endpoint_ids) != 2 or len(population_facts) != 2:
                return False
        if fact_type in {"numeric", "derived_direction"} and rendered_population_scope_errors(claim_text, population_facts):
            return False
        if fact_type == "text_proposition":
            object_text = str(
                getattr(fact, "object_text", "")
                or getattr(fact, "source_claim", "")
            )
            if (requirement.kind in {"list", "list_attributes"}
                    and requirement.predicate not in DOCUMENT_ROLE_PREDICATES
                    and str(getattr(fact, "evidence_predicate", "")) not in DOCUMENT_ROLE_PREDICATES):
                return _overlap(str(getattr(fact, "subject", "")), body) >= 0.18
            if text_fact_binding_errors(requirement, fact):
                return False
            relation_output_role = {
                "cause_or_reason": "reason",
                "explanation": "explanation",
                "purpose": "purpose",
                "intended_use": "purpose",
                "definition": "definition",
                "procedure": "method",
                "negative_relation": "excluded_entity",
                "positive_inclusion_or_count": "included_entity",
            }.get(requirement.kind, "")
            relation_arguments = {
                str(getattr(item, "role", "") or ""):
                str(getattr(item, "value", "") or "")
                for item in getattr(fact, "relation_arguments", ()) or ()
            }
            source_role = str(getattr(fact, "evidence_predicate", ""))
            if requirement.predicate in DOCUMENT_ROLE_PREDICATES or source_role in DOCUMENT_ROLE_PREDICATES:
                if rendered_document_role_errors(
                    claim_text, candidate_facts, contract.requirements,
                    cited_evidence_ids=cited_evidence_ids,
                ):
                    return False
                propositions, errors = parse_document_role_propositions(claim_text)
                if errors:
                    # Only an exact actor-only answer may be contextual.
                    return not rendered_document_role_errors(claim_text, (fact,), (requirement,))
                target_role, actor_role = RELATION_ARGUMENT_SCHEMAS[source_role]
                source = DocumentRoleProposition(source_role,
                    relation_arguments.get(actor_role, ""), relation_arguments.get(target_role, ""))
                return any(document_role_proposition_compatible(item, source) for item in propositions)
            if (
                relation_output_role
                and relation_arguments.get(relation_output_role)
                and _overlap(
                    relation_arguments[relation_output_role], body
                ) >= 0.18
            ):
                return True
            return bool(
                object_text
                and proposition_relation_supported(requirement, object_text)
                and _overlap(object_text, body) >= 0.10
                and (
                    requirement.predicate in {"stated_as", "defined_as"}
                    or _predicate_compatible(requirement, body)
                    or _overlap(requirement.predicate, body) > 0
                    or _overlap(requirement.subject, body) >= 0.10
                )
            )
        if fact_type == "derived_direction":
            if numeric_fact_binding_errors(requirement, fact, typed_facts=typed_facts,
                    requirements=contract.requirements, authorized_evidence_ids=evidence_to_citation):
                return False
            direction = str(getattr(fact, "direction", "") or "")
            return rendered_direction_supports_endpoints(claim_text, direction, population_facts)
        if numeric_fact_binding_errors(requirement, fact):
            return False
        value = str(getattr(fact, "value", "") or "")
        value_tokens = _numeric_values(value)
        if value_tokens and not value_tokens <= _numeric_values(body):
            return False
        numeric_role = str(
            getattr(fact, "numeric_role", "requested_answer_value") or ""
        )
        if numeric_role in {
            "filter_threshold", "period", "table_header", "unrelated_neighbor",
        }:
            return False
        if numeric_role == "confidence_level" and not re.search(
            r"confidence|ביטחון", " ".join((
                requirement.subject,
                requirement.predicate,
                " ".join(requirement.objects_or_metrics),
            )), re.IGNORECASE,
        ):
            return False
        unit = str(getattr(fact, "unit", "") or "")
        if unit == "%" and not _PERCENT_RE.search(body):
            return False
        if (
            unit
            and unit.casefold() not in {"%", "number", "numeric", "count"}
            and requirement.kind != "winner"
        ):
            normalized_unit = normalize_semantic_text(unit).casefold()
            normalized_body = normalize_semantic_text(body).casefold()
            equivalents = _UNIT_EQUIVALENCE.get(
                normalized_unit, (normalized_unit,)
            )
            if not any(value and value in normalized_body for value in equivalents):
                return False
        if requirement.kind in {"winner", "negative_relation"}:
            subject = str(getattr(fact, "subject", "") or "")
            if subject and _overlap(subject, body) < 0.16:
                return False
        # Source-role compatibility (subject, predicate, period, population,
        # and unit) was established above from the fact's evidence fields.
        # The rendered claim may be contextually concise, but it can only bind
        # this already source-derived fact; request slots never populate it.
        return bool(value_tokens or getattr(fact, "object_text", ""))

    allocated: set[str] = set(precovered_requirement_ids)
    allocation_counts: dict[str, int] = {
        requirement_id: 1 for requirement_id in allocated
    }
    requirements_by_id = {item.id: item for item in contract.requirements}
    if not allocated <= canonical_ids:
        raise RuntimeError("claim graph received noncanonical precoverage ID")
    nodes: list[ClaimNode] = []
    citation_links: list[CitationBinding] = []
    for index, binding in enumerate(claim_bindings, start=1):
        claim_id = str(getattr(binding, "claim_id", "") or f"claim-{index}")
        evidence_ids = tuple(dict.fromkeys(
            str(item) for item in getattr(binding, "evidence_ids", ()) if item
        ))
        candidate_facts = tuple(
            fact for evidence_id in evidence_ids
            for fact in facts_by_evidence.get(evidence_id, ())
        )
        declared_fact_ids = set(getattr(binding, "fact_ids", ()) or ())
        declared_requirement_ids = set(
            getattr(binding, "requirement_ids", ()) or ()
        )
        by_requirement: dict[str, Any] = {}
        for fact in candidate_facts:
            if declared_fact_ids and getattr(fact, "fact_id", "") not in declared_fact_ids:
                continue
            ids = tuple(
                getattr(fact, "requirement_ids", ())
                or (getattr(fact, "requirement_id", ""),)
            )
            # A single adapter fact cannot be reused to fake independent
            # coverage. Distinct fact IDs are required for distinct req IDs.
            for requirement_id in ids:
                requirement = requirements_by_id.get(requirement_id)
                if requirement is None:
                    continue
                if declared_requirement_ids and requirement_id not in declared_requirement_ids:
                    continue
                if not claim_supports_fact(
                    str(getattr(binding, "claim", "") or ""),
                    fact,
                    requirement,
                    candidate_facts,
                    evidence_ids,
                    declared_fact_ids,
                ):
                    continue
                reusable_list_slot = bool(
                    requirement
                    and requirement.kind in {"list", "list_attributes"}
                    and allocation_counts.get(requirement_id, 0)
                    < max(1, requirement.cardinality)
                )
                if requirement_id and (
                    requirement_id not in allocated or reusable_list_slot
                ):
                    by_requirement.setdefault(requirement_id, fact)
                    break
        requirement_ids = tuple(by_requirement)
        fact_ids = tuple(dict.fromkeys(
            str(getattr(fact, "fact_id", ""))
            for fact in by_requirement.values()
            if getattr(fact, "fact_id", "")
        ))
        citation_ids = tuple(dict.fromkeys(
            evidence_to_citation[evidence_id]
            for evidence_id in evidence_ids
            if evidence_id in evidence_to_citation
        ))
        declared_supported = bool(getattr(binding, "supported", False))
        supported = bool(
            declared_supported and requirement_ids and fact_ids
            and citation_ids and len(fact_ids) >= len(requirement_ids)
        )
        reasons: list[str] = []
        if not declared_supported:
            reasons.append("synthesis_binding_failed")
        if not requirement_ids:
            reasons.append("canonical_requirement_binding_missing")
        if len(fact_ids) < len(requirement_ids):
            reasons.append("distinct_fact_per_requirement_missing")
        if not citation_ids:
            reasons.append("direct_citation_binding_missing")
        # Keep admitted requirement/fact coverage as a diagnostic even when
        # the rendered claim also contains an unbound quantity. The whole
        # claim and its direct citations still fail below; coverage alone is
        # never a graph or final-response acceptance decision.
        admitted_coverage = supported
        value_errors = rendered_fact_value_errors(
            str(getattr(binding, "claim", "") or ""), tuple(by_requirement.values()))
        if value_errors:
            supported = False
            reasons.extend(value_errors)
        node = ClaimNode(
            claim_id=claim_id,
            text=str(getattr(binding, "claim", "") or ""),
            requirement_ids=requirement_ids,
            fact_ids=fact_ids,
            evidence_unit_ids=evidence_ids,
            citation_ids=citation_ids,
            supported=supported,
            reason_codes=tuple(reasons),
        )
        nodes.append(node)
        if admitted_coverage:
            allocated.update(requirement_ids)
            for requirement_id in requirement_ids:
                allocation_counts[requirement_id] = (
                    allocation_counts.get(requirement_id, 0) + 1
                )
        for evidence_id in evidence_ids:
            citation_id = evidence_to_citation.get(evidence_id)
            if citation_id:
                evidence_fact_ids = tuple(dict.fromkeys(
                    str(getattr(fact, "fact_id", ""))
                    for fact in candidate_facts
                    if evidence_id in {
                        str(getattr(fact, "evidence_id", "") or ""),
                        *(
                            str(value)
                            for value in getattr(
                                fact, "source_evidence_unit_ids", ()
                            ) or ()
                        ),
                    }
                    and getattr(fact, "fact_id", "") in fact_ids
                ))
                citation_links.append(CitationBinding(
                    citation_id=citation_id,
                    claim_id=claim_id,
                    fact_ids=evidence_fact_ids,
                    evidence_unit_id=evidence_id,
                    direct=bool(supported and evidence_fact_ids),
                ))
    mandatory = {
        item.id for item in contract.requirements if item.mandatory
    }
    satisfied = {
        requirement_id for requirement_id in allocated
        if allocation_counts.get(requirement_id, 0) >= (
            max(1, requirements_by_id[requirement_id].cardinality)
            if requirements_by_id[requirement_id].kind in {"list", "list_attributes"}
            else 1
        )
    }
    missing = tuple(sorted(mandatory - satisfied))
    provenance_nodes: list[FactProvenanceNode] = []
    for fact in typed_facts:
        context = getattr(fact, "evidence_context", None)
        context_payload = (
            context.as_dict()
            if context is not None and hasattr(context, "as_dict") else {}
        )
        provenance_nodes.append(FactProvenanceNode(
            fact_id=str(getattr(fact, "fact_id", "") or ""),
            evidence_unit_id=str(getattr(fact, "evidence_id", "") or ""),
            candidate_id=str(getattr(fact, "candidate_id", "") or ""),
            source_evidence_unit_ids=tuple(
                getattr(fact, "source_evidence_unit_ids", ()) or ()
            ),
            source_candidate_ids=tuple(
                getattr(fact, "source_candidate_ids", ()) or ()
            ),
            source_typed_fact_ids=tuple(
                getattr(fact, "source_typed_fact_ids", ()) or ()
            ),
            operator_derivation_id=str(
                getattr(fact, "operator_derivation_id", "") or ""
            ),
            derivation_description=str(
                getattr(fact, "derivation_description", "") or ""
            ),
            evidence_context=context_payload,
            evidence_atom_id=str(
                getattr(getattr(fact, "evidence_atom", None), "atom_id", "")
                or ""
            ),
        ))
    _c45_application_value = ClaimGraph(
        contract_id=contract.contract_id,
        contract_sha256=contract.contract_sha256,
        claims=tuple(nodes),
        citation_bindings=tuple(citation_links),
        covered_requirement_ids=tuple(sorted(satisfied)),
        missing_requirement_ids=missing,
        contextual_factual_correctness=bool(nodes) and all(
            item.supported for item in nodes
        ),
        standalone_completeness=bool(standalone_complete),
        fact_provenance=tuple(provenance_nodes),
    )
    _c45_observe('C44E2E2_FAMILY_04_20ba9dc99d4b')
    return _c45_application_value


@dataclass(frozen=True)
class ExecutionBoundFact:
    fact_id: str
    requirement_id: str
    requirement_ids: tuple[str, ...]
    evidence_id: str
    subject: str
    predicate: str
    object_text: str
    value: str = ""
    unit: str = ""
    period: str = ""
    time_role: str = ""
    population: str = ""
    qualifiers: tuple[str, ...] = ()
    source_claim: str = ""
    metric: str = ""
    facet: str = ""
    numeric_role: str = "requested_answer_value"
    # Empty preserves the distinction between an explicitly classified source
    # role and an older adapter that only carries the coarse numeric role.
    # ``__post_init__`` canonicalizes the latter at this boundary; request
    # semantics are never used to fill it.
    scalar_role: str = ""
    structural_type: str = "body"
    row_column_bound: bool = True
    row_header: str = ""
    column_header: str = ""
    fact_type: str = "numeric"
    evidence_subject: str = ""
    evidence_predicate: str = ""
    evidence_target_entity: str = ""
    evidence_population: str = ""
    evidence_period: str = ""
    evidence_unit: str = ""
    evidence_numeric_role: str = ""
    evidence_span: tuple[int, int] = (0, 0)
    evidence_facet: str = ""
    candidate_id: str = ""
    source_evidence_unit_ids: tuple[str, ...] = ()
    source_candidate_ids: tuple[str, ...] = ()
    source_typed_fact_ids: tuple[str, ...] = ()
    operator_derivation_id: str = ""
    derivation_description: str = ""
    origin_debug_id: str = ""
    evidence_context: Any = field(default=None, repr=False, compare=False)
    filter_rows: tuple[tuple[str, str, str], ...] = ()
    relation_arguments: tuple[RelationArgument, ...] = ()
    canonical_valid: bool = True
    canonical_validation_errors: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "scalar_role", normalize_scalar_role(
            self.scalar_role or self.numeric_role
        ))
        relation = str(self.evidence_predicate or self.predicate or "")
        errors = list(self.canonical_validation_errors)
        if relation in RELATION_ARGUMENT_SCHEMAS:
            arguments = {item.role: item for item in self.relation_arguments}
            errors.extend(
                f"relation_argument_missing:{role}"
                for role in RELATION_ARGUMENT_SCHEMAS[relation]
                if role not in arguments
            )
            if self.evidence_context is None:
                errors.append("relation_argument_provenance_missing")
            else:
                errors.extend(relation_argument_validation_errors(
                    self.evidence_context,
                    predicate=relation,
                ))
        errors = list(dict.fromkeys(errors))
        object.__setattr__(self, "canonical_validation_errors", tuple(errors))
        object.__setattr__(self, "canonical_valid", not errors)

    def as_prompt_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "requirement_id": self.requirement_id,
            "requirement_ids": list(self.requirement_ids),
            "evidence_id": self.evidence_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object_text": self.object_text,
            "value": self.value,
            "unit": self.unit,
            "period": self.period,
            "time_role": self.time_role,
            "population": self.population,
            "qualifiers": list(self.qualifiers),
            "metric": self.metric,
            "facet": self.facet,
            "numeric_role": self.numeric_role,
            "scalar_role": self.scalar_role,
            "structural_type": self.structural_type,
            "row_column_bound": self.row_column_bound,
            "row_header": self.row_header,
            "column_header": self.column_header,
            "fact_type": self.fact_type,
            "evidence_subject": self.evidence_subject,
            "evidence_predicate": self.evidence_predicate,
            "evidence_target_entity": self.evidence_target_entity,
            "evidence_population": self.evidence_population,
            "evidence_period": self.evidence_period,
            "evidence_unit": self.evidence_unit,
            "evidence_numeric_role": self.evidence_numeric_role,
            "evidence_span": list(self.evidence_span),
            "evidence_facet": self.evidence_facet,
            "candidate_id": self.candidate_id,
            "source_evidence_unit_ids": list(self.source_evidence_unit_ids),
            "source_candidate_ids": list(self.source_candidate_ids),
            "source_typed_fact_ids": list(self.source_typed_fact_ids),
            "operator_derivation_id": self.operator_derivation_id,
            "derivation_description": self.derivation_description,
            "origin_debug_id": self.origin_debug_id,
            "evidence_context": (
                self.evidence_context.as_dict()
                if self.evidence_context is not None
                and hasattr(self.evidence_context, "as_dict") else None
            ),
            "filter_rows": [list(item) for item in self.filter_rows],
            "relation_arguments": [
                item.as_dict() for item in self.relation_arguments
            ],
            "canonical_valid": self.canonical_valid,
            "canonical_validation_errors": list(
                self.canonical_validation_errors
            ),
        }


def bind_operator_output_facts(
    contract: RequestExecutionContract,
    execution: Any,
) -> tuple[tuple[ExecutionBoundFact, ...], tuple[str, ...]]:
    """Bind operator output requirements without fabricating matrix claims."""
    result = getattr(execution, "result", None)
    if result is None or not getattr(execution, "accepted", False):
        _c45_application_value = (), ()
        _c45_observe('C44E2E2_FAMILY_03_0f14aac5cc1b')
        return _c45_application_value
    internal_kinds = {
        "category_set", "category_value_bindings",
        "common_metric_unit_period_population",
    }
    precovered = tuple(
        item.id
        for item in contract.requirements
        if item.kind in internal_kinds
        # A request condition is validated by the accepted operator execution;
        # it is not a source fact about an emitted result value.  Treating the
        # latter as evidence for the former would both erase the condition's
        # scalar role and permit a result value to satisfy its own filter.
        or item.satisfaction_mode
        is RequirementSatisfactionMode.REQUEST_CONDITION
    )
    attributes = tuple(getattr(execution, "result_attributes", ()) or (result,))
    operation_name = str(
        getattr(getattr(execution, "operation", ""), "value", "")
        or getattr(execution, "operation", "")
    ).upper()
    filter_rows = (
        tuple(
            (
                str(getattr(item, "period", "") or ""),
                str(getattr(item, "value", "") or ""),
                str(getattr(item, "unit", "") or ""),
            )
            for item in attributes
            if getattr(item, "period", "") and getattr(item, "value", "")
        )
        if operation_name == "FILTER" else ()
    )
    output: list[ExecutionBoundFact] = []
    for requirement in contract.requirements:
        if (
            requirement.kind in internal_kinds
            or requirement.satisfaction_mode
            is RequirementSatisfactionMode.REQUEST_CONDITION
        ):
            continue
        source = result
        if requirement.kind == "winner_attribute":
            expected_units = {str(item).casefold() for item in requirement.units}
            expected_metrics = {str(item).casefold() for item in requirement.objects_or_metrics}
            source = next((
                fact for fact in attributes
                if (
                    not expected_units
                    or str(getattr(fact, "unit", "")).casefold() in expected_units
                    or ("%" in expected_units and str(getattr(fact, "unit", "")) == "%")
                ) and (
                    requirement.output_slot == "percentage"
                    and str(getattr(fact, "unit", "")).casefold()
                    in {"%", "percent", "percentage", "אחוז"}
                    or requirement.output_slot == "count"
                    and str(getattr(fact, "unit", "")).casefold()
                    not in {"%", "percent", "percentage", "אחוז"}
                    or
                    not expected_metrics
                    or any(metric in str(getattr(fact, "metric", "")).casefold() for metric in expected_metrics)
                    or "count" in expected_metrics and str(getattr(fact, "unit", "")).casefold() not in {"%", "percent", "percentage"}
                )
            ), None)
            if source is None:
                continue
        fact_value = str(getattr(source, "value", ""))
        fact_unit = str(getattr(source, "unit", ""))
        fact_period = str(getattr(source, "period", ""))
        if requirement.kind == "filtered_row_period":
            if not filter_rows:
                continue
            fact_value = filter_rows[0][0]
            fact_unit = ""
            fact_period = filter_rows[0][0]
        elif requirement.kind == "filtered_row_value":
            if not filter_rows:
                continue
            fact_value = filter_rows[0][1]
            fact_unit = filter_rows[0][2]
            fact_period = filter_rows[0][0]
        evidence_id = str(getattr(source, "evidence_id", "") or "")
        candidate_id = str(getattr(source, "candidate_id", "") or "")
        if not evidence_id or evidence_id.startswith("operator-hit-"):
            continue
        payload = "\x1f".join((
            requirement.id, evidence_id, str(getattr(source, "entity", "")),
            fact_value, fact_unit,
        ))
        output.append(ExecutionBoundFact(
            fact_id="fact-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24],
            requirement_id=requirement.id,
            requirement_ids=(requirement.id,),
            evidence_id=evidence_id,
            subject=str(getattr(source, "entity", "")),
            predicate=str(getattr(source, "predicate", "")),
            object_text=str(getattr(source, "source_claim", "")),
            value=fact_value,
            unit=fact_unit,
            period=fact_period,
            population=str(getattr(source, "population", "")),
            qualifiers=tuple(getattr(source, "qualifiers", ()) or ()),
            source_claim=str(getattr(source, "source_claim", "")),
            metric=str(getattr(source, "metric", "")),
            facet=str(getattr(source, "evidence_facet", "") or ""),
            numeric_role=str(
                getattr(source, "evidence_numeric_role", "")
                or "requested_answer_value"
            ),
            scalar_role=str(getattr(source, "scalar_role", "") or ""),
            fact_type=(
                "text_proposition"
                if requirement.value_type in _TEXTUAL_EVIDENCE_VALUE_TYPES
                else "numeric"
            ),
            evidence_subject=str(getattr(source, "evidence_subject", "") or ""),
            evidence_predicate=str(getattr(source, "evidence_predicate", "") or ""),
            evidence_target_entity=str(getattr(source, "evidence_target_entity", "") or ""),
            evidence_population=str(getattr(source, "evidence_population", "") or ""),
            evidence_period=str(getattr(source, "evidence_period", "") or ""),
            evidence_unit=str(getattr(source, "evidence_unit", "") or ""),
            evidence_numeric_role=str(getattr(source, "evidence_numeric_role", "") or ""),
            evidence_span=tuple(getattr(source, "evidence_span", ()) or ()),
            evidence_facet=str(getattr(source, "evidence_facet", "") or ""),
            candidate_id=candidate_id,
            source_evidence_unit_ids=tuple(
                getattr(source, "source_evidence_unit_ids", ()) or (evidence_id,)
            ),
            source_candidate_ids=tuple(
                getattr(source, "source_candidate_ids", ())
                or ((candidate_id,) if candidate_id else ())
            ),
            source_typed_fact_ids=tuple(
                getattr(source, "source_typed_fact_ids", ())
                or ((str(getattr(source, "fact_id", "")),)
                    if getattr(source, "fact_id", "") else ())
            ),
            operator_derivation_id=str(
                getattr(source, "operator_derivation_id", "") or ""
            ),
            derivation_description=str(
                getattr(source, "derivation_description", "") or ""
            ),
            origin_debug_id=str(getattr(source, "origin_debug_id", "") or ""),
            evidence_context=getattr(source, "evidence_context", None),
            filter_rows=filter_rows,
            relation_arguments=tuple(
                getattr(source, "relation_arguments", ())
                or getattr(
                    getattr(source, "evidence_context", None),
                    "relation_arguments",
                    (),
                )
            ),
        ))
    _c45_application_value = tuple(output), precovered
    _c45_observe('C44E2E2_FAMILY_03_0f14aac5cc1b')
    return _c45_application_value


_LEXICAL_CHAR_CLASS = r"A-Za-z0-9\u00c0-\u02af\u0370-\u052f\u0590-\u05ff"
_QUESTION_TOKEN = (
    r"what|which|who|why|how|do|does|did|is|are|was|were|employ|"
    r"recruit|hire|raise|מה|מהו|מהי|מהם|מהן|אילו|מי|מדוע|למה|כיצד|איך|כמה|איזה|"
    r"איזו|באיזו|האם|מעסיק|מגייס"
)


def _split_independent_requests(value: str) -> tuple[str, ...]:
    """Split only at complete conjunction/interrogative lexical tokens."""
    separator = re.compile(
        rf"(?<![{_LEXICAL_CHAR_CLASS}])(?:,\s*)?(?:and|or|ו|או)\s+"
        rf"(?=(?:{_QUESTION_TOKEN})(?![{_LEXICAL_CHAR_CLASS}]))|"
        rf"(?<![{_LEXICAL_CHAR_CLASS}])(?=ו(?:{_QUESTION_TOKEN})"
        rf"(?![{_LEXICAL_CHAR_CLASS}]))",
        re.IGNORECASE,
    )
    pieces: list[str] = []
    cursor = 0
    for match in separator.finditer(value):
        piece = value[cursor:match.start()].strip(" ,;:.?!")
        if piece:
            pieces.append(piece)
        cursor = match.end()
        if cursor == match.start() and value[cursor:cursor + 1] == "ו":
            cursor += 1
    tail = value[cursor:].strip(" ,;:.?!")
    if tail:
        pieces.append(tail)
    return tuple(pieces) or (value,)


def _negative_inclusion_request(value: str) -> bool:
    """Recognize negation independently of count nouns in the object."""
    return bool(
        _NEGATIVE_INCLUDE_RE.search(value)
        or (_NEGATION_RE.search(value) and _POSITIVE_INCLUDE_RE.search(value))
    )


def _filter_condition(value: str, metrics: Sequence[str]) -> Optional[FilterCondition]:
    match = _FILTER_CONDITION_RE.search(value)
    if match is None:
        return None
    raw_operator = normalize_semantic_text(match.group("operator")).casefold()
    if re.search(r"(?:at\s+least|לפחות)", raw_operator, re.I):
        operator = "greater_than_or_equal"
    elif re.search(r"(?:at\s+most|לכל\s+היותר)", raw_operator, re.I):
        operator = "less_than_or_equal"
    elif re.search(r"(?:less|fewer|below|under|פחות|מתחת)", raw_operator, re.I):
        operator = "less_than"
    elif re.search(r"(?:before|לפני)", raw_operator, re.I):
        operator = "before"
    elif re.search(r"(?:after|אחרי)", raw_operator, re.I):
        operator = "after"
    else:
        operator = "greater_than"
    metric = next((item for item in metrics if item not in {"count", "percentage"}), "")
    return FilterCondition(
        metric=metric,
        operator=operator,
        threshold=str(match.group("threshold") or "").replace(",", ""),
        unit=str(match.group("unit") or ""),
    )


def _scope_constraint(
    value: str,
    metrics: Sequence[str],
) -> Optional[ScopeConstraint]:
    condition = _filter_condition(value, metrics)
    if condition is None:
        return None
    return ScopeConstraint(
        metric=condition.metric,
        operator=condition.operator,
        threshold=condition.threshold,
        unit=condition.unit,
    )


def _output_filter_request(value: str) -> bool:
    """Return true only when qualifying rows/items are the requested output."""
    return bool(
        _FILTER_CONDITION_RE.search(value)
        and (
            re.search(
                r"\b(?:filter|only\s+those|where|meeting)\b|"
                r"(?:סנן|רק\s+(?:אלה|אלו)|שעבר)",
                value,
                re.IGNORECASE,
            )
            or _LIST_RE.search(value)
            or re.search(
                r"\b(?:which|what)\s+(?:years?|periods?|rows?|items?|sites?|facilities)\b|"
                r"\bwhen\b|(?:באילו|איזו|אילו)\s+(?:שנים|תקופות|שורות|אתרים|מתקנים)|מתי",
                value,
                re.IGNORECASE,
            )
        )
    )


def _scope_filter_modifier(value: str) -> bool:
    """Identify threshold clauses that constrain the requested population."""
    if _output_filter_request(value) or not _FILTER_CONDITION_RE.search(value):
        return False
    return bool(re.search(
        r"\b(?:for|among|within)\b.{1,120}\b(?:exceed(?:s|ed|ing)?|"
        r"pass(?:es|ed|ing)?|greater\s+than|more\s+than|above|over|"
        r"less\s+than|fewer\s+than|below|under|at\s+least|at\s+most)\b|"
        r"(?:עבור|בקרב|מבין|מתוך).{1,120}(?:מעל|יותר\s+מ|פחות\s+מ|"
        r"מתחת|לפחות|לכל\s+היותר)",
        value,
        re.IGNORECASE,
    ) or re.search(
        # A relative/participial clause attached to an entity narrows the
        # population of the separately requested scalar.  It does not turn
        # that scalar request into a row-producing FILTER operation.
        r"\bfor\s+(?:the\s+)?[A-Za-z][A-Za-z'’-]*(?:\s+[A-Za-z][A-Za-z'’-]*){0,5}\s+"
        r"(?:that\s+)?(?:employ|hold|operate|manage|serve|contain|use)(?:s|ing)?\b"
        r".{0,80}\b(?:above|over|more\s+than|below|under|less\s+than|at\s+least|at\s+most)\b|"
        r"(?<![\u05d0-\u05ea])ל[\u05d0-\u05ea\"'׳/-]+(?:\s+[\u05d0-\u05ea\"'׳/-]+){0,5}\s+"
        r"(?:ה?מעסיק(?:ה|ים|ות)?|ה?מחזיק(?:ה|ים|ות)?|ה?מפעיל(?:ה|ים|ות)?|"
        r"ה?מנהל(?:ת|ים|ות)?|ה?משרת(?:ת|ים|ות)?|ש(?:מעסיק|מחזיק|מפעיל|מנהל|משרת))"
        r".{0,80}(?:מעל|יותר\s+מ|מתחת|פחות\s+מ|לפחות|לכל\s+היותר)",
        value,
        re.IGNORECASE,
    ))


def _embedded_event(
    value: str,
    relation: str,
    metrics: Sequence[str],
    periods: Sequence[str],
) -> Optional[EmbeddedEvent]:
    if relation not in {"cause", "explanation"} or not _CHANGE_RE.search(value):
        return None
    direction = ""
    if re.search(r"(?:\b(?:increase|increased|rose|grew)\b|(?:עלייה|עלה|עלתה|גדל))", value, re.I):
        direction = "increase"
    elif re.search(r"(?:\b(?:decrease|decreased|fell|declined)\b|(?:ירידה|ירד|ירדה|פחת))", value, re.I):
        direction = "decrease"
    metric = next((item for item in metrics if item not in {"count", "percentage"}), "")
    return EmbeddedEvent(
        relation="changed_between",
        metric=metric,
        period_a=periods[0] if len(periods) >= 1 else "",
        period_b=periods[1] if len(periods) >= 2 else "",
        direction=direction,
    )


def _split_clauses(value: str) -> tuple[str, ...]:
    text = normalize_semantic_text(value)
    primary = re.split(r"\s*[;?]\s*|\n+", text)
    output: list[str] = []
    for part in primary:
        part = part.strip(" ,;:.?!")
        if not part:
            continue
        # Split only when a conjunction clearly begins another requested
        # claim. Ordinary noun-phrase conjunctions stay in one clause.
        # Keep dependent output attributes in the same clause.  This lets the
        # requirement builder link e.g. count+percentage or winner+attributes
        # while still splitting genuinely independent questions.
        dependent_output = bool(re.search(
            r"(?:,\s*)?(?:and\s+what|ו?מה)\s+(?:was|were|is|are|היה|הייתה|היו)?\s*"
            r"(?:(?:the|its|their)\s+)?(?:count|number|percentage|percent|share|rate|value|unit|"
            r"מספר|כמות|שיעור|אחוז|חלק|ערך|יחידה)",
            part,
            re.IGNORECASE,
        )) or bool(re.search(
            r"(?:,\s*)?ו?מה.{0,32}(?:מספר|כמות|שיעור|אחוז|חלק|ערך|יחידה)",
            part,
            re.IGNORECASE,
        )) or bool(re.search(
            r"(?:,\s*)?(?:and\s+what|ו?מה)\s+(?:was|were|is|are|היה|הייתה|היו)?"
            r".{0,64}(?:in\s+each\s+(?:year|period|row)|בכל\s+(?:שנה|תקופה|שורה))",
            part,
            re.IGNORECASE,
        ))
        pieces = [part] if dependent_output else _split_independent_requests(part)
        output.extend(piece.strip(" ,;:.?!") for piece in pieces if piece.strip(" ,;:.?!"))
    return tuple(output[:16]) or (text,)


def _relation(value: str, route_hint: str = "") -> str:
    from financial_semantics import financial_query
    financial = financial_query(value)
    if financial is not None and financial.operation == "payments":
        return "lookup"
    # Cause precedes exclusion: a why-question containing "not included" asks
    # for the reason, never for the excluded entity.
    if _OUTER_EXPLANATION_RE.search(value):
        return "explanation"
    if _CAUSE_RE.search(value) and re.search(r"(?:^\s*(?:why|מדוע|למה)|מה\s+גרם|סיבה)", value, re.I):
        return "cause"
    if _PROCEDURE_RE.search(value):
        return "procedure"
    if _DEFINITION_RE.search(value) and re.search(
        r"(?:\b(?:define|defined|definition|means?)\b|(?:מגדיר|מוגדר|הגדר|פירוש|משמעות))",
        value,
        re.I,
    ):
        return "definition"
    if _PURPOSE_RE.search(value):
        return "purpose"
    if _ARGMAX_RE.search(value):
        return "argmax"
    if _ARGMIN_RE.search(value):
        return "argmin"
    if _negative_inclusion_request(value):
        return "exclusion"
    if re.search(r"(?:\b(?:except|exception|other\s+than)\b|(?:למעט|פרט\s+ל|חריג))", value, re.I):
        return "exception"
    output_filter = _output_filter_request(value)
    scope_filter = _scope_filter_modifier(value) or bool(re.search(
        r"(?:שיעור|אחוז).{0,80}(?:עובדים|עובדות).{0,40}בעלי.{0,40}"
        r"(?:מעל|מתחת|לפחות|לכל\s+היותר)",
        value,
        re.IGNORECASE,
    ))
    if output_filter:
        return "filter_rows"
    if (
        not scope_filter
        and _CHANGE_RE.search(value)
        and (_COMPARE_RE.search(value) or len(tuple(match for match in _PERIOD_RE.finditer(value)
            if not _numeric_identifier_role(value, match.start(), match.end()))) >= 2)
    ):
        return "change_over_time"
    if _COMPARE_RE.search(value) and not scope_filter:
        return "compare"
    if _LIST_RE.search(value):
        return "list"
    if _DEFINITION_RE.search(value) and re.search(r"(?:define|defined|definition|means?|מגדיר|מוגדר|הגדר|פירוש|תפקיד|אחריות)", value, re.I):
        return "definition"
    if (
        _FILTER_RE.search(value)
        and not _request_has_count_cue(value)
        and not scope_filter
    ):
        return "filter_rows"
    if _request_has_count_cue(value):
        return "count"
    if _POSITIVE_INCLUDE_RE.search(value):
        return "inclusion"
    hint = route_hint.casefold()
    if "cause" in hint or "explanation" in hint:
        return "cause" if _CAUSE_RE.search(value) else "explanation"
    if "summary" in hint:
        return "summary"
    if "procedure" in hint:
        return "procedure"
    return "lookup"


def _predicate(
    value: str,
    relation: str,
    metrics: Sequence[str] = (),
) -> str:
    role_predicate, _target = document_role_query(value)
    if role_predicate:
        return role_predicate
    if _RAISES_CAPITAL_RE.search(value):
        return "raises_capital"
    if _RECRUITS_EMPLOYEES_RE.search(value):
        return "recruits_employees"
    if _EMPLOYS_RE.search(value):
        return "employs_workers"
    if (
        relation == "lookup"
        and not re.match(
            r"\s*(?:which|who|what\s+(?:category|entity|name)|"
            r"איזה|איזו|אילו|מי)\b",
            value,
            re.IGNORECASE,
        )
        and (
        set(metrics) & {
            "demand", "revenue", "profit", "cost", "capacity", "salary",
            "duration", "employee_count", "employee_recruitment",
            "capital_raised",
        }
        or re.search(
            r"\b(?:estimate|amount|value|total|overall|lower\s+bound|upper\s+bound)\b|"
            r"(?:אומדן|הערכה|ערך|סך\s*הכול|חסם|גבול)",
            value,
            re.IGNORECASE,
        )
        )
    ):
        return "has_count"
    return {
        "purpose": "purpose_of",
        "definition": "defined_as",
        "cause": "caused_by",
        "exclusion": "excluded_from",
        "exception": "exception_to",
        "inclusion": "included_in",
        "change_over_time": "changed_between",
        "compare": "compared_with",
        "count": "has_count",
        "list": "has_members",
        "filter": "satisfies_filter",
        "filter_rows": "satisfies_filter",
        "argmax": "maximum_of",
        "argmin": "minimum_of",
        "procedure": "method_of",
        "summary": "summarizes",
        "explanation": "explained_as",
    }.get(relation, "stated_as")


def _metrics(value: str) -> tuple[str, ...]:
    from financial_semantics import financial_request
    if financial_request(value) is not None:
        return ("cost",)
    from document_query import mask_role_bound_identifiers
    metric_text = mask_role_bound_identifiers(value)
    output = [name for name, pattern in _METRIC_PATTERNS if pattern.search(metric_text)]
    for name, pattern in _FACET_PATTERNS:
        if pattern.search(value):
            output.append(name)
    return _unique(output)


def explicit_metric_concepts(value: str) -> tuple[str, ...]:
    """Concrete measured concepts; retrieval synonyms do not supply these."""
    from document_query import mask_role_bound_identifiers
    metric_text = mask_role_bound_identifiers(value)
    return _unique(name for name, pattern in _METRIC_PATTERNS
                   if name not in {"percentage", "count", "total"} and pattern.search(metric_text))


_SCOPE_MARKER_RE = re.compile(
    r"\b(?:among|within|across)\b|\bfor\s+(?:the\s+)?(?:companies|"
    r"organizations|workers|employees|customers|sites|projects|facilities|"
    r"teams|groups|participants|respondents|households|units|categories|"
    r"category|types?|segments?)\b|"
    r"\bin\s+(?:companies|organizations|workers|employees|customers|sites|"
    r"projects|facilities|teams|groups|participants|respondents|households|"
    r"units|categories)\b|"
    r"(?:בקרב|מבין|מתוך|בחברות|בארגונים|בעובדים|בלקוחות|באתרים|במתקנים|"
    r"בקבוצות|בקטגורי(?:ה|ות)|עבור\s+(?:חברות|ארגונים|עובדים|לקוחות|אתרים|"
    r"מתקנים|קבוצות|קטגוריות))",
    re.IGNORECASE,
)


def _explicit_scope_marker(value: str) -> bool:
    """Whether the question contains an explicit narrowing population role."""
    if experience_band_facets(value):
        # This exact workforce duration is owned by the experience facet.
        # Any other scope marker (industry, company, region, etc.) survives
        # and still requires independently resolved population semantics.
        value = re.sub(
            r"\bfor\s+(?:the\s+)?(?:workers|employees)\s+with\s+"
            r"(?:(?:experience|tenure)\s+(?:of\s+)?)?" + _EXPERIENCE_BOUND_RE.pattern,
            lambda match: "" if _experience_bound_owned(value, match) else match.group(0),
            value, flags=re.I,
        )
        value = re.sub(
            r"עבור\s+(?:ה)?עובדים\s+עם\s+"
            r"(?:(?:ניסיון)\s+(?:של\s+)?)?" + _EXPERIENCE_BOUND_RE.pattern,
            lambda match: "" if _experience_bound_owned(value, match) else match.group(0),
            value, flags=re.I,
        )
    return bool(_SCOPE_MARKER_RE.search(value))


def _extract_populations(value: str) -> tuple[str, ...]:
    """Return compositional request populations, never evidence-derived roles."""
    patterns = (
        r"\b(?P<scope>(?:among|within|across)\s+(?:the\s+)?"
        r"(?:[A-Za-z0-9'’-]+\s+){0,5}(?:companies|organizations|workers|"
        r"employees|customers|sites|projects|facilities|teams|groups|participants|"
        r"respondents|households|units|categories)(?:\s+(?:in|of|with|from)\s+"
        r"(?!\d{4}\b)(?:[A-Za-z0-9'’-]+\s*){1,6})?)",
        r"\b(?P<scope>for\s+(?:the\s+)?(?:[A-Za-z0-9'’-]+\s+){0,4}"
        r"(?:companies|organizations|workers|employees|customers|sites|projects|"
        r"facilities|teams|groups|participants|respondents|households|units|categories)"
        r"(?:\s+(?:in|of|with|from)\s+(?!\d{4}\b)(?:[A-Za-z0-9'’-]+\s*){1,6})?)",
        r"\b(?P<scope>in\s+(?:companies|organizations|workers|employees|"
        r"customers|sites|projects|facilities|teams|groups|participants|respondents|"
        r"households|units|categories)(?:\s+(?:in|of|with|from)\s+"
        r"(?!\d{4}\b)(?:[A-Za-z0-9'’-]+\s*){1,6})?)",
        r"\b(?P<scope>for\s+(?:the\s+)?(?:category|type|segment)\s+"
        r"(?:[A-Za-z0-9'’-]+\s*){1,4})",
        r"(?P<scope>(?:בקרב|מבין|מתוך)\s+ה?(?:חברות|ארגונים|עובדים|לקוחות|"
        r"אתרים|פרויקטים|מתקנים|קבוצות|קטגוריות|משתתפים|משיבים)(?:\s+(?:ב|מ|ל)?(?:תחום|אזור|ענף|מגזר)\s+"
        r"[\u05d0-\u05ea\"'׳/-]+){0,2}(?:\s+[\u05d0-\u05ea\"'׳/-]+){0,4})",
        r"ב(?P<scope>(?:חברות|ארגונים|עובדים|לקוחות|אתרים|פרויקטים|מתקנים|קבוצות|קטגוריות|משתתפים|משיבים)\s+"
        r"(?:ב|מ)?(?:תחום|אזור|ענף|מגזר)\s+"
        r"[\u05d0-\u05ea\"'׳/-]+(?:\s+[\u05d0-\u05ea\"'׳/-]+){0,4})",
        r"עבור\s+(?P<scope>(?:ארגונים|חברות|עובדים|לקוחות|אתרים)\s+"
        r"ב(?:אזור|תחום|ענף|מגזר)\s+[\u05d0-\u05ea\"'׳/-]+(?:\s+[\u05d0-\u05ea\"'׳/-]+){0,4})",
        r"(?P<scope>העובדים\s+בעלי\s+[\u05d0-\u05ea\"'׳/-]+(?:\s+של\s+"
        r"(?:מעל|מתחת|לפחות|עד)\s+[\u05d0-\u05ea0-9\"'׳/-]+"
        r"(?:\s+(?:שנים?|שנה))?){0,2})",
    )
    populations: list[str] = []
    locative = source_population_scope(_experience_comparison_subject(value))
    if locative:
        populations.append(locative)
    for pattern in patterns:
        for match in re.finditer(pattern, value, re.IGNORECASE):
            scope = normalize_semantic_text(match.group("scope")).strip(" ,;:.?!")
            # A scope qualifier may legitimately contain a location ("in the
            # urban region"), but it must end before the clause predicate.
            # Keep this deliberately grammatical rather than evidence-shaped:
            # request text is the only authority for the population role.
            scope = re.sub(
                r"\s+\b(?:reported|report|completed|complete|inspected|inspect|"
                r"experienced|experience|received|receive|were|was|are|is|did|"
                r"do|does|have|has|had)\b.*$",
                "",
                scope,
                flags=re.IGNORECASE,
            ).strip(" ,;:.?!")
            if scope:
                populations.append(scope)
    return _unique(populations)


def experience_scope_populations(value: str) -> tuple[str, ...]:
    """Preserve independent workforce scopes after the duration is factored."""
    scopes = _extract_populations(_experience_comparison_subject(value))
    output: list[str] = []
    for scope in scopes:
        generic_worker_suffix = re.search(
            r"(?:\bfor\s+(?:the\s+)?(?:workers|employees)|"
            r"לעובדים|עבור\s+(?:ה)?עובדים)\s*$",
            scope,
            re.I,
        )
        if generic_worker_suffix is None:
            output.append(scope)
            continue
        independent_scope = scope[:generic_worker_suffix.start()].strip()
        if independent_scope and _SCOPE_MARKER_RE.search(independent_scope):
            output.append(independent_scope)
    return _unique(output)


def _remove_scope_only_metrics(
    metrics: Sequence[str], populations: Sequence[str]
) -> tuple[str, ...]:
    """Avoid treating a noun that narrows population as an answer metric."""
    population_text = " ".join(populations)
    if populations and re.search(r"\b(?:industry|sector)\b|(?:תחום|ענף|מגזר|תעשיי|סקטור)", population_text, re.I):
        return tuple(metric for metric in metrics if metric != "industry_sector")
    return tuple(metrics)


def _scalar_value_type(clause: CanonicalClause) -> str:
    """Classify requested answer type independently from the route enum."""
    relation = clause.semantic_relation
    from document_query import mask_role_bound_identifiers
    text = mask_role_bound_identifiers(clause.text)
    if relation in {"purpose", "definition", "cause", "explanation", "procedure", "summary", "exclusion", "exception"}:
        return "text_proposition"
    if relation == "list":
        return "list"
    slots = set(clause.requested_output_slots)
    if "count" in slots:
        return "count"
    if "percentage" in slots:
        return "percentage"
    if relation in {"compare", "change_over_time"}:
        return "comparison_result"
    if re.search(
        r"^\s*(?:what|which)\s+(?:year|month)\b|^\s*when\b|"
        r"^\s*(?:באיזו\s+(?:שנה|חודש)|באיזה\s+חודש|מתי)\b|"
        r"\b(?:date|period|month)\b|(?:תאריך|תקופה|חודש)",
        text,
        re.IGNORECASE,
    ):
        return "period"
    if re.match(r"\s*(?:which|who|what\s+(?:category|entity|name)|איזה|איזו|אילו|מי)\b", text, re.I):
        return "entity"
    if re.match(
        r"\s*(?:does|do|did|is|are|was|were|has|have|had)\b|^\s*האם\b",
        text,
        re.IGNORECASE,
    ):
        return "boolean"
    if re.search(
        r"(?:[$€£₪]|\b(?:currency|dollars?|euros?|pounds?|shekels?)\b|"
        r"(?:מטבע|שקלים?|דולרים?|אירו))",
        text,
        re.IGNORECASE,
    ):
        return "currency"
    numeric_cues = re.compile(
        r"\b(?:estimate|amount|value|total|overall|lower\s+bound|upper\s+bound|"
        r"confidence(?:\s+level)?|threshold|range|demand|revenue|profit|cost|"
        r"capacity|salary|duration)\b|"
        r"(?:אומדן|הערכה|ערך|סך\s*הכול|חסם|גבול|ביקוש|הכנס|רווח|עלות|קיבולת|שכר|משך)",
        re.IGNORECASE,
    )
    if numeric_cues.search(text) or any(
        metric in {"demand", "revenue", "profit", "cost", "capacity", "salary", "duration", "employee_count", "employee_recruitment", "capital_raised"}
        for metric in clause.objects_or_metrics
    ):
        return "numeric"
    return "text_proposition"


def _request_clause_text(value: str) -> str:
    # Source identity remains in document_scope and the original request span.
    # Its filename framing is not a factual subject, metric or answer count.
    from document_reading import named_document_request
    scoped = named_document_request(value)
    return scoped[1] if scoped is not None else value


def _subject(value: str, metrics: Sequence[str], relation: str = "") -> str:
    from subject_intent import information_subject, project_location_anchor
    information = information_subject(value)
    if information and relation == "lookup" and project_location_anchor(information):
        return normalize_semantic_text(information)[:MAX_CONTRACT_TEXT]
    role_predicate, role_target = document_role_query(value)
    if role_predicate:
        return role_target
    if relation in {"compare", "change_over_time"} and (
        experience_band_facets(value)
        or re.search(r"\bby\s+experience\s+levels?\b|לפי\s+(?:רמות\s+)?ניסיון", value, re.I)
    ):
        value = _experience_comparison_subject(value)
    from request_understanding import requested_enumerations
    for enumeration in reversed(requested_enumerations(value)):
        # Blank exactly the request-owned enumeration numeral. Source IDs,
        # dates and unrelated matching words retain their original spans.
        start,stop = enumeration.numeral_span
        value = value[:start]+' '*(stop-start)+value[stop:]
    text = re.sub(
        r"^\s*(?:what|which|who|why|how|does|do|did|is|are|was|were|"
        r"מה|מהו|מהי|מהם|מהן|מי|מדוע|למה|כיצד|איך|כמה|איזה|איזו|אילו|ציין|פרט)\b",
        " ", value, flags=re.IGNORECASE,
    )
    # Interrogatives commonly carry an auxiliary immediately after the
    # question word ("what was the purpose", "which is the category").  It
    # is grammar, not part of the semantic subject.
    text = re.sub(
        r"^\s*(?:is|are|was|were|does|do|did|has|have|had)\b\s*",
        "", text, flags=re.IGNORECASE,
    )
    from document_query import role_bound_identifier_spans
    identifier_spans = tuple(item.identifier_span for item in role_bound_identifier_spans(text))
    def preserve_identifier(match):
        return match.group(0) if any(match.start() < b and a < match.end()
            for a, b in identifier_spans) else ' '
    # Remove answer-size/value and period grammar while keeping exact entity
    # identity, including year-shaped and dotted/alphanumeric designations.
    text = _NUMBER_RE.sub(preserve_identifier, text)
    identifier_spans = tuple(item.identifier_span for item in role_bound_identifier_spans(text))
    text = _PERIOD_RE.sub(preserve_identifier, text)

    text = normalize_semantic_text(text).strip(" ,;:.?!")
    if text:
        return text[:MAX_CONTRACT_TEXT]
    return " ".join(metrics)[:MAX_CONTRACT_TEXT]


def _cardinality(value: str, default: int) -> int:
    # The plan/compiler share the same request-only interpretation. Dates,
    # amounts and document identifiers cannot become a count just because a
    # request is an enumeration. ``default`` remains the compatibility budget.
    from request_understanding import explicit_requested_items
    explicit = explicit_requested_items(value)
    return explicit if explicit is not None else default


def _answer_shape(value: str, relation: str, cardinality: int) -> str:
    from financial_semantics import financial_request, financial_query
    financial = financial_query(value)
    if relation == "lookup" and (financial_request(value) is not None or (financial is not None and financial.operation == "payments")):
        return "scalar"
    count = _request_has_count_cue(value, include_slot=True)
    percentage = bool(_PERCENT_RE.search(value))
    if relation in {"argmax", "argmin"}:
        attrs = ["winner"]
        if percentage:
            attrs.append("percentage")
        if count:
            attrs.append("count")
        return "extremum_with_" + "_and_".join(attrs)
    if relation == "filter_rows":
        return "repeated_rows_period_and_metric_value"
    if count and percentage:
        return "count_and_percentage"
    if relation in {"compare", "change_over_time"}:
        return "comparison_endpoints"
    if relation == "list":
        return "list_with_attributes" if re.search(r"(?:for\s+each|with\s+their|לכל|עבור\s+כל|אחריות|מאפיינ)", value, re.I) else "list"
    if relation in {"cause", "purpose", "definition", "explanation"}:
        return "explanation"
    if relation in {"exclusion", "exception"}:
        return "entity_or_negative_proposition"
    if count:
        return "scalar_count"
    return "scalar" if cardinality == 1 else "list"


def _endpoint_type(value: str) -> str:
    if _PERIOD_RE.fullmatch(normalize_semantic_text(value)):
        return "period"
    if re.search(
        r"\b(?:population|sample|companies|firms|operators|workers|employees|customers|sites|"
        r"projects|all)\b|(?:אוכלוסי|מדגם|חברות|מפעילים|עובדים|לקוחות|אתרים|"
        r"פרויקטים|כלל)",
        value,
        re.IGNORECASE,
    ):
        return "population"
    if re.search(
        r"\b(?:category|band|level|type|group)\b|(?:קטגור|קבוצ|רמה|סוג|טווח)",
        value,
        re.IGNORECASE,
    ):
        return "category"
    return "entity"


def _comparison_endpoint_is_outcome_phrase(value: str) -> bool:
    """Reject an endpoint hint that describes the requested result itself.

    Population endpoints identify the entities or populations being compared.
    They must not absorb a proposition such as "firms recruit more workers",
    which would turn an outcome into an impossible population constraint.
    """
    normalized = normalize_semantic_text(value)
    return bool(re.search(
        r"\b(?:recruit(?:s|ed|ing)?|hire[sd]?|employ(?:s|ed|ing)?|"
        r"raise[sd]?|raising|fundrais(?:e|ing)|increase[sd]?|decrease[sd]?|"
        r"grew|declined|more|less|fewer|higher|lower)\b|"
        r"(?:מגייס|גייס(?:ה|ו)?|מעסיק|העסיק|גייס(?:ה)?\s+הון|"
        r"עלה|עלתה|ירד|ירדה|גדל|גדלה|פחת|יותר|פחות)",
        normalized,
        re.IGNORECASE,
    ))


def _clean_endpoint(value: str) -> str:
    cleaned = normalize_semantic_text(value).strip(" ,;:.?!")
    cleaned = re.sub(
        r"^(?:the\s+report\s+)?(?:compares?|comparison\s+of|"
        r"הדוח\s+משווה|השוו|השווה)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned[:MAX_CONTRACT_TEXT]


def _comparison_endpoints(
    value: str,
    relation: str,
    periods: Sequence[str],
) -> tuple[ComparisonEndpoint, ...]:
    if relation not in {"compare", "change_over_time"}:
        return ()
    if len(periods) >= 2:
        return tuple(
            ComparisonEndpoint(
                id=f"endpoint-{index}", endpoint_type="period", value=period
            )
            for index, period in enumerate(periods[:2], start=1)
        )
    patterns = (
        re.compile(
            r"\b(?:of|among)\s+(?:the\s+)?"
            r"(?P<a>[A-Za-z][A-Za-z0-9'’-]*(?:\s+[A-Za-z][A-Za-z0-9'’-]*){0,2})"
            r"\s+and\s+"
            r"(?P<b>[A-Za-z][A-Za-z0-9'’-]*(?:\s+[A-Za-z][A-Za-z0-9'’-]*){0,2})"
            r"\s+(?P<noun>operators?|firms?|companies|workers?|employees?|"
            r"sites?|stations?|terminals?|gateways?|projects?|systems?|facilities)\b",
            re.IGNORECASE,
        ),
        re.compile(
            r"\bbetween\s+(?P<a>.+?)\s+and\s+(?P<b>.+?)"
            r"(?=\s+(?:in\s+terms\s+of|with\s+respect\s+to|by)\b|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?P<a>.+?)\s+(?:versus|vs\.?)\s+(?P<b>.+?)"
            r"(?=\s+(?:in\s+terms\s+of|with\s+respect\s+to|by)\b|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"בין\s+(?P<a>.+?)\s+לבין\s+(?P<b>.+?)"
            r"(?=\s+(?:מבחינת|לפי|בהיבט)\b|$)",
            re.IGNORECASE,
        ),
        re.compile(
            r"(?P<a>.+?)\s+לעומת\s+(?P<b>.+?)"
            r"(?=\s+(?:מבחינת|לפי|בהיבט)\b|$)",
            re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        match = pattern.search(value)
        if match is None:
            continue
        shared_noun = normalize_semantic_text(
            match.groupdict().get("noun") or ""
        )
        values = (
            _clean_endpoint(" ".join((match.group("a"), shared_noun))),
            _clean_endpoint(" ".join((match.group("b"), shared_noun))),
        )
        invalid_endpoint = re.compile(
            r"^(?:does|do|did|is|are|was|were|what|which|who|why|how|"
            r"does\s+ai|ai|האם|מה|מי|למה|מדוע|איך)$",
            re.IGNORECASE,
        )
        if (
            all(values)
            and values[0].casefold() != values[1].casefold()
            and not any(invalid_endpoint.fullmatch(item) for item in values)
            and not any(
                _comparison_endpoint_is_outcome_phrase(item)
                for item in values
            )
        ):
            return tuple(
                ComparisonEndpoint(
                    id=f"endpoint-{index}",
                    endpoint_type=_endpoint_type(endpoint),
                    value=endpoint,
                )
                for index, endpoint in enumerate(values, start=1)
            )
    return ()


def _requested_output_slots(value: str, relation: str) -> tuple[str, ...]:
    from financial_semantics import financial_request
    if relation == "lookup" and financial_request(value) is not None:
        return ("scalar",)
    count = _request_has_count_cue(value, include_slot=True)
    percentage = bool(_PERCENT_RE.search(value))
    if relation in {"argmax", "argmin"}:
        slots = ["winner"]
        if percentage:
            slots.append("percentage")
        if count:
            slots.append("count")
        return tuple(slots)
    if relation == "filter_rows":
        return ("row_period", "row_metric_value")
    if relation == "list":
        return ("items", "item_attributes") if re.search(
            r"(?:for\s+each|with\s+their|לכל|עבור\s+כל|אחריות|מאפיינ)",
            value,
            re.IGNORECASE,
        ) else ("items",)
    if relation in {"compare", "change_over_time"}:
        slots = ["endpoint_values", "direction"]
        if re.search(
            r"\b(?:statistical(?:ly)?\s+significant|significant(?:ly)?|significance)\b|"
            r"(?:מובהק|מובהקות)",
            value,
            re.IGNORECASE,
        ):
            slots.append("significance")
        return tuple(slots)
    if relation in {"exclusion", "exception"}:
        return ("entity",)
    if relation == "inclusion":
        return ("count",) if count else ("positive_inclusion",)
    if relation == "purpose" and re.search(r"(?:\bpolicy\b|מדיניות)", value, re.I):
        return ("purpose", "policy_use")
    if relation in {"purpose", "definition", "cause", "explanation", "procedure"}:
        return ("text_proposition",)
    if count and percentage:
        return ("count", "percentage")
    if percentage:
        return ("percentage",)
    if count:
        return ("count",)
    return ("scalar",)


def _semantic_targets(
    value: str,
    relation: str,
    endpoints: Sequence[ComparisonEndpoint],
    populations: Sequence[str],
) -> dict[str, str]:
    """Extract target roles without treating contextual nouns as the target."""
    text = normalize_semantic_text(value)
    counted_entity = ""
    count_match = re.search(
        r"(?:\bhow\s+many\b|^\s*(?:כמה|בכמה)\b)\s+"
        r"(?P<entity>.+?)\s+(?:were|was|are|is|did|do|does|have|has|"
        r"reported|responded|experienced|received|"
        r"היו|היה|הייתה|יש|נעשה|שימשו|נכלל|נכללו)\b",
        text,
        re.IGNORECASE,
    )
    if count_match:
        counted_entity = normalize_semantic_text(count_match.group("entity")).strip(" ,;:.?!")
    elif relation == "count":
        count_match = re.search(
            r"\b(?:count|number|quantity)\s+of\s+(?P<entity>.+?)"
            r"(?=\s+(?:in|for|among|within|that|which|who|were|was|are|is)\b|[?.,]|$)",
            text,
            re.IGNORECASE,
        )
        if count_match:
            counted_entity = normalize_semantic_text(count_match.group("entity")).strip(" ,;:.?!")

    target_entity_type = ""
    if relation in {"compare", "change_over_time"} and experience_band_facets(value):
        named_actor = re.search(
            r"\b(?:at|in|for)\s+(?:the\s+)?(?P<actor>"
            r"(?:[Cc]ompany|[Uu]nit|[Dd]epartment|[Dd]ivision|[Tt]eam|[Pp]roject)\s+"
            r"[A-Z][A-Za-z0-9'’-]*(?:\s+[A-Z][A-Za-z0-9'’-]*){0,2})", value,
        )
        if named_actor:
            target_entity_type = named_actor.group("actor")
    entity_match = re.match(
        r"^\s*(?:which|what|who|איזה|איזו|אילו|מי|מהו|מהי)\s+"
        r"(?P<entity>[A-Za-z\u05d0-\u05ea][A-Za-z\u05d0-\u05ea'׳-]*)\b",
        text,
        re.IGNORECASE,
    )
    if entity_match:
        target_entity_type = entity_match.group("entity")
        if target_entity_type.casefold() in {
            "percentage", "percent", "rate", "share", "count", "number",
            "amount", "value", "אחוז", "שיעור", "מספר", "כמות", "ערך",
            "is", "are", "was", "were", "does", "do", "did", "has",
            "have", "had",
        }:
            target_entity_type = ""
    if relation == "purpose":
        purpose_target = re.search(
            r"\bpurpose\s+of\s+(?:the\s+)?(?P<target>[^?.,;]+)|"
            r"(?:מטר(?:ה|ת))\s+(?P<target_he>[^?.,;]+)",
            text,
            re.IGNORECASE,
        )
        if purpose_target:
            target_entity_type = normalize_semantic_text(
                purpose_target.group("target") or purpose_target.group("target_he")
            ).strip(" ,;:.?!")
    elif relation == "procedure":
        procedure_target = re.search(
            r"^\s*how\s+(?:was|were|is|are)\s+(?P<target_question>[A-Za-z][A-Za-z0-9'’-]*"
            r"(?:\s+[A-Za-z][A-Za-z0-9'’-]*){0,4})\s+(?:identified|determined|"
            r"calculated|measured|performed|selected|constructed|derived|implemented)\b|"
            r"\b(?:the\s+)?(?P<target>[A-Za-z][A-Za-z0-9'’-]*(?:\s+[A-Za-z][A-Za-z0-9'’-]*){0,4})\s+"
            r"(?:identified|determined|calculated|measured|performed|selected|constructed|derived|implemented)\b|"
            r"^\s*(?:כיצד|איך)\s+(?:זוה|זוהה|זוהתה|אותר|אותרה|נבחר|נבחרה|נקבע|נקבעה|"
            r"חושב|חושבה|נמדד|נמדדה|בוצע|בוצעה|יושם|יושמה|הוגדר|הוגדרה|נבנה|נבנתה)\s+"
            r"(?P<target_he_question>[\u05d0-\u05ea][\u05d0-\u05ea\"'׳/-]*(?:\s+[\u05d0-\u05ea\"'׳/-]+){0,4})|"
            r"(?P<target_he>[\u05d0-\u05ea][\u05d0-\u05ea\"'׳/-]*(?:\s+[\u05d0-\u05ea\"'׳/-]+){0,4})\s+"
            r"(?:זוה|אותר|נבחר|נקבע|חושב|נמדד|בוצע|יושם|הוגדר|נבנה)",
            text,
            re.IGNORECASE,
        )
        if procedure_target:
            target_entity_type = normalize_semantic_text(
                procedure_target.group("target_question")
                or procedure_target.group("target")
                or procedure_target.group("target_he_question")
                or procedure_target.group("target_he")
            ).strip(" ,;:.?!")
    if relation == "count" and counted_entity:
        target_entity_type = counted_entity

    scope_population = populations[0] if populations else ""
    if not scope_population or scope_population.casefold() in {
        "population", "sample", "cohort", "אוכלוסייה", "מדגם"
    }:
        population_match = re.search(
            r"\b(?P<scope>(?:[A-Za-z0-9'-]+\s+){0,3}(?:population|sample|cohort))\b|"
            r"(?P<scope_he>(?:אוכלוסיית?|מדגם)\s+[\u05d0-\u05ea\"'׳/-]+(?:\s+[\u05d0-\u05ea\"'׳/-]+){0,3})",
            text,
            re.IGNORECASE,
        )
        if population_match:
            scope_population = normalize_semantic_text(
                population_match.group("scope") or population_match.group("scope_he")
            )
            scope_population = re.sub(
                r"^(?:identify|identifies|identified|cover|covers|covered|represent|represents)\s+(?:the\s+)?",
                "",
                scope_population,
                flags=re.IGNORECASE,
            )

    requested_attribute = ""
    if relation in {"argmax", "argmin"}:
        attrs = [slot for slot in _requested_output_slots(text, relation) if slot != "winner"]
        requested_attribute = ",".join(attrs)
    elif relation in {"exclusion", "exception"}:
        requested_attribute = target_entity_type or "excluded_entity"
    role_predicate, _role_target = document_role_query(value)
    if role_predicate:
        target_entity_type = ""
        requested_attribute = RELATION_ARGUMENT_SCHEMAS[role_predicate][1]

    comparison_subject = endpoints[0].value if len(endpoints) >= 1 else ""
    comparison_reference = endpoints[1].value if len(endpoints) >= 2 else ""
    return {
        "target_entity_type": target_entity_type[:MAX_CONTRACT_TEXT],
        "counted_entity": counted_entity[:MAX_CONTRACT_TEXT],
        "requested_attribute": requested_attribute[:MAX_CONTRACT_TEXT],
        "comparison_subject": comparison_subject[:MAX_CONTRACT_TEXT],
        "comparison_reference": comparison_reference[:MAX_CONTRACT_TEXT],
        "scope_population": scope_population[:MAX_CONTRACT_TEXT],
    }


def _relation_target(
    value: str,
    relation: str,
    subject: str,
    target_entity_type: str,
) -> str:
    """Resolve the event/entity whose relation the answer must address."""
    role_predicate, role_target = document_role_query(value)
    if role_predicate:
        return role_target
    if relation not in {
        "cause", "explanation", "purpose", "definition", "procedure",
        "exclusion", "exception", "inclusion",
    }:
        return ""
    if relation in {"exclusion", "exception", "inclusion"}:
        scope_match = re.search(
            r"\b(?:excluded|omitted|not\s+included|included)\b.{0,24}"
            r"\b(?:from|in)\s+(?:the\s+)?(?P<scope>[^?.,;]+)|"
            r"(?:לא\s+נכלל(?:ה|ו)?|אינ(?:ו|ה|ם|ן)\s+כלול(?:ה|ים|ות)?|"
            r"הוחרג(?:ה|ו)?|נכלל(?:ה|ו)?)\s+"
            r"(?:מ|ב)(?P<scope_he>[^?.,;]+)",
            value,
            re.IGNORECASE,
        )
        if scope_match:
            return normalize_semantic_text(
                scope_match.group("scope") or scope_match.group("scope_he")
            ).strip(" ,;:.?!")[:MAX_CONTRACT_TEXT]
        return ""
    if relation == "definition":
        definition_target = re.search(
            r"^\s*what\s+(?:does|do)\s+(?:the\s+)?(?P<does_target>.+?)\s+means?\s*$|"
            r"^\s*(?:define|what\s+is\s+the\s+(?:definition|meaning)\s+of)\s+"
            r"(?:the\s+)?(?P<direct_target>.+?)\s*$|"
            r"^\s*(?:מה\s+)?(?:פירוש|משמעות)\s+(?P<he_target>.+?)\s*$|"
            r"^\s*הגדר(?:י|ו)?\s+(?P<he_direct_target>.+?)\s*$",
            normalize_semantic_text(value).strip(" ,;:.?!"),
            re.IGNORECASE,
        )
        if definition_target:
            return normalize_semantic_text(next(
                group for group in definition_target.groups() if group
            )).strip(" ,;:.?!")[:MAX_CONTRACT_TEXT]
    if relation in {"cause", "explanation"}:
        event_target = re.search(
            r"^\s*why\s+(?:did|does|do|was|were|is|are)\s+(?P<why_event>.+?)\s*$|"
            r"^\s*(?:does|do|did|can)\s+(?:the\s+)?(?:report|document|study)\s+"
            r"(?:explain|describe)\s+why\s+(?P<explained_event>.+?)\s*$|"
            r"^\s*(?:מדוע|למה)\s+(?P<he_why_event>.+?)\s*$|"
            r"^\s*האם\s+(?:הדוח|המסמך|המחקר)\s+(?:מסביר|מסבירה|הסביר|הסבירה)\s+"
            r"(?:מדוע|למה)\s+(?P<he_explained_event>.+?)\s*$",
            normalize_semantic_text(value).strip(" ,;:.?!"),
            re.IGNORECASE,
        )
        if event_target:
            return normalize_semantic_text(next(
                group for group in event_target.groups() if group
            )).strip(" ,;:.?!")[:MAX_CONTRACT_TEXT]
    if relation in {"purpose", "definition", "procedure"} and target_entity_type:
        return normalize_semantic_text(target_entity_type)[:MAX_CONTRACT_TEXT]
    target = normalize_semantic_text(subject)
    target = re.sub(
        r"^(?:the\s+)?(?:reason|cause|explanation|purpose|definition|meaning|"
        r"method|procedure)\s+(?:for|of)\s+|"
        r"^(?:סיבת?|הסיבה\s+ל|מטרת?|הגדרת?|משמעות|שיטת?|אופן)\s+",
        "",
        target,
        flags=re.IGNORECASE,
    )
    target = re.sub(
        r"^(?:did|does|do|was|were|is|are|האם|היה|הייתה|היו)\s+",
        "",
        target,
        flags=re.IGNORECASE,
    )
    return target.strip(" ,;:.?!")[:MAX_CONTRACT_TEXT]


def _normalize_metrics_for_relation(
    metrics: Sequence[str],
    text: str,
    relation: str,
) -> tuple[str, ...]:
    output = list(_unique(metrics))
    output = [
        item for item in output
        if item not in {"role", "education_level", "experience_band", "population_subgroup"}
    ]
    if relation in {"argmax", "argmin"}:
        output = [item for item in output if item not in {"count", "percentage"}]
    if relation in {"compare", "change_over_time"}:
        if "employee_count" in output and "count" in output:
            output.remove("count")
        total_is_metric = bool(re.search(
            r"\b(?:total|overall)\s+(?:count|number|demand|revenue|cost|capacity)\b|"
            r"(?:סך\s*הכול|סה.?כ)\s+(?:מספר|ביקוש|הכנסות|עלות|קיבולת)",
            text,
            re.IGNORECASE,
        ))
        if "total" in output and not total_is_metric:
            output.remove("total")
        if "percentage" in output and "demand" in output and _PERCENT_RE.search(text):
            output.remove("percentage" if experience_band_facets(text) or "experience_band" in metrics else "demand")
    return tuple(output)


def _normalize_semantic_draft(
    draft: SemanticContractDraft,
) -> tuple[tuple[CanonicalClause, ...], SemanticValidationRecord]:
    """Apply exactly one deterministic generic normalization pass."""
    normalized: list[CanonicalClause] = []
    inherited_periods: tuple[str, ...] = ()
    changed = False
    violations: list[str] = []
    for clause in draft.clauses:
        relation = clause.semantic_relation
        if _ARGMAX_RE.search(clause.text) and relation != "argmax":
            violations.append("extremum_miscompiled_as_non_extremum")
            relation = "argmax"
            changed = True
        elif _ARGMIN_RE.search(clause.text) and relation != "argmin":
            violations.append("extremum_miscompiled_as_non_extremum")
            relation = "argmin"
            changed = True
        elif _LIST_RE.search(clause.text) and relation == "count":
            violations.append("named_list_miscompiled_as_count")
            relation = "list"
            changed = True
        elif _CHANGE_RE.search(clause.text) and relation == "filter":
            violations.append("change_miscompiled_as_filter")
            relation = "change_over_time"
            changed = True
        elif (
            _CHANGE_RE.search(clause.text)
            and relation != "change_over_time"
            and inherited_periods
        ):
            violations.append("parallel_change_clause_normalized")
            relation = "change_over_time"
            changed = True
        periods = clause.periods
        if relation == "change_over_time":
            if len(periods) >= 2:
                inherited_periods = periods[:2]
            elif inherited_periods:
                periods = inherited_periods
                violations.append("change_periods_inherited_from_parallel_clause")
                changed = True
        metrics = _normalize_metrics_for_relation(
            clause.objects_or_metrics, clause.text, relation
        )
        if metrics != clause.objects_or_metrics:
            violations.append("spurious_or_shape_only_metric_removed")
            changed = True
        facets = tuple(name for name in _requested_facets(clause.text,relation) if name != 'total')
        endpoints = _comparison_endpoints(clause.text, relation, periods)
        if (
            relation == "compare"
            and not endpoints
            and re.match(r"^\s*(?:do|does|did)\b", clause.text, re.I)
            and re.search(r"\b(?:more|less|fewer)\b", clause.text, re.I)
            and not re.search(
                r"\b(?:than|between|versus|vs\.?|compare)\b",
                clause.text,
                re.I,
            )
        ):
            relation = "lookup"
            violations.append("unbound_comparative_normalized_to_lookup")
            changed = True
        if (
            relation in {"compare", "change_over_time"}
            and len(endpoints) != 2
            and len(draft.endpoint_hints) >= 2
        ):
            endpoint_hints = tuple(
                value for value in draft.endpoint_hints
                if not _comparison_endpoint_is_outcome_phrase(value)
            )
            if len(endpoint_hints) >= 2:
                endpoints = tuple(
                ComparisonEndpoint(
                    id=f"endpoint-{index}",
                    endpoint_type=_endpoint_type(value),
                    value=value,
                )
                    for index, value in enumerate(endpoint_hints[:2], start=1)
                )
                violations.append("comparison_endpoints_repaired_from_plan_entities")
                changed = True
            else:
                violations.append("comparison_endpoint_outcome_phrase_rejected")
        if relation in {"argmax", "argmin"} and endpoints:
            violations.append("extremum_had_comparison_endpoints")
            endpoints = ()
            changed = True
        if relation in {"compare", "change_over_time"} and len(endpoints) != 2:
            violations.append("comparison_endpoint_missing")
        if any(item.endpoint_type == "period" for item in endpoints) and not periods:
            violations.append("synthetic_period_endpoint_rejected")
            endpoints = tuple(
                item for item in endpoints if item.endpoint_type != "period"
            )
            changed = True
        predicate = _predicate(clause.text, relation, metrics)
        targets = _semantic_targets(clause.text, relation, endpoints, clause.populations)
        relation_target = _relation_target(
            clause.text,
            relation,
            clause.subject,
            targets["target_entity_type"],
        )
        raw_condition = _filter_condition(clause.text, metrics)
        condition = raw_condition if relation == "filter_rows" else None
        scope_constraint = (
            _scope_constraint(clause.text, metrics)
            if raw_condition is not None and relation != "filter_rows" else None
        )
        embedded = _embedded_event(clause.text, relation, metrics, periods)
        normalized.append(replace(
            clause,
            semantic_relation=relation,
            predicate=predicate,
            answer_shape=_answer_shape(
                clause.text, relation, clause.requested_cardinality
            ),
            objects_or_metrics=metrics,
            periods=periods,
            comparison_endpoints=endpoints,
            requested_facets=facets,
            requested_output_slots=_requested_output_slots(clause.text, relation),
            answer_relation=relation,
            embedded_event=embedded,
            filter_condition=condition,
            scope_constraint=scope_constraint,
            relation_target=relation_target,
            output_schema=(
                ("period", "metric_value") if relation == "filter_rows" else ()
            ),
            **targets,
        ))
    draft.violations.extend(_unique(violations))
    draft.normalization_count = 1
    draft.normalization_applied = changed
    final_payload = [item.as_dict() for item in normalized]
    signature = hashlib.sha256(json.dumps(
        final_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    endpoint_types = _unique(
        endpoint.endpoint_type
        for clause in normalized for endpoint in clause.comparison_endpoints
    )
    requested_facets = _unique(
        facet for clause in normalized for facet in clause.requested_facets
    )
    requested_slots = _unique(
        slot for clause in normalized for slot in clause.requested_output_slots
    )
    unresolved = any(
        violation == "comparison_endpoint_missing" for violation in violations
    )
    semantic_role_checks = tuple(
        f"{clause.id}:"
        + (
            "output_filter"
            if clause.filter_condition is not None
            else "scope_constraint"
            if clause.scope_constraint is not None
            else "no_comparator_role"
        )
        for clause in normalized
    )
    return tuple(normalized), SemanticValidationRecord(
        draft_operation=draft.draft_operation,
        violations=tuple(draft.violations),
        normalization_applied=draft.normalization_applied,
        normalization_count=draft.normalization_count,
        valid=not unresolved,
        final_semantic_signature=signature,
        endpoint_types=endpoint_types,
        requested_facets=requested_facets,
        requested_output_slots=requested_slots,
        semantic_role_checks=semantic_role_checks,
    )


def _clause(
    value: str,
    index: int,
    route_hint: str,
    default_cardinality: int,
    relation_override: str = "",
) -> CanonicalClause:
    original_value = value
    value = _request_clause_text(value)
    relation = relation_override or _relation(value, route_hint)
    populations = _extract_populations(value)
    if relation in {"compare", "change_over_time"} and experience_band_facets(value):
        populations = experience_scope_populations(value)
    metrics = _remove_scope_only_metrics(_metrics(value), populations)
    predicate = _predicate(value, relation, metrics)
    mentions = numeric_mentions(value, relation=relation)
    raw_condition = _filter_condition(value, metrics)
    condition = raw_condition if relation == "filter_rows" else None
    scope_constraint = (
        _scope_constraint(value, metrics)
        if raw_condition is not None and relation != "filter_rows" else None
    )
    periods = _unique(match.group(0) for match in _PERIOD_RE.finditer(value)
                      if not _numeric_identifier_role(value, match.start(), match.end()))
    values = _unique(
        mention.value for mention in mentions
        if mention.role not in {"filter_threshold", "period", "table_header", "unrelated_neighbor", "document_identifier", "entity_identifier"}
        and not (
            (condition and mention.value == condition.threshold)
            or (scope_constraint and mention.value == scope_constraint.threshold)
        )
    )
    units = _unique(
        mention.unit for mention in mentions
        if mention.unit
        and mention.role not in {
            "filter_threshold", "period", "table_header", "unrelated_neighbor", "document_identifier", "entity_identifier"
        }
        and not (
            (condition and mention.value == condition.threshold)
            or (scope_constraint and mention.value == scope_constraint.threshold)
        )
    )
    facets = _requested_facets(value,relation)
    explicit_negative = _negative_inclusion_request(value)
    polarity = "negative" if explicit_negative else "positive" if _POSITIVE_INCLUDE_RE.search(value) else "neutral"
    qualifiers = list(facets)
    qualifiers.extend(experience_band_facets(value))
    if re.search(r"(?:\b(?:current|present|latest)\b|(?:נוכחי|כיום|בעת\s+הסקר))", value, re.I):
        qualifiers.append("current")
    if re.search(r"(?:\b(?:previous|historical|prior)\b|(?:קודם|היסטורי|בעבר))", value, re.I):
        qualifiers.append("historical")
    if re.search(r"(?:\b(?:all|every|each|both|complete)\b|(?<!\w)(?:כל|לכל|שני|שתי|ו?ה?מלא(?:ה|ים|ות)?)(?!\w))", value, re.I):
        qualifiers.append("complete")
    cardinality = _cardinality(value, default_cardinality if relation in {"list", "summary"} else 1)
    completeness = (
        "complete_candidate_matrix" if relation in {"argmax", "argmin"}
        else "all_requested_clauses" if "complete" in qualifiers
        else "complete_proposition"
    )
    endpoints = _comparison_endpoints(value, relation, periods)
    targets = _semantic_targets(value, relation, endpoints, populations)
    subject = _subject(value, metrics, relation)
    relation_target = _relation_target(
        value, relation, subject, targets["target_entity_type"]
    )
    embedded = _embedded_event(value, relation, metrics, periods)
    return CanonicalClause(
        id=f"clause-{index}", text=original_value, semantic_relation=relation,
        answer_shape=_answer_shape(value, relation, cardinality),
        subject=subject, predicate=predicate,
        objects_or_metrics=metrics, requested_values=values, units=units,
        periods=periods, populations=populations, polarity=polarity,
        qualifiers=_unique(qualifiers), requested_cardinality=cardinality,
        completeness=completeness, numeric_mentions=mentions,
        comparison_endpoints=endpoints,
        requested_facets=facets,
        requested_output_slots=_requested_output_slots(value, relation),
        answer_relation=relation,
        embedded_event=embedded,
        filter_condition=condition,
        scope_constraint=scope_constraint,
        relation_target=relation_target,
        output_schema=(("period", "metric_value") if relation == "filter_rows" else ()),
        **targets,
    )


def _requirements_for_clause_raw(clause: CanonicalClause) -> tuple[TypedRequirement, ...]:
    requested_quantity = _request_quantity_semantics(clause)
    base = {
        "clause_id": clause.id, "subject": clause.subject,
        "predicate": clause.predicate, "objects_or_metrics": clause.objects_or_metrics,
        "requested_values": clause.requested_values, "units": clause.units,
        "periods": clause.periods, "populations": clause.populations,
        "polarity": clause.polarity, "qualifiers": clause.qualifiers,
        "cardinality": clause.requested_cardinality,
        "facet": clause.requested_facets[0] if len(clause.requested_facets) == 1 else "",
        "target_entity_type": clause.target_entity_type,
        "counted_entity": clause.counted_entity,
        "requested_attribute": clause.requested_attribute,
        "comparison_subject": clause.comparison_subject,
        "comparison_reference": clause.comparison_reference,
        "scope_population": clause.scope_population,
        "relation_target": clause.relation_target,
        "answer_relation": clause.answer_relation or clause.semantic_relation,
        "filter_operator": (
            clause.filter_condition.operator if clause.filter_condition else ""
        ),
        "filter_threshold": (
            clause.filter_condition.threshold if clause.filter_condition else ""
        ),
        "filter_unit": (
            clause.filter_condition.unit if clause.filter_condition else ""
        ),
        "scope_operator": (
            clause.scope_constraint.operator if clause.scope_constraint else ""
        ),
        "scope_threshold": (
            clause.scope_constraint.threshold if clause.scope_constraint else ""
        ),
        "scope_unit": (
            clause.scope_constraint.unit if clause.scope_constraint else ""
        ),
        # Retain the historical serialized role for older callers.  The
        # orthogonal fields below are the sole compatibility authority.
        "scalar_role": _request_scalar_role(clause),
        "quantity_kind": requested_quantity.quantity_kind,
        "statistical_role": requested_quantity.statistical_role,
        "aggregation_role": requested_quantity.aggregation_role,
        "quantity_semantics_authoritative": True,
    }
    prefix = clause.id.replace("clause", "req")
    relation = clause.semantic_relation
    requested_slots = set(clause.requested_output_slots)
    if relation in {"argmax", "argmin"}:
        ids = [f"{prefix}-category-set", f"{prefix}-value-matrix", f"{prefix}-common-scope", f"{prefix}-winner"]
        requirements = [
            TypedRequirement(ids[0], kind="category_set", completeness="complete_candidate_matrix", output_slot="candidate_set", value_type="matrix", answer_producing=False, **base),
            TypedRequirement(ids[1], kind="category_value_bindings", completeness="each_value_bound_to_category", output_slot="candidate_values", value_type="matrix", answer_producing=False, **base),
            TypedRequirement(ids[2], kind="common_metric_unit_period_population", completeness="common_scope", output_slot="common_scope", value_type="scope", answer_producing=False, **base),
            TypedRequirement(ids[3], kind="winner", completeness="winning_category", output_slot="winner", value_type="category", **base),
        ]
        if "percentage" in requested_slots:
            requirements.append(TypedRequirement(
                f"{prefix}-winner-percentage", kind="winner_attribute", completeness="winner_percentage",
                output_slot="percentage",
                linked_requirement_ids=(ids[3],), **{**base, "objects_or_metrics": ("percentage",), "units": ("%",),
                   "quantity_kind": QuantityKind.PERCENTAGE.value,
                   "aggregation_role": AggregationRole.SHARE.value},
            ))
        if "count" in requested_slots:
            requirements.append(TypedRequirement(
                f"{prefix}-winner-count", kind="winner_attribute", completeness="winner_count",
                output_slot="count",
                linked_requirement_ids=(ids[3],), **{**base, "objects_or_metrics": ("count",),
                   "quantity_kind": QuantityKind.COUNT.value,
                   "aggregation_role": AggregationRole.CATEGORY.value},
            ))
        return tuple(requirements)
    if relation == "filter_rows":
        if clause.filter_condition is None:
            return ()
        condition_id = f"{prefix}-filter-condition"
        period_id = f"{prefix}-row-period"
        value_id = f"{prefix}-row-value"
        linked = (condition_id, period_id, value_id)
        without_threshold = {
            **base,
            "requested_values": (),
            "periods": (),
            "linked_requirement_ids": linked,
        }
        return (
            TypedRequirement(
                condition_id,
                kind="filter_condition",
                completeness="validated_filter_condition",
                output_slot="filter_condition",
                value_type="operator_condition",
                answer_producing=False,
                **{**base, "requested_values": (), "periods": (),
                   "linked_requirement_ids": (period_id, value_id),
                   "statistical_role": StatisticalRole.FILTER_THRESHOLD.value},
            ),
            TypedRequirement(
                period_id,
                kind="filtered_row_period",
                completeness="period_for_each_matching_row",
                output_slot="row_period",
                value_type="period",
                **{**without_threshold,
                   "quantity_kind": QuantityKind.PERIOD.value},
            ),
            TypedRequirement(
                value_id,
                kind="filtered_row_value",
                completeness="metric_value_for_each_matching_row",
                output_slot="row_metric_value",
                value_type="numeric",
                **without_threshold,
            ),
        )
    if relation == "cause":
        return (TypedRequirement(
            f"{prefix}-reason", kind="cause_or_reason",
            completeness="reason_not_observation", output_slot="text_proposition",
            value_type="text_proposition", **{**base, "predicate": "caused_by", "polarity": "neutral"},
        ),)
    if relation == "inclusion":
        return (TypedRequirement(
            f"{prefix}-positive-inclusion", kind="positive_inclusion_or_count",
            completeness="positive_inclusion", output_slot=clause.requested_output_slots[0],
            value_type=(
                "count" if clause.requested_output_slots[0] == "count"
                else "text_proposition"
            ),
            **{**base, "predicate": "included_in", "polarity": "positive"},
        ),)
    if relation in {"exclusion", "exception"}:
        return (TypedRequirement(
            f"{prefix}-negative-relation", kind="negative_relation",
            completeness="explicit_negative_predicate", output_slot="entity",
            value_type="text_proposition",
            **{**base, "polarity": "negative"},
        ),)
    if relation == "list":
        if clause.answer_shape == "list_with_attributes":
            item_id, attribute_id = f"{prefix}-items", f"{prefix}-item-attributes"
            return (
                TypedRequirement(
                    item_id, kind="list", completeness="requested_item_set",
                    output_slot="items", value_type="list",
                    linked_requirement_ids=(attribute_id,), **base,
                ),
                TypedRequirement(
                    attribute_id, kind="list_attributes",
                    completeness="attribute_for_each_requested_item",
                    output_slot="item_attributes", value_type="list",
                    linked_requirement_ids=(item_id,), **base,
                ),
            )
        return (TypedRequirement(
            f"{prefix}-list", kind="list", completeness="requested_item_set",
            output_slot="items", value_type="list", **base,
        ),)
    count_requested = "count" in requested_slots
    percentage_requested = "percentage" in requested_slots
    if count_requested and percentage_requested and relation not in {"compare", "change_over_time"}:
        count_id, percentage_id = f"{prefix}-count", f"{prefix}-percentage"
        return (
            TypedRequirement(
                count_id, kind="count", completeness="count_value",
                output_slot="count",
                linked_requirement_ids=(percentage_id,),
                **{**base, "objects_or_metrics": tuple(item for item in clause.objects_or_metrics if item != "percentage") or ("count",), "units": tuple(unit for unit in clause.units if unit != "%"),
                   "quantity_kind": QuantityKind.COUNT.value},
            ),
            TypedRequirement(
                percentage_id, kind="percentage", completeness="percentage_value",
                output_slot="percentage",
                linked_requirement_ids=(count_id,),
                **{**base, "predicate": "has_percentage", "objects_or_metrics": ("percentage",), "units": ("%",),
                   "quantity_kind": QuantityKind.PERCENTAGE.value},
            ),
        )
    if percentage_requested and relation not in {"compare", "change_over_time"}:
        return (TypedRequirement(
            f"{prefix}-percentage", kind="percentage",
            completeness="percentage_value", output_slot="percentage",
            value_type="percentage",
            **{**base, "predicate": "has_percentage", "objects_or_metrics": ("percentage",), "units": ("%",),
               "quantity_kind": QuantityKind.PERCENTAGE.value},
        ),)
    if count_requested and relation not in {"compare", "change_over_time"}:
        return (TypedRequirement(
            f"{prefix}-count", kind="count", completeness="count_value",
            output_slot="count",
            value_type="count",
            **{**base, "objects_or_metrics": tuple(item for item in clause.objects_or_metrics if item != "percentage") or ("count",),
               "quantity_kind": QuantityKind.COUNT.value},
        ),)
    if relation in {"compare", "change_over_time"}:
        metrics = clause.objects_or_metrics or (clause.predicate,)
        bands = experience_band_facets(clause.text)
        if not bands and re.search(
            r"\b(?:by|across)\s+(?:experience|seniority)\s+levels?\b|"
            r"לפי\s+(?:רמות\s+)?ניסיון", clause.text, re.I,
        ):
            bands = ("experience_band:lower_band", "experience_band:higher_band")
        metric_facets = tuple((metric, band) for metric in metrics for band in (bands or (base["facet"],)))
        output: list[TypedRequirement] = []
        for metric_index, (metric, band) in enumerate(metric_facets, start=1):
            comparison_base = {**base, "facet": band}
            metric_predicate = {
                "employee_count": "employs_workers",
                "employee_recruitment": "recruits_employees",
                "capital_raised": "raises_capital",
            }.get(metric, clause.predicate)
            endpoint_ids: list[str] = []
            for endpoint_index, endpoint in enumerate(clause.comparison_endpoints, start=1):
                requirement_id = f"{prefix}-metric-{metric_index}-endpoint-{endpoint_index}"
                endpoint_ids.append(requirement_id)
                output.append(TypedRequirement(
                    requirement_id,
                    kind="comparison_endpoint", completeness="bound_endpoint",
                    endpoint_type=endpoint.endpoint_type,
                    endpoint_value=endpoint.value,
                    output_slot="endpoint_values",
                    **{**comparison_base, "predicate": metric_predicate,
                       "objects_or_metrics": (metric,),
                       "periods": ((endpoint.value,) if endpoint.endpoint_type == "period" else ()),
                       "populations": ((endpoint.value,) if endpoint.endpoint_type == "population" else base["populations"])},
                ))
            if len(endpoint_ids) == 2:
                output.append(TypedRequirement(
                    f"{prefix}-metric-{metric_index}-direction",
                    kind="comparison_direction",
                    completeness="direction_between_bound_endpoints",
                    linked_requirement_ids=tuple(endpoint_ids),
                    output_slot="direction",
                    value_type="direction",
                    **{**comparison_base, "predicate": (
                        "changed_between" if relation == "change_over_time"
                        else metric_predicate
                    ), "objects_or_metrics": (metric,)},
                ))
                if "significance" in requested_slots:
                    output.append(TypedRequirement(
                        f"{prefix}-metric-{metric_index}-significance",
                        kind="comparison_significance",
                        completeness="significance_between_bound_endpoints",
                        linked_requirement_ids=tuple(endpoint_ids),
                        output_slot="significance",
                        value_type="text_proposition",
                        **{**base, "predicate": (
                            "changed_between" if relation == "change_over_time"
                            else metric_predicate
                        ), "objects_or_metrics": (metric,)},
                    ))
        return tuple(output)
    if relation == "purpose" and {"purpose", "policy_use"} <= requested_slots:
        purpose_id, use_id = f"{prefix}-purpose", f"{prefix}-policy-use"
        return (
            TypedRequirement(purpose_id, kind="purpose", completeness="stated_purpose", output_slot="purpose", value_type="text_proposition", linked_requirement_ids=(use_id,), **base),
            TypedRequirement(use_id, kind="intended_use", completeness="stated_policy_use", output_slot="policy_use", value_type="text_proposition", linked_requirement_ids=(purpose_id,), **base),
        )
    if relation in {"purpose", "definition", "explanation", "procedure"}:
        return (TypedRequirement(
            f"{prefix}-{relation}", kind=relation,
            completeness=clause.completeness,
            output_slot="text_proposition", value_type="text_proposition", **base,
        ),)
    scalar_slot = (
        clause.requested_output_slots[0]
        if clause.requested_output_slots else "scalar"
    )
    return (TypedRequirement(
        f"{prefix}-{relation}", kind=relation, completeness=clause.completeness,
        output_slot=scalar_slot,
        value_type=_scalar_value_type(clause),
        **base,
    ),)


_EXPLICIT_TOTAL_REQUEST_RE = re.compile(
    r"\b(?:total|overall|aggregate)\s+(?:count|number|amount|demand|"
    r"revenue|cost|capacity)\b|"
    r"(?:סך\s*הכול|סה.?כ)\s*(?:מספר|כמות|ביקוש|הכנסות|עלות|קיבולת)|"
    r"(?:ה)?כולל(?:ת|ים|ות)?\s+ל?(?:מספר|כמות|ביקוש|הכנסות|עלות|קיבולת|עובדים)",
    re.IGNORECASE,
)

_TOTAL_REQUEST_CUE_RE = re.compile(
    r"\b(?:total|overall|aggregate)\b|"
    r"(?:סך\s*הכול|סה.?כ|ה?כולל(?:ת|ים|ות)?)",
    re.IGNORECASE,
)
_COMPARISON_CONTEXT_TOKEN_RE = re.compile(
    r"\b(?:survey|report|year|quarter|period)\b|"
    r"(?:סקר|דוח|שנה|רבעון|תקופה)",
    re.IGNORECASE,
)


def _strip_request_subject_grammar(value: str) -> str:
    """Remove interrogative/copular wrapper tokens from a source concept."""
    value = normalize_semantic_text(value).strip(" ,;:.?!")
    value = re.sub(
        r"^(?:what|which|who|how(?:\s+many)?|many|much)\s+(?:is|are|was|were|"
        r"does|do|did|has|have|had)?\s*",
        "", value, flags=re.IGNORECASE,
    )
    return re.sub(
        r"^(?:(?:מה|איזה|איזו|אילו|מי|כמה)\s+)?(?:היה|הייתה|היו|הוא|היא|הם|הן)\s+",
        "", value,
    ).strip(" ,;:.?!")


_SUBJECT_GRAMMAR_RE = re.compile(
    r"^\s*(?:the\s+)?(?:many\s+)?(?:is|are|was|were|does|do|did|has|have|"
    r"had|of|in|for|among|within|across)\s+|"
    r"^\s*(?:ה(?:וא|יא|יו|ייתה|יהיה)?\s+|ב(?:תוך|קרב)\s+|של\s+|מן\s+)",
    re.IGNORECASE,
)
_SUBJECT_TRAILING_GRAMMAR_RE = re.compile(
    r"\s+(?:in|for|of|among|within|across|from|to|the|and|or|ו|של|ב|מן)$",
    re.IGNORECASE,
)
_DEPENDENT_OUTPUT_SUBJECT_RE = re.compile(
    r"\s*,?\s*(?:and\s+what|ומה)\b.*$",
    re.IGNORECASE,
)
_METRIC_WRAPPER_RE = re.compile(
    r"\b(?:estimated|estimate|operational|total|overall|number|count|amount|"
    r"value|rate|share)\b|(?:אומדן|הערכה|תפעולי|סך\s*הכול|מספר|כמות|ערך|שיעור|כולל(?:ת|ים|ות)?)",
    re.IGNORECASE,
)
_CLAUSE_RESIDUE_RE = re.compile(
    r"\b(?:reported|responded|experienced|received|included|employ(?:s|ed)?|"
    r"recruit(?:s|ed)?|raise(?:s|d)?|change(?:d)?|compare(?:d)?|complete(?:d|s|ing)?|"
    r"use(?:d|s|ing)?|identif(?:y|ied|ies|ying)|process(?:es|ed|ing)?|purpose|objective|reason|cause|"
    r"definition|method|procedure|was|were|is|are|conducted|handles?|and\s+what)\b|"
    r"(?:אמרו|דיווחו|נכלל(?:ה|ו)?|מעסיק(?:ה|ים|ות)?|גייס(?:ה|ו)?|"
    r"השתנה|השווה|מטפל(?:ת|ים|ות)?|אחראי(?:ת|ים|ות)?|מטרה|סיבה|הגדרה|הליך|ומה)",
    re.IGNORECASE,
)
_RELATION_COMPLEMENT_RE = re.compile(
    r"\b(?:reported|responded|said|consider(?:s|ed)?|experienced|received)\s+"
    r"(?P<complement>[A-Za-z][A-Za-z0-9'’\-]*(?:\s+[A-Za-z][A-Za-z0-9'’\-]*){0,5})",
    re.IGNORECASE,
)
_EVENT_SUBJECT_RE = re.compile(
    r"^\s*(?P<entity>[A-Za-z][A-Za-z0-9'’-]*(?:\s+[A-Za-z][A-Za-z0-9'’-]*){0,3})\s+"
    r"(?:reported|responded|said|consider(?:s|ed)?|experienced|received)\s+"
    r"(?P<metric>[^?.,;]+)|"
    r"^\s*(?P<entity_he>[\u05d0-\u05ea][\u05d0-\u05ea\u0022\u0027\u05f3-]*(?:\s+[\u05d0-\u05ea\u0022\u0027\u05f3-]+){0,3})\s+"
    r"(?:אמרו|דיווחו|שקלו|חוו|קיבלו)\s+(?P<metric_he>[^?.,;]+)",
    re.IGNORECASE,
)


def _independent_entity_value(value: str) -> str:
    """Return a bounded entity slot, rejecting request-proposition residue.

    A target/entity field is an authoritative semantic owner only when the
    parser supplied an entity-like span.  Falling back to a full clause would
    duplicate relation, metric, and output wording into a second hard
    constraint.
    """
    value = _strip_request_subject_grammar(value)
    value = re.split(r"\s*(?:,|;|\?|\band\s+what\b|ומה)\s*", value, maxsplit=1,
                   flags=re.IGNORECASE)[0]
    # A population qualifier is independently represented by the canonical
    # population dimension.  Counted-entity extraction may include it in the
    # request parser's broad entity span; retain only the entity head rather
    # than letting that scope prevent a concrete target from being compiled.
    value = re.split(
        r"\s+(?:among|within|across)\s+|\s+(?:בקרב|בתוך)\s+",
        value,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    value = normalize_semantic_text(value).strip(" ,;:.?!")
    if not value or len(value.split()) > 5 or _CLAUSE_RESIDUE_RE.search(value):
        return ""
    return value[:MAX_CONTRACT_TEXT]


def _remove_owned_subject_span(subject: str, owner: str) -> str:
    """Remove an exact request span owned by another semantic dimension."""
    owner = normalize_semantic_text(owner).strip(" ,;:.?!")
    if not owner:
        return subject
    return normalize_semantic_text(re.sub(
        re.escape(owner), " ", subject, flags=re.IGNORECASE,
    )).strip(" ,;:.?!")


def _event_entity_and_metric(clause: CanonicalClause) -> tuple[str, str]:
    """Factor a source-relevant event into entity and metric slots.

    This is intentionally limited to an explicit local proposition shape.  It
    does not turn arbitrary lexical overlap into a metric and does not inspect
    source evidence.
    """
    match = _EVENT_SUBJECT_RE.search(
        normalize_semantic_text(clause.subject).strip(" ,;:.?!")
    )
    if match is None:
        return "", ""
    entity = _independent_entity_value(
        match.group("entity") or match.group("entity_he") or ""
    )
    metric = normalize_semantic_text(
        match.group("metric") or match.group("metric_he") or ""
    ).strip(" ,;:.?!")
    return entity, metric[:MAX_CONTRACT_TEXT]


def _metric_attached_target(
    clause: CanonicalClause,
    metric: str,
) -> str:
    """Return an exact request-local entity grammatically attached to metric.

    A metric may carry its measured entity (``demand for engineers``) while
    the request parser has no standalone target slot.  This is a structural
    request relation, not source-text matching or token deletion.  It keeps
    the entity in the target owner so it need not be duplicated as subject.
    """
    metric = normalize_semantic_text(metric).strip(" ,;:.?!")
    if not metric:
        return ""
    text = normalize_semantic_text(clause.subject or clause.text)
    escaped = re.escape(metric)
    match = re.search(
        rf"{escaped}(?:\s+(?:total|overall|aggregate|"
        rf"ה?כולל(?:ת|ים|ות)?|סך\s*הכול))?\s+"
        rf"(?:(?:of|for|של)\s+|ל)(?P<target>[^,;?.]+)",
        text,
        re.IGNORECASE,
    )
    if match is None:
        return ""
    target = normalize_semantic_text(match.group("target"))
    target = re.split(
        r"\s+(?:in|on|during|between|among|within|at|"
        r"בין|בסקר|בשנת|בעת|בקרב|בתוך)\s+",
        target,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    target = re.sub(
        r"\s+(?:in|on|during|between|among|within|at|of|for|של|ל)$",
        "",
        target,
        flags=re.IGNORECASE,
    )
    target = re.sub(r"^(?:the|a|an)\s+", "", target, flags=re.IGNORECASE)
    return _independent_entity_value(target)


def _source_language_metric_value(
    metric: str,
    clause: CanonicalClause,
) -> str:
    """Keep a normalized metric comparable to a source-language proposition.

    The frozen semantic frame may represent a metric with a canonical English
    label while the request supplies an exact Hebrew source-language span.  A
    full request subject is not an acceptable substitute: preserve only the
    bounded metric match exposed by the existing metric authority.
    """
    metric = normalize_semantic_text(metric).strip(" ,;:.?!")
    text = normalize_semantic_text(clause.text)
    metric_has_latin = bool(re.search(r"[A-Za-z]", metric))
    metric_has_hebrew = bool(re.search(r"[\u05d0-\u05ea]", metric))
    text_has_latin = bool(re.search(r"[A-Za-z]", text))
    text_has_hebrew = bool(re.search(r"[\u05d0-\u05ea]", text))
    script_mismatch = (
        metric_has_latin and text_has_hebrew and not text_has_latin
        or metric_has_hebrew and text_has_latin and not text_has_hebrew
    )
    if not metric or not script_mismatch:
        return metric
    normalized_names = {
        item for item in metric.split()
        if item in {name for name, _pattern in _METRIC_PATTERNS}
    }
    matches = [
        normalize_semantic_text(pattern.search(text).group(0)).strip(" ,;:.?!")
        for name, pattern in _METRIC_PATTERNS
        if name in normalized_names and pattern.search(text) is not None
    ]
    return _unique(matches)[0] if len(_unique(matches)) == 1 else metric


def _minimal_requirement_subject(
    clause: CanonicalClause,
    *,
    metric: str = "",
    target_entity: str = "",
    relation_arguments: str = "",
    population: str = "",
    facet: str = "",
) -> str:
    """Factor a minimal independently meaningful source subject.

    This operates over request-owned structured slots.  It removes only exact
    spans already canonically owned by another dimension, never source text,
    synonyms, or broad lexical matches.  If no core remains, the subject stays
    unconstrained rather than becoming a request proposition.
    """
    # Extremum wording is an operator instruction, not an entity equality
    # constraint on the winning source fact.  Its category type and common
    # metric scope are represented by their dedicated operator requirements;
    # retaining "lowest/highest ..." here would make an operator result prove
    # an invented source subject.
    if clause.semantic_relation in {"argmax", "argmin"}:
        return ""
    text = normalize_semantic_text(clause.text)
    object_match = re.search(
        r"\b(?:include(?:d)?|employ(?:s|ed)?|recruit(?:s|ed)?|raise(?:s|d)?)\s+"
        r"(?:the\s+)?(?P<object>[A-Za-z][A-Za-z0-9'’-]*(?:\s+[A-Za-z][A-Za-z0-9'’-]*){0,4})",
        text,
        re.IGNORECASE,
    )
    # Direct inclusion/relation objects have no other entity owner in the
    # canonical frame, so they remain the source subject.
    if object_match and not target_entity and not relation_arguments:
        return object_match.group("object").strip(" ,;:.?!")[:MAX_CONTRACT_TEXT]

    subject = _strip_request_subject_grammar(clause.subject)
    # A linked output attribute (for example, count plus percentage) belongs
    # to its output requirement.  It is not a second source subject.  Clause
    # segmentation has already established this as a dependent output, so this
    # only removes request grammar from the subject candidate.
    subject = _DEPENDENT_OUTPUT_SUBJECT_RE.sub("", subject)
    # Remove enclosing structured owners first.  A population such as
    # ``among sites in the urban region`` contains the shorter counted entity
    # ``sites``; removing the latter first would destroy the exact scope span
    # and leave request-local scope words as a fabricated subject.
    for owner in sorted(
        (target_entity, metric, relation_arguments, population, facet),
        key=lambda item: len(normalize_semantic_text(item)),
        reverse=True,
    ):
        subject = _remove_owned_subject_span(subject, owner)
        # A relation target may legitimately contain a period that is owned by
        # the endpoint requirement rather than the source relation itself.
        # Remove the same bounded relation complement after that independently
        # owned period has been stripped, without treating a partial lexical
        # overlap as a new semantic owner.
        owner_without_periods = owner
        for period in clause.periods:
            owner_without_periods = _remove_owned_subject_span(
                owner_without_periods, period,
            )
        if owner_without_periods:
            subject = _remove_owned_subject_span(subject, owner_without_periods)
    # A wrapper is removable only when another canonical dimension owns the
    # measurable concept.  In an otherwise unconstrained phrase such as
    # "current estimate of specialists", stripping ``estimate`` would turn a
    # source-comparable concept into request residue.
    if metric:
        subject = _METRIC_WRAPPER_RE.sub(" ", subject)
    # Remove predicate/output framing token-by-token before deciding whether a
    # core subject remains.  Rejecting the entire residue would turn
    # ``Gateway Cedar process`` into no subject, although the bounded entity
    # ``Gateway Cedar`` is still independently meaningful.
    previous = None
    while subject and subject != previous:
        previous = subject
        subject = _CLAUSE_RESIDUE_RE.sub(" ", subject)
        subject = _strip_request_subject_grammar(subject)
        subject = _SUBJECT_GRAMMAR_RE.sub("", subject)
        subject = _SUBJECT_TRAILING_GRAMMAR_RE.sub("", subject)
        subject = normalize_semantic_text(subject).strip(" ,;:.?!")
    subject = _SUBJECT_GRAMMAR_RE.sub("", subject)
    subject = re.sub(r"^(?:the|a|an)\s+", "", subject, flags=re.IGNORECASE)
    subject = normalize_semantic_text(subject).strip(" ,;:.?!")
    # A relation complement (``reported outages``) names what happened, not
    # a distinct source actor.  Once the counted entity and population have
    # been assigned their own canonical dimensions, promoting that complement
    # into ``subject`` would make an event phrase an invented owner equality.
    # This is deliberately based on the request's bounded relation structure,
    # not on source text or a benchmark-specific vocabulary.
    complement_match = _RELATION_COMPLEMENT_RE.search(
        normalize_semantic_text(clause.subject)
    )
    if complement_match:
        complement = normalize_semantic_text(
            complement_match.group("complement")
        ).strip(" ,;:.?!")
        complement = _SUBJECT_TRAILING_GRAMMAR_RE.sub("", complement)
        if subject.casefold() == complement.casefold():
            subject = ""
    if subject.casefold() in {
        "the", "a", "an", "the there", "there", "that", "this", "it",
        "in", "for", "of", "among", "within", "across", "from", "to",
        "ה", "ב", "של", "מן",
    }:
        subject = ""

    # A short surviving noun phrase is independently meaningful.  Predicate
    # and output-clause residue has already been removed rather than being
    # converted into an entity equality constraint.
    if subject and len(subject.split()) <= 5:
        return subject[:MAX_CONTRACT_TEXT]
    return ""


_FACTORABLE_REQUEST_LEAD_RE = re.compile(
    r"^\s*(?:"
    r"how\s+many|what(?:\s+was|\s+is)?|which|who|"
    r"compare|show|give|list|"
    r"כמה|מה\s+שיעור|מהו|מה\s+היה|כיצד|השווה\s+את|באילו\s+(?:\d+|שלוש|שתי|שני)\s+קבוצות"
    r")\s+",
    re.IGNORECASE,
)
_FACTORABLE_DEPENDENT_OUTPUT_RE = re.compile(
    r"\s*(?:,|;|\band\s+what\b|\band\s+(?:state|say|tell|explain)\b|ומה)\s*"
    r"(?:was|is|the|what|how|whether|מה|היה|הייתה|שיעור(?:ן|ם)?).*$",
    re.IGNORECASE,
)
_SOURCE_RELATION_VERB_RE = re.compile(
    r"\b(?:reported|responded|said|consider(?:s|ed)?|"
    r"employ(?:s|ed)?|recruit(?:s|ed)?|raise(?:s|d)?|"
    r"changed?|compared?)\b|"
    r"(?:אמרו|דיווחו|שקלו|מעסיק(?:ה|ים|ות)?|גייס(?:ה|ו)?|"
    r"השתנה|השווה)",
    re.IGNORECASE,
)


def _lossless_subject_residual(
    clause: CanonicalClause,
    *,
    metric: str = "",
    target_entity: str = "",
    relation_arguments: str = "",
    population: str = "",
    facet: str = "",
    source_predicate: str = "",
    aggregation_role: str = "",
    allow_total_comparison_deconstraint: bool = False,
) -> str:
    """Return a bounded request-local concept not owned by another slot.

    C21 deliberately avoided clause-shaped subjects.  That is still correct,
    but an empty subject is not a license to discard a discriminator merely
    because the remaining concept has more than five tokens.  This helper
    removes only source-admissibility owners with request-local provenance and
    preserves the remaining bounded concept.  It never examines source text.
    """
    subject = normalize_semantic_text(clause.subject)
    if not subject:
        subject = normalize_semantic_text(clause.text)
    subject = _FACTORABLE_DEPENDENT_OUTPUT_RE.sub(" ", subject)
    subject = _FACTORABLE_REQUEST_LEAD_RE.sub("", subject)
    # Request periods are independently owned by their endpoint requirements;
    # remove their literal tokens before deciding whether a source subject
    # remains.  A source-local threshold is different: it distinguishes the
    # requested population and must stay with the residual concept.
    for period in clause.periods:
        subject = _remove_owned_subject_span(subject, period)
    for owner in sorted(
        (target_entity, metric, relation_arguments, population, facet),
        key=lambda item: len(normalize_semantic_text(item)),
        reverse=True,
    ):
        subject = _remove_owned_subject_span(subject, owner)
    if (
        allow_total_comparison_deconstraint
        and aggregation_role == AggregationRole.TOTAL.value
    ):
        # TOTAL remains an aggregation constraint with the original request
        # span.  It is not a second subject equality once that owner exists.
        subject = _TOTAL_REQUEST_CUE_RE.sub(" ", subject)
    if source_predicate:
        # The predicate already owns the relation token itself.  Keep its
        # actor, object modifiers, and local threshold in the subject core.
        subject = _SOURCE_RELATION_VERB_RE.sub(" ", subject)
    # A threshold in the request may delimit the requested actor/population
    # even when it is not an answer scalar.  Preserve its source-local unit in
    # the remaining subject concept rather than dropping it with the output
    # grammar or turning it into a derived operator fact.
    for mention in clause.numeric_mentions:
        if mention.role != "filter_threshold":
            continue
        threshold = normalize_semantic_text(
            f"{mention.value} {mention.unit}".strip()
        )
        if threshold and threshold not in subject:
            subject = f"{subject} {threshold}".strip()
    subject = _DEPENDENT_OUTPUT_SUBJECT_RE.sub(" ", subject)
    if allow_total_comparison_deconstraint and metric:
        subject = _METRIC_WRAPPER_RE.sub(" ", subject)
    if allow_total_comparison_deconstraint and clause.periods:
        # Endpoint containers (survey/year/quarter) are structural discourse
        # around the independently required endpoint periods, not a source
        # actor.  This does not cross proposition boundaries or inspect
        # evidence text.
        subject = _COMPARISON_CONTEXT_TOKEN_RE.sub(" ", subject)
        # Removing a comparison container can leave Hebrew prefixed
        # prepositions/conjunctions as standalone grammar tokens.  They carry
        # no actor identity after their endpoint owner has been preserved.
        subject = re.sub(
            r"(?<![\u05d0-\u05ea])(?:ו?[בל]|ו)(?![\u05d0-\u05ea])", " ", subject,
        )
    previous = None
    while subject and subject != previous:
        previous = subject
        subject = _CLAUSE_RESIDUE_RE.sub(" ", subject)
        subject = _strip_request_subject_grammar(subject)
        subject = _SUBJECT_GRAMMAR_RE.sub(" ", subject)
        subject = _SUBJECT_TRAILING_GRAMMAR_RE.sub(" ", subject)
        subject = normalize_semantic_text(subject).strip(" ,;:.?!-–—")
    subject = re.sub(
        r"^(?:many|what|which|who|how|the|a|an|כמה|מה|שיעור|איזה|איזו|אילו|מי)\s+",
        "", subject, flags=re.IGNORECASE,
    )
    subject = re.sub(r"\b(?:the|a|an|for|of|in|to)\b", " ", subject,
                     flags=re.IGNORECASE)
    subject = normalize_semantic_text(subject).strip(" ,;:.?!-–—")
    if not subject or subject.casefold() in {
        "the", "a", "an", "there", "that", "this", "it", "in", "for",
        "of", "among", "within", "across", "from", "to", "many", "ה", "ב",
        "של", "מן",
    }:
        return ""
    # A bounded residual is a semantic concept, not a reconstructed user
    # question.  A longer phrase is retained when it is the only canonical
    # owner of a discriminative modifier; the global contract bound still
    # prevents arbitrary request copying.
    words = subject.split()
    if len(words) > 14:
        return ""
    return subject[:MAX_CONTRACT_TEXT]


def _source_predicate_is_explicit(requirement: TypedRequirement) -> bool:
    """Whether the requirement predicate expresses a source proposition."""
    return str(requirement.predicate or "") not in {
        "", "stated_as", "has_count", "has_percentage", "changed_between",
        "compared_with", "maximum_of", "minimum_of", "satisfies_filter",
        "has_members",
    }


def _explicit_source_predicate(
    requirement: TypedRequirement,
    clause: CanonicalClause,
) -> str:
    """Return only a relation that the request itself semantically names."""
    if clause.predicate in DOCUMENT_ROLE_PREDICATES:
        return clause.predicate
    if _source_predicate_is_explicit(requirement):
        return requirement.predicate
    if clause.predicate in {
        "employs_workers", "recruits_employees", "raises_capital",
    }:
        return clause.predicate
    request = normalize_semantic_text(clause.text)
    if re.search(
        r"\b(?:reported|responded|said|consider(?:s|ed)?)\b|"
        r"(?:אמרו|דיווחו|שקלו)",
        request,
        re.IGNORECASE,
    ):
        return "stated_as"
    if re.search(
        r"\b(?:handles?|is\s+responsible\s+for|are\s+responsible\s+for|"
        r"performs?)\b|(?:מטפל(?:ת|ים|ות)?|אחראי(?:ת|ים|ות)?)",
        request,
        re.IGNORECASE,
    ):
        return "performed_by"
    return ""


def _explicit_relation_target(
    clause: CanonicalClause,
    predicate: str,
    fallback: str,
) -> str:
    """Compile a request-local relational target without using output labels."""
    if fallback:
        return fallback
    request = normalize_semantic_text(clause.text)

    def bounded_target(value: str) -> str:
        for period in clause.periods:
            value = _remove_owned_subject_span(value, period)
        value = re.sub(
            r"\b(?:in|on|at|during|between|ב|ב־)\s*$", "", value,
            flags=re.IGNORECASE,
        )
        return normalize_semantic_text(value).strip(" ,;:.?!")

    if predicate == "stated_as":
        match = re.search(
            r"\b(?:reported|responded|said|consider(?:s|ed)?)\s+(?P<target>[^?.,;]+)|"
            r"(?:אמרו|דיווחו|שקלו)\s+(?P<target_he>[^?.,;]+)",
            request,
            re.IGNORECASE,
        )
        if match:
            return bounded_target(
                match.group("target") or match.group("target_he") or ""
            )
        return ""
    if predicate != "performed_by":
        return ""
    match = re.search(
        r"\b(?:handles?|manages?|performs?)\s+(?P<target>[^?.,;]+)|"
        r"(?:מטפל(?:ת|ים|ות)?\s+ב|אחראי(?:ת|ים|ות)?\s+על)"
        r"(?P<target_he>[^?.,;]+)",
        request,
        re.IGNORECASE,
    )
    if not match:
        return ""
    return bounded_target(
        match.group("target") or match.group("target_he") or ""
    )


def _constraint(
    dimension: str,
    *,
    value: str = "",
    state: RequirementConstraintState = RequirementConstraintState.NOT_REQUIRED,
    origin: RequirementConstraintOrigin = RequirementConstraintOrigin.UNKNOWN,
    request_span: str,
    rule: str,
    justification: str,
) -> RequirementDimensionConstraint:
    return RequirementDimensionConstraint(
        dimension=dimension,
        state=state,
        value=value,
        origin=origin,
        request_span=request_span,
        rule=rule,
        justification=justification,
    )


def _compiled_requirement_constraints(
    requirement: TypedRequirement,
    clause: CanonicalClause,
    document_reference=None,
) -> tuple[RequirementDimensionConstraint, ...]:
    """Compile all source-admissibility dimensions exactly once.

    The requirement graph can still represent operator and discourse needs,
    but those needs are never silently promoted into SOURCE_FACT equality.
    """
    span = clause.text
    source_fact = requirement.satisfaction_mode is RequirementSatisfactionMode.SOURCE_FACT
    numeric = source_fact and (
        requirement.value_type in {"numeric", "currency", "period", "count", "percentage"}
        or requirement.kind in {
            "count", "percentage", "comparison_endpoint",
            "category_value_bindings", "winner_attribute", "filtered_row_value",
        }
    )
    quantity_kind = normalize_quantity_kind(requirement.quantity_kind)
    statistical_role = normalize_statistical_role(requirement.statistical_role)
    aggregation_role = normalize_aggregation_role(requirement.aggregation_role)
    metric = " ".join(
        item for item in requirement.objects_or_metrics
        if item not in {"count", "percentage", "number", "total"}
    )
    metric = _source_language_metric_value(metric, clause)
    explicit_periods = tuple(
        period for period in requirement.periods
        if _PERIOD_RE.fullmatch(normalize_semantic_text(period))
    )
    discourse_period = (
        "current" if "current" in clause.qualifiers
        else "historical" if "historical" in clause.qualifiers else ""
    )
    explicit_total = bool(_EXPLICIT_TOTAL_REQUEST_RE.search(span))
    # C23's only audit-proven ownership transfer is an explicit TOTAL
    # comparison endpoint.  It is deliberately not a generic subject
    # de-constraint for changes, ordinary lookups, or other comparisons.
    eligible_total_comparison = (
        source_fact
        and requirement.kind == "comparison_endpoint"
        and clause.semantic_relation == "compare"
        and explicit_total
    )
    explicit_facet = str(requirement.facet or "")
    source_predicate = _explicit_source_predicate(requirement, clause)
    target = _independent_entity_value(
        str(requirement.counted_entity or requirement.target_entity_type or "")
    )
    event_entity, event_metric = _event_entity_and_metric(clause)
    if not target:
        target = event_entity
    if not metric:
        metric = event_metric
    if not target and eligible_total_comparison:
        target = _metric_attached_target(clause, metric)
    # A comparison endpoint with no percentage/count request still requires a
    # numeric source estimate.  NUMBER here is a strict source kind, not a
    # COUNT/percentage wildcard and never rewrites source quantity semantics.
    if (
        numeric
        and quantity_kind == QuantityKind.UNSPECIFIED.value
        and requirement.kind == "comparison_endpoint"
    ):
        quantity_kind = QuantityKind.NUMBER.value
    if (
        numeric
        and aggregation_role in {
            AggregationRole.NONE.value, AggregationRole.UNSPECIFIED.value,
        }
        and eligible_total_comparison
    ):
        aggregation_role = AggregationRole.TOTAL.value
    relation_arguments = _explicit_relation_target(
        clause, source_predicate, str(requirement.relation_target or ""),
    )
    if source_predicate == "performed_by" and relation_arguments:
        # A responsibility target is a relation argument, not a measured
        # scalar metric.  Keeping it in both dimensions would force an
        # English/normalized metric equality on a source-local Hebrew task.
        metric = ""
    if (
        target
        and explicit_facet
        and normalize_semantic_text(target).casefold()
        == normalize_semantic_text(explicit_facet).casefold()
    ):
        target = ""
    analysis_clause = replace(clause,text=_request_clause_text(clause.text))
    if clause.semantic_relation == "purpose" and source_predicate == "purpose_of" and target:
        # The explicit purpose operator and its independently required target
        # own this leading question grammar. It is not a source actor.
        purpose_subject = re.sub(
            r"^(?:(?:the\s+)?(?:purpose|objective|goal|aim)\s+(?:of\s+)?|"
            r"(?:מטרה|מטרת)\s+)", "", analysis_clause.subject, flags=re.I)
        analysis_clause = replace(analysis_clause,subject=purpose_subject)
    subject = _minimal_requirement_subject(
        analysis_clause,
        metric=metric,
        target_entity=target,
        relation_arguments=relation_arguments,
        population=str(requirement.scope_population or "|".join(requirement.populations)),
        facet=explicit_facet,
    )
    residual_subject = _lossless_subject_residual(
        analysis_clause,
        metric=metric,
        target_entity=target,
        relation_arguments=relation_arguments,
        population=str(requirement.scope_population or "|".join(requirement.populations)),
        facet=explicit_facet,
        source_predicate=source_predicate,
        aggregation_role=aggregation_role,
        allow_total_comparison_deconstraint=eligible_total_comparison,
    )
    if source_predicate in DOCUMENT_ROLE_PREDICATES and relation_arguments:
        # The complete named target belongs to its relation argument. The
        # interrogative actor/verb contributes no additional subject equality.
        subject = residual_subject = ""
    # C21's bounded subject remains authoritative when present.  The residual
    # path exists only for the failure mode established by the audit: a
    # meaningful request concept was otherwise left with no canonical owner.
    if (
        clause.semantic_relation not in {"argmax", "argmin"}
        and residual_subject
        and not subject
    ):
        subject = residual_subject
    from document_query import requested_item_identifier, requested_source_locations, technical_attribute_query
    item_identifier = requested_item_identifier(span) if numeric and requirement.kind == "count" else ""
    source_locations = requested_source_locations(span) if item_identifier else ()
    attribute_label, attribute_context = technical_attribute_query(span) if not numeric and requirement.kind == "lookup" else ("", "")
    from document_query import requested_list_scope, requested_organization_owner
    list_heading, list_topic, list_order = requested_list_scope(span) if requirement.kind == "list" else ("", "", "")
    list_owner = requested_organization_owner(span, document_reference=document_reference) if list_heading else ""
    organization_request = bool(list_heading and re.search(
        r"\b(?:organized|organised|structured|arranged)\b|"
        r"(?<!\w)(?:מאורגן|מאורגנת|מאורגנים|מאורגנות)(?!\w)",
        _request_clause_text(span), re.I,
    ))
    if item_identifier:
        subject = target = item_identifier
        metric = ""
    elif attribute_label:
        subject = attribute_label
        target = ""
        metric = ""
        source_predicate = "has_attribute"
    constraints = (
        _constraint(
            "value", value="required" if numeric else "",
            state=(RequirementConstraintState.REQUIRED if numeric else RequirementConstraintState.NOT_REQUIRED),
            origin=(RequirementConstraintOrigin.OUTPUT_TYPE_REQUIRED if numeric else RequirementConstraintOrigin.UNKNOWN),
            request_span=span, rule="source_fact_output_value",
            justification="the requested output requires a bound source scalar" if numeric else "the requirement is not a direct source scalar",
        ),
        _constraint(
            "quantity_kind", value=(
                QuantityKind.PERIOD.value
                if requirement.value_type == "period" else quantity_kind
            ) if numeric and (
                requirement.value_type == "period" or quantity_kind != QuantityKind.UNSPECIFIED.value
            ) else "",
            state=(RequirementConstraintState.REQUIRED if numeric and (
                requirement.value_type == "period" or quantity_kind != QuantityKind.UNSPECIFIED.value
            ) else RequirementConstraintState.NOT_REQUIRED),
            origin=(RequirementConstraintOrigin.OUTPUT_TYPE_REQUIRED if numeric else RequirementConstraintOrigin.UNKNOWN),
            request_span=span, rule="requested_scalar_output_kind",
            justification="the request explicitly asks for this output kind" if numeric else "no source quantity kind is requested",
        ),
        _constraint(
            "statistical_role", value=statistical_role if (
                numeric and statistical_role != StatisticalRole.UNSPECIFIED.value
            ) else "",
            state=(RequirementConstraintState.REQUIRED if numeric and statistical_role != StatisticalRole.UNSPECIFIED.value else RequirementConstraintState.NOT_REQUIRED),
            origin=(RequirementConstraintOrigin.EXPLICIT_REQUEST_SEMANTIC if numeric and statistical_role != StatisticalRole.UNSPECIFIED.value else RequirementConstraintOrigin.UNKNOWN),
            request_span=span, rule="explicit_statistical_role",
            justification="a request-local statistical cue specifies this role" if numeric and statistical_role != StatisticalRole.UNSPECIFIED.value else "no source statistical role is requested",
        ),
        _constraint(
            "aggregation_role", value=aggregation_role if (
                numeric and aggregation_role not in {AggregationRole.NONE.value, AggregationRole.UNSPECIFIED.value}
                and (aggregation_role != AggregationRole.TOTAL.value or explicit_total)
            ) else "",
            state=(RequirementConstraintState.REQUIRED if numeric and aggregation_role not in {AggregationRole.NONE.value, AggregationRole.UNSPECIFIED.value} and (aggregation_role != AggregationRole.TOTAL.value or explicit_total) else RequirementConstraintState.NOT_REQUIRED),
            origin=(RequirementConstraintOrigin.EXPLICIT_REQUEST_SEMANTIC if numeric and aggregation_role not in {AggregationRole.NONE.value, AggregationRole.UNSPECIFIED.value} and (aggregation_role != AggregationRole.TOTAL.value or explicit_total) else RequirementConstraintOrigin.UNKNOWN),
            request_span=span, rule="explicit_aggregation_cue",
            justification="a request-local aggregation cue specifies this role" if numeric and aggregation_role not in {AggregationRole.NONE.value, AggregationRole.UNSPECIFIED.value} and (aggregation_role != AggregationRole.TOTAL.value or explicit_total) else "no source aggregation role is requested",
        ),
        _constraint(
            "subject", value=subject,
            state=(RequirementConstraintState.REQUIRED if subject else RequirementConstraintState.NOT_REQUIRED),
            origin=(RequirementConstraintOrigin.EXPLICIT_REQUEST_SEMANTIC if subject else RequirementConstraintOrigin.UNKNOWN),
            request_span=span, rule="minimal_semantic_subject",
            justification="bounded request concept" if subject else "no bounded semantic subject was proven",
        ),
        _constraint(
            "metric", value=metric,
            state=(RequirementConstraintState.REQUIRED if metric else RequirementConstraintState.NOT_REQUIRED),
            origin=(RequirementConstraintOrigin.EXPLICIT_REQUEST_SEMANTIC if metric else RequirementConstraintOrigin.UNKNOWN),
            request_span=span, rule="explicit_metric_slot",
            justification="request metric extracted independently of answer shape" if metric else "no concrete metric is requested",
        ),
        _constraint(
            "predicate", value=source_predicate if source_fact and source_predicate else "",
            state=(RequirementConstraintState.REQUIRED if source_fact and source_predicate else RequirementConstraintState.NOT_REQUIRED),
            origin=(RequirementConstraintOrigin.EXPLICIT_REQUEST_SEMANTIC if source_fact and source_predicate else RequirementConstraintOrigin.UNKNOWN),
            request_span=span, rule="explicit_source_relation",
            justification="the request names this source relation" if source_fact and source_predicate else "operator or output predicate does not constrain a source fact",
        ),
        _constraint(
            "period", value="|".join(explicit_periods),
            state=(RequirementConstraintState.REQUIRED if explicit_periods else RequirementConstraintState.SCOPE_ONLY if discourse_period else RequirementConstraintState.NOT_REQUIRED),
            origin=(RequirementConstraintOrigin.EXPLICIT_REQUEST_SEMANTIC if explicit_periods else RequirementConstraintOrigin.DISCOURSE_SCOPE if discourse_period else RequirementConstraintOrigin.UNKNOWN),
            request_span=span, rule="explicit_fact_period_or_discourse_scope",
            justification="explicit requested fact period" if explicit_periods else "current/previous wording selects discourse scope only" if discourse_period else "no fact-period equality is requested",
        ),
        _constraint(
            "population", value=(requirement.scope_population or "|".join(requirement.populations)),
            state=(RequirementConstraintState.REQUIRED if (requirement.scope_population or requirement.populations) else RequirementConstraintState.NOT_REQUIRED),
            origin=(RequirementConstraintOrigin.EXPLICIT_REQUEST_SEMANTIC if (requirement.scope_population or requirement.populations) else RequirementConstraintOrigin.UNKNOWN),
            request_span=span, rule="explicit_population_scope",
            justification="request population narrows the answer" if (requirement.scope_population or requirement.populations) else "no population equality is requested",
        ),
        _constraint(
            "facet", value=explicit_facet,
            state=(RequirementConstraintState.REQUIRED if explicit_facet else RequirementConstraintState.NOT_REQUIRED),
            origin=(RequirementConstraintOrigin.EXPLICIT_REQUEST_SEMANTIC if explicit_facet else RequirementConstraintOrigin.UNKNOWN),
            request_span=span, rule="explicit_requested_facet",
            justification="request names this category facet" if explicit_facet else "no source facet is requested",
        ),
        _constraint(
            "unit", value="|".join(requirement.units) if numeric and requirement.units else "",
            state=(RequirementConstraintState.REQUIRED if numeric and requirement.units else RequirementConstraintState.NOT_REQUIRED),
            origin=(RequirementConstraintOrigin.OUTPUT_TYPE_REQUIRED if numeric and requirement.units else RequirementConstraintOrigin.UNKNOWN),
            request_span=span, rule="explicit_requested_unit",
            justification="requested scalar output carries this unit" if numeric and requirement.units else "no unit equality is requested",
        ),
        _constraint(
            "polarity", value=requirement.polarity if requirement.polarity != "neutral" else "",
            state=(RequirementConstraintState.REQUIRED if requirement.polarity != "neutral" else RequirementConstraintState.NOT_REQUIRED),
            origin=(RequirementConstraintOrigin.EXPLICIT_REQUEST_SEMANTIC if requirement.polarity != "neutral" else RequirementConstraintOrigin.UNKNOWN),
            request_span=span, rule="explicit_request_polarity",
            justification="request explicitly requires positive or negative relation" if requirement.polarity != "neutral" else "neutrality does not constrain source polarity",
        ),
        _constraint(
            "target_entity", value=target,
            state=(RequirementConstraintState.REQUIRED if target else RequirementConstraintState.NOT_REQUIRED),
            origin=(RequirementConstraintOrigin.EXPLICIT_REQUEST_SEMANTIC if target else RequirementConstraintOrigin.UNKNOWN),
            request_span=span, rule="explicit_target_entity",
            justification="request names the counted or target entity" if target else "no target entity equality is requested",
        ),
        _constraint(
            "relation_arguments", value=relation_arguments,
            state=(RequirementConstraintState.REQUIRED if relation_arguments else RequirementConstraintState.NOT_REQUIRED),
            origin=(RequirementConstraintOrigin.EXPLICIT_REQUEST_SEMANTIC if relation_arguments else RequirementConstraintOrigin.UNKNOWN),
            request_span=span, rule="explicit_relation_target",
            justification="request names a relation argument" if relation_arguments else "no relation argument equality is requested",
        ),
        _constraint(
            "fact_role", value="measured_outcome" if numeric else "",
            state=(RequirementConstraintState.REQUIRED if numeric else RequirementConstraintState.NOT_REQUIRED),
            origin=(RequirementConstraintOrigin.OUTPUT_TYPE_REQUIRED if numeric else RequirementConstraintOrigin.UNKNOWN),
            request_span=span, rule="source_scalar_is_measured_outcome",
            justification="answer scalar cannot be supplied by a filter or exclusion condition" if numeric else "no source scalar role is requested",
        ),
        _constraint(
            "structural_type", value="required" if source_fact else "",
            state=(RequirementConstraintState.REQUIRED if source_fact else RequirementConstraintState.NOT_REQUIRED),
            origin=(RequirementConstraintOrigin.DERIVED_BUT_PROVEN if source_fact else RequirementConstraintOrigin.UNKNOWN),
            request_span=span, rule="source_fact_structure",
            justification="a source fact needs a bound structural source" if source_fact else "not a direct source-fact requirement",
        ),
        _constraint(
            "proposition_completeness", value="required" if source_fact else "",
            state=(RequirementConstraintState.REQUIRED if source_fact else RequirementConstraintState.NOT_REQUIRED),
            origin=(RequirementConstraintOrigin.DERIVED_BUT_PROVEN if source_fact else RequirementConstraintOrigin.UNKNOWN),
            request_span=span, rule="complete_source_proposition",
            justification="a source fact must be proposition-complete" if source_fact else "not a direct source-fact requirement",
        ),
    )
    from financial_semantics import financial_request
    financial = financial_request(span) if numeric and requirement.kind == "lookup" else None
    if financial is not None:
        # A named invoice/quotation is document scope, not the local amount's
        # subject. The exact row role and every document qualifier now have
        # separate mandatory owners and are checked again at final acceptance.
        transferred = []
        for constraint in constraints:
            if constraint.dimension in {"subject", "metric"}:
                transferred.append(replace(
                    constraint, state=RequirementConstraintState.SCOPE_ONLY,
                    hard_constraint_owner="financial_role+document_scope",
                    enforcement_path="financial_role+document_scope",
                    enforcement_equivalence_status=EnforcementEquivalenceStatus.DISCRIMINATIVE_EQUIVALENCE_PROVEN,
                    rule="monetary_role_and_document_scope",
                    justification="source row role and exact named document scope independently preserve these request distinctions",
                ))
            elif constraint.dimension == "unit" and financial.currency:
                transferred.append(_constraint(
                    "unit", value=financial.currency, state=RequirementConstraintState.REQUIRED,
                    origin=RequirementConstraintOrigin.EXPLICIT_REQUEST_SEMANTIC,
                    request_span=span, rule="explicit_financial_currency", justification="request explicitly names this currency",
                ))
            else:
                transferred.append(constraint)
        constraints = tuple(transferred)
    if list_heading and (not organization_request or list_owner):
        owners = "+".join(name for name, value in (("ordered_list_heading", list_heading), ("ordered_list_topic", list_topic), ("ordered_list_owner", list_owner), ("ordered_list_order", list_order)) if value)
        constraints = tuple(replace(c, state=RequirementConstraintState.SCOPE_ONLY,
            hard_constraint_owner=owners, enforcement_path=owners,
            enforcement_equivalence_status=EnforcementEquivalenceStatus.DISCRIMINATIVE_EQUIVALENCE_PROVEN,
            rule="source_owned_list_scope", justification="source list collection and any explicit organization actor, topic, cardinality and sequence independently retain the request distinctions")
            if c.dimension == "subject" else c for c in constraints)
    return constraints + tuple(
        _constraint(
            dimension, value=value, state=(RequirementConstraintState.REQUIRED if value else RequirementConstraintState.NOT_REQUIRED),
            origin=(RequirementConstraintOrigin.EXPLICIT_REQUEST_SEMANTIC if value else RequirementConstraintOrigin.UNKNOWN),
            request_span=span, rule=rule, justification=justification,
        )
        for dimension, value, rule, justification in (
            ("financial_role", financial.role if financial else "", "explicit_monetary_row_role", "requested payable/subtotal/tax role must be present in the bounded source row"),
            ("document_scope", financial.scope_json() if financial else document_reference.encoded() if document_reference is not None and (document_reference.kind != "presentation" or list_owner) else "", "explicit_named_document_scope", "the explicit named source must match identity-bound governing document metadata"),
            ("source_location", "|".join(source_locations), "explicit_source_location_outputs", "requested worksheet and row outputs require independently proven source-owned field coordinates"),
            ("attribute_context", attribute_context, "explicit_attribute_context", "requested governing context must be present in the same source-owned heading or proposition"),
            ("ordered_list_heading", list_heading, "explicit_list_heading", "requested list identity must match its exact source-owned heading"),
            ("ordered_list_topic", list_topic, "explicit_list_topic", "requested topic must be present in same-source governing list context"),
            ("ordered_list_owner", list_owner, "explicit_organization_owner", "requested organization actor must be independently proved in the same source-owned list introduction"),
            ("ordered_list_order", list_order, "explicit_source_sequence", "requested order must equal proven original source ordinals"),
        )
    )


def _compiled_semantic_accounting(
    requirement: TypedRequirement,
    clause: CanonicalClause,
    constraints: tuple[RequirementDimensionConstraint, ...],
) -> tuple[RequirementSemanticAccounting, ...]:
    """Account for every compiler-owned distinction exactly once.

    Ordinary hard constraints are self-enforced.  The one C23 de-constraint
    shape is deliberately narrow: an explicitly TOTAL comparison endpoint may
    release a residual subject only when target, metric, endpoint period,
    aggregation, value, and same-proposition structure are all canonical
    hard constraints.  This is the structural proof, not a lexical shortcut.
    """
    accounting = [
        RequirementSemanticAccounting(
            semantic_distinction=constraint.value,
            semantic_category=_SEMANTIC_CATEGORY_BY_DIMENSION.get(
                constraint.dimension, "OTHER",
            ),
            request_span=constraint.request_span,
            semantic_owner=constraint.semantic_owner,
            hard_constraint_owner=constraint.hard_constraint_owner,
            enforcement_path=constraint.enforcement_path,
            enforcement_equivalence_status=(
                constraint.enforcement_equivalence_status
            ),
            independently_meaningful=True,
            independently_discriminative=True,
            proof=constraint.justification,
        )
        for constraint in constraints
        if constraint.state is RequirementConstraintState.REQUIRED
    ]
    by_dimension = {item.dimension: item for item in constraints}
    required = {
        dimension for dimension, item in by_dimension.items()
        if item.state is RequirementConstraintState.REQUIRED
    }
    subject = by_dimension["subject"]
    if {"financial_role", "document_scope"} <= required:
        for dimension in ("subject", "metric"):
            transferred = by_dimension[dimension]
            if transferred.state is RequirementConstraintState.SCOPE_ONLY and transferred.value:
                accounting.append(RequirementSemanticAccounting(
                    semantic_distinction=transferred.value,
                    semantic_category=_SEMANTIC_CATEGORY_BY_DIMENSION[dimension],
                    request_span=clause.text, semantic_owner=dimension,
                    hard_constraint_owner="financial_role+document_scope",
                    enforcement_path="financial_role+document_scope",
                    enforcement_equivalence_status=EnforcementEquivalenceStatus.DISCRIMINATIVE_EQUIVALENCE_PROVEN,
                    independently_meaningful=True, independently_discriminative=False,
                    proof="local monetary role remains required; exact document kind/number and all residual named qualifiers remain required against proven source metadata",
                ))
    if by_dimension.get("ordered_list_heading") and by_dimension["ordered_list_heading"].state is RequirementConstraintState.REQUIRED and subject.state is RequirementConstraintState.SCOPE_ONLY:
        accounting.append(RequirementSemanticAccounting(
            semantic_distinction=subject.value, semantic_category="CORE_SUBJECT_IDENTITY", request_span=clause.text,
            semantic_owner="subject", hard_constraint_owner=subject.hard_constraint_owner,
            enforcement_path=subject.enforcement_path, enforcement_equivalence_status=EnforcementEquivalenceStatus.DISCRIMINATIVE_EQUIVALENCE_PROVEN,
            independently_meaningful=True, independently_discriminative=False,
            proof="explicit list collection and any requested organization owner and topic independently preserve identity; cardinality and source ordinals preserve membership and sequence",
        ))
    equivalent_owner_dimensions = {
        "metric", "target_entity", "aggregation_role", "period", "value",
        "fact_role", "structural_type", "proposition_completeness",
    }
    if (
        subject.state is RequirementConstraintState.NOT_REQUIRED
        and requirement.kind == "comparison_endpoint"
        and clause.semantic_relation == "compare"
        and bool(_EXPLICIT_TOTAL_REQUEST_RE.search(clause.text))
        and equivalent_owner_dimensions <= required
    ):
        path = "+".join(sorted(equivalent_owner_dimensions))
        accounting.append(RequirementSemanticAccounting(
            semantic_distinction=normalize_semantic_text(clause.subject),
            semantic_category="REDUNDANT_SUBJECT_MODIFIER",
            request_span=clause.text,
            semantic_owner="metric+target_entity+aggregation_role+period",
            hard_constraint_owner="",
            enforcement_path=path,
            enforcement_equivalence_status=(
                EnforcementEquivalenceStatus.DISCRIMINATIVE_EQUIVALENCE_PROVEN
            ),
            independently_meaningful=True,
            independently_discriminative=False,
            proof=(
                "explicit TOTAL comparison residual is fully represented by "
                "metric, target entity, aggregation, endpoint period, bound "
                "value, and same-proposition source structure"
            ),
        ))
    return tuple(accounting)


def _requirements_for_clause(clause: CanonicalClause, document_reference=None) -> tuple[TypedRequirement, ...]:
    """Return the frozen requirement graph with C20 semantic provenance."""
    compiled: list[TypedRequirement] = []
    for item in _requirements_for_clause_raw(clause):
        constraints = _compiled_requirement_constraints(item, clause, document_reference)
        compiled.append(replace(
            item,
            dimension_constraints=constraints,
            semantic_accounting=_compiled_semantic_accounting(
                item, clause, constraints,
            ),
        ))
    return tuple(compiled)


_PARALLEL_EXPERIENCE_SCOPE_PREFIX_RE = re.compile(
    r"^\s*(?:פרט(?:י|ו)?\s+בנפרד\s+עבור|"
    r"(?:(?:report|show|list|specify|describe|break\s+down)\s+)?"
    r"separately\s+for)\s+",
    re.IGNORECASE,
)


def _link_parallel_experience_scope_clause(
    clause_texts: Sequence[str],
    *,
    route_hint: str,
    default_cardinality: int,
    relation_override: str = "",
) -> tuple[str, ...]:
    """Attach an explicit multi-band breakdown to its temporal comparison."""
    if len(clause_texts) != 2:
        return tuple(clause_texts)
    primary_text, scope_text = clause_texts
    if not _PARALLEL_EXPERIENCE_SCOPE_PREFIX_RE.match(scope_text):
        return tuple(clause_texts)
    bands = experience_band_facets(scope_text)
    if len(set(bands)) < 2 or any(
        not facet.startswith("experience_band:")
        or len(facet.split(":")) not in {4, 5}
        for facet in bands
    ):
        return tuple(clause_texts)
    primary = _clause(
        primary_text, 1, route_hint, default_cardinality, relation_override,
    )
    if primary.semantic_relation != "change_over_time" or len(primary.periods) != 2:
        return tuple(clause_texts)
    if not any(
        name == "experience_band" and pattern.search(primary_text)
        for name, pattern in _FACET_PATTERNS
    ):
        return tuple(clause_texts)
    # Keep the original scope wording in the canonical clause. The requirement
    # builder expands its explicit facets across the two inherited periods.
    return (f"{primary_text} {scope_text}",)


def _semantic_draft(
    query: str,
    *,
    plan: Any = None,
) -> SemanticContractDraft:
    normalized = normalize_semantic_text(query)
    route_hint = _enum(getattr(plan, "request_operator", ""))
    relation_override = (
        str(getattr(plan, "resolved_semantic_relation", "") or "").casefold()
        if getattr(plan, "semantic_resolution_authoritative", False)
        else ""
    )
    default_cardinality = max(1, int(getattr(plan, "requested_items", 1) or 1))
    clause_texts = _link_parallel_experience_scope_clause(
        _split_clauses(normalized),
        route_hint=route_hint,
        default_cardinality=default_cardinality,
        relation_override=relation_override,
    )
    draft_clauses = tuple(
        _clause(
            text,
            index,
            route_hint,
            default_cardinality,
            relation_override,
        )
        for index, text in enumerate(clause_texts, start=1)
    )
    return SemanticContractDraft(
        normalized_request=normalized,
        draft_operation=(
            draft_clauses[0].semantic_relation
            if len({item.semantic_relation for item in draft_clauses}) == 1
            else "compound"
        ),
        clauses=list(draft_clauses),
        endpoint_hints=_unique(getattr(plan, "entities", ()) or ()),
    )


def _output_slot_violations(
    clauses: Sequence[CanonicalClause],
    requirements: Sequence[TypedRequirement],
) -> tuple[str, ...]:
    violations: list[str] = []
    for clause in clauses:
        requested = set(clause.requested_output_slots)
        answer_requirements = [
            item for item in requirements
            if item.answer_producing and item.clause_id == clause.id
        ]
        produced = {item.output_slot for item in answer_requirements if item.output_slot}
        if requested - produced:
            violations.append("requested_output_slot_missing_requirement")
        if produced - requested:
            violations.append("unexpected_answer_output_requirement")
        counts: dict[str, int] = {}
        for requirement in answer_requirements:
            counts[requirement.output_slot] = counts.get(requirement.output_slot, 0) + 1
        # Endpoint values are explicitly a many-to-one structural graph: each
        # endpoint/metric value independently satisfies the one requested slot.
        if any(
            count > 1 and slot not in {"endpoint_values", "direction"}
            for slot, count in counts.items()
        ):
            violations.append("duplicate_answer_output_requirement")
    return _unique(violations)


def _semantic_contradictions(
    clauses: Sequence[CanonicalClause],
) -> tuple[str, ...]:
    violations: list[str] = []
    significance = re.compile(
        r"\b(?:significant(?:ly)?|statistical(?:ly)?\s+significant|significance)\b|"
        r"(?:מובהק|מובהקות)",
        re.IGNORECASE,
    )
    for clause in clauses:
        relation = clause.semantic_relation
        slots = set(clause.requested_output_slots)
        if _explicit_scope_marker(clause.text) and not clause.scope_population:
            violations.append("explicit_scope_population_unresolved")
        if clause.scope_population and "industry_sector" in clause.objects_or_metrics:
            violations.append("scope_population_metric_role_conflict")
        if (
            "scalar" in slots
            and relation in {"lookup", "summary", "filter_rows"}
            and
            _scalar_value_type(clause) == "text_proposition"
            and re.search(
                r"\b(?:estimate|amount|value|total|lower\s+bound|upper\s+bound|demand)\b|"
                r"(?:אומדן|הערכה|ערך|סך\s*הכול|חסם|גבול|ביקוש)",
                clause.text,
                re.IGNORECASE,
            )
        ):
            violations.append("numeric_scalar_compiled_as_text_proposition")
        if clause.polarity == "negative" and relation == "count":
            violations.append("negative_relation_compiled_as_count")
        if significance.search(clause.text) and relation not in {"compare", "change_over_time"}:
            violations.append("significance_comparison_compiled_as_lookup")
        if (
            len(clause.periods) >= 2
            and relation not in {"compare", "change_over_time", "filter_rows"}
            and clause.embedded_event is None
        ):
            violations.append("temporal_endpoints_without_comparison_relation")
        if relation not in {"argmax", "argmin"} and "count" in slots and not clause.counted_entity and not any(
            item in {"count", "employee_count", "employee_recruitment"}
            for item in clause.objects_or_metrics
        ):
            violations.append("count_output_without_counted_entity")
        if relation in {"exclusion", "exception"} and "count" in slots:
            violations.append("negative_entity_compiled_with_count_output")
        if relation == "cause" and "text_proposition" not in slots:
            violations.append("cause_missing_reason_output")
        if relation in {"compare", "change_over_time"} and (
            not clause.comparison_subject or not clause.comparison_reference
        ):
            violations.append("comparative_reference_unresolved")
        if relation in {"compare", "change_over_time"} and any(
            re.fullmatch(
                r"(?:does|do|did|is|are|was|were|what|which|who|why|how|ai|"
                r"האם|מה|מי|למה|מדוע|איך)",
                endpoint.value,
                re.IGNORECASE,
            )
            for endpoint in clause.comparison_endpoints
        ):
            violations.append("comparison_endpoint_not_semantic_argument")
        if relation in {"purpose", "definition", "cause", "explanation", "procedure"} and not clause.relation_target:
            violations.append("relation_target_unresolved")
    return _unique(violations)


_RELATION_EXECUTION_ROUTES = {
    "argmax": "ARGMAX",
    "argmin": "ARGMIN",
    "cause": "CAUSE",
    "change_over_time": "CHANGE_OVER_TIME",
    "compare": "COMPARE",
    "count": "COUNT",
    "definition": "DEFINITION",
    "exception": "EXCEPTION",
    "exclusion": "EXCLUSION",
    "explanation": "EXPLANATION",
    "filter_rows": "FILTER",
    "inclusion": "COUNT",
    "list": "LIST",
    "procedure": "PROCEDURE",
    "purpose": "DIRECT_LOOKUP",
    "summary": "SUMMARY",
    "lookup": "DIRECT_LOOKUP",
}
_CONVERSATION_EXECUTION_ROUTES = {
    "ANOTHER_RESULT": "ANOTHER_RESULT",
    "VERIFY_PREVIOUS": "VERIFY_PREVIOUS",
    "CORRECT_PREVIOUS": "CORRECT_PREVIOUS",
    "EXPAND_PREVIOUS": "EXPAND_PREVIOUS",
    "CONSTRAIN_TO_SOURCE": "SOURCE_SCOPED_QUERY",
}


def _planner_draft_disagreements(
    plan: Any,
    clauses: Sequence[CanonicalClause],
) -> tuple[str, ...]:
    """Return only material conflicts between independent pre-freeze signals."""
    if plan is None or getattr(plan, "semantic_resolution_authoritative", False):
        return ()
    conversation_operation = _enum(
        getattr(plan, "conversation_operation", "NONE")
    )
    if conversation_operation not in {"", "NONE"}:
        return ()
    planned = _enum(getattr(plan, "request_operator", ""))
    if not planned or planned in {"DOCUMENT_QUIZ", "SOURCE_SCOPED_QUERY"}:
        return ()
    relations = _unique(item.semantic_relation for item in clauses)
    if len(relations) != 1:
        return ()
    relation = relations[0]
    deterministic_route = _RELATION_EXECUTION_ROUTES.get(
        relation, "DIRECT_LOOKUP"
    )
    compatible_routes = {
        frozenset(("COMPARE", "CHANGE_OVER_TIME")),
        frozenset(("LIST", "LIST_WITH_ATTRIBUTES")),
        # A scalar lookup and a scalar count/inclusion can be slot-equivalent;
        # the final requirement graph, not the enum spelling, decides.
        frozenset(("DIRECT_LOOKUP", "COUNT")),
    }
    operation_compatible = (
        planned == deterministic_route
        or frozenset((planned, deterministic_route)) in compatible_routes
        or relation == "purpose" and planned == "DIRECT_LOOKUP"
    )
    disagreements: list[str] = []
    if not operation_compatible:
        disagreements.append("planner_draft_operation_disagreement")

    known_facets = {name for name, _pattern in _FACET_PATTERNS}
    planner_facets = {
        str(item).casefold()
        for item in getattr(plan, "qualifiers", ()) or ()
        if str(item).casefold() in known_facets
    }
    draft_facets = {
        str(item).casefold()
        for clause in clauses for item in clause.requested_facets
    }
    if planner_facets and draft_facets and not planner_facets & draft_facets:
        disagreements.append("planner_draft_facet_disagreement")
    return tuple(disagreements)


def _execution_route(
    semantic_relation: str,
    conversation_operation: str,
    plan: Any,
) -> str:
    """Resolve the one route written into the immutable contract."""
    if conversation_operation in _CONVERSATION_EXECUTION_ROUTES:
        return _CONVERSATION_EXECUTION_ROUTES[conversation_operation]
    planned = _enum(getattr(plan, "request_operator", ""))
    if planned in {"DOCUMENT_QUIZ", "SOURCE_SCOPED_QUERY"}:
        return planned
    if semantic_relation == "compound":
        relations = {
            item.semantic_relation
            for item in getattr(plan, "clauses", ()) or ()
        }
        if relations == {"compare"}:
            return "COMPARE"
        return "DIRECT_LOOKUP"
    return _RELATION_EXECUTION_ROUTES.get(semantic_relation, "DIRECT_LOOKUP")


def _build_request_execution_contract_once(
    query: str,
    *,
    plan: Any = None,
    source_scope: Iterable[str] = (),
) -> RequestExecutionContract:
    from document_reading import compile_reading_contract
    reading_contract = compile_reading_contract(query, plan, source_scope)
    if reading_contract is not None:
        return reading_contract
    draft = _semantic_draft(query, plan=plan)
    normalized = draft.normalized_request
    default_cardinality = max(1, int(getattr(plan, "requested_items", 1) or 1))
    clauses, semantic_validation = _normalize_semantic_draft(draft)
    pre_requirement_violations = _semantic_contradictions(clauses)
    planner_disagreements = _planner_draft_disagreements(plan, clauses)
    if (
        not semantic_validation.valid
        or pre_requirement_violations
        or planner_disagreements
    ):
        result = SemanticCompilationResult(
            status="needs_model_resolution",
            draft=draft,
            violations=_unique((
                *semantic_validation.violations,
                *pre_requirement_violations,
                *planner_disagreements,
            )),
            reason_codes=((
                "pre_freeze_semantic_disagreement"
                if planner_disagreements else "semantic_slots_unresolved"
            ),),
            planner_operation=_enum(getattr(plan, "request_operator", "")),
            deterministic_operation=draft.draft_operation,
            deterministic_semantic_agreement=(not planner_disagreements),
            pre_freeze_disagreement=bool(planner_disagreements),
        )
        raise SemanticCompilationError(result)
    from document_query import named_document_reference
    document_reference = named_document_reference(normalized)
    raw_requirements = tuple(
        requirement for clause in clauses for requirement in _requirements_for_clause(clause, document_reference)
    )
    requirements_list: list[TypedRequirement] = []
    signatures: set[tuple[Any, ...]] = set()
    for requirement in raw_requirements:
        signature = (
            requirement.kind,
            normalize_semantic_text(requirement.subject).casefold(),
            requirement.predicate,
            requirement.objects_or_metrics,
            requirement.periods,
            requirement.populations,
            requirement.endpoint_type,
            normalize_semantic_text(requirement.endpoint_value).casefold(),
            requirement.facet,
            requirement.output_slot,
            requirement.quantity_kind,
            requirement.statistical_role,
            requirement.aggregation_role,
            normalize_semantic_text(requirement.relation_target).casefold(),
            requirement.filter_operator,
            requirement.filter_threshold,
            requirement.scope_operator,
            requirement.scope_threshold,
        )
        if signature in signatures:
            continue
        signatures.add(signature)
        requirements_list.append(requirement)
    requirements = tuple(requirements_list)
    slot_violations = _output_slot_violations(clauses, requirements)
    if slot_violations:
        raise SemanticCompilationError(SemanticCompilationResult(
            status="invalid",
            draft=draft,
            violations=_unique((*semantic_validation.violations, *slot_violations)),
            reason_codes=("requirement_graph_invariant_failed",),
        ))
    scope = list(source_scope)
    plan_source = getattr(plan, "source_id", None)
    if plan_source:
        scope.append(str(plan_source))
    scope_kind = _enum(getattr(plan, "scope", ""))
    if scope_kind:
        scope.insert(0, scope_kind)
    relations = _unique(clause.semantic_relation for clause in clauses)
    shapes = _unique(clause.answer_shape for clause in clauses)
    polarities = _unique(clause.polarity for clause in clauses)
    conversation_operation = (
        _enum(getattr(plan, "conversation_operation", "NONE")) or "NONE"
    )
    semantic_relation = relations[0] if len(relations) == 1 else "compound"
    payload = {
        "schema_version": "request-execution-contract-1.4.0",
        "resolved_user_request": normalized,
        "normalized_request": normalized,
        "conversation_operation": conversation_operation,
        "execution_route": _execution_route(
            semantic_relation, conversation_operation, plan
        ),
        "support_status": (
            "unsupported"
            if _enum(getattr(plan, "intent", ""))
            == "UNSUPPORTED_OR_CONVERSATIONAL" else "supported"
        ),
        "source_scope": list(_unique(scope)),
        "semantic_relation": semantic_relation,
        "answer_shape": shapes[0] if len(shapes) == 1 else "compound",
        "answer_scope": getattr(plan, "answer_scope", "targeted"),
        "subjects": list(_unique(clause.subject for clause in clauses)),
        "predicates": list(_unique(clause.predicate for clause in clauses)),
        "objects_or_metrics": list(_unique(item for clause in clauses for item in clause.objects_or_metrics)),
        "requested_values": list(_unique(item for clause in clauses for item in clause.requested_values)),
        "units": list(_unique(item for clause in clauses for item in clause.units)),
        "periods": list(_unique(item for clause in clauses for item in clause.periods)),
        "populations": list(_unique(item for clause in clauses for item in clause.populations)),
        "polarity": polarities[0] if len(polarities) == 1 else "mixed",
        "qualifiers": list(_unique(item for clause in clauses for item in clause.qualifiers)),
        "requested_cardinality": max((clause.requested_cardinality for clause in clauses), default=1),
        "completeness": "all_mandatory_requirements" if len(requirements) > 1 else clauses[0].completeness,
        "clauses": [item.as_dict() for item in clauses],
        "requirements": [item.as_dict() for item in requirements],
        "semantic_validation": semantic_validation.as_dict(),
        "target_entity_types": list(_unique(clause.target_entity_type for clause in clauses if clause.target_entity_type)),
        "counted_entities": list(_unique(clause.counted_entity for clause in clauses if clause.counted_entity)),
        "requested_attributes": list(_unique(clause.requested_attribute for clause in clauses if clause.requested_attribute)),
        "comparison_subjects": list(_unique(clause.comparison_subject for clause in clauses if clause.comparison_subject)),
        "comparison_references": list(_unique(clause.comparison_reference for clause in clauses if clause.comparison_reference)),
        "scope_populations": list(_unique(clause.scope_population for clause in clauses if clause.scope_population)),
        "answer_relations": list(_unique(
            clause.answer_relation or clause.semantic_relation for clause in clauses
        )),
        "embedded_events": [
            clause.embedded_event.as_dict()
            for clause in clauses if clause.embedded_event is not None
        ],
        "filter_conditions": [
            clause.filter_condition.as_dict()
            for clause in clauses if clause.filter_condition is not None
        ],
        "scope_constraints": [
            clause.scope_constraint.as_dict()
            for clause in clauses if clause.scope_constraint is not None
        ],
    }
    digest = hashlib.sha256(json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")).hexdigest()
    return RequestExecutionContract(
        contract_id="contract-" + digest[:24],
        contract_sha256=digest,
        schema_version=payload["schema_version"],
        resolved_user_request=normalized,
        normalized_request=normalized,
        conversation_operation=payload["conversation_operation"],
        execution_route=payload["execution_route"],
        support_status=payload["support_status"],
        source_scope=tuple(payload["source_scope"]),
        semantic_relation=payload["semantic_relation"],
        answer_shape=payload["answer_shape"],
        answer_scope=payload["answer_scope"],
        subjects=tuple(payload["subjects"]),
        predicates=tuple(payload["predicates"]),
        objects_or_metrics=tuple(payload["objects_or_metrics"]),
        requested_values=tuple(payload["requested_values"]),
        units=tuple(payload["units"]),
        periods=tuple(payload["periods"]),
        populations=tuple(payload["populations"]),
        polarity=payload["polarity"],
        qualifiers=tuple(payload["qualifiers"]),
        requested_cardinality=payload["requested_cardinality"],
        completeness=payload["completeness"],
        clauses=clauses,
        requirements=requirements,
        semantic_validation=semantic_validation,
        target_entity_types=tuple(payload["target_entity_types"]),
        counted_entities=tuple(payload["counted_entities"]),
        requested_attributes=tuple(payload["requested_attributes"]),
        comparison_subjects=tuple(payload["comparison_subjects"]),
        comparison_references=tuple(payload["comparison_references"]),
        scope_populations=tuple(payload["scope_populations"]),
        answer_relations=tuple(payload["answer_relations"]),
        embedded_events=tuple(
            clause.embedded_event
            for clause in clauses if clause.embedded_event is not None
        ),
        filter_conditions=tuple(
            clause.filter_condition
            for clause in clauses if clause.filter_condition is not None
        ),
        scope_constraints=tuple(
            clause.scope_constraint
            for clause in clauses if clause.scope_constraint is not None
        ),
    )


def compile_request_execution_contract(
    query: str,
    *,
    plan: Any = None,
    source_scope: Iterable[str] = (),
    model_resolver: Any = None,
) -> SemanticCompilationResult:
    """Compile once, optionally applying one bounded structured resolution."""
    try:
        contract = _build_request_execution_contract_once(
            query, plan=plan, source_scope=source_scope
        )
        return SemanticCompilationResult(
            status=(
                "repaired" if contract.semantic_validation.normalization_applied
                else "valid"
            ),
            draft=_semantic_draft(query, plan=plan),
            violations=contract.semantic_validation.violations,
            contract=contract,
            reason_codes=(
                ("deterministic_semantic_repair",)
                if contract.semantic_validation.normalization_applied else ()
            ),
            planner_operation=_enum(getattr(plan, "request_operator", "")),
            deterministic_operation=contract.semantic_relation,
            deterministic_semantic_agreement=(
                True if plan is not None else None
            ),
        )
    except SemanticCompilationError as exc:
        first = exc.result
    if model_resolver is None:
        return first
    try:
        resolution = model_resolver(first)
    except Exception:
        resolution = None
    if not isinstance(resolution, Mapping):
        return replace(
            first,
            status="needs_model_resolution",
            reason_codes=_unique((*first.reason_codes, "model_resolution_failed")),
            model_resolution_attempted=True,
        )
    resolved_query = normalize_semantic_text(resolution.get("resolved_request") or query)
    endpoint_hints = tuple(resolution.get("comparison_endpoints") or ())

    class _ResolutionPlan:
        semantic_resolution_authoritative = True
        resolved_semantic_relation = str(
            resolution.get("semantic_relation") or ""
        ).casefold()
        request_operator = resolved_semantic_relation
        requested_items = resolution.get("requested_cardinality") or getattr(plan, "requested_items", 1)
        entities = endpoint_hints or tuple(getattr(plan, "entities", ()) or ())
        source_id = getattr(plan, "source_id", None)
        scope = getattr(plan, "scope", "")
        conversation_operation = getattr(plan, "conversation_operation", "NONE")
        intent = (
            "UNSUPPORTED_OR_CONVERSATIONAL"
            if resolution.get("unsupported")
            else getattr(plan, "intent", "")
        )

    try:
        contract = _build_request_execution_contract_once(
            resolved_query, plan=_ResolutionPlan(), source_scope=source_scope
        )
    except SemanticCompilationError as exc:
        return replace(
            exc.result,
            status=("unsupported" if resolution.get("unsupported") else "invalid"),
            reason_codes=_unique((*exc.result.reason_codes, "model_resolution_invalid")),
            model_resolution_attempted=True,
            planner_operation=first.planner_operation,
            deterministic_operation=first.deterministic_operation,
            deterministic_semantic_agreement=False,
            pre_freeze_disagreement=first.pre_freeze_disagreement,
            reconciled=False,
        )
    return SemanticCompilationResult(
        status="repaired",
        draft=_semantic_draft(resolved_query, plan=_ResolutionPlan()),
        violations=_unique((*first.violations, *contract.semantic_validation.violations)),
        contract=contract,
        reason_codes=("model_semantic_reconciliation",),
        model_resolution_attempted=True,
        planner_operation=first.planner_operation,
        deterministic_operation=first.deterministic_operation,
        deterministic_semantic_agreement=False,
        pre_freeze_disagreement=first.pre_freeze_disagreement,
        reconciled=True,
    )


def build_request_execution_contract(
    query: str,
    *,
    plan: Any = None,
    source_scope: Iterable[str] = (),
) -> RequestExecutionContract:
    """Compatibility constructor; runtime should consume the result compiler."""
    result = compile_request_execution_contract(
        query, plan=plan, source_scope=source_scope
    )
    if result.contract is None:
        raise SemanticCompilationError(result)
    return result.contract


def build_canonical_contract(
    query: str,
    *,
    plan: Any = None,
    source_scope: Iterable[str] = (),
) -> RequestExecutionContract:
    """Compatibility constructor for tests and non-runtime tooling.

    Request runtime code must call ``build_request_execution_contract`` once,
    after conversation resolution, and pass the returned object downstream.
    """
    return build_request_execution_contract(
        query,
        plan=plan,
        source_scope=source_scope,
    )


def contracts_semantically_compatible(
    left: CanonicalSemanticContract,
    right: CanonicalSemanticContract,
) -> bool:
    """Compare required slots/output; internal operation enum names are absent."""
    if left.polarity != right.polarity and {left.polarity, right.polarity} <= {"positive", "negative"}:
        return False
    incompatible_predicates = {
        frozenset(("employs_workers", "recruits_employees")),
        frozenset(("employs_workers", "raises_capital")),
        frozenset(("recruits_employees", "raises_capital")),
        frozenset(("caused_by", "changed_between")),
        frozenset(("included_in", "excluded_from")),
    }
    if any(frozenset((a, b)) in incompatible_predicates for a in left.predicates for b in right.predicates):
        return False
    left_facets = {
        requirement.facet for requirement in left.requirements
        if requirement.facet
    }
    right_facets = {
        requirement.facet for requirement in right.requirements
        if requirement.facet
    }
    if left_facets and right_facets and left_facets != right_facets:
        return False
    if left.periods and right.periods and not set(left.periods) & set(right.periods):
        return False
    left_output = {(req.kind, req.completeness) for req in left.requirements if req.mandatory}
    right_output = {(req.kind, req.completeness) for req in right.requirements if req.mandatory}
    if left_output == right_output:
        return True
    # Direct lookup/count and compare/change can be equivalent when their
    # concrete required values/endpoints agree.
    countish = {"lookup", "count", "inclusion"}
    comparison = {"compare", "change_over_time"}
    relation_ok = (
        left.semantic_relation == right.semantic_relation
        or {left.semantic_relation, right.semantic_relation} <= countish
        or {left.semantic_relation, right.semantic_relation} <= comparison
    )
    metric_ok = not left.objects_or_metrics or not right.objects_or_metrics or bool(
        set(left.objects_or_metrics) & set(right.objects_or_metrics)
    )
    shape_ok = left.answer_shape == right.answer_shape or (
        {left.answer_shape, right.answer_shape} <= {"scalar", "scalar_count"}
    )
    return relation_ok and metric_ok and shape_ok


def _evidence_predicates(value: str) -> frozenset[str]:
    predicates: set[str] = set()
    for proposition, _start, _end in segment_text_propositions(value):
        predicates.update(predicate for predicate in DOCUMENT_ROLE_PREDICATES
                          if document_role_arguments(proposition, predicate))
    if _CAUSE_EVIDENCE_RE.search(value):
        predicates.add("caused_by")
    if _PURPOSE_RE.search(value):
        predicates.add("purpose_of")
    if _NEGATIVE_INCLUDE_RE.search(value):
        predicates.update(("excluded_from", "exception_to"))
    if _POSITIVE_INCLUDE_RE.search(value) and not _NEGATIVE_INCLUDE_RE.search(value):
        predicates.add("included_in")
    if _RAISES_CAPITAL_RE.search(value):
        predicates.add("raises_capital")
    if _RECRUITS_EMPLOYEES_RE.search(value):
        predicates.add("recruits_employees")
    if _EMPLOYS_RE.search(value):
        predicates.add("employs_workers")
    if _CHANGE_RE.search(value):
        predicates.add("changed_between")
    if _COMPARE_RE.search(value):
        predicates.add("compared_with")
    if _DEFINITION_RE.search(value):
        predicates.add("defined_as")
    if _NUMBER_RE.search(value):
        predicates.update(("has_count", "stated_as", "maximum_of", "minimum_of"))
    if _LIST_RE.search(value) or re.search(r"(?:[•▪◦]|(?:^|\n)\s*[-*])", value):
        predicates.add("has_members")
    return frozenset(predicates)


def _structural_type(hit: Mapping[str, Any]) -> str:
    metadata = hit.get("metadata") or {}
    if metadata.get("has_bullet_context"):
        return "bullet"
    if metadata.get("has_table_context"):
        return "table_row"
    return normalize_semantic_text(
        metadata.get("structural_type") or metadata.get("source_unit_type")
        or metadata.get("unit_type") or "body"
    ).casefold()


def _complete_proposition(text: str, structural: str) -> bool:
    if structural in _NAVIGATION_TYPES:
        return False
    if structural in {"heading", "caption", "chart_title"} and not _NUMBER_RE.search(text):
        return False
    if len(_terms(text)) < 3:
        return False
    if structural in {"table_row", "table", "chart", "chart_data"}:
        return bool(_NUMBER_RE.search(text) and re.search(r"[A-Za-z\u05d0-\u05ea]", text))
    return bool(_evidence_predicates(text) or re.search(r"[.!?׃:]", text))


def _predicate_compatible(requirement: TypedRequirement, text: str) -> bool:
    predicates = _evidence_predicates(text)
    predicate = requirement.predicate
    if requirement.kind == "cause_or_reason":
        return "caused_by" in predicates
    if requirement.kind == "positive_inclusion_or_count":
        return "excluded_from" not in predicates and bool(
            {"included_in", "has_count", "stated_as"} & predicates
        )
    if requirement.kind == "negative_relation":
        return "excluded_from" in predicates or "exception_to" in predicates
    if requirement.kind in {"purpose", "intended_use"} or predicate in {
        "purpose_of", "used_for",
    }:
        return bool(re.search(
            r"(?:\b(?:purpose|used\s+(?:for|to)|intended\s+to|designed\s+to|"
            r"created\s+to|serves?\s+to|in\s+order\s+to)\b|"
            r"(?:מטר(?:ה|תו)|נועד(?:ה|ו)?\s+ל|משמש(?:ת|ים|ות)?\s+ל|כדי\s+ל))",
            text,
            re.IGNORECASE,
        ))
    if requirement.kind in {"category_set", "category_value_bindings", "common_metric_unit_period_population", "winner", "winner_attribute"}:
        return bool(_NUMBER_RE.search(text))
    if requirement.kind in {"filter_condition", "filtered_row_period", "filtered_row_value"}:
        return bool(_NUMBER_RE.search(text))
    if predicate in {"performed_by", "method_of"}:
        return bool(re.search(r"(?:\b(?:by|using|through|step|method)\b|(?:באמצעות|לפי|שלב|שיטה))", text, re.I))
    if predicate == "summarizes":
        return True
    if predicate == "stated_as":
        return True
    if predicate in {
        "employs_workers", "recruits_employees", "raises_capital",
        "included_in", "excluded_from", "caused_by", "defined_as",
    }:
        return predicate in predicates
    if predicate in predicates:
        return True
    if predicate in {"has_count", "maximum_of", "minimum_of"}:
        return bool(_NUMBER_RE.search(text))
    if predicate in {"compared_with", "changed_between"}:
        if {"compared_with", "changed_between"} & predicates:
            return True
        # Two typed endpoints in one proposition structurally prove a
        # comparison even when the source states the values without a lexical
        # comparison verb.  The period, metric, unit, and population checks
        # below still have to agree for each endpoint.
        return (
            requirement.kind == "comparison_endpoint"
            and len(_NUMBER_RE.findall(text)) >= 2
            and bool(_PERIOD_RE.search(text))
        )
    return bool(predicates)


def _period_compatible(required_periods: Sequence[str], searchable: str) -> bool:
    if not required_periods:
        return True
    if all(period.startswith("endpoint_") for period in required_periods):
        return True
    compact = re.sub(r"(?<=\d)\s+(?=\d)", "", searchable)
    evidence_periods = {
        match.group(0) for match in _PERIOD_RE.finditer(searchable)
    } | {
        match.group(0) for match in _PERIOD_RE.finditer(compact)
    }
    for required in required_periods:
        if required in evidence_periods:
            continue
        if any(
            periods_semantically_compatible(required, observed)[0]
            for observed in evidence_periods
        ):
            continue
        # Some PDF extraction splits a school-year token inside its first
        # four-digit year ("202 2/23").  Preserve exact period checking when
        # possible, but accept that reconstructed first year only for a
        # slash-delimited school-year request.  A clear different year still
        # fails this check.
        primary = re.match(r"((?:19|20)\d{2})/\d{2,4}$", required)
        if primary and re.search(
            rf"(?<!\d){re.escape(primary.group(1))}(?!\d)", compact
        ):
            continue
        return False
    return True


def assess_candidate(
    hit: Mapping[str, Any],
    contract: CanonicalSemanticContract,
    *,
    effective_tenant: str = "",
) -> CandidateDecision:
    metadata = hit.get("metadata") or {}
    text = normalize_semantic_text(hit.get("document") or "")
    heading = normalize_semantic_text(metadata.get("section") or metadata.get("title") or "")
    searchable = f"{heading} {text}".strip()
    structural = _structural_type(hit)
    stored_tenant = str(metadata.get("tenant_id") or "")
    source_ok = not effective_tenant or not stored_tenant or stored_tenant == effective_tenant
    structure_ok = structural not in _NAVIGATION_TYPES
    proposition_ok = _complete_proposition(text, structural)
    compatible: list[str] = []
    subject_checks: list[bool] = []
    predicate_checks: list[bool] = []
    metric_checks: list[bool] = []
    period_checks: list[bool] = []
    population_checks: list[bool] = []
    polarity_checks: list[bool] = []
    unit_checks: list[bool] = []
    evidence_metrics = set(_metrics(searchable))
    if _NUMBER_RE.search(searchable):
        evidence_metrics.add("count")
    evidence_units = {
        (match.group("unit") or "").casefold()
        for match in _NUMBER_RE.finditer(searchable) if match.group("unit")
    }
    for requirement in contract.requirements:
        channels = {
            str(getattr(channel, "value", channel)).casefold()
            for channel in (hit.get("_channels") or ())
        }
        semantic_channel = "vector" in channels
        subject_ok = (
            _overlap(requirement.subject, searchable) >= 0.12
            or (
                semantic_channel
                and not requirement.objects_or_metrics
                and requirement.predicate in {"stated_as", "summarizes"}
            )
        )
        if requirement.predicate in DOCUMENT_ROLE_PREDICATES:
            from enterprise_retrieval import _relation_arguments
            target_role = RELATION_ARGUMENT_SCHEMAS[requirement.predicate][0]
            subject_ok = any(
                document_role_target_compatible(
                    requirement.relation_target or requirement.subject,
                    _relation_arguments(proposition,requirement.predicate).get(target_role,""),
                )
                for proposition, _start, _end in segment_text_propositions(str(hit.get("document") or ""))
            )
        # For concise questions the concrete metric/facet can be the strongest
        # subject anchor.
        requested_metrics = set(requirement.objects_or_metrics)
        requested_facets = (
            requested_metrics & {"role", "education_level", "experience_band"}
        ) | ({requirement.facet} if requirement.facet else set())
        evidence_facets = evidence_metrics & {"role", "education_level", "experience_band"}
        facet_ok = not requested_facets or bool(requested_facets & evidence_facets)
        if requirement.facet.startswith("experience_band:"):
            facet_ok = experience_band_compatible(
                requirement.facet, experience_band_facets(searchable),
            ) and not (
                requested_facets - {requirement.facet, "experience_band"} - evidence_facets
            )
        metric_ok = facet_ok and (
            not requested_metrics or bool(requested_metrics & evidence_metrics) or any(
            metric.replace("_", " ") in searchable.casefold() for metric in requested_metrics
            )
        )
        predicate_ok = _predicate_compatible(requirement, searchable) or (
            requirement.predicate == "has_members"
            and structural in {"bullet", "list", "table_row", "table"}
        )
        # Purpose is often stated as what a device or system does, without a
        # literal "purpose" or "used for" cue.  Candidate selection is only
        # a retrieval boundary: retain subject-matched complete propositions
        # so source-fact construction and semantic validation can decide
        # whether the passage actually answers the purpose requirement.  Do
        # not apply this relaxation to final claim/fact compatibility.
        if (
            not predicate_ok
            and requirement.kind in {"purpose", "intended_use"}
            and subject_ok
            and proposition_ok
        ):
            predicate_ok = True
        if requirement.predicate in DOCUMENT_ROLE_PREDICATES:
            predicate_ok = subject_ok
        period_ok = _period_compatible(requirement.periods, searchable)
        population_ok = not requirement.populations or any(
            _population_compatible(population, searchable)
            for population in requirement.populations
        )
        negative_evidence = "excluded_from" in _evidence_predicates(searchable)
        polarity_ok = (
            requirement.polarity == "neutral"
            or requirement.polarity == "negative" and negative_evidence
            or requirement.polarity == "positive" and not negative_evidence
        )
        requested_units = {unit.casefold() for unit in requirement.units if unit}
        unit_ok = not requested_units or bool(requested_units & evidence_units) or (
            "%" in requested_units and "percent" in evidence_metrics
        )
        subject_checks.append(subject_ok)
        predicate_checks.append(predicate_ok)
        metric_checks.append(metric_ok)
        period_checks.append(period_ok)
        population_checks.append(population_ok)
        polarity_checks.append(polarity_ok)
        unit_checks.append(unit_ok)
        # Predicate/polarity/period/facet are hard constraints.  Subject and
        # metric are alternative anchors because many indexed rows put one in
        # a parent heading rather than the row text.
        if (
            predicate_ok and period_ok and population_ok and polarity_ok and unit_ok
            and (metric_ok if requested_metrics else subject_ok)
        ):
            compatible.append(requirement.id)
    hard_mismatch = []
    if not source_ok:
        hard_mismatch.append("source_tenant_integrity_failed")
    if not structure_ok:
        hard_mismatch.append("navigation_or_debris")
    if not proposition_ok:
        hard_mismatch.append("incomplete_proposition")
    if not compatible:
        if predicate_checks and not any(predicate_checks):
            hard_mismatch.append("predicate_incompatible")
        if period_checks and not any(period_checks):
            hard_mismatch.append("period_incompatible")
        if population_checks and not any(population_checks):
            hard_mismatch.append("population_incompatible")
        if polarity_checks and not any(polarity_checks):
            hard_mismatch.append("polarity_incompatible")
        if unit_checks and not any(unit_checks):
            hard_mismatch.append("unit_incompatible")
        if metric_checks and not any(metric_checks):
            hard_mismatch.append("metric_or_facet_incompatible")
        if subject_checks and not any(subject_checks):
            hard_mismatch.append("subject_incompatible")
    operator_core_ids = {
        requirement.id for requirement in contract.requirements
        if requirement.kind in {
            "category_set", "category_value_bindings",
            "common_metric_unit_period_population", "winner",
        }
    }
    operator_core_ok = not operator_core_ids or bool(
        operator_core_ids & set(compatible)
    )
    if compatible and not operator_core_ok:
        hard_mismatch.append("facet_matrix_incompatible")
    selected = (
        source_ok and structure_ok and proposition_ok
        and bool(compatible) and operator_core_ok
    )
    heading_exact = max((_overlap(subject, heading) for subject in contract.subjects), default=0.0)
    preference = int(round(heading_exact * 40))
    preference += {
        "narrative": 30, "paragraph": 30, "body": 26,
        "table_row": 22, "table": 20, "chart_data": 18, "chart": 16,
        "composite": 14, "reconstructed": 14,
    }.get(structural, 8)
    preference += min(24, len(compatible) * 4)
    if metadata.get("parent_record_id") or metadata.get("previous_record_id") or metadata.get("next_record_id"):
        preference += 3
    target_anchors = _unique(
        item
        for requirement in contract.requirements
        for item in (
            requirement.counted_entity,
            requirement.target_entity_type,
            requirement.requested_attribute,
            requirement.subject,
        )
        if item
    )
    target_support = max(
        (_overlap(anchor, searchable) for anchor in target_anchors), default=0.0
    )
    proposition_support = int(round(target_support * 60))
    proposition_support += 25 if any(predicate_checks) else 0
    proposition_support += 15 if proposition_ok else 0
    channels = tuple(sorted({
        str(getattr(channel, "value", channel)).casefold()
        for channel in (hit.get("_channels") or ())
        if str(getattr(channel, "value", channel))
    }))
    raw_ranks = dict(hit.get("_channel_ranks") or {})
    channel_ranks = tuple(sorted(
        (str(name), max(1, int(rank)))
        for name, rank in raw_ranks.items()
        if str(rank).isdigit()
    ))
    best_rank = max(1, int(
        hit.get("_best_retrieval_rank")
        or min((rank for _, rank in channel_ranks), default=10_000)
    ))
    return CandidateDecision(
        candidate_id=candidate_id(hit),
        compatible_requirement_ids=tuple(compatible), selected=selected,
        rejected=not selected, rejection_reason_codes=_unique(hard_mismatch),
        subject_compatibility="compatible" if any(subject_checks) else "incompatible",
        predicate_compatibility="compatible" if any(predicate_checks) else "incompatible",
        metric_compatibility="compatible" if any(metric_checks) else "incompatible",
        period_compatibility="compatible" if any(period_checks) else "incompatible",
        population_compatibility="compatible" if any(population_checks) else "incompatible",
        polarity_compatibility="compatible" if any(polarity_checks) else "incompatible",
        unit_compatibility="compatible" if any(unit_checks) else "incompatible",
        structural_type_compatibility="compatible" if structure_ok else "incompatible",
        proposition_completeness="complete" if proposition_ok else "incomplete",
        source_tenant_integrity="verified" if source_ok else "failed",
        preference_score=preference,
        proposition_support_score=proposition_support,
        retrieval_score=float(hit.get("_fusion_score") or 0.0),
        retrieval_channels=channels,
        retrieval_channel_ranks=channel_ranks,
        best_retrieval_rank=best_rank,
    )


def select_semantic_candidates(
    hits: Sequence[Mapping[str, Any]],
    contract: CanonicalSemanticContract,
    *,
    effective_tenant: str = "",
    limit: int = 12,
) -> tuple[list[Mapping[str, Any]], tuple[CandidateDecision, ...]]:
    assessed = tuple(
        assess_candidate(hit, contract, effective_tenant=effective_tenant)
        for hit in hits
    )
    maximum = max(1, int(limit))
    selected_indexes: list[int] = []
    def semantic_order(index: int) -> tuple[Any, ...]:
        decision = assessed[index]
        return (
            decision.proposition_support_score,
            len(decision.retrieval_channels),
            -decision.best_retrieval_rank,
            decision.retrieval_score,
            decision.preference_score,
            -index,
        )

    # Reserve the strongest independent candidate for every mandatory
    # requirement before the global budget is applied.
    for requirement in (item for item in contract.requirements if item.mandatory):
        matches = [
            index for index in range(len(hits))
            if assessed[index].selected
            and requirement.id in assessed[index].compatible_requirement_ids
        ]
        match = max(matches, key=semantic_order) if matches else None
        if match is not None:
            if match not in selected_indexes:
                selected_indexes.append(match)
        if len(selected_indexes) >= maximum:
            break
    order = sorted(range(len(hits)), key=semantic_order, reverse=True)
    for index in order:
        if len(selected_indexes) >= maximum:
            break
        if assessed[index].selected and index not in selected_indexes:
            selected_indexes.append(index)

    selected_content = {
        normalize_semantic_text(hits[index].get("document") or "").casefold()
        for index in selected_indexes
    }
    final_decisions: list[CandidateDecision] = []
    for index, decision in enumerate(assessed):
        if index in selected_indexes:
            final_decisions.append(replace(
                decision,
                selected=True,
                rejected=False,
                rejection_reason_codes=(),
                selection_disposition="selected_for_fact_extraction",
            ))
            continue
        if not decision.selected:
            final_decisions.append(replace(
                decision,
                selected=False,
                rejected=True,
                selection_disposition=(
                    decision.rejection_reason_codes[0]
                    if decision.rejection_reason_codes else "semantic_incompatible"
                ),
            ))
            continue
        content = normalize_semantic_text(hits[index].get("document") or "").casefold()
        competitors = [
            assessed[selected]
            for selected in selected_indexes
            if set(decision.compatible_requirement_ids)
            & set(assessed[selected].compatible_requirement_ids)
        ]
        if content and content in selected_content:
            disposition = "duplicate_evidence"
        elif any(
            item.proposition_support_score > decision.proposition_support_score
            for item in competitors
        ):
            disposition = "dominated_by_higher_semantic_score"
        elif any(
            item.proposition_support_score == decision.proposition_support_score
            and len(item.retrieval_channels) > len(decision.retrieval_channels)
            for item in competitors
        ):
            disposition = "dominated_by_multi_channel_match"
        elif len(selected_indexes) >= maximum:
            disposition = "global_budget_exceeded"
        elif competitors:
            disposition = "requirement_budget_exceeded"
        else:
            disposition = "insufficient_proposition_support"
        final_decisions.append(replace(
            decision,
            selected=False,
            rejected=True,
            rejection_reason_codes=(disposition,),
            selection_disposition=disposition,
        ))
    return [hits[index] for index in selected_indexes], tuple(final_decisions)


def _material_claims(answer: str) -> tuple[str, ...]:
    text = normalize_semantic_text(answer)
    if not text:
        return ()
    claims = [
        item.strip(" -•▪◦\t")
        for item in re.split(r"(?<=[.!?׃])\s+|\s*;\s*|\n+", text)
        if len(_terms(item)) >= 2 or _NUMBER_RE.search(item)
    ]
    return tuple(claims[:MAX_CONTRACT_ITEMS])


def _claim_field_checks(requirement: TypedRequirement, claim: str) -> tuple[bool, ...]:
    predicates = _evidence_predicates(claim)
    claim_metrics = set(_metrics(claim))
    if _NUMBER_RE.search(claim):
        claim_metrics.add("count")
    list_item = requirement.kind in {"list", "list_attributes"}
    subject = (list_item and len(_terms(claim)) >= 2) or _overlap(requirement.subject, claim) >= 0.10 or bool(
        set(requirement.objects_or_metrics) & claim_metrics
    )
    predicate = list_item or _predicate_compatible(requirement, claim)
    requested_metrics = set(requirement.objects_or_metrics)
    metric = list_item or not requested_metrics or bool(requested_metrics & claim_metrics)
    claim_values = {match.group("value").replace(",", "") for match in _NUMBER_RE.finditer(claim)}
    requires_output_value = requirement.kind in {
        "count", "percentage", "winner_attribute", "comparison_endpoint",
    }
    value = (
        bool(claim_values)
        if requires_output_value
        else True
    )
    units = {(match.group("unit") or "").casefold() for match in _NUMBER_RE.finditer(claim)}
    requested_units = {unit.casefold() for unit in requirement.units if unit}
    unit = not requested_units or bool(requested_units & units) or (
        "%" in requested_units and bool(_PERCENT_RE.search(claim))
    )
    periods = {match.group(0) for match in _PERIOD_RE.finditer(claim)}
    period = not requirement.periods or bool(set(requirement.periods) & periods) or all(
        item.startswith("endpoint_") for item in requirement.periods
    )
    population = not requirement.populations or any(
        _overlap(item, claim) >= 0.16 for item in requirement.populations
    )
    negative = "excluded_from" in predicates or bool(_NEGATION_RE.search(claim))
    polarity = (
        requirement.polarity == "neutral"
        or requirement.polarity == "negative" and negative
        or requirement.polarity == "positive" and not negative
    )
    qualifier = not requirement.qualifiers or any(
        item in {"complete", "current", "historical", "role", "education_level", "experience_band", "total", "population_subgroup"}
        for item in requirement.qualifiers
    ) or any(_overlap(item, claim) > 0 for item in requirement.qualifiers)
    return subject, predicate, metric, value, unit, period, population, polarity, qualifier


def _citation_text(citation: Any) -> str:
    return normalize_semantic_text(
        getattr(citation, "evidence_quote", None)
        or getattr(citation, "_evidence_text", None)
        or ""
    )


def _citation_id(citation: Any) -> str:
    return str(getattr(citation, "id", None) or getattr(citation, "citation_number", ""))


def _numeric_values(value: str) -> set[str]:
    without_markers = _CLAIM_CITATION_RE.sub(" ", value)
    return {
        canonical_decimal_literal(match.group("value").replace(",", ""))
        for match in _NUMBER_RE.finditer(without_markers)
    }


_RENDERED_NUMERIC_LITERAL_RE = re.compile(r"(?<![\w])[-+]?\d+(?:[.,]\d+)*%?(?![\w])")


_RENDERED_DIRECTION_RE = re.compile(
    r"(?P<increase>\b(?:increase|increased|rose|grew|growth)\b|(?<![\w])ש?(?:עלייה|עלה|עלתה|גידול|גדל|גדלה)(?![\w]))|"
    r"(?P<decrease>\b(?:decrease|decreased|decline|declined|fell|dropped|reduction)\b|(?<![\w])ש?(?:ירידה|ירד|ירדה|קיטון|פחת|פחתה)(?![\w]))|"
    r"(?P<stable>\b(?:did\s+not\s+change|didn't\s+change|no\s+change|stable|unchanged)\b|"
    r"(?<![\w])(?:יציב|יציבה|ללא\s+שינוי|לא\s+השתנ(?:ה|תה))(?![\w]))", re.I,
)

@dataclass(frozen=True)
class RenderedDirectionAssertion:
    """A decision about the current clause, independent of source truth."""

    direction: str
    clause: str
    tail: str
    form: str
    asserted: bool


_ASSERTION_METRIC_SUBJECT_RE = re.compile(
    r"(?:the\s+)?(?:(?:total|overall)\s+)?"
    r"(?:(?:share|rate|percentage|count|amount)\s+of\s+(?:the\s+)?)?"
    r"(?:demand|revenue|profit|cost|price|capacity|duration|time|latency|salary|wage|"
    r"compensation|pay|share|metric|employee\s+count)|"
    r"(?:שיעור\s+)?ה?ביקוש|המדד\s+הכולל|קבוצת\s+ההתפלגות\s+המבוקשת",
    re.I,
)
_ASSERTION_OPERAND = (
    r"(?:(?:approximately|about|around|roughly|nearly|almost)\s+|כ[־-]?)?"
    r"[-+]?\d+(?:[.,]\d+)*\s*%?"
    r"(?:\s+(?:percent|percentage|employees|workers|jobs|people))?"
    r"(?:\s+(?:in\s+(?:(?:19|20)\d{2}|the\s+(?:current|previous)\s+period)|"
    r"ב[־-]?(?:19|20)\d{2}|בשנת\s+(?:19|20)\d{2}|בתקופה\s+ה(?:נוכחית|קודמת)))?"
)


def _assertion_scope_supported(value: str) -> bool:
    # Parse the entire descriptor with the established closed owner grammar;
    # extracting a convenient owner substring would discard an outer denial.
    # A renderer may retain a short demand denominator whose worker/job
    # domain is authenticated by the facts rather than its display label.
    if re.fullmatch(r"of\s+all\s+demand|מכלל\s+הביקוש", value, re.I):
        return True
    proof = (_canonical_population_scope(value) if _POPULATION_CANONICAL_DECLARATION.search(value)
             else _positive_population_scope(value))
    if proof.kind is not PopulationIdentityKind.UNRESOLVED and proof.keys:
        return True
    # A closed cohort noun can be grammatically affirmative while failing
    # semantic ownership (for example, a different occupational population).
    # Recognizing its syntax must not manufacture source identity or coverage.
    cohort = re.fullmatch(r"(?P<owner>.+?)\s+(?:among|within|across)\s+"
        r"(?:the\s+)?[A-Za-z][A-Za-z-]*", value, re.I)
    if cohort:
        owner = cohort.group("owner")
        proof = (_canonical_population_scope(owner) if _POPULATION_CANONICAL_DECLARATION.search(owner)
                 else _positive_population_scope(owner))
        return proof.kind is not PopulationIdentityKind.UNRESOLVED and bool(proof.keys)
    return False


def _assertion_band_subject(value: str) -> bool:
    prefix = re.match(r"(?:workers\s+with|עובדים\s+בעלי)\s+", value, re.I)
    return bool(prefix and _EXPERIENCE_BOUND_RE.fullmatch(value[prefix.end():]))


def _assertion_subject_supported(value: str, *, source_subjects: Sequence[str] = ()) -> bool:
    """Recognize whole noun labels; parentheses are typed descriptor slots.

    Unknown outer prose cannot become a label merely by preceding a colon.
    Population descriptors retain the separate source identity/scope checks.
    """
    text = value.strip()
    if normalize_semantic_text(text).casefold() in source_subjects:
        return True
    descriptors = []
    plain = []
    start = depth = 0
    for index, character in enumerate(text):
        if character == "(":
            if depth == 0:
                start = index
            depth += 1
        elif character == ")":
            depth -= 1
            if depth < 0:
                return False
            if depth == 0:
                descriptors.append(text[start + 1:index].strip())
        elif depth == 0:
            plain.append(character)
    if depth:
        return False
    head = normalize_semantic_text("".join(plain))
    if not head:
        return not descriptors  # Elliptical predicate after a clause boundary.
    band = re.fullmatch(r"(?:הקבוצה\s+של\s+)?(עובדים\s+בעלי\s+.+)", head)
    long_subject = re.fullmatch(
        r"(?P<metric>.+?)\s+(?:for\s+workers\s+with|לעובדים\s+בעלי)\s+(?P<band>.+)",
        head, re.I,
    )
    # Inline source ownership is another noun-phrase slot, also authenticated
    # by the existing population consumer. It cannot introduce a new predicate.
    owned_metric = re.fullmatch(
        r"(?P<metric>.+?)\s+(?P<scope>(?:at|in|within|across|among|בחברת|בקרב|מכלל)\s+.+)",
        long_subject.group("metric") if long_subject else head, re.I,
    )
    if owned_metric and not _assertion_scope_supported(owned_metric.group("scope")):
        return False
    supported = bool(_ASSERTION_METRIC_SUBJECT_RE.fullmatch(head))
    if band:
        supported = _assertion_band_subject(band.group(1))
    elif long_subject:
        supported = bool(_ASSERTION_METRIC_SUBJECT_RE.fullmatch(
                             owned_metric.group("metric") if owned_metric else long_subject.group("metric"))
                         and _EXPERIENCE_BOUND_RE.fullmatch(long_subject.group("band")))
    elif owned_metric:
        supported = bool(_ASSERTION_METRIC_SUBJECT_RE.fullmatch(owned_metric.group("metric")))
    if not supported:
        return False
    for descriptor in descriptors:
        if _assertion_band_subject(descriptor):
            continue
        # These are noun-scope slots emitted by the typed renderer. Ownership,
        # alias definitions and canonical identity are validated with facts.
        if not _assertion_scope_supported(descriptor):
            return False
    return True


def _assertion_operand_supported(value: str, *, subject: bool = False,
        source_subjects: Sequence[str] = (), source_units: Sequence[str] = ()) -> bool:
    text = value.strip()
    operand = _ASSERTION_OPERAND
    if source_units:
        operand = operand.replace('percent|percentage|employees|workers|jobs|people',
            'percent|percentage|employees|workers|jobs|people|'+'|'.join(re.escape(unit) for unit in source_units))
    if re.fullmatch(operand, text, re.I):
        return True
    # Nominal comparison first operands carry the subject and an explicit
    # period. The second operand cannot carry arbitrary free-form prose.
    if subject:
        measured = re.fullmatch(
            r"(?P<subject>.*?)\s*:?\s*(?:in\s+(?:19|20)\d{2}\s+)?"
            r"(?:was|is)\s+" + operand, text, re.I,
        )
        return bool(measured and _assertion_subject_supported(measured.group("subject").rstrip(" :"),
            source_subjects=source_subjects))
    return False


def _assertion_form(prefix: str, tail: str, direction: str, word: str, *,
        source_subjects: Sequence[str] = (), source_units: Sequence[str] = ()) -> str:
    """Positive full-form recognition, never absence of denial vocabulary."""
    prefix = prefix.strip()
    prefix = re.sub(r"^It\s+is\s+correct\s+that\s+", "", prefix, count=1, flags=re.I)
    nominal = re.fullmatch(
        r"(?P<left>.+?)\s*,?\s+(?:versus|vs\.?|compared\s+(?:with|to))\s+"
        r"(?P<right>.+?)\s*,\s*(?:representing|indicating|showing)\s+(?:a|an)",
        prefix, re.I,
    )
    if nominal:
        return "nominal_result" if (not tail.strip()
            and _assertion_operand_supported(nominal.group("left"), subject=True,
                source_subjects=source_subjects,source_units=source_units)
            and _assertion_operand_supported(nominal.group("right"),
                source_subjects=source_subjects,source_units=source_units)) else ""
    predicate_prefix = r"(?:showed|recorded|experienced|saw|was|were|is)\s+(?:a\s+|an\s+)?"
    if word.casefold() in {"increase", "decrease", "decline", "growth", "reduction", "עלייה", "גידול", "ירידה", "קיטון"}:
        predicate = re.search(r"(?:^|:\s*|\s+)" + predicate_prefix + r"$", prefix + " ", re.I)
        if predicate:
            prefix = prefix[:predicate.start()].rstrip()
        elif not prefix.endswith(":") and prefix:
            return ""
    elif direction == "stable":
        prefix = re.sub(r"(?:^|\s+)(?:remained|was|is|showed)\s*$", "", prefix, flags=re.I).rstrip()
    if not _assertion_subject_supported(prefix.rstrip(" :"),source_subjects=source_subjects):
        return ""
    if not tail.strip():
        return "declarative"  # Endpoint binding still required by consumers.
    operands = re.fullmatch(
        r"\s*(?:(?:sharply|slightly|substantially|significantly)\s+)?"
        r"(?:from\s+(?P<left>.+?)\s+to\s+(?P<right>.+)|"
        r"(?:between|at)\s+(?P<first>.+?)\s+and\s+(?P<last>.+)|"
        r"מ[־-]?\s*(?P<he_left>.+?)\s+ל[־-]?\s*(?P<he_right>.+))\s*",
        tail, re.I,
    )
    if not operands:
        return ""
    sides = (operands.group("left"), operands.group("right")) if operands.group("left") else (
        (operands.group("first"), operands.group("last")) if operands.group("first") else
        (operands.group("he_left"), operands.group("he_right")))
    return "declarative" if all(_assertion_operand_supported(side,
        source_subjects=source_subjects,source_units=source_units) for side in sides) else ""


def _rendered_direction_clauses(value: str, *, source_subjects: Sequence[str] = (),
        source_units: Sequence[str] = ()) -> tuple[str, ...]:
    # Clear independent propositions delimit negation scope. A comma alone
    # may delimit an owner label, so it is not a semantic boundary here.
    text = _CLAIM_CITATION_RE.sub(" ", value or "")
    quoted = tuple(match.span() for match in re.finditer(
        r'"[^\"]*(?:"|$)|“[^”]*(?:”|$)|‘[^’\']*(?:[’\']|$)|«[^»]*(?:»|$)|'
        r'״[^״]*(?:״|$)|(?<!\w)\'[^\']*(?:\'(?!\w)|$)', text))
    separator = re.compile(
        r"[.!?׃;](?:\s+|$)|\n+|\s+(?:but|while|whereas|אבל|בעוד|ואילו)\s+|,\s+and\s+|\s+(?=וההבדל\b)", re.I)
    output = []
    start = 0
    pending_subject = ""
    for match in separator.finditer(text):
        if any(left <= match.start() < right for left, right in quoted):
            continue
        clause = text[start:match.start()]
        if not clause.strip():
            start = match.end()
            continue
        if pending_subject and _RENDERED_DIRECTION_RE.match(clause.lstrip()):
            output.append(pending_subject + " " + clause.strip())
            pending_subject = ""
            start = match.end()
            continue
        if not re.match(r"[.!?׃;]", match.group()):
            # Layout and conjunctions cannot drop an unsupported outer wrapper.
            # Only a supported comparison or an explicit independent survey
            # proposition establishes a fresh comparison after this boundary.
            survey = re.fullmatch(r"(?P<label>.*?)\s*:?\s*(?:(?:the\s+)?survey\s+(?:was|is)\s+"
                r"(?:not\s+)?(?:audited|checked)|הסקר\s+(?:לא\s+)?נבדק)\s*,?\s*", clause, re.I)
            independent = bool(survey and _assertion_subject_supported(survey.group("label").rstrip(" :"),
                source_subjects=source_subjects))
            coordinator = bool(re.fullmatch(r"\s+(?:but|while|whereas|אבל|בעוד|ואילו)\s+",
                                            match.group(), re.I))
            right = text[match.end():]
            label = survey.group("label").rstrip(" :") if survey else ""
            if (coordinator and independent and label
                    and _RENDERED_DIRECTION_RE.match(right.lstrip())):
                # Coordinators can introduce an elliptical direction after a
                # separate survey note. Carry only the authenticated noun
                # subject; an explicit subject in the next clause stays new.
                pending_subject = label
                start = match.end()
                continue
            affirmative = any(_assertion_form(clause[:direction.start()], clause[direction.end():],
                direction.lastgroup, direction.group(),source_subjects=source_subjects,source_units=source_units)
                for direction in _RENDERED_DIRECTION_RE.finditer(clause))
            if not independent and not affirmative:
                continue
        output.append(clause)
        start = match.end()
    final_clause = text[start:]
    if pending_subject and _RENDERED_DIRECTION_RE.match(final_clause.lstrip()):
        output.append(pending_subject + " " + final_clause.strip())
    else:
        output.append(final_clause)
    return tuple(output)


def rendered_direction_assertion_decisions(value: str, *, source_subjects: Sequence[str] = (),
        source_units: Sequence[str] = ()) -> tuple[RenderedDirectionAssertion, ...]:
    """Classify material occurrences against the bounded assertion contract."""
    text = _CLAIM_CITATION_RE.sub(" ", (value or "").replace("’", "'"))
    decisions = []
    for clause in _rendered_direction_clauses(text,source_subjects=source_subjects,source_units=source_units):
        for match in _RENDERED_DIRECTION_RE.finditer(clause):
            prefix = clause[:match.start()]
            tail = clause[match.end():]
            quoted = bool(re.search(r'["“”‘«»״]|(?<!\w)\'|\'(?!\w)', clause))
            form = "" if quoted else _assertion_form(prefix, tail, match.lastgroup, match.group(),
                source_subjects=source_subjects,source_units=source_units)
            decisions.append(RenderedDirectionAssertion(match.lastgroup, clause, tail,
                form or "unsupported", bool(form)))
    return tuple(decisions)


def _rendered_direction_assertions(value: str) -> tuple[tuple[str, str, str], ...]:
    return tuple((decision.direction, decision.clause, decision.tail)
                 for decision in rendered_direction_assertion_decisions(value)
                 if decision.asserted)


def rendered_direction_present(value: str, direction: str) -> bool:
    """Whether a bounded declarative clause asserts the requested direction."""
    return any(observed == direction for observed, _clause, _tail in _rendered_direction_assertions(value))


def rendered_direction_mentioned(value: str) -> bool:
    """Material-clause detection also retains denied or quoted directions."""
    return bool(_RENDERED_DIRECTION_RE.search((value or "").replace("’", "'")))


def _source_assertion_slots(endpoints: Sequence[Any]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Authenticate optional grammar labels from both source endpoints.

    This supplies noun/unit slots, never a complete assertion or its polarity.
    The current answer still faces whole-clause, operand, population and facet
    checks. Standalone assertion recognition has no such source vocabulary.
    """
    if len(endpoints) != 2:
        return (), ()
    subjects, units = [], []
    denied = re.compile(r'\b(?:not|no|never|because|therefore|whether|if|unless|'
        r'that|is|are|was|were|does|did|can|could|must|may|said|says|denies|denied|'
        r'incorrect|false|prove|proves|ignore|override)\b|(?<!\w)(?:לא|אין|אם|אינו|אינה|כי|טען|טוען)(?!\w)',re.I)
    dependent = re.compile(r'\b(?:if|unless|when|whenever|provided|except|excluding|'
        r'conditional|hypothetical|unconfirmed|not|no|never|neither|without|denies|denied|'
        r'would|could|may|might)\b|\bsubject\s+to\b|'
        r'(?<!\w)(?:אם|אלא|כאשר|בתנאי|בכפוף|למעט|ללא|לא|אין|אינו|אינה|ייתכן)(?!\w)',re.I)
    for fact in endpoints:
        atom = getattr(fact,'evidence_atom',None)
        context = getattr(fact,'evidence_context',None)
        if (not getattr(fact,'structural_binding_valid',False) or atom is None or context is None
                or structural_binding_validation_errors(context)
                or atom.value != fact.value or atom.unit != fact.unit or atom.period != fact.period
                or atom.evidence_unit_id != fact.evidence_id or atom.candidate_id != fact.candidate_id
                or atom.numeric_binding != getattr(fact,'numeric_binding',None)):
            return (), ()
        source = str(getattr(fact,'source_claim','') or '')
        # A numeric producer may have a shorter subject than the proposition
        # that qualifies it. Optional noun/unit grammar cannot discard that
        # dependency and turn a conditional observation into an assertion.
        container = str(getattr(fact,'source_container_claim','') or '')
        structural = str(getattr(getattr(context,'structural_evidence_unit',None),'local_text','') or '')
        if not source.strip() or ' '.join(source.split()) not in ' '.join(structural.split()):
            return (), ()
        if dependent.search(' '.join((source,container,structural))):
            return (), ()
        normalized_source = normalize_semantic_text(source).casefold()
        if not normalized_source:
            return (), ()
        candidates = {str(getattr(fact,key,'') or '').strip(' :')
            for key in ('evidence_subject','evidence_metric','metric')}
        # A literal positive comparison can carry its exact source noun label
        # even when a legacy endpoint's metric field includes its predicate.
        for clause in re.split(r'[.!?;](?:\s+|$)|\n+|\s+(?:while|whereas|but|ואילו|בעוד)\s+',source,flags=re.I):
            for direction in _RENDERED_DIRECTION_RE.finditer(clause):
                candidate = clause[:direction.start()].strip(' :')
                if str(getattr(fact,'period','')) in clause and str(getattr(fact,'value','')).replace(',','') in clause.replace(',',''):
                    candidates.add(candidate)
        allowed = set()
        for candidate in candidates:
            label = normalize_semantic_text(candidate).casefold()
            if (not label or len(label)>180 or len(label.split())>16 or denied.search(label)
                    or not re.fullmatch(r'[\w\s()\-/]+',label)
                    or _RENDERED_DIRECTION_RE.search(label)
                    or label == normalize_semantic_text(str(getattr(fact,'unit','') or '')).casefold()):
                continue
            if re.search(r'(?<!\w)'+re.escape(label)+r'(?!\w)',normalized_source):
                allowed.add(label)
        subjects.append(allowed)
        unit = str(getattr(fact,'evidence_unit','') or getattr(fact,'unit','') or '').strip()
        aliases = {unit,*_UNIT_EQUIVALENCE.get(unit,())} - {'','number','numeric'}
        units.append({alias for alias in aliases if len(alias)<=60
            and re.fullmatch(r'[A-Za-z\u05d0-\u05ea%]+(?:[ /-][A-Za-z\u05d0-\u05ea]+)*',alias)})
    return tuple(sorted(set.intersection(*subjects))),tuple(sorted(set.intersection(*units)))


def _rendered_assertion_matches_endpoints(clause: str, tail: str, endpoints: Sequence[Any], direction: str) -> bool:
    if len(endpoints) != 2:
        return False
    ordered = sorted(endpoints, key=source_endpoint_order)
    expected = tuple((str(fact.period), canonical_decimal_literal(str(fact.value).replace(",", "")))
                     for fact in ordered)
    operand_start = r"(?=(?:(?:approximately|about|around|roughly|nearly|almost)\s+)?[-+]?\d)"
    explicit = re.match(r"\s*(?:(?:sharply|slightly|substantially|significantly)\s+)?"
        r"from\s+" + operand_start + r"(?P<left>.*?)\s+to\s+" + operand_start
        + r"(?P<right>.+)$", tail, re.I)
    if explicit is None:
        explicit = re.match(r"\s*מ(?:כ)?[־\-]?\s*(?=\d)(?P<left>.*?)\s+"
            r"ל(?:כ)?[־\-]?\s*(?=\d)(?P<right>.+)$", tail, re.I)
    if explicit is None:
        connector = r"(?:between|at)" if direction == "stable" else r"between"
        explicit = re.match(r"\s*" + connector + r"\s+" + operand_start
            + r"(?P<left>.*?)\s+and\s+" + operand_start + r"(?P<right>.+)$", tail, re.I)
    if explicit is not None:
        for side, endpoint in zip((explicit.group("left"), explicit.group("right")), expected):
            periods = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", side))
            values = _numeric_values(side) - periods
            if periods and periods != {endpoint[0]}:
                return False
            candidates = {pair for pair in expected if (not periods or pair[0] in periods)
                          and pair[1] in values}
            if candidates != {endpoint} or len(values) != 1:
                return False
        return True
    # A result predicate can follow a comparison whose period-owned operands
    # occur in either textual order. Only the explicit comparison immediately
    # preceding that predicate supplies operands; nearby numbers do not.
    nominal = re.fullmatch(
        r"(?P<left>.+?)\s*,?\s+"
        r"(?:versus|vs\.?|compared\s+(?:with|to))\s+"
        r"(?P<right>.+?)\s*,\s*"
        r"(?:representing|indicating|showing)\s+(?:a|an)\s+"
        r"(?:increase|decrease|decline|growth|reduction)\s*",
        clause, re.I,
    )
    if nominal is None or tail.strip():
        return False
    observed = []
    for side in (nominal.group("left"), nominal.group("right")):
        if rendered_direction_mentioned(side):
            return False
        # Experience cutoffs describe the population, not comparison values.
        masked = list(side)
        for _facet, (start, end) in experience_band_bounds(side):
            masked[start:end] = " " * (end - start)
        operand = "".join(masked)
        pairs = {(period, canonical_decimal_literal(literal.rstrip("%")))
                 for period, literal in rendered_period_number_pairs(operand)}
        periods = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", operand))
        values = _numeric_values(operand) - periods
        if (len(pairs) != 1
                or len(tuple(re.finditer(r"(?<!\d)(?:19|20)\d{2}(?!\d)", operand))) != 1
                or len(tuple(_RENDERED_NUMERIC_LITERAL_RE.finditer(operand))) != 2):
            return False
        pair = next(iter(pairs))
        if pair not in expected or periods != {pair[0]} or values != {pair[1]}:
            return False
        observed.append(pair)
    return set(observed) == set(expected)


def _rendered_direction_assertion_subject(
    clause: str, decision: RenderedDirectionAssertion, *, source_subjects: Sequence[str] = (),
) -> str:
    """Return the bounded subject attached to one material direction word."""
    nominal = re.fullmatch(
        r"(?P<left>.+?)\s*,?\s+"
        r"(?:versus|vs\.?|compared\s+(?:with|to))\s+"
        r"(?P<right>.+?)\s*,\s*"
        r"(?:representing|indicating|showing)\s+(?:a|an)\s+"
        r"(?:increase|decrease|decline|growth|reduction)\s*",
        clause, re.I,
    )
    if decision.form == "nominal_result" and nominal is not None:
        measured = re.fullmatch(
            r"(?P<subject>.*?)\s*:?\s*"
            r"(?:in\s+(?:(?:19|20)\d{2}|the\s+(?:current|previous)\s+period)\s+)?"
            r"(?:was|is)\s+" + _ASSERTION_OPERAND,
            nominal.group("left"), re.I,
        )
        if measured is not None:
            subject = measured.group("subject").rstrip(" :")
            if _assertion_subject_supported(subject,source_subjects=source_subjects):
                return subject

    occurrence = next((match for match in _RENDERED_DIRECTION_RE.finditer(clause)
                       if match.lastgroup == decision.direction), None)
    if occurrence is None:
        return ""
    prefix = clause[:occurrence.start()]
    segments = (prefix, *prefix.split(":"))
    candidates = set()
    for segment in segments:
        starts = {0, *(match.end() for match in re.finditer(r"\s+", segment))}
        for start in starts:
            candidate = segment[start:].strip().strip(" :")
            if candidate and _assertion_subject_supported(candidate,source_subjects=source_subjects):
                candidates.add(candidate)
    return max(candidates, key=lambda item: (len(item), item.casefold()), default="")


def _rendered_direction_context(
    clause: str, decision: RenderedDirectionAssertion, endpoints: Sequence[Any],
) -> Optional[str]:
    """Bind a material clause's metric/facet context to these source endpoints."""
    subjects,_units = _source_assertion_slots(endpoints)
    subject = _rendered_direction_assertion_subject(clause, decision,source_subjects=subjects)
    if not subject:
        return None
    scope = source_population_scope(subject)
    source_is_scoped = any(getattr(fact, "evidence_population", "") for fact in endpoints)
    scoped_subject = subject + ": " + decision.direction
    if source_is_scoped:
        if not scope or rendered_population_scope_errors(scoped_subject, endpoints):
            return None
    elif scope and rendered_population_scope_errors(scoped_subject, endpoints):
        return None
    core = subject
    if scope:
        start = core.casefold().find(scope.casefold())
        if start < 0:
            return None
        core = core[:start] + core[start + len(scope):]
    core = normalize_semantic_text(core).strip(" ,;:.")
    core = re.sub(r"\(\s*\)", "", core).strip(" ,;:.")
    return core or None


def _rendered_direction_has_other_authenticated_owner(
    clause: str, endpoints: Sequence[Any],
) -> bool:
    """Recognize a clearly different named owner as a separate comparison."""
    scope = source_population_scope(clause)
    if not scope:
        return False
    rendered = (_canonical_population_scope(scope)
                if _POPULATION_CANONICAL_DECLARATION.search(scope)
                else _positive_population_scope(scope))
    rendered_names = rendered.canonical_referents or rendered.named_referents
    if not rendered_names:
        return False
    source_names = set()
    for fact in endpoints:
        proof = source_population_identity_proof(fact)
        source_names.update(proof.canonical_referents or proof.named_referents)
    return bool(source_names and rendered_names.isdisjoint(source_names))


def _rendered_direction_pair_signature(clause: str) -> frozenset[tuple[str, str]]:
    pairs = rendered_period_number_pairs(clause)
    return frozenset(pairs) if len(pairs) == 2 else frozenset()


def _rendered_direction_pair_values(signature: frozenset[tuple[str, str]]) -> frozenset[str]:
    return frozenset(value for _period, value in signature)


def _rendered_direction_is_comparison_reference(clause: str) -> bool:
    """Recognize a generic anaphoric comparison reference without parsing polarity."""
    return bool(re.search(
        r"(?<![\w])(?:the|this|that|same)\s+comparison(?![\w])|"
        r"(?<![\w])ההשוואה(?![\w])",
        clause, re.I,
    ))


def rendered_direction_supports_endpoints(value: str, direction: str, endpoints: Sequence[Any]) -> bool:
    """Bind the asserted direction and polarity to its actual endpoint pair."""
    if len(endpoints) != 2:
        return False
    subjects,units = _source_assertion_slots(endpoints)
    decisions = rendered_direction_assertion_decisions(value,source_subjects=subjects,source_units=units)
    anchors = tuple(decision for decision in decisions
        if decision.asserted and decision.direction == direction
        and _rendered_assertion_matches_endpoints(
            decision.clause, decision.tail, endpoints, decision.direction))
    if not anchors:
        return False
    expected = {(str(fact.period), canonical_decimal_literal(str(fact.value).replace(",", "")))
                for fact in endpoints}
    target_periods = {period for period, _value in expected}
    anchor_ids = {id(decision) for decision in anchors}
    anchor_contexts = {
        context for decision in anchors
        if (context := _rendered_direction_context(decision.clause, decision, endpoints))
    }
    context_signatures: dict[str, set[frozenset[tuple[str, str]]]] = {}
    for decision in decisions:
        context = _rendered_direction_context(decision.clause, decision, endpoints)
        signature = _rendered_direction_pair_signature(decision.clause)
        if context and signature and decision.asserted:
            context_signatures.setdefault(context, set()).add(signature)
    target_signature = frozenset(expected)
    for context in anchor_contexts:
        context_signatures.setdefault(context, set()).add(target_signature)
    comparison_candidates = {
        (context, signature)
        for context, signatures in context_signatures.items()
        for signature in signatures
    }
    for candidate in decisions:
        if (not candidate.asserted
                or _rendered_direction_is_comparison_reference(candidate.clause)):
            continue
        signature = _rendered_direction_pair_signature(candidate.clause)
        if not signature:
            continue
        context = _rendered_direction_context(candidate.clause, candidate, endpoints)
        if context is None:
            context = _rendered_direction_assertion_subject(candidate.clause, candidate,source_subjects=subjects)
        context = normalize_semantic_text(context or "").strip(" ,;:.")
        comparison_candidates.add((context, signature))
    comparison_candidates.update(
        (context, target_signature) for context in (anchor_contexts or {""})
    )

    # A complete assertion can establish the comparison that abbreviated
    # companions refer to. Every material occurrence is then checked against
    # that same owner/metric/facet context and the source-bound endpoint pair.
    for decision in decisions:
        clause = decision.clause
        quoted = bool(re.search(r'["“”‘«»״]|(?<!\w)\'|\'(?!\w)', clause))
        if quoted or id(decision) in anchor_ids:
            continue
        if not rendered_direction_mentioned(clause):
            continue
        context = _rendered_direction_context(clause, decision, endpoints)
        binds_pair = _rendered_assertion_matches_endpoints(
            clause, decision.tail, endpoints, decision.direction)
        if context is None:
            if _rendered_direction_is_comparison_reference(clause):
                periods = {match.group(0) for match in re.finditer(
                    r"(?<!\d)(?:19|20)\d{2}(?!\d)", clause)}
                operand_values = _numeric_values(decision.tail) - periods
                if periods or operand_values:
                    matching = [candidate for candidate in comparison_candidates
                        if (not periods or periods <= {period for period, _value in candidate[1]})
                        and (not operand_values
                             or _rendered_direction_pair_values(candidate[1]) == operand_values)]
                    if len(matching) > 1:
                        return False
                    if matching:
                        candidate = matching[0]
                    elif len(comparison_candidates) == 1:
                        candidate = next(iter(comparison_candidates))
                    else:
                        return False
                else:
                    if len(comparison_candidates) != 1:
                        return False
                    candidate = next(iter(comparison_candidates))
                candidate_context, candidate_signature = candidate
                if candidate_signature != target_signature:
                    continue
                if candidate_context and candidate_context not in anchor_contexts:
                    continue
                if not decision.asserted or decision.direction != direction:
                    return False
                if (periods or operand_values) and not binds_pair:
                    return False
                if not (periods or operand_values) and decision.form != "declarative":
                    return False
                continue
            # A clause with endpoint operands but no reliably authenticated
            # owner/population/metric context is ambiguous, so it cannot be
            # silently detached from this comparison.
            if (not _rendered_direction_has_other_authenticated_owner(clause, endpoints)
                    and (binds_pair or target_periods & {
                        match.group(0) for match in re.finditer(
                            r"(?<!\d)(?:19|20)\d{2}(?!\d)", clause)
                    })):
                return False
            continue
        if context not in anchor_contexts:
            target_facets = {facet for fact in endpoints for facet in source_fact_facets(fact)
                             if facet.startswith("experience_band:")}
            observed_facets = set(experience_band_facets(
                _rendered_direction_assertion_subject(clause, decision)))
            # Values do not supply a missing band owner. If a source-bound
            # band comparison is shortened until its band disappears, fail
            # closed; a separately named metric or band remains independent.
            missing_band = bool(target_facets and not observed_facets)
            if binds_pair and missing_band and not _rendered_direction_has_other_authenticated_owner(
                    clause, endpoints):
                return False
            continue

        signature = _rendered_direction_pair_signature(clause)
        periods = {match.group(0) for match in re.finditer(
            r"(?<!\d)(?:19|20)\d{2}(?!\d)", clause)}
        related = False
        explicit_pair = False
        if signature:
            if signature == target_signature or periods == target_periods:
                related = True
                explicit_pair = True
            elif len(periods) == 2:
                # Fully explicit periods distinguish an independent
                # comparison in the same metric/population context.
                continue
            elif periods.isdisjoint(target_periods):
                continue
            else:
                related = True  # Partial endpoint reference is ambiguous.
                explicit_pair = True
        elif periods:
            if periods.isdisjoint(target_periods):
                continue
            related = True
            explicit_pair = True
        else:
            operand_values = _numeric_values(decision.tail)
            candidates = context_signatures.get(context, set())
            if operand_values:
                matching = [candidate for candidate in candidates
                            if _rendered_direction_pair_values(candidate) == operand_values]
                if len(matching) > 1:
                    return False
                if matching and matching[0] != target_signature:
                    continue
                related = True
                explicit_pair = True
            else:
                if len(candidates) > 1:
                    return False
                related = True

        if related and (not decision.asserted or decision.direction != direction):
            return False
        if related and explicit_pair and not binds_pair:
            return False
        if related and not explicit_pair and decision.form != "declarative":
            return False
    return True


def rendered_temporal_direction_errors(answer: str, linked_facts: Sequence[Any]) -> tuple[str, ...]:
    """Revalidate current publication assertions, including an earlier graph."""
    reasons = []
    numeric = tuple({str(getattr(fact, "fact_id", "") or index): fact
                     for index, fact in enumerate(linked_facts)
                     if getattr(fact, "fact_type", "numeric") == "numeric"}.values())
    for direction_fact in linked_facts:
        if getattr(direction_fact, "fact_type", "") != "derived_direction":
            continue
        ids = tuple(getattr(direction_fact, "source_typed_fact_ids", ()) or ())
        endpoints = tuple(fact for fact in numeric if getattr(fact, "fact_id", "") in ids)
        if not rendered_direction_supports_endpoints(answer, direction_fact.direction, endpoints):
            reasons.append("rendered_direction_assertion_unbound")
    # Graph admission may have rejected the direction fact already. Denied
    # or merely mentioned comparisons still cannot become supported numeric
    # claims when their authentic operands survive that rejection.
    for clause in _rendered_direction_clauses(answer):
        if not _RENDERED_DIRECTION_RE.search(clause.replace("’", "'")):
            continue
        bound = {(period, canonical_decimal_literal(literal.rstrip("%")))
                 for period, literal in rendered_period_number_pairs(clause)}
        endpoints = tuple(fact for fact in numeric if (str(fact.period),
            canonical_decimal_literal(str(fact.value).replace(",", ""))) in bound)
        if len(endpoints) != 2:
            continue
        from decimal import Decimal
        left, right = sorted(endpoints, key=source_endpoint_order)
        a, b = Decimal(str(left.value).replace(",", "")), Decimal(str(right.value).replace(",", ""))
        direction = "increase" if b > a else "decrease" if b < a else "stable"
        if not rendered_direction_supports_endpoints(answer, direction, endpoints):
            reasons.append("rendered_direction_assertion_unbound")
    return _unique(reasons)


def rendered_period_number_bindings(value: str) -> set[tuple[str, str, bool]]:
    """Bind period, measured value and its own approximation marker."""
    text = _CLAIM_CITATION_RE.sub(" ", value or "")
    pairs = set()
    for clause in re.split(r"[.!?׃;](?:\s+|$)|\n+", text):
        numbers = tuple(_RENDERED_NUMERIC_LITERAL_RE.finditer(clause))
        approximate_starts = {match.start("number") for match in re.finditer(
            r"(?:\b(?:approximately|about|around|roughly|nearly|almost)\s+|"
            r"(?:בערך|בקירוב|כמעט|משוער(?:ת|ים|ות)?|(?<![\w])(?:[מלב])?כ\s*[-־~]?)\s*)"
            r"(?P<number>[-+]?\d+(?:[.,]\d+)*%?)", clause, re.I)}
        def measured_unit(match):
            if match.group().endswith("%"):
                return "%"
            tail = re.match(r"\s*([A-Za-z\u05d0-\u05ea]+)", clause[match.end():])
            word = tail.group(1).casefold() if tail else ""
            if word in {"percent", "percentage", "אחוז", "אחוזים"}:
                return "%"
            return word if word in _COUNT_UNITS or word in _UNIT_EQUIVALENCE else ""
        periods = tuple(match for match in re.finditer(r"(?<!\d)(?:19|20)\d{2}(?!\d)", clause)
            if not any(number.start() <= match.start() < number.end() and measured_unit(number) for number in numbers))
        operands = tuple(match for match in numbers if measured_unit(match)
            or not re.fullmatch(r"(?:19|20)\d{2}", match.group()))
        for period in periods:
            def distance(number):
                return min(abs(number.end() - period.start()), abs(number.start() - period.end()))
            nearby = tuple(number for number in operands if distance(number) <= 96)
            if nearby:
                explicit_left = tuple(number for number in nearby if number.end() <= period.start() and re.fullmatch(
                    r"\s*(?:[A-Za-z\u05d0-\u05ea]+\s+)?(?:(?:in|during|for|בשנת|לשנת|ב)[־\-]?\s*|\(\s*)",
                    clause[number.end():period.start()], re.I))
                number = max(explicit_left, key=lambda item:item.end()) if explicit_left else min(nearby, key=distance)
                unit = measured_unit(number)
                literal = canonical_decimal_literal(number.group().rstrip("%").replace(",", ""))
                pairs.add((period.group(), literal + ("%" if unit in {"%", "percent", "percentage", "אחוז", "אחוזים"} else ""),
                           number.start() in approximate_starts))
    return pairs



def rendered_period_number_pairs(value: str) -> set[tuple[str, str]]:
    """Compatibility view of the canonical period/value bindings."""
    return {(period, literal) for period, literal, _approximate in rendered_period_number_bindings(value)}


def rendered_fact_value_errors(answer: str, linked_facts: Sequence[Any]) -> tuple[str, ...]:
    """Bind material numbers to facts and experience thresholds to their slots.

    A proved duration threshold may appear inside its compatible experience
    category. It never contributes a free-standing answer quantity.
    """
    from financial_semantics import (
        currency_for_span, document_reference_spans, normalize_currency,
        rendered_document_reference_spans, rendered_financial_role_errors,
    )
    body = _CLAIM_CITATION_RE.sub(" ", answer)
    body = re.sub(r"(?m)^\s*(?:[-*•]\s+|\d{1,2}[.)]\s+)", " ", body)
    source_facets = tuple(facet for fact in linked_facts for facet in source_fact_facets(fact))
    reasons = list(rendered_population_scope_errors(answer, linked_facts))
    reasons.extend(rendered_temporal_direction_errors(answer, linked_facts))
    reasons.extend(rendered_financial_role_errors(answer, linked_facts))
    proven_document_spans = set(rendered_document_reference_spans(body, linked_facts))
    from document_query import rendered_location_support
    source_spans, source_errors = rendered_location_support(body, linked_facts)
    proven_document_spans.update(source_spans)
    reasons.extend(source_errors)
    if (any(normalize_currency(getattr(fact, "unit", "")) for fact in linked_facts)
            and any(span not in proven_document_spans for span in document_reference_spans(body))):
        reasons.append("rendered_document_reference_unbound")
    percent_units = {"%", "percent", "percentage", "אחוז", "אחוזים"}
    def count_aliases(unit):
        aliases = {unit, *_UNIT_EQUIVALENCE.get(unit, ())}
        return aliases | {alias+'s' for alias in aliases if alias.isascii()}
    def unit_compatible(observed, source):
        observed_currency, source_currency = normalize_currency(observed), normalize_currency(source)
        if observed_currency or source_currency:
            return bool(observed_currency and source_currency and observed_currency == source_currency)
        return bool(observed == source or observed in percent_units and source in percent_units
            or observed in _COUNT_UNITS and source in _COUNT_UNITS
            and count_aliases(observed) & count_aliases(source))
    continuation_words = {"in", "during", "for", "and", "or", "of", "to", "from", "by", "at",
        "is", "are", "was", "were", "respectively", "the", "a", "an", "בשנת", "בשנים", "בין", "של", "היה", "היו"}
    def immediate_quantity_predicate(text, match):
        start, _end = _scalar_proposition_bounds(text, match.start("value"), match.end("value"))
        before = text[start:match.start("value")].rstrip()
        cues = tuple(_POINT_ESTIMATE_CONTEXT_RE.finditer(before))
        return bool(cues and re.fullmatch(r"\s*(?:(?:approximately|about|around|roughly)\s*)?",
            before[cues[-1].end():], re.I))
    def literal_unit(text, match):
        currency = currency_for_span(text, match.start("value"), match.end("value"))
        if currency:
            return currency
        unit = str(match.group("unit") or "").casefold()
        if unit and match.end("unit") < len(text) and text[match.end("unit")].isalpha():
            unit = ""
        tail = re.match(r"\s*([A-Za-z\u05d0-\u05ea]+)", text[match.end("value"):])
        if not unit and tail:
            word = tail.group(1).casefold()
            if word in _COUNT_UNITS or word not in continuation_words and (
                    immediate_quantity_predicate(text, match) or re.fullmatch(r"[A-Z]{3,8}", tail.group(1))):
                unit = word
        return unit
    masked = list(body)
    for start, end in proven_document_spans:
        masked[start:end] = " " * (end - start)
    for facet, (start, end) in experience_band_bounds(body):
        if not source_facet_compatible(facet, source_facets):
            reasons.append("rendered_experience_scope_unbound")
        else:
            masked[start:end] = " " * (end - start)
    allowed_values: set[str] = set()
    for fact in linked_facts:
        allowed_values.update(_numeric_values(str(getattr(fact, "value", "") or "")))
        allowed_values.update(_numeric_values(str(getattr(fact, "period", "") or "")))
        if str(getattr(fact, "fact_type", "numeric")) == "text_proposition":
            allowed_values.update(_numeric_values(str(getattr(fact, "object_text", "")
                or getattr(fact, "source_claim", "") or "")))
        for row in getattr(fact, "filter_rows", ()) or ():
            for field_value in row:
                allowed_values.update(_numeric_values(str(field_value or "")))
        for qualifier in getattr(fact, "qualifiers", ()) or ():
            allowed_values.update(_numeric_values(str(qualifier)))
    if not _numeric_values("".join(masked)) <= allowed_values:
        reasons.append("rendered_value_unbound")
    # Rendered period/value pairs belong to admitted endpoint facts, rather
    # than to a bag containing every source year and every source quantity.
    endpoint_values = {}
    for fact in linked_facts:
        if str(getattr(fact, "fact_type", "numeric")) != "numeric":
            continue
        period = str(getattr(fact, "period", "") or "")
        if period:
            endpoint_values.setdefault(period, set()).update(canonical_decimal_literal(number.group().rstrip("%").replace(",", ""))
                for number in _RENDERED_NUMERIC_LITERAL_RE.finditer(str(getattr(fact, "value", "") or "")))
    if sum(bool(re.fullmatch(r"(?:19|20)\d{2}", period)) for period in endpoint_values) >= 2:
        for period, literal in rendered_period_number_pairs("".join(masked)):
            if period in endpoint_values and canonical_decimal_literal(literal.rstrip("%")) not in endpoint_values[period]:
                reasons.append("rendered_period_value_binding_invalid")
    # Equal values in different periods remain independent observations.
    # An approximation marker on one cannot discharge the other's qualifier.
    original_clauses = re.split(r"[.!?׃;](?:\s+|$)|\n+", body)
    masked_clauses = re.split(r"[.!?׃;](?:\s+|$)|\n+", "".join(masked))
    for original_clause, masked_clause in zip(original_clauses, masked_clauses):
        rendered_bindings = rendered_period_number_bindings(masked_clause)
        rendered_pairs = {(period, canonical_decimal_literal(literal.rstrip("%")))
                          for period, literal, _approximate in rendered_bindings}
        approximate_bindings = {(period, canonical_decimal_literal(literal.rstrip("%")))
                                for period, literal, approximate in rendered_bindings if approximate}
        asserted_facets = experience_band_facets(original_clause)
        for fact in linked_facts:
            if (str(getattr(fact, "fact_type", "numeric")) != "numeric"
                    or "approximate" not in (getattr(fact, "qualifiers", ()) or ())):
                continue
            if asserted_facets and source_fact_facets(fact) and not any(
                    source_facet_compatible(facet, source_fact_facets(fact)) for facet in asserted_facets):
                continue
            period = str(getattr(fact, "period", "") or "")
            if re.fullmatch(r"(?:19|20)\d{2}", period) and any(
                    (period, value) in rendered_pairs and (period, value) not in approximate_bindings
                    for value in _numeric_values(str(getattr(fact, "value", "") or ""))):
                reasons.append("rendered_period_approximation_binding_invalid")
    # An explicit from/to expression supplies its own comparison order. A
    # correct request direction cannot justify the opposite rendered order.
    directional_patterns = (
        r"\b(?P<direction>decreased?|declined?|fell|dropped?|increased?|rose|grew|stable|unchanged)\b[^.!?;]*?\bfrom\s+(?P<left>.*?)\s+to\s+(?P<right>[^!?;]*)(?:$)",
        r"(?P<direction>ירד|פחת|עלה|גדל|יציב|ללא שינוי)[^!?;]*?מ(?:כ)?[־\-]?\s*(?=\d)(?P<left>.*?)\s+ל(?:כ)?[־\-]?\s*(?=\d)(?P<right>[^!?;]*)(?:$)",
    )
    for clause in re.split(r"[.!?׃;](?:\s+|$)|\n+", "".join(masked)):
        for pattern in directional_patterns:
            match = re.search(pattern, clause, re.I)
            if not match:
                continue
            endpoints = []
            for side in (match.group("left"), match.group("right")):
                periods = set(re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", side))
                values = {canonical_decimal_literal(number.group().rstrip("%").replace(",", ""))
                    for number in _RENDERED_NUMERIC_LITERAL_RE.finditer(side)} - periods
                candidates = {(period, value) for period, admitted in endpoint_values.items() for value in admitted
                    if (not periods or period in periods) and (not values or value in values)}
                endpoints.append(candidates)
            if any(len(endpoint) != 1 for endpoint in endpoints):
                if endpoint_values:
                    reasons.append("rendered_direction_endpoints_unbound")
                continue
            from decimal import Decimal
            left = Decimal(next(iter(endpoints[0]))[1]); right = Decimal(next(iter(endpoints[1]))[1])
            observed = "decrease" if right < left else "increase" if right > left else "stable"
            token = match.group("direction").casefold()
            claimed = "decrease" if re.fullmatch(r"decreased?|declined?|fell|dropped?|ירד|פחת", token) else (
                "increase" if re.fullmatch(r"increased?|rose|grew|עלה|גדל", token) else "stable")
            if observed != claimed:
                reasons.append("rendered_direction_order_inconsistent")
    # A period or a category threshold is not a percentage observation even
    # when its digits already occur elsewhere in the supported statement.
    percentage_values: set[str] = set()
    for fact in linked_facts:
        if str(getattr(fact, "unit", "")).casefold() in {"%", "percent", "percentage", "אחוז", "אחוזים"}:
            percentage_values.update(_numeric_values(str(getattr(fact, "value", "") or "")))
        if str(getattr(fact, "fact_type", "")) == "text_proposition":
            text = str(getattr(fact, "object_text", "") or getattr(fact, "source_claim", "") or "")
            for match in _NUMBER_RE.finditer(text):
                if re.match(r"\s*(?:%|percent(?:age)?\b|אחוז(?:ים)?)", text[match.end("value"):], re.I):
                    percentage_values.add(canonical_decimal_literal(match.group("value").replace(",", "")))
    for match in _NUMBER_RE.finditer(body):
        if not "".join(masked[match.start("value"):match.end("value")]).strip():
            continue
        # A conflicting prefix/suffix is ambiguous, rather than unitless.
        prefix_currency = currency_for_span(body[:match.end("value")], match.start("value"), match.end("value"))
        suffix_currency = currency_for_span(body[match.start("value"):], 0, match.end("value") - match.start("value"))
        if prefix_currency and suffix_currency and prefix_currency != suffix_currency:
            reasons.append("rendered_currency_ambiguous")
        if (re.match(r"\s*(?:%|percent(?:age)?\b|אחוז(?:ים)?)", body[match.end("value"):], re.I)
                and canonical_decimal_literal(match.group("value").replace(",", "")) not in percentage_values):
            reasons.append("rendered_percentage_role_unbound")
        # A calendar token cannot supply a measured quantity, and an admitted
        # percentage cannot supply a count merely because the digits match.
        # Reuse the quantitative-unit registry rather than a question-specific
        # noun or a list of benchmark values.
        value = canonical_decimal_literal(match.group("value").replace(",", ""))
        quantity_values = set()
        for fact in linked_facts:
            quantity_values.update(_numeric_values(str(getattr(fact, "value", "") or "")))
            for row in getattr(fact, "filter_rows", ()) or ():
                if len(row) >= 2:
                    quantity_values.update(_numeric_values(str(row[1])))
            if str(getattr(fact, "fact_type", "")) == "text_proposition":
                source = str(getattr(fact, "object_text", "") or getattr(fact, "source_claim", "") or "")
                quantity_values.update(canonical_decimal_literal(literal.group("value").replace(",", ""))
                    for literal in _NUMBER_RE.finditer(source) if literal_unit(source, literal) or immediate_quantity_predicate(source, literal))
        if immediate_quantity_predicate(body, match) and value not in quantity_values:
            reasons.append("rendered_quantity_value_unbound")
        observed_unit = literal_unit(body, match)
        if not observed_unit:
            continue
        unit_bound = False
        for fact in linked_facts:
            fact_type = str(getattr(fact, "fact_type", "numeric"))
            if fact_type == "text_proposition":
                source = str(getattr(fact, "object_text", "") or getattr(fact, "source_claim", "") or "")
                for literal in _NUMBER_RE.finditer(source):
                    source_unit = literal_unit(source, literal)
                    if (canonical_decimal_literal(literal.group("value").replace(",", "")) == value
                            and unit_compatible(observed_unit, source_unit)):
                        unit_bound = True
                continue
            quantity_values = _numeric_values(str(getattr(fact, "value", "") or ""))
            for row in getattr(fact, "filter_rows", ()) or ():
                if len(row) >= 2:
                    quantity_values.update(_numeric_values(str(row[1])))
            if value not in quantity_values:
                continue
            source_unit = str(getattr(fact, "unit", "") or "").casefold()
            source_count_units = {source_unit} if source_unit in _COUNT_UNITS else set()
            if source_unit in {"number", "numeric", "count"}:
                source = str(getattr(fact, "source_claim", "") or "")
                literals = tuple(literal for literal in _NUMBER_RE.finditer(source)
                    if canonical_decimal_literal(literal.group("value").replace(",", "")) == value)
                if len(literals) == 1:
                    unit = literal_unit(source, literals[0])
                    if unit in _COUNT_UNITS:
                        source_count_units.add(unit)
                # The actor/row label cannot lend its entity type to a
                # measured operand with an explicit source-local count unit.
                if not source_count_units:
                    fields = ("column_header",) if source else ("evidence_subject", "evidence_target_entity")
                    for field in fields:
                        source_count_units.update(token.group().casefold() for token in _TOKEN_RE.finditer(
                            str(getattr(fact, field, "") or "")) if token.group().casefold() in _COUNT_UNITS)
                    if len(literals) == 1:
                        before = source[:literals[0].start("value")]
                        for cue in (_SOURCE_CARDINALITY_NOUN_RELATION_RE, _SOURCE_CARDINALITY_ENTITY_RELATION_RE):
                            prefix = cue.search(before)
                            if prefix:
                                source_count_units.update(token.group().casefold() for token in _TOKEN_RE.finditer(
                                    prefix.group()) if token.group().casefold() in _COUNT_UNITS)
            count_unit = observed_unit in _COUNT_UNITS and any(
                unit_compatible(observed_unit, unit) for unit in source_count_units)
            if unit_compatible(observed_unit, source_unit) or count_unit:
                unit_bound = True
        if not unit_bound:
            reasons.append("rendered_quantity_unit_unbound")
    return _unique(reasons)


@dataclass(frozen=True)
class AcceptanceDecision:
    """The one route-independent decision made after answer rendering."""

    accepted: bool
    mandatory_requirement_coverage: bool
    claim_fact_linkage: bool
    fact_evidence_linkage: bool
    claim_citation_linkage: bool
    contextual_complete: bool
    standalone_complete: bool
    factual_bindings: bool
    structural_safety: bool
    rendered_values_bound: bool
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "mandatory_requirement_coverage": self.mandatory_requirement_coverage,
            "claim_fact_linkage": self.claim_fact_linkage,
            "fact_evidence_linkage": self.fact_evidence_linkage,
            "claim_citation_linkage": self.claim_citation_linkage,
            "contextual_complete": self.contextual_complete,
            "standalone_complete": self.standalone_complete,
            "factual_bindings": self.factual_bindings,
            "structural_safety": self.structural_safety,
            "rendered_values_bound": self.rendered_values_bound,
            "reason_codes": list(self.reason_codes),
        }


class ResponseDisposition(str, Enum):
    """Canonical terminal response state shared by final gate and API."""

    ANSWER = "ANSWER"
    ABSTAIN = "ABSTAIN"
    CLARIFY = "CLARIFY"
    DENY = "DENY"


_RESPONSE_MODE_TO_DISPOSITION = {
    "answer": ResponseDisposition.ANSWER,
    "abstain": ResponseDisposition.ABSTAIN,
    "clarify": ResponseDisposition.CLARIFY,
    # Retain the older API spelling as a compatibility input only.  The
    # terminal contract and trace use DENY.
    "unsupported_refusal": ResponseDisposition.DENY,
    "deny": ResponseDisposition.DENY,
}


@dataclass(frozen=True)
class ResponseDecision:
    """Common classification for every serialized response."""

    mode: str
    reason_codes: tuple[str, ...]
    acceptance_decision: Optional[AcceptanceDecision] = None
    disposition: ResponseDisposition = field(init=False)

    def __post_init__(self) -> None:
        normalized_mode = str(self.mode or "").casefold()
        disposition = _RESPONSE_MODE_TO_DISPOSITION.get(normalized_mode)
        if disposition is None:
            raise ValueError("invalid response decision mode")
        canonical_mode = {
            ResponseDisposition.ANSWER: "answer",
            ResponseDisposition.ABSTAIN: "abstain",
            ResponseDisposition.CLARIFY: "clarify",
            ResponseDisposition.DENY: "deny",
        }[disposition]
        object.__setattr__(self, "mode", canonical_mode)
        object.__setattr__(self, "disposition", disposition)
        if disposition is ResponseDisposition.ANSWER and not (
            self.acceptance_decision is not None
            and self.acceptance_decision.accepted
        ):
            raise ValueError("answer response requires accepted AcceptanceDecision")
        if (
            disposition is not ResponseDisposition.ANSWER
            and self.acceptance_decision is not None
            and self.acceptance_decision.accepted
        ):
            raise ValueError("non-answer response cannot carry accepted decision")

    @property
    def is_answer(self) -> bool:
        return self.disposition is ResponseDisposition.ANSWER

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "disposition": self.disposition.value,
            "reason_codes": list(self.reason_codes),
            "acceptance_decision": (
                self.acceptance_decision.as_dict()
                if self.acceptance_decision is not None else None
            ),
        }


def decide_acceptance(
    contract: RequestExecutionContract,
    answer: str,
    claim_graph: ClaimGraph,
    typed_facts: Sequence[Any],
    citations: Sequence[Any],
    *,
    structural_safety: bool = True,
) -> AcceptanceDecision:
    """Evaluate every answering route through the same fact-bound gate."""
    from workbook_project import WorkbookProjectEvidence
    if (len(contract.requirements) == 1
            and isinstance(getattr(claim_graph, 'reading_evidence', None), WorkbookProjectEvidence)):
        from document_reading import reading_acceptance
        return reading_acceptance(contract, answer, claim_graph, typed_facts, citations,
                                  structural_safety=structural_safety)
    if (len(contract.requirements) == 1
            and contract.requirements[0].kind == 'conversation_original_file'):
        from conversation_documents import original_file_acceptance
        return original_file_acceptance(contract, answer, claim_graph, typed_facts, citations,
                                        structural_safety=structural_safety)
    if (len(contract.requirements) == 1
            and contract.requirements[0].kind in {'document_contents', 'document_summary', 'financial_records', 'entity_sections', 'document_topic', 'document_reference', 'conversation_document_reference', 'document_type_listing', 'subject_documents', 'subject_work', 'technical_specification', 'contact_lookup'}):
        from document_reading import reading_acceptance
        return reading_acceptance(contract, answer, claim_graph, typed_facts, citations,
                                  structural_safety=structural_safety)
    requirements = {item.id: item for item in contract.requirements}
    facts_by_id = {
        str(getattr(fact, "fact_id", "") or ""): fact
        for fact in typed_facts if getattr(fact, "fact_id", "")
    }
    linked_fact_ids = {
        fact_id for claim in claim_graph.claims for fact_id in claim.fact_ids
    }
    linked_facts = tuple(
        facts_by_id[fact_id]
        for fact_id in linked_fact_ids if fact_id in facts_by_id
    )
    mandatory_coverage = not claim_graph.missing_requirement_ids
    graph_contract_bound = bool(
        claim_graph.contract_id == contract.contract_id
        and claim_graph.contract_sha256 == contract.contract_sha256
    )
    claim_fact_linkage = graph_contract_bound and bool(claim_graph.claims) and all(
        claim.supported and claim.requirement_ids and claim.fact_ids
        and set(claim.fact_ids) <= set(facts_by_id)
        for claim in claim_graph.claims
    )
    fact_evidence_linkage = bool(linked_facts) and all(
        bool(getattr(fact, "evidence_id", "")) for fact in linked_facts
    )
    citation_ids = {
        str(getattr(item, "id", "") or getattr(item, "citation_number", ""))
        for item in citations
    } - {""}
    claim_citation_linkage = bool(citation_ids) and all(
        claim.citation_ids
        and set(claim.citation_ids) <= citation_ids
        and any(
            binding.claim_id == claim.claim_id and binding.direct
            for binding in claim_graph.citation_bindings
        )
        for claim in claim_graph.claims
    )
    factual_bindings = bool(linked_facts)
    for claim in claim_graph.claims:
        for requirement_id, fact_id in zip(claim.requirement_ids, claim.fact_ids):
            requirement = requirements.get(requirement_id)
            fact = facts_by_id.get(fact_id)
            if requirement is None or fact is None:
                factual_bindings = False
                continue
            if str(getattr(fact, "fact_type", "numeric")) == "text_proposition":
                factual_bindings = factual_bindings and not text_fact_binding_errors(
                    requirement, fact
                )
            elif numeric_fact_binding_errors(requirement, fact, typed_facts=typed_facts,
                    requirements=contract.requirements,
                    authorized_evidence_ids={binding.evidence_unit_id for binding in claim_graph.citation_bindings if binding.direct}):
                factual_bindings = False
    # Bind the graph to the bytes actually being released, not an earlier
    # accepted rendering. Markers may differ only in display whitespace.
    if (any(item.predicate in DOCUMENT_ROLE_PREDICATES for item in contract.requirements)
            or any(str(getattr(fact, "evidence_predicate", "")) in DOCUMENT_ROLE_PREDICATES for fact in linked_facts)):
        def material_role_text(value: str) -> tuple[str, ...]:
            return tuple(_document_role_body(item) for item in
                re.split(r"[.!?׃;](?:\s+|$)|\n+", _CLAIM_CITATION_RE.sub(" ", value or ""))
                if _document_role_body(item))
        graph_text = tuple(item for claim in claim_graph.claims for item in material_role_text(claim.text))
        if material_role_text(answer) != graph_text:
            claim_fact_linkage = False
    if any(getattr(fact, "evidence_population", "") for fact in linked_facts):
        def material_scoped_text(value: str) -> tuple[str, ...]:
            body = _CLAIM_CITATION_RE.sub(" ", value or "")
            body = re.sub(r"(?m)^\s*(?:[-*•]\s+|\d{1,2}[.)]\s+)", " ", body)
            return tuple(normalize_semantic_text(clause).strip() for clause in
                         re.split(r"[.!?׃;](?:\s+|$)|\n+", body)
                         if _numeric_values(clause) or rendered_direction_mentioned(clause))
        graph_text = tuple(clause for claim in claim_graph.claims for clause in material_scoped_text(claim.text))
        if material_scoped_text(answer) != graph_text:
            claim_fact_linkage = False
    rendered_relation_errors = rendered_document_role_errors(
        answer, linked_facts, contract.requirements,
        cited_evidence_ids={binding.evidence_unit_id for binding in claim_graph.citation_bindings
                           if binding.direct and binding.citation_id in citation_ids},
    )
    factual_bindings = factual_bindings and not rendered_relation_errors
    from document_query import rendered_required_location_errors
    from ordered_list_evidence import rendered_ordered_list_errors
    rendered_values_bound = not (
        rendered_fact_value_errors(answer, linked_facts)
        or rendered_ordered_list_errors(answer, linked_facts, contract.requirements)
        or rendered_required_location_errors(answer, linked_facts, contract.requirements)
    )
    contextual_complete = bool(
        claim_graph.contextual_factual_correctness
        and mandatory_coverage
        and factual_bindings
        and rendered_values_bound
    )
    standalone_required = "self_contained" in contract.qualifiers
    effective_structural_safety = bool(
        structural_safety and answer and balanced_semantic_delimiters(answer)
    )
    components = {
        "mandatory_requirement_coverage": mandatory_coverage,
        "claim_fact_linkage": claim_fact_linkage,
        "fact_evidence_linkage": fact_evidence_linkage,
        "claim_citation_linkage": claim_citation_linkage,
        "contextual_complete": contextual_complete,
        "factual_bindings": factual_bindings,
        "structural_safety": effective_structural_safety,
        "rendered_values_bound": rendered_values_bound,
    }
    reasons = [name for name, passed in components.items() if not passed]
    reasons.extend(rendered_relation_errors)
    if not graph_contract_bound:
        reasons.append("claim_graph_contract_identity_incompatible")
    if standalone_required and not claim_graph.standalone_completeness:
        reasons.append("standalone_completeness_required")
    return AcceptanceDecision(
        accepted=not reasons,
        mandatory_requirement_coverage=mandatory_coverage,
        claim_fact_linkage=claim_fact_linkage,
        fact_evidence_linkage=fact_evidence_linkage,
        claim_citation_linkage=claim_citation_linkage,
        contextual_complete=contextual_complete,
        standalone_complete=claim_graph.standalone_completeness,
        factual_bindings=factual_bindings,
        structural_safety=effective_structural_safety,
        rendered_values_bound=rendered_values_bound,
        reason_codes=tuple(reasons),
    )


def balanced_semantic_delimiters(value: str) -> bool:
    pairs = (("(", ")"), ("[", "]"), ("{", "}"))
    return all(value.count(left) == value.count(right) for left, right in pairs)


def evaluate_response_contract(
    contract: CanonicalSemanticContract,
    answer: str,
    citations: Sequence[Any],
    *,
    operator_accepted: bool = False,
) -> ResponseContractResult:
    claims = _material_claims(answer)
    citation_map = {_citation_id(item): item for item in citations if _citation_id(item)}
    results: list[FieldEntailment] = []
    relevance_ok = bool(claims)
    binding_ok = bool(citations)
    for requirement in contract.requirements:
        if requirement.kind in {"category_set", "category_value_bindings", "common_metric_unit_period_population"}:
            passed = bool(operator_accepted)
            results.append(FieldEntailment(
                requirement.id, None, (), passed, passed, passed, passed,
                passed, passed, passed, passed, passed, passed,
                (() if passed else ("operator_typed_coverage_missing",)),
            ))
            relevance_ok = relevance_ok and passed
            binding_ok = binding_ok and passed
            continue
        evaluated_claims: list[
            tuple[int, tuple[bool, ...], tuple[str, ...], bool]
        ] = []
        for index, claim in enumerate(claims):
            checks = _claim_field_checks(requirement, claim)
            marker_ids = tuple(
                match.group("id") for match in _CLAIM_CITATION_RE.finditer(claim)
            )
            cited_text = " ".join(
                _citation_text(citation_map[item])
                for item in marker_ids if item in citation_map
            )
            evidence_checks = (
                _claim_field_checks(requirement, cited_text)
                if cited_text else (False,) * 9
            )
            claim_values = _numeric_values(claim)
            evidence_values = _numeric_values(cited_text)
            numeric_bound = not claim_values or claim_values <= evidence_values
            proposition_bound = _overlap(
                _CLAIM_CITATION_RE.sub(" ", claim), cited_text
            ) >= 0.10
            entailed = (
                bool(marker_ids) and all(evidence_checks)
                and numeric_bound and proposition_bound
            )
            evaluated_claims.append((index, checks, marker_ids, entailed))
        if requirement.kind in {"list", "list_attributes"}:
            satisfied = [
                item for item in evaluated_claims
                if all(item[1]) and item[3]
            ]
            required_items = max(1, requirement.cardinality)
            passed = len(satisfied) >= required_items
            citation_ids = _unique(
                citation_id for item in satisfied for citation_id in item[2]
            )
            checks = (passed,) * 9
            reasons = () if passed else (
                "requested_cardinality_or_item_attribute_missing",
            )
            results.append(FieldEntailment(
                requirement.id,
                satisfied[0][0] if satisfied else None,
                citation_ids,
                *checks,
                passed,
                reasons,
            ))
            relevance_ok = relevance_ok and passed
            binding_ok = binding_ok and passed
            continue
        best: Optional[tuple[int, int, tuple[bool, ...], tuple[str, ...], bool]] = None
        for index, checks, marker_ids, entailed in evaluated_claims:
            score = sum(checks) + int(entailed) * 14
            if best is None or score > best[0]:
                best = (score, index, checks, marker_ids, entailed)
        checks = best[2] if best else (False,) * 9
        marker_ids = best[3] if best else ()
        entailed = best[4] if best else False
        claim_passed = all(checks)
        passed = claim_passed and entailed
        reasons: list[str] = []
        labels = ("subject", "predicate", "metric", "value", "unit", "period", "population", "polarity", "qualifier")
        reasons.extend(f"{label}_incompatible" for label, check in zip(labels, checks) if not check)
        if not marker_ids:
            reasons.append("claim_citation_missing")
        elif not entailed:
            reasons.append("citation_field_entailment_failed")
        results.append(FieldEntailment(
            requirement.id,
            best[1] if best else None,
            marker_ids,
            *checks,
            passed,
            _unique(reasons),
        ))
        relevance_ok = relevance_ok and claim_passed
        binding_ok = binding_ok and entailed
    reason_codes: list[str] = []
    if not relevance_ok:
        reason_codes.append("mandatory_requirement_not_satisfied")
    if not binding_ok:
        reason_codes.append("material_claim_not_bound_to_typed_evidence")
    passed = relevance_ok and binding_ok and all(item.passed for item in results if any(
        req.id == item.requirement_id and req.mandatory for req in contract.requirements
    ))
    return ResponseContractResult(
        passed=passed, relevance_passed=relevance_ok,
        citation_entailment_passed=binding_ok, claims=claims,
        requirements=tuple(results), reason_codes=_unique(reason_codes),
    )
