"""Request-local lookup of identified entities inside authorized source sections.

This is evidence selection, not an answer or source-authorization boundary.
Callers supply already authorized indexed rows and still validate returned hits.
Every returned passage is a literal span of one input chunk with its own page.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import re
import unicodedata

from document_reading import ReadingEvidence
from document_query import ENTITY_ROLE_PATTERNS, role_bound_identifier_spans


_KINDS = {kind: ENTITY_ROLE_PATTERNS[kind]
          for kind in ('room', 'equipment', 'product', 'procedure', 'clause')}
_ID = r'[A-Za-z0-9]+(?:[.\-/][A-Za-z0-9]+)*'
_NUMBER = r'(?:number|no\.?|מספר|מס[\'׳.]?|#)'
_HE = r'[\u05d0-\u05ea]'
_BREADCRUMB = re.compile(r'\A(?:[ \t]*\[[^\]\r\n]{1,500}\][ \t]*(?:\r?\n|$))+')


def _words(value):
    value = unicodedata.normalize('NFKC', str(value or '')).casefold()
    return tuple(re.findall(r'[^\W_]+', value, re.UNICODE))


@dataclass(frozen=True)
class EntityReference:
    kind: str
    identifier: str
    project: str = ''
    descriptor: tuple[str, ...] = ()
    floor: str = ''


@dataclass(frozen=True)
class EntityGroupReference:
    """Explicit comparison subjects, each retaining its own section owner."""
    members: tuple[EntityReference, ...]
    operation: str = 'comparison'


@dataclass(frozen=True)
class EntityEvidence(ReadingEvidence):
    reference: object | None = None
    question: str = ''
    selection_complete: bool = True
    native_rows: tuple[dict, ...] = ()
    # A request-local presentation restriction established by the actual
    # quote verifier. Source selection and its complete witness stay intact.
    quote_eligible_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class PresentationUnit:
    """A literal display block and all of its original passage identities.

    Consolidation changes presentation only. It does not claim that two
    source files are identical, and it never replaces a citation identifier.
    """
    text: str
    citation_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    context: tuple[str, ...] = ()


def _presentation_key(value):
    """Normalize extraction layout while retaining material symbol case."""
    value = unicodedata.normalize('NFKC',str(value or ''))
    value = re.sub(r'(?<=[A-Za-z0-9])-\s+(?=[A-Za-z0-9])','-',value)
    value = re.sub(r'\s*([(),;])\s*',r'\1',' '.join(value.split()))
    return re.sub(r'\s*([–—-])\s*',r'\1',value)


def presentation_units(evidence):
    """Consolidate exact repeated blocks only under the same literal owner.

    A heading scopes following blocks within its own source. Equal quantities
    under different headings remain separate facts; unproved scope also stays
    source-local. Every original passage and citation remains in evidence.
    """
    contexts = {}
    source_tokens = {}
    output, positions = [], {}
    eligible = getattr(evidence,'quote_eligible_ids',None)
    passages = tuple(p for p in evidence.passages if eligible is None or p.citation_id in eligible)
    hyphen_lists = {p.source_id for p in passages if re.search(
        r'(?m)^[ \t]*-[ \t]+(?=[A-Za-z\u05d0-\u05ea])', p.source_claim)}
    def strip_marker(text, source=None):
        pattern = r'(?m)^\s*[*•](?:\s+|$)|^\s*-[ \t]+(?=[A-Za-z\u05d0-\u05ea])'
        if source in hyphen_lists:
            pattern = r'(?m)^\s*[-*•](?:\s+|$)'
        return re.sub(pattern, '', text)
    def key(text, source=None):
        return _presentation_key(strip_marker(text,source))
    def tokens(text, source=None):
        # Overlap proves copied content, so signs, technical identifiers,
        # numeric punctuation and qualifier/unit symbols remain material.
        return tuple(re.findall(r'[+-]?\d+(?:[.,/]\d+)*|[^\W_]+(?:[-/][^\W_]+)*|'
            r'[%₪$€°"″\'′<>≤≥=]',key(text,source),re.UNICODE))
    def counted(text):
        return bool(re.search(r'(?<![\w])[+-]?(?:\d+(?:[.,]\d+)?)(?![\w])|\b[xX]\d+\b|'
            r'\b(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|'
            r'thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|hundred)\b|'
            r'(?<!\w)(?:אחד|אחת|שני|שתי|שניים|שתיים|שלושה|שלוש|ארבעה|ארבע|חמישה|חמש|'
            r'שישה|שש|שבעה|שבע|שמונה|תשעה|תשע|עשרה|עשר|עשרים|שלושים|ארבעים|חמישים|מאה)(?!\w)', text, re.I))
    def heading_key(text):
        value = re.sub(r'^\s*[-*•]\s+', '', text).strip()
        if (not value or '\n' in value or len(value)>180 or len(value.split())>12
                or counted(value) or re.search(r'[.!?;]$', value)
                or re.search(r'\b(?:is|are|has|have|includes?|contains?)\b|(?<!\w)(?:יש|אין|כולל\w*|מכיל\w*)(?!\w)', value, re.I)):
            return ''
        return key(value)
    def blocks(text):
        ranges = [(match.start(), match.end()) for match in re.finditer(
            r'\S[\s\S]*?(?=\n[ \t]*\n|\n[ \t]*[-*•]\s+|\Z)', text)]
        joined = []
        for a,b in ranges:
            previous = text[joined[-1][0]:joined[-1][1]].strip() if joined else ''
            if joined and (re.fullmatch(r'[-*•]\s*',previous)
                    or re.fullmatch(r'[-*•]?\s*[+-]?\d+(?:[.,]\d+)?\s*',previous)
                    or re.fullmatch(ENTITY_ROLE_PATTERNS['floor'],previous,re.I)
                       and re.fullmatch(r'[+-]?\d+',text[a:b].strip())
                    or previous.endswith(('–','—','-'))):
                joined[-1] = (joined[-1][0], b)
            else:
                joined.append((a,b))
        return [text[a:b] for a,b in joined]
    for passage in passages:
        metadata = passage.metadata
        source = passage.source_id
        owner = (str(metadata.get('_entity_kind') or ''),
                 str(metadata.get('_entity_identifier') or ''),
                 str(metadata.get('_entity_project') or ''))
        prior_owner, heading, prior_index = contexts.get(source, (None, '', None))
        index = metadata.get('chunk_index')
        previous = source_tokens.setdefault(source, [])
        if owner != prior_owner:
            heading = ''
            previous = []
        text = passage.source_claim
        current = tokens(text,source)
        adjacent = (owner == prior_owner and type(index) is int
                    and type(prior_index) is int and index == prior_index+1)
        overlap = next((length for length in range(min(len(previous),len(current)),7,-1)
            if tuple(item[0] for item in previous[-length:]) == current[:length]),0) if adjacent else 0
        copied_prefix = bool(overlap)
        if overlap:
            heading = previous[-overlap][1]
        # Only the immediately preceding original chunk may prove overlap.
        # Earlier matching content can belong to another location or owner.
        if index != prior_index:
            previous = []
        literal_blocks = blocks(text)
        for block_index, block in enumerate(literal_blocks):
            clean = strip_marker(block,source)
            references = role_bound_identifier_spans(clean)
            owns = any(item.kind == owner[0] and item.identifier == owner[1]
                       for item in references)
            candidate_heading = heading_key(clean)
            if owns:
                heading = ''
            elif candidate_heading:
                heading = candidate_heading
            # An overlapping chunk can start in the middle of an already
            # selected clause. Its short prefix fragment adds no fact; the
            # long exact overlap proves its original position and owner.
            if (copied_prefix and block_index == 0 and len(tokens(clean,source)) <= 3
                    and not counted(clean) and not owns and not candidate_heading):
                previous.extend((token,heading) for token in tokens(clean,source))
                continue
            # Without an explicit matching owner/heading, equal strings from
            # unrelated sources do not prove equivalent location-bound facts.
            scope = (owner, heading) if any(owner) and (owns or heading) else (
                owner, source, passage.citation_id, block_index)
            identity = (scope, key(clean,source))
            if identity in positions:
                unit_index = positions[identity]
                unit = output[unit_index]
                output[unit_index] = replace(unit,
                    citation_ids=tuple(dict.fromkeys((*unit.citation_ids, passage.citation_id))),
                    source_ids=tuple(dict.fromkeys((*unit.source_ids, source))))
            else:
                positions[identity] = len(output)
                output.append(PresentationUnit(block, (passage.citation_id,), (source,),
                                               (heading,) if heading else ()))
            previous.extend((token,heading) for token in tokens(clean,source))
        contexts[source] = (owner, heading, index)
        source_tokens[source] = previous
    return tuple(output)


@dataclass(frozen=True)
class CountOperand:
    quantity: int
    label: str
    context: tuple[str, ...]
    citation_ids: tuple[str, ...]
    source_ids: tuple[str, ...]


@dataclass(frozen=True)
class SectionCountProof:
    operands: tuple[CountOperand, ...]
    total: int


def section_count_proof(evidence):
    """Recompute a count only from complete, unambiguous connection fields.

    This is deliberately narrower than arbitrary numeric reasoning. The
    question must request a count, every operand retains a literal location,
    and all available source representations must agree on the location map.
    No quantities from headings, models, measurements, or conditional fields
    become operands. Callers still own source authorization and quote checks.
    """
    reference = getattr(evidence, 'reference', None)
    query = getattr(evidence, 'question', '')
    if (not isinstance(reference, EntityReference) or not evidence.complete
            or not getattr(evidence, 'selection_complete', True)):
        return None
    eligible = getattr(evidence,'quote_eligible_ids',None)
    if eligible is not None and set(eligible) != {p.citation_id for p in evidence.passages}:
        # An unavailable unique field may be another operand. Exact quoted
        # fields can still be displayed, but cannot establish a complete sum.
        return None
    count_words = r'\b(?:connections?|ports?|sockets?)\b|(?<!\w)(?:חיבורי|חיבורים|שקעי|שקעים|כניסות)(?!\w)'
    openers = list(re.finditer(r'\bhow\s+many\b|(?<!\w)ו?כמה(?!\w)', query, re.I))
    if len(openers) != 1 or not re.search(count_words, query[openers[0].end():], re.I):
        return None
    topic, count_focus = _requested_focus(query, reference)
    wanted = _topic_terms(topic)
    count_topic = bool(re.search(r'\b(?:connections?|ports?|sockets?)\s*$',topic,re.I)
        or re.match(r'^(?:חיבורי|חיבורים|שקעי|שקעים|כניסות)\s+',topic))
    if (not count_focus or not wanted or not count_topic
            or re.search(r'\band\b|\s+ו(?=[\u05d0-\u05ea])',topic,re.I)):
        return None
    from config_env import hebrew_token_variants
    def forms(word):
        return set(hebrew_token_variants(word)) if re.fullmatch(_HE+'+',word) else {word}
    def matches(a,b):
        return bool(forms(a)&forms(b))
    def is_topic(value):
        observed = _words(value)
        return all(any(matches(term,word) for word in observed) for term in wanted)
    def same_label(value):
        observed = _topic_terms(value)
        return len(observed)==len(wanted) and all(matches(a,b) for a,b in zip(wanted,observed))
    sources = {p.source_id for p in evidence.passages}
    maps = {source:{} for source in sources}
    operands = {}
    hyphen_lists = {p.source_id for p in evidence.passages if re.search(
        r'(?m)^[ \t]*-[ \t]+(?=[A-Za-z\u05d0-\u05ea])', p.source_claim)}
    for unit in presentation_units(evidence):
        value = ' '.join(unicodedata.normalize('NFKC', unit.text).split())
        if value.startswith(('•','*')):
            value = value[1:].strip()
        elif value.startswith('- ') and unit.source_ids[0] in hyphen_lists:
            value = value[2:].strip()
        if not is_topic(value):
            continue
        # A pure topic heading is context, not an incomplete count field.
        if unit.context and _presentation_key(value.rstrip(':')) == _presentation_key(unit.context[-1].rstrip(':')):
            continue
        leading = re.fullmatch(r'(?:[xX]\s*)?(\d+)(?:\s*[xX])?\s+(.+?)\.?',value)
        trailing = re.fullmatch(r'(.+?)\s*:\s*(?:[xX]\s*)?(\d+)(?:\s*[xX])?\.?',value)
        if leading:
            quantity, label = int(leading[1]), leading[2].rstrip('.')
        elif trailing:
            label, quantity = trailing[1], int(trailing[2])
        else:
            return None
        if (not unit.context or not same_label(label)
                or re.search(r'\b(?:if|unless|when|except|provided|subject)\b|'
                    r'(?<!\w)(?:אם|כאשר|למעט|בתנאי|בכפוף)(?!\w)', ' '.join(unit.context), re.I)):
            return None
        context = unit.context
        for source in unit.source_ids:
            previous = maps[source].get(context)
            if previous is not None and previous != quantity:
                return None
            maps[source][context] = quantity
        if context in operands:
            previous = operands[context]
            if previous.quantity != quantity:
                return None
            operands[context] = replace(previous,
                citation_ids=tuple(dict.fromkeys((*previous.citation_ids,*unit.citation_ids))),
                source_ids=tuple(dict.fromkeys((*previous.source_ids,*unit.source_ids))))
        else:
            operands[context] = CountOperand(quantity,label,context,unit.citation_ids,unit.source_ids)
    if not operands or any(not mapping for mapping in maps.values()):
        return None
    baseline = next(iter(maps.values()))
    if any(mapping != baseline for mapping in maps.values()):
        return None
    values = tuple(operands.values())
    return SectionCountProof(values, sum(operand.quantity for operand in values))


def _descriptor_words(value):
    return tuple(word for word in _words(value) if word not in {'the','a','an'})


def _room_descriptions(text):
    """Literal room noun phrases with a separately role-bound floor.

    The floor designation remains a floor. It never becomes a room number.
    A type/size/descriptor and an adjacent floor together identify the room.
    """
    result = []
    for floor in role_bound_identifier_spans(text):
        if floor.kind!='floor':continue
        begin = max(text.rfind('\n',0,floor.entity_span[0])+1,
                    text.rfind(',',0,floor.entity_span[0])+1,floor.entity_span[0]-150)
        prefix = text[begin:floor.entity_span[0]]
        matches = list(re.finditer(r'(?<!\w)(?:[בלמכ]?ה?)?(?:חדר|כיתה|אולם|מעבדה)(?!'+_HE+r')\s+',prefix))
        if matches:
            role = matches[-1]
            raw = prefix[role.end():].strip()
            if not re.fullmatch(_HE+r'+(?:\s+'+_HE+r'+){0,3}',raw):continue
            descriptor = _descriptor_words(raw)
            start = begin+role.start()
        else:
            role = re.search(r'\b(?:room|classroom|hall|lab|laboratory)\b\s*(?:(?:on|at|in)\s+)?(?:the\s+)?$',prefix,re.I)
            if not role:continue
            raw = prefix[:role.start()].strip()
            raw = re.split(r'\b(?:in|at|for|near|beside|of|describe|explain|the)\s+',raw,flags=re.I)[-1].strip()
            if not re.fullmatch(r'[A-Za-z-]+(?:\s+[A-Za-z-]+){0,3}',raw):continue
            descriptor = _descriptor_words(raw)
            start = begin+prefix.rfind(raw,0,role.start())
        if not descriptor:continue
        result.append((start,floor.entity_span[1],descriptor,floor.identifier))
    return tuple(result)


def _descriptor_matches(wanted,observed):
    def equal(a,b):
        return a==b or (re.fullmatch(_HE+'{3,}',a) and re.fullmatch(_HE+'{3,}',b)
                       and (a=='ה'+b or b=='ה'+a))
    return bool(wanted) and len(wanted)==len(observed) and all(equal(a,b) for a,b in zip(wanted,observed))


def entity_reference(query, *, _comparison_member=False):
    """Interpret a role-bound identifier, never a requested result count."""
    value = str(query or '')
    found = [item for item in role_bound_identifier_spans(value) if item.kind in _KINDS]
    comparison = bool(re.search(r'\b(?:compare|comparison|differences?|similarities|differs?)\b|השוו?ה|השוואה|הבדל|לעומת', value, re.I))
    if comparison and not _comparison_member and len(found) != 2:
        # Unparsed or descriptive comparison endpoints belong to the normal
        # resolver; one understood subject cannot stand in for both sides.
        return None
    descriptor,floor = (),''
    if len(found)==1:
        item = found[0]
        kind, identifier = item.kind, item.identifier
        entity_span = item.entity_span
    elif not found:
        described = _room_descriptions(value)
        if len(described)!=1:return None
        a,b,descriptor,floor = described[0]
        kind,identifier,entity_span = 'room','',(a,b)
    else:
        # A comparison names its endpoints explicitly. Bare incidental IDs
        # and an implicit "and 42" are not enough to invent another owner.
        if (len(found) != 2 or found[0].kind != found[1].kind
                or found[0].identifier == found[1].identifier
                or not comparison):
            return None
        # Only a shared project selector is admitted here. Separately scoped
        # endpoints need the ordinary resolver rather than an inferred scope.
        between = value[found[0].entity_span[1]:found[1].entity_span[0]]
        if re.search(r'\b(?:at|in|for)\s+|(?<!\w)ב[\u05d0-\u05ea]{2,}', between, re.I):
            return None
        prefix = value[:found[0].entity_span[0]]
        suffix = value[found[-1].entity_span[1]:]
        # Reuse the single-subject project grammar, never a filename or ID
        # shortcut. The synthetic phrase contains only the parsed endpoints.
        individual = prefix + value[found[0].entity_span[0]:found[0].entity_span[1]] + suffix
        first = entity_reference(individual,_comparison_member=True)
        if not isinstance(first, EntityReference):
            return None
        trailing = entity_reference(value[found[-1].entity_span[0]:])
        if (isinstance(trailing, EntityReference) and trailing.project and first.project
                and _words(trailing.project) != _words(first.project)):
            return None
        return EntityGroupReference(tuple(EntityReference(item.kind, item.identifier.casefold(), first.project)
                                          for item in found))
    prefix, suffix = value[:entity_span[0]].strip(), value[entity_span[1]:].strip()
    # A resolver may combine the verified prior user subject with a new
    # attribute question. Interrogative grammar starts that new clause; it is
    # never part of the project's proper name.
    suffix = re.split(r'\s+(?=(?:how|what|which|why|כמה|מה|איך|כיצד|איזה|אילו)\b)', suffix, maxsplit=1, flags=re.I)[0]
    project = ''
    # Hebrew "At [project], ..." and English equivalents bind metadata to
    # content without requiring the project name inside the matching chunk.
    before = re.match(r'^(?:ב(?P<he>[^,;?]+)|(?:at|in|for)\s+(?P<en>[^,;?]+))\s*[,;]', prefix, re.I)
    if before:
        project = before.group('he') or before.group('en')
    else:
        after = re.search(r'(?:\b(?:at|in|for)\s+(?:project\s+)?|(?<!\w)ב)(?P<project>[^,;?]+?)\s*[?!.]*$', suffix, re.I)
        if after:
            project = after.group('project')
    project = re.sub(r'^(?:פרויקט|אתר|חברת|the\s+project)\s+', '', project, flags=re.I)
    project = project.strip(' \"\'״?.!')
    # Grammatical adjectives after an identifier are not project selectors.
    if project in {'נוי', 'בנוי', 'there', 'it'}:
        project = ''
    return EntityReference(kind, identifier.casefold(), project[:160],descriptor,floor)


def section_reference(query):
    """An explicit document restriction owns its topic before entity lookup."""
    from document_query import document_topic_reference
    return document_topic_reference(query) or entity_reference(query)


def _mapped_source_lines(rows):
    lines,joined=[],''
    for row in rows:
        text=str(row.get('document') or '')
        crumb=_BREADCRUMB.match(text)
        offset=crumb.end() if crumb else 0
        for line in re.finditer(r'[^\n]*(?:\n|$)',text[offset:]):
            if not line.group():continue
            a,b=offset+line.start(),offset+line.end()
            lines.append((len(joined),len(joined)+b-a,row,a,b))
            joined+=text[a:b]
        joined+='\n'
    return lines,joined


def _document_identity_matches(reference,rows):
    from pathlib import PurePosixPath
    from financial_semantics import document_references
    metadata=rows[0].get('metadata') or {}
    labels=[str(metadata.get(key) or '') for key in ('source_relative_path','display_name','title')]
    label_references=[item for label in labels for item in document_references(label)]
    if any(item.number!=reference.identifier or reference.kind!='document' and item.kind!=reference.kind
           for item in label_references):return False
    filename=any(reference.identifier in _words(PurePosixPath(label.replace('\\','/')).stem) for label in labels)
    labelled=bool(label_references)
    _lines,body=_mapped_source_lines(rows)
    opening='\n'.join(body.splitlines()[:40])
    declared=document_references(opening)
    owned=[item for item in declared if reference.kind=='document' or item.kind==reference.kind]
    if declared and not owned:return False
    if owned and any(item.number!=reference.identifier for item in owned):return False
    return bool(filename or labelled or any(item.number==reference.identifier for item in owned))


def _document_scope_matches(reference,rows,scoped):
    if not reference.scope_terms:return True
    wanted=tuple(reference.scope_terms)
    def contains(text):
        observed=_words(text)
        return any(wanted==observed[i:i+len(wanted)]
                   for i in range(len(observed)-len(wanted)+1))
    metadata=rows[0].get('metadata') or {}
    if any(contains(metadata.get(key) or '') for key in ('source_relative_path','title','display_name')):return True
    _,body=_mapped_source_lines(rows)
    if any(contains(match.group(1)) for match in re.finditer(
            r'(?im)^(?:customer|client|project|site|לכבוד|לקוח|פרויקט|אתר)\s*[:–—-]\s*(.+)$',body)):
        return True
    # A scope/customer phrase in the actual topic-owned work is evidence;
    # an incidental mention elsewhere in a broad fallback is not identity.
    if scoped and not scoped[0][2]['metadata'].get('_entity_document_fallback'):
        return contains(' '.join(line[2]['document'][line[3]:line[4]] for line in scoped))
    return False


def _source_matches(reference, rows):
    if not reference.project:
        return True
    metadata = rows[0].get('metadata') or {}
    identity = ' '.join(str(metadata.get(key) or '') for key in
                        ('source_relative_path', 'display_name', 'title'))
    wanted = _words(reference.project)
    def matches(text):
        observed = _words(text)
        return bool(wanted) and any(wanted == observed[i:i+len(wanted)]
                                   for i in range(len(observed)-len(wanted)+1))
    if matches(identity):
        return True
    if reference.descriptor and len(wanted)>1 and all(re.fullmatch(_HE+'+',word) for word in wanted):
        # A Hebrew locative ב contracts the definite article. Require the
        # whole remaining project phrase and definite agreement, plus a
        # source-owned room/floor heading. Never strip arbitrary proper-name
        # letters or use an incidental project mention in document prose.
        if wanted[1].startswith('ה') and len(wanted[1])>3:
            observed = _words(identity)
            contracted = ('ה'+wanted[0],)+wanted[1:]
            if any(contracted==observed[i:i+len(contracted)] for i in range(len(observed)-len(contracted)+1)):
                first = min(rows,key=lambda row:(row.get('metadata') or {}).get('chunk_index',0))
                opening = '\n'.join(str(first.get('document') or '').splitlines()[:24])
                if _entity_pattern(reference).search(opening):return True
    # A renamed file may carry project identity in an explicit opening title.
    # An incidental mention elsewhere in a specification is never ownership.
    first = min(rows, key=lambda row: (row.get('metadata') or {}).get('chunk_index', 0))
    text = str(first.get('document') or '')
    crumb = _BREADCRUMB.match(text)
    text = text[crumb.end():] if crumb else text
    opening = [line.strip() for line in text.splitlines() if line.strip()][:6]
    for line in opening:
        title = re.sub(r'^(?:project|site|customer|client|פרויקט|אתר|לקוח|חברת)\s*[:–—-]?\s+', '', line, flags=re.I)
        if _words(title) == wanted:
            return True
    return False


def _entity_pattern(reference):
    class Located:
        def __init__(self, span):self.span = span
        def start(self):return self.span[0]
        def end(self):return self.span[1]
    class Matcher:
        def search(self, text):
            if reference.descriptor:
                for a,b,descriptor,floor in _room_descriptions(text):
                    before = text[text.rfind('\n',0,a)+1:a].strip(' \t#-*•')
                    after = text[b:text.find('\n',b) if '\n' in text[b:] else len(text)].strip(' \t:.-–—')
                    if (not before and not after and floor==reference.floor
                            and _descriptor_matches(reference.descriptor,descriptor)):
                        return Located((a,b))
                return None
            return next((Located(item.entity_span) for item in role_bound_identifier_spans(text)
                         if item.kind == reference.kind and item.identifier == reference.identifier), None)
    return Matcher()


def _governing_floor_lines(lines, joined, anchor_start):
    """Map the latest literal floor header without borrowing body mentions.

    Native layout may split its label/designation or join a floor header to
    the next room heading. The role parser supplies the exact identifier;
    line-leading position and an empty suffix/child heading prove structure.
    """
    candidates = []
    for floor in role_bound_identifier_spans(joined):
        if floor.kind != 'floor' or floor.entity_span[1] > anchor_start:
            continue
        start,end = floor.entity_span
        beginning = joined.rfind('\n',0,start)+1
        following = joined.find('\n',end)
        following = len(joined) if following<0 else following
        prefix = joined[beginning:start].strip(' \t#-*•')
        suffix = joined[end:following].strip(' \t:–—-,')
        child = bool(re.match(r'^(?:[בלמכ]?ה?)?'+_KINDS['room']+
            r'(?![A-Za-z\u05d0-\u05ea])',suffix,re.I))
        if prefix or suffix and not child:
            continue
        mapped = []
        for low,high,row,a,b in lines:
            x,y = max(low,start),min(high,end)
            if x<y and joined[x:y].strip():
                if mapped and mapped[-1][2] is row:
                    previous = mapped[-1]
                    mapped[-1] = (previous[0],y,row,previous[3],a+y-low)
                else:
                    mapped.append((x,y,row,a+x-low,a+y-low))
        if mapped:
            candidates.append((start,tuple(mapped)))
    # Preserve named floor headings accepted by the existing grammar. These
    # supply literal context only; no arbitrary adjacent scalar is inferred.
    named = r'^(?:floor|קומה)\s*(?:[+-]?\d+|[A-Za-z]|'+_HE+r'+)\s*$'
    for line in lines:
        if line[1] <= anchor_start and re.fullmatch(named,joined[line[0]:line[1]].strip(),re.I):
            candidates.append((line[0],(line,)))
    return max(candidates,key=lambda item:item[0])[1] if candidates else ()


def _section_heading(text, reference):
    value = text.strip().strip('#').strip()
    value = re.sub(r'^[-*•]\s+[^:\n]{1,160}:\s*', '', value)
    if not value or len(value) > 180:
        return False
    if reference.kind=='room' and _room_descriptions(value):return True
    # The role must govern the heading. "Conduit to the room" is body text.
    # A spatial section owns its child equipment/model headings. Only a new
    # peer spatial entity or floor ends that ownership. Other entity kinds
    # similarly stop at their own peers, or at a new enclosing space.
    role = _KINDS[reference.kind]
    if reference.kind != 'room':role = '(?:' + role + '|' + _KINDS['room'] + ')'
    if re.match(r'^(?:\d+(?:\.\d+)*[.)]?\s+)?(?:[בלמכ]?ה?)?' + role + r'(?![A-Za-z\u05d0-\u05ea])', value, re.I):
        return True
    if reference.kind == 'room' and re.fullmatch(r'(?:[A-Za-z]+\s+){1,3}room\s*[-:–—]?', value, re.I):
        return True
    floor = next((item for item in role_bound_identifier_spans(value)
        if item.kind=='floor' and item.entity_span[0]==0),None)
    if floor is not None:
        suffix = value[floor.entity_span[1]:].strip(' \t:–—-,')
        if not suffix or re.match(r'^(?:[בלמכ]?ה?)?'+_KINDS['room']+
                r'(?![A-Za-z\u05d0-\u05ea])',suffix,re.I):
            return True
    return bool(re.match(r'^(?:floor|קומה)\s*(?:[+-]?\d+|[A-Za-z]|' + _HE + r'+)\s*$', value, re.I))


def numbered_property_focus(query, reference):
    """Read a bounded property question about one explicitly identified device.

    Preserve every stated property and require the optional second clause to
    refer back to the same entity. Other question shapes keep the ordinary
    coverage grammar rather than silently dropping restrictions.
    """
    if getattr(reference, 'kind', '') not in {'product', 'equipment'}:
        return ()
    identifiers = [item for item in role_bound_identifier_spans(query)
                   if item.kind == reference.kind and item.identifier == reference.identifier]
    if len(identifiers) != 1:
        return ()
    item = identifiers[0]
    prefix = query[:item.entity_span[0]].strip()
    first = re.fullmatch(r'(?:[^,;?]+,\s*)?(?:מהו|מהי)\s+([\u05d0-\u05ea]+)\s+'
                         r'[\u05d0-\u05ea]+(?:\s+[\u05d0-\u05ea]+){0,3}\s*', prefix)
    suffix = query[item.entity_span[1]:].strip()
    second = re.fullmatch(r'(?:ו?מה(?:ו|י)?\s+([\u05d0-\u05ea]+)\s+'
                          r'(?:שלו|שלה|שלהם|שלהן))?\s*[?!.]*', suffix)
    if not first or not second:
        return ()
    return (first.group(1),) + ((second.group(1),) if second.group(1) else ())


def _requested_focus(query, reference):
    """Extract a literal requested child/property, without naming domains.

    The spatial identifier selects its parent section. An interrogative object
    such as "which controller" or "how many network connections" selects a
    child row/property inside that parent. Broad descriptions keep the parent.
    """
    properties = numbered_property_focus(query, reference)
    if properties:
        return ' ו'.join(properties), False
    value = str(query or '')
    value = re.sub(r'\b(what|which)\s+(?:(?:is|are|was|were)\s+)?(?:the\s+)?(?:(?:specified|planned|proposed|offered)\s+)?',
                   r'\1 ',value,flags=re.I)
    quantities = list(re.finditer(
        r'(?:\bhow\s+many\s+|(?<!\w)ו?כמה\s+)(?P<topic>[^,;?]{1,100}?)'
        r'(?=\s+(?:are|is|were|was|specified|planned|there|מתוכנ\w*|מוגדר\w*|מפורט\w*|נדרש\w*|יש|שם)\b|[?;]|$)', value, re.I))
    nominal = list(re.finditer(
            r'(?:\b(?:which|what)\s+(?:the\s+)?|(?<!\w)(?:אילו|איזה|איזו|מהו|מהי)\s+)(?P<topic>[^,;?]{1,100}?)'
            r'(?=\s+(?:is|are|was|were|for|in|planned|proposed|specified|offered|מתוכנ\w*|מוצע\w*|נדרש\w*|מיועד\w*|מוגדר\w*|מופיע\w*)\b)', value, re.I))
    descriptions = list(re.finditer(r'\b(?:describe|explain)\s+(?:the\s+)?(?P<topic>[^,;?]{1,80}?)\s+(?:in|for)\s+',value,re.I))
    # A possessive property question names its owner separately: "the width
    # of model AX-8" / "מהו רוחב לוח המגע מדגם AX-8".  The construct's
    # first noun is the requested attribute; the identified section supplies
    # its object. Do not require that object's words in each property leaf.
    properties = list(re.finditer(
        r'\b(?:what|which)\s+(?:the\s+)?(?P<topic>[^,;?]{1,60}?)\s+of\b',value,re.I))
    reference_spans = [item for item in role_bound_identifier_spans(value)
                       if item.kind==reference.kind and item.identifier==reference.identifier]
    if reference_spans:
        prefix = value[:reference_spans[0].entity_span[0]]
        properties += list(re.finditer(
            r'(?<!\w)(?:מהו|מהי)\s+(?P<topic>[\u05d0-\u05ea]+)\s+(?=[\u05d0-\u05ea])',prefix))
    candidates = [(match.start(),match.group('topic'),True) for match in quantities]
    candidates += [(match.start(),match.group('topic'),False) for match in nominal+descriptions+properties]
    for _,topic,count in sorted(candidates,key=lambda item:(item[0],-len(item[1])),reverse=True):
        topic = ' '.join(topic.split()).strip(' :-.')
        topic = re.sub(r'^(?:(?:is|are|was|were|the|a|an|planned|proposed|specified|offered)\s+)+','',topic,flags=re.I)
        topic = re.sub(r'^(?:capabilities|features|specifications|attributes)\s+of\s+(?:the\s+)?','',topic,flags=re.I)
        topic = re.sub(r'^(?:ה?יכולות|ה?מאפיינים|ה?תכונות)\s+של\s+','',topic)
        if (not topic or topic.casefold() in {'capabilities','features','specifications','attributes','abilities','יכולות','היכולות','מאפיינים','המאפיינים'}
                or role_bound_identifier_spans(topic) or re.fullmatch(_KINDS[reference.kind],topic,re.I)):
            continue
        return topic,count
    return '',False


def _requested_topic(query, reference):
    return _requested_focus(query,reference)[0]


def _topic_terms(topic):
    # Connection/port wording supplies a requested count's grammatical unit;
    # its concrete property label (network/electric/etc.) owns source matching.
    generic = {'the','a','an','connection','connections','port','ports',
               'and','or','&','ו','חיבור','חיבורים','חיבורי','החיבורים','חיבורי','כניסות'}
    return tuple(word for word in _words(topic) if word not in generic)


_DOTTED_SECTION = re.compile(
    r'(?<![\w./-])(?:\.[ \t]*)?(?P<number>\d{1,3}(?:\.\d{1,3})+)(?![\w/-]|\.\d)\.?')


def _numbered_context(text, anchor, reference):
    """Admit a hierarchy only with an explicit parent and sibling children.

    A dotted scalar/version alone supplies no section ownership. A model in
    an explicitly numbered child inherits its nearest declared parent, whose
    next peer/ancestor ends the section, including inline extraction joins.
    """
    if reference.kind not in {'equipment','product'}:return None
    markers = []
    identifiers = [item.identifier_span for item in role_bound_identifier_spans(text)
                   if item.kind in {'equipment','product'}]
    for match in _DOTTED_SECTION.finditer(text):
        a,b = match.span('number')
        if any(a<end and start<b for start,end in identifiers):continue
        prefix = text[text.rfind('\n',0,match.start())+1:match.start()]
        if prefix.strip() and not match.group().startswith('.') and not match.group().endswith('.'):
            continue
        markers.append((match.start(),match.end(),tuple(int(n) for n in match.group('number').split('.'))))
    before = [item for item in markers if item[0]<=anchor.start()]
    if not before:return None
    child = before[-1]
    parents = [item for item in before[:-1] if len(item[2])<len(child[2])
               and child[2][:len(item[2])]==item[2]]
    for parent in sorted(parents,key=lambda item:(len(item[2]),item[0]),reverse=True):
        depth = len(parent[2])
        end = next((item[0] for item in markers if item[0]>child[0]
                    and (len(item[2])<=depth or item[2][:depth]!=parent[2])),len(text))
        children = {item[2][:depth+1] for item in markers
                    if parent[0]<item[0]<end and len(item[2])>depth
                    and item[2][:depth]==parent[2]}
        if len(children)>=2:
            return parent[0],end,tuple(item for item in markers if parent[0]<=item[0]<end)
    return None


def _split_source_lines(lines, cuts):
    """Split mapped literal lines at structural boundaries, without rewriting."""
    result = []
    for low,high,row,a,b in lines:
        offsets = [low]+sorted(cut for cut in cuts if low<cut<high)+[high]
        result.extend((x,y,row,a+x-low,a+y-low) for x,y in zip(offsets,offsets[1:]))
    return result


def _relationship_owner(query,reference):
    if reference is None:return ''
    if getattr(reference,'descriptor',()):
        spans = [(a,b) for a,b,descriptor,floor in _room_descriptions(query)
                 if floor==reference.floor and _descriptor_matches(reference.descriptor,descriptor)]
    else:
        spans = [item.entity_span for item in role_bound_identifier_spans(query)
                 if item.kind==reference.kind and item.identifier==reference.identifier]
    prefix = query[:spans[0][0]] if spans else query
    owners = list(re.finditer(r'(?:\b(?:near|beside|next\s+to)\s+(?:the\s+)?|(?<!\w)(?:ליד|בסמוך\s+ל)\s*)([^,;?]{2,100})',prefix,re.I))
    if not owners:return ''
    owner = owners[-1].group(1).strip()
    return re.sub(r'\s+(?:in|at|for|on)\s+(?:the\s*)?$','',owner,flags=re.I).strip()


def _focused_lines(lines, joined, start, stop, anchor, topic, query='',count_focus=False,numbered=None,reference=None,document_topic=False):
    """Keep source-owned rows/paragraphs plus their governing identity.

    Ordered item numbers establish table/list ownership only when a sequence
    proves that role. A standalone measurement never becomes an item boundary.
    """
    if not topic and not document_topic:return lines[start:stop], ()
    terms = _topic_terms(topic)
    if not terms and not document_topic:return [], ()
    from config_env import hebrew_token_variants
    def forms(word):
        return set(hebrew_token_variants(word)) if re.fullmatch(_HE+'+',word) else {word}
    def relevant(text,requested=terms):
        observed = _words(text)
        def matches(term,word):
            if forms(term)&forms(word):return True
            # Within a requested property label only, compare an attached
            # grammatical clitic with that exact supplied noun. Do not strip
            # arbitrary roots or use these forms for source/project identity.
            if re.fullmatch(_HE+'{3,}',term) and re.fullmatch(_HE+'+',word):
                prefixes = ('ה','ב','ל','כ','בה','לה','כה','ו','וה','וב','ול','וכ','ובה','ולה','וכה')
                return any(word==prefix+term or term==prefix+word for prefix in prefixes)
            return False
        return all(any(matches(term,word) for word in observed) for term in requested)
    if document_topic:
        from source_structure import _semantic_boundary,boundary_join_reason
        from financial_semantics import _money_matches
        low,high=lines[start][0],lines[stop-1][1]
        blocks=[(low+m.start(),low+m.end(),m.group().strip())
                for m in re.finditer(r'\S[\s\S]*?(?=\n[ \t]*\n|\Z)',joined[low:high])]
        matching=[i for i,(_,_,text) in enumerate(blocks) if terms and relevant(text)]
        ranges=[]
        for index in matching:
            a,b,text=blocks[index]
            end=index
            for following in range(index+1,len(blocks)):
                next_text=blocks[following][2]
                explicit=bool(re.match(r'^(?:includes?|including|comprises?|כולל\w*|לרבות)\b',next_text,re.I))
                if (not re.search(r'[A-Za-z\u05d0-\u05ea]',next_text)
                        or _semantic_boundary(next_text) and not explicit or _money_matches(next_text)):
                    break
                previous=blocks[end][2]
                if (len(joined[a:blocks[following][1]])>900
                        or re.search(r'[.!?]["\u05f4\])}]*$',previous) and not explicit
                        and not boundary_join_reason(previous,next_text)):
                    break
                end=following
            ranges.append((a,blocks[end][1]))
        fallback=not ranges
        if fallback:ranges=[(low,high)]
        selected=[];copies={}
        for line in lines[start:stop]:
            if not any(a<line[1] and line[0]<b for a,b in ranges):continue
            a,b,row,sa,sb=line
            if id(row) not in copies:
                copies[id(row)]={**row,'metadata':dict(row.get('metadata') or {},
                    _entity_document_fallback=fallback)}
            selected.append((a,b,copies[id(row)],sa,sb))
        return selected, ()
    owner = _relationship_owner(query,reference)
    if owner:
        from source_structure import _looks_heading
        low,high = lines[start][0],lines[stop-1][1]
        body = joined[low:high]
        blocks = [(low+m.start(),low+m.end(),m.group().strip())
                  for m in re.finditer(r'\S[\s\S]*?(?=\n[ \t]*\n|\Z)',body)]
        headings = []
        domain = None
        for index,(_,_,text) in enumerate(blocks):
            match = re.match(r'^\s*(?:[-*•]\s+)?(?:for\s+(?:the\s+)?|עבור\s+|בשביל\s+)(.+)$',text,re.I)
            if match and len(text)<180 and '\n' not in text:
                headings.append((index,match.group(1),domain))
            elif (text.endswith(':') and len(text)<180 and '\n' not in text
                  and (_looks_heading(text) or 1<=len(text.split())<=12)):
                domain=index
        owner_terms = _descriptor_words(owner)
        if not owner_terms:return [], ()
        head = owner_terms[0] if re.search(_HE,owner) else owner_terms[-1]
        # Coordinated requested labels can occupy separate source paragraphs.
        # Require every label under one owner; never union sibling groups.
        requested = tuple(word[1:] if index and re.fullmatch('ו'+_HE+'{2,}',word) else word
                          for index,word in enumerate(terms))
        coordinated = bool(re.search(r'\band\b|\s+ו'+_HE,topic,re.I))
        facets = tuple((term,) for term in requested) if coordinated else (requested,)
        candidates = []
        for position,(index,label,domain) in enumerate(headings):
            if not relevant(label,(head,)):continue
            candidates.append((relevant(label,owner_terms),position,index,domain))
        exact = [item for item in candidates if item[0]]
        candidates = exact or candidates
        # Section labels can distinguish electrical infrastructure from a
        # later carpentry section with the same object heading. Resolve this
        # ownership using source headings, before considering available data.
        contextual = [item for item in candidates if item[3] is not None
                      and any(relevant(blocks[item[3]][2],facet) for facet in facets)]
        candidates = contextual or candidates
        if len(candidates)!=1:return [], ()
        _,position,heading,domain = candidates[0]
        end = headings[position+1][0] if position+1<len(headings) else len(blocks)
        end = min((i for i in range(heading+1,end) if blocks[i][2].endswith(':')),default=end)
        fields = []
        for facet in facets:
            found = [i for i in range(heading+1,end) if relevant(blocks[i][2],facet)
                     and len(_words(blocks[i][2]))>len(facet)]
            if not found:return [], ()
            fields.extend(found)
        fields=tuple(sorted(set(fields)))
        ranges = [(blocks[heading][0],blocks[heading][1]),(lines[start][0],anchor.end())]
        if domain is not None:ranges.append((blocks[domain][0],blocks[domain][1]))
        ranges.extend((blocks[index][0],blocks[index][1]) for index in fields)
        selected = []
        for line in lines[start:stop]:
            if not any(a<line[1] and line[0]<b for a,b in ranges):continue
            a,b,row,source_start,source_end = line
            if a<=blocks[heading][0]<b:
                decoration = re.match(r'[ \t]*[-*•][ \t]+',joined[a:b])
                if decoration:
                    delta=decoration.end();a+=delta;source_start+=delta
            selected.append((a,b,row,source_start,source_end))
        return selected, ()
    if numbered:
        low,high,markers = numbered
        parent = markers[0][2]
        children = [item for item in markers if len(item[2])==len(parent)+1]
        units = [(item[0],next((other[0] for other in children if other[0]>item[0]),high))
                 for item in children]
        property_focus = numbered_property_focus(query, reference)
        if property_focus:
            # A numbered parent can list several models. A sibling model's
            # fields must not satisfy this model's requested properties.
            anchor_index = next((i for i, (a,b) in enumerate(units) if a <= anchor.start() < b), None)
            if anchor_index is None:
                return [], ()
            def competing(unit):
                a,b = unit
                return any(item.kind in {'product', 'equipment'}
                           and item.identifier != reference.identifier
                           for item in role_bound_identifier_spans(joined[a:b]))
            if competing(units[anchor_index]):
                return [], ()
            previous = [i for i in range(anchor_index) if competing(units[i])]
            following = [i for i in range(anchor_index + 1, len(units)) if competing(units[i])]
            units = units[(previous[-1] + 1 if previous else 0):
                          (following[0] if following else len(units))]
        facets = tuple((term,) for term in property_focus) if property_focus else (terms,)
        selected = [(a,b) for a,b in units if any(relevant(joined[a:b], facet) for facet in facets)]
        if not selected:return [], ()
        # Keep the complete declaration containing the exact model identifier,
        # and the parent title. Each requested leaf retains its own number and
        # literal content; a neighboring model's scalar cannot supply it.
        selected.extend((a,b) for a,b in units if a<=anchor.start()<b)
        if (children and low<children[0][0]
                and re.search(r'[A-Za-z\u05d0-\u05ea]',joined[low:children[0][0]])):
            selected.append((low,children[0][0]))
        return [line for line in lines[start:stop]
                if any(a<line[1] and line[0]<b for a,b in selected)], ()
    # Reuse source-only structural field ownership for ingested spreadsheets
    # and tables. A requested descriptor is its actual labeled field, not all
    # quantity/price/formula fields in the row or an unbound text copy.
    from structural_evidence import canonical_structural_entry
    descriptor_fields = []
    has_structured_fields = False
    seen_rows = set()
    for line in lines[start:stop]:
        row = line[2]
        if id(row) in seen_rows:continue
        seen_rows.add(id(row))
        original = str(row.get('document') or '')
        meta = row.get('metadata') or {}
        entry = canonical_structural_entry(original, source_id=str(meta.get('source_id') or ''),
            candidate_id='entity-'+hashlib.sha256(original.encode()).hexdigest()[:24],
            chunk_id=str(row.get('id') or meta.get('chunk_id') or ''),page=meta.get('page'),metadata=meta)
        owned = set()
        intervals = [(item[3],item[4]) for item in lines[start:stop] if item[2] is row]
        owner_low,owner_high = min(a for a,_ in intervals),max(b for _,b in intervals)
        for unit in entry.structural_units:
            spans = unit.source_spans
            if (not unit.structurally_valid or len(spans)!=1
                    or original[spans[0].start:spans[0].end]!=unit.local_text):continue
            for binding in unit.field_bindings:
                if binding.field_span is None:continue
                a,b = binding.source_span; fa,fb = binding.field_span
                if binding.role=='field':has_structured_fields=True
                numeric_owner = any(other.role=='value' and other.field_span==binding.field_span
                                    for other in unit.field_bindings)
                numeric_literal = bool(re.fullmatch(r'[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)\s*(?:%|[A-Za-z\u05d0-\u05ea]+)?',binding.value.strip()))
                if numeric_literal and not numeric_owner:continue
                if numeric_owner:
                    label = re.sub(r'\s*\([A-Z]{1,3}\)\s*$','',binding.column_role)
                    owned_value = str(binding.value or '').strip()
                    if any(identifier.identifier==owned_value.casefold()
                           and identifier.identifier_span[0]>=len(label)
                           for identifier in role_bound_identifier_spans(label+' '+owned_value)):
                        continue
                    from semantic_contracts import _source_metric_label_establishes_cardinality
                    if not count_focus or not _source_metric_label_establishes_cardinality(binding.column_role):continue
                if (binding.role not in {'field','entity'} or not 0<=fa<=a<b<=fb<=len(unit.local_text)
                        or not binding.column_role or not relevant(binding.column_role+' '+binding.value)
                        or unit.local_text[a:b]!=binding.value
                        or not owner_low<=spans[0].start+fa<spans[0].start+fb<=owner_high
                        or (not count_focus and len(re.findall(r'[A-Za-z\u05d0-\u05ea]',binding.value))<2)
                        or (spans[0].start+a,spans[0].start+b) in owned):continue
                owned.add((spans[0].start+a,spans[0].start+b))
                if numeric_owner:
                    # A count's source label is its semantic owner. Retain the
                    # proven label-to-value literal, excluding only the list
                    # serialization marker, so the normal quote gate can prove
                    # a readable field rather than an unowned scalar.
                    a = binding.label_span[0]
                descriptor_fields.append((row,spans[0].start+fa,spans[0].start+fb,
                                          spans[0].start+a,spans[0].start+b,unit.local_text[a:b]))
    if descriptor_fields:
        keep = [line for line in lines[start:stop]
                if line[0]<anchor.end() or any(line[2] is row and a<line[4] and line[3]<b
                                              for row,a,b,_,_,_ in descriptor_fields)]
        return keep, tuple(descriptor_fields)
    if has_structured_fields:return [], ()
    low, high = lines[start][0], lines[stop-1][1]
    body = joined[low:high]
    blocks = [(low+m.start(), low+m.end(), m.group().strip())
              for m in re.finditer(r'\S[\s\S]*?(?=\n[ \t]*\n|\Z)', body)]
    numeric = [(i,int(text)) for i,(_,_,text) in enumerate(blocks) if re.fullmatch(r'\d{1,3}',text)]
    boundaries = {i for i,(_,_,text) in enumerate(blocks)
                  if re.match(r'^\d{1,3}[.)]\s+\S',text)}
    ordered_table = bool(re.search(r'(?:\b(?:item|serial|no\.?)\b|סדורי).{0,100}(?:\b(?:product|description)\b|מוצר|תיאור|תאור)',body,re.I))
    if ordered_table:
        for position in range(len(numeric)-2):
            triple = numeric[position:position+3]
            if triple[1][1] == triple[0][1]+1 and triple[2][1] == triple[1][1]+1:
                boundaries.update(index for index,_ in triple)
    ranges = []
    # The caller already bounded this parent, so local heading recovery never
    # crosses a new room. Keep the latest explicit numbered subsection, with
    # its child headings, to distinguish wall/table/etc. property ownership.
    for index,(a,b,text) in enumerate(blocks):
        if not relevant(text):continue
        prior = max((i for i in boundaries if i<=index), default=None)
        following = min((i for i in boundaries if i>index), default=None)
        if prior is not None and following is not None:
            first = prior if not re.fullmatch(r'\d{1,3}',blocks[prior][2]) else prior+1
            ranges.append((blocks[first][0],blocks[following-1][1]))
        else:
            following_scalar = (index+1<len(blocks) and bool(re.fullmatch(r'[+-]?(?:\d+(?:[.,]\d+)?|\.\d+)\s*(?:%|[A-Za-z\u05d0-\u05ea]+)?',blocks[index+1][2])))
            previous_marker = (index and bool(re.fullmatch(r'[•▪◦*-]\s*[+-]?\d+(?:[.,]\d+)?',blocks[index-1][2])))
            if count_focus and following_scalar:
                ranges.append((a,blocks[index+1][1]))
            elif _topic_terms(text)==terms and not previous_marker:
                from source_structure import _looks_heading
                following = []
                for _,_,child in blocks[index+1:]:
                    if _looks_heading(child):break
                    following.append(child)
                if any(re.search(r'\d|[.!?:]',child) and re.search(r'[A-Za-z\u05d0-\u05ea]',child)
                       for child in following):
                    # With no reliable peer-item layout, the verified parent
                    # supplies safe context for a split child name/model.
                    return lines[start:stop], ()
                return [], ()
            # Native PDF bullet markers/scalars may be on their own line.
            # Retain that directly adjacent field marker with its label.
            first = index
            if index and re.fullmatch(r'[•▪◦*-]\s*[+-]?\d+(?:[.,]\d+)?',blocks[index-1][2]):first-=1
            ranges.append((blocks[first][0],b))
            heading_indices = []
            for previous in range(index):
                candidate = blocks[previous][2]
                ordinal = bool(previous and re.fullmatch(r'(?:\.\s*\d{1,3}|\d{1,3}[.)])',blocks[previous-1][2]))
                if (2<=len(candidate.split())<=12 and len(candidate)<=200
                        and not re.search(r'[.!?;]$',candidate)
                        and not re.match(r'^[•▪◦*-]',candidate)
                        and not (previous and re.fullmatch(r'[•▪◦*-]\s*(?:[+-]?\d+(?:[.,]\d+)?)?',blocks[previous-1][2]))):
                    if ordinal:heading_indices=[]
                    heading_indices.append(previous)
            if heading_indices:
                # Unnumbered sibling headings replace one another; explicit
                # numbering retains its governing heading and local children.
                explicit = next((i for i in reversed(heading_indices) if i and re.fullmatch(
                    r'(?:\.\s*\d{1,3}|\d{1,3}[.)])',blocks[i-1][2])), None)
                chosen = [i for i in heading_indices if explicit is not None and i>=explicit] if explicit is not None else heading_indices[-1:]
                ranges.extend((blocks[i][0],b) for i in chosen)
    if not ranges:return [], ()
    # The entity label can itself span native PDF lines. It remains the parent
    # of the selected child/property and is never borrowed from another room.
    ranges.append((lines[start][0],anchor.end()))
    return [line for line in lines[start:stop]
            if any(a<line[1] and line[0]<b for a,b in ranges)], ()


def _section_identity(hits):
    """Compare all content tokens, ignoring only proven extraction decoration.

    Keep every scalar and word, including ambiguous ordinal-looking decimals.
    Exact overlapping token sequences are copied chunk text,
    rather than a new fact. Conflicting quantities cannot collapse here.
    """
    tokens = ()
    for hit in hits:
        text = hit['document']
        text = re.sub(r'(?m)^[ \t]*[•▪◦●‣*–—](?:[ \t]+|$)', '', text)
        text = re.sub(r'(?m)^[ \t]*-(?:[ \t]+)(?=[A-Za-z\u05d0-\u05ea])','',text)
        text = re.sub(r'([+-])\s+(?=(?:\d|[.,]\d))',r'\1',text)
        current = tuple(re.findall(r'[+-]?(?:\d+(?:[.,/]\d+)*|[.,]\d+)|[^\W_]+(?:[._/-][^\W_]+)*|'
                                   r'[%₪$€£°"″\'′<>≤≥=]',
                                   unicodedata.normalize('NFKC', text), re.UNICODE))
        overlap = next((n for n in range(min(len(tokens), len(current)), 5, -1)
                        if tokens[-n:] == current[:n]), 0)
        tokens += current[overlap:]
    from semantic_answer_quality import canonical_quote_is_readable
    return tokens,all(canonical_quote_is_readable(hit['document']) for hit in hits)


def _descriptor_key(value):
    value = unicodedata.normalize('NFKC',str(value or '')).casefold()
    # Rejoin only a literal hyphenated identifier broken by extraction wrap.
    # Preserve its hyphen, every quantity, and every qualifier word.
    value = re.sub(r'(?<=[A-Za-z0-9])-\s+(?=[A-Za-z0-9])','-',value)
    value = ' '.join(value.split())
    return re.sub(r'\s*([(),;])\s*',r'\1',value)


def _nonmonetary_spans(text,start,end):
    """Select literal prose around existing canonical currency tokens."""
    from financial_semantics import _money_matches,_CURRENCY
    value=text[start:end];excluded=[]
    for match,_amount,_currency in _money_matches(value):
        a,b=match.span()
        for marker in _CURRENCY.finditer(value):
            if marker.end()<=a and not value[marker.end():a].strip():a=marker.start()
            elif marker.start()>=b and not value[b:marker.start()].strip():b=marker.end()
        excluded.append((a,b))
    cursor=0;result=[]
    for a,b in sorted(excluded):
        if cursor<a:result.append((start+cursor,start+a))
        cursor=max(cursor,b)
    if cursor<len(value):result.append((start+cursor,end))
    return result


def _literal_span_hit(row,a,b,reference,topic):
    text=str(row.get('document') or '')
    literal=text[a:b].strip()
    if not literal:return None
    a+=len(text[a:b])-len(text[a:b].lstrip());b=a+len(literal)
    metadata=dict(row.get('metadata') or {})
    metadata.update(_entity_chunk_start=a,_entity_chunk_end=b,
        _entity_original_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest(),
        _entity_kind=reference.kind,_entity_identifier=reference.identifier,_entity_focus=topic,
        _entity_project=getattr(reference,'project',''),
        _entity_offset_contract='literal span in indexed chunk; not raw PDF offsets')
    return {**row,'document':literal,'metadata':metadata,'score_kind':'entity_section'}


def _select_document_topic_sections(grouped,reference,*,query,max_chars,max_hits):
    from document_reading import passage_segments,bounded_excerpt,MAX_PUBLIC_CHARS,MAX_PUBLIC_WORDS
    admitted=[]
    for source,values in sorted(grouped.items()):
        rows=sorted(values,key=lambda row:((row.get('metadata') or {}).get('chunk_index',0),str(row.get('id') or '')))
        if not rows or not _document_identity_matches(reference,rows):continue
        lines,joined=_mapped_source_lines(rows)
        scoped=[]
        if lines:
            scoped,_=_focused_lines(lines,joined,0,len(lines),None,reference.topic,query,
                                    reference=reference,document_topic=True)
        if _document_scope_matches(reference,rows,scoped):admitted.append((source,scoped))
    # Source identity is resolved before answer availability. A duplicate ID
    # cannot disappear merely because its source has no descriptive prose.
    if len(admitted)>1:return [],True,'entity_section_ambiguous'
    if not admitted or not admitted[0][1]:return [],True,'document_topic_not_found'
    _,scoped=admitted[0]
    fallback=bool(scoped[0][2]['metadata'].get('_entity_document_fallback'))
    spans=[]
    for _,_,row,a,b in scoped:
        if spans and spans[-1][0] is row and spans[-1][2]==a:spans[-1]=(row,spans[-1][1],b)
        else:spans.append((row,a,b))
    hits=[]
    for row,a,b in spans:
        original=row['document']
        for low,high in _nonmonetary_spans(original,a,b):
            cursor=low
            for segment in passage_segments(original[low:high]):
                hit=_literal_span_hit(row,cursor,cursor+len(segment),reference,reference.topic)
                cursor+=len(segment)
                if hit:hits.append(hit)
    if not hits:return [],True,'document_topic_not_found'
    remaining=min(max_chars,MAX_PUBLIC_CHARS) if fallback else max_chars
    words=MAX_PUBLIC_WORDS if fallback else 100000
    selected=[];complete=True
    for hit in hits:
        if len(selected)>=max_hits or remaining<=0 or words<=0:
            complete=False;break
        value=bounded_excerpt(hit['document'],remaining,words)
        if not value:complete=False;break
        if value!=hit['document']:
            metadata=dict(hit['metadata']);metadata['_entity_chunk_end']=metadata['_entity_chunk_start']+len(value)
            hit={**hit,'document':value,'metadata':metadata};complete=False
        selected.append(hit);remaining-=len(value);words-=len(value.split())
        if not complete:break
    return selected,complete,'' if complete else 'partial_entity_section'


def _select_entity_sections(grouped, reference, *, query='', max_chars=16000, max_hits=12):
    """Recover governing headings and continuations, stopping at a new entity.

    Sources must already be tenant-authorized. Explicit project identity is
    checked at source level. Ambiguous same-ID sources remain separate rather
    than borrowing their headings or facts from one another.
    """
    from document_query import DocumentTopicReference
    if isinstance(reference,DocumentTopicReference):
        return _select_document_topic_sections(grouped,reference,query=query,max_chars=max_chars,max_hits=max_hits)
    if isinstance(reference, EntityGroupReference):
        selected, complete = [], True
        remaining = max(0, min(int(max_chars), 32000))
        for member in reference.members:
            hits, member_complete, reason = _select_entity_sections(
                grouped, member, query=query, max_chars=remaining,
                max_hits=max(0, max_hits-len(selected)))
            if reason == 'entity_section_ambiguous':
                return [], True, reason
            if not hits or not member_complete:
                complete = False
            selected.extend(hits)
            remaining -= sum(len(hit['document']) for hit in hits)
        return selected, complete, ('' if selected and complete else
            'partial_entity_comparison' if selected else 'entity_section_not_found')
    if not isinstance(reference, EntityReference):
        return [], True, 'entity_reference_missing'
    selected = []
    selection_complete = True
    remaining = max(0, min(int(max_chars), 32000))
    pattern = _entity_pattern(reference)
    topic,count_focus = _requested_focus(query,reference)
    seen_sections = set()
    def source_priority(item):
        metadata = item[1][0].get('metadata') or {} if item[1] else {}
        return (metadata.get('mime_type') != 'application/pdf',
                str(metadata.get('source_relative_path') or ''), str(item[0]))
    for source, source_rows in sorted(grouped.items(), key=source_priority):
        rows = sorted(source_rows, key=lambda row: ((row.get('metadata') or {}).get('chunk_index', 0), str(row.get('id') or '')))
        if not rows or not _source_matches(reference, rows):
            continue
        lines = []
        joined = ''
        for row in rows:
            text = str(row.get('document') or '')
            crumb = _BREADCRUMB.match(text)
            offset = crumb.end() if crumb else 0
            for line in re.finditer(r'[^\n]*(?:\n|$)', text[offset:]):
                if not line.group():
                    continue
                start, end = offset+line.start(), offset+line.end()
                lines.append((len(joined), len(joined)+end-start, row, start, end))
                joined += text[start:end]
            joined += '\n'
        anchor = pattern.search(joined)
        if anchor is None:
            continue
        numbered = _numbered_context(joined,anchor,reference)
        if numbered:
            lines = _split_source_lines(lines,{numbered[0],numbered[1],*(item[0] for item in numbered[2])})
        first = next((i for i, line in enumerate(lines) if line[0] <= anchor.start() < line[1]), None)
        if first is None:
            continue
        last_anchor = next((i for i, line in enumerate(lines) if line[0] < anchor.end() <= line[1]), first)
        # Include a nearby standalone subtype heading immediately preceding
        # a split "room number" label; recover the governing floor separately.
        start = first
        while start > 0 and first-start < 12:
            previous = joined[lines[start-1][0]:lines[start-1][1]].strip()
            previous_ids = role_bound_identifier_spans(previous)
            peer = any(item.kind==reference.kind and item.identifier!=reference.identifier
                       for item in previous_ids)
            floor_child = reference.kind=='room' and any(item.kind=='floor'
                and re.match(r'^(?:[בלמכ]?ה?)?'+_KINDS['room']+r'(?![A-Za-z\u05d0-\u05ea])',
                    previous[item.entity_span[1]:].strip(' \t:–—-,'),re.I)
                for item in previous_ids)
            if peer or floor_child:
                break
            if (not previous or _section_heading(previous, reference)
                    or re.fullmatch(_NUMBER + r'\s*', previous, re.I)):
                start -= 1
            else:
                break
        governing = _governing_floor_lines(lines,joined,anchor.start())
        if len({id(line[2]) for line in governing}) > 1:
            # A label/value split over original chunks has no single literal
            # quote span. Preserve the room facts and disclose partial context
            # rather than emitting uncitable fragments or an older floor.
            governing = ()
            selection_complete = False
        stop = len(lines)
        for i in range(last_anchor+1, len(lines)):
            literal = joined[lines[i][0]:lines[i][1]].strip()
            if _section_heading(literal, reference) and not pattern.search(literal):
                stop = i
                break
        if numbered:
            start = next(i for i,line in enumerate(lines) if line[0]==numbered[0])
            stop = min(stop,next((i for i,line in enumerate(lines) if line[0]>=numbered[1]),len(lines)))
        scoped, descriptor_fields = _focused_lines(lines, joined, start, stop, anchor, topic, query,count_focus,numbered,reference)
        if not scoped:continue
        governing = tuple(line for line in governing if not any(
            kept[2] is line[2] and kept[3] <= line[3] and line[4] <= kept[4] for kept in scoped))
        keep = list(governing) + scoped
        last_row = rows[-1]
        last_meta = last_row.get('metadata') or {}
        # A native page window is not the document's end. A requested leaf
        # that still runs to that window's edge can continue on an unread
        # page; retain honest partial evidence instead of asserting closure.
        window_partial = bool(last_meta.get('_entity_representation')=='native_pdf'
            and type(last_meta.get('_native_pdf_page_count')) is int
            and last_meta.get('page',0)<last_meta['_native_pdf_page_count']
            and max(line[1] for line in keep)>=max(line[1] for line in lines))
        spans = []
        for _, _, row, a, b in keep:
            if spans and spans[-1][0] is row and spans[-1][2] == a:
                spans[-1] = (row, spans[-1][1], b)
            else:
                spans.append((row, a, b))
        bounded_spans = []
        from document_reading import passage_segments
        for row, a, b in spans:
            text = str(row.get('document') or '')
            values = [(va,vb) for owner,_,_,va,vb,_ in descriptor_fields
                      if owner is row and a<=va<vb<=b]
            if not values:
                # Ingested cell context is another explicit source field.
                # Reuse its existing literal field binding; serialization
                # labels remain provenance rather than answer/quote prose.
                from structural_evidence import _structured_field_bindings
                for binding in _structured_field_bindings(text[a:b],group_id='entity-context',worksheet_row=None):
                    if (binding.role=='field' and re.fullmatch(r'Cell\s+[A-Z]{1,3}[1-9]\d*',binding.column_role,re.I)
                            and pattern.search(binding.value)):
                        va,vb = binding.source_span
                        values.append((a+va,a+vb))
            for value_start,value_end in values or [(a,b)]:
                cursor = value_start
                for segment in passage_segments(text[value_start:value_end]):
                    bounded_spans.append((row, cursor, cursor+len(segment)))
                    cursor += len(segment)
        source_hits = []
        for row, a, b in bounded_spans:
            hit=_literal_span_hit(row,a,b,reference,topic)
            if hit is None:continue
            metadata=hit['metadata']
            a,b=metadata['_entity_chunk_start'],metadata['_entity_chunk_end']
            descriptors = [(va,vb,value) for owner,_,_,va,vb,value in descriptor_fields
                           if owner is row and a<=va<vb<=b]
            if descriptors:
                metadata['_entity_descriptor_spans'] = tuple((va,vb) for va,vb,_ in descriptors)
                metadata['_entity_descriptor_text'] = '\n'.join(value for _,_,value in descriptors)
            source_hits.append(hit)
        if any(hit['metadata'].get('_entity_descriptor_text') for hit in source_hits):
            coverage = [_descriptor_key(hit['metadata'].get('_entity_descriptor_text') or hit['document'])
                        for hit in selected if reference.project
                        or hit['metadata'].get('source_id') == source]
            kept, novel_descriptor = [], False
            for hit in source_hits:
                descriptor = hit['metadata'].get('_entity_descriptor_text')
                if descriptor:
                    key = _descriptor_key(descriptor)
                    if key and key in coverage:continue
                    novel_descriptor = True
                    coverage.append(key)
                kept.append(hit)
            if not novel_descriptor:continue
            source_hits = kept
        # Text equality is not entity ownership. An unscoped same-ID subject
        # in two sources remains ambiguous even if both specifications happen
        # to be word-for-word equal; a shared project must first be explicit.
        section_identity = (_section_identity(source_hits), '' if reference.project else source)
        if section_identity in seen_sections:
            continue
        seen_sections.add(section_identity)
        if source_hits and window_partial:selection_complete = False
        for hit in source_hits:
            if len(hit['document']) > remaining or len(selected) >= max_hits:
                return selected, False, 'partial_entity_section'
            selected.append(hit)
            remaining -= len(hit['document'])
    sources = {hit['metadata'].get('source_id') for hit in selected}
    if not reference.project and len(sources) > 1:
        return [], True, 'entity_section_ambiguous'
    return selected, selection_complete, ('partial_entity_section' if selected and not selection_complete
                                           else '' if selected else 'entity_section_not_found')


def select_entity_sections(grouped, reference, *, query='', max_chars=16000, max_hits=12):
    return _select_entity_sections(grouped, reference, query=query, max_chars=max_chars, max_hits=max_hits)[0]


def collect_entity_sections(retriever, tenant, document_root, query, where=None):
    """Collect processable section evidence using the shared source boundary."""
    from document_reading import scan_authorized
    grouped, complete, count = scan_authorized(retriever, tenant, document_root, where=where)
    from entity_native_pdf import recover_entity_pdf_rows
    native_rows=()
    if isinstance(section_reference(query),EntityReference):
        native_rows=recover_entity_pdf_rows(grouped,query,document_root,tenant).native_rows
    return entity_evidence_from_rows(grouped, query, storage_complete=complete,
                                    examined_chunks=count,native_rows=native_rows)


def _selection_rows(grouped,native_rows):
    selection = {}
    for values in grouped.values():
        for row in values:
            source = (row.get('metadata') or {}).get('source_id')
            selection.setdefault(source,[]).append(row)
    recovered = {}
    for row in native_rows:
        source = (row.get('metadata') or {}).get('source_id')
        recovered.setdefault(source,[]).append(row)
    selection.update(recovered)
    return selection


def _native_window_binding_valid(grouped,native_rows,reference):
    """Recompute native page coverage from the original indexed identity.

    The final quote consumer independently rereads complete native pages.
    This check additionally prevents silently omitting a whole peer-heading
    page or stitching separate windows before that consumer receives them.
    """
    # Multi-subject indexed selection retains its normal coverage witness.
    # Native recovery windows currently prove one subject; do not silently
    # reuse one subject's recovered pages as the other's proof.
    if isinstance(reference, EntityGroupReference):
        return not native_rows
    recovered = {}
    for row in native_rows:
        recovered.setdefault((row.get('metadata') or {}).get('source_id'),[]).append(row)
    for source,rows in recovered.items():
        indexed = sorted(grouped.get(source,()),key=lambda row:((row.get('metadata') or {}).get('chunk_index',0),str(row.get('id') or '')))
        if not indexed or not _source_matches(reference,indexed):return False
        joined,positions = '',[]
        for row in indexed:
            text = str(row.get('document') or '')
            crumb = _BREADCRUMB.match(text)
            body = text[crumb.end():] if crumb else text
            positions.append((len(joined),len(joined)+len(body),row))
            joined += body+'\n'
        anchors = [item for item in role_bound_identifier_spans(joined)
                   if item.kind==reference.kind and item.identifier==reference.identifier]
        anchor_rows = [row for a,b,row in positions
                       if any(a<item.entity_span[1] and item.entity_span[0]<b for item in anchors)]
        if not anchor_rows:return False
        anchor_pages = tuple(sorted({row['metadata'].get('page') for row in anchor_rows
                                     if type(row['metadata'].get('page')) is int}))
        if (not anchor_pages or min(anchor_pages)<1
                or any(type(row['metadata'].get('page')) is not int for row in anchor_rows)):
            return False
        anchor_ids = tuple(str(row.get('id') or row['metadata'].get('chunk_id') or '') for row in anchor_rows)
        page_count = rows[0]['metadata'].get('_native_pdf_page_count')
        if type(page_count) is not int or page_count<max(anchor_pages):return False
        expected = tuple(sorted({page for anchor in anchor_pages for page in (anchor-1,anchor,anchor+1)
                                 if 1<=page<=page_count}))
        if (len(expected)>8 or expected!=tuple(range(expected[0],expected[-1]+1))
                or sorted(row['metadata'].get('page') for row in rows)!=list(expected)):
            return False
        for row in rows:
            meta = row['metadata'];page = meta.get('page')
            text_hash = hashlib.sha256(row['document'].encode('utf-8')).hexdigest()
            identifier = f'{source}::native-pdf::{page:06d}::{text_hash[:16]}'
            if (meta.get('_native_pdf_page_count')!=page_count
                    or meta.get('_native_pdf_anchor_pages')!=anchor_pages
                    or meta.get('_native_pdf_anchor_chunk_ids')!=anchor_ids
                    or meta.get('_native_pdf_requested_pages')!=expected
                    or meta.get('_native_pdf_page_text_sha256')!=text_hash
                    or not re.fullmatch(r'[0-9a-f]{64}',str(meta.get('_native_pdf_raw_text_sha256') or ''))
                    or row.get('id')!=identifier or meta.get('chunk_id')!=identifier
                    or meta.get('chunk_index')!=page-1):
                return False
    return True


def entity_evidence_from_rows(grouped, query, *, storage_complete=True, examined_chunks=None,native_rows=()):
    from document_reading import ScanWitness, make_passage
    reference = section_reference(query)
    rows = tuple(row for values in grouped.values() for row in values)
    hits, selection_complete, reason = _select_entity_sections(_selection_rows(grouped,native_rows), reference, query=query)
    passages = tuple(replace(make_passage(hit['document'], hit['document'], hit['metadata'],
                                 hit.get('id') or hit['metadata'].get('chunk_id'), i),
                                 representation=hit['metadata'].get('_entity_representation','indexed'))
                     for i, hit in enumerate(hits, 1))
    return EntityEvidence(passages, bool(storage_complete and selection_complete), len(grouped),
                          len(rows) if examined_chunks is None else examined_chunks,
                          reason,
                          ScanWitness(rows, bool(storage_complete)), reference, query, selection_complete,tuple(native_rows))


def validate_entity_evidence(evidence, query):
    """Recompute literal selection, identifiers, page provenance and counts.

    This is independent of any model answer or claimed PASS. Authorization of
    the current source bytes remains required at the caller's release gate.
    """
    from document_reading import passage_valid
    reference = section_reference(query)
    witness = getattr(evidence, 'witness', None)
    eligible = getattr(evidence,'quote_eligible_ids',None)
    if eligible is not None and (not isinstance(eligible,tuple)
            or any(not isinstance(value,str) for value in eligible)
            or len(eligible)!=len(set(eligible))
            or any(value not in {p.citation_id for p in evidence.passages} for value in eligible)):
        return False
    if (not reference or reference != getattr(evidence, 'reference', None)
            or not witness or getattr(evidence, 'question', None) != query):
        return False
    grouped = {}
    identities = {}
    for row in witness.rows:
        metadata = row.get('metadata') or {}
        source = metadata.get('source_id')
        if not source or not isinstance(row.get('document'), str):
            return False
        identity = tuple(metadata.get(k) for k in
                         ('tenant_id', 'source_id', 'source_sha256', 'source_relative_path',
                          'source_size', 'source_mtime_ns', 'mime_type'))
        if source in identities and identities[source] != identity:
            return False
        identities[source] = identity
        grouped.setdefault(source, []).append(row)
    if (not evidence.passages or evidence.examined_documents != len(grouped)
            or evidence.examined_chunks != len(witness.rows)
            or len({str(row.get('id')) for row in witness.rows}) != len(witness.rows)
            or len({identity[0] for identity in identities.values()}) != 1):
        return False
    native_rows = getattr(evidence,'native_rows',())
    if not isinstance(native_rows,tuple):return False
    for row in native_rows:
        if not isinstance(row,dict) or not isinstance(row.get('metadata'),dict):return False
        metadata = row.get('metadata') or {}
        source = metadata.get('source_id')
        identity = tuple(metadata.get(k) for k in
                         ('tenant_id', 'source_id', 'source_sha256', 'source_relative_path',
                          'source_size', 'source_mtime_ns', 'mime_type'))
        if (source not in identities or identity!=identities[source]
                or metadata.get('_entity_representation')!='native_pdf'
                or metadata.get('mime_type')!='application/pdf'
                or type(metadata.get('page')) is not int or metadata['page']<1
                or not isinstance(row.get('document'),str)):
            return False
    if not _native_window_binding_valid(grouped,native_rows,reference):return False
    expected, selection_complete, reason = _select_entity_sections(_selection_rows(grouped,native_rows), reference, query=query)
    if (evidence.complete != bool(witness.storage_complete and selection_complete)
            or evidence.selection_complete != selection_complete or evidence.reason != reason):
        return False
    if len(expected) != len(evidence.passages):
        return False
    for i, (hit, passage) in enumerate(zip(expected, evidence.passages), 1):
        if (not passage_valid(passage) or passage.text != hit['document']
                or passage.source_claim != hit['document'] or passage.citation_id != str(i)
                or passage.metadata != hit['metadata']
                or passage.representation != hit['metadata'].get('_entity_representation','indexed')
                or passage.evidence_id != (hit.get('id') or hit['metadata'].get('chunk_id'))):
            return False
    return True
