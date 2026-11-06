"""Request-local document references and source-owned structured locations.

This module never contains corpus names or answers. It interprets explicit
request grammar and verifies locations against existing canonical source fields.
"""
from __future__ import annotations
import json,re,unicodedata
from dataclasses import dataclass
from pathlib import PurePosixPath

_IDENTIFIER=re.compile(r'(?<![\w/\-])(?=[A-Za-z0-9_/-]*[A-Za-z])(?=[A-Za-z0-9_/-]*\d)[A-Za-z][A-Za-z0-9_]*(?:[-/][A-Za-z0-9_]+)+(?![\w/\-])')
_QUANTITY=re.compile(r'\b(?:quantity|qty|how\s+many|number\s+of)\b|(?:כמות|כמה)',re.I)
_GENERIC={'the','of','named','called','של','את','ה','בנושא'}

# One role grammar owns request identifiers for both semantic compilation and
# section retrieval. A role-bound number selects an entity; it is not a result
# count, date, amount or measured answer. Keep this module dependency-light.
ENTITY_ROLE_PATTERNS = {
 'room': r'(?:classroom|class|lecture\s+hall|laboratory|lab|hall|room|חדר|כיתה|כיתת(?:\s+לימוד)?|אולם|מעבדה)',
 'floor': r'(?:floor|storey|story|קומה|קומת)',
 'equipment': r'(?:equipment|device|unit|ציוד|מכשיר|יחידה)',
 'product': r'(?:product|model|מוצר|דגם)',
 'procedure': r'(?:procedure|process|נוהל|הליך)',
 'clause': r'(?:clause|section|סעיף)',
 'invoice': r'(?:invoice|חשבונית(?:\s+מס)?(?:\s+קבלה)?)',
 'receipt': r'(?:receipt|קבלה)',
 'document': r'(?:document|מסמך)',
}
_ENTITY_IDENTIFIER = r'(?=[A-Za-z0-9.\-/]*\d)[A-Za-z0-9]+(?:[.\-/][A-Za-z0-9]+)*'
_ENTITY_NUMBER_LABEL = r'(?:number|no\.?|מספר|מס[\'׳.]?|#)'
_ENTITY_SUBTYPE = r'(?:meeting|conference|training|large|small|lecture|teaching|ישיבות|הדרכה|גדולה|גדול|קטנה|קטן|לימוד)'

@dataclass(frozen=True)
class RoleBoundIdentifier:
 kind:str
 identifier:str
 identifier_span:tuple[int,int]
 entity_span:tuple[int,int]
 role_span:tuple[int,int]

def role_bound_identifier_spans(text):
 """Return literal identity spans, allowing grammatical Hebrew prefixes.

 Only an explicit role plus a numeric/alphanumeric designation is accepted.
 Arbitrary intervening verbs cannot turn a capacity into an identifier.
 """
 value=str(text or '');result=[]
 from financial_semantics import document_references
 for reference in document_references(value):
  result.append(RoleBoundIdentifier(reference.kind,reference.number,reference.number_span,
      reference.span,(reference.span[0],reference.number_span[0])))
 for kind,role in ENTITY_ROLE_PATTERNS.items():
  # A floor sign distinguishes basement/location identity. It cannot be
  # consumed as the decorative separator used by other entity roles.
  separator=r'[:=]?' if kind=='floor' else r'[-–—:=]?'
  designation=(r'[+-]?' if kind=='floor' else '')+_ENTITY_IDENTIFIER
  # Extraction can join a Hebrew role to a Latin model designation.
  # The script boundary is explicit; do not split two alphabetic words
  # in the same script or accept an identifier without a digit.
  pattern=(r'(?<!\w)(?:ו?[בלמכ]?ה?)?(?P<role>'+role+r')(?:(?![A-Za-z\u05d0-\u05ea])|(?<=[\u05d0-\u05ea])(?=[A-Za-z]))'
           r'(?:\s+'+_ENTITY_SUBTYPE+r'){0,2}\s*'
           r'(?:'+_ENTITY_NUMBER_LABEL+r'\s*)?'+separator+r'\s*'
           r'(?P<identifier>'+designation+r')(?![\w\-/]|\.(?!\s|$))')
  for match in re.finditer(pattern,value,re.I):
   identifier=match.group('identifier')
   if not any(character.isdigit() for character in identifier):continue
   # Do not accept a prefix of a grouped amount/percentage/currency literal.
   # Period-separated clause identifiers remain valid complete identities.
   if re.match(r'(?:,\d|\s*%|[₪$€£]|\s+[₪$€£](?!\s*\d))',value[match.end('identifier'):]):continue
   item=RoleBoundIdentifier(kind,identifier.casefold(),match.span('identifier'),match.span(),match.span('role'))
   # A combined invoice/receipt role is one identity, never two entities.
   if any(old.identifier_span==item.identifier_span and old.entity_span[0]<=item.entity_span[0] for old in result):continue
   result.append(item)
 return tuple(sorted(result,key=lambda item:item.entity_span))

def mask_role_bound_identifiers(text):
 """Keep offsets while excluding identity labels from result-count cues."""
 value=str(text or '')
 spans=[]
 for identifier in role_bound_identifier_spans(value):
  start,end=identifier.entity_span
  preceding_label=re.search(r'(?:\b(?:number|no\.?)|מספר)\s*$',value[:start],re.I)
  if preceding_label:start=preceding_label.start()
  spans.append((start,end))
 for start,end in reversed(spans):
  value=value[:start]+' '*(end-start)+value[end:]
 return value

@dataclass(frozen=True)
class DocumentTopicReference:
 kind:str
 identifier:str
 topic:str
 scope_text:str=''
 scope_terms:tuple[str,...]=()
 reference_span:tuple[int,int]=(0,0)

 @property
 def document_kind(self):return self.kind


def document_topic_reference(query):
 """Bind an explicit document identity to a descriptive content question.

 The canonical financial document grammar supplies quote/invoice identity.
 A price, amount or count request retains its ordinary typed request path.
 """
 from financial_semantics import document_references, financial_request, monetary_role
 value=str(query or '')
 references=list(document_references(value))
 references += [item for item in role_bound_identifier_spans(value) if item.kind=='document']
 if len(references)!=1 or financial_request(value) is not None:return None
 reference=references[0]
 span=reference.span if hasattr(reference,'span') else reference.entity_span
 identifier=reference.number if hasattr(reference,'number') else reference.identifier
 content=value[:span[0]]+' '*(span[1]-span[0])+value[span[1]:]
 if (monetary_role(content) or re.search(
     r'\b(?:price|cost|amount|total|payable|subtotal|vat|tax|how\s+much|how\s+many)\b|'
     r'(?:כמה|מחיר|עלות|עולה|סכום|לתשלום|מע["״]?מ)',content,re.I)):
  return None
 patterns=(
  r'(?<!\w)מה\s+(?:כולל(?:ת|ים|ות)?|מכיל(?:ה|ים|ות)?|מפרט(?:ת|ים|ות)?)\s+(?P<topic>[^?;]+)',
  r'\bwhat\s+(?:does|do)\s+(?P<topic>[^?;]+?)\s+(?:include|contain|cover|involve)\s*[?!.]*\s*$',
  r'(?:\b(?:describe|explain|summari[sz]e)\s+(?:the\s+)?|(?<!\w)(?:תאר|תארי|תארו|הסבר|הסבירי|הסבירו|פרט|פרטי|פרטו)\s+(?:את\s+)?)(?P<topic>[^?;]+)',
 )
 matches=[match for pattern in patterns for match in re.finditer(pattern,value,re.I)]
 if len(matches)!=1:return None
 match=matches[0];a,b=match.span('topic')
 if span[1]<=match.start():
  introduction=value[:span[0]].strip(' ,;:')
  if not re.fullmatch(r'(?:(?:according\s+to|per|in|from|לפי|על\s+פי|בהתאם\s+ל)\s*)?',introduction,re.I):return None
 elif match.start()<span[0]:
  if value[:match.start()].strip().casefold() not in {'','please'}:return None
 if a<span[0]<b:b=span[0]
 elif a<span[1] and span[0]<b:return None
 topic=value[a:b].strip(' ,;:?! .')
 topic=re.sub(r'\s+(?:according\s+to|in|from|of|לפי|במסמך)\s*$', '',topic,flags=re.I).strip()
 if not topic or len(topic)>180 or not re.search(r'[A-Za-z\u05d0-\u05ea]',topic):return None
 if span[1]<=match.start():scope=value[span[1]:match.start()]
 elif span[0]>=match.end():scope=value[span[1]:]
 else:scope=value[span[1]:] if b==span[0] else ''
 scope=scope.strip(' ,;:?! .')
 preposition=re.match(r'^(?:for|to|at|in)\s+',scope,re.I)
 if preposition:scope=scope[preposition.end():]
 elif re.match(r'^ל[\u05d0-\u05ea]',scope):scope=scope[1:]
 scope_terms=_identity_words(scope)
 return DocumentTopicReference(reference.kind,identifier,topic,scope,scope_terms,span)

def _identity_words(value):
 """Normalize spelling decoration without deleting named identity letters."""
 value=unicodedata.normalize('NFKC',str(value or '')).casefold()
 value=''.join(character for character in value if unicodedata.category(character) not in {'Mn','Me','Cf'})
 return tuple(re.findall(r'[^\W_]+',value,re.UNICODE))

def _document_identity(value):
 words=_identity_words(value)
 # Description is a grammatical document role on BOTH sides. It is never
 # an alias for a proper name and cannot disappear from only one identity.
 if words and words[0] in {'description','תיאור','תאור'}:return 'description',words[1:]
 return '',words

def _contains_identity(wanted,observed):
 return bool(wanted) and any(observed[start:start+len(wanted)]==wanted for start in range(len(observed)-len(wanted)+1))

def _heading_topic_matches(wanted,observed):
 """Use the same bounded topic grammar as the ordered-list consumer."""
 from ordered_list_evidence import heading_topic_matches
 return heading_topic_matches(wanted,observed)

def _words(value):
 value=unicodedata.normalize('NFKC',str(value or '')).casefold()
 values=re.findall(r'[a-z0-9]+|[\u05d0-\u05ea]+',value)
 return tuple({'תיאור':'תאור'}.get(v,v[1:] if v.startswith('ה') and len(v)>3 else v) for v in values if v not in _GENERIC)

@dataclass(frozen=True)
class DocumentReference:
 kind:str
 title:str
 def encoded(self):return json.dumps({'schema':'project-x-named-document-v1','kind':self.kind,'title':self.title},ensure_ascii=False,sort_keys=True,separators=(',',':'))

def named_document_reference(query):
 query=re.sub(r'^(?:הצג|הציגו|קרא|הקרא)\s+(?:לי\s+)?(?:את\s+)?תוכן\s+(?:המסמך\s+)?', 'הצג ', query or '')
 query=re.sub(r'^please\s+', '', query, flags=re.I)
 patterns=(
  ('document',r'(?:(?<!\w)מה\s+(?:כתוב|מופיע|נכתב|יש)\s+ב|(?<!\w)(?:הצג|הציגו|קרא|הקרא|פרט|סכם)\s+(?:לי\s+)?(?:את\s+)?(?:המסמך\s+)?|\b(?:read|show|reproduce|summari[sz]e)\s+(?:me\s+)?(?:the\s+)?(?:contents\s+of\s+)?(?:document\s+)?|\bwhat\s+(?:is\s+(?:written\s+)?in|does)\s+(?:the\s+)?)(?P<title>[^,;?\n]+?)(?:\s+(?:say|contain))?(?=[,;?]|$)'),
  ('spreadsheet',r'(?:בכתב\s+ה?כמויות|בחוברת\s+ה?עבודה|\b(?:in|from)\s+(?:the\s+)?(?:workbook|spreadsheet|bill\s+of\s+quantities))\s+(?:של\s+)?(?P<title>[^,;?\n]+?)(?=[,;?]|\s+(?:what|which|how)\b|$)'),
  ('document',r'(?:במסמך|\b(?:in|from)\s+(?:the\s+)?document)\s+(?P<title>[^,;?\n]+?)(?=[,;?]|\s+(?:what|which|how)\b|$)'),
  ('presentation',r'(?:במצגת|\b(?:in|from)\s+(?:the\s+)?presentation)\s+(?P<title>[^,;?\n]+?)(?=[,;?]|\s+(?:what|which|how)\b|$)'),
 )
 candidates=[]
 for order,(kind,pattern) in enumerate(patterns):
  for match in re.finditer('(?='+pattern+')',query or '',re.I):
   raw_title=match.group('title').strip()
   if kind=='document':raw_title=re.sub(r'^(?:ה?מסמך|the\s+document|document)\s+', '', raw_title, flags=re.I)
   title=raw_title.strip(' \"\'״')
   # A requested topic is content inside the named source, not part of its title.
   # Read an explicitly quoted title from its literal position in the full
   # request, so punctuation inside a filename cannot end it prematurely.
   quoted=re.match(r'(?P<quote>[\"\'״])(?P<title>.+?)(?P=quote)(?=\s|[.,;:?!]|$)',
       query[match.start('title'):])
   # Sentence punctuation after an unquoted, recognized filename is not part
   # of its identity. Quoted filenames retain every literal character below.
   filename=re.match(r'^(.+\.(?:pdf|docx?|pptx?|xlsx?|csv|tsv|txt|md|odt|ods|odp))(?=[.!]+\s*$|\s+(?:say|says|contain|contains|about|לגבי|אודות|על)\b|$)',title,re.I)
   if quoted:title=quoted.group('title')
   elif filename:title=filename.group(1)
   else:title=re.split(r'\s+(?:say|says|contain|contains)\s+about\s+|\s+(?:about|לגבי|אודות|על)\s+',title,maxsplit=1,flags=re.I)[0].strip()
   if kind=='document':
    title=re.sub(r'^(?:ה?מסמך|the\s+document|document)\s+', '', title, flags=re.I)
   if not quoted and not filename and re.fullmatch(
       r'(?:(?:the\s+)?(?:this|that|above|previous)(?:\s+(?:document|file))?|the\s+(?:document|file)|'
       r'(?:המסמך|הקובץ)(?:\s+(?:הזה|ההוא|לעיל|הקודם))?|(?:זה|הזה|ההוא|הקודם|לעיל))',title,re.I):
    continue
   if 1<len(title)<=160 and _document_identity(title)[1]:
    # A declared source owns the scope before a later instruction describes
    # what to summarize inside it. Generic content verbs remain a fallback.
    priority=0 if quoted or filename else 1 if order else 2
    candidates.append((priority,match.start(),order,DocumentReference(kind,title)))
 strongest=min((item[0] for item in candidates),default=2)
 explicit_refs = {(item[3].kind,item[3].title.casefold()) for item in candidates if item[0]==strongest and strongest<2}
 if len(explicit_refs)>1:
  # None means no named restriction to several callers. Keep an explicit
  # unresolved reference so competing declarations cannot expand the scope
  # or silently select the first source. Unknown kinds never match inventory.
  return DocumentReference('ambiguous','')
 return min(candidates,key=lambda item:item[:3])[3] if candidates else None

def reference_matches(reference,source_label,mime_type='',heading=''):
 import mimetypes
 # Canonical numbered document titles may put the number first in the
 # filename. Only complete title identities qualify, with kind and number
 # both identical; path/tenant authorization remains the inventory's job.
 from financial_documents import primary_document_identities
 if reference.kind == 'document':
  requested = primary_document_identities(str(reference.title))
  if len(requested) == 1:
   observed = primary_document_identities(PurePosixPath(str(source_label).replace('\\','/')).stem)
   return bool(len(observed) == 1 and
               (requested[0].kind, requested[0].number) == (observed[0].kind, observed[0].number))
 source=PurePosixPath(str(source_label).replace('\\','/'));extension=source.suffix.casefold()
 kinds={'spreadsheet':{'.xlsx','.xls','.csv','.tsv','.ods'},'presentation':{'.pptx','.ppt','.odp'},'document':None}
 if reference.kind not in kinds or (kinds[reference.kind] is not None and extension not in kinds[reference.kind]):return False
 title=PurePosixPath(str(reference.title).replace('\\','/'))
 explicit=bool(title.suffix and (title.suffix.casefold()==extension or mimetypes.guess_type(str(title),strict=False)[0]
                               or title.suffix.casefold() in {'.docx','.pptx','.xlsx','.odt','.ods','.odp'}))
 if explicit and title.suffix.casefold()!=extension:return False
 title_text=str(title.with_suffix('')) if explicit else str(title)
 if str(title.parent) not in {'', '.'}:
  observed_path=str(source.with_suffix('')) if explicit else str(source.with_suffix(''))
  return _identity_words(title_text)==_identity_words(observed_path)
 wanted_role,wanted=_document_identity(title_text);observed_role,observed=_document_identity(source.stem)
 if wanted_role==observed_role and wanted and (wanted==observed if explicit else _contains_identity(wanted,observed)):return True
 # A first-slide topic is separate evidence; never manufacture a title by
 # joining filename words with unrelated heading words.
 return bool(not explicit and reference.kind=='presentation' and heading
             and _heading_topic_matches(title_text,heading))

def requested_item_identifier(query):
 if not _QUANTITY.search(query or ''):return ''
 values=tuple(dict.fromkeys(m.group() for m in _IDENTIFIER.finditer(query or '')))
 return values[0] if len(values)==1 else ''

def requested_source_locations(query):
 query=query or '';result=[]
 for name,pattern in (('sheet',r'\b(?:which|what)\s+(?:work)?sheet\b|(?:באיזה|באיזו).{0,12}גיליון|גיליון\s+ושורה'),('row',r'\b(?:which|what)\s+row\b|(?:באיזה|באיזו).{0,16}שורה|גיליון\s+ושורה'),('cell',r'\b(?:which|what)\s+cell\b|(?:באיזה|באיזו)\s+תא')):
  if re.search(pattern,query,re.I):result.append(name)
 return tuple(result)

def fact_source_locations(fact):
 import hashlib
 context=getattr(fact,'evidence_context',None);unit=getattr(context,'structural_evidence_unit',None)
 if unit is None or not getattr(context,'structural_binding_valid',False) or not getattr(context,'source_id',''):return {}
 if (unit.source_id!=context.source_id or unit.candidate_id!=context.candidate_id
     or unit.chunk_id!=context.chunk_id or unit.unit_id!=context.evidence_unit_id
     or getattr(fact,'evidence_id','')!=unit.unit_id or getattr(fact,'candidate_id','')!=unit.candidate_id):return {}
 source=str(getattr(context,'source_candidate_text','') or '');text=str(unit.local_text)
 def valid_span(span,length):
  return isinstance(span,(tuple,list)) and len(span)==2 and all(type(v) is int for v in span) and 0<=span[0]<span[1]<=length
 spans=tuple(getattr(unit,'source_spans',()))
 if len(spans)!=1:return {}
 span=spans[0]
 if not valid_span((span.start,span.end),len(source)) or source[span.start:span.end]!=text:return {}
 text_hash=hashlib.sha256(text.encode('utf-8')).hexdigest()
 if span.sha256!=text_hash or unit.local_text_sha256!=text_hash:return {}
 numeric=getattr(fact,'numeric_binding',None);value_span=getattr(numeric,'value_span',None)
 if not valid_span(value_span,len(text)):return {}
 proposition_span=getattr(context,'proposition_span',None)
 if (not valid_span(proposition_span,len(text)) or text[slice(*proposition_span)]!=context.local_proposition
     or getattr(fact,'source_claim','')!=context.local_proposition or getattr(fact,'evidence_span',None)!=proposition_span
     or getattr(numeric,'source_span_sha256','')!=context.proposition_sha256
     or getattr(fact,'evidence_span_sha256','')!=context.proposition_sha256):return {}
 value=re.sub(r'\s+','',str(getattr(fact,'value','')))
 if not value or re.sub(r'\s+','',text[slice(*value_span)])!=value:return {}
 column=str(getattr(fact,'column_header','') or '')
 bindings=[b for b in unit.field_bindings if b.role=='value' and valid_span(b.source_span,len(text))
           and b.source_span[0]<=value_span[0]<value_span[1]<=b.source_span[1]]
 if len(bindings)!=1:return {}
 binding=bindings[0]
 if (binding.column_role!=column or binding.structural_group_id!=(unit.structural_group_id or unit.unit_id)
     or column not in unit.column_roles or not valid_span(binding.field_span,len(text))
     or not valid_span(binding.label_span,len(text))
     or getattr(numeric,'binding_method','')!='structural_row_column'
     or getattr(numeric,'structural_role','')!=f'row:{unit.row_label}|column:{column}'):return {}
 field_start,field_end=binding.field_span;label_start,label_end=binding.label_span
 if not (field_start<=label_start<label_end<=field_end and field_start<=binding.source_span[0]<binding.source_span[1]<=field_end):return {}
 line_start=text.rfind('\n',0,value_span[0])+1;line_end=text.find('\n',value_span[1])
 if line_end<0:line_end=len(text)
 line=text[line_start:line_end]
 parsed=re.fullmatch(r'[ \t]*[-*•▪◦]\s+(?P<label>[^:\r\n]{1,160}?):[ \t]*(?P<value>[^\r\n]*)',line)
 if (not parsed or text[field_start:field_end].strip()!=line.strip()
     or (label_start,label_end)!=(line_start+parsed.start('label'),line_start+parsed.end('label'))
     or ' '.join(parsed.group('label').split())!=column
     or not line_start+parsed.start('value')<=value_span[0]<value_span[1]<=line_start+parsed.end('value')):return {}
 row=getattr(unit,'worksheet_row',None);row_span=getattr(unit,'worksheet_row_span',None)
 if type(row) is not int or row<=0 or row_span is None or not valid_span((row_span.start,row_span.end),len(source)):return {}
 row_literal=source[row_span.start:row_span.end]
 if not row_literal.isdigit() or int(row_literal)!=row or row_span.sha256!=hashlib.sha256(row_literal.encode('utf-8')).hexdigest():return {}
 row_line_start=source.rfind('\n',0,row_span.start)+1;row_line_end=source.find('\n',row_span.end)
 if row_line_end<0:row_line_end=len(source)
 row_header=re.fullmatch(r'[ \t]*Spreadsheet row\s+(?P<number>\d+)\s*:[ \t]*',source[row_line_start:row_line_end],re.I)
 if (not row_header or (row_span.start,row_span.end)!=(row_line_start+row_header.start('number'),row_line_start+row_header.end('number'))
     or row_line_end>span.start or source[row_line_end:span.start].strip()):return {}
 if type(binding.worksheet_row) is not int or binding.worksheet_row!=row:return {}
 sheet=str(getattr(unit,'sheet','') or '');section=str(getattr(context,'section_title','') or '')
 provenance=getattr(context,'section_provenance',None)
 if (not sheet or provenance is None or provenance.value!=section or provenance.source_quote!=section
     or provenance.source_container not in {'metadata.section','metadata.section_title','metadata.heading'}
     or provenance.source_span!=(0,len(section))
     or provenance.source_sha256!=hashlib.sha256(section.encode('utf-8')).hexdigest()
     or provenance.structural_unit_id!=unit.unit_id or provenance.structural_group_id!=(unit.structural_group_id or unit.unit_id)):return {}
 section_match=re.fullmatch(r'Sheet:\s*(?P<sheet>.*?)(?:\s*/\s*(?:Rows\s+\d+(?:-\d+)?|Sheet context))?',section,re.I)
 header=re.fullmatch(r'\[(?P<workbook>.+?)\s*>\s*Sheet:\s*(?P<sheet>[^>\]\r\n]+?)(?:\s*>[^\]\r\n]*)?\]',source.splitlines()[0] if source else '',re.I)
 if (not section_match or section_match.group('sheet').strip()!=sheet or not header
     or header.group('sheet').strip()!=sheet or header.group('workbook').strip()!=unit.workbook):return {}
 result={'sheet':sheet,'row':row}
 letter=re.search(r'\(([A-Z]{1,3})\)\s*$',column);cell=str(binding.cell_coordinate or '')
 if letter:
  if binding.column_letter!=letter.group(1) or cell!=f'{letter.group(1)}{row}':return {}
  result['cell']=cell
 elif binding.column_letter or cell:return {}
 return result

def document_fact_errors(scope,fact):
 if not scope:return ()
 try:value=json.loads(scope)
 except (ValueError,TypeError):return ('document_scope_invalid',)
 if not isinstance(value,dict):return ('document_scope_invalid',)
 if value.get('schema')!='project-x-named-document-v1':return () if 'schema' not in value and value.get('kind') in {'invoice','quote'} else ('document_scope_invalid',)
 if set(value)!={'schema','kind','title'} or value['kind'] not in {'spreadsheet','document','presentation'} or not isinstance(value['title'],str):return ('document_scope_invalid',)
 from financial_semantics import source_document_label
 label=source_document_label(fact)
 return () if reference_matches(DocumentReference(value['kind'],value['title']),label) else ('named_document_scope_incompatible',)

def location_fact_errors(slots,fact):
 if not slots:return ()
 requested=tuple(slots.split('|'));locations=fact_source_locations(fact)
 if any(slot not in {'sheet','row','cell'} for slot in requested):return ('source_location_request_invalid',)
 return tuple('source_'+slot+'_unproven' for slot in requested if slot not in locations)

def _owned_identifier_values(fact):
 """Identifiers are owned by an entity/model field, never its whole row."""
 context=getattr(fact,'evidence_context',None)
 if context is None:return ()
 from evidence_context import structural_binding_validation_errors
 if structural_binding_validation_errors(context):return ()
 unit=getattr(context,'structural_evidence_unit',None)
 if fact_source_locations(fact):
  row=str(getattr(fact,'row_header','') or '')
  if not row or row!=getattr(unit,'row_label',''):return ()
  entities=[binding for binding in unit.field_bindings if binding.role=='entity' and binding.value==row
            and binding.structural_group_id==(unit.structural_group_id or unit.unit_id)
            and len(binding.source_span)==2 and 0<=binding.source_span[0]<binding.source_span[1]<=len(unit.local_text)
            and unit.local_text[slice(*binding.source_span)]==row]
  return (row,) if len(entities)==1 else ()
 if getattr(fact,'fact_type','')=='text_proposition' and getattr(fact,'predicate','')=='has_attribute':
  target=str(getattr(fact,'evidence_target_entity','') or '')
  records=[record for record in context.semantic_provenance if record.field=='evidence_target_entity']
  if (target and target==getattr(context,'evidence_target_entity','') and len(records)==1 and unit is not None):
   import hashlib
   record=records[0];span=record.source_span;source=context.local_proposition
   if (record.value==target and record.source_container=='local_proposition'
       and ' '.join(record.source_quote.split())==' '.join(target.split())
       and isinstance(span,(tuple,list)) and len(span)==2 and all(type(value) is int for value in span)
       and 0<=span[0]<span[1]<=len(source) and source[slice(*span)]==record.source_quote
       and record.source_sha256==hashlib.sha256(record.source_quote.encode('utf-8')).hexdigest()
       and record.structural_unit_id==unit.unit_id
       and record.structural_group_id==(unit.structural_group_id or unit.unit_id)):
    return (target,)
 return ()

def rendered_location_support(text,facts):
 """Return only proved identity/location spans; reject unsupported locations."""
 spans=[];errors=[];locations=[fact_source_locations(f) for f in facts]
 for match in re.finditer(r'(?:\b(?:row|cell)\s+|(?:שורה|תא)\s+)(?P<value>[A-Z]{1,3}[1-9]\d*|[1-9]\d*)',text,re.I):
  value=match.group('value');key='cell' if re.search('[A-Za-z]',value) else 'row';wanted=value.upper() if key=='cell' else int(value)
  if any(v.get(key)==wanted for v in locations):spans.append(match.span('value'))
  else:errors.append('rendered_'+key+'_unbound')
 # Every explicitly rendered sheet label needs ownership, including an extra
 # wrong sheet alongside the correct one. Sheet names may contain spaces.
 sheet_marker=r'(?<!\w)(?:(?:work)?sheet|[בה]?גיליון)\s*[:=]?\s+'
 boundary=r'(?=\s+(?:(?:and\s+)?(?:at\s+)?(?:row|cell|(?:work)?sheet)\b|(?:ו?ב?(?:שורה|תא|גיליון))\b)|[,;.!?\n\[\]]|$)'
 for match in re.finditer(sheet_marker+r'(?P<value>[^,;.!?\n\[\]]{1,160}?)'+boundary,text,re.I):
  value=match.group('value').strip(' \"\'״“”*`')
  if not any(str(item.get('sheet','')).casefold()==value.casefold() for item in locations):
   errors.append('rendered_sheet_unbound')
 for item in locations:
  if item.get('sheet'):
   for m in re.finditer(r'(?<!\w)'+re.escape(str(item['sheet']))+r'(?!\w)',text):spans.append(m.span())
  if item.get('cell'):
   for m in re.finditer(r'(?<!\w)'+re.escape(str(item['cell']))+r'(?!\w)',text,re.I):spans.append(m.span())
 allowed={m.group().casefold() for f in facts for value in _owned_identifier_values(f) for m in _IDENTIFIER.finditer(value)}
 for m in _IDENTIFIER.finditer(text):
  if m.group().casefold() in allowed:spans.append(m.span())
  else:errors.append('rendered_identifier_unbound')
 return tuple(spans),tuple(dict.fromkeys(errors))

def technical_attribute_query(query):
 """Recognize a bounded nominal equipment lookup, never an event request.

 Only role words belong to this grammar. Unknown actions or qualifications
 stay on the ordinary request path; they cannot become a has_attribute fact
 merely because the request also contains a technical noun.
 """
 value=str(query or '')
 if not re.search(r'\b(?:model|controller|panel|device|screen)\b|(?:בקר|מסך|מכשיר|דגם)',value,re.I):return '',''
 reference=named_document_reference(value)
 if reference and ',' in value:value=value.split(',',1)[1].strip()
 value=re.sub(r'^\s*(?:what\s+(?:is\s+)?|which\s+|איזה\s+|איזו\s+|מהו\s+|מה\s+)', '',value,flags=re.I)
 value=re.sub(r'^(?:the\s+)?(?:model\s+(?:of\s+)?|דגם\s+)', '',value,flags=re.I)
 context='';match=re.search(r'(?P<context>לחדר\s+[^,;?]+)|\b(?:for|in)\s+(?:the\s+)?(?P<english>[^,;?]*?\broom)\s*$',value,re.I)
 if match:
  context=match.group('context') or match.group('english');context=re.sub(r'^ל(?=חדר)','',context)
  value=value[:match.start()]
 value=re.sub(r'\s*(?:מוצע(?:ת)?|מוצעים|is\s+(?:offered|proposed)|(?:is\s+)?(?:offered|proposed))\s*$', '',value,flags=re.I).strip(' ,;?!')
 words=_words(value)
 role_words={
  'model','controller','panel','device','screen','display','management','control','touch',
  'central','main','backup','network','audio','video','room','system','administrative','wireless',
  'wired','wall','desk','desktop','portable','remote','local','digital','analog',
  'lighting','access','security','automation','meeting','conference',
  'בקר','מסך','מכשיר','דגם','לניהול','ניהול','שליטה','בקרה','מגע','תצוגה',
  'ראשי','מרכזי','גיבוי','מערכת','תקשורת','חדר','וידאו','אודיו','תאורה','אבטחה','גישה',
 }
 heads={'model','controller','panel','device','screen','display','בקר','מסך','מכשיר','דגם'}
 if not 0<len(words)<=6 or not set(words)<=role_words or not set(words)&heads:return '',''
 return ' '.join(words)[:160],context[:160]

def attribute_context_errors(requested,fact):
 if not requested:return ()
 from evidence_context import structural_binding_validation_errors
 from config_env import hebrew_token_variants
 context=getattr(fact,'evidence_context',None)
 if context is None or structural_binding_validation_errors(context):return ('attribute_context_provenance_invalid',)
 observed=str(getattr(context,'governing_context','') or '')+' '+str(getattr(fact,'source_claim','') or '')
 observed=re.sub(r'(?<![\u05d0-\u05ea])[לבה]חדר(?![\u05d0-\u05ea])','חדר',observed)
 def forms(word):return hebrew_token_variants(word) if re.fullmatch(r'[\u05d0-\u05ea]+',word) else {word}
 observed_forms=set().union(*(forms(word) for word in _words(observed))) if _words(observed) else set()
 return () if all(forms(word)&observed_forms for word in _words(requested)) else ('attribute_context_incompatible',)

def rendered_required_location_errors(text,facts,requirements):
 from semantic_contracts import requirement_dimension_value
 errors=[]
 for requirement in requirements:
  slots=requirement_dimension_value(requirement,'source_location')
  if not slots:continue
  linked=[f for f in facts if requirement.id in getattr(f,'requirement_ids',())]
  locations=[fact_source_locations(f) for f in linked]
  for slot in slots.split('|'):
   present=False
   for location in locations:
    value=location.get(slot)
    if value is None:continue
    if slot=='row':pattern=r'(?:\brow\s+|שורה\s+)'+str(value)+r'(?!\d)'
    else:pattern=r'(?<!\w)'+re.escape(str(value))+r'(?!\w)'
    if re.search(pattern,text,re.I):present=True;break
   if not present:errors.append('requested_'+slot+'_omitted')
 return tuple(dict.fromkeys(errors))

def requested_organization_owner(query, document_reference=None):
 """Return the literal actor of a strongly requested organization list.

 A supplied reference is the compiler-owned unique full-request reference;
 the individual list clause may no longer contain its source declaration.
 Unresolved pronouns stay literal rather than acquiring a document identity.
 """
 value=str(query or '')
 from request_understanding import requested_enumerations,_quoted_request_spans
 enumerations=requested_enumerations(value)
 if len(enumerations)!=1:return ''
 start=enumerations[0].item_span[1]
 stop=re.search(r'[?;\n]',value[start:])
 end=start+stop.start() if stop else len(value)
 patterns=(
  r'\b(?:around|along)\s+which\s+(?P<owner>[^?;,\n]+?)\s+(?:is|are|was|were)\s+(?:organized|organised|structured|arranged)\b',
  r'(?<!\w)ש(?:סביבם|סביבן|בהם|בהן|לאורכם|לאורכן)\s+(?P<owner>[^?;,\n]+?)\s+(?:(?:הוא|היא)\s+)?(?:מאורגן|מאורגנת|מאורגנים|מאורגנות)(?!\w)',
 )
 quoted=_quoted_request_spans(value)
 matches=[match for pattern in patterns for match in re.finditer(pattern,value,re.I)
  if start<=match.start()<match.end()<=end
  and not any(a<match.end() and match.start()<b for a,b in quoted)]
 if len(matches)!=1:return ''
 owner=matches[0].group('owner').strip()
 if not owner or len(owner)>160 or len(owner.split())>12:return ''
 local=named_document_reference(value)
 reference=local or document_reference
 if local is not None and document_reference is not None and local!=document_reference:reference=None
 scoped=(isinstance(reference,DocumentReference) and reference.kind in {'document','presentation'} and bool(reference.title))
 if scoped and owner=='it':return 'presentation' if reference.kind=='presentation' else 'document'
 if scoped and owner in {'הוא','היא'}:return 'מצגת' if reference.kind=='presentation' else 'מסמך'
 return owner


def requested_list_scope(query):
 reference=named_document_reference(query or '')
 match=re.search(r'(?:שלבי|שלבים\s+של)\s+(?P<he>.+?)(?=\s+ב(?:מצגת|מסמך|קובץ)|[,;?]|$)|\b(?:steps|stages|items)\s+(?:(?:of|in|on|for)\s+)?(?:the\s+)?(?P<en>.+?)(?=\s+(?:in|from)\s+(?:the\s+)?(?:presentation|document|file)\b|[,;?]|$)',query or '',re.I)
 if not match:
  # An explicit enumeration in an organization question selects a literal
  # source-owned collection; neither its members nor order come from query.
  if not re.search(r'\b(?:organized|organised|structured|arranged)\b|'
      r'(?<!\w)(?:מאורגן|מאורגנת|מאורגנים|מאורגנות)(?!\w)',query or '',re.I):
   return '','',''
  from request_understanding import requested_enumerations
  enumerations=requested_enumerations(query or '')
  if len(enumerations)!=1:return '','',''
  heading=(query or '')[slice(*enumerations[0].item_span)]
  collection_roles={'צירים','ממדים','מימדים','נושאים','חלקים','שלבים','תחומים','פרקים'}
  # These are closed collection-role nouns, not arbitrary owner names.
  if heading.startswith('ה') and heading[1:] in collection_roles:heading=heading[1:]
  ordered=bool(re.search(r'\b(?:in\s+(?:their\s+)?order|ordered|sequence)\b|(?:לפי\s+סדר|בסדר)',query or '',re.I))
  return heading,'','source_order' if ordered else ''
 heading=(match.group('he') or match.group('en')).strip()
 if not heading or len(heading)>160:return '','',''
 ordered=bool(re.search(r'\b(?:in\s+(?:their\s+)?order|ordered|sequence)\b|(?:לפי\s+סדר|בסדר)',query or '',re.I))
 return heading,reference.title if reference else '', 'source_order' if ordered else ''
