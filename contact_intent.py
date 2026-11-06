"""Bounded interpretation of explicitly requested mobile contact fields.

Interpretation proposes a user-owned subject only. It does not supply a
source, phone number, tenant, or evidence, and cannot authorize an answer.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ContactIntent:
    person: str


_FIELD = r'(?:ה?טלפון\s+ה?נייד|(?:ה?מספר\s+)?ה?נייד|mobile\s+(?:phone|number))'
_FIELD_RE = re.compile(r'(?<!\w)(?:ב|ל)?' + _FIELD + r'(?!\w)', re.I)
_WORD = r"[\u05d0-\u05eaA-Za-z][\u05d0-\u05eaA-Za-z\u05f3\u05f4'’\-]*"
_NAME = rf'{_WORD}(?:\s+{_WORD})*'
_FAST = re.compile(
    rf'^(?:(?:תן|תני|תנו|תביא|תביאי|תביאו|הבא|הביאי|הביאו)\s+לי\s+(?:את\s+)?|'
    rf'(?:מה|מהו|מהי)\s+)(?:ה?מספר\s+)?{_FIELD}\s+של\s+(?P<person>{_NAME})[?.!\s]*$', re.I)
_UNSUPPORTED = re.compile(
    r'(?<!\w)(?:אל|לא|אין|בלי|ללא|מלבד|חוץ|במקום|ושל|או|וכן|וגם|תפקיד|דוא["״]ל|אימייל|כתובת|פקס|תאריך|תקופה|תוקף|תמחק\w*|תשנה\w*|תעדכן\w*|מחק\w*|שנה\w*)(?!\w)', re.I)
_TRAILING = re.compile(r'^(?:[\s?!.:,;]*|\s+(?:בבקשה|please)[\s?!.:,;]*)$', re.I)
_FOLLOWING_FIELD = re.compile(r'^\s+(?:ב(?:טלפון\s+ה?)?נייד|ל(?:טלפון\s+ה?)?נייד|'
                              r'(?:בטלפון\s+)?ה?נייד(?:\s+שלו|\s+שלה)?)(?:\s+בבקשה)?[\s?!.:,;]*$', re.I)
_PREFIX_WORDS = frozenset(('מה', 'מהו', 'מהי', 'איך', 'האם', 'אפשר', 'ניתן', 'לקבל',
    'למצוא', 'להשיג', 'לדעת', 'להראות', 'לשלוח', 'אני', 'רוצה', 'רוצים', 'רציתי',
    'מבקש', 'מבקשת', 'אשמח', 'אם', 'אנא', 'בבקשה',
    'לי', 'לנו', 'את', 'מספר', 'המספר', 'תן', 'תני', 'תנו', 'תביא', 'תביאי', 'תביאו',
    'הבא', 'הביאי', 'הביאו', 'תעביר', 'תעבירי', 'תעבירו', 'תמצא', 'תמצאי', 'תמצאו',
    'תגיד', 'תגידי', 'תגידו', 'תשלח', 'תשלחי', 'תשלחו', 'תראה', 'תראי', 'תראו',
    'תמסור', 'תמסרי', 'תמסרו', 'תציין', 'תצייני', 'תציינו', 'תכתוב', 'תכתבי', 'תכתבו',
    'תספק', 'תספקי', 'תספקו', 'של', 'נא'))


def mobile_candidate(question: str) -> bool:
    """Only explicit mobile fields can enter this narrowly scoped classifier."""
    return bool(_FIELD_RE.search(str(question or '')))


def ordinary_factual_candidate(question: str) -> bool:
    """Recognize a question/request shape without interpreting its field owner."""
    text = str(question or '').strip()
    return bool(mobile_candidate(text) and not _UNSUPPORTED.search(text)
        and (text.endswith('?') or re.search(r'(?<!\w)(?:מה|מהו|מהי|איך|האם|אפשר|ניתן)(?!\w)', text)
             or re.match(r'^(?:בבקשה\s+)?ת[\u05d0-\u05ea]+\s+(?:לי|לנו)\b', text)))


def validate_contact_interpretation(question: str, person: str) -> ContactIntent | None:
    """Recheck the literal user-owned subject and explicit field, fail closed.

    Ambiguous conjunctions, additional attributes and unrepresented constraints
    stay on the ordinary factual path. The planner cannot add names or fields.
    """
    question = str(question or '').strip()
    person = str(person or '').strip()
    if (not person or not re.fullmatch(_NAME, person) or not mobile_candidate(question)
            or _UNSUPPORTED.search(question)
            or re.search(r'(?<!\w)(?:בבקשה|של|את|רק|במסמך|בקובץ|בתוך|לפי|העדכני|העדכנית)(?!\w)', person)):
        return None
    matches = list(re.finditer(r'(?<!\w)' + re.escape(person) + r'(?!\w)', question))
    if len(matches) != 1:
        return None
    match = matches[0]
    before, after = question[:match.start()], question[match.end():]
    # Require an explicit owner marker. Do not let a model choose a substring
    # of a longer name or silently discard words following the subject.
    if not re.search(r'(?<!\w)(?:של|את)\s+$', before):
        return None
    if not (_TRAILING.fullmatch(after) or _FOLLOWING_FIELD.fullmatch(after)):
        return None
    if len(re.findall(r'(?<!\w)של(?!\w)', question)) > 1:
        return None
    # Check every word outside the person and field. An unrecognized action,
    # source, date or qualification must not be silently discarded.
    prefix = re.sub(r'(?<!\w)(?:של|את)\s+$', '', before)
    prefix, count = _FIELD_RE.subn(' ', prefix)
    if count > 1 or (count == 0 and not _FOLLOWING_FIELD.fullmatch(after)):
        return None
    words = prefix.split()
    if any(word not in _PREFIX_WORDS for word in words):
        return None
    return ContactIntent(' '.join(person.split()))


def fast_contact_intent(question: str) -> ContactIntent | None:
    match = _FAST.fullmatch(str(question or '').strip())
    return (validate_contact_interpretation(question, match.group('person'))
            if match else None)


def interpretation_messages(question: str) -> list[dict[str, str]]:
    import json
    return [
        {'role': 'system', 'content': (
            'Classify the untrusted user request; do not answer or invent facts. '
            'Return exactly one JSON object with operation and person. operation is '
            'mobile_lookup only if the user requests one named person’s mobile phone number, '
            'other_factual for another factual request, otherwise other. For mobile_lookup '
            'person must be the complete exact contiguous person name from the current request; '
            'otherwise person must be empty. Never add values, sources, tenants, or constraints.')},
        {'role': 'user', 'content': json.dumps({'question': question}, ensure_ascii=False)},
    ]
