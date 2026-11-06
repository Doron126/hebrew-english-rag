"""Project-local environment loading with deterministic precedence.

The application historically used both ``.env`` and a legacy file named
``env``.  Load both without replacing values already supplied by the process:

    operating-system environment > .env > env

This module deliberately uses only the Python standard library so configuration
is available before optional application dependencies are imported.
"""

from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import MutableMapping


DEFAULT_EMBED_MODEL = "intfloat/multilingual-e5-large-instruct"
EMBEDDING_FORMAT_REVISION = "e5-v1"

_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
TENANT_ID_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9_-]{0,62}[A-Za-z0-9])?$"
)

_SEARCH_QUOTES = frozenset("\"'`\u05f3\u05f4\u2018\u2019\u201a\u201b\u201c\u201d\u201e\u201f")
_SEARCH_DASHES = frozenset("-\u05be\u2010\u2011\u2012\u2013\u2014\u2015\u2212")
_HEBREW_TOKEN_RE = re.compile(r"[\u05d0-\u05ea]+")
# Clitic stripping and inflection are intentionally allowlisted. Without a
# morphological dictionary, blindly removing ב/ל/כ/ה changes valid words into
# other valid words (כתיבה -> תיבה, לבנים -> בנים, הספק -> ספק).
_HEBREW_INFLECTION_GROUPS = (
    ("אבטחה", "אבטחת", "אבטחות"), ("אנרגיה", "אנרגיות"),
    ("ארגון", "ארגונים"), ("אתר", "אתרים"), ("גיבוי", "גיבויים"),
    ("דוח", "דוחות"), ("הטבה", "הטבות"), ("הכנסה", "הכנסות"),
    ("הסכם", "הסכמים"), ("הספק", "הספקים"), ("הוצאה", "הוצאות"),
    ("השפעה", "השפעות"), ("החלטה", "החלטות"), ("הערכה", "הערכות"),
    ("הפעלה", "הפעלות"), ("הצלחה", "הצלחות"), ("הכשרה", "הכשרות"),
    ("התחייבות", "התחייבויות"), ("השקעה", "השקעות"),
    ("חברה", "חברת", "חברות"), ("חודש", "חודשים"), ("יום", "ימים"),
    ("כוח", "כוחות"), ("לקוח", "לקוחות"), ("מדד", "מדדים"),
    ("מחיר", "מחירים"), ("מוקד", "מוקדים"), ("מסמך", "מסמכים"),
    ("מערכת", "מערכות"),
    ("מתקן", "מתקנים"), ("נתון", "נתונים"), ("נוהל", "נהלים"),
    ("עובד", "עובדים"), ("עובדת", "עובדות"), ("עלות", "עלויות"),
    ("פרויקט", "פרויקטים"), ("צריכה", "צריכות"), ("קובץ", "קבצים"),
    ("קיבולת", "קיבולות"), ("שירות", "שירותים"), ("שעה", "שעות"),
    ("שנה", "שנים"), ("סיכון", "סיכונים"), ("תאריך", "תאריכים"),
    ("תוכנית", "תוכניות"), ("תכנית", "תכניות"), ("תפוקה", "תפוקות"),
    ("תקציב", "תקציבים"), ("דרישה", "דרישות"),
)
_HEBREW_INFLECTIONS = {
    form: frozenset(group)
    for group in _HEBREW_INFLECTION_GROUPS
    for form in group
}
_HEBREW_MORPHOLOGY_BASES = frozenset(
    set(_HEBREW_INFLECTIONS).union({"ישראל", "מדיניות", "מידע", "קרמיקה"})
)



def embedding_model_kwargs(device: str) -> dict[str, str]:
    """Select native CPU precision explicitly instead of checkpoint FP16.

    Recent Transformers versions honor a checkpoint's automatic dtype. Loading
    those weights as FP16 on CPUs without native FP16 instructions can turn
    ordinary embedding inference into extremely slow emulated arithmetic.
    Strings keep this shared configuration helper independent of PyTorch.
    """
    selected = str(device or "cpu").strip().casefold().split(":", 1)[0]
    return {"dtype": "float32"} if selected == "cpu" else {}


def validate_tenant_id(value: str) -> str:
    """Return the canonical safe tenant identifier shared by every entrypoint."""
    tenant = str(value or "").strip()
    if not TENANT_ID_RE.fullmatch(tenant):
        raise ValueError(
            "Invalid tenant ID. Use 1-64 ASCII letters, numbers, hyphens, "
            "or underscores; begin and end with a letter or number."
        )
    return tenant


def _is_allowed_clitic_base(candidate: str) -> bool:
    if candidate in _HEBREW_MORPHOLOGY_BASES:
        return True
    return (
        candidate.startswith("ה")
        and candidate[1:] in _HEBREW_MORPHOLOGY_BASES
    )


def normalize_search_text(value: str) -> str:
    """Return a stable, display-independent form for lexical search.

    Hebrew PDF and user text commonly differ only by niqqud, bidi controls,
    acronym quote style, or the choice of Hebrew maqaf versus an ASCII dash.
    Those differences carry no retrieval meaning, so normalize them before
    tokenization.  The returned value is for matching only; original text is
    still retained for display and citations.
    """
    decomposed = unicodedata.normalize("NFKD", str(value or ""))
    normalized: list[str] = []
    for character in decomposed:
        category = unicodedata.category(character)
        if category in {"Mn", "Me", "Cf"}:
            # Mn/Me removes Hebrew niqqud and cantillation. Cf removes bidi and
            # zero-width formatting controls that otherwise split equal terms.
            continue
        if character in _SEARCH_QUOTES:
            # Normalize acronym spellings such as שב"כ / שב״כ / שבכ.
            continue
        if character in _SEARCH_DASHES:
            # Make תת־מערכת and תת-מערכת the same two-token phrase.
            normalized.append(" ")
        else:
            normalized.append(character)
    return re.sub(r"\s+", " ", "".join(normalized).casefold()).strip()


def _hebrew_prefix_bases(token: str) -> set[str]:
    """Peel only unambiguous clitic positions, preserving every full form."""
    forms = {token}
    base = token

    # Hebrew clitics can stack: ו + ב/ל/כ + ה, for example ובהחברות.
    # Deliberately do not peel מ or ש: both very often belong to a real root
    # (מחשב, מסמך, שירות), and treating them as prefixes hurts precision.
    if base.startswith("ו") and len(base) >= 3:
        candidate = base[1:]
        nested_candidate = candidate[1:] if candidate[:1] in "בלכ" else ""
        if _is_allowed_clitic_base(candidate) or _is_allowed_clitic_base(nested_candidate):
            base = candidate
            forms.add(base)
    if base[:1] in "בלכ" and len(base) >= 3:
        candidate = base[1:]
        if _is_allowed_clitic_base(candidate):
            base = candidate
            forms.add(base)
    if base.startswith("ה") and len(base) >= 3:
        candidate = base[1:]
        if candidate in _HEBREW_MORPHOLOGY_BASES:
            forms.add(candidate)
    return forms


def _hebrew_forward_inflections(token: str) -> set[str]:
    """Return only explicit, bidirectional inflections—never a guessed stem."""
    return set(_HEBREW_INFLECTIONS.get(token, {token}))


def hebrew_token_variants(token: str) -> set[str]:
    """Return conservative clitic and inflection forms for one Hebrew token.

    The function intentionally keeps complete surface forms and never returns a
    suffix-stripped lexical stem. This preserves useful matches such as
    חברה/חברת/חברות and מסמך/מסמכים without making חברה match חבר.
    """
    normalized = normalize_search_text(token)
    match = _HEBREW_TOKEN_RE.fullmatch(normalized)
    if not match:
        return {normalized} if normalized else set()

    bases = _hebrew_prefix_bases(match.group(0))
    variants: set[str] = set()
    for base in bases:
        variants.update(_hebrew_forward_inflections(base))
    return {variant for variant in variants if variant}


def _read_env_file(path: Path) -> dict[str, str]:
    """Read simple dotenv assignments without mutating ``os.environ``."""
    if not path.is_file():
        return {}

    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        if not _ENV_KEY_RE.fullmatch(key):
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        values[key] = value
    return values


def load_project_environment(
    project_dir: str | Path | None = None,
    *,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Load project configuration and return the values actually added.

    Values in ``environ`` are never overwritten. The preferred ``.env`` file
    replaces duplicate values from the legacy ``env`` file before the merged
    defaults are applied.
    """
    root = Path(project_dir) if project_dir is not None else Path(__file__).resolve().parent
    target = os.environ if environ is None else environ

    merged = _read_env_file(root / "env")
    merged.update(_read_env_file(root / ".env"))

    loaded: dict[str, str] = {}
    for key, value in merged.items():
        if key not in target:
            target[key] = value
            loaded[key] = value
    return loaded


def format_embedding_document(text: str, model_name: str) -> str:
    """Apply the document-side prefix required by non-instruct E5 models."""
    model = (model_name or "").casefold()
    value = text or ""
    if "e5" in model and "instruct" not in model:
        return "passage: " + value
    return value
