"""Standards-based display metadata for canonical multilingual text."""

from __future__ import annotations

from enum import Enum
import unicodedata


class BaseDirection(str, Enum):
    RTL = "rtl"
    LTR = "ltr"
    AUTO = "auto"


_RTL_CLASSES = frozenset({"R", "AL"})
_LTR_CLASSES = frozenset({"L"})


def base_direction(value: str) -> BaseDirection:
    """Match HTML ``dir=auto``: use the first strong directional character."""
    for character in str(value or ""):
        bidi_class = unicodedata.bidirectional(character)
        if bidi_class in _RTL_CLASSES:
            return BaseDirection.RTL
        if bidi_class in _LTR_CLASSES:
            return BaseDirection.LTR
    return BaseDirection.AUTO


def bidi_language(value: str) -> str:
    hebrew = sum("\u05d0" <= character <= "\u05ea" for character in str(value or ""))
    latin = sum(character.isascii() and character.isalpha() for character in str(value or ""))
    if hebrew >= 2 and latin >= 2:
        return "mixed"
    if hebrew >= 2:
        return "he"
    if latin >= 2:
        return "en"
    return "und"
