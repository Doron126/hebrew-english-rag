"""Shared bounds for request-specific output allocation.

These limits do not change model context or bypass prompt admission.  The
overview budget is configurable; the protocol cap remains a safety boundary.
"""
import os

MAX_OUTPUT_TOKENS = 1024
DEFAULT_OVERVIEW_OUTPUT_TOKENS = 1024
SECTION_OUTPUT_TOKENS = 320


def overview_output_tokens():
    """Return the configured overview allowance; reject invalid configuration."""
    raw = os.getenv("GEMMA_OVERVIEW_MAX_NEW_TOKENS", str(DEFAULT_OVERVIEW_OUTPUT_TOKENS))
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid GEMMA_OVERVIEW_MAX_NEW_TOKENS") from exc
    if not 1 <= value <= MAX_OUTPUT_TOKENS:
        raise ValueError("GEMMA_OVERVIEW_MAX_NEW_TOKENS must be between 1 and 1024")
    return value
