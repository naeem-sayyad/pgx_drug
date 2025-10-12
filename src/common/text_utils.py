"""Text helpers for PGx pipeline."""

from __future__ import annotations

import re
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")


def safe_str(value: Any) -> str:
    """Return a stripped string, replacing None/NaN with empty string."""
    if value is None:
        return ""
    text = str(value)
    if text.lower() in {"nan", "none", "na"}:
        return ""
    return text.strip()


def clean_str(value: Any) -> str:
    """Normalize whitespace and strip surrounding spaces."""
    text = safe_str(value)
    return _WHITESPACE_RE.sub(" ", text).strip()


def pct_fmt(prob: float) -> str:
    """Format a probability in percent with two decimal places."""
    return f"{prob * 100:.2f}%"
