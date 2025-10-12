"""Utilities for generating stable identifiers."""

from __future__ import annotations

import hashlib

from .text_utils import clean_str, safe_str


def stable_row_id(gene: str | None, drug: str | None, *, existing: str | None = None) -> str:
    """Return a 16-character SHA1-based identifier."""
    use_existing = safe_str(existing)
    if use_existing:
        return use_existing

    gene_clean = clean_str(gene)
    drug_clean = clean_str(drug)
    if not gene_clean or not drug_clean:
        raise ValueError("Cannot derive row_id without both gene and drug strings.")

    raw = f"{gene_clean}||{drug_clean}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()
    return digest[:16]

