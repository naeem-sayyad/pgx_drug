"""Lightweight PubMed metadata fetcher with in-memory caching."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable

import requests

logger = logging.getLogger(__name__)

EUTILS_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
TOOL_NAME = "pgx-toxicity-pipeline"
CONTACT_EMAIL = "support@example.com"


@dataclass
class PubMedMetadata:
    pmid: str
    title: str
    journal: str | None = None
    pub_year: str | None = None
    url: str | None = None


_CACHE: Dict[str, PubMedMetadata] = {}


def _normalize_pmid(pmid: str) -> str | None:
    text = (pmid or "").strip()
    if not text:
        return None
    # Remove common separators and keep digits.
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits or None


def fetch_pubmed_metadata(pmids: Iterable[str]) -> Dict[str, PubMedMetadata]:
    """Fetch PubMed metadata for the provided PMIDs.

    Returns a mapping of original normalized PMID to metadata.
    """
    normalized = []
    for pmid in pmids:
        norm = _normalize_pmid(pmid)
        if norm and norm not in _CACHE and norm not in normalized:
            normalized.append(norm)

    if normalized:
        try:
            response = requests.get(
                EUTILS_URL,
                params={
                    "db": "pubmed",
                    "id": ",".join(normalized),
                    "retmode": "json",
                    "tool": TOOL_NAME,
                    "email": CONTACT_EMAIL,
                },
                timeout=10,
            )
            response.raise_for_status()
            payload = response.json()
            result_map = payload.get("result", {})
            for pmid in normalized:
                item = result_map.get(pmid)
                if not item:
                    continue
                metadata = PubMedMetadata(
                    pmid=pmid,
                    title=(item.get("title") or "").strip(),
                    journal=(item.get("fulljournalname") or item.get("source") or "").strip()
                    or None,
                    pub_year=(item.get("pubdate") or "").split(" ", 1)[0] or None,
                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                )
                _CACHE[pmid] = metadata
        except Exception as exc:  # pragma: no cover - network failures
            logger.warning("Failed to fetch PubMed metadata: %s", exc)

    # Build result map for the caller using original input order.
    result: Dict[str, PubMedMetadata] = {}
    for pmid in pmids:
        norm = _normalize_pmid(pmid)
        if norm and norm in _CACHE:
            result[pmid] = _CACHE[norm]
    return result

