"""Helpers to resolve and join data on (Gene, Drug) pairs."""

from __future__ import annotations

from typing import Iterable, Mapping, Tuple

import pandas as pd

from ..common.id_utils import stable_row_id
from ..common.text_utils import clean_str

GENE_FIELDS: Tuple[str, ...] = ("Gene", "gene_name", "gene_id")
DRUG_FIELDS: Tuple[str, ...] = ("drug_name_pgx", "drug_name_chembl", "drug_name", "drug_norm")


def resolve_gene(row: Mapping[str, object]) -> str:
    """Resolve the best gene string from a row."""
    for field in GENE_FIELDS:
        if field in row:
            value = clean_str(row[field])
            if value:
                return value
    return ""


def resolve_drug(row: Mapping[str, object]) -> str:
    """Resolve the best drug string from a row."""
    for field in DRUG_FIELDS:
        if field in row:
            value = clean_str(row[field])
            if value:
                return value
    return ""


def resolve_entities(row: Mapping[str, object]) -> tuple[str, str]:
    """Resolve both gene and drug from a mapping."""
    gene = resolve_gene(row)
    drug = resolve_drug(row)
    return gene, drug


def ensure_resolved(
    df: pd.DataFrame,
    *,
    row_id_col: str = "row_id",
    context: str = "",
) -> pd.DataFrame:
    """Return a copy of df with canonical Gene, drug_name, and row_id columns."""
    resolved = df.copy()
    genes: list[str] = []
    drugs: list[str] = []
    row_ids: list[str] = []

    for idx, row in resolved.iterrows():
        gene, drug = resolve_entities(row)
        if not gene or not drug:
            raise ValueError(
                f"Unresolved Gene/Drug at index {idx} {context!s}: gene={gene!r} drug={drug!r}"
            )
        genes.append(gene)
        drugs.append(drug)
        row_ids.append(stable_row_id(gene, drug, existing=row.get(row_id_col)))

    resolved["Gene"] = genes
    resolved["drug_name"] = drugs
    resolved[row_id_col] = row_ids
    return resolved


def _decode_index(value, lookup: list[str]) -> str:
    idx_text = clean_str(value)
    if idx_text == "":
        return ""
    idx = int(float(idx_text))
    if idx < 0 or idx >= len(lookup):
        raise IndexError(f"Index {idx} out of bounds for lookup of size {len(lookup)}")
    return lookup[idx]


def resolve_ml_entities(
    ml_df: pd.DataFrame,
    *,
    golden_resolved: pd.DataFrame | None = None,
    gene_lookup: list[str] | None = None,
    drug_lookup: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Attach Gene/drug names to the ML dataframe using encoded indices."""
    if gene_lookup is None or drug_lookup is None:
        if golden_resolved is None:
            raise ValueError("Either golden_resolved or explicit lookup lists must be provided.")
        gene_lookup = sorted(golden_resolved["Gene"].unique())
        drug_lookup = sorted(golden_resolved["drug_name"].unique())

    resolved = ml_df.copy()

    if "Gene" not in resolved.columns:
        if "Gene_enc" not in resolved.columns:
            raise KeyError("ML dataframe requires Gene or Gene_enc column.")
        resolved["Gene"] = resolved["Gene_enc"].apply(lambda v: _decode_index(v, gene_lookup))

    if "drug_name" not in resolved.columns:
        if "drug_name_pgx_enc" not in resolved.columns:
            raise KeyError("ML dataframe requires drug_name or drug_name_pgx_enc column.")
        resolved["drug_name"] = resolved["drug_name_pgx_enc"].apply(lambda v: _decode_index(v, drug_lookup))

    resolved = ensure_resolved(resolved, context="(ml resolved)")
    return resolved, gene_lookup, drug_lookup


def merge_on_entities(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    how: str = "left",
    suffixes: tuple[str, str] = ("_left", "_right"),
) -> pd.DataFrame:
    """Merge two DataFrames using resolved Gene/Drug pairs."""
    left_prepped = ensure_resolved(left, context="(left)")
    right_prepped = ensure_resolved(right, context="(right)")
    merged = left_prepped.merge(
        right_prepped.drop_duplicates(subset=["row_id"]),
        on=["row_id", "Gene", "drug_name"],
        how=how,
        suffixes=suffixes,
    )
    return merged
