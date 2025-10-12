"""Filter PGx payloads and export gene/drug toxic interactions."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io_utils import read_csv, write_csv
from src.common.text_utils import clean_str

logger = logging.getLogger(__name__)

DETAIL_COLUMNS = [
    "Gene",
    "drug_name",
    "summary",
    "Phenotype Category",
    "Phenotype_Dominant",
    "Significance",
    "evidence_strength",
    "evidence_strength_num",
    "moa",
    "indications",
    "atc",
    "drug_frequency",
    "gene_frequency",
    "Variant/Haplotypes",
    "Alleles",
    "Sentence",
    "Notes",
    "pmids",
    "best_pmid",
    "best_title",
    "best_year",
    "weighted_prob",
    "p_xgb",
    "p_cat",
    "p_targetox",
    "y_xgb",
    "y_cat",
    "y_targetox",
    "evidence_support",
]

TOXIC_COLUMNS = [
    "Gene",
    "drug_name",
    "weighted_prob",
    "p_xgb",
    "p_cat",
    "p_targetox",
    "y_xgb",
    "y_cat",
    "y_targetox",
    "summary",
    "evidence_support",
]

DEFAULT_FALLBACK_EVIDENCE = "Evidence supported by curated pharmacogenomic knowledge base."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export toxic interactions or full details for specific gene/drug pairs.")
    parser.add_argument("--payload", default="outputs/quick_sample/final_payload.summarized.csv", help="Path to payload CSV (summarized output).")
    parser.add_argument("--gene", required=True, help="Gene symbol to filter (case-insensitive).")
    parser.add_argument("--drug", help="Optional drug name to filter (case-insensitive).")
    parser.add_argument("--out", help="Optional output CSV path.")
    return parser.parse_args()


def _norm(value: str) -> str:
    return clean_str(value).lower()


def to_float(value: str) -> float:
    text = clean_str(value)
    if not text:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def compute_evidence(row: pd.Series) -> str:
    candidates = [
        ("pmids", True),
        ("best_pmid", True),
        ("Sentence", False),
        ("Notes", False),
        ("evidence_strength", False),
    ]
    for col, is_pmid in candidates:
        if col in row:
            value = clean_str(row[col])
            if value and value != "-":
                return f"PMID: {value}" if is_pmid else value
    return DEFAULT_FALLBACK_EVIDENCE


def enrich_with_evidence(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["evidence_support"] = df.apply(compute_evidence, axis=1)
    return df


def export_gene_only(df: pd.DataFrame, gene: str, out_path: Path) -> Path:
    filtered = enrich_with_evidence(df)
    filtered["weighted_pred_numeric"] = filtered["weighted_pred"].apply(to_float)
    toxic = filtered[filtered["weighted_pred_numeric"] >= 0.5].copy()
    if toxic.empty:
        logger.warning("No toxic interactions found for gene %s; exporting empty file.", gene)
    columns = [col for col in TOXIC_COLUMNS if col in toxic.columns]
    toxic = toxic[columns]
    numeric_cols = ["weighted_prob", "p_xgb", "p_cat", "p_targetox"]
    for col in numeric_cols:
        if col in toxic.columns:
            toxic[col] = toxic[col].apply(to_float)
    if "weighted_prob" in toxic.columns:
        toxic = toxic.sort_values(by="weighted_prob", ascending=False)
    return write_csv(toxic, out_path)


def export_gene_drug(df: pd.DataFrame, out_path: Path) -> Path:
    detailed = enrich_with_evidence(df)
    columns = [col for col in DETAIL_COLUMNS if col in detailed.columns]
    detailed = detailed[columns]
    numeric_cols = ["weighted_prob", "p_xgb", "p_cat", "p_targetox"]
    for col in numeric_cols:
        if col in detailed.columns:
            detailed[col] = detailed[col].apply(to_float)
    return write_csv(detailed, out_path)


def determine_output_path(args: argparse.Namespace, gene: str, drug: str | None) -> Path:
    if args.out:
        return Path(args.out)
    safe_gene = _norm(gene).replace(" ", "_")
    if drug:
        safe_drug = _norm(drug).replace(" ", "_")
        return Path(f"export_{safe_gene}_{safe_drug}.csv")
    return Path(f"export_{safe_gene}_toxic.csv")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[export] %(message)s")
    args = parse_args()

    payload = read_csv(args.payload)
    gene_norm = _norm(args.gene)

    gene_mask = payload["Gene"].apply(lambda value: _norm(value) == gene_norm)
    gene_rows = payload[gene_mask].copy()
    if gene_rows.empty:
        raise ValueError(f"Gene '{args.gene}' not found in payload {args.payload}")

    output_path = determine_output_path(args, args.gene, args.drug)

    if args.drug:
        drug_norm = _norm(args.drug)
        drug_mask = gene_rows["drug_name"].apply(lambda value: _norm(value) == drug_norm)
        subset = gene_rows[drug_mask]
        if subset.empty:
            raise ValueError(f"No records found for gene '{args.gene}' with drug '{args.drug}'.")
        export_gene_drug(subset, output_path)
    else:
        export_gene_only(gene_rows, args.gene, output_path)

    logger.info("Exported data to %s", output_path)


if __name__ == "__main__":
    main()
