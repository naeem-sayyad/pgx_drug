"""Retrieve PMID metadata for each Gene/Drug pair."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import pandas as pd

from ..common.io_utils import read_csv, write_csv
from ..common.text_utils import clean_str
from ..features.joiners import ensure_resolved

PMID_SPLIT_RE = re.compile(r"[;,|\s]+")
PMID_EXTRACT_RE = re.compile(r"\d{5,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Attach PMID metadata to payload.")
    parser.add_argument("--inp", required=True, help="Path to final_payload.csv.")
    parser.add_argument("--gold", required=True, help="Path to golden CSV.")
    parser.add_argument("--out", required=True, help="Output directory.")
    return parser.parse_args()


def collect_pmids(value: str) -> list[str]:
    """Extract candidate PMID strings from free-form text."""
    cleaned = clean_str(value)
    if not cleaned:
        return []
    tokens: list[str] = []
    for part in PMID_SPLIT_RE.split(cleaned):
        if not part:
            continue
        for match in PMID_EXTRACT_RE.findall(part):
            if match not in tokens:
                tokens.append(match)
    return tokens


def pick_best_row(group: pd.DataFrame) -> pd.Series:
    """Select the row with strongest evidence based on numeric cues."""
    value = group.copy()
    for col in ("evidence_strength_num", "pmid_n"):
        if col in value.columns:
            try:
                value[col] = pd.to_numeric(value[col], errors="coerce").fillna(0.0)
            except Exception:
                value[col] = 0.0
        else:
            value[col] = 0.0
    value["_score"] = value["evidence_strength_num"] * 1000 + value["pmid_n"]
    value_sorted = value.sort_values("_score", ascending=False)
    return value_sorted.iloc[0]


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[pmid] %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = ensure_resolved(read_csv(args.inp), context="(payload)")
    golden = ensure_resolved(read_csv(args.gold), context="(gold)")

    pmid_cols = [col for col in golden.columns if col.lower().startswith("pmid")]
    year_cols = [col for col in golden.columns if "year" in col.lower()]
    sentence_col = "Sentence" if "Sentence" in golden.columns else None

    grouped = golden.groupby("row_id")

    pmid_list = []
    best_pmid = []
    best_title = []
    best_year = []

    logging.info("Aggregating PMID data for %d rows...", len(payload))
    for _, row in payload.iterrows():
        row_id = row["row_id"]
        if row_id in grouped.groups:
            group = grouped.get_group(row_id)
            pmids_for_row: list[str] = []
            for col in pmid_cols:
                for value in group[col]:
                    pmids_for_row.extend(collect_pmids(value))
            unique_pmids: list[str] = []
            for token in pmids_for_row:
                if token and token not in unique_pmids:
                    unique_pmids.append(token)
            pmid_list.append(",".join(unique_pmids))

            best_row = pick_best_row(group)
            best_pmid_candidate = ""
            for col in pmid_cols:
                tokens = collect_pmids(best_row[col])
                if tokens:
                    best_pmid_candidate = tokens[0]
                    break
            best_pmid.append(best_pmid_candidate)

            title_value = ""
            if sentence_col:
                title_value = clean_str(best_row[sentence_col])
            if not title_value and "Notes" in best_row.index:
                title_value = clean_str(best_row["Notes"])
            best_title.append(title_value)

            year_value = ""
            for col in year_cols:
                text = clean_str(best_row[col])
                if text:
                    year_value = text
                    break
            best_year.append(year_value)
        else:
            pmid_list.append("")
            best_pmid.append("")
            best_title.append("")
            best_year.append("")

    payload["pmids"] = [entry if entry else "-" for entry in pmid_list]
    payload["best_pmid"] = best_pmid
    payload["best_title"] = best_title
    payload["best_year"] = best_year

    output_path = out_dir / "final_payload.with_pmids.csv"
    write_csv(payload, output_path)
    logging.info("Saved PMID-augmented payload to %s", output_path)


if __name__ == "__main__":
    main()
