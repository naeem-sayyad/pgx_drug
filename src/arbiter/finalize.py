"""Finalize ensemble output and prepare payload."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ..common.io_utils import read_csv, write_csv
from ..features.joiners import ensure_resolved, merge_on_entities
from .ensemble import apply_ensemble


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Finalize predictions payload.")
    parser.add_argument("--ml", required=True, help="Path to predictions.scored.csv.")
    parser.add_argument("--gold", required=True, help="Path to golden evidence CSV.")
    parser.add_argument("--out", required=True, help="Output directory.")
    parser.add_argument("--weights", nargs=3, type=float, default=(0.4, 0.3, 0.3))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[finalize] %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading predictions...")
    pred_df = ensure_resolved(read_csv(args.ml), context="(predictions)")
    pred_df = apply_ensemble(pred_df, weights=tuple(args.weights))

    logging.info("Loading golden evidence...")
    golden_df = read_csv(args.gold)
    golden_resolved = ensure_resolved(golden_df, context="(gold)")

    logging.info("Joining predictions with golden evidence...")
    merged = merge_on_entities(pred_df, golden_resolved, how="left", suffixes=("", "_gold"))

    merged["_order"] = range(len(merged))
    if merged["row_id"].duplicated().any():
        before = len(merged)
        logging.info("Detected duplicate row_id entries; retaining strongest instance per pair.")
        merged = merged.sort_values(["row_id", "weighted_prob"], ascending=[True, False])
        merged = merged.drop_duplicates(subset=["row_id"], keep="first")
        merged = merged.sort_values("_order")
        logging.info("Deduplicated payload from %d to %d rows.", before, len(merged))

    essential_cols = [
        "row_id",
        "Gene",
        "drug_name",
        "true_label",
        "p_xgb",
        "p_cat",
        "p_targetox",
        "y_xgb",
        "y_cat",
        "y_targetox",
        "weighted_prob",
        "weighted_pred",
        "majority_pred",
    ]

    merged = merged.drop(columns="_order")

    for col in essential_cols:
        if col not in merged.columns:
            merged[col] = ""

    merged["summary"] = ""

    output_path = out_dir / "final_payload.csv"
    write_csv(merged, output_path)
    logging.info("Saved payload to %s", output_path)


if __name__ == "__main__":
    main()
