"""Ensemble arbitration utilities."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd

from ..common.io_utils import read_csv, write_csv


def apply_ensemble(
    df: pd.DataFrame,
    *,
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> pd.DataFrame:
    """Apply weighted and majority ensemble strategies."""
    w_xgb, w_cat, w_ttox = weights
    probs = (
        df["p_xgb"].astype(float) * w_xgb
        + df["p_cat"].astype(float) * w_cat
        + df["p_targetox"].astype(float) * w_ttox
    )
    df = df.copy()
    df["weighted_prob"] = probs
    df["weighted_pred"] = (df["weighted_prob"] >= 0.5).astype(int)
    votes = (
        df[["y_xgb", "y_cat", "y_targetox"]]
        .astype(int)
        .sum(axis=1)
    )
    df["majority_pred"] = (votes >= 2).astype(int)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply ensemble voting and weighting.")
    parser.add_argument("--inp", required=True, help="Path to predictions scored CSV.")
    parser.add_argument("--out", required=True, help="Output CSV with ensemble columns.")
    parser.add_argument("--weights", nargs=3, type=float, default=(0.4, 0.3, 0.3))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[ensemble] %(message)s")
    df = read_csv(args.inp)
    logging.info("Applying ensemble with weights %s", args.weights)
    enriched = apply_ensemble(df, weights=tuple(args.weights))
    write_csv(enriched, args.out)
    logging.info("Wrote %d rows to %s", len(enriched), args.out)


if __name__ == "__main__":
    main()

