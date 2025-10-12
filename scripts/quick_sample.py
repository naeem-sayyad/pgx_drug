"""Quick sample runner for the PGx pipeline."""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io_utils import read_csv, write_csv
from src.features.joiners import ensure_resolved, resolve_ml_entities
from src.models.feature_pipeline import identify_label_column


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a quick sample end-to-end pipeline.")
    parser.add_argument("--gold", default="data/pgx_clean_golden_finalultimate_fe.csv")
    parser.add_argument("--ml", default="data/pgx_ML_final.csv")
    parser.add_argument("--out", default="outputs/quick_sample")
    parser.add_argument("--models", default="models/quick_sample")
    parser.add_argument("--n", type=int, default=20, help="Number of rows for the sample ML dataset.")
    return parser.parse_args()


def run_step(args: list[str]) -> None:
    logging.info("Running: %s", " ".join(args))
    subprocess.check_call(args)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[quick_sample] %(message)s")

    ml_raw = read_csv(args.ml)
    golden_raw = read_csv(args.gold)
    golden_df = ensure_resolved(golden_raw, context="(gold sample)")
    ml_df, _, _ = resolve_ml_entities(ml_raw, golden_resolved=golden_df)

    label_col = identify_label_column(ml_df)
    if label_col != "true_label":
        ml_df["true_label"] = ml_df[label_col]

    sample_ml = ml_df.head(args.n).copy()
    if sample_ml["true_label"].nunique() < 2:
        logging.warning(
            "Sample of %d rows lacks both classes; falling back to full ML dataset.",
            args.n,
        )
        sample_ml = ml_df

    row_ids = set(sample_ml["row_id"])
    sample_gold = golden_df[golden_df["row_id"].isin(row_ids)].copy()

    out_dir = Path(args.out)
    models_dir = Path(args.models)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    sample_ml_path = out_dir / "sample_ml.csv"
    sample_gold_path = out_dir / "sample_gold.csv"
    write_csv(sample_ml, sample_ml_path)
    write_csv(sample_gold, sample_gold_path)

    run_step(
        [
            sys.executable,
            "-m",
            "src.models.train_xgb",
            "--train",
            str(sample_ml_path),
            "--gold",
            str(sample_gold_path),
            "--out",
            str(models_dir),
        ]
    )
    run_step(
        [
            sys.executable,
            "-m",
            "src.models.train_cat",
            "--train",
            str(sample_ml_path),
            "--gold",
            str(sample_gold_path),
            "--out",
            str(models_dir),
        ]
    )
    run_step(
        [
            sys.executable,
            "-m",
            "src.models.train_targettox",
            "--train",
            str(sample_ml_path),
            "--gold",
            str(sample_gold_path),
            "--out",
            str(models_dir),
        ]
    )
    run_step(
        [
            sys.executable,
            "-m",
            "src.models.predict_all",
            "--ml",
            str(sample_ml_path),
            "--out",
            str(out_dir),
            "--models-dir",
            str(models_dir),
        ]
    )
    run_step(
        [
            sys.executable,
            "-m",
            "src.arbiter.finalize",
            "--ml",
            str(out_dir / "predictions.scored.csv"),
            "--gold",
            str(sample_gold_path),
            "--out",
            str(out_dir),
        ]
    )
    run_step(
        [
            sys.executable,
            "-m",
            "src.pmid.retrieve",
            "--inp",
            str(out_dir / "final_payload.csv"),
            "--gold",
            str(sample_gold_path),
            "--out",
            str(out_dir),
        ]
    )
    run_step(
        [
            sys.executable,
            "-m",
            "src.summarize.t5_generate",
            "--inp",
            str(out_dir / "final_payload.with_pmids.csv"),
            "--out",
            str(out_dir),
        ]
    )
    run_step(
        [
            sys.executable,
            "scripts/verify_final.py",
            "--inp",
            str(out_dir / "final_payload.summarized.csv"),
        ]
    )

    logging.info("Quick sample pipeline completed successfully.")


if __name__ == "__main__":
    main()
