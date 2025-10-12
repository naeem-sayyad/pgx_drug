"""Train the XGBoost model for PGx toxicity."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

from ..common.io_utils import read_csv
from ..common.text_utils import safe_str
from ..features.joiners import ensure_resolved, resolve_ml_entities
from .feature_pipeline import identify_label_column, prepare_features
from .metrics import compute_metrics

RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train XGBoost toxicity classifier.")
    parser.add_argument("--train", required=True, help="Path to ML feature CSV.")
    parser.add_argument("--gold", required=True, help="Path to golden evidence CSV.")
    parser.add_argument("--out", required=True, help="Directory to store the trained model.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split fraction.")
    return parser.parse_args()


def load_target(series) -> np.ndarray:
    labels = series.map(safe_str)
    if labels.eq("").any():
        raise ValueError("Found empty true_label entries; cannot train.")
    values = labels.astype(float)
    return values.to_numpy(dtype=np.float32)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[train_xgb] %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading datasets...")
    ml_raw = read_csv(args.train)
    golden_raw = read_csv(args.gold)
    golden_resolved = ensure_resolved(golden_raw, context="(gold)")
    ml_df, gene_lookup, drug_lookup = resolve_ml_entities(ml_raw, golden_resolved=golden_resolved)

    label_col = identify_label_column(ml_df)
    if label_col != "true_label":
        ml_df["true_label"] = ml_df[label_col]
    y = load_target(ml_df["true_label"])
    schema_path = out_dir / "feature_schema.json"

    features, schema = prepare_features(
        ml_df,
        schema_path=schema_path,
        allow_create=True,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        features.values,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    logging.info("Training XGBoost model...")
    model = xgb.XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        reg_lambda=1.0,
        reg_alpha=0.0,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    y_val_prob = model.predict_proba(X_val)[:, 1]
    metrics = compute_metrics(y_val, y_val_prob)
    logging.info("Validation AUC=%.4f ACC=%.4f LogLoss=%.4f", metrics.auc, metrics.accuracy, metrics.logloss)

    model_path = out_dir / "xgb.bin"
    model.save_model(model_path)
    logging.info("Saved model to %s", model_path)
    logging.info("Saved feature schema to %s", schema_path)

    entity_lookup = {
        "genes": gene_lookup,
        "drugs": drug_lookup,
    }
    (out_dir / "entity_lookup.json").write_text(json.dumps(entity_lookup, indent=2), encoding="utf-8")
    logging.info("Saved entity lookup to %s", out_dir / "entity_lookup.json")


if __name__ == "__main__":
    main()
