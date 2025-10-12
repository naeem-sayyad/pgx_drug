"""Run predictions across all trained models."""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch
import xgboost as xgb
from catboost import CatBoostClassifier

from ..common.io_utils import read_csv, write_csv
from ..features.joiners import resolve_ml_entities
from .feature_pipeline import prepare_features
from .targetox_net import TargetToxNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict with XGB, CatBoost, and TargetTox.")
    parser.add_argument("--ml", required=True, help="Path to ML CSV for scoring.")
    parser.add_argument("--out", required=True, help="Output directory for prediction CSV.")
    parser.add_argument("--models-dir", default="models", help="Directory containing trained models.")
    return parser.parse_args()


def load_xgb(model_path: Path, features) -> np.ndarray:
    booster = xgb.Booster()
    booster.load_model(str(model_path))
    dmatrix = xgb.DMatrix(features.values, feature_names=list(features.columns))
    probs = booster.predict(dmatrix)
    return probs


def load_cat(model_path: Path, features) -> np.ndarray:
    model = CatBoostClassifier()
    model.load_model(str(model_path))
    probs = model.predict_proba(features.values)[:, 1]
    return probs


def load_targetox(model_path: Path, schema_path: Path, features) -> np.ndarray:
    payload = torch.load(model_path, map_location="cpu")
    with schema_path.open("r", encoding="utf-8") as handle:
        schema_payload = json.load(handle)

    input_dim = payload.get("input_dim") or schema_payload["input_dim"]
    model = TargetToxNet(input_dim=input_dim)
    model.load_state_dict(payload["state_dict"])
    model.eval()

    feature_tensor = torch.as_tensor(features.values, dtype=torch.float32)
    with torch.no_grad():
        logits = model(feature_tensor)
        probs = torch.sigmoid(logits).numpy()
    return probs


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[predict_all] %(message)s")

    models_dir = Path(args.models_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading ML dataset for scoring...")
    ml_raw = read_csv(args.ml)
    lookup_path = models_dir / "entity_lookup.json"
    if not lookup_path.exists():
        raise FileNotFoundError(f"Missing entity lookup file at {lookup_path}")
    lookup_data = json.loads(lookup_path.read_text(encoding="utf-8"))
    gene_lookup = lookup_data["genes"]
    drug_lookup = lookup_data["drugs"]
    ml_df, _, _ = resolve_ml_entities(
        ml_raw,
        gene_lookup=gene_lookup,
        drug_lookup=drug_lookup,
    )

    schema_path = models_dir / "feature_schema.json"
    features_df, _ = prepare_features(
        ml_df,
        schema_path=schema_path,
        allow_create=False,
    )

    logging.info("Generating model predictions...")
    xgb_path = models_dir / "xgb.bin"
    cat_path = models_dir / "cat.cbm"
    targetox_path = models_dir / "targetox.pt"
    targetox_schema_path = models_dir / "targetox.schema.json"

    if not xgb_path.exists() or not cat_path.exists() or not targetox_path.exists():
        missing = [path.name for path in (xgb_path, cat_path, targetox_path) if not path.exists()]
        raise FileNotFoundError(f"Missing trained models: {', '.join(missing)}")

    p_xgb = load_xgb(xgb_path, features_df)
    p_cat = load_cat(cat_path, features_df)
    p_targetox = load_targetox(targetox_path, targetox_schema_path, features_df)

    scored = ml_df.copy()
    scored["p_xgb"] = p_xgb
    scored["p_cat"] = p_cat
    scored["p_targetox"] = p_targetox
    scored["y_xgb"] = (p_xgb >= 0.5).astype(int)
    scored["y_cat"] = (p_cat >= 0.5).astype(int)
    scored["y_targetox"] = (p_targetox >= 0.5).astype(int)

    output_path = out_dir / "predictions.scored.csv"
    write_csv(scored, output_path)
    logging.info("Saved scored predictions to %s", output_path)


if __name__ == "__main__":
    main()
