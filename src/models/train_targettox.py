"""Train the TargetTox PyTorch model."""

from __future__ import annotations

import argparse
import json
import logging
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..common.io_utils import read_csv
from ..common.text_utils import safe_str
from ..features.joiners import ensure_resolved, resolve_ml_entities
from .feature_pipeline import identify_label_column, prepare_features
from .metrics import compute_metrics
from .targetox_net import TargetToxNet

RANDOM_STATE = 42
EPOCHS = 100
PATIENCE = 10


class TabularDataset(Dataset):
    """Simple tensor dataset for tabular data."""

    def __init__(self, features: np.ndarray, labels: np.ndarray | None = None):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = None if labels is None else torch.as_tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.features.shape[0]

    def __getitem__(self, index: int):
        feats = self.features[index]
        if self.labels is None:
            return feats
        return feats, self.labels[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TargetTox PyTorch model.")
    parser.add_argument("--train", required=True, help="Path to ML feature CSV.")
    parser.add_argument("--gold", required=True, help="Path to golden evidence CSV.")
    parser.add_argument("--out", required=True, help="Directory to store the trained model.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def load_target(series) -> np.ndarray:
    labels = series.map(safe_str)
    if labels.eq("").any():
        raise ValueError("Found empty true_label entries; cannot train.")
    return labels.astype(float).to_numpy(dtype=np.float32)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[train_targettox] %(message)s")
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

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
    features_df, schema = prepare_features(
        ml_df,
        schema_path=schema_path,
        allow_create=True,
    )

    features = features_df.to_numpy(dtype=np.float32)

    X_train, X_val, y_train, y_val = train_test_split(
        features,
        y,
        test_size=args.test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    train_dataset = TabularDataset(X_train, y_train)
    val_dataset = TabularDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TargetToxNet(input_dim=features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    best_state = None
    epochs_without_improvement = 0

    logging.info(
        "Training TargetTox model on %d samples (val %d) with input dim %d",
        len(train_dataset),
        len(val_dataset),
        features.shape[1],
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            logits = model(batch_features)
            loss = criterion(logits, batch_labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * batch_features.size(0)

        avg_loss = running_loss / len(train_dataset)

        model.eval()
        all_probs: list[float] = []
        with torch.no_grad():
            for batch_features, _ in val_loader:
                batch_features = batch_features.to(device)
                logits = model(batch_features)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs.tolist())

        y_val_probs = np.array(all_probs, dtype=np.float32)
        metrics = compute_metrics(y_val, y_val_probs)
        logging.info(
            "Epoch %03d | loss=%.4f | AUC=%.4f ACC=%.4f LogLoss=%.4f",
            epoch,
            avg_loss,
            metrics.auc,
            metrics.accuracy,
            metrics.logloss,
        )

        if metrics.auc > best_auc:
            best_auc = metrics.auc
            best_state = deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                logging.info("Early stopping triggered after %d epochs.", epoch)
                break

    if best_state is None:
        best_state = deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    model_path = out_dir / "targetox.pt"
    torch.save({"state_dict": best_state, "input_dim": features.shape[1]}, model_path)

    schema_path_specific = out_dir / "targetox.schema.json"
    schema_path_specific.write_text(
        json.dumps(
            {
                "input_dim": features.shape[1],
                "feature_schema": schema.to_dict(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    logging.info("Saved model to %s", model_path)
    logging.info("Saved schema to %s", schema_path_specific)

    entity_lookup = {
        "genes": gene_lookup,
        "drugs": drug_lookup,
    }
    (out_dir / "entity_lookup.json").write_text(json.dumps(entity_lookup, indent=2), encoding="utf-8")
    logging.info("Saved entity lookup to %s", out_dir / "entity_lookup.json")


if __name__ == "__main__":
    main()
