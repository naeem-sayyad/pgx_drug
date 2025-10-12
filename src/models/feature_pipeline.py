"""Feature engineering helpers shared across models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

from ..common.text_utils import clean_str


@dataclass
class FeatureSchemaField:
    """Schema description for a single feature column."""

    name: str
    kind: str  # "numeric" or "categorical"
    mapping: list[str] | None = None  # Only for categorical fields

    def to_dict(self) -> dict:
        data = {"name": self.name, "kind": self.kind}
        if self.mapping is not None:
            data["mapping"] = self.mapping
        return data

    @staticmethod
    def from_dict(data: dict) -> "FeatureSchemaField":
        return FeatureSchemaField(
            name=data["name"],
            kind=data["kind"],
            mapping=list(data.get("mapping") or []),
        )


@dataclass
class FeatureSchema:
    """Schema container describing the feature matrix."""

    fields: list[FeatureSchemaField]

    def to_dict(self) -> dict:
        return {"fields": [field.to_dict() for field in self.fields]}

    @staticmethod
    def from_dict(data: dict) -> "FeatureSchema":
        fields = [FeatureSchemaField.from_dict(field) for field in data["fields"]]
        return FeatureSchema(fields=fields)

    @property
    def feature_names(self) -> list[str]:
        return [field.name for field in self.fields]


EXTRA_EXCLUDE = {"true_label", "label", "target", "Gene", "drug_name"}
LABEL_CANDIDATES = ["true_label", "y_toxicity", "label", "target", "y_primary"]


def _detect_field(name: str, series: pd.Series) -> FeatureSchemaField:
    try:
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().all():
            return FeatureSchemaField(name=name, kind="numeric")
    except Exception:
        pass

    cleaned = [clean_str(value) for value in series.tolist()]
    mapping = [value for value in pd.unique(cleaned) if value]
    return FeatureSchemaField(name=name, kind="categorical", mapping=mapping)


def build_feature_schema(
    df: pd.DataFrame,
    *,
    exclude: Iterable[str],
) -> FeatureSchema:
    fields: list[FeatureSchemaField] = []
    seen: set[str] = set()

    for name in df.columns:
        if name in seen:
            continue
        seen.add(name)
        if name in exclude:
            continue
        field = _detect_field(name, df[name])
        fields.append(field)
    return FeatureSchema(fields=fields)


def encode_with_schema(df: pd.DataFrame, schema: FeatureSchema) -> pd.DataFrame:
    data: dict[str, np.ndarray] = {}

    for field in schema.fields:
        if field.name in df.columns:
            series = df[field.name]
        else:
            series = pd.Series([""] * len(df))

        if field.kind == "numeric":
            numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
            data[field.name] = numeric.astype("float32").to_numpy()
        else:
            cleaned = series.apply(clean_str)
            mapping = {value: idx + 1 for idx, value in enumerate(field.mapping or [])}
            encoded = cleaned.map(lambda value: mapping.get(value, 0)).astype("float32")
            data[field.name] = encoded.to_numpy()

    feature_frame = pd.DataFrame(data, index=df.index)
    return feature_frame


def prepare_features(
    df: pd.DataFrame,
    *,
    schema_path: Path,
    allow_create: bool,
    exclude: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, FeatureSchema]:
    exclude_cols = set(exclude or [])
    exclude_cols.update(EXTRA_EXCLUDE)
    exclude_cols.update({col for col in df.columns if col.lower().startswith("y_")})

    if schema_path.exists():
        with schema_path.open("r", encoding="utf-8") as handle:
            schema = FeatureSchema.from_dict(json.load(handle))
    else:
        if not allow_create:
            raise FileNotFoundError(f"Feature schema missing at {schema_path}")
        schema = build_feature_schema(df, exclude=exclude_cols)
        schema_path.parent.mkdir(parents=True, exist_ok=True)
        with schema_path.open("w", encoding="utf-8") as handle:
            json.dump(schema.to_dict(), handle, indent=2)

    encoded = encode_with_schema(df, schema)
    return encoded, schema


def identify_label_column(df: pd.DataFrame) -> str:
    """Determine which column should be used as the binary label."""
    for candidate in LABEL_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise KeyError("No label column found in dataframe.")
