"""Common I/O utilities for the PGx pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

DEFAULT_DTYPE = str
_EMPTY = ("", "NA", "NaN", "nan", None)

logger = logging.getLogger(__name__)


def read_csv(
    path: str | Path,
    *,
    dtype: type | dict[str, type] = DEFAULT_DTYPE,
    na_values: Optional[Iterable[str]] = None,
    keep_default_na: bool = False,
) -> pd.DataFrame:
    """Read a CSV ensuring string columns by default."""
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    extra_na = list(na_values or [])
    for sentinel in _EMPTY:
        if isinstance(sentinel, str):
            extra_na.append(sentinel)

    df = pd.read_csv(
        csv_path,
        dtype=dtype,
        keep_default_na=keep_default_na,
        na_filter=False,
    )
    for col in df.columns:
        df[col] = df[col].replace(_EMPTY, "").map(lambda v: (v or "").strip())

    logger.info("Loaded %s rows from %s", len(df), csv_path)
    return df


def write_csv(df: pd.DataFrame, path: str | Path, *, index: bool = False) -> Path:
    """Write a DataFrame to CSV with UTF-8 encoding."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=index)
    logger.info("Wrote %s rows to %s", len(df), out_path)
    return out_path

