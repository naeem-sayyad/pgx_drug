"""Verification script ensuring payload completeness."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.io_utils import read_csv
from src.common.text_utils import clean_str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify final payload CSV.")
    parser.add_argument("--inp", required=True, help="Path to final_payload.summarized.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[verify] %(message)s")
    df = read_csv(args.inp)

    failures: list[str] = []

    for idx, row in df.iterrows():
        gene = clean_str(row.get("Gene"))
        drug = clean_str(row.get("drug_name"))
        summary = clean_str(row.get("summary"))

        if not gene:
            failures.append(f"Row {idx}: Gene blank")
        if not drug:
            failures.append(f"Row {idx}: drug_name blank")
        if not summary:
            failures.append(f"Row {idx}: summary blank")
        else:
            lower_summary = summary.lower()
            if gene.lower() not in lower_summary:
                failures.append(f"Row {idx}: summary missing gene '{gene}'")
            if drug.lower() not in lower_summary:
                failures.append(f"Row {idx}: summary missing drug '{drug}'")
            if "gene (row" in lower_summary or "drug (row" in lower_summary:
                failures.append(f"Row {idx}: summary contains placeholder text")

    if failures:
        for failure in failures:
            logging.error(failure)
        logging.error("Verification FAILED: %d issues detected.", len(failures))
        sys.exit(1)

    logging.info("Verification PASS: %d rows validated.", len(df))


if __name__ == "__main__":
    main()
