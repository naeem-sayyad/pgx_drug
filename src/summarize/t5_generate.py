"""Generate structured summaries for each ensemble record."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

from ..common.io_utils import read_csv, write_csv
from ..common.text_utils import clean_str, pct_fmt

PMID_SPLIT_RE = re.compile(r"[;,|\s]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate deterministic summaries for final payload.")
    parser.add_argument("--inp", required=True, help="Path to payload with PMIDs CSV.")
    parser.add_argument("--out", required=True, help="Output directory.")
    # Retain legacy arguments for compatibility, even though they are unused.
    parser.add_argument("--model-name", default="template", help="Unused legacy parameter.")
    parser.add_argument("--batch-size", type=int, default=0, help="Unused legacy parameter.")
    parser.add_argument("--max-length", type=int, default=0, help="Unused legacy parameter.")
    return parser.parse_args()


def parse_float(value) -> float:
    try:
        return float(clean_str(value) or 0.0)
    except Exception:
        return 0.0


def parse_int_flag(value) -> int:
    try:
        return int(float(clean_str(value) or 0))
    except Exception:
        return 0


def normalize_gene(name: str) -> str:
    text = clean_str(name)
    if text and text == text.lower():
        return text.upper()
    return text


def normalize_drug(name: str) -> str:
    text = clean_str(name)
    if not text:
        return text
    if text == text.lower():
        return text.title()
    return text


def parse_pmids(raw_value: str) -> str:
    cleaned = clean_str(raw_value)
    if not cleaned:
        return "-"
    tokens: list[str] = []
    for token in PMID_SPLIT_RE.split(cleaned):
        token = token.strip()
        if not token:
            continue
        if token not in tokens:
            tokens.append(token)
    return ", ".join(tokens) if tokens else "-"


def decision_label(flag: int) -> str:
    return "toxic" if flag == 1 else "non-toxic"


def response_phrase(is_toxic: bool) -> str:
    if is_toxic:
        return "is likely to exhibit a toxic response"
    return "is not expected to exhibit a toxic response"


def build_summary(row) -> str:
    gene = normalize_gene(row.get("Gene", ""))
    drug = normalize_drug(row.get("drug_name", ""))

    pxgb = parse_float(row.get("p_xgb", 0.0))
    pcat = parse_float(row.get("p_cat", 0.0))
    pttox = parse_float(row.get("p_targetox", 0.0))
    pens = parse_float(row.get("weighted_prob", 0.0))

    y_xgb = decision_label(parse_int_flag(row.get("y_xgb", 0)))
    y_cat = decision_label(parse_int_flag(row.get("y_cat", 0)))
    y_ttox = decision_label(parse_int_flag(row.get("y_targetox", 0)))
    ensemble_flag = parse_int_flag(row.get("weighted_pred", 0))
    ensemble_decision = decision_label(ensemble_flag)

    pmid_text = parse_pmids(row.get("pmids", ""))

    sentences = [
        "The DNA analysis is complete.",
        f"Our ensemble model indicates that {gene} {response_phrase(ensemble_flag == 1)} when {drug} is administered, underscoring our pipeline's consolidated judgement.",
        f"Per-model probabilities — XGB {pct_fmt(pxgb)}, CatBoost {pct_fmt(pcat)}, TargetTox {pct_fmt(pttox)}; Ensemble {pct_fmt(pens)} (decision: {ensemble_decision}).",
        f"Per-model decisions — XGB {y_xgb}, CatBoost {y_cat}, TargetTox {y_ttox}; Ensemble {ensemble_decision}.",
        f"Evidence identifiers (PMID): {pmid_text}.",
    ]
    return " ".join(sentences)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[summary] %(message)s")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading payload with PMIDs...")
    df = read_csv(args.inp)
    if "summary" not in df.columns:
        df["summary"] = ""

    logging.info("Formatting summaries for %d rows...", len(df))
    for idx, row in df.iterrows():
        df.at[idx, "summary"] = build_summary(row)

    output_path = out_dir / "final_payload.summarized.csv"
    write_csv(df, output_path)
    logging.info("Saved summaries to %s", output_path)


if __name__ == "__main__":
    main()
