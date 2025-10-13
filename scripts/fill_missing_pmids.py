"""Fill missing PMIDs in the payload by querying PubMed."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.common.pubmed import fetch_pubmed_metadata

ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
TOOL = "pgx-toxicity-pipeline"
EMAIL = "support@example.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fill missing PMIDs using PubMed eSearch.")
    parser.add_argument("--inp", required=True, help="Path to final_payload.with_pmids.csv")
    parser.add_argument("--out", required=True, help="Output path for augmented CSV")
    parser.add_argument("--sleep", type=float, default=0.35, help="Delay between API calls (seconds)")
    parser.add_argument("--max", type=int, default=0, help="Optional limit on rows to update (0=all)")
    return parser.parse_args()


def search_pubmed(gene: str, drug: str) -> str:
    term = f"{gene} AND {drug} AND (pharmacogenomics OR pharmacogenetics)"
    params = {
        "db": "pubmed",
        "retmode": "json",
        "retmax": 1,
        "term": term,
        "tool": TOOL,
        "email": EMAIL,
    }
    try:
        response = requests.get(ESEARCH_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        idlist = data.get("esearchresult", {}).get("idlist", [])
        if idlist:
            return idlist[0]
    except Exception as exc:  # pragma: no cover - network
        logging.warning("PubMed search failed for %s / %s: %s", gene, drug, exc)
    return ""


def normalize_pmid(value: str | float) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    digits = "".join(ch for ch in text if ch.isdigit())
    return digits


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[pmid-fill] %(message)s")

    df = pd.read_csv(args.inp)
    updates = 0
    for idx, row in df.iterrows():
        current = normalize_pmid(row.get("best_pmid", ""))
        if current:
            continue
        gene = row.get("Gene", "")
        drug = row.get("drug_name", "")
        if not gene or not drug:
            continue

        pmid = search_pubmed(gene, drug)
        if not pmid:
            continue

        metadata = fetch_pubmed_metadata([pmid]).get(pmid)
        df.at[idx, "pmids"] = pmid
        df.at[idx, "best_pmid"] = pmid
        if metadata:
            df.at[idx, "best_title"] = metadata.title or df.at[idx, "best_title"]
            df.at[idx, "best_year"] = metadata.pub_year or df.at[idx, "best_year"]
        updates += 1
        if args.max and updates >= args.max:
            break
        time.sleep(args.sleep)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info("Filled %s PMIDs. Saved to %s", updates, output_path)


if __name__ == "__main__":
    main()
