"""PGx AI Flask web application."""

from __future__ import annotations

import logging
import math
import re
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
)

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.export_interactions import enrich_with_evidence
from src.common.pubmed import fetch_pubmed_metadata
from src.pipeline.from_vcf import run_vcf_pipeline

logging.basicConfig(level=logging.INFO, format="[webapp] %(message)s")
logger = logging.getLogger(__name__)

PMID_SPLIT_RE = re.compile(r"[;,|\s]+")
INVALID_PMID_VALUES = {"-", "na", "n/a", "null", "none", "0"}

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
SUMMARY_PATH = PROJECT_ROOT / "data" / "final_payload.summarized.csv"

UPLOAD_DIR = BASE_DIR / "uploads"
EXPORT_DIR = BASE_DIR / "exports"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)


def load_payload() -> pd.DataFrame:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Summarized payload not found at {SUMMARY_PATH}. Run the pipeline first."
        )
    df = pd.read_csv(SUMMARY_PATH)
    return enrich_with_evidence(df)


def normalize(text: str | None) -> str:
    return (text or "").strip()


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return str(value).strip()
    if pd.isna(value):
        return ""
    return str(value).strip()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and not value.strip():
            return default
        result = float(value)
        if math.isnan(result):
            return default
        return result
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    return int(round(_safe_float(value, default)))


def _collect_pmids(row: pd.Series) -> list[str]:
    pmids: list[str] = []
    primary = _as_text(row.get("best_pmid"))
    if primary and primary.lower() not in INVALID_PMID_VALUES:
        pmids.append(primary)
    raw_pmids = _as_text(row.get("pmids"))
    if raw_pmids:
        for token in PMID_SPLIT_RE.split(raw_pmids):
            token = token.strip()
            if (
                token
                and token.lower() not in INVALID_PMID_VALUES
                and token not in pmids
            ):
                pmids.append(token)
    return pmids


def _is_valid_pmid(pmid: str | None) -> bool:
    if not pmid:
        return False
    if pmid.lower() in INVALID_PMID_VALUES:
        return False
    return True


def _pmid_link(pmid: str) -> str | None:
    if not pmid:
        return None
    if pmid.isdigit():
        return f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    return None


def build_evidence_cards(row: pd.Series) -> tuple[list[dict[str, Any]], list[str]]:
    pmids = _collect_pmids(row)
    snippet = _as_text(row.get("Sentence")) or _as_text(row.get("Notes"))
    fallback_summary = _as_text(row.get("evidence_support"))
    title = _as_text(row.get("best_title"))
    year = _as_text(row.get("best_year"))
    strength = _as_text(row.get("evidence_strength"))
    significance = _as_text(row.get("Significance"))
    phenotype = _as_text(row.get("Phenotype Category")) or _as_text(row.get("Phenotype_Dominant"))
    journal = ""

    pmid_metadata = fetch_pubmed_metadata(pmids) if pmids else {}
    primary_pmid = pmids[0] if pmids else ""
    primary_meta = pmid_metadata.get(primary_pmid)
    if primary_meta:
        title = title or primary_meta.title
        year = year or primary_meta.pub_year or ""
        journal = primary_meta.journal or ""
        link = primary_meta.url or _pmid_link(primary_pmid)
    else:
        link = _pmid_link(primary_pmid) if primary_pmid else None

    citation_records: list[dict[str, Any]] = []
    for pmid in pmids:
        meta = pmid_metadata.get(pmid)
        citation_records.append(
            {
                "pmid": pmid,
                "title": (meta.title if meta else ""),
                "journal": (meta.journal if meta else ""),
                "year": (meta.pub_year if meta else ""),
                "url": (meta.url if meta else _pmid_link(pmid)),
            }
        )

    card = {
        "headline": phenotype or "Clinical Evidence",
        "snippet": snippet or fallback_summary or "Evidence sourced from curated pharmacogenomic knowledge bases.",
        "notes": _as_text(row.get("Notes")),
        "pmids": pmids,
        "primary_pmid": primary_pmid,
        "title": title,
        "year": year,
        "strength": strength,
        "significance": significance,
        "journal": journal,
        "link": link,
        "source_summary": fallback_summary,
        "citations": citation_records,
    }
    return [card], pmids


def build_record(row: pd.Series) -> dict[str, Any]:
    evidence_cards, pmids = build_evidence_cards(row)
    summary_text = _as_text(row.get("summary"))
    record = {
        "gene": _as_text(row.get("Gene")),
        "drug": _as_text(row.get("drug_name")),
        "summary": summary_text,
        "ensemble_probability": round(_safe_float(row.get("weighted_prob")) * 100, 2),
        "decisions": {
            "ensemble": _safe_int(row.get("weighted_pred")),
            "xgb": _safe_int(row.get("y_xgb")),
            "catboost": _safe_int(row.get("y_cat")),
            "targettox": _safe_int(row.get("y_targetox")),
        },
        "probabilities": {
            "xgb": round(_safe_float(row.get("p_xgb")) * 100, 2),
            "catboost": round(_safe_float(row.get("p_cat")) * 100, 2),
            "targettox": round(_safe_float(row.get("p_targetox")) * 100, 2),
        },
        "evidence": _as_text(row.get("evidence_support")),
        "evidence_details": evidence_cards,
        "phenotype_category": _as_text(row.get("Phenotype Category")),
        "phenotype_dominant": _as_text(row.get("Phenotype_Dominant")),
        "significance": _as_text(row.get("Significance")),
        "evidence_strength": _as_text(row.get("evidence_strength")),
        "sentence": _as_text(row.get("Sentence")),
        "notes": _as_text(row.get("Notes")),
        "pmids": pmids,
    }
    return record


def score_record(message_lower: str, record: dict[str, Any]) -> float:
    score = 0.0
    for term in (record.get("gene"), record.get("drug")):
        term_lower = (term or "").lower()
        if term_lower and term_lower in message_lower:
            score += 4.0
    for field in ("phenotype_category", "phenotype_dominant", "significance"):
        term_lower = (record.get(field) or "").lower()
        if term_lower and term_lower in message_lower:
            score += 1.5
    for pmid in record.get("pmids", []):
        if pmid and pmid.lower() in message_lower:
            score += 1.0
    if "toxic" in message_lower and record.get("decisions", {}).get("ensemble") == 1:
        score += 1.0
    if "safe" in message_lower and record.get("decisions", {}).get("ensemble") == 0:
        score += 1.0
    score += record.get("ensemble_probability", 0.0) / 200.0
    return score


def generate_chat_reply(message: str, entry: dict[str, Any]) -> tuple[str, list[dict[str, str]]]:
    records: list[dict[str, Any]] = entry.get("records", [])
    if not records:
        return (
            "I couldn't locate any analysis records for this session. Please re-run the DNA analysis and try again.",
            [],
        )

    message_lower = message.lower()
    if "download" in message_lower:
        return (
            "Use the “Download CSV” button to grab the full analysis. It exports the curated summaries and evidence for this run.",
            [],
        )

    ranked = sorted(records, key=lambda rec: score_record(message_lower, rec), reverse=True)
    top_records = ranked[:2] if ranked else []
    if not top_records:
        top_records = records[:1]

    lines: list[str] = []
    citations: list[dict[str, str]] = []

    for record in top_records:
        gene = record.get("gene") or "Unknown gene"
        drug = record.get("drug") or "Unknown drug"
        ensemble_flag = record.get("decisions", {}).get("ensemble", 0)
        risk_phrase = "likely toxic" if ensemble_flag == 1 else "not expected to be toxic"
        prob = record.get("ensemble_probability", 0.0)
        summary_text = record.get("summary") or ""
        lines.append(
            f"{gene} × {drug} is {risk_phrase} (ensemble probability {prob:.2f}%). {summary_text}"
        )

        evidence_cards = record.get("evidence_details", [])
        if evidence_cards:
            card = evidence_cards[0]
            snippet = card.get("snippet") or record.get("evidence") or ""
            if snippet:
                lines.append(f"Supporting evidence: {snippet}")

            citation_entries = card.get("citations") or []
            for entry in citation_entries[:3]:
                pmid = entry.get("pmid")
                if not _is_valid_pmid(pmid):
                    continue
                cite = {
                    "pmid": pmid,
                    "title": entry.get("title") or card.get("title") or "",
                }
                if entry.get("url"):
                    cite["url"] = entry["url"]
                elif _pmid_link(pmid):
                    cite["url"] = _pmid_link(pmid)
                if entry.get("journal"):
                    cite["journal"] = entry["journal"]
                if entry.get("year"):
                    cite["year"] = entry["year"]
                if cite not in citations:
                    citations.append(cite)
        elif record.get("evidence"):
            lines.append(f"Supporting evidence: {record['evidence']}")

    reply_text = "\n\n".join(lines)
    return reply_text, citations


analysis_store: dict[str, dict[str, Any]] = {}


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/analyze")
def analyze():
    dna_file = request.files.get("dnaFile")
    optional_drug = normalize(request.form.get("drug"))

    if dna_file is None or dna_file.filename == "":
        abort(400, description="DNA file is required.")

    upload_id = uuid4().hex
    upload_folder = UPLOAD_DIR / upload_id
    upload_folder.mkdir(parents=True, exist_ok=True)
    saved_path = upload_folder / dna_file.filename
    dna_file.save(saved_path)
    logger.info("Received DNA file %s", saved_path)

    filename_lower = dna_file.filename.lower()
    analysis_df: pd.DataFrame | None = None
    gene_display = ""
    used_drug = optional_drug
    detected_variants: list[str] = []

    try:
        if filename_lower.endswith((".vcf", ".vcf.gz")):
            result = run_vcf_pipeline(
                saved_path,
                optional_drug=optional_drug or None,
            )
            analysis_df = result.payload.copy()
            derived_genes = {
                normalize(value)
                for value in analysis_df.get("Gene", pd.Series(dtype=str)).dropna()
                if normalize(value)
            }
            if derived_genes:
                gene_display = ", ".join(sorted(derived_genes))
            elif result.gene_set:
                gene_display = ", ".join(sorted(result.gene_set))
            used_drug = optional_drug or result.used_drug or ""
            detected_variants = sorted(result.variant_set)
        else:
            payload_df = load_payload()
            if "Gene" not in payload_df.columns or "drug_name" not in payload_df.columns:
                abort(500, description="Payload is missing required columns.")

            payload_df["Gene"] = payload_df["Gene"].astype(str)
            payload_df["drug_name"] = payload_df["drug_name"].astype(str)

            analysis_df = payload_df.copy()
            if optional_drug:
                filtered = analysis_df[
                    analysis_df["drug_name"].str.lower() == optional_drug.lower()
                ]
                if not filtered.empty:
                    analysis_df = filtered
            gene_display = analysis_df.iloc[0]["Gene"] if not analysis_df.empty else ""
    except ValueError as exc:
        abort(400, description=str(exc))

    if analysis_df is None or analysis_df.empty:
        abort(400, description="No pharmacogenomic findings could be produced for this file.")

    analysis_id = uuid4().hex
    result_folder = EXPORT_DIR / analysis_id
    result_folder.mkdir(parents=True, exist_ok=True)

    export_path = result_folder / "analysis.csv"
    analysis_df.to_csv(export_path, index=False)

    records = [build_record(row) for _, row in analysis_df.iterrows()]

    analysis_store[analysis_id] = {
        "csv_path": str(export_path),
        "gene": gene_display,
        "drug": used_drug or "",
        "records": records,
        "variants": detected_variants,
    }

    return jsonify(
        {
            "analysis_id": analysis_id,
            "gene": gene_display,
            "drug": used_drug or "",
            "summaries": records,
            "started_at": int(time.time()),
            "variants": detected_variants,
        }
    )


@app.post("/chat")
def chat():
    payload = request.get_json(silent=True) or {}
    analysis_id = normalize(payload.get("analysis_id"))
    message = normalize(payload.get("message"))
    if not analysis_id or not message:
        abort(400, description="Both analysis_id and message are required.")

    entry = analysis_store.get(analysis_id)
    if not entry:
        abort(404, description="Analysis not found. Please rerun the workflow.")

    reply_text, citations = generate_chat_reply(message, entry)

    return jsonify(
        {
            "reply": reply_text,
            "citations": citations,
        }
    )


@app.get("/download/<analysis_id>")
def download(analysis_id: str):
    info = analysis_store.get(analysis_id)
    if not info:
        abort(404, description="Analysis not found.")
    csv_path = Path(info["csv_path"])
    if not csv_path.exists():
        abort(404, description="Export no longer available.")
    return send_file(
        csv_path,
        mimetype="text/csv",
        as_attachment=True,
        download_name=csv_path.name,
    )


if __name__ == "__main__":
    app.run(debug=True)
