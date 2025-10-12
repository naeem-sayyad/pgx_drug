"""Utilities to derive PGx predictions directly from a VCF file."""

from __future__ import annotations

import gzip
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Set

import numpy as np
import pandas as pd

from ..arbiter.ensemble import apply_ensemble
from ..common.io_utils import read_csv
from ..common.text_utils import clean_str
from ..features.joiners import ensure_resolved, merge_on_entities, resolve_ml_entities
from ..models.feature_pipeline import prepare_features
from ..models.predict_all import load_cat, load_targetox, load_xgb
from ..summarize.t5_generate import build_summary
from scripts.export_interactions import enrich_with_evidence

logger = logging.getLogger(__name__)

ML_DEFAULT_PATH = Path("data/pgx_ML_final.csv")
GOLD_DEFAULT_PATH = Path("data/pgx_clean_golden_finalultimate_fe.csv")
MODELS_DEFAULT_DIR = Path("models/quick_sample")

GENE_TOKEN_PATTERNS = (
    re.compile(r"gene(?:name)?=([^;]+)", re.IGNORECASE),
    re.compile(r"genesymbol=([^;]+)", re.IGNORECASE),
    re.compile(r"geneinfo=([^;]+)", re.IGNORECASE),
    re.compile(r"hgnc(?:_id)?=([^;]+)", re.IGNORECASE),
)

ANN_SPLIT_RE = re.compile(r"[|,]")
ID_SPLIT_RE = re.compile(r"[;,]")
STAR_ALLELE_RE = re.compile(r"^[A-Z0-9]+[*][0-9A-Z.+-]+$")
RSID_RE = re.compile(r"^RS?\d+$", re.IGNORECASE)


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return path.open("r", encoding="utf-8", errors="ignore")


def _normalise_gene(value: str) -> str:
    cleaned = clean_str(value)
    return cleaned.upper()


def _candidates_from_token(token: str) -> Iterable[str]:
    for part in ANN_SPLIT_RE.split(token):
        gene = _normalise_gene(part)
        if gene and gene not in {"", ".", "NA", "NONE"}:
            yield gene


@dataclass
class VCFSignals:
    genes: Set[str]
    variants: Set[str]
    alleles: Set[str]


def _register_variant(target: Set[str], token: str) -> None:
    cleaned = clean_str(token)
    if not cleaned or cleaned in {".", "-", "NA"}:
        return
    target.add(cleaned.upper())


def _register_allele(target: Set[str], token: str) -> None:
    cleaned = clean_str(token)
    if not cleaned or cleaned in {".", "-", "NA"}:
        return
    upper = cleaned.upper()
    if STAR_ALLELE_RE.match(upper):
        target.add(upper)


def extract_signals_from_vcf(path: Path) -> VCFSignals:
    genes: Set[str] = set()
    variants: Set[str] = set()
    alleles: Set[str] = set()
    if not path.exists():
        raise FileNotFoundError(f"VCF file not found: {path}")

    with _open_text(path) as handle:
        for raw_line in handle:
            if not raw_line or raw_line.startswith("##"):
                continue
            if raw_line.startswith("#"):
                continue
            fields = raw_line.strip().split("\t")
            if len(fields) < 8:
                fields = raw_line.strip().split()
            if len(fields) < 8:
                fields = raw_line.strip().split()
            if len(fields) < 8:
                continue
            chrom, pos, identifier = fields[0], fields[1], fields[2]
            ref = fields[3] if len(fields) > 3 else ""
            alt = fields[4] if len(fields) > 4 else ""
            info_field = fields[7]
            upper_info = info_field.upper()

            chrom_clean = clean_str(chrom)
            chrom_upper = chrom_clean.upper()
            chrom_no_prefix = chrom_upper[3:] if chrom_upper.startswith("CHR") else chrom_upper
            pos_clean = clean_str(pos)

            if identifier and identifier != ".":
                for token in ID_SPLIT_RE.split(identifier):
                    token = clean_str(token)
                    if not token:
                        continue
                    upper_token = token.upper()
                    if STAR_ALLELE_RE.match(upper_token):
                        alleles.add(upper_token)
                    elif RSID_RE.match(upper_token) or ":" in upper_token:
                        variants.add(upper_token)
                    elif upper_token and upper_token.startswith("CHR"):
                        variants.add(upper_token.replace("CHR", "", 1))

            if chrom_clean and pos_clean:
                coordinate_tokens = {
                    f"{chrom_upper}:{pos_clean}",
                    f"{chrom_no_prefix}:{pos_clean}",
                }
                for token in coordinate_tokens:
                    _register_variant(variants, token)

            ref_clean = clean_str(ref).upper()
            alt_tokens = [clean_str(token).upper() for token in alt.split(",")] if alt else []
            for alt_token in alt_tokens:
                if not alt_token:
                    continue
                combo_tokens = [
                    f"{chrom_no_prefix}:{pos_clean}:{ref_clean}>{alt_token}",
                    f"{chrom_upper}:{pos_clean}:{ref_clean}>{alt_token}",
                    f"{ref_clean}>{alt_token}",
                ]
                for combo in combo_tokens:
                    _register_variant(variants, combo)
                _register_allele(alleles, alt_token)

            for pattern in GENE_TOKEN_PATTERNS:
                match = pattern.search(info_field)
                if match:
                    for candidate in _candidates_from_token(match.group(1)):
                        genes.add(candidate)

            if "ANN=" in upper_info or "CSQ=" in upper_info:
                key, value = (
                    info_field.split("ANN=", 1)
                    if "ANN=" in upper_info
                    else info_field.split("CSQ=", 1)
                )
                annotations = value.split(";")[0]
                for item in annotations.split(","):
                    parts = item.split("|")
                    if len(parts) >= 4:
                        genes.add(_normalise_gene(parts[3]))
                    if len(parts) >= 5:
                        genes.add(_normalise_gene(parts[4]))

            for entry in info_field.split(";"):
                if "=" not in entry:
                    continue
                key, value = entry.split("=", 1)
                key_upper = key.upper()
                if key_upper in {
                    "GENE",
                    "GENES",
                    "GENE_NAME",
                    "GENE_SYMBOL",
                    "GENENAME",
                    "GENESYMBOL",
                    "SYMBOL",
                    "HGNC",
                    "HGNC_ID",
                    "GENEINFO",
                }:
                    for candidate in _candidates_from_token(value):
                        genes.add(candidate)
                elif key_upper in {
                    "RS",
                    "RSID",
                    "RSIDS",
                    "RS_ID",
                    "RS_IDs",
                    "DBSNP",
                    "DBSNP_RS",
                    "DBSNPID",
                    "DBSNP_ID",
                    "DBSNP_RSIDS",
                    "SNP",
                }:
                    for token in ID_SPLIT_RE.split(value):
                        _register_variant(variants, token)
                elif key_upper in {"ALLELE", "ALLELES"}:
                    for token in ID_SPLIT_RE.split(value):
                        _register_allele(alleles, token)
                elif key_upper in {"HAPLOTYPES", "HAPLOTYPE"}:
                    for token in ID_SPLIT_RE.split(value):
                        _register_allele(alleles, token)

            if "GENE=" not in upper_info and "ANN=" not in upper_info and "CSQ=" not in upper_info:
                # Try RefGene style annotations (Gene.refGene=)
                ref_match = re.search(r"GENE\.REFGENE=([^;]+)", upper_info)
                if ref_match:
                    for candidate in _candidates_from_token(ref_match.group(1)):
                        genes.add(candidate)

            if len(fields) > 9:
                format_field = fields[8]
                sample_fields = fields[9:]
                sample_tokens = []
                for sample in sample_fields:
                    sample_tokens.extend(sample.replace("/", ":").split(":"))
                for token in sample_tokens:
                    upper = clean_str(token).upper()
                    if not upper:
                        continue
                    if STAR_ALLELE_RE.match(upper):
                        alleles.add(upper)
                    elif RSID_RE.match(upper):
                        variants.add(upper)

    genes = {gene for gene in genes if gene}
    variants = {variant for variant in variants if variant}
    alleles = {allele for allele in alleles if allele}
    logger.info(
        "Extracted %d genes, %d variants, %d alleles from %s",
        len(genes),
        len(variants),
        len(alleles),
        path.name,
    )
    return VCFSignals(genes=genes, variants=variants, alleles=alleles)


@dataclass
class VCFAnalysisResult:
    payload: pd.DataFrame
    gene_set: Set[str]
    variant_set: Set[str]
    used_drug: Optional[str]


def _filter_ml_dataset(
    ml_df: pd.DataFrame,
    genes: Set[str],
    *,
    drug: Optional[str] = None,
) -> pd.DataFrame:
    genes_upper = {gene.upper() for gene in genes}
    filtered = ml_df[ml_df["Gene"].str.upper().isin(genes_upper)].copy()
    if drug:
        drug_upper = drug.upper()
        filtered = filtered[filtered["drug_name"].str.upper() == drug_upper]
    return filtered


def _ensure_not_empty(df: pd.DataFrame, message: str) -> pd.DataFrame:
    if df.empty:
        raise ValueError(message)
    return df


def run_vcf_pipeline(
    vcf_path: Path,
    *,
    optional_drug: Optional[str] = None,
    ml_path: Path = ML_DEFAULT_PATH,
    gold_path: Path = GOLD_DEFAULT_PATH,
    models_dir: Path = MODELS_DEFAULT_DIR,
) -> VCFAnalysisResult:
    signals = extract_signals_from_vcf(vcf_path)
    genes = {gene.upper() for gene in signals.genes}

    ml_raw = read_csv(ml_path)
    gold_raw = read_csv(gold_path)
    gold_resolved = ensure_resolved(gold_raw, context="(gold)")

    variants_upper = {
        clean_str(token).upper() for token in signals.variants if clean_str(token)
    }
    alleles_upper = {
        clean_str(token).upper() for token in signals.alleles if clean_str(token)
    }

    if not genes:
        if variants_upper:
            variant_matches = gold_resolved[
                gold_resolved["Variant/Haplotypes"]
                .astype(str)
                .str.upper()
                .isin(variants_upper)
            ]
            genes.update(variant_matches["Gene"].astype(str).str.upper())
        if alleles_upper:
            allele_matches = gold_resolved[
                gold_resolved["Alleles"].astype(str).str.upper().isin(alleles_upper)
            ]
            genes.update(allele_matches["Gene"].astype(str).str.upper())

    if not genes:
        raise ValueError("No gene annotations could be extracted from the provided VCF.")

    ml_resolved, _, _ = resolve_ml_entities(ml_raw, golden_resolved=gold_resolved)
    ml_filtered = _filter_ml_dataset(
        ml_resolved,
        genes,
        drug=optional_drug,
    )
    _ensure_not_empty(
        ml_filtered,
        "No pharmacogenomic interactions matched the provided gene selections.",
    )

    schema_path = models_dir / "feature_schema.json"
    feature_frame, _ = prepare_features(
        ml_filtered,
        schema_path=schema_path,
        allow_create=False,
    )

    xgb_path = models_dir / "xgb.bin"
    cat_path = models_dir / "cat.cbm"
    targetox_path = models_dir / "targetox.pt"
    targetox_schema = models_dir / "targetox.schema.json"

    p_xgb = load_xgb(xgb_path, feature_frame)
    p_cat = load_cat(cat_path, feature_frame)
    p_targetox = load_targetox(targetox_path, targetox_schema, feature_frame)

    predictions = ml_filtered.copy()
    predictions["p_xgb"] = p_xgb
    predictions["p_cat"] = p_cat
    predictions["p_targetox"] = p_targetox
    predictions["y_xgb"] = (p_xgb >= 0.5).astype(int)
    predictions["y_cat"] = (p_cat >= 0.5).astype(int)
    predictions["y_targetox"] = (p_targetox >= 0.5).astype(int)

    predictions = apply_ensemble(predictions)

    merged = merge_on_entities(predictions, gold_resolved, how="left", suffixes=("", "_gold"))
    merged["_order"] = np.arange(len(merged))
    if merged["row_id"].duplicated().any():
        merged = (
            merged.sort_values(["row_id", "weighted_prob"], ascending=[True, False])
            .drop_duplicates(subset=["row_id"], keep="first")
            .sort_values("_order")
        )
    merged = merged.drop(columns="_order", errors="ignore")

    merged = enrich_with_evidence(merged)
    mask = pd.Series(False, index=merged.index)
    if variants_upper and "Variant/Haplotypes" in merged.columns:
        mask = mask | merged["Variant/Haplotypes"].astype(str).str.upper().isin(variants_upper)
    if alleles_upper and "Alleles" in merged.columns:
        mask = mask | merged["Alleles"].astype(str).str.upper().isin(alleles_upper)
    if (variants_upper or alleles_upper) and mask.any():
        merged = merged[mask]

    if merged.empty:
        raise ValueError("No pharmacogenomic findings matched the detected variants or genes.")

    merged["summary"] = merged.apply(build_summary, axis=1)
    display_genes = {
        clean_str(value) for value in merged["Gene"].astype(str) if clean_str(value)
    }
    return VCFAnalysisResult(
        payload=merged.reset_index(drop=True),
        gene_set=display_genes,
        variant_set=signals.variants,
        used_drug=optional_drug,
    )
