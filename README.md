# PGx Toxicity Pipeline

Minimal, production-ready pharmacogenomic toxicity workflow.

## Setup
- Python 3.10+
- `pip install -r requirements.txt`

## Core Stages
1. Train models:
   ```bash
   python -m src.models.train_xgb --train data/pgx_ML_final.csv --gold data/pgx_clean_golden_finalultimate_fe.csv --out models/
   python -m src.models.train_cat --train data/pgx_ML_final.csv --gold data/pgx_clean_golden_finalultimate_fe.csv --out models/
   python -m src.models.train_targettox --train data/pgx_ML_final.csv --gold data/pgx_clean_golden_finalultimate_fe.csv --out models/
   ```
2. Score and ensemble:
   ```bash
   python -m src.models.predict_all --ml data/pgx_ML_final.csv --out outputs/
   python -m src.arbiter.finalize --ml outputs/predictions.scored.csv --gold data/pgx_clean_golden_finalultimate_fe.csv --out outputs/
   python -m src.pmid.retrieve --inp outputs/final_payload.csv --gold data/pgx_clean_golden_finalultimate_fe.csv --out outputs/
   python -m src.summarize.t5_generate --inp outputs/final_payload.with_pmids.csv --out outputs/
   python scripts/verify_final.py --inp outputs/final_payload.summarized.csv
   ```

## Quick Sample
Run a small end-to-end pass on the first 20 ML rows (writes artifacts into `outputs/quick_sample`):
```bash
python scripts/quick_sample.py
```
