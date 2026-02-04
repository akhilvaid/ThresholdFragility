# Fragility of Fixed Decision Thresholds in Clinical AI Screening Pipelines

Code repository for **“Fragility of Fixed Decision Thresholds in Clinical AI Screening Pipelines”**.

This project demonstrates how **fixed decision thresholds** (e.g., a single cutoff used to convert model scores into “screen positive/negative”) can be **surprisingly brittle** even when the underlying model is unchanged.

---

## Repository structure

- `main.py` — entrypoint to run the analysis
- `threshold_fragility.py` — core threshold fragility routines
- `support_functions.py` — utilities (metrics, plotting, helpers)
- `config.py` — paths/parameters
- `TestData/` — example data used for demonstration

---

## Data note (cohort composition + de-identification)

The included example evaluation data is **de-identified** and is intentionally balanced across four evaluation cohorts:

- `testing` (**n = 100**)
- `testing_prospective` (**n = 100**)
- `external_testing` (**n = 100**)
- `external_prospective_testing` (**n = 100**)

Total: **400 samples**, with **100 samples apiece** from each cohort.

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
