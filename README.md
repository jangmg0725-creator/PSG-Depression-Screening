# PSG Summary-Based Depression Screening

Source code for the PSG-summary-based depression screening analysis using 23 predictors, candidate-model comparison, repeated nested cross-validation, Optuna tuning, final Shrinkage LDA fitting, internal/external evaluation, SHAP, and subgroup/error analyses.

## Environment

Python 3.10.16. Install required Python packages:

```bash
pip install -r requirements.txt
```

Package versions below are not locked; for exact replication, export the versions from the original analysis environment.

## Input files

Place authorized, de-identified datasets at:

- `data/ANAM.csv` (development cohort)
- `data/CNUH.csv` (external validation cohort)

The data files are not included. Verify that input column names, label encodings, and numerical ranges match those specified in the script. **Do not commit patient-level data or generated patient-level prediction CSVs to a public repository.**

## Run

From the repository directory:

```bash
python psg_depression_screening.py
```

Outputs are written to `outputs/psg_depression_screening/` (excluded from Git).

## Notes

- Only local file paths and public-facing filenames/labels were adapted from the original research script. The modeling and plotting logic was not changed.
- This copy was syntax-checked, but has **not** been run on the original datasets in this environment.
- Verify PSQI valid range (global score 0–21) and sex coding against data dictionaries before interpreting subgroup figures.
- For TRIPOD+AI model availability, publish a validated final fitted model and the preprocessing/feature schema separately after checking privacy, licensing and executable artifact safety.
