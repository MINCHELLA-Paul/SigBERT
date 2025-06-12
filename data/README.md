# Processed Data for SigBERT

- This directory contains the processed tabular datasets used as inputs for survival prediction models. Each file typically corresponds to a set of patient reports that have been cleaned, timestamped, and encoded with sentence-level embeddings.
- All identifiers must be fully anonymized and compliant with local data protection regulations.
- Sentence embeddings are typically obtained from clinical narratives and are designed to preserve semantic information in a compact numerical format.
 
## Required Columns

Each CSV file in this folder must contain the following columns:

| Column Name     | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `ID`             | Unique patient identifier. Must be anonymized.                             |
| `date_creation`  | Timestamp of the original medical report (format: `YYYY-MM-DD`).           |
| `DEATH`          | Binary target variable indicating death status (`0` = censored, `1` = death). |
| `date_death`     | Date of death when available (format: `YYYY-MM-DD`), `NaN` otherwise.      |
| `date_start`     | Earliest report date per patient. Used as the start of the follow-up period. |
| `date_end`       | Latest report date per patient. Used as the end of the follow-up period.     |
| `embeddings`     | Sentence embedding vector representing the medical report. Formally, a dense vector in $\mathbb{R}^p$. Stored as a stringified list (e.g., `"[0.01, 0.43, ...]"`). |

## Format Notes

- All date fields must be in ISO format (`YYYY-MM-DD`), and parseable as `datetime` objects in Python.
- The `embeddings` column stores sentence-level representations computed using a language model (e.g., OncoBERT) and optionally reduced in dimension (e.g., via PCA).
- `DEATH` must be strictly binary: `{0, 1}`.

## Usage

These files are typically produced by the notebook [`compute_sent_embd.ipynb`](../notebooks/compute_sent_embd.ipynb) 
and are consumed by downstream survival analysis code which expects these exact column names and formats.

## Example Rows

| ID     | date_creation | DEATH | date_death | date_start | date_end  | embeddings                           |
|--------|---------------|-------|------------|------------|-----------|--------------------------------------|
| 12345  | 2020-06-18    | 1     | 2021-09-30 | 2018-03-14 | 2021-06-20 | "[0.012, -0.345, ..., 0.098]"        |
| 12345  | 2020-10-22    | 1     | 2021-09-30 | 2018-03-14 | 2021-06-20 | "[0.812, -0.450, ..., 0.930]"        |
| 12345  | 2021-01-11    | 1     | 2021-09-30 | 2018-03-14 | 2021-06-20 | "[-0.188, -0.990, ..., 0.153]"        |
