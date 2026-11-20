# Structural breaks in Turkish macro-financial time series

Code, data and results for the paper

> Weisser, C., Lukassen, F., Schlee, M., Erduran, A., Altun, K., & Bäumer, F. (2026).
> Structural breaks in Turkish macro-financial time series: Assessing ensemble detection
> and LLM-based event attribution. 7th Annual Business Administration Conference
> (ABAC 2026), Istanbul.

The paper applies an ensemble of ten structural break detectors to four monthly Turkish
series (CPI inflation, the overnight interbank rate, consumer confidence and the MSCI
Turkey ETF) and to the daily USD/TRY exchange rate. The detected breaks are compared with
a chronology of documented episodes, and a large language model is asked to name the
likely cause of each break. These attributions are rated against the record, for the main
model and for six further models.

## Contents

| Path | Content |
|---|---|
| `package/` | `structural_break_analyzer`, the detection and attribution package, in the version used for the paper |
| `code/01_data.py` | downloads and cleans the series |
| `code/02_analysis.ipynb` | detection, matching to the chronology, attributions, tables and figures |
| `code/data/raw/` | the series as downloaded on 27 May 2026 |
| `code/data/processed/` | the cleaned series read by the notebook |
| `code/output/tables/` | result tables (CSV) |
| `code/output/figures/` | Figures 1 to 3 (PDF and PNG) |
| `code/output/attributions/` | the LLM attributions, one text file per break and model |
| `code/output/cache/` | stored detector output |
| `requirements.txt` | package versions of the environment the results were produced in |

## Setup

The results were produced with Python 3.11 on Linux.

    python3.11 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install -e package

## Reproducing the results

    cd code
    python 01_data.py
    jupyter nbconvert --to notebook --execute --inplace 02_analysis.ipynb

`01_data.py` cleans the raw files in `data/raw/` and writes `data/processed/`. With
`--download` it first downloads the series again and cuts them at `--cutoff` (default
2026-05-31). Both sources revise and extend their data, so a new download need not give
the files stored here.

The notebook writes everything in `output/`. It uses the stored detector output in
`output/cache/` and the stored attributions in `output/attributions/`, so it needs no API
key. Deleting `output/cache/` makes it run the detectors again, which takes less than a
minute and gives the same tables.

To generate attributions again, delete the text files in question and set
`OPENROUTER_API_KEY` before starting the notebook. `NO_LLM=1` switches all API calls off.
The models are not deterministic at the temperature used (0.3), so new texts will differ
from the stored ones, and the ratings in the notebook refer to the stored texts.

## Where the results of the paper are

| In the paper | File |
|---|---|
| Table 2 (monthly series) | `code/data/processed/_manifest.csv` |
| Table 3 | `code/output/tables/table3_breaks_attributions.csv` |
| Table 4 | `code/output/tables/episode_matrix.csv` |
| Table 5 | `code/output/tables/table5_ratings_by_model.csv` and `table5_summary.csv` |
| Figure 1 | `code/output/figures/fig1_series_breaks` |
| Figure 2 | `code/output/figures/fig2_votes` |
| Figure 3 | `code/output/figures/fig3_usdtry_daily` |

`breaks_main.csv` and `breaks_daily.csv` in `code/output/tables/` list all consensus
breaks with their statistics.

The attributions of each model are in `code/output/attributions/models/<model>/`, named
`<series>_<break date>.md`. The files directly in `code/output/attributions/` are the
Claude Sonnet 4 texts again. The main model of the paper is Claude Opus 5. All models
were called through OpenRouter. Qwen3.5 9B returned no text for two breaks.

## Data

| Series | Source |
|---|---|
| iShares MSCI Turkey ETF (TUR), monthly close in USD | Yahoo Finance |
| USD/TRY, daily close | Yahoo Finance |
| CPI inflation, year-on-year | OECD |
| Overnight interbank rate | OECD |
| Consumer confidence index | OECD |

All series were downloaded on 27 May 2026.

## The package

`package/` contains the source of `structural_break_analyzer` as it was run for the
paper. It is under the MIT licence (`package/LICENSE`). The framework is described in

> Lukassen, F., Weisser, C., Schlee, M., Kumar, M., Thielmann, A., Saefken, B.,
> Silbersdorff, A., & Kneib, T. (2026). LLM-augmented changepoint detection: A framework
> for ensemble detection and automated explanation. arXiv.
> https://doi.org/10.48550/arXiv.2601.02957
