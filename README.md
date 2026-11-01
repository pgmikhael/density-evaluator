# Mammographic Density Evaluator

Evaluates mammographic density as a predictor of breast cancer risk across multiple sites. `unified_analysis.py` loads every site into a common schema and produces a demographics table, cross-site summary stats, and density / age-alone / time-varying-Cox analyses with clustered bootstrap CIs. Multi-site groups are site-adjusted (stratified Cox, Mantel-Haenszel RR/OR, site term in the age-adjusted c-index, within-site bootstrap).

## Environment

```bash
conda env create -f environment.yml
conda activate density
```

## Usage

Site input paths are configured at the top of `unified_analysis.py` (expected under `data/`).

```bash
python unified_analysis.py                          # first-exam-only and all-exams
python unified_analysis.py --first-exam-only        # first-exam workbook only
python unified_analysis.py --include-mgh            # also run the gated MGH + MGH-AI pipeline
python unified_analysis.py --time-varying-cox-only  # recompute only the time_varying_cox sheet
```

Other options: `--n-bootstrap` (default 1000), `--n-jobs` (default 40), `--output-dir` (default `results_unified`).

## Output

`results_unified/unified_analysis_{first_exam_only|all_exams}.xlsx` with sheets `demographics`, `summary_stats`, `density_analysis`, `age_analysis`, and (all-exams only) `time_varying_cox`. With `--include-mgh` and the AI file present, a second workbook prefixed `ai_` is written.

## History

The earlier per-site scripts (`evaluate_density.py`, `adjusted_eval*.py`, `merge_and_run.py`, notebooks) are preserved on the `archive/pre-unified-cleanup` branch.
