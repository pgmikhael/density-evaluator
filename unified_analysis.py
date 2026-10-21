"""
Unified density analysis script.

Loads every site into a common schema, produces a demographics table and
cross-site summary stats, and runs the density / age-alone / time-varying-Cox
analyses with clustered bootstrap CIs. Gates MGH behind a flag.

Multi-site cohorts (all-sites and its age/race subgroups) control for site
identity: Cox models are stratified by Site, RR/OR are Mantel-Haenszel pooled,
the age-adjusted c-index adds C(Site), and the clustered bootstrap resamples
patients within each site. Only sites with >=1 event and both exposure levels
count; if a subgroup has <2 such sites it falls back to a plain single-site
estimate. The n_sites / site_adjusted columns in the output record this.


Usage:
source /data/rbg/users/pgmikhael/environments/miniforge-pypy3/etc/profile.d/conda.sh
conda activate sybil2
cd density-evaluator
python unified_analysis.py                    # runs BOTH first-exam-only and all-exams
python unified_analysis.py --first-exam-only  # only the first-exam workbook
python unified_analysis.py --include-mgh      # also runs the MGH + MGH-AI pipeline (gated)
python unified_analysis.py --time-varying-cox-only  # recompute ONLY the time_varying_cox
                                              # sheet and patch it into the existing
                                              # all_exams workbook(s); leaves the other
                                              # sheets (and their long bootstrap runs) alone


Output (per run mode): 
- results/unified_analysis_{first_exam_only|all_exams}.xlsx with sheets demographics, 
summary_stats, density_analysis, age_analysis, and (for all-exams) time_varying_cox. 
- When --include-mgh and the AI file exists, a second workbook prefixed ai_ is written
"""
import os
import gc
import argparse
import warnings
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pandas as pd
import joblib
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from lifelines import CoxPHFitter, CoxTimeVaryingFitter
from lifelines.utils import concordance_index
import statsmodels.formula.api as smf
from statsmodels.stats.contingency_tables import StratifiedTable
from scipy.stats.contingency import relative_risk, odds_ratio


@contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager that advances a tqdm bar as joblib jobs complete."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old
        tqdm_object.close()

warnings.filterwarnings("ignore")


# ----------------------------------------------------------------------
# Site configuration
# ----------------------------------------------------------------------

SITES = [
    {"name": "Chile",       "path": "data/chile/formatted_complete_DH_data.csv",                               "loader": "csv",   "ethnicity": "hispanic"},
    {"name": "UCSF",        "path": "data/UCSF/formatted_anonymized_risk_model_preds_w_outcomes_jan_5_2025.csv","loader": "csv",   "ethnicity": "unknown"},
    {"name": "Taiwan-CMU",  "path": "data/taiwan-cmu/input_ori_20250808.xlsx",                                  "loader": "excel", "ethnicity": "east asian"},
    {"name": "Taiwan-CGMH", "path": "data/taiwan-cgmh/formatted_minus_asymptomatic_08122025.csv",              "loader": "csv",   "ethnicity": "east asian"},
    {"name": "Apollo",      "path": "data/apollo/formatted_data.csv",                                          "loader": "csv",   "ethnicity": "south asian"},
    {"name": "Emory",       "path": "data/emory/input_file_one_outcome.csv",                                   "loader": "csv",   "ethnicity": "from_column"},
]

MGH_SITE = {
    "name": "MGH",
    "path": "data/MGH/mgh_radiologist_data 2.csv",
    "ai_path": "data/MGH/mgh_ai_data 2.csv",
    "loader": "csv",
    "ethnicity": "from_column",
}

CANONICAL_RACES = ["east asian", "black", "hispanic", "south asian", "white"]

# Map raw Ethnicity column values -> canonical races. Anything not in this
# dict becomes "other" and is excluded from race-subgroup analyses.
ETHNICITY_MAP = {
    "asian": "east asian",
    "east asian": "east asian",
    "chinese": "east asian",
    "japanese": "east asian",
    "korean": "east asian",
    "vietnamese": "east asian",
    "black": "black",
    "african american": "black",
    "african-american": "black",
    "hispanic": "hispanic",
    "hispanic or latino": "hispanic",
    "latino": "hispanic",
    "chilean": "hispanic",
    "south asian": "south asian",
    "indian": "south asian",
    "white": "white",
    "caucasian": "white",
}

AGE_GROUPS = ["30-40", "40-50", "50-60", "60-70", "70-80"]
DATE_COLUMNS = ["Exam Date", "Date of Last Negative Mammogram", "Date of Cancer Diagnosis"]
REQUIRED_COLUMNS = [
    "Patient ID", "Exam ID", "Exam Date", "Density", "Age at Exam",
    "Cancer (YES | NO)", "Date of Cancer Diagnosis", "Date of Last Negative Mammogram",
]


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def _normalize_ethnicity_column(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "from_column":
        if "Ethnicity" not in df.columns:
            df["Ethnicity"] = "other"
        df["Ethnicity"] = (
            df["Ethnicity"].astype(str).str.strip().str.lower().map(ETHNICITY_MAP).fillna("other")
        )
    else:
        df["Ethnicity"] = mode
    return df


def _load_one(site: dict) -> pd.DataFrame:
    if site["loader"] == "csv":
        df = pd.read_csv(site["path"])
    elif site["loader"] == "excel":
        df = pd.read_excel(site["path"])
    else:
        raise ValueError(f"unknown loader: {site['loader']}")

    # Apollo has duplicate "Cancer (YES | NO)" headers; pandas renames to ".1"
    if "Cancer (YES | NO).1" in df.columns:
        df = df.drop(columns=["Cancer (YES | NO).1"])

    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"[{site['name']}] missing required column: {col}")

    for col in DATE_COLUMNS:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    vals = df["Cancer (YES | NO)"]
    if vals.dtype == object:
        df["Cancer (YES | NO)"] = vals.astype(str).str.upper().map({"YES": 1, "NO": 0})
    df["Cancer (YES | NO)"] = df["Cancer (YES | NO)"].astype("Int64").astype(int)

    df["Density"] = pd.to_numeric(df["Density"], errors="coerce")
    df = df.dropna(subset=["Density", "Age at Exam", "Exam Date"])
    df["Density"] = df["Density"].astype(int)

    # Patient IDs vary by site (int vs str); normalize to str so groupby
    # keys stay consistent when sites are concatenated.
    df["Patient ID"] = (site["name"] + "::" + df["Patient ID"].astype(str))

    df = _normalize_ethnicity_column(df, site["ethnicity"])
    df["Site"] = site["name"]

    # days to cancer / days to last negative (relative to exam date)
    df["days_to_last_negative"] = (df["Date of Last Negative Mammogram"] - df["Exam Date"]).dt.days
    df["days_to_cancer"] = (df["Date of Cancer Diagnosis"] - df["Exam Date"]).dt.days

    return df


def _apply_base_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep exams with cancer within 5 years, OR an explicit last-negative
    mammogram at least 1 year after the exam. A future cancer diagnosis is
    NOT treated as evidence of a negative followup.
    """
    has_cancer = df["Cancer (YES | NO)"] == 1
    days_cancer = df["days_to_cancer"]
    days_neg = df["days_to_last_negative"]

    cancer_within_5 = (
        has_cancer & days_cancer.notna() & (days_cancer >= 0) & (days_cancer <= 5 * 365)
    )
    confirmed_1yr_negative = days_neg.notna() & (days_neg >= 365)
    return df[cancer_within_5 | confirmed_1yr_negative].reset_index(drop=True)


def load_sites(include_mgh: bool) -> dict:
    sites = list(SITES)
    if include_mgh and os.path.exists(MGH_SITE["path"]):
        sites.append(MGH_SITE)
    out = {}
    for site in sites:
        print(f"Loading {site['name']} from {site['path']}")
        df = _load_one(site)
        df = _apply_base_filter(df)
        print(f"  -> {len(df)} exams, {df['Patient ID'].nunique()} patients after base filter")
        out[site["name"]] = df
    return out


def filter_for_horizon(df: pd.DataFrame, horizon_years: int) -> pd.DataFrame:
    """
    Per-horizon validity filter.

    Replaces the buggy `format_data` from adjusted_eval.py which used
    integer years and admitted exams with any positive followup.

    Negatives require an explicit last-negative mammogram at or beyond the
    horizon. A future cancer diagnosis is not itself evidence of a negative
    followup.
    """
    min_days = horizon_years * 365
    has_cancer = df["Cancer (YES | NO)"] == 1
    days_cancer = df["days_to_cancer"]
    days_neg = df["days_to_last_negative"]

    is_cancer = has_cancer & days_cancer.notna() & (days_cancer >= 0) & (days_cancer <= min_days)
    is_neg = ~is_cancer & days_neg.notna() & (days_neg >= min_days)
    keep = is_cancer | is_neg

    out = df[keep].copy()
    out["cancer_outcome"] = is_cancer[keep].astype(int).values
    out["time_to_event_days"] = np.where(
        out["cancer_outcome"] == 1, out["days_to_cancer"], min_days
    )
    out["time_to_event"] = out["time_to_event_days"] / 365.0
    return out.reset_index(drop=True)


# ----------------------------------------------------------------------
# Demographics & summary stats
# ----------------------------------------------------------------------

def _count_pair(df: pd.DataFrame, mask=None) -> tuple:
    if mask is None:
        sub = df
    else:
        sub = df[mask]
    return len(sub), sub["Patient ID"].nunique()


def build_demographics(dataframes: dict, site_filter: list = None) -> pd.DataFrame:
    rows = []
    row_labels = []

    def add_row(label, per_site_counts):
        row_labels.append(label)
        rows.append(per_site_counts)

    show = list(dataframes.keys()) if site_filter is None else [s for s in site_filter if s in dataframes]
    site_names = show

    def per_site(fn):
        out = {}
        for name in show:
            df = dataframes[name]
            out[(name, "# exams")], out[(name, "# patients")] = fn(df)
        return out

    add_row("Total", per_site(lambda d: _count_pair(d)))
    add_row("Cancer diagnosis", per_site(lambda d: _count_pair(d, d["Cancer (YES | NO)"] == 1)))
    for density in [1, 2, 3, 4]:
        add_row(f"Density {density}", per_site(lambda d, dn=density: _count_pair(d, d["Density"] == dn)))
    for race in CANONICAL_RACES:
        add_row(f"Race: {race}", per_site(lambda d, r=race: _count_pair(d, d["Ethnicity"] == r)))
    bins = [(30, 40), (40, 50), (50, 60), (60, 70), (70, 80)]
    for lo, hi in bins:
        add_row(
            f"Age {lo}-{hi}",
            per_site(lambda d, a=lo, b=hi: _count_pair(d, (d["Age at Exam"] >= a) & (d["Age at Exam"] < b))),
        )

    def tte_mask(df, kind):
        # Matches the per-horizon filter: a "T-year negative" requires an
        # explicit last-negative mammogram at >= T*365 days, regardless of
        # future cancer status.
        has_cancer = df["Cancer (YES | NO)"] == 1
        days_neg = df["days_to_last_negative"]
        days_cancer = df["days_to_cancer"]
        if kind == "1yr cancer":
            return has_cancer & (days_cancer >= 0) & (days_cancer <= 365)
        if kind == "5yr cancer":
            return has_cancer & (days_cancer >= 0) & (days_cancer <= 5 * 365)
        if kind == "1yr neg":
            is_cancer_1yr = has_cancer & (days_cancer >= 0) & (days_cancer <= 365)
            return ~is_cancer_1yr & (days_neg >= 365)
        if kind == "5yr neg":
            is_cancer_5yr = has_cancer & (days_cancer >= 0) & (days_cancer <= 5 * 365)
            return ~is_cancer_5yr & (days_neg >= 5 * 365)

    add_row("Time to event: 1yr negative followup",
            per_site(lambda d: _count_pair(d, tte_mask(d, "1yr neg"))))
    add_row("Time to event: 1yr cancer",
            per_site(lambda d: _count_pair(d, tte_mask(d, "1yr cancer"))))
    add_row("Time to event: 5yr negative followup",
            per_site(lambda d: _count_pair(d, tte_mask(d, "5yr neg"))))
    add_row("Time to event: 5yr cancer",
            per_site(lambda d: _count_pair(d, tte_mask(d, "5yr cancer"))))

    columns = pd.MultiIndex.from_tuples(
        [(s, c) for s in site_names for c in ["# exams", "# patients"]]
    )
    table = pd.DataFrame(
        [[r.get(col, 0) for col in columns] for r in rows],
        index=row_labels,
        columns=columns,
    )
    return table


def compute_summary_stats(dataframes: dict, include_all_sites: bool = True,
                          site_filter: list = None) -> pd.DataFrame:
    all_df = pd.concat(dataframes.values(), ignore_index=True) if include_all_sites else None
    show = list(dataframes.keys()) if site_filter is None else [s for s in site_filter if s in dataframes]
    shown = {name: dataframes[name] for name in show}
    cols = show + (["All-sites"] if include_all_sites else [])
    rows = {}

    def med_range(df):
        a = df["Age at Exam"].dropna()
        return f"{a.median():.0f} ({a.min():.0f}-{a.max():.0f})"

    rows["Median age (min-max)"] = {name: med_range(df) for name, df in shown.items()}
    if include_all_sites:
        rows["Median age (min-max)"]["All-sites"] = med_range(all_df)

    for race in CANONICAL_RACES:
        row = {}
        for name, df in shown.items():
            total_pids = df["Patient ID"].nunique()
            race_pids = df[df["Ethnicity"] == race]["Patient ID"].nunique()
            row[name] = f"{100.0 * race_pids / total_pids:.1f}%" if total_pids > 0 else "0.0%"
        if include_all_sites:
            total_all = all_df["Patient ID"].nunique()
            race_all = all_df[all_df["Ethnicity"] == race]["Patient ID"].nunique()
            row["All-sites"] = f"{100.0 * race_all / total_all:.1f}%" if total_all > 0 else "0.0%"
        rows[f"Race %: {race} (by patients)"] = row

    def pct_dense(df):
        return f"{100.0 * (df['Density'] >= 3).mean():.1f}%" if len(df) else "NA"
    rows["% exams dense (3 or 4)"] = {name: pct_dense(df) for name, df in shown.items()}
    if include_all_sites:
        rows["% exams dense (3 or 4)"]["All-sites"] = pct_dense(all_df)

    def cancer_prev_1yr(df):
        n = len(df)
        if n == 0:
            return "NA"
        mask = (df["Cancer (YES | NO)"] == 1) & (df["days_to_cancer"] >= 0) & (df["days_to_cancer"] <= 365)
        return f"{100.0 * mask.sum() / n:.2f}%"
    rows["Cancer prevalence (1-yr)"] = {name: cancer_prev_1yr(df) for name, df in shown.items()}
    if include_all_sites:
        rows["Cancer prevalence (1-yr)"]["All-sites"] = cancer_prev_1yr(all_df)

    def pct_preceding(df, days):
        n = len(df)
        if n == 0:
            return "NA"
        mask = (df["Cancer (YES | NO)"] == 1) & (df["days_to_cancer"] >= 0) & (df["days_to_cancer"] <= days)
        return f"{100.0 * mask.sum() / n:.2f}%"
    rows["% exams preceding cancer <=1yr"] = {name: pct_preceding(df, 365) for name, df in shown.items()}
    rows["% exams preceding cancer <=5yr"] = {name: pct_preceding(df, 5 * 365) for name, df in shown.items()}
    if include_all_sites:
        rows["% exams preceding cancer <=1yr"]["All-sites"] = pct_preceding(all_df, 365)
        rows["% exams preceding cancer <=5yr"]["All-sites"] = pct_preceding(all_df, 5 * 365)

    return pd.DataFrame.from_dict(rows, orient="index").reindex(columns=cols)


# ----------------------------------------------------------------------
# Analysis core
# ----------------------------------------------------------------------

def ci_bounds(mean, deltas, confidence_interval=0.95, num_resamples=None):
    deltas = np.sort(deltas)
    if num_resamples is None:
        num_resamples = len(deltas)
    if num_resamples == 0:
        return float("nan"), float("nan")
    idx = max(int((1 - confidence_interval) / 2.0 * num_resamples), 1)
    lower_delta = deltas[idx]
    upper_delta = deltas[-idx]
    return mean - upper_delta, mean - lower_delta


def _binarize_density(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "1_2_vs_3_4":
        out = df.copy()
        out["exposed"] = (out["Density"] >= 3).astype(int)
    elif mode == "1_2_vs_4":
        out = df[df["Density"] != 3].copy()
        out["exposed"] = (out["Density"] == 4).astype(int)
    else:
        raise ValueError(mode)
    return out


METRIC_KEYS = ["rr", "or", "hr_unadjusted", "hr_age_adjusted",
               "cindex_unadjusted", "cindex_age_adjusted"]


def _informative_sites(df: pd.DataFrame, exposure_col: str) -> list:
    """Sites within a multi-site cohort that can contribute to a stratified
    estimate: at least one event AND both exposure levels present."""
    good = []
    for site, g in df.groupby("Site"):
        if g["cancer_outcome"].sum() >= 1 and g[exposure_col].nunique() >= 2:
            good.append(site)
    return good


def _resolve_sites(df: pd.DataFrame, exposure_col: str) -> tuple:
    """Decide how a cohort is analysed w.r.t. site identity.

    Returns (df_to_use, n_sites, site_adjusted):
      - single-site cohort         -> (df, 1, False)
      - >=2 informative sites      -> (df restricted to them, k, True)  [stratified]
      - exactly 1 informative site -> (df restricted to it, 1, False)   [plain single-site]
      - 0 informative sites        -> (empty df, 0, False)              [-> NaNs downstream]
    """
    if "Site" not in df.columns:
        return df, 1, False
    n_total = df["Site"].nunique()
    if n_total <= 1:
        return df, n_total, False
    good = _informative_sites(df, exposure_col)
    if len(good) >= 2:
        return df[df["Site"].isin(good)].reset_index(drop=True), len(good), True
    if len(good) == 1:
        return df[df["Site"] == good[0]].reset_index(drop=True), 1, False
    return df.iloc[0:0].copy(), 0, False


def _rr_or(df: pd.DataFrame, exposure_col: str, stratify: bool) -> tuple:
    """RR and OR. When ``stratify`` is True these are Mantel-Haenszel pooled
    over Site (uninformative strata are dropped automatically)."""
    if stratify:
        tables = []
        for _, g in df.groupby("Site"):
            ge = g[exposure_col].values
            gy = g["cancer_outcome"].values
            a = int(((ge == 1) & (gy == 1)).sum())
            b = int(((ge == 1) & (gy == 0)).sum())
            c = int(((ge == 0) & (gy == 1)).sum())
            d = int(((ge == 0) & (gy == 0)).sum())
            if (a + b) == 0 or (c + d) == 0 or (a + c) == 0:
                continue
            tables.append([[a, b], [c, d]])
        if len(tables) < 2:
            return float("nan"), float("nan")
        try:
            st = StratifiedTable(tables)
            return float(st.riskratio_pooled), float(st.oddsratio_pooled)
        except Exception:
            return float("nan"), float("nan")

    e = df[exposure_col].values
    y = df["cancer_outcome"].values
    a = int(((e == 1) & (y == 1)).sum())
    b = int(((e == 1) & (y == 0)).sum())
    c = int(((e == 0) & (y == 1)).sum())
    d = int(((e == 0) & (y == 0)).sum())
    if (a + b) == 0 or (c + d) == 0 or (a + c) == 0:
        return float("nan"), float("nan")
    try:
        rr = relative_risk(exposed_cases=a, exposed_total=a + b,
                           control_cases=c, control_total=c + d).relative_risk
    except Exception:
        rr = float("nan")
    try:
        or_stat = odds_ratio(np.array([[a, b], [c, d]])).statistic
    except Exception:
        or_stat = float("nan")
    return rr, or_stat


def _cox_hr(df: pd.DataFrame, exposure_col: str, adjust_age: bool, stratify: bool) -> float:
    """Cox HR for ``exposure_col``. When ``stratify`` is True the model is
    stratified by Site (each site keeps its own baseline hazard)."""
    covars = [exposure_col] + (["age"] if adjust_age else [])
    cols = ["time_to_event", "cancer_outcome"] + covars + (["Site"] if stratify else [])
    try:
        cph = CoxPHFitter().fit(
            df[cols], duration_col="time_to_event", event_col="cancer_outcome",
            strata=(["Site"] if stratify else None), fit_options={"max_steps": 100},
        )
        return float(np.exp(cph.params_.loc[exposure_col]))
    except Exception:
        return float("nan")


def _logit_cindex(df: pd.DataFrame, exposure_col: str, adjust_age: bool, stratify: bool) -> float:
    """Logistic-model c-index. Site (as ``C(Site)``) is added only to the
    age-adjusted variant, per design."""
    formula = f"cancer_outcome ~ {exposure_col}"
    if adjust_age:
        formula += " + age"
        if stratify:
            formula += " + C(Site)"
    try:
        model = smf.logit(formula, data=df).fit(disp=0)
        scores = model.predict(df)
        return float(concordance_index(df["time_to_event"], -scores, df["cancer_outcome"]))
    except Exception:
        return float("nan")


def _point_estimates(df: pd.DataFrame, exposure_col: str, stratify: bool) -> dict:
    rr, or_stat = _rr_or(df, exposure_col, stratify)
    return {
        "rr": rr,
        "or": or_stat,
        "hr_unadjusted": _cox_hr(df, exposure_col, adjust_age=False, stratify=stratify),
        "hr_age_adjusted": _cox_hr(df, exposure_col, adjust_age=True, stratify=stratify),
        "cindex_unadjusted": _logit_cindex(df, exposure_col, adjust_age=False, stratify=stratify),
        "cindex_age_adjusted": _logit_cindex(df, exposure_col, adjust_age=True, stratify=stratify),
    }


def _site_stratified_pids(df: pd.DataFrame, stratify: bool) -> dict:
    """Map a resampling-stratum label -> array of patient IDs in it. When
    ``stratify`` the strata are sites; otherwise a single pooled stratum."""
    if stratify:
        return {site: g["Patient ID"].unique() for site, g in df.groupby("Site")}
    return {"_all": df["Patient ID"].unique()}


def _resample_pids(rng: np.random.Generator, strata_pids: dict) -> np.ndarray:
    return np.concatenate([rng.choice(p, size=len(p), replace=True)
                           for p in strata_pids.values()])


def _single_bootstrap(i: int, df: pd.DataFrame, p_indices: dict, strata_pids: dict,
                      exposure_col: str, stratify: bool) -> dict:
    rng = np.random.default_rng(i)
    sample_pids = _resample_pids(rng, strata_pids)
    indices = np.concatenate([p_indices[p] for p in sample_pids])
    sample = df.iloc[indices].reset_index(drop=True)
    if sample["cancer_outcome"].nunique() < 2 or sample[exposure_col].nunique() < 2:
        return None
    try:
        return _point_estimates(sample, exposure_col, stratify)
    except Exception:
        return None


def _run_with_bootstrap(df: pd.DataFrame, exposure_col: str, n_bootstrap: int, n_jobs: int,
                        confidence_level: float = 0.95) -> tuple:
    """Returns (result, meta): result maps metric -> (point, (lo, hi)),
    meta = {"n_sites": int, "site_adjusted": bool}. Multi-site cohorts are
    analysed stratified by Site and the clustered bootstrap resamples patients
    within each site."""
    df, n_sites, stratify = _resolve_sites(df, exposure_col)
    meta = {"n_sites": n_sites, "site_adjusted": stratify}
    nan_result = {k: (float("nan"), (float("nan"), float("nan"))) for k in METRIC_KEYS}

    if len(df) < 10 or df["cancer_outcome"].nunique() < 2 or df[exposure_col].nunique() < 2:
        return nan_result, meta

    point = _point_estimates(df, exposure_col, stratify)

    p_indices = {pid: np.array(idxs) for pid, idxs in df.groupby("Patient ID").indices.items()}
    strata_pids = _site_stratified_pids(df, stratify)

    with tqdm_joblib(tqdm(total=n_bootstrap, desc="    bootstrap", leave=False, ncols=100)):
        boots = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_single_bootstrap)(i, df, p_indices, strata_pids, exposure_col, stratify)
            for i in range(n_bootstrap)
        )
    boots = [b for b in boots if b is not None]

    result = {}
    for k, v in point.items():
        if not boots or not np.isfinite(v):
            result[k] = (v, (float("nan"), float("nan")))
            continue
        samples = np.array([b[k] for b in boots if np.isfinite(b[k])])
        if len(samples) == 0:
            result[k] = (v, (float("nan"), float("nan")))
            continue
        deltas = samples - v
        lo, hi = ci_bounds(v, deltas, confidence_level, len(samples))
        result[k] = (v, (lo, hi))
    return result, meta


def run_density_analysis(df: pd.DataFrame, horizon_years: int, binarization: str,
                         n_bootstrap: int, n_jobs: int) -> tuple:
    sub = filter_for_horizon(df, horizon_years)
    sub = _binarize_density(sub, binarization)
    sub = sub.rename(columns={"Age at Exam": "age"})
    return _run_with_bootstrap(sub, "exposed", n_bootstrap, n_jobs)


# ----------------------------------------------------------------------
# Age-alone analysis
# ----------------------------------------------------------------------

def run_age_analysis(df: pd.DataFrame, horizon_years: int, mode: str,
                     n_bootstrap: int, n_jobs: int) -> tuple:
    """mode: 'continuous' or 'binary_70'. Returns (result, meta).

    'binary_70' goes through the shared density machinery (so it inherits the
    MH / stratified-Cox / site-stratified-bootstrap treatment). 'continuous'
    site-stratifies the Cox HR; the c-index stays site-unadjusted because it is
    the "unadjusted" variant.
    """
    sub = filter_for_horizon(df, horizon_years)
    sub = sub.rename(columns={"Age at Exam": "age"})

    if mode == "binary_70":
        sub["age_binary"] = (sub["age"] >= 70).astype(int)
        return _run_with_bootstrap(sub, "age_binary", n_bootstrap, n_jobs)

    # continuous: site-stratified Cox HR + site-unadjusted logistic c-index
    sub, n_sites, stratify = _resolve_sites(sub, "age")
    meta = {"n_sites": n_sites, "site_adjusted": stratify}
    if len(sub) < 10 or sub["cancer_outcome"].nunique() < 2:
        return ({k: (float("nan"), (float("nan"), float("nan"))) for k in
                 ["hr_unadjusted", "cindex_unadjusted"]}, meta)

    def _cont_point(d):
        cols = ["time_to_event", "cancer_outcome", "age"] + (["Site"] if stratify else [])
        try:
            cph = CoxPHFitter().fit(
                d[cols], duration_col="time_to_event", event_col="cancer_outcome",
                strata=(["Site"] if stratify else None), fit_options={"max_steps": 100})
            hr = float(np.exp(cph.params_.loc["age"]))
        except Exception:
            hr = float("nan")
        try:
            model = smf.logit("cancer_outcome ~ age", data=d).fit(disp=0)
            scores = model.predict(d)
            cix = float(concordance_index(d["time_to_event"], -scores, d["cancer_outcome"]))
        except Exception:
            cix = float("nan")
        return {"hr_unadjusted": hr, "cindex_unadjusted": cix}

    point = _cont_point(sub)

    p_indices = {pid: np.array(idxs) for pid, idxs in sub.groupby("Patient ID").indices.items()}
    strata_pids = _site_stratified_pids(sub, stratify)

    def _boot(i):
        rng = np.random.default_rng(i)
        idx = np.concatenate([p_indices[p] for p in _resample_pids(rng, strata_pids)])
        smp = sub.iloc[idx].reset_index(drop=True)
        if smp["cancer_outcome"].nunique() < 2:
            return None
        return _cont_point(smp)

    with tqdm_joblib(tqdm(total=n_bootstrap, desc="    bootstrap", leave=False, ncols=100)):
        boots = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_boot)(i) for i in range(n_bootstrap))
    boots = [b for b in boots if b is not None]

    result = {}
    for k, v in point.items():
        if not np.isfinite(v) or not boots:
            result[k] = (v, (float("nan"), float("nan")))
            continue
        samples = np.array([b[k] for b in boots if np.isfinite(b[k])])
        if len(samples) == 0:
            result[k] = (v, (float("nan"), float("nan")))
            continue
        deltas = samples - v
        lo, hi = ci_bounds(v, deltas, 0.95, len(samples))
        result[k] = (v, (lo, hi))
    return result, meta


# ----------------------------------------------------------------------
# Time-varying Cox
# ----------------------------------------------------------------------

def _build_long_format(df: pd.DataFrame, binarization: str,
                       min_followup_days: int = 365,
                       max_followup_days: int = 5 * 365) -> pd.DataFrame:
    """
    Build (start, stop, event] rows for CoxTimeVaryingFitter.

    Inclusion (one model, no horizon stratification): a patient needs at
    least one cancer-free year of follow-up from the initial mammogram and no
    prior (pre-mammogram) breast-cancer diagnosis. Follow-up is
    administratively censored at ``max_followup_days`` from the *first* exam;
    a cancer diagnosed after that cap is treated as a censoring event at the
    cap. Exams sorted by date define intervals; covariates (binary density,
    age) update at each exam, so risk is carried by the most recent density.
    The event flags the single interval that contains the diagnosis day.
    """
    sub = _binarize_density(df, binarization)
    sub = sub.sort_values(["Patient ID", "Exam Date"]).reset_index(drop=True)

    rows = []
    for pid, g in sub.groupby("Patient ID", sort=False):
        g = g.reset_index(drop=True)
        first_date = g["Exam Date"].iloc[0]
        had_cancer = bool((g["Cancer (YES | NO)"] == 1).any())
        cancer_day = (
            g["days_to_cancer"].dropna().iloc[0]
            if had_cancer and g["days_to_cancer"].notna().any() else None
        )
        last_neg_day = g["days_to_last_negative"].max()

        if had_cancer:
            # Prevalent (<1yr) or prior (<0) disease isn't incident risk.
            if cancer_day is None or cancer_day < min_followup_days:
                continue
            if cancer_day <= max_followup_days:
                end_day, event = cancer_day, 1
            else:
                end_day, event = max_followup_days, 0
        else:
            if pd.isna(last_neg_day) or last_neg_day < min_followup_days:
                continue
            end_day, event = min(last_neg_day, max_followup_days), 0

        for i in range(len(g)):
            start = (g["Exam Date"].iloc[i] - first_date).days
            if i < len(g) - 1:
                stop = (g["Exam Date"].iloc[i + 1] - first_date).days
            else:
                stop = end_day
            if start >= end_day:
                break
            stop = min(stop, end_day)
            if stop <= start:
                continue
            rows.append({
                "id": pid,
                "start": start / 365.0,
                "stop": stop / 365.0,
                "event": event if stop == end_day else 0,
                "exposed": int(g["exposed"].iloc[i]),
                "age": float(g["Age at Exam"].iloc[i]),
                "Site": g["Site"].iloc[i],
            })
    return pd.DataFrame(rows)


def run_time_varying_cox(df: pd.DataFrame, binarization: str, adjust_age: bool,
                         max_followup_years: int = 5) -> dict:
    long_df = _build_long_format(df, binarization,
                                 max_followup_days=max_followup_years * 365)
    base = {"hr": float("nan"), "ci": (float("nan"), float("nan")),
            "n": len(long_df), "n_sites": 0, "site_adjusted": False}
    if len(long_df) == 0 or long_df["event"].nunique() < 2:
        return base

    strata = None
    site_adjusted = False
    n_sites = long_df["Site"].nunique()
    if n_sites > 1:
        good = [s for s, g in long_df.groupby("Site")
                if g["event"].sum() >= 1 and g["exposed"].nunique() >= 2]
        if len(good) >= 2:
            long_df = long_df[long_df["Site"].isin(good)].reset_index(drop=True)
            strata, site_adjusted, n_sites = ["Site"], True, len(good)
        elif len(good) == 1:
            long_df = long_df[long_df["Site"] == good[0]].reset_index(drop=True)
            n_sites = 1
        else:
            return {**base, "n": len(long_df)}
        if long_df["event"].nunique() < 2:
            return {**base, "n": len(long_df), "n_sites": n_sites}

    cols = (["id", "start", "stop", "event", "exposed"]
            + (["age"] if adjust_age else []) + (["Site"] if strata else []))
    try:
        ctv = CoxTimeVaryingFitter().fit(
            long_df[cols], id_col="id", event_col="event",
            start_col="start", stop_col="stop", strata=strata,
        )
        s = ctv.summary
        hr = float(s.loc["exposed", "exp(coef)"])
        lo = float(s.loc["exposed", "exp(coef) lower 95%"])
        hi = float(s.loc["exposed", "exp(coef) upper 95%"])
        return {"hr": hr, "ci": (lo, hi), "n": len(long_df),
                "n_sites": n_sites, "site_adjusted": site_adjusted}
    except Exception as e:
        return {"hr": float("nan"), "ci": (float("nan"), float("nan")),
                "n": len(long_df), "n_sites": n_sites, "site_adjusted": site_adjusted,
                "error": str(e)}


# ----------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------

def _flatten_result(group_label: str, horizon: int, binarization: str,
                    result: dict, meta: dict) -> list:
    rows = []
    for metric, (point, (lo, hi)) in result.items():
        rows.append({
            "group": group_label,
            "horizon_years": horizon,
            "binarization": binarization,
            "metric": metric,
            "point": point,
            "ci_lower": lo,
            "ci_upper": hi,
            "n_sites": meta["n_sites"],
            "site_adjusted": meta["site_adjusted"],
        })
    return rows


def aggregate_density_analysis(dataframes: dict, n_bootstrap: int, n_jobs: int,
                               include_all_sites: bool = True,
                               site_filter: list = None) -> pd.DataFrame:
    show = list(dataframes.keys()) if site_filter is None else [s for s in site_filter if s in dataframes]
    cohorts = {f"site:{name}": dataframes[name] for name in show}
    if include_all_sites:
        all_df = pd.concat(dataframes.values(), ignore_index=True)
        cohorts["all-sites"] = all_df
        for lo, hi in [(30, 40), (40, 50), (50, 60), (60, 70), (70, 80)]:
            cohorts[f"age:{lo}-{hi}"] = all_df[(all_df["Age at Exam"] >= lo) & (all_df["Age at Exam"] < hi)]
        for race in CANONICAL_RACES:
            cohorts[f"race:{race}"] = all_df[all_df["Ethnicity"] == race]

    tasks = [
        (horizon, binarization, label, cohort)
        for horizon in [1, 5]
        for binarization in ["1_2_vs_3_4", "1_2_vs_4"]
        for label, cohort in cohorts.items()
        if len(cohort) > 0
    ]
    rows = []
    with tqdm(tasks, desc="density", ncols=100) as bar:
        for horizon, binarization, label, cohort in bar:
            bar.set_postfix_str(f"{label} | {horizon}yr | {binarization}")
            res, meta = run_density_analysis(cohort, horizon, binarization, n_bootstrap, n_jobs)
            rows.extend(_flatten_result(label, horizon, binarization, res, meta))
            gc.collect()
    return pd.DataFrame(rows)


def aggregate_age_analysis(dataframes: dict, n_bootstrap: int, n_jobs: int,
                           include_all_sites: bool = True,
                           site_filter: list = None) -> pd.DataFrame:
    show = list(dataframes.keys()) if site_filter is None else [s for s in site_filter if s in dataframes]
    cohorts = {f"site:{name}": dataframes[name] for name in show}
    if include_all_sites:
        all_df = pd.concat(dataframes.values(), ignore_index=True)
        cohorts["all-sites"] = all_df
        for race in CANONICAL_RACES:
            cohorts[f"race:{race}"] = all_df[all_df["Ethnicity"] == race]

    tasks = [
        (horizon, mode, label, cohort)
        for horizon in [1, 5]
        for mode in ["continuous", "binary_70"]
        for label, cohort in cohorts.items()
        if len(cohort) > 0
    ]
    rows = []
    with tqdm(tasks, desc="age", ncols=100) as bar:
        for horizon, mode, label, cohort in bar:
            bar.set_postfix_str(f"{label} | {horizon}yr | {mode}")
            res, meta = run_age_analysis(cohort, horizon, mode, n_bootstrap, n_jobs)
            for metric, (point, (lo, hi)) in res.items():
                rows.append({
                    "group": label,
                    "horizon_years": horizon,
                    "age_mode": mode,
                    "metric": metric,
                    "point": point,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "n_sites": meta["n_sites"],
                    "site_adjusted": meta["site_adjusted"],
                })
            gc.collect()
    return pd.DataFrame(rows)


def aggregate_time_varying_cox(dataframes: dict,
                                include_all_sites: bool = True,
                                site_filter: list = None,
                                max_followup_years: int = 5) -> pd.DataFrame:
    show = list(dataframes.keys()) if site_filter is None else [s for s in site_filter if s in dataframes]
    cohorts = {f"site:{name}": dataframes[name] for name in show}
    if include_all_sites:
        all_df = pd.concat(dataframes.values(), ignore_index=True)
        cohorts["all-sites"] = all_df
        for race in CANONICAL_RACES:
            cohorts[f"race:{race}"] = all_df[all_df["Ethnicity"] == race]

    tasks = [
        (binarization, adjust_age, label, cohort)
        for binarization in ["1_2_vs_3_4", "1_2_vs_4"]
        for adjust_age in [False, True]
        for label, cohort in cohorts.items()
        if len(cohort) > 0
    ]
    rows = []
    with tqdm(tasks, desc="tv-cox", ncols=100) as bar:
        for binarization, adjust_age, label, cohort in bar:
            bar.set_postfix_str(f"{label} | {binarization} | adj_age={adjust_age}")
            res = run_time_varying_cox(cohort, binarization, adjust_age, max_followup_years)
            rows.append({
                "group": label,
                "max_followup_years": max_followup_years,
                "binarization": binarization,
                "adjust_age": adjust_age,
                "hr": res["hr"],
                "ci_lower": res["ci"][0],
                "ci_upper": res["ci"][1],
                "n_intervals": res["n"],
                "n_sites": res["n_sites"],
                "site_adjusted": res["site_adjusted"],
            })
    return pd.DataFrame(rows)


def patch_time_varying_cox(dataframes: dict, output_dir: str, include_all_sites: bool,
                           site_filter: list = None, prefix: str = "") -> None:
    """Recompute only the time_varying_cox sheet and write it back into the
    existing all_exams workbook, leaving every other sheet untouched.

    Note: the replaced sheet is appended at the end of the workbook by
    openpyxl, so its tab position may change; the contents of the other
    sheets are not modified.
    """
    tag = f"{prefix}_all_exams" if prefix else "all_exams"
    out_path = os.path.join(output_dir, f"unified_analysis_{tag}.xlsx")
    if not os.path.exists(out_path):
        raise FileNotFoundError(
            f"{out_path} does not exist; run the full pipeline once before "
            f"using --time-varying-cox-only.")
    print(f"Recomputing time-varying Cox -> {out_path}")
    tv = aggregate_time_varying_cox(dataframes, include_all_sites=include_all_sites,
                                    site_filter=site_filter)
    with pd.ExcelWriter(out_path, mode="a", engine="openpyxl",
                        if_sheet_exists="replace") as writer:
        tv.to_excel(writer, sheet_name="time_varying_cox")
    print(f"Updated 'time_varying_cox' sheet in {out_path}")


# ----------------------------------------------------------------------
# Pipeline runner
# ----------------------------------------------------------------------

def _take_first_exam(dataframes: dict) -> dict:
    out = {}
    for name, df in dataframes.items():
        out[name] = (
            df.sort_values(["Patient ID", "Exam Date"])
              .drop_duplicates("Patient ID", keep="first")
              .reset_index(drop=True)
        )
    return out


def run_pipeline(dataframes: dict, label: str, output_dir: str,
                 n_bootstrap: int, n_jobs: int, first_exam_only: bool,
                 include_all_sites: bool, site_filter: list = None,
                 prefix: str = "") -> None:
    suffix = "first_exam_only" if first_exam_only else "all_exams"
    tag = f"{prefix}_{suffix}" if prefix else suffix
    out_path = os.path.join(output_dir, f"unified_analysis_{tag}.xlsx")

    sheets = {}
    print(f"[{label}] Building demographics")
    sheets["demographics"] = build_demographics(dataframes, site_filter=site_filter)
    print(f"[{label}] Computing summary stats")
    sheets["summary_stats"] = compute_summary_stats(
        dataframes, include_all_sites=include_all_sites, site_filter=site_filter)

    print(f"[{label}] Running density analysis")
    sheets["density_analysis"] = aggregate_density_analysis(
        dataframes, n_bootstrap, n_jobs,
        include_all_sites=include_all_sites, site_filter=site_filter)
    print(f"[{label}] Running age-alone analysis")
    sheets["age_analysis"] = aggregate_age_analysis(
        dataframes, n_bootstrap, n_jobs,
        include_all_sites=include_all_sites, site_filter=site_filter)

    if not first_exam_only:
        print(f"[{label}] Running time-varying Cox")
        sheets["time_varying_cox"] = aggregate_time_varying_cox(
            dataframes, include_all_sites=include_all_sites, site_filter=site_filter)

    os.makedirs(output_dir, exist_ok=True)
    with pd.ExcelWriter(out_path, mode="w") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31])
    print(f"[{label}] Wrote {out_path}")


# ----------------------------------------------------------------------
# MGH AI-density helpers
# ----------------------------------------------------------------------

def load_mgh_ai(include_mgh: bool) -> pd.DataFrame | None:
    if not include_mgh or not os.path.exists(MGH_SITE["ai_path"]):
        return None
    ai_site = {**MGH_SITE, "name": "MGH-AI", "path": MGH_SITE["ai_path"]}
    df = _load_one(ai_site)
    df = _apply_base_filter(df)
    return df


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Unified density analysis.")
    parser.add_argument("--include-mgh", action=argparse.BooleanOptionalAction, default=False,
                        help="Include MGH radiologist + AI density data.")
    parser.add_argument("--first-exam-only", action=argparse.BooleanOptionalAction, default=None,
                        help="Run on first exam per patient only. If omitted, runs BOTH.")
    parser.add_argument("--n-bootstrap", type=int, default=1000)
    parser.add_argument("--n-jobs", type=int, default=40)
    parser.add_argument("--output-dir", type=str, default="results_unified")
    parser.add_argument("--time-varying-cox-only", action="store_true",
                        help="Only recompute the time_varying_cox sheet and patch it "
                             "into the existing all_exams workbook(s); skips the "
                             "demographics / density / age analyses (and their bootstraps).")
    args = parser.parse_args()

    # Radiologist / primary density data
    dataframes = load_sites(include_mgh=args.include_mgh)

    # When --include-mgh: only emit MGH per-site rows + the all-sites aggregate
    # and its subgroups. Per-site rows for the other sites live in the
    # --no-include-mgh workbook.
    # When --no-include-mgh: per-site rows for every loaded site, no aggregate.
    if args.include_mgh:
        site_filter = ["MGH"]
        include_all_sites = True
        workbook_prefix = "with_mgh"
    else:
        site_filter = None
        include_all_sites = False
        workbook_prefix = ""

    if args.time_varying_cox_only:
        patch_time_varying_cox(dataframes, args.output_dir,
                               include_all_sites=include_all_sites,
                               site_filter=site_filter, prefix=workbook_prefix)
        ai_df = load_mgh_ai(args.include_mgh)
        if ai_df is not None:
            patch_time_varying_cox({"MGH-AI": ai_df}, args.output_dir,
                                   include_all_sites=True, prefix="ai")
        return

    modes = [args.first_exam_only] if args.first_exam_only is not None else [True, False]
    for first_exam in modes:
        cohort = _take_first_exam(dataframes) if first_exam else dataframes
        run_pipeline(cohort, f"radiologist first_exam={first_exam}", args.output_dir,
                     args.n_bootstrap, args.n_jobs, first_exam,
                     include_all_sites=include_all_sites,
                     site_filter=site_filter, prefix=workbook_prefix)

    # AI-density block (MGH only) — always runs the MGH-only cohort, so the
    # "all-sites" rollup is a single-site passthrough; include it for consistency.
    ai_df = load_mgh_ai(args.include_mgh)
    if ai_df is not None:
        ai_dataframes = {"MGH-AI": ai_df}
        for first_exam in modes:
            cohort = _take_first_exam(ai_dataframes) if first_exam else ai_dataframes
            run_pipeline(cohort, f"AI first_exam={first_exam}", args.output_dir,
                         args.n_bootstrap, args.n_jobs, first_exam,
                         include_all_sites=True, prefix="ai")


if __name__ == "__main__":
    main()
