import os
os.environ["TMPDIR"] = "/Users/leslierlamb/Desktop/Density/temp"
os.environ["LOKY_TMPDIR"] = "/Users/leslierlamb/Desktop/Density/temp"
os.environ["PYTHONWARNINGS"] = "ignore"
import warnings
import logging
import numpy as np
import pandas as pd
import datetime
import argparse
from collections import defaultdict
from tqdm import tqdm
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
import gc

# logging.getLogger("joblib").setLevel(logging.ERROR)
# warnings.filterwarnigns("ignore", message=".*unknown resource type.*")

def format_data(data: pd.DataFrame, max_followup: int) -> tuple:
    """
    Prepare the data for the evaluation.

    Parameters
    ----------
    data : pd.DataFrame
        The input data
    max_followup : int
        The truncation time in years

    Returns
    -------
    tuple
        A tuple of density, cancer_label, and censor_time
    """
    # set cancer date to 100 if no cancer
    data.loc[data["Cancer (YES | NO)"] == 0, "Date of Cancer Diagnosis"] = data.loc[
        data["Cancer (YES | NO)"] == 0, "Exam Date"
    ] + datetime.timedelta(days=int(100 * 365))

    ages = data["Age at Exam"]
    ethnicity = data["Ethnicity"]
    density = np.array(data["Density"])
    date = data["Exam Date"]
    ever_has_cancer = data["Cancer (YES | NO)"]
    last_negative_date = data["Date of Last Negative Mammogram"]
    cancer_date = data["Date of Cancer Diagnosis"]
    # compute the time of censoring relative to exam date
    years_to_last_negative = np.minimum(
        (last_negative_date - date).dt.days // 365, max_followup
    )
    days_to_last_negative = (last_negative_date - date).dt.days

    # compute the time of cancer diagnosis relative to exam date or max_followup
    years_to_cancer = ((cancer_date - date).dt.days // 365).copy()

    # positives are those who developed cancer within the follow-up period
    cancer_label = (years_to_cancer >= 0) & (years_to_cancer <= max_followup) & ever_has_cancer

    # valid rows
    valid_rows = (
        (years_to_last_negative >= max_followup) & (days_to_last_negative > 0)
    ) | (cancer_label & (years_to_cancer >= 0))

    # construct censor_time array
    censor_time = np.where(cancer_label, years_to_cancer, years_to_last_negative)

    # remove invalid rows
    density = density[valid_rows]
    cancer_label = cancer_label[valid_rows]
    censor_time = censor_time[valid_rows]
    patient_id = data["Patient ID"][valid_rows]
    ages = ages[valid_rows]
    ethnicity = ethnicity[valid_rows]
    site = (
        data["SITE"][valid_rows]
        if "SITE" in data.columns
        else pd.Series([None] * valid_rows.sum(), index=data.index[valid_rows])
    )
    return density, cancer_label, censor_time, patient_id, ages, ethnicity, site


def ci_bounds(mean, deltas, confidence_interval, num_resamples):
    """
    Returns the lower and upper bounds of the confidence interval.
    mean : the empirical mean
    deltas : deltas from the empirical mean from different samples
    """
    deltas = np.sort(deltas)
    index_offset = int((1 - confidence_interval) / 2.0 * num_resamples)
    lower_delta, upper_delta = deltas[index_offset], deltas[-index_offset]
    lower_bound, upper_bound = mean - upper_delta, mean - lower_delta
    return lower_bound, upper_bound


def run_single_bootstrap(i, data, p_indices, u_patients, n_p, base_formula="age"):
    # Set seed based on iteration for reproducibility in parallel
    np.random.seed(i)

    # Fast clustered sampling
    sample_pids = np.random.choice(u_patients, size=n_p, replace=True)
    indices = [idx for pid in sample_pids for idx in p_indices[pid]]
    sample_df = data.iloc[indices].copy().reset_index(drop=True)

    if (
        sample_df["cancer_outcome"].nunique() < 2
    ):  # need at least one positive and one negative
        return None

    try:
        cph_age = CoxPHFitter().fit(
            sample_df,
            "time_to_event",
            "cancer_outcome",
            formula=base_formula,
            fit_options={"max_steps": 100},
        )
        cph_full = CoxPHFitter().fit(
            sample_df,
            "time_to_event",
            "cancer_outcome",
            formula=f"{base_formula} + density",
            fit_options={"max_steps": 100},
        )
        c_full = cph_full.concordance_index_
        c_age = cph_age.concordance_index_
        output = {
            "c_age": c_age,
            "c_full": c_full,
            "delta_c": c_full - c_age,
        }

        sample_df.loc[:, "pred_age"] = cph_age.predict_partial_hazard(sample_df)

        sample_df.loc[:, "pred_full"] = cph_full.predict_partial_hazard(sample_df)
        for group in ["30-40", "40-50", "50-60", "60-70", "70-80"]:
            try:
                age_lower, age_upper = group.split("-")
                age_lower, age_upper = int(age_lower), int(age_upper)
                group_data = (
                    sample_df[
                        (sample_df["age"] >= age_lower) & (sample_df["age"] < age_upper)
                    ]
                    .copy()
                    .reset_index(drop=True)
                )
                c_age = concordance_index(
                    group_data["time_to_event"],
                    -group_data["pred_age"],
                    group_data["cancer_outcome"],
                )
                c_full = concordance_index(
                    group_data["time_to_event"],
                    -group_data["pred_full"],
                    group_data["cancer_outcome"],
                )
                output[f"c_age_{group}"] = c_age
                output[f"c_full_{group}"] = c_full
                output[f"delta_c_{group}"] = c_full - c_age
            except:
                continue

        for group in data["ethnicity"].unique():
            try:
                group_data = (
                    sample_df[sample_df["ethnicity"] == group]
                    .copy()
                    .reset_index(drop=True)
                )
                if len(group_data) == 0:
                    continue
                c_age = concordance_index(
                    group_data["time_to_event"],
                    -group_data["pred_age"],
                    group_data["cancer_outcome"],
                )
                c_full = concordance_index(
                    group_data["time_to_event"],
                    -group_data["pred_full"],
                    group_data["cancer_outcome"],
                )
                output[f"c_age_{group}"] = c_age
                output[f"c_full_{group}"] = c_full
                output[f"delta_c_{group}"] = c_full - c_age

            except:
                continue

        return output
    except:
        return None


def run_single_bootstrap_log(i, data, p_indices, u_patients, n_p, base_formula="cancer_outcome ~ age"):
    # Set seed based on iteration for reproducibility in parallel
    np.random.seed(i)
    
    # Clustered bootstrap: replicate rows for multiply-sampled patients
    sample_pids = np.random.choice(u_patients, size=n_p, replace=True)
    indices = [idx for pid in sample_pids for idx in p_indices[pid]]
    sample_df = data.iloc[indices].copy().reset_index(drop=True)
    try:

        model_age  = smf.logit(base_formula, data=sample_df).fit(disp=0)
        model_full = smf.logit(f"{base_formula} + density", data=sample_df).fit(disp=0)

        sample_df.loc[:, "pred_age"]  = model_age.predict(sample_df)
        sample_df.loc[:,"pred_full"] = model_full.predict(sample_df)

        # predict and score
        c_age  = concordance_index(sample_df['time_to_event'], -sample_df["pred_age"],  sample_df['cancer_outcome'])
        c_full = concordance_index(sample_df['time_to_event'], -sample_df["pred_full"], sample_df['cancer_outcome'])

        output = {"c_age": c_age, "c_full": c_full, "delta_c": c_full - c_age}

        for group in ["30-40", "40-50", "50-60", "60-70", "70-80"]:
            age_lower, age_upper = int(group.split("-")[0]), int(group.split("-")[1])
            group_data = sample_df[(sample_df["age"] >= age_lower) & (sample_df["age"] < age_upper)]
            if len(group_data) == 0 or group_data["cancer_outcome"].nunique() < 2:
                continue
            try:
                c_age_g  = concordance_index(group_data["time_to_event"], -group_data["pred_age"],  group_data["cancer_outcome"])
                c_full_g = concordance_index(group_data["time_to_event"], -group_data["pred_full"], group_data["cancer_outcome"])
                output[f"c_age_{group}"]   = c_age_g
                output[f"c_full_{group}"]  = c_full_g
                output[f"delta_c_{group}"] = c_full_g - c_age_g
            except:
                continue

        for group in sample_df["ethnicity"].unique():
            group_data = sample_df[sample_df["ethnicity"] == group]
            if len(group_data) == 0 or group_data["cancer_outcome"].nunique() < 2:
                continue
            try:
                c_age_g  = concordance_index(group_data["time_to_event"], -group_data["pred_age"],  group_data["cancer_outcome"])
                c_full_g = concordance_index(group_data["time_to_event"], -group_data["pred_full"], group_data["cancer_outcome"])
                output[f"c_age_{group}"]   = c_age_g
                output[f"c_full_{group}"]  = c_full_g
                output[f"delta_c_{group}"] = c_full_g - c_age_g
            except:
                continue

        return output
    except:
        return None    
    

def calculate_age_adjusted_metrics(
    density,
    cancer_label,
    censor_time,
    ages,
    pids,
    ethnicity,
    confidence_level=0.95,
    n_bootstrap=1000,
    random_state=0,
    site=None,
):
    output = {}
    minidf = pd.DataFrame(
        {
            "density": density,
            "cancer_outcome": cancer_label.astype(int),
            "time_to_event": censor_time,
            "age": ages,
            "pids": pids,
            "ethnicity": ethnicity,
        }
    )
    if site is not None:
        minidf.loc[:,"site"] = np.array(site)
        base_formula = "cancer_outcome ~ age + C(site)"
        rhs_formula  = "age + C(site)"
    else:
        base_formula = "cancer_outcome ~ age"
        rhs_formula  = "age"

    grouped_indices = minidf.groupby("pids").indices
    unique_patients = minidf["pids"].unique()
    n_patients = len(unique_patients)

    results = Parallel(n_jobs=40, backend="multiprocessing")(
        delayed(run_single_bootstrap_log)(
            i, minidf, grouped_indices, unique_patients, n_patients, base_formula
        )
        for i in tqdm(range(n_bootstrap), ncols=100)
    )

    # bootstrappred results for c-index
    results = [c for c in results if c is not None]
    c_deltas = [c["delta_c"] for c in results]
    c_ages = [c["c_age"] for c in results]
    c_fulls = [c["c_full"] for c in results]

    # # Cox proportional hazards model for ratio
    # cph_age = CoxPHFitter().fit(
    #     minidf, "time_to_event", "cancer_outcome", formula=base_formula
    # )
    # c_age = cph_age.concordance_index_

    cph_full = CoxPHFitter().fit(
        minidf, "time_to_event", "cancer_outcome", formula=f"{rhs_formula} + density"
    )
    # c_full = cph_full.concordance_index_


    model_age  = smf.logit(base_formula, data=minidf).fit(disp=0)
    model_full = smf.logit(f"{base_formula} + density", data=minidf).fit(disp=0)

    minidf.loc[:,"pred_age"]  = model_age.predict(minidf)
    minidf.loc[:,"pred_full"] = model_full.predict(minidf)

    # predict and score
    c_age  = concordance_index(minidf['time_to_event'], -minidf['pred_age'],  minidf['cancer_outcome'])
    c_full = concordance_index(minidf['time_to_event'], -minidf['pred_full'], minidf['cancer_outcome'])


    delta_c = c_full - c_age

    # minidf.loc[:, "pred_age"] = cph_age.predict_partial_hazard(minidf)
    # minidf.loc[:, "pred_full"] = cph_full.predict_partial_hazard(minidf)

    # CI for c-index
    n_results = len(results)
    deltas = np.array(c_deltas) - delta_c
    delta_lower, delta_upper = ci_bounds(delta_c, deltas, confidence_level, n_results)
    age_lower, age_upper = ci_bounds(
        c_age, np.array(c_ages) - c_age, confidence_level, n_results
    )
    full_lower, full_upper = ci_bounds(
        c_full, np.array(c_fulls) - c_full, confidence_level, n_results
    )

    output = {
        "full_model_cindex": c_full,
        "age_only_model_cindex": c_age,
        "delta_c_index": delta_c,
        "delta_c_index_ci": (delta_lower, delta_upper),
        "full_model_cindex_ci": (full_lower, full_upper),
        "age_only_model_cindex_ci": (age_lower, age_upper),
    }

    # CI for age groups
    for group in ["30-40", "40-50", "50-60", "60-70", "70-80"]:
        c_age_group = [c[f"c_age_{group}"] for c in results if f"c_age_{group}" in c]
        c_full_group = [c[f"c_full_{group}"] for c in results if f"c_full_{group}" in c]
        delta_c_group = [
            c[f"delta_c_{group}"] for c in results if f"delta_c_{group}" in c
        ]

        if len(c_age_group) == 0:
            continue

        groupdf = (
            minidf[
                (minidf["age"] >= int(group.split("-")[0]))
                & (minidf["age"] < int(group.split("-")[1]))
            ]
            .copy()
            .reset_index(drop=True)
        )
        if groupdf["cancer_outcome"].nunique() < 2:
            continue

        c_age = concordance_index(
            groupdf["time_to_event"],
            -groupdf["pred_age"],
            groupdf["cancer_outcome"],
        )
        c_full = concordance_index(
            groupdf["time_to_event"],
            -groupdf["pred_full"],
            groupdf["cancer_outcome"],
        )
        delta_c_empirical = c_full - c_age
        deltas = np.array(delta_c_group) - delta_c_empirical
        delta_lower, delta_upper = ci_bounds(
            delta_c_empirical, deltas, confidence_level, len(deltas)
        )
        age_lower_ci, age_upper_ci = ci_bounds(
            c_age,
            np.array(c_age_group) - c_age,
            confidence_level,
            len(c_age_group),
        )
        full_lower_ci, full_upper_ci = ci_bounds(
            c_full,
            np.array(c_full_group) - c_full,
            confidence_level,
            len(c_full_group),
        )
        output.update(
            {
                f"full_model_age_{group}_cindex": c_full,
                f"age_only_model_age_{group}_cindex": c_age,
                f"delta_c_index_age_{group}": delta_c_empirical,
                f"delta_c_index_{group}_ci": (delta_lower, delta_upper),
                f"full_model_{group}_cindex_ci": (full_lower_ci, full_upper_ci),
                f"age_only_model_{group}_cindex_ci": (age_lower_ci, age_upper_ci),
            }
        )
    # CI for race / ethnicity
    for group in minidf["ethnicity"].unique():
        c_age_group = [c[f"c_age_{group}"] for c in results if f"c_age_{group}" in c]
        c_full_group = [c[f"c_full_{group}"] for c in results if f"c_full_{group}" in c]
        delta_c_group = [
            c[f"delta_c_{group}"] for c in results if f"delta_c_{group}" in c
        ]

        if len(c_age_group) == 0:
            continue

        groupdf = minidf[minidf["ethnicity"] == group].copy().reset_index(drop=True)
        if groupdf["cancer_outcome"].nunique() < 2:
            continue
        c_age = concordance_index(
            groupdf["time_to_event"],
            -groupdf["pred_age"],
            groupdf["cancer_outcome"],
        )
        c_full = concordance_index(
            groupdf["time_to_event"],
            -groupdf["pred_full"],
            groupdf["cancer_outcome"],
        )
        delta_c_empirical = c_full - c_age
        deltas = np.array(delta_c_group) - delta_c_empirical
        delta_lower, delta_upper = ci_bounds(
            delta_c_empirical, deltas, confidence_level, len(deltas)
        )
        age_lower_ci, age_upper_ci = ci_bounds(
            c_age,
            np.array(c_age_group) - c_age,
            confidence_level,
            len(c_age_group),
        )
        full_lower_ci, full_upper_ci = ci_bounds(
            c_full,
            np.array(c_full_group) - c_full,
            confidence_level,
            len(c_full_group),
        )
        output.update(
            {
                f"full_model_ethnicity_{group}_cindex": c_full,
                f"age_only_model_ethnicity_{group}_cindex": c_age,
                f"delta_c_index_ethnicity_{group}": delta_c_empirical,
                f"delta_c_index_ethnicity_{group}_ci": (delta_lower, delta_upper),
                f"full_model_ethnicity_{group}_cindex_ci": (
                    full_lower_ci,
                    full_upper_ci,
                ),
                f"age_only_model_ethnicity_{group}_cindex_ci": (
                    age_lower_ci,
                    age_upper_ci,
                ),
            }
        )

    # compute adjusted hazard ratio for density
    summary_full = cph_full.summary
    hr = summary_full.loc["density", "exp(coef)"]
    hr_lower = summary_full.loc["density", "exp(coef) lower 95%"]
    hr_upper = summary_full.loc["density", "exp(coef) upper 95%"]

    output.update(
        {
            "density_adjusted_hr": hr,
            "density_adjusted_or": np.exp(model_full.params["density"]),
            "density_adjusted_hr_ci": (hr_lower, hr_upper),
            "density_adjusted_or_ci": model_full.conf_int()
            .loc["density"]
            .apply(np.exp)
            .tolist(),
        }
    )

    # hazard ratio and odds ratio per group
    for group in ["30-40", "40-50", "50-60", "60-70", "70-80"]:
        groupdf = (
            minidf[
                (minidf["age"] >= int(group.split("-")[0]))
                & (minidf["age"] < int(group.split("-")[1]))
            ]
            .copy()
            .reset_index(drop=True)
        )
        if groupdf["cancer_outcome"].nunique() < 2:
            continue

        cph_full_group = CoxPHFitter().fit(
            groupdf,
            "time_to_event",
            "cancer_outcome",
            formula=f"{rhs_formula} + density",
        )
        summary_full_group = cph_full_group.summary
        hr_group = summary_full_group.loc["density", "exp(coef)"]
        hr_lower_group = summary_full_group.loc["density", "exp(coef) lower 95%"]
        hr_upper_group = summary_full_group.loc["density", "exp(coef) upper 95%"]

        logit_model_group = smf.logit(
            f"{base_formula} + density", data=groupdf
        ).fit(disp=0)

        output.update(
            {
                f"density_adjusted_age_{group}_hr": hr_group,
                f"density_adjusted_age_{group}_or": np.exp(
                    logit_model_group.params["density"]
                ),
                f"density_adjusted_age_{group}_hr_ci": (hr_lower_group, hr_upper_group),
                f"density_adjusted_age_{group}_or_ci": logit_model_group.conf_int()
                .loc["density"]
                .apply(np.exp)
                .tolist(),
            }
        )
    for group in minidf["ethnicity"].unique():
        groupdf = minidf[minidf["ethnicity"] == group].copy().reset_index(drop=True)
        if groupdf["cancer_outcome"].nunique() < 2:
            continue
        cph_full_group = CoxPHFitter().fit(
            groupdf,
            "time_to_event",
            "cancer_outcome",
            formula=f"{rhs_formula} + density",
        )
        summary_full_group = cph_full_group.summary
        hr_group = summary_full_group.loc["density", "exp(coef)"]
        hr_lower_group = summary_full_group.loc["density", "exp(coef) lower 95%"]
        hr_upper_group = summary_full_group.loc["density", "exp(coef) upper 95%"]
        logit_model_group = smf.logit(
            f"{base_formula} + density", data=groupdf
        ).fit(disp=0)
        output.update(
            {
                f"density_adjusted_ethnicity_{group}_hr": hr_group,
                f"density_adjusted_ethnicity_{group}_or": np.exp(
                    logit_model_group.params["density"]
                ),
                f"density_adjusted_ethnicity_{group}_hr_ci": (
                    hr_lower_group,
                    hr_upper_group,
                ),
                f"density_adjusted_ethnicity_{group}_or_ci": logit_model_group.conf_int()
                .loc["density"]
                .apply(np.exp)
                .tolist(),
            }
        )

    del minidf
    gc.collect()

    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate age-adjusted density metrics for MGH."
    )
    parser.add_argument(
        "--first-exam-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "If true, keep only the first exam per patient. "
            "Use --no-first-exam-only to evaluate on all exams."
        ),
    )
    args = parser.parse_args()

    def maybe_first_exam(df):
        if args.first_exam_only:
            df = df.sort_values(["Patient ID", "Exam Date"], ascending=[True, True])
            df = df.drop_duplicates(subset="Patient ID", keep="first")
        return df

    output_suffix = "first_exam_only" if args.first_exam_only else "all_exams"

    # parameters
    # ----------------------------------------------------------------
    confidence_level = 0.95
    n_bootstrap = 1000

    # ----------------------------------------------------------------
    # Prepare output file paths
    mgh_ai_data_file = "mgh_ai_data 2.csv"
    mgh_rad_data_file = "mgh_radiologist_data 2.csv"
    other_sites_data_file = "all_sites_minus_colombia_revised.csv"

    # ----------------------------------------------------------------
    # MGH Auomated Density Data
    mgh_ai_data = pd.read_csv(mgh_ai_data_file)

    minitable = defaultdict(list)

    for col in [
        "Exam Date",
        "Date of Last Negative Mammogram",
        "Date of Cancer Diagnosis",
    ]:
        mgh_ai_data[col] = pd.to_datetime(mgh_ai_data[col])
    mgh_ai_data = maybe_first_exam(mgh_ai_data)

    for max_followup in [0, 4]:
        density, cancer_label, censor_time, patient_ids, patient_age, ethnicity, _ = (
            format_data(mgh_ai_data, max_followup)
        )
        metrics = calculate_age_adjusted_metrics(
            density,
            cancer_label,
            censor_time,
            patient_age,
            patient_ids,
            ethnicity,
            n_bootstrap=n_bootstrap,
        )
        minitable["group"].append("AI density - Overall")
        minitable["max_followup"].append(max_followup + 1)
        minitable["full_c_index"].append(
            f"{metrics['full_model_cindex']:.2f} ({metrics['full_model_cindex_ci'][0]:.2f}-{metrics['full_model_cindex_ci'][1]:.2f})"
        )
        minitable["age_only_c_index"].append(
            f"{metrics['age_only_model_cindex']:.2f} ({metrics['age_only_model_cindex_ci'][0]:.2f}-{metrics['age_only_model_cindex_ci'][1]:.2f})"
        )
        minitable["delta_c_index"].append(
            f"{metrics['delta_c_index']:.2f} ({metrics['delta_c_index_ci'][0]:.2f}-{metrics['delta_c_index_ci'][1]:.2f})"
        )
        minitable["hazard_ratio"].append(
            f"{metrics['density_adjusted_hr']:.2f} ({metrics['density_adjusted_hr_ci'][0]:.2f}-{metrics['density_adjusted_hr_ci'][1]:.2f})"
        )
        minitable["odds_ratio"].append(
            f"{metrics['density_adjusted_or']:.2f} ({metrics['density_adjusted_or_ci'][0]:.2f}-{metrics['density_adjusted_or_ci'][1]:.2f})"
        )

        # age groups CI
        for group in ["30-40", "40-50", "50-60", "60-70", "70-80"]:
            minitable["group"].append("AI density - Age: " + group)
            minitable["max_followup"].append(max_followup + 1)
            minitable["full_c_index"].append(
                f"{metrics[f'full_model_age_{group}_cindex']:.2f} ({metrics[f'full_model_{group}_cindex_ci'][0]:.2f}-{metrics[f'full_model_{group}_cindex_ci'][1]:.2f})"
            )
            minitable["age_only_c_index"].append(
                f"{metrics[f'age_only_model_age_{group}_cindex']:.2f} ({metrics[f'age_only_model_{group}_cindex_ci'][0]:.2f}-{metrics[f'age_only_model_{group}_cindex_ci'][1]:.2f})"
            )
            minitable["delta_c_index"].append(
                f"{metrics[f'delta_c_index_age_{group}']:.2f} ({metrics[f'delta_c_index_{group}_ci'][0]:.2f}-{metrics[f'delta_c_index_{group}_ci'][1]:.2f})"
            )
            minitable["hazard_ratio"].append(
                f"{metrics[f'density_adjusted_age_{group}_hr']:.2f} ({metrics[f'density_adjusted_age_{group}_hr_ci'][0]:.2f}-{metrics[f'density_adjusted_age_{group}_hr_ci'][1]:.2f})"
            )
            minitable["odds_ratio"].append(
                f"{metrics[f'density_adjusted_age_{group}_or']:.2f} ({metrics[f'density_adjusted_age_{group}_or_ci'][0]:.2f}-{metrics[f'density_adjusted_age_{group}_or_ci'][1]:.2f})"
            )

        # race / ethnicity CI
        for group in mgh_ai_data["Ethnicity"].unique():
            if group in ["Ninguno", "Otro", "Amerindio", "Gitano (Rom)"]:
                continue
            minitable["group"].append("AI density - Ethnicity: " + group)
            minitable["max_followup"].append(max_followup + 1)
            minitable["full_c_index"].append(
                f"{metrics[f'full_model_ethnicity_{group}_cindex']:.2f} ({metrics[f'full_model_ethnicity_{group}_cindex_ci'][0]:.2f}-{metrics[f'full_model_ethnicity_{group}_cindex_ci'][1]:.2f})"
            )
            minitable["age_only_c_index"].append(
                f"{metrics[f'age_only_model_ethnicity_{group}_cindex']:.2f} ({metrics[f'age_only_model_ethnicity_{group}_cindex_ci'][0]:.2f}-{metrics[f'age_only_model_ethnicity_{group}_cindex_ci'][1]:.2f})"
            )
            minitable["delta_c_index"].append(
                f"{metrics[f'delta_c_index_ethnicity_{group}']:.2f} ({metrics[f'delta_c_index_ethnicity_{group}_ci'][0]:.2f}-{metrics[f'delta_c_index_ethnicity_{group}_ci'][1]:.2f})"
            )
            minitable["hazard_ratio"].append(
                f"{metrics[f'density_adjusted_ethnicity_{group}_hr']:.2f} ({metrics[f'density_adjusted_ethnicity_{group}_hr_ci'][0]:.2f}-{metrics[f'density_adjusted_ethnicity_{group}_hr_ci'][1]:.2f})"
            )
            minitable["odds_ratio"].append(
                f"{metrics[f'density_adjusted_ethnicity_{group}_or']:.2f} ({metrics[f'density_adjusted_ethnicity_{group}_or_ci'][0]:.2f}-{metrics[f'density_adjusted_ethnicity_{group}_or_ci'][1]:.2f})"
            )

    # ----------------------------------------------------------------
    # MGH Radiologist Density Data
    mgh_rad_data = pd.read_csv(mgh_rad_data_file)
    for col in [
        "Exam Date",
        "Date of Last Negative Mammogram",
        "Date of Cancer Diagnosis",
    ]:
        mgh_rad_data[col] = pd.to_datetime(mgh_rad_data[col])
    mgh_rad_data = maybe_first_exam(mgh_rad_data)

    for max_followup in [0, 4]:
        density, cancer_label, censor_time, patient_ids, patient_age, ethnicity, _ = (
            format_data(mgh_rad_data, max_followup)
        )
        metrics = calculate_age_adjusted_metrics(
            density,
            cancer_label,
            censor_time,
            patient_age,
            patient_ids,
            ethnicity,
            n_bootstrap=n_bootstrap,
        )
        minitable["group"].append("Radiologist density - Overall")
        minitable["max_followup"].append(max_followup + 1)
        minitable["full_c_index"].append(
            f"{metrics['full_model_cindex']:.2f} ({metrics['full_model_cindex_ci'][0]:.2f}-{metrics['full_model_cindex_ci'][1]:.2f})"
        )
        minitable["age_only_c_index"].append(
            f"{metrics['age_only_model_cindex']:.2f} ({metrics['age_only_model_cindex_ci'][0]:.2f}-{metrics['age_only_model_cindex_ci'][1]:.2f})"
        )
        minitable["delta_c_index"].append(
            f"{metrics['delta_c_index']:.2f} ({metrics['delta_c_index_ci'][0]:.2f}-{metrics['delta_c_index_ci'][1]:.2f})"
        )
        minitable["hazard_ratio"].append(
            f"{metrics['density_adjusted_hr']:.2f} ({metrics['density_adjusted_hr_ci'][0]:.2f}-{metrics['density_adjusted_hr_ci'][1]:.2f})"
        )
        minitable["odds_ratio"].append(
            f"{metrics['density_adjusted_or']:.2f} ({metrics['density_adjusted_or_ci'][0]:.2f}-{metrics['density_adjusted_or_ci'][1]:.2f})"
        )

        # age groups CI
        for group in ["30-40", "40-50", "50-60", "60-70", "70-80"]:
            minitable["group"].append("Radiologist density - Age: " + group)
            minitable["max_followup"].append(max_followup + 1)
            minitable["full_c_index"].append(
                f"{metrics[f'full_model_age_{group}_cindex']:.2f} ({metrics[f'full_model_{group}_cindex_ci'][0]:.2f}-{metrics[f'full_model_{group}_cindex_ci'][1]:.2f})"
            )
            minitable["age_only_c_index"].append(
                f"{metrics[f'age_only_model_age_{group}_cindex']:.2f} ({metrics[f'age_only_model_{group}_cindex_ci'][0]:.2f}-{metrics[f'age_only_model_{group}_cindex_ci'][1]:.2f})"
            )
            minitable["delta_c_index"].append(
                f"{metrics[f'delta_c_index_age_{group}']:.2f} ({metrics[f'delta_c_index_{group}_ci'][0]:.2f}-{metrics[f'delta_c_index_{group}_ci'][1]:.2f})"
            )
            minitable["hazard_ratio"].append(
                f"{metrics[f'density_adjusted_age_{group}_hr']:.2f} ({metrics[f'density_adjusted_age_{group}_hr_ci'][0]:.2f}-{metrics[f'density_adjusted_age_{group}_hr_ci'][1]:.2f})"
            )
            minitable["odds_ratio"].append(
                f"{metrics[f'density_adjusted_age_{group}_or']:.2f} ({metrics[f'density_adjusted_age_{group}_or_ci'][0]:.2f}-{metrics[f'density_adjusted_age_{group}_or_ci'][1]:.2f})"
            )

        # race / ethnicity CI
        for group in mgh_rad_data["Ethnicity"].unique():
            if group in ["Ninguno", "Otro", "Amerindio", "Gitano (Rom)"]:
                continue
            minitable["group"].append("Radiologist density - Ethnicity: " + group)
            minitable["max_followup"].append(max_followup + 1)
            minitable["full_c_index"].append(
                f"{metrics[f'full_model_ethnicity_{group}_cindex']:.2f} ({metrics[f'full_model_ethnicity_{group}_cindex_ci'][0]:.2f}-{metrics[f'full_model_ethnicity_{group}_cindex_ci'][1]:.2f})"
            )
            minitable["age_only_c_index"].append(
                f"{metrics[f'age_only_model_ethnicity_{group}_cindex']:.2f} ({metrics[f'age_only_model_ethnicity_{group}_cindex_ci'][0]:.2f}-{metrics[f'age_only_model_ethnicity_{group}_cindex_ci'][1]:.2f})"
            )
            minitable["delta_c_index"].append(
                f"{metrics[f'delta_c_index_ethnicity_{group}']:.2f} ({metrics[f'delta_c_index_ethnicity_{group}_ci'][0]:.2f}-{metrics[f'delta_c_index_ethnicity_{group}_ci'][1]:.2f})"
            )
            minitable["hazard_ratio"].append(
                f"{metrics[f'density_adjusted_ethnicity_{group}_hr']:.2f} ({metrics[f'density_adjusted_ethnicity_{group}_hr_ci'][0]:.2f}-{metrics[f'density_adjusted_ethnicity_{group}_hr_ci'][1]:.2f})"
            )
            minitable["odds_ratio"].append(
                f"{metrics[f'density_adjusted_ethnicity_{group}_or']:.2f} ({metrics[f'density_adjusted_ethnicity_{group}_or_ci'][0]:.2f}-{metrics[f'density_adjusted_ethnicity_{group}_or_ci'][1]:.2f})"
            )

    minitable = pd.DataFrame(minitable)
    with pd.ExcelWriter(f"mgh_output_{output_suffix}.xlsx", mode="w") as writer:
        minitable.to_excel(writer, sheet_name="MGH", index=False)

    # ----------------------------------------------------------------
    # Combined data (age-adjusted and age+site-adjusted)
    minitable = defaultdict(list)
    mgh_rad_data = pd.read_csv(
        mgh_rad_data_file,
        dtype={"Patient ID": str, "Exam ID": str, "Density": int, "Age at Exam": float},
    )
    mgh_rad_data.loc[:,"SITE"] = "MGH"
    mgh_rad_data.loc[:,"Ethnicity"] = mgh_rad_data["Ethnicity"].str.lower()
    for col in [
        "Exam Date",
        "Date of Last Negative Mammogram",
        "Date of Cancer Diagnosis",
    ]:
        mgh_rad_data[col] = pd.to_datetime(mgh_rad_data[col])
    mgh_rad_data = maybe_first_exam(mgh_rad_data)

    other_sites_data = pd.read_csv(
        other_sites_data_file,
        dtype={"Patient ID": str, "Exam ID": str, "Density": int, "Age at Exam": float},
    )
    for col in [
        "Exam Date",
        "Date of Last Negative Mammogram",
        "Date of Cancer Diagnosis",
    ]:
        other_sites_data[ col] = pd.to_datetime(other_sites_data[col], format = 'ISO8601')
    other_sites_data = maybe_first_exam(other_sites_data)

    aggregate_data = pd.concat([mgh_rad_data, other_sites_data])

    def append_rows(minitable, metrics, label_prefix, max_followup, ethnicity_groups):
        minitable["group"].append(f"{label_prefix} - Overall")
        minitable["max_followup"].append(max_followup + 1)
        minitable["full_c_index"].append(
            f"{metrics['full_model_cindex']:.2f} ({metrics['full_model_cindex_ci'][0]:.2f}-{metrics['full_model_cindex_ci'][1]:.2f})"
        )
        minitable["age_only_c_index"].append(
            f"{metrics['age_only_model_cindex']:.2f} ({metrics['age_only_model_cindex_ci'][0]:.2f}-{metrics['age_only_model_cindex_ci'][1]:.2f})"
        )
        minitable["delta_c_index"].append(
            f"{metrics['delta_c_index']:.2f} ({metrics['delta_c_index_ci'][0]:.2f}-{metrics['delta_c_index_ci'][1]:.2f})"
        )
        minitable["hazard_ratio"].append(
            f"{metrics['density_adjusted_hr']:.2f} ({metrics['density_adjusted_hr_ci'][0]:.2f}-{metrics['density_adjusted_hr_ci'][1]:.2f})"
        )
        minitable["odds_ratio"].append(
            f"{metrics['density_adjusted_or']:.2f} ({metrics['density_adjusted_or_ci'][0]:.2f}-{metrics['density_adjusted_or_ci'][1]:.2f})"
        )
        for group in ["30-40", "40-50", "50-60", "60-70", "70-80"]:
            if f"full_model_age_{group}_cindex" not in metrics:
                continue
            minitable["group"].append(f"{label_prefix} - Age: {group}")
            minitable["max_followup"].append(max_followup + 1)
            minitable["full_c_index"].append(
                f"{metrics[f'full_model_age_{group}_cindex']:.2f} ({metrics[f'full_model_{group}_cindex_ci'][0]:.2f}-{metrics[f'full_model_{group}_cindex_ci'][1]:.2f})"
            )
            minitable["age_only_c_index"].append(
                f"{metrics[f'age_only_model_age_{group}_cindex']:.2f} ({metrics[f'age_only_model_{group}_cindex_ci'][0]:.2f}-{metrics[f'age_only_model_{group}_cindex_ci'][1]:.2f})"
            )
            minitable["delta_c_index"].append(
                f"{metrics[f'delta_c_index_age_{group}']:.2f} ({metrics[f'delta_c_index_{group}_ci'][0]:.2f}-{metrics[f'delta_c_index_{group}_ci'][1]:.2f})"
            )
            minitable["hazard_ratio"].append(
                f"{metrics[f'density_adjusted_age_{group}_hr']:.2f} ({metrics[f'density_adjusted_age_{group}_hr_ci'][0]:.2f}-{metrics[f'density_adjusted_age_{group}_hr_ci'][1]:.2f})"
            )
            minitable["odds_ratio"].append(
                f"{metrics[f'density_adjusted_age_{group}_or']:.2f} ({metrics[f'density_adjusted_age_{group}_or_ci'][0]:.2f}-{metrics[f'density_adjusted_age_{group}_or_ci'][1]:.2f})"
            )
        for group in ethnicity_groups:
            if group in ["Ninguno", "Otro", "Amerindio", "Gitano (Rom)"]:
                continue
            if f"full_model_ethnicity_{group}_cindex" not in metrics:
                continue
            minitable["group"].append(f"{label_prefix} - Ethnicity: {group}")
            minitable["max_followup"].append(max_followup + 1)
            minitable["full_c_index"].append(
                f"{metrics[f'full_model_ethnicity_{group}_cindex']:.2f} ({metrics[f'full_model_ethnicity_{group}_cindex_ci'][0]:.2f}-{metrics[f'full_model_ethnicity_{group}_cindex_ci'][1]:.2f})"
            )
            minitable["age_only_c_index"].append(
                f"{metrics[f'age_only_model_ethnicity_{group}_cindex']:.2f} ({metrics[f'age_only_model_ethnicity_{group}_cindex_ci'][0]:.2f}-{metrics[f'age_only_model_ethnicity_{group}_cindex_ci'][1]:.2f})"
            )
            minitable["delta_c_index"].append(
                f"{metrics[f'delta_c_index_ethnicity_{group}']:.2f} ({metrics[f'delta_c_index_ethnicity_{group}_ci'][0]:.2f}-{metrics[f'delta_c_index_ethnicity_{group}_ci'][1]:.2f})"
            )
            minitable["hazard_ratio"].append(
                f"{metrics[f'density_adjusted_ethnicity_{group}_hr']:.2f} ({metrics[f'density_adjusted_ethnicity_{group}_hr_ci'][0]:.2f}-{metrics[f'density_adjusted_ethnicity_{group}_hr_ci'][1]:.2f})"
            )
            minitable["odds_ratio"].append(
                f"{metrics[f'density_adjusted_ethnicity_{group}_or']:.2f} ({metrics[f'density_adjusted_ethnicity_{group}_or_ci'][0]:.2f}-{metrics[f'density_adjusted_ethnicity_{group}_or_ci'][1]:.2f})"
            )

    for max_followup in [0, 4]:
        (
            density,
            cancer_label,
            censor_time,
            patient_ids,
            patient_age,
            ethnicity,
            site,
        ) = format_data(aggregate_data, max_followup)
        ethnicity_groups = aggregate_data["Ethnicity"].unique()
        
        # Age-adjusted only
        print("AGGREGATE DATA: AGE ONLY")
        metrics_age = calculate_age_adjusted_metrics(
            density,
            cancer_label,
            censor_time,
            patient_age,
            patient_ids,
            ethnicity,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
        )
        append_rows(
            minitable,
            metrics_age,
            "Combined density (age-adjusted)",
            max_followup,
            ethnicity_groups,
        )
        
        # Age + site-adjusted
        print("AGGREGATE DATA: AGE + SITE")
        metrics_site = calculate_age_adjusted_metrics(
            density,
            cancer_label,
            censor_time,
            patient_age,
            patient_ids,
            ethnicity,
            n_bootstrap=n_bootstrap,
            confidence_level=confidence_level,
            site=site,
        )
        append_rows(
            minitable,
            metrics_site,
            "Combined density (age + site-adjusted)",
            max_followup,
            ethnicity_groups,
        )

    minitable = pd.DataFrame(minitable)
    with pd.ExcelWriter(f"combined_output_{output_suffix}.xlsx", mode="w") as writer:
        minitable.to_excel(writer, sheet_name="Aggregate", index=False)
