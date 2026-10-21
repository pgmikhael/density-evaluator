import numpy as np
import pandas as pd
import datetime
import argparse
from collections import defaultdict, Counter
from tqdm import tqdm
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
import gc


def format_data(
    data: pd.DataFrame, max_followup: int, include_age: bool = False
) -> tuple:
    """
    Prepare the data for the evaluation.

    Parameters
    ----------

        parser = argparse.ArgumentParser(
            description="Evaluate age-adjusted density metrics across sites."
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
    if include_age:
        ages = data["Age at Exam"]
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
    if include_age:
        ages = ages[valid_rows]
        return density, cancer_label, censor_time, patient_id, ages

    return density, cancer_label, censor_time, patient_id


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


def run_single_bootstrap(i, data, p_indices, u_patients, n_p):
    # Set seed based on iteration for reproducibility in parallel
    np.random.seed(i)

    # Fast clustered sampling
    sample_pids = np.random.choice(u_patients, size=n_p, replace=True)
    indices = [idx for pid in sample_pids for idx in p_indices[pid]]
    sample_df = data.iloc[indices]

    if (
        sample_df["cancer_outcome"].nunique() < 2
    ):  # need at least one positive and one negative
        return None

    try:
        c_age = (
            CoxPHFitter()
            .fit(
                sample_df,
                "time_to_event",
                "cancer_outcome",
                formula="age",
                fit_options={"max_steps": 100},
            )
            .concordance_index_
        )
        c_full = (
            CoxPHFitter()
            .fit(
                sample_df,
                "time_to_event",
                "cancer_outcome",
                formula="age + density",
                fit_options={"max_steps": 100},
            )
            .concordance_index_
        )
        return (c_age, c_full, c_full - c_age)
    except:
        return None


def run_single_bootstrap_log(i, data, p_indices, u_patients, n_p):
    # Set seed based on iteration for reproducibility in parallel
    np.random.seed(i)
    
    # Fast clustered sampling
    sample_pids = np.random.choice(u_patients, size=n_p, replace=True)
    # indices = np.array([idx for pid in sample_pids for idx in p_indices[pid]])
    # sample_df = data.iloc[indices]
    pid_counts = Counter(sample_pids)
    sample_df = data[data['pids'].isin(pid_counts)].copy()
    sample_df['weight'] = sample_df['pids'].map(pid_counts)
    w = sample_df['weight'].values
    try:
        X = sample_df[['age', 'density']].values
        X_age = sample_df[['age']].values
        y = sample_df['cancer_outcome'].values
        t = sample_df['time_to_event'].values
        
        # probs_age  = LogisticRegression().fit(X_age, y, sample_weight=w).predict_proba(X_age)[:, 1]
        # probs_full = LogisticRegression().fit(X, y, sample_weight=w).predict_proba(X)[:, 1]

        # c_age  = concordance_index(t, -probs_age,  y)
        # c_full = concordance_index(t, -probs_full, y)

        model_age  = smf.logit("cancer_outcome ~ age", data=sample_df).fit(disp=0)
        model_full = smf.logit("cancer_outcome ~ age + density", data=sample_df).fit(disp=0)

        # predict and score
        c_age  = concordance_index(sample_df['time_to_event'], -model_age.predict(sample_df),  sample_df['cancer_outcome'])
        c_full = concordance_index(sample_df['time_to_event'], -model_full.predict(sample_df), sample_df['cancer_outcome'])
                
        
        return (c_age, c_full, c_full - c_age)
    except:
        return None    
    
def calculate_age_adjusted_metrics(
    density,
    cancer_label,
    censor_time,
    ages,
    pids,
    confidence_level=0.95,
    n_bootstrap=1000,
    random_state=0,
):
    minidf = pd.DataFrame(
        {
            "density": density,
            "cancer_outcome": cancer_label.astype(int),
            "time_to_event": censor_time,
            "age": ages,
            "pids": pids,
        }
    )
    grouped_indices = minidf.groupby("pids").indices
    unique_patients = minidf["pids"].unique()
    n_patients = len(unique_patients)

    results = Parallel(n_jobs=40, backend="loky")(
        delayed(run_single_bootstrap_log)(
            i, minidf, grouped_indices, unique_patients, n_patients
        )
        for i in tqdm(range(n_bootstrap), ncols=100)
    )

    # CPH for c-index
    c_ages = [c[0] for c in results if c is not None]
    c_deltas = [c[2] for c in results if c is not None]
    c_fulls = [c[1] for c in results if c is not None]

    model_age  = smf.logit("cancer_outcome ~ age", data=minidf).fit(disp=0)
    model_full = smf.logit("cancer_outcome ~ age + density", data=minidf).fit(disp=0)

    # predict and score
    c_age  = concordance_index(minidf['time_to_event'], -model_age.predict(minidf),  minidf['cancer_outcome'])
    c_full = concordance_index(minidf['time_to_event'], -model_full.predict(minidf), minidf['cancer_outcome'])
            

    # # Cox proportional hazards model for ratio
    # cph_age = CoxPHFitter().fit(
    #     minidf, "time_to_event", "cancer_outcome", formula="age"
    # )
    # c_age = cph_age.concordance_index_

    cph_full = CoxPHFitter().fit(
        minidf, "time_to_event", "cancer_outcome", formula="age + density"
    )
    # c_full = cph_full.concordance_index_

    delta_c = c_full - c_age

    # CI for c-index
    deltas = np.array(c_deltas) - delta_c
    delta_lower, delta_upper = ci_bounds(delta_c, deltas, confidence_level, n_bootstrap)
    age_lower, age_upper = ci_bounds(
        c_age, np.array(c_ages) - c_age, confidence_level, n_bootstrap
    )
    full_lower, full_upper = ci_bounds(
        c_full, np.array(c_fulls) - c_full, confidence_level, n_bootstrap
    )

    # adjusted hazard ratio for density
    summary_full = cph_full.summary
    hr = summary_full.loc["density", "exp(coef)"]
    hr_lower = summary_full.loc["density", "exp(coef) lower 95%"]
    hr_upper = summary_full.loc["density", "exp(coef) upper 95%"]

    # Logistic regression for the 3-year binary outcome
    logit_model = smf.logit("cancer_outcome ~ age + density", data=minidf).fit(disp=0)

    results = {
        "full_model_cindex": c_full,
        "full_model_cindex_ci": (full_lower, full_upper),
        "age_only_model_cindex": c_age,
        "age_only_model_cindex_ci": (age_lower, age_upper),
        "delta_c_index": delta_c,
        "delta_c_index_ci": (delta_lower, delta_upper),
        "density_adjusted_hr": hr,
        "density_adjusted_or": np.exp(logit_model.params["density"]),
        "density_adjusted_hr_ci": (hr_lower, hr_upper),
        "density_adjusted_or_ci": logit_model.conf_int()
        .loc["density"]
        .apply(np.exp)
        .tolist(),
    }

    del minidf
    gc.collect()

    return results



    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate age-adjusted density metrics across sites."
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


    data1 = pd.read_csv("data/chile/formatted_complete_DH_data.csv")
    data2 = pd.read_csv(
        "data/UCSF/formatted_anonymized_risk_model_preds_w_outcomes_jan_5_2025.csv"
    )
    data3 = pd.read_excel("data/taiwan-cmu/input_ori_20250808.xlsx")
    data4 = pd.read_csv("data/taiwan-cgmh/formatted_minus_asymptomatic_08122025.csv")
    data5 = pd.read_csv("data/apollo/formatted_data.csv")
    data6 = pd.read_csv("data/emory/input_file_one_outcome.csv")

    data_list = [
        (data1, "Chile"),
        (data2, "UCSF"),
        (data3, "Taiwan-CMU"),
        (data4, "Taiwan-CGMH"),
        (data5, "Apollo"),
        (data6, "Emory"),
    ]
    for data, site in data_list:
        for col in [
            "Exam Date",
            "Date of Last Negative Mammogram",
            "Date of Cancer Diagnosis",
        ]:
            data[col] = pd.to_datetime(data[col])

    if args.first_exam_only:
        data_list_ = []
        # Remove repeat patients, keep only first exam
        for data, site in data_list:
            # 1. Sort by ID and Date (Earliest date first)
            df = data.sort_values(["Patient ID", "Exam Date"], ascending=[True, True])

            # 2. Keep the first occurrence of each ID
            df_first = df.drop_duplicates(subset="Patient ID", keep="first")

            data_list_.append((df_first, site))

        data_list = data_list_

    minitable = defaultdict(list)
    for max_followup in [0, 4]:
        print(f"{max_followup + 1}-year cancer")
        for data, site in tqdm(data_list, ncols=100):
            density, cancer_label, censor_time, patient_ids, patient_age = format_data(
                data, max_followup, include_age=True
            )
            metrics = calculate_age_adjusted_metrics(
                density,
                cancer_label,
                censor_time,
                patient_age,
                patient_ids,
                n_bootstrap=1000,
            )
            minitable["site"].append(site)
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

    output_suffix = "first_exam_only" if args.first_exam_only else "all_exams"
    pd.DataFrame(minitable).to_csv(
        f"results/age_adjusted_metrics_by_site_{output_suffix}.csv", index=False
    )
