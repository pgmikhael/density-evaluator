from typing import NamedTuple
import os
import numpy as np
from argparse import ArgumentParser
from scipy.stats.contingency import crosstab, relative_risk, odds_ratio
from sksurv.metrics import concordance_index_ipcw
import pandas as pd
from collections import defaultdict
import datetime
from c_index_util import get_censoring_dist, concordance_index
import hashlib
import random
import pickle

class Metrics(NamedTuple):
    relative_risk: float
    odds_ratio: float
    c_index: float


def calculate_metrics(density, cancer_label, censor_time, density_cutoff=3) -> Metrics:
    """
    Calculate the relative risk, odds ratio of the model.

    Parameters
    ----------
    predictions : list
        Binary predictions of risk
    labels : list
        True cancer labels
    censor_time : list
        Time of censoring
    density_cutoff : int
        The cutoff for what's considered dense. Default is 3-4.

    Returns
    -------
    Metrics
        A named tuple with the relative risk, odds ratio
        - relative_risk : float
            The relative risk of the model
        - odds_ratio : float
            The odds ratio of the model; maximum likelihood estimates of the odds ratios
    """
    binary_density = np.int16(density >= density_cutoff)
    contingency_table = crosstab(binary_density, cancer_label)
    contingency_table_dict = {}
    dense_values, cancer_values = contingency_table.elements
    for i, dense in enumerate(dense_values):
        dense_key = "dense" if dense == 1 else "not_dense"
        for j, cancer in enumerate(cancer_values):
            cancer_key = "cancer" if cancer == 1 else "no_cancer"
            contingency_table_dict[f"{dense_key}_{cancer_key}"] = (
                contingency_table.count[i][j]
            )

    rr = relative_risk(
        exposed_cases=contingency_table_dict["dense_cancer"],
        exposed_total=contingency_table_dict["dense_cancer"]
        + contingency_table_dict["dense_no_cancer"],
        control_cases=contingency_table_dict["not_dense_cancer"],
        control_total=contingency_table_dict["not_dense_cancer"]
        + contingency_table_dict["not_dense_no_cancer"],
    )
    table = np.array(
        [
            [
                contingency_table_dict["dense_cancer"],
                contingency_table_dict["dense_no_cancer"],
            ],
            [
                contingency_table_dict["not_dense_cancer"],
                contingency_table_dict["not_dense_no_cancer"],
            ],
        ]
    )
    oratio = odds_ratio(table)

    # calculate c-index
    censoring_dist = get_censoring_dist(
        censor_time, cancer_label
    )  # NOTE: computed on test set
    normalized_density = (density - density.min()) / (density.max() - density.min())
    c_index = concordance_index(
        event_times=censor_time,
        predicted_scores=normalized_density,
        event_observed=cancer_label,
        censoring_dist=censoring_dist,
    )

    return Metrics(rr.relative_risk, oratio.statistic, c_index)


def calculate_age_metrics(
    ages, cancer_label, censor_time, age_cutoffs=[50, 60, 70]
) -> Metrics:
    """
    Calculate the relative risk, odds ratio of the model.

    Parameters
    ----------
    ages : list
        Age of the patients
    labels : list
        True cancer labels
    censor_time : list
        Time of censoring
    age_cutoffs : int
        The cutoff for what's considered positive.

    Returns
    -------
    Metrics
        A named tuple with the relative risk, odds ratio
        - relative_risk : float
            The relative risk of the model
        - odds_ratio : float
            The odds ratio of the model; maximum likelihood estimates of the odds ratios
    """
    results = []
    for age in age_cutoffs:
        binary_ages = np.int16(ages >= age)
        contingency_table = crosstab(binary_ages, cancer_label)
        contingency_table_dict = {}
        age_values, cancer_values = contingency_table.elements
        for i, binary_age in enumerate(age_values):
            age_key = "old" if binary_age == 1 else "young"
            for j, cancer in enumerate(cancer_values):
                cancer_key = "cancer" if cancer == 1 else "no_cancer"
                contingency_table_dict[f"{age_key}_{cancer_key}"] = (
                    contingency_table.count[i][j]
                )

        rr = relative_risk(
            exposed_cases=contingency_table_dict["old_cancer"],
            exposed_total=contingency_table_dict["old_cancer"]
            + contingency_table_dict["old_no_cancer"],
            control_cases=contingency_table_dict["young_cancer"],
            control_total=contingency_table_dict["young_cancer"]
            + contingency_table_dict["young_no_cancer"],
        )
        table = np.array(
            [
                [
                    contingency_table_dict["old_cancer"],
                    contingency_table_dict["old_no_cancer"],
                ],
                [
                    contingency_table_dict["young_cancer"],
                    contingency_table_dict["young_no_cancer"],
                ],
            ]
        )
        oratio = odds_ratio(table)

        # calculate c-index
        censoring_dist = get_censoring_dist(
            censor_time, cancer_label
        )  # NOTE: computed on test set
        c_index = concordance_index(
            event_times=censor_time,
            predicted_scores=binary_ages,
            event_observed=cancer_label,
            censoring_dist=censoring_dist,
        )
        results.append(Metrics(rr.relative_risk, oratio.statistic, c_index))

    return results


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
    density = np.array(data["Density"])
    date = data["Exam Date"]
    ever_has_cancer = data["Cancer (YES | NO)"]
    last_negative_date = data["Date of Last Negative Mammogram"]
    cancer_date = data["Date of Cancer Diagnosis"]
    # compute the time of censoring relative to exam date
    years_to_last_negative = np.minimum(
        (last_negative_date - date).dt.days // 365, max_followup
    )

    # compute the time of cancer diagnosis relative to exam date or max_followup
    years_to_cancer = ((cancer_date - date).dt.days // 365).copy()

    # positives are those who developed cancer within the follow-up period
    cancer_label = (years_to_cancer <= max_followup) & ever_has_cancer

    # valid rows
    valid_rows = (years_to_last_negative >= max_followup) | cancer_label & (years_to_cancer >= 0)

    # construct censor_time array
    censor_time = np.where(cancer_label, years_to_cancer, years_to_last_negative)

    # remove invalid rows
    density = density[valid_rows]
    cancer_label = np.int32(cancer_label[valid_rows])
    censor_time = np.int32(censor_time[valid_rows])
    ages = ages[valid_rows]

    return density, cancer_label, censor_time, ages

    
    
def hash(data, seed):
    """
    Hash the data with a random seed.

    Parameters
    ----------
    data : str
        The input data
    seed : int
        random seed

    Returns
    -------
    str
        A hashed string with random seed
    """
    random.seed(seed)
    randnum = random.randint(0, 2**32 - 1)
    hasher = hashlib.sha256()
    hasher.update(data.encode())
    hasher.update(randnum.to_bytes(4, 'big'))
    hashed_data = hasher.hexdigest()
    return hashed_data

def unidentify_data(data: pd.DataFrame, seed: int) -> tuple:
    """
    Prepare the data for the evaluation.

    Parameters
    ----------
    data : pd.DataFrame
        The input data
    seed : int
        random seed

    Returns
    -------
    dict
        A dictionary of data with hashed Patient ID and Exam ID
    """
    # set cancer date to 100 if no cancer
    data.loc[data["Cancer (YES | NO)"] == 0, "Date of Cancer Diagnosis"] = data.loc[
        data["Cancer (YES | NO)"] == 0, "Exam Date"
    ] + datetime.timedelta(days=int(100 * 365))

    ages = data["Age at Exam"]
    density = np.array(data["Density"])
    date = data["Exam Date"]
    ever_has_cancer = data["Cancer (YES | NO)"]
    last_negative_date = data["Date of Last Negative Mammogram"]
    cancer_date = data["Date of Cancer Diagnosis"]
    # compute the time of censoring relative to exam date
    years_to_last_negative = (last_negative_date - date).dt.days // 365

    # compute the time of cancer diagnosis relative to exam date or max_followup
    years_to_cancer = ((cancer_date - date).dt.days // 365).copy()

    # positives are those who developed cancer within the follow-up period
    cancer_label = ever_has_cancer

    censor_time = np.where(cancer_label, years_to_cancer, years_to_last_negative)

    valid_rows = (years_to_last_negative >= 0) | (cancer_label.astype(bool) & (years_to_cancer >= 0))

    processed_data = {
        "density": density[valid_rows],
        "cancer_label": cancer_label[valid_rows],
        "censor_time": censor_time[valid_rows],
        "ages": ages[valid_rows],
        "ethnicity": list(data["Ethnicity"][valid_rows]),
    }

    pid_map = {}
    data["Patient ID"] = data["Patient ID"].astype(str)
    for pid in data["Patient ID"].unique():
        new_pid = hash(pid, seed)
        pid_map[pid] = new_pid
    
    eid_map = {}
    data["Exam ID"] = data["Exam ID"].astype(str)
    for eid in data["Exam ID"].unique():
        new_eid = hash(eid, seed)
        eid_map[eid] = new_eid
    
    data["Patient ID"] = data["Patient ID"].map(pid_map)
    data["Exam ID"] = data["Exam ID"].map(eid_map)
    processed_data["pid"] = list(data["Patient ID"][valid_rows])
    processed_data["eid"] = list(data["Exam ID"][valid_rows])
    return processed_data
    

def prepare_data(data: pd.DataFrame, args: ArgumentParser) -> tuple:
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
        A tuple of density, cancer_label, censor_time
    """
    density, cancer_label, censor_time, ages = format_data(data, args.max_followup)
    return density, cancer_label, censor_time, ages


# input density, age, ethnicity

parser = ArgumentParser()
parser.add_argument(
    "--input_file", "-i", type=str, required=True, help="Path to the input CSV file."
)
parser.add_argument(
    "--max_followup",
    "-d",
    type=int,
    default=5,
    required=True,
    help="Truncation time in years. Starts at 0",
)
parser.add_argument(
    "--density_cutoff", type=int, default=3, help="Cutoff for density. Default is 3-4."
)
parser.add_argument(
    "--age_groups",
    type=str,
    default=[],
    nargs="*",
    help="Age groups to evaluate. Format: 40-50 50-60",
)
parser.add_argument(
    "--age_cutoffs",
    type=int,
    default=[40, 50, 60],
    nargs="*",
    help="Ages to use as binary predictor.",
)
parser.add_argument(
    "--deidentify_and_save",
    action="store_true",
    default=False,
    help="Save the formatted data.",
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help= "Seed for deidentification hash.",
)

if __name__ == "__main__":
    args = parser.parse_args()

    # input variables to save
    input_variables = {}

    # collect results in a dictionary / dataframe
    results = defaultdict(list)

    # read the data
    data = pd.read_csv(args.input_file)

    # check if the columns are present
    required_columns = [
        "Patient ID",
        "Exam ID",
        "Exam Date",
        "Density",
        "Age at Exam",
        "Ethnicity",
        "Cancer (YES | NO)",
        "Date of Cancer Diagnosis",
        "Date of Last Negative Mammogram",
    ]
    for col in required_columns:
        if col not in data.columns:
            raise ValueError(f"Column {col} is missing in the input file.")

    # format dates
    for col in [
        "Exam Date",
        "Date of Last Negative Mammogram",
        "Date of Cancer Diagnosis",
    ]:
        data[col] = pd.to_datetime(data[col])  #'%m/%d/%Y' # , format='%Y-%m-%d'

    if args.deidentify_and_save:
        assert args.seed, "Seed must be provided to save statistics."
        format_and_deidentify = unidentify_data(data, args.seed)
        # save the data
        data_path = os.path.join(
            os.path.dirname(args.input_file),
            os.path.basename(args.input_file).split(".")[0] + "_unidentified.p",
        )
        pickle.dump(format_and_deidentify, open(data_path, "wb"))

    
    # prepare the data with correct censoring and cancer labels
    density, cancer_label, censor_time, ages = prepare_data(data, args)

    metrics = calculate_metrics(
        density,
        cancer_label,
        censor_time,
        args.density_cutoff,
    )

    results["Group"].append("All")
    results["Censoring Year"].append(args.max_followup)
    results["Relative Risk"].append(metrics.relative_risk)
    results["Odds Ratio"].append(metrics.odds_ratio)
    results["C-Index"].append(metrics.c_index)

    # evaluate per ethnicity
    for group in data["Ethnicity"].unique():
        group_data = data[data["Ethnicity"] == group]
        density, cancer_label, censor_time, ages = prepare_data(group_data, args)
        metrics = calculate_metrics(
            density,
            cancer_label,
            censor_time,
            args.density_cutoff,
        )
        results["Group"].append(group)
        results["Censoring Year"].append(args.max_followup)
        results["Relative Risk"].append(metrics.relative_risk)
        results["Odds Ratio"].append(metrics.odds_ratio)
        results["C-Index"].append(metrics.c_index)

    # evaluate on different age groups
    for group in args.age_groups:
        age_lower, age_upper = group.split("-")
        age_lower, age_upper = int(age_lower), int(age_upper)
        group_data = data[
            (data["Age at Exam"] >= age_lower) & (data["Age at Exam"] < age_upper)
        ]
        if len(group_data) == 0:
            continue
        density, cancer_label, censor_time, ages = prepare_data(group_data, args)
        metrics = calculate_metrics(
            density,
            cancer_label,
            censor_time,
            args.density_cutoff,
        )
        results["Group"].append("Age: " + str(age_lower) + "-" + str(age_upper))
        results["Censoring Year"].append(args.max_followup)
        results["Relative Risk"].append(metrics.relative_risk)
        results["Odds Ratio"].append(metrics.odds_ratio)
        results["C-Index"].append(metrics.c_index)
    # use age as a predictor
    density, cancer_label, censor_time, ages = prepare_data(data, args)
    age_based_metrics = calculate_age_metrics(
        ages,
        cancer_label,
        censor_time,
        args.age_cutoffs,
    )
    for i, age in enumerate(args.age_cutoffs):
        results["Group"].append("Age as Predictor: " + str(age))
        results["Censoring Year"].append(args.max_followup)
        results["Relative Risk"].append(age_based_metrics[i].relative_risk)
        results["Odds Ratio"].append(age_based_metrics[i].odds_ratio)
        results["C-Index"].append(age_based_metrics[i].c_index)

    # use all ages as a predictor - increasing age is a risk factor
    censoring_dist = get_censoring_dist(
        censor_time, cancer_label
    )  # NOTE: computed on test set
    normalized_age = (ages - ages.min()) / (ages.max() - ages.min())
    c_index = concordance_index(
        event_times=censor_time,
        predicted_scores=normalized_age,
        event_observed=cancer_label,
        censoring_dist=censoring_dist,
    )
    results["Group"].append("Age as Predictor: All ages")
    results["Censoring Year"].append(args.max_followup)
    results["Relative Risk"].append("-")
    results["Odds Ratio"].append("-")
    results["C-Index"].append(c_index)

    # save the results
    results_df = pd.DataFrame(results)
    results_path = os.path.join(
        os.path.dirname(args.input_file),
        os.path.basename(args.input_file).split(".")[0] + "_results.csv",
    )
    results_df.to_csv(results_path, index=False)

    # done
    print(f"""
        Results saved to: {results_path}
        """)
