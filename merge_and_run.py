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
from evaluate_density import *

parser = ArgumentParser()
parser.add_argument(
    "--input_file", "-i", type=str, required=True, help="Path to the input CSV file."
)

age_groups = ["30-40", "40-50", "50-60", "60-70", "70-80"]
age_cutoffs = [40, 50, 60, 70]
density_cutoff = 3
max_followup_range = range(0, 10)


def include_exam_and_determine_label(censor_times, golds, followup):
    include = []
    labels = []
    times = []
    for gold, censor_time in zip(golds, censor_times):
        valid_pos = gold and censor_time <= followup 
        valid_neg = censor_time >= followup
        included, label = (valid_pos or valid_neg), valid_pos
        include.append(included)
        labels.append(label)
        times.append( min(censor_time, followup) if valid_neg else followup)
    return include, labels, times


if __name__ == "__main__":
    args = parser.parse_args()

    merged_data_unidentified = pickle.load(open("merged_data_unidentified.p", "rb"))


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


    statistics = {
        "Dates": (data['Exam Date'].min(), data['Exam Date'].max()),
        "Num exams": len(data['Exam ID'].unique),
        "Num patients": len(data['Patient ID'].unique),
        "Num cancer": sum(data['Cancer (YES | NO)']),
        "Num cancer pids": data.groupby('Patient ID')['Cancer (YES | NO)'].max().sum(),
    }



    format_and_deidentify = unidentify_data(data, args.seed)

    # combine the data
    for k,v in format_and_deidentify.items():
        if isinstance(v, list):
            merged_data_unidentified[k] = merged_data_unidentified[k] + v
        elif isinstance(v, np.ndarray):
            merged_data_unidentified[k] = np.concatenate([merged_data_unidentified[k], v])
        else:
            raise ValueError(f"Data type not supported: {type(v)}")

    
    for max_followup in max_followup_range:
        for data, site in [(merged_data_unidentified, "Merged"), (format_and_deidentify, "MGH")]:
            
            density, cancer_label, censor_time, age, ethnicity = data["density"], data["cancer_label"], data["censor_time"], data["ages"], data["ethnicity"]
            valid_row, labels, times = include_exam_and_determine_label(censor_time, cancer_label, max_followup)

            df = pd.DataFrame({
                "Density": density,
                "Cancer": labels,
                "Censor Time": times,
                "Age at Exam": age,
                "Ethnicity": ethnicity
            })[valid_row]

            metrics = calculate_metrics(
                df["Density"],
                df["Cancer"],
                df["Censor Time"],
                density_cutoff,
            )

            results["Group"].append(f"{site}-All")
            results["Censoring Year"].append(max_followup)
            results["Relative Risk"].append(metrics.relative_risk)
            results["Odds Ratio"].append(metrics.odds_ratio)
            results["C-Index"].append(metrics.c_index)
            results["Followup"].append(max_followup)

            # evaluate per ethnicity
            for group in set(ethnicity):
                group_data = df[df["Ethnicity"] == group]
                metrics = calculate_metrics(
                    group_data["Density"],
                    group_data["Cancer"],
                    group_data["Censor Time"],
                    density_cutoff,
                )
                results["Group"].append(f"{site}-{group}")
                results["Censoring Year"].append(max_followup)
                results["Relative Risk"].append(metrics.relative_risk)
                results["Odds Ratio"].append(metrics.odds_ratio)
                results["C-Index"].append(metrics.c_index)
                results["Followup"].append(max_followup)

            # evaluate on different age groups
            for group in args.age_groups:
                age_lower, age_upper = group.split("-")
                age_lower, age_upper = int(age_lower), int(age_upper)
                group_data = df[
                    (df["Age at Exam"] >= age_lower) & (df["Age at Exam"] < age_upper)
                ]
                if len(group_data) == 0:
                    continue
                
                metrics = calculate_metrics(
                    group_data["Density"],
                    group_data["Cancer"],
                    group_data["Censor Time"],
                    args.density_cutoff,
                )
                results["Group"].append(f"{site}-Age: " + str(age_lower) + "-" + str(age_upper))
                results["Censoring Year"].append(max_followup)
                results["Relative Risk"].append(metrics.relative_risk)
                results["Odds Ratio"].append(metrics.odds_ratio)
                results["C-Index"].append(metrics.c_index)

            # use age as a predictor
            age_based_metrics = calculate_age_metrics(
                df["Age at Exam"],
                df["Cancer"],
                df["Censor Time"],
                args.age_cutoffs,
            )
            for i, age in enumerate(args.age_cutoffs):
                results["Group"].append(f"{site}-Age as Predictor: " + str(age))
                results["Censoring Year"].append(max_followup)
                results["Relative Risk"].append(age_based_metrics[i].relative_risk)
                results["Odds Ratio"].append(age_based_metrics[i].odds_ratio)
                results["C-Index"].append(age_based_metrics[i].c_index)
                results["Followup"].append(max_followup)

            # use all ages as a predictor - increasing age is a risk factor
            censoring_dist = get_censoring_dist(df["Censor Time"], df["Cancer"])  
            ages = np.array(df["Age at Exam"])
            normalized_age = (ages - ages.min()) / (ages.max() - ages.min())
            c_index = concordance_index(
                event_times=df["Censor Time"],
                predicted_scores=normalized_age,
                event_observed=df["Cancer"],
                censoring_dist=censoring_dist,
            )
            results["Group"].append(f"{site}-Age as Predictor: All ages")
            results["Censoring Year"].append(max_followup)
            results["Relative Risk"].append("-")
            results["Odds Ratio"].append("-")
            results["C-Index"].append(c_index)
            results["Followup"].append(max_followup)

    
    # save the results
    results_path = "mgh_and_merged_results.p"
    pickle.dump((results,statistics), open(results_path, "wb"))
    

    # done
    print(f"""
        Results saved to: {results_path}
        """)
