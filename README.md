# Mammographic Density Evaluator

Script to evaluate mammographic density as a predictor of breast cancer risk. Given density measurements and censored cancer data, the script computes the relative risk, odds ratio, and c-statistic of the density measurements.

## Environment

```bash
conda env create -f environment.yml
conda activate density
```

## Data Format

An example of the data format is provided in [example_input.csv](example_input.csv). The data is a CSV file with the following columns:

- PatientID: Unique identifier for each patient
- ExamID: Unique identifier for each mammogram
- ExamDate: Date of the mammogram. The date should be in the format `MM/DD/YYYY`
- Density / BI-RADS: Density measurement 
- Age at Exam: Age of the patient at the time of the mammogram
- Ethnicity: Ethnicity of the patient
- Cancer (YES | NO): Whether the patient was diagnosed with cancer within `T`  years of the mammogram
- Date of Cancer Diagnosis: Date of cancer diagnosis. The date should be in the format `MM/DD/YYYY`
- Date of Last Negative Mammogram: Last known date of a negative mammogram. The date should be in the format `MM/DD/YYYY`

## Usage


To compute the relative risk and odds ratio:

```bash
python evaluate_density.py \
-i example_input.csv \
--max_followup 5 \
--density_cutoff 3
```

To compute the c-statistic, training data is required. The training data should be in the same format as the input data. Only the 
`Cancer (YES | NO)`, `Date of Cancer Diagnosis`, and `Date of Last Negative Mammogram` columns are required.

```bash
python evaluate_density.py \
-i example_input.csv \
--max_followup 5 \
--density_cutoff 3 \
--compute_c_index \
--input_train_file example_train_input.csv
```

To evaluate the performance of the density measurements on different age groups:

```bash
python evaluate_density.py \
-i example_input.csv \
--max_followup 5 \
--density_cutoff 3 \
--age_groups 40-50 50-60 60-70
```

## Output
The script outputs the relative risk, odds ratio, and optionally c-statistic of the density measurements. The output is saved in a CSV file with the same name as the input file, but with `_results` appended to the name. For example, if the input file is `example_input.csv`, the output file will be `example_input_results.csv`.