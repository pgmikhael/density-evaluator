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
- Density: Density measurement 
- Age at Exam: Age of the patient at the time of the mammogram
- Ethnicity: Ethnicity of the patient
- Cancer (YES | NO): Whether the patient was diagnosed with cancer within `T`  years of the mammogram
- Date of Cancer Diagnosis: Date of cancer diagnosis. The date should be in the format `MM/DD/YYYY`
- Date of Last Negative Mammogram: Last known date of a negative mammogram (i.e., date of the most recent negative mammogram). The date should be in the format `MM/DD/YYYY`

## Usage


To compute the relative risk and odds ratio:

```bash
python evaluate_density.py \
-i example_input.csv \
--max_followup 5 \
--density_cutoff 3
```

To evaluate the performance of the density measurements on different age groups:

```bash
python evaluate_density.py \
-i example_input.csv \
--max_followup 4 \ # Maximum follow-up time in years, indexed at 0
--density_cutoff 3 \
--age_groups 40-50 50-60 60-70
```

To format and save the data with deidentified IDs add the arguments `--deidentify_and_save --seed 123`. The seed is required to generate the deidentified IDs in a random manner.



## Output
The script outputs the relative risk and odds ratio of the density measurements. The output is saved in a CSV file with the same name as the input file, but with `_results` appended to the name. For example, if the input file is `example_input.csv`, the output file will be `example_input_results.csv`.


Breast Density Study Data Questions + Authorlist


Hi,

We’re in the process of drafting the manuscript analyzing the predictive power of breast density. Could you send me a detailed description of how the data we’re using was collected. Specifically, we need to include the following details: 
- Is this all the data that you have with density and outcome data at your institution or a subset? 
- If it is a subset, how was it obtained (randomly or through other method)? 
- How was cancer diagnosis defined? Is it biopsy-confirmed? Is it ductal carcinoma in situ or invasive disease, or something else?

Also, please let us know who should be included in the author list, as well as affiliations.

Thank you!

Peter




MGH: Leslie
CGMH: Gigin Lin <giginlin@gmail.com>, Yin-Tzu Lin <lin.ytzu@gmail.com>
CMU: 林碩彥 LIN SHUO-YEN <100998@tool.caaumed.org.tw>, 巫旻諺 WU MIN-YEN <083742@tool.caaumed.org.tw>, 姜秀穎 Hsiu-Yin Chiang <032031@tool.caaumed.org.tw>, 郭醫師 <fenderkuo@gmail.com>
Chile: Diego Kauer <diego.kauer@ug.uchile.cl>
UCSF: "Chung, Maggie" <Maggie.Chung@ucsf.edu>, "Chou, Shinn-Huey" <Shirley.Chou@ucsf.edu>, Adam Yala <yala@berkeley.edu>
Emory: "Satya Isaac, Rohan" <rohan.satya.isaac@emory.edu>, "Trivedi, Hari" <hari.trivedi@emory.edu>
India: "Dr. Sujoy Kar" <drsujoy_k@apollohospitals.com>, Bharath Potla <bharath_p@apollohospitals.com>, Tameem Ansari <tameem_a@apollohospitals.com>
