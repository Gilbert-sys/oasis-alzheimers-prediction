OASIS-2: Predicting conversion to dementia from baseline data

Status: work in progress. Risk set, labelling and preprocessing are complete. Cross-validation and write-up are in progress; sections marked TODO will be filled in as results come in.

Question

Among older adults who are cognitively normal at their first visit, can baseline demographics, MRI volumetrics and a cognitive screening score predict who goes on to develop dementia, and how soon?

This is a forecasting problem, so the model only sees information recorded at the first visit.

Data

OASIS-2: a longitudinal MRI and clinical dataset of 150 adults aged 60 to 96, with 373 sessions in total and at least two visits per subject, a year or more apart. The raw data is not included in this repository (OASIS data use terms). To reproduce, place the file in data/raw/ and update the path in src/load.py.

Population and outcome
At-risk population: subjects with CDR = 0 at their first visit. This gives 85 subjects: 72 nondemented, plus 13 of the 14 subjects labelled "Converted". The 64 subjects who are already impaired at baseline are excluded, because the outcome has already happened for them.
Event: sustained conversion, meaning CDR >= 0.5 at some visit and still CDR >= 0.5 at the last visit. This gives 13 events.
Event time: days from baseline (MR Delay) to the first visit with CDR >= 0.5. True onset lies somewhere between the previous visit and this one, so event times are interval-censored and the detection visit is used.
Censoring: non-converters are censored at their last visit. Follow-up ranges from about 1 to 7 years, which is why a time-to-event model is used rather than a binary label.

Notes on label decisions:

One "Converted" subject already had CDR 0.5 at their first visit, so they were impaired at entry and are excluded by the baseline rule.
Two subjects labelled Nondemented (OAS2_0005, OAS2_0017) have a single CDR 0.5 visit that reverts to 0. These are treated as non-events, consistent with the dataset's own labels.
Features (fixed in advance)

Age, sex, years of education, SES, eTIV, nWBV, and baseline MMSE. CDR defines the population and the outcome, so it is not a predictor. ASF is a rescaling of eTIV and is omitted. Imputation (median) and standardisation are fitted inside the training folds only.

Methods
Penalised Cox proportional hazards (primary): hazard ratios with 95% intervals.
Penalised logistic regression on "ever converts", as a binary comparison.
XGBoost with fixed, modest hyperparameters, as a flexible comparison (planned).
Evaluation: repeated stratified 5-fold cross-validation (20 repeats), reporting the mean and the spread across repeats. Metrics are C-index (Cox) and AUC (logistic, XGBoost).

With 13 events and 7 features there are roughly two events per variable, well below the usual guideline of ten, so penalisation is used and results are reported with their uncertainty.

Results

Repeated stratified 5-fold CV (20 repeats), mean and SD across repeats:

| Model | Metric | Mean | SD |
|---|---|---|---|
| Penalised logistic | AUC | 0.546 | 0.058 |
| Penalised Cox | C-index | 0.516 | 0.084 |

Cox hazard ratios (per 1 SD, 95% interval) on all 85 subjects: every interval includes 1 (e.g. nWBV 0.90, 0.55 to 1.47; MMSE 0.98, 0.62 to 1.54).

**Finding:** with 13 events, none of the seven baseline features separated converters from non-converters better than chance. This is a limit of the sample size as much as of the features, and the data cannot rule out modest effects.

Limitations
Only 13 events, so intervals are wide and model comparisons are weakly powered.
Single-site, single-cohort data, with no external validation.
Event times are approximate (detection visit, not true onset), and CDR is a noisy label.
Proportional hazards cannot be tested reliably at this sample size.
Baseline MMSE is measured before the outcome, so it is legitimate for forecasting, but it is likely the strongest predictor and should be read with that in mind.
The SD across repeats reflects variability from how the data is split, not sampling uncertainty. All repeats use the same 85 subjects, so the true uncertainty about out-of-sample performance is larger than the reported SD.
Intervals come from a penalised Cox fit, so they are approximate.
This is a research and learning project, not a clinical tool.