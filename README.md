# oasis-alzheimers-prediction
The project: predicting cognitive decline progression in Alzheimer's patients using longitudinal clinical and MRI data. Start with tabular clinical data only (the OASIS-2 dataset). Raw MRI is too complex for a first project.
Is this a good timeline?
Week 1	Register OASIS, read 1–2 papers on ML for Alzheimer's prediction, set up GitHub repo and Python environment
Week 2	Load OASIS-2 dataset, exploratory data analysis — distributions, missingness, class balance. Commit EDA notebook.
Week 3	Handle missing data, define target variable (stable vs progressive MCI at 24 months), select initial features
Week 4	Baseline logistic regression model, proper train/test split respecting patient IDs (no data leakage), evaluate with AUC
Week 5	XGBoost or random forest, compare to baseline, add SHAP values for interpretability
Week 6	Write-up: clean README, SHAP visualisation, reflection on limitations and next steps
