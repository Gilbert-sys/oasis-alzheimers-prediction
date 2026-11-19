import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

from src.load import load_raw, build_risk_set
from src.features import encode, fit_preprocess, apply_preprocess, FEATURES

N_SPLITS, N_REPEATS = 5, 20
C_LOGIT = 1.0       # fixed
PENALIZER = 0.1     # fixed

def run_cv(data):
    rskf = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=0)
    rows = []
    for i, (tr, te) in enumerate(rskf.split(data, data["event"])):
        train, test = data.iloc[tr], data.iloc[te]
        imp, scal = fit_preprocess(train)
        train_p = apply_preprocess(train, imp, scal)
        test_p = apply_preprocess(test, imp, scal)
        logit = LogisticRegression(C=C_LOGIT, max_iter=1000)    
        logit.fit(train_p[FEATURES], train_p["event"])
        auc = roc_auc_score(test_p["event"], logit.predict_proba(test_p[FEATURES])[:, 1])
 
        cph = CoxPHFitter(penalizer=PENALIZER)
        cph.fit(train_p, duration_col="time", event_col="event")
        risk_score = cph.predict_partial_hazard(test_p[FEATURES])
        cindex = concordance_index(test_p["time"], -risk_score, test_p["event"])
 
        rows.append({"repeat": i // N_SPLITS, "auc": auc, "cindex": cindex})
    return pd.DataFrame(rows)
 
def summarise(res):
    per_repeat = res.groupby("repeat")[["auc", "cindex"]].mean()
    return per_repeat.agg(["mean", "std"]).round(3)
 
def fit_final(data):
    imp, scal = fit_preprocess(data)
    full = apply_preprocess(data, imp, scal)
    cph = CoxPHFitter(penalizer=PENALIZER).fit(full, duration_col="time", event_col="event")
    return cph.summary[["exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]].round(3)
 
if __name__ == "__main__":
    data = encode(build_risk_set(load_raw()))
    res = run_cv(data)
    print(summarise(res))
    print(fit_final(data))