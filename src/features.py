import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Fixed in advance. Do not change after seeing results
FEATURES = ["Age", "M/F", "EDUC", "SES", "eTIV", "nWBV", "MMSE"]

def encode(risk):
    data = risk.copy()
    data["M/F"] = (data["M/F"] == "M").astype(int) # male is 1 female is 0
    return data[FEATURES + ["event","time"]]

def fit_preprocess(train):
    imp = SimpleImputer(strategy="median")
    filled = imp.fit_transform(train[FEATURES])
    scal = StandardScaler().fit(filled)
    return imp, scal

def apply_preprocess(df, imp, scal):
    X = scal.transform(imp.transform(df[FEATURES]))
    out = pd.DataFrame(X, columns=FEATURES, index=df.index)
    out["event"] = df["event"]
    out["time"] = df["time"]
    return out
