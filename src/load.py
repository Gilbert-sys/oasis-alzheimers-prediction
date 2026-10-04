from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
RAW  = ROOT / "data" / "raw" / "oasis_longitudinal_demographics-8d83e569fa2e2d30.xlsx"

def load_raw() -> pd.DataFrame:
    df = pd.read_excel(RAW) 
    df = df.sort_values(["Subject ID", "Visit"])
    assert df.shape[0] == 373
    assert df["Subject ID"].nunique() == 150
    df = df.sort_values(["Subject ID", "Visit"]).reset_index(drop=True)
    return df

def build_risk_set(df: pd.DataFrame) -> pd.DataFrame:
    baseline = df.drop_duplicates(subset="Subject ID", keep ="first")
    assert len(baseline) == 150

    baseline = baseline[baseline["CDR"] == 0]
    print(len(baseline))
    at_risk = baseline["Subject ID"]

    def outcome(g):
        impaired = g[g["CDR"] >= 0.5]
        sustained = g["CDR"].iloc[-1] >= 0.5
        if len(impaired) > 0 and sustained:
            event = 1
            time = impaired["MR Delay"].iloc[0]
        else:
            event = 0
            time = g["MR Delay"].max()
        
        return pd.Series({"event": event, "time": time})

    outcomes = (
    df[df["Subject ID"].isin(at_risk)]
    .groupby("Subject ID")
    .apply(outcome)
    .reset_index()
)

    risk = baseline.merge(outcomes, on="Subject ID")
    risk = risk[["Subject ID","M/F","Age","EDUC","SES","MMSE","eTIV","nWBV","Group","event","time"]]

    assert len(risk) == 85
    assert risk["event"].sum() == 13
    assert (risk["time"] > 0).all()
    assert ((risk["Group"] == "Converted") == (risk["event"] == 1)).all()

    return risk

if __name__ == "__main__":
    risk = build_risk_set(load_raw())
    print(risk.shape, risk["event"].sum())