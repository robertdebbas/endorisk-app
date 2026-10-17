"""
Prediction interface used by the app.  The app only calls load_model() and predict().

Main model : ridge logistic regression (S2 features) — see train_arms_model.py.
Control    : recalibrated random forest, returned under 'control' but not shown by default.

Study-context variables (region, design, unit) are averaged over their observed joint
distribution in the training arms (direct standardization, one arm = one context, equal
weight). The only time variable is the era, held at '2011+' (a case seen today cannot belong
to an earlier era). There is no linear calendar term.

case = dict(kinematics, alloy, operator, uses_teeth, tooth_group)
"""
from pathlib import Path
import joblib, numpy as np

MODEL_PATH = Path(__file__).parent / "models" / "arms_model.joblib"
KINEMATICS = ["Rotary", "Reciprocating"]
ALLOYS = ["Conventional NiTi", "M-Wire", "Heat-treated (CM/Blue)"]
OPERATORS = ["Endodontist", "Graduate student", "Undergraduate student", "General practitioner"]
TOOTH_GROUPS = ["Molars", "Posterior", "All", "Other/NR"]
TODAY_ERA = "2011+"

def use_policy_from_uses(k: int) -> str:
    return "Single tooth" if k <= 1 else ("Up to 4 teeth" if k <= 4 else "More than 4 teeth")

EXPECTED_NUMERIC = ["max_uses_teeth"]

def load_model(path=MODEL_PATH):
    m = joblib.load(path)
    if list(m.get("num_cols", [])) != EXPECTED_NUMERIC:
        raise RuntimeError(f"Model file {path.name} (version {m.get('version')}) expects numeric features "
                           f"{m.get('num_cols')} but this code fills {EXPECTED_NUMERIC}. Retrain with train_arms_model.py.")
    return m

def _expit(z): return 1 / (1 + np.exp(-np.clip(z, -50, 50)))

def _rows(m, case, contexts):
    """design matrix (one row per context) in the bundle's column order, numerics standardized"""
    cols = m["ridge_cols"]; X = np.zeros((len(contexts), len(cols)))
    lev_fixed = dict(kinematics=case["kinematics"], alloy2=case["alloy"], operator=case["operator"],
                     use_policy=use_policy_from_uses(int(case["uses_teeth"])), tooth_group2=case["tooth_group"], era=TODAY_ERA)
    for i, ctx in enumerate(contexts):
        lev = {**lev_fixed, "region": ctx["region"], "design": ctx["design"], "unit": ctx["unit"]}
        for f, l in lev.items():
            c = f"{f}_{l}"
            if c in cols: X[i, cols.index(c)] = 1.0
        X[i, cols.index("max_uses_teeth")] = float(case["uses_teeth"])
    Xraw = X.copy()
    X[:, m["num_idx"]] = (X[:, m["num_idx"]] - m["num_mu"]) / m["num_sd"]
    return X, Xraw, lev_fixed

def predict(m, case: dict) -> dict:
    contexts = m["context_arms"].to_dict("records")
    Xs, Xraw, levels = _rows(m, case, contexts)
    # main: ridge, standardized over contexts; interval from the study-bootstrap coefficient draws
    with np.errstate(all="ignore"):   # macOS Accelerate emits spurious matmul warnings; values are finite
        eta = m["ridge_intercept"] + Xs @ m["ridge_coef"]
        main_pct = float(100 * _expit(eta).mean())
        B = m["ridge_boot"]                                    # (n_boot, 1 + p)
        sims = 100 * _expit(B[:, 0][:, None] + B[:, 1:] @ Xs.T).mean(axis=1)
    lo, hi = np.percentile(sims, [2.5, 97.5])
    # control: recalibrated RF, standardized the same way (same columns as the ridge)
    p_rf = np.clip(m["rf"].predict(Xraw) / 100, 1e-4, 1 - 1e-4)
    a, b = m["rf_recal"]
    rf_recal_pct = float(100 * _expit(a + b * np.log(p_rf / (1 - p_rf))).mean())
    refs = [dict(factor=f, **{k: r[k] for k in ("level", "arms", "n", "fractured", "rate_pct")})
            for f in ["kinematics", "alloy2", "operator", "use_policy", "tooth_group2"]
            for r in m["crude"][f] if r["level"] == levels[f]]
    return dict(main_pct=main_pct, main_lo=float(lo), main_hi=float(hi),
                control=dict(name="forêt aléatoire recalibrée", pct=rf_recal_pct, rf_raw_pct=float(100 * p_rf.mean())),
                base_rate_pct=m["base_rate_pct"], levels=levels, references=refs, n_contexts=len(contexts),
                model_name=m["model_name"], version=m["version"], trained_on=m["trained_on"], ridge_C=m["ridge_C"],
                n_arms=m["n_arms"], n_units=m["n_units"], n_fractured=m["n_fractured"], validation=m["validation"])

if __name__ == "__main__":
    m = load_model()
    print("bundle:", m["version"], "| ridge C =", m["ridge_C"], "| features =", len(m["ridge_cols"]), "| boot draws =", m["ridge_boot"].shape[0])
    for c in [dict(kinematics="Rotary", alloy="Conventional NiTi", operator="Endodontist", uses_teeth=1, tooth_group="Molars"),
              dict(kinematics="Rotary", alloy="Conventional NiTi", operator="General practitioner", uses_teeth=8, tooth_group="Molars"),
              dict(kinematics="Reciprocating", alloy="M-Wire", operator="Graduate student", uses_teeth=3, tooth_group="All")]:
        p = predict(m, c)
        print(f"{c}\n  APRÈS  ridge S2 std = {p['main_pct']:.2f} % [{p['main_lo']:.2f}, {p['main_hi']:.2f}]   "
              f"contrôle forêt recalibrée = {p['control']['pct']:.2f} % (brute {p['control']['rf_raw_pct']:.2f} %)")
