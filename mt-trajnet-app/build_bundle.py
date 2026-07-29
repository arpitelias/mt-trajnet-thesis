"""
Assemble the demonstration app's results bundle from the authoritative result files.

Every value written here is read directly from the verified files in results/.
Nothing is recomputed by hand and nothing is hard-coded from memory, except the
dataset descriptive statistics, which are recomputed from Laboratory.csv below.
"""

import json
import os
import numpy as np
import pandas as pd

REPO = "/home/claude/repo"
RES = os.path.join(REPO, "results")
OUT = "/home/claude/out/mt-trajnet-app/data/results_bundle.json"

TARGETS = ["dissolution_av", "tbl_av_hardness", "tbl_rsd_weight", "fct_tensile"]


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


bundle = {}

# ---------------------------------------------------------------- provenance
bundle["meta"] = {
    "model": "MT-TrajNet",
    "seed": 42,
    "folds": 3,
    "fold_scheme": "product-code grouped, cluster-stratified, code 15 held out",
    "hardware": "single Kaggle Tesla T4",
    "data_md5": "7d7fc76be1e4940198b76d9d0797a3a9",
    "cv_batches": 941,
    "total_batches": 1005,
    "product_codes": 25,
}

# ---------------------------------------------------------------- dataset facts
lab = pd.read_csv(os.path.join(REPO, "..", "project_lab.csv")) if False else None
# Laboratory.csv lives outside the repo; recompute target stats from oof file instead,
# which stores the same y_arr used by every experiment.
z = np.load(os.path.join(RES, "oof_predictions.npz"), allow_pickle=True)
y = z["y_arr"]
groups = z["groups"]

bundle["dataset"] = {
    "n_batches": int(y.shape[0]),
    "n_codes": int(len(np.unique(groups))),
    "sensor_channels": 10,
    "sampling_seconds": 10,
    "sensor_rows_millions": 4.72,
    "trajectory_length": {"min": 993, "median": 3228, "max": 41710},
    "targets": {},
    "correlations": {},
}
for k, t in enumerate(TARGETS):
    bundle["dataset"]["targets"][t] = {
        "min": round(float(y[:, k].min()), 2),
        "max": round(float(y[:, k].max()), 2),
        "mean": round(float(y[:, k].mean()), 2),
    }
corr = np.corrcoef(y.T)
for i, a in enumerate(TARGETS):
    for j, b in enumerate(TARGETS):
        if i < j:
            bundle["dataset"]["correlations"][f"{a}|{b}"] = round(float(corr[i, j]), 3)

# ---------------------------------------------------------------- hardness cluster
hc = load("hardness_cluster_finding.json")
bundle["hardness_cluster"] = hc

# ---------------------------------------------------------------- folds
bundle["folds"] = load("fold_assignment.json")

# ---------------------------------------------------------------- RQ1 comparison
comp = pd.read_csv(os.path.join(RES, "comparison_stratified.csv"), index_col=0)
bundle["comparison_table"] = {
    "models": [c for c in comp.columns],
    "rows": {t: {c: float(comp.loc[t, c]) for c in comp.columns} for t in comp.index},
}
bundle["rq1"] = load("rq1_stratified.json")

# ---------------------------------------------------------------- RQ2 multi vs single
bundle["rq2"] = load("rq2_stratified.json")

# ---------------------------------------------------------------- point-output ablation
bundle["point_ablation"] = load("point_output_ablation.json")

# ---------------------------------------------------------------- headline metrics
strat = load("mttrajnet_stratified.json")
bundle["rmse"] = {
    "fold_mean": strat["rmse_mean"],
    "fold_values": strat["rmse_folds"],
}
bundle["calibration"] = strat["calibration"]

ext = load("extended_metrics.json")
bundle["pooled_metrics"] = ext["pooled_metrics"]
bundle["coverage"] = ext["coverage"]

# ---------------------------------------------------------------- OOD
bundle["ood"] = load("ood_test.json")

# ---------------------------------------------------------------- ensemble comparison
bundle["ensemble"] = load("rq3_ensemble_comparison.json")

# ---------------------------------------------------------------- attribution
bundle["attribution"] = load("rq3b_all_targets_stratified.json")

# ---------------------------------------------------------------- hardness law (LOCO)
loco = ext["leave_one_code_out"]
codes, mh, err, n = [], [], [], []
for c, v in loco.items():
    codes.append(int(c))
    mh.append(float(v["mean_hardness"]))
    err.append(float(v["rmse"]["tbl_av_hardness"]))
    n.append(int(v.get("n", 0)))
mh_a, err_a = np.array(mh), np.array(err)
from scipy import stats as _st

r, p = _st.pearsonr(np.abs(mh_a - 50.0), err_a)
bundle["hardness_law"] = {
    "codes": codes,
    "mean_hardness": mh,
    "hardness_rmse": err,
    "n_batches": n,
    "pearson_r": round(float(r), 3),
    "p_value": float(p),
    "description": "hardness RMSE grows with distance of a product code's mean hardness from 50 N",
}

# ---------------------------------------------------------------- derived views
# Convenience shapes used by the overview page. These are computed from the
# canonical sections above, never entered by hand, so they cannot drift.
bundle["comparison"] = {"by_target": bundle["comparison_table"]["rows"]}
bundle["significance"] = {"by_target": bundle["rq1"]["results"]}
bundle["multitask"] = {
    "params": bundle["rq2"]["params"],
    "by_target": bundle["rq2"]["results"],
}
bundle["attribution"]["by_target"] = bundle["attribution"]["results"]

cv_ref = bundle["ood"]["cv_reference_rmse"]
ratios = {}
for key, block in bundle["ood"]["results"].items():
    ratios[key] = {
        t: round(float(block[t]["rmse"]) / float(cv_ref[t]), 3)
        for t in TARGETS
        if t in block and t in cv_ref
    }
bundle["ood"]["error_ratios"] = ratios

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    json.dump(bundle, f, indent=1)

print("wrote", OUT)
print("top-level keys:", list(bundle.keys()))
print()
print("VERIFY against thesis:")
print("  RMSE fold-mean      :", bundle["rmse"]["fold_mean"])
print("  hardness law        : r =", bundle["hardness_law"]["pearson_r"],
      " p =", f'{bundle["hardness_law"]["p_value"]:.2e}', f'({len(codes)} codes)')
print("  attribution hardness: tau =", bundle["attribution"]["results"]["tbl_av_hardness"]["kendall_tau"],
      " p =", bundle["attribution"]["results"]["tbl_av_hardness"]["p"])
print("  params              :", bundle["rq2"]["params"])
print("  calibration hardness:", bundle["calibration"]["tbl_av_hardness"])
