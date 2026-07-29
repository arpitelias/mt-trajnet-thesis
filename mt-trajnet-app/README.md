# MT-TrajNet — research demonstration application

An interactive presentation of the MSc research project *Trustworthy Deep Learning for
Pharmaceutical Tablet Quality Prediction: End-to-End Multi-Task Modelling with Dual Uncertainty
Quantification on Industrial Compression Trajectories*.

MT-TrajNet reads a tablet press's raw ten-channel sensor trajectory and predicts four laboratory
quality attributes at once, each with a calibrated uncertainty, from a single forward pass.

## Sections

| Section | What it covers |
|---|---|
| **Overview** | The study in one screen: headline figures, the three research questions and what the evidence showed |
| **Results** | Accuracy against engineered-feature baselines, the multi-task comparison, the point-output ablation, pooled metrics with confidence intervals |
| **Uncertainty** | Interval calibration, behaviour under distribution shift, and the evidential-versus-deep-ensemble comparison |
| **Interpretability** | The uncertainty-stratified attribution result: which sensors the model leans on as its confidence drops |
| **The data** | Dataset provenance, the four targets, the bimodal hardness discovery and the fold design it forced |
| **The model** | Architecture, the evidential output, training configuration and evaluation protocol |
| **Batch explorer** | One batch at a time: the raw sensor trace, the four predictions with calibrated intervals, and the laboratory values |
| **Limitations** | The hardness law, the constraints of the study, and future work |

## Running it

```bash
pip install -r requirements.txt
streamlit run app.py
```

On Windows, if the `streamlit` command is not on your path:

```bash
python -m streamlit run app.py
```

## How it works

The application performs no inference. Every value it displays was computed once from the
authoritative fold-1 model and the study's result files, and is read from two data files:

- `data/results_bundle.json` — study-level results, assembled directly from the result files in the
  thesis repository by `build_bundle.py`. Nothing in it is typed by hand.
- `data/app_data.json` — per-batch predictions, calibrated uncertainty and 90% prediction intervals
  for 378 held-out batches, produced by notebook 17.
- `data/trajectories.json` — downsampled ten-channel sensor curves for 30 representative batches,
  used by the batch explorer.

This keeps the application fast, dependency-light and deployable anywhere, and it means every number
on screen traces back to a verified result file rather than to a live model run.

## Structure

```
mt-trajnet-app/
├── app.py                  entry point and navigation
├── config/theme.py         palette, typography, shared Plotly styling
├── components/ui.py        page headers, KPI cards, callouts, badges
├── components/charts.py    chart builders sharing one visual language
├── utils/data_loader.py    cached data access, target and channel naming
├── pages_/                 one module per section
├── data/                   the two data files described above
└── assets/figures/         the nine figures as published in the thesis
```

The light theme is pinned in `.streamlit/config.toml` so the application renders identically
regardless of the viewer's system or browser theme.

## Honesty notes

These are stated in the application itself, and repeated here.

- The hardness result is reported as it stands: MT-TrajNet trails tuned gradient boosting on this
  target, and the ablation and limitations explain why rather than smoothing it over.
- With three folds the corrected paired test has very little power, so non-significant differences
  are weak evidence in either direction, not evidence of equivalence.
- The attribution shift is significant on dissolution and hardness and not significant on weight RSD
  and tensile strength; both outcomes are shown.
- Expected calibration error is always reported with its bin count (10-bin).
- Weight RSD is the weakest calibrated target and is labelled as such.
- The phase-level attribution effect seen before the folds were corrected did not replicate, and is
  excluded from the findings rather than quietly dropped.
- Prediction intervals shown per batch are fold-1 calibrated, not the pooled cross-validated values.
- Nothing in the application implies the model replaces laboratory testing.
