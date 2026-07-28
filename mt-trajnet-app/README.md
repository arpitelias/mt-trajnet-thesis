# MT-TrajNet demo

An interactive demo of MT-TrajNet, the multi-task deep learning model from the thesis
*Trustworthy Deep Learning for Pharmaceutical Tablet Quality Prediction*. The model predicts
four tablet quality attributes (dissolution, hardness, weight RSD, tensile strength) directly
from a tablet press's raw sensor trajectory, and returns a calibrated uncertainty for each.

## What it shows

- **Predictions with calibrated 90% intervals** for any batch, next to the actual lab value.
- **A distribution-shift flag.** On products the model never trained on, its uncertainty rises
  sharply. Code 15 (an unseen high-hardness product) shows roughly 3.4x the hardness uncertainty
  of seen products, while an unseen low-hardness control stays flat. The model separates the
  regimes on its own.

## How it works

The app loads `app_data.json`, which was precomputed once from the authoritative fold-1 model
(fixed seed 42, verified to reproduce the thesis fold-1 RMSE and per-code uncertainty). The app
itself runs no model and needs no GPU; it is a viewer over verified results. Predictions,
uncertainty, and intervals for 378 batches (fold-1 test set plus the two held-out codes 15 and 25)
are stored in that file.

## Run locally

```
pip install -r requirements.txt
streamlit run app.py
```

## Honesty notes

- Intervals are single-fold (fold-1) calibrated, not the pooled cross-validation numbers reported
  in the thesis.
- The hardness result is the strongest and best-calibrated on this fold; tensile is the weakest.
- The regime flag is an illustrative decision aid derived from the model's own uncertainty, not a
  validated release rule.
