Trained model weights are excluded from version control because of their size. The raw dataset is
not included; it is publicly archived (see below).

## Reproducibility

Every experiment runs from a single assembled data file whose MD5 fingerprint
(`7d7fc76be1e4940198b76d9d0797a3a9`) is asserted at the start of each notebook, so any change to the
input data is caught immediately. Seed fixed at 42 throughout, with deterministic cuDNN. Training ran
on a single NVIDIA Tesla T4.

## Demonstration application

The Streamlit app presents the study interactively: the dataset and fold design, the model, the
results, calibration, the attribution finding, and a batch explorer showing one batch at a time with
its sensor trace, four predictions with calibrated intervals, and the measured laboratory values.
Switching between the control code and the out-of-distribution code shows the intervals widen only
for the genuine regime shift.

The app reads precomputed results rather than running the model, so it needs no GPU and starts
instantly.

```bash
cd mt-trajnet-app
pip install -r requirements.txt
streamlit run app.py
```

## Dataset

Žagar, J. and Mihelič, J. (2022) 'Big data collection in pharmaceutical manufacturing and its use
for product quality predictions', *Scientific Data*, 9(1), 99.
DOI: https://doi.org/10.1038/s41597-022-01203-x

1,005 production batches across 25 product codes, roughly 4.72 million sensor readings across ten
channels. Published under CC BY 4.0 and archived on figshare. Not included in this repository
because of its size; download it from the DOI above.

## Honest limitations

- One production line, one press, one dataset. Transfer to other equipment is untested.
- Three folds give the corrected paired test very little power.
- Hyperparameters were fixed rather than searched.
- Weight uniformity is poorly calibrated and its intervals should not be relied on per batch.
- Under distribution shift a five-seed deep ensemble was more accurate on all four targets, at
  roughly five times the training and inference cost.
- An earlier phase-level attribution effect did not replicate after the folds were corrected and is
  excluded from the findings.
- The dataset contains no release specification limits, so this work predicts continuous quality
  values and does not classify batches as passing or failing. It is a research artefact, not a
  replacement for laboratory testing.

## License

Code is released under the MIT License. The dataset remains under its original CC BY 4.0 licence.

## Citation

Elias, A. J. (2026) *Trustworthy Deep Learning for Pharmaceutical Tablet Quality Prediction:
End-to-End Multi-Task Modelling with Dual Uncertainty Quantification on Industrial Compression
Trajectories*. MSc thesis, National College of Ireland.
