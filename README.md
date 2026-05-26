# MT-TrajNet: Trustworthy Deep Learning for Pharmaceutical Tablet Quality Prediction

MSc thesis project at the National College of Ireland (2026). End-to-end multi-task deep learning with dual uncertainty quantification on industrial compression trajectories.

**Author:** Arpit Joshua Elias (24257567)
**Programme:** MSc in Artificial Intelligence
**Module:** Practicum CA2 (MSCAI1)
**Status:** In development (May – August 2026)

---

## What this project does

A deep learning system that predicts four tablet quality attributes (dissolution, hardness, weight uniformity, tensile strength) directly from the raw 10-second sensor trajectories of a tablet press machine, instead of using engineered summary features as most existing studies do. Every prediction comes with two types of uncertainty (aleatoric and epistemic) and an explanation that depends on how confident the model is.

## Research questions

- **RQ1 (Trajectory representation):** Does end-to-end deep learning on raw sensor trajectories give better predictions than models trained on engineered batch-level features?
- **RQ2 (Multi-task structure):** Does predicting all four quality attributes together with one shared encoder work better than training a separate model for each one?
- **RQ3 (Trustworthy deployment):** Can the combination of dual uncertainty plus uncertainty-stratified attribution give calibrated and explainable confidence even when applied to product codes the model has not seen during training?

## Model

**MT-TrajNet** — a shared Temporal Convolutional Network (TCN) encoder with four task-specific evidential regression heads. The TCN reads variable-length trajectories (993 to 41,710 samples), pools to a 256-dimensional representation, and the four heads each output the parameters of a Normal-Inverse-Gamma distribution following the evidential regression formulation from Amini et al. (2020). Multi-task loss balancing uses the uncertainty-weighting scheme from Kendall et al. (2018).

## Dataset

Žagar and Mihelič (2022), *Scientific Data* 9(1), 99. Publicly archived on figshare under CC BY 4.0.
DOI: https://doi.org/10.1038/s41597-022-01203-x

1,005 GMP-compliant production batches across 25 product codes, 4.72 million sensor readings across 10 channels recorded every 10 seconds. The dataset is not included in this repository — see the download script in `data/` for how to fetch it from figshare.

## Repository structure

```
mt-trajnet-thesis/
├── README.md
├── requirements.txt
├── data/
│   └── README.md          # how to download the dataset
├── notebooks/
│   └── eda.ipynb          # exploratory data analysis
├── src/
│   ├── preprocessing/     # 4-stage preprocessing pipeline
│   │   ├── stage1_standardise.py
│   │   ├── stage2_normalise.py
│   │   ├── stage3_join.py
│   │   └── stage4_gap_indicator.py
│   ├── models/            # MT-TrajNet and baselines
│   ├── training/          # training loops, nested CV
│   ├── evaluation/        # metrics, OOD protocol
│   └── attribution/       # uncertainty-stratified IG
├── scripts/
│   └── run_pipeline.py    # end-to-end preprocessing
└── results/               # output figures, tables, model checkpoints
```

## Status and milestones

| Week | Dates | Milestone |
|------|-------|-----------|
| 1-2 | 18-31 May | Data pipeline + EDA notebook |
| 3 | 1-7 Jun | Engineered-feature baselines (LASSO, XGBoost) |
| 4-5 | 8-21 Jun | MT-TrajNet single-task + input representation ablation |
| 6-7 | 22 Jun – 5 Jul | Multi-task variant with evidential UQ |
| 8 | 6-12 Jul | Deep ensemble variant |
| 9 | 13-19 Jul | Leave-one-product-code-out OOD protocol |
| 10 | 20-26 Jul | Uncertainty-stratified attribution analysis |
| 11 | 27 Jul – 2 Aug | Streamlit dashboard |
| 12 | 3-8 Aug | Thesis draft |
| 13-15 | 9-22 Aug | Revision, submission, viva preparation |

## License

To be added before the repository is made public (planned: MIT for code, dataset remains under its original CC BY 4.0).

## Citation

If this work is referenced before publication, please cite as:

> Elias, A.J. (2026). Trustworthy Deep Learning for Pharmaceutical Tablet Quality Prediction: End-to-End Multi-Task Modelling with Dual Uncertainty Quantification on Industrial Compression Trajectories. MSc thesis, National College of Ireland.
