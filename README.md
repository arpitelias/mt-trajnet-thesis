# MT-TrajNet: Trustworthy Deep Learning for Pharmaceutical Tablet Quality Prediction

MSc thesis project at the National College of Ireland (2026). End-to-end multi-task deep learning with dual uncertainty quantification on industrial compression trajectories.

Author: Arpit Joshua Elias (24257567)
Programme: MSc in Artificial Intelligence
Module: Practicum CA2 (MSCAI1)
Status: In development (May to August 2026)

## What this project does

A deep learning system that predicts four tablet quality attributes (dissolution, hardness, weight uniformity, tensile strength) directly from the raw 10-second sensor trajectories of a tablet press machine, instead of using engineered summary features as most existing studies do. Every prediction comes with two types of uncertainty (aleatoric and epistemic) and an explanation that depends on how confident the model is.

## Research questions

RQ1 (Trajectory representation): Does end-to-end deep learning on raw sensor trajectories give better predictions than models trained on engineered batch-level features?

RQ2 (Multi-task structure): Does predicting all four quality attributes together with one shared encoder work better than training a separate model for each one?

RQ3 (Trustworthy deployment): Can the combination of dual uncertainty plus uncertainty-stratified attribution give calibrated and explainable confidence even on product codes the model has not seen during training?

## Model

MT-TrajNet is a shared Temporal Convolutional Network (TCN) encoder with four task-specific evidential regression heads. The TCN reads variable-length trajectories (993 to 41,710 samples), pools to a 256-dimensional representation, and each head outputs the parameters of a Normal-Inverse-Gamma distribution following the evidential regression formulation of Amini et al. (2020). Multi-task loss balancing uses the uncertainty-weighting scheme of Kendall et al. (2018).

## Dataset

Zagar and Mihelic (2022), Scientific Data 9(1), 99. Publicly archived on figshare under CC BY 4.0. DOI: https://doi.org/10.1038/s41597-022-01203-x

1,005 GMP-compliant production batches across 25 product codes, 4.72 million sensor readings across 10 channels recorded every 10 seconds. The raw dataset is not included in this repository because of its size and because it is already publicly archived. Download it from the DOI above.

## Repository structure

notebooks   exploratory analysis, audit, feasibility, preprocessing, baselines
results     output tables and figures
deliverables   weekly submissions to the mentor, organised by topic
proposal    abstract, technical architecture, research question, CA2 proposal

The raw dataset is kept locally and is not tracked here. The deep-learning source modules (preprocessing pipeline, model, training, evaluation, attribution) will be added as they are built.

## Status and milestones

Weeks 1-2 (18-31 May): data pipeline and EDA.
Week 3 (1-7 Jun): engineered-feature baselines (LASSO, XGBoost).
Weeks 4-5 (8-21 Jun): MT-TrajNet single-task plus input representation ablation.
Weeks 6-7 (22 Jun - 5 Jul): multi-task variant with evidential uncertainty.
Week 8 (6-12 Jul): deep ensemble variant.
Week 9 (13-19 Jul): leave-one-product-code-out OOD protocol.
Week 10 (20-26 Jul): uncertainty-stratified attribution analysis.
Week 11 (27 Jul - 2 Aug): Streamlit dashboard.
Week 12 (3-8 Aug): thesis draft.
Weeks 13-15 (9-22 Aug): revision, submission, viva preparation.

## License

To be added before the repository is made public (planned: MIT for code; dataset remains under its original CC BY 4.0).

## Citation

Elias, A.J. (2026). Trustworthy Deep Learning for Pharmaceutical Tablet Quality Prediction: End-to-End Multi-Task Modelling with Dual Uncertainty Quantification on Industrial Compression Trajectories. MSc thesis, National College of Ireland.
