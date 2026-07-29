"""Overview: what the study is, and what it found."""

import numpy as np
import streamlit as st

from components import ui
from components import charts
from config import theme as T
from utils.data_loader import (
    CONTROL_CODE, FOLD1_CODES, OOD_CODE, TARGET_NAME, TARGETS,
    load_batches, load_bundle,
)


def render():
    b = load_bundle()

    ui.page_header(
        "MT-TrajNet · MSc research practicum",
        "Trustworthy tablet quality prediction from raw compression trajectories",
        "Every tablet batch leaves a trace: ten sensors on a rotary press, sampled every ten "
        "seconds for hours. Existing work reduces that trace to a handful of summary numbers "
        "before modelling it. This study models the trace directly, predicts four laboratory "
        "quality attributes at once, and attaches a calibrated uncertainty to every prediction "
        "so that a batch the model cannot judge is visible as such.",
    )

    ui.spacer("1.4rem")

    # --- headline figures, all from the verified result files ---
    params = b["multitask"]["params"]
    ood_ratio = b["ood"]["error_ratios"]["code15_fold1"]["tbl_av_hardness"]
    ctrl_ratio = b["ood"]["error_ratios"]["code25_fold1"]["tbl_av_hardness"]

    ui.kpi_row([
        dict(label="Production batches", value="1,005",
             sub="GMP batches across 25 product codes, 4.72 million sensor rows"),
        dict(label="One model, four targets", value="384,400", unit="params",
             sub=f"{params['reduction']}× fewer than four separate single-task models"),
        dict(label="Unseen, same regime", value=f"{ctrl_ratio:.2f}×", accent=T.GREEN,
             sub="hardness error on a held-out product inside the trained regime"),
        dict(label="Unseen, shifted regime", value=f"{ood_ratio:.2f}×", accent=T.AMBER,
             sub="hardness error on a held-out high-hardness product (code 15)"),
    ])

    ui.spacer("0.6rem")
    ui.callout(
        "The two figures on the right are the study's central result. A product the model has "
        "never seen costs nothing if it sits inside the regime the model was trained on, and "
        "costs a great deal if it does not. The model's own uncertainty separates the two cases "
        "without being told which is which.",
        "info",
    )

    # --- research questions ---
    ui.section("01", "Research questions and what the evidence showed")

    sig = b["significance"]["by_target"]
    mt = b["multitask"]["by_target"]
    attr = b["attribution"]["by_target"]

    hard_mt = sig["tbl_av_hardness"]["mt_mean"]
    hard_xgb = sig["tbl_av_hardness"]["xgb_mean"]
    diss_p = mt["dissolution_av"]["p"]
    tau = attr["tbl_av_hardness"]["kendall_tau"]
    tau_p = attr["tbl_av_hardness"]["p"]

    cal_h = b["calibration"]["tbl_av_hardness"]

    r1c1, r1c2 = st.columns(2, gap="large")
    with r1c1:
        st.markdown(ui.question_card(
            "RQ1 · trajectory representation",
            "Does modelling the raw trace beat engineered features?",
            "Partly. MT-TrajNet is close to the strongest tuned baselines on dissolution and "
            "weight uniformity, but behind on tensile strength and clearly behind on hardness "
            f"({hard_mt:.3f} against {hard_xgb:.3f}). Only the hardness gap is statistically "
            "significant, though with three folds the others are inconclusive rather than equal. "
            "These gaps are reported as findings, not smoothed over.",
            T.STEEL,
        ), unsafe_allow_html=True)
    with r1c2:
        st.markdown(ui.question_card(
            "RQ2 · multi-task structure",
            "Does one shared model beat four separate ones?",
            f"Yes, on the strongest evidence available here. Dissolution improves significantly "
            f"(p = {diss_p}), the remaining targets are numerically no worse, and the shared "
            f"model uses {params['reduction']}× fewer parameters than four single-task networks.",
            T.GREEN,
        ), unsafe_allow_html=True)

    ui.spacer("1rem")
    r2c1, r2c2 = st.columns(2, gap="large")
    with r2c1:
        st.markdown(ui.question_card(
            "RQ3a · calibrated uncertainty",
            "Can the confidence attached to a prediction be trusted?",
            f"Largely yes in distribution: hardness coverage is {cal_h['picp']:.3f} against a "
            f"nominal 0.90, and its uncertainty correlates {cal_h['corr']:.3f} with the actual "
            f"error. Under a genuine regime shift error rises {ood_ratio:.2f}× while an unseen "
            f"product inside the trained regime costs only {ctrl_ratio:.2f}×. Weight uniformity "
            "is the clear exception and is reported as such.",
            T.AMBER,
        ), unsafe_allow_html=True)
    with r2c2:
        st.markdown(ui.question_card(
            "RQ3b · uncertainty-stratified attribution",
            "Does the model explain itself differently when unsure?",
            f"Yes, and this is the study's one strongly significant result. As confidence falls "
            f"the model measurably reorders which sensors it depends on "
            f"(Kendall τ = {tau}, p = {tau_p}), concentrating on die fill and main compression, "
            "the signals that physically govern tablet hardness.",
            T.PLUM,
        ), unsafe_allow_html=True)

    # --- the model at a glance ---
    ui.section("02", "Accuracy against the baselines")

    comp = b["comparison"]
    display_models = ["dummy", "lasso_tuned", "xgb_tuned", "MT_TrajNet"]
    label = {"dummy": "Mean predictor", "lasso_tuned": "LASSO (tuned)",
             "xgb_tuned": "XGBoost (tuned)", "MT_TrajNet": "MT-TrajNet"}
    colour = {"dummy": T.GREY, "lasso_tuned": T.CLAY, "xgb_tuned": T.PLUM, "MT_TrajNet": T.STEEL}

    left, right = st.columns([3, 2], gap="large")
    with left:
        cats = [TARGET_NAME[t] for t in TARGETS]
        # normalise each target to its dummy baseline so four different units share one axis
        series = []
        for m in display_models:
            vals = [comp["by_target"][t][m] / comp["by_target"][t]["dummy"] for t in TARGETS]
            series.append((label[m], vals, colour[m]))
        fig = charts.grouped_bars(cats, series, ytitle="RMSE relative to mean predictor",
                                  height=340)
        fig.add_hline(y=1.0, line=dict(color=T.GREY, width=1, dash="dot"))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        ui.small(
            "Each target is divided by the mean-predictor baseline so four different units can "
            "share one axis. Lower is better; the dotted line is the naive baseline."
        )
    with right:
        ui.spacer("0.4rem")
        st.markdown("##### Reading this honestly")
        ui.small(
            "MT-TrajNet clears the naive baseline on every target and is close to the tuned "
            "baselines on three of four. Hardness is the exception, and it is the one target "
            "where the dataset is genuinely awkward: the high-hardness regime is concentrated "
            "in a handful of product codes, leaving few training examples once they are held "
            "out for honest evaluation."
        )
        ui.spacer("0.5rem")
        ui.callout(
            "With three cross-validation folds, the statistical test has low power. Differences "
            "that do not reach significance are reported as inconclusive rather than as wins.",
            "warn",
        )

    # --- what to look at ---
    ui.section("03", "How to move through this application")

    n_batches = len(load_batches()["batches"])
    guide = [
        ("The data", "Where the 1,005 batches come from, why hardness is bimodal, and how the "
                     "cross-validation had to be rebuilt to keep the evaluation honest."),
        ("The model", "The temporal convolutional encoder, the evidential heads, and why one "
                      "shared encoder is the efficient choice."),
        ("Results", "The full comparison against every baseline, with per-fold values and "
                    "corrected significance tests."),
        ("Uncertainty", "Calibration at three nominal levels, and how the evidential model "
                        "compares with a five-seed deep ensemble."),
        ("Interpretability", "Which sensors drive predictions, and how that changes when the "
                             "model is unsure."),
        ("Batch explorer", f"All {n_batches} held-out batches, one at a time, with predictions, "
                           "intervals and the actual laboratory result."),
        ("Limitations", "What this work does not show, stated plainly."),
    ]
    for i in range(0, len(guide), 2):
        cols = st.columns(2, gap="large")
        for col, (title, desc) in zip(cols, guide[i:i + 2]):
            with col:
                st.markdown(f"**{title}**")
                ui.small(desc)
                ui.spacer("0.5rem")
