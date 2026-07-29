"""Uncertainty — calibration quality, behaviour under distribution shift, and the ensemble comparison."""

import numpy as np
import pandas as pd
import streamlit as st

from components import charts, ui
from config import theme as T
from utils import data_loader as L


def render():
    b = L.load_bundle()
    batches = L.load_batches()["batches"]

    ui.page_header(
        "Findings",
        "Uncertainty and calibration",
        "A prediction is only useful in a regulated setting if the confidence attached to it can be "
        "trusted. This section asks three things: are the intervals the right width, does the "
        "uncertainty grow when the model meets a product it was never trained on, and how does the "
        "single-pass evidential estimate compare against the far more expensive deep ensemble.",
    )

    cal = b["calibration"]

    # ---------------------------------------------------------------- calibration
    ui.section("01", "Are the intervals the right width?")
    st.markdown(
        '<p class="lede">Prediction interval coverage is the share of batches whose laboratory value '
        'actually fell inside the interval. At a nominal 90% level, a well-calibrated model lands near '
        '0.90. Expected calibration error summarises the gap across all levels, and is reported here '
        'with a 10-bin estimate.</p>',
        unsafe_allow_html=True,
    )

    ui.spacer("0.9rem")
    ui.kpi_row([
        dict(label=L.TARGET_NAME[t],
             value=f'{cal[t]["picp"]:.3f}',
             sub=f'ECE {cal[t]["ece"]:.3f} · 10-bin',
             accent=T.GREEN if abs(cal[t]["picp"] - 0.90) <= 0.05 else T.AMBER)
        for t in L.TARGETS
    ])
    ui.small("Prediction interval coverage at the 90% nominal level. Closer to 0.90 is better.")

    ui.spacer("1.1rem")
    fig = charts.grouped_bars(
        [L.TARGET_NAME[t] for t in L.TARGETS],
        [("Observed coverage", [cal[t]["picp"] for t in L.TARGETS], T.STEEL)],
        ytitle="coverage at 90% nominal",
        height=300,
    )
    fig.add_hline(y=0.90, line=dict(color=T.GREY, width=1.4, dash="dot"),
                  annotation_text="nominal 0.90", annotation_position="top left",
                  annotation_font=dict(family="IBM Plex Mono", size=10, color=T.GREY))
    fig.update_yaxes(range=[0.70, 1.0])
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    ui.spacer("0.8rem")
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("##### Does the uncertainty track the error?")
        st.markdown(
            '<p class="small">A useful uncertainty estimate should be larger on the batches the '
            'model actually gets wrong. The correlation below measures that directly.</p>',
            unsafe_allow_html=True,
        )
        rows = []
        for t in L.TARGETS:
            c = cal[t]["corr"]
            rows.append({
                "Target": L.TARGET_NAME[t],
                "Correlation": f"{c:+.3f}",
                "Reading": "tracks error well" if c > 0.4 else ("weak" if c > 0.1 else "essentially none"),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Target"), use_container_width=True)

    with right:
        st.markdown("##### How to read this")
        ui.callout(
            f"Hardness is the standout: its uncertainty correlates {cal['tbl_av_hardness']['corr']:.3f} "
            "with the actual error, so the model knows which hardness predictions to distrust. On the "
            "other three targets the uncertainty is well-sized on average but carries little "
            "batch-to-batch signal.",
            "info",
        )
        ui.callout(
            "<span class='t'>Where this is weakest</span>"
            f"Weight RSD has by far the worst calibration error ({cal['tbl_rsd_weight']['ece']:.3f}, "
            "10-bin) and its uncertainty carries almost no information about the error "
            f"({cal['tbl_rsd_weight']['corr']:+.3f}). This is the clearest limitation in the "
            "uncertainty results.",
            "warn",
        )

    ui.thesis_figure(
        L.figure_path("fig7_calibration.png"),
        "Calibration at three nominal levels, from the authoritative stratified run.",
    )

    # ---------------------------------------------------------------- OOD
    ui.section("02", "What happens on a product the model has never seen")
    ood = b["ood"]
    cv_ref = ood["cv_reference_rmse"]["tbl_av_hardness"]
    ood_rmse = ood["results"]["code15_fold1"]["tbl_av_hardness"]["rmse"]
    ctl_rmse = ood["results"]["code25_fold1"]["tbl_av_hardness"]["rmse"]

    st.markdown(
        '<p class="lede">Two product codes were held out of training entirely. Code 25 sits inside the '
        'hardness range the model learned; code 15 sits in the high-hardness regime the model barely '
        'saw. Comparing them separates a genuine regime shift from ordinary unseen-product variation.</p>',
        unsafe_allow_html=True,
    )

    ui.spacer("0.9rem")
    ui.kpi_row([
        dict(label="Cross-validation reference", value=f"{cv_ref:.2f}", unit="N",
             sub="hardness RMSE on seen products"),
        dict(label="Code 25 · control", value=f"{ctl_rmse:.2f}", unit="N",
             sub=f"{ctl_rmse/cv_ref:.2f}× the reference, unseen but in-regime", accent=T.GREEN),
        dict(label="Code 15 · out of distribution", value=f"{ood_rmse:.2f}", unit="N",
             sub=f"{ood_rmse/cv_ref:.2f}× the reference, unseen and out of regime", accent=T.AMBER),
    ])

    ui.spacer("1.1rem")
    c1, c2 = st.columns([3, 2], gap="large")

    with c1:
        ind = [r["std"]["tbl_av_hardness"] for r in batches if r["code"] in L.FOLD1_CODES]
        ctl = [r["std"]["tbl_av_hardness"] for r in batches if r["code"] == L.CONTROL_CODE]
        oodv = [r["std"]["tbl_av_hardness"] for r in batches if r["code"] == L.OOD_CODE]
        fig = charts.box_by_group(
            [("in-distribution", ind, T.STEEL),
             ("control, unseen", ctl, T.GREEN),
             ("out of distribution", oodv, T.AMBER)],
            ytitle="calibrated hardness uncertainty (N)",
            height=340,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        ui.small(
            "Each point is one held-out batch. Nothing tells the model which code is unseen; the "
            "separation comes from its own uncertainty."
        )

    with c2:
        ratio = float(np.median(oodv)) / float(np.median(ind))
        st.markdown("##### The finding")
        ui.callout(
            f"Error on the unseen in-regime control barely moves ({ctl_rmse/cv_ref:.2f}× the "
            f"cross-validation reference), while error on the unseen out-of-regime code rises to "
            f"{ood_rmse/cv_ref:.2f}×. The model's own uncertainty follows the same split: median "
            f"hardness uncertainty is about {ratio:.1f}× higher on code 15 than on seen products.",
            "good",
        )
        ui.spacer("0.5rem")
        ui.callout(
            "An unfamiliar product is not by itself a problem. An unfamiliar <i>regime</i> is. That "
            "distinction is what makes the flag useful rather than alarmist.",
            "info",
        )

    # ---------------------------------------------------------------- ensemble
    ui.section("03", "Evidential against a deep ensemble")
    ens = b["ensemble"]
    n_seeds = ens.get("n_seeds", 5)

    st.markdown(
        f'<p class="lede">The deep ensemble trains {n_seeds} independent models and uses their spread '
        'as the uncertainty. It is the established reference method and costs roughly {n} times as much '
        'to train and to run. The evidential heads produce their estimate in a single forward pass.</p>'.replace("{n}", str(n_seeds)),
        unsafe_allow_html=True,
    )

    ui.spacer("0.8rem")
    ood_block = ens.get("ood_code15", {})
    if ood_block:
        cats = [L.TARGET_NAME[t] for t in L.TARGETS]
        fig = charts.faceted_bars(
            cats,
            [
                ("Deep ensemble", [ood_block["ensemble"][t]["rmse"] for t in L.TARGETS], T.PLUM),
                ("Evidential, single pass", [ood_block["evidential"][t]["rmse"] for t in L.TARGETS], T.STEEL),
            ],
            ytitle="RMSE on code 15",
            height=310,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        e_h = ood_block["ensemble"]["tbl_av_hardness"]["rmse"]
        v_h = ood_block["evidential"]["tbl_av_hardness"]["rmse"]
        ui.callout(
            f"Under distribution shift the ensemble is consistently more accurate on all four targets, "
            f"most clearly on hardness ({e_h:.2f} N against {v_h:.2f} N, about "
            f"{100*(1-e_h/v_h):.0f}% lower error). In distribution the two are comparable. The honest "
            f"conclusion is a trade-off rather than a winner: the evidential model gives most of the "
            f"benefit at a fraction of the cost, and the ensemble is the more robust choice when "
            f"distribution shift is expected and the compute is available.",
            "info",
        )
        ui.small(
            "This comparison was run on a single fold with a calibration and evaluation split, so it "
            "is noisier than the pooled cross-fitted numbers reported elsewhere. It should be read as "
            "a directional result."
        )
