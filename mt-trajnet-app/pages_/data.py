"""The data — provenance, the four targets, the bimodal hardness discovery, and fold design."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components import charts, ui
from config import theme as T
from utils import data_loader as L


def render():
    b = L.load_bundle()
    ds = b["dataset"]
    hc = b["hardness_cluster"]
    folds = b["folds"]

    ui.page_header(
        "Study design",
        "The data",
        "The study uses a public dataset of industrial tablet compression runs. What follows is not "
        "only a description of it: one property of the data, discovered early, went on to shape every "
        "design decision in the project, including how the model is evaluated and which product code "
        "was chosen to test distribution shift.",
    )

    ui.spacer("1.2rem")
    ui.kpi_row([
        dict(label="Production batches", value=f'{ds["n_batches"]:,}',
             sub=f'across {ds["n_codes"]} product codes'),
        dict(label="Sensor channels", value=str(ds["sensor_channels"]),
             sub=f'sampled every {ds["sampling_seconds"]} seconds'),
        dict(label="Sensor readings", value=f'{ds["sensor_rows_millions"]:.2f}', unit="million",
             sub="raw rows across the whole dataset"),
        dict(label="Quality targets", value="4",
             sub="measured in the laboratory after production"),
    ])

    ui.spacer("0.6rem")
    ui.small(
        "Source: Žagar and Mihelič (2022), published in Scientific Data under a CC BY 4.0 licence. "
        "The manufacturer and products were anonymised by the dataset authors before release, and no "
        "personal or clinical data is involved."
    )

    # ---------------------------------------------------------------- trajectories
    ui.section("01", "Every batch is a time series, and they are not the same length")
    tl = ds["trajectory_length"]
    st.markdown(
        f'<p class="lede">A batch runs for as long as it takes. The shortest in this dataset is '
        f'{tl["min"]:,} samples and the longest is {tl["max"]:,}, a factor of more than forty, with a '
        f'median of {tl["median"]:,}. Any model reading these traces directly has to handle that spread, '
        f'which is why the pipeline downsamples by a stride of two and pads or truncates to a fixed '
        f'6,000 steps.</p>',
        unsafe_allow_html=True,
    )
    ui.thesis_figure(
        L.figure_path("fig1_lengths.png"),
        "Distribution of trajectory lengths across the 1,005 batches.",
    )
    ui.thesis_figure(
        L.figure_path("fig3_sensors.png"),
        "Example sensor traces from a single batch. Eight of the ten channels are shown.",
    )

    # ---------------------------------------------------------------- targets
    ui.section("02", "The four quality attributes")
    st.markdown(
        '<p class="lede">Each batch carries four laboratory measurements. They are related but not '
        'interchangeable, which is the regime where sharing a model across them can help: there is '
        'common structure to exploit without the tasks being redundant.</p>',
        unsafe_allow_html=True,
    )

    ui.spacer("0.7rem")
    left, right = st.columns([2, 3], gap="large")

    with left:
        recs = []
        for t in L.TARGETS:
            s = ds["targets"][t]
            recs.append({
                "Target": L.TARGET_NAME[t],
                "Unit": L.TARGET_UNIT[t],
                "Range": f'{s["min"]:.2f} – {s["max"]:.2f}',
                "Mean": f'{s["mean"]:.2f}',
            })
        st.dataframe(pd.DataFrame(recs).set_index("Target"), use_container_width=True)

    with right:
        pairs = ds["correlations"]
        labels, values, colours = [], [], []
        for key, v in sorted(pairs.items(), key=lambda kv: -abs(kv[1])):
            a, bb = key.split("|")
            labels.append(f"{L.TARGET_NAME[a]} · {L.TARGET_NAME[bb]}")
            values.append(v)
            colours.append(T.STEEL if v >= 0 else T.CLAY)
        fig = go.Figure(go.Bar(
            y=labels[::-1], x=values[::-1], orientation="h", marker_color=colours[::-1],
            hovertemplate="%{y}<br>r = %{x:.3f}<extra></extra>",
        ))
        fig.add_vline(x=0, line=dict(color=T.GREY, width=1.2))
        fig.update_layout(bargap=0.32)
        charts.style(fig, height=260, xtitle="Pearson correlation between targets")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    ui.callout(
        "The strongest relationships are dissolution with tensile strength "
        f'({pairs["dissolution_av|fct_tensile"]:+.2f}) and hardness with tensile strength '
        f'({pairs["tbl_av_hardness|fct_tensile"]:+.2f}). Weight uniformity is close to independent of '
        "the rest, which is worth remembering later: it is also the target where the model's "
        "uncertainty carries the least information.",
        "info",
    )
    ui.thesis_figure(
        L.figure_path("fig4_targets.png"),
        "Distribution of the four laboratory targets.",
    )

    # ---------------------------------------------------------------- bimodal
    ui.section("03", "The finding that shaped the whole study")
    st.markdown(
        '<p class="lede">Hardness is not distributed the way the other targets are. It is bimodal, and '
        'when the high-hardness batches are traced back to their product codes, the split is almost '
        'perfectly clean.</p>',
        unsafe_allow_html=True,
    )

    ui.spacer("0.8rem")
    ui.kpi_row([
        dict(label="High-hardness batches", value=f'{hc["high_cluster_batches"]}',
             sub="above 70 N", accent=T.AMBER),
        dict(label="Low-hardness batches", value=f'{hc["low_cluster_batches"]}',
             sub="at or below 70 N", accent=T.STEEL),
        dict(label="Product codes involved", value=f'{len(hc["high_cluster_codes"])}',
             sub="codes " + ", ".join(str(c) for c in hc["high_cluster_codes"])),
        dict(label="Overlap between clusters", value="none",
             sub="no code appears in both", accent=T.GREEN),
    ])

    ui.spacer("0.6rem")
    ui.callout(
        "<span class='t'>Why this matters so much</span>"
        "The high-hardness regime is concentrated in five product codes, and one of them, code 15, "
        "holds 64 of the 98 batches on its own. Because the evaluation groups by product code to "
        "prevent leakage, that concentration means a naive split can place almost the entire "
        "high-hardness regime into a single fold. It also makes code 15 the natural choice for testing "
        "distribution shift, and it explains why hardness remains the hardest target to predict.",
        "warn",
    )
    ui.thesis_figure(
        L.figure_path("fig2_batches.png"),
        "Batches per product code, coloured by the role each code plays in the study.",
    )

    # ---------------------------------------------------------------- folds
    ui.section("04", "Designing the evaluation around it")
    st.markdown(
        '<p class="lede">The first attempt used a plain grouped three-fold split. It failed for a '
        'specific and instructive reason: it placed 90 of the 98 high-hardness batches into one test '
        'fold, so that fold was evaluated on 90 such batches while the model had trained on only 8. '
        'The resulting hardness error was an artefact of the split rather than a property of the '
        'model.</p>',
        unsafe_allow_html=True,
    )

    ui.spacer("0.7rem")
    c1, c2 = st.columns([3, 2], gap="large")

    with c1:
        hcf = folds["high_cluster_per_fold"]
        fig = charts.grouped_bars(
            ["Fold 1", "Fold 2", "Fold 3"],
            [
                ("Plain grouped split", [90, 6, 2], T.CLAY),
                ("Cluster-stratified split", [hcf["fold_1"], hcf["fold_2"], hcf["fold_3"]], T.STEEL),
            ],
            ytitle="high-hardness batches in the test fold",
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        st.markdown("##### The corrected design")
        rows = []
        for name, codes in folds["folds"].items():
            rows.append({
                "Fold": name.replace("fold_", "Fold "),
                "Product codes": ", ".join(str(c) for c in codes),
            })
        st.dataframe(pd.DataFrame(rows).set_index("Fold"), use_container_width=True)
        ui.callout(
            f"Code {folds['ood_code']} is withheld from cross-validation entirely and reserved for the "
            f"distribution-shift test. The remaining high-hardness codes are balanced across the three "
            f"folds, leaving {b['meta']['cv_batches']} batches in cross-validation.",
            "good",
        )

    ui.spacer("0.4rem")
    ui.callout(
        "One consequence is stated plainly rather than worked around. With code 15 withheld, only 34 "
        "high-hardness batches remain for training across all folds. Hardness therefore stays "
        "structurally difficult, and that constraint runs through every result in this study.",
        "warn",
    )
    ui.thesis_figure(
        L.figure_path("fig5_fold_diagnostic.png"),
        "High-hardness batches per fold, before and after the stratified assignment.",
    )
