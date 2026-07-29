"""Batch explorer — the interactive demonstration: a sensor trace, and what the model made of it."""

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from components import charts, ui
from config import theme as T
from utils import data_loader as L


GROUP_LABEL = {
    "in_distribution": "Seen regime · in-distribution",
    "control": "Unseen product · same regime",
    "ood": "Unseen product · shifted regime",
}
GROUP_NOTE = {
    "in_distribution": "A product code the fold-1 model was evaluated on, drawn from the regime it "
                       "was trained in.",
    "control": "Code 25. The fold-1 model never trained on it, but its hardness sits inside the "
               "range the model learned. This is the control.",
    "ood": "Code 15. Never trained on, and its hardness sits in the high regime the model barely "
           "saw. This is the distribution-shift case.",
}
GROUP_COLOUR = {"in_distribution": T.STEEL, "control": T.GREEN, "ood": T.AMBER}


def render():
    traj_file = L.load_trajectories()
    traj = traj_file["trajectories"]
    batches = {str(r["batch_id"]): r for r in L.load_batches()["batches"]}

    ui.page_header(
        "Demonstration",
        "Batch explorer",
        "Everything so far has been aggregate. This page works one batch at a time: the raw sensor "
        "trace the model reads, the four predictions it returns, the calibrated interval around each, "
        "and the laboratory value that was actually measured. Move between the three groups in the "
        "selector to see how the model's confidence responds when the product is unfamiliar.",
    )

    # ---------------------------------------------------------------- selection
    ui.spacer("1.1rem")
    grouped = {"in_distribution": [], "control": [], "ood": []}
    for bid, tv in traj.items():
        code = tv["code"]
        if code == L.OOD_CODE:
            grouped["ood"].append(bid)
        elif code == L.CONTROL_CODE:
            grouped["control"].append(bid)
        else:
            grouped["in_distribution"].append(bid)
    for g in grouped:
        grouped[g].sort(key=lambda x: int(x))

    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        group = st.radio(
            "Product group",
            list(GROUP_LABEL.keys()),
            format_func=lambda g: GROUP_LABEL[g],
            index=0,
        )
    with c2:
        ids = grouped[group]
        bid = st.selectbox(
            "Batch",
            ids,
            format_func=lambda x: f"batch {x}  ·  product code {traj[x]['code']}  ·  "
                                  f"{traj[x]['length']:,} samples",
        )

    tv = traj[bid]
    row = batches[bid]
    accent = GROUP_COLOUR[group]

    ui.spacer("0.5rem")
    ui.callout(GROUP_NOTE[group], "warn" if group == "ood" else ("good" if group == "control" else "info"))

    # ---------------------------------------------------------------- confidence status
    ind_std = np.median([
        r["std"]["tbl_av_hardness"] for r in batches.values() if r["code"] in L.FOLD1_CODES
    ])
    this_std = row["std"]["tbl_av_hardness"]
    ratio = this_std / ind_std

    ui.spacer("0.9rem")
    ui.kpi_row([
        dict(label="Product code", value=str(tv["code"]),
             sub=L.role_of(tv["code"]), accent=accent),
        dict(label="Trajectory length", value=f'{tv["length"]:,}',
             sub="raw samples before downsampling"),
        dict(label="Hardness uncertainty", value=f"{this_std:.2f}", unit="N",
             sub="calibrated standard deviation", accent=accent),
        dict(label="Against seen products", value=f"{ratio:.1f}×",
             sub=f"typical seen batch is {ind_std:.2f} N",
             accent=T.AMBER if ratio >= 2 else T.GREEN),
    ])

    # ---------------------------------------------------------------- trajectory
    ui.section("01", "What the model reads")
    st.markdown(
        '<p class="lede">These are the sensor channels recorded during the run, each shown in its own '
        'recorded units. The model receives the whole trace, not a summary of it. Add or remove '
        'channels with the selector below.</p>',
        unsafe_allow_html=True,
    )

    default_ch = ["main_comp", "tbl_fill", "ejection", "tbl_speed"]
    chosen = st.multiselect(
        "Sensor channels",
        traj_file["channels"],
        default=default_ch,
        format_func=lambda c: L.CHANNEL_NAME.get(c, c),
    )
    if not chosen:
        chosen = default_ch

    from plotly.subplots import make_subplots

    n = len(chosen)
    ncols = 2 if n > 1 else 1
    nrows = (n + ncols - 1) // ncols
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[L.CHANNEL_NAME.get(c, c) for c in chosen],
        vertical_spacing=0.16 if nrows > 1 else 0.1,
        horizontal_spacing=0.09,
    )
    x = np.linspace(0, 100, traj_file["n_points"])
    palette = [T.STEEL, T.AMBER, T.GREEN, T.PLUM, T.CLAY, T.GREY,
               "#3B7EAE", "#8A5A2B", "#2F6B57", "#6A5A8C"]
    for i_ch, ch in enumerate(chosen):
        r, c = i_ch // ncols + 1, i_ch % ncols + 1
        vals = np.asarray(tv["channels"][ch], dtype=float)
        colour = palette[traj_file["channels"].index(ch) % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=x, y=vals, mode="lines",
                line=dict(color=colour, width=1.5),
                hovertemplate="%{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=r, col=c,
        )
        fig.update_xaxes(showgrid=False, linecolor=T.LINE,
                         tickfont=dict(family="IBM Plex Mono", size=9, color=T.GREY),
                         row=r, col=c)
        fig.update_yaxes(gridcolor=T.LINE, zeroline=False, linecolor=T.LINE,
                         tickfont=dict(family="IBM Plex Mono", size=9, color=T.GREY),
                         row=r, col=c)

    fig.update_layout(
        height=175 * nrows + 40,
        margin=dict(l=10, r=10, t=32, b=28),
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter", size=11, color=T.INK),
        showlegend=False,
    )
    for ann in fig.layout.annotations:
        ann.font = dict(family="IBM Plex Mono", size=10, color=T.GREY)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    ui.small(
        "Each panel shows one sensor channel in its own recorded units against progress through the "
        "batch, displayed at 200 points per channel. Some channels are machine setpoints that are "
        "held constant for a whole run and so appear as flat lines; that is a property of the "
        "process, not a plotting artefact."
    )

    # ---------------------------------------------------------------- predictions
    ui.section("02", "What the model returned")
    st.markdown(
        '<p class="lede">The blue diamond is the prediction, the dark circle is the value the '
        'laboratory actually measured, and the band is the calibrated 90% prediction interval. A '
        'laboratory value inside the band means the model\'s stated confidence held for that '
        'attribute.</p>',
        unsafe_allow_html=True,
    )

    ui.spacer("0.7rem")
    inside_count = 0
    grid = st.columns(2, gap="large")
    for i, t in enumerate(L.TARGETS):
        with grid[i % 2]:
            pred = row["pred"][t]
            lo, hi = row["pi90_low"][t], row["pi90_high"][t]
            actual = row["true"][t]
            unit = L.TARGET_UNIT[t]
            fig, inside = charts.interval_strip(pred, lo, hi, actual, unit)
            inside_count += int(inside)
            tag = "laboratory value inside the interval" if inside else "laboratory value outside the interval"
            tag_c = T.GREEN if inside else T.AMBER
            st.markdown(
                f'<div style="margin-bottom:2px;">'
                f'<span style="font-family:{T.FONT_MONO};font-size:0.7rem;letter-spacing:0.1em;'
                f'text-transform:uppercase;color:{T.GREY};">{L.TARGET_NAME[t]}</span><br>'
                f'<span style="font-family:{T.FONT_DISPLAY};font-size:1.5rem;font-weight:600;">'
                f'{pred:.1f}</span> '
                f'<span style="color:{T.GREY};font-size:0.9rem;">{unit}</span> '
                f'<span style="color:{T.GREY};font-family:{T.FONT_MONO};font-size:0.78rem;">'
                f'&nbsp;·&nbsp; measured {actual:.1f}</span></div>',
                unsafe_allow_html=True,
            )
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown(
                f'<div style="font-family:{T.FONT_MONO};font-size:0.74rem;color:{tag_c};'
                f'margin-top:-14px;margin-bottom:14px;">{tag} &nbsp;·&nbsp; '
                f'90% interval {lo:.1f} to {hi:.1f}</div>',
                unsafe_allow_html=True,
            )

    ui.callout(
        f"On this batch the laboratory value fell inside the 90% interval for "
        f"<b>{inside_count} of the 4</b> attributes.",
        "good" if inside_count >= 3 else "info",
    )
    ui.small(
        "Intervals are shown exactly as the model produces them and are not trimmed to physically "
        "possible ranges. This is most visible on weight RSD, where the lower bound can fall below "
        "zero even though a relative standard deviation cannot be negative. It is a direct "
        "consequence of that target's weaker calibration, reported on the Uncertainty page rather "
        "than hidden by clipping the display."
    )

    # ---------------------------------------------------------------- context
    ui.section("03", "This batch against the rest")
    st.markdown(
        '<p class="lede">Where this batch\'s hardness uncertainty sits relative to every held-out '
        'batch in the study. The three groups are plotted separately; the marker shows the batch '
        'currently selected.</p>',
        unsafe_allow_html=True,
    )

    ui.spacer("0.6rem")
    ind = [r["std"]["tbl_av_hardness"] for r in batches.values() if r["code"] in L.FOLD1_CODES]
    ctl = [r["std"]["tbl_av_hardness"] for r in batches.values() if r["code"] == L.CONTROL_CODE]
    ood = [r["std"]["tbl_av_hardness"] for r in batches.values() if r["code"] == L.OOD_CODE]

    fig = charts.box_by_group(
        [("in-distribution", ind, T.STEEL),
         ("control, unseen", ctl, T.GREEN),
         ("out of distribution", ood, T.AMBER)],
        ytitle="calibrated hardness uncertainty (N)",
        height=360,
    )
    group_x = {"in_distribution": "in-distribution", "control": "control, unseen",
               "ood": "out of distribution"}
    fig.add_trace(go.Scatter(
        x=[group_x[group]], y=[this_std], mode="markers",
        marker=dict(symbol="diamond", size=15, color=T.INK,
                    line=dict(color="white", width=2)),
        hovertemplate=f"batch {bid}<br>%{{y:.2f}} N<extra>selected</extra>",
        showlegend=False,
    ))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    ui.spacer("0.4rem")
    ui.callout(
        "<span class='t'>What to take from this</span>"
        "Switching between the control group and the out-of-distribution group is the clearest way to "
        "see the study's central result. Both are products the model never trained on, yet only one of "
        "them widens the intervals. Unfamiliarity alone is not the problem; an unfamiliar operating "
        "regime is.",
        "info",
    )

    ui.spacer("0.5rem")
    ui.small(
        "Predictions and intervals shown here come from the fold-1 model with fold-1 calibration, not "
        "the pooled cross-validated values reported on the Results page. Thirty representative batches "
        "are included in this demonstration. Nothing on this page is a release decision; it is a "
        "research artefact."
    )
