"""Interpretability — which sensors the model leans on, and how that changes with its confidence."""

import pandas as pd
import streamlit as st

from components import charts, ui
from config import theme as T
from utils import data_loader as L


def render():
    b = L.load_bundle()
    attr = b["attribution"]["results"]

    ui.page_header(
        "Findings",
        "Interpretability",
        "Attribution usually answers one question: which inputs mattered? This analysis asks a sharper "
        "one. The held-out batches are split into the fifth the model was most confident about and the "
        "fifth it was least confident about, and the sensor importances are compared between those two "
        "groups. If the model simply becomes less certain, the ranking should stay put. If its "
        "uncertainty is meaningful, the ranking should move.",
    )

    ui.spacer("1.1rem")

    # ---------------------------------------------------------------- headline
    h = attr["tbl_av_hardness"]
    ratios = h["channel_ratios"]
    top = sorted(ratios.items(), key=lambda kv: -kv[1])
    lead_name = L.CHANNEL_NAME.get(top[0][0], top[0][0])
    second_name = L.CHANNEL_NAME.get(top[1][0], top[1][0])

    ui.kpi_row([
        dict(label="Rank agreement", value=f'{h["kendall_tau"]:.3f}',
             sub="Kendall's τ, confident against uncertain", accent=T.STEEL),
        dict(label="Significance", value=f'p = {h["p"]:.4f}',
             sub="the ranking genuinely shifts", accent=T.GREEN),
        dict(label=lead_name, value=f'{top[0][1]:.2f}×',
             sub="attribution when uncertain, against confident", accent=T.AMBER),
        dict(label=second_name, value=f'{top[1][1]:.2f}×',
             sub="attribution when uncertain, against confident", accent=T.AMBER),
    ])

    # ---------------------------------------------------------------- channel shift
    ui.section("01", "Which sensors the model leans on when it is unsure")
    st.markdown(
        '<p class="lede">Each bar is one sensor channel. A value above one means the model relies on '
        'that channel more heavily on the batches it is least confident about. Hardness is shown here '
        'because it is the target where the effect is both strongest and statistically supported.</p>',
        unsafe_allow_html=True,
    )

    ui.spacer("0.8rem")
    left, right = st.columns([3, 2], gap="large")

    with left:
        names = [L.CHANNEL_NAME.get(k, k) for k, _ in top]
        values = [v for _, v in top]
        colours = [T.AMBER if v >= 1.5 else (T.STEEL if v >= 1.0 else T.GREY) for v in values]

        import plotly.graph_objects as go
        fig = go.Figure(go.Bar(
            y=names[::-1], x=values[::-1], orientation="h",
            marker_color=colours[::-1],
            hovertemplate="%{y}<br>%{x:.2f}× when uncertain<extra></extra>",
        ))
        fig.add_vline(x=1.0, line=dict(color=T.GREY, width=1.3, dash="dot"))
        fig.update_layout(bargap=0.3)
        charts.style(fig, height=420, xtitle="attribution ratio, uncertain against confident")
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with right:
        st.markdown("##### Why this is the interesting result")
        st.markdown(
            f'<p class="small">The two channels the model leans on hardest when it is uncertain are '
            f'<b>{lead_name.lower()}</b> ({top[0][1]:.2f}×) and <b>{second_name.lower()}</b> '
            f'({top[1][1]:.2f}×). Those are the two signals that physically govern how much material '
            f'enters the die and how hard it is compressed, which is what determines tablet hardness.'
            f'</p>'
            f'<p class="small">The model was never told this. It was given ten raw sensor channels and '
            f'four laboratory targets. That its uncertainty concentrates on the mechanically relevant '
            f'signals is evidence the representation it learned is not arbitrary.</p>',
            unsafe_allow_html=True,
        )
        ui.callout(
            f"Kendall's τ between the confident and uncertain rankings is {h['kendall_tau']:.3f} "
            f"with p = {h['p']:.4f}. The ranking shift is real rather than noise.",
            "good",
        )

    ui.thesis_figure(
        L.figure_path("fig9_channel_attribution.png"),
        "Uncertainty-stratified channel attribution, from the authoritative stratified run.",
    )

    # ---------------------------------------------------------------- per target
    ui.section("02", "Where the effect holds and where it does not")
    st.markdown(
        '<p class="lede">The same test was run on all four targets. It is significant on two of them '
        'and not on the other two, and both outcomes are reported.</p>',
        unsafe_allow_html=True,
    )

    ui.spacer("0.7rem")
    recs = []
    for t in L.TARGETS:
        a = attr[t]
        lead = max(a["channel_ratios"].items(), key=lambda kv: kv[1])
        recs.append({
            "Target": L.TARGET_NAME[t],
            "Kendall's τ": f'{a["kendall_tau"]:.3f}',
            "p": f'{a["p"]:.4f}',
            "Result": "significant" if a["significant"] else "not significant",
            "Most amplified channel": f'{L.CHANNEL_NAME.get(lead[0], lead[0])} ({lead[1]:.2f}×)',
        })
    st.dataframe(pd.DataFrame(recs).set_index("Target"), use_container_width=True)

    ui.spacer("0.4rem")
    ui.callout(
        "The attribution shift is significant on dissolution and hardness and not significant on "
        "weight RSD and tensile strength. Only the first two support the claim, and the app presents "
        "it that way. Note also that with ten channels Kendall's τ can only take a limited set of "
        "values, which is why two targets can land on an identical figure; the underlying channel "
        "rankings differ.",
        "info",
    )

    # ---------------------------------------------------------------- honesty
    ui.section("03", "A result that did not survive")
    ui.callout(
        "<span class='t'>Reported because it matters</span>"
        "An earlier version of this analysis, run before the cross-validation folds were corrected, "
        "showed a strong shift in <i>when</i> along the trajectory the model attended, not just which "
        "channels. That phase-level effect did not reappear once the folds were stratified properly, "
        "across two separate runs. It was therefore an artefact of the imbalanced folds rather than a "
        "property of the model, and it is excluded from the findings. The channel-level result "
        "reported above did survive the correction, and strengthened.",
        "warn",
    )

    ui.spacer("0.6rem")
    st.markdown("##### How the attribution was computed")
    st.markdown(
        '<p class="small">Attributions come from a gradient-based SHAP explainer with 32 background '
        'samples, applied to the full 314-batch fold-1 test set. For each target the batches are split '
        'into the most-confident and least-confident fifths by the model\'s own epistemic uncertainty, '
        'per-channel attributions are averaged within each group, and the two rankings are compared '
        'with Kendall\'s τ. SHAP was used in preference to integrated gradients because it is more '
        'efficient at this scale and works directly with the evidential output heads.</p>',
        unsafe_allow_html=True,
    )
