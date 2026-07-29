"""The model — architecture, the evidential output, and how it was trained and evaluated."""

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from components import ui
from config import theme as T
from utils import data_loader as L


ARCH_SVG = """
<svg viewBox="0 0 940 210" xmlns="http://www.w3.org/2000/svg" style="width:100%;height:auto;">
  <defs>
    <marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3" orient="auto">
      <path d="M0,0 L0,6 L8,3 z" fill="{grey}"/>
    </marker>
  </defs>

  <!-- input -->
  <rect x="8" y="62" width="118" height="86" rx="9" fill="{soft}" stroke="{line}"/>
  <text x="67" y="92" text-anchor="middle" font-family="Inter" font-size="12.5" fill="{ink}">Trajectory</text>
  <text x="67" y="110" text-anchor="middle" font-family="IBM Plex Mono" font-size="10.5" fill="{grey}">10 channels</text>
  <text x="67" y="126" text-anchor="middle" font-family="IBM Plex Mono" font-size="10.5" fill="{grey}">6,000 steps</text>
  <line x1="128" y1="105" x2="164" y2="105" stroke="{grey}" stroke-width="1.4" marker-end="url(#ar)"/>

  <!-- tcn blocks -->
  <text x="330" y="46" text-anchor="middle" font-family="IBM Plex Mono" font-size="10" fill="{grey}"
        letter-spacing="1.6">SHARED TCN ENCODER</text>
  <rect x="166" y="62" width="330" height="86" rx="9" fill="none" stroke="{line}" stroke-dasharray="4 3"/>
  <rect x="178" y="76" width="70" height="58" rx="7" fill="{steelbg}" stroke="{steel}"/>
  <text x="213" y="100" text-anchor="middle" font-family="Inter" font-size="11" fill="{ink}">block 1</text>
  <text x="213" y="116" text-anchor="middle" font-family="IBM Plex Mono" font-size="10" fill="{steel}">d=1</text>
  <rect x="256" y="76" width="70" height="58" rx="7" fill="{steelbg}" stroke="{steel}"/>
  <text x="291" y="100" text-anchor="middle" font-family="Inter" font-size="11" fill="{ink}">block 2</text>
  <text x="291" y="116" text-anchor="middle" font-family="IBM Plex Mono" font-size="10" fill="{steel}">d=2</text>
  <rect x="334" y="76" width="70" height="58" rx="7" fill="{steelbg}" stroke="{steel}"/>
  <text x="369" y="100" text-anchor="middle" font-family="Inter" font-size="11" fill="{ink}">block 3</text>
  <text x="369" y="116" text-anchor="middle" font-family="IBM Plex Mono" font-size="10" fill="{steel}">d=4</text>
  <rect x="412" y="76" width="70" height="58" rx="7" fill="{steelbg}" stroke="{steel}"/>
  <text x="447" y="100" text-anchor="middle" font-family="Inter" font-size="11" fill="{ink}">block 4</text>
  <text x="447" y="116" text-anchor="middle" font-family="IBM Plex Mono" font-size="10" fill="{steel}">d=8</text>
  <line x1="498" y1="105" x2="534" y2="105" stroke="{grey}" stroke-width="1.4" marker-end="url(#ar)"/>

  <!-- pooling -->
  <rect x="536" y="62" width="104" height="86" rx="9" fill="{soft}" stroke="{line}"/>
  <text x="588" y="96" text-anchor="middle" font-family="Inter" font-size="12" fill="{ink}">Pooling</text>
  <text x="588" y="115" text-anchor="middle" font-family="IBM Plex Mono" font-size="10.5" fill="{grey}">128-dim</text>
  <line x1="642" y1="105" x2="676" y2="105" stroke="{grey}" stroke-width="1.4" marker-end="url(#ar)"/>

  <!-- heads -->
  <text x="800" y="46" text-anchor="middle" font-family="IBM Plex Mono" font-size="10" fill="{grey}"
        letter-spacing="1.6">FOUR EVIDENTIAL HEADS</text>
  <rect x="678" y="24" width="250" height="40" rx="7" fill="{cardbg}" stroke="{line}"/>
  <text x="694" y="49" font-family="Inter" font-size="11.5" fill="{ink}">Dissolution</text>
  <text x="912" y="49" text-anchor="end" font-family="IBM Plex Mono" font-size="10" fill="{grey}">γ ν α β</text>
  <rect x="678" y="70" width="250" height="40" rx="7" fill="{cardbg}" stroke="{line}"/>
  <text x="694" y="95" font-family="Inter" font-size="11.5" fill="{ink}">Hardness</text>
  <text x="912" y="95" text-anchor="end" font-family="IBM Plex Mono" font-size="10" fill="{grey}">γ ν α β</text>
  <rect x="678" y="116" width="250" height="40" rx="7" fill="{cardbg}" stroke="{line}"/>
  <text x="694" y="141" font-family="Inter" font-size="11.5" fill="{ink}">Weight RSD</text>
  <text x="912" y="141" text-anchor="end" font-family="IBM Plex Mono" font-size="10" fill="{grey}">γ ν α β</text>
  <rect x="678" y="162" width="250" height="40" rx="7" fill="{cardbg}" stroke="{line}"/>
  <text x="694" y="187" font-family="Inter" font-size="11.5" fill="{ink}">Tensile strength</text>
  <text x="912" y="187" text-anchor="end" font-family="IBM Plex Mono" font-size="10" fill="{grey}">γ ν α β</text>
</svg>
"""


def render():
    b = L.load_bundle()
    meta = b["meta"]
    params = b["rq2"]["params"]

    ui.page_header(
        "Study design",
        "The model",
        "MT-TrajNet is one network with a shared encoder and four prediction heads. It reads the whole "
        "compression trajectory and returns, from a single forward pass, a prediction and an "
        "uncertainty for each of the four quality attributes. No sampling, no ensembling at inference.",
    )

    ui.spacer("1.2rem")
    svg = ARCH_SVG.format(
        ink=T.INK, grey=T.GREY, line=T.LINE, soft=T.SOFT,
        steel=T.STEEL, steelbg=T.STEEL_BG, cardbg=T.CARD,
    )
    components.html(
        f"""<!doctype html><html><head><meta charset="utf-8">
<style>
  html,body {{ margin:0; padding:0; background:{T.CARD}; }}
  body {{ display:flex; align-items:center; justify-content:center; }}
</style></head><body>{svg}</body></html>""",
        height=225,
        scrolling=False,
    )
    ui.small(
        "The dilation factor d doubles at each block, so the receptive field widens quickly and a "
        "single block can span long stretches of the trajectory without any recurrent loop."
    )

    # ---------------------------------------------------------------- encoder
    ui.section("01", "Why a dilated convolutional encoder")
    c1, c2 = st.columns([3, 2], gap="large")
    with c1:
        st.markdown(
            '<p class="lede">The encoder is a temporal convolutional network: four residual blocks, '
            'each with two causal convolutions of kernel width three, ReLU activations and dropout of '
            '0.1, all at 128 channels. Doubling the dilation at every block widens the receptive field '
            'geometrically rather than linearly, which is what allows four blocks to cover a '
            '6,000-step sequence.</p>'
            '<p class="lede">A transformer encoder was considered and rejected. Attention cost grows '
            'with the square of the sequence length, and this dataset\'s longest runs are more than '
            'forty thousand samples before downsampling. The convolutional stack keeps the computation '
            'linear in sequence length, which is what makes end-to-end training on raw trajectories '
            'practical at all.</p>',
            unsafe_allow_html=True,
        )
    with c2:
        ui.callout(
            "<span class='t'>Receptive field</span>"
            "With kernel width 3 and dilations 1, 2, 4 and 8, the stacked blocks reach far enough back "
            "along the trace to relate early filling behaviour to late compression behaviour, which is "
            "the relationship that matters physically.",
            "info",
        )

    # ---------------------------------------------------------------- heads
    ui.section("02", "The evidential output, and what it buys")
    st.markdown(
        '<p class="lede">Each head is a small two-layer network that outputs four numbers rather than '
        'one. Those four parameterise a Normal-Inverse-Gamma distribution over the prediction, '
        'following the deep evidential regression formulation. Instead of a point estimate, the model '
        'returns a distribution, and the shape of that distribution is what the uncertainty is read '
        'from.</p>',
        unsafe_allow_html=True,
    )

    ui.spacer("0.7rem")
    cols = st.columns(4)
    par = [
        ("γ", "gamma", "the predicted value itself"),
        ("ν", "nu", "how much evidence supports that value"),
        ("α", "alpha", "concentration of the variance estimate"),
        ("β", "beta", "scale of the variance estimate"),
    ]
    for col, (sym, name, desc) in zip(cols, par):
        with col:
            st.markdown(
                f'<div class="kpi"><div class="k">{name}</div>'
                f'<div class="v" style="color:{T.STEEL};">{sym}</div>'
                f'<div class="s">{desc}</div></div>',
                unsafe_allow_html=True,
            )

    ui.spacer("0.7rem")
    ui.callout(
        "The practical benefit is that uncertainty arrives with the prediction at no extra cost. A deep "
        "ensemble reaches a similar result by training several models and measuring their "
        "disagreement, which multiplies both training and inference cost. The Uncertainty page compares "
        "the two directly, including where the ensemble genuinely wins.",
        "info",
    )

    # ---------------------------------------------------------------- multi-task
    ui.section("03", "One encoder, four tasks")
    c1, c2 = st.columns([2, 3], gap="large")
    with c1:
        ui.kpi_row([
            dict(label="Shared model", value=f'{params["multi_task"]:,}', unit="params",
                 sub="one encoder, four light heads", accent=T.STEEL),
        ])
        ui.spacer("0.5rem")
        ui.kpi_row([
            dict(label="Four separate models", value=f'{params["four_single_task"]:,}', unit="params",
                 sub=f'{params["reduction"]}× larger in total'),
        ])
    with c2:
        st.markdown(
            '<p class="lede">The total training loss is the sum of the four per-target evidential '
            'losses, weighted equally. Equal weighting is enough because the targets are standardised '
            'to zero mean and unit variance during training, which puts them on a comparable scale; no '
            'learned task-weighting was needed. The four losses are optimised jointly through the '
            'shared encoder, and that shared path is what allows structure learned for one attribute to '
            'benefit the others.</p>',
            unsafe_allow_html=True,
        )

    # ---------------------------------------------------------------- training
    ui.section("04", "Training and evaluation protocol")
    c1, c2 = st.columns(2, gap="large")

    with c1:
        st.markdown("##### Training configuration")
        cfg = pd.DataFrame([
            {"Setting": "Optimiser", "Value": "Adam"},
            {"Setting": "Learning rate", "Value": "5 × 10⁻⁴"},
            {"Setting": "Batch size", "Value": "16"},
            {"Setting": "Maximum epochs", "Value": "150"},
            {"Setting": "Early stopping patience", "Value": "10 epochs"},
            {"Setting": "Gradient clipping", "Value": "norm 1.0"},
            {"Setting": "Random seed", "Value": str(meta["seed"])},
            {"Setting": "Hardware", "Value": meta["hardware"]},
        ]).set_index("Setting")
        st.dataframe(cfg, use_container_width=True)
        ui.small(
            "Hyperparameters were fixed rather than searched. A nested search is listed as future work "
            "rather than claimed here."
        )

    with c2:
        st.markdown("##### Evaluation protocol")
        st.markdown(
            f'<p class="small">Cross-validation is grouped by product code, so no code appears on both '
            f'sides of a split and the estimate reflects performance on genuinely unseen products. '
            f'Code {b["folds"]["ood_code"]} is withheld entirely for the distribution-shift test, '
            f'leaving {meta["cv_batches"]} batches across {meta["folds"]} folds.</p>'
            f'<p class="small">Uncertainty is calibrated by cross-fitting: each fold\'s scale factor is '
            f'estimated only from the other folds\' out-of-fold predictions, so no batch is ever '
            f'calibrated using its own prediction. An earlier version fitted the scale on the same '
            f'predictions it reported on, which was optimistic; it was corrected, and the effect on '
            f'coverage was small.</p>'
            f'<p class="small">Model comparisons use the corrected resampled paired t-test of Nadeau '
            f'and Bengio, which accounts for the dependence between overlapping training folds. With '
            f'three folds this test has low power, and results are reported on that understanding.</p>',
            unsafe_allow_html=True,
        )

    ui.spacer("0.5rem")
    ui.callout(
        f"<span class='t'>Reproducibility</span>"
        f"Every experiment runs from a single assembled data file whose MD5 fingerprint "
        f"(<span class='mono'>{meta['data_md5']}</span>) is asserted at the start of each notebook, so "
        f"any change to the input data is caught immediately rather than discovered later in a result.",
        "good",
    )
