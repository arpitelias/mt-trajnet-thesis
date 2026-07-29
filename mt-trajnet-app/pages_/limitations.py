"""Limitations — what the study shows, what it does not, and why."""

import streamlit as st

from components import charts, ui
from config import theme as T
from utils import data_loader as L


def render():
    b = L.load_bundle()
    law = b["hardness_law"]
    cal = b["calibration"]

    ui.page_header(
        "Closing",
        "Limitations and what comes next",
        "A demonstration that only shows what worked is not much use to anyone assessing it. This "
        "section sets out where the study is weakest, what explains those weaknesses, and which of "
        "them are genuine constraints rather than things that could have been fixed with more effort.",
    )

    # ---------------------------------------------------------------- hardness law
    ui.section("01", "Why hardness is the hard target")
    st.markdown(
        '<p class="lede">Hardness is the one attribute where a tuned gradient-boosting baseline clearly '
        'beats the trajectory model. Rather than leave that as an unexplained loss, the error was '
        'examined across every product code held out in turn. The pattern is strikingly regular.</p>',
        unsafe_allow_html=True,
    )

    ui.spacer("0.9rem")
    left, right = st.columns([3, 2], gap="large")

    with left:
        xs = [abs(m - 50.0) for m in law["mean_hardness"]]
        fig = charts.scatter_with_fit(
            xs, law["hardness_rmse"], law["codes"],
            xtitle="distance of the product code's mean hardness from 50 N",
            ytitle="hardness RMSE for that code",
            height=380, colour=T.AMBER,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        ui.small(
            f"Each point is one of the {len(law['codes'])} product codes, held out in turn. Hover to "
            f"identify a code."
        )

    with right:
        ui.kpi_row([
            dict(label="Correlation", value=f'{law["pearson_r"]:.3f}',
                 sub="Pearson r across all product codes", accent=T.AMBER),
        ])
        ui.spacer("0.5rem")
        ui.kpi_row([
            dict(label="Significance", value=f'p < 10⁻¹⁰',
                 sub="this is not a chance pattern", accent=T.GREEN),
        ])
        ui.spacer("0.5rem")
        ui.callout(
            "Hardness error grows almost linearly with how far a product's typical hardness sits from "
            "the centre of the training distribution. The model is not failing at hardness in general; "
            "it is accurate near the centre and degrades predictably towards the extremes, where it "
            "has seen very few examples.",
            "info",
        )

    ui.spacer("0.5rem")
    ui.callout(
        "<span class='t'>A constraint rather than a flaw</span>"
        "The high-hardness regime lives in five product codes, and one of them holds 64 of the 98 "
        "batches. Because evaluation groups whole codes to prevent leakage, that regime cannot be "
        "spread evenly across folds. Once the dominant code is withheld for the distribution-shift "
        "test, only 34 high-hardness batches remain for training. No amount of tuning changes that; it "
        "is a property of the dataset.",
        "warn",
    )
    ui.thesis_figure(
        L.figure_path("fig6_hardness_vs_meanhardness.png"),
        "Hardness error against each product code's mean hardness.",
    )

    # ---------------------------------------------------------------- limits
    ui.section("02", "The limitations, stated plainly")

    items = [
        ("Low statistical power",
         "Three folds give the corrected paired test very little power. Most differences between "
         "models do not reach significance, and that should be read as inconclusive rather than as "
         "evidence that the models are equivalent.",
         "warn"),
        ("Weight uniformity is poorly calibrated",
         f"Weight RSD has the largest calibration error of the four targets "
         f"({cal['tbl_rsd_weight']['ece']:.3f}, 10-bin) and its uncertainty carries almost no "
         f"information about the actual error ({cal['tbl_rsd_weight']['corr']:+.3f}). Its intervals "
         f"should not be relied on at the batch level.",
         "warn"),
        ("Evidential uncertainty is less robust than an ensemble",
         "Under distribution shift the deep ensemble was more accurate on all four targets. The "
         "evidential model gives most of the benefit at a fraction of the cost, but the ensemble is "
         "the better choice where shift is expected and the compute exists.",
         "info"),
        ("A result that did not replicate",
         "An early analysis suggested the model shifted which part of the trajectory it attended to as "
         "confidence changed. That effect disappeared once the folds were corrected, across two "
         "separate runs, and has been excluded. The channel-level finding survived and strengthened.",
         "info"),
        ("Single-fold calibration in this application",
         "The per-batch intervals shown in this demonstration come from the fold-1 model and its "
         "fold-1 calibration, not the pooled cross-validated values reported in the thesis. They are "
         "labelled accordingly wherever they appear.",
         "info"),
        ("Scope reductions",
         "An input-representation ablation comparing patch tokenisation against uniform resampling was "
         "planned and not carried out, and hyperparameters were fixed rather than searched. Both are "
         "listed as future work rather than presented as completed.",
         "info"),
    ]

    for title, body, kind in items:
        ui.callout(f"<span class='t'>{title}</span>{body}", kind)
        ui.spacer("0.5rem")

    # ---------------------------------------------------------------- what it shows
    ui.section("03", "What the study does support")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(ui.question_card(
            "Contribution 1",
            "Multi-task trajectory modelling",
            "Four correlated quality attributes can be predicted from one shared trajectory encoder, "
            f"at {b['rq2']['params']['reduction']}× fewer parameters than four separate networks and "
            "with no loss of accuracy against those single-task models, significantly better on "
            "dissolution. This is a comparison against single-task variants, not against the "
            "engineered-feature baselines.",
            T.STEEL,
        ), unsafe_allow_html=True)
    with c2:
        st.markdown(ui.question_card(
            "Contribution 2",
            "Uncertainty that responds to regime shift",
            "Error on an unseen product inside the trained regime barely moves, while error on an "
            f"unseen product outside it rises "
            f"{b['ood']['error_ratios']['code15_fold1']['tbl_av_hardness']:.2f}×, and the model's own "
            "uncertainty separates the two cases.",
            T.GREEN,
        ), unsafe_allow_html=True)
    with c3:
        a = b["attribution"]["results"]["tbl_av_hardness"]
        st.markdown(ui.question_card(
            "Contribution 3",
            "Confidence-conditioned interpretability",
            f"The model measurably changes which sensors it relies on as its confidence falls "
            f"(τ = {a['kendall_tau']:.3f}, p = {a['p']:.4f}), concentrating on the signals that "
            "physically govern the attribute being predicted.",
            T.AMBER,
        ), unsafe_allow_html=True)

    # ---------------------------------------------------------------- future
    ui.section("04", "Future work")
    st.markdown(
        '<p class="lede">Four directions follow directly from the limitations above.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="small">'
        '<b>More folds and a nested search.</b> Repeated or nested cross-validation with a proper '
        'hyperparameter search would give the significance tests usable power and remove the fixed '
        'configuration caveat.<br><br>'
        '<b>Targeted data for the extremes.</b> The hardness relationship implies the most valuable '
        'additional batches are the ones furthest from the centre of the hardness distribution, not '
        'simply more batches.<br><br>'
        '<b>Hybrid uncertainty.</b> A small ensemble of evidential models might recover the ensemble\'s '
        'robustness under shift while keeping most of the single-pass efficiency.<br><br>'
        '<b>Cross-site validation.</b> Every result here comes from one production line. Testing on a '
        'second tablet press would establish whether the learned representation transfers or is '
        'specific to this equipment.'
        '</p>',
        unsafe_allow_html=True,
    )

    ui.spacer("0.6rem")
    ui.callout(
        "Nothing in this application suggests the model replaces laboratory testing. It is a research "
        "artefact demonstrating that trajectory-native prediction with calibrated, interpretable "
        "uncertainty is feasible on real production data.",
        "info",
    )
