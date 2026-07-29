"""Results — accuracy against baselines, the multi-task comparison, and the ablation."""

import pandas as pd
import streamlit as st

from components import charts, ui
from config import theme as T
from utils import data_loader as L


def render():
    b = L.load_bundle()

    ui.page_header(
        "Findings",
        "Results",
        "Three questions are answered here. Does modelling the raw trajectory beat engineered-feature "
        "baselines? Does predicting four attributes together beat four separate models? And is the "
        "evidential output layer costing accuracy? Every figure below comes from the same stratified "
        "cross-validation, so the comparisons are like for like.",
    )

    # ---------------------------------------------------------------- headline
    ui.spacer("1.2rem")
    rmse = b["rmse"]["fold_mean"]
    ui.kpi_row([
        dict(label="Dissolution", value=f'{rmse["dissolution_av"]:.3f}', unit="%",
             sub="RMSE, mean of three folds", accent=T.STEEL),
        dict(label="Hardness", value=f'{rmse["tbl_av_hardness"]:.3f}', unit="N",
             sub="the one target a baseline wins", accent=T.AMBER),
        dict(label="Weight RSD", value=f'{rmse["tbl_rsd_weight"]:.3f}', unit="%",
             sub="RMSE, mean of three folds", accent=T.GREEN),
        dict(label="Tensile", value=f'{rmse["fct_tensile"]:.3f}', unit="MPa",
             sub="RMSE, mean of three folds", accent=T.PLUM),
    ])

    # ---------------------------------------------------------------- RQ1
    ui.section("01", "Trajectory model against engineered-feature baselines")
    st.markdown(
        '<p class="lede">Every model here was fitted on the same folds, with product codes kept '
        'whole so no code appears in both training and test. The baselines use the 28 engineered '
        'batch-level features; MT-TrajNet reads the raw ten-channel trajectory.</p>',
        unsafe_allow_html=True,
    )

    rows = b["comparison_table"]["rows"]
    order = ["dummy", "lasso", "lasso_tuned", "xgb", "xgb_tuned", "MT_TrajNet"]
    pretty = {
        "dummy": "Mean predictor",
        "lasso": "LASSO",
        "lasso_tuned": "LASSO (tuned)",
        "xgb": "XGBoost",
        "xgb_tuned": "XGBoost (tuned)",
        "MT_TrajNet": "MT-TrajNet",
    }
    table = pd.DataFrame(
        {pretty[m]: {L.TARGET_NAME[t]: rows[t][m] for t in L.TARGETS} for m in order}
    )

    def highlight_best(s):
        best = s.min()
        return [
            f"background-color:{T.GREEN_BG};font-weight:600" if v == best else ""
            for v in s
        ]

    st.dataframe(
        table.style.apply(highlight_best, axis=1).format("{:.3f}"),
        use_container_width=True,
    )
    ui.small(
        "Root mean squared error, lower is better. The best value in each row is highlighted. "
        "Units follow the target: percent for dissolution and weight RSD, newtons for hardness, "
        "megapascals for tensile strength."
    )

    ui.spacer("1.2rem")
    left, right = st.columns([3, 2], gap="large")

    with left:
        cats = [L.TARGET_NAME[t] for t in L.TARGETS]
        fig = charts.faceted_bars(
            cats,
            [
                ("Best engineered-feature baseline",
                 [rows[t]["best_baseline"] for t in L.TARGETS], T.CLAY),
                ("MT-TrajNet",
                 [rows[t]["MT_TrajNet"] for t in L.TARGETS], T.STEEL),
            ],
            ytitle="RMSE (lower is better)",
            height=310,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        ui.small(
            "Each target has its own panel and its own vertical scale, because the four attributes "
            "are measured in different units and over very different ranges."
        )

    with right:
        st.markdown("##### Reading this honestly")
        rq1 = b["rq1"]["results"]
        lines = []
        for t in L.TARGETS:
            r = rq1[t]
            kind = "sig" if r["significant"] else "nsig"
            verdict = "significant" if r["significant"] else "not significant"
            pval = r["p"]
            badge = ui.badge(f"p = {pval:.3f} · {verdict}", kind)
            lines.append(
                f'<div style="margin-bottom:9px;"><b>{L.TARGET_NAME[t]}</b> {badge}<br>'
                f'<span class="small">MT-TrajNet {r["mt_mean"]:.3f} against tuned XGBoost '
                f'{r["xgb_mean"]:.3f}</span></div>'
            )
        st.markdown("".join(lines), unsafe_allow_html=True)
        ui.callout(
            "Three folds gives the corrected paired test very little power, so a non-significant "
            "result here is weak evidence in either direction rather than proof of equivalence.",
            "info",
        )

    ui.spacer("0.6rem")
    def pct_gap(t):
        return 100.0 * (rows[t]["MT_TrajNet"] / rows[t]["best_baseline"] - 1.0)

    ui.callout(
        "<span class='t'>The honest headline</span>"
        f"MT-TrajNet is close to the strongest engineered-feature baseline on dissolution "
        f"({pct_gap('dissolution_av'):+.0f}%) and weight uniformity "
        f"({pct_gap('tbl_rsd_weight'):+.0f}%). It is behind on tensile strength "
        f"({pct_gap('fct_tensile'):+.0f}%) and clearly behind on hardness "
        f"({pct_gap('tbl_av_hardness'):+.0f}%: {rows['tbl_av_hardness']['MT_TrajNet']:.3f} N against "
        f"{rows['tbl_av_hardness']['xgb_tuned']:.3f} N for tuned XGBoost). Only the hardness gap "
        f"reaches significance (p = {rq1['tbl_av_hardness']['p']:.3f}); the tensile gap does not, but "
        "with three folds that is inconclusive rather than evidence of parity. The hardness result has "
        "a structural explanation, set out on the Limitations page.",
        "warn",
    )

    # ---------------------------------------------------------------- RQ2
    ui.section("02", "One shared model against four separate ones")
    rq2 = b["rq2"]
    params = rq2["params"]

    st.markdown(
        '<p class="lede">The single-task comparison keeps the architecture identical and only removes '
        'the sharing: four networks, one per target, trained on the same folds. If sharing a trajectory '
        'encoder across correlated attributes helps, it should show up here.</p>',
        unsafe_allow_html=True,
    )

    ui.spacer("0.8rem")
    c1, c2 = st.columns([3, 2], gap="large")

    with c1:
        cats = [L.TARGET_NAME[t] for t in L.TARGETS]
        fig = charts.faceted_bars(
            cats,
            [
                ("Four single-task models",
                 [rq2["results"][t]["single_mean"] for t in L.TARGETS], T.GREY),
                ("Multi-task MT-TrajNet",
                 [rq2["results"][t]["multi_mean"] for t in L.TARGETS], T.STEEL),
            ],
            ytitle="RMSE (lower is better)",
            height=310,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    with c2:
        ui.kpi_row([
            dict(label="Parameters, shared", value=f'{params["multi_task"]:,}',
                 sub="one encoder, four heads", accent=T.STEEL),
        ])
        ui.spacer("0.5rem")
        ui.kpi_row([
            dict(label="Parameters, separate", value=f'{params["four_single_task"]:,}',
                 sub=f'{params["reduction"]}× more than the shared model'),
        ])
        ui.spacer("0.5rem")
        d = rq2["results"]["dissolution_av"]
        ui.callout(
            f"Multi-task is numerically better on all four targets and significantly better on "
            f"dissolution (p = {d['p']:.3f}), at {params['reduction']}× fewer parameters. "
            f"The other three differences are within the noise of three folds.",
            "good",
        )

    ui.thesis_figure(
        L.figure_path("fig8_multitask_vs_singletask.png"),
        "Multi-task against single-task on the stratified folds, with the parameter comparison.",
    )

    # ---------------------------------------------------------------- ablation
    ui.section("03", "Is the evidential output layer costing accuracy?")
    pa = b["point_ablation"]
    point = pa["point_output_rmse_mean"]
    evid = pa["reference_evidential_rmse"]
    xgb = pa["reference_tuned_xgb_rmse"]

    st.markdown(
        '<p class="lede">The evidential heads buy calibrated uncertainty, but they change the loss. '
        'This ablation keeps the same encoder and folds and swaps the evidential heads for a plain '
        'linear output trained with mean squared error, which isolates the cost of the objective '
        'itself.</p>',
        unsafe_allow_html=True,
    )

    ui.spacer("0.8rem")
    cats = [L.TARGET_NAME[t] for t in L.TARGETS]
    fig = charts.faceted_bars(
        cats,
        [
            ("Point output, MSE loss", [point[t] for t in L.TARGETS], T.PLUM),
            ("Evidential MT-TrajNet", [evid[t] for t in L.TARGETS], T.STEEL),
            ("Tuned XGBoost", [xgb[t] for t in L.TARGETS], T.CLAY),
        ],
        ytitle="RMSE (lower is better)",
        height=320,
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    gap_loss = evid["tbl_av_hardness"] - point["tbl_av_hardness"]
    gap_left = point["tbl_av_hardness"] - xgb["tbl_av_hardness"]
    ui.callout(
        f"On hardness the point-output model reaches {point['tbl_av_hardness']:.3f} N against "
        f"{evid['tbl_av_hardness']:.3f} N for the evidential model, so the evidential objective costs "
        f"about {gap_loss:.1f} N. It still trails tuned XGBoost by roughly {gap_left:.1f} N, which means "
        f"most of the hardness gap comes from the trajectory encoder and the small number of "
        f"high-hardness training batches, not from the choice of evidential regression. On the other "
        f"three targets the point-output model is equal or slightly worse, so the uncertainty comes at "
        f"very little accuracy cost elsewhere.",
        "info",
    )

    # ---------------------------------------------------------------- pooled
    ui.section("04", "Pooled metrics with confidence intervals")
    pooled = b["pooled_metrics"]
    recs = []
    for t in L.TARGETS:
        m = pooled[t]
        recs.append({
            "Target": L.TARGET_NAME[t],
            "RMSE": f'{m["rmse"]:.3f}',
            "RMSE 95% CI": f'{m["rmse_ci"][0]:.3f} – {m["rmse_ci"][1]:.3f}',
            "MAE": f'{m["mae"]:.3f}',
            "R²": f'{m["r2"]:.3f}',
            "R² 95% CI": f'{m["r2_ci"][0]:.3f} – {m["r2_ci"][1]:.3f}',
            "n": m["n"],
        })
    st.dataframe(pd.DataFrame(recs).set_index("Target"), use_container_width=True)
    ui.small(
        "Pooled over all 941 out-of-fold batches, with bootstrap confidence intervals. These pooled "
        "values differ slightly from the fold-mean RMSE quoted at the top of the page: both are "
        "correct, they are simply different quantities."
    )
