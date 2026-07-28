import os
import json
import numpy as np
import streamlit as st
import plotly.graph_objects as go

# ----------------------------------------------------------------------------
# MT-TrajNet demo. Reads precomputed predictions and calibrated uncertainty
# from app_data.json (fold-1 model, verified against the thesis results).
# The app runs no model: every value shown was computed once from the
# authoritative fold-1 weights and saved.
# ----------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="MT-TrajNet | tablet quality with honest uncertainty",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- palette (matched to the thesis figures) ----
INK = "#1A1D23"
BG = "#FBFBF9"
STEEL = "#2D6E9E"
GREEN = "#2E8B6B"
AMBER = "#C77D2E"
GREY = "#6B7075"
LINE = "#E4E2DB"
SOFT = "#F2F1EC"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

.stApp {{ background:{BG}; }}
html, body, [class*="css"] {{ font-family:'Inter',sans-serif; color:{INK}; }}
h1,h2,h3,h4 {{ font-family:'Space Grotesk',sans-serif; color:{INK}; letter-spacing:-0.01em; }}
.block-container {{ padding-top:2.2rem; max-width:1200px; }}

.eyebrow {{ font-family:'IBM Plex Mono',monospace; font-size:0.72rem; letter-spacing:0.16em;
            text-transform:uppercase; color:{STEEL}; margin-bottom:0.35rem; }}
.lede {{ color:{GREY}; font-size:1.02rem; line-height:1.55; max-width:60ch; }}

.chip {{ display:inline-block; background:{SOFT}; border:1px solid {LINE}; border-radius:999px;
         padding:3px 11px; font-size:0.74rem; font-family:'IBM Plex Mono',monospace; color:{GREY}; }}

.metric {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:14px; padding:16px 18px; height:100%; }}
.metric .k {{ font-size:0.74rem; color:{GREY}; text-transform:uppercase; letter-spacing:0.06em;
              font-family:'IBM Plex Mono',monospace; }}
.metric .v {{ font-family:'Space Grotesk',sans-serif; font-size:1.9rem; font-weight:600; line-height:1.15; }}
.metric .s {{ font-size:0.8rem; color:{GREY}; }}

.badge {{ display:inline-block; padding:4px 12px; border-radius:8px; font-size:0.78rem;
          font-weight:600; font-family:'IBM Plex Mono',monospace; }}
.card {{ background:#FFFFFF; border:1px solid {LINE}; border-radius:14px; padding:14px 16px 6px 16px; }}
.card .t {{ font-family:'IBM Plex Mono',monospace; font-size:0.76rem; color:{GREY};
            text-transform:uppercase; letter-spacing:0.06em; }}
.card .p {{ font-family:'Space Grotesk',sans-serif; font-size:1.5rem; font-weight:600; }}
.card .u {{ font-size:0.9rem; color:{GREY}; font-weight:400; }}
.card .row {{ font-size:0.82rem; color:{GREY}; font-family:'IBM Plex Mono',monospace; }}

hr {{ border:none; border-top:1px solid {LINE}; margin:1.4rem 0; }}
.small {{ font-size:0.82rem; color:{GREY}; line-height:1.5; }}
[data-testid="stSidebar"] {{ background:{SOFT}; border-right:1px solid {LINE}; }}
</style>
""", unsafe_allow_html=True)


# ---- data ----
@st.cache_data
def load_data():
    with open(os.path.join(HERE, "app_data.json")) as f:
        return json.load(f)

D = load_data()
BATCHES = D["batches"]
TARGETS = D["targets"]

NAME = {"dissolution_av": "Dissolution", "tbl_av_hardness": "Hardness",
        "tbl_rsd_weight": "Weight RSD", "fct_tensile": "Tensile strength"}
UNIT = {"dissolution_av": "%", "tbl_av_hardness": "N", "tbl_rsd_weight": "%", "fct_tensile": "MPa"}

FOLD1_CODES = {1, 2, 4, 6, 7, 9, 13, 18}   # in-distribution test codes
OOD_CODE = 15
CONTROL_CODE = 25

def role_of(code):
    if code == OOD_CODE:
        return "out-of-distribution"
    if code == CONTROL_CODE:
        return "control (held out)"
    return "in-distribution"

# in-distribution reference: median calibrated std per target on fold-1 test codes
REF = {}
for t in TARGETS:
    vals = [b["std"][t] for b in BATCHES if b["code"] in FOLD1_CODES]
    REF[t] = float(np.median(vals))

def group_median_std(code, t):
    vals = [b["std"][t] for b in BATCHES if b["code"] == code]
    return float(np.median(vals)) if vals else float("nan")

IND_H = np.median([b["std"]["tbl_av_hardness"] for b in BATCHES if b["code"] in FOLD1_CODES])
CTRL_H = group_median_std(CONTROL_CODE, "tbl_av_hardness")
OOD_H = group_median_std(OOD_CODE, "tbl_av_hardness")


# ---- header ----
st.markdown('<div class="eyebrow">MT-TrajNet &middot; MSc research practicum</div>', unsafe_allow_html=True)
st.markdown("# Tablet quality prediction that flags what it hasn't seen")
st.markdown(
    '<p class="lede">A single deep model reads a tablet press\'s raw sensor trajectory and predicts four '
    'quality attributes at once, each with a calibrated uncertainty. The point of this demo: on products '
    'the model was never trained on, its uncertainty rises sharply, so a low-confidence batch is visible '
    'before anyone trusts the number.</p>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="metric"><div class="k">In-distribution</div>'
                f'<div class="v">{IND_H:.1f}<span class="u"> N</span></div>'
                f'<div class="s">median hardness uncertainty on seen products</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="metric"><div class="k">Control (unseen)</div>'
                f'<div class="v">{CTRL_H:.1f}<span class="u"> N</span></div>'
                f'<div class="s">unseen product, but inside the trained regime</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="metric"><div class="k">Out-of-distribution</div>'
                f'<div class="v" style="color:{AMBER}">{OOD_H:.1f}<span class="u"> N</span></div>'
                f'<div class="s">unseen high-hardness product (code 15)</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown(f'<div class="metric"><div class="k">Uncertainty ratio</div>'
                f'<div class="v" style="color:{AMBER}">{OOD_H/IND_H:.1f}&times;</div>'
                f'<div class="s">OOD vs in-distribution, control stays flat</div></div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ---- sidebar: batch picker ----
st.sidebar.markdown('<div class="eyebrow">Select a batch</div>', unsafe_allow_html=True)

group = st.sidebar.radio(
    "Product group",
    ["In-distribution (seen)", "Control (unseen, low regime)", "Out-of-distribution (unseen, high regime)"],
    index=0,
)
if group.startswith("In"):
    pool = [b for b in BATCHES if b["code"] in FOLD1_CODES]
elif group.startswith("Control"):
    pool = [b for b in BATCHES if b["code"] == CONTROL_CODE]
else:
    pool = [b for b in BATCHES if b["code"] == OOD_CODE]

labels = [f'batch {b["batch_id"]}  ·  code {b["code"]}' for b in pool]
pick = st.sidebar.selectbox("Batch", range(len(pool)), format_func=lambda i: labels[i])
B = pool[pick]

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown(
    f'<div class="small">Product code <b>{B["code"]}</b> &mdash; {role_of(B["code"])}.<br><br>'
    f'All values come from the fold-1 model, which never trained on code 15 or code 25. '
    f'Intervals are the model\'s calibrated 90% prediction intervals.</div>',
    unsafe_allow_html=True,
)


# ---- selected batch: role banner + uncertainty status ----
role = role_of(B["code"])
ratio_h = B["std"]["tbl_av_hardness"] / REF["tbl_av_hardness"]
elevated = ratio_h >= 2.0

if elevated:
    banner_c, banner_bg, banner_txt = AMBER, "#FBF1E4", "High model uncertainty — this batch looks unlike the training data"
else:
    banner_c, banner_bg, banner_txt = GREEN, "#E9F3EE", "Within expected uncertainty for a seen regime"

st.markdown(
    f'<div style="display:flex;align-items:center;gap:14px;margin-bottom:2px;">'
    f'<h3 style="margin:0;">Batch {B["batch_id"]}</h3>'
    f'<span class="badge" style="background:{banner_bg};color:{banner_c};">{role}</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div style="background:{banner_bg};border:1px solid {LINE};border-left:3px solid {banner_c};'
    f'border-radius:10px;padding:10px 14px;margin:8px 0 4px 0;color:{banner_c};font-size:0.9rem;">'
    f'{banner_txt}. &nbsp;Hardness uncertainty is {ratio_h:.1f}&times; the in-distribution median.</div>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)


# ---- interval plot helper ----
def interval_fig(t):
    pred = B["pred"][t]; lo = B["pi90_low"][t]; hi = B["pi90_high"][t]; true = B["true"][t]
    inside = lo <= true <= hi
    accent = GREEN if inside else AMBER

    span = hi - lo
    ax_lo = min(lo, true) - 0.25 * span
    ax_hi = max(hi, true) + 0.25 * span

    fig = go.Figure()
    # interval band
    fig.add_shape(type="rect", x0=lo, x1=hi, y0=-0.28, y1=0.28,
                  fillcolor=accent, opacity=0.14, line=dict(width=0))
    fig.add_shape(type="line", x0=lo, x1=hi, y0=0, y1=0, line=dict(color=accent, width=2))
    for xv in (lo, hi):
        fig.add_shape(type="line", x0=xv, x1=xv, y0=-0.16, y1=0.16, line=dict(color=accent, width=1.5))
    # prediction
    fig.add_trace(go.Scatter(x=[pred], y=[0], mode="markers",
                             marker=dict(symbol="diamond", size=13, color=STEEL,
                                         line=dict(color="white", width=1.5)),
                             hovertemplate=f"prediction %{{x:.2f}} {UNIT[t]}<extra></extra>", name="prediction"))
    # actual lab value
    fig.add_trace(go.Scatter(x=[true], y=[0], mode="markers",
                             marker=dict(symbol="circle", size=11, color=INK,
                                         line=dict(color="white", width=1.5)),
                             hovertemplate=f"actual %{{x:.2f}} {UNIT[t]}<extra></extra>", name="actual"))
    fig.update_layout(
        height=118, margin=dict(l=8, r=8, t=6, b=20), showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(range=[ax_lo, ax_hi], showgrid=False, zeroline=False,
                   tickfont=dict(family="IBM Plex Mono", size=11, color=GREY)),
        yaxis=dict(range=[-0.6, 0.6], showticklabels=False, showgrid=False, zeroline=False),
    )
    return fig, inside

st.markdown("#### Predictions and calibrated 90% intervals")
st.markdown('<div class="small">Blue diamond is the model\'s prediction, black dot is the actual lab value, '
            'the band is the 90% prediction interval. When the dot sits inside the band, the model\'s '
            'confidence held.</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

inside_count = 0
grid = st.columns(2)
for idx, t in enumerate(TARGETS):
    with grid[idx % 2]:
        pred = B["pred"][t]; lo = B["pi90_low"][t]; hi = B["pi90_high"][t]; true = B["true"][t]
        fig, inside = interval_fig(t)
        inside_count += int(inside)
        tag_c = GREEN if inside else AMBER
        tag = "actual inside interval" if inside else "actual outside interval"
        st.markdown(
            f'<div class="card"><div class="t">{NAME[t]}</div>'
            f'<div class="p">{pred:.1f} <span class="u">{UNIT[t]}</span></div>'
            f'<div class="row">90% interval {lo:.1f} to {hi:.1f} &nbsp;·&nbsp; actual {true:.1f}</div>'
            f'<div class="row" style="color:{tag_c};">{tag}</div></div>',
            unsafe_allow_html=True,
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

st.markdown(
    f'<div class="small">On this batch, the actual value fell inside the 90% interval for '
    f'<b>{inside_count} of 4</b> targets.</div>', unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)


# ---- distribution-shift chart ----
st.markdown("#### Why the flag works: uncertainty separates the regimes")
st.markdown('<div class="small">Each point is one batch\'s hardness uncertainty. Seen products and the unseen '
            'control sit low and together; the unseen high-hardness product (code 15) sits far above. '
            'Nothing tells the model which code is which — the uncertainty finds them.</div>',
            unsafe_allow_html=True)

def shift_fig():
    fig = go.Figure()
    groups = [("in-distribution", FOLD1_CODES, STEEL),
              ("control (unseen)", {CONTROL_CODE}, GREEN),
              ("out-of-distribution", {OOD_CODE}, AMBER)]
    for label, codes, col in groups:
        ys = [b["std"]["tbl_av_hardness"] for b in BATCHES if b["code"] in codes]
        fig.add_trace(go.Box(y=ys, name=label, boxpoints="all", jitter=0.5, pointpos=0,
                             marker=dict(color=col, size=5, opacity=0.55),
                             line=dict(color=col), fillcolor="rgba(0,0,0,0)",
                             hovertemplate="%{y:.1f} N<extra></extra>"))
    fig.update_layout(
        height=340, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis=dict(title="hardness uncertainty (N)", gridcolor=LINE,
                   tickfont=dict(family="IBM Plex Mono", size=11, color=GREY),
                   titlefont=dict(family="Inter", size=12, color=GREY)),
        xaxis=dict(tickfont=dict(family="IBM Plex Mono", size=11, color=INK)),
    )
    return fig

st.plotly_chart(shift_fig(), use_container_width=True, config={"displayModeBar": False})

st.markdown("<hr>", unsafe_allow_html=True)


# ---- model card / honesty ----
with st.expander("Model card and honest limitations", expanded=False):
    mc1, mc2 = st.columns(2)
    with mc1:
        st.markdown(f"""
<div class="small">
<b>Model.</b> MT-TrajNet — a dilated temporal convolutional encoder (four residual blocks) with four
evidential regression heads, 384,400 parameters. One forward pass returns a prediction and an uncertainty
per target.<br><br>
<b>This demo.</b> Fold 1 of the stratified cross-validation, fixed seed 42, Tesla T4. Product codes 15 and 25
were never in this model's training. Data integrity is pinned by an MD5 hash on the input file.
</div>""", unsafe_allow_html=True)
    with mc2:
        st.markdown(f"""
<div class="small">
<b>What is honest to read here.</b> The hardness story is the strongest: uncertainty rises about
{OOD_H/IND_H:.1f}&times; on the unseen high-hardness product while the unseen control stays flat.<br><br>
<b>What to be careful about.</b> These are single-fold calibrated intervals, not the pooled thesis numbers.
Tensile is the least well-calibrated target on this fold. The regime flag is an illustrative aid based on the
model's own uncertainty, not a validated release rule.
</div>""", unsafe_allow_html=True)

st.markdown(
    f'<div class="small" style="text-align:center;color:{GREY};margin-top:1rem;">'
    f'MT-TrajNet &middot; trajectory-native tablet quality prediction with calibrated dual uncertainty &middot; '
    f'seed 42 &middot; fold 1</div>', unsafe_allow_html=True)
