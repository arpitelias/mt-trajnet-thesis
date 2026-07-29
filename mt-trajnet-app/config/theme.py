"""Design tokens and global styling.

One place defines every colour, font and spacing value used in the application, so the
interface stays consistent and matches the figures in the thesis.
"""

import streamlit as st

# --- surfaces -------------------------------------------------------------
INK = "#141A21"          # primary text, deep and slightly blue
PAPER = "#F4F3EF"        # page background, warm paper rather than white
CARD = "#FFFFFF"         # card surface
SOFT = "#E9E7E0"         # secondary surface
LINE = "#D8D5CC"         # hairline borders
GREY = "#5C636E"         # secondary text
MUTED = "#8A9099"        # tertiary text

# --- sidebar (dark, for contrast against the paper content area) ----------
NAVY = "#161D26"
NAVY_SOFT = "#1F2833"
NAVY_LINE = "#2C3742"
NAVY_TEXT = "#C4CCD6"
NAVY_MUTED = "#79848F"

# --- accents --------------------------------------------------------------
STEEL = "#1F5C8B"        # primary accent, model predictions
GREEN = "#1F6F55"        # in-distribution, good calibration
AMBER = "#AD6425"        # out-of-distribution, caution
PLUM = "#57457A"         # secondary comparison series
CLAY = "#96452B"         # baselines

# --- tinted backgrounds ---------------------------------------------------
GREEN_BG = "#E4EFE9"
AMBER_BG = "#F6EADC"
STEEL_BG = "#E5EDF4"

SERIES = [STEEL, GREEN, AMBER, PLUM, CLAY, GREY]

FONT_DISPLAY = "'Space Grotesk', sans-serif"
FONT_BODY = "'Inter', sans-serif"
FONT_MONO = "'IBM Plex Mono', monospace"


def inject_css() -> None:
    """Load fonts and apply global styles. Call once, at app start."""
    st.markdown(
        f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

.stApp {{ background:{PAPER}; }}
html, body, [class*="css"], p, li, span, div {{ font-family:{FONT_BODY}; color:{INK}; }}
h1, h2, h3, h4, h5 {{ font-family:{FONT_DISPLAY}; color:{INK}; letter-spacing:-0.015em; }}
h1 {{ font-size:2.05rem; font-weight:700; line-height:1.2; margin-bottom:0.2rem; }}
h2 {{ font-size:1.4rem; font-weight:600; margin-top:0.4rem; }}
h3 {{ font-size:1.08rem; font-weight:600; }}

/* generous top padding so the eyebrow label is never clipped */
.block-container {{ padding-top:4rem !important; padding-bottom:5rem; max-width:1180px; }}

/* ---------------- sidebar: dark, for contrast ---------------- */
[data-testid="stSidebar"] {{ background:{NAVY}; border-right:1px solid {NAVY_LINE}; }}
[data-testid="stSidebar"] .block-container {{ padding-top:2rem; }}
[data-testid="stSidebar"] * {{ color:{NAVY_TEXT}; }}
[data-testid="stSidebar"] hr {{ border-color:{NAVY_LINE} !important; }}

/* sidebar buttons */
[data-testid="stSidebar"] .stButton > button {{
  background:transparent; color:{NAVY_TEXT}; border:1px solid {NAVY_LINE};
  border-radius:9px; font-family:{FONT_BODY}; font-size:0.9rem; font-weight:500;
  padding:0.4rem 0.75rem; text-align:left; justify-content:flex-start;
  transition:background 120ms ease, border-color 120ms ease;
}}
[data-testid="stSidebar"] .stButton > button:hover {{
  background:{NAVY_SOFT}; border-color:{MUTED}; color:#FFFFFF;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {{
  background:{STEEL}; border-color:{STEEL}; color:#FFFFFF; font-weight:600;
}}
[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {{
  background:#26699B; border-color:#26699B;
}}

/* ---------------- typography helpers ---------------- */
.eyebrow {{
  font-family:{FONT_MONO}; font-size:0.7rem; letter-spacing:0.18em;
  text-transform:uppercase; color:{STEEL}; margin-bottom:0.5rem; font-weight:500;
  line-height:1.7;
}}
.lede {{ color:{GREY}; font-size:1.0rem; line-height:1.65; max-width:70ch; }}
.small {{ font-size:0.81rem; color:{GREY}; line-height:1.6; }}
.mono {{ font-family:{FONT_MONO}; }}

/* section header with rule */
.section-head {{
  display:flex; align-items:baseline; gap:0.7rem;
  border-bottom:1px solid {LINE}; padding-bottom:0.55rem; margin:2.4rem 0 1.2rem 0;
}}
.section-head .n {{ font-family:{FONT_MONO}; font-size:0.72rem; color:{STEEL}; letter-spacing:0.1em; }}
.section-head .t {{ font-family:{FONT_DISPLAY}; font-size:1.1rem; font-weight:600; color:{INK}; }}

/* ---------------- cards ---------------- */
.kpi {{
  background:{CARD}; border:1px solid {LINE}; border-radius:12px;
  padding:1.05rem 1.15rem; height:100%;
  box-shadow:0 1px 2px rgba(20,26,33,0.045), 0 1px 1px rgba(20,26,33,0.03);
}}
.kpi .k {{
  font-family:{FONT_MONO}; font-size:0.66rem; letter-spacing:0.11em; text-transform:uppercase;
  color:{MUTED}; margin-bottom:0.45rem;
}}
.kpi .v {{ font-family:{FONT_DISPLAY}; font-size:1.8rem; font-weight:600; line-height:1.12; color:{INK}; }}
.kpi .v .u {{ font-size:0.92rem; font-weight:500; color:{GREY}; }}
.kpi .s {{ font-size:0.78rem; color:{GREY}; line-height:1.5; margin-top:0.4rem; }}

.qcard {{
  background:{CARD}; border:1px solid {LINE}; border-radius:12px;
  padding:1.05rem 1.15rem; height:100%; border-top:3px solid {STEEL};
  box-shadow:0 1px 2px rgba(20,26,33,0.045), 0 1px 1px rgba(20,26,33,0.03);
}}
.qcard .q {{ font-family:{FONT_MONO}; font-size:0.66rem; letter-spacing:0.11em; color:{STEEL}; text-transform:uppercase; }}
.qcard .h {{ font-family:{FONT_DISPLAY}; font-size:0.98rem; font-weight:600; margin:0.35rem 0 0.5rem 0; line-height:1.35; }}
.qcard .b {{ font-size:0.845rem; color:{GREY}; line-height:1.6; }}

/* ---------------- callouts ---------------- */
.callout {{
  border-radius:10px; padding:0.85rem 1.05rem; margin:0.75rem 0;
  font-size:0.875rem; line-height:1.6; border:1px solid {LINE};
}}
.callout.info {{ background:{STEEL_BG}; border-left:3px solid {STEEL}; color:{INK}; }}
.callout.good {{ background:{GREEN_BG}; border-left:3px solid {GREEN}; color:{INK}; }}
.callout.warn {{ background:{AMBER_BG}; border-left:3px solid {AMBER}; color:{INK}; }}

/* callout heading must sit on its own line */
.callout .t {{
  display:block; font-family:{FONT_MONO}; font-size:0.67rem; letter-spacing:0.11em;
  text-transform:uppercase; margin-bottom:0.4rem; font-weight:600;
}}
.callout.info .t {{ color:{STEEL}; }}
.callout.good .t {{ color:{GREEN}; }}
.callout.warn .t {{ color:{AMBER}; }}

/* ---------------- badges ---------------- */
.badge {{
  display:inline-block; padding:3px 10px; border-radius:6px;
  font-family:{FONT_MONO}; font-size:0.7rem; font-weight:500;
}}
.badge.sig {{ background:{GREEN_BG}; color:{GREEN}; }}
.badge.nsig {{ background:{SOFT}; color:{GREY}; }}
.badge.ood {{ background:{AMBER_BG}; color:{AMBER}; }}

/* ---------------- dataframes, expanders, charts ---------------- */
[data-testid="stDataFrame"] {{ border:1px solid {LINE}; border-radius:10px; overflow:hidden; }}

[data-testid="stExpander"] {{
  border:1px solid {LINE}; border-radius:10px; background:{CARD};
  box-shadow:0 1px 2px rgba(20,26,33,0.04);
}}
[data-testid="stExpander"] summary {{
  font-family:{FONT_MONO}; font-size:0.7rem; letter-spacing:0.1em;
  text-transform:uppercase; color:{GREY};
}}

/* charts sit on white so they read as figures */
[data-testid="stPlotlyChart"] {{
  background:{CARD}; border:1px solid {LINE}; border-radius:10px;
  padding:0.5rem 0.4rem 0.2rem 0.4rem;
  box-shadow:0 1px 2px rgba(20,26,33,0.04);
}}

[data-testid="stWidgetLabel"] p {{
  font-family:{FONT_MONO}; font-size:0.68rem; letter-spacing:0.1em;
  text-transform:uppercase; color:{MUTED};
}}

/* hide streamlit chrome */
#MainMenu {{ visibility:hidden; }}
footer {{ visibility:hidden; }}
[data-testid="stDecoration"] {{ display:none; }}
</style>
""",
        unsafe_allow_html=True,
    )
