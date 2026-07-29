"""Cached access to the application's data files.

Two files are read:
  data/results_bundle.json  - study-level results, assembled from the thesis result
                              files by build_bundle.py
  data/app_data.json        - per-batch predictions, calibrated uncertainty and
                              intervals for 378 held-out batches
"""

import json
import os
from functools import lru_cache

import streamlit as st

APP_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(APP_DIR, "data")
FIG_DIR = os.path.join(APP_DIR, "assets", "figures")

TARGETS = ["dissolution_av", "tbl_av_hardness", "tbl_rsd_weight", "fct_tensile"]

TARGET_NAME = {
    "dissolution_av": "Dissolution",
    "tbl_av_hardness": "Hardness",
    "tbl_rsd_weight": "Weight RSD",
    "fct_tensile": "Tensile strength",
}

TARGET_UNIT = {
    "dissolution_av": "%",
    "tbl_av_hardness": "N",
    "tbl_rsd_weight": "%",
    "fct_tensile": "MPa",
}

# sensor channels in the order the model receives them
CHANNEL_NAME = {
    "fom": "Filling device speed",
    "cyl_pre": "Cylinder pre-compression",
    "tbl_speed": "Tablet press speed",
    "pre_comp": "Pre-compression force",
    "SREL": "Slow release indicator",
    "tbl_fill": "Tablet fill (die fill)",
    "cyl_main": "Cylinder main compression",
    "ejection": "Ejection force",
    "main_comp": "Main compression force",
    "stiffness": "Stiffness",
}

# fold-1 groupings used throughout the app
FOLD1_CODES = {1, 2, 4, 6, 7, 9, 13, 18}
OOD_CODE = 15
CONTROL_CODE = 25


@st.cache_data(show_spinner=False)
def load_bundle():
    """Study-level results assembled from the thesis result files."""
    with open(os.path.join(DATA_DIR, "results_bundle.json")) as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_batches():
    """Per-batch predictions, uncertainty and calibrated intervals."""
    with open(os.path.join(DATA_DIR, "app_data.json")) as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def batch_index():
    """Batches grouped by role, for the explorer's selector."""
    data = load_batches()
    groups = {"in_distribution": [], "control": [], "ood": []}
    for row in data["batches"]:
        code = row["code"]
        if code == OOD_CODE:
            groups["ood"].append(row)
        elif code == CONTROL_CODE:
            groups["control"].append(row)
        else:
            groups["in_distribution"].append(row)
    return groups


@st.cache_data(show_spinner=False)
def load_trajectories():
    """Downsampled sensor curves for a representative subset of batches."""
    with open(os.path.join(DATA_DIR, "trajectories.json")) as fh:
        return json.load(fh)


def role_of(code: int) -> str:
    if code == OOD_CODE:
        return "out-of-distribution"
    if code == CONTROL_CODE:
        return "control, held out"
    return "in-distribution"


@lru_cache(maxsize=None)
def figure_path(name: str) -> str:
    """Absolute path to a thesis figure, e.g. figure_path('fig7_calibration.png')."""
    return os.path.join(FIG_DIR, name)


def fmt(value, digits: int = 3) -> str:
    """Format a number for display, trimming trailing zeros sensibly."""
    if value is None:
        return "—"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}"
    return f"{value:.{digits}f}"
