"""
MT-TrajNet — research demonstration application.

Entry point and navigation. Each section lives in its own module under pages_/ and
exposes a render() function, so page order and grouping stay under explicit control
rather than relying on Streamlit's automatic page discovery.

Run with:  streamlit run app.py
"""

import streamlit as st

from config import theme as T

st.set_page_config(
    page_title="MT-TrajNet — trajectory-native tablet quality prediction",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

T.inject_css()

from pages_ import (          # noqa: E402  (import after page config, as Streamlit requires)
    overview,
    data as data_page,
    model as model_page,
    results as results_page,
    uncertainty as uncertainty_page,
    interpretability as interpretability_page,
    explorer as explorer_page,
    limitations as limitations_page,
)

SECTIONS = {
    "Overview": overview,
    "The data": data_page,
    "The model": model_page,
    "Results": results_page,
    "Uncertainty": uncertainty_page,
    "Interpretability": interpretability_page,
    "Batch explorer": explorer_page,
    "Limitations": limitations_page,
}

GROUPS = [
    ("Introduction", ["Overview"]),
    ("Study design", ["The data", "The model"]),
    ("Findings", ["Results", "Uncertainty", "Interpretability"]),
    ("Demonstration", ["Batch explorer"]),
    ("Closing", ["Limitations"]),
]


def sidebar() -> str:
    with st.sidebar:
        st.markdown(
            f'<div style="font-family:{T.FONT_DISPLAY};font-size:1.05rem;font-weight:600;'
            f'letter-spacing:-0.01em;">MT-TrajNet</div>'
            f'<div style="font-family:{T.FONT_MONO};font-size:0.68rem;letter-spacing:0.12em;'
            f'text-transform:uppercase;color:{T.GREY};margin-top:2px;">Research demonstration</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<hr style="border:none;border-top:1px solid {T.LINE};margin:0.9rem 0 0.4rem 0;">',
            unsafe_allow_html=True,
        )

        names = [n for _, group in GROUPS for n in group]
        labels = {}
        for group_title, group_names in GROUPS:
            for n in group_names:
                labels[n] = n

        if "section" not in st.session_state:
            st.session_state.section = "Overview"

        for group_title, group_names in GROUPS:
            st.markdown(
                f'<div style="font-family:{T.FONT_MONO};font-size:0.62rem;letter-spacing:0.14em;'
                f'text-transform:uppercase;color:{T.GREY};margin:0.9rem 0 0.3rem 0;">{group_title}</div>',
                unsafe_allow_html=True,
            )
            for n in group_names:
                active = st.session_state.section == n
                if st.button(
                    n,
                    key=f"nav_{n}",
                    use_container_width=True,
                    type="primary" if active else "secondary",
                ):
                    st.session_state.section = n
                    st.rerun()

        st.markdown(
            f'<hr style="border:none;border-top:1px solid {T.LINE};margin:1.2rem 0 0.6rem 0;">',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div style="font-family:{T.FONT_MONO};font-size:0.68rem;color:{T.GREY};line-height:1.7;">'
            f"MSc Artificial Intelligence<br>National College of Ireland<br>"
            f'<span style="color:{T.LINE}">——</span><br>'
            f"seed 42 · fold 1 · Tesla T4</div>",
            unsafe_allow_html=True,
        )

    return st.session_state.section


def main() -> None:
    section = sidebar()
    SECTIONS[section].render()


if __name__ == "__main__":
    main()
