"""Reusable interface pieces.

Every page builds from these so spacing, type and colour stay consistent.
"""

import streamlit as st

from config import theme as T


def page_header(eyebrow: str, title: str, lede: str = "") -> None:
    """Standard page opening: small label, title, optional intro paragraph."""
    st.markdown(f'<div class="eyebrow">{eyebrow}</div>', unsafe_allow_html=True)
    st.markdown(f"# {title}")
    if lede:
        st.markdown(f'<p class="lede">{lede}</p>', unsafe_allow_html=True)


def section(number: str, title: str) -> None:
    """Ruled section divider with a small index."""
    st.markdown(
        f'<div class="section-head"><span class="n">{number}</span>'
        f'<span class="t">{title}</span></div>',
        unsafe_allow_html=True,
    )


def kpi(label: str, value: str, unit: str = "", sub: str = "", accent: str = None) -> str:
    """KPI card markup. Render inside a column with st.markdown(..., unsafe_allow_html=True)."""
    colour = f"color:{accent};" if accent else ""
    unit_html = f'<span class="u"> {unit}</span>' if unit else ""
    return (
        f'<div class="kpi"><div class="k">{label}</div>'
        f'<div class="v" style="{colour}">{value}{unit_html}</div>'
        f'<div class="s">{sub}</div></div>'
    )


def kpi_row(items) -> None:
    """Render a row of KPI cards. items: list of dicts accepted by kpi()."""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            st.markdown(kpi(**item), unsafe_allow_html=True)


def question_card(tag: str, heading: str, body: str, accent: str = None) -> str:
    """Research-question card with its headline answer."""
    border = f"border-top-color:{accent};" if accent else ""
    return (
        f'<div class="qcard" style="{border}"><div class="q">{tag}</div>'
        f'<div class="h">{heading}</div><div class="b">{body}</div></div>'
    )


def callout(text: str, kind: str = "info") -> None:
    """Coloured note. kind: info | good | warn."""
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "nsig") -> str:
    """Inline badge. kind: sig | nsig | ood."""
    return f'<span class="badge {kind}">{text}</span>'


def small(text: str) -> None:
    st.markdown(f'<div class="small">{text}</div>', unsafe_allow_html=True)


def spacer(height: str = "1rem") -> None:
    st.markdown(f'<div style="height:{height}"></div>', unsafe_allow_html=True)


def thesis_figure(path: str, caption: str, expanded: bool = True) -> None:
    """Show a figure exactly as it appears in the thesis, tucked into an expander."""
    with st.expander("Figure as published in the thesis", expanded=expanded):
        st.image(path, caption=caption, use_container_width=True)
