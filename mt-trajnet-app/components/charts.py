"""Chart builders.

All figures share one styling function so fonts, gridlines, margins and hover
behaviour are identical across the application.
"""

import plotly.graph_objects as go

from config import theme as T


def style(fig, height: int = 340, xtitle: str = "", ytitle: str = "", legend: bool = False):
    """Apply the shared look to a Plotly figure."""
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=16, b=12),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=legend,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            font=dict(family="Inter", size=11, color=T.GREY),
        ),
        hoverlabel=dict(font=dict(family="IBM Plex Mono", size=11), bgcolor="white"),
        font=dict(family="Inter", size=12, color=T.INK),
    )
    fig.update_xaxes(
        title=dict(text=xtitle, font=dict(family="Inter", size=12, color=T.GREY)),
        gridcolor=T.LINE, zeroline=False, linecolor=T.LINE,
        tickfont=dict(family="IBM Plex Mono", size=11, color=T.GREY),
    )
    fig.update_yaxes(
        title=dict(text=ytitle, font=dict(family="Inter", size=12, color=T.GREY)),
        gridcolor=T.LINE, zeroline=False, linecolor=T.LINE,
        tickfont=dict(family="IBM Plex Mono", size=11, color=T.GREY),
    )
    return fig


def grouped_bars(categories, series, xtitle="", ytitle="", height=360, horizontal=False):
    """Grouped bar chart. series: list of (name, values, colour)."""
    fig = go.Figure()
    for name, values, colour in series:
        if horizontal:
            fig.add_trace(go.Bar(
                y=categories, x=values, name=name, orientation="h",
                marker_color=colour,
                hovertemplate="%{y}: %{x:.3f}<extra>" + name + "</extra>",
            ))
        else:
            fig.add_trace(go.Bar(
                x=categories, y=values, name=name, marker_color=colour,
                hovertemplate="%{x}: %{y:.3f}<extra>" + name + "</extra>",
            ))
    fig.update_layout(bargap=0.28, bargroupgap=0.08)
    return style(fig, height=height, xtitle=xtitle, ytitle=ytitle, legend=len(series) > 1)


def interval_strip(pred, lo, hi, actual, unit, height=110):
    """A single prediction with its interval and the actual laboratory value."""
    inside = lo <= actual <= hi
    accent = T.GREEN if inside else T.AMBER
    span = max(hi - lo, 1e-9)
    pad = 0.28 * span
    axis_lo = min(lo, actual) - pad
    axis_hi = max(hi, actual) + pad

    fig = go.Figure()
    fig.add_shape(type="rect", x0=lo, x1=hi, y0=-0.3, y1=0.3,
                  fillcolor=accent, opacity=0.13, line=dict(width=0))
    fig.add_shape(type="line", x0=lo, x1=hi, y0=0, y1=0, line=dict(color=accent, width=2))
    for x in (lo, hi):
        fig.add_shape(type="line", x0=x, x1=x, y0=-0.17, y1=0.17,
                      line=dict(color=accent, width=1.6))
    fig.add_trace(go.Scatter(
        x=[pred], y=[0], mode="markers", name="prediction",
        marker=dict(symbol="diamond", size=13, color=T.STEEL,
                    line=dict(color="white", width=1.6)),
        hovertemplate=f"prediction %{{x:.2f}} {unit}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[actual], y=[0], mode="markers", name="laboratory value",
        marker=dict(symbol="circle", size=11, color=T.INK,
                    line=dict(color="white", width=1.6)),
        hovertemplate=f"laboratory %{{x:.2f}} {unit}<extra></extra>",
    ))
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=6, b=24),
                      showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
    fig.update_xaxes(range=[axis_lo, axis_hi], showgrid=False, zeroline=False,
                     tickfont=dict(family="IBM Plex Mono", size=11, color=T.GREY))
    fig.update_yaxes(range=[-0.65, 0.65], showticklabels=False, showgrid=False, zeroline=False)
    return fig, inside


def scatter_with_fit(x, y, labels, xtitle, ytitle, height=380, colour=None):
    """Scatter with a least-squares line, used for the hardness law."""
    import numpy as np

    colour = colour or T.STEEL
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    fig = go.Figure()
    slope, intercept = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    fig.add_trace(go.Scatter(
        x=xs, y=slope * xs + intercept, mode="lines",
        line=dict(color=T.GREY, width=1.4, dash="dot"),
        hoverinfo="skip", name="least-squares fit",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers", text=labels,
        marker=dict(size=9, color=colour, opacity=0.75,
                    line=dict(color="white", width=1)),
        hovertemplate="code %{text}<br>%{x:.1f}, %{y:.2f}<extra></extra>",
        name="product code",
    ))
    return style(fig, height=height, xtitle=xtitle, ytitle=ytitle)


def box_by_group(groups, ytitle, height=360):
    """Distribution comparison. groups: list of (label, values, colour)."""
    fig = go.Figure()
    for label, values, colour in groups:
        fig.add_trace(go.Box(
            y=values, name=label, boxpoints="all", jitter=0.45, pointpos=0,
            marker=dict(color=colour, size=5, opacity=0.5),
            line=dict(color=colour), fillcolor="rgba(0,0,0,0)",
            hovertemplate="%{y:.2f}<extra>" + label + "</extra>",
        ))
    return style(fig, height=height, ytitle=ytitle)


def faceted_bars(target_names, series, ytitle="RMSE", height=300):
    """One small panel per target, so targets on very different scales stay readable.

    series: list of (name, values, colour) where values align with target_names.
    """
    from plotly.subplots import make_subplots

    n = len(target_names)
    fig = make_subplots(rows=1, cols=n, subplot_titles=target_names,
                        horizontal_spacing=0.085)

    for col in range(1, n + 1):
        for name, values, colour in series:
            fig.add_trace(
                go.Bar(
                    x=[name], y=[values[col - 1]], marker_color=colour, name=name,
                    showlegend=(col == 1),
                    hovertemplate="%{y:.3f}<extra>" + name + "</extra>",
                    width=0.62,
                ),
                row=1, col=col,
            )
        fig.update_xaxes(showticklabels=False, showgrid=False, linecolor=T.LINE,
                         row=1, col=col)
        fig.update_yaxes(gridcolor=T.LINE, zeroline=False, linecolor=T.LINE,
                         tickfont=dict(family="IBM Plex Mono", size=10, color=T.GREY),
                         row=1, col=col)

    fig.update_yaxes(title=dict(text=ytitle, font=dict(family="Inter", size=11, color=T.GREY)),
                     row=1, col=1)
    fig.update_layout(
        height=height, margin=dict(l=14, r=24, t=58, b=14),
        plot_bgcolor="white", paper_bgcolor="white", barmode="group", bargap=0.3,
        font=dict(family="Inter", size=11, color=T.INK),
        legend=dict(orientation="h", yanchor="bottom", y=1.14, xanchor="left", x=0,
                    font=dict(family="Inter", size=11, color=T.GREY)),
        hoverlabel=dict(font=dict(family="IBM Plex Mono", size=11), bgcolor="white"),
    )
    for ann in fig.layout.annotations:
        ann.font = dict(family="IBM Plex Mono", size=10, color=T.GREY)
    return fig
