"""
core/charts.py
--------------
All Plotly chart builders. Each function returns a go.Figure.
No Streamlit imports here — purely data → figure.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Colour palette
TIER_COLORS  = {"Fleet-wide": "#5DCAA5", "Common": "#85B7EB", "Isolated": "#D3D1C7"}
AC_PALETTE   = ["#5DCAA5", "#85B7EB", "#FAC775", "#D85A30", "#534AB7", "#D4537E"]

DMG_COLORS = {
    "corroded":        "#D85A30",
    "broken":          "#534AB7",
    "cracked":         "#E24B4A",
    "paint peel off":  "#378ADD",
    "eroded":          "#BA7517",
    "torn":            "#1D9E75",
    "missing":         "#888780",
    "punctured":       "#D4537E",
    "wrinkled":        "#639922",
    "dirty":           "#5DCAA5",
    "overplay":        "#AFA9EC",
    "robbed":          "#B4B2A9",
    "nicked":          "#FAC775",
    "dented":          "#F0997B",
    "scratched":       "#85B7EB",
}


def _clean_layout(fig, height=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Arial", size=11),
        margin=dict(l=10, r=10, t=40, b=10),
        height=height,
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.06)", zeroline=False)
    return fig


# ── Scatter map ────────────────────────────────────────────────────────────
def scatter_map(df: pd.DataFrame, X_2d: np.ndarray) -> go.Figure:
    plot_df = df.copy()
    plot_df["x"] = X_2d[:, 0]
    plot_df["y"] = X_2d[:, 1]
    plot_df["hover"] = plot_df["Description"].str[:100]
    plot_df["cluster_str"] = plot_df["cluster_label"].fillna("unclustered")

    fig = px.scatter(
        plot_df,
        x="x", y="y",
        color="cluster_str",
        symbol="project",
        hover_data={"hover": True, "project": True, "cluster_str": True,
                    "x": False, "y": False},
        title="NRC similarity map — each dot is one NRC",
        labels={"cluster_str": "Cluster", "hover": "Description", "project": "AC Reg"},
        height=520,
        template="plotly_white",
    )
    fig.update_traces(marker=dict(size=7, opacity=0.75))
    fig.update_layout(legend=dict(orientation="v", x=1.02, xanchor="left", font_size=10))
    return _clean_layout(fig, 520)


# ── Top-N ranked defects bar ───────────────────────────────────────────────
def ranked_bar(scores: pd.DataFrame, top_n: int = 25) -> go.Figure:
    top = scores.head(top_n).copy()
    top["label"] = top["location"] + " — " + top["damage_type"]
    top["color"] = top["tier"].map(TIER_COLORS)

    fig = go.Figure(go.Bar(
        x=top["score"],
        y=top["label"],
        orientation="h",
        marker_color=top["color"].tolist(),
        text=top["score"].apply(lambda x: f"{x:.3f}"),
        textposition="outside",
        customdata=top[["total_count", "avg_mhrs", "tier"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Score: %{x:.3f}<br>"
            "NRCs: %{customdata[0]}<br>"
            "Avg hrs: %{customdata[1]:.1f}<br>"
            "Tier: %{customdata[2]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=f"Top {top_n} defects by weighted score",
        yaxis=dict(autorange="reversed", tickfont_size=11),
        xaxis=dict(range=[0, top["score"].max() * 1.18]),
        height=max(400, top_n * 28),
        showlegend=False,
    )
    return _clean_layout(fig)


# ── Fleet-wide grouped bar ─────────────────────────────────────────────────
def fleet_grouped_bar(scores: pd.DataFrame, projects: list) -> go.Figure:
    fleet = scores[scores["tier"] == "Fleet-wide"].copy()
    fleet["label"] = fleet["location"] + " — " + fleet["damage_type"]

    count_cols = [c for c in fleet.columns if c.startswith("count_")]
    fig = go.Figure()
    for i, col in enumerate(count_cols):
        ac = col.replace("count_", "")
        fig.add_trace(go.Bar(
            name=ac,
            x=fleet["label"],
            y=fleet[col],
            marker_color=AC_PALETTE[i % len(AC_PALETTE)],
            text=fleet[col],
            textposition="outside",
        ))
    fig.update_layout(
        barmode="group",
        title="Fleet-wide defects — NRC count per aircraft",
        xaxis=dict(tickangle=-35, tickfont_size=10),
        legend=dict(orientation="h", y=1.08, x=0),
        height=420,
    )
    return _clean_layout(fig)


# ── Frequency heatmap ──────────────────────────────────────────────────────
def frequency_heatmap(scores: pd.DataFrame, top_n: int = 30) -> go.Figure:
    top = scores.head(top_n).copy()
    top["label"] = top["location"] + " — " + top["damage_type"]

    count_cols = [c for c in top.columns if c.startswith("count_")]
    ac_labels  = [c.replace("count_", "") for c in count_cols]

    z      = top[count_cols].values
    y_lbls = top["label"].tolist()

    fig = go.Figure(go.Heatmap(
        z=z,
        x=ac_labels,
        y=y_lbls,
        colorscale="Blues",
        text=z,
        texttemplate="%{text}",
        showscale=True,
        hovertemplate="<b>%{y}</b><br>%{x}: %{z} NRCs<extra></extra>",
    ))
    fig.update_layout(
        title=f"Defect frequency — top {top_n} × aircraft",
        height=max(500, top_n * 26),
        margin=dict(l=280, r=30, t=50, b=40),
    )
    fig.update_yaxes(tickfont_size=10, autorange="reversed")
    return _clean_layout(fig)


# ── Manhour impact bar ────────────────────────────────────────────────────
def manhour_bar(scores: pd.DataFrame, top_n: int = 20) -> go.Figure:
    top = scores.nlargest(top_n, "avg_mhrs").copy()
    top["label"] = top["location"] + " — " + top["damage_type"]
    top["color"] = top["avg_mhrs"].apply(
        lambda h: "#A32D2D" if h > 30 else "#BA7517" if h > 15 else "#85B7EB"
    )

    fig = go.Figure(go.Bar(
        x=top["avg_mhrs"],
        y=top["label"],
        orientation="h",
        marker_color=top["color"].tolist(),
        text=top["avg_mhrs"].apply(lambda x: f"{x:.1f}h"),
        textposition="outside",
        customdata=top[["total_count", "tier"]].values,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Avg manhours: %{x:.1f}h<br>"
            "NRCs: %{customdata[0]}<br>"
            "Tier: %{customdata[1]}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=f"Top {top_n} defects by avg actual manhours",
        yaxis=dict(autorange="reversed", tickfont_size=11),
        xaxis_title="Avg actual manhours",
        height=max(400, top_n * 28),
        showlegend=False,
    )
    return _clean_layout(fig)


# ── Score component stacked bar ───────────────────────────────────────────
def score_components_bar(scores: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top = scores.head(top_n).copy()
    top["label"] = (top["location"] + " — " + top["damage_type"]).str[:40]

    # Reconstruct component columns when raw values weren't saved (e.g. loaded from Supabase)
    has_raw = "presence_raw" in top.columns and "freq_norm" in top.columns and "mhrs_norm" in top.columns
    if not has_raw:
        # Back-calculate from what's available:
        # presence_raw = projects_count / max(projects_count)  — fraction of ACs affected
        # freq_norm    = total_count / max(total_count)         — relative NRC frequency
        # mhrs_norm    = avg_mhrs / max(avg_mhrs)              — relative manhour cost
        max_proj  = top["projects_count"].max() if "projects_count" in top.columns and top["projects_count"].max() > 0 else 1
        max_count = top["total_count"].max()    if "total_count"    in top.columns and top["total_count"].max()    > 0 else 1
        max_mhrs  = top["avg_mhrs"].max()       if "avg_mhrs"       in top.columns and top["avg_mhrs"].max()       > 0 else 1
        top["presence_raw"] = (top["projects_count"] / max_proj)  if "projects_count" in top.columns else 0
        top["freq_norm"]    = (top["total_count"]    / max_count)  if "total_count"    in top.columns else 0
        top["mhrs_norm"]    = (top["avg_mhrs"]       / max_mhrs)   if "avg_mhrs"       in top.columns else 0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Presence (50%)",
        x=top["label"],
        y=(top["presence_raw"] * 0.50).round(3),
        marker_color="#5DCAA5",
    ))
    fig.add_trace(go.Bar(
        name="Frequency (30%)",
        x=top["label"],
        y=(top["freq_norm"] * 0.30).round(3),
        marker_color="#85B7EB",
    ))
    fig.add_trace(go.Bar(
        name="Manhours (20%)",
        x=top["label"],
        y=(top["mhrs_norm"] * 0.20).round(3),
        marker_color="#FAC775",
    ))
    fig.update_layout(
        barmode="stack",
        title=f"Score breakdown — top {top_n}",
        xaxis=dict(tickangle=-35, tickfont_size=10),
        legend=dict(orientation="h", y=1.08),
        height=380,
    )
    return _clean_layout(fig)


# ── Tier donut ────────────────────────────────────────────────────────────
def tier_donut(scores: pd.DataFrame) -> go.Figure:
    counts = scores["tier"].value_counts()
    fig = go.Figure(go.Pie(
        labels=counts.index,
        values=counts.values,
        hole=0.6,
        marker_colors=[TIER_COLORS.get(t, "#888") for t in counts.index],
        textinfo="label+value",
        hovertemplate="%{label}: %{value}<extra></extra>",
    ))
    fig.update_layout(title="Defect tier breakdown", height=320,
                      legend=dict(orientation="h", y=-0.1))
    return _clean_layout(fig)


# ── Damage type distribution ──────────────────────────────────────────────
def damage_distribution(df: pd.DataFrame) -> go.Figure:
    counts = (
        df[df["damage_type"] != "other"]["damage_type"]
        .value_counts()
        .head(12)
        .reset_index()
    )
    counts.columns = ["damage_type", "count"]
    counts["color"] = counts["damage_type"].map(DMG_COLORS).fillna("#888")

    fig = go.Figure(go.Bar(
        x=counts["count"],
        y=counts["damage_type"],
        orientation="h",
        marker_color=counts["color"].tolist(),
        text=counts["count"],
        textposition="outside",
    ))
    fig.update_layout(
        title="NRC count by damage type",
        yaxis=dict(autorange="reversed", tickfont_size=11),
        height=380,
        showlegend=False,
    )
    return _clean_layout(fig)


# ── Cluster size distribution ─────────────────────────────────────────────
def cluster_size_dist(df: pd.DataFrame) -> go.Figure:
    sizes = (
        df[df["cluster_id"] != -1]
        .groupby("cluster_label")
        .size()
        .reset_index(name="count")
    )
    bins   = [0, 5, 10, 20, 50, 9999]
    labels = ["= 5", "6–10", "11–20", "21–50", "50+"]
    sizes["bin"] = pd.cut(sizes["count"], bins=bins, labels=labels)
    dist = sizes["bin"].value_counts().reindex(labels).reset_index()
    dist.columns = ["bin", "count"]

    fig = go.Figure(go.Bar(
        x=dist["bin"],
        y=dist["count"],
        marker_color="#534AB7",
        text=dist["count"],
        textposition="outside",
    ))
    fig.update_layout(title="Cluster size distribution", height=300, showlegend=False)
    return _clean_layout(fig)
