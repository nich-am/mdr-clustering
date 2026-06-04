"""
app.py
------
Main Streamlit application for NRC Findings Clustering & EDA.

Run with:
    streamlit run app.py
"""

import io
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd

from core.pipeline import run_pipeline, build_excel
from core.charts import (
    scatter_map, ranked_bar, fleet_grouped_bar,
    frequency_heatmap, manhour_bar, score_components_bar,
    tier_donut, damage_distribution, cluster_size_dist,
)

# ── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NRC Findings Clustering",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 500; }
  .tier-fleet   { background:#E1F5EE; color:#0F6E56; padding:2px 10px;
                  border-radius:10px; font-size:12px; font-weight:500; }
  .tier-common  { background:#E6F1FB; color:#185FA5; padding:2px 10px;
                  border-radius:10px; font-size:12px; font-weight:500; }
  .tier-isolated{ background:#F1EFE8; color:#5F5E5A; padding:2px 10px;
                  border-radius:10px; font-size:12px; font-weight:500; }
  div[data-testid="stExpander"] { border: 0.5px solid #e0e0e0; border-radius:8px; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ NRC Clustering")
    st.markdown("---")

    st.markdown("### 1 · Upload NRC files")
    st.markdown(
        "Upload one Excel file per aircraft. "
        "Each file must have a **Description** column."
    )

    # Dynamic file uploader — up to 10 aircraft
    n_files = st.number_input("Number of aircraft", min_value=1, max_value=10, value=3)

    uploaded = {}
    for i in range(n_files):
        col1, col2 = st.columns([2, 3])
        with col1:
            ac_reg = st.text_input(
                f"AC reg #{i+1}",
                value=f"AC-{i+1:03d}",
                key=f"reg_{i}",
                label_visibility="collapsed",
                placeholder=f"e.g. PK-GL{chr(88+i)}",
            )
        with col2:
            fobj = st.file_uploader(
                f"File for {ac_reg}",
                type=["xlsx", "xls"],
                key=f"file_{i}",
                label_visibility="collapsed",
            )
        if fobj and ac_reg:
            uploaded[ac_reg] = fobj

    st.markdown("---")
    st.markdown("### 2 · Clustering settings")
    min_cluster = st.slider(
        "Min cluster size",
        min_value=3, max_value=20, value=5,
        help="Smaller = more clusters (may be noisier). Larger = fewer, tighter clusters.",
    )
    top_n_score = st.slider(
        "Top N defects to show",
        min_value=10, max_value=50, value=25,
    )

    st.markdown("---")
    run_btn = st.button("▶ Run pipeline", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown(
        "<small style='color:gray'>Built for GMF AeroAsia MRO NRC analysis. "
        "Powered by TF-IDF · UMAP · HDBSCAN.</small>",
        unsafe_allow_html=True,
    )


# ── Main area ────────────────────────────────────────────────────────────────
st.title("NRC Findings Clustering & EDA")
st.markdown(
    "Identify recurring defect patterns across maintenance events "
    "of the same workscope and aircraft type."
)

# ── State ─────────────────────────────────────────────────────────────────────
if "results" not in st.session_state:
    st.session_state.results = None

# ── Run on button press ───────────────────────────────────────────────────────
if run_btn:
    if len(uploaded) < 1:
        st.error("Please upload at least one NRC Excel file.")
    else:
        with st.spinner("Running pipeline — preprocessing → vectorizing → clustering → scoring…"):
            try:
                results = run_pipeline(uploaded, min_cluster_size=min_cluster)
                st.session_state.results = results
                st.success(f"Done! {len(results['df'])} NRCs processed across {len(results['projects'])} aircraft.")
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                st.exception(e)

# ── Show results ─────────────────────────────────────────────────────────────
if st.session_state.results:
    R        = st.session_state.results
    df       = R["df"]
    scores   = R["scores"]
    X_2d     = R["X_2d"]
    projects = R["projects"]

    n_projects  = len(projects)
    n_clustered = (df["cluster_id"] != -1).sum()
    n_clusters  = df[df["cluster_id"] != -1]["cluster_id"].nunique()
    n_noise     = (df["cluster_id"] == -1).sum()
    n_fleet     = (scores["tier"] == "Fleet-wide").sum() if not scores.empty else 0
    n_common    = (scores["tier"] == "Common").sum()     if not scores.empty else 0

    # ── KPIs ──────────────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total NRCs",    f"{len(df):,}")
    c2.metric("Aircraft",      n_projects)
    c3.metric("Clusters found",n_clusters)
    c4.metric("NRCs clustered",f"{n_clustered:,}")
    c5.metric("Fleet-wide defects", n_fleet,
              help="Present in ALL aircraft events")
    c6.metric("One-off NRCs", n_noise)

    # ── AC registration legend ─────────────────────────────────────────────
    ac_colors = ["#5DCAA5", "#85B7EB", "#FAC775", "#D85A30", "#534AB7"]
    pills = " &nbsp; ".join(
        f"<span style='background:{ac_colors[i%len(ac_colors)]};color:#333;"
        f"padding:3px 10px;border-radius:8px;font-size:12px;font-weight:500'>"
        f"{p}</span>"
        for i, p in enumerate(projects)
    )
    st.markdown(f"**Aircraft in analysis:** {pills}", unsafe_allow_html=True)
    st.markdown("---")

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Ranked defects",
        "🌍 Fleet-wide",
        "🗺️ Similarity map",
        "🔥 Manhour impact",
        "📈 Score breakdown",
        "📋 Data tables",
    ])

    # ─────────────────────────────────────────────────────
    # TAB 1 — Ranked defects
    # ─────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Defects ranked by weighted commonality score")
        st.markdown(
            "Score = **50% presence** (across aircraft) + **30% frequency** (NRC rate) "
            "+ **20% manhour cost**. Higher = more fleet-wide concern."
        )

        tier_filter = st.radio(
            "Filter by tier",
            ["All", "Fleet-wide", "Common", "Isolated"],
            horizontal=True,
        )
        filtered = scores if tier_filter == "All" else scores[scores["tier"] == tier_filter]

        st.plotly_chart(
            ranked_bar(filtered, top_n=top_n_score),
            use_container_width=True,
        )

        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(damage_distribution(df), use_container_width=True)
        with col_r:
            st.plotly_chart(tier_donut(scores), use_container_width=True)

    # ─────────────────────────────────────────────────────
    # TAB 2 — Fleet-wide
    # ─────────────────────────────────────────────────────
    with tab2:
        fleet = scores[scores["tier"] == "Fleet-wide"]
        if fleet.empty:
            st.info("No defects found in all aircraft. Try uploading more events.")
        else:
            st.markdown(
                f"#### {len(fleet)} defects found in **all {n_projects} aircraft** — "
                "these are your strongest workscope candidates."
            )

            # Summary cards for fleet-wide
            count_cols = [c for c in fleet.columns if c.startswith("count_")]
            for _, row in fleet.iterrows():
                with st.expander(
                    f"**{row['location']} — {row['damage_type']}**  "
                    f"| Score: {row['score']:.3f}  "
                    f"| {int(row['total_count'])} NRCs  "
                    f"| {row['avg_mhrs']:.1f}h avg"
                ):
                    ac_cols = st.columns(len(count_cols))
                    for i, col in enumerate(count_cols):
                        ac_name = col.replace("count_", "")
                        ac_cols[i].metric(ac_name, int(row[col]))

                    # NRCs for this defect
                    sub = df[
                        (df["location"] == row["location"]) &
                        (df["damage_type"] == row["damage_type"])
                    ][["project", "Description", "Skill Active", "Act Mhrs"]]
                    if not sub.empty:
                        st.dataframe(sub.reset_index(drop=True), use_container_width=True)

            st.markdown("---")
            st.plotly_chart(fleet_grouped_bar(scores, projects), use_container_width=True)
            st.plotly_chart(frequency_heatmap(scores, top_n=top_n_score), use_container_width=True)

    # ─────────────────────────────────────────────────────
    # TAB 3 — Similarity map
    # ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### NRC similarity map")
        st.markdown(
            "Each dot = one NRC. NRCs close together have similar title language. "
            "Colour = cluster, shape = aircraft."
        )
        st.plotly_chart(scatter_map(df, X_2d), use_container_width=True)

        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(cluster_size_dist(df), use_container_width=True)
        with col_r:
            # Cluster list
            st.markdown("**Cluster summary**")
            cluster_summary = (
                df[df["cluster_id"] != -1]
                .groupby("cluster_label")
                .agg(count=("Description","count"), projects=("project","nunique"))
                .sort_values("count", ascending=False)
                .reset_index()
            )
            st.dataframe(cluster_summary, use_container_width=True, height=280)

    # ─────────────────────────────────────────────────────
    # TAB 4 — Manhour impact
    # ─────────────────────────────────────────────────────
    with tab4:
        st.markdown("#### Defects by manhour burden")
        st.markdown(
            "Focuses on the costliest defects. "
            "Red = >30h avg, amber = 15–30h, blue = <15h."
        )
        st.plotly_chart(manhour_bar(scores, top_n=top_n_score), use_container_width=True)

        st.markdown("---")
        st.markdown("#### Combined view — score vs manhour cost (fleet-wide only)")
        fleet = scores[scores["tier"] == "Fleet-wide"].copy()
        if not fleet.empty:
            fleet["label"] = fleet["location"] + " — " + fleet["damage_type"]
            import plotly.express as px
            bubble = px.scatter(
                fleet,
                x="total_count",
                y="avg_mhrs",
                size="score",
                color="score",
                text="label",
                color_continuous_scale="Teal",
                title="Fleet-wide defects — NRC count vs manhours (bubble = score)",
                labels={"total_count":"NRC count","avg_mhrs":"Avg manhours","score":"Score"},
                height=420,
            )
            bubble.update_traces(textposition="top center", textfont_size=10)
            bubble.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(bubble, use_container_width=True)

    # ─────────────────────────────────────────────────────
    # TAB 5 — Score breakdown
    # ─────────────────────────────────────────────────────
    with tab5:
        st.markdown("#### How each score is composed")
        st.plotly_chart(score_components_bar(scores, top_n=15), use_container_width=True)

        st.markdown("---")
        st.markdown("**Score formula reference**")
        c1, c2, c3 = st.columns(3)
        c1.info("**Presence · 50%**\nFraction of aircraft that have this defect.\n"
                "3/3 = 1.0 pt, 2/3 = 0.67 pt, 1/3 = 0.33 pt")
        c2.info("**Frequency · 30%**\nAvg NRC rate per aircraft (count ÷ project size), "
                "normalised 0–1.")
        c3.info("**Manhour cost · 20%**\nAvg actual manhours for this defect combo, "
                "normalised 0–1.")

    # ─────────────────────────────────────────────────────
    # TAB 6 — Data tables
    # ─────────────────────────────────────────────────────
    with tab6:
        st.markdown("#### All NRCs with cluster assignments")
        show_cols = [c for c in
                     ["project","Description","Skill Active","Act Mhrs",
                      "cluster_label","location","damage_type","cluster_id"]
                     if c in df.columns]

        # Filters
        f1, f2, f3 = st.columns(3)
        sel_ac  = f1.multiselect("Filter by aircraft", options=projects, default=projects)
        dmg_opts = ["All"] + sorted(df["damage_type"].dropna().unique().tolist())
        sel_dmg = f2.selectbox("Filter by damage type", dmg_opts)
        loc_opts = ["All"] + sorted(df["location"].dropna().unique().tolist())
        sel_loc = f3.selectbox("Filter by location", loc_opts)

        mask = df["project"].isin(sel_ac)
        if sel_dmg != "All":
            mask &= df["damage_type"] == sel_dmg
        if sel_loc != "All":
            mask &= df["location"] == sel_loc

        filtered_df = df[mask][show_cols].reset_index(drop=True)
        st.markdown(f"Showing **{len(filtered_df)}** NRCs")
        st.dataframe(filtered_df, use_container_width=True, height=400)

        st.markdown("---")
        st.markdown("#### EDA score table")
        score_cols = ["defect_key","tier","total_count","projects_count",
                      "avg_mhrs","score"] + \
                     [c for c in scores.columns if c.startswith("count_")]
        avail_score_cols = [c for c in score_cols if c in scores.columns]
        st.dataframe(scores[avail_score_cols], use_container_width=True, height=360)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Download results")
    excel_bytes = build_excel(df, scores)
    st.download_button(
        label="⬇️ Download full Excel report",
        data=excel_bytes,
        file_name="nrc_clustering_results.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )

else:
    # ── Landing state ──────────────────────────────────────────────────────────
    st.info(
        "👈 Upload your NRC Excel files in the sidebar, "
        "set the AC registrations, then click **Run pipeline**."
    )
    st.markdown("---")
    st.markdown("### How it works")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown("**1 · Upload**\nOne Excel file per aircraft registration. "
                "Needs a `Description` column.")
    c2.markdown("**2 · Cluster**\nTF-IDF vectorisation → UMAP reduction → "
                "HDBSCAN clustering groups similar NRC titles.")
    c3.markdown("**3 · Score**\nWeighted EDA scores each defect combo by "
                "presence across AC, frequency, and manhour cost.")
    c4.markdown("**4 · Export**\nDownload an Excel report with all NRCs, "
                "clusters, frequency matrix, and EDA scores.")
