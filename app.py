"""
app.py
------
NRC Findings Clustering + Material Request Analysis
Streamlit app for GMF AeroAsia MRO.

Run with:
    streamlit run app.py
"""

import io
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd

from core.pipeline  import run_pipeline, build_excel
from core.materials import load_mrm, build_material_summary, summarise_by_defect, top_parts_across_fleet
from core.charts    import (
    scatter_map, ranked_bar, fleet_grouped_bar,
    frequency_heatmap, manhour_bar, score_components_bar,
    tier_donut, damage_distribution, cluster_size_dist,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NRC Clustering & Material Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 500; }
  div[data-testid="stExpander"] { border: 0.5px solid #e0e0e0; border-radius:8px; }
  .pill { display:inline-block; font-size:11px; font-weight:500; padding:2px 9px;
          border-radius:10px; margin:1px; }
  .pill-fleet  { background:#E1F5EE; color:#0F6E56; }
  .pill-common { background:#E6F1FB; color:#185FA5; }
  .pill-iso    { background:#F1EFE8; color:#5F5E5A; }
  .pill-part   { background:#FFF3E0; color:#7B4400; font-family:monospace; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ NRC Clustering")
    st.markdown("---")

    st.markdown("### 1 · Aircraft & files")
    st.markdown(
        "For each aircraft upload:\n"
        "- **Findings file** `_MDR_TRACKING_*.xlsx`\n"
        "- **Material request** `_MRM_TRACKING_*.xlsx` *(optional)*"
    )

    n_files = st.number_input("Number of aircraft", min_value=1, max_value=10, value=3)

    ac_entries = []   # list of {ac_reg, nrc_file, mrm_file}
    for i in range(n_files):
        st.markdown(f"**Aircraft #{i+1}**")
        col_a, col_b = st.columns([1, 2])
        with col_a:
            ac_reg = st.text_input(
                "AC Reg", value=f"AC-{i+1:03d}",
                key=f"reg_{i}", placeholder="e.g. PK-GLZ",
                label_visibility="collapsed",
            )
        with col_b:
            rev_no = st.text_input(
                "Rev No", value="",
                key=f"rev_{i}", placeholder="e.g. 00221737",
                label_visibility="collapsed",
            )

        nrc_file = st.file_uploader(
            f"Findings (MDR)", type=["xlsx","xls"],
            key=f"nrc_{i}", label_visibility="visible",
        )
        mrm_file = st.file_uploader(
            f"Materials (MRM)", type=["xlsx","xls"],
            key=f"mrm_{i}", label_visibility="visible",
        )
        label = f"{ac_reg} ({rev_no})" if rev_no else ac_reg
        ac_entries.append({"label": label, "ac_reg": ac_reg,
                           "rev_no": rev_no, "nrc_file": nrc_file, "mrm_file": mrm_file})
        st.markdown("---")

    st.markdown("### 2 · Settings")
    min_cluster = st.slider("Min cluster size", 3, 20, 5,
        help="Lower = more clusters. Higher = fewer, tighter clusters.")
    top_n_score = st.slider("Top N defects to show", 10, 50, 25)
    min_ac_parts = st.slider(
        "Min AC for pre-provision recommendation", 1, n_files, min(2, n_files),
        help="Parts appearing in this many aircraft are flagged as fleet-level candidates."
    )

    st.markdown("---")
    run_btn = st.button("▶ Run analysis", type="primary", use_container_width=True)
    st.markdown("---")
    st.markdown(
        "<small style='color:gray'>GMF AeroAsia · NRC Clustering v2<br>"
        "TF-IDF · UMAP · HDBSCAN · Weighted EDA</small>",
        unsafe_allow_html=True,
    )


# ── Main ─────────────────────────────────────────────────────────────────────
st.title("NRC Findings Clustering & Material Analysis")
st.markdown(
    "Cluster NRC findings across maintenance events, score recurring defects "
    "by fleet-wide commonality, and surface the materials requested for each defect."
)

if "results" not in st.session_state:
    st.session_state.results = None

# ── Run ───────────────────────────────────────────────────────────────────────
if run_btn:
    nrc_uploads = {
        e["label"]: e["nrc_file"]
        for e in ac_entries if e["nrc_file"] is not None
    }
    if not nrc_uploads:
        st.error("Upload at least one Findings (MDR) file.")
    else:
        with st.spinner("Running pipeline…"):
            try:
                # NRC clustering pipeline
                results = run_pipeline(nrc_uploads, min_cluster_size=min_cluster)

                # Load MRM files
                mrm_dict = {}
                for e in ac_entries:
                    if e["mrm_file"] is not None and e["label"] in nrc_uploads:
                        try:
                            mrm_dict[e["label"]] = load_mrm(e["mrm_file"])
                        except Exception as ex:
                            st.warning(f"Could not load MRM for {e['label']}: {ex}")

                # Build material tables
                mat_detail  = pd.DataFrame()
                mat_summary = pd.DataFrame()
                top_parts   = pd.DataFrame()

                if mrm_dict:
                    mat_detail  = build_material_summary(
                        results["df"], mrm_dict, results["scores"]
                    )
                    mat_summary = summarise_by_defect(mat_detail)
                    top_parts   = top_parts_across_fleet(mat_detail, min_ac=min_ac_parts)

                results["mrm_dict"]    = mrm_dict
                results["mat_detail"]  = mat_detail
                results["mat_summary"] = mat_summary
                results["top_parts"]   = top_parts

                st.session_state.results = results
                st.success(
                    f"Done! {len(results['df'])} NRCs · "
                    f"{len(results['projects'])} aircraft · "
                    f"{len(mrm_dict)} MRM files loaded."
                )
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)


# ── Display ───────────────────────────────────────────────────────────────────
if st.session_state.results:
    R           = st.session_state.results
    df          = R["df"]
    scores      = R["scores"]
    X_2d        = R["X_2d"]
    projects    = R["projects"]
    mat_detail  = R["mat_detail"]
    mat_summary = R["mat_summary"]
    top_parts   = R["top_parts"]
    has_mrm     = not mat_detail.empty

    n_clustered = (df["cluster_id"] != -1).sum()
    n_clusters  = df[df["cluster_id"] != -1]["cluster_id"].nunique()
    n_noise     = (df["cluster_id"] == -1).sum()
    n_fleet     = int((scores["tier"] == "Fleet-wide").sum()) if not scores.empty else 0
    n_common    = int((scores["tier"] == "Common").sum())     if not scores.empty else 0

    # KPIs
    st.markdown("---")
    cols = st.columns(7 if has_mrm else 6)
    cols[0].metric("Total NRCs",       f"{len(df):,}")
    cols[1].metric("Aircraft",          len(projects))
    cols[2].metric("Clusters",          n_clusters)
    cols[3].metric("NRCs clustered",   f"{n_clustered:,}")
    cols[4].metric("Fleet-wide defects", n_fleet)
    cols[5].metric("One-off NRCs",       n_noise)
    if has_mrm:
        cols[6].metric("Unique parts (fleet)", int(top_parts["Part Number"].nunique()) if not top_parts.empty else 0)

    # AC legend
    ac_colors = ["#5DCAA5","#85B7EB","#FAC775","#D85A30","#534AB7","#D4537E"]
    pills = " &nbsp; ".join(
        f"<span style='background:{ac_colors[i%len(ac_colors)]};color:#333;"
        f"padding:3px 10px;border-radius:8px;font-size:12px;font-weight:500'>"
        f"{p}{'&nbsp;📦' if p in R.get('mrm_dict',{}) else ''}</span>"
        for i, p in enumerate(projects)
    )
    st.markdown(f"**Aircraft:** {pills} &nbsp; *(📦 = MRM file loaded)*", unsafe_allow_html=True)
    st.markdown("---")

    # Tabs — add Materials tabs only if MRM loaded
    base_tabs  = ["📊 Ranked defects", "🌍 Fleet-wide", "🗺️ Similarity map",
                  "🔥 Manhour impact", "📈 Score breakdown", "📋 Data tables"]
    mrm_tabs   = ["🔩 Materials by defect", "📦 Pre-provision list"] if has_mrm else []
    all_tabs   = base_tabs + mrm_tabs
    tab_objs   = st.tabs(all_tabs)

    tidx = 0  # tab index counter

    # ── TAB 0: Ranked defects ─────────────────────────────────────────────────
    with tab_objs[tidx]; tidx += 1:
        st.markdown("#### Defects ranked by weighted commonality score")
        st.markdown(
            "Score = **50% presence** (across aircraft) + "
            "**30% frequency** (NRC rate) + **20% manhour cost**"
        )
        tier_filter = st.radio("Filter", ["All","Fleet-wide","Common","Isolated"], horizontal=True)
        filt = scores if tier_filter == "All" else scores[scores["tier"] == tier_filter]
        st.plotly_chart(ranked_bar(filt, top_n=top_n_score), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1: st.plotly_chart(damage_distribution(df),  use_container_width=True)
        with c2: st.plotly_chart(tier_donut(scores),       use_container_width=True)

    # ── TAB 1: Fleet-wide ─────────────────────────────────────────────────────
    with tab_objs[tidx]; tidx += 1:
        fleet = scores[scores["tier"] == "Fleet-wide"]
        if fleet.empty:
            st.info("No fleet-wide defects found.")
        else:
            st.markdown(f"#### {len(fleet)} defects in **all {len(projects)} aircraft**")
            count_cols = [c for c in fleet.columns if c.startswith("count_")]

            for _, row in fleet.iterrows():
                # Material badge count
                if has_mrm:
                    parts_for = mat_detail[mat_detail["defect_key"] == row["defect_key"]]
                    n_parts   = parts_for["Part Number"].nunique()
                    mat_badge = f"&nbsp; 🔩 {n_parts} parts" if n_parts else ""
                else:
                    mat_badge = ""

                with st.expander(
                    f"**{row['location']} — {row['damage_type']}**"
                    f"  |  Score: {row['score']:.3f}"
                    f"  |  {int(row['total_count'])} NRCs"
                    f"  |  {row['avg_mhrs']:.1f}h avg"
                ):
                    ac_c = st.columns(len(count_cols))
                    for i, col in enumerate(count_cols):
                        ac_c[i].metric(col.replace("count_",""), int(row[col]))

                    sub = df[
                        (df["location"] == row["location"]) &
                        (df["damage_type"] == row["damage_type"])
                    ][["project","Description","Skill Active","Act Mhrs"]]
                    st.dataframe(sub.reset_index(drop=True), use_container_width=True)

                    # Inline materials for this defect
                    if has_mrm and n_parts > 0:
                        st.markdown("**Materials requested for this defect:**")
                        mat_show = parts_for[[
                            "ac_reg","Order No","Part Number",
                            "Material Description","Qty Req","UOM","Fulfillment Status"
                        ]].reset_index(drop=True)
                        st.dataframe(mat_show, use_container_width=True)

            st.markdown("---")
            st.plotly_chart(fleet_grouped_bar(scores, projects), use_container_width=True)
            st.plotly_chart(frequency_heatmap(scores, top_n=top_n_score), use_container_width=True)

    # ── TAB 2: Similarity map ─────────────────────────────────────────────────
    with tab_objs[tidx]; tidx += 1:
        st.markdown("#### NRC similarity map")
        st.markdown("Each dot = one NRC. Proximity = similar title language.")
        st.plotly_chart(scatter_map(df, X_2d), use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(cluster_size_dist(df), use_container_width=True)
        with c2:
            cs = (
                df[df["cluster_id"] != -1]
                .groupby("cluster_label")
                .agg(count=("Description","count"), ac=("project","nunique"))
                .sort_values("count", ascending=False)
                .reset_index()
            )
            st.markdown("**Cluster summary**")
            st.dataframe(cs, use_container_width=True, height=280)

    # ── TAB 3: Manhour impact ─────────────────────────────────────────────────
    with tab_objs[tidx]; tidx += 1:
        st.markdown("#### Defects by manhour burden")
        st.plotly_chart(manhour_bar(scores, top_n=top_n_score), use_container_width=True)

        fleet = scores[scores["tier"] == "Fleet-wide"].copy()
        if not fleet.empty:
            fleet["label"] = fleet["location"] + " — " + fleet["damage_type"]
            import plotly.express as px
            bubble = px.scatter(
                fleet, x="total_count", y="avg_mhrs",
                size="score", color="score", text="label",
                color_continuous_scale="Teal",
                title="Fleet-wide — NRC count vs manhours (bubble = score)",
                labels={"total_count":"NRC count","avg_mhrs":"Avg manhours"},
                height=420,
            )
            bubble.update_traces(textposition="top center", textfont_size=10)
            bubble.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(bubble, use_container_width=True)

    # ── TAB 4: Score breakdown ────────────────────────────────────────────────
    with tab_objs[tidx]; tidx += 1:
        st.markdown("#### Score component breakdown")
        st.plotly_chart(score_components_bar(scores, top_n=15), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.info("**Presence · 50%**\nFraction of aircraft with this defect.\n3/3=1.0 · 2/3=0.67 · 1/3=0.33")
        c2.info("**Frequency · 30%**\nAvg NRC rate per aircraft, normalised 0–1.")
        c3.info("**Manhour cost · 20%**\nAvg actual manhours, normalised 0–1.")

    # ── TAB 5: Data tables ────────────────────────────────────────────────────
    with tab_objs[tidx]; tidx += 1:
        st.markdown("#### All NRCs with cluster assignments")
        show_cols = [c for c in ["project","Description","Skill Active","Act Mhrs",
                                  "cluster_label","location","damage_type"] if c in df.columns]
        f1, f2, f3 = st.columns(3)
        sel_ac  = f1.multiselect("Aircraft", options=projects, default=projects)
        sel_dmg = f2.selectbox("Damage type", ["All"]+sorted(df["damage_type"].dropna().unique().tolist()))
        sel_loc = f3.selectbox("Location",    ["All"]+sorted(df["location"].dropna().unique().tolist()))
        mask = df["project"].isin(sel_ac)
        if sel_dmg != "All": mask &= df["damage_type"] == sel_dmg
        if sel_loc != "All": mask &= df["location"]    == sel_loc
        st.markdown(f"Showing **{mask.sum()}** NRCs")
        st.dataframe(df[mask][show_cols].reset_index(drop=True), use_container_width=True, height=380)

        st.markdown("---")
        st.markdown("#### EDA score table")
        sc_cols = ["defect_key","tier","total_count","projects_count","avg_mhrs","score"] + \
                  [c for c in scores.columns if c.startswith("count_")]
        st.dataframe(scores[[c for c in sc_cols if c in scores.columns]],
                     use_container_width=True, height=340)

    # ── TAB 6: Materials by defect (MRM only) ─────────────────────────────────
    if has_mrm:
        with tab_objs[tidx]; tidx += 1:
            st.markdown("#### Materials linked to each defect cluster")
            st.markdown(
                "For every scored defect, these are the materials (toggle=**Y**) "
                "requested on the matched orders across all aircraft."
            )

            tier_f = st.radio("Show tier", ["All","Fleet-wide","Common","Isolated"],
                              horizontal=True, key="mat_tier")
            score_f = scores if tier_f == "All" else scores[scores["tier"] == tier_f]

            for _, srow in score_f.head(top_n_score).iterrows():
                key   = srow["defect_key"]
                parts = mat_detail[mat_detail["defect_key"] == key]
                if parts.empty:
                    continue

                n_unique = parts["Part Number"].nunique()
                n_ac     = parts["ac_reg"].nunique()
                tier_cls = ("pill-fleet" if srow["tier"]=="Fleet-wide"
                            else "pill-common" if srow["tier"]=="Common"
                            else "pill-iso")

                with st.expander(
                    f"**{srow['location']} — {srow['damage_type']}**"
                    f"  |  Score: {srow['score']:.3f}"
                    f"  |  {n_unique} unique parts"
                    f"  |  {n_ac} aircraft"
                ):
                    # Pivot: one row per part, columns per AC
                    piv = (
                        parts
                        .groupby(["Part Number","Material Description","UOM","Type","ac_reg"])["Qty Req"]
                        .sum()
                        .unstack(fill_value=0)
                        .reset_index()
                    )
                    piv["Total Qty"] = piv.select_dtypes("number").sum(axis=1)
                    piv["AC count"]  = (piv.select_dtypes("number").drop(columns=["Total Qty"]) > 0).sum(axis=1)
                    piv = piv.sort_values(["AC count","Total Qty"], ascending=[False,False])
                    st.dataframe(piv, use_container_width=True)

        # ── TAB 7: Pre-provision list ─────────────────────────────────────────
        with tab_objs[tidx]; tidx += 1:
            st.markdown("#### Pre-provision recommendations")
            st.markdown(
                f"Parts appearing in **≥ {min_ac_parts} aircraft** for the same defect — "
                "strongest candidates to include in future workscope material kits."
            )

            if top_parts.empty:
                st.info(f"No parts found in ≥{min_ac_parts} aircraft. Try lowering the threshold.")
            else:
                # Summary KPIs
                k1, k2, k3 = st.columns(3)
                k1.metric("Recommended parts",       len(top_parts))
                k2.metric("Distinct defects covered", int(top_parts["defect_count"].sum()))
                k3.metric("Total qty to pre-provision", f"{top_parts['total_qty'].sum():,.0f}")

                st.markdown("---")

                # Filter by type
                types = ["All"] + sorted(top_parts["Type"].dropna().unique().tolist())
                sel_type = st.selectbox("Filter by material type", types)
                show_parts = top_parts if sel_type == "All" else top_parts[top_parts["Type"] == sel_type]

                display_cols = [
                    "Part Number","Material Description","UOM","Type",
                    "ac_count","defect_count","total_qty","defects","ac_list","avg_score"
                ]
                display_cols = [c for c in display_cols if c in show_parts.columns]

                st.dataframe(
                    show_parts[display_cols].rename(columns={
                        "ac_count":     "# Aircraft",
                        "defect_count": "# Defects",
                        "total_qty":    "Total Qty",
                        "defects":      "Defect types",
                        "ac_list":      "Found in AC",
                        "avg_score":    "Avg score",
                    }),
                    use_container_width=True,
                    height=460,
                )

                # Breakdown chart
                import plotly.express as px
                fig = px.bar(
                    show_parts.head(30),
                    x="total_qty",
                    y="Material Description",
                    color="ac_count",
                    orientation="h",
                    color_continuous_scale="Blues",
                    title="Top 30 recommended parts — total qty · shaded by # aircraft",
                    labels={"total_qty":"Total qty","ac_count":"# Aircraft",
                            "Material Description":"Part"},
                    height=max(400, len(show_parts.head(30)) * 26),
                )
                fig.update_layout(
                    yaxis=dict(autorange="reversed", tickfont_size=10),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=280, r=20, t=50, b=30),
                )
                st.plotly_chart(fig, use_container_width=True)

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Download")
    dl1, dl2 = st.columns(2)

    with dl1:
        excel_bytes = build_excel(df, scores)
        st.download_button(
            "⬇️ Download NRC clusters + EDA scores (Excel)",
            data=excel_bytes,
            file_name="nrc_clustering_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            type="primary",
        )

    if has_mrm:
        with dl2:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                if not mat_detail.empty:
                    mat_detail.to_excel(writer, sheet_name="Material_Detail", index=False)
                if not mat_summary.empty:
                    mat_summary.to_excel(writer, sheet_name="Material_by_Defect", index=False)
                if not top_parts.empty:
                    top_parts.to_excel(writer, sheet_name="PreProvision_List", index=False)
            buf.seek(0)
            st.download_button(
                "⬇️ Download material analysis (Excel)",
                data=buf.read(),
                file_name="nrc_material_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

else:
    st.info("👈 Upload NRC files in the sidebar and click **Run analysis**.")
    st.markdown("---")
    st.markdown("### How it works")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.markdown("**1 · Upload**\nOne MDR + MRM file per aircraft. MRM is optional.")
    c2.markdown("**2 · Cluster**\nTF-IDF → UMAP → HDBSCAN groups similar NRC titles.")
    c3.markdown("**3 · Score**\nWeighted EDA scores by presence, frequency & manhours.")
    c4.markdown("**4 · Materials**\nJoins MRM (toggle=Y) to each defect via Order No.")
    c5.markdown("**5 · Pre-provision**\nParts in multiple aircraft flagged as fleet-level candidates.")
