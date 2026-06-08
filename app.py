"""
app.py
------
NRC Findings Clustering + Material Analysis + Run History + PDF Export
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
import plotly.express as px

from core.pipeline   import run_pipeline, build_excel
from core.materials  import load_mrm, build_material_summary, summarise_by_defect, top_parts_across_fleet
from core.pdf_export import generate_pdf
from core.storage    import (
    save_run, load_run_history, load_run_scores,
    load_run_materials, delete_run, compare_runs,
)
from core.charts import (
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
  .pill { display:inline-block; font-size:11px; font-weight:500;
          padding:2px 9px; border-radius:10px; margin:1px; }
</style>
""", unsafe_allow_html=True)

# Check if Supabase is configured
HAS_SUPABASE = "supabase" in st.secrets if hasattr(st, "secrets") else False

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ✈️ NRC Clustering")
    st.markdown("---")

    # ── Page navigation ────────────────────────────────────────────────────
    page = st.radio(
        "Navigate",
        ["🔬 New analysis", "📂 Run history", "🔁 Compare runs"],
        label_visibility="collapsed",
    )
    st.markdown("---")

    if page == "🔬 New analysis":
        st.markdown("### 1 · Aircraft & files")
        st.markdown(
            "For each aircraft upload:\n"
            "- **Findings** `_MDR_TRACKING_*.xlsx`\n"
            "- **Materials** `_MRM_TRACKING_*.xlsx` *(optional)*"
        )

        n_files = st.number_input("Number of aircraft", min_value=1, max_value=10, value=3)

        ac_entries = []
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
            nrc_file = st.file_uploader("Findings (MDR)", type=["xlsx","xls"], key=f"nrc_{i}")
            mrm_file = st.file_uploader("Materials (MRM)", type=["xlsx","xls"], key=f"mrm_{i}")
            label = f"{ac_reg} ({rev_no})" if rev_no else ac_reg
            ac_entries.append({
                "label": label, "ac_reg": ac_reg,
                "rev_no": rev_no, "nrc_file": nrc_file, "mrm_file": mrm_file,
            })
            st.markdown("---")

        st.markdown("### 2 · Run settings")
        min_cluster  = st.slider("Min cluster size", 3, 20, 5)
        top_n_score  = st.slider("Top N defects",   10, 50, 25)
        min_ac_parts = st.slider("Min AC for pre-provision", 1, max(1,n_files), min(2,n_files))

        st.markdown("### 3 · Run metadata")
        workscope = st.text_input("Workscope", placeholder="e.g. 6Y+12Y C-Check")
        ac_type   = st.text_input("Aircraft type", placeholder="e.g. A320-200")
        notes     = st.text_area("Notes (optional)", height=80)

        st.markdown("---")
        run_btn = st.button("▶ Run analysis", type="primary", use_container_width=True)

    st.markdown(
        "<small style='color:gray'>GMF AeroAsia · NRC Clustering v3<br>"
        "TF-IDF · UMAP · HDBSCAN · Weighted EDA · Supabase</small>",
        unsafe_allow_html=True,
    )


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: NEW ANALYSIS
# ════════════════════════════════════════════════════════════════════════════
if page == "🔬 New analysis":

    st.title("NRC Findings Clustering & Material Analysis")
    st.markdown(
        "Cluster NRC findings, score defects by fleet-wide commonality, "
        "surface materials, export PDF, and save to history."
    )

    if "results" not in st.session_state:
        st.session_state.results = None

    # ── Run pipeline ──────────────────────────────────────────────────────
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
                    results = run_pipeline(nrc_uploads, min_cluster_size=min_cluster)

                    mrm_dict = {}
                    for e in ac_entries:
                        if e["mrm_file"] is not None and e["label"] in nrc_uploads:
                            try:
                                mrm_dict[e["label"]] = load_mrm(e["mrm_file"])
                            except Exception as ex:
                                st.warning(f"MRM load error ({e['label']}): {ex}")

                    mat_detail = mat_summary = top_parts = pd.DataFrame()
                    if mrm_dict:
                        mat_detail  = build_material_summary(results["df"], mrm_dict, results["scores"])
                        mat_summary = summarise_by_defect(mat_detail)
                        top_parts   = top_parts_across_fleet(mat_detail, min_ac=min_ac_parts)

                    results.update({
                        "mrm_dict": mrm_dict, "mat_detail": mat_detail,
                        "mat_summary": mat_summary, "top_parts": top_parts,
                        "workscope": workscope, "ac_type": ac_type, "notes": notes,
                    })
                    st.session_state.results = results
                    st.success(
                        f"Done! {len(results['df'])} NRCs · "
                        f"{len(results['projects'])} aircraft · "
                        f"{len(mrm_dict)} MRM file(s) loaded."
                    )
                except Exception as e:
                    st.error(f"Pipeline error: {e}")
                    st.exception(e)

    # ── Display results ───────────────────────────────────────────────────
    if st.session_state.results:
        R          = st.session_state.results
        df         = R["df"]
        scores     = R["scores"]
        X_2d       = R["X_2d"]
        projects   = R["projects"]
        mat_detail = R["mat_detail"]
        top_parts  = R["top_parts"]
        has_mrm    = not mat_detail.empty

        n_clustered = (df["cluster_id"] != -1).sum()
        n_clusters  = df[df["cluster_id"] != -1]["cluster_id"].nunique()
        n_noise     = (df["cluster_id"] == -1).sum()
        n_fleet     = int((scores["tier"] == "Fleet-wide").sum()) if not scores.empty else 0

        # KPIs
        st.markdown("---")
        kcols = st.columns(7 if has_mrm else 6)
        kcols[0].metric("Total NRCs",         f"{len(df):,}")
        kcols[1].metric("Aircraft",             len(projects))
        kcols[2].metric("Clusters",             n_clusters)
        kcols[3].metric("NRCs clustered",      f"{n_clustered:,}")
        kcols[4].metric("Fleet-wide defects",   n_fleet)
        kcols[5].metric("One-off NRCs",         n_noise)
        if has_mrm:
            kcols[6].metric(
                "Unique parts (fleet)",
                int(top_parts["Part Number"].nunique()) if not top_parts.empty else 0,
            )

        ac_colors = ["#5DCAA5","#85B7EB","#FAC775","#D85A30","#534AB7","#D4537E"]
        pills = " &nbsp; ".join(
            f"<span style='background:{ac_colors[i%len(ac_colors)]};color:#333;"
            f"padding:3px 10px;border-radius:8px;font-size:12px;font-weight:500'>"
            f"{p}{'&nbsp;📦' if p in R.get('mrm_dict',{}) else ''}</span>"
            for i, p in enumerate(projects)
        )
        st.markdown(f"**Aircraft:** {pills} &nbsp;*(📦 = MRM loaded)*", unsafe_allow_html=True)
        st.markdown("---")

        # ── Tabs ──────────────────────────────────────────────────────────
        base_names = ["📊 Ranked","🌍 Fleet-wide","🗺️ Map","🔥 Manhours",
                      "📈 Score","📋 Data"]
        mrm_names  = ["🔩 Materials","📦 Pre-provision"] if has_mrm else []
        tabs = st.tabs(base_names + mrm_names)

        tab_ranked, tab_fleet, tab_map, tab_mhrs, tab_score, tab_data = tabs[:6]
        tab_matdef  = tabs[6] if has_mrm else None
        tab_preprov = tabs[7] if has_mrm else None

        # Ranked
        with tab_ranked:
            st.markdown("#### Defects by weighted commonality score")
            st.markdown("Score = **50%** presence + **30%** frequency + **20%** manhour cost")
            tf = st.radio("Tier", ["All","Fleet-wide","Common","Isolated"], horizontal=True)
            filt = scores if tf == "All" else scores[scores["tier"] == tf]
            st.plotly_chart(ranked_bar(filt, top_n=top_n_score), use_container_width=True)
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(damage_distribution(df),  use_container_width=True)
            with c2: st.plotly_chart(tier_donut(scores),       use_container_width=True)

        # Fleet-wide
        with tab_fleet:
            fleet = scores[scores["tier"] == "Fleet-wide"]
            if fleet.empty:
                st.info("No fleet-wide defects found.")
            else:
                st.markdown(f"#### {len(fleet)} defects in all {len(projects)} aircraft")
                count_cols = [c for c in fleet.columns if c.startswith("count_")]
                for _, row in fleet.iterrows():
                    n_parts = 0
                    if has_mrm:
                        pf = mat_detail[mat_detail["defect_key"] == row["defect_key"]]
                        n_parts = pf["Part Number"].nunique()
                    with st.expander(
                        f"**{row['location']} — {row['damage_type']}**"
                        f"  |  Score: {row['score']:.3f}"
                        f"  |  {int(row['total_count'])} NRCs"
                        f"  |  {row['avg_mhrs']:.1f}h avg"
                        + (f"  |  🔩 {n_parts} parts" if n_parts else "")
                    ):
                        ac_c = st.columns(len(count_cols))
                        for i, col in enumerate(count_cols):
                            ac_c[i].metric(col.replace("count_",""), int(row[col]))
                        sub = df[
                            (df["location"] == row["location"]) &
                            (df["damage_type"] == row["damage_type"])
                        ][["project","Description","Skill Active","Act Mhrs"]]
                        st.dataframe(sub.reset_index(drop=True), use_container_width=True)
                        if has_mrm and n_parts > 0:
                            st.markdown("**Materials:**")
                            st.dataframe(
                                pf[["ac_reg","Order No","Part Number",
                                    "Material Description","Qty Req","UOM","Fulfillment Status"]
                                ].reset_index(drop=True),
                                use_container_width=True,
                            )
                st.markdown("---")
                st.plotly_chart(fleet_grouped_bar(scores, projects), use_container_width=True)
                st.plotly_chart(frequency_heatmap(scores, top_n=top_n_score), use_container_width=True)

        # Similarity map
        with tab_map:
            st.markdown("#### NRC similarity map")
            st.plotly_chart(scatter_map(df, X_2d), use_container_width=True)
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(cluster_size_dist(df), use_container_width=True)
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

        # Manhours
        with tab_mhrs:
            st.plotly_chart(manhour_bar(scores, top_n=top_n_score), use_container_width=True)
            fleet_mh = scores[scores["tier"] == "Fleet-wide"].copy()
            if not fleet_mh.empty:
                fleet_mh["label"] = fleet_mh["location"] + " — " + fleet_mh["damage_type"]
                bubble = px.scatter(
                    fleet_mh, x="total_count", y="avg_mhrs",
                    size="score", color="score", text="label",
                    color_continuous_scale="Teal",
                    title="Fleet-wide — NRC count vs manhours",
                    labels={"total_count":"NRC count","avg_mhrs":"Avg hrs"},
                    height=420,
                )
                bubble.update_traces(textposition="top center", textfont_size=10)
                bubble.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(bubble, use_container_width=True)

        # Score breakdown
        with tab_score:
            st.plotly_chart(score_components_bar(scores, top_n=15), use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.info("**Presence · 50%**\n3/3 = 1.0 · 2/3 = 0.67 · 1/3 = 0.33")
            c2.info("**Frequency · 30%**\nAvg NRC rate per aircraft, normalised 0–1.")
            c3.info("**Manhour cost · 20%**\nAvg actual manhours, normalised 0–1.")

        # Data tables
        with tab_data:
            show_cols = [c for c in ["project","Description","Skill Active","Act Mhrs",
                                      "cluster_label","location","damage_type"] if c in df.columns]
            f1, f2, f3 = st.columns(3)
            sel_ac  = f1.multiselect("Aircraft",  options=projects, default=projects)
            sel_dmg = f2.selectbox("Damage type", ["All"]+sorted(df["damage_type"].dropna().unique().tolist()))
            sel_loc = f3.selectbox("Location",    ["All"]+sorted(df["location"].dropna().unique().tolist()))
            mask = df["project"].isin(sel_ac)
            if sel_dmg != "All": mask &= df["damage_type"] == sel_dmg
            if sel_loc != "All": mask &= df["location"]    == sel_loc
            st.markdown(f"Showing **{mask.sum()}** NRCs")
            st.dataframe(df[mask][show_cols].reset_index(drop=True),
                         use_container_width=True, height=380)
            st.markdown("---")
            sc_cols = (["defect_key","tier","total_count","projects_count","avg_mhrs","score"]
                       + [c for c in scores.columns if c.startswith("count_")])
            st.dataframe(scores[[c for c in sc_cols if c in scores.columns]],
                         use_container_width=True, height=340)

        # Materials by defect
        if has_mrm and tab_matdef is not None:
            with tab_matdef:
                st.markdown("#### Materials linked to each defect")
                tf2 = st.radio("Tier", ["All","Fleet-wide","Common","Isolated"],
                               horizontal=True, key="mat_tier")
                sf  = scores if tf2 == "All" else scores[scores["tier"] == tf2]
                for _, srow in sf.head(top_n_score).iterrows():
                    key   = srow["defect_key"]
                    parts = mat_detail[mat_detail["defect_key"] == key]
                    if parts.empty: continue
                    n_unique = parts["Part Number"].nunique()
                    n_ac     = parts["ac_reg"].nunique()
                    with st.expander(
                        f"**{srow['location']} — {srow['damage_type']}**"
                        f"  |  Score: {srow['score']:.3f}"
                        f"  |  {n_unique} unique parts  |  {n_ac} aircraft"
                    ):
                        piv = (
                            parts
                            .groupby(["Part Number","Material Description","UOM","Type","ac_reg"])["Qty Req"]
                            .sum().unstack(fill_value=0).reset_index()
                        )
                        piv["Total Qty"]  = piv.select_dtypes("number").sum(axis=1)
                        piv["# Aircraft"] = (piv.select_dtypes("number")
                                               .drop(columns=["Total Qty"]).gt(0).sum(axis=1))
                        st.dataframe(piv.sort_values("# Aircraft", ascending=False),
                                     use_container_width=True)

        # Pre-provision
        if has_mrm and tab_preprov is not None:
            with tab_preprov:
                st.markdown("#### Pre-provision recommendations")
                if top_parts.empty:
                    st.info(f"No parts found in ≥{min_ac_parts} aircraft.")
                else:
                    k1, k2, k3 = st.columns(3)
                    k1.metric("Recommended parts",          len(top_parts))
                    k2.metric("Distinct defects covered",   int(top_parts["defect_count"].sum()))
                    k3.metric("Total qty",                  f"{top_parts['total_qty'].sum():,.0f}")
                    types     = ["All"] + sorted(top_parts["Type"].dropna().unique().tolist())
                    sel_type  = st.selectbox("Filter type", types)
                    show_pts  = top_parts if sel_type=="All" else top_parts[top_parts["Type"]==sel_type]
                    disp_cols = [c for c in ["Part Number","Material Description","UOM","Type",
                                             "ac_count","defect_count","total_qty","defects",
                                             "ac_list","avg_score"] if c in show_pts.columns]
                    st.dataframe(
                        show_pts[disp_cols].rename(columns={
                            "ac_count":"# AC","defect_count":"# Defects",
                            "total_qty":"Total Qty","defects":"Defect types",
                            "ac_list":"Found in AC","avg_score":"Avg score",
                        }),
                        use_container_width=True, height=460,
                    )
                    fig = px.bar(
                        show_pts.head(30), x="total_qty", y="Material Description",
                        color="ac_count", orientation="h", color_continuous_scale="Blues",
                        title="Top 30 parts — total qty",
                        labels={"total_qty":"Total qty","ac_count":"# AC"},
                        height=max(400, len(show_pts.head(30))*26),
                    )
                    fig.update_layout(
                        yaxis=dict(autorange="reversed", tickfont_size=10),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=280, r=20, t=50, b=30),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # ── Downloads + Save ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Export & Save")

        excel_bytes = build_excel(df, scores)

        # Generate PDF
        with st.spinner("Generating PDF…"):
            pdf_bytes = generate_pdf(
                df, scores,
                top_parts if not top_parts.empty else pd.DataFrame(),
                workscope or "—",
                ac_type   or "—",
                notes     or "",
            )

        col1, col2, col3 = st.columns(3)

        with col1:
            st.download_button(
                "⬇️ Download Excel report",
                data=excel_bytes,
                file_name="nrc_clustering_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary",
            )

        with col2:
            if pdf_bytes:
                st.download_button(
                    "⬇️ Download PDF report",
                    data=pdf_bytes,
                    file_name="nrc_clustering_report.pdf",
                    mime="application/pdf",
                )
            else:
                st.info("PDF unavailable (WeasyPrint not installed on server)")

        if has_mrm:
            mat_buf = io.BytesIO()
            with pd.ExcelWriter(mat_buf, engine="openpyxl") as writer:
                if not mat_detail.empty:
                    mat_detail.to_excel(writer, sheet_name="Material_Detail",     index=False)
                if not top_parts.empty:
                    top_parts.to_excel(writer,  sheet_name="PreProvision_List",   index=False)
            mat_buf.seek(0)
            with col3:
                st.download_button(
                    "⬇️ Download material analysis",
                    data=mat_buf.read(),
                    file_name="nrc_material_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

        # Save to Supabase
        if HAS_SUPABASE:
            st.markdown("---")
            st.markdown("### 💾 Save this run to history")
            save_col1, save_col2 = st.columns([3, 1])
            with save_col1:
                st.markdown(
                    f"Workscope: **{workscope or '—'}** · "
                    f"AC type: **{ac_type or '—'}** · "
                    f"Aircraft: **{', '.join(projects)}**"
                )
            with save_col2:
                if st.button("💾 Save run", type="primary"):
                    with st.spinner("Saving to Supabase…"):
                        run_id = save_run(
                            df=df, scores=scores, top_parts=top_parts,
                            excel_bytes=excel_bytes,
                            pdf_bytes=pdf_bytes,
                            workscope=workscope or "",
                            ac_type=ac_type or "",
                            notes=notes or "",
                        )
                    if run_id:
                        st.success(f"✅ Saved! Run ID: `{run_id[:8]}…`")
                    else:
                        st.error("Save failed — check Supabase credentials.")
        else:
            st.info("💡 Add Supabase credentials to Streamlit secrets to enable run history saving.")

    else:
        st.info("👈 Upload NRC files in the sidebar and click **Run analysis**.")
        st.markdown("---")
        st.markdown("### How it works")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.markdown("**1 · Upload**\nMDR + optional MRM per aircraft.")
        c2.markdown("**2 · Cluster**\nTF-IDF → UMAP → HDBSCAN.")
        c3.markdown("**3 · Score**\nWeighted EDA by presence, frequency & manhours.")
        c4.markdown("**4 · Materials**\nJoins MRM (toggle=Y) via Order No.")
        c5.markdown("**5 · Save**\nStores run to Supabase + export PDF/Excel.")


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: RUN HISTORY
# ════════════════════════════════════════════════════════════════════════════
elif page == "📂 Run history":
    st.title("📂 Run History")
    st.markdown("All past analyses saved to Supabase.")

    if not HAS_SUPABASE:
        st.warning("Supabase not configured. Add credentials to Streamlit secrets.")
        st.stop()

    if st.button("🔄 Refresh", type="secondary"):
        st.cache_data.clear()

    history = load_run_history()

    if history.empty:
        st.info("No runs saved yet. Run an analysis and click **Save run**.")
    else:
        st.markdown(f"**{len(history)} runs saved**")

        for _, run in history.iterrows():
            with st.expander(
                f"**{run.get('workscope','—')}** · "
                f"{run.get('ac_type','—')} · "
                f"{run.get('aircraft','—')} · "
                f"{run.get('created_at','—')}"
            ):
                r1, r2, r3, r4, r5 = st.columns(5)
                r1.metric("Total NRCs",       run.get("total_nrcs",0))
                r2.metric("Clusters",          run.get("n_clusters",0))
                r3.metric("Fleet-wide",        run.get("n_fleet_wide",0))
                r4.metric("Aircraft",          run.get("aircraft","—"))
                r5.metric("Date",              run.get("created_at","—"))

                if run.get("notes"):
                    st.markdown(f"*{run['notes']}*")

                # Load defect scores for this run
                run_scores = load_run_scores(run["id"])
                if not run_scores.empty:
                    st.markdown("**Defect scores:**")
                    display_cols = [c for c in ["location","damage_type","tier","score",
                                                "total_count","projects_count","avg_mhrs"]
                                    if c in run_scores.columns]
                    st.dataframe(
                        run_scores[display_cols].head(20),
                        use_container_width=True, height=280,
                    )

                # Load materials for this run
                run_mats = load_run_materials(run["id"])
                if not run_mats.empty:
                    st.markdown("**Pre-provision materials:**")
                    mat_cols = [c for c in ["part_number","material_description","uom",
                                            "ac_count","total_qty","defects","ac_list"]
                                if c in run_mats.columns]
                    st.dataframe(run_mats[mat_cols].head(20), use_container_width=True, height=240)

                # Download links
                dl1, dl2, del_col = st.columns([2, 2, 1])
                if run.get("excel_url"):
                    dl1.markdown(f"[⬇️ Download Excel]({run['excel_url']})")
                if run.get("pdf_url"):
                    dl2.markdown(f"[⬇️ Download PDF]({run['pdf_url']})")

                with del_col:
                    if st.button("🗑️ Delete", key=f"del_{run['id']}"):
                        if delete_run(run["id"]):
                            st.success("Deleted.")
                            st.rerun()


# ════════════════════════════════════════════════════════════════════════════
#  PAGE: COMPARE RUNS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔁 Compare runs":
    st.title("🔁 Compare Runs")
    st.markdown(
        "Select two past runs to see which defects got better, worse, "
        "or newly appeared between maintenance events."
    )

    if not HAS_SUPABASE:
        st.warning("Supabase not configured.")
        st.stop()

    history = load_run_history()

    if len(history) < 2:
        st.info("Need at least 2 saved runs to compare.")
    else:
        run_labels = (
            history["created_at"].astype(str) + " · " +
            history["workscope"].fillna("—") + " · " +
            history["aircraft"].fillna("—")
        ).tolist()
        run_ids = history["id"].tolist()

        c1, c2 = st.columns(2)
        with c1:
            sel_a = st.selectbox("Run A (baseline)", run_labels, index=0)
        with c2:
            sel_b = st.selectbox("Run B (compare)",  run_labels, index=min(1, len(run_labels)-1))

        id_a = run_ids[run_labels.index(sel_a)]
        id_b = run_ids[run_labels.index(sel_b)]

        if id_a == id_b:
            st.warning("Select two different runs.")
        else:
            with st.spinner("Loading and comparing…"):
                comparison = compare_runs(id_a, id_b)

            if comparison.empty:
                st.error("Could not load scores for one or both runs.")
            else:
                st.markdown("---")
                # Summary stats
                improved  = (comparison["delta"] < -0.02).sum()
                worsened  = (comparison["delta"] >  0.02).sum()
                new_defects = ((comparison["score_a"] == 0) & (comparison["score_b"] > 0)).sum()
                resolved    = ((comparison["score_a"] > 0) & (comparison["score_b"] == 0)).sum()

                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Defects worsened ▲",  worsened,     delta=int(worsened),  delta_color="inverse")
                k2.metric("Defects improved ▼",  improved,     delta=-int(improved), delta_color="normal")
                k3.metric("New defects",          new_defects)
                k4.metric("Resolved defects",     resolved)

                st.markdown("---")

                # Filter
                show_filter = st.radio(
                    "Show",
                    ["All","Worsened ▲","Improved ▼","New","Stable"],
                    horizontal=True,
                )
                if show_filter == "Worsened ▲":
                    comp_show = comparison[comparison["delta"] > 0.02]
                elif show_filter == "Improved ▼":
                    comp_show = comparison[comparison["delta"] < -0.02]
                elif show_filter == "New":
                    comp_show = comparison[comparison["score_a"] == 0]
                elif show_filter == "Stable":
                    comp_show = comparison[comparison["delta_label"].str.startswith("≈")]
                else:
                    comp_show = comparison

                # Colour the delta column
                def colour_delta(val):
                    if isinstance(val, str):
                        if val.startswith("▲"): return "color: #A32D2D; font-weight:600"
                        if val.startswith("▼"): return "color: #0F6E56; font-weight:600"
                    return "color: #888"

                disp = comp_show[["defect","tier","score_a","count_a",
                                   "score_b","count_b","delta_label"]].copy()
                disp.columns = ["Defect","Tier","Score A","NRCs A",
                                "Score B","NRCs B","Change"]

                st.dataframe(
                    disp.style.applymap(colour_delta, subset=["Change"]),
                    use_container_width=True,
                    height=500,
                )

                # Delta bar chart
                top_delta = (
                    comparison[comparison["delta"].abs() > 0.02]
                    .nlargest(20, "delta")
                )
                if not top_delta.empty:
                    fig = px.bar(
                        top_delta,
                        x="delta", y="defect",
                        orientation="h",
                        color="delta",
                        color_continuous_scale="RdYlGn_r",
                        title="Score change between runs (positive = worsened)",
                        labels={"delta":"Score change","defect":"Defect"},
                        height=max(350, len(top_delta)*28),
                    )
                    fig.update_layout(
                        yaxis=dict(autorange="reversed"),
                        paper_bgcolor="rgba(0,0,0,0)",
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=280, r=20, t=50, b=30),
                    )
                    st.plotly_chart(fig, use_container_width=True)
