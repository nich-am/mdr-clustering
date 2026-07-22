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
from datetime import timezone, timedelta
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import plotly.express as px

from core.pipeline   import run_pipeline, build_excel
from core.materials  import (load_mrm, build_material_summary, summarise_by_defect,
                             top_parts_across_fleet, load_rop_db,
                             build_workscope_material_table, workscope_table_stats,
                             load_alt_mat_db, build_alternate_material_recommendations,
                             alt_mat_stats)
from core.pdf_export import generate_pdf
from core.storage    import (
    save_run, load_run_history, load_run_scores,
    load_run_materials, load_workscope_materials,
    delete_run, compare_runs,
)
from core.charts import (
    scatter_map, ranked_bar, fleet_grouped_bar,
    frequency_heatmap, manhour_bar, score_components_bar,
    tier_donut, damage_distribution, cluster_size_dist,
)

# ── Timezone helper ───────────────────────────────────────────────────────
def _to_wib(ts_str: str) -> str:
    """Convert a UTC timestamp string to WIB (GMT+7) for display."""
    try:
        import pandas as pd
        ts = pd.Timestamp(ts_str)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        ts_wib = ts.astimezone(timezone(timedelta(hours=7)))
        return ts_wib.strftime("%Y-%m-%d %H:%M WIB")
    except Exception:
        return str(ts_str)


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

        st.markdown("### 3 · Run metadata")
        workscope = st.text_input("Workscope", placeholder="e.g. 6Y+12Y C-Check")
        ac_type   = ""
        notes     = st.text_area("Notes (optional)", height=80)

        st.markdown("---")
        run_btn = st.button("▶ Run analysis", type="primary", width='stretch')

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
                    workscope_table = pd.DataFrame()
                    rop_db = pd.DataFrame()
                    alt_mat_recs = pd.DataFrame()

                    # Load Non-ROP database from bundled file
                    try:
                        import os
                        rop_path = os.path.join(
                            os.path.dirname(__file__), "data", "rop_database.xlsx"
                        )
                        rop_db = load_rop_db(open(rop_path, "rb"))
                    except Exception as ex:
                        st.warning(f"Could not load Non-ROP DB: {ex}")
                        rop_db = pd.DataFrame()

                    if mrm_dict:
                        mat_detail      = build_material_summary(results["df"], mrm_dict, results["scores"])
                        mat_summary     = summarise_by_defect(mat_detail)
                        top_parts       = top_parts_across_fleet(mat_detail)
                        workscope_table = build_workscope_material_table(
                            mrm_dict,
                            rop_db=rop_db if not rop_db.empty else None,
                        )

                        # Load bundled Alternate Material database and build recommendations
                        try:
                            alt_mat_path = os.path.join(
                                os.path.dirname(__file__), "data", "alt_material_database.xlsx"
                            )
                            alt_mat_db = load_alt_mat_db(open(alt_mat_path, "rb"))
                            if not workscope_table.empty:
                                alt_mat_recs = build_alternate_material_recommendations(
                                    workscope_table, alt_mat_db,
                                    rop_db=rop_db if not rop_db.empty else None,
                                )
                        except Exception as ex:
                            st.warning(f"Could not load Alternate Material DB: {ex}")
                            alt_mat_recs = pd.DataFrame()

                    results.update({
                        "mrm_dict": mrm_dict, "mat_detail": mat_detail,
                        "mat_summary": mat_summary, "top_parts": top_parts,
                        "workscope_table": workscope_table, "rop_db": rop_db,
                        "alt_mat_recs": alt_mat_recs,
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
        mat_detail      = R["mat_detail"]
        top_parts       = R["top_parts"]
        workscope_table = R.get("workscope_table", pd.DataFrame())
        rop_db          = R.get("rop_db", pd.DataFrame())
        alt_mat_recs    = R.get("alt_mat_recs", pd.DataFrame())
        workscope       = R.get("workscope", workscope or "")
        ac_type         = R.get("ac_type",   ac_type   or "")
        notes           = R.get("notes",     notes     or "")
        has_mrm         = not mat_detail.empty
        has_workscope   = not workscope_table.empty
        has_alt_mat     = not alt_mat_recs.empty

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
        if has_workscope and not rop_db.empty:
            from core.materials import workscope_table_stats
            stats = workscope_table_stats(workscope_table, len(projects))
            n_matched = int((workscope_table["Min-Maxed?"] != "—").sum()) if "Min-Maxed?" in workscope_table.columns else 0
            st.info(
                f"📦 **{stats.get('fleet_wide_parts',0)}** parts used in all aircraft · "
                f"❌ **{stats.get('not_min_maxed',0)}** not yet min-maxed · "
                f"✅ **{stats.get('already_min_maxed',0)}** already min-maxed · "
                f"🔍 **{n_matched}** of {stats.get('total_unique_parts',0)} parts matched in Non-ROP DB"
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
        base_names  = ["📊 Common Defects", "🌍 Found on Every Aircraft", "🗺️ Similarity Map",
                       "🔥 Repair Time Impact", "📈 How Scoring Works", "📋 Full Data Table"]
        mrm_names   = ["🔩 Parts Needed per Defect"] if has_mrm else []
        ws_names    = ["📦 All Materials Used"] if has_workscope else []
        mm_names    = ["🎯 Min-Max Recommendation"] if has_workscope else []
        nf_names    = ["❓ Not Found in DB"] if has_workscope else []
        alt_names   = ["🔁 Alternate Materials"] if has_workscope else []
        tabs = st.tabs(base_names + mrm_names + ws_names + mm_names + nf_names + alt_names)

        tab_ranked, tab_fleet, tab_map, tab_mhrs, tab_score, tab_data = tabs[:6]
        idx = 6
        tab_matdef = None
        if has_mrm:
            tab_matdef = tabs[idx]; idx += 1
        tab_ws = None
        if has_workscope:
            tab_ws = tabs[idx]; idx += 1
        tab_minmax = None
        if has_workscope:
            tab_minmax = tabs[idx]; idx += 1
        tab_notfound = None
        if has_workscope:
            tab_notfound = tabs[idx]; idx += 1
        tab_altmat = None
        if has_workscope:
            tab_altmat = tabs[idx]; idx += 1

        n_ac = len(projects)
        n_fleet_wide   = int((scores["tier"] == "Fleet-wide").sum())
        n_common_tier  = int((scores["tier"] == "Common").sum())
        n_isolated     = int((scores["tier"] == "Isolated").sum())

        # ── Common Defects ───────────────────────────────────────────────
        with tab_ranked:
            st.markdown(
                f"#### {len(scores)} defect patterns found across {n_ac} aircraft"
            )
            st.markdown(
                f"Every defect below is scored from **0 to 1** based on how often it shows up "
                f"and how costly it is to fix. The higher the score, the more it's worth paying "
                f"attention to. Out of all defects found: **{n_fleet_wide}** showed up on "
                f"*every* aircraft, **{n_common_tier}** showed up on *most* aircraft, and "
                f"**{n_isolated}** were *one-off* findings on a single aircraft."
            )
            with st.expander("ℹ️ What do Fleet-wide / Common / Isolated mean?"):
                st.markdown(
                    "- **🌍 Fleet-wide** — this defect was found on **every single aircraft** "
                    "in this batch. Likely a recurring or systemic issue worth addressing "
                    "in the maintenance plan.\n"
                    "- **🔶 Common** — this defect was found on **most, but not all** aircraft. "
                    "Worth monitoring — it may become fleet-wide over time.\n"
                    "- **🔹 Isolated** — this defect was found on **only one aircraft**. "
                    "Likely a one-off issue specific to that aircraft's condition or history."
                )
            st.markdown(
                "Score = **50%** how many aircraft it appears on "
                "+ **30%** how often it shows up + **20%** how many repair hours it costs."
            )
            tf = st.radio("Filter by tier", ["All","Fleet-wide","Common","Isolated"], horizontal=True)
            filt = scores if tf == "All" else scores[scores["tier"] == tf]
            st.plotly_chart(ranked_bar(filt, top_n=top_n_score), width='stretch')
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(damage_distribution(df),  width='stretch')
            with c2: st.plotly_chart(tier_donut(scores),       width='stretch')

        # ── Found on Every Aircraft (Fleet-wide) ─────────────────────────
        with tab_fleet:
            fleet = scores[scores["tier"] == "Fleet-wide"]
            if fleet.empty:
                st.info(
                    f"No defects were found on all {n_ac} aircraft in this batch. "
                    "This means there isn't a single recurring issue affecting every aircraft — "
                    "check the **Common Defects** tab for issues found on most (but not all) aircraft."
                )
            else:
                st.markdown(f"#### {len(fleet)} defects found on **all {n_ac} aircraft**")
                st.markdown(
                    f"These {len(fleet)} defects appeared on every single aircraft in this batch — "
                    "they're the strongest candidates for a fleet-wide fix or a standing "
                    "inspection item, since fixing them once won't be enough; they'll likely "
                    "keep reappearing on future aircraft of the same type."
                )
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
                        st.dataframe(sub.reset_index(drop=True), width='stretch')
                        if has_mrm and n_parts > 0:
                            st.markdown("**Materials:**")
                            st.dataframe(
                                pf[["ac_reg","Order No","Part Number",
                                    "Material Description","Qty Req","UOM","Fulfillment Status"]
                                ].reset_index(drop=True),
                                width='stretch',
                            )
                st.markdown("---")
                st.plotly_chart(fleet_grouped_bar(scores, projects), width='stretch')
                st.plotly_chart(frequency_heatmap(scores, top_n=top_n_score), width='stretch')

        # ── Similarity Map ────────────────────────────────────────────────
        with tab_map:
            st.markdown("#### How similar are the NRC findings to each other?")
            st.markdown(
                "Each dot below is one NRC finding. Findings that describe similar problems "
                "are grouped close together and coloured the same — this is how the app "
                "automatically detects repeating defect patterns from free-text titles, "
                "even when two technicians wrote the same issue differently."
            )
            with st.expander("ℹ️ What do Clustering and Total Defect mean?"):
                st.markdown(
                    "**Clustering** — the app reads every NRC title and groups together the "
                    "ones describing the same underlying problem, even if they were written "
                    "with different wording or abbreviations (e.g. \"LH WING CRACK\" and "
                    "\"WING CRACKED LEFT HAND\" get grouped into the same cluster). Each group "
                    "is called a **cluster**, and becomes one row in the Common Defects and "
                    "Fleet-wide tabs. Dots that don't clearly match any group are left "
                    "unclustered (shown as noise) rather than forced into the wrong group.\n\n"
                    "**Total Defect** — the total count of individual NRC findings that were "
                    "successfully placed into a cluster, i.e. matched with at least one other "
                    "similar finding. This is different from the total number of NRCs "
                    "uploaded, since some one-off findings may not match anything else closely "
                    "enough to form a cluster."
                )
            st.plotly_chart(scatter_map(df, X_2d), width='stretch')
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(cluster_size_dist(df), width='stretch')
            with c2:
                cs = (
                    df[df["cluster_id"] != -1]
                    .groupby("cluster_label")
                    .agg(count=("Description","count"), ac=("project","nunique"))
                    .sort_values("count", ascending=False)
                    .reset_index()
                )
                st.markdown("**Cluster summary**")
                st.dataframe(cs, width='stretch', height=280)

        # ── Repair Time Impact ───────────────────────────────────────────
        with tab_mhrs:
            total_hrs = scores["avg_mhrs"].sum() if "avg_mhrs" in scores.columns else 0
            costliest = scores.nlargest(1, "avg_mhrs") if not scores.empty else None
            st.markdown("#### Which defects take the most time to fix?")
            if costliest is not None and not costliest.empty:
                c0 = costliest.iloc[0]
                st.markdown(
                    f"The most time-consuming defect is **{c0['location']} — {c0['damage_type']}**, "
                    f"averaging **{c0['avg_mhrs']:.1f} hours** per repair. Use this tab to plan "
                    "labour and downtime — defects with high repair time but low frequency can "
                    "still be major schedule risks even if they're not common."
                )
            st.plotly_chart(manhour_bar(scores, top_n=top_n_score), width='stretch')

        # ── How Scoring Works ────────────────────────────────────────────
        with tab_score:
            st.markdown("#### How is each defect's score calculated?")
            st.markdown(
                "Every defect gets a single score between 0 and 1, built from three "
                "ingredients so that the most important issues naturally rise to the top."
            )
            sc1, sc2, sc3 = st.columns(3)
            sc1.metric("Presence", "50%", help="How many aircraft have this defect")
            sc2.metric("Frequency", "30%", help="How often the defect repeats")
            sc3.metric("Repair cost", "20%", help="How many manhours it takes to fix")
            st.markdown(
                "**Presence (50%)** — defects on more aircraft score higher; this is "
                "weighted the heaviest because a problem affecting the whole fleet matters "
                "more than a one-off.\n\n"
                "**Frequency (30%)** — defects that show up many times (not just on many "
                "aircraft, but repeatedly) score higher.\n\n"
                "**Repair cost (20%)** — defects that take longer to fix get a score boost, "
                "since they have a bigger impact on maintenance schedules even if they're rare."
            )
            st.plotly_chart(score_components_bar(scores, top_n=15), width='stretch')

        # ── Full Data Table ───────────────────────────────────────────────
        with tab_data:
            st.markdown("#### Full underlying data")
            st.markdown(
                f"This is the raw data behind every chart in this app — "
                f"all **{len(df):,}** NRC findings with their assigned defect cluster, "
                "plus the full scoring table. Use the filters below to narrow it down, "
                "or export everything from the **Export & Save** section below."
            )
            f1, f2, f3 = st.columns(3)
            sel_ac  = f1.multiselect("Aircraft",  options=projects, default=projects)
            sel_dmg = f2.selectbox("Damage type", ["All"]+sorted(df["damage_type"].dropna().unique().tolist()))
            sel_loc = f3.selectbox("Location",    ["All"]+sorted(df["location"].dropna().unique().tolist()))
            mask = df["project"].isin(sel_ac)
            if sel_dmg != "All": mask &= df["damage_type"] == sel_dmg
            if sel_loc != "All": mask &= df["location"]    == sel_loc
            st.markdown(f"Showing **{mask.sum()}** of {len(df)} NRCs")
            show_cols = [c for c in ["project","Description","Skill Active","Act Mhrs",
                                      "cluster_label","location","damage_type"] if c in df.columns]
            st.dataframe(df[mask][show_cols].reset_index(drop=True), width='stretch', height=380)
            st.markdown("---")
            st.markdown("**Defect scoring table**")
            sc_cols = (["defect_key","tier","total_count","projects_count","avg_mhrs","score"]
                       + [c for c in scores.columns if c.startswith("count_")])
            st.dataframe(scores[[c for c in sc_cols if c in scores.columns]],
                         width='stretch', height=340)

        # ── Parts Needed per Defect ───────────────────────────────────────
        if has_mrm and tab_matdef is not None:
            with tab_matdef:
                n_defects_with_parts = mat_detail["defect_key"].nunique() if not mat_detail.empty else 0
                st.markdown("#### Which parts were requested for each defect?")
                st.markdown(
                    f"**{n_defects_with_parts}** defects in this batch had at least one material "
                    "requested against them. Expand a defect below to see exactly which parts "
                    "were called, in what quantity, and on which aircraft — useful for tracing "
                    "a specific repair back to its material consumption."
                )
                tf2 = st.radio("Filter by tier", ["All","Fleet-wide","Common","Isolated"],
                               horizontal=True, key="mat_tier")
                sf  = scores if tf2 == "All" else scores[scores["tier"] == tf2]
                shown_any = False
                for _, srow in sf.head(top_n_score).iterrows():
                    key   = srow["defect_key"]
                    parts = mat_detail[mat_detail["defect_key"] == key]
                    if parts.empty: continue
                    shown_any = True
                    n_unique = parts["Part Number"].nunique()
                    n_ac_p   = parts["ac_reg"].nunique()
                    with st.expander(
                        f"**{srow['location']} — {srow['damage_type']}**"
                        f"  |  Score: {srow['score']:.3f}"
                        f"  |  {n_unique} unique parts  |  {n_ac_p} aircraft"
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
                                     width='stretch')
                if not shown_any:
                    st.info("None of the defects shown have linked material requests yet.")

        # ── All Materials Used (workscope-level table) ────────────────────
        if has_workscope and tab_ws is not None:
            with tab_ws:
                from core.materials import workscope_table_stats
                stats = workscope_table_stats(workscope_table, n_ac)

                st.markdown("#### Every material requested across this workscope")
                st.markdown(
                    f"This table aggregates **every part** requested (toggle = Y) across "
                    f"all {n_ac} aircraft in this workscope, regardless of which defect it "
                    f"was tied to — **{stats.get('total_unique_parts',0)} unique parts** in "
                    f"total. Use this as a master reference; for stocking recommendations "
                    "specifically, see the **Min-Max Recommendation** tab."
                )

                wk1, wk2, wk3, wk4 = st.columns(4)
                wk1.metric("Unique parts",          stats.get("total_unique_parts",0))
                wk2.metric("Used on all aircraft",  stats.get("fleet_wide_parts",0))
                wk3.metric("❓ Not found in DB",     stats.get("not_found_in_db",0),
                           help="See the Not Found in DB tab for details")
                wk4.metric("Highest score",         f"{stats.get('top_score',0):.0f}")

                with st.expander("ℹ️ What do Calls and Occurrence mean?"):
                    st.markdown(
                        "**Calls** — the total number of maintenance events (order calls) "
                        "across all aircraft analyzed that requested this part. If the same "
                        "aircraft ordered the same part on 3 different orders, that counts as "
                        "3 calls.\n\n"
                        "**Occurrence** — the number of maintenance events that had a request "
                        "for this material. This counts *distinct events*, not how many times "
                        "the part was ordered within one event — so an aircraft calling the "
                        "same part 3 times in one event still counts as 1 occurrence.\n\n"
                        "**Weighted Score** = Total Calls + (Occurrence × 2). Occurrence is "
                        "weighted more heavily because a part needed across many separate "
                        "events is a stronger signal than one event ordering the same part "
                        "repeatedly."
                    )

                st.markdown("---")

                fc1, fc2, fc3, fc4 = st.columns(4)
                occ_filter = fc1.selectbox(
                    "AC occurrence",
                    ["All", f"All {n_ac} aircraft", f"{n_ac-1} aircraft", "1 aircraft"],
                )
                mm_filter  = fc2.selectbox(
                    "Min-Max status",
                    ["All", "❌ Not min-maxed", "✅ Already min-maxed", "— Unknown"],
                )
                type_filter = fc3.selectbox(
                    "Material type",
                    ["All"] + sorted(workscope_table["Type"].dropna().unique().tolist()),
                )
                score_min = fc4.number_input("Min score", min_value=0, value=0, step=1)

                wt = workscope_table.copy()
                if occ_filter == f"All {n_ac} aircraft":
                    wt = wt[wt["Total Occurrence"] == n_ac]
                elif occ_filter == f"{n_ac-1} aircraft":
                    wt = wt[wt["Total Occurrence"] == n_ac-1]
                elif occ_filter == "1 aircraft":
                    wt = wt[wt["Total Occurrence"] == 1]
                if mm_filter != "All":
                    wt = wt[wt["Min-Maxed?"] == mm_filter]
                if type_filter != "All":
                    wt = wt[wt["Type"] == type_filter]
                if score_min > 0:
                    wt = wt[wt["Weighted Score"] >= score_min]

                st.markdown(f"Showing **{len(wt)}** of {len(workscope_table)} materials")

                def highlight_mm(val):
                    if val == "❌ No":  return "color:#FF6B6B;font-weight:600"
                    if val == "✅ Yes": return "color:#5DCAA5;font-weight:600"
                    return ""

                def highlight_score(val):
                    try:
                        v = float(val)
                        if v >= 30: return "font-weight:700;color:#5DCAA5"
                        if v >= 15: return "font-weight:600;color:#85B7EB"
                    except: pass
                    return ""

                call_ac_cols = [c for c in wt.columns if c.startswith("calls_")]
                qty_ac_cols  = [c for c in wt.columns if c.startswith("qty_")]
                rename_map   = {
                    **{c: c.replace("calls_","") + " (calls)" for c in call_ac_cols},
                    **{c: c.replace("qty_","")   + " (qty)"   for c in qty_ac_cols},
                }
                wt_display  = wt.rename(columns=rename_map)

                styled = wt_display.style \
                    .map(highlight_mm,    subset=["Min-Maxed?"]) \
                    .map(highlight_score, subset=["Weighted Score"])

                renamed_num_cols = (
                    [c.replace("calls_","") + " (calls)" for c in call_ac_cols] +
                    [c.replace("qty_","")   + " (qty)"   for c in qty_ac_cols] +
                    ["Total Calls","Total Qty","Occurrence %","Weighted Score","Reorder Point","Max. level"]
                )
                col_cfg = {
                    c: st.column_config.NumberColumn(format="%g%%" if c == "Occurrence %" else "%g")
                    for c in renamed_num_cols if c in wt_display.columns
                }

                st.dataframe(styled, width='stretch', height=520, column_config=col_cfg)

        # ── Min-Max Recommendation (priority stocking list) ───────────────
        if has_workscope and tab_minmax is not None:
            with tab_minmax:
                qty_ac_cols  = [c for c in workscope_table.columns if c.startswith("qty_")]

                fleet_wide_all = workscope_table[
                    workscope_table["Total Occurrence"] == n_ac
                ].sort_values("Weighted Score", ascending=False)

                st.markdown("#### Which parts should the warehouse stock ahead of time?")

                if fleet_wide_all.empty:
                    st.info(
                        f"No parts were requested on all {n_ac} aircraft in this batch, "
                        "so there's no fleet-wide stocking recommendation to show. "
                        "Check the **All Materials Used** tab for the full part list."
                    )
                else:
                    mm_type_opts = ["All"] + sorted(fleet_wide_all["Type"].dropna().unique().tolist())
                    mm_type_filter = st.selectbox("Filter by material type", mm_type_opts, key="mm_type")
                    all_ac_parts = fleet_wide_all if mm_type_filter == "All" \
                                   else fleet_wide_all[fleet_wide_all["Type"] == mm_type_filter]

                    not_mm_fleet = all_ac_parts[all_ac_parts["Min-Maxed?"] == "❌ No"] \
                                   if not all_ac_parts.empty else all_ac_parts
                    already_mm   = all_ac_parts[all_ac_parts["Min-Maxed?"] == "✅ Yes"] \
                                   if not all_ac_parts.empty else all_ac_parts

                    st.markdown(
                        f"**{len(all_ac_parts)}** parts were requested on **every one of the "
                        f"{n_ac} aircraft** in this batch — these are the strongest signals "
                        f"for what to keep in stock ahead of the next maintenance event of "
                        f"this same workscope. Of those, **{len(not_mm_fleet)}** don't yet "
                        f"have a min-max stocking plan set up, while **{len(already_mm)}** "
                        "already do (worth double-checking their stock levels are still "
                        "adequate)."
                    )

                    pp1, pp2, pp3 = st.columns(3)
                    pp1.metric("Parts used on every aircraft", len(all_ac_parts))
                    pp2.metric("🎯 Not yet min-maxed", len(not_mm_fleet),
                               help="Highest priority — needed on every aircraft, no stock plan yet")
                    pp3.metric("✅ Already min-maxed", len(already_mm))

                    if not not_mm_fleet.empty:
                        st.markdown("---")
                        st.markdown(
                            f"### 🎯 Priority list — {len(not_mm_fleet)} parts to set up a min-max plan for"
                        )
                        st.markdown(
                            "> **For the warehouse team:** the table below shows exactly how "
                            "many units **each aircraft** called for every part, plus which "
                            "workcenter(s) requested it. Use the per-aircraft quantities to "
                            "estimate a sensible reorder point (ROP) and max stock level — "
                            "e.g. if an aircraft typically calls 2–4 units per event, a ROP "
                            "around that range avoids both shortages and overstock."
                        )

                        pp_display = not_mm_fleet.rename(
                            columns={c: c.replace("qty_", "") for c in qty_ac_cols}
                        )
                        ac_qty_display_cols = [c.replace("qty_", "") for c in qty_ac_cols]
                        disp_pp_cols = (
                            ["Part Number", "Material Description", "UOM", "Type", "Workcenter(s)"]
                            + ac_qty_display_cols
                            + ["Total Calls", "Total Qty", "Total Occurrence", "Occurrence %", "Weighted Score"]
                        )
                        disp_pp_cols = [c for c in disp_pp_cols if c in pp_display.columns]

                        qty_col_cfg = {
                            c: st.column_config.NumberColumn(format="%g%%" if c == "Occurrence %" else "%g")
                            for c in ac_qty_display_cols + ["Total Calls", "Total Qty", "Occurrence %", "Weighted Score"]
                            if c in pp_display.columns
                        }

                        st.dataframe(
                            pp_display[disp_pp_cols],
                            width='stretch',
                            height=min(440, len(not_mm_fleet) * 38 + 50),
                            column_config=qty_col_cfg,
                        )

                        fig_pp = px.bar(
                            not_mm_fleet.head(20).rename(
                                columns={c: c.replace("qty_", "") for c in qty_ac_cols}
                            ),
                            x="Weighted Score",
                            y="Material Description",
                            orientation="h",
                            color="Total Occurrence",
                            color_continuous_scale="Teal",
                            title="Top 20 priority parts — ranked by weighted score",
                            labels={"Total Occurrence": "# Aircraft"},
                            height=max(350, min(len(not_mm_fleet), 20) * 28),
                        )
                        fig_pp.update_layout(
                            yaxis=dict(autorange="reversed", tickfont_size=10),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=280, r=20, t=50, b=30),
                        )
                        st.plotly_chart(fig_pp, width='stretch')
                    else:
                        st.success(
                            "✅ All fleet-wide parts already have a min-max plan set up. "
                            "Check the section below to verify stock levels are still adequate."
                        )

                    if not already_mm.empty:
                        with st.expander(
                            f"✅ Already min-maxed ({len(already_mm)} parts) — "
                            "verify stock levels are still adequate before the next event"
                        ):
                            mm_display = already_mm.rename(
                                columns={c: c.replace("qty_", "") for c in qty_ac_cols}
                            )
                            ac_qty_display_cols2 = [c.replace("qty_", "") for c in qty_ac_cols]
                            disp_mm_cols = (
                                ["Part Number", "Material Description", "UOM", "Workcenter(s)"]
                                + ac_qty_display_cols2
                                + ["Total Calls", "Total Qty", "Total Occurrence", "Weighted Score",
                                   "Reorder Point", "Max. level"]
                            )
                            disp_mm_cols = [c for c in disp_mm_cols if c in mm_display.columns]
                            st.dataframe(mm_display[disp_mm_cols], width='stretch')

        # ── Not Found in DB ────────────────────────────────────────────────
        if has_workscope and tab_notfound is not None:
            with tab_notfound:
                not_found = workscope_table[workscope_table["Min-Maxed?"] == "—"] \
                            if "Min-Maxed?" in workscope_table.columns else pd.DataFrame()

                st.markdown("#### Parts that couldn't be checked against the Non-ROP database")
                st.markdown(
                    "These parts' status is genuinely **unknown**, not confirmed \"No\" — "
                    "their Part Number wasn't found anywhere in the Non-ROP database, so "
                    "there's no record to say whether they're min-maxed or not. This is "
                    "different from the ❌ **Not yet min-maxed** parts in the Min-Max "
                    "Recommendation tab, which *were* found in the database with an empty ROP."
                )

                if not_found.empty:
                    st.success(
                        "✅ Every part in this workscope was successfully matched against "
                        "the Non-ROP database — nothing to show here."
                    )
                else:
                    nf1, nf2, nf3 = st.columns(3)
                    nf1.metric("Parts not found in DB", len(not_found))
                    nf2.metric("Of total unique parts",
                               f"{len(not_found) / len(workscope_table) * 100:.0f}%"
                               if len(workscope_table) else "0%")
                    nf_fleet = not_found[not_found["Total Occurrence"] == n_ac]
                    nf3.metric("Used on all aircraft", len(nf_fleet),
                               help="Highest priority to investigate — these are common parts "
                                    "with completely unknown stocking status")

                    st.markdown(
                        "> **For the warehouse/database team:** these Part Numbers likely need "
                        "to be **added to the Non-ROP database** (or their Part Number format "
                        "reconciled) before a min-max decision can even be made. Until then, "
                        "their stocking status is a blind spot."
                    )

                    st.markdown("---")
                    if not nf_fleet.empty:
                        st.markdown(f"### 🔎 {len(nf_fleet)} parts used on all {n_ac} aircraft — highest priority to look into")
                        nf_fleet_sorted = nf_fleet.sort_values("Weighted Score", ascending=False)
                        nf_disp_cols = [c for c in ["Part Number","Material Description","UOM","Type",
                                                     "Total Calls","Total Qty","Total Occurrence",
                                                     "Occurrence %","Weighted Score"]
                                        if c in nf_fleet_sorted.columns]
                        st.dataframe(nf_fleet_sorted[nf_disp_cols], width='stretch',
                                     height=min(440, len(nf_fleet_sorted) * 38 + 50))

                    st.markdown("---")
                    st.markdown(f"### All {len(not_found)} unmatched parts")
                    nf_type_opts = ["All"] + sorted(not_found["Type"].dropna().unique().tolist()) \
                                   if "Type" in not_found.columns else ["All"]
                    nf_type_sel = st.selectbox("Material type", nf_type_opts, key="nf_type_filter")
                    nf_filtered = not_found if nf_type_sel == "All" else not_found[not_found["Type"] == nf_type_sel]

                    nf_all_cols = [c for c in ["Part Number","Material Description","UOM","Type",
                                                "Total Calls","Total Qty","Total Occurrence",
                                                "Occurrence %","Weighted Score"]
                                   if c in nf_filtered.columns]
                    st.dataframe(
                        nf_filtered[nf_all_cols].sort_values("Weighted Score", ascending=False),
                        width='stretch', height=520,
                    )


        if has_workscope and tab_altmat is not None:
            with tab_altmat:
                st.markdown("#### Is there already-stocked alternate for a part that isn't min-maxed?")

                if not has_alt_mat:
                    st.info(
                        "None of the parts in this workscope have a known alternate "
                        "listed in GMF's Alternate Material database — nothing to show here."
                    )
                else:
                    astats = alt_mat_stats(alt_mat_recs)
                    st.markdown(
                        f"**{astats.get('parts_with_alternates', 0)}** parts requested in this "
                        f"workscope have at least one known alternate — **"
                        f"{astats.get('total_relationships', 0)}** part-to-alternate relationships "
                        f"in total. Of those, **{astats.get('swap_opportunities', 0)}** are "
                        f"**swap opportunities**: the requested part has no min-max plan, but an "
                        f"interchangeable alternate is *already* min-maxed — meaning the warehouse "
                        f"may not need to set up anything new at all."
                    )

                    a1, a2, a3 = st.columns(3)
                    a1.metric("Parts with known alternates", astats.get("parts_with_alternates", 0))
                    a2.metric("🔁 Swap opportunities", astats.get("swap_opportunities", 0),
                              help="Requested part not min-maxed, but its alternate already is")
                    a3.metric("Total alternate relationships", astats.get("total_relationships", 0))

                    is_swap = (
                        (alt_mat_recs.get("Requested Min-Maxed?", "") == "❌ No") &
                        (alt_mat_recs["Alternate Min-Maxed?"] == "✅ Yes")
                    )
                    swap_df   = alt_mat_recs[is_swap]
                    other_df  = alt_mat_recs[~is_swap]

                    if not swap_df.empty:
                        st.markdown("---")
                        st.markdown(f"### 🔁 {len(swap_df)} swap opportunities")
                        st.markdown(
                            "> **For the warehouse team:** instead of setting up a brand-new "
                            "min-max plan for the part on the left, consider designating the "
                            "already-min-maxed alternate on the right for this purpose — "
                            "confirm interchangeability with engineering first."
                        )
                        swap_disp_cols = [c for c in [
                            "Part Number", "Material Description", "Type", "Requested Min-Maxed?",
                            "Alternate Part Number", "Alternate Kind", "Alternate Min-Maxed?",
                            "Weighted Score", "Total Occurrence",
                        ] if c in swap_df.columns]
                        st.dataframe(
                            swap_df[swap_disp_cols],
                            width='stretch',
                            height=min(500, len(swap_df) * 38 + 50),
                        )

                    st.markdown("---")
                    st.markdown(f"### All alternate relationships ({len(alt_mat_recs)})")
                    st.markdown(
                        "Every known alternate for every part in this workscope — "
                        "including cases where neither the requested part nor its "
                        "alternate is min-maxed yet, shown here for full traceability."
                    )

                    fk1, fk2, fk3 = st.columns(3)
                    req_mm_opts = ["All"] + sorted(alt_mat_recs["Requested Min-Maxed?"].dropna().unique().tolist()) \
                                  if "Requested Min-Maxed?" in alt_mat_recs.columns else ["All"]
                    alt_mm_opts = ["All"] + sorted(alt_mat_recs["Alternate Min-Maxed?"].dropna().unique().tolist())
                    type_opts   = ["All"] + sorted(alt_mat_recs["Type"].dropna().unique().tolist()) \
                                  if "Type" in alt_mat_recs.columns else ["All"]
                    sel_req_mm = fk1.selectbox("Requested part status", req_mm_opts)
                    sel_alt_mm = fk2.selectbox("Alternate part status", alt_mm_opts)
                    sel_type   = fk3.selectbox("Material type", type_opts, key="altmat_type")

                    filtered_alt = alt_mat_recs.copy()
                    if sel_req_mm != "All" and "Requested Min-Maxed?" in filtered_alt.columns:
                        filtered_alt = filtered_alt[filtered_alt["Requested Min-Maxed?"] == sel_req_mm]
                    if sel_alt_mm != "All":
                        filtered_alt = filtered_alt[filtered_alt["Alternate Min-Maxed?"] == sel_alt_mm]
                    if sel_type != "All" and "Type" in filtered_alt.columns:
                        filtered_alt = filtered_alt[filtered_alt["Type"] == sel_type]

                    st.markdown(f"Showing **all {len(filtered_alt)}** matching relationships")
                    all_disp_cols = [c for c in [
                        "Part Number", "Material Description", "Type", "Requested Min-Maxed?",
                        "Alternate Part Number", "Alternate Kind", "Alternate Min-Maxed?",
                        "Weighted Score", "Total Occurrence",
                    ] if c in filtered_alt.columns]
                    st.dataframe(
                        filtered_alt[all_disp_cols],
                        width='stretch',
                        height=600,
                    )

        # ── Downloads + Save ──────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Export & Save")

        excel_bytes = build_excel(df, scores)

        # Generate PDF
        with st.spinner("Generating PDF…"):
            pdf_bytes = generate_pdf(
                df, scores,
                workscope_table if not workscope_table.empty else pd.DataFrame(),
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
                    mat_detail.to_excel(writer, sheet_name="Material_Detail",      index=False)
                if not top_parts.empty:
                    top_parts.to_excel(writer,  sheet_name="PreProvision_List",    index=False)
                if not workscope_table.empty:
                    ws_export = workscope_table.copy()
                    ws_export.columns = [
                        c.replace("qty_","") for c in ws_export.columns
                    ]
                    ws_export.to_excel(writer, sheet_name="Workscope_Materials",   index=False)
                if has_alt_mat:
                    alt_mat_recs.to_excel(writer, sheet_name="Alternate_Materials", index=False)
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
                            workscope_table=workscope_table if not workscope_table.empty else None,
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

        # Run selector
        def _make_label(row):
            parts = [_to_wib(str(row["created_at"]))]
            if str(row.get("workscope", "")).strip():
                parts.append(str(row["workscope"]).strip())
            if str(row.get("ac_type", "")).strip():
                parts.append(str(row["ac_type"]).strip())
            parts.append(str(row.get("aircraft", "—")))
            return "  ·  ".join(parts)

        run_labels = history.apply(_make_label, axis=1).tolist()
        sel_label = st.selectbox("Select a run to view", run_labels, index=0)
        run_row   = history.iloc[run_labels.index(sel_label)]
        run_id    = run_row["id"]

        st.markdown("---")

        # KPIs
        h1, h2, h3, h4, h5 = st.columns(5)
        h1.metric("Total NRCs",    run_row.get("total_nrcs", 0))
        h2.metric("Clusters",      run_row.get("n_clusters", 0))
        h3.metric("Fleet-wide",    run_row.get("n_fleet_wide", 0))
        h4.metric("Aircraft",      run_row.get("aircraft", "—"))
        h5.metric("Date",          _to_wib(str(run_row.get("created_at", "—"))))

        if run_row.get("notes"):
            st.markdown(f"*{run_row['notes']}*")

        st.markdown("---")

        # Load data for this run
        with st.spinner("Loading run data…"):
            run_scores  = load_run_scores(run_id)
            run_ws      = load_workscope_materials(run_id)

        # Backfill Occurrence % for runs saved before this column existed
        if not run_ws.empty and "Occurrence %" not in run_ws.columns and "Total Occurrence" in run_ws.columns:
            _n_ac_backfill = int(run_ws["Total Occurrence"].max()) or 1
            run_ws["Occurrence %"] = (run_ws["Total Occurrence"] / _n_ac_backfill * 100).round(1)

        # Backfill Workcenter(s) for runs saved before this column existed
        if not run_ws.empty:
            if "Workcenter(s)" not in run_ws.columns:
                run_ws["Workcenter(s)"] = "—"
            else:
                run_ws["Workcenter(s)"] = run_ws["Workcenter(s)"].fillna("—")
                run_ws.loc[run_ws["Workcenter(s)"].isin(["", "None", "nan"]), "Workcenter(s)"] = "—"

        n_projects = len(run_row.get("aircraft","").split(", ")) if run_row.get("aircraft") else 1

        # ── History tabs — mirror the new analysis tab structure ─────────────
        hist_base = ["📊 Common Defects", "🌍 Found on Every Aircraft",
                     "🔥 Repair Time Impact", "📈 How Scoring Works", "📋 Full Data Table"]
        hist_ws   = ["📦 All Materials Used", "🎯 Min-Max Recommendation"] if not run_ws.empty else []

        # Alternate Materials can be recomputed live from the saved workscope table
        # (it only needs the workscope table + bundled reference DBs, neither of
        # which requires per-run storage)
        hist_alt_recs = pd.DataFrame()
        if not run_ws.empty:
            try:
                hist_rop_path = os.path.join(os.path.dirname(__file__), "data", "rop_database.xlsx")
                hist_rop_db   = load_rop_db(open(hist_rop_path, "rb"))
                hist_alt_path = os.path.join(os.path.dirname(__file__), "data", "alt_material_database.xlsx")
                hist_alt_db   = load_alt_mat_db(open(hist_alt_path, "rb"))
                hist_alt_recs = build_alternate_material_recommendations(
                    run_ws, hist_alt_db,
                    rop_db=hist_rop_db if not hist_rop_db.empty else None,
                )
            except Exception:
                hist_alt_recs = pd.DataFrame()
        hist_has_alt = not hist_alt_recs.empty
        hist_nf      = ["❓ Not Found in DB"]      if not run_ws.empty else []
        hist_altmat  = ["🔁 Alternate Materials"] if not run_ws.empty else []

        hist_tabs = st.tabs(hist_base + hist_ws + hist_nf + hist_altmat)

        ht_ranked   = hist_tabs[0]
        ht_fleet    = hist_tabs[1]
        ht_mhrs     = hist_tabs[2]
        ht_score    = hist_tabs[3]
        ht_data     = hist_tabs[4]
        ht_ws       = hist_tabs[5] if len(hist_ws) > 0 else None
        ht_minmax   = hist_tabs[6] if len(hist_ws) > 0 else None
        ht_notfound = hist_tabs[5 + len(hist_ws)] if hist_nf else None
        ht_altmat   = hist_tabs[5 + len(hist_ws) + len(hist_nf)] if hist_altmat else None

        n_fleet_h    = int((run_scores["tier"] == "Fleet-wide").sum()) if not run_scores.empty else 0
        n_common_h   = int((run_scores["tier"] == "Common").sum())     if not run_scores.empty else 0
        n_isolated_h = int((run_scores["tier"] == "Isolated").sum())   if not run_scores.empty else 0

        if run_scores.empty:
            for t in hist_tabs[:5]:
                with t:
                    st.info("No defect scores saved for this run.")
        else:
            # ── Common Defects ────────────────────────────────────────────
            with ht_ranked:
                st.markdown(f"#### {len(run_scores)} defect patterns found across {n_projects} aircraft")
                st.markdown(
                    f"Every defect below is scored from **0 to 1** based on how often it shows up "
                    f"and how costly it is to fix. Out of all defects found: **{n_fleet_h}** showed "
                    f"up on *every* aircraft, **{n_common_h}** showed up on *most* aircraft, and "
                    f"**{n_isolated_h}** were *one-off* findings on a single aircraft."
                )
                with st.expander("ℹ️ What do Fleet-wide / Common / Isolated mean?"):
                    st.markdown(
                        "- **🌍 Fleet-wide** — found on **every single aircraft** in this batch.\n"
                        "- **🔶 Common** — found on **most, but not all** aircraft.\n"
                        "- **🔹 Isolated** — found on **only one aircraft**."
                    )
                st.markdown(
                    "Score = **50%** how many aircraft it appears on "
                    "+ **30%** how often it shows up + **20%** how many repair hours it costs."
                )
                tf_h = st.radio("Filter by tier", ["All","Fleet-wide","Common","Isolated"],
                                horizontal=True, key="hist_tier")
                filt_h = run_scores if tf_h == "All" else run_scores[run_scores["tier"] == tf_h]
                st.plotly_chart(ranked_bar(filt_h, top_n=25), width='stretch')
                c1h, c2h = st.columns(2)
                with c1h: st.plotly_chart(tier_donut(run_scores), width='stretch')
                with c2h:
                    tier_counts = run_scores.groupby("tier").size().reset_index(name="count")
                    st.markdown("**Defect tier breakdown**")
                    st.dataframe(tier_counts, width='stretch', height=180)

            # ── Found on Every Aircraft ───────────────────────────────────
            with ht_fleet:
                fleet_h = run_scores[run_scores["tier"] == "Fleet-wide"]
                if fleet_h.empty:
                    st.info(
                        f"No defects were found on all {n_projects} aircraft in this run. "
                        "Check **Common Defects** for issues found on most aircraft."
                    )
                else:
                    st.markdown(f"#### {len(fleet_h)} defects found on **all {n_projects} aircraft**")
                    st.markdown(
                        f"These {len(fleet_h)} defects appeared on every aircraft in this batch — "
                        "strongest candidates for a fleet-wide fix or standing inspection item."
                    )
                    count_cols_h = [c for c in fleet_h.columns if c.startswith("count_")]
                    for _, row_h in fleet_h.iterrows():
                        with st.expander(
                            f"**{row_h['location']} — {row_h['damage_type']}**"
                            f"  |  Score: {row_h['score']:.3f}"
                            f"  |  {int(row_h['total_count'])} NRCs"
                            f"  |  {row_h['avg_mhrs']:.1f}h avg"
                        ):
                            if count_cols_h:
                                # Per-aircraft breakdown available (rare for saved runs)
                                ac_ch = st.columns(len(count_cols_h))
                                for i, col in enumerate(count_cols_h):
                                    ac_ch[i].metric(col.replace("count_",""), int(row_h[col]))
                            else:
                                # Saved runs only keep the aggregate score row —
                                # per-aircraft breakdown isn't persisted, so show
                                # what we do have instead of leaving this blank.
                                mk1, mk2, mk3 = st.columns(3)
                                mk1.metric("Total NRCs", int(row_h["total_count"]))
                                mk2.metric("Aircraft affected",
                                           int(row_h["projects_count"]) if "projects_count" in row_h else n_projects)
                                mk3.metric("Avg repair time", f"{row_h['avg_mhrs']:.1f}h")
                    st.markdown("---")
                    if count_cols_h:
                        st.plotly_chart(fleet_grouped_bar(run_scores,
                            [c.replace("count_","") for c in count_cols_h]), width='stretch')
                    else:
                        # No per-aircraft breakdown saved for this run — show a
                        # score-ranked bar instead of an empty grouped chart.
                        fleet_h_ranked = fleet_h.copy()
                        fleet_h_ranked["label"] = fleet_h_ranked["location"] + " — " + fleet_h_ranked["damage_type"]
                        fig_fleet_h = px.bar(
                            fleet_h_ranked.sort_values("score"),
                            x="score", y="label", orientation="h",
                            color="total_count", color_continuous_scale="Teal",
                            title="Fleet-wide defects — ranked by score",
                            labels={"score":"Score","label":"","total_count":"NRC count"},
                            height=max(350, len(fleet_h_ranked)*30),
                        )
                        fig_fleet_h.update_layout(
                            yaxis=dict(tickfont_size=10),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=280, r=20, t=50, b=30),
                        )
                        st.plotly_chart(fig_fleet_h, width='stretch')
                    st.plotly_chart(frequency_heatmap(run_scores, top_n=25), width='stretch')

            # ── Repair Time Impact ────────────────────────────────────────
            with ht_mhrs:
                costliest_h = run_scores.nlargest(1, "avg_mhrs") if not run_scores.empty else None
                st.markdown("#### Which defects take the most time to fix?")
                if costliest_h is not None and not costliest_h.empty:
                    c0h = costliest_h.iloc[0]
                    st.markdown(
                        f"The most time-consuming defect is **{c0h['location']} — {c0h['damage_type']}**, "
                        f"averaging **{c0h['avg_mhrs']:.1f} hours** per repair."
                    )
                st.plotly_chart(manhour_bar(run_scores, top_n=25), width='stretch')

            # ── How Scoring Works ─────────────────────────────────────────
            with ht_score:
                st.markdown("#### How is each defect's score calculated?")
                st.markdown(
                    "Every defect gets a single score between 0 and 1, built from three ingredients "
                    "so that the most important issues naturally rise to the top."
                )
                sc1h, sc2h, sc3h = st.columns(3)
                sc1h.metric("Presence", "50%", help="How many aircraft have this defect")
                sc2h.metric("Frequency", "30%", help="How often the defect repeats")
                sc3h.metric("Repair cost", "20%", help="How many manhours it takes to fix")
                st.markdown(
                    "**Presence (50%)** — defects on more aircraft score higher.\n\n"
                    "**Frequency (30%)** — defects that show up many times score higher.\n\n"
                    "**Repair cost (20%)** — defects that take longer to fix get a score boost."
                )
                st.plotly_chart(score_components_bar(run_scores, top_n=15), width='stretch')

            # ── Full Data Table ───────────────────────────────────────────
            with ht_data:
                st.markdown("#### Defect scoring table for this run")
                st.markdown(
                    "Full scoring table saved for this run. Includes tier, NRC count, "
                    "per-aircraft counts, average manhours, and final weighted score."
                )
                sc_cols_h = (
                    ["defect_key","location","sub_component","damage_type",
                     "tier","total_count","projects_count","avg_mhrs","score"]
                    + [c for c in run_scores.columns if c.startswith("count_")]
                )
                st.dataframe(
                    run_scores[[c for c in sc_cols_h if c in run_scores.columns]]
                    .reset_index(drop=True),
                    width='stretch', height=480
                )

        # ── All Materials Used ────────────────────────────────────────────
        if ht_ws is not None and not run_ws.empty:
            with ht_ws:
                st.markdown("#### Every material requested across this workscope")
                st.markdown(
                    f"This table aggregates every part requested (toggle = Y) across all aircraft "
                    f"in this saved run — **{len(run_ws)} unique parts** in total."
                )

                ws_n_ac = int(run_ws["Total Occurrence"].max()) if not run_ws.empty else n_projects

                wk1h, wk2h, wk3h, wk4h, wk5h = st.columns(5)
                wk1h.metric("Unique parts",         len(run_ws))
                wk2h.metric("Used on all aircraft", int((run_ws["Total Occurrence"] == ws_n_ac).sum()))
                wk3h.metric("Not yet min-maxed ❌",  int((run_ws.get("Min-Maxed?","") == "❌ No").sum()))
                wk4h.metric("Already min-maxed ✅",  int((run_ws.get("Min-Maxed?","") == "✅ Yes").sum()))
                wk5h.metric("❓ Not found in DB",    int((run_ws.get("Min-Maxed?","") == "—").sum()),
                            help="See the Not Found in DB tab for details")

                with st.expander("ℹ️ What do Calls and Occurrence mean?"):
                    st.markdown(
                        "**Calls** — the total number of maintenance events (order calls) "
                        "across all aircraft analyzed that requested this part. If the same "
                        "aircraft ordered the same part on 3 different orders, that counts as "
                        "3 calls.\n\n"
                        "**Occurrence** — the number of maintenance events that had a request "
                        "for this material. This counts *distinct events*, not how many times "
                        "the part was ordered within one event — so an aircraft calling the "
                        "same part 3 times in one event still counts as 1 occurrence.\n\n"
                        "**Weighted Score** = Total Calls + (Occurrence × 2). Occurrence is "
                        "weighted more heavily because a part needed across many separate "
                        "events is a stronger signal than one event ordering the same part "
                        "repeatedly."
                    )

                st.markdown("---")

                # Filters
                hfc1, hfc2, hfc3 = st.columns(3)
                h_mm_filter   = hfc1.selectbox("Min-Max status",
                    ["All","❌ Not min-maxed","✅ Already min-maxed","— Unknown"],
                    key="hist_mm")
                h_occ_filter  = hfc2.selectbox("AC occurrence",
                    ["All"] + [str(i) for i in range(ws_n_ac, 0, -1)],
                    key="hist_occ")
                h_type_filter = hfc3.selectbox("Material type",
                    ["All"] + sorted(run_ws["Type"].dropna().unique().tolist()) if "Type" in run_ws.columns else ["All"],
                    key="hist_type")

                wth = run_ws.copy()
                if h_mm_filter != "All":
                    mm_map = {"❌ Not min-maxed": "❌ No", "✅ Already min-maxed": "✅ Yes",
                              "— Unknown": "—"}
                    wth = wth[wth["Min-Maxed?"] == mm_map.get(h_mm_filter, h_mm_filter)]
                if h_occ_filter != "All":
                    wth = wth[wth["Total Occurrence"] == int(h_occ_filter)]
                if h_type_filter != "All" and "Type" in wth.columns:
                    wth = wth[wth["Type"] == h_type_filter]

                st.markdown(f"Showing **{len(wth)}** of {len(run_ws)} materials")

                def _h_mm(val):
                    if val == "❌ No":  return "color:#FF6B6B;font-weight:600"
                    if val == "✅ Yes": return "color:#5DCAA5;font-weight:600"
                    return ""
                def _h_sc(val):
                    try:
                        v = float(val)
                        if v >= 30: return "font-weight:700;color:#5DCAA5"
                        if v >= 15: return "font-weight:600;color:#85B7EB"
                    except: pass
                    return ""

                h_call_cols = [c for c in wth.columns if c.startswith("calls_")]
                h_qty_cols  = [c for c in wth.columns if c.startswith("qty_")]
                h_rename    = {
                    **{c: c.replace("calls_","") + " (calls)" for c in h_call_cols},
                    **{c: c.replace("qty_","")   + " (qty)"   for c in h_qty_cols},
                }
                wth_disp = wth.rename(columns=h_rename)

                h_styled = wth_disp.style \
                    .map(_h_mm, subset=["Min-Maxed?"]) \
                    .map(_h_sc, subset=["Weighted Score"])

                h_num_cols = (
                    [c.replace("calls_","") + " (calls)" for c in h_call_cols] +
                    [c.replace("qty_","")   + " (qty)"   for c in h_qty_cols] +
                    ["Grand Total","Total Calls","Total Qty","Occurrence %","Weighted Score",
                     "Reorder Point","Max. level"]
                )
                h_col_cfg = {
                    c: st.column_config.NumberColumn(format="%g%%" if c == "Occurrence %" else "%g")
                    for c in h_num_cols if c in wth_disp.columns
                }
                st.dataframe(h_styled, width='stretch', height=520, column_config=h_col_cfg)

        # ── Min-Max Recommendation ────────────────────────────────────────
        if ht_minmax is not None and not run_ws.empty:
            with ht_minmax:
                ws_n_ac2 = int(run_ws["Total Occurrence"].max()) if not run_ws.empty else n_projects
                qty_ac_cols_h = [c for c in run_ws.columns if c.startswith("qty_")]

                fleet_wide_h_all = run_ws[
                    run_ws["Total Occurrence"] == ws_n_ac2
                ].sort_values("Weighted Score", ascending=False)

                st.markdown("#### Which parts should the warehouse stock ahead of time?")

                if fleet_wide_h_all.empty:
                    st.info("No parts were requested on every aircraft in this saved run.")
                else:
                    h_type_opts = ["All"] + sorted(fleet_wide_h_all["Type"].dropna().unique().tolist()) \
                                  if "Type" in fleet_wide_h_all.columns else ["All"]
                    h_mm_type_filter = st.selectbox("Filter by material type", h_type_opts, key="hist_mm_type")
                    all_ac_h = fleet_wide_h_all if h_mm_type_filter == "All" \
                               else fleet_wide_h_all[fleet_wide_h_all["Type"] == h_mm_type_filter]

                    not_mm_h  = all_ac_h[all_ac_h["Min-Maxed?"] == "❌ No"]  if not all_ac_h.empty else all_ac_h
                    already_h = all_ac_h[all_ac_h["Min-Maxed?"] == "✅ Yes"] if not all_ac_h.empty else all_ac_h

                    st.markdown(
                        f"**{len(all_ac_h)}** parts were requested on every one of the "
                        f"{ws_n_ac2} aircraft in this run. Of those, **{len(not_mm_h)}** "
                        f"don't yet have a min-max plan, while **{len(already_h)}** already do."
                    )
                    pp1h, pp2h, pp3h = st.columns(3)
                    pp1h.metric("Parts used on every aircraft", len(all_ac_h))
                    pp2h.metric("🎯 Not yet min-maxed",         len(not_mm_h))
                    pp3h.metric("✅ Already min-maxed",          len(already_h))

                    if not not_mm_h.empty:
                        st.markdown("---")
                        st.markdown(f"### 🎯 Priority list — {len(not_mm_h)} parts to set up a min-max plan for")
                        pp_h_disp = not_mm_h.rename(
                            columns={c: c.replace("qty_","") for c in qty_ac_cols_h}
                        )
                        ac_qty_h = [c.replace("qty_","") for c in qty_ac_cols_h]
                        disp_h_cols = (
                            ["Part Number","Material Description","UOM","Type","Workcenter(s)"]
                            + ac_qty_h
                            + ["Total Calls","Total Qty","Total Occurrence","Occurrence %","Weighted Score"]
                        )
                        disp_h_cols = [c for c in disp_h_cols if c in pp_h_disp.columns]
                        h_qty_cfg = {
                            c: st.column_config.NumberColumn(format="%g%%" if c == "Occurrence %" else "%g")
                            for c in ac_qty_h + ["Total Calls","Total Qty","Occurrence %","Weighted Score"]
                            if c in pp_h_disp.columns
                        }
                        st.dataframe(
                            pp_h_disp[disp_h_cols],
                            width='stretch',
                            height=min(440, len(not_mm_h) * 38 + 50),
                            column_config=h_qty_cfg,
                        )
                        fig_h = px.bar(
                            not_mm_h.head(20).rename(
                                columns={c: c.replace("qty_","") for c in qty_ac_cols_h}
                            ),
                            x="Weighted Score", y="Material Description",
                            orientation="h", color="Total Occurrence",
                            color_continuous_scale="Teal",
                            title="Top 20 priority parts — ranked by weighted score",
                            labels={"Total Occurrence": "# Aircraft"},
                            height=max(350, min(len(not_mm_h), 20) * 28),
                        )
                        fig_h.update_layout(
                            yaxis=dict(autorange="reversed", tickfont_size=10),
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(l=280, r=20, t=50, b=30),
                        )
                        st.plotly_chart(fig_h, width='stretch')
                    else:
                        st.success("✅ All fleet-wide parts already have a min-max plan set up.")

                    if not already_h.empty:
                        with st.expander(
                            f"✅ Already min-maxed ({len(already_h)} parts) — "
                            "verify stock levels before next event"
                        ):
                            mm_h_disp = already_h.rename(
                                columns={c: c.replace("qty_","") for c in qty_ac_cols_h}
                            )
                            ac_qty_h2 = [c.replace("qty_","") for c in qty_ac_cols_h]
                            disp_mm_h = (
                                ["Part Number","Material Description","UOM","Workcenter(s)"]
                                + ac_qty_h2
                                + ["Total Calls","Total Qty","Total Occurrence","Weighted Score",
                                   "Reorder Point","Max. level"]
                            )
                            disp_mm_h = [c for c in disp_mm_h if c in mm_h_disp.columns]
                            st.dataframe(mm_h_disp[disp_mm_h], width='stretch')

        # ── Not Found in DB ──────────────────────────────────────────────────
        if ht_notfound is not None and not run_ws.empty:
            with ht_notfound:
                nf_h = run_ws[run_ws["Min-Maxed?"] == "—"] \
                       if "Min-Maxed?" in run_ws.columns else pd.DataFrame()

                st.markdown("#### Parts that couldn't be checked against the Non-ROP database")
                st.markdown(
                    "These parts' status is genuinely **unknown**, not confirmed \"No\" — "
                    "their Part Number wasn't found anywhere in the Non-ROP database, so "
                    "there's no record to say whether they're min-maxed or not. This is "
                    "different from the ❌ **Not yet min-maxed** parts in the Min-Max "
                    "Recommendation tab, which *were* found in the database with an empty ROP."
                )

                if nf_h.empty:
                    st.success(
                        "✅ Every part in this saved run was successfully matched against "
                        "the Non-ROP database — nothing to show here."
                    )
                else:
                    nf_ws_n_ac = int(run_ws["Total Occurrence"].max()) if not run_ws.empty else n_projects
                    nf1h, nf2h, nf3h = st.columns(3)
                    nf1h.metric("Parts not found in DB", len(nf_h))
                    nf2h.metric("Of total unique parts",
                                f"{len(nf_h) / len(run_ws) * 100:.0f}%" if len(run_ws) else "0%")
                    nf_fleet_h = nf_h[nf_h["Total Occurrence"] == nf_ws_n_ac]
                    nf3h.metric("Used on all aircraft", len(nf_fleet_h),
                                help="Highest priority to investigate — these are common parts "
                                     "with completely unknown stocking status")

                    st.markdown(
                        "> **For the warehouse/database team:** these Part Numbers likely need "
                        "to be **added to the Non-ROP database** (or their Part Number format "
                        "reconciled) before a min-max decision can even be made."
                    )

                    st.markdown("---")
                    if not nf_fleet_h.empty:
                        st.markdown(f"### 🔎 {len(nf_fleet_h)} parts used on all {nf_ws_n_ac} aircraft — highest priority")
                        nf_fleet_h_sorted = nf_fleet_h.sort_values("Weighted Score", ascending=False)
                        nf_h_disp_cols = [c for c in ["Part Number","Material Description","UOM","Type",
                                                       "Total Calls","Total Qty","Total Occurrence",
                                                       "Occurrence %","Weighted Score"]
                                          if c in nf_fleet_h_sorted.columns]
                        st.dataframe(nf_fleet_h_sorted[nf_h_disp_cols], width='stretch',
                                     height=min(440, len(nf_fleet_h_sorted) * 38 + 50))

                    st.markdown("---")
                    st.markdown(f"### All {len(nf_h)} unmatched parts")
                    nf_h_all_cols = [c for c in ["Part Number","Material Description","UOM","Type",
                                                  "Total Calls","Total Qty","Total Occurrence",
                                                  "Occurrence %","Weighted Score"]
                                     if c in nf_h.columns]
                    st.dataframe(
                        nf_h[nf_h_all_cols].sort_values("Weighted Score", ascending=False),
                        width='stretch', height=520,
                    )

        # ── Alternate Materials ──────────────────────────────────────────────
        if ht_altmat is not None:
            with ht_altmat:
                st.markdown("#### Is there already-stocked alternate for a part that isn't min-maxed?")

                if not hist_has_alt:
                    st.info(
                        "None of the parts in this saved workscope have a known alternate "
                        "listed in GMF's Alternate Material database — nothing to show here."
                    )
                else:
                    h_astats = alt_mat_stats(hist_alt_recs)
                    st.markdown(
                        f"**{h_astats.get('parts_with_alternates', 0)}** parts requested in this "
                        f"workscope have at least one known alternate — **"
                        f"{h_astats.get('total_relationships', 0)}** part-to-alternate relationships "
                        f"in total. Of those, **{h_astats.get('swap_opportunities', 0)}** are "
                        f"**swap opportunities**: the requested part has no min-max plan, but an "
                        f"interchangeable alternate is *already* min-maxed — meaning the warehouse "
                        f"may not need to set up anything new at all."
                    )

                    ha1, ha2, ha3 = st.columns(3)
                    ha1.metric("Parts with known alternates", h_astats.get("parts_with_alternates", 0))
                    ha2.metric("🔁 Swap opportunities", h_astats.get("swap_opportunities", 0),
                               help="Requested part not min-maxed, but its alternate already is")
                    ha3.metric("Total alternate relationships", h_astats.get("total_relationships", 0))

                    h_is_swap = (
                        (hist_alt_recs.get("Requested Min-Maxed?", "") == "❌ No") &
                        (hist_alt_recs["Alternate Min-Maxed?"] == "✅ Yes")
                    )
                    h_swap_df  = hist_alt_recs[h_is_swap]

                    if not h_swap_df.empty:
                        st.markdown("---")
                        st.markdown(f"### 🔁 {len(h_swap_df)} swap opportunities")
                        st.markdown(
                            "> **For the warehouse team:** instead of setting up a brand-new "
                            "min-max plan for the part on the left, consider designating the "
                            "already-min-maxed alternate on the right for this purpose — "
                            "confirm interchangeability with engineering first."
                        )
                        h_swap_cols = [c for c in [
                            "Part Number", "Material Description", "Type", "Requested Min-Maxed?",
                            "Alternate Part Number", "Alternate Kind", "Alternate Min-Maxed?",
                            "Weighted Score", "Total Occurrence",
                        ] if c in h_swap_df.columns]
                        st.dataframe(
                            h_swap_df[h_swap_cols],
                            width='stretch',
                            height=min(500, len(h_swap_df) * 38 + 50),
                        )

                    st.markdown("---")
                    st.markdown(f"### All alternate relationships ({len(hist_alt_recs)})")
                    st.markdown(
                        "Every known alternate for every part in this workscope — "
                        "including cases where neither the requested part nor its "
                        "alternate is min-maxed yet, shown here for full traceability."
                    )

                    h_fk1, h_fk2, h_fk3 = st.columns(3)
                    h_req_opts = ["All"] + sorted(hist_alt_recs["Requested Min-Maxed?"].dropna().unique().tolist()) \
                                 if "Requested Min-Maxed?" in hist_alt_recs.columns else ["All"]
                    h_alt_opts = ["All"] + sorted(hist_alt_recs["Alternate Min-Maxed?"].dropna().unique().tolist())
                    h_type_opts = ["All"] + sorted(hist_alt_recs["Type"].dropna().unique().tolist()) \
                                  if "Type" in hist_alt_recs.columns else ["All"]
                    h_sel_req = h_fk1.selectbox("Requested part status", h_req_opts, key="hist_alt_req")
                    h_sel_alt = h_fk2.selectbox("Alternate part status", h_alt_opts, key="hist_alt_alt")
                    h_sel_type = h_fk3.selectbox("Material type", h_type_opts, key="hist_alt_type")

                    h_filtered = hist_alt_recs.copy()
                    if h_sel_req != "All" and "Requested Min-Maxed?" in h_filtered.columns:
                        h_filtered = h_filtered[h_filtered["Requested Min-Maxed?"] == h_sel_req]
                    if h_sel_alt != "All":
                        h_filtered = h_filtered[h_filtered["Alternate Min-Maxed?"] == h_sel_alt]
                    if h_sel_type != "All" and "Type" in h_filtered.columns:
                        h_filtered = h_filtered[h_filtered["Type"] == h_sel_type]

                    st.markdown(f"Showing **all {len(h_filtered)}** matching relationships")
                    h_all_cols = [c for c in [
                        "Part Number", "Material Description", "Type", "Requested Min-Maxed?",
                        "Alternate Part Number", "Alternate Kind", "Alternate Min-Maxed?",
                        "Weighted Score", "Total Occurrence",
                    ] if c in h_filtered.columns]
                    st.dataframe(
                        h_filtered[h_all_cols],
                        width='stretch',
                        height=600,
                    )

        # ── Downloads ─────────────────────────────────────────────────────
        st.markdown("---")
        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            excel_url = run_row.get("excel_url","")
            if excel_url:
                st.link_button("⬇️ Download Excel", excel_url, type="primary")
        with dl2:
            pdf_url = run_row.get("pdf_url","")
            if pdf_url:
                st.link_button("⬇️ Download PDF", pdf_url)
        with dl3:
            if st.button("🗑️ Delete this run", type="secondary"):
                delete_run(run_id)
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
        def _make_compare_label(row):
            parts = [_to_wib(str(row["created_at"]))]
            if str(row.get("workscope", "")).strip():
                parts.append(str(row["workscope"]).strip())
            parts.append(str(row.get("aircraft", "—")))
            return " · ".join(parts)

        run_labels = history.apply(_make_compare_label, axis=1).tolist()
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
                    disp.style.map(colour_delta, subset=["Change"]),
                    width='stretch',
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
                    st.plotly_chart(fig, width='stretch')
