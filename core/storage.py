"""
core/storage.py
---------------
Supabase integration for saving and loading pipeline run history.
Uses direct HTTP for storage uploads (supabase-py v2 compatibility).
"""

import uuid
from typing import Optional

import requests
import pandas as pd
import streamlit as st


def _get_creds():
    """Return (url, key) from Streamlit secrets, URL stripped of trailing slash."""
    url = st.secrets["supabase"]["url"].rstrip("/")
    key = st.secrets["supabase"]["key"]
    return url, key


def _client():
    """Create Supabase client from Streamlit secrets."""
    try:
        from supabase import create_client
        url, key = _get_creds()
        return create_client(url, key)
    except Exception as e:
        st.error(f"Supabase connection failed: {e}")
        return None


def _upload_file(run_id: str, filename: str, data: bytes, content_type: str) -> str:
    """
    Upload a file to the nrc-reports bucket via direct REST API.
    Returns the public URL string, or "" on failure.
    """
    url, key = _get_creds()
    path = f"{run_id}/{filename}"
    endpoint = f"{url}/storage/v1/object/nrc-reports/{path}"

    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": content_type,
        "x-upsert": "true",          # overwrite if exists
    }

    try:
        resp = requests.post(endpoint, headers=headers, data=data, timeout=30)
        if resp.status_code in (200, 201):
            public_url = f"{url}/storage/v1/object/public/nrc-reports/{path}"
            return public_url
        else:
            st.warning(f"Could not upload {filename}: HTTP {resp.status_code} — {resp.text[:200]}")
            return ""
    except Exception as e:
        st.warning(f"Could not upload {filename}: {e}")
        return ""


# ── Save a run ─────────────────────────────────────────────────────────────
def save_run(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    top_parts: pd.DataFrame,
    excel_bytes: bytes,
    pdf_bytes: Optional[bytes],
    workscope: str,
    ac_type: str,
    notes: str,
    workscope_table: Optional[pd.DataFrame] = None,
) -> Optional[str]:
    """
    Save a complete pipeline run to Supabase.
    Returns the run_id (UUID string) or None on failure.
    """
    sb = _client()
    if sb is None:
        return None

    run_id     = str(uuid.uuid4())
    projects   = sorted(df["project"].unique().tolist())
    n_fleet    = int((scores["tier"] == "Fleet-wide").sum()) if not scores.empty else 0
    n_clusters = int(df[df["cluster_id"] != -1]["cluster_id"].nunique())

    # ── Upload Excel ───────────────────────────────────────────────────────
    excel_url = _upload_file(
        run_id, "nrc_results.xlsx", excel_bytes,
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # ── Upload PDF ─────────────────────────────────────────────────────────
    pdf_url = ""
    if pdf_bytes:
        pdf_url = _upload_file(run_id, "nrc_report.pdf", pdf_bytes, "application/pdf")

    # ── Insert run row ─────────────────────────────────────────────────────
    try:
        sb.table("runs").insert({
            "id":           run_id,
            "workscope":    workscope,
            "ac_type":      ac_type,
            "aircraft":     projects,
            "total_nrcs":   len(df),
            "n_clusters":   n_clusters,
            "n_fleet_wide": n_fleet,
            "notes":        notes,
            "excel_url":    excel_url,
            "pdf_url":      pdf_url,
        }).execute()
    except Exception as e:
        st.error(f"Could not save run metadata: {e}")
        return None

    # ── Insert defect scores ───────────────────────────────────────────────
    if not scores.empty:
        score_rows = []
        for _, row in scores.iterrows():
            score_rows.append({
                "run_id":         run_id,
                "location":       row.get("location", ""),
                "damage_type":    row.get("damage_type", ""),
                "tier":           row.get("tier", ""),
                "score":          float(row.get("score", 0)),
                "total_count":    int(row.get("total_count", 0)),
                "projects_count": int(row.get("projects_count", 0)),
                "avg_mhrs":       float(row.get("avg_mhrs", 0)),
                "presence_raw":   float(row.get("presence_raw", 0)),
                "freq_norm":      float(row.get("freq_norm", 0)),
                "mhrs_norm":      float(row.get("mhrs_norm", 0)),
            })
        try:
            for i in range(0, len(score_rows), 100):
                sb.table("defect_scores").insert(score_rows[i:i+100]).execute()
        except Exception as e:
            st.warning(f"Could not save defect scores: {e}")

    # ── Insert material recommendations ────────────────────────────────────
    if not top_parts.empty:
        mat_rows = []
        for _, row in top_parts.iterrows():
            mat_rows.append({
                "run_id":               run_id,
                "part_number":          str(row.get("Part Number", "")),
                "material_description": str(row.get("Material Description", "")),
                "uom":                  str(row.get("UOM", "")),
                "material_type":        str(row.get("Type", "")),
                "ac_count":             int(row.get("ac_count", 0)),
                "total_qty":            float(row.get("total_qty", 0)),
                "avg_score":            float(row.get("avg_score", 0)),
                "defects":              str(row.get("defects", "")),
                "ac_list":              str(row.get("ac_list", "")),
            })
        try:
            for i in range(0, len(mat_rows), 100):
                sb.table("material_recommendations").insert(mat_rows[i:i+100]).execute()
        except Exception as e:
            st.warning(f"Could not save material recommendations: {e}")

    # ── Insert workscope materials ─────────────────────────────────────────
    if workscope_table is not None and not workscope_table.empty:
        ws_rows = []
        for _, row in workscope_table.iterrows():
            # Support both old (Grand Total) and new (Total Calls) column names
            calls_val = row.get("Total Calls", row.get("Grand Total", 0))
            qty_val   = row.get("Total Qty",   row.get("Grand Total", 0))
            maxlv_val = row.get("Max. level",  row.get("Max Level", 0))
            ws_rows.append({
                "run_id":               run_id,
                "part_number":          str(row.get("Part Number", "")),
                "material_description": str(row.get("Material Description", "")),
                "uom":                  str(row.get("UOM", "")),
                "material_type":        str(row.get("Type", "")),
                "workcenter":           str(row.get("Workcenter(s)", "—")),
                "total_occurrence":     int(row.get("Total Occurrence", 0)),
                "grand_total":          float(calls_val or 0),
                "total_qty":            float(qty_val   or 0),
                "weighted_score":       float(row.get("Weighted Score", 0)),
                "min_maxed":            str(row.get("Min-Maxed?", "—")),
                "reorder_point":        float(row.get("Reorder Point", 0) or 0),
                "max_level":            float(maxlv_val or 0),
            })
        try:
            for i in range(0, len(ws_rows), 100):
                sb.table("workscope_materials").insert(ws_rows[i:i+100]).execute()
        except Exception as e:
            st.warning(f"Could not save workscope materials: {e}")

    return run_id


# ── Load run history list ──────────────────────────────────────────────────
def load_run_history() -> pd.DataFrame:
    """Load all past runs as a DataFrame, newest first."""
    sb = _client()
    if sb is None:
        return pd.DataFrame()
    try:
        resp = (
            sb.table("runs")
            .select("id, created_at, workscope, ac_type, aircraft, "
                    "total_nrcs, n_clusters, n_fleet_wide, notes, excel_url, pdf_url")
            .order("created_at", desc=True)
            .execute()
        )
        if not resp.data:
            return pd.DataFrame()
        df = pd.DataFrame(resp.data)
        df["created_at"] = pd.to_datetime(df["created_at"]).dt.strftime("%Y-%m-%d %H:%M")
        df["aircraft"]   = df["aircraft"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else str(x)
        )
        return df
    except Exception as e:
        st.warning(f"Could not load run history: {e}")
        return pd.DataFrame()


# ── Load scores for a specific run ────────────────────────────────────────
def load_run_scores(run_id: str) -> pd.DataFrame:
    sb = _client()
    if sb is None:
        return pd.DataFrame()
    try:
        resp = (
            sb.table("defect_scores")
            .select("*")
            .eq("run_id", run_id)
            .order("score", desc=True)
            .execute()
        )
        return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not load scores: {e}")
        return pd.DataFrame()


# ── Load materials for a specific run ─────────────────────────────────────
def load_run_materials(run_id: str) -> pd.DataFrame:
    sb = _client()
    if sb is None:
        return pd.DataFrame()
    try:
        resp = (
            sb.table("material_recommendations")
            .select("*")
            .eq("run_id", run_id)
            .order("ac_count", desc=True)
            .execute()
        )
        return pd.DataFrame(resp.data) if resp.data else pd.DataFrame()
    except Exception as e:
        st.warning(f"Could not load materials: {e}")
        return pd.DataFrame()


# ── Load workscope materials for a specific run ───────────────────────────
def load_workscope_materials(run_id: str) -> pd.DataFrame:
    sb = _client()
    if sb is None:
        return pd.DataFrame()
    try:
        resp = (
            sb.table("workscope_materials")
            .select("*")
            .eq("run_id", run_id)
            .order("weighted_score", desc=True)
            .execute()
        )
        if not resp.data:
            return pd.DataFrame()
        df = pd.DataFrame(resp.data)
        # Rename to display-friendly column names
        df = df.rename(columns={
            "part_number":          "Part Number",
            "material_description": "Material Description",
            "uom":                  "UOM",
            "material_type":        "Type",
            "workcenter":           "Workcenter(s)",
            "total_occurrence":     "Total Occurrence",
            "grand_total":          "Total Calls",
            "total_qty":            "Total Qty",
            "weighted_score":       "Weighted Score",
            "min_maxed":            "Min-Maxed?",
            "reorder_point":        "Reorder Point",
            "max_level":            "Max Level",
        })
        return df
    except Exception as e:
        st.warning(f"Could not load workscope materials: {e}")
        return pd.DataFrame()


# ── Delete a run ───────────────────────────────────────────────────────────
def delete_run(run_id: str) -> bool:
    sb = _client()
    if sb is None:
        return False
    try:
        for fname in ["nrc_results.xlsx", "nrc_report.pdf"]:
            try:
                sb.storage.from_("nrc-reports").remove([f"{run_id}/{fname}"])
            except Exception:
                pass
        sb.table("runs").delete().eq("id", run_id).execute()
        return True
    except Exception as e:
        st.warning(f"Could not delete run: {e}")
        return False


# ── Compare two runs: defect score delta ──────────────────────────────────
def compare_runs(run_id_a: str, run_id_b: str) -> pd.DataFrame:
    """Side-by-side comparison of defect scores between two runs."""
    scores_a = load_run_scores(run_id_a).rename(
        columns={"score": "score_a", "total_count": "count_a"}
    )
    scores_b = load_run_scores(run_id_b).rename(
        columns={"score": "score_b", "total_count": "count_b"}
    )

    if scores_a.empty or scores_b.empty:
        return pd.DataFrame()

    key_cols = ["location", "damage_type"]
    merged = scores_a[key_cols + ["score_a", "count_a", "tier"]].merge(
        scores_b[key_cols + ["score_b", "count_b"]],
        on=key_cols,
        how="outer",
    ).fillna(0)

    merged["delta"]       = (merged["score_b"] - merged["score_a"]).round(4)
    merged["delta_label"] = merged["delta"].apply(
        lambda d: f"▲ {d:+.3f}" if d > 0.02 else (f"▼ {d:+.3f}" if d < -0.02 else "≈ stable")
    )
    merged["defect"]      = merged["location"] + " — " + merged["damage_type"]
    merged = merged.sort_values("score_b", ascending=False).reset_index(drop=True)
    return merged
