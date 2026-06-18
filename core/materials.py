"""
core/materials.py
-----------------
Material request analysis for NRC findings clustering.

Two separate views:
  A) Per-defect materials  — which parts were requested for each defect combo
  B) Workscope-level table — aggregate view across the whole event, like the
     Min-Max Recommendation table, with:
       • per-AC qty columns
       • Total Calls (order count) and Total Qty (actual quantity)
       • Total Occurrence (how many AC called this part)
       • Weighted Score  = total_qty + (ac_count × 2)
       • Min-Max status  from the Non-ROP database

Join keys:
  MRM  Part Number  (format "PN" or "PN:VendorNumber" e.g. "NSA936020-01" or "NSA936020-01:F5442")
  NRC  Order No     (matches MRM Order column)
  ROP  Material     ("PN:VendorNumber" format — joined on full key first, base PN as fallback)
"""

import io
import pandas as pd


# Columns we keep from the MRM file
MRM_KEEP = [
    "Order",
    "Part Number",
    "Material Description",
    "Qty Req",
    "UOM",
    "Type",
    "Material Fullfilment Status",
    "Workcenter",
]


# ── Load MRM ───────────────────────────────────────────────────────────────
def load_mrm(fileobj) -> pd.DataFrame:
    """Read MRM Excel, keep only toggle='y' rows."""
    df = pd.read_excel(fileobj)
    df.columns = df.columns.str.strip()

    toggle_col = next((c for c in df.columns if c.lower() == "toggle"), None)
    if toggle_col is None:
        raise ValueError("No 'toggle' column found in MRM file.")
    df = df[df[toggle_col].astype(str).str.strip().str.lower() == "y"].copy()

    keep = [c for c in MRM_KEEP if c in df.columns]
    df   = df[keep].copy()
    df["Order"] = df["Order"].astype(str).str.strip()
    return df.reset_index(drop=True)


# ── Load Non-ROP / Min-Max database ───────────────────────────────────────
def load_rop_db(fileobj, ac_type_filter: str = "") -> pd.DataFrame:
    """
    Read the Non-ROP database Excel.
    Optionally filter by ObjectType (e.g. '320-200').
    Returns a DataFrame with columns: Material, MPN, ROP, Reorder Point, Max. level
    """
    df = pd.read_excel(fileobj)
    df.columns = df.columns.str.strip()

    # ac_type_filter is used as a preference, not a hard filter.
    # If a part appears under the matching AC type, that row takes priority.
    # Parts not found under the matching type still get looked up across all types.
    # This prevents common parts (shared across 320/737 fleets) from being missed.
    if ac_type_filter and "ObjectType" in df.columns:
        preferred = df[df["ObjectType"].astype(str).str.strip() == ac_type_filter.strip()]
        others    = df[~df.index.isin(preferred.index)]
        # Keep preferred rows first; for parts not in preferred, also keep "others"
        # but deduplicate on Material so preferred takes priority
        df = pd.concat([preferred, others], ignore_index=True)
        df = df.drop_duplicates(subset=["Material"], keep="first")

    # Normalise ROP: Yes → min-maxed, No / blank → not min-maxed
    if "ROP" in df.columns:
        df["min_maxed"] = df["ROP"].astype(str).str.strip().str.lower() == "yes"
    else:
        df["min_maxed"] = False

    keep = [c for c in ["Material","MPN","Material Description","Reorder Point",
                         "Max. level","BUn","min_maxed","ObjectType","MTyp","PGr"]
            if c in df.columns]
    return df[keep].reset_index(drop=True)


# ── A) Per-defect material detail ─────────────────────────────────────────
def join_materials_to_nrc(nrc_df: pd.DataFrame, mrm_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join MRM onto NRC via Order No."""
    nrc = nrc_df.copy()
    nrc["_order_key"] = nrc["Order No"].astype(str).str.strip()
    mrm = mrm_df.copy()
    mrm["_order_key"] = mrm["Order"].astype(str).str.strip()
    joined = nrc.merge(mrm.drop(columns=["Order"]), on="_order_key", how="left")
    return joined.drop(columns=["_order_key"])


def build_material_summary(
    nrc_df: pd.DataFrame,
    mrm_dict: dict,
    scores_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each defect combo, pull all Y-toggle materials from matching orders.
    Returns detail table: one row per (defect, ac_reg, part_number).
    """
    rows = []
    for _, score_row in scores_df.iterrows():
        loc  = score_row["location"]
        dmg  = score_row["damage_type"]
        key  = score_row["defect_key"]
        tier = score_row["tier"]
        sc   = score_row["score"]

        matching = nrc_df[
            (nrc_df["location"] == loc) & (nrc_df["damage_type"] == dmg)
        ][["project","Order No","Description"]].copy()
        matching["Order No"] = matching["Order No"].astype(str).str.strip()
        if matching.empty:
            continue

        for ac_reg, mrm_df in mrm_dict.items():
            ac_nrcs = matching[matching["project"] == ac_reg]
            if ac_nrcs.empty:
                continue
            order_nos = set(ac_nrcs["Order No"].unique())
            mats = mrm_df[mrm_df["Order"].astype(str).isin(order_nos)].copy()
            if mats.empty:
                continue
            for _, mat in mats.iterrows():
                rows.append({
                    "defect_key":           key,
                    "location":             loc,
                    "damage_type":          dmg,
                    "tier":                 tier,
                    "score":                sc,
                    "ac_reg":               ac_reg,
                    "Order No":             mat["Order"],
                    "Part Number":          mat.get("Part Number", ""),
                    "Material Description": mat.get("Material Description", ""),
                    "Qty Req":              mat.get("Qty Req", 0),
                    "UOM":                  mat.get("UOM", ""),
                    "Type":                 mat.get("Type", ""),
                    "Fulfillment Status":   mat.get("Material Fullfilment Status", ""),
                })
    return pd.DataFrame(rows)


def summarise_by_defect(mat_detail: pd.DataFrame) -> pd.DataFrame:
    if mat_detail.empty:
        return pd.DataFrame()
    return (
        mat_detail
        .groupby(["defect_key","tier","score","Part Number",
                  "Material Description","UOM","Type"])
        .agg(ac_count=("ac_reg","nunique"), total_qty=("Qty Req","sum"),
             ac_list=("ac_reg", lambda x: ", ".join(sorted(x.unique()))))
        .reset_index()
        .sort_values(["score","ac_count","total_qty"], ascending=[False,False,False])
    )


def top_parts_across_fleet(mat_detail: pd.DataFrame, min_ac: int = 2) -> pd.DataFrame:
    if mat_detail.empty:
        return pd.DataFrame()
    agg = (
        mat_detail
        .groupby(["Part Number","Material Description","UOM","Type"])
        .agg(defect_count=("defect_key","nunique"), ac_count=("ac_reg","nunique"),
             total_qty=("Qty Req","sum"),
             defects=("defect_key", lambda x: " · ".join(sorted(x.unique())[:3])),
             ac_list=("ac_reg", lambda x: ", ".join(sorted(x.unique()))),
             avg_score=("score","mean"))
        .reset_index()
    )
    agg = agg[agg["ac_count"] >= min_ac]
    return agg.sort_values(["ac_count","avg_score","total_qty"],
                           ascending=[False,False,False]).reset_index(drop=True)


# ── B) Workscope-level material table (the "Min-Max Recommendation" view) ──
def build_workscope_material_table(
    mrm_dict: dict,                     # {ac_reg: mrm_df}
    rop_db: pd.DataFrame = None,        # from load_rop_db(), optional
) -> pd.DataFrame:
    """
    Aggregate ALL y-toggle materials across all aircraft for the whole workscope.

    Columns produced:
      Part Number, Material Description, UOM, Type,
      qty_{ac_reg} for each AC,
      Total Calls, Total Qty,
      Total Occurrence   (how many AC called this part),
      Weighted Score     = Total Calls + (Total Occurrence × 2),
      Min-Maxed?         (Yes / No from ROP DB, or N/A if no DB uploaded),
      Reorder Point,
      Max. Level
    """
    if not mrm_dict:
        return pd.DataFrame()

    ac_regs = sorted(mrm_dict.keys())

    # Aggregate per part per AC:
    #   - call_count  = number of distinct orders that requested this part (for scoring)
    #   - total_qty   = sum of Qty Req (for reference, kept as a separate column)
    per_ac = []
    for ac, mrm_df in mrm_dict.items():
        agg = (
            mrm_df
            .groupby(["Part Number","Material Description","UOM","Type"])
            .agg(
                **{f"calls_{ac}": ("Order", "nunique"),
                   f"qty_{ac}":   ("Qty Req", "sum")}
            )
            .reset_index()
        )
        per_ac.append(agg)

    # Merge all ACs
    from functools import reduce
    merged = reduce(
        lambda l, r: pd.merge(l, r,
                               on=["Part Number","Material Description","UOM","Type"],
                               how="outer"),
        per_ac
    )

    call_cols = [f"calls_{ac}" for ac in ac_regs]
    qty_cols  = [f"qty_{ac}"   for ac in ac_regs]

    for col in call_cols + qty_cols:
        if col not in merged.columns:
            merged[col] = 0
    merged[call_cols] = merged[call_cols].fillna(0).astype(int)
    merged[qty_cols]  = merged[qty_cols].fillna(0).round(2)

    # Derived columns
    # Total Calls      = number of order-calls across all ACs (used for scoring)
    # Total Occurrence  = how many ACs called this part at least once
    # Weighted Score   = Total Calls + (Total Occurrence × 2)
    #                     heaviest weight on parts called across many ACs,
    #                     secondary weight on repeat calls within an AC
    merged["Total Calls"]      = merged[call_cols].sum(axis=1)
    merged["Total Qty"]        = merged[qty_cols].sum(axis=1).round(2)
    merged["Total Occurrence"] = (merged[call_cols] > 0).sum(axis=1)
    merged["Weighted Score"]   = merged["Total Calls"] + merged["Total Occurrence"] * 2

    # ── Join ROP database ──────────────────────────────────────────────────
    if rop_db is not None and not rop_db.empty and "Material" in rop_db.columns:
        # ROP Material format:  "PN:VendorNumber"  e.g. "ASG33:36131"
        # MRM Part Number may:  include vendor number  → "ASG33:36131"  (exact match)
        #                    or omit  vendor number    → "ASG33"         (base-PN match)
        # Strategy: try exact match first; for unmatched rows fall back to
        # matching on the base PN (everything before the colon).

        rop_lookup = rop_db[["Material","min_maxed","Reorder Point","Max. level"]].copy()
        rop_lookup["_key_full"] = rop_lookup["Material"].astype(str).str.strip().str.upper()
        rop_lookup["_key_base"] = rop_lookup["_key_full"].str.split(":").str[0]
        # For base-PN lookup, keep the first (preferred AC type) row per base PN
        rop_base = rop_lookup.drop_duplicates(subset=["_key_base"], keep="first")

        merged["_key_full"] = merged["Part Number"].astype(str).str.strip().str.upper()
        merged["_key_base"] = merged["_key_full"].str.split(":").str[0]

        # Pass 1 — exact full-key join
        merged = merged.merge(
            rop_lookup[["_key_full","min_maxed","Reorder Point","Max. level"]]
            .drop_duplicates(subset=["_key_full"]),
            on="_key_full", how="left"
        )

        # Pass 2 — fill unmatched rows via base-PN join
        unmatched = merged["min_maxed"].isna()
        if unmatched.any():
            base_fill = merged.loc[unmatched, ["_key_base"]].merge(
                rop_base[["_key_base","min_maxed","Reorder Point","Max. level"]],
                on="_key_base", how="left"
            )
            merged.loc[unmatched, "min_maxed"]      = base_fill["min_maxed"].values
            merged.loc[unmatched, "Reorder Point"]  = base_fill["Reorder Point"].values
            merged.loc[unmatched, "Max. level"]     = base_fill["Max. level"].values

        merged["Min-Maxed?"] = merged["min_maxed"].map(
            {True: "✅ Yes", False: "❌ No"}
        ).fillna("—")
        merged["Reorder Point"] = merged["Reorder Point"].fillna(0).round(2)
        merged["Max. level"]    = merged["Max. level"].fillna(0).round(2)
        merged = merged.drop(columns=["min_maxed","_key_full","_key_base"], errors="ignore")
    else:
        merged["Min-Maxed?"]    = "—"
        merged["Reorder Point"] = 0
        merged["Max. level"]    = 0

    # Sort by Weighted Score descending
    merged = merged.sort_values("Weighted Score", ascending=False).reset_index(drop=True)

    # Reorder columns for display:
    #   calls_{ac} = number of orders that called this part  ← used for scoring
    #   qty_{ac}   = total quantity requested                ← kept for reference
    front_cols = ["Part Number","Material Description","UOM","Type"]
    ac_call_cols = call_cols   # e.g. calls_PK-GLV, calls_PK-GLX, calls_PK-GLZ
    ac_qty_cols  = qty_cols    # e.g. qty_PK-GLV,   qty_PK-GLX,   qty_PK-GLZ
    end_cols   = ["Total Calls","Total Qty","Total Occurrence","Weighted Score",
                  "Min-Maxed?","Reorder Point","Max. level"]
    all_cols   = front_cols + ac_call_cols + ac_qty_cols + end_cols
    result = merged[[c for c in all_cols if c in merged.columns]].copy()

    # Smart number formatting: whole numbers shown without decimals
    numeric_display_cols = call_cols + qty_cols + ["Total Calls","Total Qty","Weighted Score","Reorder Point","Max. level"]
    for col in numeric_display_cols:
        if col not in result.columns:
            continue
        col_data = pd.to_numeric(result[col], errors="coerce")
        rounded  = col_data.round(2)
        if (rounded.dropna() == rounded.dropna().apply(lambda x: int(x))).all():
            result[col] = rounded.fillna(0).astype(int)
        else:
            result[col] = rounded.apply(
                lambda x: int(x) if pd.notna(x) and x == int(x) else x
            )

    return result


def workscope_table_stats(df: pd.DataFrame, n_ac: int) -> dict:
    """Summary KPIs for the workscope material table."""
    if df.empty:
        return {}
    # Support both old ("Grand Total") and new ("Total Calls") column names
    calls_col = "Total Calls" if "Total Calls" in df.columns else "Grand Total"
    mm_col_ok = "Min-Maxed?" in df.columns
    return {
        "total_unique_parts":    len(df),
        "fleet_wide_parts":      int((df["Total Occurrence"] == n_ac).sum()),
        "not_min_maxed":         int((df["Min-Maxed?"] == "❌ No").sum())  if mm_col_ok else 0,
        "already_min_maxed":     int((df["Min-Maxed?"] == "✅ Yes").sum()) if mm_col_ok else 0,
        "top_score":             float(df["Weighted Score"].max()),
        "total_qty_all":         float(df[calls_col].sum()),
    }
