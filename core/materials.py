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
def load_rop_db(fileobj) -> pd.DataFrame:
    """
    Read the Non-ROP database Excel.
    No AC type filtering — the database covers all aircraft types.
    Deduplicates on Material so each part number appears only once.
    Returns a DataFrame with columns: Material, MPN, ROP, Reorder Point, Max. level
    """
    df = pd.read_excel(fileobj)
    df.columns = df.columns.str.strip()

    # Deduplicate on Material (part + vendor code) — keep first occurrence
    if "Material" in df.columns:
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


# ── Load Actual Consumption transactions ───────────────────────────────────
# Movement types where a negative Quantity means the material was actually
# consumed/issued, and a positive Quantity means it was returned/reversed
# (i.e. un-consumed) back into stock on the SAME order.
_CONSUME_MVT = {"963", "965", "Z02", "Z04"}   # issues — negative qty
_RETURN_MVT  = {"964", "Z11", "Z12"}          # reversals/removes — positive qty

def load_actual_consumption(fileobj) -> pd.DataFrame:
    """
    Read a SAP goods-movement transaction export (consume + reversal rows)
    and net them per Order + Material to get the TRUE actual consumption.

    A part issued then later reversed on the same order nets to ~0 —
    this is exactly why we can't just sum "Unplanned Issue" rows alone.

    Expected columns: MvT, Order, Material, Material Description, Quantity, BUn
    (matches the raw SAP export format, e.g. Actual_Consumption_*.xlsx)

    Returns a DataFrame with one row per Order + Material:
        Order, Part Number, Actual Consumption Qty, UOM
    Net quantity is signed (can be negative if returns exceed issues —
    shown as-is, not clipped, since that usually flags a data issue worth
    investigating rather than hiding).
    """
    df = pd.read_excel(fileobj)
    df.columns = df.columns.str.strip()

    required = {"Order", "Material", "Quantity"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Actual consumption file missing columns: {missing}")

    df["Order"]    = df["Order"].astype(str).str.strip()
    df["Material"] = df["Material"].astype(str).str.strip()

    uom_col = "BUn" if "BUn" in df.columns else ("EUn" if "EUn" in df.columns else None)

    grouped = (
        df.groupby(["Order", "Material"], as_index=False)
        .agg(**{
            "Actual Consumption Qty": ("Quantity", "sum"),
            **({"UOM": (uom_col, "first")} if uom_col else {}),
        })
    )
    # Net consumption = -(sum of signed transaction quantities)
    # Issues are negative, reversals/returns are positive in the source data,
    # so negating gives a positive "amount consumed" figure — but we leave
    # the sign as computed (not clipped) per the requirement to show it as-is.
    grouped["Actual Consumption Qty"] = (-grouped["Actual Consumption Qty"]).round(2)
    grouped = grouped.rename(columns={"Material": "Part Number"})
    return grouped.reset_index(drop=True)


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
    consumption_dict: dict = None,      # {ac_reg: consumption_df}, optional
) -> pd.DataFrame:
    """
    Aggregate ALL y-toggle materials across all aircraft for the whole workscope.

    Columns produced:
      Part Number, Material Description, UOM, Type, Workcenter(s),
      qty_{ac_reg} for each AC,
      consumed_{ac_reg} for each AC   (only if consumption_dict provided),
      Total Calls, Total Qty,
      Total Actual Consumption        (only if consumption_dict provided),
      Total Occurrence   (how many AC called this part),
      Occurrence %       = Total Occurrence / total AC count × 100,
      Weighted Score     = Total Calls + (Total Occurrence × 2),
      Min-Maxed?         (Yes / No from ROP DB, or N/A if no DB uploaded),
      Reorder Point,
      Max. Level

    consumption_dict values come from load_actual_consumption(), keyed with
    one row per Order + Part Number. Since this table is Part-Number-level
    (not per-order), consumption is summed across all orders for that AC.
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

    # ── Merge actual consumption (optional) ─────────────────────────────────
    # consumption_dict values are per Order + Part Number (net of reversals).
    # This table is Part-Number-level, so sum consumption across all orders
    # for each AC to get one consumed_{ac} figure per part.
    has_consumption = bool(consumption_dict)
    consumed_cols = []
    if has_consumption:
        consumed_cols = [f"consumed_{ac}" for ac in ac_regs]
        for ac in ac_regs:
            cons_df = consumption_dict.get(ac)
            col_name = f"consumed_{ac}"
            if cons_df is None or cons_df.empty:
                merged[col_name] = 0.0
                continue
            per_part = (
                cons_df.groupby("Part Number", as_index=False)["Actual Consumption Qty"]
                .sum()
                .rename(columns={"Actual Consumption Qty": col_name})
            )
            merged = merged.merge(per_part, on="Part Number", how="left")
            merged[col_name] = merged[col_name].fillna(0).round(2)
        merged["Total Actual Consumption"] = merged[consumed_cols].sum(axis=1).round(2)

    # Derived columns
    # Total Calls      = number of order-calls across all ACs (used for scoring)
    # Total Occurrence  = how many ACs called this part at least once
    # Weighted Score   = Total Calls + (Total Occurrence × 2)
    #                     heaviest weight on parts called across many ACs,
    #                     secondary weight on repeat calls within an AC
    merged["Total Calls"]      = merged[call_cols].sum(axis=1)
    merged["Total Qty"]        = merged[qty_cols].sum(axis=1).round(2)
    merged["Total Occurrence"] = (merged[call_cols] > 0).sum(axis=1)
    merged["Occurrence %"]     = (merged["Total Occurrence"] / len(ac_regs) * 100).round(1)
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

    # ── Aggregate Workcenter (responsible unit) per part, across all ACs ────
    # A part can be requested by more than one workcenter across different
    # orders/aircraft — collect the distinct set as a readable list so the
    # warehouse team can see who's responsible for managing it.
    workcenter_frames = []
    for ac, mrm_df in mrm_dict.items():
        if "Workcenter" in mrm_df.columns:
            wc = mrm_df[["Part Number", "Workcenter"]].copy()
            wc = wc[wc["Workcenter"].notna()]
            wc["Workcenter"] = wc["Workcenter"].astype(str).str.strip()
            wc = wc[(wc["Workcenter"] != "") & (wc["Workcenter"].str.lower() != "nan")]
            workcenter_frames.append(wc)

    if workcenter_frames:
        wc_all = pd.concat(workcenter_frames, ignore_index=True)
        wc_summary = (
            wc_all.groupby("Part Number")["Workcenter"]
            .agg(lambda x: ", ".join(sorted(set(x))))
            .reset_index()
            .rename(columns={"Workcenter": "Workcenter(s)"})
        )
        merged = merged.merge(wc_summary, on="Part Number", how="left")
        merged["Workcenter(s)"] = merged["Workcenter(s)"].fillna("—")
    else:
        merged["Workcenter(s)"] = "—"

    # Sort by Weighted Score descending
    merged = merged.sort_values("Weighted Score", ascending=False).reset_index(drop=True)

    # Reorder columns for display:
    #   calls_{ac}    = number of orders that called this part  ← used for scoring
    #   qty_{ac}      = total quantity requested                ← kept for reference
    #   consumed_{ac} = net actual consumption (issues − reversals), if available
    front_cols = ["Part Number","Material Description","UOM","Type","Workcenter(s)"]
    ac_call_cols = call_cols   # e.g. calls_PK-GLV, calls_PK-GLX, calls_PK-GLZ
    ac_qty_cols  = qty_cols    # e.g. qty_PK-GLV,   qty_PK-GLX,   qty_PK-GLZ
    end_cols = ["Total Calls","Total Qty"]
    if has_consumption:
        end_cols += consumed_cols + ["Total Actual Consumption"]
    end_cols += ["Total Occurrence","Occurrence %","Weighted Score",
                 "Min-Maxed?","Reorder Point","Max. level"]
    all_cols   = front_cols + ac_call_cols + ac_qty_cols + end_cols
    result = merged[[c for c in all_cols if c in merged.columns]].copy()

    # Smart number formatting: whole numbers shown without decimals
    numeric_display_cols = (
        call_cols + qty_cols + consumed_cols +
        ["Total Calls","Total Qty","Total Actual Consumption",
         "Occurrence %","Weighted Score","Reorder Point","Max. level"]
    )
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
        "not_found_in_db":       int((df["Min-Maxed?"] == "—").sum())     if mm_col_ok else 0,
        "top_score":             float(df["Weighted Score"].max()),
        "total_qty_all":         float(df[calls_col].sum()),
    }


# ── C) Alternate material recommendation ───────────────────────────────────
# The bundled data/alt_material_database.xlsx is a pre-processed long-format
# lookup built from GMF's Alternate Material master (Material, Leading Part,
# Alternate 1-6, One Way Alternate 1-5). It only keeps rows where an alternate
# actually exists, in the form:
#   base_part | base_description | alt_part | alt_kind
# alt_kind is one of: "Leading Part", "Alternate", "One-Way Alternate"

def load_alt_mat_db(fileobj) -> pd.DataFrame:
    """
    Load the bundled Alternate Material lookup table.
    Returns columns: base_part, base_description, alt_part, alt_kind
    """
    df = pd.read_excel(fileobj, dtype=str)
    df.columns = df.columns.str.strip()
    for c in ["base_part", "base_description", "alt_part", "alt_kind"]:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    return df.reset_index(drop=True)


def build_alternate_material_recommendations(
    workscope_table: pd.DataFrame,
    alt_lookup: pd.DataFrame,
    rop_db: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    For every part in the workscope material table, find any known
    alternates (in either direction) and check whether each alternate
    is already min-maxed in the Non-ROP database.

    A "swap opportunity" is a row where the requested part is NOT
    min-maxed but a known alternate already IS — meaning the warehouse
    could use the existing min-max plan instead of creating a new one.
    """
    if workscope_table is None or workscope_table.empty:
        return pd.DataFrame()
    if alt_lookup is None or alt_lookup.empty:
        return pd.DataFrame()

    wt = workscope_table.copy()
    wt["_key"] = wt["Part Number"].astype(str).str.strip().str.upper()

    alt = alt_lookup.copy()
    alt["_base_key"] = alt["base_part"].astype(str).str.strip().str.upper()
    alt["_alt_key"]  = alt["alt_part"].astype(str).str.strip().str.upper()

    ws_cols = [c for c in ["Part Number","Material Description","Type","Min-Maxed?",
                            "Weighted Score","Total Occurrence"] if c in wt.columns]

    # Forward: workscope part is the base part → alternate is alt_part
    fwd = wt.merge(alt, left_on="_key", right_on="_base_key", how="inner")
    if not fwd.empty:
        fwd["Alternate Part Number"] = fwd["alt_part"]
        fwd["Alternate Kind"]        = fwd["alt_kind"]

    # Reverse: workscope part appears as someone else's alternate →
    # that someone else's part number is also an interchangeable option
    rev = wt.merge(alt, left_on="_key", right_on="_alt_key", how="inner")
    if not rev.empty:
        rev["Alternate Part Number"] = rev["base_part"]
        rev["Alternate Kind"]        = rev["alt_kind"] + " (reverse)"

    combined = pd.concat([fwd, rev], ignore_index=True, sort=False)
    if combined.empty:
        return pd.DataFrame()

    combined["_alt_norm"] = combined["Alternate Part Number"].astype(str).str.strip().str.upper()
    # Drop cases where the "alternate" is just the same part as requested
    combined = combined[combined["_alt_norm"] != combined["_key"]]
    if combined.empty:
        return pd.DataFrame()

    # ROP lookup for the alternate part
    rop_map = {}
    if rop_db is not None and not rop_db.empty and "Material" in rop_db.columns:
        rop_norm = rop_db.copy()
        rop_norm["_key"] = rop_norm["Material"].astype(str).str.strip().str.upper()
        rop_map = dict(zip(rop_norm["_key"], rop_norm["min_maxed"]))

    def _mm_label(key):
        val = rop_map.get(key)
        if val is True:  return "✅ Yes"
        if val is False: return "❌ No"
        return "— Unknown"

    combined["Alternate Min-Maxed?"] = combined["_alt_norm"].map(_mm_label)
    combined = combined.rename(columns={"Min-Maxed?": "Requested Min-Maxed?"})

    out_cols = [c.replace("Min-Maxed?", "Requested Min-Maxed?") if c == "Min-Maxed?" else c
                for c in ws_cols] + ["Alternate Part Number", "Alternate Kind", "Alternate Min-Maxed?"]
    out_cols = [c for c in out_cols if c in combined.columns]

    result = combined[out_cols].drop_duplicates(
        subset=["Part Number", "Alternate Part Number"]
    ).reset_index(drop=True)

    # Sort: swap opportunities first (requested=No, alternate=Yes), then by score
    result["_is_swap"] = (
        (result.get("Requested Min-Maxed?", "") == "❌ No") &
        (result["Alternate Min-Maxed?"] == "✅ Yes")
    )
    sort_cols = ["_is_swap"]
    sort_asc  = [False]
    if "Weighted Score" in result.columns:
        sort_cols.append("Weighted Score")
        sort_asc.append(False)
    result = result.sort_values(sort_cols, ascending=sort_asc).drop(columns=["_is_swap"])

    return result.reset_index(drop=True)


def alt_mat_stats(df: pd.DataFrame) -> dict:
    """Summary KPIs for the alternate material recommendation table."""
    if df.empty:
        return {}
    is_swap = (
        (df.get("Requested Min-Maxed?", "") == "❌ No") &
        (df["Alternate Min-Maxed?"] == "✅ Yes")
    )
    return {
        "total_relationships": len(df),
        "parts_with_alternates": df["Part Number"].nunique(),
        "swap_opportunities":  int(is_swap.sum()),
        "swap_parts":          int(df.loc[is_swap, "Part Number"].nunique()),
    }
