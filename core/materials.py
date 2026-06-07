"""
core/materials.py
-----------------
Loads Material Request (MRM) files, filters toggle='y',
then joins materials to NRC findings via Order number.

Key join logic:
  NRC file  : Order No  (int)
  MRM file  : Order     (int)
  → joined on str(Order No) == str(Order)
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
    "Vendor Name",
    "PO Net Value",
]


def load_mrm(fileobj) -> pd.DataFrame:
    """
    Read one MRM Excel file, keep only toggle='y' rows,
    and return a clean DataFrame.
    """
    df = pd.read_excel(fileobj)

    # Normalise column names (strip whitespace, lower for toggle)
    df.columns = df.columns.str.strip()

    # Filter toggle = 'y'  (column is lowercase 'toggle')
    toggle_col = next((c for c in df.columns if c.lower() == "toggle"), None)
    if toggle_col is None:
        raise ValueError("No 'toggle' column found in MRM file.")
    df = df[df[toggle_col].astype(str).str.strip().str.lower() == "y"].copy()

    # Keep only relevant columns that exist
    keep = [c for c in MRM_KEEP if c in df.columns]
    df = df[keep].copy()

    # Normalise Order to string for joining
    df["Order"] = df["Order"].astype(str).str.strip()

    return df.reset_index(drop=True)


def join_materials_to_nrc(nrc_df: pd.DataFrame, mrm_df: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join MRM materials onto NRC rows via Order No → Order.
    Returns a DataFrame with one row per (NRC, material) combination.
    NRCs with no material requests will have NaN in material columns.
    """
    nrc = nrc_df.copy()
    nrc["_order_key"] = nrc["Order No"].astype(str).str.strip()

    mrm = mrm_df.copy()
    mrm["_order_key"] = mrm["Order"].astype(str).str.strip()

    joined = nrc.merge(
        mrm.drop(columns=["Order"]),
        on="_order_key",
        how="left",
    ).drop(columns=["_order_key"])

    return joined


def build_material_summary(
    nrc_df: pd.DataFrame,
    mrm_dict: dict,          # {ac_reg: mrm_df}
    scores_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each scored defect combo (location | damage_type), look up all
    NRC Order Nos that belong to that combo, then pull all y-toggle
    materials linked to those orders across all AC.

    Returns a DataFrame with columns:
      defect_key, tier, score, ac_reg, Order No, Part Number,
      Material Description, Qty Req, UOM, Type,
      Material Fullfilment Status
    """
    rows = []

    for _, score_row in scores_df.iterrows():
        key  = score_row["defect_key"]
        loc  = score_row["location"]
        dmg  = score_row["damage_type"]
        tier = score_row["tier"]
        sc   = score_row["score"]

        # NRCs that match this defect combo
        matching_nrcs = nrc_df[
            (nrc_df["location"] == loc) & (nrc_df["damage_type"] == dmg)
        ][["project", "Order No", "Description"]].copy()
        matching_nrcs["Order No"] = matching_nrcs["Order No"].astype(str).str.strip()

        if matching_nrcs.empty:
            continue

        for ac_reg, mrm_df in mrm_dict.items():
            # Only consider NRCs for this AC
            ac_nrcs = matching_nrcs[matching_nrcs["project"] == ac_reg]
            if ac_nrcs.empty:
                continue

            order_nos = set(ac_nrcs["Order No"].unique())

            # Pull matching materials
            mats = mrm_df[mrm_df["Order"].astype(str).isin(order_nos)].copy()
            if mats.empty:
                continue

            for _, mat_row in mats.iterrows():
                rows.append({
                    "defect_key":          key,
                    "location":            loc,
                    "damage_type":         dmg,
                    "tier":                tier,
                    "score":               sc,
                    "ac_reg":              ac_reg,
                    "Order No":            mat_row["Order"],
                    "Part Number":         mat_row.get("Part Number", ""),
                    "Material Description":mat_row.get("Material Description", ""),
                    "Qty Req":             mat_row.get("Qty Req", 0),
                    "UOM":                 mat_row.get("UOM", ""),
                    "Type":                mat_row.get("Type", ""),
                    "Fulfillment Status":  mat_row.get("Material Fullfilment Status", ""),
                    "Workcenter":          mat_row.get("Workcenter", ""),
                })

    return pd.DataFrame(rows)


def summarise_by_defect(mat_detail: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot material detail into a per-defect summary:
    For each defect_key, list unique part numbers with:
    - how many AC it appeared in
    - total qty requested
    - part description
    """
    if mat_detail.empty:
        return pd.DataFrame()

    summary = (
        mat_detail
        .groupby(["defect_key", "tier", "score", "Part Number", "Material Description", "UOM", "Type"])
        .agg(
            ac_count=("ac_reg", "nunique"),
            total_qty=("Qty Req", "sum"),
            ac_list=("ac_reg", lambda x: ", ".join(sorted(x.unique()))),
        )
        .reset_index()
        .sort_values(["score", "ac_count", "total_qty"], ascending=[False, False, False])
    )

    return summary


def top_parts_across_fleet(mat_detail: pd.DataFrame, min_ac: int = 2) -> pd.DataFrame:
    """
    Parts that appear in at least `min_ac` aircraft for the same defect.
    These are the strongest candidates for pre-provisioning.
    """
    if mat_detail.empty:
        return pd.DataFrame()

    agg = (
        mat_detail
        .groupby(["Part Number", "Material Description", "UOM", "Type"])
        .agg(
            defect_count=("defect_key", "nunique"),
            ac_count=("ac_reg", "nunique"),
            total_qty=("Qty Req", "sum"),
            defects=("defect_key", lambda x: " · ".join(sorted(x.unique())[:3])),
            ac_list=("ac_reg", lambda x: ", ".join(sorted(x.unique()))),
            avg_score=("score", "mean"),
        )
        .reset_index()
    )

    agg = agg[agg["ac_count"] >= min_ac]
    agg = agg.sort_values(["ac_count", "avg_score", "total_qty"], ascending=[False, False, False])

    return agg.reset_index(drop=True)
