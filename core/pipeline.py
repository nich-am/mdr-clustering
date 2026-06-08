"""
core/pipeline.py
----------------
Vectorization → UMAP reduction → HDBSCAN clustering → weighted EDA scoring.
All functions are pure (no Streamlit imports) so they can be tested standalone.
"""

import re
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import hdbscan
import umap

from core.preprocess import (
    clean_text, get_damage, get_location, get_sub_component, build_cluster_label
)

warnings.filterwarnings("ignore")


# ── 1. Load & validate uploaded files ─────────────────────────────────────
def load_files(uploaded_files: dict) -> pd.DataFrame:
    """
    uploaded_files: {ac_reg_label: BytesIO}
    Returns combined DataFrame with 'project' and 'ac_reg' columns.
    """
    dfs = []
    for ac_label, fileobj in uploaded_files.items():
        df = pd.read_excel(fileobj)
        df["project"] = ac_label
        df["ac_reg"]  = ac_label
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.dropna(subset=["Description"])
    combined = combined[combined["Description"].str.strip() != ""]
    return combined


# ── 2. Preprocess ─────────────────────────────────────────────────────────
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["preprocessed"] = df["Description"].apply(clean_text)
    df["damage_type"]  = df["Description"].apply(get_damage)
    df["location"]     = df["Description"].apply(get_location)
    df["sub_component"]  = df["Description"].apply(get_sub_component)
    df["defect_key"]     = df["location"] + " | " + df["damage_type"]
    return df


# ── 3. Vectorise (TF-IDF) ─────────────────────────────────────────────────
def vectorize(texts: list):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.85,
        sublinear_tf=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    X = vectorizer.fit_transform(texts)
    X = normalize(X)
    return X, vectorizer


# ── 4. UMAP reduction ─────────────────────────────────────────────────────
def reduce_dimensions(X_dense, n_components=15, n_neighbors=12):
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=0.0,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(X_dense)


def reduce_2d(X_dense, n_neighbors=12):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    return reducer.fit_transform(X_dense)


# ── 5. HDBSCAN clustering ─────────────────────────────────────────────────
def cluster(X_reduced, min_cluster_size=5):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=2,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    labels = clusterer.fit_predict(X_reduced)
    return labels


# ── 6. Label clusters ─────────────────────────────────────────────────────
def label_clusters(df: pd.DataFrame, vectorizer, X_tfidf) -> dict:
    cluster_labels = {}
    raw_labels = {}

    for cid in sorted(df["cluster_id"].unique()):
        if cid == -1:
            cluster_labels[-1] = "Unclustered / one-off"
            continue
        titles = df.loc[df["cluster_id"] == cid, "Description"].tolist()
        raw_labels[cid] = build_cluster_label(titles)

    # Disambiguate true duplicates with variant suffix
    label_count = Counter(raw_labels.values())
    used = Counter()
    for cid, base in raw_labels.items():
        if label_count[base] > 1:
            used[base] += 1
            cluster_labels[cid] = f"{base} (v{used[base]})"
        else:
            cluster_labels[cid] = base

    return cluster_labels


def merge_same_defect_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post-clustering merge: clusters that share the SAME (location, damage_type)
    are collapsed into a single cluster ID (the smallest one wins).

    This fixes over-clustering where HDBSCAN splits the same defect into
    multiple clusters due to minor phrasing differences (e.g. LH vs RH,
    SIDFLOOR vs SIDE FLOOR, UPPER vs LOWER).

    Sub-component is intentionally NOT used as a merge key — we want
    "wing screw broken" and "wing bonding broken" to stay separate.
    """
    df = df.copy()
    # Build (location, damage_type) → min cluster_id mapping
    valid = df[df["cluster_id"] != -1].copy()
    if valid.empty:
        return df

    merge_map = (
        valid.groupby(["location", "damage_type"])["cluster_id"]
        .min()
        .to_dict()
    )

    def remap(row):
        if row["cluster_id"] == -1:
            return -1
        return merge_map.get((row["location"], row["damage_type"]), row["cluster_id"])

    df["cluster_id"] = df.apply(remap, axis=1)
    return df


# ── 7. Weighted EDA scoring ───────────────────────────────────────────────
def compute_eda_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each location|damage combo, compute:
      - presence_score  (0.50): fraction of projects that have this defect
      - frequency_score (0.30): avg NRC rate per project (normalised)
      - manhour_score   (0.20): avg actual manhours (normalised)
    Returns sorted DataFrame with weighted score.
    """
    WEIGHTS = {"presence": 0.50, "frequency": 0.30, "manhour": 0.20}

    classified = df[
        (df["location"] != "unclassified") & (df["damage_type"] != "other")
    ].copy()

    projects     = sorted(classified["project"].unique())
    n_projects   = len(projects)
    project_sizes = classified.groupby("project").size().to_dict()

    rows = []
    for key, grp in classified.groupby("defect_key"):
        loc, dmg = key.split(" | ")
        present  = set(grp["project"].unique())
        presence = len(present) / n_projects

        counts = grp.groupby("project").size().to_dict()
        avg_freq = sum(counts.get(p, 0) / project_sizes[p] for p in projects) / n_projects

        avg_mhrs = grp["Act Mhrs"].mean() if "Act Mhrs" in grp.columns else 0.0

        per_proj = {f"count_{p}": counts.get(p, 0) for p in projects}

        rows.append({
            "defect_key":    key,
            "location":      loc,
            "damage_type":   dmg,
            "total_count":   len(grp),
            "projects_count":len(present),
            "presence_raw":  presence,
            "avg_freq_raw":  avg_freq,
            "avg_mhrs":      round(avg_mhrs, 2),
            **per_proj,
        })

    scores = pd.DataFrame(rows)
    if scores.empty:
        return scores

    scores["freq_norm"]  = scores["avg_freq_raw"] / scores["avg_freq_raw"].max()
    scores["mhrs_norm"]  = scores["avg_mhrs"] / scores["avg_mhrs"].max() if scores["avg_mhrs"].max() > 0 else 0

    scores["score"] = (
        WEIGHTS["presence"]  * scores["presence_raw"]  +
        WEIGHTS["frequency"] * scores["freq_norm"]      +
        WEIGHTS["manhour"]   * scores["mhrs_norm"]
    ).round(4)

    def tier(row):
        if row["projects_count"] == n_projects:   return "Fleet-wide"
        if row["projects_count"] >= n_projects-1: return "Common"
        return "Isolated"

    scores["tier"] = scores.apply(tier, axis=1)
    scores = scores.sort_values("score", ascending=False).reset_index(drop=True)
    return scores


# ── 8. Full pipeline (one call) ───────────────────────────────────────────
def run_pipeline(uploaded_files: dict, min_cluster_size: int = 5) -> dict:
    """
    Run the complete pipeline.
    Returns a dict with keys: df, scores, X_2d, projects
    """
    df = load_files(uploaded_files)
    df = preprocess(df)

    texts = df["preprocessed"].tolist()
    X_tfidf, vectorizer = vectorize(texts)

    X_dense   = X_tfidf.toarray()
    X_reduced = reduce_dimensions(X_dense, n_neighbors=min(12, len(df)-1))
    X_2d      = reduce_2d(X_dense,        n_neighbors=min(12, len(df)-1))

    labels = cluster(X_reduced, min_cluster_size=min_cluster_size)
    df["cluster_id"] = labels

    # Preprocess: extract location + damage before merge
    df = preprocess(df) if "location" not in df.columns else df

    # Post-merge: collapse clusters with same (location, damage_type) into one
    df = merge_same_defect_clusters(df)

    cluster_map = label_clusters(df, vectorizer, X_tfidf)
    df["cluster_label"] = df["cluster_id"].map(cluster_map)

    scores   = compute_eda_scores(df)
    projects = sorted(df["project"].unique())

    return {
        "df":       df,
        "scores":   scores,
        "X_2d":     X_2d,
        "projects": projects,
    }


# ── 9. Excel export ───────────────────────────────────────────────────────
def build_excel(df: pd.DataFrame, scores: pd.DataFrame) -> bytes:
    import io
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        # Sheet 1 – All NRCs with cluster
        cols = [c for c in ["project","ac_reg","Seq","Description","Skill Active",
                             "Pmhrs","Act Mhrs","cluster_id","cluster_label",
                             "location","damage_type"] if c in df.columns]
        df[cols].to_excel(writer, sheet_name="NRC_Clusters", index=False)

        # Sheet 2 – EDA scores
        scores.to_excel(writer, sheet_name="EDA_Scores", index=False)

        # Sheet 3 – Frequency matrix
        clustered = df[df["cluster_id"] != -1].copy()
        if not clustered.empty:
            freq = (
                clustered
                .groupby(["cluster_label", "project"])
                .size()
                .unstack(fill_value=0)
                .reset_index()
            )
            freq["Total"] = freq.drop(columns="cluster_label").sum(axis=1)
            freq.sort_values("Total", ascending=False).to_excel(
                writer, sheet_name="Frequency_Matrix", index=False
            )
    buf.seek(0)
    return buf.read()
