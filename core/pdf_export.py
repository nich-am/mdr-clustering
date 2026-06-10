"""
core/pdf_export.py
------------------
Generates a summary PDF report from pipeline results.
Uses WeasyPrint to convert an HTML template to PDF.

Report sections:
  1. Cover
  2. Executive Summary  — KPIs + finding/material headline stats
  3. Fleet-Wide Defects
  4. Common Defects
  5. Top Defects by Manhour Impact
  6. Workscope Material Recommendation  (order-call counts + min-max status)
  7. Notes
"""

from datetime import datetime, timezone, timedelta
from typing import Optional
import pandas as pd


# ── Constants ─────────────────────────────────────────────────────────────
TIER_BG  = {"Fleet-wide": "#E1F5EE", "Common": "#E6F1FB", "Isolated": "#F1EFE8"}
TIER_FG  = {"Fleet-wide": "#0F6E56", "Common": "#185FA5", "Isolated": "#5F5E5A"}
AC_COLORS = ["#5DCAA5", "#85B7EB", "#FAC775", "#D85A30", "#534AB7", "#D4537E"]
WIB = timezone(timedelta(hours=7))


def _now_wib() -> str:
    return datetime.now(tz=WIB).strftime("%d %B %Y, %H:%M WIB")


def _tier_badge(tier: str) -> str:
    bg = TIER_BG.get(tier, "#eee")
    fg = TIER_FG.get(tier, "#333")
    return (
        f"<span style='background:{bg};color:{fg};"
        f"padding:2px 8px;border-radius:8px;font-size:10px;"
        f"font-weight:600'>{tier}</span>"
    )


def _score_bar(score: float, max_score: float = 1.0) -> str:
    pct   = min(100, score / max_score * 100) if max_score else 0
    color = "#0F6E56" if score >= 0.6 else "#185FA5" if score >= 0.5 else "#BA7517"
    return (
        f"<div style='display:flex;align-items:center;gap:6px'>"
        f"<div style='width:70px;background:#eee;border-radius:3px;height:7px'>"
        f"<div style='width:{pct:.0f}%;background:{color};height:100%;border-radius:3px'></div>"
        f"</div>"
        f"<span style='font-size:10px;color:{color};font-weight:600'>{score:.3f}</span>"
        f"</div>"
    )


def _scores_table(scores: pd.DataFrame, tier_filter: Optional[str] = None,
                  top_n: int = 30) -> str:
    df = scores.copy()
    if tier_filter:
        df = df[df["tier"] == tier_filter]
    df = df.head(top_n)
    if df.empty:
        return "<p style='color:#888;font-size:11px'>No data.</p>"

    count_cols = [c for c in df.columns if c.startswith("count_")]
    ac_headers = "".join(
        f"<th style='text-align:center'>{c.replace('count_','')}</th>"
        for c in count_cols
    )
    rows = ""
    for _, row in df.iterrows():
        ac_cells = "".join(
            f"<td style='text-align:center;font-weight:500'>"
            f"{int(row[c]) if row[c] else '—'}</td>"
            for c in count_cols
        )
        rows += f"""
        <tr>
          <td style='font-weight:500'>{row['location']}</td>
          <td>{row.get('sub_component','') or ''}</td>
          <td>{row['damage_type']}</td>
          <td>{_tier_badge(row['tier'])}</td>
          <td style='text-align:center'>{int(row['total_count'])}</td>
          {ac_cells}
          <td style='text-align:right'>{row['avg_mhrs']:.1f}h</td>
          <td>{_score_bar(row['score'])}</td>
        </tr>"""

    return f"""
    <table class="data-table">
      <thead>
        <tr>
          <th>Location</th>
          <th>Sub-component</th>
          <th>Damage type</th>
          <th>Tier</th>
          <th style='text-align:center'>NRCs</th>
          {ac_headers}
          <th style='text-align:right'>Avg hrs</th>
          <th style='min-width:110px'>Score</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def _workscope_material_table(workscope_table: pd.DataFrame, top_n: int = 40) -> str:
    """Render the workscope-level material recommendation table."""
    df = workscope_table.head(top_n).copy()
    if df.empty:
        return "<p style='color:#888;font-size:11px'>No material data uploaded.</p>"

    call_cols = [c for c in df.columns if c.startswith("calls_")]
    qty_cols  = [c for c in df.columns if c.startswith("qty_")]

    ac_call_headers = "".join(
        f"<th style='text-align:center'>{c.replace('calls_','')} calls</th>"
        for c in call_cols
    )
    ac_qty_headers = "".join(
        f"<th style='text-align:center'>{c.replace('qty_','')} qty</th>"
        for c in qty_cols
    )

    def _mm_cell(val):
        if val == "✅ Yes":
            return "<td style='text-align:center;color:#0F6E56;font-weight:700'>✅ Yes</td>"
        if val == "❌ No":
            return "<td style='text-align:center;color:#C0392B;font-weight:700'>❌ No</td>"
        return "<td style='text-align:center;color:#999'>—</td>"

    rows = ""
    for _, row in df.iterrows():
        call_cells = "".join(
            f"<td style='text-align:center'>{int(row[c]) if pd.notna(row.get(c)) else 0}</td>"
            for c in call_cols
        )
        qty_cells = "".join(
            f"<td style='text-align:center'>{row.get(c, 0)}</td>"
            for c in qty_cols
        )
        score_val = row.get("Weighted Score", 0)
        score_style = "font-weight:700;color:#0F6E56" if score_val >= 10 else \
                      "font-weight:600;color:#185FA5" if score_val >= 5 else ""
        rows += f"""
        <tr>
          <td style='font-family:monospace;font-size:10px'>{row.get('Part Number','')}</td>
          <td>{row.get('Material Description','')}</td>
          <td style='text-align:center'>{row.get('UOM','')}</td>
          <td style='text-align:center'>{row.get('Type','')}</td>
          {call_cells}
          {qty_cells}
          <td style='text-align:center'>{int(row.get('Grand Total', 0))}</td>
          <td style='text-align:center'>{int(row.get('Total Occurrence', 0))}</td>
          <td style='text-align:center;{score_style}'>{score_val}</td>
          {_mm_cell(row.get('Min-Maxed?','—'))}
          <td style='text-align:right'>{row.get('Reorder Point', 0)}</td>
          <td style='text-align:right'>{row.get('Max. level', 0)}</td>
        </tr>"""

    return f"""
    <table class="data-table">
      <thead>
        <tr>
          <th>Part Number</th>
          <th>Description</th>
          <th>UOM</th>
          <th>Type</th>
          {ac_call_headers}
          {ac_qty_headers}
          <th style='text-align:center'>Grand Total<br>Calls</th>
          <th style='text-align:center'>AC<br>Occurrence</th>
          <th style='text-align:center'>Weighted<br>Score</th>
          <th style='text-align:center'>Min-Maxed?</th>
          <th style='text-align:right'>Reorder<br>Point</th>
          <th style='text-align:right'>Max.<br>Level</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def build_html_report(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    workscope_table: pd.DataFrame,
    workscope: str,
    ac_type: str,
    notes: str,
) -> str:
    projects   = sorted(df["project"].unique().tolist())
    n_nrcs     = len(df)
    n_clusters = int(df[df["cluster_id"] != -1]["cluster_id"].nunique())
    n_fleet    = int((scores["tier"] == "Fleet-wide").sum()) if not scores.empty else 0
    n_common   = int((scores["tier"] == "Common").sum())     if not scores.empty else 0
    n_isolated = int((scores["tier"] == "Isolated").sum())   if not scores.empty else 0
    n_noise    = int((df["cluster_id"] == -1).sum())
    run_date   = _now_wib()

    # Material stats
    has_mat          = not workscope_table.empty
    n_unique_parts   = len(workscope_table) if has_mat else 0
    n_fleet_parts    = int((workscope_table["Total Occurrence"] == len(projects)).sum()) if has_mat else 0
    n_not_minmax     = int((workscope_table.get("Min-Maxed?","") == "❌ No").sum()) if has_mat else 0
    n_yes_minmax     = int((workscope_table.get("Min-Maxed?","") == "✅ Yes").sum()) if has_mat else 0
    top_score        = float(workscope_table["Weighted Score"].max()) if has_mat else 0

    # Top fleet-wide defect for summary callout
    fleet_scores = scores[scores["tier"] == "Fleet-wide"] if not scores.empty else pd.DataFrame()
    top_defect_str = ""
    if not fleet_scores.empty:
        top = fleet_scores.iloc[0]
        top_defect_str = f"{top['location']} {top.get('sub_component','') or ''} {top['damage_type']}".strip()

    # Top material for summary callout
    top_mat_str = ""
    if has_mat:
        top_mat_row = workscope_table.iloc[0]
        top_mat_str = f"{top_mat_row.get('Material Description','')} (score {top_mat_row.get('Weighted Score',0)})"

    ac_pills = "".join(
        f"<span style='background:{AC_COLORS[i % len(AC_COLORS)]};color:#333;"
        f"padding:3px 12px;border-radius:8px;font-size:12px;font-weight:600;"
        f"margin-right:6px'>{p}</span>"
        for i, p in enumerate(projects)
    )

    fleet_table  = _scores_table(scores, tier_filter="Fleet-wide", top_n=30)
    common_table = _scores_table(scores, tier_filter="Common",     top_n=20)
    mhrs_table   = _scores_table(
        scores.nlargest(15, "avg_mhrs") if not scores.empty else scores,
        tier_filter=None, top_n=15
    )

    # Material section
    mat_section = ""
    if has_mat:
        n_not_mm_fleet = int(
            ((workscope_table["Total Occurrence"] == len(projects)) &
             (workscope_table.get("Min-Maxed?", pd.Series()) == "❌ No")).sum()
        )
        mat_section = f"""
        <div class="section page-break">
          <h2>Workscope Material Recommendation</h2>
          <p class="subtitle">
            All materials with toggle = Y across all aircraft, ranked by weighted score.
            Score = Grand Total Calls + (AC Occurrence × 2).
            Parts used in all {len(projects)} aircraft are the strongest pre-provision candidates.
          </p>
          <div class="kpi-strip" style="grid-template-columns:repeat(5,1fr);margin-bottom:16px">
            <div class="kpi-card">
              <div class="val">{n_unique_parts:,}</div>
              <div class="lbl">Unique parts</div>
            </div>
            <div class="kpi-card" style="background:#E1F5EE;border-color:#9FE1CB">
              <div class="val" style="color:#0F6E56">{n_fleet_parts}</div>
              <div class="lbl">Used in all {len(projects)} AC</div>
            </div>
            <div class="kpi-card" style="background:#FDECEA;border-color:#F5B7B1">
              <div class="val" style="color:#C0392B">{n_not_minmax}</div>
              <div class="lbl">Not min-maxed ❌</div>
            </div>
            <div class="kpi-card" style="background:#E1F5EE;border-color:#9FE1CB">
              <div class="val" style="color:#0F6E56">{n_yes_minmax}</div>
              <div class="lbl">Already min-maxed ✅</div>
            </div>
            <div class="kpi-card">
              <div class="val">{int(top_score)}</div>
              <div class="lbl">Highest score</div>
            </div>
          </div>
          {"" if n_not_mm_fleet == 0 else f"<div class='formula-box'>⚠️ <strong>{n_not_mm_fleet} part(s)</strong> are used in all {len(projects)} aircraft but are <strong>not yet min-maxed</strong> — priority candidates for stock review.</div>"}
          {_workscope_material_table(workscope_table, top_n=40)}
        </div>"""

    notes_section = ""
    if notes and notes.strip():
        notes_section = f"""
        <div class="section">
          <h2>Notes</h2>
          <p style='color:#444;line-height:1.6'>{notes}</p>
        </div>"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>NRC Clustering Report — {workscope}</title>
  <style>
    @page {{
      size: A4 landscape;
      margin: 12mm 12mm 18mm 12mm;
      @bottom-center {{
        content: "GMF AeroAsia · NRC Clustering Report · " counter(page) " / " counter(pages);
        font-size: 9px; color: #999; font-family: Arial, sans-serif;
      }}
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: Arial, sans-serif; font-size: 11px; color: #222; line-height: 1.4; }}

    .cover {{
      height: 100vh; display: flex; flex-direction: column;
      justify-content: center; padding: 40px;
      background: linear-gradient(135deg, #1a3a6b 0%, #2c5fa8 100%);
      color: white; page-break-after: always;
    }}
    .cover .logo {{ font-size: 48px; margin-bottom: 10px; }}
    .cover h1 {{ font-size: 30px; font-weight: 700; margin-bottom: 8px; }}
    .cover .sub {{ font-size: 17px; opacity: 0.85; margin-bottom: 24px; }}
    .cover .meta-grid {{
      display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px;
      background: rgba(255,255,255,0.12); border-radius: 8px;
      padding: 18px; margin-top: 16px;
    }}
    .cover .meta-item .label {{ font-size: 10px; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.08em; }}
    .cover .meta-item .value {{ font-size: 14px; font-weight: 600; margin-top: 3px; }}

    .kpi-strip {{
      display: grid; grid-template-columns: repeat(7, 1fr); gap: 10px; margin: 16px 0;
    }}
    .kpi-card {{
      background: #f5f7ff; border-radius: 8px; padding: 10px 12px;
      border: 0.5px solid #dde3f5;
    }}
    .kpi-card .val {{ font-size: 22px; font-weight: 700; color: #1a3a6b; }}
    .kpi-card .lbl {{ font-size: 9px; color: #666; text-transform: uppercase;
                      letter-spacing: 0.05em; margin-top: 3px; }}

    .section {{ margin-bottom: 24px; page-break-inside: avoid; }}
    .section h2 {{
      font-size: 15px; font-weight: 700; color: #1a3a6b;
      border-bottom: 2px solid #c5d4f0; padding-bottom: 5px; margin-bottom: 8px;
    }}
    .subtitle {{ font-size: 10px; color: #666; margin-bottom: 8px; }}

    .data-table {{ width: 100%; border-collapse: collapse; font-size: 10px; }}
    .data-table thead tr {{ background: #1a3a6b; color: white; }}
    .data-table thead th {{
      padding: 6px 7px; text-align: left; font-weight: 600;
      font-size: 9px; letter-spacing: 0.03em; white-space: nowrap;
    }}
    .data-table tbody tr:nth-child(even) {{ background: #f8f9ff; }}
    .data-table td {{ padding: 5px 7px; border-bottom: 0.5px solid #e8ecf5; }}

    .formula-box {{
      background: #f0f4ff; border-left: 4px solid #2c5fa8; border-radius: 4px;
      padding: 10px 14px; margin-bottom: 12px; font-size: 11px; color: #333;
    }}
    .formula-box strong {{ color: #1a3a6b; }}

    .callout-grid {{
      display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 12px 0;
    }}
    .callout {{
      border-radius: 6px; padding: 12px 14px; font-size: 11px;
    }}
    .callout .callout-title {{
      font-size: 9px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.06em; margin-bottom: 6px;
    }}
    .callout .callout-value {{ font-size: 13px; font-weight: 600; }}

    .page-break {{ page-break-before: always; }}
  </style>
</head>
<body>

  <!-- COVER PAGE -->
  <div class="cover">
    <div class="logo">✈</div>
    <h1>NRC Findings Clustering Report</h1>
    <div class="sub">{workscope} · {ac_type}</div>
    <div style="margin-top:12px">{ac_pills}</div>
    <div class="meta-grid">
      <div class="meta-item">
        <div class="label">Report date</div>
        <div class="value">{run_date}</div>
      </div>
      <div class="meta-item">
        <div class="label">Workscope</div>
        <div class="value">{workscope}</div>
      </div>
      <div class="meta-item">
        <div class="label">Aircraft type</div>
        <div class="value">{ac_type}</div>
      </div>
      <div class="meta-item">
        <div class="label">Aircraft analysed</div>
        <div class="value">{len(projects)} AC</div>
      </div>
      <div class="meta-item">
        <div class="label">Total NRCs</div>
        <div class="value">{n_nrcs:,}</div>
      </div>
      <div class="meta-item">
        <div class="label">Defect clusters</div>
        <div class="value">{n_clusters}</div>
      </div>
    </div>
  </div>

  <!-- EXECUTIVE SUMMARY -->
  <div class="section">
    <h2>Executive Summary</h2>

    <div class="kpi-strip">
      <div class="kpi-card">
        <div class="val">{n_nrcs:,}</div><div class="lbl">Total NRCs</div>
      </div>
      <div class="kpi-card">
        <div class="val">{len(projects)}</div><div class="lbl">Aircraft</div>
      </div>
      <div class="kpi-card">
        <div class="val">{n_clusters}</div><div class="lbl">Defect clusters</div>
      </div>
      <div class="kpi-card" style="background:#E1F5EE;border-color:#9FE1CB">
        <div class="val" style="color:#0F6E56">{n_fleet}</div>
        <div class="lbl">Fleet-wide defects</div>
      </div>
      <div class="kpi-card" style="background:#E6F1FB;border-color:#B5D4F4">
        <div class="val" style="color:#185FA5">{n_common}</div>
        <div class="lbl">Common defects</div>
      </div>
      <div class="kpi-card">
        <div class="val">{n_isolated}</div><div class="lbl">Isolated defects</div>
      </div>
      <div class="kpi-card">
        <div class="val">{n_noise}</div><div class="lbl">One-off NRCs</div>
      </div>
    </div>

    <div class="callout-grid">
      <div class="callout" style="background:#E1F5EE;border-left:4px solid #0F6E56">
        <div class="callout-title" style="color:#0F6E56">🔍 Top fleet-wide finding</div>
        <div class="callout-value">{top_defect_str or "—"}</div>
        <div style="font-size:10px;color:#555;margin-top:4px">
          Found in all {len(projects)} aircraft · {n_fleet} fleet-wide defects total
        </div>
      </div>
      <div class="callout" style="background:#E6F1FB;border-left:4px solid #185FA5">
        <div class="callout-title" style="color:#185FA5">🔩 Top material by workscope score</div>
        <div class="callout-value">{top_mat_str or "No material data uploaded"}</div>
        <div style="font-size:10px;color:#555;margin-top:4px">
          {f"{n_unique_parts:,} unique parts · {n_fleet_parts} used in all {len(projects)} AC · {n_not_minmax} not yet min-maxed" if has_mat else "Upload MRM files to enable material analysis"}
        </div>
      </div>
    </div>

    <div class="formula-box">
      <strong>Defect scoring:</strong>
      Score = <strong>50%</strong> Presence (across aircraft)
      + <strong>30%</strong> Frequency (NRC rate per AC)
      + <strong>20%</strong> Manhour cost → range 0.0 – 1.0
      &nbsp;&nbsp;|&nbsp;&nbsp;
      <strong>Material scoring:</strong>
      Weighted Score = Grand Total Calls + (AC Occurrence × 2)
    </div>
  </div>

  <!-- FLEET-WIDE DEFECTS -->
  <div class="section page-break">
    <h2>Fleet-Wide Defects — Found in All {len(projects)} Aircraft</h2>
    <p class="subtitle">
      Strongest candidates for inclusion in the planned workscope.
      Sorted by weighted score descending.
    </p>
    {fleet_table}
  </div>

  <!-- COMMON DEFECTS -->
  <div class="section page-break">
    <h2>Common Defects — Found in {len(projects)-1} of {len(projects)} Aircraft</h2>
    <p class="subtitle">
      Present in most but not all aircraft. Monitor for fleet-wide recurrence
      in future maintenance events.
    </p>
    {common_table}
  </div>

  <!-- MANHOUR IMPACT -->
  <div class="section page-break">
    <h2>Top 15 Defects by Manhour Impact</h2>
    <p class="subtitle">
      Ranked by average actual manhours regardless of frequency.
      High-cost defects warrant dedicated tooling and material pre-positioning.
    </p>
    {mhrs_table}
  </div>

  <!-- MATERIAL RECOMMENDATION -->
  {mat_section}

  <!-- NOTES -->
  {notes_section}

</body>
</html>"""


def generate_pdf(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    workscope_table: pd.DataFrame,
    workscope: str,
    ac_type: str,
    notes: str,
) -> Optional[bytes]:
    """Render the HTML report to PDF bytes using WeasyPrint."""
    try:
        from weasyprint import HTML
        html_str = build_html_report(
            df, scores, workscope_table, workscope, ac_type, notes
        )
        return HTML(string=html_str).write_pdf()
    except ImportError:
        return None
    except Exception as e:
        import streamlit as st
        st.warning(f"PDF generation failed: {e}")
        return None
