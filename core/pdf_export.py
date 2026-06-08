"""
core/pdf_export.py
------------------
Generates a clean summary PDF report from pipeline results.
Uses WeasyPrint to convert an HTML template to PDF.

Report sections:
  1. Cover: run metadata (date, workscope, AC type, aircraft)
  2. Executive summary KPIs
  3. Fleet-wide defects table (scored, with per-AC counts)
  4. Common defects table (2/3 aircraft)
  5. Manhour impact table (top 15)
  6. Pre-provision materials list (if available)
"""

from datetime import datetime
from typing import Optional
import base64
import pandas as pd


# ── Colour constants ───────────────────────────────────────────────────────
TIER_BG  = {"Fleet-wide": "#E1F5EE", "Common": "#E6F1FB", "Isolated": "#F1EFE8"}
TIER_FG  = {"Fleet-wide": "#0F6E56", "Common": "#185FA5", "Isolated": "#5F5E5A"}
AC_COLORS = ["#5DCAA5", "#85B7EB", "#FAC775", "#D85A30", "#534AB7", "#D4537E"]


def _tier_badge(tier: str) -> str:
    bg = TIER_BG.get(tier, "#eee")
    fg = TIER_FG.get(tier, "#333")
    return (
        f"<span style='background:{bg};color:{fg};"
        f"padding:2px 8px;border-radius:8px;font-size:11px;"
        f"font-weight:600'>{tier}</span>"
    )


def _score_bar(score: float, max_score: float = 1.0) -> str:
    pct   = min(100, score / max_score * 100)
    color = "#0F6E56" if score >= 0.6 else "#185FA5" if score >= 0.5 else "#BA7517"
    return (
        f"<div style='display:flex;align-items:center;gap:6px'>"
        f"<div style='width:80px;background:#eee;border-radius:3px;height:8px'>"
        f"<div style='width:{pct:.0f}%;background:{color};height:100%;border-radius:3px'></div>"
        f"</div>"
        f"<span style='font-size:11px;color:{color};font-weight:600'>{score:.3f}</span>"
        f"</div>"
    )


def _scores_table(scores: pd.DataFrame, tier_filter: Optional[str] = None,
                  top_n: int = 30) -> str:
    """Render a defect scores table as HTML."""
    df = scores.copy()
    if tier_filter:
        df = df[df["tier"] == tier_filter]
    df = df.head(top_n)

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
          <th>Damage type</th>
          <th>Tier</th>
          <th style='text-align:center'>NRCs</th>
          {ac_headers}
          <th style='text-align:right'>Avg hrs</th>
          <th style='min-width:120px'>Score</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def _materials_table(top_parts: pd.DataFrame, top_n: int = 30) -> str:
    """Render the pre-provision parts list as HTML."""
    df = top_parts.head(top_n)
    rows = ""
    for i, row in df.iterrows():
        rows += f"""
        <tr>
          <td style='font-family:monospace;font-size:11px'>{row.get('Part Number','')}</td>
          <td>{row.get('Material Description','')}</td>
          <td style='text-align:center'>{row.get('UOM','')}</td>
          <td style='text-align:center'>{row.get('Type','')}</td>
          <td style='text-align:center;font-weight:600'>{int(row.get('ac_count',0))}</td>
          <td style='text-align:center'>{row.get('total_qty',0):.0f}</td>
          <td style='font-size:11px;color:#666'>{row.get('defects','')[:60]}</td>
        </tr>"""

    return f"""
    <table class="data-table">
      <thead>
        <tr>
          <th>Part Number</th>
          <th>Description</th>
          <th>UOM</th>
          <th>Type</th>
          <th style='text-align:center'># AC</th>
          <th style='text-align:center'>Total Qty</th>
          <th>Defect types</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>"""


def build_html_report(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    top_parts: pd.DataFrame,
    workscope: str,
    ac_type: str,
    notes: str,
) -> str:
    """Build the full HTML string for the PDF report."""

    projects    = sorted(df["project"].unique().tolist())
    n_nrcs      = len(df)
    n_clusters  = int(df[df["cluster_id"] != -1]["cluster_id"].nunique())
    n_fleet     = int((scores["tier"] == "Fleet-wide").sum()) if not scores.empty else 0
    n_common    = int((scores["tier"] == "Common").sum())     if not scores.empty else 0
    n_noise     = int((df["cluster_id"] == -1).sum())
    run_date    = datetime.now().strftime("%d %B %Y, %H:%M")

    # AC registration pills
    ac_pills = "".join(
        f"<span style='background:{AC_COLORS[i % len(AC_COLORS)]};color:#333;"
        f"padding:3px 12px;border-radius:8px;font-size:12px;font-weight:600;"
        f"margin-right:6px'>{p}</span>"
        for i, p in enumerate(projects)
    )

    fleet_table  = _scores_table(scores, tier_filter="Fleet-wide", top_n=30)
    common_table = _scores_table(scores, tier_filter="Common",     top_n=20)
    mhrs_table   = _scores_table(
        scores.nlargest(15, "avg_mhrs"), tier_filter=None, top_n=15
    )
    mat_section = ""
    if not top_parts.empty:
        mat_section = f"""
        <div class="section">
          <h2>Pre-Provision Material Recommendations</h2>
          <p class="subtitle">
            Parts appearing in multiple aircraft for the same defect —
            candidates for inclusion in future workscope material kits.
          </p>
          {_materials_table(top_parts, top_n=30)}
        </div>"""

    notes_section = ""
    if notes.strip():
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
      margin: 15mm 15mm 20mm 15mm;
      @bottom-center {{
        content: "GMF AeroAsia · NRC Clustering Report · " counter(page) " / " counter(pages);
        font-size: 9px; color: #999; font-family: Arial, sans-serif;
      }}
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: Arial, sans-serif;
      font-size: 12px;
      color: #222;
      line-height: 1.4;
    }}

    /* Cover */
    .cover {{
      height: 100vh;
      display: flex;
      flex-direction: column;
      justify-content: center;
      padding: 40px;
      background: linear-gradient(135deg, #1a3a6b 0%, #2c5fa8 100%);
      color: white;
      page-break-after: always;
    }}
    .cover .logo {{ font-size: 48px; margin-bottom: 10px; }}
    .cover h1 {{ font-size: 32px; font-weight: 700; margin-bottom: 8px; }}
    .cover .sub {{ font-size: 18px; opacity: 0.85; margin-bottom: 30px; }}
    .cover .meta-grid {{
      display: grid; grid-template-columns: 1fr 1fr; gap: 20px;
      background: rgba(255,255,255,0.12);
      border-radius: 8px; padding: 20px; margin-top: 20px;
    }}
    .cover .meta-item .label {{ font-size: 11px; opacity: 0.7; text-transform: uppercase; letter-spacing: 0.08em; }}
    .cover .meta-item .value {{ font-size: 15px; font-weight: 600; margin-top: 3px; }}

    /* KPI strip */
    .kpi-strip {{
      display: grid;
      grid-template-columns: repeat(6, 1fr);
      gap: 12px;
      margin: 20px 0;
    }}
    .kpi-card {{
      background: #f5f7ff;
      border-radius: 8px;
      padding: 12px 14px;
      border: 0.5px solid #dde3f5;
    }}
    .kpi-card .val {{
      font-size: 24px; font-weight: 700;
      color: #1a3a6b;
    }}
    .kpi-card .lbl {{
      font-size: 10px; color: #666;
      text-transform: uppercase; letter-spacing: 0.05em;
      margin-top: 3px;
    }}

    /* Sections */
    .section {{ margin-bottom: 30px; page-break-inside: avoid; }}
    .section h2 {{
      font-size: 16px; font-weight: 700; color: #1a3a6b;
      border-bottom: 2px solid #c5d4f0;
      padding-bottom: 6px; margin-bottom: 10px;
    }}
    .subtitle {{ font-size: 11px; color: #666; margin-bottom: 10px; }}

    /* Tables */
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 11px;
    }}
    .data-table thead tr {{
      background: #1a3a6b;
      color: white;
    }}
    .data-table thead th {{
      padding: 7px 8px;
      text-align: left;
      font-weight: 600;
      font-size: 10px;
      letter-spacing: 0.04em;
      white-space: nowrap;
    }}
    .data-table tbody tr:nth-child(even) {{ background: #f8f9ff; }}
    .data-table tbody tr:hover {{ background: #eef2ff; }}
    .data-table td {{ padding: 6px 8px; border-bottom: 0.5px solid #e8ecf5; }}

    /* Formula box */
    .formula-box {{
      background: #f0f4ff;
      border-left: 4px solid #2c5fa8;
      border-radius: 4px;
      padding: 12px 16px;
      margin-bottom: 16px;
      font-size: 12px;
      color: #333;
    }}
    .formula-box strong {{ color: #1a3a6b; }}

    /* Page break */
    .page-break {{ page-break-before: always; }}
  </style>
</head>
<body>

  <!-- COVER PAGE -->
  <div class="cover">
    <div class="logo">✈</div>
    <h1>NRC Findings Clustering Report</h1>
    <div class="sub">{workscope} · {ac_type}</div>
    <div style="margin-top:16px">{ac_pills}</div>
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
        <div class="value">{len(projects)} AC · {n_nrcs:,} NRCs</div>
      </div>
    </div>
  </div>

  <!-- EXECUTIVE SUMMARY -->
  <div class="section">
    <h2>Executive Summary</h2>
    <div class="kpi-strip">
      <div class="kpi-card"><div class="val">{n_nrcs:,}</div><div class="lbl">Total NRCs</div></div>
      <div class="kpi-card"><div class="val">{len(projects)}</div><div class="lbl">Aircraft</div></div>
      <div class="kpi-card"><div class="val">{n_clusters}</div><div class="lbl">Clusters</div></div>
      <div class="kpi-card" style="background:#E1F5EE;border-color:#9FE1CB">
        <div class="val" style="color:#0F6E56">{n_fleet}</div>
        <div class="lbl">Fleet-wide defects</div>
      </div>
      <div class="kpi-card" style="background:#E6F1FB;border-color:#B5D4F4">
        <div class="val" style="color:#185FA5">{n_common}</div>
        <div class="lbl">Common defects (2/3)</div>
      </div>
      <div class="kpi-card">
        <div class="val">{n_noise}</div>
        <div class="lbl">One-off NRCs</div>
      </div>
    </div>

    <div class="formula-box">
      <strong>Weighted scoring formula:</strong>
      &nbsp; Score = <strong>50%</strong> Presence (across aircraft)
      + <strong>30%</strong> Frequency (NRC rate per aircraft)
      + <strong>20%</strong> Manhour cost (avg actual hrs) &nbsp;→&nbsp; range 0.0 – 1.0
    </div>
  </div>

  <!-- FLEET-WIDE DEFECTS -->
  <div class="section page-break">
    <h2>Fleet-Wide Defects — Found in All {len(projects)} Aircraft</h2>
    <p class="subtitle">
      These defects appeared in every aircraft in this analysis.
      Strongest candidates for inclusion in the planned workscope.
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
    <h2>Top Defects by Manhour Impact</h2>
    <p class="subtitle">
      Ranked by average actual manhours. High-impact defects regardless of frequency.
    </p>
    {mhrs_table}
  </div>

  <!-- MATERIALS -->
  {mat_section}

  <!-- NOTES -->
  {notes_section}

</body>
</html>"""


def generate_pdf(
    df: pd.DataFrame,
    scores: pd.DataFrame,
    top_parts: pd.DataFrame,
    workscope: str,
    ac_type: str,
    notes: str,
) -> Optional[bytes]:
    """
    Render the HTML report to PDF bytes using WeasyPrint.
    Returns None if WeasyPrint is not available.
    """
    try:
        from weasyprint import HTML
        html_str = build_html_report(df, scores, top_parts, workscope, ac_type, notes)
        pdf_bytes = HTML(string=html_str).write_pdf()
        return pdf_bytes
    except ImportError:
        return None
    except Exception as e:
        import streamlit as st
        st.warning(f"PDF generation failed: {e}")
        return None
