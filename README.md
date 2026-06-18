# NRC Findings Clustering & Material Analysis

A Streamlit app for clustering Non-Routine Card (NRC) findings across aircraft maintenance events of the same workscope and type — surfacing fleet-wide defect patterns, scoring them by commonality and repair cost, and generating material stocking recommendations for the warehouse team.

**Live app:** deployed on Streamlit Community Cloud, connected to Supabase for run history persistence.

---

## What it does

1. **Clusters NRC titles** using NLP (TF-IDF + UMAP + HDBSCAN) — grouping free-text findings that describe the same defect, even when written differently by different technicians
2. **Scores each defect** by how commonly it appears across aircraft (presence), how frequently it recurs (frequency), and how costly it is to fix (manhour impact)
3. **Tiers defects** into Fleet-wide / Common / Isolated based on how many aircraft they appear on
4. **Links material requests** (MRM files) to each defect, showing which parts were called and in what quantity per aircraft
5. **Generates a workscope material table** — aggregating all Y-toggle parts across the fleet with a weighted score to identify pre-provision candidates
6. **Checks the Non-ROP database** to flag which parts already have a min-max stocking plan and which don't
7. **Exports** a full Excel report and PDF summary
8. **Saves run history** to Supabase so past analyses can be reviewed and compared

---

## Project structure

```
mdr-clustering/
├── app.py                    ← main Streamlit app
├── requirements.txt          ← Python dependencies
├── packages.txt              ← system packages (for Streamlit Cloud)
├── core/
│   ├── pipeline.py           ← NLP pipeline: vectorise → cluster → score
│   ├── preprocess.py         ← text cleaning, location/damage extraction
│   ├── materials.py          ← MRM loading, workscope aggregation, ROP join
│   ├── storage.py            ← Supabase read/write for run history
│   ├── charts.py             ← Plotly chart builders
│   └── pdf_export.py         ← WeasyPrint PDF generation
├── data/
│   └── rop_database.xlsx     ← Non-ROP / min-max database (bundled)
└── .streamlit/
    └── config.toml           ← theme config (dark mode)
```

---

## Analysis tabs

### New Analysis

| Tab | What it shows |
|-----|---------------|
| 📊 Common Defects | All defects ranked by weighted score, tier filter, donut chart |
| 🌍 Found on Every Aircraft | Fleet-wide defects only, expandable cards + ranked bar chart |
| 🗺️ Similarity Map | 2D UMAP scatter — each dot is one NRC, coloured by cluster |
| 🔥 Repair Time Impact | Top 25 by avg manhours, bubble chart (count vs hours) |
| 📈 How Scoring Works | Score breakdown with formula explanation |
| 📋 Full Data Table | Filterable full NRC list + defect score table |
| 🔩 Parts Needed per Defect | Per-defect material pivot (requires MRM upload) |
| 📦 All Materials Used | Workscope-level aggregated material table with filters |
| 🎯 Min-Max Recommendation | Fleet-wide parts not yet min-maxed — warehouse priority list |

### Run History

Same tabs as New Analysis (minus Similarity Map), loaded from Supabase for any saved run.

---

## Input files

### NRC Excel file (required — one per aircraft)

| Column | Description |
|--------|-------------|
| `Description` | NRC title text — the main input for clustering |
| `Order No` | Links NRCs to MRM material requests |
| `Act Mhrs` | Actual manhours for repair cost scoring |
| `Pmhrs` | Planned manhours (fallback if Act Mhrs missing) |
| `Skill Active` | Skill/trade code |

### MRM file (optional — one per aircraft)

Material request file. Only rows with `Toggle = Y` are used.

| Column | Description |
|--------|-------------|
| `Order` | Links to NRC `Order No` |
| `Part Number` | Format `PN:VendorCode` (e.g. `ASG33:36131`) — matched against Non-ROP DB |
| `Material Description` | Part description |
| `Qty Req` | Quantity requested |
| `UOM` | Unit of measure |
| `Type` | Material type (EXP, ROT, etc.) |

### Non-ROP / Min-Max database (optional — loaded from `data/rop_database.xlsx`)

| Column | Description |
|--------|-------------|
| `Material` (col C) | `PN:VendorCode` — join key matched against MRM `Part Number` |
| `MPN` (col B) | Bare part number |
| `ROP` | `Yes` = already min-maxed, `No` = not min-maxed |
| `Reorder Point` | Current reorder point |
| `Max. level` | Current max stock level |
| `ObjectType` | Aircraft type filter (e.g. `737-800`) |

---

## Weighted scoring formula

```
score = 0.50 × presence + 0.30 × frequency + 0.20 × manhour_cost
```

| Component | Weight | Description |
|-----------|--------|-------------|
| Presence | 50% | Fraction of aircraft that have this defect (1/3, 2/3, 3/3) |
| Frequency | 30% | Avg NRC count per aircraft, normalised 0–1 |
| Manhour cost | 20% | Avg actual manhours, normalised 0–1 |

**Tier labels:**
- **Fleet-wide** — defect found on every aircraft in the batch
- **Common** — found on all but one aircraft
- **Isolated** — found on only one aircraft

---

## Workscope material weighted score

```
Weighted Score = Total Calls + (Total Occurrence × 2)
```

- **Total Calls** — how many order-lines called this part across all aircraft
- **Total Occurrence** — how many aircraft called it at least once
- The `× 2` multiplier means a part called once on every aircraft scores higher than a part called many times on only one — intentionally surfacing fleet-wide demand

---

## Supabase schema

Run this SQL once in your Supabase SQL editor to create all required tables:

```sql
-- Stores each pipeline run
create table runs (
  id            uuid default gen_random_uuid() primary key,
  created_at    timestamptz default now(),
  workscope     text,
  ac_type       text,
  aircraft      text[],
  total_nrcs    int,
  n_clusters    int,
  n_fleet_wide  int,
  notes         text,
  excel_url     text,
  pdf_url       text
);
alter table runs disable row level security;

-- Defect scores (one row per defect combo per run)
create table defect_scores (
  id             uuid default gen_random_uuid() primary key,
  run_id         uuid references runs(id) on delete cascade,
  created_at     timestamptz default now(),
  location       text,
  damage_type    text,
  tier           text,
  score          numeric,
  total_count    int,
  projects_count int,
  avg_mhrs       numeric
);
alter table defect_scores disable row level security;

-- Material recommendations (pre-provision candidates)
create table material_recommendations (
  id                   uuid default gen_random_uuid() primary key,
  run_id               uuid references runs(id) on delete cascade,
  part_number          text,
  material_description text,
  uom                  text,
  material_type        text,
  ac_count             int,
  total_qty            numeric,
  avg_score            numeric,
  defects              text,
  ac_list              text
);
alter table material_recommendations disable row level security;

-- Workscope-level material table
create table workscope_materials (
  id                   uuid default gen_random_uuid() primary key,
  run_id               uuid references runs(id) on delete cascade,
  part_number          text,
  material_description text,
  uom                  text,
  material_type        text,
  total_occurrence     int,
  grand_total          numeric,
  total_qty            numeric default 0,
  weighted_score       numeric,
  min_maxed            text,
  reorder_point        numeric,
  max_level            numeric
);
alter table workscope_materials disable row level security;
```

Also create the storage bucket:

1. Go to **Storage** → **New bucket**
2. Name: `nrc-reports`
3. Toggle **Public bucket** ON

Then allow uploads with this policy:
```sql
create policy "allow all uploads"
on storage.objects
for all
using (bucket_id = 'nrc-reports')
with check (bucket_id = 'nrc-reports');
```

---

## Streamlit secrets

In Streamlit Cloud → your app → Settings → Secrets:

```toml
[supabase]
url = "https://xxxxxxxxxxxx.supabase.co"
key = "eyJhbGci..."
```

No trailing slash on the URL. Use the **anon/public** key, not the service_role key.

---

## Local setup

### Step 1 — Install Python 3.10 or 3.11
Download from https://python.org/downloads. Tick **Add Python to PATH** during install.

### Step 2 — Create a virtual environment

```bash
cd path/to/mdr-clustering
python -m venv venv

# Activate — Windows:
venv\Scripts\activate
# Activate — Mac / Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** WeasyPrint requires system libraries. On Ubuntu/Debian: `sudo apt install libglib2.0-dev libpango-1.0-0 libcairo2`. On Mac: `brew install pango cairo`. On Windows: use WSL or skip PDF export.

### Step 4 — Add Streamlit secrets locally

Create `.streamlit/secrets.toml`:

```toml
[supabase]
url = "https://xxxxxxxxxxxx.supabase.co"
key = "eyJhbGci..."
```

### Step 5 — Run the app

```bash
streamlit run app.py
```

Opens at http://localhost:8501

---

## Deploying on Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to https://share.streamlit.io → **New app**
3. Select your repo, branch `main`, file `app.py`
4. Add secrets (see above)
5. Click **Deploy**

The `packages.txt` file handles system dependency installation automatically on Streamlit Cloud.

---

## Tuning the pipeline

### Min cluster size (sidebar slider)
- **3–4**: more clusters, catches smaller patterns, may be noisier
- **5** (default): good balance for most workscopes
- **8–10**: fewer, tighter clusters — only clear repeating patterns

### Adding abbreviations
Edit `core/preprocess.py` → `ABBREV_MAP`:
```python
r"\bYOUR_ABBREV\b": "expanded form"
```

### Adding locations or damage types
Edit `core/preprocess.py`:
- New locations → add to `LOCATION_VOCAB` (more specific terms first)
- New damage types → add to `DAMAGE_VOCAB`
