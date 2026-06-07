# MDR Findings Clustering App

Streamlit app for clustering and EDA of Maintenance Discrepancy and Rectification (MDR) card findings
across maintenance events of the same workscope and aircraft type.

---

## Project structure

```
nrc_app/
├── app.py                  ← main Streamlit app (run this)
├── requirements.txt        ← Python dependencies
├── core/
│   ├── preprocess.py       ← text cleaning, location/damage extraction
│   ├── pipeline.py         ← full ML pipeline (vectorize → cluster → score)
│   └── charts.py           ← all Plotly chart builders
└── README.md
```

---

## Local setup (run on your laptop)

### Step 1 — Install Python
Download Python 3.10 or 3.11 from https://python.org/downloads
Make sure to tick "Add Python to PATH" during install.

### Step 2 — Create a virtual environment
Open a terminal (Command Prompt or PowerShell on Windows):

```bash
# Navigate to the app folder
cd path/to/nrc_app

# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Mac / Linux:
source venv/bin/activate
```

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

This takes 2–5 minutes the first time.

### Step 4 — Run the app

```bash
streamlit run app.py
```

The app opens automatically in your browser at http://localhost:8501

---

## How to use

1. In the sidebar, set **how many aircraft** you are comparing.
2. For each aircraft, type the **AC registration** (e.g. PK-GLZ) and
   upload its NRC Excel file.
3. Adjust **min cluster size** if needed (default 5 works well).
4. Click **Run pipeline**.
5. Explore the 6 tabs:
   - **Ranked defects** — all defects sorted by weighted score
   - **Fleet-wide** — defects found in ALL aircraft (highest priority)
   - **Similarity map** — 2D scatter of NRC title similarity
   - **Manhour impact** — costliest defect types
   - **Score breakdown** — how each score is composed
   - **Data tables** — filterable full NRC list + score table
6. Click **Download full Excel report** to export.

---

## Excel file requirements

Your NRC Excel files must have at least these columns:

| Column       | Description                        |
|--------------|------------------------------------|
| Description  | NRC title text (required)          |
| Skill Active | Skill/trade code (optional)        |
| Act Mhrs     | Actual manhours (optional)         |
| Pmhrs        | Planned manhours (optional)        |
| Seq          | Sequence number (optional)         |

---

## Deploy online (free) with Streamlit Community Cloud

So your team can use it from a browser without installing anything:

### Step 1 — Push code to GitHub
1. Create a free account at https://github.com
2. Create a new repository (e.g. `mdr-clustering`)
3. Upload all files from the `mdr_app/` folder to the repo root

### Step 2 — Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with your GitHub account
3. Click **New app**
4. Select your repository, branch `main`, and main file `app.py`
5. Click **Deploy**

The app will be live at a URL like:
`https://your-username-mdr-clustering-app-xxxxx.streamlit.app`

Share that URL with your team — no installation needed.

---

## Tuning the pipeline

### Min cluster size
- **3–4**: more clusters, catches smaller patterns, may be noisier
- **5** (default): good balance
- **8–10**: fewer, tighter clusters, only clear patterns

### Adding new abbreviations
Edit `core/preprocess.py` → `ABBREV_MAP` dict.
Add entries like: `r"\bYOUR_ABBREV\b": "expanded form"`

### Adding new locations or damage types
Edit `core/preprocess.py`:
- New locations → add to `LOCATION_VOCAB` list (order matters — more specific first)
- New damage types → add to `DAMAGE_VOCAB` dict

---

## Weighted scoring formula

```
score = 0.50 × presence + 0.30 × frequency + 0.20 × manhour_cost
```

- **presence**: fraction of aircraft that have this defect (1/3, 2/3, or 3/3)
- **frequency**: average NRC rate per aircraft, normalised 0–1
- **manhour_cost**: average actual manhours, normalised 0–1

Tier labels:
- **Fleet-wide**: defect found in ALL aircraft
- **Common**: found in N-1 aircraft (missing from one)
- **Isolated**: found in only one aircraft

---

## Requirements

- Python 3.10 or 3.11
- See requirements.txt for package versions
