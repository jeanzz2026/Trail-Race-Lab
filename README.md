# ITRA Race Results Analyzer 🏃‍♂️

A web application for extracting and analyzing race results from the International Trail Running Association (ITRA) website. The application provides both a Flask-based web interface and a Streamlit dashboard for comprehensive race data analysis.

## Features

### Web Interface (Flask)
- Extract race results from ITRA race URLs
- Display runner information including:
  - Position
  - Name
  - Finish Time
  - Age
  - Gender
  - Nationality
  - ITRA Profile Link
- Clean and responsive UI with topographic background
- Loading states and error handling

### Analytics Dashboard (Streamlit)
- Interactive data visualization
- GPX-based race and nutrition planning:
  - course distance, smoothed elevation, gain and loss
  - checkpoint-to-checkpoint pace, arrival and departure targets
  - target-finish-time mode or reference-performance estimate
  - per-leg carbohydrate, fluid, flask, gel and solid-food plan
  - downloadable race-plan and nutrition-plan CSV files
- Performance analysis charts:
  - Race Time vs ITRA Performance Index
  - Finish Time vs Position scatter plot
  - Age Distribution histogram
- Multiple filtering options:
  - Gender
  - Nationality
  - Age Groups
- Downloadable results table

## Setup Instructions

### One-command local development setup

Windows PowerShell:

```powershell
.\scripts\setup_local.ps1
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

macOS/Linux:

```bash
./scripts/setup_local.sh
.venv/bin/python -m streamlit run streamlit_app.py
```

The repository includes the model, public modern labels, validation outputs,
and the aggregated TRAP race features used by training. See
[`THIRD_PARTY_DATA.md`](THIRD_PARTY_DATA.md) for provenance and the optional
script that downloads the participant-level TRAP source data from its original
repository.

### Manual setup

1. Clone the repository:
```bash
git clone https://github.com/jeanzz2026/Trail-Race-Lab.git
cd Trail-Race-Lab
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Run the Flask application:
```bash
python main.py
```
The Flask app will be available at `http://localhost:5000`

4. Run the Streamlit dashboard:
```bash
streamlit run streamlit_app.py
```
The Streamlit dashboard will be available at `http://localhost:8501`

## Example Usage

1. Visit either the Flask web interface or Streamlit dashboard
2. Enter an ITRA race results URL, for example:
   `https://itra.run/Races/RaceResults/70K/2024/94006`
3. Click "Get Results" or "Analyze Results" to fetch and display the data
4. For the Streamlit dashboard:
   - Use the filters to analyze specific runner groups
   - Interact with the charts to explore patterns
   - Download the results table for further analysis

## Technologies Used
- Flask (Web Framework)
- Streamlit (Analytics Dashboard)
- Beautiful Soup 4 (Web Scraping)
- Pandas (Data Processing)
- Plotly (Data Visualization)
- Bootstrap (UI Components)

## Offline Race Score prototype

The repository also contains an experimental, non-official ITRA Race Score
estimator. It uses the public TRAP corpus as an unlabelled course/time prior and
modern public score/time pairs for calibration.

Train and validate it from the checked-in aggregate features with:

```powershell
.\.venv\Scripts\python.exe scripts\train_offline_model.py
```

To rebuild the aggregate from the original participant-level TRAP files, first
run `scripts/fetch_trap_data.ps1` (or `.sh`), then pass
`--trap-dir vendor/TRAP-data/ITRA`. Raw upstream data is intentionally fetched
from its original repository because that repository does not declare an
explicit redistribution license; details are in `THIRD_PARTY_DATA.md`.

Estimate from course data only (low confidence):

```powershell
.\.venv\Scripts\python.exe scripts\predict_race_score.py `
  --distance 61 --elevation 3400 --time 07:30:00
```

Estimate from one score/time pair from the same race (high confidence):

```powershell
.\.venv\Scripts\python.exe scripts\predict_race_score.py `
  --distance 61 --elevation 3400 --time 07:30:00 `
  --anchor-time 05:35:13 --anchor-score 853
```

Generated metrics and limitations are documented in
`model_artifacts/REPORT.md`. The estimator does not reproduce or claim to be an
official ITRA calculation.

The same estimator is available in the Streamlit UI without an ITRA account:

- Generating a GPX race plan automatically displays a course-only estimated
  Race Score and its 80% interval.
- After scraping a public race-results page, the app follows its public
  **RACE DETAILS** link, reads distance/ascent automatically, and immediately
  estimates every finisher. Manual fields appear only when ITRA does not
  return complete course metadata. One known same-race score/time pair can
  optionally provide much stronger calibration. The anchor editor accepts any
  number of rows; multiple anchors are combined with a robust median of
  `score × finish time`, obvious outliers are excluded when enough anchors are
  available, and uncertainty reflects both anchor count and disagreement.
- The results table and the separate **Race Score** tab display the estimated
  values. Personal ITRA Index values remain a separate field.

## Race and nutrition planner

Open the Streamlit dashboard, scroll to **比赛计划与补给计划（离线原型）**,
then:

1. Upload a GPX file that contains elevation values by dropping it into the page
   or selecting it with **Browse files**.
2. Choose a target elapsed time, or enter a reference distance and result.
3. Add intermediate checkpoints and planned stop durations.
4. Set the end-of-race fatigue adjustment and whether altitude should be used.
5. Enter only carbohydrate and fluid rates that you have tested in training.
6. Generate the plan and download the two CSV files if needed.

The independent implementation in `race_planner.py` uses these public modelling
ideas from
[trail-race-planner](https://github.com/yama-asobi-lab/trail-race-planner):

- flat-equivalent distance: `distance_km + elevation_gain_m / 100`
- a distance-dependent Riegel exponent for reference-performance projection
- a piecewise grade-adjusted pace curve
- linear fatigue and a basic altitude multiplier
- normalization of point-level effort weights to the chosen moving time

The reference repository did not declare a software license when this feature
was implemented, so its source code was not copied. This planner is not an
official ITRA tool. It also does not yet model weather, technical terrain,
congestion, darkness, mandatory gear weight, or individual gastrointestinal
tolerance.

For sports nutrition, the default is a configurable planning aid rather than a
prescription. Public guidance commonly describes 30–60 g carbohydrate/hour for
1–2.5-hour endurance exercise and up to 90 g/hour for longer events when the
athlete has trained the strategy. Fluid targets should be individualized from
sweat-rate observations and conditions, while avoiding overdrinking. See the
[Australian Institute of Sport](https://www.ausport.gov.au/ais/nutrition/supplements/group_a/sports-foods2/sports-bar2/how-and-when-do-i-use-it)
and the
[NATA fluid-replacement position statement](https://pubmed.ncbi.nlm.nih.gov/28985128/).

Run all offline tests with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Online deployment

The Streamlit UI uses only browser uploads and does not read local filesystem
paths from the web form. No host address or port is hard-coded in
`.streamlit/config.toml`.

For Streamlit Community Cloud, set `streamlit_app.py` as the entrypoint; the
included `requirements.txt` contains the web-app dependencies.

### Browser-assisted ITRA capture

ITRA may challenge requests from shared cloud-server IP addresses. The companion
Chrome/Edge extension in `browser_extension/` keeps ITRA access in the user's
verified browser session and sends only normalized race-result fields to the
Streamlit app. It never transfers ITRA cookies or credentials.

For local extension testing:

1. Open `chrome://extensions` or `edge://extensions`.
2. Enable **Developer mode** and choose **Load unpacked**.
3. Select the repository's `browser_extension` directory.
4. Open an ITRA race-results page and complete any verification shown by ITRA.
5. When the results table is visible, click the extension and choose
   **抓取成绩并打开分析**.

The extension is intentionally restricted to ITRA race-results pages and
`https://trail-race-lab.streamlit.app/`. If the deployed app URL changes, update
the two matching entries in `browser_extension/manifest.json`, `APP_URL` in
`service-worker.js`, and `APP_HOST` in `content.js`.

Captured races can be persisted in the browser's IndexedDB from the
**Saved races and JSON backup** panel. Saving is always explicit: review the
results and any Race Score anchors, then click **Save / overwrite current
race**. The same panel can load or delete saved races and export/import a
portable JSON backup. Browser records stay on the current device and browser
profile; clearing site data removes them.

For a container platform:

```bash
docker build -t trail-race-lab .
docker run --rm -p 8501:8501 trail-race-lab
```

The container command reads the platform-provided `PORT` environment variable,
falling back to port 8501.
