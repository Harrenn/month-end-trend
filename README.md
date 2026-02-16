# Month-End Trend Calculator

Flask web app for month-end trend estimation and backtesting using a Legacy RMLA-style historical attainment model.

## What it does

- Supports two trend types:
- `collection` using `collection_data.csv`
- `releases` using `releases_data.csv` (with optional `loan_type` segmentation)
- Lets users:
- Upload CSV data
- Generate live month-end trend projections from current MTD values
- Run multi-month backtests
- Run single-calendar-month backtests across years
- Shows a freshness reminder when the latest dataset month is older than the previous calendar month.

## Tech stack

- Python + Flask
- Pandas + NumPy
- Bootstrap + jQuery (server-rendered templates)

## Project layout

- `app.py`: main Flask app, data prep, model logic, and API endpoints
- `api/index.py`: Vercel entrypoint exposing the Flask app
- `templates/`: UI pages (`index`, `live_trend`, `backtest`, `upload`, base layout)
- `data/collection/collection_data.csv`: collection dataset
- `data/releases/releases_data.csv`: releases dataset
- `tests/test_app.py`: storage and data-freshness unit tests
- `vercel.json`: Vercel routing/build config
- `DEPLOYMENT.md`: deployment notes and env var setup

## Local development

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set environment variables

```bash
export FLASK_SECRET_KEY="change-this-in-real-envs"
```

Optional debug mode:

```bash
export FLASK_DEBUG=1
```

### 4. Run the app

```bash
python app.py
```

App runs at `http://127.0.0.1:5002`.

## Running tests

```bash
python -m unittest discover -s tests
```

## Data format requirements

### Collection trend CSV

Required columns:

- `date`
- `collection`

Example:

```csv
date,collection
2026-01-31,9539989.06
2026-01-30,9226222.34
```

### Releases trend CSV

Required columns:

- `date`
- `releases`

Optional:

- `loan_type` (enables segment selection in the UI/API session flow)

Example:

```csv
date,releases,loan_type
2025-05-07,25,BN ROPALI
2021-11-24,54,REPO
```

## Storage backends

Storage is configured with `STORAGE_BACKEND`:

- `local` (default): reads/writes CSVs in `data/...`
- `vercel_blob`: reads/writes shared CSVs in Vercel Blob storage

For `vercel_blob`, set:

- `STORAGE_BACKEND=vercel_blob`
- `BLOB_READ_WRITE_TOKEN`

Optional:

- `VERCEL_BLOB_BASE_URL` (defaults to `https://blob.vercel-storage.com`)
- `VERCEL_BLOB_PATH_PREFIX` (namespace prefix)

## API endpoints

Most API endpoints require a selected trend stored in session. In browser flows this is set by `POST /select_trend`.

- `GET /api/data_status`
- `POST /api/live_trend`
- `POST /api/backtest`
- `POST /api/single_month_backtest`
- `POST /api/upload` (multipart form with `file`)

If calling by script, keep cookies between requests so session state persists.

## Deployment

- Vercel runtime entrypoint: `api/index.py`
- Route config: `vercel.json`
- Deployment setup details: `DEPLOYMENT.md`
