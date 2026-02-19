# Deployment Guide (Vercel)

This project is deployed as a Python serverless app on Vercel.

## 1. Prerequisites

- A Vercel account and a linked project.
- Python 3.10+ for local development.
- Vercel Blob store enabled in your Vercel project.

## 2. Runtime layout

- `vercel.json` routes all requests to `api/index.py`.
- `api/index.py` exposes the Flask app from `app.py`.
- Existing Flask routes and templates remain unchanged.

## 3. Required environment variables

Set these in Vercel Project Settings -> Environment Variables:

- `FLASK_SECRET_KEY`: random secret for Flask sessions.
- `STORAGE_BACKEND=vercel_blob`
- `BLOB_READ_WRITE_TOKEN`: Vercel Blob read/write token.
- `VERCEL_BLOB_BASE_URL`: your Blob store host URL (required in production).

Optional:

- `VERCEL_BLOB_PATH_PREFIX` (for namespacing blob paths)

## 4. Shared data keys

The app uses fixed shared blob paths:

- `collection/collection_data.csv`
- `releases/releases_data.csv`

Uploads overwrite these paths (single shared dataset behavior).

## 5. Deploy

```bash
vercel
```

For production:

```bash
vercel --prod
```

## 6. Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

For local filesystem mode (default), no Blob env vars are required.
For Blob-backed local testing, set `STORAGE_BACKEND=vercel_blob` and `BLOB_READ_WRITE_TOKEN`.

## 7. Data freshness reminder

The UI shows a non-blocking reminder when dataset latest month is older than the previous calendar month.

Example on February 16, 2026:
- Latest month `2025-12` -> stale (reminder shown)
- Latest month `2026-01` -> fresh (no reminder)

## 8. Persistence verification checklist (production)

Use this after any env var or deployment change:

1. Confirm both users are on the same production URL.
2. Confirm production env vars include only the required Blob contract:
   - `STORAGE_BACKEND=vercel_blob`
   - `BLOB_READ_WRITE_TOKEN`
   - `VERCEL_BLOB_BASE_URL`
   - `FLASK_SECRET_KEY`
3. Upload one CSV for each trend (`collection`, `releases`).
4. Check:
   - `/api/storage_health?trend=collection`
   - `/api/storage_health?trend=releases`
5. Verify both endpoints return `read_status: "found"` and expected `storage_target.expected_path`.
