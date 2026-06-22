# Running the DORA App

This document covers how to launch the **DORA** (Dengue Outbreak Response Assistant) application — the FastAPI backend and Streamlit dashboard that serve the forecasting pipeline's outputs. For how the underlying ML pipeline works, see [README.md](README.md).

## Architecture

```
Streamlit dashboard (dashboard.py)
   │  via api_client.py (requests, cached 300s)
   ▼
FastAPI backend (main.py)
   │  via data.py
   ▼
Trained model artifacts (ml/results/<run_dir>/*.csv, *.pkl)
```

The Streamlit dashboard never reads model artifacts directly — all data flows through the FastAPI backend's HTTP endpoints. The default ports are:

- **Backend (FastAPI)**: `8000`
- **Frontend (Streamlit)**: configured by the operator; the dashboard finds the backend via the `DENGUE_API_URL` environment variable (default `http://localhost:8000`)

In production (Docker), one image (`app/Dockerfile`) serves either role depending on the `SERVICE` environment variable, dispatched by `app/docker-entrypoint.sh`. **Locally, you run the two processes directly with `uv run` in two terminals** — you don't need Docker or the entrypoint script for local development.

## Prerequisites

- `uv` installed, Python `>=3.13` (see [README.md](README.md#reproducibility) for `uv sync`).
- Trained model artifacts available — either:
  - `ml/results/production_run/` (committed to the repo, used out of the box), or
  - a fresh `ml/results/run_<timestamp>/` produced by running the ML pipeline (see [README.md](README.md#pipeline-architecture); not covered here).
- Know the following environment variables before starting (names/purpose only — set real values via your own `.env` or shell exports; a local `.env` file is gitignored and should never be committed):
  - `DENGUE_RUN_DIR` — which run directory to read artifacts from. If unset, the backend falls back to `ml/results/latest_run.txt`.
  - `DENGUE_API_URL` — backend URL the dashboard calls (default `http://localhost:8000`).
  - `DENGUE_ADMIN_TOKEN` — required header value (`X-Admin-Token`) to call `POST /admin/refresh` on the backend.
  - `DENGUE_DASHBOARD_PASSWORD` — password gate for the dashboard. **If unset, the password gate is skipped entirely.** Note: the dashboard's "Refresh data & forecast" button sends this same password as the `X-Admin-Token` value when calling `/admin/refresh` — so for the refresh button to work, `DENGUE_ADMIN_TOKEN` (on the backend) and `DENGUE_DASHBOARD_PASSWORD` (on the dashboard) must be set to the same value.

## Starting the Backend

```bash
uv run --package dengue-app uvicorn dengue_app.main:app --host 0.0.0.0 --port 8000
```

Add `--reload` for local development to auto-restart on code changes (not used in the Docker entrypoint, which is meant for production).

If you need a specific run directory rather than the latest one, set `DENGUE_RUN_DIR` before starting:

```bash
export DENGUE_RUN_DIR=/path/to/ml/results/run_20260101_120000
```

Verify it's up:

```bash
curl http://localhost:8000/health
```

## Starting the Frontend

```bash
uv run --package dengue-app streamlit run app/src/dengue_app/dashboard.py --server.port 8501 --server.address 0.0.0.0
```

**Note:** use a different port than the backend (e.g. `8501`) since both run on `localhost` simultaneously — unlike the Docker deployment, where each service is a separate container and can reuse the same default port.

If the backend is running on the default `http://localhost:8000`, you don't need to set `DENGUE_API_URL`. Otherwise:

```bash
export DENGUE_API_URL=http://localhost:8000
```

If `DENGUE_DASHBOARD_PASSWORD` is unset, the dashboard loads directly with no login prompt.

## Running the Complete Application

1. `uv sync` (if dependencies aren't already installed).
2. Confirm model artifacts exist: `ml/results/production_run/` or a `run_<timestamp>/` you've generated, and set `DENGUE_RUN_DIR` if needed.
3. (Optional) set `DENGUE_ADMIN_TOKEN` / `DENGUE_DASHBOARD_PASSWORD` if you want the refresh button and login gate active.
4. **Terminal 1** — start the backend (see above).
5. Confirm `curl http://localhost:8000/health` returns `{"status": "ok"}`.
6. **Terminal 2** — start the frontend on a different port (see above).
7. Open `http://localhost:8501` in a browser.
8. Use the dashboard: select a city, switch between current forecast / future forecast / historical views, and (if configured) use the sidebar "Refresh data & forecast" button to pull the latest InfoDengue data and re-run inference.

## API Endpoints

All endpoints are served by `app/src/dengue_app/main.py`:

| Method | Path | Query params | Purpose |
|---|---|---|---|
| GET | `/health` | — | Health check |
| POST | `/admin/refresh` | header `X-Admin-Token` | Fetch latest InfoDengue data and regenerate the forecast with the currently-served model (no retraining) |
| GET | `/cities` | — | List of cities with coordinates and population |
| GET | `/risk/recommendations` | — | Decision-support text by risk tier |
| GET | `/forecast/quarterly` | `city` | Next 4 quarters' predicted cases, 95% CI, and risk tier |
| GET | `/backtest/quarterly` | `city`, `start`, `end` | Out-of-fold predicted vs. actual cases per quarter |
| GET | `/history/quarterly` | `city`, `start`, `end` | Actual quarterly case counts |
| GET | `/history/seasonal-profile` | `city` | Per-season (Q4→Q1→Q2→Q3) incidence profile |
| GET | `/history/monthly` | `city` | Actual monthly case counts |
| GET | `/forecast/monthly` | `city` | Next ~12 months' predicted cases (no CI) |

## Troubleshooting

- **Dashboard shows no data / connection errors**: confirm the backend is running and reachable — `curl $DENGUE_API_URL/health` (or `http://localhost:8000/health` if unset). Check that `DENGUE_API_URL` on the dashboard process matches where the backend is actually listening.
- **Port conflicts**: change `--server.port` (Streamlit) or `--port` (uvicorn) to a free port.
- **Missing model artifacts / empty forecasts**: check that `ml/results/production_run/` exists, or that `DENGUE_RUN_DIR` points at a valid run directory containing `final_quarterly_forecast.csv`/`final_weekly_forecast.csv`. If `DENGUE_RUN_DIR` is unset, the backend falls back to whatever `ml/results/latest_run.txt` points to.
- **`/admin/refresh` returns 401**: the `X-Admin-Token` header doesn't match `DENGUE_ADMIN_TOKEN` on the backend process. If triggered via the dashboard's refresh button, remember it sends `DENGUE_DASHBOARD_PASSWORD` as that header — make sure both processes were started with the same value (e.g. from a shared `.env`).
- **Missing environment variables**: most variables have safe defaults (password gate and `DENGUE_API_URL` default work for local single-machine use) — only `DENGUE_ADMIN_TOKEN`/`DENGUE_DASHBOARD_PASSWORD` need to be set deliberately, and only if you want the refresh feature or login gate active.
- **Dependency problems**: re-run `uv sync` from the repo root; if `dengue-ml`/`dengue-core` imports fail, confirm you're using `uv run --package dengue-app` (not a bare `python`/`streamlit` invocation) so the workspace packages are on the path.
- **No automated tests**: `app/` has no test suite today (only `ml/tests/` exists — see [README.md](README.md#testing)), so the checks above (`curl`, browser) are the only available diagnostics.
- **TODO**: Render deployment specifics. No `render.yaml`, `docker-compose.yml`, or `Procfile` is committed to this repo. The `Dockerfile` + `docker-entrypoint.sh` (`SERVICE=api`/`SERVICE=dashboard`) suggest deployment as two Render services built from the same image, but the actual Render-side service configuration (env vars, build/start commands) is presumably set up directly in the Render dashboard and isn't reproducible from this repo alone — document it here once confirmed.
