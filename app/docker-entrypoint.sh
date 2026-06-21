#!/bin/sh
set -e

PORT="${PORT:-8000}"

if [ "${SERVICE:-api}" = "dashboard" ]; then
  exec uv run --package dengue-app streamlit run app/src/dengue_app/dashboard.py \
    --server.port "$PORT" --server.address 0.0.0.0
else
  exec uv run --package dengue-app uvicorn dengue_app.main:app \
    --host 0.0.0.0 --port "$PORT"
fi
