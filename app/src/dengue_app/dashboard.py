"""DORA -- Dengue Outbreak Response Assistant.

Streamlit dashboard, talks to the dengue_app FastAPI backend over HTTP only
(no direct dengue_ml/filesystem access from this module -- keeps the API
independently usable by other consumers). Entrypoint only: password gate,
admin refresh, header/nav, then dispatch to the selected page.
"""

import hmac
import os

import requests
import streamlit as st
from dotenv import load_dotenv

from dengue_app import api_client, layout
from dengue_app.layout import DORA_LOGO_PATH
from dengue_app.views.about import render_about
from dengue_app.views.home import render_home

load_dotenv()

API_URL = os.environ.get("DENGUE_API_URL", "http://localhost:8000")
DASHBOARD_PASSWORD = os.environ.get("DENGUE_DASHBOARD_PASSWORD")

st.set_page_config(page_title="DORA", page_icon=str(DORA_LOGO_PATH), layout="wide")

# Password gate -- skipped entirely if DENGUE_DASHBOARD_PASSWORD isn't set
# (so local dev without the env var still works; production deployments must
# set it). One shared secret, no per-user accounts.
if DASHBOARD_PASSWORD and not st.session_state.get("authenticated"):
    st.title("DORA")
    pw = st.text_input("Password", type="password")
    if st.button("Log in"):
        if hmac.compare_digest(pw, DASHBOARD_PASSWORD):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    st.stop()

with st.sidebar:
    st.markdown("### Admin")
    if st.button("Refresh data & forecast"):
        with st.spinner("Fetching latest InfoDengue data and re-running inference..."):
            try:
                r = requests.post(
                    f"{API_URL}/admin/refresh",
                    headers={"X-Admin-Token": DASHBOARD_PASSWORD or ""},
                    timeout=120,
                )
                r.raise_for_status()
                summary = r.json()
                api_client.api_get.clear()  # bust the 300s cache so the refreshed data shows immediately
                st.success(
                    f"Refreshed. Rows added: {summary.get('rows_added')}. "
                    f"Forecast now covers: {', '.join(summary.get('forecast_quarters', []))}"
                )
                st.rerun()
            except requests.HTTPError as e:
                detail = e.response.text if e.response is not None else str(e)
                st.error(f"Refresh failed: {detail}")
            except requests.RequestException as e:
                st.error(f"Refresh failed: {e}")

layout.inject_css()
layout.render_header()

if st.session_state["page"] == "Home":
    render_home()
else:
    render_about()
