"""Shared CSS and header (logo / nav / logo) used by every page."""

from pathlib import Path

import streamlit as st

_ASSETS_DIR = Path(__file__).resolve().parents[2]
DORA_LOGO_PATH = _ASSETS_DIR / "DORA_logo.png"
INTELLISHORE_LOGO_PATH = _ASSETS_DIR / "Intellishore_logo.png"

PAGES = ["Home", "About us"]


def inject_css() -> None:
    # Font (Inter) + larger mode selector + larger city selector + larger
    # subtitle/title -- several selector variants stacked since Streamlit's
    # internal data-testids can shift between versions.
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

        .dora-title h1 { font-size: 3.2rem !important; margin: 0 !important; line-height: 1; }
        .dora-subtitle { font-size: 1.25rem !important; color: #555; margin-top: 4px; }

        div[data-testid="stRadio"] [data-testid="stWidgetLabel"] p { font-size: 1.3rem !important; }
        div[data-testid="stRadio"] label p { font-size: 1.2rem !important; }

        div[data-testid="stSelectbox"] label p { font-size: 1.25rem !important; }
        div[data-testid="stSelectbox"] div[data-baseweb="select"] * { font-size: 1.15rem !important; }
        div[data-baseweb="popover"] li { font-size: 1.15rem !important; }
        ul[role="listbox"] * { font-size: 1.15rem !important; }

        .dora-header { border-bottom: 1px solid #e6e6e6; padding-bottom: 12px; margin-bottom: 18px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """DORA logo (left) / Home·About nav (middle) / Intellishore logo
    (right), in one row -- mirrors the InfoDengue header the user wants to
    match. Active page is highlighted via st.button's built-in `type`
    rather than custom per-button CSS classes, which Streamlit doesn't make
    easy to target reliably across versions."""
    st.session_state.setdefault("page", "Home")

    st.markdown('<div class="dora-header">', unsafe_allow_html=True)
    logo_col, nav_col, logo2_col = st.columns([1, 3, 1], vertical_alignment="center")

    with logo_col:
        st.image(str(DORA_LOGO_PATH), width=200)

    with nav_col:
        _, *btn_cols, _ = st.columns([2, 1, 1, 2])
        for col, page_name in zip(btn_cols, PAGES):
            with col:
                is_active = st.session_state["page"] == page_name
                if st.button(page_name, key=f"nav_{page_name}",
                             type="primary" if is_active else "secondary", width="stretch"):
                    st.session_state["page"] = page_name
                    st.rerun()

    with logo2_col:
        st.image(str(INTELLISHORE_LOGO_PATH), width=140)
    st.markdown("</div>", unsafe_allow_html=True)
