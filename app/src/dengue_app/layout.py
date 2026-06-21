"""Shared CSS and header (logo / title+subtitle / nav / logo) used by every page."""

from pathlib import Path

import streamlit as st

_ASSETS_DIR = Path(__file__).resolve().parents[2]
DORA_LOGO_PATH = _ASSETS_DIR / "DORA_logo.png"
INTELLISHORE_LOGO_PATH = _ASSETS_DIR / "Intellishore_logo.png"

PAGES = ["Home", "About us"]

NAV_BLUE = "#2563eb"
NAV_BLUE_HOVER = "#1d4ed8"


def _nav_key(page_name: str) -> str:
    return f"nav_{page_name.lower().replace(' ', '_')}"


def inject_css() -> None:
    # Font (Inter) + larger mode selector + larger city selector + larger
    # subtitle/title -- several selector variants stacked since Streamlit's
    # internal data-testids can shift between versions.
    nav_selectors = ", ".join(f'.st-key-{_nav_key(p)} button[kind="primary"]' for p in PAGES)
    nav_hover_selectors = ", ".join(f'.st-key-{_nav_key(p)} button[kind="primary"]:hover' for p in PAGES)
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
        html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

        .dora-header-title {{
            text-align: center; font-size: 2.1rem; font-weight: 400;
            color: darkgreen; margin: 0; line-height: 1.2;
        }}
        .dora-header-title strong {{ font-weight: 800; }}
        .dora-header-subtitle {{
            text-align: center; font-style: italic; font-size: 1.3rem;
            color: #555; margin: 4px 0 14px 0;
        }}

        div[data-testid="stRadio"] [data-testid="stWidgetLabel"] p {{ font-size: 1.3rem !important; }}
        div[data-testid="stRadio"] label p {{ font-size: 1.2rem !important; }}

        div[data-testid="stSelectbox"] label p {{ font-size: 1.25rem !important; }}
        div[data-testid="stSelectbox"] div[data-baseweb="select"] * {{ font-size: 1.15rem !important; }}
        div[data-baseweb="popover"] li {{ font-size: 1.15rem !important; }}
        ul[role="listbox"] * {{ font-size: 1.15rem !important; }}

        .st-key-dora_header {{
            border-bottom: 2px solid #c7d6e8; padding-bottom: 14px; margin-bottom: 22px;
        }}
        .st-key-dora_filters {{
            background-color: #eaf2fb; padding: 18px 20px 4px 20px;
            border-radius: 10px; margin-bottom: 20px;
        }}

        {nav_selectors} {{ background-color: {NAV_BLUE} !important; border-color: {NAV_BLUE} !important; }}
        {nav_hover_selectors} {{ background-color: {NAV_BLUE_HOVER} !important; border-color: {NAV_BLUE_HOVER} !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """DORA logo (left) / title+subtitle+nav (middle, stacked, centered) /
    Intellishore logo (right). Wrapped in a real st.container (not a raw
    unclosed <div>, which Streamlit's HTML parser would auto-close before
    the next element and so wouldn't actually enclose the columns below it)
    so the CSS bottom-border below acts as a single separator bar between
    the header and the page content that follows. Active nav page is
    highlighted via st.button's `type="primary"`, recolored blue by
    inject_css() above (scoped to these two buttons via Streamlit's
    auto-generated `st-key-<key>` class, so the rest of the app's
    primary-colored widgets are untouched)."""
    st.session_state.setdefault("page", "Home")

    with st.container(key="dora_header"):
        logo_col, center_col, logo2_col = st.columns([1, 4, 1], vertical_alignment="center")

        with logo_col:
            st.image(str(DORA_LOGO_PATH), width=160)

        with center_col:
            st.markdown(
                '<h1 class="dora-header-title"><strong>DORA</strong> : Dengue Outbreak Response Assistant</h1>'
                '<p class="dora-header-subtitle">AI-powered dengue forecasting and risk monitoring '
                "for Southeast Brazil</p>",
                unsafe_allow_html=True,
            )
            _, *btn_cols, _ = st.columns([3, 1, 1, 3])
            for col, page_name in zip(btn_cols, PAGES):
                with col:
                    is_active = st.session_state["page"] == page_name
                    if st.button(page_name, key=_nav_key(page_name),
                                 type="primary" if is_active else "secondary", width="stretch"):
                        st.session_state["page"] = page_name
                        st.rerun()

        with logo2_col:
            st.image(str(INTELLISHORE_LOGO_PATH), width=120)
