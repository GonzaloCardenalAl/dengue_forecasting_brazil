"""Static About page: project description + contact info."""

import streamlit as st


def render_about() -> None:
    st.title("About DORA")
    st.markdown(
        """
        **DORA (Dengue Outbreak Response Assistant)** forecasts quarterly dengue case
        counts and epidemic risk for four state capitals in Southeast Brazil --
        Vitória, Belo Horizonte, Rio de Janeiro, and São Paulo -- using historical
        InfoDengue surveillance data, climate indicators, and El Niño/SST indices.

        The forecasts feed a set of decision-support recommendations (surveillance,
        healthcare staffing, vector control, supply chain) tiered by predicted case
        volume and the model's own epidemic probability, intended to give public
        health teams lead time ahead of outbreak season.
        """
    )
    st.markdown("---")
    st.markdown("**Contact**")
    st.markdown("gonzalocardenalal@gmail.com")
