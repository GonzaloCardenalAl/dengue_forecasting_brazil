"""Decision-support risk tiers: a presentation-layer business rule, not a
trained model, so it lives here rather than in dengue_ml.

A forecast row is classified into one of 4 ordinal tiers by ranking its
predicted_cases against that *same city's* own historical quarterly
distribution (cities differ by orders of magnitude -- a global threshold
would misclassify Vitória vs São Paulo) combined with the classifier's
epidemic probability for that quarter.
"""

import pandas as pd

RISK_TIERS = ["low", "moderate", "high", "very_high"]

# Tunable thresholds -- named constants so they're easy to adjust after
# seeing real forecast output, rather than magic numbers buried in logic.
EPIDEMIC_PROBA_THRESHOLD = 0.5
VERY_HIGH_PERCENTILE = 90.0
HIGH_PERCENTILE = 75.0
MODERATE_PERCENTILE = 50.0


def compute_risk_tier(
    predicted_cases: float,
    historical_quarterly_cases: pd.Series,
    epidemic_proba: float | None,
) -> str:
    """
    Returns one of RISK_TIERS.

    `historical_quarterly_cases` should be that city's own past quarterly
    casos_est values (excluding the forecast horizon). If empty, percentile
    rank defaults to the moderate midpoint rather than erroring.
    """
    historical_quarterly_cases = historical_quarterly_cases.dropna()
    if historical_quarterly_cases.empty:
        percentile = MODERATE_PERCENTILE
    else:
        percentile = float((historical_quarterly_cases < predicted_cases).mean() * 100)

    is_epidemic = epidemic_proba is not None and epidemic_proba >= EPIDEMIC_PROBA_THRESHOLD

    if is_epidemic and percentile >= VERY_HIGH_PERCENTILE:
        return "very_high"
    if is_epidemic or percentile >= HIGH_PERCENTILE:
        return "high"
    if percentile >= MODERATE_PERCENTILE:
        return "moderate"
    return "low"


RECOMMENDATIONS: dict[str, dict] = {
    "low": {
        "emoji": "🟢",
        "label": "Low Risk",
        "description": "Expected cases below historical threshold",
        "recommendations": {
            "Public Health": [
                "Maintain routine surveillance.",
                "Continue standard vector-control activities.",
            ],
            "Healthcare System": [
                "Maintain baseline hospital staffing.",
            ],
            "Supply Chain": [
                "Standard vaccine inventory levels.",
            ],
        },
    },
    "moderate": {
        "emoji": "🟡",
        "label": "Moderate Risk",
        "description": "Elevated transmission expected",
        "recommendations": {
            "Surveillance": [
                "Increase mosquito surveillance frequency.",
                "Monitor early outbreak indicators weekly.",
            ],
            "Communication": [
                "Launch public awareness campaigns.",
            ],
            "Supply Chain": [
                "Review vaccine inventory and safety stock.",
            ],
            "Healthcare": [
                "Prepare additional personnel if needed.",
            ],
        },
    },
    "high": {
        "emoji": "🟠",
        "label": "High Risk",
        "description": "Significant outbreak probability",
        "recommendations": {
            "Public Health": [
                "Intensify vector-control interventions.",
                "Increase laboratory capacity.",
            ],
            "Healthcare System": [
                "Prepare emergency departments.",
                "Pre-position diagnostics and treatments.",
            ],
            "Supply Chain": [
                "Increase vaccine distribution.",
                "Activate contingency inventory plans.",
            ],
            "Monitoring": [
                "Weekly forecast updates.",
            ],
        },
    },
    "very_high": {
        "emoji": "🔴",
        "label": "Very High Risk",
        "description": "Severe outbreak scenario",
        "recommendations": {
            "Emergency Preparedness": [
                "Activate outbreak response protocols.",
            ],
            "Healthcare": [
                "Expand hospital capacity.",
                "Increase staffing.",
            ],
            "Supply Chain": [
                "Prioritize vaccine allocation.",
                "Mobilize reserve inventories.",
            ],
            "Government": [
                "Intensify communication campaigns.",
                "Coordinate across municipalities.",
            ],
        },
    },
}
