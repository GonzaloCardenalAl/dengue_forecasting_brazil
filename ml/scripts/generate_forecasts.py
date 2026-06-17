#!/usr/bin/env python
"""Load the trained final model and generate 4-quarter-ahead forecasts with 95% CI."""
import joblib
from dengue_ml.run_dir import get_latest_run_dir
from dengue_ml.preprocessing import prepare_model_table
from dengue_ml.forecasting.forecast_next_4q import generate_next_4q_forecast
from dengue_ml.reporting.results_tables import final_forecast_table
from dengue_ml.reporting.plots import plot_final_forecast

if __name__ == "__main__":
    run_dir = get_latest_run_dir()
    print(f"Run directory: {run_dir}")

    model_path = run_dir / "final_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found — run train_final_model.py first."
        )

    artifact    = joblib.load(model_path)
    df          = prepare_model_table()
    forecast_df = generate_next_4q_forecast(artifact, df)

    # Unfiltered history (includes the still-converging final quarter) so the
    # forecast plot can show its genuinely wide credible interval.
    full_history_df = prepare_model_table(apply_reliability_cutoff=False)

    final_forecast_table(forecast_df, outputs_dir=run_dir)
    plot_final_forecast(forecast_df, full_history_df, outputs_dir=run_dir)

    print(forecast_df.to_string(index=False))
    print(f"\nForecast saved to {run_dir / 'final_4q_forecast.csv'}")
