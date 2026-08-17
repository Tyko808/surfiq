from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.database import save_forecasts_to_db
from src.open_meteo import fetch_spot_forecast
from src.recommendations import add_recommendation_reasons
from src.scoring import add_surf_scores


def load_spots(config_path: str | Path = "config/spots.yaml") -> list[dict]:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    return config["spots"]


def build_forecast_dataset(
    config_path: str | Path = "config/spots.yaml",
    forecast_days: int = 7,
    user_skill_level: str = "intermediate",
    wind_tolerance: str = "medium",
    preferred_session_time: str = "any",
) -> pd.DataFrame:
    spots = load_spots(config_path)
    frames = []

    for spot in spots:
        spot_df = fetch_spot_forecast(
            spot=spot,
            forecast_days=forecast_days,
        )

        frames.append(spot_df)

    df = pd.concat(frames, ignore_index=True)

    df = add_surf_scores(
        df,
        user_skill_level=user_skill_level,
        wind_tolerance=wind_tolerance,
        preferred_session_time=preferred_session_time,
    )

    df = add_recommendation_reasons(
        df,
        user_skill_level=user_skill_level,
    )

    df = df.sort_values(["time", "spot"]).reset_index(drop=True)

    return df


def save_forecast_csv(
    df: pd.DataFrame,
    output_path: str | Path = "data/surf_forecast.csv",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(output_path, index=False)


def run_etl_and_store(
    forecast_days: int = 7,
    user_skill_level: str = "intermediate",
    wind_tolerance: str = "medium",
    preferred_session_time: str = "any",
) -> pd.DataFrame:
    df = build_forecast_dataset(
        forecast_days=forecast_days,
        user_skill_level=user_skill_level,
        wind_tolerance=wind_tolerance,
        preferred_session_time=preferred_session_time,
    )

    save_forecasts_to_db(df)
    save_forecast_csv(df)

    return df


if __name__ == "__main__":
    data = run_etl_and_store()
    print(f"Saved {len(data):,} rows to SQLite and CSV.")
