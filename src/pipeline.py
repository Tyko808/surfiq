from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from numbers import Real
from pathlib import Path

import pandas as pd
import yaml

from src.database import save_forecasts_to_db
from src.open_meteo import fetch_spot_forecast
from src.recommendations import add_recommendation_reasons
from src.scoring import add_surf_scores

REQUIRED_SPOT_FIELDS = [
    "name",
    "latitude",
    "longitude",
    "skill_level",
    "offshore_wind_from_degrees",
    "ideal_wave_min_ft",
    "ideal_wave_max_ft",
    "ideal_swell_period_min",
    "ideal_swell_period_max",
    "ideal_tide",
]

ALLOWED_SKILL_LEVELS = {"beginner", "intermediate", "advanced"}
ALLOWED_TIDES = {"low", "mid", "high", "low_to_mid", "mid_to_high"}
DEFAULT_MAX_WORKERS = 6


def _is_number(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, Real)


def validate_spots(spots: list[dict]) -> None:
    errors = []
    seen_names = set()

    for index, spot in enumerate(spots, start=1):
        if not isinstance(spot, dict):
            errors.append(f"spot #{index}: each spot must be a mapping")
            continue

        spot_name = spot.get("name", f"spot #{index}")

        missing_fields = [field for field in REQUIRED_SPOT_FIELDS if field not in spot]
        if missing_fields:
            errors.append(f"{spot_name}: missing {', '.join(missing_fields)}")
            continue

        if not isinstance(spot["name"], str) or not spot["name"].strip():
            errors.append(f"spot #{index}: name must be a non-empty string")
            continue

        spot_name = spot["name"]

        if spot["name"] in seen_names:
            errors.append(f"{spot_name}: duplicate spot name")
        seen_names.add(spot["name"])

        if "region" in spot and (
            not isinstance(spot["region"], str) or not spot["region"].strip()
        ):
            errors.append(f"{spot_name}: region must be a non-empty string")

        if spot["skill_level"] not in ALLOWED_SKILL_LEVELS:
            errors.append(
                f"{spot_name}: skill_level must be one of {sorted(ALLOWED_SKILL_LEVELS)}"
            )

        if spot["ideal_tide"] not in ALLOWED_TIDES:
            errors.append(f"{spot_name}: ideal_tide must be one of {sorted(ALLOWED_TIDES)}")

        numeric_fields = [
            "latitude",
            "longitude",
            "offshore_wind_from_degrees",
            "ideal_wave_min_ft",
            "ideal_wave_max_ft",
            "ideal_swell_period_min",
            "ideal_swell_period_max",
        ]

        numeric_errors = []

        for field in numeric_fields:
            if not _is_number(spot[field]):
                numeric_errors.append(f"{spot_name}: {field} must be numeric")

        if numeric_errors:
            errors.extend(numeric_errors)
            continue

        if not -90 <= spot["latitude"] <= 90:
            errors.append(f"{spot_name}: latitude must be between -90 and 90")

        if not -180 <= spot["longitude"] <= 180:
            errors.append(f"{spot_name}: longitude must be between -180 and 180")

        if not 0 <= spot["offshore_wind_from_degrees"] <= 360:
            errors.append(f"{spot_name}: offshore_wind_from_degrees must be 0-360")

        if spot["ideal_wave_min_ft"] <= 0 or spot["ideal_wave_max_ft"] <= 0:
            errors.append(f"{spot_name}: ideal wave range must be positive")

        if spot["ideal_wave_min_ft"] > spot["ideal_wave_max_ft"]:
            errors.append(f"{spot_name}: ideal_wave_min_ft cannot exceed ideal_wave_max_ft")

        if spot["ideal_swell_period_min"] <= 0 or spot["ideal_swell_period_max"] <= 0:
            errors.append(f"{spot_name}: ideal swell period range must be positive")

        if spot["ideal_swell_period_min"] > spot["ideal_swell_period_max"]:
            errors.append(
                f"{spot_name}: ideal_swell_period_min cannot exceed ideal_swell_period_max"
            )

        if "big_wave_spot" in spot and not isinstance(spot["big_wave_spot"], bool):
            errors.append(f"{spot_name}: big_wave_spot must be true or false")

    if errors:
        detail = "\n- ".join(errors)
        raise ValueError(f"Invalid spot configuration:\n- {detail}")


def load_spots(config_path: str | Path = "config/spots.yaml") -> list[dict]:
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    spots = config.get("spots")

    if not isinstance(spots, list) or not spots:
        raise ValueError("Spot configuration must contain a non-empty `spots` list.")

    validate_spots(spots)

    return spots


def fetch_spot_forecasts(
    spots: list[dict],
    forecast_days: int,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> list[pd.DataFrame]:
    if max_workers <= 1 or len(spots) == 1:
        frames = []

        for spot in spots:
            try:
                frames.append(
                    fetch_spot_forecast(
                        spot=spot,
                        forecast_days=forecast_days,
                    )
                )
            except Exception as exc:
                raise RuntimeError(f"Could not fetch forecast for {spot['name']}.") from exc

        return frames

    frames: list[pd.DataFrame | None] = [None] * len(spots)
    worker_count = min(max_workers, len(spots))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(
                fetch_spot_forecast,
                spot=spot,
                forecast_days=forecast_days,
            ): index
            for index, spot in enumerate(spots)
        }

        for future in as_completed(futures):
            index = futures[future]
            spot = spots[index]

            try:
                frames[index] = future.result()
            except Exception as exc:
                raise RuntimeError(f"Could not fetch forecast for {spot['name']}.") from exc

    return [frame for frame in frames if frame is not None]


def build_forecast_dataset(
    config_path: str | Path = "config/spots.yaml",
    forecast_days: int = 7,
    user_skill_level: str = "intermediate",
    wind_tolerance: str = "medium",
    preferred_session_time: str = "any",
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> pd.DataFrame:
    spots = load_spots(config_path)
    frames = fetch_spot_forecasts(
        spots=spots,
        forecast_days=forecast_days,
        max_workers=max_workers,
    )

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
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> pd.DataFrame:
    df = build_forecast_dataset(
        forecast_days=forecast_days,
        user_skill_level=user_skill_level,
        wind_tolerance=wind_tolerance,
        preferred_session_time=preferred_session_time,
        max_workers=max_workers,
    )

    save_forecasts_to_db(df)
    save_forecast_csv(df)

    return df


if __name__ == "__main__":
    data = run_etl_and_store()
    print(f"Saved {len(data):,} rows to SQLite and CSV.")
