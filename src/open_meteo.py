from __future__ import annotations

import pandas as pd
import requests

MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"
WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

MARINE_HOURLY = [
    "wave_height",
    "wave_direction",
    "wave_period",
    "swell_wave_height",
    "swell_wave_direction",
    "swell_wave_period",
    "sea_surface_temperature",
]

WEATHER_HOURLY = [
    "temperature_2m",
    "precipitation_probability",
    "wind_speed_10m",
    "wind_direction_10m",
]


def _get_json(url: str, params: dict) -> dict:
    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def _hourly_to_df(payload: dict) -> pd.DataFrame:
    hourly = payload.get("hourly", {})

    if not hourly or "time" not in hourly:
        raise ValueError(f"API response missing hourly data: {payload}")

    df = pd.DataFrame(hourly)
    df["time"] = pd.to_datetime(df["time"])

    return df


def fetch_marine_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 7,
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(MARINE_HOURLY),
        "timezone": "auto",
        "forecast_days": forecast_days,
    }

    payload = _get_json(MARINE_URL, params)
    return _hourly_to_df(payload)


def fetch_weather_forecast(
    latitude: float,
    longitude: float,
    forecast_days: int = 7,
) -> pd.DataFrame:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ",".join(WEATHER_HOURLY),
        "timezone": "auto",
        "forecast_days": forecast_days,
    }

    payload = _get_json(WEATHER_URL, params)
    return _hourly_to_df(payload)


def fetch_spot_forecast(
    spot: dict,
    forecast_days: int = 7,
) -> pd.DataFrame:
    marine = fetch_marine_forecast(
        latitude=spot["latitude"],
        longitude=spot["longitude"],
        forecast_days=forecast_days,
    )

    weather = fetch_weather_forecast(
        latitude=spot["latitude"],
        longitude=spot["longitude"],
        forecast_days=forecast_days,
    )

    merged = marine.merge(weather, on="time", how="inner")

    for key, value in spot.items():
        merged[key] = value

    merged = merged.rename(columns={"name": "spot"})

    return merged