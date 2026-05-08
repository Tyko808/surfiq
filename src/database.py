from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/surfiq.db")


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def initialize_database(db_path: Path = DB_PATH) -> None:
    with get_connection(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS surf_forecasts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                spot TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                time TEXT NOT NULL,
                skill_level TEXT,
                wave_height REAL,
                wave_height_ft REAL,
                wave_direction REAL,
                wave_period REAL,
                swell_wave_height REAL,
                swell_wave_height_ft REAL,
                swell_wave_direction REAL,
                swell_wave_period REAL,
                sea_surface_temperature REAL,
                temperature_2m REAL,
                precipitation_probability REAL,
                wind_speed_10m REAL,
                wind_speed_mph REAL,
                wind_direction_10m REAL,
                offshore_wind_from_degrees REAL,
                ideal_wave_min_ft REAL,
                ideal_wave_max_ft REAL,
                ideal_swell_period_min REAL,
                ideal_swell_period_max REAL,
                ideal_tide TEXT,
                surf_score REAL,
                surf_rating TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_surf_forecasts_spot_time
            ON surf_forecasts (spot, time)
            """
        )


def save_forecasts_to_db(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    initialize_database(db_path)

    columns = [
        "spot",
        "latitude",
        "longitude",
        "time",
        "skill_level",
        "wave_height",
        "wave_height_ft",
        "wave_direction",
        "wave_period",
        "swell_wave_height",
        "swell_wave_height_ft",
        "swell_wave_direction",
        "swell_wave_period",
        "sea_surface_temperature",
        "temperature_2m",
        "precipitation_probability",
        "wind_speed_10m",
        "wind_speed_mph",
        "wind_direction_10m",
        "offshore_wind_from_degrees",
        "ideal_wave_min_ft",
        "ideal_wave_max_ft",
        "ideal_swell_period_min",
        "ideal_swell_period_max",
        "ideal_tide",
        "surf_score",
        "surf_rating",
    ]

    available_columns = [col for col in columns if col in df.columns]
    clean_df = df[available_columns].copy()
    clean_df["time"] = clean_df["time"].astype(str)

    with get_connection(db_path) as conn:
        clean_df.to_sql(
            "surf_forecasts",
            conn,
            if_exists="append",
            index=False,
        )


def load_forecasts_from_db(db_path: Path = DB_PATH) -> pd.DataFrame:
    initialize_database(db_path)

    with get_connection(db_path) as conn:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM surf_forecasts
            ORDER BY time, spot
            """,
            conn,
        )

    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
        df["created_at"] = pd.to_datetime(df["created_at"])

    return df


def clear_database(db_path: Path = DB_PATH) -> None:
    initialize_database(db_path)

    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM surf_forecasts")