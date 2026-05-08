from __future__ import annotations

import pandas as pd


def generate_recommendation_reason(row: pd.Series) -> str:
    wave_ft = row.get("wave_height_ft")
    swell_period = row.get("swell_wave_period")
    wind_mph = row.get("wind_speed_mph")
    wind_direction = row.get("wind_direction_10m")

    parts = []

    if pd.notna(wave_ft):
        parts.append(f"waves around {wave_ft:.1f} ft")

    if pd.notna(swell_period):
        if swell_period >= 10:
            parts.append(f"a strong {swell_period:.0f}-second swell")
        else:
            parts.append(f"a shorter {swell_period:.0f}-second swell")

    if pd.notna(wind_mph):
        if wind_mph <= 10:
            parts.append(f"light {wind_mph:.1f} mph wind")
        elif wind_mph <= 18:
            parts.append(f"manageable {wind_mph:.1f} mph wind")
        else:
            parts.append(f"stronger {wind_mph:.1f} mph wind")

    if pd.notna(wind_direction):
        parts.append(f"wind coming from {wind_direction:.0f}°")

    if not parts:
        return "This window ranks highly based on the combined surf score."

    return "This looks like a strong surf window because it has " + ", ".join(parts[:4]) + "."


def add_recommendation_reasons(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["recommendation_reason"] = output.apply(generate_recommendation_reason, axis=1)

    return output


def get_top_recommendations(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    top = df.sort_values("surf_score", ascending=False).head(n).copy()
    top = add_recommendation_reasons(top)

    return top