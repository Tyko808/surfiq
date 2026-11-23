from __future__ import annotations

import pandas as pd


def circular_distance_degrees(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def scale_score(value: float, ideal_min: float, ideal_max: float, max_score: float) -> float:
    if pd.isna(value):
        return 0

    if ideal_min <= value <= ideal_max:
        return max_score

    if value < ideal_min:
        ratio = value / ideal_min if ideal_min != 0 else 0
        return max(0, max_score * ratio * 0.75)

    overage = value - ideal_max
    tolerance = ideal_max * 0.75

    penalty_ratio = max(0, 1 - (overage / tolerance))
    return max_score * penalty_ratio * 0.75


def wave_height_score(row: pd.Series, user_skill_level: str) -> float:
    wave_ft = row.get("wave_height_ft")

    spot_min = float(row.get("ideal_wave_min_ft", 2))
    spot_max = float(row.get("ideal_wave_max_ft", 5))
    big_wave_spot = row.get("big_wave_spot", False)

    skill_adjustments = {
        "beginner": (0.5, 3.0),
        "intermediate": (2.0, 6.0),
        "advanced": (3.0, 12.0),
    }

    skill_min, skill_max = skill_adjustments.get(
        user_skill_level,
        skill_adjustments["intermediate"],
    )

    is_big_wave_spot = pd.notna(big_wave_spot) and bool(big_wave_spot)

    if is_big_wave_spot and user_skill_level != "advanced":
        spot_score = scale_score(wave_ft, spot_min, spot_max, 30)
        return min(spot_score, 10)

    if user_skill_level == "advanced" and is_big_wave_spot:
        skill_max = max(skill_max, spot_max)

    ideal_min = max(spot_min, skill_min)
    ideal_max = min(spot_max, skill_max)

    if ideal_min > ideal_max:
        ideal_min, ideal_max = skill_min, skill_max

    score = scale_score(wave_ft, ideal_min, ideal_max, 30)

    # A spot's configured skill level is a safety constraint: surfers below
    # that level should not see a high total score driven by ideal wave size.
    skill_rank = {"beginner": 0, "intermediate": 1, "advanced": 2}
    spot_skill_level = row.get("skill_level")
    if (
        spot_skill_level in skill_rank
        and user_skill_level in skill_rank
        and skill_rank[user_skill_level] < skill_rank[spot_skill_level]
    ):
        score = min(score, 10)

    return score


def swell_period_score(row: pd.Series) -> float:
    period = row.get("swell_wave_period")
    ideal_min = row.get("ideal_swell_period_min", 8)
    ideal_max = row.get("ideal_swell_period_max", 15)

    return scale_score(period, float(ideal_min), float(ideal_max), 20)


def wind_speed_score(row: pd.Series, wind_tolerance: str) -> float:
    wind_mph = row.get("wind_speed_mph")

    limits = {
        "low": 8,
        "medium": 14,
        "high": 22,
    }

    max_clean_wind = limits.get(wind_tolerance, 14)

    if pd.isna(wind_mph):
        return 0

    if wind_mph <= max_clean_wind:
        return 20

    if wind_mph <= max_clean_wind + 6:
        return 12

    if wind_mph <= max_clean_wind + 12:
        return 6

    return 2


def wind_direction_score(row: pd.Series) -> float:
    wind_direction = row.get("wind_direction_10m")
    offshore_direction = row.get("offshore_wind_from_degrees")

    if pd.isna(wind_direction) or pd.isna(offshore_direction):
        return 7

    distance = circular_distance_degrees(
        float(wind_direction),
        float(offshore_direction),
    )

    if distance <= 35:
        return 15

    if distance <= 75:
        return 10

    if distance <= 120:
        return 5

    return 1


def water_temp_score(row: pd.Series) -> float:
    temp_c = row.get("sea_surface_temperature")

    if pd.isna(temp_c):
        return 0

    if temp_c >= 20:
        return 5

    if temp_c >= 16:
        return 3

    if temp_c >= 12:
        return 1

    return 0


def time_of_day_score(row: pd.Series, preferred_session_time: str) -> float:
    hour = row["time"].hour

    if preferred_session_time == "morning":
        return 10 if 5 <= hour <= 10 else 3

    if preferred_session_time == "afternoon":
        return 10 if 12 <= hour <= 16 else 3

    if preferred_session_time == "evening":
        return 10 if 16 <= hour <= 20 else 3

    return 7


def get_score_components(
    row: pd.Series,
    user_skill_level: str = "intermediate",
    wind_tolerance: str = "medium",
    preferred_session_time: str = "any",
) -> dict:
    return {
        "wave_score": wave_height_score(row, user_skill_level),
        "swell_score": swell_period_score(row),
        "wind_speed_score": wind_speed_score(row, wind_tolerance),
        "wind_direction_score": wind_direction_score(row),
        "water_temp_score": water_temp_score(row),
        "time_score": time_of_day_score(row, preferred_session_time),
    }


def label_score(score: float) -> str:
    if score >= 85:
        return "Excellent"

    if score >= 70:
        return "Good"

    if score >= 55:
        return "Okay"

    if score >= 40:
        return "Poor"

    return "Bad"


def add_surf_scores(
    df: pd.DataFrame,
    user_skill_level: str = "intermediate",
    wind_tolerance: str = "medium",
    preferred_session_time: str = "any",
) -> pd.DataFrame:
    scored = df.copy()

    scored["wave_height_ft"] = scored["wave_height"] * 3.28084
    scored["swell_wave_height_ft"] = scored["swell_wave_height"] * 3.28084
    scored["wind_speed_mph"] = scored["wind_speed_10m"] * 0.621371

    component_rows = scored.apply(
        lambda row: get_score_components(
            row,
            user_skill_level=user_skill_level,
            wind_tolerance=wind_tolerance,
            preferred_session_time=preferred_session_time,
        ),
        axis=1,
    )

    component_df = pd.DataFrame(component_rows.tolist(), index=scored.index)

    scored = pd.concat([scored, component_df], axis=1)

    scored["surf_score"] = scored[
        [
            "wave_score",
            "swell_score",
            "wind_speed_score",
            "wind_direction_score",
            "water_temp_score",
            "time_score",
        ]
    ].sum(axis=1)

    scored["surf_score"] = scored["surf_score"].clip(0, 100).round(1)
    scored["surf_rating"] = scored["surf_score"].apply(label_score)

    return scored
