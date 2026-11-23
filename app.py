from __future__ import annotations

import numpy as np
import pandas as pd
import pydeck as pdk
import streamlit as st

from src.database import clear_database, load_forecasts_from_db, save_forecasts_to_db
from src.pipeline import build_forecast_dataset

st.set_page_config(
    page_title="SurfIQ Dashboard",
    page_icon="🌊",
    layout="wide",
)

st.title("🌊 SurfIQ: Surf Forecast Dashboard")
st.caption(
    "An API-driven surf analytics dashboard that ranks surf windows using wave, swell, wind, and spot-specific scoring."
)
st.caption("Score guide: 85+ Excellent | 70–84 Good | 55–69 Okay | 40–54 Poor | Below 40 Bad")
st.caption(
    "Tide preferences are stored per spot, but tide is not included in the score "
    "because the current forecast data has no tide observations."
)
st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %I:%M %p')}")


def infer_region(spot_name: str) -> str:
    """
    Adds a region label based on the surf spot name.
    """
    name = spot_name.lower()

    if any(place in name for place in ["bali", "lombok", "uluwatu", "canggu", "padang padang", "desert point"]):
        return "Indonesia"

    if any(place in name for place in ["hawaii", "pipeline", "sunset beach", "honolua", "waikiki"]):
        return "Hawaii"

    if any(place in name for place in ["portugal", "ericeira", "nazaré", "nazare", "peniche", "sagres"]):
        return "Portugal"

    if any(place in name for place in ["morocco", "taghazout", "anchor point", "imsouane", "dakhla"]):
        return "Morocco"

    if any(place in name for place in ["france", "spain", "hossegor", "biarritz", "mundaka", "zarautz"]):
        return "France & Spain"

    if any(place in name for place in ["australia", "bondi", "bells beach", "snapper rocks", "margaret river"]):
        return "Australia"

    if any(place in name for place in ["costa rica", "mexico", "tamarindo", "santa teresa", "pavones", "puerto escondido", "sayulita"]):
        return "Central America & Mexico"

    if any(place in name for place in ["south africa", "jeffreys bay", "muizenberg"]):
        return "South Africa"

    if any(place in name for place in ["japan", "chiba", "shonan"]):
        return "Japan"

    if any(place in name for place in ["rockaway", "long beach, ny", "montauk", "narragansett", "cape cod"]):
        return "Northeast US"

    if any(place in name for place in ["cocoa beach", "sebastian inlet", "wrightsville", "folly beach"]):
        return "Southeast US"

    if any(
        place in name
        for place in [
            "malibu",
            "huntington beach",
            "trestles",
            "ocean beach, san francisco",
            "santa cruz",
            "la jolla",
        ]
    ):
        return "California"

    return "Other"


def get_score_color(score: float) -> list[int]:
    """
    Returns RGBA color for the interactive map.
    """
    if score >= 85:
        return [34, 197, 94, 190]
    if score >= 70:
        return [132, 204, 22, 190]
    if score >= 55:
        return [234, 179, 8, 190]
    if score >= 40:
        return [249, 115, 22, 190]
    return [239, 68, 68, 190]


def spread_nearby_points(
    df: pd.DataFrame,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
    precision: int = 1,
    offset_radius: float = 0.75,
) -> pd.DataFrame:
    """
    Slightly offsets nearby points so overlapping surf spots become visible.

    Original latitude and longitude stay unchanged.
    The map uses plot_lat and plot_lon only for display.
    """
    out = df.copy()

    out["plot_lat"] = out[lat_col]
    out["plot_lon"] = out[lon_col]

    out["cluster_lat"] = out[lat_col].round(precision)
    out["cluster_lon"] = out[lon_col].round(precision)

    for _, idx in out.groupby(["cluster_lat", "cluster_lon"]).groups.items():
        idx = list(idx)
        n = len(idx)

        if n == 1:
            continue

        for i, row_idx in enumerate(idx):
            angle = 2 * np.pi * i / n
            out.at[row_idx, "plot_lat"] = out.at[row_idx, lat_col] + offset_radius * np.sin(angle)
            out.at[row_idx, "plot_lon"] = out.at[row_idx, lon_col] + offset_radius * np.cos(angle)

    return out.drop(columns=["cluster_lat", "cluster_lon"])


def display_score_breakdown(row: pd.Series) -> None:
    """
    Shows how the total surf score was calculated.
    """
    required_cols = [
        "wave_score",
        "swell_score",
        "wind_speed_score",
        "wind_direction_score",
        "water_temp_score",
        "time_score",
    ]

    missing = [col for col in required_cols if col not in row.index]

    if missing:
        st.warning(
            "Score breakdown is unavailable because component score columns are missing. "
            "Make sure `src/scoring.py` is updated."
        )
        return

    breakdown = pd.DataFrame(
        {
            "Component": [
                "Wave Height",
                "Swell Period",
                "Wind Speed",
                "Wind Direction",
                "Water Temperature",
                "Time Preference",
            ],
            "Score": [
                row["wave_score"],
                row["swell_score"],
                row["wind_speed_score"],
                row["wind_direction_score"],
                row["water_temp_score"],
                row["time_score"],
            ],
            "Max Score": [30, 20, 20, 15, 5, 10],
        }
    )

    breakdown["Percent"] = (breakdown["Score"] / breakdown["Max Score"] * 100).round(1)

    st.dataframe(
        breakdown,
        width="stretch",
        hide_index=True,
    )

    st.bar_chart(breakdown.set_index("Component")["Score"])


def display_score_heatmap(df: pd.DataFrame) -> None:
    """
    Shows a spot-by-time score heatmap for the currently filtered forecast window.
    """
    heatmap_df = df.copy()

    top_spots = (
        heatmap_df.groupby("spot")["surf_score"]
        .max()
        .sort_values(ascending=False)
        .head(12)
        .index
    )

    heatmap_df = heatmap_df[heatmap_df["spot"].isin(top_spots)].copy()
    heatmap_df["time_label"] = heatmap_df["time"].dt.strftime("%a %m/%d %I %p")
    heatmap_df["surf_score"] = heatmap_df["surf_score"].round(1)
    time_labels = heatmap_df.sort_values("time")["time_label"].drop_duplicates().tolist()

    st.vega_lite_chart(
        heatmap_df,
        {
            "mark": {"type": "rect", "tooltip": True},
            "encoding": {
                "x": {
                    "field": "time_label",
                    "type": "ordinal",
                    "title": "Forecast Time",
                    "sort": time_labels,
                    "axis": {"labelAngle": -45, "labelOverlap": "greedy"},
                },
                "y": {
                    "field": "spot",
                    "type": "nominal",
                    "title": "Surf Spot",
                    "sort": list(top_spots),
                },
                "color": {
                    "field": "surf_score",
                    "type": "quantitative",
                    "title": "Score",
                    "scale": {
                        "domain": [0, 40, 55, 70, 85, 100],
                        "range": ["#ef4444", "#f97316", "#eab308", "#84cc16", "#22c55e", "#16a34a"],
                    },
                },
                "tooltip": [
                    {"field": "spot", "type": "nominal", "title": "Spot"},
                    {"field": "region", "type": "nominal", "title": "Region"},
                    {"field": "time_label", "type": "ordinal", "title": "Time"},
                    {"field": "surf_score", "type": "quantitative", "title": "Score"},
                    {"field": "surf_rating", "type": "nominal", "title": "Rating"},
                    {"field": "wave_height_ft", "type": "quantitative", "title": "Wave ft", "format": ".1f"},
                    {"field": "swell_wave_period", "type": "quantitative", "title": "Swell sec", "format": ".0f"},
                    {"field": "wind_speed_mph", "type": "quantitative", "title": "Wind mph", "format": ".1f"},
                ],
            },
            "height": max(260, min(520, 34 * len(top_spots))),
        },
        use_container_width=True,
    )


def display_condition_scatter(df: pd.DataFrame) -> None:
    """
    Interactive condition explorer for surf score relationships.
    """
    metric_options = {
        "Wave Height": "wave_height_ft",
        "Swell Period": "swell_wave_period",
        "Wind Speed": "wind_speed_mph",
        "Wind Direction": "wind_direction_10m",
        "Water Temp": "sea_surface_temperature",
        "Surf Score": "surf_score",
    }

    control_cols = st.columns(3)

    with control_cols[0]:
        x_label = st.selectbox(
            "X axis",
            list(metric_options),
            index=0,
            key="visual_x_axis",
        )

    with control_cols[1]:
        y_label = st.selectbox(
            "Y axis",
            list(metric_options),
            index=5,
            key="visual_y_axis",
        )

    with control_cols[2]:
        color_label = st.selectbox(
            "Color",
            ["surf_rating", "region", "spot"],
            index=0,
            key="visual_color",
        )

    x_col = metric_options[x_label]
    y_col = metric_options[y_label]

    scatter_df = df.dropna(subset=[x_col, y_col]).copy()
    scatter_df["time_label"] = scatter_df["time"].dt.strftime("%a %b %d, %I:%M %p")
    scatter_df["point_size"] = scatter_df["surf_score"].clip(12, 100)

    st.vega_lite_chart(
        scatter_df,
        {
            "mark": {"type": "circle", "opacity": 0.72},
            "encoding": {
                "x": {
                    "field": x_col,
                    "type": "quantitative",
                    "title": x_label,
                    "scale": {"zero": False},
                },
                "y": {
                    "field": y_col,
                    "type": "quantitative",
                    "title": y_label,
                    "scale": {"zero": False},
                },
                "color": {"field": color_label, "type": "nominal", "title": color_label.replace("_", " ").title()},
                "size": {
                    "field": "point_size",
                    "type": "quantitative",
                    "title": "Score",
                    "legend": None,
                },
                "tooltip": [
                    {"field": "spot", "type": "nominal", "title": "Spot"},
                    {"field": "region", "type": "nominal", "title": "Region"},
                    {"field": "time_label", "type": "nominal", "title": "Time"},
                    {"field": "surf_score", "type": "quantitative", "title": "Score", "format": ".1f"},
                    {"field": "surf_rating", "type": "nominal", "title": "Rating"},
                    {"field": "wave_height_ft", "type": "quantitative", "title": "Wave ft", "format": ".1f"},
                    {"field": "swell_wave_period", "type": "quantitative", "title": "Swell sec", "format": ".0f"},
                    {"field": "wind_speed_mph", "type": "quantitative", "title": "Wind mph", "format": ".1f"},
                ],
            },
            "height": 440,
        },
        use_container_width=True,
    )


def display_wind_direction_summary(df: pd.DataFrame) -> None:
    """
    Summarizes score and wind speed by compass bucket.
    """
    wind_df = df.dropna(subset=["wind_direction_10m"]).copy()

    if wind_df.empty:
        st.warning("No wind direction data available for the selected filters.")
        return

    labels = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    wind_df["wind_bucket"] = (
        ((wind_df["wind_direction_10m"] + 22.5) // 45).astype(int) % 8
    ).map(dict(enumerate(labels)))

    wind_summary = (
        wind_df.groupby("wind_bucket")
        .agg(
            avg_score=("surf_score", "mean"),
            avg_wind_mph=("wind_speed_mph", "mean"),
            windows=("surf_score", "size"),
        )
        .reindex(labels)
        .dropna()
        .reset_index()
    )

    wind_summary["avg_score"] = wind_summary["avg_score"].round(1)
    wind_summary["avg_wind_mph"] = wind_summary["avg_wind_mph"].round(1)

    st.vega_lite_chart(
        wind_summary,
        {
            "mark": {"type": "bar", "cornerRadiusTopLeft": 2, "cornerRadiusTopRight": 2},
            "encoding": {
                "x": {
                    "field": "wind_bucket",
                    "type": "ordinal",
                    "title": "Wind Direction",
                    "sort": labels,
                },
                "y": {
                    "field": "avg_score",
                    "type": "quantitative",
                    "title": "Average Surf Score",
                    "scale": {"domain": [0, 100]},
                },
                "color": {
                    "field": "avg_score",
                    "type": "quantitative",
                    "title": "Score",
                    "scale": {
                        "domain": [0, 40, 55, 70, 85, 100],
                        "range": ["#ef4444", "#f97316", "#eab308", "#84cc16", "#22c55e", "#16a34a"],
                    },
                },
                "tooltip": [
                    {"field": "wind_bucket", "type": "ordinal", "title": "Wind"},
                    {"field": "avg_score", "type": "quantitative", "title": "Avg Score"},
                    {"field": "avg_wind_mph", "type": "quantitative", "title": "Avg Wind mph"},
                    {"field": "windows", "type": "quantitative", "title": "Windows"},
                ],
            },
            "height": 300,
        },
        use_container_width=True,
    )


def filter_by_session_time(df: pd.DataFrame, preferred_session_time: str) -> pd.DataFrame:
    """
    Actually filters forecast rows based on the selected session time.

    This is different from the scoring function, which only gives a small
    score bonus for the preferred time.
    """
    if preferred_session_time == "morning":
        return df[(df["time"].dt.hour >= 5) & (df["time"].dt.hour <= 11)].copy()

    if preferred_session_time == "afternoon":
        return df[(df["time"].dt.hour >= 12) & (df["time"].dt.hour <= 17)].copy()

    if preferred_session_time == "evening":
        return df[(df["time"].dt.hour >= 17) & (df["time"].dt.hour <= 21)].copy()

    return df.copy()


@st.cache_data(ttl=60 * 60)
def load_live_data(
    forecast_days: int,
    user_skill_level: str,
    wind_tolerance: str,
    preferred_session_time: str,
) -> pd.DataFrame:
    return build_forecast_dataset(
        forecast_days=forecast_days,
        user_skill_level=user_skill_level,
        wind_tolerance=wind_tolerance,
        preferred_session_time=preferred_session_time,
    )


with st.sidebar:
    st.header("Surfer Profile")

    user_skill_level = st.selectbox(
        "Skill level",
        ["beginner", "intermediate", "advanced"],
        index=1,
    )

    wind_tolerance = st.selectbox(
        "Wind tolerance",
        ["low", "medium", "high"],
        index=1,
    )

    preferred_session_time = st.selectbox(
        "Preferred session time",
        ["any", "morning", "afternoon", "evening"],
        index=0,
    )

    st.markdown("---")

    st.header("Forecast Controls")

    forecast_days = st.slider(
        "Forecast days",
        min_value=1,
        max_value=7,
        value=2,
    )

    refresh = st.button("Refresh API Data")

    st.markdown("---")

    st.header("Database Controls")

    save_to_db = st.button("Save Current Forecast to SQLite")
    view_database = st.checkbox("View historical database records")
    wipe_db = st.button("Clear SQLite Database")

    st.markdown("---")
    st.write("Edit `config/spots.yaml` to add surf spots or adjust scoring rules.")


if refresh:
    st.cache_data.clear()

if wipe_db:
    clear_database()
    st.success("SQLite database cleared.")


try:
    df = load_live_data(
        forecast_days=forecast_days,
        user_skill_level=user_skill_level,
        wind_tolerance=wind_tolerance,
        preferred_session_time=preferred_session_time,
    )
except Exception as exc:
    st.error("Could not load forecast data. Check your internet connection and spot coordinates.")
    st.exception(exc)
    st.stop()


if "region" not in df.columns:
    df["region"] = df["spot"].apply(infer_region)
else:
    df["region"] = df["region"].fillna(df["spot"].apply(infer_region))


if save_to_db:
    save_forecasts_to_db(df)
    st.success("Current forecast saved to SQLite.")


if view_database:
    st.subheader("Historical Forecast Records from SQLite")

    db_df = load_forecasts_from_db()

    if db_df.empty:
        st.info("No database records yet. Click `Save Current Forecast to SQLite` first.")
    else:
        st.write(f"Database contains {len(db_df):,} saved forecast rows.")
        st.dataframe(
            db_df.sort_values("created_at", ascending=False).head(500),
            width="stretch",
            hide_index=True,
        )

    st.markdown("---")


regions = sorted(df["region"].dropna().unique())

default_regions = [
    region
    for region in ["Northeast US", "California", "Hawaii", "Portugal", "Morocco", "Indonesia"]
    if region in regions
]

if not default_regions:
    default_regions = regions[:3]

selected_regions = st.sidebar.multiselect(
    "Regions",
    regions,
    default=default_regions,
)

df = df[df["region"].isin(selected_regions)].copy()

if df.empty:
    st.warning("Select at least one region.")
    st.stop()


spots = sorted(df["spot"].unique())

selected_spots = st.sidebar.multiselect(
    "Surf spots",
    spots,
    default=spots[:6],
)

df = df[df["spot"].isin(selected_spots)].copy()

if df.empty:
    st.warning("Select at least one surf spot.")
    st.stop()


min_time = df["time"].min().to_pydatetime()
max_time = df["time"].max().to_pydatetime()

selected_range = st.sidebar.slider(
    "Forecast window",
    min_value=min_time,
    max_value=max_time,
    value=(min_time, max_time),
    format="MM/DD HH:mm",
)

df_window = df[
    (df["time"] >= selected_range[0])
    & (df["time"] <= selected_range[1])
].copy()

df_window = filter_by_session_time(df_window, preferred_session_time)

if df_window.empty:
    st.warning(
        f"No records available for the selected filters and preferred session time: {preferred_session_time}."
    )
    st.stop()


best_row = df_window.sort_values("surf_score", ascending=False).iloc[0]

st.subheader(
    f"Best overall window: {best_row['spot']} at {best_row['time'].strftime('%A, %b %d at %I:%M %p')}"
)

metric_cols = st.columns(6)

metric_cols[0].metric("Best Region", best_row["region"])
metric_cols[1].metric("Surf Score", f"{best_row['surf_score']:.0f}/100")
metric_cols[2].metric("Rating", best_row["surf_rating"])
metric_cols[3].metric("Wave Height", f"{best_row['wave_height_ft']:.1f} ft")
metric_cols[4].metric("Wind Speed", f"{best_row['wind_speed_mph']:.1f} mph")

if "board_recommendation" in best_row.index:
    metric_cols[5].metric("Board", best_row["board_recommendation"])
else:
    metric_cols[5].metric("Board", "N/A")

st.info(best_row["recommendation_reason"])

with st.expander("See why this score is high or low"):
    display_score_breakdown(best_row)

st.markdown("---")


tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Overview",
        "Visual Explorer",
        "Interactive Map",
        "Compare Spots",
        "Spot Detail",
    ]
)


with tab1:
    st.subheader("Top 5 Surf Windows")

    overview_cols = [
        "spot",
        "region",
        "time",
        "surf_score",
        "surf_rating",
        "wave_height_ft",
        "swell_wave_period",
        "wind_speed_mph",
    ]

    if "board_recommendation" in df_window.columns:
        overview_cols.append("board_recommendation")

    top_5 = df_window.sort_values("surf_score", ascending=False).head(5)[overview_cols].copy()

    top_5["time"] = top_5["time"].dt.strftime("%a %b %d, %I:%M %p")
    top_5["wave_height_ft"] = top_5["wave_height_ft"].round(1)
    top_5["wind_speed_mph"] = top_5["wind_speed_mph"].round(1)

    top_5 = top_5.rename(
        columns={
            "spot": "Spot",
            "region": "Region",
            "time": "Best Time",
            "surf_score": "Score",
            "surf_rating": "Rating",
            "wave_height_ft": "Wave ft",
            "swell_wave_period": "Swell sec",
            "wind_speed_mph": "Wind mph",
            "board_recommendation": "Recommended Board",
        }
    )

    st.dataframe(
        top_5,
        width="stretch",
        hide_index=True,
    )

    st.markdown("---")

    st.subheader("Surf Score Heatmap")
    display_score_heatmap(df_window)

    st.markdown("---")

    left, right = st.columns([2, 1])

    with left:
        st.subheader("Hourly Surf Score")

        chart_df = df_window.pivot_table(
            index="time",
            columns="spot",
            values="surf_score",
            aggfunc="mean",
        )

        st.line_chart(chart_df)

    with right:
        st.subheader("Spot Comparison")

        summary = (
            df_window.groupby(["spot", "region"])
            .agg(
                avg_score=("surf_score", "mean"),
                max_score=("surf_score", "max"),
                avg_wave_ft=("wave_height_ft", "mean"),
                avg_swell_period=("swell_wave_period", "mean"),
                avg_wind_mph=("wind_speed_mph", "mean"),
            )
            .reset_index()
            .sort_values("max_score", ascending=False)
        )

        summary = summary.round(
            {
                "avg_score": 1,
                "max_score": 1,
                "avg_wave_ft": 1,
                "avg_swell_period": 1,
                "avg_wind_mph": 1,
            }
        )

        summary = summary.rename(
            columns={
                "spot": "Spot",
                "region": "Region",
                "avg_score": "Avg Score",
                "max_score": "Best Score",
                "avg_wave_ft": "Avg Wave ft",
                "avg_swell_period": "Avg Swell sec",
                "avg_wind_mph": "Avg Wind mph",
            }
        )

        st.dataframe(
            summary,
            width="stretch",
            hide_index=True,
        )


with tab2:
    st.subheader("Condition Explorer")
    display_condition_scatter(df_window)

    st.markdown("---")
    st.subheader("Wind Direction Impact")
    display_wind_direction_summary(df_window)


with tab3:
    st.subheader("Interactive Surf Score Map")
    st.caption(
        "Each point shows the best forecasted surf window for that spot. "
        "Nearby surf spots are slightly offset so overlapping locations are easier to read."
    )

    map_df = (
        df_window.sort_values("surf_score", ascending=False)
        .groupby("spot")
        .head(1)
        .copy()
    )

    if map_df.empty:
        st.warning("No map data available for the selected filters.")
    else:
        map_df["latitude"] = pd.to_numeric(map_df["latitude"], errors="coerce")
        map_df["longitude"] = pd.to_numeric(map_df["longitude"], errors="coerce")
        map_df = map_df.dropna(subset=["latitude", "longitude"])

        if map_df.empty:
            st.warning("No valid latitude/longitude data available for the selected spots.")
        else:
            map_df = spread_nearby_points(map_df, offset_radius=0.05)

            map_df["color"] = map_df["surf_score"].apply(get_score_color)

            # Fixed small radius keeps nearby spots readable.
            # Color shows the score, so size does not need to.
            map_df["radius"] = 12000

            midpoint_lat = map_df["plot_lat"].mean()
            midpoint_lon = map_df["plot_lon"].mean()

            lat_range = map_df["plot_lat"].max() - map_df["plot_lat"].min()
            lon_range = map_df["plot_lon"].max() - map_df["plot_lon"].min()
            max_range = max(lat_range, lon_range)

            if max_range > 120:
                zoom_level = 1
            elif max_range > 60:
                zoom_level = 1.5
            elif max_range > 25:
                zoom_level = 2.3
            elif max_range > 10:
                zoom_level = 3.2
            else:
                zoom_level = 4.5

            bubble_layer = pdk.Layer(
                "ScatterplotLayer",
                data=map_df,
                get_position="[plot_lon, plot_lat]",
                get_radius="radius",
                get_fill_color="color",
                pickable=True,
                opacity=0.85,
                stroked=True,
                filled=True,
                line_width_min_pixels=2,
                get_line_color=[255, 255, 255],
            )

            tooltip = {
                "html": """
                <div style="font-family: Arial; padding: 6px;">
                    <b>{spot}</b><br/>
                    Region: {region}<br/>
                    Score: {surf_score}/100<br/>
                    Rating: {surf_rating}<br/>
                    Wave: {wave_height_ft} ft<br/>
                    Swell: {swell_wave_period} sec<br/>
                    Wind: {wind_speed_mph} mph<br/>
                    Board: {board_recommendation}
                </div>
                """,
                "style": {
                    "backgroundColor": "black",
                    "color": "white",
                },
            }

            view_state = pdk.ViewState(
                latitude=midpoint_lat,
                longitude=midpoint_lon,
                zoom=zoom_level,
                pitch=0,
            )

            st.pydeck_chart(
                pdk.Deck(
                    layers=[bubble_layer],
                    initial_view_state=view_state,
                    tooltip=tooltip,
                    map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
                )
            )

            st.caption(
                "Nearby points are slightly separated for readability. Hover over a point to see the exact spot name and forecast details."
            )

            st.markdown(
                """
                **Score Legend:**  
                🟢 Excellent: 85+ &nbsp;&nbsp; 🟡 Good/Okay: 55–84 &nbsp;&nbsp; 🟠 Poor: 40–54 &nbsp;&nbsp; 🔴 Bad: Below 40
                """
            )

            st.markdown("---")
            st.subheader("Best Surf Window by Location")

            map_cols = [
                "spot",
                "region",
                "time",
                "surf_score",
                "surf_rating",
                "wave_height_ft",
                "swell_wave_period",
                "wind_speed_mph",
            ]

            if "board_recommendation" in map_df.columns:
                map_cols.append("board_recommendation")

            map_table = map_df[map_cols].copy()

            map_table["time"] = map_table["time"].dt.strftime("%a %b %d, %I:%M %p")
            map_table["wave_height_ft"] = map_table["wave_height_ft"].round(1)
            map_table["wind_speed_mph"] = map_table["wind_speed_mph"].round(1)
            map_table = map_table.sort_values("surf_score", ascending=False)

            st.dataframe(
                map_table.rename(
                    columns={
                        "spot": "Spot",
                        "region": "Region",
                        "time": "Best Time",
                        "surf_score": "Best Score",
                        "surf_rating": "Rating",
                        "wave_height_ft": "Wave ft",
                        "swell_wave_period": "Swell sec",
                        "wind_speed_mph": "Wind mph",
                        "board_recommendation": "Recommended Board",
                    }
                ),
                width="stretch",
                hide_index=True,
            )


with tab4:
    st.subheader("Compare Two Surf Spots")

    available_compare_spots = sorted(df_window["spot"].unique())

    if len(available_compare_spots) < 2:
        st.warning("Select at least two surf spots to compare.")
    else:
        compare_cols = st.columns(2)

        with compare_cols[0]:
            spot_a = st.selectbox(
                "First spot",
                available_compare_spots,
                index=0,
            )

        with compare_cols[1]:
            spot_b = st.selectbox(
                "Second spot",
                available_compare_spots,
                index=1,
            )

        compare_df = df_window[df_window["spot"].isin([spot_a, spot_b])].copy()

        if compare_df.empty:
            st.warning("No comparison data available for the selected spots.")
        else:
            comparison_summary = (
                compare_df.groupby("spot")
                .agg(
                    avg_score=("surf_score", "mean"),
                    best_score=("surf_score", "max"),
                    avg_wave_ft=("wave_height_ft", "mean"),
                    avg_swell_sec=("swell_wave_period", "mean"),
                    avg_wind_mph=("wind_speed_mph", "mean"),
                )
                .reset_index()
                .round(1)
            )

            st.dataframe(
                comparison_summary.rename(
                    columns={
                        "spot": "Spot",
                        "avg_score": "Average Score",
                        "best_score": "Best Score",
                        "avg_wave_ft": "Average Wave ft",
                        "avg_swell_sec": "Average Swell sec",
                        "avg_wind_mph": "Average Wind mph",
                    }
                ),
                width="stretch",
                hide_index=True,
            )

            st.subheader("Score Trend Comparison")

            compare_chart = compare_df.pivot_table(
                index="time",
                columns="spot",
                values="surf_score",
                aggfunc="mean",
            )

            st.line_chart(compare_chart)

            st.subheader("Best Window for Each Spot")

            best_each = (
                compare_df.sort_values("surf_score", ascending=False)
                .groupby("spot")
                .head(1)
                .copy()
            )

            if best_each.empty:
                st.warning("No best windows available for the selected comparison.")
            else:
                for _, row in best_each.iterrows():
                    st.markdown(f"### {row['spot']}")
                    st.write(f"**Best time:** {row['time'].strftime('%A, %b %d at %I:%M %p')}")
                    st.write(f"**Score:** {row['surf_score']:.0f}/100 — {row['surf_rating']}")

                    if "board_recommendation" in row.index:
                        st.write(f"**Recommended board:** {row['board_recommendation']}")

                    st.write(row["recommendation_reason"])

                    with st.expander(f"See score breakdown for {row['spot']}"):
                        display_score_breakdown(row)


with tab5:
    st.subheader("Spot Detail")

    available_spots = sorted(df_window["spot"].unique())

    if not available_spots:
        st.warning("No surf spots available for the selected filters and time window.")
    else:
        selected_detail_spot = st.selectbox(
            "Detailed spot view",
            available_spots,
        )

        detail = df_window[df_window["spot"] == selected_detail_spot].copy()

        if detail.empty:
            st.warning("No forecast data available for this spot in the selected time window.")
        else:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader(f"Wave Height: {selected_detail_spot}")
                st.line_chart(detail.set_index("time")[["wave_height_ft"]])

            with col2:
                st.subheader(f"Wind Speed: {selected_detail_spot}")
                st.line_chart(detail.set_index("time")[["wind_speed_mph"]])

            col3, col4 = st.columns(2)

            with col3:
                st.subheader(f"Swell Period: {selected_detail_spot}")
                st.line_chart(detail.set_index("time")[["swell_wave_period"]])

            with col4:
                st.subheader(f"Surf Score: {selected_detail_spot}")
                st.line_chart(detail.set_index("time")[["surf_score"]])

            best_detail = detail.sort_values("surf_score", ascending=False).iloc[0]

            st.markdown("---")
            st.subheader("Best Window Breakdown for This Spot")
            st.write(f"**Best time:** {best_detail['time'].strftime('%A, %b %d at %I:%M %p')}")
            st.write(f"**Score:** {best_detail['surf_score']:.0f}/100 — {best_detail['surf_rating']}")

            if "board_recommendation" in best_detail.index:
                st.write(f"**Recommended board:** {best_detail['board_recommendation']}")

            st.write(best_detail["recommendation_reason"])

            with st.expander("See score breakdown"):
                display_score_breakdown(best_detail)
