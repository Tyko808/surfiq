from __future__ import annotations

import pandas as pd
import pydeck as pdk
import streamlit as st

from src.database import clear_database, load_forecasts_from_db, save_forecasts_to_db
from src.pipeline import build_forecast_dataset
from src.recommendations import get_top_recommendations

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
st.caption(f"Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %I:%M %p')}")


def infer_region(spot_name: str) -> str:
    """
    Adds a simple region label based on the spot name.
    Uses specific checks so spots like Canggu do not get mislabeled as California.
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
        return [34, 197, 94, 180]  # green
    if score >= 70:
        return [132, 204, 22, 180]  # light green
    if score >= 55:
        return [234, 179, 8, 180]  # yellow
    if score >= 40:
        return [249, 115, 22, 180]  # orange
    return [239, 68, 68, 180]  # red


def display_score_breakdown(row: pd.Series) -> None:
    """
    Shows how the total surf score was calculated.
    Requires the upgraded scoring.py that creates component score columns.
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
            "Score breakdown is unavailable because the component score columns are missing. "
            "Make sure you replaced `src/scoring.py` with the upgraded version."
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
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(breakdown.set_index("Component")["Score"])


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
            use_container_width=True,
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

if df_window.empty:
    st.warning("No records available for the selected time range.")
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


tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
    [
        "Overview",
        "Recommendations",
        "Interactive Map",
        "Compare Spots",
        "Alerts",
        "Spot Detail",
        "Raw Data",
        "Methodology",
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
        use_container_width=True,
        hide_index=True,
    )

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
            use_container_width=True,
            hide_index=True,
        )


with tab2:
    st.subheader("Top Recommended Surf Windows")

    top_recs = get_top_recommendations(df_window, n=15)

    rec_cols = [
        "spot",
        "region",
        "time",
        "surf_score",
        "surf_rating",
        "wave_height_ft",
        "swell_wave_period",
        "wind_speed_mph",
    ]

    if "board_recommendation" in top_recs.columns:
        rec_cols.append("board_recommendation")

    rec_cols.append("recommendation_reason")

    rec_table = top_recs[rec_cols].copy()

    rec_table["time"] = rec_table["time"].dt.strftime("%a %b %d, %I:%M %p")
    rec_table["wave_height_ft"] = rec_table["wave_height_ft"].round(1)
    rec_table["wind_speed_mph"] = rec_table["wind_speed_mph"].round(1)

    rec_table = rec_table.rename(
        columns={
            "spot": "Spot",
            "region": "Region",
            "time": "Time",
            "surf_score": "Score",
            "surf_rating": "Rating",
            "wave_height_ft": "Wave ft",
            "swell_wave_period": "Swell sec",
            "wind_speed_mph": "Wind mph",
            "board_recommendation": "Recommended Board",
            "recommendation_reason": "Why It Ranks Well",
        }
    )

    st.dataframe(
        rec_table,
        use_container_width=True,
        hide_index=True,
    )


with tab3:
    st.subheader("Interactive Surf Score Map")

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

        map_df["color"] = map_df["surf_score"].apply(get_score_color)

        # Large enough to see, but not too massive.
        map_df["radius"] = map_df["surf_score"] * 12000

        midpoint_lat = map_df["latitude"].mean()
        midpoint_lon = map_df["longitude"].mean()

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_df,
            get_position="[longitude, latitude]",
            get_radius="radius",
            get_fill_color="color",
            pickable=True,
            opacity=0.75,
            stroked=True,
            filled=True,
            line_width_min_pixels=2,
            get_line_color=[255, 255, 255],
        )

        tooltip = {
            "html": """
            <b>{spot}</b><br/>
            Region: {region}<br/>
            Score: {surf_score}/100<br/>
            Rating: {surf_rating}<br/>
            Wave: {wave_height_ft} ft<br/>
            Wind: {wind_speed_mph} mph
            """,
            "style": {
                "backgroundColor": "black",
                "color": "white",
            },
        }

        view_state = pdk.ViewState(
            latitude=midpoint_lat,
            longitude=midpoint_lon,
            zoom=1,
            pitch=0,
        )

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
            )
        )

        st.caption(
            "Bubble size represents surf score. Green means excellent conditions, yellow/orange means mixed, and red means poor."
        )

        map_cols = [
            "spot",
            "region",
            "time",
            "surf_score",
            "surf_rating",
            "wave_height_ft",
            "wind_speed_mph",
        ]

        if "board_recommendation" in map_df.columns:
            map_cols.append("board_recommendation")

        map_table = map_df[map_cols].copy()

        map_table["time"] = map_table["time"].dt.strftime("%a %b %d, %I:%M %p")
        map_table["wave_height_ft"] = map_table["wave_height_ft"].round(1)
        map_table["wind_speed_mph"] = map_table["wind_speed_mph"].round(1)

        st.dataframe(
            map_table.rename(
                columns={
                    "spot": "Spot",
                    "region": "Region",
                    "time": "Best Time",
                    "surf_score": "Best Score",
                    "surf_rating": "Rating",
                    "wave_height_ft": "Wave ft",
                    "wind_speed_mph": "Wind mph",
                    "board_recommendation": "Recommended Board",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


with tab4:
    st.subheader("Compare Two Surf Spots")

    compare_cols = st.columns(2)

    with compare_cols[0]:
        spot_a = st.selectbox(
            "First spot",
            spots,
            index=0,
        )

    with compare_cols[1]:
        default_index = 1 if len(spots) > 1 else 0
        spot_b = st.selectbox(
            "Second spot",
            spots,
            index=default_index,
        )

    compare_df = df_window[df_window["spot"].isin([spot_a, spot_b])].copy()

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
        use_container_width=True,
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
    st.subheader("Surf Alert Candidates")

    st.write(
        "Use this section to find upcoming windows that pass your personal surf threshold."
    )

    alert_cols = st.columns(3)

    with alert_cols[0]:
        alert_score = st.slider(
            "Minimum surf score",
            min_value=40,
            max_value=100,
            value=80,
        )

    with alert_cols[1]:
        alert_rating = st.selectbox(
            "Minimum rating",
            ["Any", "Okay", "Good", "Excellent"],
            index=0,
        )

    with alert_cols[2]:
        max_alerts = st.slider(
            "Number of alerts to show",
            min_value=5,
            max_value=30,
            value=10,
        )

    alert_df = df_window[df_window["surf_score"] >= alert_score].copy()

    rating_order = {
        "Bad": 0,
        "Poor": 1,
        "Okay": 2,
        "Good": 3,
        "Excellent": 4,
    }

    if alert_rating != "Any":
        min_rating_value = rating_order[alert_rating]
        alert_df = alert_df[
            alert_df["surf_rating"].map(rating_order) >= min_rating_value
        ]

    alert_df = alert_df.sort_values("surf_score", ascending=False).head(max_alerts)

    if alert_df.empty:
        st.warning("No surf windows match your alert settings.")
    else:
        alert_cols_to_show = [
            "spot",
            "region",
            "time",
            "surf_score",
            "surf_rating",
            "wave_height_ft",
            "swell_wave_period",
            "wind_speed_mph",
        ]

        if "board_recommendation" in alert_df.columns:
            alert_cols_to_show.append("board_recommendation")

        alert_cols_to_show.append("recommendation_reason")

        alert_table = alert_df[alert_cols_to_show].copy()

        alert_table["time"] = alert_table["time"].dt.strftime("%a %b %d, %I:%M %p")
        alert_table["wave_height_ft"] = alert_table["wave_height_ft"].round(1)
        alert_table["wind_speed_mph"] = alert_table["wind_speed_mph"].round(1)

        st.dataframe(
            alert_table.rename(
                columns={
                    "spot": "Spot",
                    "region": "Region",
                    "time": "Time",
                    "surf_score": "Score",
                    "surf_rating": "Rating",
                    "wave_height_ft": "Wave ft",
                    "swell_wave_period": "Swell sec",
                    "wind_speed_mph": "Wind mph",
                    "board_recommendation": "Recommended Board",
                    "recommendation_reason": "Reason",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


with tab6:
    st.subheader("Spot Detail")

    selected_detail_spot = st.selectbox(
        "Detailed spot view",
        spots,
    )

    detail = df_window[df_window["spot"] == selected_detail_spot].copy()

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


with tab7:
    st.subheader("Raw Forecast Data")

    raw_cols = [
        "spot",
        "region",
        "time",
        "surf_score",
        "surf_rating",
        "wave_height_ft",
        "swell_wave_period",
        "wind_speed_mph",
        "wind_direction_10m",
        "sea_surface_temperature",
    ]

    if "board_recommendation" in df_window.columns:
        raw_cols.append("board_recommendation")

    raw_cols.append("recommendation_reason")

    raw = df_window[raw_cols].copy()
    raw["time"] = raw["time"].dt.strftime("%Y-%m-%d %H:%M")

    raw = raw.rename(
        columns={
            "spot": "Spot",
            "region": "Region",
            "time": "Time",
            "surf_score": "Score",
            "surf_rating": "Rating",
            "wave_height_ft": "Wave ft",
            "swell_wave_period": "Swell sec",
            "wind_speed_mph": "Wind mph",
            "wind_direction_10m": "Wind Direction",
            "sea_surface_temperature": "Sea Temp C",
            "board_recommendation": "Recommended Board",
            "recommendation_reason": "Recommendation Reason",
        }
    )

    st.dataframe(
        raw,
        use_container_width=True,
        hide_index=True,
    )

    csv = df_window.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download filtered forecast CSV",
        data=csv,
        file_name="surfiq_forecast.csv",
        mime="text/csv",
    )


with tab8:
    st.subheader("Methodology")

    st.markdown(
        """
        **SurfIQ** ranks surf windows by combining live marine and weather forecast data with custom scoring logic.

        **Data sources**
        - Marine forecast data: wave height, wave direction, wave period, swell height, swell direction, swell period, and sea surface temperature.
        - Weather forecast data: wind speed, wind direction, air temperature, and precipitation probability.

        **Scoring logic**
        The surf score is calculated from several features:
        - **Wave height:** compares forecasted wave size to the ideal range for each surf spot and surfer skill level.
        - **Swell period:** rewards longer and more organized swell.
        - **Wind speed:** rewards cleaner conditions with lighter wind.
        - **Wind direction:** rewards wind that is closer to the estimated offshore direction for each spot.
        - **Water temperature:** adds a small comfort bonus.
        - **Preferred session time:** adjusts scores based on whether the user prefers morning, afternoon, evening, or any time.

        **Interactive features**
        - Region and surf spot filters
        - Personalized surfer profile settings
        - Score breakdown explanations
        - Board recommendations
        - Interactive map with score-based bubbles
        - Two-spot comparison tool
        - Alert candidates for high-quality surf windows
        - Raw data download

        **Storage**
        The app can save forecast snapshots into a local SQLite database. This allows the project to support historical forecast tracking and future trend analysis.

        **Limitations**
        - Offshore wind directions are estimates.
        - Tide scoring is not fully implemented yet.
        - The model is rule-based, not trained on real surf-session ratings yet.
        - Forecast quality depends on the external API data.
        """
    )