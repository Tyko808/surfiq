SurfIQ is an interactive surf forecasting dashboard that helps users decide where and when to surf. The app collects live marine and weather forecast data for global surf spots and ranks surf windows using a custom scoring algorithm based on wave height, swell period, wind speed, wind direction, water temperature, and user preferences.

## Features

- Live marine and weather forecast data from the Open-Meteo API
- Custom surf score from 0 to 100
- Region and surf spot filtering
- Personalized surfer profile settings
- Score breakdown explanations
- Surfboard recommendations
- Interactive surf score map
- Two-spot comparison tool
- Alert candidates for high-quality surf windows
- SQLite storage for saved forecast snapshots
- Raw forecast data export

## How It Works

The dashboard pulls hourly marine and weather forecast data for each surf spot. It then calculates a surf score using multiple factors, including wave height, swell period, wind speed, wind direction, water temperature, surfer skill level, wind tolerance, and preferred session time.

Higher scores indicate better surf conditions. The app also explains why each surf window ranks well and recommends a board type based on the forecasted wave conditions.