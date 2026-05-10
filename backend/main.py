"""
Mise FastAPI backend.
All demand adjustments come from the trained LightGBM model.
The backend maps user inputs → feature vectors → model predictions.
"""
import os
import pickle
from contextlib import asynccontextmanager
from datetime import date, timedelta
from typing import Literal, Optional

import numpy as np
import pandas as pd
import requests
import google.generativeai as genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "ml", "model.pkl")
DATA_PATH = os.path.join(ROOT, "data", "sales_history.csv")

PORTUGUESE_HOLIDAYS = {
    date(2024, 1, 1), date(2024, 3, 29), date(2024, 4, 25), date(2024, 5, 1),
    date(2024, 6, 10), date(2024, 6, 13), date(2024, 8, 15), date(2024, 10, 5),
    date(2024, 11, 1), date(2024, 12, 1), date(2024, 12, 8), date(2024, 12, 25),
    date(2025, 1, 1), date(2025, 4, 18), date(2025, 4, 25), date(2025, 5, 1),
    date(2025, 6, 10), date(2025, 6, 19), date(2025, 8, 15), date(2025, 10, 5),
    date(2025, 11, 1), date(2025, 12, 1), date(2025, 12, 8), date(2025, 12, 25),
    date(2026, 1, 1), date(2026, 4, 3), date(2026, 4, 25), date(2026, 5, 1),
    date(2026, 6, 10), date(2026, 8, 15), date(2026, 10, 5),
    date(2026, 11, 1), date(2026, 12, 1), date(2026, 12, 8), date(2026, 12, 25),
}

TEMPERATURE_MAP: dict[str, float] = {
    "Cool": 12.0,
    "Mild": 19.0,
    "Warm": 28.0,
}

ITEM_PRICES = {
    "Margherita Pizza": 14.0,
    "Pasta Carbonara": 16.0,
    "Grilled Salmon": 22.0,
    "Caesar Salad": 12.0,
    "Tiramisu": 7.0,
    "Risotto": 17.0,
    "Bruschetta": 9.0,
    "Grilled Chicken": 18.0,
}

artifact: dict | None = None
sales_df: pd.DataFrame | None = None


def get_lisbon_weather() -> dict:
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": 38.7,
            "longitude": -9.14,
            "daily": ["temperature_2m_max", "precipitation_sum"],
            "timezone": "Europe/Lisbon",
            "forecast_days": 7
        }
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        daily_temps = data["daily"]["temperature_2m_max"]
        daily_rainfall = data["daily"]["precipitation_sum"]
        avg_temp = sum(daily_temps) / 7
        total_rain = sum(daily_rainfall)
        return {
            "daily_temps": daily_temps,
            "daily_rainfall": daily_rainfall,
            "avg_temp": round(avg_temp, 1),
            "total_rain": round(total_rain, 1),
            "temperature_level": "Warm" if avg_temp > 24 else ("Cool" if avg_temp < 15 else "Mild"),
            "rainy_days": sum(1 for r in daily_rainfall if r > 2.0)
        }
    except Exception:
        return {
            "daily_temps": [19.0] * 7,
            "daily_rainfall": [1.0] * 7,
            "avg_temp": 19.0,
            "total_rain": 7.0,
            "temperature_level": "Mild",
            "rainy_days": 0
        }


def get_ai_insights(recommendations: list, weather: dict, restaurant_name: str, owner_notes: str = "") -> str:
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")

        items_summary = "\n".join([
            f"- {r.menu_item}: {r.predicted_demand} units predicted, {r.vs_last_week} vs last week, confidence: {r.confidence}, flagged: {r.flag}"
            for r in recommendations
        ])

        daily_weather = ", ".join([
            f"Day {i+1}: {weather['daily_temps'][i]:.1f}C, {weather['daily_rainfall'][i]:.1f}mm rain"
            for i in range(7)
        ])

        notes_section = f"\nRestaurant owner notes: {owner_notes}" if owner_notes else ""

        prompt = f"""
You are an expert restaurant demand analyst for {restaurant_name}.

This week's forecast data:
{items_summary}

Real weather this week (day by day):
{daily_weather}

Average: {weather['avg_temp']}C, {weather['rainy_days']} rainy days.{notes_section}

Your job is to reason across ALL these signals simultaneously — weather patterns, demand changes vs last week, confidence levels, flagged items, and any owner notes — and produce a single confident recommendation paragraph.

If the owner has provided notes, factor them into your reasoning and adjust your recommendations accordingly. For example if they mention a private event on a specific day, account for that in your order recommendations.

Do NOT just narrate the numbers. Cross-reference the signals and explain WHY the pattern is happening and WHAT specific action the restaurant should take. Include which specific days to front-load or reduce orders. The output should read like advice from an expert who understands both the data and the restaurant business — not a template.

Be specific, be direct, maximum 4 sentences.
"""
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"AI insights unavailable: {str(e)}"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global artifact, sales_df
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"model.pkl not found at {MODEL_PATH}. Run ml/train.py first.")
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)
    if os.path.exists(DATA_PATH):
        sales_df = pd.read_csv(DATA_PATH, parse_dates=["date"])
    print("Model and data loaded.")
    yield


app = FastAPI(title="Mise Forecast API", version="2.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ForecastRequest(BaseModel):
    restaurant_name: str = "Da Mario"
    cuisine: str = "Italian"
    location: str = "Lisbon"
    owner_notes: str = ""
    seating_capacity: Optional[int] = 60
    upcoming_events: bool = False
    is_holiday_week: bool = False
    is_tourist_season: bool = False
    forecast_month: Optional[int] = None


class ItemForecast(BaseModel):
    menu_item: str
    predicted_demand: int
    recommended_order: int
    vs_last_week: str
    reasoning: str
    confidence: str
    flag: bool
    daily_predictions: list[int] = []

class ForecastResponse(BaseModel):
    restaurant: str
    week: str
    recommendations: list[ItemForecast]
    total_estimated_waste_saved: str
    without_mise_waste: float = 0.0
    with_mise_waste: float = 0.0
    model_accuracy: str
    ai_insights: str = ""
    daily_temps: list[float] = []
    daily_rainfall: list[float] = []


def build_feature_row(
    target_date: date,
    rainfall_mm: float,
    temperature_c: float,
    local_event: int,
    is_holiday: int,
    is_tourist_season: int,
    rolling_7day_avg: float,
    lag_7: float,
    lag_14: float,
    lag_28: float,
    month_override: Optional[int] = None,
) -> dict:
    month = month_override if month_override is not None else target_date.month
    return {
        "day_of_week": target_date.weekday(),
        "week_of_year": target_date.isocalendar()[1],
        "month": month,
        "is_weekend": int(target_date.weekday() >= 5),
        "is_holiday": is_holiday,
        "is_tourist_season": is_tourist_season,
        "rainfall_mm": rainfall_mm,
        "temperature_c": temperature_c,
        "local_event": local_event,
        "rolling_7day_avg": rolling_7day_avg,
        "lag_7": lag_7,
        "lag_14": lag_14,
        "lag_28": lag_28,
    }


def get_lag_values(item: str, target_date: date) -> tuple[float, float, float]:
    if sales_df is None or artifact is None:
        return 20.0, 20.0, 20.0
    stats = artifact["item_stats"].get(item, {})
    last_35 = stats.get("last_35_sales", [])
    if not last_35:
        fallback = stats.get("last_7_avg", 20.0)
        return fallback, fallback, fallback
    sales_lookup = {
        pd.Timestamp(r["date"]).date(): r["units_sold"]
        for r in last_35
    }
    last_date = max(sales_lookup.keys())

    def get_lag(n: int) -> float:
        candidates = [
            target_date - timedelta(days=n),
            last_date - timedelta(days=(n - 1)),
            last_date,
        ]
        for d in candidates:
            if d in sales_lookup:
                return float(sales_lookup[d])
        return stats.get("last_7_avg", 20.0)

    return get_lag(7), get_lag(14), get_lag(28)


def get_rolling_avg(item: str) -> float:
    if artifact is None:
        return 20.0
    stats = artifact["item_stats"].get(item, {})
    return stats.get("last_7_avg", stats.get("mean", 20.0))


def build_reasoning(req: ForecastRequest, week_start: date) -> str:
    parts = []
    if req.upcoming_events:
        parts.append("Local event")
    if req.is_holiday_week:
        parts.append("Holiday week")
    if req.is_tourist_season:
        parts.append("Tourist season")
    if week_start.weekday() >= 5 or (week_start + timedelta(days=5)).weekday() >= 5:
        parts.append("Weekend in window")
    return ", ".join(parts) if parts else "Normal conditions"


def confidence_label(mape: float) -> str:
    if mape < 10:
        return "high"
    if mape < 18:
        return "medium"
    return "low"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    if artifact is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    models = artifact["models"]
    mape_scores = artifact["mape_scores"]
    item_stats = artifact["item_stats"]
    all_features = artifact["features"]

    today = date.today()
    days_until_monday = (7 - today.weekday()) % 7 or 7
    week_start = today + timedelta(days=days_until_monday)
    week_end = week_start + timedelta(days=6)
    week_str = f"{week_start.strftime('%b %d')} - {week_end.strftime('%b %d, %Y')}"

    weather = get_lisbon_weather()
    daily_rainfall = weather["daily_rainfall"]
    daily_temps = weather["daily_temps"]
    local_event = int(req.upcoming_events)
    is_tourist = int(req.is_tourist_season)

    recommendations = []

    for item in models:
        model = models[item]
        stats = item_stats[item]
        mape = mape_scores.get(item, 15.0)
        lag_7, lag_14, lag_28 = get_lag_values(item, week_start)
        rolling_avg = get_rolling_avg(item)

        weekly_pred = 0.0
        daily_preds = []
        for d in range(7):
            target_date = week_start + timedelta(days=d)
            is_holiday = int(target_date in PORTUGUESE_HOLIDAYS or req.is_holiday_week)
            row = build_feature_row(
                target_date=target_date,
                rainfall_mm=daily_rainfall[d],
                temperature_c=daily_temps[d],
                local_event=local_event,
                is_holiday=is_holiday,
                is_tourist_season=is_tourist,
                rolling_7day_avg=rolling_avg,
                lag_7=lag_7,
                lag_14=lag_14,
                lag_28=lag_28,
                month_override=req.forecast_month,
            )
            X = pd.DataFrame([row])[all_features]
            daily_pred = max(0.0, float(model.predict(X)[0]))
            daily_preds.append(int(round(daily_pred)))
            weekly_pred += daily_pred

        predicted_demand = max(1, int(round(weekly_pred)))
        recommended_order = int(np.ceil(predicted_demand * 1.10))

        last_week_total = int(round(stats["last_7_avg"] * 7))
        if last_week_total > 0:
            pct = (predicted_demand - last_week_total) / last_week_total * 100
            vs_last_week = f"{pct:+.0f}%"
        else:
            vs_last_week = "N/A"

        historical_weekly = stats["mean"] * 7
        flag = abs(predicted_demand - historical_weekly) / historical_weekly > 0.30

        reasoning = build_reasoning(req, week_start)
        conf = confidence_label(mape)

        recommendations.append(ItemForecast(
            menu_item=item,
            predicted_demand=predicted_demand,
            recommended_order=recommended_order,
            vs_last_week=vs_last_week,
            reasoning=reasoning,
            confidence=conf,
            flag=flag,
            daily_predictions=daily_preds,
        ))

    recommendations.sort(key=lambda x: (-int(x.flag), -x.predicted_demand))

    avg_mape = float(np.mean(list(mape_scores.values())))

    # Waste rates aligned with frontend: 8% industry avg without Mise, 3% with Mise
    total_predicted_units = sum(r.predicted_demand for r in recommendations)
    without_mise_waste = total_predicted_units * 0.08 * 2.50
    with_mise_waste    = total_predicted_units * 0.03 * 2.50
    waste_saved        = without_mise_waste - with_mise_waste

    ai_insights = get_ai_insights(recommendations, weather, req.restaurant_name, req.owner_notes)

    return ForecastResponse(
        restaurant=req.restaurant_name,
        week=week_str,
        recommendations=recommendations,
        total_estimated_waste_saved=f"€{waste_saved:.0f}",
        without_mise_waste=without_mise_waste,
        with_mise_waste=with_mise_waste,
        model_accuracy=f"MAPE: {avg_mape:.1f}%",
        ai_insights=ai_insights,
        daily_temps=daily_temps,
        daily_rainfall=daily_rainfall,
    )

