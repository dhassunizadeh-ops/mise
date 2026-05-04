"""
Generates 730 days (2 years) of synthetic sales data for Da Mario, Lisbon.
All demand patterns are encoded as genuine statistical relationships in the data
so the ML model can learn them from the features — no hardcoded adjustments elsewhere.
"""
import math
import os
from datetime import date, timedelta

import numpy as np
import pandas as pd

np.random.seed(42)

MENU_ITEMS = [
    "Margherita Pizza",
    "Pasta Carbonara",
    "Grilled Salmon",
    "Caesar Salad",
    "Tiramisu",
    "Risotto",
    "Bruschetta",
    "Grilled Chicken",
]

BASE_DEMAND = {
    "Margherita Pizza": 30,
    "Pasta Carbonara": 28,
    "Grilled Salmon": 18,
    "Caesar Salad": 15,
    "Tiramisu": 20,
    "Risotto": 22,
    "Bruschetta": 25,
    "Grilled Chicken": 20,
}

PORTUGUESE_HOLIDAYS = {
    date(2024, 1, 1), date(2024, 3, 29), date(2024, 4, 25), date(2024, 5, 1),
    date(2024, 6, 10), date(2024, 6, 13), date(2024, 8, 15), date(2024, 10, 5),
    date(2024, 11, 1), date(2024, 12, 1), date(2024, 12, 8), date(2024, 12, 25),
    date(2025, 1, 1), date(2025, 4, 18), date(2025, 4, 25), date(2025, 5, 1),
    date(2025, 6, 10), date(2025, 6, 19), date(2025, 8, 15), date(2025, 10, 5),
    date(2025, 11, 1), date(2025, 12, 1), date(2025, 12, 8), date(2025, 12, 25),
}


def lisbon_temperature(day_of_year: int) -> float:
    """Sinusoidal Lisbon temperature model. Trough ~12°C Jan, peak ~29°C Jul."""
    base = 20.5 - 8.5 * math.cos(2 * math.pi * (day_of_year - 15) / 365)
    noise = np.random.normal(0, 2.5)
    return round(float(np.clip(base + noise, 8.0, 38.0)), 1)


def lisbon_rainfall(month: int) -> float:
    """Monthly rain probability and exponential intensity for Lisbon."""
    monthly = {
        1: (0.45, 9.0), 2: (0.40, 8.0), 3: (0.32, 6.0), 4: (0.28, 5.0),
        5: (0.18, 3.0), 6: (0.08, 1.5), 7: (0.03, 0.8), 8: (0.03, 0.8),
        9: (0.15, 3.0), 10: (0.30, 7.0), 11: (0.38, 9.0), 12: (0.42, 9.0),
    }
    prob, mean = monthly[month]
    if np.random.random() < prob:
        return round(float(min(np.random.exponential(mean), 25.0)), 1)
    return 0.0


def item_specific_multiplier(
    item: str, rainfall_mm: float, temperature_c: float,
    is_tourist_season: int, month: int,
) -> float:
    """
    Item-level weather/season multipliers. These create learnable patterns
    in the training data. All values are within the specified ranges.
    """
    mul = 1.0
    high_rain = rainfall_mm > 10.0
    warm_temp = temperature_c > 25.0
    cool_temp = temperature_c < 15.0
    warm_sunny = temperature_c > 22.0 and rainfall_mm < 3.0

    if item == "Grilled Salmon":
        if high_rain:
            mul *= np.random.uniform(0.75, 0.85)
        if warm_temp:
            mul *= np.random.uniform(1.15, 1.25)
        if is_tourist_season:
            mul *= np.random.uniform(1.20, 1.30)

    elif item in ("Pasta Carbonara", "Risotto"):
        if high_rain:
            mul *= np.random.uniform(1.10, 1.20)
        if cool_temp:
            mul *= np.random.uniform(1.15, 1.25)
        if month in (11, 12, 1, 2):
            mul *= np.random.uniform(1.15, 1.20)

    elif item in ("Caesar Salad", "Bruschetta"):
        if high_rain:
            mul *= np.random.uniform(0.80, 0.90)
        if warm_sunny:
            mul *= np.random.uniform(1.20, 1.30)

    return mul


def generate_sales_history() -> pd.DataFrame:
    start = date(2024, 1, 1)
    records = []
    rolling_buffers = {item: [] for item in MENU_ITEMS}

    for day_offset in range(730):
        current_date = start + timedelta(days=day_offset)
        dow = current_date.weekday()
        week = current_date.isocalendar()[1]
        month = current_date.month
        day_of_year = current_date.timetuple().tm_yday
        is_weekend = int(dow >= 5)
        is_holiday = int(current_date in PORTUGUESE_HOLIDAYS)
        is_tourist_season = int(month in (6, 7, 8, 9))

        temperature_c = lisbon_temperature(day_of_year)
        rainfall_mm = lisbon_rainfall(month)
        local_event = int(np.random.random() < 0.15)

        for item in MENU_ITEMS:
            base = BASE_DEMAND[item]

            # Universal multipliers (spec: these apply to all items)
            weekend_mul = np.random.uniform(1.30, 1.40) if is_weekend else 1.0
            holiday_mul = np.random.uniform(1.15, 1.25) if is_holiday else 1.0
            event_mul = np.random.uniform(1.20, 1.30) if local_event else 1.0
            tourist_mul = np.random.uniform(1.10, 1.20) if is_tourist_season else 1.0

            # Item-specific weather/season multipliers
            specific_mul = item_specific_multiplier(
                item, rainfall_mm, temperature_c, is_tourist_season, month
            )

            expected = base * weekend_mul * holiday_mul * event_mul * tourist_mul * specific_mul
            noise = np.random.normal(0, expected * 0.10)
            units_sold = max(1, int(round(expected + noise)))

            buf = rolling_buffers[item]
            buf.append(units_sold)
            if len(buf) > 7:
                buf.pop(0)
            rolling_7day_avg = round(sum(buf) / len(buf), 2)

            records.append({
                "date": current_date.isoformat(),
                "menu_item": item,
                "units_sold": units_sold,
                "day_of_week": dow,
                "week_of_year": week,
                "month": month,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "is_tourist_season": is_tourist_season,
                "rainfall_mm": rainfall_mm,
                "temperature_c": temperature_c,
                "local_event": local_event,
                "rolling_7day_avg": rolling_7day_avg,
            })

    df = pd.DataFrame(records)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/sales_history.csv", index=False)
    print(f"sales_history.csv written — {len(df)} rows")
    print(f"Columns: {list(df.columns)}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    return df


def generate_suppliers() -> pd.DataFrame:
    suppliers = [
        ("Quinta Verde", "produce", 2, 8.5, 0.92, 0.95),
        ("Mercado Lisboa", "produce", 1, 7.8, 0.88, 0.97),
        ("Horta Fresca", "produce", 3, 9.1, 1.05, 0.91),
        ("Carnes Premium", "meat", 3, 9.3, 1.20, 0.93),
        ("Talho Central", "meat", 2, 8.0, 1.10, 0.96),
        ("Bovino Select", "meat", 4, 8.8, 1.25, 0.88),
        ("Peixaria Atlantico", "fish", 1, 9.5, 1.15, 0.94),
        ("Mar Fresco", "fish", 2, 8.7, 1.08, 0.90),
        ("Costa Prata", "fish", 1, 9.0, 1.22, 0.98),
        ("Distribuidora Sol", "dry_goods", 5, 7.5, 0.85, 0.92),
        ("Armazem Lisboa", "dry_goods", 4, 7.2, 0.82, 0.89),
        ("Grao & Cereal", "dry_goods", 3, 8.1, 0.90, 0.95),
        ("Laticínios Serra", "dairy", 2, 9.2, 0.95, 0.97),
        ("Manteigaria Real", "dairy", 3, 8.6, 1.00, 0.93),
        ("Queijaria Norte", "dairy", 2, 8.9, 1.03, 0.91),
    ]
    df = pd.DataFrame(suppliers, columns=[
        "supplier_name", "category", "avg_delivery_days",
        "quality_score", "price_index", "on_time_rate",
    ])
    df.to_csv("data/suppliers.csv", index=False)
    print(f"suppliers.csv written — {len(df)} rows")
    return df


if __name__ == "__main__":
    generate_sales_history()
    generate_suppliers()
    print("Data generation complete.")
