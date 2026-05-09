"""
Trains one LightGBM model per menu item on the full feature set including
weather, temperature, tourist season, and lag features.
Also fits a KMeans cold-start model over synthetic restaurant profiles.
"""
import os
import pickle
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_percentage_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import lightgbm as lgb
    USE_LGBM = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor
    USE_LGBM = False
    print("LightGBM not found — falling back to GradientBoostingRegressor")

CLUSTER_LABELS: dict[int, str] = {
    0: "Small casual venue, low volume",
    1: "Mid-size tourist-area restaurant",
    2: "High-volume urban bistro",
    3: "Large evening dining venue",
    4: "Compact specialist restaurant",
}

# cuisine_encoded mapping used by both train and frontend
CUISINE_ENC: dict[str, int] = {
    "Italian": 0, "Portuguese": 1, "Mediterranean": 2, "Spanish": 3, "French": 4,
}

# All contextual features the model learns from
FEATURES = [
    "day_of_week",
    "week_of_year",
    "month",
    "is_weekend",
    "is_holiday",
    "is_tourist_season",
    "rainfall_mm",
    "temperature_c",
    "local_event",
    "rolling_7day_avg",
]


def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/sales_history.csv", parse_dates=["date"])
    return df.sort_values(["menu_item", "date"]).reset_index(drop=True)


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for item, group in df.groupby("menu_item"):
        g = group.copy().sort_values("date")
        g["lag_7"] = g["units_sold"].shift(7)
        g["lag_14"] = g["units_sold"].shift(14)
        g["lag_28"] = g["units_sold"].shift(28)
        frames.append(g)
    return pd.concat(frames).sort_values(["menu_item", "date"]).dropna().reset_index(drop=True)


def train_models(df: pd.DataFrame):
    models, feature_importances, mape_scores, naive_mape_scores = {}, {}, {}, {}
    all_features = FEATURES + ["lag_7", "lag_14", "lag_28"]

    print(f"\n{'='*65}")
    print("MISE — ML Training Results (LightGBM, 730-day dataset)")
    print(f"{'='*65}")
    print(f"{'Menu Item':<25} {'LightGBM MAPE':>15} {'Naive MAPE':>12} {'Improvement':>13}")
    print(f"{'-'*65}")

    for item in sorted(df["menu_item"].unique()):
        item_df = df[df["menu_item"] == item].copy().reset_index(drop=True)

        split_train = int(len(item_df) * 0.60)
        split_val = int(len(item_df) * 0.80)
        train = item_df.iloc[:split_train]
        val = item_df.iloc[split_train:split_val]
        test = item_df.iloc[split_val:]

        X_train = train[all_features]
        y_train = train["units_sold"]
        X_val = val[all_features]
        y_val = val["units_sold"]
        X_test = test[all_features]
        y_test = test["units_sold"]

        if USE_LGBM:
            params = {
                "objective": "regression",
                "metric": "mape",
                "learning_rate": 0.04,
                "num_leaves": 63,
                "n_estimators": 500,
                "min_child_samples": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.1,
                "reg_lambda": 0.1,
                "verbose": -1,
                "random_state": 42,
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(60, verbose=False),
                    lgb.log_evaluation(period=-1),
                ],
            )
            fi = dict(zip(all_features, model.feature_importances_))
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            model = GradientBoostingRegressor(
                n_estimators=300, learning_rate=0.04, max_depth=5,
                subsample=0.8, random_state=42,
            )
            model.fit(X_train, y_train)
            fi = dict(zip(all_features, model.feature_importances_))

        preds = np.maximum(model.predict(X_test), 0)
        mape = mean_absolute_percentage_error(y_test, preds) * 100

        naive_preds = test["lag_7"].values
        naive_mape = mean_absolute_percentage_error(y_test, naive_preds) * 100

        improvement = (naive_mape - mape) / naive_mape * 100

        models[item] = model
        feature_importances[item] = fi
        mape_scores[item] = round(mape, 2)
        naive_mape_scores[item] = round(naive_mape, 2)

        print(f"{item:<25} {mape:>13.1f}%  {naive_mape:>10.1f}%  {improvement:>+11.1f}%")

    avg_mape = np.mean(list(mape_scores.values()))
    avg_naive = np.mean(list(naive_mape_scores.values()))
    avg_improvement = (avg_naive - avg_mape) / avg_naive * 100

    print(f"{'-'*65}")
    print(f"{'AVERAGE':<25} {avg_mape:>13.1f}%  {avg_naive:>10.1f}%  {avg_improvement:>+11.1f}%")
    print(f"{'='*65}")
    print(f"\nLightGBM beats naive baseline by {avg_improvement:.1f}% on average MAPE\n")

    return models, feature_importances, mape_scores, naive_mape_scores


def build_kmeans() -> KMeans:
    """Generate 50 synthetic restaurant profiles and fit a 5-cluster KMeans model."""
    rng = np.random.default_rng(42)
    n = 50
    cuisine_enc     = rng.integers(0, 5, n).astype(float)
    avg_daily_vol   = rng.uniform(50, 400, n)
    seating_cap     = rng.uniform(20, 120, n)
    price_tier      = rng.integers(1, 4, n).astype(float)
    X = np.column_stack([cuisine_enc, avg_daily_vol, seating_cap, price_tier])
    km = KMeans(n_clusters=5, random_state=42, n_init=10)
    km.fit(X)
    return km


def assign_cluster(kmeans: KMeans, cuisine: str, seating_capacity: int) -> tuple[int, str]:
    c_enc   = float(CUISINE_ENC.get(cuisine, 0))
    avg_vol = min(seating_capacity * 2.5, 400.0)
    X = np.array([[c_enc, avg_vol, float(seating_capacity), 2.0]])
    cluster_num = int(kmeans.predict(X)[0])
    return cluster_num, CLUSTER_LABELS[cluster_num]


def save_artifacts(models, feature_importances, mape_scores, naive_mape_scores, df, kmeans_model):
    os.makedirs("ml", exist_ok=True)

    item_stats = {}
    for item in df["menu_item"].unique():
        idf = df[df["menu_item"] == item].sort_values("date")
        item_stats[item] = {
            "mean": float(idf["units_sold"].mean()),
            "std": float(idf["units_sold"].std()),
            "last_7_avg": float(idf.tail(7)["units_sold"].mean()),
            "last_28_avg": float(idf.tail(28)["units_sold"].mean()),
            # Store last 35 daily values for lag lookups
            "last_35_sales": idf.tail(35)[["date", "units_sold"]].to_dict("records"),
        }

    artifact = {
        "models": models,
        "feature_importances": feature_importances,
        "mape_scores": mape_scores,
        "naive_mape_scores": naive_mape_scores,
        "item_stats": item_stats,
        "features": FEATURES + ["lag_7", "lag_14", "lag_28"],
        "base_features": FEATURES,
        "menu_items": sorted(models.keys()),
        "kmeans_model": kmeans_model,
        "cluster_labels": CLUSTER_LABELS,
        "cuisine_enc": CUISINE_ENC,
    }

    with open("ml/model.pkl", "wb") as f:
        pickle.dump(artifact, f)

    print(f"Saved ml/model.pkl")
    print(f"Features: {FEATURES}")
    print(f"Items: {sorted(models.keys())}")
    print(f"KMeans clusters: {kmeans_model.n_clusters}")


if __name__ == "__main__":
    print("Loading sales_history.csv …")
    raw_df = load_data()
    print(f"  {len(raw_df)} rows, {raw_df['menu_item'].nunique()} items, "
          f"{raw_df['date'].min().date()} to {raw_df['date'].max().date()}")

    print("Adding lag features …")
    df = add_lag_features(raw_df)
    print(f"  {len(df)} rows after dropna")

    models, fi, mape_scores, naive_mape_scores = train_models(df)

    print("Fitting KMeans cold-start model …")
    kmeans_model = build_kmeans()
    print(f"  5 clusters fitted on 50 synthetic restaurant profiles")

    save_artifacts(models, fi, mape_scores, naive_mape_scores, df, kmeans_model)
    print("\nTraining complete.")
