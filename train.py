"""
AI-Based Harbor Sedimentation Prediction and Forecasting System
---------------------------------------------------------------
Run:  python train.py                      (uses synthetic demo data)
      python train.py --data my_data.csv   (uses your real data)

Your CSV needs a 'date' column and a target column (default: 'sed_rate').
Any other numeric columns (wave_height, river_discharge, ...) are used
as input features automatically.
"""

import argparse
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")  # saves figures to files; works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


# ----------------------------------------------------------------------
# 1. DATA
# ----------------------------------------------------------------------
def make_demo_data(n_months: int = 240, seed: int = 42) -> pd.DataFrame:
    """Synthetic monthly harbor data so you can test the pipeline.
    REPLACE with real data (bathymetry surveys, wave/tide/river records)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-01", periods=n_months, freq="MS")
    m = dates.month.values
    season = np.sin(2 * np.pi * (m - 1) / 12)

    wave_height = 1.2 + 0.5 * season + rng.normal(0, 0.15, n_months)       # m
    wind_speed = 6 + 2 * season + rng.normal(0, 0.8, n_months)             # m/s
    rainfall = np.clip(80 + 70 * -season + rng.normal(0, 25, n_months), 0, None)  # mm
    river_discharge = np.clip(200 + 1.5 * rainfall + rng.normal(0, 30, n_months), 0, None)  # m3/s
    tidal_range = 1.5 + 0.4 * np.sin(2 * np.pi * np.arange(n_months) / 6) + rng.normal(0, 0.05, n_months)  # m
    dredging = (rng.random(n_months) < 0.08).astype(int)

    sed_rate = (
        0.02 * river_discharge
        + 1.5 * wave_height
        - 1.0 * tidal_range
        - 2.5 * dredging
        + 0.3 * wind_speed
        + rng.normal(0, 0.4, n_months)
    )  # cm/month
    return pd.DataFrame({
        "date": dates, "wave_height": wave_height, "wind_speed": wind_speed,
        "rainfall": rainfall, "river_discharge": river_discharge,
        "tidal_range": tidal_range, "dredging": dredging, "sed_rate": sed_rate,
    })


def load_data(path: str | None) -> pd.DataFrame:
    df = make_demo_data() if path is None else pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    # fill small gaps in time series by interpolation
    num = df.select_dtypes("number").columns
    df[num] = df[num].interpolate(limit_direction="both")
    return df


# ----------------------------------------------------------------------
# 2. FEATURE ENGINEERING  (forecast target `horizon` steps ahead)
# ----------------------------------------------------------------------
def build_features(df: pd.DataFrame, target: str, horizon: int = 1,
                   lags=(1, 2, 3, 6, 12), roll=(3, 6)):
    df = df.copy()
    drivers = [c for c in df.select_dtypes("number").columns if c != target]

    # seasonality
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

    # lags of target and drivers (past information only -> no data leakage)
    for col in [target] + drivers:
        for l in lags:
            df[f"{col}_lag{l}"] = df[col].shift(l)
    for col in [target] + drivers:
        for w in roll:
            df[f"{col}_roll{w}"] = df[col].shift(1).rolling(w).mean()

    # target = value `horizon` steps in the future
    df["y"] = df[target].shift(-horizon)
    df = df.dropna().reset_index(drop=True)

    feat_cols = [c for c in df.columns if c not in ("date", "y", target)
             and not c.startswith(f"{target}_lag")
             and not c.startswith(f"{target}_roll")]
    return df, feat_cols


# ----------------------------------------------------------------------
# 3. TRAIN / EVALUATE
# ----------------------------------------------------------------------
def metrics(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def main(args):
    df_raw = load_data(args.data)
    df, feats = build_features(df_raw, args.target, args.horizon)

    # chronological split (never shuffle time series!)
    split = int(len(df) * (1 - args.test_size))
    train, test = df.iloc[:split], df.iloc[split:]
    X_tr, y_tr = train[feats], train["y"]
    X_te, y_te = test[feats], test["y"]
    print(f"Train: {len(train)} rows | Test: {len(test)} rows | Features: {len(feats)}")

    tscv = TimeSeriesSplit(n_splits=5)
    candidates = {
        "Ridge": (
            Pipeline([("sc", StandardScaler()), ("m", Ridge())]),
            {"m__alpha": [0.1, 1, 10, 100]},
        ),
        "RandomForest": (
            RandomForestRegressor(random_state=42, n_jobs=-1),
            {"n_estimators": [200, 400], "max_depth": [None, 8, 16], "min_samples_leaf": [1, 3]},
        ),
        "GradientBoosting": (
            GradientBoostingRegressor(random_state=42),
            {"n_estimators": [200, 400], "learning_rate": [0.03, 0.1], "max_depth": [2, 3]},
        ),
    }

    results, fitted, preds = {}, {}, {}
    for name, (est, grid) in candidates.items():
        gs = GridSearchCV(est, grid, cv=tscv, scoring="neg_root_mean_squared_error", n_jobs=-1)
        gs.fit(X_tr, y_tr)
        p = gs.predict(X_te)
        results[name] = metrics(y_te, p)
        fitted[name], preds[name] = gs.best_estimator_, p
        print(f"{name:17s} best={gs.best_params_}")

    # naive baseline: "next value = current value" (persistence)
    if args.target in X_te.columns:
        results["Persistence"] = metrics(y_te, X_te[args.target])
        preds["Persistence"] = X_te[args.target].values

    res_df = pd.DataFrame(results).T.sort_values("RMSE")
    print("\nTest-set results:\n", res_df.round(4))
    res_df.to_csv(OUT / "model_comparison.csv")

    best = res_df.drop(index="Persistence", errors="ignore").index[0]
    print(f"\nBest model: {best}")
    joblib.dump({"model": fitted[best], "features": feats, "target": args.target,
                 "horizon": args.horizon}, OUT / "best_model.joblib")

    # ---- plots ----
    plt.figure(figsize=(11, 4))
    plt.plot(test["date"], y_te.values, label="Observed", lw=2)
    plt.plot(test["date"], preds[best], label=f"Predicted ({best})", lw=2, ls="--")
    plt.title("Sedimentation: observed vs predicted (test period)")
    plt.xlabel("Date"); plt.ylabel(args.target); plt.legend(); plt.tight_layout()
    plt.savefig(OUT / "pred_vs_obs.png", dpi=200); plt.close()

    plt.figure(figsize=(5, 5))
    plt.scatter(y_te, preds[best], alpha=0.7)
    lim = [min(y_te.min(), preds[best].min()), max(y_te.max(), preds[best].max())]
    plt.plot(lim, lim, "r--"); plt.xlabel("Observed"); plt.ylabel("Predicted")
    plt.title(f"{best}  R²={results[best]['R2']:.3f}"); plt.tight_layout()
    plt.savefig(OUT / "scatter.png", dpi=200); plt.close()

    model = fitted[best]
    inner = model.named_steps["m"] if isinstance(model, Pipeline) else model
    if hasattr(inner, "feature_importances_"):
        imp = pd.Series(inner.feature_importances_, index=feats).nlargest(15)[::-1]
        imp.plot.barh(figsize=(7, 5), title="Top 15 feature importances")
        plt.tight_layout(); plt.savefig(OUT / "feature_importance.png", dpi=200); plt.close()

    # ---- forecast the next period from the latest data ----
    latest = build_features_for_forecast(df_raw, args.target, feats)
    print(f"\nForecast for {args.horizon} step(s) after last record: "
          f"{model.predict(latest)[0]:.3f}")
    print(f"Saved model, tables and figures in ./{OUT}/")


def build_features_for_forecast(df_raw, target, feats):
    """Build the feature row for the most recent date (no future target needed)."""
    df = df_raw.copy()
    drivers = [c for c in df.select_dtypes("number").columns if c != target]
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    for col in [target] + drivers:
        for l in (1, 2, 3, 6, 12):
            df[f"{col}_lag{l}"] = df[col].shift(l)
        for w in (3, 6):
            df[f"{col}_roll{w}"] = df[col].shift(1).rolling(w).mean()
    return df.iloc[[-1]][feats]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="path to CSV (omit for demo data)")
    ap.add_argument("--target", default="sed_rate")
    ap.add_argument("--horizon", type=int, default=1, help="steps ahead to forecast")
    ap.add_argument("--test_size", type=float, default=0.2)
    main(ap.parse_args())
