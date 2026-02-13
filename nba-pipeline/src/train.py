# nba-pipeline/src/train.py
"""
NBA Game Outcome Model — Training Script

Trains an XGBoost binary classifier to predict home team wins.
- Target: GAME_OUTCOME (1=home win, 0=away win)
- Features: 52 rolling team stats + 8 derived difference features = 60 total
- Split: chronological 80/20 (no future leakage)
- Evaluation: ROC-AUC, accuracy, classification report, top feature importances
- Output: saved model at nba-pipeline/models/nba_winner.json

Usage: python train.py
"""

import sys
from pathlib import Path
from datetime import datetime, timezone

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "nba-pipeline" / "src"))

import os
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
import json
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report,
    confusion_matrix, precision_score, recall_score, f1_score,
    log_loss,
)
from dotenv import load_dotenv

load_dotenv()

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("supabase").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Import paginated fetch + supabase client from gamelogs
from gamelogs import fetch_paginated, supabase

MODEL_DIR = REPO_ROOT / "nba-pipeline" / "models"
MODEL_PATH = MODEL_DIR / "nba_winner.json"
REPORT_PATH = MODEL_DIR / "nba_winner_report.json"

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
ROLLING_STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "PF",
                 "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS"]

ROLLING_COLS = []
for _stat in ROLLING_STATS + ["WIN_RATE", "GAMES"]:
    for _w in [10, 30]:
        ROLLING_COLS.append(f"{_stat}_{_w}")

# 52 raw rolling feature columns
FEATURE_COLS = [f"HOME_{c}" for c in ROLLING_COLS] + [f"AWAY_{c}" for c in ROLLING_COLS]

# 8 derived difference features (home minus away)
DIFF_FEATURES = {
    "DIFF_PTS_10": ("HOME_PTS_10", "AWAY_PTS_10"),
    "DIFF_PTS_30": ("HOME_PTS_30", "AWAY_PTS_30"),
    "DIFF_WIN_RATE_10": ("HOME_WIN_RATE_10", "AWAY_WIN_RATE_10"),
    "DIFF_WIN_RATE_30": ("HOME_WIN_RATE_30", "AWAY_WIN_RATE_30"),
    "DIFF_PLUS_MINUS_10": ("HOME_PLUS_MINUS_10", "AWAY_PLUS_MINUS_10"),
    "DIFF_PLUS_MINUS_30": ("HOME_PLUS_MINUS_30", "AWAY_PLUS_MINUS_30"),
    "DIFF_FG_PCT_10": ("HOME_FG_PCT_10", "AWAY_FG_PCT_10"),
    "DIFF_FG_PCT_30": ("HOME_FG_PCT_30", "AWAY_FG_PCT_30"),
}

ALL_FEATURES = FEATURE_COLS + list(DIFF_FEATURES.keys())


# ---------------------------------------------------------------------------
# 1. Fetch training data
# ---------------------------------------------------------------------------
def fetch_training_data():
    """Fetch completed gamelogs (GAME_STATUS 3 or 4) with known outcome."""
    logger.info("Fetching completed gamelogs from Supabase...")

    # Fetch status=3 and status=4 separately, then combine
    rows_3 = fetch_paginated("gamelogs", "*", [("eq", "GAME_STATUS", 3)])
    rows_4 = fetch_paginated("gamelogs", "*", [("eq", "GAME_STATUS", 4)])
    all_rows = rows_3 + rows_4

    if not all_rows:
        logger.error("No completed gamelogs found. Run gamelogs.py first.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    logger.info(f"  Fetched {len(df)} rows (status 3: {len(rows_3)}, status 4: {len(rows_4)})")

    # Filter to rows with known outcome
    df["GAME_OUTCOME"] = pd.to_numeric(df["GAME_OUTCOME"], errors="coerce")
    df = df[df["GAME_OUTCOME"].notna()].copy()
    logger.info(f"  {len(df)} rows with known GAME_OUTCOME")

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], utc=True)
    df["SEASON_ID"] = pd.to_numeric(df.get("SEASON_ID"), errors="coerce")

    # Cast feature columns to float
    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# 2. Build feature matrix
# ---------------------------------------------------------------------------
def add_diff_features(df):
    """Add derived difference features (home minus away)."""
    for name, (home_col, away_col) in DIFF_FEATURES.items():
        if home_col in df.columns and away_col in df.columns:
            df[name] = df[home_col] - df[away_col]
        else:
            df[name] = np.nan
    return df


def build_feature_matrix(df):
    """Extract features and target, dropping rows with null features."""
    df = add_diff_features(df)

    # Drop rows where any raw rolling feature is null (early-season)
    before = len(df)
    df = df.dropna(subset=FEATURE_COLS)
    dropped = before - len(df)
    if dropped > 0:
        logger.info(f"  Dropped {dropped} rows with null rolling features (early season)")

    if len(df) < 100:
        logger.warning(f"Only {len(df)} training samples — results may be unreliable")

    X = df[ALL_FEATURES].astype(float)
    y = df["GAME_OUTCOME"].astype(int)

    return X, y, df


# ---------------------------------------------------------------------------
# 3. Chronological train/test split
# ---------------------------------------------------------------------------
def time_split(X, y, df, test_fraction=0.20):
    """Split data chronologically. Last test_fraction by date goes to test."""
    sorted_idx = df["GAME_DATE"].sort_values().index
    X = X.loc[sorted_idx]
    y = y.loc[sorted_idx]
    df = df.loc[sorted_idx]

    cutoff = int(len(X) * (1 - test_fraction))

    X_train, X_test = X.iloc[:cutoff], X.iloc[cutoff:]
    y_train, y_test = y.iloc[:cutoff], y.iloc[cutoff:]
    df_test = df.iloc[cutoff:]

    train_end = df.iloc[cutoff - 1]["GAME_DATE"]
    test_start = df.iloc[cutoff]["GAME_DATE"]

    logger.info(f"  Train: {len(X_train)} games (through {train_end.date()})")
    logger.info(f"  Test:  {len(X_test)} games (from {test_start.date()})")
    logger.info(f"  Train home-win rate: {y_train.mean():.3f}")
    logger.info(f"  Test  home-win rate: {y_test.mean():.3f}")

    return X_train, X_test, y_train, y_test, df_test


# ---------------------------------------------------------------------------
# 4. Train model
# ---------------------------------------------------------------------------
def train_model(X_train, y_train, X_test, y_test):
    """Train XGBoost classifier with early stopping."""
    logger.info("Training XGBoost classifier...")

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="auc",
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        verbosity=1,
        early_stopping_rounds=30,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=True,
    )

    logger.info(f"  Best iteration: {model.best_iteration}")
    return model


# ---------------------------------------------------------------------------
# 5. Evaluate model
# ---------------------------------------------------------------------------
def _season_id_to_label(season_id):
    """Convert numeric SEASON_ID (e.g. 22024) to human-readable label (e.g. '2024-25')."""
    if season_id is None or pd.isna(season_id):
        return "unknown"
    sid = int(season_id)
    year = sid % 10000  # e.g. 22024 → 2024
    return f"{year}-{str(year + 1)[-2:]}"


def compute_season_breakdown(y_test, y_pred, y_prob, df_test):
    """Compute accuracy, AUC, and record counts per season in the test set."""
    results = {}
    season_ids = df_test["SEASON_ID"].values

    for sid in sorted(set(season_ids)):
        mask = season_ids == sid
        y_t = y_test.values[mask]
        y_p = y_pred[mask]
        y_pr = y_prob[mask]

        n = int(mask.sum())
        acc = round(float(accuracy_score(y_t, y_p)), 4)

        # AUC requires both classes present
        if len(set(y_t)) > 1:
            auc = round(float(roc_auc_score(y_t, y_pr)), 4)
        else:
            auc = None

        label = _season_id_to_label(sid)
        results[label] = {
            "season_id": int(sid),
            "games": n,
            "accuracy": acc,
            "roc_auc": auc,
            "home_win_rate": round(float(y_t.mean()), 4),
        }

    return results


def evaluate_model(model, X_test, y_test, df_test, train_info=None):
    """Evaluate model, print metrics, and return a full report dict."""
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    prec_away = precision_score(y_test, y_pred, pos_label=0)
    rec_away = recall_score(y_test, y_pred, pos_label=0)
    f1_away = f1_score(y_test, y_pred, pos_label=0)
    prec_home = precision_score(y_test, y_pred, pos_label=1)
    rec_home = recall_score(y_test, y_pred, pos_label=1)
    f1_home = f1_score(y_test, y_pred, pos_label=1)

    # Feature importances (all features, sorted)
    importance = model.feature_importances_
    feat_imp = pd.Series(importance, index=ALL_FEATURES).sort_values(ascending=False)

    # Per-season breakdown
    season_breakdown = compute_season_breakdown(y_test, y_pred, y_prob, df_test)

    # --- Print to console ---
    print("\n" + "=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"  ROC-AUC:  {auc:.4f}")
    print(f"  Accuracy: {acc:.4f} ({(y_pred == y_test).sum()}/{len(y_test)})")
    print(f"  Log Loss: {logloss:.4f}")
    print()
    print(f"  Confusion Matrix:")
    print(f"                  Predicted Away  Predicted Home")
    print(f"    Actual Away   {cm[0][0]:>14d}  {cm[0][1]:>14d}")
    print(f"    Actual Home   {cm[1][0]:>14d}  {cm[1][1]:>14d}")
    print()
    print(classification_report(y_test, y_pred, target_names=["Away Win", "Home Win"]))

    print("Top 15 Feature Importances:")
    print("-" * 40)
    for feat, imp in feat_imp.head(15).items():
        print(f"  {feat:<30s} {imp:.4f}")
    print()

    print("Per-Season Accuracy (test set):")
    print("-" * 50)
    for label, info in season_breakdown.items():
        auc_str = f"{info['roc_auc']:.4f}" if info['roc_auc'] is not None else "N/A"
        print(f"  {label:<10s}  {info['games']:>5d} games  Acc={info['accuracy']:.4f}  AUC={auc_str}")
    print()

    # --- Build report dict ---
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_file": MODEL_PATH.name,
        "xgboost_params": {
            "objective": "binary:logistic",
            "max_depth": 4,
            "learning_rate": 0.05,
            "n_estimators": 300,
            "early_stopping_rounds": 30,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 5,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "best_iteration": model.best_iteration,
        },
        "dataset": {
            "total_samples": (train_info or {}).get("total_samples"),
            "train_samples": (train_info or {}).get("train_samples"),
            "test_samples": int(len(y_test)),
            "train_date_range": (train_info or {}).get("train_date_range"),
            "test_date_range": (train_info or {}).get("test_date_range"),
            "home_win_rate_overall": (train_info or {}).get("home_win_rate_overall"),
            "home_win_rate_train": (train_info or {}).get("home_win_rate_train"),
            "home_win_rate_test": round(float(y_test.mean()), 4),
            "features_count": len(ALL_FEATURES),
            "features": ALL_FEATURES,
        },
        "performance": {
            "roc_auc": round(auc, 4),
            "accuracy": round(acc, 4),
            "log_loss": round(logloss, 4),
            "confusion_matrix": {
                "true_away_pred_away": int(cm[0][0]),
                "true_away_pred_home": int(cm[0][1]),
                "true_home_pred_away": int(cm[1][0]),
                "true_home_pred_home": int(cm[1][1]),
            },
            "away_win": {
                "precision": round(prec_away, 4),
                "recall": round(rec_away, 4),
                "f1_score": round(f1_away, 4),
                "support": int((y_test == 0).sum()),
            },
            "home_win": {
                "precision": round(prec_home, 4),
                "recall": round(rec_home, 4),
                "f1_score": round(f1_home, 4),
                "support": int((y_test == 1).sum()),
            },
        },
        "season_accuracy": season_breakdown,
        "feature_importances": {
            feat: round(float(imp), 6) for feat, imp in feat_imp.items()
        },
    }

    return auc, acc, report


def save_report(report, path):
    """Save model performance report as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report saved to {path}")


# ---------------------------------------------------------------------------
# 6. Save model
# ---------------------------------------------------------------------------
def save_model(model, path):
    """Save XGBoost model to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))
    logger.info(f"Model saved to {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start = datetime.now(timezone.utc)
    logger.info("=== NBA GAME OUTCOME MODEL — TRAINING ===")

    # Fetch
    df = fetch_training_data()

    # Features
    logger.info("Building feature matrix...")
    X, y, df = build_feature_matrix(df)
    logger.info(f"  {len(X)} samples, {len(ALL_FEATURES)} features")
    logger.info(f"  Overall home-win rate: {y.mean():.3f}")

    # Split
    logger.info("Splitting train/test (chronological 80/20)...")
    X_train, X_test, y_train, y_test, df_test = time_split(X, y, df)

    # Train
    model = train_model(X_train, y_train, X_test, y_test)

    # Build training info for the report
    sorted_df = df.sort_values("GAME_DATE")
    cutoff = int(len(sorted_df) * 0.80)
    train_info = {
        "total_samples": len(X),
        "train_samples": len(X_train),
        "train_date_range": f"{sorted_df.iloc[0]['GAME_DATE'].date()} to {sorted_df.iloc[cutoff - 1]['GAME_DATE'].date()}",
        "test_date_range": f"{sorted_df.iloc[cutoff]['GAME_DATE'].date()} to {sorted_df.iloc[-1]['GAME_DATE'].date()}",
        "home_win_rate_overall": round(float(y.mean()), 4),
        "home_win_rate_train": round(float(y_train.mean()), 4),
    }

    # Evaluate
    auc, acc, report = evaluate_model(model, X_test, y_test, df_test, train_info)

    # Save model + report
    save_model(model, MODEL_PATH)
    save_report(report, REPORT_PATH)

    elapsed = (datetime.now(timezone.utc) - start).total_seconds()
    logger.info(f"=== TRAINING COMPLETE in {elapsed:.1f}s | AUC={auc:.4f} | Acc={acc:.4f} ===")


if __name__ == "__main__":
    main()
