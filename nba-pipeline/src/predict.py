# nba-pipeline/src/predict.py
"""
NBA Game Outcome Model — Prediction Script

Loads a trained XGBoost model and predicts outcomes for today's scheduled games.
- Reads scheduled gamelogs (GAME_STATUS=1) for today from Supabase
- Skips games that already have a prediction (PREDICTION is not null)
- Skips games missing rolling features (data not ready)
- Writes PREDICTION (1=home, 0=away) and PREDICTION_PCT (home win prob) to gamelogs
- Prints a formatted table of ALL today's predictions (new + existing)

Usage: python predict.py
"""

import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "nba-pipeline" / "src"))

import os
import logging
import pandas as pd
import numpy as np
import xgboost as xgb
from dotenv import load_dotenv

load_dotenv()

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("supabase").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

from gamelogs import fetch_paginated, supabase

MODEL_DIR = REPO_ROOT / "nba-pipeline" / "models"
MODEL_PATH = MODEL_DIR / "nba_winner.json"

# ---------------------------------------------------------------------------
# Feature definitions (must match train.py exactly)
# ---------------------------------------------------------------------------
ROLLING_STATS = ["PTS", "REB", "AST", "STL", "BLK", "TOV", "PF",
                 "FG_PCT", "FG3_PCT", "FT_PCT", "PLUS_MINUS"]

ROLLING_COLS = []
for _stat in ROLLING_STATS + ["WIN_RATE", "GAMES"]:
    for _w in [10, 30]:
        ROLLING_COLS.append(f"{_stat}_{_w}")

FEATURE_COLS = [f"HOME_{c}" for c in ROLLING_COLS] + [f"AWAY_{c}" for c in ROLLING_COLS]

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
# 1. Load model
# ---------------------------------------------------------------------------
def load_model(path):
    """Load XGBoost model from JSON file. Returns None if not found."""
    if not path.exists():
        logger.warning(f"Model not found: {path}")
        logger.warning("Run train.py first to generate the model.")
        return None

    model = xgb.XGBClassifier()
    model.load_model(str(path))
    logger.info(f"Model loaded from {path}")
    return model


# ---------------------------------------------------------------------------
# 2. Fetch today's scheduled games
# ---------------------------------------------------------------------------
def fetch_todays_games():
    """Fetch gamelogs for today's date with GAME_STATUS=1 (scheduled).

    "Today" is defined in US/Eastern time since all NBA games are scheduled
    in ET. GAME_DATE is stored as the EST date at midnight (e.g.
    2024-02-13T00:00:00+00:00), so we simply match on today's date.
    """
    eastern = ZoneInfo("America/New_York")
    now_et = datetime.now(eastern)
    today_str = now_et.strftime("%Y-%m-%dT00:00:00+00:00")

    filters = [
        ("eq", "GAME_STATUS", 1),
        ("eq", "GAME_DATE", today_str),
    ]

    rows = fetch_paginated("gamelogs", "*", filters)
    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], utc=True)
    df["GAME_ID"] = pd.to_numeric(df["GAME_ID"], errors="coerce").astype("Int64")
    df["PREDICTION"] = pd.to_numeric(df.get("PREDICTION"), errors="coerce")
    df["PREDICTION_PCT"] = pd.to_numeric(df.get("PREDICTION_PCT"), errors="coerce")

    for col in FEATURE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# 3. Add derived features
# ---------------------------------------------------------------------------
def add_diff_features(df):
    """Add derived difference features (home minus away)."""
    for name, (home_col, away_col) in DIFF_FEATURES.items():
        if home_col in df.columns and away_col in df.columns:
            df[name] = df[home_col] - df[away_col]
        else:
            df[name] = np.nan
    return df


# ---------------------------------------------------------------------------
# 4. Predict and write back
# ---------------------------------------------------------------------------
def predict_and_write(model, games_df):
    """Predict outcomes for new games and write to Supabase.

    Returns:
        tuple: (new_predictions_df, skipped_count)
    """
    # Separate already-predicted vs new
    already_predicted = games_df[games_df["PREDICTION"].notna()].copy()
    new_games = games_df[games_df["PREDICTION"].isna()].copy()

    logger.info(f"  {len(already_predicted)} games already predicted")
    logger.info(f"  {len(new_games)} games need prediction")

    if new_games.empty:
        return pd.DataFrame(), 0

    # Check for games with missing rolling features
    new_games = add_diff_features(new_games)
    has_features = new_games[FEATURE_COLS].notna().all(axis=1)
    skipped = new_games[~has_features]
    predictable = new_games[has_features].copy()

    if len(skipped) > 0:
        logger.warning(f"  Skipping {len(skipped)} games with missing features:")
        for _, row in skipped.iterrows():
            logger.warning(f"    GAME_ID={row['GAME_ID']}: {row.get('AWAY_NAME', '?')} @ {row.get('HOME_NAME', '?')}")

    if predictable.empty:
        return pd.DataFrame(), len(skipped)

    # Build feature matrix and predict
    X = predictable[ALL_FEATURES].astype(float)
    probs = model.predict_proba(X)[:, 1]

    predictable["PREDICTION_PCT"] = probs
    predictable["PREDICTION"] = (probs >= 0.5).astype(int)

    # Write predictions to Supabase
    logger.info(f"  Writing {len(predictable)} predictions to Supabase...")
    for _, row in predictable.iterrows():
        game_id = int(row["GAME_ID"])
        pred = int(row["PREDICTION"])
        prob = round(float(row["PREDICTION_PCT"]), 3)

        try:
            supabase.table("gamelogs").update({
                "PREDICTION": pred,
                "PREDICTION_PCT": prob,
            }).eq("GAME_ID", game_id).execute()
        except Exception as e:
            logger.error(f"  Failed to write GAME_ID={game_id}: {e}")

    return predictable, len(skipped)


# ---------------------------------------------------------------------------
# 5. Format and print output
# ---------------------------------------------------------------------------
def print_predictions(games_df, new_predictions_df):
    """Print formatted table of ALL today's predictions."""
    today_str = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    # Mark which games are new vs existing
    new_ids = set(new_predictions_df["GAME_ID"].tolist()) if not new_predictions_df.empty else set()

    # Merge new predictions back into the full games list
    display = games_df.copy()
    if not new_predictions_df.empty:
        for _, row in new_predictions_df.iterrows():
            mask = display["GAME_ID"] == row["GAME_ID"]
            display.loc[mask, "PREDICTION"] = row["PREDICTION"]
            display.loc[mask, "PREDICTION_PCT"] = row["PREDICTION_PCT"]

    # Filter to games with predictions
    display = display[display["PREDICTION"].notna()].copy()

    if display.empty:
        print(f"\n  No predictions available for {today_str}\n")
        return

    print(f"\n{'=' * 90}")
    print(f"  NBA PREDICTIONS FOR {today_str}")
    print(f"{'=' * 90}")
    print(f"  {'Away Team':<26s}    {'Home Team':<26s} {'Pick':<8s} {'Prob':>7s}  {'Status'}")
    print(f"  {'-' * 84}")

    for _, row in display.sort_values("GAME_DATE").iterrows():
        away = row.get("AWAY_NAME", "???")
        home = row.get("HOME_NAME", "???")
        pred = int(row["PREDICTION"])
        prob = float(row["PREDICTION_PCT"])
        game_id = row["GAME_ID"]

        pick = home if pred == 1 else away
        # Prob displayed is always the predicted winner's probability
        pick_prob = prob if pred == 1 else (1 - prob)
        status = "NEW" if game_id in new_ids else "EXISTING"

        # Truncate team names if needed
        away_disp = away[:25] if isinstance(away, str) else "???"
        home_disp = home[:25] if isinstance(home, str) else "???"
        pick_disp = pick[:7] if isinstance(pick, str) else "???"

        print(f"  {away_disp:<26s} @  {home_disp:<26s} {pick_disp:<8s} {pick_prob:>6.1%}  {status}")

    new_count = len(new_ids & set(display["GAME_ID"].tolist()))
    existing_count = len(display) - new_count
    print(f"\n  {len(display)} games | {new_count} new, {existing_count} existing | Model: {MODEL_PATH.name}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    logger.info("=== NBA GAME OUTCOME MODEL — PREDICTIONS ===")

    # Load model
    model = load_model(MODEL_PATH)
    if model is None:
        print("\n  No model found — skipping predictions. Run train.py first.\n")
        return

    # Fetch today's scheduled games
    logger.info("Fetching today's scheduled games...")
    games_df = fetch_todays_games()

    if games_df.empty:
        print("\n  No scheduled games found for today.\n")
        return

    logger.info(f"  {len(games_df)} scheduled games found")

    # Predict new games and write to Supabase
    new_predictions, skipped = predict_and_write(model, games_df)

    # Print all predictions
    print_predictions(games_df, new_predictions)

    if skipped > 0:
        print(f"  ⚠ {skipped} games skipped (missing features — run gamelogs.py current first)\n")


if __name__ == "__main__":
    main()
