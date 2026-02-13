# pipeline/src/players.py
"""
NBA Players Pipeline:
- Fetches active players per season using commonallplayers
- Upserts to Supabase 'players' table using PERSON_ID (bigint PK).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared.nba.nba_constants import (
    TEAM_NAME_TO_ID,  # assuming you have this, but not needed here
)

import os
import logging
from datetime import datetime
import pandas as pd
from supabase import create_client, Client
from nba_api.stats.endpoints import commonallplayers
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# Suppress noisy library logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("supabase").setLevel(logging.WARNING)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase env vars")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


def get_active_players(season):
    """Fetch active players for a season using commonallplayers."""
    try:
        players = commonallplayers.CommonAllPlayers(is_only_current_season=0, season=season)
        df = players.get_data_frames()[0]
        if df.empty:
            print(f"0 players found for {season} from NBA API")
            return pd.DataFrame()

        active_players = df[df['ROSTERSTATUS'] == 1][['PERSON_ID', 'DISPLAY_FIRST_LAST', 'TEAM_ID', 'TEAM_NAME', 'FROM_YEAR', 'TO_YEAR', 'PLAYER_SLUG']].copy()
        active_players['SEASON_YEAR'] = int(season[:4])  # Add temp column for sorting latest

        active_players['PERSON_ID'] = pd.to_numeric(active_players['PERSON_ID'], errors='coerce').astype('Int64')
        active_players['TEAM_ID'] = pd.to_numeric(active_players['TEAM_ID'], errors='coerce').astype('Int64')
        active_players['FROM_YEAR'] = pd.to_numeric(active_players['FROM_YEAR'], errors='coerce').astype('Int64')
        active_players['TO_YEAR'] = pd.to_numeric(active_players['TO_YEAR'], errors='coerce').astype('Int64')

        print(f"{len(active_players)} players found for {season} from NBA API")
        return active_players
    except Exception as e:
        logger.error(f"Failed to fetch players for {season}: {e}")
        return pd.DataFrame()


def get_historical_players(seasons_to_fetch):
    """Fetch players for specified seasons."""
    all_rows = []
    for season in seasons_to_fetch:
        df = get_active_players(season)
        if not df.empty:
            all_rows.append(df)

    if all_rows:
        combined = pd.concat(all_rows, ignore_index=True)
        # Keep latest per PERSON_ID (sort by SEASON_YEAR)
        combined = combined.sort_values('SEASON_YEAR').drop_duplicates('PERSON_ID', keep='last')
        # Drop temp column
        combined = combined.drop(columns=['SEASON_YEAR'])
        return combined
    return pd.DataFrame()


def fetch_db_players():
    """Fetch all existing players from Supabase."""
    try:
        response = supabase.table("players").select("*").execute()
        if response.data:
            db_df = pd.DataFrame(response.data)
            if 'PERSON_ID' in db_df.columns:
                db_df['PERSON_ID'] = pd.to_numeric(db_df['PERSON_ID'], errors='coerce').astype('Int64')
            if 'TEAM_ID' in db_df.columns:
                db_df['TEAM_ID'] = pd.to_numeric(db_df['TEAM_ID'], errors='coerce').astype('Int64')
            if 'FROM_YEAR' in db_df.columns:
                db_df['FROM_YEAR'] = pd.to_numeric(db_df['FROM_YEAR'], errors='coerce').astype('Int64')
            if 'TO_YEAR' in db_df.columns:
                db_df['TO_YEAR'] = pd.to_numeric(db_df['TO_YEAR'], errors='coerce').astype('Int64')
            return db_df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to fetch DB players: {e}")
        return pd.DataFrame()


def find_deltas(new_df, db_df):
    """Find rows in new_df that differ from db_df or are missing in DB."""
    if db_df.empty:
        return new_df

    # Ensure consistent types
    for col in ['PERSON_ID', 'TEAM_ID', 'FROM_YEAR', 'TO_YEAR']:
        if col in new_df.columns:
            new_df[col] = pd.to_numeric(new_df[col], errors='coerce').astype('Int64')
        if col in db_df.columns:
            db_df[col] = pd.to_numeric(db_df[col], errors='coerce').astype('Int64')

    merged = new_df.merge(db_df, on="PERSON_ID", suffixes=("_new", "_db"), how="left")

    compare_cols = ["DISPLAY_FIRST_LAST", "TEAM_ID", "TEAM_NAME", "FROM_YEAR", "TO_YEAR", "PLAYER_SLUG"]

    delta_mask = merged['DISPLAY_FIRST_LAST_db'].isna()  # new rows

    for col in compare_cols:
        delta_mask |= (
            (merged[f"{col}_new"] != merged[f"{col}_db"]) &
            (merged[f"{col}_new"].notna() | merged[f"{col}_db"].notna())
        )

    # Select the original new columns + unsuffixed PERSON_ID
    new_cols = [c for c in merged.columns if c.endswith('_new')]
    selected_cols = new_cols + ['PERSON_ID']

    deltas = merged.loc[delta_mask, selected_cols].copy()

    # Rename _new back to original
    rename_dict = {c: c.replace('_new', '') for c in new_cols}
    deltas.rename(columns=rename_dict, inplace=True)

    return deltas


def upsert_players_to_supabase(df: pd.DataFrame):
    if df.empty:
        print("Upserting 0 players")
        return

    # Ensure correct types
    for col in ['PERSON_ID', 'TEAM_ID', 'FROM_YEAR', 'TO_YEAR']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    records = df.to_dict(orient="records")

    success_count = 0
    with tqdm(total=len(records), desc="Upserting players", unit="player", leave=False) as pbar:
        for r in records:
            payload = {
                "PERSON_ID": int(r["PERSON_ID"]),
                "DISPLAY_FIRST_LAST": r.get("DISPLAY_FIRST_LAST"),
                "TEAM_ID": int(r["TEAM_ID"]) if pd.notna(r.get("TEAM_ID")) else None,
                "TEAM_NAME": r.get("TEAM_NAME"),
                "FROM_YEAR": int(r["FROM_YEAR"]) if pd.notna(r.get("FROM_YEAR")) else None,
                "TO_YEAR": int(r["TO_YEAR"]) if pd.notna(r.get("TO_YEAR")) else None,
                "PLAYER_SLUG": r.get("PLAYER_SLUG"),
            }
            try:
                supabase.table("players").upsert(payload, on_conflict="PERSON_ID").execute()
                success_count += 1
            except Exception as e:
                logger.error(f"Upsert failed for PERSON_ID {payload['PERSON_ID']}: {e}")
            pbar.update(1)

    print(f"Upsert complete: {success_count}/{len(records)} players processed")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "current"

    if mode == "full":
        print("\n=== Running FULL mode: historical backfill + current update ===")
        today = datetime.today()
        current_year = today.year if today.month >= 10 else today.year - 1
        seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(2020, current_year + 1)]
        hist_df = get_historical_players(seasons)
        if not hist_df.empty:
            upsert_players_to_supabase(hist_df)
    else:
        print("\n=== Running INCREMENTAL mode: current + delta check ===")
        today = datetime.now()
        current_year = today.year if today.month >= 10 else today.year - 1
        current_season = f"{current_year}-{str(current_year + 1)[-2:]}"

        new_df = get_active_players(current_season)
        db_df = fetch_db_players()

        api_count = len(new_df) if not new_df.empty else 0
        db_count = len(db_df) if not db_df.empty else 0
        print(f"{api_count} players found for {current_season} from NBA API")
        print(f"{db_count} players found in players table")

        if not new_df.empty:
            deltas = find_deltas(new_df, db_df)
            print(f"Deltas found: {len(deltas)}")
            if not deltas.empty:
                print(f"Upserting {len(deltas)} players")
                upsert_players_to_supabase(deltas)
            else:
                print("Upserting 0 players (no deltas)")
        else:
            print("Deltas found: 0")
            print("Upserting 0 players")


if __name__ == "__main__":
    main()
