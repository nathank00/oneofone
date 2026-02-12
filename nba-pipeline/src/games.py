# pipeline/src/games.py
"""
NBA Games Pipeline:
- Historical: leaguegamefinder (bulk for past seasons)
- Current/live/upcoming: scoreboard (real-time status & scores)
Upserts to Supabase 'games' table using GAME_ID (bigint PK).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared.nba.nba_constants import (
    TEAM_ABBR_TO_FULL,
    TEAM_SHORT_TO_FULL,
    TEAM_NAME_TO_ID,
)

import os
import logging
from datetime import datetime
import pandas as pd
from supabase import create_client, Client
from nba_api.stats.endpoints import leaguegamefinder
from nba_api.live.nba.endpoints import scoreboard
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

GAME_STATUS_SCHEDULED = 1
GAME_STATUS_LIVE      = 2
GAME_STATUS_FINAL     = 3
GAME_STATUS_POSTPONED = 4

GAME_OUTCOME_HOME_WIN = 1
GAME_OUTCOME_AWAY_WIN = 0
GAME_OUTCOME_TIE      = None


def get_historical_games(seasons_to_fetch):
    """Fetch games for specified seasons using leaguegamefinder."""
    all_rows = []
    for season in seasons_to_fetch:
        try:
            finder = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                league_id_nullable="00",
                season_type_nullable="Regular Season"
            )
            df = finder.get_data_frames()[0]
            if df.empty:
                print(f"0 games found for {season} from NBA API")
                continue

            # Standardize GAME_ID to Int64 early
            df['GAME_ID'] = pd.to_numeric(df['GAME_ID'], errors='coerce').astype('Int64')

            # Process to single row per game + derive outcome
            games = df.groupby('GAME_ID').agg({
                'SEASON_ID': 'first',
                'GAME_DATE': 'first',
                'MATCHUP': 'first',
            }).reset_index()

            games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'])
            games['GAME_ID'] = pd.to_numeric(games['GAME_ID'], errors='coerce').astype('Int64')

            games['is_away_home'] = games['MATCHUP'].str.contains('@')
            games['HOME_NAME_RAW'] = games.apply(
                lambda x: x['MATCHUP'].split('@')[1].strip() if x['is_away_home'] else x['MATCHUP'].split('vs.')[0].strip(),
                axis=1
            )
            games['AWAY_NAME_RAW'] = games.apply(
                lambda x: x['MATCHUP'].split('@')[0].strip() if x['is_away_home'] else x['MATCHUP'].split('vs.')[1].strip(),
                axis=1
            )

            games['HOME_NAME'] = games['HOME_NAME_RAW'].replace(TEAM_ABBR_TO_FULL)
            games['AWAY_NAME'] = games['AWAY_NAME_RAW'].replace(TEAM_ABBR_TO_FULL)
            games['HOME_ID'] = games['HOME_NAME'].map(TEAM_NAME_TO_ID).astype("Int64")
            games['AWAY_ID'] = games['AWAY_NAME'].map(TEAM_NAME_TO_ID).astype("Int64")

            def get_outcome(game_id):
                game_rows = df[df['GAME_ID'] == game_id]
                if len(game_rows) == 0:
                    return None
                row = game_rows.iloc[0]
                if pd.isna(row['WL']):
                    return None
                is_home = 'vs.' in row['MATCHUP']
                if row['WL'] == 'W':
                    return GAME_OUTCOME_HOME_WIN if is_home else GAME_OUTCOME_AWAY_WIN
                elif row['WL'] == 'L':
                    return GAME_OUTCOME_AWAY_WIN if is_home else GAME_OUTCOME_HOME_WIN
                return None

            games['GAME_OUTCOME'] = games['GAME_ID'].apply(get_outcome)
            games['GAME_STATUS'] = GAME_STATUS_FINAL if games['GAME_OUTCOME'].notna().all() else GAME_STATUS_SCHEDULED  # Adjust if needed

            games = games[["GAME_ID", "SEASON_ID", "GAME_DATE", "AWAY_NAME", "HOME_NAME", "AWAY_ID", "HOME_ID", "GAME_STATUS", "GAME_OUTCOME"]]
            games["SEASON_ID"] = games["SEASON_ID"].astype(str).str[-4:].astype(int)

            print(f"{len(games)} games found for {season} from NBA API")
            all_rows.append(games)
        except Exception as e:
            logger.error(f"Failed to fetch/process {season}: {e}")

    if all_rows:
        return pd.concat(all_rows, ignore_index=True).drop_duplicates("GAME_ID")
    return pd.DataFrame()


def fetch_db_games_for_seasons(season_ids):
    """Fetch existing games from Supabase for given SEASON_IDs."""
    try:
        response = supabase.table("games").select("*").in_("SEASON_ID", season_ids).execute()
        if response.data:
            db_df = pd.DataFrame(response.data)
            if 'GAME_ID' in db_df.columns:
                db_df['GAME_ID'] = pd.to_numeric(db_df['GAME_ID'], errors='coerce').astype('Int64')
            if 'GAME_DATE' in db_df.columns:
                db_df['GAME_DATE'] = pd.to_datetime(db_df['GAME_DATE'], errors='coerce', utc=True)
            return db_df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to fetch DB games for seasons {season_ids}: {e}")
        return pd.DataFrame()


def find_deltas(new_df, db_df):
    """Find rows in new_df that differ from db_df or are missing in DB."""
    if db_df.empty:
        return new_df

    # Ensure consistent numeric types
    for df in [new_df, db_df]:
        df['GAME_ID'] = pd.to_numeric(df['GAME_ID'], errors='coerce').astype('Int64')
        if 'GAME_DATE' in df.columns:
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce', utc=True)

    merged = new_df.merge(db_df, on="GAME_ID", suffixes=("_new", "_db"), how="left")

    compare_cols = ["SEASON_ID", "GAME_DATE", "AWAY_NAME", "HOME_NAME", "AWAY_ID", "HOME_ID", "GAME_STATUS", "GAME_OUTCOME"]

    delta_mask = merged['SEASON_ID_db'].isna()  # new rows

    for col in compare_cols:
        new_col = f"{col}_new"
        db_col = f"{col}_db"

        if col == "GAME_DATE":
            delta_mask |= (
                (merged[new_col].dt.date.ne(merged[db_col].dt.date)) &
                (merged[new_col].notna() | merged[db_col].notna())
            )
        else:
            delta_mask |= (
                (merged[new_col].ne(merged[db_col])) &
                (merged[new_col].notna() | merged[db_col].notna())
            )

    # Select the original new columns (via _new suffix) + unsuffixed GAME_ID
    new_cols = [c for c in merged.columns if c.endswith('_new')]
    selected_cols = new_cols + ['GAME_ID']

    deltas = merged.loc[delta_mask, selected_cols].copy()

    # Rename _new back to original column names
    rename_dict = {c: c.replace('_new', '') for c in new_cols}
    deltas.rename(columns=rename_dict, inplace=True)

    return deltas


def get_current_and_live_games():
    """Fetch today's/upcoming + live games from scoreboard."""
    try:
        sb = scoreboard.ScoreBoard()
        games_data = sb.games.get_dict()
        rows = []
        for g in games_data:
            gid = int(g['gameId'])
            status = g.get('gameStatus', GAME_STATUS_SCHEDULED)
            status_text = g.get('gameStatusText', '')

            outcome = None
            if status == GAME_STATUS_FINAL:
                home_score = g.get('homeTeam', {}).get('score', 0)
                away_score = g.get('awayTeam', {}).get('score', 0)
                if home_score > away_score:
                    outcome = GAME_OUTCOME_HOME_WIN
                elif away_score > home_score:
                    outcome = GAME_OUTCOME_AWAY_WIN
                else:
                    outcome = GAME_OUTCOME_TIE

            if 'Final' in status_text:
                status = GAME_STATUS_FINAL
            elif any(q in status_text for q in ['Q1','Q2','Q3','Q4','OT']):
                status = GAME_STATUS_LIVE

            game_time_utc = g.get('gameTimeUTC')
            game_date = pd.to_datetime(game_time_utc) if game_time_utc else pd.NaT

            rows.append({
                "GAME_ID": gid,
                "SEASON_ID": int(f"20{str(g['gameId'])[3:5]}"),
                "GAME_DATE": game_date,
                "AWAY_NAME": TEAM_SHORT_TO_FULL.get(g['awayTeam']['teamName'], g['awayTeam']['teamName']),
                "HOME_NAME": TEAM_SHORT_TO_FULL.get(g['homeTeam']['teamName'], g['homeTeam']['teamName']),
                "AWAY_ID": int(g['awayTeam']['teamId']),
                "HOME_ID": int(g['homeTeam']['teamId']),
                "GAME_STATUS": status,
                "GAME_OUTCOME": outcome,
            })
        df = pd.DataFrame(rows)
        print(f"{len(df)} games found for today's/current from scoreboard API")
        return df
    except Exception as e:
        logger.error(f"Scoreboard fetch failed: {e}")
        return pd.DataFrame()


def upsert_games_to_supabase(df: pd.DataFrame):
    if df.empty:
        print("Upserting 0 games")
        return

    # Ensure correct integer types
    for col in ['GAME_OUTCOME', 'GAME_STATUS', 'SEASON_ID', 'HOME_ID', 'AWAY_ID', 'GAME_ID']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    records = df.to_dict(orient="records")

    success_count = 0
    with tqdm(total=len(records), desc="Upserting games", unit="game", leave=False) as pbar:
        for r in records:
            game_date_iso = (
                pd.Timestamp(r["GAME_DATE"]).isoformat()
                if pd.notna(r.get("GAME_DATE")) and r.get("GAME_DATE") is not None
                else None
            )

            payload = {
                "GAME_ID": int(r["GAME_ID"]),
                "SEASON_ID": int(r["SEASON_ID"]) if pd.notna(r.get("SEASON_ID")) else None,
                "AWAY_NAME": r.get("AWAY_NAME"),
                "HOME_NAME": r.get("HOME_NAME"),
                "AWAY_ID": int(r["AWAY_ID"]) if pd.notna(r.get("AWAY_ID")) else None,
                "HOME_ID": int(r["HOME_ID"]) if pd.notna(r.get("HOME_ID")) else None,
                "GAME_STATUS": int(r["GAME_STATUS"]) if pd.notna(r.get("GAME_STATUS")) else None,
                "GAME_OUTCOME": int(r["GAME_OUTCOME"]) if pd.notna(r.get("GAME_OUTCOME")) else None,
                "GAME_DATE": game_date_iso,
            }
            try:
                supabase.table("games").upsert(payload, on_conflict="GAME_ID").execute()
                success_count += 1
            except Exception as e:
                logger.error(f"Upsert failed for GAME_ID {payload['GAME_ID']}: {e}")
            pbar.update(1)

    print(f"Upsert complete: {success_count}/{len(records)} games processed")


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "incremental"

    if mode == "full":
        print("\n=== Running FULL mode: historical backfill + current update ===")
        seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(2020, 2026)]
        hist_df = get_historical_games(seasons)
        if not hist_df.empty:
            upsert_games_to_supabase(hist_df)
    else:
        print("\n=== Running INCREMENTAL mode: current + recent delta check ===")
        today = datetime.now()
        recent_years = [today.year - 1, today.year]
        recent_seasons = [f"{y}-{str(y+1)[-2:]}" for y in recent_years if y >= 2020]
        recent_ids = recent_years

        new_df = get_historical_games(recent_seasons)
        db_df = fetch_db_games_for_seasons(recent_ids)

        for year in recent_years:
            season_str = f"{year}-{str(year+1)[-2:]}"
            api_count = len(new_df[new_df['SEASON_ID'] == year]) if not new_df.empty else 0
            db_count = len(db_df[db_df['SEASON_ID'] == year]) if not db_df.empty else 0
            print(f"{api_count} games found for {season_str} from NBA API")
            print(f"{db_count} games found for {season_str} in games table")

        if not new_df.empty:
            deltas = find_deltas(new_df, db_df)
            print(f"Deltas found: {len(deltas)}")
            if not deltas.empty:
                print(f"Upserting {len(deltas)} games")
                upsert_games_to_supabase(deltas)
            else:
                print("Upserting 0 games (no deltas)")
        else:
            print("Deltas found: 0")
            print("Upserting 0 games")

    print("\n=== Updating current/live/upcoming games ===")
    current_df = get_current_and_live_games()
    if not current_df.empty:
        print(f"Upserting {len(current_df)} games")
        upsert_games_to_supabase(current_df)
    else:
        print("0 games found for today's/current from scoreboard API")
        print("Upserting 0 games")


if __name__ == "__main__":
    main()
