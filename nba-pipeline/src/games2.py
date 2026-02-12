# games2.py
"""
NBA Games Pipeline v2:
Modes:
- full: backfill all regular season games 2020-21 to present
- incremental: process entire current season (delta upsert)
- current: process only games from last 2 days (default mode, delta upsert)
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from shared.nba.nba_constants import (
    TEAM_ABBR_TO_FULL,
    TEAM_SHORT_TO_FULL,
    TEAM_NAME_TO_ID,
    TEAM_ID_TO_NAME,
)

import os
import logging
import pandas as pd
import numpy as np
from supabase import create_client, Client
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv3, boxscoreadvancedv3, commonteamroster
from nba_api.live.nba.endpoints import scoreboard
from dotenv import load_dotenv
from tqdm import tqdm
from multiprocessing import Pool
import random
from time import sleep
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import functools

load_dotenv()

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

team_id_to_name = TEAM_ID_TO_NAME

def create_session_with_retries():
    session = requests.Session()
    retry_strategy = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount('https://', adapter)
    return session

def patch_requests_get():
    original_get = requests.get
    session = create_session_with_retries()
    @functools.wraps(original_get)
    def patched_get(*args, **kwargs):
        return session.get(*args, **kwargs)
    requests.get = patched_get
    return original_get

def restore_requests_get(original_get):
    requests.get = original_get

def _empty_game_row(game_id, game_date, season_id, home_name, home_id, away_name, away_id):
    return {'type': 'success', 'data': pd.DataFrame([{
        'game_id': game_id,
        'game_date': pd.to_datetime(game_date),
        'SEASON_ID': season_id,
        'home_team_name': home_name,
        'home_team_id': str(home_id),
        'away_team_name': away_name,
        'away_team_id': str(away_id),
        'home_team_score': np.nan,
        'away_team_score': np.nan,
        'total_points': np.nan,
        'home_team_players': [[]],
        'away_team_players': [[]],
        'player_stats': [[]]
    }])}

def process_game(args):
    game_id, game_date, home_team_id, away_team_id, SEASON_ID = args
    home_team_name = team_id_to_name.get(str(home_team_id), "Unknown")
    away_team_name = team_id_to_name.get(str(away_team_id), "Unknown")

    try:
        sleep(random.uniform(1.0, 2.0))

        trad_df = None
        try:
            boxscore_trad = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
            trad_frames = boxscore_trad.get_data_frames()
            if trad_frames and len(trad_frames) > 0 and len(trad_frames[0]) > 0:
                trad_df = trad_frames[0]
        except Exception:
            pass

        adv_df = None
        if trad_df is not None:
            try:
                boxscore_adv = boxscoreadvancedv3.BoxScoreAdvancedV3(game_id=game_id)
                adv_frames = boxscore_adv.get_data_frames()
                if adv_frames and len(adv_frames) > 0 and len(adv_frames[0]) > 0:
                    adv_df = adv_frames[0]
            except Exception:
                pass

            if any(col not in trad_df.columns for col in ['teamId', 'personId', 'points']):
                return _empty_game_row(game_id, game_date, SEASON_ID, home_team_name, home_team_id, away_team_name, away_team_id)

            home_team_score = trad_df[trad_df['teamId'] == int(home_team_id)]['points'].sum() or 0
            away_team_score = trad_df[trad_df['teamId'] == int(away_team_id)]['points'].sum() or 0
            total_points = home_team_score + away_team_score

            home_team_players = trad_df[trad_df['teamId'] == int(home_team_id)]['personId'].tolist()
            away_team_players = trad_df[trad_df['teamId'] == int(away_team_id)]['personId'].tolist()

            trad_columns = ['personId', 'teamId', 'points', 'reboundsTotal', 'assists', 'steals', 'blocks',
                            'turnovers', 'fieldGoalsMade', 'fieldGoalsAttempted', 'threePointersMade',
                            'threePointersAttempted', 'freeThrowsMade', 'freeThrowsAttempted']
            if 'minutesCalculated' in trad_df.columns:
                trad_columns.append('minutesCalculated')

            player_stats_df = trad_df[trad_columns].copy()
            if adv_df is not None:
                adv_cols = ['personId', 'teamId', 'offensiveRating', 'defensiveRating', 'usagePercentage']
                if all(col in adv_df.columns for col in adv_cols):
                    player_stats_df = pd.merge(player_stats_df, adv_df[adv_cols], on=['personId', 'teamId'], how='left')

            player_stats = [player_stats_df.to_dict('records')]

        else:
            season_year = SEASON_ID
            next_short = (season_year % 100) + 1
            season_str = f"{season_year}-{next_short:02d}"

            home_team_players = []
            away_team_players = []

            try:
                home_roster = commonteamroster.CommonTeamRoster(team_id=str(home_team_id), season=season_str, league_id_nullable="00")
                home_df = home_roster.get_data_frames()[0]
                home_team_players = home_df['PLAYER_ID'].astype(int).tolist()
            except Exception as e:
                logger.warning(f"Home roster failed {home_team_id}: {e}")

            try:
                away_roster = commonteamroster.CommonTeamRoster(team_id=str(away_team_id), season=season_str, league_id_nullable="00")
                away_df = away_roster.get_data_frames()[0]
                away_team_players = away_df['PLAYER_ID'].astype(int).tolist()
            except Exception as e:
                logger.warning(f"Away roster failed {away_team_id}: {e}")

            home_team_score = np.nan
            away_team_score = np.nan
            total_points = np.nan
            player_stats = [[]]

        game_info = {
            'game_id': game_id,
            'game_date': pd.to_datetime(game_date),
            'SEASON_ID': SEASON_ID,
            'home_team_name': home_team_name,
            'home_team_id': str(home_team_id),
            'away_team_name': away_team_name,
            'away_team_id': str(away_team_id),
            'home_team_score': home_team_score,
            'away_team_score': away_team_score,
            'total_points': total_points,
            'home_team_players': [home_team_players],
            'away_team_players': [away_team_players],
            'player_stats': player_stats
        }
        return {'type': 'success', 'data': pd.DataFrame([game_info])}

    except Exception as e:
        logger.error(f"Critical error processing {game_id}: {e}")
        return {'type': 'error', 'data': {'game_id': game_id, 'SEASON_ID': SEASON_ID, 'error': str(e)}}

def get_historical_games(seasons_to_fetch):
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
                print(f"0 games found for {season}")
                continue

            df['GAME_ID'] = pd.to_numeric(df['GAME_ID'], errors='coerce').astype('Int64')

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

            # Outcome from WL (exact original logic)
            def get_outcome(game_id):
                game_rows = df[df['GAME_ID'] == game_id]
                if len(game_rows) == 0:
                    return None
                row = game_rows.iloc[0]
                if 'WL' not in row or pd.isna(row['WL']):
                    return None
                is_home = 'vs.' in row['MATCHUP']
                if row['WL'] == 'W':
                    return GAME_OUTCOME_HOME_WIN if is_home else GAME_OUTCOME_AWAY_WIN
                elif row['WL'] == 'L':
                    return GAME_OUTCOME_AWAY_WIN if is_home else GAME_OUTCOME_HOME_WIN
                return None

            games['GAME_OUTCOME'] = games['GAME_ID'].apply(get_outcome)
            games['GAME_STATUS'] = np.where(games['GAME_OUTCOME'].notna(), GAME_STATUS_FINAL, GAME_STATUS_SCHEDULED)

            # Attempt PTS safely (only if present)
            def get_pts(game_id):
                game_rows = df[df['GAME_ID'] == game_id]
                pts = game_rows['PTS'].dropna()
                if len(pts) == 2:
                    # Assume first row home if 'vs.', else reverse
                    if 'vs.' in game_rows.iloc[0]['MATCHUP']:
                        return int(pts.iloc[0]), int(pts.iloc[1]), int(pts.sum())
                    else:
                        return int(pts.iloc[1]), int(pts.iloc[0]), int(pts.sum())
                return None, None, None

            pts_stats = games['GAME_ID'].apply(get_pts)
            games['HOME_PTS'] = [s[0] for s in pts_stats]
            games['AWAY_PTS'] = [s[1] for s in pts_stats]
            games['TOTAL_PTS'] = [s[2] for s in pts_stats]

            keep = ["GAME_ID", "SEASON_ID", "GAME_DATE", "AWAY_NAME", "HOME_NAME", "AWAY_ID", "HOME_ID",
                    "GAME_STATUS", "GAME_OUTCOME", "HOME_PTS", "AWAY_PTS", "TOTAL_PTS"]
            games = games[keep]
            games["SEASON_ID"] = games["SEASON_ID"].astype(str).str[-4:].astype(int)

            print(f"{len(games)} games found for {season}")
            all_rows.append(games)
        except Exception as e:
            logger.error(f"Failed to fetch/process {season}: {e}")
            continue

    if all_rows:
        return pd.concat(all_rows, ignore_index=True).drop_duplicates("GAME_ID")
    return pd.DataFrame()

def get_games_by_date_range(date_from, date_to):
    try:
        finder = leaguegamefinder.LeagueGameFinder(
            date_from_nullable=date_from.strftime('%m/%d/%Y'),
            date_to_nullable=date_to.strftime('%m/%d/%Y'),
            league_id_nullable="00",
            season_type_nullable="Regular Season"
        )
        df = finder.get_data_frames()[0]
        if df.empty:
            return pd.DataFrame()

        # Reuse same logic as historical for consistency
        return get_historical_games([get_current_season_str()])  # fallback to season logic for recent dates

    except Exception as e:
        logger.error(f"Date range fetch failed {date_from} to {date_to}: {e}")
        return pd.DataFrame()

def get_current_metadata():
    try:
        sb = scoreboard.ScoreBoard()
        games_data = sb.games.get_dict()
        rows = []
        for g in games_data:
            gid = int(g['gameId'])
            game_time_utc = g.get('gameTimeUTC')
            game_date = pd.to_datetime(game_time_utc) if game_time_utc else pd.NaT
            season_year = g.get('seasonYear') or int(str(g['gameId'])[3:5]) + 2000
            rows.append({
                "GAME_ID": gid,
                "SEASON_ID": season_year,
                "GAME_DATE": game_date,
                "AWAY_NAME": TEAM_SHORT_TO_FULL.get(g['awayTeam']['teamName'], g['awayTeam']['teamName']),
                "HOME_NAME": TEAM_SHORT_TO_FULL.get(g['homeTeam']['teamName'], g['homeTeam']['teamName']),
                "AWAY_ID": int(g['awayTeam']['teamId']),
                "HOME_ID": int(g['homeTeam']['teamId']),
            })
        df = pd.DataFrame(rows)
        print(f"{len(df)} current/upcoming games from scoreboard")
        return df
    except Exception as e:
        logger.error(f"Scoreboard fetch failed: {e}")
        return pd.DataFrame()

def get_today_games_dict():
    try:
        sb = scoreboard.ScoreBoard()
        games_data = sb.games.get_dict()
        today_dict = {}
        for g in games_data:
            gid = int(g['gameId'])
            status_text = g.get('gameStatusText', '')
            status = g.get('gameStatus', GAME_STATUS_SCHEDULED)
            if 'Final' in status_text:
                status = GAME_STATUS_FINAL
            elif any(q in status_text for q in ['Q','OT','Half']):
                status = GAME_STATUS_LIVE
            outcome = None
            home_score = g.get('homeTeam', {}).get('score', 0)
            away_score = g.get('awayTeam', {}).get('score', 0)
            if status == GAME_STATUS_FINAL:
                if home_score > away_score:
                    outcome = GAME_OUTCOME_HOME_WIN
                elif away_score > home_score:
                    outcome = GAME_OUTCOME_AWAY_WIN
            total = home_score + away_score if status == GAME_STATUS_FINAL else np.nan
            today_dict[gid] = {
                'GAME_STATUS': status,
                'GAME_OUTCOME': outcome,
                'HOME_PTS': home_score if status == GAME_STATUS_FINAL else np.nan,
                'AWAY_PTS': away_score if status == GAME_STATUS_FINAL else np.nan,
                'TOTAL_PTS': total,
            }
        return today_dict
    except Exception as e:
        logger.error(f"Scoreboard status fetch failed: {e}")
        return {}

def process_games_to_full(metadata_df):
    if metadata_df.empty:
        return pd.DataFrame()

    args = []
    for _, row in metadata_df.iterrows():
        args.append((
            str(row['GAME_ID']),
            str(row['GAME_DATE']),
            int(row['HOME_ID']) if pd.notna(row['HOME_ID']) else None,
            int(row['AWAY_ID']) if pd.notna(row['AWAY_ID']) else None,
            int(row['SEASON_ID'])
        ))

    original_get = patch_requests_get()
    try:
        game_logs = []
        errors = []

        seasons = sorted(metadata_df['SEASON_ID'].unique())
        for SEASON_ID in seasons:
            season_args = [a for a in args if a[4] == SEASON_ID]
            logger.info(f"Season {SEASON_ID}: {len(season_args)} games")

            season_logs = []
            batch_size = 500
            for batch_start in range(0, len(season_args), batch_size):
                batch_args = season_args[batch_start:batch_start + batch_size]
                logger.info(f"  Season {SEASON_ID} Batch {batch_start // batch_size + 1} ({len(batch_args)} games)")
                with Pool(8) as pool:
                    batch_results = list(tqdm(
                        pool.imap_unordered(process_game, batch_args),
                        total=len(batch_args),
                        desc=f"Season {SEASON_ID} Batch {batch_start // batch_size + 1}"
                    ))

                batch_success = [r['data'] for r in batch_results if r['type'] == 'success' and not r['data'].empty]
                batch_errors = [r['data'] for r in batch_results if r['type'] == 'error']
                errors.extend(batch_errors)

                if batch_success:
                    season_logs.append(pd.concat(batch_success, ignore_index=True))

            if season_logs:
                game_logs.append(pd.concat(season_logs, ignore_index=True))

        if game_logs:
            return pd.concat(game_logs, ignore_index=True)
        return pd.DataFrame()
    finally:
        restore_requests_get(original_get)

def fetch_db_games_for_seasons(season_ids):
    try:
        response = supabase.table("games").select("*").in_("SEASON_ID", season_ids).execute()
        if response.data:
            db_df = pd.DataFrame(response.data)
            db_df['GAME_ID'] = pd.to_numeric(db_df['GAME_ID'], errors='coerce').astype('Int64')
            db_df['GAME_DATE'] = pd.to_datetime(db_df['GAME_DATE'], errors='coerce', utc=True)
            return db_df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to fetch DB games: {e}")
        return pd.DataFrame()

def find_deltas(new_df, db_df):
    if db_df.empty:
        return new_df
    for df in [new_df, db_df]:
        df['GAME_ID'] = pd.to_numeric(df['GAME_ID'], errors='coerce').astype('Int64')
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='coerce', utc=True)
    merged = new_df.merge(db_df, on="GAME_ID", suffixes=("_new", "_db"), how="left")
    compare_cols = ["SEASON_ID", "GAME_DATE", "AWAY_NAME", "HOME_NAME", "AWAY_ID", "HOME_ID",
                    "GAME_STATUS", "GAME_OUTCOME", "AWAY_PTS", "HOME_PTS", "TOTAL_PTS",
                    "HOME_TEAM_PLAYERS", "AWAY_TEAM_PLAYERS"]
    delta_mask = merged['SEASON_ID_db'].isna()
    for col in compare_cols:
        n_col = f"{col}_new"
        db_col = f"{col}_db"
        if col == "GAME_DATE":
            delta_mask |= (merged[n_col].dt.date.ne(merged[db_col].dt.date)) & (merged[n_col].notna() | merged[db_col].notna())
        else:
            delta_mask |= (merged[n_col].ne(merged[db_col])) & (merged[n_col].notna() | merged[db_col].notna())
    new_cols = [c for c in merged.columns if c.endswith('_new')]
    deltas = merged.loc[delta_mask, new_cols + ['GAME_ID']].copy()
    deltas.rename(columns={c: c.replace('_new', '') for c in new_cols}, inplace=True)
    return deltas

def upsert_games_to_supabase(df):
    if df.empty:
        print("Upserting 0 games")
        return
    for col in ['GAME_ID', 'SEASON_ID', 'HOME_ID', 'AWAY_ID', 'GAME_STATUS', 'GAME_OUTCOME', 'HOME_PTS', 'AWAY_PTS', 'TOTAL_PTS']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
    records = df.to_dict(orient="records")
    success = 0
    with tqdm(total=len(records), desc="Upserting") as pbar:
        for r in records:
            payload = {
                "GAME_ID": int(r["GAME_ID"]),
                "SEASON_ID": int(r["SEASON_ID"]) if pd.notna(r.get("SEASON_ID")) else None,
                "AWAY_NAME": r.get("AWAY_NAME"),
                "HOME_NAME": r.get("HOME_NAME"),
                "AWAY_ID": int(r["AWAY_ID"]) if pd.notna(r.get("AWAY_ID")) else None,
                "HOME_ID": int(r["HOME_ID"]) if pd.notna(r.get("HOME_ID")) else None,
                "GAME_STATUS": int(r.get("GAME_STATUS", GAME_STATUS_SCHEDULED)),
                "GAME_OUTCOME": int(r["GAME_OUTCOME"]) if pd.notna(r.get("GAME_OUTCOME")) else None,
                "GAME_DATE": pd.Timestamp(r["GAME_DATE"]).isoformat() if pd.notna(r.get("GAME_DATE")) else None,
                "AWAY_PTS": int(r["AWAY_PTS"]) if pd.notna(r.get("AWAY_PTS")) else None,
                "HOME_PTS": int(r["HOME_PTS"]) if pd.notna(r.get("HOME_PTS")) else None,
                "TOTAL_PTS": int(r["TOTAL_PTS"]) if pd.notna(r.get("TOTAL_PTS")) else None,
                "HOME_TEAM_PLAYERS": r.get("HOME_TEAM_PLAYERS", []),
                "AWAY_TEAM_PLAYERS": r.get("AWAY_TEAM_PLAYERS", []),
            }
            try:
                supabase.table("games").upsert(payload, on_conflict="GAME_ID").execute()
                success += 1
            except Exception as e:
                logger.error(f"Upsert failed GAME_ID {payload['GAME_ID']}: {e}")
            pbar.update(1)
    print(f"Upserted {success}/{len(records)} games")

def get_current_season_str():
    today = datetime.now()
    year = today.year
    if today.month < 7:
        year -= 1
    return f"{year}-{str(year+1)[-2:]}"

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "current"

    now = datetime.now()
    current_season = get_current_season_str()

    if mode == "full":
        print("\n=== FULL MODE: backfill 2020-21 to present ===")
        seasons = [f"{y}-{str(y+1)[-2:]}" for y in range(2020, now.year + 1)]
        if now.month >= 10:
            seasons.append(f"{now.year}-{str(now.year+1)[-2:]}")
        metadata_df = get_historical_games(seasons)

    elif mode == "incremental":
        print(f"\n=== INCREMENTAL MODE: current season {current_season} (delta) ===")
        metadata_df = get_historical_games([current_season])

    else:  # current (default)
        print("\n=== CURRENT MODE: last 2 days (delta) ===")
        date_from = (now - timedelta(days=2)).date()
        date_to = now.date()
        metadata_df = get_games_by_date_range(date_from, date_to)
        current_df = get_current_metadata()
        if not current_df.empty:
            metadata_df = pd.concat([metadata_df, current_df], ignore_index=True).drop_duplicates("GAME_ID")

    if metadata_df.empty:
        print("No games fetched")
        return

    full_df = process_games_to_full(metadata_df)

    if full_df.empty:
        print("No boxscore data processed")
        return

    full_df.rename(columns={
        'game_id': 'GAME_ID',
        'game_date': 'GAME_DATE',
        'home_team_name': 'HOME_NAME',
        'away_team_name': 'AWAY_NAME',
        'home_team_id': 'HOME_ID',
        'away_team_id': 'AWAY_ID',
        'home_team_score': 'HOME_PTS',
        'away_team_score': 'AWAY_PTS',
        'total_points': 'TOTAL_PTS',
    }, inplace=True)

    full_df['GAME_ID'] = pd.to_numeric(full_df['GAME_ID'], errors='coerce').astype('Int64')
    full_df['HOME_ID'] = pd.to_numeric(full_df['HOME_ID'], errors='coerce').astype('Int64')
    full_df['AWAY_ID'] = pd.to_numeric(full_df['AWAY_ID'], errors='coerce').astype('Int64')
    full_df['HOME_PTS'] = pd.to_numeric(full_df['HOME_PTS'], errors='coerce').astype('Int64')
    full_df['AWAY_PTS'] = pd.to_numeric(full_df['AWAY_PTS'], errors='coerce').astype('Int64')
    full_df['TOTAL_PTS'] = pd.to_numeric(full_df['TOTAL_PTS'], errors='coerce').astype('Int64')

    full_df['HOME_TEAM_PLAYERS'] = full_df['home_team_players'].apply(lambda x: x[0] if isinstance(x, list) and x else [])
    full_df['AWAY_TEAM_PLAYERS'] = full_df['away_team_players'].apply(lambda x: x[0] if isinstance(x, list) and x else [])
    full_df.drop(columns=['home_team_players', 'away_team_players', 'player_stats'], errors='ignore', inplace=True)

    today_dict = get_today_games_dict()
    for idx, row in full_df.iterrows():
        gid = int(row['GAME_ID'])
        if gid in today_dict:
            s = today_dict[gid]
            full_df.at[idx, 'GAME_STATUS'] = s['GAME_STATUS']
            full_df.at[idx, 'GAME_OUTCOME'] = s['GAME_OUTCOME']
            full_df.at[idx, 'HOME_PTS'] = s['HOME_PTS']
            full_df.at[idx, 'AWAY_PTS'] = s['AWAY_PTS']
            full_df.at[idx, 'TOTAL_PTS'] = s['TOTAL_PTS']

    if mode == "full":
        upsert_games_to_supabase(full_df)
    else:
        seasons = full_df['SEASON_ID'].unique().tolist()
        db_df = fetch_db_games_for_seasons(seasons)
        deltas = find_deltas(full_df, db_df)
        print(f"Found {len(deltas)} deltas to upsert")
        upsert_games_to_supabase(deltas)

if __name__ == "__main__":
    main()
