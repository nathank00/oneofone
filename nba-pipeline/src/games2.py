<<<<<<< HEAD
# games2.py
"""
NBA Games Pipeline v2:
Modes:
- full: backfill all regular season games 2020-21 to present
- incremental: process entire current season (delta upsert)
- current: process only games from last 2 days (default mode, delta upsert)
=======
# nba-pipeline/src/games2.py
"""
NBA Games Pipeline v2:
Modes:
- full: backfill all regular season games 2020-21 to present (full upsert/overwrite)
- incremental: refresh current season rows (delta upsert)
- current: games from 3 days ago to today (delta upsert on diffs only)

Data sources:
- LeagueGameFinder: bulk game metadata + scores for completed games (1 call per season)
- ScheduleLeagueV2: full season schedule including future/upcoming games
- ScoreboardV2: today's games with live scores and game status
- CommonTeamRoster: roster player IDs per team per season (30 calls per season)
>>>>>>> claude/festive-noyce
"""

import sys
from pathlib import Path
<<<<<<< HEAD
from datetime import datetime, timedelta
=======
from datetime import datetime, timedelta, timezone
>>>>>>> claude/festive-noyce

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
<<<<<<< HEAD
from nba_api.stats.endpoints import leaguegamefinder, boxscoretraditionalv3, boxscoreadvancedv3, commonteamroster
from nba_api.live.nba.endpoints import scoreboard
=======
from nba_api.stats.endpoints import (
    leaguegamefinder,
    scoreboardv2,
    scheduleleaguev2,
    commonteamroster,
)
>>>>>>> claude/festive-noyce
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
<<<<<<< HEAD
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
=======
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
>>>>>>> claude/festive-noyce
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase env vars")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

GAME_STATUS_SCHEDULED = 1
<<<<<<< HEAD
GAME_STATUS_LIVE      = 2
GAME_STATUS_FINAL     = 3
=======
GAME_STATUS_LIVE = 2
GAME_STATUS_FINAL = 3
>>>>>>> claude/festive-noyce
GAME_STATUS_POSTPONED = 4

GAME_OUTCOME_HOME_WIN = 1
GAME_OUTCOME_AWAY_WIN = 0

<<<<<<< HEAD
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
=======
# All 30 NBA team IDs for roster fetching
ALL_TEAM_IDS = list(TEAM_NAME_TO_ID.values())


# ---------------------------------------------------------------------------
# Rate-limit-safe HTTP session (retries on 429/5xx with backoff)
# ---------------------------------------------------------------------------
def create_session_with_retries():
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
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


# ---------------------------------------------------------------------------
# Season helpers
# ---------------------------------------------------------------------------
def get_current_season_str():
    """Return current NBA season string like '2024-25'."""
    today = datetime.now()
    year = today.year
    if today.month < 7:
        year -= 1
    return f"{year}-{str(year + 1)[-2:]}"


def get_current_season_year():
    """Return start year of the current NBA season (e.g. 2024 for 2024-25)."""
    today = datetime.now()
    year = today.year
    if today.month < 7:
        year -= 1
    return year


def season_str_to_year(season_str):
    """'2024-25' -> 2024"""
    return int(season_str.split("-")[0])


# ---------------------------------------------------------------------------
# 1. LeagueGameFinder — bulk completed games with scores (1 call per season)
#    Returns 2 rows per game (one per team). We pivot to 1 row per game.
# ---------------------------------------------------------------------------
def fetch_completed_games_for_season(season_str):
    """
    Fetch all completed regular season games for a season using LeagueGameFinder.
    Returns a DataFrame with one row per game, including scores and outcome.
    """
    try:
        sleep(random.uniform(0.8, 1.5))
        finder = leaguegamefinder.LeagueGameFinder(
            season_nullable=season_str,
            league_id_nullable="00",
            season_type_nullable="Regular Season",
        )
        raw = finder.get_data_frames()[0]
        if raw.empty:
            logger.info(f"  0 games from LeagueGameFinder for {season_str}")
            return pd.DataFrame()

        raw["GAME_ID"] = pd.to_numeric(raw["GAME_ID"], errors="coerce").astype("Int64")

        # Each game has 2 rows (home team row has "vs.", away team row has "@")
        home_rows = raw[raw["MATCHUP"].str.contains("vs.", na=False)].copy()
        away_rows = raw[raw["MATCHUP"].str.contains("@", na=False)].copy()

        if home_rows.empty and away_rows.empty:
            logger.warning(f"  No parseable matchup rows for {season_str}")
            return pd.DataFrame()

        # Build home side
        home = home_rows[["GAME_ID", "GAME_DATE", "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS", "WL"]].copy()
        home.rename(columns={
            "TEAM_ID": "HOME_ID",
            "TEAM_ABBREVIATION": "HOME_ABBR",
            "PTS": "HOME_PTS",
            "WL": "HOME_WL",
        }, inplace=True)

        # Build away side
        away = away_rows[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS", "WL"]].copy()
        away.rename(columns={
            "TEAM_ID": "AWAY_ID",
            "TEAM_ABBREVIATION": "AWAY_ABBR",
            "PTS": "AWAY_PTS",
            "WL": "AWAY_WL",
        }, inplace=True)

        # Merge on GAME_ID to get 1 row per game
        games = home.merge(away, on="GAME_ID", how="outer")

        # Resolve team names
        games["HOME_NAME"] = games["HOME_ABBR"].map(TEAM_ABBR_TO_FULL)
        games["AWAY_NAME"] = games["AWAY_ABBR"].map(TEAM_ABBR_TO_FULL)

        # Parse SEASON_ID to just the start year (e.g. "22024" -> 2024)
        games["SEASON_ID"] = games["SEASON_ID"].astype(str).str[-4:].astype("Int64")

        games["GAME_DATE"] = pd.to_datetime(games["GAME_DATE"], errors="coerce")

        # Scores
        games["HOME_PTS"] = pd.to_numeric(games["HOME_PTS"], errors="coerce").astype("Int64")
        games["AWAY_PTS"] = pd.to_numeric(games["AWAY_PTS"], errors="coerce").astype("Int64")
        games["TOTAL_PTS"] = games["HOME_PTS"] + games["AWAY_PTS"]
        games["TOTAL_PTS"] = games["TOTAL_PTS"].astype("Int64")

        # IDs
        games["HOME_ID"] = pd.to_numeric(games["HOME_ID"], errors="coerce").astype("Int64")
        games["AWAY_ID"] = pd.to_numeric(games["AWAY_ID"], errors="coerce").astype("Int64")

        # Outcome
        def derive_outcome(row):
            if pd.notna(row.get("HOME_WL")):
                if row["HOME_WL"] == "W":
                    return GAME_OUTCOME_HOME_WIN
                elif row["HOME_WL"] == "L":
                    return GAME_OUTCOME_AWAY_WIN
            return None

        games["GAME_OUTCOME"] = games.apply(derive_outcome, axis=1)
        games["GAME_STATUS"] = np.where(games["GAME_OUTCOME"].notna(), GAME_STATUS_FINAL, GAME_STATUS_SCHEDULED)
        games["GAME_STATUS"] = games["GAME_STATUS"].astype("Int64")

        keep = [
            "GAME_ID", "SEASON_ID", "GAME_DATE",
            "AWAY_NAME", "HOME_NAME", "AWAY_ID", "HOME_ID",
            "GAME_STATUS", "GAME_OUTCOME",
            "HOME_PTS", "AWAY_PTS", "TOTAL_PTS",
        ]
        games = games[[c for c in keep if c in games.columns]]
        games = games.drop_duplicates("GAME_ID")

        logger.info(f"  {len(games)} completed games from LeagueGameFinder for {season_str}")
        return games

    except Exception as e:
        logger.error(f"  LeagueGameFinder failed for {season_str}: {e}")
        return pd.DataFrame()


def fetch_completed_games(season_list):
    """Fetch completed games for multiple seasons (sequential, 1 API call each)."""
    all_dfs = []
    for season_str in season_list:
        df = fetch_completed_games_for_season(season_str)
        if not df.empty:
            all_dfs.append(df)
    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True).drop_duplicates("GAME_ID")
    return pd.DataFrame()


# ---------------------------------------------------------------------------
# 2. ScheduleLeagueV2 — full season schedule including future games
# ---------------------------------------------------------------------------
def fetch_schedule_for_season(season_str):
    """
    Fetch the full season schedule using ScheduleLeagueV2.
    Includes upcoming/scheduled games that LeagueGameFinder won't have.
    """
    try:
        sleep(random.uniform(0.8, 1.5))
        sched = scheduleleaguev2.ScheduleLeagueV2(
            season=season_str,
            league_id="00",
        )
        frames = sched.get_data_frames()
        if not frames or frames[0].empty:
            logger.info(f"  0 games from ScheduleLeagueV2 for {season_str}")
            return pd.DataFrame()

        raw = frames[0]

        games = pd.DataFrame()
        games["GAME_ID"] = pd.to_numeric(raw["gameId"], errors="coerce").astype("Int64")
        games["SEASON_ID"] = season_str_to_year(season_str)
        games["GAME_DATE"] = pd.to_datetime(raw["gameDateTimeUTC"], errors="coerce", utc=True)

        # Team info
        games["HOME_ID"] = pd.to_numeric(raw["homeTeam_teamId"], errors="coerce").astype("Int64")
        games["AWAY_ID"] = pd.to_numeric(raw["awayTeam_teamId"], errors="coerce").astype("Int64")

        # Resolve full names from team ID
        id_to_name = {int(v): k for k, v in TEAM_NAME_TO_ID.items()}
        games["HOME_NAME"] = games["HOME_ID"].map(id_to_name)
        games["AWAY_NAME"] = games["AWAY_ID"].map(id_to_name)

        # Game status from the schedule endpoint
        games["GAME_STATUS"] = pd.to_numeric(raw["gameStatus"], errors="coerce").astype("Int64")

        # Scores (0 for scheduled games)
        home_score = pd.to_numeric(raw["homeTeam_score"], errors="coerce")
        away_score = pd.to_numeric(raw["awayTeam_score"], errors="coerce")

        games["HOME_PTS"] = np.where(games["GAME_STATUS"] == GAME_STATUS_FINAL, home_score, np.nan)
        games["AWAY_PTS"] = np.where(games["GAME_STATUS"] == GAME_STATUS_FINAL, away_score, np.nan)
        games["HOME_PTS"] = pd.to_numeric(games["HOME_PTS"], errors="coerce").astype("Int64")
        games["AWAY_PTS"] = pd.to_numeric(games["AWAY_PTS"], errors="coerce").astype("Int64")
        games["TOTAL_PTS"] = (games["HOME_PTS"] + games["AWAY_PTS"]).astype("Int64")

        # Outcome for final games
        def derive_outcome(row):
            if row["GAME_STATUS"] == GAME_STATUS_FINAL:
                if pd.notna(row["HOME_PTS"]) and pd.notna(row["AWAY_PTS"]):
                    if row["HOME_PTS"] > row["AWAY_PTS"]:
                        return GAME_OUTCOME_HOME_WIN
                    elif row["AWAY_PTS"] > row["HOME_PTS"]:
                        return GAME_OUTCOME_AWAY_WIN
            return None

        games["GAME_OUTCOME"] = games.apply(derive_outcome, axis=1)

        # Handle postponed
        if "postponedStatus" in raw.columns:
            postponed_mask = raw["postponedStatus"].notna() & (raw["postponedStatus"] != "")
            games.loc[postponed_mask, "GAME_STATUS"] = GAME_STATUS_POSTPONED

        games = games.drop_duplicates("GAME_ID")
        logger.info(f"  {len(games)} games from ScheduleLeagueV2 for {season_str}")
        return games

    except Exception as e:
        logger.error(f"  ScheduleLeagueV2 failed for {season_str}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 3. ScoreboardV2 — today's games with live status/scores
# ---------------------------------------------------------------------------
def fetch_scoreboard_for_date(game_date):
    """
    Fetch games for a specific date using ScoreboardV2.
    Returns game metadata with status and scores.
    """
    try:
        sleep(random.uniform(0.8, 1.5))
        date_str = game_date.strftime("%Y-%m-%d")
        sb = scoreboardv2.ScoreboardV2(game_date=date_str, league_id="00")
        frames = sb.get_data_frames()

        game_header = frames[0]  # GameHeader
        line_score = frames[1]   # LineScore

        if game_header.empty:
            return pd.DataFrame()

        games = pd.DataFrame()
        games["GAME_ID"] = pd.to_numeric(game_header["GAME_ID"], errors="coerce").astype("Int64")
        games["GAME_DATE"] = pd.to_datetime(game_header["GAME_DATE_EST"], errors="coerce")
        games["HOME_ID"] = pd.to_numeric(game_header["HOME_TEAM_ID"], errors="coerce").astype("Int64")
        games["AWAY_ID"] = pd.to_numeric(game_header["VISITOR_TEAM_ID"], errors="coerce").astype("Int64")
        games["GAME_STATUS"] = pd.to_numeric(game_header["GAME_STATUS_ID"], errors="coerce").astype("Int64")

        # Derive season from game date
        gd = game_date
        season_year = gd.year if gd.month >= 7 else gd.year - 1
        games["SEASON_ID"] = season_year

        # Resolve team names from ID
        id_to_name = {int(v): k for k, v in TEAM_NAME_TO_ID.items()}
        games["HOME_NAME"] = games["HOME_ID"].map(id_to_name)
        games["AWAY_NAME"] = games["AWAY_ID"].map(id_to_name)

        # Extract scores from LineScore (2 rows per game: home and visitor)
        if not line_score.empty and "PTS" in line_score.columns:
            line_score["GAME_ID"] = pd.to_numeric(line_score["GAME_ID"], errors="coerce").astype("Int64")
            line_score["TEAM_ID"] = pd.to_numeric(line_score["TEAM_ID"], errors="coerce").astype("Int64")
            line_score["PTS"] = pd.to_numeric(line_score["PTS"], errors="coerce")

            for idx, row in games.iterrows():
                gid = row["GAME_ID"]
                ls_game = line_score[line_score["GAME_ID"] == gid]
                if len(ls_game) == 2:
                    home_ls = ls_game[ls_game["TEAM_ID"] == row["HOME_ID"]]
                    away_ls = ls_game[ls_game["TEAM_ID"] == row["AWAY_ID"]]
                    if not home_ls.empty and not away_ls.empty:
                        h_pts = home_ls.iloc[0]["PTS"]
                        a_pts = away_ls.iloc[0]["PTS"]
                        if row["GAME_STATUS"] == GAME_STATUS_FINAL:
                            games.at[idx, "HOME_PTS"] = h_pts
                            games.at[idx, "AWAY_PTS"] = a_pts
                            games.at[idx, "TOTAL_PTS"] = h_pts + a_pts if pd.notna(h_pts) and pd.notna(a_pts) else np.nan

        # Ensure numeric types
        for col in ["HOME_PTS", "AWAY_PTS", "TOTAL_PTS"]:
            if col in games.columns:
                games[col] = pd.to_numeric(games[col], errors="coerce").astype("Int64")
            else:
                games[col] = pd.array([pd.NA] * len(games), dtype="Int64")

        # Outcome
        def derive_outcome(row):
            if row["GAME_STATUS"] == GAME_STATUS_FINAL:
                if pd.notna(row.get("HOME_PTS")) and pd.notna(row.get("AWAY_PTS")):
                    if row["HOME_PTS"] > row["AWAY_PTS"]:
                        return GAME_OUTCOME_HOME_WIN
                    elif row["AWAY_PTS"] > row["HOME_PTS"]:
                        return GAME_OUTCOME_AWAY_WIN
            return None

        games["GAME_OUTCOME"] = games.apply(derive_outcome, axis=1)

        games = games.drop_duplicates("GAME_ID")
        logger.info(f"  {len(games)} games from ScoreboardV2 for {date_str}")
        return games

    except Exception as e:
        logger.error(f"  ScoreboardV2 failed for {game_date}: {e}")
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# 4. CommonTeamRoster — player IDs per team per season
#    Uses the same batching/multiprocessing pattern for rate-limit safety.
# ---------------------------------------------------------------------------
def _fetch_roster_for_team(args):
    """Fetch roster for a single team. Used in multiprocessing pool."""
    team_id, season_str = args
    try:
        sleep(random.uniform(1.0, 2.0))
        roster = commonteamroster.CommonTeamRoster(
            team_id=str(team_id),
            season=season_str,
            league_id_nullable="00",
        )
        df = roster.get_data_frames()[0]
        if df.empty:
            return {"team_id": int(team_id), "player_ids": []}
        player_ids = df["PLAYER_ID"].astype(int).tolist()
        return {"team_id": int(team_id), "player_ids": player_ids}
    except Exception as e:
        logger.warning(f"  Roster fetch failed team_id={team_id} season={season_str}: {e}")
        return {"team_id": int(team_id), "player_ids": []}


def fetch_rosters_for_season(season_str):
    """
    Fetch rosters for all 30 teams for a given season.
    Returns dict: {team_id: [player_id, ...]}.
    Uses multiprocessing pool with rate-limit-safe batching.
    """
    args = [(tid, season_str) for tid in ALL_TEAM_IDS]

    original_get = patch_requests_get()
    try:
        rosters = {}
        with Pool(8) as pool:
            results = list(tqdm(
                pool.imap_unordered(_fetch_roster_for_team, args),
                total=len(args),
                desc=f"Rosters {season_str}",
            ))
        for r in results:
            rosters[r["team_id"]] = r["player_ids"]
        return rosters
    finally:
        restore_requests_get(original_get)


def attach_rosters(games_df, season_rosters):
    """
    Attach HOME_TEAM_PLAYERS and AWAY_TEAM_PLAYERS columns to games_df
    using the pre-fetched roster dict: {season_year: {team_id: [player_ids]}}.
    """
    home_players = []
    away_players = []

    for _, row in games_df.iterrows():
        season_year = int(row["SEASON_ID"]) if pd.notna(row["SEASON_ID"]) else None
        home_id = int(row["HOME_ID"]) if pd.notna(row["HOME_ID"]) else None
        away_id = int(row["AWAY_ID"]) if pd.notna(row["AWAY_ID"]) else None

        rosters = season_rosters.get(season_year, {})
        home_players.append(rosters.get(home_id, []))
        away_players.append(rosters.get(away_id, []))

    games_df["HOME_TEAM_PLAYERS"] = home_players
    games_df["AWAY_TEAM_PLAYERS"] = away_players
    return games_df


# ---------------------------------------------------------------------------
# 5. Supabase DB interaction
# ---------------------------------------------------------------------------
def fetch_db_games(season_ids=None, game_ids=None):
    """Fetch existing games from Supabase, filtered by season or game IDs."""
    try:
        query = supabase.table("games").select("*")
        if season_ids:
            query = query.in_("SEASON_ID", [int(s) for s in season_ids])
        elif game_ids:
            query = query.in_("GAME_ID", [int(g) for g in game_ids])

        response = query.execute()
        if response.data:
            db_df = pd.DataFrame(response.data)
            db_df["GAME_ID"] = pd.to_numeric(db_df["GAME_ID"], errors="coerce").astype("Int64")
            db_df["GAME_DATE"] = pd.to_datetime(db_df["GAME_DATE"], errors="coerce", utc=True)
>>>>>>> claude/festive-noyce
            return db_df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to fetch DB games: {e}")
        return pd.DataFrame()

<<<<<<< HEAD
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
=======

def find_deltas(new_df, db_df):
    """Find rows in new_df that differ from db_df or are missing in DB."""
    if db_df.empty:
        return new_df

    for df in [new_df, db_df]:
        df["GAME_ID"] = pd.to_numeric(df["GAME_ID"], errors="coerce").astype("Int64")
        if "GAME_DATE" in df.columns:
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce", utc=True)

    merged = new_df.merge(db_df, on="GAME_ID", suffixes=("_new", "_db"), how="left")

    compare_cols = [
        "SEASON_ID", "GAME_DATE", "AWAY_NAME", "HOME_NAME", "AWAY_ID", "HOME_ID",
        "GAME_STATUS", "GAME_OUTCOME", "AWAY_PTS", "HOME_PTS", "TOTAL_PTS",
        "HOME_TEAM_PLAYERS", "AWAY_TEAM_PLAYERS",
    ]

    # Start with new rows (no match in DB)
    delta_mask = merged["SEASON_ID_db"].isna()

    for col in compare_cols:
        n_col = f"{col}_new"
        db_col = f"{col}_db"
        if n_col not in merged.columns or db_col not in merged.columns:
            continue

        if col == "GAME_DATE":
            delta_mask |= (
                merged[n_col].dt.date.ne(merged[db_col].dt.date)
                & (merged[n_col].notna() | merged[db_col].notna())
            )
        elif col in ("HOME_TEAM_PLAYERS", "AWAY_TEAM_PLAYERS"):
            # List comparison: convert to string for comparison
            delta_mask |= (
                merged[n_col].astype(str).ne(merged[db_col].astype(str))
                & (merged[n_col].notna() | merged[db_col].notna())
            )
        else:
            delta_mask |= (
                merged[n_col].ne(merged[db_col])
                & (merged[n_col].notna() | merged[db_col].notna())
            )

    new_cols = [c for c in merged.columns if c.endswith("_new")]
    deltas = merged.loc[delta_mask, new_cols + ["GAME_ID"]].copy()
    deltas.rename(columns={c: c.replace("_new", "") for c in new_cols}, inplace=True)
    return deltas


def upsert_games(df):
    """Upsert games DataFrame to Supabase 'games' table."""
    if df.empty:
        logger.info("Upserting 0 games")
        return

    # Ensure correct types before serialization
    int_cols = ["GAME_ID", "SEASON_ID", "HOME_ID", "AWAY_ID", "GAME_STATUS", "GAME_OUTCOME", "HOME_PTS", "AWAY_PTS", "TOTAL_PTS"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    records = df.to_dict(orient="records")
    success = 0

    with tqdm(total=len(records), desc="Upserting games") as pbar:
>>>>>>> claude/festive-noyce
        for r in records:
            payload = {
                "GAME_ID": int(r["GAME_ID"]),
                "SEASON_ID": int(r["SEASON_ID"]) if pd.notna(r.get("SEASON_ID")) else None,
                "AWAY_NAME": r.get("AWAY_NAME"),
                "HOME_NAME": r.get("HOME_NAME"),
                "AWAY_ID": int(r["AWAY_ID"]) if pd.notna(r.get("AWAY_ID")) else None,
                "HOME_ID": int(r["HOME_ID"]) if pd.notna(r.get("HOME_ID")) else None,
<<<<<<< HEAD
                "GAME_STATUS": int(r.get("GAME_STATUS", GAME_STATUS_SCHEDULED)),
=======
                "GAME_STATUS": int(r["GAME_STATUS"]) if pd.notna(r.get("GAME_STATUS")) else GAME_STATUS_SCHEDULED,
>>>>>>> claude/festive-noyce
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
<<<<<<< HEAD
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
=======

    logger.info(f"Upserted {success}/{len(records)} games")


# ---------------------------------------------------------------------------
# 6. Merge helper — combine completed games with schedule (prefer completed data)
# ---------------------------------------------------------------------------
def merge_completed_and_schedule(completed_df, schedule_df):
    """
    Merge completed games (from LeagueGameFinder) with schedule (from ScheduleLeagueV2).
    Completed games take priority for scores/status. Schedule fills in future games.
    """
    if completed_df.empty:
        return schedule_df
    if schedule_df.empty:
        return completed_df

    # Start from completed games
    merged = completed_df.copy()

    # Find game IDs in schedule that aren't in completed
    existing_ids = set(completed_df["GAME_ID"].dropna().astype(int))
    schedule_new = schedule_df[~schedule_df["GAME_ID"].isin(existing_ids)].copy()

    if not schedule_new.empty:
        # Align columns before concat
        for col in merged.columns:
            if col not in schedule_new.columns:
                schedule_new[col] = pd.NA
        for col in schedule_new.columns:
            if col not in merged.columns:
                merged[col] = pd.NA

        merged = pd.concat([merged, schedule_new], ignore_index=True)

    # Also update any games that exist in both but schedule has newer status
    # (e.g., game went from scheduled to final between LGF and schedule fetch)
    overlap_sched = schedule_df[schedule_df["GAME_ID"].isin(existing_ids)].copy()
    if not overlap_sched.empty:
        for _, srow in overlap_sched.iterrows():
            gid = srow["GAME_ID"]
            mask = merged["GAME_ID"] == gid
            existing = merged.loc[mask]
            if not existing.empty:
                existing_status = existing.iloc[0].get("GAME_STATUS")
                sched_status = srow.get("GAME_STATUS")
                # If schedule says final but LGF didn't have it, update
                if pd.notna(sched_status) and sched_status == GAME_STATUS_FINAL and existing_status != GAME_STATUS_FINAL:
                    merged.loc[mask, "GAME_STATUS"] = srow["GAME_STATUS"]
                    if pd.notna(srow.get("HOME_PTS")):
                        merged.loc[mask, "HOME_PTS"] = srow["HOME_PTS"]
                        merged.loc[mask, "AWAY_PTS"] = srow["AWAY_PTS"]
                        merged.loc[mask, "TOTAL_PTS"] = srow["TOTAL_PTS"]
                        merged.loc[mask, "GAME_OUTCOME"] = srow["GAME_OUTCOME"]

    return merged.drop_duplicates("GAME_ID")


# ---------------------------------------------------------------------------
# 7. Mode implementations
# ---------------------------------------------------------------------------
def run_full_mode():
    """Full backfill: all seasons 2020-21 to present. Full upsert overwrite."""
    now = datetime.now()
    current_year = get_current_season_year()

    # Build season list: 2020-21 through current
    seasons = [f"{y}-{str(y + 1)[-2:]}" for y in range(2020, current_year + 1)]
    logger.info(f"Full mode: {len(seasons)} seasons to process: {seasons}")

    # 1. Fetch completed games from LeagueGameFinder (1 API call per season)
    logger.info("Step 1: Fetching completed games from LeagueGameFinder...")
    completed_df = fetch_completed_games(seasons)
    logger.info(f"  Total completed games: {len(completed_df)}")

    # 2. Fetch schedule for current season (for upcoming games)
    current_season = get_current_season_str()
    logger.info(f"Step 2: Fetching schedule for current season {current_season}...")
    schedule_df = fetch_schedule_for_season(current_season)

    # 3. Merge: completed data takes priority, schedule fills in future games
    logger.info("Step 3: Merging completed + scheduled games...")
    all_games = merge_completed_and_schedule(completed_df, schedule_df)
    logger.info(f"  Total games after merge: {len(all_games)}")

    if all_games.empty:
        logger.info("No games to process")
        return

    # 4. Fetch rosters for each season (30 teams × N seasons, batched with multiprocessing)
    logger.info("Step 4: Fetching rosters...")
    season_rosters = {}
    for season_str in seasons:
        year = season_str_to_year(season_str)
        logger.info(f"  Fetching rosters for {season_str}...")
        season_rosters[year] = fetch_rosters_for_season(season_str)

    # 5. Attach roster player IDs to games
    logger.info("Step 5: Attaching rosters to games...")
    all_games = attach_rosters(all_games, season_rosters)

    # 6. Full upsert (overwrite)
    logger.info(f"Step 6: Upserting {len(all_games)} games (full overwrite)...")
    upsert_games(all_games)


def run_incremental_mode():
    """Refresh current season. Delta upsert only changed rows."""
    current_season = get_current_season_str()
    current_year = get_current_season_year()
    logger.info(f"Incremental mode: season {current_season}")

    # 1. Fetch completed games for current season
    logger.info("Step 1: Fetching completed games for current season...")
    completed_df = fetch_completed_games_for_season(current_season)

    # 2. Fetch full schedule for current season
    logger.info("Step 2: Fetching schedule for current season...")
    schedule_df = fetch_schedule_for_season(current_season)

    # 3. Merge
    logger.info("Step 3: Merging completed + scheduled games...")
    new_df = merge_completed_and_schedule(completed_df, schedule_df)
    logger.info(f"  Total games after merge: {len(new_df)}")

    if new_df.empty:
        logger.info("No games to process")
        return

    # 4. Fetch rosters for current season
    logger.info("Step 4: Fetching rosters for current season...")
    season_rosters = {current_year: fetch_rosters_for_season(current_season)}

    # 5. Attach rosters
    logger.info("Step 5: Attaching rosters to games...")
    new_df = attach_rosters(new_df, season_rosters)

    # 6. Delta check against DB
    logger.info("Step 6: Finding deltas against DB...")
    db_df = fetch_db_games(season_ids=[current_year])
    deltas = find_deltas(new_df, db_df)
    logger.info(f"  {len(deltas)} deltas found")

    # 7. Upsert deltas only
    if not deltas.empty:
        logger.info(f"Step 7: Upserting {len(deltas)} changed games...")
        upsert_games(deltas)
    else:
        logger.info("No changes to upsert")


def run_current_mode():
    """Games from 3 days ago to today. Delta upsert on diffs only."""
    now = datetime.now(timezone.utc)
    current_season = get_current_season_str()
    current_year = get_current_season_year()

    date_from = (now - timedelta(days=3)).date()
    date_to = (now + timedelta(hours=12)).date()
    logger.info(f"Current mode: {date_from} to {date_to}")

    # 1. Fetch recent completed games via LeagueGameFinder with date range
    logger.info("Step 1: Fetching recent completed games...")
    try:
        sleep(random.uniform(0.8, 1.5))
        finder = leaguegamefinder.LeagueGameFinder(
            date_from_nullable=date_from.strftime("%m/%d/%Y"),
            date_to_nullable=date_to.strftime("%m/%d/%Y"),
            league_id_nullable="00",
            season_type_nullable="Regular Season",
        )
        raw = finder.get_data_frames()[0]
    except Exception as e:
        logger.error(f"LeagueGameFinder date range failed: {e}")
        raw = pd.DataFrame()

    recent_completed = pd.DataFrame()
    if not raw.empty:
        raw["GAME_ID"] = pd.to_numeric(raw["GAME_ID"], errors="coerce").astype("Int64")

        home_rows = raw[raw["MATCHUP"].str.contains("vs.", na=False)].copy()
        away_rows = raw[raw["MATCHUP"].str.contains("@", na=False)].copy()

        if not home_rows.empty or not away_rows.empty:
            home = home_rows[["GAME_ID", "GAME_DATE", "SEASON_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS", "WL"]].copy()
            home.rename(columns={"TEAM_ID": "HOME_ID", "TEAM_ABBREVIATION": "HOME_ABBR", "PTS": "HOME_PTS", "WL": "HOME_WL"}, inplace=True)

            away = away_rows[["GAME_ID", "TEAM_ID", "TEAM_ABBREVIATION", "PTS", "WL"]].copy()
            away.rename(columns={"TEAM_ID": "AWAY_ID", "TEAM_ABBREVIATION": "AWAY_ABBR", "PTS": "AWAY_PTS", "WL": "AWAY_WL"}, inplace=True)

            recent_completed = home.merge(away, on="GAME_ID", how="outer")
            recent_completed["HOME_NAME"] = recent_completed["HOME_ABBR"].map(TEAM_ABBR_TO_FULL)
            recent_completed["AWAY_NAME"] = recent_completed["AWAY_ABBR"].map(TEAM_ABBR_TO_FULL)
            recent_completed["SEASON_ID"] = recent_completed["SEASON_ID"].astype(str).str[-4:].astype("Int64")
            recent_completed["GAME_DATE"] = pd.to_datetime(recent_completed["GAME_DATE"], errors="coerce")
            recent_completed["HOME_PTS"] = pd.to_numeric(recent_completed["HOME_PTS"], errors="coerce").astype("Int64")
            recent_completed["AWAY_PTS"] = pd.to_numeric(recent_completed["AWAY_PTS"], errors="coerce").astype("Int64")
            recent_completed["TOTAL_PTS"] = (recent_completed["HOME_PTS"] + recent_completed["AWAY_PTS"]).astype("Int64")
            recent_completed["HOME_ID"] = pd.to_numeric(recent_completed["HOME_ID"], errors="coerce").astype("Int64")
            recent_completed["AWAY_ID"] = pd.to_numeric(recent_completed["AWAY_ID"], errors="coerce").astype("Int64")

            def derive_outcome(row):
                if pd.notna(row.get("HOME_WL")):
                    if row["HOME_WL"] == "W":
                        return GAME_OUTCOME_HOME_WIN
                    elif row["HOME_WL"] == "L":
                        return GAME_OUTCOME_AWAY_WIN
                return None

            recent_completed["GAME_OUTCOME"] = recent_completed.apply(derive_outcome, axis=1)
            recent_completed["GAME_STATUS"] = np.where(
                recent_completed["GAME_OUTCOME"].notna(), GAME_STATUS_FINAL, GAME_STATUS_SCHEDULED
            )
            recent_completed["GAME_STATUS"] = recent_completed["GAME_STATUS"].astype("Int64")

            keep = [
                "GAME_ID", "SEASON_ID", "GAME_DATE",
                "AWAY_NAME", "HOME_NAME", "AWAY_ID", "HOME_ID",
                "GAME_STATUS", "GAME_OUTCOME",
                "HOME_PTS", "AWAY_PTS", "TOTAL_PTS",
            ]
            recent_completed = recent_completed[[c for c in keep if c in recent_completed.columns]]
            recent_completed = recent_completed.drop_duplicates("GAME_ID")

    logger.info(f"  {len(recent_completed)} recent completed games")

    # 2. Fetch today's scoreboard for scheduled/live/just-finished games
    logger.info("Step 2: Fetching today's scoreboard...")
    today_df = fetch_scoreboard_for_date(now)

    # 3. Merge: completed data takes priority
    logger.info("Step 3: Merging recent + today's games...")
    new_df = merge_completed_and_schedule(recent_completed, today_df)
    logger.info(f"  Total games: {len(new_df)}")

    if new_df.empty:
        logger.info("No games to process")
        return

    # 4. Fetch rosters for current season only
    logger.info("Step 4: Fetching rosters for current season...")
    season_rosters = {current_year: fetch_rosters_for_season(current_season)}

    # 5. Attach rosters
    logger.info("Step 5: Attaching rosters to games...")
    new_df = attach_rosters(new_df, season_rosters)

    # 6. Delta check
    logger.info("Step 6: Finding deltas against DB...")
    game_ids = new_df["GAME_ID"].dropna().astype(int).tolist()
    db_df = fetch_db_games(game_ids=game_ids)
    deltas = find_deltas(new_df, db_df)
    logger.info(f"  {len(deltas)} deltas found")

    # 7. Upsert deltas only
    if not deltas.empty:
        logger.info(f"Step 7: Upserting {len(deltas)} changed games...")
        upsert_games(deltas)
    else:
        logger.info("No changes to upsert")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "current"

    if mode == "full":
        print("\n=== FULL MODE: backfill 2020-21 to present ===")
        run_full_mode()

    elif mode == "incremental":
        print(f"\n=== INCREMENTAL MODE: current season {get_current_season_str()} (delta) ===")
        run_incremental_mode()

    elif mode == "current":
        print("\n=== CURRENT MODE: last 3 days (delta) ===")
        run_current_mode()

    else:
        print(f"Unknown mode: {mode}. Use 'full', 'incremental', or 'current'.")
        sys.exit(1)

>>>>>>> claude/festive-noyce

if __name__ == "__main__":
    main()
