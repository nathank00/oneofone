#!/usr/bin/env python3
"""
Debug: fetch today's NBA games using shared client and print them.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shared.nba.nba_api_client import fetch_scoreboard, fetch_schedule, fetch_game_finder
from shared.nba.nba_constants import TEAM_NAME_TO_ID

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
now_et = datetime.now(ET)
today = now_et.strftime("%Y-%m-%d")

id_to_name = {int(v): k for k, v in TEAM_NAME_TO_ID.items()}

print(f"Today (ET): {today}")
print(f"Now (ET):   {now_et.strftime('%H:%M:%S')}")
print()

# ScoreboardV2
print("--- ScoreboardV2 ---")
try:
    frames = fetch_scoreboard(today)
    gh = frames[0]
    print(f"Games: {len(gh)}")
    for _, row in gh.iterrows():
        home = id_to_name.get(int(row["HOME_TEAM_ID"]), f"ID:{row['HOME_TEAM_ID']}")
        away = id_to_name.get(int(row["VISITOR_TEAM_ID"]), f"ID:{row['VISITOR_TEAM_ID']}")
        print(f"  {row['GAME_ID']}  {away} @ {home}  status={row['GAME_STATUS_ID']} ({str(row.get('GAME_STATUS_TEXT', '')).strip()})")
except Exception as e:
    print(f"FAILED: {e}")

print()

# ScheduleLeagueV2
print("--- ScheduleLeagueV2 ---")
try:
    year = now_et.year if now_et.month >= 7 else now_et.year - 1
    season = f"{year}-{str(year + 1)[-2:]}"
    frames = fetch_schedule(season)
    sched = frames[0]
    import pandas as pd
    utc_dt = pd.to_datetime(sched["gameDateTimeUTC"], errors="coerce", utc=True)
    sched["game_date_et"] = utc_dt.dt.tz_convert("America/New_York").dt.date
    today_games = sched[sched["game_date_et"] == now_et.date()]
    print(f"Games on {today}: {len(today_games)}")
    for _, row in today_games.iterrows():
        home = id_to_name.get(int(row["homeTeam_teamId"]), "?")
        away = id_to_name.get(int(row["awayTeam_teamId"]), "?")
        print(f"  {row['gameId']}  {away} @ {home}  status={row.get('gameStatus')}")
except Exception as e:
    print(f"FAILED: {e}")

print()

# LeagueGameFinder
print("--- LeagueGameFinder ---")
try:
    frames = fetch_game_finder(
        date_from=now_et.strftime("%m/%d/%Y"),
        date_to=now_et.strftime("%m/%d/%Y"),
    )
    raw = frames[0]
    unique = raw["GAME_ID"].nunique() if not raw.empty else 0
    print(f"Rows: {len(raw)} ({unique} games)")
except Exception as e:
    print(f"FAILED: {e}")
