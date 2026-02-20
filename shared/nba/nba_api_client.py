# shared/nba/nba_api_client.py
"""
Direct HTTP client for stats.nba.com — replaces nba_api library.

nba_api's internal session gets blocked by stats.nba.com. Direct requests
with the right headers (x-nba-stats-origin, x-nba-stats-token) work
instantly.

Each function returns a list of DataFrames matching the nba_api pattern
(get_data_frames()[0], etc.) so callers need minimal changes.
"""

import logging
import pandas as pd
import requests
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

logger = logging.getLogger(__name__)

BASE_URL = "https://stats.nba.com/stats"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
    "Referer": "https://www.nba.com/",
    "Accept": "application/json",
    "x-nba-stats-origin": "stats",
    "x-nba-stats-token": "true",
}

DEFAULT_TIMEOUT = 60


def _get_session():
    """Create a requests session with retry logic for 429/5xx."""
    session = requests.Session()
    session.headers.update(HEADERS)
    retry = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


_session = None


def _get_shared_session():
    global _session
    if _session is None:
        _session = _get_session()
    return _session


def _api_get(endpoint, params, timeout=DEFAULT_TIMEOUT):
    """Make a GET request to stats.nba.com and return the JSON response."""
    url = f"{BASE_URL}/{endpoint}"
    resp = _get_shared_session().get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _result_sets_to_dataframes(data):
    """Convert resultSets JSON to a list of DataFrames."""
    frames = []
    for rs in data.get("resultSets", []):
        headers = rs.get("headers", [])
        rows = rs.get("rowSet", [])
        frames.append(pd.DataFrame(rows, columns=headers))
    return frames


# ---------------------------------------------------------------------------
# Endpoint wrappers
# ---------------------------------------------------------------------------

def fetch_scoreboard(game_date, timeout=DEFAULT_TIMEOUT):
    """ScoreboardV2 — returns [GameHeader, LineScore, ...]."""
    params = {
        "GameDate": game_date,
        "LeagueID": "00",
        "DayOffset": "0",
    }
    data = _api_get("scoreboardv2", params, timeout=timeout)
    return _result_sets_to_dataframes(data)


def fetch_schedule(season, timeout=DEFAULT_TIMEOUT):
    """
    ScheduleLeagueV2 — returns the full season schedule.

    The response format differs from other endpoints (nested JSON, not
    resultSets), so we flatten it into a single DataFrame matching the
    column names that nba_api would produce.
    """
    params = {"Season": season, "LeagueID": "00"}
    url = f"{BASE_URL}/scheduleleaguev2"
    resp = _get_shared_session().get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()

    game_dates = data.get("leagueSchedule", {}).get("gameDates", [])
    rows = []
    for gd in game_dates:
        for g in gd.get("games", []):
            home = g.get("homeTeam", {})
            away = g.get("awayTeam", {})
            rows.append({
                "gameId": g.get("gameId"),
                "gameDateTimeUTC": g.get("gameDateTimeUTC"),
                "gameStatus": g.get("gameStatus"),
                "postponedStatus": g.get("postponedStatus"),
                "homeTeam_teamId": home.get("teamId"),
                "homeTeam_score": home.get("score"),
                "awayTeam_teamId": away.get("teamId"),
                "awayTeam_score": away.get("score"),
            })

    return [pd.DataFrame(rows)] if rows else [pd.DataFrame()]


def fetch_game_finder(
    season=None,
    date_from=None,
    date_to=None,
    timeout=DEFAULT_TIMEOUT,
):
    """LeagueGameFinder — bulk completed games with scores."""
    params = {
        "LeagueID": "00",
        "SeasonType": "Regular Season",
    }
    if season:
        params["Season"] = season
    if date_from:
        params["DateFrom"] = date_from
    if date_to:
        params["DateTo"] = date_to

    data = _api_get("leaguegamefinder", params, timeout=timeout)
    return _result_sets_to_dataframes(data)


def fetch_team_roster(team_id, season, timeout=DEFAULT_TIMEOUT):
    """CommonTeamRoster — roster for a single team/season."""
    params = {
        "TeamID": str(team_id),
        "Season": season,
        "LeagueID": "00",
    }
    data = _api_get("commonteamroster", params, timeout=timeout)
    return _result_sets_to_dataframes(data)


def fetch_all_players(season, is_only_current_season=0, timeout=DEFAULT_TIMEOUT):
    """CommonAllPlayers — all players for a season."""
    params = {
        "IsOnlyCurrentSeason": str(is_only_current_season),
        "Season": season,
        "LeagueID": "00",
    }
    data = _api_get("commonallplayers", params, timeout=timeout)
    return _result_sets_to_dataframes(data)


def fetch_league_game_log(
    season,
    player_or_team="P",
    season_type="Regular Season",
    date_from=None,
    date_to=None,
    timeout=DEFAULT_TIMEOUT,
):
    """LeagueGameLog — all player/team game logs for a season."""
    params = {
        "Season": season,
        "PlayerOrTeam": player_or_team,
        "SeasonType": season_type,
        "LeagueID": "00",
    }
    if date_from:
        params["DateFrom"] = date_from
    if date_to:
        params["DateTo"] = date_to

    data = _api_get("leaguegamelog", params, timeout=timeout)
    return _result_sets_to_dataframes(data)
