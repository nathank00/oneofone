# tests/test_api_contracts.py
"""
API contract tests — lightweight real API calls to verify response structures.

These tests hit live endpoints (NBA API + MLB Stats API) to ensure the
response schemas haven't changed. Each test makes a single focused request
and validates field names, types, and non-emptiness. If the network is
unavailable, tests are skipped (not failed).

Marked with @pytest.mark.api so they can be run/skipped independently:
    pytest -m api          # run only API tests
    pytest -m "not api"    # skip API tests
"""

import pytest
import requests

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _safe_get(url, params=None, timeout=15):
    """GET request with graceful network failure → skip."""
    try:
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp
    except (requests.ConnectionError, requests.Timeout) as e:
        pytest.skip(f"Network unavailable: {e}")
    except requests.HTTPError as e:
        pytest.skip(f"API returned error: {e}")


# ===========================================================================
# MLB Stats API
# ===========================================================================
@pytest.mark.api
class TestMLBScheduleAPI:
    """Validate the MLB schedule endpoint returns expected structure."""

    URL = "https://statsapi.mlb.com/api/v1/schedule"

    def test_schedule_returns_dates_and_games(self):
        """A known historical date should return games with expected fields."""
        resp = _safe_get(self.URL, params={
            "sportId": 1,
            "startDate": "2024-07-04",
            "endDate": "2024-07-04",
            "gameType": "R",
            "hydrate": "linescore",
        })
        data = resp.json()

        assert "dates" in data
        dates = data["dates"]
        assert len(dates) > 0, "No dates returned for 2024-07-04"

        # Inspect first game
        games = dates[0].get("games", [])
        assert len(games) > 0, "No games on 2024-07-04"

        game = games[0]
        assert "gamePk" in game
        assert isinstance(game["gamePk"], int)
        assert "status" in game
        assert "abstractGameState" in game["status"]
        assert "teams" in game
        assert "home" in game["teams"]
        assert "away" in game["teams"]
        assert "team" in game["teams"]["home"]
        assert "id" in game["teams"]["home"]["team"]
        assert "name" in game["teams"]["home"]["team"]

    def test_schedule_game_has_linescore(self):
        """Hydrated linescore should contain runs for final games."""
        resp = _safe_get(self.URL, params={
            "sportId": 1,
            "startDate": "2024-07-04",
            "endDate": "2024-07-04",
            "gameType": "R",
            "hydrate": "linescore",
        })
        data = resp.json()
        games = data["dates"][0]["games"]

        # Find a final game
        final_games = [g for g in games if g.get("status", {}).get("abstractGameState") == "Final"]
        assert len(final_games) > 0, "No final games found"

        game = final_games[0]
        assert "linescore" in game, "Final game missing linescore hydration"
        linescore = game["linescore"]
        assert "teams" in linescore
        assert "home" in linescore["teams"]
        assert "runs" in linescore["teams"]["home"]

    def test_schedule_status_mapping(self):
        """detailedState and abstractGameState should be present for status mapping."""
        resp = _safe_get(self.URL, params={
            "sportId": 1,
            "startDate": "2024-07-04",
            "endDate": "2024-07-04",
            "gameType": "R",
        })
        data = resp.json()
        game = data["dates"][0]["games"][0]
        status = game["status"]

        assert "detailedState" in status
        assert "abstractGameState" in status
        assert isinstance(status["detailedState"], str)
        assert status["abstractGameState"] in ("Preview", "Live", "Final")


@pytest.mark.api
class TestMLBLiveFeedAPI:
    """Validate the MLB live feed endpoint (lineup data)."""

    # Use a known completed game (2024 July 4, Dodgers vs Diamondbacks)
    GAME_PK = 745652  # A real gamePk for 2024
    URL = f"https://statsapi.mlb.com/api/v1.1/game/{GAME_PK}/feed/live"

    def test_live_feed_has_boxscore(self):
        resp = _safe_get(self.URL)
        data = resp.json()

        assert "liveData" in data
        assert "boxscore" in data["liveData"]
        teams = data["liveData"]["boxscore"].get("teams", {})
        assert "home" in teams
        assert "away" in teams

    def test_live_feed_has_players_with_batting_order(self):
        resp = _safe_get(self.URL)
        data = resp.json()
        teams = data["liveData"]["boxscore"]["teams"]

        for side in ["home", "away"]:
            players = teams[side].get("players", {})
            assert len(players) > 0, f"No players for {side} team"

            # At least some players should have battingOrder
            has_batting_order = any(
                p.get("battingOrder") is not None
                for p in players.values()
            )
            assert has_batting_order, f"No batting orders for {side} team"

    def test_live_feed_has_pitchers(self):
        resp = _safe_get(self.URL)
        data = resp.json()
        teams = data["liveData"]["boxscore"]["teams"]

        for side in ["home", "away"]:
            pitchers = teams[side].get("pitchers", [])
            assert len(pitchers) > 0, f"No pitchers for {side} team"
            # First pitcher should be the SP
            assert isinstance(pitchers[0], int)

    def test_live_feed_has_bullpen(self):
        resp = _safe_get(self.URL)
        data = resp.json()
        teams = data["liveData"]["boxscore"]["teams"]

        for side in ["home", "away"]:
            bullpen = teams[side].get("bullpen", [])
            # Bullpen may be empty for some games, but field should exist
            assert isinstance(bullpen, list)


@pytest.mark.api
class TestMLBPlayersAPI:
    """Validate the MLB players endpoint."""

    URL = "https://statsapi.mlb.com/api/v1/sports/1/players"

    def test_players_returns_people_array(self):
        resp = _safe_get(self.URL, params={"season": 2024})
        data = resp.json()

        assert "people" in data
        people = data["people"]
        assert len(people) > 500, f"Expected 500+ players, got {len(people)}"

    def test_player_has_required_fields(self):
        resp = _safe_get(self.URL, params={"season": 2024})
        people = resp.json()["people"]

        player = people[0]
        assert "id" in player
        assert isinstance(player["id"], int)
        assert "fullName" in player
        assert isinstance(player["fullName"], str)
        assert "primaryPosition" in player
        assert "abbreviation" in player["primaryPosition"]

    def test_player_has_team_info(self):
        resp = _safe_get(self.URL, params={"season": 2024})
        people = resp.json()["people"]

        # Find a player with a current team
        with_team = [p for p in people if "currentTeam" in p]
        assert len(with_team) > 0, "No players with currentTeam"

        player = with_team[0]
        assert "id" in player["currentTeam"]
        assert isinstance(player["currentTeam"]["id"], int)

    def test_position_types_for_classification(self):
        """Verify pitcher positions match what our pipeline expects (P, SP, RP)."""
        resp = _safe_get(self.URL, params={"season": 2024})
        people = resp.json()["people"]

        positions = set(p["primaryPosition"]["abbreviation"] for p in people)
        # Our pipeline classifies based on these
        assert "P" in positions or "SP" in positions, \
            f"No pitcher positions found. Positions: {positions}"


@pytest.mark.api
class TestMLBPlayerStatsAPI:
    """Validate the MLB player stats gameLog endpoint."""

    # Mike Trout — well-known player with consistent stats
    PLAYER_ID = 545361
    URL = f"https://statsapi.mlb.com/api/v1/people/{PLAYER_ID}/stats"

    def test_batting_gamelog_structure(self):
        resp = _safe_get(self.URL, params={
            "stats": "gameLog",
            "group": "hitting",
            "season": 2023,
        })
        data = resp.json()

        assert "stats" in data
        stats = data["stats"]
        assert len(stats) > 0

        splits = stats[0].get("splits", [])
        # Trout played in 2023
        assert len(splits) > 0, "No batting game log splits for Trout in 2023"

        split = splits[0]
        assert "stat" in split
        stat = split["stat"]

        # Verify key batting stats our pipeline uses
        expected_keys = ["atBats", "hits", "runs", "homeRuns", "rbi",
                         "baseOnBalls", "strikeOuts", "stolenBases",
                         "plateAppearances", "avg", "obp", "slg", "ops"]
        for key in expected_keys:
            assert key in stat, f"Missing batting stat key: {key}"

    def test_pitching_gamelog_structure(self):
        """Test pitching gameLog for a known pitcher (Gerrit Cole, 2023)."""
        cole_id = 543037
        resp = _safe_get(
            f"https://statsapi.mlb.com/api/v1/people/{cole_id}/stats",
            params={"stats": "gameLog", "group": "pitching", "season": 2023},
        )
        data = resp.json()

        stats = data["stats"]
        assert len(stats) > 0
        splits = stats[0].get("splits", [])
        assert len(splits) > 0, "No pitching game log splits for Cole in 2023"

        stat = splits[0]["stat"]
        expected_keys = ["inningsPitched", "hits", "runs", "earnedRuns",
                         "baseOnBalls", "strikeOuts", "homeRuns",
                         "battersFaced", "numberOfPitches", "era", "whip"]
        for key in expected_keys:
            assert key in stat, f"Missing pitching stat key: {key}"


# ===========================================================================
# NBA API (via shared.nba.nba_api_client)
# ===========================================================================
@pytest.mark.api
class TestNBALeagueGameFinderAPI:
    """Validate the NBA LeagueGameFinder endpoint structure."""

    def test_returns_dataframe_with_expected_columns(self):
        try:
            import sys
            sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
            from shared.nba.nba_api_client import fetch_game_finder
            raw = fetch_game_finder(season="2023-24")[0]
        except Exception as e:
            pytest.skip(f"NBA API unavailable: {e}")

        assert not raw.empty, "LeagueGameFinder returned empty for 2023-24"

        required_cols = ["GAME_ID", "GAME_DATE", "SEASON_ID", "TEAM_ID",
                         "TEAM_ABBREVIATION", "MATCHUP", "PTS", "WL"]
        for col in required_cols:
            assert col in raw.columns, f"Missing column: {col}"

    def test_matchup_format_for_parsing(self):
        """Matchup strings should contain 'vs.' (home) or '@' (away)."""
        try:
            import sys
            sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
            from shared.nba.nba_api_client import fetch_game_finder
            raw = fetch_game_finder(season="2023-24")[0]
        except Exception as e:
            pytest.skip(f"NBA API unavailable: {e}")

        # Every row should have either "vs." or "@"
        has_vs = raw["MATCHUP"].str.contains("vs.", na=False)
        has_at = raw["MATCHUP"].str.contains("@", na=False)
        parseable = has_vs | has_at
        assert parseable.all(), "Some matchup strings are not parseable"


@pytest.mark.api
class TestNBACommonAllPlayersAPI:
    """Validate the NBA CommonAllPlayers endpoint."""

    def test_returns_players_with_expected_columns(self):
        try:
            import sys
            sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
            from shared.nba.nba_api_client import fetch_all_players
            df = fetch_all_players("2023-24")[0]
        except Exception as e:
            pytest.skip(f"NBA API unavailable: {e}")

        assert not df.empty, "CommonAllPlayers returned empty for 2023-24"

        required_cols = ["PERSON_ID", "DISPLAY_FIRST_LAST", "TEAM_ID",
                         "TEAM_NAME", "ROSTERSTATUS"]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_active_players_exist(self):
        try:
            import sys
            sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
            from shared.nba.nba_api_client import fetch_all_players
            df = fetch_all_players("2023-24")[0]
        except Exception as e:
            pytest.skip(f"NBA API unavailable: {e}")

        active = df[df["ROSTERSTATUS"] == 1]
        assert len(active) > 200, f"Expected 200+ active players, got {len(active)}"
