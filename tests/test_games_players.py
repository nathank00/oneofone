# tests/test_games_players.py
"""
Tests for games.py and players.py parsing and transformation logic
for both NBA and MLB pipelines.

Tests the pure functions (season helpers, status mapping, find_deltas,
merge logic, player classification) using synthetic data.
No network calls, no Supabase.
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

REPO_ROOT = Path(__file__).resolve().parents[1]

# We import the pipeline modules using importlib to avoid the
# Supabase init-at-import issue (same pattern as conftest.py).
import importlib.util
import os

os.environ.setdefault("SUPABASE_URL", "https://fake.supabase.co")
os.environ.setdefault("SUPABASE_KEY", "fake-key-for-testing")


def _load(name, path):
    """Load a module from path with mocked Supabase."""
    with patch("supabase.create_client", return_value=MagicMock()):
        spec = importlib.util.spec_from_file_location(name, str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[name] = mod
        spec.loader.exec_module(mod)
    return mod


nba_games = _load("nba_games", REPO_ROOT / "nba-pipeline" / "src" / "games.py")
nba_players = _load("nba_players", REPO_ROOT / "nba-pipeline" / "src" / "players.py")
mlb_games = _load("mlb_games", REPO_ROOT / "mlb-pipeline" / "src" / "games.py")
mlb_players = _load("mlb_players", REPO_ROOT / "mlb-pipeline" / "src" / "players.py")


# ===========================================================================
# NBA Games — Season Helpers
# ===========================================================================
class TestNBASeasonHelpers:

    def test_season_str_to_year(self):
        assert nba_games.season_str_to_year("2024-25") == 2024
        assert nba_games.season_str_to_year("2020-21") == 2020

    def test_get_current_season_str_format(self):
        """Should return a string like '2024-25'."""
        result = nba_games.get_current_season_str()
        assert isinstance(result, str)
        parts = result.split("-")
        assert len(parts) == 2
        assert len(parts[0]) == 4
        assert len(parts[1]) == 2
        # The second part should be the last 2 digits of (first + 1)
        year = int(parts[0])
        assert parts[1] == str(year + 1)[-2:]

    def test_get_current_season_year_type(self):
        result = nba_games.get_current_season_year()
        assert isinstance(result, int)
        assert 2020 <= result <= 2030


# ===========================================================================
# NBA Games — Status Constants
# ===========================================================================
class TestNBAGameStatusConstants:

    def test_status_values(self):
        assert nba_games.GAME_STATUS_SCHEDULED == 1
        assert nba_games.GAME_STATUS_LIVE == 2
        assert nba_games.GAME_STATUS_FINAL == 3
        assert nba_games.GAME_STATUS_POSTPONED == 4

    def test_outcome_values(self):
        assert nba_games.GAME_OUTCOME_HOME_WIN == 1
        assert nba_games.GAME_OUTCOME_AWAY_WIN == 0


# ===========================================================================
# NBA Games — Merge Logic
# ===========================================================================
class TestNBAMergeCompletedAndSchedule:

    def test_schedule_fills_missing_games(self):
        completed = pd.DataFrame({
            "GAME_ID": pd.array([1, 2], dtype="Int64"),
            "GAME_STATUS": pd.array([3, 3], dtype="Int64"),
            "HOME_PTS": pd.array([100, 105], dtype="Int64"),
        })
        schedule = pd.DataFrame({
            "GAME_ID": pd.array([2, 3], dtype="Int64"),
            "GAME_STATUS": pd.array([3, 1], dtype="Int64"),
            "HOME_PTS": pd.array([105, pd.NA], dtype="Int64"),
        })
        result = nba_games.merge_completed_and_schedule(completed, schedule)
        assert set(result["GAME_ID"].tolist()) == {1, 2, 3}

    def test_completed_takes_priority(self):
        completed = pd.DataFrame({
            "GAME_ID": pd.array([1], dtype="Int64"),
            "GAME_STATUS": pd.array([3], dtype="Int64"),
            "HOME_PTS": pd.array([110], dtype="Int64"),
            "AWAY_PTS": pd.array([100], dtype="Int64"),
            "TOTAL_PTS": pd.array([210], dtype="Int64"),
            "GAME_OUTCOME": pd.array([1], dtype="Int64"),
            "SEASON_ID": pd.array([2024], dtype="Int64"),
            "GAME_DATE": pd.to_datetime(["2024-01-01"]),
            "HOME_NAME": ["Lakers"],
            "AWAY_NAME": ["Warriors"],
            "HOME_ID": pd.array([1610612747], dtype="Int64"),
            "AWAY_ID": pd.array([1610612744], dtype="Int64"),
        })
        schedule = pd.DataFrame({
            "GAME_ID": pd.array([1], dtype="Int64"),
            "GAME_STATUS": pd.array([1], dtype="Int64"),  # schedule says scheduled
            "HOME_PTS": pd.array([pd.NA], dtype="Int64"),
            "AWAY_PTS": pd.array([pd.NA], dtype="Int64"),
            "TOTAL_PTS": pd.array([pd.NA], dtype="Int64"),
            "GAME_OUTCOME": pd.array([pd.NA], dtype="Int64"),
            "SEASON_ID": pd.array([2024], dtype="Int64"),
            "GAME_DATE": pd.to_datetime(["2024-01-01"]),
            "HOME_NAME": ["Lakers"],
            "AWAY_NAME": ["Warriors"],
            "HOME_ID": pd.array([1610612747], dtype="Int64"),
            "AWAY_ID": pd.array([1610612744], dtype="Int64"),
        })
        result = nba_games.merge_completed_and_schedule(completed, schedule)
        row = result[result["GAME_ID"] == 1].iloc[0]
        # Completed game has final status and scores — should be preserved
        assert int(row["HOME_PTS"]) == 110

    def test_empty_completed(self):
        schedule = pd.DataFrame({
            "GAME_ID": pd.array([1], dtype="Int64"),
            "GAME_STATUS": pd.array([1], dtype="Int64"),
        })
        result = nba_games.merge_completed_and_schedule(pd.DataFrame(), schedule)
        assert len(result) == 1

    def test_empty_schedule(self):
        completed = pd.DataFrame({
            "GAME_ID": pd.array([1], dtype="Int64"),
            "GAME_STATUS": pd.array([3], dtype="Int64"),
        })
        result = nba_games.merge_completed_and_schedule(completed, pd.DataFrame())
        assert len(result) == 1


# ===========================================================================
# NBA Games — find_deltas
# ===========================================================================
class TestNBAFindDeltas:

    def test_all_new_when_db_empty(self):
        new_df = pd.DataFrame({
            "GAME_ID": pd.array([1, 2, 3], dtype="Int64"),
            "SEASON_ID": pd.array([2024, 2024, 2024], dtype="Int64"),
            "GAME_DATE": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"], utc=True),
            "GAME_STATUS": pd.array([3, 3, 1], dtype="Int64"),
        })
        result = nba_games.find_deltas(new_df, pd.DataFrame())
        assert len(result) == 3

    def test_detects_status_change(self):
        new_df = pd.DataFrame({
            "GAME_ID": pd.array([1], dtype="Int64"),
            "SEASON_ID": pd.array([2024], dtype="Int64"),
            "GAME_DATE": pd.to_datetime(["2024-01-01"], utc=True),
            "GAME_STATUS": pd.array([3], dtype="Int64"),
            "GAME_OUTCOME": pd.array([1], dtype="Int64"),
            "AWAY_NAME": ["Warriors"],
            "HOME_NAME": ["Lakers"],
            "AWAY_ID": pd.array([1610612744], dtype="Int64"),
            "HOME_ID": pd.array([1610612747], dtype="Int64"),
            "AWAY_PTS": pd.array([100], dtype="Int64"),
            "HOME_PTS": pd.array([110], dtype="Int64"),
            "TOTAL_PTS": pd.array([210], dtype="Int64"),
            "HOME_TEAM_PLAYERS": [[]],
            "AWAY_TEAM_PLAYERS": [[]],
        })
        db_df = pd.DataFrame({
            "GAME_ID": pd.array([1], dtype="Int64"),
            "SEASON_ID": pd.array([2024], dtype="Int64"),
            "GAME_DATE": pd.to_datetime(["2024-01-01"], utc=True),
            "GAME_STATUS": pd.array([1], dtype="Int64"),  # Was scheduled
            "GAME_OUTCOME": pd.array([pd.NA], dtype="Int64"),
            "AWAY_NAME": ["Warriors"],
            "HOME_NAME": ["Lakers"],
            "AWAY_ID": pd.array([1610612744], dtype="Int64"),
            "HOME_ID": pd.array([1610612747], dtype="Int64"),
            "AWAY_PTS": pd.array([pd.NA], dtype="Int64"),
            "HOME_PTS": pd.array([pd.NA], dtype="Int64"),
            "TOTAL_PTS": pd.array([pd.NA], dtype="Int64"),
            "HOME_TEAM_PLAYERS": [[]],
            "AWAY_TEAM_PLAYERS": [[]],
        })
        result = nba_games.find_deltas(new_df, db_df)
        assert len(result) == 1  # Status changed from 1 → 3


# ===========================================================================
# NBA Games — Roster Attachment
# ===========================================================================
class TestNBAAttachRosters:

    def test_attaches_player_lists(self):
        games = pd.DataFrame({
            "GAME_ID": pd.array([1], dtype="Int64"),
            "SEASON_ID": pd.array([2024], dtype="Int64"),
            "HOME_ID": pd.array([1610612747], dtype="Int64"),
            "AWAY_ID": pd.array([1610612744], dtype="Int64"),
        })
        season_rosters = {
            2024: {
                1610612747: [101, 102, 103],
                1610612744: [201, 202, 203],
            }
        }
        result = nba_games.attach_rosters(games, season_rosters)
        assert result.iloc[0]["HOME_TEAM_PLAYERS"] == [101, 102, 103]
        assert result.iloc[0]["AWAY_TEAM_PLAYERS"] == [201, 202, 203]

    def test_missing_roster_returns_empty_list(self):
        games = pd.DataFrame({
            "GAME_ID": pd.array([1], dtype="Int64"),
            "SEASON_ID": pd.array([2024], dtype="Int64"),
            "HOME_ID": pd.array([1610612747], dtype="Int64"),
            "AWAY_ID": pd.array([9999999], dtype="Int64"),  # Unknown team
        })
        season_rosters = {
            2024: {1610612747: [101, 102]},
        }
        result = nba_games.attach_rosters(games, season_rosters)
        assert result.iloc[0]["HOME_TEAM_PLAYERS"] == [101, 102]
        assert result.iloc[0]["AWAY_TEAM_PLAYERS"] == []


# ===========================================================================
# NBA Players — find_deltas
# ===========================================================================
class TestNBAPlayerFindDeltas:

    def test_all_new_when_db_empty(self):
        new_df = pd.DataFrame({
            "PERSON_ID": pd.array([101, 102], dtype="Int64"),
            "DISPLAY_FIRST_LAST": ["LeBron James", "Anthony Davis"],
            "TEAM_ID": pd.array([1610612747, 1610612747], dtype="Int64"),
            "TEAM_NAME": ["Los Angeles Lakers", "Los Angeles Lakers"],
            "FROM_YEAR": pd.array([2003, 2012], dtype="Int64"),
            "TO_YEAR": pd.array([2024, 2024], dtype="Int64"),
            "PLAYER_SLUG": ["lebron-james", "anthony-davis"],
        })
        result = nba_players.find_deltas(new_df, pd.DataFrame())
        assert len(result) == 2

    def test_detects_team_change(self):
        new_df = pd.DataFrame({
            "PERSON_ID": pd.array([101], dtype="Int64"),
            "DISPLAY_FIRST_LAST": ["Player X"],
            "TEAM_ID": pd.array([1610612747], dtype="Int64"),  # New team
            "TEAM_NAME": ["Los Angeles Lakers"],
            "FROM_YEAR": pd.array([2020], dtype="Int64"),
            "TO_YEAR": pd.array([2024], dtype="Int64"),
            "PLAYER_SLUG": ["player-x"],
        })
        db_df = pd.DataFrame({
            "PERSON_ID": pd.array([101], dtype="Int64"),
            "DISPLAY_FIRST_LAST": ["Player X"],
            "TEAM_ID": pd.array([1610612744], dtype="Int64"),  # Old team
            "TEAM_NAME": ["Golden State Warriors"],
            "FROM_YEAR": pd.array([2020], dtype="Int64"),
            "TO_YEAR": pd.array([2024], dtype="Int64"),
            "PLAYER_SLUG": ["player-x"],
        })
        result = nba_players.find_deltas(new_df, db_df)
        assert len(result) == 1  # Team changed

    def test_no_deltas_when_identical(self):
        data = {
            "PERSON_ID": pd.array([101], dtype="Int64"),
            "DISPLAY_FIRST_LAST": ["Player X"],
            "TEAM_ID": pd.array([1610612747], dtype="Int64"),
            "TEAM_NAME": ["Los Angeles Lakers"],
            "FROM_YEAR": pd.array([2020], dtype="Int64"),
            "TO_YEAR": pd.array([2024], dtype="Int64"),
            "PLAYER_SLUG": ["player-x"],
        }
        new_df = pd.DataFrame(data)
        db_df = pd.DataFrame(data)
        result = nba_players.find_deltas(new_df, db_df)
        assert len(result) == 0


# ===========================================================================
# MLB Games — Status Mapping
# ===========================================================================
class TestMLBGameStatusConstants:

    def test_status_values(self):
        assert mlb_games.GAME_STATUS_SCHEDULED == 1
        assert mlb_games.GAME_STATUS_LIVE == 2
        assert mlb_games.GAME_STATUS_FINAL == 3
        assert mlb_games.GAME_STATUS_POSTPONED == 4


# ===========================================================================
# MLB Games — merge_games_and_lineups
# ===========================================================================
class TestMLBMergeGamesAndLineups:

    def test_merges_lineup_data(self):
        games = [
            {"GAME_ID": 100, "HOME_NAME": "Dodgers"},
            {"GAME_ID": 200, "HOME_NAME": "Yankees"},
        ]
        lineups = {
            100: {
                "HOME_LINEUP": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                "AWAY_LINEUP": [11, 12, 13, 14, 15, 16, 17, 18, 19],
                "HOME_SP": 501,
                "AWAY_SP": 601,
                "HOME_BULLPEN": [502, 503],
                "AWAY_BULLPEN": [602, 603],
            },
            200: {
                "HOME_LINEUP": [],
                "AWAY_LINEUP": [],
                "HOME_SP": None,
                "AWAY_SP": None,
                "HOME_BULLPEN": [],
                "AWAY_BULLPEN": [],
            },
        }
        result = mlb_games.merge_games_and_lineups(games, lineups)

        assert result[0]["HOME_LINEUP"] == [1, 2, 3, 4, 5, 6, 7, 8, 9]
        assert result[0]["HOME_SP"] == 501
        assert result[1]["HOME_LINEUP"] == []
        assert result[1]["HOME_SP"] is None

    def test_missing_lineup_uses_defaults(self):
        games = [{"GAME_ID": 999, "HOME_NAME": "Test"}]
        lineups = {}  # No lineup data
        result = mlb_games.merge_games_and_lineups(games, lineups)

        assert result[0]["HOME_LINEUP"] == []
        assert result[0]["AWAY_LINEUP"] == []
        assert result[0]["HOME_SP"] is None
        assert result[0]["AWAY_SP"] is None


# ===========================================================================
# MLB Games — find_deltas
# ===========================================================================
class TestMLBFindDeltas:

    def test_all_new_when_db_empty(self):
        games = [
            {"GAME_ID": 100, "GAME_STATUS": 1, "HOME_RUNS": None},
            {"GAME_ID": 200, "GAME_STATUS": 3, "HOME_RUNS": 5},
        ]
        result = mlb_games.find_deltas(games, [])
        assert len(result) == 2

    def test_detects_status_change(self):
        new_games = [
            {
                "GAME_ID": 100,
                "GAME_STATUS": 3,
                "GAME_OUTCOME": 1,
                "HOME_RUNS": 5,
                "AWAY_RUNS": 3,
                "TOTAL_RUNS": 8,
                "HOME_SP": 501,
                "AWAY_SP": 601,
                "HOME_LINEUP": [1, 2, 3],
                "AWAY_LINEUP": [4, 5, 6],
                "HOME_BULLPEN": [],
                "AWAY_BULLPEN": [],
            }
        ]
        db_games = [
            {
                "GAME_ID": 100,
                "GAME_STATUS": 1,  # Was scheduled
                "GAME_OUTCOME": None,
                "HOME_RUNS": None,
                "AWAY_RUNS": None,
                "TOTAL_RUNS": None,
                "HOME_SP": None,
                "AWAY_SP": None,
                "HOME_LINEUP": [],
                "AWAY_LINEUP": [],
                "HOME_BULLPEN": [],
                "AWAY_BULLPEN": [],
            }
        ]
        result = mlb_games.find_deltas(new_games, db_games)
        assert len(result) == 1

    def test_no_deltas_when_identical(self):
        game_data = {
            "GAME_ID": 100,
            "GAME_STATUS": 3,
            "GAME_OUTCOME": 1,
            "HOME_RUNS": 5,
            "AWAY_RUNS": 3,
            "TOTAL_RUNS": 8,
            "HOME_SP": 501,
            "AWAY_SP": 601,
            "HOME_LINEUP": [1, 2, 3],
            "AWAY_LINEUP": [4, 5, 6],
            "HOME_BULLPEN": [502],
            "AWAY_BULLPEN": [602],
        }
        result = mlb_games.find_deltas([game_data], [game_data.copy()])
        assert len(result) == 0

    def test_detects_lineup_change(self):
        base = {
            "GAME_ID": 100,
            "GAME_STATUS": 1,
            "GAME_OUTCOME": None,
            "HOME_RUNS": None,
            "AWAY_RUNS": None,
            "TOTAL_RUNS": None,
            "HOME_SP": None,
            "AWAY_SP": None,
            "HOME_BULLPEN": [],
            "AWAY_BULLPEN": [],
        }
        new_game = {**base, "HOME_LINEUP": [1, 2, 3, 4, 5, 6, 7, 8, 9], "AWAY_LINEUP": []}
        db_game = {**base, "HOME_LINEUP": [], "AWAY_LINEUP": []}

        result = mlb_games.find_deltas([new_game], [db_game])
        assert len(result) == 1  # Lineup changed


# ===========================================================================
# MLB Games — _clean_payload
# ===========================================================================
class TestMLBCleanPayload:

    def test_basic_payload(self):
        game = {
            "GAME_ID": 745652,
            "SEASON_ID": 2024,
            "GAME_DATE": "2024-07-04",
            "HOME_NAME": "Dodgers",
            "AWAY_NAME": "Diamondbacks",
            "HOME_ID": 119,
            "AWAY_ID": 109,
            "GAME_STATUS": 3,
            "GAME_OUTCOME": 1,
            "HOME_RUNS": 5,
            "AWAY_RUNS": 3,
            "TOTAL_RUNS": 8,
            "HOME_LINEUP": [1, 2, 3, None, 5, 6, 7, 8, 9],
            "AWAY_LINEUP": [11, 12, 13, 14, 15, 16, 17, 18, 19],
            "HOME_SP": 501,
            "AWAY_SP": 601,
            "HOME_BULLPEN": [502, 503],
            "AWAY_BULLPEN": [602],
        }
        payload = mlb_games._clean_payload(game)

        assert payload["GAME_ID"] == 745652
        assert isinstance(payload["GAME_ID"], int)
        assert payload["GAME_STATUS"] == 3
        # None values in lineup should be filtered
        assert None not in payload["HOME_LINEUP"]
        assert len(payload["HOME_LINEUP"]) == 8  # 1 None filtered
        assert all(isinstance(x, int) for x in payload["HOME_LINEUP"])

    def test_null_fields(self):
        game = {
            "GAME_ID": 100,
            "SEASON_ID": None,
            "GAME_DATE": None,
            "HOME_NAME": None,
            "AWAY_NAME": None,
            "HOME_ID": None,
            "AWAY_ID": None,
            "GAME_STATUS": None,
            "GAME_OUTCOME": None,
            "HOME_RUNS": None,
            "AWAY_RUNS": None,
            "TOTAL_RUNS": None,
            "HOME_LINEUP": [],
            "AWAY_LINEUP": [],
            "HOME_SP": None,
            "AWAY_SP": None,
            "HOME_BULLPEN": [],
            "AWAY_BULLPEN": [],
        }
        payload = mlb_games._clean_payload(game)
        assert payload["GAME_ID"] == 100
        assert payload["SEASON_ID"] is None
        assert payload["GAME_OUTCOME"] is None
        assert payload["HOME_SP"] is None


# ===========================================================================
# MLB Players — Player Type Classification
# ===========================================================================
class TestMLBPlayerTypeClassification:

    def test_pitcher_positions(self):
        assert "P" in mlb_players.PITCHER_POSITIONS
        assert "SP" in mlb_players.PITCHER_POSITIONS
        assert "RP" in mlb_players.PITCHER_POSITIONS

    def test_two_way_position(self):
        assert mlb_players.TWP_POSITION == "TWP"

    def test_classification_logic(self):
        """Simulate the classification logic from fetch_players_for_season."""
        test_cases = [
            ("SS", "batter"),
            ("OF", "batter"),
            ("1B", "batter"),
            ("C", "batter"),
            ("P", "pitcher"),
            ("SP", "pitcher"),
            ("RP", "pitcher"),
            ("TWP", "two_way"),
            ("DH", "batter"),
        ]
        for position, expected_type in test_cases:
            if position in mlb_players.PITCHER_POSITIONS:
                player_type = "pitcher"
            elif position == mlb_players.TWP_POSITION:
                player_type = "two_way"
            else:
                player_type = "batter"
            assert player_type == expected_type, \
                f"Position {position}: expected {expected_type}, got {player_type}"


# ===========================================================================
# MLB Players — find_deltas
# ===========================================================================
class TestMLBPlayerFindDeltas:

    def test_all_new_when_db_empty(self):
        new_players = [
            {"PLAYER_ID": 100, "FULL_NAME": "Mike Trout", "TEAM_ID": 108,
             "TEAM_NAME": "Angels", "POSITION": "OF", "PLAYER_TYPE": "batter"},
        ]
        result = mlb_players.find_deltas(new_players, [])
        assert len(result) == 1

    def test_detects_team_change(self):
        new_players = [
            {"PLAYER_ID": 100, "FULL_NAME": "Player X", "TEAM_ID": 119,
             "TEAM_NAME": "Dodgers", "POSITION": "SS", "PLAYER_TYPE": "batter"},
        ]
        db_players = [
            {"PLAYER_ID": 100, "FULL_NAME": "Player X", "TEAM_ID": 137,
             "TEAM_NAME": "Giants", "POSITION": "SS", "PLAYER_TYPE": "batter"},
        ]
        result = mlb_players.find_deltas(new_players, db_players)
        assert len(result) == 1

    def test_no_deltas_when_identical(self):
        player = {"PLAYER_ID": 100, "FULL_NAME": "Player X", "TEAM_ID": 119,
                   "TEAM_NAME": "Dodgers", "POSITION": "SS", "PLAYER_TYPE": "batter"}
        result = mlb_players.find_deltas([player], [player.copy()])
        assert len(result) == 0

    def test_detects_position_change(self):
        new_players = [
            {"PLAYER_ID": 100, "FULL_NAME": "Player X", "TEAM_ID": 119,
             "TEAM_NAME": "Dodgers", "POSITION": "TWP", "PLAYER_TYPE": "two_way"},
        ]
        db_players = [
            {"PLAYER_ID": 100, "FULL_NAME": "Player X", "TEAM_ID": 119,
             "TEAM_NAME": "Dodgers", "POSITION": "OF", "PLAYER_TYPE": "batter"},
        ]
        result = mlb_players.find_deltas(new_players, db_players)
        assert len(result) == 1
