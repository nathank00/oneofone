# tests/test_nba_gamelogs.py
"""
Tests for NBA gamelogs feature engineering logic.

Tests the pure computational functions using synthetic data:
- _extract_team_abbr: matchup string parsing
- compute_team_games: player stat aggregation to team level
- add_rolling_features: shift(1) rolling window computation
- build_gamelogs: final gamelog assembly with HOME_/AWAY_ prefixes
- prepare_records: JSON-safe record cleaning
"""

import numpy as np
import pandas as pd

# Import the pre-loaded NBA modules from conftest (avoids name collisions)
from conftest import nba_gamelogs


# ---------------------------------------------------------------------------
# Helpers to build synthetic data
# ---------------------------------------------------------------------------
def make_games_df(n=5, home_id=1610612747, away_id=1610612744, start_game_id=100):
    """Create a synthetic games DataFrame."""
    dates = pd.date_range("2024-01-01", periods=n, freq="2D", tz="UTC")
    return pd.DataFrame({
        "GAME_ID": range(start_game_id, start_game_id + n),
        "SEASON_ID": [22024] * n,
        "GAME_DATE": dates,
        "AWAY_NAME": ["Golden State Warriors"] * n,
        "HOME_NAME": ["Los Angeles Lakers"] * n,
        "AWAY_ID": pd.array([away_id] * n, dtype="Int64"),
        "HOME_ID": pd.array([home_id] * n, dtype="Int64"),
        "GAME_STATUS": pd.array([3] * n, dtype="Int64"),
        "GAME_OUTCOME": pd.array([1, 0, 1, 0, 1][:n], dtype="Int64"),
        "AWAY_PTS": pd.array([100, 110, 105, 115, 102][:n], dtype="Int64"),
        "HOME_PTS": pd.array([110, 105, 108, 100, 107][:n], dtype="Int64"),
        "TOTAL_PTS": pd.array([210, 215, 213, 215, 209][:n], dtype="Int64"),
    })


def make_playerstats_df(games_df, players_per_team=3):
    """Create synthetic playerstats for home and away teams."""
    rng = np.random.RandomState(42)
    rows = []

    for _, game in games_df.iterrows():
        for side in ["HOME", "AWAY"]:
            team_id = int(game[f"{side}_ID"])
            base_pid = team_id * 100

            for p in range(players_per_team):
                pid = base_pid + p
                matchup = "LAL vs. GSW" if side == "HOME" else "GSW @ LAL"
                wl = ("W" if game["GAME_OUTCOME"] == 1 else "L") if side == "HOME" else \
                     ("L" if game["GAME_OUTCOME"] == 1 else "W")

                rows.append({
                    "GAME_ID": int(game["GAME_ID"]),
                    "GAME_DATE": game["GAME_DATE"],
                    "SEASON_ID": int(game["SEASON_ID"]),
                    "PLAYER_ID": pid,
                    "MATCHUP": matchup,
                    "WL": wl,
                    "FGM": rng.randint(3, 10),
                    "FGA": rng.randint(10, 20),
                    "FG3M": rng.randint(0, 5),
                    "FG3A": rng.randint(3, 10),
                    "FTM": rng.randint(1, 5),
                    "FTA": rng.randint(2, 7),
                    "OREB": rng.randint(0, 4),
                    "DREB": rng.randint(2, 8),
                    "REB": rng.randint(3, 12),
                    "AST": rng.randint(1, 8),
                    "STL": rng.randint(0, 3),
                    "BLK": rng.randint(0, 3),
                    "TOV": rng.randint(0, 4),
                    "PF": rng.randint(0, 5),
                    "PTS": rng.randint(8, 30),
                    "PLUS_MINUS": rng.randint(-15, 15),
                })

    df = pd.DataFrame(rows)
    int_cols = ["GAME_ID", "PLAYER_ID", "SEASON_ID"] + nba_gamelogs.COUNTING_STATS
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestExtractTeamAbbr:

    def test_away_format(self):
        assert nba_gamelogs._extract_team_abbr("GSW @ LAL") == "GSW"

    def test_home_format(self):
        assert nba_gamelogs._extract_team_abbr("LAL vs. GSW") == "LAL"

    def test_none_input(self):
        assert nba_gamelogs._extract_team_abbr(None) is None

    def test_numeric_input(self):
        assert nba_gamelogs._extract_team_abbr(12345) is None

    def test_empty_string(self):
        assert nba_gamelogs._extract_team_abbr("") is None

    def test_weird_format(self):
        """No recognized separator — returns None."""
        assert nba_gamelogs._extract_team_abbr("LAL-GSW") is None


class TestComputeTeamGames:

    def test_produces_two_rows_per_game(self):
        games = make_games_df(n=3)
        stats = make_playerstats_df(games)
        team_df = nba_gamelogs.compute_team_games(games, stats)

        # 3 games × 2 sides = 6 rows
        assert len(team_df) == 6

    def test_sides_are_home_and_away(self):
        games = make_games_df(n=2)
        stats = make_playerstats_df(games)
        team_df = nba_gamelogs.compute_team_games(games, stats)

        sides = set(team_df["SIDE"].tolist())
        assert sides == {"HOME", "AWAY"}

    def test_aggregates_player_stats(self):
        """Team-level PTS should be the sum of all players on that side."""
        games = make_games_df(n=1)
        stats = make_playerstats_df(games, players_per_team=3)
        team_df = nba_gamelogs.compute_team_games(games, stats)

        for _, row in team_df.iterrows():
            side = row["SIDE"]
            game_id = row["GAME_ID"]
            team_id = row["TEAM_ID"]

            side_stats = stats[
                (stats["GAME_ID"] == game_id) &
                (stats["PLAYER_ID"].isin(range(int(team_id) * 100, int(team_id) * 100 + 3)))
            ]
            assert int(row["PTS"]) == int(side_stats["PTS"].sum())

    def test_shooting_percentages_from_totals(self):
        """FG_PCT should be FGM/FGA, not averaged per-player."""
        games = make_games_df(n=1)
        stats = make_playerstats_df(games, players_per_team=3)
        team_df = nba_gamelogs.compute_team_games(games, stats)

        for _, row in team_df.iterrows():
            if row["FGA"] > 0:
                expected_fg = row["FGM"] / row["FGA"]
                assert abs(row["FG_PCT"] - expected_fg) < 1e-6

    def test_win_indicator(self):
        games = make_games_df(n=1)
        games["GAME_OUTCOME"] = pd.array([1], dtype="Int64")  # home win
        stats = make_playerstats_df(games)
        team_df = nba_gamelogs.compute_team_games(games, stats)

        home = team_df[team_df["SIDE"] == "HOME"]
        away = team_df[team_df["SIDE"] == "AWAY"]
        assert int(home.iloc[0]["WIN"]) == 1
        assert int(away.iloc[0]["WIN"]) == 0

    def test_empty_inputs(self):
        empty = pd.DataFrame()
        result = nba_gamelogs.compute_team_games(empty, empty)
        assert result.empty


class TestAddRollingFeatures:

    def test_rolling_columns_created(self):
        games = make_games_df(n=5)
        stats = make_playerstats_df(games)
        team_df = nba_gamelogs.compute_team_games(games, stats)
        rolled = nba_gamelogs.add_rolling_features(team_df)

        for stat in nba_gamelogs.ROLLING_STATS:
            for window in [10, 30]:
                col = f"{stat}_{window}"
                assert col in rolled.columns, f"Missing rolling column: {col}"

    def test_first_game_is_nan(self):
        """shift(1) means the first game for each team should have NaN rolling."""
        games = make_games_df(n=3)
        stats = make_playerstats_df(games)
        team_df = nba_gamelogs.compute_team_games(games, stats)
        rolled = nba_gamelogs.add_rolling_features(team_df)

        for tid, group in rolled.groupby("TEAM_ID"):
            first = group.sort_values("GAME_DATE").iloc[0]
            assert pd.isna(first["PTS_10"]), f"Team {tid} first game PTS_10 should be NaN"

    def test_no_data_leakage(self):
        """The rolling average for game N should NOT include game N's stats."""
        games = make_games_df(n=5)
        stats = make_playerstats_df(games, players_per_team=2)
        team_df = nba_gamelogs.compute_team_games(games, stats)
        rolled = nba_gamelogs.add_rolling_features(team_df)

        for tid, group in rolled.groupby("TEAM_ID"):
            sorted_group = group.sort_values("GAME_DATE").reset_index(drop=True)
            if len(sorted_group) >= 3:
                expected = (sorted_group.loc[0, "PTS"] + sorted_group.loc[1, "PTS"]) / 2
                actual = sorted_group.loc[2, "PTS_10"]
                assert abs(actual - expected) < 1e-6, \
                    f"Data leakage detected: expected {expected}, got {actual}"

    def test_win_rate_rolling(self):
        games = make_games_df(n=5)
        stats = make_playerstats_df(games)
        team_df = nba_gamelogs.compute_team_games(games, stats)
        rolled = nba_gamelogs.add_rolling_features(team_df)

        assert "WIN_RATE_10" in rolled.columns
        assert "WIN_RATE_30" in rolled.columns
        assert "GAMES_10" in rolled.columns
        assert "GAMES_30" in rolled.columns

    def test_empty_input(self):
        result = nba_gamelogs.add_rolling_features(pd.DataFrame())
        assert result.empty


class TestBuildGamelogs:

    def test_output_has_home_away_columns(self):
        games = make_games_df(n=5)
        stats = make_playerstats_df(games)
        team_df = nba_gamelogs.compute_team_games(games, stats)
        team_df = nba_gamelogs.add_rolling_features(team_df)
        result = nba_gamelogs.build_gamelogs(games, team_df)

        home_cols = [c for c in result.columns if c.startswith("HOME_")]
        away_cols = [c for c in result.columns if c.startswith("AWAY_")]
        assert len(home_cols) > 0, "Missing HOME_ columns"
        assert len(away_cols) > 0, "Missing AWAY_ columns"

    def test_one_row_per_game(self):
        games = make_games_df(n=5)
        stats = make_playerstats_df(games)
        team_df = nba_gamelogs.compute_team_games(games, stats)
        team_df = nba_gamelogs.add_rolling_features(team_df)
        result = nba_gamelogs.build_gamelogs(games, team_df)

        assert len(result) == len(games)

    def test_metadata_preserved(self):
        games = make_games_df(n=3)
        stats = make_playerstats_df(games)
        team_df = nba_gamelogs.compute_team_games(games, stats)
        team_df = nba_gamelogs.add_rolling_features(team_df)
        result = nba_gamelogs.build_gamelogs(games, team_df)

        assert set(result["GAME_ID"].tolist()) == set(games["GAME_ID"].tolist())
        assert "GAME_STATUS" in result.columns
        assert "GAME_OUTCOME" in result.columns

    def test_empty_games(self):
        result = nba_gamelogs.build_gamelogs(pd.DataFrame(), pd.DataFrame())
        assert result.empty


class TestPrepareRecords:

    def test_nan_converted_to_none(self):
        df = pd.DataFrame({"GAME_ID": [1], "VALUE": [float("nan")]})
        records = nba_gamelogs.prepare_records(df)
        assert records[0]["VALUE"] is None

    def test_numpy_types_converted(self):
        df = pd.DataFrame({
            "GAME_ID": pd.array([1], dtype="Int64"),
            "VALUE": np.array([1.23456], dtype=np.float64),
        })
        records = nba_gamelogs.prepare_records(df)
        assert isinstance(records[0]["GAME_ID"], int)
        assert isinstance(records[0]["VALUE"], float)

    def test_output_is_json_safe(self):
        """All values in prepared records should be JSON-serializable types."""
        import json
        df = pd.DataFrame({
            "GAME_ID": pd.array([1], dtype="Int64"),
            "PTS_10": np.array([0.123456789], dtype=np.float64),
            "GAME_DATE": pd.to_datetime(["2024-01-01"], utc=True),
            "EMPTY": [np.nan],
        })
        records = nba_gamelogs.prepare_records(df)
        # Should not raise — all types are JSON-serializable
        json.dumps(records)
