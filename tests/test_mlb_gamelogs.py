# tests/test_mlb_gamelogs.py
"""
Tests for MLB gamelogs feature engineering logic.

Tests the pure computational functions using synthetic data:
- compute_player_batting_rolling: per-player rolling with shift(1)
- compute_player_pitching_rolling: same for pitching stats
- get_latest_player_rolling: latest values extraction
- compute_team_win_rolling: team win rate rolling
- build_gamelogs: lineup-weighted feature assembly
- LINEUP_WEIGHTS: normalization correctness
"""

import numpy as np
import pandas as pd

# Import the pre-loaded modules from conftest (avoids name collisions)
from conftest import mlb_gamelogs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_batting_df(player_ids, n_games_per_player=5):
    """Create synthetic batting stats for multiple players."""
    rng = np.random.RandomState(42)
    rows = []
    base_game_id = 1000

    for pid in player_ids:
        dates = pd.date_range("2024-04-01", periods=n_games_per_player, freq="D", tz="UTC")
        for i, date in enumerate(dates):
            rows.append({
                "GAME_ID": base_game_id + i,
                "PLAYER_ID": pid,
                "GAME_DATE": date,
                "TEAM_ID": 119,  # Dodgers
                "OPPONENT_ID": 137,
                "IS_HOME": True,
                "AB": rng.randint(2, 5),
                "H": rng.randint(0, 3),
                "R": rng.randint(0, 2),
                "HR": rng.randint(0, 2),
                "RBI": rng.randint(0, 3),
                "BB": rng.randint(0, 2),
                "SO": rng.randint(0, 3),
                "SB": rng.randint(0, 2),
                "PA": rng.randint(3, 6),
                "BA": round(rng.uniform(0.200, 0.350), 3),
                "OBP": round(rng.uniform(0.280, 0.420), 3),
                "SLG": round(rng.uniform(0.350, 0.550), 3),
                "OPS": round(rng.uniform(0.650, 0.950), 3),
            })

    df = pd.DataFrame(rows)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], utc=True)
    for col in ["GAME_ID", "PLAYER_ID", "TEAM_ID"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in mlb_gamelogs.BATTING_ROLLING_STATS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def make_pitching_df(player_ids, n_games_per_player=5):
    """Create synthetic pitching stats for multiple players."""
    rng = np.random.RandomState(42)
    rows = []
    base_game_id = 1000

    for pid in player_ids:
        dates = pd.date_range("2024-04-01", periods=n_games_per_player, freq="D", tz="UTC")
        for i, date in enumerate(dates):
            rows.append({
                "GAME_ID": base_game_id + i,
                "PLAYER_ID": pid,
                "GAME_DATE": date,
                "TEAM_ID": 119,
                "OPPONENT_ID": 137,
                "IS_HOME": True,
                "IP": round(rng.uniform(4.0, 7.0), 1),
                "H_P": rng.randint(3, 8),
                "R_P": rng.randint(1, 5),
                "ER": rng.randint(1, 4),
                "BB_P": rng.randint(0, 4),
                "SO_P": rng.randint(3, 10),
                "HR_P": rng.randint(0, 2),
                "BF": rng.randint(20, 30),
                "PIT": rng.randint(60, 100),
                "ERA": round(rng.uniform(2.5, 5.0), 2),
                "WHIP": round(rng.uniform(0.9, 1.5), 3),
            })

    df = pd.DataFrame(rows)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], utc=True)
    for col in ["GAME_ID", "PLAYER_ID", "TEAM_ID"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in mlb_gamelogs.PITCHING_ROLLING_STATS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def make_mlb_games_df(n=5):
    """Create synthetic MLB games DataFrame."""
    dates = pd.date_range("2024-04-01", periods=n, freq="D", tz="UTC")
    rng = np.random.RandomState(42)

    return pd.DataFrame({
        "GAME_ID": pd.array(range(1000, 1000 + n), dtype="Int64"),
        "SEASON_ID": pd.array([2024] * n, dtype="Int64"),
        "GAME_DATE": dates,
        "AWAY_NAME": ["San Francisco Giants"] * n,
        "HOME_NAME": ["Los Angeles Dodgers"] * n,
        "AWAY_ID": pd.array([137] * n, dtype="Int64"),
        "HOME_ID": pd.array([119] * n, dtype="Int64"),
        "GAME_STATUS": pd.array([3] * n, dtype="Int64"),
        "GAME_OUTCOME": pd.array(rng.randint(0, 2, n), dtype="Int64"),
        "AWAY_RUNS": pd.array(rng.randint(1, 8, n), dtype="Int64"),
        "HOME_RUNS": pd.array(rng.randint(1, 8, n), dtype="Int64"),
        "TOTAL_RUNS": pd.array(rng.randint(3, 15, n), dtype="Int64"),
        "HOME_SP": pd.array([501] * n, dtype="Int64"),
        "AWAY_SP": pd.array([601] * n, dtype="Int64"),
        "HOME_LINEUP": [[101, 102, 103, 104, 105, 106, 107, 108, 109]] * n,
        "AWAY_LINEUP": [[201, 202, 203, 204, 205, 206, 207, 208, 209]] * n,
        "HOME_BULLPEN": [[502, 503, 504]] * n,
        "AWAY_BULLPEN": [[602, 603, 604]] * n,
    })


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestLineupWeights:

    def test_weights_sum_to_one(self):
        assert abs(mlb_gamelogs.LINEUP_WEIGHTS_NORM.sum() - 1.0) < 1e-10

    def test_weights_are_decreasing(self):
        """Batter 1 gets highest weight, batter 9 gets lowest."""
        weights = mlb_gamelogs.LINEUP_WEIGHTS_NORM
        for i in range(len(weights) - 1):
            assert weights[i] > weights[i + 1], \
                f"Weight {i} ({weights[i]}) should be > weight {i+1} ({weights[i+1]})"

    def test_nine_weights(self):
        assert len(mlb_gamelogs.LINEUP_WEIGHTS) == 9
        assert len(mlb_gamelogs.LINEUP_WEIGHTS_NORM) == 9

    def test_raw_weights_are_9_to_1(self):
        expected = [9, 8, 7, 6, 5, 4, 3, 2, 1]
        assert list(mlb_gamelogs.LINEUP_WEIGHTS) == expected


class TestComputePlayerBattingRolling:

    def test_returns_dict_structure(self):
        batting = make_batting_df([100, 200], n_games_per_player=3)
        result = mlb_gamelogs.compute_player_batting_rolling(batting)

        assert isinstance(result, dict)
        assert 100 in result
        assert 200 in result

    def test_rolling_keys_per_game(self):
        batting = make_batting_df([100], n_games_per_player=3)
        result = mlb_gamelogs.compute_player_batting_rolling(batting)

        # Each game should have keys like "BA_10", "BA_50", etc.
        game_vals = list(result[100].values())[1]  # second game
        for stat in mlb_gamelogs.BATTING_ROLLING_STATS:
            for window in mlb_gamelogs.WINDOWS:
                key = f"{stat}_{window}"
                assert key in game_vals, f"Missing rolling key: {key}"

    def test_first_game_is_none(self):
        """shift(1) means the first game has NaN → None."""
        batting = make_batting_df([100], n_games_per_player=3)
        result = mlb_gamelogs.compute_player_batting_rolling(batting)

        first_game_id = min(result[100].keys())
        first_vals = result[100][first_game_id]
        assert first_vals["BA_10"] is None, "First game rolling should be None (shift)"

    def test_no_data_leakage(self):
        """Game N rolling should not include game N's stats."""
        batting = make_batting_df([100], n_games_per_player=5)
        result = mlb_gamelogs.compute_player_batting_rolling(batting)

        game_ids = sorted(result[100].keys())
        # Game 2 (index 2) rolling_10 should be avg of game 0 and game 1
        game_2_ba = result[100][game_ids[2]]["BA_10"]

        ba_values = batting[batting["PLAYER_ID"] == 100].sort_values("GAME_DATE")["BA"].values
        expected = (ba_values[0] + ba_values[1]) / 2

        assert abs(game_2_ba - expected) < 1e-6, \
            f"Leakage: game 2 rolling ({game_2_ba}) != avg of games 0+1 ({expected})"

    def test_empty_input(self):
        result = mlb_gamelogs.compute_player_batting_rolling(pd.DataFrame())
        assert result == {}


class TestComputePlayerPitchingRolling:

    def test_returns_dict_structure(self):
        pitching = make_pitching_df([500, 600], n_games_per_player=3)
        result = mlb_gamelogs.compute_player_pitching_rolling(pitching)

        assert 500 in result
        assert 600 in result

    def test_stat_name_remapping(self):
        """Pitching rolling should use output names (e.g., SO_P → SO)."""
        pitching = make_pitching_df([500], n_games_per_player=3)
        result = mlb_gamelogs.compute_player_pitching_rolling(pitching)

        game_ids = sorted(result[500].keys())
        vals = result[500][game_ids[1]]  # second game

        for output_stat in mlb_gamelogs.PITCHING_OUTPUT_STATS:
            for window in mlb_gamelogs.WINDOWS:
                key = f"{output_stat}_{window}"
                assert key in vals, f"Missing remapped key: {key}"


class TestGetLatestPlayerRolling:

    def test_returns_latest_game_values(self):
        batting = make_batting_df([100], n_games_per_player=5)
        rolling = mlb_gamelogs.compute_player_batting_rolling(batting)
        latest = mlb_gamelogs.get_latest_player_rolling(rolling)

        assert 100 in latest
        # Latest should be from the last game
        last_game_id = max(rolling[100].keys())
        assert latest[100] == rolling[100][last_game_id]

    def test_empty_input(self):
        result = mlb_gamelogs.get_latest_player_rolling({})
        assert result == {}


class TestComputeTeamWinRolling:

    def test_returns_two_dicts(self):
        games = make_mlb_games_df(n=10)
        rolling, latest = mlb_gamelogs.compute_team_win_rolling(games)

        assert isinstance(rolling, dict)
        assert isinstance(latest, dict)

    def test_both_teams_present(self):
        games = make_mlb_games_df(n=5)
        rolling, latest = mlb_gamelogs.compute_team_win_rolling(games)

        assert 119 in rolling  # Dodgers (home)
        assert 137 in rolling  # Giants (away)

    def test_win_rate_keys(self):
        games = make_mlb_games_df(n=5)
        rolling, latest = mlb_gamelogs.compute_team_win_rolling(games)

        for tid in rolling:
            for gid in rolling[tid]:
                vals = rolling[tid][gid]
                for w in mlb_gamelogs.WINDOWS:
                    assert f"WIN_RATE_{w}" in vals
                    assert f"GAMES_{w}" in vals

    def test_win_rate_bounded(self):
        """Win rate should be between 0 and 1."""
        games = make_mlb_games_df(n=10)
        rolling, _ = mlb_gamelogs.compute_team_win_rolling(games)

        for tid in rolling:
            for gid in rolling[tid]:
                for w in mlb_gamelogs.WINDOWS:
                    wr = rolling[tid][gid].get(f"WIN_RATE_{w}")
                    if wr is not None:
                        assert 0.0 <= wr <= 1.0, f"Win rate {wr} out of bounds"

    def test_empty_input(self):
        empty = pd.DataFrame(columns=["GAME_ID", "GAME_DATE", "HOME_ID", "AWAY_ID",
                                       "GAME_STATUS", "GAME_OUTCOME"])
        rolling, latest = mlb_gamelogs.compute_team_win_rolling(empty)
        assert rolling == {}
        assert latest == {}


class TestBuildGamelogs:

    def test_returns_list_of_dicts(self):
        games = make_mlb_games_df(n=3)
        batting = make_batting_df(list(range(101, 110)) + list(range(201, 210)), n_games_per_player=3)
        pitching = make_pitching_df([501, 502, 503, 601, 602, 603], n_games_per_player=3)

        b_rolling = mlb_gamelogs.compute_player_batting_rolling(batting)
        p_rolling = mlb_gamelogs.compute_player_pitching_rolling(pitching)
        b_latest = mlb_gamelogs.get_latest_player_rolling(b_rolling)
        p_latest = mlb_gamelogs.get_latest_player_rolling(p_rolling)
        tw_rolling, tw_latest = mlb_gamelogs.compute_team_win_rolling(games)

        records = mlb_gamelogs.build_gamelogs(
            games, b_rolling, p_rolling, tw_rolling,
            b_latest, p_latest, tw_latest
        )

        assert isinstance(records, list)
        assert len(records) == 3  # one per game

    def test_has_batting_sp_bp_columns(self):
        games = make_mlb_games_df(n=3)
        batting = make_batting_df(list(range(101, 110)) + list(range(201, 210)), n_games_per_player=3)
        pitching = make_pitching_df([501, 502, 503, 601, 602, 603], n_games_per_player=3)

        b_rolling = mlb_gamelogs.compute_player_batting_rolling(batting)
        p_rolling = mlb_gamelogs.compute_player_pitching_rolling(pitching)
        b_latest = mlb_gamelogs.get_latest_player_rolling(b_rolling)
        p_latest = mlb_gamelogs.get_latest_player_rolling(p_rolling)
        tw_rolling, tw_latest = mlb_gamelogs.compute_team_win_rolling(games)

        records = mlb_gamelogs.build_gamelogs(
            games, b_rolling, p_rolling, tw_rolling,
            b_latest, p_latest, tw_latest
        )

        rec = records[0]
        # Batting columns
        assert "HOME_BA_10" in rec
        assert "AWAY_OPS_50" in rec
        # SP columns
        assert "HOME_SP_ERA_10" in rec
        assert "AWAY_SP_WHIP_50" in rec
        # BP columns
        assert "HOME_BP_ERA_10" in rec
        assert "AWAY_BP_SO_50" in rec
        # Win rate columns
        assert "HOME_WIN_RATE_10" in rec
        assert "AWAY_GAMES_50" in rec

    def test_game_metadata_preserved(self):
        games = make_mlb_games_df(n=2)
        records = mlb_gamelogs.build_gamelogs(
            games, {}, {}, {}, {}, {}, {}
        )
        rec = records[0]
        assert "GAME_ID" in rec
        assert "GAME_DATE" in rec
        assert "HOME_NAME" in rec
        assert "AWAY_NAME" in rec
        assert "HOME_LINEUP" in rec
        assert "HOME_SP" in rec

    def test_empty_games(self):
        empty = pd.DataFrame()
        records = mlb_gamelogs.build_gamelogs(empty, {}, {}, {}, {}, {}, {})
        assert records == []


class TestRollingWindowConfig:

    def test_windows_are_10_and_50(self):
        assert mlb_gamelogs.WINDOWS == [10, 50]

    def test_batting_rolling_stats_count(self):
        assert len(mlb_gamelogs.BATTING_ROLLING_STATS) == 10

    def test_pitching_stat_map_count(self):
        assert len(mlb_gamelogs.PITCHING_STAT_MAP) == 6

    def test_pitching_output_stat_names(self):
        expected = ["ERA", "WHIP", "SO", "BB", "HR", "IP"]
        assert mlb_gamelogs.PITCHING_OUTPUT_STATS == expected
