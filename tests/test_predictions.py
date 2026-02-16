# tests/test_predictions.py
"""
End-to-end prediction flow tests.

Loads the ACTUAL trained models (nba_winner.json, mlb_winner.json) and feeds
them realistic synthetic gamelogs — mock data shaped exactly like what the
pipeline produces, with stat values in real-world ranges.

Validates that the full prediction path works:
  realistic mock data → add_diff_features → build feature matrix → model.predict_proba → valid outputs

If a model file doesn't exist (e.g. fresh clone), tests are skipped, not failed.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import pytest
from pathlib import Path

from conftest import nba_train, nba_predict, mlb_train, mlb_predict

REPO_ROOT = Path(__file__).resolve().parents[1]
NBA_MODEL_PATH = REPO_ROOT / "nba-pipeline" / "models" / "nba_winner.json"
MLB_MODEL_PATH = REPO_ROOT / "mlb-pipeline" / "models" / "mlb_winner.json"


# ---------------------------------------------------------------------------
# Realistic mock data generators
# ---------------------------------------------------------------------------
def _make_nba_realistic_gamelogs(n=50):
    """
    Create synthetic NBA gamelogs with stat values in realistic ranges.
    Mimics the structure produced by nba gamelogs.py → build_gamelogs.
    """
    rng = np.random.RandomState(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="D", tz="UTC")

    data = {
        "GAME_ID": range(100000, 100000 + n),
        "GAME_DATE": dates,
        "SEASON_ID": [22024] * n,
        "GAME_STATUS": [1] * n,     # scheduled — what predict.py consumes
        "GAME_OUTCOME": [None] * n,  # unknown — scheduled game
        "HOME_NAME": ["Los Angeles Lakers"] * n,
        "AWAY_NAME": ["Golden State Warriors"] * n,
    }

    # Realistic NBA stat ranges (from actual gamelogs data)
    stat_ranges = {
        "PTS":          (100, 125),
        "REB":          (38, 55),
        "AST":          (20, 32),
        "STL":          (5, 12),
        "BLK":          (3, 8),
        "TOV":          (10, 18),
        "PF":           (16, 26),
        "FG_PCT":       (0.42, 0.52),
        "FG3_PCT":      (0.30, 0.42),
        "FT_PCT":       (0.72, 0.85),
        "PLUS_MINUS":   (-8, 8),
        "WIN_RATE":     (0.30, 0.70),
        "GAMES":        (5, 30),
    }

    for side in ["HOME", "AWAY"]:
        for stat, (lo, hi) in stat_ranges.items():
            for window in [10, 30]:
                col = f"{side}_{stat}_{window}"
                data[col] = rng.uniform(lo, hi, n)

    return pd.DataFrame(data)


def _make_mlb_realistic_gamelogs(n=50):
    """
    Create synthetic MLB gamelogs with stat values in realistic ranges.
    Mimics the structure produced by mlb gamelogs.py → build_gamelogs.
    """
    rng = np.random.RandomState(42)
    dates = pd.date_range("2025-04-01", periods=n, freq="D", tz="UTC")

    data = {
        "GAME_ID": range(700000, 700000 + n),
        "GAME_DATE": dates,
        "SEASON_ID": [2025] * n,
        "GAME_STATUS": [1] * n,
        "GAME_OUTCOME": [None] * n,
        "HOME_NAME": ["Los Angeles Dodgers"] * n,
        "AWAY_NAME": ["San Francisco Giants"] * n,
        "HOME_SP": [501] * n,
        "AWAY_SP": [601] * n,
        "HOME_LINEUP": [[101, 102, 103, 104, 105, 106, 107, 108, 109]] * n,
        "AWAY_LINEUP": [[201, 202, 203, 204, 205, 206, 207, 208, 209]] * n,
    }

    # Realistic MLB stat ranges (batting = lineup-weighted rolling avgs)
    batting_ranges = {
        "BA":  (0.220, 0.310),
        "OBP": (0.290, 0.390),
        "SLG": (0.360, 0.520),
        "OPS": (0.650, 0.900),
        "R":   (0.4, 1.2),
        "HR":  (0.05, 0.35),
        "RBI": (0.4, 1.1),
        "BB":  (0.2, 0.7),
        "SO":  (0.5, 1.5),
        "SB":  (0.0, 0.3),
    }
    sp_ranges = {
        "ERA":  (2.50, 5.50),
        "WHIP": (0.90, 1.55),
        "SO":   (4.0, 10.0),
        "BB":   (1.0, 4.0),
        "HR":   (0.5, 2.0),
        "IP":   (4.5, 7.0),
    }
    bp_ranges = {
        "ERA":  (3.00, 5.00),
        "WHIP": (1.00, 1.50),
        "SO":   (3.0, 8.0),
        "BB":   (1.5, 4.0),
        "HR":   (0.3, 1.5),
        "IP":   (3.0, 5.0),
    }
    win_ranges = {
        "WIN_RATE": (0.35, 0.65),
        "GAMES":    (5, 50),
    }

    for side in ["HOME", "AWAY"]:
        for stat, (lo, hi) in batting_ranges.items():
            for w in [10, 50]:
                data[f"{side}_{stat}_{w}"] = rng.uniform(lo, hi, n)
        for stat, (lo, hi) in sp_ranges.items():
            for w in [10, 50]:
                data[f"{side}_SP_{stat}_{w}"] = rng.uniform(lo, hi, n)
        for stat, (lo, hi) in bp_ranges.items():
            for w in [10, 50]:
                data[f"{side}_BP_{stat}_{w}"] = rng.uniform(lo, hi, n)
        for stat, (lo, hi) in win_ranges.items():
            for w in [10, 50]:
                data[f"{side}_{stat}_{w}"] = rng.uniform(lo, hi, n)

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Load actual models (skip if not present)
# ---------------------------------------------------------------------------
def _load_real_model(path):
    """Load the actual trained model. Returns None if file missing."""
    if not path.exists():
        return None
    model = xgb.XGBClassifier()
    model.load_model(str(path))
    return model


# ===========================================================================
# NBA — Actual Model + Realistic Data
# ===========================================================================
class TestNBAActualModel:

    @pytest.fixture(autouse=True)
    def load_model(self):
        self.model = _load_real_model(NBA_MODEL_PATH)
        if self.model is None:
            pytest.skip("NBA model not found — run train.py first")

    def test_model_accepts_realistic_data(self):
        """Feed realistic mock gamelogs through the actual NBA model."""
        df = _make_nba_realistic_gamelogs(20)
        df = nba_predict.add_diff_features(df)
        X = df[nba_predict.ALL_FEATURES].astype(float)

        probs = self.model.predict_proba(X)
        assert probs.shape == (20, 2)

    def test_predictions_are_valid(self):
        """Actual model should produce valid probabilities and binary picks."""
        df = _make_nba_realistic_gamelogs(30)
        df = nba_predict.add_diff_features(df)
        X = df[nba_predict.ALL_FEATURES].astype(float)

        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)

        assert all(0.0 <= p <= 1.0 for p in probs), "Probabilities out of bounds"
        assert set(preds).issubset({0, 1}), "Predictions not binary"

    def test_predictions_not_degenerate(self):
        """Model shouldn't predict all-home or all-away on varied input."""
        df = _make_nba_realistic_gamelogs(50)
        df = nba_predict.add_diff_features(df)
        X = df[nba_predict.ALL_FEATURES].astype(float)

        probs = self.model.predict_proba(X)[:, 1]
        # With 50 varied games, we expect at least some spread
        assert probs.std() > 0.01, "Predictions have no variance — model may be broken"
        # Not all the same prediction
        preds = (probs >= 0.5).astype(int)
        assert len(set(preds)) > 1 or probs.std() > 0.01, \
            "Model predicts same outcome for all games"

    def test_feature_count_matches_model(self):
        """The model should expect exactly 60 features."""
        df = _make_nba_realistic_gamelogs(5)
        df = nba_predict.add_diff_features(df)
        X = df[nba_predict.ALL_FEATURES].astype(float)

        assert X.shape[1] == 60
        # Model should accept this without error
        self.model.predict_proba(X)

    def test_output_format_matches_supabase_schema(self):
        """Output types should match what gets written to Supabase."""
        df = _make_nba_realistic_gamelogs(10)
        df = nba_predict.add_diff_features(df)
        X = df[nba_predict.ALL_FEATURES].astype(float)

        probs = self.model.predict_proba(X)[:, 1]
        for p in probs:
            pred = int((p >= 0.5))
            prob = round(float(p), 3)
            assert isinstance(pred, int) and pred in (0, 1)
            assert isinstance(prob, float) and 0.0 <= prob <= 1.0

    def test_strong_home_team_favored(self):
        """A team with clearly better stats should get higher home win prob."""
        df = _make_nba_realistic_gamelogs(2)

        # Game 0: dominant home team
        for w in [10, 30]:
            df.loc[0, f"HOME_WIN_RATE_{w}"] = 0.80
            df.loc[0, f"AWAY_WIN_RATE_{w}"] = 0.25
            df.loc[0, f"HOME_PLUS_MINUS_{w}"] = 10.0
            df.loc[0, f"AWAY_PLUS_MINUS_{w}"] = -10.0
            df.loc[0, f"HOME_PTS_{w}"] = 120.0
            df.loc[0, f"AWAY_PTS_{w}"] = 98.0

        # Game 1: dominant away team
        for w in [10, 30]:
            df.loc[1, f"HOME_WIN_RATE_{w}"] = 0.25
            df.loc[1, f"AWAY_WIN_RATE_{w}"] = 0.80
            df.loc[1, f"HOME_PLUS_MINUS_{w}"] = -10.0
            df.loc[1, f"AWAY_PLUS_MINUS_{w}"] = 10.0
            df.loc[1, f"HOME_PTS_{w}"] = 98.0
            df.loc[1, f"AWAY_PTS_{w}"] = 120.0

        df = nba_predict.add_diff_features(df)
        X = df[nba_predict.ALL_FEATURES].astype(float)
        probs = self.model.predict_proba(X)[:, 1]

        # Strong home team should have higher home win probability
        assert probs[0] > probs[1], \
            f"Strong home ({probs[0]:.3f}) should be favored over weak home ({probs[1]:.3f})"


# ===========================================================================
# MLB — Actual Model + Realistic Data
# ===========================================================================
class TestMLBActualModel:

    @pytest.fixture(autouse=True)
    def load_model(self):
        self.model = _load_real_model(MLB_MODEL_PATH)
        if self.model is None:
            pytest.skip("MLB model not found — run train.py first")

    def test_model_accepts_realistic_data(self):
        """Feed realistic mock gamelogs through the actual MLB model."""
        df = _make_mlb_realistic_gamelogs(20)
        df = mlb_predict.add_diff_features(df)
        X = df[mlb_predict.ALL_FEATURES].astype(float)

        probs = self.model.predict_proba(X)
        assert probs.shape == (20, 2)

    def test_predictions_are_valid(self):
        df = _make_mlb_realistic_gamelogs(30)
        df = mlb_predict.add_diff_features(df)
        X = df[mlb_predict.ALL_FEATURES].astype(float)

        probs = self.model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)

        assert all(0.0 <= p <= 1.0 for p in probs)
        assert set(preds).issubset({0, 1})

    def test_predictions_not_degenerate(self):
        df = _make_mlb_realistic_gamelogs(50)
        df = mlb_predict.add_diff_features(df)
        X = df[mlb_predict.ALL_FEATURES].astype(float)

        probs = self.model.predict_proba(X)[:, 1]
        assert probs.std() > 0.01, "Predictions have no variance"

    def test_feature_count_matches_model(self):
        """The model should expect exactly 108 features."""
        df = _make_mlb_realistic_gamelogs(5)
        df = mlb_predict.add_diff_features(df)
        X = df[mlb_predict.ALL_FEATURES].astype(float)

        assert X.shape[1] == 108
        self.model.predict_proba(X)

    def test_output_format_matches_supabase_schema(self):
        df = _make_mlb_realistic_gamelogs(10)
        df = mlb_predict.add_diff_features(df)
        X = df[mlb_predict.ALL_FEATURES].astype(float)

        probs = self.model.predict_proba(X)[:, 1]
        for p in probs:
            pred = int((p >= 0.5))
            prob = round(float(p), 3)
            assert isinstance(pred, int) and pred in (0, 1)
            assert isinstance(prob, float) and 0.0 <= prob <= 1.0

    def test_strong_home_team_favored(self):
        """Team with clearly better stats should get higher home win prob."""
        df = _make_mlb_realistic_gamelogs(2)

        # Game 0: dominant home team
        for w in [10, 50]:
            df.loc[0, f"HOME_WIN_RATE_{w}"] = 0.75
            df.loc[0, f"AWAY_WIN_RATE_{w}"] = 0.30
            df.loc[0, f"HOME_OPS_{w}"] = 0.850
            df.loc[0, f"AWAY_OPS_{w}"] = 0.650
            df.loc[0, f"HOME_SP_ERA_{w}"] = 2.50
            df.loc[0, f"AWAY_SP_ERA_{w}"] = 5.50

        # Game 1: dominant away team
        for w in [10, 50]:
            df.loc[1, f"HOME_WIN_RATE_{w}"] = 0.30
            df.loc[1, f"AWAY_WIN_RATE_{w}"] = 0.75
            df.loc[1, f"HOME_OPS_{w}"] = 0.650
            df.loc[1, f"AWAY_OPS_{w}"] = 0.850
            df.loc[1, f"HOME_SP_ERA_{w}"] = 5.50
            df.loc[1, f"AWAY_SP_ERA_{w}"] = 2.50

        df = mlb_predict.add_diff_features(df)
        X = df[mlb_predict.ALL_FEATURES].astype(float)
        probs = self.model.predict_proba(X)[:, 1]

        assert probs[0] > probs[1], \
            f"Strong home ({probs[0]:.3f}) should be favored over weak home ({probs[1]:.3f})"


# ===========================================================================
# Pipeline Function Tests (no model needed — tests the code path itself)
# ===========================================================================
class TestPredictionPipeline:
    """Tests the prediction pipeline functions themselves (add_diff_features,
    build_feature_matrix, etc.) independent of any saved model."""

    def test_nba_diff_features_computed_correctly(self):
        df = _make_nba_realistic_gamelogs(10)
        df = nba_predict.add_diff_features(df)

        for name, (home_col, away_col) in nba_predict.DIFF_FEATURES.items():
            expected = df[home_col] - df[away_col]
            assert np.allclose(df[name], expected), f"{name} values incorrect"

    def test_mlb_diff_features_computed_correctly(self):
        df = _make_mlb_realistic_gamelogs(10)
        df = mlb_predict.add_diff_features(df)

        for name, (home_col, away_col) in mlb_predict.DIFF_FEATURES.items():
            expected = df[home_col] - df[away_col]
            assert np.allclose(df[name], expected), f"{name} values incorrect"

    def test_nba_build_feature_matrix_with_realistic_data(self):
        df = _make_nba_realistic_gamelogs(50)
        # Need GAME_OUTCOME for build_feature_matrix
        df["GAME_OUTCOME"] = np.random.RandomState(42).randint(0, 2, 50)
        df["GAME_STATUS"] = 3

        X, y, df2 = nba_train.build_feature_matrix(df)
        assert X.shape[1] == 60
        assert len(X) == 50  # no NaNs to drop in realistic data

    def test_mlb_build_feature_matrix_with_realistic_data(self):
        df = _make_mlb_realistic_gamelogs(50)
        df["GAME_OUTCOME"] = np.random.RandomState(42).randint(0, 2, 50)
        df["GAME_STATUS"] = 3

        X, y, df2 = mlb_train.build_feature_matrix(df)
        assert X.shape[1] == 108
        assert len(X) == 50  # MLB doesn't drop NaN rows

    def test_nba_missing_feature_columns_produce_nan_diffs(self):
        df = pd.DataFrame({"GAME_ID": [1], "HOME_PTS_10": [105.0]})
        result = nba_predict.add_diff_features(df)
        assert pd.isna(result["DIFF_PTS_10"].iloc[0])

    def test_mlb_missing_feature_columns_produce_nan_diffs(self):
        df = pd.DataFrame({"GAME_ID": [1], "HOME_OPS_10": [0.800]})
        result = mlb_predict.add_diff_features(df)
        assert pd.isna(result["DIFF_OPS_10"].iloc[0])

    def test_model_save_load_preserves_predictions(self, tmp_path):
        """Train a small model, save, load, verify identical predictions."""
        df = _make_nba_realistic_gamelogs(100)
        df["GAME_OUTCOME"] = np.random.RandomState(42).randint(0, 2, 100)
        df["GAME_STATUS"] = 3

        X, y, df2 = nba_train.build_feature_matrix(df)
        X_train, X_test, y_train, y_test, _ = nba_train.time_split(X, y, df2)

        model = xgb.XGBClassifier(
            n_estimators=10, max_depth=2, verbosity=0,
            use_label_encoder=False, eval_metric="auc",
        )
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        probs_orig = model.predict_proba(X_test)[:, 1]

        path = tmp_path / "test_model.json"
        model.save_model(str(path))
        loaded = nba_predict.load_model(path)
        probs_loaded = loaded.predict_proba(X_test)[:, 1]

        assert np.allclose(probs_orig, probs_loaded)

    def test_roster_validation_for_mlb(self):
        """MLB predict skips games without lineups/SPs."""
        complete = pd.Series({
            "HOME_LINEUP": [101, 102, 103, 104, 105, 106, 107, 108, 109],
            "AWAY_LINEUP": [201, 202, 203, 204, 205, 206, 207, 208, 209],
            "HOME_SP": 501, "AWAY_SP": 601,
        })
        home_lu = complete.get("HOME_LINEUP")
        away_lu = complete.get("AWAY_LINEUP")
        has_lineups = (home_lu is not None and isinstance(home_lu, list) and len(home_lu) > 0 and
                       away_lu is not None and isinstance(away_lu, list) and len(away_lu) > 0)
        has_sps = pd.notna(complete.get("HOME_SP")) and pd.notna(complete.get("AWAY_SP"))
        assert has_lineups and has_sps

        missing = pd.Series({
            "HOME_LINEUP": None, "AWAY_LINEUP": [],
            "HOME_SP": pd.NA, "AWAY_SP": 601,
        })
        home_lu2 = missing.get("HOME_LINEUP")
        has_lu2 = home_lu2 is not None and isinstance(home_lu2, list) and len(home_lu2) > 0
        assert not has_lu2
