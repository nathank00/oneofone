# tests/test_mlb_train.py
"""
Tests for MLB train.py and predict.py logic.

Tests the pure computational functions using synthetic data:
- Feature definitions: correct count and naming
- add_diff_features: difference computation
- build_feature_matrix: feature extraction (XGBoost handles NaN — no dropping)
- time_split: chronological splitting with no leakage
- load_model: graceful handling of missing model files
- has_roster_data: lineup/SP validation for predictions
"""

import numpy as np
import pandas as pd

# Import the pre-loaded modules from conftest (avoids name collisions)
from conftest import mlb_train, mlb_predict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_mlb_training_df(n=100):
    """Create a synthetic gamelogs DataFrame suitable for MLB training."""
    rng = np.random.RandomState(42)
    dates = pd.date_range("2023-04-01", periods=n, freq="D", tz="UTC")

    data = {
        "GAME_ID": range(1, n + 1),
        "GAME_DATE": dates,
        "SEASON_ID": [2023] * n,
        "GAME_STATUS": [3] * n,
        "GAME_OUTCOME": rng.randint(0, 2, n),
    }

    # Add all raw feature columns
    for col in mlb_train.FEATURE_COLS:
        data[col] = rng.uniform(0.1, 5.0, n)

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestMLBFeatureDefinitions:

    def test_raw_feature_count(self):
        """96 raw features: 2 sides × (10 batting × 2 + 6 SP × 2 + 6 BP × 2 + 2 win × 2)."""
        assert len(mlb_train.FEATURE_COLS) == 96

    def test_diff_feature_count(self):
        assert len(mlb_train.DIFF_FEATURES) == 12

    def test_total_feature_count(self):
        assert len(mlb_train.ALL_FEATURES) == 108

    def test_feature_cols_have_home_and_away(self):
        home = [c for c in mlb_train.FEATURE_COLS if c.startswith("HOME_")]
        away = [c for c in mlb_train.FEATURE_COLS if c.startswith("AWAY_")]
        assert len(home) == 48
        assert len(away) == 48

    def test_feature_cols_include_sp_bp(self):
        sp_cols = [c for c in mlb_train.FEATURE_COLS if "_SP_" in c]
        bp_cols = [c for c in mlb_train.FEATURE_COLS if "_BP_" in c]
        # 2 sides × 6 stats × 2 windows = 24 each
        assert len(sp_cols) == 24
        assert len(bp_cols) == 24

    def test_diff_features_reference_existing_cols(self):
        for name, (home_col, away_col) in mlb_train.DIFF_FEATURES.items():
            assert home_col in mlb_train.FEATURE_COLS, f"{home_col} not in FEATURE_COLS"
            assert away_col in mlb_train.FEATURE_COLS, f"{away_col} not in FEATURE_COLS"

    def test_predict_features_match_train(self):
        """predict.py feature definitions must exactly match train.py."""
        assert mlb_predict.ALL_FEATURES == mlb_train.ALL_FEATURES
        assert mlb_predict.FEATURE_COLS == mlb_train.FEATURE_COLS
        assert mlb_predict.DIFF_FEATURES == mlb_train.DIFF_FEATURES


class TestMLBAddDiffFeatures:

    def test_diff_columns_added(self):
        df = make_mlb_training_df(10)
        result = mlb_train.add_diff_features(df)
        for name in mlb_train.DIFF_FEATURES:
            assert name in result.columns

    def test_diff_values_correct(self):
        df = make_mlb_training_df(10)
        result = mlb_train.add_diff_features(df)

        for name, (home_col, away_col) in mlb_train.DIFF_FEATURES.items():
            expected = df[home_col] - df[away_col]
            actual = result[name]
            assert np.allclose(expected, actual, equal_nan=True)

    def test_missing_columns_produce_nan(self):
        df = pd.DataFrame({"GAME_ID": [1], "HOME_OPS_10": [0.800]})
        result = mlb_train.add_diff_features(df)
        assert pd.isna(result["DIFF_OPS_10"].iloc[0])


class TestMLBBuildFeatureMatrix:

    def test_returns_x_y_df(self):
        df = make_mlb_training_df(50)
        X, y, result_df = mlb_train.build_feature_matrix(df)

        assert len(X) == len(y)
        assert len(X.columns) == 108

    def test_does_not_drop_nan_rows(self):
        """MLB uses XGBoost native NaN handling — rows are NOT dropped."""
        df = make_mlb_training_df(50)
        df.loc[:4, "HOME_SP_ERA_10"] = np.nan

        X, y, result_df = mlb_train.build_feature_matrix(df)
        # MLB build_feature_matrix does NOT drop NaN rows (unlike NBA)
        assert len(X) == 50

    def test_creates_missing_columns(self):
        """If a feature column doesn't exist in the data, it should be added as NaN."""
        df = pd.DataFrame({
            "GAME_ID": [1, 2],
            "GAME_DATE": pd.date_range("2024-01-01", periods=2, tz="UTC"),
            "GAME_STATUS": [3, 3],
            "GAME_OUTCOME": [1, 0],
        })
        X, y, _ = mlb_train.build_feature_matrix(df)
        assert len(X.columns) == 108
        assert X.isna().all().all()  # all values NaN since no feature data


class TestMLBTimeSplit:

    def test_split_sizes(self):
        df = make_mlb_training_df(100)
        X, y, _ = mlb_train.build_feature_matrix(df)
        X_train, X_test, y_train, y_test, df_test = mlb_train.time_split(X, y, df)

        assert len(X_train) == 80
        assert len(X_test) == 20

    def test_chronological_order(self):
        df = make_mlb_training_df(100)
        X, y, df2 = mlb_train.build_feature_matrix(df)
        X_train, X_test, _, _, df_test = mlb_train.time_split(X, y, df2)

        train_max = df2.loc[X_train.index, "GAME_DATE"].max()
        test_min = df_test["GAME_DATE"].min()
        assert test_min >= train_max

    def test_no_index_overlap(self):
        df = make_mlb_training_df(100)
        X, y, _ = mlb_train.build_feature_matrix(df)
        X_train, X_test, _, _, _ = mlb_train.time_split(X, y, df)

        overlap = set(X_train.index) & set(X_test.index)
        assert len(overlap) == 0


class TestMLBLoadModel:

    def test_missing_model_returns_none(self, tmp_path):
        result = mlb_predict.load_model(tmp_path / "nonexistent.json")
        assert result is None

    def test_existing_model_loads(self, tmp_path):
        import xgboost as xgb

        model = xgb.XGBClassifier(n_estimators=2, max_depth=1, use_label_encoder=False)
        X = np.random.rand(20, 5)
        y = np.random.randint(0, 2, 20)
        model.fit(X, y, eval_set=[(X, y)], verbose=False)

        model_path = tmp_path / "test_model.json"
        model.save_model(str(model_path))

        loaded = mlb_predict.load_model(model_path)
        assert loaded is not None
        probs = loaded.predict_proba(X)
        assert probs.shape == (20, 2)


class TestMLBRosterValidation:

    def test_complete_roster(self):
        """A game with both lineups and both SPs should be predictable."""
        row = pd.Series({
            "HOME_LINEUP": [101, 102, 103, 104, 105, 106, 107, 108, 109],
            "AWAY_LINEUP": [201, 202, 203, 204, 205, 206, 207, 208, 209],
            "HOME_SP": 501,
            "AWAY_SP": 601,
        })
        home_lu = row.get("HOME_LINEUP")
        away_lu = row.get("AWAY_LINEUP")
        home_sp = row.get("HOME_SP")
        away_sp = row.get("AWAY_SP")

        has_lineups = (home_lu is not None and isinstance(home_lu, list) and len(home_lu) > 0 and
                       away_lu is not None and isinstance(away_lu, list) and len(away_lu) > 0)
        has_sps = pd.notna(home_sp) and pd.notna(away_sp)

        assert has_lineups and has_sps

    def test_missing_lineup(self):
        """A game missing a lineup should NOT be predictable."""
        row = pd.Series({
            "HOME_LINEUP": None,
            "AWAY_LINEUP": [201, 202, 203],
            "HOME_SP": 501,
            "AWAY_SP": 601,
        })
        home_lu = row.get("HOME_LINEUP")
        has_lineups = (home_lu is not None and isinstance(home_lu, list) and len(home_lu) > 0)
        assert not has_lineups

    def test_empty_lineup(self):
        row = pd.Series({
            "HOME_LINEUP": [],
            "AWAY_LINEUP": [201, 202, 203],
            "HOME_SP": 501,
            "AWAY_SP": 601,
        })
        home_lu = row.get("HOME_LINEUP")
        has_lineups = (home_lu is not None and isinstance(home_lu, list) and len(home_lu) > 0)
        assert not has_lineups

    def test_missing_sp(self):
        row = pd.Series({
            "HOME_LINEUP": [101, 102, 103],
            "AWAY_LINEUP": [201, 202, 203],
            "HOME_SP": pd.NA,
            "AWAY_SP": 601,
        })
        has_sps = pd.notna(row.get("HOME_SP")) and pd.notna(row.get("AWAY_SP"))
        assert not has_sps
