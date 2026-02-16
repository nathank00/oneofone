# tests/test_nba_train.py
"""
Tests for NBA train.py and predict.py logic.

Tests the pure computational functions using synthetic data:
- Feature definitions: correct count and naming
- add_diff_features: difference computation
- build_feature_matrix: feature extraction and NaN handling
- time_split: chronological splitting with no leakage
- load_model: graceful handling of missing model files
"""

import numpy as np
import pandas as pd

# Import the pre-loaded modules from conftest (avoids name collisions)
from conftest import nba_train, nba_predict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_training_df(n=100):
    """Create a synthetic gamelogs DataFrame suitable for training."""
    rng = np.random.RandomState(42)
    dates = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")

    data = {
        "GAME_ID": range(1, n + 1),
        "GAME_DATE": dates,
        "SEASON_ID": [22023] * n,
        "GAME_STATUS": [3] * n,
        "GAME_OUTCOME": rng.randint(0, 2, n),
    }

    # Add all rolling feature columns with random data
    for col in nba_train.FEATURE_COLS:
        data[col] = rng.uniform(0.1, 1.0, n)

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestFeatureDefinitions:

    def test_feature_count(self):
        """52 raw rolling + 8 diff = 60 total features."""
        assert len(nba_train.FEATURE_COLS) == 52
        assert len(nba_train.DIFF_FEATURES) == 8
        assert len(nba_train.ALL_FEATURES) == 60

    def test_feature_cols_have_home_and_away(self):
        home_cols = [c for c in nba_train.FEATURE_COLS if c.startswith("HOME_")]
        away_cols = [c for c in nba_train.FEATURE_COLS if c.startswith("AWAY_")]
        assert len(home_cols) == 26
        assert len(away_cols) == 26

    def test_diff_features_reference_existing_cols(self):
        """Every diff feature should reference valid HOME_ and AWAY_ columns."""
        for name, (home_col, away_col) in nba_train.DIFF_FEATURES.items():
            assert home_col in nba_train.FEATURE_COLS, f"{home_col} not in FEATURE_COLS"
            assert away_col in nba_train.FEATURE_COLS, f"{away_col} not in FEATURE_COLS"

    def test_predict_features_match_train(self):
        """predict.py feature definitions must match train.py exactly."""
        assert nba_predict.ALL_FEATURES == nba_train.ALL_FEATURES
        assert nba_predict.FEATURE_COLS == nba_train.FEATURE_COLS
        assert nba_predict.DIFF_FEATURES == nba_train.DIFF_FEATURES


class TestAddDiffFeatures:

    def test_diff_columns_added(self):
        df = make_training_df(10)
        result = nba_train.add_diff_features(df)

        for name in nba_train.DIFF_FEATURES:
            assert name in result.columns, f"Missing diff column: {name}"

    def test_diff_values_correct(self):
        df = make_training_df(10)
        result = nba_train.add_diff_features(df)

        for name, (home_col, away_col) in nba_train.DIFF_FEATURES.items():
            expected = df[home_col] - df[away_col]
            actual = result[name]
            assert np.allclose(expected, actual, equal_nan=True), \
                f"Diff values incorrect for {name}"

    def test_missing_columns_produce_nan(self):
        df = pd.DataFrame({"GAME_ID": [1], "HOME_PTS_10": [100.0]})
        result = nba_train.add_diff_features(df)
        # AWAY_PTS_10 doesn't exist, so DIFF_PTS_10 should be NaN
        assert pd.isna(result["DIFF_PTS_10"].iloc[0])


class TestBuildFeatureMatrix:

    def test_returns_x_y_df(self):
        df = make_training_df(50)
        X, y, result_df = nba_train.build_feature_matrix(df)

        assert len(X) == len(y)
        assert len(X.columns) == 60
        assert set(y.unique()).issubset({0, 1})

    def test_drops_rows_with_nan_features(self):
        df = make_training_df(50)
        # Set first 5 rows' HOME_PTS_10 to NaN
        df.loc[:4, "HOME_PTS_10"] = np.nan

        X, y, result_df = nba_train.build_feature_matrix(df)
        assert len(X) == 45

    def test_x_is_float(self):
        df = make_training_df(50)
        X, _, _ = nba_train.build_feature_matrix(df)
        assert X.dtypes.apply(lambda d: np.issubdtype(d, np.floating)).all()

    def test_y_is_integer(self):
        df = make_training_df(50)
        _, y, _ = nba_train.build_feature_matrix(df)
        assert y.dtype in [np.int64, np.int32, int]


class TestTimeSplit:

    def test_split_sizes(self):
        df = make_training_df(100)
        X, y, _ = nba_train.build_feature_matrix(df)

        X_train, X_test, y_train, y_test, df_test = nba_train.time_split(X, y, df)

        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_chronological_order(self):
        """All test dates should be after all train dates."""
        df = make_training_df(100)
        X, y, df2 = nba_train.build_feature_matrix(df)

        X_train, X_test, y_train, y_test, df_test = nba_train.time_split(X, y, df2)

        train_max_date = df2.loc[X_train.index, "GAME_DATE"].max()
        test_min_date = df_test["GAME_DATE"].min()
        assert test_min_date >= train_max_date, "Data leakage: test data precedes training data"

    def test_no_overlap(self):
        """Train and test indices should not overlap."""
        df = make_training_df(100)
        X, y, _ = nba_train.build_feature_matrix(df)

        X_train, X_test, _, _, _ = nba_train.time_split(X, y, df)

        overlap = set(X_train.index) & set(X_test.index)
        assert len(overlap) == 0, f"Train/test overlap: {overlap}"

    def test_custom_split_fraction(self):
        df = make_training_df(100)
        X, y, _ = nba_train.build_feature_matrix(df)

        X_train, X_test, _, _, _ = nba_train.time_split(X, y, df, test_fraction=0.30)

        assert len(X_train) == 70
        assert len(X_test) == 30


class TestLoadModel:

    def test_missing_model_returns_none(self, tmp_path):
        result = nba_predict.load_model(tmp_path / "nonexistent.json")
        assert result is None

    def test_existing_model_loads(self, tmp_path):
        """Create a minimal XGBoost model, save it, then load it."""
        import xgboost as xgb

        model = xgb.XGBClassifier(n_estimators=2, max_depth=1, use_label_encoder=False)
        X = np.random.rand(20, 5)
        y = np.random.randint(0, 2, 20)
        model.fit(X, y, eval_set=[(X, y)], verbose=False)

        model_path = tmp_path / "test_model.json"
        model.save_model(str(model_path))

        loaded = nba_predict.load_model(model_path)
        assert loaded is not None

        # Verify it can predict
        probs = loaded.predict_proba(X)
        assert probs.shape == (20, 2)
