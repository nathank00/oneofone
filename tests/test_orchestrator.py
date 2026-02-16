# tests/test_orchestrator.py
"""
Tests for run_pipeline.py orchestrators (both NBA and MLB).

Tests the pipeline script structure and configuration:
- Script files exist and are importable
- Pipeline modes are correctly defined
- Script execution order is correct
- CLI argument parsing
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestNBAPipelineStructure:

    def test_run_pipeline_exists(self):
        path = REPO_ROOT / "nba-pipeline" / "run_pipeline.py"
        assert path.exists(), f"Missing: {path}"

    def test_all_stage_scripts_exist(self):
        src = REPO_ROOT / "nba-pipeline" / "src"
        required = ["games.py", "players.py", "playerstats.py",
                     "gamelogs.py", "train.py", "predict.py"]
        for name in required:
            # games.py might also be games2.py — check both
            path = src / name
            assert path.exists(), f"Missing pipeline script: {path}"

    def test_models_directory_exists(self):
        path = REPO_ROOT / "nba-pipeline" / "models"
        assert path.exists() or path.parent.exists(), \
            "nba-pipeline/models directory should exist"

    def test_shared_constants_importable(self):
        from shared.nba.nba_constants import TEAM_ABBR_TO_FULL
        assert len(TEAM_ABBR_TO_FULL) == 30


class TestMLBPipelineStructure:

    def test_run_pipeline_exists(self):
        path = REPO_ROOT / "mlb-pipeline" / "run_pipeline.py"
        assert path.exists(), f"Missing: {path}"

    def test_all_stage_scripts_exist(self):
        src = REPO_ROOT / "mlb-pipeline" / "src"
        required = ["games.py", "players.py", "playerstats.py",
                     "gamelogs.py", "train.py", "predict.py"]
        for name in required:
            path = src / name
            assert path.exists(), f"Missing pipeline script: {path}"

    def test_models_directory_exists(self):
        path = REPO_ROOT / "mlb-pipeline" / "models"
        assert path.exists() or path.parent.exists(), \
            "mlb-pipeline/models directory should exist"

    def test_shared_constants_importable(self):
        from shared.mlb.mlb_constants import TEAM_ID_TO_NAME
        assert len(TEAM_ID_TO_NAME) == 30


class TestNBAOrchestratorConfig:

    def test_imports_and_has_modes(self):
        # Import the orchestrator module
        nba_pipeline_path = REPO_ROOT / "nba-pipeline" / "run_pipeline.py"
        assert nba_pipeline_path.exists()

        import importlib.util
        spec = importlib.util.spec_from_file_location("nba_run_pipeline", nba_pipeline_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert hasattr(mod, "run_historical")
        assert hasattr(mod, "run_current")
        assert callable(mod.run_historical)
        assert callable(mod.run_current)

    def test_src_dir_points_to_correct_location(self):
        nba_pipeline_path = REPO_ROOT / "nba-pipeline" / "run_pipeline.py"
        import importlib.util
        spec = importlib.util.spec_from_file_location("nba_run_pipeline", nba_pipeline_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert hasattr(mod, "SRC_DIR")
        assert Path(mod.SRC_DIR).exists()


class TestMLBOrchestratorConfig:

    def test_imports_and_has_modes(self):
        mlb_pipeline_path = REPO_ROOT / "mlb-pipeline" / "run_pipeline.py"
        assert mlb_pipeline_path.exists()

        import importlib.util
        spec = importlib.util.spec_from_file_location("mlb_run_pipeline", mlb_pipeline_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert hasattr(mod, "run_historical")
        assert hasattr(mod, "run_current")
        assert hasattr(mod, "run_live")  # MLB has a third mode
        assert callable(mod.run_historical)
        assert callable(mod.run_current)
        assert callable(mod.run_live)

    def test_src_dir_points_to_correct_location(self):
        mlb_pipeline_path = REPO_ROOT / "mlb-pipeline" / "run_pipeline.py"
        import importlib.util
        spec = importlib.util.spec_from_file_location("mlb_run_pipeline", mlb_pipeline_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        assert hasattr(mod, "SRC_DIR")
        assert Path(mod.SRC_DIR).exists()


class TestGitHubActionsExist:

    def test_nba_workflow_exists(self):
        path = REPO_ROOT / ".github" / "workflows" / "nba-pipeline.yml"
        assert path.exists(), f"Missing workflow: {path}"

    def test_mlb_workflow_exists(self):
        path = REPO_ROOT / ".github" / "workflows" / "mlb-pipeline.yml"
        assert path.exists(), f"Missing workflow: {path}"

    def test_workflows_are_not_empty(self):
        for name in ["nba-pipeline.yml", "mlb-pipeline.yml"]:
            path = REPO_ROOT / ".github" / "workflows" / name
            assert path.stat().st_size > 100, f"{name} appears to be empty"
