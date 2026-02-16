# tests/test_shared.py
"""Tests for shared constants modules — data integrity checks."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from shared.nba.nba_constants import (
    TEAM_ABBR_TO_FULL,
    TEAM_NAME_TO_ID,
    TEAM_ID_TO_NAME,
    TEAM_SHORT_TO_FULL,
    TEAM_FULL_TO_ID,
)
from shared.mlb.mlb_constants import TEAM_ID_TO_NAME as MLB_TEAM_ID_TO_NAME


# ---------------------------------------------------------------------------
# NBA constants
# ---------------------------------------------------------------------------
class TestNBAConstants:

    def test_all_30_teams_in_abbr_map(self):
        assert len(TEAM_ABBR_TO_FULL) == 30, f"Expected 30 NBA teams, got {len(TEAM_ABBR_TO_FULL)}"

    def test_all_30_teams_in_id_map(self):
        assert len(TEAM_NAME_TO_ID) == 30

    def test_reverse_map_consistency(self):
        """TEAM_ID_TO_NAME should be the exact inverse of TEAM_NAME_TO_ID."""
        for name, tid in TEAM_NAME_TO_ID.items():
            assert TEAM_ID_TO_NAME[tid] == name, f"Reverse lookup failed for {name} -> {tid}"

    def test_abbr_keys_are_three_chars(self):
        for abbr in TEAM_ABBR_TO_FULL:
            assert len(abbr) == 3, f"Abbreviation '{abbr}' is not 3 characters"
            assert abbr == abbr.upper(), f"Abbreviation '{abbr}' is not uppercase"

    def test_full_to_id_maps_abbr_to_full_values(self):
        """TEAM_FULL_TO_ID should map full names back to abbreviations."""
        for abbr, full_name in TEAM_ABBR_TO_FULL.items():
            assert full_name in TEAM_FULL_TO_ID, f"{full_name} missing from TEAM_FULL_TO_ID"

    def test_short_names_are_unique(self):
        """Short display names should be unique across teams."""
        short_names = list(TEAM_SHORT_TO_FULL.keys())
        assert len(short_names) == len(set(short_names)), "Duplicate short names found"

    def test_known_teams_present(self):
        """Spot-check a few well-known teams."""
        assert "LAL" in TEAM_ABBR_TO_FULL
        assert TEAM_ABBR_TO_FULL["LAL"] == "Los Angeles Lakers"
        assert "BOS" in TEAM_ABBR_TO_FULL
        assert TEAM_ABBR_TO_FULL["BOS"] == "Boston Celtics"
        assert "GSW" in TEAM_ABBR_TO_FULL


# ---------------------------------------------------------------------------
# MLB constants
# ---------------------------------------------------------------------------
class TestMLBConstants:

    def test_all_30_teams(self):
        assert len(MLB_TEAM_ID_TO_NAME) == 30, f"Expected 30 MLB teams, got {len(MLB_TEAM_ID_TO_NAME)}"

    def test_ids_are_integers(self):
        for tid in MLB_TEAM_ID_TO_NAME:
            assert isinstance(tid, int), f"Team ID {tid} should be an integer"

    def test_names_are_strings(self):
        for name in MLB_TEAM_ID_TO_NAME.values():
            assert isinstance(name, str) and len(name) > 0

    def test_known_teams_present(self):
        """Spot-check a few well-known teams."""
        assert 119 in MLB_TEAM_ID_TO_NAME  # Dodgers
        assert MLB_TEAM_ID_TO_NAME[119] == "Los Angeles Dodgers"
        assert 147 in MLB_TEAM_ID_TO_NAME  # Yankees
        assert 111 in MLB_TEAM_ID_TO_NAME  # Red Sox

    def test_no_duplicate_names(self):
        names = list(MLB_TEAM_ID_TO_NAME.values())
        assert len(names) == len(set(names)), "Duplicate team names found"
