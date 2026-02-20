# nba-pipeline/src/playerstats.py
"""
NBA Player Stats Pipeline:
Modes:
- full: backfill all player game logs 2020-21 to present (full upsert/overwrite)
- incremental: refresh current season rows (delta upsert)
- current: player stats from games in last 3 days to now+12h UTC (delta upsert)

Data source:
- LeagueGameLog (player_or_team="P"): all players' per-game stat lines for a
  whole season in ONE API call. ~6 calls total for full backfill instead of
  thousands of per-player calls.
- Supabase games table: used in 'current' mode to find game IDs in date range.

Each row = one player's statline in one game.
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta, timezone

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import os
import logging
import pandas as pd
import numpy as np
from supabase import create_client, Client
from dotenv import load_dotenv
from tqdm import tqdm
import random
from time import sleep

from shared.nba.nba_api_client import fetch_league_game_log as _fetch_league_game_log


def make_playerstats_id(game_id, player_id):
    """Generate a deterministic ID from (GAME_ID, PLAYER_ID)."""
    return f"{game_id}_{player_id}"

load_dotenv()

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("supabase").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing Supabase env vars")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------------------------------------------------------------------------
# Season helpers
# ---------------------------------------------------------------------------
def get_current_season_str():
    """Return current NBA season string like '2024-25'."""
    today = datetime.now()
    year = today.year
    if today.month < 7:
        year -= 1
    return f"{year}-{str(year + 1)[-2:]}"


def get_current_season_year():
    """Return start year of the current NBA season (e.g. 2024 for 2024-25)."""
    today = datetime.now()
    year = today.year
    if today.month < 7:
        year -= 1
    return year


def season_str_to_year(season_str):
    """'2024-25' -> 2024"""
    return int(season_str.split("-")[0])


# ---------------------------------------------------------------------------
# 1. LeagueGameLog — all players' stats for an entire season in ONE call
# ---------------------------------------------------------------------------
def fetch_league_gamelog_for_season(season_str, date_from=None, date_to=None):
    """
    Fetch all player game logs for a season using LeagueGameLog.
    With player_or_team_abbreviation="P", returns one row per player per game —
    the entire season in a single API call.

    Optional date_from / date_to (MM/DD/YYYY strings) to narrow the window.
    Returns a raw DataFrame.
    """
    try:
        sleep(random.uniform(0.8, 1.5))
        df = _fetch_league_game_log(
            season=season_str,
            player_or_team="P",
            date_from=date_from,
            date_to=date_to,
        )[0]
        if df.empty:
            logger.info(f"  0 rows from LeagueGameLog for {season_str}")
            return pd.DataFrame()

        logger.info(f"  {len(df)} player-game rows from LeagueGameLog for {season_str}")
        return df
    except Exception as e:
        logger.error(f"  LeagueGameLog failed for {season_str}: {e}")
        return pd.DataFrame()


def normalize_league_gamelog(df, season_str):
    """
    Normalize raw LeagueGameLog (player mode) DataFrame to match the
    playerstats schema.

    When player_or_team_abbreviation="P", the NBA API returns these extra
    columns beyond the documented schema: PLAYER_ID, PLAYER_NAME.
    The stat columns (MIN, FGM, FGA, etc.) are the same.
    """
    if df.empty:
        return pd.DataFrame()

    out = pd.DataFrame()

    # GAME_ID
    game_id_col = "GAME_ID" if "GAME_ID" in df.columns else "Game_ID"
    out["GAME_ID"] = pd.to_numeric(df[game_id_col], errors="coerce").astype("Int64")

    # PLAYER_ID — returned by NBA API in player mode
    player_id_col = "PLAYER_ID" if "PLAYER_ID" in df.columns else "Player_ID"
    out["PLAYER_ID"] = pd.to_numeric(df[player_id_col], errors="coerce").astype("Int64")

    out["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="coerce")
    out["MATCHUP"] = df["MATCHUP"].astype(str)
    out["WL"] = df["WL"].astype(str)

    # MIN — parse from various formats
    out["MIN"] = df["MIN"].apply(_parse_minutes)

    int_cols = [
        "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA",
        "OREB", "DREB", "REB", "AST", "STL", "BLK",
        "TOV", "PF", "PTS", "PLUS_MINUS",
    ]
    for col in int_cols:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        else:
            out[col] = pd.array([pd.NA] * len(df), dtype="Int64")

    float_cols = ["FG_PCT", "FG3_PCT", "FT_PCT"]
    for col in float_cols:
        if col in df.columns:
            out[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        else:
            out[col] = np.nan

    out["SEASON_ID"] = season_str_to_year(season_str)

    out = out.drop_duplicates(subset=["GAME_ID", "PLAYER_ID"])
    return out


def _parse_minutes(val):
    """Parse minutes value — could be 'MM:SS', integer, float, or NaN."""
    if pd.isna(val):
        return pd.NA
    val_str = str(val).strip()
    if not val_str or val_str.lower() == "nan":
        return pd.NA
    if ":" in val_str:
        parts = val_str.split(":")
        try:
            return int(parts[0])
        except (ValueError, IndexError):
            return pd.NA
    try:
        return int(float(val_str))
    except (ValueError, TypeError):
        return pd.NA


# ---------------------------------------------------------------------------
# 2. Get game IDs for date range — from Supabase games table
# ---------------------------------------------------------------------------
def get_game_ids_in_date_range(date_from, date_to):
    """
    Get GAME_ID list from the games table for a date range.
    Returns list of integer game IDs.
    """
    try:
        response = (
            supabase.table("games")
            .select("GAME_ID")
            .gte("GAME_DATE", date_from.isoformat())
            .lte("GAME_DATE", date_to.isoformat())
            .execute()
        )
        if not response.data:
            return []
        return [int(r["GAME_ID"]) for r in response.data if r.get("GAME_ID") is not None]
    except Exception as e:
        logger.error(f"Failed to get game IDs for date range: {e}")
        return []


# ---------------------------------------------------------------------------
# 3. Supabase DB interaction
# ---------------------------------------------------------------------------
def fetch_db_playerstats(season_ids=None, game_ids=None):
    """Fetch existing playerstats from Supabase."""
    try:
        query = supabase.table("playerstats").select("*")
        if season_ids:
            query = query.in_("SEASON_ID", [int(s) for s in season_ids])
        elif game_ids:
            # Supabase .in_() has a limit; batch if needed
            all_data = []
            batch_size = 200
            for i in range(0, len(game_ids), batch_size):
                batch = game_ids[i:i + batch_size]
                resp = supabase.table("playerstats").select("*").in_("GAME_ID", [int(g) for g in batch]).execute()
                if resp.data:
                    all_data.extend(resp.data)
            if all_data:
                db_df = pd.DataFrame(all_data)
                db_df["GAME_ID"] = pd.to_numeric(db_df["GAME_ID"], errors="coerce").astype("Int64")
                db_df["PLAYER_ID"] = pd.to_numeric(db_df["PLAYER_ID"], errors="coerce").astype("Int64")
                return db_df
            return pd.DataFrame()

        response = query.execute()
        if response.data:
            db_df = pd.DataFrame(response.data)
            db_df["GAME_ID"] = pd.to_numeric(db_df["GAME_ID"], errors="coerce").astype("Int64")
            db_df["PLAYER_ID"] = pd.to_numeric(db_df["PLAYER_ID"], errors="coerce").astype("Int64")
            return db_df
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to fetch DB playerstats: {e}")
        return pd.DataFrame()


def find_deltas(new_df, db_df):
    """
    Find rows in new_df that are new or changed compared to db_df.
    Uses (GAME_ID, PLAYER_ID) as the composite key.
    """
    if db_df.empty:
        return new_df

    for df in [new_df, db_df]:
        df["GAME_ID"] = pd.to_numeric(df["GAME_ID"], errors="coerce").astype("Int64")
        df["PLAYER_ID"] = pd.to_numeric(df["PLAYER_ID"], errors="coerce").astype("Int64")

    new_df = new_df.copy()
    db_df = db_df.copy()
    new_df["_key"] = new_df["GAME_ID"].astype(str) + "_" + new_df["PLAYER_ID"].astype(str)
    db_df["_key"] = db_df["GAME_ID"].astype(str) + "_" + db_df["PLAYER_ID"].astype(str)

    merged = new_df.merge(
        db_df[["_key"] + [c for c in db_df.columns if c != "_key"]],
        on="_key", how="left", suffixes=("_new", "_db"),
    )

    # New rows: no match in DB (check a DB-only column)
    delta_mask = merged["GAME_ID_db"].isna()

    # Changed rows: compare stat columns
    compare_cols = [
        "MATCHUP", "WL", "MIN", "FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT",
        "FTM", "FTA", "FT_PCT", "OREB", "DREB", "REB", "AST", "STL", "BLK",
        "TOV", "PF", "PTS", "PLUS_MINUS",
    ]
    for col in compare_cols:
        n_col = f"{col}_new"
        d_col = f"{col}_db"
        if n_col in merged.columns and d_col in merged.columns:
            delta_mask |= (
                merged[n_col].ne(merged[d_col])
                & (merged[n_col].notna() | merged[d_col].notna())
            )

    # Extract the new-side columns only
    new_cols = [c for c in merged.columns if c.endswith("_new")]
    deltas = merged.loc[delta_mask, new_cols + ["_key"]].copy()
    deltas.rename(columns={c: c.replace("_new", "") for c in new_cols}, inplace=True)
    deltas = deltas.drop(columns=["_key"])

    return deltas


def _build_payload(r):
    """Convert a single record dict into a Supabase-ready payload."""
    game_id = int(r["GAME_ID"]) if pd.notna(r.get("GAME_ID")) else None
    player_id = int(r["PLAYER_ID"]) if pd.notna(r.get("PLAYER_ID")) else None
    return {
        "id": make_playerstats_id(game_id, player_id) if game_id and player_id else None,
        "GAME_ID": game_id,
        "PLAYER_ID": player_id,
        "GAME_DATE": pd.Timestamp(r["GAME_DATE"]).isoformat() if pd.notna(r.get("GAME_DATE")) else None,
        "MATCHUP": r.get("MATCHUP"),
        "WL": r.get("WL"),
        "MIN": int(r["MIN"]) if pd.notna(r.get("MIN")) else None,
        "FGM": int(r["FGM"]) if pd.notna(r.get("FGM")) else None,
        "FGA": int(r["FGA"]) if pd.notna(r.get("FGA")) else None,
        "FG_PCT": float(r["FG_PCT"]) if pd.notna(r.get("FG_PCT")) else None,
        "FG3M": int(r["FG3M"]) if pd.notna(r.get("FG3M")) else None,
        "FG3A": int(r["FG3A"]) if pd.notna(r.get("FG3A")) else None,
        "FG3_PCT": float(r["FG3_PCT"]) if pd.notna(r.get("FG3_PCT")) else None,
        "FTM": int(r["FTM"]) if pd.notna(r.get("FTM")) else None,
        "FTA": int(r["FTA"]) if pd.notna(r.get("FTA")) else None,
        "FT_PCT": float(r["FT_PCT"]) if pd.notna(r.get("FT_PCT")) else None,
        "OREB": int(r["OREB"]) if pd.notna(r.get("OREB")) else None,
        "DREB": int(r["DREB"]) if pd.notna(r.get("DREB")) else None,
        "REB": int(r["REB"]) if pd.notna(r.get("REB")) else None,
        "AST": int(r["AST"]) if pd.notna(r.get("AST")) else None,
        "STL": int(r["STL"]) if pd.notna(r.get("STL")) else None,
        "BLK": int(r["BLK"]) if pd.notna(r.get("BLK")) else None,
        "TOV": int(r["TOV"]) if pd.notna(r.get("TOV")) else None,
        "PF": int(r["PF"]) if pd.notna(r.get("PF")) else None,
        "PTS": int(r["PTS"]) if pd.notna(r.get("PTS")) else None,
        "PLUS_MINUS": int(r["PLUS_MINUS"]) if pd.notna(r.get("PLUS_MINUS")) else None,
        "SEASON_ID": int(r["SEASON_ID"]) if pd.notna(r.get("SEASON_ID")) else None,
    }


UPSERT_BATCH_SIZE = 500


def upsert_playerstats(df):
    """Upsert playerstats DataFrame to Supabase 'playerstats' table in batches."""
    if df.empty:
        logger.info("Upserting 0 playerstats rows")
        return

    int_cols = [
        "GAME_ID", "PLAYER_ID", "MIN", "FGM", "FGA", "FG3M", "FG3A",
        "FTM", "FTA", "OREB", "DREB", "REB", "AST", "STL", "BLK",
        "TOV", "PF", "PTS", "PLUS_MINUS", "SEASON_ID",
    ]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    float_cols = ["FG_PCT", "FG3_PCT", "FT_PCT"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    records = df.to_dict(orient="records")
    payloads = [_build_payload(r) for r in records]

    # Sanitize NaN → None for JSON serialization (in-progress games have null stats)
    import math
    for p in payloads:
        for k, v in p.items():
            if isinstance(v, float) and math.isnan(v):
                p[k] = None

    success = 0
    total_batches = (len(payloads) + UPSERT_BATCH_SIZE - 1) // UPSERT_BATCH_SIZE

    with tqdm(total=len(payloads), desc="Upserting playerstats") as pbar:
        for i in range(0, len(payloads), UPSERT_BATCH_SIZE):
            batch = payloads[i : i + UPSERT_BATCH_SIZE]
            try:
                supabase.table("playerstats").upsert(
                    batch, on_conflict="id"
                ).execute()
                success += len(batch)
            except Exception as e:
                logger.warning(f"Batch upsert failed (batch {i // UPSERT_BATCH_SIZE + 1}/{total_batches}, {len(batch)} rows): {e}")
                logger.info("  Falling back to row-by-row for this batch...")
                for payload in batch:
                    try:
                        supabase.table("playerstats").upsert(
                            payload, on_conflict="id"
                        ).execute()
                        success += 1
                    except Exception as row_err:
                        logger.error(f"  Row upsert failed GAME_ID={payload.get('GAME_ID')} PLAYER_ID={payload.get('PLAYER_ID')}: {row_err}")
            pbar.update(len(batch))

    logger.info(f"Upserted {success}/{len(payloads)} playerstats rows")


# ---------------------------------------------------------------------------
# 4. Mode implementations
# ---------------------------------------------------------------------------
def run_full_mode():
    """Full backfill: all seasons 2020-21 to present.
    Deletes all existing rows then inserts fresh data.
    Uses LeagueGameLog — ~1 API call per season instead of thousands."""
    current_year = get_current_season_year()
    seasons = [f"{y}-{str(y + 1)[-2:]}" for y in range(2020, current_year + 1)]
    logger.info(f"Full mode: {len(seasons)} seasons to process: {seasons}")

    all_dfs = []

    # 1. Fetch all player gamelogs per season (~1 API call each)
    for season_str in seasons:
        logger.info(f"Step: Fetching LeagueGameLog for {season_str}...")
        raw = fetch_league_gamelog_for_season(season_str)
        if not raw.empty:
            norm = normalize_league_gamelog(raw, season_str)
            if not norm.empty:
                all_dfs.append(norm)
                logger.info(f"  {len(norm)} normalized rows for {season_str}")

    if not all_dfs:
        logger.info("No player stats to process")
        return

    new_df = pd.concat(all_dfs, ignore_index=True)
    new_df = new_df.drop_duplicates(subset=["GAME_ID", "PLAYER_ID"])
    logger.info(f"Total stat rows across all seasons: {len(new_df)}")

    # 2. Delete all existing rows — full overwrite
    logger.info("Deleting all existing playerstats rows...")
    try:
        supabase.table("playerstats").delete().neq("id", "00000000-0000-0000-0000-000000000000").execute()
        logger.info("  Existing rows deleted")
    except Exception as e:
        logger.error(f"  Failed to delete existing rows: {e}")
        return

    # 3. Insert fresh data (no conflict possible on empty table)
    logger.info(f"Inserting {len(new_df)} playerstats rows...")
    upsert_playerstats(new_df)


def run_incremental_mode():
    """Refresh current season. Delta upsert only changed rows.
    Uses LeagueGameLog — 1 API call for the whole season."""
    current_season = get_current_season_str()
    current_year = get_current_season_year()
    logger.info(f"Incremental mode: season {current_season}")

    # 1. Fetch all player gamelogs for current season (1 API call)
    logger.info("Step 1: Fetching LeagueGameLog for current season...")
    raw = fetch_league_gamelog_for_season(current_season)

    if raw.empty:
        logger.info("No player stats to process")
        return

    new_df = normalize_league_gamelog(raw, current_season)
    logger.info(f"  {len(new_df)} normalized rows")

    if new_df.empty:
        logger.info("No player stats to process")
        return

    # 2. Delta check against DB
    logger.info("Step 2: Finding deltas against DB...")
    db_df = fetch_db_playerstats(season_ids=[current_year])
    deltas = find_deltas(new_df, db_df)
    logger.info(f"  {len(deltas)} deltas found")

    # 3. Upsert deltas only
    if not deltas.empty:
        logger.info(f"Step 3: Upserting {len(deltas)} changed playerstats rows...")
        upsert_playerstats(deltas)
    else:
        logger.info("No changes to upsert")


def run_current_mode():
    """Player stats from games in last 3 days to now+12h UTC. Delta upsert.
    Uses LeagueGameLog with date range — 1 API call."""
    now = datetime.now(timezone.utc)
    current_season = get_current_season_str()
    current_year = get_current_season_year()

    date_from = (now - timedelta(days=3)).date()
    date_to = (now + timedelta(hours=12)).date()
    logger.info(f"Current mode: {date_from} to {date_to}")

    # 1. Fetch player gamelogs for date range (1 API call)
    logger.info("Step 1: Fetching LeagueGameLog for date range...")
    raw = fetch_league_gamelog_for_season(
        current_season,
        date_from=date_from.strftime("%m/%d/%Y"),
        date_to=date_to.strftime("%m/%d/%Y"),
    )

    if raw.empty:
        logger.info("No player stats for date range")
        return

    new_df = normalize_league_gamelog(raw, current_season)
    logger.info(f"  {len(new_df)} normalized rows for date range")

    if new_df.empty:
        logger.info("No player stats for date range")
        return

    # 2. Delta check against DB
    logger.info("Step 2: Finding deltas against DB...")
    game_ids = new_df["GAME_ID"].dropna().unique().tolist()
    db_df = fetch_db_playerstats(game_ids=game_ids)
    deltas = find_deltas(new_df, db_df)
    logger.info(f"  {len(deltas)} deltas found")

    # 3. Upsert deltas only
    if not deltas.empty:
        logger.info(f"Step 3: Upserting {len(deltas)} changed playerstats rows...")
        upsert_playerstats(deltas)
    else:
        logger.info("No changes to upsert")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "current"

    if mode == "full":
        print("\n=== FULL MODE: backfill 2020-21 to present ===")
        run_full_mode()

    elif mode == "incremental":
        print(f"\n=== INCREMENTAL MODE: current season {get_current_season_str()} (delta) ===")
        run_incremental_mode()

    elif mode == "current":
        print("\n=== CURRENT MODE: last 3 days (delta) ===")
        run_current_mode()

    else:
        print(f"Unknown mode: {mode}. Use 'full', 'incremental', or 'current'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
