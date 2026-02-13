export interface GameLog {
  GAME_ID: number;
  GAME_DATE: string;
  AWAY_NAME: string;
  HOME_NAME: string;
  GAME_STATUS: number;
  GAME_OUTCOME: number | null;
  PREDICTION: number | null;
  PREDICTION_PCT: number | null;
  AWAY_PTS: number | null;
  HOME_PTS: number | null;
}

export interface WLRecord {
  wins: number;
  losses: number;
}
