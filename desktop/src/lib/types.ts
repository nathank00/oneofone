export interface MatchedGame {
  gameId: number;
  homeName: string;
  awayName: string;
  predictedWinner: string;
  modelProb: number;
  marketImpliedProb: number;
  edge: number;
  marketTicker: string;
  marketTitle: string;
  yesAsk: number | null;
  noAsk: number | null;
  betSide: string;
}

export interface OrderResult {
  orderId: string;
  ticker: string;
  status: string;
  side: string;
  action: string;
  fillCount: number | null;
  remainingCount: number | null;
}

export interface PositionItem {
  ticker: string;
  exposure: number;
  totalTraded: number;
  restingOrders: number;
}

export interface PortfolioOverview {
  connected: boolean;
  balance: number;
  portfolioValue: number;
  positionsCount: number;
  positions: PositionItem[];
  error: string | null;
}

export interface ScannerEvent {
  timestamp: string;
  eventType: string;
  message: string;
  game: string | null;
  edge: number | null;
  orderId: string | null;
}

export interface PredictionDisplay {
  gameId: number;
  homeName: string;
  awayName: string;
  predictedWinner: string;
  winProbability: number;
  gameStatus: number;
}

export type SizingMode = "contracts" | "dollars";

export interface AppSettings {
  kalshiKeyId: string;
  kalshiPemPath: string;
  edgeThreshold: number;
  sizingMode: SizingMode;
  betAmount: number;
  useDemoApi: boolean;
}

export type Tab = "home" | "manual" | "auto" | "settings";
