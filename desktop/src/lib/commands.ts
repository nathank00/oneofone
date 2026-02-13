import { invoke } from "@tauri-apps/api/core";
import type {
  AppSettings,
  MatchedGame,
  OrderResult,
  PredictionDisplay,
  PortfolioOverview,
} from "./types";

export async function loadSettings(): Promise<AppSettings> {
  return invoke<AppSettings>("load_settings");
}

export async function saveSettings(settings: AppSettings): Promise<void> {
  return invoke("save_settings", { settings });
}

export async function pickPemFile(): Promise<string | null> {
  return invoke<string | null>("pick_pem_file");
}

export async function testKalshiConnection(): Promise<string> {
  return invoke<string>("test_kalshi_connection");
}

export async function getPortfolioOverview(): Promise<PortfolioOverview> {
  return invoke<PortfolioOverview>("get_portfolio_overview");
}

export async function getTodaysPredictionsRaw(): Promise<PredictionDisplay[]> {
  return invoke<PredictionDisplay[]>("get_todays_predictions_raw");
}

export async function getPredictions(): Promise<MatchedGame[]> {
  return invoke<MatchedGame[]>("get_predictions");
}

export async function placeBet(
  ticker: string,
  side: string,
  count: number,
  priceDollars: string,
): Promise<OrderResult> {
  return invoke<OrderResult>("place_bet", {
    ticker,
    side,
    count,
    priceDollars,
  });
}

export async function startScanner(): Promise<void> {
  return invoke("start_scanner");
}

export async function stopScanner(): Promise<void> {
  return invoke("stop_scanner");
}

export async function getScannerStatus(): Promise<boolean> {
  return invoke<boolean>("get_scanner_status");
}
