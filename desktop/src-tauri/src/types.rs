use serde::{Deserialize, Serialize};

// ── Supabase predictions ─────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Prediction {
    #[serde(rename = "GAME_ID")]
    pub game_id: i64,
    #[serde(rename = "GAME_DATE")]
    pub game_date: String,
    #[serde(rename = "HOME_NAME")]
    pub home_name: String,
    #[serde(rename = "AWAY_NAME")]
    pub away_name: String,
    #[serde(rename = "PREDICTION")]
    pub prediction: Option<i32>,
    #[serde(rename = "PREDICTION_PCT")]
    pub prediction_pct: Option<f64>,
    #[serde(rename = "GAME_STATUS")]
    pub game_status: Option<i32>,
    #[serde(rename = "GAME_OUTCOME")]
    pub game_outcome: Option<i32>,
}

/// Frontend-friendly prediction for display when no Kalshi markets exist.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PredictionDisplay {
    pub game_id: i64,
    pub home_name: String,
    pub away_name: String,
    pub predicted_winner: String,
    pub win_probability: f64,
    pub game_status: i32,
}

// ── Kalshi market data ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalshiMarket {
    pub ticker: String,
    pub title: String,
    pub subtitle: Option<String>,
    pub event_ticker: String,
    pub status: String,
    #[serde(default)]
    pub yes_bid: Option<f64>,
    #[serde(default)]
    pub yes_ask: Option<f64>,
    #[serde(default)]
    pub no_bid: Option<f64>,
    #[serde(default)]
    pub no_ask: Option<f64>,
    #[serde(default)]
    pub last_price: Option<f64>,
    #[serde(default)]
    pub volume: Option<i64>,
    #[serde(default)]
    pub open_interest: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalshiMarketsResponse {
    pub markets: Vec<KalshiMarketRaw>,
    pub cursor: Option<String>,
}

/// Raw market from Kalshi API (prices as strings)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalshiMarketRaw {
    pub ticker: String,
    pub title: String,
    pub subtitle: Option<String>,
    pub event_ticker: String,
    pub status: String,
    pub yes_bid_dollars: Option<String>,
    pub yes_ask_dollars: Option<String>,
    pub no_bid_dollars: Option<String>,
    pub no_ask_dollars: Option<String>,
    pub last_price_dollars: Option<String>,
    pub volume: Option<i64>,
    pub open_interest: Option<i64>,
}

impl KalshiMarketRaw {
    pub fn into_market(self) -> KalshiMarket {
        KalshiMarket {
            ticker: self.ticker,
            title: self.title,
            subtitle: self.subtitle,
            event_ticker: self.event_ticker,
            status: self.status,
            yes_bid: self.yes_bid_dollars.as_deref().and_then(|s| s.parse().ok()),
            yes_ask: self.yes_ask_dollars.as_deref().and_then(|s| s.parse().ok()),
            no_bid: self.no_bid_dollars.as_deref().and_then(|s| s.parse().ok()),
            no_ask: self.no_ask_dollars.as_deref().and_then(|s| s.parse().ok()),
            last_price: self.last_price_dollars.as_deref().and_then(|s| s.parse().ok()),
            volume: self.volume,
            open_interest: self.open_interest,
        }
    }
}

// ── Matched game (prediction + market combined) ──────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MatchedGame {
    pub game_id: i64,
    pub home_name: String,
    pub away_name: String,
    pub predicted_winner: String,
    pub model_prob: f64,
    pub market_implied_prob: f64,
    pub edge: f64,
    pub market_ticker: String,
    pub market_title: String,
    pub yes_ask: Option<f64>,
    pub no_ask: Option<f64>,
    /// "yes" if betting on the market's yes side, "no" otherwise
    pub bet_side: String,
}

// ── Orders ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OrderResult {
    pub order_id: String,
    pub ticker: String,
    pub status: String,
    pub side: String,
    pub action: String,
    pub fill_count: Option<i64>,
    pub remaining_count: Option<i64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalshiOrderResponse {
    pub order: KalshiOrderRaw,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalshiOrderRaw {
    pub order_id: String,
    pub ticker: String,
    pub status: String,
    pub side: String,
    pub action: String,
    #[serde(default)]
    pub fill_count: Option<i64>,
    #[serde(default)]
    pub remaining_count: Option<i64>,
}

// ── Portfolio ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PositionItem {
    pub ticker: String,
    pub exposure: f64,
    pub total_traded: f64,
    pub resting_orders: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PortfolioOverview {
    pub connected: bool,
    pub balance: f64,
    pub portfolio_value: f64,
    pub positions_count: i64,
    pub positions: Vec<PositionItem>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalshiBalanceResponse {
    pub balance: Option<f64>,
    pub balance_dollars: Option<String>,
    pub portfolio_value: Option<f64>,
    pub portfolio_value_dollars: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalshiPositionsResponse {
    pub market_positions: Vec<KalshiPosition>,
    pub cursor: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KalshiPosition {
    pub ticker: String,
    pub market_exposure: Option<i64>,
    pub market_exposure_dollars: Option<String>,
    pub total_traded: Option<i64>,
    pub total_traded_dollars: Option<String>,
    pub resting_orders_count: Option<i64>,
}

// ── Scanner events ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ScannerEvent {
    pub timestamp: String,
    pub event_type: String, // "scan", "edge_found", "bet_placed", "error"
    pub message: String,
    pub game: Option<String>,
    pub edge: Option<f64>,
    pub order_id: Option<String>,
}

// ── Settings ─────────────────────────────────────────────────────────

/// Position sizing method: fixed contract count or dollar amount.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
pub enum SizingMode {
    Contracts,
    Dollars,
}

impl Default for SizingMode {
    fn default() -> Self {
        SizingMode::Contracts
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AppSettings {
    pub kalshi_key_id: String,
    pub kalshi_pem_path: String,
    pub edge_threshold: f64,
    /// "contracts" or "dollars"
    pub sizing_mode: SizingMode,
    /// Number of contracts (when sizing_mode == Contracts)
    /// or dollar amount (when sizing_mode == Dollars)
    pub bet_amount: f64,
    pub use_demo_api: bool,
}

impl Default for AppSettings {
    fn default() -> Self {
        Self {
            kalshi_key_id: String::new(),
            kalshi_pem_path: String::new(),
            edge_threshold: 10.0,
            sizing_mode: SizingMode::Contracts,
            bet_amount: 10.0,
            use_demo_api: true,
        }
    }
}

impl AppSettings {
    /// Compute the number of contracts to buy given the price per contract.
    /// In "contracts" mode, returns bet_amount directly (as whole contracts).
    /// In "dollars" mode, computes floor(bet_amount / price) to stay under budget.
    pub fn compute_contract_count(&self, price_dollars: f64) -> i64 {
        match self.sizing_mode {
            SizingMode::Contracts => (self.bet_amount as i64).max(1),
            SizingMode::Dollars => {
                if price_dollars <= 0.0 {
                    return 1;
                }
                let count = (self.bet_amount / price_dollars).floor() as i64;
                count.max(1)
            }
        }
    }
}
