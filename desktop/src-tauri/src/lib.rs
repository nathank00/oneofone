mod kalshi;
mod matcher;
mod scanner;
mod supabase;
mod types;

use reqwest::Client;
use std::sync::Arc;
use tauri::State;
use tokio::sync::{watch, Mutex};
use types::*;

/// Shared application state across all commands.
pub struct AppState {
    client: Client,
    scanner_stop_tx: Option<watch::Sender<bool>>,
}

// ── Commands ─────────────────────────────────────────────────────────

#[tauri::command]
async fn load_settings(app: tauri::AppHandle) -> Result<AppSettings, String> {
    use tauri_plugin_store::StoreExt;
    let store = app.store("settings.json").map_err(|e| format!("{:?}", e))?;
    match store.get("settings") {
        Some(val) => serde_json::from_value(val.clone()).map_err(|e| format!("{:?}", e)),
        None => Ok(AppSettings::default()),
    }
}

#[tauri::command]
async fn save_settings(app: tauri::AppHandle, settings: AppSettings) -> Result<(), String> {
    use tauri_plugin_store::StoreExt;
    let store = app.store("settings.json").map_err(|e| format!("{:?}", e))?;
    let val = serde_json::to_value(&settings).map_err(|e| format!("{:?}", e))?;
    store.set("settings", val);
    store.save().map_err(|e| format!("{:?}", e))?;
    Ok(())
}

#[tauri::command]
async fn pick_pem_file(app: tauri::AppHandle) -> Result<Option<String>, String> {
    use tauri_plugin_dialog::DialogExt;
    let file = app
        .dialog()
        .file()
        .add_filter("Key Files", &["pem", "txt"])
        .add_filter("All Files", &["*"])
        .blocking_pick_file();

    match file {
        Some(fp) => {
            let path = fp.into_path().map_err(|e| format!("{:?}", e))?;
            Ok(Some(path.to_string_lossy().to_string()))
        }
        None => Ok(None),
    }
}

#[tauri::command]
async fn test_kalshi_connection(
    state: State<'_, Arc<Mutex<AppState>>>,
    app: tauri::AppHandle,
) -> Result<String, String> {
    let settings = load_settings(app).await?;
    if settings.kalshi_key_id.is_empty() || settings.kalshi_pem_path.is_empty() {
        return Err("API key or PEM path not configured".into());
    }
    let auth = kalshi::load_auth(&settings.kalshi_key_id, &settings.kalshi_pem_path)?;
    let s = state.lock().await;
    kalshi::test_connection(&s.client, &auth, settings.use_demo_api).await
}

#[tauri::command]
async fn get_portfolio_overview(
    state: State<'_, Arc<Mutex<AppState>>>,
    app: tauri::AppHandle,
) -> Result<PortfolioOverview, String> {
    let settings = match load_settings(app).await {
        Ok(s) => s,
        Err(_) => {
            return Ok(PortfolioOverview {
                connected: false,
                balance: 0.0,
                portfolio_value: 0.0,
                positions_count: 0,
                positions: vec![],
                error: Some("Failed to load settings".into()),
            });
        }
    };

    if settings.kalshi_key_id.is_empty() || settings.kalshi_pem_path.is_empty() {
        return Ok(PortfolioOverview {
            connected: false,
            balance: 0.0,
            portfolio_value: 0.0,
            positions_count: 0,
            positions: vec![],
            error: Some("API key not configured".into()),
        });
    }

    let auth = match kalshi::load_auth(&settings.kalshi_key_id, &settings.kalshi_pem_path) {
        Ok(a) => a,
        Err(e) => {
            return Ok(PortfolioOverview {
                connected: false,
                balance: 0.0,
                portfolio_value: 0.0,
                positions_count: 0,
                positions: vec![],
                error: Some(e),
            });
        }
    };

    let s = state.lock().await;

    let balance_resp =
        match kalshi::fetch_balance(&s.client, &auth, settings.use_demo_api).await {
            Ok(b) => b,
            Err(e) => {
                return Ok(PortfolioOverview {
                    connected: false,
                    balance: 0.0,
                    portfolio_value: 0.0,
                    positions_count: 0,
                    positions: vec![],
                    error: Some(e),
                });
            }
        };

    let positions =
        kalshi::fetch_positions(&s.client, &auth, settings.use_demo_api)
            .await
            .unwrap_or_default();

    // balance_dollars is a string like "51.58"; balance is in cents (5158)
    let balance = balance_resp
        .balance_dollars
        .and_then(|s| s.parse::<f64>().ok())
        .or_else(|| balance_resp.balance.map(|c| c / 100.0))
        .unwrap_or(0.0);

    let portfolio_value = balance_resp
        .portfolio_value_dollars
        .and_then(|s| s.parse::<f64>().ok())
        .or_else(|| balance_resp.portfolio_value.map(|c| c / 100.0))
        .unwrap_or(0.0);

    let position_items: Vec<PositionItem> = positions
        .iter()
        .map(|p| {
            let exposure = p
                .market_exposure_dollars
                .as_deref()
                .and_then(|s| s.parse::<f64>().ok())
                .or_else(|| p.market_exposure.map(|c| c as f64 / 100.0))
                .unwrap_or(0.0);
            let traded = p
                .total_traded_dollars
                .as_deref()
                .and_then(|s| s.parse::<f64>().ok())
                .or_else(|| p.total_traded.map(|c| c as f64 / 100.0))
                .unwrap_or(0.0);
            PositionItem {
                ticker: p.ticker.clone(),
                exposure,
                total_traded: traded,
                resting_orders: p.resting_orders_count.unwrap_or(0),
            }
        })
        .collect();

    Ok(PortfolioOverview {
        connected: true,
        balance,
        portfolio_value,
        positions_count: positions.len() as i64,
        positions: position_items,
        error: None,
    })
}

/// Fetch today's model predictions from Supabase (no Kalshi auth needed).
#[tauri::command]
async fn get_todays_predictions_raw(
    state: State<'_, Arc<Mutex<AppState>>>,
) -> Result<Vec<PredictionDisplay>, String> {
    let s = state.lock().await;
    let predictions = supabase::fetch_todays_predictions(&s.client)
        .await
        .unwrap_or_default();

    Ok(predictions
        .into_iter()
        .filter(|p| p.prediction.is_some() && p.prediction_pct.is_some())
        .map(|p| {
            let pct = p.prediction_pct.unwrap_or(0.5);
            let pred = p.prediction.unwrap_or(1);
            let winner = if pred == 1 {
                p.home_name.clone()
            } else {
                p.away_name.clone()
            };
            // pct is the raw model output (probability of home win).
            // If pred==1 (home win), confidence = pct.
            // If pred==0 (away win), confidence = 1.0 - pct.
            let confidence = if pred == 1 { pct } else { 1.0 - pct };
            PredictionDisplay {
                game_id: p.game_id,
                home_name: p.home_name,
                away_name: p.away_name,
                predicted_winner: winner,
                win_probability: confidence,
                game_status: p.game_status.unwrap_or(1),
            }
        })
        .collect())
}

/// Fetch predictions matched to Kalshi markets (needs Kalshi auth).
/// Returns empty vec on any error — never panics.
#[tauri::command]
async fn get_predictions(
    state: State<'_, Arc<Mutex<AppState>>>,
    app: tauri::AppHandle,
) -> Result<Vec<MatchedGame>, String> {
    let settings = match load_settings(app).await {
        Ok(s) => s,
        Err(_) => return Ok(vec![]),
    };

    if settings.kalshi_key_id.is_empty() || settings.kalshi_pem_path.is_empty() {
        return Ok(vec![]);
    }

    let auth = match kalshi::load_auth(&settings.kalshi_key_id, &settings.kalshi_pem_path) {
        Ok(a) => a,
        Err(_) => return Ok(vec![]),
    };

    let s = state.lock().await;

    // Gracefully handle empty predictions or markets
    let predictions = supabase::fetch_todays_predictions(&s.client)
        .await
        .unwrap_or_default();

    let markets = kalshi::fetch_nba_markets(&s.client, &auth, settings.use_demo_api)
        .await
        .unwrap_or_default();

    Ok(matcher::match_predictions_to_markets(&predictions, &markets))
}

#[tauri::command]
async fn place_bet(
    state: State<'_, Arc<Mutex<AppState>>>,
    app: tauri::AppHandle,
    ticker: String,
    side: String,
    count: i64,
    price_dollars: String,
) -> Result<OrderResult, String> {
    let settings = load_settings(app).await?;
    let auth = kalshi::load_auth(&settings.kalshi_key_id, &settings.kalshi_pem_path)?;
    let s = state.lock().await;

    kalshi::place_order(
        &s.client,
        &auth,
        settings.use_demo_api,
        &ticker,
        &side,
        count,
        &price_dollars,
    )
    .await
}

#[tauri::command]
async fn start_scanner(
    state: State<'_, Arc<Mutex<AppState>>>,
    app: tauri::AppHandle,
) -> Result<(), String> {
    let settings = load_settings(app.clone()).await?;

    let mut s = state.lock().await;

    // Stop existing scanner if running
    if let Some(tx) = s.scanner_stop_tx.take() {
        let _ = tx.send(true);
    }

    let (stop_tx, stop_rx) = watch::channel(false);
    s.scanner_stop_tx = Some(stop_tx);

    // Spawn the scanner as a background task
    tauri::async_runtime::spawn(scanner::run_scanner(app, stop_rx, settings));

    Ok(())
}

#[tauri::command]
async fn stop_scanner(state: State<'_, Arc<Mutex<AppState>>>) -> Result<(), String> {
    let mut s = state.lock().await;
    if let Some(tx) = s.scanner_stop_tx.take() {
        let _ = tx.send(true);
    }
    Ok(())
}

#[tauri::command]
async fn get_scanner_status(state: State<'_, Arc<Mutex<AppState>>>) -> Result<bool, String> {
    let s = state.lock().await;
    Ok(s.scanner_stop_tx.is_some())
}

// ── App entry point ──────────────────────────────────────────────────

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Install a custom panic hook that prints the full backtrace so we can
    // identify which exact line triggers the Display-panic.
    std::panic::set_hook(Box::new(|info| {
        let bt = std::backtrace::Backtrace::force_capture();
        eprintln!("\n=== PANIC ===\n{}\nBacktrace:\n{}", info, bt);
    }));

    let app_state = Arc::new(Mutex::new(AppState {
        client: Client::new(),
        scanner_stop_tx: None,
    }));

    tauri::Builder::default()
        .plugin(tauri_plugin_store::Builder::new().build())
        .plugin(tauri_plugin_dialog::init())
        .manage(app_state)
        .invoke_handler(tauri::generate_handler![
            load_settings,
            save_settings,
            pick_pem_file,
            test_kalshi_connection,
            get_portfolio_overview,
            get_todays_predictions_raw,
            get_predictions,
            place_bet,
            start_scanner,
            stop_scanner,
            get_scanner_status,
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
