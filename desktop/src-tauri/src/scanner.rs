use chrono::Utc;
use reqwest::Client;
use tauri::{AppHandle, Emitter};
use tokio::sync::watch;

use crate::kalshi;
use crate::matcher;
use crate::supabase;
use crate::types::*;

/// Emit a scanner event to the frontend.
fn emit_event(app: &AppHandle, event_type: &str, message: &str, game: Option<&str>, edge: Option<f64>, order_id: Option<&str>) {
    let event = ScannerEvent {
        timestamp: Utc::now().format("%H:%M:%S").to_string(),
        event_type: event_type.to_string(),
        message: message.to_string(),
        game: game.map(|s| s.to_string()),
        edge,
        order_id: order_id.map(|s| s.to_string()),
    };
    let _ = app.emit("scanner-event", &event);
}

/// The main scanner loop. Runs until the watch channel signals stop.
pub async fn run_scanner(
    app: AppHandle,
    mut stop_rx: watch::Receiver<bool>,
    settings: AppSettings,
) {
    let client = Client::new();

    let auth = match kalshi::load_auth(&settings.kalshi_key_id, &settings.kalshi_pem_path) {
        Ok(a) => a,
        Err(e) => {
            emit_event(&app, "error", &format!("Failed to load API key: {}", e), None, None, None);
            return;
        }
    };

    let sizing_label = match settings.sizing_mode {
        SizingMode::Contracts => format!("{} contracts", settings.bet_amount as i64),
        SizingMode::Dollars => format!("${:.0}/bet", settings.bet_amount),
    };

    emit_event(&app, "scan", &format!("Scanner started ({sizing_label}, threshold {:.0}%)", settings.edge_threshold), None, None, None);

    loop {
        // Check if we should stop
        if *stop_rx.borrow() {
            emit_event(&app, "scan", "Scanner stopped", None, None, None);
            return;
        }

        // 1. Fetch predictions (hardcoded Supabase credentials)
        let predictions = match supabase::fetch_todays_predictions(&client).await {
            Ok(p) => p,
            Err(e) => {
                emit_event(&app, "error", &format!("Supabase error: {}", e), None, None, None);
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                continue;
            }
        };

        if predictions.is_empty() {
            emit_event(&app, "scan", "No predictions found for today", None, None, None);
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            continue;
        }

        // 2. Fetch markets
        let markets = match kalshi::fetch_nba_markets(&client, &auth, settings.use_demo_api).await {
            Ok(m) => m,
            Err(e) => {
                emit_event(&app, "error", &format!("Kalshi error: {}", e), None, None, None);
                tokio::time::sleep(std::time::Duration::from_secs(30)).await;
                continue;
            }
        };

        if markets.is_empty() {
            emit_event(&app, "scan", "No open NBA markets found", None, None, None);
            tokio::time::sleep(std::time::Duration::from_secs(60)).await;
            continue;
        }

        // 3. Match and find edges
        let matched = matcher::match_predictions_to_markets(&predictions, &markets);
        let threshold = settings.edge_threshold;

        emit_event(
            &app,
            "scan",
            &format!("Scanned {} games, {} matched to markets", predictions.len(), matched.len()),
            None,
            None,
            None,
        );

        for game in &matched {
            if game.edge >= threshold {
                let game_desc = format!("{} @ {}", game.away_name, game.home_name);
                emit_event(
                    &app,
                    "edge_found",
                    &format!(
                        "Edge {:.1}% on {} (model {:.0}% vs market {:.0}%)",
                        game.edge,
                        game.predicted_winner,
                        game.model_prob * 100.0,
                        game.market_implied_prob * 100.0,
                    ),
                    Some(&game_desc),
                    Some(game.edge),
                    None,
                );

                // Compute price: buy at the current ask
                let price = if game.bet_side == "yes" {
                    game.yes_ask.unwrap_or(0.0)
                } else {
                    game.no_ask.unwrap_or(0.0)
                };

                if price <= 0.0 || price >= 1.0 {
                    emit_event(
                        &app,
                        "error",
                        &format!("Invalid price ${:.2} for {}", price, game.market_ticker),
                        Some(&game_desc),
                        None,
                        None,
                    );
                    continue;
                }

                // Compute contract count based on sizing mode
                let count = settings.compute_contract_count(price);
                let price_str = format!("{:.2}", price);

                // Execute the bet
                match kalshi::place_order(
                    &client,
                    &auth,
                    settings.use_demo_api,
                    &game.market_ticker,
                    &game.bet_side,
                    count,
                    &price_str,
                )
                .await
                {
                    Ok(result) => {
                        let cost_str = format!("{:.2}", price * count as f64);
                        emit_event(
                            &app,
                            "bet_placed",
                            &format!(
                                "Placed {} {} x{} @ ${} (${}) on {}",
                                game.bet_side,
                                result.action,
                                count,
                                price_str,
                                cost_str,
                                game.market_ticker,
                            ),
                            Some(&game_desc),
                            Some(game.edge),
                            Some(&result.order_id),
                        );
                    }
                    Err(e) => {
                        emit_event(
                            &app,
                            "error",
                            &format!("Order failed: {}", e),
                            Some(&game_desc),
                            None,
                            None,
                        );
                    }
                }
            }
        }

        // Wait before next scan cycle (check for stop signal during sleep)
        tokio::select! {
            _ = tokio::time::sleep(std::time::Duration::from_secs(30)) => {},
            _ = stop_rx.changed() => {
                if *stop_rx.borrow() {
                    emit_event(&app, "scan", "Scanner stopped", None, None, None);
                    return;
                }
            }
        }
    }
}
