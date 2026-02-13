use crate::types::*;

/// Team name fragments used to match Kalshi market titles to our predictions.
/// Kalshi titles are like "Will the Lakers win?" or "Lakers vs Celtics"
/// Our predictions use full names like "Los Angeles Lakers".
fn extract_team_keyword(full_name: &str) -> &str {
    // Handle special cases
    if full_name.contains("Trail Blazers") {
        return "Blazers";
    }
    if full_name.contains("76ers") {
        return "76ers";
    }
    // Default: use last word (e.g., "Los Angeles Lakers" -> "Lakers")
    full_name.split_whitespace().last().unwrap_or(full_name)
}

/// Match predictions to Kalshi markets by team name.
/// Returns a MatchedGame for each prediction that has a corresponding open market.
pub fn match_predictions_to_markets(
    predictions: &[Prediction],
    markets: &[KalshiMarket],
) -> Vec<MatchedGame> {
    let mut matched = Vec::new();

    for pred in predictions {
        let prediction_val = match pred.prediction {
            Some(v) => v,
            None => continue,
        };
        let prediction_pct = match pred.prediction_pct {
            Some(v) => v,
            None => continue,
        };

        let home_keyword = extract_team_keyword(&pred.home_name);
        let away_keyword = extract_team_keyword(&pred.away_name);

        let predicted_winner = if prediction_val == 1 {
            &pred.home_name
        } else {
            &pred.away_name
        };
        let predicted_winner_keyword = extract_team_keyword(predicted_winner);

        // Model probability for the predicted winner
        let model_prob = if prediction_val == 1 {
            prediction_pct
        } else {
            1.0 - prediction_pct
        };

        // Find matching market — look for market titles containing team names
        for market in markets {
            let title_lower = market.title.to_lowercase();
            let subtitle_lower = market
                .subtitle
                .as_deref()
                .unwrap_or("")
                .to_lowercase();
            let combined = format!("{title_lower} {subtitle_lower}");

            let has_home = combined.contains(&home_keyword.to_lowercase());
            let has_away = combined.contains(&away_keyword.to_lowercase());

            if !has_home && !has_away {
                continue;
            }

            // Determine if this market is for the team our model predicts to win
            let market_is_for_predicted_winner =
                combined.contains(&predicted_winner_keyword.to_lowercase());

            // Market implied probability (yes_ask = cost to buy "yes" = implied prob)
            let yes_ask = market.yes_ask.unwrap_or(0.0);
            let no_ask = market.no_ask.unwrap_or(0.0);

            let (market_implied_prob, bet_side) = if market_is_for_predicted_winner {
                // Market is for our predicted winner — buy YES
                (yes_ask, "yes".to_string())
            } else {
                // Market is for the other team — buy NO (betting against them)
                (no_ask, "no".to_string())
            };

            // Skip if no valid price
            if market_implied_prob <= 0.0 || market_implied_prob >= 1.0 {
                continue;
            }

            let edge = (model_prob - market_implied_prob) * 100.0;

            matched.push(MatchedGame {
                game_id: pred.game_id,
                home_name: pred.home_name.clone(),
                away_name: pred.away_name.clone(),
                predicted_winner: predicted_winner.to_string(),
                model_prob,
                market_implied_prob,
                edge,
                market_ticker: market.ticker.clone(),
                market_title: market.title.clone(),
                yes_ask: market.yes_ask,
                no_ask: market.no_ask,
                bet_side,
            });

            break; // Only match the first market per prediction
        }
    }

    // Sort by edge descending (best opportunities first)
    matched.sort_by(|a, b| b.edge.partial_cmp(&a.edge).unwrap_or(std::cmp::Ordering::Equal));
    matched
}
