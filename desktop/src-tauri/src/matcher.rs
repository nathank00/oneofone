use crate::types::*;
use std::collections::HashMap;

/// Build a map from full team name → Kalshi ticker abbreviation.
/// These match the suffixes on Kalshi market tickers, e.g., "-ORL", "-SAC".
fn team_abbr_map() -> HashMap<&'static str, &'static str> {
    let mut m = HashMap::new();
    m.insert("Atlanta Hawks", "ATL");
    m.insert("Boston Celtics", "BOS");
    m.insert("Brooklyn Nets", "BKN");
    m.insert("Charlotte Hornets", "CHA");
    m.insert("Chicago Bulls", "CHI");
    m.insert("Cleveland Cavaliers", "CLE");
    m.insert("Dallas Mavericks", "DAL");
    m.insert("Denver Nuggets", "DEN");
    m.insert("Detroit Pistons", "DET");
    m.insert("Golden State Warriors", "GSW");
    m.insert("Houston Rockets", "HOU");
    m.insert("Indiana Pacers", "IND");
    m.insert("Los Angeles Clippers", "LAC");
    m.insert("Los Angeles Lakers", "LAL");
    m.insert("Memphis Grizzlies", "MEM");
    m.insert("Miami Heat", "MIA");
    m.insert("Milwaukee Bucks", "MIL");
    m.insert("Minnesota Timberwolves", "MIN");
    m.insert("New Orleans Pelicans", "NOP");
    m.insert("New York Knicks", "NYK");
    m.insert("Oklahoma City Thunder", "OKC");
    m.insert("Orlando Magic", "ORL");
    m.insert("Philadelphia 76ers", "PHI");
    m.insert("Phoenix Suns", "PHX");
    m.insert("Portland Trail Blazers", "POR");
    m.insert("Sacramento Kings", "SAC");
    m.insert("San Antonio Spurs", "SAS");
    m.insert("Toronto Raptors", "TOR");
    m.insert("Utah Jazz", "UTA");
    m.insert("Washington Wizards", "WAS");
    m
}

/// Extract the team abbreviation suffix from a Kalshi ticker.
/// e.g., "KXNBAGAME-26FEB19ORLSAC-ORL" → "ORL"
fn ticker_team_suffix(ticker: &str) -> &str {
    ticker.rsplit('-').next().unwrap_or("")
}

/// Match predictions to Kalshi markets by team abbreviation.
///
/// Strategy (YES-only):
/// 1. Look up the predicted winner's abbreviation (e.g., "Orlando Magic" → "ORL")
/// 2. Look up both teams' abbreviations to identify the correct game event
/// 3. Find the market whose ticker ends with the winner's abbreviation
///    AND whose event ticker contains both teams' abbreviations
/// 4. Always bet YES on that market (never bet NO)
pub fn match_predictions_to_markets(
    predictions: &[Prediction],
    markets: &[KalshiMarket],
) -> Vec<MatchedGame> {
    let abbr_map = team_abbr_map();
    let mut matched = Vec::new();

    eprintln!("[matcher] {} predictions, {} markets", predictions.len(), markets.len());

    for pred in predictions {
        let prediction_val = match pred.prediction {
            Some(v) => v,
            None => continue,
        };
        let prediction_pct = match pred.prediction_pct {
            Some(v) => v,
            None => continue,
        };

        let home_abbr = match abbr_map.get(pred.home_name.as_str()) {
            Some(a) => *a,
            None => {
                eprintln!("[matcher] WARN: no abbreviation for {:?}", pred.home_name);
                continue;
            }
        };
        let away_abbr = match abbr_map.get(pred.away_name.as_str()) {
            Some(a) => *a,
            None => {
                eprintln!("[matcher] WARN: no abbreviation for {:?}", pred.away_name);
                continue;
            }
        };

        let predicted_winner = if prediction_val == 1 {
            &pred.home_name
        } else {
            &pred.away_name
        };
        let winner_abbr = if prediction_val == 1 { home_abbr } else { away_abbr };

        // Model probability for the predicted winner
        let model_prob = if prediction_val == 1 {
            prediction_pct
        } else {
            1.0 - prediction_pct
        };

        eprintln!("[matcher] pred: {} ({}) vs {} ({}) | winner={} ({}) | prob={:.3}",
            pred.home_name, home_abbr, pred.away_name, away_abbr,
            predicted_winner, winner_abbr, model_prob);

        // Find the market where:
        // 1. The event ticker contains BOTH team abbreviations (correct game)
        // 2. The market ticker ends with the winner's abbreviation (YES = our pick)
        for market in markets {
            let event = &market.event_ticker;
            let ticker_suffix = ticker_team_suffix(&market.ticker);

            // Check that this event involves both teams
            if !event.contains(home_abbr) || !event.contains(away_abbr) {
                continue;
            }

            // Only take the market where YES = our predicted winner
            if ticker_suffix != winner_abbr {
                continue;
            }

            eprintln!("[matcher]   MATCH! ticker={} (YES={})", market.ticker, winner_abbr);

            let yes_ask = market.yes_ask.unwrap_or(0.0);

            // Skip if no valid price
            if yes_ask <= 0.0 || yes_ask >= 1.0 {
                eprintln!("[matcher]   SKIP (invalid yes_ask={})", yes_ask);
                continue;
            }

            let edge = (model_prob - yes_ask) * 100.0;

            matched.push(MatchedGame {
                game_id: pred.game_id,
                home_name: pred.home_name.clone(),
                away_name: pred.away_name.clone(),
                predicted_winner: predicted_winner.to_string(),
                model_prob,
                market_implied_prob: yes_ask,
                edge,
                market_ticker: market.ticker.clone(),
                market_title: market.title.clone(),
                yes_ask: market.yes_ask,
                no_ask: market.no_ask,
                bet_side: "yes".to_string(),
            });

            break; // Found the correct market for this prediction
        }
    }

    // Sort by edge descending (best opportunities first)
    matched.sort_by(|a, b| b.edge.partial_cmp(&a.edge).unwrap_or(std::cmp::Ordering::Equal));
    matched
}
