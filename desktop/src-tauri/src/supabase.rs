use reqwest::Client;
use chrono::Utc;
use chrono_tz::America::New_York;

use crate::types::Prediction;

const SUPABASE_URL: &str = "https://erwayvqnooppeelqmebd.supabase.co";
const SUPABASE_KEY: &str = "sb_publishable_0rKYvWyHIyulmY9ButyYow_luus8rFi";

/// Fetch today's predictions from the Supabase gamelogs table.
/// Uses hardcoded publishable key and the same date format as the web app.
pub async fn fetch_todays_predictions(
    client: &Client,
) -> Result<Vec<Prediction>, String> {
    // Get today's date in Eastern time, formatted as the GAME_DATE stored in DB
    let now_utc = Utc::now();
    let now_et = now_utc.with_timezone(&New_York);
    // Format date part with chrono, then append the timezone suffix manually.
    // IMPORTANT: %2B must NOT be inside chrono's format() string — chrono treats
    // %2 as an invalid format specifier, causing Display::fmt to return Err,
    // which panics inside .to_string(). We URL-encode the + as %2B for Supabase.
    let date_part = now_et.format("%Y-%m-%d").to_string();
    let today_str = format!("{}T00:00:00%2B00:00", date_part);

    let url = format!(
        "{}/rest/v1/gamelogs?select=GAME_ID,GAME_DATE,HOME_NAME,AWAY_NAME,PREDICTION,PREDICTION_PCT,GAME_STATUS,GAME_OUTCOME&GAME_DATE=eq.{}&PREDICTION=not.is.null",
        SUPABASE_URL, today_str
    );

    let resp = match client
        .get(&url)
        .header("apikey", SUPABASE_KEY)
        .header("Authorization", format!("Bearer {}", SUPABASE_KEY))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return Err(format!("Supabase request failed: {:?}", e)),
    };

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Supabase error {}: {}", status, body));
    }

    let body_text = resp.text().await.unwrap_or_default();
    let predictions: Vec<Prediction> = serde_json::from_str(&body_text)
        .map_err(|e| format!("Failed to parse predictions: {:?}", e))?;

    Ok(predictions)
}
