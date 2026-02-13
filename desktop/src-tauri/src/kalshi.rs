use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use reqwest::Client;
use rsa::pkcs1::DecodeRsaPrivateKey;
use rsa::pkcs8::DecodePrivateKey;
use rsa::pss::BlindedSigningKey;
use rsa::sha2::Sha256;
use rsa::signature::RandomizedSigner;
use rsa::signature::SignatureEncoding;
use rsa::RsaPrivateKey;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::types::*;

const PROD_BASE: &str = "https://api.elections.kalshi.com/trade-api/v2";
const DEMO_BASE: &str = "https://demo-api.kalshi.co/trade-api/v2";

pub struct KalshiAuth {
    pub key_id: String,
    pub signing_key: BlindedSigningKey<Sha256>,
}

pub fn base_url(use_demo: bool) -> &'static str {
    if use_demo { DEMO_BASE } else { PROD_BASE }
}

/// Load a PEM private key from file and create an auth context.
/// Supports both PKCS#8 ("BEGIN PRIVATE KEY") and PKCS#1 ("BEGIN RSA PRIVATE KEY") formats.
pub fn load_auth(key_id: &str, pem_path: &str) -> Result<KalshiAuth, String> {
    let pem_str = std::fs::read_to_string(pem_path)
        .map_err(|e| format!("Failed to read PEM file: {:?}", e))?;

    // Try PKCS#8 first, then fall back to PKCS#1
    let private_key = RsaPrivateKey::from_pkcs8_pem(&pem_str)
        .or_else(|_| RsaPrivateKey::from_pkcs1_pem(&pem_str))
        .map_err(|e| format!("Failed to parse PEM key (tried PKCS#8 and PKCS#1): {:?}", e))?;

    let signing_key = BlindedSigningKey::<Sha256>::new(private_key);

    Ok(KalshiAuth {
        key_id: key_id.to_string(),
        signing_key,
    })
}

/// Generate signed headers for a Kalshi API request.
fn sign_headers(auth: &KalshiAuth, method: &str, path: &str) -> HashMap<String, String> {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis()
        .to_string();

    let message = format!("{timestamp}{method}{path}");
    let mut rng = rand::thread_rng();
    let signature = auth.signing_key.sign_with_rng(&mut rng, message.as_bytes());
    let sig_b64 = BASE64.encode(signature.to_bytes());

    let mut headers = HashMap::new();
    headers.insert("KALSHI-ACCESS-KEY".to_string(), auth.key_id.clone());
    headers.insert("KALSHI-ACCESS-TIMESTAMP".to_string(), timestamp);
    headers.insert("KALSHI-ACCESS-SIGNATURE".to_string(), sig_b64);
    headers
}

fn build_request(
    client: &Client,
    method: &str,
    url: &str,
    path: &str,
    auth: Option<&KalshiAuth>,
) -> reqwest::RequestBuilder {
    let mut req = match method {
        "POST" => client.post(url),
        "DELETE" => client.delete(url),
        _ => client.get(url),
    };

    if let Some(auth) = auth {
        let headers = sign_headers(auth, method, path);
        for (k, v) in &headers {
            req = req.header(k.as_str(), v.as_str());
        }
    }

    req
}

// ── Public API functions ─────────────────────────────────────────────

pub async fn fetch_nba_markets(
    client: &Client,
    auth: &KalshiAuth,
    use_demo: bool,
) -> Result<Vec<KalshiMarket>, String> {
    let base = base_url(use_demo);
    let path = "/trade-api/v2/markets";
    let url = format!("{base}/markets?series_ticker=KXNBAGAME&status=open&limit=200");

    let resp = match build_request(client, "GET", &url, path, Some(auth))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return Err(format!("Market request failed: {:?}", e)),
    };

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Kalshi markets API error {}: {}", status, body));
    }

    // Try to parse the response; if it fails, return empty list
    // (e.g. the series doesn't exist on demo API)
    let body_text = resp.text().await.unwrap_or_default();
    match serde_json::from_str::<KalshiMarketsResponse>(&body_text) {
        Ok(data) => Ok(data.markets.into_iter().map(|m| m.into_market()).collect()),
        Err(_) => Ok(vec![]),
    }
}

pub async fn fetch_balance(
    client: &Client,
    auth: &KalshiAuth,
    use_demo: bool,
) -> Result<KalshiBalanceResponse, String> {
    let base = base_url(use_demo);
    let path = "/trade-api/v2/portfolio/balance";
    let url = format!("{base}/portfolio/balance");

    let resp = match build_request(client, "GET", &url, path, Some(auth))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return Err(format!("Balance request failed: {:?}", e)),
    };

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Kalshi balance API error {}: {}", status, body));
    }

    let body_text = resp.text().await.unwrap_or_default();
    serde_json::from_str(&body_text)
        .map_err(|e| format!("Failed to parse balance: {:?}", e))
}

pub async fn fetch_positions(
    client: &Client,
    auth: &KalshiAuth,
    use_demo: bool,
) -> Result<Vec<KalshiPosition>, String> {
    let base = base_url(use_demo);
    let path = "/trade-api/v2/portfolio/positions";
    let url = format!("{base}/portfolio/positions?limit=200");

    let resp = match build_request(client, "GET", &url, path, Some(auth))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return Err(format!("Positions request failed: {:?}", e)),
    };

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Kalshi positions API error {}: {}", status, body));
    }

    let body_text = resp.text().await.unwrap_or_default();
    match serde_json::from_str::<KalshiPositionsResponse>(&body_text) {
        Ok(data) => Ok(data.market_positions),
        Err(_) => Ok(vec![]),
    }
}

pub async fn place_order(
    client: &Client,
    auth: &KalshiAuth,
    use_demo: bool,
    ticker: &str,
    side: &str,
    count: i64,
    price_dollars: &str,
) -> Result<OrderResult, String> {
    let base = base_url(use_demo);
    let path = "/trade-api/v2/portfolio/orders";
    let url = format!("{base}/portfolio/orders");

    let price_field = if side == "yes" {
        "yes_price_dollars"
    } else {
        "no_price_dollars"
    };

    let body = serde_json::json!({
        "ticker": ticker,
        "side": side,
        "action": "buy",
        "count": count,
        price_field: price_dollars,
        "type": "limit",
    });

    let resp = match build_request(client, "POST", &url, path, Some(auth))
        .header("Content-Type", "application/json")
        .body(body.to_string())
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => return Err(format!("Order request failed: {:?}", e)),
    };

    if !resp.status().is_success() {
        let status = resp.status().as_u16();
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("Order failed {}: {}", status, body));
    }

    let body_text = resp.text().await.unwrap_or_default();
    let data: KalshiOrderResponse = serde_json::from_str(&body_text)
        .map_err(|e| format!("Failed to parse order response: {:?}", e))?;

    Ok(OrderResult {
        order_id: data.order.order_id,
        ticker: data.order.ticker,
        status: data.order.status,
        side: data.order.side,
        action: data.order.action,
        fill_count: data.order.fill_count,
        remaining_count: data.order.remaining_count,
    })
}

/// Test connection by fetching balance. Returns "ok" on success.
pub async fn test_connection(
    client: &Client,
    auth: &KalshiAuth,
    use_demo: bool,
) -> Result<String, String> {
    let bal = fetch_balance(client, auth, use_demo).await?;
    // balance_dollars is a string like "51.58"; balance is in cents (5158)
    let balance_str = bal
        .balance_dollars
        .unwrap_or_else(|| {
            let cents = bal.balance.unwrap_or(0.0);
            format!("{:.2}", cents / 100.0)
        });
    Ok(format!("Connected. Balance: ${}", balance_str))
}
