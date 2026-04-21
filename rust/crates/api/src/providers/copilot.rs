//! GitHub Copilot provider — uses the Copilot API as an Anthropic-compatible backend.
//!
//! Authentication flow:
//! 1. Read the GitHub Copilot OAuth token from `~/.config/github-copilot/apps.json`
//! 2. Exchange it for a short-lived Copilot session token via
//!    `GET https://api.github.com/copilot_internal/v2/token`
//! 3. Use the session token as a Bearer token against
//!    `https://api.githubcopilot.com/v1/messages`

use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::Deserialize;

use crate::error::ApiError;
use crate::providers::anthropic::{AnthropicClient, AuthSource};

const COPILOT_API_BASE_URL: &str = "https://api.githubcopilot.com";
const COPILOT_TOKEN_ENDPOINT: &str = "https://api.github.com/copilot_internal/v2/token";
const COPILOT_APPS_JSON_RELATIVE: &str = ".config/github-copilot/apps.json";
const EDITOR_VERSION: &str = "vscode/1.100.0";
const EDITOR_PLUGIN_VERSION: &str = "copilot-chat/0.26.0";
/// Refresh the session token 60 seconds before it expires.
const TOKEN_REFRESH_MARGIN_SECS: u64 = 60;

/// Response from the Copilot token exchange endpoint.
#[derive(Debug, Deserialize)]
struct CopilotTokenResponse {
    token: String,
    expires_at: u64,
}

/// Cached session token with its expiration timestamp.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct CachedToken {
    token: String,
    expires_at: u64,
}

#[allow(dead_code)]
impl CachedToken {
    fn is_valid(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now + TOKEN_REFRESH_MARGIN_SECS < self.expires_at
    }
}

/// Reads the GitHub Copilot OAuth token from the local config file.
fn read_copilot_oauth_token() -> Result<String, ApiError> {
    // First check env var override
    if let Ok(token) = std::env::var("GITHUB_COPILOT_OAUTH_TOKEN") {
        if !token.is_empty() {
            return Ok(token);
        }
    }

    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map_err(|_| {
            ApiError::Auth("cannot determine home directory for Copilot config".to_string())
        })?;
    let apps_path = std::path::Path::new(&home).join(COPILOT_APPS_JSON_RELATIVE);

    let content = std::fs::read_to_string(&apps_path).map_err(|e| {
        ApiError::Auth(format!(
            "cannot read GitHub Copilot config at {}: {e}. \
             Make sure GitHub Copilot is installed and authenticated in VS Code.",
            apps_path.display()
        ))
    })?;

    // apps.json format: {"github.com:<app_id>": {"user": "...", "oauth_token": "ghu_..."}}
    let parsed: serde_json::Value = serde_json::from_str(&content).map_err(|e| {
        ApiError::Auth(format!("failed to parse Copilot apps.json: {e}"))
    })?;

    let obj = parsed.as_object().ok_or_else(|| {
        ApiError::Auth("Copilot apps.json is not a JSON object".to_string())
    })?;

    for (_key, entry) in obj {
        if let Some(token) = entry.get("oauth_token").and_then(|v| v.as_str()) {
            if !token.is_empty() {
                return Ok(token.to_string());
            }
        }
    }

    Err(ApiError::Auth(
        "no OAuth token found in GitHub Copilot apps.json. \
         Make sure you are signed in to GitHub Copilot in VS Code."
            .to_string(),
    ))
}

/// Exchanges a GitHub Copilot OAuth token for a short-lived session token.
async fn exchange_copilot_token(
    http: &reqwest::Client,
    oauth_token: &str,
) -> Result<CopilotTokenResponse, ApiError> {
    let response = http
        .get(COPILOT_TOKEN_ENDPOINT)
        .header("Authorization", format!("token {oauth_token}"))
        .header("User-Agent", "claw-code/0.1.0")
        .header("editor-version", EDITOR_VERSION)
        .header("editor-plugin-version", EDITOR_PLUGIN_VERSION)
        .timeout(Duration::from_secs(10))
        .send()
        .await
        .map_err(|e| ApiError::Auth(format!("Copilot token exchange failed: {e}")))?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        return Err(ApiError::Auth(format!(
            "Copilot token exchange returned {status}: {body}"
        )));
    }

    response
        .json::<CopilotTokenResponse>()
        .await
        .map_err(|e| ApiError::Auth(format!("failed to parse Copilot token response: {e}")))
}

/// Creates an `AnthropicClient` configured to use GitHub Copilot as the backend.
///
/// The returned client points to `api.githubcopilot.com` and includes the
/// required `editor-version` / `editor-plugin-version` headers.
pub fn create_copilot_client() -> Result<AnthropicClient, ApiError> {
    let oauth_token = read_copilot_oauth_token()?;
    let http = reqwest::Client::new();
    let token_cache: Arc<Mutex<Option<CachedToken>>> = Arc::new(Mutex::new(None));

    // Do the initial token exchange synchronously at startup.
    let session_token = {
        let rt = tokio::runtime::Handle::try_current().ok();
        match rt {
            Some(handle) => {
                // Already inside a tokio runtime — spawn a blocking task.
                std::thread::scope(|s| {
                    s.spawn(|| {
                        let rt = tokio::runtime::Runtime::new().map_err(|e| {
                            ApiError::Auth(format!("failed to create tokio runtime: {e}"))
                        })?;
                        rt.block_on(exchange_copilot_token(&http, &oauth_token))
                    })
                    .join()
                    .expect("thread should not panic")
                })?
            }
            None => {
                let rt = tokio::runtime::Runtime::new().map_err(|e| {
                    ApiError::Auth(format!("failed to create tokio runtime: {e}"))
                })?;
                rt.block_on(exchange_copilot_token(&http, &oauth_token))?
            }
        }
    };

    // Cache the token
    {
        let mut cache = token_cache.lock().unwrap_or_else(|e| e.into_inner());
        *cache = Some(CachedToken {
            token: session_token.token.clone(),
            expires_at: session_token.expires_at,
        });
    }

    let client = AnthropicClient::from_auth(AuthSource::BearerToken(session_token.token))
        .with_base_url(COPILOT_API_BASE_URL)
        .with_no_betas()
        .with_extra_headers(vec![
            ("editor-version".to_string(), EDITOR_VERSION.to_string()),
            (
                "editor-plugin-version".to_string(),
                EDITOR_PLUGIN_VERSION.to_string(),
            ),
            (
                "copilot-integration-id".to_string(),
                "vscode-chat".to_string(),
            ),
        ]);

    Ok(client)
}

/// Check whether GitHub Copilot credentials are available on this machine.
#[must_use]
pub fn has_copilot_credentials() -> bool {
    read_copilot_oauth_token().is_ok()
}

/// The base URL used by the Copilot provider.
#[must_use]
pub fn copilot_base_url() -> &'static str {
    COPILOT_API_BASE_URL
}
