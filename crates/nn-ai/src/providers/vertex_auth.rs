//! OAuth access tokens for Vertex AI from a service-account file.
//!
//! Implements the JWT-bearer grant: sign an RS256 assertion with the
//! service account's private key, exchange it at the token endpoint and cache
//! the access token until shortly before it expires.
//!
//! Every problem with the credentials file itself (missing, unreadable, not
//! JSON, wrong shape, bad key) is reported with one generic message. Echoing
//! the underlying cause would let anyone who can test a Vertex credential
//! probe the server's filesystem.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use jsonwebtoken::{Algorithm, EncodingKey, Header};
use nn_llm::{AiError, AiResult};
use serde::Serialize;
use serde_json::Value;

use crate::http::Http;

pub(crate) const CREDENTIALS_FILE_ERROR: &str = "Invalid or inaccessible credentials file";
const DEFAULT_TOKEN_URI: &str = "https://oauth2.googleapis.com/token";
const SCOPE: &str = "https://www.googleapis.com/auth/cloud-platform";
const ASSERTION_LIFETIME_SECS: u64 = 3600;
/// Tokens are refreshed this long before they expire.
const EXPIRY_MARGIN: Duration = Duration::from_secs(60);

fn credentials_file_error() -> AiError {
    AiError::Configuration(CREDENTIALS_FILE_ERROR.to_string())
}

struct CachedToken {
    token: String,
    expires_at: Instant,
}

/// Process-wide cache keyed by credentials path, so short-lived model
/// objects do not each pay for a token exchange.
fn cache() -> &'static Mutex<HashMap<String, CachedToken>> {
    static CACHE: OnceLock<Mutex<HashMap<String, CachedToken>>> = OnceLock::new();
    CACHE.get_or_init(Default::default)
}

#[derive(Serialize)]
struct Claims<'a> {
    iss: &'a str,
    scope: &'a str,
    aud: &'a str,
    iat: u64,
    exp: u64,
}

#[derive(Clone)]
pub(crate) struct ServiceAccount {
    path: String,
    http: Http,
}

impl ServiceAccount {
    pub fn new(path: &str, http: Http) -> Self {
        Self { path: path.to_string(), http }
    }

    fn cached(&self) -> Option<String> {
        let cache = cache().lock().unwrap_or_else(|e| e.into_inner());
        cache
            .get(&self.path)
            .filter(|entry| entry.expires_at > Instant::now() + EXPIRY_MARGIN)
            .map(|entry| entry.token.clone())
    }

    /// Reads the file and signs the assertion. Returns `(token_uri, jwt)`.
    async fn assertion(&self) -> AiResult<(String, String)> {
        let raw = tokio::fs::read(&self.path).await.map_err(|_| credentials_file_error())?;
        let file: Value = serde_json::from_slice(&raw).map_err(|_| credentials_file_error())?;
        let field = |name: &str| file.get(name).and_then(Value::as_str).filter(|s| !s.trim().is_empty());
        let client_email = field("client_email").ok_or_else(credentials_file_error)?;
        let private_key = field("private_key").ok_or_else(credentials_file_error)?;
        let token_uri = field("token_uri").unwrap_or(DEFAULT_TOKEN_URI).to_string();

        let key = EncodingKey::from_rsa_pem(private_key.as_bytes()).map_err(|_| credentials_file_error())?;
        let mut header = Header::new(Algorithm::RS256);
        header.kid = field("private_key_id").map(str::to_string);
        let now = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or_default();
        let claims = Claims {
            iss: client_email,
            scope: SCOPE,
            aud: &token_uri,
            iat: now,
            exp: now + ASSERTION_LIFETIME_SECS,
        };
        let jwt = jsonwebtoken::encode(&header, &claims, &key).map_err(|_| credentials_file_error())?;
        Ok((token_uri, jwt))
    }

    /// A valid access token, from the cache when possible.
    pub async fn token(&self) -> AiResult<String> {
        if let Some(token) = self.cached() {
            return Ok(token);
        }
        let (token_uri, jwt) = self.assertion().await?;
        let form = [("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer"), ("assertion", jwt.as_str())];
        let response = match self.http.json(self.http.client.post(&token_uri).form(&form)).await {
            Ok(json) => json,
            Err(err @ (AiError::Network(_) | AiError::Timeout(_))) => return Err(err),
            Err(err) => {
                return Err(AiError::Authentication(format!(
                    "{}: the service account could not obtain an access token ({err})",
                    self.http.provider
                )))
            }
        };
        let token = response
            .get("access_token")
            .and_then(Value::as_str)
            .ok_or_else(|| self.http.malformed("no access_token in the token response"))?
            .to_string();
        let lifetime = response.get("expires_in").and_then(Value::as_u64).unwrap_or(ASSERTION_LIFETIME_SECS);
        let entry = CachedToken { token: token.clone(), expires_at: Instant::now() + Duration::from_secs(lifetime) };
        cache().lock().unwrap_or_else(|e| e.into_inner()).insert(self.path.clone(), entry);
        Ok(token)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::config::ProviderConfig;
    use crate::http::shared_client;
    use base64::Engine;
    use jsonwebtoken::{DecodingKey, Validation};
    use serde::Deserialize;
    use std::io::Write;
    use wiremock::matchers::{body_string_contains, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    // Throwaway key pair generated for these tests only; it guards nothing.
    const TEST_PRIVATE_KEY: &str = include_str!("../../assets/test_only_rsa_private.pem");
    const TEST_PUBLIC_KEY: &str = include_str!("../../assets/test_only_rsa_public.pem");

    /// Writes a service-account file whose token endpoint is `token_uri`.
    pub fn service_account_file(token_uri: &str) -> tempfile::NamedTempFile {
        let mut file = tempfile::NamedTempFile::new().unwrap();
        let json = serde_json::json!({
            "type": "service_account",
            "project_id": "demo",
            "private_key_id": "kid-1",
            "private_key": TEST_PRIVATE_KEY,
            "client_email": "robot@demo.iam.gserviceaccount.com",
            "token_uri": token_uri,
        });
        file.write_all(json.to_string().as_bytes()).unwrap();
        file
    }

    /// Mounts a token endpoint at `/token` that hands out `token`.
    pub async fn mount_token_endpoint(server: &MockServer, token: &str) {
        Mock::given(method("POST"))
            .and(path("/token"))
            .and(body_string_contains("grant_type=urn%3Aietf%3Aparams%3Aoauth%3Agrant-type%3Ajwt-bearer"))
            .respond_with(ResponseTemplate::new(200).set_body_json(
                serde_json::json!({"access_token": token, "expires_in": 3600, "token_type": "Bearer"}),
            ))
            .mount(server)
            .await;
    }

    fn account(path: &str) -> ServiceAccount {
        let http = Http::new(&shared_client(), "vertex", &ProviderConfig::default(), Duration::from_secs(5));
        ServiceAccount::new(path, http)
    }

    #[derive(Deserialize)]
    struct DecodedClaims {
        iss: String,
        scope: String,
        aud: String,
        iat: u64,
        exp: u64,
    }

    #[tokio::test]
    async fn exchanges_a_signed_assertion_and_caches_the_token() {
        let server = MockServer::start().await;
        mount_token_endpoint(&server, "ya29.test-token").await;
        let token_uri = format!("{}/token", server.uri());
        let file = service_account_file(&token_uri);
        let account = account(file.path().to_str().unwrap());

        assert_eq!(account.token().await.unwrap(), "ya29.test-token");
        assert_eq!(account.token().await.unwrap(), "ya29.test-token");
        let requests = server.received_requests().await.unwrap();
        assert_eq!(requests.len(), 1, "the second call must come from the cache");

        // The assertion is a real RS256 JWT with the claims Google expects.
        let body = String::from_utf8(requests[0].body.clone()).unwrap();
        let jwt = body.split('&').find_map(|pair| pair.strip_prefix("assertion=")).unwrap();
        let mut validation = Validation::new(Algorithm::RS256);
        validation.set_audience(&[token_uri.as_str()]);
        let key = DecodingKey::from_rsa_pem(TEST_PUBLIC_KEY.as_bytes()).unwrap();
        let decoded = jsonwebtoken::decode::<DecodedClaims>(jwt, &key, &validation).unwrap();
        assert_eq!(decoded.header.kid.as_deref(), Some("kid-1"));
        assert_eq!(decoded.claims.iss, "robot@demo.iam.gserviceaccount.com");
        assert_eq!(decoded.claims.scope, "https://www.googleapis.com/auth/cloud-platform");
        assert_eq!(decoded.claims.aud, token_uri);
        assert_eq!(decoded.claims.exp - decoded.claims.iat, 3600);
        // Sanity-check the segment encoding too.
        let payload = jwt.split('.').nth(1).unwrap();
        assert!(base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(payload).is_ok());
    }

    #[tokio::test]
    async fn expired_tokens_are_fetched_again() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({"access_token": "short", "expires_in": 30})))
            .mount(&server)
            .await;
        let file = service_account_file(&format!("{}/token", server.uri()));
        let account = account(file.path().to_str().unwrap());
        account.token().await.unwrap();
        account.token().await.unwrap();
        // 30 s is inside the refresh margin, so nothing is served from cache.
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn every_credentials_file_problem_gives_the_same_message() {
        let missing = account("/definitely/not/here.json").token().await.unwrap_err();

        let mut not_json = tempfile::NamedTempFile::new().unwrap();
        not_json.write_all(b"root:x:0:0:root:/root:/bin/sh").unwrap();
        let not_json = account(not_json.path().to_str().unwrap()).token().await.unwrap_err();

        let mut wrong_shape = tempfile::NamedTempFile::new().unwrap();
        wrong_shape.write_all(br#"{"type": "authorized_user", "refresh_token": "x"}"#).unwrap();
        let wrong_shape = account(wrong_shape.path().to_str().unwrap()).token().await.unwrap_err();

        let mut bad_key = tempfile::NamedTempFile::new().unwrap();
        bad_key
            .write_all(br#"{"client_email": "a@b", "private_key": "-----BEGIN PRIVATE KEY-----\nnope\n-----END PRIVATE KEY-----\n"}"#)
            .unwrap();
        let bad_key = account(bad_key.path().to_str().unwrap()).token().await.unwrap_err();

        for err in [missing, not_json, wrong_shape, bad_key] {
            assert!(matches!(&err, AiError::Configuration(m) if m == CREDENTIALS_FILE_ERROR), "{err:?}");
        }
    }

    #[tokio::test]
    async fn a_rejected_assertion_is_an_authentication_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(400).set_body_json(
                serde_json::json!({"error": "invalid_grant", "error_description": "Invalid JWT Signature."}),
            ))
            .mount(&server)
            .await;
        let file = service_account_file(&format!("{}/token", server.uri()));
        let err = account(file.path().to_str().unwrap()).token().await.unwrap_err();
        assert!(matches!(err, AiError::Authentication(_)), "{err:?}");
        assert!(err.to_string().contains("invalid_grant"));
    }
}
