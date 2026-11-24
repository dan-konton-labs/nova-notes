//! Outbound HTTP for AI providers and fetched content, and the limits on
//! where a stored credential may send its key.
//!
//! * [`provider_client`] is the client every provider call goes through. It
//!   follows redirects only within the same host (custom key headers such
//!   as `x-api-key` would follow one to another host) and resolves host names through
//!   [`PolicyResolver`], which vets every address at connect time, so a
//!   name that re-resolves to a metadata address after validation
//!   (DNS rebinding) is still refused.
//! * [`content_client`] is the shared client for page fetches and web
//!   search: bounded redirects that may not hop to a link-local or metadata
//!   address literal, and a total timeout.
//! * [`prepare_client_config`] turns a resolved provider configuration into
//!   the one a client is built with. For configurations read from the
//!   database it refuses a base URL the provider does not accept (a key
//!   for OpenAI only ever goes to OpenAI's domain), confines the Vertex AI
//!   service-account file to the configured directories, and re-validates
//!   every URL under the configured [`UrlPolicy`].

use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use nn_content::url_validation::{check_resolved_ip, Resolver, SystemResolver};
use nn_content::UrlPolicy;
use reqwest::dns::{Addrs, Name, Resolve, Resolving};

use crate::config::ProviderNetConfig;
use crate::domain::credential::ConfigSource;
use crate::domain::ResolvedProviderConfig;
use crate::error::{AppError, AppResult};

const USER_AGENT: &str = concat!("nova-notes/", env!("CARGO_PKG_VERSION"));
const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Total limit for one request of the content client, unless the request
/// sets its own.
pub const CONTENT_REQUEST_TIMEOUT: Duration = Duration::from_secs(600);
/// Redirect hops the content client follows.
pub const MAX_CONTENT_REDIRECTS: usize = 10;

/// A DNS resolver that refuses names resolving to an address the policy
/// blocks (always link-local and cloud metadata; with
/// [`UrlPolicy::public_only`] every non-public address too).
#[derive(Clone)]
pub struct PolicyResolver {
    policy: UrlPolicy,
    resolver: Arc<dyn Resolver>,
}

impl PolicyResolver {
    pub fn new(policy: UrlPolicy, resolver: Arc<dyn Resolver>) -> Self {
        Self { policy, resolver }
    }
}

impl Resolve for PolicyResolver {
    fn resolve(&self, name: Name) -> Resolving {
        let this = self.clone();
        Box::pin(async move {
            let host = name.as_str().to_string();
            let ips = this.resolver.resolve(&host).await?;
            for ip in &ips {
                check_resolved_ip(&host, *ip, &this.policy)?;
            }
            let addrs: Addrs = Box::new(ips.into_iter().map(|ip| SocketAddr::new(ip, 0)).collect::<Vec<_>>().into_iter());
            Ok(addrs)
        })
    }
}

/// The client for AI provider calls (see the module docs). No total
/// timeout: the provider clients set one per request, and streamed replies
/// must outlive it.
pub fn provider_client(policy: UrlPolicy) -> AppResult<reqwest::Client> {
    provider_client_with(policy, Arc::new(SystemResolver))
}

/// [`provider_client`] with a given resolver (tests).
pub fn provider_client_with(policy: UrlPolicy, resolver: Arc<dyn Resolver>) -> AppResult<reqwest::Client> {
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(CONNECT_TIMEOUT)
        .redirect(same_host_redirects())
        .dns_resolver(Arc::new(PolicyResolver::new(policy, resolver)))
        .build()
        .map_err(|e| AppError::internal(format!("Could not build the HTTP client: {e}")))
}

/// Redirects for provider calls: followed only to the same host (an
/// `http` → `https` upgrade, a moved path), at most five. A redirect to
/// another host is returned as the 3xx response, so the key headers
/// (`x-api-key`, `api-key`, ...) never follow it.
fn same_host_redirects() -> reqwest::redirect::Policy {
    reqwest::redirect::Policy::custom(|attempt| {
        let next = attempt.url();
        // Same host and port, or the same host upgraded to https on its default port.
        let same_host = attempt.previous().first().is_some_and(|first| {
            first.host_str() == next.host_str()
                && (first.port_or_known_default() == next.port_or_known_default()
                    || (first.scheme() == "http" && next.scheme() == "https" && next.port().is_none()))
        });
        if attempt.previous().len() > 5 || !same_host {
            attempt.stop()
        } else {
            attempt.follow()
        }
    })
}

/// The shared client for fetched pages and web search (see the module docs).
pub fn content_client() -> AppResult<reqwest::Client> {
    let redirects = reqwest::redirect::Policy::custom(|attempt| {
        if attempt.previous().len() >= MAX_CONTENT_REDIRECTS {
            return attempt.error("too many redirects");
        }
        let blocked = match attempt.url().host() {
            Some(url::Host::Ipv4(ip)) => check_resolved_ip("", ip.into(), &UrlPolicy::self_hosted()).is_err(),
            Some(url::Host::Ipv6(ip)) => check_resolved_ip("", ip.into(), &UrlPolicy::self_hosted()).is_err(),
            _ => false,
        };
        if blocked {
            attempt.error("redirect to a link-local or metadata address refused")
        } else {
            attempt.follow()
        }
    });
    reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(CONNECT_TIMEOUT)
        .timeout(CONTENT_REQUEST_TIMEOUT)
        .redirect(redirects)
        .build()
        .map_err(|e| AppError::internal(format!("Could not build the HTTP client: {e}")))
}

// ---------------------------------------------------------------------------
// What a stored credential may point at
// ---------------------------------------------------------------------------

/// Why `base_url` cannot be used with `provider`'s stored credentials, or
/// `None` when it can.
pub fn base_url_problem(net: &ProviderNetConfig, provider: &str, base_url: &str) -> Option<String> {
    let base_url = base_url.trim();
    if base_url.is_empty() {
        return None;
    }
    match nn_ai::base_url_rule(provider) {
        nn_ai::BaseUrlRule::AnyHost => None,
        nn_ai::BaseUrlRule::None => Some(format!("A base URL cannot be set for {provider} credentials")),
        _ if net.allow_custom_base_urls => None,
        rule @ nn_ai::BaseUrlRule::Domains(domains) => {
            let allowed = url::Url::parse(base_url)
                .ok()
                .and_then(|url| url.host_str().map(|host| rule.allows(url.scheme(), host)))
                .unwrap_or(false);
            (!allowed).then(|| {
                format!(
                    "The base URL for {provider} must be an https URL on {} (use the OpenAI- or \
                     Anthropic-compatible provider for other servers)",
                    domains.join(" or ")
                )
            })
        }
    }
}

/// A Vertex AI location as it goes into the API host name
/// (`{location}-aiplatform.googleapis.com`): letters, digits and hyphens.
pub fn check_vertex_location(location: &str) -> Result<(), String> {
    let location = location.trim();
    if location.is_empty() || location.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
        Ok(())
    } else {
        Err("Invalid Vertex AI location: use a region name such as us-central1".to_string())
    }
}

/// The canonical path of a Vertex AI service-account file, which must be an
/// existing file inside one of the configured directories, or one of the
/// configured files.
pub fn confine_vertex_credentials(net: &ProviderNetConfig, path: &str) -> Result<PathBuf, String> {
    let problem = || {
        let dirs: Vec<String> = net.vertex_credentials_dirs.iter().map(|d| d.display().to_string()).collect();
        // (The configured single files are not listed: naming them adds nothing.)
        format!(
            "The Vertex AI service account file must be an existing file in {} (set OPEN_NOTEBOOK_VERTEX_CREDENTIALS_DIR)",
            dirs.join(" or ")
        )
    };
    let file = Path::new(path.trim()).canonicalize().map_err(|_| problem())?;
    if !file.is_file() {
        return Err(problem());
    }
    let inside = net
        .vertex_credentials_dirs
        .iter()
        .filter_map(|dir| dir.canonicalize().ok())
        .any(|dir| file.starts_with(dir))
        || net.vertex_credentials_files.iter().filter_map(|f| f.canonicalize().ok()).any(|f| f == file);
    if inside {
        Ok(file)
    } else {
        Err(problem())
    }
}

/// Validate every URL of a client configuration under `policy`.
pub async fn validate_config_urls(config: &nn_ai::ProviderConfig, policy: &UrlPolicy) -> Result<(), String> {
    for url in [
        &config.base_url,
        &config.endpoint,
        &config.endpoint_llm,
        &config.endpoint_embedding,
        &config.endpoint_stt,
        &config.endpoint_tts,
    ]
    .into_iter()
    .flatten()
    {
        if url.trim().is_empty() {
            continue;
        }
        nn_content::url_validation::validate_url_with(url.trim(), policy, &SystemResolver)
            .await
            .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// The client configuration for `provider` from a resolved configuration
/// (see the module docs). The error is a user-facing reason.
pub async fn prepare_client_config(
    net: &ProviderNetConfig,
    provider: &str,
    resolved: Option<(ResolvedProviderConfig, ConfigSource)>,
) -> Result<nn_ai::ProviderConfig, String> {
    let Some((resolved, source)) = resolved else { return Ok(nn_ai::ProviderConfig::default()) };
    let mut config: nn_ai::ProviderConfig = resolved.into();
    let provider = nn_ai::normalize_provider_name(provider);
    if source == ConfigSource::Database {
        // Refused, not dropped: dropping it would send the key to the
        // provider's default host, which it may never have been meant for.
        // (Vertex has no base URL at all: it is ignored below.)
        let checked = config.base_url.as_deref().filter(|_| provider != "vertex");
        if let Some(problem) = checked.and_then(|url| base_url_problem(net, &provider, url)) {
            return Err(format!(
                "{problem}. Edit the credential, or set OPEN_NOTEBOOK_ALLOW_CUSTOM_PROVIDER_BASE_URLS=true to allow it."
            ));
        }
        if provider == "vertex" {
            if let Some(path) = config.credentials_path.as_deref().filter(|p| !p.trim().is_empty()) {
                let confined = confine_vertex_credentials(net, path)?;
                config.credentials_path = Some(confined.to_string_lossy().into_owned());
            }
        }
    }
    if provider == "vertex" {
        config.base_url = None;
        if let Some(location) = &config.location {
            check_vertex_location(location)?;
        }
    }
    validate_config_urls(&config, &net.url_policy).await?;
    Ok(config)
}

#[cfg(test)]
mod tests {
    use std::net::IpAddr;

    use nn_content::url_validation::StaticResolver;
    use wiremock::matchers::method;
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::config::Config;

    fn net(dir: &Path) -> ProviderNetConfig {
        Config::for_tests(dir).providers
    }

    fn resolved(base_url: &str) -> Option<(ResolvedProviderConfig, ConfigSource)> {
        let config = ResolvedProviderConfig {
            api_key: Some("sk-stored".into()),
            base_url: Some(base_url.into()),
            ..ResolvedProviderConfig::default()
        };
        Some((config, ConfigSource::Database))
    }

    #[tokio::test]
    async fn stored_first_party_base_urls_off_the_provider_domain_are_refused() {
        let dir = tempfile::tempdir().unwrap();
        let net = net(dir.path());
        let err = prepare_client_config(&net, "openai", resolved("https://attacker.example/v1")).await.unwrap_err();
        assert!(err.contains("OPEN_NOTEBOOK_ALLOW_CUSTOM_PROVIDER_BASE_URLS"), "{err}");
        let config = prepare_client_config(&net, "anthropic", resolved("https://api.anthropic.com/v1")).await.unwrap();
        assert_eq!(config.base_url.as_deref(), Some("https://api.anthropic.com/v1"));
        let config = prepare_client_config(&net, "openai_compatible", resolved("http://127.0.0.1:1234/v1")).await.unwrap();
        assert_eq!(config.base_url.as_deref(), Some("http://127.0.0.1:1234/v1"), "custom servers keep theirs");

        // The opt-out restores any host for first-party providers.
        let open = ProviderNetConfig { allow_custom_base_urls: true, ..net.clone() };
        let config = prepare_client_config(&open, "openai", resolved("https://gateway.example/v1")).await.unwrap();
        assert_eq!(config.base_url.as_deref(), Some("https://gateway.example/v1"));

        // Environment configurations are the operator's and pass unchanged.
        let env = Some((
            ResolvedProviderConfig { base_url: Some("https://gateway.example/v1".into()), ..Default::default() },
            ConfigSource::Environment,
        ));
        assert_eq!(prepare_client_config(&net, "openai", env).await.unwrap().base_url.as_deref(), Some("https://gateway.example/v1"));
    }

    #[tokio::test]
    async fn vertex_ignores_base_urls_and_confines_the_key_file() {
        let dir = tempfile::tempdir().unwrap();
        let net = net(dir.path());
        std::fs::create_dir_all(dir.path().join("vertex")).unwrap();
        let inside = dir.path().join("vertex/sa.json");
        std::fs::write(&inside, "{}").unwrap();
        let outside = dir.path().join("sa.json");
        std::fs::write(&outside, "{}").unwrap();

        let vertex = |path: &Path, location: &str| {
            Some((
                ResolvedProviderConfig {
                    base_url: Some("https://attacker.example".into()),
                    vertex_project: Some("p".into()),
                    vertex_location: Some(location.into()),
                    credentials_path: Some(path.display().to_string()),
                    ..Default::default()
                },
                ConfigSource::Database,
            ))
        };
        let config = prepare_client_config(&net, "vertex", vertex(&inside, "us-central1")).await.unwrap();
        assert_eq!(config.base_url, None);
        assert_eq!(config.credentials_path.as_deref(), Some(inside.canonicalize().unwrap().to_str().unwrap()));

        let err = prepare_client_config(&net, "vertex", vertex(&outside, "us-central1")).await.unwrap_err();
        assert!(err.contains("service account file"), "{err}");
        let sneaky = dir.path().join("vertex/../sa.json");
        assert!(prepare_client_config(&net, "vertex", vertex(&sneaky, "us-central1")).await.is_err());
        // GOOGLE_APPLICATION_CREDENTIALS allows that one file, not its folder.
        let env_file = ProviderNetConfig { vertex_credentials_files: vec![outside.clone()], ..net.clone() };
        assert!(prepare_client_config(&env_file, "vertex", vertex(&outside, "us-central1")).await.is_ok());
        let sibling = dir.path().join("other.json");
        std::fs::write(&sibling, "{}").unwrap();
        assert!(prepare_client_config(&env_file, "vertex", vertex(&sibling, "us-central1")).await.is_err());
        let err = prepare_client_config(&net, "vertex", vertex(&inside, "attacker.example/x")).await.unwrap_err();
        assert!(err.contains("location"), "{err}");
    }

    #[tokio::test]
    async fn the_provider_client_refuses_blocked_addresses_at_connect_time() {
        let server = MockServer::start().await;
        Mock::given(method("GET")).respond_with(ResponseTemplate::new(200)).mount(&server).await;
        let port = server.address().port();
        let loopback: IpAddr = "127.0.0.1".parse().unwrap();
        let resolver = Arc::new(
            StaticResolver::new()
                .with("metadata.test", &["169.254.169.254".parse().unwrap()])
                .with("local.test", &[loopback]),
        );

        // Validation saw a harmless name; at connect it is a metadata address.
        let client = provider_client_with(UrlPolicy::self_hosted(), resolver.clone()).unwrap();
        let err = client.get(format!("http://metadata.test:{port}/")).send().await.unwrap_err();
        assert!(err.is_connect() || err.to_string().contains("error"), "{err}");
        // Local servers stay reachable by default...
        assert!(client.get(format!("http://local.test:{port}/")).send().await.unwrap().status().is_success());
        // ...and not under public_only.
        let strict = provider_client_with(UrlPolicy::public_only(), resolver).unwrap();
        assert!(strict.get(format!("http://local.test:{port}/")).send().await.is_err());
    }

    #[tokio::test]
    async fn the_content_client_does_not_redirect_to_metadata_literals() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(302).insert_header("location", "http://169.254.169.254/latest/meta-data/"))
            .mount(&server)
            .await;
        let err = content_client().unwrap().get(server.uri()).send().await.unwrap_err();
        assert!(err.is_redirect(), "{err}");
    }

    #[tokio::test]
    async fn public_only_policy_applies_to_provider_urls() {
        let dir = tempfile::tempdir().unwrap();
        let strict = ProviderNetConfig { url_policy: UrlPolicy::public_only(), ..net(dir.path()) };
        let err = prepare_client_config(&strict, "ollama", resolved("http://127.0.0.1:11434")).await.unwrap_err();
        assert!(err.contains("not allowed"), "{err}");
        assert!(prepare_client_config(&net(dir.path()), "ollama", resolved("http://127.0.0.1:11434")).await.is_ok());
    }

    #[tokio::test]
    async fn the_provider_client_follows_redirects_only_within_the_host() {
        let other = MockServer::start().await;
        Mock::given(method("GET")).respond_with(ResponseTemplate::new(200)).mount(&other).await;
        let server = MockServer::start().await;
        Mock::given(wiremock::matchers::path("/moved"))
            .respond_with(ResponseTemplate::new(301).insert_header("location", format!("{}/v1/models", server.uri())))
            .mount(&server)
            .await;
        Mock::given(wiremock::matchers::path("/away"))
            .respond_with(ResponseTemplate::new(302).insert_header("location", format!("{}/stolen", other.uri())))
            .mount(&server)
            .await;
        Mock::given(wiremock::matchers::path("/v1/models")).respond_with(ResponseTemplate::new(200)).mount(&server).await;
        let client = provider_client(UrlPolicy::self_hosted()).unwrap();
        let moved = client.get(format!("{}/moved", server.uri())).header("x-api-key", "k").send().await.unwrap();
        assert_eq!(moved.status(), 200);
        let away = client.get(format!("{}/away", server.uri())).header("x-api-key", "k").send().await.unwrap();
        assert_eq!(away.status(), 302, "a cross-host redirect is not followed");
        assert!(other.received_requests().await.unwrap().is_empty());
    }
}
