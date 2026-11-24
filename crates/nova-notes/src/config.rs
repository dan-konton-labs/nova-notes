//! Runtime configuration, read once from the environment at startup.
//!
//! Variable names match the Python build so an existing `.env` keeps working.

use std::path::{Path, PathBuf};
use std::time::Duration;

/// Default cap on request bodies, in megabytes.
pub const DEFAULT_MAX_UPLOAD_SIZE_MB: f64 = 100.0;
/// Default per-call model timeout. The Python build raises esperanto's 60 s
/// default to this because long answers and local models need it.
pub const DEFAULT_LLM_TIMEOUT_SECONDS: u64 = 180;

/// `Debug` is written by hand (below) so secrets never reach a log line.
#[derive(Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    /// Root for uploads, podcast audio, note attachments and the embedded database.
    pub data_dir: PathBuf,
    /// `None` means "no SURREAL_URL given": run the embedded engine under `data_dir`.
    pub surreal_url: Option<String>,
    pub surreal_user: Option<String>,
    pub surreal_password: String,
    pub surreal_namespace: String,
    pub surreal_database: String,
    /// Instance password. `None` disables authentication entirely.
    pub password: Option<String>,
    /// Passphrase that credential encryption keys are derived from.
    pub encryption_key: Option<String>,
    /// Origins allowed to call the API from another site (`CORS_ORIGINS`).
    /// Empty (the default) means no cross-origin access at all.
    pub cors_origins: Vec<String>,
    /// True when CORS_ORIGINS was not set at all.
    pub cors_is_default: bool,
    /// Extra host names the server answers to, beyond localhost, 127.0.0.1
    /// and [::1]: `OPEN_NOTEBOOK_ALLOWED_HOSTS` plus the hosts of
    /// `OPEN_NOTEBOOK_PUBLIC_URL` / `API_URL` and a specific `API_HOST` address.
    /// Requests with any other `Host` are refused (DNS rebinding defence).
    pub allowed_hosts: Vec<String>,
    /// Origins (`scheme://host[:port]`) of `OPEN_NOTEBOOK_PUBLIC_URL` / `API_URL`:
    /// browser writes from them are same-site even when a proxy rewrites `Host`.
    pub public_origins: Vec<String>,
    /// `OPEN_NOTEBOOK_ALLOW_UNAUTHENTICATED`: start with no password even when
    /// listening on a non-loopback address.
    pub allow_unauthenticated: bool,
    /// `OPEN_NOTEBOOK_TRUST_PROXY_HEADERS`: believe `X-Forwarded-Proto`,
    /// `-Host` and `-For` (set only behind a reverse proxy that overwrites them).
    pub trust_proxy_headers: bool,
    pub max_upload_bytes: usize,
    /// How many background jobs run at once.
    pub worker_max_tasks: usize,
    pub llm_timeout: Duration,
    /// A self-hosted SearXNG instance (`SEARXNG_URL`) that Discover searches
    /// first; without it Discover uses the free public engines only.
    pub searxng_url: Option<String>,
    /// Obsidian vault sync over CouchDB (Self-hosted LiveSync). Unset = off.
    pub vault: VaultConfig,
    /// Where AI provider calls may go (stored base URLs, Vertex key files).
    pub providers: ProviderNetConfig,
    /// Which addresses link sources (and Discover results) may be fetched
    /// from: `OPEN_NOTEBOOK_SOURCE_URL_POLICY`. The default `public_only`
    /// refuses loopback, private, carrier-grade NAT (tailnet) and other
    /// internal addresses, on every redirect hop; `self_hosted` allows them
    /// for people who ingest from services on their LAN. Link-local and
    /// cloud-metadata addresses are refused under both.
    pub source_url_policy: nn_content::UrlPolicy,
    /// Longest pasted text a text source may have, in characters
    /// (`OPEN_NOTEBOOK_MAX_TEXT_SOURCE_CHARS`). The text is stored in the
    /// job arguments as well as on the source.
    pub max_text_source_chars: usize,
}

/// Default for [`Config::max_text_source_chars`].
pub const DEFAULT_MAX_TEXT_SOURCE_CHARS: usize = 10_000_000;

fn max_text_source_chars() -> usize {
    match env_nonempty("OPEN_NOTEBOOK_MAX_TEXT_SOURCE_CHARS") {
        None => DEFAULT_MAX_TEXT_SOURCE_CHARS,
        Some(raw) => raw.trim().parse::<usize>().ok().filter(|n| *n > 0).unwrap_or_else(|| {
            tracing::warn!(
                "OPEN_NOTEBOOK_MAX_TEXT_SOURCE_CHARS={raw:?} is not a positive number; using {DEFAULT_MAX_TEXT_SOURCE_CHARS}"
            );
            DEFAULT_MAX_TEXT_SOURCE_CHARS
        }),
    }
}

/// Parse `OPEN_NOTEBOOK_SOURCE_URL_POLICY`. Only `self_hosted` loosens the
/// policy; unset or anything else is `public_only`.
pub fn parse_source_url_policy(raw: Option<&str>) -> nn_content::UrlPolicy {
    match raw.map(|v| v.trim().to_ascii_lowercase()) {
        Some(v) if v == "self_hosted" => nn_content::UrlPolicy::self_hosted(),
        Some(v) if v.is_empty() || v == "public_only" => nn_content::UrlPolicy::public_only(),
        None => nn_content::UrlPolicy::public_only(),
        Some(other) => {
            tracing::warn!(
                "OPEN_NOTEBOOK_SOURCE_URL_POLICY={other} is not recognised (use public_only or self_hosted); \
                 using public_only"
            );
            nn_content::UrlPolicy::public_only()
        }
    }
}

/// Limits on the URLs and files a stored AI credential can point at.
#[derive(Debug, Clone)]
pub struct ProviderNetConfig {
    /// `OPEN_NOTEBOOK_PROVIDER_URL_POLICY=public_only` also refuses loopback
    /// and private addresses for provider URLs. The default (`self_hosted`)
    /// refuses only link-local and cloud-metadata addresses, so a local
    /// Ollama or LM Studio keeps working.
    pub url_policy: nn_content::UrlPolicy,
    /// `OPEN_NOTEBOOK_ALLOW_CUSTOM_PROVIDER_BASE_URLS=true` lets a stored
    /// credential of a first-party provider (OpenAI, Anthropic, ...) point
    /// its base URL at any host, for a proxy or gateway. Off by default:
    /// such a URL must be on the provider's own domain.
    pub allow_custom_base_urls: bool,
    /// Directories a Vertex AI credential's service-account file may be in:
    /// `OPEN_NOTEBOOK_VERTEX_CREDENTIALS_DIR` (default `DATA_FOLDER/vertex`).
    pub vertex_credentials_dirs: Vec<PathBuf>,
    /// Single files also allowed: `GOOGLE_APPLICATION_CREDENTIALS`, when set.
    pub vertex_credentials_files: Vec<PathBuf>,
}

impl ProviderNetConfig {
    fn from_env(data_dir: &Path) -> Self {
        let vertex_credentials_dirs = vec![env_nonempty("OPEN_NOTEBOOK_VERTEX_CREDENTIALS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| data_dir.join("vertex"))];
        Self {
            url_policy: match env_nonempty("OPEN_NOTEBOOK_PROVIDER_URL_POLICY").map(|v| v.trim().to_ascii_lowercase()) {
                Some(v) if v == "public_only" => nn_content::UrlPolicy::public_only(),
                _ => nn_content::UrlPolicy::self_hosted(),
            },
            allow_custom_base_urls: env_nonempty("OPEN_NOTEBOOK_ALLOW_CUSTOM_PROVIDER_BASE_URLS")
                .is_some_and(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")),
            vertex_credentials_dirs,
            vertex_credentials_files: env_nonempty("GOOGLE_APPLICATION_CREDENTIALS").map(PathBuf::from).into_iter().collect(),
        }
    }

    fn for_tests(data_dir: &Path) -> Self {
        Self {
            url_policy: nn_content::UrlPolicy::self_hosted(),
            allow_custom_base_urls: false,
            vertex_credentials_dirs: vec![data_dir.join("vertex")],
            vertex_credentials_files: Vec::new(),
        }
    }
}

/// Where the Obsidian vault's CouchDB lives and how to sign in. All optional:
/// with no URL the vault endpoints report "not configured" and nothing is
/// ever contacted. The password comes through `VAULT_COUCHDB_PASSWORD` (or
/// `_FILE`, for Docker secrets) and is never logged or echoed by the API.
#[derive(Clone, Default)]
pub struct VaultConfig {
    /// `VAULT_COUCHDB_URL`, e.g. `http://127.0.0.1:5984`.
    pub couchdb_url: Option<String>,
    /// `VAULT_COUCHDB_DATABASE`, e.g. `my_vault`.
    pub database: Option<String>,
    /// `VAULT_COUCHDB_USER`.
    pub user: Option<String>,
    /// `VAULT_COUCHDB_PASSWORD` / `VAULT_COUCHDB_PASSWORD_FILE`.
    pub password: Option<String>,
    /// `VAULT_FOLDER`: the folder inside the vault this app writes under.
    /// Everything outside it is never touched. Default `NovaNotes`.
    pub folder: String,
    /// `VAULT_CASE_SENSITIVE`: set to `true` only when the vault's LiveSync
    /// has "Handle files as case-sensitive" turned on. Off (the LiveSync
    /// default), entry document ids are the lowercased path.
    pub case_sensitive: bool,
}

impl VaultConfig {
    fn from_env() -> Self {
        // Credentials written into the URL (`http://user:pass@host:5984`) move
        // to `user` / `password`, so the URL itself can be shown and logged.
        // The explicit variables win over the URL's.
        let (couchdb_url, url_user, url_password) = match env_nonempty("VAULT_COUCHDB_URL") {
            Some(raw) => {
                let (url, user, password) = split_userinfo(raw.trim().trim_end_matches('/'));
                (Some(url), user, password)
            }
            None => (None, None, None),
        };
        if let Some(url) = couchdb_url.as_deref().filter(|u| is_remote_plain_http(u)) {
            tracing::warn!(
                "VAULT_COUCHDB_URL ({url}) is plain http:// to a non-loopback host: the vault credentials and notes \
                 cross the network unencrypted. Use https:// or a private tunnel."
            );
        }
        Self {
            couchdb_url,
            database: env_nonempty("VAULT_COUCHDB_DATABASE"),
            user: env_nonempty("VAULT_COUCHDB_USER").or(url_user),
            password: secret_from_env("VAULT_COUCHDB_PASSWORD").or(url_password),
            folder: default_vault_folder(env_nonempty("VAULT_FOLDER")),
            case_sensitive: env_nonempty("VAULT_CASE_SENSITIVE")
                .is_some_and(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")),
        }
    }
}

/// `url` without its userinfo, plus the (percent-decoded) user and password
/// it carried. A value that does not parse as a URL is returned unchanged.
pub fn split_userinfo(url: &str) -> (String, Option<String>, Option<String>) {
    let Ok(mut parsed) = url::Url::parse(url) else {
        return (url.to_string(), None, None);
    };
    let user = Some(percent_decode(parsed.username())).filter(|u| !u.is_empty());
    let password = parsed.password().map(percent_decode).filter(|p| !p.is_empty());
    if user.is_none() && password.is_none() {
        return (url.to_string(), None, None);
    }
    let _ = parsed.set_username("");
    let _ = parsed.set_password(None);
    let mut clean = parsed.to_string();
    // `Url` adds a `/` path to a bare origin; keep the configured form.
    if !url.ends_with('/') && clean.ends_with('/') && parsed.path() == "/" {
        clean.pop();
    }
    (clean, user, password)
}

/// `url` with any userinfo removed, for display.
pub fn strip_userinfo(url: &str) -> String {
    split_userinfo(url).0
}

fn percent_decode(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = |b: u8| (b as char).to_digit(16).map(|d| d as u8);
        match (bytes[i], bytes.get(i + 1).copied().and_then(hex), bytes.get(i + 2).copied().and_then(hex)) {
            (b'%', Some(hi), Some(lo)) => {
                out.push(hi << 4 | lo);
                i += 3;
            }
            (b, _, _) => {
                out.push(b);
                i += 1;
            }
        }
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// True for an `http://` URL whose host is not loopback.
fn is_remote_plain_http(url: &str) -> bool {
    let Ok(parsed) = url::Url::parse(url) else {
        return false;
    };
    let loopback = match parsed.host() {
        Some(url::Host::Ipv4(ip)) => ip.is_loopback(),
        Some(url::Host::Ipv6(ip)) => ip.is_loopback(),
        Some(url::Host::Domain(host)) => host.eq_ignore_ascii_case("localhost"),
        None => true,
    };
    parsed.scheme() == "http" && !loopback
}

const REDACTED: &str = "<redacted>";

fn redact<T>(value: &Option<T>) -> Option<&'static str> {
    value.as_ref().map(|_| REDACTED)
}

impl std::fmt::Debug for VaultConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VaultConfig")
            .field("couchdb_url", &self.couchdb_url.as_deref().map(strip_userinfo))
            .field("database", &self.database)
            .field("user", &self.user)
            .field("password", &redact(&self.password))
            .field("folder", &self.folder)
            .field("case_sensitive", &self.case_sensitive)
            .finish()
    }
}

impl std::fmt::Debug for Config {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Config")
            .field("host", &self.host)
            .field("port", &self.port)
            .field("data_dir", &self.data_dir)
            .field("surreal_url", &self.surreal_url.as_deref().map(strip_userinfo))
            .field("surreal_user", &self.surreal_user)
            .field("surreal_password", &REDACTED)
            .field("surreal_namespace", &self.surreal_namespace)
            .field("surreal_database", &self.surreal_database)
            .field("password", &redact(&self.password))
            .field("encryption_key", &redact(&self.encryption_key))
            .field("cors_origins", &self.cors_origins)
            .field("allowed_hosts", &self.allowed_hosts)
            .field("public_origins", &self.public_origins)
            .field("allow_unauthenticated", &self.allow_unauthenticated)
            .field("max_upload_bytes", &self.max_upload_bytes)
            .field("worker_max_tasks", &self.worker_max_tasks)
            .field("llm_timeout", &self.llm_timeout)
            .field("searxng_url", &self.searxng_url.as_deref().map(strip_userinfo))
            .field("vault", &self.vault)
            .field("providers", &self.providers)
            .field("source_url_policy", &self.source_url_policy)
            .field("max_text_source_chars", &self.max_text_source_chars)
            .finish_non_exhaustive()
    }
}

fn default_vault_folder(raw: Option<String>) -> String {
    raw.map(|f| f.trim().trim_matches('/').to_string()).filter(|f| !f.is_empty()).unwrap_or_else(|| "NovaNotes".into())
}

/// Read `NAME_FILE` (Docker secrets) first, then `NAME`. Empty values count as unset.
pub fn secret_from_env(name: &str) -> Option<String> {
    if let Ok(path) = std::env::var(format!("{name}_FILE")) {
        match std::fs::read_to_string(&path) {
            Ok(contents) => {
                let secret = contents.trim().to_string();
                if !secret.is_empty() {
                    return Some(secret);
                }
                tracing::warn!("{name}_FILE points to an empty file: {path}");
            }
            Err(e) => tracing::warn!("could not read {name}_FILE ({path}): {e}"),
        }
    }
    env_nonempty(name)
}

/// An environment variable, treating unset and blank the same.
pub fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|v| !v.trim().is_empty())
}

fn parse_cors_origins(raw: &str) -> Vec<String> {
    let value = raw.trim();
    if value == "*" {
        return vec!["*".to_string()];
    }
    value.split(',').map(str::trim).filter(|o| !o.is_empty()).map(String::from).collect()
}

/// `scheme://host[:port]` and the lowercased host of an absolute URL.
fn url_origin_and_host(raw: &str) -> Option<(String, String)> {
    let url = url::Url::parse(raw.trim()).ok()?;
    let host = url.host_str()?.trim_start_matches('[').trim_end_matches(']').to_ascii_lowercase();
    Some((url.origin().ascii_serialization(), host))
}

/// One `OPEN_NOTEBOOK_ALLOWED_HOSTS` entry, normalised: lowercase, no port,
/// no brackets, no trailing dot. `*` and a leading `.` are kept as written.
fn normalise_host_entry(raw: &str) -> Option<String> {
    let entry = raw.trim().to_ascii_lowercase();
    if entry.is_empty() {
        return None;
    }
    if entry == "*" {
        return Some(entry);
    }
    // Accept a full URL or host:port too, since that is what people paste.
    let entry = match url_origin_and_host(&entry) {
        Some((_, host)) if entry.contains("://") => host,
        _ => strip_port(&entry).unwrap_or(entry),
    };
    let entry = entry.trim_end_matches('.').to_string();
    (!entry.is_empty()).then_some(entry)
}

/// The host part of a `Host` header value (`name`, `name:port`, `[v6]:port`),
/// lowercased, without a trailing dot. `None` when the value is malformed.
pub fn strip_port(value: &str) -> Option<String> {
    let value = value.trim();
    let host = if let Some(rest) = value.strip_prefix('[') {
        let (inside, after) = rest.split_once(']')?;
        if !(after.is_empty() || after.strip_prefix(':').is_some_and(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()))) {
            return None;
        }
        inside
    } else {
        match value.rsplit_once(':') {
            Some((host, port)) if port.bytes().all(|b| b.is_ascii_digit()) && !port.is_empty() => host,
            Some(_) => return None,
            None => value,
        }
    };
    if host.is_empty() || (host.contains(':') && !value.starts_with('[')) {
        return None;
    }
    Some(host.trim_end_matches('.').to_ascii_lowercase())
}

/// Hosts and origins the server accepts, from the allow-list variable, the
/// public URL variables and the listen address.
fn host_allow_list(allowed: Option<&str>, public_urls: &[Option<String>], bind_host: &str) -> (Vec<String>, Vec<String>) {
    let mut hosts: Vec<String> = allowed.unwrap_or("").split(',').filter_map(normalise_host_entry).collect();
    let mut origins = Vec::new();
    for url in public_urls.iter().flatten() {
        if let Some((origin, host)) = url_origin_and_host(url) {
            hosts.push(host);
            origins.push(origin);
        }
    }
    // Listening on one specific address: that address is a valid Host.
    if let Ok(ip) = bind_host.trim_start_matches('[').trim_end_matches(']').parse::<std::net::IpAddr>() {
        if !ip.is_unspecified() && !ip.is_loopback() {
            hosts.push(ip.to_string());
        }
    }
    hosts.sort();
    hosts.dedup();
    origins.dedup();
    (hosts, origins)
}

fn truthy(value: Option<String>) -> bool {
    value.is_some_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
}

fn max_upload_bytes() -> usize {
    let raw = std::env::var("OPEN_NOTEBOOK_MAX_UPLOAD_SIZE_MB").unwrap_or_default();
    let mut mb = raw.trim().parse::<f64>().unwrap_or(DEFAULT_MAX_UPLOAD_SIZE_MB);
    if mb.is_nan() || mb <= 0.0 {
        tracing::warn!(
            "OPEN_NOTEBOOK_MAX_UPLOAD_SIZE_MB={raw:?} is not a positive size; using the default of {DEFAULT_MAX_UPLOAD_SIZE_MB}MB"
        );
        mb = DEFAULT_MAX_UPLOAD_SIZE_MB;
    }
    (mb * 1024.0 * 1024.0) as usize
}

/// The database URL, honouring the legacy SURREAL_ADDRESS / SURREAL_PORT pair.
fn surreal_url() -> Option<String> {
    if let Some(url) = env_nonempty("SURREAL_URL") {
        return Some(url);
    }
    let address = env_nonempty("SURREAL_ADDRESS")?;
    let has_port = address.rsplit(']').next().is_some_and(|tail| tail.contains(':'));
    let address = if has_port {
        address
    } else {
        format!("{address}:{}", env_nonempty("SURREAL_PORT").unwrap_or_else(|| "8000".into()))
    };
    Some(format!("ws://{address}/rpc"))
}

impl Config {
    pub fn from_env() -> Self {
        let cors_raw = std::env::var("CORS_ORIGINS").ok();
        let host = env_nonempty("API_HOST").unwrap_or_else(|| "127.0.0.1".into());
        let (allowed_hosts, public_origins) = host_allow_list(
            env_nonempty("OPEN_NOTEBOOK_ALLOWED_HOSTS").as_deref(),
            &[env_nonempty("OPEN_NOTEBOOK_PUBLIC_URL"), env_nonempty("API_URL")],
            &host,
        );
        Self {
            host,
            port: env_nonempty("API_PORT").and_then(|p| p.parse().ok()).unwrap_or(5055),
            data_dir: PathBuf::from(env_nonempty("DATA_FOLDER").unwrap_or_else(|| "./data".into())),
            surreal_url: surreal_url(),
            surreal_user: env_nonempty("SURREAL_USER"),
            surreal_password: env_nonempty("SURREAL_PASSWORD")
                .or_else(|| env_nonempty("SURREAL_PASS"))
                .unwrap_or_else(|| "root".into()),
            surreal_namespace: env_nonempty("SURREAL_NAMESPACE").unwrap_or_else(|| "open_notebook".into()),
            surreal_database: env_nonempty("SURREAL_DATABASE").unwrap_or_else(|| "open_notebook".into()),
            password: secret_from_env("OPEN_NOTEBOOK_PASSWORD"),
            encryption_key: secret_from_env("OPEN_NOTEBOOK_ENCRYPTION_KEY"),
            cors_origins: parse_cors_origins(cors_raw.as_deref().unwrap_or("")),
            cors_is_default: cors_raw.is_none(),
            allowed_hosts,
            public_origins,
            allow_unauthenticated: truthy(env_nonempty("OPEN_NOTEBOOK_ALLOW_UNAUTHENTICATED")),
            trust_proxy_headers: truthy(env_nonempty("OPEN_NOTEBOOK_TRUST_PROXY_HEADERS")),
            max_upload_bytes: max_upload_bytes(),
            worker_max_tasks: env_nonempty("OPEN_NOTEBOOK_WORKER_MAX_TASKS")
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|n| *n > 0)
                .unwrap_or(5),
            llm_timeout: Duration::from_secs(
                env_nonempty("ESPERANTO_LLM_TIMEOUT")
                    .and_then(|v| v.parse::<f64>().ok())
                    .filter(|s| *s > 0.0)
                    .map(|s| s as u64)
                    .unwrap_or(DEFAULT_LLM_TIMEOUT_SECONDS),
            ),
            searxng_url: env_nonempty("SEARXNG_URL").map(|u| u.trim().trim_end_matches('/').to_string()),
            vault: VaultConfig::from_env(),
            providers: ProviderNetConfig::from_env(Path::new(&env_nonempty("DATA_FOLDER").unwrap_or_else(|| "./data".into()))),
            source_url_policy: parse_source_url_policy(env_nonempty("OPEN_NOTEBOOK_SOURCE_URL_POLICY").as_deref()),
            max_text_source_chars: max_text_source_chars(),
        }
    }

    /// A configuration for tests: in-memory database, data under `data_dir`.
    pub fn for_tests(data_dir: &Path) -> Self {
        Self {
            host: "127.0.0.1".into(),
            port: 0,
            data_dir: data_dir.to_path_buf(),
            surreal_url: Some("mem://".into()),
            surreal_user: None,
            surreal_password: "root".into(),
            surreal_namespace: "open_notebook".into(),
            surreal_database: "open_notebook".into(),
            password: None,
            encryption_key: Some("test-encryption-key".into()),
            cors_origins: Vec::new(),
            cors_is_default: true,
            allowed_hosts: Vec::new(),
            public_origins: Vec::new(),
            allow_unauthenticated: false,
            trust_proxy_headers: false,
            max_upload_bytes: (DEFAULT_MAX_UPLOAD_SIZE_MB * 1024.0 * 1024.0) as usize,
            worker_max_tasks: 5,
            llm_timeout: Duration::from_secs(DEFAULT_LLM_TIMEOUT_SECONDS),
            searxng_url: None,
            vault: VaultConfig { folder: "NovaNotes".into(), ..VaultConfig::default() },
            providers: ProviderNetConfig::for_tests(data_dir),
            // Tests serve pages from loopback mock servers. The production
            // default (`public_only`) is covered by its own tests.
            source_url_policy: nn_content::UrlPolicy::self_hosted(),
            max_text_source_chars: DEFAULT_MAX_TEXT_SOURCE_CHARS,
        }
    }

    /// The URL the database layer connects to: SURREAL_URL when given, the
    /// embedded engine under the data directory otherwise.
    pub fn database_url(&self) -> String {
        match &self.surreal_url {
            Some(url) => url.clone(),
            None => format!("surrealkv://{}", self.data_dir.join("surrealdb").display()),
        }
    }

    pub fn uploads_dir(&self) -> PathBuf {
        self.data_dir.join("uploads")
    }
    pub fn podcasts_dir(&self) -> PathBuf {
        self.data_dir.join("podcasts")
    }
    pub fn note_files_dir(&self) -> PathBuf {
        self.data_dir.join("note_files")
    }

    /// Create the data directories the app writes into.
    pub fn ensure_dirs(&self) -> std::io::Result<()> {
        for dir in [self.uploads_dir(), self.podcasts_dir(), self.note_files_dir()] {
            std::fs::create_dir_all(dir)?;
        }
        Ok(())
    }

    /// With CORS_ORIGINS resolving to a wildcard we never allow credentials:
    /// the two together would reflect any Origin.
    pub fn cors_allow_credentials(&self) -> bool {
        !self.cors_wildcard()
    }

    /// `CORS_ORIGINS=*`, honoured only when a password is set: without one a
    /// wildcard would hand the whole API to every website.
    pub fn cors_wildcard(&self) -> bool {
        self.password.is_some() && self.cors_origins.iter().any(|o| o == "*")
    }

    /// Whether `host` (a `Host` header's host part, no port) is one this
    /// server answers to.
    pub fn host_is_allowed(&self, host: &str) -> bool {
        const LOOPBACK: [&str; 3] = ["localhost", "127.0.0.1", "::1"];
        let host = host.trim_end_matches('.').to_ascii_lowercase();
        LOOPBACK.contains(&host.as_str())
            || self.allowed_hosts.iter().any(|entry| {
                entry == "*"
                    || *entry == host
                    || entry.strip_prefix('.').is_some_and(|suffix| host == suffix || host.ends_with(entry.as_str()))
            })
    }

    /// Refuse to serve an unauthenticated app on a network address. Loopback
    /// (and `localhost`) is fine; anything else needs `OPEN_NOTEBOOK_PASSWORD`
    /// or the explicit `OPEN_NOTEBOOK_ALLOW_UNAUTHENTICATED=true` opt-in.
    pub fn check_listen_safety(&self) -> Result<(), String> {
        if self.password.is_some() || self.allow_unauthenticated {
            return Ok(());
        }
        let bare = self.host.trim_start_matches('[').trim_end_matches(']');
        let loopback = bare.eq_ignore_ascii_case("localhost")
            || bare.parse::<std::net::IpAddr>().is_ok_and(|ip| ip.is_loopback());
        if loopback {
            return Ok(());
        }
        Err(format!(
            "Refusing to start: API_HOST={} listens on the network but OPEN_NOTEBOOK_PASSWORD is not set, \
             so anyone who can reach this port could use the app and its stored API keys. \
             Set OPEN_NOTEBOOK_PASSWORD (recommended), or set OPEN_NOTEBOOK_ALLOW_UNAUTHENTICATED=true \
             if something else (a firewall, Tailscale ACLs) already limits who can connect. \
             In Docker, pass one of them with -e or in the compose file's environment.",
            self.host
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debug_output_omits_secrets() {
        let mut config = Config::for_tests(Path::new("/tmp/nn-debug"));
        config.password = Some("instance-pass-123".into());
        config.encryption_key = Some("enc-key-456".into());
        config.surreal_password = "surreal-pass-789".into();
        config.surreal_url = Some("ws://root:url-pass-000@db:8000/rpc".into());
        config.vault = VaultConfig {
            couchdb_url: Some("http://admin:couch-url-pass@couch.lan:5984".into()),
            user: Some("admin".into()),
            password: Some("vault-pass-321".into()),
            ..VaultConfig::default()
        };
        let printed = format!("{config:?} {:#?}", config.vault);
        for secret in ["instance-pass-123", "enc-key-456", "surreal-pass-789", "url-pass-000", "couch-url-pass", "vault-pass-321"] {
            assert!(!printed.contains(secret), "{secret} leaked: {printed}");
        }
        assert!(printed.contains("<redacted>") && printed.contains("couch.lan"), "{printed}");
    }

    #[test]
    fn couchdb_url_credentials_move_out_of_the_url() {
        assert_eq!(
            split_userinfo("http://admin:p%40ss%3Aw0rd@couch.lan:5984"),
            ("http://couch.lan:5984".to_string(), Some("admin".to_string()), Some("p@ss:w0rd".to_string()))
        );
        assert_eq!(split_userinfo("https://couch.example/db/"), ("https://couch.example/db/".to_string(), None, None));
        assert_eq!(strip_userinfo("https://u:p@couch.example/base"), "https://couch.example/base");
        assert!(is_remote_plain_http("http://couch.lan:5984"));
        assert!(!is_remote_plain_http("http://127.0.0.1:5984"));
        assert!(!is_remote_plain_http("http://localhost:5984"));
        assert!(!is_remote_plain_http("https://couch.example"));
    }

    #[test]
    fn source_url_policy_fails_closed() {
        use nn_content::UrlPolicy;
        assert_eq!(parse_source_url_policy(None), UrlPolicy::public_only());
        assert_eq!(parse_source_url_policy(Some("")), UrlPolicy::public_only());
        assert_eq!(parse_source_url_policy(Some("public_only")), UrlPolicy::public_only());
        assert_eq!(parse_source_url_policy(Some(" Self_Hosted ")), UrlPolicy::self_hosted());
        assert_eq!(parse_source_url_policy(Some("off")), UrlPolicy::public_only());
    }

    #[test]
    fn cors_origins_parse() {
        assert_eq!(parse_cors_origins("*"), vec!["*"]);
        assert_eq!(parse_cors_origins(" https://a.example , https://b.example ,"), vec!["https://a.example", "https://b.example"]);
        assert!(parse_cors_origins("").is_empty());
    }

    #[test]
    fn host_headers_lose_their_port() {
        assert_eq!(strip_port("Notes.Example.ts.net:5055").as_deref(), Some("notes.example.ts.net"));
        assert_eq!(strip_port("localhost").as_deref(), Some("localhost"));
        assert_eq!(strip_port("localhost.").as_deref(), Some("localhost"));
        assert_eq!(strip_port("[::1]:5055").as_deref(), Some("::1"));
        assert_eq!(strip_port("[::1]").as_deref(), Some("::1"));
        assert_eq!(strip_port("::1"), None);
        assert_eq!(strip_port("host:port"), None);
        assert_eq!(strip_port("[::1]x"), None);
        assert_eq!(strip_port(""), None);
    }

    #[test]
    fn allowed_hosts_come_from_the_list_the_public_urls_and_the_bind_address() {
        let (hosts, origins) = host_allow_list(
            Some(" Lab.Example.ts.net , .example-tailnet.ts.net, http://nas.lan:8080 , box:5055,"),
            &[Some("https://notes.example.ts.net:8502/".into()), None, Some("not a url".into())],
            "192.168.1.20",
        );
        for host in ["lab.example.ts.net", ".example-tailnet.ts.net", "nas.lan", "box", "notes.example.ts.net", "192.168.1.20"] {
            assert!(hosts.contains(&host.to_string()), "{host} in {hosts:?}");
        }
        assert_eq!(origins, vec!["https://notes.example.ts.net:8502"]);
        let (hosts, _) = host_allow_list(None, &[], "0.0.0.0");
        assert!(hosts.is_empty(), "{hosts:?}");

        let mut cfg = Config::for_tests(Path::new("/tmp/nn"));
        for host in ["localhost", "127.0.0.1", "::1", "LOCALHOST."] {
            assert!(cfg.host_is_allowed(host), "{host}");
        }
        assert!(!cfg.host_is_allowed("attacker.example"));
        cfg.allowed_hosts = hosts_of(&[".example-tailnet.ts.net", "nas.lan"]);
        assert!(cfg.host_is_allowed("lab.example-tailnet.ts.net"));
        assert!(cfg.host_is_allowed("example-tailnet.ts.net"));
        assert!(cfg.host_is_allowed("nas.lan"));
        assert!(!cfg.host_is_allowed("evilexample-tailnet.ts.net"));
        assert!(!cfg.host_is_allowed("nas.lan.evil.example"));
        cfg.allowed_hosts = hosts_of(&["*"]);
        assert!(cfg.host_is_allowed("anything.example"));
    }

    fn hosts_of(list: &[&str]) -> Vec<String> {
        list.iter().map(|h| h.to_string()).collect()
    }

    #[test]
    fn no_password_on_a_network_address_needs_an_explicit_opt_in() {
        let mut cfg = Config::for_tests(Path::new("/tmp/nn"));
        for host in ["127.0.0.1", "localhost", "::1", "[::1]", "127.0.0.2"] {
            cfg.host = host.into();
            assert!(cfg.check_listen_safety().is_ok(), "{host}");
        }
        for host in ["0.0.0.0", "::", "192.168.1.20", "nas.lan"] {
            cfg.host = host.into();
            let error = cfg.check_listen_safety().unwrap_err();
            assert!(error.contains("OPEN_NOTEBOOK_PASSWORD") && error.contains("OPEN_NOTEBOOK_ALLOW_UNAUTHENTICATED"), "{error}");
        }
        cfg.allow_unauthenticated = true;
        assert!(cfg.check_listen_safety().is_ok());
        cfg.allow_unauthenticated = false;
        cfg.password = Some("hunter2".into());
        assert!(cfg.check_listen_safety().is_ok());
    }

    #[test]
    fn a_cors_wildcard_needs_a_password() {
        let mut cfg = Config::for_tests(Path::new("/tmp/nn"));
        cfg.cors_origins = vec!["*".into()];
        assert!(!cfg.cors_wildcard());
        cfg.password = Some("hunter2".into());
        assert!(cfg.cors_wildcard());
        assert!(!cfg.cors_allow_credentials());
    }

    #[test]
    fn embedded_database_is_the_default() {
        let mut cfg = Config::for_tests(Path::new("/tmp/nn"));
        cfg.surreal_url = None;
        assert_eq!(cfg.database_url(), "surrealkv:///tmp/nn/surrealdb");
    }
}
