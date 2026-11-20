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

#[derive(Debug, Clone)]
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
    pub cors_origins: Vec<String>,
    /// True when CORS_ORIGINS was not set at all (so a startup warning is due).
    pub cors_is_default: bool,
    pub max_upload_bytes: usize,
    /// How many background jobs run at once.
    pub worker_max_tasks: usize,
    pub llm_timeout: Duration,
    /// A self-hosted SearXNG instance (`SEARXNG_URL`) that Discover searches
    /// first; without it Discover uses the free public engines only.
    pub searxng_url: Option<String>,
    /// Obsidian vault sync over CouchDB (Self-hosted LiveSync). Unset = off.
    pub vault: VaultConfig,
}

/// Where the Obsidian vault's CouchDB lives and how to sign in. All optional:
/// with no URL the vault endpoints report "not configured" and nothing is
/// ever contacted. The password comes through `VAULT_COUCHDB_PASSWORD` (or
/// `_FILE`, for Docker secrets) and is never logged or echoed by the API.
#[derive(Debug, Clone, Default)]
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
        Self {
            couchdb_url: env_nonempty("VAULT_COUCHDB_URL").map(|u| u.trim_end_matches('/').to_string()),
            database: env_nonempty("VAULT_COUCHDB_DATABASE"),
            user: env_nonempty("VAULT_COUCHDB_USER"),
            password: secret_from_env("VAULT_COUCHDB_PASSWORD"),
            folder: default_vault_folder(env_nonempty("VAULT_FOLDER")),
            case_sensitive: env_nonempty("VAULT_CASE_SENSITIVE")
                .is_some_and(|v| matches!(v.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on")),
        }
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
        Self {
            host: env_nonempty("API_HOST").unwrap_or_else(|| "127.0.0.1".into()),
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
            cors_origins: parse_cors_origins(cors_raw.as_deref().unwrap_or("*")),
            cors_is_default: cors_raw.is_none(),
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
            cors_origins: vec!["*".into()],
            cors_is_default: true,
            max_upload_bytes: (DEFAULT_MAX_UPLOAD_SIZE_MB * 1024.0 * 1024.0) as usize,
            worker_max_tasks: 5,
            llm_timeout: Duration::from_secs(DEFAULT_LLM_TIMEOUT_SECONDS),
            searxng_url: None,
            vault: VaultConfig { folder: "NovaNotes".into(), ..VaultConfig::default() },
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
        !self.cors_origins.iter().any(|o| o == "*")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cors_origins_parse() {
        assert_eq!(parse_cors_origins("*"), vec!["*"]);
        assert_eq!(parse_cors_origins(" https://a.example , https://b.example ,"), vec!["https://a.example", "https://b.example"]);
        assert!(parse_cors_origins("").is_empty());
    }

    #[test]
    fn embedded_database_is_the_default() {
        let mut cfg = Config::for_tests(Path::new("/tmp/nn"));
        cfg.surreal_url = None;
        assert_eq!(cfg.database_url(), "surrealkv:///tmp/nn/surrealdb");
    }
}
