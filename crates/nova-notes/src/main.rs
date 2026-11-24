//! The `nova-notes` binary: one process serving the JSON API, the web UI and
//! the background worker.

use std::net::SocketAddr;

use nova_notes::{api, services, AppState, Config};

const USAGE: &str = "\
nova-notes: a Rust rebuild of the Open Notebook research assistant

USAGE:
    nova-notes [COMMAND]

COMMANDS:
    serve      Run the server (default)
    migrate    Apply pending database migrations and exit
    import-chat-history FILE [--dry-run]
               Import chat history exported from the Python build
               (scripts/export_langgraph_chats.py) and exit
    version    Print the version and exit
    help       Print this message

Configuration comes from the environment (a .env file in the working
directory is loaded if present). The main variables:

    API_HOST, API_PORT             Listen address (default 127.0.0.1:5055)
    DATA_FOLDER                    Data directory (default ./data)
    SURREAL_URL                    SurrealDB v2 server, e.g. ws://localhost:8000/rpc.
                                   Unset: use the embedded database in DATA_FOLDER.
    SURREAL_USER, SURREAL_PASSWORD, SURREAL_NAMESPACE, SURREAL_DATABASE
    OPEN_NOTEBOOK_ENCRYPTION_KEY   Passphrase protecting stored API keys
    OPEN_NOTEBOOK_PASSWORD         Password for the app (unset: no login; then the
                                   server only starts on a loopback API_HOST)
    OPEN_NOTEBOOK_ALLOWED_HOSTS    Extra host names to answer to, comma-separated
                                   (e.g. notes.tailnet-name.ts.net or .tailnet-name.ts.net)
    OPEN_NOTEBOOK_ALLOW_UNAUTHENTICATED
                                   true: allow no password on a non-loopback API_HOST
    CORS_ORIGINS                   Other sites allowed to call the API (default: none)
";

fn main() -> anyhow::Result<()> {
    let command = std::env::args().nth(1).unwrap_or_else(|| "serve".to_string());
    match command.as_str() {
        "help" | "--help" | "-h" => {
            print!("{USAGE}");
            return Ok(());
        }
        "version" | "--version" | "-V" => {
            println!("nova-notes {} (Open Notebook API {})", nova_notes::VERSION, nova_notes::API_COMPAT_VERSION);
            return Ok(());
        }
        "serve" | "migrate" | "import-chat-history" => {}
        other => {
            eprint!("Unknown command: {other}\n\n{USAGE}");
            std::process::exit(2);
        }
    }

    // A missing .env is fine; a malformed one is worth knowing about.
    if let Err(e) = dotenvy::dotenv() {
        if !e.not_found() {
            eprintln!("Warning: could not read .env: {e}");
        }
    }
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info,surrealdb=warn,surrealdb_core=warn")),
        )
        .init();

    // The embedded database engine recurses deeply while planning queries;
    // the default 2 MiB worker stack is not enough for it.
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .thread_stack_size(16 * 1024 * 1024)
        .build()?
        .block_on(run(&command))
}

async fn run(command: &str) -> anyhow::Result<()> {
    let config = Config::from_env();
    if command == "serve" {
        // Checked before touching the database, so a misconfigured container
        // stops at once with the reason instead of serving an open app.
        config.check_listen_safety().map_err(|message| anyhow::anyhow!(message))?;
    }
    tracing::info!("Nova Notes {} starting; database: {}", nova_notes::VERSION, describe_database(&config));

    // AppState::init waits for nothing: retry here so a database container
    // that is still starting does not kill the server.
    let state = connect_with_retry(config).await?;
    match command {
        "migrate" => {
            tracing::info!("Migrations are up to date");
            return Ok(());
        }
        "import-chat-history" => return import_chat_history(&state).await,
        _ => {}
    }

    if state.config.encryption_key.is_none() {
        tracing::warn!(
            "OPEN_NOTEBOOK_ENCRYPTION_KEY is not set. Saving AI provider API keys will fail until it is. Set it to any secret string."
        );
    }
    if state.config.cors_origins.iter().any(|o| o == "*") && state.config.password.is_none() {
        tracing::warn!("CORS_ORIGINS=* is ignored because OPEN_NOTEBOOK_PASSWORD is not set: no other site may call the API.");
    }
    if state.config.allow_unauthenticated && state.config.password.is_none() {
        tracing::warn!("OPEN_NOTEBOOK_ALLOW_UNAUTHENTICATED is set: serving without a password on {}.", state.config.host);
    }
    tracing::info!(
        "Answering to localhost, 127.0.0.1 and [::1]{}. Other host names (a Tailscale name, a LAN name) need \
         OPEN_NOTEBOOK_ALLOWED_HOSTS or OPEN_NOTEBOOK_PUBLIC_URL.",
        if state.config.allowed_hosts.is_empty() { String::new() } else { format!(", plus {}", state.config.allowed_hosts.join(", ")) }
    );
    match &state.config.searxng_url {
        Some(url) => tracing::info!("Discover searches SearXNG at {url} first, then the public engines."),
        None => tracing::info!(
            "SEARXNG_URL is not set: Discover uses public search engines only, which often block automated searches."
        ),
    }
    if state.config.password.is_none() {
        tracing::warn!("OPEN_NOTEBOOK_PASSWORD is not set: anyone who can reach this server can use it.");
    }
    if remote_database_uses_default_password(&state.config) {
        tracing::warn!(
            "SURREAL_PASSWORD is not set: signing in to the remote database {} with the default password \"root\". \
             Set SURREAL_PASSWORD (and SURREAL_USER) to the database's real credentials.",
            state.config.surreal_url.as_deref().unwrap_or_default()
        );
    }

    services::start(&state).await?;

    let address: SocketAddr = format!("{}:{}", state.config.host, state.config.port)
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid API_HOST/API_PORT: {e}"))?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    tracing::info!("Listening on http://{}", listener.local_addr()?);

    axum::serve(listener, api::build_router(state).into_make_service_with_connect_info::<SocketAddr>())
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
            tracing::info!("Shutting down");
        })
        .await?;
    Ok(())
}

/// `nova-notes import-chat-history FILE [--dry-run]`: no server, no worker.
async fn import_chat_history(state: &AppState) -> anyhow::Result<()> {
    let args: Vec<String> = std::env::args().skip(2).collect();
    let dry_run = args.iter().any(|a| a == "--dry-run");
    let Some(path) = args.iter().find(|a| !a.starts_with("--")) else {
        anyhow::bail!("Usage: nova-notes import-chat-history FILE [--dry-run]");
    };
    let file = std::fs::File::open(path).map_err(|e| anyhow::anyhow!("Could not open {path}: {e}"))?;
    let report =
        services::chat_import::import_chat_history(state, std::io::BufReader::new(file), dry_run).await?;
    let verb = if dry_run { "Would import" } else { "Imported" };
    println!(
        "{verb} {} messages into {} chat sessions. Already had messages: {}. Nothing to import: {}. Left out (system/tool) messages: {}.",
        report.imported_messages,
        report.imported_sessions,
        report.skipped_existing,
        report.skipped_empty,
        report.ignored_messages
    );
    if !report.missing_sessions.is_empty() {
        println!(
            "Sessions in the export that this database does not have ({}): {}",
            report.missing_sessions.len(),
            report.missing_sessions.join(", ")
        );
    }
    Ok(())
}

/// A remote database (ws/wss/http/https) with neither `SURREAL_PASSWORD` nor
/// `SURREAL_PASS` set is signed in to with the fallback password "root".
fn remote_database_uses_default_password(config: &Config) -> bool {
    password_is_defaulted(config.surreal_url.as_deref(), |name| std::env::var(name).ok())
}

fn password_is_defaulted(surreal_url: Option<&str>, env: impl Fn(&str) -> Option<String>) -> bool {
    let remote = surreal_url
        .is_some_and(|url| ["ws://", "wss://", "http://", "https://"].iter().any(|p| url.starts_with(p)));
    let set = |name: &str| env(name).is_some_and(|v| !v.trim().is_empty());
    remote && !set("SURREAL_PASSWORD") && !set("SURREAL_PASS")
}

fn describe_database(config: &Config) -> String {
    match &config.surreal_url {
        Some(url) => format!("{url} (namespace {}, database {})", config.surreal_namespace, config.surreal_database),
        None => format!("embedded, in {}", config.data_dir.join("surrealdb").display()),
    }
}

async fn connect_with_retry(config: Config) -> anyhow::Result<AppState> {
    const ATTEMPTS: u32 = 12;
    let mut delay = std::time::Duration::from_secs(1);
    for attempt in 1..=ATTEMPTS {
        match AppState::init(config.clone()).await {
            Ok(state) => return Ok(state),
            // Only an unreachable remote database is worth waiting for.
            Err(e) if attempt < ATTEMPTS && config.surreal_url.is_some() && e.to_string().contains("Could not connect") => {
                tracing::warn!("Database is not reachable yet (attempt {attempt}/{ATTEMPTS}): {e}. Retrying in {delay:?}...");
                tokio::time::sleep(delay).await;
                delay = (delay * 2).min(std::time::Duration::from_secs(5));
            }
            Err(e) => return Err(anyhow::anyhow!("Startup failed: {e}")),
        }
    }
    unreachable!("the last attempt always returns")
}

#[cfg(test)]
mod tests {
    use super::password_is_defaulted;

    #[test]
    fn a_remote_database_without_a_password_is_flagged() {
        let none = |_: &str| None;
        let pass = |name: &str| (name == "SURREAL_PASS").then(|| "s3cret".to_string());
        assert!(password_is_defaulted(Some("ws://db:8000/rpc"), none));
        assert!(password_is_defaulted(Some("https://db.example"), |_: &str| Some("  ".to_string())));
        assert!(!password_is_defaulted(Some("ws://db:8000/rpc"), pass));
        assert!(!password_is_defaulted(Some("surrealkv://./data/db"), none));
        assert!(!password_is_defaulted(None, none));
    }
}
