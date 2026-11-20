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
    OPEN_NOTEBOOK_PASSWORD         Password for the app (unset: no login)
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
    if state.config.cors_is_default {
        tracing::warn!("CORS_ORIGINS is not set: the API accepts cross-origin requests from any origin.");
    }
    match &state.config.searxng_url {
        Some(url) => tracing::info!("Discover searches SearXNG at {url} first, then the public engines."),
        None => tracing::info!(
            "SEARXNG_URL is not set: Discover uses public search engines only, which often block automated searches."
        ),
    }
    if state.config.password.is_none() {
        tracing::warn!("OPEN_NOTEBOOK_PASSWORD is not set: anyone who can reach this server can use it.");
    }

    services::start(&state).await?;

    let address: SocketAddr = format!("{}:{}", state.config.host, state.config.port)
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid API_HOST/API_PORT: {e}"))?;
    let listener = tokio::net::TcpListener::bind(address).await?;
    tracing::info!("Listening on http://{}", listener.local_addr()?);

    axum::serve(listener, api::build_router(state))
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
