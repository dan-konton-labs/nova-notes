//! Schema migrations.
//!
//! Three tracks, each recorded in its own table so they can never collide:
//!
//! | Track | Files | Recorded in |
//! |---|---|---|
//! | upstream Open Notebook | `migrations/1..25.surrealql` | `_sbl_migrations` |
//! | Nova Notes fork | `migrations/100..106.surrealql` | `_nova_migrations` |
//! | Rust build additions | `migrations/rust/1..N.surrealql` | `_nova_rust_migrations` |
//!
//! The first two are the Python build's files, byte for byte, with the same
//! version bookkeeping, so a database created by either build opens in the
//! other. The third only ever adds things the Python build ignores.

use include_dir::{include_dir, Dir};

use super::{ensure_safe_identifier, Db, Json, Params};
use crate::error::{AppError, AppResult};

static MIGRATIONS: Dir<'static> = include_dir!("$CARGO_MANIFEST_DIR/migrations");

pub const UPSTREAM_TABLE: &str = "_sbl_migrations";
pub const NOVA_TABLE: &str = "_nova_migrations";
pub const RUST_TABLE: &str = "_nova_rust_migrations";

/// Number of upstream migrations (files `1..=UPSTREAM_COUNT`).
pub const UPSTREAM_COUNT: i64 = 25;
/// Fork migration `k` (1-based) is file / version `NOVA_BASE + k`.
pub const NOVA_BASE: i64 = 99;
pub const NOVA_COUNT: i64 = 7;
pub const RUST_COUNT: i64 = 5;

/// Before the fork got its own track its migrations were upstream-track
/// versions 26-29. Databases from that time get those records moved once.
const LEGACY_FORK_FIRST_VERSION: i64 = 26;

struct Track {
    table: &'static str,
    /// Version recorded when nothing is applied; migration `k` is `base + k`.
    base: i64,
    count: i64,
    /// Directory inside the embedded `migrations/` folder ("" for the root).
    dir: &'static str,
    label: &'static str,
}

const TRACKS: [Track; 3] = [
    Track { table: UPSTREAM_TABLE, base: 0, count: UPSTREAM_COUNT, dir: "", label: "upstream" },
    Track { table: NOVA_TABLE, base: NOVA_BASE, count: NOVA_COUNT, dir: "", label: "fork" },
    Track { table: RUST_TABLE, base: 0, count: RUST_COUNT, dir: "rust/", label: "rust" },
];

/// Strip comment lines and blank lines and join the rest with spaces, exactly
/// as the Python runner prepared each file.
fn clean_sql(raw: &str) -> String {
    raw.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with("--"))
        .collect::<Vec<_>>()
        .join(" ")
}

fn migration_sql(track: &Track, version: i64) -> AppResult<String> {
    let path = format!("{}{version}.surrealql", track.dir);
    let file = MIGRATIONS
        .get_file(&path)
        .ok_or_else(|| AppError::internal(format!("Migration file {path} is missing from the build")))?;
    let raw = file
        .contents_utf8()
        .ok_or_else(|| AppError::internal(format!("Migration file {path} is not UTF-8")))?;
    Ok(clean_sql(raw))
}

/// Versions recorded in a migrations table. A missing table reads as empty.
async fn recorded_versions(db: &Db, table: &str) -> Vec<i64> {
    let rows = db
        .query("SELECT version FROM type::table($table) ORDER BY version", Params::new().set("table", table))
        .await
        .unwrap_or_default();
    rows.iter().filter_map(|row| row.get("version").and_then(Json::as_i64)).collect()
}

async fn latest_version(db: &Db, track: &Track) -> i64 {
    recorded_versions(db, track.table).await.into_iter().max().unwrap_or(track.base)
}

/// Current version of every track.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SchemaVersion {
    pub upstream: i64,
    pub fork: i64,
    pub rust: i64,
}

impl SchemaVersion {
    pub fn is_current(&self) -> bool {
        self.upstream >= UPSTREAM_COUNT && self.fork >= NOVA_BASE + NOVA_COUNT && self.rust >= RUST_COUNT
    }
}

pub async fn current_version(db: &Db) -> SchemaVersion {
    SchemaVersion {
        upstream: latest_version(db, &TRACKS[0]).await,
        fork: latest_version(db, &TRACKS[1]).await,
        rust: latest_version(db, &TRACKS[2]).await,
    }
}

/// Move fork migrations recorded on the upstream track (versions 26+) onto the
/// fork track. Idempotent: does nothing once the fork track has any record.
async fn adopt_legacy_fork_versions(db: &Db) -> AppResult<()> {
    if !recorded_versions(db, NOVA_TABLE).await.is_empty() {
        return Ok(());
    }
    let legacy: Vec<i64> = recorded_versions(db, UPSTREAM_TABLE)
        .await
        .into_iter()
        .filter(|v| *v >= LEGACY_FORK_FIRST_VERSION)
        .collect();
    if legacy.is_empty() {
        return Ok(());
    }
    let offset = NOVA_BASE + 1 - LEGACY_FORK_FIRST_VERSION;
    let mut statements = vec!["BEGIN TRANSACTION;".to_string()];
    for old in &legacy {
        let new = old + offset;
        statements.push(format!(
            "CREATE type::thing('{NOVA_TABLE}', {new}) SET version = {new}, applied_at = time::now(), moved_from = {old};"
        ));
        statements.push(format!("DELETE type::thing('{UPSTREAM_TABLE}', {old});"));
    }
    statements.push("COMMIT TRANSACTION;".to_string());
    db.query_all(&statements.join(" "), Params::new()).await?;
    tracing::info!("Moved fork migration records {legacy:?} from {UPSTREAM_TABLE} to {NOVA_TABLE}");
    Ok(())
}

async fn run_track(db: &Db, track: &Track) -> AppResult<()> {
    let applied = latest_version(db, track).await - track.base;
    for k in (applied + 1)..=track.count {
        let version = track.base + k;
        tracing::info!("Running {} migration {version}", track.label);
        let sql = migration_sql(track, version)?;
        if !sql.is_empty() {
            // Statements are independent. The Python runner never looked at
            // per-statement results, and at least one upstream file relies on
            // that (migration 7 removes a table that does not exist on a new
            // database). Keep going, but say what failed.
            let (_, errors) = db
                .execute(&sql, Params::new())
                .await
                .map_err(|e| AppError::database(format!("Migration {version} failed: {e}")))?;
            for (statement, message) in errors {
                tracing::warn!("{} migration {version}, statement {statement}: {message}", track.label);
            }
        }
        db.query_all(
            "CREATE type::thing($table, $version) SET version = $version, applied_at = time::now()",
            Params::new().set("table", track.table).set("version", version),
        )
        .await
        .map_err(|e| AppError::database(format!("Could not record migration {version}: {e}")))?;
    }
    Ok(())
}

/// Apply every pending migration on every track. Safe to call on each start.
pub async fn run(db: &Db) -> AppResult<SchemaVersion> {
    let before = current_version(db).await;
    if before.is_current() {
        tracing::info!("Database schema is up to date ({before:?})");
        return Ok(before);
    }
    tracing::info!("Database schema needs migrating (currently {before:?})");
    adopt_legacy_fork_versions(db).await?;
    for track in &TRACKS {
        run_track(db, track).await?;
    }
    finish_index_builds(db).await?;
    let after = current_version(db).await;
    tracing::info!("Database schema migrated to {after:?}");
    Ok(after)
}

/// Make sure every index is fully built before the schema is used.
///
/// Several migrations define indexes `CONCURRENTLY`, among them the unique
/// `name` indexes of the podcast profiles. SurrealDB builds those in the
/// background, and a build started inside a migration's transaction can stall
/// for good: on small machines the profile name indexes stay "indexing" with the
/// seeded profiles' batches pending, for as long as the process runs. Every
/// later write to the table is then only queued for the index, so a profile
/// that was just created is "not found" by name until the queue catches up,
/// and the unique index cannot refuse a duplicate in the meantime.
///
/// `REBUILD INDEX` aborts such a build and indexes the records again in the
/// foreground, returning when the index is ready. Nothing else writes while
/// migrations run, so this finishes. Only indexes that are not ready are
/// rebuilt; on a fresh database those tables hold a handful of rows.
async fn finish_index_builds(db: &Db) -> AppResult<()> {
    let info = db.query_value("INFO FOR DB", Params::new()).await?;
    for table in info["tables"].as_object().map(|t| t.keys().cloned().collect::<Vec<_>>()).unwrap_or_default() {
        let table = ensure_safe_identifier(&table, "table")?;
        let table_info = db.query_value(&format!("INFO FOR TABLE {table}"), Params::new()).await?;
        for index in table_info["indexes"].as_object().map(|i| i.keys().cloned().collect::<Vec<_>>()).unwrap_or_default() {
            let index = ensure_safe_identifier(&index, "index")?;
            if index_status(db, table, index).await? == "ready" {
                continue;
            }
            tracing::info!("Building index {index} on {table}");
            db.query_all(&format!("REBUILD INDEX IF EXISTS {index} ON {table}"), Params::new()).await?;
            let status = index_status(db, table, index).await?;
            if status != "ready" {
                return Err(AppError::database(format!("Index {index} on {table} is not usable after rebuilding ({status})")));
            }
        }
    }
    Ok(())
}

/// The build status SurrealDB reports for an index: "ready" once it is usable.
async fn index_status(db: &Db, table: &str, index: &str) -> AppResult<String> {
    let info = db.query_value(&format!("INFO FOR INDEX {index} ON {table}"), Params::new()).await?;
    Ok(info["building"]["status"].as_str().unwrap_or("ready").to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_migration_file_is_embedded() {
        for track in &TRACKS {
            for k in 1..=track.count {
                migration_sql(track, track.base + k).unwrap();
            }
        }
    }

    #[test]
    fn comments_and_blank_lines_are_dropped() {
        assert_eq!(clean_sql("-- hello\n\n  DEFINE TABLE a;\n   -- x\nDEFINE TABLE b;  \n"), "DEFINE TABLE a; DEFINE TABLE b;");
    }

    #[tokio::test]
    async fn a_new_database_migrates_to_current_and_stays_there() {
        let db = Db::memory().await.unwrap();
        assert_eq!(current_version(&db).await, SchemaVersion { upstream: 0, fork: NOVA_BASE, rust: 0 });

        let version = run(&db).await.unwrap();
        assert!(version.is_current(), "{version:?}");
        assert_eq!(version, SchemaVersion { upstream: 25, fork: 106, rust: 5 });

        // The schema is really there.
        let info = db.query_value("INFO FOR DB", Params::new()).await.unwrap();
        let tables = info["tables"].as_object().unwrap();
        for table in ["notebook", "source", "note", "source_embedding", "source_insight", "chat_session", "transformation", "model", "credential", "episode", "episode_profile", "speaker_profile", "folder", "section", "meeting", "chat_message", "command", "note_pin", "login_session"] {
            assert!(tables.contains_key(table), "missing table {table}");
        }
        assert!(info["functions"].as_object().unwrap().contains_key("text_search"));
        assert!(info["functions"].as_object().unwrap().contains_key("vector_search"));

        // Running again applies nothing and records nothing new.
        assert_eq!(run(&db).await.unwrap(), version);
        assert_eq!(recorded_versions(&db, UPSTREAM_TABLE).await.len(), 25);
        assert_eq!(recorded_versions(&db, NOVA_TABLE).await, vec![100, 101, 102, 103, 104, 105, 106]);
    }

    /// Every index is usable once `run` returns: a profile created straight
    /// away is found by its (concurrently built, unique) name index, and the
    /// unique index refuses a duplicate.
    #[tokio::test]
    async fn indexes_are_built_when_run_returns() {
        let db = Db::memory().await.unwrap();
        run(&db).await.unwrap();
        let indexes = [("speaker_profile", "idx_speaker_profile_name"), ("episode_profile", "idx_episode_profile_name"), ("episode", "idx_episode_command")];
        for (table, index) in indexes {
            assert_eq!(index_status(&db, table, index).await.unwrap(), "ready", "{table}.{index}");
        }
        db.query("CREATE speaker_profile SET name = 'zz', speakers = []", Params::new()).await.unwrap();
        let found = db.query("SELECT * FROM speaker_profile WHERE name = 'zz'", Params::new()).await.unwrap();
        assert_eq!(found.len(), 1);
        assert!(db.query("CREATE speaker_profile SET name = 'zz', speakers = []", Params::new()).await.is_err());

        // What finish_index_builds runs on an index that is not ready: it
        // returns with the index ready and complete.
        db.query_all("REBUILD INDEX IF EXISTS idx_speaker_profile_name ON speaker_profile", Params::new()).await.unwrap();
        assert_eq!(index_status(&db, "speaker_profile", "idx_speaker_profile_name").await.unwrap(), "ready");
        let found = db.query("SELECT * FROM speaker_profile WHERE name = 'zz'", Params::new()).await.unwrap();
        assert_eq!(found.len(), 1);
    }

    #[tokio::test]
    async fn legacy_fork_records_move_to_the_fork_track() {
        let db = Db::memory().await.unwrap();
        // A database from before the split: upstream 1..25 plus fork 26..29.
        for v in 1..=29 {
            db.query_all(
                "CREATE type::thing($table, $version) SET version = $version, applied_at = time::now()",
                Params::new().set("table", UPSTREAM_TABLE).set("version", v),
            )
            .await
            .unwrap();
        }
        adopt_legacy_fork_versions(&db).await.unwrap();
        assert_eq!(recorded_versions(&db, UPSTREAM_TABLE).await.len(), 25);
        assert_eq!(recorded_versions(&db, NOVA_TABLE).await, vec![100, 101, 102, 103]);
        // Idempotent.
        adopt_legacy_fork_versions(&db).await.unwrap();
        assert_eq!(recorded_versions(&db, NOVA_TABLE).await, vec![100, 101, 102, 103]);
    }
}
