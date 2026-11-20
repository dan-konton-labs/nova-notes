//! The notes library as an Obsidian vault, and the plumbing to push it into
//! the vault's CouchDB (the Self-hosted LiveSync plugin's server).
//!
//! Two layers, both additive — nothing else in the app depends on them:
//!
//! 1. **The vault tree** ([`vault_tree`]): every visible note rendered as a
//!    markdown file under `<VAULT_FOLDER>/<Folder>/<Notebook>/<Title>.md`,
//!    with YAML frontmatter carrying the note id and timestamps. This is the
//!    exact shape an Obsidian vault expects, whatever carries it there.
//! 2. **The CouchDB push** ([`push`]): writes that tree into the LiveSync
//!    database as LiveSync documents (an entry doc per file whose `children`
//!    name chunk docs holding the text). Configured entirely through the
//!    `VAULT_COUCHDB_*` environment variables; unset means off, and `status`
//!    says so instead of failing.
//!
//! Entry ids follow LiveSync: the lowercased vault path, unless
//! `VAULT_CASE_SENSITIVE` says the vault handles files case-sensitively
//! (see [`entry_id`] and issue #13).
//!
//! Safety: the push only ever writes documents whose path starts with
//! `VAULT_FOLDER` (default `NovaNotes/`), so the rest of the vault cannot be
//! touched. Nothing is deleted remotely: files that no longer exist locally
//! are only *reported* as stale. A remote file edited (in Obsidian) more
//! recently than the local note is never overwritten either: it is skipped
//! and reported as a conflict. Pulling vault edits back is a later step.

use std::collections::{HashMap, HashSet};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value as Json};
use sha2::{Digest, Sha256};

use super::notebooks::{self, NotebookQuery};
use crate::config::VaultConfig;
use crate::domain::Note;
use crate::error::{AppError, AppResult};
use crate::jobs::JobQueue;
use crate::state::AppState;

pub fn register_jobs(_jobs: &JobQueue) {}

// ---------------------------------------------------------------------------
// The vault tree
// ---------------------------------------------------------------------------

/// One note as a vault file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VaultFile {
    /// Vault-relative path, `<folder>/.../<title>.md`.
    pub path: String,
    pub note_id: String,
    /// The full markdown file: frontmatter, then the note.
    pub content: String,
    /// Milliseconds since the epoch, as LiveSync stores times.
    pub ctime: i64,
    pub mtime: i64,
}

/// A file-name piece Obsidian and every OS accept: the characters Obsidian
/// refuses (`\ / : * ? " < > | # ^ [ ]`) become spaces, whitespace collapses,
/// leading dots go (hidden files), and the result is capped at 120 characters.
pub fn clean_component(raw: &str) -> String {
    let cleaned: String = raw
        .chars()
        .map(|c| match c {
            '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|' | '#' | '^' | '[' | ']' => ' ',
            c if c.is_control() => ' ',
            c => c,
        })
        .collect();
    let cleaned = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");
    let cleaned = cleaned.trim_start_matches('.').trim().to_string();
    let capped: String = cleaned.chars().take(120).collect();
    let capped = capped.trim().to_string();
    if capped.is_empty() { "Untitled".to_string() } else { capped }
}

/// The note body a vault file carries: the content without the generated
/// attached-files section (its links point at this server, not the vault).
fn vault_body(content: &str) -> &str {
    match content.find(super::note_docs::ATTACHMENTS_MARKER) {
        Some(at) => content[..at].trim_end().trim_end_matches("---").trim_end(),
        None => content,
    }
}

fn frontmatter(note_id: &str, created: Option<chrono::DateTime<chrono::Utc>>, updated: Option<chrono::DateTime<chrono::Utc>>) -> String {
    let stamp = |value: Option<chrono::DateTime<chrono::Utc>>| {
        value.map(|d| d.to_rfc3339_opts(chrono::SecondsFormat::Secs, true)).unwrap_or_default()
    };
    format!(
        "---\nnova-note-id: {note_id}\nnova-created: {}\nnova-updated: {}\n---\n\n",
        stamp(created),
        stamp(updated)
    )
}

/// Every visible note as a vault file, stable paths, no two files sharing
/// one. Layout: `<VAULT_FOLDER>/[<folder group>/]<notebook>/<title>.md`;
/// notes outside any notebook land in `Unfiled/`.
pub async fn vault_tree(state: &AppState) -> AppResult<Vec<VaultFile>> {
    let config = &state.config.vault;
    let prefix = &config.folder;
    let notes = Note::list(state).await?;
    let notebooks = notebooks::list_notebooks(state, &NotebookQuery::default()).await?;
    let folders = notebooks::list_folders(state).await?;
    let folder_names: HashMap<&str, &str> = folders.iter().map(|f| (f.id.as_str(), f.name.as_str())).collect();
    // notebook id -> its directory under the vault folder.
    let notebook_dirs: HashMap<&str, String> = notebooks
        .iter()
        .map(|nb| {
            let dir = match nb.folder_id.as_deref().and_then(|id| folder_names.get(id)) {
                Some(group) => format!("{}/{}", clean_component(group), clean_component(&nb.name)),
                None => clean_component(&nb.name),
            };
            (nb.id.as_str(), dir)
        })
        .collect();
    // note id -> first notebook (sorted, Quick Notes last, as the UI shows it).
    let rows = state.db.query("SELECT in AS note, out AS notebook FROM artifact", crate::db::Params::new()).await?;
    let mut membership: HashMap<String, Vec<String>> = HashMap::new();
    for row in rows {
        if let (Some(note), Some(nb)) = (row.get("note").and_then(Json::as_str), row.get("notebook").and_then(Json::as_str)) {
            if note.starts_with("note:") {
                membership.entry(note.to_string()).or_default().push(nb.to_string());
            }
        }
    }

    let mut taken: HashSet<String> = HashSet::new();
    let mut out = Vec::with_capacity(notes.len());
    for note in notes {
        let mut ids = membership.remove(&note.id).unwrap_or_default();
        ids.sort();
        let home = ids
            .iter()
            .find(|id| id.as_str() != super::workspace::QUICK_NOTES_ID)
            .or_else(|| ids.first())
            .map(String::as_str);
        let dir = home.and_then(|id| notebook_dirs.get(id).cloned()).unwrap_or_else(|| "Unfiled".to_string());
        let content = note.content.as_deref().unwrap_or_default();
        let (title, _) = super::workspace::title_and_snippet(note.title.as_deref(), content);
        let stem = clean_component(&title);
        // First come, first named; a second note with the same name gets
        // " 2", " 3", ... Collisions are decided on the entry id, not the
        // path, so titles differing only by case ("Todo"/"todo") still get
        // distinct ids — one file per id, or a push could never write both.
        let mut path = format!("{prefix}/{dir}/{stem}.md");
        let mut n = 2;
        while !taken.insert(entry_id(config, &path)) {
            path = format!("{prefix}/{dir}/{stem} {n}.md");
            n += 1;
        }
        let body = vault_body(content);
        out.push(VaultFile {
            content: format!("{}{}\n", frontmatter(&note.id, note.created, note.updated), body.trim_end()),
            ctime: note.created.map(|d| d.timestamp_millis()).unwrap_or_default(),
            mtime: note.updated.map(|d| d.timestamp_millis()).unwrap_or_default(),
            path,
            note_id: note.id,
        });
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// LiveSync documents
// ---------------------------------------------------------------------------

/// Characters of text per LiveSync chunk document.
const CHUNK_CHARS: usize = 1000;

/// Split on character boundaries into chunks a LiveSync reader concatenates
/// back together (the reader accepts any chunking).
pub fn chunk_content(content: &str) -> Vec<String> {
    if content.is_empty() {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut current = String::with_capacity(CHUNK_CHARS);
    for c in content.chars() {
        current.push(c);
        if current.chars().count() >= CHUNK_CHARS {
            chunks.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// A chunk's document id: content-addressed, in LiveSync's `h:` namespace
/// with an `nn-` marker so our chunks are recognisable.
pub fn leaf_id(data: &str) -> String {
    format!("h:nn-{}", &hex::encode(Sha256::digest(data.as_bytes()))[..40])
}

/// The id LiveSync gives a file's entry document: the vault path, lowercased
/// unless the vault has "Handle files as case-sensitive" on (`path` inside
/// the document keeps its case either way). Everything that builds or looks
/// up an entry id must come through here, or a real LiveSync client cannot
/// find our documents (issue #13).
pub fn entry_id(config: &VaultConfig, path: &str) -> String {
    if config.case_sensitive {
        path.to_string()
    } else {
        path.to_lowercase()
    }
}

/// The LiveSync documents for one vault file: the chunk ("leaf") docs and the
/// entry doc naming them, with its `_id` from [`entry_id`] and the original
/// path inside, as LiveSync stores plain (non-obfuscated) paths.
pub fn livesync_docs(config: &VaultConfig, file: &VaultFile) -> (Vec<Json>, Json) {
    let chunks = chunk_content(&file.content);
    let children: Vec<String> = chunks.iter().map(|c| leaf_id(c)).collect();
    let leaves = chunks
        .iter()
        .zip(&children)
        .map(|(data, id)| json!({ "_id": id, "data": data, "type": "leaf" }))
        .collect();
    let entry = json!({
        "_id": entry_id(config, &file.path),
        "path": file.path,
        "children": children,
        "type": "plain",
        "ctime": file.ctime,
        "mtime": file.mtime,
        "size": file.content.len(),
        "eden": {},
    });
    (leaves, entry)
}

// ---------------------------------------------------------------------------
// CouchDB client
// ---------------------------------------------------------------------------

/// A thin client for one CouchDB database, built from [`VaultConfig`].
pub struct Couch {
    client: reqwest::Client,
    base: String,
    database: String,
    user: Option<String>,
    password: Option<String>,
}

impl Couch {
    /// `Configuration` (422) when the vault is not configured.
    pub fn from_config(config: &VaultConfig) -> AppResult<Self> {
        let (Some(url), Some(database)) = (config.couchdb_url.as_deref(), config.database.as_deref()) else {
            return Err(AppError::config(
                "Vault sync is not configured. Set VAULT_COUCHDB_URL and VAULT_COUCHDB_DATABASE \
                 (and VAULT_COUCHDB_USER / VAULT_COUCHDB_PASSWORD) to point at the vault's CouchDB.",
            ));
        };
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(20))
            .build()
            .map_err(|e| AppError::internal(format!("Could not build the vault HTTP client: {e}")))?;
        Ok(Self {
            client,
            base: url.trim_end_matches('/').to_string(),
            database: database.to_string(),
            user: config.user.clone(),
            password: config.password.clone(),
        })
    }

    fn request(&self, method: reqwest::Method, path: &str) -> reqwest::RequestBuilder {
        let request = self.client.request(method, format!("{}/{path}", self.base));
        match &self.user {
            Some(user) => request.basic_auth(user, self.password.as_deref()),
            None => request,
        }
    }

    /// One percent-encoded document id (slashes in vault paths included).
    fn doc_path(&self, id: &str) -> String {
        format!("{}/{}", self.database, url::form_urlencoded::byte_serialize(id.as_bytes()).collect::<String>())
    }

    async fn send(&self, request: reqwest::RequestBuilder) -> AppResult<reqwest::Response> {
        request.send().await.map_err(|e| AppError::Network(format!("Could not reach the vault CouchDB: {e}")))
    }

    /// The database's document count, proving the server answers and the
    /// database exists. Auth or a missing database are reported plainly.
    pub async fn db_info(&self) -> AppResult<Json> {
        let response = self.send(self.request(reqwest::Method::GET, &self.database)).await?;
        let status = response.status();
        let body: Json = response.json().await.unwrap_or_default();
        if !status.is_success() {
            let reason = body.get("reason").and_then(Json::as_str).unwrap_or("no reason given");
            return Err(AppError::ExternalService(format!("Vault CouchDB answered {status} for the database: {reason}")));
        }
        Ok(body)
    }

    /// A document, or `None` when it does not exist.
    pub async fn get_doc(&self, id: &str) -> AppResult<Option<Json>> {
        let response = self.send(self.request(reqwest::Method::GET, &self.doc_path(id))).await?;
        match response.status() {
            reqwest::StatusCode::NOT_FOUND => Ok(None),
            status if status.is_success() => Ok(Some(
                response.json().await.map_err(|e| AppError::ExternalService(format!("Bad CouchDB document body: {e}")))?,
            )),
            status => Err(AppError::ExternalService(format!("Vault CouchDB answered {status} reading a document"))),
        }
    }

    /// Several documents by id (`_all_docs` with `include_docs`).
    pub async fn get_docs(&self, ids: &[String]) -> AppResult<Vec<Json>> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let response = self
            .send(
                self.request(reqwest::Method::POST, &format!("{}/_all_docs?include_docs=true", self.database))
                    .json(&json!({ "keys": ids })),
            )
            .await?;
        let status = response.status();
        if !status.is_success() {
            return Err(AppError::ExternalService(format!("Vault CouchDB answered {status} reading documents")));
        }
        let body: Json = response.json().await.map_err(|e| AppError::ExternalService(format!("Bad CouchDB body: {e}")))?;
        Ok(body
            .get("rows")
            .and_then(Json::as_array)
            .map(|rows| rows.iter().filter_map(|r| r.get("doc").filter(|d| !d.is_null()).cloned()).collect())
            .unwrap_or_default())
    }

    /// Write documents in one `_bulk_docs` call. Conflicts are reported.
    pub async fn bulk_docs(&self, docs: &[Json]) -> AppResult<()> {
        if docs.is_empty() {
            return Ok(());
        }
        let response = self
            .send(self.request(reqwest::Method::POST, &format!("{}/_bulk_docs", self.database)).json(&json!({ "docs": docs })))
            .await?;
        let status = response.status();
        if !status.is_success() {
            return Err(AppError::ExternalService(format!("Vault CouchDB answered {status} writing documents")));
        }
        let body: Json = response.json().await.map_err(|e| AppError::ExternalService(format!("Bad CouchDB body: {e}")))?;
        let errors: Vec<String> = body
            .as_array()
            .into_iter()
            .flatten()
            .filter(|row| row.get("error").is_some())
            .map(|row| {
                format!(
                    "{}: {}",
                    row.get("id").and_then(Json::as_str).unwrap_or("?"),
                    row.get("error").and_then(Json::as_str).unwrap_or("error")
                )
            })
            .collect();
        if !errors.is_empty() {
            return Err(AppError::ExternalService(format!("Vault CouchDB refused some writes: {}", errors.join("; "))));
        }
        Ok(())
    }

    /// Ids (and revs) of every document whose id starts with `prefix`.
    pub async fn ids_with_prefix(&self, prefix: &str) -> AppResult<Vec<String>> {
        let start: String = url::form_urlencoded::byte_serialize(format!("\"{prefix}\"").as_bytes()).collect();
        let end: String = url::form_urlencoded::byte_serialize(format!("\"{prefix}\u{fff0}\"").as_bytes()).collect();
        let response = self
            .send(self.request(
                reqwest::Method::GET,
                &format!("{}/_all_docs?startkey={start}&endkey={end}", self.database),
            ))
            .await?;
        let status = response.status();
        if !status.is_success() {
            return Err(AppError::ExternalService(format!("Vault CouchDB answered {status} listing documents")));
        }
        let body: Json = response.json().await.map_err(|e| AppError::ExternalService(format!("Bad CouchDB body: {e}")))?;
        Ok(body
            .get("rows")
            .and_then(Json::as_array)
            .map(|rows| rows.iter().filter_map(|r| r.get("id").and_then(Json::as_str)).map(String::from).collect())
            .unwrap_or_default())
    }
}

// ---------------------------------------------------------------------------
// Status and push
// ---------------------------------------------------------------------------

/// What `GET /api/vault/status` reports. Never includes credentials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VaultStatus {
    pub configured: bool,
    /// The CouchDB base URL (no credentials live in it).
    pub url: Option<String>,
    pub database: Option<String>,
    /// The vault folder this app owns.
    pub folder: String,
    /// Local notes that would sync.
    pub local_files: usize,
    /// Whether the database answered, and its total document count.
    pub reachable: Option<bool>,
    pub remote_doc_count: Option<i64>,
    /// Entry documents already under the folder on the server.
    pub remote_files: Option<usize>,
    pub error: Option<String>,
}

pub async fn status(state: &AppState) -> AppResult<VaultStatus> {
    let config = &state.config.vault;
    let local_files = vault_tree(state).await?.len();
    let mut out = VaultStatus {
        configured: config.couchdb_url.is_some() && config.database.is_some(),
        url: config.couchdb_url.clone(),
        database: config.database.clone(),
        folder: config.folder.clone(),
        local_files,
        reachable: None,
        remote_doc_count: None,
        remote_files: None,
        error: None,
    };
    if !out.configured {
        return Ok(out);
    }
    let couch = Couch::from_config(config)?;
    match couch.db_info().await {
        Ok(info) => {
            out.reachable = Some(true);
            out.remote_doc_count = info.get("doc_count").and_then(Json::as_i64);
            out.remote_files = Some(couch.ids_with_prefix(&entry_id(config, &format!("{}/", config.folder))).await?.len());
        }
        Err(e) => {
            out.reachable = Some(false);
            out.error = Some(e.detail());
        }
    }
    Ok(out)
}

/// What a push did.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PushReport {
    /// Paths written (new or changed).
    pub pushed: Vec<String>,
    /// Paths already up to date.
    pub unchanged: usize,
    /// Entry documents under our folder on the server with no local note any
    /// more. Left alone (nothing is deleted remotely), listed for review.
    pub stale: Vec<String>,
    /// Paths whose remote copy was modified (in Obsidian) after the local
    /// note last changed. Skipped, not overwritten, listed for review.
    pub conflicts: Vec<String>,
}

/// Whether the remote entry was modified after the local note: then a push
/// would overwrite an Obsidian-side edit, so the file is skipped instead.
fn remote_is_newer(entry: &Json, file: &VaultFile) -> bool {
    entry.get("mtime").and_then(Json::as_i64).is_some_and(|remote| remote > file.mtime)
}

/// Assemble a remote entry's text from its chunk documents.
async fn remote_content(couch: &Couch, entry: &Json) -> AppResult<Option<String>> {
    let Some(children) = entry.get("children").and_then(Json::as_array) else {
        return Ok(None);
    };
    let ids: Vec<String> = children.iter().filter_map(Json::as_str).map(String::from).collect();
    let docs = couch.get_docs(&ids).await?;
    if docs.len() != ids.len() {
        return Ok(None); // a chunk is missing; treat as changed
    }
    let by_id: HashMap<&str, &str> = docs
        .iter()
        .filter_map(|d| Some((d.get("_id")?.as_str()?, d.get("data")?.as_str()?)))
        .collect();
    let mut content = String::new();
    for id in &ids {
        match by_id.get(id.as_str()) {
            Some(data) => content.push_str(data),
            None => return Ok(None),
        }
    }
    Ok(Some(content))
}

/// Push the vault tree into CouchDB, one way. New and changed files are
/// written as LiveSync documents; unchanged ones are left alone, so the
/// LiveSync clients see no spurious edits. Nothing is ever deleted.
pub async fn push(state: &AppState) -> AppResult<PushReport> {
    let config = &state.config.vault;
    // LiveSync gives ids starting with `_` a `/` prefix, which [`entry_id`]
    // does not model; such a folder would write documents no client reads.
    if config.folder.starts_with('_') {
        return Err(AppError::config("VAULT_FOLDER must not start with an underscore: LiveSync rewrites such ids."));
    }
    let couch = Couch::from_config(config)?;
    couch.db_info().await?;
    let tree = vault_tree(state).await?;
    let mut report = PushReport::default();

    let local_ids: HashSet<String> = tree.iter().map(|f| entry_id(config, &f.path)).collect();
    for remote in couch.ids_with_prefix(&entry_id(config, &format!("{}/", config.folder))).await? {
        if !local_ids.contains(remote.as_str()) && !remote.starts_with("h:") {
            report.stale.push(remote);
        }
    }

    for file in &tree {
        let existing = couch.get_doc(&entry_id(config, &file.path)).await?;
        if let Some(entry) = &existing {
            let deleted = entry.get("deleted").and_then(Json::as_bool).unwrap_or(false);
            if !deleted {
                if let Some(remote) = remote_content(&couch, entry).await? {
                    if remote == file.content {
                        report.unchanged += 1;
                        continue;
                    }
                }
                if remote_is_newer(entry, file) {
                    report.conflicts.push(file.path.clone());
                    continue;
                }
            }
        }
        let (leaves, mut entry) = livesync_docs(config, file);
        // Chunks are content-addressed: write only the missing ones, and never
        // rewrite an existing chunk (its _rev is unknown and its data fixed).
        let have: HashSet<String> = couch
            .get_docs(&leaves.iter().filter_map(|l| l.get("_id")?.as_str().map(String::from)).collect::<Vec<_>>())
            .await?
            .iter()
            .filter_map(|d| d.get("_id")?.as_str().map(String::from))
            .collect();
        let mut docs: Vec<Json> =
            leaves.into_iter().filter(|l| !have.contains(l.get("_id").and_then(Json::as_str).unwrap_or_default())).collect();
        if let Some(rev) = existing.as_ref().and_then(|e| e.get("_rev")).cloned() {
            entry["_rev"] = rev;
        }
        docs.push(entry);
        couch.bulk_docs(&docs).await?;
        report.pushed.push(file.path.clone());
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::notes::{self, NewNote};
    use crate::testing::TestApp;

    type Store = std::sync::Arc<std::sync::Mutex<HashMap<String, Json>>>;

    /// A CouchDB double speaking exactly the five requests [`Couch`] makes,
    /// over a real socket, with CouchDB's `_rev` conflict rule.
    async fn spawn_couch(store: Store) -> String {
        use axum::extract::{Path as AxPath, Query, State};
        use axum::response::IntoResponse;
        use axum::routing::{get, post};

        async fn info(State(s): State<Store>) -> axum::Json<Json> {
            axum::Json(json!({ "doc_count": s.lock().unwrap().len() }))
        }
        async fn all_docs_range(State(s): State<Store>, Query(q): Query<HashMap<String, String>>) -> axum::Json<Json> {
            let strip = |k: &str| q.get(k).map(|v| v.trim_matches('"').to_string());
            let start = strip("startkey").unwrap_or_default();
            let end = strip("endkey").unwrap_or_else(|| "\u{10ffff}".into());
            let rows: Vec<Json> =
                s.lock().unwrap().keys().filter(|k| **k >= start && **k <= end).map(|k| json!({ "id": k })).collect();
            axum::Json(json!({ "rows": rows }))
        }
        async fn all_docs_keys(State(s): State<Store>, axum::Json(body): axum::Json<Json>) -> axum::Json<Json> {
            let store = s.lock().unwrap();
            let rows: Vec<Json> = body["keys"]
                .as_array()
                .cloned()
                .unwrap_or_default()
                .iter()
                .filter_map(Json::as_str)
                .map(|k| match store.get(k) {
                    Some(doc) => json!({ "id": k, "doc": doc }),
                    None => json!({ "key": k, "error": "not_found" }),
                })
                .collect();
            axum::Json(json!({ "rows": rows }))
        }
        async fn bulk(State(s): State<Store>, axum::Json(body): axum::Json<Json>) -> axum::Json<Json> {
            let mut store = s.lock().unwrap();
            let mut out = Vec::new();
            for mut doc in body["docs"].as_array().cloned().unwrap_or_default() {
                let id = doc["_id"].as_str().unwrap_or_default().to_string();
                let current = store.get(&id).and_then(|d| d.get("_rev")).cloned();
                if current.is_some() && doc.get("_rev") != current.as_ref() {
                    out.push(json!({ "id": id, "error": "conflict" }));
                    continue;
                }
                let n: i64 =
                    current.and_then(|r| r.as_str()?.split('-').next()?.parse().ok()).unwrap_or(0);
                doc["_rev"] = json!(format!("{}-x", n + 1));
                store.insert(id.clone(), doc);
                out.push(json!({ "id": id, "ok": true }));
            }
            axum::Json(json!(out))
        }
        async fn get_one(State(s): State<Store>, AxPath((_db, id)): AxPath<(String, String)>) -> axum::response::Response {
            // `Couch` form-encodes ids, so a space arrives as `+`, as CouchDB reads it.
            let id = id.replace('+', " ");
            match s.lock().unwrap().get(&id) {
                Some(doc) => axum::Json(doc.clone()).into_response(),
                None => (axum::http::StatusCode::NOT_FOUND, axum::Json(json!({ "error": "not_found" }))).into_response(),
            }
        }

        let router = axum::Router::new()
            .route("/{db}", get(info))
            .route("/{db}/_all_docs", get(all_docs_range).post(all_docs_keys))
            .route("/{db}/_bulk_docs", post(bulk))
            .route("/{db}/{*id}", get(get_one))
            .with_state(store);
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        format!("http://{addr}")
    }

    #[tokio::test]
    async fn push_writes_lowercase_ids_and_respects_remote_edits() {
        let app = TestApp::new().await;
        let (_, nb) = app.post("/api/notebooks", serde_json::json!({ "name": "Garden", "description": "" })).await;
        let nb_id = nb["id"].as_str().unwrap().to_string();
        notes::create_note(
            &app.state,
            &NewNote { title: Some("Plan v2 Final".into()), content: "Plant in May".into(), notebook_id: Some(nb_id), ..Default::default() },
        )
        .await
        .unwrap();

        let store = Store::default();
        let url = spawn_couch(store.clone()).await;
        let mut state = app.state.clone();
        let mut config = (*state.config).clone();
        config.vault =
            VaultConfig { couchdb_url: Some(url), database: Some("vault".into()), folder: "NovaNotes".into(), ..Default::default() };
        state.config = std::sync::Arc::new(config);

        let path = "NovaNotes/Garden/Plan v2 Final.md";
        let id = "novanotes/garden/plan v2 final.md";
        let report = push(&state).await.unwrap();
        assert_eq!(report.pushed, vec![path.to_string()]);
        {
            let store = store.lock().unwrap();
            let entry = store.get(id).expect("entry under the lowercased id");
            assert_eq!(entry["path"], path, "the path inside keeps its case");
            assert!(!store.contains_key(path), "no mixed-case document is written");
        }

        // Idempotent: the second push finds the entry by its lowercase id.
        let report = push(&state).await.unwrap();
        assert_eq!((report.pushed.len(), report.unchanged, report.conflicts.len()), (0, 1, 0));

        // A remote (Obsidian) edit newer than the note is skipped as a conflict.
        {
            let mut store = store.lock().unwrap();
            let leaf = leaf_id("Edited in Obsidian");
            store.insert(leaf.clone(), json!({ "_id": leaf, "data": "Edited in Obsidian", "type": "leaf" }));
            let entry = store.get_mut(id).unwrap();
            entry["children"] = json!([leaf]);
            entry["mtime"] = json!(entry["mtime"].as_i64().unwrap() + 60_000);
        }
        let report = push(&state).await.unwrap();
        assert_eq!((report.pushed.len(), report.unchanged, report.conflicts), (0, 0, vec![path.to_string()]));
        assert_eq!(
            store.lock().unwrap().get(id).unwrap()["children"],
            json!([leaf_id("Edited in Obsidian")]),
            "the Obsidian edit is untouched"
        );

        // A remote file with no local note is reported stale, never deleted.
        store.lock().unwrap().insert(
            "novanotes/gone.md".into(),
            json!({ "_id": "novanotes/gone.md", "path": "NovaNotes/Gone.md", "children": [], "type": "plain", "ctime": 1, "mtime": 1, "size": 0, "eden": {}, "_rev": "1-x" }),
        );
        let report = push(&state).await.unwrap();
        assert_eq!(report.stale, vec!["novanotes/gone.md".to_string()]);
        assert!(store.lock().unwrap().contains_key("novanotes/gone.md"));
    }

    #[tokio::test]
    async fn titles_differing_only_by_case_get_distinct_ids() {
        let app = TestApp::new().await;
        let (_, nb) = app.post("/api/notebooks", serde_json::json!({ "name": "Garden", "description": "" })).await;
        let nb_id = nb["id"].as_str().unwrap().to_string();
        for title in ["Todo", "todo"] {
            notes::create_note(
                &app.state,
                &NewNote { title: Some(title.into()), content: format!("the {title} list"), notebook_id: Some(nb_id.clone()), ..Default::default() },
            )
            .await
            .unwrap();
        }

        let tree = vault_tree(&app.state).await.unwrap();
        let ids: HashSet<String> = tree.iter().map(|f| entry_id(&app.state.config.vault, &f.path)).collect();
        assert_eq!(ids.len(), tree.len(), "one id per file: {:?}", tree.iter().map(|f| &f.path).collect::<Vec<_>>());
        assert!(tree.iter().any(|f| f.path.ends_with(" 2.md")), "the later note is renamed");

        // Both reach the vault: nothing is reported as a conflict, ever.
        let store = Store::default();
        let url = spawn_couch(store.clone()).await;
        let mut state = app.state.clone();
        let mut config = (*state.config).clone();
        config.vault =
            VaultConfig { couchdb_url: Some(url), database: Some("vault".into()), folder: "NovaNotes".into(), ..Default::default() };
        state.config = std::sync::Arc::new(config);
        let report = push(&state).await.unwrap();
        assert_eq!((report.pushed.len(), report.conflicts.len()), (2, 0));
        let report = push(&state).await.unwrap();
        assert_eq!((report.pushed.len(), report.unchanged, report.conflicts.len()), (0, 2, 0));
    }

    #[tokio::test]
    async fn folders_starting_with_an_underscore_are_refused() {
        let app = TestApp::new().await;
        let mut state = app.state.clone();
        let mut config = (*state.config).clone();
        config.vault = VaultConfig {
            couchdb_url: Some("http://127.0.0.1:1".into()),
            database: Some("vault".into()),
            folder: "_private".into(),
            ..Default::default()
        };
        state.config = std::sync::Arc::new(config);
        let err = push(&state).await.unwrap_err();
        assert!(err.detail().contains("underscore"), "{}", err.detail());
    }

    #[test]
    fn remote_files_edited_after_the_note_are_conflicts() {
        let file = VaultFile { path: "NovaNotes/A/x.md".into(), note_id: "note:x".into(), content: "a".into(), ctime: 1, mtime: 100 };
        assert!(remote_is_newer(&serde_json::json!({ "mtime": 101 }), &file), "an Obsidian edit after the note's update is a conflict");
        assert!(!remote_is_newer(&serde_json::json!({ "mtime": 100 }), &file));
        assert!(!remote_is_newer(&serde_json::json!({ "mtime": 99 }), &file));
        assert!(!remote_is_newer(&serde_json::json!({}), &file), "no remote mtime means nothing to protect");
    }

    #[test]
    fn components_are_safe_for_obsidian_and_filesystems() {
        assert_eq!(clean_component("Plan: v2 *final*?"), "Plan v2 final");
        assert_eq!(clean_component("a/b\\c|d"), "a b c d");
        assert_eq!(clean_component(".hidden"), "hidden");
        assert_eq!(clean_component("   "), "Untitled");
        assert_eq!(clean_component("tags #work [x]"), "tags work x");
        assert!(clean_component(&"x".repeat(500)).chars().count() <= 120);
    }

    #[test]
    fn chunks_reassemble_and_leaves_are_content_addressed() {
        let text: String = ("abc ".repeat(600)).trim_end().to_string();
        let chunks = chunk_content(&text);
        assert!(chunks.len() > 1);
        assert_eq!(chunks.concat(), text);
        assert!(chunk_content("").is_empty());
        assert_eq!(leaf_id("same"), leaf_id("same"));
        assert_ne!(leaf_id("one"), leaf_id("two"));
        assert!(leaf_id("x").starts_with("h:nn-"));
    }

    #[test]
    fn livesync_entry_names_its_chunks() {
        let file = VaultFile {
            path: "NovaNotes/Garden/Tomatoes.md".into(),
            note_id: "note:1".into(),
            content: "---\nnova-note-id: note:1\n---\n\nPlant in May\n".into(),
            ctime: 1,
            mtime: 2,
        };
        let config = VaultConfig { folder: "NovaNotes".into(), ..Default::default() };
        let (leaves, entry) = livesync_docs(&config, &file);
        // LiveSync derives ids from the lowercased path (issue #13); the
        // path inside the document keeps its case.
        assert_eq!(entry["_id"], "novanotes/garden/tomatoes.md");
        assert_eq!(entry["path"], "NovaNotes/Garden/Tomatoes.md");
        let sensitive = VaultConfig { case_sensitive: true, ..config.clone() };
        assert_eq!(livesync_docs(&sensitive, &file).1["_id"], "NovaNotes/Garden/Tomatoes.md");
        assert_eq!(entry["type"], "plain");
        assert_eq!(entry["mtime"], 2);
        let children: Vec<&str> = entry["children"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
        assert_eq!(children.len(), leaves.len());
        let data: String = leaves.iter().map(|l| l["data"].as_str().unwrap()).collect();
        assert_eq!(data, file.content);
    }

    #[tokio::test]
    async fn the_tree_mirrors_notebooks_and_dedupes_names() {
        let app = TestApp::new().await;
        let (_, nb) = app.post("/api/notebooks", serde_json::json!({ "name": "Garden", "description": "" })).await;
        let nb_id = nb["id"].as_str().unwrap().to_string();
        for content in ["First tomato note", "Second tomato note"] {
            notes::create_note(
                &app.state,
                &NewNote { title: Some("Tomatoes".into()), content: content.into(), notebook_id: Some(nb_id.clone()), ..Default::default() },
            )
            .await
            .unwrap();
        }
        notes::create_note(&app.state, &NewNote { title: Some("Loose".into()), content: "No notebook".into(), ..Default::default() })
            .await
            .unwrap();

        let tree = vault_tree(&app.state).await.unwrap();
        let paths: Vec<&str> = tree.iter().map(|f| f.path.as_str()).collect();
        assert!(paths.contains(&"NovaNotes/Garden/Tomatoes.md"), "{paths:?}");
        assert!(paths.contains(&"NovaNotes/Garden/Tomatoes 2.md"), "{paths:?}");
        assert!(paths.contains(&"NovaNotes/Unfiled/Loose.md"), "{paths:?}");
        let file = tree.iter().find(|f| f.path == "NovaNotes/Unfiled/Loose.md").unwrap();
        assert!(file.content.starts_with("---\nnova-note-id: note:"), "{}", file.content);
        assert!(file.content.ends_with("No notebook\n"), "{}", file.content);
        assert!(file.mtime > 0);
    }

    #[tokio::test]
    async fn unconfigured_vault_reports_and_refuses_plainly() {
        let app = TestApp::new().await;
        let status = status(&app.state).await.unwrap();
        assert!(!status.configured);
        assert_eq!(status.folder, "NovaNotes");
        assert!(status.reachable.is_none());
        let error = push(&app.state).await.unwrap_err();
        assert_eq!(error.status(), axum::http::StatusCode::UNPROCESSABLE_ENTITY);
        assert!(error.detail().contains("VAULT_COUCHDB_URL"), "{error}");
    }
}
