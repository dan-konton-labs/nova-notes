//! Sources: adding them (text, link, upload), processing them in the
//! background, and everything the source pages show.
//!
//! The Rust counterpart of `api/routers/sources.py`,
//! `commands/source_commands.py` (`process_source`) and
//! `open_notebook/graphs/source.py`. Processing is: extract the content
//! (`nn-content`), save title and full text, queue the embedding if asked,
//! then run the requested transformations, each of which queues an insight.

use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

use axum::http::StatusCode;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::notebooks::{ensure_table, not_found_as, timestamp};
use super::notes::upload_file_name;
use super::transformations::apply_transformation;
use crate::config::{env_nonempty, Config};
use crate::db::{retry_conflicts, Doc, Json, Params};
use crate::domain::{null_default, Asset, ContentSettings, Notebook, Source, Transformation};
use crate::error::{AppError, AppResult};
use crate::jobs::{status, JobContext, JobQueue, RetryPolicy};
use crate::state::AppState;

pub const SOURCE_NOT_FOUND: &str = "Source not found";
/// Title of a source until processing finds a better one.
pub const PLACEHOLDER_TITLE: &str = "Processing...";
/// Most notebooks or transformations one request may name.
pub const MAX_SOURCE_LIST_ITEMS: usize = 50;
/// How long `POST /sources` waits for synchronous processing.
pub const SYNC_PROCESSING_TIMEOUT: Duration = Duration::from_secs(300);
const UPLOADS_CONTAINMENT_ERROR: &str = "Invalid file path: must be within the uploads directory";

/// `process_source`: up to 15 attempts (1-120 s backoff), because deep queues
/// meet SurrealDB transaction conflicts; validation, configuration,
/// context-length, incomplete-generation and missing-record errors are final.
pub fn register_jobs(jobs: &JobQueue) {
    jobs.register(
        "process_source",
        RetryPolicy::attempts(15, Duration::from_secs(1), Duration::from_secs(120)),
        process_source_job,
    );
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

/// A source as the API reports it (`SourceResponse`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceView {
    pub id: String,
    pub title: Option<String>,
    pub topics: Option<Vec<String>>,
    pub asset: Option<Asset>,
    pub full_text: Option<String>,
    pub embedded: bool,
    pub embedded_chunks: i64,
    /// Whether the uploaded original is still on disk (`None`: no file).
    pub file_available: Option<bool>,
    pub created: String,
    pub updated: String,
    pub command_id: Option<String>,
    pub status: Option<String>,
    pub processing_info: Option<Value>,
    pub notebooks: Option<Vec<String>>,
}

impl SourceView {
    fn new(source: Source, embedded_chunks: i64) -> Self {
        Self {
            id: source.id,
            title: source.title,
            topics: Some(source.topics),
            asset: source.asset,
            full_text: source.full_text,
            embedded: embedded_chunks > 0,
            embedded_chunks,
            file_available: None,
            created: timestamp(source.created),
            updated: timestamp(source.updated),
            command_id: None,
            status: None,
            processing_info: None,
            notebooks: None,
        }
    }
}

/// One row of `GET /sources` (`SourceListResponse`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceListItem {
    pub id: String,
    pub title: Option<String>,
    pub topics: Option<Vec<String>>,
    pub asset: Option<Asset>,
    pub embedded: bool,
    /// Always 0 in the list (see `GET /sources/{id}` for the count).
    pub embedded_chunks: i64,
    pub insights_count: i64,
    pub created: String,
    pub updated: String,
    pub file_available: Option<bool>,
    pub command_id: Option<String>,
    pub status: Option<String>,
    pub processing_info: Option<Value>,
}

/// Answer of `GET /sources/{id}/status` (`SourceStatusResponse`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceStatusView {
    pub status: Option<String>,
    pub message: String,
    pub processing_info: Option<Value>,
    pub command_id: Option<String>,
}

// ---------------------------------------------------------------------------
// Loading and listing
// ---------------------------------------------------------------------------

/// A source by id; another table's id or a missing record → 404 "Source not found".
pub async fn load_source(state: &AppState, id: &str) -> AppResult<Source> {
    ensure_table(id, Source::TABLE, SOURCE_NOT_FOUND)?;
    Source::get(state, id).await.map_err(not_found_as(SOURCE_NOT_FOUND))
}

/// Cap error text shown to clients at 200 characters.
fn truncate_error(message: Option<String>) -> Option<String> {
    message.map(|m| if m.chars().count() <= 200 { m } else { format!("{}…", m.chars().take(200).collect::<String>()) })
}

/// `GET /sources` parameters.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceListQuery {
    pub notebook_id: Option<String>,
    /// 1 to 100.
    pub limit: i64,
    pub offset: i64,
    /// `type`, `title`, `created`, `updated`, `insights_count` or `embedded`.
    pub sort_by: String,
    /// `asc` or `desc`.
    pub sort_order: String,
}

impl Default for SourceListQuery {
    fn default() -> Self {
        Self { notebook_id: None, limit: 50, offset: 0, sort_by: "updated".into(), sort_order: "desc".into() }
    }
}

/// `sort_by` → the column or alias sorted on. `title` has a full-text index
/// SurrealDB cannot sort with, so it sorts on a lower-cased alias.
fn sort_column(sort_by: &str) -> Option<&'static str> {
    Some(match sort_by {
        "created" => "created",
        "updated" => "updated",
        "title" => "title_sort",
        "insights_count" => "insights_count",
        "embedded" => "embedded",
        "type" => "type",
        _ => return None,
    })
}

#[derive(Deserialize)]
struct ListRow {
    id: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default, deserialize_with = "null_default")]
    topics: Vec<String>,
    #[serde(default)]
    asset: Option<Asset>,
    #[serde(default, deserialize_with = "null_default")]
    embedded: bool,
    #[serde(default, deserialize_with = "null_default")]
    insights_count: i64,
    #[serde(default)]
    created: Option<DateTime<Utc>>,
    #[serde(default)]
    updated: Option<DateTime<Utc>>,
    #[serde(default)]
    command: Json,
}

/// A page of sources, optionally only one notebook's.
pub async fn list_sources(state: &AppState, query: &SourceListQuery) -> AppResult<Vec<SourceListItem>> {
    let column = sort_column(&query.sort_by).ok_or_else(|| {
        AppError::bad_request("sort_by must be one of: type, title, created, updated, insights_count, embedded")
    })?;
    let direction = query.sort_order.to_lowercase();
    if direction != "asc" && direction != "desc" {
        return Err(AppError::bad_request("sort_order must be 'asc' or 'desc'"));
    }
    let mut params = Params::new().set("limit", query.limit.clamp(1, 100)).set("offset", query.offset.max(0));
    let from = match query.notebook_id.as_deref().filter(|id| !id.is_empty()) {
        Some(notebook_id) => {
            ensure_table(notebook_id, Notebook::TABLE, "Notebook not found")?;
            Notebook::get(state, notebook_id).await.map_err(not_found_as("Notebook not found"))?;
            params = params.record("notebook_id", notebook_id);
            "(SELECT VALUE in FROM reference WHERE out = $notebook_id)"
        }
        None => "source",
    };
    let sql = format!(
        "SELECT id, asset, created, title, updated, topics, command, \
            string::lowercase(title OR '') AS title_sort, \
            (IF asset.file_path != NONE THEN 'file' ELSE IF asset.url != NONE THEN 'link' ELSE 'text' END) AS type, \
            array::len((SELECT id FROM source_insight WHERE source = $parent.id)) AS insights_count, \
            (SELECT VALUE id FROM source_embedding WHERE source = $parent.id LIMIT 1) != [] AS embedded \
         FROM {from} ORDER BY {column} {}, id ASC LIMIT $limit START $offset FETCH command",
        direction.to_uppercase()
    );
    let rows: Vec<ListRow> = state.db.query_as(&sql, params).await?;
    Ok(rows
        .into_iter()
        .map(|row| {
            let (command_id, status, processing_info) = match &row.command {
                Json::Object(command) => {
                    let metadata = command.get("result").and_then(|r| r.get("execution_metadata"));
                    let error = command.get("error_message").and_then(Json::as_str).map(String::from);
                    (
                        command.get("id").and_then(Json::as_str).map(String::from),
                        command.get("status").and_then(Json::as_str).map(String::from),
                        Some(json!({
                            "started_at": metadata.and_then(|m| m.get("started_at")),
                            "completed_at": metadata.and_then(|m| m.get("completed_at")),
                            "error": truncate_error(error),
                        })),
                    )
                }
                // A link whose job record is gone.
                Json::String(id) => (Some(id.clone()), Some("unknown".to_string()), None),
                _ => (None, None, None),
            };
            SourceListItem {
                id: row.id,
                title: row.title,
                topics: Some(row.topics),
                asset: row.asset,
                embedded: row.embedded,
                embedded_chunks: 0,
                insights_count: row.insights_count,
                created: timestamp(row.created),
                updated: timestamp(row.updated),
                file_available: None,
                command_id,
                status,
                processing_info,
            }
        })
        .collect())
}

/// A source with its status, chunk count and notebooks; stamps it as viewed.
pub async fn get_source(state: &AppState, id: &str) -> AppResult<SourceView> {
    let source = load_source(state, id).await?;
    // Best effort: a failed stamp never fails the read.
    if let Err(e) = state.db.query("UPDATE $id SET last_viewed_at = time::now()", Params::new().record("id", &source.id)).await {
        tracing::warn!("Failed to stamp last_viewed_at for source {}: {e}", source.id);
    }
    let (status, processing_info) = match source.command {
        Some(_) => (source.status(state).await, source.processing_progress(state).await),
        None => (None, None),
    };
    let embedded_chunks = source.embedded_chunks(state).await?;
    let notebooks = source.notebook_ids(state).await?;
    let file_available = file_available(&state.config, &source);
    let command_id = source.command.clone();
    Ok(SourceView {
        file_available,
        command_id,
        status,
        processing_info,
        notebooks: Some(notebooks),
        ..SourceView::new(source, embedded_chunks)
    })
}

/// Body of `PUT /sources/{id}`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceUpdate {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub topics: Option<Vec<String>>,
}

pub async fn update_source(state: &AppState, id: &str, update: &SourceUpdate) -> AppResult<SourceView> {
    let mut source = load_source(state, id).await?;
    if let Some(title) = &update.title {
        source.title = Some(title.clone());
    }
    if let Some(topics) = &update.topics {
        source.topics = topics.clone();
    }
    source.save(state).await.map_err(not_found_as(SOURCE_NOT_FOUND))?;
    let embedded_chunks = source.embedded_chunks(state).await?;
    Ok(SourceView::new(source, embedded_chunks))
}

/// Delete a source, its uploaded file, chunks and insights.
pub async fn delete_source(state: &AppState, id: &str) -> AppResult<()> {
    load_source(state, id).await?.delete(state).await
}

/// Processing status of a source, with a sentence describing it.
pub async fn source_status(state: &AppState, id: &str) -> AppResult<SourceStatusView> {
    let source = load_source(state, id).await?;
    let Some(command_id) = source.command.clone() else {
        return Ok(SourceStatusView {
            status: None,
            message: "Legacy source (completed before async processing)".to_string(),
            processing_info: None,
            command_id: None,
        });
    };
    let status = source.status(state).await;
    let processing_info = source.processing_progress(state).await;
    let message = match status.as_deref() {
        Some("completed") => "Source processing completed successfully".to_string(),
        Some("failed") => "Source processing failed".to_string(),
        Some("running") => "Source processing in progress".to_string(),
        Some("queued") => "Source processing queued".to_string(),
        Some("unknown") => "Source processing status unknown".to_string(),
        Some(other) => format!("Source processing status: {other}"),
        None => "Source processing status: None".to_string(),
    };
    Ok(SourceStatusView { status, message, processing_info, command_id: Some(command_id) })
}

// ---------------------------------------------------------------------------
// Uploaded files
// ---------------------------------------------------------------------------

/// `path` made absolute and free of `.` / `..`, with symlinks resolved as far
/// as the path exists.
fn resolve_path(path: &Path) -> PathBuf {
    let absolute = std::path::absolute(path).unwrap_or_else(|_| path.to_path_buf());
    let mut normal = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::ParentDir => {
                normal.pop();
            }
            Component::CurDir => {}
            other => normal.push(other),
        }
    }
    if let Ok(canonical) = normal.canonicalize() {
        return canonical;
    }
    match (normal.parent().and_then(|p| p.canonicalize().ok()), normal.file_name()) {
        (Some(parent), Some(name)) => parent.join(name),
        _ => normal,
    }
}

/// Whether `path` lies inside the uploads directory (not merely next to it
/// under a name that starts the same).
pub fn is_within_uploads(config: &Config, path: &Path) -> bool {
    let root = resolve_path(&config.uploads_dir());
    let resolved = resolve_path(path);
    resolved != root && resolved.starts_with(&root)
}

/// `path` resolved, if it lies inside the uploads directory.
pub fn confine_to_uploads(config: &Config, path: &str) -> AppResult<PathBuf> {
    let path = Path::new(path);
    if is_within_uploads(config, path) {
        Ok(resolve_path(path))
    } else {
        Err(AppError::bad_request(UPLOADS_CONTAINMENT_ERROR))
    }
}

/// `None` without a file, `false` when the file is outside the uploads
/// directory or gone.
fn file_available(config: &Config, source: &Source) -> Option<bool> {
    let path = source.asset.as_ref()?.file_path.as_deref().filter(|p| !p.is_empty())?;
    let path = Path::new(path);
    Some(is_within_uploads(config, path) && resolve_path(path).is_file())
}

/// Store an upload under its own name in the uploads directory, adding
/// ` (1)`, ` (2)`, ... when the name is taken. The name is claimed atomically
/// (create-new), so concurrent uploads of the same name never overwrite each
/// other. Returns the stored file's absolute path.
pub async fn save_uploaded_file(config: &Config, filename: &str, data: &[u8]) -> AppResult<PathBuf> {
    let name = upload_file_name(filename);
    if name.is_empty() || name == "." || name == ".." || name.contains('\0') {
        return Err(AppError::invalid("Invalid filename"));
    }
    let folder = config.uploads_dir();
    tokio::fs::create_dir_all(&folder).await?;
    let root = resolve_path(&folder);
    let as_path = Path::new(&name);
    let stem = as_path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| name.clone());
    let suffix = as_path.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();

    let mut counter = 0u32;
    let (path, mut file) = loop {
        let candidate = if counter == 0 { name.clone() } else { format!("{stem} ({counter}){suffix}") };
        let path = root.join(&candidate);
        if path.parent() != Some(root.as_path()) {
            return Err(AppError::invalid("Invalid filename: path traversal detected"));
        }
        match tokio::fs::OpenOptions::new().write(true).create_new(true).open(&path).await {
            Ok(file) => break (path, file),
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => counter += 1,
            Err(e) => return Err(e.into()),
        }
    };
    use tokio::io::AsyncWriteExt;
    let written = async {
        file.write_all(data).await?;
        file.flush().await
    }
    .await;
    if let Err(e) = written {
        tracing::error!("Failed to save uploaded file: {e}");
        let _ = tokio::fs::remove_file(&path).await;
        return Err(e.into());
    }
    tracing::info!("Saved uploaded file to: {}", path.display());
    Ok(path)
}

/// Reject an upload extraction cannot handle (415). A failed check (for
/// example a file gone on retry) is not a verdict: extraction reports it.
pub async fn assert_file_supported(path: &Path) -> AppResult<()> {
    match nn_content::check_file_support(path, None, None).await {
        Ok(support) if !support.supported => {
            let mut detail = support.reason.unwrap_or_else(|| "Unsupported file type".to_string());
            if !support.identified_type.is_empty() {
                detail = format!("{detail} (detected type: {})", support.identified_type);
            }
            Err(AppError::UnsupportedType(detail))
        }
        Ok(_) => Ok(()),
        Err(e) => {
            tracing::debug!("Pre-flight file-support check skipped for {}: {e}", path.display());
            Ok(())
        }
    }
}

/// An uploaded original, ready to be served.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceFile {
    pub path: PathBuf,
    pub filename: String,
}

/// The uploaded original of a source. Only files inside the uploads
/// directory are ever served (403 otherwise).
pub async fn source_file(state: &AppState, id: &str) -> AppResult<SourceFile> {
    let source = load_source(state, id).await?;
    let path = source
        .asset
        .as_ref()
        .and_then(|a| a.file_path.as_deref())
        .filter(|p| !p.is_empty())
        .ok_or_else(|| AppError::not_found("Source has no file to download"))?;
    if !is_within_uploads(&state.config, Path::new(path)) {
        tracing::warn!("Blocked download outside uploads directory for source {id}: {path}");
        return Err(AppError::http(StatusCode::FORBIDDEN, "Access to file denied"));
    }
    let resolved = resolve_path(Path::new(path));
    if !resolved.is_file() {
        return Err(AppError::not_found("File not found on server"));
    }
    let filename = resolved.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    Ok(SourceFile { path: resolved, filename })
}

// ---------------------------------------------------------------------------
// Creating sources
// ---------------------------------------------------------------------------

/// What to extract, as handed to the `process_source` job (`content_state`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContentState {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub url: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub file_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    /// Delete the uploaded file once its text is extracted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub delete_source: Option<bool>,
}

/// A new source (`SourceCreate`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NewSource {
    /// Deprecated single-notebook form of `notebooks`.
    #[serde(default)]
    pub notebook_id: Option<String>,
    #[serde(default)]
    pub notebooks: Option<Vec<String>>,
    /// `link`, `upload` or `text`.
    #[serde(rename = "type")]
    pub source_type: String,
    #[serde(default)]
    pub url: Option<String>,
    /// A file already in the uploads directory (instead of an upload).
    #[serde(default)]
    pub file_path: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    /// Transformations to run once the content is extracted.
    #[serde(default)]
    pub transformations: Option<Vec<String>>,
    /// Embed the content for vector search.
    #[serde(default)]
    pub embed: bool,
    /// Delete the uploaded file after processing.
    #[serde(default)]
    pub delete_source: bool,
    /// Queue processing and answer at once (otherwise wait for it).
    #[serde(default)]
    pub async_processing: bool,
}

impl NewSource {
    /// The checks `SourceCreate` makes, as Pydantic-style messages. On
    /// success `notebooks` holds the full list (`notebook_id` folded in).
    pub fn validate(&mut self) -> Result<(), Vec<String>> {
        let mut problems = Vec::new();
        for (field, list) in [("notebooks", &self.notebooks), ("transformations", &self.transformations)] {
            if let Some(list) = list.as_ref().filter(|l| l.len() > MAX_SOURCE_LIST_ITEMS) {
                problems.push(format!(
                    "{field}: List should have at most {MAX_SOURCE_LIST_ITEMS} items after validation, not {}",
                    list.len()
                ));
            }
        }
        if !problems.is_empty() {
            return Err(problems);
        }
        if self.notebook_id.is_some() && self.notebooks.is_some() {
            return Err(vec![
                "Value error, Cannot specify both 'notebook_id' and 'notebooks'. Use 'notebooks' for multi-notebook support."
                    .to_string(),
            ]);
        }
        if let Some(notebook_id) = &self.notebook_id {
            self.notebooks = Some(vec![notebook_id.clone()]);
        }
        self.notebooks.get_or_insert_with(Vec::new);
        self.transformations.get_or_insert_with(Vec::new);
        Ok(())
    }
}

/// A file sent with `POST /sources`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceUpload {
    pub filename: String,
    pub data: Vec<u8>,
}

/// Check the type-specific input and build the job's `content_state`. The
/// SSRF and uploads-directory guards run here (and again in the job).
async fn build_content_state(state: &AppState, new: &NewSource, uploaded: Option<&Path>) -> AppResult<ContentState> {
    match new.source_type.as_str() {
        "link" => {
            let url = new.url.clone().filter(|u| !u.is_empty()).ok_or_else(|| AppError::bad_request("URL is required for link type"))?;
            nn_content::validate_url(&url).await.map_err(|e| AppError::bad_request(e.to_string()))?;
            Ok(ContentState { url: Some(url), ..ContentState::default() })
        }
        "upload" => {
            let path = match uploaded {
                Some(path) => path.to_string_lossy().into_owned(),
                None => new.file_path.clone().filter(|p| !p.is_empty()).ok_or_else(|| {
                    AppError::bad_request("File upload or file_path is required for upload type")
                })?,
            };
            let resolved = confine_to_uploads(&state.config, &path)?;
            assert_file_supported(&resolved).await?;
            Ok(ContentState { file_path: Some(path), delete_source: Some(new.delete_source), ..ContentState::default() })
        }
        "text" => {
            let content =
                new.content.clone().filter(|c| !c.is_empty()).ok_or_else(|| AppError::bad_request("Content is required for text type"))?;
            Ok(ContentState { content: Some(content), ..ContentState::default() })
        }
        _ => Err(AppError::bad_request("Invalid source type. Must be link, upload, or text")),
    }
}

/// Add a source. With `async_processing` the source is created and its
/// `process_source` job queued, and the answer comes at once (status `new`);
/// otherwise the call waits for processing (up to five minutes) and returns
/// the processed source. An upload that does not make it into a source is
/// removed again.
pub async fn create_source(state: &AppState, mut new: NewSource, upload: Option<SourceUpload>) -> AppResult<SourceView> {
    new.validate().map_err(|problems| AppError::unprocessable(format!("Invalid source data: {}", problems.join("; "))))?;
    for notebook_id in new.notebooks.iter().flatten() {
        let missing = format!("Notebook {notebook_id} not found");
        ensure_table(notebook_id, Notebook::TABLE, &missing)?;
        Notebook::get(state, notebook_id).await.map_err(|e| if e.is_not_found() { AppError::not_found(missing.clone()) } else { e })?;
    }

    let mut uploaded: Option<PathBuf> = None;
    if let Some(upload) = upload.as_ref().filter(|_| new.source_type == "upload") {
        match save_uploaded_file(&state.config, &upload.filename, &upload.data).await {
            Ok(path) => uploaded = Some(path),
            Err(e) => {
                tracing::error!("File upload failed: {e}");
                return Err(AppError::bad_request("File upload failed"));
            }
        }
    }

    let outcome = create_from_input(state, &new, uploaded.as_deref()).await;
    if outcome.is_err() {
        if let Some(path) = &uploaded {
            let _ = tokio::fs::remove_file(path).await;
        }
    }
    outcome
}

async fn create_from_input(state: &AppState, new: &NewSource, uploaded: Option<&Path>) -> AppResult<SourceView> {
    let content_state = build_content_state(state, new, uploaded).await?;
    let transformations = new.transformations.clone().unwrap_or_default();
    for id in &transformations {
        let missing = format!("Transformation {id} not found");
        ensure_table(id, Transformation::TABLE, &missing)?;
        Transformation::get(state, id).await.map_err(|e| if e.is_not_found() { AppError::not_found(missing.clone()) } else { e })?;
    }
    let notebooks = new.notebooks.clone().unwrap_or_default();
    let args = |source_id: &str| {
        json!({
            "source_id": source_id,
            "content_state": content_state,
            "notebook_ids": notebooks,
            "transformations": transformations,
            "embed": new.embed,
        })
    };
    let title = new.title.clone().filter(|t| !t.is_empty()).unwrap_or_else(|| PLACEHOLDER_TITLE.to_string());

    if new.async_processing {
        // The asset is stored up front so a failed run can be retried.
        let asset = match new.source_type.as_str() {
            "link" => Some(Asset { url: content_state.url.clone(), file_path: None }),
            "upload" => Some(Asset { file_path: content_state.file_path.clone(), url: None }),
            _ => None,
        };
        let draft = Source { title: Some(title), asset, ..Source::default() };
        let mut source = retry_conflicts(|| Source::create(state, &draft)).await?;
        let queued = async {
            for notebook_id in &notebooks {
                retry_conflicts(|| source.add_to_notebook(state, notebook_id)).await?;
            }
            state.jobs.submit("process_source", args(&source.id)).await
        }
        .await;
        let command_id = match queued {
            Ok(command_id) => command_id,
            Err(e) => {
                if let Err(cleanup) = source.delete(state).await {
                    tracing::warn!("Failed to remove source {} after a failed submission: {cleanup}", source.id);
                }
                if e.status().is_client_error() || e.is_retryable() {
                    return Err(e);
                }
                tracing::error!("Failed to submit async processing command: {e}");
                return Err(AppError::http(StatusCode::INTERNAL_SERVER_ERROR, "Failed to queue processing"));
            }
        };
        tracing::info!("Submitted async processing command: {command_id}");
        // Only the link is written: the job may already have saved results
        // (and may be writing this record right now, hence the retry).
        retry_conflicts(|| state.db.update(&source.id, Doc::new().record("command", &command_id))).await?;
        source.command = Some(command_id.clone());
        return Ok(SourceView {
            asset: None,
            full_text: None,
            command_id: Some(command_id),
            status: Some(status::NEW.to_string()),
            processing_info: Some(json!({ "async": true, "queued": true })),
            ..SourceView::new(source, 0)
        });
    }

    let draft = Source { title: Some(title), ..Source::default() };
    let source = retry_conflicts(|| Source::create(state, &draft)).await?;
    for notebook_id in &notebooks {
        retry_conflicts(|| source.add_to_notebook(state, notebook_id)).await?;
    }
    let command_id = state.jobs.submit("process_source", args(&source.id)).await?;
    let record = match state.jobs.wait(&command_id, SYNC_PROCESSING_TIMEOUT).await {
        Ok(record) => record,
        Err(e) => {
            let _ = state.jobs.cancel(&command_id).await;
            tracing::error!("Sync processing failed: {e}");
            return discard_failed(state, &source, format!("Processing failed: {}", truncate_error(Some(e.to_string())).unwrap_or_default())).await;
        }
    };
    if record.status_str() != status::COMPLETED {
        tracing::error!("Sync processing failed: {:?}", record.error_message);
        let message = truncate_error(record.error_message).unwrap_or_else(|| "None".to_string());
        return discard_failed(state, &source, format!("Processing failed: {message}")).await;
    }
    let processed = Source::get(state, &source.id)
        .await
        .map_err(|_| AppError::http(StatusCode::INTERNAL_SERVER_ERROR, "Processed source not found"))?;
    let embedded_chunks = processed.embedded_chunks(state).await?;
    Ok(SourceView::new(processed, embedded_chunks))
}

/// Remove a source whose synchronous processing failed and answer 500.
async fn discard_failed<T>(state: &AppState, source: &Source, message: String) -> AppResult<T> {
    if let Err(cleanup) = source.delete(state).await {
        tracing::warn!("Failed to remove source {} after failed processing: {cleanup}", source.id);
    }
    Err(AppError::http(StatusCode::INTERNAL_SERVER_ERROR, message))
}

/// Queue processing again for a failed or stuck source, with the content it
/// was created from. Embeds, and runs no transformations.
pub async fn retry_source(state: &AppState, id: &str) -> AppResult<SourceView> {
    let mut source = load_source(state, id).await?;
    if source.command.is_some() {
        if let Some(current) = source.status(state).await {
            if matches!(current.as_str(), "running" | "queued" | "new") {
                return Err(AppError::bad_request(
                    "Source is already processing. Cannot retry while processing is active.",
                ));
            }
        }
    }
    let notebook_ids = source.notebook_ids(state).await?;
    if notebook_ids.is_empty() {
        return Err(AppError::bad_request("Source is not associated with any notebooks"));
    }

    let file_path = source.asset.as_ref().and_then(|a| a.file_path.clone()).filter(|p| !p.is_empty());
    let url = source.asset.as_ref().and_then(|a| a.url.clone()).filter(|u| !u.is_empty());
    let content_state = if let Some(file_path) = file_path {
        assert_file_supported(Path::new(&file_path)).await?;
        ContentState { file_path: Some(file_path), delete_source: Some(false), ..ContentState::default() }
    } else if let Some(url) = url {
        ContentState { url: Some(url), ..ContentState::default() }
    } else if let Some(text) = source.full_text.clone().filter(|t| !t.is_empty()) {
        ContentState { content: Some(text), ..ContentState::default() }
    } else {
        return Err(AppError::bad_request("Cannot determine source content for retry"));
    };

    let args = json!({
        "source_id": source.id,
        "content_state": content_state,
        "notebook_ids": notebook_ids,
        "transformations": [],
        "embed": true,
    });
    let command_id = match state.jobs.submit("process_source", args).await {
        Ok(command_id) => command_id,
        Err(e) => {
            tracing::error!("Failed to submit retry processing command for source {id}: {e}");
            return Err(AppError::http(StatusCode::INTERNAL_SERVER_ERROR, "Failed to queue retry processing"));
        }
    };
    tracing::info!("Submitted retry processing command: {command_id} for source {id}");
    state.db.update(&source.id, Doc::new().record("command", &command_id)).await?;
    source.command = Some(command_id.clone());
    let embedded_chunks = source.embedded_chunks(state).await?;
    Ok(SourceView {
        command_id: Some(command_id),
        status: Some("queued".to_string()),
        processing_info: Some(json!({ "retry": true, "queued": true })),
        ..SourceView::new(source, embedded_chunks)
    })
}

// ---------------------------------------------------------------------------
// Extraction
// ---------------------------------------------------------------------------

/// The configured engine, or `auto` when its runtime is not available here
/// (the choice lives in the database and can outlive the runtime).
fn usable_engine(engine: Option<&str>, kind: &str) -> Option<String> {
    let engine = engine?.trim().to_string();
    match super::settings::engine_runtime_missing(Some(&engine)) {
        None => Some(engine),
        Some(env_var) => {
            tracing::warn!(
                "Configured {kind} engine '{engine}' is selected in Content Settings but its runtime is not available in \
                 this build; falling back to 'auto'. Set {env_var}=true to enable it."
            );
            Some("auto".to_string())
        }
    }
}

fn env_number<T: std::str::FromStr>(name: &str) -> Option<T> {
    env_nonempty(name).and_then(|v| v.trim().parse().ok())
}

/// Everything extraction needs: engine choices from the content settings,
/// engine credentials from the environment (the variables the Python build
/// and content-core read), the shared HTTP client and the default
/// speech-to-text model.
pub async fn extraction_context(state: &AppState) -> nn_content::ExtractContext {
    let settings = match ContentSettings::load(state).await {
        Ok(settings) => settings,
        Err(e) => {
            tracing::warn!("Failed to load content settings, using defaults: {e}");
            ContentSettings::default()
        }
    };
    let mut ctx = nn_content::ExtractContext { client: state.http.clone(), ..nn_content::ExtractContext::default() };

    let url_engine = usable_engine(settings.default_content_processing_engine_url.as_deref(), "url");
    if let Some(engine) = url_engine.as_deref() {
        ctx.url_engine = engine.parse().unwrap_or_else(|_| {
            tracing::warn!("Unknown URL engine '{engine}' in Content Settings; using 'auto'");
            nn_content::UrlEngine::Auto
        });
    }
    // Documents have one engine here; a stored "docling" falls back to it.
    let document_engine = usable_engine(settings.default_content_processing_engine_doc.as_deref(), "document");
    if let Some(languages) = settings.youtube_preferred_languages.filter(|l| !l.is_empty()) {
        ctx.youtube_languages = languages;
    }

    ctx.jina_api_key = env_nonempty("JINA_API_KEY");
    ctx.firecrawl_api_key = env_nonempty("FIRECRAWL_API_KEY");
    if let Some(url) = env_nonempty("FIRECRAWL_API_URL") {
        ctx.firecrawl_api_url = url;
    }
    if let Some(proxy) = env_nonempty("CCORE_FIRECRAWL_PROXY") {
        ctx.firecrawl_proxy = proxy;
    }
    if let Some(wait) = env_number::<u64>("CCORE_FIRECRAWL_WAIT_FOR") {
        ctx.firecrawl_wait_for_ms = wait;
    }
    ctx.crawl4ai_api_url = env_nonempty("CRAWL4AI_API_URL");
    ctx.crawl4ai_api_token = env_nonempty("CRAWL4AI_API_TOKEN");
    ctx.youtube_proxy = env_nonempty("CCORE_YOUTUBE_PROXY");
    ctx.youtube_cookies_file = env_nonempty("CCORE_YOUTUBE_COOKIES_FILE").map(PathBuf::from);
    if let Some(minutes) = env_number::<f64>("CCORE_AUDIO_SEGMENT_MINUTES").filter(|m| *m >= 0.0) {
        ctx.audio.segment_seconds = (minutes * 60.0) as u64;
    }
    if let Some(concurrency) = env_number::<usize>("CCORE_AUDIO_CONCURRENCY") {
        ctx.audio.concurrency = concurrency.clamp(1, 10);
    }
    if let Some(seconds) = env_number::<f64>("CCORE_STT_TIMEOUT").filter(|s| *s > 0.0) {
        ctx.limits.stt_timeout = Duration::from_secs_f64(seconds);
    }

    ctx.speech_to_text = match state.models.speech_to_text_opt(state).await {
        Ok(model) => model,
        Err(e) => {
            tracing::warn!("Failed to retrieve speech-to-text model configuration: {e}");
            None
        }
    };
    tracing::info!(
        "Extracting via nn-content (url_engine={}, document_engine={})",
        ctx.url_engine,
        document_engine.as_deref().unwrap_or("auto")
    );
    ctx
}

/// The input guards, applied where the file is read and the URL fetched
/// (retries and sync processing reach the job without the API's checks).
pub async fn validate_content_state(config: &Config, content_state: &ContentState) -> AppResult<()> {
    if let Some(path) = content_state.file_path.as_deref().filter(|p| !p.is_empty()) {
        if !is_within_uploads(config, Path::new(path)) {
            return Err(AppError::invalid(UPLOADS_CONTAINMENT_ERROR));
        }
    }
    if let Some(url) = content_state.url.as_deref().filter(|u| !u.is_empty()) {
        nn_content::validate_url(url).await.map_err(|e| AppError::invalid(e.to_string()))?;
    }
    Ok(())
}

/// Extract the content a `content_state` names. Every extraction failure is
/// permanent (`InvalidInput`) with a fixed, user-facing message; the detail
/// goes to the log only.
pub async fn extract_source_content(state: &AppState, content_state: &ContentState) -> AppResult<nn_content::Extracted> {
    validate_content_state(&state.config, content_state).await?;
    let url = content_state.url.clone().filter(|u| !u.is_empty());
    let input = if let Some(url) = &url {
        nn_content::ExtractInput::url(url)
    } else if let Some(path) = content_state.file_path.as_deref().filter(|p| !p.is_empty()) {
        nn_content::ExtractInput::file(resolve_path(Path::new(path)))
    } else if let Some(content) = &content_state.content {
        nn_content::ExtractInput::text(content)
    } else {
        return Err(AppError::invalid("This URL or input is not valid."));
    };
    let ctx = extraction_context(state).await;
    let extracted = nn_content::extract_content(input, &ctx).await.map_err(|e| {
        tracing::warn!("Content extraction failed: {e}");
        AppError::invalid(e.user_message(url.as_deref()))
    })?;

    if content_state.delete_source == Some(true) {
        if let Some(path) = content_state.file_path.as_deref() {
            match tokio::fs::remove_file(path).await {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    tracing::warn!("File not found while trying to delete: {path}")
                }
                Err(e) => tracing::warn!("Failed to delete source file {path}: {e}"),
            }
        }
    }
    Ok(extracted)
}

// ---------------------------------------------------------------------------
// The process_source job
// ---------------------------------------------------------------------------

/// Arguments of the `process_source` job (`SourceProcessingInput`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceProcessingArgs {
    pub source_id: String,
    pub content_state: ContentState,
    #[serde(default)]
    pub notebook_ids: Vec<String>,
    #[serde(default)]
    pub transformations: Vec<String>,
    #[serde(default)]
    pub embed: bool,
}

async fn process_source_job(ctx: JobContext, args: Value) -> AppResult<Value> {
    let started = Instant::now();
    let args: SourceProcessingArgs =
        serde_json::from_value(args).map_err(|e| AppError::invalid(format!("Invalid process_source input: {e}")))?;
    tracing::info!(
        "Starting source processing for source: {} (notebooks: {:?}, transformations: {:?}, embed: {})",
        args.source_id,
        args.notebook_ids,
        args.transformations,
        args.embed
    );
    match process_source(&ctx, &args).await {
        Ok(insights_created) => {
            let processing_time = started.elapsed().as_secs_f64();
            tracing::info!("Successfully processed source: {} in {processing_time:.2}s", args.source_id);
            Ok(json!({
                "success": true,
                "source_id": args.source_id,
                "embedded_chunks": 0,
                "insights_created": insights_created,
                "processing_time": processing_time,
                "error_message": null,
            }))
        }
        Err(e) => {
            if e.is_retryable() {
                tracing::debug!("Transient error processing source {} (attempt {}): {e}", args.source_id, ctx.attempt);
            } else {
                tracing::error!("Source processing failed (permanent) for source {}: {e}", args.source_id);
            }
            Err(e)
        }
    }
}

/// Load a record for the job: a missing one means the job can never succeed.
async fn load_for_job<T, F>(load: F, gone: String) -> AppResult<T>
where
    F: std::future::Future<Output = AppResult<T>>,
{
    load.await.map_err(|e| if e.is_not_found() { AppError::invalid(gone) } else { e })
}

/// Returns the number of insights the source has afterwards.
async fn process_source(ctx: &JobContext, args: &SourceProcessingArgs) -> AppResult<usize> {
    let state = &ctx.state;
    let mut transformations = Vec::with_capacity(args.transformations.len());
    for id in &args.transformations {
        transformations.push(load_for_job(Transformation::get(state, id), format!("Transformation '{id}' no longer exists")).await?);
    }
    let source = load_for_job(
        Source::get(state, &args.source_id),
        format!("Source '{}' no longer exists (deleted before processing?)", args.source_id),
    )
    .await?;
    state.db.update(&source.id, Doc::new().record("command", &ctx.command_id)).await?;

    let extracted = extract_source_content(state, &args.content_state).await?;

    // Re-read: the record may have changed (or gone) while extraction ran.
    let mut source = Source::get(state, &args.source_id).await?;
    let url = args.content_state.url.clone().filter(|u| !u.is_empty());
    let file_path = args.content_state.file_path.clone().filter(|p| !p.is_empty());
    // As in Python, the asset always records what was extracted (both
    // fields empty for text).
    source.asset = Some(Asset { url, file_path });
    source.full_text = Some(extracted.content);
    if let Some(title) = extracted.title.filter(|t| !t.trim().is_empty()) {
        // A title the user gave is kept.
        if source.title.as_deref().is_none_or(|t| t.is_empty() || t == PLACEHOLDER_TITLE) {
            source.title = Some(title);
        }
    }
    source.save(state).await?;

    let has_text = source.full_text.as_deref().is_some_and(|t| !t.trim().is_empty());
    if args.embed {
        if has_text {
            source.vectorize(state).await?;
        } else {
            tracing::warn!("Source {} has no text content to embed, skipping vectorization", source.id);
        }
    }

    if has_text && !transformations.is_empty() {
        let runs = transformations
            .iter()
            .map(|t| apply_transformation(state, t, source.full_text.as_deref(), Some(&source), t.model_id.as_deref()));
        futures::future::try_join_all(runs).await?;
    }

    Ok(source.insights(state).await?.len())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn uploads_get_unique_names_and_stay_inside_the_folder() {
        let (state, _dir) = AppState::for_tests().await.unwrap();
        let config = &state.config;
        let first = save_uploaded_file(config, "doc.txt", b"first").await.unwrap();
        let second = save_uploaded_file(config, "doc.txt", b"second").await.unwrap();
        let third = save_uploaded_file(config, "../../etc/doc.txt", b"third").await.unwrap();
        let names: Vec<String> = [&first, &second, &third].iter().map(|p| p.file_name().unwrap().to_string_lossy().into_owned()).collect();
        assert_eq!(names, ["doc.txt", "doc (1).txt", "doc (2).txt"]);
        assert_eq!(std::fs::read(&second).unwrap(), b"second");
        for path in [&first, &second, &third] {
            assert!(is_within_uploads(config, path));
        }
        for bad in ["", "..", "dir/.."] {
            assert!(save_uploaded_file(config, bad, b"x").await.is_err(), "{bad:?}");
        }

        // Concurrent uploads of one name all survive.
        let writes = (0..8).map(|i| {
            let config = config.clone();
            async move { save_uploaded_file(&config, "report.pdf", format!("content-{i}").as_bytes()).await.unwrap() }
        });
        let paths = futures::future::join_all(writes).await;
        let contents: std::collections::HashSet<Vec<u8>> = paths.iter().map(|p| std::fs::read(p).unwrap()).collect();
        assert_eq!(contents.len(), 8);
    }

    #[tokio::test]
    async fn paths_are_confined_to_the_uploads_directory() {
        let (state, dir) = AppState::for_tests().await.unwrap();
        let config = &state.config;
        let uploads = config.uploads_dir();
        let sibling = dir.path().join("uploads_evil");
        std::fs::create_dir_all(&sibling).unwrap();
        std::fs::write(sibling.join("secret.txt"), b"no").unwrap();

        assert!(is_within_uploads(config, &uploads.join("doc.pdf")), "a missing file inside is still inside");
        for outside in [
            PathBuf::from("/etc/passwd"),
            uploads.join("..").join("secret.txt"),
            sibling.join("secret.txt"),
            uploads.clone(),
        ] {
            assert!(!is_within_uploads(config, &outside), "{}", outside.display());
            let error = confine_to_uploads(config, &outside.to_string_lossy()).unwrap_err();
            assert_eq!(error.detail(), UPLOADS_CONTAINMENT_ERROR);
        }
        let state_ok = ContentState { content: Some("hello".into()), ..ContentState::default() };
        validate_content_state(config, &state_ok).await.unwrap();
        let link_local = ContentState { url: Some("http://169.254.169.254/latest/meta-data/".into()), ..ContentState::default() };
        let error = validate_content_state(config, &link_local).await.unwrap_err();
        assert!(error.detail().contains("Link-local") || error.detail().to_lowercase().contains("link-local"), "{error}");
    }

    #[test]
    fn new_sources_validate_like_source_create() {
        let ids = |n: usize, prefix: &str| (0..n).map(|i| format!("{prefix}:{i}")).collect::<Vec<_>>();
        let mut ok = NewSource { source_type: "text".into(), notebooks: Some(ids(50, "notebook")), ..NewSource::default() };
        ok.validate().unwrap();
        assert_eq!(ok.transformations, Some(vec![]));
        let mut too_many = NewSource { source_type: "text".into(), notebooks: Some(ids(51, "notebook")), ..NewSource::default() };
        assert!(too_many.validate().unwrap_err()[0].contains("at most 50 items"));
        let mut too_many = NewSource { source_type: "text".into(), transformations: Some(ids(51, "t")), ..NewSource::default() };
        assert!(too_many.validate().is_err());
        let mut both = NewSource {
            source_type: "text".into(),
            notebook_id: Some("notebook:1".into()),
            notebooks: Some(vec![]),
            ..NewSource::default()
        };
        assert!(both.validate().unwrap_err()[0].contains("Cannot specify both"));
        let mut single = NewSource { source_type: "text".into(), notebook_id: Some("notebook:1".into()), ..NewSource::default() };
        single.validate().unwrap();
        assert_eq!(single.notebooks, Some(vec!["notebook:1".to_string()]));
        let mut none = NewSource { source_type: "text".into(), ..NewSource::default() };
        none.validate().unwrap();
        assert_eq!(none.notebooks, Some(vec![]));
    }

    #[test]
    fn long_errors_are_truncated() {
        assert_eq!(truncate_error(None), None);
        assert_eq!(truncate_error(Some("short".into())).as_deref(), Some("short"));
        let long = truncate_error(Some("x".repeat(300))).unwrap();
        assert_eq!(long.chars().count(), 201);
        assert!(long.ends_with('…'));
    }
}
