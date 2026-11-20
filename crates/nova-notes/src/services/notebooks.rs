//! Notebooks, and the folders and sections that organise them
//! (section > folder > notebook).
//!
//! The Rust counterpart of `api/routers/notebooks.py`, `folders.py` and
//! `sections.py`. Error details are the Python routers' strings.

use chrono::{DateTime, SecondsFormat, Utc};
use serde::{Deserialize, Deserializer, Serialize};

use crate::db::{table_of, validate_order_by, Doc, Json, Params};
use crate::domain::{null_default, Folder, Notebook, Section, Source};
use crate::error::{AppError, AppResult};
use crate::jobs::JobQueue;
use crate::state::AppState;

/// Notebooks, folders and sections have no background jobs.
pub fn register_jobs(_jobs: &JobQueue) {}

// ---------------------------------------------------------------------------
// Helpers shared with the other organisational services
// ---------------------------------------------------------------------------

/// A record timestamp as the API reports it (RFC 3339, `""` when unset).
pub(crate) fn timestamp(value: Option<DateTime<Utc>>) -> String {
    value.map(|d| d.to_rfc3339_opts(SecondsFormat::AutoSi, true)).unwrap_or_default()
}

/// Deserialize a field that distinguishes "absent" (`None`, with
/// `#[serde(default)]`), "explicit null" (`Some(None)`) and a value.
pub(crate) fn double_option<'de, D, T>(deserializer: D) -> Result<Option<Option<T>>, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de>,
{
    Option::<T>::deserialize(deserializer).map(Some)
}

/// Refuse an id that belongs to another table before it is used to load,
/// change or delete a record: `missing` becomes the 404 detail.
pub(crate) fn ensure_table(id: &str, table: &str, missing: &str) -> AppResult<()> {
    match table_of(id) {
        Some(found) if found == table => Ok(()),
        Some(_) => Err(AppError::not_found(missing)),
        None => Err(AppError::invalid(format!("Invalid record id: {id:?}"))),
    }
}

/// Replace a "not found" error's message with the router's detail string.
pub(crate) fn not_found_as(detail: &'static str) -> impl Fn(AppError) -> AppError {
    move |e| if e.is_not_found() { AppError::not_found(detail) } else { e }
}

async fn load_notebook(state: &AppState, id: &str, missing: &'static str) -> AppResult<Notebook> {
    ensure_table(id, Notebook::TABLE, missing)?;
    Notebook::get(state, id).await.map_err(not_found_as(missing))
}

// ---------------------------------------------------------------------------
// Notebooks
// ---------------------------------------------------------------------------

const NOTEBOOK_NOT_FOUND: &str = "Notebook not found";

/// Fields a notebook list may be sorted by.
pub const NOTEBOOK_SORT_FIELDS: [&str; 3] = ["created", "name", "updated"];

const SELECT_WITH_COUNTS: &str =
    "SELECT *, count(<-reference.in) AS source_count, count(<-artifact[WHERE in.untouched_draft != true].in) AS note_count";

/// A notebook with its source and note counts (`NotebookResponse`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotebookSummary {
    pub id: String,
    pub name: String,
    pub description: String,
    pub archived: bool,
    pub created: String,
    pub updated: String,
    pub source_count: i64,
    pub note_count: i64,
    pub folder_id: Option<String>,
}

#[derive(Deserialize)]
struct NotebookRow {
    #[serde(flatten)]
    notebook: Notebook,
    #[serde(default, deserialize_with = "null_default")]
    source_count: i64,
    #[serde(default, deserialize_with = "null_default")]
    note_count: i64,
}

impl From<NotebookRow> for NotebookSummary {
    fn from(row: NotebookRow) -> Self {
        let nb = row.notebook;
        Self {
            id: nb.id,
            name: nb.name,
            description: nb.description,
            archived: nb.archived,
            created: timestamp(nb.created),
            updated: timestamp(nb.updated),
            source_count: row.source_count,
            note_count: row.note_count,
            folder_id: nb.folder,
        }
    }
}

fn notebook_summaries(rows: Vec<Json>) -> Vec<NotebookSummary> {
    rows.into_iter()
        .filter_map(|row| match serde_json::from_value::<NotebookRow>(row) {
            Ok(row) => Some(row.into()),
            Err(e) => {
                tracing::error!("Skipping unreadable notebook row: {e}");
                None
            }
        })
        .collect()
}

/// Filters and ordering for [`list_notebooks`].
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotebookQuery {
    /// Only archived (`true`) or only active (`false`) notebooks.
    #[serde(default)]
    pub archived: Option<bool>,
    /// `<field> [asc|desc]` on one of [`NOTEBOOK_SORT_FIELDS`]. Default `updated desc`.
    #[serde(default)]
    pub order_by: Option<String>,
    /// A folder id, or `"none"` for notebooks outside any folder.
    #[serde(default)]
    pub folder_id: Option<String>,
}

/// A single clause on one of the sortable notebook fields, normalised.
fn notebook_order(order_by: &str) -> AppResult<String> {
    let invalid = || {
        AppError::invalid(format!(
            "Invalid order_by: '{order_by}'. Allowed fields: {}. Allowed directions: asc, desc",
            NOTEBOOK_SORT_FIELDS.join(", ")
        ))
    };
    let validated = validate_order_by(order_by).map_err(|_| invalid())?;
    let field = validated.split_whitespace().next().unwrap_or_default();
    if validated.contains(',') || !NOTEBOOK_SORT_FIELDS.contains(&field) {
        return Err(invalid());
    }
    Ok(validated)
}

/// Every notebook with its counts, filtered and ordered.
pub async fn list_notebooks(state: &AppState, query: &NotebookQuery) -> AppResult<Vec<NotebookSummary>> {
    let order = notebook_order(query.order_by.as_deref().unwrap_or("updated desc"))?;
    let rows = state.db.query(&format!("{SELECT_WITH_COUNTS} FROM notebook ORDER BY {order}"), Params::new()).await?;
    let mut notebooks = notebook_summaries(rows);
    if let Some(archived) = query.archived {
        notebooks.retain(|nb| nb.archived == archived);
    }
    if let Some(folder_id) = query.folder_id.as_deref() {
        let wanted = (folder_id != "none").then_some(folder_id);
        notebooks.retain(|nb| nb.folder_id.as_deref() == wanted);
    }
    Ok(notebooks)
}

/// Body of `POST /notebooks`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NewNotebook {
    pub name: String,
    #[serde(default)]
    pub description: String,
}

pub async fn create_notebook(state: &AppState, new: &NewNotebook) -> AppResult<NotebookSummary> {
    let notebook = Notebook::create(state, &new.name, &new.description, None).await?;
    Ok(NotebookRow { notebook, source_count: 0, note_count: 0 }.into())
}

/// One notebook with its counts. Does not count as a view; see [`view_notebook`].
pub async fn get_notebook(state: &AppState, notebook_id: &str) -> AppResult<NotebookSummary> {
    ensure_table(notebook_id, Notebook::TABLE, NOTEBOOK_NOT_FOUND)?;
    let rows = state
        .db
        .query(&format!("{SELECT_WITH_COUNTS} FROM $notebook_id"), Params::new().record("notebook_id", notebook_id))
        .await?;
    notebook_summaries(rows).into_iter().next().ok_or_else(|| AppError::not_found(NOTEBOOK_NOT_FOUND))
}

/// One notebook with its counts, recording that it was just opened (which
/// feeds [`recently_viewed`]). A failed stamp never fails the read.
pub async fn view_notebook(state: &AppState, notebook_id: &str) -> AppResult<NotebookSummary> {
    let notebook = get_notebook(state, notebook_id).await?;
    if let Err(e) = Notebook::touch_viewed(state, notebook_id).await {
        tracing::warn!("Failed to stamp last_viewed_at for notebook {notebook_id}: {e}");
    }
    Ok(notebook)
}

/// Body of `PUT /notebooks/{id}`. Absent fields are left alone.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotebookUpdate {
    #[serde(default)]
    pub name: Option<String>,
    /// `Some(None)` (JSON `null`) and `""` both clear the description.
    #[serde(default, deserialize_with = "double_option", skip_serializing_if = "Option::is_none")]
    pub description: Option<Option<String>>,
    #[serde(default)]
    pub archived: Option<bool>,
    /// `Some(None)` (JSON `null`) and `""` take the notebook out of its
    /// folder; an id moves it there.
    #[serde(default, deserialize_with = "double_option", skip_serializing_if = "Option::is_none")]
    pub folder_id: Option<Option<String>>,
}

pub async fn update_notebook(state: &AppState, notebook_id: &str, update: &NotebookUpdate) -> AppResult<NotebookSummary> {
    let mut notebook = load_notebook(state, notebook_id, NOTEBOOK_NOT_FOUND).await?;

    if let Some(name) = &update.name {
        notebook.name = name.clone();
    }
    if let Some(description) = &update.description {
        notebook.description = description.clone().unwrap_or_default();
    }
    if let Some(archived) = update.archived {
        notebook.archived = archived;
    }
    // Resolve the target folder before anything is written.
    let folder_change = match &update.folder_id {
        None => None,
        Some(raw) => {
            let target = raw.as_deref().unwrap_or_default().trim();
            if target.is_empty() {
                Some(None)
            } else {
                // Any failure to resolve the id (unknown, malformed, another table) is a 404.
                let folder = load_folder(state, target).await.map_err(|e| match e {
                    AppError::InvalidInput(_) => AppError::not_found(FOLDER_NOT_FOUND),
                    other => other,
                })?;
                Some(Some(folder.id))
            }
        }
    };

    notebook.save(state).await.map_err(not_found_as(NOTEBOOK_NOT_FOUND))?;
    if let Some(folder) = folder_change {
        Notebook::set_folder(state, &notebook.id, folder.as_deref()).await.map_err(not_found_as(NOTEBOOK_NOT_FOUND))?;
    }
    get_notebook(state, &notebook.id).await
}

/// What deleting a notebook would remove (`NotebookDeletePreview`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotebookDeletePreview {
    pub notebook_id: String,
    pub notebook_name: String,
    pub note_count: i64,
    pub exclusive_source_count: i64,
    pub shared_source_count: i64,
}

pub async fn notebook_delete_preview(state: &AppState, notebook_id: &str) -> AppResult<NotebookDeletePreview> {
    let notebook = load_notebook(state, notebook_id, NOTEBOOK_NOT_FOUND).await?;
    let preview = notebook.delete_preview(state).await?;
    Ok(NotebookDeletePreview {
        notebook_id: notebook.id,
        notebook_name: notebook.name,
        note_count: preview.note_count,
        exclusive_source_count: preview.exclusive_source_count,
        shared_source_count: preview.shared_source_count,
    })
}

/// What deleting a notebook removed (`NotebookDeleteResponse`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotebookDeleted {
    pub message: String,
    pub deleted_notes: i64,
    pub deleted_sources: i64,
    pub unlinked_sources: i64,
    pub deleted_chat_sessions: i64,
}

/// Delete a notebook with its notes and chat sessions. Its sources are
/// unlinked; with `delete_exclusive_sources`, those in no other notebook are
/// deleted too.
pub async fn delete_notebook(state: &AppState, notebook_id: &str, delete_exclusive_sources: bool) -> AppResult<NotebookDeleted> {
    let notebook = load_notebook(state, notebook_id, NOTEBOOK_NOT_FOUND).await?;
    let summary = notebook.delete(state, delete_exclusive_sources).await?;
    Ok(NotebookDeleted {
        message: "Notebook deleted successfully".to_string(),
        deleted_notes: summary.deleted_notes,
        deleted_sources: summary.deleted_sources,
        unlinked_sources: summary.unlinked_sources,
        deleted_chat_sessions: summary.deleted_chat_sessions,
    })
}

/// Link an existing source to a notebook. Linking twice is a no-op.
pub async fn add_source_to_notebook(state: &AppState, notebook_id: &str, source_id: &str) -> AppResult<()> {
    const MISSING: &str = "Notebook or source not found";
    let notebook = load_notebook(state, notebook_id, MISSING).await?;
    ensure_table(source_id, Source::TABLE, MISSING)?;
    let source = Source::get(state, source_id).await.map_err(not_found_as(MISSING))?;

    let existing = state
        .db
        .query(
            "SELECT id FROM reference WHERE in = $source_id AND out = $notebook_id LIMIT 1",
            Params::new().record("source_id", &source.id).record("notebook_id", &notebook.id),
        )
        .await?;
    if existing.is_empty() {
        state.db.relate(&source.id, "reference", &notebook.id, Doc::new()).await?;
    }
    Ok(())
}

/// Unlink a source from a notebook. The source itself is kept.
pub async fn remove_source_from_notebook(state: &AppState, notebook_id: &str, source_id: &str) -> AppResult<()> {
    let notebook = load_notebook(state, notebook_id, NOTEBOOK_NOT_FOUND).await?;
    state
        .db
        .query(
            "DELETE FROM reference WHERE out = $notebook_id AND in = $source_id",
            Params::new().record("notebook_id", &notebook.id).record("source_id", source_id),
        )
        .await?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Recently viewed
// ---------------------------------------------------------------------------

/// A notebook or source that was opened recently (`RecentlyViewedResponse`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RecentlyViewed {
    /// `"notebook"` or `"source"`.
    #[serde(rename = "type")]
    pub kind: String,
    pub id: String,
    pub title: String,
    pub last_viewed_at: String,
}

#[derive(Deserialize)]
struct ViewedRow {
    #[serde(default, deserialize_with = "null_default")]
    id: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    last_viewed_at: Option<DateTime<Utc>>,
}

/// The most recently opened notebooks and sources, newest first.
pub async fn recently_viewed(state: &AppState, limit: usize) -> AppResult<Vec<RecentlyViewed>> {
    let limit = limit.max(1);
    let viewed = |table: &'static str, title: &'static str| async move {
        state
            .db
            .query_as::<ViewedRow>(
                &format!(
                    "SELECT id, {title} AS title, last_viewed_at FROM {table} \
                     WHERE last_viewed_at != NONE AND last_viewed_at != NULL \
                     ORDER BY last_viewed_at DESC LIMIT $limit"
                ),
                Params::new().set("limit", limit),
            )
            .await
    };
    let notebooks = viewed("notebook", "name").await?;
    let sources = viewed("source", "title").await?;

    let tagged = |kind: &'static str, untitled: &'static str, rows: Vec<ViewedRow>| {
        rows.into_iter().map(move |row| (kind, untitled, row))
    };
    let mut rows: Vec<_> = tagged("notebook", "Untitled notebook", notebooks)
        .chain(tagged("source", "Untitled source", sources))
        .collect();
    rows.sort_by_key(|row| std::cmp::Reverse(row.2.last_viewed_at));
    rows.truncate(limit);
    Ok(rows
        .into_iter()
        .map(|(kind, untitled, row)| RecentlyViewed {
            kind: kind.to_string(),
            id: row.id,
            title: row.title.filter(|t| !t.is_empty()).unwrap_or_else(|| untitled.to_string()),
            last_viewed_at: timestamp(row.last_viewed_at),
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Names shared by folders and sections
// ---------------------------------------------------------------------------

/// Longest folder or section name, in characters, after trimming.
pub const MAX_GROUP_NAME_LENGTH: usize = 80;

fn clean_name(kind: &str, raw: &str) -> AppResult<String> {
    let name = raw.trim();
    if name.is_empty() {
        return Err(AppError::invalid(format!("{kind} name cannot be empty")));
    }
    if name.chars().count() > MAX_GROUP_NAME_LENGTH {
        return Err(AppError::invalid(format!("{kind} name cannot be longer than {MAX_GROUP_NAME_LENGTH} characters")));
    }
    Ok(name.to_string())
}

fn same_name(a: &str, b: &str) -> bool {
    a.trim().to_lowercase() == b.trim().to_lowercase()
}

/// Case-insensitive name order with the id as tie-break.
fn by_name<T>(items: &mut [T], key: impl Fn(&T) -> (&str, &str)) {
    items.sort_by(|a, b| {
        let (a_name, a_id) = key(a);
        let (b_name, b_id) = key(b);
        (a_name.to_lowercase(), a_id).cmp(&(b_name.to_lowercase(), b_id))
    });
}

async fn count_where(state: &AppState, table: &str, field: &str, id: &str) -> AppResult<i64> {
    let rows = state
        .db
        .query(&format!("SELECT count() AS n FROM {table} WHERE {field} = $id GROUP ALL"), Params::new().record("id", id))
        .await?;
    Ok(rows.first().and_then(|row| row.get("n")).and_then(Json::as_i64).unwrap_or(0))
}

// ---------------------------------------------------------------------------
// Sections
// ---------------------------------------------------------------------------

const SECTION_NOT_FOUND: &str = "Section not found";

/// A section with its folder count (`SectionResponse`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectionSummary {
    pub id: String,
    pub name: String,
    pub created: String,
    pub updated: String,
    pub folder_count: i64,
}

impl SectionSummary {
    fn new(section: Section, folder_count: i64) -> Self {
        Self {
            id: section.id,
            name: section.name,
            created: timestamp(section.created),
            updated: timestamp(section.updated),
            folder_count,
        }
    }
}

#[derive(Deserialize)]
struct SectionRow {
    #[serde(flatten)]
    section: Section,
    #[serde(default, deserialize_with = "null_default")]
    folder_count: i64,
}

/// Body of `POST /sections` and `PUT /sections/{id}`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectionInput {
    /// 1-80 characters after trimming.
    pub name: String,
}

/// What deleting a section did (`SectionDeleteResponse`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SectionDeleted {
    pub message: String,
    pub folders_ungrouped: usize,
}

async fn load_section(state: &AppState, section_id: &str) -> AppResult<Section> {
    ensure_table(section_id, Section::TABLE, SECTION_NOT_FOUND)?;
    Section::get(state, section_id).await.map_err(not_found_as(SECTION_NOT_FOUND))
}

/// Section names are unique, ignoring case.
async fn ensure_section_name_free(state: &AppState, name: &str, exclude_id: Option<&str>) -> AppResult<()> {
    let taken = Section::list(state)
        .await?
        .iter()
        .any(|section| Some(section.id.as_str()) != exclude_id && same_name(&section.name, name));
    if taken {
        return Err(AppError::conflict(format!("A section named '{name}' already exists")));
    }
    Ok(())
}

/// Sections sorted by name, with their folder counts.
pub async fn list_sections(state: &AppState) -> AppResult<Vec<SectionSummary>> {
    let mut rows: Vec<SectionRow> = state
        .db
        .query_as(
            "SELECT *, count(SELECT id FROM folder WHERE section = $parent.id) AS folder_count FROM section",
            Params::new(),
        )
        .await?;
    by_name(&mut rows, |row| (&row.section.name, &row.section.id));
    Ok(rows.into_iter().map(|row| SectionSummary::new(row.section, row.folder_count)).collect())
}

pub async fn create_section(state: &AppState, input: &SectionInput) -> AppResult<SectionSummary> {
    let name = clean_name("Section", &input.name)?;
    ensure_section_name_free(state, &name, None).await?;
    Ok(SectionSummary::new(Section::create(state, &name).await?, 0))
}

pub async fn rename_section(state: &AppState, section_id: &str, input: &SectionInput) -> AppResult<SectionSummary> {
    let section = load_section(state, section_id).await?;
    let name = clean_name("Section", &input.name)?;
    ensure_section_name_free(state, &name, Some(&section.id)).await?;
    let renamed = Section::rename(state, &section.id, &name).await.map_err(not_found_as(SECTION_NOT_FOUND))?;
    let folder_count = count_where(state, Folder::TABLE, "section", &section.id).await?;
    Ok(SectionSummary::new(renamed, folder_count))
}

/// Delete a section. Its folders are kept and become unsectioned.
pub async fn delete_section(state: &AppState, section_id: &str) -> AppResult<SectionDeleted> {
    let section = load_section(state, section_id).await?;
    let ungrouped = state
        .db
        .query(
            "UPDATE folder SET section = NONE WHERE section = $section RETURN id",
            Params::new().record("section", &section.id),
        )
        .await?;
    state.db.delete(&section.id).await?;
    Ok(SectionDeleted { message: "Section deleted successfully".to_string(), folders_ungrouped: ungrouped.len() })
}

// ---------------------------------------------------------------------------
// Folders
// ---------------------------------------------------------------------------

const FOLDER_NOT_FOUND: &str = "Folder not found";

/// A folder with its notebook count (`FolderResponse`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FolderSummary {
    pub id: String,
    pub name: String,
    pub created: String,
    pub updated: String,
    pub notebook_count: i64,
    pub section_id: Option<String>,
}

impl FolderSummary {
    fn new(folder: Folder, notebook_count: i64) -> Self {
        Self {
            id: folder.id,
            name: folder.name,
            created: timestamp(folder.created),
            updated: timestamp(folder.updated),
            notebook_count,
            section_id: folder.section,
        }
    }
}

#[derive(Deserialize)]
struct FolderRow {
    #[serde(flatten)]
    folder: Folder,
    #[serde(default, deserialize_with = "null_default")]
    notebook_count: i64,
}

/// Body of `POST /folders` and `PUT /folders/{id}`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FolderInput {
    /// 1-80 characters after trimming.
    pub name: String,
    /// Absent leaves the section alone (on create: no section); JSON `null`
    /// or `""` takes the folder out of its section; an id places it there.
    #[serde(default, deserialize_with = "double_option", skip_serializing_if = "Option::is_none")]
    pub section_id: Option<Option<String>>,
}

/// What deleting a folder did (`FolderDeleteResponse`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FolderDeleted {
    pub message: String,
    pub notebooks_ungrouped: usize,
}

async fn load_folder(state: &AppState, folder_id: &str) -> AppResult<Folder> {
    ensure_table(folder_id, Folder::TABLE, FOLDER_NOT_FOUND)?;
    Folder::get(state, folder_id).await.map_err(not_found_as(FOLDER_NOT_FOUND))
}

/// Blank means no section; an id that does not resolve to a section is a 404.
async fn resolve_section(state: &AppState, raw: Option<&str>) -> AppResult<Option<String>> {
    let target = raw.unwrap_or_default().trim();
    if target.is_empty() {
        return Ok(None);
    }
    match load_section(state, target).await {
        Ok(section) => Ok(Some(section.id)),
        Err(AppError::InvalidInput(_)) => Err(AppError::not_found(SECTION_NOT_FOUND)),
        Err(e) => Err(e),
    }
}

/// Folder names are unique, ignoring case, within one section.
async fn ensure_folder_name_free(state: &AppState, name: &str, exclude_id: Option<&str>, section_id: Option<&str>) -> AppResult<()> {
    let taken = Folder::list(state).await?.iter().any(|folder| {
        Some(folder.id.as_str()) != exclude_id && folder.section.as_deref() == section_id && same_name(&folder.name, name)
    });
    if taken {
        return Err(AppError::conflict(format!("A folder named '{name}' already exists")));
    }
    Ok(())
}

/// Folders sorted by name, with their notebook counts.
pub async fn list_folders(state: &AppState) -> AppResult<Vec<FolderSummary>> {
    let mut rows: Vec<FolderRow> = state
        .db
        .query_as(
            "SELECT *, count(SELECT id FROM notebook WHERE folder = $parent.id) AS notebook_count FROM folder",
            Params::new(),
        )
        .await?;
    by_name(&mut rows, |row| (&row.folder.name, &row.folder.id));
    Ok(rows.into_iter().map(|row| FolderSummary::new(row.folder, row.notebook_count)).collect())
}

pub async fn create_folder(state: &AppState, input: &FolderInput) -> AppResult<FolderSummary> {
    let name = clean_name("Folder", &input.name)?;
    let section_id = resolve_section(state, input.section_id.as_ref().and_then(|s| s.as_deref())).await?;
    ensure_folder_name_free(state, &name, None, section_id.as_deref()).await?;
    Ok(FolderSummary::new(Folder::create(state, &name, section_id.as_deref()).await?, 0))
}

/// Rename a folder and, when `section_id` is given, move it.
pub async fn update_folder(state: &AppState, folder_id: &str, input: &FolderInput) -> AppResult<FolderSummary> {
    let folder = load_folder(state, folder_id).await?;
    let name = clean_name("Folder", &input.name)?;
    let section_change = match &input.section_id {
        Some(raw) => Some(resolve_section(state, raw.as_deref()).await?),
        None => None,
    };
    let section_id = section_change.as_ref().unwrap_or(&folder.section);
    ensure_folder_name_free(state, &name, Some(&folder.id), section_id.as_deref()).await?;

    let mut updated = Folder::rename(state, &folder.id, &name).await.map_err(not_found_as(FOLDER_NOT_FOUND))?;
    if let Some(section) = &section_change {
        updated = Folder::set_section(state, &folder.id, section.as_deref()).await.map_err(not_found_as(FOLDER_NOT_FOUND))?;
    }
    let notebook_count = count_where(state, Notebook::TABLE, "folder", &folder.id).await?;
    Ok(FolderSummary::new(updated, notebook_count))
}

/// Delete a folder. Its notebooks are kept and become ungrouped.
pub async fn delete_folder(state: &AppState, folder_id: &str) -> AppResult<FolderDeleted> {
    let folder = load_folder(state, folder_id).await?;
    let ungrouped = state
        .db
        .query(
            "UPDATE notebook SET folder = NONE WHERE folder = $folder RETURN id",
            Params::new().record("folder", &folder.id),
        )
        .await?;
    state.db.delete(&folder.id).await?;
    Ok(FolderDeleted { message: "Folder deleted successfully".to_string(), notebooks_ungrouped: ungrouped.len() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn notebook_order_is_one_clause_on_a_sortable_field() {
        for (given, expected) in
            [("updated desc", "updated desc"), ("name asc", "name asc"), ("created", "created"), ("  Name   DESC ", "name desc")]
        {
            assert_eq!(notebook_order(given).unwrap(), expected);
        }
        for bad in [
            "updated; DROP TABLE notebook",
            "name) FETCH (SELECT * FROM credential",
            "updated LIMIT 1",
            "name desc extra",
            "archived desc",
            "id",
            "name asc, updated desc",
            "updated sideways",
            "",
        ] {
            let error = notebook_order(bad).unwrap_err();
            assert_eq!(error.status(), axum::http::StatusCode::BAD_REQUEST, "{bad}");
            assert!(error.detail().contains("Allowed fields: created, name, updated"), "{bad}: {error}");
        }
    }

    #[test]
    fn update_bodies_tell_absent_from_null() {
        let update: NotebookUpdate = serde_json::from_str(r#"{"name": "x"}"#).unwrap();
        assert_eq!(update.description, None);
        assert_eq!(update.folder_id, None);
        let update: NotebookUpdate = serde_json::from_str(r#"{"description": null, "folder_id": "folder:a"}"#).unwrap();
        assert_eq!(update.description, Some(None));
        assert_eq!(update.folder_id, Some(Some("folder:a".to_string())));
        // An absent field stays absent through a round trip (for web UI callers).
        let json = serde_json::to_value(NotebookUpdate { name: Some("x".into()), ..Default::default() }).unwrap();
        assert!(json.get("folder_id").is_none());
    }

    #[test]
    fn ids_of_other_tables_are_not_found() {
        assert!(ensure_table("notebook:a", "notebook", "nope").is_ok());
        assert!(ensure_table("source:a", "notebook", "nope").unwrap_err().is_not_found());
        assert_eq!(ensure_table("a", "notebook", "nope").unwrap_err().status(), axum::http::StatusCode::BAD_REQUEST);
    }
}
