//! The notes workspace: the notes-first, three-pane view (library sidebar,
//! note list, editor) in the style of Apple Notes, Obsidian and AFFiNE.
//!
//! Everything here is a read model over the existing tables, plus two small
//! additions that the Python build never sees:
//!
//! - **Pins** live in their own `note_pin` table (`note_pin:<note key>`), so
//!   pinning a note never touches the note itself and never bumps its
//!   `updated` time (which would reorder every list).
//! - **Daily notes** are ordinary notes, titled `YYYY-MM-DD`, filed in a
//!   "Daily Notes" notebook that is created on first use, like Quick Notes.
//!
//! Tags are not stored: they are the `#hashtags` written in a note's text.

use std::collections::{BTreeMap, HashMap, HashSet};

use chrono::NaiveDate;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};

use super::note_docs::ATTACHMENTS_MARKER;
use super::notebooks::{self, ensure_table, NotebookQuery};
use super::notes::{self, NewNote};
use crate::db::{Doc, Json, Params};
use crate::domain::{Note, Notebook};
use crate::error::{AppError, AppResult};
use crate::jobs::JobQueue;
use crate::state::AppState;

pub fn register_jobs(_jobs: &JobQueue) {}

/// Record id of the notebook that holds daily notes.
pub const DAILY_NOTES_ID: &str = "notebook:daily_notes";
/// Characters of preview text shown under a note's title in the list.
const SNIPPET_CHARS: usize = 160;

/// A note as the workspace list shows it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkspaceNote {
    pub id: String,
    /// The note's title, or its first line when it has none.
    pub title: String,
    /// Plain-text preview of the content after the first line.
    pub snippet: String,
    /// Every notebook the note is filed in (usually one).
    pub notebook_ids: Vec<String>,
    /// The first notebook the note is filed in, if any.
    pub notebook_id: Option<String>,
    pub notebook_name: Option<String>,
    /// RFC 3339.
    pub updated: String,
    pub pinned: bool,
    /// `#hashtags` in the note, lower-cased, without the `#`, sorted.
    pub tags: Vec<String>,
    /// File extension of an attached original (`PDF`, `HTML`, ...), if any.
    pub file_kind: Option<String>,
    /// `"ai"` for model-written notes.
    pub note_type: Option<String>,
}

/// Every visible note, pinned ones first, then most recently updated.
pub async fn list_workspace_notes(state: &AppState) -> AppResult<Vec<WorkspaceNote>> {
    let notes = Note::list(state).await?;
    let membership = note_notebooks(state).await?;
    let names: HashMap<String, String> = notebooks::list_notebooks(state, &NotebookQuery::default())
        .await?
        .into_iter()
        .map(|nb| (nb.id, nb.name))
        .collect();
    let pins = pinned_ids(state).await?;

    let mut out: Vec<WorkspaceNote> = notes
        .into_iter()
        .map(|note| {
            let content = visible_content(note.content.as_deref().unwrap_or_default());
            let (title, snippet) = title_and_snippet(note.title.as_deref(), content);
            let notebook_ids = membership.get(&note.id).cloned().unwrap_or_default();
            // Show a note under its real notebook rather than the Quick Notes inbox.
            let notebook_id = notebook_ids.iter().find(|id| id.as_str() != QUICK_NOTES_ID).or_else(|| notebook_ids.first()).cloned();
            let notebook_name = notebook_id.as_ref().and_then(|id| names.get(id).cloned());
            let file_kind = match (&note.attachment_name, &note.attachment_file) {
                (Some(name), Some(file)) if !name.is_empty() && !file.is_empty() => {
                    Some(name.rsplit('.').next().unwrap_or("file").to_uppercase())
                }
                _ => None,
            };
            WorkspaceNote {
                pinned: pins.contains(&note.id),
                tags: extract_tags(content),
                updated: note.updated.map(|d| d.to_rfc3339()).unwrap_or_default(),
                id: note.id,
                title,
                snippet,
                notebook_ids,
                notebook_id,
                notebook_name,
                file_kind,
                note_type: note.note_type,
            }
        })
        .collect();
    // `Note::list` is already newest first; a stable sort keeps that order within each group.
    out.sort_by_key(|note| !note.pinned);
    Ok(out)
}

/// note id → the notebooks it is filed in, sorted by id.
async fn note_notebooks(state: &AppState) -> AppResult<HashMap<String, Vec<String>>> {
    let rows = state.db.query("SELECT in AS note, out AS notebook FROM artifact", Params::new()).await?;
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    for row in rows {
        if let (Some(note), Some(notebook)) = (row.get("note").and_then(Json::as_str), row.get("notebook").and_then(Json::as_str)) {
            if note.starts_with("note:") {
                map.entry(note.to_string()).or_default().push(notebook.to_string());
            }
        }
    }
    for ids in map.values_mut() {
        ids.sort();
        ids.dedup();
    }
    Ok(map)
}

/// A note's content without the generated attached-files section.
fn visible_content(content: &str) -> &str {
    match content.find(ATTACHMENTS_MARKER) {
        Some(at) => content[..at].trim_end().trim_end_matches("---").trim_end(),
        None => content,
    }
}

/// Strip the markdown that would look like noise in a one-line preview.
fn plain_line(line: &str) -> String {
    static LINK: Lazy<Regex> = Lazy::new(|| Regex::new(r"!?\[([^\]]*)\]\([^)]*\)").expect("valid regex"));
    let line = line.trim().trim_start_matches(['#', '>', '-', '*', '+', ' ']).trim();
    let line = line.strip_prefix("[ ]").or_else(|| line.strip_prefix("[x]")).unwrap_or(line).trim();
    let line = LINK.replace_all(line, "$1");
    line.replace(['*', '`'], "").replace("__", "").trim().to_string()
}

/// The list's title (stored title, or the first line of text) and preview.
pub fn title_and_snippet(title: Option<&str>, content: &str) -> (String, String) {
    let mut lines = content.lines().map(plain_line).filter(|line| !line.is_empty());
    let title = match title.map(str::trim).filter(|t| !t.is_empty()) {
        Some(title) => title.to_string(),
        None => lines.next().unwrap_or_else(|| "New Note".to_string()),
    };
    let mut snippet = String::new();
    for line in lines {
        if line == title {
            continue;
        }
        if !snippet.is_empty() {
            snippet.push(' ');
        }
        snippet.push_str(&line);
        if snippet.chars().count() >= SNIPPET_CHARS {
            break;
        }
    }
    let title: String = title.chars().take(200).collect();
    (title, snippet.chars().take(SNIPPET_CHARS).collect())
}

/// `#hashtags` in markdown text, lower-cased and de-duplicated. A tag starts
/// with a letter and may hold letters, digits, `_`, `-` and `/` (nested tags,
/// as in Obsidian). Headings (`# Title`), URL fragments and code are skipped.
pub fn extract_tags(content: &str) -> Vec<String> {
    static TAG: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?:^|[\s(,;])#([A-Za-z][A-Za-z0-9_/-]*)").expect("valid regex"));
    static CODE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?s)```.*?```|`[^`\n]*`").expect("valid regex"));
    let text = CODE.replace_all(content, " ");
    let mut tags: Vec<String> = TAG
        .captures_iter(&text)
        .map(|c| c[1].trim_end_matches(['/', '-']).to_lowercase())
        .filter(|t| !t.is_empty() && t.len() <= 64)
        .collect();
    tags.sort();
    tags.dedup();
    tags
}

/// Tag → number of notes carrying it, alphabetical.
pub fn tag_counts(notes: &[WorkspaceNote]) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();
    for note in notes {
        for tag in &note.tags {
            *counts.entry(tag.clone()).or_insert(0) += 1;
        }
    }
    counts
}

// ---------------------------------------------------------------------------
// Pins
// ---------------------------------------------------------------------------

fn pin_id(note_id: &str) -> String {
    format!("note_pin:{}", note_id.strip_prefix("note:").unwrap_or(note_id))
}

/// Ids of pinned notes.
pub async fn pinned_ids(state: &AppState) -> AppResult<HashSet<String>> {
    let rows = state.db.query("SELECT VALUE note FROM note_pin", Params::new()).await?;
    Ok(rows.iter().filter_map(Json::as_str).map(String::from).collect())
}

/// Pin or unpin a note. Pinning a missing note → `NotFound`; unpinning
/// always works (it also tidies up after a deleted note).
pub async fn set_pinned(state: &AppState, note_id: &str, pinned: bool) -> AppResult<bool> {
    ensure_table(note_id, Note::TABLE, "Note not found")?;
    let pin = pin_id(note_id);
    if pinned {
        Note::get(state, note_id).await.map_err(|_| AppError::not_found("Note not found"))?;
        state
            .db
            .query("UPSERT $pin SET note = $note, pinned_at = time::now()", Params::new().record("pin", &pin).record("note", note_id))
            .await?;
    } else {
        state.db.query("DELETE $pin", Params::new().record("pin", &pin)).await?;
    }
    Ok(pinned)
}

/// Flip a note's pin. Returns whether it is now pinned.
pub async fn toggle_pin(state: &AppState, note_id: &str) -> AppResult<bool> {
    let pinned = pinned_ids(state).await?.contains(note_id);
    set_pinned(state, note_id, !pinned).await
}

// ---------------------------------------------------------------------------
// Notes in notebooks
// ---------------------------------------------------------------------------

/// The notebooks a note is filed in, sorted by id (filings carry no date).
pub async fn note_notebook_ids(state: &AppState, note_id: &str) -> AppResult<Vec<String>> {
    ensure_table(note_id, Note::TABLE, "Note not found")?;
    let rows = state
        .db
        .query("SELECT VALUE out FROM artifact WHERE in = $id", Params::new().record("id", note_id))
        .await?;
    let mut ids: Vec<String> = Vec::new();
    for id in rows.iter().filter_map(Json::as_str) {
        if !ids.iter().any(|known| known == id) {
            ids.push(id.to_string());
        }
    }
    ids.sort();
    Ok(ids)
}

async fn unfile(state: &AppState, note_id: &str, notebook_id: &str) -> AppResult<()> {
    let _write = state.db.lock_record(note_id).await;
    crate::db::retry_conflicts(|| {
        state.db.query(
            "DELETE artifact WHERE in = $note AND out = $notebook",
            Params::new().record("note", note_id).record("notebook", notebook_id),
        )
    })
    .await?;
    Ok(())
}

/// What changed when a note's notebooks were set.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotebookChange {
    pub added: Vec<String>,
    pub removed: Vec<String>,
}

/// File a note in exactly these notebooks. A note keeps at least one
/// notebook: that is what gives it to a notebook's chat, search scope and
/// studio, and what the editor saves through. Unknown notebooks → `NotFound`.
pub async fn set_note_notebooks(state: &AppState, note_id: &str, wanted: &[String]) -> AppResult<NotebookChange> {
    let mut wanted_ids: Vec<String> = Vec::new();
    for id in wanted.iter().map(|id| id.trim()).filter(|id| !id.is_empty()) {
        if !wanted_ids.iter().any(|known| known == id) {
            wanted_ids.push(id.to_string());
        }
    }
    if wanted_ids.is_empty() {
        return Err(AppError::invalid("A note needs at least one notebook. Pick another before removing this one."));
    }
    Note::get(state, note_id).await.map_err(|_| AppError::not_found("Note not found"))?;
    for id in &wanted_ids {
        ensure_table(id, Notebook::TABLE, "Notebook not found")?;
        Notebook::get(state, id).await.map_err(|_| AppError::not_found("Notebook not found"))?;
    }
    let current = note_notebook_ids(state, note_id).await?;
    let mut change = NotebookChange::default();
    let note = Note { id: note_id.to_string(), ..Note::default() };
    for id in wanted_ids.iter().filter(|id| !current.contains(id)) {
        note.add_to_notebook(state, id).await?;
        change.added.push(id.clone());
    }
    for id in current.iter().filter(|id| !wanted_ids.contains(id)) {
        unfile(state, note_id, id).await?;
        change.removed.push(id.clone());
    }
    Ok(change)
}

/// Drop a note on a notebook: file it there. A note that was only in Quick
/// Notes (the inbox) leaves it, so filing a quick capture is one gesture.
/// Returns the notebooks it is in afterwards.
pub async fn file_note(state: &AppState, note_id: &str, notebook_id: &str) -> AppResult<Vec<String>> {
    let current = note_notebook_ids(state, note_id).await?;
    let only_in_inbox = current.len() == 1 && current[0] == QUICK_NOTES_ID;
    let mut wanted = if only_in_inbox && notebook_id != QUICK_NOTES_ID { Vec::new() } else { current };
    if !wanted.iter().any(|id| id == notebook_id) {
        wanted.push(notebook_id.to_string());
    }
    set_note_notebooks(state, note_id, &wanted).await?;
    note_notebook_ids(state, note_id).await
}

/// Record id of the Quick Notes notebook (the inbox for captures).
pub const QUICK_NOTES_ID: &str = "notebook:quick_notes";

// ---------------------------------------------------------------------------
// Deleting a folder
// ---------------------------------------------------------------------------

/// What deleting a folder from the Notes view removed and kept.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct FolderDeleted {
    /// Notes that lived only in this folder, deleted with it.
    pub deleted_notes: i64,
    /// Notes also filed in other folders: unfiled from this one and kept.
    pub kept_notes: i64,
    pub deleted_chat_sessions: i64,
}

/// Delete a folder (a notebook) from the Notes view. A note that is also
/// filed in another folder is **kept**: it only loses its filing here. Notes
/// that live nowhere else go with the folder, as do its meetings and chat
/// sessions; sources are unlinked, never deleted. (The JSON API's notebook
/// delete is unchanged and still removes every note, as the Python build did.)
pub async fn delete_folder(state: &AppState, notebook_id: &str) -> AppResult<FolderDeleted> {
    ensure_table(notebook_id, Notebook::TABLE, "Folder not found")?;
    let notebook = Notebook::get(state, notebook_id).await.map_err(|_| AppError::not_found("Folder not found"))?;
    let mut kept = 0;
    for note in notebook.notes(state, false).await? {
        if note_notebook_ids(state, &note.id).await?.iter().any(|id| id != notebook_id) {
            unfile(state, &note.id, notebook_id).await?;
            kept += 1;
        }
    }
    let summary = notebook.delete(state, false).await?;
    Ok(FolderDeleted {
        deleted_notes: summary.deleted_notes,
        kept_notes: kept,
        deleted_chat_sessions: summary.deleted_chat_sessions,
    })
}

/// notebook id → how many meetings it stores (for the delete confirmation).
pub async fn meeting_counts(state: &AppState) -> AppResult<HashMap<String, i64>> {
    let rows = state.db.query("SELECT notebook_id, count() AS n FROM meeting GROUP BY notebook_id", Params::new()).await?;
    Ok(rows
        .iter()
        .filter_map(|row| Some((row.get("notebook_id")?.as_str()?.to_string(), row.get("n")?.as_i64()?)))
        .collect())
}

/// notebook id → how many chat sessions are attached to it.
pub async fn chat_counts(state: &AppState) -> AppResult<HashMap<String, i64>> {
    let rows = state.db.query("SELECT out, count() AS n FROM refers_to GROUP BY out", Params::new()).await?;
    Ok(rows
        .iter()
        .filter_map(|row| Some((row.get("out")?.as_str()?.to_string(), row.get("n")?.as_i64()?)))
        .filter(|(id, _)| id.starts_with("notebook:"))
        .collect())
}

// ---------------------------------------------------------------------------
// Daily notes
// ---------------------------------------------------------------------------

/// The Daily Notes notebook, created (or un-archived) on first use.
pub async fn daily_notes_notebook(state: &AppState) -> AppResult<&'static str> {
    let _write = state.db.lock_record(DAILY_NOTES_ID).await;
    match Notebook::get(state, DAILY_NOTES_ID).await {
        Ok(notebook) if notebook.archived => {
            state.db.update(DAILY_NOTES_ID, Doc::new().set("archived", false)).await?;
        }
        Ok(_) => {}
        Err(error) if error.is_not_found() => {
            let notebook = Doc::new()
                .set("name", "Daily Notes")
                .set("description", "One note per day, opened with Today.")
                .set("archived", false);
            state.db.create_with_id(DAILY_NOTES_ID, notebook).await?;
        }
        Err(error) => return Err(error),
    }
    Ok(DAILY_NOTES_ID)
}

/// The starting text of a new daily note.
pub fn daily_template(date: NaiveDate) -> String {
    format!("{}\n\n## Tasks\n\n- [ ] Top task for today\n\n## Notes\n\n", date.format("%A %-d %B %Y"))
}

/// Open the daily note for `date` (the writer's local date), creating it
/// from [`daily_template`] the first time. Returns the note id.
pub async fn daily_note(state: &AppState, date: NaiveDate) -> AppResult<String> {
    let notebook_id = daily_notes_notebook(state).await?;
    let title = date.format("%Y-%m-%d").to_string();
    // Serialise "find or create" so two quick clicks make one note.
    let _write = state.db.lock_record(&format!("{notebook_id}_{title}")).await;
    let existing = notes::list_notes(state, Some(notebook_id)).await?;
    if let Some(note) = existing.into_iter().find(|n| n.title.as_deref() == Some(title.as_str())) {
        return Ok(note.id);
    }
    let new = NewNote {
        title: Some(title),
        content: daily_template(date),
        note_type: Some(Some("human".to_string())),
        notebook_id: Some(notebook_id.to_string()),
    };
    Ok(notes::create_note(state, &new).await?.id)
}

/// Parse a `YYYY-MM-DD` date from a form.
pub fn parse_date(raw: &str) -> AppResult<NaiveDate> {
    NaiveDate::parse_from_str(raw.trim(), "%Y-%m-%d").map_err(|_| AppError::invalid("date must be YYYY-MM-DD"))
}

// ---------------------------------------------------------------------------
// Quick switcher
// ---------------------------------------------------------------------------

/// What the command palette searches: every note and notebook, light.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaletteIndex {
    pub notes: Vec<PaletteNote>,
    pub notebooks: Vec<PaletteNotebook>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaletteNote {
    pub id: String,
    pub title: String,
    pub notebook: Option<String>,
    pub updated: String,
    pub pinned: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PaletteNotebook {
    pub id: String,
    pub name: String,
}

pub async fn palette_index(state: &AppState) -> AppResult<PaletteIndex> {
    let notes = list_workspace_notes(state).await?;
    let tags = tag_counts(&notes).into_keys().collect();
    let notebooks = notebooks::list_notebooks(state, &NotebookQuery { archived: Some(false), ..NotebookQuery::default() })
        .await?
        .into_iter()
        .map(|nb| PaletteNotebook { id: nb.id, name: nb.name })
        .collect();
    let notes = notes
        .into_iter()
        .map(|n| PaletteNote { id: n.id, title: n.title, notebook: n.notebook_name, updated: n.updated, pinned: n.pinned })
        .collect();
    Ok(PaletteIndex { notes, notebooks, tags })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::TestApp;

    #[tokio::test]
    async fn deleting_a_folder_keeps_notes_filed_elsewhere() {
        let app = TestApp::new().await;
        let (_, a) = app.post("/api/notebooks", serde_json::json!({ "name": "A", "description": "" })).await;
        let (_, b) = app.post("/api/notebooks", serde_json::json!({ "name": "B", "description": "" })).await;
        let a_id = a["id"].as_str().unwrap().to_string();
        let b_id = b["id"].as_str().unwrap().to_string();
        let shared = notes::create_note(
            &app.state,
            &NewNote { title: Some("Shared".into()), content: "in both".into(), notebook_id: Some(a_id.clone()), ..Default::default() },
        )
        .await
        .unwrap();
        file_note(&app.state, &shared.id, &b_id).await.unwrap();
        let only = notes::create_note(
            &app.state,
            &NewNote { title: Some("Only in A".into()), content: "goes".into(), notebook_id: Some(a_id.clone()), ..Default::default() },
        )
        .await
        .unwrap();
        app.state.db.create("meeting", Doc::new().set("notebook_id", &a_id).set("title", "Standup")).await.unwrap();
        assert_eq!(meeting_counts(&app.state).await.unwrap().get(&a_id), Some(&1));

        let report = delete_folder(&app.state, &a_id).await.unwrap();
        assert_eq!((report.deleted_notes, report.kept_notes), (1, 1));

        // The shared note survives, filed only in B now; the exclusive one is gone.
        assert_eq!(note_notebook_ids(&app.state, &shared.id).await.unwrap(), vec![b_id.clone()]);
        assert!(notes::get_note(&app.state, &only.id).await.is_err());
        // The folder's meeting went with it.
        assert_eq!(meeting_counts(&app.state).await.unwrap().get(&a_id), None);
    }

    #[test]
    fn tags_are_hashtags_not_headings_or_code() {
        let text = "# Heading\nSome #Rust and #rust/async ideas, (#todo) a#b `#code` #9 #ok-\n```\n#fenced\n```";
        assert_eq!(extract_tags(text), vec!["ok", "rust", "rust/async", "todo"]);
        assert!(extract_tags("## Section\n\n#  spaced").is_empty());
    }

    #[test]
    fn titles_fall_back_to_the_first_line_and_snippets_skip_markdown() {
        let (title, snippet) = title_and_snippet(None, "# Groceries\n\n- [ ] **milk**\n- [x] [eggs](http://x)\n");
        assert_eq!(title, "Groceries");
        assert_eq!(snippet, "milk eggs");
        let (title, snippet) = title_and_snippet(Some("  Plan "), "Plan\nStep one");
        assert_eq!((title.as_str(), snippet.as_str()), ("Plan", "Step one"));
        assert_eq!(title_and_snippet(Some(""), "").0, "New Note");
    }

    #[test]
    fn attachment_sections_are_not_previewed() {
        let content = format!("Body text\n\n---\n{ATTACHMENTS_MARKER}\n\n**Attached files**\n- [a.pdf](/note-blobs/x)");
        assert_eq!(visible_content(&content), "Body text");
    }

    #[tokio::test]
    async fn notes_list_with_notebooks_pins_tags_and_daily_notes() {
        let app = TestApp::new().await;
        let (_, nb) = app.post("/api/notebooks", serde_json::json!({ "name": "Garden", "description": "" })).await;
        let nb_id = nb["id"].as_str().unwrap().to_string();
        let first = notes::create_note(
            &app.state,
            &NewNote { title: Some("Tomatoes".into()), content: "Plant in May #garden #Spring".into(), notebook_id: Some(nb_id.clone()), ..Default::default() },
        )
        .await
        .unwrap();
        let second = notes::create_note(&app.state, &NewNote { content: "Loose thought\nmore".into(), ..Default::default() }).await.unwrap();

        let list = list_workspace_notes(&app.state).await.unwrap();
        assert_eq!(list.len(), 2);
        assert_eq!(list[0].id, second.id, "newest first");
        assert_eq!(list[0].title, "Loose thought");
        assert_eq!(list[0].notebook_id, None);
        assert_eq!(list[1].notebook_name.as_deref(), Some("Garden"));
        assert_eq!(list[1].tags, vec!["garden", "spring"]);

        // Pinning moves a note to the top and leaves its updated time alone.
        let before = list[1].updated.clone();
        assert!(toggle_pin(&app.state, &first.id).await.unwrap());
        let list = list_workspace_notes(&app.state).await.unwrap();
        assert_eq!(list[0].id, first.id);
        assert!(list[0].pinned);
        assert_eq!(list[0].updated, before);
        assert!(!toggle_pin(&app.state, &first.id).await.unwrap());
        assert!(pinned_ids(&app.state).await.unwrap().is_empty());
        assert!(set_pinned(&app.state, "note:missing", true).await.unwrap_err().is_not_found());

        // The same date opens the same daily note.
        let date = parse_date("2026-11-16").unwrap();
        let daily = daily_note(&app.state, date).await.unwrap();
        assert_eq!(daily_note(&app.state, date).await.unwrap(), daily);
        let note = notes::get_note(&app.state, &daily).await.unwrap();
        assert_eq!(note.title.as_deref(), Some("2026-11-16"));
        assert!(note.content.unwrap().starts_with("Monday 16 November 2026"));
        assert!(parse_date("16/11/2026").is_err());

        // Notebooks: add, move, keep at least one; dropping files a Quick Note out of the inbox.
        let (_, other) = app.post("/api/notebooks", serde_json::json!({ "name": "Kitchen", "description": "" })).await;
        let other_id = other["id"].as_str().unwrap().to_string();
        let change = set_note_notebooks(&app.state, &first.id, &[nb_id.clone(), other_id.clone()]).await.unwrap();
        assert_eq!(change.added, vec![other_id.clone()]);
        let both = { let mut v = vec![nb_id.clone(), other_id.clone()]; v.sort(); v };
        assert_eq!(note_notebook_ids(&app.state, &first.id).await.unwrap(), both);
        let change = set_note_notebooks(&app.state, &first.id, std::slice::from_ref(&other_id)).await.unwrap();
        assert_eq!(change.removed, vec![nb_id.clone()]);
        assert!(set_note_notebooks(&app.state, &first.id, &[]).await.unwrap_err().to_string().contains("at least one"));
        assert!(set_note_notebooks(&app.state, &first.id, &["notebook:nope".into()]).await.unwrap_err().is_not_found());
        let quick = crate::services::note_docs::quick_notes_notebook(&app.state).await.unwrap();
        let captured = notes::create_note(&app.state, &NewNote { content: "capture".into(), notebook_id: Some(quick.into()), ..Default::default() }).await.unwrap();
        assert_eq!(file_note(&app.state, &captured.id, &nb_id).await.unwrap(), vec![nb_id.clone()]);
        assert_eq!(file_note(&app.state, &captured.id, &other_id).await.unwrap(), both);
        // A note in no notebook at all (made through the API) can be filed too.
        assert_eq!(file_note(&app.state, &second.id, &nb_id).await.unwrap(), vec![nb_id.clone()]);

        let index = palette_index(&app.state).await.unwrap();
        assert_eq!(index.notes.len(), 4);
        assert!(index.notebooks.iter().any(|nb| nb.name == "Daily Notes"));
        assert_eq!(index.tags, vec!["garden", "spring"]);
    }
}
