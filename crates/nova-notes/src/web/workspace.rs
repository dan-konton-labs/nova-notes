//! The notes workspace (`/notes`): a notes-first, three-pane view in the
//! spirit of Apple Notes, Obsidian and AFFiNE.
//!
//! - **Library sidebar**: All Notes, Pinned, Today (daily note), Quick Notes,
//!   the notebooks (grouped by folder) and the `#tags` found in notes. The
//!   rest of the app (classic notebook view, Ask, Podcasts, Settings, ...) is
//!   one click away at the bottom.
//! - **Note list**: title, date and preview, pinned first, filtered as you
//!   type. The client groups it by Today / Yesterday / Previous 7 days / month.
//! - **Editor**: the same rich editor (and file viewer) the notebook page
//!   uses, opened in place.
//!
//! The command palette (Ctrl/Cmd+K, on every page) reads `/notes/palette`.
//! Nothing here replaces an existing page; the classic views are unchanged.

use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::{Form, Router};
use maud::{html, Markup, PreEscaped, DOCTYPE};
use serde::Deserialize;

use super::notebooks::{note_body, note_editor};
use super::{head, human_date_str, markdown, safe_path_or, WebResult};
use crate::error::AppError;
use crate::services::note_docs;
use crate::services::notebooks::{self, FolderInput, FolderSummary, NewNotebook, NotebookQuery, NotebookSummary, NotebookUpdate};
use crate::services::notes::{self, NoteView};
use crate::services::workspace::{self, tag_counts, WorkspaceNote};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/notes", get(workspace_page))
        .route("/notes/palette", get(palette))
        .route("/notes/new", post(new_note))
        .route("/notes/daily", post(daily_note))
        .route("/notes/notebooks", post(new_notebook))
        .route("/notes/{note_id}/pin", post(toggle_pin))
        .route("/notes/{note_id}/delete", post(delete_note))
        .route("/notes/{note_id}/notebooks", post(set_notebooks))
        .route("/notes/{note_id}/file", post(file_note))
        .route("/notes/{note_id}/move", post(move_note))
        .route("/notes/folders/{id}/rename", post(rename_folder))
        .route("/notes/folders/{id}/move", post(move_folder))
        .route("/notes/folders/{id}/delete", post(delete_folder))
        .route("/notes/groups/{id}/rename", post(rename_group))
        .route("/notes/groups/{id}/delete", post(delete_group))
}

// ---------------------------------------------------------------------------
// Views
// ---------------------------------------------------------------------------

/// Which notes the list shows, and which note is open.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ViewQuery {
    /// Only this notebook's notes.
    #[serde(default)]
    nb: Option<String>,
    /// Only notes carrying this `#tag`.
    #[serde(default)]
    tag: Option<String>,
    /// Only pinned notes (`1`).
    #[serde(default)]
    pinned: Option<String>,
    /// The open note.
    #[serde(default)]
    note: Option<String>,
    /// The open note was just created (focus its title).
    #[serde(default)]
    new: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum View {
    All,
    Pinned,
    Notebook(String),
    Tag(String),
}

impl ViewQuery {
    fn view(&self) -> View {
        let some = |v: &Option<String>| v.as_deref().map(str::trim).filter(|v| !v.is_empty()).map(String::from);
        if let Some(nb) = some(&self.nb) {
            View::Notebook(nb)
        } else if let Some(tag) = some(&self.tag) {
            View::Tag(tag.trim_start_matches('#').to_lowercase())
        } else if self.pinned.as_deref().is_some_and(|p| !p.is_empty() && p != "0") {
            View::Pinned
        } else {
            View::All
        }
    }

    fn note(&self) -> Option<&str> {
        self.note.as_deref().map(str::trim).filter(|n| !n.is_empty())
    }
}

impl View {
    /// Query string selecting this view (without the leading `?`).
    fn query(&self) -> String {
        match self {
            View::All => String::new(),
            View::Pinned => "pinned=1".into(),
            View::Notebook(id) => format!("nb={}", enc(id)),
            View::Tag(tag) => format!("tag={}", enc(tag)),
        }
    }

    /// `/notes` in this view, optionally with a note open.
    fn href(&self, note: Option<&str>) -> String {
        let mut parts: Vec<String> = Vec::new();
        let q = self.query();
        if !q.is_empty() {
            parts.push(q);
        }
        if let Some(note) = note {
            parts.push(format!("note={}", enc(note)));
        }
        if parts.is_empty() { "/notes".into() } else { format!("/notes?{}", parts.join("&")) }
    }

    fn matches(&self, note: &WorkspaceNote) -> bool {
        match self {
            View::All => true,
            View::Pinned => note.pinned,
            View::Notebook(id) => note.notebook_ids.iter().any(|nb| nb == id),
            View::Tag(tag) => note.tags.iter().any(|t| t == tag || t.starts_with(&format!("{tag}/"))),
        }
    }
}

fn enc(value: &str) -> String {
    url::form_urlencoded::byte_serialize(value.as_bytes()).collect()
}

/// Inline SVG icons (stroke = currentColor), so the page needs no icon font.
fn icon(name: &str) -> PreEscaped<&'static str> {
    PreEscaped(match name {
        "search" => r#"<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" aria-hidden="true"><circle cx="7" cy="7" r="4.5"/><path d="m10.5 10.5 3.5 3.5"/></svg>"#,
        "compose" => r#"<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><path d="M13.5 8.5v4a1.5 1.5 0 0 1-1.5 1.5H4A1.5 1.5 0 0 1 2.5 12.5v-8A1.5 1.5 0 0 1 4 3h4"/><path d="m11.5 2 2.5 2.5L8.5 10H6V7.5Z"/></svg>"#,
        "notes" => r#"<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><rect x="3" y="1.5" width="10" height="13" rx="2"/><path d="M5.5 5h5M5.5 8h5M5.5 11h3"/></svg>"#,
        "pin" => r#"<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><path d="M9.8 1.8 14.2 6.2l-2 .6-2.6 2.6-.4 3.1-1.3 1.3L5.6 11 2.5 14.1M5.6 11 1.9 7.3l1.3-1.3 3.1-.4 2.6-2.6Z" stroke-linejoin="round"/></svg>"#,
        "today" => r#"<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><rect x="2" y="3" width="12" height="11" rx="2"/><path d="M2 6.5h12M5.5 1.5v3M10.5 1.5v3"/><circle cx="8" cy="10" r="1.2" fill="currentColor" stroke="none"/></svg>"#,
        "bolt" => r#"<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><path d="M9 1.5 3.5 9H8l-1 5.5L12.5 7H8Z" stroke-linejoin="round"/></svg>"#,
        "book" => r#"<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><path d="M2.5 3a1.5 1.5 0 0 1 1.5-1.5h8.5v11H4a1.5 1.5 0 0 0-1.5 1.5Zm0 11A1.5 1.5 0 0 0 4 15.5h8.5v-3"/></svg>"#,
        "folder" => r#"<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><path d="M1.5 4.5A1.5 1.5 0 0 1 3 3h3l1.5 1.5H13A1.5 1.5 0 0 1 14.5 6v6A1.5 1.5 0 0 1 13 13.5H3A1.5 1.5 0 0 1 1.5 12Z"/></svg>"#,
        "hash" => r#"<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><path d="M6 2 5 14M11 2l-1 12M2.5 5.5h11M2 10.5h11"/></svg>"#,
        "sidebar" => r#"<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3" aria-hidden="true"><rect x="1.5" y="2.5" width="13" height="11" rx="2"/><path d="M6 2.5v11"/></svg>"#,
        "focus" => r#"<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><path d="M2 6V3a1 1 0 0 1 1-1h3M10 2h3a1 1 0 0 1 1 1v3M14 10v3a1 1 0 0 1-1 1h-3M6 14H3a1 1 0 0 1-1-1v-3"/></svg>"#,
        "trash" => r#"<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><path d="M2.5 4h11M6 4V2.5h4V4M4 4l.7 9.5a1 1 0 0 0 1 .9h4.6a1 1 0 0 0 1-.9L12 4"/></svg>"#,
        "open" => r#"<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><path d="M9 2.5h4.5V7M13.5 2.5 7 9M11.5 9.5v3a1 1 0 0 1-1 1h-7a1 1 0 0 1-1-1v-7a1 1 0 0 1 1-1h3"/></svg>"#,
        "back" => r#"<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" aria-hidden="true"><path d="M10 3 5 8l5 5"/></svg>"#,
        "grid" => r#"<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><rect x="2" y="2" width="5" height="5" rx="1"/><rect x="9" y="2" width="5" height="5" rx="1"/><rect x="2" y="9" width="5" height="5" rx="1"/><rect x="9" y="9" width="5" height="5" rx="1"/></svg>"#,
        _ => "",
    })
}

// ---------------------------------------------------------------------------
// The page
// ---------------------------------------------------------------------------

struct Library {
    notes: Vec<WorkspaceNote>,
    notebooks: Vec<NotebookSummary>,
    folders: Vec<FolderSummary>,
    /// notebook id → meetings stored in it; absent means none.
    meetings: std::collections::HashMap<String, i64>,
    /// notebook id → chat sessions attached to it; absent means none.
    chats: std::collections::HashMap<String, i64>,
}

async fn load_library(state: &AppState) -> WebResult<Library> {
    let notes = workspace::list_workspace_notes(state).await?;
    let notebooks = active_notebooks(state).await?;
    let folders = notebooks::list_folders(state).await?;
    let meetings = workspace::meeting_counts(state).await?;
    let chats = workspace::chat_counts(state).await?;
    Ok(Library { notes, notebooks, folders, meetings, chats })
}

/// Active notebooks by name: the targets a note can be filed in.
pub(crate) async fn active_notebooks(state: &AppState) -> WebResult<Vec<NotebookSummary>> {
    Ok(notebooks::list_notebooks(
        state,
        &NotebookQuery { archived: Some(false), order_by: Some("name asc".into()), ..NotebookQuery::default() },
    )
    .await?)
}

async fn workspace_page(State(state): State<AppState>, Query(query): Query<ViewQuery>) -> WebResult<Response> {
    let view = query.view();
    let library = load_library(&state).await?;
    // A notebook view of a notebook that no longer exists falls back to All Notes.
    if let View::Notebook(id) = &view {
        if !library.notebooks.iter().any(|nb| &nb.id == id) && notebooks::get_notebook(&state, id).await.is_err() {
            return Ok(Redirect::to("/notes").into_response());
        }
    }
    let open = match query.note() {
        Some(id) => Some(open_note(&state, id).await?),
        None => None,
    };
    let fresh = query.new.as_deref().is_some_and(|v| !v.is_empty() && v != "0");
    let title = match &open {
        Some(open) => open.title.clone(),
        None => view_title(&library, &view),
    };
    Ok(shell(&state, &title, &library, &view, open.as_ref(), fresh).into_response())
}

/// A note opened in the editor pane.
struct OpenNote {
    note: NoteView,
    title: String,
    /// The notebooks the note is filed in.
    notebook_ids: Vec<String>,
    /// The notebook the editor saves through.
    notebook_id: Option<String>,
    notebook_name: Option<String>,
    pinned: bool,
}

/// The notebook a note is shown under: any but the Quick Notes inbox, if it has one.
pub(crate) fn home_notebook(ids: &[String]) -> Option<String> {
    ids.iter().find(|id| id.as_str() != workspace::QUICK_NOTES_ID).or_else(|| ids.first()).cloned()
}

async fn open_note(state: &AppState, id: &str) -> WebResult<OpenNote> {
    let note = notes::get_note(state, id).await?;
    let notebook_ids = workspace::note_notebook_ids(state, &note.id).await?;
    let notebook_id = home_notebook(&notebook_ids);
    let notebook_name = match &notebook_id {
        Some(nb) => notebooks::get_notebook(state, nb).await.ok().map(|nb| nb.name),
        None => None,
    };
    let pinned = workspace::pinned_ids(state).await?.contains(&note.id);
    let title = note.title.clone().filter(|t| !t.trim().is_empty()).unwrap_or_else(|| "New Note".into());
    Ok(OpenNote { note, title, notebook_ids, notebook_id, notebook_name, pinned })
}

fn notebook_name(library: &Library, id: &str) -> String {
    library.notebooks.iter().find(|nb| nb.id == id).map(|nb| nb.name.clone()).unwrap_or_else(|| "Notebook".into())
}

fn view_title(library: &Library, view: &View) -> String {
    match view {
        View::All => "All Notes".into(),
        View::Pinned => "Pinned".into(),
        View::Notebook(id) => notebook_name(library, id),
        View::Tag(tag) => format!("#{tag}"),
    }
}

fn shell(state: &AppState, title: &str, library: &Library, view: &View, open: Option<&OpenNote>, fresh: bool) -> Markup {
    let auth = state.config.password.is_some();
    html! {
        (DOCTYPE)
        html lang="en" {
            head {
                (head(title))
                script { (PreEscaped(r#"
                    try { if (localStorage.getItem("nn-ws-focus") === "on") document.documentElement.classList.add("ws-focus"); } catch (error) {}
                "#)) }
            }
            body.ws-body hx-headers=r#"{"X-Requested-With": "htmx"}"# {
                div.ws #ws data-ws-pane=(if open.is_some() { "editor" } else { "list" }) {
                    (sidebar(library, view, auth))
                    (note_list(library, view, open))
                    (editor_pane(library, view, open, fresh))
                }
                div.ws-scrim data-ws-drawer-close {}
                button.btn.menu-toggle type="button" data-menu-toggle aria-controls="site-sidebar" aria-expanded="false" { "Menu" }
                (new_notebook_dialog())
                div #toasts aria-live="polite" {}
            }
        }
    }
}

/// A coloured sidebar dot, as the app's main sidebar draws them.
fn dot(colour: &str) -> Markup {
    html! { span.dot style=(format!("background:{colour}")) {} }
}

/// The library sidebar. It is the app's own sidebar (`.sidebar`, `.nav-link`,
/// the same collapse button and mobile drawer), with notes in it.
fn sidebar(library: &Library, view: &View, auth: bool) -> Markup {
    let pinned = library.notes.iter().filter(|n| n.pinned).count();
    let tags = tag_counts(&library.notes);
    let count = |id: &str| library.notes.iter().filter(|n| n.notebook_ids.iter().any(|nb| nb == id)).count();
    let new_target = match view {
        View::Notebook(id) => id.clone(),
        _ => String::new(),
    };
    let nb_row = |nb: &NotebookSummary| {
        let current = *view == View::Notebook(nb.id.clone());
        let colour = if nb.id == workspace::QUICK_NOTES_ID {
            "var(--orange)"
        } else if nb.id == workspace::DAILY_NOTES_ID {
            "var(--red)"
        } else {
            "var(--teal-sys)"
        };
        html! {
            div.ws-nav-row {
                a.nav-link href=(View::Notebook(nb.id.clone()).href(None)) aria-current=[current.then_some("page")]
                    data-ws-drop=(nb.id) data-ws-drop-name=(nb.name) title=(format!("{} (drop a note here to file it)", nb.name)) {
                    (dot(colour))
                    span.ws-nav-label { (nb.name) }
                    span.ws-count { (count(&nb.id)) }
                }
                @let exclusive = library.notes.iter().filter(|n| n.notebook_ids.len() == 1 && n.notebook_ids[0] == nb.id).count();
                (folder_menu(nb, count(&nb.id), exclusive, library))
            }
        }
    };
    let loose: Vec<&NotebookSummary> = library.notebooks.iter().filter(|nb| nb.folder_id.is_none()).collect();
    html! {
        nav.sidebar.ws-sidebar #site-sidebar aria-label="Main" {
            div.sidebar-header {
                a.brand href="/notes" { span.pebbles aria-hidden="true" { i {} i {} i {} } "Nova Notes" }
                button.btn.btn-ghost.btn-icon.sidebar-toggle type="button" data-sidebar-toggle
                    aria-controls="sidebar-links" aria-expanded="true" aria-label="Collapse sidebar" title="Collapse sidebar (Ctrl+\\)" { (icon("sidebar")) }
            }
            div.sidebar-links #sidebar-links {
                button.ws-search type="button" data-palette-open {
                    (icon("search")) span { "Search" } kbd { "Ctrl K" }
                }
                div.nav-group {
                    div.nav-title { "Notes" }
                    form method="post" action="/notes/new" data-note-create {
                        input type="hidden" name="nb" value=(new_target);
                        button.nav-link.nav-action type="submit" title="New note (Ctrl+Alt+N)" {
                            span.dot.quick-note-dot aria-hidden="true" { "+" } "New Note"
                        }
                    }
                    a.nav-link href="/notes" aria-current=[(*view == View::All).then_some("page")] {
                        (dot("var(--yellow)")) span.ws-nav-label { "All Notes" } span.ws-count { (library.notes.len()) }
                    }
                    a.nav-link href="/notes?pinned=1" aria-current=[(*view == View::Pinned).then_some("page")] {
                        (dot("var(--orange)")) span.ws-nav-label { "Pinned" } span.ws-count { (pinned) }
                    }
                    form method="post" action="/notes/daily" data-note-create data-ws-daily {
                        input type="hidden" name="date" value="";
                        button.nav-link.nav-action type="submit" title="Today's note (Ctrl+Alt+D)" {
                            (dot("var(--red)")) span.ws-nav-label { "Today" }
                        }
                    }
                }
                div.nav-group {
                    div.nav-title.ws-title-row {
                        span { "Folders" }
                        button.ws-title-add type="button" data-open="#ws-new-notebook" title="New folder" aria-label="New folder" { "+" }
                    }
                    @for nb in &loose { (nb_row(nb)) }
                    @for folder in &library.folders {
                        @let inside: Vec<&NotebookSummary> = library.notebooks.iter().filter(|nb| nb.folder_id.as_deref() == Some(folder.id.as_str())).collect();
                        @if !inside.is_empty() {
                            @let open = matches!(view, View::Notebook(id) if inside.iter().any(|nb| &nb.id == id));
                            div.ws-nav-row.ws-group-row {
                                details.ws-folder open[open] {
                                    summary.nav-link { (dot("var(--grey)")) span.ws-nav-label { (folder.name) } span.ws-count { (inside.len()) } }
                                    div.ws-folder-body { @for nb in &inside { (nb_row(nb)) } }
                                }
                                (group_menu(folder))
                            }
                        }
                    }
                    @if library.notebooks.is_empty() {
                        p.faint.small.ws-hint { "No folders yet." }
                    }
                }
                @if !tags.is_empty() {
                    div.nav-group {
                        div.nav-title { "Tags" }
                        div.ws-tags {
                            @for (tag, n) in &tags {
                                a.badge.ws-tag href=(View::Tag(tag.clone()).href(None)) aria-current=[(*view == View::Tag(tag.clone())).then_some("page")] {
                                    "#" (tag) span.ws-count { (n) }
                                }
                            }
                        }
                    }
                }
                a.ws-studio href="/notebooks" title="Open the studio: notebooks with their sources, chat, podcasts and settings" {
                    (icon("grid"))
                    span { "Studio" }
                }
                div.sidebar-foot {
                    button.btn.btn-ghost.btn-sm type="button" data-theme-toggle { "Toggle theme" }
                    @if auth { form method="post" action="/logout" { button.btn.btn-ghost.btn-sm type="submit" { "Sign out" } } }
                    span { "v" (crate::VERSION) }
                }
            }
        }
    }
}

fn new_notebook_dialog() -> Markup {
    html! {
        dialog #ws-new-notebook aria-labelledby="ws-new-notebook-title" {
            form method="post" action="/notes/notebooks" {
                div.dialog-head { h2 #ws-new-notebook-title { "New notebook" } button.btn.btn-ghost.btn-sm type="button" data-close { "Close" } }
                div.dialog-body {
                    div.field {
                        label for="ws-new-notebook-name" { "Name" }
                        input #ws-new-notebook-name type="text" name="name" required placeholder="Notebook name" autocomplete="off";
                    }
                    p.hint { "A notebook gathers notes and sources, and gives them to its chat, search and studio." }
                }
                div.dialog-foot {
                    button.btn type="button" data-close { "Cancel" }
                    button.btn.btn-primary type="submit" { "Create" }
                }
            }
        }
    }
}

/// The "⋯" menu on a folder row (a notebook, as the Notes view presents
/// them): rename it, move it between groups, delete it. Notes also filed in
/// other folders survive a delete, and the confirmation says exactly what
/// goes: notes only here, meetings, chats.
fn folder_menu(nb: &NotebookSummary, notes_in: usize, exclusive: usize, library: &Library) -> Markup {
    let groups = &library.folders;
    let kept = notes_in.saturating_sub(exclusive);
    let meetings = library.meetings.get(&nb.id).copied().unwrap_or(0);
    let chats = library.chats.get(&nb.id).copied().unwrap_or(0);
    let s = |n: i64| if n == 1 { "" } else { "s" };
    let mut confirm = format!("Delete \"{}\"?", nb.name);
    match (exclusive, kept) {
        (0, 0) => confirm.push_str(" It has no notes."),
        (e, 0) => confirm.push_str(&format!(" Its {e} note{} will be deleted.", s(e as i64))),
        (0, k) => confirm.push_str(&format!(" Its {k} note{} are also in other folders and will be kept.", s(k as i64))),
        (e, k) => confirm.push_str(&format!(
            " {e} note{} only in this folder will be deleted; {k} note{} also in other folders will be kept.",
            s(e as i64),
            s(k as i64)
        )),
    }
    if meetings > 0 {
        confirm.push_str(&format!(" Its {meetings} meeting{} will be deleted.", s(meetings)));
    }
    if chats > 0 {
        confirm.push_str(&format!(" Its {chats} chat session{} will be deleted.", s(chats)));
    }
    confirm.push_str(" This cannot be undone.");
    html! {
        details.nb-menu.ws-row-menu {
            summary.ws-row-more title="Folder actions" aria-label=(format!("Actions for {}", nb.name)) { "⋯" }
            div.nb-menu-body.stack-sm {
                form.stack-sm method="post" action=(format!("/notes/folders/{}/rename", enc(&nb.id))) {
                    div.nb-menu-label { "Rename folder" }
                    div.row.ws-menu-form {
                        input type="text" name="name" value=(nb.name) required aria-label="Folder name" autocomplete="off";
                        button.btn.btn-sm type="submit" { "Save" }
                    }
                }
                form.stack-sm method="post" action=(format!("/notes/folders/{}/move", enc(&nb.id))) {
                    div.nb-menu-label { "Move to group" }
                    div.nn-file-list {
                        @if nb.folder_id.is_some() {
                            button.btn.btn-sm type="submit" name="folder_id" value="" { "No group" }
                        }
                        @for group in groups {
                            @if nb.folder_id.as_deref() != Some(group.id.as_str()) {
                                button.btn.btn-sm type="submit" name="folder_id" value=(group.id) { (group.name) }
                            }
                        }
                    }
                    div.row.ws-menu-form {
                        input type="text" name="new_group" placeholder="New group…" aria-label="New group name" autocomplete="off";
                        button.btn.btn-sm type="submit" { "Move" }
                    }
                }
                form method="post" action=(format!("/notes/folders/{}/delete", enc(&nb.id))) data-ws-confirm=(confirm) {
                    button.btn.btn-sm.ws-danger type="submit" { "Delete folder…" }
                }
            }
        }
    }
}

/// The "⋯" menu on a folder group: rename it, or delete it keeping its folders.
fn group_menu(folder: &FolderSummary) -> Markup {
    html! {
        details.nb-menu.ws-row-menu {
            summary.ws-row-more title="Group actions" aria-label=(format!("Actions for {}", folder.name)) { "⋯" }
            div.nb-menu-body.stack-sm {
                form.stack-sm method="post" action=(format!("/notes/groups/{}/rename", enc(&folder.id))) {
                    div.nb-menu-label { "Rename group" }
                    div.row.ws-menu-form {
                        input type="text" name="name" value=(folder.name) required aria-label="Group name" autocomplete="off";
                        button.btn.btn-sm type="submit" { "Save" }
                    }
                }
                form method="post" action=(format!("/notes/groups/{}/delete", enc(&folder.id)))
                    data-ws-confirm=(format!("Delete the group \"{}\"? Its folders are kept and move to the top level.", folder.name)) {
                    button.btn.btn-sm.ws-danger type="submit" { "Delete group" }
                }
            }
        }
    }
}

/// The "⋯" menu on a note in the list: pin, move to another folder, delete.
/// It works for every kind of note: text pages, edgeless canvases and files.
fn note_menu(view: &View, note: &WorkspaceNote, current: bool, all: &[NotebookSummary]) -> Markup {
    // Keep an open note open; otherwise stay on the list.
    let back = view.href(current.then_some(note.id.as_str()));
    let list_back = view.href(None);
    html! {
        details.nb-menu.ws-note-menu {
            summary.ws-row-more title="Note actions" aria-label=(format!("Actions for {}", note.title)) { "⋯" }
            div.nb-menu-body.stack-sm {
                form method="post" action=(format!("/notes/{}/pin", enc(&note.id))) data-note-create {
                    input type="hidden" name="back" value=(back);
                    button.btn.btn-sm type="submit" { (if note.pinned { "Unpin" } else { "Pin to the top" }) }
                }
                form method="post" action=(format!("/notes/{}/move", enc(&note.id))) data-note-create {
                    input type="hidden" name="back" value=(back);
                    div.nb-menu-label { "Move to folder" }
                    div.nn-file-list {
                        @for nb in all {
                            @if !(note.notebook_ids.len() == 1 && note.notebook_ids[0] == nb.id) {
                                button.btn.btn-sm type="submit" name="nb" value=(nb.id) { (nb.name) }
                            }
                        }
                    }
                }
                form method="post" action=(format!("/notes/{}/delete", enc(&note.id))) data-note-create
                    data-ws-confirm="Delete this note? This cannot be undone." {
                    input type="hidden" name="back" value=(list_back);
                    button.btn.btn-sm.ws-danger type="submit" { "Delete note" }
                }
            }
        }
    }
}

fn note_list(library: &Library, view: &View, open: Option<&OpenNote>) -> Markup {
    let shown: Vec<&WorkspaceNote> = library.notes.iter().filter(|n| view.matches(n)).collect();
    let open_id = open.map(|o| o.note.id.as_str());
    // A brand-new draft is not listed until its first save; show it anyway.
    let open_missing = open.filter(|o| !shown.iter().any(|n| n.id == o.note.id));
    let new_target = match view {
        View::Notebook(id) => id.clone(),
        _ => String::new(),
    };
    html! {
        section.panel.ws-list aria-label="Notes" {
            header.panel-head.ws-list-head {
                button.btn.btn-ghost.btn-icon.ws-show-sidebar type="button" data-ws-toggle="sidebar" title="Show sidebar" aria-label="Show sidebar" { (icon("sidebar")) }
                div.ws-list-title {
                    h1 { (view_title(library, view)) }
                    span.faint.small data-ws-count { (shown.len()) " " (if shown.len() == 1 { "note" } else { "notes" }) }
                }
                div.row.ws-list-actions {
                    @if let View::Notebook(id) = view {
                        a.btn.btn-sm href=(format!("/notebooks/{id}")) title="Open the notebook with its sources, chat and studio" { "Open notebook" }
                    }
                    form method="post" action="/notes/new" data-note-create {
                        input type="hidden" name="nb" value=(new_target);
                        button.btn.btn-ghost.btn-icon type="submit" title="New note" aria-label="New note" { (icon("compose")) }
                    }
                }
            }
            div.ws-filter {
                input type="search" placeholder="Filter notes" aria-label="Filter notes" data-ws-filter autocomplete="off";
            }
            @if shown.is_empty() && open_missing.is_none() {
                (super::empty_state("No notes here yet", "Write one with New Note, or drag a note onto a notebook in the sidebar to file it."))
            }
            ol.ws-notes data-ws-notes {
                @if let Some(open) = open_missing {
                    (note_item(view, &WorkspaceNote {
                        id: open.note.id.clone(),
                        title: open.title.clone(),
                        snippet: String::new(),
                        notebook_ids: open.notebook_ids.clone(),
                        notebook_id: open.notebook_id.clone(),
                        notebook_name: open.notebook_name.clone(),
                        updated: open.note.updated.clone(),
                        pinned: open.pinned,
                        tags: Vec::new(),
                        file_kind: None,
                        note_type: open.note.note_type.clone(),
                    }, true, false, &library.notebooks))
                }
                @for note in &shown { (note_item(view, note, open_id == Some(note.id.as_str()), !matches!(view, View::Notebook(_)), &library.notebooks)) }
            }
            p.ws-no-match.faint.small hidden data-ws-no-match { "No notes match." }
        }
    }
}

fn note_item(view: &View, note: &WorkspaceNote, current: bool, show_notebook: bool, all: &[NotebookSummary]) -> Markup {
    let search = format!(
        "{} {} {} {}",
        note.title,
        note.snippet,
        note.tags.iter().map(|t| format!("#{t}")).collect::<Vec<_>>().join(" "),
        note.notebook_name.as_deref().unwrap_or_default()
    )
    .to_lowercase();
    html! {
        li.ws-note-item data-pinned[note.pinned] data-updated=(note.updated) data-search=(search) data-note-id=(note.id) {
            a.ws-note href=(view.href(Some(&note.id))) aria-current=[current.then_some("true")] draggable="true" data-ws-drag=(note.id) {
                div.ws-note-title {
                    @if note.pinned { span.ws-pin-mark title="Pinned" { (icon("pin")) } }
                    span.truncate data-ws-title { (note.title) }
                }
                div.ws-note-line {
                    time.ws-note-date datetime=(note.updated) data-ws-date { (human_date_str(&note.updated)) }
                    span.ws-note-snippet data-ws-snippet { @if note.snippet.is_empty() { "No additional text" } @else { (note.snippet) } }
                }
                @if (show_notebook && note.notebook_name.is_some()) || note.file_kind.is_some() || note.note_type.as_deref() == Some("ai") || !note.tags.is_empty() {
                    div.ws-note-chips {
                        @if show_notebook { @if let Some(name) = &note.notebook_name { span.badge { (name) } } }
                        @if let Some(kind) = &note.file_kind { span.badge.info { (kind) } }
                        @if note.note_type.as_deref() == Some("ai") { span.badge.ok { "AI" } }
                        @for tag in note.tags.iter().take(3) { span.badge.ws-chip-tag { "#" (tag) } }
                    }
                }
            }
            (note_menu(view, note, current, all))
        }
    }
}

/// "In Garden, Kitchen ▾": which notebooks a note is filed in, as a pop-over
/// with a checkbox per notebook and a field for a new one. Changing a box
/// saves at once. Used by the Notes view and by the notebook page.
pub(crate) fn notebooks_menu(note_id: &str, in_ids: &[String], all: &[NotebookSummary], back: &str) -> Markup {
    let names: Vec<&str> = all.iter().filter(|nb| in_ids.contains(&nb.id)).map(|nb| nb.name.as_str()).collect();
    let label = match names.as_slice() {
        [] => "Not in a notebook".to_string(),
        [one] => format!("In {one}"),
        [one, two] => format!("In {one}, {two}"),
        [one, rest @ ..] => format!("In {one} +{}", rest.len()),
    };
    html! {
        details.nb-menu.nn-file-menu {
            summary.btn.btn-sm title="Notebooks this note is in" { (icon("book")) span.truncate { (label) } span.faint aria-hidden="true" { "▾" } }
            div.nb-menu-body.stack-sm {
                form.stack-sm method="post" action=(format!("/notes/{}/notebooks", enc(note_id))) data-note-create data-nn-autosave {
                    input type="hidden" name="back" value=(back);
                    div.nb-menu-label { "File in notebooks" }
                    div.nn-file-list {
                        @for nb in all {
                            label.check {
                                input type="checkbox" name="nb" value=(nb.id) checked[in_ids.contains(&nb.id)];
                                span.truncate { (nb.name) }
                            }
                        }
                    }
                    div.row.nn-file-new {
                        input type="text" name="new_notebook" placeholder="New notebook…" aria-label="New notebook name" autocomplete="off";
                        button.btn.btn-sm.btn-primary type="submit" { "Save" }
                    }
                    p.hint { "A note can live in several notebooks; each one's chat, search and studio can use it." }
                }
            }
        }
    }
}

fn editor_pane(library: &Library, view: &View, open: Option<&OpenNote>, fresh: bool) -> Markup {
    let Some(open) = open else {
        return html! {
            section.nn-page.ws-editor.ws-editor-empty aria-label="Editor" {
                div.ws-welcome {
                    span.pebbles aria-hidden="true" { i {} i {} i {} }
                    h2 { "Pick a note, or start a new one" }
                    p.muted { (library.notes.len()) " notes in your library." }
                    div.row {
                        form method="post" action="/notes/new" data-note-create {
                            input type="hidden" name="nb" value=(match view { View::Notebook(id) => id.as_str(), _ => "" });
                            button.btn.btn-primary type="submit" { "New Note" }
                        }
                        form method="post" action="/notes/daily" data-note-create data-ws-daily {
                            input type="hidden" name="date" value="";
                            button.btn type="submit" { "Today's note" }
                        }
                    }
                    dl.ws-keys {
                        div { dt { kbd { "Ctrl" } kbd { "K" } } dd { "Search and jump anywhere" } }
                        div { dt { kbd { "Ctrl" } kbd { "Alt" } kbd { "N" } } dd { "New note" } }
                        div { dt { kbd { "Ctrl" } kbd { "Alt" } kbd { "D" } } dd { "Today's note" } }
                        div { dt { kbd { "Ctrl" } kbd { "\\" } } dd { "Collapse or expand the sidebar" } }
                        div { dt { kbd { "Ctrl" } kbd { "." } } dd { "Focus mode" } }
                        div { dt { kbd { "@" } } dd { "Link another note, in the editor" } }
                        div { dt { "Drag" } dd { "a note onto a notebook to file it" } }
                    }
                }
            }
        };
    };
    let note = &open.note;
    let back = view.href(None);
    let here = view.href(Some(&note.id));
    let note_href = format!("{}{}note=__ID__", back, if back.contains('?') { "&" } else { "?" });
    html! {
        section.nn-page.ws-editor aria-label="Editor" data-note-href=(note_href) {
            header.nn-page-bar {
                a.btn.btn-ghost.btn-icon.ws-back href=(back) title="Back to the list" aria-label="Back to the list" { (icon("back")) }
                div.nn-page-bar-main { (notebooks_menu(&note.id, &open.notebook_ids, &library.notebooks, &here)) }
                div.row.nn-page-actions {
                    form method="post" action=(format!("/notes/{}/pin", enc(&note.id))) data-note-create {
                        input type="hidden" name="back" value=(here);
                        button.btn.btn-ghost.btn-icon type="submit" aria-pressed=(open.pinned.to_string())
                            title=(if open.pinned { "Unpin" } else { "Pin to the top" }) aria-label=(if open.pinned { "Unpin" } else { "Pin" }) { (icon("pin")) }
                    }
                    @if let Some(nb) = &open.notebook_id {
                        a.btn.btn-ghost.btn-icon href=(format!("/notebooks/{nb}?note={}", enc(&note.id))) title="Open in its notebook, beside sources and chat" aria-label="Open in notebook" { (icon("open")) }
                    }
                    button.btn.btn-ghost.btn-icon type="button" data-ws-toggle="focus" title="Focus mode (Ctrl+.)" aria-label="Focus mode" { (icon("focus")) }
                    form method="post" action=(format!("/notes/{}/delete", enc(&note.id))) data-note-create data-ws-confirm="Delete this note? This cannot be undone." {
                        input type="hidden" name="back" value=(back);
                        button.btn.btn-ghost.btn-icon.ws-danger type="submit" title="Delete note" aria-label="Delete note" { (icon("trash")) }
                    }
                }
            }
            div.nn-page-body {
                @match (&open.notebook_id, note.attachment.is_some()) {
                    (Some(nb), false) => { (note_editor(nb, note, fresh)) }
                    (Some(nb), true) => { (note_body(nb, note, false)) }
                    (None, _) => {
                        h1.nn-page-title { (open.title) }
                        (markdown(note.content.as_deref().unwrap_or_default()))
                        p.flash.warn.small { "This note is not in a notebook yet, so it opens read-only. File it with the notebook menu above to edit it." }
                    }
                }
                footer.nn-page-foot.faint.small {
                    span { "Edited " time data-ws-date datetime=(note.updated) { (human_date_str(&note.updated)) } }
                    span data-ws-words {}
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

async fn palette(State(state): State<AppState>) -> Result<axum::Json<workspace::PaletteIndex>, AppError> {
    Ok(axum::Json(workspace::palette_index(&state).await?))
}

#[derive(Debug, Default, Deserialize)]
struct NewNoteForm {
    #[serde(default)]
    nb: String,
    #[serde(default)]
    title: String,
}

/// A new note in the given notebook (Quick Notes when none), opened in the
/// workspace. With a title (from the palette's "Create note"), it is created
/// at once with that title; otherwise it is a blank draft, as everywhere else.
async fn new_note(State(state): State<AppState>, Form(form): Form<NewNoteForm>) -> WebResult<Redirect> {
    let nb = form.nb.trim();
    let (notebook_id, view) = if nb.is_empty() {
        (note_docs::quick_notes_notebook(&state).await?.to_string(), View::All)
    } else {
        notebooks::get_notebook(&state, nb).await?;
        (nb.to_string(), View::Notebook(nb.to_string()))
    };
    let title = form.title.trim();
    if title.is_empty() {
        let id = note_docs::create_draft(&state, &notebook_id).await?;
        Ok(Redirect::to(&format!("{}&new=1", view.href(Some(&id)))))
    } else {
        let note = note_docs::create_linked_note(&state, &notebook_id, title).await?;
        Ok(Redirect::to(&view.href(Some(&note.id))))
    }
}

#[derive(Debug, Default, Deserialize)]
struct DailyForm {
    /// The writer's local date; the server's (UTC) date when absent.
    #[serde(default)]
    date: String,
}

async fn daily_note(State(state): State<AppState>, Form(form): Form<DailyForm>) -> WebResult<Redirect> {
    let date = if form.date.trim().is_empty() { chrono::Utc::now().date_naive() } else { workspace::parse_date(&form.date)? };
    let id = workspace::daily_note(&state, date).await?;
    Ok(Redirect::to(&View::Notebook(workspace::DAILY_NOTES_ID.into()).href(Some(&id))))
}

#[derive(Debug, Default, Deserialize)]
struct BackForm {
    #[serde(default)]
    back: Option<String>,
}

async fn toggle_pin(State(state): State<AppState>, headers: HeaderMap, Path(note_id): Path<String>, Form(form): Form<BackForm>) -> WebResult<Response> {
    let pinned = workspace::toggle_pin(&state, &note_id).await?;
    let back = safe_path_or(form.back.as_deref(), &View::All.href(Some(&note_id)));
    Ok(super::with_toast(super::redirect(&headers, &back), if pinned { "Pinned" } else { "Unpinned" }))
}

async fn delete_note(State(state): State<AppState>, headers: HeaderMap, Path(note_id): Path<String>, Form(form): Form<BackForm>) -> WebResult<Response> {
    notes::delete_note(&state, &note_id).await?;
    workspace::set_pinned(&state, &note_id, false).await?;
    let back = safe_path_or(form.back.as_deref(), "/notes");
    Ok(super::with_toast(super::redirect(&headers, &back), "Note deleted"))
}

#[derive(Debug, Default, Deserialize)]
struct NotebookNameForm {
    #[serde(default)]
    name: String,
}

/// The sidebar's "+ New notebook": create it and show it.
async fn new_notebook(State(state): State<AppState>, Form(form): Form<NotebookNameForm>) -> WebResult<Redirect> {
    let created = notebooks::create_notebook(&state, &NewNotebook { name: form.name.trim().to_string(), description: String::new() }).await?;
    Ok(Redirect::to(&View::Notebook(created.id).href(None)))
}

/// The notebook menu: file the note in exactly the ticked notebooks, plus a
/// new one when a name was typed.
async fn set_notebooks(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(note_id): Path<String>,
    Form(fields): Form<Vec<(String, String)>>,
) -> WebResult<Response> {
    let mut wanted: Vec<String> = fields.iter().filter(|(k, _)| k == "nb").map(|(_, v)| v.clone()).collect();
    let new_name = fields.iter().find(|(k, _)| k == "new_notebook").map(|(_, v)| v.trim()).unwrap_or_default();
    if !new_name.is_empty() {
        let created = notebooks::create_notebook(&state, &NewNotebook { name: new_name.to_string(), description: String::new() }).await?;
        wanted.push(created.id);
    }
    let change = workspace::set_note_notebooks(&state, &note_id, &wanted).await?;
    let back = fields.iter().find(|(k, _)| k == "back").map(|(_, v)| v.as_str());
    let back = safe_path_or(back, &View::All.href(Some(&note_id)));
    let toast = match (change.added.len(), change.removed.len()) {
        (0, 0) => "No change",
        (_, 0) => "Added to notebook",
        (0, _) => "Removed from notebook",
        _ => "Notebooks updated",
    };
    Ok(super::with_toast(super::redirect(&headers, &back), toast))
}

#[derive(Debug, Default, Deserialize)]
struct FileForm {
    #[serde(default)]
    nb: String,
}

/// A note dropped on a notebook in the sidebar. Answers JSON for the script.
async fn file_note(
    State(state): State<AppState>,
    Path(note_id): Path<String>,
    Form(form): Form<FileForm>,
) -> Result<axum::Json<serde_json::Value>, AppError> {
    let notebook = notebooks::get_notebook(&state, form.nb.trim()).await?;
    let ids = workspace::file_note(&state, &note_id, &notebook.id).await?;
    Ok(axum::Json(serde_json::json!({ "notebook": notebook.name, "notebooks": ids })))
}

#[derive(Debug, Default, Deserialize)]
struct MoveNoteForm {
    #[serde(default)]
    nb: String,
    #[serde(default)]
    back: Option<String>,
}

/// The note menu's "Move to folder": file the note in exactly that folder.
async fn move_note(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(note_id): Path<String>,
    Form(form): Form<MoveNoteForm>,
) -> WebResult<Response> {
    let notebook = notebooks::get_notebook(&state, form.nb.trim()).await?;
    workspace::set_note_notebooks(&state, &note_id, std::slice::from_ref(&notebook.id)).await?;
    let back = safe_path_or(form.back.as_deref(), "/notes");
    Ok(super::with_toast(super::redirect(&headers, &back), &format!("Moved to {}", notebook.name)))
}

#[derive(Debug, Default, Deserialize)]
struct NameForm {
    #[serde(default)]
    name: String,
}

/// Rename a folder (a notebook) from its sidebar menu.
async fn rename_folder(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Form(form): Form<NameForm>,
) -> WebResult<Response> {
    let name = form.name.trim();
    if name.is_empty() {
        return Err(AppError::invalid("Folder name is required").into());
    }
    notebooks::update_notebook(&state, &id, &NotebookUpdate { name: Some(name.to_string()), ..Default::default() }).await?;
    Ok(super::with_toast(super::redirect(&headers, &View::Notebook(id).href(None)), "Folder renamed"))
}

#[derive(Debug, Default, Deserialize)]
struct MoveFolderForm {
    /// The group to move into; blank takes the folder out of its group.
    #[serde(default)]
    folder_id: Option<String>,
    /// A new group to create and move into; wins over `folder_id`.
    #[serde(default)]
    new_group: String,
}

async fn move_folder(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Form(form): Form<MoveFolderForm>,
) -> WebResult<Response> {
    let target = match form.new_group.trim() {
        "" => form.folder_id.unwrap_or_default().trim().to_string(),
        name => notebooks::create_folder(&state, &FolderInput { name: name.to_string(), section_id: None }).await?.id,
    };
    let update = NotebookUpdate { folder_id: Some(Some(target).filter(|t| !t.is_empty())), ..Default::default() };
    notebooks::update_notebook(&state, &id, &update).await?;
    Ok(super::with_toast(super::redirect(&headers, &View::Notebook(id).href(None)), "Folder moved"))
}

/// Delete a folder (a notebook), as the confirmation said: notes also filed
/// in other folders are kept and only unfiled from this one; notes that live
/// nowhere else are deleted with it. Sources are kept, only unlinked.
async fn delete_folder(State(state): State<AppState>, headers: HeaderMap, Path(id): Path<String>) -> WebResult<Response> {
    let deleted = workspace::delete_folder(&state, &id).await?;
    let s = |n: i64| if n == 1 { "" } else { "s" };
    let mut toast = match deleted.deleted_notes {
        0 => "Folder deleted".to_string(),
        n => format!("Folder deleted, with {n} note{}", s(n)),
    };
    if deleted.kept_notes > 0 {
        toast.push_str(&format!("; {} note{} kept in other folders", deleted.kept_notes, s(deleted.kept_notes)));
    }
    Ok(super::with_toast(super::redirect(&headers, "/notes"), &toast))
}

async fn rename_group(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Form(form): Form<NameForm>,
) -> WebResult<Response> {
    notebooks::update_folder(&state, &id, &FolderInput { name: form.name, section_id: None }).await?;
    Ok(super::with_toast(super::redirect(&headers, "/notes"), "Group renamed"))
}

/// Delete a folder group. Its folders are kept and become top-level.
async fn delete_group(State(state): State<AppState>, headers: HeaderMap, Path(id): Path<String>) -> WebResult<Response> {
    notebooks::delete_folder(&state, &id).await?;
    Ok(super::with_toast(super::redirect(&headers, "/notes"), "Group deleted; its folders are kept"))
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{header, Method, Request, StatusCode};

    use super::*;
    use crate::services::notes::NewNote;
    use crate::testing::TestApp;

    async fn form(app: &TestApp, path: &str, body: &str) -> (StatusCode, String) {
        let response = app
            .send(
                Request::builder()
                    .method(Method::POST)
                    .uri(path)
                    .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await;
        let location = response.headers().get(header::LOCATION).map(|v| v.to_str().unwrap().to_string()).unwrap_or_default();
        (response.status(), location)
    }

    #[test]
    fn views_round_trip_through_their_links() {
        let q = |nb: Option<&str>, tag: Option<&str>, pinned: Option<&str>| ViewQuery {
            nb: nb.map(String::from),
            tag: tag.map(String::from),
            pinned: pinned.map(String::from),
            ..ViewQuery::default()
        };
        assert_eq!(q(None, None, None).view(), View::All);
        assert_eq!(q(None, None, Some("1")).view(), View::Pinned);
        assert_eq!(q(None, None, Some("0")).view(), View::All);
        assert_eq!(q(None, Some("#Rust"), None).view(), View::Tag("rust".into()));
        assert_eq!(q(Some("notebook:abc"), Some("x"), None).view(), View::Notebook("notebook:abc".into()));
        assert_eq!(View::Notebook("notebook:abc".into()).href(Some("note:1")), "/notes?nb=notebook%3Aabc&note=note%3A1");
        assert_eq!(View::All.href(None), "/notes");
        assert_eq!(View::All.href(Some("note:1")), "/notes?note=note%3A1");
        let tagged = WorkspaceNote {
            id: "note:1".into(), title: "t".into(), snippet: String::new(), notebook_ids: vec![], notebook_id: None, notebook_name: None,
            updated: String::new(), pinned: false, tags: vec!["rust/async".into()], file_kind: None, note_type: None,
        };
        assert!(View::Tag("rust".into()).matches(&tagged));
        assert!(!View::Tag("ru".into()).matches(&tagged));
    }

    #[tokio::test]
    async fn workspace_lists_opens_pins_and_deletes_notes() {
        let app = TestApp::new().await;
        let (_, nb) = app.post("/api/notebooks", serde_json::json!({ "name": "Garden", "description": "" })).await;
        let nb_id = nb["id"].as_str().unwrap().to_string();
        let note = notes::create_note(
            &app.state,
            &NewNote { title: Some("Tomatoes".into()), content: "Plant in May #garden".into(), notebook_id: Some(nb_id.clone()), ..Default::default() },
        )
        .await
        .unwrap();

        let (status, body) = app.text(Method::GET, "/notes", None).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("All Notes") && body.contains("Tomatoes") && body.contains("#garden"), "{body}");
        assert!(body.contains("data-palette-open"));

        // Open in place: the rich editor, saving through the note's notebook.
        let (status, body) = app.text(Method::GET, &format!("/notes?note={}", enc(&note.id)), None).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("data-nb-rich"), "{body}");
        assert!(body.contains(&format!("/notebooks/{nb_id}/notes/{}/editor", note.id)));
        assert!(body.contains("data-note-href=\"/notes?note=__ID__\""), "{body}");
        // The notebook menu shows where it is filed, with a box per notebook.
        assert!(body.contains("In Garden") && body.contains(&format!("value=\"{nb_id}\" checked")), "{body}");
        // The Studio is one full-width door, not a list of links.
        assert!(body.contains("ws-studio") && body.contains(">Studio</span>"), "{body}");
        assert!(!body.contains("href=\"/podcasts\""), "{body}");

        // Views filter the list.
        let (_, body) = app.text(Method::GET, "/notes?tag=garden", None).await;
        assert!(body.contains("Tomatoes"));
        let (_, body) = app.text(Method::GET, "/notes?pinned=1", None).await;
        assert!(!body.contains("data-note-id=\"note:"), "{body}");

        let (status, location) = form(&app, &format!("/notes/{}/pin", enc(&note.id)), "back=%2Fnotes%3Fpinned%3D1").await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(location, "/notes?pinned=1");
        let (_, body) = app.text(Method::GET, "/notes?pinned=1", None).await;
        assert!(body.contains("Tomatoes"));

        let (status, location) = form(&app, &format!("/notes/{}/delete", enc(&note.id)), "back=%2F%2Fevil.example").await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(location, "/notes");
        assert!(notes::get_note(&app.state, &note.id).await.is_err());
        assert!(workspace::pinned_ids(&app.state).await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn new_notes_daily_notes_and_the_palette() {
        let app = TestApp::new().await;
        // A blank draft in Quick Notes, opened fresh.
        let (status, location) = form(&app, "/notes/new", "nb=").await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert!(location.starts_with("/notes?note=note%3A") && location.ends_with("&new=1"), "{location}");
        let (status, body) = app.text(Method::GET, &location, None).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("data-nb-fresh") && body.contains("New Note"), "{body}");

        // A titled note from the palette is saved at once.
        let (_, location) = form(&app, "/notes/new", "title=Shopping+list").await;
        let (_, body) = app.text(Method::GET, &location, None).await;
        assert!(body.contains("Shopping list"));

        // Unknown notebooks are refused.
        let (status, _) = form(&app, "/notes/new", "nb=notebook%3Anope").await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        let (status, location) = form(&app, "/notes/daily", "date=2026-11-16").await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert!(location.starts_with("/notes?nb=notebook%3Adaily_notes&note="), "{location}");
        let (_, again) = form(&app, "/notes/daily", "date=2026-11-16").await;
        assert_eq!(location, again);
        let (status, _) = form(&app, "/notes/daily", "date=tomorrow").await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let (status, index) = app.get("/notes/palette").await;
        assert_eq!(status, StatusCode::OK);
        let titles: Vec<&str> = index["notes"].as_array().unwrap().iter().filter_map(|n| n["title"].as_str()).collect();
        assert!(titles.contains(&"Shopping list") && titles.contains(&"2026-11-16"), "{titles:?}");

        // The old note route still redirects into the notebook page.
        let (_, list) = app.get("/notes/palette").await;
        let id = list["notes"][0]["id"].as_str().unwrap().to_string();
        let response = app.send(Request::builder().uri(format!("/notes/{id}")).body(Body::empty()).unwrap()).await;
        assert_eq!(response.status(), StatusCode::SEE_OTHER);
        assert!(response.headers()[header::LOCATION].to_str().unwrap().starts_with("/notebooks/"));
    }

    #[tokio::test]
    async fn notes_are_filed_in_notebooks_from_the_menu_and_by_dropping() {
        let app = TestApp::new().await;
        let (_, nb) = app.post("/api/notebooks", serde_json::json!({ "name": "Garden", "description": "" })).await;
        let nb_id = nb["id"].as_str().unwrap().to_string();
        // A Quick Note, as the sidebar's New Note makes one.
        let (_, location) = form(&app, "/notes/new", "title=Seeds").await;
        let note_id = location.split("note=").nth(1).unwrap().replace("%3A", ":");
        assert_eq!(workspace::note_notebook_ids(&app.state, &note_id).await.unwrap(), vec![workspace::QUICK_NOTES_ID]);

        // Dropping it on Garden files it there and out of the Quick Notes inbox.
        let response = app
            .send(
                Request::builder()
                    .method(Method::POST)
                    .uri(format!("/notes/{}/file", enc(&note_id)))
                    .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
                    .body(Body::from(format!("nb={}", enc(&nb_id))))
                    .unwrap(),
            )
            .await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(workspace::note_notebook_ids(&app.state, &note_id).await.unwrap(), vec![nb_id.clone()]);

        // The menu adds a brand-new notebook alongside, keeping the ticked one.
        let body = format!("back=%2Fnotes&nb={}&new_notebook=Kitchen", enc(&nb_id));
        let (status, location) = form(&app, &format!("/notes/{}/notebooks", enc(&note_id)), &body).await;
        assert_eq!((status, location.as_str()), (StatusCode::SEE_OTHER, "/notes"));
        let ids = workspace::note_notebook_ids(&app.state, &note_id).await.unwrap();
        assert_eq!(ids.len(), 2);
        // Unticking everything is refused: a note keeps one notebook.
        let (status, _) = form(&app, &format!("/notes/{}/notebooks", enc(&note_id)), "back=%2Fnotes").await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        // The sidebar's + makes a notebook and opens it.
        let (status, location) = form(&app, "/notes/notebooks", "name=Reading").await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert!(location.starts_with("/notes?nb=notebook%3A"), "{location}");
        let (_, page) = app.text(Method::GET, &location, None).await;
        assert!(page.contains("Reading") && page.contains("Open notebook"), "{page}");
    }

    #[tokio::test]
    async fn notes_move_between_folders_from_the_menu() {
        let app = TestApp::new().await;
        let (_, a) = app.post("/api/notebooks", serde_json::json!({ "name": "Garden", "description": "" })).await;
        let (_, b) = app.post("/api/notebooks", serde_json::json!({ "name": "Kitchen", "description": "" })).await;
        let (a, b) = (a["id"].as_str().unwrap().to_string(), b["id"].as_str().unwrap().to_string());
        let note = notes::create_note(
            &app.state,
            &NewNote { title: Some("Seeds".into()), content: "x".into(), notebook_id: Some(a.clone()), ..Default::default() },
        )
        .await
        .unwrap();

        // The list shows a per-note menu with Move and Delete.
        let (_, body) = app.text(Method::GET, "/notes", None).await;
        assert!(body.contains("ws-note-menu") && body.contains("Move to folder"), "{body}");

        let (status, location) = form(&app, &format!("/notes/{}/move", enc(&note.id)), &format!("nb={}&back=%2Fnotes", enc(&b))).await;
        assert_eq!((status, location.as_str()), (StatusCode::SEE_OTHER, "/notes"));
        assert_eq!(workspace::note_notebook_ids(&app.state, &note.id).await.unwrap(), vec![b.clone()]);
        // A move replaces every filing, even when the note was in several.
        workspace::set_note_notebooks(&app.state, &note.id, &[a.clone(), b.clone()]).await.unwrap();
        form(&app, &format!("/notes/{}/move", enc(&note.id)), &format!("nb={}", enc(&a))).await;
        assert_eq!(workspace::note_notebook_ids(&app.state, &note.id).await.unwrap(), vec![a.clone()]);
        let (status, _) = form(&app, &format!("/notes/{}/move", enc(&note.id)), "nb=notebook%3Anope").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn folders_rename_move_and_delete_from_the_sidebar() {
        let app = TestApp::new().await;
        let (_, nb) = app.post("/api/notebooks", serde_json::json!({ "name": "Garden", "description": "" })).await;
        let nb_id = nb["id"].as_str().unwrap().to_string();
        notes::create_note(
            &app.state,
            &NewNote { title: Some("Seeds".into()), content: "x".into(), notebook_id: Some(nb_id.clone()), ..Default::default() },
        )
        .await
        .unwrap();

        // Rename.
        let (status, location) = form(&app, &format!("/notes/folders/{}/rename", enc(&nb_id)), "name=Allotment").await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert!(location.starts_with("/notes?nb="), "{location}");
        assert_eq!(notebooks::get_notebook(&app.state, &nb_id).await.unwrap().name, "Allotment");
        let (status, _) = form(&app, &format!("/notes/folders/{}/rename", enc(&nb_id)), "name=").await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        // Move into a brand-new group, then out of it.
        form(&app, &format!("/notes/folders/{}/move", enc(&nb_id)), "new_group=Life").await;
        let folders = notebooks::list_folders(&app.state).await.unwrap();
        let group = folders.iter().find(|f| f.name == "Life").expect("group created");
        assert_eq!(notebooks::get_notebook(&app.state, &nb_id).await.unwrap().folder_id.as_deref(), Some(group.id.as_str()));
        form(&app, &format!("/notes/folders/{}/move", enc(&nb_id)), "folder_id=").await;
        assert_eq!(notebooks::get_notebook(&app.state, &nb_id).await.unwrap().folder_id, None);

        // Group rename and delete (folders are kept).
        let (status, _) = form(&app, &format!("/notes/groups/{}/rename", enc(&group.id)), "name=Living").await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        let (status, location) = form(&app, &format!("/notes/groups/{}/delete", enc(&group.id)), "").await;
        assert_eq!((status, location.as_str()), (StatusCode::SEE_OTHER, "/notes"));
        assert!(notebooks::list_folders(&app.state).await.unwrap().is_empty());

        // Delete the folder: a note also filed elsewhere is kept, the rest go.
        let (_, other) = app.post("/api/notebooks", serde_json::json!({ "name": "Keep", "description": "" })).await;
        let other_id = other["id"].as_str().unwrap().to_string();
        let shared = notes::create_note(
            &app.state,
            &NewNote { title: Some("Shared".into()), content: "x".into(), notebook_id: Some(nb_id.clone()), ..Default::default() },
        )
        .await
        .unwrap();
        workspace::file_note(&app.state, &shared.id, &other_id).await.unwrap();
        let (status, location) = form(&app, &format!("/notes/folders/{}/delete", enc(&nb_id)), "").await;
        assert_eq!((status, location.as_str()), (StatusCode::SEE_OTHER, "/notes"));
        assert!(notebooks::get_notebook(&app.state, &nb_id).await.is_err());
        let remaining = workspace::list_workspace_notes(&app.state).await.unwrap();
        assert_eq!(remaining.len(), 1, "the shared note survives, Seeds went with the folder");
        assert_eq!(remaining[0].id, shared.id);
        assert_eq!(remaining[0].notebook_ids, vec![other_id]);
    }
}
