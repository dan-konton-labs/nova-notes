//! Notebook pages: the notebook list (with sections and folders) and the
//! notebook page itself, with its sources, notes and chat panes and the
//! studio panel.
//!
//! Every handler calls the same services as the JSON API. Forms post to the
//! routes below and answer with a fragment for htmx or a redirect otherwise,
//! so the pages also work without JavaScript.
//!
//! Chat context: each source and note row carries a `<select>` that belongs
//! to the chat form (`form="nb-chat-form"`), named `ctx_<record id>`. The
//! send handler turns those into the `context_config` the old frontend built
//! (`insights`, `full content` or `not in` per item).

use axum::extract::{Multipart, Path, Query, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::{Form, Router};
use maud::{html, Markup, PreEscaped};
use serde::Deserialize;
use serde_json::json;

use super::{
    empty_state, fragment_or_redirect, human_date_str, is_htmx, markdown, page, page_head, read_multipart, safe_path_or, status_badge,
    with_trigger, Fields, PageContext, UploadedFile, WebResult,
};
use crate::error::AppError;
use crate::services::chat::{self, ChatSessionResponse, CreateSessionRequest, ExecuteChatRequest, UpdateSessionRequest};
use crate::services::context::{build_context, BuildContextRequest, ContextConfig, ItemStatuses};
use crate::services::notebooks::{
    self, FolderInput, FolderSummary, NewNotebook, NotebookQuery, NotebookSummary, NotebookUpdate, SectionInput, SectionSummary,
};
use crate::api::ApiJson;
use crate::services::note_docs::{self, NoteDocView, SaveNoteDoc};
use crate::services::notes::{self, NewNote, NoteUpdate, NoteUpload, NoteView};
use crate::services::sources::{self, NewSource, SourceListItem, SourceListQuery, SourceUpload};
use crate::state::AppState;

/// Sources shown per page in the sources pane.
const SOURCE_PAGE: i64 = 50;
/// Extensions a note can be created from (the notes service's allow-list).
const NOTE_FILE_ACCEPT: &str = ".pdf,.html,.htm,.docx,.txt,.md,.markdown";

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/notebooks", get(list_page).post(create_notebook))
        .route("/notebooks/{id}", get(notebook_page))
        .route("/notebooks/{id}/rename", post(rename_notebook))
        .route("/notebooks/{id}/archive", post(archive_notebook))
        .route("/notebooks/{id}/move", post(move_notebook))
        .route("/notebooks/{id}/delete", get(delete_page).post(delete_notebook))
        .route("/notebook-folders", post(create_folder))
        .route("/notebook-folders/{id}/rename", post(rename_folder))
        .route("/notebook-folders/{id}/delete", post(delete_folder))
        .route("/notebook-sections", post(create_section))
        .route("/notebook-sections/{id}/rename", post(rename_section))
        .route("/notebook-sections/{id}/delete", post(delete_section))
        // Sources pane
        .route("/notebooks/{id}/sources", get(sources_fragment).post(add_sources))
        .route("/notebooks/{id}/sources/existing", get(existing_sources_page).post(link_sources))
        .route("/notebooks/{id}/sources/{source_id}/row", get(source_row_fragment))
        .route("/notebooks/{id}/sources/{source_id}/remove", post(remove_source))
        // Notes pane
        .route("/notebooks/{id}/transport", get(transport_page).post(do_transport))
        .route("/notebooks/{id}/notes", get(notes_fragment).post(create_note))
        .route("/notebooks/{id}/notes/new", get(new_note_page).post(write_note))
        .route("/notebooks/{id}/notes/upload", post(upload_notes))
        .route("/notebooks/{id}/notes/{note_id}", get(note_page).post(update_note))
        .route("/notebooks/{id}/notes/{note_id}/delete", post(delete_note))
        .route("/notes/quick", post(quick_note))
        .route("/notes/{note_id}", get(note_redirect))
        // Rich note editor (JSON for static/js/notebooks.js and the editor bundle)
        .route("/notebooks/{id}/notes/editor", get(new_note_doc).post(create_note_doc))
        .route("/notebooks/{id}/notes/{note_id}/editor", get(note_doc).post(update_note_doc))
        .route("/notebooks/{id}/notes/link", post(create_linked_note))
        .route("/note-blobs/{key}", get(get_note_blob).put(put_note_blob))
        // Chat pane
        .route("/notebooks/{id}/chat", get(chat_fragment))
        .route("/notebooks/{id}/chat/send", post(send_message))
        .route("/notebooks/{id}/chat/context", post(context_stats))
        .route("/notebooks/{id}/chat/save-note", post(save_message_as_note))
        .route("/notebooks/{id}/chat/sessions", post(create_session))
        .route("/notebooks/{id}/chat/sessions/{session_id}/rename", post(rename_session))
        .route("/notebooks/{id}/chat/sessions/{session_id}/delete", post(delete_session))
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn notebook_href(id: &str) -> String {
    format!("/notebooks/{id}")
}

/// Percent-encode a query-string value.
fn encode(value: &str) -> String {
    crate::api::auth::encode_query_value(value)
}

fn dialog(id: &str, title: &str, body: Markup) -> Markup {
    html! {
        dialog.nb-dialog id=(id) aria-label=(title) {
            div.dialog-head {
                h3 { (title) }
                button.btn.btn-ghost.btn-sm type="button" data-close aria-label="Close" { "Close" }
            }
            (body)
        }
    }
}

// ---------------------------------------------------------------------------
// Notebook list
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
struct ListQuery {
    folder: Option<String>,
    section: Option<String>,
    q: Option<String>,
}

impl ListQuery {
    /// The URL of this view, for returning to it after a form post.
    fn href(&self) -> String {
        let mut parts = Vec::new();
        if let Some(folder) = self.folder.as_deref().filter(|f| !f.is_empty()) {
            parts.push(format!("folder={}", encode(folder)));
        } else if let Some(section) = self.section.as_deref().filter(|s| !s.is_empty()) {
            parts.push(format!("section={}", encode(section)));
        }
        if parts.is_empty() {
            "/notebooks".to_string()
        } else {
            format!("/notebooks?{}", parts.join("&"))
        }
    }
}

fn matches_query(name: &str, query: &str) -> bool {
    query.is_empty() || name.to_lowercase().contains(query)
}

async fn list_page(State(state): State<AppState>, Query(query): Query<ListQuery>) -> WebResult<Markup> {
    let active = notebooks::list_notebooks(&state, &NotebookQuery { archived: Some(false), ..Default::default() }).await?;
    let archived = notebooks::list_notebooks(&state, &NotebookQuery { archived: Some(true), ..Default::default() }).await?;
    let folders = notebooks::list_folders(&state).await?;
    let sections = notebooks::list_sections(&state).await?;
    let search = query.q.as_deref().unwrap_or_default().trim().to_lowercase();
    let searching = !search.is_empty();
    let back = query.href();

    let current_folder = query.folder.as_deref().and_then(|id| folders.iter().find(|f| f.id == id));
    let current_section_id = match current_folder {
        Some(folder) => folder.section_id.clone(),
        None => query.section.clone(),
    };
    let current_section = current_section_id.as_deref().and_then(|id| sections.iter().find(|s| s.id == id));

    let ctx = PageContext::new(&state, "/notebooks");
    let discover = super::discover::discover_controls(&state).await?;
    let actions = html! {
        form.nb-search method="get" action="/notebooks" role="search" {
            @if let Some(folder) = &query.folder { input type="hidden" name="folder" value=(folder); }
            @if let Some(section) = &query.section { input type="hidden" name="section" value=(section); }
            input type="search" name="q" value=[query.q.as_deref()] placeholder="Search notebooks..." aria-label="Search notebooks";
        }
        button.btn.btn-primary type="button" data-open="#nb-create-notebook" { "New Notebook" }
        button.btn type="button" data-open="#nb-create-section" { "New section" }
        button.btn type="button" data-open="#nb-create-folder" { "New folder" }
        (discover)
    };

    let body = html! {
        (page_head("Notebooks", None, actions))
        div.stack.nb-list {
            @if let Some(folder) = current_folder {
                (breadcrumb(current_section, Some(&folder.name)))
                @let inside: Vec<&NotebookSummary> = active.iter()
                    .filter(|nb| nb.folder_id.as_deref() == Some(folder.id.as_str()) && matches_query(&nb.name, &search)).collect();
                (notebook_grid(&folder.name, &inside, &folders, &back,
                    if searching { empty_state("No matches", "Try a different search.") }
                    else { empty_state("This folder is empty", "Move notebooks here from the notebook menu.") }))
                @let archived_here: Vec<&NotebookSummary> = archived.iter()
                    .filter(|nb| nb.folder_id.as_deref() == Some(folder.id.as_str()) && matches_query(&nb.name, &search)).collect();
                @if !archived_here.is_empty() { (archived_list(&archived_here, &folders, &back)) }
            } @else if let Some(section) = current_section {
                (breadcrumb(None, Some(&section.name)))
                @let inside: Vec<&FolderSummary> = folders.iter()
                    .filter(|f| f.section_id.as_deref() == Some(section.id.as_str()) && matches_query(&f.name, &search)).collect();
                @if inside.is_empty() {
                    @if searching { (empty_state("No matches", "Try a different search.")) }
                    @else { (empty_state("This section is empty", "Use New folder to add a folder here, or move an existing folder into this section from its menu.")) }
                } @else {
                    section.stack-sm {
                        h2 { (section.name) " " span.faint.small { "(" (inside.len()) ")" } }
                        div.grid { @for folder in &inside { (folder_card(folder, &sections, &back)) } }
                    }
                }
            } @else {
                @if !searching { (recently_viewed(&state).await) }
                @let shown_sections: Vec<&SectionSummary> = sections.iter().filter(|s| matches_query(&s.name, &search)).collect();
                @if !shown_sections.is_empty() {
                    section.stack-sm {
                        h2 { "Sections " span.faint.small { "(" (shown_sections.len()) ")" } }
                        div.grid { @for section in &shown_sections { (section_card(section, &back)) } }
                    }
                }
                // At the top level: folders outside any (existing) section; a search looks in all.
                @let root_folders: Vec<&FolderSummary> = folders.iter().filter(|f| {
                    matches_query(&f.name, &search)
                        && (searching || f.section_id.as_deref().is_none_or(|s| !sections.iter().any(|sec| sec.id == s)))
                }).collect();
                @if !root_folders.is_empty() {
                    section.stack-sm {
                        h2 { "Folders " span.faint.small { "(" (root_folders.len()) ")" } }
                        div.grid { @for folder in &root_folders { (folder_card(folder, &sections, &back)) } }
                    }
                }
                @let loose: Vec<&NotebookSummary> = active.iter().filter(|nb| {
                    matches_query(&nb.name, &search)
                        && (searching || nb.folder_id.as_deref().is_none_or(|f| !folders.iter().any(|folder| folder.id == f)))
                }).collect();
                @let grouped = !folders.is_empty() || !sections.is_empty();
                @if searching || !loose.is_empty() || !grouped {
                    (notebook_grid(
                        if !searching && grouped { "Ungrouped" } else { "Active Notebooks" },
                        &loose, &folders, &back,
                        if searching { empty_state("No matches", "Try a different search.") }
                        else { empty_state("No notebooks yet", "Start by creating your first notebook to organize your research.") }))
                }
                @let archived_shown: Vec<&NotebookSummary> = archived.iter().filter(|nb| matches_query(&nb.name, &search)).collect();
                @if !archived_shown.is_empty() { (archived_list(&archived_shown, &folders, &back)) }
            }
        }
        (create_dialogs(&sections, current_folder.map(|f| f.id.as_str()), current_section.map(|s| s.id.as_str()), &back))
    };
    Ok(page(&ctx, "Notebooks", body))
}

fn breadcrumb(section: Option<&SectionSummary>, current: Option<&str>) -> Markup {
    html! {
        nav.crumbs aria-label="Folder navigation" {
            a href="/notebooks" { "All notebooks" }
            @if let Some(section) = section {
                " › " a href=(format!("/notebooks?section={}", encode(&section.id))) { (section.name) }
            }
            @if let Some(current) = current { " › " span aria-current="page" { (current) } }
        }
    }
}

async fn recently_viewed(state: &AppState) -> Markup {
    let items = match notebooks::recently_viewed(state, 6).await {
        Ok(items) => items,
        Err(e) => {
            tracing::warn!("Could not load recently viewed items: {e}");
            return html! {};
        }
    };
    if items.is_empty() {
        return html! {};
    }
    html! {
        section.stack-sm.nb-recent {
            h2 { "Recently Viewed" }
            div.row {
                @for item in &items {
                    @let (href, label) = if item.kind == "source" { (format!("/sources/{}", item.id), "Source") } else { (notebook_href(&item.id), "Notebook") };
                    a.item.nb-recent-item href=(href) {
                        span.badge { (label) }
                        span.truncate { (item.title) }
                        span.faint.small { (human_date_str(&item.last_viewed_at)) }
                    }
                }
            }
        }
    }
}

fn notebook_grid(title: &str, notebooks: &[&NotebookSummary], folders: &[FolderSummary], back: &str, empty: Markup) -> Markup {
    html! {
        section.stack-sm {
            h2 { (title) " " span.faint.small { "(" (notebooks.len()) ")" } }
            @if notebooks.is_empty() { (empty) }
            @else { div.grid { @for nb in notebooks { (notebook_card(nb, folders, back)) } } }
        }
    }
}

fn archived_list(notebooks: &[&NotebookSummary], folders: &[FolderSummary], back: &str) -> Markup {
    html! {
        details.nb-archived {
            summary { h2 { "Archived Notebooks " span.faint.small { "(" (notebooks.len()) ")" } } }
            div.grid { @for nb in notebooks { (notebook_card(nb, folders, back)) } }
        }
    }
}

fn notebook_card(nb: &NotebookSummary, folders: &[FolderSummary], back: &str) -> Markup {
    // Draggable onto a folder card (static/js/notebooks.js); the menu's
    // "Move to folder" does the same without a mouse or JavaScript.
    html! {
        article.card.nb-card draggable="true" data-nb-drag=(nb.id) {
            div.row-between {
                a.card-title.truncate.grow href=(notebook_href(&nb.id)) { (nb.name) }
                (notebook_menu(nb, folders, back))
            }
            @if nb.archived { span.badge.warn { "Archived" } }
            p.muted.small.clamp-2 {
                @if nb.description.is_empty() { "No description" } @else { (nb.description) }
            }
            div.row.faint.small.nb-card-foot {
                span { "Updated " (human_date_str(&nb.updated)) }
                span title="Sources" { (nb.source_count) " sources" }
                span title="Notes" { (nb.note_count) " notes" }
            }
        }
    }
}

/// The actions menu of a notebook card: rename, archive, move, delete.
fn notebook_menu(nb: &NotebookSummary, folders: &[FolderSummary], back: &str) -> Markup {
    html! {
        details.nb-menu {
            summary.btn.btn-ghost.btn-sm aria-label="Notebook actions" { "⋯" }
            div.nb-menu-body.stack-sm {
                form.stack-sm method="post" action=(format!("/notebooks/{}/rename", nb.id)) {
                    input type="hidden" name="back" value=(back);
                    label { "Name" input type="text" name="name" value=(nb.name) required; }
                    button.btn.btn-sm type="submit" { "Rename" }
                }
                form method="post" action=(format!("/notebooks/{}/archive", nb.id)) {
                    input type="hidden" name="back" value=(back);
                    input type="hidden" name="archived" value=(if nb.archived { "false" } else { "true" });
                    button.btn.btn-sm type="submit" { @if nb.archived { "Unarchive" } @else { "Archive" } }
                }
                (move_form(nb, folders, back))
                a.btn.btn-sm.btn-danger href=(format!("/notebooks/{}/delete", nb.id)) { "Delete" }
            }
        }
    }
}

fn move_form(nb: &NotebookSummary, folders: &[FolderSummary], back: &str) -> Markup {
    html! {
        form.stack-sm method="post" action=(format!("/notebooks/{}/move", nb.id)) {
            input type="hidden" name="back" value=(back);
            label {
                "Move to folder"
                select name="folder_id" {
                    option value="" selected[nb.folder_id.is_none()] { "No folder" }
                    @for folder in folders {
                        option value=(folder.id) selected[nb.folder_id.as_deref() == Some(folder.id.as_str())] { (folder.name) }
                    }
                }
            }
            button.btn.btn-sm type="submit" { "Move" }
        }
    }
}

fn folder_card(folder: &FolderSummary, sections: &[SectionSummary], back: &str) -> Markup {
    let href = match &folder.section_id {
        Some(section) => format!("/notebooks?section={}&folder={}", encode(section), encode(&folder.id)),
        None => format!("/notebooks?folder={}", encode(&folder.id)),
    };
    html! {
        article.card.nb-card.nb-folder data-nb-drop-folder=(folder.id) data-nb-back=(back) {
            div.row-between {
                a.card-title.truncate.grow href=(href) aria-label=(format!("Open folder {}", folder.name)) { "▸ " (folder.name) }
                details.nb-menu {
                    summary.btn.btn-ghost.btn-sm aria-label="Folder actions" { "⋯" }
                    div.nb-menu-body.stack-sm {
                        form.stack-sm method="post" action=(format!("/notebook-folders/{}/rename", folder.id)) {
                            input type="hidden" name="back" value=(back);
                            label { "Folder name" input type="text" name="name" value=(folder.name) required maxlength="80"; }
                            label {
                                "Section"
                                select name="section_id" {
                                    option value="" selected[folder.section_id.is_none()] { "No section" }
                                    @for section in sections {
                                        option value=(section.id) selected[folder.section_id.as_deref() == Some(section.id.as_str())] { (section.name) }
                                    }
                                }
                            }
                            button.btn.btn-sm type="submit" { "Rename" }
                        }
                        form method="post" action=(format!("/notebook-folders/{}/delete", folder.id))
                            data-nb-confirm=(format!("Delete the folder \"{}\"? Its notebooks are kept and return to Ungrouped.", folder.name)) {
                            input type="hidden" name="back" value=(back);
                            button.btn.btn-sm.btn-danger type="submit" { "Delete folder" }
                        }
                    }
                }
            }
            p.faint.small { (folder.notebook_count) " notebook(s)" }
        }
    }
}

fn section_card(section: &SectionSummary, back: &str) -> Markup {
    html! {
        article.card.nb-card.nb-section {
            div.row-between {
                a.card-title.truncate.grow href=(format!("/notebooks?section={}", encode(&section.id)))
                    aria-label=(format!("Open section {}", section.name)) { "▤ " (section.name) }
                details.nb-menu {
                    summary.btn.btn-ghost.btn-sm aria-label="Section actions" { "⋯" }
                    div.nb-menu-body.stack-sm {
                        form.stack-sm method="post" action=(format!("/notebook-sections/{}/rename", section.id)) {
                            input type="hidden" name="back" value=(back);
                            label { "Section name" input type="text" name="name" value=(section.name) required maxlength="80"; }
                            button.btn.btn-sm type="submit" { "Rename" }
                        }
                        form method="post" action=(format!("/notebook-sections/{}/delete", section.id))
                            data-nb-confirm=(format!("Delete the section \"{}\"? Its folders and notebooks are kept and move to the top level.", section.name)) {
                            input type="hidden" name="back" value=(back);
                            button.btn.btn-sm.btn-danger type="submit" { "Delete section" }
                        }
                    }
                }
            }
            p.faint.small { (section.folder_count) " folder(s)" }
        }
    }
}

fn create_dialogs(sections: &[SectionSummary], folder: Option<&str>, section: Option<&str>, back: &str) -> Markup {
    html! {
        (dialog("nb-create-notebook", "Create New Notebook", html! {
            form method="post" action="/notebooks" {
                @if let Some(folder) = folder { input type="hidden" name="folder_id" value=(folder); }
                div.dialog-body {
                    p.muted { "Enter a name and optional description to get started." }
                    div.field { label for="nb-new-name" { "Name" } input #nb-new-name type="text" name="name" required placeholder="Notebook name"; }
                    div.field { label for="nb-new-desc" { "Description" } textarea #nb-new-desc name="description" placeholder="Add more info about this notebook here..." {} }
                }
                div.dialog-foot {
                    button.btn type="button" data-close { "Cancel" }
                    button.btn.btn-primary type="submit" { "Create" }
                }
            }
        }))
        (dialog("nb-create-folder", "Create folder", html! {
            form method="post" action="/notebook-folders" {
                input type="hidden" name="back" value=(back);
                div.dialog-body {
                    p.muted { "Group related notebooks together in a folder." }
                    div.field { label for="nb-folder-name" { "Folder name" } input #nb-folder-name type="text" name="name" required maxlength="80" placeholder="e.g. Research"; }
                    div.field {
                        label for="nb-folder-section" { "Section" }
                        select #nb-folder-section name="section_id" {
                            option value="" { "No section" }
                            @for s in sections { option value=(s.id) selected[section == Some(s.id.as_str())] { (s.name) } }
                        }
                    }
                }
                div.dialog-foot {
                    button.btn type="button" data-close { "Cancel" }
                    button.btn.btn-primary type="submit" { "Create" }
                }
            }
        }))
        (dialog("nb-create-section", "Create section", html! {
            form method="post" action="/notebook-sections" {
                input type="hidden" name="back" value=(back);
                div.dialog-body {
                    p.muted { "Sections group folders, e.g. Personal > True crime > your notebooks." }
                    div.field { label for="nb-section-name" { "Section name" } input #nb-section-name type="text" name="name" required maxlength="80" placeholder="e.g. Personal"; }
                }
                div.dialog-foot {
                    button.btn type="button" data-close { "Cancel" }
                    button.btn.btn-primary type="submit" { "Create" }
                }
            }
        }))
    }
}

// ---------------------------------------------------------------------------
// Notebook, folder and section forms
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
struct NotebookForm {
    #[serde(default)]
    name: String,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    folder_id: Option<String>,
    #[serde(default)]
    back: Option<String>,
}

async fn create_notebook(State(state): State<AppState>, Form(form): Form<NotebookForm>) -> WebResult<Response> {
    let name = form.name.trim();
    if name.is_empty() {
        return Err(AppError::invalid("Notebook name is required").into());
    }
    let created = notebooks::create_notebook(
        &state,
        &NewNotebook { name: name.to_string(), description: form.description.unwrap_or_default().trim().to_string() },
    )
    .await?;
    if let Some(folder) = form.folder_id.as_deref().filter(|f| !f.is_empty()) {
        let update = NotebookUpdate { folder_id: Some(Some(folder.to_string())), ..Default::default() };
        notebooks::update_notebook(&state, &created.id, &update).await?;
    }
    Ok(Redirect::to(&notebook_href(&created.id)).into_response())
}

async fn rename_notebook(State(state): State<AppState>, Path(id): Path<String>, Form(form): Form<NotebookForm>) -> WebResult<Response> {
    let name = form.name.trim();
    if name.is_empty() {
        return Err(AppError::invalid("Notebook name is required").into());
    }
    let update = NotebookUpdate {
        name: Some(name.to_string()),
        description: form.description.map(|d| Some(d.trim().to_string())),
        ..Default::default()
    };
    notebooks::update_notebook(&state, &id, &update).await?;
    Ok(Redirect::to(&safe_path_or(form.back.as_deref(), &notebook_href(&id))).into_response())
}

#[derive(Debug, Deserialize)]
struct ArchiveForm {
    archived: String,
    #[serde(default)]
    back: Option<String>,
}

async fn archive_notebook(State(state): State<AppState>, Path(id): Path<String>, Form(form): Form<ArchiveForm>) -> WebResult<Response> {
    let update = NotebookUpdate { archived: Some(form.archived == "true"), ..Default::default() };
    notebooks::update_notebook(&state, &id, &update).await?;
    Ok(Redirect::to(&safe_path_or(form.back.as_deref(), &notebook_href(&id))).into_response())
}

async fn move_notebook(State(state): State<AppState>, Path(id): Path<String>, Form(form): Form<NotebookForm>) -> WebResult<Response> {
    let update = NotebookUpdate { folder_id: Some(form.folder_id.filter(|f| !f.is_empty())), ..Default::default() };
    notebooks::update_notebook(&state, &id, &update).await?;
    Ok(Redirect::to(&safe_path_or(form.back.as_deref(), &notebook_href(&id))).into_response())
}

async fn delete_page(State(state): State<AppState>, Path(id): Path<String>) -> WebResult<Markup> {
    let preview = notebooks::notebook_delete_preview(&state, &id).await?;
    let ctx = PageContext::new(&state, "/notebooks");
    let body = html! {
        div.crumbs { a href="/notebooks" { "Notebooks" } " › " a href=(notebook_href(&id)) { (preview.notebook_name) } }
        (page_head("Delete Notebook", None, html! {}))
        form.card.stack.nb-narrow method="post" action=(format!("/notebooks/{id}/delete")) {
            p { "Are you sure you want to delete \"" (preview.notebook_name) "\"? This action cannot be undone." }
            @if preview.note_count > 0 {
                p.nb-danger-text { (preview.note_count) " note(s) will be permanently deleted." }
            } @else { p.muted { "No notes to delete." } }
            @if preview.shared_source_count > 0 {
                p.muted { (preview.shared_source_count) " source(s) are shared with other notebooks and will be unlinked." }
            }
            @if preview.exclusive_source_count == 0 && preview.shared_source_count == 0 {
                p.muted { "No sources in this notebook." }
            }
            @if preview.exclusive_source_count > 0 {
                div.stack-sm.nb-divided {
                    p.nb-danger-text { (preview.exclusive_source_count) " source(s) exist only in this notebook." }
                    label.check { input type="radio" name="sources" value="delete"; "Delete exclusive sources" }
                    label.check { input type="radio" name="sources" value="keep" checked; "Unlink and keep them" }
                }
            }
            div.row {
                a.btn href=(notebook_href(&id)) { "Cancel" }
                button.btn.btn-danger type="submit" { "Delete" }
            }
        }
    };
    Ok(page(&ctx, "Delete Notebook", body))
}

#[derive(Debug, Default, Deserialize)]
struct DeleteForm {
    #[serde(default)]
    sources: Option<String>,
}

async fn delete_notebook(State(state): State<AppState>, Path(id): Path<String>, Form(form): Form<DeleteForm>) -> WebResult<Response> {
    let result = notebooks::delete_notebook(&state, &id, form.sources.as_deref() == Some("delete")).await?;
    let response = Redirect::to("/notebooks").into_response();
    Ok(with_trigger(response, json!({ "toast": result.message })))
}

#[derive(Debug, Default, Deserialize)]
struct NamedForm {
    #[serde(default)]
    name: String,
    #[serde(default)]
    section_id: Option<String>,
    #[serde(default)]
    back: Option<String>,
}

async fn create_folder(State(state): State<AppState>, Form(form): Form<NamedForm>) -> WebResult<Response> {
    let input = FolderInput { name: form.name, section_id: Some(form.section_id.filter(|s| !s.is_empty())) };
    notebooks::create_folder(&state, &input).await?;
    Ok(Redirect::to(&safe_path_or(form.back.as_deref(), "/notebooks")).into_response())
}

async fn rename_folder(State(state): State<AppState>, Path(id): Path<String>, Form(form): Form<NamedForm>) -> WebResult<Response> {
    let input = FolderInput { name: form.name, section_id: form.section_id.map(|s| Some(s).filter(|s| !s.is_empty())) };
    notebooks::update_folder(&state, &id, &input).await?;
    Ok(Redirect::to(&safe_path_or(form.back.as_deref(), "/notebooks")).into_response())
}

async fn delete_folder(State(state): State<AppState>, Path(id): Path<String>, Form(form): Form<NamedForm>) -> WebResult<Response> {
    notebooks::delete_folder(&state, &id).await?;
    // The folder's own page is gone; never return to it.
    let back = safe_path_or(form.back.as_deref(), "/notebooks");
    let back = if back.contains(&encode(&id)) { "/notebooks".to_string() } else { back };
    Ok(Redirect::to(&back).into_response())
}

async fn create_section(State(state): State<AppState>, Form(form): Form<NamedForm>) -> WebResult<Response> {
    notebooks::create_section(&state, &SectionInput { name: form.name }).await?;
    Ok(Redirect::to(&safe_path_or(form.back.as_deref(), "/notebooks")).into_response())
}

async fn rename_section(State(state): State<AppState>, Path(id): Path<String>, Form(form): Form<NamedForm>) -> WebResult<Response> {
    notebooks::rename_section(&state, &id, &SectionInput { name: form.name }).await?;
    Ok(Redirect::to(&safe_path_or(form.back.as_deref(), "/notebooks")).into_response())
}

async fn delete_section(State(state): State<AppState>, Path(id): Path<String>) -> WebResult<Response> {
    notebooks::delete_section(&state, &id).await?;
    Ok(Redirect::to("/notebooks").into_response())
}

// ---------------------------------------------------------------------------
// Notebook page
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
struct NotebookPageQuery {
    session: Option<String>,
    /// A note open as a page in the Notes column.
    note: Option<String>,
    /// That note was just created (focus its title).
    new: Option<String>,
}

async fn notebook_page(State(state): State<AppState>, Path(id): Path<String>, Query(query): Query<NotebookPageQuery>) -> WebResult<Markup> {
    let notebook = notebooks::view_notebook(&state, &id).await?;
    let folders = notebooks::list_folders(&state).await?;
    let sources = sources_pane(&state, &id).await?;
    let notes = notes_pane(&state, &id).await?;
    let chat = chat_pane(&state, &id, query.session.as_deref()).await?;
    let studio = super::studio::studio_panel(&state, &id).await?;
    let podcasts = super::podcasts::notebook_podcasts_panel(&state, &id).await?;
    let add_source = add_source_dialog(&state, &id).await?;
    let open_note = match query.note.as_deref().map(str::trim).filter(|n| !n.is_empty()) {
        Some(note_id) => Some(notes::get_note(&state, note_id).await?),
        None => None,
    };
    let notes_column = match &open_note {
        // The note is a page in the Notes column. The grid stays in the page,
        // hidden, so each note's chat-context choice still goes with the chat.
        Some(note) => {
            let fresh = query.new.as_deref().is_some_and(|v| !v.is_empty() && v != "0" && v != "false");
            let page = note_page_in_notebook(&state, &id, note, fresh).await?;
            html! { (page) div.nb-notes-behind hidden { (notes) (podcasts) (studio) } }
        }
        None => html! { (notes) (podcasts) (studio) },
    };

    let ctx = PageContext::new(&state, format!("/notebooks/{id}"));
    let body = html! {
        div.nb-workspace {
            (notebook_header(&notebook, &folders))
            // Three panes, each collapsible to a rail. Sources starts collapsed and Notes takes the free space.
            // The script runs while the page is parsed so the saved layout is in place before first paint.
            div.nb-cols #nb-cols data-sources="closed" data-notes="open" data-chat="open" data-tab="notes" {
                script { (PreEscaped(COLUMNS_INIT)) }
                (column("sources", "Sources", sources))
                (column("notes", "Notes", notes_column))
                (column("chat", "Chat", chat))
            }
        }
        (add_source)
        (dialog("nb-modal", "Details", html! { div #nb-modal-body .dialog-body {} }))
    };
    let title = match &open_note {
        Some(note) => note.title.clone().filter(|t| !t.trim().is_empty()).unwrap_or_else(|| "Untitled Note".into()),
        None => notebook.name.clone(),
    };
    Ok(page(&ctx, &title, body))
}

/// A note open as a floating page in the notebook's Notes column: just the
/// page (borderless title and rich text), with a bar to go back to the
/// notes, expand the page over Sources and Chat, file it in other
/// notebooks, or open it in the Notes view.
async fn note_page_in_notebook(state: &AppState, notebook_id: &str, note: &NoteView, fresh: bool) -> WebResult<Markup> {
    let all = super::workspace::active_notebooks(state).await?;
    let in_ids = crate::services::workspace::note_notebook_ids(state, &note.id).await?;
    let here = format!("{}?note={}", notebook_href(notebook_id), encode(&note.id));
    let item = format!("{}/notes/{}", notebook_href(notebook_id), note.id);
    Ok(html! {
        article.nn-page.nb-note-open data-note-href=(format!("{}?note=__ID__", notebook_href(notebook_id))) {
            header.nn-page-bar {
                a.btn.btn-ghost.btn-sm href=(notebook_href(notebook_id)) title="Back to this notebook's notes" { "‹ All notes" }
                div.nn-page-bar-main { (super::workspace::notebooks_menu(&note.id, &in_ids, &all, &here)) }
                div.row.nn-page-actions {
                    button.btn.btn-ghost.btn-sm type="button" data-nb-expand-page aria-pressed="false"
                        title="Expand the page: fold Sources and Chat away (again to bring them back)" { "Expand" }
                    a.btn.btn-ghost.btn-sm href=(format!("/notes?note={}", encode(&note.id))) title="Open in the Notes view" { "Open in Notes" }
                    form.inline method="post" action=(format!("{item}/delete")) data-note-create data-ws-confirm="Delete this note? This cannot be undone." {
                        button.btn.btn-ghost.btn-sm.nb-destructive type="submit" { "Delete" }
                    }
                    (collapse_button("notes", "Notes"))
                }
            }
            div.nn-page-body {
                @if note.attachment.is_some() { (note_body(notebook_id, note, false)) } @else { (note_editor(notebook_id, note, fresh)) }
            }
        }
    })
}

/// Restores the saved pane layout (`nn-nb-cols` in localStorage) onto the `.nb-cols` element that contains
/// this script. `notebooks.js` writes the key and handles the toggles.
const COLUMNS_INIT: &str = r#"(function(){var el=document.currentScript.parentElement;try{var s=JSON.parse(localStorage.getItem("nn-nb-cols")||"null");if(s){["sources","notes","chat"].forEach(function(k){if(s[k]==="open"||s[k]==="closed")el.setAttribute("data-"+k,s[k]);});}}catch(e){}})();"#;

const ICON_LEFT: &str = r#"<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M10 3.5 5.5 8l4.5 4.5"/></svg>"#;
const ICON_RIGHT: &str = r#"<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6 3.5 10.5 8 6 12.5"/></svg>"#;
const ICON_SQUEEZE: &str = r#"<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M1.5 8h4.2M10.3 8h4.2M4 5.8 6 8l-2 2.2M12 5.8 10 8l2 2.2"/></svg>"#;

/// The icon for collapsing a pane: it points the way the pane folds away.
fn collapse_icon(key: &str) -> &'static str {
    match key {
        "sources" => ICON_LEFT,
        "chat" => ICON_RIGHT,
        _ => ICON_SQUEEZE,
    }
}

const ICON_EXPAND: &str = r#"<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M6.2 8H1.5M9.8 8h4.7M3.5 5.8 1.5 8l2 2.2M12.5 5.8l2 2.2-2 2.2"/></svg>"#;

/// The icon on a collapsed pane's rail: it points the way the pane unfolds.
fn rail_icon(key: &str) -> &'static str {
    match key {
        "sources" => ICON_RIGHT,
        "chat" => ICON_LEFT,
        _ => ICON_EXPAND,
    }
}

/// The button in a pane's header that folds the pane into its rail.
fn collapse_button(key: &str, label: &str) -> Markup {
    html! {
        button.btn.btn-ghost.btn-icon.nb-collapse type="button" data-nb-toggle=(key) data-nb-from="pane"
            aria-label=(format!("Collapse {label}")) title=(format!("Collapse {label}")) { (PreEscaped(collapse_icon(key))) }
    }
}

/// One of the three notebook columns: the pane, and the rail that stands in for it while it is collapsed.
fn column(key: &str, label: &str, pane: Markup) -> Markup {
    html! {
        div class=(format!("nb-col nb-col-{key}")) data-col=(key) {
            div.nb-col-inner { (pane) }
            button.nb-rail type="button" data-nb-toggle=(key) data-nb-from="rail"
                aria-label=(format!("Show {label}")) title=(format!("Show {label}")) {
                span.nb-rail-icon { (PreEscaped(rail_icon(key))) }
                span.nb-rail-label { (label) }
            }
        }
    }
}

fn pane_switch(key: &str, label: &str, pressed: bool) -> Markup {
    html! {
        button.btn type="button" data-nb-toggle=(key) data-nb-from="toolbar" aria-pressed=(if pressed { "true" } else { "false" })
            title=(format!("Show or hide {label}")) { (label) }
    }
}

fn notebook_header(nb: &NotebookSummary, folders: &[FolderSummary]) -> Markup {
    let back = notebook_href(&nb.id);
    let folder = nb.folder_id.as_deref().and_then(|id| folders.iter().find(|f| f.id == id));
    html! {
        header.nb-toolbar {
            div.nb-titleblock {
                div.crumbs {
                    a href="/notebooks" { "Notebooks" }
                    @if let Some(folder) = folder { " › " a href=(format!("/notebooks?folder={}", encode(&folder.id))) { (folder.name) } }
                }
                div.row.nb-title-row {
                    h1 { (nb.name) }
                    @if nb.archived { span.badge.warn { "Archived" } }
                }
                p.faint.small.nb-meta {
                    @if nb.description.is_empty() { "Add description..." } @else { span.nb-desc { (nb.description) } " · " }
                    "Created " (human_date_str(&nb.created)) " · Updated " (human_date_str(&nb.updated))
                }
            }
            div.nb-toolbar-actions {
                div.btn-group.nb-switch role="group" aria-label="Panes" {
                    (pane_switch("sources", "Sources", false))
                    (pane_switch("notes", "Notes", true))
                    (pane_switch("chat", "Chat", true))
                }
                a.btn href=(format!("/notes?nb={}", encode(&nb.id))) title="This notebook's notes in the Notes view" { "Open in Notes" }
                div.btn-group role="group" aria-label="Notebook" {
                    button.btn type="button" data-open="#nb-edit-notebook" { "Edit" }
                    form.inline method="post" action=(format!("/notebooks/{}/archive", nb.id)) {
                        input type="hidden" name="back" value=(back);
                        input type="hidden" name="archived" value=(if nb.archived { "false" } else { "true" });
                        button.btn type="submit" { @if nb.archived { "Unarchive" } @else { "Archive" } }
                    }
                    a.btn.nb-destructive href=(format!("/notebooks/{}/delete", nb.id)) { "Delete" }
                }
            }
        }
        (dialog("nb-edit-notebook", "Edit notebook", html! {
            form method="post" action=(format!("/notebooks/{}/rename", nb.id)) {
                input type="hidden" name="back" value=(back);
                div.dialog-body {
                    div.field { label for="nb-edit-name" { "Name" } input #nb-edit-name type="text" name="name" value=(nb.name) required placeholder="Notebook name"; }
                    div.field { label for="nb-edit-desc" { "Description" } textarea #nb-edit-desc name="description" placeholder="Add description..." { (nb.description) } }
                }
                div.dialog-foot {
                    button.btn type="button" data-close { "Cancel" }
                    button.btn.btn-primary type="submit" { "Save" }
                }
            }
            div.dialog-body.nb-divided { (move_form(nb, folders, &back)) }
        }))
    }
}

// ---------------------------------------------------------------------------
// Sources pane
// ---------------------------------------------------------------------------

/// What a source row shows.
#[derive(Debug, Clone)]
struct SourceRow {
    id: String,
    title: String,
    kind: &'static str,
    status: Option<String>,
    error: Option<String>,
    insights_count: i64,
}

fn source_kind(asset: Option<&crate::domain::Asset>) -> &'static str {
    match asset {
        Some(asset) if asset.file_path.is_some() => "File",
        Some(asset) if asset.url.is_some() => "Link",
        _ => "Text",
    }
}

impl From<SourceListItem> for SourceRow {
    fn from(item: SourceListItem) -> Self {
        let error = item
            .processing_info
            .as_ref()
            .and_then(|info| info.get("error"))
            .and_then(|e| e.as_str())
            .map(str::to_string);
        Self {
            kind: source_kind(item.asset.as_ref()),
            title: item.title.filter(|t| !t.is_empty()).unwrap_or_else(|| "Untitled Source".to_string()),
            id: item.id,
            status: item.status,
            error,
            insights_count: item.insights_count,
        }
    }
}

fn is_processing(status: Option<&str>) -> bool {
    matches!(status, Some("new" | "queued" | "pending" | "running"))
}

async fn load_source_row(state: &AppState, source_id: &str) -> WebResult<SourceRow> {
    let source = sources::load_source(state, source_id).await?;
    let status = sources::source_status(state, source_id).await?;
    let insights = source.insights(state).await?;
    Ok(SourceRow {
        kind: source_kind(source.asset.as_ref()),
        title: source.title.clone().filter(|t| !t.is_empty()).unwrap_or_else(|| "Untitled Source".to_string()),
        id: source.id.clone(),
        error: status.processing_info.as_ref().and_then(|i| i.get("error")).and_then(|e| e.as_str()).map(str::to_string),
        status: status.status,
        insights_count: insights.len() as i64,
    })
}

async fn source_page_rows(state: &AppState, notebook_id: &str, offset: i64) -> WebResult<Vec<SourceRow>> {
    let query = SourceListQuery { notebook_id: Some(notebook_id.to_string()), limit: SOURCE_PAGE, offset, ..Default::default() };
    Ok(sources::list_sources(state, &query).await?.into_iter().map(SourceRow::from).collect())
}

fn source_row(notebook_id: &str, row: &SourceRow) -> Markup {
    let polling = is_processing(row.status.as_deref());
    let default_mode = if row.insights_count > 0 { "insights" } else { "full" };
    html! {
        div.item.nb-source
            hx-get=[polling.then(|| format!("/notebooks/{notebook_id}/sources/{}/row", row.id))]
            hx-trigger=[polling.then_some("every 3s")]
            hx-target=[polling.then_some("this")]
            hx-swap=[polling.then_some("outerHTML")] {
            div.grow.stack-sm {
                div.row {
                    span.badge { (row.kind) }
                    @if row.status.as_deref() != Some("completed") { (status_badge(row.status.as_deref())) }
                    @if row.insights_count > 0 { span.faint.small { (row.insights_count) " insights" } }
                }
                a.item-title.nb-source-title href=(format!("/sources/{}", row.id)) title=(row.title) { (row.title) }
                @if let Some(error) = row.error.as_deref().filter(|_| row.status.as_deref() == Some("failed")) {
                    p.small.nb-danger-text.clamp-2 title=(error) { (error) }
                }
                div.row.nb-row-actions {
                    select.nb-ctx name=(format!("ctx_{}", row.id)) form="nb-chat-form" aria-label="Chat context" data-default=(default_mode) {
                        option value="insights" selected[default_mode == "insights"] disabled[row.insights_count == 0] { "Insights" }
                        option value="full" selected[default_mode == "full"] { "Full content" }
                        option value="off" { "Not in context" }
                    }
                    form.inline method="post" action=(format!("/notebooks/{notebook_id}/sources/{}/remove", row.id))
                        hx-post=(format!("/notebooks/{notebook_id}/sources/{}/remove", row.id))
                        hx-target="closest .nb-source" hx-swap="outerHTML"
                        hx-confirm="Are you sure you want to remove this from the notebook?" {
                        button.btn.btn-ghost.btn-sm type="submit" title="Remove from notebook" { "Remove" }
                    }
                }
            }
        }
    }
}

fn source_rows(notebook_id: &str, rows: &[SourceRow], offset: i64) -> Markup {
    html! {
        @for row in rows { (source_row(notebook_id, row)) }
        @if rows.len() as i64 == SOURCE_PAGE {
            @let next = offset + SOURCE_PAGE;
            a.btn.btn-sm.nb-more href=(format!("/notebooks/{notebook_id}?sources_offset={next}"))
                hx-get=(format!("/notebooks/{notebook_id}/sources?offset={next}")) hx-target="this" hx-swap="outerHTML" { "Load more" }
        }
    }
}

async fn sources_pane(state: &AppState, notebook_id: &str) -> WebResult<Markup> {
    let rows = source_page_rows(state, notebook_id, 0).await?;
    Ok(html! {
        section.panel.nb-pane.nb-sources-pane {
            div.panel-head {
                (collapse_button("sources", "Sources"))
                h3.nb-pane-title { "Sources" }
                div.row.nb-head-actions {
                    button.btn.btn-primary.btn-sm type="button" data-open="#nb-add-source" title="Add Source" aria-label="Add Source" { "Add" }
                    // The column is narrow, so the secondary actions live in one menu.
                    details.nb-menu {
                        summary.btn.btn-sm title="More source actions" aria-label="More source actions" { "More" }
                        div.nb-menu-body.stack-sm {
                            a.btn.btn-sm href=(format!("/notebooks/{notebook_id}/sources/existing"))
                                hx-get=(format!("/notebooks/{notebook_id}/sources/existing")) hx-target="#nb-modal-body" hx-swap="innerHTML" { "Add existing" }
                            div.nb-menu-label { "Chat context" }
                            button.btn.btn-sm type="button" data-nb-bulk="source" data-mode="insights" { "Include all (insights only)" }
                            button.btn.btn-sm type="button" data-nb-bulk="source" data-mode="full" { "Include all (full content)" }
                            button.btn.btn-sm type="button" data-nb-bulk="source" data-mode="off" { "Exclude all from context" }
                        }
                    }
                }
            }
            div.panel-body #nb-sources { (sources_list(notebook_id, &rows)) }
        }
    })
}

fn sources_list(notebook_id: &str, rows: &[SourceRow]) -> Markup {
    html! {
        @if rows.is_empty() {
            (empty_state("No sources yet", "Add your first source to start building your knowledge base."))
        } @else { (source_rows(notebook_id, rows, 0)) }
    }
}

async fn add_source_dialog(state: &AppState, notebook_id: &str) -> WebResult<Markup> {
    let transformations = crate::services::transformations::list_transformations(state).await?;
    let settings = crate::services::settings::get_settings(state).await?;
    let embed_option = settings.default_embedding_option.as_deref().unwrap_or("ask");
    Ok(dialog("nb-add-source", "Add Source", html! {
        form method="post" action=(format!("/notebooks/{notebook_id}/sources")) enctype="multipart/form-data"
            hx-post=(format!("/notebooks/{notebook_id}/sources")) hx-encoding="multipart/form-data"
            hx-target="#nb-sources" hx-swap="innerHTML" data-close-on-success data-reset {
            div.dialog-body {
                fieldset.nb-type-picker {
                    legend { "Type" }
                    label.check { input type="radio" name="type" value="link" checked; "Link" }
                    label.check { input type="radio" name="type" value="upload"; "File" }
                    label.check { input type="radio" name="type" value="text"; "Text" }
                }
                div.field data-nb-type="link" {
                    label for="nb-src-url" { "URL(s)" }
                    textarea #nb-src-url name="url" rows="3" placeholder="Enter URLs, one per line" {}
                    p.hint { "Paste multiple URLs (one per line) to batch import" }
                }
                div.field data-nb-type="upload" {
                    label for="nb-src-file" { "File(s)" }
                    input #nb-src-file type="file" name="file" multiple;
                    p.hint { "Documents (PDF, DOCX, PPTX, EPUB, HTML, TXT, MD and more), media (MP4, MP3, WAV, M4A) and images." }
                }
                div.field data-nb-type="text" {
                    label for="nb-src-text" { "Text Content" }
                    textarea #nb-src-text name="content" rows="6" placeholder="Paste or type your content here..." {}
                }
                div.field {
                    label for="nb-src-title" { "Title" }
                    input #nb-src-title type="text" name="title" placeholder="Give your source a descriptive title";
                    p.hint { "Leave empty to have one generated. Ignored when adding several items at once." }
                }
                @if !transformations.is_empty() {
                    fieldset.stack-sm {
                        legend { "Transformations" }
                        @for t in &transformations {
                            label.check { input type="checkbox" name="transformations" value=(t.id) checked[t.apply_default]; (t.title) }
                        }
                    }
                }
                @if embed_option != "never" {
                    label.check { input type="checkbox" name="embed" value="true" checked[embed_option == "always" || embed_option == "ask"]; "Embed for vector search" }
                }
            }
            div.dialog-foot {
                span.htmx-indicator.spinner {}
                button.btn type="button" data-close { "Cancel" }
                button.btn.btn-primary type="submit" { "Add Source" }
            }
        }
    }))
}

#[derive(Debug, Default, Deserialize)]
struct OffsetQuery {
    #[serde(default)]
    offset: Option<i64>,
}

async fn sources_fragment(State(state): State<AppState>, Path(id): Path<String>, Query(query): Query<OffsetQuery>) -> WebResult<Markup> {
    let offset = query.offset.unwrap_or(0).max(0);
    let rows = source_page_rows(&state, &id, offset).await?;
    if offset == 0 {
        notebooks::get_notebook(&state, &id).await?;
        return Ok(sources_list(&id, &rows));
    }
    Ok(source_rows(&id, &rows, offset))
}

async fn source_row_fragment(State(state): State<AppState>, Path((id, source_id)): Path<(String, String)>) -> WebResult<Markup> {
    let row = load_source_row(&state, &source_id).await?;
    Ok(source_row(&id, &row))
}

async fn remove_source(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, source_id)): Path<(String, String)>,
) -> WebResult<Response> {
    notebooks::remove_source_from_notebook(&state, &id, &source_id).await?;
    let response = fragment_or_redirect(&headers, html! {}, &notebook_href(&id));
    Ok(with_trigger(response, json!({ "toast": "Source removed from notebook successfully", "nb-context": true })))
}

/// The sources to create from one add-source form: one per URL line or file.
fn sources_from_form(notebook_id: &str, fields: &Fields, files: Vec<UploadedFile>) -> Vec<(NewSource, Option<SourceUpload>)> {
    let source_type = fields.text("type").unwrap_or_else(|| "link".to_string());
    let base = NewSource {
        notebooks: Some(vec![notebook_id.to_string()]),
        source_type: source_type.clone(),
        title: fields.text("title"),
        transformations: Some(fields.all("transformations")),
        embed: fields.checked("embed"),
        async_processing: true,
        ..Default::default()
    };
    match source_type.as_str() {
        "link" => {
            let urls: Vec<String> =
                fields.text("url").unwrap_or_default().lines().map(str::trim).filter(|u| !u.is_empty()).map(str::to_string).collect();
            if urls.is_empty() {
                return vec![(base, None)];
            }
            let single = urls.len() == 1;
            urls.into_iter()
                .map(|url| (NewSource { url: Some(url), title: if single { base.title.clone() } else { None }, ..base.clone() }, None))
                .collect()
        }
        "upload" => {
            if files.is_empty() {
                return vec![(base, None)];
            }
            let single = files.len() == 1;
            files
                .into_iter()
                .map(|file| {
                    let upload = SourceUpload { filename: file.filename, data: file.data };
                    (NewSource { title: if single { base.title.clone() } else { None }, ..base.clone() }, Some(upload))
                })
                .collect()
        }
        _ => vec![(NewSource { content: fields.text("content"), ..base }, None)],
    }
}

async fn add_sources(State(state): State<AppState>, headers: HeaderMap, Path(id): Path<String>, multipart: Multipart) -> WebResult<Response> {
    notebooks::get_notebook(&state, &id).await?;
    let (fields, files) = read_multipart(multipart).await?;
    let mut added = 0;
    let mut failures = Vec::new();
    for (new, upload) in sources_from_form(&id, &fields, files) {
        let label = upload.as_ref().map(|u| u.filename.clone()).or_else(|| new.url.clone());
        match sources::create_source(&state, new, upload).await {
            Ok(_) => added += 1,
            Err(e) if added == 0 && failures.is_empty() && label.is_none() => return Err(e.into()),
            Err(e) => failures.push(format!("{}: {}", label.unwrap_or_else(|| "Source".into()), e.detail())),
        }
    }
    if added == 0 {
        if let Some(first) = failures.first() {
            return Err(AppError::invalid(first.clone()).into());
        }
    }
    let rows = source_page_rows(&state, &id, 0).await?;
    let fragment = html! {
        @if !failures.is_empty() {
            div.flash.bad role="alert" {
                (added) " source(s) added, " (failures.len()) " failed"
                ul { @for failure in &failures { li { (failure) } } }
            }
        }
        (sources_list(&id, &rows))
    };
    let toast = if added == 1 { "Source submitted for background processing.".to_string() } else { format!("{added} source(s) added to notebook") };
    Ok(with_trigger(fragment_or_redirect(&headers, fragment, &notebook_href(&id)), json!({ "toast": toast, "nb-context": true })))
}

async fn existing_sources_page(State(state): State<AppState>, headers: HeaderMap, Path(id): Path<String>) -> WebResult<Markup> {
    let notebook = notebooks::get_notebook(&state, &id).await?;
    let linked: std::collections::HashSet<String> = sources::list_sources(
        &state,
        &SourceListQuery { notebook_id: Some(id.clone()), limit: 100, ..Default::default() },
    )
    .await?
    .into_iter()
    .map(|s| s.id)
    .collect();
    let all = sources::list_sources(&state, &SourceListQuery { limit: 100, ..Default::default() }).await?;
    let candidates: Vec<SourceRow> = all.into_iter().filter(|s| !linked.contains(&s.id)).map(SourceRow::from).collect();
    let form = html! {
        form.stack method="post" action=(format!("/notebooks/{id}/sources/existing"))
            hx-post=(format!("/notebooks/{id}/sources/existing")) hx-target="#nb-sources" hx-swap="innerHTML" data-close-on-success {
            h3 { "Add Existing Source" }
            @if candidates.is_empty() {
                (empty_state("No other sources", "Every source is already in this notebook."))
            } @else {
                p.muted.small { "The most recent sources that are not in this notebook." }
                div.stack-sm.nb-pick-list {
                    @for row in &candidates {
                        label.check.item {
                            input type="checkbox" name="source_id" value=(row.id);
                            span.badge { (row.kind) }
                            span.truncate { (row.title) }
                        }
                    }
                }
                div.row { button.btn.btn-primary type="submit" { "Add to notebook" } }
            }
        }
    };
    if is_htmx(&headers) {
        return Ok(form);
    }
    let ctx = PageContext::new(&state, format!("/notebooks/{id}"));
    let body = html! {
        div.crumbs { a href="/notebooks" { "Notebooks" } " › " a href=(notebook_href(&id)) { (notebook.name) } }
        div.card.nb-narrow { (form) }
    };
    Ok(page(&ctx, "Add Existing Source", body))
}

async fn link_sources(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Form(fields): Form<Vec<(String, String)>>,
) -> WebResult<Response> {
    let fields = Fields(fields);
    let ids = fields.all("source_id");
    if ids.is_empty() {
        return Err(AppError::invalid("Choose at least one source").into());
    }
    for source_id in &ids {
        notebooks::add_source_to_notebook(&state, &id, source_id).await?;
    }
    let rows = source_page_rows(&state, &id, 0).await?;
    let response = fragment_or_redirect(&headers, sources_list(&id, &rows), &notebook_href(&id));
    Ok(with_trigger(response, json!({ "toast": format!("{} source(s) added to notebook", ids.len()), "nb-context": true })))
}

// ---------------------------------------------------------------------------
// Notes pane
// ---------------------------------------------------------------------------

async fn notes_pane(state: &AppState, notebook_id: &str) -> WebResult<Markup> {
    let notes = notes::list_notes(state, Some(notebook_id)).await?;
    Ok(html! {
        section.panel.nb-pane.nb-notes-pane data-nb-dropzone {
            div.panel-head {
                (collapse_button("notes", "Notes"))
                h3.nb-pane-title { "Notes" }
                div.row.nb-head-actions {
                    details.nb-menu {
                        summary.btn.btn-ghost.btn-sm title="Context" { "Context" }
                        div.nb-menu-body.stack-sm {
                            button.btn.btn-sm type="button" data-nb-bulk="note" data-mode="full" { "Include all in context" }
                            button.btn.btn-sm type="button" data-nb-bulk="note" data-mode="off" { "Exclude all from context" }
                        }
                    }
                    form.inline.nb-upload method="post" action=(format!("/notebooks/{notebook_id}/notes/upload")) enctype="multipart/form-data"
                        hx-post=(format!("/notebooks/{notebook_id}/notes/upload")) hx-encoding="multipart/form-data"
                        hx-target="#nb-notes" hx-swap="outerHTML" {
                        label.btn.btn-sm title="Add file" {
                            "Add file"
                            input.nb-file-input type="file" name="file" multiple accept=(NOTE_FILE_ACCEPT) hidden;
                        }
                        noscript { button.btn.btn-sm type="submit" { "Upload" } }
                    }
                    button.btn.btn-sm type="button" title="Copy a note or folder from your Notes library into this notebook"
                        hx-get=(format!("/notebooks/{notebook_id}/transport")) hx-target="#nb-modal-body" hx-swap="innerHTML" { "Transport" }
                    form.inline method="post" action=(format!("/notebooks/{notebook_id}/notes/new")) data-note-create {
                        button.btn.btn-primary.btn-sm type="submit" { "Write Note" }
                    }
                }
            }
            div.panel-body {
                div.nb-drop-hint { "Drop files to add them as notes" br; span.faint.small { "PDF, HTML, DOCX, TXT or Markdown" } }
                (notes_grid(notebook_id, &notes, &[]))
            }
        }
    })
}

/// The notes grid. It refreshes itself when the studio panel or the chat adds
/// a note (`notes-changed`).
fn notes_grid(notebook_id: &str, notes: &[NoteView], failures: &[String]) -> Markup {
    html! {
        div #nb-notes .nb-notes hx-get=(format!("/notebooks/{notebook_id}/notes")) hx-trigger="notes-changed from:body" hx-swap="outerHTML" {
            @if !failures.is_empty() {
                div.flash.bad role="alert" { ul { @for failure in failures { li { (failure) } } } }
            }
            @if notes.is_empty() {
                (empty_state("No notes yet", "Create your first note to capture insights and observations. Or drop a PDF, HTML, DOCX, TXT or Markdown file here."))
            } @else {
                div.nb-note-grid { @for note in notes { (note_card(notebook_id, note)) } }
            }
        }
    }
}

fn note_kind(note: &NoteView) -> (&'static str, String) {
    if note.explainer {
        ("info", "Explainer".to_string())
    } else if note.infographic {
        ("info", "Infographic".to_string())
    } else if let Some(attachment) = &note.attachment {
        ("", attachment.name.rsplit('.').next().unwrap_or("file").to_uppercase())
    } else if note.note_type.as_deref() == Some("ai") {
        ("ok", "AI Generated".to_string())
    } else {
        ("", "Human".to_string())
    }
}

fn file_size(bytes: i64) -> String {
    match bytes {
        b if b < 1024 => format!("{b} B"),
        b if b < 1024 * 1024 => format!("{:.1} KB", b as f64 / 1024.0),
        b => format!("{:.1} MB", b as f64 / (1024.0 * 1024.0)),
    }
}

fn note_card(notebook_id: &str, note: &NoteView) -> Markup {
    let (badge_class, badge) = note_kind(note);
    let href = format!("/notebooks/{notebook_id}/notes/{}", note.id);
    // Notes open as a page in the Notes column; files open in the viewer.
    let open_href = if note.attachment.is_some() { href.clone() } else { format!("{}?note={}", notebook_href(notebook_id), encode(&note.id)) };
    let title = note.title.clone().filter(|t| !t.is_empty()).unwrap_or_else(|| "Untitled Note".to_string());
    html! {
        article.nb-note {
            div.row-between {
                span class=(format!("badge {badge_class}")) { (badge) }
                select.nb-ctx name=(format!("ctx_{}", note.id)) form="nb-chat-form" aria-label="Chat context" data-default="full" {
                    option value="full" selected { "In context" }
                    option value="off" { "Not in context" }
                }
            }
            a.nb-note-title href=(open_href) hx-get=[note.attachment.as_ref().map(|_| &href)]
                hx-target=[note.attachment.as_ref().map(|_| "#nb-modal-body")]
                hx-swap=[note.attachment.as_ref().map(|_| "innerHTML")] title=(title) { (title) }
            @if note.infographic {
                a href=(href) hx-get=(href) hx-target="#nb-modal-body" hx-swap="innerHTML" tabindex="-1" aria-hidden="true" {
                    img.nb-thumb src=(format!("/api/notes/{}/file", note.id)) alt="" loading="lazy";
                }
            } @else if let Some(content) = note.content.as_deref().filter(|c| !c.is_empty()) {
                p.small.muted.nb-note-preview { (content.chars().take(240).collect::<String>()) }
            }
            div.row-between.faint.small.nb-note-foot {
                span { (human_date_str(&note.updated)) }
                @if let Some(attachment) = &note.attachment { span { (file_size(attachment.size)) } }
                form.inline method="post" action=(format!("{href}/delete"))
                    hx-post=(format!("{href}/delete")) hx-target="#nb-notes" hx-swap="outerHTML"
                    hx-confirm="Are you sure you want to delete this note? This action cannot be undone." {
                    button.btn.btn-ghost.btn-sm type="submit" title="Delete Note" { "Delete" }
                }
            }
        }
    }
}

async fn notes_fragment(State(state): State<AppState>, Path(id): Path<String>) -> WebResult<Markup> {
    let notes = notes::list_notes(&state, Some(&id)).await?;
    Ok(notes_grid(&id, &notes, &[]))
}

// ---------------------------------------------------------------------------
// Transport: copy notes from the Notes library into this notebook
// ---------------------------------------------------------------------------

/// The Transport dialog: every folder and note of the Notes library, each
/// with a Copy button. Copying duplicates — nothing is moved or linked.
async fn transport_page(State(state): State<AppState>, Path(id): Path<String>) -> WebResult<Markup> {
    let notebook = notebooks::get_notebook(&state, &id).await?;
    let folders: Vec<_> = crate::web::workspace::active_notebooks(&state)
        .await?
        .into_iter()
        .filter(|nb| nb.id != notebook.id)
        .collect();
    let notes = crate::services::workspace::list_workspace_notes(&state).await?;
    let action = format!("/notebooks/{id}/transport");
    let row_attrs = |search: String| search.to_lowercase();
    Ok(html! {
        div.stack.nb-transport data-nb-transport {
            h4 { "Transport into " (notebook.name) }
            p.muted.small {
                "Copy a note, or a whole folder of notes, from your Notes library into this notebook. "
                "A transported note is an independent copy: editing or deleting it never touches the original."
            }
            input type="search" placeholder="Filter notes and folders" aria-label="Filter notes and folders" data-nb-transport-filter autocomplete="off";
            @if !folders.is_empty() {
                div.nb-menu-label { "Folders" }
                div.stack-sm.nb-transport-list {
                    @for nb in &folders {
                        form.row-between.nb-transport-row method="post" action=(action)
                            hx-post=(action) hx-target="#nb-notes" hx-swap="outerHTML" data-search=(row_attrs(nb.name.clone())) {
                            span.truncate { (nb.name) " " span.faint.small { "· " (nb.note_count) (if nb.note_count == 1 { " note" } else { " notes" }) } }
                            button.btn.btn-sm type="submit" name="nb" value=(nb.id) disabled[nb.note_count == 0] { "Copy all" }
                        }
                    }
                }
            }
            div.nb-menu-label { "Notes" }
            @if notes.is_empty() {
                p.muted.small { "No notes in the library yet." }
            }
            div.stack-sm.nb-transport-list {
                @for note in &notes {
                    form.row-between.nb-transport-row method="post" action=(action)
                        hx-post=(action) hx-target="#nb-notes" hx-swap="outerHTML"
                        data-search=(row_attrs(format!("{} {}", note.title, note.notebook_name.as_deref().unwrap_or_default()))) {
                        span.truncate {
                            (note.title)
                            @if let Some(name) = &note.notebook_name { " " span.badge { (name) } }
                        }
                        button.btn.btn-sm type="submit" name="note" value=(note.id) { "Copy" }
                    }
                }
            }
            p.small.faint hidden data-nb-transport-empty { "Nothing matches." }
        }
    })
}

#[derive(Debug, Default, Deserialize)]
struct TransportForm {
    #[serde(default)]
    note: Option<String>,
    #[serde(default)]
    nb: Option<String>,
}

/// Perform a transport and answer with the refreshed notes grid (htmx) or a
/// redirect back to the notebook page (plain form).
async fn do_transport(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Form(form): Form<TransportForm>,
) -> WebResult<Response> {
    use crate::services::transport;
    let toast = match (form.note.as_deref().filter(|n| !n.is_empty()), form.nb.as_deref().filter(|n| !n.is_empty())) {
        (Some(note_id), _) => {
            let copy = transport::transport_note(&state, note_id, &id).await?;
            format!("Copied \"{}\"", copy.title.as_deref().unwrap_or("Untitled"))
        }
        (None, Some(from_id)) => {
            let report = transport::transport_notebook(&state, from_id, &id).await?;
            match report.failed.len() {
                0 => format!("Copied {} note{}", report.copied.len(), if report.copied.len() == 1 { "" } else { "s" }),
                n => format!("Copied {} note(s); {n} failed", report.copied.len()),
            }
        }
        (None, None) => return Err(AppError::invalid("Pick a note or a folder to transport").into()),
    };
    let response = if is_htmx(&headers) {
        let notes = notes::list_notes(&state, Some(&id)).await?;
        notes_grid(&id, &notes, &[]).into_response()
    } else {
        Redirect::to(&notebook_href(&id)).into_response()
    };
    Ok(with_trigger(response, json!({ "toast": toast })))
}

/// The rich editor, with the plain form as the no-JavaScript fallback.
pub(crate) fn note_editor(notebook_id: &str, note: &NoteView, fresh: bool) -> Markup {
    let url = format!("/notebooks/{notebook_id}/notes/{}", note.id);
    html! {
        div.stack.nb-rich data-nb-rich data-notebook=(notebook_id) data-nb-fresh[fresh]
            data-editor=(format!("{url}/editor")) data-link=(format!("/notebooks/{notebook_id}/notes/link"))
            data-bundle=(super::asset_url("editor/editor.js")) data-css=(super::asset_url("editor/editor.css")) {
            div.row-between.nb-viewer-head {
                div.row {
                    span.faint.small role="status" data-nb-status { "Saved" }
                    button.btn.btn-sm type="button" data-nb-retry hidden { "Retry save" }
                }
                div.row {
                    div.nb-mode role="group" aria-label="Editor mode" {
                        button.btn.btn-sm type="button" data-nb-mode="page" aria-pressed="true" disabled title="Page: a document" { "Page" }
                        button.btn.btn-sm type="button" data-nb-mode="edgeless" aria-pressed="false" disabled title="Edgeless: a whiteboard" { "Edgeless" }
                    }
                }
            }
            input.nb-rich-title type="text" value=[note.title.as_deref()] placeholder="Add a title..." aria-label="Title" disabled;
            div.nb-rich-surface {
                div.nb-rich-loading.faint.small { span.spinner {} " Loading editor…" }
            }
            div.nb-backlinks hidden {}
        }
        noscript { style { ".nb-rich { display: none }" } (note_plain_form(notebook_id, Some(note))) }
    }
}

/// The plain note form, for browsers without JavaScript.
fn note_plain_form(notebook_id: &str, note: Option<&NoteView>) -> Markup {
    let action = match note {
        Some(note) => format!("/notebooks/{notebook_id}/notes/{}", note.id),
        None => format!("/notebooks/{notebook_id}/notes"),
    };
    let content = note.and_then(|n| n.content.clone()).unwrap_or_default();
    html! {
        form.stack.nb-editor method="post" action=(action) {
            input type="text" name="title" value=[note.and_then(|n| n.title.as_deref())] placeholder="Add a title..." aria-label="Title";
            textarea.nb-note-text name="content" rows="14" placeholder="Write your note content here..." aria-label="Content" required { (content) }
            div.row {
                button.btn.btn-primary type="submit" { @if note.is_some() { "Save Note" } @else { "Create Note" } }
            }
        }
    }
}

/// A note made from a file, shown safely: SVG as an image, HTML in a
/// sandboxed frame (explainers may run scripts, nothing else may), PDF in a
/// frame, anything else as its extracted text.
fn note_viewer(note: &NoteView) -> Markup {
    let Some(attachment) = &note.attachment else {
        return markdown(note.content.as_deref().unwrap_or_default());
    };
    let src = format!("/api/notes/{}/file", note.id);
    let title = note.title.clone().unwrap_or_else(|| attachment.name.clone());
    html! {
        div.stack-sm {
            @if note.infographic {
                a href=(src) target="_blank" rel="noopener" { img.nb-infographic src=(src) alt=(title); }
            } @else if note.explainer {
                iframe.nb-frame src=(src) sandbox="allow-scripts" title=(title) referrerpolicy="no-referrer" {}
            } @else if attachment.mime == "text/html" {
                iframe.nb-frame src=(src) sandbox="" title=(title) referrerpolicy="no-referrer" {}
            } @else if attachment.mime == "application/pdf" {
                iframe.nb-frame src=(src) title=(title) {}
            } @else {
                (markdown(note.content.as_deref().unwrap_or_default()))
            }
            p.faint.small {
                "Original file, " (file_size(attachment.size)) ". The extracted text is used for search and chat."
                @if !note.explainer { " " a href=(format!("{src}?download=true")) { "Download original" } }
            }
        }
    }
}

async fn new_note_page(State(state): State<AppState>, Path(id): Path<String>) -> WebResult<Markup> {
    let notebook = notebooks::get_notebook(&state, &id).await?;
    Ok(note_full_page(&state, Some(&notebook), "Create note", note_plain_form(&id, None)))
}

async fn write_note(State(state): State<AppState>, Path(id): Path<String>) -> WebResult<Redirect> {
    open_blank_note(&state, &id).await
}

/// Quick Note (from any page) opens a blank note in the Notes view.
async fn quick_note(State(state): State<AppState>) -> WebResult<Redirect> {
    let note_id = note_docs::create_draft(&state, note_docs::quick_notes_notebook(&state).await?).await?;
    Ok(Redirect::to(&format!("/notes?note={}&new=1", encode(&note_id))))
}

/// Write Note opens a blank note as a page in the notebook's Notes column.
async fn open_blank_note(state: &AppState, notebook_id: &str) -> WebResult<Redirect> {
    let note_id = note_docs::create_draft(state, notebook_id).await?;
    Ok(Redirect::to(&format!("{}?note={}&new=1", notebook_href(notebook_id), encode(&note_id))))
}

fn note_full_page(state: &AppState, notebook: Option<&NotebookSummary>, title: &str, body: Markup) -> Markup {
    let path = notebook.map(|nb| notebook_href(&nb.id)).unwrap_or_else(|| "/notebooks".to_string());
    let ctx = PageContext::new(state, path);
    page(
        &ctx,
        title,
        html! {
            div.crumbs {
                a href="/notebooks" { "Notebooks" }
                @if let Some(nb) = notebook { " › " a href=(notebook_href(&nb.id)) { (nb.name) } }
            }
            div.nn-page.nb-note-page { div.nn-page-body { (body) } }
        },
    )
}

#[derive(Debug, Default, Deserialize)]
struct NotePageQuery {
    #[serde(default, rename = "new")]
    fresh: bool,
}

async fn note_page(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, note_id)): Path<(String, String)>,
    Query(query): Query<NotePageQuery>,
) -> WebResult<Markup> {
    let note = notes::get_note(&state, &note_id).await?;
    if is_htmx(&headers) {
        return Ok(note_body(&id, &note, true));
    }
    let notebook = notebooks::get_notebook(&state, &id).await?;
    let body = if note.attachment.is_some() { note_body(&id, &note, false) } else { note_editor(&id, &note, query.fresh) };
    let title = note.title.as_deref().filter(|title| !title.is_empty()).unwrap_or("Untitled Note");
    Ok(note_full_page(&state, Some(&notebook), title, body))
}

/// A note's view or editor. In the modal (`in_modal`), a file note also gets
/// an Expand button (fills the window) and a link to its own page.
pub(crate) fn note_body(notebook_id: &str, note: &NoteView, in_modal: bool) -> Markup {
    if note.attachment.is_some() {
        let page = format!("/notebooks/{notebook_id}/notes/{}", note.id);
        html! {
            div.row-between.nb-viewer-head {
                h3 { (note.title.as_deref().unwrap_or("Untitled Note")) }
                @if in_modal {
                    div.row {
                        button.btn.btn-sm type="button" data-nb-expand aria-pressed="false" { "Expand" }
                        a.btn.btn-sm href=(page) target="_blank" rel="noopener" { "Open in new tab" }
                    }
                }
            }
            (note_viewer(note))
        }
    } else {
        note_editor(notebook_id, note, false)
    }
}

/// `/notes/{id}`: a note's page in its notebook, or on its own when it is in none.
async fn note_redirect(State(state): State<AppState>, Path(note_id): Path<String>) -> WebResult<Response> {
    let note = notes::get_note(&state, &note_id).await?;
    let domain_note = crate::domain::Note::get(&state, &note.id).await?;
    if let Some(notebook_id) = domain_note.notebook_ids(&state).await?.into_iter().next() {
        return Ok(Redirect::to(&format!("/notebooks/{notebook_id}/notes/{}", note.id)).into_response());
    }
    let title = note.title.clone().unwrap_or_else(|| "Note".to_string());
    let body = html! { h3 { (title) } (note_viewer(&note)) };
    Ok(note_full_page(&state, None, &title, body).into_response())
}

#[derive(Debug, Default, Deserialize)]
struct NoteForm {
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    content: String,
}

async fn create_note(State(state): State<AppState>, headers: HeaderMap, Path(id): Path<String>, Form(form): Form<NoteForm>) -> WebResult<Response> {
    if form.content.trim().is_empty() {
        return Err(AppError::invalid("Note content is required").into());
    }
    let new = NewNote {
        title: form.title.map(|t| t.trim().to_string()).filter(|t| !t.is_empty()),
        content: form.content,
        note_type: Some(Some("human".to_string())),
        notebook_id: Some(id.clone()),
    };
    notes::create_note(&state, &new).await?;
    notes_response(&state, &headers, &id, "Note created successfully").await
}

async fn update_note(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, note_id)): Path<(String, String)>,
    Form(form): Form<NoteForm>,
) -> WebResult<Response> {
    let update = NoteUpdate { title: Some(form.title.unwrap_or_default().trim().to_string()), content: Some(form.content), note_type: None };
    notes::update_note(&state, &note_id, &update).await?;
    notes_response(&state, &headers, &id, "Note updated successfully").await
}

async fn delete_note(State(state): State<AppState>, headers: HeaderMap, Path((id, note_id)): Path<(String, String)>) -> WebResult<Response> {
    notes::delete_note(&state, &note_id).await?;
    notes_response(&state, &headers, &id, "Note deleted successfully").await
}

async fn notes_response(state: &AppState, headers: &HeaderMap, notebook_id: &str, toast: &str) -> WebResult<Response> {
    let fragment = if is_htmx(headers) { notes_grid(notebook_id, &notes::list_notes(state, Some(notebook_id)).await?, &[]) } else { html! {} };
    Ok(with_trigger(fragment_or_redirect(headers, fragment, &notebook_href(notebook_id)), json!({ "toast": toast, "nb-context": true })))
}

async fn upload_notes(State(state): State<AppState>, headers: HeaderMap, Path(id): Path<String>, multipart: Multipart) -> WebResult<Response> {
    notebooks::get_notebook(&state, &id).await?;
    let (_, files) = read_multipart(multipart).await?;
    if files.is_empty() {
        return Err(AppError::invalid("Choose a file to add").into());
    }
    let total = files.len();
    let mut failures = Vec::new();
    let mut first_error = None;
    for UploadedFile { filename, data, .. } in files {
        let upload = NoteUpload { filename: filename.clone(), data, notebook_id: id.clone() };
        if let Err(e) = notes::upload_note_file(&state, &upload).await {
            failures.push(format!("Could not add {filename}: {}", e.detail()));
            first_error.get_or_insert(e);
        }
    }
    if failures.len() == total {
        if let (1, Some(e)) = (total, first_error) {
            return Err(e.into());
        }
        return Err(AppError::invalid(failures.join("; ")).into());
    }
    let added = total - failures.len();
    let fragment = notes_grid(&id, &notes::list_notes(&state, Some(&id)).await?, &failures);
    let response = fragment_or_redirect(&headers, fragment, &notebook_href(&id));
    Ok(with_trigger(response, json!({ "toast": format!("{added} file(s) added"), "nb-context": true })))
}

// ---------------------------------------------------------------------------
// Rich note editor (JSON)
// ---------------------------------------------------------------------------

async fn new_note_doc(State(state): State<AppState>, Path(id): Path<String>) -> Result<axum::Json<NoteDocView>, AppError> {
    notebooks::get_notebook(&state, &id).await?;
    Ok(axum::Json(note_docs::load_note_doc(&state, &id, None).await?))
}

async fn note_doc(State(state): State<AppState>, Path((id, note_id)): Path<(String, String)>) -> Result<axum::Json<NoteDocView>, AppError> {
    Ok(axum::Json(note_docs::load_note_doc(&state, &id, Some(&note_id)).await?))
}

async fn create_note_doc(
    State(state): State<AppState>,
    Path(id): Path<String>,
    ApiJson(input): ApiJson<SaveNoteDoc>,
) -> Result<axum::Json<NoteView>, AppError> {
    notebooks::get_notebook(&state, &id).await?;
    Ok(axum::Json(note_docs::save_note_doc(&state, &id, None, &input).await?))
}

async fn update_note_doc(
    State(state): State<AppState>,
    Path((id, note_id)): Path<(String, String)>,
    ApiJson(input): ApiJson<SaveNoteDoc>,
) -> Result<axum::Json<NoteView>, AppError> {
    Ok(axum::Json(note_docs::save_note_doc(&state, &id, Some(&note_id), &input).await?))
}

#[derive(Debug, Deserialize)]
struct LinkedNoteInput {
    #[serde(default)]
    title: String,
}

async fn create_linked_note(
    State(state): State<AppState>,
    Path(id): Path<String>,
    ApiJson(input): ApiJson<LinkedNoteInput>,
) -> Result<axum::Json<NoteView>, AppError> {
    notebooks::get_notebook(&state, &id).await?;
    Ok(axum::Json(note_docs::create_linked_note(&state, &id, &input.title).await?))
}

/// A file placed inside a note. Content-addressed, so it can be cached for
/// good; served inert (sandboxed, never sniffed) whatever its type.
async fn get_note_blob(State(state): State<AppState>, Path(key): Path<String>) -> Result<Response, AppError> {
    use axum::http::{header, HeaderValue};
    let blob = note_docs::get_blob(&state.config, &key).await?;
    let inline = blob.mime.starts_with("image/") || blob.mime == "application/pdf";
    let headers = [
        (header::CONTENT_TYPE, HeaderValue::from_str(&blob.mime).unwrap_or(HeaderValue::from_static("application/octet-stream"))),
        (header::CACHE_CONTROL, HeaderValue::from_static("private, max-age=31536000, immutable")),
        (header::X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff")),
        (header::CONTENT_SECURITY_POLICY, HeaderValue::from_static("sandbox; default-src 'none'; img-src data:; style-src 'unsafe-inline'")),
        (header::CONTENT_DISPOSITION, HeaderValue::from_static(if inline { "inline" } else { "attachment" })),
    ];
    Ok((headers, blob.data).into_response())
}

async fn put_note_blob(
    State(state): State<AppState>,
    Path(key): Path<String>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Result<axum::Json<serde_json::Value>, AppError> {
    let mime = headers.get(axum::http::header::CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or_default();
    note_docs::put_blob(&state.config, &key, mime, &body).await?;
    Ok(axum::Json(json!({ "key": key })))
}

// ---------------------------------------------------------------------------
// Chat pane
// ---------------------------------------------------------------------------

/// The session to show: the requested one if it belongs to this notebook,
/// else the most recently updated.
fn pick_session<'a>(sessions: &'a [ChatSessionResponse], wanted: Option<&str>) -> Option<&'a ChatSessionResponse> {
    let wanted = wanted.map(|w| chat::normalize_record_id("chat_session", w));
    wanted.and_then(|w| sessions.iter().find(|s| s.id == w)).or_else(|| sessions.first())
}

async fn chat_pane(state: &AppState, notebook_id: &str, wanted: Option<&str>) -> WebResult<Markup> {
    let sessions = chat::list_sessions(state, notebook_id).await?;
    let current = pick_session(&sessions, wanted);
    let messages = match current {
        Some(session) => chat::get_session(state, &session.id).await?.messages,
        None => Vec::new(),
    };
    let models = crate::services::models::list_models(state, Some("language")).await?;
    let current_model = current.and_then(|s| s.model_override.clone());
    let chat_url = format!("/notebooks/{notebook_id}/chat");

    Ok(html! {
        section.panel.nb-pane.nb-chat-pane #nb-chat {
            div.panel-head {
                (collapse_button("chat", "Chat"))
                h3.nb-pane-title { "Chat with Notebook" }
                div #nb-ctx-stats .faint.small.nb-ctx-stats hx-post=(format!("{chat_url}/context")) hx-include="#nb-chat-form"
                    hx-trigger="load, nb-context from:body delay:300ms" hx-swap="innerHTML" {}
            }
            div.panel-body.chat {
                details.nb-sessions {
                    summary {
                        "Sessions (" (sessions.len()) ")"
                        @if let Some(session) = current { ": " span.nb-session-current { (session.title) } }
                    }
                    div.stack-sm {
                        form.row method="post" action=(format!("{chat_url}/sessions")) hx-post=(format!("{chat_url}/sessions"))
                            hx-target="#nb-chat" hx-swap="outerHTML" {
                            input.grow type="text" name="title" placeholder="Type a title here..." aria-label="New session title";
                            button.btn.btn-sm type="submit" { "New session" }
                        }
                        @if sessions.is_empty() { p.faint.small { "No chat sessions yet" } }
                        @for session in &sessions {
                            @let link = format!("/notebooks/{notebook_id}?session={}", encode(&session.id));
                            div.item.nb-session aria-current=[(current.map(|c| c.id.as_str()) == Some(session.id.as_str())).then_some("true")] {
                                div.grow.stack-sm {
                                    a href=(link) hx-get=(format!("{chat_url}?session={}", encode(&session.id))) hx-target="#nb-chat" hx-swap="outerHTML" {
                                        (session.title)
                                    }
                                    span.faint.small { (session.message_count.unwrap_or(0)) " messages · " (human_date_str(&session.updated)) }
                                    details.nb-inline-edit {
                                        summary.small { "Rename" }
                                        form.row method="post" action=(format!("{chat_url}/sessions/{}/rename", session.id))
                                            hx-post=(format!("{chat_url}/sessions/{}/rename", session.id)) hx-target="#nb-chat" hx-swap="outerHTML" {
                                            input.grow type="text" name="title" value=(session.title) required aria-label="Session title";
                                            button.btn.btn-sm type="submit" { "Save" }
                                        }
                                    }
                                }
                                form.inline method="post" action=(format!("{chat_url}/sessions/{}/delete", session.id))
                                    hx-post=(format!("{chat_url}/sessions/{}/delete", session.id)) hx-target="#nb-chat" hx-swap="outerHTML"
                                    hx-confirm="Are you sure you want to delete this chat session? This action cannot be undone." {
                                    button.btn.btn-ghost.btn-sm type="submit" title="Delete Session" { "Delete" }
                                }
                            }
                        }
                    }
                }
                div.chat-log.nb-chat-log {
                    @if messages.is_empty() {
                        div.empty.nb-chat-empty { strong { "Start a conversation about this notebook" } "Ask questions to understand the content better" }
                    }
                    @for message in &messages {
                        @if message.message_type == "human" {
                            div.msg.human { p.pre-wrap { (message.content) } }
                        } @else {
                            div.msg.ai {
                                (markdown(&message.content))
                                div.row.nb-msg-actions {
                                    button.btn.btn-ghost.btn-sm type="button" data-copy=(message.content) { "Copy" }
                                    @if let Some(session) = current {
                                        form.inline method="post" action=(format!("{chat_url}/save-note"))
                                            hx-post=(format!("{chat_url}/save-note")) hx-swap="none" {
                                            input type="hidden" name="session_id" value=(session.id);
                                            input type="hidden" name="message_id" value=(message.id);
                                            button.btn.btn-ghost.btn-sm type="submit" { "Save to note" }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                form #nb-chat-form .chat-form.nb-chat-form method="post" action=(format!("{chat_url}/send"))
                    hx-post=(format!("{chat_url}/send")) hx-target="#nb-chat" hx-swap="outerHTML" hx-disabled-elt="find button[type=submit]" {
                    input type="hidden" name="session_id" value=[current.map(|s| s.id.as_str())];
                    div.grow.stack-sm {
                        textarea.nb-chat-input name="message" required placeholder="Ask anything about your sources..." aria-label="Message" {}
                        div.row.faint.small {
                            label.nb-model {
                                "Model "
                                select name="model_override" aria-label="Model" {
                                    option value="" selected[current_model.is_none()] { "Default model" }
                                    @for model in &models {
                                        option value=(model.id) selected[current_model.as_deref() == Some(model.id.as_str())] {
                                            (model.name) " (" (model.provider) ")"
                                        }
                                    }
                                }
                            }
                            span { "Press Enter to send, Shift+Enter for a new line" }
                        }
                    }
                    button.btn.btn-primary type="submit" { "Send" span.htmx-indicator.spinner {} }
                }
            }
        }
    })
}

#[derive(Debug, Default, Deserialize)]
struct ChatQuery {
    session: Option<String>,
}

async fn chat_fragment(State(state): State<AppState>, Path(id): Path<String>, Query(query): Query<ChatQuery>) -> WebResult<Markup> {
    chat_pane(&state, &id, query.session.as_deref()).await
}

/// The chat context chosen in the panes: `ctx_<id>` fields with `insights`,
/// `full` or `off`, as the `context_config` the old frontend sent.
fn context_config(fields: &[(String, String)]) -> ContextConfig {
    let (mut sources, mut notes) = (Vec::new(), Vec::new());
    for (name, value) in fields {
        let Some(id) = name.strip_prefix("ctx_") else { continue };
        let status = match value.as_str() {
            "insights" => "insights",
            "full" => "full content",
            _ => "not in",
        };
        match crate::db::table_of(id) {
            Some("source") => sources.push((id.to_string(), status.to_string())),
            Some("note") => notes.push((id.to_string(), if status == "insights" { "not in" } else { status }.to_string())),
            _ => {}
        }
    }
    ContextConfig { sources: Some(ItemStatuses(sources)), notes: Some(ItemStatuses(notes)) }
}

async fn context_stats(State(state): State<AppState>, Path(id): Path<String>, Form(fields): Form<Vec<(String, String)>>) -> WebResult<Markup> {
    let config = context_config(&fields);
    let count = |list: &Option<ItemStatuses>, status: &str| list.as_ref().map_or(0, |l| l.0.iter().filter(|(_, s)| s == status).count());
    let (insights, full, notes_in) = (count(&config.sources, "insights"), count(&config.sources, "full content"), count(&config.notes, "full content"));
    let built = build_context(&state, &BuildContextRequest { notebook_id: id, context_config: config }).await?;
    Ok(html! {
        span title=(format!("{} characters", built.char_count)) {
            (insights + full) " sources (" (insights) " insights, " (full) " full) · " (notes_in) " notes · ~" (built.token_count) " tokens"
        }
    })
}

async fn send_message(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Form(fields): Form<Vec<(String, String)>>,
) -> WebResult<Response> {
    let fields = Fields(fields);
    let message = fields.text("message").ok_or_else(|| AppError::invalid("Message is required"))?;
    let model_override = fields.text("model_override");
    let sessions = chat::list_sessions(&state, &id).await?;
    let requested = fields.text("session_id").map(|s| chat::normalize_record_id("chat_session", &s));

    let session_id = match requested.and_then(|wanted| sessions.iter().find(|s| s.id == wanted)) {
        Some(session) => {
            if session.model_override != model_override {
                let update = UpdateSessionRequest { model_override: Some(model_override.clone()), ..Default::default() };
                chat::update_session(&state, &session.id, &update).await?;
            }
            session.id.clone()
        }
        None => {
            let title = if message.chars().count() > 30 { format!("{}...", message.chars().take(30).collect::<String>()) } else { message.clone() };
            let request = CreateSessionRequest { notebook_id: id.clone(), title: Some(title), model_override: model_override.clone() };
            chat::create_session(&state, &request).await?.id
        }
    };

    let built = build_context(&state, &BuildContextRequest { notebook_id: id.clone(), context_config: context_config(&fields.0) }).await?;
    let context = match serde_json::to_value(&built.context).map_err(|e| AppError::internal(e.to_string()))? {
        serde_json::Value::Object(map) => map,
        _ => serde_json::Map::new(),
    };
    chat::execute_chat(&state, &ExecuteChatRequest { session_id: session_id.clone(), message, context, model_override: None }).await?;

    if is_htmx(&headers) {
        return Ok(chat_pane(&state, &id, Some(&session_id)).await?.into_response());
    }
    Ok(Redirect::to(&format!("/notebooks/{id}?session={}", encode(&session_id))).into_response())
}

#[derive(Debug, Default, Deserialize)]
struct SaveNoteForm {
    session_id: String,
    message_id: String,
}

async fn save_message_as_note(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Form(form): Form<SaveNoteForm>,
) -> WebResult<Response> {
    let session = chat::get_session(&state, &form.session_id).await?;
    if session.session.notebook_id.as_deref() != Some(id.as_str()) {
        return Err(AppError::not_found("Session not found").into());
    }
    let message = session.messages.iter().find(|m| m.id == form.message_id).ok_or_else(|| AppError::not_found("Message not found"))?;
    let new = NewNote { title: None, content: message.content.clone(), note_type: Some(Some("ai".to_string())), notebook_id: Some(id.clone()) };
    notes::create_note(&state, &new).await?;
    let target = format!("/notebooks/{id}?session={}", encode(&session.session.id));
    Ok(with_trigger(fragment_or_redirect(&headers, html! {}, &target), json!({ "toast": "Note saved", "notes-changed": true })))
}

#[derive(Debug, Default, Deserialize)]
struct SessionForm {
    #[serde(default)]
    title: Option<String>,
}

async fn create_session(State(state): State<AppState>, headers: HeaderMap, Path(id): Path<String>, Form(form): Form<SessionForm>) -> WebResult<Response> {
    let request = CreateSessionRequest { notebook_id: id.clone(), title: form.title.map(|t| t.trim().to_string()), model_override: None };
    let session = chat::create_session(&state, &request).await?;
    chat_response(&state, &headers, &id, Some(&session.id), "Chat session created").await
}

async fn rename_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, session_id)): Path<(String, String)>,
    Form(form): Form<SessionForm>,
) -> WebResult<Response> {
    ensure_session_in_notebook(&state, &id, &session_id).await?;
    let title = form.title.map(|t| t.trim().to_string()).filter(|t| !t.is_empty()).ok_or_else(|| AppError::invalid("Title is required"))?;
    chat::update_session(&state, &session_id, &UpdateSessionRequest { title: Some(Some(title)), ..Default::default() }).await?;
    chat_response(&state, &headers, &id, Some(&session_id), "Session updated").await
}

async fn delete_session(State(state): State<AppState>, headers: HeaderMap, Path((id, session_id)): Path<(String, String)>) -> WebResult<Response> {
    ensure_session_in_notebook(&state, &id, &session_id).await?;
    chat::delete_session(&state, &session_id).await?;
    chat_response(&state, &headers, &id, None, "Session deleted").await
}

async fn ensure_session_in_notebook(state: &AppState, notebook_id: &str, session_id: &str) -> WebResult<()> {
    let wanted = chat::normalize_record_id("chat_session", session_id);
    if chat::list_sessions(state, notebook_id).await?.iter().any(|s| s.id == wanted) {
        Ok(())
    } else {
        Err(AppError::not_found("Session not found").into())
    }
}

async fn chat_response(state: &AppState, headers: &HeaderMap, notebook_id: &str, session: Option<&str>, toast: &str) -> WebResult<Response> {
    let response = if is_htmx(headers) {
        chat_pane(state, notebook_id, session).await?.into_response()
    } else {
        let target = match session {
            Some(session) => format!("/notebooks/{notebook_id}?session={}", encode(session)),
            None => notebook_href(notebook_id),
        };
        Redirect::to(&target).into_response()
    };
    Ok(with_trigger(response, json!({ "toast": toast })))
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{header, Method, Request, StatusCode};
    use http_body_util::BodyExt;
    use serde_json::{json, Value};

    use crate::ai::AnyModel;
    use crate::testing::{multipart_body, MockLanguageModel, TestApp};

    struct Reply {
        status: StatusCode,
        location: Option<String>,
        trigger: Option<String>,
        body: String,
    }

    async fn send(app: &TestApp, method: Method, path: &str, content_type: Option<&str>, body: Vec<u8>, htmx: bool) -> Reply {
        let mut builder = Request::builder().method(method).uri(path);
        if let Some(content_type) = content_type {
            builder = builder.header(header::CONTENT_TYPE, content_type);
        }
        if htmx {
            builder = builder.header("HX-Request", "true");
        }
        let response = app.send(builder.body(Body::from(body)).unwrap()).await;
        let status = response.status();
        let header_str = |name: &str| response.headers().get(name).map(|v| v.to_str().unwrap().to_string());
        let (location, trigger) = (header_str("location"), header_str("hx-trigger"));
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        Reply { status, location, trigger, body: String::from_utf8_lossy(&bytes).into_owned() }
    }

    async fn form(app: &TestApp, path: &str, body: &str, htmx: bool) -> Reply {
        send(app, Method::POST, path, Some("application/x-www-form-urlencoded"), body.as_bytes().to_vec(), htmx).await
    }

    async fn page(app: &TestApp, path: &str) -> Reply {
        send(app, Method::GET, path, None, Vec::new(), false).await
    }

    async fn htmx_get(app: &TestApp, path: &str) -> Reply {
        send(app, Method::GET, path, None, Vec::new(), true).await
    }

    fn enc(value: &str) -> String {
        crate::api::auth::encode_query_value(value)
    }

    async fn notebook(app: &TestApp, name: &str) -> String {
        let (_, nb) = app.post("/api/notebooks", json!({ "name": name, "description": "About things" })).await;
        nb["id"].as_str().unwrap().to_string()
    }

    #[tokio::test]
    async fn the_notebook_page_has_three_collapsible_panes_with_sources_closed() {
        let app = TestApp::new().await;
        let nb = notebook(&app, "Panes").await;
        let reply = page(&app, &format!("/notebooks/{}", enc(&nb))).await;
        assert_eq!(reply.status, StatusCode::OK);
        let body = &reply.body;
        // Sources starts minimised; Notes and Chat start open.
        assert!(body.contains(r#"id="nb-cols""#) && body.contains(r#"data-sources="closed""#), "{body}");
        assert!(body.contains(r#"data-notes="open""#) && body.contains(r#"data-chat="open""#), "{body}");
        // Each pane sits in a column with a rail to bring it back, and has a collapse button.
        for pane in ["sources", "notes", "chat"] {
            assert!(body.contains(&format!(r#"data-col="{pane}""#)), "no {pane} column: {body}");
            assert!(body.contains(&format!(r#"data-nb-toggle="{pane}" data-nb-from="rail""#)), "no {pane} rail: {body}");
            assert!(body.contains(&format!(r#"data-nb-toggle="{pane}" data-nb-from="pane""#)), "no {pane} collapse button: {body}");
        }
        // The page still carries what the panes need.
        assert!(body.contains("nb-add-source") && body.contains("Write Note"), "{body}");
    }

    #[tokio::test]
    async fn transport_copies_notes_and_folders_into_the_notebook() {
        let app = TestApp::new().await;
        let target = notebook(&app, "Project").await;
        let library = notebook(&app, "Garden").await;
        let (_, note) = app
            .post("/api/notes", json!({ "title": "Tomatoes", "content": "Plant in May", "notebook_id": library }))
            .await;
        let note_id = note["id"].as_str().unwrap().to_string();

        // The notebook page offers Transport; the dialog lists folder and note.
        let reply = page(&app, &format!("/notebooks/{}", enc(&target))).await;
        assert!(reply.body.contains("Transport"), "{}", reply.body);
        let dialog = htmx_get(&app, &format!("/notebooks/{target}/transport")).await;
        assert_eq!(dialog.status, StatusCode::OK);
        assert!(dialog.body.contains("Transport into Project"), "{}", dialog.body);
        assert!(dialog.body.contains("Garden") && dialog.body.contains("Tomatoes"), "{}", dialog.body);
        assert!(!dialog.body.contains(&format!("value=\"{target}\">Copy all")), "the target is not a source");

        // Copying one note duplicates it into the notebook.
        let reply = form(&app, &format!("/notebooks/{target}/transport"), &format!("note={}", enc(&note_id)), true).await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        assert!(reply.trigger.unwrap().contains("Copied \\\"Tomatoes\\\""));
        let (_, listed) = app.get(&format!("/api/notes?notebook_id={target}")).await;
        assert_eq!(listed.as_array().unwrap().len(), 1);
        let copy_id = listed[0]["id"].as_str().unwrap().to_string();
        assert_ne!(copy_id, note_id);

        // Copying the folder brings its notes; the original keeps its own.
        let reply = form(&app, &format!("/notebooks/{target}/transport"), &format!("nb={}", enc(&library)), true).await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        let (_, listed) = app.get(&format!("/api/notes?notebook_id={target}")).await;
        assert_eq!(listed.as_array().unwrap().len(), 2);
        let (_, source_notes) = app.get(&format!("/api/notes?notebook_id={library}")).await;
        assert_eq!(source_notes.as_array().unwrap().len(), 1);

        // Nothing picked, or a missing record, is refused plainly.
        let reply = form(&app, &format!("/notebooks/{target}/transport"), "", true).await;
        assert_eq!(reply.status, StatusCode::BAD_REQUEST);
        let reply = form(&app, &format!("/notebooks/{target}/transport"), "note=note%3Anope", true).await;
        assert_eq!(reply.status, StatusCode::NOT_FOUND);
    }

    async fn text_source(app: &TestApp, nb: &str, title: &str) -> String {
        let (status, source) = app
            .post("/api/sources/json", json!({ "type": "text", "content": "Waves move energy.", "title": title, "notebooks": [nb], "async_processing": true }))
            .await;
        assert_eq!(status, StatusCode::OK, "{source}");
        let id = source["id"].as_str().unwrap().to_string();
        app.wait_for_job(source["command_id"].as_str().unwrap()).await;
        id
    }

    #[tokio::test]
    async fn empty_list_page() {
        let app = TestApp::new().await;
        let reply = page(&app, "/notebooks").await;
        assert_eq!(reply.status, StatusCode::OK);
        assert!(reply.body.contains("No notebooks yet"), "{}", reply.body);
        assert!(reply.body.contains("New Notebook"));
    }

    #[tokio::test]
    async fn writing_starts_with_a_saved_note_and_autosaves_update_the_same_record() {
        let app = TestApp::new().await;
        let notebook_id = notebook(&app, "Writing").await;
        let quick_notebook = "notebook:quick_notes";
        for (create, notebook) in [
            ("/notes/quick".to_string(), quick_notebook),
            (format!("/notebooks/{notebook_id}/notes/new"), notebook_id.as_str()),
        ] {
            let reply = form(&app, &create, "", false).await;
            assert_eq!(reply.status, StatusCode::SEE_OTHER, "{}", reply.body);
            let location = reply.location.unwrap();
            // Quick Note opens in the Notes view; Write Note in the notebook's Notes column.
            let opened_in = if notebook == quick_notebook { "/notes?note=".to_string() } else { format!("/notebooks/{notebook}?note=") };
            assert!(location.starts_with(&opened_in) && location.ends_with("&new=1"), "{location}");
            let note_id = location[opened_in.len()..].trim_end_matches("&new=1").replace("%3A", ":");
            let note_id = note_id.as_str();
            assert!(note_id.starts_with("note:"));
            let url = format!("/notebooks/{notebook}/notes/{note_id}");
            let (_, blank) = app.get(&format!("/api/notes/{note_id}")).await;
            assert!(blank["content"].is_null());
            let (_, all_notes) = app.get("/api/notes").await;
            assert!(!all_notes.as_array().unwrap().iter().any(|note| note["id"] == note_id));
            let (_, listed) = app.get(&format!("/api/notes?notebook_id={notebook}")).await;
            assert_eq!(listed, json!([]));
            let (_, summary) = app.get(&format!("/api/notebooks/{notebook}")).await;
            assert_eq!(summary["note_count"], 0);
            let notebook_record = crate::domain::Notebook::get(&app.state, notebook).await.unwrap();
            let (context, _) = crate::services::context::build_notebook_context(&app.state, &notebook_record, None).await.unwrap();
            assert!(context.notes.is_empty());
            let full = page(&app, &location).await;
            assert!(full.body.contains("data-nb-fresh") && full.body.contains("data-nb-status"));

            let editor_url = format!("{url}/editor");
            for (title, content, mode) in [("Draft", "First entry", "page"), ("", "", "edgeless")] {
                let (status, saved) = app
                    .post(&editor_url, json!({ "title": title, "content": content, "doc_state": "AQID", "doc_mode": mode }))
                    .await;
                assert_eq!(status, StatusCode::OK, "{saved}");
                assert_eq!(saved["id"], note_id);
                let (_, doc) = app.get(&editor_url).await;
                assert_eq!(doc["note"]["title"], title);
                assert_eq!(doc["note"]["content"].as_str().unwrap_or_default(), content);
                assert_eq!(doc["doc_state"], "AQID");
                assert_eq!(doc["doc_mode"], mode);
            }
            let (_, listed) = app.get(&format!("/api/notes?notebook_id={notebook}")).await;
            assert_eq!(listed.as_array().unwrap().len(), 1);
            let (_, summary) = app.get(&format!("/api/notebooks/{notebook}")).await;
            assert_eq!(summary["note_count"], 1);
        }

        let quick_url = format!("/api/notebooks/{quick_notebook}");
        let (status, _) = app.put(&quick_url, json!({ "archived": true })).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(form(&app, "/notes/quick", "", false).await.status, StatusCode::SEE_OTHER);
        let (_, summary) = app.get(&quick_url).await;
        assert_eq!(summary["archived"], false);
        assert_eq!(summary["note_count"], 1);
    }

    #[tokio::test]
    async fn create_rename_archive_and_list_notebooks() {
        let app = TestApp::new().await;
        let reply = form(&app, "/notebooks", "name=Physics&description=Waves", false).await;
        assert_eq!(reply.status, StatusCode::SEE_OTHER);
        let location = reply.location.unwrap();
        let id = location.trim_start_matches("/notebooks/").to_string();
        let (_, nb) = app.get(&format!("/api/notebooks/{id}")).await;
        assert_eq!(nb["name"], "Physics");
        assert_eq!(nb["description"], "Waves");

        let reply = form(&app, &format!("/notebooks/{id}/rename"), "name=Optics&description=Light&back=%2Fnotebooks", false).await;
        assert_eq!(reply.location.as_deref(), Some("/notebooks"));
        let (_, nb) = app.get(&format!("/api/notebooks/{id}")).await;
        assert_eq!(nb["name"], "Optics");

        let listed = page(&app, "/notebooks").await;
        assert!(listed.body.contains("Optics") && listed.body.contains("Active Notebooks"));

        form(&app, &format!("/notebooks/{id}/archive"), "archived=true", false).await;
        let (_, nb) = app.get(&format!("/api/notebooks/{id}")).await;
        assert_eq!(nb["archived"], true);
        let listed = page(&app, "/notebooks").await;
        assert!(listed.body.contains("Archived Notebooks"));

        let searched = page(&app, "/notebooks?q=zzz").await;
        assert!(searched.body.contains("No matches"));

        let reply = form(&app, "/notebooks", "name=+++", false).await;
        assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn folders_and_sections() {
        let app = TestApp::new().await;
        let nb = notebook(&app, "Cases").await;
        assert_eq!(form(&app, "/notebook-sections", "name=Personal", false).await.status, StatusCode::SEE_OTHER);
        let (_, sections) = app.get("/api/sections").await;
        let section = sections[0]["id"].as_str().unwrap().to_string();
        let reply = form(&app, "/notebook-folders", &format!("name=Crime&section_id={}", enc(&section)), false).await;
        assert_eq!(reply.status, StatusCode::SEE_OTHER);
        let (_, folders) = app.get("/api/folders").await;
        let folder = folders[0]["id"].as_str().unwrap().to_string();
        assert_eq!(folders[0]["section_id"], section.as_str());

        // Notebook cards are draggable and folder cards accept the drop; the
        // drop posts the same move form, with the view to come back to.
        let root = page(&app, "/notebooks").await;
        assert!(root.body.contains(&format!(r#"draggable="true" data-nb-drag="{nb}""#)), "{}", root.body);
        let section_view = page(&app, &format!("/notebooks?section={}", enc(&section))).await;
        assert!(section_view.body.contains(&format!(r#"data-nb-drop-folder="{folder}""#)), "{}", section_view.body);
        let back = format!("/notebooks?section={}", enc(&section));
        let reply = form(&app, &format!("/notebooks/{nb}/move"), &format!("folder_id={}&back={}", enc(&folder), enc(&back)), false).await;
        assert_eq!(reply.status, StatusCode::SEE_OTHER);
        assert_eq!(reply.location.as_deref(), Some(back.as_str()));
        let (_, moved) = app.get(&format!("/api/notebooks/{nb}")).await;
        assert_eq!(moved["folder_id"], folder.as_str());

        let root = page(&app, "/notebooks").await;
        assert!(root.body.contains("Sections") && root.body.contains("Personal"));
        let in_section = page(&app, &format!("/notebooks?section={}", enc(&section))).await;
        assert!(in_section.body.contains("Crime"), "{}", in_section.body);
        let in_folder = page(&app, &format!("/notebooks?folder={}", enc(&folder))).await;
        assert!(in_folder.body.contains("Cases") && in_folder.body.contains("All notebooks"));

        form(&app, &format!("/notebook-folders/{folder}/rename"), "name=True+crime", false).await;
        let (_, folders) = app.get("/api/folders").await;
        assert_eq!(folders[0]["name"], "True crime");
        // A duplicate section name is refused.
        assert_eq!(form(&app, "/notebook-sections", "name=personal", false).await.status, StatusCode::CONFLICT);

        form(&app, &format!("/notebook-folders/{folder}/delete"), "", false).await;
        form(&app, &format!("/notebook-sections/{section}/delete"), "", false).await;
        let (_, folders) = app.get("/api/folders").await;
        assert_eq!(folders.as_array().unwrap().len(), 0);
        let (_, kept) = app.get(&format!("/api/notebooks/{nb}")).await;
        assert!(kept["folder_id"].is_null());
    }

    #[tokio::test]
    async fn delete_notebook_with_preview() {
        let app = TestApp::new().await;
        let nb = notebook(&app, "Doomed").await;
        app.post("/api/notes", json!({ "content": "x", "notebook_id": nb })).await;
        let preview = page(&app, &format!("/notebooks/{nb}/delete")).await;
        assert_eq!(preview.status, StatusCode::OK);
        assert!(preview.body.contains("1 note(s) will be permanently deleted."), "{}", preview.body);
        let reply = form(&app, &format!("/notebooks/{nb}/delete"), "sources=keep", false).await;
        assert_eq!(reply.location.as_deref(), Some("/notebooks"));
        assert_eq!(app.get(&format!("/api/notebooks/{nb}")).await.0, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn missing_notebook_is_404() {
        let app = TestApp::new().await;
        let reply = page(&app, "/notebooks/notebook:nope").await;
        assert_eq!(reply.status, StatusCode::NOT_FOUND);
        assert!(reply.body.contains("data-error-detail"));
        assert_eq!(page(&app, "/notebooks/notebook:nope/delete").await.status, StatusCode::NOT_FOUND);
        assert_eq!(page(&app, "/notes/note:nope").await.status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn notebook_page_shows_empty_panes() {
        let app = TestApp::new().await;
        let nb = notebook(&app, "Fresh").await;
        let reply = page(&app, &format!("/notebooks/{nb}")).await;
        assert_eq!(reply.status, StatusCode::OK);
        for text in ["Fresh", "No sources yet", "No notes yet", "Chat with Notebook", "Start a conversation", "nb-chat-form"] {
            assert!(reply.body.contains(text), "{text}: {}", reply.body);
        }
        // Opening it counts as a view.
        let list = page(&app, "/notebooks").await;
        assert!(list.body.contains("Recently Viewed"));
    }

    #[tokio::test]
    async fn add_a_text_source_and_remove_it() {
        let app = TestApp::new().await;
        let nb = notebook(&app, "Sources").await;
        let (content_type, body) =
            multipart_body(&[("type", None, b"text"), ("content", None, b"Light is a wave."), ("title", None, b"Light notes")]);
        let reply = send(&app, Method::POST, &format!("/notebooks/{nb}/sources"), Some(&content_type), body, true).await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        assert!(reply.body.contains("Light notes"), "{}", reply.body);
        assert!(reply.trigger.unwrap().contains("toast"));
        let (_, sources) = app.get(&format!("/api/sources?notebook_id={nb}")).await;
        let source = sources[0]["id"].as_str().unwrap().to_string();

        // The row polls while processing and stops once done.
        let command = sources[0]["command_id"].as_str().unwrap().to_string();
        app.wait_for_job(&command).await;
        let row = htmx_get(&app, &format!("/notebooks/{nb}/sources/{source}/row")).await;
        assert_eq!(row.status, StatusCode::OK);
        assert!(!row.body.contains("every 3s"), "{}", row.body);
        assert!(row.body.contains(&format!("ctx_{source}")));

        // A text source without content is refused with the service's message.
        let (content_type, body) = multipart_body(&[("type", None, b"text")]);
        let reply = send(&app, Method::POST, &format!("/notebooks/{nb}/sources"), Some(&content_type), body, true).await;
        assert_eq!(reply.status, StatusCode::BAD_REQUEST);
        assert!(reply.body.contains("Content is required for text type"));

        let reply = form(&app, &format!("/notebooks/{nb}/sources/{source}/remove"), "", true).await;
        assert_eq!(reply.status, StatusCode::OK);
        let (_, sources) = app.get(&format!("/api/sources?notebook_id={nb}")).await;
        assert_eq!(sources.as_array().unwrap().len(), 0);
        // The source itself is kept and can be linked back.
        let existing = page(&app, &format!("/notebooks/{nb}/sources/existing")).await;
        assert!(existing.body.contains("Light notes"));
        let reply = form(&app, &format!("/notebooks/{nb}/sources/existing"), &format!("source_id={}", enc(&source)), false).await;
        assert_eq!(reply.status, StatusCode::SEE_OTHER);
        let (_, sources) = app.get(&format!("/api/sources?notebook_id={nb}")).await;
        assert_eq!(sources.as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn notes_create_edit_view_and_delete() {
        let app = TestApp::new().await;
        let nb = notebook(&app, "Notes").await;
        let reply = form(&app, &format!("/notebooks/{nb}/notes"), "title=First&content=Hello+**world**", true).await;
        assert_eq!(reply.status, StatusCode::OK);
        assert!(reply.body.contains("First") && reply.body.contains("id=\"nb-notes\""), "{}", reply.body);
        let (_, notes) = app.get(&format!("/api/notes?notebook_id={nb}")).await;
        let note = notes[0]["id"].as_str().unwrap().to_string();

        let editor = htmx_get(&app, &format!("/notebooks/{nb}/notes/{note}")).await;
        // The rich editor (mounted by script), with the plain form as the no-script fallback.
        assert!(editor.body.contains("data-nb-rich") && editor.body.contains(&format!("/notebooks/{nb}/notes/{note}/editor")), "{}", editor.body);
        assert!(editor.body.contains("Hello **world**"), "{}", editor.body);
        let full = page(&app, &format!("/notes/{note}")).await;
        assert_eq!(full.status, StatusCode::SEE_OTHER);
        assert_eq!(full.location.unwrap(), format!("/notebooks/{nb}/notes/{note}"));
        assert_eq!(page(&app, &format!("/notebooks/{nb}/notes/{note}")).await.status, StatusCode::OK);

        // In the notebook, the card opens the note as a page in the Notes column,
        // beside the collapsible Sources and Chat panes.
        let grid = page(&app, &format!("/notebooks/{nb}")).await;
        let open = format!("/notebooks/{nb}?note={}", enc(&note));
        assert!(grid.body.contains(&format!("href=\"{open}\"")), "{}", grid.body);
        let opened = page(&app, &open).await;
        assert_eq!(opened.status, StatusCode::OK);
        for text in ["nn-page", "data-nb-rich", "data-nb-expand-page", "nb-notes-behind", "In Notes", "nb-chat-form", r#"data-nb-toggle="chat" data-nb-from="rail""#] {
            assert!(opened.body.contains(text), "{text}: {}", opened.body);
        }
        assert!(opened.body.contains(&format!("data-note-href=\"/notebooks/{nb}?note=__ID__\"")), "{}", opened.body);
        assert_eq!(page(&app, &format!("/notebooks/{nb}?note=note:missing")).await.status, StatusCode::NOT_FOUND);

        let reply = form(&app, &format!("/notebooks/{nb}/notes/{note}"), "title=Renamed&content=Bye", false).await;
        assert_eq!(reply.status, StatusCode::SEE_OTHER);
        let (_, saved) = app.get(&format!("/api/notes/{note}")).await;
        assert_eq!(saved["title"], "Renamed");
        assert_eq!(saved["content"], "Bye");

        assert_eq!(form(&app, &format!("/notebooks/{nb}/notes"), "content=+", true).await.status, StatusCode::BAD_REQUEST);

        let reply = form(&app, &format!("/notebooks/{nb}/notes/{note}/delete"), "", true).await;
        assert!(reply.body.contains("No notes yet"));
        assert_eq!(app.get(&format!("/api/notes/{note}")).await.0, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn upload_a_note_file() {
        let app = TestApp::new().await;
        let nb = notebook(&app, "Files").await;
        let (content_type, body) = multipart_body(&[("file", Some("readme.md"), b"# Read me\n\nSome text.")]);
        let reply = send(&app, Method::POST, &format!("/notebooks/{nb}/notes/upload"), Some(&content_type), body, true).await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        assert!(reply.body.contains("readme.md") && reply.body.contains("MD"), "{}", reply.body);
        let (_, notes) = app.get(&format!("/api/notes?notebook_id={nb}")).await;
        let note = notes[0]["id"].as_str().unwrap();
        let viewer = htmx_get(&app, &format!("/notebooks/{nb}/notes/{note}")).await;
        assert!(viewer.body.contains("Download original") && viewer.body.contains("<h1>Read me</h1>"), "{}", viewer.body);

        let (content_type, body) = multipart_body(&[("file", Some("tool.exe"), b"MZ")]);
        let reply = send(&app, Method::POST, &format!("/notebooks/{nb}/notes/upload"), Some(&content_type), body, true).await;
        assert_eq!(reply.status, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    }

    #[tokio::test]
    async fn html_notes_render_in_a_sandbox() {
        let note = super::NoteView {
            id: "note:x".into(),
            title: Some("Page".into()),
            content: Some("text".into()),
            note_type: Some("human".into()),
            created: String::new(),
            updated: String::new(),
            command_id: None,
            attachment: Some(crate::services::notes::NoteAttachment { name: "p.html".into(), mime: "text/html".into(), size: 3 }),
            explainer: false,
            infographic: false,
        };
        let out = super::note_viewer(&note).into_string();
        assert!(out.contains("sandbox=\"\""), "{out}");
        let explainer = super::note_viewer(&super::NoteView { explainer: true, ..note.clone() }).into_string();
        assert!(explainer.contains("sandbox=\"allow-scripts\"") && !explainer.contains("allow-same-origin"), "{explainer}");
        assert!(!explainer.contains("Download original"));
        let svg = super::NoteView {
            infographic: true,
            attachment: Some(crate::services::notes::NoteAttachment { name: "i.svg".into(), mime: "image/svg+xml".into(), size: 3 }),
            ..note
        };
        assert!(super::note_viewer(&svg).into_string().contains("<img"));

        // The card's links fill the modal body rather than replacing it (the
        // notes grid's own hx-swap="outerHTML" would otherwise be inherited).
        let card = super::note_card("notebook:n", &svg).into_string();
        assert_eq!(card.matches("hx-swap=\"innerHTML\"").count(), 2, "{card}");
        assert!(super::note_body("notebook:n", &svg, true).into_string().contains("data-nb-expand"));
        assert!(!super::note_body("notebook:n", &svg, false).into_string().contains("data-nb-expand"));
    }

    #[tokio::test]
    async fn chat_sends_with_selected_context() {
        let app = TestApp::new().await;
        let model = MockLanguageModel::new(["Waves carry **energy**.", "A short title"]);
        app.state.models.set_override(AnyModel::Language(model.clone()));
        let nb = notebook(&app, "Physics").await;
        let source = text_source(&app, &nb, "Wave basics").await;
        let excluded = text_source(&app, &nb, "Unrelated").await;

        let body = format!("message=What+do+waves+do%3F&ctx_{}=full&ctx_{}=off", enc(&source), enc(&excluded));
        let reply = form(&app, &format!("/notebooks/{nb}/chat/send"), &body, true).await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        assert!(reply.body.contains("<strong>energy</strong>"), "{}", reply.body);
        assert!(reply.body.contains("What do waves do?"));
        let prompt = model.prompt(model.call_count() - 1);
        assert!(prompt.contains("Waves move energy."), "{prompt}");
        assert!(prompt.matches("Waves move energy.").count() == 1, "the excluded source stays out: {prompt}");

        // The session was created and titled from the message.
        let (_, sessions) = app.get(&format!("/api/chat/sessions?notebook_id={nb}")).await;
        let session = sessions[0]["id"].as_str().unwrap().to_string();
        assert_eq!(sessions[0]["title"], "What do waves do?");
        assert_eq!(sessions[0]["message_count"], 2);

        // Context stats count what is selected.
        let stats = form(&app, &format!("/notebooks/{nb}/chat/context"), &format!("ctx_{}=full", enc(&source)), true).await;
        assert!(stats.body.contains("1 sources (0 insights, 1 full)"), "{}", stats.body);

        // Save the answer as an AI note (titled by the model).
        let (_, full) = app.get(&format!("/api/chat/sessions/{session}")).await;
        let message = full["messages"][1]["id"].as_str().unwrap();
        let reply = form(&app, &format!("/notebooks/{nb}/chat/save-note"), &format!("session_id={}&message_id={}", enc(&session), enc(message)), true).await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        assert!(reply.trigger.unwrap().contains("notes-changed"));
        let (_, notes): (_, Value) = app.get(&format!("/api/notes?notebook_id={nb}")).await;
        assert_eq!(notes[0]["note_type"], "ai");
    }

    #[tokio::test]
    async fn chat_sessions_can_be_created_renamed_and_deleted() {
        let app = TestApp::new().await;
        let nb = notebook(&app, "Talk").await;
        let reply = form(&app, &format!("/notebooks/{nb}/chat/sessions"), "title=Planning", true).await;
        assert_eq!(reply.status, StatusCode::OK);
        assert!(reply.body.contains("Planning"));
        let (_, sessions) = app.get(&format!("/api/chat/sessions?notebook_id={nb}")).await;
        let session = sessions[0]["id"].as_str().unwrap().to_string();

        let reply = form(&app, &format!("/notebooks/{nb}/chat/sessions/{session}/rename"), "title=Review", false).await;
        assert_eq!(reply.status, StatusCode::SEE_OTHER);
        let (_, sessions) = app.get(&format!("/api/chat/sessions?notebook_id={nb}")).await;
        assert_eq!(sessions[0]["title"], "Review");

        // Another notebook cannot touch it.
        let other = notebook(&app, "Other").await;
        assert_eq!(form(&app, &format!("/notebooks/{other}/chat/sessions/{session}/delete"), "", true).await.status, StatusCode::NOT_FOUND);

        let chat = htmx_get(&app, &format!("/notebooks/{nb}/chat?session={}", enc(&session))).await;
        assert!(chat.body.contains("Review"));

        let reply = form(&app, &format!("/notebooks/{nb}/chat/sessions/{session}/delete"), "", true).await;
        assert!(reply.body.contains("No chat sessions yet"), "{}", reply.body);
    }
}
