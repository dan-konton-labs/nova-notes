//! Sources pages: the list of every source (`/sources`) and one source's
//! page (`/sources/{id}`) with its content, insights, details and chat.
//!
//! Everything calls the same services as the JSON API
//! (`services::sources`, `insights`, `source_chat`, `embedding`, ...).
//! Source chat runs the turn on the server and answers with the finished
//! chat panel (no streaming): the service produces the whole answer in one
//! step anyway, so a stream would add nothing but moving parts.

use std::collections::BTreeSet;

use axum::extract::{Multipart, Path, Query, State};
use axum::http::{HeaderMap, HeaderValue};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Form, Router};
use bytes::Bytes;
use chrono::{DateTime, Utc};
use maud::{html, Markup};
use serde::Deserialize;

use super::{
    empty_state, fragment_or_redirect, human_date, markdown, page, page_head, redirect, safe_path, status_badge, with_toast,
    Fields, PageContext, WebResult,
};
use crate::domain::Asset;
use crate::error::AppError;
use crate::services::embedding::{self, EmbedRequest};
use crate::services::insights::{self, InsightView, NewSourceInsight};
use crate::services::notebooks::{self, NotebookQuery, NotebookSummary};
use crate::services::source_chat::{
    self, CreateSourceChatSessionRequest, SendMessageRequest, SourceChatSessionWithMessages, UpdateSourceChatSessionRequest,
};
use crate::services::sources::{self, NewSource, SourceListQuery, SourceUpdate, SourceUpload, SourceView};
use crate::services::{models, settings, transformations};
use crate::state::AppState;

/// Rows per page of `/sources` (the Next.js page loaded 30 at a time).
const PAGE_SIZE: i64 = 30;
/// Most URLs or files one "Add source" may carry, as in the Next.js dialog.
const MAX_BATCH: usize = 50;
/// Sort fields of the list, with their column labels.
const SORT_FIELDS: [(&str, &str); 6] = [
    ("type", "Type"),
    ("title", "Title"),
    ("created", "Created"),
    ("updated", "Updated"),
    ("insights_count", "Insights"),
    ("embedded", "Embedded"),
];
/// Polls of a finished transformation while its insight is still being stored.
const INSIGHT_SETTLE_POLLS: u32 = 20;
/// Polls before the insights panel gives up on a pending transformation.
const INSIGHT_MAX_POLLS: u32 = 400;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/sources", get(list_page).post(add_sources))
        .route("/sources/{id}", get(detail_page))
        .route("/sources/{id}/status", get(status_fragment))
        .route("/sources/{id}/retry", post(retry))
        .route("/sources/{id}/delete", post(delete))
        .route("/sources/{id}/title", post(rename))
        .route("/sources/{id}/embed", post(embed))
        .route("/sources/{id}/notebooks", post(set_notebooks))
        .route("/sources/{id}/insights", get(insights_fragment).post(create_insight))
        .route("/sources/{id}/insights/{insight_id}/delete", post(delete_insight))
        .route("/sources/{id}/insights/{insight_id}/note", post(insight_to_note))
        .route("/sources/{id}/chat", get(chat_fragment))
        .route("/sources/{id}/chat/send", post(chat_send))
        .route("/sources/{id}/chat/sessions", post(chat_create_session))
        .route("/sources/{id}/chat/sessions/{session_id}/rename", post(chat_rename_session))
        .route("/sources/{id}/chat/sessions/{session_id}/delete", post(chat_delete_session))
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

/// Accept `abc` as well as `source:abc` in the address.
fn source_id(raw: &str) -> String {
    if raw.contains(':') { raw.to_string() } else { format!("source:{raw}") }
}

fn source_href(id: &str) -> String {
    format!("/sources/{id}")
}

fn parse_time(value: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value).ok().map(|d| d.with_timezone(&Utc))
}

fn date_text(value: Option<&str>) -> String {
    human_date(value.and_then(parse_time))
}

fn type_label(asset: Option<&Asset>) -> (&'static str, &'static str) {
    match asset {
        Some(a) if a.url.as_deref().is_some_and(|u| !u.is_empty()) => ("Link", "link"),
        Some(a) if a.file_path.as_deref().is_some_and(|p| !p.is_empty()) => ("File", "file"),
        _ => ("Text", "text"),
    }
}

fn display_title(title: Option<&str>) -> String {
    title.filter(|t| !t.trim().is_empty()).unwrap_or("Untitled Source").to_string()
}

/// Only http(s) links are rendered as links.
fn safe_external(url: &str) -> Option<String> {
    let parsed = url::Url::parse(url).ok()?;
    matches!(parsed.scheme(), "http" | "https").then(|| parsed.to_string())
}

fn is_active(status: Option<&str>) -> bool {
    matches!(status, Some("new" | "queued" | "pending" | "running"))
}

async fn all_notebooks(state: &AppState) -> WebResult<Vec<NotebookSummary>> {
    Ok(notebooks::list_notebooks(state, &NotebookQuery { archived: Some(false), ..NotebookQuery::default() }).await?)
}

// ---------------------------------------------------------------------------
// Status
// ---------------------------------------------------------------------------

/// The processing status of a source: a badge, the error when it failed, a
/// retry button, and polling while it is still working. `refresh` reloads
/// the page once processing ends (the detail page then shows the content).
fn status_block(id: &str, status: Option<&str>, error: Option<&str>, compact: bool) -> Markup {
    let active = is_active(status);
    let failed = matches!(status, Some("failed" | "error" | "canceled" | "cancelled"));
    let poll_url = format!("{}/status?compact={}&was_active={}", source_href(id), compact, active);
    html! {
        div.src-status.row
            hx-get=[active.then_some(poll_url.as_str())]
            hx-trigger=[active.then_some("every 3s")]
            hx-swap=[active.then_some("outerHTML")] {
            (status_badge(status))
            @if status.is_none() && !compact { span.faint.small { "Processed" } }
            @if failed {
                @if let Some(error) = error.filter(|e| !e.is_empty()) {
                    span.small.src-error title=(error) { (error) }
                }
                form.inline method="post" action=(format!("{}/retry", source_href(id)))
                    hx-post=(format!("{}/retry", source_href(id))) hx-target="closest .src-status" hx-swap="outerHTML" {
                    input type="hidden" name="compact" value=(compact);
                    button.btn.btn-sm type="submit" { "Retry" }
                }
            }
        }
    }
}

#[derive(Debug, Default, Deserialize)]
struct StatusParams {
    #[serde(default)]
    compact: Option<String>,
    #[serde(default)]
    was_active: Option<String>,
}

fn flag(value: Option<&str>) -> bool {
    matches!(value, Some("true" | "1" | "on"))
}

async fn status_fragment(State(state): State<AppState>, Path(id): Path<String>, Query(params): Query<StatusParams>) -> WebResult<Response> {
    let id = source_id(&id);
    let status = sources::source_status(&state, &id).await?;
    let error = status.processing_info.as_ref().and_then(|i| i.get("error")).and_then(|e| e.as_str()).map(String::from);
    let compact = flag(params.compact.as_deref());
    let mut response = status_block(&id, status.status.as_deref(), error.as_deref(), compact).into_response();
    // The detail page reloads once processing ends, to show the content.
    if !compact && flag(params.was_active.as_deref()) && !is_active(status.status.as_deref()) {
        response.headers_mut().insert("HX-Refresh", HeaderValue::from_static("true"));
    }
    Ok(response)
}

#[derive(Debug, Default, Deserialize)]
struct CompactForm {
    #[serde(default)]
    compact: Option<String>,
}

async fn retry(State(state): State<AppState>, Path(id): Path<String>, headers: HeaderMap, Form(form): Form<CompactForm>) -> WebResult<Response> {
    let id = source_id(&id);
    let view = sources::retry_source(&state, &id).await?;
    let block = status_block(&id, view.status.as_deref(), None, flag(form.compact.as_deref()));
    Ok(with_toast(fragment_or_redirect(&headers, block, &source_href(&id)), "The source has been requeued for processing."))
}

// ---------------------------------------------------------------------------
// /sources
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
struct ListParams {
    #[serde(default)]
    page: Option<i64>,
    #[serde(default)]
    sort_by: Option<String>,
    #[serde(default)]
    sort_order: Option<String>,
}

fn list_href(page: i64, sort_by: &str, sort_order: &str) -> String {
    format!("/sources?page={page}&sort_by={sort_by}&sort_order={sort_order}")
}

async fn list_page(State(state): State<AppState>, Query(params): Query<ListParams>) -> WebResult<Markup> {
    let sort_by = params.sort_by.filter(|s| SORT_FIELDS.iter().any(|(f, _)| f == s)).unwrap_or_else(|| "updated".into());
    let sort_order = match params.sort_order.as_deref() {
        Some("asc") => "asc",
        _ => "desc",
    };
    let page_number = params.page.unwrap_or(1).max(1);
    let query = SourceListQuery {
        notebook_id: None,
        limit: PAGE_SIZE + 1,
        offset: (page_number - 1) * PAGE_SIZE,
        sort_by: sort_by.clone(),
        sort_order: sort_order.into(),
    };
    let mut rows = sources::list_sources(&state, &query).await?;
    let has_more = rows.len() as i64 > PAGE_SIZE;
    rows.truncate(PAGE_SIZE as usize);
    let dialog = add_source_dialog(&state, None, "/sources").await?;

    let header = |field: &str, label: &str| {
        let active = sort_by == field;
        let next_order = if active && sort_order == "desc" { "asc" } else { "desc" };
        let arrow = if !active { "" } else if sort_order == "asc" { " ↑" } else { " ↓" };
        html! {
            th aria-sort=[active.then_some(if sort_order == "asc" { "ascending" } else { "descending" })] {
                a.src-sort href=(list_href(1, field, next_order)) { (label) (arrow) }
            }
        }
    };

    let body = html! {
        (page_head(
            "All Sources",
            Some("View all your sources here. You can add new sources or manage existing ones."),
            html! { button.btn.btn-primary type="button" data-open="#src-add" { "New Source" } },
        ))
        @if rows.is_empty() && page_number == 1 {
            div.empty {
                strong { "No sources yet" }
                "View all your sources here."
                div.row.src-center { button.btn type="button" data-open="#src-add" { "New Source" } }
            }
        } @else if rows.is_empty() {
            (empty_state("No more sources", "This page is past the last source."))
        } @else {
            div.table-wrap {
                table.src-table {
                    thead {
                        tr {
                            @for (field, label) in SORT_FIELDS { (header(field, label)) }
                            th { "Status" }
                            th.src-actions { "Actions" }
                        }
                    }
                    tbody {
                        @for row in &rows {
                            @let (type_name, type_class) = type_label(row.asset.as_ref());
                            @let title = display_title(row.title.as_deref());
                            @let error = row.processing_info.as_ref().and_then(|i| i.get("error")).and_then(|e| e.as_str());
                            tr {
                                td { span class=(format!("src-type src-type-{type_class}")) { (type_name) } }
                                td.src-title-cell {
                                    a.item-title href=(source_href(&row.id)) { (title) }
                                    @if let Some(url) = row.asset.as_ref().and_then(|a| a.url.as_deref()) {
                                        div.faint.small.truncate { (url) }
                                    }
                                }
                                td.muted.small { (date_text(Some(&row.created))) }
                                td.muted.small { (date_text(Some(&row.updated))) }
                                td { (row.insights_count) }
                                td {
                                    @if row.embedded { span.badge.ok { "Yes" } } @else { span.badge { "No" } }
                                }
                                td { (status_block(&row.id, row.status.as_deref(), error, true)) }
                                td.src-actions {
                                    form.inline method="post" action=(format!("{}/delete", source_href(&row.id)))
                                        hx-post=(format!("{}/delete", source_href(&row.id)))
                                        hx-target="closest tr" hx-swap="outerHTML"
                                        hx-confirm=(format!("Are you sure you want to delete \"{title}\"?")) {
                                        button.btn.btn-sm.btn-danger type="submit" { "Delete" }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        @if page_number > 1 || has_more {
            nav.row-between.src-pager aria-label="Pages" {
                @if page_number > 1 {
                    a.btn.btn-sm href=(list_href(page_number - 1, &sort_by, sort_order)) { "← Previous" }
                } @else { span {} }
                span.faint.small { "Page " (page_number) }
                @if has_more {
                    a.btn.btn-sm href=(list_href(page_number + 1, &sort_by, sort_order)) { "Next →" }
                } @else { span {} }
            }
        }
        (dialog)
    };
    Ok(page(&PageContext::new(&state, "/sources"), "Sources", body))
}

// ---------------------------------------------------------------------------
// Adding sources
// ---------------------------------------------------------------------------

/// The "Add Source" dialog (`#src-add`), opened by `data-open="#src-add"`.
/// `notebook_id` is ticked in advance; after adding, the browser goes to
/// `return_to`. Other pages (the notebook page) can embed it.
pub async fn add_source_dialog(state: &AppState, notebook_id: Option<&str>, return_to: &str) -> WebResult<Markup> {
    let notebooks = all_notebooks(state).await?;
    let transformations = transformations::list_transformations(state).await?;
    let embed_option = settings::get_settings(state).await?.default_embedding_option.unwrap_or_else(|| "ask".into());
    Ok(html! {
        dialog #src-add.wide aria-labelledby="src-add-title" {
            form.src-add-form method="post" action="/sources" enctype="multipart/form-data"
                hx-post="/sources" hx-encoding="multipart/form-data" hx-disabled-elt="find button[type=submit]" {
                input type="hidden" name="return_to" value=(return_to);
                div.dialog-head {
                    h2 #src-add-title { "Add Source" }
                    button.btn.btn-ghost.btn-sm type="button" data-close { "Close" }
                }
                div.dialog-body {
                    p.muted.small { "Content will be processed and analyzed by AI." }
                    fieldset.src-type-pick {
                        legend.src-sr { "Source type" }
                        label.check { input type="radio" name="type" value="link" checked; "Add URL" }
                        label.check { input type="radio" name="type" value="upload"; "Upload File" }
                        label.check { input type="radio" name="type" value="text"; "Enter Text" }
                    }
                    div.field data-src-for="link" {
                        label for="src-url" { "URL(s) *" }
                        textarea #src-url name="url" rows="3" placeholder="Enter URLs, one per line" {}
                        div.hint { "Paste multiple URLs (one per line) to batch import" }
                    }
                    div.field data-src-for="upload" {
                        label for="src-file" { "File(s) *" }
                        input #src-file type="file" name="file" multiple;
                        div.hint {
                            "Select multiple files to batch import. Supported: documents (PDF, DOCX, PPTX, XLSX, EPUB, HTML, TXT, MD), media (MP4, MP3, WAV, M4A) and images."
                        }
                    }
                    div.field data-src-for="text" {
                        label for="src-content" { "Text Content *" }
                        textarea #src-content name="content" rows="6" placeholder="Paste or type your content here..." {}
                    }
                    div.field {
                        label for="src-new-title" { "Title" }
                        input #src-new-title type="text" name="title" placeholder="Give your source a descriptive title";
                        div.hint { "Required for text. If left empty for links and files, a title is generated from the content (batches always generate titles)." }
                    }
                    div.field {
                        label { "Notebooks" }
                        @if notebooks.is_empty() {
                            p.faint.small { "No notebooks found." }
                        } @else {
                            div.src-checks {
                                @for nb in &notebooks {
                                    label.check {
                                        input type="checkbox" name="notebooks" value=(nb.id) checked[notebook_id == Some(nb.id.as_str())];
                                        (nb.name)
                                    }
                                }
                            }
                        }
                    }
                    @if !transformations.is_empty() {
                        div.field {
                            label { "Transformations" }
                            div.src-checks {
                                @for t in &transformations {
                                    label.check title=(t.description) {
                                        input type="checkbox" name="transformations" value=(t.id) checked[t.apply_default];
                                        (t.title)
                                    }
                                }
                            }
                        }
                    }
                    @match embed_option.as_str() {
                        "always" => {
                            input type="hidden" name="embed" value="true";
                            p.small.muted { strong { "Embedding enabled automatically. " } "Your settings are configured to always embed content for vector search." }
                        }
                        "never" => {
                            p.small.muted { strong { "Embedding disabled. " } "Your settings are configured to skip embedding. Vector search won't be available for this source." }
                        }
                        _ => {
                            label.check {
                                input type="checkbox" name="embed" value="true" checked;
                                "Enable embedding for search"
                            }
                            div.hint { "Allows this source to be found in vector searches and AI queries" }
                        }
                    }
                }
                div.dialog-foot {
                    span.spinner.htmx-indicator aria-hidden="true" {}
                    button.btn type="button" data-close { "Cancel" }
                    button.btn.btn-primary type="submit" { "Add Source" }
                }
            }
        }
    })
}

/// What the "Add Source" form sent.
#[derive(Debug, Default)]
struct AddForm {
    source_type: String,
    urls: Vec<String>,
    content: Option<String>,
    title: Option<String>,
    notebooks: Vec<String>,
    transformations: Vec<String>,
    embed: bool,
    files: Vec<SourceUpload>,
    return_to: Option<String>,
}

async fn read_add_form(mut multipart: Multipart) -> Result<AddForm, AppError> {
    let mut form = AddForm::default();
    while let Some(field) = multipart.next_field().await.map_err(|e| AppError::bad_request(e.body_text()))? {
        let name = field.name().unwrap_or_default().to_string();
        if name == "file" {
            let filename = field.file_name().unwrap_or_default().to_string();
            let data = field.bytes().await.map_err(|e| AppError::bad_request(e.body_text()))?;
            // An empty file input arrives as a nameless, empty part.
            if !(filename.is_empty() && data.is_empty()) {
                form.files.push(SourceUpload { filename, data: data.to_vec() });
            }
            continue;
        }
        let value = field.text().await.map_err(|e| AppError::bad_request(e.body_text()))?;
        match name.as_str() {
            "type" => form.source_type = value,
            "url" => form.urls = value.lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect(),
            "content" => form.content = Some(value).filter(|v| !v.trim().is_empty()),
            "title" => form.title = Some(value.trim().to_string()).filter(|v| !v.is_empty()),
            "notebooks" if !value.is_empty() => form.notebooks.push(value),
            "transformations" if !value.is_empty() => form.transformations.push(value),
            "embed" => form.embed = flag(Some(value.as_str())),
            "return_to" => form.return_to = safe_path(Some(&value)),
            _ => {}
        }
    }
    Ok(form)
}

/// One `NewSource` per URL / file / text, and the upload that goes with it.
fn plan_sources(form: AddForm) -> Result<Vec<(NewSource, Option<SourceUpload>)>, AppError> {
    let base = NewSource {
        source_type: form.source_type.clone(),
        notebooks: Some(form.notebooks.clone()),
        transformations: Some(form.transformations.clone()),
        embed: form.embed,
        delete_source: false,
        async_processing: true,
        ..NewSource::default()
    };
    let batch = |count: usize| -> Result<(), AppError> {
        if count > MAX_BATCH {
            return Err(AppError::bad_request(format!("Maximum {MAX_BATCH} items allowed per batch")));
        }
        Ok(())
    };
    match form.source_type.as_str() {
        "link" => {
            if form.urls.is_empty() {
                return Err(AppError::bad_request("URL is required for link type"));
            }
            batch(form.urls.len())?;
            let single = form.urls.len() == 1;
            Ok(form
                .urls
                .into_iter()
                .map(|url| {
                    let title = if single { form.title.clone() } else { None };
                    (NewSource { url: Some(url), title, ..base.clone() }, None)
                })
                .collect())
        }
        "upload" => {
            if form.files.is_empty() {
                return Err(AppError::bad_request("File upload or file_path is required for upload type"));
            }
            batch(form.files.len())?;
            let single = form.files.len() == 1;
            Ok(form
                .files
                .into_iter()
                .map(|file| {
                    let title = if single { form.title.clone() } else { None };
                    (NewSource { title, ..base.clone() }, Some(file))
                })
                .collect())
        }
        "text" => {
            if form.title.is_none() {
                return Err(AppError::bad_request("A title is required for text content"));
            }
            let content = form.content.clone().ok_or_else(|| AppError::bad_request("Content is required for text type"))?;
            Ok(vec![(NewSource { content: Some(content), title: form.title.clone(), ..base }, None)])
        }
        _ => Err(AppError::bad_request("Invalid source type. Must be link, upload, or text")),
    }
}

async fn add_sources(State(state): State<AppState>, headers: HeaderMap, multipart: Multipart) -> WebResult<Response> {
    let form = read_add_form(multipart).await?;
    let return_to = form.return_to.clone().unwrap_or_else(|| "/sources".to_string());
    let planned = plan_sources(form)?;
    let total = planned.len();
    let mut created = 0usize;
    let mut first_error: Option<AppError> = None;
    for (new, upload) in planned {
        match sources::create_source(&state, new, upload).await {
            Ok(_) => created += 1,
            Err(e) => {
                tracing::warn!("Adding a source from the web UI failed: {e}");
                first_error.get_or_insert(e);
            }
        }
    }
    if let Some(error) = first_error {
        if created == 0 {
            return Err(error.into());
        }
        return Err(AppError::bad_request(format!(
            "{created} succeeded, {} failed. First error: {}",
            total - created,
            error.detail()
        ))
        .into());
    }
    Ok(redirect(&headers, &return_to))
}

// ---------------------------------------------------------------------------
// Deleting, renaming, embedding
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
struct ReturnForm {
    #[serde(default)]
    return_to: Option<String>,
}

async fn delete(State(state): State<AppState>, Path(id): Path<String>, headers: HeaderMap, Form(form): Form<ReturnForm>) -> WebResult<Response> {
    let id = source_id(&id);
    sources::delete_source(&state, &id).await?;
    let target = safe_path(form.return_to.as_deref());
    let response = match target {
        // From the detail page: leave it.
        Some(target) => redirect(&headers, &target),
        // From a list row: the empty answer removes the row.
        None => fragment_or_redirect(&headers, html! {}, "/sources"),
    };
    Ok(with_toast(response, "Source deleted successfully"))
}

fn title_block(id: &str, title: Option<&str>) -> Markup {
    let action = format!("{}/title", source_href(id));
    html! {
        div #src-title.src-title {
            h1 { (display_title(title)) }
            details.src-rename {
                summary.btn.btn-ghost.btn-sm { "Rename" }
                form.row method="post" action=(action) hx-post=(action) hx-target="#src-title" hx-swap="outerHTML" {
                    input type="text" name="title" value=(title.unwrap_or_default()) placeholder="Give your source a descriptive title" required aria-label="Title";
                    button.btn.btn-primary.btn-sm type="submit" { "Save" }
                }
            }
        }
    }
}

#[derive(Debug, Deserialize)]
struct TitleForm {
    title: String,
}

async fn rename(State(state): State<AppState>, Path(id): Path<String>, headers: HeaderMap, Form(form): Form<TitleForm>) -> WebResult<Response> {
    let id = source_id(&id);
    let title = form.title.trim().to_string();
    if title.is_empty() {
        return Err(AppError::bad_request("Title cannot be empty").into());
    }
    let view = sources::update_source(&state, &id, &SourceUpdate { title: Some(title), topics: None }).await?;
    let response = fragment_or_redirect(&headers, title_block(&id, view.title.as_deref()), &source_href(&id));
    Ok(with_toast(response, "Source updated successfully"))
}

async fn embed(State(state): State<AppState>, Path(id): Path<String>, headers: HeaderMap) -> WebResult<Response> {
    let id = source_id(&id);
    sources::load_source(&state, &id).await?;
    let result = embedding::embed_item(&state, &EmbedRequest { item_id: id.clone(), item_type: "source".into(), async_processing: true }).await?;
    let fragment = html! { span.badge.info { "Embedding queued" } };
    Ok(with_toast(fragment_or_redirect(&headers, fragment, &source_href(&id)), &result.message))
}

// ---------------------------------------------------------------------------
// Notebook associations
// ---------------------------------------------------------------------------

fn notebooks_form(id: &str, all: &[NotebookSummary], linked: &[String]) -> Markup {
    let action = format!("{}/notebooks", source_href(id));
    html! {
        form #src-notebooks.stack-sm method="post" action=(action) hx-post=(action) hx-target="this" hx-swap="outerHTML" {
            h3 { "Notebooks" }
            p.faint.small { "Manage which notebooks contain this source" }
            @if all.is_empty() {
                p.faint.small { "No notebooks available" }
            } @else {
                div.src-checks {
                    @for nb in all {
                        label.check {
                            input type="checkbox" name="notebooks" value=(nb.id) checked[linked.contains(&nb.id)];
                            a href=(format!("/notebooks/{}", nb.id)) { (nb.name) }
                        }
                    }
                }
                div.row { button.btn.btn-sm type="submit" { "Save notebooks" } }
            }
        }
    }
}

async fn set_notebooks(State(state): State<AppState>, Path(id): Path<String>, headers: HeaderMap, body: Bytes) -> WebResult<Response> {
    let id = source_id(&id);
    let wanted: BTreeSet<String> =
        Fields::parse(&body).all("notebooks").into_iter().collect();
    let view = sources::get_source(&state, &id).await?;
    let current: BTreeSet<String> = view.notebooks.unwrap_or_default().into_iter().collect();
    for notebook_id in wanted.difference(&current) {
        notebooks::add_source_to_notebook(&state, notebook_id, &id).await?;
    }
    for notebook_id in current.difference(&wanted) {
        notebooks::remove_source_from_notebook(&state, notebook_id, &id).await?;
    }
    let all = all_notebooks(&state).await?;
    let linked: Vec<String> = wanted.into_iter().collect();
    let target = format!("{}?tab=details", source_href(&id));
    Ok(with_toast(fragment_or_redirect(&headers, notebooks_form(&id, &all, &linked), &target), "Notebooks updated"))
}

// ---------------------------------------------------------------------------
// Insights
// ---------------------------------------------------------------------------

/// A transformation still being turned into an insight.
#[derive(Debug, Clone, Default, Deserialize)]
struct Pending {
    #[serde(default)]
    pending: Option<String>,
    /// Insights the source had when the transformation started.
    #[serde(default)]
    base: Option<usize>,
    #[serde(default)]
    tries: Option<u32>,
    /// Polls since the transformation's job finished.
    #[serde(default)]
    settle: Option<u32>,
}

struct InsightsPanel<'a> {
    id: &'a str,
    insights: &'a [InsightView],
    transformations: &'a [transformations::TransformationView],
    notebooks: &'a [NotebookSummary],
    pending: Option<Pending>,
}

fn insights_panel(panel: InsightsPanel<'_>) -> Markup {
    let id = panel.id;
    let base_url = format!("{}/insights", source_href(id));
    let poll = panel.pending.as_ref().and_then(|p| {
        let command = p.pending.as_deref()?;
        Some(format!(
            "{base_url}?pending={command}&base={}&tries={}&settle={}",
            p.base.unwrap_or(0),
            p.tries.unwrap_or(0),
            p.settle.unwrap_or(0)
        ))
    });
    html! {
        section #src-insights.stack
            hx-get=[poll.as_deref()] hx-trigger=[poll.as_ref().map(|_| "every 3s")] hx-swap=[poll.as_ref().map(|_| "outerHTML")] {
            div {
                h2 { "Insights" }
                p.muted.small { "Insights generated from model analysis" }
            }
            div.card.flat.stack-sm {
                h3 { "Generate New Insight" }
                @if panel.transformations.is_empty() {
                    p.faint.small { "No transformations yet. Create one under " a href="/transformations" { "Transformations" } "." }
                } @else {
                    form.row method="post" action=(base_url) hx-post=(base_url) hx-target="#src-insights" hx-swap="outerHTML"
                        hx-disabled-elt="find button" {
                        select.grow name="transformation_id" required aria-label="Transformation" {
                            option value="" { "Select a transformation..." }
                            @for t in panel.transformations { option value=(t.id) { (t.title) } }
                        }
                        button.btn.btn-primary type="submit" { "Create" }
                    }
                }
                @if poll.is_some() {
                    div.row.small.muted { span.spinner aria-hidden="true" {} "Insight generation started. It will appear shortly." }
                }
            }
            @if panel.insights.is_empty() {
                (empty_state("No insights yet", "Create your first insight using a transformation above"))
            } @else {
                @for insight in panel.insights { (insight_item(id, insight, panel.notebooks)) }
            }
        }
    }
}

fn insight_item(id: &str, insight: &InsightView, notebooks: &[NotebookSummary]) -> Markup {
    let base = format!("{}/insights/{}", source_href(id), insight.id);
    html! {
        details.card.flat.src-insight {
            summary.row-between {
                span.row {
                    span.badge.info { (insight.insight_type) }
                    span.faint.small { (date_text(insight.created.as_deref())) }
                }
                span.faint.small { "View Insight" }
            }
            div.stack-sm.src-insight-body {
                (markdown(&insight.content))
                div.row-between {
                    form.row method="post" action=(format!("{base}/note")) hx-post=(format!("{base}/note")) hx-swap="none" {
                        select name="notebook_id" aria-label="Notebook for the note" {
                            option value="" { "No notebook" }
                            @for nb in notebooks { option value=(nb.id) { (nb.name) } }
                        }
                        button.btn.btn-sm type="submit" { "Save as note" }
                    }
                    form.inline method="post" action=(format!("{base}/delete")) hx-post=(format!("{base}/delete"))
                        hx-target="#src-insights" hx-swap="outerHTML"
                        hx-confirm="Are you sure you want to delete this insight? This action cannot be undone." {
                        button.btn.btn-sm.btn-danger type="submit" { "Delete Insight" }
                    }
                }
            }
        }
    }
}

/// The notebooks a source belongs to, for "Save as note".
async fn source_notebooks(state: &AppState, view: &SourceView) -> WebResult<Vec<NotebookSummary>> {
    let linked = view.notebooks.clone().unwrap_or_default();
    Ok(all_notebooks(state).await?.into_iter().filter(|nb| linked.contains(&nb.id)).collect())
}

async fn render_insights(state: &AppState, id: &str, pending: Option<Pending>) -> WebResult<Markup> {
    let view = sources::get_source(state, id).await?;
    let insights = insights::list_source_insights(state, id).await?;
    let transformations = transformations::list_transformations(state).await?;
    let notebooks = source_notebooks(state, &view).await?;
    Ok(insights_panel(InsightsPanel { id, insights: &insights, transformations: &transformations, notebooks: &notebooks, pending }))
}

/// The insights panel; with `pending` it checks on a running transformation
/// and keeps polling until its insight is stored (or it failed).
async fn insights_fragment(State(state): State<AppState>, Path(id): Path<String>, Query(pending): Query<Pending>) -> WebResult<Response> {
    let id = source_id(&id);
    let Some(command) = pending.pending.clone().filter(|c| !c.is_empty()) else {
        return Ok(render_insights(&state, &id, None).await?.into_response());
    };
    let tries = pending.tries.unwrap_or(0) + 1;
    let base = pending.base.unwrap_or(0);
    let record = state.jobs.get(&command).await?;
    let count = insights::list_source_insights(&state, &id).await?.len();
    let (still_pending, settle, failure) = match record {
        None => (false, 0, Some("The insight job could not be found".to_string())),
        Some(record) if !record.is_terminal() => (true, 0, None),
        Some(record) if record.status_str() != crate::jobs::status::COMPLETED => {
            (false, 0, Some(record.error_message.unwrap_or_else(|| "Insight generation failed".to_string())))
        }
        // Finished: the insight is stored by a follow-up job, so wait for it a little.
        Some(_) => {
            let settle = pending.settle.unwrap_or(0) + 1;
            (count <= base && settle < INSIGHT_SETTLE_POLLS, settle, None)
        }
    };
    let keep = still_pending && tries < INSIGHT_MAX_POLLS;
    let next = keep.then_some(Pending { pending: Some(command), base: Some(base), tries: Some(tries), settle: Some(settle) });
    let response = render_insights(&state, &id, next).await?.into_response();
    Ok(match failure {
        Some(message) => with_toast(response, &message),
        None => response,
    })
}

async fn create_insight(State(state): State<AppState>, Path(id): Path<String>, headers: HeaderMap, Form(form): Form<NewSourceInsight>) -> WebResult<Response> {
    let id = source_id(&id);
    if form.transformation_id.is_empty() {
        return Err(AppError::bad_request("Select a transformation...").into());
    }
    let base = insights::list_source_insights(&state, &id).await?.len();
    let created = insights::create_source_insight(&state, &id, &form).await?;
    let pending = Pending { pending: created.command_id, base: Some(base), tries: Some(0), settle: Some(0) };
    let panel = render_insights(&state, &id, Some(pending)).await?;
    let target = format!("{}?tab=insights", source_href(&id));
    Ok(with_toast(fragment_or_redirect(&headers, panel, &target), "Insight generation started. It will appear shortly."))
}

/// The insight, if it belongs to this source.
async fn source_insight(state: &AppState, id: &str, insight_id: &str) -> WebResult<InsightView> {
    let insight = insights::get_insight(state, insight_id).await?;
    if insight.source_id != id {
        return Err(AppError::not_found(insights::INSIGHT_NOT_FOUND).into());
    }
    Ok(insight)
}

async fn delete_insight(State(state): State<AppState>, Path((id, insight_id)): Path<(String, String)>, headers: HeaderMap) -> WebResult<Response> {
    let id = source_id(&id);
    source_insight(&state, &id, &insight_id).await?;
    insights::delete_insight(&state, &insight_id).await?;
    let panel = render_insights(&state, &id, None).await?;
    let target = format!("{}?tab=insights", source_href(&id));
    Ok(with_toast(fragment_or_redirect(&headers, panel, &target), "Insight deleted"))
}

#[derive(Debug, Default, Deserialize)]
struct NoteForm {
    #[serde(default)]
    notebook_id: Option<String>,
}

async fn insight_to_note(
    State(state): State<AppState>,
    Path((id, insight_id)): Path<(String, String)>,
    headers: HeaderMap,
    Form(form): Form<NoteForm>,
) -> WebResult<Response> {
    let id = source_id(&id);
    source_insight(&state, &id, &insight_id).await?;
    insights::save_insight_as_note(&state, &insight_id, form.notebook_id.as_deref()).await?;
    let target = format!("{}?tab=insights", source_href(&id));
    Ok(with_toast(fragment_or_redirect(&headers, html! {}, &target), "Saved as note"))
}

// ---------------------------------------------------------------------------
// Source chat
// ---------------------------------------------------------------------------

/// The chat column: sessions, messages and the message form.
async fn chat_panel(state: &AppState, id: &str, session: Option<&str>) -> WebResult<Markup> {
    let sessions = source_chat::list_sessions(state, id).await?;
    let selected = session.filter(|s| !s.is_empty()).map(String::from).or_else(|| sessions.first().map(|s| s.id.clone()));
    let current: Option<SourceChatSessionWithMessages> = match selected.as_deref() {
        Some(session_id) => Some(source_chat::get_session(state, id, session_id).await?),
        None => None,
    };
    let language_models = models::list_models(state, Some("language")).await?;
    let href = source_href(id);
    let current_id = current.as_ref().map(|c| c.session.id.clone()).unwrap_or_default();
    let model_override = current.as_ref().and_then(|c| c.session.model_override.clone()).unwrap_or_default();
    Ok(html! {
        section #src-chat.panel.src-chat {
            div.panel-head {
                h2 { "Chat with Source" }
                form.row method="post" action=(format!("{href}/chat/sessions")) hx-post=(format!("{href}/chat/sessions"))
                    hx-target="#src-chat" hx-swap="outerHTML" {
                    button.btn.btn-sm type="submit" { "New session" }
                }
            }
            div.panel-body {
                @if sessions.is_empty() {
                    p.faint.small { "No chat sessions yet" }
                } @else {
                    details.src-sessions {
                        summary.small {
                            "Sessions (" (sessions.len()) ")"
                            @if let Some(c) = &current { " · " strong { (c.session.title) } }
                        }
                        ul.src-session-list {
                            @for s in &sessions {
                                li.row-between {
                                    a href=(format!("{href}?session={}", s.id)) aria-current=[(s.id == current_id).then_some("true")] {
                                        (s.title)
                                    }
                                    span.faint.small { (s.message_count.unwrap_or(0)) " messages" }
                                }
                            }
                        }
                        @if let Some(c) = &current {
                            @let base = format!("{href}/chat/sessions/{}", c.session.id);
                            div.row.src-session-tools {
                                form.row.grow method="post" action=(format!("{base}/rename")) hx-post=(format!("{base}/rename"))
                                    hx-target="#src-chat" hx-swap="outerHTML" {
                                    input.grow type="text" name="title" value=(c.session.title) placeholder="Type a title here..." aria-label="Session title" required;
                                    button.btn.btn-sm type="submit" { "Rename" }
                                }
                                form.inline method="post" action=(format!("{base}/delete")) hx-post=(format!("{base}/delete"))
                                    hx-target="#src-chat" hx-swap="outerHTML"
                                    hx-confirm="Are you sure you want to delete this chat session? This action cannot be undone." {
                                    button.btn.btn-sm.btn-danger type="submit" { "Delete Session" }
                                }
                            }
                        }
                    }
                }
                div.chat-log.src-chat-log {
                    @match &current {
                        Some(c) if !c.messages.is_empty() => {
                            @for m in &c.messages {
                                @if m.message_type == "human" {
                                    div.msg.human.pre-wrap { (m.content) }
                                } @else {
                                    div.msg.ai { (markdown(&m.content)) }
                                }
                            }
                        }
                        _ => {
                            div.empty {
                                strong { "Start a conversation about this source" }
                                "Ask questions to understand the content better"
                            }
                        }
                    }
                }
                @if let Some(ind) = current.as_ref().and_then(|c| c.context_indicators.as_ref()) {
                    p.faint.small {
                        "Context: " (ind.sources.len()) " source, " (ind.insights.len()) " insights"
                    }
                }
                form.chat-form.src-chat-form method="post" action=(format!("{href}/chat/send")) hx-post=(format!("{href}/chat/send"))
                    hx-target="#src-chat" hx-swap="outerHTML" hx-disabled-elt="find button" {
                    input type="hidden" name="session_id" value=(current_id);
                    div.grow.stack-sm {
                        textarea name="message" required placeholder="Ask anything about this source..." aria-label="Message" {}
                        @if !language_models.is_empty() {
                            select name="model_override" aria-label="Model" {
                                option value="" { "Default model" }
                                @for m in &language_models {
                                    option value=(m.id) selected[m.id == model_override] { (m.name) " (" (m.provider) ")" }
                                }
                            }
                        }
                    }
                    button.btn.btn-primary type="submit" { span.spinner.htmx-indicator aria-hidden="true" {} "Send" }
                }
                p.faint.small { "Press Enter to send, Shift+Enter for a new line" }
            }
        }
    })
}

#[derive(Debug, Default, Deserialize)]
struct SessionParam {
    #[serde(default)]
    session: Option<String>,
}

async fn chat_fragment(State(state): State<AppState>, Path(id): Path<String>, Query(params): Query<SessionParam>) -> WebResult<Markup> {
    let id = source_id(&id);
    chat_panel(&state, &id, params.session.as_deref()).await
}

fn session_target(id: &str, session_id: Option<&str>) -> String {
    match session_id {
        Some(s) => format!("{}?session={s}", source_href(id)),
        None => source_href(id),
    }
}

#[derive(Debug, Default, Deserialize)]
struct NewSessionForm {
    #[serde(default)]
    title: Option<String>,
}

async fn chat_create_session(State(state): State<AppState>, Path(id): Path<String>, headers: HeaderMap, Form(form): Form<NewSessionForm>) -> WebResult<Response> {
    let id = source_id(&id);
    let request = CreateSourceChatSessionRequest { source_id: id.clone(), title: form.title.filter(|t| !t.trim().is_empty()), model_override: None };
    let session = source_chat::create_session(&state, &id, &request).await?;
    let panel = chat_panel(&state, &id, Some(&session.id)).await?;
    Ok(with_toast(fragment_or_redirect(&headers, panel, &session_target(&id, Some(&session.id))), "Chat session created"))
}

async fn chat_rename_session(
    State(state): State<AppState>,
    Path((id, session_id)): Path<(String, String)>,
    headers: HeaderMap,
    Form(form): Form<TitleForm>,
) -> WebResult<Response> {
    let id = source_id(&id);
    let title = form.title.trim().to_string();
    if title.is_empty() {
        return Err(AppError::bad_request("Title cannot be empty").into());
    }
    let request = UpdateSourceChatSessionRequest { title: Some(title), model_override: None };
    let session = source_chat::update_session(&state, &id, &session_id, &request).await?;
    let panel = chat_panel(&state, &id, Some(&session.id)).await?;
    Ok(fragment_or_redirect(&headers, panel, &session_target(&id, Some(&session.id))))
}

async fn chat_delete_session(State(state): State<AppState>, Path((id, session_id)): Path<(String, String)>, headers: HeaderMap) -> WebResult<Response> {
    let id = source_id(&id);
    source_chat::delete_session(&state, &id, &session_id).await?;
    let panel = chat_panel(&state, &id, None).await?;
    Ok(with_toast(fragment_or_redirect(&headers, panel, &source_href(&id)), "Chat session deleted"))
}

#[derive(Debug, Default, Deserialize)]
struct SendForm {
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    message: String,
    #[serde(default)]
    model_override: Option<String>,
}

/// Send a message: creates a session when there is none, runs the turn and
/// answers with the updated chat panel.
async fn chat_send(State(state): State<AppState>, Path(id): Path<String>, headers: HeaderMap, Form(form): Form<SendForm>) -> WebResult<Response> {
    let id = source_id(&id);
    let message = form.message.trim().to_string();
    if message.is_empty() {
        return Err(AppError::bad_request("Message content is required").into());
    }
    // An empty choice is "Default model", which clears a saved override.
    let picked_default = form.model_override.as_deref() == Some("");
    let model_override = form.model_override.filter(|m| !m.is_empty());
    let session_id = match form.session_id.filter(|s| !s.is_empty()) {
        Some(session_id) => {
            if picked_default {
                // Before the turn, so this message already uses the default.
                source_chat::set_model_override(&state, &id, &session_id, None).await?;
            }
            session_id
        }
        None => {
            let request = CreateSourceChatSessionRequest { source_id: id.clone(), title: None, model_override: model_override.clone() };
            source_chat::create_session(&state, &id, &request).await?.id
        }
    };
    let turn = source_chat::prepare_message(
        &state,
        &id,
        &session_id,
        &SendMessageRequest { message, model_override: model_override.clone() },
    )
    .await?;
    source_chat::run_turn(&state, &turn).await?;
    // The model picked here becomes the session's model, as in the Next.js panel.
    if let Some(model) = model_override {
        let request = UpdateSourceChatSessionRequest { title: None, model_override: Some(model) };
        source_chat::update_session(&state, &id, &session_id, &request).await?;
    }
    let panel = chat_panel(&state, &id, Some(&session_id)).await?;
    Ok(fragment_or_redirect(&headers, panel, &session_target(&id, Some(&session_id))))
}

// ---------------------------------------------------------------------------
// /sources/{id}
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Deserialize)]
struct DetailParams {
    #[serde(default)]
    tab: Option<String>,
    #[serde(default)]
    session: Option<String>,
}

async fn detail_page(State(state): State<AppState>, Path(id): Path<String>, Query(params): Query<DetailParams>) -> WebResult<Markup> {
    let id = source_id(&id);
    let view = sources::get_source(&state, &id).await?;
    let tab = match params.tab.as_deref() {
        Some("insights") => "insights",
        Some("details") => "details",
        _ => "content",
    };
    let insight_count = insights::list_source_insights(&state, &id).await?.len();
    let tab_body = match tab {
        "insights" => render_insights(&state, &id, None).await?,
        "details" => details_tab(&state, &view).await?,
        _ => content_tab(&view),
    };
    let chat = chat_panel(&state, &id, params.session.as_deref()).await?;
    let error = view.processing_info.as_ref().and_then(|i| i.get("error")).and_then(|e| e.as_str());
    let href = source_href(&id);
    let tab_href = |name: &str| match params.session.as_deref().filter(|s| !s.is_empty()) {
        Some(s) => format!("{href}?tab={name}&session={s}"),
        None => format!("{href}?tab={name}"),
    };
    let (type_name, _) = type_label(view.asset.as_ref());

    let body = html! {
        div.crumbs { a href="/sources" { "← Sources" } }
        div.src-detail {
            div.stack.src-main {
                div.row-between.src-head {
                    div.stack-sm.grow {
                        (title_block(&id, view.title.as_deref()))
                        div.row.small.faint {
                            span.badge { (type_name) }
                            span { "Source ID: " code { (id) } }
                        }
                        (status_block(&id, view.status.as_deref(), error, false))
                    }
                    div.row.src-head-actions {
                        @match view.file_available {
                            Some(true) => { a.btn.btn-sm href=(format!("/api/sources/{id}/download")) download { "Download File" } }
                            Some(false) => { button.btn.btn-sm type="button" disabled title="This file is currently unavailable due to storage system reasons." { "File unavailable" } }
                            None => {}
                        }
                        @if view.embedded {
                            span.badge.ok { "Embedded" }
                        } @else {
                            form.inline method="post" action=(format!("{href}/embed")) hx-post=(format!("{href}/embed")) hx-swap="outerHTML" {
                                button.btn.btn-sm type="submit" { "Embed Content" }
                            }
                        }
                        form.inline method="post" action=(format!("{href}/delete")) hx-post=(format!("{href}/delete"))
                            hx-confirm="Are you sure you want to delete this source?" {
                            input type="hidden" name="return_to" value="/sources";
                            button.btn.btn-sm.btn-danger type="submit" { "Delete Source" }
                        }
                    }
                }
                nav.tabs aria-label="Source sections" {
                    a.tab href=(tab_href("content")) aria-current=[(tab == "content").then_some("page")] { "Content" }
                    a.tab href=(tab_href("insights")) aria-current=[(tab == "insights").then_some("page")] {
                        "Insights" @if insight_count > 0 { " (" (insight_count) ")" }
                    }
                    a.tab href=(tab_href("details")) aria-current=[(tab == "details").then_some("page")] { "Details" }
                }
                (tab_body)
            }
            div.src-side { (chat) }
        }
    };
    let title = display_title(view.title.as_deref());
    Ok(page(&PageContext::new(&state, href.clone()), &title, body))
}

fn content_tab(view: &SourceView) -> Markup {
    let url = view.asset.as_ref().and_then(|a| a.url.as_deref()).filter(|u| !u.is_empty());
    html! {
        section.card.stack {
            @if let Some(url) = url {
                p.small {
                    @match safe_external(url) {
                        Some(link) => { a href=(link) target="_blank" rel="noopener noreferrer" { (url) } }
                        None => { (url) }
                    }
                }
            }
            @match view.full_text.as_deref().filter(|t| !t.trim().is_empty()) {
                Some(text) => { (markdown(text)) }
                None => { p.muted { "No content available" } }
            }
        }
    }
}

async fn details_tab(state: &AppState, view: &SourceView) -> WebResult<Markup> {
    let all = all_notebooks(state).await?;
    let linked = view.notebooks.clone().unwrap_or_default();
    let href = source_href(&view.id);
    let url = view.asset.as_ref().and_then(|a| a.url.as_deref()).filter(|u| !u.is_empty());
    let file = view.asset.as_ref().and_then(|a| a.file_path.as_deref()).filter(|p| !p.is_empty());
    let topics = view.topics.clone().unwrap_or_default();
    Ok(html! {
        section.card.stack {
            h2 { "Details" }
            @if !view.embedded {
                div.flash.warn.stack-sm {
                    strong { "Content Not Embedded" }
                    p.small { "This content hasn't been embedded for vector search. Embedding enables advanced search capabilities and better content discovery." }
                    form.inline method="post" action=(format!("{href}/embed")) hx-post=(format!("{href}/embed")) hx-swap="outerHTML" {
                        button.btn.btn-sm type="submit" { "Embed Content" }
                    }
                }
            }
            @if let Some(url) = url {
                div.stack-sm {
                    h3 { "URL" }
                    div.row {
                        @match safe_external(url) {
                            Some(link) => { a.truncate href=(link) target="_blank" rel="noopener noreferrer" { (url) } }
                            None => { span.truncate { (url) } }
                        }
                        button.btn.btn-ghost.btn-sm type="button" data-copy=(url) { "Copy" }
                    }
                }
            }
            @if let Some(file) = file {
                div.stack-sm {
                    h3 { "Uploaded file" }
                    div.row {
                        code.truncate { (std::path::Path::new(file).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()) }
                        @if view.file_available == Some(true) {
                            a.btn.btn-sm href=(format!("/api/sources/{}/download", view.id)) download { "Download" }
                        } @else {
                            span.small.faint { "This file is currently unavailable due to storage system reasons." }
                        }
                    }
                }
            }
            @if !topics.is_empty() {
                div.stack-sm {
                    h3 { "Topics" }
                    div.row { @for t in &topics { span.badge { (t) } } }
                }
            }
            div.stack-sm {
                h3 { "Metadata" }
                dl.src-meta {
                    dt { "Source ID" } dd { code { (view.id) } }
                    dt { "Embedding" } dd {
                        @if view.embedded { span.badge.ok { "Embedded" } " " span.faint.small { (view.embedded_chunks) " chunks" } }
                        @else { span.badge { "Not Embedded" } }
                    }
                    dt { "Created" } dd { (date_text(Some(&view.created))) }
                    dt { "Updated" } dd { (date_text(Some(&view.updated))) }
                }
            }
            (notebooks_form(&view.id, &all, &linked))
        }
    })
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{header, Method, Request, StatusCode};
    use http_body_util::BodyExt;

    use crate::ai::AnyModel;
    use crate::domain::Source;
    use crate::testing::{multipart_body, MockEmbeddingModel, MockLanguageModel, TestApp};

    struct Reply {
        status: StatusCode,
        headers: axum::http::HeaderMap,
        body: String,
    }

    async fn send(app: &TestApp, request: Request<Body>) -> Reply {
        let response = app.send(request).await;
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        Reply { status, headers, body: String::from_utf8_lossy(&bytes).into_owned() }
    }

    async fn get(app: &TestApp, path: &str) -> Reply {
        send(app, Request::builder().uri(path).body(Body::empty()).unwrap()).await
    }

    /// A URL-encoded form post; `htmx` adds the htmx request header.
    async fn post_form(app: &TestApp, path: &str, form: &str, htmx: bool) -> Reply {
        let mut builder = Request::builder()
            .method(Method::POST)
            .uri(path)
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded");
        if htmx {
            builder = builder.header("hx-request", "true");
        }
        send(app, builder.body(Body::from(form.to_string())).unwrap()).await
    }

    async fn post_multipart(app: &TestApp, parts: &[(&str, Option<&str>, &[u8])], htmx: bool) -> Reply {
        let (content_type, body) = multipart_body(parts);
        let mut builder = Request::builder().method(Method::POST).uri("/sources").header(header::CONTENT_TYPE, content_type);
        if htmx {
            builder = builder.header("hx-request", "true");
        }
        send(app, builder.body(Body::from(body)).unwrap()).await
    }

    fn enc(value: &str) -> String {
        url::form_urlencoded::byte_serialize(value.as_bytes()).collect()
    }

    async fn notebook(app: &TestApp, name: &str) -> String {
        let (status, body) = app.post("/api/notebooks", serde_json::json!({ "name": name, "description": "" })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body["id"].as_str().unwrap().to_string()
    }

    async fn text_source(app: &TestApp, title: &str, text: &str) -> Source {
        Source::create(&app.state, &Source { title: Some(title.into()), full_text: Some(text.into()), ..Source::default() })
            .await
            .unwrap()
    }

    fn mock(app: &TestApp, replies: &[&str]) -> std::sync::Arc<MockLanguageModel> {
        let model = MockLanguageModel::new(replies.iter().copied());
        app.state.models.set_override(AnyModel::Language(model.clone()));
        app.state.models.set_override(AnyModel::Embedding(MockEmbeddingModel::new()));
        model
    }

    #[tokio::test]
    async fn empty_list_shows_the_empty_state_and_add_dialog() {
        let app = TestApp::new().await;
        let reply = get(&app, "/sources").await;
        assert_eq!(reply.status, StatusCode::OK);
        assert!(reply.body.contains("No sources yet"), "{}", reply.body);
        assert!(reply.body.contains("id=\"src-add\""));
        assert!(reply.body.contains("Enable embedding for search"));
    }

    #[tokio::test]
    async fn list_shows_sources_sorted_and_paged() {
        let app = TestApp::new().await;
        for i in 0..32 {
            text_source(&app, &format!("Doc {i:02}"), "text").await;
        }
        let reply = get(&app, "/sources?sort_by=title&sort_order=asc").await;
        assert_eq!(reply.status, StatusCode::OK);
        assert!(reply.body.contains("Doc 00"));
        assert!(!reply.body.contains("Doc 31"));
        assert!(reply.body.contains("page=2&amp;sort_by=title&amp;sort_order=asc"), "{}", reply.body);
        let second = get(&app, "/sources?page=2&sort_by=title&sort_order=asc").await;
        assert!(second.body.contains("Doc 31") && !second.body.contains("Doc 00"));
        // Unknown sort fields fall back to the default instead of failing.
        assert_eq!(get(&app, "/sources?sort_by=nope").await.status, StatusCode::OK);
    }

    #[tokio::test]
    async fn adding_a_text_source_queues_processing() {
        let app = TestApp::new().await;
        mock(&app, &["unused"]);
        let nb = notebook(&app, "Research").await;
        let reply = post_multipart(
            &app,
            &[
                ("type", None, b"text"),
                ("title", None, b"My text"),
                ("content", None, b"Some body text"),
                ("notebooks", None, nb.as_bytes()),
                ("embed", None, b"true"),
                ("return_to", None, b"/sources"),
            ],
            true,
        )
        .await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        assert_eq!(reply.headers["HX-Redirect"], "/sources");
        let sources: Vec<Source> = app.state.db.query_as("SELECT * FROM source", crate::db::Params::new()).await.unwrap();
        assert_eq!(sources.len(), 1);
        assert_eq!(sources[0].title.as_deref(), Some("My text"));
        assert_eq!(sources[0].notebook_ids(&app.state).await.unwrap(), vec![nb]);

        // A text source needs a title.
        let reply = post_multipart(&app, &[("type", None, b"text"), ("content", None, b"x")], true).await;
        assert_eq!(reply.status, StatusCode::BAD_REQUEST);
        assert!(reply.body.contains("A title is required"));
        // A plain form post is redirected.
        let reply = post_multipart(&app, &[("type", None, b"text"), ("title", None, b"T"), ("content", None, b"x")], false).await;
        assert_eq!(reply.status, StatusCode::SEE_OTHER, "{}", reply.body);
    }

    #[tokio::test]
    async fn adding_an_upload_stores_the_file() {
        let app = TestApp::new().await;
        mock(&app, &["unused"]);
        let reply = post_multipart(&app, &[("type", None, b"upload"), ("file", Some("notes.txt"), b"hello file")], true).await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        let sources: Vec<Source> = app.state.db.query_as("SELECT * FROM source", crate::db::Params::new()).await.unwrap();
        assert_eq!(sources.len(), 1);
        assert!(sources[0].asset.as_ref().and_then(|a| a.file_path.as_deref()).is_some_and(|p| p.ends_with("notes.txt")));
        // No file at all is a clear error.
        let reply = post_multipart(&app, &[("type", None, b"upload")], true).await;
        assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn delete_from_the_list_removes_the_row() {
        let app = TestApp::new().await;
        let source = text_source(&app, "Gone", "x").await;
        let reply = post_form(&app, &format!("/sources/{}/delete", source.id), "", true).await;
        assert_eq!(reply.status, StatusCode::OK);
        assert!(reply.body.is_empty());
        assert!(reply.headers["HX-Trigger"].to_str().unwrap().contains("deleted"));
        assert!(Source::get(&app.state, &source.id).await.is_err());
        // From the detail page it leaves for the list.
        let other = text_source(&app, "Other", "x").await;
        let reply = post_form(&app, &format!("/sources/{}/delete", other.id), "return_to=%2Fsources", true).await;
        assert_eq!(reply.headers["HX-Redirect"], "/sources");
    }

    #[tokio::test]
    async fn detail_page_renders_content_and_tabs() {
        let app = TestApp::new().await;
        let source = text_source(&app, "Deep dive", "# Heading\n\nBody <script>alert(1)</script>").await;
        let reply = get(&app, &format!("/sources/{}", source.id)).await;
        assert_eq!(reply.status, StatusCode::OK);
        assert!(reply.body.contains("Deep dive"));
        assert!(reply.body.contains("<h1>Heading</h1>"));
        assert!(!reply.body.contains("<script>alert"));
        assert!(reply.body.contains("Start a conversation about this source"));
        assert!(reply.body.contains("Embed Content"));
        // The bare id works too.
        let bare = source.id.trim_start_matches("source:");
        assert_eq!(get(&app, &format!("/sources/{bare}")).await.status, StatusCode::OK);

        let details = get(&app, &format!("/sources/{}?tab=details", source.id)).await;
        assert!(details.body.contains("Content Not Embedded") && details.body.contains("No notebooks available"));
        let insights = get(&app, &format!("/sources/{}?tab=insights", source.id)).await;
        assert!(insights.body.contains("No insights yet"));
    }

    #[tokio::test]
    async fn missing_source_is_a_404_page() {
        let app = TestApp::new().await;
        let reply = get(&app, "/sources/source:nope").await;
        assert_eq!(reply.status, StatusCode::NOT_FOUND);
        assert!(reply.body.contains("Source not found"));
        let reply = post_form(&app, "/sources/source:nope/delete", "", true).await;
        assert_eq!(reply.status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn rename_updates_the_title() {
        let app = TestApp::new().await;
        let source = text_source(&app, "Old", "x").await;
        let reply = post_form(&app, &format!("/sources/{}/title", source.id), "title=New+name", true).await;
        assert_eq!(reply.status, StatusCode::OK);
        assert!(reply.body.contains("New name"));
        assert_eq!(Source::get(&app.state, &source.id).await.unwrap().title.as_deref(), Some("New name"));
        let reply = post_form(&app, &format!("/sources/{}/title", source.id), "title=+", true).await;
        assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn notebook_associations_are_saved() {
        let app = TestApp::new().await;
        let a = notebook(&app, "A").await;
        let b = notebook(&app, "B").await;
        let source = text_source(&app, "S", "x").await;
        source.add_to_notebook(&app.state, &a).await.unwrap();
        let form = format!("notebooks={}", enc(&b));
        let reply = post_form(&app, &format!("/sources/{}/notebooks", source.id), &form, true).await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        assert_eq!(source.notebook_ids(&app.state).await.unwrap(), vec![b]);
    }

    #[tokio::test]
    async fn status_polls_while_processing_and_retry_requeues() {
        let app = TestApp::new().await;
        let source = text_source(&app, "S", "text").await;
        // Legacy source: no job, no polling.
        let reply = get(&app, &format!("/sources/{}/status?compact=true", source.id)).await;
        assert_eq!(reply.status, StatusCode::OK);
        assert!(!reply.body.contains("every 3s"));
        // Retry needs a notebook.
        let reply = post_form(&app, &format!("/sources/{}/retry", source.id), "compact=true", true).await;
        assert_eq!(reply.status, StatusCode::BAD_REQUEST);
        assert!(reply.body.contains("not associated with any notebooks"));

        mock(&app, &["unused"]);
        let nb = notebook(&app, "N").await;
        source.add_to_notebook(&app.state, &nb).await.unwrap();
        let reply = post_form(&app, &format!("/sources/{}/retry", source.id), "compact=true", true).await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        let command = Source::get(&app.state, &source.id).await.unwrap().command.unwrap();
        // The card polls while the job is unfinished; on a slow machine it can
        // already be done by the time the reply is built.
        let finished = app.state.jobs.get(&command).await.unwrap().is_some_and(|j| crate::jobs::status::is_terminal(j.status_str()));
        assert!(finished || reply.body.contains("every 3s"), "{}", reply.body);
        app.wait_for_job(&command).await;
        let reply = get(&app, &format!("/sources/{}/status?compact=false&was_active=true", source.id)).await;
        assert!(!reply.body.contains("every 3s"));
        assert_eq!(reply.headers["HX-Refresh"], "true");
    }

    #[tokio::test]
    async fn insights_are_generated_saved_and_deleted() {
        let app = TestApp::new().await;
        mock(&app, &["The key point."]);
        let nb = notebook(&app, "N").await;
        let source = text_source(&app, "S", "Long text about things.").await;
        source.add_to_notebook(&app.state, &nb).await.unwrap();
        let (_, t) = app
            .post("/api/transformations", serde_json::json!({ "name": "summary", "title": "Summary", "description": "d", "prompt": "Summarise." }))
            .await;
        let transformation = t["id"].as_str().unwrap();

        let reply = post_form(&app, &format!("/sources/{}/insights", source.id), &format!("transformation_id={}", enc(transformation)), true).await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        assert!(reply.body.contains("every 3s") && reply.body.contains("pending=command:"), "{}", reply.body);

        // Once the jobs are done, the polled fragment shows the insight.
        app.wait_for_idle().await;
        let poll = reply.body.split("hx-get=\"").nth(1).unwrap().split('"').next().unwrap().replace("&amp;", "&");
        let mut url = poll;
        let mut body = String::new();
        for _ in 0..100 {
            body = get(&app, &url).await.body;
            match body.split("hx-get=\"").nth(1) {
                Some(rest) if body.contains("every 3s") => url = rest.split('"').next().unwrap().replace("&amp;", "&"),
                _ => break,
            }
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
        assert!(body.contains("The key point."), "{body}");
        assert!(!body.contains("every 3s"));

        let insight = crate::services::insights::list_source_insights(&app.state, &source.id).await.unwrap().remove(0);
        let reply = post_form(&app, &format!("/sources/{}/insights/{}/note", source.id, insight.id), &format!("notebook_id={}", enc(&nb)), true).await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        let (_, notes) = app.get(&format!("/api/notes?notebook_id={}", enc(&nb))).await;
        assert_eq!(notes.as_array().unwrap().len(), 1, "{notes}");

        let reply = post_form(&app, &format!("/sources/{}/insights/{}/delete", source.id, insight.id), "", true).await;
        assert_eq!(reply.status, StatusCode::OK);
        assert!(reply.body.contains("No insights yet"));

        // A missing transformation is refused.
        let reply = post_form(&app, &format!("/sources/{}/insights", source.id), "transformation_id=transformation%3Anope", true).await;
        assert_eq!(reply.status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn chat_sessions_and_messages() {
        let app = TestApp::new().await;
        let model = mock(&app, &["Here is the answer."]);
        let source = text_source(&app, "S", "The sky is blue.").await;

        // First message creates a session.
        let reply = post_form(&app, &format!("/sources/{}/chat/send", source.id), "session_id=&message=What+colour%3F", true).await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        assert!(reply.body.contains("What colour?") && reply.body.contains("Here is the answer."), "{}", reply.body);
        assert_eq!(model.call_count(), 1);
        let sessions = crate::services::source_chat::list_sessions(&app.state, &source.id).await.unwrap();
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].message_count, Some(2));
        let session = sessions[0].id.clone();

        let reply = post_form(&app, &format!("/sources/{}/chat/sessions/{session}/rename", source.id), "title=Colours", true).await;
        assert_eq!(reply.status, StatusCode::OK);
        assert!(reply.body.contains("Colours"));

        let reply = post_form(&app, &format!("/sources/{}/chat/sessions", source.id), "", true).await;
        assert_eq!(reply.status, StatusCode::OK);
        assert_eq!(crate::services::source_chat::list_sessions(&app.state, &source.id).await.unwrap().len(), 2);

        let page = get(&app, &format!("/sources/{}?session={session}", source.id)).await;
        assert!(page.body.contains("Here is the answer."));

        let reply = post_form(&app, &format!("/sources/{}/chat/sessions/{session}/delete", source.id), "", true).await;
        assert_eq!(reply.status, StatusCode::OK);
        assert_eq!(crate::services::source_chat::list_sessions(&app.state, &source.id).await.unwrap().len(), 1);

        // An empty message is refused.
        let reply = post_form(&app, &format!("/sources/{}/chat/send", source.id), "message=+", true).await;
        assert_eq!(reply.status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn choosing_default_model_clears_a_saved_override() {
        let app = TestApp::new().await;
        let model = mock(&app, &["Answer."]);
        let source = text_source(&app, "S", "The sky is blue.").await;
        let request = crate::services::source_chat::CreateSourceChatSessionRequest {
            source_id: source.id.clone(),
            title: None,
            model_override: Some("model:old".into()),
        };
        let session = crate::services::source_chat::create_session(&app.state, &source.id, &request).await.unwrap().id;

        let reply =
            post_form(&app, &format!("/sources/{}/chat/send", source.id), &format!("session_id={session}&message=Hi&model_override="), true).await;
        assert_eq!(reply.status, StatusCode::OK, "{}", reply.body);
        assert_eq!(model.call_count(), 1);
        let sessions = crate::services::source_chat::list_sessions(&app.state, &source.id).await.unwrap();
        assert_eq!(sessions[0].model_override, None, "the saved override is cleared");

        // A form without the picker leaves the saved choice alone.
        crate::services::source_chat::set_model_override(&app.state, &source.id, &session, Some("model:kept")).await.unwrap();
        let _ = post_form(&app, &format!("/sources/{}/chat/send", source.id), &format!("session_id={session}&message=Again"), true).await;
        let sessions = crate::services::source_chat::list_sessions(&app.state, &source.id).await.unwrap();
        assert_eq!(sessions[0].model_override.as_deref(), Some("model:kept"));
    }
}
