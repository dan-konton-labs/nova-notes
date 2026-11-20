//! The notebook's studio panel: meetings (recorded in the browser,
//! transcribed in chunks, summarised, reviewed and approved into a note),
//! explainers and infographics (background jobs that end as notes).
//!
//! The notebook page embeds [`studio_panel`]. Every action has its own route
//! under `/notebooks/{id}/studio/...` that answers htmx with a fragment and a
//! plain form post with a redirect back to the notebook. Recording itself is
//! `static/js/studio.js`: microphone (and optionally shared tab audio), 20 s
//! mono 16-bit WAV chunks, uploaded one at a time.

use axum::extract::{Multipart, Path, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Form, Json, Router};
use chrono::{DateTime, Utc};
use maud::{html, Markup};
use serde::Deserialize;
use serde_json::json;

use super::{empty_state, fragment_or_redirect, is_htmx, page, page_head, seg, with_trigger, PageContext, WebResult};
use crate::domain::meeting::{MeetingAction, MAX_SUMMARY, MAX_TITLE, MAX_TRANSCRIPT};
use crate::domain::Notebook;
use crate::error::AppError;
use crate::services::explainers::{self, ExplainerCreateRequest, ExplainerMode};
use crate::services::generation_jobs::{GenerationJobSummary, MAX_INSTRUCTIONS_CHARS};
use crate::services::infographics::{self, Aspect, InfographicCreateRequest, LAYOUTS, STYLES};
use crate::services::meetings::{self, EditMeeting, FinishMeeting, MeetingView, StartMeeting};
use crate::state::AppState;

/// A queued job older than this gets the "waiting for the worker" hint.
const QUEUED_WARNING_SECS: i64 = 60;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/notebooks/{id}/studio", get(panel_route))
        .route("/notebooks/{id}/studio/jobs", get(jobs_route))
        .route("/notebooks/{id}/studio/explainers", post(start_explainer))
        .route("/notebooks/{id}/studio/infographics", post(start_infographic))
        .route("/notebooks/{id}/studio/meetings", get(meetings_route).post(start_meeting))
        .route("/notebooks/{id}/studio/meetings/{meeting}", get(meeting_route).post(save_meeting))
        .route("/notebooks/{id}/studio/meetings/{meeting}/edit", get(edit_route))
        .route("/notebooks/{id}/studio/meetings/{meeting}/chunks/{sequence}", post(upload_chunk))
        .route("/notebooks/{id}/studio/meetings/{meeting}/finish", post(finish_meeting))
        .route("/notebooks/{id}/studio/meetings/{meeting}/summarize", post(retry_summary))
        .route("/notebooks/{id}/studio/meetings/{meeting}/approve", post(approve_meeting))
        .route("/notebooks/{id}/studio/meetings/{meeting}/discard", post(discard_meeting))
}

// ---------------------------------------------------------------------------
// URLs and responses
// ---------------------------------------------------------------------------

fn notebook_url(notebook_id: &str) -> String {
    format!("/notebooks/{}", seg(notebook_id))
}

fn studio_url(notebook_id: &str) -> String {
    format!("{}/studio", notebook_url(notebook_id))
}

fn meeting_url(notebook_id: &str, meeting_id: &str) -> String {
    format!("{}/meetings/{}", studio_url(notebook_id), seg(meeting_id))
}

/// htmx gets the fragment; a plain form post goes back to the notebook.
fn respond(headers: &HeaderMap, notebook_id: &str, fragment: Markup) -> Response {
    fragment_or_redirect(headers, fragment, &notebook_url(notebook_id))
}

/// A fragment GET opened directly (no htmx) renders as a page of its own.
async fn fragment_or_page(
    state: &AppState,
    headers: &HeaderMap,
    notebook_id: &str,
    path: String,
    fragment: Markup,
) -> WebResult<Response> {
    if is_htmx(headers) {
        return Ok(fragment.into_response());
    }
    let notebook = Notebook::get(state, notebook_id).await?;
    let body = html! {
        div.crumbs { a href="/notebooks" { "Notebooks" } " / " a href=(notebook_url(notebook_id)) { (notebook.name) } }
        (page_head("Studio", Some("Meetings, explainers and infographics"), html! {}))
        (fragment)
    };
    Ok(page(&PageContext::new(state, path), "Studio", body).into_response())
}

// ---------------------------------------------------------------------------
// The panel
// ---------------------------------------------------------------------------

/// The notebook's studio panel (meetings, explainers, infographics), embedded
/// by the notebook page. Missing notebook → `NotFound`.
pub async fn studio_panel(state: &AppState, notebook_id: &str) -> super::WebResult<Markup> {
    let notebook = Notebook::get(state, notebook_id).await?;
    let has_sources = !notebook.sources(state, false).await?.is_empty();
    let jobs = load_jobs(state, notebook_id).await?;
    let meetings = meetings::list_meetings(state, notebook_id).await?;
    Ok(html! {
        section.st-panel.panel aria-label="Studio" data-st-panel {
            div.panel-head {
                h3 { "Studio" }
                div.row {
                    button.btn.btn-sm type="button" data-open="#st-explainer-dialog" { "Explainer" }
                    button.btn.btn-sm type="button" data-open="#st-infographic-dialog" { "Infographic" }
                }
            }
            div.panel-body {
                (jobs_list(notebook_id, &jobs, Utc::now()))
                (recorder(notebook_id))
                (meetings_list(notebook_id, &meetings))
            }
            (explainer_dialog(notebook_id, has_sources))
            (infographic_dialog(notebook_id, has_sources))
        }
    })
}

async fn panel_route(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> WebResult<Response> {
    let panel = studio_panel(&state, &id).await?;
    fragment_or_page(&state, &headers, &id, studio_url(&id), panel).await
}

// ---------------------------------------------------------------------------
// Explainer and infographic jobs
// ---------------------------------------------------------------------------

/// Which generator a job belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Explainer,
    Infographic,
}

impl Kind {
    fn as_str(self) -> &'static str {
        match self {
            Kind::Explainer => "explainer",
            Kind::Infographic => "infographic",
        }
    }
    fn failed(self) -> &'static str {
        match self {
            Kind::Explainer => "Could not generate the explainer",
            Kind::Infographic => "Could not generate the infographic",
        }
    }
    fn generating(self) -> &'static str {
        match self {
            Kind::Explainer => "Generating explainer... this can take a few minutes",
            Kind::Infographic => "Drawing infographic... this can take a minute or two",
        }
    }
    fn started(self) -> &'static str {
        match self {
            Kind::Explainer => "Explainer started. It will appear in your notes when it's ready.",
            Kind::Infographic => "Infographic started. It will appear in your notes when it's ready.",
        }
    }
}

fn is_active(job: &GenerationJobSummary) -> bool {
    matches!(job.status.as_str(), "queued" | "running")
}

fn requested(job: &GenerationJobSummary) -> Option<DateTime<Utc>> {
    job.requested_at.as_deref().and_then(|s| DateTime::parse_from_rfc3339(s).ok()).map(|d| d.with_timezone(&Utc))
}

/// Both generators' jobs, newest first.
async fn load_jobs(state: &AppState, notebook_id: &str) -> WebResult<Vec<(Kind, GenerationJobSummary)>> {
    let mut jobs: Vec<(Kind, GenerationJobSummary)> = explainers::list_explainer_jobs(state, notebook_id)
        .await?
        .into_iter()
        .map(|j| (Kind::Explainer, j))
        .chain(infographics::list_infographic_jobs(state, notebook_id).await?.into_iter().map(|j| (Kind::Infographic, j)))
        .collect();
    jobs.sort_by_key(|(_, job)| std::cmp::Reverse(requested(job)));
    Ok(jobs)
}

/// Running and failed jobs (completed ones are notes by now). Polls while a
/// job is active. Completed jobs stay as hidden markers so the script can
/// tell when one finishes.
fn jobs_list(notebook_id: &str, jobs: &[(Kind, GenerationJobSummary)], now: DateTime<Utc>) -> Markup {
    let active = jobs.iter().any(|(_, j)| is_active(j));
    html! {
        div #st-jobs.st-jobs
            hx-get=(format!("{}/jobs", studio_url(notebook_id)))
            hx-trigger=[active.then_some("every 4s")]
            hx-swap="outerHTML"
            data-st-jobs
        {
            @for (kind, job) in jobs {
                @if is_active(job) {
                    @let stuck = job.status == "queued"
                        && requested(job).is_some_and(|at| (now - at).num_seconds() > QUEUED_WARNING_SECS);
                    div.st-job.st-job-pending role="status"
                        data-job-id=(job.job_id) data-job-status=(job.status) data-job-kind=(kind.as_str())
                    {
                        span.spinner aria-hidden="true" {}
                        span {
                            @if stuck {
                                "Waiting for the background worker. If this doesn't start soon, the worker may not be running."
                            } @else {
                                (kind.generating())
                            }
                        }
                    }
                } @else if job.status == "failed" {
                    div.st-job.st-job-failed role="alert"
                        data-job-id=(job.job_id) data-job-status="failed" data-job-kind=(kind.as_str())
                        data-job-error=(job.error_message.as_deref().unwrap_or(""))
                    {
                        div.grow {
                            strong { (kind.failed()) }
                            p.small.muted.st-wrap { (job.error_message.as_deref().filter(|m| !m.is_empty()).unwrap_or("Something went wrong. Please try again.")) }
                        }
                        button.btn.btn-ghost.btn-sm type="button" data-st-dismiss aria-label="Dismiss" title="Dismiss" { "Dismiss" }
                    }
                } @else {
                    span hidden data-job-id=(job.job_id) data-job-status=(job.status) data-job-kind=(kind.as_str())
                        data-job-title=(job.title.as_deref().unwrap_or("")) {}
                }
            }
        }
    }
}

async fn jobs_route(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> WebResult<Response> {
    let jobs = load_jobs(&state, &id).await?;
    fragment_or_page(&state, &headers, &id, format!("{}/jobs", studio_url(&id)), jobs_list(&id, &jobs, Utc::now())).await
}

/// The form's optional focus, trimmed, at most 2000 characters.
fn instructions(value: Option<String>) -> WebResult<Option<String>> {
    let value = value.map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    if value.as_ref().is_some_and(|v| v.chars().count() > MAX_INSTRUCTIONS_CHARS) {
        return Err(AppError::unprocessable(format!(
            "instructions: String should have at most {MAX_INSTRUCTIONS_CHARS} characters"
        ))
        .into());
    }
    Ok(value)
}

#[derive(Debug, Default, Deserialize)]
struct ExplainerForm {
    instructions: Option<String>,
}

async fn start_explainer(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Form(form): Form<ExplainerForm>,
) -> WebResult<Response> {
    let request = ExplainerCreateRequest { mode: ExplainerMode::Sources, instructions: instructions(form.instructions)? };
    explainers::create_explainer(&state, &id, &request).await?;
    let jobs = load_jobs(&state, &id).await?;
    let response = respond(&headers, &id, jobs_list(&id, &jobs, Utc::now()));
    Ok(with_trigger(response, json!({ "toast": Kind::Explainer.started() })))
}

#[derive(Debug, Default, Deserialize)]
struct InfographicForm {
    style: Option<String>,
    layout: Option<String>,
    aspect: Option<String>,
    instructions: Option<String>,
}

async fn start_infographic(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Form(form): Form<InfographicForm>,
) -> WebResult<Response> {
    let defaults = InfographicCreateRequest::default();
    let aspect = match form.aspect.as_deref().filter(|a| !a.is_empty()) {
        None => defaults.aspect,
        Some(value) => Aspect::parse(value).ok_or_else(|| AppError::invalid(format!("Unknown infographic aspect: {value}")))?,
    };
    let request = InfographicCreateRequest {
        style: form.style.filter(|s| !s.is_empty()).unwrap_or(defaults.style),
        layout: form.layout.filter(|s| !s.is_empty()).unwrap_or(defaults.layout),
        aspect,
        instructions: instructions(form.instructions)?,
    };
    infographics::create_infographic(&state, &id, &request).await?;
    let jobs = load_jobs(&state, &id).await?;
    let response = respond(&headers, &id, jobs_list(&id, &jobs, Utc::now()));
    Ok(with_trigger(response, json!({ "toast": Kind::Infographic.started() })))
}

fn no_sources_alert(has_sources: bool) -> Markup {
    html! {
        @if !has_sources { p.flash.bad role="alert" { "Add a source to this notebook first." } }
    }
}

fn explainer_dialog(notebook_id: &str, has_sources: bool) -> Markup {
    let action = format!("{}/explainers", studio_url(notebook_id));
    html! {
        dialog #st-explainer-dialog aria-labelledby="st-explainer-title" {
            form method="post" action=(action) hx-post=(action) hx-target="#st-jobs" hx-swap="outerHTML"
                data-close-on-success data-reset
            {
                div.dialog-head {
                    h3 #st-explainer-title { "Generate Explainer" }
                    button.btn.btn-ghost.btn-sm type="button" data-close aria-label="Close" { "Close" }
                }
                div.dialog-body {
                    p.muted { "Build an interactive HTML explainer you can explore, saved as a note in this notebook." }
                    div.st-option.st-option-selected {
                        strong { "From saved sources" }
                        span.small.muted { "Turn this notebook's sources into an interactive dossier." }
                    }
                    (no_sources_alert(has_sources))
                    div.field {
                        label for="st-explainer-focus" { "Focus (optional)" }
                        textarea #st-explainer-focus name="instructions" rows="3" maxlength=(MAX_INSTRUCTIONS_CHARS)
                            placeholder="e.g. Explain it for a beginner, focus on costs and risks" {}
                        p.hint { "Generating takes a few minutes. The explainer appears in your notes when it's ready." }
                    }
                }
                div.dialog-foot {
                    button.btn type="button" data-close { "Cancel" }
                    button.btn.btn-primary type="submit" disabled[!has_sources] { "Generate" }
                }
            }
        }
    }
}

/// Short descriptions and a palette preview (paper, ink, accent) per style,
/// as the Next.js dialog shows them. The preview is the style's own colours,
/// not app chrome, so it is not themed.
fn style_preview(id: &str) -> (&'static str, [&'static str; 3]) {
    match id {
        "clean-editorial" => ("Magazine grid, lots of white space, one accent colour", ["#FAFAF7", "#1A1A1A", "#0F766E"]),
        "craft-handmade" => ("Paper cut-outs, warm pastels, cartoon icons", ["#FFF8F0", "#F4A261", "#2A9D8F"]),
        "chalkboard" => ("Coloured chalk sketches on a dark board", ["#1C2B1C", "#F5F5F5", "#FFE566"]),
        "technical-schematic" => ("White line work on engineering blue", ["#1E3A5F", "#E0F2FE", "#F59E0B"]),
        "subway-map" => ("Coloured lines and stations, like a metro map", ["#F5F5F5", "#E63946", "#1D4ED8"]),
        "corporate-memphis" => ("Bright flat shapes and friendly figures", ["#FFFFFF", "#6C5CE7", "#FF7675"]),
        "cyberpunk-neon" => ("Glowing neon lines on a dark cyberpunk grid", ["#0B0B1A", "#FF2BD6", "#00F0FF"]),
        "aged-academia" => ("Sepia science plate with fine ink lines", ["#F2E6CF", "#3E2C1C", "#9B2226"]),
        "kawaii" => ("Cute pastels with rounded shapes", ["#FFF7FB", "#FFC8DD", "#CDB4DB"]),
        "pixel-art" => ("Retro 8-bit game look", ["#1B1B3A", "#FFD60A", "#3DDC97"]),
        "ikea-manual" => ("Black line art with numbered steps", ["#FFFFFF", "#111111", "#0058A3"]),
        "bold-graphic" => ("Bold outlines, halftone dots, speech bubbles", ["#FFD60A", "#111111", "#E63946"]),
        "morandi-journal" => ("Bullet-journal doodles in muted tones", ["#F7F3EE", "#C9A9A6", "#A3B18A"]),
        "ui-wireframe" => ("Greyscale interface mock-up", ["#F1F1F1", "#9CA3AF", "#2563EB"]),
        _ => ("", ["#FFFFFF", "#111111", "#888888"]),
    }
}

fn aspect_label(aspect: Aspect) -> &'static str {
    match aspect {
        Aspect::Portrait => "Portrait",
        Aspect::Landscape => "Landscape",
        Aspect::Square => "Square",
    }
}

fn infographic_dialog(notebook_id: &str, has_sources: bool) -> Markup {
    let action = format!("{}/infographics", studio_url(notebook_id));
    html! {
        dialog.wide #st-infographic-dialog aria-labelledby="st-infographic-title" {
            form method="post" action=(action) hx-post=(action) hx-target="#st-jobs" hx-swap="outerHTML"
                data-close-on-success data-reset
            {
                div.dialog-head {
                    h3 #st-infographic-title { "Generate Infographic" }
                    button.btn.btn-ghost.btn-sm type="button" data-close aria-label="Close" { "Close" }
                }
                div.dialog-body {
                    p.muted { "Turn this notebook's sources into a one-page visual summary, saved as an image note you can download." }
                    fieldset.st-fieldset {
                        legend { "Style" }
                        div.st-styles {
                            @for option in STYLES {
                                @let (desc, [paper, ink, accent]) = style_preview(option.id);
                                label.st-style {
                                    input type="radio" name="style" value=(option.id) checked[option.id == infographics::DEFAULT_STYLE];
                                    span.st-swatch aria-hidden="true" style=(format!("background:{paper}")) {
                                        i style=(format!("background:{ink};height:12px")) {}
                                        i style=(format!("background:{accent};height:20px")) {}
                                        i style=(format!("background:{ink};height:8px")) {}
                                    }
                                    span.grow {
                                        span.st-style-name { (option.name) }
                                        span.st-style-desc { (desc) }
                                    }
                                }
                            }
                        }
                    }
                    div.cols-2 {
                        div.field {
                            label for="st-infographic-layout" { "Layout" }
                            select #st-infographic-layout name="layout" {
                                @for option in LAYOUTS {
                                    option value=(option.id) selected[option.id == infographics::DEFAULT_LAYOUT] { (option.name) }
                                }
                            }
                        }
                        fieldset.st-fieldset {
                            legend { "Shape" }
                            div.st-aspects {
                                @for aspect in Aspect::ALL {
                                    label.st-aspect {
                                        input type="radio" name="aspect" value=(aspect.as_str()) checked[aspect == Aspect::default()];
                                        span { (aspect_label(aspect)) }
                                    }
                                }
                            }
                        }
                    }
                    (no_sources_alert(has_sources))
                    div.field {
                        label for="st-infographic-focus" { "Focus (optional)" }
                        textarea #st-infographic-focus name="instructions" rows="3" maxlength=(MAX_INSTRUCTIONS_CHARS)
                            placeholder="e.g. Focus on the key numbers, keep it simple for beginners" {}
                        p.hint { "Generating takes a minute or two. The infographic appears in your notes when it's ready." }
                    }
                }
                div.dialog-foot {
                    button.btn type="button" data-close { "Cancel" }
                    button.btn.btn-primary type="submit" disabled[!has_sources] { "Generate" }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Meetings
// ---------------------------------------------------------------------------

/// The in-browser recorder's controls; `studio.js` drives them.
fn recorder(notebook_id: &str) -> Markup {
    html! {
        div.st-recorder data-st-recorder data-start-url=(format!("{}/meetings", studio_url(notebook_id))) {
            div.row-between {
                h4 { "Meetings" }
            }
            p.small.muted { "Audio is sent to your configured transcription provider, used temporarily, then deleted locally. Only text is kept. Tell participants before recording." }
            p.small.faint { "Microphone requires HTTPS or localhost. Shared audio varies by browser and OS; Chrome or Edge with tab audio is recommended. Four-hour limit." }
            input type="text" name="title" aria-label="Meeting title" placeholder="Meeting title" maxlength=(MAX_TITLE) data-st-title;
            label.check {
                input type="checkbox" data-st-shared;
                "Include shared-tab or system audio"
            }
            p.hint data-st-mic-help { "Microphone only: suitable for in-person meetings." }
            p.hint data-st-shared-help hidden { "Choose the meeting tab and enable Share audio. Use headphones to reduce echo." }
            div.row {
                button.btn.btn-primary.btn-sm type="button" data-st-record { "Record meeting" }
                span.row data-st-status role="status" hidden {
                    span.st-rec-dot aria-hidden="true" {}
                    span data-st-status-text { "Recording" }
                    span.st-timer data-st-timer { "0:00" }
                }
                button.btn.btn-sm type="button" data-st-stop hidden { "Stop" }
                span.spinner data-st-busy hidden aria-hidden="true" {}
            }
            p.flash.bad role="alert" data-st-error hidden {}
            noscript { p.flash.warn { "Recording a meeting needs JavaScript." } }
        }
    }
}

fn meetings_list(notebook_id: &str, meetings: &[MeetingView]) -> Markup {
    html! {
        div #st-meetings.st-meetings.stack-sm data-st-meetings {
            @if meetings.is_empty() {
                div.st-meetings-empty { (empty_state("No meetings yet", "Record a meeting to get a transcript, a summary and action items.")) }
            }
            @for meeting in meetings {
                (meeting_card(notebook_id, meeting))
            }
        }
    }
}

async fn meetings_route(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> WebResult<Response> {
    let list = meetings::list_meetings(&state, &id).await?;
    fragment_or_page(&state, &headers, &id, format!("{}/meetings", studio_url(&id)), meetings_list(&id, &list)).await
}

fn status_label(status: &str) -> (&'static str, &'static str) {
    match status {
        "recording" => ("info", "Recording"),
        "review" => ("warn", "To review"),
        "approved" => ("ok", "Approved"),
        _ => ("", "Draft"),
    }
}

fn summary_pending(view: &MeetingView) -> bool {
    matches!(view.summary_status.as_str(), "new" | "queued" | "running")
}

/// A small form whose button posts to `url` and swaps the card.
fn card_action(url: &str, label: &str, class: &str, confirm: Option<&str>, extra: Markup) -> Markup {
    html! {
        form.inline method="post" action=(url) hx-post=(url) hx-target="closest .st-meeting" hx-swap="outerHTML"
            hx-confirm=[confirm]
        {
            (extra)
            button type="submit" class=(format!("btn btn-sm {class}")) { (label) }
        }
    }
}

/// One meeting, read-only. Polls itself while its summary is being generated.
fn meeting_card(notebook_id: &str, view: &MeetingView) -> Markup {
    let meeting = &view.meeting;
    let url = meeting_url(notebook_id, &meeting.id);
    let pending = summary_pending(view);
    let (badge, label) = status_label(&meeting.status);
    let recording = meeting.status == "recording";
    html! {
        article.st-meeting.item
            data-meeting-id=(meeting.id) data-card-url=(url) data-status=(meeting.status)
            hx-get=[pending.then_some(url.as_str())] hx-trigger=[pending.then_some("every 3s")] hx-swap="outerHTML"
        {
            div.st-meeting-body {
                div.row-between {
                    h4.st-meeting-title { (meeting.title) }
                    span class=(format!("badge {badge}")) { (label) }
                }
                @if let Some(created) = meeting.created {
                    p.small.faint { (super::human_date(Some(created))) }
                }
                @if meeting.interrupted {
                    p.flash.warn.small role="alert" { "The recording was interrupted. A final section may be missing." }
                }
                @if !meeting.missing_chunks.is_empty() {
                    p.flash.warn.small role="alert" {
                        (meeting.missing_chunks.len()) " audio sections are missing. Review the gaps before approving."
                    }
                }
                @if recording {
                    div.stack-sm data-st-inactive-only {
                        p.flash.warn.small { "No recording is active in this window. If it was interrupted, recover completed sections below. A final section may be missing." }
                        div.row {
                            (card_action(&format!("{url}/finish"), "Finish interrupted recording", "", None, html! {
                                input type="hidden" name="total_chunks" value=(meeting.total_chunks);
                                input type="hidden" name="interrupted" value="true";
                            }))
                        }
                    }
                }
                @if pending {
                    p.small.st-pending role="status" {
                        span.spinner aria-hidden="true" {}
                        " Generating summary and actions. If this stays queued, check the background worker."
                    }
                }
                @if !meeting.summary_error.is_empty() {
                    p.flash.bad.small role="alert" {
                        "Summary failed. Check the default chat model and background worker, then retry. The transcript is retained."
                    }
                }
                details.st-section open[!recording] {
                    summary { "Summary" }
                    @if meeting.summary.is_empty() {
                        p.small.muted { "No summary yet." }
                    } @else {
                        p.pre-wrap.small { (meeting.summary) }
                    }
                }
                details.st-section {
                    summary { "Actions (" (meeting.actions.len()) ")" }
                    @if meeting.actions.is_empty() {
                        p.small.muted { "No actions captured." }
                    }
                    ul.st-actions {
                        @for action in &meeting.actions {
                            li.st-action {
                                label.check {
                                    input type="checkbox" checked[action.done] disabled;
                                    span { (action.text) }
                                }
                                @let meta = [action.owner.as_str(), action.due.as_str()].into_iter().filter(|s| !s.is_empty()).collect::<Vec<_>>().join(" · ");
                                @if !meta.is_empty() { p.small.muted { (meta) } }
                                blockquote.st-evidence { (action.evidence) }
                            }
                        }
                    }
                }
                details.st-section open[recording] {
                    summary { "Transcript" }
                    @if meeting.transcript.is_empty() {
                        p.small.muted { "Waiting for transcribed audio…" }
                    } @else {
                        p.pre-wrap.small.st-transcript { (meeting.transcript) }
                    }
                }
                div.row {
                    @if meeting.status == "approved" {
                        span.small.muted { "Saved as a note in this notebook." }
                    }
                    @if !recording {
                        a.btn.btn-sm href=(format!("{url}/edit")) hx-get=(format!("{url}/edit"))
                            hx-target="closest .st-meeting" hx-swap="outerHTML" { "Edit" }
                    }
                    @if meeting.status == "review" {
                        (card_action(&format!("{url}/approve"), "Approve for notebook chat", "btn-primary", None, html! {}))
                        @if !pending {
                            (card_action(&format!("{url}/summarize"), "Generate summary again", "", None, html! {}))
                        }
                    }
                    @if meeting.status != "approved" {
                        span data-st-inactive-only {
                            (card_action(&format!("{url}/discard"), "Discard", "btn-danger",
                                Some("Permanently discard this meeting draft and its transcript?"), html! {}))
                        }
                    }
                }
            }
        }
    }
}

async fn meeting_route(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, meeting)): Path<(String, String)>,
) -> WebResult<Response> {
    let view = meetings::get_meeting_view(&state, &id, &meeting).await?;
    fragment_or_page(&state, &headers, &id, meeting_url(&id, &meeting), meeting_card(&id, &view)).await
}

#[derive(Debug, Deserialize)]
struct StartForm {
    #[serde(default)]
    title: String,
    #[serde(default)]
    capture_mode: Option<String>,
}

/// Start a recording draft; answers with its card (the script reads the
/// meeting id from it).
async fn start_meeting(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
    Form(form): Form<StartForm>,
) -> WebResult<Response> {
    let title = form.title.trim();
    let data = StartMeeting {
        title: if title.is_empty() { "Meeting".into() } else { title.to_string() },
        capture_mode: form.capture_mode.filter(|m| !m.is_empty()).unwrap_or_else(|| "microphone".into()),
    };
    let view = meetings::start_meeting(&state, &id, &data).await?;
    Ok(respond(&headers, &id, meeting_card(&id, &view)))
}

/// One WAV chunk (multipart field `audio`); answers `{"text": ...}`.
async fn upload_chunk(
    State(state): State<AppState>,
    Path((id, meeting, sequence)): Path<(String, String, String)>,
    mut multipart: Multipart,
) -> WebResult<Response> {
    let sequence: i64 =
        sequence.trim().parse().map_err(|_| AppError::invalid("The chunk number must be a whole number"))?;
    let mut audio = None;
    while let Some(field) = multipart.next_field().await.map_err(|e| AppError::invalid(e.body_text()))? {
        if field.name() == Some("audio") {
            audio = Some(field.bytes().await.map_err(|e| AppError::invalid(e.body_text()))?);
        }
    }
    let audio = audio.ok_or_else(|| AppError::invalid("No audio was uploaded"))?;
    let result = meetings::upload_chunk(&state, &id, &meeting, sequence, &audio).await?;
    Ok(Json(result).into_response())
}

#[derive(Debug, Deserialize)]
struct FinishForm {
    #[serde(default)]
    total_chunks: String,
    #[serde(default)]
    interrupted: Option<String>,
}

async fn finish_meeting(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, meeting)): Path<(String, String)>,
    Form(form): Form<FinishForm>,
) -> WebResult<Response> {
    let total_chunks: i64 =
        form.total_chunks.trim().parse().map_err(|_| AppError::invalid("total_chunks must be a whole number"))?;
    let interrupted = form.interrupted.as_deref().is_some_and(|v| matches!(v, "true" | "1" | "on" | "yes"));
    let data = FinishMeeting { total_chunks, interrupted };
    data.check().map_err(AppError::from)?;
    let view = meetings::finish_meeting(&state, &id, &meeting, &data).await?;
    Ok(respond(&headers, &id, meeting_card(&id, &view)))
}

async fn retry_summary(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, meeting)): Path<(String, String)>,
) -> WebResult<Response> {
    let view = meetings::retry_summary(&state, &id, &meeting).await?;
    Ok(respond(&headers, &id, meeting_card(&id, &view)))
}

async fn approve_meeting(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, meeting)): Path<(String, String)>,
) -> WebResult<Response> {
    meetings::approve(&state, &id, &meeting).await?;
    let view = meetings::get_meeting_view(&state, &id, &meeting).await?;
    let response = respond(&headers, &id, meeting_card(&id, &view));
    Ok(with_trigger(response, json!({ "toast": "Meeting saved as a note", "notes-changed": true })))
}

async fn discard_meeting(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, meeting)): Path<(String, String)>,
) -> WebResult<Response> {
    meetings::discard_meeting(&state, &id, &meeting).await?;
    let response = respond(&headers, &id, html! {});
    Ok(with_trigger(response, json!({ "toast": "Meeting discarded" })))
}

// --- Editing ---------------------------------------------------------------

fn edit_form(notebook_id: &str, view: &MeetingView) -> Markup {
    let meeting = &view.meeting;
    let url = meeting_url(notebook_id, &meeting.id);
    html! {
        article.st-meeting.item data-meeting-id=(meeting.id) data-card-url=(url) data-status=(meeting.status) {
            form.st-meeting-body.stack-sm method="post" action=(url) hx-post=(url)
                hx-target="closest .st-meeting" hx-swap="outerHTML"
            {
                input type="hidden" name="revision" value=(meeting.revision);
                div.field {
                    label { "Meeting title" }
                    input type="text" name="title" value=(meeting.title) maxlength=(MAX_TITLE) required;
                }
                div.field {
                    label { "Summary" }
                    textarea name="summary" rows="8" maxlength=(MAX_SUMMARY) { (meeting.summary) }
                }
                fieldset.st-fieldset {
                    legend { "Actions" }
                    @if meeting.actions.is_empty() {
                        p.small.muted { "No actions captured." }
                    }
                    @for (i, action) in meeting.actions.iter().enumerate() {
                        div.st-action.stack-sm {
                            div.row {
                                input type="checkbox" name=(format!("a{i}_done")) value="true" checked[action.done] aria-label="Done";
                                input.grow type="text" name=(format!("a{i}_text")) value=(action.text) maxlength="2000"
                                    aria-label="Action" required;
                            }
                            div.cols-2 {
                                input type="text" name=(format!("a{i}_owner")) value=(action.owner) maxlength="200"
                                    placeholder="Owner (if stated)" aria-label="Owner (if stated)";
                                input type="text" name=(format!("a{i}_due")) value=(action.due) maxlength="200"
                                    placeholder="Due date (if stated)" aria-label="Due date (if stated)";
                            }
                            input type="hidden" name=(format!("a{i}_evidence")) value=(action.evidence);
                            blockquote.st-evidence { (action.evidence) }
                            label.check.small {
                                input type="checkbox" name=(format!("a{i}_remove")) value="true";
                                "Remove"
                            }
                        }
                    }
                }
                div.field {
                    label { "Transcript" }
                    textarea name="transcript" rows="12" maxlength=(MAX_TRANSCRIPT) required { (meeting.transcript) }
                }
                div.row {
                    button.btn.btn-primary.btn-sm type="submit" { "Save" }
                    a.btn.btn-ghost.btn-sm href=(notebook_url(notebook_id)) hx-get=(url)
                        hx-target="closest .st-meeting" hx-swap="outerHTML" { "Cancel" }
                }
            }
        }
    }
}

async fn edit_route(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, meeting)): Path<(String, String)>,
) -> WebResult<Response> {
    let view = meetings::get_meeting_view(&state, &id, &meeting).await?;
    if view.meeting.status == "recording" {
        return Err(AppError::invalid("Finish recording before editing").into());
    }
    fragment_or_page(&state, &headers, &id, format!("{}/edit", meeting_url(&id, &meeting)), edit_form(&id, &view)).await
}

/// The edit form's fields: `revision`, `title`, `summary`, `transcript` and
/// `a{i}_{text,owner,due,evidence,done,remove}` per action.
fn parse_edit(fields: &[(String, String)]) -> Result<EditMeeting, AppError> {
    let mut revision = None;
    let (mut title, mut summary, mut transcript) = (String::new(), String::new(), String::new());
    let mut actions: std::collections::BTreeMap<usize, (MeetingAction, bool)> = Default::default();
    for (name, value) in fields {
        match name.as_str() {
            "revision" => {
                revision = Some(value.trim().parse::<i64>().map_err(|_| AppError::invalid("Invalid revision"))?)
            }
            "title" => title = value.trim().to_string(),
            "summary" => summary = value.trim().to_string(),
            "transcript" => transcript = value.clone(),
            other => {
                let Some((index, field)) = other.strip_prefix('a').and_then(|rest| rest.split_once('_')) else {
                    continue;
                };
                let Ok(index) = index.parse::<usize>() else { continue };
                let (action, removed) = actions.entry(index).or_default();
                match field {
                    "text" => action.text = value.trim().to_string(),
                    "owner" => action.owner = value.trim().to_string(),
                    "due" => action.due = value.trim().to_string(),
                    "evidence" => action.evidence = value.clone(),
                    "done" => action.done = true,
                    "remove" => *removed = true,
                    _ => {}
                }
            }
        }
    }
    Ok(EditMeeting {
        revision: revision.ok_or_else(|| AppError::invalid("Missing revision; reload the meeting"))?,
        title,
        summary,
        actions: actions.into_values().filter(|(_, removed)| !removed).map(|(a, _)| a).collect(),
        transcript,
    })
}

async fn save_meeting(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((id, meeting)): Path<(String, String)>,
    Form(fields): Form<Vec<(String, String)>>,
) -> WebResult<Response> {
    let data = parse_edit(&fields)?;
    data.check().map_err(AppError::from)?;
    let view = meetings::edit_meeting(&state, &id, &meeting, &data).await?;
    let response = respond(&headers, &id, meeting_card(&id, &view));
    let trigger = if view.meeting.status == "approved" {
        json!({ "toast": "Meeting saved", "notes-changed": true })
    } else {
        json!({ "toast": "Meeting saved" })
    };
    Ok(with_trigger(response, trigger))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{header, Method, Request, StatusCode};
    use http_body_util::BodyExt;

    use crate::ai::AnyModel;
    use crate::db::Params;
    use crate::domain::meeting::Meeting;
    use crate::testing::{multipart_body, MockEmbeddingModel, MockLanguageModel, MockSpeechToTextModel, TestApp};

    const QUOTE: &str = "Priya will send the budget to finance by Friday.";

    fn wav(frames: u32) -> Vec<u8> {
        let data_len = frames * 2;
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data_len).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&16_000u32.to_le_bytes());
        out.extend_from_slice(&32_000u32.to_le_bytes());
        out.extend_from_slice(&2u16.to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data_len.to_le_bytes());
        out.extend(std::iter::repeat_n(0u8, data_len as usize));
        out
    }

    fn encode(value: &str) -> String {
        url::form_urlencoded::byte_serialize(value.as_bytes()).collect()
    }

    /// POST a urlencoded form, as htmx (`htmx`) or a plain browser.
    async fn form(app: &TestApp, path: &str, fields: &[(&str, &str)], htmx: bool) -> (StatusCode, HeaderMap, String) {
        let body = fields.iter().map(|(k, v)| format!("{}={}", encode(k), encode(v))).collect::<Vec<_>>().join("&");
        let mut builder = Request::builder()
            .method(Method::POST)
            .uri(path)
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded");
        if htmx {
            builder = builder.header("HX-Request", "true");
        }
        let response = app.send(builder.body(Body::from(body)).unwrap()).await;
        let status = response.status();
        let headers = response.headers().clone();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, headers, String::from_utf8_lossy(&bytes).into_owned())
    }

    async fn htmx_get(app: &TestApp, path: &str) -> (StatusCode, String) {
        let request = Request::builder().uri(path).header("HX-Request", "true").body(Body::empty()).unwrap();
        let response = app.send(request).await;
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    async fn notebook(app: &TestApp) -> String {
        let (status, body) = app.post("/api/notebooks", json!({ "name": "Studio test" })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body["id"].as_str().unwrap().to_string()
    }

    async fn add_source(app: &TestApp, notebook_id: &str) {
        let (status, body) = app
            .post(
                "/api/sources/json",
                json!({ "type": "text", "notebook_id": notebook_id, "content": "Budgets rose by ten percent.", "embed": false }),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
    }

    fn with_stt(app: &TestApp) {
        app.state.models.set_override(AnyModel::SpeechToText(MockSpeechToTextModel::new(QUOTE)));
        app.state.models.set_override(AnyModel::Embedding(MockEmbeddingModel::new()));
    }

    fn summary_reply() -> String {
        json!({
            "summary": "Budget follow-up agreed.",
            "actions": [{ "text": "Send budget", "owner": "Priya", "due": "Friday", "evidence": QUOTE }],
        })
        .to_string()
    }

    fn meeting_id_in(html: &str) -> String {
        let start = html.find("data-meeting-id=\"").expect("card has a meeting id") + "data-meeting-id=\"".len();
        html[start..start + html[start..].find('"').unwrap()].to_string()
    }

    async fn upload(app: &TestApp, nb: &str, meeting: &str, sequence: i64, audio: &[u8]) -> (StatusCode, String) {
        let (content_type, body) = multipart_body(&[("audio", Some("chunk.wav"), audio)]);
        let request = Request::builder()
            .method(Method::POST)
            .uri(format!("{}/chunks/{sequence}", meeting_url(nb, meeting)))
            .header(header::CONTENT_TYPE, content_type)
            .header("HX-Request", "true")
            .body(Body::from(body))
            .unwrap();
        let response = app.send(request).await;
        let status = response.status();
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    /// Wait until the meeting's summary job has finished.
    async fn wait_summary(app: &TestApp, meeting: &str) {
        let stored = Meeting::get(&app.state, meeting).await.unwrap();
        assert!(!stored.summary_job.is_empty());
        app.wait_for_job(&stored.summary_job).await;
    }

    #[test]
    fn path_segments_are_encoded() {
        assert_eq!(seg("notebook:abc"), "notebook:abc");
        assert_eq!(seg("a b/c"), "a%20b%2Fc");
        assert_eq!(meeting_url("notebook:x", "meeting:y"), "/notebooks/notebook:x/studio/meetings/meeting:y");
    }

    #[test]
    fn edit_fields_parse_into_an_edit() {
        let fields = [
            ("revision", "2"),
            ("title", " Sync "),
            ("summary", "S"),
            ("transcript", "T"),
            ("a0_text", "One"),
            ("a0_evidence", "q1"),
            ("a0_done", "true"),
            ("a1_text", "Two"),
            ("a1_evidence", "q2"),
            ("a1_remove", "true"),
            ("a2_text", "Three"),
            ("a2_owner", "Sam"),
            ("a2_evidence", "q3"),
        ]
        .map(|(k, v)| (k.to_string(), v.to_string()));
        let edit = parse_edit(&fields).unwrap();
        assert_eq!(edit.revision, 2);
        assert_eq!(edit.title, "Sync");
        assert_eq!(edit.actions.len(), 2);
        assert!(edit.actions[0].done);
        assert_eq!(edit.actions[1].owner, "Sam");
        assert!(!edit.actions[1].done);
        assert!(parse_edit(&[("title".to_string(), "x".to_string())]).is_err());
    }

    #[tokio::test]
    async fn panel_renders_empty_states_and_options() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        let panel = studio_panel(&app.state, &nb).await.unwrap().into_string();
        assert!(panel.contains("data-st-panel"));
        assert!(panel.contains("No meetings yet"));
        assert!(panel.contains("Record meeting"));
        assert!(panel.contains("Generate Explainer") && panel.contains("Generate Infographic"));
        assert!(panel.contains("Add a source to this notebook first."));
        for style in STYLES {
            assert!(panel.contains(&format!("value=\"{}\"", style.id)), "{}", style.id);
        }
        for layout in LAYOUTS {
            assert!(panel.contains(&format!("value=\"{}\"", layout.id)), "{}", layout.id);
        }
        assert!(panel.contains("value=\"landscape\""));
        // Nothing is running, so the job list does not poll.
        assert!(!panel.contains("every 4s"));

        // The panel route: a fragment for htmx, a full page otherwise.
        let (status, body) = htmx_get(&app, &studio_url(&nb)).await;
        assert_eq!(status, StatusCode::OK);
        assert!(!body.contains("<html"));
        let (status, body) = app.text(Method::GET, &studio_url(&nb), None).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("<html") && body.contains("Studio test"), "{body}");
    }

    #[tokio::test]
    async fn missing_notebook_and_meeting_are_404() {
        let app = TestApp::new().await;
        assert!(matches!(studio_panel(&app.state, "notebook:nope").await, Err(super::super::WebError(AppError::NotFound(_)))));
        let (status, body) = app.text(Method::GET, "/notebooks/notebook:nope/studio", None).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body.contains("data-error-detail"));
        let nb = notebook(&app).await;
        let (status, _) = htmx_get(&app, &meeting_url(&nb, "meeting:nope")).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _, _) = form(&app, &format!("{}/approve", meeting_url(&nb, "meeting:nope")), &[], true).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn starting_a_meeting_needs_a_speech_to_text_model() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        let (status, _, body) = form(&app, &format!("{}/meetings", studio_url(&nb)), &[("title", "Sync")], true).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert!(body.contains("speech-to-text"), "{body}");
    }

    #[tokio::test]
    async fn meeting_lifecycle_through_the_studio_routes() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        with_stt(&app);
        let language = MockLanguageModel::new([summary_reply()]);
        app.state.models.set_override(AnyModel::Language(language));

        // Start: the card comes back with the meeting id.
        let (status, _, card) =
            form(&app, &format!("{}/meetings", studio_url(&nb)), &[("title", "Budget sync"), ("capture_mode", "microphone")], true)
                .await;
        assert_eq!(status, StatusCode::OK, "{card}");
        assert!(card.contains("Budget sync") && card.contains("Recording"));
        assert!(card.contains("Finish interrupted recording"));
        let meeting = meeting_id_in(&card);

        // A chunk is transcribed; bad audio is a 400 with a readable reason.
        let (status, body) = upload(&app, &nb, &meeting, 0, &wav(16_000)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(serde_json::from_str::<serde_json::Value>(&body).unwrap()["text"], QUOTE);
        let (status, body) = upload(&app, &nb, &meeting, 1, b"not a wav").await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("mono 16-bit PCM WAV"), "{body}");

        // While recording, the card shows the live transcript.
        let (_, card) = htmx_get(&app, &meeting_url(&nb, &meeting)).await;
        assert!(card.contains(QUOTE) && card.contains("[00:00]"), "{card}");

        // Finish: review, summary queued.
        let (status, _, card) =
            form(&app, &format!("{}/finish", meeting_url(&nb, &meeting)), &[("total_chunks", "1")], true).await;
        assert_eq!(status, StatusCode::OK, "{card}");
        assert!(card.contains("To review"));
        wait_summary(&app, &meeting).await;
        let (_, card) = htmx_get(&app, &meeting_url(&nb, &meeting)).await;
        assert!(card.contains("Budget follow-up agreed.") && card.contains("Send budget"), "{card}");
        assert!(card.contains("Priya · Friday"));
        assert!(!card.contains("every 3s"), "a finished summary stops polling");

        // Edit form, then save an edit that removes nothing and renames.
        let (status, edit) = htmx_get(&app, &format!("{}/edit", meeting_url(&nb, &meeting))).await;
        assert_eq!(status, StatusCode::OK);
        assert!(edit.contains("name=\"a0_text\"") && edit.contains("name=\"revision\""));
        let stored = Meeting::get(&app.state, &meeting).await.unwrap();
        let revision = stored.revision.to_string();
        let (status, headers, card) = form(
            &app,
            &meeting_url(&nb, &meeting),
            &[
                ("revision", &revision),
                ("title", "Budget review"),
                ("summary", "Edited summary"),
                ("transcript", &stored.transcript),
                ("a0_text", "Send the budget"),
                ("a0_owner", "Priya"),
                ("a0_due", ""),
                ("a0_evidence", QUOTE),
                ("a0_done", "true"),
            ],
            true,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{card}");
        assert!(card.contains("Budget review") && card.contains("Edited summary"));
        assert!(headers["HX-Trigger"].to_str().unwrap().contains("toast"));
        let stored = Meeting::get(&app.state, &meeting).await.unwrap();
        assert_eq!(stored.title, "Budget review");
        assert!(stored.actions[0].done);

        // A stale revision is refused.
        let (status, _, body) = form(
            &app,
            &meeting_url(&nb, &meeting),
            &[("revision", "0"), ("title", "X"), ("summary", ""), ("transcript", "T")],
            true,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

        // Approve: a note appears and the page hears about it.
        let (status, headers, card) = form(&app, &format!("{}/approve", meeting_url(&nb, &meeting)), &[], true).await;
        assert_eq!(status, StatusCode::OK, "{card}");
        assert!(card.contains("Approved"));
        assert!(headers["HX-Trigger"].to_str().unwrap().contains("notes-changed"));
        let note_id = format!("note:meeting_{}", meeting.split_once(':').unwrap().1);
        let rows = app.state.db.query("SELECT title FROM $id", Params::new().record("id", &note_id)).await.unwrap();
        assert_eq!(rows[0]["title"], "Budget review");

        // An approved meeting cannot be discarded.
        let (status, _, _) = form(&app, &format!("{}/discard", meeting_url(&nb, &meeting)), &[], true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        // The list shows it.
        let (status, list) = htmx_get(&app, &format!("{}/meetings", studio_url(&nb))).await;
        assert_eq!(status, StatusCode::OK);
        assert!(list.contains("Budget review") && !list.contains("No meetings yet"));
    }

    #[tokio::test]
    async fn failed_summary_retry_and_discard() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        with_stt(&app);
        app.state.models.set_override(AnyModel::Language(MockLanguageModel::new(["not json", "still not json"])));

        let (_, _, card) = form(&app, &format!("{}/meetings", studio_url(&nb)), &[("title", "")], true).await;
        assert!(card.contains(">Meeting<"), "blank titles become 'Meeting': {card}");
        let meeting = meeting_id_in(&card);
        upload(&app, &nb, &meeting, 0, &wav(8_000)).await;
        // A plain form post (no htmx) redirects back to the notebook.
        let (status, headers, _) =
            form(&app, &format!("{}/finish", meeting_url(&nb, &meeting)), &[("total_chunks", "1")], false).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(headers[header::LOCATION], notebook_url(&nb));
        wait_summary(&app, &meeting).await;

        let (_, card) = htmx_get(&app, &meeting_url(&nb, &meeting)).await;
        assert!(card.contains("Summary failed."), "{card}");
        assert!(card.contains("Generate summary again"));

        let (status, _, card) = form(&app, &format!("{}/summarize", meeting_url(&nb, &meeting)), &[], true).await;
        assert_eq!(status, StatusCode::OK, "{card}");
        wait_summary(&app, &meeting).await;

        let (status, _, body) = form(&app, &format!("{}/discard", meeting_url(&nb, &meeting)), &[], true).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.is_empty());
        assert!(Meeting::get(&app.state, &meeting).await.is_err());
    }

    #[tokio::test]
    async fn interrupted_recording_can_be_recovered() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        with_stt(&app);
        app.state.models.set_override(AnyModel::Language(MockLanguageModel::new([summary_reply()])));
        let (_, _, card) = form(&app, &format!("{}/meetings", studio_url(&nb)), &[("title", "Lost")], true).await;
        let meeting = meeting_id_in(&card);
        upload(&app, &nb, &meeting, 0, &wav(8_000)).await;
        let (status, _, card) = form(
            &app,
            &format!("{}/finish", meeting_url(&nb, &meeting)),
            &[("total_chunks", "2"), ("interrupted", "true")],
            true,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{card}");
        assert!(card.contains("interrupted") && card.contains("1 audio sections are missing"), "{card}");
        // Editing a recording is refused; this one is in review now, so it works.
        let (status, _) = htmx_get(&app, &format!("{}/edit", meeting_url(&nb, &meeting))).await;
        assert_eq!(status, StatusCode::OK);
        let (status, _, _) = form(&app, &format!("{}/finish", meeting_url(&nb, &meeting)), &[("total_chunks", "x")], true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn generators_need_sources_and_list_their_jobs() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        let (status, _, body) = form(&app, &format!("{}/explainers", studio_url(&nb)), &[], true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        let (status, _, _) = form(&app, &format!("{}/infographics", studio_url(&nb)), &[("style", "nope")], true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        add_source(&app, &nb).await;
        // The model fails, so the job ends failed with its reason on screen.
        app.state.models.set_override(AnyModel::Language(MockLanguageModel::new(["no html here", "no svg here"])));
        let (status, headers, jobs) =
            form(&app, &format!("{}/explainers", studio_url(&nb)), &[("instructions", "For beginners")], true).await;
        assert_eq!(status, StatusCode::OK, "{jobs}");
        assert!(headers["HX-Trigger"].to_str().unwrap().contains("Explainer started"));
        assert!(jobs.contains("st-jobs") && jobs.contains("data-job-kind=\"explainer\""), "{jobs}");

        let (status, _, jobs) = form(
            &app,
            &format!("{}/infographics", studio_url(&nb)),
            &[("style", "chalkboard"), ("layout", "funnel"), ("aspect", "square"), ("instructions", "")],
            true,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{jobs}");
        assert!(jobs.contains("data-job-kind=\"infographic\""));
        let (status, _, _) = form(&app, &format!("{}/infographics", studio_url(&nb)), &[("aspect", "round")], true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let long = "x".repeat(MAX_INSTRUCTIONS_CHARS + 1);
        let (status, _, _) = form(&app, &format!("{}/explainers", studio_url(&nb)), &[("instructions", &long)], true).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

        for job in [explainers::list_explainer_jobs(&app.state, &nb).await.unwrap(), infographics::list_infographic_jobs(&app.state, &nb).await.unwrap()].concat() {
            app.wait_for_job(&job.job_id).await;
        }
        let (status, jobs) = htmx_get(&app, &format!("{}/jobs", studio_url(&nb))).await;
        assert_eq!(status, StatusCode::OK);
        assert!(jobs.contains("Could not generate the explainer"), "{jobs}");
        assert!(jobs.contains("Could not generate the infographic"), "{jobs}");
        assert!(!jobs.contains("every 4s"), "finished jobs stop polling: {jobs}");
        // With sources, the generate buttons are enabled.
        let panel = studio_panel(&app.state, &nb).await.unwrap().into_string();
        assert!(!panel.contains("Add a source to this notebook first."));
    }

    #[test]
    fn running_and_stuck_jobs_poll() {
        let now = Utc::now();
        let job = |status: &str, age: i64| GenerationJobSummary {
            job_id: format!("command:{status}{age}"),
            status: status.into(),
            requested_at: Some((now - chrono::Duration::seconds(age)).to_rfc3339()),
            note_id: None,
            title: Some("Done".into()),
            error_message: None,
        };
        let html = jobs_list(
            "notebook:a",
            &[(Kind::Explainer, job("running", 5)), (Kind::Infographic, job("queued", 120)), (Kind::Explainer, job("completed", 9))],
            now,
        )
        .into_string();
        assert!(html.contains("every 4s"));
        assert!(html.contains("Generating explainer"));
        assert!(html.contains("Waiting for the background worker"));
        assert!(html.contains("data-job-status=\"completed\"") && html.contains("hidden"));
    }
}
