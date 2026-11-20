//! The podcasts pages: episodes (list, player, details, publish, retry,
//! delete), generating from a notebook, reading a script, episode and speaker
//! profiles, and the private Podcast Feed (URL, rotation, show details and
//! artwork).
//!
//! The counterpart of `app/(dashboard)/podcasts/page.tsx` and
//! `components/podcasts` in the Next.js frontend. Every action calls
//! `services::podcasts` / `services::podcast_feed`, the same functions the
//! JSON API uses.

use std::collections::HashMap;

use axum::extract::{Multipart, Path, Query, State};
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::{Form, Router};
use chrono::{DateTime, Utc};
use maud::{html, Markup};
use nn_podcast::feed::{FeedSettingsUpdate, FeedSettingsView};
use serde::Deserialize;
use serde_json::{Map, Value as Json};

use crate::api::auth::encode_query_value;
use crate::api::podcast_feed::forwarded_origin;
use crate::error::AppError;
use crate::jobs::status;
use crate::services::models::{list_models, ModelResponse};
use crate::services::notebooks::{list_notebooks, NotebookQuery, NotebookSummary};
use crate::services::podcast_feed;
use crate::services::podcasts::{
    self, EpisodeProfileInput, EpisodeProfileView, EpisodeView, GenerateRequest, ScriptRequest, SpeakerProfileInput,
    SpeakerProfileView,
};
use crate::state::AppState;
use crate::web::{
    dom_id, empty_state, human_date, is_htmx, multipart_error, page, page_head, redirect, safe_path, seg, with_toast, Fields,
    PageContext, WebError, WebResult,
};

/// Speakers a profile can have (the API's limit).
const MAX_SPEAKERS: usize = 4;
/// The longest script the dialog accepts (the frontend's `MAX_SCRIPT_CHARS`).
const MAX_SCRIPT_CHARS: usize = 500_000;
/// The default feed description is cut to this many characters.
const SUMMARY_EXCERPT_CHARS: usize = 300;
/// The longest feed description the server accepts.
const SUMMARY_MAX_CHARS: usize = 4000;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/podcasts", get(episodes_page))
        .route("/podcasts/episodes", get(episodes_fragment))
        .route("/podcasts/episodes/{id}", get(episode_page))
        .route("/podcasts/episodes/{id}/card", get(episode_card_fragment))
        .route("/podcasts/episodes/{id}/delete", post(delete_episode))
        .route("/podcasts/episodes/{id}/retry", post(retry_episode))
        .route("/podcasts/episodes/{id}/publish", post(publish_episode))
        .route("/podcasts/episodes/{id}/unpublish", post(unpublish_episode))
        .route("/podcasts/generate", get(generate_page).post(generate))
        .route("/podcasts/read-script", get(read_script_page).post(read_script))
        .route("/podcasts/profiles", get(profiles_page))
        .route("/podcasts/profiles/speakers/new", get(new_speaker_page).post(create_speaker))
        .route("/podcasts/profiles/speakers/{id}", get(edit_speaker_page).post(update_speaker))
        .route("/podcasts/profiles/speakers/{id}/duplicate", post(duplicate_speaker))
        .route("/podcasts/profiles/speakers/{id}/delete", post(delete_speaker))
        .route("/podcasts/profiles/episodes/new", get(new_episode_profile_page).post(create_episode_profile))
        .route("/podcasts/profiles/episodes/{id}", get(edit_episode_profile_page).post(update_episode_profile))
        .route("/podcasts/profiles/episodes/{id}/duplicate", post(duplicate_episode_profile))
        .route("/podcasts/profiles/episodes/{id}/delete", post(delete_episode_profile))
        .route("/podcasts/feed", get(feed_page))
        .route("/podcasts/feed/rotate", post(rotate_feed))
        .route("/podcasts/feed/settings", post(save_feed_settings))
        .route("/podcasts/feed/artwork", post(upload_artwork))
        .route("/notebooks/{id}/podcasts", get(notebook_podcasts_fragment))
}

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

fn non_empty(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

fn parse_date(value: Option<&str>) -> Option<DateTime<Utc>> {
    value.and_then(|v| DateTime::parse_from_rfc3339(v).ok()).map(|d| d.with_timezone(&Utc))
}

/// A non-empty string field of a JSON snapshot.
fn text<'a>(map: &'a Map<String, Json>, key: &str) -> Option<&'a str> {
    map.get(key).and_then(Json::as_str).filter(|s| !s.trim().is_empty())
}

/// A scalar field of a JSON snapshot for display, or a dash.
fn shown(map: &Map<String, Json>, key: &str) -> String {
    match map.get(key) {
        Some(Json::String(s)) if !s.is_empty() => s.clone(),
        Some(Json::Number(n)) => n.to_string(),
        _ => "—".into(),
    }
}

/// "provider / name" from the resolved display fields, then the legacy snapshot strings.
fn model_label(map: &Map<String, Json>, provider: &str, name: &str, legacy_provider: &str, legacy_name: &str) -> String {
    let provider = text(map, provider).or_else(|| text(map, legacy_provider)).unwrap_or("—");
    let name = text(map, name).or_else(|| text(map, legacy_name)).unwrap_or("—");
    format!("{provider} / {name}")
}

/// Model id → "provider / name".
async fn model_names(state: &AppState) -> HashMap<String, String> {
    match list_models(state, None).await {
        Ok(models) => models.into_iter().map(|m| (m.id, format!("{} / {}", m.provider, m.name))).collect(),
        Err(e) => {
            tracing::warn!("Could not list models for the podcasts page: {e}");
            HashMap::new()
        }
    }
}

async fn active_notebooks(state: &AppState) -> WebResult<Vec<NotebookSummary>> {
    Ok(list_notebooks(state, &NotebookQuery { archived: Some(false), ..Default::default() }).await?)
}

fn episode_needs_setup(profile: &EpisodeProfileView) -> bool {
    profile.outline_llm.is_none() || profile.transcript_llm.is_none()
}

fn speaker_needs_setup(profile: &SpeakerProfileView) -> bool {
    profile.voice_model.is_none()
}

fn setup_alert() -> Markup {
    html! {
        div.flash.warn role="alert" {
            strong { "Setup required" }
            " Some profiles don't have models configured yet. Edit them to select models before generating podcasts."
        }
    }
}

/// Episodes / Profiles / Private feed.
fn area_tabs(current: &str) -> Markup {
    let tabs = [("/podcasts", "Episodes"), ("/podcasts/profiles", "Profiles"), ("/podcasts/feed", "Private feed")];
    html! {
        nav.tabs aria-label="Podcast views" {
            @for (href, label) in tabs {
                a.tab href=(href) aria-current=[(href == current).then_some("page")] { (label) }
            }
        }
    }
}

fn podcasts_head() -> Markup {
    page_head("Podcasts", Some("Keep track of generated episodes and manage reusable profiles."), html! {})
}

// ---------------------------------------------------------------------------
// Episode status
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Group {
    Running,
    Pending,
    Completed,
    Failed,
}

fn group_of(status: Option<&str>) -> Group {
    match status.unwrap_or("unknown") {
        "running" | "processing" => Group::Running,
        "completed" => Group::Completed,
        "failed" | "error" => Group::Failed,
        _ => Group::Pending,
    }
}

/// Still generating: the card shows progress and polls.
fn is_active(status: Option<&str>) -> bool {
    matches!(status, Some("running" | "processing" | "pending" | "submitted" | "new" | "queued"))
}

fn is_failed(status: Option<&str>) -> bool {
    matches!(status, Some("failed" | "error"))
}

fn episode_badge(status: Option<&str>) -> Markup {
    let (class, label) = match status.unwrap_or("unknown") {
        "completed" => return html! {},
        "running" | "processing" => ("warn", "Processing"),
        "failed" | "error" => ("bad", "Failed"),
        "pending" | "submitted" | "new" | "queued" => ("info", "Pending"),
        _ => ("", "Unknown"),
    };
    html! { span class=(format!("badge {class}")) { (label) } }
}

fn is_script(episode: &EpisodeView) -> bool {
    text(&episode.episode_profile, "mode") == Some("script")
}

fn profile_label(episode: &EpisodeView) -> String {
    if is_script(episode) {
        "Script reading".into()
    } else {
        text(&episode.episode_profile, "name").unwrap_or("Unknown").to_string()
    }
}

fn episode_meta(episode: &EpisodeView) -> String {
    let created = human_date(parse_date(episode.created.as_deref()));
    if created.is_empty() {
        format!("Profile: {}", profile_label(episode))
    } else {
        format!("Profile: {} · Created {created}", profile_label(episode))
    }
}

/// The feed description the server would build: the outline's segment
/// descriptions (whitespace collapsed), else the episode name, cut to 300
/// characters at a word boundary.
fn default_summary(episode: &EpisodeView) -> String {
    let segments = episode.outline.as_ref().and_then(|o| o.get("segments")).and_then(Json::as_array);
    let parts: Vec<String> = segments
        .into_iter()
        .flatten()
        .filter_map(|segment| segment.get("description"))
        .map(|d| match d {
            Json::String(s) => s.split_whitespace().collect::<Vec<_>>().join(" "),
            Json::Null => String::new(),
            other => other.to_string(),
        })
        .filter(|s| !s.is_empty())
        .collect();
    let text = if parts.is_empty() { episode.name.clone() } else { parts.join(" ") };
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= SUMMARY_EXCERPT_CHARS {
        return text;
    }
    let mut cut: String = chars[..SUMMARY_EXCERPT_CHARS - 1].iter().collect();
    if !chars[SUMMARY_EXCERPT_CHARS - 1].is_whitespace() && cut.contains(' ') {
        if let Some(space) = cut.rfind(' ') {
            cut.truncate(space);
        }
    }
    format!("{}…", cut.trim_end())
}

// ---------------------------------------------------------------------------
// Episode cards and the list
// ---------------------------------------------------------------------------

fn progress_bar(episode: &EpisodeView) -> Markup {
    if !is_active(episode.job_status.as_deref()) {
        return html! {};
    }
    let value = episode.progress.map(|p| p.clamp(0, 100));
    html! {
        div.row.pod-progress {
            @match value {
                Some(v) => {
                    progress max="100" value=(v) aria-label="Generation progress" {}
                    span.small.faint { (v) "%" }
                }
                None => { progress aria-label="Generation progress" {} }
            }
        }
    }
}

fn player(episode: &EpisodeView) -> Markup {
    match &episode.audio_url {
        Some(url) => html! { audio.pod-audio controls preload="none" src=(url) {} },
        None => html! {},
    }
}

/// Publish / Unpublish. Offered for completed episodes with audio; a
/// published one always shows its state and Unpublish.
fn publish_control(episode: &EpisodeView, in_list: bool) -> Markup {
    let base = format!("/podcasts/episodes/{}", episode.id);
    if episode.published_at.is_some() {
        return html! {
            span.badge.ok { "In your feed" }
            form.inline method="post" action=(format!("{base}/unpublish"))
                hx-post=[in_list.then(|| format!("{base}/unpublish"))]
                hx-target=[in_list.then_some("#pod-episodes")] hx-swap=[in_list.then_some("outerHTML")] {
                button.btn.btn-sm type="submit" { "Unpublish" }
            }
        };
    }
    let publishable = episode.job_status.as_deref() == Some("completed") && episode.audio_url.is_some();
    if !publishable {
        return html! {};
    }
    let dialog = format!("pod-publish-{}", dom_id(&episode.id));
    html! {
        button.btn.btn-sm type="button" data-open=(format!("#{dialog}")) { "Publish" }
        dialog id=(dialog) {
            form method="post" action=(format!("{base}/publish"))
                hx-post=[in_list.then(|| format!("{base}/publish"))]
                hx-target=[in_list.then_some("#pod-episodes")] hx-swap=[in_list.then_some("outerHTML")]
                data-close-on-success {
                div.dialog-head {
                    h2 { "Publish to your feed" }
                    button.btn.btn-ghost.btn-sm type="button" data-close { "Close" }
                }
                div.dialog-body {
                    p.muted { "Podcast apps will show this episode in your private feed. Edit the description they display if you like." }
                    div.field {
                        label for=(format!("{dialog}-summary")) { "Episode description" }
                        textarea id=(format!("{dialog}-summary")) name="summary" rows="6" maxlength=(SUMMARY_MAX_CHARS) {
                            (default_summary(episode))
                        }
                        p.hint { "4000 characters at most." }
                    }
                }
                div.dialog-foot {
                    button.btn type="button" data-close { "Cancel" }
                    button.btn.btn-primary type="submit" { "Publish episode" }
                }
            }
        }
    }
}

/// One action button as a form. In the list it posts with htmx and swaps the list.
fn action_form(action: &str, label: &str, class: &str, confirm: Option<&str>, in_list: bool) -> Markup {
    html! {
        form.inline method="post" action=(action)
            hx-post=[in_list.then_some(action)]
            hx-target=[in_list.then_some("#pod-episodes")] hx-swap=[in_list.then_some("outerHTML")]
            hx-confirm=[confirm] data-pod-confirm=[(!in_list).then_some(confirm).flatten()] {
            button type="submit" class=(class) { (label) }
        }
    }
}

fn episode_card(episode: &EpisodeView) -> Markup {
    let status = episode.job_status.as_deref();
    let base = format!("/podcasts/episodes/{}", episode.id);
    let active = is_active(status);
    let delete_confirm = format!("Delete episode? This will remove “{}” and its audio file permanently.", episode.name);
    html! {
        article.card.stack-sm.pod-episode id=(format!("pod-ep-{}", dom_id(&episode.id)))
            hx-get=[active.then(|| format!("{base}/card"))]
            hx-trigger=[active.then_some("every 3s")] hx-swap=[active.then_some("outerHTML")] {
            div.row-between {
                div.grow {
                    div.row {
                        h3 { a href=(base) { (episode.name) } }
                        (episode_badge(status))
                    }
                    p.small.faint { (episode_meta(episode)) }
                }
                div.row {
                    a.btn.btn-sm href=(base) { "Details" }
                    @if let Some(url) = &episode.audio_url {
                        a.btn.btn-sm href=(url) download { "Download" }
                    }
                    (publish_control(episode, true))
                    @if is_failed(status) {
                        (action_form(&format!("{base}/retry"), "Retry", "btn btn-sm", None, true))
                    }
                    (action_form(&format!("{base}/delete"), "Delete", "btn btn-sm btn-ghost pod-danger", Some(&delete_confirm), true))
                }
            }
            (progress_bar(episode))
            (player(episode))
            @if is_failed(status) {
                @if let Some(error) = &episode.error_message {
                    div.flash.bad {
                        strong.small { "Error details" }
                        p.small.pre-wrap { (error) }
                    }
                }
            }
        }
    }
}

const GROUPS: [(Group, &str, &str); 4] = [
    (Group::Running, "Currently Processing", "Episodes that are actively generating assets."),
    (Group::Pending, "Queued / Pending", "Submitted episodes waiting to start processing."),
    (Group::Completed, "Completed Episodes", "Ready to review, download, or publish."),
    (Group::Failed, "Failed Episodes", "Episodes that encountered issues during generation."),
];

/// The episode list. With `watch` (a job just submitted from this page) the
/// whole list polls until that job's episode shows up; each unfinished card
/// polls on its own and refreshes the list when it finishes.
fn episode_list(episodes: &[EpisodeView], watch: Option<&str>) -> Markup {
    let count = |group: Group| episodes.iter().filter(|e| group_of(e.job_status.as_deref()) == group).count();
    let url = match watch {
        Some(job) => format!("/podcasts/episodes?watch={}", encode_query_value(job)),
        None => "/podcasts/episodes".to_string(),
    };
    let trigger = if watch.is_some() { "every 3s, pod-episodes-changed from:body" } else { "pod-episodes-changed from:body" };
    html! {
        div #pod-episodes .stack hx-get=(url) hx-trigger=(trigger) hx-swap="outerHTML" {
            div.row.pod-counts {
                @for (label, value) in [
                    ("Total", episodes.len()),
                    ("Processing", count(Group::Running)),
                    ("Completed", count(Group::Completed)),
                    ("Failed", count(Group::Failed)),
                    ("Pending", count(Group::Pending)),
                ] {
                    span.badge { (label) " " strong { (value) } }
                }
            }
            @if watch.is_some() {
                p.row.small.muted { span.spinner {} "Starting the new episode…" }
            }
            @if episodes.is_empty() && watch.is_none() {
                (empty_state("No podcast episodes yet", "Click Generate Podcast to create your first one."))
            }
            @for (group, title, description) in GROUPS {
                @let items: Vec<&EpisodeView> = episodes.iter().filter(|e| group_of(e.job_status.as_deref()) == group).collect();
                @if !items.is_empty() {
                    @if group == Group::Completed {
                        details #pod-completed .pod-group {
                            summary {
                                h3 { (title) " " span.faint.small { "(" (items.len()) ")" } }
                                p.small.muted { (description) }
                            }
                            div.pod-grid { @for episode in &items { (episode_card(episode)) } }
                        }
                    } @else {
                        section.pod-group.stack-sm {
                            div {
                                h3 { (title) }
                                p.small.muted { (description) }
                            }
                            @for episode in &items { (episode_card(episode)) }
                        }
                    }
                }
            }
        }
    }
}

/// The job to keep watching: it has not started yet, or it is running but its
/// episode is not in the list as unfinished yet.
async fn still_watching(state: &AppState, watch: Option<&str>, episodes: &[EpisodeView]) -> Option<String> {
    let job_id = watch.filter(|w| !w.trim().is_empty())?;
    let job = state.jobs.get(job_id).await.ok().flatten()?;
    let current = job.status_str();
    if status::is_terminal(current) {
        return None;
    }
    if current == status::RUNNING && episodes.iter().any(|e| is_active(e.job_status.as_deref())) {
        return None;
    }
    Some(job_id.to_string())
}

async fn render_list(state: &AppState, watch: Option<&str>) -> WebResult<Markup> {
    let episodes = podcasts::list_episodes(state, None).await?;
    let watch = still_watching(state, watch, &episodes).await;
    Ok(episode_list(&episodes, watch.as_deref()))
}

#[derive(Debug, Default, Deserialize)]
struct ListQuery {
    watch: Option<String>,
}

// ---------------------------------------------------------------------------
// A notebook's podcasts (embedded in the notebook page)
// ---------------------------------------------------------------------------

/// The podcasts filed under a notebook, for the notebook page: each with its
/// status, progress and player, and a Generate link with the notebook chosen.
/// The panel polls while any of its episodes is still generating.
pub async fn notebook_podcasts_panel(state: &AppState, notebook_id: &str) -> WebResult<Markup> {
    let episodes = podcasts::list_episodes(state, Some(notebook_id)).await?;
    let notebook_href = format!("/notebooks/{}", seg(notebook_id));
    let generate_href = format!(
        "/podcasts/generate?notebook_id={}&return_to={}",
        encode_query_value(notebook_id),
        encode_query_value(&notebook_href)
    );
    let polling = episodes.iter().any(|e| is_active(e.job_status.as_deref()));
    let refresh = format!("{notebook_href}/podcasts");
    Ok(html! {
        section.card.stack-sm.pod-notebook #pod-notebook aria-label="Notebook podcasts"
            hx-get=[polling.then_some(refresh.as_str())] hx-trigger=[polling.then_some("every 5s")] hx-swap=[polling.then_some("outerHTML")] {
            div.row-between {
                h3.pod-notebook-title { "Podcasts" }
                a.btn.btn-sm href=(generate_href) { "Generate" }
            }
            @if episodes.is_empty() {
                p.small.muted.pod-notebook-empty { "No podcasts for this notebook yet." }
            }
            @for episode in &episodes {
                @let status = episode.job_status.as_deref();
                article.stack-sm.pod-notebook-item {
                    div.row-between {
                        div.grow {
                            a href=(format!("/podcasts/episodes/{}", episode.id)) { strong { (episode.name) } }
                            @if let Some(created) = parse_date(episode.created.as_deref()) {
                                p.small.faint { "Created " (human_date(Some(created))) }
                            }
                        }
                        @if is_failed(status) || is_active(status) { (episode_badge(status)) }
                        @if episode.published_at.is_some() { span.badge.ok { "In your feed" } }
                    }
                    (progress_bar(episode))
                    @if status == Some("completed") { (player(episode)) }
                }
            }
        }
    })
}

async fn notebook_podcasts_fragment(State(state): State<AppState>, Path(id): Path<String>) -> WebResult<Markup> {
    notebook_podcasts_panel(&state, &id).await
}

// ---------------------------------------------------------------------------
// Episodes page
// ---------------------------------------------------------------------------

async fn episodes_page(State(state): State<AppState>) -> WebResult<Markup> {
    let list = render_list(&state, None).await?;
    let episode_profiles = podcasts::list_episode_profiles(&state).await?;
    let speaker_profiles = podcasts::list_speaker_profiles(&state).await?;
    let notebooks = active_notebooks(&state).await?;
    let needs_setup =
        episode_profiles.iter().any(episode_needs_setup) || speaker_profiles.iter().any(speaker_needs_setup);
    let ctx = PageContext::new(&state, "/podcasts");
    let body = html! {
        (podcasts_head())
        (area_tabs("/podcasts"))
        div.stack {
            @if needs_setup { (setup_alert()) }
            div.row-between {
                div {
                    h2 { "Episodes overview" }
                    p.small.muted { "Monitor podcast generation jobs and review the final artefacts." }
                }
                div.row {
                    a.btn.btn-primary.pod-dialog-link href="/podcasts/generate" data-open="#pod-generate" { "Generate Podcast" }
                    a.btn.pod-dialog-link href="/podcasts/read-script" data-open="#pod-script" { "Read a Script" }
                    a.btn.btn-sm href="/podcasts" { "Refresh" }
                }
            }
            (list)
        }
        dialog #pod-generate .wide {
            (generate_form(&notebooks, &episode_profiles, &speaker_profiles, true, &GeneratePreset::default()))
        }
        dialog #pod-script .wide {
            (script_form(&notebooks, &speaker_profiles, true))
        }
    };
    Ok(page(&ctx, "Podcasts", body))
}

async fn episodes_fragment(State(state): State<AppState>, Query(query): Query<ListQuery>) -> WebResult<Markup> {
    render_list(&state, query.watch.as_deref()).await
}

/// One card, for its own polling. Once the episode has finished (or is gone)
/// the list is told to refresh so the card moves to its new group.
async fn episode_card_fragment(State(state): State<AppState>, Path(id): Path<String>) -> WebResult<Response> {
    let changed = |body: Markup| {
        let mut response = body.into_response();
        response.headers_mut().insert("HX-Trigger", HeaderValue::from_static("pod-episodes-changed"));
        response
    };
    match podcasts::get_episode(&state, &id).await {
        Ok(episode) if is_active(episode.job_status.as_deref()) => Ok(episode_card(&episode).into_response()),
        Ok(episode) => Ok(changed(episode_card(&episode))),
        Err(e) if e.is_not_found() => Ok(changed(html! {})),
        Err(e) => Err(e.into()),
    }
}

#[derive(Debug, Default, Deserialize)]
struct DetailQuery {
    tab: Option<String>,
}

async fn episode_page(
    State(state): State<AppState>,
    Path(id): Path<String>,
    Query(query): Query<DetailQuery>,
) -> WebResult<Markup> {
    let episode = podcasts::get_episode(&state, &id).await?;
    let status = episode.job_status.as_deref();
    let base = format!("/podcasts/episodes/{}", episode.id);
    let tab = match query.tab.as_deref() {
        Some("outline") => "outline",
        Some("transcript") => "transcript",
        _ => "summary",
    };
    let ctx = PageContext::new(&state, base.clone());
    let delete_confirm = format!("Delete episode? This will remove “{}” and its audio file permanently.", episode.name);
    let body = html! {
        div.crumbs { a href="/podcasts" { "Podcasts" } " / " (episode.name) }
        (page_head(&episode.name, Some(&episode_meta(&episode)), html! {
            (episode_badge(status))
            @if let Some(url) = &episode.audio_url {
                a.btn.btn-sm href=(url) download { "Download" }
            }
            @if episode.published_at.is_some() {
                span.badge.ok { "In your feed" }
                (action_form(&format!("{base}/unpublish"), "Unpublish", "btn btn-sm", None, false))
            }
            @if is_failed(status) {
                (action_form(&format!("{base}/retry"), "Retry", "btn btn-sm", None, false))
            }
            (action_form(&format!("{base}/delete"), "Delete", "btn btn-sm btn-danger", Some(&delete_confirm), false))
        }))
        div.stack {
            (progress_bar(&episode))
            @if is_active(status) {
                p.small.muted { "This episode is still generating. " a href=(base) { "Refresh" } " to see its progress." }
            }
            (player(&episode))
            @if is_failed(status) {
                @if let Some(error) = &episode.error_message {
                    div.flash.bad { strong.small { "Error details" } p.small.pre-wrap { (error) } }
                }
            }
            @if episode.published_at.is_none()
                && status == Some("completed") && episode.audio_url.is_some() {
                details.card.flat.pod-publish {
                    summary { strong { "Publish to your feed" } }
                    form.stack-sm method="post" action=(format!("{base}/publish")) {
                        p.small.muted { "Podcast apps will show this episode in your private feed. Edit the description they display if you like." }
                        div.field {
                            label for="pod-summary" { "Episode description" }
                            textarea #pod-summary name="summary" rows="6" maxlength=(SUMMARY_MAX_CHARS) { (default_summary(&episode)) }
                        }
                        div.row { button.btn.btn-primary type="submit" { "Publish episode" } }
                    }
                }
            }
            nav.tabs aria-label="Episode details" {
                @for (key, label) in [("summary", "Summary"), ("outline", "Outline"), ("transcript", "Transcript")] {
                    a.tab href=(format!("{base}?tab={key}")) aria-current=[(key == tab).then_some("page")] { (label) }
                }
            }
            @match tab {
                "outline" => (outline_tab(&episode)),
                "transcript" => (transcript_tab(&episode)),
                _ => (summary_tab(&episode)),
            }
        }
    };
    Ok(page(&ctx, &episode.name, body))
}

fn summary_tab(episode: &EpisodeView) -> Markup {
    let ep = &episode.episode_profile;
    let sp = &episode.speaker_profile;
    let speakers = sp.get("speakers").and_then(Json::as_array).cloned().unwrap_or_default();
    html! {
        div.stack {
            @if !is_script(episode) {
                section.stack-sm {
                    h3 { "Episode Profile" }
                    div.pod-facts {
                        div { p.faint.small { "Outline model" } p { (model_label(ep, "outline_model_provider", "outline_model_name", "outline_provider", "outline_model")) } }
                        div { p.faint.small { "Transcript model" } p { (model_label(ep, "transcript_model_provider", "transcript_model_name", "transcript_provider", "transcript_model")) } }
                        div { p.faint.small { "Segments" } p { (shown(ep, "num_segments")) } }
                        div { p.faint.small { "Max output tokens" } p { (shown(ep, "max_tokens")) } }
                    }
                    @if let Some(briefing) = text(ep, "default_briefing") {
                        div.pod-box.small.pre-wrap { (briefing) }
                    }
                }
            }
            section.stack-sm {
                h3 { "Speaker Profile" }
                p.small.faint { (model_label(sp, "voice_model_provider", "voice_model_name", "tts_provider", "tts_model")) }
                @for speaker in &speakers {
                    @if let Some(speaker) = speaker.as_object() { (speaker_box(speaker, &HashMap::new())) }
                }
            }
            @if !episode.briefing.trim().is_empty() {
                section.stack-sm {
                    h3 { "Briefing" }
                    div.pod-box.small.pre-wrap { (episode.briefing) }
                }
            }
        }
    }
}

fn outline_tab(episode: &EpisodeView) -> Markup {
    let segments = episode.outline.as_ref().and_then(|o| o.get("segments")).and_then(Json::as_array).cloned().unwrap_or_default();
    html! {
        @if segments.is_empty() {
            p.small.muted { "No outline available." }
        } @else {
            div.stack-sm {
                @for (index, segment) in segments.iter().enumerate() {
                    div.pod-box.small {
                        div.row-between {
                            strong { (segment.get("name").and_then(Json::as_str).map(String::from).unwrap_or_else(|| format!("Segment {}", index + 1))) }
                            @if let Some(size) = segment.get("size").and_then(Json::as_str) { span.badge { (size) } }
                        }
                        p.muted.pre-wrap { (segment.get("description").and_then(Json::as_str).unwrap_or("No description provided.")) }
                    }
                }
            }
        }
    }
}

fn transcript_tab(episode: &EpisodeView) -> Markup {
    let entries = episode.transcript.as_ref().and_then(|t| t.get("transcript")).and_then(Json::as_array).cloned().unwrap_or_default();
    html! {
        @if entries.is_empty() {
            p.small.muted { "No transcript available." }
        } @else {
            div.stack-sm {
                @for entry in &entries {
                    div.pod-box.small {
                        strong { (entry.get("speaker").and_then(Json::as_str).unwrap_or("Speaker")) }
                        p.muted.pre-wrap { (entry.get("dialogue").and_then(Json::as_str).unwrap_or("")) }
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Episode actions
// ---------------------------------------------------------------------------

/// After an action: htmx gets the refreshed list and a toast, a plain form a redirect.
async fn list_or_redirect(state: &AppState, headers: &HeaderMap, watch: Option<&str>, toast: &str) -> WebResult<Response> {
    if is_htmx(headers) {
        Ok(with_toast(render_list(state, watch).await?.into_response(), toast))
    } else {
        Ok(Redirect::to("/podcasts").into_response())
    }
}

async fn delete_episode(State(state): State<AppState>, Path(id): Path<String>, headers: HeaderMap) -> WebResult<Response> {
    podcasts::delete_episode(&state, &id).await?;
    list_or_redirect(&state, &headers, None, "Episode deleted").await
}

async fn retry_episode(State(state): State<AppState>, Path(id): Path<String>, headers: HeaderMap) -> WebResult<Response> {
    let submitted = podcasts::retry_episode(&state, &id).await?;
    list_or_redirect(&state, &headers, Some(&submitted.job_id), "A new podcast generation job has been submitted.").await
}

#[derive(Debug, Default, Deserialize)]
struct PublishForm {
    #[serde(default)]
    summary: Option<String>,
}

async fn publish_episode(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
    Form(form): Form<PublishForm>,
) -> WebResult<Response> {
    let summary = form.summary.as_deref().and_then(non_empty);
    podcasts::publish_episode(&state, &id, summary.as_deref()).await?;
    episode_or_list(&state, &headers, &id, "Published to your feed").await
}

async fn unpublish_episode(State(state): State<AppState>, Path(id): Path<String>, headers: HeaderMap) -> WebResult<Response> {
    podcasts::unpublish_episode(&state, &id).await?;
    episode_or_list(&state, &headers, &id, "Removed from your feed").await
}

/// Publishing changes keep a plain form on the episode's page.
async fn episode_or_list(state: &AppState, headers: &HeaderMap, id: &str, toast: &str) -> WebResult<Response> {
    if is_htmx(headers) {
        return list_or_redirect(state, headers, None, toast).await;
    }
    Ok(Redirect::to(&format!("/podcasts/episodes/{id}")).into_response())
}

// ---------------------------------------------------------------------------
// Generate
// ---------------------------------------------------------------------------

fn notebook_options(notebooks: &[NotebookSummary], placeholder: &str, selected: Option<&str>) -> Markup {
    html! {
        option value="" { (placeholder) }
        @for notebook in notebooks {
            option value=(notebook.id) selected[selected == Some(notebook.id.as_str())] { (notebook.name) }
        }
    }
}

/// The generate form, in the dialog (`in_dialog`, htmx, swaps the list) or on its own page.
fn generate_form(
    notebooks: &[NotebookSummary],
    episode_profiles: &[EpisodeProfileView],
    speaker_profiles: &[SpeakerProfileView],
    in_dialog: bool,
    preset: &GeneratePreset,
) -> Markup {
    let body = html! {
        p.muted { "Choose the notebook to talk about and an episode profile, then name the episode." }
        @if let Some(back) = &preset.return_to { input type="hidden" name="return_to" value=(back); }
        @if episode_profiles.is_empty() {
            div.flash.warn {
                "No episode profiles found. " a href="/podcasts/profiles" { "Create an episode profile" } " before generating a podcast."
            }
        }
        div.field {
            label for="pod-gen-notebook" { "Notebook" }
            select #pod-gen-notebook name="notebook_id" {
                (notebook_options(notebooks, "Select a notebook", preset.notebook_id.as_deref()))
            }
            p.hint { "The episode is made from the notebook's sources and notes, and filed under it." }
        }
        details {
            summary.small.muted { "Or paste the content instead" }
            div.field {
                label for="pod-gen-content" { "Content" }
                textarea #pod-gen-content name="content" rows="5" placeholder="Used instead of the notebook's content when filled in" {}
            }
        }
        div.cols-2 {
            div.field {
                label for="pod-gen-profile" { "Episode profile" }
                select #pod-gen-profile name="episode_profile" required data-pod-profile {
                    option value="" { "Select an episode profile" }
                    @for profile in episode_profiles {
                        option value=(profile.name) data-speaker=(profile.speaker_config_name.as_deref().unwrap_or("")) { (profile.name) }
                    }
                }
                p.hint data-pod-uses-speaker {}
            }
            div.field {
                label for="pod-gen-speaker" { "Speaker Profile" }
                select #pod-gen-speaker name="speaker_profile" {
                    option value="" { "Use the episode profile's speaker" }
                    @for profile in speaker_profiles {
                        option value=(profile.id) { (profile.name) }
                    }
                }
            }
        }
        div.field {
            label for="pod-gen-name" { "Episode name" }
            input #pod-gen-name type="text" name="episode_name" required autocomplete="off" placeholder="e.g., AI and the Future of Work";
        }
        div.field {
            label for="pod-gen-instructions" { "Additional instructions" }
            textarea #pod-gen-instructions name="instructions" rows="3" autocomplete="off"
                placeholder="Any supplementary advice to append to the episode briefing..." {}
        }
    };
    if in_dialog {
        html! {
            form method="post" action="/podcasts/generate" hx-post="/podcasts/generate" hx-target="#pod-episodes" hx-swap="outerHTML"
                data-close-on-success data-reset {
                div.dialog-head {
                    h2 { "Generate Podcast Episode" }
                    button.btn.btn-ghost.btn-sm type="button" data-close { "Close" }
                }
                div.dialog-body { (body) }
                div.dialog-foot {
                    button.btn type="button" data-close { "Cancel" }
                    button.btn.btn-primary type="submit" { span.spinner.htmx-indicator {} "Generate" }
                }
            }
        }
    } else {
        html! {
            form.card.stack method="post" action="/podcasts/generate" {
                (body)
                div.row {
                    button.btn.btn-primary type="submit" { "Generate" }
                    a.btn href=(preset.return_to.as_deref().unwrap_or("/podcasts")) { "Cancel" }
                }
            }
        }
    }
}

/// Pre-filled choices for the generate form, from `?notebook_id=&return_to=`
/// (the notebook page's Generate link).
#[derive(Debug, Default, Deserialize)]
struct GeneratePreset {
    notebook_id: Option<String>,
    return_to: Option<String>,
}

async fn generate_page(State(state): State<AppState>, Query(preset): Query<GeneratePreset>) -> WebResult<Markup> {
    let preset = GeneratePreset { notebook_id: preset.notebook_id, return_to: safe_path(preset.return_to.as_deref()) };
    let notebooks = active_notebooks(&state).await?;
    let episode_profiles = podcasts::list_episode_profiles(&state).await?;
    let speaker_profiles = podcasts::list_speaker_profiles(&state).await?;
    let ctx = PageContext::new(&state, "/podcasts/generate");
    Ok(page(&ctx, "Generate Podcast Episode", html! {
        div.crumbs { a href="/podcasts" { "Podcasts" } " / Generate" }
        (page_head("Generate Podcast Episode", None, html! {}))
        (generate_form(&notebooks, &episode_profiles, &speaker_profiles, false, &preset))
    }))
}

#[derive(Debug, Default, Deserialize)]
struct GenerateForm {
    #[serde(default)]
    notebook_id: String,
    #[serde(default)]
    content: String,
    #[serde(default)]
    episode_profile: String,
    #[serde(default)]
    speaker_profile: String,
    #[serde(default)]
    episode_name: String,
    #[serde(default)]
    instructions: String,
    #[serde(default)]
    return_to: String,
}

async fn generate(State(state): State<AppState>, headers: HeaderMap, Form(form): Form<GenerateForm>) -> WebResult<Response> {
    let episode_profile = non_empty(&form.episode_profile)
        .ok_or_else(|| AppError::invalid("Select an episode profile before generating a podcast."))?;
    let episode_name = non_empty(&form.episode_name).ok_or_else(|| AppError::invalid("Provide a name for the episode."))?;
    // No speaker picked: the episode profile's own (by record id, so a rename can't break it).
    let speaker_profile = match non_empty(&form.speaker_profile) {
        Some(speaker) => speaker,
        None => {
            let profiles = podcasts::list_episode_profiles(&state).await?;
            let profile = profiles
                .iter()
                .find(|p| p.name == episode_profile)
                .ok_or_else(|| AppError::not_found(format!("Episode profile '{episode_profile}' not found")))?;
            match (&profile.speaker_config, &profile.speaker_config_name) {
                (Some(id), Some(_)) => id.clone(),
                _ => {
                    return Err(WebError(AppError::invalid(
                        "This episode profile references a speaker profile that no longer exists. Edit the profile and select a speaker profile.",
                    )))
                }
            }
        }
    };
    let content = non_empty(&form.content);
    let notebook_id = non_empty(&form.notebook_id);
    if content.is_none() && notebook_id.is_none() {
        return Err(WebError(AppError::invalid("Choose a notebook, or paste the content to talk about.")));
    }
    let request = GenerateRequest {
        episode_profile,
        speaker_profile,
        episode_name,
        content,
        notebook_id,
        briefing_suffix: non_empty(&form.instructions),
    };
    let submitted = podcasts::submit_generation(&state, &request).await?;
    // Started from a notebook page: go back there, where its podcast list polls.
    if let (false, Some(back)) = (is_htmx(&headers), safe_path(Some(&form.return_to))) {
        return Ok(Redirect::to(&back).into_response());
    }
    list_or_redirect(&state, &headers, Some(&submitted.job_id), "Podcast task started").await
}

// ---------------------------------------------------------------------------
// Read a script
// ---------------------------------------------------------------------------

fn script_form(notebooks: &[NotebookSummary], speaker_profiles: &[SpeakerProfileView], in_dialog: bool) -> Markup {
    let body = html! {
        p.muted { "Paste a script or drop in a .txt / .md file. It is read exactly as written by the voices in the speaker profile you pick. No AI rewriting." }
        div.cols-2 {
            div.field {
                label for="pod-script-profile" { "Speaker Profile" }
                select #pod-script-profile name="speaker_profile" required {
                    option value="" { "Choose a voice profile" }
                    @for profile in speaker_profiles {
                        @let names: Vec<&str> = profile.speakers.iter().filter_map(|s| s.get("name").and_then(Json::as_str)).collect();
                        option value=(profile.id) disabled[profile.voice_model.is_none()] {
                            (profile.name) " (" (names.join(", ")) ")"
                            @if profile.voice_model.is_none() { " - no voice model" }
                        }
                    }
                }
            }
            div.field {
                label for="pod-script-name" { "Episode name" }
                input #pod-script-name type="text" name="episode_name" autocomplete="off" placeholder="e.g., AI and the Future of Work";
            }
        }
        div.field.pod-drop data-pod-drop {
            div.row-between {
                label for="pod-script-text" { "Script" }
                label.btn.btn-sm.pod-file {
                    "Choose file"
                    input type="file" name="file" accept=".txt,.md,.markdown,text/plain,text/markdown" data-pod-script-file;
                }
            }
            textarea #pod-script-text .pod-script name="script" rows="12"
                placeholder="Paste your script here, or drop a .txt or .md file.\n\nStart a line with a speaker's name to switch voice, e.g.\nMax: Welcome back to the show." {}
            div.row-between.small.faint {
                span data-pod-filename { "Each paragraph is one clip. \"Name:\" at the start of a line switches to that speaker; otherwise the first voice reads." }
                span data-pod-chars data-max=(MAX_SCRIPT_CHARS) { "0 characters" }
            }
        }
        div.field {
            label for="pod-script-notebook" { "File under notebook (optional)" }
            select #pod-script-notebook name="notebook_id" { (notebook_options(notebooks, "No notebook", None)) }
        }
    };
    if in_dialog {
        html! {
            form method="post" action="/podcasts/read-script" enctype="multipart/form-data"
                hx-post="/podcasts/read-script" hx-encoding="multipart/form-data" hx-target="#pod-episodes" hx-swap="outerHTML"
                data-close-on-success data-reset {
                div.dialog-head {
                    h2 { "Read a script" }
                    button.btn.btn-ghost.btn-sm type="button" data-close { "Close" }
                }
                div.dialog-body { (body) }
                div.dialog-foot {
                    button.btn type="button" data-close { "Cancel" }
                    button.btn.btn-primary type="submit" { span.spinner.htmx-indicator {} "Read script" }
                }
            }
        }
    } else {
        html! {
            form.card.stack method="post" action="/podcasts/read-script" enctype="multipart/form-data" {
                (body)
                div.row {
                    button.btn.btn-primary type="submit" { "Read script" }
                    a.btn href="/podcasts" { "Cancel" }
                }
            }
        }
    }
}

async fn read_script_page(State(state): State<AppState>) -> WebResult<Markup> {
    let notebooks = active_notebooks(&state).await?;
    let speaker_profiles = podcasts::list_speaker_profiles(&state).await?;
    let ctx = PageContext::new(&state, "/podcasts/read-script");
    Ok(page(&ctx, "Read a script", html! {
        div.crumbs { a href="/podcasts" { "Podcasts" } " / Read a script" }
        (page_head("Read a script", None, html! {}))
        (script_form(&notebooks, &speaker_profiles, false))
    }))
}

fn is_script_file(name: Option<&str>, content_type: Option<&str>) -> bool {
    let name = name.unwrap_or("").to_ascii_lowercase();
    [".txt", ".md", ".markdown"].iter().any(|ext| name.ends_with(ext))
        || matches!(content_type, Some("text/plain" | "text/markdown"))
}

async fn read_script(State(state): State<AppState>, headers: HeaderMap, mut multipart: Multipart) -> WebResult<Response> {
    let mut fields: HashMap<String, String> = HashMap::new();
    let mut file: Option<(String, String)> = None;
    while let Some(field) = multipart.next_field().await.map_err(multipart_error)? {
        let name = field.name().unwrap_or_default().to_string();
        if name == "file" {
            let file_name = field.file_name().map(String::from);
            let content_type = field.content_type().map(String::from);
            let data = field.bytes().await.map_err(multipart_error)?;
            if data.is_empty() && file_name.as_deref().unwrap_or("").is_empty() {
                continue;
            }
            if !is_script_file(file_name.as_deref(), content_type.as_deref()) {
                return Err(WebError(AppError::invalid("Only .txt and .md script files are supported")));
            }
            let text = String::from_utf8(data.to_vec())
                .map_err(|_| AppError::invalid("The script file is not UTF-8 text"))?;
            file = Some((file_name.unwrap_or_default(), text));
        } else {
            let value = field.text().await.map_err(multipart_error)?;
            fields.insert(name, value);
        }
    }
    let get = |key: &str| fields.get(key).map(String::as_str).unwrap_or("");

    let mut script = get("script").to_string();
    let mut episode_name = non_empty(get("episode_name"));
    if script.trim().is_empty() {
        if let Some((file_name, text)) = file {
            script = text;
            if episode_name.is_none() {
                let lower = file_name.to_ascii_lowercase();
                let stem = [".markdown", ".txt", ".md"]
                    .iter()
                    .find(|ext| lower.ends_with(*ext))
                    .map(|ext| &file_name[..file_name.len() - ext.len()])
                    .unwrap_or(&file_name);
                episode_name = non_empty(stem);
            }
        }
    }
    let speaker_profile = non_empty(get("speaker_profile")).ok_or_else(|| AppError::invalid("Choose a voice profile."))?;
    let episode_name = episode_name.ok_or_else(|| AppError::invalid("Provide a name for the episode."))?;
    if script.trim().is_empty() {
        return Err(WebError(AppError::invalid("Paste a script, or choose a .txt or .md file.")));
    }
    if script.chars().count() > MAX_SCRIPT_CHARS {
        return Err(WebError(AppError::invalid("The script is too long (500,000 characters at most).")));
    }
    let request = ScriptRequest { speaker_profile, episode_name, script, notebook_id: non_empty(get("notebook_id")) };
    let submitted = podcasts::submit_script(&state, &request).await?;
    list_or_redirect(&state, &headers, Some(&submitted.job_id), "Reading the script").await
}

// ---------------------------------------------------------------------------
// Profiles page
// ---------------------------------------------------------------------------

fn speaker_box(speaker: &Map<String, Json>, models: &HashMap<String, String>) -> Markup {
    let field = |key: &str| speaker.get(key).and_then(Json::as_str).unwrap_or("").to_string();
    let voice_model = speaker.get("voice_model").and_then(Json::as_str).filter(|v| !v.is_empty());
    html! {
        div.pod-box.small {
            div.row-between {
                strong { (field("name")) }
                span.row {
                    span.faint { "Voice ID: " (field("voice_id")) }
                    @if let Some(model) = voice_model {
                        span.badge { (models.get(model).map(String::as_str).unwrap_or(model)) }
                    }
                }
            }
            p.muted.pre-wrap { strong { "Backstory:" } " " (field("backstory")) }
            p.muted.pre-wrap { strong { "Personality:" } " " (field("personality")) }
        }
    }
}

fn confirm_form(action: &str, label: &str, class: &str, confirm: Option<&str>, disabled: bool, title: Option<&str>) -> Markup {
    html! {
        form.inline method="post" action=(action) data-pod-confirm=[confirm] {
            button type="submit" class=(class) disabled[disabled] title=[title] { (label) }
        }
    }
}

fn speaker_card(profile: &SpeakerProfileView, usage: usize, models: &HashMap<String, String>) -> Markup {
    let base = format!("/podcasts/profiles/speakers/{}", profile.id);
    let confirm = format!("Delete speaker profile? Deleting “{}” cannot be undone.", profile.name);
    html! {
        div.card.stack-sm {
            div.row-between {
                div {
                    div.row {
                        h3 { (profile.name) }
                        @if speaker_needs_setup(profile) { span.badge.warn { "Setup required" } }
                    }
                    p.small.muted { @if profile.description.is_empty() { "No description provided." } @else { (profile.description) } }
                }
                span.badge {
                    @match &profile.voice_model {
                        Some(model) => (models.get(model).map(String::as_str).unwrap_or(model)),
                        None => "Not configured",
                    }
                }
            }
            div.row {
                @if usage > 0 {
                    span.badge.info { "Used by " (usage) @if usage == 1 { " episode profile" } @else { " episode profiles" } }
                } @else {
                    span.badge { "Unused" }
                }
            }
            @for speaker in &profile.speakers { (speaker_box(speaker, models)) }
            div.row.pod-actions {
                a.btn.btn-sm.btn-ghost href=(base) { "Edit" }
                (confirm_form(&format!("{base}/duplicate"), "Duplicate", "btn btn-sm btn-ghost", None, false, None))
                (confirm_form(&format!("{base}/delete"), "Delete", "btn btn-sm btn-ghost pod-danger", Some(&confirm), usage > 0,
                    (usage > 0).then_some("Remove this speaker from episode profiles before deleting it.")))
            }
        }
    }
}

fn episode_profile_card(
    profile: &EpisodeProfileView,
    speakers: &[SpeakerProfileView],
    models: &HashMap<String, String>,
) -> Markup {
    let base = format!("/podcasts/profiles/episodes/{}", profile.id);
    let confirm = format!(
        "Delete profile? This will remove “{}”. Existing episodes keep their data, but new ones will no longer use this configuration.",
        profile.name
    );
    let model = |id: &Option<String>| match id {
        Some(id) => models.get(id).cloned().unwrap_or_else(|| id.clone()),
        None => "Not configured".into(),
    };
    let speaker = speakers.iter().find(|s| Some(&s.id) == profile.speaker_config.as_ref());
    html! {
        div.card.stack-sm {
            div.row-between {
                div {
                    div.row {
                        h3 { (profile.name) }
                        @if episode_needs_setup(profile) { span.badge.warn { "Setup required" } }
                    }
                    p.small.muted { @if profile.description.is_empty() { "No description provided." } @else { (profile.description) } }
                }
                div.row.pod-actions {
                    a.btn.btn-sm.btn-ghost href=(base) { "Edit" }
                    (confirm_form(&format!("{base}/duplicate"), "Duplicate", "btn btn-sm btn-ghost", None, false, None))
                    (confirm_form(&format!("{base}/delete"), "Delete", "btn btn-sm btn-ghost pod-danger", Some(&confirm), false, None))
                }
            }
            div.pod-facts.small {
                div { p.faint { "Outline model" } p { (model(&profile.outline_llm)) } }
                div { p.faint { "Transcript model" } p { (model(&profile.transcript_llm)) } }
                div { p.faint { "Segments" } p { (profile.num_segments) } }
                @if let Some(language) = &profile.language { div { p.faint { "Language" } p { (language) } } }
                div {
                    p.faint { "Speaker Profile" }
                    p.row {
                        (profile.speaker_config_name.as_deref().unwrap_or("Not configured"))
                        @if let Some(voice) = speaker.and_then(|s| s.voice_model.as_ref()) {
                            span.badge { (models.get(voice).map(String::as_str).unwrap_or(voice)) }
                        }
                    }
                }
            }
            @if !profile.default_briefing.trim().is_empty() {
                details {
                    summary.small.muted { "Default briefing" }
                    div.pod-box.small.pre-wrap { (profile.default_briefing) }
                }
            }
        }
    }
}

fn profiles_explainer() -> Markup {
    html! {
        details.card.flat.pod-explainer {
            summary { strong { "How profiles power podcast generation" } }
            div.stack-sm.small.muted {
                p { "Profiles split the podcast workflow into two reusable building blocks. Mix and match them whenever you generate a new episode." }
                strong { "Episode profiles set the format" }
                ul {
                    li { "Outline the number of segments and how the story flows" }
                    li { "Pick the language models used for briefing, outlining, and script writing" }
                    li { "Store default briefings so every episode starts with a consistent tone" }
                }
                strong { "Speaker profiles bring voices to life" }
                ul {
                    li { "Choose the text-to-speech provider and model" }
                    li { "Capture personality, backstory, and pronunciation notes per speaker" }
                    li { "Reuse the same host or guest voices across different episode formats" }
                }
                strong { "Recommended workflow" }
                ol {
                    li { "Create speaker profiles for each voice you need" }
                    li { "Build episode profiles that reference those speakers" }
                    li { "Generate podcasts by selecting the episode profile that fits the story" }
                }
            }
        }
    }
}

async fn profiles_page(State(state): State<AppState>) -> WebResult<Markup> {
    let episode_profiles = podcasts::list_episode_profiles(&state).await?;
    let speaker_profiles = podcasts::list_speaker_profiles(&state).await?;
    let models = model_names(&state).await;
    let needs_setup =
        episode_profiles.iter().any(episode_needs_setup) || speaker_profiles.iter().any(speaker_needs_setup);
    let usage = |id: &str| episode_profiles.iter().filter(|p| p.speaker_config.as_deref() == Some(id)).count();
    let ctx = PageContext::new(&state, "/podcasts/profiles");
    let body = html! {
        (podcasts_head())
        (area_tabs("/podcasts/profiles"))
        div.stack {
            @if needs_setup { (setup_alert()) }
            div {
                h2 { "Profiles workspace" }
                p.small.muted { "Build reusable episode and speaker configurations for fast podcast production." }
            }
            (profiles_explainer())
            div.cols-2.pod-profiles {
                section.stack {
                    div.row-between {
                        div {
                            h2 { "Speaker profiles" }
                            p.small.muted { "Configure voices and personalities for generated episodes." }
                        }
                        a.btn.btn-primary href="/podcasts/profiles/speakers/new" { "Create speaker" }
                    }
                    @if speaker_profiles.is_empty() {
                        (empty_state("No speaker profiles yet", "Create one to make episode profiles available."))
                    }
                    @for profile in &speaker_profiles { (speaker_card(profile, usage(&profile.id), &models)) }
                }
                section.stack {
                    div.row-between {
                        div {
                            h2 { "Episode profiles" }
                            p.small.muted { "Define reusable generation settings for your shows." }
                        }
                        @if speaker_profiles.is_empty() {
                            span.btn.btn-primary aria-disabled="true" { "Create profile" }
                        } @else {
                            a.btn.btn-primary href="/podcasts/profiles/episodes/new" { "Create profile" }
                        }
                    }
                    @if speaker_profiles.is_empty() {
                        div.flash.warn { "Create a speaker profile before adding an episode profile." }
                    }
                    @if episode_profiles.is_empty() {
                        (empty_state("No episode profiles yet", "Create one to kickstart podcast generation."))
                    }
                    @for profile in &episode_profiles { (episode_profile_card(profile, &speaker_profiles, &models)) }
                }
            }
        }
    };
    Ok(page(&ctx, "Podcast profiles", body))
}

fn model_select(id: &str, name: &str, models: &[ModelResponse], current: Option<&str>, placeholder: &str, required: bool) -> Markup {
    html! {
        select id=(id) name=(name) required[required] {
            option value="" { (placeholder) }
            @for model in models {
                option value=(model.id) selected[current == Some(model.id.as_str())] { (model.provider) " / " (model.name) }
            }
            // A configured model that is no longer listed stays selectable.
            @if let Some(current) = current.filter(|c| !c.is_empty() && !models.iter().any(|m| m.id == *c)) {
                option value=(current) selected { (current) }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Speaker profile forms
// ---------------------------------------------------------------------------

fn speaker_form(profile: Option<&SpeakerProfileView>, tts_models: &[ModelResponse]) -> Markup {
    let action = match profile {
        Some(p) => format!("/podcasts/profiles/speakers/{}", p.id),
        None => "/podcasts/profiles/speakers/new".to_string(),
    };
    let speakers: Vec<Map<String, Json>> = profile.map(|p| p.speakers.clone()).unwrap_or_default();
    let filled = speakers.len().max(1);
    let value = |index: usize, key: &str| {
        speakers.get(index).and_then(|s| s.get(key)).and_then(Json::as_str).unwrap_or("").to_string()
    };
    html! {
        form.stack method="post" action=(action) hx-post=(action) {
            div.card.stack {
                div.cols-2 {
                    div.field {
                        label for="pod-sp-name" { "Profile name *" }
                        input #pod-sp-name type="text" name="name" required placeholder="e.g., Tech discussion"
                            value=(profile.map(|p| p.name.as_str()).unwrap_or(""));
                    }
                    div.field {
                        label for="pod-sp-description" { "Description" }
                        textarea #pod-sp-description name="description" rows="3" placeholder="Short summary of when to use this profile" {
                            (profile.map(|p| p.description.as_str()).unwrap_or(""))
                        }
                    }
                }
                div.field {
                    label for="pod-sp-voice-model" { "Voice model *" }
                    (model_select("pod-sp-voice-model", "voice_model", tts_models, profile.and_then(|p| p.voice_model.as_deref()), "Select voice model", true))
                    @if tts_models.is_empty() {
                        p.hint { "No text-to-speech models yet. Add one in " a href="/settings/models" { "Manage → Models" } "." }
                    }
                }
            }
            div.stack-sm {
                h3 { "Speakers" }
                p.small.muted { "Configure between one and four voices for this profile. Leave a speaker blank to leave it out." }
            }
            @for index in 0..MAX_SPEAKERS {
                @let fields = html! {
                    div.cols-2 {
                        div.field {
                            label for=(format!("pod-s{index}-name")) { "Name *" }
                            input id=(format!("pod-s{index}-name")) type="text" name=(format!("speaker{index}_name")) autocomplete="off"
                                placeholder=(format!("Host {}", index + 1)) value=(value(index, "name"));
                        }
                        div.field {
                            label for=(format!("pod-s{index}-voice")) { "Voice ID *" }
                            input id=(format!("pod-s{index}-voice")) type="text" name=(format!("speaker{index}_voice_id")) autocomplete="off"
                                placeholder="voice_123" value=(value(index, "voice_id"));
                        }
                    }
                    div.field {
                        label for=(format!("pod-s{index}-backstory")) { "Backstory *" }
                        textarea id=(format!("pod-s{index}-backstory")) name=(format!("speaker{index}_backstory")) rows="2"
                            placeholder="Short biography or context for the speaker" { (value(index, "backstory")) }
                    }
                    div.field {
                        label for=(format!("pod-s{index}-personality")) { "Personality *" }
                        textarea id=(format!("pod-s{index}-personality")) name=(format!("speaker{index}_personality")) rows="2"
                            placeholder="Describe style and tone" { (value(index, "personality")) }
                    }
                    div.field {
                        label for=(format!("pod-s{index}-model")) { "Per-speaker TTS override (optional)" }
                        @let current = value(index, "voice_model");
                        (model_select(&format!("pod-s{index}-model"), &format!("speaker{index}_voice_model"), tts_models,
                            (!current.is_empty()).then_some(current.as_str()), "Use profile default", false))
                    }
                };
                @if index < filled {
                    div.card.stack-sm { strong { "Speaker " (index + 1) } (fields) }
                } @else {
                    details.card.flat.stack-sm {
                        summary { strong { "Add speaker " (index + 1) } }
                        (fields)
                    }
                }
            }
            div.row {
                button.btn.btn-primary type="submit" { @if profile.is_some() { "Save changes" } @else { "Create profile" } }
                a.btn href="/podcasts/profiles" { "Cancel" }
            }
        }
    }
}

/// The speaker profile a form describes. Speakers left blank are dropped;
/// keys the form does not show are kept from `existing`.
fn speaker_input(fields: &Fields, existing: Option<&SpeakerProfileView>) -> Result<SpeakerProfileInput, AppError> {
    let name = fields.required("name", "Name is required")?;
    let voice_model = fields.required("voice_model", "Voice model is required")?;
    let mut speakers = Vec::new();
    for index in 0..MAX_SPEAKERS {
        let key = |field: &str| format!("speaker{index}_{field}");
        let values: Vec<Option<String>> =
            ["name", "voice_id", "backstory", "personality"].iter().map(|f| fields.text(&key(f))).collect();
        if values.iter().all(Option::is_none) {
            continue;
        }
        let number = index + 1;
        let messages = ["Name is required", "Voice ID is required", "Backstory is required", "Personality is required"];
        let mut speaker = existing.and_then(|p| p.speakers.get(index)).cloned().unwrap_or_default();
        for ((field, value), message) in ["name", "voice_id", "backstory", "personality"].iter().zip(values).zip(messages) {
            let value = value.ok_or_else(|| AppError::invalid(format!("Speaker {number}: {message}")))?;
            speaker.insert(field.to_string(), Json::String(value));
        }
        match fields.text(&key("voice_model")) {
            Some(model) => speaker.insert("voice_model".into(), Json::String(model)),
            None => speaker.remove("voice_model"),
        };
        speakers.push(speaker);
    }
    if speakers.is_empty() {
        return Err(AppError::invalid("At least one speaker is required"));
    }
    Ok(SpeakerProfileInput {
        name,
        description: Some(fields.string("description").trim().to_string()),
        voice_model: Some(Some(voice_model)),
        speakers,
    })
}

fn profile_page(state: &AppState, path: &str, title: &str, description: &str, form: Markup) -> Markup {
    let ctx = PageContext::new(state, path);
    page(&ctx, title, html! {
        div.crumbs { a href="/podcasts" { "Podcasts" } " / " a href="/podcasts/profiles" { "Profiles" } " / " (title) }
        (page_head(title, Some(description), html! {}))
        (form)
    })
}

const SPEAKER_FORM_DESC: &str = "Configure text-to-speech settings and define up to four speakers.";
const EPISODE_FORM_DESC: &str =
    "Define how episodes should be generated and which speaker configuration they use by default.";

async fn new_speaker_page(State(state): State<AppState>) -> WebResult<Markup> {
    let models = list_models(&state, Some("text_to_speech")).await?;
    let form = speaker_form(None, &models);
    Ok(profile_page(&state, "/podcasts/profiles/speakers/new", "Create Speaker Profile", SPEAKER_FORM_DESC, form))
}

async fn edit_speaker_page(State(state): State<AppState>, Path(id): Path<String>) -> WebResult<Markup> {
    let profile = SpeakerProfileView::from(podcasts::load_speaker_profile(&state, &id).await?);
    let models = list_models(&state, Some("text_to_speech")).await?;
    let form = speaker_form(Some(&profile), &models);
    let path = format!("/podcasts/profiles/speakers/{}", profile.id);
    Ok(profile_page(&state, &path, "Edit Speaker Profile", SPEAKER_FORM_DESC, form))
}

async fn create_speaker(
    State(state): State<AppState>,
    headers: HeaderMap,
    Form(fields): Form<Vec<(String, String)>>,
) -> WebResult<Response> {
    let input = speaker_input(&Fields(fields), None)?;
    podcasts::create_speaker_profile(&state, &input).await?;
    Ok(redirect(&headers, "/podcasts/profiles"))
}

async fn update_speaker(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
    Form(fields): Form<Vec<(String, String)>>,
) -> WebResult<Response> {
    let existing = SpeakerProfileView::from(podcasts::load_speaker_profile(&state, &id).await?);
    let input = speaker_input(&Fields(fields), Some(&existing))?;
    podcasts::update_speaker_profile(&state, &id, &input).await?;
    Ok(redirect(&headers, "/podcasts/profiles"))
}

async fn duplicate_speaker(State(state): State<AppState>, Path(id): Path<String>, headers: HeaderMap) -> WebResult<Response> {
    podcasts::duplicate_speaker_profile(&state, &id).await?;
    Ok(redirect(&headers, "/podcasts/profiles"))
}

async fn delete_speaker(State(state): State<AppState>, Path(id): Path<String>, headers: HeaderMap) -> WebResult<Response> {
    let profile = podcasts::load_speaker_profile(&state, &id).await?;
    let in_use = podcasts::list_episode_profiles(&state).await?.iter().any(|p| p.speaker_config.as_deref() == Some(&profile.id));
    if in_use {
        return Err(WebError(AppError::invalid("Remove this speaker from episode profiles before deleting it.")));
    }
    podcasts::delete_speaker_profile(&state, &id).await?;
    Ok(redirect(&headers, "/podcasts/profiles"))
}

// ---------------------------------------------------------------------------
// Episode profile forms
// ---------------------------------------------------------------------------

/// What the episode profile form shows, from a stored profile or blank.
#[derive(Default)]
struct EpisodeProfileValues {
    name: String,
    description: String,
    speaker_config: Option<String>,
    outline_llm: Option<String>,
    transcript_llm: Option<String>,
    language: Option<String>,
    default_briefing: String,
    num_segments: i64,
    max_tokens: Option<i64>,
}

fn episode_profile_form(
    id: Option<&str>,
    values: &EpisodeProfileValues,
    speakers: &[SpeakerProfileView],
    llms: &[ModelResponse],
) -> Markup {
    let action = match id {
        Some(id) => format!("/podcasts/profiles/episodes/{id}"),
        None => "/podcasts/profiles/episodes/new".to_string(),
    };
    html! {
        form.card.stack method="post" action=(action) hx-post=(action) {
            @if speakers.is_empty() {
                div.flash.warn {
                    strong { "No speaker profiles available" }
                    " Create a speaker profile before configuring an episode profile."
                }
            }
            div.cols-2 {
                div.field {
                    label for="pod-ep-name" { "Profile name *" }
                    input #pod-ep-name type="text" name="name" required placeholder="e.g., Tech discussion" value=(values.name);
                }
                div.row {
                    div.field.grow {
                        label for="pod-ep-segments" { "Segments *" }
                        input #pod-ep-segments type="number" name="num_segments" min="3" max="20" step="1" required value=(values.num_segments);
                    }
                    div.field.grow {
                        label for="pod-ep-max-tokens" { "Max output tokens" }
                        input #pod-ep-max-tokens type="number" name="max_tokens" min="1" step="1" placeholder="Leave blank for defaults"
                            value=[values.max_tokens];
                    }
                }
            }
            p.hint { "Max output tokens is optional. It caps how many tokens each generation step can produce (the outline and the full transcript). Leave it blank to use the built-in default (8192 for both)." }
            div.field {
                label for="pod-ep-description" { "Description" }
                textarea #pod-ep-description name="description" rows="2" placeholder="Short summary of when to use this profile" { (values.description) }
            }
            div.field {
                label for="pod-ep-speaker" { "Speaker Profile *" }
                select #pod-ep-speaker name="speaker_config" required {
                    option value="" { "Select a speaker profile" }
                    @for speaker in speakers {
                        option value=(speaker.id) selected[values.speaker_config.as_deref() == Some(speaker.id.as_str())] { (speaker.name) }
                    }
                }
            }
            div.cols-2 {
                div.field {
                    label for="pod-ep-outline" { "Outline model *" }
                    (model_select("pod-ep-outline", "outline_llm", llms, values.outline_llm.as_deref(), "Select outline model", true))
                }
                div.field {
                    label for="pod-ep-transcript" { "Transcript model *" }
                    (model_select("pod-ep-transcript", "transcript_llm", llms, values.transcript_llm.as_deref(), "Select transcript model", true))
                }
            }
            div.field {
                label for="pod-ep-language" { "Language" }
                select #pod-ep-language name="language" {
                    option value="" { "Select a language (optional)" }
                    @for language in crate::services::settings::languages() {
                        option value=(language.code) selected[values.language.as_deref() == Some(language.code)] { (language.name) }
                    }
                }
            }
            div.field {
                label for="pod-ep-briefing" { "Default briefing *" }
                textarea #pod-ep-briefing name="default_briefing" rows="6" required
                    placeholder="Outline the structure, tone, and goals for this episode format" { (values.default_briefing) }
            }
            div.row {
                button.btn.btn-primary type="submit" disabled[speakers.is_empty()] { @if id.is_some() { "Save changes" } @else { "Create profile" } }
                a.btn href="/podcasts/profiles" { "Cancel" }
            }
        }
    }
}

fn episode_profile_input(fields: &Fields) -> Result<EpisodeProfileInput, AppError> {
    let name = fields.required("name", "Name is required")?;
    let speaker_config = fields.required("speaker_config", "Speaker profile is required")?;
    let outline = fields.required("outline_llm", "Outline model is required")?;
    let transcript = fields.required("transcript_llm", "Transcript model is required")?;
    let default_briefing = fields.required("default_briefing", "Default briefing is required")?;
    let num_segments = match fields.text("num_segments") {
        Some(text) => text.parse::<i64>().map_err(|_| AppError::invalid("Number of segments must be between 3 and 20"))?,
        None => 3,
    };
    let max_tokens = match fields.text("max_tokens") {
        Some(text) => match text.parse::<i64>() {
            Ok(n) if n > 0 => Some(n),
            _ => return Err(AppError::invalid("Max output tokens must be a positive integer")),
        },
        None => None,
    };
    Ok(EpisodeProfileInput {
        name,
        description: Some(fields.string("description").trim().to_string()),
        speaker_config,
        outline_llm: Some(Some(outline)),
        transcript_llm: Some(Some(transcript)),
        language: Some(fields.text("language")),
        default_briefing,
        num_segments: Some(num_segments),
        max_tokens: Some(max_tokens),
    })
}

async fn new_episode_profile_page(State(state): State<AppState>) -> WebResult<Markup> {
    let speakers = podcasts::list_speaker_profiles(&state).await?;
    let llms = list_models(&state, Some("language")).await?;
    let values = EpisodeProfileValues { num_segments: 3, ..Default::default() };
    let form = episode_profile_form(None, &values, &speakers, &llms);
    Ok(profile_page(&state, "/podcasts/profiles/episodes/new", "Create Episode Profile", EPISODE_FORM_DESC, form))
}

async fn edit_episode_profile_page(State(state): State<AppState>, Path(id): Path<String>) -> WebResult<Markup> {
    let profile = podcasts::load_episode_profile(&state, &id).await?;
    let speakers = podcasts::list_speaker_profiles(&state).await?;
    let llms = list_models(&state, Some("language")).await?;
    let values = EpisodeProfileValues {
        name: profile.name,
        description: profile.description.unwrap_or_default(),
        speaker_config: profile.speaker_config,
        outline_llm: profile.outline_llm,
        transcript_llm: profile.transcript_llm,
        language: profile.language,
        default_briefing: profile.default_briefing,
        num_segments: profile.num_segments,
        max_tokens: profile.max_tokens,
    };
    let form = episode_profile_form(Some(&profile.id), &values, &speakers, &llms);
    let path = format!("/podcasts/profiles/episodes/{}", profile.id);
    Ok(profile_page(&state, &path, "Edit Episode Profile", EPISODE_FORM_DESC, form))
}

async fn create_episode_profile(
    State(state): State<AppState>,
    headers: HeaderMap,
    Form(fields): Form<Vec<(String, String)>>,
) -> WebResult<Response> {
    let input = episode_profile_input(&Fields(fields))?;
    podcasts::create_episode_profile(&state, &input).await?;
    Ok(redirect(&headers, "/podcasts/profiles"))
}

async fn update_episode_profile(
    State(state): State<AppState>,
    Path(id): Path<String>,
    headers: HeaderMap,
    Form(fields): Form<Vec<(String, String)>>,
) -> WebResult<Response> {
    let input = episode_profile_input(&Fields(fields))?;
    podcasts::update_episode_profile(&state, &id, &input).await?;
    Ok(redirect(&headers, "/podcasts/profiles"))
}

async fn duplicate_episode_profile(State(state): State<AppState>, Path(id): Path<String>, headers: HeaderMap) -> WebResult<Response> {
    podcasts::duplicate_episode_profile(&state, &id).await?;
    Ok(redirect(&headers, "/podcasts/profiles"))
}

async fn delete_episode_profile(State(state): State<AppState>, Path(id): Path<String>, headers: HeaderMap) -> WebResult<Response> {
    podcasts::delete_episode_profile(&state, &id).await?;
    Ok(redirect(&headers, "/podcasts/profiles"))
}

// ---------------------------------------------------------------------------
// Private feed
// ---------------------------------------------------------------------------

fn feed_url_card(result: Result<String, AppError>) -> Markup {
    html! {
        section #pod-feed-url .card.stack-sm {
            h2 { "Listen in a podcast app" }
            p.small.muted { "Publish episodes, then follow this private feed in Apple Podcasts or any podcast app." }
            @match result {
                Ok(url) => {
                    div.row.pod-feed-row {
                        input.grow.pod-mono type="text" readonly value=(url) aria-label="Podcast feed URL" data-pod-select;
                        button.btn type="button" data-copy=(url) aria-label="Copy feed URL" { "Copy" }
                        form.inline method="post" action="/podcasts/feed/rotate" hx-post="/podcasts/feed/rotate"
                            hx-target="#pod-feed-url" hx-swap="outerHTML"
                            hx-confirm="Rotate the feed link? This creates a new link and stops the current one working. Every podcast app subscribed with the old link will stop updating until you follow the new one." {
                            button.btn type="submit" { "Rotate link" }
                        }
                    }
                    p.small.muted {
                        "In Apple Podcasts open Library, then the More menu, then \"Follow a Show by URL\", and paste this link. "
                        "Anyone who has this link can listen to your published episodes, so keep it private."
                    }
                }
                Err(e) if e.status() == StatusCode::UNPROCESSABLE_ENTITY => {
                    div.flash.warn role="alert" {
                        strong { "HTTPS address needed" }
                        p.small {
                            "Podcast apps need a secure address to reach this feed. Set OPEN_NOTEBOOK_PUBLIC_URL to this server's HTTPS address "
                            "(for example https://your-host.your-tailnet.ts.net:8502) and restart."
                        }
                    }
                }
                Err(e) => {
                    div.flash.bad role="alert" { "Could not load the feed link: " (e.detail()) }
                }
            }
        }
    }
}

fn feed_settings_card(settings: &FeedSettingsView) -> Markup {
    html! {
        section #pod-feed-settings .card.stack {
            div {
                h2 { "About your show" }
                p.small.muted { "How your show appears in podcast apps. Blank fields use the defaults." }
            }
            form.stack-sm method="post" action="/podcasts/feed/settings" hx-post="/podcasts/feed/settings"
                hx-target="#pod-feed-settings" hx-swap="outerHTML" {
                div.field {
                    label for="pod-feed-title" { "Show title" }
                    input #pod-feed-title type="text" name="title" value=(settings.title);
                }
                div.field {
                    label for="pod-feed-author" { "Author" }
                    input #pod-feed-author type="text" name="author" value=(settings.author);
                }
                div.field {
                    label for="pod-feed-description" { "Description" }
                    textarea #pod-feed-description name="description" rows="3" { (settings.description) }
                }
                div.field.pod-narrow {
                    label for="pod-feed-language" { "Language" }
                    input #pod-feed-language type="text" name="language" value=(settings.language);
                    p.hint { "A language code, for example en or pt-BR." }
                }
                div.row { button.btn.btn-primary type="submit" { "Save" } }
            }
            div.stack-sm {
                strong { "Artwork" }
                div.row.pod-artwork-row {
                    @if settings.has_artwork {
                        img.pod-artwork alt="Current show artwork"
                            src=(format!("/api/podcasts/feed-settings/artwork?v={}", encode_query_value(settings.artwork_version.as_deref().unwrap_or(""))));
                    } @else {
                        div.pod-artwork.pod-artwork-empty { "No artwork yet" }
                    }
                    form.stack-sm method="post" action="/podcasts/feed/artwork" enctype="multipart/form-data"
                        hx-post="/podcasts/feed/artwork" hx-encoding="multipart/form-data"
                        hx-target="#pod-feed-settings" hx-swap="outerHTML" {
                        input type="file" name="file" accept="image/png,image/jpeg" required aria-label="Choose artwork image" data-pod-artwork;
                        div.row {
                            button.btn type="submit" { @if settings.has_artwork { "Replace artwork" } @else { "Upload artwork" } }
                        }
                        p.hint { "Square PNG or JPEG, 1400 to 3000 pixels, up to 10 MB." }
                    }
                }
            }
        }
    }
}

async fn feed_page(State(state): State<AppState>, headers: HeaderMap) -> WebResult<Markup> {
    let url = podcast_feed::feed_url(&state, &forwarded_origin(&headers)).await.map(|u| u.feed_url);
    let settings = podcast_feed::feed_settings(&state).await?;
    let ctx = PageContext::new(&state, "/podcasts/feed");
    Ok(page(&ctx, "Podcast feed", html! {
        (podcasts_head())
        (area_tabs("/podcasts/feed"))
        div.stack {
            (feed_url_card(url))
            (feed_settings_card(&settings))
        }
    }))
}

async fn rotate_feed(State(state): State<AppState>, headers: HeaderMap) -> WebResult<Response> {
    let url = podcast_feed::rotate_feed_url(&state, &forwarded_origin(&headers)).await?;
    if is_htmx(&headers) {
        return Ok(with_toast(feed_url_card(Ok(url.feed_url)).into_response(), "Feed link rotated. Follow the new link in your podcast app."));
    }
    Ok(Redirect::to("/podcasts/feed").into_response())
}

#[derive(Debug, Default, Deserialize)]
struct FeedSettingsForm {
    title: Option<String>,
    author: Option<String>,
    description: Option<String>,
    language: Option<String>,
}

async fn save_feed_settings(
    State(state): State<AppState>,
    headers: HeaderMap,
    Form(form): Form<FeedSettingsForm>,
) -> WebResult<Response> {
    let update = FeedSettingsUpdate {
        title: form.title.map(Some),
        author: form.author.map(Some),
        description: form.description.map(Some),
        language: form.language.map(Some),
    };
    let settings = podcast_feed::update_feed_settings(&state, &update).await?;
    if is_htmx(&headers) {
        return Ok(with_toast(feed_settings_card(&settings).into_response(), "Show details saved"));
    }
    Ok(Redirect::to("/podcasts/feed").into_response())
}

async fn upload_artwork(State(state): State<AppState>, headers: HeaderMap, mut multipart: Multipart) -> WebResult<Response> {
    let max = nn_podcast::artwork::MAX_BYTES;
    let mut data: Option<Vec<u8>> = None;
    while let Some(mut field) = multipart.next_field().await.map_err(multipart_error)? {
        if field.name() != Some("file") {
            continue;
        }
        // Keep at most one byte over the limit: enough to know it is too large.
        let mut bytes = Vec::new();
        while let Some(chunk) = field.chunk().await.map_err(multipart_error)? {
            let room = (max + 1).saturating_sub(bytes.len());
            bytes.extend_from_slice(&chunk[..chunk.len().min(room)]);
            if bytes.len() > max {
                break;
            }
        }
        data = Some(bytes);
        break;
    }
    let data = data.filter(|d| !d.is_empty()).ok_or_else(|| AppError::invalid("Choose a PNG or JPEG image to upload."))?;
    let settings = podcast_feed::upload_artwork(&state, data).await?;
    if is_htmx(&headers) {
        return Ok(with_toast(feed_settings_card(&settings).into_response(), "Artwork uploaded"));
    }
    Ok(Redirect::to("/podcasts/feed").into_response())
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{header, HeaderMap, Method, Request, StatusCode};
    use http_body_util::BodyExt;
    use nn_llm::AiError;
    use serde_json::json;

    use crate::ai::AnyModel;
    use crate::api::auth::encode_query_value;
    use crate::api::episode_profiles::tests::create_episode_profile;
    use crate::api::podcast_feed::tests::{episode_with_audio, HOST};
    use crate::api::speaker_profiles::tests::{create_speaker_profile, model};
    use crate::testing::{multipart_body, silent_mp3_frames, MockLanguageModel, MockTextToSpeechModel, TestApp};

    const OUTLINE: &str = r#"{"segments": [
        {"name": "Opening", "description": "Why offline-first apps matter.", "size": "short"},
        {"name": "Middle", "description": "How sync works.", "size": "short"},
        {"name": "Close", "description": "What to remember.", "size": "short"}
    ]}"#;
    const TRANSCRIPT: &str =
        r#"{"transcript": [{"speaker": "Ana", "dialogue": "Welcome to the show."}, {"speaker": "Ben", "dialogue": "Glad to be here."}]}"#;

    async fn read(app: &TestApp, request: Request<Body>) -> (StatusCode, HeaderMap, String) {
        let response = app.send(request).await;
        let status = response.status();
        let headers = response.headers().clone();
        let body = response.into_body().collect().await.unwrap().to_bytes();
        (status, headers, String::from_utf8_lossy(&body).into_owned())
    }

    async fn page(app: &TestApp, path: &str) -> (StatusCode, String) {
        app.text(Method::GET, path, None).await
    }

    fn encode(pairs: &[(&str, &str)]) -> String {
        pairs.iter().map(|(k, v)| format!("{}={}", encode_query_value(k), encode_query_value(v))).collect::<Vec<_>>().join("&")
    }

    async fn form(app: &TestApp, path: &str, pairs: &[(&str, &str)], htmx: bool) -> (StatusCode, HeaderMap, String) {
        let mut builder = Request::builder()
            .method(Method::POST)
            .uri(path)
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded");
        if htmx {
            builder = builder.header("hx-request", "true");
        }
        read(app, builder.body(Body::from(encode(pairs))).unwrap()).await
    }

    async fn multipart(app: &TestApp, path: &str, parts: &[(&str, Option<&str>, &[u8])]) -> (StatusCode, HeaderMap, String) {
        let (content_type, body) = multipart_body(parts);
        let request = Request::builder()
            .method(Method::POST)
            .uri(path)
            .header(header::CONTENT_TYPE, content_type)
            .header("hx-request", "true")
            .body(Body::from(body))
            .unwrap();
        read(app, request).await
    }

    /// `zz_show` / `zz_speakers` with mock models behind them; returns the speaker profile id.
    async fn setup(app: &TestApp) -> String {
        let llm = model(app, "llm1", "language").await;
        let tts = model(app, "tts1", "text_to_speech").await;
        let speaker = create_speaker_profile(app, "zz_speakers", Some(&tts)).await;
        let speaker_id = speaker["id"].as_str().unwrap().to_string();
        create_episode_profile(app, "zz_show", &speaker_id, &llm).await;
        app.state.models.set_override(AnyModel::Language(MockLanguageModel::new([OUTLINE, TRANSCRIPT])));
        app.state.models.set_override(AnyModel::TextToSpeech(MockTextToSpeechModel::new()));
        speaker_id
    }

    /// The job a form just submitted. The list fragment watches it while it is
    /// unfinished; a job that already finished (the mocks are fast, a small
    /// machine is slow) is not watched, and is then the newest job.
    async fn watched_job(app: &TestApp, fragment: &str) -> String {
        if let Some(start) = fragment.find("watch=") {
            let rest = &fragment[start + "watch=".len()..];
            let end = rest.find('"').unwrap();
            return rest[..end].replace("%3A", ":");
        }
        let newest = app.state.jobs.list(None, None, 1).await.unwrap();
        let job = newest.first().expect("a job was submitted");
        assert!(crate::jobs::status::is_terminal(job.status_str()), "an unfinished job is watched: {fragment}");
        job.id.clone()
    }

    async fn episodes(app: &TestApp) -> Vec<crate::services::podcasts::EpisodeView> {
        crate::services::podcasts::list_episodes(&app.state, None).await.unwrap()
    }

    #[tokio::test]
    async fn empty_pages_render() {
        let app = TestApp::new().await;
        let (status, body) = page(&app, "/podcasts").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("No podcast episodes yet"), "{body}");
        assert!(body.contains("Generate Podcast") && body.contains("Read a Script"));
        assert!(body.contains("id=\"pod-generate\"") && body.contains("id=\"pod-script\""));

        let (status, body) = page(&app, "/podcasts/profiles").await;
        assert_eq!(status, StatusCode::OK);
        // The migrations seed default profiles; with no models they need setup.
        assert!(body.contains("Speaker profiles") && body.contains("Episode profiles"), "{body}");

        for path in ["/podcasts/generate", "/podcasts/read-script", "/podcasts/profiles/speakers/new", "/podcasts/profiles/episodes/new"] {
            let (status, _) = page(&app, path).await;
            assert_eq!(status, StatusCode::OK, "{path}");
        }
        let (status, body) = page(&app, "/podcasts/episodes").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.starts_with("<div") && body.contains("id=\"pod-episodes\"") && !body.contains("<html"), "a fragment: {body}");
    }

    #[tokio::test]
    async fn missing_records_are_404_pages() {
        let app = TestApp::new().await;
        for path in [
            "/podcasts/episodes/episode:nope",
            "/podcasts/profiles/speakers/speaker_profile:nope",
            "/podcasts/profiles/episodes/episode_profile:nope",
        ] {
            let (status, body) = page(&app, path).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{path}");
            assert!(body.contains("data-error-detail"), "{body}");
        }
        let (status, _, _) = form(&app, "/podcasts/episodes/episode:nope/delete", &[], true).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        // A polled card that is gone empties itself and refreshes the list.
        let (status, headers, body) = read(&app, Request::builder().uri("/podcasts/episodes/episode:nope/card").body(Body::empty()).unwrap()).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.is_empty());
        assert_eq!(headers["hx-trigger"], "pod-episodes-changed");
    }

    #[tokio::test]
    async fn an_episode_with_audio_plays_publishes_and_deletes() {
        let app = TestApp::new().await;
        let id = episode_with_audio(&app, "web1", "episodes/web1/audio/web1.mp3", &silent_mp3_frames(20), None).await;

        let (status, body) = page(&app, "/podcasts").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("Episode web1") && body.contains("Completed Episodes"), "{body}");
        assert!(body.contains(&format!("src=\"/api/podcasts/episodes/{id}/audio\"")), "{body}");
        assert!(body.contains("Publish"));

        let (status, body) = page(&app, &format!("/podcasts/episodes/{id}")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("Secret generation briefing") && body.contains("<audio"), "{body}");
        let (status, body) = page(&app, &format!("/podcasts/episodes/{id}?tab=outline")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("No outline available."));
        let (_, body) = page(&app, &format!("/podcasts/episodes/{id}?tab=transcript")).await;
        assert!(body.contains("No transcript available."));

        // Publish with htmx: the list comes back with the episode in the feed.
        let (status, headers, body) = form(&app, &format!("/podcasts/episodes/{id}/publish"), &[("summary", "  My notes  ")], true).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("In your feed"), "{body}");
        assert!(headers["hx-trigger"].to_str().unwrap().contains("Published to your feed"));
        let row = app.state.db.select(&id).await.unwrap().unwrap();
        assert!(row["published_at"].is_string());
        assert_eq!(row["feed_summary"], "My notes");

        // Unpublish as a plain form: a redirect.
        let (status, headers, _) = form(&app, &format!("/podcasts/episodes/{id}/unpublish"), &[], false).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(headers[header::LOCATION], format!("/podcasts/episodes/{id}").as_str());
        let row = app.state.db.select(&id).await.unwrap().unwrap();
        assert!(row.get("published_at").is_none_or(|v| v.is_null()), "{row}");

        // A completed episode cannot be retried.
        let (status, _, body) = form(&app, &format!("/podcasts/episodes/{id}/retry"), &[], true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("not in a failed state"), "{body}");

        let (status, _, body) = form(&app, &format!("/podcasts/episodes/{id}/delete"), &[], true).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("No podcast episodes yet"), "{body}");
        assert!(app.state.db.select(&id).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn a_notebook_page_lists_its_podcasts() {
        let app = TestApp::new().await;
        let (_, nb) = app.post("/api/notebooks", json!({ "name": "Pods" })).await;
        let nb = nb["id"].as_str().unwrap().to_string();

        // Empty: a hint and a Generate link with this notebook chosen.
        let (status, body) = page(&app, &format!("/notebooks/{nb}")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("No podcasts for this notebook yet."), "{body}");
        let generate = format!("/podcasts/generate?notebook_id={}", encode_query_value(&nb));
        assert!(body.contains(&generate.replace('&', "&amp;")), "{body}");

        // The Generate page comes with the notebook selected.
        let (status, body) = page(&app, &format!("{generate}&return_to=%2Fnotebooks%2F{}", encode_query_value(&nb))).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains(&format!(r#"value="{nb}" selected"#)), "{body}");
        assert!(body.contains(r#"name="return_to""#), "{body}");

        // An episode filed under the notebook shows with its player; others do not.
        let id = episode_with_audio(&app, "nb1", "episodes/nb1/audio/nb1.mp3", &silent_mp3_frames(5), None).await;
        app.state.db.update(&id, crate::db::Doc::new().record("notebook", &nb)).await.unwrap();
        episode_with_audio(&app, "other", "episodes/other/audio/other.mp3", &silent_mp3_frames(5), None).await;
        let (status, body) = page(&app, &format!("/notebooks/{nb}/podcasts")).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("Episode nb1") && !body.contains("Episode other"), "{body}");
        assert!(body.contains(&format!("src=\"/api/podcasts/episodes/{id}/audio\"")), "{body}");
        assert!(!body.contains("hx-trigger"), "nothing generating, so no polling: {body}");
    }

    #[tokio::test]
    async fn generating_from_a_notebook_page_returns_there() {
        let app = TestApp::new().await;
        setup(&app).await;
        let (_, nb) = app.post("/api/notebooks", json!({ "name": "Pods" })).await;
        let nb = nb["id"].as_str().unwrap().to_string();
        let back = format!("/notebooks/{nb}");
        let (status, headers, _) = form(
            &app,
            "/podcasts/generate",
            &[("content", "Some content"), ("episode_profile", "zz_show"), ("episode_name", "From the page"), ("return_to", &back)],
            false,
        )
        .await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(headers[header::LOCATION], back.as_str());

        // Only same-site paths are followed.
        let (status, headers, _) = form(
            &app,
            "/podcasts/generate",
            &[("content", "Some content"), ("episode_profile", "zz_show"), ("episode_name", "Again"), ("return_to", "//evil.example")],
            false,
        )
        .await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(headers[header::LOCATION], "/podcasts");
    }

    #[test]
    fn default_summary_matches_the_server() {
        let mut episode: crate::services::podcasts::EpisodeView = serde_json::from_value(json!({
            "id": "episode:x", "name": "Short name", "episode_profile": {}, "speaker_profile": {}, "briefing": "",
            "audio_file": null, "audio_url": null, "transcript": null, "outline": null, "created": null,
            "job_status": "completed", "error_message": null, "notebook_id": null, "progress": null,
            "published_at": null, "duration_seconds": null, "feed_summary": null
        }))
        .unwrap();
        assert_eq!(super::default_summary(&episode), "Short name");
        episode.outline = Some(json!({ "segments": [{ "description": "One\n two" }, { "description": "three" }] }));
        assert_eq!(super::default_summary(&episode), "One two three");
        let long = "word ".repeat(100);
        episode.outline = Some(json!({ "segments": [{ "description": long }] }));
        let summary = super::default_summary(&episode);
        assert!(summary.ends_with("word…") && summary.chars().count() <= 300, "{summary}");
    }

    #[tokio::test]
    async fn generating_from_content_and_from_a_notebook() {
        let app = TestApp::new().await;
        setup(&app).await;

        // Validation happens before anything is queued.
        let (status, _, body) = form(&app, "/podcasts/generate", &[("episode_profile", "zz_show"), ("episode_name", "X")], true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("Choose a notebook"), "{body}");
        let (status, _, body) =
            form(&app, "/podcasts/generate", &[("episode_profile", "nope"), ("episode_name", "X"), ("content", "c")], true).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body.contains("Episode profile &#39;nope&#39; not found") || body.contains("Episode profile 'nope' not found"), "{body}");

        // Content pasted in, speaker taken from the episode profile.
        let (status, headers, body) = form(
            &app,
            "/podcasts/generate",
            &[
                ("notebook_id", ""),
                ("content", "Offline-first apps keep working without a network."),
                ("episode_profile", "zz_show"),
                ("speaker_profile", ""),
                ("episode_name", "Offline first"),
                ("instructions", "Keep it short."),
            ],
            true,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(headers["hx-trigger"].to_str().unwrap().contains("Podcast task started"));
        let job = watched_job(&app, &body).await;
        assert_eq!(app.wait_for_job(&job).await, "completed");
        let list = episodes(&app).await;
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].name, "Offline first");
        assert!(list[0].briefing.contains("Keep it short."));

        // Once finished, the list stops watching.
        let (_, body) = page(&app, &format!("/podcasts/episodes?watch={}", encode_query_value(&job))).await;
        assert!(!body.contains("every 3s"), "{body}");
        assert!(body.contains("Offline first"));

        // From a notebook: the episode is made from its notes and filed under it.
        let notebook = crate::services::notebooks::create_notebook(
            &app.state,
            &crate::services::notebooks::NewNotebook { name: "Sync".into(), description: String::new() },
        )
        .await
        .unwrap();
        let (status, note) = app
            .post("/api/notes", json!({ "title": "Sync notes", "content": "CRDTs merge edits.", "notebook_id": notebook.id }))
            .await;
        assert_eq!(status, StatusCode::OK, "{note}");
        app.state.models.set_override(AnyModel::Language(MockLanguageModel::new([OUTLINE, TRANSCRIPT])));
        let (status, _, body) = form(
            &app,
            "/podcasts/generate",
            &[("notebook_id", &notebook.id), ("episode_profile", "zz_show"), ("episode_name", "From notes")],
            true,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(app.wait_for_job(&watched_job(&app, &body).await).await, "completed");
        let filed = crate::services::podcasts::list_episodes(&app.state, Some(&notebook.id)).await.unwrap();
        assert_eq!(filed.len(), 1);
        assert_eq!(filed[0].name, "From notes");

        // As a plain form the answer is a redirect.
        app.state.models.set_override(AnyModel::Language(MockLanguageModel::new([OUTLINE, TRANSCRIPT])));
        let (status, headers, _) =
            form(&app, "/podcasts/generate", &[("content", "More."), ("episode_profile", "zz_show"), ("episode_name", "Plain")], false).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(headers[header::LOCATION], "/podcasts");
    }

    #[tokio::test]
    async fn reading_a_script_pasted_or_from_a_file() {
        let app = TestApp::new().await;
        let speaker = setup(&app).await;

        let (status, headers, body) = multipart(
            &app,
            "/podcasts/read-script",
            &[
                ("speaker_profile", None, speaker.as_bytes()),
                ("episode_name", None, b"Read aloud"),
                ("script", None, b"Ana: Hello there.\n\nBen: Hi, Ana."),
                ("notebook_id", None, b""),
                ("file", Some(""), b""),
            ],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(headers["hx-trigger"].to_str().unwrap().contains("Reading the script"));
        assert_eq!(app.wait_for_job(&watched_job(&app, &body).await).await, "completed");

        // An empty textarea falls back to the file, and the file names the episode.
        let (status, _, body) = multipart(
            &app,
            "/podcasts/read-script",
            &[("speaker_profile", None, speaker.as_bytes()), ("script", None, b""), ("file", Some("Pilot.md"), b"Ana: From a file.")],
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(app.wait_for_job(&watched_job(&app, &body).await).await, "completed");
        let names: Vec<String> = episodes(&app).await.into_iter().map(|e| e.name).collect();
        assert!(names.contains(&"Read aloud".to_string()) && names.contains(&"Pilot".to_string()), "{names:?}");
        let (_, body) = page(&app, "/podcasts").await;
        assert!(body.contains("Script reading"), "{body}");

        // Refusals.
        let (status, _, body) = multipart(
            &app,
            "/podcasts/read-script",
            &[("speaker_profile", None, speaker.as_bytes()), ("episode_name", None, b"X"), ("file", Some("cover.png"), b"\x89PNG")],
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("Only .txt and .md script files are supported"), "{body}");
        let (status, _, body) = multipart(
            &app,
            "/podcasts/read-script",
            &[("speaker_profile", None, speaker.as_bytes()), ("episode_name", None, b"X"), ("script", None, b"   ")],
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("Paste a script"), "{body}");
        assert_eq!(episodes(&app).await.len(), 2, "nothing else was queued");
    }

    #[tokio::test]
    async fn a_failed_episode_shows_its_error_and_retries() {
        let app = TestApp::new().await;
        setup(&app).await;
        app.state.models.set_override(AnyModel::Language(MockLanguageModel::failing(AiError::Authentication("Invalid API key".into()))));
        let (status, _, body) = form(
            &app,
            "/podcasts/generate",
            &[("content", "Some content"), ("episode_profile", "zz_show"), ("episode_name", "Doomed")],
            true,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(app.wait_for_job(&watched_job(&app, &body).await).await, "failed");
        let failed = episodes(&app).await.remove(0);

        let (_, body) = page(&app, "/podcasts").await;
        assert!(body.contains("Failed Episodes") && body.contains("Error details") && body.contains("Retry"), "{body}");
        // A finished card tells the list to refresh.
        let (_, headers, body) =
            read(&app, Request::builder().uri(format!("/podcasts/episodes/{}/card", failed.id)).body(Body::empty()).unwrap()).await;
        assert_eq!(headers["hx-trigger"], "pod-episodes-changed");
        assert!(!body.contains("every 3s"));

        app.state.models.set_override(AnyModel::Language(MockLanguageModel::new([OUTLINE, TRANSCRIPT])));
        let (status, headers, body) = form(&app, &format!("/podcasts/episodes/{}/retry", failed.id), &[], true).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(headers["hx-trigger"].to_str().unwrap().contains("submitted"));
        assert_eq!(app.wait_for_job(&watched_job(&app, &body).await).await, "completed");
        let list = episodes(&app).await;
        assert_eq!(list.len(), 1);
        assert_eq!(list[0].job_status.as_deref(), Some("completed"));
        assert_ne!(list[0].id, failed.id);
    }

    #[tokio::test]
    async fn speaker_profiles_are_created_edited_duplicated_and_deleted() {
        let app = TestApp::new().await;
        let tts = model(&app, "tts1", "text_to_speech").await;
        let (_, body) = page(&app, "/podcasts/profiles/speakers/new").await;
        assert!(body.contains("tts1"), "the voice model is offered: {body}");

        // No speakers → refused.
        let (status, _, body) = form(&app, "/podcasts/profiles/speakers/new", &[("name", "zz_web"), ("voice_model", &tts)], true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("At least one speaker is required"), "{body}");
        // A half-filled speaker → refused with its number.
        let (status, _, body) = form(
            &app,
            "/podcasts/profiles/speakers/new",
            &[("name", "zz_web"), ("voice_model", &tts), ("speaker0_name", "Ana")],
            true,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("Speaker 1: Voice ID is required"), "{body}");

        let speaker = |n: &str, name: &str, voice: &str| {
            vec![
                (format!("speaker{n}_name"), name.to_string()),
                (format!("speaker{n}_voice_id"), voice.to_string()),
                (format!("speaker{n}_backstory"), "A host.".to_string()),
                (format!("speaker{n}_personality"), "Warm.".to_string()),
                (format!("speaker{n}_voice_model"), String::new()),
            ]
        };
        let mut pairs: Vec<(String, String)> =
            vec![("name".into(), "zz_web".into()), ("description".into(), "Two hosts".into()), ("voice_model".into(), tts.clone())];
        pairs.extend(speaker("0", "Ana", "alloy"));
        pairs.extend(speaker("1", "Ben", "echo"));
        pairs.extend(["speaker2_name", "speaker2_voice_id", "speaker2_backstory", "speaker2_personality"].map(|k| (k.to_string(), String::new())));
        let borrowed: Vec<(&str, &str)> = pairs.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        let (status, headers, body) = form(&app, "/podcasts/profiles/speakers/new", &borrowed, true).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(headers["hx-redirect"], "/podcasts/profiles");

        let created = crate::services::podcasts::get_speaker_profile(&app.state, "zz_web").await.unwrap();
        assert_eq!(created.speakers.len(), 2);
        assert_eq!(created.description, "Two hosts");
        assert_eq!(created.voice_model.as_deref(), Some(tts.as_str()));
        assert!(created.speakers[0].get("voice_model").is_none());

        let (status, body) = page(&app, "/podcasts/profiles").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("zz_web") && body.contains("Voice ID: echo") && body.contains("openai / tts1"), "{body}");
        let (status, body) = page(&app, &format!("/podcasts/profiles/speakers/{}", created.id)).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("value=\"Ben\""), "{body}");

        // Edit as a plain form: rename and drop Ben.
        let mut pairs: Vec<(String, String)> = vec![("name".into(), "zz_web2".into()), ("voice_model".into(), tts.clone())];
        pairs.extend(speaker("0", "Ana", "nova"));
        let borrowed: Vec<(&str, &str)> = pairs.iter().map(|(k, v)| (k.as_str(), v.as_str())).collect();
        let (status, headers, _) = form(&app, &format!("/podcasts/profiles/speakers/{}", created.id), &borrowed, false).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(headers[header::LOCATION], "/podcasts/profiles");
        let edited = crate::services::podcasts::get_speaker_profile(&app.state, "zz_web2").await.unwrap();
        assert_eq!(edited.speakers.len(), 1);
        assert_eq!(edited.speakers[0]["voice_id"], "nova");

        let (status, _, _) = form(&app, &format!("/podcasts/profiles/speakers/{}/duplicate", created.id), &[], false).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        let copy = crate::services::podcasts::get_speaker_profile(&app.state, "zz_web2 - Copy").await.unwrap();

        // A speaker profile in use cannot be deleted.
        let llm = model(&app, "llm1", "language").await;
        create_episode_profile(&app, "zz_uses_copy", &copy.id, &llm).await;
        let (status, _, body) = form(&app, &format!("/podcasts/profiles/speakers/{}/delete", copy.id), &[], true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("Remove this speaker from episode profiles"), "{body}");

        let (status, _, _) = form(&app, &format!("/podcasts/profiles/speakers/{}/delete", created.id), &[], false).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert!(crate::services::podcasts::get_speaker_profile(&app.state, "zz_web2").await.is_err());
    }

    #[tokio::test]
    async fn episode_profiles_are_created_edited_duplicated_and_deleted() {
        let app = TestApp::new().await;
        let llm = model(&app, "llm1", "language").await;
        let tts = model(&app, "tts1", "text_to_speech").await;
        let speaker = create_speaker_profile(&app, "zz_voices", Some(&tts)).await;
        let speaker_id = speaker["id"].as_str().unwrap();

        let (_, body) = page(&app, "/podcasts/profiles/episodes/new").await;
        assert!(body.contains("zz_voices") && body.contains("openai / llm1"), "{body}");

        let base = |name: &'static str, segments: &'static str| {
            vec![
                ("name", name),
                ("description", "Weekly show"),
                ("speaker_config", speaker_id),
                ("outline_llm", llm.as_str()),
                ("transcript_llm", llm.as_str()),
                ("language", "pt-BR"),
                ("default_briefing", "Discuss it."),
                ("num_segments", segments),
                ("max_tokens", ""),
            ]
        };
        let (status, _, body) = form(&app, "/podcasts/profiles/episodes/new", &base("zz_weekly", "2"), true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("between 3 and 20"), "{body}");

        let (status, headers, body) = form(&app, "/podcasts/profiles/episodes/new", &base("zz_weekly", "4"), true).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(headers["hx-redirect"], "/podcasts/profiles");
        let created = crate::services::podcasts::get_episode_profile(&app.state, "zz_weekly").await.unwrap();
        assert_eq!(created.num_segments, 4);
        assert_eq!(created.language.as_deref(), Some("pt-BR"));
        assert_eq!(created.max_tokens, None);
        assert_eq!(created.speaker_config_name.as_deref(), Some("zz_voices"));

        let (status, body) = page(&app, "/podcasts/profiles").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("zz_weekly") && body.contains("Used by 1 episode profile"), "{body}");
        let (status, body) = page(&app, &format!("/podcasts/profiles/episodes/{}", created.id)).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("value=\"zz_weekly\"") && body.contains("Discuss it."), "{body}");

        let mut edit = base("zz_weekly", "5");
        edit[5] = ("language", "");
        edit[8] = ("max_tokens", "4096");
        let (status, _, _) = form(&app, &format!("/podcasts/profiles/episodes/{}", created.id), &edit, false).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        let edited = crate::services::podcasts::get_episode_profile(&app.state, "zz_weekly").await.unwrap();
        assert_eq!((edited.num_segments, edited.max_tokens, edited.language), (5, Some(4096), None));

        let (status, _, _) = form(&app, &format!("/podcasts/profiles/episodes/{}/duplicate", created.id), &[], true).await;
        assert_eq!(status, StatusCode::OK);
        assert!(crate::services::podcasts::get_episode_profile(&app.state, "zz_weekly - Copy").await.is_ok());

        let (status, _, _) = form(&app, &format!("/podcasts/profiles/episodes/{}/delete", created.id), &[], false).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert!(crate::services::podcasts::get_episode_profile(&app.state, "zz_weekly").await.is_err());
    }

    /// A structurally valid JPEG header of the given size (what the artwork check reads).
    fn jpeg(size: u16) -> Vec<u8> {
        let mut out = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        out.extend(b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0");
        out.extend([0xFF, 0xDB, 0x00, 0x43, 0x00]);
        out.extend([1u8; 64]);
        out.extend([0xFF, 0xC0, 0x00, 0x11, 0x08]);
        out.extend(size.to_be_bytes());
        out.extend(size.to_be_bytes());
        out.extend([0x03, 0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01]);
        out.extend([0xFF, 0xDA, 0x00, 0x0C, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3F, 0x00]);
        out.extend([7u8; 40]);
        out.extend([0xFF, 0xD9]);
        out
    }

    #[tokio::test]
    async fn the_private_feed_page() {
        let app = TestApp::new().await;
        // No HTTPS address: the page explains what to set.
        let (status, body) = page(&app, "/podcasts/feed").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("HTTPS address needed") && body.contains("About your show"), "{body}");

        let forwarded = |method: Method, path: &str, htmx: bool| {
            let mut builder =
                Request::builder().method(method).uri(path).header("x-forwarded-proto", "https").header("x-forwarded-host", HOST);
            if htmx {
                builder = builder.header("hx-request", "true");
            }
            builder.body(Body::empty()).unwrap()
        };
        let token = crate::services::podcast_feed::ensure_feed_token(&app.state).await.unwrap();
        let (status, _, body) = read(&app, forwarded(Method::GET, "/podcasts/feed", false)).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains(&token) && body.contains("Rotate link") && body.contains("data-copy"), "{body}");

        let (status, headers, body) = read(&app, forwarded(Method::POST, "/podcasts/feed/rotate", true)).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(headers["hx-trigger"].to_str().unwrap().contains("rotated"));
        let rotated = crate::services::podcast_feed::ensure_feed_token(&app.state).await.unwrap();
        assert_ne!(rotated, token);
        assert!(body.contains(&rotated) && !body.contains(&token));

        // Show details.
        let (status, _, body) = form(
            &app,
            "/podcasts/feed/settings",
            &[("title", "Lab Notes"), ("author", "Dan"), ("description", ""), ("language", "en")],
            true,
        )
        .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("value=\"Lab Notes\""), "{body}");
        let settings = crate::services::podcast_feed::feed_settings(&app.state).await.unwrap();
        assert_eq!((settings.title.as_str(), settings.author.as_str()), ("Lab Notes", "Dan"));
        let (status, _, body) = form(&app, "/podcasts/feed/settings", &[("language", "not a language!")], true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("language code"), "{body}");

        // Artwork.
        let (status, _, _) = multipart(&app, "/podcasts/feed/artwork", &[("file", Some("cover.jpg"), b"not an image")]).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _, body) = multipart(&app, "/podcasts/feed/artwork", &[("file", Some("cover.jpg"), &jpeg(1400))]).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("Replace artwork") && body.contains("/api/podcasts/feed-settings/artwork?v="), "{body}");
        assert!(crate::services::podcast_feed::feed_settings(&app.state).await.unwrap().has_artwork);
    }
}
