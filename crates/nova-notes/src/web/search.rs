//! Ask and Search (`/search`) and Transformations (`/transformations`).
//!
//! - Search runs [`services::search::search`] from a plain GET form, so a
//!   result page has a shareable URL; htmx swaps just the result list.
//! - Ask posts to `/search/ask`, which runs the whole three-stage Ask and
//!   renders every stage (works without JavaScript). With JavaScript,
//!   `static/js/search.js` posts the same form as JSON to
//!   `/search/ask/stream` instead and fills the stages in as the
//!   server-sent events arrive. Each event carries server-rendered,
//!   sanitised HTML, so the page never renders model output itself.
//! - The final answer can be saved as an AI note in one or more notebooks.
//! - Transformations: list, create, edit, delete, the default prompt and a
//!   playground that runs one transformation on pasted text.

use axum::body::Bytes;
use axum::extract::{Path, RawQuery, State};
use axum::http::HeaderMap;
use axum::response::{IntoResponse, Redirect, Response};
use axum::routing::{get, post};
use axum::Router;
use futures::StreamExt;
use maud::{html, Markup};
use once_cell::sync::Lazy;
use regex::Regex;
use serde_json::{json, Value};

use super::{empty_state, is_htmx, markdown, page, page_head, with_toast, Fields, PageContext, WebResult};
use crate::api::source_chat::sse_response;
use crate::api::ApiJson;
use crate::error::AppError;
use crate::services::models::{get_default_models, list_models, ModelResponse};
use crate::services::notebooks::{list_notebooks, NotebookQuery, NotebookSummary};
use crate::services::notes::{create_note, NewNote};
use crate::services::search::{self, AskEvent, AskRequest, AskSearch, NotebookScope, SearchRequest, SearchResponse, SearchType};
use crate::services::transformations::{
    self, DefaultPrompt, ExecuteTransformation, NewTransformation, TransformationUpdate, TransformationView,
};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/search", get(search_page))
        .route("/search/ask", post(ask_submit))
        .route("/search/ask/stream", post(ask_stream))
        .route("/search/save-note", post(save_note))
        .route("/transformations", get(transformations_page).post(create_transformation))
        .route("/transformations/new", get(new_transformation_page))
        .route("/transformations/default-prompt", post(save_default_prompt))
        .route("/transformations/playground", post(run_playground))
        .route("/transformations/{id}", get(edit_transformation_page).post(update_transformation))
        .route("/transformations/{id}/delete", post(delete_transformation))
}

// ---------------------------------------------------------------------------
// Form and query parsing
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Shared pieces
// ---------------------------------------------------------------------------

async fn active_notebooks(state: &AppState) -> WebResult<Vec<NotebookSummary>> {
    Ok(list_notebooks(state, &NotebookQuery { archived: Some(false), ..NotebookQuery::default() }).await?)
}

async fn language_models(state: &AppState) -> WebResult<Vec<ModelResponse>> {
    Ok(list_models(state, Some("language")).await?)
}

/// A checkbox list of notebooks, `name` repeated once per checked box.
fn notebook_checkboxes(name: &str, notebooks: &[NotebookSummary], selected: &[String], empty: &str) -> Markup {
    html! {
        @if notebooks.is_empty() {
            p.muted.small { (empty) }
        } @else {
            div.sr-checklist {
                @for nb in notebooks {
                    label.check {
                        input type="checkbox" name=(name) value=(nb.id) checked[selected.contains(&nb.id)];
                        span.grow {
                            (nb.name)
                            @if !nb.description.is_empty() { span.faint.small.sr-desc { (nb.description) } }
                        }
                    }
                }
            }
        }
    }
}

/// The optional notebook scope shared by Ask and Search.
fn scope_picker(notebooks: &[NotebookSummary], selected: &[String]) -> Markup {
    let summary = if selected.is_empty() { "All notebooks".to_string() } else { format!("{} selected", selected.len()) };
    html! {
        details.sr-scope open[!selected.is_empty()] {
            summary {
                span { "Notebooks" }
                span class=(if selected.is_empty() { "badge" } else { "badge info" }) { (summary) }
            }
            div.stack-sm.sr-scope-body {
                (notebook_checkboxes("notebook_ids", notebooks, selected, "No notebooks yet."))
                p.hint { "Leave all unchecked to search your whole knowledge base." }
            }
        }
    }
}

fn model_select(id: &str, name: &str, label: &str, models: &[ModelResponse], selected: Option<&str>, empty_label: Option<&str>) -> Markup {
    html! {
        div.field {
            label for=(id) { (label) }
            select id=(id) name=(name) required[empty_label.is_none()] {
                @match empty_label {
                    Some(text) => option value="" selected[selected.is_none_or(str::is_empty)] { (text) },
                    None => @if selected.is_none() { option value="" selected disabled { "Select a model" } },
                }
                @for model in models {
                    option value=(model.id) selected[selected == Some(model.id.as_str())] {
                        (model.name) " (" (model.provider) ")"
                    }
                }
            }
        }
    }
}

static REFERENCE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(source_insight|insight|note|source):([A-Za-z0-9_]+)").expect("valid regex"));

/// Turn `source:<id>` and `note:<id>` references in model output into
/// markdown links to their pages, as the Next.js frontend made them
/// clickable. Insight references have no page of their own and stay text.
fn link_references(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 64);
    let mut last = 0;
    for caps in REFERENCE.captures_iter(text) {
        let whole = caps.get(0).expect("match");
        // Part of a longer word or already inside a URL.
        let before = text[..whole.start()].chars().next_back();
        if before.is_some_and(|c| c.is_alphanumeric() || c == '_' || c == '/' || c == ':') {
            continue;
        }
        let page = match &caps[1] {
            "source" => "sources",
            "note" => "notes",
            _ => continue,
        };
        if caps[2].len() > 100 {
            continue;
        }
        out.push_str(&text[last..whole.start()]);
        out.push_str(&format!("[{0}](/{page}/{0})", whole.as_str()));
        last = whole.end();
    }
    out.push_str(&text[last..]);
    out
}

fn answer_markdown(text: &str) -> Markup {
    markdown(&link_references(text))
}

// ---------------------------------------------------------------------------
// /search
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Ask,
    Search,
}

/// What the Search form holds.
struct SearchForm {
    query: String,
    search_type: SearchType,
    search_sources: bool,
    search_notes: bool,
    notebook_ids: Vec<String>,
}

impl SearchForm {
    fn from_fields(fields: &Fields) -> Self {
        // A submitted form always carries the `type` radio, and unchecked
        // boxes are then absent. A bare link (`?q=...`) has no `type`, so it
        // searches both, as the page's defaults do.
        let submitted = fields.has("type");
        Self {
            query: fields.string("q"),
            search_type: if fields.raw("type") == Some("vector") { SearchType::Vector } else { SearchType::Text },
            search_sources: !submitted || fields.has("search_sources"),
            search_notes: !submitted || fields.has("search_notes"),
            notebook_ids: fields.all("notebook_ids"),
        }
    }

    fn request(&self) -> SearchRequest {
        SearchRequest {
            query: self.query.clone(),
            search_type: self.search_type,
            limit: 100,
            search_sources: self.search_sources,
            search_notes: self.search_notes,
            minimum_score: 0.2,
            scope: NotebookScope { notebook_id: None, notebook_ids: Some(self.notebook_ids.clone()) },
        }
    }
}

/// What the Ask form holds.
#[derive(Default)]
struct AskForm {
    question: String,
    notebook_ids: Vec<String>,
    strategy_model: Option<String>,
    answer_model: Option<String>,
    final_answer_model: Option<String>,
}

impl AskForm {
    fn from_fields(fields: &Fields) -> Self {
        let model = |key: &str| fields.raw(key).filter(|v| !v.is_empty()).map(str::to_string);
        Self {
            question: fields.string("question"),
            notebook_ids: fields.all("notebook_ids"),
            strategy_model: model("strategy_model"),
            answer_model: model("answer_model"),
            final_answer_model: model("final_answer_model"),
        }
    }

    fn request(&self) -> AskRequest {
        AskRequest {
            question: self.question.clone(),
            strategy_model: self.strategy_model.clone().unwrap_or_default(),
            answer_model: self.answer_model.clone().unwrap_or_default(),
            final_answer_model: self.final_answer_model.clone().unwrap_or_default(),
            scope: NotebookScope { notebook_id: None, notebook_ids: Some(self.notebook_ids.clone()) },
        }
    }
}

/// Everything an Ask produced, from the event stream.
#[derive(Debug, Default)]
struct AskOutcome {
    strategy: Option<(String, Vec<AskSearch>)>,
    answers: Vec<String>,
    final_answer: Option<String>,
    error: Option<String>,
}

impl AskOutcome {
    fn from_events(events: Vec<AskEvent>) -> Self {
        let mut outcome = Self::default();
        for event in events {
            match event {
                AskEvent::Strategy { reasoning, searches } => outcome.strategy = Some((reasoning, searches)),
                AskEvent::Answer { content } => outcome.answers.push(content),
                AskEvent::FinalAnswer { content } => outcome.final_answer = Some(content),
                AskEvent::Complete { .. } => {}
                AskEvent::Error { message } => outcome.error = Some(message),
            }
        }
        outcome
    }
}

/// Data both tabs need.
struct SearchPageData {
    notebooks: Vec<NotebookSummary>,
    models: Vec<ModelResponse>,
    default_chat: Option<String>,
    has_embedding: bool,
}

async fn search_page_data(state: &AppState) -> WebResult<SearchPageData> {
    let defaults = get_default_models(state).await?;
    Ok(SearchPageData {
        notebooks: active_notebooks(state).await?,
        models: language_models(state).await?,
        default_chat: defaults.default_chat_model,
        has_embedding: state.models.embedding_opt(state).await?.is_some(),
    })
}

fn mode_tabs(mode: Mode) -> Markup {
    html! {
        p.sr-eyebrow { "Choose a mode" }
        nav.tabs aria-label="Search your knowledge base" {
            a.tab href="/search?mode=ask" aria-current=[(mode == Mode::Ask).then_some("page")] { "Ask (beta)" }
            a.tab href="/search?mode=search" aria-current=[(mode == Mode::Search).then_some("page")] { "Search" }
        }
    }
}

fn strategy_card(reasoning: &str, searches: &[AskSearch]) -> Markup {
    html! {
        details.card.sr-stage {
            summary { span.sr-stage-title { "Strategy" } }
            div.stack-sm.sr-stage-body {
                div {
                    p.muted.small { "Reasoning:" }
                    p { (reasoning) }
                }
                @if !searches.is_empty() {
                    div.stack-sm {
                        p.muted.small { "Search terms:" }
                        @for (i, s) in searches.iter().enumerate() {
                            div.sr-term {
                                span.badge { (i + 1) }
                                div {
                                    div.item-title { (s.term) }
                                    div.muted.small { (s.instructions) }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn partial_answer(answer: &str) -> Markup {
    html! { div.sr-answer { (answer_markdown(answer)) } }
}

fn save_form(question: &str, answer: &str, notebooks: &[NotebookSummary]) -> Markup {
    html! {
        form.sr-save method="post" action="/search/save-note" hx-post="/search/save-note" hx-swap="outerHTML" {
            input type="hidden" name="question" value=(question);
            input type="hidden" name="answer" value=(answer);
            details {
                summary.btn.btn-sm { "Save to Notebooks" }
                div.stack-sm.sr-save-body {
                    p.muted.small { "Select Notebook" }
                    (notebook_checkboxes("notebook_ids", notebooks, &[], "No notebooks found."))
                    @if !notebooks.is_empty() {
                        div.row { button.btn.btn-primary.btn-sm type="submit" { "Save to Notebook" } }
                    }
                }
            }
        }
    }
}

fn final_card(question: &str, answer: &str, notebooks: &[NotebookSummary]) -> Markup {
    html! {
        div.card.sr-final.stack-sm {
            h3 { "Final Answer" }
            (answer_markdown(answer))
            (save_form(question, answer, notebooks))
        }
    }
}

fn error_flash(message: &str) -> Markup {
    html! { div.flash.bad role="alert" { (message) } }
}

/// The Ask result area. The slots are filled by the server for a plain
/// form post, or one by one by `search.js` while streaming.
fn ask_output(outcome: Option<&AskOutcome>, question: &str, notebooks: &[NotebookSummary]) -> Markup {
    let empty = AskOutcome::default();
    let outcome = outcome.unwrap_or(&empty);
    html! {
        div #sr-ask-out .stack.sr-ask-out aria-live="polite" {
            div data-sr-slot="error" {
                @if let Some(error) = &outcome.error { (error_flash(error)) }
            }
            div data-sr-slot="strategy" {
                @if let Some((reasoning, searches)) = &outcome.strategy { (strategy_card(reasoning, searches)) }
            }
            details.card.sr-stage data-sr-slot="answers" hidden[outcome.answers.is_empty()] {
                summary {
                    span.sr-stage-title { "Individual Answers (" span data-sr-answer-count { (outcome.answers.len()) } ")" }
                }
                div.stack-sm.sr-stage-body data-sr-answer-list {
                    @for answer in &outcome.answers { (partial_answer(answer)) }
                }
            }
            div data-sr-slot="final" {
                @if let Some(answer) = &outcome.final_answer { (final_card(question, answer, notebooks)) }
            }
            div.row.muted.small data-sr-progress hidden {
                span.spinner {}
                "Processing your question..."
            }
        }
    }
}

fn ask_tab(data: &SearchPageData, form: &AskForm, outcome: Option<&AskOutcome>) -> Markup {
    let pick = |chosen: &Option<String>| chosen.clone().or_else(|| data.default_chat.clone());
    let (strategy, answer, final_answer) = (pick(&form.strategy_model), pick(&form.answer_model), pick(&form.final_answer_model));
    let custom = [&strategy, &answer, &final_answer].iter().any(|m| m.as_ref() != data.default_chat.as_ref());
    let name_of = |id: &Option<String>| {
        id.as_deref()
            .map(|id| data.models.iter().find(|m| m.id == id).map(|m| m.name.clone()).unwrap_or_else(|| id.to_string()))
            .unwrap_or_else(|| "Not set".to_string())
    };
    html! {
        section.card.stack {
            div {
                h2 { "Ask Your Knowledge Base (beta)" }
                p.muted { "The LLM will answer your query based on the documents in your knowledge base." }
            }
            form.stack method="post" action="/search/ask" data-sr-ask {
                div.field {
                    label for="sr-question" { "Question" }
                    textarea #sr-question name="question" rows="3" placeholder="Enter your question..." required { (form.question) }
                    p.hint { "Press Cmd/Ctrl+Enter to submit" }
                }
                (scope_picker(&data.notebooks, &form.notebook_ids))
                @if !data.has_embedding {
                    div.flash.warn {
                        "You can't use this feature because you have no embedding model selected. Please set one up in the "
                        a href="/settings/models" { "Models page" } "."
                    }
                } @else if data.models.is_empty() {
                    div.flash.warn {
                        "Ask needs a language model. Add one in "
                        a href="/settings/models" { "Manage → Models" } "."
                    }
                } @else {
                    details.sr-models {
                        summary {
                            span.muted.small { (if custom { "Using Custom Models" } else { "Using Default Models" }) }
                            span.badge.sr-mono { "Strategy: " (name_of(&strategy)) }
                            span.badge.sr-mono { "Answer: " (name_of(&answer)) }
                            span.badge.sr-mono { "Final: " (name_of(&final_answer)) }
                            span.btn.btn-ghost.btn-sm { "Advanced" }
                        }
                        div.stack-sm.sr-models-body {
                            p.muted.small { "Choose specific models for each stage of the Ask process" }
                            div.sr-model-grid {
                                (model_select("sr-strategy-model", "strategy_model", "Strategy Model", &data.models, strategy.as_deref(), None))
                                (model_select("sr-answer-model", "answer_model", "Answer Model", &data.models, answer.as_deref(), None))
                                (model_select("sr-final-model", "final_answer_model", "Final Answer Model", &data.models, final_answer.as_deref(), None))
                            }
                        }
                    }
                    div.row {
                        button.btn.btn-primary type="submit" data-sr-ask-button { "Ask" }
                    }
                }
            }
            (ask_output(outcome, &form.question, &data.notebooks))
        }
    }
}

fn score_of(result: &Value) -> Option<f64> {
    ["final_score", "similarity", "relevance"].iter().find_map(|k| result.get(*k).and_then(Value::as_f64))
}

/// Where a hit links to: notes and sources by id, insights and chunks by
/// their parent source.
fn result_href(id: &str, parent: &str) -> Option<String> {
    if id.starts_with("note:") {
        Some(format!("/notes/{id}"))
    } else if id.starts_with("source:") {
        Some(format!("/sources/{id}"))
    } else if parent.starts_with("source:") {
        Some(format!("/sources/{parent}"))
    } else {
        None
    }
}

fn search_results(query: &str, outcome: Option<&Result<SearchResponse, AppError>>) -> Markup {
    html! {
        div #sr-results .stack-sm {
            @match outcome {
                None => {}
                Some(Err(e)) => (error_flash(&e.detail())),
                Some(Ok(response)) => {
                    div.row-between {
                        h3 { (response.total_count) " results found" }
                        span.badge { (if response.search_type == "vector" { "Vector Search" } else { "Text Search" }) }
                    }
                    @if response.results.is_empty() {
                        (empty_state(&format!("No results found for “{query}”"), ""))
                    }
                    @for result in &response.results {
                        @let id = result.get("id").and_then(Value::as_str).unwrap_or_default();
                        @let parent = result.get("parent_id").and_then(Value::as_str).unwrap_or_default();
                        @let title = result.get("title").and_then(Value::as_str).filter(|t| !t.is_empty()).unwrap_or("Untitled");
                        @let matches: Vec<&str> = result.get("matches").and_then(Value::as_array).map(|m| m.iter().filter_map(Value::as_str).collect()).unwrap_or_default();
                        div.item.sr-hit {
                            div.grow.stack-sm {
                                div.row {
                                    @match result_href(id, parent) {
                                        Some(href) => a.item-title href=(href) { (title) },
                                        None => span.item-title { (title) },
                                    }
                                    @if let Some(score) = score_of(result) {
                                        span.badge.sr-mono { (format!("{score:.2}")) }
                                    }
                                    @if let Some(kind) = id.split(':').next().filter(|k| !k.is_empty()) {
                                        span.faint.small { (kind.replace('_', " ")) }
                                    }
                                }
                                @if !matches.is_empty() {
                                    details.sr-matches {
                                        summary.small.muted { "Matches (" (matches.len()) ")" }
                                        @for m in &matches { div.sr-match.small { (m) } }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn search_tab(data: &SearchPageData, form: &SearchForm, outcome: Option<&Result<SearchResponse, AppError>>) -> Markup {
    let vector = form.search_type == SearchType::Vector;
    html! {
        section.card.stack {
            div {
                h2 { "Search" }
                p.muted { "Search your knowledge base for specific keywords or concepts" }
            }
            form.stack method="get" action="/search"
                hx-get="/search" hx-select="#sr-results" hx-target="#sr-results" hx-swap="outerHTML" hx-push-url="true" hx-indicator="#sr-search-spin" {
                input type="hidden" name="mode" value="search";
                div.stack-sm {
                    label for="sr-query" { "Search" }
                    div.sr-query-row {
                        input #sr-query type="search" name="q" value=(form.query) placeholder="Enter search query..." autocomplete="off" required;
                        button.btn.btn-primary type="submit" { span #sr-search-spin .spinner.htmx-indicator {} "Search" }
                    }
                    p.hint { "Press Enter to search" }
                    p.hint {
                        @if vector {
                            "Vector search matches source content, insights, and note content by semantic similarity. Titles are not matched."
                        } @else {
                            "Text search matches source titles and content, insights, and note titles and content."
                        }
                    }
                }
                (scope_picker(&data.notebooks, &form.notebook_ids))
                div.sr-options {
                    fieldset.sr-fieldset {
                        legend { "Search Type" }
                        @if !data.has_embedding {
                            p.small.sr-warn { "Vector search requires an embedding model. Only text search is available." }
                        }
                        label.check { input type="radio" name="type" value="text" checked[!vector]; "Text Search" }
                        label.check {
                            input type="radio" name="type" value="vector" checked[vector] disabled[!data.has_embedding];
                            "Vector Search"
                        }
                    }
                    fieldset.sr-fieldset {
                        legend { "Search In" }
                        label.check { input type="checkbox" name="search_sources" value="true" checked[form.search_sources]; "Search Sources" }
                        label.check { input type="checkbox" name="search_notes" value="true" checked[form.search_notes]; "Search Notes" }
                    }
                }
            }
            (search_results(&form.query, outcome))
        }
    }
}

fn search_layout(state: &AppState, mode: Mode, body: Markup) -> Markup {
    let ctx = PageContext::new(state, "/search");
    page(
        &ctx,
        "Ask and Search",
        html! {
            (page_head("Ask and Search", None, html! {}))
            (mode_tabs(mode))
            (body)
        },
    )
}

async fn search_page(State(state): State<AppState>, RawQuery(raw): RawQuery) -> WebResult<Markup> {
    let fields = Fields::query(raw.as_deref());
    let mode = if fields.raw("mode") == Some("search") { Mode::Search } else { Mode::Ask };
    let data = search_page_data(&state).await?;
    let body = match mode {
        Mode::Search => {
            let form = SearchForm::from_fields(&fields);
            let outcome = if form.query.trim().is_empty() {
                None
            } else {
                let request = form.request();
                Some(match request.validate() {
                    Ok(()) => search::search(&state, &request).await,
                    Err(message) => Err(AppError::unprocessable(message)),
                })
            };
            search_tab(&data, &form, outcome.as_ref())
        }
        Mode::Ask => {
            // A `q` prefills the question (links from elsewhere); it is not run
            // on a GET, since an Ask spends model calls.
            let form = AskForm { question: fields.string("q"), ..AskForm::default() };
            ask_tab(&data, &form, None)
        }
    };
    Ok(search_layout(&state, mode, body))
}

fn check_ask(request: &AskRequest) -> Result<(), AppError> {
    if request.question.trim().is_empty() {
        return Err(AppError::bad_request("Please enter a question."));
    }
    request.validate().map_err(AppError::unprocessable)
}

/// The whole Ask in one request: the fallback for browsers without JavaScript.
async fn ask_submit(State(state): State<AppState>, body: Bytes) -> WebResult<Markup> {
    let form = AskForm::from_fields(&Fields::parse(&body));
    let request = form.request();
    check_ask(&request)?;
    let plan = search::prepare_ask(&state, &request).await?;
    let events: Vec<AskEvent> = search::ask_stream(state.clone(), plan).collect().await;
    let outcome = AskOutcome::from_events(events);
    let data = search_page_data(&state).await?;
    Ok(search_layout(&state, Mode::Ask, ask_tab(&data, &form, Some(&outcome))))
}

/// One Ask event for the page: the stage rendered to (sanitised) HTML.
fn web_event(event: &AskEvent, question: &str, notebooks: &[NotebookSummary]) -> Value {
    match event {
        AskEvent::Strategy { reasoning, searches } => {
            json!({ "type": "strategy", "html": strategy_card(reasoning, searches).into_string() })
        }
        AskEvent::Answer { content } => json!({ "type": "answer", "html": partial_answer(content).into_string() }),
        AskEvent::FinalAnswer { content } => {
            json!({ "type": "final_answer", "html": final_card(question, content, notebooks).into_string() })
        }
        AskEvent::Complete { .. } => json!({ "type": "complete" }),
        AskEvent::Error { message } => json!({ "type": "error", "message": message, "html": error_flash(message).into_string() }),
    }
}

/// The Ask as server-sent events, for `search.js`. Body: the `AskRequest` JSON.
async fn ask_stream(State(state): State<AppState>, ApiJson(request): ApiJson<AskRequest>) -> WebResult<Response> {
    check_ask(&request)?;
    let plan = search::prepare_ask(&state, &request).await?;
    let notebooks = active_notebooks(&state).await?;
    let question = request.question.clone();
    let events = search::ask_stream(state, plan).map(move |event| web_event(&event, &question, &notebooks));
    Ok(sse_response(events))
}

/// Save the final answer as an AI note in each chosen notebook.
async fn save_note(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> WebResult<Response> {
    let fields = Fields::parse(&body);
    let notebook_ids = fields.all("notebook_ids");
    if notebook_ids.is_empty() {
        return Err(AppError::bad_request("Select Notebook").into());
    }
    let (question, answer) = (fields.string("question"), fields.string("answer"));
    if answer.trim().is_empty() {
        return Err(AppError::bad_request("There is no answer to save.").into());
    }
    for notebook_id in &notebook_ids {
        let note = NewNote {
            title: Some(question.clone()).filter(|q| !q.trim().is_empty()),
            content: answer.clone(),
            note_type: Some(Some("ai".into())),
            notebook_id: Some(notebook_id.clone()),
        };
        create_note(&state, &note).await?;
    }
    if !is_htmx(&headers) {
        return Ok(Redirect::to(&format!("/notebooks/{}", notebook_ids[0])).into_response());
    }
    let notebooks = active_notebooks(&state).await?;
    let saved: Vec<&NotebookSummary> = notebooks.iter().filter(|nb| notebook_ids.contains(&nb.id)).collect();
    Ok(with_toast(
        html! {
            div.flash.ok.sr-save {
                "Saved to "
                @for (i, nb) in saved.iter().enumerate() {
                    @if i > 0 { ", " }
                    a href=(format!("/notebooks/{}", nb.id)) { (nb.name) }
                }
                "."
            }
        }
        .into_response(),
        "Successfully saved to notebook",
    ))
}

// ---------------------------------------------------------------------------
// /transformations
// ---------------------------------------------------------------------------

const TRANSFORMATIONS_DESC: &str =
    "Transformations are prompts that will be used by the LLM to process a source and extract insights, summaries, etc.";

fn flash_message(key: Option<&str>) -> Option<&'static str> {
    match key? {
        "created" => Some("Transformation created successfully"),
        "updated" => Some("Transformation updated successfully"),
        "deleted" => Some("Transformation deleted successfully"),
        "prompt" => Some("Default prompt saved"),
        _ => None,
    }
}

fn transformation_tabs(playground: bool) -> Markup {
    html! {
        p.sr-eyebrow { "Choose a workspace" }
        nav.tabs aria-label="Transformation views" {
            a.tab href="/transformations" aria-current=[(!playground).then_some("page")] { "Transformations" }
            a.tab href="/transformations?tab=playground" aria-current=[playground.then_some("page")] { "Playground" }
        }
    }
}

fn default_prompt_card(prompt: &str) -> Markup {
    html! {
        details.card.sr-stage {
            summary {
                div {
                    div.sr-stage-title { "Default Transformation Prompt" }
                    div.muted.small { "This will be added to all your transformation prompts" }
                }
            }
            form.stack-sm.sr-stage-body method="post" action="/transformations/default-prompt"
                hx-post="/transformations/default-prompt" hx-swap="none" {
                label.sr-only for="sr-default-prompt" { "Default Transformation Prompt" }
                textarea #sr-default-prompt .sr-mono.sr-tall name="transformation_instructions"
                    placeholder="Enter your default transformation instructions..." { (prompt) }
                div.row.sr-end { button.btn.btn-primary type="submit" { "Save" } }
            }
        }
    }
}

fn transformation_card(t: &TransformationView) -> Markup {
    html! {
        div.card.sr-tf-card.stack-sm {
            div.row-between {
                div.grow {
                    div.row {
                        span.card-title { (t.name) }
                        @if t.apply_default { span.badge.ok { "Default" } }
                    }
                    @if !t.description.is_empty() { p.muted.small { (t.description) } }
                }
                div.row {
                    a.btn.btn-sm href=(format!("/transformations?tab=playground&id={}", t.id)) { "Playground" }
                    a.btn.btn-sm href=(format!("/transformations/{}", t.id)) { "Edit" }
                    form.inline method="post" action=(format!("/transformations/{}/delete", t.id))
                        hx-post=(format!("/transformations/{}/delete", t.id))
                        hx-confirm="Are you sure you want to delete this transformation?"
                        hx-target="closest .sr-tf-card" hx-swap="outerHTML" {
                        button.btn.btn-danger.btn-sm type="submit" aria-label=(format!("Delete {}", t.name)) { "Delete" }
                    }
                }
            }
            details.sr-tf-details {
                summary.small.muted { "Details" }
                dl.sr-dl {
                    dt { "Title" } dd { (if t.title.is_empty() { "Untitled" } else { &t.title }) }
                    @if !t.description.is_empty() { dt { "Description" } dd { (t.description) } }
                    dt { "System Prompt" } dd { pre.sr-prompt { (t.prompt) } }
                }
            }
        }
    }
}

fn list_tab(list: &[TransformationView], default_prompt: &str) -> Markup {
    html! {
        div.stack {
            (default_prompt_card(default_prompt))
            div.row-between {
                h2 { "Custom Transformations" }
                a.btn.btn-primary href="/transformations/new" { "Create New" }
            }
            @if list.is_empty() {
                (empty_state("No transformations yet", "Create a transformation to get started"))
            } @else {
                div.stack-sm { @for t in list { (transformation_card(t)) } }
            }
        }
    }
}

/// What the playground form holds.
#[derive(Default)]
struct PlaygroundForm {
    transformation_id: String,
    model_id: String,
    input_text: String,
}

fn playground_output(output: Option<&str>) -> Markup {
    html! {
        div #sr-tf-output .stack-sm {
            @if let Some(output) = output {
                span.sr-stage-title { "Output" }
                div.card.flat.sr-output { (markdown(output)) }
            }
        }
    }
}

fn playground_tab(list: &[TransformationView], models: &[ModelResponse], form: &PlaygroundForm, output: Option<&str>) -> Markup {
    html! {
        section.card.stack {
            div {
                h2 { "Playground" }
                p.muted { (TRANSFORMATIONS_DESC) }
            }
            @if list.is_empty() {
                (empty_state("No transformations yet", "Create a transformation to get started"))
            } @else {
                form.stack method="post" action="/transformations/playground"
                    hx-post="/transformations/playground" hx-target="#sr-tf-output" hx-swap="outerHTML" hx-indicator="#sr-tf-spin" {
                    div.cols-2 {
                        div.field {
                            label for="sr-tf-select" { "Transformation" }
                            select #sr-tf-select name="transformation_id" required {
                                option value="" disabled selected[form.transformation_id.is_empty()] { "Select a transformation to start" }
                                @for t in list {
                                    option value=(t.id) selected[form.transformation_id == t.id] { (t.name) }
                                }
                            }
                        }
                        (model_select("sr-tf-model", "model_id", "Model", models, Some(form.model_id.as_str()), Some("Default transformation model")))
                    }
                    div.field {
                        label for="sr-tf-input" { "Input Text" }
                        textarea #sr-tf-input .sr-mono name="input_text" rows="8" placeholder="Enter some text to transform..." required { (form.input_text) }
                    }
                    div.row {
                        button.btn.btn-primary type="submit" { span #sr-tf-spin .spinner.htmx-indicator {} "Run Transformation" }
                    }
                }
            }
            (playground_output(output))
        }
    }
}

fn transformations_layout(state: &AppState, playground: bool, flash: Option<&str>, body: Markup) -> Markup {
    let ctx = PageContext::new(state, "/transformations");
    page(
        &ctx,
        "Transformations",
        html! {
            (page_head("Transformations", Some(TRANSFORMATIONS_DESC), html! {}))
            @if let Some(flash) = flash { div.flash.ok.sr-flash role="status" { (flash) } }
            (transformation_tabs(playground))
            (body)
        },
    )
}

async fn transformations_page(State(state): State<AppState>, RawQuery(raw): RawQuery) -> WebResult<Markup> {
    let fields = Fields::query(raw.as_deref());
    let list = transformations::list_transformations(&state).await?;
    let flash = flash_message(fields.raw("flash"));
    if fields.raw("tab") == Some("playground") {
        let models = language_models(&state).await?;
        let form = PlaygroundForm { transformation_id: fields.string("id"), ..PlaygroundForm::default() };
        return Ok(transformations_layout(&state, true, flash, playground_tab(&list, &models, &form, None)));
    }
    let prompt = transformations::get_default_prompt(&state).await?;
    Ok(transformations_layout(&state, false, flash, list_tab(&list, &prompt.transformation_instructions)))
}

async fn run_playground(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> WebResult<Markup> {
    let fields = Fields::parse(&body);
    let form = PlaygroundForm {
        transformation_id: fields.string("transformation_id"),
        model_id: fields.string("model_id"),
        input_text: fields.string("input_text"),
    };
    if form.transformation_id.is_empty() {
        return Err(AppError::bad_request("Select a transformation to start").into());
    }
    if form.input_text.trim().is_empty() {
        return Err(AppError::bad_request("Enter some text to transform.").into());
    }
    let request = ExecuteTransformation {
        transformation_id: form.transformation_id.clone(),
        input_text: form.input_text.clone(),
        model_id: Some(form.model_id.clone()).filter(|m| !m.is_empty()),
    };
    let result = transformations::execute_transformation(&state, &request).await?;
    if is_htmx(&headers) {
        return Ok(playground_output(Some(&result.output)));
    }
    let list = transformations::list_transformations(&state).await?;
    let models = language_models(&state).await?;
    Ok(transformations_layout(&state, true, None, playground_tab(&list, &models, &form, Some(&result.output))))
}

async fn save_default_prompt(State(state): State<AppState>, headers: HeaderMap, body: Bytes) -> WebResult<Response> {
    let fields = Fields::parse(&body);
    let update = DefaultPrompt { transformation_instructions: fields.string("transformation_instructions") };
    transformations::update_default_prompt(&state, &update).await?;
    if is_htmx(&headers) {
        return Ok(with_toast(html! {}.into_response(), "Default prompt saved"));
    }
    Ok(Redirect::to("/transformations?flash=prompt").into_response())
}

/// The create / edit form.
fn editor(existing: Option<&TransformationView>, models: &[ModelResponse]) -> Markup {
    let action = existing.map(|t| format!("/transformations/{}", t.id)).unwrap_or_else(|| "/transformations".into());
    let value = |f: fn(&TransformationView) -> &str| existing.map(f).unwrap_or_default().to_string();
    html! {
        form.card.stack.sr-editor method="post" action=(action) {
            div.field {
                label for="sr-tf-name" { "Name" }
                input #sr-tf-name type="text" name="name" value=(value(|t| &t.name)) placeholder="Unique identifier, e.g. key_topics" autocomplete="off" required;
            }
            div.cols-2 {
                div.field {
                    label for="sr-tf-title" { "Title" }
                    input #sr-tf-title type="text" name="title" value=(value(|t| &t.title)) placeholder="Displayed title, defaults to name" autocomplete="off";
                }
                (model_select("sr-tf-model-id", "model_id", "Model", models, Some(existing.and_then(|t| t.model_id.as_deref()).unwrap_or_default()), Some("System Default")))
            }
            label.check {
                input type="checkbox" name="apply_default" value="true" checked[existing.is_some_and(|t| t.apply_default)];
                "Suggest by default on new sources"
            }
            div.field {
                label for="sr-tf-description" { "Description" }
                textarea #sr-tf-description name="description" rows="2" placeholder="Describe what this transformation does." { (value(|t| &t.description)) }
            }
            div.field {
                label for="sr-tf-prompt" { "System Prompt" }
                textarea #sr-tf-prompt .sr-mono.sr-prompt-editor name="prompt" placeholder="Write the prompt that will power this transformation..." required { (value(|t| &t.prompt)) }
                p.hint { "Prompts should be written with the source content in mind. You can ask the model to summarise, extract insights, or produce structured outputs such as tables." }
            }
            div.row.sr-end {
                a.btn href="/transformations" { "Cancel" }
                button.btn.btn-primary type="submit" { (if existing.is_some() { "Save Transformation" } else { "Create New" }) }
            }
        }
    }
}

fn editor_page(state: &AppState, existing: Option<&TransformationView>, models: &[ModelResponse]) -> Markup {
    let ctx = PageContext::new(state, "/transformations");
    let title = if existing.is_some() { "Edit Transformation" } else { "New Transformation" };
    page(
        &ctx,
        title,
        html! {
            div.crumbs { a href="/transformations" { "Transformations" } " / " (existing.map(|t| t.name.as_str()).unwrap_or("New")) }
            (page_head(title, None, html! {}))
            (editor(existing, models))
        },
    )
}

async fn new_transformation_page(State(state): State<AppState>) -> WebResult<Markup> {
    let models = language_models(&state).await?;
    Ok(editor_page(&state, None, &models))
}

async fn edit_transformation_page(State(state): State<AppState>, Path(id): Path<String>) -> WebResult<Markup> {
    let transformation = transformations::get_transformation(&state, &id).await?;
    let models = language_models(&state).await?;
    Ok(editor_page(&state, Some(&transformation), &models))
}

/// Name and prompt are required, as the old form's schema had them.
fn editor_fields(fields: &Fields) -> Result<(String, String), AppError> {
    let name = fields.string("name").trim().to_string();
    let prompt = fields.string("prompt");
    if name.is_empty() {
        return Err(AppError::bad_request("Name is required"));
    }
    if prompt.trim().is_empty() {
        return Err(AppError::bad_request("Prompt is required"));
    }
    Ok((name, prompt))
}

async fn create_transformation(State(state): State<AppState>, body: Bytes) -> WebResult<Response> {
    let fields = Fields::parse(&body);
    let (name, prompt) = editor_fields(&fields)?;
    let title = Some(fields.string("title").trim().to_string()).filter(|t| !t.is_empty()).unwrap_or_else(|| name.clone());
    let new = NewTransformation {
        name,
        title,
        description: fields.string("description"),
        prompt,
        apply_default: fields.has("apply_default"),
        model_id: Some(fields.string("model_id")).filter(|m| !m.is_empty()),
    };
    transformations::create_transformation(&state, &new).await?;
    Ok(Redirect::to("/transformations?flash=created").into_response())
}

async fn update_transformation(State(state): State<AppState>, Path(id): Path<String>, body: Bytes) -> WebResult<Response> {
    let fields = Fields::parse(&body);
    let (name, prompt) = editor_fields(&fields)?;
    let update = TransformationUpdate {
        name: Some(name),
        title: Some(fields.string("title").trim().to_string()).filter(|t| !t.is_empty()),
        // A cleared description is saved as cleared.
        description: Some(Some(fields.string("description"))),
        prompt: Some(prompt),
        apply_default: Some(fields.has("apply_default")),
        model_id: Some(Some(fields.string("model_id")).filter(|m| !m.is_empty())),
    };
    transformations::update_transformation(&state, &id, &update).await?;
    Ok(Redirect::to("/transformations?flash=updated").into_response())
}

async fn delete_transformation(State(state): State<AppState>, Path(id): Path<String>, headers: HeaderMap) -> WebResult<Response> {
    transformations::delete_transformation(&state, &id).await?;
    if is_htmx(&headers) {
        return Ok(with_toast(html! {}.into_response(), "Transformation deleted successfully"));
    }
    Ok(Redirect::to("/transformations?flash=deleted").into_response())
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{header, Method, Request, StatusCode};
    use http_body_util::BodyExt;

    use super::*;
    use crate::ai::AnyModel;
    use crate::db::Doc;
    use crate::domain::Source;
    use crate::services::notes::{get_note, list_notes};
    use crate::testing::{MockEmbeddingModel, MockLanguageModel, TestApp};

    /// Post a urlencoded form; returns status, headers and body text.
    async fn form(app: &TestApp, path: &str, body: &str, htmx: bool) -> (StatusCode, HeaderMap, String) {
        let mut builder = Request::builder()
            .method(Method::POST)
            .uri(path)
            .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded");
        if htmx {
            builder = builder.header("hx-request", "true");
        }
        let response = app.send(builder.body(Body::from(body.to_string())).unwrap()).await;
        let (status, headers) = (response.status(), response.headers().clone());
        let bytes = response.into_body().collect().await.unwrap().to_bytes();
        (status, headers, String::from_utf8_lossy(&bytes).into_owned())
    }

    fn enc(value: &str) -> String {
        url::form_urlencoded::byte_serialize(value.as_bytes()).collect()
    }

    async fn page(app: &TestApp, path: &str) -> (StatusCode, String) {
        app.text(Method::GET, path, None).await
    }

    async fn notebook(app: &TestApp, name: &str) -> String {
        let (_, nb) = app.post("/api/notebooks", json!({ "name": name })).await;
        nb["id"].as_str().unwrap().to_string()
    }

    async fn language_model(app: &TestApp) -> String {
        let row = app.state.db.create("model", Doc::new().set("name", "mock-llm").set("provider", "openai").set("type", "language")).await.unwrap();
        row["id"].as_str().unwrap().to_string()
    }

    async fn embedded_source(app: &TestApp, embedder: &MockEmbeddingModel, notebook: &str, title: &str, text: &str) -> Source {
        let source = Source::create(&app.state, &Source { title: Some(title.into()), full_text: Some(text.into()), ..Source::default() })
            .await
            .unwrap();
        source.add_to_notebook(&app.state, notebook).await.unwrap();
        app.state
            .db
            .create(
                "source_embedding",
                Doc::new().record("source", &source.id).set("order", 0).set("content", text).set("embedding", embedder.vector(text)),
            )
            .await
            .unwrap();
        source
    }

    #[test]
    fn references_become_links() {
        let out = link_references("See [source:abc1] and note:n_2, insight:i9, http://x/source:zz, mysource:q.");
        assert!(out.contains("[[source:abc1](/sources/source:abc1)]"), "{out}");
        assert!(out.contains("[note:n_2](/notes/note:n_2)"), "{out}");
        assert!(out.contains("insight:i9,"), "{out}");
        assert!(out.contains("http://x/source:zz"), "{out}");
        assert!(out.contains("mysource:q"), "{out}");
        let html = answer_markdown("Waves [source:x]").into_string();
        assert!(html.contains(r#"href="/sources/source:x""#), "{html}");
    }

    #[test]
    fn fields_keep_repeated_keys() {
        let fields = Fields::parse(b"notebook_ids=a&notebook_ids=b&notebook_ids=&notebook_ids=a&q=x+y");
        assert_eq!(fields.all("notebook_ids"), ["a", "b"]);
        assert_eq!(fields.string("q"), "x y");
        assert!(!fields.has("missing"));
    }

    #[tokio::test]
    async fn search_page_renders_both_modes_and_empty_states() {
        let app = TestApp::new().await;
        let (status, body) = page(&app, "/search").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("Ask Your Knowledge Base (beta)"), "{body}");
        assert!(body.contains("no embedding model selected"), "{body}");
        assert!(body.contains("No notebooks yet."));

        let (status, body) = page(&app, "/search?mode=search").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("Search your knowledge base for specific keywords or concepts"));
        assert!(body.contains("Only text search is available"));
        assert!(!body.contains("results found"));

        let (status, body) = page(&app, "/search?mode=search&q=nothingmatcheshere&type=text&search_sources=true").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("0 results found"), "{body}");
        assert!(body.contains("No results found for “nothingmatcheshere”"), "{body}");

        // Vector search without an embedding model: the service's message, inline.
        let (_, body) = page(&app, "/search?mode=search&q=x&type=vector").await;
        assert!(body.contains("Vector search requires an embedding model. Please configure one"), "{body}");

        // An unknown notebook in the scope is reported, not ignored.
        let (_, body) = page(&app, "/search?mode=search&q=x&notebook_ids=notebook:nope").await;
        assert!(body.contains("flash bad"), "{body}");
    }

    #[tokio::test]
    async fn search_results_link_to_sources_and_notes_and_respect_scope() {
        let app = TestApp::new().await;
        let embedder = MockEmbeddingModel::new();
        app.state.models.set_override(AnyModel::Embedding(embedder.clone()));
        let physics = notebook(&app, "Physics").await;
        let music = notebook(&app, "Music").await;
        let waves = embedded_source(&app, &embedder, &physics, "Ocean waves", "ocean waves carry energy").await;
        let sound = embedded_source(&app, &embedder, &music, "Sound waves", "sound waves carry music").await;
        let (_, note) = app.post("/api/notes", json!({ "title": "Wave note", "content": "waves in a note", "notebook_id": physics })).await;
        let note_id = note["id"].as_str().unwrap().to_string();

        let (status, body) = page(&app, "/search?mode=search&q=waves&type=text&search_sources=true&search_notes=true").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains(&format!(r#"href="/sources/{}""#, waves.id)), "{body}");
        assert!(body.contains(&format!(r#"href="/notes/{note_id}""#)), "{body}");
        assert!(body.contains("Text Search"));

        // Notes unchecked.
        let (_, body) = page(&app, "/search?mode=search&q=waves&type=text&search_sources=true").await;
        assert!(!body.contains(&format!("/notes/{note_id}")), "{body}");

        // A bare link with only a query searches sources and notes.
        let (_, body) = page(&app, "/search?mode=search&q=waves").await;
        assert!(body.contains(&format!(r#"href="/sources/{}""#, waves.id)), "{body}");
        assert!(body.contains(&format!(r#"href="/notes/{note_id}""#)), "{body}");

        // Scoped to Music, vector search, with matches.
        let (_, body) =
            page(&app, &format!("/search?mode=search&q=waves&type=vector&search_sources=true&search_notes=true&notebook_ids={}", enc(&music))).await;
        assert!(body.contains(&format!("/sources/{}", sound.id)), "{body}");
        assert!(!body.contains(&format!("/sources/{}", waves.id)), "{body}");
        assert!(body.contains("Vector Search") && body.contains("Matches (1)"), "{body}");
        assert!(body.contains(&format!(r#"value="{music}" checked"#)), "the scope stays checked: {body}");
    }

    fn mock_ask(app: &TestApp) {
        let strategy = json!({ "reasoning": "Look for waves", "searches": [{ "term": "waves carry", "instructions": "What do they carry?" }] });
        app.state.models.set_override(AnyModel::Language(MockLanguageModel::new([
            strategy.to_string(),
            "Waves carry energy [source:x]".to_string(),
            "**Energy**, see [source:abc]".to_string(),
        ])));
    }

    #[tokio::test]
    async fn ask_without_javascript_renders_every_stage() {
        let app = TestApp::new().await;
        let embedder = MockEmbeddingModel::new();
        app.state.models.set_override(AnyModel::Embedding(embedder.clone()));
        let physics = notebook(&app, "Physics").await;
        embedded_source(&app, &embedder, &physics, "Ocean waves", "ocean waves carry energy").await;
        let model = language_model(&app).await;

        let (status, body) = page(&app, "/search").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("data-sr-ask-button") && body.contains("mock-llm"), "{body}");

        mock_ask(&app);
        let m = enc(&model);
        let request = format!(
            "question={}&notebook_ids={}&strategy_model={m}&answer_model={m}&final_answer_model={m}",
            enc("What do waves carry?"),
            enc(&physics)
        );
        let (status, _, body) = form(&app, "/search/ask", &request, false).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("Look for waves") && body.contains("waves carry"), "{body}");
        assert!(body.contains("Individual Answers (") && body.contains("Waves carry energy"), "{body}");
        assert!(body.contains("Final Answer") && body.contains("<strong>Energy</strong>"), "{body}");
        assert!(body.contains(r#"href="/sources/source:abc""#), "{body}");
        assert!(body.contains(r#"action="/search/save-note""#), "{body}");

        // Empty question and unknown model.
        let (status, _, body) = form(&app, "/search/ask", &format!("question=&strategy_model={m}&answer_model={m}&final_answer_model={m}"), false).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        let (status, _, _) = form(&app, "/search/ask", "question=hi&strategy_model=model:gone&answer_model=model:gone&final_answer_model=model:gone", false).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn ask_stream_sends_rendered_stages() {
        let app = TestApp::new().await;
        let embedder = MockEmbeddingModel::new();
        app.state.models.set_override(AnyModel::Embedding(embedder.clone()));
        let physics = notebook(&app, "Physics").await;
        embedded_source(&app, &embedder, &physics, "Ocean waves", "ocean waves carry energy").await;
        let model = language_model(&app).await;
        mock_ask(&app);

        let body = json!({ "question": "What do waves carry?", "strategy_model": model, "answer_model": model, "final_answer_model": model, "notebook_ids": [physics] });
        let (status, text) = app.text(Method::POST, "/search/ask/stream", Some(body)).await;
        assert_eq!(status, StatusCode::OK, "{text}");
        let events: Vec<Value> = text
            .split("\n\n")
            .filter(|c| !c.is_empty())
            .map(|c| serde_json::from_str(c.strip_prefix("data: ").unwrap()).unwrap())
            .collect();
        let types: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
        assert_eq!(types, ["strategy", "answer", "final_answer", "complete"]);
        assert!(events[0]["html"].as_str().unwrap().contains("Look for waves"));
        let final_html = events[2]["html"].as_str().unwrap();
        assert!(final_html.contains("Final Answer") && final_html.contains("/sources/source:abc"), "{final_html}");
        assert!(final_html.contains("Physics"), "the save form lists notebooks: {final_html}");

        // Errors before streaming are ordinary error responses.
        let (status, text) = app
            .text(Method::POST, "/search/ask/stream", Some(json!({ "question": " ", "strategy_model": model, "answer_model": model, "final_answer_model": model })))
            .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(text.contains("Please enter a question."));
    }

    #[tokio::test]
    async fn save_answer_as_note_in_notebooks() {
        let app = TestApp::new().await;
        let a = notebook(&app, "Alpha").await;
        let b = notebook(&app, "Beta").await;
        let request = format!("question={}&answer={}&notebook_ids={}&notebook_ids={}", enc("Why?"), enc("Because."), enc(&a), enc(&b));

        let (status, headers, body) = form(&app, "/search/save-note", &request, true).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(headers["hx-trigger"].to_str().unwrap().contains("Successfully saved to notebook"));
        assert!(body.contains("Alpha") && body.contains("Beta"), "{body}");
        for nb in [&a, &b] {
            let notes = list_notes(&app.state, Some(nb)).await.unwrap();
            assert_eq!(notes.len(), 1);
            let note = serde_json::to_value(get_note(&app.state, &notes[0].id).await.unwrap()).unwrap();
            assert_eq!((note["title"].as_str(), note["content"].as_str(), note["note_type"].as_str()), (Some("Why?"), Some("Because."), Some("ai")));
        }

        // Plain form: redirect to the first notebook.
        let (status, headers, _) = form(&app, "/search/save-note", &format!("question=q&answer=a&notebook_ids={}", enc(&a)), false).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(headers[header::LOCATION], format!("/notebooks/{a}"));

        let (status, _, body) = form(&app, "/search/save-note", "question=q&answer=a", true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("Select Notebook"));
        let (status, _, _) = form(&app, "/search/save-note", "question=q&answer=a&notebook_ids=notebook:nope", true).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    async fn transformation_id(app: &TestApp, name: &str) -> Option<TransformationView> {
        transformations::list_transformations(&app.state).await.unwrap().into_iter().find(|t| t.name == name)
    }

    #[tokio::test]
    async fn transformations_crud_through_forms() {
        let app = TestApp::new().await;
        let (status, body) = page(&app, "/transformations").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("Custom Transformations") && body.contains("Default Transformation Prompt"), "{body}");
        let (status, body) = page(&app, "/transformations/new").await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("Suggest by default on new sources"));

        // Create; a blank title falls back to the name.
        let (status, headers, _) =
            form(&app, "/transformations", "name=zz_topics&title=&description=Key+topics&prompt=List+the+topics.&apply_default=true&model_id=", false).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(headers[header::LOCATION], "/transformations?flash=created");
        let created = transformation_id(&app, "zz_topics").await.expect("created");
        assert_eq!((created.title.as_str(), created.apply_default, created.model_id.clone()), ("zz_topics", true, None));
        let (_, body) = page(&app, "/transformations?flash=created").await;
        assert!(body.contains("Transformation created successfully") && body.contains("zz_topics"));

        let (status, _, body) = form(&app, "/transformations", "name=+&prompt=x", false).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body.contains("Name is required"));

        // Edit: the form shows the values; clearing the description persists.
        let (status, body) = page(&app, &format!("/transformations/{}", created.id)).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("List the topics.") && body.contains("Key topics"), "{body}");
        let (status, _, _) =
            form(&app, &format!("/transformations/{}", created.id), "name=zz_topics&title=Topics&description=&prompt=New+prompt", false).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        let updated = transformations::get_transformation(&app.state, &created.id).await.unwrap();
        assert_eq!((updated.title.as_str(), updated.description.as_str(), updated.prompt.as_str(), updated.apply_default), ("Topics", "", "New prompt", false));

        // Missing records.
        let (status, body) = page(&app, "/transformations/transformation:nope").await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body.contains("Transformation not found"));
        let (status, _, _) = form(&app, "/transformations/transformation:nope/delete", "", true).await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        // Delete through htmx.
        let (status, headers, body) = form(&app, &format!("/transformations/{}/delete", created.id), "", true).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.is_empty());
        assert!(headers["hx-trigger"].to_str().unwrap().contains("Transformation deleted successfully"));
        assert!(transformation_id(&app, "zz_topics").await.is_none());
    }

    #[tokio::test]
    async fn default_prompt_and_empty_list() {
        let app = TestApp::new().await;
        for t in transformations::list_transformations(&app.state).await.unwrap() {
            transformations::delete_transformation(&app.state, &t.id).await.unwrap();
        }
        let (_, body) = page(&app, "/transformations").await;
        assert!(body.contains("No transformations yet"), "{body}");
        let (_, body) = page(&app, "/transformations?tab=playground").await;
        assert!(body.contains("No transformations yet"), "{body}");

        let (status, headers, _) = form(&app, "/transformations/default-prompt", "transformation_instructions=Be+brief.", false).await;
        assert_eq!(status, StatusCode::SEE_OTHER);
        assert_eq!(headers[header::LOCATION], "/transformations?flash=prompt");
        assert_eq!(transformations::get_default_prompt(&app.state).await.unwrap().transformation_instructions, "Be brief.");
        let (status, headers, _) = form(&app, "/transformations/default-prompt", "transformation_instructions=Shorter.", true).await;
        assert_eq!(status, StatusCode::OK);
        assert!(headers.contains_key("hx-trigger"));
        let (_, body) = page(&app, "/transformations").await;
        assert!(body.contains("Shorter."));
    }

    #[tokio::test]
    async fn playground_runs_a_transformation() {
        let app = TestApp::new().await;
        let (_, created) =
            app.post("/api/transformations", json!({ "name": "zz_play", "title": "Play", "description": "", "prompt": "Summarise." })).await;
        let id = created["id"].as_str().unwrap().to_string();
        let (status, body) = page(&app, &format!("/transformations?tab=playground&id={}", enc(&id))).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains(&format!(r#"value="{id}" selected"#)), "{body}");

        app.state.models.set_override(AnyModel::Language(MockLanguageModel::new(["# Summary\n\nShort."])));
        let request = format!("transformation_id={}&model_id=&input_text={}", enc(&id), enc("A long text."));
        let (status, _, body) = form(&app, "/transformations/playground", &request, true).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert!(body.contains("<h1>Summary</h1>") && body.contains("sr-tf-output") && !body.contains("<html"), "{body}");

        app.state.models.set_override(AnyModel::Language(MockLanguageModel::new(["Plain output"])));
        let (status, _, body) = form(&app, "/transformations/playground", &request, false).await;
        assert_eq!(status, StatusCode::OK);
        assert!(body.contains("<html") && body.contains("Plain output") && body.contains("A long text."), "{body}");

        let (status, _, _) = form(&app, "/transformations/playground", "transformation_id=transformation:nope&input_text=x", true).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let (status, _, _) = form(&app, "/transformations/playground", &format!("transformation_id={}&input_text=", enc(&id)), true).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }
}
