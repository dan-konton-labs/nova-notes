//! Interactive HTML explainers generated from a notebook (background job):
//! the Rust counterpart of `api/routers/explainers.py`,
//! `commands/explainer_commands.py` and `open_notebook/utils/explainer.py`.
//!
//! An explainer is one self-contained HTML page written by the chat model from
//! a notebook's sources, saved as a note with an HTML attachment flagged
//! `explainer` (migration 104) so the viewer may run its scripts.
//!
//! Safety model (ADR-015):
//!
//! - The page is shown in an iframe sandboxed with `allow-scripts` only, so it
//!   runs in an opaque origin and can never read the app's token or call the API.
//! - [`harden_html`] injects a Content-Security-Policy that blocks subresource
//!   requests. CSPs combine restrictively, so a policy the model writes itself
//!   cannot loosen it. It also strips resource hints and
//!   `<meta http-equiv=refresh>` (defence in depth; CSP doesn't cover hints).
//! - The same policy is sent as a header when the file is served
//!   ([`crate::services::notes::EXPLAINER_CSP_HEADER`]).
//!
//! Source text is untrusted: the prompt keeps instructions in the system
//! message and passes sources only inside delimited data blocks.

use std::sync::LazyLock;
use std::time::Instant;

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::generation_jobs::{
    self, char_prefix, clean_instructions, int_env, GeneratedNote, GenerationJobStarted,
    GenerationJobSummary, JobMessages, MAX_INSTRUCTIONS_CHARS, MAX_TITLE_CHARS,
};
use crate::db::Json;
use crate::domain::{Note, Notebook};
use crate::error::{AppError, AppResult};
use crate::jobs::{JobQueue, RetryPolicy};
use crate::state::AppState;

/// The job's command name.
pub const COMMAND_NAME: &str = "generate_explainer";

/// Total characters of source text sent to the model (roughly 4 chars/token,
/// so ~60k tokens), shared evenly between sources. Override with
/// `OPEN_NOTEBOOK_EXPLAINER_CONTEXT_CHARS`.
pub const DEFAULT_CONTEXT_CHARS: usize = 240_000;
/// Every source gets at least this many characters.
pub const MIN_PER_SOURCE_CHARS: usize = 2_000;
/// Output budget for the page. Override with `OPEN_NOTEBOOK_EXPLAINER_MAX_TOKENS`.
/// A full page from a two-source notebook measured ~30k output tokens on a
/// reasoning model, so the default leaves generous headroom.
pub const DEFAULT_MAX_TOKENS: usize = 48_000;
/// The note's text content (used for search, embeddings and chat context).
pub const MAX_NOTE_TEXT_CHARS: usize = 50_000;

/// Appended to a source cut to fit the budget.
pub const TRUNCATION_NOTICE: &str = "\n\n[Source truncated to fit the explainer context budget.]";

/// The no-network policy put into every page.
pub const EXPLAINER_CSP: &str = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; \
     img-src data:; font-src data:; media-src data:; connect-src 'none'; form-action 'none'; base-uri 'none'";

pub const NO_SOURCES_MESSAGE: &str =
    "Add at least one source to this notebook before generating an explainer.";
pub const WORKER_FAILED_MESSAGE: &str = "The background worker could not run the explainer job.";
pub const STALE_JOB_MESSAGE: &str =
    "The explainer job stopped unexpectedly or never started. Check that the \
     worker is running, then start a new explainer.";
const START_FAILED_MESSAGE: &str = "Failed to start the explainer";
const UNEXPECTED_MESSAGE: &str = "Explainer generation failed unexpectedly. Check the worker logs.";
const NO_SOURCE_TEXT_MESSAGE: &str =
    "This notebook has no source text yet. Add sources (and wait for them \
     to finish processing) before generating from them.";

const MESSAGES: JobMessages = JobMessages {
    worker_failed: WORKER_FAILED_MESSAGE,
    stale: STALE_JOB_MESSAGE,
};

/// Register the `generate_explainer` job. Not retried automatically (avoids
/// duplicate notes); the user starts a new one from the UI instead.
pub fn register_jobs(jobs: &JobQueue) {
    jobs.register(COMMAND_NAME, RetryPolicy::once(), |ctx, args| async move {
        let input: GenerateExplainerInput = serde_json::from_value(args)
            .map_err(|e| AppError::invalid(format!("Invalid {COMMAND_NAME} input: {e}")))?;
        Ok(generate_explainer_job(&ctx.state, &input).await)
    });
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

/// What to build from. Only `sources` (the notebook's saved sources) exists.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ExplainerMode {
    #[default]
    Sources,
}

/// `POST /notebooks/{id}/explainers` body (`ExplainerCreateRequest`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ExplainerCreateRequest {
    #[serde(default)]
    pub mode: ExplainerMode,
    /// Optional focus or angle for the reader, at most 2000 characters.
    #[serde(default, deserialize_with = "generation_jobs::instructions_field")]
    pub instructions: Option<String>,
}

/// Start generating an explainer; the result appears as a note in the
/// notebook. Missing notebook → 404, no sources → 400.
pub async fn create_explainer(
    state: &AppState,
    notebook_id: &str,
    request: &ExplainerCreateRequest,
) -> AppResult<GenerationJobStarted> {
    let notebook =
        generation_jobs::notebook_with_sources(state, notebook_id, NO_SOURCES_MESSAGE).await?;
    let args = json!({
        "notebook_id": notebook.id,
        "mode": request.mode,
        "instructions": clean_instructions(request.instructions.as_deref()),
        "requested_at": generation_jobs::requested_now(),
    });
    generation_jobs::submit_generation(
        state,
        COMMAND_NAME,
        &notebook.id,
        args,
        START_FAILED_MESSAGE,
    )
    .await
}

/// Explainer jobs for this notebook: every queued or running one, plus those
/// requested in the last 24 hours, newest first. A job queued or running for
/// more than 30 minutes is reported as failed. Missing notebook → 404.
pub async fn list_explainer_jobs(
    state: &AppState,
    notebook_id: &str,
) -> AppResult<Vec<GenerationJobSummary>> {
    let notebook = Notebook::get(state, notebook_id).await?;
    generation_jobs::list_generation_jobs(state, COMMAND_NAME, &notebook.id, MESSAGES).await
}

// ---------------------------------------------------------------------------
// The job
// ---------------------------------------------------------------------------

/// `generate_explainer` arguments (`GenerateExplainerInput`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateExplainerInput {
    pub notebook_id: String,
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default)]
    pub instructions: Option<String>,
    /// ISO time the API queued the job; lets the notes pane list and age jobs.
    #[serde(default)]
    pub requested_at: Option<String>,
}

fn default_mode() -> String {
    "sources".into()
}

/// Run one explainer job: the result JSON (`success`, `note_id`, `title`,
/// `error_message`, ...). Failures are reported, not raised.
pub async fn generate_explainer_job(state: &AppState, input: &GenerateExplainerInput) -> Json {
    let started = Instant::now();
    let outcome = run_explainer(state, input).await;
    if let Ok(note) = &outcome {
        tracing::info!("Explainer {} created for {}", note.id, input.notebook_id);
    }
    generation_jobs::generation_output(&input.notebook_id, started, outcome, UNEXPECTED_MESSAGE)
}

/// Build the explainer and save it as a note in the notebook.
pub async fn run_explainer(state: &AppState, input: &GenerateExplainerInput) -> AppResult<Note> {
    if input.mode != "sources" {
        return Err(AppError::invalid(format!(
            "Unsupported explainer mode: {}",
            input.mode
        )));
    }
    let notebook = Notebook::get(state, &input.notebook_id).await?;
    let block = notebook_sources_block(state, &notebook, context_chars()).await?;

    let instructions = input.instructions.as_deref().unwrap_or("").trim();
    let instructions = char_prefix(instructions, MAX_INSTRUCTIONS_CHARS);
    let system = state
        .prompts
        .render("explainer/system", json!({ "instructions": instructions }))?;
    let human = format!(
        "Notebook title: {}\nNotebook description: {}\n\n<sources>\n{block}\n</sources>\n\nWrite the complete HTML explainer now.",
        notebook.name,
        if notebook.description.is_empty() { "(none)" } else { &notebook.description },
    );

    let reply =
        generation_jobs::generate_reply(state, &system, &human, max_output_tokens()).await?;
    let html = harden_html(&extract_html(&reply)?);

    let title = match html_title(&html) {
        t if t.is_empty() => format!("Explainer: {}", notebook.name),
        t => t,
    };
    let title = char_prefix(&title, MAX_TITLE_CHARS).to_string();
    let content = match html_to_text(&html) {
        t if t.is_empty() => title.clone(),
        t => t,
    };
    let generated = GeneratedNote {
        file_name: generation_jobs::attachment_file_name(&title, "explainer", ".html"),
        title,
        content,
        explainer: Some(true),
        ext: ".html",
        mime: "text/html",
        bytes: html.into_bytes(),
    };
    generation_jobs::save_generated_note(state, &input.notebook_id, generated).await
}

/// The notebook's sources (with full text) as delimited data blocks.
pub(crate) async fn notebook_sources_block(
    state: &AppState,
    notebook: &Notebook,
    budget_chars: usize,
) -> AppResult<String> {
    let sources = notebook.sources(state, true).await?;
    let triples: Vec<(String, String, String)> = sources
        .into_iter()
        .map(|s| {
            let title = s
                .title
                .filter(|t| !t.is_empty())
                .unwrap_or_else(|| "Untitled source".into());
            (s.id, title, s.full_text.unwrap_or_default())
        })
        .collect();
    build_sources_block(&triples, budget_chars)
}

/// Source text budget (characters).
pub fn context_chars() -> usize {
    int_env(
        "OPEN_NOTEBOOK_EXPLAINER_CONTEXT_CHARS",
        DEFAULT_CONTEXT_CHARS,
    )
}

/// Output token budget.
pub fn max_output_tokens() -> usize {
    int_env("OPEN_NOTEBOOK_EXPLAINER_MAX_TOKENS", DEFAULT_MAX_TOKENS)
}

// ---------------------------------------------------------------------------
// Prompt building and output checks
// ---------------------------------------------------------------------------

/// Render `(id, title, text)` sources as delimited data blocks.
///
/// Sources without text are skipped (none left → 400). The budget is shared
/// evenly; a source that needs less than its share leaves the rest to the others.
pub fn build_sources_block(
    sources: &[(String, String, String)],
    budget_chars: usize,
) -> AppResult<String> {
    let usable: Vec<(&str, &str, &str)> = sources
        .iter()
        .map(|(id, title, text)| (id.as_str(), title.as_str(), text.trim()))
        .filter(|(_, _, text)| !text.is_empty())
        .collect();
    if usable.is_empty() {
        return Err(AppError::invalid(NO_SOURCE_TEXT_MESSAGE));
    }

    // Hand out the budget smallest-first so short sources don't waste it.
    let lengths: Vec<i64> = usable
        .iter()
        .map(|(_, _, text)| text.chars().count() as i64)
        .collect();
    let count = usable.len() as i64;
    let min = MIN_PER_SOURCE_CHARS as i64;
    let mut remaining = (budget_chars as i64).max(min * count);
    let mut limits = vec![0i64; usable.len()];
    let mut order: Vec<usize> = (0..usable.len()).collect();
    order.sort_by_key(|&i| lengths[i]);
    for (pos, &i) in order.iter().enumerate() {
        let share = remaining.div_euclid(count - pos as i64);
        let take = lengths[i].min(share.max(min));
        limits[i] = take;
        remaining -= take;
    }

    let parts: Vec<String> = usable
        .iter()
        .enumerate()
        .map(|(i, (id, title, text))| {
            let body = if lengths[i] > limits[i] {
                let cut = char_prefix(text, usize::try_from(limits[i]).unwrap_or(0)).trim_end();
                format!("{cut}{TRUNCATION_NOTICE}")
            } else {
                text.to_string()
            };
            format!(
                "<source n=\"{}\" id=\"{}\" title=\"{}\">\n{body}\n</source>",
                i + 1,
                attr(id),
                attr(title)
            )
        })
        .collect();
    Ok(parts.join("\n\n"))
}

fn attr(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
}

static FENCE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?is)```(?:html)?\s*\n(.*?)```").expect("regex"));
static FENCE_START: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)\A```(?:html)?\s*\n").expect("regex"));
static DOC_START: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)<!doctype\s+html|<html[\s>]").expect("regex"));
static HEAD_OPEN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)<head(\s[^>]*)?>").expect("regex"));
static HTML_OPEN: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?i)<html(\s[^>]*)?>").expect("regex"));
static RESOURCE_HINT: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?i)<link\b[^>]*\brel\s*=\s*["']?[^"'>]*\b(?:preconnect|dns-prefetch|prefetch|prerender|modulepreload|preload)\b[^>]*>"#,
    )
    .expect("regex")
});
static META_REFRESH: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r#"(?i)<meta\b[^>]*\bhttp-equiv\s*=\s*["']?\s*refresh\b[^>]*>"#).expect("regex")
});
static TITLE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?is)<title[^>]*>(.*?)</title>").expect("regex"));
static WS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"[ \t]+").expect("regex"));
static BLANKS: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"\n\s*\n+").expect("regex"));

/// Pull the HTML document out of a model reply.
///
/// Accepts a bare document or one wrapped in a code fence (with or without
/// chatter around it). A reply that stops before `</html>` was cut off by the
/// output limit and is rejected (`IncompleteGeneration`) rather than saved.
pub fn extract_html(text: &str) -> AppResult<String> {
    let mut candidate = text.trim().to_string();
    match FENCE
        .captures(&candidate)
        .and_then(|c| c.get(1))
        .filter(|inner| DOC_START.is_match(inner.as_str()))
    {
        Some(inner) => candidate = inner.as_str().trim().to_string(),
        // Unterminated fence (reply cut off) or prose before the document.
        None => candidate = FENCE_START.replace(&candidate, "").into_owned(),
    }

    let Some(start) = DOC_START.find(&candidate) else {
        return Err(AppError::IncompleteGeneration(
            "The model did not return an HTML document. Try again, or pick a stronger default chat model.".into(),
        ));
    };
    let candidate = &candidate[start.start()..];
    let Some(end) = candidate.to_ascii_lowercase().rfind("</html>") else {
        return Err(AppError::IncompleteGeneration(
            "The explainer was cut off before it finished (the model hit its output limit; hidden reasoning \
             counts towards it). Try again with fewer or shorter sources, a model with a larger output limit, \
             or raise OPEN_NOTEBOOK_EXPLAINER_MAX_TOKENS."
                .into(),
        ));
    };
    Ok(candidate[..end + "</html>".len()].to_string())
}

/// Strip resource hints and refresh redirects, then insert the no-network CSP
/// as the first element of `<head>` (creating one if needed). Regex on model
/// output: defence in depth only.
pub fn harden_html(html: &str) -> String {
    let meta = format!("<meta http-equiv=\"Content-Security-Policy\" content=\"{EXPLAINER_CSP}\">");
    let html = RESOURCE_HINT.replace_all(html, "");
    let html = META_REFRESH.replace_all(&html, "").into_owned();
    if let Some(head) = HEAD_OPEN.find(&html) {
        return format!("{}{meta}{}", &html[..head.end()], &html[head.end()..]);
    }
    if let Some(root) = HTML_OPEN.find(&html) {
        return format!(
            "{}<head>{meta}</head>{}",
            &html[..root.end()],
            &html[root.end()..]
        );
    }
    format!("<head>{meta}</head>{html}")
}

/// The page's `<title>` text, entities decoded and whitespace collapsed.
pub fn html_title(html: &str) -> String {
    let Some(inner) = TITLE.captures(html).and_then(|c| c.get(1)) else {
        return String::new();
    };
    let text = extract_text(&scraper::Html::parse_fragment(inner.as_str()));
    WS.replace_all(&text, " ").trim().to_string()
}

/// Readable text of the page (scripts, styles, SVG and `<head>` dropped) for
/// the note body, at most [`MAX_NOTE_TEXT_CHARS`] characters.
pub fn html_to_text(html: &str) -> String {
    let text = extract_text(&scraper::Html::parse_document(html));
    let lines: Vec<String> = text
        .lines()
        .map(|line| WS.replace_all(line, " ").trim().to_string())
        .collect();
    let text = lines.join("\n");
    let text = BLANKS.replace_all(&text, "\n\n");
    char_prefix(text.trim(), MAX_NOTE_TEXT_CHARS).to_string()
}

const SKIP_TAGS: [&str; 6] = ["script", "style", "template", "noscript", "svg", "head"];
const BLOCK_TAGS: [&str; 26] = [
    "p",
    "div",
    "section",
    "article",
    "header",
    "footer",
    "li",
    "tr",
    "br",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "details",
    "summary",
    "dt",
    "dd",
    "table",
    "ul",
    "ol",
    "blockquote",
    "pre",
    "figure",
    "figcaption",
];

fn extract_text(document: &scraper::Html) -> String {
    let mut out = String::new();
    visit(document.root_element(), &mut out);
    out
}

fn visit(element: scraper::ElementRef<'_>, out: &mut String) {
    let name = element.value().name();
    if SKIP_TAGS.contains(&name) {
        return;
    }
    let block = BLOCK_TAGS.contains(&name);
    if block {
        out.push('\n');
    }
    for child in element.children() {
        match child.value() {
            scraper::Node::Text(text) => out.push_str(text),
            scraper::Node::Element(_) => {
                if let Some(child) = scraper::ElementRef::wrap(child) {
                    visit(child, out);
                }
            }
            _ => {}
        }
    }
    if block {
        out.push('\n');
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use axum::http::StatusCode;
    use serde_json::json;

    use super::*;
    use crate::ai::AnyModel;
    use crate::domain::Source;
    use crate::testing::{MockEmbeddingModel, MockLanguageModel, TestApp};

    pub(crate) const DOC: &str = "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>\
        <title>How &amp; why tides work</title><style>body{}</style></head>\
        <body><h1>Tides</h1><p>The moon pulls the sea.</p>\
        <script>document.body.dataset.ok='1'</script></body></html>";

    #[test]
    fn extract_html_accepts_bare_fenced_and_chatty_replies() {
        assert_eq!(extract_html(DOC).unwrap(), DOC);
        assert_eq!(extract_html(&format!("```html\n{DOC}\n```")).unwrap(), DOC);
        assert_eq!(
            extract_html(&format!("Sure! Here it is:\n\n{DOC}\n\nEnjoy.")).unwrap(),
            DOC
        );
    }

    #[test]
    fn extract_html_rejects_missing_or_cut_off_documents() {
        for reply in [
            "",
            "No HTML here",
            "```html\n<!DOCTYPE html><html><body>cut",
        ] {
            assert!(
                matches!(extract_html(reply), Err(AppError::IncompleteGeneration(_))),
                "{reply:?}"
            );
        }
    }

    #[test]
    fn harden_puts_the_no_network_csp_first_in_head() {
        let hardened = harden_html(DOC);
        let head = hardened.find("<head>").unwrap() + "<head>".len();
        assert!(hardened[head..].starts_with("<meta http-equiv=\"Content-Security-Policy\""));
        assert!(hardened.contains("default-src 'none'") && hardened.contains("connect-src 'none'"));
        assert!(harden_html("<html><body>x</body></html>").contains("Content-Security-Policy"));
        let with_header = harden_html("<html><header>h</header></html>");
        assert!(with_header.contains("<header>") && with_header.contains("<head><meta"));
        assert!(harden_html("<p>bare</p>").starts_with("<head><meta"));
    }

    #[test]
    fn harden_strips_resource_hints_and_refresh() {
        let page = "<!DOCTYPE html><html><head>\
            <link rel='preconnect' href='https://x.evil'>\
            <link rel=\"dns-prefetch\" href=\"//leak.evil\">\
            <LINK REL=prefetch HREF=https://y.evil/a>\
            <meta http-equiv=\"refresh\" content=\"0;url=https://z.evil\">\
            <link rel='stylesheet' href='data:text/css,'>\
            </head><body>ok</body></html>";
        let hardened = harden_html(page);
        for gone in ["x.evil", "leak.evil", "y.evil", "z.evil"] {
            assert!(!hardened.contains(gone), "{gone}");
        }
        assert!(
            hardened.contains("rel='stylesheet'") && hardened.contains("Content-Security-Policy")
        );
    }

    #[test]
    fn title_and_text_skip_scripts_and_styles() {
        assert_eq!(html_title(DOC), "How & why tides work");
        assert_eq!(html_title("<html><body>no title</body></html>"), "");
        let text = html_to_text(DOC);
        assert!(
            text.contains("The moon pulls the sea.") && text.contains("Tides"),
            "{text}"
        );
        assert!(
            !text.contains("dataset") && !text.contains("body{}") && !text.contains("How &"),
            "{text}"
        );
        let spaced = html_to_text("<html><body><p>a  \t b</p>\n\n\n<p>c</p></body></html>");
        assert_eq!(spaced, "a b\n\nc");
    }

    #[test]
    fn sources_block_shares_the_budget_and_skips_empty_sources() {
        let sources = vec![
            (
                "source:a".to_string(),
                "Short".to_string(),
                "tiny".to_string(),
            ),
            (
                "source:b".to_string(),
                "Big \"one\"".to_string(),
                "x".repeat(50_000),
            ),
            (
                "source:c".to_string(),
                "Empty".to_string(),
                "  ".to_string(),
            ),
        ];
        let block = build_sources_block(&sources, 10_000).unwrap();
        assert_eq!(block.matches("<source ").count(), 2);
        assert!(block.contains("title=\"Big &quot;one&quot;\""));
        assert!(block.contains(TRUNCATION_NOTICE.trim()));
        assert!(block.len() < 12_000, "{}", block.len());
        assert!(
            block.starts_with("<source n=\"1\" id=\"source:a\" title=\"Short\">\ntiny\n</source>")
        );
    }

    #[test]
    fn sources_block_without_text_is_invalid_input() {
        let sources = vec![("source:a".to_string(), "A".to_string(), String::new())];
        assert!(matches!(
            build_sources_block(&sources, 1000),
            Err(AppError::InvalidInput(_))
        ));
    }

    // A full page from a 2-source notebook took ~30k output tokens on a
    // reasoning model (9k hidden), so the default must leave headroom.
    const _: () = assert!(DEFAULT_MAX_TOKENS >= 48_000);

    // --- through the API ------------------------------------------------------

    pub(crate) async fn notebook_with_source(app: &TestApp) -> (String, Source) {
        let (_, nb) = app
            .post(
                "/api/notebooks",
                json!({ "name": "Tides", "description": "Ocean basics" }),
            )
            .await;
        let notebook_id = nb["id"].as_str().unwrap().to_string();
        let source = Source::create(
            &app.state,
            &Source {
                title: Some("Moon".into()),
                full_text: Some("The moon pulls the sea.".into()),
                ..Source::default()
            },
        )
        .await
        .unwrap();
        source
            .add_to_notebook(&app.state, &notebook_id)
            .await
            .unwrap();
        (notebook_id, source)
    }

    pub(crate) fn use_models(app: &TestApp, replies: &[&str]) -> std::sync::Arc<MockLanguageModel> {
        let model = MockLanguageModel::new(replies.iter().copied());
        app.state
            .models
            .set_override(AnyModel::Language(model.clone()));
        app.state
            .models
            .set_override(AnyModel::Embedding(MockEmbeddingModel::new()));
        model
    }

    async fn command(app: &TestApp, job_id: &str) -> Json {
        app.state
            .db
            .select(job_id)
            .await
            .unwrap()
            .expect("command record")
    }

    #[tokio::test]
    async fn explainer_job_saves_a_hardened_explainer_note() {
        let app = TestApp::new().await;
        let (notebook_id, source) = notebook_with_source(&app).await;
        let model = use_models(&app, &[&format!("```html\n{DOC}\n```")]);

        let (status, body) = app
            .post(
                &format!("/api/notebooks/{notebook_id}/explainers"),
                json!({ "instructions": "  for kids  " }),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["status"], "queued");
        assert_eq!(body["notebook_id"], notebook_id.as_str());
        let job_id = body["job_id"].as_str().unwrap().to_string();
        assert!(job_id.starts_with("command:"));
        assert_eq!(app.wait_for_job(&job_id).await, "completed");

        let record = command(&app, &job_id).await;
        assert_eq!(record["args"]["mode"], "sources");
        assert_eq!(record["args"]["instructions"], "for kids");
        assert!(record["args"]["requested_at"]
            .as_str()
            .unwrap()
            .ends_with("+00:00"));
        let result = &record["result"];
        assert_eq!(result["success"], true, "{result}");
        assert_eq!(result["title"], "How & why tides work");
        let note_id = result["note_id"].as_str().unwrap().to_string();

        // Instructions go in the system prompt's focus block; sources are data blocks.
        let calls = model.calls.lock().unwrap().clone();
        assert!(
            calls[0][0].content.contains("for kids")
                && calls[0][0].content.contains("DARK MODE BY DEFAULT")
        );
        let human = &calls[0][1].content;
        assert!(
            human.contains(&format!(
                "<source n=\"1\" id=\"{}\" title=\"Moon\">",
                source.id
            )),
            "{human}"
        );
        assert!(human.starts_with("Notebook title: Tides\nNotebook description: Ocean basics\n"));
        assert_eq!(
            model.options.lock().unwrap()[0].max_tokens,
            Some(DEFAULT_MAX_TOKENS as u32)
        );

        let note = Note::get(&app.state, &note_id).await.unwrap();
        assert_eq!(note.explainer, Some(true));
        assert_eq!(note.note_type.as_deref(), Some("ai"));
        assert_eq!(note.attachment_mime.as_deref(), Some("text/html"));
        assert_eq!(
            note.attachment_name.as_deref(),
            Some("How-why-tides-work.html")
        );
        assert!(note
            .content
            .as_deref()
            .unwrap()
            .contains("The moon pulls the sea."));
        let path = crate::domain::note::note_file_path(
            &app.state.config,
            note.attachment_file.as_deref().unwrap(),
        )
        .unwrap();
        let stored = std::fs::read_to_string(path).unwrap();
        assert!(stored.contains("Content-Security-Policy"));
        assert_eq!(note.attachment_size, Some(stored.len() as i64));
        assert_eq!(
            note.notebook_ids(&app.state).await.unwrap(),
            vec![notebook_id.clone()]
        );

        let (status, viewed) = app.get(&format!("/api/notes/{note_id}")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(viewed["explainer"], true);

        let (status, jobs) = app
            .get(&format!("/api/notebooks/{notebook_id}/explainers/jobs"))
            .await;
        assert_eq!(status, StatusCode::OK, "{jobs}");
        assert_eq!(
            jobs,
            json!([{
                "job_id": job_id,
                "status": "completed",
                "requested_at": record["args"]["requested_at"],
                "note_id": note_id,
                "title": "How & why tides work",
                "error_message": null,
            }])
        );
    }

    #[tokio::test]
    async fn failures_are_reported_in_the_job_list() {
        let app = TestApp::new().await;
        let (notebook_id, _) = notebook_with_source(&app).await;
        // No chat model configured: the job completes with the reason.
        let (_, body) = app
            .post(
                &format!("/api/notebooks/{notebook_id}/explainers"),
                json!({}),
            )
            .await;
        let job_id = body["job_id"].as_str().unwrap().to_string();
        assert_eq!(app.wait_for_job(&job_id).await, "completed");
        let record = command(&app, &job_id).await;
        assert_eq!(record["result"]["success"], false);
        assert_eq!(record["args"]["instructions"], Json::Null);

        // A cut-off page.
        use_models(&app, &["```html\n<!DOCTYPE html><html><body>cut"]);
        let (_, body) = app
            .post(
                &format!("/api/notebooks/{notebook_id}/explainers"),
                json!({}),
            )
            .await;
        let cut_id = body["job_id"].as_str().unwrap().to_string();
        assert_eq!(app.wait_for_job(&cut_id).await, "completed");

        let (status, jobs) = app
            .get(&format!("/api/notebooks/{notebook_id}/explainers/jobs"))
            .await;
        assert_eq!(status, StatusCode::OK);
        let jobs = jobs.as_array().unwrap();
        assert_eq!(jobs.len(), 2);
        assert_eq!(jobs[0]["job_id"], cut_id.as_str(), "newest first");
        assert_eq!(jobs[0]["status"], "failed");
        assert!(
            jobs[0]["error_message"]
                .as_str()
                .unwrap()
                .contains("cut off"),
            "{}",
            jobs[0]
        );
        assert_eq!(jobs[1]["status"], "failed");
        assert!(
            jobs[1]["error_message"]
                .as_str()
                .unwrap()
                .contains("Models"),
            "{}",
            jobs[1]
        );
        assert_eq!(jobs[1]["note_id"], Json::Null);
        // Nothing was saved.
        assert!(Notebook::get(&app.state, &notebook_id)
            .await
            .unwrap()
            .notes(&app.state, false)
            .await
            .unwrap()
            .is_empty());
    }

    #[tokio::test]
    async fn create_explainer_needs_sources_a_known_mode_and_short_instructions() {
        let app = TestApp::new().await;
        let (_, nb) = app.post("/api/notebooks", json!({ "name": "Empty" })).await;
        let url = format!("/api/notebooks/{}/explainers", nb["id"].as_str().unwrap());
        let (status, body) = app.post(&url, json!({})).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["detail"], NO_SOURCES_MESSAGE);
        assert_eq!(
            app.post(&url, json!({ "mode": "repo" })).await.0,
            StatusCode::UNPROCESSABLE_ENTITY
        );
        let long = "x".repeat(MAX_INSTRUCTIONS_CHARS + 1);
        assert_eq!(
            app.post(&url, json!({ "instructions": long })).await.0,
            StatusCode::UNPROCESSABLE_ENTITY
        );
        let (status, _) = app
            .post("/api/notebooks/notebook:missing/explainers", json!({}))
            .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn explainer_jobs_keep_failures_visible_and_skip_other_notebooks() {
        let app = TestApp::new().await;
        let (notebook_id, _) = notebook_with_source(&app).await;
        let now = chrono::Utc::now();
        let iso = |minutes: i64| (now - chrono::Duration::minutes(minutes)).to_rfc3339();
        let rows = [
            (
                "command:done",
                COMMAND_NAME,
                notebook_id.as_str(),
                "completed",
                iso(5),
                json!({ "success": true, "note_id": "note:1", "title": "Tides" }),
            ),
            (
                "command:stuck",
                COMMAND_NAME,
                notebook_id.as_str(),
                "running",
                iso(45),
                Json::Null,
            ),
            (
                "command:crash",
                COMMAND_NAME,
                notebook_id.as_str(),
                "failed",
                iso(3),
                Json::Null,
            ),
            (
                "command:other",
                COMMAND_NAME,
                "notebook:other",
                "new",
                iso(1),
                Json::Null,
            ),
            (
                "command:infographic",
                "generate_infographic",
                notebook_id.as_str(),
                "new",
                iso(1),
                Json::Null,
            ),
        ];
        for (id, name, nb, status, at, result) in rows {
            app.state
                .db
                .query(
                    "CREATE $id CONTENT $content",
                    crate::db::Params::new().record("id", id).set(
                        "content",
                        json!({
                            "app": "open_notebook", "name": name, "status": status, "result": result,
                            "error_message": if status == "failed" { json!("boom") } else { Json::Null },
                            "args": { "notebook_id": nb, "requested_at": at },
                        }),
                    ),
                )
                .await
                .unwrap();
        }
        let (status, jobs) = app
            .get(&format!("/api/notebooks/{notebook_id}/explainers/jobs"))
            .await;
        assert_eq!(status, StatusCode::OK, "{jobs}");
        let ids: Vec<&str> = jobs
            .as_array()
            .unwrap()
            .iter()
            .map(|j| j["job_id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, ["command:crash", "command:done", "command:stuck"]);
        assert_eq!(jobs[0]["error_message"], "boom");
        assert_eq!(jobs[1]["note_id"], "note:1");
        assert_eq!(jobs[2]["status"], "failed");
        assert_eq!(jobs[2]["error_message"], STALE_JOB_MESSAGE);

        let (status, _) = app
            .get("/api/notebooks/notebook:missing/explainers/jobs")
            .await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn unsupported_mode_in_the_job_is_reported() {
        let app = TestApp::new().await;
        let input = GenerateExplainerInput {
            notebook_id: "notebook:n1".into(),
            mode: "repo".into(),
            instructions: None,
            requested_at: None,
        };
        let out = generate_explainer_job(&app.state, &input).await;
        assert_eq!(out["success"], false);
        assert_eq!(out["error_message"], "Unsupported explainer mode: repo");
        assert_eq!(out["notebook_id"], "notebook:n1");
    }
}
