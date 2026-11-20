//! Context building for chat: the Rust counterpart of
//! `open_notebook/utils/context_builder.py`, plus model selection by context
//! size (`open_notebook/ai/provision.py`).
//!
//! - [`build_context`] backs `POST /api/chat/context`: a notebook's sources and
//!   notes chosen by a per-item status config.
//! - [`build_source_context`] backs source chat: one source plus its insights
//!   under a token budget, truncating the source text rather than dropping it.
//!
//! The status strings ("not in context", "insights", "full content") are the
//! protocol the frontend uses; they are matched by substring, as in Python.

use std::fmt;

use nn_llm::SharedLanguageModel;
use serde::de::{MapAccess, Visitor};
use serde::ser::SerializeMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::db::{with_table, Json};
use crate::domain::{DefaultModelType, Note, Notebook, Source, SourceInsight};
use crate::error::{AppError, AppResult};
use crate::jobs::JobQueue;
use crate::services::notebooks::{ensure_table, not_found_as};
use crate::state::AppState;

/// Context building has no background jobs.
pub fn register_jobs(_jobs: &JobQueue) {}

/// Appended to a source text cut to fit the budget.
pub const SOURCE_TRUNCATION_NOTICE: &str = "\n\n[Source content truncated to fit the context token budget.]";
/// Share of the budget reserved for insights when the source alone is too big.
pub const SOURCE_INSIGHT_BUDGET_RATIO: f64 = 0.2;
/// Token budget of a source chat's context.
pub const SOURCE_CHAT_CONTEXT_TOKENS: usize = 50_000;
/// Above this many prompt tokens the large-context model is used.
pub const LARGE_CONTEXT_THRESHOLD: usize = 105_000;

// ---------------------------------------------------------------------------
// Model selection
// ---------------------------------------------------------------------------

/// Pick the language model for a prompt, as Python's `provision_langchain_model`:
/// the large-context model when `content` exceeds [`LARGE_CONTEXT_THRESHOLD`]
/// tokens, otherwise `model_id` when given, otherwise the default for `kind`.
pub async fn provision_language_model(
    state: &AppState,
    content: &str,
    model_id: Option<&str>,
    kind: DefaultModelType,
) -> AppResult<SharedLanguageModel> {
    let tokens = nn_content::token_count(content);
    if tokens > LARGE_CONTEXT_THRESHOLD {
        tracing::debug!("Using the large context model because the content has {tokens} tokens");
        return state.models.language(state, DefaultModelType::LargeContext).await;
    }
    state.models.language_or_default(state, model_id, kind).await
}

// ---------------------------------------------------------------------------
// Python-style rendering
// ---------------------------------------------------------------------------

/// Render a JSON value the way Python's `str()` renders the equivalent
/// dict / list / str (`{'id': 'source:1', 'title': None}`). The Python build
/// put such strings into prompts and measured context size with them.
pub(crate) fn py_repr(value: &Json) -> String {
    let mut out = String::new();
    write_py_repr(value, &mut out);
    out
}

fn write_py_repr(value: &Json, out: &mut String) {
    match value {
        Json::Null => out.push_str("None"),
        Json::Bool(true) => out.push_str("True"),
        Json::Bool(false) => out.push_str("False"),
        Json::Number(n) => match (n.as_i64(), n.as_u64(), n.as_f64()) {
            (Some(i), _, _) => out.push_str(&i.to_string()),
            (_, Some(u), _) => out.push_str(&u.to_string()),
            (_, _, Some(f)) if f.is_finite() && f.fract() == 0.0 && f.abs() < 1e16 => out.push_str(&format!("{f:.1}")),
            (_, _, Some(f)) => out.push_str(&f.to_string()),
            _ => out.push_str(&n.to_string()),
        },
        Json::String(s) => out.push_str(&py_str_repr(s)),
        Json::Array(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                write_py_repr(item, out);
            }
            out.push(']');
        }
        Json::Object(map) => {
            out.push('{');
            for (i, (key, item)) in map.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&py_str_repr(key));
                out.push_str(": ");
                write_py_repr(item, out);
            }
            out.push('}');
        }
    }
}

/// Python's `repr(str)`: single quotes unless the text has a single quote
/// and no double quote; backslashes, the quote and control characters escaped.
pub(crate) fn py_str_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if (c as u32) < 0x20 || c as u32 == 0x7f => out.push_str(&format!("\\x{:02x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

// ---------------------------------------------------------------------------
// Notebook context (POST /api/chat/context)
// ---------------------------------------------------------------------------

/// Item id → status string, in the order the client sent them.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ItemStatuses(pub Vec<(String, String)>);

impl Serialize for ItemStatuses {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let mut map = serializer.serialize_map(Some(self.0.len()))?;
        for (id, status) in &self.0 {
            map.serialize_entry(id, status)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for ItemStatuses {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct StatusVisitor;
        impl<'de> Visitor<'de> for StatusVisitor {
            type Value = ItemStatuses;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an object of item id to status")
            }
            fn visit_unit<E>(self) -> Result<Self::Value, E> {
                Ok(ItemStatuses::default())
            }
            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut entries = Vec::new();
                while let Some((id, status)) = map.next_entry::<String, Json>()? {
                    // A status that is not a string matches no rule.
                    if let Json::String(status) = status {
                        entries.push((id, status));
                    }
                }
                Ok(ItemStatuses(entries))
            }
        }
        deserializer.deserialize_any(StatusVisitor)
    }
}

/// Which sources and notes go into a notebook chat's context.
///
/// Status per item: containing "not in" skips it, "insights" includes a
/// source's short context (title + insights), "full content" its long
/// context (plus full text). Notes only support "full content".
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sources: Option<ItemStatuses>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub notes: Option<ItemStatuses>,
}

impl ContextConfig {
    /// Python's falsy `{}`: no config at all means "every source and note,
    /// short context".
    pub fn is_empty(&self) -> bool {
        self.sources.is_none() && self.notes.is_none()
    }
}

/// The included context items (`{"sources": [...], "notes": [...]}`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct NotebookContext {
    pub sources: Vec<Json>,
    pub notes: Vec<Json>,
}

/// `BuildContextRequest`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuildContextRequest {
    pub notebook_id: String,
    /// Must be a JSON object (Python `Dict[str, Any]`); a list is a 422.
    #[serde(deserialize_with = "object_config")]
    pub context_config: ContextConfig,
}

fn object_config<'de, D: Deserializer<'de>>(deserializer: D) -> Result<ContextConfig, D::Error> {
    use serde::de::Error as _;
    match Json::deserialize(deserializer)? {
        value @ Json::Object(_) => serde_json::from_value(value).map_err(D::Error::custom),
        _ => Err(D::Error::custom("context_config: Input should be a valid dictionary")),
    }
}

/// `BuildContextResponse`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuildContextResponse {
    pub context: NotebookContext,
    pub token_count: usize,
    pub char_count: usize,
}

const NOTEBOOK_NOT_FOUND: &str = "Notebook not found";

/// Load a notebook by id; anything that is not an existing notebook is
/// `NotFound` with `missing` as the detail.
pub(crate) async fn load_notebook(state: &AppState, id: &str, missing: &'static str) -> AppResult<Notebook> {
    ensure_table(id, Notebook::TABLE, missing)?;
    Notebook::get(state, id).await.map_err(not_found_as(missing))
}

/// `POST /api/chat/context`: the selected context of a notebook with its
/// size. `char_count` / `token_count` measure the Python `str()` rendering
/// of every included item, as the Python build did.
pub async fn build_context(state: &AppState, request: &BuildContextRequest) -> AppResult<BuildContextResponse> {
    let notebook = load_notebook(state, &request.notebook_id, NOTEBOOK_NOT_FOUND).await?;
    let config = (!request.context_config.is_empty()).then_some(&request.context_config);
    let (context, total) = build_notebook_context(state, &notebook, config).await?;
    let token_count = if total.is_empty() { 0 } else { nn_content::token_count(&total) };
    Ok(BuildContextResponse { context, token_count, char_count: total.chars().count() })
}

/// Assemble a notebook's context from `config`, or every source and note
/// (short context) without one. A broken item is logged and skipped; it
/// never fails the whole request. Returns the items and the concatenated
/// Python `str()` of each, used for size accounting.
pub async fn build_notebook_context(
    state: &AppState,
    notebook: &Notebook,
    config: Option<&ContextConfig>,
) -> AppResult<(NotebookContext, String)> {
    let mut context = NotebookContext::default();
    let mut total = String::new();
    let mut include = |list: &mut Vec<Json>, item: Json| {
        total.push_str(&py_repr(&item));
        list.push(item);
    };

    if let Some(config) = config {
        for (source_id, status) in config.sources.as_ref().map(|s| s.0.as_slice()).unwrap_or_default() {
            if status.contains("not in") {
                continue;
            }
            let long = if status.contains("insights") {
                false
            } else if status.contains("full content") {
                true
            } else {
                continue;
            };
            let Ok(source) = Source::get(state, &with_table(Source::TABLE, source_id)).await else {
                continue;
            };
            match source.context(state, long, None).await {
                Ok(item) => include(&mut context.sources, item),
                Err(e) => tracing::warn!("Error processing source {source_id}: {e}"),
            }
        }
        for (note_id, status) in config.notes.as_ref().map(|n| n.0.as_slice()).unwrap_or_default() {
            if status.contains("not in") || !status.contains("full content") {
                continue;
            }
            match Note::get(state, &with_table(Note::TABLE, note_id)).await {
                Ok(note) if !note.untouched_draft => include(&mut context.notes, note.context(true)),
                Ok(_) => {}
                Err(e) => tracing::warn!("Error processing note {note_id}: {e}"),
            }
        }
    } else {
        let sources = notebook.sources(state, false).await?;
        let ids: Vec<String> = sources.iter().map(|s| s.id.clone()).collect();
        let mut insights = SourceInsight::for_sources(state, &ids).await.unwrap_or_else(|e| {
            tracing::warn!("Error batch-fetching source insights: {e}");
            Default::default()
        });
        for source in &sources {
            let source_insights = insights.remove(&source.id).unwrap_or_default();
            match source.context(state, false, Some(source_insights)).await {
                Ok(item) => include(&mut context.sources, item),
                Err(e) => tracing::warn!("Error processing source {}: {e}", source.id),
            }
        }
        for note in notebook.notes(state, false).await? {
            include(&mut context.notes, note.context(false));
        }
    }
    Ok((context, total))
}

// ---------------------------------------------------------------------------
// Source context (source chat)
// ---------------------------------------------------------------------------

/// The source part of a source chat's context.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextSource {
    pub id: String,
    pub title: Option<String>,
    pub full_text: Option<String>,
}

/// One insight of a source chat's context.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ContextInsight {
    pub id: String,
    pub source_id: String,
    pub insight_type: String,
    pub content: String,
}

/// What happened to the source text under the budget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SourceTextStatus {
    /// In full.
    Available,
    /// A prefix, followed by [`SOURCE_TRUNCATION_NOTICE`].
    Truncated,
    /// The source has no text.
    Missing,
    /// Not even a prefix fit the budget.
    OmittedBudget,
    /// No such source.
    NotFound,
}

impl SourceTextStatus {
    /// Whether any source text reaches the model.
    pub fn is_available(&self) -> bool {
        matches!(self, Self::Available | Self::Truncated)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SourceContextMetadata {
    pub source_count: usize,
    pub note_count: usize,
    pub insight_count: usize,
    pub source_text_status: SourceTextStatus,
    pub source_truncated: bool,
}

/// A source and its insights, budgeted (`build_source_context`'s result).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SourceContext {
    /// Zero or one source.
    pub sources: Vec<ContextSource>,
    /// Always empty; kept for the shape of the Python result.
    pub notes: Vec<Json>,
    pub insights: Vec<ContextInsight>,
    pub total_tokens: usize,
    pub total_items: usize,
    pub metadata: SourceContextMetadata,
}

impl SourceContext {
    /// The Markdown sent to the model.
    pub fn formatted(&self) -> String {
        format_source_context(&self.sources, &self.insights)
    }
}

/// Render sources and insights as the Markdown block Source Chat sends.
pub fn format_source_context(sources: &[ContextSource], insights: &[ContextInsight]) -> String {
    let mut parts: Vec<String> = Vec::new();
    if !sources.is_empty() {
        parts.push("## SOURCE CONTENT".into());
        for source in sources {
            parts.push(format!("**Source ID:** {}", source.id));
            parts.push(format!("**Title:** {}", source.title.as_deref().unwrap_or("No title")));
            match source.full_text.as_deref().filter(|t| !t.trim().is_empty()) {
                Some(text) => parts.push(format!("**Content:**\n{text}")),
                None => parts.push("**Content:**\n[Source text is unavailable in this context.]".into()),
            }
            parts.push(String::new());
        }
    }
    if !insights.is_empty() {
        parts.push("## SOURCE INSIGHTS".into());
        for insight in insights {
            parts.push(format!("**Insight ID:** {}", insight.id));
            parts.push(format!("**Type:** {}", insight.insight_type));
            parts.push(format!("**Content:** {}", insight.content));
            parts.push(String::new());
        }
    }
    parts.join("\n")
}

fn rendered_tokens(source: Option<&ContextSource>, insights: &[ContextInsight]) -> usize {
    let sources = source.map(std::slice::from_ref).unwrap_or_default();
    nn_content::token_count(&format_source_context(sources, insights))
}

fn floor_char_boundary(text: &str, mut index: usize) -> usize {
    index = index.min(text.len());
    while !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

/// Cut an over-budget source's text to the longest prefix that, with the
/// truncation notice and the already selected insights, fits `max_tokens`.
/// `None` when not even a non-blank prefix fits. The second value says
/// whether the text was cut.
fn truncate_source_to_budget(
    source: &ContextSource,
    max_tokens: usize,
    insights: &[ContextInsight],
) -> (Option<ContextSource>, bool) {
    let Some(text) = source.full_text.as_deref().filter(|t| !t.trim().is_empty()) else {
        return (None, false);
    };
    let candidate = |len: usize| ContextSource {
        full_text: Some(format!("{}{SOURCE_TRUNCATION_NOTICE}", &text[..len])),
        ..source.clone()
    };
    if rendered_tokens(Some(&candidate(0)), insights) > max_tokens {
        return (None, true);
    }
    let fits = |len: usize| {
        let len = floor_char_boundary(text, len);
        !text[..len].trim().is_empty() && rendered_tokens(Some(&candidate(len)), insights) <= max_tokens
    };

    // A token is at least one byte, and usually about four: find an upper
    // bound that does not fit by doubling, then binary-search below it.
    let mut low = 1usize;
    let mut high = text.len().min(max_tokens.saturating_mul(4).max(1));
    while high < text.len() && fits(high) {
        low = high;
        high = (high.saturating_mul(2)).min(text.len());
    }
    let mut best = 0usize;
    while low <= high {
        let mid = low + (high - low) / 2;
        if fits(mid) {
            best = floor_char_boundary(text, mid);
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    if best == 0 {
        return (None, true);
    }
    (Some(candidate(best)), true)
}

/// One source's full text plus its insights, as context for source chat.
///
/// With `max_tokens`, the source text is kept in full when it fits and
/// insights are added in fetch order while space remains. When the source
/// alone is over budget, a [`SOURCE_INSIGHT_BUDGET_RATIO`] share is reserved
/// for insights and the source text is truncated into the rest. A missing
/// source gives an empty context with status `not_found`.
pub async fn build_source_context(state: &AppState, source_id: &str, max_tokens: Option<usize>) -> AppResult<SourceContext> {
    let fail = |e: AppError| AppError::database(format!("Failed to build context: {e}"));
    let source = match Source::get(state, &with_table(Source::TABLE, source_id)).await {
        Ok(source) => Some(source),
        Err(AppError::NotFound(_) | AppError::InvalidInput(_)) => None,
        Err(e) => return Err(fail(e)),
    };

    let mut sources = Vec::new();
    let mut insights: Vec<ContextInsight> = Vec::new();
    let mut truncated = false;
    let mut status = SourceTextStatus::NotFound;
    let mut total_tokens = 0;

    if let Some(source) = source {
        let items: Vec<ContextInsight> = source
            .insights(state)
            .await
            .map_err(fail)?
            .into_iter()
            .map(|i| ContextInsight { id: i.id, source_id: source.id.clone(), insight_type: i.insight_type, content: i.content })
            .collect();
        let context = ContextSource { id: source.id.clone(), title: source.title.clone(), full_text: source.full_text.clone() };
        let mut budgeted = Some(context.clone());

        match max_tokens {
            None => insights = items,
            Some(max) if rendered_tokens(Some(&context), &[]) > max => {
                let insight_budget = (max as f64 * SOURCE_INSIGHT_BUDGET_RATIO) as usize;
                for item in items {
                    insights.push(item);
                    if rendered_tokens(None, &insights) > insight_budget {
                        insights.pop();
                    }
                }
                let (cut, was_cut) = truncate_source_to_budget(&context, max, &insights);
                budgeted = cut;
                truncated = was_cut;
            }
            Some(max) => {
                for item in items {
                    insights.push(item);
                    if rendered_tokens(Some(&context), &insights) > max {
                        insights.pop();
                    }
                }
            }
        }

        status = match &budgeted {
            None => SourceTextStatus::OmittedBudget,
            Some(s) if s.full_text.as_deref().is_some_and(|t| !t.trim().is_empty()) => {
                if truncated {
                    SourceTextStatus::Truncated
                } else {
                    SourceTextStatus::Available
                }
            }
            Some(_) => SourceTextStatus::Missing,
        };
        total_tokens = rendered_tokens(budgeted.as_ref(), &insights);
        sources.extend(budgeted);
    } else {
        tracing::warn!("Source {source_id} not found");
    }

    let total_items = sources.len() + insights.len();
    tracing::info!("Built context with {total_items} items, {total_tokens} tokens");
    Ok(SourceContext {
        metadata: SourceContextMetadata {
            source_count: sources.len(),
            note_count: 0,
            insight_count: insights.len(),
            source_text_status: status,
            source_truncated: truncated,
        },
        sources,
        notes: Vec::new(),
        insights,
        total_tokens,
        total_items,
    })
}

#[cfg(test)]
mod tests {
    use axum::http::StatusCode;
    use serde_json::json;

    use super::*;
    use crate::db::Doc;
    use crate::testing::TestApp;

    #[test]
    fn python_repr_matches_str_of_a_dict() {
        let value = json!({ "id": "source:s2", "insights": ["i"], "title": null, "n": 1, "f": 1.0, "ok": true });
        assert_eq!(py_repr(&value), "{'id': 'source:s2', 'insights': ['i'], 'title': None, 'n': 1, 'f': 1.0, 'ok': True}");
        assert_eq!(py_str_repr("it's"), "\"it's\"");
        assert_eq!(py_str_repr("a'b\"c\n\\"), "'a\\'b\"c\\n\\\\'");
    }

    #[test]
    fn status_maps_keep_their_order() {
        let config: ContextConfig =
            serde_json::from_str(r#"{"sources": {"z": "insights", "a": "full content", "m": 3}, "notes": {}}"#).unwrap();
        let sources = config.sources.clone().unwrap();
        assert_eq!(sources.0, vec![("z".to_string(), "insights".to_string()), ("a".into(), "full content".into())]);
        assert!(!config.is_empty());
        assert!(serde_json::from_str::<ContextConfig>("{}").unwrap().is_empty());
        assert_eq!(serde_json::to_value(&config).unwrap(), json!({ "sources": { "z": "insights", "a": "full content" }, "notes": {} }));
    }

    async fn source(app: &TestApp, title: &str, text: &str) -> Source {
        Source::create(&app.state, &Source { title: Some(title.into()), full_text: Some(text.into()), ..Source::default() })
            .await
            .unwrap()
    }

    async fn insight(app: &TestApp, source: &Source, kind: &str, content: &str) -> String {
        let row = app
            .state
            .db
            .create("source_insight", Doc::new().record("source", &source.id).set("insight_type", kind).set("content", content))
            .await
            .unwrap();
        row["id"].as_str().unwrap().to_string()
    }

    #[tokio::test]
    async fn context_endpoint_follows_the_status_strings() {
        let app = TestApp::new().await;
        let (_, nb) = app.post("/api/notebooks", json!({ "name": "NB" })).await;
        let nb = nb["id"].as_str().unwrap().to_string();
        let skipped = source(&app, "Skipped", "skip me").await;
        let short = source(&app, "Short", "short body").await;
        let long = source(&app, "Long", "the long body").await;
        insight(&app, &short, "summary", "a summary").await;
        let note = Note::create(&app.state, Some("N"), Some("note body"), Some("human")).await.unwrap();
        let other_note = Note::create(&app.state, Some("O"), Some("other"), Some("human")).await.unwrap();
        let bare = |id: &str| id.split_once(':').unwrap().1.to_string();

        let (status, body) = app
            .post(
                "/api/chat/context",
                json!({
                    "notebook_id": nb,
                    "context_config": {
                        "sources": { bare(&skipped.id): "not in context", bare(&short.id): "insights", long.id.clone(): "full content", "source:gone": "insights" },
                        "notes": { other_note.id.clone(): "not in context", bare(&note.id): "full content" },
                    },
                }),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let sources = body["context"]["sources"].as_array().unwrap();
        assert_eq!(sources.len(), 2, "{body}");
        assert_eq!(sources[0]["id"], short.id);
        assert_eq!(sources[0]["insights"][0]["content"], "a summary");
        assert!(sources[0].get("full_text").is_none(), "insights is the short context");
        assert_eq!(sources[1]["id"], long.id);
        assert_eq!(sources[1]["full_text"], "the long body");
        assert_eq!(body["context"]["notes"], json!([{ "id": note.id, "title": "N", "content": "note body" }]));

        let expected: String = sources.iter().chain(body["context"]["notes"].as_array().unwrap()).map(py_repr).collect();
        assert_eq!(body["char_count"], expected.chars().count());
        assert_eq!(body["token_count"], nn_content::token_count(&expected));

        // An explicit but empty config includes nothing.
        let (_, body) =
            app.post("/api/chat/context", json!({ "notebook_id": nb, "context_config": { "sources": {}, "notes": {} } })).await;
        assert_eq!(body, json!({ "context": { "sources": [], "notes": [] }, "token_count": 0, "char_count": 0 }));
    }

    #[tokio::test]
    async fn empty_config_includes_every_source_and_note_short() {
        let app = TestApp::new().await;
        let (_, nb) = app.post("/api/notebooks", json!({ "name": "NB" })).await;
        let nb = nb["id"].as_str().unwrap().to_string();
        let s = source(&app, "S", "full text").await;
        s.add_to_notebook(&app.state, &nb).await.unwrap();
        insight(&app, &s, "summary", "sum").await;
        let note = Note::create(&app.state, Some("N"), Some("note body"), None).await.unwrap();
        note.add_to_notebook(&app.state, &nb).await.unwrap();

        let (status, body) = app.post("/api/chat/context", json!({ "notebook_id": nb, "context_config": {} })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["context"]["sources"].as_array().unwrap().len(), 1);
        assert_eq!(body["context"]["sources"][0]["insights"][0]["content"], "sum");
        assert!(body["context"]["sources"][0].get("full_text").is_none());
        // The notebook listing omits note bodies, so the short note context has none.
        assert_eq!(body["context"]["notes"], json!([{ "id": note.id, "title": "N", "content": null }]));
        assert!(body["token_count"].as_u64().unwrap() > 0);
    }

    #[tokio::test]
    async fn context_endpoint_404s_and_validates() {
        let app = TestApp::new().await;
        let s = source(&app, "S", "x").await;
        for id in ["notebook:missing", s.id.as_str()] {
            let (status, body) = app.post("/api/chat/context", json!({ "notebook_id": id, "context_config": {} })).await;
            assert_eq!(status, StatusCode::NOT_FOUND, "{id}");
            assert_eq!(body["detail"], "Notebook not found");
        }
        let (status, body) = app.post("/api/chat/context", json!({ "notebook_id": "notebook:x" })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(body["detail"][0]["loc"], json!(["body"]));
        let (status, _) = app.post("/api/chat/context", json!({ "notebook_id": "notebook:x", "context_config": [] })).await;
        assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[tokio::test]
    async fn source_context_fits_the_budget() {
        let app = TestApp::new().await;
        let text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ".repeat(400);
        let s = source(&app, "Big", &text).await;
        let small_insight = insight(&app, &s, "summary", "short insight").await;
        insight(&app, &s, "huge", &"word ".repeat(2000)).await;

        // No budget: everything.
        let all = build_source_context(&app.state, &s.id, None).await.unwrap();
        assert_eq!(all.insights.len(), 2);
        assert_eq!(all.metadata.source_text_status, SourceTextStatus::Available);

        // Plenty of budget: source and every insight that fits.
        let roomy = build_source_context(&app.state, &s.id, Some(50_000)).await.unwrap();
        assert_eq!(roomy.sources[0].full_text.as_deref(), Some(text.as_str()));
        assert_eq!(roomy.insights.len(), 2);

        // A tight budget: the source is cut, the small insight kept, the total fits.
        let tight = build_source_context(&app.state, &s.id, Some(1000)).await.unwrap();
        assert_eq!(tight.metadata.source_text_status, SourceTextStatus::Truncated);
        assert!(tight.metadata.source_truncated);
        let cut = tight.sources[0].full_text.as_deref().unwrap();
        assert!(cut.ends_with(SOURCE_TRUNCATION_NOTICE));
        assert!(cut.len() < text.len());
        assert_eq!(tight.insights.iter().map(|i| i.id.as_str()).collect::<Vec<_>>(), [small_insight.as_str()]);
        assert!(tight.total_tokens <= 1000, "{}", tight.total_tokens);
        assert_eq!(tight.total_tokens, nn_content::token_count(&tight.formatted()));
        // Near-maximal: the cut leaves little room unused.
        assert!(tight.total_tokens > 900, "{}", tight.total_tokens);
        let formatted = tight.formatted();
        assert!(formatted.starts_with("## SOURCE CONTENT\n**Source ID:** source:"), "{formatted}");
        assert!(formatted.contains("## SOURCE INSIGHTS"));

        // A budget too small for even the notice omits the source.
        let none = build_source_context(&app.state, &s.id, Some(5)).await.unwrap();
        assert_eq!(none.metadata.source_text_status, SourceTextStatus::OmittedBudget);
        assert!(none.sources.is_empty());

        let missing = build_source_context(&app.state, "source:gone", Some(100)).await.unwrap();
        assert_eq!(missing.metadata.source_text_status, SourceTextStatus::NotFound);
        assert_eq!(missing.total_items, 0);

        let empty = Source::create(&app.state, &Source { title: Some("Empty".into()), ..Source::default() }).await.unwrap();
        let ctx = build_source_context(&app.state, &empty.id, Some(100)).await.unwrap();
        assert_eq!(ctx.metadata.source_text_status, SourceTextStatus::Missing);
        assert!(ctx.formatted().contains("[Source text is unavailable in this context.]"));
    }
}
