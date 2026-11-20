//! Search and Ask: the Rust counterpart of `api/routers/search.py` and
//! `open_notebook/graphs/ask.py`.
//!
//! - [`search`] runs a keyword (BM25) or vector search, optionally scoped to
//!   notebooks (ADR-008).
//! - Ask answers a question in three stages, as the Python LangGraph did:
//!   the strategy model plans up to five searches (`ask/entry`, JSON out),
//!   the answer model answers each search from its vector-search hits
//!   (`ask/query_process`, run concurrently), and the final-answer model
//!   writes the answer from those (`ask/final_answer`). [`ask_stream`] emits
//!   the stages as [`AskEvent`]s; [`ask_simple`] returns only the answer.

use futures::stream::BoxStream;
use futures::StreamExt;
use nn_llm::{ChatMessage, ChatOptions};
use serde::{Deserialize, Serialize};

use crate::db::Json;
use crate::domain::search::{resolve_notebook_scope, text_search, vector_search};
use crate::domain::{DefaultModelType, Model};
use crate::error::{AppError, AppResult};
use crate::jobs::JobQueue;
use crate::services::context::{provision_language_model, py_repr, py_str_repr};
use crate::state::AppState;

/// Search has no background jobs.
pub fn register_jobs(_jobs: &JobQueue) {}

/// Output budget of each Ask stage (Python `ASK_MAX_TOKENS`).
pub const ASK_MAX_TOKENS: u32 = 8192;
/// Hits per Ask search.
const ASK_SEARCH_RESULTS: usize = 10;
/// Minimum similarity of an Ask search hit (Python's `vector_search` default).
const ASK_MINIMUM_SCORE: f64 = 0.2;
/// Most notebooks one request may be scoped to.
pub const MAX_SCOPE_NOTEBOOKS: usize = 50;

const NO_SEARCH_TERMS: &str = "The strategy model returned no search terms for this question. This usually means the model \
spent its output budget on reasoning or returned an empty response. Pick a different strategy model in the Ask page's \
advanced model options, or rephrase the question.";
const EMPTY_FINAL_ANSWER: &str = "The final answer model returned an empty response. Try again, or pick a different final \
answer model in the Ask page's advanced model options.";

// ---------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------

/// `NotebookScopeMixin`: `notebook_id` and / or `notebook_ids`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotebookScope {
    #[serde(default)]
    pub notebook_id: Option<String>,
    #[serde(default)]
    pub notebook_ids: Option<Vec<String>>,
}

impl NotebookScope {
    /// Both fields merged, duplicates dropped, order kept. Empty strings
    /// stay, so validation rejects them instead of widening the scope.
    pub fn ids(&self) -> Vec<String> {
        let mut merged: Vec<String> = Vec::new();
        for id in self.notebook_id.iter().chain(self.notebook_ids.iter().flatten()) {
            if !merged.contains(id) {
                merged.push(id.clone());
            }
        }
        merged
    }

    fn validate(&self) -> Result<(), String> {
        match &self.notebook_ids {
            Some(ids) if ids.len() > MAX_SCOPE_NOTEBOOKS => {
                Err(format!("notebook_ids: List should have at most {MAX_SCOPE_NOTEBOOKS} items after validation, not {}", ids.len()))
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchType {
    #[default]
    Text,
    Vector,
}

impl SearchType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Text => "text",
            Self::Vector => "vector",
        }
    }
}

fn default_limit() -> i64 {
    100
}
fn default_true() -> bool {
    true
}
fn default_minimum_score() -> f64 {
    0.2
}

/// `SearchRequest`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default, rename = "type")]
    pub search_type: SearchType,
    /// 1..=1000.
    #[serde(default = "default_limit")]
    pub limit: i64,
    #[serde(default = "default_true")]
    pub search_sources: bool,
    #[serde(default = "default_true")]
    pub search_notes: bool,
    /// 0..=1, vector search only.
    #[serde(default = "default_minimum_score")]
    pub minimum_score: f64,
    #[serde(flatten)]
    pub scope: NotebookScope,
}

impl SearchRequest {
    /// The field constraints FastAPI enforced (answered with a 422).
    pub fn validate(&self) -> Result<(), String> {
        if self.limit < 1 {
            return Err("limit: Input should be greater than or equal to 1".into());
        }
        if self.limit > 1000 {
            return Err("limit: Input should be less than or equal to 1000".into());
        }
        if !(0.0..=1.0).contains(&self.minimum_score) {
            return Err("minimum_score: Input should be between 0 and 1".into());
        }
        self.scope.validate()
    }
}

/// `SearchResponse`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SearchResponse {
    pub results: Vec<Json>,
    pub total_count: usize,
    pub search_type: String,
}

/// `AskRequest`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AskRequest {
    pub question: String,
    pub strategy_model: String,
    pub answer_model: String,
    pub final_answer_model: String,
    #[serde(flatten)]
    pub scope: NotebookScope,
}

impl AskRequest {
    pub fn validate(&self) -> Result<(), String> {
        self.scope.validate()
    }
}

/// `AskResponse`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AskResponse {
    pub answer: String,
    pub question: String,
}

// ---------------------------------------------------------------------------
// Search
// ---------------------------------------------------------------------------

/// A database failure as the search router reported it: `Search failed: …`,
/// without the driver's own text for a failed scope check.
fn search_failed(e: AppError) -> AppError {
    match e {
        AppError::Database { message, .. } => {
            let message = match message.find(": ") {
                Some(i) if message.starts_with("Failed to resolve notebook scope") => message[..i].to_string(),
                _ => message,
            };
            AppError::internal(format!("Search failed: {message}"))
        }
        other => other,
    }
}

/// `POST /api/search`.
pub async fn search(state: &AppState, request: &SearchRequest) -> AppResult<SearchResponse> {
    let notebook_ids = resolve_notebook_scope(state, &request.scope.ids()).await.map_err(search_failed)?;
    let limit = request.limit.clamp(1, 1000) as usize;
    let results = match request.search_type {
        SearchType::Vector => {
            if state.models.embedding_opt(state).await?.is_none() {
                return Err(AppError::bad_request(
                    "Vector search requires an embedding model. Please configure one in the Models section.",
                ));
            }
            vector_search(
                state,
                &request.query,
                limit,
                request.search_sources,
                request.search_notes,
                request.minimum_score,
                Some(&notebook_ids),
            )
            .await
        }
        SearchType::Text => {
            text_search(state, &request.query, limit, request.search_sources, request.search_notes, Some(&notebook_ids)).await
        }
    }
    .map_err(search_failed)?;
    Ok(SearchResponse { total_count: results.len(), results, search_type: request.search_type.as_str().into() })
}

// ---------------------------------------------------------------------------
// Ask
// ---------------------------------------------------------------------------

/// One planned search.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AskSearch {
    #[serde(default, deserialize_with = "crate::domain::null_default")]
    pub term: String,
    #[serde(default, deserialize_with = "crate::domain::null_default")]
    pub instructions: String,
}

/// The strategy model's plan.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AskStrategy {
    #[serde(default, deserialize_with = "crate::domain::null_default")]
    pub reasoning: String,
    #[serde(default, deserialize_with = "crate::domain::null_default")]
    pub searches: Vec<AskSearch>,
}

impl AskStrategy {
    /// Python's `str(Strategy)`, as the final-answer prompt shows it.
    fn py_str(&self) -> String {
        let searches: Vec<String> = self
            .searches
            .iter()
            .map(|s| format!("Search(term={}, instructions={})", py_str_repr(&s.term), py_str_repr(&s.instructions)))
            .collect();
        format!("reasoning={} searches=[{}]", py_str_repr(&self.reasoning), searches.join(", "))
    }
}

/// One server-sent event of an Ask (`data: <json>`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum AskEvent {
    Strategy { reasoning: String, searches: Vec<AskSearch> },
    Answer { content: String },
    FinalAnswer { content: String },
    Complete { final_answer: Option<String> },
    Error { message: String },
}

/// A validated Ask, ready to run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AskPlan {
    pub question: String,
    pub strategy_model: String,
    pub answer_model: String,
    pub final_answer_model: String,
    /// Empty: the whole knowledge base.
    pub notebook_ids: Vec<String>,
}

/// Validate an Ask before anything runs: the scope first (cheapest), then
/// the three models, then that an embedding model is configured.
pub async fn prepare_ask(state: &AppState, request: &AskRequest) -> AppResult<AskPlan> {
    let notebook_ids = resolve_notebook_scope(state, &request.scope.ids()).await?;
    let strategy = Model::get(state, &request.strategy_model).await?;
    let answer = Model::get(state, &request.answer_model).await?;
    let final_answer = Model::get(state, &request.final_answer_model).await?;
    if state.models.embedding_opt(state).await?.is_none() {
        return Err(AppError::bad_request(
            "Ask feature requires an embedding model. Please configure one in the Models section.",
        ));
    }
    Ok(AskPlan {
        question: request.question.clone(),
        strategy_model: strategy.id,
        answer_model: answer.id,
        final_answer_model: final_answer.id,
        notebook_ids,
    })
}

const STRATEGY_SCHEMA: &str = r##"{"$defs": {"Search": {"properties": {"term": {"title": "Term", "type": "string"}, "instructions": {"description": "Tell the answeting LLM what information you need extracted from this search", "title": "Instructions", "type": "string"}}, "required": ["term", "instructions"], "title": "Search", "type": "object"}}, "properties": {"reasoning": {"title": "Reasoning", "type": "string"}, "searches": {"description": "You can add up to five searches to this strategy", "items": {"$ref": "#/$defs/Search"}, "title": "Searches", "type": "array"}}, "required": ["reasoning"]}"##;

/// LangChain's `PydanticOutputParser(Strategy).get_format_instructions()`.
fn strategy_format_instructions() -> String {
    format!(
        "The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\n\
As an example, for the schema {{\"properties\": {{\"foo\": {{\"title\": \"Foo\", \"description\": \"a list of strings\", \"type\": \"array\", \"items\": {{\"type\": \"string\"}}}}}}, \"required\": [\"foo\"]}}\n\
the object {{\"foo\": [\"bar\", \"baz\"]}} is a well-formatted instance of the schema. The object {{\"properties\": {{\"foo\": [\"bar\", \"baz\"]}}}} is not well-formatted.\n\n\
Here is the output schema:\n```\n{STRATEGY_SCHEMA}\n```"
    )
}

/// Parse the strategy JSON leniently: code fences and text around the
/// object are ignored.
fn parse_strategy(text: &str) -> AppResult<AskStrategy> {
    let trimmed = text.trim();
    let unfenced = trimmed
        .strip_prefix("```json")
        .or_else(|| trimmed.strip_prefix("```"))
        .map(|rest| rest.trim_end().trim_end_matches("```"))
        .unwrap_or(trimmed);
    if let Ok(strategy) = serde_json::from_str::<AskStrategy>(unfenced.trim()) {
        return Ok(strategy);
    }
    if let (Some(start), Some(end)) = (unfenced.find('{'), unfenced.rfind('}')) {
        if start < end {
            if let Ok(strategy) = serde_json::from_str::<AskStrategy>(&unfenced[start..=end]) {
                return Ok(strategy);
            }
        }
    }
    Err(AppError::ExternalService(
        "The strategy model did not return a valid JSON search strategy. Pick a different strategy model, or try again.".into(),
    ))
}

/// Send one prompt as a single user message and return the answer without
/// thinking content.
async fn ask_model(state: &AppState, prompt: &str, model_id: &str, json: bool) -> AppResult<String> {
    let model = provision_language_model(state, prompt, Some(model_id), DefaultModelType::Tools).await?;
    let options = ChatOptions { json, ..ChatOptions::default() }.with_max_tokens(ASK_MAX_TOKENS);
    let reply = model.chat(&[ChatMessage::user(prompt)], &options).await?;
    Ok(nn_content::clean_thinking_content(&reply.content))
}

/// Stage 1: plan the searches. Blank terms are dropped; none left is an
/// error rather than a silent "no documents found".
pub async fn plan_strategy(state: &AppState, plan: &AskPlan) -> AppResult<AskStrategy> {
    let prompt = state.prompts.render(
        "ask/entry",
        serde_json::json!({ "question": plan.question, "format_instructions": strategy_format_instructions() }),
    )?;
    let reply = ask_model(state, &prompt, &plan.strategy_model, true).await?;
    let mut strategy = parse_strategy(&reply)?;
    strategy.searches.retain(|s| !s.term.trim().is_empty());
    if strategy.searches.is_empty() {
        return Err(AppError::ExternalService(NO_SEARCH_TERMS.into()));
    }
    Ok(strategy)
}

/// Stage 2: answer one search from its vector-search hits. `None` when
/// nothing was found or the model said nothing beyond its thinking.
pub async fn answer_search(state: &AppState, plan: &AskPlan, search: &AskSearch) -> AppResult<Option<String>> {
    let scope = (!plan.notebook_ids.is_empty()).then_some(plan.notebook_ids.as_slice());
    let results = vector_search(state, &search.term, ASK_SEARCH_RESULTS, true, true, ASK_MINIMUM_SCORE, scope).await?;
    if results.is_empty() {
        return Ok(None);
    }
    let ids: Vec<Json> = results.iter().map(|r| r.get("id").cloned().unwrap_or(Json::Null)).collect();
    let prompt = state.prompts.render(
        "ask/query_process",
        serde_json::json!({
            "question": plan.question,
            "term": search.term,
            "instructions": search.instructions,
            "results": py_repr(&Json::Array(results)),
            "ids": py_repr(&Json::Array(ids)),
        }),
    )?;
    let answer = ask_model(state, &prompt, &plan.answer_model, false).await?;
    Ok((!answer.trim().is_empty()).then_some(answer))
}

/// Stage 3: the final answer from the strategy and the partial answers.
pub async fn write_final_answer(state: &AppState, plan: &AskPlan, strategy: &AskStrategy, answers: &[String]) -> AppResult<String> {
    let answers_json = Json::Array(answers.iter().cloned().map(Json::String).collect());
    let prompt = state.prompts.render(
        "ask/final_answer",
        serde_json::json!({ "question": plan.question, "strategy": strategy.py_str(), "answers": py_repr(&answers_json) }),
    )?;
    let answer = ask_model(state, &prompt, &plan.final_answer_model, false).await?;
    if answer.trim().is_empty() {
        return Err(AppError::IncompleteGeneration(EMPTY_FINAL_ANSWER.into()));
    }
    Ok(answer)
}

/// Every search answered concurrently, in plan order.
fn answer_all(state: AppState, plan: AskPlan, searches: Vec<AskSearch>) -> BoxStream<'static, AppResult<Option<String>>> {
    futures::stream::iter(searches)
        .map(move |search| {
            let state = state.clone();
            let plan = plan.clone();
            async move { answer_search(&state, &plan, &search).await }
        })
        .buffered(5)
        .boxed()
}

/// Run an Ask as the event stream the Ask page consumes: `strategy`, one
/// `answer` per answered search, `final_answer`, then `complete`; or an
/// `error` event, after which the stream ends.
pub fn ask_stream(state: AppState, plan: AskPlan) -> BoxStream<'static, AskEvent> {
    async_stream::stream! {
        let strategy = match plan_strategy(&state, &plan).await {
            Ok(strategy) => strategy,
            Err(e) => {
                tracing::error!("Error in ask streaming: {e}");
                yield AskEvent::Error { message: e.detail() };
                return;
            }
        };
        yield AskEvent::Strategy { reasoning: strategy.reasoning.clone(), searches: strategy.searches.clone() };

        let mut answers = Vec::new();
        let mut partials = answer_all(state.clone(), plan.clone(), strategy.searches.clone());
        while let Some(result) = partials.next().await {
            match result {
                Ok(Some(answer)) => {
                    answers.push(answer.clone());
                    yield AskEvent::Answer { content: answer };
                }
                Ok(None) => {}
                Err(e) => {
                    tracing::error!("Error in ask streaming: {e}");
                    yield AskEvent::Error { message: e.detail() };
                    return;
                }
            }
        }

        match write_final_answer(&state, &plan, &strategy, &answers).await {
            Ok(final_answer) => {
                yield AskEvent::FinalAnswer { content: final_answer.clone() };
                yield AskEvent::Complete { final_answer: Some(final_answer) };
            }
            Err(e) => {
                tracing::error!("Error in ask streaming: {e}");
                yield AskEvent::Error { message: e.detail() };
            }
        }
    }
    .boxed()
}

/// `POST /api/search/ask/simple`: the whole Ask, answer only.
pub async fn ask_simple(state: &AppState, request: &AskRequest) -> AppResult<AskResponse> {
    let plan = prepare_ask(state, request).await?;
    let strategy = plan_strategy(state, &plan).await?;
    let mut answers = Vec::new();
    let mut partials = answer_all(state.clone(), plan.clone(), strategy.searches.clone());
    while let Some(result) = partials.next().await {
        answers.extend(result?);
    }
    let answer = write_final_answer(state, &plan, &strategy, &answers).await?;
    if answer.is_empty() {
        return Err(AppError::internal("No answer generated"));
    }
    Ok(AskResponse { answer, question: request.question.clone() })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use axum::http::{Method, StatusCode};
    use serde_json::{json, Value};

    use super::*;
    use crate::ai::AnyModel;
    use crate::db::Doc;
    use crate::domain::{Note, Source};
    use crate::testing::{MockEmbeddingModel, MockLanguageModel, TestApp};

    fn events(body: &str) -> Vec<Value> {
        body.split("\n\n")
            .filter(|chunk| !chunk.is_empty())
            .map(|chunk| serde_json::from_str(chunk.strip_prefix("data: ").expect("data line")).unwrap())
            .collect()
    }

    async fn notebook(app: &TestApp, name: &str) -> String {
        let (_, nb) = app.post("/api/notebooks", json!({ "name": name })).await;
        nb["id"].as_str().unwrap().to_string()
    }

    /// A source with one embedded chunk, in `notebook`.
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

    async fn language_model(app: &TestApp) -> String {
        let row = app.state.db.create("model", Doc::new().set("name", "m").set("provider", "openai").set("type", "language")).await.unwrap();
        row["id"].as_str().unwrap().to_string()
    }

    fn ask_body(model: &str, scope: Value) -> Value {
        let mut body = json!({ "question": "What do waves carry?", "strategy_model": model, "answer_model": model, "final_answer_model": model });
        body.as_object_mut().unwrap().extend(scope.as_object().unwrap().clone());
        body
    }

    #[test]
    fn scope_merges_and_dedupes() {
        let scope = NotebookScope { notebook_id: Some("notebook:a".into()), notebook_ids: Some(vec!["notebook:b".into(), "notebook:a".into()]) };
        assert_eq!(scope.ids(), ["notebook:a", "notebook:b"]);
        assert!(NotebookScope::default().ids().is_empty());
    }

    #[test]
    fn strategy_parsing_is_lenient() {
        let s = parse_strategy("```json\n{\"reasoning\": \"r\", \"searches\": [{\"term\": \"t\", \"instructions\": \"i\"}]}\n```").unwrap();
        assert_eq!(s.searches[0].term, "t");
        let s = parse_strategy("Sure! {\"reasoning\": \"r\", \"searches\": []} hope that helps").unwrap();
        assert!(s.searches.is_empty());
        assert!(parse_strategy("no json here").is_err());
        let s = AskStrategy { reasoning: "it's".into(), searches: vec![AskSearch { term: "RAG".into(), instructions: "x".into() }] };
        assert_eq!(s.py_str(), "reasoning=\"it's\" searches=[Search(term='RAG', instructions='x')]");
        assert!(strategy_format_instructions().contains("\"title\": \"Searches\""));
    }

    #[tokio::test]
    async fn search_validation_and_scope_errors() {
        let app = TestApp::new().await;
        for body in [
            json!({ "query": "x", "limit": 0 }),
            json!({ "query": "x", "limit": -5 }),
            json!({ "query": "x", "limit": 1001 }),
            json!({ "query": "x", "minimum_score": 1.5 }),
            json!({ "query": "x", "type": "fuzzy" }),
            json!({ "limit": 5 }),
            json!({ "query": "x", "notebook_ids": (0..51).map(|i| format!("notebook:{i}")).collect::<Vec<_>>() }),
        ] {
            let (status, response) = app.post("/api/search", body.clone()).await;
            assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY, "{body} -> {response}");
            assert!(response["detail"].is_array());
        }
        let (status, body) = app.post("/api/search", json!({ "query": "x", "notebook_ids": ["notebook:zzz"] })).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(body["detail"].as_str().unwrap().contains("notebook:zzz"));
        for scope in [json!({ "notebook_ids": ["source:abc"] }), json!({ "notebook_id": "" })] {
            let mut body = json!({ "query": "x" });
            body.as_object_mut().unwrap().extend(scope.as_object().unwrap().clone());
            let (status, _) = app.post("/api/search", body).await;
            assert_eq!(status, StatusCode::BAD_REQUEST);
        }
        let (status, body) = app.post("/api/search", json!({ "query": "x", "type": "vector" })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["detail"], "Vector search requires an embedding model. Please configure one in the Models section.");
        let (status, body) = app.post("/api/search", json!({ "query": "" })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["detail"], "Search keyword cannot be empty");
    }

    #[tokio::test]
    async fn text_and_vector_search_respect_the_notebook_scope() {
        let app = TestApp::new().await;
        let embedder = MockEmbeddingModel::new();
        app.state.models.set_override(AnyModel::Embedding(embedder.clone()));
        let physics = notebook(&app, "Physics").await;
        let music = notebook(&app, "Music").await;
        let waves = embedded_source(&app, &embedder, &physics, "Ocean waves", "ocean waves carry energy").await;
        let sound = embedded_source(&app, &embedder, &music, "Sound waves", "sound waves carry music").await;
        let note = Note::create(&app.state, Some("Wave note"), Some("waves in a note"), Some("human")).await.unwrap();
        note.add_to_notebook(&app.state, &physics).await.unwrap();

        let ids = |body: &Value| -> Vec<String> {
            body["results"].as_array().unwrap().iter().map(|r| r["id"].as_str().unwrap_or_default().to_string()).collect()
        };

        let (status, body) = app.post("/api/search", json!({ "query": "waves", "type": "vector", "minimum_score": 0.1 })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["search_type"], "vector");
        let all = ids(&body);
        assert!(all.contains(&waves.id) && all.contains(&sound.id), "{body}");
        assert_eq!(body["total_count"], all.len());

        let (_, body) =
            app.post("/api/search", json!({ "query": "waves", "type": "vector", "minimum_score": 0.1, "notebook_id": music })).await;
        assert_eq!(ids(&body), std::slice::from_ref(&sound.id), "{body}");

        let (status, body) = app.post("/api/search", json!({ "query": "waves", "notebook_ids": [physics] })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["search_type"], "text");
        let scoped = ids(&body);
        assert!(!scoped.is_empty(), "{body}");
        assert!(!scoped.contains(&sound.id), "{body}");

        let (_, body) = app.post("/api/search", json!({ "query": "waves", "search_notes": false })).await;
        assert!(!ids(&body).contains(&note.id));
    }

    #[tokio::test]
    async fn ask_streams_every_stage() {
        let app = TestApp::new().await;
        let embedder = MockEmbeddingModel::new();
        app.state.models.set_override(AnyModel::Embedding(embedder.clone()));
        let physics = notebook(&app, "Physics").await;
        let music = notebook(&app, "Music").await;
        let waves = embedded_source(&app, &embedder, &physics, "Ocean waves", "ocean waves carry energy").await;
        let sound = embedded_source(&app, &embedder, &music, "Sound waves", "sound waves carry music").await;
        let strategy = json!({ "reasoning": "Look for waves", "searches": [
            { "term": "waves carry", "instructions": "What do they carry?" },
            { "term": "  ", "instructions": "blank, dropped" },
        ] });
        let llm = MockLanguageModel::new([
            format!("<think>plan</think>```json\n{strategy}\n```"),
            "Waves carry energy [source:x]".to_string(),
            "Final: energy".to_string(),
        ]);
        app.state.models.set_override(AnyModel::Language(llm.clone()));
        let model = language_model(&app).await;

        let (status, body) = app.text(Method::POST, "/api/search/ask", Some(ask_body(&model, json!({ "notebook_ids": [physics] })))).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        let e = events(&body);
        assert_eq!(
            e,
            [
                json!({ "type": "strategy", "reasoning": "Look for waves", "searches": [{ "term": "waves carry", "instructions": "What do they carry?" }] }),
                json!({ "type": "answer", "content": "Waves carry energy [source:x]" }),
                json!({ "type": "final_answer", "content": "Final: energy" }),
                json!({ "type": "complete", "final_answer": "Final: energy" }),
            ]
        );
        assert_eq!(llm.call_count(), 3);
        assert!(llm.options.lock().unwrap()[0].json, "the strategy asks for JSON");
        assert!(llm.options.lock().unwrap().iter().all(|o| o.max_tokens == Some(ASK_MAX_TOKENS)));
        assert!(llm.prompt(0).contains("# USER QUESTION\n\nWhat do waves carry?"));
        assert!(llm.prompt(0).contains("Here is the output schema"));
        // The per-search prompt holds only the scoped hits.
        let answer_prompt = llm.prompt(1);
        assert!(answer_prompt.contains(&waves.id), "{answer_prompt}");
        assert!(!answer_prompt.contains(&sound.id), "{answer_prompt}");
        assert!(answer_prompt.contains("What do they carry?"));
        let final_prompt = llm.prompt(2);
        assert!(final_prompt.contains("reasoning='Look for waves' searches=[Search(term='waves carry'"), "{final_prompt}");
        assert!(final_prompt.contains("['Waves carry energy [source:x]']"));
    }

    #[tokio::test]
    async fn ask_simple_and_failures() {
        let app = TestApp::new().await;
        let embedder = MockEmbeddingModel::new();
        let model = language_model(&app).await;
        let strategy = json!({ "reasoning": "r", "searches": [{ "term": "nothing matches", "instructions": "i" }] }).to_string();

        // Validation order: scope, models, embedding model.
        let (status, _) = app.post("/api/search/ask", ask_body(&model, json!({ "notebook_ids": ["notebook:zzz"] }))).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        let mut body = ask_body(&model, json!({}));
        body["answer_model"] = json!("model:gone");
        let (status, response) = app.post("/api/search/ask/simple", body).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
        assert!(response["detail"].as_str().unwrap().contains("model:gone"));
        let (status, response) = app.post("/api/search/ask", ask_body(&model, json!({}))).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(response["detail"], "Ask feature requires an embedding model. Please configure one in the Models section.");
        app.state.models.set_override(AnyModel::Embedding(embedder.clone()));

        // No hits: no partial answers, but still a final answer.
        let llm = MockLanguageModel::new([strategy.as_str(), "The final answer"]);
        app.state.models.set_override(AnyModel::Language(llm.clone()));
        let (status, response) = app.post("/api/search/ask/simple", ask_body(&model, json!({}))).await;
        assert_eq!(status, StatusCode::OK, "{response}");
        assert_eq!(response, json!({ "answer": "The final answer", "question": "What do waves carry?" }));
        assert_eq!(llm.call_count(), 2);
        assert!(llm.prompt(1).contains("# RESULTS\n\nHere are the answers you received for each of your queries.\n\n[]"));

        // Only blank terms: a 502 for the simple endpoint, an error event when streaming.
        let blank = json!({ "reasoning": "r", "searches": [{ "term": "", "instructions": "i" }] }).to_string();
        app.state.models.set_override(AnyModel::Language(MockLanguageModel::new([blank.as_str()])));
        let (status, response) = app.post("/api/search/ask/simple", ask_body(&model, json!({}))).await;
        assert_eq!(status, StatusCode::BAD_GATEWAY);
        assert!(response["detail"].as_str().unwrap().contains("no search terms"));
        let (_, text) = app.text(Method::POST, "/api/search/ask", Some(ask_body(&model, json!({})))).await;
        let e = events(&text);
        assert_eq!(e.len(), 1);
        assert_eq!(e[0]["type"], "error");
        assert!(e[0]["message"].as_str().unwrap().contains("no search terms"));

        // An empty final answer is an error after the strategy event.
        let llm: Arc<MockLanguageModel> = MockLanguageModel::new([strategy.as_str(), "<think>only thinking</think>"]);
        app.state.models.set_override(AnyModel::Language(llm));
        let (_, text) = app.text(Method::POST, "/api/search/ask", Some(ask_body(&model, json!({})))).await;
        let kinds: Vec<String> = events(&text).iter().map(|e| e["type"].as_str().unwrap().to_string()).collect();
        assert_eq!(kinds, ["strategy", "error"]);
        assert!(text.contains("final answer model returned an empty response"));
    }

    #[tokio::test]
    async fn thinking_only_partial_answers_are_skipped() {
        let app = TestApp::new().await;
        let embedder = MockEmbeddingModel::new();
        app.state.models.set_override(AnyModel::Embedding(embedder.clone()));
        let nb = notebook(&app, "N").await;
        embedded_source(&app, &embedder, &nb, "Rag", "rag retrieval").await;
        let plan = AskPlan {
            question: "q".into(),
            strategy_model: language_model(&app).await,
            answer_model: language_model(&app).await,
            final_answer_model: language_model(&app).await,
            notebook_ids: Vec::new(),
        };
        let search = AskSearch { term: "rag retrieval".into(), instructions: "extract".into() };
        for reply in ["<think>only reasoning</think>", "<think>cut off mid"] {
            app.state.models.set_override(AnyModel::Language(MockLanguageModel::new([reply])));
            assert_eq!(answer_search(&app.state, &plan, &search).await.unwrap(), None);
        }
    }
}
