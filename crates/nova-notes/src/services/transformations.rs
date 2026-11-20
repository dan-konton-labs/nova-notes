//! Transformations: reusable prompts run against text or a source.
//!
//! The Rust counterpart of `api/routers/transformations.py` and
//! `open_notebook/graphs/transformation.py`. [`apply_transformation`] is the
//! graph: it is what `POST /transformations/execute`, the `run_transformation`
//! job and source processing all run.

use std::time::Instant;

use nn_llm::{ChatMessage, ChatOptions, SharedLanguageModel};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use super::notebooks::{double_option, ensure_table, not_found_as, timestamp};
use crate::domain::{DefaultModelType, DefaultPrompts, Model, Source, Transformation};
use crate::error::{AppError, AppResult};
use crate::jobs::{JobContext, JobQueue, RetryPolicy};
use crate::state::AppState;

pub const TRANSFORMATION_NOT_FOUND: &str = "Transformation not found";
pub const MODEL_NOT_FOUND: &str = "Model not found";

/// Above this many prompt tokens the large-context model is used instead
/// (Python's `provision_langchain_model`).
pub const LARGE_CONTEXT_THRESHOLD: usize = 105_000;
/// Output budget of a transformation run.
pub const TRANSFORMATION_MAX_TOKENS: u32 = 8192;

pub const NO_TEXT_TO_TRANSFORM: &str = "There is no text content to transform";
pub const GENERATION_LIMIT_MESSAGE: &str = "The model reached its generation limit before completing the \
     transformation. Try a shorter transformation or a different model.";
pub const NO_USABLE_TEXT_MESSAGE: &str = "The model returned no usable text for the transformation after \
     removing thinking content. Try a shorter transformation or a different model.";

/// `run_transformation`: retried up to 5 times (1-60 s backoff) unless the
/// error is permanent (validation, configuration, context length,
/// incomplete generation, missing record).
pub fn register_jobs(jobs: &JobQueue) {
    jobs.register(
        "run_transformation",
        RetryPolicy::attempts(5, std::time::Duration::from_secs(1), std::time::Duration::from_secs(60)),
        run_transformation_job,
    );
}

// ---------------------------------------------------------------------------
// Views and inputs
// ---------------------------------------------------------------------------

/// A transformation as the API reports it (`TransformationResponse`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformationView {
    pub id: String,
    pub name: String,
    pub title: String,
    pub description: String,
    pub prompt: String,
    pub apply_default: bool,
    pub model_id: Option<String>,
    pub created: String,
    pub updated: String,
}

impl From<Transformation> for TransformationView {
    fn from(t: Transformation) -> Self {
        Self {
            id: t.id,
            name: t.name,
            title: t.title,
            description: t.description,
            prompt: t.prompt,
            apply_default: t.apply_default,
            model_id: t.model_id,
            created: timestamp(t.created),
            updated: timestamp(t.updated),
        }
    }
}

/// Body of `POST /transformations` (`TransformationCreate`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NewTransformation {
    pub name: String,
    pub title: String,
    pub description: String,
    pub prompt: String,
    #[serde(default)]
    pub apply_default: bool,
    #[serde(default)]
    pub model_id: Option<String>,
}

/// Body of `PUT /transformations/{id}`. Absent fields are left alone; for
/// `description` and `model_id` an explicit `null` (or `""`) clears them.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformationUpdate {
    #[serde(default)]
    pub name: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default, deserialize_with = "double_option", skip_serializing_if = "Option::is_none")]
    pub description: Option<Option<String>>,
    #[serde(default)]
    pub prompt: Option<String>,
    #[serde(default)]
    pub apply_default: Option<bool>,
    #[serde(default, deserialize_with = "double_option", skip_serializing_if = "Option::is_none")]
    pub model_id: Option<Option<String>>,
}

/// Body of `POST /transformations/execute`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecuteTransformation {
    pub transformation_id: String,
    pub input_text: String,
    #[serde(default)]
    pub model_id: Option<String>,
}

/// Result of `POST /transformations/execute`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransformationOutput {
    pub output: String,
    pub transformation_id: String,
    /// The model the run asked for; `None` means the default transformation model.
    pub model_id: Option<String>,
}

/// The instructions prepended to every transformation (`DefaultPromptResponse`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct DefaultPrompt {
    pub transformation_instructions: String,
}

// ---------------------------------------------------------------------------
// CRUD
// ---------------------------------------------------------------------------

pub(crate) async fn load_transformation(state: &AppState, id: &str) -> AppResult<Transformation> {
    ensure_table(id, Transformation::TABLE, TRANSFORMATION_NOT_FOUND)?;
    Transformation::get(state, id).await.map_err(not_found_as(TRANSFORMATION_NOT_FOUND))
}

/// A referenced model must exist (404 otherwise).
async fn ensure_model_exists(state: &AppState, model_id: &str) -> AppResult<()> {
    ensure_table(model_id, Model::TABLE, MODEL_NOT_FOUND)?;
    Model::get(state, model_id).await.map(|_| ()).map_err(not_found_as(MODEL_NOT_FOUND))
}

/// Every transformation, by name.
pub async fn list_transformations(state: &AppState) -> AppResult<Vec<TransformationView>> {
    Ok(Transformation::list(state, Some("name asc")).await?.into_iter().map(Into::into).collect())
}

pub async fn create_transformation(state: &AppState, new: &NewTransformation) -> AppResult<TransformationView> {
    let model_id = new.model_id.clone().filter(|m| !m.is_empty());
    if let Some(model_id) = model_id.as_deref() {
        ensure_model_exists(state, model_id).await?;
    }
    let mut transformation = Transformation {
        name: new.name.clone(),
        title: new.title.clone(),
        description: new.description.clone(),
        prompt: new.prompt.clone(),
        apply_default: new.apply_default,
        model_id,
        ..Transformation::default()
    };
    transformation.save(state).await?;
    Ok(transformation.into())
}

pub async fn get_transformation(state: &AppState, id: &str) -> AppResult<TransformationView> {
    Ok(load_transformation(state, id).await?.into())
}

pub async fn update_transformation(state: &AppState, id: &str, update: &TransformationUpdate) -> AppResult<TransformationView> {
    let mut transformation = load_transformation(state, id).await?;
    if let Some(name) = &update.name {
        transformation.name = name.clone();
    }
    if let Some(title) = &update.title {
        transformation.title = title.clone();
    }
    if let Some(description) = &update.description {
        transformation.description = description.clone().unwrap_or_default();
    }
    if let Some(prompt) = &update.prompt {
        transformation.prompt = prompt.clone();
    }
    if let Some(apply_default) = update.apply_default {
        transformation.apply_default = apply_default;
    }
    if let Some(model_id) = &update.model_id {
        let model_id = model_id.clone().filter(|m| !m.is_empty());
        if let Some(model_id) = model_id.as_deref() {
            ensure_model_exists(state, model_id).await?;
        }
        transformation.model_id = model_id;
    }
    transformation.save(state).await.map_err(not_found_as(TRANSFORMATION_NOT_FOUND))?;
    Ok(transformation.into())
}

pub async fn delete_transformation(state: &AppState, id: &str) -> AppResult<()> {
    load_transformation(state, id).await?.delete(state).await
}

pub async fn get_default_prompt(state: &AppState) -> AppResult<DefaultPrompt> {
    let prompts = DefaultPrompts::load(state).await?;
    Ok(DefaultPrompt { transformation_instructions: prompts.transformation_instructions.unwrap_or_default() })
}

pub async fn update_default_prompt(state: &AppState, update: &DefaultPrompt) -> AppResult<DefaultPrompt> {
    let prompts = DefaultPrompts { transformation_instructions: Some(update.transformation_instructions.clone()) };
    let saved = prompts.save(state).await?;
    Ok(DefaultPrompt { transformation_instructions: saved.transformation_instructions.unwrap_or_default() })
}

/// Run a transformation on the given text and return its output. Nothing is
/// stored.
pub async fn execute_transformation(state: &AppState, request: &ExecuteTransformation) -> AppResult<TransformationOutput> {
    let transformation = load_transformation(state, &request.transformation_id).await?;
    let model_id = request
        .model_id
        .clone()
        .filter(|m| !m.is_empty())
        .or_else(|| transformation.model_id.clone().filter(|m| !m.is_empty()));
    if let Some(model_id) = model_id.as_deref() {
        ensure_model_exists(state, model_id).await?;
    }
    let output = apply_transformation(state, &transformation, Some(&request.input_text), None, model_id.as_deref()).await?;
    Ok(TransformationOutput { output, transformation_id: request.transformation_id.clone(), model_id })
}

// ---------------------------------------------------------------------------
// The transformation graph
// ---------------------------------------------------------------------------

/// Pick the language model for a prompt: the large-context model above
/// [`LARGE_CONTEXT_THRESHOLD`] tokens, else `model_id`, else the default for
/// `kind` (Python's `provision_langchain_model`).
pub(crate) async fn provision_language_model(
    state: &AppState,
    prompt_text: &str,
    model_id: Option<&str>,
    kind: DefaultModelType,
) -> AppResult<SharedLanguageModel> {
    let tokens = nn_content::token_count(prompt_text);
    if tokens > LARGE_CONTEXT_THRESHOLD {
        tracing::debug!("Using large context model because the content has {tokens} tokens");
        return state.models.language(state, DefaultModelType::LargeContext).await;
    }
    state.models.language_or_default(state, model_id, kind).await
}

/// Run `transformation` over `input_text` (or, when that is empty, the
/// source's full text) and return the cleaned output. With a source, the
/// output is also queued as an insight on it (`create_insight`).
///
/// No text → `InvalidInput`; a truncated or empty answer → `IncompleteGeneration`.
pub async fn apply_transformation(
    state: &AppState,
    transformation: &Transformation,
    input_text: Option<&str>,
    source: Option<&Source>,
    model_id: Option<&str>,
) -> AppResult<String> {
    let content = match input_text.filter(|t| !t.is_empty()) {
        Some(text) => text.to_string(),
        None => source.and_then(|s| s.full_text.clone()).unwrap_or_default(),
    };
    if content.trim().is_empty() {
        return Err(AppError::invalid(NO_TEXT_TO_TRANSFORM));
    }

    // The user's prompt is a value rendered into a fixed template, never
    // compiled as template source.
    let mut instructions = transformation.prompt.clone();
    match DefaultPrompts::load(state).await {
        Ok(defaults) => {
            if let Some(default) = defaults.transformation_instructions.filter(|d| !d.trim().is_empty()) {
                instructions = format!("{default}\n\n{instructions}");
            }
        }
        Err(e) => tracing::warn!("Could not load the default transformation instructions: {e}"),
    }
    let system_prompt = state
        .prompts
        .render("transformation/execute", json!({ "instructions": instructions, "input_text": content }))?;

    let messages = [ChatMessage::system(system_prompt), ChatMessage::user(content)];
    let prompt_text = format!("{}\n{}", messages[0].content, messages[1].content);
    let model = provision_language_model(state, &prompt_text, model_id, DefaultModelType::Transformation).await?;
    let response = model.chat(&messages, &ChatOptions::default().with_max_tokens(TRANSFORMATION_MAX_TOKENS)).await?;

    if response.finish_reason.as_deref().is_some_and(|r| matches!(r.to_lowercase().as_str(), "length" | "max_tokens")) {
        return Err(AppError::IncompleteGeneration(GENERATION_LIMIT_MESSAGE.to_string()));
    }
    let cleaned = nn_content::clean_thinking_content(&response.content);
    if cleaned.trim().is_empty() {
        return Err(AppError::IncompleteGeneration(NO_USABLE_TEXT_MESSAGE.to_string()));
    }

    if let Some(source) = source {
        source.add_insight(state, &transformation.title, &cleaned).await?;
    }
    Ok(cleaned)
}

/// Arguments of the `run_transformation` job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunTransformationArgs {
    pub source_id: String,
    pub transformation_id: String,
}

/// `run_transformation`: generate an insight on an existing source.
async fn run_transformation_job(ctx: JobContext, args: Value) -> AppResult<Value> {
    let started = Instant::now();
    let args: RunTransformationArgs =
        serde_json::from_value(args).map_err(|e| AppError::invalid(format!("Invalid run_transformation input: {e}")))?;
    let state = &ctx.state;
    tracing::info!("Running transformation {} on source {}", args.transformation_id, args.source_id);

    let outcome = async {
        let source = Source::get(state, &args.source_id).await?;
        let transformation = Transformation::get(state, &args.transformation_id).await?;
        apply_transformation(state, &transformation, None, Some(&source), transformation.model_id.as_deref()).await
    }
    .await;

    match outcome {
        Ok(_) => {
            let processing_time = started.elapsed().as_secs_f64();
            tracing::info!(
                "Successfully ran transformation {} on source {} in {processing_time:.2}s",
                args.transformation_id,
                args.source_id
            );
            Ok(json!({
                "success": true,
                "source_id": args.source_id,
                "transformation_id": args.transformation_id,
                "processing_time": processing_time,
                "error_message": null,
            }))
        }
        Err(e) => {
            if e.is_retryable() {
                tracing::warn!(
                    "Transient error running transformation {} on source {} (attempt {}): {e}",
                    args.transformation_id,
                    args.source_id,
                    ctx.attempt
                );
            } else {
                tracing::error!(
                    "Generation failed (permanent) for transformation {} on source {}: {e}",
                    args.transformation_id,
                    args.source_id
                );
            }
            Err(e)
        }
    }
}
