//! Model resolution: from a model id or a default slot in the database to a
//! ready-to-call client from `nn-ai`.
//!
//! The Rust counterpart of Python's `ModelManager`. Every feature that calls
//! a model goes through here:
//!
//! ```ignore
//! let llm = state.models.language(&state, DefaultModelType::Chat).await?;      // default slot, error if unset
//! let llm = state.models.language_by_id(&state, &model_id).await?;             // a specific model
//! let embedder = state.models.embedding(&state).await?;
//! ```
//!
//! Tests replace real providers with mocks through [`ModelManager::set_override`]:
//! once an override is set for a model type, every request for that type
//! returns it, whatever the database says.

pub mod net;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use nn_llm::{SharedEmbeddingModel, SharedLanguageModel, SharedSpeechToTextModel, SharedTextToSpeechModel};

use crate::config::Config;
use crate::db::Db;
use crate::domain::{DefaultModelType, DefaultModels, Model};
use crate::error::{AppError, AppResult};
use crate::state::AppState;

/// A model client of any type.
#[derive(Clone)]
pub enum AnyModel {
    Language(SharedLanguageModel),
    Embedding(SharedEmbeddingModel),
    TextToSpeech(SharedTextToSpeechModel),
    SpeechToText(SharedSpeechToTextModel),
}

impl AnyModel {
    fn type_name(&self) -> &'static str {
        match self {
            Self::Language(_) => "language",
            Self::Embedding(_) => "embedding",
            Self::TextToSpeech(_) => "text_to_speech",
            Self::SpeechToText(_) => "speech_to_text",
        }
    }
}

/// Resolves configured models and their credentials into clients.
#[derive(Clone)]
pub struct ModelManager {
    #[allow(dead_code)]
    db: Db,
    config: Arc<Config>,
    clients: nn_ai::Clients,
    overrides: Arc<RwLock<HashMap<&'static str, AnyModel>>>,
}

/// The user-facing message when a default slot is empty.
fn no_default_message(kind: DefaultModelType) -> String {
    let what = match kind {
        DefaultModelType::Chat => "chat model",
        DefaultModelType::Transformation => "transformation model",
        DefaultModelType::Tools => "tools model",
        DefaultModelType::LargeContext => "large context model",
        DefaultModelType::Embedding => "embedding model",
        DefaultModelType::TextToSpeech => "text-to-speech model",
        DefaultModelType::SpeechToText => "speech-to-text model",
    };
    format!("No {what} configured. Please go to Manage → Models and set a default {what}.")
}

impl ModelManager {
    pub fn new(db: Db, config: Arc<Config>, http: reqwest::Client) -> Self {
        Self { db, config, clients: nn_ai::Clients::with_client(http), overrides: Arc::new(RwLock::new(HashMap::new())) }
    }

    /// The `nn-ai` client set (shared connection pool), for discovery and
    /// connection tests.
    pub fn clients(&self) -> &nn_ai::Clients {
        &self.clients
    }

    /// Answer every request for this model's type with `model`. For tests.
    pub fn set_override(&self, model: AnyModel) {
        self.overrides.write().expect("model override lock poisoned").insert(model.type_name(), model);
    }

    /// Remove every override.
    pub fn clear_overrides(&self) {
        self.overrides.write().expect("model override lock poisoned").clear();
    }

    fn overridden(&self, model_type: &str) -> Option<AnyModel> {
        self.overrides.read().expect("model override lock poisoned").get(model_type).cloned()
    }

    /// Whether any override is active (callers skip database checks then).
    pub fn has_override(&self, model_type: &str) -> bool {
        self.overrides.read().expect("model override lock poisoned").contains_key(model_type)
    }

    /// Build the client for a model record, with its credential (or the
    /// provider's default credential, or the environment).
    pub async fn build(&self, state: &AppState, model: &Model) -> AppResult<AnyModel> {
        if let Some(model) = self.overridden(&model.model_type) {
            return Ok(model);
        }
        // Stored URLs and files are checked again at use time (see `net`):
        // DNS may have changed since the credential was saved, and rows
        // written before these checks existed are covered too. Local
        // addresses stay allowed by default (self-hosted Ollama and friends).
        let mut config = net::prepare_client_config(&self.config.providers, &model.provider, model.provider_config(state).await?)
            .await
            .map_err(|e| AppError::config(format!("The endpoint configured for {} is not allowed: {e}", model.provider)))?;
        if config.timeout.is_none() && model.model_type == "language" {
            config.timeout = Some(self.config.llm_timeout);
        }
        let clients = &self.clients;
        Ok(match model.model_type.as_str() {
            "language" => AnyModel::Language(clients.language_model(&model.provider, &model.name, &config)?),
            "embedding" => AnyModel::Embedding(clients.embedding_model(&model.provider, &model.name, &config)?),
            "text_to_speech" => AnyModel::TextToSpeech(clients.text_to_speech_model(&model.provider, &model.name, &config)?),
            "speech_to_text" => AnyModel::SpeechToText(clients.speech_to_text_model(&model.provider, &model.name, &config)?),
            other => return Err(AppError::config(format!("Invalid model type: {other}"))),
        })
    }

    /// The client for a model id. Missing or invalid → `Configuration`.
    pub async fn by_id(&self, state: &AppState, model_id: &str) -> AppResult<AnyModel> {
        let model = Model::get_for_use(state, model_id).await?;
        self.build(state, &model).await
    }

    /// The client for a default slot, or `None` when the slot is empty or
    /// its model is gone (both logged).
    pub async fn default_model(&self, state: &AppState, kind: DefaultModelType) -> AppResult<Option<AnyModel>> {
        if let Some(model) = self.overridden(kind.model_type()) {
            return Ok(Some(model));
        }
        match DefaultModels::model_for(state, kind).await? {
            Some(model) => Ok(Some(self.build(state, &model).await?)),
            None => Ok(None),
        }
    }

    /// A language model for a default slot. Empty slot → `Configuration`.
    pub async fn language(&self, state: &AppState, kind: DefaultModelType) -> AppResult<SharedLanguageModel> {
        match self.default_model(state, kind).await? {
            Some(AnyModel::Language(model)) => Ok(model),
            Some(other) => Err(wrong_type("language", &other)),
            None => Err(AppError::config(no_default_message(kind))),
        }
    }

    /// A language model by id, or the slot's default when `model_id` is `None`
    /// (the common "override or default" pattern).
    pub async fn language_or_default(
        &self,
        state: &AppState,
        model_id: Option<&str>,
        kind: DefaultModelType,
    ) -> AppResult<SharedLanguageModel> {
        match model_id.filter(|id| !id.trim().is_empty()) {
            Some(id) => self.language_by_id(state, id).await,
            None => self.language(state, kind).await,
        }
    }

    pub async fn language_by_id(&self, state: &AppState, model_id: &str) -> AppResult<SharedLanguageModel> {
        match self.by_id(state, model_id).await? {
            AnyModel::Language(model) => Ok(model),
            other => Err(wrong_type("language", &other)),
        }
    }

    /// The default embedding model. Unset → `Configuration` with the Python message.
    pub async fn embedding(&self, state: &AppState) -> AppResult<SharedEmbeddingModel> {
        match self.default_model(state, DefaultModelType::Embedding).await? {
            Some(AnyModel::Embedding(model)) => Ok(model),
            Some(other) => Err(wrong_type("embedding", &other)),
            None => Err(AppError::config("No embedding model configured. Please configure one in the Models section.")),
        }
    }

    /// The default embedding model, or `None` when none is configured.
    pub async fn embedding_opt(&self, state: &AppState) -> AppResult<Option<SharedEmbeddingModel>> {
        match self.default_model(state, DefaultModelType::Embedding).await? {
            Some(AnyModel::Embedding(model)) => Ok(Some(model)),
            Some(other) => Err(wrong_type("embedding", &other)),
            None => Ok(None),
        }
    }

    pub async fn text_to_speech_by_id(&self, state: &AppState, model_id: &str) -> AppResult<SharedTextToSpeechModel> {
        match self.by_id(state, model_id).await? {
            AnyModel::TextToSpeech(model) => Ok(model),
            other => Err(wrong_type("text_to_speech", &other)),
        }
    }

    /// The default speech-to-text model, or `None` when none is configured.
    pub async fn speech_to_text_opt(&self, state: &AppState) -> AppResult<Option<SharedSpeechToTextModel>> {
        match self.default_model(state, DefaultModelType::SpeechToText).await? {
            Some(AnyModel::SpeechToText(model)) => Ok(Some(model)),
            Some(other) => Err(wrong_type("speech_to_text", &other)),
            None => Ok(None),
        }
    }
}

fn wrong_type(expected: &str, got: &AnyModel) -> AppError {
    AppError::config(format!("Expected a {expected} model but the configured model is a {} model", got.type_name()))
}

/// Embed one piece of text with the default embedding model.
///
/// Text longer than one chunk is chunked, each chunk embedded, and the
/// vectors mean-pooled into one (as the Python `generate_embedding` does).
pub async fn generate_embedding(state: &AppState, text: &str) -> AppResult<Vec<f32>> {
    let text = text.trim();
    if text.is_empty() {
        return Err(AppError::invalid("Cannot generate embedding for empty text"));
    }
    let model = state.models.embedding(state).await?;
    let config = chunk_config();
    let pieces = if nn_content::token_count(text) <= config.chunk_size {
        vec![text.to_string()]
    } else {
        let chunks = nn_content::chunk_text_with(text, None, None, &config);
        if chunks.is_empty() {
            return Err(AppError::invalid("Text chunking produced no chunks"));
        }
        chunks
    };
    let vectors = generate_embeddings(state, &model, &pieces).await?;
    if vectors.len() == 1 {
        return Ok(normalize(vectors.into_iter().next().unwrap_or_default()));
    }
    mean_pool(&vectors)
}

/// The chunking configuration from `OPEN_NOTEBOOK_CHUNK_SIZE` / `_CHUNK_OVERLAP` / `_MIN_CHUNK_SIZE`.
pub fn chunk_config() -> nn_content::ChunkConfig {
    let env = |name: &str| std::env::var(name).ok();
    nn_content::ChunkConfig::from_settings(
        env("OPEN_NOTEBOOK_CHUNK_SIZE").as_deref(),
        env("OPEN_NOTEBOOK_CHUNK_OVERLAP").as_deref(),
        env("OPEN_NOTEBOOK_MIN_CHUNK_SIZE").as_deref(),
    )
}

/// Batch size for embedding calls (`OPEN_NOTEBOOK_EMBEDDING_BATCH_SIZE`, default 50).
pub fn embedding_batch_size() -> usize {
    std::env::var("OPEN_NOTEBOOK_EMBEDDING_BATCH_SIZE")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(50)
}

/// Embed many texts in batches, retrying each batch up to three times on a
/// retryable error, preserving order.
pub async fn generate_embeddings(state: &AppState, model: &SharedEmbeddingModel, texts: &[String]) -> AppResult<Vec<Vec<f32>>> {
    let _ = state;
    const MAX_ATTEMPTS: u32 = 3;
    let batch_size = embedding_batch_size();
    let total = texts.len().div_ceil(batch_size);
    let mut out = Vec::with_capacity(texts.len());
    for (index, batch) in texts.chunks(batch_size).enumerate() {
        let mut attempt = 1;
        loop {
            match model.embed(batch).await {
                Ok(vectors) if vectors.len() == batch.len() => {
                    out.extend(vectors);
                    break;
                }
                Ok(vectors) => {
                    return Err(AppError::ExternalService(format!(
                        "Embedding model returned {} vectors for {} texts",
                        vectors.len(),
                        batch.len()
                    )))
                }
                Err(e) if e.is_retryable() && attempt < MAX_ATTEMPTS => {
                    tracing::debug!("Embedding batch {}/{total} attempt {attempt} failed: {e}. Retrying...", index + 1);
                    attempt += 1;
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                }
                Err(e) => {
                    let wrapped = AppError::from(e.clone());
                    return Err(match wrapped {
                        AppError::ExternalService(m) => AppError::ExternalService(format!(
                            "Failed to generate embeddings using model '{}' (batch {}/{total}, {} texts): {m}",
                            model.model_name(),
                            index + 1,
                            batch.len()
                        )),
                        other => other,
                    });
                }
            }
        }
    }
    Ok(out)
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>().sqrt();
    if norm > 0.0 {
        v.iter_mut().for_each(|x| *x = (*x as f64 / norm) as f32);
    }
    v
}

/// Mean of L2-normalised vectors, normalised again.
pub fn mean_pool(vectors: &[Vec<f32>]) -> AppResult<Vec<f32>> {
    let Some(first) = vectors.first() else {
        return Err(AppError::invalid("Cannot mean pool empty list of embeddings"));
    };
    let dims = first.len();
    if vectors.iter().any(|v| v.len() != dims) {
        return Err(AppError::invalid("Cannot mean pool embeddings of different sizes"));
    }
    let mut mean = vec![0f64; dims];
    for v in vectors {
        let norm = v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>().sqrt();
        let norm = if norm > 0.0 { norm } else { 1.0 };
        for (m, x) in mean.iter_mut().zip(v) {
            *m += *x as f64 / norm;
        }
    }
    let n = vectors.len() as f64;
    mean.iter_mut().for_each(|m| *m /= n);
    let norm = mean.iter().map(|m| m * m).sum::<f64>().sqrt();
    if norm > 0.0 {
        mean.iter_mut().for_each(|m| *m /= norm);
    }
    Ok(mean.into_iter().map(|m| m as f32).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Doc;
    use crate::testing::{MockEmbeddingModel, MockLanguageModel};

    #[test]
    fn mean_pooling_normalises() {
        let pooled = mean_pool(&[vec![2.0, 0.0], vec![0.0, 3.0]]).unwrap();
        let expected = std::f32::consts::FRAC_1_SQRT_2;
        assert!((pooled[0] - expected).abs() < 1e-6 && (pooled[1] - expected).abs() < 1e-6, "{pooled:?}");
        assert!(mean_pool(&[]).is_err());
        assert!(mean_pool(&[vec![1.0], vec![1.0, 2.0]]).is_err());
    }

    #[tokio::test]
    async fn empty_slots_are_configuration_errors() {
        let (state, _dir) = AppState::for_tests().await.unwrap();
        let e = state.models.language(&state, DefaultModelType::Chat).await.err().unwrap();
        assert!(matches!(e, AppError::Configuration(_)), "{e}");
        assert!(e.to_string().contains("Manage → Models"));
        let e = state.models.embedding(&state).await.err().unwrap();
        assert_eq!(e.to_string(), "No embedding model configured. Please configure one in the Models section.");
        assert!(state.models.embedding_opt(&state).await.unwrap().is_none());
        assert!(matches!(generate_embedding(&state, "hello").await, Err(AppError::Configuration(_))));
        assert!(matches!(state.models.language_by_id(&state, "model:missing").await, Err(AppError::Configuration(_))));
    }

    #[tokio::test]
    async fn configured_models_build_real_clients() {
        let (state, _dir) = AppState::for_tests().await.unwrap();
        let mut cred = crate::domain::Credential {
            name: "Local".into(),
            provider: "ollama".into(),
            modalities: vec!["language".into()],
            base_url: Some("http://localhost:11434".into()),
            ..Default::default()
        };
        cred.save(&state).await.unwrap();
        let row = state
            .db
            .create("model", Doc::new().set("name", "llama3").set("provider", "ollama").set("type", "language").record("credential", &cred.id))
            .await
            .unwrap();
        let model_id = row["id"].as_str().unwrap().to_string();
        let mut defaults = DefaultModels::load(&state).await.unwrap();
        defaults.default_chat_model = Some(model_id.clone());
        defaults.save(&state).await.unwrap();

        let chat = state.models.language(&state, DefaultModelType::Chat).await.unwrap();
        assert_eq!(chat.provider(), "ollama");
        assert_eq!(chat.model_name(), "llama3");
        // Transformation falls back to the chat model.
        let t = state.models.language(&state, DefaultModelType::Transformation).await.unwrap();
        assert_eq!(t.model_name(), "llama3");
        // Asking for the wrong type is a clear error.
        assert!(state.models.text_to_speech_by_id(&state, &model_id).await.is_err());
    }

    #[tokio::test]
    async fn overrides_replace_providers_and_embeddings_pool() {
        let (state, _dir) = AppState::for_tests().await.unwrap();
        let llm = MockLanguageModel::new(["hi"]);
        state.models.set_override(AnyModel::Language(llm.clone()));
        let embedder = MockEmbeddingModel::new();
        state.models.set_override(AnyModel::Embedding(embedder.clone()));

        let chat = state.models.language(&state, DefaultModelType::Chat).await.unwrap();
        assert_eq!(chat.complete("q").await.unwrap(), "hi");

        let short = generate_embedding(&state, "a short text").await.unwrap();
        assert_eq!(short, embedder.vector("a short text"));
        let long = "word ".repeat(3000);
        let pooled = generate_embedding(&state, &long).await.unwrap();
        assert_eq!(pooled.len(), embedder.dimensions);
        assert!(embedder.calls.lock().unwrap().last().unwrap().len() > 1, "long text is chunked");
        assert!(generate_embedding(&state, "   ").await.is_err());
    }
}
