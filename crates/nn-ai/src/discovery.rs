//! Model discovery: asking a provider which models a credential can use.
//!
//! A port of `open_notebook/ai/model_discovery.py` (listing endpoints, static
//! fallbacks, name-based type classification) combined with the
//! credential-driven URL rules of `discover_with_config()` in
//! `api/credentials_service.py`. The database parts (registering models) are
//! the server's job.
//!
//! Result conventions of [`discover_models`]:
//!
//! * a credential that lacks what the listing needs (no key, no base URL)
//!   yields `Ok(vec![])`, as in the original;
//! * a listing that fails (bad key, unreachable server) yields `Err`, so the
//!   caller can tell the user why, except for Anthropic, which falls back to
//!   a static list as the original does;
//! * an unknown provider yields [`AiError::Unsupported`].

use std::time::Duration;

use nn_llm::{AiError, AiResult};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::{non_blank, ProviderConfig};
use crate::factory::Clients;
use crate::http::{encode_component, join_url, Http};
use crate::providers::anthropic::{normalize_anthropic_compatible_base_url, ANTHROPIC_API_ROOT, ANTHROPIC_VERSION};
use crate::providers::cohere::COHERE_API_ROOT;
use crate::providers::deepgram::DEEPGRAM_VOICES;
use crate::providers::google::gemini_root;
use crate::providers::ollama::OLLAMA_DEFAULT_BASE_URL;
use crate::providers::openai::{DEFAULT_AZURE_API_VERSION, OMLX_DEFAULT_BASE_URL};
use crate::registry::{self, Modality};

const LISTING_TIMEOUT: Duration = Duration::from_secs(30);
const OLLAMA_TIMEOUT: Duration = Duration::from_secs(10);
/// Safety net against a misbehaving pagination cursor.
const MAX_ANTHROPIC_PAGES: usize = 20;

/// A model a provider reports as available.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DiscoveredModel {
    pub name: String,
    pub provider: String,
    /// Best guess from the provider's metadata and the model name; the user
    /// can override it when registering the model.
    pub model_type: Modality,
    /// Human-readable name, when the provider supplies one.
    pub description: Option<String>,
}

impl DiscoveredModel {
    fn new(name: &str, provider: &str, model_type: Modality) -> Self {
        Self { name: name.to_string(), provider: provider.to_string(), model_type, description: None }
    }
}

// ---------------------------------------------------------------------------
// Name-based classification
// ---------------------------------------------------------------------------

/// Substring patterns per model type for one provider.
#[derive(Default)]
struct TypePatterns {
    language: &'static [&'static str],
    embedding: &'static [&'static str],
    speech_to_text: &'static [&'static str],
    text_to_speech: &'static [&'static str],
}

fn type_patterns(provider: &str) -> TypePatterns {
    match provider {
        "openai" => TypePatterns {
            language: &["gpt-4", "gpt-3.5", "o1", "o3", "chatgpt", "text-davinci", "davinci", "curie", "babbage", "ada"],
            embedding: &["text-embedding", "embedding"],
            speech_to_text: &["whisper"],
            text_to_speech: &["tts"],
        },
        "google" => TypePatterns {
            language: &["gemini", "palm", "bison", "chat"],
            embedding: &["embedding", "textembedding"],
            // Google STT reuses plain Gemini names and cannot be told apart by
            // name, so it has no pattern; users assign that type manually.
            text_to_speech: &["tts"],
            ..Default::default()
        },
        "ollama" => TypePatterns {
            language: &[
                "llama", "mistral", "mixtral", "codellama", "phi", "gemma", "qwen", "deepseek", "vicuna", "falcon",
                "orca", "neural", "dolphin", "openchat", "starling", "solar", "yi", "nous", "wizard", "zephyr",
                "tinyllama",
            ],
            embedding: &["nomic-embed", "mxbai-embed", "all-minilm", "bge-", "e5-"],
            ..Default::default()
        },
        "mistral" => TypePatterns {
            language: &["mistral", "mixtral", "codestral", "ministral", "pixtral", "open-mistral", "open-mixtral"],
            embedding: &["mistral-embed"],
            // STT is checked before TTS, so the STT patterns are the explicit
            // non-tts Voxtral names.
            speech_to_text: &["voxtral-mini-latest", "voxtral-small-latest"],
            text_to_speech: &["voxtral-mini-tts", "voxtral-tts"],
        },
        "groq" => TypePatterns {
            language: &["llama", "mixtral", "gemma", "whisper"],
            speech_to_text: &["whisper"],
            ..Default::default()
        },
        "deepseek" => TypePatterns {
            language: &["deepseek-chat", "deepseek-reasoner", "deepseek-coder"],
            ..Default::default()
        },
        "xai" => TypePatterns { language: &["grok"], ..Default::default() },
        "voyage" => TypePatterns { embedding: &["voyage"], ..Default::default() },
        "elevenlabs" => {
            TypePatterns { text_to_speech: &["eleven"], speech_to_text: &["scribe"], ..Default::default() }
        }
        // STT is checked first, so the "-en" Aura voices stay TTS.
        "deepgram" => TypePatterns {
            speech_to_text: &["nova", "whisper", "enhanced", "base"],
            text_to_speech: &["aura"],
            ..Default::default()
        },
        "dashscope" => TypePatterns { language: &["qwen"], ..Default::default() },
        // SiliconFlow's catalog mixes chat models with embedding, rerank and
        // audio models; keep those out of the language slot.
        "siliconflow" => TypePatterns {
            embedding: &["bge-", "bce-embedding", "embedding"],
            speech_to_text: &["sensevoice", "telespeech"],
            text_to_speech: &["cosyvoice", "fish-speech", "moss-tts"],
            ..Default::default()
        },
        "minimax" => {
            TypePatterns { text_to_speech: &["speech-"], language: &["minimax", "abab"], ..Default::default() }
        }
        // PPQ proxies many upstream providers, so its ids carry their names.
        "ppq" => TypePatterns {
            embedding: &["text-embedding", "embed"],
            speech_to_text: &["whisper", "nova-", "scribe"],
            text_to_speech: &["aura", "tts", "eleven"],
            ..Default::default()
        },
        _ => TypePatterns::default(),
    }
}

/// Classifies a model by its name, using the provider's naming patterns.
///
/// Types are checked from most to least specific (speech-to-text,
/// text-to-speech, embedding, language); unknown names default to language.
pub fn classify_model_type(model_name: &str, provider: &str) -> Modality {
    let name = model_name.to_lowercase();
    let patterns = type_patterns(provider);
    let ordered = [
        (Modality::SpeechToText, patterns.speech_to_text),
        (Modality::TextToSpeech, patterns.text_to_speech),
        (Modality::Embedding, patterns.embedding),
        (Modality::Language, patterns.language),
    ];
    ordered
        .into_iter()
        .find(|(_, patterns)| patterns.iter().any(|pattern| name.contains(pattern)))
        .map(|(modality, _)| modality)
        .unwrap_or(Modality::Language)
}

// ---------------------------------------------------------------------------
// Static lists
// ---------------------------------------------------------------------------

/// Used only when Anthropic's model listing API is unreachable or errors.
pub const ANTHROPIC_FALLBACK_MODELS: &[&str] = &[
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-5",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-haiku-4-5",
];

const VOYAGE_MODELS: &[&str] =
    &["voyage-3", "voyage-3-lite", "voyage-code-3", "voyage-finance-2", "voyage-law-2", "voyage-multilingual-2"];

const ELEVENLABS_TTS_MODELS: &[&str] = &[
    "eleven_v3",
    "eleven_flash_v2_5",
    "eleven_flash_v2",
    "eleven_multilingual_v2",
    "eleven_turbo_v2_5",
    "eleven_turbo_v2",
    "eleven_monolingual_v1",
    "eleven_multilingual_v1",
];
const ELEVENLABS_STT_MODELS: &[&str] = &["scribe_v2", "scribe_v1"];

const DEEPGRAM_STT_MODELS: &[&str] =
    &["nova-3", "nova-2", "whisper-large", "whisper-medium", "whisper-small", "whisper-base", "whisper-tiny"];

/// Vertex needs service-account OAuth to list models; a curated list of
/// well-known models stands in.
const VERTEX_MODELS: &[&str] = &["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-flash-lite", "text-embedding-005"];

/// Audio models that a provider serves but does not list: `(name, type)`.
///
/// OpenRouter's and MiniMax's `/models` listings are dominated by language
/// models, so the speech models known to work are seeded on top of a
/// successful listing. Users can add any other id by hand.
pub fn audio_seed(provider: &str) -> &'static [(&'static str, Modality)] {
    match provider {
        "openrouter" => &[
            ("microsoft/mai-voice-2", Modality::TextToSpeech),
            ("openai/whisper-1", Modality::SpeechToText),
            ("openai/whisper-large-v3", Modality::SpeechToText),
        ],
        "minimax" => &[("speech-2.8-hd", Modality::TextToSpeech), ("speech-2.8-turbo", Modality::TextToSpeech)],
        _ => &[],
    }
}

fn static_models(provider: &str) -> Vec<DiscoveredModel> {
    let typed = |names: &[&str], model_type| -> Vec<DiscoveredModel> {
        names.iter().map(|name| DiscoveredModel::new(name, provider, model_type)).collect()
    };
    match provider {
        "voyage" => typed(VOYAGE_MODELS, Modality::Embedding),
        "elevenlabs" => {
            [typed(ELEVENLABS_TTS_MODELS, Modality::TextToSpeech), typed(ELEVENLABS_STT_MODELS, Modality::SpeechToText)]
                .concat()
        }
        "deepgram" => {
            [typed(DEEPGRAM_VOICES, Modality::TextToSpeech), typed(DEEPGRAM_STT_MODELS, Modality::SpeechToText)].concat()
        }
        "vertex" => VERTEX_MODELS
            .iter()
            .map(|name| DiscoveredModel::new(name, provider, classify_model_type(name, "google")))
            .collect(),
        _ => Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Planning: which endpoint, with which credentials
// ---------------------------------------------------------------------------

/// Joins a base URL with `/models` without doubling an existing suffix.
fn models_endpoint(url: &str) -> String {
    let trimmed = url.trim().trim_end_matches('/');
    if trimmed.ends_with("/models") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/models")
    }
}

/// How one provider's models are found for a given credential.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Plan {
    /// The credential lacks what the listing needs.
    Nothing,
    /// The provider has no listing API.
    Static,
    /// OpenAI-style `GET /models` returning `{"data": [{"id": ...}]}`.
    OpenAiList { url: String, bearer: Option<String> },
    Azure { url: String, api_key: String },
    Anthropic { url: String, api_key: String, fallback: bool },
    Google { url: String, api_key: String },
    Ollama { url: String, bearer: Option<String> },
    Cohere { url: String, api_key: String },
}

fn plan(provider: &str, config: &ProviderConfig) -> AiResult<Plan> {
    let spec = registry::provider(provider)
        .ok_or_else(|| AiError::Unsupported(format!("Unknown provider: {}", provider.trim())))?;
    let key = config.key().map(str::to_string);
    let base = config.url();

    Ok(match spec.name {
        "vertex" => Plan::Static,
        "voyage" | "elevenlabs" | "deepgram" => {
            if key.is_some() {
                Plan::Static
            } else {
                Plan::Nothing
            }
        }
        "anthropic" => match key {
            Some(api_key) => {
                let root = base.map(normalize_anthropic_compatible_base_url);
                let root = root.as_deref().unwrap_or(ANTHROPIC_API_ROOT);
                Plan::Anthropic { url: join_url(root, "models"), api_key, fallback: true }
            }
            None => Plan::Nothing,
        },
        "anthropic_compatible" => match (base, key) {
            (Some(base), Some(api_key)) => Plan::Anthropic {
                url: join_url(&normalize_anthropic_compatible_base_url(base), "models"),
                api_key,
                fallback: false,
            },
            _ => Plan::Nothing,
        },
        "ollama" => {
            let base = base.unwrap_or(OLLAMA_DEFAULT_BASE_URL);
            let base = base.strip_suffix("/v1").unwrap_or(base);
            Plan::Ollama { url: join_url(base, "api/tags"), bearer: key }
        }
        "openai_compatible" => match base {
            Some(base) => Plan::OpenAiList { url: models_endpoint(base), bearer: key },
            None => Plan::Nothing,
        },
        "omlx" => Plan::OpenAiList { url: models_endpoint(base.unwrap_or(OMLX_DEFAULT_BASE_URL)), bearer: key },
        "azure" => {
            let endpoint = non_blank(&config.endpoint).map(|e| e.trim_end_matches('/')).or(base);
            match (endpoint, key) {
                (Some(endpoint), Some(api_key)) => {
                    let version = non_blank(&config.api_version).unwrap_or(DEFAULT_AZURE_API_VERSION);
                    let url = format!("{endpoint}/openai/models?api-version={}", encode_component(version));
                    Plan::Azure { url, api_key }
                }
                _ => Plan::Nothing,
            }
        }
        "google" => match key {
            // v1beta, like the model calls: v1 does not list preview models.
            Some(api_key) => Plan::Google { url: join_url(&gemini_root(base), "models"), api_key },
            None => Plan::Nothing,
        },
        "cohere" => match key {
            Some(api_key) => {
                let root = base.unwrap_or(COHERE_API_ROOT);
                let root = root.strip_suffix("/v2").or_else(|| root.strip_suffix("/v1")).unwrap_or(root);
                Plan::Cohere { url: join_url(root, "v1/models"), api_key }
            }
            None => Plan::Nothing,
        },
        // Table-driven OpenAI-compatible providers.
        _ => match (spec.openai_compat_discovery_url, key) {
            (Some(registry_url), Some(api_key)) => {
                // A credential's base URL is also where its models are listed,
                // but only for OpenAI (gateways) and providers that declare a
                // *_BASE_URL override (ADR-012). The others keep their registry
                // URL, which may carry provider-specific query parameters.
                let honors_base_url = spec.name == "openai" || spec.base_url_env().is_some();
                let url = match base {
                    Some(base) if honors_base_url => models_endpoint(base),
                    _ => registry_url.to_string(),
                };
                Plan::OpenAiList { url, bearer: Some(api_key) }
            }
            _ => Plan::Nothing,
        },
    })
}

// ---------------------------------------------------------------------------
// Parsing
// ---------------------------------------------------------------------------

fn text_field<'a>(item: &'a Value, key: &str) -> Option<&'a str> {
    item.get(key).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty())
}

/// Parses an OpenAI-style listing, applying the per-provider quirks.
fn parse_openai_list(provider: &str, json: &Value) -> Vec<DiscoveredModel> {
    let items = json.get("data").and_then(Value::as_array).or_else(|| json.as_array());
    let mut models: Vec<DiscoveredModel> = items
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let id = text_field(item, "id")?;
            let model_type = match provider {
                // Trust Mistral's capability flag over name patterns.
                "mistral" if item.pointer("/capabilities/completion_chat").and_then(Value::as_bool) == Some(true) => {
                    Modality::Language
                }
                // OpenRouter's listing holds language models.
                "openrouter" => Modality::Language,
                // Unknown servers mostly serve OpenAI-named models.
                "openai_compatible" | "omlx" | "azure" => classify_model_type(id, "openai"),
                _ => classify_model_type(id, provider),
            };
            Some(DiscoveredModel {
                name: id.to_string(),
                provider: provider.to_string(),
                model_type,
                description: text_field(item, "name").map(str::to_string),
            })
        })
        .collect();

    // Seed only on top of a listing that returned something: an empty result
    // usually means the call did not really work.
    if !models.is_empty() {
        for (name, model_type) in audio_seed(provider) {
            if !models.iter().any(|m| m.name == *name && m.model_type == *model_type) {
                models.push(DiscoveredModel::new(name, provider, *model_type));
            }
        }
    }
    models
}

fn parse_google_list(json: &Value) -> Vec<DiscoveredModel> {
    let items = json.get("models").and_then(Value::as_array);
    items
        .into_iter()
        .flatten()
        .filter_map(|item| {
            // Names arrive as "models/gemini-2.5-flash".
            let name = text_field(item, "name")?.trim_start_matches("models/");
            let methods: Vec<&str> = item
                .get("supportedGenerationMethods")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            let by_name = classify_model_type(name, "google");
            let model_type = if methods.iter().any(|m| m.starts_with("embed")) {
                Modality::Embedding
            } else if by_name == Modality::TextToSpeech {
                // Speech models also list generateContent; the name decides.
                by_name
            } else if methods.contains(&"generateContent") {
                Modality::Language
            } else {
                by_name
            };
            Some(DiscoveredModel {
                name: name.to_string(),
                provider: "google".to_string(),
                model_type,
                description: text_field(item, "displayName").map(str::to_string),
            })
        })
        .collect()
}

fn parse_ollama_list(json: &Value) -> Vec<DiscoveredModel> {
    let items = json.get("models").and_then(Value::as_array);
    items
        .into_iter()
        .flatten()
        .filter_map(|item| text_field(item, "name"))
        .map(|name| DiscoveredModel::new(name, "ollama", classify_model_type(name, "ollama")))
        .collect()
}

/// Cohere tags each model with the endpoints it serves; only chat and embed
/// models are relevant (reranking is out of scope).
fn parse_cohere_list(json: &Value) -> Vec<DiscoveredModel> {
    let items = json.get("models").and_then(Value::as_array);
    items
        .into_iter()
        .flatten()
        .filter_map(|item| {
            let name = text_field(item, "name")?;
            let endpoints: Vec<&str> =
                item.get("endpoints").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).collect();
            let model_type = if endpoints.contains(&"chat") {
                Modality::Language
            } else if endpoints.contains(&"embed") {
                Modality::Embedding
            } else {
                return None;
            };
            Some(DiscoveredModel::new(name, "cohere", model_type))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

/// Fetches every page of an Anthropic-style model listing.
async fn fetch_anthropic_model_ids(http: &Http, url: &str, api_key: &str) -> AiResult<Vec<String>> {
    let mut ids = Vec::new();
    let mut after_id: Option<String> = None;
    for _ in 0..MAX_ANTHROPIC_PAGES {
        let mut request = http
            .client
            .get(url)
            .header("x-api-key", api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .query(&[("limit", "100")]);
        if let Some(after) = &after_id {
            request = request.query(&[("after_id", after)]);
        }
        let page = http.json(request).await?;
        ids.extend(
            page.get("data").and_then(Value::as_array).into_iter().flatten().filter_map(|m| text_field(m, "id")).map(str::to_string),
        );
        let has_more = page.get("has_more").and_then(Value::as_bool).unwrap_or(false);
        match text_field(&page, "last_id") {
            Some(last) if has_more => after_id = Some(last.to_string()),
            _ => break,
        }
    }
    Ok(ids)
}

async fn execute(provider: &str, plan: Plan, config: &ProviderConfig, client: &reqwest::Client) -> AiResult<Vec<DiscoveredModel>> {
    let mut http = Http::new(client, provider, config, LISTING_TIMEOUT);
    http.timeout = if matches!(plan, Plan::Ollama { .. }) { OLLAMA_TIMEOUT } else { LISTING_TIMEOUT };
    let bearer = |request: reqwest::RequestBuilder, key: &Option<String>| match key {
        Some(key) => request.bearer_auth(key),
        None => request,
    };

    match plan {
        Plan::Nothing => Ok(Vec::new()),
        Plan::Static => Ok(static_models(provider)),
        Plan::OpenAiList { url, bearer: key } => {
            let json = http.json(bearer(http.client.get(&url), &key)).await?;
            Ok(parse_openai_list(provider, &json))
        }
        Plan::Azure { url, api_key } => {
            let json = http.json(http.client.get(&url).header("api-key", api_key)).await?;
            Ok(parse_openai_list(provider, &json))
        }
        Plan::Anthropic { url, api_key, fallback } => {
            let ids = match fetch_anthropic_model_ids(&http, &url, &api_key).await {
                Ok(ids) => ids,
                Err(err) if fallback => {
                    tracing::warn!(provider, error = %err, "model listing failed, using the static fallback");
                    ANTHROPIC_FALLBACK_MODELS.iter().map(|m| m.to_string()).collect()
                }
                Err(err) => return Err(err),
            };
            Ok(ids.iter().map(|id| DiscoveredModel::new(id, provider, Modality::Language)).collect())
        }
        Plan::Google { url, api_key } => {
            let request = http.client.get(&url).header("x-goog-api-key", api_key).query(&[("pageSize", "1000")]);
            Ok(parse_google_list(&http.json(request).await?))
        }
        Plan::Ollama { url, bearer: key } => {
            let json = http.json(bearer(http.client.get(&url), &key)).await?;
            Ok(parse_ollama_list(&json))
        }
        Plan::Cohere { url, api_key } => {
            let request = http.client.get(&url).bearer_auth(api_key).query(&[("page_size", "1000")]);
            Ok(parse_cohere_list(&http.json(request).await?))
        }
    }
}

impl Clients {
    /// See [`discover_models`].
    pub async fn discover_models(&self, provider: &str, config: &ProviderConfig) -> AiResult<Vec<DiscoveredModel>> {
        let name = registry::normalize_provider_name(provider);
        let plan = plan(&name, config)?;
        execute(&name, plan, config, self.http()).await
    }
}

/// Lists the models `config` can use with `provider`, each with a guessed
/// model type. See the module documentation for the result conventions.
pub async fn discover_models(provider: &str, config: &ProviderConfig) -> AiResult<Vec<DiscoveredModel>> {
    Clients::new().discover_models(provider, config).await
}

/// [`discover_models`] on the caller's connection pool.
pub async fn discover_models_with_client(
    provider: &str,
    config: &ProviderConfig,
    client: &reqwest::Client,
) -> AiResult<Vec<DiscoveredModel>> {
    Clients::with_client(client.clone()).discover_models(provider, config).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::shared_client;
    use crate::providers::testing::{header, request};
    use serde_json::json;
    use wiremock::matchers::{method, path, query_param, query_param_is_missing};
    use wiremock::{Mock, MockServer, ResponseTemplate};
    use Modality::{Embedding, Language, SpeechToText, TextToSpeech};

    fn keyed() -> ProviderConfig {
        ProviderConfig::new().with_api_key("key-123")
    }

    fn summary(models: &[DiscoveredModel]) -> Vec<(&str, Modality)> {
        models.iter().map(|m| (m.name.as_str(), m.model_type)).collect()
    }

    // --- classification ---------------------------------------------------

    #[test]
    fn openai_names_are_classified_by_pattern() {
        for (name, expected) in [
            ("gpt-4o", Language),
            ("gpt-4.1-mini", Language),
            ("gpt-3.5-turbo", Language),
            ("o3-mini", Language),
            ("chatgpt-4o-latest", Language),
            ("gpt-5", Language),
            ("text-embedding-3-small", Embedding),
            ("text-embedding-ada-002", Embedding),
            ("whisper-1", SpeechToText),
            ("tts-1", TextToSpeech),
            ("tts-1-hd", TextToSpeech),
            ("gpt-4o-mini-tts", TextToSpeech),
            ("dall-e-3", Language),
        ] {
            assert_eq!(classify_model_type(name, "openai"), expected, "{name}");
        }
    }

    #[test]
    fn classification_is_case_insensitive_and_defaults_to_language() {
        assert_eq!(classify_model_type("Whisper-Large-V3", "openai"), SpeechToText);
        assert_eq!(classify_model_type("anything", "unknown-provider"), Language);
        assert_eq!(classify_model_type("claude-sonnet-4-5", "anthropic"), Language);
        assert_eq!(classify_model_type("", "openai"), Language);
    }

    #[test]
    fn specific_types_win_over_language() {
        // Groq lists "whisper" under both language and speech-to-text; the
        // more specific type is checked first.
        assert_eq!(classify_model_type("whisper-large-v3", "groq"), SpeechToText);
        assert_eq!(classify_model_type("llama-3.3-70b", "groq"), Language);
        // Mistral: the -tts model must not be caught by the STT names.
        assert_eq!(classify_model_type("voxtral-mini-tts-2603", "mistral"), TextToSpeech);
        assert_eq!(classify_model_type("voxtral-mini-latest", "mistral"), SpeechToText);
        assert_eq!(classify_model_type("voxtral-small-latest", "mistral"), SpeechToText);
        assert_eq!(classify_model_type("mistral-embed", "mistral"), Embedding);
        assert_eq!(classify_model_type("mistral-small-latest", "mistral"), Language);
        // Deepgram: Aura voices are TTS, Nova/Whisper are STT.
        assert_eq!(classify_model_type("aura-2-thalia-en", "deepgram"), TextToSpeech);
        assert_eq!(classify_model_type("nova-3", "deepgram"), SpeechToText);
        assert_eq!(classify_model_type("whisper-large", "deepgram"), SpeechToText);
    }

    #[test]
    fn remaining_provider_tables_match_the_original() {
        for (name, provider, expected) in [
            ("gemini-2.5-flash", "google", Language),
            ("gemini-2.5-flash-preview-tts", "google", TextToSpeech),
            ("gemini-embedding-001", "google", Embedding),
            ("text-embedding-004", "google", Embedding),
            ("llama3.2:latest", "ollama", Language),
            ("nomic-embed-text:latest", "ollama", Embedding),
            ("mxbai-embed-large", "ollama", Embedding),
            ("bge-m3", "ollama", Embedding),
            ("voyage-3-lite", "voyage", Embedding),
            ("eleven_multilingual_v2", "elevenlabs", TextToSpeech),
            ("scribe_v1", "elevenlabs", SpeechToText),
            ("qwen-plus", "dashscope", Language),
            ("grok-4", "xai", Language),
            ("deepseek-chat", "deepseek", Language),
            ("BAAI/bge-m3", "siliconflow", Embedding),
            ("netease-youdao/bce-embedding-base_v1", "siliconflow", Embedding),
            ("FunAudioLLM/SenseVoiceSmall", "siliconflow", SpeechToText),
            ("FunAudioLLM/CosyVoice2-0.5B", "siliconflow", TextToSpeech),
            ("deepseek-ai/DeepSeek-V3", "siliconflow", Language),
            ("speech-2.8-hd", "minimax", TextToSpeech),
            ("MiniMax-M3", "minimax", Language),
        ] {
            assert_eq!(classify_model_type(name, provider), expected, "{provider}/{name}");
        }
    }

    #[test]
    fn ppq_classification_by_substring() {
        assert_eq!(classify_model_type("openai/text-embedding-3-small", "ppq"), Embedding);
        assert_eq!(classify_model_type("nova-3", "ppq"), SpeechToText);
        assert_eq!(classify_model_type("deepgram_aura_2", "ppq"), TextToSpeech);
        assert_eq!(classify_model_type("auto", "ppq"), Language);
    }

    // --- planning ---------------------------------------------------------

    fn list_url(provider: &str, config: &ProviderConfig) -> String {
        match plan(provider, config).unwrap() {
            Plan::OpenAiList { url, .. }
            | Plan::Azure { url, .. }
            | Plan::Anthropic { url, .. }
            | Plan::Google { url, .. }
            | Plan::Ollama { url, .. }
            | Plan::Cohere { url, .. } => url,
            other => panic!("{provider}: no URL in {other:?}"),
        }
    }

    #[test]
    fn registry_providers_list_at_their_registry_url() {
        assert_eq!(list_url("groq", &keyed()), "https://api.groq.com/openai/v1/models");
        assert_eq!(list_url("ppq", &keyed()), "https://api.ppq.ai/v1/models?type=all");
        assert_eq!(list_url("novita", &keyed()), "https://api.novita.ai/openai/models");
        assert_eq!(list_url("openai", &keyed()), "https://api.openai.com/v1/models");
        assert_eq!(
            plan("groq", &keyed()).unwrap(),
            Plan::OpenAiList { url: "https://api.groq.com/openai/v1/models".into(), bearer: Some("key-123".into()) }
        );
    }

    #[test]
    fn base_url_overrides_apply_only_where_declared() {
        let custom = keyed().with_base_url("https://gateway.test/v1/");
        // OpenAI and providers with a *_BASE_URL override honour the base URL.
        assert_eq!(list_url("openai", &custom), "https://gateway.test/v1/models");
        assert_eq!(list_url("siliconflow", &custom), "https://gateway.test/v1/models");
        assert_eq!(list_url("zai", &custom), "https://gateway.test/v1/models");
        assert_eq!(list_url("minimax", &custom), "https://gateway.test/v1/models");
        // Everyone else keeps the registry URL and its query parameters.
        assert_eq!(list_url("ppq", &custom), "https://api.ppq.ai/v1/models?type=all");
        assert_eq!(list_url("groq", &custom), "https://api.groq.com/openai/v1/models");
        // A pasted /models URL is not doubled.
        let pasted = keyed().with_base_url("https://api.siliconflow.cn/v1/models");
        assert_eq!(list_url("siliconflow", &pasted), "https://api.siliconflow.cn/v1/models");
    }

    #[test]
    fn missing_credentials_plan_nothing() {
        let empty = ProviderConfig::new();
        for provider in [
            "openai", "groq", "anthropic", "google", "cohere", "voyage", "elevenlabs", "deepgram",
            "openai_compatible", "anthropic_compatible", "azure",
        ] {
            assert_eq!(plan(provider, &empty).unwrap(), Plan::Nothing, "{provider}");
        }
        // anthropic_compatible needs both halves.
        assert_eq!(plan("anthropic_compatible", &keyed()).unwrap(), Plan::Nothing);
        assert_eq!(plan("anthropic_compatible", &ProviderConfig::new().with_base_url("http://x")).unwrap(), Plan::Nothing);
        assert_eq!(plan("azure", &keyed()).unwrap(), Plan::Nothing);
        assert!(matches!(plan("nope", &keyed()), Err(AiError::Unsupported(_))));
    }

    #[test]
    fn bespoke_providers_plan_their_own_endpoints() {
        let empty = ProviderConfig::new();
        assert_eq!(list_url("ollama", &empty), "http://localhost:11434/api/tags");
        assert_eq!(list_url("ollama", &ProviderConfig::new().with_base_url("http://box:11434/")), "http://box:11434/api/tags");
        assert_eq!(list_url("omlx", &empty), "http://localhost:11435/v1/models");
        assert_eq!(list_url("openai_compatible", &ProviderConfig::new().with_base_url("http://lm:1234/v1")), "http://lm:1234/v1/models");
        assert_eq!(list_url("anthropic", &keyed()), "https://api.anthropic.com/v1/models");
        assert_eq!(list_url("anthropic_compatible", &keyed().with_base_url("https://proxy.test")), "https://proxy.test/v1/models");
        assert_eq!(list_url("google", &keyed()), "https://generativelanguage.googleapis.com/v1beta/models");
        assert_eq!(list_url("cohere", &keyed()), "https://api.cohere.com/v1/models");
        let azure = ProviderConfig { endpoint: Some("https://r.openai.azure.com/".into()), ..keyed() };
        assert_eq!(list_url("azure", &azure), "https://r.openai.azure.com/openai/models?api-version=2024-10-21");
        let azure = ProviderConfig { api_version: Some("2025-01-01-preview".into()), ..keyed().with_base_url("https://form.openai.azure.com") };
        assert_eq!(list_url("azure", &azure), "https://form.openai.azure.com/openai/models?api-version=2025-01-01-preview");
        assert_eq!(plan("vertex", &empty).unwrap(), Plan::Static);
        assert_eq!(plan("voyage", &keyed()).unwrap(), Plan::Static);
    }

    // --- static lists -----------------------------------------------------

    #[tokio::test]
    async fn static_providers_return_typed_lists() {
        let voyage = discover_models("voyage", &keyed()).await.unwrap();
        assert_eq!(voyage.len(), 6);
        assert!(voyage.iter().all(|m| m.model_type == Embedding && m.provider == "voyage"));

        let deepgram = discover_models("deepgram", &keyed()).await.unwrap();
        assert!(deepgram.iter().any(|m| m.name == "nova-3" && m.model_type == SpeechToText));
        assert!(deepgram.iter().any(|m| m.name == "aura-2-thalia-en" && m.model_type == TextToSpeech));

        let elevenlabs = discover_models("elevenlabs", &keyed()).await.unwrap();
        assert!(elevenlabs.iter().any(|m| m.name == "eleven_multilingual_v2" && m.model_type == TextToSpeech));
        assert!(elevenlabs.iter().any(|m| m.name == "scribe_v1" && m.model_type == SpeechToText));

        let vertex = discover_models("vertex", &ProviderConfig::new()).await.unwrap();
        assert_eq!(
            summary(&vertex),
            vec![
                ("gemini-3.5-flash", Language),
                ("gemini-2.5-flash", Language),
                ("gemini-2.5-flash-lite", Language),
                ("text-embedding-005", Embedding),
            ]
        );

        assert!(discover_models("voyage", &ProviderConfig::new()).await.unwrap().is_empty());
        assert!(discover_models("openai", &ProviderConfig::new()).await.unwrap().is_empty());
    }

    // --- parsing ----------------------------------------------------------

    #[test]
    fn openai_listing_is_classified_per_provider() {
        let groq = parse_openai_list("groq", &json!({"data": [{"id": "whisper-large-v3"}, {"id": "llama-3.3-70b"}, {"id": ""}, {"object": "model"}]}));
        assert_eq!(summary(&groq), vec![("whisper-large-v3", SpeechToText), ("llama-3.3-70b", Language)]);
        assert!(groq.iter().all(|m| m.provider == "groq" && m.description.is_none()));

        // Unknown servers are classified with the OpenAI name patterns.
        let local = parse_openai_list("openai_compatible", &json!([{"id": "text-embedding-nomic"}, {"id": "qwen3"}, {"id": "whisper-1"}]));
        assert_eq!(summary(&local), vec![("text-embedding-nomic", Embedding), ("qwen3", Language), ("whisper-1", SpeechToText)]);
    }

    #[test]
    fn mistral_capabilities_quirk() {
        let models = parse_openai_list(
            "mistral",
            &json!({"data": [
                // The capability flag wins over name-based classification.
                {"id": "magistral-medium", "capabilities": {"completion_chat": true}},
                // No chat capability: fall back to name patterns.
                {"id": "mistral-embed", "capabilities": {"completion_chat": false}},
                {"id": "voxtral-mini-tts-2603", "capabilities": {"completion_chat": false}},
            ]}),
        );
        assert_eq!(
            summary(&models),
            vec![("magistral-medium", Language), ("mistral-embed", Embedding), ("voxtral-mini-tts-2603", TextToSpeech)]
        );
    }

    #[test]
    fn openrouter_models_are_language_with_descriptions_and_an_audio_seed() {
        let models = parse_openai_list("openrouter", &json!({"data": [{"id": "acme/embedding-x", "name": "Acme Embedding X"}]}));
        assert_eq!(models[0].model_type, Language);
        assert_eq!(models[0].description.as_deref(), Some("Acme Embedding X"));
        assert_eq!(
            summary(&models[1..]),
            vec![
                ("microsoft/mai-voice-2", TextToSpeech),
                ("openai/whisper-1", SpeechToText),
                ("openai/whisper-large-v3", SpeechToText),
            ]
        );
        assert!(models.iter().all(|m| m.provider == "openrouter"));
    }

    #[test]
    fn minimax_tts_models_are_seeded_once_and_only_on_success() {
        let models = parse_openai_list("minimax", &json!({"data": [{"id": "MiniMax-M3"}, {"id": "speech-2.8-hd"}]}));
        assert_eq!(
            summary(&models),
            vec![("MiniMax-M3", Language), ("speech-2.8-hd", TextToSpeech), ("speech-2.8-turbo", TextToSpeech)]
        );
        // An empty listing means discovery did not really work: seed nothing.
        assert!(parse_openai_list("minimax", &json!({"data": []})).is_empty());
        assert!(parse_openai_list("openrouter", &json!({})).is_empty());
    }

    #[test]
    fn google_listing_uses_generation_methods_and_names() {
        let models = parse_google_list(&json!({"models": [
            {"name": "models/gemini-2.5-flash", "displayName": "Gemini 2.5 Flash", "supportedGenerationMethods": ["generateContent", "countTokens"]},
            {"name": "models/gemini-embedding-001", "displayName": "Gemini Embedding", "supportedGenerationMethods": ["embedContent"]},
            {"name": "models/gemini-2.5-flash-preview-tts", "supportedGenerationMethods": ["generateContent"]},
            {"name": "models/aqa", "supportedGenerationMethods": ["generateAnswer"]},
            {"displayName": "nameless"},
        ]}));
        assert_eq!(
            summary(&models),
            vec![
                ("gemini-2.5-flash", Language),
                ("gemini-embedding-001", Embedding),
                ("gemini-2.5-flash-preview-tts", TextToSpeech),
                ("aqa", Language),
            ]
        );
        assert_eq!(models[0].description.as_deref(), Some("Gemini 2.5 Flash"));
    }

    #[test]
    fn cohere_listing_keeps_chat_and_embed_models_only() {
        let models = parse_cohere_list(&json!({"models": [
            {"name": "command-a-03-2025", "endpoints": ["chat"]},
            {"name": "embed-v4.0", "endpoints": ["embed", "classify"]},
            {"name": "rerank-v3.5", "endpoints": ["rerank"]},
            {"name": "mystery"},
        ]}));
        assert_eq!(summary(&models), vec![("command-a-03-2025", Language), ("embed-v4.0", Embedding)]);
    }

    // --- execution against a mock ----------------------------------------

    #[tokio::test]
    async fn openai_style_discovery_sends_the_bearer_key() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": [{"id": "gpt-4o"}, {"id": "text-embedding-3-small"}, {"id": "whisper-1"}, {"id": "tts-1"}]})))
            .mount(&server)
            .await;
        let config = keyed().with_base_url(format!("{}/v1", server.uri()));
        let models = discover_models("openai", &config).await.unwrap();
        assert_eq!(
            summary(&models),
            vec![("gpt-4o", Language), ("text-embedding-3-small", Embedding), ("whisper-1", SpeechToText), ("tts-1", TextToSpeech)]
        );
        assert_eq!(header(&request(&server, 0).await, "authorization"), Some("Bearer key-123"));
    }

    #[tokio::test]
    async fn local_servers_are_listed_without_a_key() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": [{"id": "mlx-community/Qwen3-4bit"}]})))
            .mount(&server)
            .await;
        let config = ProviderConfig::new().with_base_url(format!("{}/v1", server.uri()));
        for provider in ["openai_compatible", "omlx"] {
            let models = discover_models(provider, &config).await.unwrap();
            assert_eq!(models.len(), 1);
            assert_eq!(models[0].provider, provider);
        }
        assert!(header(&request(&server, 0).await, "authorization").is_none());
    }

    #[tokio::test]
    async fn failed_listings_are_errors() {
        let server = MockServer::start().await;
        Mock::given(method("GET")).respond_with(ResponseTemplate::new(401).set_body_json(json!({"error": {"message": "bad key"}}))).mount(&server).await;
        let config = keyed().with_base_url(format!("{}/v1", server.uri()));
        let err = discover_models("openai", &config).await.unwrap_err();
        assert!(matches!(err, AiError::Authentication(_)), "{err:?}");
        let err = discover_models("anthropic_compatible", &config).await.unwrap_err();
        assert!(matches!(err, AiError::Authentication(_)), "{err:?}");
    }

    #[tokio::test]
    async fn anthropic_listing_follows_pagination() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .and(query_param("after_id", "claude-opus-4-8"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": [{"id": "claude-haiku-4-5"}], "has_more": false, "last_id": "claude-haiku-4-5"})))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .and(query_param("limit", "100"))
            .and(query_param_is_missing("after_id"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": [{"id": "claude-opus-4-8"}], "has_more": true, "last_id": "claude-opus-4-8"})))
            .mount(&server)
            .await;
        let config = keyed().with_base_url(server.uri());
        let models = discover_models("anthropic", &config).await.unwrap();
        assert_eq!(summary(&models), vec![("claude-opus-4-8", Language), ("claude-haiku-4-5", Language)]);
        assert!(models.iter().all(|m| m.provider == "anthropic"));

        let first = request(&server, 0).await;
        assert_eq!(header(&first, "x-api-key"), Some("key-123"));
        assert_eq!(header(&first, "anthropic-version"), Some("2023-06-01"));
        assert_eq!(server.received_requests().await.unwrap().len(), 2);
    }

    #[tokio::test]
    async fn anthropic_falls_back_to_the_static_list_on_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET")).respond_with(ResponseTemplate::new(500).set_body_json(json!({"error": "boom"}))).mount(&server).await;
        let config = keyed().with_base_url(server.uri());
        let models = discover_models("anthropic", &config).await.unwrap();
        assert_eq!(models.iter().map(|m| m.name.as_str()).collect::<Vec<_>>(), ANTHROPIC_FALLBACK_MODELS);
        assert!(models.iter().all(|m| m.provider == "anthropic" && m.model_type == Language));
    }

    #[tokio::test]
    async fn bespoke_listings_send_their_own_auth() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/api/tags"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"models": [{"name": "llama3.2:latest"}, {"name": "nomic-embed-text:latest"}]})))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1beta/models"))
            .and(query_param("pageSize", "1000"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"models": [{"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]}]})))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/v1/models"))
            .and(query_param("page_size", "1000"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"models": [{"name": "embed-v4.0", "endpoints": ["embed"]}]})))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/openai/models"))
            .and(query_param("api-version", "2024-10-21"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": [{"id": "gpt-4o"}, {"id": "text-embedding-3-large"}]})))
            .mount(&server)
            .await;
        let client = shared_client();

        let ollama = discover_models_with_client("ollama", &ProviderConfig::new().with_base_url(server.uri()), &client).await.unwrap();
        assert_eq!(summary(&ollama), vec![("llama3.2:latest", Language), ("nomic-embed-text:latest", Embedding)]);

        let google = discover_models("google", &keyed().with_base_url(format!("{}/v1beta", server.uri()))).await.unwrap();
        assert_eq!(summary(&google), vec![("gemini-2.5-flash", Language)]);
        assert_eq!(header(&request(&server, 1).await, "x-goog-api-key"), Some("key-123"));

        let cohere = discover_models("cohere", &keyed().with_base_url(server.uri())).await.unwrap();
        assert_eq!(summary(&cohere), vec![("embed-v4.0", Embedding)]);
        assert_eq!(header(&request(&server, 2).await, "authorization"), Some("Bearer key-123"));

        let azure = ProviderConfig { endpoint: Some(server.uri()), ..keyed() };
        let azure = discover_models("azure", &azure).await.unwrap();
        assert_eq!(summary(&azure), vec![("gpt-4o", Language), ("text-embedding-3-large", Embedding)]);
        assert_eq!(header(&request(&server, 3).await, "api-key"), Some("key-123"));
        assert!(azure.iter().all(|m| m.provider == "azure"));
    }

    #[test]
    fn discovered_models_serialise_for_the_api() {
        let model = DiscoveredModel { name: "gpt-4o".into(), provider: "openai".into(), model_type: Language, description: None };
        assert_eq!(
            serde_json::to_value(&model).unwrap(),
            json!({"name": "gpt-4o", "provider": "openai", "model_type": "language", "description": null})
        );
    }
}
