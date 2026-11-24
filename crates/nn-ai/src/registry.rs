//! Provider registry: the single source of truth for AI provider metadata.
//!
//! A faithful port of `open_notebook/ai/provider_registry.py`. The table is
//! pure static data and its declaration order is the display order the
//! frontend renders (`GET /api/providers` returns it as declared).

use serde::{Deserialize, Serialize};

/// The four kinds of model a provider can offer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Modality {
    Language,
    Embedding,
    SpeechToText,
    TextToSpeech,
}

impl Modality {
    pub const ALL: [Modality; 4] = [
        Modality::Language,
        Modality::Embedding,
        Modality::SpeechToText,
        Modality::TextToSpeech,
    ];

    /// The identifier stored on model records (`language`, `embedding`,
    /// `speech_to_text`, `text_to_speech`).
    pub fn as_str(&self) -> &'static str {
        match self {
            Modality::Language => "language",
            Modality::Embedding => "embedding",
            Modality::SpeechToText => "speech_to_text",
            Modality::TextToSpeech => "text_to_speech",
        }
    }

    pub fn parse(value: &str) -> Option<Modality> {
        match value.trim().to_ascii_lowercase().as_str() {
            "language" => Some(Modality::Language),
            "embedding" => Some(Modality::Embedding),
            "speech_to_text" => Some(Modality::SpeechToText),
            "text_to_speech" => Some(Modality::TextToSpeech),
            _ => None,
        }
    }

    /// Wording used in error messages ("a text-to-speech model").
    pub(crate) fn human(&self) -> &'static str {
        match self {
            Modality::Language => "language",
            Modality::Embedding => "embedding",
            Modality::SpeechToText => "speech-to-text",
            Modality::TextToSpeech => "text-to-speech",
        }
    }
}

impl std::fmt::Display for Modality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Everything the backend needs to know about one AI provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ProviderSpec {
    pub name: &'static str,
    pub display_name: &'static str,
    /// Default modalities offered when creating a credential for this provider.
    pub modalities: &'static [Modality],
    /// Env vars that must ALL be set for the provider to count as configured.
    pub required_env: &'static [&'static str],
    /// Env vars of which at least ONE must be set.
    pub required_any_env: &'static [&'static str],
    /// Env vars read during migration but not required.
    pub optional_env: &'static [&'static str],
    /// Cheapest model used by the provider connection test. `None` means the
    /// test is dynamic (first available model) or handled by a bespoke tester.
    pub test_model: Option<&'static str>,
    pub test_model_type: Modality,
    /// Where users get an API key / set the provider up.
    pub docs_url: Option<&'static str>,
    /// For providers exposing an OpenAI-compatible `GET /models` endpoint,
    /// the discovery URL.
    pub openai_compat_discovery_url: Option<&'static str>,
}

/// Env var configuration in the legacy `PROVIDER_ENV_CONFIG` shape: keys are
/// omitted when they have no entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct EnvConfig {
    #[serde(skip_serializing_if = "<[&str]>::is_empty")]
    pub required: &'static [&'static str],
    #[serde(skip_serializing_if = "<[&str]>::is_empty")]
    pub required_any: &'static [&'static str],
    #[serde(skip_serializing_if = "<[&str]>::is_empty")]
    pub optional: &'static [&'static str],
}

/// One entry of the `GET /api/providers` response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ProviderInfo {
    pub name: &'static str,
    pub display_name: &'static str,
    pub modalities: &'static [Modality],
    pub docs_url: Option<&'static str>,
    pub env_configured: bool,
}

impl ProviderSpec {
    /// Optional env var overriding the provider's endpoint (`*_BASE_URL`).
    ///
    /// Providers that declare one accept a user-supplied base URL for
    /// discovery, e.g. regional endpoints (ADR-012).
    pub fn base_url_env(&self) -> Option<&'static str> {
        self.optional_env.iter().copied().find(|v| v.ends_with("_BASE_URL"))
    }

    pub fn env_config(&self) -> EnvConfig {
        EnvConfig {
            required: self.required_env,
            required_any: self.required_any_env,
            optional: self.optional_env,
        }
    }

    pub fn supports(&self, modality: Modality) -> bool {
        self.modalities.contains(&modality)
    }

    /// Modalities as the strings stored on credentials and models.
    pub fn modality_names(&self) -> Vec<&'static str> {
        self.modalities.iter().map(Modality::as_str).collect()
    }

    /// Whether the provider has enough env vars set to be migrated to a
    /// credential. This crate never reads the environment itself: the caller
    /// passes the lookup (`|k| std::env::var(k).ok()`).
    pub fn is_env_configured(&self, lookup: impl Fn(&str) -> Option<String>) -> bool {
        let set = |name: &&str| lookup(name).is_some_and(|v| !v.trim().is_empty());
        if !self.required_any_env.is_empty() {
            self.required_any_env.iter().any(set)
        } else if !self.required_env.is_empty() {
            self.required_env.iter().all(set)
        } else {
            false
        }
    }

    /// The `GET /api/providers` entry for this provider.
    pub fn info(&self, env_configured: bool) -> ProviderInfo {
        ProviderInfo {
            name: self.name,
            display_name: self.display_name,
            modalities: self.modalities,
            docs_url: self.docs_url,
            env_configured,
        }
    }
}

use Modality::{Embedding, Language, SpeechToText, TextToSpeech};

const LANGUAGE_ONLY: &[Modality] = &[Language];
const ALL_MODALITIES: &[Modality] = &[Language, Embedding, SpeechToText, TextToSpeech];

const BASE: ProviderSpec = ProviderSpec {
    name: "",
    display_name: "",
    modalities: LANGUAGE_ONLY,
    required_env: &[],
    required_any_env: &[],
    optional_env: &[],
    test_model: None,
    test_model_type: Language,
    docs_url: None,
    openai_compat_discovery_url: None,
};

/// Every supported provider, in display order.
pub static PROVIDERS: &[ProviderSpec] = &[
    ProviderSpec {
        name: "openai",
        display_name: "OpenAI",
        modalities: ALL_MODALITIES,
        required_env: &["OPENAI_API_KEY"],
        test_model: Some("gpt-3.5-turbo"),
        docs_url: Some("https://platform.openai.com/api-keys"),
        openai_compat_discovery_url: Some("https://api.openai.com/v1/models"),
        ..BASE
    },
    ProviderSpec {
        name: "anthropic",
        display_name: "Anthropic",
        modalities: LANGUAGE_ONLY,
        required_env: &["ANTHROPIC_API_KEY"],
        test_model: Some("claude-haiku-4-5-20251001"),
        docs_url: Some("https://console.anthropic.com/settings/keys"),
        ..BASE
    },
    ProviderSpec {
        name: "google",
        display_name: "Google AI",
        modalities: ALL_MODALITIES,
        required_any_env: &["GOOGLE_API_KEY", "GEMINI_API_KEY"],
        test_model: Some("gemini-flash-latest"),
        docs_url: Some("https://aistudio.google.com/app/apikey"),
        ..BASE
    },
    ProviderSpec {
        name: "groq",
        display_name: "Groq",
        modalities: &[Language, SpeechToText],
        required_env: &["GROQ_API_KEY"],
        test_model: Some("llama-3.1-8b-instant"),
        docs_url: Some("https://console.groq.com/keys"),
        openai_compat_discovery_url: Some("https://api.groq.com/openai/v1/models"),
        ..BASE
    },
    ProviderSpec {
        name: "mistral",
        display_name: "Mistral AI",
        modalities: &[Language, Embedding, SpeechToText, TextToSpeech],
        required_env: &["MISTRAL_API_KEY"],
        test_model: Some("mistral-small-latest"),
        docs_url: Some("https://console.mistral.ai/api-keys/"),
        openai_compat_discovery_url: Some("https://api.mistral.ai/v1/models"),
        ..BASE
    },
    ProviderSpec {
        name: "deepseek",
        display_name: "DeepSeek",
        modalities: LANGUAGE_ONLY,
        required_env: &["DEEPSEEK_API_KEY"],
        test_model: Some("deepseek-chat"),
        docs_url: Some("https://platform.deepseek.com/api_keys"),
        openai_compat_discovery_url: Some("https://api.deepseek.com/models"),
        ..BASE
    },
    ProviderSpec {
        name: "xai",
        display_name: "xAI (Grok)",
        modalities: &[Language, TextToSpeech],
        required_env: &["XAI_API_KEY"],
        test_model: Some("grok-beta"),
        docs_url: Some("https://console.x.ai/"),
        openai_compat_discovery_url: Some("https://api.x.ai/v1/models"),
        ..BASE
    },
    ProviderSpec {
        name: "openrouter",
        display_name: "OpenRouter",
        modalities: ALL_MODALITIES,
        required_env: &["OPENROUTER_API_KEY"],
        test_model: Some("openai/gpt-3.5-turbo"),
        docs_url: Some("https://openrouter.ai/keys"),
        openai_compat_discovery_url: Some("https://openrouter.ai/api/v1/models"),
        ..BASE
    },
    ProviderSpec {
        name: "dashscope",
        display_name: "DashScope (Qwen)",
        modalities: LANGUAGE_ONLY,
        required_env: &["DASHSCOPE_API_KEY"],
        test_model: Some("qwen-plus"),
        docs_url: Some("https://help.aliyun.com/zh/model-studio/getting-started/"),
        openai_compat_discovery_url: Some(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/models",
        ),
        ..BASE
    },
    ProviderSpec {
        name: "minimax",
        display_name: "MiniMax",
        modalities: &[Language, TextToSpeech],
        required_env: &["MINIMAX_API_KEY"],
        // Keys are region-specific; mainland China uses https://api.minimax.cn/v1
        // for both chat and TTS.
        optional_env: &["MINIMAX_BASE_URL"],
        test_model: Some("MiniMax-M3"),
        docs_url: Some("https://platform.minimaxi.com/document/Guides"),
        openai_compat_discovery_url: Some("https://api.minimax.io/v1/models"),
        ..BASE
    },
    ProviderSpec {
        name: "novita",
        display_name: "Novita",
        modalities: LANGUAGE_ONLY,
        required_env: &["NOVITA_API_KEY"],
        test_model: Some("moonshotai/kimi-k2.5"),
        docs_url: Some("https://novita.ai/settings/key-management"),
        openai_compat_discovery_url: Some("https://api.novita.ai/openai/models"),
        ..BASE
    },
    ProviderSpec {
        name: "siliconflow",
        display_name: "SiliconFlow",
        modalities: LANGUAGE_ONLY,
        required_env: &["SILICONFLOW_API_KEY"],
        // Mainland China accounts set the credential's base URL to
        // https://api.siliconflow.cn/v1.
        optional_env: &["SILICONFLOW_BASE_URL"],
        test_model: Some("deepseek-ai/DeepSeek-V3.1-Terminus"),
        docs_url: Some("https://cloud.siliconflow.com/account/ak"),
        openai_compat_discovery_url: Some("https://api.siliconflow.com/v1/models"),
        ..BASE
    },
    ProviderSpec {
        name: "zai",
        display_name: "Z.ai",
        modalities: LANGUAGE_ONLY,
        required_env: &["ZAI_API_KEY"],
        optional_env: &["ZAI_BASE_URL"],
        test_model: Some("glm-5.2"),
        docs_url: Some("https://z.ai/manage-apikey/apikey-list"),
        openai_compat_discovery_url: Some("https://api.z.ai/api/paas/v4/models"),
        ..BASE
    },
    ProviderSpec {
        name: "ppq",
        display_name: "PayPerQ",
        modalities: ALL_MODALITIES,
        required_env: &["PPQ_API_KEY"],
        test_model: Some("auto"),
        docs_url: Some("https://ppq.ai"),
        // PPQ's bare /models returns only chat models; ?type=all is required to
        // also list embedding, STT and TTS models.
        openai_compat_discovery_url: Some("https://api.ppq.ai/v1/models?type=all"),
        ..BASE
    },
    ProviderSpec {
        name: "cohere",
        display_name: "Cohere",
        // Native v2 API (/v2/chat, /v2/embed), not OpenAI-compatible, so
        // discovery is bespoke. Reranking is out of scope.
        modalities: &[Language, Embedding],
        required_env: &["COHERE_API_KEY"],
        test_model: Some("command-a-03-2025"),
        docs_url: Some("https://dashboard.cohere.com/api-keys"),
        ..BASE
    },
    ProviderSpec {
        name: "voyage",
        display_name: "Voyage AI",
        modalities: &[Embedding],
        required_env: &["VOYAGE_API_KEY"],
        test_model: Some("voyage-3-lite"),
        test_model_type: Embedding,
        docs_url: Some("https://dash.voyageai.com/api-keys"),
        ..BASE
    },
    ProviderSpec {
        name: "elevenlabs",
        display_name: "ElevenLabs",
        modalities: &[TextToSpeech, SpeechToText],
        required_env: &["ELEVENLABS_API_KEY"],
        test_model: Some("eleven_multilingual_v2"),
        test_model_type: TextToSpeech,
        docs_url: Some("https://elevenlabs.io/app/settings/api-keys"),
        ..BASE
    },
    ProviderSpec {
        name: "deepgram",
        display_name: "Deepgram",
        modalities: &[TextToSpeech, SpeechToText],
        required_env: &["DEEPGRAM_API_KEY"],
        test_model: Some("aura-2-thalia-en"),
        test_model_type: TextToSpeech,
        docs_url: Some("https://console.deepgram.com/"),
        ..BASE
    },
    ProviderSpec {
        name: "ollama",
        display_name: "Ollama",
        modalities: &[Language, Embedding],
        required_env: &["OLLAMA_API_BASE"],
        ..BASE
    },
    ProviderSpec {
        name: "omlx",
        display_name: "oMLX",
        modalities: &[Language, Embedding],
        required_env: &["OMLX_API_BASE"],
        optional_env: &["OMLX_API_KEY"],
        docs_url: Some(
            "https://github.com/lfnovo/open-notebook/blob/main/docs/5-CONFIGURATION/omlx.md",
        ),
        // No fixed discovery URL: the base URL is user-supplied (default
        // http://localhost:11435/v1) and the API key is optional.
        ..BASE
    },
    ProviderSpec {
        name: "azure",
        display_name: "Azure OpenAI",
        modalities: ALL_MODALITIES,
        required_env: &[
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_VERSION",
        ],
        optional_env: &[
            "AZURE_OPENAI_ENDPOINT_LLM",
            "AZURE_OPENAI_ENDPOINT_EMBEDDING",
            "AZURE_OPENAI_ENDPOINT_STT",
            "AZURE_OPENAI_ENDPOINT_TTS",
        ],
        // An Azure OpenAI deployment name.
        test_model: Some("gpt-35-turbo"),
        docs_url: Some(
            "https://portal.azure.com/#view/Microsoft_Azure_ProjectOxford/CognitiveServicesHub/~/OpenAI",
        ),
        ..BASE
    },
    ProviderSpec {
        name: "vertex",
        display_name: "Google Vertex AI",
        modalities: &[Language, Embedding, TextToSpeech],
        required_env: &["VERTEX_PROJECT", "VERTEX_LOCATION"],
        optional_env: &["GOOGLE_APPLICATION_CREDENTIALS"],
        test_model: Some("gemini-flash-latest"),
        docs_url: Some("https://cloud.google.com/vertex-ai/docs/start/cloud-environment"),
        ..BASE
    },
    ProviderSpec {
        name: "openai_compatible",
        display_name: "OpenAI Compatible",
        modalities: ALL_MODALITIES,
        required_any_env: &["OPENAI_COMPATIBLE_BASE_URL", "OPENAI_COMPATIBLE_API_KEY"],
        docs_url: Some(
            "https://github.com/lfnovo/open-notebook/blob/main/docs/5-CONFIGURATION/openai-compatible.md",
        ),
        ..BASE
    },
    ProviderSpec {
        name: "anthropic_compatible",
        display_name: "Anthropic Compatible",
        modalities: LANGUAGE_ONLY,
        required_env: &["ANTHROPIC_COMPATIBLE_BASE_URL", "ANTHROPIC_COMPATIBLE_API_KEY"],
        docs_url: Some(
            "https://github.com/lfnovo/open-notebook/blob/main/docs/5-CONFIGURATION/ai-providers.md",
        ),
        // Discovery uses Anthropic's GET /v1/models with x-api-key and
        // anthropic-version headers, not the OpenAI-compatible table.
        ..BASE
    },
];

/// All providers in declaration (display) order.
pub fn providers() -> &'static [ProviderSpec] {
    PROVIDERS
}

/// Canonical provider id: lowercase with underscores (`openai-compatible`
/// and `OpenAI_Compatible` both become `openai_compatible`).
pub fn normalize_provider_name(name: &str) -> String {
    name.trim().to_ascii_lowercase().replace('-', "_")
}

/// Look a provider up by name. Case-insensitive; hyphens and underscores are
/// interchangeable.
pub fn provider(name: &str) -> Option<&'static ProviderSpec> {
    let wanted = normalize_provider_name(name);
    PROVIDERS.iter().find(|spec| spec.name == wanted)
}

/// Where a stored credential's `base_url` may point for this provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseUrlRule {
    /// Any host: the provider is a server the user runs or picks (Ollama,
    /// oMLX, Azure, the OpenAI- and Anthropic-compatible providers).
    AnyHost,
    /// Only `https` URLs on one of these domains or their subdomains: the
    /// provider's own API, including its regional endpoints.
    Domains(&'static [&'static str]),
    /// The provider takes no base URL (Vertex AI builds its host from the
    /// location).
    None,
}

/// The [`BaseUrlRule`] of a provider (normalised name). Unknown providers
/// get [`BaseUrlRule::None`].
pub fn base_url_rule(provider: &str) -> BaseUrlRule {
    let domains: &'static [&'static str] = match normalize_provider_name(provider).as_str() {
        "ollama" | "omlx" | "azure" | "openai_compatible" | "anthropic_compatible" => return BaseUrlRule::AnyHost,
        "openai" => &["openai.com"],
        "anthropic" => &["anthropic.com"],
        // Exact API hosts where the parent domain also serves other
        // tenants' content (Cloud Storage, Function Compute, ...).
        "google" => &["generativelanguage.googleapis.com"],
        "groq" => &["groq.com"],
        "mistral" => &["mistral.ai"],
        "deepseek" => &["deepseek.com"],
        "xai" => &["x.ai"],
        "openrouter" => &["openrouter.ai"],
        "dashscope" => &["dashscope.aliyuncs.com", "dashscope-intl.aliyuncs.com"],
        "minimax" => &["minimax.io", "minimaxi.com", "minimax.cn", "minimax.chat"],
        "novita" => &["novita.ai"],
        "siliconflow" => &["siliconflow.com", "siliconflow.cn"],
        "zai" => &["z.ai", "bigmodel.cn"],
        "ppq" => &["ppq.ai"],
        "cohere" => &["cohere.com", "cohere.ai"],
        "voyage" => &["voyageai.com"],
        "elevenlabs" => &["elevenlabs.io"],
        "deepgram" => &["deepgram.com"],
        _ => return BaseUrlRule::None,
    };
    BaseUrlRule::Domains(domains)
}

impl BaseUrlRule {
    /// Whether a URL with this scheme and host (lower-case ASCII) is allowed.
    pub fn allows(&self, scheme: &str, host: &str) -> bool {
        match self {
            BaseUrlRule::AnyHost => true,
            BaseUrlRule::None => false,
            BaseUrlRule::Domains(domains) => {
                let host = host.trim_end_matches('.').to_ascii_lowercase();
                scheme.eq_ignore_ascii_case("https")
                    && domains.iter().any(|d| host == *d || host.strip_suffix(d).is_some_and(|rest| rest.ends_with('.')))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn base_url_rules_cover_every_provider() {
        for spec in PROVIDERS {
            let rule = base_url_rule(spec.name);
            match spec.name {
                "vertex" => assert_eq!(rule, BaseUrlRule::None),
                "ollama" | "omlx" | "azure" | "openai_compatible" | "anthropic_compatible" => {
                    assert_eq!(rule, BaseUrlRule::AnyHost)
                }
                other => assert!(matches!(rule, BaseUrlRule::Domains(d) if !d.is_empty()), "{other}"),
            }
        }
        let openai = base_url_rule("openai");
        assert!(openai.allows("https", "api.openai.com"));
        assert!(openai.allows("HTTPS", "API.OpenAI.com."));
        assert!(!openai.allows("http", "api.openai.com"), "plain http is refused");
        assert!(!openai.allows("https", "attacker.example"));
        assert!(!openai.allows("https", "evilopenai.com"));
        assert!(!openai.allows("https", "openai.com.attacker.example"));
        assert!(base_url_rule("zai").allows("https", "open.bigmodel.cn"));
        // Shared cloud domains are narrowed to the API host.
        assert!(base_url_rule("google").allows("https", "generativelanguage.googleapis.com"));
        assert!(!base_url_rule("google").allows("https", "storage.googleapis.com"));
        assert!(base_url_rule("dashscope").allows("https", "dashscope-intl.aliyuncs.com"));
        assert!(!base_url_rule("dashscope").allows("https", "attacker.oss-cn-hangzhou.aliyuncs.com"));
        assert!(!base_url_rule("dashscope").allows("https", "fc.aliyuncs.com"));
        assert!(base_url_rule("ollama").allows("http", "192.168.1.10"));
        assert!(!base_url_rule("vertex").allows("https", "aiplatform.googleapis.com"));
    }

    #[test]
    fn names_are_unique_and_ordered_as_declared() {
        let names: Vec<&str> = PROVIDERS.iter().map(|s| s.name).collect();
        let unique: HashSet<&str> = names.iter().copied().collect();
        assert_eq!(unique.len(), names.len());
        assert_eq!(
            names,
            [
                "openai",
                "anthropic",
                "google",
                "groq",
                "mistral",
                "deepseek",
                "xai",
                "openrouter",
                "dashscope",
                "minimax",
                "novita",
                "siliconflow",
                "zai",
                "ppq",
                "cohere",
                "voyage",
                "elevenlabs",
                "deepgram",
                "ollama",
                "omlx",
                "azure",
                "vertex",
                "openai_compatible",
                "anthropic_compatible",
            ]
        );
    }

    #[test]
    fn every_provider_is_well_formed() {
        for spec in PROVIDERS {
            assert!(!spec.display_name.is_empty(), "{}", spec.name);
            assert!(!spec.modalities.is_empty(), "{}", spec.name);
            assert!(
                !spec.required_env.is_empty() || !spec.required_any_env.is_empty(),
                "{} has no env configuration",
                spec.name
            );
            if spec.test_model.is_some() {
                assert!(spec.supports(spec.test_model_type), "{}", spec.name);
            }
            // The API key env var of a table-driven provider is its single
            // required env var.
            if spec.openai_compat_discovery_url.is_some() {
                assert_eq!(spec.required_env.len(), 1, "{}", spec.name);
            }
        }
    }

    #[test]
    fn openai_compat_table_covers_the_same_providers_as_the_original() {
        let table: HashSet<&str> = PROVIDERS
            .iter()
            .filter(|s| s.openai_compat_discovery_url.is_some())
            .map(|s| s.name)
            .collect();
        let expected: HashSet<&str> = [
            "openai",
            "groq",
            "mistral",
            "deepseek",
            "xai",
            "openrouter",
            "dashscope",
            "minimax",
            "novita",
            "siliconflow",
            "zai",
            "ppq",
        ]
        .into_iter()
        .collect();
        assert_eq!(table, expected);
        assert_eq!(
            provider("ppq").unwrap().openai_compat_discovery_url,
            Some("https://api.ppq.ai/v1/models?type=all")
        );
    }

    #[test]
    fn lookup_normalises_case_and_hyphens() {
        assert_eq!(provider("OpenAI-Compatible").unwrap().name, "openai_compatible");
        assert_eq!(provider(" anthropic ").unwrap().name, "anthropic");
        assert!(provider("nope").is_none());
    }

    #[test]
    fn base_url_env_is_derived_from_optional_env() {
        assert_eq!(provider("minimax").unwrap().base_url_env(), Some("MINIMAX_BASE_URL"));
        assert_eq!(provider("siliconflow").unwrap().base_url_env(), Some("SILICONFLOW_BASE_URL"));
        assert_eq!(provider("zai").unwrap().base_url_env(), Some("ZAI_BASE_URL"));
        assert_eq!(provider("openai").unwrap().base_url_env(), None);
        // OMLX_API_KEY and the Azure endpoints are optional but not base URLs.
        assert_eq!(provider("omlx").unwrap().base_url_env(), None);
        assert_eq!(provider("azure").unwrap().base_url_env(), None);
    }

    #[test]
    fn env_config_serialises_in_the_legacy_shape() {
        let openai = serde_json::to_value(provider("openai").unwrap().env_config()).unwrap();
        assert_eq!(openai, serde_json::json!({"required": ["OPENAI_API_KEY"]}));
        let google = serde_json::to_value(provider("google").unwrap().env_config()).unwrap();
        assert_eq!(
            google,
            serde_json::json!({"required_any": ["GOOGLE_API_KEY", "GEMINI_API_KEY"]})
        );
        let omlx = serde_json::to_value(provider("omlx").unwrap().env_config()).unwrap();
        assert_eq!(
            omlx,
            serde_json::json!({"required": ["OMLX_API_BASE"], "optional": ["OMLX_API_KEY"]})
        );
    }

    #[test]
    fn env_configured_follows_required_any_then_required() {
        let env = |pairs: &'static [(&'static str, &'static str)]| {
            move |key: &str| {
                pairs.iter().find(|(k, _)| *k == key).map(|(_, v)| v.to_string())
            }
        };
        let google = provider("google").unwrap();
        assert!(google.is_env_configured(env(&[("GEMINI_API_KEY", "k")])));
        assert!(!google.is_env_configured(env(&[("GEMINI_API_KEY", "  ")])));
        let azure = provider("azure").unwrap();
        assert!(!azure.is_env_configured(env(&[("AZURE_OPENAI_API_KEY", "k")])));
        assert!(azure.is_env_configured(env(&[
            ("AZURE_OPENAI_API_KEY", "k"),
            ("AZURE_OPENAI_ENDPOINT", "https://x"),
            ("AZURE_OPENAI_API_VERSION", "2024-10-21"),
        ])));
    }

    #[test]
    fn provider_info_matches_the_api_response_shape() {
        let info = provider("groq").unwrap().info(true);
        assert_eq!(
            serde_json::to_value(info).unwrap(),
            serde_json::json!({
                "name": "groq",
                "display_name": "Groq",
                "modalities": ["language", "speech_to_text"],
                "docs_url": "https://console.groq.com/keys",
                "env_configured": true,
            })
        );
        let ollama = serde_json::to_value(provider("ollama").unwrap().info(false)).unwrap();
        assert_eq!(ollama["docs_url"], serde_json::Value::Null);
    }

    #[test]
    fn modality_round_trips() {
        for m in Modality::ALL {
            assert_eq!(Modality::parse(m.as_str()), Some(m));
        }
        assert_eq!(Modality::parse("vision"), None);
    }
}
