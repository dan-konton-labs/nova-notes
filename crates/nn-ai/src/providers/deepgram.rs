//! Deepgram: Aura text-to-speech (`/v1/speak`) and Nova/Whisper
//! speech-to-text (`/v1/listen`).
//!
//! Deepgram has no separate voice parameter: each Aura voice is its own model
//! id (`aura-2-thalia-en`). A voice passed to `synthesize` therefore selects
//! the model for that call; an empty voice uses the configured model.

use async_trait::async_trait;
use nn_llm::{AiResult, AudioData, SpeechToTextModel, TextToSpeechModel, TranscriptionRequest, Voice};
use serde_json::{json, Value};

use super::require_key;
use crate::audio::audio_mime;
use crate::config::{non_blank, ProviderConfig, DEFAULT_SPEECH_TIMEOUT};
use crate::http::{join_url, Http};

pub(crate) const DEEPGRAM_API_ROOT: &str = "https://api.deepgram.com/v1";

/// A representative subset of the Aura-2 English voices.
pub(crate) const DEEPGRAM_VOICES: &[&str] = &[
    "aura-2-thalia-en",
    "aura-2-andromeda-en",
    "aura-2-helena-en",
    "aura-2-apollo-en",
    "aura-2-arcas-en",
    "aura-2-asteria-en",
    "aura-2-athena-en",
    "aura-2-hera-en",
    "aura-2-hermes-en",
    "aura-2-atlas-en",
];

struct DeepgramEndpoint {
    http: Http,
    model: String,
    base: String,
    api_key: String,
}

impl DeepgramEndpoint {
    fn new(client: &reqwest::Client, model: &str, config: &ProviderConfig) -> AiResult<Self> {
        Ok(Self {
            http: Http::new(client, "deepgram", config, DEFAULT_SPEECH_TIMEOUT),
            model: model.to_string(),
            base: config.url().unwrap_or(DEEPGRAM_API_ROOT).to_string(),
            api_key: require_key("deepgram", config)?,
        })
    }

    fn post(&self, path: &str) -> reqwest::RequestBuilder {
        // Deepgram's scheme is "Token", not "Bearer".
        self.http.client.post(join_url(&self.base, path)).header("authorization", format!("Token {}", self.api_key))
    }
}

pub(crate) struct DeepgramSpeech {
    endpoint: DeepgramEndpoint,
}

impl DeepgramSpeech {
    pub fn new(client: &reqwest::Client, model: &str, config: &ProviderConfig) -> AiResult<Self> {
        Ok(Self { endpoint: DeepgramEndpoint::new(client, model, config)? })
    }
}

#[async_trait]
impl TextToSpeechModel for DeepgramSpeech {
    fn provider(&self) -> &str {
        "deepgram"
    }

    fn model_name(&self) -> &str {
        &self.endpoint.model
    }

    async fn synthesize(&self, text: &str, voice: &str) -> AiResult<AudioData> {
        let ep = &self.endpoint;
        let model = if voice.trim().is_empty() { ep.model.as_str() } else { voice.trim() };
        let request = ep.post("speak").query(&[("model", model), ("encoding", "mp3")]).json(&json!({"text": text}));
        let (bytes, content_type) = ep.http.bytes(request).await?;
        if bytes.is_empty() {
            return Err(ep.http.malformed("the audio response was empty"));
        }
        let content_type = content_type.filter(|ct| ct.starts_with("audio/")).unwrap_or_else(|| "audio/mpeg".to_string());
        Ok(AudioData { bytes, content_type })
    }

    async fn voices(&self) -> AiResult<Vec<Voice>> {
        Ok(DEEPGRAM_VOICES
            .iter()
            .map(|id| {
                // "aura-2-thalia-en" -> name "Thalia", language "en".
                let mut parts = id.split('-').skip(2);
                let name = parts.next().unwrap_or(id);
                let name = name[..1].to_ascii_uppercase() + &name[1..];
                Voice { id: id.to_string(), name, language: parts.next().map(str::to_string), gender: None }
            })
            .collect())
    }
}

pub(crate) struct DeepgramTranscriber {
    endpoint: DeepgramEndpoint,
}

impl DeepgramTranscriber {
    pub fn new(client: &reqwest::Client, model: &str, config: &ProviderConfig) -> AiResult<Self> {
        Ok(Self { endpoint: DeepgramEndpoint::new(client, model, config)? })
    }
}

#[async_trait]
impl SpeechToTextModel for DeepgramTranscriber {
    fn provider(&self) -> &str {
        "deepgram"
    }

    fn model_name(&self) -> &str {
        &self.endpoint.model
    }

    async fn transcribe(&self, request: TranscriptionRequest) -> AiResult<String> {
        let ep = &self.endpoint;
        let mut query = vec![("model", ep.model.clone()), ("smart_format", "true".to_string())];
        match non_blank(&request.language) {
            Some(language) => query.push(("language", language.to_string())),
            None => query.push(("detect_language", "true".to_string())),
        }
        let mime = non_blank(&request.content_type).unwrap_or_else(|| audio_mime(&request.filename)).to_string();
        // The audio is the raw request body; Deepgram has no prompt parameter.
        let builder = ep.post("listen").query(&query).header("content-type", mime).body(request.audio.clone());
        let json = ep.http.json(builder).await?;
        json.pointer("/results/channels/0/alternatives/0/transcript")
            .and_then(Value::as_str)
            .map(|text| text.trim().to_string())
            .ok_or_else(|| ep.http.malformed("no transcript in the response"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::shared_client;
    use crate::providers::testing::{header, request, request_json};
    use bytes::Bytes;
    use nn_llm::AiError;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config(server: &MockServer) -> ProviderConfig {
        ProviderConfig::new().with_api_key("dg-test").with_base_url(format!("{}/v1", server.uri()))
    }

    #[tokio::test]
    async fn speech_selects_the_voice_through_the_model_parameter() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/speak"))
            .respond_with(ResponseTemplate::new(200).insert_header("content-type", "audio/mpeg").set_body_bytes(b"ID3".to_vec()))
            .mount(&server)
            .await;
        let tts = DeepgramSpeech::new(&shared_client(), "aura-2-thalia-en", &config(&server)).unwrap();
        let audio = tts.synthesize("Hello", "aura-2-apollo-en").await.unwrap();
        assert_eq!(audio.content_type, "audio/mpeg");
        tts.synthesize("Hello", "").await.unwrap();

        let first = request(&server, 0).await;
        assert_eq!(header(&first, "authorization"), Some("Token dg-test"));
        assert_eq!(first.url.query(), Some("model=aura-2-apollo-en&encoding=mp3"));
        assert_eq!(request_json(&server, 0).await, json!({"text": "Hello"}));
        assert_eq!(request(&server, 1).await.url.query(), Some("model=aura-2-thalia-en&encoding=mp3"));

        let voices = tts.voices().await.unwrap();
        assert_eq!(voices[0], Voice { id: "aura-2-thalia-en".into(), name: "Thalia".into(), language: Some("en".into()), gender: None });
    }

    #[tokio::test]
    async fn transcription_posts_raw_audio() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/listen"))
            .and(query_param("model", "nova-3"))
            .and(query_param("smart_format", "true"))
            .and(query_param("language", "en"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "results": {"channels": [{"alternatives": [{"transcript": "Hello there.", "confidence": 0.99}]}]},
            })))
            .mount(&server)
            .await;
        let stt = DeepgramTranscriber::new(&shared_client(), "nova-3", &config(&server)).unwrap();
        let text = stt
            .transcribe(TranscriptionRequest {
                audio: Bytes::from_static(b"RIFFaudio"),
                filename: "a.wav".into(),
                content_type: None,
                language: Some("en".into()),
                prompt: None,
            })
            .await
            .unwrap();
        assert_eq!(text, "Hello there.");
        let sent = request(&server, 0).await;
        assert_eq!(header(&sent, "content-type"), Some("audio/wav"));
        assert_eq!(sent.body, b"RIFFaudio");
    }

    #[tokio::test]
    async fn deepgram_errors_are_classified() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({"err_code": "INVALID_AUTH", "err_msg": "Invalid credentials."})))
            .mount(&server)
            .await;
        let tts = DeepgramSpeech::new(&shared_client(), "aura-2-thalia-en", &config(&server)).unwrap();
        let err = tts.synthesize("x", "").await.unwrap_err();
        assert!(matches!(&err, AiError::Authentication(m) if m == "deepgram: Invalid credentials. (HTTP 401)"), "{err:?}");
    }
}
