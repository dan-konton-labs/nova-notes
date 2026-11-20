//! MiniMax text-to-speech through its native `/v1/t2a_v2` endpoint.
//!
//! MiniMax chat is OpenAI-compatible and lives in [`super::openai`]; speech is
//! not. The audio comes back hex-encoded inside a JSON envelope, and failures
//! are reported in `base_resp` with HTTP status 200.

use async_trait::async_trait;
use nn_llm::{AiError, AiResult, AudioData, TextToSpeechModel, Voice};
use serde_json::{json, Value};

use super::require_key;
use crate::config::{ProviderConfig, DEFAULT_SPEECH_TIMEOUT};
use crate::error::redact;
use crate::http::{join_url, Http};

const DEFAULT_VOICE: &str = "English_Graceful_Lady";
/// Offered when the account's voice list cannot be fetched.
const FALLBACK_VOICES: &[&str] = &[DEFAULT_VOICE, "English_expressive_narrator"];

pub(crate) struct MiniMaxSpeech {
    http: Http,
    model: String,
    base: String,
    api_key: String,
}

impl MiniMaxSpeech {
    pub fn new(client: &reqwest::Client, model: &str, config: &ProviderConfig) -> AiResult<Self> {
        Ok(Self {
            http: Http::new(client, "minimax", config, DEFAULT_SPEECH_TIMEOUT),
            model: model.to_string(),
            // The same base URL serves chat and speech (region-specific keys).
            base: config.url().or_else(|| super::openai::default_base_url("minimax")).unwrap_or_default().to_string(),
            api_key: require_key("minimax", config)?,
        })
    }

    fn post(&self, path: &str, body: &Value) -> reqwest::RequestBuilder {
        self.http.client.post(join_url(&self.base, path)).bearer_auth(&self.api_key).json(body)
    }

    /// MiniMax signals errors in `base_resp.status_code` with HTTP 200.
    fn check(&self, json: &Value) -> AiResult<()> {
        let code = json.pointer("/base_resp/status_code").and_then(Value::as_i64).unwrap_or(0);
        if code == 0 {
            return Ok(());
        }
        let detail = json.pointer("/base_resp/status_msg").and_then(Value::as_str).unwrap_or("request failed");
        let message = format!("minimax: {} (code {code})", redact(detail, Some(&self.api_key)));
        Err(match code {
            1004 | 2049 => AiError::Authentication(message),
            // Rate limit, insufficient balance, token-per-minute limit.
            1002 | 1008 | 1039 => AiError::RateLimit(message),
            2013 => AiError::InvalidRequest(message),
            _ => AiError::Provider { status: None, message },
        })
    }
}

fn static_voices() -> Vec<Voice> {
    FALLBACK_VOICES
        .iter()
        .map(|id| Voice { id: id.to_string(), name: id.replace('_', " "), language: None, gender: None })
        .collect()
}

#[async_trait]
impl TextToSpeechModel for MiniMaxSpeech {
    fn provider(&self) -> &str {
        "minimax"
    }

    fn model_name(&self) -> &str {
        &self.model
    }

    async fn synthesize(&self, text: &str, voice: &str) -> AiResult<AudioData> {
        let voice = if voice.trim().is_empty() { DEFAULT_VOICE } else { voice.trim() };
        let body = json!({
            "model": self.model,
            "text": text,
            "stream": false,
            "output_format": "hex",
            "voice_setting": {"voice_id": voice, "speed": 1, "vol": 1, "pitch": 0},
            "audio_setting": {"sample_rate": 32000, "bitrate": 128000, "format": "mp3", "channel": 1},
        });
        let json = self.http.json(self.post("t2a_v2", &body)).await?;
        self.check(&json)?;
        let audio = json
            .pointer("/data/audio")
            .and_then(Value::as_str)
            .and_then(|encoded| hex::decode(encoded.trim()).ok())
            .filter(|bytes| !bytes.is_empty())
            .ok_or_else(|| self.http.malformed("no audio in the response"))?;
        Ok(AudioData { bytes: audio.into(), content_type: "audio/mpeg".to_string() })
    }

    /// System voices plus the account's cloned and designed voices. Falls
    /// back to a few well-known system voices when the listing fails, so
    /// callers always have a valid voice to offer.
    async fn voices(&self) -> AiResult<Vec<Voice>> {
        let listed = match self.http.json(self.post("get_voice", &json!({"voice_type": "all"}))).await {
            Ok(json) if self.check(&json).is_ok() => json,
            _ => return Ok(static_voices()),
        };
        let voices: Vec<Voice> = ["system_voice", "voice_cloning", "voice_generation"]
            .iter()
            .filter_map(|group| listed.get(*group).and_then(Value::as_array))
            .flatten()
            .filter_map(|voice| {
                let id = voice.get("voice_id")?.as_str()?.to_string();
                let name = voice.get("voice_name").and_then(Value::as_str).unwrap_or(&id).to_string();
                Some(Voice { id, name, language: None, gender: None })
            })
            .collect();
        Ok(if voices.is_empty() { static_voices() } else { voices })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::shared_client;
    use crate::providers::testing::{header, request, request_json};
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn speech(server: &MockServer) -> MiniMaxSpeech {
        let config = ProviderConfig::new().with_api_key("mm-test").with_base_url(format!("{}/v1", server.uri()));
        MiniMaxSpeech::new(&shared_client(), "speech-2.8-hd", &config).unwrap()
    }

    #[test]
    fn default_base_url_is_the_international_endpoint() {
        let tts = MiniMaxSpeech::new(&shared_client(), "speech-2.8-hd", &ProviderConfig::new().with_api_key("k")).unwrap();
        assert_eq!(tts.base, "https://api.minimax.io/v1");
    }

    #[tokio::test]
    async fn speech_decodes_hex_audio() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/t2a_v2"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "data": {"audio": hex::encode(b"ID3mp3"), "status": 2},
                "extra_info": {"audio_format": "mp3"},
                "base_resp": {"status_code": 0, "status_msg": "success"},
            })))
            .mount(&server)
            .await;
        let tts = speech(&server);
        let audio = tts.synthesize("Hello", "").await.unwrap();
        assert_eq!(&audio.bytes[..], b"ID3mp3");
        assert_eq!(audio.content_type, "audio/mpeg");
        assert_eq!(header(&request(&server, 0).await, "authorization"), Some("Bearer mm-test"));
        let body = request_json(&server, 0).await;
        assert_eq!(body["model"], json!("speech-2.8-hd"));
        assert_eq!(body["text"], json!("Hello"));
        assert_eq!(body["voice_setting"]["voice_id"], json!("English_Graceful_Lady"));
        assert_eq!(body["audio_setting"]["format"], json!("mp3"));
        assert_eq!(body["stream"], json!(false));
    }

    #[tokio::test]
    async fn base_resp_failures_are_errors_despite_status_200() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"base_resp": {"status_code": 1004, "status_msg": "login fail: invalid api key"}})))
            .mount(&server)
            .await;
        let tts = speech(&server);
        let err = tts.synthesize("Hello", "voice").await.unwrap_err();
        assert!(matches!(&err, AiError::Authentication(m) if m.contains("login fail")), "{err:?}");
        // The voice listing degrades to the static voices instead of failing.
        assert_eq!(tts.voices().await.unwrap()[0].id, "English_Graceful_Lady");
    }

    #[tokio::test]
    async fn voices_merge_system_and_account_voices() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/get_voice"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "system_voice": [{"voice_id": "Wise_Woman", "voice_name": "Wise Woman"}],
                "voice_cloning": [{"voice_id": "my-clone"}],
                "base_resp": {"status_code": 0, "status_msg": "success"},
            })))
            .mount(&server)
            .await;
        let voices = speech(&server).voices().await.unwrap();
        assert_eq!(voices.iter().map(|v| v.id.as_str()).collect::<Vec<_>>(), vec!["Wise_Woman", "my-clone"]);
        assert_eq!(voices[0].name, "Wise Woman");
        assert_eq!(request_json(&server, 0).await, json!({"voice_type": "all"}));
    }
}
