//! ElevenLabs text-to-speech (`/v1/text-to-speech/{voice_id}`) and Scribe
//! speech-to-text (`/v1/speech-to-text`).

use async_trait::async_trait;
use nn_llm::{AiError, AiResult, AudioData, SpeechToTextModel, TextToSpeechModel, TranscriptionRequest, Voice};
use reqwest::multipart::{Form, Part};
use serde_json::{json, Value};

use super::require_key;
use crate::audio::audio_mime;
use crate::config::{non_blank, ProviderConfig, DEFAULT_SPEECH_TIMEOUT};
use crate::http::{encode_component, join_url, Http};

pub(crate) const ELEVENLABS_API_ROOT: &str = "https://api.elevenlabs.io/v1";

struct ElevenLabsEndpoint {
    http: Http,
    model: String,
    base: String,
    api_key: String,
}

impl ElevenLabsEndpoint {
    fn new(client: &reqwest::Client, model: &str, config: &ProviderConfig) -> AiResult<Self> {
        Ok(Self {
            http: Http::new(client, "elevenlabs", config, DEFAULT_SPEECH_TIMEOUT),
            model: model.to_string(),
            base: config.url().unwrap_or(ELEVENLABS_API_ROOT).to_string(),
            api_key: require_key("elevenlabs", config)?,
        })
    }
}

pub(crate) struct ElevenLabsSpeech {
    endpoint: ElevenLabsEndpoint,
    output_format: String,
}

impl ElevenLabsSpeech {
    pub fn new(client: &reqwest::Client, model: &str, config: &ProviderConfig) -> AiResult<Self> {
        Ok(Self {
            endpoint: ElevenLabsEndpoint::new(client, model, config)?,
            output_format: config.extra_str("output_format").unwrap_or("mp3_44100_128").to_string(),
        })
    }
}

/// MIME type of an ElevenLabs `output_format` such as `mp3_44100_128`.
fn format_mime(output_format: &str) -> &'static str {
    match output_format.split('_').next().unwrap_or_default() {
        "pcm" => "audio/pcm",
        "ulaw" => "audio/basic",
        "opus" => "audio/ogg",
        "wav" => "audio/wav",
        _ => "audio/mpeg",
    }
}

#[async_trait]
impl TextToSpeechModel for ElevenLabsSpeech {
    fn provider(&self) -> &str {
        "elevenlabs"
    }

    fn model_name(&self) -> &str {
        &self.endpoint.model
    }

    async fn synthesize(&self, text: &str, voice: &str) -> AiResult<AudioData> {
        let ep = &self.endpoint;
        let voice = voice.trim();
        if voice.is_empty() {
            return Err(AiError::InvalidRequest("elevenlabs: a voice id is required".to_string()));
        }
        let url = join_url(&ep.base, &format!("text-to-speech/{}", encode_component(voice)));
        let request = ep
            .http
            .client
            .post(url)
            .query(&[("output_format", self.output_format.as_str())])
            .header("xi-api-key", &ep.api_key)
            .json(&json!({"text": text, "model_id": ep.model}));
        let (bytes, content_type) = ep.http.bytes(request).await?;
        if bytes.is_empty() {
            return Err(ep.http.malformed("the audio response was empty"));
        }
        let content_type = content_type
            .filter(|ct| ct.starts_with("audio/"))
            .unwrap_or_else(|| format_mime(&self.output_format).to_string());
        Ok(AudioData { bytes, content_type })
    }

    async fn voices(&self) -> AiResult<Vec<Voice>> {
        let ep = &self.endpoint;
        let request = ep.http.client.get(join_url(&ep.base, "voices")).header("xi-api-key", &ep.api_key);
        let json = ep.http.json(request).await?;
        let label = |voice: &Value, name: &str| {
            voice.pointer(&format!("/labels/{name}")).and_then(Value::as_str).map(str::to_string)
        };
        Ok(json
            .get("voices")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter_map(|voice| {
                let id = voice.get("voice_id")?.as_str()?.to_string();
                Some(Voice {
                    name: voice.get("name").and_then(Value::as_str).unwrap_or(&id).to_string(),
                    language: label(voice, "language"),
                    gender: label(voice, "gender"),
                    id,
                })
            })
            .collect())
    }
}

pub(crate) struct ElevenLabsTranscriber {
    endpoint: ElevenLabsEndpoint,
}

impl ElevenLabsTranscriber {
    pub fn new(client: &reqwest::Client, model: &str, config: &ProviderConfig) -> AiResult<Self> {
        Ok(Self { endpoint: ElevenLabsEndpoint::new(client, model, config)? })
    }
}

#[async_trait]
impl SpeechToTextModel for ElevenLabsTranscriber {
    fn provider(&self) -> &str {
        "elevenlabs"
    }

    fn model_name(&self) -> &str {
        &self.endpoint.model
    }

    async fn transcribe(&self, request: TranscriptionRequest) -> AiResult<String> {
        let ep = &self.endpoint;
        let length = request.audio.len() as u64;
        let mime = non_blank(&request.content_type).unwrap_or_else(|| audio_mime(&request.filename)).to_string();
        let file = Part::stream_with_length(request.audio.clone(), length).file_name(request.filename.clone());
        let file = match file.mime_str(&mime) {
            Ok(part) => part,
            Err(_) => Part::stream_with_length(request.audio.clone(), length).file_name(request.filename.clone()),
        };
        // Scribe has no prompt parameter; the hint is not sent.
        let mut form = Form::new().text("model_id", ep.model.clone()).part("file", file);
        if let Some(language) = non_blank(&request.language) {
            form = form.text("language_code", language.to_string());
        }
        let builder =
            ep.http.client.post(join_url(&ep.base, "speech-to-text")).header("xi-api-key", &ep.api_key).multipart(form);
        let json = ep.http.json(builder).await?;
        json.get("text")
            .and_then(Value::as_str)
            .map(|text| text.trim().to_string())
            .ok_or_else(|| ep.http.malformed("no text in the transcription"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::http::shared_client;
    use crate::providers::testing::{header, request, request_json};
    use bytes::Bytes;
    use wiremock::matchers::{method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn config(server: &MockServer) -> ProviderConfig {
        ProviderConfig::new().with_api_key("xi-test").with_base_url(format!("{}/v1", server.uri()))
    }

    #[tokio::test]
    async fn speech_posts_to_the_voice_url() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"))
            .and(query_param("output_format", "mp3_44100_128"))
            .respond_with(ResponseTemplate::new(200).insert_header("content-type", "audio/mpeg").set_body_bytes(b"ID3".to_vec()))
            .mount(&server)
            .await;
        let tts = ElevenLabsSpeech::new(&shared_client(), "eleven_multilingual_v2", &config(&server)).unwrap();
        let audio = tts.synthesize("Hello", "21m00Tcm4TlvDq8ikWAM").await.unwrap();
        assert_eq!(audio.bytes, Bytes::from_static(b"ID3"));
        assert_eq!(audio.content_type, "audio/mpeg");
        let sent = request(&server, 0).await;
        assert_eq!(header(&sent, "xi-api-key"), Some("xi-test"));
        assert_eq!(request_json(&server, 0).await, json!({"text": "Hello", "model_id": "eleven_multilingual_v2"}));

        let err = tts.synthesize("Hello", " ").await.unwrap_err();
        assert!(matches!(err, AiError::InvalidRequest(_)));
    }

    #[tokio::test]
    async fn voices_are_listed_with_labels() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/v1/voices"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"voices": [
                {"voice_id": "v1", "name": "Rachel", "labels": {"gender": "female", "language": "en"}},
                {"voice_id": "v2", "name": "Adam"},
            ]})))
            .mount(&server)
            .await;
        let tts = ElevenLabsSpeech::new(&shared_client(), "eleven_v3", &config(&server)).unwrap();
        let voices = tts.voices().await.unwrap();
        assert_eq!(voices.len(), 2);
        assert_eq!(voices[0], Voice { id: "v1".into(), name: "Rachel".into(), language: Some("en".into()), gender: Some("female".into()) });
        assert_eq!(voices[1].gender, None);
    }

    #[tokio::test]
    async fn transcription_uploads_multipart_with_model_id() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/speech-to-text"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"language_code": "en", "text": "Hello there"})))
            .mount(&server)
            .await;
        let stt = ElevenLabsTranscriber::new(&shared_client(), "scribe_v1", &config(&server)).unwrap();
        let text = stt
            .transcribe(TranscriptionRequest {
                audio: Bytes::from_static(b"OggSaudio"),
                filename: "memo.ogg".into(),
                content_type: Some("audio/ogg".into()),
                language: Some("en".into()),
                prompt: None,
            })
            .await
            .unwrap();
        assert_eq!(text, "Hello there");
        let sent = request(&server, 0).await;
        assert_eq!(header(&sent, "xi-api-key"), Some("xi-test"));
        let body = String::from_utf8_lossy(&sent.body).to_lowercase();
        for expected in ["name=\"model_id\"\r\n\r\nscribe_v1", "name=\"language_code\"\r\n\r\nen", "filename=\"memo.ogg\"", "content-type: audio/ogg", "oggsaudio"] {
            assert!(body.contains(expected), "missing {expected:?}");
        }
    }

    #[tokio::test]
    async fn detail_errors_are_classified() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .respond_with(ResponseTemplate::new(401).set_body_json(json!({"detail": {"status": "invalid_api_key", "message": "Invalid API key"}})))
            .mount(&server)
            .await;
        let tts = ElevenLabsSpeech::new(&shared_client(), "eleven_v3", &config(&server)).unwrap();
        let err = tts.synthesize("x", "voice").await.unwrap_err();
        assert!(matches!(&err, AiError::Authentication(m) if m == "elevenlabs: Invalid API key (HTTP 401)"), "{err:?}");
    }
}
