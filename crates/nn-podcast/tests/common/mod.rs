//! Mock models shared by the integration tests. Nothing here touches the
//! network: replies are scripted and audio is synthesized in memory.

#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use nn_llm::{
    AiError, AiResult, AudioData, ChatMessage, ChatOptions, ChatResponse, LanguageModel, TextStream,
    TextToSpeechModel,
};
use nn_podcast::models::{Speaker, SpeakerProfile};

// ---------------------------------------------------------------------------
// Language model
// ---------------------------------------------------------------------------

/// One recorded chat call.
#[derive(Debug, Clone)]
pub struct Call {
    pub messages: Vec<ChatMessage>,
    pub options: ChatOptions,
}

impl Call {
    /// The first user message: the rendered prompt.
    pub fn prompt(&self) -> &str {
        &self.messages[0].content
    }
}

/// A language model that answers from a script, in order.
pub struct ScriptedModel {
    name: String,
    replies: Mutex<VecDeque<AiResult<ChatResponse>>>,
    calls: Mutex<Vec<Call>>,
}

impl ScriptedModel {
    pub fn new(name: &str, replies: Vec<AiResult<ChatResponse>>) -> Arc<Self> {
        Arc::new(Self { name: name.to_string(), replies: Mutex::new(replies.into()), calls: Mutex::new(Vec::new()) })
    }

    pub fn calls(&self) -> Vec<Call> {
        self.calls.lock().unwrap().clone()
    }

    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }
}

#[async_trait]
impl LanguageModel for ScriptedModel {
    fn provider(&self) -> &str {
        "mock"
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    async fn chat(&self, messages: &[ChatMessage], options: &ChatOptions) -> AiResult<ChatResponse> {
        self.calls.lock().unwrap().push(Call { messages: messages.to_vec(), options: options.clone() });
        self.replies
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Err(AiError::Other("the test scripted no further reply".into())))
    }

    async fn chat_stream(&self, _: &[ChatMessage], _: &ChatOptions) -> AiResult<TextStream> {
        Err(AiError::Unsupported("streaming is not used by the podcast engine".into()))
    }
}

/// A normal reply.
pub fn reply(content: &str) -> AiResult<ChatResponse> {
    reply_with(content, "stop")
}

pub fn reply_with(content: &str, finish_reason: &str) -> AiResult<ChatResponse> {
    Ok(ChatResponse {
        content: content.to_string(),
        model: "mock-model".into(),
        finish_reason: Some(finish_reason.to_string()),
        usage: None,
    })
}

pub fn refused() -> AiResult<ChatResponse> {
    reply_with("", "refusal")
}

pub const OUTLINE_JSON: &str = r#"{"segments": [
    {"name": "Setting out", "description": "Why   the topic\nmatters.", "size": "short"},
    {"name": "Wrapping up", "description": "What to remember.", "size": "medium"}
]}"#;

pub fn transcript_json(turns: &[(&str, &str)]) -> String {
    let entries: Vec<serde_json::Value> = turns
        .iter()
        .map(|(speaker, dialogue)| serde_json::json!({"speaker": speaker, "dialogue": dialogue}))
        .collect();
    serde_json::json!({"transcript": entries}).to_string()
}

// ---------------------------------------------------------------------------
// Speakers
// ---------------------------------------------------------------------------

pub fn speakers() -> Vec<Speaker> {
    vec![
        Speaker::new("Velvet Noir", "af_bella", "A late-night radio host.", "Wry and warm."),
        Speaker::new("Max", "am_adam", "A field reporter.", "Direct."),
    ]
}

pub fn speaker_profile() -> SpeakerProfile {
    SpeakerProfile {
        id: Some("speaker_profile:sp1".into()),
        name: "velvet_noir".into(),
        description: None,
        voice_model: Some("model:tts1".into()),
        speakers: speakers(),
    }
}

// ---------------------------------------------------------------------------
// Text to speech
// ---------------------------------------------------------------------------

/// MPEG-2 layer III, 64 kbit/s, 24 kHz, mono: one silent 192-byte frame is 24 ms.
pub const MP3_FRAME_HEADER: [u8; 4] = [0xFF, 0xF3, 0x84, 0xC0];
pub const MP3_FRAME_SECONDS: f64 = 0.024;

/// A silent MP3 clip of `frames` frames, tagged the way real clips are (an
/// ID3v2 tag in front), so joining has something to strip.
pub fn mp3_clip(frames: usize) -> Vec<u8> {
    let mut frame = MP3_FRAME_HEADER.to_vec();
    frame.resize(192, 0);
    let mut clip = b"ID3\x04\x00\x00\x00\x00\x00\x10".to_vec();
    clip.extend([0u8; 16]);
    clip.extend(frame.repeat(frames));
    clip
}

/// A silent 16-bit mono WAV clip.
pub fn wav_clip(sample_rate: u32, seconds: f64) -> Vec<u8> {
    let samples = (f64::from(sample_rate) * seconds).round() as u32;
    let data_len = samples * 2;
    let mut out = b"RIFF".to_vec();
    out.extend((36 + data_len).to_le_bytes());
    out.extend(b"WAVEfmt ");
    out.extend(16u32.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(sample_rate.to_le_bytes());
    out.extend((sample_rate * 2).to_le_bytes());
    out.extend(2u16.to_le_bytes());
    out.extend(16u16.to_le_bytes());
    out.extend(b"data");
    out.extend(data_len.to_le_bytes());
    out.extend(std::iter::repeat_n(0u8, data_len as usize));
    out
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipFormat {
    Mp3,
    Wav,
}

/// A speech model that returns silent clips: one MP3 frame (24 ms) per
/// character of text, so every clip's length is known to the test.
pub struct MockTts {
    name: String,
    format: ClipFormat,
    /// How long each call takes; earlier calls can be made slower than later
    /// ones to scramble the completion order.
    delay: Box<dyn Fn(&str) -> Duration + Send + Sync>,
    /// Errors returned before any audio, in order.
    failures: Mutex<VecDeque<AiError>>,
    calls: Mutex<Vec<(String, String)>>,
    in_flight: AtomicUsize,
    max_in_flight: AtomicUsize,
}

impl MockTts {
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            format: ClipFormat::Mp3,
            delay: Box::new(|_| Duration::ZERO),
            failures: Mutex::new(VecDeque::new()),
            calls: Mutex::new(Vec::new()),
            in_flight: AtomicUsize::new(0),
            max_in_flight: AtomicUsize::new(0),
        }
    }

    pub fn format(mut self, format: ClipFormat) -> Self {
        self.format = format;
        self
    }

    pub fn delay(mut self, delay: impl Fn(&str) -> Duration + Send + Sync + 'static) -> Self {
        self.delay = Box::new(delay);
        self
    }

    pub fn failing_with(self, failures: Vec<AiError>) -> Self {
        *self.failures.lock().unwrap() = failures.into();
        self
    }

    pub fn shared(self) -> Arc<Self> {
        Arc::new(self)
    }

    /// `(voice, text)` of every call, in the order the calls started.
    pub fn calls(&self) -> Vec<(String, String)> {
        self.calls.lock().unwrap().clone()
    }

    pub fn max_in_flight(&self) -> usize {
        self.max_in_flight.load(Ordering::SeqCst)
    }
}

#[async_trait]
impl TextToSpeechModel for MockTts {
    fn provider(&self) -> &str {
        "mock"
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    async fn synthesize(&self, text: &str, voice: &str) -> AiResult<AudioData> {
        self.calls.lock().unwrap().push((voice.to_string(), text.to_string()));
        let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
        self.max_in_flight.fetch_max(now, Ordering::SeqCst);
        tokio::time::sleep((self.delay)(text)).await;
        self.in_flight.fetch_sub(1, Ordering::SeqCst);

        if let Some(error) = self.failures.lock().unwrap().pop_front() {
            return Err(error);
        }
        let frames = text.chars().count();
        Ok(match self.format {
            ClipFormat::Mp3 => {
                AudioData { bytes: Bytes::from(mp3_clip(frames)), content_type: "audio/mpeg".into() }
            }
            ClipFormat::Wav => AudioData {
                bytes: Bytes::from(wav_clip(24_000, frames as f64 * MP3_FRAME_SECONDS)),
                content_type: "audio/wav".into(),
            },
        })
    }
}
