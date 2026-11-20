//! Test support: an in-memory app to send requests to, and scripted models.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use axum::body::Body;
use axum::http::{header, Method, Request, StatusCode};
use axum::Router;
use bytes::Bytes;
use futures::StreamExt;
use http_body_util::BodyExt;
use nn_llm::{
    AiError, AiResult, AudioData, ChatMessage, ChatOptions, ChatResponse, EmbeddingModel, LanguageModel,
    SpeechToTextModel, TextStream, TextToSpeechModel, TranscriptionRequest,
};
use tower::ServiceExt;

use crate::db::Json;
use crate::state::AppState;

/// A migrated in-memory instance with every job handler registered and the
/// worker running, plus the full router.
pub struct TestApp {
    pub state: AppState,
    pub router: Router,
    _dir: tempfile::TempDir,
}

impl TestApp {
    pub async fn new() -> Self {
        let (state, dir) = AppState::for_tests().await.expect("test state");
        crate::services::start(&state).await.expect("services start");
        Self { router: crate::api::build_router(state.clone()), state, _dir: dir }
    }

    /// Send any request and return the raw response.
    pub async fn send(&self, request: Request<Body>) -> axum::response::Response {
        self.router.clone().oneshot(request).await.expect("router is infallible")
    }

    /// Send a request with an optional JSON body; returns status and the
    /// response parsed as JSON (`Null` when it is not JSON).
    pub async fn request(&self, method: Method, path: &str, body: Option<Json>) -> (StatusCode, Json) {
        let builder = Request::builder().method(method).uri(path);
        let request = match body {
            Some(body) => builder
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("serializable body")))
                .expect("valid request"),
            None => builder.body(Body::empty()).expect("valid request"),
        };
        let response = self.send(request).await;
        let status = response.status();
        let bytes = response.into_body().collect().await.expect("readable body").to_bytes();
        (status, serde_json::from_slice(&bytes).unwrap_or(Json::Null))
    }

    /// Send a request and return status and the body as text (for streams, XML, HTML).
    pub async fn text(&self, method: Method, path: &str, body: Option<Json>) -> (StatusCode, String) {
        let builder = Request::builder().method(method).uri(path);
        let request = match body {
            Some(body) => builder
                .header(header::CONTENT_TYPE, "application/json")
                .body(Body::from(serde_json::to_vec(&body).expect("serializable body")))
                .expect("valid request"),
            None => builder.body(Body::empty()).expect("valid request"),
        };
        let response = self.send(request).await;
        let status = response.status();
        let bytes = response.into_body().collect().await.expect("readable body").to_bytes();
        (status, String::from_utf8_lossy(&bytes).into_owned())
    }

    pub async fn get(&self, path: &str) -> (StatusCode, Json) {
        self.request(Method::GET, path, None).await
    }
    pub async fn post(&self, path: &str, body: Json) -> (StatusCode, Json) {
        self.request(Method::POST, path, Some(body)).await
    }
    pub async fn put(&self, path: &str, body: Json) -> (StatusCode, Json) {
        self.request(Method::PUT, path, Some(body)).await
    }
    pub async fn patch(&self, path: &str, body: Json) -> (StatusCode, Json) {
        self.request(Method::PATCH, path, Some(body)).await
    }
    pub async fn delete(&self, path: &str) -> (StatusCode, Json) {
        self.request(Method::DELETE, path, None).await
    }

    /// Wait for a background job and return its final status string.
    pub async fn wait_for_job(&self, command_id: &str) -> String {
        self.state
            .jobs
            .wait(command_id, std::time::Duration::from_secs(20))
            .await
            .expect("job finishes")
            .status_str()
            .to_string()
    }

    /// Wait until no background job is waiting or running. For follow-up jobs
    /// a test has no id for, such as the `create_insight` that
    /// `run_transformation` queues.
    pub async fn wait_for_idle(&self) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        loop {
            let busy = self
                .state
                .db
                .query(
                    "SELECT id, name, status FROM command WHERE status NOT IN $done",
                    crate::db::Params::new().set(
                        "done",
                        [crate::jobs::status::COMPLETED, crate::jobs::status::FAILED, crate::jobs::status::CANCELED],
                    ),
                )
                .await
                .expect("job query");
            if busy.is_empty() {
                return;
            }
            assert!(std::time::Instant::now() < deadline, "background jobs still busy: {busy:?}");
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
        }
    }
}

/// Build a `multipart/form-data` body. Each part is `(name, filename, bytes)`;
/// a `None` filename makes it a plain form field.
pub fn multipart_body(parts: &[(&str, Option<&str>, &[u8])]) -> (String, Vec<u8>) {
    const BOUNDARY: &str = "----nova-notes-test-boundary";
    let mut body = Vec::new();
    for (name, filename, bytes) in parts {
        body.extend_from_slice(format!("--{BOUNDARY}\r\n").as_bytes());
        match filename {
            Some(filename) => body.extend_from_slice(
                format!("Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: application/octet-stream\r\n\r\n").as_bytes(),
            ),
            None => body.extend_from_slice(format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes()),
        }
        body.extend_from_slice(bytes);
        body.extend_from_slice(b"\r\n");
    }
    body.extend_from_slice(format!("--{BOUNDARY}--\r\n").as_bytes());
    (format!("multipart/form-data; boundary={BOUNDARY}"), body)
}

/// A language model that answers from a script and records what it was asked.
///
/// Replies are consumed in order; when the script runs out the last reply (or
/// `"mock reply"` if there never was one) repeats.
#[derive(Default)]
pub struct MockLanguageModel {
    replies: Mutex<VecDeque<AiResult<String>>>,
    last: Mutex<Option<String>>,
    /// Every call's messages, in order.
    pub calls: Mutex<Vec<Vec<ChatMessage>>>,
    /// Every call's options, in order.
    pub options: Mutex<Vec<ChatOptions>>,
    /// While set, calls wait before answering (see [`hold`](Self::hold)).
    held: std::sync::atomic::AtomicBool,
    released: tokio::sync::Notify,
}

impl MockLanguageModel {
    pub fn new<I, S>(replies: I) -> Arc<Self>
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        let model = Self::default();
        model.replies.lock().unwrap().extend(replies.into_iter().map(|r| Ok(r.into())));
        Arc::new(model)
    }

    /// A model whose every call fails with `error`.
    pub fn failing(error: AiError) -> Arc<Self> {
        let model = Self::default();
        model.replies.lock().unwrap().push_back(Err(error));
        Arc::new(model)
    }

    pub fn push_reply(&self, reply: impl Into<String>) {
        self.replies.lock().unwrap().push_back(Ok(reply.into()));
    }

    /// Make calls wait until [`release`](Self::release), so a test can look
    /// at the state between "job queued" and "job finished".
    pub fn hold(&self) {
        self.held.store(true, std::sync::atomic::Ordering::SeqCst);
    }

    pub fn release(&self) {
        self.held.store(false, std::sync::atomic::Ordering::SeqCst);
        self.released.notify_waiters();
    }

    async fn wait_while_held(&self) {
        loop {
            let released = self.released.notified();
            if !self.held.load(std::sync::atomic::Ordering::SeqCst) {
                return;
            }
            released.await;
        }
    }

    pub fn call_count(&self) -> usize {
        self.calls.lock().unwrap().len()
    }

    /// All message text of call `index`, joined by newlines.
    pub fn prompt(&self, index: usize) -> String {
        self.calls.lock().unwrap()[index].iter().map(|m| m.content.as_str()).collect::<Vec<_>>().join("\n")
    }

    fn next(&self, messages: &[ChatMessage], options: &ChatOptions) -> AiResult<String> {
        self.calls.lock().unwrap().push(messages.to_vec());
        self.options.lock().unwrap().push(options.clone());
        let mut replies = self.replies.lock().unwrap();
        match replies.pop_front() {
            Some(Ok(reply)) => {
                *self.last.lock().unwrap() = Some(reply.clone());
                Ok(reply)
            }
            Some(Err(e)) => {
                // A failing model keeps failing.
                replies.push_front(Err(e.clone()));
                Err(e)
            }
            None => Ok(self.last.lock().unwrap().clone().unwrap_or_else(|| "mock reply".to_string())),
        }
    }
}

#[async_trait]
impl LanguageModel for MockLanguageModel {
    fn provider(&self) -> &str {
        "mock"
    }
    fn model_name(&self) -> &str {
        "mock-language"
    }
    async fn chat(&self, messages: &[ChatMessage], options: &ChatOptions) -> AiResult<ChatResponse> {
        self.wait_while_held().await;
        let content = self.next(messages, options)?;
        Ok(ChatResponse { content, model: "mock-language".into(), finish_reason: Some("stop".into()), usage: None })
    }
    async fn chat_stream(&self, messages: &[ChatMessage], options: &ChatOptions) -> AiResult<TextStream> {
        self.wait_while_held().await;
        let content = self.next(messages, options)?;
        // Split on spaces so consumers see several deltas.
        let deltas: Vec<AiResult<String>> = content.split_inclusive(' ').map(|part| Ok(part.to_string())).collect();
        Ok(futures::stream::iter(deltas).boxed())
    }
}

/// An embedding model producing small deterministic vectors: texts sharing
/// words get similar vectors, so semantic search behaves plausibly in tests.
pub struct MockEmbeddingModel {
    pub dimensions: usize,
    /// Every batch it was asked to embed.
    pub calls: Mutex<Vec<Vec<String>>>,
}

impl MockEmbeddingModel {
    pub fn new() -> Arc<Self> {
        Arc::new(Self { dimensions: 16, calls: Mutex::new(Vec::new()) })
    }

    pub fn vector(&self, text: &str) -> Vec<f32> {
        let mut vector = vec![0.0f32; self.dimensions];
        for word in text.to_lowercase().split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()) {
            let bucket = word.bytes().fold(7usize, |acc, b| acc.wrapping_mul(31).wrapping_add(b as usize)) % self.dimensions;
            vector[bucket] += 1.0;
        }
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            vector.iter_mut().for_each(|v| *v /= norm);
        } else {
            vector[0] = 1.0;
        }
        vector
    }
}

#[async_trait]
impl EmbeddingModel for MockEmbeddingModel {
    fn provider(&self) -> &str {
        "mock"
    }
    fn model_name(&self) -> &str {
        "mock-embedding"
    }
    async fn embed(&self, texts: &[String]) -> AiResult<Vec<Vec<f32>>> {
        self.calls.lock().unwrap().push(texts.to_vec());
        Ok(texts.iter().map(|t| self.vector(t)).collect())
    }
}

/// A text-to-speech model returning a tiny fake MP3 per call.
#[derive(Default)]
pub struct MockTextToSpeechModel {
    /// Every `(text, voice)` it was asked to speak.
    pub calls: Mutex<Vec<(String, String)>>,
}

impl MockTextToSpeechModel {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }
}

#[async_trait]
impl TextToSpeechModel for MockTextToSpeechModel {
    fn provider(&self) -> &str {
        "mock"
    }
    fn model_name(&self) -> &str {
        "mock-tts"
    }
    async fn synthesize(&self, text: &str, voice: &str) -> AiResult<AudioData> {
        self.calls.lock().unwrap().push((text.to_string(), voice.to_string()));
        Ok(AudioData { bytes: Bytes::from(silent_mp3_frames(4)), content_type: "audio/mpeg".into() })
    }
}

/// `count` valid, silent MPEG-1 Layer III frames (128 kbps, 44.1 kHz, mono).
pub fn silent_mp3_frames(count: usize) -> Vec<u8> {
    const FRAME_LEN: usize = 417; // 144 * 128000 / 44100
    let mut out = Vec::with_capacity(count * FRAME_LEN);
    for _ in 0..count {
        let mut frame = vec![0u8; FRAME_LEN];
        frame[..4].copy_from_slice(&[0xFF, 0xFB, 0x90, 0xC0]);
        out.extend_from_slice(&frame);
    }
    out
}

/// A speech-to-text model returning a fixed transcript.
pub struct MockSpeechToTextModel {
    pub transcript: String,
    /// File names it was asked to transcribe.
    pub calls: Mutex<Vec<String>>,
}

impl MockSpeechToTextModel {
    pub fn new(transcript: impl Into<String>) -> Arc<Self> {
        Arc::new(Self { transcript: transcript.into(), calls: Mutex::new(Vec::new()) })
    }
}

#[async_trait]
impl SpeechToTextModel for MockSpeechToTextModel {
    fn provider(&self) -> &str {
        "mock"
    }
    fn model_name(&self) -> &str {
        "mock-stt"
    }
    async fn transcribe(&self, request: TranscriptionRequest) -> AiResult<String> {
        self.calls.lock().unwrap().push(request.filename);
        Ok(self.transcript.clone())
    }
}
