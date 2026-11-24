//! Reassembly of streamed responses: Server-Sent Events and NDJSON.
//!
//! Network chunks split lines, events and even UTF-8 characters at arbitrary
//! points. [`LineBuffer`] reassembles lines from raw bytes, [`SseParser`]
//! builds events from lines following the WHATWG event-stream rules, and
//! [`text_stream`] turns a streamed HTTP response into a
//! [`TextStream`] using a per-provider decoder. Every provider's
//! `chat_stream` goes through this one module.

use std::time::Duration;

use bytes::Bytes;
use futures::{Stream, StreamExt};
use nn_llm::{AiError, AiResult, TextStream};

use crate::error::transport_error;

/// Longest line accepted in a streamed response (SSE or NDJSON).
pub const MAX_LINE_BYTES: usize = 8 * 1024 * 1024;
/// Largest SSE event (all its `data:` lines together).
pub const MAX_EVENT_BYTES: usize = 16 * 1024 * 1024;
/// Longest a streamed reply may run in total, however steadily data arrives.
pub const MAX_STREAM_DURATION: Duration = Duration::from_secs(60 * 60);

/// Splits a byte stream into lines. Accepts `\n`, `\r\n` and bare `\r`
/// terminators, and only decodes UTF-8 once a line is complete.
///
/// A line longer than [`MAX_LINE_BYTES`] is not buffered: the excess is
/// dropped and [`overflowed`](Self::overflowed) turns true for good, so the
/// caller can end the stream.
#[derive(Debug, Default)]
pub struct LineBuffer {
    buf: Vec<u8>,
    overflowed: bool,
    /// The previous byte was `\r`, so a directly following `\n` belongs to
    /// the same terminator.
    after_cr: bool,
    started: bool,
}

impl LineBuffer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Feeds a chunk and returns every line it completed, without terminators.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        let mut lines = Vec::new();
        for &byte in chunk {
            match byte {
                b'\n' if self.after_cr => self.after_cr = false,
                b'\n' | b'\r' => {
                    self.after_cr = byte == b'\r';
                    lines.push(self.take_line());
                }
                _ => {
                    self.after_cr = false;
                    if self.buf.len() < MAX_LINE_BYTES {
                        self.buf.push(byte);
                    } else {
                        self.overflowed = true;
                    }
                }
            }
        }
        lines
    }

    /// Whether a line ever exceeded [`MAX_LINE_BYTES`].
    pub fn overflowed(&self) -> bool {
        self.overflowed
    }

    /// Returns the unterminated remainder at end of stream, if any.
    pub fn finish(&mut self) -> Option<String> {
        (!self.buf.is_empty()).then(|| self.take_line())
    }

    fn take_line(&mut self) -> String {
        let mut bytes = std::mem::take(&mut self.buf);
        if !self.started {
            self.started = true;
            if bytes.starts_with(&[0xEF, 0xBB, 0xBF]) {
                bytes.drain(..3);
            }
        }
        String::from_utf8(bytes).unwrap_or_else(|e| String::from_utf8_lossy(e.as_bytes()).into_owned())
    }
}

/// One dispatched server-sent event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SseEvent {
    /// The `event:` field, when the server sent one.
    pub event: Option<String>,
    /// All `data:` lines of the event joined by `\n`.
    pub data: String,
}

/// Incremental Server-Sent Events parser.
///
/// Comment lines (starting with `:`) and `id`/`retry` fields are ignored,
/// multiple `data:` lines are joined with newlines, and an event is dispatched
/// at each blank line if it carried data.
#[derive(Debug, Default)]
pub struct SseParser {
    lines: LineBuffer,
    event: Option<String>,
    data: Vec<String>,
    data_bytes: usize,
    overflowed: bool,
}

impl SseParser {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, chunk: &[u8]) -> Vec<SseEvent> {
        let mut events = Vec::new();
        for line in self.lines.push(chunk) {
            if let Some(event) = self.line(&line) {
                events.push(event);
            }
        }
        events
    }

    /// Whether a line exceeded [`MAX_LINE_BYTES`] or an event
    /// [`MAX_EVENT_BYTES`]. Oversized events are dropped, not dispatched.
    pub fn overflowed(&self) -> bool {
        self.overflowed || self.lines.overflowed()
    }

    /// Flushes an event the server did not terminate with a blank line.
    pub fn finish(&mut self) -> Option<SseEvent> {
        if let Some(line) = self.lines.finish() {
            self.line(&line);
        }
        self.dispatch()
    }

    fn line(&mut self, line: &str) -> Option<SseEvent> {
        if line.is_empty() {
            return self.dispatch();
        }
        if line.starts_with(':') {
            return None;
        }
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line, ""),
        };
        match field {
            "data" => {
                self.data_bytes += value.len() + 1;
                if self.data_bytes > MAX_EVENT_BYTES {
                    self.overflowed = true;
                } else {
                    self.data.push(value.to_string());
                }
            }
            "event" => self.event = Some(value.to_string()),
            _ => {}
        }
        None
    }

    fn dispatch(&mut self) -> Option<SseEvent> {
        let event = self.event.take();
        self.data_bytes = 0;
        if self.data.is_empty() || self.overflowed {
            self.data.clear();
            return None;
        }
        let data = std::mem::take(&mut self.data).join("\n");
        Some(SseEvent { event, data })
    }
}

/// How a streamed body is cut into frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Framing {
    /// Server-Sent Events; a `[DONE]` data payload ends the stream.
    Sse,
    /// One JSON document per line (Ollama).
    NdJson,
}

/// One unit handed to a provider decoder: an SSE event or an NDJSON line.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Frame {
    pub event: Option<String>,
    pub data: String,
}

/// What a provider decoder makes of one frame.
#[derive(Debug)]
pub(crate) enum Step {
    /// A text delta to yield (empty deltas are dropped).
    Text(String),
    /// Nothing for the caller (keep-alives, metadata, usage).
    Skip,
    /// The provider signalled the end of the reply.
    Done,
    /// The provider reported an error mid-stream.
    Fail(AiError),
}

enum Framer {
    Sse(SseParser),
    NdJson(LineBuffer),
}

impl Framer {
    fn new(framing: Framing) -> Self {
        match framing {
            Framing::Sse => Framer::Sse(SseParser::new()),
            Framing::NdJson => Framer::NdJson(LineBuffer::new()),
        }
    }

    fn push(&mut self, chunk: &[u8]) -> Vec<Frame> {
        match self {
            Framer::Sse(parser) => parser.push(chunk).into_iter().map(Frame::from).collect(),
            Framer::NdJson(lines) => lines.push(chunk).into_iter().filter_map(Frame::from_line).collect(),
        }
    }

    fn overflowed(&self) -> bool {
        match self {
            Framer::Sse(parser) => parser.overflowed(),
            Framer::NdJson(lines) => lines.overflowed(),
        }
    }

    fn finish(&mut self) -> Vec<Frame> {
        match self {
            Framer::Sse(parser) => parser.finish().into_iter().map(Frame::from).collect(),
            Framer::NdJson(lines) => lines.finish().into_iter().filter_map(Frame::from_line).collect(),
        }
    }
}

impl From<SseEvent> for Frame {
    fn from(event: SseEvent) -> Self {
        Frame { event: event.event, data: event.data }
    }
}

impl Frame {
    fn from_line(line: String) -> Option<Frame> {
        (!line.trim().is_empty()).then_some(Frame { event: None, data: line })
    }
}

/// Decodes a stream of body chunks into text deltas.
///
/// `idle` bounds the wait for each chunk, so long answers are not cut off
/// while a stalled connection still fails; [`MAX_STREAM_DURATION`] bounds the
/// whole reply, and [`MAX_LINE_BYTES`] / [`MAX_EVENT_BYTES`] its frames. The
/// stream ends after the first error.
pub(crate) fn decode_stream<S, D>(body: S, framing: Framing, idle: Duration, provider: String, mut decode: D) -> TextStream
where
    S: Stream<Item = AiResult<Bytes>> + Send + 'static,
    D: FnMut(Frame) -> Step + Send + 'static,
{
    let stream = async_stream::stream! {
        let mut body = Box::pin(body);
        let mut framer = Framer::new(framing);
        let deadline = tokio::time::Instant::now() + MAX_STREAM_DURATION;
        'read: loop {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                yield Err(AiError::Timeout(format!(
                    "{provider}: the streamed reply took longer than {} minutes",
                    MAX_STREAM_DURATION.as_secs() / 60
                )));
                break;
            }
            let (frames, at_end) = match tokio::time::timeout(idle.min(remaining), body.next()).await {
                Err(_) if remaining <= idle => {
                    yield Err(AiError::Timeout(format!(
                        "{provider}: the streamed reply took longer than {} minutes",
                        MAX_STREAM_DURATION.as_secs() / 60
                    )));
                    break;
                }
                Err(_) => {
                    yield Err(AiError::Timeout(format!(
                        "{provider}: no data received for {} seconds while streaming",
                        idle.as_secs()
                    )));
                    break;
                }
                Ok(Some(Err(err))) => {
                    yield Err(err);
                    break;
                }
                Ok(Some(Ok(chunk))) => {
                    let frames = framer.push(&chunk);
                    if framer.overflowed() {
                        yield Err(AiError::Provider {
                            status: None,
                            message: format!("{provider}: unexpected response: a streamed event is too large"),
                        });
                        break;
                    }
                    (frames, false)
                }
                Ok(None) => (framer.finish(), true),
            };
            for frame in frames {
                if framing == Framing::Sse && frame.data.trim() == "[DONE]" {
                    break 'read;
                }
                match decode(frame) {
                    Step::Text(text) => {
                        if !text.is_empty() {
                            yield Ok(text);
                        }
                    }
                    Step::Skip => {}
                    Step::Done => break 'read,
                    Step::Fail(err) => {
                        yield Err(err);
                        break 'read;
                    }
                }
            }
            if at_end {
                break;
            }
        }
    };
    Box::pin(stream)
}

/// [`decode_stream`] over the body of an HTTP response.
pub(crate) fn text_stream<D>(response: reqwest::Response, framing: Framing, idle: Duration, provider: &str, decode: D) -> TextStream
where
    D: FnMut(Frame) -> Step + Send + 'static,
{
    let name = provider.to_string();
    let body = response.bytes_stream().map(move |chunk| chunk.map_err(|e| transport_error(&name, e)));
    decode_stream(body, framing, idle, provider.to_string(), decode)
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::stream;

    fn all_lines(chunks: &[&[u8]]) -> Vec<String> {
        let mut buffer = LineBuffer::new();
        let mut lines: Vec<String> = chunks.iter().flat_map(|c| buffer.push(c)).collect();
        lines.extend(buffer.finish());
        lines
    }

    #[test]
    fn lines_survive_any_chunking() {
        let text = "alpha\nbeta\r\ngamma\rdelta\n\nlast";
        let expected = vec!["alpha", "beta", "gamma", "delta", "", "last"];
        assert_eq!(all_lines(&[text.as_bytes()]), expected);
        // Every possible split point, including between \r and \n.
        for cut in 0..text.len() {
            let (a, b) = text.as_bytes().split_at(cut);
            assert_eq!(all_lines(&[a, b]), expected, "cut at {cut}");
        }
        let bytes: Vec<&[u8]> = text.as_bytes().chunks(1).collect();
        assert_eq!(all_lines(&bytes), expected);
    }

    #[test]
    fn multibyte_characters_split_across_chunks_are_reassembled() {
        let text = "héllo wörld ✓ 日本語\n";
        for cut in 0..text.len() {
            let (a, b) = text.as_bytes().split_at(cut);
            assert_eq!(all_lines(&[a, b]), vec!["héllo wörld ✓ 日本語"], "cut at {cut}");
        }
    }

    #[test]
    fn leading_byte_order_mark_is_dropped() {
        assert_eq!(all_lines(&[b"\xEF\xBB\xBFdata: x\n"]), vec!["data: x"]);
    }

    fn all_events(chunks: &[&[u8]]) -> Vec<SseEvent> {
        let mut parser = SseParser::new();
        let mut events: Vec<SseEvent> = chunks.iter().flat_map(|c| parser.push(c)).collect();
        events.extend(parser.finish());
        events
    }

    fn event(name: Option<&str>, data: &str) -> SseEvent {
        SseEvent { event: name.map(str::to_string), data: data.to_string() }
    }

    #[test]
    fn sse_events_are_parsed_per_the_event_stream_rules() {
        let raw = concat!(
            ": keep-alive comment\n",
            "\n",
            "event: message_start\n",
            "data: {\"a\":1}\n",
            "\n",
            "data: first line\n",
            "data: second line\n",
            "id: 7\n",
            "retry: 1000\n",
            "\n",
            "data:no-space\n",
            "\n",
            "event: ping\n",
            "\n",
            "data\n",
            "\n",
            "data: [DONE]\n",
            "\n",
        );
        let expected = vec![
            event(Some("message_start"), "{\"a\":1}"),
            event(None, "first line\nsecond line"),
            event(None, "no-space"),
            // "event: ping" without data dispatches nothing and must not leak
            // its name into the next event.
            event(None, ""),
            event(None, "[DONE]"),
        ];
        assert_eq!(all_events(&[raw.as_bytes()]), expected);
        for cut in 0..raw.len() {
            let (a, b) = raw.as_bytes().split_at(cut);
            assert_eq!(all_events(&[a, b]), expected, "cut at {cut}");
        }
        let crlf = raw.replace('\n', "\r\n");
        let bytes: Vec<&[u8]> = crlf.as_bytes().chunks(3).collect();
        assert_eq!(all_events(&bytes), expected);
    }

    #[test]
    fn unterminated_final_event_is_flushed() {
        assert_eq!(all_events(&[b"data: tail"]), vec![event(None, "tail")]);
        assert_eq!(all_events(&[b"data: a\n\ndata: b\n"]), vec![event(None, "a"), event(None, "b")]);
    }

    fn chunks(parts: &[&str]) -> impl Stream<Item = AiResult<Bytes>> + Send + 'static {
        let owned: Vec<AiResult<Bytes>> = parts.iter().map(|p| Ok(Bytes::from(p.to_string()))).collect();
        stream::iter(owned)
    }

    fn echo(frame: Frame) -> Step {
        match frame.data.as_str() {
            "skip" => Step::Skip,
            "stop" => Step::Done,
            "boom" => Step::Fail(AiError::Other("boom".into())),
            other => Step::Text(other.to_string()),
        }
    }

    async fn collect(stream: TextStream) -> Vec<Result<String, String>> {
        stream.map(|item| item.map_err(|e| e.to_string())).collect().await
    }

    const IDLE: Duration = Duration::from_secs(5);

    #[tokio::test]
    async fn sse_stream_yields_deltas_and_stops_at_done() {
        let body = chunks(&["data: Hel", "lo\n\ndata: skip\n\n: ping\n\nda", "ta: world\n\ndata: [DONE]\n\ndata: ignored\n\n"]);
        let out = collect(decode_stream(body, Framing::Sse, IDLE, "test".into(), echo)).await;
        assert_eq!(out, vec![Ok("Hello".to_string()), Ok("world".to_string())]);
    }

    #[tokio::test]
    async fn decoder_can_end_or_fail_the_stream() {
        let body = chunks(&["data: a\n\ndata: stop\n\ndata: b\n\n"]);
        let out = collect(decode_stream(body, Framing::Sse, IDLE, "test".into(), echo)).await;
        assert_eq!(out, vec![Ok("a".to_string())]);

        let body = chunks(&["data: a\n\ndata: boom\n\ndata: b\n\n"]);
        let out = collect(decode_stream(body, Framing::Sse, IDLE, "test".into(), echo)).await;
        assert_eq!(out, vec![Ok("a".to_string()), Err("boom".to_string())]);
    }

    #[tokio::test]
    async fn ndjson_stream_handles_split_lines_and_missing_final_newline() {
        let body = chunks(&["on", "e\n\ntwo\nthr", "ee"]);
        let out = collect(decode_stream(body, Framing::NdJson, IDLE, "test".into(), echo)).await;
        assert_eq!(out, vec![Ok("one".to_string()), Ok("two".to_string()), Ok("three".to_string())]);
    }

    #[tokio::test]
    async fn transport_errors_end_the_stream() {
        let items: Vec<AiResult<Bytes>> = vec![
            Ok(Bytes::from("data: a\n\n")),
            Err(AiError::Network("reset".into())),
            Ok(Bytes::from("data: b\n\n")),
        ];
        let out = collect(decode_stream(stream::iter(items), Framing::Sse, IDLE, "test".into(), echo)).await;
        assert_eq!(out, vec![Ok("a".to_string()), Err("reset".to_string())]);
    }

    #[tokio::test(start_paused = true)]
    async fn a_stalled_stream_times_out() {
        let body = stream::iter(vec![Ok(Bytes::from("data: a\n\n"))]).chain(stream::pending());
        let out = collect(decode_stream(body, Framing::Sse, Duration::from_secs(30), "test".into(), echo)).await;
        assert_eq!(out.len(), 2);
        assert_eq!(out[0], Ok("a".to_string()));
        assert!(out[1].as_ref().unwrap_err().contains("no data received for 30 seconds"));
    }

    #[test]
    fn an_endless_line_is_not_buffered() {
        let mut lines = LineBuffer::new();
        let chunk = vec![b'a'; 1024 * 1024];
        for _ in 0..(MAX_LINE_BYTES / chunk.len() + 4) {
            assert!(lines.push(&chunk).is_empty());
        }
        assert!(lines.overflowed());
        assert!(lines.buf.len() <= MAX_LINE_BYTES);
    }

    #[test]
    fn an_oversized_event_is_dropped_and_flagged() {
        let mut parser = SseParser::new();
        let line = format!("data: {}\n", "x".repeat(1024 * 1024));
        for _ in 0..(MAX_EVENT_BYTES / (1024 * 1024) + 2) {
            assert!(parser.push(line.as_bytes()).is_empty());
        }
        assert!(parser.overflowed());
        assert!(parser.data.iter().map(String::len).sum::<usize>() <= MAX_EVENT_BYTES);
        assert!(parser.push(b"\n").is_empty(), "the oversized event is not dispatched");
    }

    #[tokio::test]
    async fn an_oversized_stream_line_ends_the_stream() {
        let big = "a".repeat(MAX_LINE_BYTES + 10);
        let body = stream::iter(vec![Ok(Bytes::from("data: ok\n\n")), Ok(Bytes::from(big))]);
        let out = collect(decode_stream(body, Framing::NdJson, IDLE, "test".into(), echo)).await;
        assert_eq!(out.len(), 2, "{:?}", out.iter().map(|r| r.as_ref().map(String::len)).collect::<Vec<_>>());
        assert!(out[1].as_ref().unwrap_err().contains("too large"), "{:?}", out[1]);
    }

    #[tokio::test(start_paused = true)]
    async fn a_trickled_stream_hits_the_total_deadline() {
        // A keep-alive comment every 10 s never trips the idle timeout.
        let body = stream::unfold((), |()| async {
            tokio::time::sleep(Duration::from_secs(10)).await;
            Some((Ok(Bytes::from(": keep-alive\n")), ()))
        });
        let out = collect(decode_stream(body, Framing::Sse, Duration::from_secs(30), "test".into(), echo)).await;
        assert_eq!(out.len(), 1);
        assert!(out[0].as_ref().unwrap_err().contains("took longer than 60 minutes"), "{:?}", out[0]);
    }
}
