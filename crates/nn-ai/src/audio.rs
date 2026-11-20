//! Small audio helpers: WAV framing for raw PCM, MIME guessing and the clips
//! used by connection tests.

use bytes::Bytes;

/// A short clip of speech ("Hello there") used to validate speech-to-text
/// models, so a passing test visibly returns text.
pub const TEST_SPEECH_MP3: &[u8] = include_bytes!("../assets/test_speech.mp3");

/// Wraps signed 16-bit little-endian PCM in a RIFF/WAVE header so the result
/// plays as-is.
pub fn pcm16_to_wav(pcm: &[u8], sample_rate: u32, channels: u16) -> Bytes {
    const BITS_PER_SAMPLE: u16 = 16;
    let block_align = channels * BITS_PER_SAMPLE / 8;
    let byte_rate = sample_rate * u32::from(block_align);
    let data_len = pcm.len() as u32;

    let mut wav = Vec::with_capacity(44 + pcm.len());
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&(36 + data_len).to_le_bytes());
    wav.extend_from_slice(b"WAVE");
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&BITS_PER_SAMPLE.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_len.to_le_bytes());
    wav.extend_from_slice(pcm);
    Bytes::from(wav)
}

/// Half a second of silence (16 kHz, 16-bit mono WAV).
pub fn silent_wav() -> Bytes {
    const SAMPLE_RATE: u32 = 16_000;
    pcm16_to_wav(&vec![0u8; SAMPLE_RATE as usize], SAMPLE_RATE, 1)
}

/// True for the MIME types Gemini uses for raw PCM
/// (`audio/L16;codec=pcm;rate=24000`, `audio/pcm`).
pub(crate) fn is_raw_pcm(mime: &str) -> bool {
    let mime = mime.to_ascii_lowercase();
    mime.contains("l16") || mime.contains("pcm")
}

/// Reads a MIME parameter such as `rate` from `audio/L16;codec=pcm;rate=24000`.
pub(crate) fn mime_param(mime: &str, name: &str) -> Option<u32> {
    mime.split(';').skip(1).find_map(|part| {
        let (key, value) = part.split_once('=')?;
        key.trim().eq_ignore_ascii_case(name).then(|| value.trim().parse().ok())?
    })
}

/// MIME type for an audio file name, by extension.
pub(crate) fn audio_mime(filename: &str) -> &'static str {
    let ext = filename.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase()).unwrap_or_default();
    match ext.as_str() {
        "mp3" | "mpga" | "mpeg" => "audio/mpeg",
        "wav" | "wave" => "audio/wav",
        "m4a" | "mp4" => "audio/mp4",
        "ogg" | "oga" | "opus" => "audio/ogg",
        "flac" => "audio/flac",
        "webm" => "audio/webm",
        "aac" => "audio/aac",
        "aiff" | "aif" => "audio/aiff",
        _ => "application/octet-stream",
    }
}

/// Container/extension name for an audio file (`mp3`, `wav`...), as some JSON
/// speech APIs want it.
pub(crate) fn audio_format(filename: &str) -> String {
    filename
        .rsplit_once('.')
        .map(|(_, e)| e.to_ascii_lowercase())
        .filter(|e| !e.is_empty() && e.len() <= 5)
        .unwrap_or_else(|| "mp3".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_header_describes_the_pcm_payload() {
        let pcm = vec![1u8; 48_000];
        let wav = pcm16_to_wav(&pcm, 24_000, 1);
        assert_eq!(wav.len(), 44 + 48_000);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()), 36 + 48_000);
        assert_eq!(&wav[8..16], b"WAVEfmt ");
        assert_eq!(u32::from_le_bytes(wav[16..20].try_into().unwrap()), 16);
        assert_eq!(u16::from_le_bytes(wav[20..22].try_into().unwrap()), 1);
        assert_eq!(u16::from_le_bytes(wav[22..24].try_into().unwrap()), 1);
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 24_000);
        assert_eq!(u32::from_le_bytes(wav[28..32].try_into().unwrap()), 48_000);
        assert_eq!(u16::from_le_bytes(wav[32..34].try_into().unwrap()), 2);
        assert_eq!(u16::from_le_bytes(wav[34..36].try_into().unwrap()), 16);
        assert_eq!(&wav[36..40], b"data");
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 48_000);
        assert_eq!(&wav[44..], &pcm[..]);
    }

    #[test]
    fn silent_wav_is_half_a_second() {
        let wav = silent_wav();
        assert_eq!(wav.len(), 44 + 16_000);
        assert!(wav[44..].iter().all(|b| *b == 0));
    }

    #[test]
    fn pcm_mime_types_are_recognised_and_parsed() {
        assert!(is_raw_pcm("audio/L16;codec=pcm;rate=24000"));
        assert!(is_raw_pcm("audio/pcm"));
        assert!(!is_raw_pcm("audio/wav"));
        assert_eq!(mime_param("audio/L16;codec=pcm;rate=24000", "rate"), Some(24_000));
        assert_eq!(mime_param("audio/L16; Rate = 16000 ; channels=2", "rate"), Some(16_000));
        assert_eq!(mime_param("audio/L16; rate=16000; channels=2", "channels"), Some(2));
        assert_eq!(mime_param("audio/L16", "rate"), None);
    }

    #[test]
    fn mime_and_format_follow_the_extension() {
        assert_eq!(audio_mime("talk.MP3"), "audio/mpeg");
        assert_eq!(audio_mime("a.b.wav"), "audio/wav");
        assert_eq!(audio_mime("noext"), "application/octet-stream");
        assert_eq!(audio_format("clip.M4A"), "m4a");
        assert_eq!(audio_format("noext"), "mp3");
    }

    #[test]
    fn bundled_speech_clip_is_present() {
        assert!(TEST_SPEECH_MP3.len() > 1000);
    }
}
