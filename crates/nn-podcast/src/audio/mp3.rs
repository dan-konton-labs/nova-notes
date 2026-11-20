//! MPEG audio (MP3) without a decoder: frame headers, duration, and joining
//! clips frame by frame.
//!
//! A text-to-speech clip is a run of MPEG frames, usually preceded by an ID3v2
//! tag and a Xing/Info header frame that states the clip's own length. Gluing
//! such files together byte for byte leaves the first clip's header claiming
//! the length of the whole file, so players show the wrong duration. Joining
//! here keeps only the audio frames and writes one fresh header for the result.

use std::ops::Range;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MpegVersion {
    V1,
    V2,
    V25,
}

/// The four header bytes of one MPEG audio frame, decoded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct FrameHeader {
    pub version: MpegVersion,
    /// 1, 2 or 3.
    pub layer: u8,
    pub has_crc: bool,
    pub bitrate_kbps: u32,
    pub sample_rate: u32,
    pub padding: bool,
    pub channels: u8,
    raw: [u8; 4],
}

const BITRATES_V1_L1: [u32; 15] = [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448];
const BITRATES_V1_L2: [u32; 15] = [0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384];
const BITRATES_V1_L3: [u32; 15] = [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320];
const BITRATES_V2_L1: [u32; 15] = [0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256];
const BITRATES_V2_L23: [u32; 15] = [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160];

fn bitrate_table(version: MpegVersion, layer: u8) -> &'static [u32; 15] {
    match (version, layer) {
        (MpegVersion::V1, 1) => &BITRATES_V1_L1,
        (MpegVersion::V1, 2) => &BITRATES_V1_L2,
        (MpegVersion::V1, _) => &BITRATES_V1_L3,
        (_, 1) => &BITRATES_V2_L1,
        _ => &BITRATES_V2_L23,
    }
}

impl FrameHeader {
    /// Decode a frame header at the start of `bytes`. Free-format frames
    /// (bitrate index 0) are not supported and count as invalid.
    pub fn parse(bytes: &[u8]) -> Option<Self> {
        let raw: [u8; 4] = bytes.get(..4)?.try_into().ok()?;
        if raw[0] != 0xFF || raw[1] & 0xE0 != 0xE0 {
            return None;
        }
        let version = match (raw[1] >> 3) & 0b11 {
            0b00 => MpegVersion::V25,
            0b10 => MpegVersion::V2,
            0b11 => MpegVersion::V1,
            _ => return None,
        };
        let layer = match (raw[1] >> 1) & 0b11 {
            0b01 => 3,
            0b10 => 2,
            0b11 => 1,
            _ => return None,
        };
        let bitrate_index = usize::from(raw[2] >> 4);
        if bitrate_index == 0 || bitrate_index == 15 {
            return None;
        }
        let sample_rate = match (version, (raw[2] >> 2) & 0b11) {
            (_, 0b11) => return None,
            (MpegVersion::V1, index) => [44_100, 48_000, 32_000][usize::from(index)],
            (MpegVersion::V2, index) => [22_050, 24_000, 16_000][usize::from(index)],
            (MpegVersion::V25, index) => [11_025, 12_000, 8_000][usize::from(index)],
        };
        Some(Self {
            version,
            layer,
            has_crc: raw[1] & 0x01 == 0,
            bitrate_kbps: bitrate_table(version, layer)[bitrate_index],
            sample_rate,
            padding: raw[2] & 0x02 != 0,
            channels: if raw[3] >> 6 == 0b11 { 1 } else { 2 },
            raw,
        })
    }

    /// Size of the whole frame in bytes, header included.
    pub fn frame_len(&self) -> usize {
        let bits = self.bitrate_kbps as usize * 1000;
        let rate = self.sample_rate as usize;
        let padding = usize::from(self.padding);
        match (self.layer, self.version) {
            (1, _) => (12 * bits / rate + padding) * 4,
            (2, _) | (3, MpegVersion::V1) => 144 * bits / rate + padding,
            _ => 72 * bits / rate + padding,
        }
    }

    pub fn samples_per_frame(&self) -> u32 {
        match (self.layer, self.version) {
            (1, _) => 384,
            (2, _) | (3, MpegVersion::V1) => 1152,
            _ => 576,
        }
    }

    /// Layer III side information size, which is where a Xing header starts.
    fn side_info_len(&self) -> usize {
        match (self.version, self.channels) {
            (MpegVersion::V1, 1) => 17,
            (MpegVersion::V1, _) => 32,
            (_, 1) => 9,
            _ => 17,
        }
    }

    /// Frames of one stream agree on these; a header that differs is not the
    /// next frame but a coincidence in other data.
    fn same_stream(&self, other: &FrameHeader) -> bool {
        self.version == other.version && self.layer == other.layer && self.sample_rate == other.sample_rate
    }

    fn describe(&self) -> String {
        let version = match self.version {
            MpegVersion::V1 => "MPEG-1",
            MpegVersion::V2 => "MPEG-2",
            MpegVersion::V25 => "MPEG-2.5",
        };
        let channels = if self.channels == 1 { "mono" } else { "stereo" };
        format!("{version} layer {}, {} Hz, {channels}", self.layer, self.sample_rate)
    }
}

/// Totals stated by a Xing/Info or VBRI header frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct VbrHeader {
    frames: Option<u32>,
}

fn be32(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
}

/// Read the Xing/Info or VBRI header carried by `frame`, if it is one.
fn vbr_header(header: &FrameHeader, frame: &[u8]) -> Option<VbrHeader> {
    if header.layer != 3 {
        return None;
    }
    let xing_at = 4 + header.side_info_len();
    for at in [xing_at, xing_at + 2] {
        let tag = frame.get(at..at + 4);
        if tag == Some(b"Xing") || tag == Some(b"Info") {
            let flags = be32(frame, at + 4)?;
            let frames = if flags & 0x1 != 0 { be32(frame, at + 8) } else { None };
            return Some(VbrHeader { frames });
        }
    }
    if frame.get(36..40) == Some(b"VBRI") {
        // version(2) delay(2) quality(2) bytes(4) frames(4)
        return Some(VbrHeader { frames: be32(frame, 36 + 14) });
    }
    None
}

/// Size of an ID3v2 tag starting at `at`, or `None` when there is none.
fn id3v2_len(data: &[u8], at: usize) -> Option<usize> {
    let head = data.get(at..at + 10)?;
    if &head[..3] != b"ID3" || head[6..10].iter().any(|b| b & 0x80 != 0) {
        return None;
    }
    let size = head[6..10].iter().fold(0usize, |size, &b| (size << 7) | usize::from(b));
    let footer = if head[5] & 0x10 != 0 { 10 } else { 0 };
    Some(10 + size + footer)
}

/// True when a metadata tag that ends the audio starts at `at`.
fn is_trailing_tag(data: &[u8], at: usize) -> bool {
    let rest = &data[at..];
    rest.starts_with(b"TAG") || rest.starts_with(b"APETAGEX") || rest.starts_with(b"LYRICSBEGIN")
}

/// A frame whose header parses and which is followed by another frame of the
/// same stream, by a trailing tag, or by the end of the data.
fn confirmed_frame(data: &[u8], at: usize, stream: Option<&FrameHeader>) -> Option<FrameHeader> {
    let header = FrameHeader::parse(data.get(at..)?)?;
    if stream.is_some_and(|stream| !stream.same_stream(&header)) {
        return None;
    }
    let end = at + header.frame_len();
    if end > data.len() {
        return None;
    }
    if data.len() - end < 4 || is_trailing_tag(data, end) || id3v2_len(data, end).is_some() {
        return Some(header);
    }
    FrameHeader::parse(&data[end..]).filter(|next| next.same_stream(&header)).map(|_| header)
}

/// The frames of one MP3 file.
#[derive(Debug)]
pub(crate) struct Mp3Stream {
    /// Header of the first audio frame.
    pub first: FrameHeader,
    /// Byte ranges of the audio frames, in order. The Xing/Info/VBRI header
    /// frame, tags and anything that is not a frame are left out.
    pub frames: Vec<Range<usize>>,
    /// Frame count stated by the file's own Xing/Info/VBRI header.
    pub stated_frames: Option<u32>,
    /// Samples in `frames`.
    pub samples: u64,
    /// True when the frames do not all have the same bitrate.
    pub variable_bitrate: bool,
}

impl Mp3Stream {
    /// Length in seconds: from the file's own header when it has one (as
    /// players read it), else from counting every frame.
    pub fn duration_seconds(&self) -> f64 {
        let rate = f64::from(self.first.sample_rate);
        match self.stated_frames {
            Some(frames) if frames > 0 => {
                f64::from(frames) * f64::from(self.first.samples_per_frame()) / rate
            }
            _ => self.samples as f64 / rate,
        }
    }
}

/// Walk every frame of an MP3 file held in memory.
pub(crate) fn scan(data: &[u8]) -> Result<Mp3Stream, String> {
    let mut position = 0;
    while let Some(len) = id3v2_len(data, position) {
        position += len;
    }
    let not_mp3 = || "no MPEG audio frames found".to_string();
    if position >= data.len() {
        return Err(not_mp3());
    }

    let mut position = (position..data.len().saturating_sub(3))
        .find(|&at| data[at] == 0xFF && confirmed_frame(data, at, None).is_some())
        .ok_or_else(not_mp3)?;
    let mut first = confirmed_frame(data, position, None).ok_or_else(not_mp3)?;

    let mut stated_frames = None;
    let first_frame = &data[position..position + first.frame_len()];
    if let Some(vbr) = vbr_header(&first, first_frame) {
        stated_frames = vbr.frames;
        position += first.frame_len();
        // The header frame may use another bitrate or channel mode than the
        // audio; the audio's own first frame describes the stream.
        match FrameHeader::parse(data.get(position..).unwrap_or_default()) {
            Some(next) if next.same_stream(&first) && position + next.frame_len() <= data.len() => {
                first = next;
            }
            _ => {
                return Ok(Mp3Stream {
                    first,
                    frames: Vec::new(),
                    stated_frames,
                    samples: 0,
                    variable_bitrate: false,
                })
            }
        }
    }

    let mut frames = Vec::new();
    let mut variable_bitrate = false;
    while position + 4 <= data.len() {
        let header = match FrameHeader::parse(&data[position..]).filter(|h| h.same_stream(&first)) {
            Some(header) => header,
            None if is_trailing_tag(data, position) => break,
            None => {
                // Junk between frames: resynchronise on the next real frame.
                match (position + 1..data.len().saturating_sub(3))
                    .find(|&at| data[at] == 0xFF && confirmed_frame(data, at, Some(&first)).is_some())
                {
                    Some(next) => {
                        position = next;
                        continue;
                    }
                    None => break,
                }
            }
        };
        let end = position + header.frame_len();
        if end > data.len() {
            break; // cut-off last frame
        }
        variable_bitrate |= header.bitrate_kbps != first.bitrate_kbps;
        frames.push(position..end);
        position = end;
    }

    let samples = frames.len() as u64 * u64::from(first.samples_per_frame());
    Ok(Mp3Stream { first, frames, stated_frames, samples, variable_bitrate })
}

/// Length of an MP3 file in seconds.
pub(crate) fn duration_seconds(data: &[u8]) -> Result<f64, String> {
    let stream = scan(data)?;
    let seconds = stream.duration_seconds();
    if seconds > 0.0 {
        Ok(seconds)
    } else {
        Err("the MP3 file has no audio frames".into())
    }
}

/// A frame that decodes to silence in the stream's format: the first audio
/// frame's header (no CRC, no padding) over an all-zero body.
fn silent_frame(format: &FrameHeader) -> Vec<u8> {
    let mut raw = format.raw;
    raw[1] |= 0x01; // protection bit set: no CRC follows the header
    raw[2] &= !0x02; // no padding
    let header = FrameHeader::parse(&raw).expect("a header derived from a valid one is valid");
    let mut frame = vec![0u8; header.frame_len()];
    frame[..4].copy_from_slice(&raw);
    frame
}

/// Build the Xing/Info header frame for a joined stream.
///
/// `offsets` are the byte offsets of every audio frame relative to the first,
/// and `audio_bytes` their total size. Returns `None` for layers that have no
/// such header, or when no bitrate gives a frame large enough to hold it.
fn xing_frame(format: &FrameHeader, offsets: &[usize], audio_bytes: usize, variable: bool) -> Option<Vec<u8>> {
    if format.layer != 3 {
        return None;
    }
    let tag_at = 4 + format.side_info_len();
    let needed = tag_at + 4 + 4 + 4 + 4 + 100;

    let mut raw = format.raw;
    raw[1] |= 0x01;
    raw[2] &= !0x02;
    let header = (1u8..15).find_map(|index| {
        raw[2] = (raw[2] & 0x0F) | (index << 4);
        FrameHeader::parse(&raw).filter(|h| h.frame_len() >= needed)
    })?;

    let mut frame = vec![0u8; header.frame_len()];
    frame[..4].copy_from_slice(&header.raw);
    let header_len = frame.len();
    let total_bytes = header_len + audio_bytes;
    let body = &mut frame[tag_at..];
    body[..4].copy_from_slice(if variable { b"Xing" } else { b"Info" });
    body[4..8].copy_from_slice(&0x7u32.to_be_bytes()); // frames, bytes, TOC
    body[8..12].copy_from_slice(&u32::try_from(offsets.len()).ok()?.to_be_bytes());
    body[12..16].copy_from_slice(&u32::try_from(total_bytes).ok()?.to_be_bytes());
    for (percent, entry) in body[16..116].iter_mut().enumerate() {
        // Seek table: where in the file (as a 256th) each percent of the
        // running time starts.
        let frame_index = (percent * offsets.len() / 100).min(offsets.len().saturating_sub(1));
        let byte = header_len + offsets.get(frame_index).copied().unwrap_or(0);
        *entry = (byte * 256 / total_bytes.max(1)).min(255) as u8;
    }
    Some(frame)
}

/// Join MP3 clips of one format into a single stream, with `gap_ms` of silence
/// between consecutive clips.
///
/// Tags and each clip's own Xing/Info/VBRI header frame are dropped, and one
/// new header frame describing the whole result is written first.
pub(crate) fn concat(clips: &[Vec<u8>], gap_ms: u32) -> Result<Vec<u8>, String> {
    let mut streams = Vec::with_capacity(clips.len());
    for (index, clip) in clips.iter().enumerate() {
        let stream = scan(clip).map_err(|e| format!("clip {index:04} is not a usable MP3 file: {e}"))?;
        if stream.frames.is_empty() {
            return Err(format!("clip {index:04} has no audio frames"));
        }
        streams.push(stream);
    }
    let format = streams.first().ok_or("there are no clips to combine")?.first;
    for (index, stream) in streams.iter().enumerate() {
        if !stream.first.same_stream(&format) || stream.first.channels != format.channels {
            return Err(format!(
                "clips have different MP3 formats (clip 0000 is {}, clip {index:04} is {})",
                format.describe(),
                stream.first.describe()
            ));
        }
    }

    let silence = silent_frame(&format);
    let frame_ms = f64::from(format.samples_per_frame()) * 1000.0 / f64::from(format.sample_rate);
    let gap_frames = (f64::from(gap_ms) / frame_ms).round() as usize;

    let mut audio = Vec::with_capacity(clips.iter().map(Vec::len).sum::<usize>());
    let mut offsets = Vec::new();
    let mut variable = streams.iter().any(|s| s.variable_bitrate || s.first.bitrate_kbps != format.bitrate_kbps);
    for (index, (clip, stream)) in clips.iter().zip(&streams).enumerate() {
        if index > 0 {
            for _ in 0..gap_frames {
                offsets.push(audio.len());
                audio.extend_from_slice(&silence);
            }
        }
        for range in &stream.frames {
            offsets.push(audio.len());
            audio.extend_from_slice(&clip[range.clone()]);
        }
    }
    if gap_frames > 0 && streams.len() > 1 {
        variable |= silence.len() != streams[0].frames[0].len();
    }

    let mut out = xing_frame(&format, &offsets, audio.len(), variable).unwrap_or_default();
    out.extend_from_slice(&audio);
    Ok(out)
}

#[cfg(test)]
pub(crate) mod testing {
    //! Synthetic MP3 data: valid frame headers over silent bodies.

    /// One silent frame. `header` is e.g. `[0xFF, 0xFB, 0x90, 0x00]`
    /// (MPEG-1 layer III, 128 kbit/s, 44.1 kHz, stereo: 417 bytes).
    pub fn frame(header: [u8; 4]) -> Vec<u8> {
        let parsed = super::FrameHeader::parse(&header).expect("valid test header");
        let mut frame = vec![0u8; parsed.frame_len()];
        frame[..4].copy_from_slice(&header);
        frame
    }

    pub fn frames(header: [u8; 4], count: usize) -> Vec<u8> {
        frame(header).repeat(count)
    }

    pub fn id3v2(payload_len: usize) -> Vec<u8> {
        let mut tag = b"ID3\x04\x00\x00".to_vec();
        tag.extend((0..4).rev().map(|shift| ((payload_len >> (7 * shift)) & 0x7F) as u8));
        tag.extend(std::iter::repeat_n(0u8, payload_len));
        tag
    }

    pub fn id3v1() -> Vec<u8> {
        let mut tag = b"TAG".to_vec();
        tag.resize(128, b' ');
        tag
    }

    /// A Xing (or Info) header frame claiming `frames` frames.
    pub fn xing(header: [u8; 4], tag: &[u8; 4], frames: u32) -> Vec<u8> {
        let parsed = super::FrameHeader::parse(&header).expect("valid test header");
        let mut frame = frame(header);
        let at = 4 + parsed.side_info_len();
        frame[at..at + 4].copy_from_slice(tag);
        frame[at + 4..at + 8].copy_from_slice(&1u32.to_be_bytes());
        frame[at + 8..at + 12].copy_from_slice(&frames.to_be_bytes());
        frame
    }

    pub fn vbri(header: [u8; 4], frames: u32) -> Vec<u8> {
        let mut frame = frame(header);
        frame[36..40].copy_from_slice(b"VBRI");
        frame[50..54].copy_from_slice(&frames.to_be_bytes());
        frame
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    /// MPEG-1 layer III, 128 kbit/s, 44.1 kHz, stereo.
    const STEREO_44K: [u8; 4] = [0xFF, 0xFB, 0x90, 0x00];
    /// MPEG-2 layer III, 64 kbit/s, 24 kHz, mono: what most TTS services send.
    const MONO_24K: [u8; 4] = [0xFF, 0xF3, 0x84, 0xC0];
    /// Same stream as `MONO_24K` at 32 kbit/s.
    const MONO_24K_32: [u8; 4] = [0xFF, 0xF3, 0x44, 0xC0];

    fn close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
    }

    #[test]
    fn header_fields_are_decoded() {
        let stereo = FrameHeader::parse(&STEREO_44K).unwrap();
        assert_eq!(stereo.version, MpegVersion::V1);
        assert_eq!((stereo.layer, stereo.bitrate_kbps, stereo.sample_rate), (3, 128, 44_100));
        assert_eq!((stereo.channels, stereo.has_crc, stereo.padding), (2, false, false));
        assert_eq!((stereo.frame_len(), stereo.samples_per_frame()), (417, 1152));

        let mono = FrameHeader::parse(&MONO_24K).unwrap();
        assert_eq!(mono.version, MpegVersion::V2);
        assert_eq!((mono.bitrate_kbps, mono.sample_rate, mono.channels), (64, 24_000, 1));
        assert_eq!((mono.frame_len(), mono.samples_per_frame()), (192, 576));

        let padded = FrameHeader::parse(&[0xFF, 0xFB, 0x92, 0x00]).unwrap();
        assert_eq!(padded.frame_len(), 418);
    }

    #[test]
    fn invalid_headers_are_rejected() {
        for bad in [
            [0xFF, 0xEB, 0x90, 0x00], // reserved version
            [0xFF, 0xF9, 0x90, 0x00], // reserved layer
            [0xFF, 0xFB, 0xF0, 0x00], // bad bitrate
            [0xFF, 0xFB, 0x00, 0x00], // free format
            [0xFF, 0xFB, 0x9C, 0x00], // reserved sample rate
            [0x00, 0xFB, 0x90, 0x00], // no sync
        ] {
            assert!(FrameHeader::parse(&bad).is_none(), "{bad:02X?}");
        }
        assert!(FrameHeader::parse(&[0xFF, 0xFB]).is_none());
    }

    #[test]
    fn cbr_duration_counts_frames() {
        // The original's fixture: 1000 frames x 1152 samples / 44100 Hz.
        close(duration_seconds(&frames(STEREO_44K, 1000)).unwrap(), 1000.0 * 1152.0 / 44_100.0);
        close(duration_seconds(&frames(MONO_24K, 125)).unwrap(), 3.0);
    }

    #[test]
    fn tags_do_not_count_as_audio() {
        let mut data = id3v2(300);
        data.extend(frames(MONO_24K, 50));
        data.extend(id3v1());
        let stream = scan(&data).unwrap();
        assert_eq!(stream.frames.len(), 50);
        assert_eq!(stream.frames[0].start, 310);
        close(stream.duration_seconds(), 1.2);
    }

    #[test]
    fn xing_and_info_headers_are_honoured() {
        for tag in [b"Xing", b"Info"] {
            let mut data = xing(MONO_24K, tag, 250);
            data.extend(frames(MONO_24K, 10));
            let stream = scan(&data).unwrap();
            assert_eq!(stream.stated_frames, Some(250));
            assert_eq!(stream.frames.len(), 10, "the header frame is not audio");
            close(stream.duration_seconds(), 6.0);
        }
    }

    #[test]
    fn vbri_header_is_honoured() {
        let mut data = vbri(STEREO_44K, 500);
        data.extend(frames(STEREO_44K, 3));
        let stream = scan(&data).unwrap();
        assert_eq!(stream.stated_frames, Some(500));
        close(stream.duration_seconds(), 500.0 * 1152.0 / 44_100.0);
    }

    #[test]
    fn variable_bitrate_frames_are_counted_individually() {
        let mut data = frames(MONO_24K, 10);
        data.extend(frames(MONO_24K_32, 15));
        let stream = scan(&data).unwrap();
        assert!(stream.variable_bitrate);
        assert_eq!(stream.frames.len(), 25);
        close(stream.duration_seconds(), 0.6);
    }

    #[test]
    fn junk_between_frames_is_skipped_and_a_cut_off_frame_dropped() {
        let mut data = frames(MONO_24K, 5);
        data.extend(b"garbage that is not audio \xFF\xFF\x00");
        data.extend(frames(MONO_24K, 5));
        data.extend(&frame(MONO_24K)[..100]);
        assert_eq!(scan(&data).unwrap().frames.len(), 10);
    }

    #[test]
    fn non_audio_is_rejected() {
        assert!(duration_seconds(b"definitely not audio").is_err());
        assert!(duration_seconds(b"").is_err());
        assert!(duration_seconds(&id3v2(64)).is_err());
        assert!(duration_seconds(&xing(MONO_24K, b"Info", 0)).is_err());
    }

    #[test]
    fn concat_drops_tags_and_headers_and_writes_one_new_header() {
        let mut first = id3v2(120);
        first.extend(xing(MONO_24K, b"Info", 40));
        first.extend(frames(MONO_24K, 40));
        first.extend(id3v1());
        let mut second = xing(MONO_24K, b"Xing", 60);
        second.extend(frames(MONO_24K, 60));

        let joined = concat(&[first, second], 0).unwrap();
        assert!(!joined.windows(3).any(|w| w == b"ID3" || w == b"TAG"));
        assert_eq!(joined.windows(4).filter(|w| w == b"Info" || w == b"Xing").count(), 1);

        let stream = scan(&joined).unwrap();
        assert_eq!(stream.stated_frames, Some(100));
        assert_eq!(stream.frames.len(), 100);
        close(stream.duration_seconds(), 100.0 * 576.0 / 24_000.0);

        // The header states the size of the whole file.
        let at = 4 + 9 + 12;
        assert_eq!(be32(&joined, at), Some(joined.len() as u32));
        // The seek table rises from the start of the audio to near the end.
        let toc = &joined[at + 4..at + 104];
        assert!(toc.windows(2).all(|w| w[0] <= w[1]));
        assert!(toc[0] < 8 && toc[99] > 240, "{toc:?}");
    }

    #[test]
    fn concat_inserts_silence_between_clips_only() {
        let clip = frames(MONO_24K, 25); // 0.6 s
        let joined = concat(&[clip.clone(), clip.clone(), clip], 400).unwrap();
        let stream = scan(&joined).unwrap();
        // 400 ms / 24 ms per frame rounds to 17 frames, twice.
        assert_eq!(stream.frames.len(), 75 + 2 * 17);
        assert_eq!(stream.stated_frames, Some(109));
        close(duration_seconds(&joined).unwrap(), 109.0 * 0.024);
    }

    #[test]
    fn concat_of_mixed_bitrates_is_marked_variable() {
        let joined = concat(&[frames(MONO_24K, 10), frames(MONO_24K_32, 10)], 0).unwrap();
        assert_eq!(joined.windows(4).filter(|w| w == b"Xing").count(), 1);
        assert_eq!(scan(&joined).unwrap().stated_frames, Some(20));
    }

    #[test]
    fn concat_of_a_single_clip_strips_and_rewrites_the_header() {
        let mut clip = id3v2(10);
        clip.extend(xing(STEREO_44K, b"Info", 999));
        clip.extend(frames(STEREO_44K, 20));
        let joined = concat(&[clip], 400).unwrap();
        let stream = scan(&joined).unwrap();
        assert_eq!(stream.stated_frames, Some(20));
        assert_eq!(stream.frames.len(), 20);
    }

    #[test]
    fn concat_refuses_mixed_formats_and_non_mp3() {
        let mixed = concat(&[frames(MONO_24K, 5), frames(STEREO_44K, 5)], 0).unwrap_err();
        assert!(mixed.contains("different MP3 formats"), "{mixed}");
        assert!(mixed.contains("24000 Hz, mono") && mixed.contains("44100 Hz, stereo"), "{mixed}");

        let not_mp3 = concat(&[frames(MONO_24K, 5), b"RIFF....WAVE".to_vec()], 0).unwrap_err();
        assert!(not_mp3.contains("clip 0001 is not a usable MP3 file"), "{not_mp3}");

        assert!(concat(&[], 0).is_err());
    }
}
