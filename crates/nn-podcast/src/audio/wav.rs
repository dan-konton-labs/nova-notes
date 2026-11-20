//! RIFF/WAVE without a decoder: reading the format, measuring, and joining
//! uncompressed clips by rewriting the header.

use std::ops::Range;

const FORMAT_PCM: u16 = 1;
const FORMAT_IEEE_FLOAT: u16 = 3;
const FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// What the `fmt ` chunk says, plus where the samples are.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WavInfo {
    /// The `fmt ` chunk payload, kept verbatim for the rewritten header.
    pub fmt: Vec<u8>,
    pub format_tag: u16,
    pub channels: u16,
    pub sample_rate: u32,
    pub block_align: u16,
    pub bits_per_sample: u16,
    /// Byte range of the sample data, a whole number of sample frames.
    pub data: Range<usize>,
}

fn le16(bytes: &[u8], at: usize) -> Option<u16> {
    Some(u16::from_le_bytes(bytes.get(at..at + 2)?.try_into().ok()?))
}

fn le32(bytes: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.get(at..at + 4)?.try_into().ok()?))
}

impl WavInfo {
    pub fn duration_seconds(&self) -> f64 {
        let frames = self.data.len() / usize::from(self.block_align);
        frames as f64 / f64::from(self.sample_rate)
    }

    fn same_format(&self, other: &WavInfo) -> bool {
        self.format_tag == other.format_tag
            && self.channels == other.channels
            && self.sample_rate == other.sample_rate
            && self.block_align == other.block_align
            && self.bits_per_sample == other.bits_per_sample
    }

    fn describe(&self) -> String {
        format!("{} Hz, {} channel(s), {} bit", self.sample_rate, self.channels, self.bits_per_sample)
    }

    /// The byte that encodes silence: 8-bit PCM is unsigned, everything else
    /// is centred on zero.
    fn silence_byte(&self) -> u8 {
        if self.format_tag == FORMAT_PCM && self.bits_per_sample == 8 {
            0x80
        } else {
            0
        }
    }
}

/// Read the format and locate the samples of a WAV file held in memory.
pub(crate) fn parse(data: &[u8]) -> Result<WavInfo, String> {
    if data.len() < 12 || &data[..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }

    let mut fmt: Option<&[u8]> = None;
    let mut samples: Option<Range<usize>> = None;
    let mut position = 12;
    while position + 8 <= data.len() {
        let id = &data[position..position + 4];
        let declared = le32(data, position + 4).unwrap_or(0) as usize;
        let start = position + 8;
        if id == b"data" {
            // Streamed WAV files state a size of 0 or 0xFFFFFFFF because the
            // length was unknown when the header was sent: the samples then
            // run to the end of the file.
            let available = data.len() - start;
            let len = if declared == 0 || declared > available { available } else { declared };
            samples = Some(start..start + len);
            break;
        }
        let end = start.checked_add(declared).filter(|&end| end <= data.len());
        let Some(end) = end else {
            return Err("the WAV file is cut off inside a chunk".into());
        };
        if id == b"fmt " {
            fmt = Some(&data[start..end]);
        }
        position = end + (declared & 1); // chunks are padded to an even size
    }

    let fmt = fmt.filter(|fmt| fmt.len() >= 16).ok_or("the WAV file has no format chunk")?;
    let samples = samples.ok_or("the WAV file has no data chunk")?;
    let format_tag = le16(fmt, 0).unwrap_or(0);
    let channels = le16(fmt, 2).unwrap_or(0);
    let sample_rate = le32(fmt, 4).unwrap_or(0);
    let block_align = le16(fmt, 12).unwrap_or(0);
    let bits_per_sample = le16(fmt, 14).unwrap_or(0);
    if !matches!(format_tag, FORMAT_PCM | FORMAT_IEEE_FLOAT | FORMAT_EXTENSIBLE) {
        return Err(format!("unsupported WAV encoding (format tag {format_tag}); only uncompressed audio can be joined"));
    }
    if channels == 0 || sample_rate == 0 || block_align == 0 {
        return Err("the WAV format chunk is invalid".into());
    }

    let whole_frames = samples.len() - samples.len() % usize::from(block_align);
    Ok(WavInfo {
        fmt: fmt.to_vec(),
        format_tag,
        channels,
        sample_rate,
        block_align,
        bits_per_sample,
        data: samples.start..samples.start + whole_frames,
    })
}

/// Length of a WAV file in seconds.
pub(crate) fn duration_seconds(data: &[u8]) -> Result<f64, String> {
    let info = parse(data)?;
    if info.data.is_empty() {
        return Err("the WAV file has no audio data".into());
    }
    Ok(info.duration_seconds())
}

/// Join WAV clips of one format into a single file with a fresh header and
/// `gap_ms` of silence between consecutive clips.
pub(crate) fn concat(clips: &[Vec<u8>], gap_ms: u32) -> Result<Vec<u8>, String> {
    let mut infos = Vec::with_capacity(clips.len());
    for (index, clip) in clips.iter().enumerate() {
        infos.push(parse(clip).map_err(|e| format!("clip {index:04} is not a usable WAV file: {e}"))?);
    }
    let format = infos.first().ok_or("there are no clips to combine")?;
    for (index, info) in infos.iter().enumerate() {
        if !info.same_format(format) {
            return Err(format!(
                "clips have different WAV formats (clip 0000 is {}, clip {index:04} is {})",
                format.describe(),
                info.describe()
            ));
        }
    }

    let gap_frames = u64::from(format.sample_rate) * u64::from(gap_ms) / 1000;
    let gap = vec![format.silence_byte(); gap_frames as usize * usize::from(format.block_align)];

    let mut samples = Vec::with_capacity(clips.iter().map(Vec::len).sum::<usize>());
    for (index, (clip, info)) in clips.iter().zip(&infos).enumerate() {
        if index > 0 {
            samples.extend_from_slice(&gap);
        }
        samples.extend_from_slice(&clip[info.data.clone()]);
    }

    let fmt_padding = format.fmt.len() & 1;
    let data_padding = samples.len() & 1;
    let riff_size = 4 + (8 + format.fmt.len() + fmt_padding) + (8 + samples.len() + data_padding);
    let too_large = || "the combined audio is too large for a WAV file (4 GB limit)".to_string();
    let riff_size = u32::try_from(riff_size).map_err(|_| too_large())?;
    let data_size = u32::try_from(samples.len()).map_err(|_| too_large())?;

    let mut out = Vec::with_capacity(riff_size as usize + 8);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&(format.fmt.len() as u32).to_le_bytes());
    out.extend_from_slice(&format.fmt);
    out.extend(std::iter::repeat_n(0u8, fmt_padding));
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());
    out.extend_from_slice(&samples);
    out.extend(std::iter::repeat_n(0u8, data_padding));
    Ok(out)
}

#[cfg(test)]
pub(crate) mod testing {
    /// A 16-bit PCM WAV file holding `samples` (interleaved when stereo).
    pub fn pcm16(sample_rate: u32, channels: u16, samples: &[i16]) -> Vec<u8> {
        let data: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        with_header(sample_rate, channels, data.len() as u32, &data)
    }

    /// Like [`pcm16`] but with an explicit `data` size field.
    pub fn with_header(sample_rate: u32, channels: u16, declared: u32, data: &[u8]) -> Vec<u8> {
        let block_align = channels * 2;
        let mut out = b"RIFF".to_vec();
        out.extend((36 + data.len() as u32).to_le_bytes());
        out.extend(b"WAVEfmt ");
        out.extend(16u32.to_le_bytes());
        out.extend(1u16.to_le_bytes());
        out.extend(channels.to_le_bytes());
        out.extend(sample_rate.to_le_bytes());
        out.extend((sample_rate * u32::from(block_align)).to_le_bytes());
        out.extend(block_align.to_le_bytes());
        out.extend(16u16.to_le_bytes());
        out.extend(b"data");
        out.extend(declared.to_le_bytes());
        out.extend(data);
        out
    }

    /// `seconds` of a sine tone, mono 16-bit.
    pub fn tone(sample_rate: u32, seconds: f64, hz: f64) -> Vec<u8> {
        let count = (f64::from(sample_rate) * seconds).round() as usize;
        let samples: Vec<i16> = (0..count)
            .map(|i| {
                let t = i as f64 / f64::from(sample_rate);
                ((t * hz * std::f64::consts::TAU).sin() * 12_000.0) as i16
            })
            .collect();
        pcm16(sample_rate, 1, &samples)
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    #[test]
    fn header_fields_and_duration_are_read() {
        let wav = pcm16(24_000, 1, &vec![7i16; 12_000]);
        let info = parse(&wav).unwrap();
        assert_eq!((info.format_tag, info.channels, info.sample_rate), (1, 1, 24_000));
        assert_eq!((info.block_align, info.bits_per_sample), (2, 16));
        assert_eq!(info.data, 44..44 + 24_000);
        assert_eq!(duration_seconds(&wav).unwrap(), 0.5);

        let stereo = pcm16(48_000, 2, &vec![0i16; 96_000]);
        assert_eq!(duration_seconds(&stereo).unwrap(), 1.0);
    }

    #[test]
    fn extra_chunks_are_skipped() {
        let plain = pcm16(8_000, 1, &[1, 2, 3, 4]);
        // Insert an odd-sized LIST chunk (padded to even) before `data`.
        let mut wav = plain[..36].to_vec();
        wav.extend(b"LIST");
        wav.extend(3u32.to_le_bytes());
        wav.extend(b"abc\0");
        wav.extend(&plain[36..]);
        let info = parse(&wav).unwrap();
        assert_eq!(&wav[info.data.clone()], &plain[44..]);
    }

    #[test]
    fn streamed_headers_without_a_length_run_to_the_end() {
        let samples: Vec<u8> = (0..2_000u32).flat_map(|i| (i as i16).to_le_bytes()).collect();
        for declared in [0u32, u32::MAX] {
            let wav = with_header(16_000, 1, declared, &samples);
            assert_eq!(parse(&wav).unwrap().data.len(), 4_000);
        }
        // A trailing half sample frame is dropped.
        let wav = with_header(16_000, 1, u32::MAX, &samples[..3_999]);
        assert_eq!(parse(&wav).unwrap().data.len(), 3_998);
    }

    #[test]
    fn non_wav_and_damaged_files_are_rejected() {
        assert!(parse(b"RIFF....WAVE").unwrap_err().contains("no format chunk"));
        assert!(parse(b"definitely not audio").unwrap_err().contains("not a RIFF/WAVE"));
        let wav = pcm16(8_000, 1, &[1, 2, 3]);
        assert!(parse(&wav[..30]).unwrap_err().contains("cut off"));
        assert!(duration_seconds(&pcm16(8_000, 1, &[])).is_err());

        let mut adpcm = pcm16(8_000, 1, &[1, 2, 3]);
        adpcm[20] = 2;
        assert!(parse(&adpcm).unwrap_err().contains("unsupported WAV encoding"));
    }

    #[test]
    fn concat_joins_samples_under_one_header() {
        let first = pcm16(24_000, 1, &[1, 2, 3]);
        let second = pcm16(24_000, 1, &[4, 5]);
        let joined = concat(&[first, second], 0).unwrap();

        assert_eq!(joined.len(), 44 + 10);
        assert_eq!(le32(&joined, 4), Some(joined.len() as u32 - 8));
        assert_eq!(le32(&joined, 40), Some(10));
        let info = parse(&joined).unwrap();
        let samples: Vec<i16> =
            joined[info.data].chunks(2).map(|b| i16::from_le_bytes([b[0], b[1]])).collect();
        assert_eq!(samples, [1, 2, 3, 4, 5]);
    }

    #[test]
    fn concat_inserts_silence_between_clips_only() {
        let clip = pcm16(8_000, 2, &[100; 16]); // 8 stereo frames
        let joined = concat(&[clip.clone(), clip.clone(), clip], 250).unwrap();
        let info = parse(&joined).unwrap();
        // 250 ms at 8 kHz is 2000 frames of 4 bytes, twice.
        assert_eq!(info.data.len(), 3 * 32 + 2 * 8_000);
        assert!(joined[info.data.start + 32..info.data.start + 32 + 8_000].iter().all(|&b| b == 0));
        assert_eq!(info.duration_seconds(), (24.0 + 4_000.0) / 8_000.0);
    }

    #[test]
    fn concat_refuses_mixed_formats_and_non_wav() {
        let mixed =
            concat(&[pcm16(24_000, 1, &[1, 2]), pcm16(44_100, 1, &[1, 2])], 0).unwrap_err();
        assert!(mixed.contains("different WAV formats"), "{mixed}");
        assert!(mixed.contains("24000 Hz") && mixed.contains("44100 Hz"), "{mixed}");

        let not_wav = concat(&[pcm16(24_000, 1, &[1]), b"ID3 nope".to_vec()], 0).unwrap_err();
        assert!(not_wav.contains("clip 0001 is not a usable WAV file"), "{not_wav}");
        assert!(concat(&[], 0).is_err());
    }
}
