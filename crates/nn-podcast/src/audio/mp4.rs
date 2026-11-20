//! MP4 / M4A duration from the container's own header boxes.
//!
//! Only box headers and the small `mvhd` / `mdhd` boxes are read, so a large
//! file whose `moov` box sits after the audio data costs a few seeks, not a
//! full read.

use std::io::{Read, Seek, SeekFrom};

struct BoxHeader {
    kind: [u8; 4],
    /// Offset of the first payload byte.
    payload: u64,
    /// Offset one past the last payload byte.
    end: u64,
}

fn read_box<R: Read + Seek>(reader: &mut R, at: u64, limit: u64) -> Result<Option<BoxHeader>, String> {
    if at + 8 > limit {
        return Ok(None);
    }
    let io = |e: std::io::Error| format!("could not read the MP4 file: {e}");
    reader.seek(SeekFrom::Start(at)).map_err(io)?;
    let mut head = [0u8; 8];
    reader.read_exact(&mut head).map_err(io)?;
    let kind = [head[4], head[5], head[6], head[7]];
    let (size, header_len) = match u32::from_be_bytes([head[0], head[1], head[2], head[3]]) {
        0 => (limit - at, 8), // the box runs to the end of its container
        1 => {
            let mut large = [0u8; 8];
            reader.read_exact(&mut large).map_err(io)?;
            (u64::from_be_bytes(large), 16)
        }
        size => (u64::from(size), 8),
    };
    if size < header_len || at + size > limit {
        return Err("the MP4 file is damaged (a box runs past the end of the file)".into());
    }
    Ok(Some(BoxHeader { kind, payload: at + header_len, end: at + size }))
}

/// Find the first child box of type `kind` inside `start..end`.
fn find_box<R: Read + Seek>(
    reader: &mut R,
    kind: &[u8; 4],
    start: u64,
    end: u64,
) -> Result<Option<BoxHeader>, String> {
    let mut at = start;
    while let Some(found) = read_box(reader, at, end)? {
        if &found.kind == kind {
            return Ok(Some(found));
        }
        at = found.end;
    }
    Ok(None)
}

/// `(timescale, duration)` from an `mvhd` or `mdhd` box: both start with a
/// version byte, three flag bytes and two timestamps, then these two fields.
fn timescale_and_duration<R: Read + Seek>(reader: &mut R, header: &BoxHeader) -> Result<(u32, u64), String> {
    let io = |e: std::io::Error| format!("could not read the MP4 file: {e}");
    let mut payload = vec![0u8; (header.end - header.payload).min(32) as usize];
    reader.seek(SeekFrom::Start(header.payload)).map_err(io)?;
    reader.read_exact(&mut payload).map_err(io)?;
    let cut_off = || "the MP4 file is damaged (a header box is cut off)".to_string();
    let be32 = |at: usize| -> Result<u32, String> {
        payload.get(at..at + 4).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]])).ok_or_else(cut_off)
    };
    match payload.first() {
        Some(1) => {
            let timescale = be32(20)?;
            let duration = (u64::from(be32(24)?) << 32) | u64::from(be32(28)?);
            Ok((timescale, duration))
        }
        Some(_) => Ok((be32(12)?, u64::from(be32(16)?))),
        None => Err(cut_off()),
    }
}

/// Length in seconds of an MP4 / M4A file.
///
/// The movie header (`mvhd`) is authoritative. When it states no duration, the
/// longest track's media header (`mdhd`) is used instead.
pub(crate) fn duration_seconds<R: Read + Seek>(reader: &mut R, file_len: u64) -> Result<f64, String> {
    match read_box(reader, 0, file_len)? {
        Some(first) if &first.kind == b"ftyp" => {}
        _ => return Err("not an MP4 file".into()),
    }
    let moov = find_box(reader, b"moov", 0, file_len)?.ok_or("the MP4 file has no movie header")?;

    if let Some(mvhd) = find_box(reader, b"mvhd", moov.payload, moov.end)? {
        let (timescale, duration) = timescale_and_duration(reader, &mvhd)?;
        if timescale > 0 && duration > 0 {
            return Ok(duration as f64 / f64::from(timescale));
        }
    }

    let mut longest: f64 = 0.0;
    let mut at = moov.payload;
    while let Some(child) = read_box(reader, at, moov.end)? {
        if &child.kind == b"trak" {
            if let Some(mdia) = find_box(reader, b"mdia", child.payload, child.end)? {
                if let Some(mdhd) = find_box(reader, b"mdhd", mdia.payload, mdia.end)? {
                    let (timescale, duration) = timescale_and_duration(reader, &mdhd)?;
                    if timescale > 0 {
                        longest = longest.max(duration as f64 / f64::from(timescale));
                    }
                }
            }
        }
        at = child.end;
    }
    if longest > 0.0 {
        Ok(longest)
    } else {
        Err("the MP4 file states no duration".into())
    }
}

#[cfg(test)]
pub(crate) mod testing {
    pub fn boxed(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
        out.extend(kind);
        out.extend(payload);
        out
    }

    pub fn mvhd_v0(timescale: u32, duration: u32) -> Vec<u8> {
        let mut payload = vec![0u8; 12]; // version, flags, created, modified
        payload.extend(timescale.to_be_bytes());
        payload.extend(duration.to_be_bytes());
        payload.extend([0u8; 80]);
        boxed(b"mvhd", &payload)
    }

    pub fn mvhd_v1(timescale: u32, duration: u64) -> Vec<u8> {
        let mut payload = vec![1u8, 0, 0, 0];
        payload.extend([0u8; 16]); // created, modified (64-bit each)
        payload.extend(timescale.to_be_bytes());
        payload.extend(duration.to_be_bytes());
        payload.extend([0u8; 80]);
        boxed(b"mvhd", &payload)
    }

    pub fn ftyp() -> Vec<u8> {
        boxed(b"ftyp", b"M4A \0\0\0\0M4A mp42isom")
    }

    /// A minimal M4A file: `ftyp`, some audio data, then the movie header.
    pub fn m4a(timescale: u32, duration: u32) -> Vec<u8> {
        let mut file = ftyp();
        file.extend(boxed(b"mdat", &[0xAA; 500]));
        file.extend(boxed(b"moov", &mvhd_v0(timescale, duration)));
        file
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::testing::*;
    use super::*;

    fn duration(file: &[u8]) -> Result<f64, String> {
        duration_seconds(&mut Cursor::new(file), file.len() as u64)
    }

    #[test]
    fn mvhd_gives_the_duration() {
        assert_eq!(duration(&m4a(1000, 3000)).unwrap(), 3.0);
        assert_eq!(duration(&m4a(44_100, 44_100 * 26 + 22_050)).unwrap(), 26.5);
    }

    #[test]
    fn moov_before_mdat_and_version_1_headers_are_read() {
        let mut file = ftyp();
        file.extend(boxed(b"moov", &mvhd_v1(48_000, 48_000 * 7_200)));
        file.extend(boxed(b"mdat", &[0u8; 64]));
        assert_eq!(duration(&file).unwrap(), 7_200.0);
    }

    #[test]
    fn large_size_boxes_are_skipped() {
        let mut file = ftyp();
        let mut mdat = 1u32.to_be_bytes().to_vec();
        mdat.extend(b"mdat");
        mdat.extend((16u64 + 100).to_be_bytes());
        mdat.extend([0u8; 100]);
        file.extend(mdat);
        file.extend(boxed(b"moov", &mvhd_v0(600, 1_500)));
        assert_eq!(duration(&file).unwrap(), 2.5);
    }

    #[test]
    fn track_header_is_used_when_the_movie_header_states_nothing() {
        let mut mdhd = vec![0u8; 12];
        mdhd.extend(24_000u32.to_be_bytes());
        mdhd.extend(96_000u32.to_be_bytes());
        mdhd.extend([0u8; 4]);
        let trak = boxed(b"trak", &boxed(b"mdia", &boxed(b"mdhd", &mdhd)));
        let mut moov = mvhd_v0(1000, 0);
        moov.extend(trak);
        let mut file = ftyp();
        file.extend(boxed(b"moov", &moov));
        assert_eq!(duration(&file).unwrap(), 4.0);
    }

    #[test]
    fn damaged_and_foreign_files_are_rejected() {
        assert!(duration(b"definitely not audio").is_err());
        assert!(duration(b"").is_err());
        assert!(duration(&ftyp()).unwrap_err().contains("no movie header"));

        let whole = m4a(1000, 3000);
        assert!(duration(&whole[..whole.len() - 20]).unwrap_err().contains("damaged"));

        let mut empty = ftyp();
        empty.extend(boxed(b"moov", &mvhd_v0(1000, 0)));
        assert!(duration(&empty).unwrap_err().contains("no duration"));
    }
}
