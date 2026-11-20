//! Podcast Feed artwork: validation and storage (Show Metadata).
//!
//! Apple Podcasts wants square PNG or JPEG artwork between 1400 and 3000
//! pixels. The original checked uploads with Pillow, which reads only the
//! header for format and size and, for PNG, verifies every chunk checksum.
//! The same rules are enforced here with a small header parser: nothing is
//! decoded, and the original bytes are stored untouched.
//!
//! The format always comes from the content, never from a file name.

use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::audio_paths::podcasts_root;
use crate::error::{PodcastError, Result};

pub const MIN_PIXELS: u32 = 1400;
pub const MAX_PIXELS: u32 = 3000;
pub const MAX_BYTES: usize = 10 * 1024 * 1024;

/// File extension to media type, for the two accepted formats.
pub const MEDIA_TYPES: [(&str, &str); 2] = [(".png", "image/png"), (".jpg", "image/jpeg")];

/// Pixel count above which Pillow refused to open an image at all
/// (twice its `MAX_IMAGE_PIXELS`).
const DECOMPRESSION_BOMB_PIXELS: u64 = 2 * 89_478_485;

const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";

/// A validated artwork upload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Artwork {
    /// `.png` or `.jpg`.
    pub extension: &'static str,
    pub media_type: &'static str,
    /// Short content hash; changes whenever the image does.
    pub version: String,
}

fn invalid<T>(message: impl Into<String>) -> Result<T> {
    Err(PodcastError::InvalidInput(message.into()))
}

fn not_an_image<T>() -> Result<T> {
    invalid("The artwork is not a PNG or JPEG image.")
}

fn damaged<T>() -> Result<T> {
    invalid("Could not read the artwork; the image file looks damaged.")
}

const fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut n = 0;
    while n < 256 {
        let mut c = n as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        table[n] = c;
        n += 1;
    }
    table
}

static CRC32_TABLE: [u32; 256] = crc32_table();

fn crc32(parts: &[&[u8]]) -> u32 {
    let mut crc = u32::MAX;
    for byte in parts.iter().flat_map(|part| part.iter()) {
        crc = CRC32_TABLE[((crc ^ u32::from(*byte)) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

fn be32(data: &[u8], at: usize) -> Option<u32> {
    Some(u32::from_be_bytes(data.get(at..at + 4)?.try_into().ok()?))
}

/// One PNG chunk at `at`: `(type, payload, offset of the next chunk)`, with
/// the checksum verified. `None` when it is cut off, malformed or corrupt.
fn png_chunk(data: &[u8], at: usize) -> Option<(&[u8], &[u8], usize)> {
    let length = be32(data, at)? as usize;
    let kind = data.get(at + 4..at + 8)?;
    if !kind.iter().all(u8::is_ascii_alphabetic) {
        return None;
    }
    let payload = data.get(at + 8..(at + 8).checked_add(length)?)?;
    let stated = be32(data, at + 8 + length)?;
    (crc32(&[kind, payload]) == stated).then_some((kind, payload, at + 12 + length))
}

/// Width and height from the header chunk that must open a PNG file.
fn png_size(data: &[u8]) -> Option<(u32, u32)> {
    match png_chunk(data, PNG_SIGNATURE.len())? {
        (b"IHDR", header, _) if header.len() == 13 => Some((be32(header, 0)?, be32(header, 4)?)),
        _ => None,
    }
}

/// Every chunk is whole and matches its checksum, up to the end marker.
fn png_is_intact(data: &[u8]) -> bool {
    let mut at = PNG_SIGNATURE.len();
    let mut has_image_data = false;
    loop {
        // The end marker only has to be present: Pillow stops reading there.
        if data.get(at + 4..at + 8) == Some(b"IEND") {
            return has_image_data;
        }
        match png_chunk(data, at) {
            Some((kind, _, next)) => {
                has_image_data |= kind == b"IDAT";
                at = next;
            }
            None => return false,
        }
    }
}

enum JpegHeader {
    /// Markers are well formed up to the start of the image data.
    Size(u32, u32),
    /// The file ends before any image data.
    NotAnImage,
    /// A segment is cut off.
    Damaged,
}

/// Walk the JPEG segments up to the start of scan, taking the size from the
/// frame header on the way.
fn jpeg_header(data: &[u8]) -> JpegHeader {
    let mut size = (0, 0);
    let mut at = 2;
    loop {
        // Skip to the next marker: fill bytes and stray data are legal here.
        while data.get(at).is_some_and(|&byte| byte != 0xFF) {
            at += 1;
        }
        while data.get(at + 1) == Some(&0xFF) {
            at += 1;
        }
        let Some(&marker) = data.get(at + 1) else {
            return JpegHeader::NotAnImage;
        };
        at += 2;
        match marker {
            // Stuffed byte, restart markers, start/end of image, TEM: no payload.
            0x00 | 0x01 | 0xD0..=0xD9 => continue,
            _ => {}
        }
        let Some(length) = data.get(at..at + 2).map(|b| usize::from(u16::from_be_bytes([b[0], b[1]]))) else {
            return JpegHeader::NotAnImage;
        };
        let Some(segment) = (length >= 2).then(|| data.get(at + 2..at + length)).flatten() else {
            return JpegHeader::Damaged;
        };
        match marker {
            0xDA => return JpegHeader::Size(size.0, size.1),
            // Start-of-frame markers; C4, C8 and CC share the range but are tables.
            0xC0..=0xCF if !matches!(marker, 0xC4 | 0xC8 | 0xCC) => {
                if segment.len() < 6 {
                    return JpegHeader::Damaged;
                }
                let height = u32::from(u16::from_be_bytes([segment[1], segment[2]]));
                let width = u32::from(u16::from_be_bytes([segment[3], segment[4]]));
                size = (width, height);
            }
            _ => {}
        }
        at += length;
    }
}

/// Name of a well-known image format that is neither PNG nor JPEG.
fn other_format(data: &[u8]) -> Option<&'static str> {
    let at = |range: std::ops::Range<usize>| data.get(range);
    if data.starts_with(b"GIF87a") || data.starts_with(b"GIF89a") {
        Some("GIF")
    } else if data.starts_with(b"RIFF") && at(8..12) == Some(b"WEBP") {
        Some("WEBP")
    } else if data.starts_with(b"II*\0") || data.starts_with(b"MM\0*") {
        Some("TIFF")
    } else if data.starts_with(b"BM") && matches!(at(14..18).and_then(be32_le), Some(12 | 40 | 52 | 56 | 64 | 108 | 124)) {
        Some("BMP")
    } else if data.starts_with(b"\0\0\x01\0") && at(4..6).is_some_and(|count| count != [0, 0]) {
        Some("ICO")
    } else if data.starts_with(b"8BPS") {
        Some("PSD")
    } else if data.starts_with(b"\0\0\0\x0cjP  \r\n\x87\n") || data.starts_with(b"\xFF\x4F\xFF\x51") {
        Some("JPEG2000")
    } else if at(4..12) == Some(b"ftypavif") {
        Some("AVIF")
    } else {
        None
    }
}

fn be32_le(bytes: &[u8]) -> Option<u32> {
    Some(u32::from_le_bytes(bytes.try_into().ok()?))
}

/// Check an uploaded image; fail with a specific reason.
///
/// Rules, in the order they are checked: not empty; at most 10 MB; a PNG or
/// JPEG by content; square; at least 1400 and at most 3000 pixels a side; and,
/// for PNG, every chunk intact.
pub fn validate_artwork(data: &[u8]) -> Result<Artwork> {
    if data.is_empty() {
        return invalid("The artwork file is empty.");
    }
    if data.len() > MAX_BYTES {
        return invalid("The artwork file is too large (10 MB at most).");
    }

    let is_png = data.starts_with(PNG_SIGNATURE);
    let (width, height) = if is_png {
        match png_size(data) {
            Some(size) => size,
            None => return not_an_image(),
        }
    } else if data.starts_with(b"\xFF\xD8\xFF") {
        match jpeg_header(data) {
            JpegHeader::Size(width, height) => (width, height),
            JpegHeader::NotAnImage => return not_an_image(),
            JpegHeader::Damaged => return damaged(),
        }
    } else {
        return match other_format(data) {
            Some(format) => invalid(format!("The artwork must be a PNG or JPEG image (got {format}).")),
            None => not_an_image(),
        };
    };

    if u64::from(width.max(1)) * u64::from(height.max(1)) > DECOMPRESSION_BOMB_PIXELS {
        return invalid(format!("The artwork must be at most {MAX_PIXELS}x{MAX_PIXELS} pixels."));
    }
    if width != height {
        return invalid(format!("The artwork must be square (this one is {width}x{height} pixels)."));
    }
    if width < MIN_PIXELS {
        return invalid(format!(
            "The artwork must be at least {MIN_PIXELS}x{MIN_PIXELS} pixels (this one is {width}x{height})."
        ));
    }
    if width > MAX_PIXELS {
        return invalid(format!(
            "The artwork must be at most {MAX_PIXELS}x{MAX_PIXELS} pixels (this one is {width}x{height})."
        ));
    }
    if is_png && !png_is_intact(data) {
        return damaged();
    }

    let (extension, media_type) = if is_png { MEDIA_TYPES[0] } else { MEDIA_TYPES[1] };
    Ok(Artwork { extension, media_type, version: hex::encode(Sha256::digest(data))[..12].to_string() })
}

/// The media type of a stored artwork extension (`.png`, `.jpg`).
pub fn media_type_for(extension: &str) -> Option<&'static str> {
    MEDIA_TYPES.iter().find(|(known, _)| *known == extension).map(|(_, media_type)| *media_type)
}

/// Folder holding the artwork, inside the podcasts root (never under `episodes/`).
pub fn artwork_dir(podcasts_folder: &Path) -> PathBuf {
    podcasts_root(podcasts_folder).join("feed")
}

pub fn artwork_path(podcasts_folder: &Path, extension: &str) -> PathBuf {
    artwork_dir(podcasts_folder).join(format!("artwork{extension}"))
}

/// Write the file (atomically), then remove a leftover of the other format.
pub fn save_artwork(podcasts_folder: &Path, data: &[u8], artwork: &Artwork) -> Result<()> {
    let folder = artwork_dir(podcasts_folder);
    let failed = |e: std::io::Error| PodcastError::io("Failed to save the artwork", e);
    std::fs::create_dir_all(&folder).map_err(failed)?;
    let target = artwork_path(podcasts_folder, artwork.extension);
    let temporary = folder.join(format!("artwork{}.tmp", artwork.extension));
    std::fs::write(&temporary, data).map_err(failed)?;
    std::fs::rename(&temporary, &target).map_err(failed)?;
    for (extension, _) in MEDIA_TYPES {
        if extension != artwork.extension {
            match std::fs::remove_file(artwork_path(podcasts_folder, extension)) {
                Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(failed(e)),
                _ => {}
            }
        }
    }
    Ok(())
}

/// The stored artwork file and its media type, or `None` if there is none.
///
/// Only a known extension is accepted and the path is built from a fixed
/// name, so a bad record value can never point outside the artwork folder.
/// A record whose file vanished counts as no artwork.
pub fn current_artwork(podcasts_folder: &Path, extension: Option<&str>) -> Option<(PathBuf, &'static str)> {
    let extension = extension?;
    let media_type = media_type_for(extension)?;
    let path = artwork_path(podcasts_folder, extension);
    path.is_file().then_some((path, media_type))
}

#[cfg(test)]
pub(crate) mod testing {
    use super::*;

    fn chunk(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = (payload.len() as u32).to_be_bytes().to_vec();
        out.extend(kind);
        out.extend(payload);
        out.extend(crc32(&[kind, payload]).to_be_bytes());
        out
    }

    /// A structurally valid PNG of the given size. `seed` varies the content.
    pub fn png(width: u32, height: u32, seed: u8) -> Vec<u8> {
        let mut header = width.to_be_bytes().to_vec();
        header.extend(height.to_be_bytes());
        header.extend([8, 2, 0, 0, 0]); // 8-bit RGB
        let mut out = PNG_SIGNATURE.to_vec();
        out.extend(chunk(b"IHDR", &header));
        out.extend(chunk(b"tEXt", b"Software\0nn-podcast tests"));
        out.extend(chunk(b"IDAT", &[seed; 300]));
        out.extend(chunk(b"IEND", b""));
        out
    }

    /// A structurally valid JPEG header of the given size, with some scan data.
    pub fn jpeg(width: u16, height: u16, seed: u8) -> Vec<u8> {
        let mut out = vec![0xFF, 0xD8];
        out.extend([0xFF, 0xE0, 0x00, 0x10]);
        out.extend(b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0");
        out.extend([0xFF, 0xDB, 0x00, 0x43, 0x00]);
        out.extend([seed.max(1); 64]);
        out.extend([0xFF, 0xC0, 0x00, 0x11, 0x08]);
        out.extend(height.to_be_bytes());
        out.extend(width.to_be_bytes());
        out.extend([0x03, 0x01, 0x22, 0x00, 0x02, 0x11, 0x01, 0x03, 0x11, 0x01]);
        out.extend([0xFF, 0xDA, 0x00, 0x0C, 0x03, 0x01, 0x00, 0x02, 0x11, 0x03, 0x11, 0x00, 0x3F, 0x00]);
        out.extend([seed; 40]);
        out.extend([0xFF, 0xD9]);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;

    fn refusal(data: &[u8]) -> String {
        let error = validate_artwork(data).unwrap_err();
        assert!(matches!(error, PodcastError::InvalidInput(_)), "{error:?}");
        error.to_string()
    }

    #[test]
    fn crc32_matches_the_standard_check_value() {
        assert_eq!(crc32(&[b"123456789"]), 0xCBF4_3926);
        assert_eq!(crc32(&[b"1234", b"56789"]), 0xCBF4_3926);
        // The checksum every PNG ends with.
        assert_eq!(crc32(&[b"IEND", b""]), 0xAE42_6082);
    }

    #[test]
    fn a_square_png_is_accepted() {
        let data = png(1400, 1400, 1);
        let artwork = validate_artwork(&data).unwrap();
        assert_eq!((artwork.extension, artwork.media_type), (".png", "image/png"));
        assert_eq!(artwork.version.len(), 12);
        assert!(artwork.version.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn a_square_jpeg_at_the_top_of_the_range_is_accepted() {
        let artwork = validate_artwork(&jpeg(3000, 3000, 1)).unwrap();
        assert_eq!((artwork.extension, artwork.media_type), (".jpg", "image/jpeg"));
    }

    #[test]
    fn both_ends_of_the_size_range_are_accepted() {
        for size in [1400, 3000] {
            validate_artwork(&png(size, size, 0)).unwrap();
            validate_artwork(&jpeg(size as u16, size as u16, 0)).unwrap();
        }
    }

    #[test]
    fn one_pixel_outside_the_range_is_refused() {
        assert_eq!(
            refusal(&png(1399, 1399, 0)),
            "The artwork must be at least 1400x1400 pixels (this one is 1399x1399)."
        );
        assert_eq!(
            refusal(&png(3001, 3001, 0)),
            "The artwork must be at most 3000x3000 pixels (this one is 3001x3001)."
        );
        assert!(refusal(&jpeg(1399, 1399, 0)).contains("at least 1400"));
        assert!(refusal(&jpeg(3001, 3001, 0)).contains("at most 3000"));
    }

    #[test]
    fn the_version_follows_the_content() {
        let navy = validate_artwork(&png(1400, 1400, 1)).unwrap();
        let red = validate_artwork(&png(1400, 1400, 2)).unwrap();
        assert_ne!(navy.version, red.version);
        assert_eq!(navy.version, validate_artwork(&png(1400, 1400, 1)).unwrap().version);
    }

    #[test]
    fn every_refusal_has_its_own_message() {
        let mut over_10mb = PNG_SIGNATURE.to_vec();
        over_10mb.resize(8 + MAX_BYTES + 1, b'0');
        let mut gif = b"GIF89a".to_vec();
        gif.extend([0x78, 0x05, 0x78, 0x05, 0, 0, 0]);

        let cases: [(&str, Vec<u8>, &str); 8] = [
            ("text_file", b"just some text, not an image".to_vec(), "not a png or jpeg"),
            ("gif", gif, "got gif"),
            ("not_square", png(1400, 1500, 0), "square"),
            ("too_small", png(800, 800, 0), "at least 1400"),
            ("too_large", png(3200, 3200, 0), "at most 3000"),
            ("empty", Vec::new(), "empty"),
            ("over_10mb", over_10mb, "10 mb"),
            ("truncated", png(1400, 1400, 0)[..200].to_vec(), "could not read"),
        ];
        let mut messages = Vec::new();
        for (name, data, keyword) in &cases {
            let message = refusal(data);
            assert!(message.to_lowercase().contains(keyword), "{name}: {message}");
            messages.push(message);
        }
        messages.sort();
        messages.dedup();
        assert_eq!(messages.len(), cases.len());
    }

    #[test]
    fn refusal_messages_match_the_original() {
        assert_eq!(refusal(b""), "The artwork file is empty.");
        assert_eq!(refusal(b"hello"), "The artwork is not a PNG or JPEG image.");
        assert_eq!(
            refusal(&png(1400, 1500, 0)),
            "The artwork must be square (this one is 1400x1500 pixels)."
        );
        assert_eq!(refusal(b"RIFF\x10\0\0\0WEBPVP8 "), "The artwork must be a PNG or JPEG image (got WEBP).");
        assert_eq!(refusal(b"II*\0\x08\0\0\0"), "The artwork must be a PNG or JPEG image (got TIFF).");
    }

    #[test]
    fn a_huge_image_header_is_refused_before_anything_else() {
        // 15000 x 15000: a tiny file claiming 225 million pixels.
        assert_eq!(refusal(&png(15_000, 15_000, 0)), "The artwork must be at most 3000x3000 pixels.");
        // Not square either, but the pixel count is what gets it refused.
        assert_eq!(refusal(&png(60_000, 4_000, 0)), "The artwork must be at most 3000x3000 pixels.");
    }

    #[test]
    fn png_damage_is_detected_by_checksum() {
        let good = png(1400, 1400, 0);

        let mut flipped = good.clone();
        let idat = good.windows(4).position(|w| w == b"IDAT").unwrap();
        flipped[idat + 10] ^= 0xFF;
        assert_eq!(refusal(&flipped), "Could not read the artwork; the image file looks damaged.");

        // Cut off inside the image data, and right before the end marker.
        assert!(refusal(&good[..idat + 50]).contains("looks damaged"));
        assert!(refusal(&good[..good.len() - 12]).contains("looks damaged"));

        // A damaged header chunk means the size cannot be trusted at all.
        let mut bad_header = good.clone();
        bad_header[18] ^= 0x01;
        assert_eq!(refusal(&bad_header), "The artwork is not a PNG or JPEG image.");
        assert_eq!(refusal(&good[..20]), "The artwork is not a PNG or JPEG image.");
    }

    #[test]
    fn size_problems_are_reported_before_damage() {
        let small_and_cut = &png(800, 800, 0)[..200];
        assert!(refusal(small_and_cut).contains("at least 1400"));
    }

    #[test]
    fn jpeg_markers_are_walked_to_the_image_data() {
        let good = jpeg(1500, 1500, 3);
        validate_artwork(&good).unwrap();

        // Progressive frames and fill bytes before a marker are fine.
        let mut progressive = good.clone();
        let sof = good.windows(2).position(|w| w == [0xFF, 0xC0]).unwrap();
        progressive[sof + 1] = 0xC2;
        progressive.insert(sof, 0xFF);
        validate_artwork(&progressive).unwrap();

        // The file ends before any image data.
        let sos = good.windows(2).position(|w| w == [0xFF, 0xDA]).unwrap();
        assert_eq!(refusal(&good[..sos]), "The artwork is not a PNG or JPEG image.");
        // A segment is cut off.
        assert!(refusal(&good[..sof + 8]).contains("looks damaged"));
        // Only the start-of-image marker.
        assert_eq!(refusal(b"\xFF\xD8\xFF"), "The artwork is not a PNG or JPEG image.");
    }

    #[test]
    fn jpeg_is_not_square() {
        assert_eq!(
            refusal(&jpeg(1600, 1400, 0)),
            "The artwork must be square (this one is 1600x1400 pixels)."
        );
    }

    #[test]
    fn a_real_png_is_read_correctly() {
        // The Python repository's logo, when it is checked out next to the
        // workspace: a real encoder's output through the chunk walker.
        let logo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../open-notebook/logo.png");
        let Ok(data) = std::fs::read(logo) else {
            return;
        };
        assert!(png_size(&data).is_some());
        assert!(png_is_intact(&data));
        assert!(!png_is_intact(&data[..data.len() / 2]));
    }

    #[test]
    fn saving_replaces_the_other_format_and_overwrites_in_place() {
        let root = tempfile::tempdir().unwrap();
        let feed = artwork_dir(root.path());
        assert!(feed.ends_with("feed"));
        assert_eq!(current_artwork(root.path(), Some(".png")), None);

        let first = png(1400, 1400, 1);
        let artwork = validate_artwork(&first).unwrap();
        save_artwork(root.path(), &first, &artwork).unwrap();
        assert_eq!(std::fs::read(feed.join("artwork.png")).unwrap(), first);
        assert_eq!(
            current_artwork(root.path(), Some(".png")),
            Some((feed.join("artwork.png"), "image/png"))
        );

        let second = jpeg(1600, 1600, 2);
        let replaced = validate_artwork(&second).unwrap();
        save_artwork(root.path(), &second, &replaced).unwrap();
        assert!(!feed.join("artwork.png").exists());
        assert_eq!(std::fs::read(feed.join("artwork.jpg")).unwrap(), second);
        assert_ne!(artwork.version, replaced.version);
        assert_eq!(current_artwork(root.path(), Some(".png")), None);
        assert_eq!(current_artwork(root.path(), Some(".jpg")).unwrap().1, "image/jpeg");

        let third = jpeg(1500, 1500, 3);
        save_artwork(root.path(), &third, &validate_artwork(&third).unwrap()).unwrap();
        assert_eq!(std::fs::read(feed.join("artwork.jpg")).unwrap(), third);
        let names: Vec<_> =
            std::fs::read_dir(&feed).unwrap().map(|entry| entry.unwrap().file_name()).collect();
        assert_eq!(names, ["artwork.jpg"]);
    }

    #[test]
    fn only_known_extensions_can_name_the_artwork() {
        let root = tempfile::tempdir().unwrap();
        let data = png(1400, 1400, 1);
        save_artwork(root.path(), &data, &validate_artwork(&data).unwrap()).unwrap();
        for bad in [None, Some(""), Some(".gif"), Some("png"), Some("/../../etc/passwd"), Some(".png/../x")] {
            assert_eq!(current_artwork(root.path(), bad), None, "{bad:?}");
        }
        // A record whose file vanished counts as no artwork.
        std::fs::remove_file(artwork_path(root.path(), ".png")).unwrap();
        assert_eq!(current_artwork(root.path(), Some(".png")), None);
    }
}
