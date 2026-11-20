//! File type detection: magic bytes first, extension second, declared MIME
//! type last.

use std::io::{Read, Seek};

/// What a file is, as far as extraction is concerned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum FileKind {
    Pdf,
    Docx,
    Pptx,
    Odt,
    Odp,
    Xlsx,
    Xlsb,
    Xls,
    Ods,
    Csv,
    Tsv,
    Epub,
    Html,
    Markdown,
    Text,
    Audio,
    Video,
    Unsupported { reason: String },
}

impl FileKind {
    /// The value reported as `Extracted::identified_type`.
    pub fn identified_type(&self) -> &'static str {
        match self {
            FileKind::Pdf => "pdf",
            FileKind::Docx => "docx",
            FileKind::Pptx => "pptx",
            FileKind::Odt => "odt",
            FileKind::Odp => "odp",
            FileKind::Xlsx => "xlsx",
            FileKind::Xlsb => "xlsb",
            FileKind::Xls => "xls",
            FileKind::Ods => "ods",
            FileKind::Csv | FileKind::Tsv => "csv",
            FileKind::Epub => "epub",
            FileKind::Html => "html",
            FileKind::Markdown => "markdown",
            FileKind::Text => "text",
            FileKind::Audio => "audio",
            FileKind::Video => "video",
            FileKind::Unsupported { .. } => "unsupported",
        }
    }

    pub fn unsupported_reason(&self) -> Option<String> {
        match self {
            FileKind::Unsupported { reason } => Some(reason.clone()),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Detected {
    pub kind: FileKind,
    pub mime: String,
    /// Lower-case extension without the dot, when the name has one.
    pub extension: Option<String>,
}

/// What a ZIP container holds, judged by its entries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ZipFlavor {
    Docx,
    Pptx,
    Xlsx,
    Xlsb,
    Odt,
    Ods,
    Odp,
    Epub,
    Other,
}

/// Lower-case extension of `name`, without the dot.
pub(crate) fn extension_of(name: &str) -> Option<String> {
    let file = name.rsplit(['/', '\\']).next()?;
    let (stem, extension) = file.rsplit_once('.')?;
    if stem.is_empty() || extension.is_empty() || extension.len() > 12 {
        return None;
    }
    extension.chars().all(|c| c.is_ascii_alphanumeric()).then(|| extension.to_ascii_lowercase())
}

const AUDIO_EXTENSIONS: &[&str] =
    &["mp3", "wav", "m4a", "m4b", "aac", "ogg", "oga", "opus", "flac", "wma", "aiff", "aif", "amr", "mka", "weba", "mpga"];
const VIDEO_EXTENSIONS: &[&str] =
    &["mp4", "m4v", "mov", "mkv", "webm", "avi", "wmv", "flv", "mpg", "mpeg", "ts", "m2ts", "3gp", "ogv"];
const IMAGE_EXTENSIONS: &[&str] =
    &["png", "jpg", "jpeg", "gif", "bmp", "tif", "tiff", "webp", "heic", "heif", "avif", "ico"];

const IMAGE_REASON: &str = "images need OCR, which is not available";
const LEGACY_OFFICE_REASON: &str =
    "legacy binary Office files are not supported; save the file as docx, pptx or pdf";

/// Classify a ZIP archive by the parts it contains.
pub(crate) fn zip_flavor<R: Read + Seek>(reader: R) -> Option<ZipFlavor> {
    let mut archive = zip::ZipArchive::new(reader).ok()?;
    if let Ok(mut entry) = archive.by_name("mimetype") {
        let mut declared = String::new();
        if entry.by_ref().take(256).read_to_string(&mut declared).is_ok() {
            match declared.trim() {
                "application/epub+zip" => return Some(ZipFlavor::Epub),
                "application/vnd.oasis.opendocument.text" => return Some(ZipFlavor::Odt),
                "application/vnd.oasis.opendocument.spreadsheet" => return Some(ZipFlavor::Ods),
                "application/vnd.oasis.opendocument.presentation" => return Some(ZipFlavor::Odp),
                _ => {}
            }
        }
    }
    let has = |name: &str| archive.index_for_name(name).is_some();
    Some(if has("word/document.xml") {
        ZipFlavor::Docx
    } else if has("ppt/presentation.xml") {
        ZipFlavor::Pptx
    } else if has("xl/workbook.xml") {
        ZipFlavor::Xlsx
    } else if has("xl/workbook.bin") {
        ZipFlavor::Xlsb
    } else if has("META-INF/container.xml") {
        ZipFlavor::Epub
    } else {
        ZipFlavor::Other
    })
}

fn supported(kind: FileKind, mime: &str, extension: Option<&str>) -> Detected {
    Detected { kind, mime: mime.to_string(), extension: extension.map(str::to_string) }
}

fn unsupported(mime: &str, reason: &str, extension: Option<&str>) -> Detected {
    Detected {
        kind: FileKind::Unsupported { reason: reason.to_string() },
        mime: mime.to_string(),
        extension: extension.map(str::to_string),
    }
}

/// Detect the type of a file from its first bytes.
///
/// `head` is the start of the file (a few kilobytes are enough). `zip` is
/// called only when the file is a ZIP container, to look inside it.
pub(crate) fn detect(
    head: &[u8],
    zip: impl FnOnce() -> Option<ZipFlavor>,
    extension: Option<&str>,
    declared_mime: Option<&str>,
) -> Detected {
    let ext = extension;
    let mime = declared_mime.map(|m| m.split(';').next().unwrap_or("").trim().to_ascii_lowercase()).unwrap_or_default();
    let ext_is = |candidates: &[&str]| ext.is_some_and(|e| candidates.contains(&e));

    if head.windows(5).take(1024).any(|w| w == b"%PDF-") {
        return supported(FileKind::Pdf, "application/pdf", ext);
    }

    if head.starts_with(b"PK\x03\x04") || head.starts_with(b"PK\x05\x06") {
        let flavor = zip();
        return match (flavor, ext) {
            (Some(ZipFlavor::Docx), _) | (None, Some("docx")) => supported(
                FileKind::Docx,
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                ext,
            ),
            (Some(ZipFlavor::Pptx), _) | (None, Some("pptx")) => supported(
                FileKind::Pptx,
                "application/vnd.openxmlformats-officedocument.presentationml.presentation",
                ext,
            ),
            (Some(ZipFlavor::Xlsx), _) | (None, Some("xlsx" | "xlsm")) => {
                supported(FileKind::Xlsx, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", ext)
            }
            (Some(ZipFlavor::Xlsb), _) | (None, Some("xlsb")) => {
                supported(FileKind::Xlsb, "application/vnd.ms-excel.sheet.binary.macroenabled.12", ext)
            }
            (Some(ZipFlavor::Odt), _) | (None, Some("odt")) => {
                supported(FileKind::Odt, "application/vnd.oasis.opendocument.text", ext)
            }
            (Some(ZipFlavor::Ods), _) | (None, Some("ods")) => {
                supported(FileKind::Ods, "application/vnd.oasis.opendocument.spreadsheet", ext)
            }
            (Some(ZipFlavor::Odp), _) | (None, Some("odp")) => {
                supported(FileKind::Odp, "application/vnd.oasis.opendocument.presentation", ext)
            }
            (Some(ZipFlavor::Epub), _) | (None, Some("epub")) => supported(FileKind::Epub, "application/epub+zip", ext),
            _ => unsupported("application/zip", "archives are not supported; add the files inside one by one", ext),
        };
    }

    if head.starts_with(&[0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1]) {
        if ext_is(&["xls", "xlt"]) || (ext.is_none() && mime == "application/vnd.ms-excel") {
            return supported(FileKind::Xls, "application/vnd.ms-excel", ext);
        }
        let legacy_mime = match ext {
            Some("doc" | "dot") => "application/msword",
            Some("ppt" | "pps" | "pot") => "application/vnd.ms-powerpoint",
            _ => "application/x-ole-storage",
        };
        return unsupported(legacy_mime, LEGACY_OFFICE_REASON, ext);
    }

    // Images.
    if head.starts_with(b"\x89PNG\r\n\x1a\n") {
        return unsupported("image/png", IMAGE_REASON, ext);
    }
    if head.starts_with(&[0xFF, 0xD8, 0xFF]) {
        return unsupported("image/jpeg", IMAGE_REASON, ext);
    }
    if head.starts_with(b"GIF87a") || head.starts_with(b"GIF89a") {
        return unsupported("image/gif", IMAGE_REASON, ext);
    }
    if head.starts_with(b"II*\x00") || head.starts_with(b"MM\x00*") {
        return unsupported("image/tiff", IMAGE_REASON, ext);
    }
    if head.starts_with(b"BM") && ext_is(&["bmp"]) {
        return unsupported("image/bmp", IMAGE_REASON, ext);
    }

    // RIFF containers.
    if head.len() >= 12 && head.starts_with(b"RIFF") {
        match &head[8..12] {
            b"WAVE" => return supported(FileKind::Audio, "audio/wav", ext),
            b"AVI " => return supported(FileKind::Video, "video/x-msvideo", ext),
            b"WEBP" => return unsupported("image/webp", IMAGE_REASON, ext),
            _ => {}
        }
    }
    if head.len() >= 12 && head.starts_with(b"FORM") && matches!(&head[8..12], b"AIFF" | b"AIFC") {
        return supported(FileKind::Audio, "audio/aiff", ext);
    }
    if head.starts_with(b"ID3") {
        return supported(FileKind::Audio, "audio/mpeg", ext);
    }
    if head.starts_with(b"fLaC") {
        return supported(FileKind::Audio, "audio/flac", ext);
    }
    if head.starts_with(b"#!AMR") {
        return supported(FileKind::Audio, "audio/amr", ext);
    }
    if head.starts_with(b"OggS") {
        return if ext_is(&["ogv"]) || mime.starts_with("video/") {
            supported(FileKind::Video, "video/ogg", ext)
        } else {
            supported(FileKind::Audio, "audio/ogg", ext)
        };
    }
    if head.starts_with(b"FLV\x01") {
        return supported(FileKind::Video, "video/x-flv", ext);
    }
    if head.starts_with(&[0x00, 0x00, 0x01, 0xBA]) || head.starts_with(&[0x00, 0x00, 0x01, 0xB3]) {
        return supported(FileKind::Video, "video/mpeg", ext);
    }
    if head.starts_with(&[0x30, 0x26, 0xB2, 0x75, 0x8E, 0x66, 0xCF, 0x11]) {
        return if ext_is(&["wma"]) || mime.starts_with("audio/") {
            supported(FileKind::Audio, "audio/x-ms-wma", ext)
        } else {
            supported(FileKind::Video, "video/x-ms-wmv", ext)
        };
    }
    if head.starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        return if ext_is(&["mka", "weba"]) || mime.starts_with("audio/") {
            supported(FileKind::Audio, "audio/webm", ext)
        } else if ext_is(&["mkv"]) {
            supported(FileKind::Video, "video/x-matroska", ext)
        } else {
            supported(FileKind::Video, "video/webm", ext)
        };
    }
    if head.len() >= 12 && &head[4..8] == b"ftyp" {
        let brand = &head[8..12];
        if matches!(brand, b"heic" | b"heix" | b"hevc" | b"mif1" | b"msf1" | b"avif" | b"avis") {
            return unsupported("image/heic", IMAGE_REASON, ext);
        }
        if matches!(brand, b"M4A " | b"M4B " | b"M4P ") || ext_is(AUDIO_EXTENSIONS) || mime.starts_with("audio/") {
            return supported(FileKind::Audio, "audio/mp4", ext);
        }
        if brand == b"qt  " {
            return supported(FileKind::Video, "video/quicktime", ext);
        }
        return supported(FileKind::Video, "video/mp4", ext);
    }

    if head.starts_with(b"{\\rtf") {
        return unsupported("application/rtf", "RTF is not supported; save the file as docx or pdf", ext);
    }

    if looks_like_text(head) {
        return detect_text(head, ext, &mime);
    }

    // Binary data without a recognised signature: the extension and the
    // declared type are all that is left. MPEG audio without an ID3 tag and
    // raw AAC land here.
    if ext_is(AUDIO_EXTENSIONS) || (ext.is_none() && mime.starts_with("audio/")) {
        return supported(FileKind::Audio, &audio_mime(ext), ext);
    }
    if ext_is(VIDEO_EXTENSIONS) || (ext.is_none() && mime.starts_with("video/")) {
        return supported(FileKind::Video, "video/mp4", ext);
    }
    if ext_is(IMAGE_EXTENSIONS) || mime.starts_with("image/") {
        return unsupported(&format!("image/{}", ext.unwrap_or("unknown")), IMAGE_REASON, ext);
    }
    if ext_is(&["doc", "ppt"]) {
        return unsupported("application/msword", LEGACY_OFFICE_REASON, ext);
    }
    let reported = if mime.is_empty() { "application/octet-stream".to_string() } else { mime };
    unsupported(&reported, "the file is binary data of an unrecognised type", ext)
}

fn audio_mime(extension: Option<&str>) -> String {
    match extension {
        Some("mp3" | "mpga") | None => "audio/mpeg".to_string(),
        Some("aac") => "audio/aac".to_string(),
        Some(other) => format!("audio/{other}"),
    }
}

/// Whether `head` is plausibly text in some encoding.
fn looks_like_text(head: &[u8]) -> bool {
    if head.is_empty() {
        return true;
    }
    if head.starts_with(&[0xFF, 0xFE]) || head.starts_with(&[0xFE, 0xFF]) {
        // UTF-16 with a byte-order mark; every other byte is typically NUL.
        return true;
    }
    if head.contains(&0) {
        return false;
    }
    let control =
        head.iter().filter(|&&b| b < 0x09 || ((0x0E..0x20).contains(&b) && b != 0x1B) || b == 0x7F).count();
    if control * 100 > head.len() {
        return false;
    }
    match std::str::from_utf8(head) {
        Ok(_) => true,
        // A multi-byte character cut off by the end of the sample is fine.
        Err(e) if e.error_len().is_none() => true,
        // Not UTF-8: accept single-byte encodings (Latin-1, Windows-1252)
        // when the high bytes are a minority, as they are in real prose.
        Err(_) => head.iter().filter(|&&b| b >= 0x80).count() * 100 < head.len() * 30,
    }
}

fn detect_text(head: &[u8], ext: Option<&str>, mime: &str) -> Detected {
    match ext {
        Some("html" | "htm" | "xhtml") => return supported(FileKind::Html, "text/html", ext),
        Some("md" | "markdown" | "mdown" | "mkd") => return supported(FileKind::Markdown, "text/markdown", ext),
        Some("csv") => return supported(FileKind::Csv, "text/csv", ext),
        Some("tsv" | "tab") => return supported(FileKind::Tsv, "text/tab-separated-values", ext),
        _ => {}
    }
    let sample = String::from_utf8_lossy(&head[..head.len().min(2048)]).to_ascii_lowercase();
    let sample = sample.trim_start_matches('\u{feff}').trim_start();
    if sample.starts_with("<!doctype html") || sample.starts_with("<html") || mime == "text/html" {
        return supported(FileKind::Html, "text/html", ext);
    }
    if ext.is_none() {
        match mime {
            "text/csv" => return supported(FileKind::Csv, "text/csv", ext),
            "text/tab-separated-values" => return supported(FileKind::Tsv, "text/tab-separated-values", ext),
            "text/markdown" | "text/x-markdown" => return supported(FileKind::Markdown, "text/markdown", ext),
            _ => {}
        }
    }
    let text_mime = match ext {
        Some("json") => "application/json",
        Some("yaml" | "yml") => "application/yaml",
        Some("xml" | "svg") => "application/xml",
        _ => "text/plain",
    };
    supported(FileKind::Text, text_mime, ext)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind(head: &[u8], ext: Option<&str>, mime: Option<&str>) -> FileKind {
        detect(head, || None, ext, mime).kind
    }

    fn is_unsupported(head: &[u8], ext: Option<&str>) -> bool {
        matches!(kind(head, ext, None), FileKind::Unsupported { .. })
    }

    #[test]
    fn extension_parsing() {
        assert_eq!(extension_of("report.PDF").as_deref(), Some("pdf"));
        assert_eq!(extension_of("/a/b.c/archive.tar.gz").as_deref(), Some("gz"));
        assert_eq!(extension_of("C:\\docs\\x.docx").as_deref(), Some("docx"));
        assert_eq!(extension_of(".bashrc"), None);
        assert_eq!(extension_of("Makefile"), None);
        assert_eq!(extension_of("trailing."), None);
        assert_eq!(extension_of("weird.na me"), None);
    }

    #[test]
    fn magic_bytes_win_over_extension() {
        assert_eq!(kind(b"%PDF-1.7\n...", Some("txt"), None), FileKind::Pdf);
        assert_eq!(kind(b"\n\n%PDF-1.4", None, None), FileKind::Pdf);
        assert!(is_unsupported(b"\x89PNG\r\n\x1a\n....", Some("pdf")));
        assert_eq!(kind(b"ID3\x04\x00....", Some("txt"), None), FileKind::Audio);
        assert_eq!(kind(b"RIFF\x24\x00\x00\x00WAVEfmt ", None, None), FileKind::Audio);
        assert_eq!(kind(b"RIFF\x24\x00\x00\x00AVI LIST", None, None), FileKind::Video);
        assert!(is_unsupported(b"RIFF\x24\x00\x00\x00WEBPVP8 ", None));
        assert_eq!(kind(b"fLaC\x00\x00\x00\x22", None, None), FileKind::Audio);
        assert_eq!(kind(b"OggS\x00\x02", Some("ogg"), None), FileKind::Audio);
        assert_eq!(kind(b"OggS\x00\x02", Some("ogv"), None), FileKind::Video);
    }

    #[test]
    fn mp4_family_and_matroska() {
        assert_eq!(kind(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00", Some("mp4"), None), FileKind::Video);
        assert_eq!(kind(b"\x00\x00\x00\x20ftypM4A \x00\x00\x02\x00", None, None), FileKind::Audio);
        assert_eq!(kind(b"\x00\x00\x00\x20ftypisom\x00\x00\x02\x00", Some("m4a"), None), FileKind::Audio);
        assert_eq!(kind(b"\x00\x00\x00\x14ftypqt  \x00\x00\x02\x00", None, None), FileKind::Video);
        assert!(is_unsupported(b"\x00\x00\x00\x18ftypheic\x00\x00\x00\x00", None));
        assert_eq!(kind(b"\x1A\x45\xDF\xA3\x01\x00", Some("webm"), None), FileKind::Video);
        assert_eq!(kind(b"\x1A\x45\xDF\xA3\x01\x00", Some("webm"), Some("audio/webm")), FileKind::Audio);
        assert_eq!(kind(b"\x1A\x45\xDF\xA3\x01\x00", Some("mka"), None), FileKind::Audio);
    }

    #[test]
    fn zip_containers_are_classified_by_content() {
        let pk = b"PK\x03\x04\x14\x00";
        let with = |flavor: Option<ZipFlavor>, ext: Option<&str>| detect(pk, || flavor, ext, None).kind;
        assert_eq!(with(Some(ZipFlavor::Docx), Some("zip")), FileKind::Docx);
        assert_eq!(with(Some(ZipFlavor::Pptx), None), FileKind::Pptx);
        assert_eq!(with(Some(ZipFlavor::Xlsx), None), FileKind::Xlsx);
        assert_eq!(with(Some(ZipFlavor::Ods), None), FileKind::Ods);
        assert_eq!(with(Some(ZipFlavor::Odt), None), FileKind::Odt);
        assert_eq!(with(Some(ZipFlavor::Epub), None), FileKind::Epub);
        // A corrupt archive with a document extension is still routed to the
        // document extractor, which reports it as unreadable.
        assert_eq!(with(None, Some("docx")), FileKind::Docx);
        assert!(matches!(with(Some(ZipFlavor::Other), Some("zip")), FileKind::Unsupported { .. }));
        assert!(matches!(with(Some(ZipFlavor::Other), Some("docx")), FileKind::Unsupported { .. }));
    }

    #[test]
    fn ole_files() {
        let ole = [0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1, 0, 0];
        assert_eq!(kind(&ole, Some("xls"), None), FileKind::Xls);
        assert_eq!(kind(&ole, None, Some("application/vnd.ms-excel")), FileKind::Xls);
        let doc = detect(&ole, || None, Some("doc"), None);
        assert_eq!(doc.mime, "application/msword");
        assert!(doc.kind.unsupported_reason().unwrap().contains("legacy"));
        assert!(is_unsupported(&ole, Some("ppt")));
    }

    #[test]
    fn text_subtypes() {
        assert_eq!(kind(b"# Title\n\ntext", Some("md"), None), FileKind::Markdown);
        assert_eq!(kind(b"a,b\n1,2\n", Some("csv"), None), FileKind::Csv);
        assert_eq!(kind(b"a\tb\n1\t2\n", Some("tsv"), None), FileKind::Tsv);
        assert_eq!(kind(b"<!DOCTYPE html><html>", None, None), FileKind::Html);
        assert_eq!(kind(b"\xEF\xBB\xBF  <HTML><body>", Some("txt"), None), FileKind::Html);
        assert_eq!(kind(b"<p>fragment</p>", Some("htm"), None), FileKind::Html);
        assert_eq!(kind(b"{\"a\": 1}", Some("json"), None), FileKind::Text);
        assert_eq!(kind(b"fn main() {}", Some("rs"), None), FileKind::Text);
        assert_eq!(kind(b"plain", None, None), FileKind::Text);
        assert_eq!(kind(b"a,b\n", None, Some("text/csv; charset=utf-8")), FileKind::Csv);
        assert_eq!(kind("caf\u{e9} na\u{ef}ve".as_bytes(), Some("txt"), None), FileKind::Text);
        // Latin-1 bytes that are not valid UTF-8.
        assert_eq!(kind(b"caf\xe9 au lait, s'il vous pla\xeet", Some("txt"), None), FileKind::Text);
        // UTF-16 with BOM.
        assert_eq!(kind(b"\xFF\xFEh\x00i\x00", Some("txt"), None), FileKind::Text);
        assert_eq!(detect(b"{}", || None, Some("json"), None).mime, "application/json");
        assert!(is_unsupported(b"{\\rtf1\\ansi", Some("rtf")));
    }

    #[test]
    fn unknown_binary_falls_back_to_extension_then_fails() {
        let binary = [0x00, 0x01, 0x02, 0xFF, 0xFB, 0x90, 0x00, 0x00];
        assert_eq!(kind(&binary, Some("mp3"), None), FileKind::Audio);
        assert_eq!(kind(&binary, Some("mpg"), None), FileKind::Video);
        assert_eq!(kind(&binary, None, Some("audio/mpeg")), FileKind::Audio);
        let image = detect(&binary, || None, Some("ico"), None);
        assert_eq!(image.mime, "image/ico");
        let unknown = detect(&binary, || None, Some("bin"), None);
        assert_eq!(unknown.mime, "application/octet-stream");
        assert!(unknown.kind.unsupported_reason().unwrap().contains("unrecognised"));
        assert_eq!(unknown.kind.identified_type(), "unsupported");
    }
}
