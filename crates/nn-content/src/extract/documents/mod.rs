//! Document parsers. Everything here is synchronous and works on bytes in
//! memory; `files.rs` runs it on a blocking thread and catches panics.

pub(crate) mod epub;
pub(crate) mod odf;
pub(crate) mod office;
pub(crate) mod pdf;
pub(crate) mod sheets;

use std::io::{Cursor, Read};

use serde_json::{json, Map, Value};

use super::detect::FileKind;
use super::html::{html_to_markdown, HtmlOptions};
use super::xml::{self, XmlNode};
use crate::error::ContentError;

/// Largest decompressed size accepted for one part of a ZIP-based document.
const MAX_PART_BYTES: u64 = 64 * 1024 * 1024;
/// Largest decompressed size read from one ZIP-based document, all parts
/// together. A part read twice counts twice.
pub(crate) const MAX_DOCUMENT_BYTES: u64 = 256 * 1024 * 1024;

/// Result of parsing one document.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct DocOutput {
    pub title: Option<String>,
    pub content: String,
    pub metadata: Map<String, Value>,
}

impl DocOutput {
    fn set(&mut self, key: &str, value: impl Into<Value>) {
        self.metadata.insert(key.to_string(), value.into());
    }

    fn set_opt(&mut self, key: &str, value: Option<String>) {
        if let Some(value) = value.filter(|v| !v.trim().is_empty()) {
            self.metadata.insert(key.to_string(), json!(value.trim()));
        }
    }
}

/// Parse `bytes` as a document of `kind`. Audio, video and unsupported kinds
/// are not documents and must not reach this function.
pub(crate) fn extract_document(kind: &FileKind, bytes: &[u8]) -> Result<DocOutput, ContentError> {
    match kind {
        FileKind::Pdf => pdf::extract(bytes),
        FileKind::Docx => office::extract_docx(bytes),
        FileKind::Pptx => office::extract_pptx(bytes),
        FileKind::Odt | FileKind::Odp => odf::extract(bytes),
        FileKind::Xlsx | FileKind::Xlsb | FileKind::Xls | FileKind::Ods => sheets::extract_workbook(kind, bytes),
        FileKind::Csv => sheets::extract_delimited(bytes, None),
        FileKind::Tsv => sheets::extract_delimited(bytes, Some(b'\t')),
        FileKind::Epub => epub::extract(bytes),
        FileKind::Html => {
            let page = html_to_markdown(&decode_text(bytes), &HtmlOptions::default());
            let mut out = DocOutput { title: page.title, content: page.markdown, ..DocOutput::default() };
            out.set_opt("description", page.description);
            out.set_opt("author", page.author);
            out.set_opt("language", page.language);
            Ok(out)
        }
        FileKind::Markdown => {
            let content = decode_text(bytes);
            let title = content
                .lines()
                .find(|line| !line.trim().is_empty())
                .and_then(|line| line.trim().strip_prefix("# "))
                .map(|t| t.trim().trim_end_matches('#').trim().to_string())
                .filter(|t| !t.is_empty());
            Ok(DocOutput { title, content, ..DocOutput::default() })
        }
        FileKind::Text => Ok(DocOutput { content: decode_text(bytes), ..DocOutput::default() }),
        FileKind::Audio | FileKind::Video | FileKind::Unsupported { .. } => Err(ContentError::UnsupportedType {
            identified_type: kind.identified_type().to_string(),
            reason: "not a document".to_string(),
        }),
    }
}

/// Decode a text file: a byte-order mark decides (UTF-8, UTF-16); otherwise
/// UTF-8 when valid, else Windows-1252.
pub(crate) fn decode_text(bytes: &[u8]) -> String {
    if let Some((encoding, bom_length)) = encoding_rs::Encoding::for_bom(bytes) {
        return encoding.decode_without_bom_handling(&bytes[bom_length..]).0.into_owned();
    }
    match std::str::from_utf8(bytes) {
        Ok(text) => text.to_string(),
        Err(_) => encoding_rs::WINDOWS_1252.decode_without_bom_handling(bytes).0.into_owned(),
    }
}

/// Decode a body using a charset label (from HTTP or a `<meta>` tag), falling
/// back to [`decode_text`] when the label is missing or unknown.
pub(crate) fn decode_with_charset(bytes: &[u8], charset: Option<&str>) -> String {
    match charset.and_then(|label| encoding_rs::Encoding::for_label(label.trim().as_bytes())) {
        Some(encoding) => encoding.decode(bytes).0.into_owned(),
        None => decode_text(bytes),
    }
}

/// A ZIP-based document package (OOXML, OpenDocument, EPUB).
struct Package<'a> {
    archive: zip::ZipArchive<Cursor<&'a [u8]>>,
    /// Decompressed bytes this document may still read ([`MAX_DOCUMENT_BYTES`] in all).
    remaining: u64,
}

impl<'a> Package<'a> {
    fn open(bytes: &'a [u8]) -> Result<Self, ContentError> {
        zip::ZipArchive::new(Cursor::new(bytes))
            .map(|archive| Self { archive, remaining: MAX_DOCUMENT_BYTES })
            .map_err(|e| ContentError::FileOperation(format!("not a readable ZIP container: {e}")))
    }

    #[cfg(test)]
    fn with_budget(bytes: &'a [u8], budget: u64) -> Result<Self, ContentError> {
        Self::open(bytes).map(|package| Self { remaining: budget, ..package })
    }

    fn names(&self) -> Vec<String> {
        self.archive.file_names().filter_map(Result::ok).map(|name| name.into_owned()).collect()
    }

    /// Bytes of the part called `name`, or `None` when the package has no such part.
    fn read(&mut self, name: &str) -> Result<Option<Vec<u8>>, ContentError> {
        let Some(index) = self.archive.index_for_name(name) else {
            return Ok(None);
        };
        let entry = self
            .archive
            .by_index(index)
            .map_err(|e| ContentError::FileOperation(format!("cannot read '{name}' from the document: {e}")))?;
        let limit = MAX_PART_BYTES.min(self.remaining);
        let too_large = || {
            ContentError::FileOperation(if limit < MAX_PART_BYTES {
                format!("the document expands to more than {} MiB", MAX_DOCUMENT_BYTES >> 20)
            } else {
                format!("'{name}' in the document is too large")
            })
        };
        // The declared size is checked first (the reader enforces it too),
        // and the read is capped in case it lies.
        if entry.size() > limit {
            return Err(too_large());
        }
        let mut data = Vec::new();
        entry
            .take(limit + 1)
            .read_to_end(&mut data)
            .map_err(|e| ContentError::FileOperation(format!("cannot read '{name}' from the document: {e}")))?;
        if data.len() as u64 > limit {
            return Err(too_large());
        }
        self.remaining -= data.len() as u64;
        Ok(Some(data))
    }

    fn read_text(&mut self, name: &str) -> Result<Option<String>, ContentError> {
        Ok(self.read(name)?.map(|bytes| xml::decode(&bytes)))
    }

    /// The part parsed as XML, or `None` when missing. A part that is present
    /// but not well-formed is an error.
    fn read_xml(&mut self, name: &str) -> Result<Option<XmlNode>, ContentError> {
        match self.read_text(name)? {
            None => Ok(None),
            Some(text) => xml::parse(&text)
                .map(Some)
                .map_err(|e| ContentError::FileOperation(format!("'{name}' in the document is not valid XML: {e}"))),
        }
    }

    /// Like [`read_xml`](Self::read_xml) but a missing or broken part is `None`.
    /// For optional parts (properties, numbering, notes).
    fn read_xml_lenient(&mut self, name: &str) -> Option<XmlNode> {
        self.read_xml(name).ok().flatten()
    }
}

/// Check a ZIP-based workbook before a third-party parser opens it: the
/// declared decompressed size of its XML and binary parts (not pictures)
/// must fit the per-document budget. The ZIP reader refuses to inflate an
/// entry past its declared size, so this bounds what the parser can read.
pub(crate) fn check_zip_budget(bytes: &[u8]) -> Result<(), ContentError> {
    check_zip_budget_within(bytes, MAX_DOCUMENT_BYTES)
}

fn check_zip_budget_within(bytes: &[u8], budget: u64) -> Result<(), ContentError> {
    let Ok(mut archive) = zip::ZipArchive::new(Cursor::new(bytes)) else {
        // Not a ZIP: the parser reports it.
        return Ok(());
    };
    let mut total: u64 = 0;
    for index in 0..archive.len() {
        let Ok(entry) = archive.by_index_raw(index) else { continue };
        let Ok(name) = entry.name().map(|name| name.to_ascii_lowercase()) else { continue };
        let parsed = [".xml", ".bin", ".rels"].iter().any(|ext| name.ends_with(ext));
        if parsed {
            total = total.saturating_add(entry.size());
        }
    }
    if total > budget {
        return Err(ContentError::FileOperation(format!("the document expands to more than {} MiB", budget >> 20)));
    }
    Ok(())
}

/// Resolve a relationship `target` against the directory of the part that
/// declares it (`ppt/slides` + `../notesSlides/n1.xml` gives `ppt/notesSlides/n1.xml`).
fn resolve_part(base_dir: &str, target: &str) -> String {
    let target = target.split(['#', '?']).next().unwrap_or(target);
    if let Some(absolute) = target.strip_prefix('/') {
        return absolute.to_string();
    }
    let mut segments: Vec<&str> = base_dir.split('/').filter(|s| !s.is_empty()).collect();
    for segment in target.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                segments.pop();
            }
            other => segments.push(other),
        }
    }
    segments.join("/")
}

fn parent_dir(part: &str) -> &str {
    part.rsplit_once('/').map_or("", |(dir, _)| dir)
}

/// Decode `%XX` escapes in a package-internal reference.
fn percent_decode(text: &str) -> String {
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() && bytes[i + 1].is_ascii_hexdigit() && bytes[i + 2].is_ascii_hexdigit()
        {
            let hex = |b: u8| (b as char).to_digit(16).unwrap_or(0) as u8;
            out.push(hex(bytes[i + 1]) * 16 + hex(bytes[i + 2]));
            i += 3;
            continue;
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

/// Dublin Core title and creator from an OOXML `docProps/core.xml` or an
/// OpenDocument `meta.xml`.
fn core_properties(package: &mut Package, part: &str) -> (Option<String>, Option<String>) {
    let Some(root) = package.read_xml_lenient(part) else {
        return (None, None);
    };
    let field = |local: &str| {
        root.find_local(local).map(|node| node.text().trim().to_string()).filter(|text| !text.is_empty())
    };
    (field("title"), field("creator"))
}

#[cfg(test)]
pub(crate) mod testutil {
    use std::io::{Cursor, Write};

    /// Build a ZIP archive from `(name, content)` pairs. A part named
    /// `mimetype` is stored uncompressed, as EPUB and OpenDocument require.
    pub(crate) fn zip_bytes(parts: &[(&str, &[u8])]) -> Vec<u8> {
        let mut writer = zip::ZipWriter::new(Cursor::new(Vec::new()));
        for (name, content) in parts {
            let method =
                if *name == "mimetype" { zip::CompressionMethod::Stored } else { zip::CompressionMethod::Deflated };
            let options = zip::write::SimpleFileOptions::default().compression_method(method);
            writer.start_file(*name, options).unwrap();
            writer.write_all(content).unwrap();
        }
        writer.finish().unwrap().into_inner()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn part_paths_resolve() {
        assert_eq!(resolve_part("ppt/slides", "../notesSlides/notesSlide1.xml"), "ppt/notesSlides/notesSlide1.xml");
        assert_eq!(resolve_part("ppt", "slides/slide1.xml"), "ppt/slides/slide1.xml");
        assert_eq!(resolve_part("OEBPS", "/abs/ch1.xhtml#frag"), "abs/ch1.xhtml");
        assert_eq!(resolve_part("", "./a/../b.xml"), "b.xml");
        assert_eq!(parent_dir("OEBPS/content.opf"), "OEBPS");
        assert_eq!(parent_dir("content.opf"), "");
        assert_eq!(percent_decode("my%20chapter%2F1.xhtml"), "my chapter/1.xhtml");
        assert_eq!(percent_decode("100%"), "100%");
        assert_eq!(percent_decode("a%zzb"), "a%zzb");
    }

    #[test]
    fn text_decoding() {
        assert_eq!(decode_text("héllo".as_bytes()), "héllo");
        assert_eq!(decode_text(b"\xEF\xBB\xBFbom"), "bom");
        assert_eq!(decode_text(b"\xFF\xFEh\x00i\x00"), "hi");
        assert_eq!(decode_text(b"caf\xe9"), "café");
        assert_eq!(decode_with_charset(b"caf\xe9", Some("iso-8859-1")), "café");
        assert_eq!(decode_with_charset(&[0x82, 0xb1, 0x82, 0xf1], Some("Shift_JIS")), "こん");
        assert_eq!(decode_with_charset("ok".as_bytes(), Some("no-such-charset")), "ok");
    }

    #[test]
    fn plain_markdown_and_html_files() {
        let text = extract_document(&FileKind::Text, b"fn main() {}\n").unwrap();
        assert_eq!(text.content, "fn main() {}\n");
        assert_eq!(text.title, None);

        let md = extract_document(&FileKind::Markdown, b"\n# My Notes #\n\nBody *text*.\n").unwrap();
        assert_eq!(md.title.as_deref(), Some("My Notes"));
        assert_eq!(md.content, "\n# My Notes #\n\nBody *text*.\n");
        let untitled = extract_document(&FileKind::Markdown, b"Intro\n\n# Later\n").unwrap();
        assert_eq!(untitled.title, None);

        let html = extract_document(
            &FileKind::Html,
            b"<html lang=\"fr\"><head><title>Page</title><meta name=\"author\" content=\"Zed\"></head><body><nav>menu</nav><h1>T</h1><p>Texte</p></body></html>",
        )
        .unwrap();
        assert_eq!(html.title.as_deref(), Some("Page"));
        assert_eq!(html.content, "# T\n\nTexte");
        assert_eq!(html.metadata["language"], "fr");
        assert_eq!(html.metadata["author"], "Zed");
    }

    #[test]
    fn decompression_is_budgeted_per_document() {
        use testutil::zip_bytes;
        let part = vec![b'a'; 4096];
        let bytes = zip_bytes(&[("one.xml", &part), ("two.xml", &part), ("three.xml", &part), ("photo.jpg", &part)]);

        // Each read, re-reads included, draws on one budget for the document.
        let mut package = Package::with_budget(&bytes, 10_000).unwrap();
        assert_eq!(package.read("one.xml").unwrap().unwrap().len(), 4096);
        assert_eq!(package.read("one.xml").unwrap().unwrap().len(), 4096);
        let error = package.read("two.xml").unwrap_err();
        assert!(matches!(error, ContentError::FileOperation(ref m) if m.contains("expands to more than")), "{error:?}");
        assert_eq!(Package::open(&bytes).unwrap().remaining, MAX_DOCUMENT_BYTES);

        // Third-party parsers are checked on declared sizes; pictures do not count.
        assert!(check_zip_budget_within(&bytes, 3 * 4096).is_ok());
        assert!(check_zip_budget_within(&bytes, 3 * 4096 - 1).is_err());
        assert!(check_zip_budget(&bytes).is_ok());
        assert!(check_zip_budget(b"not a zip").is_ok());
    }

    #[test]
    fn corrupt_zip_is_a_file_error() {
        for kind in [FileKind::Docx, FileKind::Pptx, FileKind::Epub, FileKind::Odt] {
            let error = extract_document(&kind, b"PK\x03\x04 this is not really a zip").unwrap_err();
            assert!(matches!(error, ContentError::FileOperation(_)), "{kind:?}: {error:?}");
        }
    }
}
