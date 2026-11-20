//! The file flow: detect the type, then parse the document or transcribe the
//! media. URL downloads of documents re-enter here through a temporary file.

use std::io::Read;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::path::Path;

use serde_json::{json, Value};

use super::detect::{self, Detected, FileKind};
use super::{documents, media, ExtractContext, Extracted};
use crate::error::ContentError;

/// Bytes read from the start of a file for type detection.
const HEAD_BYTES: usize = 8192;

/// Detect the type of the file at `path`. `file_name` (the name the user gave
/// the file) supplies the extension when it has one; otherwise the extension
/// of `path` is used.
pub(crate) async fn detect_file(
    path: &Path,
    file_name: Option<&str>,
    mime_type: Option<&str>,
) -> Result<Detected, ContentError> {
    let extension = file_name
        .and_then(detect::extension_of)
        .or_else(|| path.file_name().and_then(|n| n.to_str()).and_then(detect::extension_of));
    let path = path.to_path_buf();
    let mime_type = mime_type.map(str::to_string);
    run_blocking(move || {
        let metadata = std::fs::metadata(&path).map_err(|e| ContentError::FileOperation(format!("cannot open the file: {e}")))?;
        if !metadata.is_file() {
            return Err(ContentError::FileOperation("the path is not a regular file".to_string()));
        }
        let mut file =
            std::fs::File::open(&path).map_err(|e| ContentError::FileOperation(format!("cannot open the file: {e}")))?;
        let mut head = Vec::with_capacity(HEAD_BYTES);
        file.by_ref()
            .take(HEAD_BYTES as u64)
            .read_to_end(&mut head)
            .map_err(|e| ContentError::FileOperation(format!("cannot read the file: {e}")))?;
        let zip = || std::fs::File::open(&path).ok().and_then(detect::zip_flavor);
        Ok(detect::detect(&head, zip, extension.as_deref(), mime_type.as_deref()))
    })
    .await
}

/// Extract the file at `path`. The returned value has no `url` or `file_path`
/// set: the caller knows which of the two the input was.
pub(crate) async fn extract_file(
    path: &Path,
    file_name: Option<&str>,
    mime_type: Option<&str>,
    ctx: &ExtractContext,
) -> Result<Extracted, ContentError> {
    let detected = detect_file(path, file_name, mime_type).await?;
    match &detected.kind {
        FileKind::Unsupported { reason } => Err(ContentError::UnsupportedType {
            identified_type: detected.mime.clone(),
            reason: reason.clone(),
        }),
        FileKind::Audio | FileKind::Video => media::transcribe_file(path, &detected, ctx).await,
        kind => {
            let bytes = tokio::fs::read(path)
                .await
                .map_err(|e| ContentError::FileOperation(format!("cannot read the file: {e}")))?;
            let size = bytes.len();
            let parse_kind = kind.clone();
            let document = run_blocking(move || {
                // Third-party parsers can panic on malformed input; that is a
                // bad file, not a bug to crash the worker over.
                catch_unwind(AssertUnwindSafe(|| documents::extract_document(&parse_kind, &bytes))).unwrap_or_else(|_| {
                    Err(ContentError::FileOperation(format!(
                        "the {} parser failed on this file; it is probably corrupt",
                        parse_kind.identified_type()
                    )))
                })
            })
            .await?;

            let mut metadata = document.metadata;
            metadata.insert("mime_type".to_string(), json!(detected.mime));
            metadata.insert("size_bytes".to_string(), json!(size));
            if let Some(extension) = &detected.extension {
                metadata.insert("extension".to_string(), json!(extension));
            }
            Ok(Extracted {
                title: document.title.map(|t| t.trim().to_string()).filter(|t| !t.is_empty()),
                content: document.content,
                identified_type: kind.identified_type().to_string(),
                url: None,
                file_path: None,
                metadata: Value::Object(metadata),
            })
        }
    }
}

async fn run_blocking<T, F>(work: F) -> Result<T, ContentError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, ContentError> + Send + 'static,
{
    tokio::task::spawn_blocking(work)
        .await
        .unwrap_or_else(|e| Err(ContentError::FileOperation(format!("file processing task failed: {e}"))))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extract::documents::testutil::zip_bytes;
    use crate::extract::{check_file_support, extract_content, ExtractInput};

    fn write(dir: &tempfile::TempDir, name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, bytes).unwrap();
        path
    }

    async fn run(path: &Path) -> Result<Extracted, ContentError> {
        extract_content(ExtractInput::file(path), &ExtractContext::default()).await
    }

    #[tokio::test]
    async fn every_document_type_through_the_public_entry_point() {
        use crate::extract::documents as docs;
        let dir = tempfile::tempdir().unwrap();
        // (file name, bytes, identified type, title, expected snippet)
        type Case = (&'static str, Vec<u8>, &'static str, Option<&'static str>, &'static str);
        let cases: Vec<Case> = vec![
            (
                "report.pdf",
                docs::pdf::tests::pdf_bytes(&[&["Hello from a PDF page."]], Some("PDF Title")),
                "pdf",
                Some("PDF Title"),
                "Hello from a PDF page.",
            ),
            ("plan.docx", docs::office::tests::docx_bytes(), "docx", Some("Project Plan 2025"), "# Goals"),
            ("deck.pptx", docs::office::tests::pptx_bytes(), "pptx", Some("Welcome & Agenda"), "## Slide 2: Roadmap"),
            ("stock.xlsx", docs::sheets::tests::xlsx_bytes(), "xlsx", None, "| Item | Qty | In stock |"),
            ("notes.odt", docs::odf::tests::odt_bytes(), "odt", Some("Marsh Survey"), "## Species seen"),
            ("book.epub", docs::epub::tests::epub_bytes(), "epub", Some("The Little Handbook"), "# Chapter Two"),
            ("data.csv", b"a,b\n1,2\n".to_vec(), "csv", None, "| a | b |\n| --- | --- |\n| 1 | 2 |"),
            ("data.tsv", b"a\tb\n1\t2\n".to_vec(), "csv", None, "| 1 | 2 |"),
            (
                "page.html",
                b"<html><head><title>Saved Page</title></head><body><h1>Saved</h1><p>Text</p></body></html>".to_vec(),
                "html",
                Some("Saved Page"),
                "# Saved\n\nText",
            ),
            ("README.md", b"# Read Me\n\nDetails.\n".to_vec(), "markdown", Some("Read Me"), "Details."),
            ("notes.txt", b"plain notes\n".to_vec(), "text", None, "plain notes"),
            ("config.yaml", b"key: value\n".to_vec(), "text", None, "key: value"),
            ("data.json", b"{\"k\": [1, 2]}".to_vec(), "text", None, "{\"k\": [1, 2]}"),
            ("feed.xml", b"<?xml version=\"1.0\"?><a><b>c</b></a>".to_vec(), "text", None, "<b>c</b>"),
            ("main.rs", b"fn main() {}\n".to_vec(), "text", None, "fn main() {}"),
        ];
        for (name, bytes, expected_type, expected_title, expected_snippet) in cases {
            let path = write(&dir, name, &bytes);
            let out = run(&path).await.unwrap_or_else(|e| panic!("{name}: {e:?}"));
            assert_eq!(out.identified_type, expected_type, "{name}");
            assert_eq!(out.title.as_deref(), expected_title, "{name}");
            assert!(out.content.contains(expected_snippet), "{name}: {}", out.content);
            assert_eq!(out.file_path.as_deref(), Some(path.as_path()), "{name}");
            assert_eq!(out.url, None);
            assert_eq!(out.metadata["size_bytes"], bytes.len());
            assert!(out.metadata["mime_type"].is_string());

            let support = check_file_support(&path, None, None).await.unwrap();
            assert!(support.supported, "{name}");
            assert_eq!(support.identified_type, expected_type, "{name}");
        }
    }

    #[tokio::test]
    async fn content_wins_over_a_misleading_name() {
        let dir = tempfile::tempdir().unwrap();
        // A docx stored under a server-generated name without extension.
        let path = write(&dir, "upload-7f3a", &crate::extract::documents::office::tests::docx_bytes());
        let out = run(&path).await.unwrap();
        assert_eq!(out.identified_type, "docx");

        // A PDF uploaded as "notes.txt".
        let pdf = crate::extract::documents::pdf::tests::pdf_bytes(&[&["Really a PDF."]], None);
        let path = write(&dir, "blob", &pdf);
        let input = ExtractInput::file(&path).with_file_name("notes.txt").with_mime_type("text/plain");
        let out = extract_content(input, &ExtractContext::default()).await.unwrap();
        assert_eq!(out.identified_type, "pdf");
        assert!(out.content.contains("Really a PDF."));
        assert_eq!(out.title, None);
    }

    #[tokio::test]
    async fn original_file_name_supplies_the_extension() {
        let dir = tempfile::tempdir().unwrap();
        let path = write(&dir, "c0ffee", b"x,y\n3,4\n");
        let as_text = run(&path).await.unwrap();
        assert_eq!(as_text.identified_type, "text");
        let as_csv = extract_content(ExtractInput::file(&path).with_file_name("table.CSV"), &ExtractContext::default())
            .await
            .unwrap();
        assert_eq!(as_csv.identified_type, "csv");
        assert_eq!(as_csv.content, "| x | y |\n| --- | --- |\n| 3 | 4 |");
        let by_mime = extract_content(ExtractInput::file(&path).with_mime_type("text/csv"), &ExtractContext::default())
            .await
            .unwrap();
        assert_eq!(by_mime.identified_type, "csv");
    }

    #[tokio::test]
    async fn unsupported_files_name_their_type() {
        let dir = tempfile::tempdir().unwrap();
        let cases: Vec<(&str, Vec<u8>, &str)> = vec![
            ("photo.png", b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR".to_vec(), "image/png"),
            ("photo.jpg", vec![0xFF, 0xD8, 0xFF, 0xE0, 0, 16, b'J', b'F', b'I', b'F'], "image/jpeg"),
            ("old.doc", vec![0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1, 0, 0, 0, 0], "application/msword"),
            ("bundle.zip", zip_bytes(&[("a.txt", b"hello")]), "application/zip"),
            ("blob.bin", vec![0, 159, 146, 150, 0, 1, 2, 3], "application/octet-stream"),
        ];
        for (name, bytes, mime) in cases {
            let path = write(&dir, name, &bytes);
            match run(&path).await {
                Err(ContentError::UnsupportedType { identified_type, reason }) => {
                    assert_eq!(identified_type, mime, "{name}");
                    assert!(!reason.is_empty());
                }
                other => panic!("{name}: expected UnsupportedType, got {other:?}"),
            }
            let support = check_file_support(&path, None, None).await.unwrap();
            assert!(!support.supported, "{name}");
            assert_eq!(support.identified_type, mime);
            assert!(support.reason.is_some());
        }
    }

    #[tokio::test]
    async fn missing_corrupt_and_empty_files() {
        let dir = tempfile::tempdir().unwrap();
        let missing = run(&dir.path().join("nope.pdf")).await.unwrap_err();
        assert!(matches!(missing, ContentError::FileOperation(_)), "{missing:?}");
        let directory = run(dir.path()).await.unwrap_err();
        assert!(matches!(directory, ContentError::FileOperation(_)), "{directory:?}");

        let corrupt = write(&dir, "broken.docx", b"PK\x03\x04 definitely not a zip archive");
        let error = run(&corrupt).await.unwrap_err();
        assert!(matches!(error, ContentError::FileOperation(_)), "{error:?}");

        let empty = write(&dir, "empty.txt", b"");
        let error = run(&empty).await.unwrap_err();
        assert!(matches!(error, ContentError::EmptyContent(_)), "{error:?}");

        let blank_pdf = write(&dir, "scan.pdf", &crate::extract::documents::pdf::tests::pdf_bytes(&[&[]], None));
        let error = run(&blank_pdf).await.unwrap_err();
        assert!(matches!(error, ContentError::EmptyContent(_)), "{error:?}");
    }
}
