//! PDF text extraction.
//!
//! Text comes from the PDF's text layer via `pdf-extract`, page by page, with
//! `lopdf`'s simpler extractor as the fallback for pages it cannot handle.
//! There is no OCR: a scanned PDF has no text layer and yields nothing.

use std::panic::{catch_unwind, AssertUnwindSafe};

use lopdf::{Document, Object};

use super::DocOutput;
use crate::error::ContentError;
use crate::extract::markdown;

pub(super) fn extract(bytes: &[u8]) -> Result<DocOutput, ContentError> {
    let mut document =
        Document::load_mem(bytes).map_err(|e| ContentError::FileOperation(format!("cannot parse PDF: {e}")))?;
    if document.is_encrypted() {
        document.decrypt("").map_err(|e| {
            ContentError::FileOperation(format!("the PDF is password-protected and cannot be read: {e}"))
        })?;
    }

    let page_numbers: Vec<u32> = document.get_pages().keys().copied().collect();
    let mut pages = Vec::with_capacity(page_numbers.len());
    let mut failed_pages = 0usize;
    for number in &page_numbers {
        match page_text(&document, *number) {
            Some(text) => pages.push(markdown::tidy(&text.replace('\r', "\n").replace('\u{0}', ""))),
            None => failed_pages += 1,
        }
    }

    let content = pages.into_iter().filter(|page| !page.is_empty()).collect::<Vec<_>>().join("\n\n");
    let mut out = DocOutput { title: info_string(&document, b"Title"), content, ..DocOutput::default() };
    out.set("pages", page_numbers.len());
    if failed_pages > 0 {
        out.set("unreadable_pages", failed_pages);
    }
    out.set_opt("author", info_string(&document, b"Author"));
    if out.content.trim().is_empty() && !page_numbers.is_empty() {
        out.set("note", "no text layer found; the PDF may be scanned (OCR is not available)");
    }
    Ok(out)
}

/// Text of one page. `None` when neither extractor can read it.
fn page_text(document: &Document, number: u32) -> Option<String> {
    // pdf-extract panics on some malformed fonts and content streams.
    let primary = catch_unwind(AssertUnwindSafe(|| {
        let mut text = String::new();
        let mut output = pdf_extract::PlainTextOutput::new(&mut text);
        pdf_extract::output_doc_page(document, &mut output, number).map(|()| text)
    }));
    if let Ok(Ok(text)) = primary {
        return Some(text);
    }
    catch_unwind(AssertUnwindSafe(|| document.extract_text(&[number]))).ok()?.ok()
}

/// A text entry of the document information dictionary.
fn info_string(document: &Document, key: &[u8]) -> Option<String> {
    let info = document.trailer.get(b"Info").ok()?;
    let (_, info) = document.dereference(info).ok()?;
    let value = info.as_dict().ok()?.get(key).ok()?;
    let (_, value) = document.dereference(value).ok()?;
    if !matches!(value, Object::String(..)) {
        return None;
    }
    let text = lopdf::decode_text_string(value).ok()?;
    let text = text.trim().trim_matches('\u{0}').trim().to_string();
    (!text.is_empty()).then_some(text)
}

#[cfg(test)]
pub(crate) mod tests {
    use lopdf::content::{Content, Operation};
    use lopdf::{dictionary, Stream};

    use super::*;

    /// Build a PDF with one page per entry of `pages`, each line of a page as
    /// its own text line, and optionally a title in the info dictionary.
    pub(crate) fn pdf_bytes(pages: &[&[&str]], title: Option<&str>) -> Vec<u8> {
        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let font_id = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "Helvetica",
            "Encoding" => "WinAnsiEncoding",
        });
        let resources_id = doc.add_object(dictionary! {
            "Font" => dictionary! { "F1" => font_id },
        });
        let mut kids = Vec::new();
        for lines in pages {
            let mut operations = vec![
                Operation::new("BT", vec![]),
                Operation::new("Tf", vec!["F1".into(), 12.into()]),
                Operation::new("TL", vec![16.into()]),
                Operation::new("Td", vec![72.into(), 720.into()]),
            ];
            for line in *lines {
                operations.push(Operation::new("Tj", vec![Object::string_literal(*line)]));
                operations.push(Operation::new("T*", vec![]));
            }
            operations.push(Operation::new("ET", vec![]));
            let content = Content { operations };
            let content_id = doc.add_object(Stream::new(dictionary! {}, content.encode().unwrap()));
            let page_id = doc.add_object(dictionary! {
                "Type" => "Page",
                "Parent" => pages_id,
                "Contents" => content_id,
            });
            kids.push(Object::Reference(page_id));
        }
        let count = kids.len() as i64;
        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Kids" => kids,
                "Count" => count,
                "Resources" => resources_id,
                "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
            }),
        );
        let catalog_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        doc.trailer.set("Root", catalog_id);
        if let Some(title) = title {
            let info_id = doc.add_object(dictionary! {
                "Title" => Object::string_literal(title),
                "Author" => Object::string_literal("Test Author"),
            });
            doc.trailer.set("Info", info_id);
        }
        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).unwrap();
        bytes
    }

    #[test]
    fn extracts_text_pages_and_title() {
        let bytes = pdf_bytes(
            &[&["Quarterly report", "Revenue grew by ten percent."], &["Second page text."]],
            Some("Quarterly Report 2024"),
        );
        let out = extract(&bytes).unwrap();
        assert_eq!(out.title.as_deref(), Some("Quarterly Report 2024"));
        assert_eq!(out.metadata["pages"], 2);
        assert_eq!(out.metadata["author"], "Test Author");
        assert!(out.content.contains("Quarterly report"), "{}", out.content);
        assert!(out.content.contains("Revenue grew by ten percent."), "{}", out.content);
        let first = out.content.find("Revenue").unwrap();
        let second = out.content.find("Second page text.").unwrap();
        assert!(first < second);
    }

    #[test]
    fn pdf_without_text_is_empty_with_a_note() {
        let bytes = pdf_bytes(&[&[]], None);
        let out = extract(&bytes).unwrap();
        assert_eq!(out.title, None);
        assert_eq!(out.content, "");
        assert!(out.metadata["note"].as_str().unwrap().contains("scanned"));
    }

    #[test]
    fn garbage_is_a_file_error() {
        let error = extract(b"%PDF-1.4\nthis is not a pdf at all").unwrap_err();
        assert!(matches!(error, ContentError::FileOperation(_)), "{error:?}");
    }
}
