//! EPUB extraction: the chapters named by the package spine, in reading
//! order, each converted from XHTML to Markdown.

use super::{parent_dir, percent_decode, resolve_part, DocOutput, Package};
use crate::error::ContentError;
use crate::extract::html::{html_to_markdown, HtmlOptions};

pub(super) fn extract(bytes: &[u8]) -> Result<DocOutput, ContentError> {
    let mut package = Package::open(bytes)?;
    let opf_path = package
        .read_xml("META-INF/container.xml")?
        .and_then(|container| {
            container.find_local("rootfile").and_then(|r| r.attr("full-path")).map(percent_decode)
        })
        .or_else(|| package.names().into_iter().find(|name| name.to_ascii_lowercase().ends_with(".opf")))
        .ok_or_else(|| ContentError::FileOperation("the EPUB has no package document".to_string()))?;
    let opf = package
        .read_xml(&opf_path)?
        .ok_or_else(|| ContentError::FileOperation(format!("the EPUB package document '{opf_path}' is missing")))?;
    let base = parent_dir(&opf_path).to_string();

    // id -> (part path, media type, properties)
    let manifest: Vec<(String, String, String, String)> = opf
        .find_local("manifest")
        .map(|manifest| {
            manifest
                .elements()
                .filter(|n| n.local() == "item")
                .filter_map(|item| {
                    Some((
                        item.attr("id")?.to_string(),
                        resolve_part(&base, &percent_decode(item.attr("href")?)),
                        item.attr("media-type").unwrap_or("").to_ascii_lowercase(),
                        item.attr("properties").unwrap_or("").to_string(),
                    ))
                })
                .collect()
        })
        .unwrap_or_default();

    let spine: Vec<String> = opf
        .find_local("spine")
        .map(|spine| {
            spine.elements().filter(|n| n.local() == "itemref").filter_map(|i| i.attr("idref")).map(String::from).collect()
        })
        .unwrap_or_default();

    let options = HtmlOptions { base_url: None, main_content: false };
    let mut chapters = Vec::new();
    for idref in &spine {
        let Some((_, part, media_type, properties)) = manifest.iter().find(|(id, ..)| id == idref) else {
            continue;
        };
        let is_markup = media_type.contains("html") || media_type.contains("xml") || media_type.is_empty();
        if !is_markup || properties.split_whitespace().any(|p| p == "nav") {
            continue;
        }
        let Some(markup) = package.read_text(part)? else { continue };
        let chapter = html_to_markdown(&markup, &options).markdown;
        if !chapter.trim().is_empty() {
            chapters.push(chapter);
        }
    }

    let metadata = opf.find_local("metadata");
    let field = |local: &str| {
        metadata
            .and_then(|m| m.elements().find(|n| n.local() == local))
            .map(|node| node.text().split_whitespace().collect::<Vec<_>>().join(" "))
            .filter(|text| !text.is_empty())
    };
    let mut out = DocOutput { title: field("title"), content: chapters.join("\n\n"), ..DocOutput::default() };
    out.set("chapters", chapters.len());
    out.set_opt("author", field("creator"));
    out.set_opt("language", field("language"));
    out.set_opt("publisher", field("publisher"));
    Ok(out)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::extract::documents::testutil::zip_bytes;

    pub(crate) fn epub_bytes() -> Vec<u8> {
        let container = r#"<?xml version="1.0"?><container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container"><rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles></container>"#;
        let opf = r#"<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0"><metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>The  Little
 Handbook</dc:title><dc:creator>Jo Author</dc:creator><dc:language>en</dc:language></metadata>
<manifest>
  <item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>
  <item id="c2" href="text/chapter%202.xhtml" media-type="application/xhtml+xml"/>
  <item id="c1" href="text/chapter1.xhtml" media-type="application/xhtml+xml"/>
  <item id="css" href="style.css" media-type="text/css"/>
  <item id="cover" href="cover.jpg" media-type="image/jpeg"/>
</manifest>
<spine><itemref idref="nav"/><itemref idref="c1"/><itemref idref="cover"/><itemref idref="c2"/><itemref idref="ghost"/></spine></package>"#;
        let chapter1 = r#"<?xml version="1.0" encoding="utf-8"?><html xmlns="http://www.w3.org/1999/xhtml"><head><title>ignored</title><style>p { margin: 0 }</style></head><body><h1>Chapter One</h1><p>It began with a <em>question</em>.</p></body></html>"#;
        let chapter2 = r#"<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>Chapter Two</h1><p>And ended with an answer.</p><ul><li>First</li><li>Second</li></ul></body></html>"#;
        let nav = r#"<html><body><nav><ol><li><a href="text/chapter1.xhtml">TOC entry</a></li></ol></nav></body></html>"#;
        zip_bytes(&[
            ("mimetype", b"application/epub+zip"),
            ("META-INF/container.xml", container.as_bytes()),
            ("OEBPS/content.opf", opf.as_bytes()),
            ("OEBPS/nav.xhtml", nav.as_bytes()),
            ("OEBPS/text/chapter1.xhtml", chapter1.as_bytes()),
            ("OEBPS/text/chapter 2.xhtml", chapter2.as_bytes()),
            ("OEBPS/style.css", b"p { margin: 0 }"),
            ("OEBPS/cover.jpg", &[0xFF, 0xD8, 0xFF]),
        ])
    }

    #[test]
    fn chapters_follow_the_spine() {
        let out = extract(&epub_bytes()).unwrap();
        assert_eq!(out.title.as_deref(), Some("The Little Handbook"));
        assert_eq!(out.metadata["author"], "Jo Author");
        assert_eq!(out.metadata["language"], "en");
        assert_eq!(out.metadata["chapters"], 2);
        assert_eq!(
            out.content,
            "# Chapter One\n\nIt began with a *question*.\n\n# Chapter Two\n\nAnd ended with an answer.\n\n- First\n- Second"
        );
    }

    #[test]
    fn epub_without_package_document() {
        let error = extract(&zip_bytes(&[("mimetype", b"application/epub+zip")])).unwrap_err();
        assert!(matches!(error, ContentError::FileOperation(_)));
    }
}
