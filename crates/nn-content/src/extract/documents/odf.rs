//! OpenDocument text (`.odt`) and presentation (`.odp`) extraction.
//!
//! Both keep their content in `content.xml`: headings (`text:h`), paragraphs,
//! lists and tables, with presentation pages as `## Slide N` sections.

use super::{core_properties, DocOutput, Package};
use crate::error::ContentError;
use crate::extract::markdown;
use crate::extract::xml::{XmlChild, XmlNode};

pub(super) fn extract(bytes: &[u8]) -> Result<DocOutput, ContentError> {
    let mut package = Package::open(bytes)?;
    let content = package
        .read_xml("content.xml")?
        .ok_or_else(|| ContentError::FileOperation("the file has no content.xml part".to_string()))?;

    let mut blocks = Vec::new();
    let mut slides = 0usize;
    if let Some(body) = content.child("office:body") {
        walk(body, 0, &mut slides, &mut blocks);
    }

    let (title, author) = core_properties(&mut package, "meta.xml");
    let mut out = DocOutput { title, content: blocks.join("\n\n"), ..DocOutput::default() };
    out.set_opt("author", author);
    if slides > 0 {
        out.set("slides", slides);
    }
    Ok(out)
}

fn walk(node: &XmlNode, list_depth: usize, slides: &mut usize, blocks: &mut Vec<String>) {
    for child in node.elements() {
        match child.name.as_str() {
            "text:h" => {
                let level = child
                    .attr("text:outline-level")
                    .and_then(|v| v.parse::<usize>().ok())
                    .unwrap_or(1)
                    .clamp(1, 6);
                let text = inline(child).split_whitespace().collect::<Vec<_>>().join(" ");
                if !text.is_empty() {
                    blocks.push(format!("{} {text}", "#".repeat(level)));
                }
            }
            "text:p" => {
                let text = inline(child);
                let text = text.trim();
                if !text.is_empty() {
                    blocks.push(text.to_string());
                }
            }
            "text:list" => {
                let mut lines = Vec::new();
                list(child, list_depth, &mut lines);
                if !lines.is_empty() {
                    blocks.push(lines.join("\n"));
                }
            }
            "table:table" => {
                let rendered = table(child);
                if !rendered.is_empty() {
                    blocks.push(rendered);
                }
            }
            "draw:page" => {
                *slides += 1;
                let mut page = Vec::new();
                walk(child, 0, slides, &mut page);
                if !page.is_empty() {
                    blocks.push(format!("## Slide {slides}"));
                    blocks.append(&mut page);
                }
            }
            // Declarations, change tracking, form definitions and the
            // slide thumbnail on notes pages carry no document text.
            "text:sequence-decls" | "text:tracked-changes" | "office:forms" | "text:table-of-content-source"
            | "draw:page-thumbnail" => {}
            _ => walk(child, list_depth, slides, blocks),
        }
    }
}

fn list(node: &XmlNode, depth: usize, lines: &mut Vec<String>) {
    let indent = "  ".repeat(depth.min(8));
    for item in node.elements().filter(|n| matches!(n.name.as_str(), "text:list-item" | "text:list-header")) {
        for child in item.elements() {
            match child.name.as_str() {
                "text:list" => list(child, depth + 1, lines),
                _ => {
                    let text = inline(child);
                    let text = text.trim();
                    if !text.is_empty() {
                        lines.push(format!("{indent}- {}", text.replace('\n', &format!("\n{indent}  "))));
                    }
                }
            }
        }
    }
}

fn table(node: &XmlNode) -> String {
    let mut row_nodes = Vec::new();
    node.find_all("table:table-row", &mut row_nodes);
    let rows: Vec<Vec<String>> = row_nodes
        .into_iter()
        .map(|row| {
            row.elements()
                .filter(|n| n.name == "table:table-cell")
                .map(|cell| {
                    let mut blocks = Vec::new();
                    walk(cell, 0, &mut 0, &mut blocks);
                    blocks.join(" ")
                })
                .collect::<Vec<String>>()
        })
        .filter(|cells| cells.iter().any(|c| !c.trim().is_empty()))
        .collect();
    markdown::table(&rows)
}

/// Text of a paragraph-level element with ODF's whitespace elements expanded.
fn inline(node: &XmlNode) -> String {
    let mut out = String::new();
    inline_into(node, &mut out);
    out
}

fn inline_into(node: &XmlNode, out: &mut String) {
    for child in &node.children {
        match child {
            XmlChild::Text(text) => out.push_str(text),
            XmlChild::Element(element) => match element.name.as_str() {
                "text:s" => {
                    let count = element.attr("text:c").and_then(|v| v.parse::<usize>().ok()).unwrap_or(1);
                    out.push_str(&" ".repeat(count.min(256)));
                }
                "text:tab" => out.push('\t'),
                "text:line-break" => out.push('\n'),
                // Footnote bodies and comments are not part of the running text.
                "text:note" | "office:annotation" | "text:tracked-changes" => {}
                _ => inline_into(element, out),
            },
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::extract::documents::testutil::zip_bytes;

    const NS: &str = r#"xmlns:office="o" xmlns:text="t" xmlns:table="tb" xmlns:draw="d" xmlns:presentation="p""#;

    pub(crate) fn odt_bytes() -> Vec<u8> {
        let content = format!(
            r#"<office:document-content {NS}><office:body><office:text>
<text:sequence-decls><text:sequence-decl text:name="Illustration"/></text:sequence-decls>
<text:h text:outline-level="1">Field Notes</text:h>
<text:p>Two<text:s text:c="2"/>spaces, a<text:tab/>tab and<text:line-break/>a break<text:note><text:note-body><text:p>footnote text</text:p></text:note-body></text:note>.</text:p>
<text:h text:outline-level="2">Species <text:span>seen</text:span></text:h>
<text:list><text:list-item><text:p>Heron</text:p><text:list><text:list-item><text:p>Grey</text:p></text:list-item></text:list></text:list-item><text:list-item><text:p>Kingfisher</text:p></text:list-item></text:list>
<table:table><table:table-row><table:table-cell><text:p>Bird</text:p></table:table-cell><table:table-cell><text:p>Count</text:p></table:table-cell></table:table-row>
<table:table-row><table:table-cell><text:p>Heron</text:p></table:table-cell><table:table-cell><text:p>3</text:p></table:table-cell></table:table-row></table:table>
<text:p/>
</office:text></office:body></office:document-content>"#
        );
        let meta = r#"<office:document-meta xmlns:office="o" xmlns:dc="dc" xmlns:meta="m"><office:meta><dc:title>Marsh Survey</dc:title><dc:creator>R. Warden</dc:creator></office:meta></office:document-meta>"#;
        zip_bytes(&[
            ("mimetype", b"application/vnd.oasis.opendocument.text"),
            ("content.xml", content.as_bytes()),
            ("meta.xml", meta.as_bytes()),
        ])
    }

    #[test]
    fn odt_structure() {
        let out = extract(&odt_bytes()).unwrap();
        assert_eq!(out.title.as_deref(), Some("Marsh Survey"));
        assert_eq!(out.metadata["author"], "R. Warden");
        assert_eq!(
            out.content,
            "# Field Notes\n\n\
             Two  spaces, a\ttab and\na break.\n\n\
             ## Species seen\n\n\
             - Heron\n  - Grey\n- Kingfisher\n\n\
             | Bird | Count |\n| --- | --- |\n| Heron | 3 |"
        );
    }

    #[test]
    fn odp_pages_become_slides() {
        let content = format!(
            r#"<office:document-content {NS}><office:body><office:presentation>
<draw:page draw:name="page1"><draw:frame><draw:text-box><text:p>Opening remarks</text:p></draw:text-box></draw:frame>
  <presentation:notes><draw:page-thumbnail/><draw:frame><draw:text-box><text:p>Speak slowly.</text:p></draw:text-box></draw:frame></presentation:notes></draw:page>
<draw:page draw:name="page2"><draw:frame><draw:text-box><text:list><text:list-item><text:p>Point A</text:p></text:list-item></text:list></draw:text-box></draw:frame></draw:page>
<draw:page draw:name="empty"/>
</office:presentation></office:body></office:document-content>"#
        );
        let bytes = zip_bytes(&[
            ("mimetype", b"application/vnd.oasis.opendocument.presentation"),
            ("content.xml", content.as_bytes()),
        ]);
        let out = extract(&bytes).unwrap();
        assert_eq!(out.title, None);
        assert_eq!(out.metadata["slides"], 3);
        assert_eq!(out.content, "## Slide 1\n\nOpening remarks\n\nSpeak slowly.\n\n## Slide 2\n\n- Point A");
    }

    #[test]
    fn missing_content_part() {
        let error = extract(&zip_bytes(&[("meta.xml", b"<x/>")])).unwrap_err();
        assert!(matches!(error, ContentError::FileOperation(_)));
    }
}
