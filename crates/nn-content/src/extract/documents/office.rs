//! Word (`.docx`) and PowerPoint (`.pptx`) extraction.
//!
//! Word: body paragraphs in order, heading styles as Markdown headings,
//! numbered and bulleted paragraphs as lists, tables as Markdown tables.
//! Headers, footers, footnotes, comments and tracked deletions are left out.
//!
//! PowerPoint: one `## Slide N: title` section per slide in presentation
//! order, with text boxes, tables and speaker notes.

use std::collections::HashMap;

use super::{core_properties, parent_dir, resolve_part, DocOutput, Package};
use crate::error::ContentError;
use crate::extract::markdown;
use crate::extract::xml::XmlNode;

// --- Word -------------------------------------------------------------------

/// `(numId, level)` to "is a numbered (not bulleted) list".
type Numbering = HashMap<(String, u32), bool>;

pub(super) fn extract_docx(bytes: &[u8]) -> Result<DocOutput, ContentError> {
    let mut package = Package::open(bytes)?;
    let document = package
        .read_xml("word/document.xml")?
        .ok_or_else(|| ContentError::FileOperation("the file has no word/document.xml part".to_string()))?;
    let numbering = package.read_xml_lenient("word/numbering.xml").map(|n| parse_numbering(&n)).unwrap_or_default();

    let mut blocks: Vec<Block> = Vec::new();
    if let Some(body) = document.child("w:body") {
        docx_blocks(body, &numbering, &mut blocks);
    }
    let content = join_blocks(&blocks);

    let (title, author) = core_properties(&mut package, "docProps/core.xml");
    let mut out = DocOutput { title, content, ..DocOutput::default() };
    out.set_opt("author", author);
    Ok(out)
}

struct Block {
    text: String,
    list_item: bool,
}

/// Paragraphs are separated by a blank line, consecutive list items by a newline.
fn join_blocks(blocks: &[Block]) -> String {
    let mut out = String::new();
    let mut previous_was_list = false;
    for block in blocks {
        if !out.is_empty() {
            out.push_str(if block.list_item && previous_was_list { "\n" } else { "\n\n" });
        }
        out.push_str(&block.text);
        previous_was_list = block.list_item;
    }
    out
}

fn parse_numbering(root: &XmlNode) -> Numbering {
    let mut abstract_levels: HashMap<String, Vec<(u32, bool)>> = HashMap::new();
    for definition in root.elements().filter(|n| n.name == "w:abstractNum") {
        let Some(id) = definition.attr("w:abstractNumId") else { continue };
        let levels = definition
            .elements()
            .filter(|n| n.name == "w:lvl")
            .map(|level| {
                let index = level.attr("w:ilvl").and_then(|v| v.parse().ok()).unwrap_or(0);
                let format = level.child("w:numFmt").and_then(|f| f.attr("w:val")).unwrap_or("bullet");
                (index, !matches!(format, "bullet" | "none"))
            })
            .collect();
        abstract_levels.insert(id.to_string(), levels);
    }
    let mut numbering = Numbering::new();
    for instance in root.elements().filter(|n| n.name == "w:num") {
        let (Some(id), Some(abstract_id)) =
            (instance.attr("w:numId"), instance.child("w:abstractNumId").and_then(|a| a.attr("w:val")))
        else {
            continue;
        };
        for (level, ordered) in abstract_levels.get(abstract_id).into_iter().flatten() {
            numbering.insert((id.to_string(), *level), *ordered);
        }
    }
    numbering
}

fn docx_blocks(container: &XmlNode, numbering: &Numbering, blocks: &mut Vec<Block>) {
    for node in container.elements() {
        match node.name.as_str() {
            "w:p" => {
                if let Some(block) = docx_paragraph(node, numbering) {
                    blocks.push(block);
                }
            }
            "w:tbl" => {
                let table = docx_table(node, numbering);
                if !table.is_empty() {
                    blocks.push(Block { text: table, list_item: false });
                }
            }
            // Content controls and revision wrappers hold ordinary blocks.
            "w:sdt" | "w:sdtContent" | "w:ins" | "w:customXml" | "w:smartTag" => docx_blocks(node, numbering, blocks),
            _ => {}
        }
    }
}

fn docx_paragraph(paragraph: &XmlNode, numbering: &Numbering) -> Option<Block> {
    let mut text = String::new();
    docx_inline(paragraph, &mut text);
    let text = text.trim();
    if text.is_empty() {
        return None;
    }

    let properties = paragraph.child("w:pPr");
    let style = properties.and_then(|p| p.child("w:pStyle")).and_then(|s| s.attr("w:val")).unwrap_or("");
    let outline = properties
        .and_then(|p| p.child("w:outlineLvl"))
        .and_then(|o| o.attr("w:val"))
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|level| *level < 9)
        .map(|level| level + 1);
    if let Some(level) = heading_level(style).or(outline) {
        let single_line = text.split_whitespace().collect::<Vec<_>>().join(" ");
        return Some(Block { text: format!("{} {single_line}", "#".repeat(level.min(6))), list_item: false });
    }

    let list = properties.and_then(|p| p.child("w:numPr")).and_then(|num| {
        let id = num.child("w:numId")?.attr("w:val")?;
        let level: u32 = num.child("w:ilvl").and_then(|l| l.attr("w:val")).and_then(|v| v.parse().ok()).unwrap_or(0);
        (id != "0").then(|| (id.to_string(), level))
    });
    if let Some((id, level)) = list {
        let ordered = numbering.get(&(id, level)).copied().unwrap_or(false);
        let marker = if ordered { "1. " } else { "- " };
        let indent = "  ".repeat(level.min(8) as usize);
        let continuation = format!("\n{indent}{}", " ".repeat(marker.len()));
        return Some(Block { text: format!("{indent}{marker}{}", text.replace('\n', &continuation)), list_item: true });
    }
    Some(Block { text: text.to_string(), list_item: false })
}

/// Markdown heading level for a Word paragraph style id (`Heading1`,
/// `heading 2`, `Title`, `Subtitle`). Localised style ids are not guessed;
/// those documents still get headings through `w:outlineLvl` when present.
fn heading_level(style: &str) -> Option<usize> {
    let lowered = style.to_ascii_lowercase().replace([' ', '-', '_'], "");
    if lowered == "title" {
        return Some(1);
    }
    if lowered == "subtitle" {
        return Some(2);
    }
    let level = lowered.strip_prefix("heading")?.parse::<usize>().ok()?;
    (1..=9).contains(&level).then_some(level)
}

fn docx_inline(node: &XmlNode, out: &mut String) {
    for child in node.elements() {
        match child.name.as_str() {
            "w:t" => out.push_str(&child.text()),
            "w:tab" | "w:ptab" => out.push('\t'),
            "w:br" | "w:cr" => out.push('\n'),
            "w:noBreakHyphen" => out.push('-'),
            // Properties, deleted text, field codes, and the fallback copy of
            // content already read from mc:Choice.
            "w:pPr" | "w:rPr" | "w:del" | "w:delText" | "w:instrText" | "w:moveFrom" | "mc:Fallback"
            | "w:footnoteReference" | "w:endnoteReference" | "w:commentReference" => {}
            // Paragraphs nested in a run are text boxes: keep them apart.
            "w:p" => {
                if !out.is_empty() && !out.ends_with('\n') {
                    out.push('\n');
                }
                docx_inline(child, out);
                out.push('\n');
            }
            _ => docx_inline(child, out),
        }
    }
}

fn docx_table(table: &XmlNode, numbering: &Numbering) -> String {
    let mut rows = Vec::new();
    for row in table.elements().filter(|n| n.name == "w:tr") {
        let mut cells = Vec::new();
        for cell in row.elements().filter(|n| n.name == "w:tc") {
            let mut blocks = Vec::new();
            docx_blocks(cell, numbering, &mut blocks);
            cells.push(blocks.iter().map(|b| b.text.trim()).collect::<Vec<_>>().join(" "));
            let span = cell
                .child("w:tcPr")
                .and_then(|p| p.child("w:gridSpan"))
                .and_then(|g| g.attr("w:val"))
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(1);
            for _ in 1..span.min(64) {
                cells.push(String::new());
            }
        }
        if cells.iter().any(|c| !c.is_empty()) {
            rows.push(cells);
        }
    }
    markdown::table(&rows)
}

// --- PowerPoint -------------------------------------------------------------

pub(super) fn extract_pptx(bytes: &[u8]) -> Result<DocOutput, ContentError> {
    let mut package = Package::open(bytes)?;
    let slide_parts = slide_order(&mut package)?;
    if slide_parts.is_empty() {
        return Err(ContentError::FileOperation("the presentation contains no slides".to_string()));
    }

    let mut sections = Vec::new();
    let mut first_title = None;
    for (index, part) in slide_parts.iter().enumerate() {
        let Some(slide) = package.read_xml(part)? else { continue };
        let mut title = None;
        let mut blocks = Vec::new();
        if let Some(tree) = slide.find("p:spTree") {
            slide_shapes(tree, &mut title, &mut blocks);
        }
        if let Some(notes) = slide_notes(&mut package, part) {
            blocks.push(format!("**Notes:** {notes}"));
        }
        if first_title.is_none() {
            first_title = title.clone();
        }
        let heading = match &title {
            Some(title) => format!("## Slide {}: {title}", index + 1),
            None => format!("## Slide {}", index + 1),
        };
        if title.is_some() || !blocks.is_empty() {
            blocks.insert(0, heading);
            sections.push(blocks.join("\n\n"));
        }
    }

    let (title, author) = core_properties(&mut package, "docProps/core.xml");
    // PowerPoint stamps every new deck with this placeholder title.
    let title = title.filter(|t| t != "PowerPoint Presentation").or(first_title);
    let mut out = DocOutput { title, content: sections.join("\n\n"), ..DocOutput::default() };
    out.set("slides", slide_parts.len());
    out.set_opt("author", author);
    Ok(out)
}

/// Relationship id to target part for the `.rels` file next to `part`.
fn relationships(package: &mut Package, part: &str) -> Vec<(String, String, String)> {
    let dir = parent_dir(part);
    let file = part.rsplit('/').next().unwrap_or(part);
    let rels_part = if dir.is_empty() { format!("_rels/{file}.rels") } else { format!("{dir}/_rels/{file}.rels") };
    let Some(root) = package.read_xml_lenient(&rels_part) else {
        return Vec::new();
    };
    root.elements()
        .filter(|n| n.local() == "Relationship")
        .filter_map(|rel| {
            Some((
                rel.attr("Id")?.to_string(),
                rel.attr("Type").unwrap_or("").to_string(),
                resolve_part(dir, rel.attr("Target")?),
            ))
        })
        .collect()
}

/// Slide parts in presentation order. Falls back to numeric file order when
/// the presentation part or its relationships are unusable.
fn slide_order(package: &mut Package) -> Result<Vec<String>, ContentError> {
    let names = package.names();
    let mut ordered = Vec::new();
    if let Some(presentation) = package.read_xml_lenient("ppt/presentation.xml") {
        let rels = relationships(package, "ppt/presentation.xml");
        if let Some(list) = presentation.find("p:sldIdLst") {
            for slide in list.elements().filter(|n| n.name == "p:sldId") {
                let target = slide
                    .attr("r:id")
                    .and_then(|id| rels.iter().find(|(rel_id, _, _)| rel_id == id))
                    .map(|(_, _, target)| target.clone());
                if let Some(target) = target.filter(|t| names.contains(t)) {
                    ordered.push(target);
                }
            }
        }
    }
    if ordered.is_empty() {
        let mut numbered: Vec<(u32, String)> = names
            .iter()
            .filter_map(|name| {
                let number = name.strip_prefix("ppt/slides/slide")?.strip_suffix(".xml")?.parse().ok()?;
                Some((number, name.clone()))
            })
            .collect();
        numbered.sort();
        ordered = numbered.into_iter().map(|(_, name)| name).collect();
    }
    Ok(ordered)
}

fn slide_shapes(tree: &XmlNode, title: &mut Option<String>, blocks: &mut Vec<String>) {
    for shape in tree.elements() {
        match shape.name.as_str() {
            "p:sp" => {
                let placeholder = shape.find("p:ph").and_then(|ph| ph.attr("type")).unwrap_or("");
                if matches!(placeholder, "sldNum" | "dt" | "ftr" | "hdr") {
                    continue;
                }
                let Some(body) = shape.child("p:txBody") else { continue };
                if matches!(placeholder, "title" | "ctrTitle") && title.is_none() {
                    let text = drawing_paragraphs(body)
                        .into_iter()
                        .map(|(_, text)| text)
                        .collect::<Vec<_>>()
                        .join(" ")
                        .split_whitespace()
                        .collect::<Vec<_>>()
                        .join(" ");
                    if !text.is_empty() {
                        *title = Some(text);
                    }
                    continue;
                }
                let text = text_body(body);
                if !text.is_empty() {
                    blocks.push(text);
                }
            }
            "p:grpSp" => slide_shapes(shape, title, blocks),
            "p:graphicFrame" => {
                let mut tables = Vec::new();
                shape.find_all("a:tbl", &mut tables);
                for table in tables {
                    let rows: Vec<Vec<String>> = table
                        .elements()
                        .filter(|n| n.name == "a:tr")
                        .map(|row| {
                            row.elements()
                                .filter(|n| n.name == "a:tc")
                                .map(|cell| {
                                    cell.child("a:txBody")
                                        .map(|body| {
                                            drawing_paragraphs(body)
                                                .into_iter()
                                                .map(|(_, text)| text)
                                                .collect::<Vec<_>>()
                                                .join(" ")
                                        })
                                        .unwrap_or_default()
                                })
                                .collect()
                        })
                        .collect();
                    let rendered = markdown::table(&rows);
                    if !rendered.is_empty() {
                        blocks.push(rendered);
                    }
                }
            }
            // Fallback content repeats what mc:Choice already provided.
            "mc:AlternateContent" => {
                if let Some(choice) = shape.child("mc:Choice") {
                    slide_shapes(choice, title, blocks);
                }
            }
            _ => {}
        }
    }
}

/// Non-empty paragraphs of a DrawingML text body with their indent level.
fn drawing_paragraphs(body: &XmlNode) -> Vec<(u32, String)> {
    body.elements()
        .filter(|n| n.name == "a:p")
        .filter_map(|paragraph| {
            let level =
                paragraph.child("a:pPr").and_then(|p| p.attr("lvl")).and_then(|v| v.parse().ok()).unwrap_or(0u32);
            let mut text = String::new();
            for run in paragraph.elements() {
                match run.name.as_str() {
                    "a:r" | "a:fld" => {
                        if let Some(t) = run.child("a:t") {
                            text.push_str(&t.text());
                        }
                    }
                    "a:br" => text.push('\n'),
                    _ => {}
                }
            }
            let text = text.trim().to_string();
            (!text.is_empty()).then_some((level, text))
        })
        .collect()
}

/// A text box as Markdown: one paragraph stays a paragraph, several become a
/// bullet list that keeps the outline levels.
fn text_body(body: &XmlNode) -> String {
    let paragraphs = drawing_paragraphs(body);
    match paragraphs.as_slice() {
        [] => String::new(),
        [(0, only)] => only.clone(),
        many => many
            .iter()
            .map(|(level, text)| {
                let indent = "  ".repeat((*level).min(8) as usize);
                format!("{indent}- {}", text.replace('\n', &format!("\n{indent}  ")))
            })
            .collect::<Vec<_>>()
            .join("\n"),
    }
}

fn slide_notes(package: &mut Package, slide_part: &str) -> Option<String> {
    let notes_part = relationships(package, slide_part)
        .into_iter()
        .find(|(_, kind, _)| kind.ends_with("/notesSlide"))
        .map(|(_, _, target)| target)?;
    let notes = package.read_xml_lenient(&notes_part)?;
    let mut shapes = Vec::new();
    notes.find_all("p:sp", &mut shapes);
    let text = shapes
        .into_iter()
        .filter(|shape| shape.find("p:ph").and_then(|ph| ph.attr("type")) == Some("body"))
        .filter_map(|shape| shape.child("p:txBody"))
        .flat_map(drawing_paragraphs)
        .map(|(_, text)| text)
        .collect::<Vec<_>>()
        .join("\n");
    (!text.is_empty()).then_some(text)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::extract::documents::testutil::zip_bytes;

    const W_NS: &str = r#"xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006""#;

    pub(crate) fn docx_bytes() -> Vec<u8> {
        let document = format!(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document {W_NS}><w:body>
  <w:p><w:pPr><w:pStyle w:val="Title"/></w:pPr><w:r><w:t>Project Plan</w:t></w:r></w:p>
  <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>Goals</w:t></w:r></w:p>
  <w:p><w:r><w:t xml:space="preserve">Ship the </w:t></w:r><w:r><w:rPr><w:b/></w:rPr><w:t>rewrite</w:t></w:r><w:r><w:t xml:space="preserve"> &amp; keep parity.</w:t></w:r><w:del><w:r><w:delText>deleted words</w:delText></w:r></w:del></w:p>
  <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="1"/></w:numPr></w:pPr><w:r><w:t>First bullet</w:t></w:r></w:p>
  <w:p><w:pPr><w:numPr><w:ilvl w:val="1"/><w:numId w:val="1"/></w:numPr></w:pPr><w:r><w:t>Nested bullet</w:t></w:r></w:p>
  <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr></w:pPr><w:r><w:t>Step one</w:t></w:r></w:p>
  <w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="2"/></w:numPr></w:pPr><w:r><w:t>Step two</w:t></w:r></w:p>
  <w:p/>
  <w:p><w:pPr><w:pStyle w:val="Heading2"/></w:pPr><w:hyperlink><w:r><w:t>Budget</w:t></w:r></w:hyperlink></w:p>
  <w:tbl>
    <w:tr><w:tc><w:p><w:r><w:t>Item</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>Cost</w:t></w:r></w:p></w:tc></w:tr>
    <w:tr><w:tc><w:p><w:r><w:t>Servers</w:t></w:r></w:p><w:p><w:r><w:t>(two)</w:t></w:r></w:p></w:tc><w:tc><w:p><w:r><w:t>1|200</w:t></w:r></w:p></w:tc></w:tr>
  </w:tbl>
  <w:sdt><w:sdtContent><w:p><w:r><w:t>Inside a content control.</w:t><w:br/><w:t>Second line.</w:t></w:r></w:p></w:sdtContent></w:sdt>
  <w:sectPr/>
</w:body></w:document>"#
        );
        let numbering = format!(
            r#"<w:numbering {W_NS}>
  <w:abstractNum w:abstractNumId="0"><w:lvl w:ilvl="0"><w:numFmt w:val="bullet"/></w:lvl><w:lvl w:ilvl="1"><w:numFmt w:val="bullet"/></w:lvl></w:abstractNum>
  <w:abstractNum w:abstractNumId="1"><w:lvl w:ilvl="0"><w:numFmt w:val="decimal"/></w:lvl></w:abstractNum>
  <w:num w:numId="1"><w:abstractNumId w:val="0"/></w:num>
  <w:num w:numId="2"><w:abstractNumId w:val="1"/></w:num>
</w:numbering>"#
        );
        let core = r#"<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>Project Plan 2025</dc:title><dc:creator>Dana Writer</dc:creator></cp:coreProperties>"#;
        zip_bytes(&[
            ("[Content_Types].xml", b"<Types/>"),
            ("word/document.xml", document.as_bytes()),
            ("word/numbering.xml", numbering.as_bytes()),
            ("docProps/core.xml", core.as_bytes()),
        ])
    }

    #[test]
    fn docx_headings_lists_tables() {
        let out = extract_docx(&docx_bytes()).unwrap();
        assert_eq!(out.title.as_deref(), Some("Project Plan 2025"));
        assert_eq!(out.metadata["author"], "Dana Writer");
        assert_eq!(
            out.content,
            "# Project Plan\n\n\
             # Goals\n\n\
             Ship the rewrite & keep parity.\n\n\
             - First bullet\n  - Nested bullet\n1. Step one\n1. Step two\n\n\
             ## Budget\n\n\
             | Item | Cost |\n| --- | --- |\n| Servers (two) | 1\\|200 |\n\n\
             Inside a content control.\nSecond line."
        );
    }

    #[test]
    fn docx_without_optional_parts() {
        let document = format!(
            r#"<w:document {W_NS}><w:body><w:p><w:pPr><w:numPr><w:ilvl w:val="0"/><w:numId w:val="7"/></w:numPr></w:pPr><w:r><w:t>Only item</w:t></w:r></w:p></w:body></w:document>"#
        );
        let out = extract_docx(&zip_bytes(&[("word/document.xml", document.as_bytes())])).unwrap();
        assert_eq!(out.title, None);
        assert_eq!(out.content, "- Only item");
    }

    #[test]
    fn docx_missing_or_broken_main_part() {
        let missing = extract_docx(&zip_bytes(&[("word/other.xml", b"<x/>")])).unwrap_err();
        assert!(matches!(missing, ContentError::FileOperation(_)));
        let broken = extract_docx(&zip_bytes(&[("word/document.xml", b"<w:document><w:body></w:document>")]));
        assert!(matches!(broken, Err(ContentError::FileOperation(_))), "{broken:?}");
    }

    #[test]
    fn heading_styles() {
        assert_eq!(heading_level("Heading1"), Some(1));
        assert_eq!(heading_level("heading 3"), Some(3));
        assert_eq!(heading_level("Title"), Some(1));
        assert_eq!(heading_level("Subtitle"), Some(2));
        assert_eq!(heading_level("Normal"), None);
        assert_eq!(heading_level("Heading"), None);
        assert_eq!(heading_level("Heading10"), None);
    }

    const P_NS: &str = r#"xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships""#;

    fn slide(title: &str, body: &str) -> String {
        format!(
            r#"<p:sld {P_NS}><p:cSld><p:spTree>
  <p:sp><p:nvSpPr><p:nvPr><p:ph type="title"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>{title}</a:t></a:r></a:p></p:txBody></p:sp>
  {body}
  <p:sp><p:nvSpPr><p:nvPr><p:ph type="sldNum"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:fld><a:t>7</a:t></a:fld></a:p></p:txBody></p:sp>
</p:spTree></p:cSld></p:sld>"#
        )
    }

    pub(crate) fn pptx_bytes() -> Vec<u8> {
        let presentation = format!(
            r#"<p:presentation {P_NS}><p:sldIdLst><p:sldId id="256" r:id="rId2"/><p:sldId id="257" r:id="rId1"/></p:sldIdLst></p:presentation>"#
        );
        // Presentation order (rId2 then rId1) differs from file order on purpose.
        let presentation_rels = r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide2.xml"/>
</Relationships>"#;
        let slide1 = slide(
            "Roadmap",
            r#"<p:sp><p:nvSpPr><p:nvPr><p:ph type="body"/></p:nvPr></p:nvSpPr><p:txBody>
                 <a:p><a:r><a:t>Phase one</a:t></a:r></a:p>
                 <a:p><a:pPr lvl="1"/><a:r><a:t>Design </a:t></a:r><a:r><a:t>review</a:t></a:r></a:p>
                 <a:p><a:endParaRPr/></a:p>
               </p:txBody></p:sp>
               <p:graphicFrame><a:graphic><a:graphicData><a:tbl>
                 <a:tr><a:tc><a:txBody><a:p><a:r><a:t>Q</a:t></a:r></a:p></a:txBody></a:tc><a:tc><a:txBody><a:p><a:r><a:t>Goal</a:t></a:r></a:p></a:txBody></a:tc></a:tr>
                 <a:tr><a:tc><a:txBody><a:p><a:r><a:t>1</a:t></a:r></a:p></a:txBody></a:tc><a:tc><a:txBody><a:p><a:r><a:t>Launch</a:t></a:r></a:p></a:txBody></a:tc></a:tr>
               </a:tbl></a:graphicData></a:graphic></p:graphicFrame>"#,
        );
        let slide2 = slide(
            "Welcome &amp; Agenda",
            r#"<p:grpSp><p:sp><p:txBody><a:p><a:r><a:t>A single text box.</a:t></a:r></a:p></p:txBody></p:sp></p:grpSp>"#,
        );
        let slide2_rels = r#"<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide" Target="../notesSlides/notesSlide1.xml"/>
</Relationships>"#;
        let notes = format!(
            r#"<p:notes {P_NS}><p:cSld><p:spTree>
  <p:sp><p:nvSpPr><p:nvPr><p:ph type="sldImg"/></p:nvPr></p:nvSpPr></p:sp>
  <p:sp><p:nvSpPr><p:nvPr><p:ph type="body" idx="1"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:r><a:t>Greet everyone first.</a:t></a:r></a:p></p:txBody></p:sp>
  <p:sp><p:nvSpPr><p:nvPr><p:ph type="sldNum"/></p:nvPr></p:nvSpPr><p:txBody><a:p><a:fld><a:t>1</a:t></a:fld></a:p></p:txBody></p:sp>
</p:spTree></p:cSld></p:notes>"#
        );
        let core = r#"<cp:coreProperties xmlns:cp="x" xmlns:dc="y"><dc:title>PowerPoint Presentation</dc:title></cp:coreProperties>"#;
        zip_bytes(&[
            ("ppt/presentation.xml", presentation.as_bytes()),
            ("ppt/_rels/presentation.xml.rels", presentation_rels.as_bytes()),
            ("ppt/slides/slide1.xml", slide1.as_bytes()),
            ("ppt/slides/slide2.xml", slide2.as_bytes()),
            ("ppt/slides/_rels/slide2.xml.rels", slide2_rels.as_bytes()),
            ("ppt/notesSlides/notesSlide1.xml", notes.as_bytes()),
            ("docProps/core.xml", core.as_bytes()),
        ])
    }

    #[test]
    fn pptx_slides_in_presentation_order_with_notes_and_tables() {
        let out = extract_pptx(&pptx_bytes()).unwrap();
        assert_eq!(out.title.as_deref(), Some("Welcome & Agenda"));
        assert_eq!(out.metadata["slides"], 2);
        assert_eq!(
            out.content,
            "## Slide 1: Welcome & Agenda\n\n\
             A single text box.\n\n\
             **Notes:** Greet everyone first.\n\n\
             ## Slide 2: Roadmap\n\n\
             - Phase one\n  - Design review\n\n\
             | Q | Goal |\n| --- | --- |\n| 1 | Launch |"
        );
    }

    #[test]
    fn pptx_falls_back_to_numeric_slide_order() {
        let s = |title: &str| slide(title, "");
        let bytes = zip_bytes(&[
            ("ppt/presentation.xml", b"<p:presentation/>"),
            ("ppt/slides/slide10.xml", s("Ten").as_bytes()),
            ("ppt/slides/slide2.xml", s("Two").as_bytes()),
        ]);
        let out = extract_pptx(&bytes).unwrap();
        assert_eq!(out.content, "## Slide 1: Two\n\n## Slide 2: Ten");
        assert_eq!(out.title.as_deref(), Some("Two"));
    }

    #[test]
    fn pptx_without_slides_is_an_error() {
        let error = extract_pptx(&zip_bytes(&[("ppt/presentation.xml", b"<p:presentation/>")])).unwrap_err();
        assert!(matches!(error, ContentError::FileOperation(_)));
    }
}
