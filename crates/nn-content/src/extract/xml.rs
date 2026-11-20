//! A minimal XML tree for walking Office and EPUB package parts.
//!
//! Names keep their prefix as written (`w:p`, `text:h`): the formats read
//! here use fixed, conventional prefixes, so no namespace resolution is done.

use quick_xml::events::Event;
use quick_xml::Reader;

/// Deepest element nesting accepted; protects the recursive walkers.
const MAX_DEPTH: usize = 512;

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct XmlNode {
    pub name: String,
    pub attrs: Vec<(String, String)>,
    pub children: Vec<XmlChild>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum XmlChild {
    Element(XmlNode),
    Text(String),
}

impl XmlNode {
    /// The name without its namespace prefix.
    pub fn local(&self) -> &str {
        self.name.rsplit(':').next().unwrap_or(&self.name)
    }

    /// Attribute by qualified name (`w:val`).
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attrs.iter().find(|(key, _)| key == name).map(|(_, value)| value.as_str())
    }

    pub fn elements(&self) -> impl Iterator<Item = &XmlNode> {
        self.children.iter().filter_map(|child| match child {
            XmlChild::Element(node) => Some(node),
            XmlChild::Text(_) => None,
        })
    }

    /// First direct child with this qualified name.
    pub fn child(&self, name: &str) -> Option<&XmlNode> {
        self.elements().find(|node| node.name == name)
    }

    /// First descendant (depth first) with this qualified name.
    pub fn find(&self, name: &str) -> Option<&XmlNode> {
        self.elements().find_map(|node| if node.name == name { Some(node) } else { node.find(name) })
    }

    /// First descendant with this local name, whatever its prefix.
    pub fn find_local(&self, local: &str) -> Option<&XmlNode> {
        self.elements().find_map(|node| if node.local() == local { Some(node) } else { node.find_local(local) })
    }

    /// Every descendant with this qualified name, in document order. Matches
    /// are not searched for nested matches.
    pub fn find_all<'a>(&'a self, name: &str, out: &mut Vec<&'a XmlNode>) {
        for node in self.elements() {
            if node.name == name {
                out.push(node);
            } else {
                node.find_all(name, out);
            }
        }
    }

    /// All text below this node, concatenated.
    pub fn text(&self) -> String {
        let mut out = String::new();
        self.collect_text(&mut out);
        out
    }

    fn collect_text(&self, out: &mut String) {
        for child in &self.children {
            match child {
                XmlChild::Text(text) => out.push_str(text),
                XmlChild::Element(node) => node.collect_text(out),
            }
        }
    }
}

/// Decode an XML part: UTF-8 unless a byte-order mark says UTF-16.
pub(crate) fn decode(bytes: &[u8]) -> String {
    let (text, _, _) = encoding_rs::UTF_8.decode(bytes);
    text.into_owned()
}

/// Parse `xml` into a tree and return its root element.
pub(crate) fn parse(xml: &str) -> Result<XmlNode, String> {
    let mut reader = Reader::from_str(xml.trim_start_matches('\u{feff}'));
    let mut stack: Vec<XmlNode> = vec![XmlNode::default()];

    let element = |start: &quick_xml::events::BytesStart| -> XmlNode {
        let attrs = start
            .attributes()
            .with_checks(false)
            .flatten()
            .map(|attr| {
                let value = quick_xml::escape::unescape(&attr.value).map(|v| v.into_owned()).unwrap_or_else(|_| attr.value.to_string());
                (attr.key.as_ref().to_string(), value)
            })
            .collect();
        XmlNode { name: start.name().as_ref().to_string(), attrs, children: Vec::new() }
    };
    fn push_text(stack: &mut [XmlNode], text: &str) {
        if let Some(top) = stack.last_mut() {
            match top.children.last_mut() {
                Some(XmlChild::Text(existing)) => existing.push_str(text),
                _ => top.children.push(XmlChild::Text(text.to_string())),
            }
        }
    }

    loop {
        match reader.read_event().map_err(|e| e.to_string())? {
            Event::Start(start) => {
                if stack.len() > MAX_DEPTH {
                    return Err("XML is nested too deeply".to_string());
                }
                stack.push(element(&start));
            }
            Event::Empty(start) => {
                let node = element(&start);
                if let Some(top) = stack.last_mut() {
                    top.children.push(XmlChild::Element(node));
                }
            }
            Event::End(_) => {
                if stack.len() < 2 {
                    return Err("unbalanced closing tag".to_string());
                }
                let node = stack.pop().unwrap_or_default();
                if let Some(top) = stack.last_mut() {
                    top.children.push(XmlChild::Element(node));
                }
            }
            Event::Text(text) => push_text(&mut stack, &text),
            Event::CData(data) => push_text(&mut stack, &data),
            Event::GeneralRef(reference) => {
                let resolved = match reference.resolve_char_ref() {
                    Ok(Some(c)) => Some(c),
                    _ => match &*reference {
                        "amp" => Some('&'),
                        "lt" => Some('<'),
                        "gt" => Some('>'),
                        "quot" => Some('"'),
                        "apos" => Some('\''),
                        "nbsp" => Some('\u{a0}'),
                        _ => None,
                    },
                };
                if let Some(c) = resolved {
                    push_text(&mut stack, c.encode_utf8(&mut [0; 4]));
                }
            }
            Event::Eof => break,
            _ => {}
        }
    }

    // Unclosed elements at end of input are folded into their parents.
    while stack.len() > 1 {
        let node = stack.pop().unwrap_or_default();
        if let Some(top) = stack.last_mut() {
            top.children.push(XmlChild::Element(node));
        }
    }
    stack
        .pop()
        .and_then(|document| document.children.into_iter().find_map(|child| match child {
            XmlChild::Element(node) => Some(node),
            XmlChild::Text(_) => None,
        }))
        .ok_or_else(|| "no root element".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_elements_attributes_text_and_entities() {
        let xml = "\u{feff}<?xml version=\"1.0\"?><w:doc xmlns:w=\"x\"><w:p w:val=\"a &amp; b\">Tom &amp; Jerry &#233;<w:br/>&lt;end&gt;</w:p><!-- c --><w:p><![CDATA[raw <x>]]></w:p></w:doc>";
        let root = parse(xml).unwrap();
        assert_eq!(root.name, "w:doc");
        assert_eq!(root.local(), "doc");
        let paragraphs: Vec<&XmlNode> = root.elements().collect();
        assert_eq!(paragraphs.len(), 2);
        assert_eq!(paragraphs[0].attr("w:val"), Some("a & b"));
        assert_eq!(paragraphs[0].text(), "Tom & Jerry é<end>");
        assert_eq!(paragraphs[1].text(), "raw <x>");
        assert!(root.find("w:br").is_some());
        assert!(root.find_local("br").is_some());
        let mut all = Vec::new();
        root.find_all("w:p", &mut all);
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse("").is_err());
        assert!(parse("no markup").is_err());
        assert!(parse("</a>").is_err());
    }
}
