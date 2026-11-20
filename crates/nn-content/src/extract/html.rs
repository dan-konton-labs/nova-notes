//! HTML to Markdown, with main-content extraction.
//!
//! This is the built-in (`simple`) web engine's converter, also used for
//! local HTML files, EPUB chapters and pasted HTML. It
//!
//! * takes the title from `og:title`, `twitter:title`, `<title>` or the
//!   first `<h1>`, in that order;
//! * narrows the page to its main content (`<article>`, `<main>`,
//!   `[role=main]` and a few well-known content containers) when one holds a
//!   substantial share of the text;
//! * drops scripts, styles, navigation, sidebars, footers, short forms,
//!   hidden elements and common advertising / cookie / sharing blocks;
//! * keeps headings, paragraphs, lists (nested), links (resolved against the
//!   base URL), emphasis, inline code, fenced code blocks with their
//!   language, block quotes, tables and images that carry alt text.
//!
//! It does not run JavaScript: pages rendered client-side come out empty or
//! partial, which is what the remote engines are for.

use std::sync::OnceLock;

use regex::Regex;
use scraper::{ElementRef, Html, Node, Selector};
use url::Url;

use super::markdown;

/// Conversion settings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HtmlOptions {
    /// URL the document came from; relative links are resolved against it.
    pub base_url: Option<Url>,
    /// Narrow to the main content and drop page chrome (navigation, footers,
    /// sidebars, advertising). Scripts, styles and hidden elements are
    /// dropped either way.
    pub main_content: bool,
}

impl Default for HtmlOptions {
    fn default() -> Self {
        Self { base_url: None, main_content: true }
    }
}

/// A converted page.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HtmlPage {
    pub title: Option<String>,
    pub markdown: String,
    pub description: Option<String>,
    pub author: Option<String>,
    /// Publication time as the page states it (usually ISO 8601).
    pub published: Option<String>,
    /// The `lang` attribute of the document element.
    pub language: Option<String>,
    pub site_name: Option<String>,
}

/// Convert an HTML document or fragment to Markdown.
pub fn html_to_markdown(html: &str, options: &HtmlOptions) -> HtmlPage {
    let document = Html::parse_document(html);
    let base = effective_base(&document, options.base_url.as_ref());
    let converter = Converter { base: base.as_ref(), prune: options.main_content };

    let body = select_first(&document, "body").unwrap_or_else(|| document.root_element());
    let root = if options.main_content { main_content_root(&document, body) } else { body };

    let mut blocks = Vec::new();
    converter.render_children(root, &mut blocks);
    let markdown = blocks.join("\n\n").trim().to_string();

    let title = meta_content(&document, &["og:title", "twitter:title"])
        .or_else(|| {
            select_first(&document, "head > title")
                .map(|t| collapse(&t.text().collect::<String>()))
                .filter(|t| !t.is_empty())
        })
        .or_else(|| {
            select_first(&document, "h1").map(|h| collapse(&converter.plain_text(h))).filter(|t| !t.is_empty())
        });

    HtmlPage {
        title,
        markdown,
        description: meta_content(&document, &["description", "og:description", "twitter:description"]),
        author: meta_content(&document, &["author", "article:author", "parsely-author"]),
        published: meta_content(&document, &["article:published_time", "date", "pubdate", "dc.date"]),
        language: select_first(&document, "html")
            .and_then(|h| h.value().attr("lang"))
            .map(|l| l.trim().to_string())
            .filter(|l| !l.is_empty()),
        site_name: meta_content(&document, &["og:site_name", "application-name"]),
    }
}

fn select_first<'a>(document: &'a Html, selector: &str) -> Option<ElementRef<'a>> {
    let selector = Selector::parse(selector).ok()?;
    document.select(&selector).next()
}

/// First non-empty `content` of a `<meta>` whose `property` or `name` is one
/// of `keys` (earlier keys win).
fn meta_content(document: &Html, keys: &[&str]) -> Option<String> {
    static META: OnceLock<Selector> = OnceLock::new();
    let selector = META.get_or_init(|| Selector::parse("meta").expect("valid selector"));
    let metas: Vec<ElementRef> = document.select(selector).collect();
    keys.iter().find_map(|key| {
        metas.iter().find_map(|meta| {
            let element = meta.value();
            let name = element.attr("property").or_else(|| element.attr("name"))?;
            if !name.eq_ignore_ascii_case(key) {
                return None;
            }
            let content = collapse(element.attr("content")?);
            (!content.is_empty()).then_some(content)
        })
    })
}

fn effective_base(document: &Html, base_url: Option<&Url>) -> Option<Url> {
    let declared = select_first(document, "base[href]").and_then(|b| b.value().attr("href"));
    match (declared, base_url) {
        (Some(href), Some(base)) => base.join(href.trim()).ok().or_else(|| Some(base.clone())),
        (Some(href), None) => Url::parse(href.trim()).ok(),
        (None, base) => base.cloned(),
    }
}

fn collapse(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Elements whose content is never text for a reader.
fn is_non_content(name: &str) -> bool {
    matches!(
        name,
        "script"
            | "style"
            | "noscript"
            | "template"
            | "iframe"
            | "object"
            | "embed"
            | "svg"
            | "canvas"
            | "head"
            | "link"
            | "meta"
            | "title"
            | "select"
            | "option"
            | "datalist"
            | "button"
            | "input"
            | "textarea"
            | "dialog"
            | "audio"
            | "video"
            | "source"
            | "track"
            | "map"
            | "area"
            | "math"
    )
}

fn is_block(name: &str) -> bool {
    matches!(
        name,
        "address"
            | "article"
            | "aside"
            | "blockquote"
            | "body"
            | "caption"
            | "center"
            | "dd"
            | "details"
            | "div"
            | "dl"
            | "dt"
            | "fieldset"
            | "figcaption"
            | "figure"
            | "footer"
            | "form"
            | "h1"
            | "h2"
            | "h3"
            | "h4"
            | "h5"
            | "h6"
            | "header"
            | "hgroup"
            | "hr"
            | "html"
            | "li"
            | "main"
            | "nav"
            | "ol"
            | "p"
            | "pre"
            | "section"
            | "summary"
            | "table"
            | "tbody"
            | "td"
            | "tfoot"
            | "th"
            | "thead"
            | "tr"
            | "ul"
    )
}

/// Length of the text a reader would see under `element`.
fn visible_text_len(element: ElementRef) -> usize {
    let mut total = 0;
    for child in element.children() {
        match child.value() {
            Node::Text(text) => total += text.trim().len(),
            Node::Element(el) if !is_non_content(el.name()) => {
                if let Some(child) = ElementRef::wrap(child) {
                    total += visible_text_len(child);
                }
            }
            _ => {}
        }
    }
    total
}

/// Pick the element that holds the article: the smallest well-known content
/// container that still carries most of the text of the largest one. Falls
/// back to `body` when no container holds a substantial share of the page
/// (index pages, pages without semantic markup).
fn main_content_root<'a>(document: &'a Html, body: ElementRef<'a>) -> ElementRef<'a> {
    static CANDIDATES: OnceLock<Selector> = OnceLock::new();
    let selector = CANDIDATES.get_or_init(|| {
        Selector::parse(
            "article, main, [role=main], [itemprop=articleBody], #content, #main, #main-content, \
             #mw-content-text, .post-content, .entry-content, .article-content, .article-body, \
             .post-body, .markdown-body",
        )
        .expect("valid selector")
    });

    let body_len = visible_text_len(body);
    let floor = 250.max(body_len / 4);
    let candidates: Vec<(ElementRef, usize)> =
        document.select(selector).map(|el| (el, visible_text_len(el))).filter(|(_, len)| *len >= floor).collect();
    let Some(largest) = candidates.iter().map(|(_, len)| *len).max() else {
        return body;
    };
    candidates
        .into_iter()
        .filter(|(_, len)| *len * 10 >= largest * 6)
        .min_by_key(|(_, len)| *len)
        .map_or(body, |(el, _)| el)
}

fn chrome_token_pattern() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"(?x)^(
                ad|ads|advert|advertisement|adsbygoogle|ad-(slot|unit|container|banner|wrapper)|sponsored
                |cookie-(banner|consent|notice|bar)|gdpr
                |newsletter(-signup|-form)?|subscribe-form|paywall
                |social(-share|-links|-media|-icons)?|share(-buttons|-bar|-tools|-links)?|sharing
                |sidebar|side-bar|breadcrumb|breadcrumbs|navbar|nav-bar|navigation
                |site-(header|footer|nav)|skip-link|skip-to-content
                |screen-reader-text|sr-only|visually-hidden
                |related(-posts|-articles|-content|-links)?|popular-posts
                |comments|comment-list|comment-respond|comments-area|disqus-thread
                |mw-editsection|mw-jump-link|navbox|catlinks|printfooter|noprint
            )$",
        )
        .expect("valid regex")
    })
}

struct Converter<'a> {
    base: Option<&'a Url>,
    prune: bool,
}

impl Converter<'_> {
    fn skip(&self, element: ElementRef) -> bool {
        let el = element.value();
        let name = el.name();
        if is_non_content(name) {
            return true;
        }
        if el.attr("hidden").is_some() || el.attr("aria-hidden").is_some_and(|v| v.eq_ignore_ascii_case("true")) {
            return true;
        }
        if let Some(style) = el.attr("style") {
            let style: String = style.chars().filter(|c| !c.is_whitespace()).collect::<String>().to_ascii_lowercase();
            if style.contains("display:none") || style.contains("visibility:hidden") {
                return true;
            }
        }
        if !self.prune || matches!(name, "html" | "body" | "main" | "article") {
            return false;
        }

        match name {
            "nav" | "aside" | "footer" => return true,
            "header" => {
                static NAV: OnceLock<Selector> = OnceLock::new();
                let nav = NAV.get_or_init(|| Selector::parse("nav").expect("valid selector"));
                if element.select(nav).next().is_some() {
                    return true;
                }
            }
            // Search, login and newsletter forms are short; some sites wrap
            // the whole page in one form, which must survive.
            "form" if visible_text_len(element) < 2000 => return true,
            _ => {}
        }
        if let Some(role) = el.attr("role") {
            if matches!(
                role.to_ascii_lowercase().as_str(),
                "navigation"
                    | "banner"
                    | "contentinfo"
                    | "complementary"
                    | "search"
                    | "dialog"
                    | "alertdialog"
                    | "menu"
                    | "menubar"
                    | "toolbar"
            ) {
                return true;
            }
        }
        let pattern = chrome_token_pattern();
        el.classes().chain(el.attr("id")).any(|token| pattern.is_match(&token.to_ascii_lowercase().replace('_', "-")))
    }

    /// Render the children of `parent` as a sequence of Markdown blocks.
    fn render_children(&self, parent: ElementRef, blocks: &mut Vec<String>) {
        let mut inline = String::new();
        for child in parent.children() {
            match child.value() {
                Node::Text(text) => push_text(&mut inline, text),
                Node::Element(_) => {
                    let Some(element) = ElementRef::wrap(child) else { continue };
                    if self.skip(element) {
                        continue;
                    }
                    let name = element.value().name();
                    if is_block(name) {
                        flush_inline(&mut inline, blocks);
                        self.render_block(element, blocks);
                    } else if contains_block(element) {
                        flush_inline(&mut inline, blocks);
                        self.render_children(element, blocks);
                    } else {
                        self.render_inline(element, &mut inline, false);
                    }
                }
                _ => {}
            }
        }
        flush_inline(&mut inline, blocks);
    }

    fn render_block(&self, element: ElementRef, blocks: &mut Vec<String>) {
        let name = element.value().name();
        match name {
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                let level = usize::from(name.as_bytes()[1] - b'0');
                let text = collapse(&self.plain_text(element));
                if !text.is_empty() {
                    blocks.push(format!("{} {text}", "#".repeat(level)));
                }
            }
            "p" | "summary" | "figcaption" | "caption" | "address" => {
                if contains_block(element) {
                    self.render_children(element, blocks);
                } else {
                    let mut inline = String::new();
                    self.inline_children(element, &mut inline, false);
                    flush_inline(&mut inline, blocks);
                }
            }
            "dt" => {
                let text = collapse(&self.plain_text(element));
                if !text.is_empty() {
                    blocks.push(format!("**{text}**"));
                }
            }
            "hr" => blocks.push("---".to_string()),
            "pre" => {
                if let Some(code) = self.code_block(element) {
                    blocks.push(code);
                }
            }
            "blockquote" => {
                let mut inner = Vec::new();
                self.render_children(element, &mut inner);
                if !inner.is_empty() {
                    let quoted: Vec<String> = inner
                        .join("\n\n")
                        .lines()
                        .map(|line| if line.is_empty() { ">".to_string() } else { format!("> {line}") })
                        .collect();
                    blocks.push(quoted.join("\n"));
                }
            }
            "ul" | "ol" => {
                let list = self.list(element, name == "ol");
                if !list.is_empty() {
                    blocks.push(list);
                }
            }
            "table" => self.table(element, blocks),
            _ => self.render_children(element, blocks),
        }
    }

    fn list(&self, list: ElementRef, ordered: bool) -> String {
        let mut number: i64 = list.value().attr("start").and_then(|s| s.trim().parse().ok()).unwrap_or(1);
        let mut lines: Vec<String> = Vec::new();
        for item in list.child_elements() {
            if self.skip(item) {
                continue;
            }
            let mut blocks = Vec::new();
            if item.value().name() == "li" {
                self.render_children(item, &mut blocks);
            } else {
                // Stray non-<li> children still carry content.
                self.render_block(item, &mut blocks);
            }
            if blocks.is_empty() {
                continue;
            }
            let marker = if ordered { format!("{number}. ") } else { "- ".to_string() };
            number += 1;
            let indent = " ".repeat(marker.len());
            for (index, line) in blocks.join("\n").lines().enumerate() {
                if index == 0 {
                    lines.push(format!("{marker}{line}"));
                } else if line.is_empty() {
                    lines.push(String::new());
                } else {
                    lines.push(format!("{indent}{line}"));
                }
            }
        }
        lines.join("\n")
    }

    fn code_block(&self, pre: ElementRef) -> Option<String> {
        let mut text = String::new();
        raw_text(pre, &mut text);
        let text = text.trim_matches(|c| c == '\n' || c == '\r').trim_end();
        if text.trim().is_empty() {
            return None;
        }
        static CODE: OnceLock<Selector> = OnceLock::new();
        let code = CODE.get_or_init(|| Selector::parse("code").expect("valid selector"));
        let language = code_language(pre).or_else(|| pre.select(code).next().and_then(code_language)).unwrap_or_default();
        let fence = if text.contains("```") { "````" } else { "```" };
        Some(format!("{fence}{language}\n{text}\n{fence}"))
    }

    fn table(&self, table: ElementRef, blocks: &mut Vec<String>) {
        let mut caption = None;
        let mut rows: Vec<ElementRef> = Vec::new();
        for child in table.child_elements() {
            match child.value().name() {
                "caption" => caption = Some(collapse(&self.plain_text(child))).filter(|c: &String| !c.is_empty()),
                "tr" => rows.push(child),
                "thead" | "tbody" | "tfoot" => {
                    rows.extend(child.child_elements().filter(|r| r.value().name() == "tr"));
                }
                _ => {}
            }
        }

        static NESTED: OnceLock<Selector> = OnceLock::new();
        let nested = NESTED.get_or_init(|| Selector::parse("table").expect("valid selector"));
        fn cells_of<'a>(row: ElementRef<'a>) -> Vec<ElementRef<'a>> {
            row.child_elements().filter(|c| matches!(c.value().name(), "td" | "th")).collect()
        }
        let widest = rows.iter().map(|row| cells_of(*row).len()).max().unwrap_or(0);

        // Tables used for page layout (nested tables, a single column) read
        // better as plain blocks than as a grid.
        if widest <= 1 || table.select(nested).next().is_some() {
            if let Some(caption) = caption {
                blocks.push(caption);
            }
            for row in rows {
                for cell in cells_of(row) {
                    if !self.skip(cell) {
                        self.render_children(cell, blocks);
                    }
                }
            }
            return;
        }

        let mut grid: Vec<Vec<String>> = Vec::new();
        for row in rows {
            let mut line = Vec::new();
            for cell in cells_of(row) {
                let mut cell_blocks = Vec::new();
                if !self.skip(cell) {
                    self.render_children(cell, &mut cell_blocks);
                }
                line.push(cell_blocks.join(" "));
                let span: usize = cell.value().attr("colspan").and_then(|s| s.trim().parse().ok()).unwrap_or(1);
                for _ in 1..span.min(50) {
                    line.push(String::new());
                }
            }
            if line.iter().any(|c| !c.trim().is_empty()) {
                grid.push(line);
            }
        }
        if grid.is_empty() {
            return;
        }
        if let Some(caption) = caption {
            blocks.push(format!("**{caption}**"));
        }
        blocks.push(markdown::table(&grid));
    }

    /// Text of `element` with whitespace collapsed by the caller; links and
    /// emphasis are flattened to their text.
    fn plain_text(&self, element: ElementRef) -> String {
        let mut out = String::new();
        self.inline_children(element, &mut out, true);
        out
    }

    fn inline_children(&self, parent: ElementRef, out: &mut String, plain: bool) {
        for child in parent.children() {
            match child.value() {
                Node::Text(text) => push_text(out, text),
                Node::Element(_) => {
                    if let Some(element) = ElementRef::wrap(child) {
                        if !self.skip(element) {
                            self.render_inline(element, out, plain);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn render_inline(&self, element: ElementRef, out: &mut String, plain: bool) {
        let name = element.value().name();
        match name {
            "br" => {
                trim_trailing_spaces(out);
                if !out.is_empty() {
                    out.push('\n');
                }
            }
            "img" => {
                if plain {
                    return;
                }
                let alt = collapse(element.value().attr("alt").unwrap_or(""));
                let source = element.value().attr("src").or_else(|| element.value().attr("data-src"));
                if let (false, Some(source)) = (alt.is_empty(), source.and_then(|s| self.resolve(s))) {
                    push_spaced(out, &format!("![{alt}]({source})"), false, false);
                }
            }
            "a" => {
                let mut inner = String::new();
                self.inline_children(element, &mut inner, plain);
                let (lead, trail) = (leading_space(element), inner.ends_with(' '));
                let text = inner.trim();
                if text.is_empty() {
                    return;
                }
                if plain {
                    // Permalink markers next to headings are not part of the title.
                    if !matches!(text, "¶" | "#" | "§" | "🔗") {
                        push_spaced(out, text, lead, trail);
                    }
                    return;
                }
                match element.value().attr("href").and_then(|href| self.resolve(href)) {
                    Some(href) if !text.contains('\n') => {
                        push_spaced(out, &format!("[{text}]({href})"), lead, trail);
                    }
                    _ => push_spaced(out, text, lead, trail),
                }
            }
            "strong" | "b" => self.wrapped(element, out, "**", plain),
            "em" | "i" | "cite" | "dfn" => self.wrapped(element, out, "*", plain),
            "del" | "s" | "strike" => self.wrapped(element, out, "~~", plain),
            "code" | "kbd" | "samp" | "tt" => {
                let text = collapse(&element.text().collect::<String>());
                if text.is_empty() {
                    return;
                }
                if plain {
                    push_spaced(out, &text, false, false);
                } else if text.contains('`') {
                    push_spaced(out, &format!("`` {text} ``"), false, false);
                } else {
                    push_spaced(out, &format!("`{text}`"), false, false);
                }
            }
            "q" => {
                let mut inner = String::new();
                self.inline_children(element, &mut inner, plain);
                let (lead, trail) = (leading_space(element), inner.ends_with(' '));
                if !inner.trim().is_empty() {
                    push_spaced(out, &format!("\"{}\"", inner.trim()), lead, trail);
                }
            }
            _ => {
                // Block elements reached in inline context (invalid nesting)
                // are flattened, separated from their neighbours by a space.
                let block = is_block(name);
                if block {
                    push_text(out, " ");
                }
                self.inline_children(element, out, plain);
                if block {
                    push_text(out, " ");
                }
            }
        }
    }

    fn wrapped(&self, element: ElementRef, out: &mut String, marker: &str, plain: bool) {
        let mut inner = String::new();
        self.inline_children(element, &mut inner, plain);
        let (lead, trail) = (leading_space(element), inner.ends_with(' '));
        let text = inner.trim();
        if text.is_empty() {
            return;
        }
        if plain || text.contains('\n') || text.starts_with(marker) {
            push_spaced(out, text, lead, trail);
        } else {
            push_spaced(out, &format!("{marker}{text}{marker}"), lead, trail);
        }
    }

    /// Absolute form of a link target. `None` for targets that are not
    /// useful outside the page (fragments, script and data URLs).
    fn resolve(&self, href: &str) -> Option<String> {
        let href = href.trim();
        if href.is_empty() || href.starts_with('#') {
            return None;
        }
        let lowered = href.to_ascii_lowercase();
        if ["javascript:", "data:", "vbscript:", "blob:", "about:"].iter().any(|scheme| lowered.starts_with(scheme)) {
            return None;
        }
        let resolved = match self.base {
            Some(base) => base.join(href).map(String::from).unwrap_or_else(|_| href.to_string()),
            None => href.to_string(),
        };
        Some(resolved.replace(' ', "%20").replace('(', "%28").replace(')', "%29"))
    }
}

fn contains_block(element: ElementRef) -> bool {
    element.descendent_elements().skip(1).any(|d| is_block(d.value().name()))
}

/// Append text in HTML's collapsed-whitespace sense.
fn push_text(out: &mut String, text: &str) {
    for c in text.chars() {
        if c.is_whitespace() {
            if !out.is_empty() && !out.ends_with(' ') && !out.ends_with('\n') {
                out.push(' ');
            }
        } else {
            out.push(c);
        }
    }
}

/// Whether the first text inside `element` starts with whitespace. The
/// inline buffer cannot tell: it never starts with a space.
fn leading_space(element: ElementRef) -> bool {
    element.text().next().is_some_and(|t| t.starts_with(char::is_whitespace))
}

/// Append an already-formatted inline fragment, restoring the spaces that
/// surrounded it inside its element.
fn push_spaced(out: &mut String, fragment: &str, lead: bool, trail: bool) {
    if lead {
        push_text(out, " ");
    }
    out.push_str(fragment);
    if trail {
        out.push(' ');
    }
}

fn trim_trailing_spaces(out: &mut String) {
    while out.ends_with(' ') {
        out.pop();
    }
}

fn flush_inline(inline: &mut String, blocks: &mut Vec<String>) {
    let text: Vec<&str> = inline.lines().map(str::trim).collect();
    let text = text.join("\n");
    let text = text.trim();
    if !text.is_empty() {
        blocks.push(text.to_string());
    }
    inline.clear();
}

/// Text of a `<pre>` as written, with `<br>` as line breaks.
fn raw_text(element: ElementRef, out: &mut String) {
    for child in element.children() {
        match child.value() {
            Node::Text(text) => out.push_str(text),
            Node::Element(el) => {
                if el.name() == "br" {
                    out.push('\n');
                } else if !is_non_content(el.name()) {
                    if let Some(child) = ElementRef::wrap(child) {
                        raw_text(child, out);
                    }
                }
            }
            _ => {}
        }
    }
}

fn code_language(element: ElementRef) -> Option<String> {
    let el = element.value();
    if let Some(lang) = el.attr("data-lang").or_else(|| el.attr("data-language")) {
        let lang = lang.trim();
        if !lang.is_empty() {
            return Some(lang.to_string());
        }
    }
    el.classes().find_map(|class| {
        let lang = class.strip_prefix("language-").or_else(|| class.strip_prefix("lang-"))?;
        (!lang.is_empty() && lang.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '#' | '_')))
            .then(|| lang.to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn convert(html: &str) -> String {
        html_to_markdown(html, &HtmlOptions::default()).markdown
    }

    fn with_base(html: &str, base: &str) -> HtmlPage {
        html_to_markdown(html, &HtmlOptions { base_url: Some(Url::parse(base).unwrap()), main_content: true })
    }

    const ARTICLE: &str = r##"<!DOCTYPE html>
<html lang="en-GB">
<head>
  <meta charset="utf-8">
  <title>How Tides Work | Example Science</title>
  <meta property="og:title" content="How Tides Work">
  <meta name="description" content="A short explanation of tides.">
  <meta name="author" content="Ada Marine">
  <meta property="article:published_time" content="2024-03-05T10:00:00Z">
  <meta property="og:site_name" content="Example Science">
  <style>body { color: red }</style>
  <script>window.tracking = "do not include";</script>
</head>
<body class="has-sidebar">
  <header class="site-header"><nav><a href="/">Home</a> <a href="/about">About</a></nav></header>
  <div class="cookie-banner">We use cookies. <button>Accept</button></div>
  <main>
    <article>
      <h1>How Tides Work <a class="headerlink" href="#top">¶</a></h1>
      <p class="byline">By <a href="/authors/ada">Ada Marine</a></p>
      <p>Tides are caused by the <strong>gravitational pull</strong> of the
         <em>Moon</em> and, to a lesser degree, the Sun. See the
         <a href="https://example.org/tides?a=1&amp;b=2">full reference</a>.</p>
      <div class="advertisement"><p>Buy our telescope!</p></div>
      <h2>Kinds of tide</h2>
      <ul>
        <li>Spring tides
          <ul><li>Around full and new moon</li><li>Largest range</li></ul>
        </li>
        <li>Neap tides</li>
      </ul>
      <ol start="3"><li>Third</li><li>Fourth</li></ol>
      <blockquote><p>Time and tide wait for no man.</p><p>Proverb</p></blockquote>
      <pre><code class="language-python">def height(t):
    return 2 * cos(t)  # metres
</code></pre>
      <p>Call <code>height(0)</code> for high water.<br>Units are metres.</p>
      <table>
        <thead><tr><th>Place</th><th>Range (m)</th></tr></thead>
        <tbody>
          <tr><td>Bay of Fundy</td><td>16.3</td></tr>
          <tr><td>Bristol <b>Channel</b></td><td>15|0</td></tr>
        </tbody>
      </table>
      <figure><img src="/img/tide.png" alt="Tide diagram"><figcaption>Figure 1: a tide.</figcaption></figure>
      <img src="/img/spacer.gif">
      <hr>
      <p style="display: none">Hidden paragraph</p>
      <p hidden>Also hidden</p>
    </article>
    <aside><h2>Related</h2><p>Unrelated teaser</p></aside>
    <div class="comments"><p>First!</p></div>
  </main>
  <footer><p>Copyright Example Science</p></footer>
</body>
</html>"##;

    #[test]
    fn article_page_converts_to_clean_markdown() {
        let page = with_base(ARTICLE, "https://science.example.com/articles/tides");
        let expected = "\
# How Tides Work

By [Ada Marine](https://science.example.com/authors/ada)

Tides are caused by the **gravitational pull** of the *Moon* and, to a lesser degree, the Sun. See the [full reference](https://example.org/tides?a=1&b=2).

## Kinds of tide

- Spring tides
  - Around full and new moon
  - Largest range
- Neap tides

3. Third
4. Fourth

> Time and tide wait for no man.
>
> Proverb

```python
def height(t):
    return 2 * cos(t)  # metres
```

Call `height(0)` for high water.
Units are metres.

| Place | Range (m) |
| --- | --- |
| Bay of Fundy | 16.3 |
| Bristol **Channel** | 15\\|0 |

![Tide diagram](https://science.example.com/img/tide.png)

Figure 1: a tide.

---";
        assert_eq!(page.markdown, expected);
        assert_eq!(page.title.as_deref(), Some("How Tides Work"));
        assert_eq!(page.description.as_deref(), Some("A short explanation of tides."));
        assert_eq!(page.author.as_deref(), Some("Ada Marine"));
        assert_eq!(page.published.as_deref(), Some("2024-03-05T10:00:00Z"));
        assert_eq!(page.language.as_deref(), Some("en-GB"));
        assert_eq!(page.site_name.as_deref(), Some("Example Science"));
    }

    #[test]
    fn page_chrome_is_removed() {
        let markdown = with_base(ARTICLE, "https://science.example.com/a").markdown;
        for unwanted in [
            "Home", "About", "cookies", "Accept", "telescope", "Unrelated teaser", "First!", "Copyright",
            "tracking", "color: red", "Hidden paragraph", "Also hidden", "spacer", "¶",
        ] {
            assert!(!markdown.contains(unwanted), "{unwanted:?} leaked into:\n{markdown}");
        }
    }

    #[test]
    fn title_fallback_order() {
        let og = r#"<html><head><title>T</title><meta property="og:title" content=" OG  Title "></head><body><h1>H</h1></body></html>"#;
        assert_eq!(html_to_markdown(og, &HtmlOptions::default()).title.as_deref(), Some("OG Title"));
        let title = "<html><head><title>\n Plain   Title \n</title></head><body><h1>H</h1></body></html>";
        assert_eq!(html_to_markdown(title, &HtmlOptions::default()).title.as_deref(), Some("Plain Title"));
        let h1 = "<html><body><p>x</p><h1>First <em>Heading</em></h1></body></html>";
        assert_eq!(html_to_markdown(h1, &HtmlOptions::default()).title.as_deref(), Some("First Heading"));
        assert_eq!(html_to_markdown("<p>nothing</p>", &HtmlOptions::default()).title, None);
        // An SVG <title> in the body is not the page title.
        let svg = "<html><body><svg><title>icon</title></svg><h1>Real</h1></body></html>";
        assert_eq!(html_to_markdown(svg, &HtmlOptions::default()).title.as_deref(), Some("Real"));
    }

    #[test]
    fn body_is_used_when_no_container_dominates() {
        let html = format!(
            "<html><body><div><h1>Index</h1><p>{}</p></div><article><p>Short teaser one.</p></article><article><p>Short teaser two.</p></article></body></html>",
            "Long introduction text. ".repeat(40)
        );
        let markdown = convert(&html);
        assert!(markdown.starts_with("# Index"));
        assert!(markdown.contains("Short teaser one."));
        assert!(markdown.contains("Short teaser two."));
    }

    #[test]
    fn smallest_dominant_container_wins() {
        let body = "Article sentence with real content. ".repeat(30);
        let html = format!(
            "<html><body><div id=\"content\"><div class=\"promo-strip\"><p>Subscribe today for more.</p></div><article><h1>Story</h1><p>{body}</p></article></div></body></html>"
        );
        let markdown = convert(&html);
        assert!(markdown.starts_with("# Story"));
        assert!(!markdown.contains("Subscribe today"));
    }

    #[test]
    fn whole_document_mode_keeps_everything_but_scripts() {
        let html = "<html><body><nav><a href=\"/x\">Menu item</a></nav><p>Body</p><footer>Foot</footer><script>var a;</script></body></html>";
        let page = html_to_markdown(html, &HtmlOptions { base_url: None, main_content: false });
        assert_eq!(page.markdown, "[Menu item](/x)\n\nBody\n\nFoot");
    }

    #[test]
    fn links_are_resolved_and_unsafe_targets_dropped() {
        let html = r##"<p><a href="../up">Up</a> <a href="//cdn.example.com/x">CDN</a> <a href="#frag">Frag</a>
            <a href="javascript:void(0)">Script</a> <a href="mailto:a@example.com">Mail</a>
            <a href="/path with space/(1)">Odd</a><a href="/empty"></a></p>"##;
        let page = with_base(html, "https://example.com/a/b/c");
        assert_eq!(
            page.markdown,
            "[Up](https://example.com/a/up) [CDN](https://cdn.example.com/x) Frag Script [Mail](mailto:a@example.com) [Odd](https://example.com/path%20with%20space/%281%29)"
        );
    }

    #[test]
    fn base_element_is_honoured() {
        let html = r#"<html><head><base href="/docs/"></head><body><p><a href="page">P</a></p></body></html>"#;
        assert_eq!(with_base(html, "https://example.com/x/y").markdown, "[P](https://example.com/docs/page)");
    }

    #[test]
    fn inline_whitespace_is_collapsed_and_preserved_around_markup() {
        assert_eq!(convert("<p>a<b> bold </b>c  and\n\n <i>it</i>.</p>"), "a **bold** c and *it*.");
        assert_eq!(convert("<p>one<span>two</span> three</p>"), "onetwo three");
        assert_eq!(convert("<div>loose text <span>inline</span><p>para</p>tail</div>"), "loose text inline\n\npara\n\ntail");
        assert_eq!(convert("<p>non&nbsp;breaking&amp;entity &lt;tag&gt;</p>"), "non breaking&entity <tag>");
        assert_eq!(convert("<p><b></b><a href=\"/x\"> </a>text</p>"), "text");
    }

    #[test]
    fn code_blocks_keep_formatting() {
        let html = "<pre>line 1\n  indented\n\n<span class=\"k\">fn</span> main() {}</pre>";
        assert_eq!(convert(html), "```\nline 1\n  indented\n\nfn main() {}\n```");
        let lang = "<pre class=\"lang-rust\"><code>let a = \"```\";</code></pre>";
        assert_eq!(convert(lang), "````rust\nlet a = \"```\";\n````");
        let highlighted = "<div class=\"highlight\"><pre><code data-lang=\"sh\">ls<br>pwd</code></pre></div>";
        assert_eq!(convert(highlighted), "```sh\nls\npwd\n```");
        assert_eq!(convert("<p>Use <code>a `tick`</code>.</p>"), "Use `` a `tick` ``.");
    }

    #[test]
    fn tables_headers_colspans_and_layout_tables() {
        let no_header = "<table><tr><td>a</td><td>b</td></tr><tr><td>c</td><td>d</td></tr></table>";
        assert_eq!(convert(no_header), "| a | b |\n| --- | --- |\n| c | d |");

        let span = "<table><caption>Scores</caption><tr><th colspan=\"2\">Name</th><th>N</th></tr><tr><td>x</td><td>y</td><td><p>1</p><p>2</p></td></tr></table>";
        assert_eq!(convert(span), "**Scores**\n\n| Name |  | N |\n| --- | --- | --- |\n| x | y | 1 2 |");

        let layout = "<table><tr><td><h2>Title</h2><p>Text</p></td></tr><tr><td><p>More</p></td></tr></table>";
        assert_eq!(convert(layout), "## Title\n\nText\n\nMore");

        let nested = "<table><tr><td><table><tr><td>in</td><td>ner</td></tr></table></td><td>right</td></tr></table>";
        assert_eq!(convert(nested), "| in | ner |\n| --- | --- |\n\nright");
    }

    #[test]
    fn lists_with_paragraphs_and_definition_lists() {
        let html = "<ul><li><p>First para</p><p>Second para</p></li><li>Plain <a href=\"https://e.com\">link</a></li><li></li></ul>";
        assert_eq!(convert(html), "- First para\n  Second para\n- Plain [link](https://e.com)");
        let dl = "<dl><dt>Term</dt><dd>Definition text</dd></dl>";
        assert_eq!(convert(dl), "**Term**\n\nDefinition text");
        let ordered = "<ol><li>One<ol><li>Nested</li></ol></li><li>Two</li></ol>";
        assert_eq!(convert(ordered), "1. One\n   1. Nested\n2. Two");
    }

    #[test]
    fn page_wrapping_form_is_kept_but_small_forms_are_dropped() {
        let text = "Meaningful sentence inside the page form. ".repeat(60);
        let html = format!(
            "<html><body><form id=\"aspnetForm\"><input type=\"hidden\" value=\"state\"><h1>Report</h1><p>{text}</p></form><form><label>Email</label><input><button>Join</button></form></body></html>"
        );
        let markdown = convert(&html);
        assert!(markdown.starts_with("# Report"));
        assert!(markdown.contains("Meaningful sentence"));
        assert!(!markdown.contains("Email"));
        assert!(!markdown.contains("state"));
    }

    #[test]
    fn malformed_html_does_not_panic() {
        assert_eq!(convert("<p>unclosed <b>bold <i>both</p><p>next"), "unclosed **bold *both***\n\n***next***");
        assert_eq!(convert(""), "");
        assert_eq!(convert("just text"), "just text");
        assert_eq!(convert("<h2></h2><ul></ul><table></table><pre>  </pre>"), "");
    }
}
