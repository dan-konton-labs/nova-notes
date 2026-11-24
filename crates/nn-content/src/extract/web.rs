//! The URL flow for everything that is not YouTube: pick the engines, route
//! documents and media to the file flow, convert pages to Markdown.

use std::io::Write;
use std::sync::OnceLock;

use regex::bytes::Regex;
use serde_json::{json, Map, Value};
use url::Url;

use super::detect::{self, FileKind};
use super::engines::{self, EnginePage};
use super::fetch::{self, Fetched};
use super::html::{html_to_markdown_blocking, HtmlOptions};
use super::{documents, files, ExtractContext, Extracted, UrlEngine};
use crate::error::ContentError;

const PAGE_ACCEPT: &str = "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8";

/// Extensions that mean "this URL is a file, not a page".
const FILE_EXTENSIONS: &[&str] = &[
    "pdf", "docx", "pptx", "xlsx", "xlsm", "xlsb", "xls", "ods", "odt", "odp", "epub", "csv", "tsv", "txt", "md",
    "markdown", "json", "yaml", "yml", "doc", "ppt", "mp3", "wav", "m4a", "aac", "ogg", "oga", "opus", "flac", "wma",
    "mp4", "m4v", "mov", "mkv", "webm", "avi", "wmv",
];

pub(crate) async fn extract_web(url: &Url, ctx: &ExtractContext) -> Result<Extracted, ContentError> {
    if detect::extension_of(url.path()).is_some_and(|ext| FILE_EXTENSIONS.contains(&ext.as_str())) {
        let fetched = fetch::get(url, ctx, "*/*").await?;
        return extract_fetched(fetched, ctx).await;
    }

    let chain = engine_chain(ctx)?;
    // A remote engine would be handed the document as if it were a page; ask
    // the server what it is first. The built-in engine finds out from its GET.
    if chain.first() != Some(&UrlEngine::Simple) {
        if let Some(content_type) = fetch::head_content_type(url, ctx).await? {
            if is_file_media_type(&content_type) {
                let fetched = fetch::get(url, ctx, "*/*").await?;
                return extract_fetched(fetched, ctx).await;
            }
        }
    }

    let mut last_error = None;
    for engine in &chain {
        let attempt = match engine {
            UrlEngine::Firecrawl => engines::firecrawl(url, ctx).await.map(|page| engine_result(page, *engine)),
            UrlEngine::Jina => engines::jina(url, ctx).await.map(|page| engine_result(page, *engine)),
            UrlEngine::Crawl4ai => engines::crawl4ai(url, ctx).await.map(|page| engine_result(page, *engine)),
            UrlEngine::Simple | UrlEngine::Auto => match fetch::get(url, ctx, PAGE_ACCEPT).await {
                Ok(fetched) => extract_fetched(fetched, ctx).await,
                Err(error) => Err(error),
            },
        };
        match attempt {
            Ok(extracted) if !extracted.content.trim().is_empty() => return Ok(extracted),
            Ok(_) => {
                tracing::warn!(engine = %engine, "URL engine returned no content");
                last_error = Some(ContentError::EmptyContent(format!("the {engine} engine found no text on the page")));
            }
            // The guard's verdict does not depend on the engine.
            Err(error @ ContentError::UrlNotAllowed(_)) => return Err(error),
            Err(error) => {
                tracing::warn!(engine = %engine, "URL engine failed: {error}");
                last_error = Some(error);
            }
        }
    }
    Err(last_error.unwrap_or_else(|| ContentError::Configuration("no URL engine is available".to_string())))
}

/// The concrete engines to try, in order.
fn engine_chain(ctx: &ExtractContext) -> Result<Vec<UrlEngine>, ContentError> {
    let has = |value: &Option<String>| value.as_deref().is_some_and(|v| !v.trim().is_empty());
    let auto = || {
        let mut chain = Vec::new();
        if has(&ctx.firecrawl_api_key) {
            chain.push(UrlEngine::Firecrawl);
        }
        if has(&ctx.jina_api_key) {
            chain.push(UrlEngine::Jina);
        }
        if has(&ctx.crawl4ai_api_url) {
            chain.push(UrlEngine::Crawl4ai);
        }
        chain.push(UrlEngine::Simple);
        chain
    };
    match ctx.url_engine {
        UrlEngine::Auto => Ok(auto()),
        UrlEngine::Simple => Ok(vec![UrlEngine::Simple]),
        UrlEngine::Jina => Ok(vec![UrlEngine::Jina]),
        UrlEngine::Firecrawl => {
            let self_hosted = ctx.firecrawl_api_url.trim_end_matches('/') != super::DEFAULT_FIRECRAWL_API_URL;
            if has(&ctx.firecrawl_api_key) || self_hosted {
                Ok(vec![UrlEngine::Firecrawl])
            } else {
                Err(ContentError::Configuration(
                    "the Firecrawl engine is selected but no Firecrawl API key is configured".to_string(),
                ))
            }
        }
        // A stored choice can outlive the server it pointed at; the original
        // falls back to `auto` in that case instead of failing every source.
        UrlEngine::Crawl4ai if !has(&ctx.crawl4ai_api_url) => {
            tracing::warn!("the crawl4ai URL engine is selected but no Crawl4AI server is configured; using auto");
            Ok(auto())
        }
        UrlEngine::Crawl4ai => Ok(vec![UrlEngine::Crawl4ai]),
    }
}

fn is_file_media_type(media_type: &str) -> bool {
    media_type.starts_with("audio/")
        || media_type.starts_with("video/")
        || matches!(
            media_type,
            "application/pdf" | "application/epub+zip" | "application/vnd.ms-excel" | "text/csv" | "application/msword"
        )
        || media_type.starts_with("application/vnd.openxmlformats-officedocument.")
        || media_type.starts_with("application/vnd.oasis.opendocument.")
}

fn engine_result(page: EnginePage, engine: UrlEngine) -> Extracted {
    let mut metadata = page.metadata;
    metadata.insert("engine".to_string(), json!(engine.as_str()));
    Extracted {
        title: page.title,
        content: page.content,
        identified_type: "url".to_string(),
        url: None,
        file_path: None,
        metadata: Value::Object(metadata),
    }
}

/// Turn a downloaded body into text: HTML is converted here, anything else
/// goes through a temporary file into the file flow.
async fn extract_fetched(fetched: Fetched, ctx: &ExtractContext) -> Result<Extracted, ContentError> {
    let name = fetched
        .file_name
        .clone()
        .or_else(|| fetched.final_url.path_segments().and_then(|mut s| s.next_back().map(str::to_string)))
        .filter(|n| !n.is_empty());
    let extension = name.as_deref().and_then(detect::extension_of);
    let media_type = fetched.content_type.as_deref();

    let declared_html = media_type.is_some_and(|m| m.contains("html"));
    let is_html = declared_html && !fetched.body.starts_with(b"%PDF-") || {
        let head = &fetched.body[..fetched.body.len().min(8192)];
        let sniffed = detect::detect(head, || None, extension.as_deref(), media_type);
        sniffed.kind == FileKind::Html
    };

    if is_html {
        let charset = fetched.charset.clone().or_else(|| meta_charset(&fetched.body));
        let html = documents::decode_with_charset(&fetched.body, charset.as_deref());
        let options = HtmlOptions { base_url: Some(fetched.final_url.clone()), main_content: true };
        let page = html_to_markdown_blocking(html, options).await?;

        let mut metadata = Map::new();
        metadata.insert("engine".to_string(), json!("simple"));
        metadata.insert("mime_type".to_string(), json!(media_type.unwrap_or("text/html")));
        metadata.insert("final_url".to_string(), json!(fetched.final_url.as_str()));
        for (key, value) in [
            ("description", page.description),
            ("author", page.author),
            ("published", page.published),
            ("language", page.language),
            ("site_name", page.site_name),
        ] {
            if let Some(value) = value {
                metadata.insert(key.to_string(), json!(value));
            }
        }
        return Ok(Extracted {
            title: page.title,
            content: page.markdown,
            identified_type: "url".to_string(),
            url: None,
            file_path: None,
            metadata: Value::Object(metadata),
        });
    }

    let suffix = extension.as_deref().map(|e| format!(".{e}")).unwrap_or_default();
    let body = fetched.body.clone();
    let temp = tokio::task::spawn_blocking(move || -> std::io::Result<tempfile::NamedTempFile> {
        let mut file = tempfile::Builder::new().prefix("nn-content-").suffix(&suffix).tempfile()?;
        file.write_all(&body)?;
        file.flush()?;
        Ok(file)
    })
    .await
    .map_err(|e| ContentError::FileOperation(format!("download task failed: {e}")))?
    .map_err(|e| ContentError::FileOperation(format!("cannot store the download in a temporary file: {e}")))?;

    let mut extracted = files::extract_file(temp.path(), name.as_deref(), media_type, ctx).await?;
    if let Value::Object(metadata) = &mut extracted.metadata {
        metadata.insert("final_url".to_string(), json!(fetched.final_url.as_str()));
        metadata.insert("downloaded".to_string(), json!(true));
    }
    Ok(extracted)
}

/// The charset an HTML document declares for itself in its first kilobytes.
fn meta_charset(body: &[u8]) -> Option<String> {
    static RE: OnceLock<Regex> = OnceLock::new();
    let pattern =
        RE.get_or_init(|| Regex::new(r#"(?i)<meta[^>]{0,200}?charset\s*=\s*["']?\s*([a-z0-9_:.\-]+)"#).expect("valid regex"));
    let head = &body[..body.len().min(4096)];
    let label = pattern.captures(head)?.get(1)?.as_bytes();
    Some(String::from_utf8_lossy(label).into_owned())
}

#[cfg(test)]
mod tests {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    use super::*;
    use crate::extract::{extract_content, ExtractInput};

    const PAGE: &str = r#"<!DOCTYPE html><html lang="en"><head><title>Tea Guide | Site</title>
<meta property="og:title" content="A Guide to Tea"><meta name="description" content="All about tea."></head>
<body><nav><a href="/">Home</a></nav><article><h1>A Guide to Tea</h1>
<p>Green tea is <a href="/kinds/green">unoxidised</a>. Steep at 80 degrees for two minutes so the leaves do not turn bitter, and use soft water when you can.</p>
<p>Black tea is fully oxidised and takes boiling water. Longer steeping gives a stronger cup, up to about five minutes for most leaves.</p></article>
<footer>Site footer</footer></body></html>"#;

    async fn html_server() -> MockServer {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/tea"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(PAGE, "text/html; charset=utf-8"))
            .mount(&server)
            .await;
        server
    }

    async fn run(url: String, ctx: &ExtractContext) -> Result<Extracted, ContentError> {
        extract_content(ExtractInput::url(url), ctx).await
    }

    #[tokio::test]
    async fn simple_engine_converts_a_page() {
        let server = html_server().await;
        let url = format!("{}/tea", server.uri());
        let out = run(format!("  {url} "), &ExtractContext::default()).await.unwrap();
        assert_eq!(out.title.as_deref(), Some("A Guide to Tea"));
        assert_eq!(out.identified_type, "url");
        assert_eq!(out.url.as_deref(), Some(url.as_str()));
        assert_eq!(out.file_path, None);
        assert!(out.content.starts_with("# A Guide to Tea\n\nGreen tea is [unoxidised]("), "{}", out.content);
        assert!(out.content.contains(&format!("]({}/kinds/green)", server.uri())));
        assert!(!out.content.contains("Home") && !out.content.contains("Site footer"));
        assert_eq!(out.metadata["engine"], "simple");
        assert_eq!(out.metadata["description"], "All about tea.");
        assert_eq!(out.metadata["language"], "en");
        assert_eq!(out.metadata["final_url"], url);
    }

    #[tokio::test]
    async fn charset_from_header_and_from_meta() {
        let server = MockServer::start().await;
        // "Café crème" in ISO-8859-1.
        let latin = b"<html><head><title>Caf\xe9</title></head><body><p>Caf\xe9 cr\xe8me</p></body></html>".to_vec();
        Mock::given(method("GET"))
            .and(path("/header"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(latin, "text/html; charset=ISO-8859-1"))
            .mount(&server)
            .await;
        // Shift_JIS declared only in the document.
        let mut sjis = b"<html><head><meta http-equiv=\"Content-Type\" content=\"text/html; charset=Shift_JIS\"><title>t</title></head><body><p>".to_vec();
        sjis.extend_from_slice(&[0x82, 0xb1, 0x82, 0xf1, 0x82, 0xc9, 0x82, 0xbf, 0x82, 0xcd]);
        sjis.extend_from_slice(b"</p></body></html>");
        Mock::given(method("GET"))
            .and(path("/meta"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(sjis, "text/html"))
            .mount(&server)
            .await;

        let ctx = ExtractContext::default();
        let header = run(format!("{}/header", server.uri()), &ctx).await.unwrap();
        assert_eq!(header.title.as_deref(), Some("Café"));
        assert_eq!(header.content, "Café crème");
        let meta = run(format!("{}/meta", server.uri()), &ctx).await.unwrap();
        assert_eq!(meta.content, "こんにちは");
    }

    #[tokio::test]
    async fn document_urls_go_through_the_file_flow() {
        let server = MockServer::start().await;
        let pdf = crate::extract::documents::pdf::tests::pdf_bytes(&[&["Annual figures inside."]], Some("Annual Report"));
        let docx = crate::extract::documents::office::tests::docx_bytes();
        // By extension.
        Mock::given(method("GET"))
            .and(path("/files/report.pdf"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(pdf.clone(), "application/octet-stream"))
            .mount(&server)
            .await;
        // By content type, on a URL that looks like a page.
        Mock::given(method("GET"))
            .and(path("/download"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(pdf.clone(), "application/pdf"))
            .mount(&server)
            .await;
        // By content alone, with a file name from Content-Disposition.
        Mock::given(method("GET"))
            .and(path("/export"))
            .respond_with(
                ResponseTemplate::new(200)
                    .insert_header("content-disposition", "attachment; filename=\"plan.docx\"")
                    .set_body_raw(docx, "application/octet-stream"),
            )
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/raw/notes.md"))
            .respond_with(ResponseTemplate::new(200).set_body_raw("# Notes\n\nRaw markdown.", "text/plain; charset=utf-8"))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/logo"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(b"\x89PNG\r\n\x1a\n\x00\x00".to_vec(), "image/png"))
            .mount(&server)
            .await;

        let ctx = ExtractContext::default();
        for route in ["/files/report.pdf", "/download"] {
            let url = format!("{}{route}", server.uri());
            let out = run(url.clone(), &ctx).await.unwrap();
            assert_eq!(out.identified_type, "pdf", "{route}");
            assert_eq!(out.title.as_deref(), Some("Annual Report"));
            assert!(out.content.contains("Annual figures inside."));
            assert_eq!(out.url.as_deref(), Some(url.as_str()));
            assert_eq!(out.file_path, None);
            assert_eq!(out.metadata["downloaded"], true);
        }
        let docx = run(format!("{}/export", server.uri()), &ctx).await.unwrap();
        assert_eq!(docx.identified_type, "docx");
        assert!(docx.content.contains("# Goals"));

        let markdown = run(format!("{}/raw/notes.md", server.uri()), &ctx).await.unwrap();
        assert_eq!(markdown.identified_type, "markdown");
        assert_eq!(markdown.title.as_deref(), Some("Notes"));
        assert_eq!(markdown.content, "# Notes\n\nRaw markdown.");

        let image = run(format!("{}/logo", server.uri()), &ctx).await.unwrap_err();
        assert!(
            matches!(&image, ContentError::UnsupportedType { identified_type, .. } if identified_type == "image/png"),
            "{image:?}"
        );
    }

    #[tokio::test]
    async fn errors_from_the_simple_engine() {
        let server = MockServer::start().await;
        Mock::given(method("GET")).and(path("/missing")).respond_with(ResponseTemplate::new(404)).mount(&server).await;
        Mock::given(method("GET"))
            .and(path("/spa"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(
                "<html><head><title>App</title></head><body><div id=\"root\"></div><script>render()</script></body></html>",
                "text/html",
            ))
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/hop"))
            .respond_with(ResponseTemplate::new(302).insert_header("location", "http://169.254.169.254/latest/"))
            .mount(&server)
            .await;
        let ctx = ExtractContext::default();
        assert!(matches!(run(format!("{}/missing", server.uri()), &ctx).await, Err(ContentError::NotFound(_))));
        assert!(matches!(run(format!("{}/spa", server.uri()), &ctx).await, Err(ContentError::EmptyContent(_))));
        assert!(matches!(run(format!("{}/hop", server.uri()), &ctx).await, Err(ContentError::UrlNotAllowed(_))));
    }

    #[test]
    fn engine_chains() {
        let base = ExtractContext::default();
        assert_eq!(engine_chain(&base).unwrap(), vec![UrlEngine::Simple]);

        let all = ExtractContext {
            firecrawl_api_key: Some("f".into()),
            jina_api_key: Some("j".into()),
            crawl4ai_api_url: Some("http://c4:11235".into()),
            ..ExtractContext::default()
        };
        assert_eq!(
            engine_chain(&all).unwrap(),
            vec![UrlEngine::Firecrawl, UrlEngine::Jina, UrlEngine::Crawl4ai, UrlEngine::Simple]
        );
        let blank_key = ExtractContext { jina_api_key: Some("  ".into()), ..ExtractContext::default() };
        assert_eq!(engine_chain(&blank_key).unwrap(), vec![UrlEngine::Simple]);

        let with = |engine| ExtractContext { url_engine: engine, ..ExtractContext::default() };
        assert_eq!(engine_chain(&with(UrlEngine::Simple)).unwrap(), vec![UrlEngine::Simple]);
        // Jina works without a key.
        assert_eq!(engine_chain(&with(UrlEngine::Jina)).unwrap(), vec![UrlEngine::Jina]);
        assert!(matches!(engine_chain(&with(UrlEngine::Firecrawl)), Err(ContentError::Configuration(_))));
        let self_hosted = ExtractContext {
            url_engine: UrlEngine::Firecrawl,
            firecrawl_api_url: "http://firecrawl.lan:3002".into(),
            ..ExtractContext::default()
        };
        assert_eq!(engine_chain(&self_hosted).unwrap(), vec![UrlEngine::Firecrawl]);
        // Crawl4AI selected without a server falls back to auto.
        assert_eq!(engine_chain(&with(UrlEngine::Crawl4ai)).unwrap(), vec![UrlEngine::Simple]);
        let c4 = ExtractContext { url_engine: UrlEngine::Crawl4ai, ..all.clone() };
        assert_eq!(engine_chain(&c4).unwrap(), vec![UrlEngine::Crawl4ai]);
    }

    #[tokio::test]
    async fn auto_uses_the_remote_engine_and_falls_back_to_simple() {
        let site = html_server().await;
        Mock::given(method("HEAD"))
            .and(path("/tea"))
            .respond_with(ResponseTemplate::new(200).insert_header("content-type", "text/html"))
            .mount(&site)
            .await;
        let page = format!("{}/tea", site.uri());

        let jina = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": {"title": "From Jina", "content": "Jina markdown"}})))
            .mount(&jina)
            .await;
        let ctx = ExtractContext {
            jina_api_key: Some("k".into()),
            jina_reader_url: jina.uri(),
            ..ExtractContext::default()
        };
        let out = run(page.clone(), &ctx).await.unwrap();
        assert_eq!(out.metadata["engine"], "jina");
        assert_eq!(out.content, "Jina markdown");
        assert_eq!(out.title.as_deref(), Some("From Jina"));
        assert_eq!(out.identified_type, "url");
        assert_eq!(out.url.as_deref(), Some(page.as_str()));

        // The remote engine fails: the built-in one takes over.
        let broken = MockServer::start().await;
        Mock::given(method("GET")).respond_with(ResponseTemplate::new(500)).mount(&broken).await;
        let ctx = ExtractContext { jina_reader_url: broken.uri(), ..ctx };
        let out = run(page.clone(), &ctx).await.unwrap();
        assert_eq!(out.metadata["engine"], "simple");
        assert_eq!(out.title.as_deref(), Some("A Guide to Tea"));

        // The remote engine returns nothing: same.
        let empty = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"data": {"title": "", "content": "  "}})))
            .mount(&empty)
            .await;
        let ctx = ExtractContext { jina_reader_url: empty.uri(), ..ctx };
        assert_eq!(run(page.clone(), &ctx).await.unwrap().metadata["engine"], "simple");

        // An explicitly selected engine does not fall back.
        let ctx = ExtractContext { url_engine: UrlEngine::Jina, jina_reader_url: broken.uri(), ..ctx };
        assert!(matches!(run(page, &ctx).await, Err(ContentError::ExternalService(_))));
    }

    #[tokio::test]
    async fn remote_engine_is_bypassed_for_documents() {
        let site = MockServer::start().await;
        let pdf = crate::extract::documents::pdf::tests::pdf_bytes(&[&["Served as a download."]], None);
        Mock::given(method("HEAD"))
            .and(path("/paper"))
            .respond_with(ResponseTemplate::new(200).insert_header("content-type", "application/pdf"))
            .mount(&site)
            .await;
        Mock::given(method("GET"))
            .and(path("/paper"))
            .respond_with(ResponseTemplate::new(200).set_body_raw(pdf, "application/pdf"))
            .mount(&site)
            .await;
        let jina = MockServer::start().await;
        Mock::given(method("GET")).respond_with(ResponseTemplate::new(200).set_body_string("should not be used")).mount(&jina).await;

        let ctx = ExtractContext { url_engine: UrlEngine::Jina, jina_reader_url: jina.uri(), ..ExtractContext::default() };
        let out = run(format!("{}/paper", site.uri()), &ctx).await.unwrap();
        assert_eq!(out.identified_type, "pdf");
        assert!(out.content.contains("Served as a download."));
        assert!(jina.received_requests().await.unwrap().is_empty());
    }

    #[test]
    fn meta_charset_sniffing() {
        assert_eq!(meta_charset(b"<html><head><meta charset=\"utf-8\">").as_deref(), Some("utf-8"));
        assert_eq!(meta_charset(b"<META HTTP-EQUIV='Content-Type' CONTENT='text/html; charset=windows-1251'>").as_deref(), Some("windows-1251"));
        assert_eq!(meta_charset(b"<html><body>charset=nope</body></html>"), None);
    }
}
