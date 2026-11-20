//! The rich note editor (BlockSuite, AFFiNE's editor engine): loading and
//! saving a note's editor document, links between notes and backlinks, and
//! the content-addressed store for files placed inside notes.
//!
//! A note keeps its plain `content` (markdown exported by the editor) for
//! search, chat, embeddings and the notes grid, exactly as before. The editor
//! document (`doc_state`, a Yjs update) sits beside it with the hash of the
//! content it was saved with; when the content has since changed elsewhere
//! (the JSON API, the Python build), the editor rebuilds from the content.
//!
//! Files dropped or pasted into a note are stored under
//! `<data>/note_blobs/<key>`, where the key is the editor's own content hash
//! (URL-safe base64 SHA-256). The server checks the hash, so a key always
//! names the bytes it was computed from.

use std::path::PathBuf;

use base64::Engine;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::notebooks::{ensure_table, not_found_as};
use super::notes::{self, NewNote, NoteView};
use crate::config::Config;
use crate::db::Doc;
use crate::domain::note::{NoteDoc, NoteRef};
use crate::domain::{Note, Notebook};
use crate::error::{AppError, AppResult};
use crate::jobs::JobQueue;
use crate::state::AppState;

pub fn register_jobs(_jobs: &JobQueue) {}

/// Editor modes.
pub const MODES: [&str; 2] = ["page", "edgeless"];
/// Starts the section of a note's content that lists the files attached in
/// the editor (and their text). It is rebuilt on every save and is never part
/// of the editor document.
pub const ATTACHMENTS_MARKER: &str = "<!-- nova-notes:attachments -->";
/// At most this many attached files have their text added to the content.
const MAX_ATTACHMENT_TEXTS: usize = 10;
/// Characters of text taken from each attached file.
const MAX_ATTACHMENT_TEXT: usize = 20_000;

/// The first Quick Note creates a normal notebook for captures.
pub async fn quick_notes_notebook(state: &AppState) -> AppResult<&'static str> {
    const QUICK_NOTES_ID: &str = "notebook:quick_notes";
    let _write = state.db.lock_record(QUICK_NOTES_ID).await;
    match Notebook::get(state, QUICK_NOTES_ID).await {
        Ok(notebook) if notebook.archived => {
            state.db.update(QUICK_NOTES_ID, Doc::new().set("archived", false)).await?;
        }
        Ok(_) => {}
        Err(error) if error.is_not_found() => {
            let notebook = Doc::new().set("name", "Quick Notes").set("description", "").set("archived", false);
            state.db.create_with_id(QUICK_NOTES_ID, notebook).await?;
        }
        Err(error) => return Err(error),
    }
    Ok(QUICK_NOTES_ID)
}

pub async fn create_draft(state: &AppState, notebook_id: &str) -> AppResult<String> {
    ensure_table(notebook_id, Notebook::TABLE, "Notebook not found")?;
    Note::create_draft(state, notebook_id).await
}

// ---------------------------------------------------------------------------
// Loading and saving
// ---------------------------------------------------------------------------

/// What the editor needs to open a note.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoteDocView {
    /// `None` for a new note.
    pub note: Option<NoteView>,
    /// The saved editor document, when it still matches the note's content.
    pub doc_state: Option<String>,
    /// `"page"` or `"edgeless"`.
    pub doc_mode: String,
    /// The notebook's other notes, which the note can link to.
    pub notes: Vec<NoteRef>,
    /// Notes linking to this one.
    pub backlinks: Vec<NoteRef>,
}

fn content_hash(content: &str) -> String {
    hex::encode(Sha256::digest(content.as_bytes()))
}

/// Load a note (or a blank one, `note_id` `None`) for the editor, with the
/// notebook's notes as link targets.
pub async fn load_note_doc(state: &AppState, notebook_id: &str, note_id: Option<&str>) -> AppResult<NoteDocView> {
    let notebook_notes = notes::list_notes(state, Some(notebook_id)).await?;
    let mut targets: Vec<NoteRef> =
        notebook_notes.into_iter().map(|n| NoteRef { id: n.id, title: n.title }).collect();

    let Some(note_id) = note_id else {
        return Ok(NoteDocView { note: None, doc_state: None, doc_mode: "page".into(), notes: targets, backlinks: Vec::new() });
    };
    let mut note = notes::get_note(state, note_id).await?;
    let doc = Note::doc_state(state, &note.id).await?;
    let fresh = doc.doc_hash.as_deref() == Some(content_hash(note.content.as_deref().unwrap_or_default()).as_str());
    let backlinks = Note::backlinks(state, &note.id).await?;
    // Notes from other notebooks that link here can be linked back to.
    for link in &backlinks {
        if !targets.iter().any(|t| t.id == link.id) {
            targets.push(link.clone());
        }
    }
    targets.retain(|t| t.id != note.id);
    // The editor rebuilds from the content without the attachments section.
    if let Some(content) = note.content.as_mut() {
        if let Some(at) = content.find(ATTACHMENTS_MARKER) {
            content.truncate(at);
            *content = content.trim_end().trim_end_matches("---").trim_end().to_string();
        }
    }
    Ok(NoteDocView {
        doc_state: if fresh { doc.doc_state } else { None },
        doc_mode: doc.doc_mode.filter(|m| MODES.contains(&m.as_str())).unwrap_or_else(|| "page".into()),
        note: Some(note),
        notes: targets,
        backlinks,
    })
}

/// A save from the editor.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SaveNoteDoc {
    #[serde(default)]
    pub title: Option<String>,
    /// Markdown exported from the document.
    #[serde(default)]
    pub content: String,
    /// Base64 Yjs update.
    #[serde(default)]
    pub doc_state: Option<String>,
    #[serde(default)]
    pub doc_mode: Option<String>,
    /// Ids of the notes the document links to.
    #[serde(default)]
    pub links: Vec<String>,
    /// Files attached in the document.
    #[serde(default)]
    pub attachments: Vec<AttachedFile>,
}

/// A file attached in the editor: its blob key and file name.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AttachedFile {
    pub key: String,
    #[serde(default)]
    pub name: String,
}

/// Create or update a note from the editor. Existing notes may be blank,
/// including when the writer clears the page or starts with a title.
pub async fn save_note_doc(
    state: &AppState,
    notebook_id: &str,
    note_id: Option<&str>,
    input: &SaveNoteDoc,
) -> AppResult<NoteView> {
    let content = with_attachments(&state.config, input.content.trim_end(), &input.attachments).await;
    if note_id.is_none() && content.trim().is_empty() {
        return Err(AppError::invalid("Note content is required"));
    }
    let mode = input.doc_mode.clone().unwrap_or_else(|| "page".into());
    if !MODES.contains(&mode.as_str()) {
        return Err(AppError::invalid("doc_mode must be 'page' or 'edgeless'"));
    }
    if let Some(doc_state) = &input.doc_state {
        base64::engine::general_purpose::STANDARD
            .decode(doc_state)
            .map_err(|_| AppError::invalid("doc_state is not valid base64"))?;
    }
    let title = input.title.as_deref().unwrap_or_default().trim();

    let (id, create_command) = match note_id {
        Some(id) => {
            ensure_table(id, Note::TABLE, "Note not found")?;
            (id.to_string(), None)
        }
        None => {
            let new = NewNote {
                title: (!title.is_empty()).then(|| title.to_string()),
                content: content.clone(),
                note_type: Some(Some("human".to_string())),
                notebook_id: Some(notebook_id.to_string()),
            };
            let note = notes::create_note(state, &new).await?;
            (note.id, note.command_id)
        }
    };

    let mut wanted: Vec<String> = Vec::new();
    for link in &input.links {
        let link = if link.contains(':') { link.clone() } else { format!("note:{link}") };
        if link.starts_with("note:") && link != id && !wanted.contains(&link) {
            wanted.push(link);
        }
    }
    let links = Note::existing_ids(state, &wanted).await?;
    let doc = NoteDoc {
        doc_state: input.doc_state.clone(),
        doc_hash: Some(content_hash(&content)),
        doc_mode: Some(mode),
    };
    let content = (!content.trim().is_empty()).then_some(content.as_str());
    let title = (note_id.is_some() || !title.is_empty()).then_some(title);
    let (note, command) = Note::save_doc_state(state, &id, title, content, &doc, &links)
        .await
        .map_err(not_found_as("Note not found"))?;
    Ok(NoteView::new(note, command.or(create_command)))
}

/// The note content: the editor's markdown, then (after
/// [`ATTACHMENTS_MARKER`]) a link to each attached file and the text of those
/// that are documents (PDF, DOCX, HTML, text, markdown), so chat, search and
/// embeddings see them.
async fn with_attachments(config: &Config, markdown: &str, attachments: &[AttachedFile]) -> String {
    let mut seen = Vec::new();
    let files: Vec<&AttachedFile> = attachments
        .iter()
        .filter(|a| is_blob_key(&a.key) && !seen.contains(&a.key) && { seen.push(a.key.clone()); true })
        .collect();
    if files.is_empty() {
        return markdown.to_string();
    }
    let mut out = String::from(markdown);
    out.push_str("\n\n---\n");
    out.push_str(ATTACHMENTS_MARKER);
    out.push_str("\n\n**Attached files**\n\n");
    for file in &files {
        let name = file.name.replace(['[', ']', '\n', '\r'], " ");
        out.push_str(&format!("- [{}](/note-blobs/{})\n", name.trim(), file.key));
    }
    for file in files.iter().take(MAX_ATTACHMENT_TEXTS) {
        match attachment_text(config, file).await {
            Ok(Some(text)) if !text.trim().is_empty() => {
                let text: String = text.chars().take(MAX_ATTACHMENT_TEXT).collect();
                out.push_str(&format!("\n### {}\n\n{}\n", file.name.replace(['\n', '\r'], " ").trim(), text.trim()));
            }
            Ok(_) => {}
            Err(e) => tracing::warn!("Could not read the text of an attached file: {e}"),
        }
    }
    out
}

/// The text of an attached document, cached beside the file. `None` for a
/// type that has no text to extract (images, archives, ...).
async fn attachment_text(config: &Config, file: &AttachedFile) -> AppResult<Option<String>> {
    let ext = notes::file_extension(&file.name);
    if !notes::ALLOWED_TYPES.iter().any(|(allowed, _)| *allowed == ext) {
        return Ok(None);
    }
    let (path, _) = blob_paths(config, &file.key)?;
    let cache = path.with_extension("txt");
    if let Ok(text) = tokio::fs::read_to_string(&cache).await {
        return Ok(Some(text));
    }
    if !tokio::fs::try_exists(&path).await.unwrap_or(false) {
        return Ok(None);
    }
    // Extraction goes by extension, so work on a copy that has one.
    let copy = path.with_extension(format!("extract-{}{ext}", uuid::Uuid::new_v4().simple()));
    tokio::fs::copy(&path, &copy).await?;
    let text = notes::extract_note_text(&copy).await;
    let _ = tokio::fs::remove_file(&copy).await;
    let text = text?;
    tokio::fs::write(&cache, &text).await?;
    Ok(Some(text))
}

/// Create a note to link to from the editor's `@` menu: titled `title`, with
/// the title as its first line of content, filed in the notebook.
pub async fn create_linked_note(state: &AppState, notebook_id: &str, title: &str) -> AppResult<NoteView> {
    let title = title.trim();
    let title = if title.is_empty() { "Untitled" } else { title };
    let new = NewNote {
        title: Some(title.to_string()),
        content: title.to_string(),
        note_type: Some(Some("human".to_string())),
        notebook_id: Some(notebook_id.to_string()),
    };
    notes::create_note(state, &new).await
}

// ---------------------------------------------------------------------------
// Files inside notes
// ---------------------------------------------------------------------------

/// A stored file: its bytes and MIME type.
#[derive(Debug, Clone)]
pub struct NoteBlob {
    pub data: Vec<u8>,
    pub mime: String,
}

pub fn blobs_dir(config: &Config) -> PathBuf {
    config.data_dir.join("note_blobs")
}

/// A key is URL-safe base64 of a SHA-256 digest (43 characters, plus `=`).
pub fn is_blob_key(key: &str) -> bool {
    let body = key.strip_suffix('=').unwrap_or(key);
    body.len() == 43 && body.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
}

fn blob_key_of(data: &[u8]) -> String {
    base64::engine::general_purpose::URL_SAFE.encode(Sha256::digest(data))
}

/// `type/subtype`, lower-cased, or `application/octet-stream`.
fn clean_mime(mime: &str) -> String {
    let mime = mime.split(';').next().unwrap_or_default().trim().to_ascii_lowercase();
    let valid = mime.split_once('/').is_some_and(|(a, b)| {
        let ok = |s: &str| !s.is_empty() && s.len() <= 100 && s.bytes().all(|c| c.is_ascii_alphanumeric() || b".+-_".contains(&c));
        ok(a) && ok(b)
    });
    if valid { mime } else { "application/octet-stream".to_string() }
}

fn blob_paths(config: &Config, key: &str) -> AppResult<(PathBuf, PathBuf)> {
    if !is_blob_key(key) {
        return Err(AppError::invalid("Invalid file key"));
    }
    // `=` padding is optional in keys; files are stored under the unpadded form.
    let stem = key.trim_end_matches('=');
    let dir = blobs_dir(config);
    Ok((dir.join(stem), dir.join(format!("{stem}.type"))))
}

/// Store a file under its content hash. A key that is not the hash of the
/// bytes is refused.
pub async fn put_blob(config: &Config, key: &str, mime: &str, data: &[u8]) -> AppResult<()> {
    let (path, type_path) = blob_paths(config, key)?;
    if blob_key_of(data).trim_end_matches('=') != key.trim_end_matches('=') {
        return Err(AppError::invalid("The file does not match its key"));
    }
    tokio::fs::create_dir_all(blobs_dir(config)).await?;
    let tmp = path.with_extension(format!("tmp-{}", uuid::Uuid::new_v4().simple()));
    tokio::fs::write(&tmp, data).await?;
    tokio::fs::rename(&tmp, &path).await?;
    tokio::fs::write(&type_path, clean_mime(mime)).await?;
    Ok(())
}

/// A stored file. Unknown key → `NotFound`.
pub async fn get_blob(config: &Config, key: &str) -> AppResult<NoteBlob> {
    let (path, type_path) = blob_paths(config, key)?;
    let data = match tokio::fs::read(&path).await {
        Ok(data) => data,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Err(AppError::not_found("File not found")),
        Err(e) => return Err(e.into()),
    };
    let mime = tokio::fs::read_to_string(&type_path).await.map(|m| clean_mime(&m)).unwrap_or_else(|_| clean_mime(""));
    Ok(NoteBlob { data, mime })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys() {
        let key = blob_key_of(b"hello");
        assert_eq!(key.len(), 44);
        assert!(is_blob_key(&key));
        assert!(is_blob_key(key.trim_end_matches('=')));
        assert!(!is_blob_key("../../etc/passwd"));
        assert!(!is_blob_key(""));
        assert!(!is_blob_key(&format!("{}/", &key[..43])));
    }

    #[test]
    fn mime_is_cleaned() {
        assert_eq!(clean_mime("image/png"), "image/png");
        assert_eq!(clean_mime("Text/HTML; charset=utf-8"), "text/html");
        assert_eq!(clean_mime("text/html\r\nX-Evil: 1"), "application/octet-stream");
        assert_eq!(clean_mime(""), "application/octet-stream");
    }

    /// The editor bundle must make no third-party requests. `editor/build.mjs`
    /// rewrites these hosts and `editor/src/editor.js` drops the embed blocks;
    /// this catches a BlockSuite upgrade or build change that brings one back.
    /// It reads `static/editor` from disk (the files the server embeds), since
    /// `include_dir!` is not recompiled when only those files change.
    #[test]
    fn editor_bundle_has_no_third_party_hosts() {
        use std::path::Path;
        const BLOCKED: &[&str] = &[
            "api.github.com/repos",
            "unpkg.com",
            "cdn.jsdelivr.net",
            "youtube.com/embed",
            "youtube.com/oembed",
            "img.youtube.com",
            "loom.com/embed",
            "loom.com/v1/oembed",
            "figma.com/embed",
            "toeverything.workers.dev",
            "fxtwitter",
        ];

        fn scan(dir: &Path, scanned: &mut usize, hits: &mut Vec<String>) {
            for entry in std::fs::read_dir(dir).expect("read static/editor") {
                let path = entry.expect("read static/editor entry").path();
                if path.is_dir() {
                    scan(&path, scanned, hits);
                    continue;
                }
                *scanned += 1;
                let body = std::fs::read(&path).expect("read bundle file");
                for host in BLOCKED {
                    if body.windows(host.len()).any(|w| w == host.as_bytes()) {
                        hits.push(format!("{} contains {host}", path.display()));
                    }
                }
            }
        }

        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("static/editor");
        let (mut scanned, mut hits) = (0, Vec::new());
        scan(&root, &mut scanned, &mut hits);
        assert!(scanned > 100, "the editor bundle looks missing ({scanned} files)");
        assert!(hits.is_empty(), "third-party hosts in static/editor:\n{}", hits.join("\n"));
    }

    // -----------------------------------------------------------------------
    // Through the web routes
    // -----------------------------------------------------------------------

    use crate::testing::TestApp;
    use axum::body::Body;
    use axum::http::{header, Method, Request, StatusCode};
    use http_body_util::BodyExt;
    use serde_json::{json, Value as Json};

    async fn notebook(app: &TestApp) -> String {
        let (status, body) = app.post("/api/notebooks", json!({ "name": "Rich", "description": "" })).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        body["id"].as_str().unwrap().to_string()
    }

    async fn put_raw(app: &TestApp, path: &str, mime: &str, data: &[u8]) -> StatusCode {
        let request = Request::builder()
            .method(Method::PUT)
            .uri(path)
            .header(header::CONTENT_TYPE, mime)
            .body(Body::from(data.to_vec()))
            .unwrap();
        app.send(request).await.status()
    }

    #[tokio::test]
    async fn editor_save_load_links_and_backlinks() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        let (_, target) = app.post("/api/notes", json!({ "title": "Target", "content": "# Target\n\ntext", "notebook_id": nb })).await;
        let target_id = target["id"].as_str().unwrap().to_string();

        // A new note from the editor is filed in the notebook it was opened in.
        let (status, saved) = app
            .post(
                &format!("/notebooks/{nb}/notes/editor"),
                json!({
                    "title": "Linker",
                    "content": format!("See [Target](/notes/{target_id})"),
                    "doc_state": "AQID",
                    "doc_mode": "edgeless",
                    "links": [target_id, "note:missing", "source:abc"],
                }),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{saved}");
        let id = saved["id"].as_str().unwrap().to_string();
        assert_eq!(saved["note_type"], "human");
        let (_, listed) = app.get(&format!("/api/notes?notebook_id={nb}")).await;
        assert!(listed.as_array().unwrap().iter().any(|n| n["id"] == id.as_str()));

        let (status, doc) = app.get(&format!("/notebooks/{nb}/notes/{id}/editor")).await;
        assert_eq!(status, StatusCode::OK, "{doc}");
        assert_eq!(doc["doc_state"], "AQID");
        assert_eq!(doc["doc_mode"], "edgeless");
        assert_eq!(doc["note"]["title"], "Linker");
        let targets: Vec<&str> = doc["notes"].as_array().unwrap().iter().map(|n| n["id"].as_str().unwrap()).collect();
        assert!(targets.contains(&target_id.as_str()));
        assert!(!targets.contains(&id.as_str()), "a note is not its own link target");

        // Only existing notes are stored as links; the target sees the backlink.
        let rows = app
            .state
            .db
            .query("SELECT VALUE links FROM ONLY $id", crate::db::Params::new().record("id", &id))
            .await
            .unwrap();
        assert_eq!(rows, vec![json!(target_id)]);
        let (_, target_doc) = app.get(&format!("/notebooks/{nb}/notes/{target_id}/editor")).await;
        assert_eq!(target_doc["backlinks"], json!([{ "id": id, "title": "Linker" }]));
        assert_eq!(target_doc["doc_state"], Json::Null, "never opened in the editor");

        // Content changed outside the editor: the stale document is not offered.
        let (status, _) = app.put(&format!("/api/notes/{id}"), json!({ "content": "Rewritten elsewhere" })).await;
        assert_eq!(status, StatusCode::OK);
        let (_, doc) = app.get(&format!("/notebooks/{nb}/notes/{id}/editor")).await;
        assert_eq!(doc["doc_state"], Json::Null);
        assert_eq!(doc["note"]["content"], "Rewritten elsewhere");

        // Updating through the editor keeps the id and replaces the links.
        let (status, _) = app
            .post(&format!("/notebooks/{nb}/notes/{id}/editor"), json!({ "title": "Linker", "content": "no links", "doc_state": "AQID" }))
            .await;
        assert_eq!(status, StatusCode::OK);
        let (_, target_doc) = app.get(&format!("/notebooks/{nb}/notes/{target_id}/editor")).await;
        assert_eq!(target_doc["backlinks"], json!([]));
    }

    #[tokio::test]
    async fn editor_save_is_validated() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        let url = format!("/notebooks/{nb}/notes/editor");
        let (status, body) = app.post(&url, json!({ "title": "x", "content": "   " })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert_eq!(body["detail"], "Note content is required");
        let (status, _) = app.post(&url, json!({ "content": "x", "doc_mode": "sideways" })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _) = app.post(&url, json!({ "content": "x", "doc_state": "not base64!" })).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        let (status, _) = app.post("/notebooks/notebook:nope/notes/editor", json!({ "content": "x" })).await;
        assert_eq!(status, StatusCode::NOT_FOUND);

        // The @ menu's "Create note" makes a real note in the notebook.
        let (status, created) = app.post(&format!("/notebooks/{nb}/notes/link"), json!({ "title": "Ideas" })).await;
        assert_eq!(status, StatusCode::OK, "{created}");
        assert_eq!(created["title"], "Ideas");
        let (_, listed) = app.get(&format!("/api/notes?notebook_id={nb}")).await;
        assert_eq!(listed.as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn blobs_are_content_addressed_and_inert() {
        let app = TestApp::new().await;
        let data = b"<script>alert(1)</script>";
        let key = blob_key_of(data);
        assert_eq!(put_raw(&app, &format!("/note-blobs/{key}"), "text/html", data).await, StatusCode::OK);

        let response = app.send(Request::builder().uri(format!("/note-blobs/{key}")).body(Body::empty()).unwrap()).await;
        assert_eq!(response.status(), StatusCode::OK);
        let headers = response.headers().clone();
        assert_eq!(headers[header::CONTENT_TYPE], "text/html");
        assert_eq!(headers[header::CONTENT_DISPOSITION], "attachment");
        assert_eq!(headers[header::X_CONTENT_TYPE_OPTIONS], "nosniff");
        assert!(headers[header::CONTENT_SECURITY_POLICY].to_str().unwrap().starts_with("sandbox"));
        let body = response.into_body().collect().await.unwrap().to_bytes();
        assert_eq!(&body[..], data);

        // A key must be the hash of the bytes; bad keys never reach the disk.
        assert_eq!(put_raw(&app, &format!("/note-blobs/{key}"), "text/plain", b"other").await, StatusCode::BAD_REQUEST);
        assert_eq!(put_raw(&app, "/note-blobs/..%2Fescape", "text/plain", b"x").await, StatusCode::BAD_REQUEST);
        let (status, _) = app.get(&format!("/note-blobs/{}", blob_key_of(b"never stored"))).await;
        assert_eq!(status, StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn attached_documents_add_their_text_to_the_content() {
        let app = TestApp::new().await;
        let nb = notebook(&app).await;
        let text = b"Quarterly numbers: 42 widgets";
        let key = blob_key_of(text);
        assert_eq!(put_raw(&app, &format!("/note-blobs/{key}"), "text/plain", text).await, StatusCode::OK);
        let image = blob_key_of(b"\x89PNG fake");
        assert_eq!(put_raw(&app, &format!("/note-blobs/{image}"), "image/png", b"\x89PNG fake").await, StatusCode::OK);

        let (status, saved) = app
            .post(
                &format!("/notebooks/{nb}/notes/editor"),
                json!({
                    "title": "Report",
                    "content": "Body text",
                    "attachments": [{ "key": key, "name": "report.txt" }, { "key": image, "name": "chart.png" }],
                }),
            )
            .await;
        assert_eq!(status, StatusCode::OK, "{saved}");
        let content = saved["content"].as_str().unwrap();
        assert!(content.starts_with("Body text"));
        assert!(content.contains(ATTACHMENTS_MARKER));
        assert!(content.contains(&format!("- [report.txt](/note-blobs/{key})")));
        assert!(content.contains(&format!("- [chart.png](/note-blobs/{image})")));
        assert!(content.contains("Quarterly numbers: 42 widgets"));

        // The editor gets the content without the generated section.
        let id = saved["id"].as_str().unwrap();
        let (_, doc) = app.get(&format!("/notebooks/{nb}/notes/{id}/editor")).await;
        assert_eq!(doc["note"]["content"], "Body text");
    }
}
