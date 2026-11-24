//! Notes, and the original files of notes created by dropping a file.
//!
//! The Rust counterpart of `api/routers/notes.py` and
//! `open_notebook/utils/note_files.py`. Error details are the Python strings.
//!
//! Text extraction for [`upload_note_file`] is [`extract_note_text`]
//! (`nn-content`); the model-written title of an untitled AI note is
//! [`generate_note_title`].

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::notebooks::{double_option, ensure_table, not_found_as, timestamp};
use crate::config::Config;
use crate::domain::note::{note_file_path, remove_note_file};
use crate::domain::{Note, Notebook};
use crate::error::{AppError, AppResult};
use crate::jobs::JobQueue;
use crate::state::AppState;

/// Notes have no background jobs of their own (`embed_note` belongs to the
/// embedding service).
pub fn register_jobs(_jobs: &JobQueue) {}

const NOTE_NOT_FOUND: &str = "Note not found";
const NOTEBOOK_NOT_FOUND: &str = "Notebook not found";

/// MIME type of a generated infographic's attachment.
pub const SVG_MIME: &str = "image/svg+xml";
/// `Content-Security-Policy` for a served infographic: sanitised when it was
/// generated, and never allowed to run or load anything.
pub const INFOGRAPHIC_CSP_HEADER: &str =
    "sandbox; default-src 'none'; style-src 'unsafe-inline'; img-src data:; font-src data:";
/// `Content-Security-Policy` for a served explainer: its scripts may run, but
/// only in an opaque origin and with every subresource request blocked.
pub const EXPLAINER_CSP_HEADER: &str = "sandbox allow-scripts; default-src 'none'; script-src 'unsafe-inline'; \
     style-src 'unsafe-inline'; img-src data:; font-src data:; media-src data:; connect-src 'none'; \
     form-action 'none'; base-uri 'none'";

/// `Content-Security-Policy` for an uploaded HTML note (and any other inert
/// file): an opaque origin with no scripts, and nothing loaded from anywhere
/// but inline `data:` images and styles, so it cannot fetch a remote URL.
pub const UPLOADED_HTML_CSP_HEADER: &str = "sandbox; default-src 'none'; img-src data:; style-src 'unsafe-inline'";

/// Longest title (and stored file name) taken from an uploaded file's name.
pub const MAX_NOTE_FILE_TITLE: usize = 200;
/// Content of a note whose file yielded no text.
pub const NO_TEXT_PLACEHOLDER: &str = "(No text could be extracted from this file. Open it to view the original.)";
/// The instruction [`generate_note_title`] gives the model.
pub const NOTE_TITLE_PROMPT: &str =
    "Based on the Note below, please provide a Title for this content, with max 15 words";

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

/// The original file of a note created by dropping a file (`NoteAttachment`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoteAttachment {
    pub name: String,
    /// MIME type of the original file.
    #[serde(rename = "type")]
    pub mime: String,
    pub size: i64,
}

/// A note as the API reports it (`NoteResponse`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoteView {
    pub id: String,
    pub title: Option<String>,
    pub content: Option<String>,
    pub note_type: Option<String>,
    pub created: String,
    pub updated: String,
    /// The `embed_note` job queued by the save that produced this view.
    pub command_id: Option<String>,
    pub attachment: Option<NoteAttachment>,
    /// True for a generated interactive HTML explainer.
    pub explainer: bool,
    /// True for a generated static SVG infographic.
    pub infographic: bool,
}

impl NoteView {
    pub fn new(note: Note, command_id: Option<String>) -> Self {
        let attachment = match (&note.attachment_name, &note.attachment_file) {
            (Some(name), Some(file)) if !name.is_empty() && !file.is_empty() => Some(NoteAttachment {
                name: name.clone(),
                mime: note
                    .attachment_mime
                    .clone()
                    .filter(|m| !m.is_empty())
                    .unwrap_or_else(|| "application/octet-stream".to_string()),
                size: note.attachment_size.unwrap_or(0),
            }),
            _ => None,
        };
        Self {
            explainer: note.explainer.unwrap_or(false) && attachment.is_some(),
            infographic: attachment.is_some() && note.attachment_mime.as_deref() == Some(SVG_MIME),
            id: note.id,
            title: note.title,
            content: note.content,
            note_type: note.note_type,
            created: timestamp(note.created),
            updated: timestamp(note.updated),
            command_id,
            attachment,
        }
    }
}

// ---------------------------------------------------------------------------
// CRUD
// ---------------------------------------------------------------------------

async fn load_note(state: &AppState, note_id: &str) -> AppResult<Note> {
    ensure_table(note_id, Note::TABLE, NOTE_NOT_FOUND)?;
    Note::get(state, note_id).await.map_err(not_found_as(NOTE_NOT_FOUND))
}

async fn load_notebook(state: &AppState, notebook_id: &str) -> AppResult<Notebook> {
    ensure_table(notebook_id, Notebook::TABLE, NOTEBOOK_NOT_FOUND)?;
    Notebook::get(state, notebook_id).await.map_err(not_found_as(NOTEBOOK_NOT_FOUND))
}

/// `"human"` and `"ai"` are the only note types.
fn validate_note_type(note_type: &str) -> AppResult<()> {
    if matches!(note_type, "human" | "ai") {
        Ok(())
    } else {
        Err(AppError::bad_request("note_type must be 'human' or 'ai'"))
    }
}

/// Every note, most recently updated first. With a notebook id: that
/// notebook's notes, without their content (as the Python list did).
pub async fn list_notes(state: &AppState, notebook_id: Option<&str>) -> AppResult<Vec<NoteView>> {
    let notes = match notebook_id.filter(|id| !id.is_empty()) {
        Some(notebook_id) => load_notebook(state, notebook_id).await?.notes(state, false).await?,
        None => Note::list(state).await?,
    };
    Ok(notes.into_iter().map(|note| NoteView::new(note, None)).collect())
}

/// Body of `POST /notes`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NewNote {
    #[serde(default)]
    pub title: Option<String>,
    pub content: String,
    /// Absent means `"human"`; JSON `null` stores no type.
    #[serde(default, deserialize_with = "double_option", skip_serializing_if = "Option::is_none")]
    pub note_type: Option<Option<String>>,
    /// The notebook to file the note in.
    #[serde(default)]
    pub notebook_id: Option<String>,
}

/// Create a note, queue its embedding and file it in a notebook if one is given.
pub async fn create_note(state: &AppState, new: &NewNote) -> AppResult<NoteView> {
    let note_type = match &new.note_type {
        None => Some("human".to_string()),
        Some(note_type) => note_type.clone(),
    };
    let mut title = new.title.clone();
    if title.as_deref().unwrap_or_default().is_empty() && note_type.as_deref() == Some("ai") && !new.content.is_empty() {
        title = Some(generate_note_title(state, &new.content).await?);
    }
    if let Some(note_type) = note_type.as_deref() {
        validate_note_type(note_type)?;
    }
    // Checked before saving, so an unknown notebook does not leave a stray note.
    let notebook_id = new.notebook_id.as_deref().filter(|id| !id.is_empty());
    if let Some(notebook_id) = notebook_id {
        load_notebook(state, notebook_id).await?;
    }

    let mut note = Note { title, content: Some(new.content.clone()), note_type, ..Note::default() };
    let command_id = note.save(state).await?;
    if let Some(notebook_id) = notebook_id {
        note.add_to_notebook(state, notebook_id).await.map_err(not_found_as(NOTEBOOK_NOT_FOUND))?;
    }
    Ok(NoteView::new(note, command_id))
}

pub async fn get_note(state: &AppState, note_id: &str) -> AppResult<NoteView> {
    Ok(NoteView::new(load_note(state, note_id).await?, None))
}

/// Body of `PUT /notes/{id}`. Absent and `null` fields are left alone.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoteUpdate {
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub content: Option<String>,
    #[serde(default)]
    pub note_type: Option<String>,
}

/// Update a note and queue its re-embedding.
pub async fn update_note(state: &AppState, note_id: &str, update: &NoteUpdate) -> AppResult<NoteView> {
    let mut note = load_note(state, note_id).await?;
    if let Some(title) = &update.title {
        note.title = Some(title.clone());
    }
    if let Some(content) = &update.content {
        note.content = Some(content.clone());
    }
    if let Some(note_type) = &update.note_type {
        validate_note_type(note_type)?;
        note.note_type = Some(note_type.clone());
    }
    let command_id = note.save(state).await.map_err(not_found_as(NOTE_NOT_FOUND))?;
    Ok(NoteView::new(note, command_id))
}

/// Delete a note, the meeting it transcribes (if any) and its original file.
pub async fn delete_note(state: &AppState, note_id: &str) -> AppResult<()> {
    load_note(state, note_id).await?.delete(state).await
}

/// Write a title for an untitled AI note with the default transformation
/// model (Python: the `pattern/generic` prompt with [`NOTE_TITLE_PROMPT`] as
/// its instruction and the note as input, at most 5000 output tokens, with
/// thinking tags stripped from the reply). No model configured →
/// `Configuration` (422).
pub async fn generate_note_title(state: &AppState, content: &str) -> AppResult<String> {
    let system_prompt = state
        .prompts
        .render("pattern/generic", serde_json::json!({ "prompt": NOTE_TITLE_PROMPT, "input_text": content }))?;
    let messages = [nn_llm::ChatMessage::system(system_prompt), nn_llm::ChatMessage::user(content)];
    let prompt_text = format!("{}\n{}", messages[0].content, messages[1].content);
    let model = super::transformations::provision_language_model(
        state,
        &prompt_text,
        None,
        crate::domain::DefaultModelType::Transformation,
    )
    .await?;
    let reply = model.chat(&messages, &nn_llm::ChatOptions::default().with_max_tokens(5000)).await?;
    Ok(nn_content::clean_thinking_content(&reply.content))
}

// ---------------------------------------------------------------------------
// Note files
// ---------------------------------------------------------------------------

/// Extension → MIME type of the files a note can be created from. The type
/// served back always comes from here, never from the upload, so a crafted
/// file cannot choose how it is rendered.
pub const ALLOWED_TYPES: [(&str, &str); 7] = [
    (".pdf", "application/pdf"),
    (".html", "text/html"),
    (".htm", "text/html"),
    (".docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"),
    (".txt", "text/plain"),
    (".md", "text/markdown"),
    (".markdown", "text/markdown"),
];

fn allowed_mime(ext: &str) -> Option<&'static str> {
    ALLOWED_TYPES.iter().find(|(allowed, _)| *allowed == ext).map(|(_, mime)| *mime)
}

/// The final path component of a client-supplied file name, with either kind
/// of separator. The user's directories are never kept.
pub fn upload_file_name(raw: &str) -> String {
    raw.replace('\\', "/").split('/').rfind(|part| !part.is_empty() && *part != ".").unwrap_or_default().to_string()
}

/// The lower-cased extension of a file name, with its dot (`".pdf"`), or `""`.
pub fn file_extension(filename: &str) -> String {
    match filename.rfind('.') {
        Some(dot) if dot > 0 && dot < filename.len() - 1 => filename[dot..].to_lowercase(),
        _ => String::new(),
    }
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|window| window == needle)
}

/// `(extension, mime)` for an upload, or the reason it is refused.
///
/// Checks the extension against the allow-list (415 otherwise) and
/// sanity-checks the first bytes, so an executable renamed to `.pdf` is
/// rejected (400).
pub fn classify_upload(filename: &str, head: &[u8]) -> AppResult<(String, &'static str)> {
    let ext = file_extension(filename);
    let Some(mime) = allowed_mime(&ext) else {
        return Err(AppError::UnsupportedType("Unsupported file type. Use PDF, HTML, DOCX, TXT or Markdown.".to_string()));
    };
    let head = &head[..head.len().min(8192)];
    match ext.as_str() {
        ".pdf" if !contains(&head[..head.len().min(1024)], b"%PDF") => Err(AppError::invalid("This file is not a valid PDF")),
        ".docx" if !head.starts_with(b"PK\x03\x04") => Err(AppError::invalid("This file is not a valid DOCX document")),
        ".pdf" | ".docx" => Ok((ext, mime)),
        _ if head.contains(&0) => Err(AppError::invalid("This file does not look like a text document")),
        _ => Ok((ext, mime)),
    }
}

/// Store the bytes in the note files folder and return the stored name
/// (`<32 hex chars><ext>`). The user's file name is never used as a path.
pub async fn store_note_file(config: &Config, data: &[u8], ext: &str) -> AppResult<String> {
    let stored = format!("{}{ext}", uuid::Uuid::new_v4().simple());
    let path = note_file_path(config, &stored)?;
    if let Some(parent) = path.parent() {
        tokio::fs::create_dir_all(parent).await?;
    }
    tokio::fs::write(&path, data).await?;
    Ok(stored)
}

/// Extract readable text (markdown where possible) from a stored note file,
/// trimmed; `""` when the file holds no text. An unreadable file is
/// `InvalidInput` ("Could not read this file. ...").
pub async fn extract_note_text(path: &Path) -> AppResult<String> {
    let input = nn_content::ExtractInput::file(path);
    match nn_content::extract_content(input, &nn_content::ExtractContext::default()).await {
        Ok(extracted) => Ok(extracted.content.trim().to_string()),
        // A readable file without text becomes a note with the placeholder.
        Err(nn_content::ContentError::EmptyContent(_)) => Ok(String::new()),
        Err(e) => {
            let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            tracing::warn!("Text extraction failed for {name}: {e}");
            Err(AppError::invalid("Could not read this file. It may be corrupted or password protected."))
        }
    }
}

/// A file dropped onto a notebook's notes pane.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NoteUpload {
    /// The client's file name; any directories in it are discarded.
    pub filename: String,
    pub data: Vec<u8>,
    pub notebook_id: String,
}

/// Create a note from a dropped file (PDF, HTML, DOCX, TXT, Markdown).
///
/// The original is kept (served by [`note_file`]) and the text extracted from
/// it becomes the note content, so it is searchable and usable as chat context
/// like any other note. A failure at any step leaves neither a file nor a note
/// behind. Served by `POST /notes/upload`.
pub async fn upload_note_file(state: &AppState, upload: &NoteUpload) -> AppResult<NoteView> {
    load_notebook(state, &upload.notebook_id).await?;

    let filename = upload_file_name(&upload.filename);
    if upload.data.is_empty() {
        return Err(AppError::invalid("The file is empty"));
    }
    let (ext, mime) = classify_upload(&filename, &upload.data)?;
    let stored = store_note_file(&state.config, &upload.data, &ext).await?;

    let mut note = Note::default();
    match note_from_stored_file(state, &mut note, upload, &filename, &stored, mime).await {
        Ok(command_id) => Ok(NoteView::new(note, command_id)),
        Err(e) => {
            // Undo: a saved-but-unlinked note (which also removes its file), or just the file.
            if note.id.is_empty() {
                remove_note_file(&state.config, &stored);
            } else if let Err(cleanup) = note.delete(state).await {
                tracing::warn!("Failed to remove orphan note {}: {cleanup}", note.id);
                remove_note_file(&state.config, &stored);
            }
            Err(e)
        }
    }
}

async fn note_from_stored_file(
    state: &AppState,
    note: &mut Note,
    upload: &NoteUpload,
    filename: &str,
    stored: &str,
    mime: &str,
) -> AppResult<Option<String>> {
    let text = extract_note_text(&note_file_path(&state.config, stored)?).await?;
    let title: String = filename.chars().take(MAX_NOTE_FILE_TITLE).collect();
    *note = Note {
        title: Some(title.clone()),
        content: Some(if text.trim().is_empty() { NO_TEXT_PLACEHOLDER.to_string() } else { text }),
        note_type: Some("human".to_string()),
        attachment_name: Some(title),
        attachment_file: Some(stored.to_string()),
        attachment_mime: Some(mime.to_string()),
        attachment_size: Some(upload.data.len() as i64),
        ..Note::default()
    };
    let command_id = note.save(state).await?;
    note.add_to_notebook(state, &upload.notebook_id).await.map_err(not_found_as(NOTEBOOK_NOT_FOUND))?;
    Ok(command_id)
}

/// The original file of a note, ready to be served.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NoteFile {
    pub path: PathBuf,
    /// From the allow-list (by extension), not from the upload.
    pub mime: &'static str,
    /// The user's name for the file.
    pub filename: String,
    /// The `Content-Security-Policy` the response must carry, if any.
    pub content_security_policy: Option<&'static str>,
}

/// Locate the original file of a note created from a dropped or generated file.
pub async fn note_file(state: &AppState, note_id: &str) -> AppResult<NoteFile> {
    let note = load_note(state, note_id).await?;
    let stored = note
        .attachment_file
        .as_deref()
        .filter(|stored| !stored.is_empty())
        .ok_or_else(|| AppError::not_found("This note has no file"))?;
    let path = note_file_path(&state.config, stored)?;
    if !tokio::fs::metadata(&path).await.map(|meta| meta.is_file()).unwrap_or(false) {
        return Err(AppError::not_found("The file is no longer on disk"));
    }

    // SVG is not uploadable; only generated infographics are stored as .svg.
    let ext = file_extension(stored);
    let mime = if ext == ".svg" && note.attachment_mime.as_deref() == Some(SVG_MIME) {
        SVG_MIME
    } else {
        allowed_mime(&ext).unwrap_or("application/octet-stream")
    };
    let content_security_policy = match mime {
        SVG_MIME => Some(INFOGRAPHIC_CSP_HEADER),
        "text/html" if note.explainer.unwrap_or(false) => Some(EXPLAINER_CSP_HEADER),
        // Uploaded HTML must never run scripts in the app's origin.
        "text/html" => Some(UPLOADED_HTML_CSP_HEADER),
        _ => None,
    };
    Ok(NoteFile {
        path,
        mime,
        filename: note.attachment_name.filter(|name| !name.is_empty()).unwrap_or_else(|| stored.to_string()),
        content_security_policy,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    fn files_in(dir: &Path) -> usize {
        std::fs::read_dir(dir).map(|entries| entries.count()).unwrap_or(0)
    }

    #[test]
    fn uploads_are_classified_by_extension_and_first_bytes() {
        let docx = ALLOWED_TYPES[3].1;
        for (name, data, mime) in [
            ("a.pdf", &b"%PDF-1.4\n..."[..], "application/pdf"),
            ("A.PDF", b"%PDF-1.4", "application/pdf"),
            ("a.html", b"<html></html>", "text/html"),
            ("a.htm", b"<p>x</p>", "text/html"),
            ("a.docx", b"PK\x03\x04rest", docx),
            ("a.txt", b"plain text here", "text/plain"),
            ("a.md", b"# Title\n\nbody", "text/markdown"),
            ("notes.final.markdown", b"body", "text/markdown"),
        ] {
            let (ext, found) = classify_upload(name, data).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(found, mime, "{name}");
            assert_eq!(ext, file_extension(name));
        }
        for (name, data, status, detail) in [
            ("run.exe", &b"MZ\x90\x00"[..], 415, "Unsupported file type. Use PDF, HTML, DOCX, TXT or Markdown."),
            ("noext", b"hello", 415, "Unsupported file type. Use PDF, HTML, DOCX, TXT or Markdown."),
            (".txt", b"hello", 415, "Unsupported file type. Use PDF, HTML, DOCX, TXT or Markdown."),
            ("a.svg", b"<svg/>", 415, "Unsupported file type. Use PDF, HTML, DOCX, TXT or Markdown."),
            ("fake.pdf", b"not a pdf at all", 400, "This file is not a valid PDF"),
            ("fake.docx", b"not a zip", 400, "This file is not a valid DOCX document"),
            ("bin.txt", b"abc\x00def", 400, "This file does not look like a text document"),
        ] {
            let error = classify_upload(name, data).unwrap_err();
            assert_eq!(error.status().as_u16(), status, "{name}");
            assert_eq!(error.detail(), detail, "{name}");
        }
    }

    #[test]
    fn file_names_lose_their_directories() {
        assert_eq!(upload_file_name("../../etc/evil.txt"), "evil.txt");
        assert_eq!(upload_file_name("C:\\Users\\me\\report.pdf"), "report.pdf");
        assert_eq!(upload_file_name("plain.md"), "plain.md");
        assert_eq!(upload_file_name("dir/"), "dir");
        assert_eq!(upload_file_name(""), "");
        assert_eq!(file_extension("a.tar.GZ"), ".gz");
        assert_eq!(file_extension("trailing."), "");
        assert_eq!(file_extension(".hidden"), "");
    }

    #[tokio::test]
    async fn stored_files_get_a_generated_name() {
        let (state, _dir) = AppState::for_tests().await.unwrap();
        let stored = store_note_file(&state.config, b"hello", ".txt").await.unwrap();
        assert!(crate::domain::note::is_stored_note_file_name(&stored), "{stored}");
        assert_eq!(std::fs::read(note_file_path(&state.config, &stored).unwrap()).unwrap(), b"hello");
        // An extension outside the stored-name pattern is refused before anything is written.
        assert!(store_note_file(&state.config, b"x", ".tar.gz").await.is_err());
        assert_eq!(files_in(&state.config.note_files_dir()), 1);
    }

    #[tokio::test]
    async fn rejected_uploads_store_nothing() {
        let (state, _dir) = AppState::for_tests().await.unwrap();
        let notebook = Notebook::create(&state, "NB", "", None).await.unwrap();
        let upload = |filename: &str, data: &[u8], notebook_id: &str| NoteUpload {
            filename: filename.to_string(),
            data: data.to_vec(),
            notebook_id: notebook_id.to_string(),
        };

        for (name, data, status) in [
            ("run.exe", &b"MZ\x90\x00"[..], StatusCode::UNSUPPORTED_MEDIA_TYPE),
            ("noext", b"hello", StatusCode::UNSUPPORTED_MEDIA_TYPE),
            ("fake.pdf", b"not a pdf at all", StatusCode::BAD_REQUEST),
            ("fake.docx", b"not a zip", StatusCode::BAD_REQUEST),
            ("bin.txt", b"abc\x00def", StatusCode::BAD_REQUEST),
            ("empty.txt", b"", StatusCode::BAD_REQUEST),
        ] {
            let error = upload_note_file(&state, &upload(name, data, &notebook.id)).await.unwrap_err();
            assert_eq!(error.status(), status, "{name}");
        }
        let error = upload_note_file(&state, &upload("empty.txt", b"", &notebook.id)).await.unwrap_err();
        assert_eq!(error.detail(), "The file is empty");

        let error = upload_note_file(&state, &upload("a.txt", b"hello", "notebook:missing")).await.unwrap_err();
        assert_eq!((error.status(), error.detail().as_str()), (StatusCode::NOT_FOUND, "Notebook not found"));

        // Nothing was stored by any of the rejected uploads.
        assert_eq!(files_in(&state.config.note_files_dir()), 0);
        assert!(Note::list(&state).await.unwrap().is_empty());

        // A file whose text cannot be extracted is refused, and its stored
        // original is removed again.
        let error = upload_note_file(&state, &upload("broken.pdf", b"%PDF-1.4 garbage", &notebook.id)).await.unwrap_err();
        assert_eq!(error.status(), StatusCode::BAD_REQUEST, "{error}");
        assert_eq!(error.detail(), "Could not read this file. It may be corrupted or password protected.");
        assert_eq!(files_in(&state.config.note_files_dir()), 0);
        assert!(Note::list(&state).await.unwrap().is_empty());

        // A valid file becomes a note titled after its name (directories dropped).
        let view = upload_note_file(&state, &upload("../a.txt", b"hello", &notebook.id)).await.unwrap();
        assert_eq!((view.title.as_deref(), view.content.as_deref()), (Some("a.txt"), Some("hello")));
        assert_eq!(files_in(&state.config.note_files_dir()), 1);
    }

    #[tokio::test]
    async fn ai_titles_need_a_model() {
        let (state, _dir) = AppState::for_tests().await.unwrap();
        let error = generate_note_title(&state, "some content").await.unwrap_err();
        assert_eq!(error.status(), StatusCode::UNPROCESSABLE_ENTITY);
    }

    #[test]
    fn attachments_show_only_for_file_notes() {
        let plain = NoteView::new(Note { id: "note:p".into(), content: Some("c".into()), ..Note::default() }, None);
        assert_eq!(plain.attachment, None);
        assert!(!plain.explainer && !plain.infographic);

        let file_note = |mime: &str, explainer: Option<bool>| Note {
            id: "note:f".into(),
            attachment_name: Some("a.html".into()),
            attachment_file: Some(format!("{}.html", "a".repeat(32))),
            attachment_mime: Some(mime.into()),
            attachment_size: Some(12),
            explainer,
            ..Note::default()
        };
        let view = NoteView::new(file_note("text/html", Some(true)), Some("command:1".into()));
        assert_eq!(view.attachment, Some(NoteAttachment { name: "a.html".into(), mime: "text/html".into(), size: 12 }));
        assert!(view.explainer && !view.infographic);
        assert_eq!(view.command_id.as_deref(), Some("command:1"));
        assert!(NoteView::new(file_note(SVG_MIME, None), None).infographic);

        // The explainer flag means nothing without a file.
        let flagged = NoteView::new(Note { explainer: Some(true), ..Note::default() }, None);
        assert!(!flagged.explainer);
    }
}
