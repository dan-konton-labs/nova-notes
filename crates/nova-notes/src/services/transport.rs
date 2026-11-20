//! Transporting notes into a notebook: a **copy**, not a filing.
//!
//! "Transport" takes a note (or every note of a folder) from the Notes
//! library and duplicates it into a studio notebook. The copy is a new,
//! independent note: editing or deleting it never touches the original, and
//! nothing keeps the two in sync. The rich-editor document (page or edgeless)
//! travels with the copy, as does an attached original file (the file itself
//! is copied, so deleting either note cannot take the other's attachment).

use serde::{Deserialize, Serialize};

use super::notebooks::{ensure_table, not_found_as};
use super::notes::{self, NoteView};
use crate::db::Doc;
use crate::domain::{Note, Notebook};
use crate::error::{AppError, AppResult};
use crate::jobs::JobQueue;
use crate::state::AppState;

pub fn register_jobs(_jobs: &JobQueue) {}

/// What a transport did.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransportReport {
    /// Ids of the new copies, in the order they were made.
    pub copied: Vec<String>,
    /// Notes that could not be copied, as `"title: reason"` lines.
    pub failed: Vec<String>,
}

/// Copy one note into `notebook_id`. Returns the copy.
pub async fn transport_note(state: &AppState, note_id: &str, notebook_id: &str) -> AppResult<NoteView> {
    ensure_table(notebook_id, Notebook::TABLE, "Notebook not found")?;
    Notebook::get(state, notebook_id).await.map_err(not_found_as("Notebook not found"))?;
    ensure_table(note_id, Note::TABLE, "Note not found")?;
    let source = Note::get(state, note_id).await.map_err(not_found_as("Note not found"))?;
    copy_note(state, &source, notebook_id).await
}

/// Copy every note of the notebook `from_id` (the "folder" in the Notes
/// library) into `notebook_id`. Notes that fail to copy are reported, not
/// fatal, so one bad note does not stop a folder.
pub async fn transport_notebook(state: &AppState, from_id: &str, notebook_id: &str) -> AppResult<TransportReport> {
    if from_id == notebook_id {
        return Err(AppError::invalid("A folder cannot be transported into itself."));
    }
    ensure_table(notebook_id, Notebook::TABLE, "Notebook not found")?;
    Notebook::get(state, notebook_id).await.map_err(not_found_as("Notebook not found"))?;
    ensure_table(from_id, Notebook::TABLE, "Folder not found")?;
    let from = Notebook::get(state, from_id).await.map_err(not_found_as("Folder not found"))?;

    let mut report = TransportReport::default();
    for note in from.notes(state, true).await? {
        let title = note.title.clone().filter(|t| !t.trim().is_empty()).unwrap_or_else(|| "Untitled".into());
        match copy_note(state, &note, notebook_id).await {
            Ok(copy) => report.copied.push(copy.id),
            Err(e) => report.failed.push(format!("{title}: {}", e.detail())),
        }
    }
    Ok(report)
}

/// The duplication itself: content, type and attachment first (a normal
/// create, so the embedding job is queued), then the editor document
/// (page/edgeless mode, Yjs state and links), copied verbatim.
async fn copy_note(state: &AppState, source: &Note, notebook_id: &str) -> AppResult<NoteView> {
    // An attached original is copied on disk so the two notes never share a file.
    let attachment_file = match source.attachment_file.as_deref().filter(|f| !f.is_empty()) {
        Some(stored) => {
            let path = crate::domain::note::note_file_path(&state.config, stored)?;
            let data = tokio::fs::read(&path).await.map_err(|e| {
                AppError::FileOperation(format!("Could not read the attached file of {}: {e}", source.id))
            })?;
            let ext = stored.split_once('.').map(|(_, ext)| format!(".{ext}")).unwrap_or_default();
            Some(notes::store_note_file(&state.config, &data, &ext).await?)
        }
        None => None,
    };

    let mut copy = Note {
        title: source.title.clone(),
        content: source.content.clone(),
        note_type: source.note_type.clone(),
        attachment_name: source.attachment_name.clone(),
        attachment_file,
        attachment_mime: source.attachment_mime.clone(),
        attachment_size: source.attachment_size,
        explainer: source.explainer,
        ..Note::default()
    };
    copy.save(state).await?;
    copy.add_to_notebook(state, notebook_id).await?;

    // The editor document rides along, so an edgeless canvas stays a canvas.
    let doc = Note::doc_state(state, &source.id).await?;
    if doc.doc_state.is_some() || doc.doc_mode.is_some() {
        state
            .db
            .update(
                &copy.id,
                Doc::new()
                    .set_some("doc_state", doc.doc_state.as_ref())
                    .set_some("doc_hash", doc.doc_hash.as_ref())
                    .set_some("doc_mode", doc.doc_mode.as_ref()),
            )
            .await?;
    }
    Ok(NoteView::new(copy, None))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::notes::NewNote;
    use crate::testing::TestApp;

    async fn notebook(app: &TestApp, name: &str) -> String {
        let (_, nb) = app.post("/api/notebooks", serde_json::json!({ "name": name, "description": "" })).await;
        nb["id"].as_str().unwrap().to_string()
    }

    #[tokio::test]
    async fn a_transported_note_is_an_independent_copy() {
        let app = TestApp::new().await;
        let from = notebook(&app, "Library").await;
        let to = notebook(&app, "Project").await;
        let original = notes::create_note(
            &app.state,
            &NewNote { title: Some("Plan".into()), content: "Step one".into(), notebook_id: Some(from.clone()), ..Default::default() },
        )
        .await
        .unwrap();
        // Give the original an edgeless editor document.
        app.state
            .db
            .update(&original.id, Doc::new().set("doc_state", "AQID").set("doc_hash", "x").set("doc_mode", "edgeless"))
            .await
            .unwrap();

        let copy = transport_note(&app.state, &original.id, &to).await.unwrap();
        assert_ne!(copy.id, original.id);
        assert_eq!(copy.title.as_deref(), Some("Plan"));
        assert_eq!(copy.content.as_deref(), Some("Step one"));
        let filed = crate::services::workspace::note_notebook_ids(&app.state, &copy.id).await.unwrap();
        assert_eq!(filed, vec![to.clone()]);
        // The editor document (edgeless mode) travelled with the copy.
        let doc = Note::doc_state(&app.state, &copy.id).await.unwrap();
        assert_eq!(doc.doc_mode.as_deref(), Some("edgeless"));
        assert_eq!(doc.doc_state.as_deref(), Some("AQID"));

        // Editing the copy leaves the original alone; deleting it too.
        app.put(&format!("/api/notes/{}", copy.id), serde_json::json!({ "content": "Changed" })).await;
        let kept = notes::get_note(&app.state, &original.id).await.unwrap();
        assert_eq!(kept.content.as_deref(), Some("Step one"));
        notes::delete_note(&app.state, &copy.id).await.unwrap();
        assert!(notes::get_note(&app.state, &original.id).await.is_ok());
    }

    #[tokio::test]
    async fn a_folder_transport_copies_every_note() {
        let app = TestApp::new().await;
        let from = notebook(&app, "Garden").await;
        let to = notebook(&app, "Project").await;
        for title in ["Tomatoes", "Compost"] {
            notes::create_note(
                &app.state,
                &NewNote { title: Some(title.into()), content: title.into(), notebook_id: Some(from.clone()), ..Default::default() },
            )
            .await
            .unwrap();
        }
        let report = transport_notebook(&app.state, &from, &to).await.unwrap();
        assert_eq!(report.copied.len(), 2);
        assert!(report.failed.is_empty());
        let copied = notes::list_notes(&app.state, Some(&to)).await.unwrap();
        assert_eq!(copied.len(), 2);
        // The source folder still has its own notes.
        assert_eq!(notes::list_notes(&app.state, Some(&from)).await.unwrap().len(), 2);

        assert!(transport_notebook(&app.state, &from, &from).await.unwrap_err().to_string().contains("itself"));
        assert!(transport_note(&app.state, "note:nope", &to).await.unwrap_err().is_not_found());
        assert!(transport_notebook(&app.state, &from, "notebook:nope").await.unwrap_err().is_not_found());
    }
}
