//! Domain behaviour against a migrated in-memory database.

use serde_json::json;

use super::*;
use crate::db::{Doc, Params};
use crate::jobs::RetryPolicy;
use crate::state::AppState;

async fn state() -> (AppState, tempfile::TempDir) {
    AppState::for_tests().await.unwrap()
}

async fn source_with_text(state: &AppState, title: &str, text: &str) -> Source {
    Source::create(
        state,
        &Source { title: Some(title.into()), full_text: Some(text.into()), topics: vec!["t".into()], ..Source::default() },
    )
    .await
    .unwrap()
}

#[tokio::test]
async fn notebooks_validate_list_and_update() {
    let (state, _dir) = state().await;
    assert!(Notebook::create(&state, "  ", "", None).await.is_err());

    let mut nb = Notebook::create(&state, "Research", "About things", None).await.unwrap();
    assert!(nb.id.starts_with("notebook:"));
    assert!(!nb.archived);
    assert!(nb.created.is_some());

    nb.archived = true;
    nb.name = "Renamed".into();
    nb.save(&state).await.unwrap();
    let again = Notebook::get(&state, &nb.id).await.unwrap();
    assert!(again.archived);
    assert_eq!(again.name, "Renamed");

    Notebook::create(&state, "Another", "", None).await.unwrap();
    let all = Notebook::list(&state, Some("name asc")).await.unwrap();
    assert_eq!(all.iter().map(|n| n.name.as_str()).collect::<Vec<_>>(), vec!["Another", "Renamed"]);
    assert!(Notebook::list(&state, Some("name; DROP")).await.is_err());
    assert!(Notebook::get(&state, "notebook:missing").await.unwrap_err().is_not_found());
}

#[tokio::test]
async fn folders_and_sections_organise_notebooks() {
    let (state, _dir) = state().await;
    let section = Section::create(&state, "Work").await.unwrap();
    assert!(Section::create(&state, "").await.is_err());
    let folder = Folder::create(&state, "Projects", Some(&section.id)).await.unwrap();
    assert_eq!(folder.section.as_deref(), Some(section.id.as_str()));

    let nb = Notebook::create(&state, "In a folder", "", Some(&folder.id)).await.unwrap();
    assert_eq!(nb.folder.as_deref(), Some(folder.id.as_str()));
    let nb = Notebook::set_folder(&state, &nb.id, None).await.unwrap();
    assert_eq!(nb.folder, None);

    let folder = Folder::set_section(&state, &folder.id, None).await.unwrap();
    assert_eq!(folder.section, None);
    assert_eq!(Folder::rename(&state, &folder.id, "Renamed").await.unwrap().name, "Renamed");
    assert_eq!(Section::list(&state).await.unwrap().len(), 1);
    assert_eq!(Folder::list(&state).await.unwrap().len(), 1);
}

#[tokio::test]
async fn a_notebook_gathers_its_sources_notes_and_context() {
    let (state, _dir) = state().await;
    let nb = Notebook::create(&state, "NB", "", None).await.unwrap();

    let source = source_with_text(&state, "Paper", "The full text of the paper.").await;
    source.add_to_notebook(&state, &nb.id).await.unwrap();
    assert!(source.add_to_notebook(&state, "notebook:missing").await.unwrap_err().is_not_found());
    state
        .db
        .create(
            "source_insight",
            Doc::new().record("source", &source.id).set("insight_type", "Summary").set("content", "It is a paper."),
        )
        .await
        .unwrap();

    let note = Note::create(&state, Some("Thought"), Some("A note body that is long enough to matter."), Some("human")).await.unwrap();
    note.add_to_notebook(&state, &nb.id).await.unwrap();

    let sources = nb.sources(&state, false).await.unwrap();
    assert_eq!(sources.len(), 1);
    assert_eq!(sources[0].title.as_deref(), Some("Paper"));
    assert_eq!(sources[0].full_text, None, "full text is omitted unless asked for");
    assert_eq!(nb.sources(&state, true).await.unwrap()[0].full_text.as_deref(), Some("The full text of the paper."));

    let notes = nb.notes(&state, false).await.unwrap();
    assert_eq!(notes.len(), 1);
    assert_eq!(notes[0].content, None);
    assert_eq!(nb.notes(&state, true).await.unwrap()[0].content.as_deref(), Some("A note body that is long enough to matter."));

    let context = nb.context(&state).await.unwrap();
    assert_eq!(
        context,
        "## Source: Paper\n\nThe full text of the paper.\n\nInsights:\n- Summary: It is a paper.\n\n## Note: Thought\n\nA note body that is long enough to matter."
    );

    assert_eq!(source.notebook_ids(&state).await.unwrap(), vec![nb.id.clone()]);
    assert_eq!(note.notebook_ids(&state).await.unwrap(), vec![nb.id.clone()]);
    assert_eq!(source.insights(&state).await.unwrap().len(), 1);
    let long = source.context(&state, true, None).await.unwrap();
    assert_eq!(long["full_text"], "The full text of the paper.");
    assert_eq!(long["insights"][0]["insight_type"], "Summary");
    assert!(source.context(&state, false, None).await.unwrap().get("full_text").is_none());
    assert_eq!(note.context(false)["content"].as_str().unwrap().chars().count(), 42);
}

#[tokio::test]
async fn deleting_a_notebook_keeps_shared_sources() {
    let (state, _dir) = state().await;
    let nb = Notebook::create(&state, "Doomed", "", None).await.unwrap();
    let other = Notebook::create(&state, "Survivor", "", None).await.unwrap();

    let exclusive = source_with_text(&state, "Exclusive", "only here").await;
    exclusive.add_to_notebook(&state, &nb.id).await.unwrap();
    let shared = source_with_text(&state, "Shared", "in both").await;
    shared.add_to_notebook(&state, &nb.id).await.unwrap();
    shared.add_to_notebook(&state, &other.id).await.unwrap();

    let note = Note::create(&state, Some("N"), Some("note content"), None).await.unwrap();
    note.add_to_notebook(&state, &nb.id).await.unwrap();
    let session = ChatSession::create(&state, Some("Chat"), None).await.unwrap();
    session.relate_to(&state, &nb.id).await.unwrap();
    StoredMessage::append(&state, &session.id, StoredMessage::HUMAN, "hello").await.unwrap();

    let preview = nb.delete_preview(&state).await.unwrap();
    assert_eq!(preview, notebook::DeletePreview { note_count: 1, exclusive_source_count: 1, shared_source_count: 1 });

    let summary = nb.delete(&state, true).await.unwrap();
    assert_eq!(
        summary,
        notebook::DeleteSummary { deleted_notes: 1, deleted_sources: 1, unlinked_sources: 1, deleted_chat_sessions: 1 }
    );
    assert!(Notebook::get(&state, &nb.id).await.unwrap_err().is_not_found());
    assert!(Source::get(&state, &exclusive.id).await.unwrap_err().is_not_found());
    assert!(Note::get(&state, &note.id).await.unwrap_err().is_not_found());
    assert!(ChatSession::get(&state, &session.id).await.unwrap_err().is_not_found());
    assert_eq!(StoredMessage::count(&state, &session.id).await.unwrap(), 0);
    // The shared source survives, still in the other notebook.
    assert_eq!(Source::get(&state, &shared.id).await.unwrap().notebook_ids(&state).await.unwrap(), vec![other.id.clone()]);
    assert_eq!(other.sources(&state, false).await.unwrap().len(), 1);
}

#[tokio::test]
async fn deleting_without_the_flag_only_unlinks_sources() {
    let (state, _dir) = state().await;
    let nb = Notebook::create(&state, "NB", "", None).await.unwrap();
    let source = source_with_text(&state, "Kept", "text").await;
    source.add_to_notebook(&state, &nb.id).await.unwrap();
    let summary = nb.delete(&state, false).await.unwrap();
    assert_eq!(summary.unlinked_sources, 1);
    assert_eq!(summary.deleted_sources, 0);
    assert!(Source::get(&state, &source.id).await.unwrap().notebook_ids(&state).await.unwrap().is_empty());
}

#[tokio::test]
async fn deleting_a_source_removes_its_file_chunks_and_insights() {
    let (state, dir) = state().await;
    let file = dir.path().join("uploads").join("doc.txt");
    std::fs::write(&file, "x").unwrap();
    let source = Source::create(
        &state,
        &Source {
            title: Some("Doc".into()),
            asset: Some(Asset { file_path: Some(file.to_string_lossy().into()), url: None }),
            ..Source::default()
        },
    )
    .await
    .unwrap();
    state
        .db
        .create(
            "source_embedding",
            Doc::new().record("source", &source.id).set("order", 0).set("content", "chunk").set("embedding", [0.1, 0.2]),
        )
        .await
        .unwrap();
    assert_eq!(source.embedded_chunks(&state).await.unwrap(), 1);

    source.delete(&state).await.unwrap();
    assert!(!file.exists());
    let left = state.db.query("SELECT * FROM source_embedding", Params::new()).await.unwrap();
    assert!(left.is_empty());
}

#[tokio::test]
async fn saving_a_note_queues_its_embedding() {
    let (state, _dir) = state().await;
    assert!(Note::create(&state, Some("Empty"), Some("   "), None).await.is_err());

    // Without the handler registered the save still succeeds.
    let mut note = Note::create(&state, Some("T"), Some("content"), Some("ai")).await.unwrap();
    assert_eq!(note.note_type.as_deref(), Some("ai"));

    state.jobs.register("embed_note", RetryPolicy::once(), |_ctx, args| async move { Ok(json!({ "embedded": args["note_id"] })) });
    note.title = Some("Changed".into());
    let command = note.save(&state).await.unwrap().expect("a job is queued once the handler exists");
    let record = state.jobs.get(&command).await.unwrap().unwrap();
    assert_eq!(record.args.unwrap()["note_id"], note.id);
    assert_eq!(Note::get(&state, &note.id).await.unwrap().title.as_deref(), Some("Changed"));
    assert_eq!(Note::list(&state).await.unwrap().len(), 1);
}

#[tokio::test]
async fn source_jobs_validate_their_input() {
    let (state, _dir) = state().await;
    state.jobs.register("embed_source", RetryPolicy::once(), |_ctx, _args| async move { Ok(json!({})) });
    state.jobs.register("create_insight", RetryPolicy::once(), |_ctx, _args| async move { Ok(json!({})) });

    let empty = Source::create(&state, &Source { title: Some("Empty".into()), ..Source::default() }).await.unwrap();
    assert!(empty.vectorize(&state).await.is_err());
    assert!(empty.add_insight(&state, "", "x").await.is_err());
    assert_eq!(empty.status(&state).await, None);

    let mut source = source_with_text(&state, "Full", "text").await;
    let command = source.vectorize(&state).await.unwrap();
    assert!(command.starts_with("command:"));
    assert!(source.add_insight(&state, "Summary", "content").await.unwrap().starts_with("command:"));

    // The processing job link round-trips as a record link.
    source.command = Some(command.clone());
    source.save(&state).await.unwrap();
    let reloaded = Source::get(&state, &source.id).await.unwrap();
    assert_eq!(reloaded.command.as_deref(), Some(command.as_str()));
    assert_eq!(reloaded.status(&state).await.as_deref(), Some("new"));
    assert_eq!(reloaded.processing_progress(&state).await.unwrap()["status"], "new");
}

#[tokio::test]
async fn insights_load_in_batches_and_become_notes() {
    let (state, _dir) = state().await;
    let nb = Notebook::create(&state, "NB", "", None).await.unwrap();
    let a = source_with_text(&state, "A", "a").await;
    let b = source_with_text(&state, "B", "b").await;
    let row = state
        .db
        .create("source_insight", Doc::new().record("source", &a.id).set("insight_type", "Key points").set("content", "Point."))
        .await
        .unwrap();

    let grouped = SourceInsight::for_sources(&state, &[a.id.clone(), b.id.clone()]).await.unwrap();
    assert_eq!(grouped[&a.id].len(), 1);
    assert!(grouped[&b.id].is_empty());
    assert!(SourceInsight::for_sources(&state, &[]).await.unwrap().is_empty());

    let insight = SourceInsight::get(&state, row["id"].as_str().unwrap()).await.unwrap();
    assert_eq!(insight.source(&state).await.unwrap().id, a.id);
    let note = insight.save_as_note(&state, Some(&nb.id)).await.unwrap();
    assert_eq!(note.title.as_deref(), Some("Key points from source A"));
    assert_eq!(nb.notes(&state, true).await.unwrap()[0].content.as_deref(), Some("Point."));
}

#[tokio::test]
async fn chat_sessions_keep_ordered_history() {
    let (state, _dir) = state().await;
    let nb = Notebook::create(&state, "NB", "", None).await.unwrap();
    let mut session = ChatSession::create(&state, Some("First"), None).await.unwrap();
    session.relate_to(&state, &nb.id).await.unwrap();
    assert!(session.belongs_to(&state, &nb.id).await.unwrap());

    StoredMessage::append(&state, &session.id, StoredMessage::HUMAN, "question").await.unwrap();
    StoredMessage::append(&state, &session.id, StoredMessage::AI, "answer").await.unwrap();
    let history = session.messages(&state).await.unwrap();
    assert_eq!(history.iter().map(|m| (m.role.as_str(), m.content.as_str())).collect::<Vec<_>>(), vec![("human", "question"), ("ai", "answer")]);
    assert_eq!(history[1].seq, 1);

    session.model_override = Some("model:abc".into());
    session.save(&state).await.unwrap();
    assert_eq!(ChatSession::get(&state, &session.id).await.unwrap().model_override.as_deref(), Some("model:abc"));
    session.model_override = None;
    session.save(&state).await.unwrap();
    assert_eq!(ChatSession::get(&state, &session.id).await.unwrap().model_override, None);

    let second = ChatSession::create(&state, Some("Second"), None).await.unwrap();
    second.relate_to(&state, &nb.id).await.unwrap();
    let sessions = nb.chat_sessions(&state).await.unwrap();
    assert_eq!(sessions.len(), 2);

    StoredMessage::clear(&state, &session.id).await.unwrap();
    assert!(session.messages(&state).await.unwrap().is_empty());
}

#[tokio::test]
async fn transformations_and_settings_round_trip() {
    let (state, _dir) = state().await;
    // The migrations seed the stock transformations and default prompt.
    let seeded = Transformation::list(&state, Some("name asc")).await.unwrap();
    assert!(!seeded.is_empty());
    assert!(DefaultPrompts::load(&state).await.unwrap().transformation_instructions.is_some());

    let mut t = Transformation {
        name: "mine".into(),
        title: "Mine".into(),
        description: "d".into(),
        prompt: "Summarise".into(),
        apply_default: true,
        ..Transformation::default()
    };
    t.save(&state).await.unwrap();
    assert!(t.id.starts_with("transformation:"));
    assert!(Transformation::defaults(&state).await.unwrap().iter().any(|d| d.id == t.id));
    t.apply_default = false;
    t.save(&state).await.unwrap();
    assert!(!Transformation::get(&state, &t.id).await.unwrap().apply_default);
    t.delete(&state).await.unwrap();

    let defaults = ContentSettings::load(&state).await.unwrap();
    assert_eq!(defaults.default_embedding_option.as_deref(), Some("ask"));
    let mut settings = defaults.clone();
    settings.default_embedding_option = Some("always".into());
    settings.youtube_preferred_languages = Some(vec!["en".into()]);
    let saved = settings.save(&state).await.unwrap();
    assert_eq!(saved, settings);
    assert_eq!(ContentSettings::load(&state).await.unwrap(), settings);
}

#[tokio::test]
async fn text_search_finds_sources_and_notes_within_scope() {
    let (state, _dir) = state().await;
    let rust = Notebook::create(&state, "Rust", "", None).await.unwrap();
    let cooking = Notebook::create(&state, "Cooking", "", None).await.unwrap();

    let source = source_with_text(&state, "Ownership explained", "Borrowing and lifetimes in the Rust language.").await;
    source.add_to_notebook(&state, &rust.id).await.unwrap();
    let note = Note::create(&state, Some("Sourdough"), Some("Flour, water, salt and patience."), None).await.unwrap();
    note.add_to_notebook(&state, &cooking.id).await.unwrap();

    let hits = search::text_search(&state, "lifetimes", 10, true, true, None).await.unwrap();
    assert_eq!(hits.len(), 1, "{hits:?}");
    assert_eq!(hits[0]["id"], source.id);

    let hits = search::text_search(&state, "sourdough", 10, true, true, None).await.unwrap();
    assert_eq!(hits.len(), 1, "{hits:?}");
    assert_eq!(hits[0]["id"], note.id);

    // Scoped to the other notebook, the source is not found.
    let scope = vec![cooking.id.clone()];
    assert!(search::text_search(&state, "lifetimes", 10, true, true, Some(&scope)).await.unwrap().is_empty());
    assert!(search::text_search(&state, "", 10, true, true, None).await.is_err());

    // An insight hit carries its source as `parent_id` (migration 25), so
    // the UI can link it to the source page.
    let insight = state
        .db
        .create("source_insight", Doc::new().record("source", &source.id).set("insight_type", "Key points").set("content", "Zygomorphic petals."))
        .await
        .unwrap();
    let hits = search::text_search(&state, "zygomorphic", 10, true, false, None).await.unwrap();
    assert_eq!(hits.len(), 1, "{hits:?}");
    assert_eq!(hits[0]["id"], insight["id"]);
    assert_eq!(hits[0]["parent_id"], source.id);

    assert_eq!(search::resolve_notebook_scope(&state, &scope).await.unwrap(), scope);
    assert!(search::resolve_notebook_scope(&state, &["source:x".into()]).await.is_err());
    assert!(search::resolve_notebook_scope(&state, &["notebook:missing".into()]).await.unwrap_err().is_not_found());
}

#[tokio::test]
async fn vector_search_ranks_by_similarity() {
    let (state, _dir) = state().await;
    let near = source_with_text(&state, "Near", "near").await;
    let far = source_with_text(&state, "Far", "far").await;
    for (source, vector) in [(&near, [1.0, 0.0, 0.0]), (&far, [0.6, 0.8, 0.0])] {
        state
            .db
            .create(
                "source_embedding",
                Doc::new().record("source", &source.id).set("order", 0).set("content", "chunk").set("embedding", vector),
            )
            .await
            .unwrap();
    }
    let hits = search::vector_search_by_embedding(&state, &[1.0, 0.0, 0.0], 10, true, true, 0.2, None).await.unwrap();
    assert_eq!(hits.len(), 2, "{hits:?}");
    assert!(hits[0]["similarity"].as_f64().unwrap() > hits[1]["similarity"].as_f64().unwrap());
    // A high threshold drops the weaker match.
    let hits = search::vector_search_by_embedding(&state, &[1.0, 0.0, 0.0], 10, true, true, 0.9, None).await.unwrap();
    assert_eq!(hits.len(), 1, "{hits:?}");
}
