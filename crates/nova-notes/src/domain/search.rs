//! Full-text and vector search over sources and notes.
//!
//! Both delegate to the SurrealQL functions the migrations define
//! (`fn::text_search`, `fn::vector_search`); rows are returned as they come.

use crate::db::{Json, Params};
use crate::error::{AppError, AppResult};
use crate::state::AppState;

/// Check that every id names an existing notebook and return the list
/// unchanged. Malformed ids → `InvalidInput`; unknown ones → `NotFound`.
pub async fn resolve_notebook_scope(state: &AppState, notebook_ids: &[String]) -> AppResult<Vec<String>> {
    if notebook_ids.is_empty() {
        return Ok(Vec::new());
    }
    let invalid: Vec<&str> = notebook_ids
        .iter()
        .map(String::as_str)
        .filter(|id| id.strip_prefix("notebook:").is_none_or(str::is_empty) || crate::db::thing(id).is_err())
        .collect();
    if !invalid.is_empty() {
        return Err(AppError::invalid(format!("Invalid notebook id(s): {}", invalid.join(", "))));
    }
    let rows = state
        .db
        .query("SELECT id FROM notebook WHERE id IN $ids", Params::new().records("ids", notebook_ids))
        .await
        .map_err(|e| AppError::database(format!("Failed to resolve notebook scope: {e}")))?;
    let found: Vec<&str> = rows.iter().filter_map(|r| r.get("id").and_then(Json::as_str)).collect();
    let missing: Vec<&str> =
        notebook_ids.iter().map(String::as_str).filter(|id| !found.contains(id)).collect();
    if !missing.is_empty() {
        return Err(AppError::not_found(format!("Notebook(s) not found: {}", missing.join(", "))));
    }
    Ok(notebook_ids.to_vec())
}

fn scope(params: Params, notebook_ids: Option<&[String]>) -> Params {
    match notebook_ids.filter(|ids| !ids.is_empty()) {
        Some(ids) => params.records("notebook_ids", ids),
        None => params.set("notebook_ids", Json::Null),
    }
}

/// BM25 keyword search. `notebook_ids` restricts results to those notebooks.
pub async fn text_search(
    state: &AppState,
    keyword: &str,
    results: usize,
    source: bool,
    note: bool,
    notebook_ids: Option<&[String]>,
) -> AppResult<Vec<Json>> {
    if keyword.is_empty() {
        return Err(AppError::invalid("Search keyword cannot be empty"));
    }
    let params = scope(
        Params::new().set("keyword", keyword).set("results", results).set("source", source).set("note", note),
        notebook_ids,
    );
    match state
        .db
        .query("SELECT * FROM fn::text_search($keyword, $results, $source, $note, $notebook_ids)", params)
        .await
    {
        Ok(rows) => Ok(rows),
        // A SurrealDB highlighter bug on some documents; semantic search still works.
        Err(e) if e.to_string().contains("position overflow") => {
            tracing::warn!("Highlight position overflow, falling back to vector search: {e}");
            vector_search(state, keyword, results, source, note, 0.2, notebook_ids).await
        }
        Err(e) => Err(e),
    }
}

/// Semantic search for `keyword` using the default embedding model.
pub async fn vector_search(
    state: &AppState,
    keyword: &str,
    results: usize,
    source: bool,
    note: bool,
    minimum_score: f64,
    notebook_ids: Option<&[String]>,
) -> AppResult<Vec<Json>> {
    if keyword.is_empty() {
        return Err(AppError::invalid("Search keyword cannot be empty"));
    }
    let embedding = crate::ai::generate_embedding(state, keyword).await?;
    vector_search_by_embedding(state, &embedding, results, source, note, minimum_score, notebook_ids).await
}

/// Semantic search with a query vector the caller already has. Results are
/// ordered by similarity, highest first (ties by id, for stable output).
pub async fn vector_search_by_embedding(
    state: &AppState,
    embedding: &[f32],
    results: usize,
    source: bool,
    note: bool,
    minimum_score: f64,
    notebook_ids: Option<&[String]>,
) -> AppResult<Vec<Json>> {
    let params = scope(
        Params::new()
            .set("embed", embedding)
            .set("results", results)
            .set("source", source)
            .set("note", note)
            .set("minimum_score", minimum_score),
        notebook_ids,
    );
    let mut rows = state
        .db
        .query("SELECT * FROM fn::vector_search($embed, $results, $source, $note, $minimum_score, $notebook_ids)", params)
        .await?;
    let similarity = |row: &Json| row.get("similarity").and_then(Json::as_f64).unwrap_or(0.0);
    let id = |row: &Json| row.get("id").and_then(Json::as_str).unwrap_or("").to_string();
    rows.sort_by(|a, b| similarity(b).total_cmp(&similarity(a)).then_with(|| id(a).cmp(&id(b))));
    Ok(rows)
}
