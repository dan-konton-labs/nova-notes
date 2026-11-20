//! The database layer: one shared SurrealDB connection and a small,
//! JSON-in/JSON-out repository API on top of it.
//!
//! This is the Rust counterpart of `open_notebook/database/repository.py`.
//! Rows come back as `serde_json::Value` with record ids rendered as
//! `"table:id"` strings and datetimes as RFC 3339 strings, exactly what the
//! Python `parse_record_ids` produced, so the rest of the app can deserialize
//! them into plain structs.
//!
//! Going the other way SurrealDB is strict about types: a `record<source>`
//! field rejects the string `"source:abc"`. [`Params`] and [`Doc`] are the two
//! builders that let callers say which values are record links or datetimes;
//! everything else is converted from JSON as-is (with JSON `null` becoming
//! SurrealDB `NONE`, which is what `option<T>` fields accept).

pub mod migrate;

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use chrono::{DateTime, SecondsFormat, Utc};
use serde::de::DeserializeOwned;
use serde::Serialize;
use surrealdb::engine::any::Any;
use surrealdb::opt::auth::Root;
use surrealdb::sql::{self, Id, Thing, Value as SqlValue};
use surrealdb::Surreal;

use crate::config::Config;
use crate::error::{AppError, AppResult};

pub type Json = serde_json::Value;

// ---------------------------------------------------------------------------
// Record ids and identifiers
// ---------------------------------------------------------------------------

/// True for a bare table / relation name that is safe to splice into a query.
pub fn is_safe_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

/// Validate a table / relation name before it is spliced into a query.
pub fn ensure_safe_identifier<'a>(value: &'a str, kind: &str) -> AppResult<&'a str> {
    if is_safe_identifier(value) {
        Ok(value)
    } else {
        Err(AppError::invalid(format!("Invalid {kind} name: {value:?}")))
    }
}

/// Split `"table:id"` into its parts. Brackets or backticks around the id
/// (SurrealDB's escaping for ids with punctuation) are removed.
pub fn split_record_id(id: &str) -> AppResult<(&str, &str)> {
    let (table, key) = id
        .split_once(':')
        .filter(|(t, k)| !t.is_empty() && !k.is_empty())
        .ok_or_else(|| AppError::invalid(format!("Invalid record id: {id:?}")))?;
    ensure_safe_identifier(table, "table")?;
    let key = key
        .strip_prefix('⟨')
        .and_then(|k| k.strip_suffix('⟩'))
        .or_else(|| key.strip_prefix('`').and_then(|k| k.strip_suffix('`')))
        .unwrap_or(key);
    Ok((table, key))
}

/// The table part of a `"table:id"` string, if it has one.
pub fn table_of(id: &str) -> Option<&str> {
    id.split_once(':').map(|(t, _)| t)
}

/// Prefix `id` with `table:` unless it already carries that table.
pub fn with_table(table: &str, id: &str) -> String {
    if id.starts_with(&format!("{table}:")) {
        id.to_string()
    } else {
        format!("{table}:{id}")
    }
}

/// `"table:id"` → a SurrealDB record link. The key is always a string key,
/// matching how the Python client parsed ids.
pub fn thing(id: &str) -> AppResult<Thing> {
    let (table, key) = split_record_id(id)?;
    Ok(Thing::from((table.to_string(), Id::String(key.to_string()))))
}

fn thing_to_string(thing: &Thing) -> String {
    match &thing.id {
        Id::String(s) => format!("{}:{}", thing.tb, s),
        Id::Number(n) => format!("{}:{}", thing.tb, n),
        _ => thing.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Value conversion
// ---------------------------------------------------------------------------

/// SurrealDB value → JSON. Record ids become `"table:id"`, datetimes RFC 3339.
pub fn to_json(value: SqlValue) -> Json {
    match value {
        SqlValue::None | SqlValue::Null => Json::Null,
        SqlValue::Bool(b) => Json::Bool(b),
        SqlValue::Number(sql::Number::Int(i)) => Json::from(i),
        SqlValue::Number(sql::Number::Float(f)) => serde_json::Number::from_f64(f).map(Json::Number).unwrap_or(Json::Null),
        SqlValue::Strand(s) => Json::String(s.0),
        SqlValue::Datetime(d) => Json::String(d.0.to_rfc3339_opts(SecondsFormat::AutoSi, true)),
        SqlValue::Uuid(u) => Json::String(u.0.to_string()),
        SqlValue::Thing(t) => Json::String(thing_to_string(&t)),
        SqlValue::Array(a) => Json::Array(a.0.into_iter().map(to_json).collect()),
        SqlValue::Object(o) => Json::Object(o.0.into_iter().map(|(k, v)| (k, to_json(v))).collect()),
        other => other.into_json(),
    }
}

/// JSON → SurrealDB value. `null` becomes `NONE`; strings stay strings (use
/// [`Params::record`] / [`Doc::record`] for record links).
pub fn from_json(value: Json) -> SqlValue {
    match value {
        Json::Null => SqlValue::None,
        Json::Bool(b) => SqlValue::Bool(b),
        Json::Number(n) => {
            if let Some(i) = n.as_i64() {
                SqlValue::from(i)
            } else {
                SqlValue::from(n.as_f64().unwrap_or(0.0))
            }
        }
        Json::String(s) => SqlValue::from(s),
        Json::Array(items) => SqlValue::from(items.into_iter().map(from_json).collect::<Vec<_>>()),
        Json::Object(map) => SqlValue::from(map.into_iter().map(|(k, v)| (k, from_json(v))).collect::<BTreeMap<_, _>>()),
    }
}

fn serialize(value: impl Serialize) -> AppResult<SqlValue> {
    Ok(from_json(serde_json::to_value(value)?))
}

fn datetime_value(dt: DateTime<Utc>) -> SqlValue {
    SqlValue::Datetime(sql::Datetime::from(dt))
}

// ---------------------------------------------------------------------------
// Doc: a record body
// ---------------------------------------------------------------------------

/// The body of a record being written (`CONTENT` / `MERGE`).
///
/// ```ignore
/// let doc = Doc::new()
///     .set("title", &title)
///     .set_some("topics", topics)        // skipped when None
///     .record("source", &source_id)      // record link
///     .none("embedding");                // explicit NONE (clears the field)
/// ```
#[derive(Debug, Default, Clone)]
pub struct Doc {
    fields: BTreeMap<String, SqlValue>,
    error: Option<AppError>,
}

impl Doc {
    pub fn new() -> Self {
        Self::default()
    }

    /// Every field of a serializable struct / JSON object. Nulls become NONE.
    pub fn from(value: impl Serialize) -> Self {
        let mut doc = Self::new();
        match serde_json::to_value(value) {
            Ok(Json::Object(map)) => {
                for (k, v) in map {
                    doc.fields.insert(k, from_json(v));
                }
            }
            Ok(other) => doc.error = Some(AppError::internal(format!("Record body must be an object, got {other}"))),
            Err(e) => doc.error = Some(e.into()),
        }
        doc
    }

    /// Like [`Doc::from`] but drops null fields, the way the Python models
    /// skipped `None` attributes when saving.
    pub fn from_non_null(value: impl Serialize) -> Self {
        let mut doc = Self::from(value);
        doc.fields.retain(|_, v| !matches!(v, SqlValue::None));
        doc
    }

    pub fn set(mut self, key: &str, value: impl Serialize) -> Self {
        match serialize(value) {
            Ok(v) => {
                self.fields.insert(key.to_string(), v);
            }
            Err(e) => self.error = Some(e),
        }
        self
    }

    /// Set the field only when the value is `Some`.
    pub fn set_some<T: Serialize>(self, key: &str, value: Option<T>) -> Self {
        match value {
            Some(v) => self.set(key, v),
            None => self,
        }
    }

    /// A record link field.
    pub fn record(mut self, key: &str, id: impl AsRef<str>) -> Self {
        match thing(id.as_ref()) {
            Ok(t) => {
                self.fields.insert(key.to_string(), SqlValue::Thing(t));
            }
            Err(e) => self.error = Some(e),
        }
        self
    }

    /// A record link, or NONE (clearing the field) when `id` is `None`.
    pub fn record_or_none(self, key: &str, id: Option<impl AsRef<str>>) -> Self {
        match id {
            Some(id) => self.record(key, id),
            None => self.none(key),
        }
    }

    pub fn datetime(mut self, key: &str, dt: DateTime<Utc>) -> Self {
        self.fields.insert(key.to_string(), datetime_value(dt));
        self
    }

    /// Explicit NONE: on a MERGE this removes the field.
    pub fn none(mut self, key: &str) -> Self {
        self.fields.insert(key.to_string(), SqlValue::None);
        self
    }

    /// Escape hatch for a value built by hand.
    pub fn raw(mut self, key: &str, value: SqlValue) -> Self {
        self.fields.insert(key.to_string(), value);
        self
    }

    pub fn remove(mut self, key: &str) -> Self {
        self.fields.remove(key);
        self
    }

    pub fn contains(&self, key: &str) -> bool {
        self.fields.contains_key(key)
    }

    pub fn is_empty(&self) -> bool {
        self.fields.is_empty()
    }

    fn into_value(self) -> AppResult<SqlValue> {
        match self.error {
            Some(e) => Err(e),
            None => Ok(SqlValue::from(self.fields)),
        }
    }
}

// ---------------------------------------------------------------------------
// Params: query variables
// ---------------------------------------------------------------------------

/// Variables bound into a query (`$name`).
///
/// ```ignore
/// db.query(
///     "SELECT * FROM source WHERE id IN $ids AND title = $title",
///     Params::new().records("ids", &ids).set("title", title),
/// ).await?
/// ```
#[derive(Debug, Default, Clone)]
pub struct Params {
    vars: Vec<(String, SqlValue)>,
    error: Option<AppError>,
}

impl Params {
    pub fn new() -> Self {
        Self::default()
    }

    /// Any serializable value, converted from JSON (null → NONE).
    pub fn set(mut self, name: &str, value: impl Serialize) -> Self {
        match serialize(value) {
            Ok(v) => self.vars.push((name.to_string(), v)),
            Err(e) => self.error = Some(e),
        }
        self
    }

    /// A `"table:id"` string bound as a record link.
    pub fn record(mut self, name: &str, id: impl AsRef<str>) -> Self {
        match thing(id.as_ref()) {
            Ok(t) => self.vars.push((name.to_string(), SqlValue::Thing(t))),
            Err(e) => self.error = Some(e),
        }
        self
    }

    /// A record link, or NONE when `id` is `None`.
    pub fn record_opt(mut self, name: &str, id: Option<impl AsRef<str>>) -> Self {
        match id {
            Some(id) => self.record(name, id),
            None => {
                self.vars.push((name.to_string(), SqlValue::None));
                self
            }
        }
    }

    /// Several ids bound as an array of record links.
    pub fn records<I, S>(mut self, name: &str, ids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let mut things = Vec::new();
        for id in ids {
            match thing(id.as_ref()) {
                Ok(t) => things.push(SqlValue::Thing(t)),
                Err(e) => {
                    self.error = Some(e);
                    return self;
                }
            }
        }
        self.vars.push((name.to_string(), SqlValue::from(things)));
        self
    }

    pub fn datetime(mut self, name: &str, dt: DateTime<Utc>) -> Self {
        self.vars.push((name.to_string(), datetime_value(dt)));
        self
    }

    /// A record body.
    pub fn doc(mut self, name: &str, doc: Doc) -> Self {
        match doc.into_value() {
            Ok(v) => self.vars.push((name.to_string(), v)),
            Err(e) => self.error = Some(e),
        }
        self
    }

    /// Several record bodies (bulk insert).
    pub fn docs(mut self, name: &str, docs: Vec<Doc>) -> Self {
        let mut values = Vec::with_capacity(docs.len());
        for doc in docs {
            match doc.into_value() {
                Ok(v) => values.push(v),
                Err(e) => {
                    self.error = Some(e);
                    return self;
                }
            }
        }
        self.vars.push((name.to_string(), SqlValue::from(values)));
        self
    }

    /// Escape hatch for a value built by hand.
    pub fn raw(mut self, name: &str, value: SqlValue) -> Self {
        self.vars.push((name.to_string(), value));
        self
    }
}

// ---------------------------------------------------------------------------
// Db
// ---------------------------------------------------------------------------

/// Variable names SurrealDB sets itself; binding one fails at run time.
const PROTECTED_VARIABLES: [&str; 12] =
    ["session", "auth", "token", "scope", "access", "before", "after", "value", "input", "this", "parent", "event"];

/// A handle to the database. Cheap to clone; all clones share one connection.
#[derive(Clone)]
pub struct Db {
    client: Surreal<Any>,
    record_locks: Arc<RecordLocks>,
}

/// One async mutex per record that has a writer holding or waiting for it.
#[derive(Default)]
struct RecordLocks(std::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>);

/// Held while writing a record; see [`Db::lock_record`].
pub struct RecordGuard {
    guard: Option<tokio::sync::OwnedMutexGuard<()>>,
    locks: Arc<RecordLocks>,
    id: String,
}

impl Drop for RecordGuard {
    fn drop(&mut self) {
        let mut map = self.locks.0.lock().expect("record lock map poisoned");
        // The map and this guard are the only holders: nobody is waiting, so
        // the entry can go. A waiter holds a clone, which keeps it.
        if map.get(&self.id).is_some_and(|m| Arc::strong_count(m) <= 2) {
            map.remove(&self.id);
        }
        self.guard.take();
    }
}

impl std::fmt::Debug for Db {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Db")
    }
}

/// `ws://host:8000/rpc` is what the Python client wants; the Rust client adds
/// the `/rpc` path itself.
fn normalize_url(url: &str) -> String {
    let url = url.trim();
    let is_remote = ["ws://", "wss://", "http://", "https://"].iter().any(|p| url.starts_with(p));
    if is_remote {
        url.trim_end_matches('/').trim_end_matches("/rpc").trim_end_matches('/').to_string()
    } else {
        url.to_string()
    }
}

fn is_embedded(url: &str) -> bool {
    !["ws://", "wss://", "http://", "https://"].iter().any(|p| url.starts_with(p))
}

impl Db {
    /// Wait for, then hold, this process's write lock on one record.
    ///
    /// SurrealDB can keep the older of two overlapping writes to the same
    /// record without reporting a conflict: when a background job's write (an
    /// `embed_note` storing the embedding) commits while a user's edit of that
    /// note is in flight, the note silently keeps its old content. Writers that
    /// can meet a background job on the same record hold this lock around their
    /// write so the two never overlap. It covers this process only, which
    /// runs both the API and the worker.
    pub async fn lock_record(&self, id: &str) -> RecordGuard {
        let mutex = self.record_locks.0.lock().expect("record lock map poisoned").entry(id.to_string()).or_default().clone();
        RecordGuard { guard: Some(mutex.lock_owned().await), locks: self.record_locks.clone(), id: id.to_string() }
    }

    /// Connect using the app configuration: SURREAL_URL when set (a running
    /// SurrealDB v2 server), otherwise the embedded engine in the data folder.
    pub async fn connect(config: &Config) -> AppResult<Self> {
        let url = normalize_url(&config.database_url());
        if let Some(path) = url.strip_prefix("surrealkv://") {
            if let Some(parent) = std::path::Path::new(path).parent() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let client = surrealdb::engine::any::connect(url.as_str())
            .await
            .map_err(|e| AppError::database(format!("Could not connect to the database at {url}: {e}")))?;
        if !is_embedded(&url) {
            let username = config.surreal_user.clone().unwrap_or_else(|| "root".to_string());
            client
                .signin(Root { username: &username, password: &config.surreal_password })
                .await
                .map_err(|e| AppError::database(format!("Database sign-in failed: {e}")))?;
        }
        client
            .use_ns(config.surreal_namespace.as_str())
            .use_db(config.surreal_database.as_str())
            .await?;
        Ok(Self { client, record_locks: Arc::default() })
    }

    /// A fresh in-memory database (no migrations applied). For tests.
    pub async fn memory() -> AppResult<Self> {
        let client = surrealdb::engine::any::connect("mem://").await?;
        client.use_ns("open_notebook").use_db("open_notebook").await?;
        Ok(Self { client, record_locks: Arc::default() })
    }

    /// Run a query and return the raw result of every statement, in order.
    /// Fails if any statement failed.
    pub async fn query_all(&self, sql: &str, params: Params) -> AppResult<Vec<Json>> {
        let (results, errors) = self.execute(sql, params).await?;
        if let Some((index, message)) = errors.into_iter().next() {
            tracing::debug!(statement = index, "query failed: {message}");
            return Err(AppError::database(message));
        }
        Ok(results)
    }

    /// Run a query and return each statement's result alongside the errors of
    /// the statements that failed (their result slot is `null`). Used by the
    /// migration runner, which tolerates individual statement failures.
    pub async fn execute(&self, sql: &str, params: Params) -> AppResult<(Vec<Json>, Vec<(usize, String)>)> {
        if let Some(e) = params.error {
            return Err(e);
        }
        if let Some((name, _)) = params.vars.iter().find(|(name, _)| PROTECTED_VARIABLES.contains(&name.as_str())) {
            return Err(AppError::internal(format!("${name} is reserved by SurrealDB and cannot be used as a query variable")));
        }
        let mut query = self.client.query(sql);
        for (name, value) in params.vars {
            query = query.bind((name, value));
        }
        let mut response = query.await?;
        let count = response.num_statements();
        let mut errors: Vec<(usize, String)> =
            response.take_errors().into_iter().map(|(i, e)| (i, e.to_string())).collect();
        errors.sort_by_key(|(i, _)| *i);
        let mut results = Vec::with_capacity(count);
        for index in 0..count {
            if errors.iter().any(|(i, _)| *i == index) {
                results.push(Json::Null);
                continue;
            }
            let value: surrealdb::Value = response.take(index)?;
            results.push(to_json(value.into_inner()));
        }
        Ok((results, errors))
    }

    /// Run a query and return the raw result of statement `index`.
    pub async fn query_statement(&self, sql: &str, params: Params, index: usize) -> AppResult<Json> {
        let mut results = self.query_all(sql, params).await?;
        if index < results.len() {
            Ok(results.swap_remove(index))
        } else {
            Ok(Json::Null)
        }
    }

    /// Run a query and return the raw result of its first statement.
    pub async fn query_value(&self, sql: &str, params: Params) -> AppResult<Json> {
        self.query_statement(sql, params, 0).await
    }

    /// Run a query and return the rows of its first statement. A statement
    /// that yields a single non-array value gives a one-element vector; NONE
    /// gives an empty one. This is the equivalent of Python's `repo_query`.
    pub async fn query(&self, sql: &str, params: Params) -> AppResult<Vec<Json>> {
        Ok(rows(self.query_value(sql, params).await?))
    }

    /// Like [`query`](Self::query) but for the last statement, for queries
    /// that open with `LET` statements.
    pub async fn query_last(&self, sql: &str, params: Params) -> AppResult<Vec<Json>> {
        let mut results = self.query_all(sql, params).await?;
        Ok(rows(results.pop().unwrap_or(Json::Null)))
    }

    /// Run a query and deserialize the rows of its first statement.
    pub async fn query_as<T: DeserializeOwned>(&self, sql: &str, params: Params) -> AppResult<Vec<T>> {
        self.query(sql, params)
            .await?
            .into_iter()
            .map(|row| serde_json::from_value(row).map_err(AppError::from))
            .collect()
    }

    /// Fetch one record by id, or `None` when it does not exist.
    pub async fn select(&self, id: &str) -> AppResult<Option<Json>> {
        let mut found = self.query("SELECT * FROM $id", Params::new().record("id", id)).await?;
        Ok(if found.is_empty() { None } else { Some(found.swap_remove(0)) })
    }

    /// Fetch one record by id and deserialize it. Missing → `NotFound`.
    pub async fn get<T: DeserializeOwned>(&self, id: &str) -> AppResult<T> {
        match self.select(id).await? {
            Some(row) => Ok(serde_json::from_value(row)?),
            None => {
                let table = table_of(id).unwrap_or(id);
                Err(AppError::not_found(format!("{table} with id {id} not found")))
            }
        }
    }

    /// Whether a record exists.
    pub async fn exists(&self, id: &str) -> AppResult<bool> {
        Ok(self.select(id).await?.is_some())
    }

    /// Every record of a table. `order_by` must already be validated (see
    /// [`validate_order_by`]).
    pub async fn select_all(&self, table: &str, order_by: Option<&str>) -> AppResult<Vec<Json>> {
        ensure_safe_identifier(table, "table")?;
        let sql = match order_by {
            Some(order) => format!("SELECT * FROM {table} ORDER BY {}", validate_order_by(order)?),
            None => format!("SELECT * FROM {table}"),
        };
        self.query(&sql, Params::new()).await
    }

    /// Create a record with a generated id. Sets `created` and `updated`.
    pub async fn create(&self, table: &str, doc: Doc) -> AppResult<Json> {
        ensure_safe_identifier(table, "table")?;
        let now = Utc::now();
        let doc = doc.remove("id").datetime("created", now).datetime("updated", now);
        let created = self
            .query(&format!("CREATE {table} CONTENT $data"), Params::new().doc("data", doc))
            .await?;
        created.into_iter().next().ok_or_else(|| AppError::database("Failed to create record"))
    }

    /// Create a record under a specific id. Sets `created` and `updated`.
    pub async fn create_with_id(&self, id: &str, doc: Doc) -> AppResult<Json> {
        let now = Utc::now();
        let doc = doc.remove("id").datetime("created", now).datetime("updated", now);
        let created = self
            .query("CREATE $id CONTENT $data", Params::new().record("id", id).doc("data", doc))
            .await?;
        created.into_iter().next().ok_or_else(|| AppError::database("Failed to create record"))
    }

    /// Insert many records in batches. Returns the created rows.
    pub async fn insert_many(&self, table: &str, docs: Vec<Doc>, batch_size: usize) -> AppResult<Vec<Json>> {
        ensure_safe_identifier(table, "table")?;
        let mut created = Vec::with_capacity(docs.len());
        let mut docs = docs.into_iter().peekable();
        while docs.peek().is_some() {
            let batch: Vec<Doc> = docs.by_ref().take(batch_size.max(1)).collect();
            let rows = self
                .query(&format!("INSERT INTO {table} $data"), Params::new().docs("data", batch))
                .await?;
            created.extend(rows);
        }
        Ok(created)
    }

    /// Merge fields into an existing record. Sets `updated`. Missing → `NotFound`.
    pub async fn update(&self, id: &str, doc: Doc) -> AppResult<Json> {
        let doc = doc.remove("id").datetime("updated", Utc::now());
        let updated = self
            .query("UPDATE $id MERGE $data", Params::new().record("id", id).doc("data", doc))
            .await?;
        updated.into_iter().next().ok_or_else(|| {
            let table = table_of(id).unwrap_or(id);
            AppError::not_found(format!("{table} with id {id} not found"))
        })
    }

    /// Create-or-merge a record under a specific id (settings singletons).
    pub async fn upsert(&self, id: &str, doc: Doc) -> AppResult<Json> {
        let rows = self
            .query("UPSERT $id MERGE $data", Params::new().record("id", id).doc("data", doc.remove("id")))
            .await?;
        rows.into_iter().next().ok_or_else(|| AppError::database("Failed to upsert record"))
    }

    /// Delete a record. Deleting a missing record is not an error.
    pub async fn delete(&self, id: &str) -> AppResult<()> {
        self.query("DELETE $id", Params::new().record("id", id)).await?;
        Ok(())
    }

    /// `RELATE from->relation->to`, with optional edge data.
    pub async fn relate(&self, from: &str, relation: &str, to: &str, data: Doc) -> AppResult<Vec<Json>> {
        ensure_safe_identifier(relation, "relationship")?;
        self.query(
            &format!("RELATE $source->{relation}->$target CONTENT $data"),
            Params::new().record("source", from).record("target", to).doc("data", data),
        )
        .await
    }

    /// Whether the database answers at all.
    pub async fn ping(&self) -> AppResult<()> {
        self.query_value("RETURN true", Params::new()).await.map(|_| ())
    }
}

fn rows(value: Json) -> Vec<Json> {
    match value {
        Json::Array(items) => items,
        Json::Null => Vec::new(),
        other => vec![other],
    }
}

/// Validate and normalise an `ORDER BY` clause supplied by a client:
/// `field`, `field asc|desc`, or a comma-separated list of those.
pub fn validate_order_by(order_by: &str) -> AppResult<String> {
    fn is_field(s: &str) -> bool {
        let mut chars = s.chars();
        matches!(chars.next(), Some(c) if c.is_ascii_lowercase() || c == '_')
            && chars.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
    }
    let mut clauses = Vec::new();
    for clause in order_by.split(',') {
        let clause = clause.trim();
        let parts: Vec<String> = clause.split_whitespace().map(str::to_ascii_lowercase).collect();
        match parts.as_slice() {
            [field] if is_field(field) => clauses.push(field.clone()),
            [field] => return Err(AppError::invalid(format!("Invalid order_by field: '{field}'"))),
            [field, direction] if is_field(field) && (direction == "asc" || direction == "desc") => {
                clauses.push(format!("{field} {direction}"))
            }
            _ => return Err(AppError::invalid(format!("Invalid order_by clause: '{clause}'"))),
        }
    }
    Ok(clauses.join(", "))
}

/// Run a database write again when SurrealDB reports a read/write conflict
/// (usually a background job touched the same record at the same moment).
/// Nothing of a conflicting transaction is committed, so the closure must be
/// safe to repeat: one write, or a transaction that re-checks its guards.
pub async fn retry_conflicts<F, Fut, T>(mut run: F) -> AppResult<T>
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = AppResult<T>>,
{
    const ATTEMPTS: u32 = 5;
    let mut attempt = 1;
    loop {
        match run().await {
            Err(AppError::Database { retryable: true, .. }) if attempt < ATTEMPTS => {
                tracing::debug!("Retrying a database write after a transaction conflict (attempt {attempt})");
                tokio::time::sleep(std::time::Duration::from_millis(25 * u64::from(attempt))).await;
                attempt += 1;
            }
            result => return result,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[tokio::test]
    async fn record_locks_serialise_writers_of_one_record() {
        let db = Db::memory().await.unwrap();
        let first = db.lock_record("note:a").await;
        // Another record is not blocked.
        drop(db.lock_record("note:b").await);
        let waiter = {
            let db = db.clone();
            tokio::spawn(async move { db.lock_record("note:a").await })
        };
        tokio::time::sleep(std::time::Duration::from_millis(20)).await;
        assert!(!waiter.is_finished(), "the second writer waits");
        drop(first);
        drop(waiter.await.unwrap());
        assert!(db.record_locks.0.lock().unwrap().is_empty(), "unused locks are dropped");
    }

    #[tokio::test]
    async fn retry_conflicts_retries_only_conflicts() {
        let conflict = || AppError::database("Failed to commit transaction due to a read or write conflict. This transaction can be retried");
        let mut calls = 0;
        let result = retry_conflicts(|| {
            calls += 1;
            let n = calls;
            async move { if n < 3 { Err(conflict()) } else { Ok(n) } }
        })
        .await;
        assert_eq!(result.unwrap(), 3);

        let mut calls = 0;
        let result: AppResult<()> = retry_conflicts(|| {
            calls += 1;
            async { Err(AppError::invalid("bad")) }
        })
        .await;
        assert!(result.is_err());
        assert_eq!(calls, 1, "other errors are not retried");

        let mut calls = 0;
        let result: AppResult<()> = retry_conflicts(|| {
            calls += 1;
            async { Err(conflict()) }
        })
        .await;
        assert!(result.unwrap_err().is_retryable());
        assert_eq!(calls, 5, "gives up after five attempts");
    }

    #[test]
    fn record_ids_split_and_escape() {
        assert_eq!(split_record_id("source:abc123").unwrap(), ("source", "abc123"));
        assert_eq!(split_record_id("source:⟨a-b⟩").unwrap(), ("source", "a-b"));
        assert_eq!(split_record_id("source:`a-b`").unwrap(), ("source", "a-b"));
        assert!(split_record_id("source").is_err());
        assert!(split_record_id("source:").is_err());
        assert!(split_record_id("bad table:1").is_err());
        assert_eq!(with_table("note", "abc"), "note:abc");
        assert_eq!(with_table("note", "note:abc"), "note:abc");
    }

    #[test]
    fn order_by_is_an_allowlist() {
        assert_eq!(validate_order_by("updated desc").unwrap(), "updated desc");
        assert_eq!(validate_order_by("Name ASC, created").unwrap(), "name asc, created");
        assert!(validate_order_by("updated; DROP TABLE x").is_err());
        assert!(validate_order_by("updated sideways").is_err());
        assert!(validate_order_by("a b c").is_err());
    }

    #[test]
    fn json_round_trip_maps_null_to_none() {
        assert!(matches!(from_json(Json::Null), SqlValue::None));
        let back = to_json(from_json(json!({"a": 1, "b": [1.5, "x", null], "c": {"d": true}})));
        assert_eq!(back, json!({"a": 1, "b": [1.5, "x", null], "c": {"d": true}}));
    }

    #[tokio::test]
    async fn crud_on_a_schemaless_table() {
        let db = Db::memory().await.unwrap();
        let created = db.create("thing", Doc::new().set("name", "one").set("n", 1)).await.unwrap();
        let id = created["id"].as_str().unwrap().to_string();
        assert!(id.starts_with("thing:"));
        assert!(created["created"].is_string());

        let updated = db.update(&id, Doc::new().set("n", 2).none("name")).await.unwrap();
        assert_eq!(updated["n"], 2);
        assert!(updated.get("name").is_none());

        assert!(db.exists(&id).await.unwrap());
        assert!(db.update("thing:missing", Doc::new().set("n", 3)).await.unwrap_err().is_not_found());

        db.delete(&id).await.unwrap();
        assert!(db.select(&id).await.unwrap().is_none());
        assert!(db.get::<Json>(&id).await.unwrap_err().is_not_found());
    }

    #[tokio::test]
    async fn record_links_bind_as_records() {
        let db = Db::memory().await.unwrap();
        db.query_all("DEFINE TABLE parent SCHEMAFULL; DEFINE TABLE child SCHEMAFULL; DEFINE FIELD parent ON child TYPE record<parent>; DEFINE FIELD label ON child TYPE option<string>;", Params::new())
            .await
            .unwrap();
        let parent = db.create("parent", Doc::new()).await.unwrap();
        let parent_id = parent["id"].as_str().unwrap();

        // A plain string is rejected by the schema, a record link is accepted.
        assert!(db.create("child", Doc::new().set("parent", parent_id)).await.is_err());
        let child = db.create("child", Doc::new().record("parent", parent_id).set("label", Json::Null)).await.unwrap();
        assert_eq!(child["parent"], parent_id);

        let found = db
            .query("SELECT * FROM child WHERE parent = $p", Params::new().record("p", parent_id))
            .await
            .unwrap();
        assert_eq!(found.len(), 1);
        let found = db
            .query("SELECT * FROM child WHERE parent IN $ps", Params::new().records("ps", [parent_id]))
            .await
            .unwrap();
        assert_eq!(found.len(), 1);
    }

    #[tokio::test]
    async fn statement_errors_surface() {
        let db = Db::memory().await.unwrap();
        assert!(db.query("SELEKT nonsense", Params::new()).await.is_err());
        let (results, errors) = db.execute("RETURN 1; REMOVE TABLE nope; RETURN 3;", Params::new()).await.unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0], 1);
        assert_eq!(results[2], 3);
        assert_eq!(errors.len(), 1);
        assert_eq!(errors[0].0, 1);
        assert!(db.query_all("RETURN 1; REMOVE TABLE nope;", Params::new()).await.is_err());
        // Reserved variable names are caught with a clear message.
        let e = db.query("RETURN $session", Params::new().set("session", 1)).await.unwrap_err();
        assert!(e.to_string().contains("reserved"), "{e}");
        // A bad record id is reported when the query runs, not as a panic.
        assert!(db.query("SELECT * FROM $id", Params::new().record("id", "nocolon")).await.is_err());
    }
}
