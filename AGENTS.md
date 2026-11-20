# Nova Notes (Rust) — Agent Rules

A from-scratch Rust rebuild of the Nova-Notes fork of Open Notebook. One binary serves the JSON API, the web UI and the background worker.

The Python original, [Open Notebook](https://github.com/lfnovo/open-notebook), is the behavioural reference: read it, never copy files from it without updating `NOTICE.md`.

## Layout

| Path | What lives there |
|---|---|
| `crates/nn-llm` | Provider-neutral AI traits (`LanguageModel`, `EmbeddingModel`, `TextToSpeechModel`, `SpeechToTextModel`, `AiError`). The contract between the other crates. |
| `crates/nn-ai` | Provider clients, provider registry, model discovery, connection tests. No database. |
| `crates/nn-content` | URL / file / text extraction, chunking, text and token utilities, URL validation, web search. No database. |
| `crates/nn-podcast` | Podcast generation, audio combining, RSS feed, artwork and script helpers. No database. |
| `crates/nova-notes` | The server. See below. |

Inside `crates/nova-notes/src`:

| Module | Role |
|---|---|
| `config`, `error`, `state` | Environment config, `AppError` (maps to the Python API's status codes), `AppState`. |
| `db` | SurrealDB connection, `Params` / `Doc` builders, repository helpers, migrations. |
| `jobs` | In-process background worker over the `command` table. |
| `prompts` | Embedded Jinja templates (`prompts/`), rendered with minijinja. |
| `ai` | Model manager: configured models and credentials → `nn-ai` clients. |
| `domain` | Records and their queries (`Notebook`, `Source`, `Note`, `ChatSession`, ...). |
| `services` | Feature logic as plain async functions over `&AppState`. |
| `api` | JSON API routers. Thin: extract, call a service, return JSON. |
| `web` | Server-rendered UI. Calls the same services. |
| `testing` | (tests only) `TestApp` and mock models. |

## Commands

Cargo is installed by rustup; in a non-login shell run `source "$HOME/.cargo/env"` first.

- Check: `cargo check -p <crate>` · Test: `cargo test -p <crate>` · Lint: `cargo clippy -p <crate> --all-targets`
- Run: `cargo run -p nova-notes` (listens on `127.0.0.1:5055`, embedded database under `./data`)
- Build output goes to `target.nosync/` (the workspace is in iCloud Drive; `.nosync` keeps it local). Never `cargo clean`; disk is tight.

## Hard rules

- **API compatibility.** Paths, methods, query and body field names, response shapes, status codes and `{"detail": "..."}` error bodies match the Python routers in `open-notebook/api/routers/`. When unsure, the Python router and `open-notebook/api/models.py` decide. The Python tests in `open-notebook/tests/` are behavioural specs: port the cases that apply.
- **Database compatibility.** Same tables, fields and record ids as the Python build, so either build can open the same SurrealDB. Never edit files `1..25` or `100..106` under `crates/nova-notes/migrations/`. Schema this build needs beyond those goes in `migrations/rust/N.surrealql` (additive and idempotent), with `RUST_COUNT` in `db/migrate.rs` bumped.
- **Services own the logic.** A route handler does extraction and response shaping only. Anything the web UI would also need is a `pub async fn` in `services/` with typed, `Serialize`/`Deserialize` inputs and outputs.
- **No stubs.** No `todo!()`, `unimplemented!()`, or functions that pretend to work. If something cannot be done, return a clear `AppError` and say so in your report.
- **Never commit secrets**, and never put an API key in a log line or error message.
- Do not `git commit`; the user commits.

## Database conventions (`crate::db`)

- `state.db.query(sql, Params)` returns the first statement's rows as `serde_json::Value`, record ids as `"table:id"` strings, datetimes as RFC 3339. `query_as::<T>` deserializes rows; `query_last` returns the last statement's rows (for `LET ...; SELECT`); `query_all` returns every statement's result.
- **Record links must be bound as records.** A `record<source>` field or an `id IN $ids` comparison needs `Params::record("name", id)` / `Params::records("name", ids)` and `Doc::record("field", id)`. A plain string will not match and schemafull tables reject it.
- JSON `null` is written as SurrealDB `NONE`. In a `Doc`, `.none("field")` clears a field on update; `.set_some("field", option)` skips `None`.
- `db.create(table, Doc)` and `db.update(id, Doc)` set `created` / `updated`. `update` on a missing record is `AppError::NotFound`.
- Never splice user input into SurrealQL. Bind it. Table and relation names that must be spliced go through `ensure_safe_identifier`; `ORDER BY` from clients goes through `validate_order_by`.
- These variable names are reserved by SurrealDB and rejected by `Params`: `session`, `auth`, `token`, `scope`, `access`, `before`, `after`, `value`, `input`, `this`, `parent`, `event`.
- Deserializing rows: use `#[serde(default, deserialize_with = "crate::domain::null_default")]` for non-optional fields that may be NONE, and `crate::domain::link_id` for record links (handles id strings and `FETCH`ed objects).

## Requests and models

- Route handlers take bodies and query strings through `crate::api::ApiJson<T>` / `crate::api::ApiQuery<T>` (FastAPI-style 422 on bad input), not `axum::Json` / `Query`. `crate::api::lenient_bool` parses FastAPI-style boolean query values.
- Models come from `state.models` (`crate::ai::ModelManager`): `language(&state, DefaultModelType::Chat)`, `language_or_default(&state, model_id, kind)`, `language_by_id`, `embedding`, `embedding_opt`, `speech_to_text_opt`, `text_to_speech_by_id`. An empty default slot is `AppError::Configuration` with a "Manage → Models" message. `crate::ai::generate_embedding(&state, text)` / `generate_embeddings(&state, &model, &texts)` embed with chunking, batching and retries; `crate::ai::chunk_config()` gives the chunking settings.
- In tests, `app.state.models.set_override(AnyModel::Language(MockLanguageModel::new([...])))` (and `Embedding`, `TextToSpeech`, `SpeechToText`) makes every request for that model type return the mock.
- Library crates, all finished and tested: `nn_ai` (providers, `registry`, `discovery`, `tester`), `nn_content` (`extract_content`, `chunk_text_with`, `validate_url`, `clean_thinking_content`, `token_count`, `web_search`), `nn_podcast` (`create_podcast`, `feed`, `artwork`, `publishing`, `audio_paths`, `script`). Read their `src/lib.rs` for the public API before re-implementing anything.

## Errors (`crate::error::AppError`)

Pick the variant by the status the Python API returned: `NotFound` 404, `InvalidInput` 400, `Configuration` 422, `Authentication` 401, `RateLimit` 429, `Network` / `ExternalService` / `ContextLength` / `IncompleteGeneration` 502, `UnsupportedType` 415, `Database` / `FileOperation` / `Internal` 500. For an explicit status (FastAPI `HTTPException(status, detail)`) use `AppError::http(status, detail)`. Reuse the Python detail strings.

## Background jobs (`crate::jobs`)

- Each feature registers its handlers in its service module's `register_jobs(jobs)`; `services::register_jobs` already calls every module. Handler names match the Python command names (`process_source`, `embed_note`, `generate_podcast`, ...) and take / return the same JSON fields.
- `state.jobs.submit(name, args)` returns `command:<id>` at once. `state.jobs.run_inline(&state, name, args)` runs it in the calling task.
- A handler returning an `AppError` for which `is_retryable()` is false fails for good; others retry per the handler's `RetryPolicy`. Mirror the Python command's `retry=` settings.

## Tests

- `crate::testing::TestApp::new().await` gives a migrated in-memory app with jobs running: `app.get("/api/...")`, `app.post(path, json!({...}))`, `app.state`, `app.wait_for_job(id)`.
- Mock models live in `crate::testing` (`MockLanguageModel`, `MockEmbeddingModel`, `MockTextToSpeechModel`, `MockSpeechToTextModel`). Tests never touch the network.
- Test behaviour through the HTTP surface where there is one; assert on status, body and resulting database state.

## Working alongside other agents

Several agents may be editing different files of the same crate at once. Edit only the files you were given. If `cargo check` reports errors in a file you do not own, that agent is mid-edit: wait a little and check again rather than fixing it. "Blocking waiting for file lock on build directory" is normal.

## Git workflow and licence

- This is a standalone repo, not a fork of Open Notebook. Do not add an `upstream` remote or merge from `lfnovo/open-notebook`: the Rust code was ported from the Nova-Notes fork and no longer tracks upstream.
- Keep `LICENSE` (MIT, both copyright lines) and `NOTICE.md` intact. Any new file copied unchanged from Open Notebook must be added to the list in `NOTICE.md`.
- Work on a branch and open a pull request; use conventional commits.
- CI runs on every PR and push to `main` (`.github/workflows/ci.yml`): `cargo test --workspace` and `cargo clippy -D warnings`. Run both locally before opening a PR.
- Never commit `.env` files, API keys, tokens or private feed URLs.
