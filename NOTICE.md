# Notice

Nova Notes is a Rust rebuild of [Open Notebook](https://github.com/lfnovo/open-notebook)
by Luis Novo, by way of the Nova-Notes fork of it.
Open Notebook is released under the MIT License, Copyright (c) 2024 Luis Novo,
and that notice is kept in [LICENSE](LICENSE).

What this rebuild takes from Open Notebook:
- **The API contract:** paths, request and response shapes, status codes and error bodies.
- **The database schema:** the SurrealDB migrations under `crates/nova-notes/migrations/`
  (files `1`–`25` from Open Notebook, `100`–`106` from the Nova-Notes fork).
- **The prompt templates** under `crates/nova-notes/prompts/` (Jinja, rendered here with minijinja).
- **Behaviour**, as specified by Open Notebook's code and tests.

The Rust code is new.

Bundled third-party files:
- `crates/nova-notes/static/vendor/htmx.min.js`: [htmx](https://htmx.org),
  Zero-Clause BSD licence.

Rust dependencies keep their own licences. See `Cargo.lock`.

## Note editor

`crates/nova-notes/static/editor/` bundles BlockSuite 0.19.5 (MPL-2.0, © toeverything),
the editor engine of AFFiNE, with its dependencies; their licence texts are in
`static/editor/*.LEGAL.txt`. The Inter and Kalam fonts under `static/editor/fonts/` are
under the SIL Open Font License (texts alongside).
The editor makes no third-party requests. To get there, the bundled code is changed in two ways:
- **Removed features:** the YouTube, Loom, Figma, GitHub and raw-HTML embed blocks are left out
  of the editor (`crates/nova-notes/editor/src/editor.js`), so links stay plain links. A note
  saved with one of the URL embeds opens with it shown as a bookmark card.
- **Rewritten URLs:** the build (`crates/nova-notes/editor/build.mjs`) rewrites, in the
  BlockSuite and Lottie player files it bundles, every hard-coded third-party URL to a local
  path the server does not serve. That covers AFFiNE's hosted link-preview and image-proxy
  services, the Twitter preview API, the embed iframes, oEmbed and GitHub API calls, and the
  Lottie player's WebAssembly, which it would otherwise load from unpkg or jsDelivr. Link
  previews and Lottie animations therefore do not load.
- **Regression test:** `services::note_docs` has a test that fails if any of these hosts
  appears in `static/editor/`.
