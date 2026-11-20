# Nova Notes

A Rust rebuild of the Nova-Notes fork of [Open Notebook](https://github.com/lfnovo/open-notebook).
One binary serves the JSON API (compatible with Open Notebook 1.15.1), the server-rendered
web UI and the background worker. It opens the same SurrealDB database as the Python build.

## Run

```bash
cargo run -p nova-notes          # http://127.0.0.1:5055, embedded database under ./data
```

Settings come from the environment; see `.env.example`. A container image builds from the
`Dockerfile`.

## Layout

| Path | What lives there |
|---|---|
| `crates/nn-llm` | Provider-neutral AI traits |
| `crates/nn-ai` | Provider clients, model discovery, connection tests |
| `crates/nn-content` | Extraction, chunking, URL checks, web search |
| `crates/nn-podcast` | Podcast generation, audio, RSS feed |
| `crates/nova-notes` | The server: API, web UI, jobs, database |
| `crates/nova-notes/editor` | Source of the note editor bundle (see below) |
| `scripts/` | One-off migration helpers (LangGraph chat-history export) |

## Note editor

Notes open in a rich editor with a Page / Edgeless (whiteboard) toggle, built on
BlockSuite 0.19.5, the editor engine of [AFFiNE](https://github.com/toeverything/AFFiNE).

New notes are saved as you write. A blank draft stays accessible at its editor URL
and appears in the notebook after its first edit is saved. If you edit the same
note in two tabs, the last successful save wins.

The built bundle lives in `crates/nova-notes/static/editor/` and is compiled into the
binary, so a normal `cargo build` (and the Docker build) needs no Node. After changing
`crates/nova-notes/editor/src/` rebuild it with Node 20+:

```bash
cd crates/nova-notes/editor && npm ci && npm run build
```

Keep `node_modules` out of iCloud: create `node_modules.nosync` and symlink
`node_modules` to it before installing.

Working rules for agents and contributors are in [AGENTS.md](AGENTS.md).

## Licence

MIT. See [LICENSE](LICENSE) and [NOTICE.md](NOTICE.md).
