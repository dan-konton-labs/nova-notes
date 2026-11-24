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

The server refuses requests for host names it does not know, cross-site form posts and API
writes, and (with no password) listening on anything but loopback. To reach it by another name,
such as a Tailscale MagicDNS name, set `OPEN_NOTEBOOK_ALLOWED_HOSTS=notes.tailnet-name.ts.net`
(or `OPEN_NOTEBOOK_PUBLIC_URL`). The container listens on `0.0.0.0`, so run it with
`-e OPEN_NOTEBOOK_PASSWORD=...`, or with `-e OPEN_NOTEBOOK_ALLOW_UNAUTHENTICATED=true` when
something else already limits who can connect. Behind an HTTPS reverse proxy, set
`OPEN_NOTEBOOK_PUBLIC_URL` to the exact browser-facing HTTPS origin (including any non-default
port). Preserve the public `Host` where possible; if the proxy rewrites it, ensure the backend
receives an allowed host. A public URL does not tell Nova which IP belongs to each client: without
trusted forwarded headers, sign-in and Bearer requests share the proxy address for throttling.
Enable `OPEN_NOTEBOOK_TRUST_PROXY_HEADERS=true` only when clients cannot reach the backend directly
and the proxy controls the forwarded values. Details in `.env.example`.

The container runs as the unprivileged user `nova` (UID/GID 10001). It does not change ownership
of mounted data. Before upgrading, back up the existing data, prepare a copy so UID/GID 10001 can
read and write the directory and its contents, and smoke-test both the upgrade and restore on that
copy. Startup checks access to the top-level directory and stops with instructions if it cannot use
it. Run with `--user 10001:10001` to skip starting as root; `NOVA_RUN_AS_ROOT=true` is a temporary
recovery override, not a migration step. With a remote database (`SURREAL_URL`), set
`SURREAL_PASSWORD`: the fallback is `root`, and the server warns at startup when it is used.

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
