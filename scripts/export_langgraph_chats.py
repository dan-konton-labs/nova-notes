#!/usr/bin/env python3
"""Export chat history from the Python build's LangGraph checkpoints.

The Python build of Open Notebook / Nova-Notes keeps notebook and source chat
messages in a LangGraph SQLite checkpoint (data/sqlite-db/checkpoints.sqlite).
The Rust build keeps them in its database and imports them with:

    nova-notes import-chat-history /app/data/chat-export.jsonl [--dry-run]

Run this with the *Python* image (it has LangGraph, which reads its own
checkpoint format), on the same data volume, for example:

    docker compose run --rm --no-deps \\
        --entrypoint /app/.venv/bin/python \\
        -v /opt/nova-build/nova-notes-rust/scripts:/scripts:ro \\
        open_notebook /scripts/export_langgraph_chats.py

The checkpoint file is never modified: it is copied with SQLite's backup API
(a consistent snapshot even if the app is running) and the copy is read.

Output: one JSON object per chat session (thread):
    {"session": "chat_session:abc",
     "messages": [{"role": "human", "content": "..."}, {"role": "ai", "content": "..."}],
     "context_indicators": {...} or null}
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from typing import Any

DEFAULT_CHECKPOINTS = "/app/data/sqlite-db/checkpoints.sqlite"
DEFAULT_OUT = "/app/data/chat-export.jsonl"


def text_of(content: Any) -> str:
    """LangChain message content as plain text (strings, or lists of parts)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(p for p in parts if p)
    return "" if content is None else str(content)


def snapshot(path: str) -> str:
    """Copy the checkpoint database to a temporary file and return its path."""
    source = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    fd, copy_path = tempfile.mkstemp(suffix=".sqlite")
    os.close(fd)
    target = sqlite3.connect(copy_path)
    with target:
        source.backup(target)
    source.close()
    target.close()
    return copy_path


def export(checkpoints: str, out: str) -> dict:
    from langgraph.checkpoint.sqlite import SqliteSaver  # the image's own version

    copy_path = snapshot(checkpoints)
    try:
        conn = sqlite3.connect(copy_path, check_same_thread=False)
        saver = SqliteSaver(conn)
        threads = [
            row[0]
            for row in conn.execute(
                "SELECT DISTINCT thread_id FROM checkpoints WHERE checkpoint_ns = '' ORDER BY thread_id"
            )
        ]
        stats = {"threads": 0, "messages": 0, "empty_threads": 0}
        with open(out, "w", encoding="utf-8") as handle:
            for thread_id in threads:
                found = saver.get_tuple({"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}})
                values = (found.checkpoint.get("channel_values") or {}) if found else {}
                messages = []
                for message in values.get("messages") or []:
                    role = getattr(message, "type", None)
                    if role is None and isinstance(message, dict):
                        role = message.get("type") or message.get("role")
                    content = getattr(message, "content", None)
                    if content is None and isinstance(message, dict):
                        content = message.get("content")
                    messages.append({"role": str(role or "unknown"), "content": text_of(content)})
                if not messages:
                    stats["empty_threads"] += 1
                    continue
                record = {
                    "session": str(thread_id),
                    "messages": messages,
                    "context_indicators": values.get("context_indicators"),
                }
                handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                stats["threads"] += 1
                stats["messages"] += len(messages)
        conn.close()
        return stats
    finally:
        os.unlink(copy_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoints", default=DEFAULT_CHECKPOINTS, help=f"default: {DEFAULT_CHECKPOINTS}")
    parser.add_argument("--out", default=DEFAULT_OUT, help=f"default: {DEFAULT_OUT}")
    args = parser.parse_args()
    if not os.path.exists(args.checkpoints):
        print(f"No checkpoint file at {args.checkpoints}: there is no chat history to export.", file=sys.stderr)
        return 1
    stats = export(args.checkpoints, args.out)
    print(
        f"Exported {stats['messages']} messages from {stats['threads']} chat sessions to {args.out} "
        f"({stats['empty_threads']} sessions had no messages)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
