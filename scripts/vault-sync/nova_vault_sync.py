#!/usr/bin/env python3
"""Copy Nova Notes into an Obsidian vault that uses Self-hosted LiveSync end-to-end encryption.

Nova Notes can render every note as a vault file (`GET /api/vault/files?content=true`). Its own CouchDB
push writes plain documents, which an encrypted vault's clients cannot read, so for an encrypted vault the
files are written through the official LiveSync CLI instead: the CLI holds the vault's encryption settings
and does the encrypting, exactly as Obsidian would.

Safety model:
  * One dedicated writer directory holds the CLI's settings.json (mode 0600), its local database, a
    journal and this tool's state. The settings must name exactly one remote, the database given with
    --expect-db, and must have encryption on (unless --allow-unencrypted, for throwaway tests only).
  * Only `<folder>/**.md` is ever read or written (default folder `NovaNotes`). Everything else in the
    vault is invisible to this tool.
  * One way, and nothing is ever deleted: a file that no longer exists in Nova Notes is only reported as
    `stale`.
  * A remote file that differs from the Nova copy and was modified after the Nova note last changed is an
    Obsidian-side edit: it is skipped and reported as a `conflict`. When Nova's note is the newer one the
    remote file is overwritten, and the text it replaces is first kept under `overwritten/`.
  * Re-check before writing (a changed revision aborts the run), a durable journal before the first write,
    read-back verification after, and a leftover journal blocks later runs until a person looks at it.
  * Note text and CLI diagnostics never appear in logs or error messages.
  * Nova Notes is called with `Authorization: Bearer <OPEN_NOTEBOOK_PASSWORD>` when one is given (the
    NOVA_NOTES_TOKEN environment variable or --token-file, a 0600 file); plain http:// is refused for
    anything but a loopback address unless --allow-insecure-http.

Standard library only. Needs Docker (the CLI runs as `novahq/livesync-cli`).
"""
import argparse
import fcntl
import hashlib
import ipaddress
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

# A tag can be re-pushed. Pin the image you have checked by digest with --image (or edit this line), e.g.
#   docker image inspect --format '{{index .RepoDigests 0}}' novahq/livesync-cli:1.0.40
# gives `novahq/livesync-cli@sha256:...`; for an image only built locally use its ID
# (`docker image inspect --format '{{.Id}}' ...`). See README.md, "Pinning the CLI image".
DEFAULT_IMAGE = "novahq/livesync-cli:1.0.40"
TOKEN_ENV = "NOVA_NOTES_TOKEN"
DEFAULT_FOLDER = "NovaNotes"
DEFAULT_NOVA = "http://127.0.0.1:5055"


class Refused(Exception):
    """A safety rule stopped the run before anything was written (or before more was written)."""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def in_scope(folder, path):
    """A markdown file strictly inside `folder`, with no empty, `.` or `..` parts."""
    if not path.startswith(folder + "/") or not path.endswith(".md"):
        return False
    if "\\" in path or any(ord(c) < 32 for c in path):
        return False
    return all(part not in ("", ".", "..") for part in path.split("/"))


# ---------------------------------------------------------------------------
# The writer directory and its settings
# ---------------------------------------------------------------------------


def private_mkdir(path, top):
    """Create `path` and every missing level below `top` with mode 0700 (mkdir's mode only covers the leaf)."""
    path, top = Path(path), Path(top)
    missing = []
    while path != top and not path.exists():
        missing.append(path)
        path = path.parent
    for level in reversed(missing):
        level.mkdir(mode=0o700)
        os.chmod(level, 0o700)  # mkdir's mode is filtered by the umask


def is_pinned(image):
    """An image named by digest (`name@sha256:...`) or by image ID (`sha256:...`)."""
    return "@sha256:" in image or image.startswith("sha256:")


def load_token(token_file=None, environ=None):
    """The Nova Notes password to send as a bearer token: --token-file (must be 0600) or NOVA_NOTES_TOKEN."""
    environ = os.environ if environ is None else environ
    if token_file:
        path = Path(token_file)
        if not path.is_file():
            raise Refused("--token-file does not exist")
        if path.stat().st_mode & 0o077:
            raise Refused("--token-file must be mode 0600")
        token = path.read_text().strip()
    else:
        token = (environ.get(TOKEN_ENV) or "").strip()
    if any(ord(c) < 33 or ord(c) > 126 for c in token):
        raise Refused("the Nova Notes token has spaces or characters a header cannot carry")
    return token or None


def check_nova_url(nova_url, allow_insecure_http=False):
    """Plain http:// is fine to a loopback address; elsewhere notes (and the token) would cross the network in clear."""
    parts = urlsplit(nova_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise Refused("--nova-url must be an http:// or https:// URL")
    if parts.scheme == "https" or allow_insecure_http:
        return
    host = parts.hostname
    try:
        loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        loopback = host == "localhost"
    if not loopback:
        raise Refused("--nova-url is plain http:// to another machine; use https:// or pass --allow-insecure-http")


def check_settings(writer, expect_db, allow_unencrypted=False):
    """Validate the CLI settings; returns the active remote id (or None for a single legacy remote)."""
    path = Path(writer) / "settings.json"
    if not path.is_file():
        raise Refused("settings.json is missing from the writer directory")
    if path.stat().st_mode & 0o077:
        raise Refused("settings.json must be mode 0600")
    settings = json.loads(path.read_text())
    remotes = settings.get("remoteConfigurations") or {}
    remote_id = None
    if remotes:
        if len(remotes) != 1 or settings.get("activeConfigurationId") not in remotes:
            raise Refused("the settings must name exactly one remote")
        remote_id = settings["activeConfigurationId"]
    elif settings.get("couchDB_DBNAME") != expect_db:
        raise Refused("the LiveSync settings target a different database than --expect-db")
    if not allow_unencrypted and not (settings.get("encrypt") and (settings.get("passphrase") or settings.get("encryptedPassphrase"))):
        raise Refused("the LiveSync settings do not have end-to-end encryption and a passphrase")
    if settings.get("resolveConflictsByNewerFile"):
        raise Refused("turn off automatic newest-file conflict resolution for this writer")
    if settings.get("additionalRemote") or settings.get("additionalRemotes"):
        raise Refused("the settings must name exactly one remote")
    return remote_id


class Cli:
    """The LiveSync CLI in a throwaway container, one call per command."""

    def __init__(self, writer, image=DEFAULT_IMAGE, network="host", runner=None):
        self.command = [
            "docker", "run", "--rm", "-i", "--network", network,
            "-e", "LIVESYNC_DB_PATH=/state/db", "-v", f"{Path(writer).resolve()}:/state",
            image, "--settings", "/state/settings.json",
        ]
        if runner:
            self.run = runner

    def run(self, *args, content=None):
        result = subprocess.run(self.command + list(args), input=content, capture_output=True, timeout=480)
        if result.returncode:
            # The CLI's diagnostics can carry connection settings: keep them out of our output.
            raise RuntimeError(f"LiveSync {args[0]} failed (exit {result.returncode}); inspect privately")
        return result.stdout


# ---------------------------------------------------------------------------
# The vault, as the CLI sees it
# ---------------------------------------------------------------------------


class Vault:
    def __init__(self, cli, folder, expect_db, remote_id):
        self.cli, self.folder, self.expect_db, self.remote_id = cli, folder, expect_db, remote_id
        self.verified = remote_id is None

    def verify_remote(self):
        if self.verified:
            return
        lines = self.cli.run("remote-export", self.remote_id).decode().strip().splitlines()
        target = urlsplit(lines[-1] if lines else "")
        database = (parse_qs(target.query).get("db") or [target.path.strip("/")])[0]
        if database != self.expect_db:
            raise Refused("the LiveSync remote is a different database than --expect-db")
        self.verified = True

    def sync(self):
        self.verify_remote()
        self.cli.run("sync")

    def index(self):
        """path -> (mtime in ms, revision) for every markdown file under the folder."""
        out = {}
        for line in self.cli.run("ls", self.folder + "/").decode().splitlines():
            if not line.strip():
                continue
            fields = line.split("\t")
            if len(fields) != 4 or not fields[0].startswith(self.folder + "/"):
                raise Refused(f"unexpected LiveSync ls output ({len(fields)} fields)")
            path, _size, mtime, rev = fields
            if not path.endswith(".md"):
                continue
            if not rev or rev.endswith("*"):
                raise Refused("a file in the vault folder has a LiveSync conflict; resolve it in Obsidian first")
            try:
                out[path] = (int(mtime), rev)
            except ValueError:
                raise Refused("unexpected LiveSync ls output (mtime)")
        return out

    def cat(self, path):
        return self.cli.run("cat", path)

    def put(self, path, data):
        self.cli.run("put", path, content=data)


# ---------------------------------------------------------------------------
# Planning (pure) and applying
# ---------------------------------------------------------------------------


def key_of(path, case_sensitive):
    return path if case_sensitive else path.lower()


def plan(tree, index, remote_sha, case_sensitive=False):
    """Decide what to do with each Nova file.

    tree: [{path, content (str), mtime (ms)}]; index: path -> (mtime, rev); remote_sha(path) -> sha256 of the
    remote text. Returns a dict of lists: put (new files), overwrite, unchanged, conflicts, stale."""
    by_key = {key_of(p, case_sensitive): p for p in index}
    out = {"put": [], "overwrite": [], "unchanged": [], "conflicts": [], "stale": []}
    seen = set()
    for file in tree:
        path, desired = file["path"], file["content"].encode("utf-8")
        key = key_of(path, case_sensitive)
        if key in seen:
            raise Refused("two Nova files would share one vault document")
        seen.add(key)
        remote = by_key.get(key)
        if remote is None:
            out["put"].append((path, desired, None))
        elif remote != path:
            # A file differing only by letter case: it would be the same LiveSync document.
            out["conflicts"].append(path)
        elif sha(desired) == remote_sha(path):
            out["unchanged"].append(path)
        elif index[path][0] > file["mtime"]:
            out["conflicts"].append(path)
        else:
            out["overwrite"].append((path, desired, index[path][1]))
    out["stale"] = sorted(p for k, p in by_key.items() if k not in seen)
    return out


def fetch_tree(nova_url, folder, token=None):
    request = urllib.request.Request(nova_url.rstrip("/") + "/api/vault/files?content=true", headers={"Accept": "application/json"})
    if token:
        # Unredirected: urllib would otherwise copy it onto a redirect to another host.
        request.add_unredirected_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=60) as response:
        body = json.load(response)
    if body.get("folder") != folder:
        raise Refused("Nova Notes writes to a different vault folder than --folder (check VAULT_FOLDER)")
    tree = body.get("files") or []
    for file in tree:
        if not in_scope(folder, file["path"]):
            raise Refused("Nova Notes returned a path outside the vault folder")
        if file.get("content") is None or not isinstance(file.get("mtime"), int):
            raise Refused("Nova Notes returned an incomplete file list")
    return tree


class State:
    """path -> {rev, sha}: what we last saw or wrote, so unchanged files need no CLI read."""

    def __init__(self, writer):
        self.path = Path(writer) / "state.json"
        self.files = {}
        if self.path.exists():
            self.files = json.loads(self.path.read_text()).get("files", {})

    def save(self):
        tmp = self.path.with_suffix(".tmp")
        fd = os.open(tmp, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as handle:
            json.dump({"version": 1, "files": self.files}, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)


def run_once(writer, tree, vault, state, dry_run=False, case_sensitive=False, sleep=time.sleep, now=time.time):
    writer = Path(writer)
    pending = writer / "pending-write.json"
    if pending.exists():
        raise Refused("an interrupted write needs a person to look at pending-write.json first")
    vault.sync()
    index = vault.index()

    def remote_sha(path):
        _mtime, rev = index[path]
        known = state.files.get(path)
        if known and known["rev"] == rev:
            return known["sha"]
        digest = sha(vault.cat(path))
        state.files[path] = {"rev": rev, "sha": digest}
        return digest

    result = plan(tree, index, remote_sha, case_sensitive)
    report = {
        "pushed": [], "unchanged": result["unchanged"], "conflicts": result["conflicts"],
        "stale": result["stale"], "overwritten": [], "dry_run": dry_run,
    }
    updates = result["put"] + result["overwrite"]
    if dry_run or not updates:
        report["would_push"] = [p for p, _d, _r in updates] if dry_run else []
        if not dry_run:
            state.save()
        return report

    # Re-check: nothing we are about to touch may have changed since the plan was made.
    vault.sync()
    again = vault.index()
    for path, _data, rev in updates:
        if (again[path][1] if path in again else None) != rev:
            raise Refused("a vault file changed while planning; try again on the next run")

    # Keep what an overwrite replaces.
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(now()))
    for path, _data, rev in result["overwrite"]:
        old = vault.cat(path)
        target = writer / "overwritten" / stamp / path
        private_mkdir(target.parent, writer)
        fd = os.open(target, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "wb") as handle:
            handle.write(old)
        report["overwritten"].append(path)

    # Durable journal (hashes only) before the first write.
    record = {"expected": {p: r for p, _d, r in updates}, "desired": {p: sha(d) for p, d, _r in updates}}
    fd = os.open(pending, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(record, handle)
        handle.flush()
        os.fsync(handle.fileno())
    for path, data, _rev in updates:
        vault.put(path, data)

    for attempt in range(3):
        vault.sync()
        current = vault.index()
        if all(path in current and vault.cat(path) == data for path, data, _r in updates):
            for path, data, _r in updates:
                state.files[path] = {"rev": current[path][1], "sha": sha(data)}
                report["pushed"].append(path)
            pending.unlink()
            state.save()
            return report
        if attempt < 2:
            sleep(2)
    raise RuntimeError("read-back verification failed; the journal was kept for recovery")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Copy Nova Notes into an encrypted LiveSync vault through the official CLI.")
    parser.add_argument("--writer", required=True, help="directory with the CLI's settings.json (0600); state and journal live here")
    parser.add_argument("--expect-db", required=True, help="the one CouchDB database this writer may talk to")
    parser.add_argument("--nova-url", default=DEFAULT_NOVA)
    parser.add_argument("--token-file", help=f"file (mode 0600) holding Nova Notes' OPEN_NOTEBOOK_PASSWORD; or set {TOKEN_ENV}")
    parser.add_argument("--allow-insecure-http", action="store_true", help="allow a plain http:// --nova-url that is not loopback")
    parser.add_argument("--folder", default=DEFAULT_FOLDER, help="vault folder Nova Notes writes under (VAULT_FOLDER)")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="the LiveSync CLI image; pin it by digest (name@sha256:...)")
    parser.add_argument("--network", default="host",
                        help="docker network for the CLI container. host (the default) lets it reach a CouchDB that "
                             "listens only on the host's loopback; use bridge when CouchDB has a routable address")
    parser.add_argument("--dry-run", action="store_true", help="report what would change, write nothing")
    parser.add_argument("--allow-unencrypted", action="store_true", help="throwaway tests only")
    parser.add_argument("--case-sensitive", action="store_true", help="the vault has 'Handle files as case-sensitive' on")
    args = parser.parse_args(argv)

    writer = Path(args.writer)
    if not is_pinned(args.image):
        print(f"warning: {args.image} is a mutable tag; pin the image by digest with --image (see README.md)", file=sys.stderr)
    try:
        check_nova_url(args.nova_url, args.allow_insecure_http)
        token = load_token(args.token_file)
        remote_id = check_settings(writer, args.expect_db, args.allow_unencrypted)
        private_mkdir(writer / "db", writer)
        lock = os.fdopen(os.open(writer / "lock", os.O_CREAT | os.O_WRONLY, 0o600), "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise Refused("another run is in progress")
        vault = Vault(Cli(writer, args.image, args.network), args.folder, args.expect_db, remote_id)
        tree = fetch_tree(args.nova_url, args.folder, token)
        report = run_once(writer, tree, vault, State(writer), args.dry_run, args.case_sensitive)
    except Refused as error:
        print(json.dumps({"refused": str(error)}))
        return 3
    except Exception as error:  # noqa: BLE001 - message only, never the note text
        print(json.dumps({"error": f"{type(error).__name__}: {error}"}))
        return 1
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
