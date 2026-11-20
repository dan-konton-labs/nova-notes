# nova-vault-sync: Nova Notes into an encrypted Obsidian vault

Copies every Nova Notes note into the `NovaNotes/` folder of an Obsidian vault that uses **Self-hosted
LiveSync end-to-end encryption**, by going through the official LiveSync CLI. The CLI holds the vault's
encryption settings and does the encrypting, exactly like Obsidian. Nova Notes' own CouchDB push
(`VAULT_COUCHDB_*`) writes plain documents that an encrypted vault's clients cannot read, so **leave
`VAULT_COUCHDB_*` unset** when the vault is encrypted and use this tool instead.

```
Nova Notes  --GET /api/vault/files?content=true-->  nova_vault_sync.py  --livesync-cli put-->  CouchDB (encrypted)  <-->  Obsidian
```

One file, standard library only (Python 3.9+), needs Docker. Tests: `python3 -m unittest -v` in this directory.

## What it does and never does

- Writes only `NovaNotes/**.md` (or `--folder`, which must match Nova's `VAULT_FOLDER`). Nothing else in the vault is read or written.
- One way. **Never deletes**: a note removed from Nova Notes is reported as `stale` and stays in the vault.
- A vault file that differs from Nova's copy **and was modified after the Nova note last changed** is an Obsidian-side edit: skipped and reported under `conflicts`. If Nova's note is newer, the file is overwritten, and the text it replaced is first kept under `<writer>/overwritten/<UTC timestamp>/`.
- Two Nova files that would share one LiveSync document (names differing only by letter case) stop the run. Update Nova Notes to a build that includes the case-aware naming (PR #14).
- Safety rules: dedicated writer directory, `settings.json` must be mode 0600 with exactly one remote and the database you pass as `--expect-db`, encryption and a passphrase required, automatic newest-file conflict resolution refused, re-check before writing, a journal before the first write, read-back verification after, and a leftover journal blocks later runs until someone looks at it.
- Note text and CLI diagnostics never appear in its output. The journal and state hold hashes and revisions only.

## One-time setup (done by a person; the secrets never go in chat, a script or shell history)

Use a **separate writer directory and device identity** from any other LiveSync client or bridge you run.

```bash
sudo install -d -m 700 /opt/nova-vault-sync/writer
IMG=novahq/livesync-cli:1.0.40
CLI="docker run --rm -i --network host -e LIVESYNC_DB_PATH=/state/db -v /opt/nova-vault-sync/writer:/state $IMG --settings /state/settings.json"
docker run --rm -v /opt/nova-vault-sync/writer:/state $IMG init-settings /state/settings.json
read -r -p 'Setup URI (from Obsidian: LiveSync settings, copy setup URI): ' URI
read -rs -p 'Setup URI passphrase: ' P; echo
printf '%s\n' "$P" | $CLI setup "$URI"; unset P URI
sudo chmod 600 /opt/nova-vault-sync/writer/settings.json
$CLI remote-status          # check it points at the right database
$CLI sync                   # first sync: downloads the whole vault into the writer's local database
```

That first `sync` is expected to take a while and fills `writer/db` with a **decrypted** copy of the whole vault
(Personal and Work included), as any LiveSync client does. Keep the writer directory
mode 700, owned by root, and out of backups you do not trust.

Only if `sync` says the remote database is locked and this device is not accepted, run `$CLI unlock-remote`
once (it accepts this device, as the other clients were accepted). On a vault that is not locked, skip it.

## Running it

```bash
# what would change? writes nothing
python3 nova_vault_sync.py --writer /opt/nova-vault-sync/writer --expect-db my_vault --dry-run
# for real
python3 nova_vault_sync.py --writer /opt/nova-vault-sync/writer --expect-db my_vault
```

Options: `--nova-url` (default `http://127.0.0.1:5055`), `--folder` (default `NovaNotes`), `--network` (docker
network for the CLI container, default `host`), `--image`, `--case-sensitive` (only if the vault has "Handle
files as case-sensitive" on), `--allow-unencrypted` (throwaway tests only).

Output is one JSON line: `pushed`, `unchanged`, `conflicts`, `stale`, `overwritten`. Exit 0 = ran, 3 = a
safety rule stopped it (the reason is in the JSON), 1 = unexpected error.

Run it on a timer once you trust it, for example every 15 minutes (`/etc/systemd/system/nova-vault-sync.service` and `.timer`):

```ini
[Service]
Type=oneshot
ExecStart=/usr/bin/python3 /opt/nova-vault-sync/nova_vault_sync.py --writer /opt/nova-vault-sync/writer --expect-db my_vault
```

## Recovering

- **`pending-write.json` exists**: an earlier run stopped between writing and verifying. Look at the vault in Obsidian (the journal lists the paths and hashes), then delete the file. The next run is idempotent.
- **`conflicts` listed**: either keep the Obsidian version (edit the note in Nova Notes so the two agree, or ignore it) or take Nova's (edit the note in Nova Notes afterwards: a newer Nova note overwrites the vault file and keeps the old text under `overwritten/`).
- **A vault conflict (`*` revision) stops the run**: resolve it in Obsidian first.

## How it was verified

Against a throwaway CouchDB with end-to-end encryption and path obfuscation on: the server held no plaintext (no note text, titles or paths in any document), an independent CLI client with the passphrase read every file byte-identically, a second run was a no-op, an Obsidian-side edit was reported as a conflict and kept, a later Nova edit overwrote it and kept the old text, a deleted note was reported `stale` and not deleted, and the refusal rules fired (loose settings permissions, wrong database, no encryption).
