"""Tests for nova_vault_sync. Standard library only: `python3 -m unittest` in this directory."""
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path

import nova_vault_sync as nvs


class FakeVault:
    """What the LiveSync CLI would show: path -> (text, mtime ms, rev counter). Speaks the commands we use."""

    def __init__(self, remote_db="vault"):
        self.files = {}
        self.clock = 1_000
        self.calls = []
        self.remote_db = remote_db
        self.drop_puts = False
        self.conflict_paths = set()

    def tick(self):
        self.clock += 10
        return self.clock

    def edit(self, path, text):
        """An Obsidian-side edit: new text, newer mtime, next revision."""
        rev = self.files[path][2] + 1 if path in self.files else 1
        self.files[path] = (text, self.tick(), rev)

    def run(self, *args, content=None):
        self.calls.append(args[0])
        cmd = args[0]
        if cmd == "sync":
            return b""
        if cmd == "remote-export":
            return f"sls+http://u:p@couch:5984/?db={self.remote_db}\n".encode()
        if cmd == "ls":
            prefix = args[1]
            lines = []
            for path, (text, mtime, rev) in sorted(self.files.items()):
                if path.startswith(prefix):
                    star = "*" if path in self.conflict_paths else ""
                    lines.append(f"{path}\t{len(text.encode())}\t{mtime}\t{rev}-abc{star}")
            return ("\n".join(lines) + "\n").encode()
        if cmd == "cat":
            return self.files[args[1]][0].encode()
        if cmd == "put":
            if not self.drop_puts:
                self.edit(args[1], content.decode())
            return b""
        raise AssertionError(f"unexpected command {args}")


def file(path, content, mtime):
    return {"path": path, "content": content, "mtime": mtime}


class Case(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.writer = Path(self.tmp.name)
        self.fake = FakeVault()
        self.vault = nvs.Vault(nvs.Cli(self.writer, runner=self.fake.run), "NovaNotes", "vault", None)
        self.state = nvs.State(self.writer)

    def tearDown(self):
        self.tmp.cleanup()

    def run_sync(self, tree, **kw):
        return nvs.run_once(self.writer, tree, self.vault, self.state, sleep=lambda _s: None, **kw)


class Planning(Case):
    def test_new_files_are_pushed_and_then_unchanged(self):
        tree = [file("NovaNotes/A/x.md", "one", 100)]
        report = self.run_sync(tree)
        self.assertEqual(report["pushed"], ["NovaNotes/A/x.md"])
        self.assertEqual(self.fake.files["NovaNotes/A/x.md"][0], "one")
        again = self.run_sync(tree)
        self.assertEqual((again["pushed"], again["unchanged"], again["conflicts"]), ([], ["NovaNotes/A/x.md"], []))

    def test_unchanged_files_need_no_cli_read_on_later_runs(self):
        tree = [file("NovaNotes/A/x.md", "one", 100)]
        self.run_sync(tree)
        self.fake.calls.clear()
        self.run_sync(tree)
        self.assertNotIn("cat", self.fake.calls)
        self.assertNotIn("put", self.fake.calls)

    def test_obsidian_edit_after_the_note_is_a_conflict_and_is_kept(self):
        tree = [file("NovaNotes/A/x.md", "one", 100)]
        self.run_sync(tree)
        self.fake.edit("NovaNotes/A/x.md", "edited in obsidian")
        report = self.run_sync(tree)
        self.assertEqual((report["pushed"], report["conflicts"]), ([], ["NovaNotes/A/x.md"]))
        self.assertEqual(self.fake.files["NovaNotes/A/x.md"][0], "edited in obsidian")

    def test_newer_nova_note_overwrites_and_keeps_the_replaced_text(self):
        self.run_sync([file("NovaNotes/A/x.md", "one", 100)])
        self.fake.edit("NovaNotes/A/x.md", "edited in obsidian")
        newer = [file("NovaNotes/A/x.md", "two, edited later in nova", self.fake.clock + 50)]
        report = self.run_sync(newer)
        self.assertEqual((report["pushed"], report["overwritten"]), (["NovaNotes/A/x.md"], ["NovaNotes/A/x.md"]))
        self.assertEqual(self.fake.files["NovaNotes/A/x.md"][0], "two, edited later in nova")
        saved = list((self.writer / "overwritten").rglob("x.md"))
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0].read_text(), "edited in obsidian")
        self.assertEqual(stat.S_IMODE(saved[0].stat().st_mode), 0o600)

    def test_files_gone_from_nova_are_reported_stale_never_deleted(self):
        self.run_sync([file("NovaNotes/A/x.md", "one", 100), file("NovaNotes/A/y.md", "two", 100)])
        report = self.run_sync([file("NovaNotes/A/x.md", "one", 100)])
        self.assertEqual(report["stale"], ["NovaNotes/A/y.md"])
        self.assertIn("NovaNotes/A/y.md", self.fake.files)
        self.assertNotIn("rm", self.fake.calls)

    def test_a_case_variant_already_in_the_vault_is_a_conflict_not_a_write(self):
        self.fake.edit("NovaNotes/A/todo.md", "obsidian's")
        report = self.run_sync([file("NovaNotes/A/Todo.md", "nova's", 100)])
        self.assertEqual((report["pushed"], report["conflicts"]), ([], ["NovaNotes/A/Todo.md"]))
        self.assertEqual(self.fake.files["NovaNotes/A/todo.md"][0], "obsidian's")

    def test_two_nova_files_that_share_a_document_are_refused(self):
        tree = [file("NovaNotes/A/Todo.md", "1", 100), file("NovaNotes/A/todo.md", "2", 100)]
        with self.assertRaises(nvs.Refused):
            self.run_sync(tree)
        self.assertEqual(self.fake.files, {})

    def test_dry_run_writes_nothing(self):
        report = self.run_sync([file("NovaNotes/A/x.md", "one", 100)], dry_run=True)
        self.assertEqual((report["pushed"], report["would_push"]), ([], ["NovaNotes/A/x.md"]))
        self.assertEqual(self.fake.files, {})
        self.assertFalse((self.writer / "pending-write.json").exists())

    def test_only_markdown_under_the_folder_is_touched(self):
        self.fake.files["NovaNotes/img.png"] = ("binary", 5, 1)
        self.fake.files["Personal/secret.md"] = ("not ours", 5, 1)
        report = self.run_sync([file("NovaNotes/A/x.md", "one", 100)])
        self.assertEqual(report["stale"], [])
        self.assertEqual(self.fake.files["Personal/secret.md"][0], "not ours")


class Safety(Case):
    def test_a_leftover_journal_blocks_the_run(self):
        (self.writer / "pending-write.json").write_text("{}")
        with self.assertRaises(nvs.Refused):
            self.run_sync([file("NovaNotes/A/x.md", "one", 100)])
        self.assertEqual(self.fake.files, {})

    def test_a_file_changing_between_plan_and_write_aborts(self):
        self.run_sync([file("NovaNotes/A/x.md", "one", 100)])
        newer = [file("NovaNotes/A/x.md", "two", self.fake.clock + 50)]
        original = self.fake.run
        syncs = []

        def run(*args, content=None):
            if args[0] == "sync":
                syncs.append(1)
                if len(syncs) == 2:  # the re-check sync: someone edits in Obsidian meanwhile
                    self.fake.edit("NovaNotes/A/x.md", "concurrent edit")
            return original(*args, content=content)

        self.vault.cli.run = run
        with self.assertRaises(nvs.Refused):
            self.run_sync(newer)
        self.assertEqual(self.fake.files["NovaNotes/A/x.md"][0], "concurrent edit")

    def test_failed_verification_keeps_the_journal(self):
        self.fake.drop_puts = True
        with self.assertRaises(RuntimeError):
            self.run_sync([file("NovaNotes/A/x.md", "one", 100)])
        self.assertTrue((self.writer / "pending-write.json").exists())
        with self.assertRaises(nvs.Refused):
            self.run_sync([file("NovaNotes/A/x.md", "one", 100)])

    def test_a_vault_conflict_stops_the_run(self):
        self.fake.edit("NovaNotes/A/x.md", "t")
        self.fake.conflict_paths.add("NovaNotes/A/x.md")
        with self.assertRaises(nvs.Refused):
            self.run_sync([file("NovaNotes/A/x.md", "one", 100)])

    def test_the_remote_must_be_the_expected_database(self):
        vault = nvs.Vault(nvs.Cli(self.writer, runner=self.fake.run), "NovaNotes", "my_vault", "remote-1")
        with self.assertRaises(nvs.Refused):
            nvs.run_once(self.writer, [], vault, self.state)
        self.assertNotIn("sync", self.fake.calls)

    def test_journal_and_state_hold_no_note_text(self):
        self.fake.drop_puts = True
        try:
            self.run_sync([file("NovaNotes/A/x.md", "SECRET BODY", 100)])
        except RuntimeError:
            pass
        self.assertNotIn("SECRET BODY", (self.writer / "pending-write.json").read_text())
        self.assertEqual(stat.S_IMODE((self.writer / "pending-write.json").stat().st_mode), 0o600)


class Settings(unittest.TestCase):
    def write(self, settings, mode=0o600):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "settings.json"
        path.write_text(json.dumps(settings))
        os.chmod(path, mode)
        return tmp.name

    GOOD = {"couchDB_DBNAME": "vault", "encrypt": True, "passphrase": "x"}

    def test_a_good_encrypted_single_remote_passes(self):
        self.assertIsNone(nvs.check_settings(self.write(self.GOOD), "vault"))

    def test_refusals(self):
        cases = {
            "group-readable": (self.GOOD, 0o640),
            "wrong database": ({**self.GOOD, "couchDB_DBNAME": "other"}, 0o600),
            "no encryption": ({"couchDB_DBNAME": "vault"}, 0o600),
            "encryption without passphrase": ({"couchDB_DBNAME": "vault", "encrypt": True}, 0o600),
            "newest wins": ({**self.GOOD, "resolveConflictsByNewerFile": True}, 0o600),
            "extra remote": ({**self.GOOD, "additionalRemote": "x"}, 0o600),
            "two remotes": ({"remoteConfigurations": {"a": {}, "b": {}}, "activeConfigurationId": "a", "encrypt": True, "passphrase": "x"}, 0o600),
        }
        for name, (settings, mode) in cases.items():
            with self.subTest(name), self.assertRaises(nvs.Refused):
                nvs.check_settings(self.write(settings, mode), "vault")

    def test_unencrypted_is_only_allowed_by_the_flag(self):
        directory = self.write({"couchDB_DBNAME": "vault"})
        with self.assertRaises(nvs.Refused):
            nvs.check_settings(directory, "vault")
        nvs.check_settings(directory, "vault", allow_unencrypted=True)


class Scope(unittest.TestCase):
    def test_in_scope(self):
        self.assertTrue(nvs.in_scope("NovaNotes", "NovaNotes/Garden/Plan v2 Final.md"))
        for bad in ["Personal/x.md", "NovaNotes/x.png", "NovaNotes/../Personal/x.md", "NovaNotes//x.md", "NovaNotes/./x.md",
                    "NovaNotesX/x.md", "NovaNotes/a\\b.md", "NovaNotes/a\nb.md", "NovaNotes.md"]:
            self.assertFalse(nvs.in_scope("NovaNotes", bad), bad)

    def test_a_nova_list_outside_the_folder_is_refused(self):
        import io
        import unittest.mock as mock

        def fake(path):
            body = json.dumps({"folder": "NovaNotes", "files": [{"path": path, "content": "x", "mtime": 1}]}).encode()
            return mock.MagicMock(__enter__=lambda s: io.BytesIO(body), __exit__=lambda *a: False)

        with mock.patch("urllib.request.urlopen", return_value=fake("Work/Plans.md")):
            with self.assertRaises(nvs.Refused):
                nvs.fetch_tree("http://nova", "NovaNotes")
        with mock.patch("urllib.request.urlopen", return_value=fake("NovaNotes/ok.md")):
            self.assertEqual(len(nvs.fetch_tree("http://nova", "NovaNotes")), 1)
        with mock.patch("urllib.request.urlopen", return_value=fake("NovaNotes/ok.md")):
            with self.assertRaises(nvs.Refused):
                nvs.fetch_tree("http://nova", "OtherFolder")



def fake_response(body):
    import io
    import unittest.mock as mock

    data = json.dumps(body).encode()
    return mock.MagicMock(__enter__=lambda s: io.BytesIO(data), __exit__=lambda *a: False)


class NovaAccess(unittest.TestCase):
    """Talking to a Nova Notes that has OPEN_NOTEBOOK_PASSWORD set."""

    TREE = {"folder": "NovaNotes", "files": [{"path": "NovaNotes/ok.md", "content": "x", "mtime": 1}]}

    def fetch(self, token):
        import unittest.mock as mock

        with mock.patch("urllib.request.urlopen", return_value=fake_response(self.TREE)) as urlopen:
            nvs.fetch_tree("http://127.0.0.1:5055", "NovaNotes", token)
        return urlopen.call_args[0][0]

    def test_the_token_is_sent_as_a_bearer_and_not_carried_across_redirects(self):
        request = self.fetch("s3cret")
        self.assertEqual(request.unredirected_hdrs.get("Authorization"), "Bearer s3cret")
        self.assertNotIn("Authorization", request.headers)
        self.assertEqual(request.get_header("Authorization"), "Bearer s3cret")

    def test_no_token_sends_no_header(self):
        self.assertIsNone(self.fetch(None).get_header("Authorization"))

    def test_the_token_comes_from_the_environment_or_a_private_file(self):
        self.assertEqual(nvs.load_token(environ={nvs.TOKEN_ENV: " s3cret\n"}), "s3cret")
        self.assertIsNone(nvs.load_token(environ={}))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "token"
            path.write_text("from-file\n")
            os.chmod(path, 0o644)
            with self.assertRaises(nvs.Refused):
                nvs.load_token(path, environ={})
            os.chmod(path, 0o600)
            self.assertEqual(nvs.load_token(path, environ={nvs.TOKEN_ENV: "ignored"}), "from-file")
            with self.assertRaises(nvs.Refused):
                nvs.load_token(Path(tmp) / "missing", environ={})
        with self.assertRaises(nvs.Refused):
            nvs.load_token(environ={nvs.TOKEN_ENV: "two words"})

    def test_plain_http_is_only_for_loopback_unless_allowed(self):
        for url in ["http://127.0.0.1:5055", "http://localhost:5055", "http://[::1]:5055", "https://notes.example.ts.net"]:
            nvs.check_nova_url(url)
        for url in ["http://notes.example.ts.net:5055", "http://192.168.1.20:5055", "ftp://127.0.0.1", "127.0.0.1:5055"]:
            with self.assertRaises(nvs.Refused, msg=url):
                nvs.check_nova_url(url)
        nvs.check_nova_url("http://notes.example.ts.net:5055", allow_insecure_http=True)

    def test_a_401_is_reported_without_the_token(self):
        import io
        import unittest.mock as mock
        import urllib.error

        error = urllib.error.HTTPError("http://127.0.0.1:5055", 401, "Unauthorized", {}, io.BytesIO(b""))
        out = io.StringIO()
        with tempfile.TemporaryDirectory() as tmp:
            settings = Path(tmp) / "settings.json"
            settings.write_text(json.dumps({"couchDB_DBNAME": "vault", "encrypt": True, "passphrase": "p"}))
            os.chmod(settings, 0o600)
            with mock.patch("urllib.request.urlopen", side_effect=error), mock.patch("sys.stdout", out), \
                    mock.patch.dict(os.environ, {nvs.TOKEN_ENV: "s3cret"}), mock.patch("sys.stderr", io.StringIO()):
                code = nvs.main(["--writer", tmp, "--expect-db", "vault"])
        self.assertEqual(code, 1)
        self.assertIn("401", out.getvalue())
        self.assertNotIn("s3cret", out.getvalue())


class Hardening(Case):
    def test_every_new_directory_level_is_private(self):
        old_umask = os.umask(0o022)
        try:
            self.run_sync([file("NovaNotes/Deep/Er/a.md", "one", 1)])
            self.fake.edit("NovaNotes/Deep/Er/a.md", "obsidian")
            self.run_sync([file("NovaNotes/Deep/Er/a.md", "two", 10**12)])
            nvs.private_mkdir(self.writer / "db", self.writer)
        finally:
            os.umask(old_umask)
        overwritten = self.writer / "overwritten"
        dirs = [overwritten, self.writer / "db"] + [p for p in overwritten.rglob("*") if p.is_dir()]
        self.assertGreater(len(dirs), 4)
        for path in dirs:
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o700, path)

    def test_only_digest_pinned_images_count_as_pinned(self):
        self.assertFalse(nvs.is_pinned(nvs.DEFAULT_IMAGE))
        self.assertTrue(nvs.is_pinned("novahq/livesync-cli@sha256:" + "0" * 64))
        self.assertTrue(nvs.is_pinned("sha256:" + "0" * 64))


if __name__ == "__main__":
    unittest.main()
