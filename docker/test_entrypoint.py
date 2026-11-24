"""Run with python3 -m unittest discover -s docker -v (Linux)."""

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ENTRYPOINT = Path(__file__).with_name("entrypoint.sh").resolve()


class EntrypointTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.work = Path(tmp.name)
        self.data = self.work / "data volume"
        self.ownership_changes = self.work / "chown.log"
        self.runtime_switch = self.work / "runtime.log"
        self.env = {"PATH": os.defpath, "DATA_FOLDER": str(self.data)}

        if os.getuid() == 0:
            # The real runtime user must be able to traverse the fixture's parent.
            self.work.chmod(0o755)
        else:
            # Run real filesystem checks as the test user. Only emulate the
            # initial root identity and the boundary that drops privileges.
            self.stub("id", 'if [ "$1" = -u ]; then echo 0; else exec /usr/bin/id "$@"; fi')
            self.stub(
                "setpriv",
                'while [ "$1" != -- ]; do\n'
                '    printf "%s\\n" "$1" >> "$RUNTIME_SWITCH_LOG"\n'
                '    shift\n'
                'done\nshift\nexec "$@"',
            )
            self.env["RUNTIME_SWITCH_LOG"] = str(self.runtime_switch)

        # Never let a regression recursively chown even our throwaway fixture.
        self.env["OWNERSHIP_LOG"] = str(self.ownership_changes)
        self.stub("chown", '/usr/bin/touch "$OWNERSHIP_LOG"')

    def stub(self, name, body):
        tools = self.work / "bin"
        tools.mkdir(exist_ok=True)
        self.env["PATH"] = f"{tools}:{os.defpath}"
        path = tools / name
        path.write_text(f"#!/bin/sh\nset -eu\n{body}\n")
        path.chmod(0o755)

    def start(self, command="echo app-started", *, already_unprivileged=False):
        args = ["/bin/sh", ENTRYPOINT, "/bin/sh", "-c", command]
        if already_unprivileged:
            if os.getuid() == 0:
                args = ["setpriv", "--reuid=10001", "--regid=10001", "--clear-groups", "--", *args]
            else:
                (self.work / "bin" / "id").unlink()
        return subprocess.run(
            args,
            env=self.env,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_existing_incompatible_volume_is_rejected_without_chown(self):
        self.data.mkdir()
        stored_note = self.data / "note.txt"
        stored_note.write_text("existing note")
        self.data.chmod(0o555)
        original_owner = (stored_note.stat().st_uid, stored_note.stat().st_gid)
        self.addCleanup(self.data.chmod, 0o755)

        result = self.start()

        self.assertEqual(
            (result.returncode, result.stdout, self.ownership_changes.exists()),
            (1, "", False),
            result.stderr,
        )
        self.assertIn(str(self.data), result.stderr)
        self.assertIn("10001:10001", result.stderr)
        self.assertIn("backup", result.stderr)
        self.assertEqual(stored_note.read_text(), "existing note")
        self.assertEqual((stored_note.stat().st_uid, stored_note.stat().st_gid), original_owner)

    def prepare_data_dir(self):
        self.data.mkdir()
        if os.getuid() == 0:
            os.chown(self.data, 10001, 10001)

    def test_prepared_volume_with_existing_data_runs_unprivileged(self):
        self.prepare_data_dir()
        note = self.data / "note.txt"
        note.write_text("existing note")

        result = self.start('/usr/bin/id -u; /usr/bin/id -g; cat "$DATA_FOLDER/note.txt"')

        self.assertEqual(result.returncode, 0, result.stderr)
        uid, gid, note_content = result.stdout.splitlines()
        self.assertEqual(note_content, "existing note")
        self.assertFalse(self.ownership_changes.exists())
        if os.getuid() == 0:
            self.assertEqual((uid, gid), ("10001", "10001"))
        else:
            self.assertEqual((uid, gid), (str(os.getuid()), str(os.getgid())))
            self.assertEqual(
                self.runtime_switch.read_text().splitlines(),
                ["--reuid=10001", "--regid=10001", "--clear-groups"],
            )

    def test_fresh_seeded_volume_is_writable_at_startup(self):
        self.prepare_data_dir()

        result = self.start('printf "first run" > "$DATA_FOLDER/note.txt"')

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.data / "note.txt").read_text(), "first run")
        self.assertFalse(self.ownership_changes.exists())
        if os.getuid() == 0:
            note = (self.data / "note.txt").stat()
            self.assertEqual((note.st_uid, note.st_gid), (10001, 10001))

    def test_explicit_root_override_starts_without_changing_ownership(self):
        self.data.mkdir(mode=0o555)
        self.env["NOVA_RUN_AS_ROOT"] = "true"

        result = self.start('/usr/bin/id -u')

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), str(os.getuid()))
        self.assertFalse(self.runtime_switch.exists())
        self.assertFalse(self.ownership_changes.exists())

    def test_explicit_non_root_user_still_gets_volume_diagnostic(self):
        self.data.mkdir(mode=0o555)
        self.addCleanup(self.data.chmod, 0o755)

        result = self.start(already_unprivileged=True)

        self.assertEqual((result.returncode, result.stdout), (1, ""), result.stderr)
        self.assertIn(str(self.data), result.stderr)
        self.assertIn("backup", result.stderr)


if __name__ == "__main__":
    unittest.main()
