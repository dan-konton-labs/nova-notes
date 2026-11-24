"""Exercise the optional systemd drop-in through the Vault Sync CLI."""
import json
import os
import shlex
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


HERE = Path(__file__).resolve().parent
DROP_IN = HERE / "systemd/nova-vault-sync.service.d/credential.conf"


class SystemdCredential(unittest.TestCase):
    def test_pr16_service_can_sync_to_password_protected_nova_from_a_private_file(self):
        # This is PR #16's ExecStart, kept as a fixture so the test does not
        # depend on the PR being checked out or installed on the host.
        commands = [
            "/usr/bin/python3 /opt/nova-vault-sync/nova_vault_sync.py "
            "--writer /opt/nova-vault-sync/writer --expect-db my_vault"
        ]
        self.assertTrue(DROP_IN.is_file(), "missing opt-in credential drop-in for PR #16")
        for line in DROP_IN.read_text().splitlines():
            if line.startswith("ExecStart="):
                command = line.removeprefix("ExecStart=")
                if command:
                    commands.append(command)
                else:
                    commands.clear()
        self.assertEqual(len(commands), 1, "the drop-in must replace PR #16's ExecStart")
        argv = shlex.split(commands[0])
        self.assertIn("--token-file", argv)
        token_arg = argv.index("--token-file") + 1
        self.assertEqual(argv[token_arg], "/etc/nova-vault-sync/nova-token")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            writer = root / "writer"
            writer.mkdir(mode=0o700)
            settings = writer / "settings.json"
            settings.write_text(json.dumps({"couchDB_DBNAME": "my_vault", "encrypt": True, "passphrase": "fixture"}))
            settings.chmod(0o600)
            token = os.urandom(16).hex()
            credential = root / "nova-token"
            fd = os.open(credential, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                stream.write(token + "\n")
            docker = root / "docker"
            docker.write_text('#!/bin/sh\ncase "$*" in\n  *" sync"|*" ls NovaNotes/") exit 0;;\n  *) exit 1;;\nesac\n')
            docker.chmod(0o700)

            expected_authorization = "Bearer " + token

            class Nova(BaseHTTPRequestHandler):
                def log_message(self, format, *args):
                    pass

                def do_GET(self):
                    if self.path != "/api/vault/files?content=true":
                        self.send_error(404)
                    elif self.headers.get("Authorization") != expected_authorization:
                        self.send_error(401)
                    else:
                        self.send_response(200)
                        self.end_headers()
                        self.wfile.write(b'{"folder":"NovaNotes","files":[]}')

            with HTTPServer(("127.0.0.1", 0), Nova) as server:
                thread = threading.Thread(target=server.serve_forever)
                thread.start()
                try:
                    argv[:2] = [sys.executable, str(HERE / "nova_vault_sync.py")]
                    argv[argv.index("--writer") + 1] = str(writer)
                    argv[token_arg] = str(credential)
                    argv += ["--nova-url", f"http://127.0.0.1:{server.server_port}", "--dry-run"]
                    env = {"PATH": str(root), "LC_ALL": "C"}
                    with self.subTest("private credential file"):
                        result = subprocess.run(argv, capture_output=True, text=True, timeout=10, env=env)
                        self.assertNotIn(token, " ".join(argv) + result.stdout + result.stderr)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertEqual(json.loads(result.stdout)["would_push"], [])

                    with self.subTest("group or world readable file is refused"):
                        credential.chmod(0o644)
                        result = subprocess.run(argv, capture_output=True, text=True, timeout=10, env=env)
                        self.assertNotIn(token, result.stdout + result.stderr)
                        self.assertEqual(result.returncode, 3, result.stdout + result.stderr)
                        self.assertIn("must be mode 0600", json.loads(result.stdout)["refused"])

                    with self.subTest("base unit can still run without credentials"):
                        credential.unlink()
                        expected_authorization = None
                        base_argv = argv[:token_arg - 1] + argv[token_arg + 1:]
                        result = subprocess.run(base_argv, capture_output=True, text=True, timeout=10, env=env)
                        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                        self.assertEqual(json.loads(result.stdout)["would_push"], [])
                finally:
                    server.shutdown()
                    thread.join()
