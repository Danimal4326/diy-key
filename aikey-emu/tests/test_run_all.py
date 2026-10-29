# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 hutchx86
"""Unit tests for run_all.py's stage wiring (no child processes are started)."""
import os
import sys
import tempfile
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import run_all  # noqa: E402


class TestStageWiring(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.d, True))

    def _run(self, argv):
        proc = mock.MagicMock()
        proc.pid = 1234
        proc.poll.return_value = 1  # triggers one supervised-exit pass
        argv = ["--state-dir", self.d, *argv]  # never touch the real stage1_http/
        with mock.patch.object(run_all.os, "geteuid", return_value=0), \
                mock.patch.object(run_all.config, "load_config_and_logging",
                                  return_value=dict(run_all.config.DEFAULTS)), \
                mock.patch.object(run_all.config, "resolve_ip", return_value="10.9.9.9"), \
                mock.patch.object(run_all.subprocess, "Popen", return_value=proc) as popen, \
                mock.patch.object(run_all.time, "sleep"), \
                mock.patch.object(run_all.sys, "argv", ["run_all.py", *argv]):
            rc = run_all.main()
        return rc, [c.args[0] for c in popen.call_args_list]

    def test_all_stages_share_identity_and_state(self):
        rc, cmds = self._run(["--iface", "eth0",
                             "--mac", "FC:EC:DA:AA:11:01", "--device-id", "dev-1",
                             "--hostname", "aikey-test"])
        self.assertEqual(rc, 1)  # the fake child "exits", so supervision stops
        names = [os.path.basename(c[1]) for c in cmds]
        self.assertEqual(names, ["beacon.py", "adopt_http.py", "ucp4_client.py",
                                 "model_server.py", "server.py"])
        by_name = dict(zip(names, cmds))
        for stage in ("beacon.py", "adopt_http.py", "ucp4_client.py"):
            args = by_name[stage]
            self.assertIn("--state-dir", args)
            self.assertEqual(args[args.index("--state-dir") + 1], self.d)
            self.assertEqual(args[args.index("--mac") + 1], "FC:EC:DA:AA:11:01")
            self.assertIn("--iface", args)

        # gRPC and the model server do not take --state-dir.
        self.assertNotIn("--state-dir", by_name["server.py"])
        self.assertIn("--bind-ip", by_name["model_server.py"])
        self.assertNotIn("--state-dir", by_name["model_server.py"])

    def test_skip_grpc(self):
        _, cmds = self._run(["--iface", "eth0", "--skip-grpc"])
        names = [os.path.basename(c[1]) for c in cmds]
        self.assertNotIn("server.py", names)

    def test_fallback_from_cli(self):
        _, cmds = self._run(["--iface", "eth0", "--skip-grpc",
                             "--fallback-host-port", "192.0.2.5:7442"])
        wss = [c for c in cmds if os.path.basename(c[1]) == "ucp4_client.py"][0]
        self.assertEqual(wss[wss.index("--fallback-host-port") + 1], "192.0.2.5:7442")

    def test_no_fallback_when_console_unset(self):
        # aikey.cfg ships with a blank console host: never hardcode one.
        _, cmds = self._run(["--iface", "eth0", "--skip-grpc"])
        wss = [c for c in cmds if os.path.basename(c[1]) == "ucp4_client.py"][0]
        self.assertNotIn("--fallback-host-port", wss)


class TestStageArgParsers(unittest.TestCase):
    """Every stage run_all starts must accept --config/--debug. The other four
    are covered indirectly by TestStageWiring's argv assertions; model_server is
    the one that previously lacked the common flags, so it gets a direct test."""

    def test_model_server_accepts_common_flags(self):
        stage_dir = os.path.join(_ROOT, "stage4_modelserver")
        if stage_dir not in sys.path:
            sys.path.insert(0, stage_dir)
        import model_server  # noqa: E402

        with mock.patch.object(model_server.http.server, "ThreadingHTTPServer") as srv, \
                mock.patch.object(model_server.sys, "argv", [
                    "model_server.py", "--config", os.path.join(self.tmp, "none.cfg"),
                    "--debug", "--bind-ip", "127.0.0.1", "--port", "17789"]):
            model_server.main()
        srv.return_value.serve_forever.assert_called_once()

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.tmp, True))


if __name__ == "__main__":
    unittest.main()
