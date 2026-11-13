# SPDX-License-Identifier: AGPL-3.0-or-later
"""stage2_grpc server: mutations restart or fail, never return a false success."""
import os
import sys
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "stage2_grpc")):
    if p not in sys.path:
        sys.path.insert(0, p)

import grpc  # noqa: E402
import server as grpc_server  # noqa: E402


class FakeContext:
    def __init__(self):
        self.code = None
        self.details = None

    def peer(self):
        return "peer"

    def abort(self, code, details):
        self.code, self.details = code, details
        raise RuntimeError(details)


class MutationTest(unittest.TestCase):
    def setUp(self):
        self.svc = grpc_server.StubAIKeyService("host", "fcecdaaabbcc", "2.2.8")

    def test_restart_requests_a_real_service_restart(self):
        with mock.patch.object(grpc_server, "_restart_after") as restart:
            out = self.svc.RestartDevice(grpc_server.api_pb2.RestartDeviceRequest(),
                                         FakeContext())
        self.assertIsNotNone(out)
        restart.assert_called_once()

    def test_update_device_is_unimplemented(self):
        ctx = FakeContext()
        with self.assertRaises(RuntimeError):
            self.svc.UpdateDevice(grpc_server.api_pb2.UpdateDeviceRequest(), ctx)
        self.assertEqual(ctx.code, grpc.StatusCode.UNIMPLEMENTED)

    def test_update_settings_is_unimplemented(self):
        ctx = FakeContext()
        with self.assertRaises(RuntimeError):
            self.svc.UpdateSettings(grpc_server.api_pb2.UpdateSettingsRequest(), ctx)
        self.assertEqual(ctx.code, grpc.StatusCode.UNIMPLEMENTED)


if __name__ == "__main__":
    unittest.main()
