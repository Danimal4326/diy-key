# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 hutchx86
"""Unit tests for core/instance_manager.py (no systemd, no network)."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from core import instance_manager  # noqa: E402


def entry(**over):
    e = {
        "name": "main",
        "mode": "macvlan",
        "iface": "dk-main",
        "parent_iface": "enP3p49s0",
        "mac": "FCECDAAA1101",
        "mac_display": "FC:EC:DA:AA:11:01",
        "device_id": "11111111-2222-3333-4444-555555555555",
        "hostname": "diy-key-main",
        "state_dir": "/srv/diy-key/instances/main",
        "netmode": "dhcp",
        "ip": "",
        "netmask": "",
        "gateway": "",
    }
    e.update(over)
    return e


class TestNames(unittest.TestCase):
    def test_unit_name(self):
        self.assertEqual(instance_manager._unit_name("main"),
                         "diy-key-instance-main.service")

    def test_mac_helpers(self):
        self.assertEqual(instance_manager._mac_with_colons("FCECDAAA1101"),
                         "FC:EC:DA:AA:11:01")
        mac = instance_manager._random_mac()
        self.assertEqual(len(mac), 12)
        self.assertTrue(mac.startswith("FCECDA"))
        int(mac, 16)

    def test_state_dir_is_under_project(self):
        self.assertEqual(instance_manager.INSTANCES_DIR,
                         os.path.join(instance_manager.PROJECT_DIR, "instances"))


class TestBuildUnit(unittest.TestCase):
    def test_dhcp_macvlan_unit(self):
        unit = instance_manager._build_unit(entry())
        self.assertIn("[Unit]", unit)
        self.assertIn("Description=AI Key emulator instance 'main' (macvlan)", unit)
        self.assertIn("WorkingDirectory=" + instance_manager.PROJECT_DIR, unit)
        self.assertIn("ip link add link enP3p49s0 name dk-main "
                      "address FC:EC:DA:AA:11:01 type macvlan mode bridge", unit)
        self.assertIn("dhclient", unit)
        self.assertIn("-cf /srv/diy-key/instances/main/dhclient.conf", unit)
        self.assertIn("--state-dir /srv/diy-key/instances/main", unit)
        self.assertIn("--mac FCECDAAA1101", unit)
        self.assertIn("--device-id 11111111-2222-3333-4444-555555555555", unit)
        self.assertIn("Restart=on-failure", unit)
        self.assertIn("RestartSec=5", unit)
        self.assertIn("KillMode=control-group", unit)
        self.assertIn("WantedBy=multi-user.target", unit)

    def test_static_unit_uses_prefix(self):
        unit = instance_manager._build_unit(entry(
            netmode="static", ip="10.0.0.50", netmask="255.255.255.0"))
        self.assertIn("ip addr add 10.0.0.50/24 dev dk-main", unit)
        self.assertNotIn("dhclient", unit)

    def test_static_with_gateway(self):
        unit = instance_manager._build_unit(entry(
            netmode="static", ip="10.0.0.50", netmask="255.255.0.0",
            gateway="10.0.0.1"))
        self.assertIn("ip addr add 10.0.0.50/16 dev dk-main", unit)
        self.assertIn("ip route add default via 10.0.0.1 dev dk-main", unit)

    def test_paths_with_spaces_are_quoted(self):
        unit = instance_manager._build_unit(entry(state_dir="/srv/with space/x"))
        self.assertIn("'/srv/with space/x'", unit)

    def test_prepare_state_writes_dhclient_conf(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        instance_manager._prepare_state(entry(state_dir=d))
        with open(os.path.join(d, "dhclient.conf")) as f:
            self.assertEqual(f.read(), 'send host-name "diy-key-main";\n')

    def test_prepare_state_skips_static(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        instance_manager._prepare_state(entry(state_dir=d, netmode="static"))
        self.assertFalse(os.path.exists(os.path.join(d, "dhclient.conf")))


class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)
        self._orig_dir = instance_manager.INSTANCES_DIR
        self._orig_reg = instance_manager.REGISTRY_PATH
        instance_manager.INSTANCES_DIR = self.d
        instance_manager.REGISTRY_PATH = os.path.join(self.d, "registry.json")
        self.addCleanup(self._restore)

    def _restore(self):
        instance_manager.INSTANCES_DIR = self._orig_dir
        instance_manager.REGISTRY_PATH = self._orig_reg

    def test_roundtrip(self):
        reg = {"main": entry()}
        instance_manager._save_registry(reg)
        self.assertEqual(instance_manager._load_registry(), reg)
        with open(instance_manager.REGISTRY_PATH) as f:
            self.assertEqual(json.load(f)["main"]["iface"], "dk-main")

    def test_missing_registry_is_empty(self):
        self.assertEqual(instance_manager._load_registry(), {})

    def test_corrupt_registry_is_empty(self):
        with open(instance_manager.REGISTRY_PATH, "w") as f:
            f.write("{oops")
        self.assertEqual(instance_manager._load_registry(), {})


class TestGuards(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)
        self._orig_dir = instance_manager.INSTANCES_DIR
        self._orig_reg = instance_manager.REGISTRY_PATH
        instance_manager.INSTANCES_DIR = os.path.join(self.d, "instances")
        instance_manager.REGISTRY_PATH = os.path.join(self.d, "registry.json")
        self.addCleanup(self._restore)

    def _restore(self):
        instance_manager.INSTANCES_DIR = self._orig_dir
        instance_manager.REGISTRY_PATH = self._orig_reg

    def test_bad_name_rejected_before_any_side_effect(self):
        with self.assertRaises(instance_manager.InstanceError):
            instance_manager.create_instance("this-name-is-way-too-long")
        with self.assertRaises(instance_manager.InstanceError):
            instance_manager.create_instance("bad name")

    def test_destroy_unknown(self):
        with self.assertRaises(instance_manager.InstanceError):
            instance_manager.destroy_instance("nope")

    def test_destroy_physical_refused(self):
        instance_manager._save_registry({"phys": entry(mode="physical")})
        with self.assertRaises(instance_manager.InstanceError):
            instance_manager.destroy_instance("phys")

    def _destroy_with(self, state_dir):
        os.makedirs(state_dir, exist_ok=True)
        marker = os.path.join(state_dir, "adopt_state.json")
        with open(marker, "w") as f:
            f.write("{}")
        instance_manager._save_registry({"main": entry(state_dir=state_dir)})
        with mock.patch.object(instance_manager, "_systemctl"), \
                mock.patch.object(instance_manager, "_unit_path",
                                  return_value=os.path.join(self.d, "unit.service")):
            instance_manager.destroy_instance("main")
        return marker

    def test_destroy_removes_owned_state(self):
        marker = self._destroy_with(os.path.join(self.d, "instances", "main"))
        self.assertFalse(os.path.exists(marker))

    def test_destroy_keeps_external_state(self):
        marker = self._destroy_with(os.path.join(self.d, "shared-state"))
        self.assertTrue(os.path.exists(marker))


class TestCreateIdentity(unittest.TestCase):
    """create_instance is the migration path for an already-adopted device."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.d, True)
        self._orig_dir = instance_manager.INSTANCES_DIR
        self._orig_reg = instance_manager.REGISTRY_PATH
        instance_manager.INSTANCES_DIR = os.path.join(self.d, "instances")
        instance_manager.REGISTRY_PATH = os.path.join(self.d, "registry.json")
        self.addCleanup(self._restore)
        self._patches = [
            mock.patch.object(instance_manager, "_iface_exists", return_value=True),
            mock.patch.object(instance_manager, "_write_and_start_unit"),
            mock.patch.object(instance_manager, "_get_ip_on_iface", return_value="10.0.0.9"),
            mock.patch.object(instance_manager.config, "load_config",
                              return_value=dict(instance_manager.config.DEFAULTS)),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def _restore(self):
        instance_manager.INSTANCES_DIR = self._orig_dir
        instance_manager.REGISTRY_PATH = self._orig_reg

    def test_reuses_supplied_identity_and_state(self):
        state = os.path.join(self.d, "kept-state")
        e = instance_manager.create_instance("main", parent_iface="eth0",
                                             mac="fc:ec:da:aa:11:01", device_id="dev-1",
                                             state_dir=state)
        self.assertEqual(e["mac"], "FCECDAAA1101")
        self.assertEqual(e["mac_display"], "FC:EC:DA:AA:11:01")
        self.assertEqual(e["device_id"], "dev-1")
        self.assertEqual(e["state_dir"], os.path.abspath(state))
        self.assertEqual(e["iface"], "dk-main")
        self.assertEqual(e["ip"], "10.0.0.9")

    def test_fresh_identity_defaults(self):
        e = instance_manager.create_instance("two", parent_iface="eth0")
        self.assertTrue(e["mac"].startswith(instance_manager.config.UBNT_OUI))
        self.assertEqual(len(e["mac"]), 12)
        self.assertEqual(len(e["device_id"]), 36)  # uuid4
        self.assertEqual(e["state_dir"],
                         os.path.join(instance_manager.INSTANCES_DIR, "two"))

    def test_invalid_mac_rejected(self):
        with self.assertRaises(instance_manager.InstanceError):
            instance_manager.create_instance("bad", parent_iface="eth0", mac="nope")

    def test_duplicate_rejected(self):
        instance_manager.create_instance("dup", parent_iface="eth0")
        with self.assertRaises(instance_manager.InstanceError):
            instance_manager.create_instance("dup", parent_iface="eth0")


if __name__ == "__main__":
    unittest.main()
