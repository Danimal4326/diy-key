#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 hutchx86
"""Run multiple independent AI Key emulator instances, each as its own systemd
unit that survives reboot and keeps a fixed identity (MAC/device-id,
generated once at creation).

Two modes: macvlan (own sub-interface + MAC + DHCP lease, a separate L2
device) and physical (bound directly to a real interface, no macvlan/dhclient).

CLI usage:
  sudo python3 core/instance_manager.py create <name> [--parent-iface <iface>]
  sudo python3 core/instance_manager.py destroy <name>
  sudo python3 core/instance_manager.py refresh <name>
  python3 core/instance_manager.py list
"""
import argparse
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_THIS_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from core import config  # noqa: E402

PROJECT_DIR = _PROJECT_DIR
INSTANCES_DIR = os.path.join(PROJECT_DIR, "instances")
REGISTRY_PATH = os.path.join(INSTANCES_DIR, "registry.json")
RUN_ALL = os.path.join(PROJECT_DIR, "run_all.py")
SYSTEMD_DIR = "/etc/systemd/system"

NAME_RE = re.compile(r"^[a-zA-Z0-9_-]{1,12}$")
DHCP_TIMEOUT_S = 20


class InstanceError(Exception):
    pass


def _unit_name(name):
    return f"diy-key-instance-{name}.service"


def _unit_path(name):
    return os.path.join(SYSTEMD_DIR, _unit_name(name))


def _load_registry():
    try:
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as e:
        # Returning {} here would make create overwrite every existing instance.
        raise RuntimeError(f"instance registry {REGISTRY_PATH} is unreadable: {e}") from e


def _save_registry(reg):
    config.atomic_write_json(REGISTRY_PATH, reg)


def _iface_exists(iface):
    return os.path.exists(f"/sys/class/net/{iface}")


def _random_mac():
    return config.UBNT_OUI + "".join(f"{random.randint(0, 255):02X}" for _ in range(3))


def _mac_with_colons(mac_nosep):
    return ":".join(mac_nosep[i:i + 2] for i in range(0, 12, 2))


def _systemctl(*args):
    return subprocess.run(["systemctl", *args], capture_output=True, text=True)


def _is_active(name):
    r = _systemctl("is-active", _unit_name(name))
    return r.stdout.strip() == "active"


def _get_ip_on_iface(iface):
    try:
        out = subprocess.run(["ip", "-4", "-json", "addr", "show", iface],
                             capture_output=True, text=True, timeout=5)
        data = json.loads(out.stdout or "[]")
        for link in data:
            for addr in link.get("addr_info", []):
                if addr.get("family") == "inet":
                    return addr.get("local")
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        pass
    return None


def list_instances():
    """Registry entries enriched with live status; `running` cross-checks the
    real systemd unit state rather than trusting the registry."""
    reg = _load_registry()
    out = []
    for name, entry in sorted(reg.items()):
        out.append({
            **entry, "name": name,
            "running": _is_active(name),
            "ip": _get_ip_on_iface(entry["iface"]),
        })
    return out


def _build_unit(entry):
    common_args = [
        "--iface", entry["iface"],
        "--mac", entry["mac"],
        "--device-id", entry["device_id"],
        "--hostname", entry["hostname"],
        "--state-dir", entry["state_dir"],
    ]
    exec_start = " ".join(shlex.quote(a) for a in [sys.executable, RUN_ALL] + common_args)

    pre_lines = []
    post_lines = []
    if entry["mode"] == "macvlan":
        iface, parent, mac_colons = entry["iface"], entry["parent_iface"], entry["mac_display"]
        # '-' prefix tolerates nonzero exit: cleanup only does real work after a
        # crash left the interface behind (a normal first start has nothing to delete).
        pre_lines = [
            f"ExecStartPre=-/sbin/ip link del {shlex.quote(iface)}",
            f"ExecStartPre=/sbin/ip link add link {shlex.quote(parent)} "
            f"name {shlex.quote(iface)} address {shlex.quote(mac_colons)} type macvlan mode bridge",
            f"ExecStartPre=/sbin/ip link set {shlex.quote(iface)} up",
        ]
        if entry.get("netmode") == "static":
            prefix = config.netmask_to_prefix(entry.get("netmask")) or 24
            pre_lines.append(
                f"ExecStartPre=/sbin/ip addr add {shlex.quote(str(entry['ip']))}/{prefix} "
                f"dev {shlex.quote(iface)}")
            if entry.get("gateway"):
                pre_lines.append(
                    f"ExecStartPre=-/sbin/ip route add default via "
                    f"{shlex.quote(str(entry['gateway']))} dev {shlex.quote(iface)}")
            post_lines = [
                f"ExecStopPost=-/sbin/ip link del {iface}",
            ]
        else:
            leases = os.path.join(entry["state_dir"], "dhclient.leases")
            pidfile = f"/run/dhclient-{iface}.pid"
            # dhclient 4.4.3-P1 has no -H flag, so override DHCP option 12 via a
            # per-interface config file with `send host-name "...";` (-cf).
            dhclient_conf = os.path.join(entry["state_dir"], "dhclient.conf")
            pre_lines.append(f"ExecStartPre=/usr/sbin/dhclient -pf {shlex.quote(pidfile)} "
                             f"-lf {shlex.quote(leases)} -cf {shlex.quote(dhclient_conf)} "
                             f"{shlex.quote(iface)}")
            post_lines = [
                f"ExecStopPost=-/usr/bin/pkill -f 'dhclient.* {iface}$'",
                f"ExecStopPost=-/sbin/ip link del {iface}",
            ]

    lines = [
        "[Unit]",
        f"Description=AI Key emulator instance '{entry['name']}' ({entry['mode']})",
        # postgresql.service orders after the console's result database when one exists;
        # a host without PostgreSQL still starts (ordering only, no hard dependency).
        "After=network-online.target postgresql.service",
        "Wants=network-online.target",
        "",
        "[Service]",
        "Type=simple",
        f"WorkingDirectory={PROJECT_DIR}",
        # Optional per-instance AI settings (DIYKEY_RAM_MODEL, DIYKEY_CLIP, ...); missing is fine.
        f"EnvironmentFile=-{os.path.join(entry['state_dir'], 'diykey.env')}",
        *pre_lines,
        f"ExecStart={exec_start}",
        *post_lines,
        "Restart=on-failure",
        "RestartSec=5",
        "KillMode=control-group",
        "TimeoutStopSec=10",
        "",
        "[Install]",
        "WantedBy=multi-user.target",
        "",
    ]
    return "\n".join(lines)


def _prepare_state(entry):
    """Write the per-instance dhclient config used by the unit (capability
    files are written by _build_unit's caller, not by the renderer)."""
    if entry["mode"] == "macvlan" and entry.get("netmode") != "static":
        os.makedirs(entry["state_dir"], exist_ok=True)
        with open(os.path.join(entry["state_dir"], "dhclient.conf"), "w") as f:
            f.write(f'send host-name "{entry["hostname"]}";\n')


def _write_and_start_unit(entry):
    _prepare_state(entry)
    with open(_unit_path(entry["name"]), "w") as f:
        f.write(_build_unit(entry))
    _systemctl("daemon-reload")
    r = _systemctl("enable", "--now", _unit_name(entry["name"]))
    if r.returncode != 0:
        raise InstanceError(f"systemd failed to start the instance: {r.stderr.strip()}")


def refresh_instance(name):
    """Rewrite and restart an existing instance's unit from the registry,
    picking up the current interpreter (sys.executable) and paths while
    preserving identity (MAC/device-id)."""
    reg = _load_registry()
    entry = reg.get(name)
    if not entry:
        raise InstanceError(f"no such instance {name!r}")
    _write_and_start_unit(entry)
    _systemctl("restart", _unit_name(name))
    return {**entry, "name": name, "running": True}


def create_instance(name, parent_iface=None, ip=None, netmask=None, gateway=None,
                    mac=None, device_id=None, state_dir=None):
    if not NAME_RE.match(name):
        raise InstanceError(
            "name must be 1-12 characters, letters/digits/-/_ only (kept short: "
            "it becomes part of a Linux interface name, 15-char hard limit)")
    reg = _load_registry()
    if name in reg:
        raise InstanceError(f"instance {name!r} already exists")

    cfg = config.load_config()
    parent_iface = parent_iface or config.resolve_parent_iface(cfg)
    if not _iface_exists(parent_iface):
        raise InstanceError(
            f"parent interface {parent_iface!r} does not exist -- pass --parent-iface or "
            f"set [network] parent_iface/[network] iface in diykey.cfg")

    # An already-adopted device must keep its MAC/device-id/state dir: passing
    # all three migrates the existing identity instead of minting a new one.
    if mac:
        mac_clean = mac.replace(":", "").replace("-", "").upper()
        if not re.fullmatch(r"[0-9A-F]{12}", mac_clean):
            raise InstanceError(f"invalid MAC {mac!r}")
    else:
        mac_clean = _random_mac()
    if device_id is not None and not str(device_id).strip():
        raise InstanceError("--device-id must not be empty")

    iface = f"dk-{name}"[:15]
    state_dir = os.path.abspath(state_dir) if state_dir else os.path.join(INSTANCES_DIR, name)
    os.makedirs(state_dir, exist_ok=True)

    netmode = "dhcp"
    if ip or (str(cfg["mode"]).lower() == "static" and cfg.get("ip")):
        netmode = "static"
        ip = ip or cfg["ip"]
        netmask = netmask or cfg.get("netmask") or "255.255.255.0"
        gateway = gateway or cfg.get("gateway") or ""

    entry = {
        "name": name,
        "mode": "macvlan",
        "iface": iface,
        "parent_iface": parent_iface,
        "mac": mac_clean,
        "device_id": str(device_id) if device_id else str(uuid.uuid4()),
        "hostname": f"{cfg['hostname']}-{name}",
        "state_dir": state_dir,
        "netmode": netmode,
        "ip": ip or "",
        "netmask": netmask or "",
        "gateway": gateway or "",
        "created": time.time(),
    }
    entry["mac_display"] = _mac_with_colons(entry["mac"])

    reg[name] = entry
    _save_registry(reg)
    try:
        _write_and_start_unit(entry)
    except Exception:
        del reg[name]
        _save_registry(reg)
        raise

    # A failed DHCP lease isn't a systemd failure (dhclient exits 0 on timeout),
    # so poll for an address to surface a useful error.
    deadline = time.time() + DHCP_TIMEOUT_S
    ip_addr = None
    while time.time() < deadline:
        ip_addr = _get_ip_on_iface(iface)
        if ip_addr:
            break
        time.sleep(0.5)
    if not ip_addr:
        if netmode == "static":
            detail = f"static IP {entry['ip']} did not appear on {iface}"
        else:
            detail = f"no DHCP lease appeared on {iface} -- check the LAN's DHCP server"
        raise InstanceError(
            f"instance created and started, but {detail} within "
            f"{DHCP_TIMEOUT_S}s; instance is left running, tear it down if you want to retry")

    return {**entry, "name": name, "running": True, "ip": ip_addr}


def destroy_instance(name):
    reg = _load_registry()
    entry = reg.get(name)
    if not entry:
        raise InstanceError(f"no such instance {name!r}")
    if entry["mode"] == "physical":
        raise InstanceError(
            f"{name!r} is bound to a physical interface ({entry['iface']}) -- refusing to "
            "auto-tear-down its network config to avoid taking down the box's real "
            "connectivity; stop it manually with systemctl if you really mean to")

    _systemctl("disable", "--now", _unit_name(name))
    try:
        os.remove(_unit_path(name))
    except OSError:
        pass
    _systemctl("daemon-reload")

    # Destroy is a real teardown: delete only state dirs this manager owns, never an
    # external --state-dir (the shared stage1_http/ location must survive).
    state_dir = entry.get("state_dir")
    if state_dir:
        owned_root = os.path.abspath(INSTANCES_DIR) + os.sep
        if os.path.abspath(state_dir).startswith(owned_root):
            shutil.rmtree(state_dir, ignore_errors=True)
        else:
            print(f"kept external state dir {state_dir} (not under {INSTANCES_DIR})",
                  file=sys.stderr)

    del reg[name]
    _save_registry(reg)


def register_physical_instance(name, iface, mac, device_id, hostname, state_dir):
    """One-time migration path for an already-adopted physical AI Key: brings it
    under the systemd lifecycle without touching the interface or identity.
    Run by hand once; not exposed as a CLI verb."""
    reg = _load_registry()
    if name in reg:
        raise InstanceError(f"instance {name!r} already exists")
    entry = {
        "name": name,
        "mode": "physical",
        "iface": iface,
        "parent_iface": None,
        "mac": mac,
        "mac_display": _mac_with_colons(mac),
        "device_id": device_id,
        "hostname": hostname,
        "state_dir": state_dir,
        "created": time.time(),
    }
    reg[name] = entry
    _save_registry(reg)
    try:
        _write_and_start_unit(entry)
    except Exception:
        del reg[name]
        _save_registry(reg)
        raise
    return entry


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_create = sub.add_parser("create")
    p_create.add_argument("name")
    p_create.add_argument("--parent-iface", default=None,
                          help="physical interface to build the macvlan on; defaults to "
                               "cfg [network] parent_iface, then [network] iface, then the "
                               "default-route interface")
    p_create.add_argument("--ip", default=None,
                          help="static IP for this instance (overrides cfg [network] mode/ip)")
    p_create.add_argument("--netmask", default=None)
    p_create.add_argument("--gateway", default=None)
    p_create.add_argument("--mac", default=None,
                          help="reuse an existing identity's MAC (e.g. migrating an already-"
                               "adopted device); default is a fresh Ubiquiti-OUI MAC")
    p_create.add_argument("--device-id", default=None,
                          help="reuse an existing identity's device-id (default: fresh UUID)")
    p_create.add_argument("--state-dir", default=None,
                          help="existing adopt_state.json/server.crt/server.key directory to "
                               "keep (default: instances/<name>/)")

    p_destroy = sub.add_parser("destroy")
    p_destroy.add_argument("name")

    p_refresh = sub.add_parser("refresh",
                               help="rewrite an existing instance's unit (current interpreter/"
                                    "paths) and restart it, preserving its identity")
    p_refresh.add_argument("name")

    sub.add_parser("list")

    args = ap.parse_args()

    if os.geteuid() != 0 and args.cmd in ("create", "destroy", "refresh"):
        sys.exit(f"{args.cmd} needs root (creates/deletes network interfaces and systemd units)")

    try:
        if args.cmd == "create":
            entry = create_instance(args.name, args.parent_iface,
                                    ip=args.ip, netmask=args.netmask, gateway=args.gateway,
                                    mac=args.mac, device_id=args.device_id,
                                    state_dir=args.state_dir)
            print(json.dumps(entry, indent=2))
        elif args.cmd == "destroy":
            destroy_instance(args.name)
            print(f"destroyed {args.name!r}")
        elif args.cmd == "refresh":
            entry = refresh_instance(args.name)
            print(json.dumps(entry, indent=2))
        elif args.cmd == "list":
            print(json.dumps(list_instances(), indent=2))
    except InstanceError as e:
        sys.exit(f"error: {e}")


if __name__ == "__main__":
    main()
