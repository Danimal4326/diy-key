#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 hutchx86
"""Run every AI Key stage as one supervised process group.

All listeners are on unprivileged ports (>1024); root is needed for interface
scoping/macvlan (SO_BINDTODEVICE, dhclient), not for the ports themselves.
Settings come from diykey.cfg; identity and state paths are resolved once here
and passed to every stage so they cannot disagree.

Usage:
  sudo python3 run_all.py --iface eth0
  sudo python3 run_all.py --state-dir instances/main --mac FC:EC:DA:00:00:01
"""
import argparse
import os
import signal
import subprocess
import sys
import time

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from core import config  # noqa: E402

HERE = _THIS_DIR


def _stop_all(procs):
    """Terminate the whole stage group, then kill anything that ignored TERM."""
    for p in procs:
        if p.poll() is None:
            p.terminate()
    deadline = time.monotonic() + 5
    for p in procs:
        try:
            p.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            p.kill()
    # Reap the killed ones too, so no zombie is left behind on shutdown.
    for p in procs:
        try:
            p.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass


def main():
    cfg = config.load_config_and_logging(sys.argv[1:])
    ap = argparse.ArgumentParser(description=__doc__)
    config.add_common_flags(ap)
    ap.add_argument("--iface", default=None,
                    help="interface to bind/derive identity from; defaults to cfg "
                         "[network] iface, else the default-route interface")
    ap.add_argument("--bind-ip", default="0.0.0.0",
                    help="address for the TCP listeners; defaults to --iface's own IP so "
                         "several instances can share a box")
    ap.add_argument("--mac", default=None,
                    help="defaults to a real Ubiquiti OUI (FC:EC:DA) + the interface's "
                         "own last 3 octets")
    ap.add_argument("--hostname", default=cfg["hostname"])
    ap.add_argument("--fallback-host-port", default=None,
                    help="console host:port for a tokenless reconnect; defaults to "
                         "[console] host/port from diykey.cfg")
    ap.add_argument("--state-dir", default=None,
                    help="adopt_state.json/server.crt/server.key directory; defaults to "
                         "stage1_http/ (keeps an already-adopted identity)")
    ap.add_argument("--device-id", default=cfg["device_id"])
    ap.add_argument("--skip-grpc", action="store_true",
                    help="skip the gRPC stage (also skipped when generated/ is empty)")
    args = ap.parse_args()

    if os.geteuid() != 0 and sys.platform != "darwin":
        sys.exit("run_all.py needs root (interface scoping/macvlan; ports are unprivileged)")

    args.iface = config.resolve_iface(args.iface, cfg)
    print(f"Using interface {args.iface}")

    if args.mac is None:
        args.mac = config.resolve_mac(args.iface, None, cfg)
        print(f"No --mac given; using MAC {args.mac}")

    # Bind each listener to the interface's own IP so a second instance's ports
    # stay free. macOS without root can only bind the wildcard address.
    if args.bind_ip == "0.0.0.0" and not (
            sys.platform == "darwin" and os.geteuid() != 0):
        resolved = config.resolve_ip(args.iface, cfg)
        if resolved and resolved != "0.0.0.0":
            args.bind_ip = resolved
            print(f"No --bind-ip given; using {args.iface}'s own IP {resolved}")
        else:
            print(f"WARNING: could not resolve {args.iface}'s IP, binding 0.0.0.0 -- "
                  f"this instance will collide with any other instance's ports")

    args.state_dir = config.resolve_state_dir(args.state_dir)
    print(f"Using state dir {args.state_dir}")

    fallback = args.fallback_host_port
    if not fallback and cfg.get("host"):
        fallback = f"{cfg['host']}:{cfg.get('port') or '7442'}"

    common = []
    if args.config:
        common += ["--config", args.config]
    if args.debug:
        common += ["--debug"]

    def stage(script, *extra):
        return [sys.executable, os.path.join(HERE, *script.split("/")), *extra, *common]

    identity = ["--iface", args.iface, "--bind-ip", args.bind_ip, "--mac", args.mac,
                "--state-dir", args.state_dir]
    net_identity = ["--iface", args.iface, "--bind-ip", args.bind_ip, "--mac", args.mac]

    beacon_args = stage("stage1_discovery/beacon.py", *identity, "--hostname", args.hostname,
                        "--device-id", args.device_id)
    http_args = stage("stage1_http/adopt_http.py", *identity, "--hostname", args.hostname,
                      "--device-id", args.device_id)
    wss_args = stage("stage3_wss/ucp4_client.py", *identity,
                     "--device-id", args.device_id, "--guid", cfg["guid"])
    model_args = stage("stage4_modelserver/model_server.py", "--bind-ip", args.bind_ip)
    if fallback:
        wss_args += ["--fallback-host-port", fallback]

    procs = [
        subprocess.Popen(beacon_args),
        subprocess.Popen(http_args),
        subprocess.Popen(wss_args),
        subprocess.Popen(model_args),
    ]

    if not args.skip_grpc:
        generated = os.path.join(HERE, "stage2_grpc", "generated")
        if not os.path.isdir(generated) or not os.listdir(generated):
            print("!! stage2_grpc/generated is empty -- run stage2_grpc/generate_pb2.sh "
                  "first, or pass --skip-grpc", file=sys.stderr)
        else:
            procs.append(subprocess.Popen(stage(
                "stage2_grpc/server.py", *net_identity, "--hostname", args.hostname)))

    print(f"Started {len(procs)} components. Ctrl-C to stop all.")

    def _on_term(*_):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _on_term)
    signal.signal(signal.SIGINT, _on_term)
    try:
        while True:
            for p in procs:
                rc = p.poll()
                if rc is not None:
                    print(f"Component {p.pid} exited rc={rc}; stopping the rest")
                    _stop_all(procs)
                    return 1
            time.sleep(1)
    except KeyboardInterrupt:
        _stop_all(procs)
        return 0


if __name__ == "__main__":
    sys.exit(main())
