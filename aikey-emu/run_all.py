#!/usr/bin/env python3
"""Runs all stage1/stage2 components together. Needs root (binds :22, :10001).

Usage:
  sudo python3 run_all.py --iface eth0
"""
import argparse
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--hostname", default="aikey-emu")
    ap.add_argument("--skip-grpc", action="store_true",
                     help="skip gRPC stage (use if generate_pb2.sh hasn't been run yet)")
    args = ap.parse_args()

    if os.geteuid() != 0:
        sys.exit("run_all.py needs root (binds ports 22, 10001, 50051)")

    procs = []

    procs.append(subprocess.Popen(
        [sys.executable, os.path.join(HERE, "stage1_discovery", "beacon.py"),
         "--iface", args.iface, "--hostname", args.hostname]))

    procs.append(subprocess.Popen(
        [sys.executable, os.path.join(HERE, "stage1_http", "adopt_http.py"),
         "--iface", args.iface, "--hostname", args.hostname]))

    procs.append(subprocess.Popen(
        [sys.executable, os.path.join(HERE, "stage3_wss", "ucp4_client.py"),
         "--iface", args.iface, "--fallback-host-port", "192.0.2.5:7442"]))

    procs.append(subprocess.Popen(
        [sys.executable, os.path.join(HERE, "stage4_modelserver", "model_server.py")]))

    if not args.skip_grpc:
        generated = os.path.join(HERE, "stage2_grpc", "generated")
        if not os.path.isdir(generated) or not os.listdir(generated):
            print("!! stage2_grpc/generated is empty -- run stage2_grpc/generate_pb2.sh "
                  "first, or pass --skip-grpc", file=sys.stderr)
        else:
            procs.append(subprocess.Popen(
                [sys.executable, os.path.join(HERE, "stage2_grpc", "server.py"),
                 "--iface", args.iface, "--hostname", args.hostname]))

    print(f"Started {len(procs)} components. Ctrl-C to stop all.")
    try:
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        for p in procs:
            p.terminate()


if __name__ == "__main__":
    main()
