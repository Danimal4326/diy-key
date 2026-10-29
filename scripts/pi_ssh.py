# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 hutchx86
import os, sys, paramiko

HOST = os.environ.get("ORANGEPI_HOST")
PORT = int(os.environ.get("ORANGEPI_SSH_PORT", "2200"))
USER = "root"
PASS = os.environ.get("ORANGEPI_PASS")
if not HOST:
    sys.exit("ORANGEPI_HOST is unset")
if not PASS:
    sys.exit("ORANGEPI_PASS is unset")

def run(cmd, timeout=120):
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, port=PORT, username=USER, password=PASS, timeout=15, look_for_keys=False, allow_agent=False)
    stdin, stdout, stderr = c.exec_command(cmd, timeout=timeout)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    rc = stdout.channel.recv_exit_status()
    c.close()
    return rc, out, err

if __name__ == "__main__":
    cmd = sys.argv[1]
    rc, out, err = run(cmd)
    sys.stdout.write(out)
    sys.stderr.write(err)
    sys.exit(rc)
