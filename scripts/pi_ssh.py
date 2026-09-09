#!/usr/bin/env python3
import sys, paramiko

HOST = "192.0.2.110"
PORT = 2200
USER = "root"
PASS = "REDACTED"

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
