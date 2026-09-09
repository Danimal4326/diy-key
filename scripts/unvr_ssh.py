#!/usr/bin/env python3
import sys, time, paramiko

HOST = "192.0.2.5"
USER = "root"
PASS = "REDACTED"

def _handler(title, instructions, prompt_list):
    return [PASS for _ in prompt_list]

def run(cmd, timeout=60):
    t = paramiko.Transport((HOST, 22))
    t.connect()
    t.auth_interactive(USER, _handler)
    chan = t.open_session()
    chan.settimeout(timeout)
    chan.exec_command(cmd)
    out, err = b"", b""
    while not chan.exit_status_ready():
        while chan.recv_ready():
            out += chan.recv(65536)
        while chan.recv_stderr_ready():
            err += chan.recv_stderr(65536)
        time.sleep(0.05)
    time.sleep(0.2)
    while chan.recv_ready():
        out += chan.recv(65536)
    while chan.recv_stderr_ready():
        err += chan.recv_stderr(65536)
    rc = chan.recv_exit_status()
    t.close()
    return rc, out.decode(errors="replace"), err.decode(errors="replace")

if __name__ == "__main__":
    rc, out, err = run(sys.argv[1])
    sys.stdout.write(out)
    sys.stderr.write(err)
    sys.exit(rc)
