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

local_root = sys.argv[1]
remote_root = sys.argv[2]

t = paramiko.Transport((HOST, PORT))
t.connect(username=USER, password=PASS)
sftp = paramiko.SFTPClient.from_transport(t)

def mkdirs(remote_path):
    parts = remote_path.strip("/").split("/")
    cur = ""
    for p in parts:
        cur += "/" + p
        try:
            sftp.mkdir(cur)
        except IOError:
            pass

for dirpath, dirnames, filenames in os.walk(local_root):
    rel = os.path.relpath(dirpath, local_root)
    remote_dir = remote_root if rel == "." else remote_root + "/" + rel
    mkdirs(remote_dir)
    for fn in filenames:
        local_file = os.path.join(dirpath, fn)
        remote_file = remote_dir + "/" + fn
        sftp.put(local_file, remote_file)
        print("put", remote_file)

sftp.close()
t.close()
