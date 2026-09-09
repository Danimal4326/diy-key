#!/usr/bin/env python3
import os, sys, paramiko

HOST = "192.0.2.110"
PORT = 2200
USER = "root"
PASS = "REDACTED"

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
