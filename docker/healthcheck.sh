#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Healthy when PostgreSQL accepts connections and the adoption API listens on :8080. run_all.py
# binds its listeners to the interface's own address, not 127.0.0.1, so connect to that.
set -euo pipefail
/usr/lib/postgresql/17/bin/pg_isready -q -h /var/run/postgresql
ip=$(ip -4 -o addr show "${DIYKEY_IFACE:-eth0}" | awk '{print $4}' | cut -d/ -f1 | head -1)
python3 -c 'import socket, sys; socket.create_connection((sys.argv[1], 8080), 5).close()' "${ip:-127.0.0.1}"
