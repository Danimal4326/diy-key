#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Healthy when PostgreSQL accepts connections and the adoption API completes a TLS handshake on
# :8080. run_all.py binds its listeners to the interface's own address, not 127.0.0.1, so connect
# to that. A full handshake, not just a TCP connect: adopt_http.py logs a warning for every
# connection that closes before TLS (one per check, live 2026-11-10), and none for one that
# completes the handshake and closes without a request.
set -euo pipefail
/usr/lib/postgresql/17/bin/pg_isready -q -h /var/run/postgresql
ip=$(ip -4 -o addr show "${DIYKEY_IFACE:-eth0}" | awk '{print $4}' | cut -d/ -f1 | head -1)
python3 - "${ip:-127.0.0.1}" <<'PY'
import socket, ssl, sys
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE  # the device's own self-signed cert
with ctx.wrap_socket(socket.create_connection((sys.argv[1], 8080), 5)):
    pass
PY
