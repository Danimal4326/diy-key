#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Healthy when PostgreSQL accepts connections and the adoption API on :8080 answers an HTTPS
# request. run_all.py binds its listeners to the interface's own address, not 127.0.0.1, so
# connect to that.
#
# A whole HEAD request, read to the end, because adopt_http.py turns anything less into log
# noise (live 2026-11-10): a bare TCP connect logs "TLS handshake ... failed" every check, and a
# handshake closed without a request dies in readline with BrokenPipeError (the server's TLS 1.3
# session tickets hit the closed socket) and prints a traceback. HEAD has no handler, so
# http.server answers 501 through log_message, which adopt_http.py silences; GET would log an
# INFO line per check.
set -euo pipefail
/usr/lib/postgresql/17/bin/pg_isready -q -h /var/run/postgresql
ip=$(ip -4 -o addr show "${DIYKEY_IFACE:-eth0}" | awk '{print $4}' | cut -d/ -f1 | head -1)
python3 - "${ip:-127.0.0.1}" <<'PY'
import http.client, ssl, sys
conn = http.client.HTTPSConnection(sys.argv[1], 8080, timeout=5,
                                   context=ssl._create_unverified_context())  # self-signed cert
conn.request("HEAD", "/api/info")
resp = conn.getresponse()
resp.read()
conn.close()
sys.exit(0 if 100 <= resp.status < 600 else 1)
PY
