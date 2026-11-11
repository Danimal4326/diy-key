#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Container entrypoint: PostgreSQL (+ pgvector) for the console's AI results, then run_all.py.
#
# First start: initdb under /data/pg, a self-signed server cert (the console requires SSL but
# doesn't validate it), and role/db "unifi-protect" (password "unifi-protect", SUPERUSER: the
# console's own unconditional CREATE EXTENSION vector needs it) with the vector extension, as
# diykey/README.md "PostgreSQL setup" describes. Later starts reuse it as is, so a password
# the console rotated (changeUserPassword) is kept.
#
# Settings (env): the DIYKEY_* identity/console overrides from core/config.py, plus
#   DIYKEY_IFACE      interface run_all.py binds to (default eth0)
#   DIYKEY_HOSTNAME   advertised hostname (default diy-key)
#   DIYKEY_PG_ALLOW   pg_hba address for the console's hostssl login (default samenet: any
#                     subnet this container is directly on)
set -euo pipefail

PGBIN=/usr/lib/postgresql/17/bin
PGDATA=/data/pg
RUNDIR=/var/run/postgresql
IFACE=${DIYKEY_IFACE:-eth0}
HOSTNAME_=${DIYKEY_HOSTNAME:-diy-key}
PG_ALLOW=${DIYKEY_PG_ALLOW:-samenet}

log() { echo "entrypoint: $*"; }
as_pg() { runuser -u postgres -- "$@"; }
psql_pg() { as_pg "$PGBIN/psql" -h "$RUNDIR" -d "${2:-postgres}" -qtAX -v ON_ERROR_STOP=1 -c "$1"; }

mkdir -p /data/state /data/firmware /data/cache /data/models "$RUNDIR"
# PostgreSQL runs as the postgres user, which must get through /data to its own (0700) pg/.
# A bind mount's root can be 0700 (e.g. made by mktemp); add only the traverse bit.
chmod a+x /data
chown postgres:postgres "$RUNDIR"
chmod 2775 "$RUNDIR"

if [ ! -s "$PGDATA/PG_VERSION" ]; then
    log "creating PostgreSQL cluster in $PGDATA"
    mkdir -p "$PGDATA"
    chown postgres:postgres "$PGDATA"
    chmod 700 "$PGDATA"
    as_pg "$PGBIN/initdb" -D "$PGDATA" -U postgres --auth-local=peer --auth-host=scram-sha-256 \
        --encoding=UTF8 --locale=C.UTF-8 >/dev/null
    openssl req -x509 -nodes -newkey rsa:2048 -days 3650 -subj "/CN=$HOSTNAME_" \
        -keyout "$PGDATA/server.key" -out "$PGDATA/server.crt" 2>/dev/null
    chown postgres:postgres "$PGDATA/server.key" "$PGDATA/server.crt"
    chmod 600 "$PGDATA/server.key"
    cat >> "$PGDATA/postgresql.conf" <<EOF

# diy-key (docker/entrypoint.sh)
listen_addresses = '*'
unix_socket_directories = '$RUNDIR'
ssl = on
ssl_cert_file = 'server.crt'
ssl_key_file = 'server.key'
EOF
    cat > "$PGDATA/pg_hba.conf" <<EOF
# diy-key (docker/entrypoint.sh): peer for local admin (ucp4_client.py's password rotation),
# SSL + password for the console.
local   all             all                                     peer
hostssl unifi-protect   unifi-protect   $PG_ALLOW               scram-sha-256
EOF
    chown postgres:postgres "$PGDATA/pg_hba.conf"
    NEW_CLUSTER=1
else
    NEW_CLUSTER=0
fi

log "starting PostgreSQL"
as_pg "$PGBIN/pg_ctl" -D "$PGDATA" -l "$PGDATA/postgresql.log" -w -t 60 start >/dev/null

if [ "$(psql_pg "SELECT 1 FROM pg_roles WHERE rolname = 'unifi-protect'")" != 1 ]; then
    log "creating role and database unifi-protect"
    psql_pg "CREATE ROLE \"unifi-protect\" WITH LOGIN PASSWORD 'unifi-protect' SUPERUSER"
fi
if [ "$(psql_pg "SELECT 1 FROM pg_database WHERE datname = 'unifi-protect'")" != 1 ]; then
    psql_pg "CREATE DATABASE \"unifi-protect\" OWNER \"unifi-protect\""
fi
psql_pg "CREATE EXTENSION IF NOT EXISTS vector" unifi-protect
[ "$NEW_CLUSTER" = 1 ] && log "PostgreSQL ready (new cluster; console login allowed from $PG_ALLOW)"

stop_all() {
    log "stopping"
    [ -n "${RUN_PID:-}" ] && kill -TERM "$RUN_PID" 2>/dev/null || true
    [ -n "${RUN_PID:-}" ] && wait "$RUN_PID" 2>/dev/null || true
    as_pg "$PGBIN/pg_ctl" -D "$PGDATA" -m fast -w stop >/dev/null 2>&1 || true
    exit 0
}
trap stop_all TERM INT

log "starting run_all.py on $IFACE"
cd /app/diykey
python3 run_all.py --iface "$IFACE" --state-dir /data/state --hostname "$HOSTNAME_" "$@" &
RUN_PID=$!
set +e
wait "$RUN_PID"
rc=$?
set -e
log "run_all.py exited ($rc); stopping PostgreSQL"
as_pg "$PGBIN/pg_ctl" -D "$PGDATA" -m fast -w stop >/dev/null 2>&1 || true
exit "$rc"
