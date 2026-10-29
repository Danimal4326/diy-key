# SPDX-License-Identifier: AGPL-3.0-or-later
# Where a DIY-Key instance keeps its files, sourced by diykey_service.sh and diykey_pg_init.sh
# (set USER_HOME first). Each instance has a folder of its own:
#
#   ~/.diy-key/$DIYKEY_INSTANCE/     (DIYKEY_INSTANCE is "main" when not set)
#     diykey.env         settings: DIYKEY_* identity and network
#     state/             adopt state, server.crt/server.key (DIYKEY_STATE_DIR overrides)
#     pg/                this AI Key's own PostgreSQL cluster (Protect connects to <bind ip>:5432)
#     logs/diykey.log    DIY-Key's log, rotated beside it
#     logs/postgres.log  PostgreSQL's log
#     restart            touch it to restart DIY-Key
#     service.pid
#
# DIYKEY_ENV_FILE, DIYKEY_LOG, DIYKEY_RESTART_FILE and DIYKEY_PIDFILE still override one path.

INSTANCE=${DIYKEY_INSTANCE:-main}
if ! [[ "$INSTANCE" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
    echo "DIYKEY_INSTANCE must be lowercase letters, digits and -, not: $INSTANCE" >&2
    exit 1
fi
BASE=$USER_HOME/.diy-key/$INSTANCE

ENV_FILE=${DIYKEY_ENV_FILE:-$BASE/diykey.env}
TRIGGER=${DIYKEY_RESTART_FILE:-$BASE/restart}
LOG=${DIYKEY_LOG:-$BASE/logs/diykey.log}
PIDFILE=${DIYKEY_PIDFILE:-$BASE/service.pid}
DEFAULT_STATE_DIR=$BASE/state
PG_DIR=$BASE/pg
PG_LOG=$BASE/logs/postgres.log
