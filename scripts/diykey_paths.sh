# SPDX-License-Identifier: AGPL-3.0-or-later
# Instance layout: ~/.diy-key/$DIYKEY_INSTANCE/ (diykey.env, state/, pg/, logs/, restart).
# DIYKEY_ENV_FILE, DIYKEY_LOG, DIYKEY_RESTART_FILE and DIYKEY_PIDFILE override one path.

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
