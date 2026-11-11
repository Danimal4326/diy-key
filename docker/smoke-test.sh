#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
#
# Smoke test for the image: start it with the console pointed at 127.0.0.1 (nothing outside the
# container is contacted), wait for docker/healthcheck.sh, then check what the console relies on:
# the PostgreSQL role/db/vector extension, an SSL login as unifi-protect (and no non-SSL one),
# /api/info, and a clean stop.
#
#   docker/smoke-test.sh <image> [extra run args...]
#   DOCKER=podman docker/smoke-test.sh diykey:test --network podman   # rootless Podman: its
#                                                                       # default pasta has no eth0
set -euo pipefail

IMAGE=${1:?usage: smoke-test.sh <image> [docker run args...]}
shift
DOCKER=${DOCKER:-docker}
NAME=diykey-smoke-$$
PSQL=/usr/lib/postgresql/17/bin/psql

cleanup() { "$DOCKER" rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT
fail() { echo "FAIL: $*" >&2; "$DOCKER" logs "$NAME" 2>&1 | tail -30 >&2 || true; exit 1; }

"$DOCKER" run -d --name "$NAME" --cap-add NET_ADMIN --cap-add NET_RAW \
    -e DIYKEY_CONSOLE_HOST=127.0.0.1 -e DIYKEY_MAC=FCECDA000099 \
    -e DIYKEY_DEVICE_ID=00000000-0000-4000-8000-000000000099 \
    -e DIYKEY_GUID=00000000-0000-4000-8000-000000000098 "$@" "$IMAGE" >/dev/null

for _ in $(seq 1 60); do
    "$DOCKER" exec "$NAME" /app/healthcheck.sh >/dev/null 2>&1 && break
    [ "$("$DOCKER" inspect -f '{{.State.Running}}' "$NAME")" = true ] || fail "container exited"
    sleep 2
done
"$DOCKER" exec "$NAME" /app/healthcheck.sh || fail "healthcheck never passed"
echo "ok: healthcheck"

logs=$("$DOCKER" logs "$NAME" 2>&1)
grep -q "PostgreSQL ready (new cluster" <<<"$logs" || fail "no 'PostgreSQL ready' in the log"
grep -qE "Started [0-9]+ components" <<<"$logs" || fail "run_all.py didn't start its components"
echo "ok: PostgreSQL created, run_all.py started"

"$DOCKER" exec -i "$NAME" bash -s <<EOF || fail "database/API checks"
set -euo pipefail
ip=\$(ip -4 -o addr show eth0 | awk '{print \$4}' | cut -d/ -f1)
got=\$(PGPASSWORD=unifi-protect $PSQL "host=\$ip dbname=unifi-protect user=unifi-protect sslmode=require" -tAc \
    "select s.ssl::text || ' ' || (select extversion from pg_extension where extname = 'vector') from pg_stat_ssl s where s.pid = pg_backend_pid()")
case "\$got" in true\ *) echo "ok: SSL login as unifi-protect, vector \${got#true }" ;; *) echo "SSL login: \$got"; exit 1 ;; esac
if PGPASSWORD=unifi-protect $PSQL "host=\$ip dbname=unifi-protect user=unifi-protect sslmode=disable" -tAc "select 1" >/dev/null 2>&1; then
    echo "a non-SSL login was accepted"; exit 1
fi
echo "ok: non-SSL login refused"
python3 - "\$ip" <<'PY'
import json, ssl, sys, urllib.request
info = json.loads(urllib.request.urlopen(f"https://{sys.argv[1]}:8080/api/info",
                                         context=ssl._create_unverified_context(), timeout=5).read())
assert info.get("sysid") == "0xa5f0", info
print("ok: /api/info sysid", info["sysid"], "mac", info.get("mac"))
PY
EOF

"$DOCKER" stop -t 30 "$NAME" >/dev/null
code=$("$DOCKER" inspect -f '{{.State.ExitCode}}' "$NAME")
[ "$code" = 0 ] || fail "stop exited $code"
"$DOCKER" logs "$NAME" 2>&1 | grep -q "^entrypoint: stopping" || fail "no clean shutdown in the log"
echo "ok: clean stop (exit 0)"
echo "smoke test passed: $IMAGE"
